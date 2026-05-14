"""
test_stress.py
Stress-test the SCM across extreme boundary conditions and forcings.

For each case the model runs for RUN_HOURS (default 6 h) and the test records:
  - Whether the run crashed (exception) or produced NaN / Inf in any snapshot
  - Final cloud state (LWP, cloud fraction)
  - State bounds (T_min, T_max, qv_min)
  - Inversion height evolution

The script prints a wide summary table and a categorised verdict.
The only hard assertion is that the *baseline* case stays numerically healthy
with cloud present.  All other outcomes are recorded as observations.

Usage:
    python3 test_stress.py             # 6-h runs (default)
    python3 test_stress.py --hours 12  # longer runs for each case
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, '.')
import config as _cfg_module
from scm import StratocumulusSCM


# -----------------------------------------------------------------------
# Test matrix
# Each entry: (case_id, human description, {cfg_field: value, ...})
# -----------------------------------------------------------------------
CASES = [
    # ---- Baseline ----
    ("baseline",           "Reference DYCOMS-II RF01",              {}),

    # ---- SST sweep ----
    ("sst_280",            "Cold SST 280 K",                        {"sst_K": 280.0}),
    ("sst_285",            "Cold SST 285 K",                        {"sst_K": 285.0}),
    ("sst_297",            "Warm SST 297 K",                        {"sst_K": 297.0}),
    ("sst_300",            "Warm SST 300 K",                        {"sst_K": 300.0}),
    ("sst_305",            "Hot SST 305 K",                         {"sst_K": 305.0}),
    ("sst_310",            "Extreme hot SST 310 K",                 {"sst_K": 310.0}),

    # ---- Large-scale divergence sweep ----
    ("div_1e-6",           "Weak divergence D=1e-6 s⁻¹",           {"divergence": 1.0e-6}),
    ("div_3e-6",           "D=3e-6 s⁻¹",                           {"divergence": 3.0e-6}),
    ("div_10e-6",          "Strong D=10e-6 s⁻¹",                   {"divergence": 10.0e-6}),
    ("div_20e-6",          "Extreme D=20e-6 s⁻¹",                  {"divergence": 20.0e-6}),

    # ---- Inversion strength sweep ----
    ("inv_0K",             "No inversion Δθ=0 K",                  {"theta_FT_jump": 0.0}),
    ("inv_2K",             "Weak inversion Δθ=2 K",                {"theta_FT_jump": 2.0}),
    ("inv_5K",             "Moderate inversion Δθ=5 K",            {"theta_FT_jump": 5.0}),
    ("inv_15K",            "Strong inversion Δθ=15 K",             {"theta_FT_jump": 15.0}),
    ("inv_25K",            "Very strong inversion Δθ=25 K",        {"theta_FT_jump": 25.0}),

    # ---- BL moisture sweep ----
    ("qt_3g",              "Very dry BL qt=3 g/kg",                {"qt_BL": 3.0e-3}),
    ("qt_5g",              "Dry BL qt=5 g/kg",                     {"qt_BL": 5.0e-3}),
    ("qt_7g",              "Slightly dry qt=7 g/kg",               {"qt_BL": 7.0e-3}),
    ("qt_12g",             "Moist BL qt=12 g/kg",                  {"qt_BL": 12.0e-3}),
    ("qt_16g",             "Very moist BL qt=16 g/kg",             {"qt_BL": 16.0e-3}),

    # ---- CCN / droplet number sweep ----
    ("Nc_5",               "Ultra-clean Nc=5 /cm³",                {"Nc_prescribed": 5.0e6}),
    ("Nc_20",              "Clean Nc=20 /cm³",                     {"Nc_prescribed": 20.0e6}),
    ("Nc_500",             "Polluted Nc=500 /cm³",                 {"Nc_prescribed": 500.0e6}),
    ("Nc_2000",            "Heavily polluted Nc=2000 /cm³",        {"Nc_prescribed": 2000.0e6}),

    # ---- Wind speed sweep ----
    ("wind_0p3",           "Near-calm u=0.3 m/s",                  {"u_ref": 0.3}),
    ("wind_1",             "Light wind u=1 m/s",                   {"u_ref": 1.0}),
    ("wind_15",            "Strong wind u=15 m/s",                 {"u_ref": 15.0}),
    ("wind_25",            "Storm-force u=25 m/s",                 {"u_ref": 25.0}),

    # ---- Initial inversion height sweep ----
    ("zi_100",             "Very low inversion zi=100 m",          {"zi_init": 100.0}),
    ("zi_300",             "Low inversion zi=300 m",               {"zi_init": 300.0}),
    ("zi_1500",            "High inversion zi=1500 m",             {"zi_init": 1500.0}),
    ("zi_2500",            "Near model-top zi=2500 m",             {"zi_init": 2500.0}),

    # ---- Combined edge cases ----
    ("warm_dry",           "Warm SST 298K + dry BL 5 g/kg",
     {"sst_K": 298.0, "qt_BL": 5.0e-3}),
    ("cold_moist",         "Cold SST 284K + moist BL 13 g/kg",
     {"sst_K": 284.0, "qt_BL": 13.0e-3}),
    ("strong_sub_weak_inv","Strong D=12e-6 + weak inversion 3K",
     {"divergence": 12.0e-6, "theta_FT_jump": 3.0}),
    ("high_zi_near_top",   "zi=2400m close to model top (3000m)",  {"zi_init": 2400.0}),
]


# -----------------------------------------------------------------------
# Helper: run one case
# -----------------------------------------------------------------------

def _check_snaps(history):
    """Return (has_nan, has_inf) by scanning all snapshots."""
    for snap in history:
        T  = snap['T']
        qv = snap['qv']
        if np.any(np.isnan(T)) or np.any(np.isnan(qv)):
            return True, False
        if np.any(np.isinf(T)) or np.any(np.isinf(qv)):
            return False, True
    return False, False


def run_case(name, overrides, t_end):
    """Run one stress case.  Returns a result dict."""
    cfg = _cfg_module.cfg
    orig = {k: getattr(cfg, k) for k in overrides}
    for k, v in overrides.items():
        setattr(cfg, k, v)

    r = dict(
        name=name,
        status='ok',          # 'ok' | 'no_cloud' | 'nan' | 'inf' | 'crash'
        crash_reason='',
        LWP_init=np.nan, LWP_final=np.nan,
        zi_init=np.nan,  zi_final=np.nan,
        T_min=np.nan, T_max=np.nan,
        qv_min_gkg=np.nan,
        precip_max_mmhr=0.0,
        cloud_final=False,
        n_snaps=0,
    )

    try:
        model = StratocumulusSCM()
        model.initialize()
        history = model.run(t_end=t_end, verbose=False)

        if not history:
            r['status'] = 'crash'
            r['crash_reason'] = 'empty history'
        else:
            has_nan, has_inf = _check_snaps(history)
            if has_nan:
                r['status'] = 'nan'
            elif has_inf:
                r['status'] = 'inf'
            else:
                snap0 = history[0]
                snapf = history[-1]
                r['LWP_init']  = snap0['LWP'] * 1000.0
                r['LWP_final'] = snapf['LWP'] * 1000.0
                r['zi_init']   = snap0['zi']
                r['zi_final']  = snapf['zi']
                r['cloud_final'] = snapf['LWP'] > 1e-3   # > 1 g/m²
                r['T_min']     = float(np.min(snapf['T']))
                r['T_max']     = float(np.max(snapf['T']))
                r['qv_min_gkg'] = float(np.min(snapf['qv'])) * 1000.0
                r['precip_max_mmhr'] = max(s['precip_rate'] for s in history) * 3600.0
                r['n_snaps']   = len(history)
                if not r['cloud_final']:
                    r['status'] = 'no_cloud'
                elif (not np.isfinite(r['precip_max_mmhr'])
                      or r['precip_max_mmhr'] > _PRECIP_MAX_PHYSICAL_MMHR):
                    r['status'] = 'precip_overflow'

    except Exception as exc:
        r['status'] = 'crash'
        r['crash_reason'] = f"{type(exc).__name__}: {str(exc)[:120]}"

    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)

    return r


# -----------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------

_STATUS_LABEL = {
    'ok':          'STABLE',
    'no_cloud':    'dissipated',
    'precip_overflow': '*** precip_OVF',
    'nan':         '*** NaN',
    'inf':         '*** Inf',
    'crash':       '*** CRASH',
}

# Precipitation rate above this is physically impossible and indicates overflow
_PRECIP_MAX_PHYSICAL_MMHR = 500.0

def print_table(results):
    hdr = (f"{'CASE':<28} {'STATUS':<12} {'LWP0':>7} {'LWPf':>7} "
           f"{'zi0':>6} {'zif':>6} {'T_min':>7} {'T_max':>7} "
           f"{'qvmin':>7} {'precip':>8}  NOTE")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in results:
        label = _STATUS_LABEL.get(r['status'], r['status'])
        note  = r['crash_reason'][:40] if r['crash_reason'] else ''
        print(f"{r['name']:<28} {label:<12} "
              f"{r['LWP_init']:>7.1f} {r['LWP_final']:>7.1f} "
              f"{r['zi_init']:>6.0f} {r['zi_final']:>6.0f} "
              f"{r['T_min']:>7.1f} {r['T_max']:>7.1f} "
              f"{r['qv_min_gkg']:>7.3f} {r['precip_max_mmhr']:>8.4f}  {note}")
    print("=" * len(hdr))


def print_summary(results):
    by_status = {}
    for r in results:
        by_status.setdefault(r['status'], []).append(r['name'])

    print("\n--- Summary ---")
    for status in ('ok', 'no_cloud', 'precip_overflow', 'nan', 'inf', 'crash'):
        names = by_status.get(status, [])
        if names:
            print(f"  {_STATUS_LABEL[status]:<14}: {', '.join(names)}")

    # Physical observations for stable cases
    stable = [r for r in results if r['status'] == 'ok']
    if stable:
        lwp_vals   = [(r['name'], r['LWP_final']) for r in stable]
        max_lwp    = max(lwp_vals, key=lambda x: x[1])
        min_lwp    = min(lwp_vals, key=lambda x: x[1])
        max_precip = max(stable, key=lambda r: r['precip_max_mmhr'])
        print(f"\n  Highest LWP : {max_lwp[0]} → {max_lwp[1]:.1f} g/m²")
        print(f"  Lowest  LWP : {min_lwp[0]} → {min_lwp[1]:.1f} g/m²")
        print(f"  Max precip  : {max_precip['name']} → {max_precip['precip_max_mmhr']:.4f} mm/hr")


# -----------------------------------------------------------------------
# Identify limitations from results
# -----------------------------------------------------------------------

def identify_limitations(results):
    """
    Classify outcomes into model limitation categories.
    Returns a dict mapping category name -> list of (case_id, description).
    """
    lims = {
        'numerical_instability': [],
        'cloud_dissipation':     [],
        'domain_boundary':       [],
        'physics_extreme':       [],
    }

    # Build a lookup for overrides
    case_map = {name: overrides for name, _, overrides in CASES}

    for r in results:
        ov = case_map.get(r['name'], {})

        # NaN / Inf / crash / precip overflow → numerical instability
        if r['status'] in ('nan', 'inf', 'crash', 'precip_overflow'):
            lims['numerical_instability'].append(
                (r['name'], r['crash_reason'] or r['status'])
            )

        # Cloud dissipated
        if r['status'] == 'no_cloud':
            lims['cloud_dissipation'].append(
                (r['name'], f"LWP_final={r['LWP_final']:.1f} g/m²")
            )

        # Domain boundary: zi near model top (> 2400 m of 3000 m)
        if r['status'] == 'ok' and r.get('zi_final', 0) > 2400:
            lims['domain_boundary'].append(
                (r['name'], f"zi_final={r['zi_final']:.0f} m (model top=3000 m)")
            )

        # Physical extremes: T out of range or negative qv
        if r['status'] == 'ok':
            if r['T_min'] < 220 or r['T_max'] > 340:
                lims['physics_extreme'].append(
                    (r['name'], f"T range [{r['T_min']:.0f}, {r['T_max']:.0f}] K")
                )
            if r['qv_min_gkg'] < -0.01:
                lims['physics_extreme'].append(
                    (r['name'], f"qv_min={r['qv_min_gkg']:.4f} g/kg (negative)")
                )

    return lims


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SCM stress test')
    parser.add_argument('--hours', type=float, default=6.0,
                        help='Simulation length per case in hours (default 6)')
    args = parser.parse_args()

    t_end = args.hours * 3600.0
    print(f"SCM stress test: {len(CASES)} cases × {args.hours:.0f} h each\n")

    results = []
    for name, desc, overrides in CASES:
        print(f"  [{name:<28}] {desc[:55]:<55}", end='  ', flush=True)
        r = run_case(name, overrides, t_end)
        results.append(r)
        label = _STATUS_LABEL.get(r['status'], r['status'])
        if r['status'] == 'ok':
            print(f"{label}  LWP={r['LWP_final']:.1f} g/m²  zi={r['zi_final']:.0f} m")
        elif r['status'] == 'no_cloud':
            print(f"{label}  LWP={r['LWP_final']:.3f} g/m²")
        elif r['status'] == 'precip_overflow':
            print(f"{label}  precip={r['precip_max_mmhr']:.2e} mm/hr")
        else:
            print(f"{label}  {r['crash_reason'][:60]}")

    print_table(results)
    print_summary(results)

    lims = identify_limitations(results)
    print("\n--- Identified limitations ---")
    for cat, items in lims.items():
        if items:
            print(f"\n  [{cat}]")
            for name, note in items:
                print(f"    {name:<28}  {note}")

    # Hard assertion: baseline must be numerically healthy and cloudy
    baseline = next(r for r in results if r['name'] == 'baseline')
    assert baseline['status'] == 'ok', \
        f"Baseline failed: {baseline['status']} — {baseline['crash_reason']}"
    assert baseline['cloud_final'], "Baseline has no cloud at end of run!"
    print("\nBaseline assertions passed.")

    return results, lims


if __name__ == '__main__':
    main()
