"""
test_twomey.py
Test the Twomey (first indirect aerosol) effect.

The Twomey effect predicts that for fixed liquid water path, higher cloud
droplet number concentration (Nc) produces more and smaller droplets,
increasing cloud optical depth as:

    tau_vis ∝ LWP^(2/3) * Nc^(1/3)

This test runs the SCM with Nc_prescribed spanning one decade (50–500 /cm³
equivalent) and verifies that the time-mean optical depth follows the
expected 1/3-power scaling with Nc.

Effective radius per layer is computed from the two-moment microphysics:

    r_eff = (3 * rho * qc / (4 * pi * rho_w * Nc_vol))^(1/3)

giving a proper Nc-dependent optical depth:

    tau_vis = sum_k [ 3 * rho[k] * qc[k] * dz / (2 * r_eff[k]) ]

The precipitation-suppression (lifetime) effect is also observable: higher
Nc suppresses drizzle, raising LWP.  The scaling test corrects for LWP
variation so that the pure Twomey exponent is isolated.

Reference: Twomey (1977), Atmos. Environ., 11, 1251-1256.
"""

import numpy as np
import sys

from config import cfg
from scm import StratocumulusSCM
import microphysics as micro


# ----------------------------------------------------------------
# CCN concentrations to test (#/kg)
# At BL density rho ~ 1.1 kg/m³: 1 /cm³ ≈ 1.1e6 /kg
# Approximate /cm³: 45, 90, 135, 270, 455 /cm³
# ----------------------------------------------------------------
NC_VALUES = np.array([50e6, 100e6, 150e6, 300e6, 500e6])   # /kg


# ----------------------------------------------------------------
# Helper: per-column optical depth using layer r_eff
# ----------------------------------------------------------------

def tau_vis_reff(snap):
    """
    Column cloud optical depth computed from layer-by-layer effective radius.

        tau_vis = sum_k [ 3 * rho[k] * qc[k] * dz / (2 * r_eff[k]) ]

    This correctly captures the Nc dependence; the fixed-r_eff diagnostic
    in radiation.py does not.
    """
    qc  = snap['qc']
    Nc  = snap['Nc']
    rho = snap['rho']
    dz  = cfg.dz

    cloudy = qc > cfg.ql_min
    if not np.any(cloudy):
        return 0.0

    r_eff = micro.r_eff_cloud(rho, qc, Nc)
    r_eff = np.maximum(r_eff, 1.0e-7)   # guard against zero
    tau   = np.sum(3.0 * rho[cloudy] * qc[cloudy] * dz / (2.0 * r_eff[cloudy]))
    return float(tau)


# ----------------------------------------------------------------
# Single SCM run
# ----------------------------------------------------------------

def run_twomey_case(nc_prescribed, run_hours=6, verbose=False):
    """
    Run the SCM with the given Nc_prescribed value.

    Temporarily overrides cfg.Nc_prescribed and cfg.t_end, then restores them
    so successive calls are independent.

    Returns a dict of time-series diagnostics.
    """
    nc_save    = cfg.Nc_prescribed
    t_end_save = cfg.t_end

    cfg.Nc_prescribed = nc_prescribed
    cfg.t_end         = run_hours * 3600.0

    try:
        model = StratocumulusSCM(sst_mode='fixed', sst_K=cfg.sst_K)
        model.initialize()
        history = model.run(t_end=cfg.t_end, verbose=verbose)
    finally:
        cfg.Nc_prescribed = nc_save
        cfg.t_end         = t_end_save

    if not history:
        return None

    LWPs  = np.array([s['LWP']          for s in history])
    taus  = np.array([tau_vis_reff(s)   for s in history])
    times = np.array([s['time']         for s in history])

    return {
        'nc'       : nc_prescribed,
        'LWP_mean' : float(np.mean(LWPs)),
        'tau_mean' : float(np.mean(taus)),
        'LWP_final': float(LWPs[-1]),
        'tau_final': float(taus[-1]),
        'times'    : times,
        'LWPs'     : LWPs,
        'taus'     : taus,
    }


# ----------------------------------------------------------------
# Scaling exponent
# ----------------------------------------------------------------

def twomey_exponent(nc_vals, tau_vals, lwp_vals):
    """
    Fit the exponent beta in tau ∝ LWP^(2/3) * Nc^beta via log-log OLS.

    Dividing tau by LWP^(2/3) removes the concurrent LWP change (lifetime
    effect) so that only the pure Twomey component is measured.
    """
    tau_norm = tau_vals / np.maximum(lwp_vals, 1.0e-6) ** (2.0 / 3.0)
    log_nc   = np.log(nc_vals)
    log_tau  = np.log(np.maximum(tau_norm, 1.0e-10))
    beta     = np.polyfit(log_nc, log_tau, 1)[0]
    return float(beta)


# ----------------------------------------------------------------
# Main test
# ----------------------------------------------------------------

def test_twomey_effect(run_hours=6, verbose=False):
    """
    Verify the Twomey (1977) optical-depth–droplet-number scaling.

    Assertions
    ----------
    1. Cloud is present (tau > 0.1) for every Nc value.
    2. tau increases monotonically with Nc (more droplets → smaller r_eff
       → higher optical depth for the same liquid water).
    3. The LWP-corrected log-log slope d(ln tau) / d(ln Nc) lies in
       [0.20, 0.50], consistent with the theoretical value of 1/3.
    """
    rho_bl   = 1.1          # kg/m³, representative BL density for display
    nc_cm3   = NC_VALUES * rho_bl * 1.0e-6

    print("=" * 60)
    print("  Twomey Effect Test")
    print(f"  Run length    : {run_hours} h per case")
    print(f"  Nc values     : {nc_cm3.round(0)} /cm³ (approx)")
    print(f"  Theoretical   : tau ∝ LWP^(2/3) * Nc^(1/3)")
    print("=" * 60)

    results = []
    for nc in NC_VALUES:
        nc_display = nc * rho_bl * 1.0e-6
        print(f"\n--- Nc = {nc:.0e} /kg  (~{nc_display:.0f} /cm³) ---")
        r = run_twomey_case(nc, run_hours=run_hours, verbose=verbose)
        if r is None:
            raise RuntimeError(f"Model produced no output for Nc = {nc:.0e} /kg")
        print(f"  LWP (mean) = {r['LWP_mean'] * 1000:.1f} g/m²")
        print(f"  tau (mean) = {r['tau_mean']:.2f}")
        results.append(r)

    nc_vals  = np.array([r['nc']       for r in results])
    tau_vals = np.array([r['tau_mean'] for r in results])
    lwp_vals = np.array([r['LWP_mean'] for r in results])

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("  Summary")
    print(f"  {'Nc (/kg)':>12}  {'Nc (/cm³)':>10}  {'LWP (g/m²)':>12}  {'tau_vis':>8}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['nc']:12.0f}  {r['nc']*rho_bl*1e-6:10.1f}  "
              f"{r['LWP_mean']*1000:12.1f}  {r['tau_mean']:8.2f}")

    # ---- Assertion 1: cloud present ----
    assert all(tau > 0.1 for tau in tau_vals), (
        f"Cloud absent or nearly absent in at least one case.\n"
        f"  tau_vals = {tau_vals}"
    )
    print("\n[PASS] Cloud present (tau > 0.1) for all Nc values")

    # ---- Assertion 2: tau monotonically increases with Nc ----
    assert all(tau_vals[i + 1] > tau_vals[i] for i in range(len(tau_vals) - 1)), (
        f"tau does not increase monotonically with Nc.\n"
        f"  nc_vals  = {nc_vals}\n"
        f"  tau_vals = {tau_vals}"
    )
    print("[PASS] tau increases monotonically with Nc")

    # ---- Assertion 3: Twomey exponent ~ 1/3 ----
    beta = twomey_exponent(nc_vals, tau_vals, lwp_vals)
    theory = 1.0 / 3.0
    print(f"\n  Fitted Twomey exponent (LWP-corrected) : {beta:.3f}")
    print(f"  Theoretical value 1/3                  : {theory:.3f}")
    assert 0.20 <= beta <= 0.50, (
        f"Twomey exponent {beta:.3f} outside expected range [0.20, 0.50].\n"
        f"  Theory predicts 1/3 ≈ 0.333."
    )
    print(f"[PASS] Exponent {beta:.3f} is within tolerance [0.20, 0.50]")

    print("\n[ALL TESTS PASSED]\n")
    return results


# ----------------------------------------------------------------
# Optional diagnostic plot
# ----------------------------------------------------------------

def plot_twomey(results):
    """Plot optical depth and LWP vs. Nc, with the Twomey reference line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    rho_bl   = 1.1
    nc_vals  = np.array([r['nc']       for r in results])
    tau_vals = np.array([r['tau_mean'] for r in results])
    lwp_vals = np.array([r['LWP_mean'] for r in results])
    nc_cm3   = nc_vals * rho_bl * 1.0e-6

    # Twomey reference line anchored to middle point
    i_ref     = len(nc_vals) // 2
    nc_ref    = nc_vals[i_ref]
    tau_ref   = tau_vals[i_ref]
    nc_line   = np.logspace(np.log10(nc_vals[0] * 0.8), np.log10(nc_vals[-1] * 1.2), 60)
    tau_theory = tau_ref * (nc_line / nc_ref) ** (1.0 / 3.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Twomey Effect — Nighttime Stratocumulus SCM', fontsize=13)

    # Panel 1: tau vs Nc (log-log)
    ax = axes[0]
    ax.loglog(nc_cm3, tau_vals, 'o-', ms=8, label='SCM')
    ax.loglog(nc_line * rho_bl * 1.0e-6, tau_theory, 'k--', lw=1.5,
              label=r'$\tau \propto N_c^{1/3}$ (Twomey 1977)')
    ax.set_xlabel(r'$N_c$ (cm$^{-3}$, approx)')
    ax.set_ylabel(r'Cloud optical depth $\tau_{vis}$')
    ax.set_title('Optical depth vs. droplet number')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: LWP vs Nc (lifetime / precipitation-suppression effect)
    ax = axes[1]
    ax.semilogx(nc_cm3, lwp_vals * 1000, 's-', ms=8, color='tab:blue')
    ax.set_xlabel(r'$N_c$ (cm$^{-3}$, approx)')
    ax.set_ylabel(r'LWP (g/m²)')
    ax.set_title('LWP vs. droplet number\n(precipitation-suppression effect)')
    ax.grid(True, alpha=0.3)

    # Panel 3: LWP-corrected tau vs Nc (isolates pure Twomey)
    ax = axes[2]
    tau_norm = tau_vals / np.maximum(lwp_vals, 1.0e-6) ** (2.0 / 3.0)
    beta     = twomey_exponent(nc_vals, tau_vals, lwp_vals)
    norm_ref   = tau_norm[i_ref]
    norm_theory = norm_ref * (nc_line / nc_ref) ** (1.0 / 3.0)
    ax.loglog(nc_cm3, tau_norm, 'D-', ms=8, color='tab:orange',
              label='LWP-corrected tau')
    ax.loglog(nc_line * rho_bl * 1.0e-6, norm_theory, 'k--', lw=1.5,
              label=r'slope $= 1/3$')
    ax.set_xlabel(r'$N_c$ (cm$^{-3}$, approx)')
    ax.set_ylabel(r'$\tau_{vis}$ / LWP$^{2/3}$')
    ax.set_title(f'Pure Twomey scaling\nfitted exponent = {beta:.3f}  (theory: 1/3)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('twomey_effect.png', dpi=150, bbox_inches='tight')
    print("Saved twomey_effect.png")

    import os
    if os.environ.get('DISPLAY') or sys.platform == 'darwin':
        plt.show()


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Twomey effect test for nighttime Sc SCM')
    p.add_argument('--hours',   type=float, default=6,
                   help='Run length per Nc case in hours (default: 6)')
    p.add_argument('--plot',    action='store_true',
                   help='Generate diagnostic plots after the test')
    p.add_argument('--verbose', action='store_true',
                   help='Print per-timestep SCM output')
    args = p.parse_args()

    results = test_twomey_effect(run_hours=args.hours, verbose=args.verbose)

    if args.plot:
        plot_twomey(results)
