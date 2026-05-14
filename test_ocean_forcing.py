"""
test_ocean_forcing.py
Test the cloud-property response to a uniform prescribed heat flux into the
slab ocean (Q_ocean, W/m²).

Physical setup
--------------
The slab-ocean energy budget is:

    C_slab * dT_sst/dt = -F_lw_net_sfc - SHF - LHF + Q_ocean

where C_slab = rho_ocean * cp_ocean * d_slab ~ 4.1e7 J/(m² K).

Q_ocean > 0  : additional heat into the ocean (e.g. subsurface upwelling
               suppressed, warm-ocean anomaly).  SST warms → stronger surface
               fluxes → potentially cloud thinning.
Q_ocean < 0  : heat extracted from the ocean (cold-ocean anomaly, upwelling).
               SST cools → higher lower-tropospheric stability → cloud thickening.

This is the canonical "ocean-forcing" perturbation used in subtropical Sc
cloud feedback studies (e.g. Bretherton et al. 2013; Nuijens & Stevens 2012).

Expected responses (from obs and LES):
  - Negative forcing → SST cools, LTS increases → thicker, deeper cloud
  - Positive forcing → SST warms, LTS decreases → thinner cloud / dissipation
  - LWP and cloud-top height both decrease with positive Q_ocean
  - SHF and LHF increase with positive Q_ocean (larger air–sea contrast)

Assertions
----------
1. SST change is monotonically increasing with Q_ocean.
2. LWP is monotonically decreasing with increasing Q_ocean.
3. Negative forcing → LWP ≥ baseline (cloud maintained / deepened).
4. Positive forcing → LWP ≤ baseline (cloud thinned).
5. Cloud fraction decreases monotonically with increasing Q_ocean.
6. Cloud depth (z_ctop − z_cbase) decreases monotonically with increasing Q_ocean.
7. Cloud-top height decreases monotonically with increasing Q_ocean.
8. Inversion height is monotonically non-increasing with Q_ocean.

References:
  Bretherton et al. (2013), J. Adv. Model. Earth Syst.
  Nuijens & Stevens (2012), J. Atmos. Sci.
"""

import numpy as np
import sys

from config import cfg
from scm import StratocumulusSCM


# ----------------------------------------------------------------
# Forcing values to sweep (W/m²)
# ----------------------------------------------------------------
Q_VALUES = np.array([-50.0, -25.0, 0.0, +25.0, +50.0])   # W/m²


# ----------------------------------------------------------------
# Single SCM run
# ----------------------------------------------------------------

def run_forcing_case(Q_ocean, run_hours=48, verbose=False):
    """
    Run the slab-ocean SCM with a prescribed ocean heat flux Q_ocean (W/m²).

    Returns time-series of bulk cloud diagnostics plus initial/final SST.
    """
    t_end_save = cfg.t_end
    cfg.t_end  = run_hours * 3600.0

    try:
        model = StratocumulusSCM(
            sst_mode='slab',
            sst_K=cfg.sst_K,
            Q_ocean=Q_ocean,
        )
        model.initialize()
        T_sst_init = model.surface.T_sst
        history = model.run(t_end=cfg.t_end, verbose=verbose)
        T_sst_final = model.surface.T_sst
    finally:
        cfg.t_end = t_end_save

    if not history:
        return None

    times       = np.array([s['time']    for s in history])
    LWPs        = np.array([s['LWP']     for s in history])
    zis         = np.array([s['zi']      for s in history])
    z_cbases    = np.array([s['z_cbase'] for s in history])
    z_ctops     = np.array([s['z_ctop']  for s in history])
    taus        = np.array([s['tau_vis'] for s in history])
    shfs        = np.array([s['shf']     for s in history])
    lhfs        = np.array([s['lhf']     for s in history])
    T_ssts      = np.array([s['T_sst']   for s in history])

    # Cloud fraction: fraction of model layers with qc > ql_min
    cfs         = np.array([np.mean(s['qc'] > cfg.ql_min) for s in history])

    # Cloud depth (NaN where no cloud)
    cloud_depths = z_ctops - z_cbases   # (nz,) array; NaN when cloud absent

    # Use the second half of the run as quasi-equilibrium mean
    half = len(history) // 2
    return {
        'Q_ocean'        : Q_ocean,
        'T_sst_init'     : T_sst_init,
        'T_sst_final'    : T_sst_final,
        'dT_sst'         : T_sst_final - T_sst_init,
        'LWP_mean'       : float(np.mean(LWPs[half:])),
        'LWP_final'      : float(LWPs[-1]),
        'zi_mean'        : float(np.nanmean(zis[half:])),
        'z_ctop_mean'    : float(np.nanmean(z_ctops[half:])),
        'z_cbase_mean'   : float(np.nanmean(z_cbases[half:])),
        'cloud_depth_mean': float(np.nanmean(cloud_depths[half:])),
        'cf_mean'        : float(np.mean(cfs[half:])),
        'tau_mean'       : float(np.mean(taus[half:])),
        'shf_mean'       : float(np.mean(shfs[half:])),
        'lhf_mean'       : float(np.mean(lhfs[half:])),
        'times'          : times,
        'LWPs'           : LWPs,
        'T_ssts'         : T_ssts,
        'zis'            : zis,
        'z_cbases'       : z_cbases,
        'z_ctops'        : z_ctops,
        'cloud_depths'   : cloud_depths,
        'cfs'            : cfs,
        'taus'           : taus,
    }


# ----------------------------------------------------------------
# Main test
# ----------------------------------------------------------------

def test_ocean_forcing(run_hours=48, verbose=False):
    """
    Verify that cloud properties respond physically to uniform ocean heat forcing.

    Assertions
    ----------
    1. dT_sst increases monotonically with Q_ocean.
    2. LWP decreases monotonically with increasing Q_ocean.
    3. Negative forcing → LWP ≥ baseline (cloud maintained / deepened).
    4. Positive forcing → LWP ≤ baseline (cloud thinned).
    5. Cloud fraction decreases monotonically with increasing Q_ocean.
    6. Cloud depth (z_ctop − z_cbase) decreases monotonically with Q_ocean.
    7. Cloud-top height decreases monotonically with increasing Q_ocean.
    8. Inversion height is monotonically non-increasing with Q_ocean.
    """
    print("=" * 65)
    print("  Ocean Forcing Cloud-Response Test")
    print(f"  Run length : {run_hours} h per case  (slab ocean)")
    print(f"  Q_ocean    : {Q_VALUES} W/m²")
    print(f"  Baseline   : Q_ocean = 0 W/m²")
    print("=" * 65)

    results = []
    for Q in Q_VALUES:
        sign = '+' if Q >= 0 else ''
        print(f"\n--- Q_ocean = {sign}{Q:.0f} W/m² ---")
        r = run_forcing_case(Q, run_hours=run_hours, verbose=verbose)
        if r is None:
            raise RuntimeError(f"Model produced no output for Q_ocean = {Q} W/m²")
        print(f"  dT_sst              = {r['dT_sst']:+.3f} K")
        print(f"  LWP      (mean)     = {r['LWP_mean']*1000:.1f} g/m²")
        print(f"  tau_vis  (mean)     = {r['tau_mean']:.2f}")
        print(f"  CF       (mean)     = {r['cf_mean']*100:.1f} %")
        print(f"  z_cbase  (mean)     = {r['z_cbase_mean']:.0f} m")
        print(f"  z_ctop   (mean)     = {r['z_ctop_mean']:.0f} m")
        print(f"  depth    (mean)     = {r['cloud_depth_mean']:.0f} m")
        print(f"  zi       (mean)     = {r['zi_mean']:.0f} m")
        print(f"  SHF      (mean)     = {r['shf_mean']:.1f} W/m²")
        print(f"  LHF      (mean)     = {r['lhf_mean']:.1f} W/m²")
        results.append(r)

    Q_vals        = np.array([r['Q_ocean']         for r in results])
    LWP_vals      = np.array([r['LWP_mean']        for r in results])
    dT_vals       = np.array([r['dT_sst']          for r in results])
    zi_vals       = np.array([r['zi_mean']          for r in results])
    z_ctop_vals   = np.array([r['z_ctop_mean']      for r in results])
    cf_vals       = np.array([r['cf_mean']          for r in results])
    depth_vals    = np.array([r['cloud_depth_mean'] for r in results])

    # Reference: index where Q = 0
    i_ref = np.argmin(np.abs(Q_vals))

    # ---- Summary table ----
    print("\n" + "=" * 85)
    print("  Summary (quasi-equilibrium means, second half of run)")
    print(f"  {'Q (W/m²)':>10}  {'dT_sst (K)':>11}  {'LWP (g/m²)':>11}"
          f"  {'CF (%)':>7}  {'z_cbase':>8}  {'z_ctop':>7}  {'depth':>7}  {'zi':>6}")
    print("  " + "-" * 80)
    for r in results:
        sign = '+' if r['Q_ocean'] >= 0 else ''
        print(f"  {sign}{r['Q_ocean']:9.0f}  {r['dT_sst']:+11.3f}  "
              f"{r['LWP_mean']*1000:11.1f}  {r['cf_mean']*100:7.1f}  "
              f"{r['z_cbase_mean']:8.0f}  {r['z_ctop_mean']:7.0f}  "
              f"{r['cloud_depth_mean']:7.0f}  {r['zi_mean']:6.0f}")

    # ---- Assertion 1: SST change monotonically increases with Q_ocean ----
    # The ocean runs cool overall (turbulent + LW losses dominate), so dT_sst
    # may remain negative even for positive Q.  The correct check is that a
    # larger Q produces a relatively warmer SST (less cooling / more warming).
    assert all(dT_vals[i + 1] > dT_vals[i] for i in range(len(dT_vals) - 1)), (
        f"dT_sst does not increase monotonically with Q_ocean.\n"
        f"  Q_vals  (W/m²) = {Q_vals}\n"
        f"  dT_sst  (K)    = {dT_vals}"
    )
    print("\n[PASS] SST change increases monotonically with Q_ocean "
          "(larger forcing → relatively warmer ocean)")

    # ---- Assertion 2: LWP monotonically decreasing with Q ----
    assert all(LWP_vals[i+1] < LWP_vals[i] for i in range(len(LWP_vals)-1)), (
        f"LWP does not decrease monotonically with increasing Q_ocean.\n"
        f"  Q_vals  = {Q_vals}\n"
        f"  LWP (g/m²) = {LWP_vals*1000}"
    )
    print("[PASS] LWP decreases monotonically with increasing Q_ocean")

    # ---- Assertion 3: negative forcing → LWP ≥ baseline ----
    LWP_ref = LWP_vals[i_ref]
    neg_cases = [(r['Q_ocean'], r['LWP_mean']) for r in results if r['Q_ocean'] < -1.0]
    for Q, lwp in neg_cases:
        assert lwp >= LWP_ref * 0.95, (   # 5% tolerance for numerical noise
            f"Q_ocean={Q:.0f} W/m² (cooling) should maintain or deepen cloud,\n"
            f"but LWP ({lwp*1000:.1f} g/m²) < baseline ({LWP_ref*1000:.1f} g/m²)"
        )
    print("[PASS] Negative ocean forcing maintains or deepens cloud")

    # ---- Assertion 4: positive forcing → LWP ≤ baseline ----
    pos_cases = [(r['Q_ocean'], r['LWP_mean']) for r in results if r['Q_ocean'] > 1.0]
    for Q, lwp in pos_cases:
        assert lwp <= LWP_ref * 1.05, (
            f"Q_ocean={Q:.0f} W/m² (warming) should thin cloud,\n"
            f"but LWP ({lwp*1000:.1f} g/m²) > baseline ({LWP_ref*1000:.1f} g/m²)"
        )
    print("[PASS] Positive ocean forcing thins cloud")

    # ---- Assertion 5: cloud fraction monotonically decreasing with Q ----
    # Warmer SST → BL warms from below → LCL rises → fewer cloudy layers.
    assert all(cf_vals[i+1] <= cf_vals[i] + 0.02 for i in range(len(cf_vals)-1)), (
        f"Cloud fraction does not decrease monotonically with Q_ocean.\n"
        f"  Q_vals  (W/m²) = {Q_vals}\n"
        f"  CF             = {cf_vals*100}"
    )
    print("[PASS] Cloud fraction decreases (or stays flat) with increasing Q_ocean")

    # ---- Assertion 6: cloud depth monotonically decreasing with Q ----
    # Warmer/drier BL → LCL rises toward cloud top → shallower cloud layer.
    assert all(depth_vals[i+1] <= depth_vals[i] + cfg.dz
               for i in range(len(depth_vals)-1)), (
        f"Cloud depth does not decrease monotonically with Q_ocean.\n"
        f"  Q_vals     (W/m²) = {Q_vals}\n"
        f"  depth      (m)    = {depth_vals}"
    )
    print("[PASS] Cloud depth decreases (or stays flat) with increasing Q_ocean")

    # ---- Assertion 7: cloud-top height monotonically decreasing with Q ----
    # Stronger positive forcing → BL warms → less LW cooling at cloud top
    # → cloud top descends.
    assert all(z_ctop_vals[i+1] <= z_ctop_vals[i] + cfg.dz
               for i in range(len(z_ctop_vals)-1)), (
        f"Cloud-top height does not decrease monotonically with Q_ocean.\n"
        f"  Q_vals     (W/m²) = {Q_vals}\n"
        f"  z_ctop     (m)    = {z_ctop_vals}"
    )
    print("[PASS] Cloud-top height decreases (or stays flat) with increasing Q_ocean")

    # ---- Assertion 8: zi monotonically non-increasing with Q ----
    assert all(zi_vals[i+1] <= zi_vals[i] + 50 for i in range(len(zi_vals)-1)), (
        f"Inversion height does not decrease with increasing Q_ocean.\n"
        f"  Q_vals  = {Q_vals}\n"
        f"  zi (m)  = {zi_vals}"
    )
    print("[PASS] Inversion height decreases (or stays flat) with increasing Q_ocean")

    print("\n[ALL TESTS PASSED]\n")
    return results


# ----------------------------------------------------------------
# Optional diagnostic plots
# ----------------------------------------------------------------

def plot_ocean_forcing(results):
    """Plot cloud and SST response as a function of Q_ocean."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    colors  = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(results)))
    times_h = results[0]['times'] / 3600.0

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle('Cloud Response to Ocean Heat Forcing — Slab Ocean SCM', fontsize=13)

    # --- Row 0: time series ---

    # Panel (0,0): SST time series
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        ax.plot(times_h, r['T_ssts'], color=c,
                label=f"Q={r['Q_ocean']:+.0f} W/m²")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('SST (K)')
    ax.set_title('Sea surface temperature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (0,1): LWP time series
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        ax.plot(times_h, r['LWPs'] * 1000, color=c,
                label=f"Q={r['Q_ocean']:+.0f} W/m²")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('LWP (g/m²)')
    ax.set_title('Liquid water path')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (0,2): Cloud fraction time series
    ax = axes[0, 2]
    for r, c in zip(results, colors):
        ax.plot(times_h, r['cfs'] * 100, color=c,
                label=f"Q={r['Q_ocean']:+.0f} W/m²")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Cloud fraction (%)')
    ax.set_title('Cloud fraction (% of column layers)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: more time series ---

    # Panel (1,0): Cloud-top height time series
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        ax.plot(times_h, r['z_ctops'], color=c,
                label=f"Q={r['Q_ocean']:+.0f} W/m²")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('z_ctop (m)')
    ax.set_title('Cloud-top height')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (1,1): Cloud depth time series
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        ax.plot(times_h, r['cloud_depths'], color=c,
                label=f"Q={r['Q_ocean']:+.0f} W/m²")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Cloud depth (m)')
    ax.set_title('Cloud depth (z_ctop − z_cbase)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (1,2): Inversion height time series
    ax = axes[1, 2]
    for r, c in zip(results, colors):
        ax.plot(times_h, r['zis'], color=c,
                label=f"Q={r['Q_ocean']:+.0f} W/m²")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('z_i (m)')
    ax.set_title('Inversion height')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 2: quasi-equilibrium sensitivity ---
    Q_vals      = np.array([r['Q_ocean']         for r in results])
    LWP_vals    = np.array([r['LWP_mean']         for r in results])
    dT_vals     = np.array([r['dT_sst']           for r in results])
    cf_vals     = np.array([r['cf_mean']          for r in results])
    z_ctop_vals = np.array([r['z_ctop_mean']      for r in results])
    depth_vals  = np.array([r['cloud_depth_mean'] for r in results])
    tau_vals    = np.array([r['tau_mean']         for r in results])

    # Panel (2,0): LWP vs Q
    ax = axes[2, 0]
    ax.plot(Q_vals, LWP_vals * 1000, 'o-', ms=8)
    ax.axvline(0, color='k', lw=0.7, ls='--')
    ax.set_xlabel('Q_ocean (W/m²)')
    ax.set_ylabel('LWP (g/m²)')
    ax.set_title('LWP vs. ocean forcing\n(2nd-half mean)')
    ax.grid(True, alpha=0.3)

    # Panel (2,1): Cloud fraction and depth vs Q
    ax = axes[2, 1]
    ax2 = ax.twinx()
    lns1 = ax.plot(Q_vals, cf_vals * 100, 's-', ms=8, color='tab:blue',
                   label='CF (%)')
    lns2 = ax2.plot(Q_vals, depth_vals, 'D--', ms=7, color='tab:green',
                    label='Depth (m)')
    ax.axvline(0, color='k', lw=0.7, ls='--')
    ax.set_xlabel('Q_ocean (W/m²)')
    ax.set_ylabel('Cloud fraction (%)', color='tab:blue')
    ax2.set_ylabel('Cloud depth (m)', color='tab:green')
    ax.set_title('Cloud fraction & depth vs. forcing')
    lns = lns1 + lns2
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (2,2): SST change and cloud-top height vs Q
    ax = axes[2, 2]
    ax2 = ax.twinx()
    lns1 = ax.plot(Q_vals, dT_vals, 's-', ms=8, color='tab:red',
                   label='ΔT_SST (K)')
    lns2 = ax2.plot(Q_vals, z_ctop_vals, 'D--', ms=7, color='tab:purple',
                    label='z_ctop (m)')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.7, ls='--')
    ax.set_xlabel('Q_ocean (W/m²)')
    ax.set_ylabel('ΔT_SST (K)', color='tab:red')
    ax2.set_ylabel('Cloud-top height (m)', color='tab:purple')
    ax.set_title('SST change & cloud-top height')
    lns = lns1 + lns2
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ocean_forcing.png', dpi=150, bbox_inches='tight')
    print("Saved ocean_forcing.png")

    import os
    if os.environ.get('DISPLAY') or sys.platform == 'darwin':
        plt.show()


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description='Ocean heat forcing cloud-response test for nighttime Sc SCM'
    )
    p.add_argument('--hours',   type=float, default=48,
                   help='Simulation length per case in hours (default: 48)')
    p.add_argument('--plot',    action='store_true',
                   help='Generate diagnostic plots after the test')
    p.add_argument('--verbose', action='store_true',
                   help='Print per-timestep SCM output')
    args = p.parse_args()

    results = test_ocean_forcing(run_hours=args.hours, verbose=args.verbose)

    if args.plot:
        plot_ocean_forcing(results)
