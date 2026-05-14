"""
run_scm.py
Driver script for the nighttime stratocumulus SCM.

Usage:
  python run_scm.py                    # 5-day run, fixed SST, nighttime
  python run_scm.py --sst slab         # slab ocean
  python run_scm.py --days 2           # shorter run
  python run_scm.py --plot             # show plots after run

Output:
  Saved to 'scm_output.npz' (numpy archive).
  Plots (optional): vertical profiles, time series.
"""

import argparse
import numpy as np
import os
import sys

from config import cfg
from scm import StratocumulusSCM
import thermo as th
import dynamics as dyn
import radiation as rad


# ============================================================
#  Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='Nighttime Stratocumulus SCM')
    p.add_argument('--sst',      default='fixed', choices=['fixed', 'slab'],
                   help='SST mode: fixed or slab ocean (default: fixed)')
    p.add_argument('--sst_k',    type=float, default=cfg.sst_K,
                   help=f'Fixed SST in Kelvin (default: {cfg.sst_K} K)')
    p.add_argument('--days',     type=float, default=cfg.t_end / 86400,
                   help='Simulation length in days (default: 5)')
    p.add_argument('--dt',       type=float, default=cfg.dt,
                   help=f'Time step in seconds (default: {cfg.dt} s)')
    p.add_argument('--output',   default='scm_output.npz',
                   help='Output file (default: scm_output.npz)')
    p.add_argument('--plot',     action='store_true',
                   help='Show plots after run')
    p.add_argument('--no-verbose', action='store_true',
                   help='Suppress per-step print output')
    return p.parse_args()


# ============================================================
#  Save / load history
# ============================================================

def save_history(history, filename):
    """Save the model history list to a .npz file."""
    keys = history[0].keys()
    arrays = {}
    for key in keys:
        vals = [snap[key] for snap in history]
        arrays[key] = np.array(vals)
    np.savez(filename, **arrays)
    print(f"Saved {len(history)} snapshots to '{filename}'")


def load_history(filename):
    """Load history from .npz file. Returns dict of arrays."""
    data = np.load(filename)
    return dict(data)


# ============================================================
#  Plotting
# ============================================================

def plot_results(history):
    """Generate diagnostic plots from model history."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    times = np.array([s['time'] for s in history]) / 3600.0   # hours

    # ------ Figure 1: Time series of bulk diagnostics ------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Stratocumulus SCM - Nighttime (no SW)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(times, [s['LWP'] * 1000 for s in history])
    ax.set_ylabel('LWP (g/m²)')
    ax.set_xlabel('Time (h)')
    ax.set_title('Liquid Water Path')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, [s['zi']     for s in history], label='Inversion')
    ax.plot(times, [s['z_cbase'] for s in history], label='Cloud base', ls='--')
    ax.plot(times, [s['z_ctop']  for s in history], label='Cloud top',  ls=':')
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('Time (h)')
    ax.set_title('Boundary Layer Heights')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(times, [s['T_sst'] for s in history])
    ax.set_ylabel('T_SST (K)')
    ax.set_xlabel('Time (h)')
    ax.set_title('Sea Surface Temperature')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(times, [s['shf'] for s in history], label='SHF')
    ax.plot(times, [s['lhf'] for s in history], label='LHF')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Flux (W/m²)')
    ax.set_xlabel('Time (h)')
    ax.set_title('Surface Fluxes')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(times, [s['precip_rate'] * 3600 for s in history])
    ax.set_ylabel('Precip (mm/hr)')
    ax.set_xlabel('Time (h)')
    ax.set_title('Surface Precipitation Rate')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(times, [s['tau_vis'] for s in history])
    ax.set_ylabel('Optical depth τ')
    ax.set_xlabel('Time (h)')
    ax.set_title('Cloud Visible Optical Depth')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scm_timeseries.png', dpi=150, bbox_inches='tight')
    print("Saved scm_timeseries.png")

    # ------ Figure 2: Vertical profiles (initial, 24 h, final) ------
    snapshots = {}
    idx_init  = 0
    idx_24h   = np.argmin(np.abs(times - 24.0))
    idx_final = len(history) - 1
    snap_labels = {idx_init: 'Initial', idx_24h: '24 h', idx_final: f'{times[-1]:.0f} h'}

    fig, axes = plt.subplots(1, 5, figsize=(16, 7), sharey=True)
    fig.suptitle('Vertical Profiles — Stratocumulus SCM', fontsize=13)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ci, (idx, label) in enumerate(snap_labels.items()):
        s = history[idx]
        z = s['z']

        axes[0].plot(s['theta'], z, color=colors[ci], label=label)
        axes[1].plot(s['qv'] * 1000, z, color=colors[ci], label=label)
        axes[2].plot(s['qc'] * 1000, z, color=colors[ci], label=label)
        axes[3].plot(s['qr'] * 1e6, z, color=colors[ci], label=label)
        axes[4].plot(s['T'],  z, color=colors[ci], label=label)

    titles   = ['θ (K)', 'qv (g/kg)', 'qc (g/kg)', 'qr (mg/kg)', 'T (K)']
    xlabels  = ['Potential Temp.', 'Water Vapour', 'Cloud Liquid', 'Rain Water', 'Temperature']
    for ax, t, xl in zip(axes, titles, xlabels):
        ax.set_xlabel(t)
        ax.set_title(xl)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Height (m)')
    axes[0].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('scm_profiles.png', dpi=150, bbox_inches='tight')
    print("Saved scm_profiles.png")

    if os.environ.get('DISPLAY') or sys.platform == 'darwin':
        plt.show()


# ============================================================
#  Initial condition helper: print sounding info
# ============================================================

def print_initial_sounding(model):
    """Print summary of the initial sounding."""
    z  = model.grid.z
    T  = model.T
    qv = model.qv
    qc = model.qc
    p  = model.p

    theta  = th.theta_from_T(T, p)
    qs     = th.q_sat(T, p)

    print("\n=== Initial sounding ===")
    print(f"{'z(m)':>7}  {'T(K)':>7}  {'θ(K)':>7}  "
          f"{'qv(g/kg)':>9}  {'qc(g/kg)':>9}  {'RH(%)':>6}  {'p(hPa)':>7}")
    for k in range(0, model.grid.nz, 5):
        rh = qv[k] / max(qs[k], 1e-8) * 100
        print(f"{z[k]:7.0f}  {T[k]:7.2f}  {theta[k]:7.2f}  "
              f"{qv[k]*1000:9.3f}  {qc[k]*1000:9.4f}  {rh:6.1f}  {p[k]/100:7.2f}")


# ============================================================
#  Main
# ============================================================

def main():
    args = parse_args()

    # Override config with CLI args
    cfg.dt    = args.dt
    cfg.t_end = args.days * 86400.0
    cfg.sst_mode = args.sst
    cfg.sst_K    = args.sst_k

    print("=" * 60)
    print(" Nighttime Stratocumulus Single-Column Model")
    print(f" SST mode  : {cfg.sst_mode}  ({cfg.sst_K:.1f} K)")
    print(f" Run length: {args.days:.1f} days  ({cfg.t_end/3600:.0f} h)")
    print(f" Timestep  : {cfg.dt:.0f} s")
    print(f" Grid      : {cfg.nz} layers, dz={cfg.dz:.0f} m, top={cfg.nz*cfg.dz:.0f} m")
    print(f" Divergence: D={cfg.divergence:.1e} s^-1  "
          f"-> w(1km)={-cfg.divergence*1000*1000:.1f} mm/s")
    print("=" * 60)

    # --- Build and initialise model ---
    model = StratocumulusSCM(sst_mode=cfg.sst_mode, sst_K=cfg.sst_K)
    model.initialize()

    print_initial_sounding(model)
    model.current_state_summary()

    # --- Run ---
    print(f"\nStarting integration ...")
    history = model.run(
        t_end=cfg.t_end,
        output_interval=cfg.output_interval,
        verbose=(not args.no_verbose),
    )

    # --- Save ---
    save_history(history, args.output)

    # --- Final summary ---
    model.current_state_summary()

    # --- Plot ---
    if args.plot:
        plot_results(history)


if __name__ == '__main__':
    main()
