# Nighttime Stratocumulus Single-Column Model

A single-column climate model for the nocturnal marine stratocumulus-topped boundary layer, with warm-phase cloud microphysics based on the CAM6 MG2 scheme.

## Physics

### Why a layer-by-layer SCM?

A box model was explicitly considered and rejected. Stratocumulus dynamics are driven by longwave cooling concentrated at cloud **top** — a box model averages this forcing over the whole cloud, destroying the key thermodynamic feedback. The sharp temperature inversion (~50 m thick), vertical profile of cloud liquid water, and drizzle sedimentation all require vertical resolution. The model uses **60 layers, dz = 50 m, z = 0–3000 m**.

### Processes

| Module | Process |
|---|---|
| `microphysics.py` | CAM6 MG2 warm-phase: droplet activation, KK2000 autoconversion/accretion, Stokes cloud sedimentation, Marshall-Palmer rain sedimentation, ventilated rain evaporation |
| `radiation.py` | LW two-stream grey-body (no SW — nighttime). Broadband κ = 0.40 m²/kg calibrated to reproduce ~80 W/m² cloud-top flux divergence (DYCOMS-II RF01) |
| `turbulence.py` | O'Brien K-profile driven by cloud-top LW buoyancy flux + surface fluxes. K → 0 exactly at cloud-top to preserve the inversion. Fully-implicit tridiagonal diffusion of conserved variables θ_l and q_t |
| `dynamics.py` | Large-scale subsidence w = −D·z (no horizontal heat divergence). Free-troposphere nudging above the inversion (τ = 1 hr) |
| `surface.py` | Bulk aerodynamic SHF and LHF. SST fixed or slab-ocean (prognostic) |

### Model schematic

```
 z (m)
  3000 ┤· · · · · · · · · · · · · · model top · · · · · · · · · · ·
       │                              FT nudging (τ = 1 hr) active
       │         w↓ = −D·z  (subsidence throughout column)
       │
       │  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       │              free troposphere  (θ, qv nudged to sounding)
       │  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       │
   900 ┤══════════════════  temperature inversion  ═══════════════
       │                    (Δθ ≈ 8.5 K over ~50 m, K → 0 here)
       │
       │  ↑↑↑  LW flux divergence ~80 W/m² drives cloud-top cooling
   840 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  cloud top  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
       │▓▓▓▓▓▓▓▓  stratocumulus  ▓▓▓▓▓▓▓▓   two-stream LW radiation
       │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (qc, Nc, qr, Nr)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   700 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  cloud base  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
       │                    ↓↓  drizzle sedimentation (Stokes / Marshall-Palmer)
       │
       │  ╔══════════════════════════════════════════════════╗
       │  ║  K-profile turbulent mixing                      ║
       │  ║  K ∝ w★ · z · (1 − z/h_BL)²   (O'Brien 1970)  ║
       │  ║  w★ = (B_surf · h_BL)^(1/3);  B driven by LW   ║
       │  ║  implicit θ_l, q_t diffusion  (Thomas algorithm) ║
       │  ╚══════════════════════════════════════════════════╝
       │
     0 ┤─────────────────  ocean surface (z = 0)  ──────────────
       │   SST = 292 K    SHF ↑   LHF ↑     (bulk aerodynamic)
       │   fixed SST  or  slab ocean: C_slab · dT_sst/dt = Q_net + Q_ocean
```

Key variables: `θ_l` = liquid-water potential temperature, `q_t` = total water, `Nc` = droplet number concentration. The 60-layer column (dz = 50 m) resolves the ~140 m cloud layer, the sharp inversion, and the sub-cloud mixed layer within a 3 km domain.

### Initial sounding

DYCOMS-II RF01 / CGILS S12-like:

```
θ_BL = 289 K,  q_t,BL = 9 g/kg
θ_jump = 8.5 K at z_i = 840 m
γ_FT = 6 K/km,  q_v,FT = 1.5 g/kg
D = 5 × 10⁻⁶ s⁻¹  →  w(1 km) = −5 mm/s
N_c = 150 /cm³ (clean marine CCN)
SST = 292 K
```

### Time integration

Operator splitting per time step (dt = 20 s):

1. Large-scale subsidence (upwind advection of θ, q_t)
2. Free-troposphere nudging (above inversion only)
3. Surface flux computation
4. Turbulent mixing (implicit, conserved variables θ_l and q_t)
5. Longwave radiation (explicit heating rate)
6. Saturation adjustment (Newton-Raphson, 10 iterations)
7. Cloud microphysics (3 sub-steps)

## Usage

```bash
# 5-day run, fixed SST (default)
python3 run_scm.py

# Slab ocean SST (prognostic)
python3 run_scm.py --sst slab

# 2-day run, custom SST
python3 run_scm.py --days 2 --sst_k 290.0

# Generate diagnostic plots (requires matplotlib)
python3 run_scm.py --days 5 --plot

# All options
python3 run_scm.py --help
```

Output is saved to `scm_output.npz` (numpy archive). Load with:

```python
import numpy as np
data = np.load('scm_output.npz')
# Keys: time, z, T, theta, qv, qc, Nc, qr, Nr, p, rho,
#       T_sst, shf, lhf, LWP, tau_vis, zi, z_cbase, z_ctop, precip_rate
```

## Verified behaviour

**Fixed SST, 5-day run:**
- Inversion locked at 850 m (stable)
- Cloud persists throughout; LWP ≈ 50–170 g/m² (nighttime deepening)
- SHF ≈ 15–22 W/m², LHF ≈ 55–95 W/m²

**Slab ocean, 2-day run:**
- SST cools at ~0.21 K/day, consistent with net −100 W/m² ocean heat loss
- Cloud deepens as SST decreases

## File structure

```
config.py       — all tunable parameters (ModelConfig / cfg singleton)
grid.py         — VerticalGrid: z, zi, hydrostatic pressure
thermo.py       — e_sat (Bolton), q_sat, exner, saturation adjustment
microphysics.py — CAM6 MG2 warm-phase microphysics
radiation.py    — LW two-stream solver
turbulence.py   — K-profile + implicit diffusion
dynamics.py     — subsidence, FT nudging, inversion detection
surface.py      — SST modes, bulk surface fluxes
scm.py          — StratocumulusSCM integrator class
run_scm.py      — CLI driver, output, optional plots
```

## References

- Gettelman & Morrison (2015), J. Climate — MG2 microphysics
- Khairoutdinov & Kogan (2000), J. Atmos. Sci. — autoconversion/accretion
- Stevens et al. (2003), MWR — DYCOMS-II RF01 case
- Wood (2012), MWR — stratocumulus clouds review
- Troen & Mahrt (1986) / O'Brien (1970) — K-profile parameterisation
- Bolton (1980), MWR — saturation vapour pressure formula
