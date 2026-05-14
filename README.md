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

## Known limitations

Identified by a stress test (`test_stress.py`, 37 cases × 6 h each) sweeping SST, divergence, inversion strength, BL moisture, CCN, wind speed, and inversion height.

### Valid operating range

The model is calibrated for the marine stratocumulus regime:

| Parameter | Calibrated range | Notes |
|---|---|---|
| SST | 285–300 K | Outside this range cloud persists but behaviour is unrealistic (see below) |
| Large-scale divergence | 1–12 × 10⁻⁶ s⁻¹ | D > ~15 × 10⁻⁶ s⁻¹ dissipates the cloud |
| BL total water | 9–13 g/kg | < ~8 g/kg: LCL rises above BL, no cloud forms |
| Initial inversion height | 400–1500 m | < 400 m: BL too shallow; > 2000 m: microphysics overflows |
| Inversion jump | 2–20 K | Δθ = 0 breaks the inversion-height detector |
| Wind speed | 0.3–20 m/s | Neutral-stability bulk coefficients assumed throughout |
| Nc | 5–2000 /cm³ | Full CCN range numerically stable (precipitation suppression effect captured) |

### Regime limits (cloud dissipation)

The following conditions produce a cloud-free column within 6 h:

| Case | Cause |
|---|---|
| D > ~15 × 10⁻⁶ s⁻¹ | Subsidence compression overcomes LW-driven moistening |
| qt_BL ≤ 7 g/kg | LCL is above the inversion — BL air cannot saturate |
| zi_init ≤ 300 m | BL too shallow; no room for a Sc layer to establish |
| Warm SST + dry BL (SST=298 K, qt=5 g/kg) | Combined: moisture too low and SST-driven LHF cannot compensate |

### Numerical limits

| Issue | Trigger | Symptom |
|---|---|---|
| **Precipitation overflow** | zi_init ≥ ~2400 m (cloud depth > ~700 m) | `precip_rate` → 10⁷⁰ mm/hr; T and qv remain finite. Root cause: the Marshall-Palmer slope parameter `λ = (π ρ_w N₀ / ρ qr)^(1/4)` diverges as `qr → 0` over a very deep column, causing `Nr` to overflow in the accretion step. |

### Physical design assumptions and their consequences

1. **No shortwave radiation (nighttime only)**
   SW is hardcoded to zero (`F_sw_toa = 0`). For SST ≥ 305 K the model produces *more* cloud as warm SST drives higher LHF → higher qt → more condensate. In reality, daytime SW would burn the cloud off; this model cannot simulate that transition.

2. **No cloud-top entrainment instability (CTEI)**
   Turbulent entrainment of warm, dry FT air at the inversion is parameterised only through the O'Brien K-profile. The model does not implement an explicit CTEI criterion (Randall 1980; Deardorff 1980), so it cannot represent the radiatively driven cloud dissolution that occurs when Δθ is very small or the cloud is very optically thick.

3. **Inversion-height detector fails at Δθ = 0**
   `find_inversion_height()` picks the level of maximum dθ/dz. With no inversion (Δθ = 0 K), it returns the model-top boundary (z ≈ 2900 m), causing the FT nudging and K-profile to use a spurious inversion height.

4. **Slab-ocean thermal inertia limits SST response**
   C_slab = ρ_ocean · cp_ocean · d_slab ≈ 4.1 × 10⁷ J/(m² K). A 100 W/m² forcing shift produces only ≈ 0.4 K SST change over 48 h, so SST-mediated cloud feedbacks develop slowly.

5. **No horizontal advection or mesoscale organisation**
   The column is closed to horizontal heat and moisture transport. Real Sc regions are maintained partly by cold advection off continents and by organised Sc-to-Cu transitions driven by SST gradients — none of which are represented.

6. **Fixed neutral-stability surface-flux coefficients**
   Cd = Ch = Cq = 1.1 × 10⁻³ at all wind speeds and stabilities. This overestimates heat exchange in near-calm conditions (u < 2 m/s) and may underestimate it in strongly stable BLs.

### Stress-test outcome table

Run `python3 test_stress.py` to reproduce. Each case is 6 model hours. Status: **STABLE** = cloud present and numerically healthy; *dissipated* = cloud lost; `precip_OVF` = precipitation rate overflow.

| Case | Δ from baseline | Status | LWP_f (g/m²) | zi_f (m) |
|---|---|---|---|---|
| baseline | — | STABLE | 57 | 850 |
| sst_280 | SST −12 K | STABLE | 313 | 850 |
| sst_310 | SST +18 K | STABLE† | 209 | 850 |
| div_10e-6 | D doubled | STABLE | 21 | 850 |
| div_20e-6 | D × 4 | *dissipated* | 0 | 850 |
| inv_0K | no inversion | STABLE‡ | 118 | 850 |
| inv_25K | very strong inversion | STABLE | 13 | 850 |
| qt_7g | dry BL | *dissipated* | 0 | 850 |
| qt_16g | very moist BL | STABLE | 1012 | 800 |
| Nc_5 | ultra-clean | STABLE | 41 | 850 |
| Nc_2000 | heavily polluted | STABLE | 64 | 850 |
| wind_25 | storm-force | STABLE | 251 | 850 |
| zi_300 | shallow BL | *dissipated* | 0 | 300 |
| zi_1500 | deep BL | STABLE | 631 | 1450 |
| zi_2500 | near model top | `precip_OVF` | — | 2450 |

† Physically unrealistic — no SW to burn cloud off.
‡ Inversion-height detector returns spurious value.

## References

- Gettelman & Morrison (2015), J. Climate — MG2 microphysics
- Khairoutdinov & Kogan (2000), J. Atmos. Sci. — autoconversion/accretion
- Stevens et al. (2003), MWR — DYCOMS-II RF01 case
- Wood (2012), MWR — stratocumulus clouds review
- Troen & Mahrt (1986) / O'Brien (1970) — K-profile parameterisation
- Bolton (1980), MWR — saturation vapour pressure formula
