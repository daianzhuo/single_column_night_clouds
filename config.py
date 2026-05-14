"""
config.py
Model configuration and physical constants for the stratocumulus SCM.

References:
  - Gettelman & Morrison (2015), J. Climate: MG2 microphysics
  - Wood (2012), MWR: Stratocumulus clouds review
  - Stevens et al. (2003), MWR: DYCOMS-II RF01 case
  - de Roode & Duynkerke (1996): ASTEX case
"""

import numpy as np


class ModelConfig:
    """All model parameters in one place. Override fields for sensitivity runs."""

    # -----------------------------------------------------------------------
    # Vertical grid
    # -----------------------------------------------------------------------
    nz    = 60        # number of layers
    dz    = 50.0      # uniform layer thickness (m)
    # -> model top = nz * dz = 3000 m

    # -----------------------------------------------------------------------
    # Time stepping
    # -----------------------------------------------------------------------
    dt        = 20.0          # large time step (s)
    t_end     = 86400.0 * 5   # total simulation time (s), 5 days
    n_micro_substep = 3       # microphysics substeps per dt

    # -----------------------------------------------------------------------
    # Initial sounding  (DYCOMS-II RF01 / CGILS S12-like)
    # -----------------------------------------------------------------------
    p_surface   = 1017.8e2    # surface pressure (Pa)
    zi_init     = 840.0       # initial inversion height (m)
    theta_BL    = 289.0       # BL liquid-water potential temperature (K)
    theta_FT_jump = 8.5       # potential-temp jump at inversion (K)
    gamma_FT    = 6.0e-3      # FT dry lapse rate above inversion (K/m)
    qt_BL       = 9.0e-3      # BL total water mixing ratio (kg/kg)
    qt_FT       = 1.5e-3      # FT water vapour mixing ratio (kg/kg)

    # -----------------------------------------------------------------------
    # Large-scale subsidence  (no horizontal heat divergence)
    # -----------------------------------------------------------------------
    # Divergence D so that w_ls(z) = -D * z  (positive D -> subsidence)
    # Typical subtropical Sc: D ~ 3-7 x 10^-6 s^-1
    divergence  = 5.0e-6      # s^-1
    # -> w_ls(1 km) = -5 mm/s

    # -----------------------------------------------------------------------
    # Sea surface temperature
    # -----------------------------------------------------------------------
    sst_mode    = 'fixed'     # 'fixed' | 'slab'
    sst_K       = 292.0       # fixed SST (K)
    slab_depth  = 10.0        # slab-ocean depth (m) for 'slab' mode
    rho_ocean   = 1025.0      # sea-water density (kg/m^3)
    cp_ocean    = 3994.0      # sea-water specific heat (J/kg/K)

    # -----------------------------------------------------------------------
    # Surface (bulk aerodynamic)
    # -----------------------------------------------------------------------
    u_ref       = 7.0         # reference 10-m wind speed (m/s)
    z_ref       = 10.0        # reference height for wind (m)
    z0          = 1.5e-4      # roughness length (m)
    Cd          = 1.1e-3      # drag coefficient (neutral, approximate)
    Ch          = 1.1e-3      # heat transfer coefficient
    Cq          = 1.1e-3      # moisture transfer coefficient

    # -----------------------------------------------------------------------
    # Radiation  (nighttime -> SW off; only LW active)
    # -----------------------------------------------------------------------
    # SW disabled: this is the "night clouds" configuration
    F_sw_toa    = 0.0         # TOA solar flux (W/m^2) [0 = night]
    F_lw_dn_top = 10.0        # downwelling LW at model top (W/m^2)
    kappa_lw    = 85.0        # cloud LW mass-absorption coeff (m^2/kg)
    # εcloud ≈ 1 - exp(-kappa_lw * LWP): fully opaque for LWP > ~30 g/m^2

    # -----------------------------------------------------------------------
    # Turbulence  (K-diffusion)
    # -----------------------------------------------------------------------
    K_min       = 0.01        # minimum diffusivity (m^2/s)
    K_max       = 300.0       # maximum diffusivity in BL (m^2/s)
    l_mix       = 50.0        # mixing length (m)
    Pr_t        = 1.0         # turbulent Prandtl number

    # -----------------------------------------------------------------------
    # Cloud microphysics  (CAM6 MG2, warm phase)
    # -----------------------------------------------------------------------
    # --- Aerosol / CCN ---
    Nc_prescribed = 150.0e6   # prescribed cloud droplet number (#/kg)
    # activated whenever supersaturated; MBL value for clean marine Sc
    # (~ 100-200 /cm^3 is typical; here we use per-mass units)

    # --- Autoconversion  (Khairoutdinov & Kogan 2000) ---
    # dqr/dt = kc * ql^2.47 * Nc_cm3^(-1.79)
    kc_auto      = 1350.0     # coefficient (kg^-1 s^-1, dimensionally adjusted)
    alpha_auto   = 2.47       # ql exponent
    beta_auto    = -1.79      # Nc exponent (Nc in cm^-3)

    # --- Accretion  (Khairoutdinov & Kogan 2000) ---
    # dqr/dt = kc_acc * (ql * qr)^1.15
    kc_acc       = 67.0       # coefficient
    gamma_acc    = 1.15       # exponent

    # --- Rain evaporation ---
    # Using simplified form based on ventilation-corrected diffusion
    N0r          = 8.0e6      # Marshall-Palmer intercept (#/m^4)
    a_vt         = 841.997    # rain terminal velocity coefficient (m^1-b s^-1)
    b_vt         = 0.8        # rain terminal velocity exponent
    Ka_therm     = 2.5e-2     # thermal conductivity of air (W/m/K)
    Dv_vapor     = 2.26e-5    # molecular diffusivity of water vapor (m^2/s)

    # --- Cloud sedimentation ---
    # Stokes settling for cloud droplets
    mu_air       = 1.81e-5    # dynamic viscosity of air (Pa s)

    # --- Thresholds ---
    ql_min       = 1.0e-9     # minimum cloud water for microphysics (kg/kg)
    qr_min       = 1.0e-9     # minimum rain water (kg/kg)
    Nr_min       = 1.0e-4     # minimum rain number (#/kg)

    # -----------------------------------------------------------------------
    # Physical constants
    # -----------------------------------------------------------------------
    Rd    = 287.04            # gas constant, dry air (J/kg/K)
    Rv    = 461.50            # gas constant, water vapour (J/kg/K)
    cp    = 1005.7            # specific heat, dry air (J/kg/K)
    Lv    = 2.501e6           # latent heat of vaporisation (J/kg)
    g     = 9.81              # gravitational acceleration (m/s^2)
    rho_w = 1000.0            # density of liquid water (kg/m^3)
    p0    = 1.0e5             # reference pressure (Pa)
    kappa = Rd / cp           # Poisson constant (dimensionless)
    sigma = 5.6704e-8         # Stefan-Boltzmann (W/m^2/K^4)
    eps   = Rd / Rv           # ratio of gas constants ~ 0.622

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    output_interval = 300.0   # seconds between output snapshots


# Singleton instance used by all modules
cfg = ModelConfig()
