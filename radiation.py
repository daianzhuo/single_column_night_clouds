"""
radiation.py
Longwave radiative transfer for the nighttime stratocumulus SCM.

Approach: two-stream grey-body radiation with a spectrally-averaged
absorption coefficient for cloud liquid water.

Physics:
  - Upwelling (F+) and downwelling (F-) fluxes computed layer by layer.
  - Cloud emissivity: epsilon = 1 - exp(-kappa_lw * LWP)
    where kappa_lw ~ 85 m^2/kg (Stephens 1978; Wood 2012).
  - Clear-sky water-vapour emission included via a simple emissivity
    parameterisation based on precipitable water.
  - No solar radiation (nighttime configuration).

The net radiative heating rate:
  Q_rad[k] = -(F+[k+1] - F+[k] + F-[k] - F-[k+1]) / (rho[k] * cp * dz)
            = -dF_net/dz / (rho * cp)

where F_net = F+ - F- is the net upward flux.

References:
  Stephens (1978), J. Atmos. Sci., 35, 2111-2122.
  Larson et al. (2007), J. Climate, 20, 1586.
  Wood (2012), MWR review.
"""

import numpy as np
from config import cfg


# ============================================================
#  Layer optical properties
# ============================================================

def layer_emissivity(qc, qr, dz, rho):
    """
    Broadband LW emissivity of a single model layer.

    epsilon = 1 - exp(-kappa_lw * LWP)
    LWP = rho * (qc + qr) * dz   [kg/m^2]

    Parameters
    ----------
    qc, qr : mixing ratios (kg/kg)
    dz     : layer thickness (m)
    rho    : air density (kg/m^3)

    Returns
    -------
    eps : emissivity [0, 1]
    """
    LWP = rho * (np.maximum(qc, 0.0) + np.maximum(qr, 0.0)) * dz
    return 1.0 - np.exp(-cfg.kappa_lw * LWP)


def clear_sky_emissivity(qv, dz, rho, T, p):
    """
    Clear-sky LW emissivity of a single 50-m layer due to water vapour.

    The Sasamori formula is a column-integrated expression and MUST NOT be
    applied per layer.  For individual thin layers the broadband optical
    depth is:
        tau_wv = kappa_eff * W_layer
    where W_layer = rho * qv * dz  (kg/m^2) and kappa_eff is an effective
    broadband absorption coefficient (~0.1 m^2/kg) that represents only
    the atmospheric window region (8-12 μm) relevant to Sc LW cooling
    (Duynkerke 1993; Larson et al. 2007).  The full FT column is nearly
    transparent in this window, consistent with a prescribed ~10 W/m^2
    downwelling LW at model top.
    """
    # Effective broadband (full LW spectrum) absorption coefficient.
    # This ~0.4 m^2/kg combines the H2O rotational band, 6.3 μm band,
    # CO2 15 μm band, and window continuum.  Calibrated so that the
    # 43-layer FT column (z=875 m to 3 km, qv=1.5 g/kg) produces
    # F_dn ~ 295 W/m^2 above cloud top, consistent with DYCOMS-II RF01
    # radiosondes and matching the observed ~65 W/m^2 cloud-top LW flux
    # divergence that drives Sc convection (Stevens et al. 2003).
    W_layer  = rho * np.maximum(qv, 0.0) * dz   # kg/m^2
    kappa_wv = 0.40                               # m^2/kg (broadband effective)
    tau_wv   = kappa_wv * W_layer
    return 1.0 - np.exp(-tau_wv)


# ============================================================
#  Two-stream LW solver (grey body, layer-by-layer)
# ============================================================

def lw_two_stream(T, p, qv, qc, qr, rho, dz, T_sfc,
                  F_lw_dn_top=None):
    """
    Compute LW upwelling and downwelling fluxes at each layer interface.

    Interface ordering:
      zi[0] = surface (z=0)
      zi[k] = top of layer k-1 / bottom of layer k
      zi[nz] = model top

    Algorithm (starting from surface upward for F+, model top downward for F-):

      F+[k+1] = F+[k] * (1-eps[k]) + eps[k] * sigma*T[k]^4

      F-[k]   = F-[k+1] * (1-eps[k]) + eps[k] * sigma*T[k]^4

    Combined emissivity = cloud + clear-sky water vapour (saturated grey gas).

    Parameters
    ----------
    T      : (nz,) temperature (K)
    p      : (nz,) pressure (Pa)
    qv     : (nz,) water vapour mixing ratio (kg/kg)
    qc     : (nz,) cloud liquid (kg/kg)
    qr     : (nz,) rain (kg/kg)
    rho    : (nz,) density (kg/m^3)
    dz     : scalar layer thickness (m)
    T_sfc  : scalar surface temperature (K)
    F_lw_dn_top : scalar downwelling LW at model top (W/m^2)
                  defaults to cfg.F_lw_dn_top

    Returns
    -------
    F_up   : (nz+1,) upwelling flux at interfaces (W/m^2)
    F_dn   : (nz+1,) downwelling flux at interfaces (W/m^2)
    Q_rad  : (nz,) radiative heating rate (K/s)
    F_net_sfc : net upward flux at surface (W/m^2) [>0 = upward, cooling surface]
    """
    if F_lw_dn_top is None:
        F_lw_dn_top = cfg.F_lw_dn_top

    nz   = len(T)
    nzi  = nz + 1

    # --- Emissivity per layer ---
    eps_cld = layer_emissivity(qc, qr, dz, rho)
    eps_wv  = clear_sky_emissivity(qv, dz, rho, T, p)
    # Combined (treat as serial absorbers: total tau = tau_cld + tau_wv)
    tau_cld = -np.log(np.maximum(1.0 - eps_cld, 1.0e-10))
    tau_wv  = -np.log(np.maximum(1.0 - eps_wv,  1.0e-10))
    tau_tot = tau_cld + tau_wv
    eps_tot = 1.0 - np.exp(-tau_tot)

    # Blackbody emission per layer
    B = cfg.sigma * T**4       # (nz,) W/m^2 per unit emissivity

    # --- Upwelling flux  F+  (surface -> top) ---
    F_up = np.zeros(nzi)
    F_up[0] = cfg.sigma * T_sfc**4    # surface emits as blackbody

    for k in range(nz):
        # Transmitted + emitted by layer k
        F_up[k+1] = F_up[k] * (1.0 - eps_tot[k]) + eps_tot[k] * B[k]

    # --- Downwelling flux  F-  (top -> surface) ---
    F_dn = np.zeros(nzi)
    F_dn[nz] = F_lw_dn_top

    for k in range(nz-1, -1, -1):
        F_dn[k] = F_dn[k+1] * (1.0 - eps_tot[k]) + eps_tot[k] * B[k]

    # --- Radiative heating rate ---
    # Q = -d(F_net)/dz / (rho * cp)
    # F_net[k] = F_up[k] - F_dn[k]  at lower interface of layer k
    Q_rad = np.zeros(nz)
    for k in range(nz):
        F_net_bottom = F_up[k]   - F_dn[k]
        F_net_top    = F_up[k+1] - F_dn[k+1]
        dF_net       = F_net_top - F_net_bottom   # positive = divergence (cooling)
        Q_rad[k]     = -dF_net / (rho[k] * cfg.cp * dz)

    # Net LW flux at surface (upward positive)
    F_net_sfc = F_up[0] - F_dn[0]

    return F_up, F_dn, Q_rad, F_net_sfc


# ============================================================
#  Diagnostic: cloud-top LW cooling
# ============================================================

def cloud_top_flux_jump(F_up, F_dn, k_top):
    """
    Compute the net LW flux divergence at cloud top (W/m^2).
    This is the dominant thermodynamic driver of Sc convection.

    delta_F = F_net(just above cloud top) - F_net(just below cloud top)
    """
    if k_top is None:
        return 0.0
    F_net_above = F_up[k_top+1] - F_dn[k_top+1]
    F_net_below = F_up[k_top]   - F_dn[k_top]
    return F_net_above - F_net_below


# ============================================================
#  Liquid water path  (diagnostic)
# ============================================================

def liquid_water_path(qc, rho, dz):
    """Vertically integrated cloud liquid water path (kg/m^2)."""
    return np.sum(rho * np.maximum(qc, 0.0) * dz)


def cloud_optical_depth(qc, qr, rho, dz):
    """
    Visible optical depth of cloud column (diagnostic).
    tau_vis = 3 * LWP / (2 * rho_w * r_eff)
    Simplified using constant r_eff ~ 10 um for marine Sc.
    """
    LWP  = liquid_water_path(qc, rho, dz)
    r_eff = 10.0e-6   # 10 microns
    return 3.0 * LWP / (2.0 * cfg.rho_w * r_eff)
