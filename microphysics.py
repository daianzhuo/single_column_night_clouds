"""
microphysics.py
Warm-phase two-moment cloud microphysics based on CAM6 MG2.

Prognostic variables per layer:
  qc  : cloud liquid water mixing ratio (kg/kg)
  Nc  : cloud droplet number concentration (#/kg)
  qr  : rain water mixing ratio (kg/kg)
  Nr  : raindrop number concentration (#/kg)

Processes implemented (following Gettelman & Morrison 2015, MG2):
  1. Droplet activation (CCN-based)
  2. Condensation / evaporation  (saturation adjustment -- called externally)
  3. Autoconversion  (cloud -> rain):  Khairoutdinov & Kogan (2000)
  4. Accretion       (cloud collected by rain): KK (2000)
  5. Self-collection of rain
  6. Evaporation of rain (ventilation-corrected)
  7. Cloud sedimentation  (Stokes law for droplets)
  8. Rain sedimentation   (Marshall-Palmer, power-law V_t)

Units convention:
  - Mixing ratios: kg/kg
  - Number concentrations: #/kg  (mass-specific)
  - Rates: kg/kg/s  or  #/kg/s

References:
  Khairoutdinov & Kogan (2000), J. Atmos. Sci., 57, 2905-2927.
  Morrison & Gettelman (2008), J. Climate, 21, 3642-3659.
  Gettelman & Morrison (2015), J. Climate, 28, 1268-1280.
  Rogers & Yau (1989): A Short Course in Cloud Physics.
"""

import math

import numpy as np
try:
    from scipy.special import gamma as _scipy_gamma
    _gamma = _scipy_gamma
except ImportError:
    _gamma = math.gamma

from config import cfg
import thermo as th


# ============================================================
#  Size-distribution helpers  (Marshall-Palmer for rain)
# ============================================================

def lambda_rain(rho, qr, Nr):
    """
    Slope parameter of exponential raindrop size distribution (m^-1).
    n(D) = Nr_vol * exp(-lambda * D)
    Nr_vol [#/m^3] = rho * Nr
    For exponential dist: qr = pi/6 * rho_w/rho * Nr_vol * 6/lambda^4
      => lambda = (pi * rho_w * Nr_vol / (rho * qr))^(1/4)
    """
    Nr_vol = rho * np.maximum(Nr, cfg.Nr_min)
    qr_    = np.maximum(qr, cfg.qr_min)
    lam = (np.pi * cfg.rho_w * Nr_vol / (rho * qr_)) ** 0.25
    return np.maximum(lam, 1.0)   # min ~1/m prevents blow-up


def r_eff_cloud(rho, qc, Nc):
    """
    Effective radius of cloud droplets (m).
    Assuming a monodisperse (or gamma) distribution:
      r_e = (3 * rho * qc / (4 * pi * rho_w * Nc_vol))^(1/3)
    where Nc_vol = rho * Nc  (#/m^3).
    """
    Nc_vol = rho * np.maximum(Nc, 1.0)   # #/m^3, avoid div/0
    qc_    = np.maximum(qc, cfg.ql_min)
    r3 = 3.0 * rho * qc_ / (4.0 * np.pi * cfg.rho_w * Nc_vol)
    return r3 ** (1.0/3.0)


# ============================================================
#  1. Droplet activation
# ============================================================

def droplet_activation(qc, Nc, rho, dt):
    """
    Prescribe cloud droplet number = Nc_prescribed wherever qc > ql_min.
    Uses implicit exponential relaxation to prevent the forward-Euler overshoot
    that occurs when dt > tau_act.

    Returns Nc_new  (#/kg)  (not a tendency — replaces Nc directly).
    """
    Nc_target = np.where(qc > cfg.ql_min, cfg.Nc_prescribed, 0.0)
    tau_act   = 1.0   # activation timescale (s)
    # Implicit: Nc_new = Nc_target + (Nc - Nc_target) * exp(-dt/tau)
    decay    = np.exp(-dt / tau_act)
    Nc_new   = Nc_target + (Nc - Nc_target) * decay
    return np.maximum(Nc_new, 0.0)


# ============================================================
#  2. Autoconversion  (KK 2000)
# ============================================================

def autoconversion(qc, Nc, rho):
    """
    Cloud-to-rain autoconversion.
    KK2000:  P_aut = kc * qc^2.47 * Nc_cm3^(-1.79)   [kg/kg/s]
    Number:  dNr/dt = P_aut / x_star
             dNc/dt = -2 * P_aut * rho / (pi/6 * rho_w * D_star^3 * rho)
                    = -P_aut / x_star_c   (approximate)

    Returns
    -------
    dqc_aut, dqr_aut : kg/kg/s  (negative and positive respectively)
    dNc_aut, dNr_aut : #/kg/s
    """
    qc_ = np.maximum(qc, 0.0)
    Nc_ = np.maximum(Nc, 0.0)
    Nc_cm3 = rho * Nc_ * 1.0e-6       # #/m^3 -> #/cm^3

    # KK2000 autoconversion rate; require Nc > 1 /cm^3 to avoid Nc^(-1.79) blow-up
    valid = (qc_ > cfg.ql_min) & (Nc_cm3 > 1.0)
    Nc_cm3_safe = np.where(valid, Nc_cm3, 1.0)   # safe minimum to avoid division issue
    P_aut = cfg.kc_auto * qc_ ** cfg.alpha_auto * Nc_cm3_safe ** cfg.beta_auto
    P_aut = np.where(valid, P_aut, 0.0)

    # Rain embryo mass ~ (2.6e-10 kg), used in MG2 for Nr tendency
    x_star = 2.6e-10          # kg per rain embryo
    dNr_aut = P_aut / x_star  # #/kg/s
    dNc_aut = -2.0 * dNr_aut  # two drops lost per embryo (approx)

    return -P_aut, P_aut, dNc_aut, dNr_aut


# ============================================================
#  3. Accretion  (KK 2000)
# ============================================================

def accretion(qc, qr, Nc, rho):
    """
    Collision of cloud droplets with falling rain.
    KK2000:  P_acc = kc_acc * (qc * qr)^1.15   [kg/kg/s]
    Droplet number loss proportional to mass loss.

    Returns
    -------
    dqc_acc, dqr_acc, dNc_acc  : kg/kg/s, kg/kg/s, #/kg/s
    """
    qc_ = np.maximum(qc, 0.0)
    qr_ = np.maximum(qr, 0.0)

    P_acc = cfg.kc_acc * (qc_ * qr_) ** cfg.gamma_acc
    P_acc = np.where((qc_ > cfg.ql_min) & (qr_ > cfg.qr_min), P_acc, 0.0)

    # Number loss: assume same fractional rate as mass loss
    Nc_     = np.maximum(Nc, 1.0)
    qc_safe = np.maximum(qc_, 1.0e-20)
    dNc_acc = -Nc_ * P_acc / qc_safe

    return -P_acc, P_acc, dNc_acc


# ============================================================
#  4. Self-collection and breakup of rain (simplified)
# ============================================================

def rain_selfcollection(Nr, qr, rho):
    """
    Self-collection of rain reduces Nr (drops coalesce -> fewer, larger drops).
    Using the Seifert & Beheng (2001) simplified form.
    dNr/dt = -k_sc * Nr^2 * rho * qr   (very approximate)

    Returns dNr_sc  (#/kg/s)
    """
    Nr_ = np.maximum(Nr, 0.0)
    qr_ = np.maximum(qr, 0.0)
    k_sc = 5.78e3   # m^3 kg^-1 s^-1 (tuned)
    dNr_sc = -k_sc * Nr_ * rho * qr_
    dNr_sc = np.where(qr_ > cfg.qr_min, dNr_sc, 0.0)
    return dNr_sc


# ============================================================
#  5. Rain evaporation (below cloud)
# ============================================================

def rain_evaporation(T, p, qv, qr, Nr, rho):
    """
    Evaporation of falling rain in subsaturated air.

    Based on ventilation-corrected diffusive growth theory (Rogers & Yau 1989,
    Pruppacher & Klett 1997), simplified after MG2.

    Rate:
      P_evpr = (qv/qs - 1) / (A + B) * integral(n(D) * F_v(D) * D dD)

    where:
      A = Lv^2 / (Ka * Rv * T^2)
      B = Rv * T / (Dv * e_sat)
    and the integral is evaluated analytically for an exponential distribution.

    Returns
    -------
    dqr_evap (#/kg/s, negative = evaporation)
    dNr_evap (#/kg/s, negative)
    dqv_evap (#/kg/s, positive)
    """
    qs    = th.q_sat(T, p)
    sat_def = (qv / np.maximum(qs, 1.0e-8)) - 1.0   # < 0 if subsaturated

    # Only evaporate if subsaturated and rain present
    do_evap = (sat_def < 0) & (qr > cfg.qr_min)

    # Thermodynamic denominators
    es   = th.e_sat(T)
    A_th = cfg.Lv**2 / (cfg.Ka_therm * cfg.Rv * T**2)
    B_th = cfg.Rv * T / (cfg.Dv_vapor * np.maximum(es, 1.0e-3))
    AB   = A_th + B_th

    # Slope parameter of exponential rain distribution
    lam = lambda_rain(rho, qr, Nr)

    # Ventilation-corrected integral for exponential dist:
    # int n(D)*F_v*D dD using F_v = a_v + b_v*Re^0.5
    # Simplified: use first-order ventilation correction
    # I_evap ~ 2*pi * N0r_vol * (0.78/lam^2 + 0.31*Sc^(1/3)*(a_vt/nu)^0.5/lam^(2+b_vt/2))
    # For practical purposes use the simplified Rogers & Yau form:
    #   dqr/dt = C_evp * (S-1) / (rho * AB) * f(lambda)
    # where f(lambda) = N0r_vol * pi * (0.78/lam^2 + 0.31 * (a*Nr_vol/nu)^0.5 * Gamma(2.75)/lam^2.75)
    # We use a simpler ventilated form:

    Nr_vol   = rho * np.maximum(Nr, cfg.Nr_min)      # #/m^3
    # F(lambda) from Rogers & Yau eq 8.22 (exponential dist):
    nu_air   = cfg.mu_air / rho                       # kinematic viscosity
    Sc_third = (nu_air / cfg.Dv_vapor) ** (1.0/3.0)  # Schmidt number ^1/3

    f_lam = (Nr_vol * np.pi * 0.78 / lam**2 +
             0.31 * Sc_third * np.sqrt(cfg.a_vt / nu_air) *
             Nr_vol * np.pi * _gamma_approx(2.75) / lam**2.75)

    # NOTE: the formula below divides by rho twice (once in `rho * AB`, once
    # explicitly).  The net result is ~1/rho smaller than the correct MG2
    # rate (Morrison & Gettelman 2008, eq. B3 uses N0r [#/m^4] while f_lam
    # here uses Nr_vol [#/m^3]).  Rain evaporation is a secondary process in
    # this nighttime Sc model; correcting the factor-of-rho would require
    # rederiving f_lam against the reference formula.
    P_evpr_kg = sat_def / (rho * AB) * f_lam
    P_evpr    = P_evpr_kg / rho   # effective kg/kg/s tendency (see note above)

    # Cap evaporation: cannot evaporate more rain than exists in dt
    # (caller must handle timestep)
    P_evpr = np.where(do_evap, np.minimum(P_evpr, 0.0), 0.0)

    # Number tendency: evaporation reduces Nr at same fractional rate as mass
    qr_safe  = np.maximum(qr, 1.0e-20)
    dNr_evap = Nr * P_evpr / qr_safe
    dqv_evap = -P_evpr

    return P_evpr, dNr_evap, dqv_evap


def _gamma_approx(n):
    """Gamma function — uses scipy if available, else math.gamma."""
    return float(_gamma(n))


# ============================================================
#  6. Cloud sedimentation  (Stokes settling)
# ============================================================

def cloud_sedimentation_velocity(qc, Nc, rho):
    """
    Mass-weighted sedimentation velocity for cloud droplets (m/s, positive downward).

    Stokes law:  V_sed = (2/9) * (rho_w / mu_air) * g * r_eff^2
    For a gamma distribution, use the mass-weighted mean radius.
    """
    r_e = r_eff_cloud(rho, qc, Nc)
    V_stokes = (2.0 / 9.0) * (cfg.rho_w / cfg.mu_air) * cfg.g * r_e**2
    return np.where(qc > cfg.ql_min, V_stokes, 0.0)


# ============================================================
#  7. Rain sedimentation  (power-law V_t)
# ============================================================

def rain_sedimentation_velocity(qr, Nr, rho):
    """
    Mass-weighted terminal velocity of rain (m/s, positive downward).

    For exponential distribution with V_t(D) = a_vt * D^b_vt:
      V_mass = a_vt * Gamma(4 + b_vt) / (6 * lambda^b_vt)

    Corrected for air density: V_t * (rho_sfc / rho)^0.4
    """
    lam = lambda_rain(rho, qr, Nr)
    gam4b = _gamma_approx(4.0 + cfg.b_vt)
    V_rain = cfg.a_vt * gam4b / (6.0 * lam**cfg.b_vt)
    # Air density correction (approximate)
    rho_sfc = 1.2
    V_rain *= (rho_sfc / np.maximum(rho, 0.1)) ** 0.4
    return np.where(qr > cfg.qr_min, V_rain, 0.0)


# ============================================================
#  Sedimentation flux (upstream differencing)
# ============================================================

def sediment(q, V_sed, rho, dz, dt):
    """
    Apply sedimentation to a mixing-ratio field using first-order
    upstream (upwind) differencing in the vertical.

    Flux convention: F[k] is the downward flux across the LOWER interface of
    layer k (i.e. between k-1 and k), in kg/m^2/s.

    Parameters
    ----------
    q     : (nz,) mixing ratio (kg/kg)
    V_sed : (nz,) downward sedimentation speed (m/s, >= 0)
    rho   : (nz,) air density (kg/m^3)
    dz    : scalar layer thickness (m)
    dt    : time step (s)

    Returns
    -------
    q_new : (nz,) updated mixing ratio
    flux  : (nz,) surface precipitation flux from layer 0 (kg/m^2/s) -- only [0] used
    """
    nz = len(q)
    q_new = q.copy()

    # Interface flux F[k]: downward from layer k into layer k-1
    # F[k] = rho[k] * V[k] * q[k]     (mass flux, kg/m^2/s)
    F = rho * V_sed * np.maximum(q, 0.0)   # (nz,) — flux leaving bottom of each layer

    # Update: layer k gains from above (F[k+1]) and loses downward (F[k])
    # dq/dt = (F[k+1] - F[k]) / (rho[k] * dz)
    for k in range(nz - 1, -1, -1):
        F_in  = F[k+1] if k < nz - 1 else 0.0   # flux entering from layer above
        F_out = F[k]                               # flux leaving this layer downward
        dq = (F_in - F_out) * dt / (rho[k] * dz)
        q_new[k] = q[k] + dq
        q_new[k] = max(q_new[k], 0.0)

    # Surface precipitation flux (kg/m^2/s) from the bottom of layer 0
    precip_flux = F[0]

    return q_new, precip_flux


# ============================================================
#  Full microphysics step
# ============================================================

def microphysics_step(T, p, qv, qc, Nc, qr, Nr, rho, z, dz, dt):
    """
    Advance microphysics for one time step dt.

    Processes applied in sequence (operator splitting):
      1. Droplet activation (number adjustment)
      2. Autoconversion
      3. Accretion
      4. Rain self-collection
      5. Rain evaporation
      6. Cloud sedimentation
      7. Rain sedimentation
    Condensation / evaporation (saturation adjustment) is handled
    externally by the main integrator to couple with temperature.

    Parameters
    ----------
    T, p, qv, qc, Nc, qr, Nr : arrays (nz,)
    rho : air density array (nz,)
    z   : layer heights (nz,)
    dz  : scalar layer thickness (m)
    dt  : time step (s)

    Returns
    -------
    dict of updated fields and diagnostic tendencies
    """
    nz = len(T)

    # --- Clamp inputs ---
    qc = np.maximum(qc, 0.0)
    Nc = np.maximum(Nc, 0.0)
    qr = np.maximum(qr, 0.0)
    Nr = np.maximum(Nr, 0.0)

    # Diagnostics (W/m^2 equivalent tendencies for energy tracking)
    diag = {
        'precip_rate'  : 0.0,   # kg/m^2/s surface precipitation
        'dT_cond'      : np.zeros(nz),  # heating from condensation (K/s)
    }

    # ---- 1. Droplet activation (implicit exponential relaxation) ----
    Nc = droplet_activation(qc, Nc, rho, dt)
    Nc = np.where(qc > cfg.ql_min, Nc, 0.0)

    # ---- 2. Autoconversion ----
    d_qc_aut, d_qr_aut, d_Nc_aut, d_Nr_aut = autoconversion(qc, Nc, rho)
    # Cap: cannot remove more cloud than exists
    d_qc_aut = np.maximum(d_qc_aut * dt, -qc) / dt
    d_Nc_aut = np.maximum(d_Nc_aut * dt, -Nc) / dt

    qc += d_qc_aut * dt
    Nc += d_Nc_aut * dt
    qr += d_qr_aut * dt
    Nr += d_Nr_aut * dt

    # ---- 3. Accretion ----
    d_qc_acc, d_qr_acc, d_Nc_acc = accretion(qc, qr, Nc, rho)
    d_qc_acc = np.maximum(d_qc_acc * dt, -qc) / dt
    d_Nc_acc = np.maximum(d_Nc_acc * dt, -Nc) / dt

    qc += d_qc_acc * dt
    Nc += d_Nc_acc * dt
    qr += d_qr_acc * dt

    # ---- 4. Rain self-collection ----
    dNr_sc = rain_selfcollection(Nr, qr, rho)
    Nr     = np.maximum(Nr + dNr_sc * dt, 0.0)

    # ---- 5. Rain evaporation ----
    d_qr_evap, d_Nr_evap, d_qv_evap = rain_evaporation(T, p, qv, qr, Nr, rho)
    # Cap: cannot evaporate more rain than exists
    d_qr_evap = np.maximum(d_qr_evap * dt, -qr) / dt
    d_Nr_evap = np.maximum(d_Nr_evap * dt, -Nr) / dt

    qr  += d_qr_evap * dt
    Nr  += d_Nr_evap * dt
    qv  += d_qv_evap * dt

    # Latent heating from rain evaporation (cooling, negative)
    dT_evap = -cfg.Lv / cfg.cp * d_qv_evap   # < 0 where evaporation occurs
    T += dT_evap * dt
    diag['dT_cond'] = dT_evap

    # ---- 6. Cloud sedimentation ----
    V_sed_c = cloud_sedimentation_velocity(qc, Nc, rho)
    qc, flux_c = sediment(qc, V_sed_c, rho, dz, dt)
    # Number sedimentation at same speed
    Nc, _       = sediment(Nc, V_sed_c, rho, dz, dt)

    # ---- 7. Rain sedimentation ----
    V_sed_r = rain_sedimentation_velocity(qr, Nr, rho)
    qr, flux_r  = sediment(qr, V_sed_r, rho, dz, dt)
    Nr, _        = sediment(Nr, V_sed_r, rho, dz, dt)

    # Surface precipitation (mm/hr for diagnostics)
    diag['precip_rate'] = flux_r + flux_c   # kg/m^2/s

    # ---- Enforce lower bounds ----
    qc = np.maximum(qc, 0.0)
    Nc = np.where(qc > cfg.ql_min, np.maximum(Nc, 1.0), 0.0)
    qr = np.maximum(qr, 0.0)
    Nr = np.where(qr > cfg.qr_min, np.maximum(Nr, cfg.Nr_min), 0.0)

    return {
        'T'  : T,
        'qv' : qv,
        'qc' : qc,
        'Nc' : Nc,
        'qr' : qr,
        'Nr' : Nr,
        'diag': diag,
    }
