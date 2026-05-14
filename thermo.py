"""
thermo.py
Thermodynamic utility functions used throughout the SCM.

All inputs/outputs use SI units (K, Pa, kg/kg) unless noted.

References:
  - Bolton (1980): Saturation vapour pressure formula
  - Emanuel (1994): Atmospheric Convection
"""

import numpy as np
from config import cfg


# ============================================================
#  Saturation vapour pressure / mixing ratio
# ============================================================

def e_sat(T):
    """
    Saturation vapour pressure over liquid water (Pa).
    Bolton (1980) formula, valid ~200-330 K.
    """
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def de_sat_dT(T):
    """d(e_sat)/dT  (Pa/K)."""
    Tc = T - 273.15
    return e_sat(T) * 17.67 * (273.15 - 29.65) / (T - 29.65)**2


def q_sat(T, p):
    """
    Saturation specific humidity (kg/kg).
    q_sat = eps * e_sat / (p - (1-eps)*e_sat)
    """
    es = e_sat(T)
    return cfg.eps * es / (p - (1.0 - cfg.eps) * es)


def dq_sat_dT(T, p):
    """d(q_sat)/dT  (kg/kg/K)."""
    es  = e_sat(T)
    des = de_sat_dT(T)
    denom = p - (1.0 - cfg.eps) * es
    return cfg.eps * p * des / denom**2


# ============================================================
#  Potential temperature / Exner
# ============================================================

def exner(p):
    """Exner function pi = (p/p0)^kappa."""
    return (p / cfg.p0) ** cfg.kappa


def theta_from_T(T, p):
    """Potential temperature (K)."""
    return T / exner(p)


def T_from_theta(theta, p):
    """Temperature from potential temperature (K)."""
    return theta * exner(p)


def theta_l(T, p, ql):
    """
    Liquid-water potential temperature (K).
    theta_l = theta - (Lv/cp) * ql / exner(p)
    Conserved during adiabatic processes.
    """
    return theta_from_T(T, p) - cfg.Lv * ql / (cfg.cp * exner(p))


def T_from_theta_l(thl, p, ql):
    """Recover T from theta_l and ql (first-order)."""
    return thl * exner(p) + cfg.Lv * ql / cfg.cp


# ============================================================
#  Density / virtual temperature
# ============================================================

def T_virtual(T, qv, ql=None):
    """
    Virtual temperature (K).
    If ql provided, includes liquid-loading correction:
      Tv = T * (1 + qv/eps) / (1 + qv + ql)
    """
    if ql is None:
        ql = 0.0
    return T * (1.0 + qv / cfg.eps) / (1.0 + qv + ql)


def air_density(T, p, qv, ql=None):
    """Moist air density (kg/m^3), optionally including liquid loading."""
    Tv = T_virtual(T, qv, ql)
    return p / (cfg.Rd * Tv)


# ============================================================
#  Buoyancy
# ============================================================

def buoyancy(thl, qt, thl_ref, qt_ref, p, T_ref=None):
    """
    Buoyancy (m/s^2) of a parcel relative to environment.
    Linearised: b = g * (Tv_parcel - Tv_env) / Tv_env
    Uses liquid-water virtual potential temperature:
      theta_v ≈ theta * (1 + 0.61*qv - ql)
    """
    # Approximate: linearised liquid-water virtual potential temperature
    theta_v_parcel = thl * (1.0 + 0.61 * qt)
    theta_v_env    = thl_ref * (1.0 + 0.61 * qt_ref)
    return cfg.g * (theta_v_parcel - theta_v_env) / theta_v_env


# ============================================================
#  Saturation adjustment (condensation / evaporation)
# ============================================================

def saturation_adjustment(T, qv, ql, p, n_iter=10):
    """
    All-or-nothing saturation adjustment.

    Given total water qt = qv + ql:
      - If supersaturated: condense until T, qv, ql satisfy qs(T,p).
      - If subsaturated:   evaporate cloud until ql = 0 or saturation.

    Uses Newton-Raphson iteration.

    Parameters
    ----------
    T, qv, ql : scalar or array (K, kg/kg, kg/kg)
    p         : pressure (Pa)

    Returns
    -------
    T_new, qv_new, ql_new, delta_ql  (same shapes as input)
    delta_ql > 0 means condensation, < 0 means evaporation.
    """
    scalar_input = np.ndim(T) == 0
    T  = np.atleast_1d(np.array(T,  dtype=float))
    qv = np.atleast_1d(np.array(qv, dtype=float))
    ql = np.atleast_1d(np.array(ql, dtype=float))
    p  = np.atleast_1d(np.array(p,  dtype=float))

    qt = qv + ql
    T_new  = T.copy()
    ql_new = ql.copy()

    for _ in range(n_iter):
        qs    = q_sat(T_new, p)
        dqs   = dq_sat_dT(T_new, p)
        f     = qt - ql_new - qs           # residual: want f = 0

        # Layers that are supersaturated or still have cloud
        needs = (f > 0) | (ql_new > 0)
        if not np.any(needs):
            break

        # Newton step: d(ql) ~ f / (1 + Lv/cp * dqs/dT)
        denom     = 1.0 + cfg.Lv / cfg.cp * dqs
        d_ql      = np.where(needs, f / denom, 0.0)

        # Prevent ql from going negative
        d_ql      = np.maximum(d_ql, -ql_new)

        ql_new   += d_ql
        T_new    += cfg.Lv / cfg.cp * d_ql

    # Where subsaturated and no cloud, ensure ql = 0
    ql_new = np.where((qt < q_sat(T_new, p)) & (ql_new < 0), 0.0, ql_new)
    ql_new = np.maximum(ql_new, 0.0)
    qv_new = qt - ql_new
    delta_ql = ql_new - ql

    if scalar_input:
        return float(T_new[0]), float(qv_new[0]), float(ql_new[0]), float(delta_ql[0])
    return T_new, qv_new, ql_new, delta_ql


# ============================================================
#  Moist static energy and conserved variables
# ============================================================

def moist_static_energy(T, qv, z):
    """
    Moist static energy h = cp*T + g*z + Lv*qv  (J/kg).
    Approximately conserved in moist adiabatic ascent.
    """
    return cfg.cp * T + cfg.g * z + cfg.Lv * qv


def liquid_water_static_energy(T, ql, z):
    """
    Liquid-water static energy sl = cp*T + g*z - Lv*ql  (J/kg).
    Conserved through condensation/evaporation.
    """
    return cfg.cp * T + cfg.g * z - cfg.Lv * ql


# ============================================================
#  Lifting condensation level (LCL) estimate
# ============================================================

def LCL_height(T_sfc, q_sfc, p_sfc):
    """
    Approximate LCL height (m) using Bolton (1980).
    Used only for diagnostics.
    """
    # Bolton (1980) eq 21-22: T_LCL from RH = q_sfc/q_sat
    T_LCL = 1.0 / (1.0/(T_sfc - 55.0) - np.log(q_sfc / q_sat(T_sfc, p_sfc)) / 2840.0) + 55.0
    return (T_sfc - T_LCL) / 9.8e-3  # dry adiabatic lapse rate
