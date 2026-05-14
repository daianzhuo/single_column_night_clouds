"""
dynamics.py
Large-scale dynamical forcing for the stratocumulus SCM.

The only dynamical process is large-scale subsidence.
No horizontal heat or moisture divergence is imposed.

Subsidence profile:
  w_ls(z) = -D * z      [m/s, negative = downward]

where D is the large-scale divergence [s^-1], typically 3-7e-6 s^-1
for subtropical marine stratocumulus (e.g. CGILS, DYCOMS-II).

The subsidence tendencies are:
  d(phi)/dt|_sub = -w_ls * d(phi)/dz

Applied to:  theta (or T), qt (= qv + qc), and passive tracers.

Note: In a column model with no horizontal divergence, conservation of
dry-air mass requires a compensating vertical mass flux. The subsidence
is applied as a large-scale forcing (relaxation on the free troposphere
profile is optional).

Reference:
  Bretherton et al. (2004), MWR: CGILS idealized case.
  Stevens et al. (2003), MWR: DYCOMS-II.
"""

import numpy as np
from config import cfg


# ============================================================
#  Subsidence velocity profile
# ============================================================

def w_subsidence(z, divergence=None):
    """
    Large-scale subsidence velocity (m/s) at height z.
    w_ls(z) = -D * z   (linear in z, from mass continuity with D = const divergence)

    Parameters
    ----------
    z          : scalar or array of heights (m)
    divergence : horizontal divergence D (s^-1), defaults to cfg.divergence

    Returns
    -------
    w_ls : (same shape as z) subsidence velocity (m/s, negative = downward)
    """
    if divergence is None:
        divergence = cfg.divergence
    return -divergence * np.asarray(z, dtype=float)


# ============================================================
#  Subsidence tendency  (upwind differencing)
# ============================================================

def subsidence_tendency(phi, w_ls, dz):
    """
    Compute the advective tendency due to large-scale subsidence:
      dphi/dt|_sub = -w_ls * dphi/dz

    Uses first-order upwind differencing:
      If w_ls < 0  (downward): dphi/dz ~ (phi[k+1] - phi[k]) / dz
      If w_ls > 0  (upward):   dphi/dz ~ (phi[k] - phi[k-1]) / dz

    Parameters
    ----------
    phi   : (nz,) field (theta, qt, qv, etc.)
    w_ls  : (nz,) subsidence velocity at layer centres (m/s)
    dz    : scalar layer thickness (m)

    Returns
    -------
    dphi_dt : (nz,) subsidence tendency (same units as phi per second)
    """
    nz = len(phi)
    dphi_dt = np.zeros(nz)

    for k in range(nz):
        if w_ls[k] < 0:
            # Downward: use information from above (upwind)
            if k < nz - 1:
                dphi_dz = (phi[k+1] - phi[k]) / dz
            else:
                dphi_dz = 0.0   # top boundary: zero gradient
        else:
            # Upward: use information from below (upwind)
            if k > 0:
                dphi_dz = (phi[k] - phi[k-1]) / dz
            else:
                dphi_dz = 0.0   # bottom boundary: zero gradient

        dphi_dt[k] = -w_ls[k] * dphi_dz

    return dphi_dt


# ============================================================
#  Free-troposphere relaxation  (optional nudging)
# ============================================================

def ft_relaxation(phi, phi_ref, z, zi_inv, tau_relax=3600.0):
    """
    Relax the free-troposphere profile back to a reference sounding
    on a timescale tau_relax (s) to prevent runaway warming above the BL.

    This represents the effect of large-scale dynamics maintaining the
    FT stratification that drives subsidence (but no horizontal divergence
    of heat — only a nudging term above the inversion).

    Parameters
    ----------
    phi       : (nz,) current profile
    phi_ref   : (nz,) reference profile (initial sounding)
    z         : (nz,) layer heights
    zi_inv    : inversion height (m) — only relax above this
    tau_relax : relaxation timescale (s), default 1 hour

    Returns
    -------
    dphi_dt_relax : (nz,) relaxation tendency (same units/s)
    """
    above_inv = z > zi_inv
    dphi_dt   = np.zeros_like(phi)
    dphi_dt[above_inv] = -(phi[above_inv] - phi_ref[above_inv]) / tau_relax
    return dphi_dt


# ============================================================
#  Inversion height tracker
# ============================================================

def find_inversion_height(theta, z):
    """
    Estimate the boundary layer top / inversion height (m) as the level of
    maximum d(theta)/dz (sharpest potential-temperature gradient).

    Parameters
    ----------
    theta : (nz,) potential temperature profile (K)
    z     : (nz,) layer heights (m)

    Returns
    -------
    zi : inversion height (m)
    k_inv : layer index just below the inversion
    """
    nz = len(theta)
    if nz < 2:
        return z[0], 0

    dtheta_dz = np.diff(theta) / np.diff(z)   # (nz-1,)
    k_inv     = int(np.argmax(dtheta_dz))
    zi        = 0.5 * (z[k_inv] + z[k_inv+1])
    return zi, k_inv
