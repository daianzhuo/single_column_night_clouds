"""
turbulence.py
Turbulent mixing parameterisation for the stratocumulus SCM.

Scheme: K-diffusion with TKE-based eddy diffusivity.

The boundary layer in a nocturnal stratocumulus deck is driven by:
  1. Longwave cooling at cloud top  (dominant in nighttime Sc)
  2. Sensible heat flux from the ocean surface  (secondary)
  3. Wind shear at the surface (via u_ref, not explicitly resolved here)

Eddy diffusivities K_h (heat/scalars) and K_m (momentum) are
computed from local TKE production/dissipation following
a simplified Mellor-Yamada level-2.5 approach, or alternatively
the simpler Lenderink & Holtslag (2004) approach for SCM use.

For this implementation we use a diagnostic K-profile scheme
after O'Brien (1970) in the mixed layer and a buoyancy-based
TKE closure above the LCL (in cloud).

No horizontal diffusion: all mixing is strictly vertical.

References:
  Mellor & Yamada (1982), Rev. Geophys. Space Phys.
  Lenderink & Holtslag (2004), J. Atmos. Sci.
  Stevens et al. (2003), MWR: DYCOMS-II RF01.
  Siebesma et al. (2007), J. Atmos. Sci.
"""

import numpy as np
from config import cfg
import thermo as th


# ============================================================
#  Eddy diffusivity profile
# ============================================================

def compute_K_profile(T, qv, qc, rho, z, zi, shf, lhf, zi_cloud_top):
    """
    Compute eddy diffusivity K (m^2/s) at layer interfaces.

    Strategy:
      - Within the boundary layer (z < zi_cloud_top + margin):
          Use a TKE-based K that scales with buoyancy flux.
      - Above the BL: K = K_min (free-troposphere, stable).

    The buoyancy flux is estimated from:
      w'b' = g/T * (w'T' + (Lv/cp)*w'qv')
    Driven by cloud-top radiative cooling and surface fluxes.

    Parameters
    ----------
    T, qv, qc : (nz,) state arrays
    rho        : (nz,) density
    z          : (nz,) layer centre heights
    zi         : (nz+1,) interface heights
    shf        : surface sensible heat flux (W/m^2, upward positive)
    lhf        : surface latent heat flux (W/m^2, upward positive)
    zi_cloud_top : height of cloud top (m), None if no cloud

    Returns
    -------
    K_h : (nz+1,) eddy diffusivity for heat/scalars (m^2/s) at interfaces
    K_m : (nz+1,) eddy diffusivity for momentum (m^2/s)
    """
    nzi = len(zi)
    K_h = np.full(nzi, cfg.K_min)
    K_m = np.full(nzi, cfg.K_min)

    # BL height = cloud top height: O'Brien profile naturally goes to zero at h_bl,
    # preventing K from diffusing across the sharp inversion above cloud top.
    if zi_cloud_top is None:
        zi_cloud_top = 800.0
    h_bl = zi_cloud_top   # K -> 0 exactly at h_bl

    # Mean BL properties
    mask_bl = z < h_bl
    T_mean_bl = np.mean(T[mask_bl]) if np.any(mask_bl) else T[0]
    rho_bl    = np.mean(rho[mask_bl]) if np.any(mask_bl) else rho[0]

    # Surface buoyancy flux
    flux_buoy_surf = max((cfg.g / T_mean_bl) * shf / (rho_bl * cfg.cp), 0.0)
    flux_buoy_lhf  = max((cfg.g / T_mean_bl) * 0.61 * lhf / (rho_bl * cfg.Lv), 0.0)
    # Cloud-top LW cooling: nighttime Sc typically ~70 W/m^2 flux divergence
    flux_buoy_ctop = (cfg.g / T_mean_bl) * 70.0 / (rho_bl * cfg.cp)

    flux_buoy_total = max(flux_buoy_surf + flux_buoy_lhf + flux_buoy_ctop, 1e-6)
    w_star = (flux_buoy_total * h_bl) ** (1.0 / 3.0)
    kv = 0.4

    # --- O'Brien K-profile: zero at and above h_bl ---
    for j in range(nzi):
        z_ifc = zi[j]
        if z_ifc <= 0.0 or z_ifc >= h_bl:
            K_h[j] = cfg.K_min
            K_m[j] = cfg.K_min
            continue
        zeta   = z_ifc / h_bl
        K_val  = kv * w_star * z_ifc * (1.0 - zeta)**2
        K_h[j] = max(K_val, cfg.K_min)
        K_m[j] = max(K_val / cfg.Pr_t, cfg.K_min)

    K_h = np.clip(K_h, cfg.K_min, cfg.K_max)
    K_m = np.clip(K_m, cfg.K_min, cfg.K_max)
    K_h[0] = cfg.K_min
    K_h[-1] = cfg.K_min
    K_m[0] = cfg.K_min
    K_m[-1] = cfg.K_min

    return K_h, K_m


# ============================================================
#  Implicit diffusion solver (Thomas algorithm / tridiagonal)
# ============================================================

def implicit_diffuse(phi, K, rho, dz, dt, flux_bottom=0.0, flux_top=0.0):
    """
    Solve the fully-implicit vertical diffusion equation:
      rho[k] * (phi_new[k] - phi[k]) / dt
        = [ K[k+1]*(phi_new[k+1] - phi_new[k])
            - K[k] *(phi_new[k]   - phi_new[k-1]) ] / dz^2

    Boundary conditions (Neumann, flux-specified):
      - Bottom (k=0) : upward kinematic flux = flux_bottom  (K * dphi/dz at z=0)
      - Top   (k=nz-1): zero flux  (K * dphi/dz = 0 at z_top)

    Fully implicit is unconditionally stable; uses Thomas algorithm.

    Parameters
    ----------
    phi          : (nz,) field to diffuse (theta, qv, qc, ...)
    K            : (nz+1,) eddy diffusivity at interfaces (m^2/s)
    rho          : (nz,) air density (kg/m^3)
    dz           : uniform layer thickness (m)
    dt           : time step (s)
    flux_bottom  : kinematic surface flux (phi-units * m/s), upward positive
                   e.g., shf/(rho*cp) for theta; evap for qv
    flux_top     : kinematic flux at model top (usually 0)

    Returns
    -------
    phi_new : (nz,) updated field
    """
    nz  = len(phi)
    dz2 = dz * dz

    # Tridiagonal coefficients  a*phi[k-1] + b*phi[k] + c*phi[k+1] = d
    a = np.zeros(nz)   # subdiagonal  (coefficient of phi[k-1])
    b = np.zeros(nz)   # main diagonal
    c = np.zeros(nz)   # superdiagonal (coefficient of phi[k+1])
    d = np.zeros(nz)   # right-hand side

    for k in range(nz):
        Klo = K[k]      # at lower interface of layer k  (between k-1 and k)
        Khi = K[k+1]    # at upper interface of layer k  (between k   and k+1)
        alpha = dt / (rho[k] * dz2)

        if k == 0:
            # Lower boundary: prescribed flux via ghost cell
            # K[0]*(phi[-1] - phi[0])/dz = -flux_bottom  (ghost cell below)
            # Eliminates a[0]:
            a[k] = 0.0
            b[k] = 1.0 + alpha * Khi            # only upper coupling
            c[k] = -alpha * Khi
            d[k] = phi[k] + dt / (rho[k] * dz) * flux_bottom
        elif k == nz - 1:
            # Upper boundary: zero flux (ghost cell = phi[nz-1])
            a[k] = -alpha * Klo
            b[k] = 1.0 + alpha * Klo            # only lower coupling
            c[k] = 0.0
            d[k] = phi[k] + dt / (rho[k] * dz) * flux_top
        else:
            a[k] = -alpha * Klo
            c[k] = -alpha * Khi
            b[k] = 1.0 - a[k] - c[k]
            d[k] = phi[k]

    phi_new = _tridiag_solve(a, b, c, d)
    return phi_new


def _tridiag_solve(a, b, c, d):
    """
    Solve tridiagonal system using the Thomas algorithm.
    a: subdiagonal, b: main diagonal, c: superdiagonal, d: RHS
    """
    n   = len(d)
    c_  = np.zeros(n)
    d_  = np.zeros(n)
    x   = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for k in range(1, n):
        denom = b[k] - a[k] * c_[k-1]
        if abs(denom) < 1.0e-30:
            denom = 1.0e-30
        c_[k] = c[k] / denom
        d_[k] = (d[k] - a[k] * d_[k-1]) / denom

    x[-1] = d_[-1]
    for k in range(n-2, -1, -1):
        x[k] = d_[k] - c_[k] * x[k+1]

    return x


