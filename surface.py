"""
surface.py
Surface boundary conditions: SST and bulk aerodynamic fluxes.

Two SST modes:
  'fixed' : SST is constant (Dirichlet lower boundary for temperature).
  'slab'  : SST evolves via net energy exchange with the atmosphere.
    d(T_sst)/dt = (F_dn_lw_sfc - F_up_lw_sfc + SHF + LHF) / (rho_ocean * cp_ocean * d_slab)
    (Note: SW = 0 at night, so only LW + turbulent fluxes drive SST.)

Bulk aerodynamic fluxes (Andreas & Murphy 1986; Fairall et al. 2003):
  SHF = rho * Cd * Ch * |U| * cp * (T_sfc - T_atm_0)  [W/m^2, upward positive]
  LHF = rho * Cd * Cq * |U| * Lv * (q_sat(T_sst) - qv_0)  [W/m^2]

where subscript 0 refers to the lowest model layer (~25 m AGL).

For simplicity the transfer coefficients Cd = Ch = Cq = 1.1e-3
(appropriate for |U| ~ 7 m/s over the open ocean, neutral stability).

References:
  Fairall et al. (2003), J. Climate.
  Andreas & Murphy (1986), J. Phys. Oceanogr.
"""

import numpy as np
from config import cfg
import thermo as th


class SurfaceModel:
    """Manages SST and computes surface turbulent + radiative fluxes."""

    def __init__(self, T_sst_init=None, mode=None):
        self.mode  = mode  if mode  is not None else cfg.sst_mode
        self.T_sst = T_sst_init if T_sst_init is not None else cfg.sst_K

        # Diagnostic storage
        self.shf  = 0.0   # sensible heat flux (W/m^2, upward positive)
        self.lhf  = 0.0   # latent heat flux   (W/m^2, upward positive)
        self.evap = 0.0   # surface evaporation (kg/m^2/s)

    # ------------------------------------------------------------------
    #  Bulk surface fluxes
    # ------------------------------------------------------------------

    def compute_fluxes(self, T0, qv0, rho0, p0):
        """
        Compute surface sensible and latent heat fluxes.

        Parameters
        ----------
        T0, qv0 : temperature and water-vapour mixing ratio in lowest layer (K, kg/kg)
        rho0    : air density in lowest layer (kg/m^3)
        p0      : pressure in lowest layer (Pa)

        Returns
        -------
        shf  : sensible heat flux (W/m^2, upward positive)
        lhf  : latent heat flux   (W/m^2, upward positive)
        """
        U   = cfg.u_ref             # wind speed (m/s)
        qs_sst = th.q_sat(self.T_sst, cfg.p_surface)  # sat. specific humidity at SST

        # Sensible heat flux
        shf = rho0 * cfg.Ch * U * cfg.cp * (self.T_sst - T0)

        # Latent heat flux
        lhf = rho0 * cfg.Cq * U * cfg.Lv * (qs_sst - qv0)
        lhf = max(lhf, 0.0)   # ocean does not condense moisture

        self.shf  = shf
        self.lhf  = lhf
        self.evap = lhf / cfg.Lv   # kg/m^2/s

        return shf, lhf

    def surface_flux_theta(self, rho0):
        """SHF converted to kinematic theta flux (K m/s)."""
        return self.shf / (rho0 * cfg.cp)

    def surface_flux_qv(self, rho0):
        """LHF converted to kinematic qv flux (kg/kg m/s)."""
        return self.lhf / (rho0 * cfg.Lv)

    # ------------------------------------------------------------------
    #  SST update (slab ocean)
    # ------------------------------------------------------------------

    def update_sst(self, F_lw_net_sfc, dt, Q_ocean=0.0):
        """
        Update SST for the slab ocean mode.

        Energy budget:
          rho_o * cp_o * d_slab * dT_sst/dt
            = -F_lw_net_sfc  +  SHF  +  LHF  +  Q_ocean

        F_lw_net_sfc : net upward LW flux at surface (W/m^2)
                       positive = ocean is losing energy to atmosphere
        SHF, LHF are already stored from compute_fluxes().
        Q_ocean      : prescribed external heat flux into the ocean (W/m^2),
                       e.g. from ocean heat transport convergence or
                       a uniform surface forcing experiment.
                       Positive = ocean gains heat, SST warms.

        Convention: all terms positive when ocean GAINS energy.
        """
        if self.mode != 'slab':
            return   # SST fixed, nothing to do

        # Ocean gains energy from:
        #   - downwelling LW  = -F_lw_net_sfc (because F_net = up - dn)
        #   - sensible heat   : -SHF (negative = ocean cooled by SHF going up)
        #   - latent heat     : -LHF
        #   - external forcing: +Q_ocean
        net_flux_into_ocean = (-F_lw_net_sfc - self.shf - self.lhf + Q_ocean)

        C_slab = cfg.rho_ocean * cfg.cp_ocean * cfg.slab_depth
        dT     = net_flux_into_ocean * dt / C_slab
        self.T_sst += dT

    # ------------------------------------------------------------------
    #  Diagnostic
    # ------------------------------------------------------------------

    def Bowen_ratio(self):
        """Bowen ratio Bo = SHF / LHF."""
        if abs(self.lhf) < 1.0e-3:
            return float('inf')
        return self.shf / self.lhf

    def info(self):
        return (f"SST={self.T_sst:.2f} K  "
                f"SHF={self.shf:.1f} W/m2  "
                f"LHF={self.lhf:.1f} W/m2  "
                f"Bo={self.Bowen_ratio():.2f}")
