"""
scm.py
Main single-column model integrator for the nighttime stratocumulus SCM.

State vector per layer k (k = 0 at surface, k = nz-1 at top):
  T[k]   : temperature            (K)
  qv[k]  : water vapour           (kg/kg)
  qc[k]  : cloud liquid water     (kg/kg)
  Nc[k]  : cloud droplet number   (#/kg)
  qr[k]  : rain water             (kg/kg)
  Nr[k]  : raindrop number        (#/kg)

Pressure is diagnosed from hydrostatic balance.
Density is diagnosed from the ideal-gas law for moist air.

Time integration (operator splitting):
  1. Large-scale subsidence (upwind advection)
  2. Free-troposphere nudging (above inversion only)
  3. Surface fluxes (update boundary condition)
  4. Turbulent mixing  (implicit CN diffusion)
  5. Longwave radiation  (explicit heating tendency)
  6. Cloud microphysics  (explicit, sub-stepped)
  7. Saturation adjustment (condensation / evaporation)

Energy and moisture are tracked to diagnose conservation errors.

No horizontal divergence of heat or moisture is included.
"""

import numpy as np
from config import cfg
from grid import VerticalGrid
import thermo   as th
import microphysics as micro
import radiation    as rad
import turbulence   as turb
import dynamics     as dyn
from surface import SurfaceModel


class StratocumulusSCM:
    """Single-column model for the nighttime stratocumulus-topped boundary layer."""

    def __init__(self, sst_mode=None, sst_K=None, Q_ocean=0.0):
        self.grid    = VerticalGrid()
        self.surface = SurfaceModel(T_sst_init=sst_K, mode=sst_mode)
        self.Q_ocean = Q_ocean   # prescribed ocean heat flux (W/m^2, + = warms ocean)

        nz = self.grid.nz
        # --- State arrays ---
        self.T  = np.zeros(nz)
        self.qv = np.zeros(nz)
        self.qc = np.zeros(nz)
        self.Nc = np.zeros(nz)
        self.qr = np.zeros(nz)
        self.Nr = np.zeros(nz)

        # Pressure and density (diagnosed each step)
        self.p   = np.zeros(nz)
        self.rho = np.zeros(nz)

        # Reference sounding for FT nudging
        self._T_ref  = None
        self._qv_ref = None

        # Time
        self.time = 0.0

        # Output history
        self.history = []

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def initialize(self, sounding=None):
        """
        Set initial conditions.

        If sounding is None, use the DYCOMS-II RF01-like profile from config.
        Otherwise, sounding must be a dict with keys: 'T', 'qv', 'qc', 'Nc', 'qr', 'Nr'
        all of length nz.
        """
        nz = self.grid.nz
        z  = self.grid.z

        if sounding is not None:
            self.T  = np.array(sounding['T'])
            self.qv = np.array(sounding['qv'])
            self.qc = np.array(sounding.get('qc', np.zeros(nz)))
            self.Nc = np.array(sounding.get('Nc', np.zeros(nz)))
            self.qr = np.array(sounding.get('qr', np.zeros(nz)))
            self.Nr = np.array(sounding.get('Nr', np.zeros(nz)))
        else:
            self._init_dycoms()

        self._update_pressure_density()

        # Store reference sounding for FT nudging
        self._T_ref  = self.T.copy()
        self._qv_ref = self.qv.copy()

    def _init_dycoms(self):
        """
        DYCOMS-II RF01 / CGILS S12-like initial sounding.
        """
        z  = self.grid.z
        nz = self.grid.nz

        zi  = cfg.zi_init   # inversion height (m)
        p0  = cfg.p_surface

        for k in range(nz):
            zk = z[k]

            # --- Potential temperature ---
            if zk <= zi:
                theta_k = cfg.theta_BL
            else:
                theta_k = (cfg.theta_BL + cfg.theta_FT_jump
                           + cfg.gamma_FT * (zk - zi))

            # --- Total water mixing ratio ---
            if zk <= zi:
                qt_k = cfg.qt_BL
            else:
                qt_k = cfg.qt_FT

            # Rough pressure estimate (iterative)
            p_est = p0 * np.exp(-cfg.g * zk / (cfg.Rd * 290.0))
            T_k   = th.T_from_theta(theta_k, p_est)

            # Saturation adjustment: put cloud in the upper part of BL
            # LCL ~ z where T = T_dew
            T_new, qv_k, qc_k, _ = th.saturation_adjustment(T_k, qt_k, 0.0, p_est)
            T_k = T_new

            self.T[k]  = T_k
            self.qv[k] = qv_k
            self.qc[k] = qc_k
            self.Nc[k] = cfg.Nc_prescribed if qc_k > cfg.ql_min else 0.0

    def _update_pressure_density(self):
        """Recompute pressure and density from current state."""
        self.p   = self.grid.compute_pressure(self.T, self.qv)
        self.rho = th.air_density(self.T, self.p, self.qv, self.qc)

    # ------------------------------------------------------------------
    #  Single time step
    # ------------------------------------------------------------------

    def step(self, dt=None):
        """Advance the model by one time step dt (s)."""
        if dt is None:
            dt = cfg.dt

        z  = self.grid.z
        zi = self.grid.zi
        dz = self.grid.dz
        nz = self.grid.nz

        T, qv, qc, Nc, qr, Nr = (self.T, self.qv, self.qc,
                                  self.Nc, self.qr, self.Nr)
        rho = self.rho
        p   = self.p

        # -------- 1. Large-scale subsidence --------
        w_ls   = dyn.w_subsidence(z)
        theta  = th.theta_from_T(T, p)
        qt     = qv + qc    # total water conserved under subsidence

        dtheta_sub = dyn.subsidence_tendency(theta, w_ls, dz)
        dqt_sub    = dyn.subsidence_tendency(qt,    w_ls, dz)
        dqr_sub    = dyn.subsidence_tendency(qr,    w_ls, dz)

        theta += dtheta_sub * dt
        qt    += dqt_sub    * dt
        qr    += np.maximum(dqr_sub * dt, -qr)   # prevent negative qr

        # Partition qt back into qv + qc: keep existing cloud fraction
        # (saturation adjustment will fix this below)
        qv = np.where(qc > cfg.ql_min, qt - qc, qt)
        qc = np.maximum(qt - qv, 0.0)

        # Convert back to T
        T = th.T_from_theta(theta, p)

        # -------- 2. FT nudging above inversion --------
        zi_inv, _ = dyn.find_inversion_height(theta, z)
        if self._T_ref is not None:
            dT_nudge  = dyn.ft_relaxation(T,  self._T_ref,  z, zi_inv)
            dqv_nudge = dyn.ft_relaxation(qv, self._qv_ref, z, zi_inv)
            T  += dT_nudge  * dt
            qv += dqv_nudge * dt

        self._update_pressure_density()
        rho = self.rho
        p   = self.p

        # -------- 3. Surface fluxes --------
        shf, lhf = self.surface.compute_fluxes(T[0], qv[0], rho[0], p[0])

        # Surface fluxes as kinematic (per unit mass) for diffusion BCs
        flux_theta_sfc = self.surface.surface_flux_theta(rho[0])
        flux_qv_sfc    = self.surface.surface_flux_qv(rho[0])

        # -------- 4. Turbulent mixing (implicit) --------
        _, k_ct = self.grid.cloud_base_top(qc)   # cloud top index
        z_ct    = z[k_ct] if k_ct is not None else zi_inv

        K_h, K_m = turb.compute_K_profile(T, qv, qc, rho, z, zi, shf, lhf, z_ct)

        # Diffuse conserved thermodynamic variables: theta_l and q_t
        # theta_l = theta - (Lv/cp) * qc / exner(p)  (conserved through condensation)
        # q_t     = qv + qc                           (conserved through phase changes)
        exner_p  = th.exner(p)
        theta_l  = th.theta_from_T(T, p) - cfg.Lv * qc / (cfg.cp * exner_p)
        qt       = qv + qc

        thl_new = turb.implicit_diffuse(theta_l, K_h, rho, dz, dt,
                                        flux_bottom=flux_theta_sfc)
        qt_new  = turb.implicit_diffuse(qt,      K_h, rho, dz, dt,
                                        flux_bottom=flux_qv_sfc)
        qt_new  = np.maximum(qt_new, 0.0)

        # Recover T from theta_l (first estimate: assume qc unchanged, sat-adj follows)
        T  = thl_new * exner_p + cfg.Lv * qc / cfg.cp
        qv = np.minimum(qt_new, th.q_sat(T, p))   # first guess
        qc = np.maximum(qt_new - qv, 0.0)
        Nc = np.where(qc > cfg.ql_min, Nc, 0.0)

        # -------- 5. Longwave radiation --------
        _, _, Q_rad, F_net_sfc = rad.lw_two_stream(
            T, p, qv, qc, qr, rho, dz, self.surface.T_sst
        )
        T += Q_rad * dt

        # Update SST for slab mode
        self.surface.update_sst(F_net_sfc, dt, self.Q_ocean)

        # -------- 6. Saturation adjustment --------
        T, qv, qc, _ = th.saturation_adjustment(T, qv, qc, p)
        Nc = np.where(qc > cfg.ql_min, Nc, 0.0)

        # -------- 7. Cloud microphysics  (sub-stepped) --------
        self._update_pressure_density()
        rho = self.rho
        p   = self.p

        result = {'diag': {'precip_rate': 0.0}}   # default if n_micro_substep == 0
        dt_micro = dt / max(cfg.n_micro_substep, 1)
        for _ in range(cfg.n_micro_substep):
            result = micro.microphysics_step(T, p, qv, qc, Nc, qr, Nr,
                                             rho, z, dz, dt_micro)
            T  = result['T']
            qv = result['qv']
            qc = result['qc']
            Nc = result['Nc']
            qr = result['qr']
            Nr = result['Nr']

            # Saturation adjustment after each micro substep
            T, qv, qc, _ = th.saturation_adjustment(T, qv, qc, p)

        # -------- Store state --------
        self.T, self.qv, self.qc = T, qv, qc
        self.Nc, self.qr, self.Nr = Nc, qr, Nr

        self._update_pressure_density()
        self.time += dt

        return result['diag']

    # ------------------------------------------------------------------
    #  Run for n_steps or until t_end
    # ------------------------------------------------------------------

    def run(self, t_end=None, output_interval=None, verbose=True):
        """
        Integrate the model forward from current time to t_end.

        Parameters
        ----------
        t_end           : end time (s), default cfg.t_end
        output_interval : seconds between saved snapshots, default cfg.output_interval
        verbose         : print progress to stdout
        """
        if t_end is None:
            t_end = cfg.t_end
        if output_interval is None:
            output_interval = cfg.output_interval

        dt = cfg.dt
        t_next_output = self.time

        while self.time < t_end - 0.5 * dt:
            diag = self.step(dt)

            if self.time >= t_next_output - 0.5 * dt:
                snap = self._snapshot(diag)
                self.history.append(snap)
                t_next_output += output_interval

                if verbose:
                    zi_inv, _ = dyn.find_inversion_height(
                        th.theta_from_T(self.T, self.p), self.grid.z)
                    LWP = rad.liquid_water_path(self.qc, self.rho, self.grid.dz)
                    print(f"t={self.time/3600:.2f} h  "
                          f"zi={zi_inv:.0f} m  "
                          f"LWP={LWP*1000:.1f} g/m2  "
                          f"{self.surface.info()}  "
                          f"precip={diag['precip_rate']*3600:.3f} mm/hr")

        return self.history

    # ------------------------------------------------------------------
    #  Diagnostics / snapshots
    # ------------------------------------------------------------------

    def _snapshot(self, diag=None):
        """Save a copy of the current state and key diagnostics."""
        z   = self.grid.z
        rho = self.rho
        dz  = self.grid.dz

        theta  = th.theta_from_T(self.T, self.p)
        theta_v = th.T_virtual(self.T, self.qv, self.qc) / th.exner(self.p)

        zi_inv, k_inv = dyn.find_inversion_height(theta, z)
        k_base, k_top = self.grid.cloud_base_top(self.qc)
        LWP           = rad.liquid_water_path(self.qc, rho, dz)
        tau_vis       = rad.cloud_optical_depth(self.qc, self.qr, rho, dz)

        snap = {
            'time'       : self.time,
            'z'          : z.copy(),
            'T'          : self.T.copy(),
            'theta'      : theta.copy(),
            'theta_v'    : theta_v.copy(),
            'qv'         : self.qv.copy(),
            'qc'         : self.qc.copy(),
            'Nc'         : self.Nc.copy(),
            'qr'         : self.qr.copy(),
            'Nr'         : self.Nr.copy(),
            'p'          : self.p.copy(),
            'rho'        : rho.copy(),
            'T_sst'      : self.surface.T_sst,
            'shf'        : self.surface.shf,
            'lhf'        : self.surface.lhf,
            'LWP'        : LWP,
            'tau_vis'    : tau_vis,
            'zi'         : zi_inv,
            'z_cbase'    : z[k_base] if k_base is not None else np.nan,
            'z_ctop'     : z[k_top]  if k_top  is not None else np.nan,
            'precip_rate': diag['precip_rate'] if diag else 0.0,
        }
        return snap

    def current_state_summary(self):
        """Print a summary of the current column state."""
        snap = self._snapshot()
        print(f"\n--- SCM state at t = {self.time/3600:.2f} h ---")
        print(f"  SST             = {snap['T_sst']:.2f} K")
        print(f"  Inversion ht    = {snap['zi']:.0f} m")
        print(f"  Cloud base      = {snap['z_cbase']:.0f} m")
        print(f"  Cloud top       = {snap['z_ctop']:.0f} m")
        print(f"  LWP             = {snap['LWP']*1000:.1f} g/m2")
        print(f"  Cloud opt depth = {snap['tau_vis']:.1f}")
        print(f"  SHF             = {snap['shf']:.1f} W/m2")
        print(f"  LHF             = {snap['lhf']:.1f} W/m2")
        print(f"  Precip rate     = {snap['precip_rate']*3600:.3f} mm/hr")
