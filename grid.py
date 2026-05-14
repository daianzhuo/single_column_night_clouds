"""
grid.py
Vertical grid definition for the single-column model.

Coordinate convention:
  - Layer centres (thermodynamic levels): z[k], k = 0 (surface) .. nz-1 (top)
  - Layer interfaces (flux levels):       zi[k], k = 0 (surface) .. nz
  - z[k]  = (k + 0.5) * dz
  - zi[k] = k * dz
  - Lowest layer centre at z[0] = dz/2 above the surface

Pressure is computed from hydrostatic balance given a temperature profile,
so it is updated each time step by the main integrator.
"""

import numpy as np
from config import cfg


class VerticalGrid:
    """Holds fixed geometry and provides index utilities."""

    def __init__(self):
        self.nz  = cfg.nz
        self.dz  = cfg.dz
        self.nzi = cfg.nz + 1   # number of interface levels

        # Layer centres
        self.z  = (np.arange(self.nz)  + 0.5) * self.dz   # (nz,)
        # Layer interfaces (including surface z=0 and model top)
        self.zi = np.arange(self.nzi)  * self.dz            # (nz+1,)

        # Precomputed reciprocal for flux convergence
        self.inv_dz = 1.0 / self.dz

    # ------------------------------------------------------------------
    # Pressure utilities (call after updating the state)
    # ------------------------------------------------------------------
    def compute_pressure(self, T, qv):
        """
        Hydrostatic pressure at layer centres (Pa).
        Integrates upward from surface using the virtual temperature.
        T, qv : arrays of shape (nz,)
        Returns p : array of shape (nz,)
        """
        p = np.empty(self.nz)
        p[0] = cfg.p_surface * np.exp(
            -cfg.g * self.z[0] / (cfg.Rd * T[0] * (1.0 + (cfg.Rv/cfg.Rd - 1.0)*qv[0]))
        )
        for k in range(1, self.nz):
            Tv_mean = 0.5 * (T[k-1] * (1.0 + (cfg.Rv/cfg.Rd - 1.0)*qv[k-1]) +
                             T[k]   * (1.0 + (cfg.Rv/cfg.Rd - 1.0)*qv[k]))
            p[k] = p[k-1] * np.exp(-cfg.g * self.dz / (cfg.Rd * Tv_mean))
        return p

    def compute_pressure_interfaces(self, T, qv):
        """
        Hydrostatic pressure at layer interfaces (Pa).
        Surface interface: p_surface.
        """
        pi = np.empty(self.nzi)
        pi[0] = cfg.p_surface
        p_centres = self.compute_pressure(T, qv)
        for k in range(1, self.nzi - 1):
            pi[k] = 0.5 * (p_centres[k-1] + p_centres[k])
        pi[-1] = 2.0 * p_centres[-1] - pi[-2]
        return pi

    # ------------------------------------------------------------------
    # Layer index utilities
    # ------------------------------------------------------------------
    def layer_of(self, z_query):
        """Return the layer index containing height z_query (m)."""
        idx = int(z_query / self.dz)
        return min(max(idx, 0), self.nz - 1)

    def cloud_base_top(self, ql):
        """
        Return (k_base, k_top) indices where ql > ql_min.
        Returns (None, None) if no cloud.
        """
        cloudy = np.where(ql > cfg.ql_min)[0]
        if len(cloudy) == 0:
            return None, None
        return int(cloudy[0]), int(cloudy[-1])
