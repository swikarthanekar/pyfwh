# Farassat Formulation 1A:
#
#   4*pi*p'(x,t) = p'_T + p'_L
#
# Thickness noise p'_T comes from the fluid displaced by a moving surface.
# Loading noise p'_L comes from unsteady pressure forces on the surface.
#
# For a stationary surface Mr=0 and the integrands simplify considerably.
# We use the d/dt-outside form: differentiate the summed integral over all
# panels with respect to observer time, rather than differentiating each
# panel's integrand pointwise. More stable at typical CFD panel counts.
#
# ref: Farassat (2007), NASA/TM-2007-214853

import numpy as np
from typing import Optional
from .surface       import FWHSurface
from .observer      import Observer
from .retarded_time import RetardedTimeSolver


class FWHSolver:
    """
    Geometry-agnostic FWH solver. Hand it any FWHSurface and it runs.

    c0   : speed of sound [m/s]
    rho0 : ambient density [kg/m^3]
    """

    def __init__(self, c0=340.0, rho0=1.225):
        if c0 <= 0 or rho0 <= 0:
            raise ValueError("c0 and rho0 must be positive")
        self.c0   = c0
        self.rho0 = rho0
        self._rt  = RetardedTimeSolver(c0=c0)

    def solve(self, surface, observer, t_out=None):
        """
        Compute far-field pressure at observer and store results in-place.
        Returns the observer for convenience.
        """
        c0, rho0 = self.c0, self.rho0
        x_obs = observer.position

        r_vec = x_obs - surface.centroids                    # (N, 3)
        r     = np.linalg.norm(r_vec, axis=1)                # (N,)
        r_hat = r_vec / r[:, None]                           # (N, 3)
        dS    = surface.areas                                 # (N,)

        if t_out is None:
            t_out = surface.time.copy()

        tau = self._rt.stationary(r, t_out)                  # (M, N)

        def interp(q):
            # q: (M, N) → interpolate each column at tau[:, n]
            M, N = tau.shape
            out  = np.empty((M, N))
            for n in range(N):
                out[:, n] = np.interp(tau[:, n], surface.time, q[:, n])
            return out

        # thickness noise — driven by fluid normal velocity on the surface
        Un     = surface.fluid_normal_velocity()             # (M, N)
        Un_ret = interp(Un)
        I_T    = np.sum(Un_ret * dS / r, axis=1)
        p_T    = rho0 * np.gradient(I_T, surface.dt) / (4 * np.pi)

        # loading noise — driven by unsteady surface pressure
        n_dot_r = np.einsum("nk,nk->n", surface.normals, r_hat)   # (N,)
        Lr      = surface.pressure * n_dot_r                        # (M, N)
        Lr_ret  = interp(Lr)

        I_ff = np.sum(Lr_ret * dS / r,    axis=1)
        I_nf = np.sum(Lr_ret * dS / r**2, axis=1)
        p_L  = np.gradient(I_ff, surface.dt) / (c0 * 4*np.pi) + I_nf / (4*np.pi)

        observer._store_results(t_out, p_T, p_L)
        return observer

    def solve_batch(self, surface, observers):
        """Run solve() for a list of observers (mic array, directivity sweep)."""
        return [self.solve(surface, obs) for obs in observers]