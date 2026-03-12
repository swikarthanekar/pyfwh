# Retarded (emission) time solver.
#
# For a stationary surface the equation tau = t - r/c0 is explicit.
# For a moving surface it's implicit (tau appears inside r(tau)) and
# we solve it with Newton-Raphson.
#
# ref: Farassat (2007), NASA/TM-2007-214853

import numpy as np


class RetardedTimeSolver:

    def __init__(self, c0=340.0, max_iter=50, tol=1e-10):
        if c0 <= 0:
            raise ValueError(f"c0 must be positive, got {c0}")
        self.c0       = c0
        self.max_iter = max_iter
        self.tol      = tol

    def stationary(self, r, t_obs):
        """
        tau*(t, n) = t - r_n / c0  for a fixed surface.

        r     : (N,)  panel distances
        t_obs : (M,)  observer times
        returns (M, N)
        """
        return t_obs[:, None] - r[None, :] / self.c0

    def solve_stationary(self, x_obs, panel_pos, t_obs):
        """Same as stationary() but takes positions instead of pre-computed r."""
        diff = x_obs[None, :] - panel_pos        # (N, 3)
        r    = np.linalg.norm(diff, axis=1)       # (N,)
        tau  = t_obs[:, None] - r[None, :] / self.c0
        return tau, r

    @staticmethod
    def interpolate_to_retarded(data, source_time, tau):
        """
        Interpolate data from source time grid onto retarded times.

        data        : (T, N) or (T, N, 3)
        source_time : (T,)
        tau         : (M, N)
        returns (M, N) or (M, N, 3)
        """
        T  = len(source_time)
        dt = source_time[1] - source_time[0]
        t0 = source_time[0]

        idx_f  = np.clip((tau - t0) / dt, 0, T - 1 - 1e-9)
        idx_lo = np.floor(idx_f).astype(int)
        idx_hi = np.minimum(idx_lo + 1, T - 1)
        alpha  = idx_f - idx_lo

        M, N = tau.shape
        rows_lo = idx_lo.ravel()
        rows_hi = idx_hi.ravel()
        cols    = np.tile(np.arange(N), M)

        if data.ndim == 2:
            d_lo = data[rows_lo, cols].reshape(M, N)
            d_hi = data[rows_hi, cols].reshape(M, N)
            return d_lo + alpha * (d_hi - d_lo)
        else:
            d_lo = data[rows_lo, cols, :]
            d_hi = data[rows_hi, cols, :]
            a    = alpha.ravel()[:, None]
            return (d_lo + a * (d_hi - d_lo)).reshape(M, N, 3)