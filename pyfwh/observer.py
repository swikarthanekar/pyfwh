import numpy as np


class Observer:
    """Far-field listener. Stores position and results after solving."""

    def __init__(self, position, name="observer"):
        self.position = np.asarray(position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError(f"position must be length-3, got {self.position.shape}")
        self.name = name
        self._time        = None
        self._p_thickness = None
        self._p_loading   = None

    @property
    def p_total(self):
        if self._p_thickness is None or self._p_loading is None:
            return None
        return self._p_thickness + self._p_loading

    @property
    def is_solved(self):
        return self._time is not None

    def oaspl(self, p_ref=2e-5):
        """Overall SPL in dB re p_ref. Returns None if not yet solved."""
        if not self.is_solved:
            return None
        p_rms = np.sqrt(np.mean(self.p_total ** 2))
        return float(20.0 * np.log10(max(p_rms, 1e-30) / p_ref))

    def spl(self, p_ref=2e-5):
        """One-sided SPL spectrum. Returns (freq, spl_db)."""
        if not self.is_solved:
            return None
        p  = self.p_total
        dt = float(self._time[1] - self._time[0])
        N  = len(p)
        P  = np.fft.rfft(p) / N
        freq = np.fft.rfftfreq(N, d=dt)
        amp  = np.abs(P)
        amp[1:-1] *= 2.0
        p_rms = np.maximum(amp / np.sqrt(2.0), 1e-30)
        return freq, 20.0 * np.log10(p_rms / p_ref)

    def distance_to(self, point):
        return float(np.linalg.norm(self.position - np.asarray(point, dtype=float)))

    # called by FWHSolver
    def _store_results(self, time, p_thickness, p_loading):
        self._time        = time
        self._p_thickness = p_thickness
        self._p_loading   = p_loading

    def __repr__(self):
        x, y, z = self.position
        status = "solved" if self.is_solved else "pending"
        return f"Observer('{self.name}', pos=({x:.2f},{y:.2f},{z:.2f}), {status})"