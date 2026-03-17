import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class FWHSurface:
    """
    Holds panel geometry and time-resolved flow data for a FWH integration surface.
    N panels, M time steps. Works for any geometry — the solver doesn't care
    whether this came from a cylinder, airfoil, or anything else.

    centroids        : (N, 3)    panel centroid positions [m]
    normals          : (N, 3)    outward unit normals
    areas            : (N,)      panel areas [m^2]
    time             : (M,)      uniformly spaced time axis [s]
    pressure         : (M, N)    surface pressure [Pa]
    velocity         : (M, N, 3) fluid velocity at surface [m/s]
    surface_velocity : (M, N, 3) panel velocity [m/s], zeros for stationary walls
    """
    
    centroids:        np.ndarray
    normals:          np.ndarray
    areas:            np.ndarray
    time:             np.ndarray
    pressure:         np.ndarray
    velocity:         np.ndarray
    surface_velocity: Optional[np.ndarray] = None
    name: str = "surface"

    def __post_init__(self):
        if self.surface_velocity is None:
            self.surface_velocity = np.zeros_like(self.velocity)
        self._validate()

    def _validate(self):
        N, M = self.n_panels, self.n_timesteps
        expected = {
            "centroids":        (N, 3),
            "normals":          (N, 3),
            "areas":            (N,),
            "pressure":         (M, N),
            "velocity":         (M, N, 3),
            "surface_velocity": (M, N, 3),
        }
        for attr, shape in expected.items():
            if getattr(self, attr).shape != shape:
                raise ValueError(f"{attr}: expected {shape}, got {getattr(self, attr).shape}")

        norms = np.linalg.norm(self.normals, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("normals are not unit vectors")

        if not np.all(np.diff(self.time) > 0):
            raise ValueError("time must be strictly increasing")

        if np.any(self.areas <= 0):
            raise ValueError("all panel areas must be positive")

    @property
    def n_panels(self):
        return self.centroids.shape[0]

    @property
    def n_timesteps(self):
        return self.time.shape[0]

    @property
    def dt(self):
        return float(self.time[1] - self.time[0])

    @property
    def T(self):
        return float(self.time[-1] - self.time[0])

    def normal_velocity(self):
        """Un = surface_velocity · n,  shape (M, N)"""
        return np.einsum("mni,ni->mn", self.surface_velocity, self.normals)

    def fluid_normal_velocity(self):
        """vn = velocity · n,  shape (M, N)"""
        return np.einsum("mni,ni->mn", self.velocity, self.normals)

    def panel_distances(self, x_obs):
        """Distances (N,) from each panel centroid to observer x_obs."""
        return np.linalg.norm(x_obs - self.centroids, axis=1)

    def panel_unit_vectors(self, x_obs):
        """Unit vectors (N, 3) from each panel toward x_obs."""
        r_vec = x_obs - self.centroids
        return r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True)

    @classmethod
    def from_dict(cls, data, name="surface"):
        keys = ["centroids", "normals", "areas", "time",
                "pressure", "velocity", "surface_velocity"]
        for k in keys:
            if k not in data:
                raise KeyError(f"missing key: {k}")
        return cls(**{k: np.asarray(data[k], dtype=float) for k in keys}, name=name)

    @classmethod
    def from_csv(cls, data_dir, dt=1e-4, name='surface'):
        from .io.csv_reader import CSVReader
        return CSVReader(data_dir, dt=dt, name=name).read()

    def __repr__(self):
        return (f"FWHSurface('{self.name}', {self.n_panels} panels, "
                f"{self.n_timesteps} steps, dt={self.dt:.2e}s)")
