# CSV reader for FWH surface data.
#
# Expected layout:
#   <data_dir>/
#       geometry.csv       static panel geometry
#       t_<XXXXXX>.csv     one file per time step
#
# geometry.csv columns:  x, y, z, nx, ny, nz, area
# timestep columns:      panel_id, p, vx, vy, vz [, usx, usy, usz]
#
# Surface velocity columns (usx, usy, usz) are optional.
# If absent, the surface is treated as stationary.

import os
import re
import numpy as np
from .base     import SurfaceReader
from ..surface import FWHSurface


class CSVReader(SurfaceReader):
    """
    Reads FWH surface data from a directory of CSV files.

    data_dir : path to directory with geometry.csv and t_*.csv files
    dt       : time step [s]; if None, uses dt_hint * file index
    name     : label for the returned FWHSurface
    """

    _GEOM    = "geometry.csv"
    _PATTERN = re.compile(r"t_(\d+)\.csv$")

    def __init__(self, data_dir, dt=None, dt_hint=1e-4, name="surface"):
        self.data_dir = data_dir
        self.dt       = dt
        self.dt_hint  = dt_hint
        self.name     = name

    def read(self):
        centroids, areas, normals = self._read_geom()
        time, pressure, velocity, surf_vel = self._read_steps(len(areas))
        return FWHSurface(
            centroids=centroids, areas=areas, normals=normals,
            time=time, pressure=pressure, velocity=velocity,
            surface_velocity=surf_vel, name=self.name,
        )

    def _read_geom(self):
        path = os.path.join(self.data_dir, self._GEOM)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"geometry.csv not found in {self.data_dir}")
        d         = np.genfromtxt(path, delimiter=",", names=True)
        centroids = np.column_stack([d["x"], d["y"], d["z"]])
        areas     = d["area"].astype(float)
        n_raw     = np.column_stack([d["nx"], d["ny"], d["nz"]])
        normals   = n_raw / np.linalg.norm(n_raw, axis=1, keepdims=True)
        return centroids, areas, normals

    def _read_steps(self, n_panels):
        files = {}
        for fname in os.listdir(self.data_dir):
            m = self._PATTERN.match(fname)
            if m:
                files[int(m.group(1))] = os.path.join(self.data_dir, fname)
        if not files:
            raise FileNotFoundError(f"no t_*.csv files found in {self.data_dir}")

        indices  = sorted(files.keys())
        T        = len(indices)
        pressure = np.empty((T, n_panels))
        velocity = np.empty((T, n_panels, 3))
        surf_vel = None

        for i, idx in enumerate(indices):
            d    = np.genfromtxt(files[idx], delimiter=",", names=True)
            cols = d.dtype.names
            pressure[i] = d["p"]
            velocity[i] = np.column_stack([d["vx"], d["vy"], d["vz"]])
            if "usx" in cols:
                if surf_vel is None:
                    surf_vel = np.zeros((T, n_panels, 3))
                surf_vel[i] = np.column_stack([d["usx"], d["usy"], d["usz"]])

        dt   = self.dt if self.dt is not None else self.dt_hint
        time = np.array(indices, dtype=float) * dt
        return time, pressure, velocity, surf_vel