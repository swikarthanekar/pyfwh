# pyFWH

Python implementation of Farassat's Formulation 1A for far-field aeroacoustic
noise prediction. The goal is a solver that works on any surface geometry
without needing case-specific modifications.

## Background

The Ffowcs Williams-Hawkings equation lets you compute far-field acoustic
pressure by integrating flow quantities on a surface around your noise source.
Farassat's Formulation 1A splits this into two parts:

```
p'(x, t) = p'_T  +  p'_L
```

p'_T is thickness noise (from the moving body displacing fluid), p'_L is
loading noise (from unsteady pressure forces on the surface). For most
stationary CFD surfaces, loading noise dominates.

The solver uses the d/dt-outside formulation — differentiating the full
surface integral with respect to observer time rather than pointwise at each
panel. This is more stable numerically, especially at lower panel counts.

## Structure

```
pyfwh/
├── surface.py        FWHSurface — holds panel geometry + flow data
├── observer.py       Observer — listener position + results
├── solver.py         FWHSolver — the actual Formulation 1A integration
├── retarded_time.py  retarded time solver (stationary + moving surfaces)
├── io/
│   ├── base.py       abstract reader
│   └── csv_reader.py CSV format reader
└── utils/
    └── test_cases.py analytical test cases for validation
```

The solver itself has no geometry-specific code. You hand it any FWHSurface
and it runs. Adding support for a new CFD output format means writing one
reader subclass — nothing else changes.

## Install

```bash
pip install numpy scipy
# or for dev:
pip install -e ".[dev]"
```

## Usage

```python
from pyfwh import FWHSurface, Observer, FWHSolver
from pyfwh.utils import monopole_on_sphere

surface, exact = monopole_on_sphere(n_panels=256, frequency=100.0)
obs = Observer(position=[10.0, 0.0, 0.0])

solver = FWHSolver(c0=340.0)
solver.solve(surface, obs)

print(f"OASPL = {obs.oaspl():.1f} dB")
```

Loading your own CFD data:

```python
# from a directory of CSV files
# expects: data_dir/geometry.csv  +  data_dir/t_000001.csv, t_000002.csv, ...
surface = CSVReader("my_data_dir/", dt=1e-4).read()

# or directly from arrays
surface = FWHSurface.from_dict({
    "centroids":        centroids,         # (N, 3)
    "normals":          normals,           # (N, 3) outward unit normals
    "areas":            areas,             # (N,)
    "time":             time,              # (M,)
    "pressure":         pressure,          # (M, N)
    "velocity":         velocity,          # (M, N, 3)
    "surface_velocity": surface_velocity,  # (M, N, 3), zeros if stationary
})
```

Multiple observers (directivity sweep, mic array):

```python
import numpy as np
angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
observers = [Observer([10*np.cos(a), 10*np.sin(a), 0.0]) for a in angles]
solver.solve_batch(surface, observers)
```

Adding a new file format:

```python
from pyfwh.io import BaseReader

class MyReader(BaseReader):
    def read(self, name="surface"):
        # parse your format, return FWHSurface.from_dict({...})
        ...
```

## Tests

```bash
python run_tests.py
```

18 tests covering surface validation, observer behaviour, and physical
accuracy against the analytical monopole solution (0.18% amplitude error).

## References

- Ffowcs Williams & Hawkings (1969), Phil. Trans. Roy. Soc. A264
- Farassat (2007), NASA/TM-2007-214853
- Farassat & Succi (1980), J. Sound Vib. 71(3)