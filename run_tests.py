import sys
import numpy as np

sys.path.insert(0, '.')

from pyfwh import FWHSurface, Observer, FWHSolver
from pyfwh.utils import monopole_on_sphere

passed = 0
failed = 0

def check(name, cond, msg=""):
    global passed, failed
    if cond:
        print(f"  ok  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f": {msg}" if msg else ""))
        failed += 1

def dummy_surface(N=4, M=10):
    n = np.random.randn(N, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return {
        'centroids': np.random.randn(N, 3),
        'normals': n,
        'areas': np.ones(N) * 0.01,
        'time': np.linspace(0., 0.1, M),
        'pressure': np.random.randn(M, N) * 10 + 101325.,
        'velocity': np.zeros((M, N, 3)),
        'surface_velocity': np.zeros((M, N, 3)),
    }


print("\n-- surface --")

d = dummy_surface(10, 20)
s = FWHSurface.from_dict(d, name="test")
check("construction", s.n_panels == 10 and s.n_timesteps == 20)
check("dt", np.isclose(s.dt, d['time'][1] - d['time'][0]))
check("repr", "test" in repr(s))

d2 = dummy_surface(4, 10)
d2['normals'] *= 2.0
try:
    FWHSurface.from_dict(d2)
    check("rejects bad normals", False)
except ValueError:
    check("rejects bad normals", True)

d3 = dummy_surface(4, 10)
d3['areas'][0] = -1.0
try:
    FWHSurface.from_dict(d3)
    check("rejects negative area", False)
except ValueError:
    check("rejects negative area", True)

d4 = dummy_surface(4, 10)
d4['time'] = np.array([0., .1, .05, .15, .2, .25, .3, .35, .4, .45])
try:
    FWHSurface.from_dict(d4)
    check("rejects non-monotonic time", False)
except ValueError:
    check("rejects non-monotonic time", True)

N, M = 3, 5
d5 = dummy_surface(N, M)
d5['surface_velocity'] = np.tile([1., 0., 0.], (M, N, 1))
d5['normals'] = np.tile([1., 0., 0.], (N, 1))
s5 = FWHSurface.from_dict(d5)
check("normal velocity", np.allclose(s5.normal_velocity(), 1.0))


print("\n-- observer --")

obs = Observer(position=[10., 0., 0.])
check("not solved initially", not obs.is_solved)
check("p_total is None", obs.p_total is None)
check("oaspl is None", obs.oaspl() is None)

try:
    Observer(position=[1., 0.])
    check("rejects 2d position", False)
except ValueError:
    check("rejects 2d position", True)

check("distance_to", np.isclose(Observer([3., 0., 0.]).distance_to([0., 0., 0.]), 3.0))


print("\n-- monopole validation --")

surface, exact = monopole_on_sphere(n_panels=256, n_timesteps=1024, n_periods=10)
obs = Observer(position=exact['observer_position'])
FWHSolver(c0=340.0).solve(surface, obs)

check("solved", obs.is_solved)
check("output shape", obs.p_total.shape == (surface.n_timesteps,))

skip = int(0.25 * surface.n_timesteps)
p_pred = float(np.sqrt(np.mean(obs.p_total[skip:] ** 2)))
p_ref  = float(np.sqrt(np.mean(exact['p_exact'][skip:] ** 2)))
err    = abs(p_pred - p_ref) / p_ref * 100
check("amplitude < 2%", err < 2.0, f"got {err:.2f}%")

oaspl_pred  = obs.oaspl()
oaspl_exact = 20 * np.log10(p_ref / 2e-5)
check("oaspl within 1.5 dB", abs(oaspl_pred - oaspl_exact) < 1.5,
      f"{oaspl_pred:.2f} vs {oaspl_exact:.2f}")

freqs = np.fft.rfftfreq(len(obs.p_total[skip:]), d=surface.dt)
spec  = np.abs(np.fft.rfft(obs.p_total[skip:]))
peak  = freqs[np.argmax(spec)]
df    = freqs[1] - freqs[0]
check("frequency", abs(peak - exact['frequency']) <= 2 * df,
      f"peak={peak:.1f} expected={exact['frequency']:.1f}")

pT = float(np.sqrt(np.mean(obs._p_thickness[skip:] ** 2)))
check("thickness noise present (permeable surface)", pT > 0)


print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)