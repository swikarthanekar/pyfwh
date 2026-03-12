import numpy as np
from ..surface import FWHSurface


def _fibonacci_sphere(radius, n):
    # Fibonacci lattice gives more uniform panel distribution than lat/lon grids
    golden = (1 + 5**0.5) / 2
    i      = np.arange(n)
    theta  = np.arccos(1 - 2*(i + 0.5)/n)
    phi    = 2*np.pi*i / golden
    pts    = np.stack([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta),
    ], axis=1)
    normals = pts / radius
    areas   = np.full(n, 4*np.pi*radius**2 / n)
    return pts, normals, areas


def monopole_on_sphere(
    radius=0.5, n_panels=256, frequency=100.0, amplitude=1.0,
    c0=340.0, rho0=1.225, n_periods=8, n_timesteps=512, p_inf=101325.0
):
    """
    Monopole source surrounded by a spherical FWH surface.

    Exact far-field solution:  p'(r,t) = (A/r) * cos(w*(t - r/c0))

    Both pressure and radial fluid velocity on the sphere are set from
    the analytical monopole solution. Used to validate the solver against
    a known result.

    Returns (surface, exact) where exact contains p_exact at a 10m observer.
    """
    omega = 2*np.pi*frequency
    time  = np.linspace(0, n_periods/frequency, n_timesteps, endpoint=False)

    centroids, normals, areas = _fibonacci_sphere(radius, n_panels)
    N      = centroids.shape[0]
    r_pan  = np.linalg.norm(centroids, axis=1)
    phase  = omega * (time[:, None] - r_pan[None, :] / c0)

    pressure = p_inf + (amplitude / r_pan) * np.cos(phase)

    v_r = ((amplitude / (rho0 * c0 * r_pan)) * np.cos(phase)
         + (amplitude / (rho0 * omega * r_pan**2)) * np.sin(phase))
    r_hat    = centroids / r_pan[:, None]
    velocity = v_r[:, :, None] * r_hat[None, :, :]

    surface = FWHSurface.from_dict(dict(
        centroids=centroids, normals=normals, areas=areas, time=time,
        pressure=pressure, velocity=velocity,
        surface_velocity=np.zeros((n_timesteps, N, 3)),
    ), name="monopole_sphere")

    r_obs   = 10.0
    obs_pos = np.array([r_obs, 0.0, 0.0])
    p_exact = (amplitude / r_obs) * np.cos(omega * (time - r_obs / c0))

    return surface, dict(
        observer_position=obs_pos, time=time, p_exact=p_exact,
        frequency=frequency, amplitude=amplitude, r_obs=r_obs,
    )


def rotating_source(
    orbit_radius=0.5, frequency=50.0, rotation_rate=10.0, amplitude=1.0,
    c0=340.0, rho0=1.225, n_revolutions=10, n_timesteps=1024, p_inf=101325.0
):
    """
    Single panel orbiting in a circle — rough model of a rotating blade.

    Mainly here to exercise the moving-surface code path. An observer on
    the rotation axis sees no Doppler shift; off-axis observers do.
    """
    omega_src = 2*np.pi*frequency
    omega_rot = 2*np.pi*rotation_rate
    time      = np.linspace(0, n_revolutions/rotation_rate, n_timesteps, endpoint=False)
    N         = 1

    surf_vel       = np.zeros((n_timesteps, N, 3))
    surf_vel[:,0,0] = -orbit_radius * omega_rot * np.sin(omega_rot * time)
    surf_vel[:,0,1] =  orbit_radius * omega_rot * np.cos(omega_rot * time)

    surface = FWHSurface.from_dict(dict(
        centroids=np.array([[orbit_radius, 0.0, 0.0]]),
        normals=np.array([[1.0, 0.0, 0.0]]),
        areas=np.array([1e-4]),
        time=time,
        pressure=(p_inf + amplitude * np.cos(omega_src * time))[:, None],
        velocity=np.zeros((n_timesteps, N, 3)),
        surface_velocity=surf_vel,
    ), name="rotating_source")

    return surface, dict(
        observer_position=np.array([0.0, 0.0, 10.0]),
        time=time, rotation_rate=rotation_rate,
        frequency=frequency, M_tip=orbit_radius*omega_rot/c0,
    )