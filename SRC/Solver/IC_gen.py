import numpy as np
import jax.numpy as jnp
from SRC.utils import bilinear_sample_periodic, Specteral_Upsampling


import jax
import jax.numpy as jnp


def init_particles_vector(n, u, v, x_range, y_range, L, rng=None, r=4):
    """
    Initialize particle state vector:
    (x1, y1, u1, v1, x2, y2, u2, v2, ..., xN, yN, uN, vN).
    
    Parameters
    ----------
    n : int
        Number of particles.
    x_range : tuple(float, float)
        (min, max) range for x coordinates.
    y_range : tuple(float, float)
        (min, max) range for y coordinates.
    rng : np.random.Generator or None
        Random generator for reproducibility.
    
    Returns
    -------
    z : np.ndarray, shape (4*n,)
        State vector for all particles.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    xs = rng.uniform(x_range[0], x_range[1], size=n)
    ys = rng.uniform(y_range[0], y_range[1], size=n)

    us = bilinear_sample_periodic(u, xs, ys, L, L)
    vs = bilinear_sample_periodic(v, xs, ys, L, L)


    
    return xs, ys, us, vs

