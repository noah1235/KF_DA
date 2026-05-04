import numpy as np
import jax.numpy as jnp
from kf_da.utils.utils import bilinear_sample_periodic, Specteral_Upsampling


import jax
import jax.numpy as jnp

def init_particles_vector(
    n: int,
    u: jnp.ndarray,
    v: jnp.ndarray,
    x_range,
    y_range,
    L: float,
    seed: int = 0,
):


    key = jax.random.PRNGKey(seed)
    kx, ky = jax.random.split(key, 2)
    xs = jax.random.uniform(kx, shape=(n,), minval=x_range[0], maxval=x_range[1])
    ys = jax.random.uniform(ky, shape=(n,), minval=y_range[0], maxval=y_range[1])

    us = bilinear_sample_periodic(u, xs, ys, L, L)
    vs = bilinear_sample_periodic(v, xs, ys, L, L)

    return xs, ys, us, vs
