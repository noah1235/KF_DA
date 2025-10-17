import jax.numpy as jnp

def every_nth_one(length, n):
    """
    Create a JAX array of given length with 1s every nth index,
    starting at index 0, and 0s elsewhere.
    """
    idx = jnp.arange(length)
    arr = (idx % n == 0).astype(jnp.float32)
    return arr

print(every_nth_one(10, 2))