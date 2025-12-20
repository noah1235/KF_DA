from ml_dtypes import float6_e2m3fn
import jax.numpy as jnp
import numpy as np

t = np.ones(4, dtype=jnp.float8_e3m4) * -15
print(t.dtype)
print(t)


