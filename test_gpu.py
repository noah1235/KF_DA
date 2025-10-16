# test_jax_gpu.py
import jax
import jax.numpy as jnp

def main():
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
    print("Default device:", jax.default_backend())

    # Create random matrices on default device
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.uniform(key1, (4000, 4000), dtype=jnp.float32)
    y = jax.random.uniform(key2, (4000, 4000), dtype=jnp.float32)

    # Matrix multiply (forces compute on the device)
    z = jnp.dot(x, y)

    # Trigger computation and measure device
    z_block = z.block_until_ready()

    print("Computation done on device:", z_block.device())
    print("z mean:", z_block.mean())

if __name__ == "__main__":
    main()
