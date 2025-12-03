import jax.numpy as jnp

class IC_init:
    @staticmethod
    def calc_attractor_size(attractor_snapshots: jnp.ndarray) -> jnp.ndarray:
        """
        Estimate a characteristic size of the attractor as the mean distance
        of snapshots to their mean state (in the full-state Euclidean norm).

        Parameters
        ----------
        attractor_snapshots : jnp.ndarray
            Array of shape (Nsamples, Nstate) containing sampled states from the attractor.

        Returns
        -------
        jnp.ndarray
            Scalar mean distance (0-D array) representing an attractor "size" scale.
        """
        mean = jnp.mean(attractor_snapshots, axis=0)
        dist = jnp.linalg.norm(attractor_snapshots - mean.reshape((1, -1)), axis=1)
        return jnp.mean(dist)