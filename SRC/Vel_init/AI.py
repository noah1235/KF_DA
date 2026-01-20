import numpy as np
import jax.numpy as jnp
from SRC.Vel_init.IC_init import IC_init
import jax

class AI(IC_init):
    def __init__(self, min_norm, max_norm):
        self.min_norm = min_norm
        self.max_norm = max_norm

    def get_attractor_snaps(self, attractor_snapshots):
        self.attractor_snapshots = attractor_snapshots
        self.unused_IC_mask = np.ones(attractor_snapshots.shape[0], dtype=np.bool)
        self.attractor_rad = self.calc_attractor_size(attractor_snapshots)

    def __repr__(self):
        return "AI"
    
    def __call__(self, U_0, pIC, DA_loss_fn_base, opt_init_seed_num):
        key = jax.random.PRNGKey(opt_init_seed_num)

        # uniform in [min_norm, max_norm)
        norm_dist = jax.random.uniform(
            key,
            shape=(),
            minval=self.min_norm,
            maxval=self.max_norm,
        )

        dist = self.attractor_rad * norm_dist

        true_IC_dist = jnp.linalg.norm(
            self.attractor_snapshots[self.unused_IC_mask, :] - jnp.expand_dims(U_0, axis=0),
            axis=(1, 2)
        )

        # find snapshot whose distance to U_0 is closest to dist
        IC_idx = jnp.argmin(jnp.abs(true_IC_dist - dist))

        U_0_guess = self.attractor_snapshots[IC_idx, :]

        # NOTE: this line mutates state; fine if you're not jitting this function.
        self.unused_IC_mask[IC_idx] = False

        actual_norm_dist = jnp.linalg.norm(U_0_guess - U_0) / self.attractor_rad

        return U_0_guess, actual_norm_dist