import numpy as np
import jax.numpy as jnp
from kf_da.velInit.IC_init import IC_init

import jax

class AI(IC_init):
    def __init__(self, min_norm, max_norm):
        self.min_norm = min_norm
        self.max_norm = max_norm

    def get_attractor_snaps(self, attractor_snapshots):
        self.attractor_snapshots = attractor_snapshots
        self.N = attractor_snapshots.shape[0]
        self.attractor_rad = self.calc_attractor_size(attractor_snapshots)
        return self.attractor_rad
    def set_unused_mask(self):
        self.unused_IC_mask = np.ones(self.N, dtype=np.bool)

    def __repr__(self):
        return "AI"
    
    def __call__dec(self, U_0, pIC, DA_loss_fn_base, opt_init_seed_num):
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
    
    def __call__(self, U_0, pIC, DA_loss_fn_base, key_num):
        key = jax.random.PRNGKey(key_num)

        # Distances to all snapshots (global indexing)
        dists = jnp.linalg.norm(
            self.attractor_snapshots - jnp.expand_dims(U_0, axis=0),
            axis=(1, 2),
        )
        norm_dists = dists / self.attractor_rad

        # Eligible = within [min_norm, max_norm]
        eligible = (norm_dists >= self.min_norm) & (norm_dists <= self.max_norm)
        eligible_idx = jnp.where(eligible)[0]

        if eligible_idx.size > 0:
            # pick uniformly at random from eligible snapshots
            IC_idx = jax.random.choice(key, eligible_idx)
        else:
            # fallback: pick snapshot closest to the middle of the band
            target = 0.5 * (self.min_norm + self.max_norm)
            IC_idx = jnp.argmin(jnp.abs(norm_dists - target))

        U_0_guess = self.attractor_snapshots[IC_idx, :]

        actual_norm_dist = jnp.linalg.norm(U_0_guess - U_0) / self.attractor_rad
        return U_0_guess, actual_norm_dist
