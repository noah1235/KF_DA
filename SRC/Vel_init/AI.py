import numpy as np
import jax.numpy as jnp
from SRC.Vel_init.IC_init import IC_init

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
    
    def __call__(self, U_0, pIC, DA_loss_fn_base):
        norm_dist = np.random.uniform(
                low=self.min_norm,
                high=self.max_norm,
            )
        
        dist = self.attractor_rad * norm_dist
        true_IC_dist = jnp.linalg.norm(
        self.attractor_snapshots[self.unused_IC_mask, :] - jnp.expand_dims(U_0, axis=0), axis=(1, 2)
        )

        IC_idx = jnp.argmin(jnp.abs(true_IC_dist - dist))
        U_0_guess = self.attractor_snapshots[
            IC_idx, :
        ]
        self.unused_IC_mask[IC_idx] = False
        actual_norm_dist = jnp.linalg.norm(U_0_guess - U_0)/self.attractor_rad
        return U_0_guess, actual_norm_dist