import jax.numpy as jnp
from SRC.Solver.KF_intergrators import RK4_Step, Particle_Stepper
from SRC.utils import build_div_free_proj
from jax import lax


def create_loss_fn(crit, stepper, target_parts, pIC, transform=True):
    transform_fn = build_div_free_proj(
                    stepper
    )

    def loss_fn(U0):
        if transform:
            U0 = transform_fn(U0)
            if True:
                U0_re = U0.reshape((2, stepper.step.rhs.KF_RHS.N, stepper.step.rhs.KF_RHS.N))
                u_hat = jnp.fft.rfft2(U0_re[0])
                v_hat = jnp.fft.rfft2(U0_re[1])

                dxop = stepper.step.rhs.KF_RHS.KX * 1j
                dyop = stepper.step.rhs.KF_RHS.KY * 1j
                div = jnp.fft.irfft2(dxop * u_hat + dyop * v_hat)
                #print(jnp.max(div))
                #print("---------")
        
        X0 = jnp.concatenate([pIC, U0])

        def body(X, data):
            tgt, i = data
            particles = X[: stepper.n_particles * 4]
            part_x = particles[::4].real
            part_y = particles[1::4].real

            loss_t = crit.g(part_x, part_y, tgt[::4], tgt[1::4], i)
            
            X_next = stepper(X)
            return X_next, loss_t

        nsteps = target_parts.shape[0]
        idxs = jnp.arange(nsteps, dtype=jnp.int32)
        xs = (target_parts.real, idxs)

        _, losses = lax.scan(body, X0, xs=xs)

        return jnp.sum(losses)

    return loss_fn



class MSE:
    def __init__(self, t_mask):
        self.t_mask = t_mask
        self.num_frames = jnp.sum(t_mask)

    def g(self, part_x, part_y, target_part_x, target_part_y, i):
        MSE_x = jnp.mean((part_x - target_part_x)**2)
        MSE_y = jnp.mean((part_y - target_part_y)**2)
        return self.t_mask[i] * ((MSE_x + MSE_y)/2) / self.num_frames

    def __repr__(self):
        return "MSE"