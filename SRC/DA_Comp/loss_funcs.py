import jax.numpy as jnp
from SRC.Solver.KF_intergrators import RK4_Step, Particle_Stepper
from SRC.utils import real_concat_to_complex
from jax import lax

def build_transform_fn(NDOF, KX, KY, K2):
    def transform_fn(X):
        X = X.reshape((2, NDOF, NDOF))
        X = project_divfree_rfft2_safe(X, KX, KY, K2).reshape(-1)
        return X
    
    return transform_fn

def project_divfree_rfft2_safe(U, KX, KY, K2):
    """
    U_hat: (2, Ny, Nx//2+1) complex, rfft2 layout
    KX,KY,K2: real arrays (Ny, Nx//2+1)
    """
    u_hat = jnp.fft.rfft2(U[0])
    v_hat = jnp.fft.rfft2(U[1])

    # Build a denom that is never zero (only modify the DC cell)
    eps = jnp.finfo(K2.dtype).tiny  # or use .eps
    mask = (K2 > 0).astype(K2.dtype)         # 1 for non-DC, 0 at DC
    K2_safe = K2 + (1.0 - mask) * eps        # add eps only where K2==0

    k_dot_U = KX * u_hat + KY * v_hat  # complex
    # Single safe division + mask (no NaNs from hidden branches)
    scale = (k_dot_U / K2_safe) * mask
    scale = scale.astype(u_hat.dtype)        # keep complex dtype

    Ux_proj = u_hat - KX * scale
    Uy_proj = v_hat - KY * scale

    #if zero_mean:
    #    Ux_proj = Ux_proj.at[0, 0].set(0)
    #    Uy_proj = Uy_proj.at[0, 0].set(0)

    return jnp.stack([jnp.fft.irfft2(Ux_proj), jnp.fft.irfft2(Uy_proj)], axis=0)

def create_loss_fn(crit, stepper, target_parts, pIC, transform=True):
    transform_fn = build_transform_fn(
                    NDOF = stepper.step.rhs.KF_RHS.N,
                    KX = stepper.step.rhs.KF_RHS.KX,
                    KY = stepper.step.rhs.KF_RHS.KY,
                    K2 = stepper.step.rhs.KF_RHS.K2,
    )

    def loss_fn(U0):
        if transform:
            U0 = transform_fn(U0)


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