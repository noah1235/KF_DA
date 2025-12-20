import jax.numpy as jnp
from SRC.Solver.KF_intergrators import RK4_Step, Particle_Stepper
from SRC.utils import build_div_free_proj, Vel_Part_Transformations, Specteral_Upsampling, bilinear_sample_periodic
from jax import lax



def create_loss_fn(crit, stepper: Particle_Stepper, target_trj, pIC, vel_part_trans: Vel_Part_Transformations):
    transform_fn = build_div_free_proj(
                    stepper,
                    vel_part_trans
    )
    upsample_factor = stepper.step.rhs.r
    M = stepper.step.rhs.KF_RHS.M
    def loss_fn(U0_fourier):
        U0 = transform_fn(U0_fourier, M=M)

        X0 = jnp.concatenate([pIC, U0])

        def body(X, data):
            target_snapshot, i = data
            particles, U_flat = vel_part_trans.split_part_and_vel(X)
            xp, yp, up, vp = vel_part_trans.get_part_pos_and_vel(particles)

            #extracting target data
            trg_particles, trg_U_flat = vel_part_trans.split_part_and_vel(target_snapshot)
            trg_xp, trg_yp, trg_up, trg_vp = vel_part_trans.get_part_pos_and_vel(trg_particles)

            def have_measurment(_):
                return jnp.concatenate([trg_particles, U_flat], axis=0), crit.g(xp, yp, trg_xp, trg_yp, U_flat, trg_U_flat, upsample_factor, i)

            def no_measurment(_):
                return jnp.concatenate([particles, U_flat], axis=0), jnp.array(0.0, dtype=X.dtype)

            X, loss_t = lax.cond(crit.t_mask[i] != 0, have_measurment, no_measurment, operand=None)


            
            X_next = stepper(X)
            return X_next, loss_t

        nsteps = target_trj.shape[0]
        idxs = jnp.arange(nsteps, dtype=jnp.int32)
        xs = (target_trj.real, idxs)

        _, losses = lax.scan(body, X0, xs=xs)

        return jnp.sum(losses)

    return loss_fn

class Loss_fn:
    def init_obj(self, t_mask, L, vel_part_trans: Vel_Part_Transformations):
        self.t_mask = t_mask
        self.num_frames = jnp.sum(t_mask)
        self.vel_part_trans = vel_part_trans
        self.L = L


class MSE_PP(Loss_fn):
    def g(self, part_x, part_y, target_part_x, target_part_y,
          U_flat, trg_U_flat, upsample_factor, i):

        # periodic (minimum-image) differences in x and y
        dx = part_x - target_part_x
        dy = part_y - target_part_y

        dx = dx - self.L * jnp.round(dx / self.L)
        dy = dy - self.L * jnp.round(dy / self.L)

        MSE_x = jnp.mean(dx**2)
        MSE_y = jnp.mean(dy**2)

        return self.t_mask[i] * ((MSE_x + MSE_y) / 2.0) / self.num_frames

    def __repr__(self):
        return "PP_MSE"
    
class MSE_Vel(Loss_fn):
    def g(self, part_x, part_y, target_part_x, target_part_y, U_flat, trg_U_flat, upsample_factor, i):
        U = self.vel_part_trans.reshape_flattened_vel(U_flat)
        trg_U = self.vel_part_trans.reshape_flattened_vel(trg_U_flat)

        # upsample to a finer grid for particle sampling
        trg_u_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(trg_U[0]), upsample_factor)
        trg_v_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(trg_U[1]), upsample_factor)

        u_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(U[0]), upsample_factor)
        v_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(U[1]), upsample_factor)

        # bilinear samples (periodic) at particle positions
        u = bilinear_sample_periodic(u_fine, target_part_x, target_part_y, self.L, self.L)
        v = bilinear_sample_periodic(v_fine, target_part_x, target_part_y, self.L, self.L)

        trg_u = bilinear_sample_periodic(trg_u_fine, target_part_x, target_part_y, self.L, self.L)
        trg_v = bilinear_sample_periodic(trg_v_fine, target_part_x, target_part_y, self.L, self.L)


        MSE_u = jnp.mean((u - trg_u)**2)
        MSE_v = jnp.mean((v - trg_v)**2)


        return self.t_mask[i] * ((MSE_u + MSE_v)/2) / self.num_frames

    def __repr__(self):
        return "Vel_MSE"