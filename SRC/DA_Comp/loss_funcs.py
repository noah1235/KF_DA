import jax.numpy as jnp
from SRC.Solver.solver import KF_TP_Stepper
from SRC.utils import bilinear_sample_periodic
from jax import lax



def create_loss_fn(crit, stepper: KF_TP_Stepper, target_trj, inv_transform):
    omega_traj, xp_traj, yp_traj, up_traj, vp_traj = target_trj
    def loss_fn(Z0):
        omega0_hat = inv_transform(Z0)

        def body(X, data):
            omega_DA, xp_DA, yp_DA, up_DA, vp_DA = X
            omega_trg, xp_trg, yp_trg, up_trg, vp_trg, i = data

            def have_measurment(_):
                return xp_trg, yp_trg, up_trg, vp_trg, crit.g(xp_DA, yp_DA, xp_trg, yp_trg, omega_DA, omega_trg, stepper.NS.vort_hat_2_vel_hat, i)
                #return xp_DA, yp_DA, up_DA, vp_DA, crit.g(xp_DA, yp_DA, xp_trg, yp_trg, omega_DA, omega_trg, stepper.NS.vort_hat_2_vel_hat, i)

            def no_measurment(_):
                return xp_DA, yp_DA, up_DA, vp_DA, jnp.array(0.0)

            xp_DA, yp_DA, up_DA, vp_DA, loss_t = lax.cond(crit.t_mask[i] != 0, have_measurment, no_measurment, operand=None)


            omega_DA, xp_DA, yp_DA, up_DA, vp_DA = stepper(omega_DA, xp_DA, yp_DA, up_DA, vp_DA)
            return (omega_DA, xp_DA, yp_DA, up_DA, vp_DA), loss_t

        nsteps = omega_traj.shape[0]
        idxs = jnp.arange(nsteps, dtype=jnp.int32)
        xs = (omega_traj, xp_traj, yp_traj, up_traj, vp_traj, idxs)
        X0 = (omega0_hat, xp_traj[0], yp_traj[0], up_traj[0], vp_traj[0])
        _, losses = lax.scan(body, X0, xs=xs)

        return jnp.sum(losses)

    return loss_fn

class Loss_fn:
    def init_obj(self, t_mask, L):
        self.t_mask = t_mask
        self.num_frames = jnp.sum(t_mask)
        self.L = L


class MSE_PP(Loss_fn):
    def g(self, xp_DA, yp_DA, xp_trg, yp_trg, 
          omega_DA, omega_trg, vort_2_vel_hat_fn, i):

        # periodic (minimum-image) differences in x and y
        dx = xp_DA - xp_trg
        dy = yp_DA - yp_trg

        dx = dx - self.L * jnp.round(dx / self.L)
        dy = dy - self.L * jnp.round(dy / self.L)

        MSE_x = jnp.mean(dx**2)
        MSE_y = jnp.mean(dy**2)

        return self.t_mask[i] * ((MSE_x + MSE_y) / 2.0) / self.num_frames

    def __repr__(self):
        return "PP_MSE"
    
class MSE_Vel(Loss_fn):
    @staticmethod
    def _vort_2_vel(omega, vort_2_vel_hat_fn):
        u_hat, v_hat = vort_2_vel_hat_fn(omega)
        return jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)

    def g(self, xp_DA, yp_DA, xp_trg, yp_trg, 
          omega_DA, omega_trg, vort_2_vel_hat_fn, i):
        u_trj, v_trj = self._vort_2_vel(omega_trg, vort_2_vel_hat_fn)
        u_DA, v_DA = self._vort_2_vel(omega_DA, vort_2_vel_hat_fn)


        # bilinear samples (periodic) at particle positions
        u_DA = bilinear_sample_periodic(u_DA, xp_trg, yp_trg, self.L, self.L)
        v_DA = bilinear_sample_periodic(v_DA, xp_trg, yp_trg, self.L, self.L)

        u_trj = bilinear_sample_periodic(u_trj, xp_trg, yp_trg, self.L, self.L)
        v_trj = bilinear_sample_periodic(v_trj, xp_trg, yp_trg, self.L, self.L)


        MSE_u = jnp.mean((u_DA - u_trj)**2)
        MSE_v = jnp.mean((v_DA - v_trj)**2)


        return self.t_mask[i] * ((MSE_u + MSE_v)/2) / self.num_frames

    def __repr__(self):
        return "Vel_MSE"