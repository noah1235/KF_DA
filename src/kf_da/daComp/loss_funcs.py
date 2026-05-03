import jax.numpy as jnp
from kf_da.solver.solver import KF_TP_Stepper
from kf_da.utils.utils import bilinear_sample_periodic
from jax import lax
import jax

def create_loss_fn_bu(crit, stepper: KF_TP_Stepper, target_trj, pp_sigma, meas_part_pos, inv_transform):
    # omega_traj, xp_traj, yp_traj, up_traj, vp_traj = target_trj
    omega_traj, _, _, up_traj, vp_traj = target_trj
    xp_meas_traj, yp_meas_traj = meas_part_pos

    num_parts = xp_meas_traj.shape[0] * xp_meas_traj.shape[1]

    if pp_sigma is not None:
        sigma_x, sigma_y = pp_sigma
    else:
        sigma_x, sigma_y = 1, 1

    def loss_fn(Z0, PP_opt):    
        xp_opt_traj = PP_opt[:num_parts].reshape(xp_meas_traj.shape)              # (m, n)
        yp_opt_traj = PP_opt[num_parts:2 * num_parts].reshape(yp_meas_traj.shape)      # (m, n)

        omega0_hat = inv_transform(Z0)

        def body(X, data):
            omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx = X
            omega_trg, up_trg, vp_trg, i = data

            has_meas = crit.t_mask[i]

            def have_measurement(carry):
                omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx = carry

                # Pull the current measurement row k = meas_idx from the (m, n) arrays
                xp_meas = lax.dynamic_index_in_dim(xp_meas_traj, meas_idx, axis=0, keepdims=False)
                yp_meas = lax.dynamic_index_in_dim(yp_meas_traj, meas_idx, axis=0, keepdims=False)
                xp_opt  = lax.dynamic_index_in_dim(xp_opt_traj,  meas_idx, axis=0, keepdims=False)
                yp_opt  = lax.dynamic_index_in_dim(yp_opt_traj,  meas_idx, axis=0, keepdims=False)

                def use_DA(_):
                    return crit.g(
                        xp_DA, yp_DA,
                        #xp_meas, yp_meas,
                        xp_opt, yp_opt,
                        sigma_x, sigma_y,
                        omega_DA, omega_trg,
                        stepper.NS.vort_hat_2_vel_hat,
                        i,
                    )

                def use_trg(_):
                    return crit.g(
                        #xp_meas, yp_meas,
                        #xp_meas, yp_meas,
                        xp_opt, yp_opt,
                        xp_opt, yp_opt,
                        sigma_x, sigma_y,
                        omega_DA, omega_trg,
                        stepper.NS.vort_hat_2_vel_hat,
                        i,
                    )

                loss_t = lax.cond(init_part, use_DA, use_trg, operand=None)

                # After a measurement, reset DA particle state to optimized particle state
                return (
                    omega_DA,
                    xp_opt,
                    yp_opt,
                    up_trg,
                    vp_trg,
                    True,
                    meas_idx + 1,
                ), loss_t

            def no_measurement(carry):
                omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx = carry
                return (
                    omega_DA,
                    xp_DA,
                    yp_DA,
                    up_DA,
                    vp_DA,
                    init_part,
                    meas_idx,
                ), jnp.array(0.0)

            (omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx), loss_t = lax.cond(
                has_meas,
                have_measurement,
                no_measurement,
                X,
            )

            omega_DA, xp_DA, yp_DA, up_DA, vp_DA = stepper(
                omega_DA, xp_DA, yp_DA, up_DA, vp_DA
            )

            return (omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx), loss_t

        nsteps = omega_traj.shape[0]
        idxs = jnp.arange(nsteps, dtype=jnp.int32)

        xs = (omega_traj, up_traj, vp_traj, idxs)

        # Initial particle positions should come from the first measurement row
        X0 = (
            omega0_hat,
            xp_opt_traj[0],
            yp_opt_traj[0],
            up_traj[0],
            vp_traj[0],
            False,   # init_part
            0,       # meas_idx
        )

        _, losses = lax.scan(body, X0, xs)

        #return pp_opt_reg
        return (jnp.sum(losses)) * jnp.sqrt(sigma_x * sigma_y)

    return loss_fn


def create_loss_fn(
    crit,
    stepper,
    target_trj,
    pp_sigma,
    meas_part_pos,
    inv_transform,
    checkpoint=False,
):
    omega_traj, _, _, up_traj, vp_traj = target_trj
    xp_meas_traj, yp_meas_traj = meas_part_pos

    num_parts = xp_meas_traj.shape[0] * xp_meas_traj.shape[1]

    if pp_sigma is not None:
        sigma_x, sigma_y = pp_sigma
    else:
        sigma_x, sigma_y = 1.0, 1.0

    def loss_fn(Z0, PP_opt):
        xp_opt_traj = PP_opt[:num_parts].reshape(xp_meas_traj.shape)
        yp_opt_traj = PP_opt[num_parts:2 * num_parts].reshape(yp_meas_traj.shape)

        omega0_hat = inv_transform(Z0)

        def body(X, data):
            omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx = X
            omega_trg, up_trg, vp_trg, i = data

            has_meas = crit.t_mask[i]

            def have_measurement(carry):
                omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx = carry

                xp_meas = lax.dynamic_index_in_dim(
                    xp_meas_traj, meas_idx, axis=0, keepdims=False
                )
                yp_meas = lax.dynamic_index_in_dim(
                    yp_meas_traj, meas_idx, axis=0, keepdims=False
                )
                xp_opt = lax.dynamic_index_in_dim(
                    xp_opt_traj, meas_idx, axis=0, keepdims=False
                )
                yp_opt = lax.dynamic_index_in_dim(
                    yp_opt_traj, meas_idx, axis=0, keepdims=False
                )

                def use_DA(_):
                    return crit.g(
                        xp_DA,
                        yp_DA,
                        xp_opt,
                        yp_opt,
                        sigma_x,
                        sigma_y,
                        omega_DA,
                        omega_trg,
                        stepper.NS.vort_hat_2_vel_hat,
                        i,
                    )

                def use_trg(_):
                    return crit.g(
                        xp_opt,
                        yp_opt,
                        xp_opt,
                        yp_opt,
                        sigma_x,
                        sigma_y,
                        omega_DA,
                        omega_trg,
                        stepper.NS.vort_hat_2_vel_hat,
                        i,
                    )

                loss_t = lax.cond(init_part, use_DA, use_trg, operand=None)

                return (
                    omega_DA,
                    xp_opt,
                    yp_opt,
                    up_trg,
                    vp_trg,
                    True,
                    meas_idx + 1,
                ), loss_t

            def no_measurement(carry):
                omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx = carry
                return (
                    omega_DA,
                    xp_DA,
                    yp_DA,
                    up_DA,
                    vp_DA,
                    init_part,
                    meas_idx,
                ), jnp.array(0.0, dtype=omega_DA.real.dtype if jnp.iscomplexobj(omega_DA) else omega_DA.dtype)

            (omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx), loss_t = lax.cond(
                has_meas,
                have_measurement,
                no_measurement,
                X,
            )

            omega_DA, xp_DA, yp_DA, up_DA, vp_DA = stepper(
                omega_DA, xp_DA, yp_DA, up_DA, vp_DA
            )

            return (omega_DA, xp_DA, yp_DA, up_DA, vp_DA, init_part, meas_idx), loss_t

        # Apply remat/checkpoint to the scan body if requested.
        scan_body = jax.checkpoint(body) if checkpoint else body

        nsteps = omega_traj.shape[0]
        idxs = jnp.arange(nsteps, dtype=jnp.int32)
        xs = (omega_traj, up_traj, vp_traj, idxs)

        X0 = (
            omega0_hat,
            xp_opt_traj[0],
            yp_opt_traj[0],
            up_traj[0],
            vp_traj[0],
            False,
            0,
        )

        _, losses = lax.scan(scan_body, X0, xs)
        return jnp.sum(losses) * jnp.sqrt(sigma_x * sigma_y)

    return loss_fn

class Loss_fn:
    def init_obj(self, t_mask, L):
        self.t_mask = t_mask
        self.num_frames = jnp.sum(t_mask)
        self.L = L


class MSE_PP(Loss_fn):
    def g(self, xp_DA, yp_DA, xp_trg, yp_trg, sigma_x, sigma_y,
          omega_DA, omega_trg, vort_2_vel_hat_fn, i):

        # periodic (minimum-image) differences in x and y
        dx = xp_DA - xp_trg
        dy = yp_DA - yp_trg

        dx = dx - self.L * jnp.round(dx / self.L)
        dy = dy - self.L * jnp.round(dy / self.L)

        MSE_x = jnp.mean(dx**2) / sigma_x**2
        MSE_y = jnp.mean(dy**2) / sigma_y**2

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
