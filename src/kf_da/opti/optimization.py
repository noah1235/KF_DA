import jax.numpy as jnp
import jax
from jax import lax
from kf_da.opti.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from kf_da.opti.LS_TR import ArmijoLineSearch
from kf_da.opti.Quasi_Newton import LBFGS_Update, BFGS_Update
import numpy as np
from kf_da.utils.utils import bilinear_sample_periodic

class Joint_Opt:
    def __init__(self, state_opt, PP_opt_its, opt_loops):
        self.state_opt = state_opt
        self.psuedo_proj = state_opt.psuedo_proj
        self.PP_opt_its = PP_opt_its
        self.opt_loops = opt_loops
        self.inertial = False

    @staticmethod
    def periodic_diff(a, b):
        return (a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def set_pp_loss_fn(self, gen_loss_fn, PP_opt_default, pp_sigma, spatial_L, vel_trj_gen_fn, t_mask, part_arr_shape, dt):
        sigma_x, sigma_y = pp_sigma
        n_parts = PP_opt_default.shape[0] // 2
        self.n_parts = n_parts
        xp_meas = PP_opt_default[:n_parts]
        yp_meas = PP_opt_default[n_parts:2 * n_parts]
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        def PP_opt_loss_fn(Z0, PP_opt):
            PP_opt = jnp.mod(PP_opt, spatial_L)
            xp_opt = PP_opt[:n_parts]
            yp_opt = PP_opt[n_parts:2 * n_parts]

            dx = self.periodic_diff(xp_opt, xp_meas)
            dy = self.periodic_diff(yp_opt, yp_meas)

            pp_opt_reg = 0.5 * (
                jnp.mean(dx**2) / sigma_x**2
                + jnp.mean(dy**2) / sigma_y**2
            ) * jnp.sqrt(sigma_x * sigma_y)

            return gen_loss_fn(Z0, PP_opt) + pp_opt_reg

        self.loss_grad_fn_pp = jax.jit(jax.value_and_grad(PP_opt_loss_fn, argnums=1))
        self.hess_fn_pp = jax.jit(jax.hessian(PP_opt_loss_fn, argnums=1))
        self.PP_opt = PP_opt_default.copy()
        self.state_opt.PP_opt = self.PP_opt
        self.PP_opt_loss_fn = jax.jit(PP_opt_loss_fn)
        self.spatial_L = spatial_L
        self.vel_trj_gen_fn = vel_trj_gen_fn
        self.t_mask = t_mask
        self.meas_indices = jnp.where(t_mask)[0]
        self.part_arr_shape = part_arr_shape
        self.dt = dt

    def set_inertial_pp_loss_fn(self, gen_loss_fn, PP_opt_default, pp_sigma, spatial_L, vel_trj_gen_fn, t_mask, part_arr_shape, dt, vel_sigma):
        self.inertial = True
        sigma_x, sigma_y = pp_sigma
        sigma_vx, sigma_vy = vel_sigma
        n_parts = PP_opt_default.shape[0] // 4
        self.n_parts = n_parts
        xp_meas = PP_opt_default[:n_parts]
        yp_meas = PP_opt_default[n_parts:2 * n_parts]
        up_meas = PP_opt_default[2 * n_parts:3 * n_parts]
        vp_meas = PP_opt_default[3 * n_parts:4 * n_parts]
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        def PP_opt_loss_fn(Z0, PP_opt):
            pos_wrapped = jnp.mod(PP_opt[:2 * n_parts], spatial_L)
            PP_opt_w = jnp.concatenate([pos_wrapped, PP_opt[2 * n_parts:]])

            xp_opt = PP_opt_w[:n_parts]
            yp_opt = PP_opt_w[n_parts:2 * n_parts]
            up_opt = PP_opt_w[2 * n_parts:3 * n_parts]
            vp_opt = PP_opt_w[3 * n_parts:4 * n_parts]

            dx  = self.periodic_diff(xp_opt, xp_meas)
            dy  = self.periodic_diff(yp_opt, yp_meas)
            dux = up_opt - up_meas
            dvy = vp_opt - vp_meas

            pp_opt_reg = 0.5 * (
                jnp.mean(dx**2) / sigma_x**2
                + jnp.mean(dy**2) / sigma_y**2
            ) * jnp.sqrt(sigma_x * sigma_y)

            vp_opt_reg = 0.5 * (
                jnp.mean(dux**2) / sigma_vx**2
                + jnp.mean(dvy**2) / sigma_vy**2
            ) * jnp.sqrt(sigma_vx * sigma_vy)

            return gen_loss_fn(Z0, PP_opt_w) + pp_opt_reg + vp_opt_reg

        self.loss_grad_fn_pp = jax.jit(jax.value_and_grad(PP_opt_loss_fn, argnums=1))
        self.hess_fn_pp = jax.jit(jax.hessian(PP_opt_loss_fn, argnums=1))
        self.PP_opt = PP_opt_default.copy()
        self.state_opt.PP_opt = self.PP_opt
        self.PP_opt_loss_fn = jax.jit(PP_opt_loss_fn)
        self.spatial_L = spatial_L
        self.vel_trj_gen_fn = vel_trj_gen_fn
        self.t_mask = t_mask
        self.meas_indices = jnp.where(t_mask)[0]
        self.part_arr_shape = part_arr_shape
        self.dt = dt

    def _wrap_pp_opt(self, PP_opt):
        if self.inertial:
            pos_wrapped = jnp.mod(PP_opt[:2 * self.n_parts], self.spatial_L)
            return jnp.concatenate([pos_wrapped, PP_opt[2 * self.n_parts:]])
        return jnp.mod(PP_opt, self.spatial_L)

    def opt_pp(self, Z0, get_IC_state_fn):
        c1 = 1e-4
        tau = 0.5
        alpha_init = 1.0
        max_bt_its = 5
        eps = 1e-12

        for i in range(self.PP_opt_its):
            loss, grad = self.loss_grad_fn_pp(Z0, self.PP_opt)
            print(loss)

            H = self.hess_fn_pp(Z0, self.PP_opt)
            eigvals, eigvecs = jnp.linalg.eigh(H)

            eigvals_reg = jnp.abs(eigvals)
            mask = eigvals_reg > eps

            eigvals_keep = eigvals_reg[mask]
            Q = eigvecs[:, mask]

            # truncated spectral inverse
            H_inv = Q @ jnp.diag(1.0 / eigvals_keep) @ Q.T
            pk = -H_inv @ grad

            alpha = alpha_init
            gTp = jnp.dot(grad, pk)

            def trial_loss(alpha):
                PP_trial = self._wrap_pp_opt(self.PP_opt + alpha * pk)
                return self.PP_opt_loss_fn(Z0, PP_trial)

            loss_trial = trial_loss(alpha)

            bt_it = 0
            while loss_trial > loss + c1 * alpha * gTp:
                alpha *= tau
                bt_it += 1

                if bt_it >= max_bt_its:
                    alpha = 0.0
                    break

                loss_trial = trial_loss(alpha)

            if alpha == 0.0:
                print("Backtracking failed; terminating PP optimization.")
                break

            self.PP_opt = self._wrap_pp_opt(self.PP_opt + alpha * pk)

        self.state_opt.PP_opt = self.PP_opt

    def opt_loop(self, Z0, loss_fn_and_derivs, get_IC_state_fn, omega0_hat, attractor_rad):
        opt_data_all = None
        for i in range(self.opt_loops):
            do_Hg_int = i == 0
            do_last_it_logic = i == self.opt_loops - 1
            Z0_opt, opt_data = self.state_opt.opt_loop(Z0, loss_fn_and_derivs, get_IC_state_fn, omega0_hat, attractor_rad, do_Hg_int=do_Hg_int, do_last_it_logic=do_last_it_logic)
            if opt_data_all is None:
                opt_data_all = opt_data
            else:
                opt_data_all += opt_data
            Z0 = Z0_opt
            if i < self.opt_loops - 1:
                self.opt_pp(Z0_opt, get_IC_state_fn)

        return Z0_opt, opt_data_all

    def __repr__(self):
        return f"JO-{self.opt_loops}X-{self.state_opt}"

class BFGS(LS_TR_Opt):
    def __init__(self, ls, its, max_mem, eps_H, limited_memory=True, psuedo_proj=None, print_loss=False):
        super().__init__(its, psuedo_proj, print_loss)
        self.ls = ls
        self.ls_method = ls.name
        self.max_mem = max_mem
        self.eps_H = eps_H
        self.limited_memory = limited_memory
        if limited_memory:
            self.name = "L-BFGS"
        else:
            self.name = "BFGS"

    def init_opt_params(self, N):
        pass

    def Hvp_init(self, grad, Hg):
        # Initialize (L-)BFGS memory using curvature info
        if self.limited_memory:
            self.H = LBFGS_Update(grad.shape[0], self.max_mem, init_gamma=(1.0 / self.eps_H))
        else:
            self.H = BFGS_Update(grad.shape[0], init_gamma=(1.0 / self.eps_H))
        self.H.update(grad, Hg)


    def update_memory(self, Z, Z_next, grad, grad_next):
        sk = Z_next - Z
        yk = grad_next - grad
        return self.H.update(sk, yk)


    def inner_loop(self, Z0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, iter, last_iteration):
        loss_grad_cond_fn_base = loss_fn_and_derivs.conditional_loss_grad_fn
        loss_grad_cond_fn = lambda x1, x2: loss_grad_cond_fn_base(x1, x2, PP_opt=self.PP_opt)

        # 1) Choose search direction
        pk =  self.H.get_step_dir(grad)

        # 2) Line search / step
        alpha, Z_next, loss_next, grad_next, debug_str = self.ls_choice_logic(
            iter,
            loss,
            Z0,
            pk,
            grad,
            loss_grad_cond_fn,
            last_iteration,
        )

        alpha_pk = alpha * pk

        # 3) Early exit on final iteration (no update / no extra evals expected)
        if last_iteration:
            return Z_next, jnp.nan, jnp.nan, alpha, alpha_pk, ""

        # 4) Update quasi-Newton memory
        did_update = self.update_memory(Z0, Z_next, grad, grad_next)

        # 5) Debug string
        debug_out = f"{debug_str} | Update: {did_update}"

        return Z_next, loss_next, grad_next, alpha, alpha_pk, debug_out


