import jax.numpy as jnp
import jax
import optax
from jax import lax
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import eigsh
import random
from kf_da.opti.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from kf_da.opti.LS_TR import ArmijoLineSearch
from kf_da.opti.Quasi_Newton import L_SR1, HVP_Update, L_BK, LBFGS_Update, BFGS_Update
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import LinearOperator, minres
from kf_da.utils import bilinear_sample_periodic


def equal_component_Q(g, k, key=None):
    """
    Construct Q ∈ R^{k×n} with orthonormal rows such that:
      - q_i^T g = ||g|| / sqrt(k)   for all i
      - Q Q^T g = g
    
    Args:
        g:  (n,) array, nonzero vector in R^n
        k:  int, number of orthonormal vectors (1 <= k <= n)
        key: PRNGKey for randomization (optional)

    Returns:
        Q: (k, n) array, with rows q_i satisfying the above.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = g.shape[0]
    assert 1 <= k <= n, "Need 1 <= k <= n"

    g_norm = jnp.linalg.norm(g)
    g_hat = g / g_norm

    # Special case k = 1: just take q_1 = g_hat
    if k == 1:
        return g_hat[None, :]

    # -----------------------------
    # 1. Build U with orthonormal rows spanning a subspace containing g_hat
    #    First row is exactly g_hat, remaining k-1 rows random GS-orthonormalized.
    # -----------------------------
    key_U = key
    # random initial rows for i >= 1
    U_init = jax.random.normal(key_U, (k, n))
    U_init = U_init.at[0].set(g_hat)  # first row is g_hat

    def gs_rows(M):
        """Orthonormalize rows of M with Gram-Schmidt, keeping row 0 as is (unit)."""
        k, n = M.shape
        Q = jnp.zeros_like(M)
        # row 0 is already unit: g_hat
        Q = Q.at[0].set(M[0])

        def body_fun(i, Q):
            v = M[i]
            def proj_sub(j, v_inner):
                qj = Q[j]
                return v_inner - jnp.dot(qj, v_inner) * qj
            v = jax.lax.fori_loop(0, i, proj_sub, v)
            v = v / jnp.linalg.norm(v)
            Q = Q.at[i].set(v)
            return Q

        Q = jax.lax.fori_loop(1, k, body_fun, Q)
        return Q

    U = gs_rows(U_init)   # shape (k, n), rows orthonormal, first row = g_hat

    # -----------------------------
    # 2. Build R ∈ O(k) with first column v = (1/√k) * 1
    #    Use a Householder reflection mapping e1 -> v.
    # -----------------------------
    v = jnp.ones((k,)) / jnp.sqrt(k)
    e1 = jnp.zeros_like(v).at[0].set(1.0)

    # u = (e1 - v) / ||e1 - v|| ; H = I - 2 u u^T ; H e1 = v
    u = e1 - v
    u = u / jnp.linalg.norm(u)
    H = jnp.eye(k) - 2.0 * jnp.outer(u, u)  # orthogonal, H e1 = v
    R = H  # columns orthonormal, first column is v

    # -----------------------------
    # 3. Q = R U
    #    - rows of Q are orthonormal
    #    - Q g_hat = R e1 = v = (1/√k) 1
    # -----------------------------
    Q = R @ U  # shape (k, n)

    return Q

class Joint_Opt:
    def __init__(self, state_opt, PP_opt_its, opt_loops):
        self.state_opt = state_opt
        self.psuedo_proj = state_opt.psuedo_proj

        self.PP_opt_its = PP_opt_its
        self.opt_loops = opt_loops

    @staticmethod
    def periodic_diff(a, b):
        return (a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def set_pp_loss_fn(self, gen_loss_fn, PP_opt_default, pp_sigma, spatial_L, vel_trj_gen_fn, t_mask, part_arr_shape, dt):
        sigma_x, sigma_y = pp_sigma
        n_parts = PP_opt_default.shape[0]//2
        self.n_parts = n_parts
        xp_meas = PP_opt_default[:n_parts]
        yp_meas = PP_opt_default[n_parts:2*n_parts]
        self.xp_meas_flat = xp_meas
        self.yp_meas_flat = yp_meas
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y


        def PP_opt_loss_fn(Z0, PP_opt):
            PP_opt = jnp.mod(PP_opt, self.spatial_L)
            xp_opt = PP_opt[:n_parts]
            yp_opt = PP_opt[n_parts:2*n_parts]

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
        self.meas_indices = jnp.where(t_mask == True)[0]
        self.part_arr_shape = part_arr_shape
        self.dt = dt


    @staticmethod
    def integrate_particle(part_IC, u_chunk, v_chunk, dt):
        L = 2.0 * jnp.pi
        time_steps = u_chunk.shape[0]

        def body_fun(i, part_pos):
            u_grid = u_chunk[i]
            v_grid = v_chunk[i]

            u = bilinear_sample_periodic(u_grid, part_pos[0], part_pos[1], L, L)
            v = bilinear_sample_periodic(v_grid, part_pos[0], part_pos[1], L, L)

            part_pos = part_pos + dt * jnp.array([u, v])
            part_pos = jnp.mod(part_pos, L)
            return part_pos

        return lax.fori_loop(0, time_steps, body_fun, part_IC)


    def _build_all_track_segments(self, xp_all, yp_all, u_traj, v_traj):
        """
        xp_all, yp_all have shape (n_states, n_tracks)
        """
        idx_start = jnp.asarray(self.meas_indices[:-1])
        idx_end = jnp.asarray(self.meas_indices[1:])

        seg_len = int(idx_end[0] - idx_start[0] + 1)

        part_ics = jnp.stack([xp_all[:-1, :], yp_all[:-1, :]], axis=-1)  # (n_states-1, n_tracks, 2)

        offsets = jnp.arange(seg_len)[None, :]
        time_idx = idx_start[:, None] + offsets

        u_chunks = u_traj[time_idx, :, :]  # (n_states-1, seg_len, Nx, Ny)
        v_chunks = v_traj[time_idx, :, :]  # (n_states-1, seg_len, Nx, Ny)

        return part_ics, u_chunks, v_chunks


    def _all_track_model_data(self, xp_all, yp_all, u_traj, v_traj):
        """
        Returns arrays of shape (n_states, n_tracks)
        """
        part_ics, u_chunks, v_chunks = self._build_all_track_segments(
            xp_all, yp_all, u_traj, v_traj
        )

        integrate_one = lambda ic, uc, vc: self.integrate_particle(ic, uc, vc, self.dt)
        integrate_tracks = jax.vmap(integrate_one, in_axes=(0, None, None))
        integrate_segments = jax.vmap(integrate_tracks, in_axes=(0, 0, 0))

        jac_one = jax.jacrev(integrate_one, argnums=0)
        jac_tracks = jax.vmap(jac_one, in_axes=(0, None, None))
        jac_segments = jax.vmap(jac_tracks, in_axes=(0, 0, 0))

        final_pos = integrate_segments(part_ics, u_chunks, v_chunks)  # (n_states-1, n_tracks, 2)
        J = jac_segments(part_ics, u_chunks, v_chunks)                # (n_states-1, n_tracks, 2, 2)

        x_pred = final_pos[:, :, 0]
        y_pred = final_pos[:, :, 1]

        s_x = jnp.zeros_like(xp_all).at[1:, :].set(
            self.periodic_diff(xp_all[1:, :], x_pred)
        )
        s_y = jnp.zeros_like(yp_all).at[1:, :].set(
            self.periodic_diff(yp_all[1:, :], y_pred)
        )

        dxt_dxi = jnp.zeros_like(xp_all).at[:-1, :].set(J[:, :, 0, 0])
        dxt_dyi = jnp.zeros_like(xp_all).at[:-1, :].set(J[:, :, 0, 1])
        dyt_dxi = jnp.zeros_like(yp_all).at[:-1, :].set(J[:, :, 1, 0])
        dyt_dyi = jnp.zeros_like(yp_all).at[:-1, :].set(J[:, :, 1, 1])

        return s_x, s_y, dxt_dxi, dxt_dyi, dyt_dxi, dyt_dyi


    def _all_track_gradient(
        self,
        xp_all,
        yp_all,
        xp_meas_all,
        yp_meas_all,
        s_x,
        s_y,
        dxt_dxi,
        dxt_dyi,
        dyt_dxi,
        dyt_dyi,
    ):
        """
        All arrays shape (n_states, n_tracks)
        """
        n_states = xp_all.shape[0]
        sigx2 = self.sigma_x ** 2
        sigy2 = self.sigma_y ** 2

        r_x = self.periodic_diff(xp_all, xp_meas_all)
        r_y = self.periodic_diff(yp_all, yp_meas_all)

        grad_x = r_x / (n_states * sigx2)
        grad_y = r_y / (n_states * sigy2)

        grad_x = grad_x.at[1:, :].add(s_x[1:, :] / (n_states * sigx2))
        grad_y = grad_y.at[1:, :].add(s_y[1:, :] / (n_states * sigy2))

        grad_x = grad_x.at[:-1, :].add(
            -(s_x[1:, :] * dxt_dxi[:-1, :]) / (n_states * sigx2)
            -(s_y[1:, :] * dyt_dxi[:-1, :]) / (n_states * sigy2)
        )
        grad_y = grad_y.at[:-1, :].add(
            -(s_x[1:, :] * dxt_dyi[:-1, :]) / (n_states * sigx2)
            -(s_y[1:, :] * dyt_dyi[:-1, :]) / (n_states * sigy2)
        )

        return grad_x, grad_y, r_x, r_y


    def _all_track_hvp(
        self,
        v_all,
        dxt_dxi,
        dxt_dyi,
        dyt_dxi,
        dyt_dyi,
    ):
        """
        v_all has shape (2*n_states, n_tracks)
        derivative arrays have shape (n_states, n_tracks)
        """
        n_states, n_tracks = dxt_dxi.shape
        sigx2 = self.sigma_x ** 2
        sigy2 = self.sigma_y ** 2

        vx = v_all[:n_states, :]
        vy = v_all[n_states:, :]

        a = dxt_dxi
        b = dxt_dyi
        c = dyt_dxi
        d = dyt_dyi

        diag_x = jnp.full((n_states, n_tracks), 2.0 / (n_states * sigx2))
        diag_y = jnp.full((n_states, n_tracks), 2.0 / (n_states * sigy2))

        diag_x = diag_x.at[:-1, :].add(
            (a[:-1, :] ** 2) / (n_states * sigx2)
            + (c[:-1, :] ** 2) / (n_states * sigy2)
        )
        diag_y = diag_y.at[:-1, :].add(
            (b[:-1, :] ** 2) / (n_states * sigx2)
            + (d[:-1, :] ** 2) / (n_states * sigy2)
        )

        mixed = jnp.zeros((n_states, n_tracks))
        mixed = mixed.at[:-1, :].set(
            (a[:-1, :] * b[:-1, :]) / (n_states * sigx2)
            + (c[:-1, :] * d[:-1, :]) / (n_states * sigy2)
        )

        hx = diag_x * vx
        hx = hx.at[:-1, :].add(-(a[:-1, :] / (n_states * sigx2)) * vx[1:, :])
        hx = hx.at[1:, :].add(-(a[:-1, :] / (n_states * sigx2)) * vx[:-1, :])
        hx = hx + mixed * vy
        hx = hx.at[:-1, :].add(-(c[:-1, :] / (n_states * sigy2)) * vy[1:, :])
        hx = hx.at[1:, :].add(-(c[:-1, :] / (n_states * sigy2)) * vy[:-1, :])

        hy = diag_y * vy
        hy = hy.at[:-1, :].add(-(d[:-1, :] / (n_states * sigy2)) * vy[1:, :])
        hy = hy.at[1:, :].add(-(d[:-1, :] / (n_states * sigy2)) * vy[:-1, :])
        hy = hy + mixed * vx
        hy = hy.at[:-1, :].add(-(b[:-1, :] / (n_states * sigx2)) * vx[1:, :])
        hy = hy.at[1:, :].add(-(b[:-1, :] / (n_states * sigx2)) * vx[:-1, :])

        return jnp.concatenate([hx, hy], axis=0)


    def _apply_step_all_tracks(self, xp_all, yp_all, step_all, alpha):
        L = 2.0 * jnp.pi
        n_states = xp_all.shape[0]

        step_x = step_all[:n_states, :]
        step_y = step_all[n_states:, :]

        xp_new = jnp.mod(xp_all + alpha * step_x, L)
        yp_new = jnp.mod(yp_all + alpha * step_y, L)

        return xp_new, yp_new


    def _all_track_loss(self, xp_all, yp_all, xp_meas_all, yp_meas_all, u_traj, v_traj):
        n_states = xp_all.shape[0]
        sigx2 = self.sigma_x ** 2
        sigy2 = self.sigma_y ** 2

        r_x = self.periodic_diff(xp_all, xp_meas_all)
        r_y = self.periodic_diff(yp_all, yp_meas_all)

        s_x, s_y, _, _, _, _ = self._all_track_model_data(
            xp_all, yp_all, u_traj, v_traj
        )

        J_r = 0.5 / n_states * (
            jnp.sum(r_x ** 2) / sigx2 + jnp.sum(r_y ** 2) / sigy2
        )

        J_m = 0.5 / n_states * (
            jnp.sum(s_x[1:, :] ** 2) / sigx2 + jnp.sum(s_y[1:, :] ** 2) / sigy2
        )

        return J_r + J_m


    def opt_pp_bu(self, Z0, get_IC_state_fn):
        """
        Batched particle-position optimization using:
        - manual gradient
        - manual Hessian-vector product
        - one global CG solve
        - one global backtracking line search

        Updates self.PP_opt in place and returns nothing.
        """
        n_total = self.n_parts                    # total x entries, same as total y entries
        n_states, n_tracks = self.part_arr_shape  # e.g. (7, 45)

        c1 = 1e-4
        tau = 0.5
        alpha_init = 1.0
        max_bt_its = 10
        newton_its = 10
        damping = 1e-6

        omega0_DA_hat = get_IC_state_fn(Z0)
        u_traj, v_traj = self.vel_trj_gen_fn(omega0_DA_hat)

        xp_opt = self.PP_opt[:n_total].reshape(self.part_arr_shape)
        yp_opt = self.PP_opt[n_total:2 * n_total].reshape(self.part_arr_shape)
        xp_meas = self.xp_meas_flat.reshape(self.part_arr_shape)
        yp_meas = self.yp_meas_flat.reshape(self.part_arr_shape)

        print("Starting batched particle-position optimization")
        print(
            f"n_total={n_total}, n_states={n_states}, "
            f"n_tracks={n_tracks}, newton_its={newton_its}"
        )

        for k in range(newton_its):
            J0 = self._all_track_loss(
                xp_opt, yp_opt, xp_meas, yp_meas, u_traj, v_traj
            )

            s_x, s_y, dxt_dxi, dxt_dyi, dyt_dxi, dyt_dyi = self._all_track_model_data(
                xp_opt, yp_opt, u_traj, v_traj
            )

            grad_x, grad_y, _, _ = self._all_track_gradient(
                xp_opt,
                yp_opt,
                xp_meas,
                yp_meas,
                s_x,
                s_y,
                dxt_dxi,
                dxt_dyi,
                dyt_dxi,
                dyt_dyi,
            )

            grad_all = jnp.concatenate([grad_x, grad_y], axis=0)   # (2*n_states, n_tracks)
            grad_vec = np.asarray(grad_all.reshape(-1))
            grad_norm = float(jnp.linalg.norm(grad_all))

            print(f"iter={k:02d}, loss={float(J0):.3e}, grad_norm={grad_norm:.3e}")

            n_dof = grad_vec.size  # should be 2 * n_states * n_tracks

            hvp_fn = jax.jit(
                lambda v_flat: self._all_track_hvp(
                    v_flat.reshape(2 * n_states, n_tracks),
                    dxt_dxi,
                    dxt_dyi,
                    dyt_dxi,
                    dyt_dyi,
                ).reshape(-1)
            )

            def matvec(v_np):
                v_jax = jnp.asarray(v_np).reshape(-1)
                Hv_jax = hvp_fn(v_jax) + damping * v_jax
                return np.asarray(Hv_jax)

            H_linop = LinearOperator(
                shape=(n_dof, n_dof),
                matvec=matvec,
                dtype=np.float64,
            )

            rhs = -grad_vec
            step_np, info = scipy_cg(
                H_linop,
                rhs,
            )

            step_all = jnp.asarray(step_np).reshape(2 * n_states, n_tracks)

            gTp = float(jnp.dot(grad_all.reshape(-1), step_all.reshape(-1)))
            if (not np.isfinite(gTp)) or (gTp >= 0.0):
                print(f"iter={k:02d}: CG step not descent, using steepest descent")
                step_all = -grad_all
                gTp = float(jnp.dot(grad_all.reshape(-1), step_all.reshape(-1)))

            alpha = alpha_init
            accepted = False

            for bt_it in range(max_bt_its):
                xp_trial, yp_trial = self._apply_step_all_tracks(
                    xp_opt, yp_opt, step_all, alpha
                )

                J_trial = self._all_track_loss(
                    xp_trial, yp_trial, xp_meas, yp_meas, u_traj, v_traj
                )

                if J_trial <= J0 + c1 * alpha * gTp:
                    accepted = True
                    print(
                        f"iter={k:02d}: cg_info={info}, bt_it={bt_it}, "
                        f"alpha={alpha:.3e}, new_loss={float(J_trial):.6e}"
                    )
                    break

                alpha *= tau

            if not accepted:
                print(f"iter={k:02d}: line search failed, skipping update")
                break

            xp_opt = xp_trial
            yp_opt = yp_trial

        self.PP_opt = jnp.concatenate([
            xp_opt.reshape(-1),
            yp_opt.reshape(-1),
        ])

        print("Finished batched particle-position optimization")

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
                PP_trial = jnp.mod(self.PP_opt + alpha * pk, self.spatial_L)
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

            self.PP_opt = jnp.mod(self.PP_opt + alpha * pk, self.spatial_L)

        self.state_opt.PP_opt = self.PP_opt

    def opt_loop(self, Z0, loss_fn_and_derivs, get_IC_state_fn, omega0_hat, attractor_rad):
        opt_data_all = None
        for i in range(self.opt_loops):
            do_Hg_int = i==0
            do_last_it_logic = i==self.opt_loops-1
            Z0_opt, opt_data = self.state_opt.opt_loop(Z0, loss_fn_and_derivs, get_IC_state_fn, omega0_hat, attractor_rad, do_Hg_int=do_Hg_int, do_last_it_logic=do_last_it_logic)
            if opt_data_all is None:
                opt_data_all = opt_data
            else:
                opt_data_all += opt_data
            Z0 = Z0_opt
            if i < self.opt_loops-1:
                self.opt_pp(Z0_opt, get_IC_state_fn)

        return Z0_opt, opt_data_all
    
    def __repr__(self):
        return f"JO-{self.opt_loops}X-{self.state_opt}"

class TN(LS_TR_Opt):
    def __init__(self, ls, its, psuedo_proj=None, print_loss=False):
        super().__init__(its, psuedo_proj, print_loss)
        self.ls = ls
        self.ls_method = ls.name
        self.name = "TN"

    def init_opt_params(self, N):
        pass

    def Hvp_init(self, grad, Hg):
        pass

    def inner_loop(self, Z0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, iter, last_iteration):
        loss_grad_cond_fn = loss_fn_and_derivs.conditional_loss_grad_fn

        #Hv = loss_fn_and_derivs.HVP_fn(Z0, v)
        damping = 1e-6
        def matvec(v):
            Hv = loss_fn_and_derivs.HVP_fn(Z0, v.reshape(grad.shape))
            Hv = np.asarray(Hv, dtype=float).ravel()
            return Hv + damping * v
        n = Z0.shape[0]
        A = LinearOperator((n, n), matvec=matvec, dtype=float)
        maxiter = 20
        pk, info = minres(A, -grad, maxiter=maxiter)
        print(info)

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


        return Z_next, loss_next, grad_next, alpha, alpha_pk, debug_str

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

class NCSR1(LS_TR_Opt, L_SR1, HVP_Update):
    def __init__(self, its, eps_H, max_memory,
                ls,
                SR1_type="conv",
                psuedo_proj=None,
                print_loss=False):
        LS_TR_Opt.__init__(self, its, psuedo_proj, print_loss)
        self.set_SR1_update_type(SR1_type)
        self.name = "NCNSR1"
        self.ls_method = ls.name
        if SR1_type == "mod":
            self.name += "M"
        
        self.eps_H = eps_H
        self._max_memory = max_memory
        self.ls = ls
        

    def NCN_dir(self, Lam, Q, grad):
        Lam_reg = jnp.abs(Lam)
        Lam_reg = jnp.where(Lam_reg > self.eps_H, Lam_reg, self.eps_H)
        null_space_comp = (-grad) - Q @ (Q.T @ -grad)
        pk = Q @ ((Q.T @ -grad) / Lam_reg) + (1.0 / self.eps_H) * null_space_comp
        return pk

    def Hvp_init(self, grad, Hg):
        print("init w/ HVP")
        self.Bk.eps = self.eps_H
        gTHg = jnp.dot(grad, Hg)
        self.HVP_Bk_update(gTHg, grad.reshape((-1, 1)))

    def second_order_logic(self, grad, hvp, Z0, iter):
        if False:
            for i in range(qTHq.shape[0]):
                xi = Q[:, i]
                t = jnp.dot(xi, self.Bk @ xi)
                print(t, qTHq[i])
            print("----")

        Bk_eigs, Bk_eig_vec = self.Bk.eig_decomp(which="LM", num_eig=len(self.Bk))
        pk = self.NCN_dir(Bk_eigs, Bk_eig_vec, grad)
        if jnp.any(jnp.isnan(pk)):
            pk = -grad
        step_type = "NCN"

        return pk, step_type


    @staticmethod
    def check_eig(hvp, x, v):
        vTHv = jnp.dot(v, hvp(x, v))
        return vTHv/jnp.linalg.norm(v)
    
    def init_opt_params(self, N):
        self.init_Bk = False
        self.Bk = L_BK(self._max_memory, N)
        self.current_memory = 0

    def inner_loop(self, Z0, grad, loss, loss_fn_and_derivs, iter, last_iteration):
        loss_fn = loss_fn_and_derivs.loss_fn
        #loss_grad_fn = loss_fn_and_derivs.loss_grad_fn
        hvp = loss_fn_and_derivs.HVP_fn
        loss_grad_cond_fn = loss_fn_and_derivs.conditional_loss_grad_fn


        pk, step_type = self.second_order_logic(grad, hvp, Z0, iter)
        alpha, U_0_next, loss_next, grad_next, debug_str = self.ls_choice_logic(loss_fn, loss, Z0, pk, grad, loss_grad_cond_fn, last_iteration)

        alpha_pk = pk * alpha
        debug_str = f"step type: {step_type} | " + debug_str
        
        if last_iteration:
            return U_0_next, np.nan, np.nan, alpha, alpha_pk, debug_str
        else:
            self.SR1_update(U_0_next, Z0, grad_next, grad, loss_next, loss)
            return U_0_next, loss_next, grad_next, alpha, alpha_pk, debug_str

class NCSR1_and_LBFGS:
    def __init__(self, NCSR1_opt: NCSR1, BFGS_opt: BFGS):
        self.NCSR1_opt = NCSR1_opt
        self.BFGS_opt = BFGS_opt
        self.psuedo_proj = self.BFGS_opt.psuedo_proj

    # Z0, loss_fn_and_derivs: Loss_and_Deriv_fns, inv_transform, omega0_hat_trg, attractor_rad
    def opt_loop(self, Z0, loss_fn_and_derivs, inv_transform, omega0_hat_trg, attractor_rad):
        loss, grad = loss_fn_and_derivs.loss_grad_fn(Z0)
        Hg = loss_fn_and_derivs.HVP_fn(Z0, grad)
        gTHg = jnp.dot(grad, Hg)
        print(f"Initial gTHg: {gTHg/jnp.linalg.norm(grad)**2}")
        if gTHg < 0:
            print("NCNSR1 Init")
            Z0, opt_data = self.NCSR1_opt.opt_loop(Z0, loss_fn_and_derivs, inv_transform, omega0_hat_trg, attractor_rad, init_loss=loss, init_grad=grad, init_Hg=Hg)
        Z0, opt_data_2 = self.BFGS_opt.opt_loop(Z0, loss_fn_and_derivs, inv_transform, omega0_hat_trg, attractor_rad)
        opt_data += opt_data_2

        return Z0, opt_data
    
    def __repr__(self):
        name = f"{self.NCSR1_opt}__{self.BFGS_opt}"
        return name

