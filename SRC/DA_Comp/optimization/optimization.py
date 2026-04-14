import jax.numpy as jnp
import jax
import optax
from jax import lax
from SRC.iterative_methods import lanczos_eigs, pcg_curve_detection
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import eigsh
import random
from SRC.DA_Comp.optimization.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch
from SRC.DA_Comp.optimization.Quasi_Newton import L_SR1, HVP_Update, L_BK, LBFGS_Update, BFGS_Update
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import minres, cg
from scipy.sparse.linalg import LinearOperator, minres
from jax.scipy.sparse.linalg import cg

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

    def set_pp_loss_fn(self, gen_loss_fn, PP_opt_default, pp_sigma, spatial_L):
        sigma_x, sigma_y = pp_sigma
        n_parts = PP_opt_default.shape[0]//2
        xp_meas = PP_opt_default[:n_parts]
        yp_meas = PP_opt_default[n_parts:2*n_parts]

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

    def opt_pp_bu(self, Z0):
        alpha = 1.0

        for i in range(self.PP_opt_its):
            loss, grad = self.loss_grad_fn_pp(Z0, self.PP_opt)
            print(loss)
            H = self.hess_fn_pp(Z0, self.PP_opt)
            eigvals, eigvecs = jnp.linalg.eigh(H)
            eigvals = jnp.abs(eigvals)
            H_inv = eigvecs @ jnp.diag(1/eigvals) @ eigvecs.T
            pk = -H_inv @ grad


            self.PP_opt = jnp.mod(self.PP_opt + alpha * pk, self.spatial_L)
        self.state_opt.PP_opt = self.PP_opt

    def opt_pp(self, Z0):
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
                self.opt_pp(Z0_opt)

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

