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
from SRC.DA_Comp.optimization.Quasi_Newton import L_SR1, HVP_Update, L_BK, LBFGS_Update
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import minres, cg


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


class L_BFGS(LS_TR_Opt):
    name = "LBFGS"
    def __init__(self, ls, its, max_mem, eps_H, print_loss=False):
        self.ls = ls
        self.ls_method = ls.name
        self.its = its
        self.print_loss = print_loss
        self.max_mem = max_mem
        self.H_init = None
        self.eps_H = eps_H

    def init_opt_params(self, N):
        self.H = self.H_init
        self.H_init = None

    def inner_loop(self, Z0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, iter, last_iteration):
        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_cond_fn = loss_fn_and_derivs.conditional_loss_grad_fn

        if self.H is None:
            Hg = loss_fn_and_derivs.HVP_fn(Z0, grad)
            gTHg = jnp.dot(grad, Hg)
            if gTHg > 0:
                print("diag init BFGS")
                self.H = LBFGS_Update(Z0.shape[0], self.max_mem, init_gamma=(1/self.eps_H))
                self.H.update(grad, Hg)
                pk = self.H.get_step_dir(grad)
            else:
                grad_norm = jnp.linalg.norm(grad)
                Rk = gTHg/grad_norm**2
                pk = (Rk/grad_norm) * grad
        else:
            pk = self.H.get_step_dir(grad)

        alpha, U_0_next, loss_next, grad_next, debug_str = self.ls_choice_logic(loss_fn, loss, Z0, pk, grad, loss_grad_cond_fn, last_iteration)
        alpha_pk = pk * alpha
        if last_iteration:
            return U_0_next, jnp.nan, jnp.nan, alpha, alpha_pk, ""
        else:
            if self.H is not None:
                sk = U_0_next - Z0
                yk = grad_next - grad
                did_update = self.H.update(sk, yk)
            return U_0_next, loss_next, grad_next, alpha, alpha_pk, debug_str + f" | Update: {did_update}"

class NCSR1(LS_TR_Opt, L_SR1, HVP_Update):
    def __init__(self, its, eps_H, max_memory,
                ls,

                SR1_type="conv",
                print_loss=False):
        LS_TR_Opt.__init__(self, its, print_loss)
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

    def second_order_logic(self, grad, hvp, Z0, iter):
        if len(self.Bk) == 0:
            print("init w/ HVP")
            self.Bk.eps = self.eps_H
            gTHg = jnp.dot(grad, hvp(Z0, grad))
            self.HVP_Bk_update(gTHg, grad.reshape((-1, 1)))

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
    def __init__(self, NCSR1_opt: NCSR1, BFGS_opt: L_BFGS, loops=1):
        self.NCSR1_opt = NCSR1_opt
        self.BFGS_opt = BFGS_opt

    def set_Bk_inv_for_BFGS(self):
        Bk_eigs, Bk_eig_vec = self.NCSR1_opt.Bk.eig_decomp(which="LA", num_eig=min(len(self.NCSR1_opt.Bk), self.BFGS_opt.max_mem))
        Bk_eigs = jnp.abs(Bk_eigs)
        H = LBFGS_Update(self.NCSR1_opt.Bk.N, self.BFGS_opt.max_mem, init_gamma=(self.NCSR1_opt.eps_H))
        for i in range(Bk_eigs.shape[0]):
            eig = Bk_eigs[i]
            vec = Bk_eig_vec[:, i]
            s = vec
            y = vec * eig
            H.update(s, y)
        self.BFGS_opt.H_init = H

    def opt_loop(self, U_0_DA_fourier, loss_fn_and_derivs):
        opt_data = None
        
        U_0_DA_fourier, opt_data = self.NCSR1_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs)
        #self.set_Bk_inv_for_BFGS()
        U_0_DA_fourier, opt_data_2 = self.BFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs)
        opt_data += opt_data_2

        return U_0_DA_fourier, opt_data
    
    def __repr__(self):
        name = f"{self.NCSR1_opt}__{self.BFGS_opt}"
        return name

