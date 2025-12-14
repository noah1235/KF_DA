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
from SRC.DA_Comp.optimization.LS_TR import Cubic_TR
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch, Cubic_TR
from SRC.DA_Comp.optimization.Quasi_Newton import L_SR1, HVP_Update, L_BK, BFGS_Update
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import minres, cg

def to_writable_np(x, dtype=np.float64):
    # Ensures a real NumPy array with writeable memory
    arr = np.asarray(x, dtype=dtype)
    if not arr.flags.writeable:
        arr = arr.copy()
    else:
        # still a good idea to avoid weird views
        arr = arr.copy()
    return arr

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


class BFGS(LS_TR_Opt, BFGS_Update):
    name = "BFGS"
    def __init__(self, ls, its, print_loss=False):
        self.ls = ls
        self.ls_method = ls.name
        self.its = its
        self.print_loss = print_loss
        self.Bk_inv_init = None

    def init_opt_params(self, N):
        if isinstance(self.ls, Cubic_TR):
            self.ls.init_opt()
        self.Bk_inv = self.Bk_inv_init
        self.Bk_inv_init = None

    def set_alpha_and_return(self, loss_fn, loss, U_0, pk, grad, div_free_proj, last_iteration, loss_grad_fn, eps=1e-12):
        def jit_block(U_0_next):
            # Step and new loss/grad
            loss_next, grad_next = loss_grad_fn(U_0_next)

            # Quasi-Newton pair
            sk = U_0_next - U_0
            yk = grad_next - grad

            # Curvature
            ys = jnp.vdot(yk, sk).real
            return loss_next, grad_next, ys, sk, yk


        if isinstance(self.ls, ArmijoLineSearch):
            alpha = self.ls(loss_fn, loss, U_0, pk, grad)
            debug_str = ""
        elif isinstance(self.ls, Cubic_TR):
            pTHp = jnp.dot(pk, -grad)
            alpha = self.ls.get_alpha(pk, grad, pTHp, loss_fn, U_0, loss)
            debug_str = f"eta: {self.ls.eta}"

        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        alpha_pk = pk * alpha  # return value
        if last_iteration:
            return U_0_next, jnp.nan, jnp.nan, alpha, alpha_pk, ""
        else:
            loss_next, grad_next, ys, sk, yk = jit_block(U_0_next)

            if ys > eps:
                self.Bk_inv_update(ys, sk, yk)
                return U_0_next, loss_next, grad_next, alpha, alpha_pk, debug_str + f" | Update: {True}"

            else:
                return U_0_next, loss_next, grad_next, alpha, alpha_pk, debug_str + f" | Update: {False}"

    def inner_loop(self, U_0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, div_free_proj, iter, last_iteration, eps=1e-12):
        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_fn = loss_fn_and_derivs.loss_grad_adj_fn

        if iter == 0 and self.Bk_inv is None:
            gTHg = jnp.dot(grad, loss_fn_and_derivs.Hvp_adj_fn(grad.reshape((-1, 1))))
            curvature = gTHg / jnp.linalg.norm(grad)**2
            self.Bk_inv = jnp.eye(grad.shape[0]) / jnp.abs(curvature)


        # Search direction and line search
        pk = -self.Bk_inv @ grad
        return self.set_alpha_and_return(loss_fn, loss, U_0, pk, grad, div_free_proj, last_iteration, loss_grad_fn, eps=1e-12)


class PCGBFGS(BFGS):
    name = "PCGBFGS"
    def __init__(self, ls, its, n_hvp, print_loss=False):
        super().__init__(ls, its, print_loss)
        self.n_hvp = n_hvp
        #end it, method, eps
        self.sch_idx = 0
        self.schedule = [
            (its, "MINRES", 1e-8),


        ]


    def inner_loop(self, U_0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, div_free_proj, iter, last_iteration):
        opt_params = self.schedule[self.sch_idx]
        if opt_params[0] - 1 == iter:
             self.sch_idx += 1
             opt_params = self.schedule[self.sch_idx]
        _, method, eps = opt_params

        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_fn = loss_fn_and_derivs.loss_grad_adj_fn
        hvp_fn = loss_fn_and_derivs.Hvp_adj_fn
        if iter == 0 and self.Bk_inv is None:
            gTHg = jnp.dot(grad, loss_fn_and_derivs.Hvp_adj_fn(grad.reshape((-1, 1))))
            curvature = gTHg / jnp.linalg.norm(grad)**2
            self.Bk_inv = jnp.eye(grad.shape[0]) / jnp.abs(curvature)

        S = []
        Y = []
        def matvec_base(v):
            return hvp_fn(v.reshape((-1, 1))).squeeze()

        def Hvp_record_fn(v):
            v_jax = jnp.array(v, dtype=jnp.float64)
            Hv = matvec_base(v_jax) + v_jax*eps
            if jnp.dot(v_jax, Hv) > 0:
                S.append(v_jax)
                Y.append(Hv)
                
            return Hv
        
        M = np.array(self.Bk_inv)
        Aop = LinearOperator((U_0.shape[0], U_0.shape[0]), matvec=Hvp_record_fn)
        if method == "MINRES":
            pk, info = minres(Aop, -grad, M=M, maxiter=self.n_hvp)
        else:
            pk, info = cg(Aop, -grad, M=M, maxiter=self.n_hvp)

        added_p_hvp = len(S) > 0
        if added_p_hvp:
            S = jnp.vstack(S).T
            Y = jnp.vstack(Y).T
            for i in range(S.shape[1]):
                s = S[:, i]
                y = Y[:, i]
                ys = jnp.dot(y, s)
                if ys > 1e-12:
                    self.Bk_inv_update(ys, s, y)

        #debug
        if True:
            Hpk = matvec_base(pk)
            error = jnp.dot(Hpk, -grad) / (jnp.linalg.norm(Hpk) * jnp.linalg.norm(grad))

            def Hvp_record_fn(v):
                v_jax = jnp.array(v, dtype=jnp.float64)
                Hv = matvec_base(v_jax)                
                return np.array(Hv) + v*eps
            Aop = LinearOperator((U_0.shape[0], U_0.shape[0]), matvec=Hvp_record_fn)
            pk2, info = minres(Aop, -grad, maxiter=self.n_hvp)
            Hpk2 = matvec_base(pk2)
            error2 = jnp.dot(Hpk2, -grad) / (jnp.linalg.norm(Hpk2) * jnp.linalg.norm(grad))
            print(f"{method} error: {error} | error no precond: {error2}")

        return self.set_alpha_and_return(loss_fn, loss, U_0, pk, grad, div_free_proj, last_iteration, loss_grad_fn, eps=1e-12)


class NCSR1(LS_TR_Opt, L_SR1, HVP_Update):
    def __init__(self, its, eps_H, max_memory,
                ls,
                num_batch_hvp=5,
                num_power_iters=2,
                SR1_type="conv",
                print_loss=False):
        LS_TR_Opt.__init__(self, its, print_loss)
        self.set_SR1_update_type(SR1_type)
        self.name = "NCSR1"
        self.ls_method = ls.name
        if SR1_type == "mod":
            self.name += "M"

        self.eps_H = eps_H
        self._max_memory = max_memory
        self.ls = ls
        
        self.num_batch_hvp = num_batch_hvp
        self.num_power_iters = num_power_iters


    def second_order_logic(self, grad, hvp, U_0, div_free_proj, iter):
        v_list = []
        vTHv_list = []
        eps = 0

        if iter != 0:
            Q = equal_component_Q(grad, 5, key=jax.random.PRNGKey(iter)).T
            HQ = hvp(Q)
            Lam = jnp.sum(Q * HQ, axis=0)
            #Lam, Q = self.Bk.eig_decomp()
            min_eig = 1e-10
            #print(Lam)
            #Lam = jnp.abs(Lam)
            #mask = Lam > min_eig
            #Lam = Lam[mask]
            #Q = Q[:, mask]
            Bk_inv = Q @ jnp.diag(1/(eps + Lam)) @ Q.T
            def precond(v):
                null_comp = v - Q @ Q.T @ v
                return Bk_inv @ v + (1/min_eig) * null_comp
            M = LinearOperator((U_0.shape[0], U_0.shape[0]), matvec=precond)

        print(f"eps: {eps}")
        def Hvp_record_fn(v):
            v = jnp.array(v, dtype=jnp.float64)
            Hv = hvp(v.reshape((-1, 1))).squeeze()
            vTHv = jnp.dot(v, Hv)
            if jnp.abs(vTHv) > 1e-8:
                v_list.append(v)
                vTHv_list.append(vTHv)
            return np.array(Hv) + np.array(v) * eps
        
        Aop = LinearOperator((U_0.shape[0], U_0.shape[0]), matvec=Hvp_record_fn)
        pk, info = minres(Aop, -grad, maxiter=3)
        v_list = jnp.vstack(v_list).T
        vTHv_list = jnp.array(vTHv_list)
        self.HVP_Bk_update(vTHv_list, v_list)

        if iter != 0:
            def Hvp_record_fn(v):
                v = jnp.array(v, dtype=jnp.float64)
                Hv = hvp(v.reshape((-1, 1))).squeeze()
                return np.array(Hv) + np.array(v) * eps
            Aop = LinearOperator((U_0.shape[0], U_0.shape[0]), matvec=Hvp_record_fn)
            pk2, info = minres(Aop, -grad, M=M, maxiter=3)
            Hpk = hvp(pk.reshape((-1, 1))).squeeze() + pk*eps
            Hpk2 = hvp(pk2.reshape((-1, 1))).squeeze() + pk2*eps

            error = jnp.dot(Hpk, -grad) / (jnp.linalg.norm(Hpk) * jnp.linalg.norm(grad))
            error_2 = jnp.dot(Hpk2, -grad) / (jnp.linalg.norm(Hpk2) * jnp.linalg.norm(grad))
            print(f"error: {error} | error precond: {error_2}")

        return pk, "MINRES"


    def second_order_logic_dec(self, grad, hvp, U_0, div_free_proj, iter):
        
        if len(self.Bk) == 0:
            Q = equal_component_Q(grad, self.num_batch_hvp, key=jax.random.PRNGKey(iter)).T
        else:
            _, Q = self.Bk.eig_decomp(which="LM", num_eig=self.num_batch_hvp)
            g_proj = Q @ (Q.T @ grad)
            r = grad - g_proj
            u = r / jnp.linalg.norm(r)
            Q = jnp.concat([Q, u.reshape(-1, 1)], axis=1)
            #Q = (grad/jnp.linalg.norm(grad)).reshape((-1, 1))


        HQ = hvp(Q)
        qTHq = jnp.sum(Q * HQ, axis=0)
        self.HVP_Bk_update(qTHq, Q)
        if False:
            for i in range(qTHq.shape[0]):
                xi = Q[:, i]
                t = jnp.dot(xi, self.Bk @ xi)
                print(t, qTHq[i])
            print("----")

            

        NCN_min_eig = self.eps_H
        Bk_eigs, Bk_eig_vec = self.Bk.eig_decomp(which="LM", num_eig=len(self.Bk))
        if False and jnp.min(Bk_eigs) < -self.eps_H:
            step_type = "neg curve"
            pk = Bk_eig_vec[:, jnp.argmin(Bk_eigs)]
            if jnp.dot(pk, grad) > 0:
                pk = -pk
        else:
            Lam_reg = jnp.abs(Bk_eigs)
            Q = Bk_eig_vec
            Lam_reg = jnp.where(Lam_reg > NCN_min_eig, Lam_reg, NCN_min_eig)

            null_space_comp = (-grad) - Q @ (Q.T @ -grad)
            pk = Q @ ((Q.T @ -grad) / Lam_reg) + (1.0 / self.eps_H) * null_space_comp

            pk = div_free_proj(pk)

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
        if isinstance(self.ls, Cubic_TR):
            self.ls.init_opt()

    def inner_loop(self, U_0, grad, loss, loss_fn_and_derivs, div_free_proj, iter, last_iteration):
        N = U_0.shape[0]

        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_fn = loss_fn_and_derivs.loss_grad_adj_fn
        hvp = loss_fn_and_derivs.Hvp_adj_fn

        pk, step_type = self.second_order_logic(grad, hvp, U_0, div_free_proj, iter)
        if isinstance(self.ls, ArmijoLineSearch):
            alpha = self.ls(loss_fn, loss, U_0, pk, grad)
            debug_str = ""
        elif isinstance(self.ls, Cubic_TR):
            pTHp = jnp.dot(pk, -grad)
            alpha = self.ls.get_alpha(pk, grad, pTHp, loss_fn, U_0, loss)
            debug_str = f"eta: {self.ls.eta}"
        alpha_pk = pk * alpha

        # Step and new loss/grad
        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        debug_str = f"step type: {step_type} | " + debug_str
        
        if last_iteration:
            return U_0_next, np.nan, np.nan, alpha, alpha_pk, debug_str
        else:
            loss_next, grad_next = loss_grad_fn(U_0_next)
            self.SR1_update(U_0_next, U_0, grad_next, grad, loss_next, loss)

            return U_0_next, loss_next, grad_next, alpha, alpha_pk, debug_str

class BFGS_2_PCGBFGS:
    def __init__(self, BFGS_opt: BFGS, PCGBFGS_opt: PCGBFGS):
        self.BFGS_opt = BFGS_opt
        self.PCGBFGS_opt = PCGBFGS_opt
    
    def opt_loop(self, U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj):
        U_0_DA_fourier, opt_data_1 = self.BFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)
        self.PCGBFGS_opt.Bk_inv_init = self.BFGS_opt.Bk_inv
        U_0_DA_fourier, opt_data_2 = self.PCGBFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)

        return opt_data_1 + opt_data_2

class NCSR1_and_BFGS:
    def __init__(self, NCSR1_opt: NCSR1, BFGS_opt: BFGS, loops=1):
        self.NCSR1_opt = NCSR1_opt
        self.BFGS_opt = BFGS_opt
        self.loops = loops

    def set_Bk_inv_for_BFGS(self):
        Bk_eigs, Bk_eig_vec = self.NCSR1_opt.Bk.eig_decomp(which="LM")
        n = self.NCSR1_opt.Bk.N        
        k = Bk_eig_vec.shape[1]
        min_eig = 1e-6
        Bk_eigs_clipped = np.maximum(Bk_eigs, min_eig)
        pad = np.full(n - k, min_eig, dtype=Bk_eigs.dtype)
        Bk_eigs_full = np.concatenate([Bk_eigs_clipped, pad], axis=0)
        U, _, _ = np.linalg.svd(Bk_eig_vec, full_matrices=True)
        Q_perp = U[:, k:]
        Bk_eig_vec_full = np.concatenate([Bk_eig_vec, Q_perp], axis=1)

        Bk_inv = (Bk_eig_vec_full * (1/Bk_eigs_full)) @ Bk_eig_vec_full.T
        Bk_inv = 0.5 * (Bk_inv + Bk_inv.T)
        Bk_inv = jnp.array(Bk_inv)
        self.BFGS_opt.Bk_inv_init = Bk_inv

    def set_Bk_for_SR1(self):
        n = self.NCSR1_opt._max_memory
        Bk_inv = self.BFGS_opt.Bk_inv
        eigs, eig_vecs = jnp.linalg.eigh(Bk_inv)
        idx = jnp.argsort(eigs, descending=False)[:n][::-1]
        scalars = 1/eigs[idx]
        vecs = eig_vecs[:, idx]
        self.NCSR1_opt.Bk.set_Bk(vecs, scalars)


    def opt_loop(self, U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj):
        opt_data = None
        
        for i in range(self.loops):
            U_0_DA_fourier, opt_data_1 = self.NCSR1_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)
            self.set_Bk_inv_for_BFGS()
            U_0_DA_fourier, opt_data_2 = self.BFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)
            #self.set_Bk_for_SR1()
            if opt_data is None:
                opt_data = opt_data_1 + opt_data_2
            else:
                opt_data += opt_data_1 + opt_data_2

        return U_0_DA_fourier, opt_data
    
    def __repr__(self):
        name = f"{self.NCSR1_opt}__{self.BFGS_opt}__L={self.loops}"
        return name

