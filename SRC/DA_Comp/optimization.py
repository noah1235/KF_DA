import jax.numpy as jnp
import jax
import optax
import numpy as np
from jax import lax
from SRC.iterative_methods import lanczos_tridiagonal, lanczos_2_eig_decomp, lanczos_eigs
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import eigsh
import random

class Optimizer:
    name = ""
    ls_method = ""
    its = -1

    def __repr__(self):
        return f"{self.name}_{self.ls_method}-{self.its}"
    
class LS_Opt(Optimizer):
    def step(
        self,
        x: jnp.ndarray,      # current x, shape (m,)
        alpha: float,
        p: jnp.ndarray       # search direction, shape (m,)
    ) -> jnp.ndarray:
        return x + alpha * p
    
class Opt_Data:
    def __init__(self, its):
        self.loss_record = np.zeros(its)
        self.grad_norm_record = np.zeros(its)
        self.alpha_gTp_record = np.zeros(its)
    
    def __call__(self, n, loss, grad, alpha_p):
        self.loss_record[n] = loss
        self.grad_norm_record[n] = jnp.linalg.norm(grad)
        self.alpha_gTp_record[n] = jnp.dot(alpha_p, grad)

    def early_stop_update(self, iters):
        self.loss_record = self.loss_record[:iters]
        self.grad_norm_record = self.grad_norm_record[:iters]
        self.alpha_gTp_record = self.alpha_gTp_record[:iters]

class LBFGS(Optimizer):
    name = "L-BFGS"
    def __init__(self, its):
        self.its = its
        self.ls_method = "BT"
        self.alg = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=20))
    
class ADAM(Optimizer):
    name = "ADAM"
    def __init__(self, lr, its):
        self.its = its
        self.alg = optax.adam(learning_rate=lr)
        
    def __repr__(self):
        return f"{self.name}-{self.its}"

class BFGS(LS_Opt):
    name = "BFGS"
    def __init__(self, ls, its, fallback_opt, print_loss=False):
        self.ls = ls
        self.ls_method = ls.name
        self.its = its
        self.print_loss = print_loss

        if fallback_opt == "eye":
            self.fallback = self.eye_fallback
    @staticmethod
    def eye_fallback(N):
        return jnp.eye(N)
    
    @staticmethod
    @jax.jit
    def Bk_inv_update(ys, sk, yk, Bk_inv):
        I = jnp.eye(yk.shape[0], dtype=Bk_inv.dtype)
        rho = 1.0 / ys
        Sy = jnp.outer(sk, yk)
        Ys = jnp.outer(yk, sk)
        Ss = jnp.outer(sk, sk)
        Bk_inv_next = (I - rho * Sy) @ Bk_inv @ (I - rho * Ys) + rho * Ss
        return Bk_inv_next

    def opt_loop(self, U_0, loss_fn_base, loss_grad_fn, div_check):
        N = U_0.shape[0]
        loss_fn = jax.jit(loss_fn_base)

        @jax.jit
        def jit_block(U_0_next):
            # Step and new loss/grad
            loss_next, grad_next = loss_grad_fn(U_0_next)

            # Quasi-Newton pair
            sk = U_0_next - U_0
            yk = grad_next - grad

            # Curvature
            ys = jnp.vdot(yk, sk).real
            return loss_next, grad_next, ys, sk, yk
        


        def inner_loop(U_0, grad, Bk_inv, eps=1e-8):
            # Search direction and line search
            pk = -Bk_inv @ grad
            alpha = self.ls(loss_fn, U_0, pk, grad)
            U_0_next = self.step(U_0, alpha, pk)
            alpha_pk = pk * alpha  # return value
            loss_next, grad_next, ys, sk, yk = jit_block(U_0_next)

            if ys > eps:
                Bk_inv_next = self.Bk_inv_update(ys, sk, yk, Bk_inv)
                return U_0_next, loss_next, grad_next, Bk_inv_next, alpha, alpha_pk, True

            else:
                return U_0_next, loss_next, grad_next, Bk_inv, alpha, alpha_pk, False


        
        Bk_inv = self.fallback(N)

        opt_data = Opt_Data(self.its)
        for i in range(self.its):
            if i == 0:
                loss, grad = loss_grad_fn(U_0)
            loss_prev = loss
            grad_prev = grad
            U_0, loss, grad, Bk_inv, alpha, alpha_pk, Bk_inv_update = inner_loop(U_0, grad, Bk_inv)
            opt_data(i, loss_prev, grad_prev, alpha_pk)
            if self.print_loss:
                print(f"i:{i} | loss: {loss} | Div: {div_check(U_0)} | alpha: {alpha} | Bk_inv_update: {Bk_inv_update}")

            if alpha <= self.ls.min_alpha:
                if self.print_loss:
                    print(f"optimizer stalled | alpha={alpha}")

                opt_data.early_stop_update(i)
                break


        return U_0, opt_data, i

class DA_SR1(LS_Opt):
    name = "DA_SR1"
    def __init__(self, its, fallback_opt, eps_g, eps_H, 
                 max_memory, NCN_kappa,
                 eta_0=1e4,
                 print_loss=False):
        self.its = its
        self.print_loss = print_loss
        self.eps_g = eps_g
        self.eps_H = eps_H
        self.max_memory = max_memory
        self.eta = eta_0
        self.eta_0 = eta_0
        self.rho = 5
        self.eta_min = 1e-3
        self.NCN_kappa = NCN_kappa

        self.BT_ls = ArmijoLineSearch()

        self.no_grad_next = False

        if fallback_opt == "eye":
            self.fallback = self.eye_fallback
    
    @staticmethod
    def eye_fallback(N):
        return jnp.eye(N) * 0 
    
    @staticmethod
    def Bk_mat_vec(Bk_vecs, Bk_scalars, v, memory):
        result = jnp.zeros(v.shape[0])
        for i in range(memory):
            x = Bk_vecs[:, i]
            result += Bk_scalars[i] * x * jnp.dot(x, v)
        return result
    
    def build_Bk(self, Bk_vecs, Bk_scalars):
        n = Bk_vecs.shape[0]
        result = jnp.zeros((n, n))
        for i in range(self.current_memory):
            x = Bk_vecs[:, i]
            result += jnp.outer(x, x) * Bk_scalars[i]
        return result

    def SR1_update(self, Bk_vecs, Bk_scalars, U_0_next, U_0, grad_next, grad, N, eps=1e-8):
        s = U_0_next - U_0
        y = grad_next - grad

        # r = y - B_k s
        r = y - self.Bk_mat_vec(Bk_vecs, Bk_scalars, s, self.current_memory)
        denom = jnp.vdot(r, s)

        if jnp.abs(denom) <= eps:
            # no change
            print("Skip SR1")
            return Bk_vecs, Bk_scalars

        V, A = Bk_vecs, Bk_scalars
        if self.current_memory < self.max_memory:
            idx = int(self.current_memory)
            V = V.at[:, idx].set(r)
            A = A.at[idx].set(1.0 / denom)
            self.current_memory = idx + 1
        else:
            N = U_0.shape[0]
            V = jnp.concatenate([V[:, 1:], jnp.zeros((N, 1), dtype=V.dtype)], axis=1)
            A = jnp.concatenate([A[1:], jnp.zeros((1,), dtype=A.dtype)], axis=0)
            idx = self.max_memory - 1
            V = V.at[:, idx].set(r)
            A = A.at[idx].set(1.0 / denom)
            self.current_memory = self.max_memory

        return V, A

    def HVP_Bk_update(self, vTAv_array, v_array, Bk_vecs, Bk_scalars, cos_tol = .95):
        """
        Insert n_new_vecs columns (xi) with scalars (-mu) into limited-memory buffers.

        Args
        ----
        vTAv_array : (n_new_vecs,)   # v_i^T A v_i
        v_array    : (N, n_new_vecs) # columns are x_i to insert
        Bk_vecs    : (N, M)          # memory buffer of vectors (M = self.max_memory)
        Bk_scalars : (M,)            # memory buffer of scalars
        """

        n_new = vTAv_array.shape[0]
        if n_new > 1:
            VTV = v_array.T @ v_array
            ortho_error = jnp.linalg.norm(VTV - jnp.eye(VTV.shape[0]))
            if ortho_error > 1e-8:
                print("vecs not ortho")
                return

        N, M = Bk_vecs.shape
        J = self.current_memory
        #linear indep check
        for i in range(n_new):
            xi = v_array[:, i]
            for j in range(J):
                xj = Bk_vecs[:, j]
                cos_sim = jnp.dot(xj, xi) / (jnp.linalg.norm(xi) * jnp.linalg.norm(xj))
                if jnp.abs(cos_sim) > cos_tol:
                    #delete
                    zeros_cols = jnp.zeros((N, 1), dtype=Bk_vecs.dtype)
                    Bk_vecs = jnp.concat([jnp.delete(Bk_vecs, j, axis=1), zeros_cols], axis=1)
                    Bk_scalars = jnp.concat([jnp.delete(Bk_scalars, j), jnp.zeros(1, dtype=Bk_scalars.dtype)])
                    self.current_memory -= 1

        cmem = self.current_memory
        n_free = self.max_memory - cmem

        # If not enough free slots, evict the oldest n_del pairs by shifting left.
        if n_free < n_new:
            n_del = n_new - n_free
            # shift left by n_del, append zeros at end
            zeros_cols = jnp.zeros((N, n_del), dtype=Bk_vecs.dtype)
            Bk_vecs    = jnp.concatenate([Bk_vecs[:, n_del:], zeros_cols], axis=1)
            Bk_scalars = jnp.concatenate([Bk_scalars[n_del:], jnp.zeros((n_del,), dtype=Bk_scalars.dtype)])
            cmem = cmem - n_del                      # current filled after eviction
            start_idx = cmem                         # first free slot
        else:
            start_idx = cmem

        # Insert new pairs sequentially; use the number of active pairs BEFORE each insert
        # when evaluating Bk_mat_vec.
        for i in range(n_new):
            xi = v_array[:, i]
            k_used = start_idx + i                   # active pairs available now
            # Compute mu from current memory (do NOT include the new xi yet)
            mu = jnp.dot(xi, self.Bk_mat_vec(Bk_vecs, Bk_scalars, xi, k_used)) - vTAv_array[i]
            # Write into buffers
            Bk_vecs    = Bk_vecs.at[:, k_used].set(xi)
            Bk_scalars = Bk_scalars.at[k_used].set(-mu)


        # Update the "filled slots" counter deterministically
        self.current_memory = int(min(self.max_memory, start_idx + n_new))

        return Bk_vecs, Bk_scalars

    def TR_opt(self, pk, g, pTHp, loss_fn, x0, loss):
        """
        Cubic regularized trust-region step solver.
        Solves for step length α in m(α) = loss + αc + ½α²b + (a/3)α³
        where a = (η/2)||p||³.
        """
        c = jnp.dot(pk, g)
        b = pTHp
        p_norm = jnp.linalg.norm(pk)
        a = 0.5 * self.eta * (p_norm ** 3)
        eps = 1e-12

        # Solve aα² + bα + c = 0 → α = (-b + sqrt(b² - 4ac)) / (2a)
        disc = b * b - 4 * a * c
        if disc < 0 or a == 0:
            alpha = self.BT_ls(loss_fn, x0, pk, g)
            return alpha

        alpha = (-b + jnp.sqrt(disc)) / (2 * a)
        if jnp.isnan(alpha) or jnp.isinf(alpha) or alpha <= 0:
            print(f"alpha invalid | a={a}, b={b}, c={c}")
            alpha = self.BT_ls(loss_fn, x0, pk, g)
            return alpha

        # Evaluate model and new loss
        model = loss + alpha * c + 0.5 * alpha**2 * b + (a / 3) * alpha**3
        new_loss = loss_fn(x0 + alpha * pk)

        # Compute trust-region ratio
        pred_red = loss - model
        act_red = loss - new_loss
        rho = act_red / (pred_red + eps)

        # Update eta
        if pred_red <= 0 or act_red <= 0 or jnp.isnan(rho):
            self.eta *= self.rho
        elif rho > 0.9:
            self.eta /= self.rho
        elif rho < 0.25:
            self.eta *= self.rho

        self.eta = float(max(self.eta, self.eta_min))
        return float(alpha)
    
    @staticmethod
    def check_eig(hvp, x, v):
        vTHv = jnp.dot(v, hvp(x, v))
        return vTHv/jnp.linalg.norm(v)
    
    @staticmethod
    def weibull_thresh_prob(x, x_thr=-5e-4, q_thr=0.3, k=1.5):
        lam = (-x_thr) / (-jnp.log1p(-q_thr))**(1.0/k)
        t = jnp.maximum(0.0, -x) / lam
        return 1.0 - jnp.exp(-jnp.power(t, k))
    
    def opt_loop(self, U_0, loss_fn, loss_grad_fn_base, div_check, div_free_proj):
        N = U_0.shape[0]
        loss_grad_fn = jax.jit(loss_grad_fn_base)

        @jax.jit
        def hvp(x, v):
            return jax.jvp(jax.grad(loss_fn), (x,), (v,))[1]

        def inner_loop(U_0, grad, Bk_vecs, Bk_scalars, loss):
            # Search direction and line search
            grad_norm = jnp.linalg.norm(grad)
            grad_norm_cond = grad_norm > self.eps_g
            gTHg = jnp.dot(grad, hvp(U_0, grad))
            Rk = gTHg / grad_norm**2
            Bk_vecs, Bk_scalars = self.HVP_Bk_update(jnp.array([gTHg]), grad.reshape((-1, 1)), Bk_vecs, Bk_scalars)
            step_type = "none"
            if Rk < -self.eps_H and grad_norm_cond and (not self.no_grad_next):
                print("grad neg")
                pk = -grad
                step_type = "grad"

            elif Rk >= -self.eps_H and Rk <= self.eps_H and grad_norm_cond and (not self.no_grad_next):
                print("grad flat")
                pk = -grad
                step_type = "grad"

            #second order methods
            else:
                #compute min Bk Eig
                if False:
                    vTAv_array = []
                    v_array = []
                    def hvp_store(v):
                        Av = hvp(U_0, v)
                        vTAv_array.append(jnp.dot(v, Av))
                        v_array.append(v)
                        return Av
                    A_op = LinearOperator((N, N), matvec=hvp_store, dtype=np.float64)
                    w, Q = eigsh(A_op, k=1, which='SA', tol=1, v0=min_Bk_eig_vec, ncv=5, maxiter=5)
                    vTAv_array = jnp.array(vTAv_array)
                    v_array = jnp.array(v_array).T
                    print(v_array.shape, vTAv_array.shape)
                
                computed_min_eig = False
                if random.random() < 0.1:
                    w, Q, vTAv_array, v_array  = lanczos_eigs(lambda v: hvp(U_0, v), U_0.shape[0], m=4)
                    Bk_vecs, Bk_scalars = self.HVP_Bk_update(vTAv_array, v_array, Bk_vecs, Bk_scalars)

                    min_eig_idx = jnp.argmin(w)
                    max_eig_idx = jnp.argmax(w)
                    max_eig_vec = Q[:, max_eig_idx]
                    min_eig_vec = Q[:, min_eig_idx]
                    min_eig = w[min_eig_idx]
                    max_eig = w[max_eig_idx]
                    if jnp.dot(min_eig_vec, grad) > 0:
                            min_eig_vec = -min_eig_vec
                    computed_min_eig = True

                ###
                if False:
                    print("-------")
                    for i in range(4):
                        xi = v_array[:, i]
                        t = jnp.dot(xi, self.Bk_mat_vec(Bk_vecs, Bk_scalars, xi, self.max_memory))
                        print(t, vTAv_array[i])

                    print("------")
                ###

                #print(f"hvp_min: {jnp.min(w)}, Bk_min: {jnp.min(Bk_eigs)}")
                #print(self.check_eig(hvp, U_0, max_eig_vec), max_eig)
                #print(self.check_eig(hvp, U_0, min_eig_vec), min_eig)
                #print("-----")

                if computed_min_eig and min_eig < -self.eps_H:
                    print(f"neg curve: {min_eig:.4e}")
                    pk = min_eig_vec
                    step_type = "neg_curve"

                elif grad_norm_cond:
                    NCN_min_eig = self.eps_H
                    #reg newton
                    #eigen decomp
                    Bk_mat_vec = lambda v: self.Bk_mat_vec(Bk_vecs, Bk_scalars, v, self.current_memory)
                    Bk_mat_vec = jax.jit(Bk_mat_vec)
                    #Bk_eigs, Bk_eig_vec, _, _ = lanczos_eigs(Bk_mat_vec, U_0.shape[0], m=self.current_memory*2, v0=max_eig_vec)
                    A_op = LinearOperator((N, N), matvec=Bk_mat_vec, dtype=np.float64)
                    Bk_eigs, Bk_eig_vec = eigsh(A_op, k=self.current_memory, which='LM')
                    
                    Q = Bk_eig_vec
                    Lam_reg = jnp.abs(Bk_eigs)
                    idx = jnp.argsort(Lam_reg)[-self.current_memory:][::-1]
                    Lam_reg = Lam_reg[idx]
                    Q = Bk_eig_vec[:, idx]
                    Lam_reg = jnp.where(Lam_reg > NCN_min_eig, Lam_reg, NCN_min_eig)

                    null_space_comp = (-grad) - Q @ (Q.T @ -grad)
                    proj_error = jnp.linalg.norm(null_space_comp)/jnp.linalg.norm(grad)
                    t = Q.T @ Q
                    ortho_error = jnp.linalg.norm(t - jnp.eye(t.shape[0]))
                    #print(f"proj error: {proj_error:.4f} | ortho error: {ortho_error:.3e}")
                    if proj_error < .5:
                        print("reg_newton")
                        step_type = "reg_Newton"
                        pk = Q @ ((Q.T @ -grad) / Lam_reg) + (1.0 / NCN_min_eig) * null_space_comp
                    else:
                        print("no NCN, grad")
                        pk = -grad
                        step_type = "grad"

            if step_type == "none":
                print("converged")
                return
            
            if True:
                if step_type == "grad":
                    pTHp = gTHg
                else:
                    pTHp = jnp.dot(pk, hvp(U_0, pk))
                    Bk_vecs, Bk_scalars = self.HVP_Bk_update(jnp.array([pTHp]), pk.reshape((-1, 1)), Bk_vecs, Bk_scalars)
                    if self.no_grad_next:
                        self.no_grad_next = False

                alpha = self.TR_opt(pk, grad, pTHp, loss_fn, U_0, loss)
    
            alpha_pk = pk * alpha

            # Step and new loss/grad
            U_0_next = self.step(U_0, alpha, pk)
            loss_next, grad_next = loss_grad_fn(U_0_next)
            Bk_vecs, Bk_scalars = self.SR1_update(Bk_vecs, Bk_scalars, U_0_next, U_0, grad_next, grad, N)
            
            return U_0_next, loss_next, grad_next, alpha, alpha_pk, Rk, Bk_vecs, Bk_scalars
            
        
        opt_data = Opt_Data(self.its)
        Bk_vecs = jnp.zeros((N, self.max_memory))
        Bk_scalars = jnp.zeros(self.max_memory)
        self.current_memory = 0
        for i in range(self.its):
            if i == 0:
                loss, grad = loss_grad_fn(U_0)
                print(f"loss: {loss}")
            loss_prev = loss
            grad_prev = grad
            U_0, loss, grad, alpha, alpha_pk, Rk, Bk_vecs, Bk_scalars = inner_loop(U_0, grad, Bk_vecs, Bk_scalars, loss)
            if alpha == 0:
                if self.print_loss:
                    print(f"optimizer stalled | alpha={alpha}")

                opt_data.early_stop_update(i)
                break
            opt_data(i, loss_prev, grad_prev, alpha_pk)

            #U_0_div = div_check(U_0)
            #if U_0_div > 1e-10:
            #    U_0 = div_free_proj(U_0)


            if self.print_loss:
                print(f"i:{i} | loss: {loss:.3e} | Div: {div_check(U_0):.3e} | alpha: {alpha:.3e} | Rk: {Rk:.3e} | eta: {self.eta:.3e}")


        return U_0, opt_data, i
        
def jit_eigen_decomp(H):
    return jnp.linalg.eigh(H)

class NCN(LS_Opt):
    name = "NCN"
    def __init__(self, ls_method, its, cond_num_cutoff):
        self.ls_method = ls_method
        self.set_ls()
        self.its = its
        self.cond_num_cutoff = cond_num_cutoff


    def direction(
        self,
        grad: jnp.ndarray,     # ∇f(x), shape (m,)
        hess: jnp.ndarray,      # ∇²f(x), shape (m,m)
    ) -> jnp.ndarray:

        # 2) eigen-decompose
        λ, Q = jit_eigen_decomp(hess)

        # 3) abs + truncate small modes
        λ_abs_clipped = jnp.maximum(jnp.abs(λ), jnp.max(jnp.abs(λ)) * self.cond_num_cutoff)
        # 4) build |H|⁻¹ = Q_tr · diag(1/λ_tr) · Q_trᵀ
        inv_abs_H = Q @ jnp.diag(1.0 / λ_abs_clipped) @ Q.T  # (m,m)
        
        # 5) saddle-free Newton step direction
        return - inv_abs_H @ grad                         # (m,)

class ArmijoLineSearch:
    name = "Arm_BT"
    def __init__(
        self,
        alpha_init: float = 1.0,
        rho: float        = 0.5,
        c: float          = 1e-4,
        max_iters: int    = 10
    ):
        self.alpha_init = alpha_init
        self.rho        = rho
        self.c          = c
        self.max_iters  = max_iters
        self.min_alpha = rho**max_iters


    def __call__(
        self,
        f,
        x: jnp.ndarray,
        p: jnp.ndarray,
        grad: jnp.ndarray,
    ) -> float:
        alpha  = self.alpha_init
        f0 = f(x)
        g0 = jnp.dot(grad, p)

        for _ in range(self.max_iters):
            new_loss = f(x + alpha*p)
            max_loss = f0 + self.c*alpha*g0
            if new_loss <= max_loss:
                break
            alpha *= self.rho
        return alpha
    
class Cubic_BT_LS:
    name = "Cub_BT"
    def __init__(
        self,
        alpha_init: float = 1.0,
        rho: float        = 0.5,
        eta: float          = 1,
        max_iters: int    = 10
    ):
        self.alpha_init = alpha_init
        self.rho        = rho
        self.eta          = eta
        self.max_iters  = max_iters
        self.min_alpha = rho**max_iters


    def __call__(
        self,
        f,
        x: jnp.ndarray,
        p: jnp.ndarray,
    ) -> float:
        alpha  = self.alpha_init
        f0 = f(x)
        step_norm = jnp.linalg.norm(p)
        for _ in range(self.max_iters):
            new_loss = f(x + alpha*p)
            max_loss = f0 - (self.eta/6) * alpha**3 * step_norm**3
            if new_loss <= max_loss:
                break
            alpha *= self.rho
        return alpha

def Hessian_Opt(U_0, loss_fn, grad_fn, Hess_fn, optimizer: LS_Opt, div_check):
    #grad_fn = jax.jit(grad_fn)
    #Hess_fn = jax.jit(Hess_fn)

    jax.jit
    def step(U_0):
        loss = loss_fn(U_0)
        grad = grad_fn(U_0)
        Hess = Hess_fn(U_0)

        p = optimizer.direction(grad, Hess)
        #alpha = optimizer.ls(loss_fn, U_0, p, grad)
        alpha = .5
        U_0 = optimizer.step(U_0, alpha, p)

        return U_0, loss, grad, Hess

    for it in range(optimizer.its):
        print("start")
        U_0, loss, grad, Hess = step(U_0)
        print(grad.shape, Hess.shape, U_0.shape)
        print(f"i:{it} | loss: {loss}")

def optax_opt(U_0, loss_fn, loss_grad_fn, optimizer_config, div_check, div_free_proj):
    optimizer = optimizer_config.alg
    opt_state = optimizer.init(U_0)
    opt_data = Opt_Data(optimizer_config.its)

    @jax.jit
    def inner_loop(U_0, opt_state):
        loss, grad = loss_grad_fn(U_0)

        updates, opt_state = optimizer.update(
            grad, opt_state, U_0, value=loss, grad=grad, value_fn=loss_fn
        )

        div_updates = div_check(updates)
        U_0_next = optax.apply_updates(U_0, updates)
        U_0_next = div_free_proj(U_0_next)
        return U_0_next, opt_state, loss, grad, updates, div_updates

    for i in range(optimizer_config.its):
        U_0, opt_state, loss, grad, alpha_p, div_updates = inner_loop(U_0, opt_state)
        opt_data(i, loss, grad, alpha_p)
        print(f"i:{i} | loss: {loss} | Div: {div_check(U_0)} | div updates: {div_updates} | {jnp.linalg.norm(alpha_p)}")

    
    
    del optimizer

    return U_0, opt_data

