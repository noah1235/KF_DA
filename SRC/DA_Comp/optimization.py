import jax.numpy as jnp
import jax
import optax
import numpy as np
from jax import lax

class Optimizer:
    name = ""
    ls_method = ""
    its = -1

    def __repr__(self):
        return f"{self.name}_{self.ls_method}-{self.its}"
    
class Opt_Data:
    def __init__(self, its):
        self.loss_record = np.zeros(its)
        self.grad_norm_record = np.zeros(its)
        self.alpha_gTp_record = np.zeros(its)
    
    def __call__(self, n, loss, grad, alpha_p):
        self.loss_record[n] = loss
        self.grad_norm_record[n] = jnp.linalg.norm(grad)
        self.alpha_gTp_record[n] = jnp.dot(alpha_p, grad)

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

class LS_Opt(Optimizer):
    def set_ls(self):
        if self.ls_method == "BT":
            self.ls = ArmijoLineSearch(alpha_init=1.0, rho=0.25, c=1e-4, max_iters=20)

    def step(
        self,
        x: jnp.ndarray,      # current x, shape (m,)
        alpha: float,
        p: jnp.ndarray       # search direction, shape (m,)
    ) -> jnp.ndarray:
        return x + alpha * p
    
class BFGS(LS_Opt):
    name = "BFGS"
    def __init__(self, ls_method, its, fallback_opt):
        self.ls_method = ls_method
        self.set_ls()
        self.its = its

        if fallback_opt == "eye":
            self.fallback = self.eye_fallback
    @staticmethod
    def eye_fallback(N):
        return jnp.eye(N)
        

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
    """
    Find α such that
      f(x + α p) ≤ f(x) + c α ∇f(x)ᵀp
    via backtracking (ρ < 1).
    """
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

    def __call___dec(
        self,
        f,
        x: jnp.ndarray,
        p: jnp.ndarray,
        grad: jnp.ndarray,
    ) -> float:
        α  = self.alpha_init
        f0 = f(x)
        g0 = jnp.dot(grad, p)
        for _ in range(self.max_iters):
            if f(x + α*p) <= f0 + self.c*α*g0:
                break
            α *= self.rho
        return α
    

    def __call__(self, f, x: jnp.ndarray, p: jnp.ndarray, grad: jnp.ndarray) -> jnp.ndarray:
        # f(x) must return a scalar shape ().
        f0 = f(x)                             # ()
        g0 = jnp.vdot(grad, p)                # scalar (inner product)

        # If not a descent direction, return alpha = 0 without branching in Python.
        def do_search(_):
            alpha0 = jnp.asarray(self.alpha_init, dtype=f0.dtype)
            f_alpha0 = f(x + alpha0 * p)

            def cond_fun(state):
                alpha, f_alpha, k = state
                armijo = f0 + self.c * alpha * g0
                need_more = jnp.logical_and(f_alpha > armijo, k < self.max_iters)
                return need_more

            def body_fun(state):
                alpha, _, k = state
                alpha = alpha * self.rho
                f_alpha = f(x + alpha * p)
                return (alpha, f_alpha, k + 1)

            alpha_fin, _, _ = lax.while_loop(cond_fun, body_fun, (alpha0, f_alpha0, 0))
            return alpha_fin

        alpha = lax.cond(g0 < 0.0, do_search, lambda _: jnp.asarray(0.0, dtype=f0.dtype), operand=None)
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

def BFGS_opt(U_0, loss_fn, loss_grad_fn, optimizer: BFGS, div_check, div_free_proj):
    N = U_0.shape[0]

    @jax.jit
    def inner_loop(U_0, grad, Bk_inv):
        pk = -Bk_inv @ grad
        alpha = optimizer.ls(loss_fn, U_0, pk, grad)

        U_0_next = optimizer.step(U_0, alpha, pk)

        loss_next, grad_next = loss_grad_fn(U_0_next)

        sk = U_0_next - U_0
        yk = grad_next - grad

        rho_k = 1/(jnp.dot(yk, sk))

        Bk_inv_next = (I - rho_k * jnp.outer(sk, yk)) @ Bk_inv @ (I - rho_k * jnp.outer(yk, sk)) + rho_k * jnp.outer(sk, sk)

        return U_0_next, loss_next, grad_next, Bk_inv_next, alpha, pk * alpha

    
    Bk_inv = optimizer.fallback(N)
    I = jnp.eye(N)

    opt_data = Opt_Data(optimizer.its)
    for i in range(optimizer.its):
        if i == 0:
            loss, grad = loss_grad_fn(U_0)
        loss_prev = loss
        grad_prev = grad
        U_0, loss, grad, Bk_inv, alpha, alpha_pk = inner_loop(U_0, grad, Bk_inv)
        opt_data(i, loss_prev, grad_prev, alpha_pk)
        
        print(f"i:{i} | loss: {loss} | Div: {div_check(U_0)} | alpha: {alpha}")


    return U_0, opt_data


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

