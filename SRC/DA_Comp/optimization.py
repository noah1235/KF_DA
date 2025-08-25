import jax.numpy as jnp
import jax
import optax
import numpy as np
from SRC.utils import real_concat_to_complex

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
        max_iters: int    = 20
    ):
        self.alpha_init = alpha_init
        self.rho        = rho
        self.c          = c
        self.max_iters  = max_iters

    def __call__(
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

def Hessian_Opt(U_0_real_guess, loss_fn, grad_fn, Hess_fn, optimizer: LS_Opt):
    grad_fn = jax.jit(grad_fn)
    Hess_fn = jax.jit(Hess_fn)

    def step(U_0_real):
        loss = loss_fn(U_0_real)
        grad = grad_fn(U_0_real)
        Hess = Hess_fn(U_0_real)
        print(grad.shape, Hess.shape)

        p = optimizer.direction(grad, Hess)
        alpha = optimizer.ls(loss_fn, U_0_real, p, grad)
        U_0_real = optimizer.step(U_0_real, alpha, p)

        return U_0_real, loss, grad, Hess
    U_0_real = U_0_real_guess
    for it in range(optimizer.its):
        print(it)
        U_0_real, loss, grad, Hess = step(U_0_real)
        print(loss)

def optax_opt(U_0_real, loss_fn, loss_grad_fn, optimizer_config, div_check, div_free_proj):
    optimizer = optimizer_config.alg
    opt_state = optimizer.init(U_0_real)
    opt_data = Opt_Data(optimizer_config.its)

    @jax.jit
    def inner_loop(U_0_real, opt_state):
        loss, grad = loss_grad_fn(U_0_real)
        updates, opt_state = optimizer.update(
            grad, opt_state, U_0_real, value=loss, grad=grad, value_fn=loss_fn
        )
        div_updates = div_check(updates)
        U_0_real_next = optax.apply_updates(U_0_real, updates)
        U_0_real_next = div_free_proj(U_0_real_next)
        return U_0_real_next, opt_state, loss, grad, updates, div_updates

    for i in range(optimizer_config.its):
        U_0_real, opt_state, loss, grad, alpha_p, div_updates = inner_loop(U_0_real, opt_state)
        opt_data(i, loss, grad, alpha_p)
        print(f"i:{i} | loss: {loss} | Div: {div_check(U_0_real)} | {div_updates} | {jnp.linalg.norm(alpha_p)}")

    
    
    del optimizer

    return U_0_real, opt_data

