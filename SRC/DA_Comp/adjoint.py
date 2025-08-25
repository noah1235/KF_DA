import jax
import jax.numpy as jnp
from jax import lax

def symmetric_error(A):
    sym_err = jnp.linalg.norm(A - A.conj().T) / jnp.linalg.norm(A)
    print(f"sym error: {sym_err}")

def JT_times_vec(fn, x, lam_out):
    _, vjp = jax.vjp(fn, x)     # pullback: cotangent -> input space
    (res,) = vjp(lam_out)
    return res

def JT_times_matrix(f, u, S):
    """
    Compute (J_f(u))^H @ S, where S is a matrix (n, p).
    For real-valued problems, (.)^H == (.)^T.

    f : C^n -> C^n (vector -> vector)
    u : (n,) complex (state at step n)
    S : (n,) or (n, p) complex  (e.g., dλ_{n+1}/dβ)

    Returns
    -------
    out : same shape as S
        (n,) if S is (n,)
        (n, p) if S is (n, p)
    """
    _, vjp_fun = jax.vjp(f, u)  # vjp_fun(c) = (J_f(u))^H @ c
    # apply columnwise
    def _col(c):
        (res,) = vjp_fun(c)
        return res
    return jax.vmap(_col, in_axes=1, out_axes=1)(S)

class Adjoint_Stepper_1:
    def __init__(self, part_idx, crit, target_part_x_array, target_part_y_array, f):
        self.crit = crit
        self.target_part_x_array = target_part_x_array
        self.target_part_y_array = target_part_y_array
        self.f = f
    
        self.loss_dg__du_fn = jax.value_and_grad(self.g)
        self.part_idx = part_idx

    def g(self, U, i):
        part = U[:self.part_idx].real
        part_x = part[::4]
        part_y = part[1::4]
        return self.crit.g(part_x, part_y, self.target_part_x_array[i], self.target_part_y_array[i], i)
    
    def df__du_v_fn(self, u_n, v):
        _, vjp_fun = jax.vjp(self.f, u_n)
        return vjp_fun(v)[0]

    def __call__(self, lam_n_1, u_n, i):
        _, vjp_fun = jax.vjp(self.f, u_n)
        g_val, dg__du = self.loss_dg__du_fn(u_n, i)

        #lam_n = vjp_fun(lam_n_1)[0] + dg__du
        lam_n = self.df__du_v_fn(u_n, lam_n_1) + dg__du
        #lam_n = jvp_val + dg__du

        return lam_n, g_val
        
class Adjoint_Stepper_2:
    def __init__(self, adj_step_1):
        self.adj_step_1 = adj_step_1
        self.g_Hess_fn = jax.hessian(adj_step_1.g)
        self.df__du_fn = jax.jacobian(adj_step_1.f)


    def g_Hess_V(self, U, i, V):
        g_hess = self.g_Hess_fn(U, i)
        return g_hess @ V

    def g_Hess_V_dec(self, U, i, V):
        # H_g(U) @ V without forming H_g
        gU = lambda U_: self.adj_step_1.g(U_, i)
        grad_g = jax.grad(gU)
        _, Hv = jax.jvp(grad_g, (U,), (V,))
        return Hv
    

    def g_Hess_V(self, U, i, V):
        # H_g(U) @ V without building H_g(U)
        gU = lambda U_: self.adj_step_1.g(U_, i)
        grad_g = jax.grad(gU)
        # Linearize grad once; lin : R^n -> R^n
        _, lin = jax.linearize(grad_g, U)
                    # V is (n, k)
        return jax.vmap(lin, in_axes=1, out_axes=1)(V)  # (n, k)

    @staticmethod
    def lambda_Hf_mat(f, u, lam, V):
        """
        Computes W = lam^H * f''(u) * V, columnwise.
        V : (n, k) complex
        returns: (n, k) complex with columns w_j = lam^H f''(u) v_j
        """
        def pullback(u_):
            _, vjp = jax.vjp(f, u_)
            (g,) = vjp(lam)            # g(u) = J(u)^H lam
            return g
        # Linearize once, then apply to all columns efficiently
        _, lin = jax.linearize(pullback, u)   # lin: R^n -> R^n (handles complex as 2R)
        return jax.vmap(lin, in_axes=1, out_axes=1)(V)
        
    def __call__(self, lam_n_1, dlam_n_1, u_n, d_u_n, i):
        lam_n, g_val = self.adj_step_1(lam_n_1, u_n, i)

        loss_term = self.g_Hess_V(u_n, i, d_u_n)
        solver_2_term = self.lambda_Hf_mat(self.adj_step_1.f, u_n, lam_n_1, d_u_n)
        dlam_term = JT_times_matrix(self.adj_step_1.f, u_n, dlam_n_1)

        dlam_n = dlam_term + solver_2_term + loss_term
        return lam_n, dlam_n, g_val

def make_second_term_matvec(Jt_fn, x, grad_f_t):
    """
    Returns mv(v) = S(x) @ v, where S(x) = Σ_i grad_f_t[i] * Hess_x t_i(x).
    No (m,n,n) tensor is formed; uses a single JVP per v.
    """
    grad_f_t = jax.lax.stop_gradient(grad_f_t)
    def mv(v):
        # DJ_t[x][v] has shape (m, n)
        DJ = jax.jvp(Jt_fn, (x,), (v,))[1]
        return DJ.T @ grad_f_t  # (n,)
    return mv

def second_term_apply_to_matrix(Jt_fn, x, grad_f_t, V):
    """
    Columnwise S(x) @ V without forming S or any 3rd-order tensor.
    V: (n, k)
    """
    mv = make_second_term_matvec(Jt_fn, x, grad_f_t)
    return jax.vmap(mv, in_axes=1, out_axes=1)(V)  # (n, k)



#               _ _       _       _      _____                _                   _
#      /\      | (_)     (_)     | |    / ____|              | |                 | |
#     /  \   __| |_  ___  _ _ __ | |_  | |     ___  _ __  ___| |_ _ __ _   _  ___| |_ ___  _ __ ___
#    / /\ \ / _` | |/ _ \| | '_ \| __| | |    / _ \| '_ \/ __| __| '__| | | |/ __| __/ _ \| '__/ __|
#   / ____ \ (_| | | (_) | | | | | |_  | |___| (_) | | | \__ \ |_| |  | |_| | (__| || (_) | |  \__ \
#  /_/    \_\__,_| |\___/|_|_| |_|\__|  \_____\___/|_| |_|___/\__|_|   \__,_|\___|\__\___/|_|  |___/
#               _/ |
#              |__/


def build_adjoint_grad_fn(pIC, crit, target_part, trj_gen_fn, f, transform_fn):
    p_idx = pIC.shape[0]
    target_part = target_part.real
    target_part_x_array = target_part[:, ::4]
    target_part_y_array = target_part[:, 1::4]
    N = target_part.shape[0]
    #target_part_array = jnp.concatenate([target_part_x_array, target_part_y_array], axis=1)


    adj_step = Adjoint_Stepper_1(p_idx, crit, target_part_x_array, target_part_y_array, f)
    
    def loss_grad_fn(u_0_vel_raw):
        u_0_vel = transform_fn(u_0_vel_raw)
        DA_trj = trj_gen_fn(pIC, u_0_vel)

        loss, lam_N = adj_step.loss_dg__du_fn(DA_trj[-1], N-1)
        lam = lam_N
        for i in range(N-2, -1, -1):
            lam, g_val = adj_step(lam, DA_trj[i], i)
            loss += g_val
        

        grad = JT_times_vec(transform_fn, u_0_vel_raw, lam[p_idx:])

        return loss, grad
        
    
    return loss_grad_fn
        
def build_adjoint_Hess_fn(pIC, crit, target_part, trj_sens_gen_fn, f, transform_fn):
    p_idx = pIC.shape[0]
    target_part = target_part.real
    target_part_x_array = target_part[:, ::4]
    target_part_y_array = target_part[:, 1::4]
    N = target_part.shape[0]    

    adj_step_1 = Adjoint_Stepper_1(p_idx, crit, target_part_x_array, target_part_y_array, f)
    adj_step_2 = Adjoint_Stepper_2(adj_step_1)

    J_fn = jax.jacrev(transform_fn)

    @jax.jit
    def loss_grad_Hess_fn(u_0_vel_raw):
        # Transform initial state and generate trajectory + sensitivities
        u_0_vel = transform_fn(u_0_vel_raw)
        DA_trj, DA_sens = trj_sens_gen_fn(pIC, u_0_vel)   # shapes (N, ...)

        # Terminal loss/adjoints
        loss0, lam_N = adj_step_1.loss_dg__du_fn(DA_trj[-1], N-1)
        dlam_N = adj_step_2.g_Hess_V(DA_trj[-1], N-1, DA_sens[-1])

        # Backward sweep with scan (reverse=True)
        # xs carries per-step inputs; i goes 0..N-2, but scan runs in reverse
        xs = (DA_trj[:-1], DA_sens[:-1], jnp.arange(N-1, dtype=jnp.int32))

        def step(carry, x):
            lam, dlam, loss_acc = carry
            trj_i, sens_i, i = x
            lam, dlam, g_val = adj_step_2(lam, dlam, trj_i, sens_i, i)
            return (lam, dlam, loss_acc + g_val), None

        (lam, dlam, loss), _ = lax.scan(step,
                                        (lam_N, dlam_N, loss0),
                                        xs,
                                        reverse=True)

        # Slice to the block of interest in t-space
        grad_t = lam[p_idx:]                  # (m,)
        Hess_t = dlam[p_idx:, p_idx:]         # (m,m)

        # Gradient transform: ∇_x h = J^T ∇_t f
        J = J_fn(u_0_vel_raw)                 # (m,n)
        grad_x = J.T @ grad_t                 # (n,)

        # First Hessian term: J^T H_f J
        t1 = J.T @ Hess_t @ J                 # (n,n)

        # Second Hessian term via JVP matvecs, no 3rd-order tensor:
        n = J.shape[1]
        I = jnp.eye(n, dtype=J.dtype)
        t2 = second_term_apply_to_matrix(J_fn, u_0_vel_raw, grad_t, I)  # (n,n)

        Hess_x = t1 + t2
        return loss, grad_x, Hess_x
        
    
    return loss_grad_Hess_fn


