import jax, jax.numpy as jnp
from jax import debug
from jax.scipy.linalg import eigh_tridiagonal
from jax import lax

def max_eig_power_iterations(hvp, n, iters=10, key=0):
    key = jax.random.PRNGKey(key)
    v = jax.random.normal(key, (n,))
    v = v / jnp.linalg.norm(v)
    def step(v, _):
        w = hvp(v)
        v_new = w / (jnp.linalg.norm(w))
        lam = jnp.vdot(v_new, hvp(v_new)).real
        debug.print("λ ≈ {lam:.6f}", lam=lam)
        return v_new, lam
    v, lams = jax.lax.scan(step, v, None, length=iters)
    return lams, v



def lanczos_tridiagonal(matvec, n, m, v0=None, tol=1e-12, store_basis=True, dtype=jnp.float64):
    """
    Run m Lanczos steps for a symmetric operator A given by `matvec(x)=A@x`.

    Args:
      matvec: function R^n -> R^n (must be linear; A symmetric)
      n: dimension
      m: number of Lanczos steps (typ. 20–200)
      v0: optional initial vector (shape (n,)); if None, uses PRNGKey(0)
      tol: breakdown tolerance on beta
      store_basis: if True, returns V with Lanczos basis (m_eff x n)

    Returns:
      alphas: shape (m_eff,)
      betas:  shape (m_eff-1,)
      V:      shape (m_eff, n) if store_basis else None
    """
    if v0 is None:
        # deterministic seed for JIT-ability; pass a custom v0 for randomness
        v0 = jnp.arange(n, dtype=dtype)
        v0 = v0 / (jnp.linalg.norm(v0) + 1e-32)
    else:
        v0 = v0 / (jnp.linalg.norm(v0) + 1e-32)

    # Preallocate containers
    V = jnp.zeros((m, n), dtype=v0.dtype)
    alphas = jnp.zeros((m,), dtype=v0.dtype)
    betas  = jnp.zeros((m-1,), dtype=v0.dtype)

    def body(carry, t):
        v_prev, v, beta_prev, V_acc, alphas_acc, betas_acc = carry

        # w = A v - beta_{t-1} v_{t-1}
        w = matvec(v) - beta_prev * v_prev

        # alpha_t
        alpha = jnp.vdot(v, w).real
        w = w - alpha * v

        # (Optional: light reorth could go here against V_acc[:t], but we keep 3-term)

        beta = jnp.linalg.norm(w)
        # normalize next v (avoid NaN on breakdown)
        v_next = jnp.where(beta > 0, w / beta, jnp.zeros_like(v))

        # write into buffers
        V_acc = V_acc.at[t].set(v)
        alphas_acc = alphas_acc.at[t].set(alpha)
        betas_acc = lax.cond(
            t < (m - 1),
            lambda b: b.at[t].set(beta),
            lambda b: b,
            betas_acc
        )
        carry_next = (v, v_next, beta, V_acc, alphas_acc, betas_acc)
        return carry_next, None

    # Initialize with v_{0} = 0, v_{1} = v0, beta_0 = 0
    v_prev0 = jnp.zeros_like(v0)
    init_carry = (v_prev0, v0, jnp.array(0., dtype=v0.dtype), V, alphas, betas)
    (v_prev, v_last, beta_last, V, alphas, betas), _ = lax.scan(body, init_carry, jnp.arange(m))

    # Detect (first) breakdown to compute effective dimension k
    # betas has length m-1; find first index where beta < tol
    has_break = jnp.any(betas < tol)
    first_break = jnp.argmax(betas < tol)                     # index in [0..m-2]
    k = jnp.where(has_break, first_break + 1, m)              # effective steps

    # Slice to effective size
    alphas_eff = alphas[:k]
    betas_eff  = betas[:(k-1)]
    V_eff = V[:k] if store_basis else None
    return alphas_eff, betas_eff, V_eff


# ------------------------------------------------------------
# Extremal eigenpairs via Lanczos (Ritz)
# ------------------------------------------------------------
def lanczos_extremal(matvec, n, m=64, v0=None, tol=1e-12, return_vectors=False):
    alphas, betas, V = lanczos_tridiagonal(matvec, n, m, v0=v0, tol=tol, store_basis=return_vectors)
    k = alphas.shape[0]

    if return_vectors:
        # Build dense tridiagonal T and use jnp.linalg.eigh (vectors available)
        T = jnp.diag(alphas) + jnp.diag(betas, 1) + jnp.diag(betas, -1)
        d, Z = jnp.linalg.eigh(T)               # d: (k,), Z: (k,k)
        idx_min = jnp.argmin(d); idx_max = jnp.argmax(d)
        lam_min = d[idx_min]; lam_max = d[idx_max]

        y_min = Z[:, idx_min]; y_max = Z[:, idx_max]
        v_min = (V.T @ y_min); v_max = (V.T @ y_max)
        v_min = v_min / (jnp.linalg.norm(v_min) + 1e-32)
        v_max = v_max / (jnp.linalg.norm(v_max) + 1e-32)

        res_min = jnp.linalg.norm(matvec(v_min) - lam_min * v_min)
        res_max = jnp.linalg.norm(matvec(v_max) - lam_max * v_max)

        return dict(lambda_min=lam_min, lambda_max=lam_max,
                    v_min=v_min, v_max=v_max, res_min=res_min, res_max=res_max,
                    iters=k)
    
    else:
        # If you only want eigenvalues, tridiagonal eigvals are fine (if supported)
        # from jax.scipy.linalg import eigh_tridiagonal
        # d = eigh_tridiagonal(alphas, betas, eigvals_only=True)
        # idx_min = jnp.argmin(d); idx_max = jnp.argmax(d)
        # return dict(lambda_min=d[idx_min], lambda_max=d[idx_max],
        #             v_min=None, v_max=None, res_min=None, res_max=None, iters=k)
        # Fallback: dense eig even for values (keeps things simple/portable)
        T = jnp.diag(alphas) + jnp.diag(betas, 1) + jnp.diag(betas, -1)
        d = jnp.linalg.eigh(T)[0]
        idx_min = jnp.argmin(d); idx_max = jnp.argmax(d)
        return dict(lambda_min=d[idx_min], lambda_max=d[idx_max],
                    v_min=None, v_max=None, res_min=None, res_max=None, iters=k)