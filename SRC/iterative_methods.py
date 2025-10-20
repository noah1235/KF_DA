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

def lanczos_eigs(matvec, n, m, v0=None, tol=1e-12, dtype=jnp.float64):
    """
    Lanczos with m steps for a symmetric operator A (given by matvec).
    Simple Python for-loop; no JIT tricks, just clarity.

    Returns:
      evals : (k,)            — Ritz eigenvalues of A (from k×k T)
      evecs : (n, k)          — Ritz eigenvectors in R^n, V^T @ y
      vTAv_array : (m,)       — v_t^T A v_t for t=0..k-1 (zeros after k)
      v_array : (n, m)        — columns store v_t (zeros after k)
    """
    if v0 is None:
        v0 = jnp.arange(n, dtype=dtype)
    v0 = v0 / (jnp.linalg.norm(v0) + 1e-32)

    # Preallocate
    V = jnp.zeros((m, n), dtype=dtype)          # store v_t as rows V[t] = v_t^T
    alphas = jnp.zeros((m,), dtype=dtype)
    betas  = jnp.zeros((m-1,), dtype=dtype)

    vTAv_array = jnp.zeros((m,),   dtype=dtype)
    v_array    = jnp.zeros((n, m), dtype=dtype)

    # init
    v_prev = jnp.zeros_like(v0)
    v = v0
    beta_prev = jnp.array(0., dtype=dtype)

    k = 0  # effective number of steps actually taken
    for t in range(m):
        Av = matvec(v)
        vTAv_array = vTAv_array.at[t].set(jnp.vdot(v, Av).real)
        v_array = v_array.at[:, t].set(v)

        w = Av - beta_prev * v_prev
        alpha = jnp.vdot(v, w).real
        w = w - alpha * v

        V = V.at[t].set(v)
        alphas = alphas.at[t].set(alpha)

        beta = jnp.linalg.norm(w)
        if t < m-1:
            betas = betas.at[t].set(beta)

        # prepare next
        v_next = jnp.where(beta > 0, w / (beta + 1e-32), jnp.zeros_like(v))
        v_prev, v, beta_prev = v, v_next, beta

        k = t + 1
        if beta < tol:
            break

    # Effective sizes
    alphas_eff = alphas[:k]
    betas_eff  = betas[:max(k-1, 0)]
    V_eff = V[:k]                 # shape (k, n); rows are basis vectors

    # Build the k x k tridiagonal T
    if k == 0:
        # degenerate edge case: return empties
        return (jnp.array([], dtype=dtype),
                jnp.zeros((n, 0), dtype=dtype),
                vTAv_array, v_array)

    T = jnp.diag(alphas_eff)
    if k > 1:
        T = T + jnp.diag(betas_eff, 1) + jnp.diag(betas_eff, -1)

    # Ritz decomposition of T
    evals, Y = jnp.linalg.eigh(T)     # Y: (k, k)

    # Lift Ritz vectors back to R^n: u_i = V_k^T y_i
    # V_eff: (k, n) with rows v_t^T  → V_eff.T: (n, k)
    evecs = V_eff.T @ Y                # (n, k)

    return evals, evecs, vTAv_array, v_array

def lanczos_tridiagonal(matvec, n, m, v0=None, tol=1e-12, dtype=jnp.float64):
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

    vTAv_array = jnp.zeros((m,),   dtype=v0.dtype)
    v_array    = jnp.zeros((n, m), dtype=v0.dtype)

    def body(carry, t):
        v_prev, v, beta_prev, V_acc, alphas_acc, betas_acc, vTAv_array, v_array = carry

        # w = A v - beta_{t-1} v_{t-1}
        Av = matvec(v)
        vTAv_array = vTAv_array.at[t].set(jnp.dot(v, Av))
        v_array = v_array.at[:, t].set(v)

        w = Av - beta_prev * v_prev

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
        carry_next = (v, v_next, beta, V_acc, alphas_acc, betas_acc, vTAv_array, v_array)
        return carry_next, None

    # Initialize with v_{0} = 0, v_{1} = v0, beta_0 = 0
    v_prev0 = jnp.zeros_like(v0)
    init_carry = (v_prev0, v0, jnp.array(0., dtype=v0.dtype), V, alphas, betas, vTAv_array, v_array)
    (v_prev, v_last, beta_last, V, alphas, betas, vTAv_array, v_array), _ = lax.scan(body, init_carry, jnp.arange(m))

    # Detect (first) breakdown to compute effective dimension k
    # betas has length m-1; find first index where beta < tol
    has_break = jnp.any(betas < tol)
    first_break = jnp.argmax(betas < tol)                     # index in [0..m-2]
    k = jnp.where(has_break, first_break + 1, m)              # effective steps

    # Slice to effective size
    alphas_eff = alphas[:k]
    betas_eff  = betas[:(k-1)]
    V_eff = V[:k]
    return alphas_eff, betas_eff, V_eff, vTAv_array, v_array

def lanczos_2_eig_decomp(alphas: jnp.ndarray,
                            betas: jnp.ndarray,
                            V: jnp.ndarray):
    """
    Given Lanczos outputs (alphas, betas, V), form the tridiagonal T,
    compute its eigendecomposition T = S Λ Sᵀ, and lift the Ritz vectors
    back to the original space: Q = V S.

    Args
    ----
    alphas : (m,)  diagonal entries of T
    betas  : (m-1,) off-diagonal entries of T (super/sub diagonal)
    V      : (n, m) Lanczos basis with orthonormal columns

    Returns
    -------
    w : (m,)        Ritz eigenvalues (sorted descending)
    Q : (n, m)      Ritz eigenvectors in the original space (columns), Q = V S
    S : (m, m)      Eigenvectors of T in the Krylov subspace (optional to use externally)
    """
    m = alphas.shape[0]
    # Build symmetric tridiagonal T
    T = jnp.diag(alphas) \
        + jnp.diag(betas, k=1) \
        + jnp.diag(betas, k=-1)

    # Eigendecomposition of symmetric T
    # jnp.linalg.eigh returns ascending; we’ll flip to descending for convenience.
    w, S = jnp.linalg.eigh(T)
    idx = jnp.argsort(w)[::-1]
    w = w[idx]
    S = S[:, idx]

    # Lift eigenvectors back: Q = V S
    Q = V.T @ S

    # (Optional) tiny numerical cleanup: re-normalize columns of Q
    # (in exact arithmetic it’s already orthonormal since V and S are)
    norms = jnp.linalg.norm(Q, axis=0)
    Q = Q / jnp.where(norms > 0, norms, 1.0)

    return w, Q, S

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