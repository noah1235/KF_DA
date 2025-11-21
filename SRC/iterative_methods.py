import jax, jax.numpy as jnp
from jax import debug
from jax.scipy.linalg import eigh_tridiagonal
from jax import lax

def pcg(matvec, M, b, x0=None, maxiter=10, tol=1e-8, eps=1e-12):
    """
    Preconditioned Conjugate Gradient in JAX, but using plain Python loops.

    Solves A x ≈ b where:
      - matvec: function v -> A v   (e.g. Hv)
      - M: preconditioner (matrix or linear operator), approx A^{-1}
      - b: RHS
      - x0: optional initial guess (defaults to zeros)
      - maxiter: max iterations
      - tol: stopping tolerance on ||r||

    Returns:
      x    : approximate solution
      info : dict with {'num_iter', 'res_norm'}
    """

    b = jnp.asarray(b)
    M = jnp.asarray(M)

    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = jnp.asarray(x0)

    def apply_M(r):
        return M @ r

    # Initial residual & search direction
    r = b - matvec(x)
    z = apply_M(r)
    p = z

    rz = jnp.dot(r, z)
    res_norm = jnp.linalg.norm(r)

    num_iter = 0

    for k in range(maxiter):
        # stopping criterion
        if res_norm <= tol:
            break

        Ap = matvec(p)
        pAp = jnp.dot(p, Ap)

        alpha = rz / (pAp + eps)

        x = x + alpha * p
        r = r - alpha * Ap

        z = apply_M(r)
        rz_new = jnp.dot(r, z)

        beta = rz_new / (rz + eps)
        p = z + beta * p

        rz = rz_new
        res_norm = jnp.linalg.norm(r)
        num_iter = k + 1

    info = {
        "num_iter": num_iter,
        "res_norm": res_norm,
    }
    return x, info


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