import jax, jax.numpy as jnp
from jax import debug
from jax.scipy.linalg import eigh_tridiagonal
from jax import lax

import jax.numpy as jnp


def cg_curve_detection(
    matvec,        # function: v -> H v
    b,             # right-hand side (usually -grad)
    max_iters: int = 50,
    tol: float = 1e-8,
):
    """
    Conjugate Gradient with negative curvature detection (no preconditioning).

    Runs CG to approximately solve H p = b. During the iterations, if a
    direction d is found such that d^T H d <= 0 (up to a small tolerance),
    the routine returns a direction of (approximately) most negative
    curvature among all directions tested so far.

    Parameters
    ----------
    matvec : callable
        Function that applies H: matvec(v) = H v.
    b : array_like
        Right-hand side vector (often -grad).
    max_iters : int, optional
        Maximum number of CG iterations. Default is 50.
    tol : float, optional
        Convergence tolerance on the residual norm ||r||. Default is 1e-8.

    Returns
    -------
    p :
        If converged: the approximate CG solution.
        If negative curvature detected: a tuple (v_nc, curv) where v_nc is
        the selected negative-curvature direction and curv = v_nc^T H v_nc.
        If max_iters reached: the last iterate p.
    flag : {"converged", "indef", "max_iter"}
        Status flag indicating how the routine terminated.
    """

    # Initial guess p = 0
    p = jnp.zeros_like(b)

    # Initial residual r = b - H p (p = 0 => r = b)
    r = b.copy()
    d = -r
    rr_old = jnp.dot(r, r)

    hvp_count = 0  # number of matvec calls (not returned, but tracked)

    # Store all tested directions and their curvatures d^T H d
    dirs = []
    curvs = []

    for k in range(max_iters):
        Hd = matvec(d)
        hvp_count += 1

        dHd = jnp.dot(d, Hd)

        # Record this direction + curvature
        dirs.append(d)
        curvs.append(dHd)

        # ------------ negative curvature handling ------------
        if dHd <= 1e-16:
            # Pick the direction with the most negative curvature
            curvs_arr = jnp.array(curvs)
            idx = int(jnp.argmin(curvs_arr))   # most negative x^T H x
            v_nc = dirs[idx]

            return (v_nc, curvs_arr[idx]), "indef"

        # Standard CG update
        alpha = rr_old / dHd
        p_new = p + alpha * d
        r_new = r + alpha * Hd

        # Convergence check
        if jnp.linalg.norm(r_new) < tol:
            return p_new, "converged"

        rr_new = jnp.dot(r_new, r_new)
        beta = rr_new / rr_old

        d = -r_new + beta * d

        # Shift iterates
        p, r = p_new, r_new
        rr_old = rr_new

    # If we hit max iters with no NC and no convergence, just return p
    return p, "max_iter"


import jax.numpy as jnp


def pcg_curve_detection(
    matvec,        # function: v -> H v
    M_inv,         # matrix: approximates H^{-1} (can be identity)
    b,             # right-hand side (usually -grad)
    p,
    max_iters: int = 3,
    tol: float = 1e-14,
    curv_tol: float = -1e-12,  # small positive tolerance for "non-positive" curvature
):
    """
    Preconditioned Conjugate Gradient with negative curvature detection.

    Solves H p = b approximately. If during PCG we see a direction d with
    d^T H d <= curv_tol, we return a negative-curvature (or non-positive)
    direction instead of a CG step.

    Parameters
    ----------
    matvec : callable
        Function that applies H: matvec(v) = H v.
    M_inv : array_like
        Preconditioner approximating H^{-1}. Used via M_inv @ r.
    b : array_like
        Right-hand side vector (often -grad).
    max_iters : int, optional
        Maximum number of PCG iterations.
    tol : float, optional
        Convergence tolerance on ||r||.
    curv_tol : float, optional
        Threshold for detecting non-positive curvature. Default 0.0.

    Returns
    -------
    p :
        If converged: approximate solution to H p = b.
        If negative curvature detected: (v_nc, curv) where
           v_nc is the selected NC direction and curv = v_nc^T H v_nc.
        If max_iters reached: last iterate p.
    flag : {"converged", "indef", "max_iter"}
        Termination status.
    """


    # Residual r = b - H p  (here p = 0, so r = b)
    r = b - matvec(p)
    z = M_inv @ r
    d = z                        # <- standard PCG: search dir = preconditioned residual
    rz_old = jnp.dot(r, z)

    hvp_count = 0  # number of HVPs / matvec calls (not returned, but useful to keep)

    # Store all tested directions and their curvatures d^T H d
    dirs = []
    curvs = []

    for k in range(max_iters):
        Hd = matvec(d)
        hvp_count += 1

        dHd = jnp.dot(d, Hd)

        # Record this direction + curvature
        dirs.append(d)
        curvs.append(dHd)

        # ------------ negative curvature handling / breakdown ------------
        # If curvature is non-positive (or numerically tiny), treat as indef
        if dHd <= curv_tol:
            curvs_arr = jnp.array(curvs)
            idx = int(jnp.argmin(curvs_arr))   # most negative d^T H d
            v_nc = dirs[idx]
            return (v_nc, curvs_arr[idx]), "indef"

        # Standard PCG step
        alpha = rz_old / dHd

        p_new = p + alpha * d
        r_new = r - alpha * Hd    # <- **minus** here for PCG

        # Convergence check on residual
        if jnp.linalg.norm(r_new) < tol:
            return p_new, "converged"

        z_new = M_inv @ r_new
        rz_new = jnp.dot(r_new, z_new)

        # If rz_new is tiny, we’re basically done or breaking down
        if rz_new <= 0.0:
            # Either numerical breakdown or indef/preconditioner issue.
            # Just return current iterate.
            return p_new, "max_iter"

        beta = rz_new / rz_old

        # New search direction
        d = z_new + beta * d

        # Shift iterates
        p, r, z = p_new, r_new, z_new
        rz_old = rz_new

    # If we hit max_iters with no curvature issue and no convergence, return last iterate
    return p, "max_iter"


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