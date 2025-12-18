import jax.numpy as jnp

def abs_eig_decomp(Bk_vecs, Bk_scalars):
    r = Bk_scalars.shape[0]
    H = Bk_vecs @ jnp.diag(Bk_scalars) @ Bk_vecs.T

    eigt, eig_vect = jnp.linalg.eigh(H)
    pd_H = eig_vect @ jnp.diag(jnp.abs(eigt)) @ eig_vect.T
    # Rank-1 shortcut
    if r == 1:
        v = Bk_vecs[:, 0]
        v_norm = jnp.linalg.norm(v)
        eig = (v_norm**2) * jnp.abs(Bk_scalars[0])
        eigvec = v / v_norm
        return jnp.array([eig]), eigvec.reshape((-1, 1))

    A = Bk_vecs[:, :r]
    lam = jnp.abs(Bk_scalars[:r])
    B = A * jnp.sqrt(lam)[None, :]
    print(jnp.linalg.norm(pd_H - B @ B.T) / jnp.linalg.norm(pd_H))

    U, S, _ = jnp.linalg.svd(B, full_matrices=False)

    eigs = S**2
    eigvecs = U

    return eigs, eigvecs

def test_abs_eig_decomp(Bk_vecs, Bk_scalars, tol=1e-10):
    """
    Tests abs_eig_decomp on an indefinite low-rank symmetric matrix.
    """

    N, r = Bk_vecs.shape

    # --- construct dense Bk ---
    Bk = jnp.zeros((N, N))
    for i in range(r):
        Bk += Bk_scalars[i] * jnp.outer(Bk_vecs[:, i], Bk_vecs[:, i])

    # --- reference |Bk| via dense eigendecomposition ---
    Lam, Q = jnp.linalg.eigh(Bk)
    Bk_abs_ref = Q @ jnp.diag(jnp.abs(Lam)) @ Q.T

    # --- method under test ---
    eigs_abs, eigvecs = abs_eig_decomp(Bk_vecs, Bk_scalars)
    Bk_abs_test = eigvecs @ jnp.diag(eigs_abs) @ eigvecs.T

    # --- errors ---
    fro_err = jnp.linalg.norm(Bk_abs_test - Bk_abs_ref, ord="fro")
    fro_ref = jnp.linalg.norm(Bk_abs_ref, ord="fro")
    rel_err = fro_err / fro_ref

    print("Relative Frobenius error:", rel_err)

    # --- eigenvalue sanity check ---
    ref_eigs = jnp.sort(jnp.abs(Lam))[::-1][:r]
    test_eigs = jnp.sort(eigs_abs)[::-1]

    print("Reference |eigs|:", ref_eigs)
    print("Computed eigs :", test_eigs)

    print(rel_err)
    assert rel_err < tol, "❌ abs_eig_decomp failed"
    print("✅ abs_eig_decomp PASSED")

import numpy as np

key = np.random.default_rng(1)

N = 200
r = 8

Bk_vecs = jnp.array(key.normal(size=(N, r)))
Bk_scalars = jnp.array(key.normal(size=r))  # includes negative values

test_abs_eig_decomp(Bk_vecs, Bk_scalars)
