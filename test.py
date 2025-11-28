from SRC.iterative_methods import pcg_curve_detection, cg_curve_detection
import numpy as np


import jax
import jax.numpy as jnp


def test_pcg_curve_detection():
    print("\n================ PCG NEG-CURVE DETECTION TEST ================")

    # ----------------------------------------------------------
    # 1. SPD test (should converge)
    # ----------------------------------------------------------
    print("\n[TEST 1] SPD matrix -> should converge")

    n = 5
    # Make a random SPD matrix: A = Q diag(λ) Qᵀ
    key = jax.random.PRNGKey(0)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    lam = jnp.linspace(1.0, 5.0, n)
    A = Q @ (lam * Q.T)

    b = jnp.ones(n)

    matvec = lambda v: A @ v
    M_inv = jnp.eye(n)   # identity preconditioner

    p, flag = pcg_curve_detection(matvec, M_inv, b, max_iters=10)
    print("Flag:", flag)
    print("Solution:", p)
    print("Residual norm ‖A p − b‖ =", float(jnp.linalg.norm(A @ p - b)))

    # Assert we are close
    if flag == "converged" and jnp.linalg.norm(A @ p - b) < 1e-5:
        print(" --> PASS")
    else:
        print(" --> FAIL")

    return
    # ----------------------------------------------------------
    # 2. Indefinite test (should detect negative curvature)
    # ----------------------------------------------------------
    print("\n[TEST 2] Indefinite matrix -> should detect NC")

    # Make an indefinite matrix: A = Q diag(λ_pos..., λ_neg...) Qᵀ
    lam_indef = jnp.array([5.0, 4.0, -1.0, 2.0, -0.5])
    A2 = Q @ (lam_indef * Q.T)

    matvec2 = lambda v: A2 @ v
    b2 = jnp.ones(n)

    p2, flag2 = pcg_curve_detection(matvec2, M_inv, b2, max_iters=50)

    print("Flag:", flag2)
    print("Returned direction / curvature:", p2)

    if flag2 == "indef":
        print(" --> PASS")
    else:
        print(" --> FAIL")


    # ----------------------------------------------------------
    # 3. Max-iter test (matrix extremely ill-conditioned)
    # ----------------------------------------------------------
    print("\n[TEST 3] Ill-conditioned SPD -> expect max_iter")

    lam_bad = jnp.array([1e-6, 1e-3, 1e-2, 1, 5])
    A3 = Q @ (lam_bad * Q.T)

    matvec3 = lambda v: A3 @ v
    b3 = jnp.ones(n)

    p3, flag3 = pcg_curve_detection(matvec3, M_inv, b3, max_iters=3)
    print("Flag:", flag3)

    if flag3 == "max_iter":
        print(" --> PASS")
    else:
        print(" --> FAIL")

    print("\n==============================================================\n")

test_pcg_curve_detection()