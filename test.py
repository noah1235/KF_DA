from SRC.iterative_methods import pcg
import numpy as np


# ---------------------------------------------------------------
# Build a random symmetric positive-definite matrix A
# ---------------------------------------------------------------
n = 20
rng = np.random.default_rng(0)

Q = rng.standard_normal((n, n))
A = Q.T @ Q + 0.1 * np.eye(n)    # SPD

# matvec wrapper
def matvec(v):
    return A @ v

# ---------------------------------------------------------------
# Preconditioner: approximate A^{-1}
# ---------------------------------------------------------------
# Use diagonal of A as simple Jacobi preconditioner
M = np.diag(1.0 / np.diag(A))

# ---------------------------------------------------------------
# RHS vector (e.g. minus gradient)
# ---------------------------------------------------------------
grad = rng.standard_normal(n)
b = -grad

# ---------------------------------------------------------------
# Call PCG
# ---------------------------------------------------------------
x, info = pcg(matvec, M, b, maxiter=100)

# ---------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------
print("Converged in:", info["num_iter"])
print("Residual norm:", info["res_norm"])


# Check accuracy: A x ≈ b
res = np.linalg.norm(A @ x - b)
print("||A x - b|| =", res)
