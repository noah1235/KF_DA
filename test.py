import jax
import jax.numpy as jnp


# ------------------------------------------------------------
# Random orthogonal matrices (Haar via QR)
# ------------------------------------------------------------
def random_orthogonal_matrices(key, N, d, dtype=jnp.float32):

    keys = jax.random.split(key, N)

    def single(k):
        M = jax.random.normal(k, (d, d), dtype=dtype)
        Q, R = jnp.linalg.qr(M)

        # Fix QR sign ambiguity → Haar distribution
        s = jnp.sign(jnp.diag(R))
        s = jnp.where(s == 0, 1.0, s)
        Q = Q * s[None, :]
        return Q

    return jax.vmap(jax.jit(single))(keys)


# ------------------------------------------------------------
# Monte-Carlo verification
# ------------------------------------------------------------
def verify_expectation(key, N_samples, d):

    Qs = random_orthogonal_matrices(key, N_samples, d)

    # First column of each Q
    q1 = Qs[:, :, 0]            # shape (N_samples, d)

    # Square entries
    sq = q1 ** 2

    # empirical E[(Q_{i1})^2] for each i
    mean_per_i = jnp.mean(sq, axis=0)

    # overall mean (should also be 1/d)
    mean_all = jnp.mean(sq)

    return mean_per_i, mean_all


# ------------------------------------------------------------
# Run test
# ------------------------------------------------------------
key = jax.random.PRNGKey(0)

N_samples = 50000
d = 16

mean_per_i, mean_all = verify_expectation(key, N_samples, d)

print("Expected value 1/d =", 1/d)
print("Mean over all entries =", mean_all)
print("Per-coordinate means =", mean_per_i)
print("d * mean_all =", d * mean_all)