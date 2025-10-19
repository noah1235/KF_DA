import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ---------- 1. Weibull defined by single (x_thr, q_thr) ----------
def weibull_thresh_prob(x, x_thr=-1e-6, q_thr=0.5, k=2):
    lam = (-x_thr) / (-jnp.log1p(-q_thr))**(1.0/k)
    t = jnp.maximum(0.0, -x) / lam
    return 1.0 - jnp.exp(-jnp.power(t, k))




# ---------- Plot setup ----------
x = np.linspace(-1e-5, 1e-7, 500)

y1 = weibull_thresh_prob(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y1, label="One-quantile (x_thr=−1e−6, q=0.9, k=1.5)", lw=2)
ax.axvline(-1e-6)
ax.axvline(0, color='gray', linestyle=':', lw=1)
ax.set_xlabel("x")
ax.set_ylabel("Probability")
ax.set_title("Weibull / cloglog CDF near negative threshold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
