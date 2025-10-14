import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# rfft2 output shape is (Nx, Ny//2 + 1)

N = 128
L = 2*jnp.pi
kx = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi))
ky = jnp.fft.fftfreq(N, d=L / (N * 2 * jnp.pi))
KX, KY = jnp.meshgrid(kx, ky, indexing='xy')
K2 = KX**2 + KY**2

# ---------- plotting ----------
def plot_velocity_field(u, step=2, scale=None, color_by_magnitude=True, title=None):
    """
    Vector-plot a 2D velocity field u[..., 0], u[..., 1] using quiver.

    Parameters
    ----------
    u : array (Nx, Ny, 2)
    step : int          Subsampling stride for arrows
    scale : float|None  Matplotlib quiver scale
    color_by_magnitude : bool  Color arrows by |u| if True, else solid black
    title : str|None
    """
    Nx, Ny, _ = u.shape
    x = jnp.arange(Nx)
    y = jnp.arange(Ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Subsample for clarity
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    U = u[::step, ::step, 0]
    V = u[::step, ::step, 1]

    fig, ax = plt.subplots(figsize=(6, 6))

    if color_by_magnitude:
        mag = jnp.sqrt(U**2 + V**2)
        Q = ax.quiver(
            Xs, Ys, U, V, mag,
            scale=scale, pivot="middle", cmap="viridis",
            width=0.003, headwidth=3,
        )
        cbar = plt.colorbar(Q, ax=ax, orientation="vertical")
        cbar.set_label("|u|", fontsize=12)
    else:
        Q = ax.quiver(
            Xs, Ys, U, V,
            scale=scale, pivot="middle",
            width=0.003, headwidth=3,
            color="black",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x grid index")
    ax.set_ylabel("y grid index")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ---------- random velocity field ----------
def get_rand_hat(key, shape):
    """Random complex Fourier coefficients ~ N(0,1) for real & imag."""
    a = random.normal(key, shape=shape)
    b = random.normal(key, shape=shape)
    return a + 1j * b

def div_free_proj(U):
    u = U[..., 0]
    v = U[..., 1]

    u_hat = jnp.fft.rfft2(u)
    v_hat = jnp.fft.rfft2(v)

    invK2 = jnp.where(K2 > 0.0, 1.0 / K2, 0.0)

    k_dot_U = KX * u_hat + KY * v_hat

    Ux_proj = u_hat - KX * k_dot_U * invK2
    Uy_proj = v_hat - KY * k_dot_U * invK2

    ux = jnp.fft.irfft2(Ux_proj, s=u.shape[:2])
    uy = jnp.fft.irfft2(Uy_proj, s=u.shape[:2])
    return jnp.stack([ux, uy], axis=-1)


def check_divergence(U):
    u_hat = jnp.fft.rfft2(U[:, :, 0])
    v_hat = jnp.fft.rfft2(U[:, :, 1])


    div = jnp.fft.irfft2(1j * KX* u_hat + 1j * KY * v_hat)
    max_div = jnp.max(jnp.abs(div))
    print(max_div)

def main(spectral_width=10.0):
    # RNG
    key = random.PRNGKey(0)
    key_u, key_v = random.split(key)


    # Smooth spectral envelope
    env = jnp.exp(-K2 / (spectral_width**2))

    # Random spectra (smoothed)
    u_hat = get_rand_hat(key_u, KX.shape) * env
    v_hat = get_rand_hat(key_v, KX.shape) * env


    # Back to real space, explicit size for safety
    u = jnp.fft.irfft2(u_hat, s=(N, N))
    v = jnp.fft.irfft2(v_hat, s=(N, N))

    # Stack with velocity on the LAST axis -> shape (N, N, 2)
    U = jnp.stack([u, v], axis=-1)

    #plot_velocity_field(U, step=2, title="Random 2D velocity field")
    check_divergence(U)
    check_divergence(div_free_proj(U))

if __name__ == "__main__":
    main()
