import jax.numpy as jnp
from SRC.DA_Comp.configs import KF_Opts
from create_results_dir import create_results_dir
import os
import numpy as np

def complex_to_real_concat(z: jnp.ndarray) -> jnp.ndarray:
    """Flatten z and return [Re(z), Im(z)] as a real 1D vector."""
    z = jnp.asarray(z)
    return jnp.concatenate([jnp.real(z).ravel(), jnp.imag(z).ravel()], axis=0)

def real_concat_to_complex(x: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse of complex_to_real_concat without needing the original shape.
    x is [Re(z), Im(z)] flattened; returns a FLAT complex vector z.
    (You can reshape z outside JIT.)
    """
    x = jnp.asarray(x)
    n = x.shape[0] // 2              # uses static shape metadata; JAX-safe
    re = x[:n]
    im = x[n:2*n]
    return re + 1j * im              # dtype promotes appropriately

def bilinear_sample_periodic(F, x, y, Lx, Ly):
    """
    Sample a 2D field F(iy, ix) defined on a periodic grid of shape (Ny, Nx)
    at continuous positions (x, y) in [0, Lx) x [0, Ly) using bilinear interpolation.

    Parameters
    ----------
    F : jnp.ndarray, shape (Ny, Nx), real or complex ok (real used here)
    x, y : jnp.ndarray, shape (P,)  particle positions
    Lx, Ly : float  domain lengths

    Returns
    -------
    vals : jnp.ndarray, shape (P,)
    """
    Ny, Nx = F.shape

    # map positions to fractional grid coordinates
    gx = (x / Lx) * Nx
    gy = (y / Ly) * Ny

    ix0 = jnp.floor(gx).astype(jnp.int32)
    iy0 = jnp.floor(gy).astype(jnp.int32)
    ix1 = (ix0 + 1) % Nx
    iy1 = (iy0 + 1) % Ny

    tx = gx - ix0.astype(gx.dtype)   # in [0,1)
    ty = gy - iy0.astype(gy.dtype)

    # gather four neighbors; note array indexing order is [iy, ix]
    f00 = F[iy0, ix0]
    f10 = F[iy0, ix1]
    f01 = F[iy1, ix0]
    f11 = F[iy1, ix1]

    # bilinear blend
    return ((1 - tx) * (1 - ty) * f00 +
            tx       * (1 - ty) * f10 +
            (1 - tx) * ty       * f01 +
            tx       * ty       * f11)

def load_data(kf_opts: KF_Opts):
    path = os.path.join(create_results_dir(), "Trjs", "KF_trjs", f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_dt={kf_opts.dt}_T={kf_opts.T}_n={kf_opts.n}", "trj.npy")
    start_idx = int(kf_opts.min_samp_T/kf_opts.dt)
    trj = np.load(path)[start_idx:, :]
    return trj

class Specteral_Upsampling:
    @classmethod
    def spectral_upsample_from_hat2d_rfft(cls, U_hat_r, r: int):
        """
        Periodic spectral upsampling by integer factor r using ONLY rfft/irfft domain ops.
        U_hat_r: (Ny, Nx_r) = rfft2 half-spectrum of a real field on (Ny, Nx), Nx_r = Nx//2+1.
        Returns: real field on (r*Ny, r*Nx) equal to the trigonometric interpolant.
        """
        Ny, Nx_r = U_hat_r.shape
        Nx = (Nx_r - 1) * 2
        Ry, Rx = r * Ny, r * Nx
        Rx_r = Rx // 2 + 1

        # 1) Centered pad along y (full complex axis)
        Uy_pad = cls._pad_centered_axis0(U_hat_r, Ry)   # shape (Ry, Nx_r)

        # 2) Right-pad along rfft axis x (nonnegative kx only)
        Uyr_pad = cls._pad_rfft_axis1_right(Uy_pad, Rx_r)  # shape (Ry, Rx_r)

        # 3) Inverse rfft2 to real grid and scale by r^2 (energy-consistent with zero-padding in k)
        f_hi = jnp.fft.irfft2(Uyr_pad, s=(Ry, Rx)) * (r * r)
        return f_hi.real
    
    @staticmethod
    def _pad_centered_axis0(F, Ry):
        """
        Centered zero-padding along the first axis (y).
        Works for complex arrays; preserves Hermitian symmetry per rfft column.
        """
        Ny = F.shape[0]
        if Ry == Ny:
            return F
        py1 = (Ry - Ny) // 2
        py2 = Ry - Ny - py1

        # center low-freqs along y
        Fy_shift = jnp.fft.fftshift(F, axes=(0,))

        # ✅ each axis needs a (before, after) pair; use (0, 0) not (0,)
        Fy_pad_shift = jnp.pad(
            Fy_shift,
            ((py1, py2), (0, 0)),
            mode='constant',
            constant_values=0.0,
        )

        # unshift back
        Fy_pad = jnp.fft.ifftshift(Fy_pad_shift, axes=(0,))
        return Fy_pad
   
    @staticmethod
    def _pad_rfft_axis1_right(F, Rx_r):
        """
        Zero-pad the rfft (last-axis) to the right (higher +kx only).
        Input F shape: (Ry, Nx_r_old). Output: (Ry, Rx_r).
        """
        Ry, Nx_r_old = F.shape
        if Rx_r == Nx_r_old:
            return F
        # put existing bins at identical nonnegative-kx indices; new higher-k bins are zero
        pad_cols = Rx_r - Nx_r_old
        return jnp.pad(F, ((0, 0), (0, pad_cols)), mode='constant', constant_values=0.0)
