import jax.numpy as jnp
from SRC.DA_Comp.configs import KF_Opts
from create_results_dir import create_results_dir
import os
import numpy as np
import jax

class Vel_Reshaper:
    def __init__(self, NDOF):
        self.NDOF = NDOF
    
    def reshape_flattened_vel(self, U_flat):
        return U_flat.reshape((2, self.NDOF, self.NDOF))
    
    def get_vel_hat_from_flat(self, U_flat):
        U = self.reshape_flattened_vel(U_flat)
        u_hat = jnp.fft.rfft2(U[0])
        v_hat = jnp.fft.rfft2(U[1])

        U_hat = jnp.stack([u_hat, v_hat], axis=0)

        return U_hat, U
    
    def vel_flat_2_vel_Fourier(self, U_flat):
        U_hat, _ = self.get_vel_hat_from_flat(U_flat)
        U_hat_flat = U_hat.reshape(-1)
        
        U_fourier = np.concatenate([U_hat_flat.real, U_hat_flat.imag])
        return U_fourier
    
    
    def vel_Fourier_2_vel_hat(self, U_fourier):
        """
        Inverse of vel_flat_2_vel_Fourier:
        Convert concatenated real vector back to complex Fourier representation.
        """
        n_half = U_fourier.size // 2
        real_part = U_fourier[:n_half]
        imag_part = U_fourier[n_half:]

        # Recombine into complex vector
        U_hat_flat = real_part + 1j * imag_part
        U_hat = U_hat_flat.reshape((2, self.NDOF, self.NDOF//2 + 1))
        return U_hat

    
    @staticmethod
    def flatten_from_comps(u, v):
        return jnp.stack([u, v], axis=0).reshape(-1)
    

class Vel_Part_Transformations(Vel_Reshaper):
    def __init__(self, NDOF, n_particles):
        super().__init__(NDOF)
        self.n_particles = n_particles

    def split_part_and_vel(self, X):
        part = X[:self.n_particles * 4]
        U_flat = X[self.n_particles * 4:]
        return part, U_flat
    
    @staticmethod
    def get_part_pos_and_vel(part):
        xp = part[0::4]
        yp = part[1::4]
        up = part[2::4]
        vp = part[3::4]

        return xp, yp, up, vp

def build_hvp(f, x):
    
    @jax.jit
    def hvp(v):
        return jax.jvp(jax.grad(f), (x,), (v,))[1]
    
    return hvp

def build_div_free_proj(stepper):
    NDOF = stepper.step.rhs.KF_RHS.N
    KX = stepper.step.rhs.KF_RHS.KX
    KY = stepper.step.rhs.KF_RHS.KY
    K2 = stepper.step.rhs.KF_RHS.K2
    M = stepper.step.rhs.KF_RHS.M
    

    def transform_fn(U_hat):
        X_proj = project_divfree_rfft2(U_hat, KX, KY, K2, M)
        X = X_proj.reshape(-1)
        return X
    
    return transform_fn

def project_divfree_rfft2(U_hat, KX, KY, K2, M, zero_dc=True):
    # rFFT of components
    #Ux = jnp.fft.rfft2(U[0]) * M
    #Uy = jnp.fft.rfft2(U[1]) * M
    Ux = U_hat[0] * M
    Uy = U_hat[1] * M

    # Longitudinal scale = (k·U)/|k|^2  (zero at DC)
    invK2 = jnp.where(K2 > 0.0, 1.0 / K2, 0.0)
    k_dot_U = KX * Ux + KY * Uy
    scale = k_dot_U * invK2

    # Helmholtz projection in k-space
    Ux_proj = Ux - KX * scale
    Uy_proj = Uy - KY * scale

    if zero_dc:
        Ux_proj = Ux_proj.at[0, 0].set(0.0)
        Uy_proj = Uy_proj.at[0, 0].set(0.0)

    u = jnp.fft.irfft2(Ux_proj)
    v = jnp.fft.irfft2(Uy_proj)

    return jnp.stack([u, v], axis=0)

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
