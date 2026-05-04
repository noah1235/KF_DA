import jax.numpy as jnp
from kf_da.utils.create_results_dir import create_results_dir 
#from kf_da.daComp.configs import KF_Opts
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
    KX = stepper.step.rhs.KF_RHS.KX
    KY = stepper.step.rhs.KF_RHS.KY
    K2 = stepper.step.rhs.KF_RHS.K2

    def transform_fn(U_hat, M=stepper.step.rhs.KF_RHS.M):
        return project_divfree_rfft2(U_hat, KX, KY, K2, M)
    
    return transform_fn

def project_divfree_rfft2(U_hat, KX, KY, K2, M):
    # rFFT of components
    Ux = U_hat[0]
    Uy = U_hat[1]
    if M is not None:
        Ux = Ux * M
        Uy = Uy * M

    # Longitudinal scale = (k·U)/|k|^2  (zero at DC)
    invK2 = jnp.where(K2 > 0.0, 1.0 / K2, 0.0)
    k_dot_U = KX * Ux + KY * Uy
    scale = k_dot_U * invK2

    # Helmholtz projection in k-space
    Ux_proj = Ux - KX * scale
    Uy_proj = Uy - KY * scale
    return Ux_proj, Uy_proj




def bilinear_sample_periodic(F, x, y, Lx, Ly):
    Ny, Nx = F.shape

    # map positions to fractional grid coordinates
    gx = (x / Lx) * Nx
    gy = (y / Ly) * Ny

    ix0 = jnp.floor(gx).astype(jnp.int32)
    iy0 = jnp.floor(gy).astype(jnp.int32)
    ix1 = (ix0 + 1) % Nx
    iy1 = (iy0 + 1) % Ny

    tx = gx - ix0.astype(gx.dtype)  
    ty = gy - iy0.astype(gy.dtype)

    f00 = F[iy0, ix0]
    f10 = F[iy0, ix1]
    f01 = F[iy1, ix0]
    f11 = F[iy1, ix1]

    return ((1 - tx) * (1 - ty) * f00 +
            tx       * (1 - ty) * f10 +
            (1 - tx) * ty       * f01 +
            tx       * ty       * f11)


def _cubic_kernel(t, a=-0.5):
    """
    Keys cubic convolution kernel.
    t can be any real array. Returns weights same shape as t.
    """
    t = jnp.abs(t)
    t2 = t * t
    t3 = t2 * t

    w0 = (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0              # |t| < 1
    w1 = a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a      # 1 <= |t| < 2
    return jnp.where(t < 1.0, w0, jnp.where(t < 2.0, w1, 0.0))

def bilinear_sample_periodic_dec(F: jnp.ndarray,
                            x: jnp.ndarray,
                            y: jnp.ndarray,
                            Lx: float,
                            Ly: float,
                            a: float = -0.5):
    """
    Periodic bicubic interpolation of F[iy, ix] on a uniform grid.

    Assumes samples at x_i = i*Lx/Nx, y_j = j*Ly/Ny (corner grid).
    x,y can be any real values; periodic wrap is applied.

    Returns: values with shape broadcast(x,y) (typically (P,))
    """
    Ny, Nx = F.shape
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    # Periodize positions to [0, L)
    x = jnp.mod(x, Lx)
    y = jnp.mod(y, Ly)

    # Continuous grid coordinates
    gx = x * (Nx / Lx)
    gy = y * (Ny / Ly)

    ix = jnp.floor(gx).astype(jnp.int32)
    iy = jnp.floor(gy).astype(jnp.int32)

    tx = gx - ix.astype(gx.dtype)  # in [0,1)
    ty = gy - iy.astype(gy.dtype)

    # Neighbor indices: i-1, i, i+1, i+2 (periodic)
    ixm1 = (ix - 1) % Nx
    ix0  = ix % Nx
    ixp1 = (ix + 1) % Nx
    ixp2 = (ix + 2) % Nx

    iym1 = (iy - 1) % Ny
    iy0  = iy % Ny
    iyp1 = (iy + 1) % Ny
    iyp2 = (iy + 2) % Ny

    # Weights in x and y for the four neighbors
    # Distances to the four sample points relative to cell origin:
    # x: -1, 0, 1, 2 corresponds to t = tx - offset
    wxm1 = _cubic_kernel(tx + 1.0, a)
    wx0  = _cubic_kernel(tx + 0.0, a)
    wxp1 = _cubic_kernel(tx - 1.0, a)
    wxp2 = _cubic_kernel(tx - 2.0, a)

    wym1 = _cubic_kernel(ty + 1.0, a)
    wy0  = _cubic_kernel(ty + 0.0, a)
    wyp1 = _cubic_kernel(ty - 1.0, a)
    wyp2 = _cubic_kernel(ty - 2.0, a)

    # Gather 4x4 stencil values with explicit advanced indexing (shape-safe)
    # Row -1
    rym1 = (wxm1 * F[iym1, ixm1] + wx0 * F[iym1, ix0] +
            wxp1 * F[iym1, ixp1] + wxp2 * F[iym1, ixp2])
    # Row 0
    ry0  = (wxm1 * F[iy0,  ixm1] + wx0 * F[iy0,  ix0] +
            wxp1 * F[iy0,  ixp1] + wxp2 * F[iy0,  ixp2])
    # Row +1
    ryp1 = (wxm1 * F[iyp1, ixm1] + wx0 * F[iyp1, ix0] +
            wxp1 * F[iyp1, ixp1] + wxp2 * F[iyp1, ixp2])
    # Row +2
    ryp2 = (wxm1 * F[iyp2, ixm1] + wx0 * F[iyp2, ix0] +
            wxp1 * F[iyp2, ixp1] + wxp2 * F[iyp2, ixp2])

    # Now interpolate in y
    val = (wym1 * rym1 + wy0 * ry0 + wyp1 * ryp1 + wyp2 * ryp2)
    return val


def load_data_dec(kf_opts: KF_Opts):
    path = os.path.join(create_results_dir(), "Trjs", "KF_datasets", f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_dt={kf_opts.dt}_n={kf_opts.n}_sampT={kf_opts.min_samp_T}_total_T={kf_opts.total_T}", "dataset.npy")
    skip = int(kf_opts.t_skip / kf_opts.dt)
    trj = np.load(path)[::skip, :]
    return trj

def load_data(kf_opts: KF_Opts):
    path = os.path.join(
        create_results_dir(),
        "Trjs",
        "KF_datasets",
        f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_dt={kf_opts.dt}_n={kf_opts.n}_sampT={kf_opts.min_samp_T}_total_T={kf_opts.total_T}",
        "dataset.npy",
    )

    skip = int(kf_opts.t_skip / kf_opts.dt)

    trj = np.load(path, mmap_mode="r")  # memory-mapped array
    trj = trj[::skip, :]                # lazy slice (does not load full array)

    return trj

def is_jitted(fn):
    return hasattr(fn, "lower")

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
