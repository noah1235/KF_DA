import jax.numpy as jnp

class Fourier_Param:
    def __init__(self, Nx: int, K: int):
        self.Nx = int(Nx)
        self.K = int(K)

        if self.K < 0:
            raise ValueError("K must be nonnegative.")
        if self.K > self.Nx // 2:
            raise ValueError(f"K={self.K} must be <= Nx//2={self.Nx//2}.")

        # rfft2 spectrum shape for an (Nx, Nx) real field
        self.full_shape = (self.Nx, self.Nx // 2 + 1)

        # kx is stored only for nonnegative modes in rfft2
        self.kx_idx = jnp.arange(self.K + 1)  # (nkx,)

        # ky uses wrap-around indexing to represent [-K..K]
        if (self.Nx % 2 == 0) and (self.K == self.Nx // 2):
            # keeping all ky (Nyquist edge); avoids duplicates/oddities
            self.ky_idx = jnp.arange(self.Nx)
        else:
            ky_pos = jnp.arange(self.K + 1)              # 0..K
            ky_neg = jnp.arange(self.Nx - self.K, self.Nx)  # Nx-K..Nx-1
            self.ky_idx = jnp.concatenate([ky_pos, ky_neg], axis=0)

        self.nky = int(self.ky_idx.shape[0])
        self.nkx = int(self.kx_idx.shape[0])
        self.small_shape = (self.nky, self.nkx)

        self.nn = self.nky * self.nkx
        self.out_dim = 2 * self.nn

        # cached broadcasted indices (useful for gather/scatter)
        self._KY = self.ky_idx[:, None]  # (nky, 1)
        self._KX = self.kx_idx[None, :]  # (1, nkx)

    def transform(self, omega_hat: jnp.ndarray) -> jnp.ndarray:
        """
        omega_hat: complex array, shape (Nx, Nx//2+1)
        returns: real array, shape (2*nky*nkx,)
        """
        if omega_hat.shape != self.full_shape:
            raise ValueError(f"Expected omega_hat shape {self.full_shape}, got {omega_hat.shape}")

        # Ensure we're packing from a complex array
        #omega_hat = omega_hat.astype(jnp.complex128, copy=False)

        U_small = omega_hat[self._KY, self._KX]     # (nky, nkx) complex
        flat = U_small.reshape(-1)                  # (nn,) complex

        # Pack into real vector
        return jnp.concatenate([flat.real, flat.imag], axis=0)

    def inv_transform(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z: real array, shape (2*nky*nkx,)
        returns: complex128 rfft2 spectrum, shape (Nx, Nx//2+1)
        """
        z = jnp.asarray(z)
        if z.shape != (self.out_dim,):
            raise ValueError(f"Expected z shape ({self.out_dim},), got {z.shape}")
        if z.dtype == jnp.float32:
            c_dtype = jnp.complex64
        elif z.dtype == jnp.float64:
            c_dtype = jnp.complex128
        else:
            raise TypeError(f"Unsupported dtype: {z.dtype}")

        # Work in float64 so complex128 comes out cleanly
        #z = z.astype(jnp.float64, copy=False)

        re = z[: self.nn].reshape(self.small_shape)
        im = z[self.nn :].reshape(self.small_shape)
        U_small = re + 1j * im                      # complex128

        omega_hat_full = jnp.zeros(self.full_shape, dtype=c_dtype)
        omega_hat_full = omega_hat_full.at[self._KY, self._KX].set(U_small)
        return omega_hat_full

    def __repr__(self) -> str:
        return f"Fourier_K={self.K}"
