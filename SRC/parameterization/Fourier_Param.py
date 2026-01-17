from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp


@dataclass
class Fourier_Param:
    """
    Low-wavenumber truncation for 2D real FFTs (rfft2).

    Keeps:
      - ky: |ky| <= kmax via wrap-around on the full ky axis
            (i.e., [0..kmax] ∪ [N-kmax..N-1] when kmax < N//2)
      - kx: 0 <= kx <= kmax on the rfft2 half-spectrum axis

    For an NxN grid:
      rfft2 spectrum shape: (N, N//2 + 1)

    Stored truncated spectrum shape:
      ky_kept = N                 if kmax >= N//2 else (2*kmax + 1)
      kx_kept = (min(kmax, N//2) + 1) but capped by (N//2 + 1)
    """
    num_k: int

    # injected later
    vel_reshaper: Optional[object] = None
    div_free_proj: Optional[object] = None

    # cached after attach()
    _N: Optional[int] = None
    _kmax: Optional[int] = None
    _full_spec_shape: Optional[Tuple[int, int]] = None
    _ky_idx: Optional[jnp.ndarray] = None
    _kx_idx: Optional[jnp.ndarray] = None
    _kept_spec_shape: Optional[Tuple[int, int]] = None
    _flat_len_complex: Optional[int] = None

    def attach(self, vel_reshaper, div_free_proj):
        """
        Attach dependencies and precompute index sets/shapes that depend on N.
        Call this once before using transform/inv_transform.
        """
        self.vel_reshaper = vel_reshaper
        self.div_free_proj = div_free_proj

        N = int(self.vel_reshaper.NDOF)
        if N <= 0:
            raise ValueError(f"Invalid NDOF={N} from vel_reshaper.")

        kmax = int(min(int(self.num_k), N // 2))

        full_shape = (N, N // 2 + 1)
        Ny, Nx_half = full_shape

        # ky indices
        if kmax >= N // 2:
            ky_idx = jnp.arange(Ny)
        else:
            ky_idx = jnp.concatenate(
                [jnp.arange(0, kmax + 1), jnp.arange(Ny - kmax, Ny)],
                axis=0,
            )

        # kx indices (half-spectrum, so cap at Nx_half-1)
        kmax_x = int(min(kmax, Nx_half - 1))
        kx_idx = jnp.arange(0, kmax_x + 1)

        kept_shape = (int(ky_idx.size), int(kx_idx.size))
        flat_len_complex = 2 * kept_shape[0] * kept_shape[1]  # 2 components (u,v)

        # save caches
        self._N = N
        self._kmax = kmax
        self._full_spec_shape = full_shape
        self._ky_idx = ky_idx
        self._kx_idx = kx_idx
        self._kept_spec_shape = kept_shape
        self._flat_len_complex = flat_len_complex
        return self

    def _check_attached(self) -> None:
        if self.vel_reshaper is None or self.div_free_proj is None:
            raise RuntimeError("Call attach(vel_reshaper, div_free_proj) before using FourierParam.")
        if self._N is None:
            raise RuntimeError("Internal cache missing. Did attach() complete?")

    @property
    def N(self) -> int:
        self._check_attached()
        return int(self._N)

    @property
    def full_spec_shape(self) -> Tuple[int, int]:
        self._check_attached()
        return self._full_spec_shape  # type: ignore[return-value]

    @property
    def kept_spec_shape(self) -> Tuple[int, int]:
        self._check_attached()
        return self._kept_spec_shape  # type: ignore[return-value]

    @property
    def ky_idx(self) -> jnp.ndarray:
        self._check_attached()
        return self._ky_idx  # type: ignore[return-value]

    @property
    def kx_idx(self) -> jnp.ndarray:
        self._check_attached()
        return self._kx_idx  # type: ignore[return-value]

    @property
    def flat_len_realimag(self) -> int:
        """
        Length of the real vector returned by transform:
          concat(real(flat_complex), imag(flat_complex))
        """
        self._check_attached()
        return 2 * int(self._flat_len_complex)

    def transform(self, U_flat: jnp.ndarray) -> jnp.ndarray:
        """
        U_flat -> truncated Fourier vector (real/imag concatenated).

        Returns shape: (2 * ncomplex,) where ncomplex = 2*ky_kept*kx_kept
        (factor 2 for velocity components).
        """
        self._check_attached()

        # (2, N, N)
        U = self.vel_reshaper.reshape_flattened_vel(U_flat)

        # (2, N, N//2+1)
        U_hat = jnp.stack([jnp.fft.rfft2(U[0]), jnp.fft.rfft2(U[1])], axis=0)

        # (2, ky_kept, kx_kept)
        U_hat_small = U_hat[:, self.ky_idx, :][:, :, self.kx_idx]

        flat_c = U_hat_small.reshape(-1)  # complex
        return jnp.concatenate([flat_c.real, flat_c.imag], axis=0)

    def inv_transform(self, U_hat_flat: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse of transform:
          - rebuild complex truncated spectrum
          - scatter into full rfft2 spectrum (zero elsewhere)
          - optional div-free projection in Fourier space
          - return flattened spatial velocity
        """
        self._check_attached()

        if U_hat_flat.ndim != 1:
            raise ValueError(f"U_hat_flat must be 1D, got shape {U_hat_flat.shape}.")
        if U_hat_flat.size % 2 != 0:
            raise ValueError("U_hat_flat length must be even (real/imag concatenated).")

        n_half = U_hat_flat.size // 2
        small_flat_c = U_hat_flat[:n_half] + 1j * U_hat_flat[n_half:]

        ky_kept, kx_kept = self.kept_spec_shape
        expected_ncomplex = 2 * ky_kept * kx_kept
        if small_flat_c.size != expected_ncomplex:
            raise ValueError(
                f"Size mismatch: got {small_flat_c.size} complex coeffs, "
                f"expected {expected_ncomplex} (2*{ky_kept}*{kx_kept})."
            )

        U_hat_small = small_flat_c.reshape((2, ky_kept, kx_kept))

        Ny, Nx_half = self.full_spec_shape
        U_hat_full = jnp.zeros((2, Ny, Nx_half), dtype=U_hat_small.dtype)

        # scatter into full spectrum
        U_hat_full = U_hat_full.at[:, self.ky_idx[:, None], self.kx_idx[None, :]].set(U_hat_small)

        # project (assumed Fourier-space operator)
        u_hat, v_hat = self.div_free_proj(U_hat_full)

        # back to real space
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)
        U = jnp.stack([u, v], axis=0)

        return U.reshape(-1)
    
    def __repr__(self):
        return f"Fourier_n={self.num_k}"
