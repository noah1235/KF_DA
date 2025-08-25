import numpy as np
import jax.numpy as jnp
def make_incompressible_ic(Nx, Ny, Lx=2*np.pi, Ly=2*np.pi, amp=1e-3, kmax=4, seed=None):
    """
    Divergence-free IC returned in rfft2 layout: shape (2, Ny, Nx_r) with Nx_r=Nx//2+1.
    """
    rng = np.random.default_rng(seed)

    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2

    # random complex streamfunction spectrum ψ_hat with bandlimit
    mask = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax) & (K2 > 0)
    psi_hat = np.zeros((Ny, Nx), dtype=np.complex128)
    coeffs = rng.normal(size=mask.sum()) + 1j*rng.normal(size=mask.sum())
    psi_hat[mask] = coeffs

    # velocity in full spectral: u_hat = i ky ψ_hat, v_hat = -i kx ψ_hat
    u_hat_full =  1j * KY * psi_hat
    v_hat_full = -1j * KX * psi_hat

    # go to real space to normalize amplitude
    u = np.fft.ifft2(u_hat_full).real
    v = np.fft.ifft2(v_hat_full).real
    rms = np.sqrt(np.mean(u**2 + v**2))
    if rms > 0:
        u *= (amp / rms); v *= (amp / rms)

    # zero-mean (not strictly necessary for velocity, but fine)
    u -= u.mean(); v -= v.mean()

    # store in rfft layout
    #u_hat_r = np.fft.rfft2(u)
    #v_hat_r = np.fft.rfft2(v)
    U = np.stack((u, v), axis=0)      # (2, Ny, Nx_r)
    return U

def init_particles_vector(n, x_range, y_range, rng=None):
    """
    Initialize particle state vector:
    (x1, y1, u1, v1, x2, y2, u2, v2, ..., xN, yN, uN, vN).
    
    Parameters
    ----------
    n : int
        Number of particles.
    x_range : tuple(float, float)
        (min, max) range for x coordinates.
    y_range : tuple(float, float)
        (min, max) range for y coordinates.
    rng : np.random.Generator or None
        Random generator for reproducibility.
    
    Returns
    -------
    z : np.ndarray, shape (4*n,)
        State vector for all particles.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    xs = rng.uniform(x_range[0], x_range[1], size=n)
    ys = rng.uniform(y_range[0], y_range[1], size=n)
    us = np.zeros(n)
    vs = np.zeros(n)
    
    # Interleave into [x1,y1,u1,v1, x2,y2,u2,v2, ...]
    z = np.empty(4*n)
    z[0::4] = xs
    z[1::4] = ys
    z[2::4] = us
    z[3::4] = vs
    
    return z

