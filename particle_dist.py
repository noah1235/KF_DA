import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
#from Solver.KF_intergrators import KF_PS_RHS, Time_Stepper, make_divergence_monitor, make_incompressible_ic, KF_LPT_PS_RHS, init_particles_vector
from SRC.Solver.trj_animation import animate_particles_and_flow, animate_vorticity
from SRC.Solver.KF_intergrators import KF_PS_RHS, Time_Stepper, KF_LPT_PS_RHS, Maxey_Riley_RHS, RK4_Step, Time_Stepper
from SRC.Solver.IC_gen import init_particles_vector, make_incompressible_ic
from SRC.utils import Specteral_Upsampling, bilinear_sample_periodic
from create_results_dir import create_results_dir
from SRC.Solver.ploting import plot_vorticity, plot_div, plot_D_vs_time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from jax import config
from SRC.utils import load_data
from SRC.DA_Comp.configs import KF_Opts
from scipy.stats import norm, uniform

config.update("jax_enable_x64", True)

class rho_RHS:
    def __init__(self, KF_trj, maxey_riley_rhs, x, y, up, vp, kx, ky, ku, kv, L, KF_RHS):
        self.KF_trj = KF_trj
        self.x = x
        self.y = y
        self.up = up
        self.vp = vp

        self.dx_op = 1j * kx[:, None, None, None]  # (Nx, 1, 1, 1)
        self.dy_op = 1j * ky[None, :, None, None]  # (1, Ny, 1, 1)
        self.dup_op = 1j * ku[None, None, :, None]  # (1, 1, Nup, 1)
        self.dvp_op = 1j * kv[None, None, None, :]  # (1, 1, 1, Nvp)
        self.maxey_riley_rhs = maxey_riley_rhs
        self.r = 4
        self.L = L
        self.KF_RHS = KF_RHS


        self.x_4d, self.y_4d, self.up_4d, self.vp_4d = jnp.meshgrid(
        x, y, up, vp, indexing="ij"
    )

        
    def __call__(self, rho, i):
        """
        Compute d rho / dt at time index i using the Liouville equation:

            ∂_t ρ = -[ ∂_x(ρ v_x) + ∂_y(ρ v_y) + ∂_{v_x}(ρ g_x) + ∂_{v_y}(ρ g_y) ].

        Parameters
        ----------
        rho : array, shape (Nx, Ny, Nup, Nvp)
            Current PDF on the phase-space grid.
        i : int
            Time index into the precomputed Kolmogorov flow trajectory KF_trj.

        Returns
        -------
        drho_dt : array, same shape as rho
        """
        x_4d_flat = self.x_4d.reshape(-1)
        y_4d_flat = self.y_4d.reshape(-1)
        up_4d_flat = self.up_4d.reshape(-1)
        vp_4d_flat = self.vp_4d.reshape(-1)
        space_NDOF = self.x.shape[0]
        vel_NDOF = self.up.shape[0]
        # ----- 1. Pull out background flow snapshot -----
        U = self.KF_trj[i]  # shape e.g. (2, N_flow, N_flow)

        # upsample flow to a "fine" grid and sample on Liouville x,y grid
        u_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(
            jnp.fft.rfft2(U[0]), self.r
        )
        v_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(
            jnp.fft.rfft2(U[1]), self.r
        )



        u_xy = bilinear_sample_periodic(u_fine, x_4d_flat, y_4d_flat, self.L, self.L)
        v_xy = bilinear_sample_periodic(v_fine, x_4d_flat, y_4d_flat, self.L, self.L)
        _, u_t_field, v_t_field = self.KF_RHS(U.reshape(-1))
        g_x, g_y = self.maxey_riley_rhs(
                            u_xy, v_xy,
                            x_4d_flat, y_4d_flat,
                            up_4d_flat, vp_4d_flat,
                            u_t_field, v_t_field
                        )

        # ----- 4. Build fluxes in each direction -----
        F_x  = rho * self.up_4d   # ρ v_x
        F_y  = rho * self.vp_4d   # ρ v_y
        F_vx = rho * g_x.reshape((space_NDOF, space_NDOF, vel_NDOF, vel_NDOF))        # ρ g_x
        F_vy = rho * g_y.reshape((space_NDOF, space_NDOF, vel_NDOF, vel_NDOF))        # ρ g_y
        # ----- 5. Spectral divergence: ∇·F = ∂_x F_x + ∂_y F_y + ∂_{v_x} F_vx + ∂_{v_y} F_vy -----
        # FFTs of fluxes over all four axes
        F_x_hat  = jnp.fft.fftn(F_x,  axes=(0, 1, 2, 3))
        F_y_hat  = jnp.fft.fftn(F_y,  axes=(0, 1, 2, 3))
        F_vx_hat = jnp.fft.fftn(F_vx, axes=(0, 1, 2, 3))
        F_vy_hat = jnp.fft.fftn(F_vy, axes=(0, 1, 2, 3))

        # Multiply by ik in each direction, then sum in Fourier space
        div_hat = (
            self.dx_op  * F_x_hat  +
            self.dy_op  * F_y_hat  +
            self.dup_op * F_vx_hat +
            self.dvp_op * F_vy_hat
        )

        # Inverse FFT to get divergence in physical space
        div_F = jnp.fft.ifftn(div_hat, axes=(0, 1, 2, 3)).real

        # ----- 6. Liouville RHS: ∂_t ρ = - ∇·F -----
        drho_dt = -div_F
        return drho_dt

def check_rho_normalization(rho, dx, dy, du, dv):
    """
    Compute the quadruple integral of rho(x, y, u, v) over the full grid.

    Parameters
    ----------
    rho : jnp.ndarray
        Array of shape (Nx, Ny, Nu, Nv) containing samples of the *continuous* PDF.
    dx, dy : float
        Grid spacings in x and y.
    du, dv : float
        Grid spacings in u and v.

    Returns
    -------
    total_mass : float
        Approximate value of ∫ rho dx dy du dv (should be ~1 for a normalized PDF).
    """
    cell_volume = dx * dy * du * dv
    total_mass = jnp.sum(rho) * cell_volume
    return float(total_mass)


def integrate_rho_rk4(rho0, rho_rhs, dt, nsteps):
    """
    Integrate the Liouville equation forward in time using explicit RK4.

    Parameters
    ----------
    rho0 : array, shape (Nx, Ny, Nup, Nvp)
        Initial PDF at t = 0.
    rho_rhs : callable
        Instance of rho_RHS, called as rho_rhs(rho, i), where i is the
        time-step index into the precomputed background trajectory.
    dt : float
        Time step size.
    nsteps : int
        Number of time steps to take.

    Returns
    -------
    rho_trj : array, shape (nsteps+1, Nx, Ny, Nup, Nvp)
        Time history of rho, including the initial state.
    """
    rho = rho0
    traj = [rho0]

    for i in range(nsteps):
        # RK4 stages; we keep using the same time index "i" for the RHS,
        # matching the existing logic that uses KF_trj[i] inside rho_rhs.
        k1 = rho_rhs(rho, i)
        k2 = rho_rhs(rho + 0.5 * dt * k1, i)
        k3 = rho_rhs(rho + 0.5 * dt * k2, i)
        k4 = rho_rhs(rho + dt * k3, i)

        rho = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(rho)

    return jnp.stack(traj, axis=0)

def main():
    # --- particle & flow parameters ---
    beta = 0.0            # density ratio (Maxey–Riley)
    St = 1e-2             # Stokes number
    T = 1.0              # total physical time for KF trajectory

    # --- Kolmogorov flow options (for generating background flow trajectory) ---
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=32,
        dt=1e-2,
        T=1e3,
        min_samp_T=500,
        t_skip=1e-1,
    )
    attractor_snapshots = load_data(kf_opts)

    # --- Liouville grid resolution ---
    space_NDOF = 32       # number of grid points in x and y
    vel_NDOF = 32         # number of grid points in u_p and v_p
    p_vel_min = -6.0      # min particle velocity (periodic)
    p_vel_max =  6.0      # max particle velocity (periodic)
    vel_std = 2.0         # std of initial Gaussian in velocity space

    # --- Kolmogorov flow RHS / domain length ---
    rhs = KF_PS_RHS(kf_opts.NDOF, kf_opts.Re, kf_opts.n)
    L = rhs.L             # assume spatial domain is [0, L) x [0, L)

    # === Generate background flow trajectory U(t) for use in Maxey–Riley ===
    nsteps = int(T / kf_opts.dt)

    # load a snapshot from the Kolmogorov attractor as initial condition
    
    U0 = attractor_snapshots[0, :]

    # Maxey–Riley RHS for single-particle dynamics (if needed later)
    maxey_riley_rhs = Maxey_Riley_RHS(beta, St, L)

    # Time-stepper for the Navier–Stokes / Kolmogorov flow
    integrator = Time_Stepper(rhs, kf_opts.dt, method="RK4")

    # integrate flow equations to get trajectory in phase space
    trj = integrator.integrate_scan(U0, nsteps).reshape((nsteps+1, 2, kf_opts.NDOF, kf_opts.NDOF))

    # === Build phase-space grids (x, y, u_p, v_p) ===
    # Spatial grids: periodic on [0, L)
    x = jnp.linspace(0.0, L, space_NDOF, endpoint=False)
    y = jnp.linspace(0.0, L, space_NDOF, endpoint=False)

    # Velocity grids: periodic on [p_vel_min, p_vel_max)
    up = jnp.linspace(p_vel_min, p_vel_max, vel_NDOF, endpoint=False)
    vp = jnp.linspace(p_vel_min, p_vel_max, vel_NDOF, endpoint=False)

    # Grid spacings
    dspace = L / space_NDOF
    dvel = (p_vel_max - p_vel_min) / vel_NDOF

    # === Construct factorized initial PDF rho_0(x, y, u_p, v_p) ===
    rho_0_x = uniform.pdf(x, loc=0.0, scale=L)
    rho_0_y = uniform.pdf(x, loc=0.0, scale=L)

    # Gaussian in u_p, v_p (first as PDF, then convert to discrete probs)
    rho_0_up = jnp.array(norm.pdf(np.array(up), loc=0.0, scale=vel_std))
    rho_0_vp = jnp.array(norm.pdf(np.array(vp), loc=0.0, scale=vel_std))

    # Factorized 4D distribution: rho_0(x,y,u_p,v_p) = rho_x * rho_y * rho_up * rho_vp
    rho_0 = (
        rho_0_x[:, None, None, None] *
        rho_0_y[None, :, None, None] *
        rho_0_up[None, None, :, None] *
        rho_0_vp[None, None, None, :]
    )
    # rho_0 shape: (space_NDOF, space_NDOF, vel_NDOF, vel_NDOF)

    # Sanity check: discrete normalization over all dimensions
    print("Initial rho_0 total", check_rho_normalization(rho_0, dspace, dspace, dvel, dvel))

    # === Build wavenumber arrays for spectral derivatives in all 4 directions ===

    # spatial wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(space_NDOF, d=dspace)
    ky = 2.0 * np.pi * np.fft.fftfreq(space_NDOF, d=dspace)

    # velocity-space wavenumbers
    ku = 2.0 * np.pi * np.fft.fftfreq(vel_NDOF, d=dvel)
    kv = 2.0 * np.pi * np.fft.fftfreq(vel_NDOF, d=dvel)

    # === Instantiate Liouville stepper ===
    maxey_riley_rhs = Maxey_Riley_RHS(beta, St, rhs.L)
    KF_RHS = KF_PS_RHS(kf_opts.NDOF, kf_opts.Re, kf_opts.n, calc_mat_deriv=True)
    rho_rhs = rho_RHS(trj, maxey_riley_rhs, x, y, up, vp, kx, ky, ku, kv, L, KF_RHS)
    rho_rhs = jax.jit(rho_rhs)

    rho_trj = integrate_rho_rk4(rho_0, rho_rhs, kf_opts.dt, nsteps)
    print(rho_trj[-1])
    print(rho_trj[0].shape)


if __name__ == "__main__":
    main()