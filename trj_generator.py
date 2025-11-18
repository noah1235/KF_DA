import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
#from Solver.KF_intergrators import KF_PS_RHS, Time_Stepper, make_divergence_monitor, make_incompressible_ic, KF_LPT_PS_RHS, init_particles_vector
from SRC.Solver.trj_animation import animate_particles_and_flow, animate_vorticity
from SRC.Solver.KF_intergrators import KF_PS_RHS, Time_Stepper, KF_LPT_PS_RHS
from SRC.Solver.IC_gen import init_particles_vector, make_incompressible_ic
from SRC.utils import Specteral_Upsampling
from create_results_dir import create_results_dir
from SRC.Solver.ploting import plot_vorticity, plot_div, plot_D_vs_time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from jax import config
config.update("jax_enable_x64", True)

jax.config.update("jax_default_device", jax.devices("cpu")[0])
# or for GPU:
# jax.config.update("jax_default_device", jax.devices("gpu")[0])

def generate_KF_flow():
    NDOF = 32
    Re = 100
    n  = 4
    dt = 1e-2
    T = 1e3

    nsteps = int(T/dt)
    rhs = KF_PS_RHS(NDOF, Re, n)
    L = rhs.L
    U_0 = make_incompressible_ic(NDOF, NDOF, L, L, amp=5e-1).reshape(-1)
    integrator = Time_Stepper(rhs, dt, method="RK4")

    trj = integrator.integrate_scan(U_0, nsteps)

    root = os.path.join(create_results_dir(), "Trjs", "KF_trjs", f"Re={Re}_NDOF={NDOF}_dt={dt}_T={T}_n={n}")
    os.makedirs(root, exist_ok=True)
    trj = np.asarray(trj)
    U_T = trj[-1].reshape((2, NDOF, NDOF))
    U_hat = jnp.stack([jnp.fft.rfft2(U_T[0]), jnp.fft.rfft2(U_T[1])])
    oemga = rhs.vorticity_real(U_hat[0], U_hat[1])
    oemga = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(np.fft.rfft2(oemga), int(round(256/NDOF)))
    fig, _, _, _ = plot_vorticity(oemga)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "final_vort.png"))
    plt.close()

    div = jnp.fft.irfft2(rhs.dxop * U_hat[0] + rhs.dyop * U_hat[1], s=(NDOF, NDOF))
    fig, _, _, _ = plot_div(div, L, L)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "final_div.png"))
    plt.close()

    np.save(os.path.join(root, "trj.npy"), trj)

def generate_KF_energy_plots():
    NDOF   = 16
    Re_list = [8, 22, 40, 100]
    n      = 4
    dt     = 1e-2
    T      = 500.0
    T_trans = 100.0
    nsteps_trans = int(T_trans / dt)
    nsteps = int(T / dt)
    r      = 2  # upsample factor for gradient evaluation

    def run_one_re(Re):
        # Reference laminar scalings
        KE_lam = Re**2 / (4 * n**4)
        D_lam  = Re / (2 * n**2)

        rhs = KF_PS_RHS(NDOF, Re, n)
        L = rhs.L

        # Initial condition
        U_0 = make_incompressible_ic(NDOF, NDOF, L, L, amp=1e-1).reshape(-1)

        # Build integrator
        integrator = Time_Stepper(rhs, dt, method="RK4")

        # Pre-alloc (host numpy is fine for logging)
        KE_list   = np.zeros(nsteps, dtype=float)
        diss_list = np.zeros(nsteps, dtype=float)
        I_list    = np.zeros(nsteps, dtype=float)

        # Real-space forcing for injection integral (one-time)
        fx_real = jnp.fft.irfft2(rhs.f_hat_x, s=(NDOF, NDOF))

        def energy_callback(step, U):
            # step is 1-based in your snippet; store at step-1
            U = U.reshape((2, NDOF, NDOF))

            # velocities
            u = U[0]
            u_hat = jnp.fft.rfft2(u)
            v = U[1]
            v_hat = jnp.fft.rfft2(v)

            # KE (spatial mean)
            KE_list[step - 1] = 0.5 * float(jnp.mean(u**2 + v**2))

            # Gradients (upsampled real-space via your spectral-upsampling helper)
            ux = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(rhs.dxop * u_hat, r)
            uy = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(rhs.dyop * u_hat, r)
            vx = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(rhs.dxop * v_hat, r)
            vy = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(rhs.dyop * v_hat, r)

            # Dissipation (spatial mean of gradient form)
            diss = (1.0 / Re) * jnp.mean(ux**2 + uy**2 + vx**2 + vy**2)
            diss_list[step - 1] = float(diss)

            # Energy injection ⟨u · f⟩ with f = (sin(n y), 0)
            I_list[step - 1] = float(jnp.mean(u * fx_real))

        # Integrate
        _ = integrator.integrate(U_0, nsteps, callback=energy_callback)

        # Post-process: trim transients and nondimensionalize if desired
        # KE_trim = KE_list[nsteps_trans:] / KE_lam   # uncomment if you want KE/KE_lam
        diss_trim = diss_list[nsteps_trans:] / D_lam
        time_trim = jnp.linspace(T_trans, T, diss_trim.shape[0])

        return Re, np.asarray(time_trim), np.asarray(diss_trim)

    # --- Run in parallel threads ---
    max_workers = min(len(Re_list), os.cpu_count() or 4)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one_re, Re): Re for Re in Re_list}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Keep the curves ordered by Re_list
    results.sort(key=lambda tup: Re_list.index(tup[0]))

    # --- Plot ---
    plt.figure(figsize=(6,4.8), dpi=120)
    for Re, t, diss in results:
        plt.plot(t, diss, label=f"Re={Re}", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Dissipation / D_lam")
    plt.title(f"Dissipation Rate (N={NDOF}, dt={dt}, n={n})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate", f"NDOF={NDOF}_dt={dt}")
    os.makedirs(root, exist_ok=True)
    outpath = os.path.join(root, "diss_rate.png")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved: {outpath}")

def generate_sample_case_ani():
    NDOF = 64
    Re = 100
    beta = 0
    St = 0
    n  = 4
    dt = 2e-2
    T = 50
    vort=False

    n_particles = 100
    nsteps = int(T/dt)
    rhs = KF_LPT_PS_RHS(NDOF, Re, n, n_particles, beta=beta, St=St)
    L = rhs.KF_RHS.L
    U_0 = make_incompressible_ic(NDOF, NDOF, L, L, amp=2).reshape(-1)
    particles = init_particles_vector(n_particles, U_0.reshape((2, NDOF, NDOF)), (0, L), (0, L), L, rng=None)
    X0 = jnp.concat([particles, U_0])

    integrator = Time_Stepper(rhs, dt, method="RK4", n_particles=n_particles)

    trj = integrator.integrate_scan(X0, nsteps)
    root = os.path.join(create_results_dir(), "Trjs", "Animations", f"Re={Re}_St={St:.1e}_beta={beta:.1e}")
    os.makedirs(root, exist_ok=True)
    fig, anim = animate_particles_and_flow(
                    trj, L, n_particles, NDOF,
                    interval=1, s=15, qskip=2,
                    repeat=True, blit=True, dpi=120, ax=None,
                    title="Particles + Velocity Field", skip=1
                )
    anim.save(os.path.join(root, "particles.mp4"), writer="ffmpeg", fps=60, dpi=150)

    if vort:
        vel_hat_trj = trj[:, n_particles*4:]
        fig, anim = animate_vorticity(vel_hat_trj, NDOF, L, L, rhs.KF_RHS.vorticity_real,
                        cmap="icefire", interval=1, repeat=True, blit=False,
                        dpi=120, skip=1, cbar=True, sym=True, clim=None, ax=None,
                        title=r"Vorticity $\omega_z$")
        anim.save(os.path.join(root, "vorticity.mp4"), writer="ffmpeg", fps=10, dpi=300)

if __name__ == "__main__":
    generate_sample_case_ani()