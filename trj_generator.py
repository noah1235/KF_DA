import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
#from Solver.KF_intergrators import KF_PS_RHS, Time_Stepper, make_divergence_monitor, make_incompressible_ic, KF_LPT_PS_RHS, init_particles_vector
from SRC.Solver.trj_animation import animate_particles_and_flow
from SRC.Solver.IC_gen import init_particles_vector
from SRC.utils import Specteral_Upsampling
from create_results_dir import create_results_dir
from SRC.Solver.ploting import plot_vorticity, plot_div, plot_D_vs_time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from SRC.Solver.solver import KF_Stepper, Omega_Integrator, KF_TP_Stepper, create_vel_part_gen_fn
from jax import config
from multiprocessing import Pool, cpu_count
config.update("jax_enable_x64", True)
import multiprocessing as mp

#jax.config.update("jax_default_device", jax.devices("cpu")[0])
# or for GPU:
# jax.config.update("jax_default_device", jax.devices("gpu")[0])

def generate_rand_IC(NDOF, key_num=0, sigma=1.0, kcut_frac=0.15):
    """
    Random vorticity IC with energy concentrated at low wavenumbers.
    kcut_frac ~ 0.10-0.25 is a good range.
    """
    key = jax.random.PRNGKey(key_num)

    # white noise in physical space
    omega0 = sigma * jax.random.normal(key, (NDOF, NDOF))
    omega0 = omega0 - jnp.mean(omega0)

    omega0_hat = jnp.fft.rfft2(omega0)

    # wavenumbers for rfft2 shape (N, N//2+1)
    ky = jnp.fft.fftfreq(NDOF) * NDOF
    kx = jnp.fft.rfftfreq(NDOF) * NDOF
    KY, KX = jnp.meshgrid(ky, kx, indexing="ij")
    K2 = KX**2 + KY**2
    K = jnp.sqrt(K2)

    kcut = kcut_frac * (NDOF / 2)
    mask = (K <= kcut)

    omega0_hat = omega0_hat * mask
    return omega0_hat

def generate_KF_dataset():
    NDOF = 128
    Re = 100
    n  = 4
    dt = 1e-2
    T = 1e6
    T_samp = 100
    nsteps = int(T / dt)
    sample_steps = int(T_samp / dt)
    chunk_size = 100000

    omega0_hat = generate_rand_IC(NDOF)

    stepper = jax.jit(KF_Stepper(Re, n, NDOF, dt))
    integrator = Omega_Integrator(stepper)

    root = os.path.join(
        create_results_dir(),
        "Trjs",
        "KF_datasets",
        f"Re={Re}_NDOF={NDOF}_dt={dt}_n={n}_sampT={T_samp}_total_T={int(T)}"
    )
    os.makedirs(root, exist_ok=True)

    omega0_hat = integrator.fv_integrate(omega0_hat, sample_steps)
    #integrator.integrate_scan_checkpoint(omega0_hat, nsteps, chunk_size, os.path.join(root, "dataset.npy"))
    trj = integrator.integrate_scan(omega0_hat, sample_steps)
    np.save(os.path.join(root, "dataset.npy"), np.array(trj))

def generate_KF_diss_plots():
    max_workers = 4
    NDOF   = 256
    Re_list = [100]
    n      = 4
    dt     = 1e-2
    T      = 10.0
    T_warmup = 10.0
    plot_diss_vs_time = False
    nsteps_warmup = int(T_warmup / dt)
    nsteps = int(T / dt)
    key_num = 1
    omega0_hat_base = generate_rand_IC(NDOF, key_num=key_num)

    @jax.jit
    def get_diss_vs_time(Re):
        stepper = KF_Stepper(Re, n, NDOF, dt)
        integrator = Omega_Integrator(stepper)  
        D_lam  = Re / (2 * n**2)
        omega0_hat = integrator.fv_integrate(omega0_hat_base, nsteps_warmup)
        def body(omega_hat, _):
            omega_hat = stepper(omega_hat)
            u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega_hat)
            du__dx, dv__dy = jnp.fft.irfft2(stepper.NS.dxop * u_hat), jnp.fft.irfft2(stepper.NS.dyop * v_hat)
            du__dy, dv__dx = jnp.fft.irfft2(stepper.NS.dyop * u_hat), jnp.fft.irfft2(stepper.NS.dxop * v_hat)
            diss = (1.0 / Re) * jnp.mean(du__dx**2 + dv__dy**2 + du__dy**2 + dv__dx**2)
            return omega_hat, diss

        _, diss_list = jax.lax.scan(body, omega0_hat, xs=None, length=nsteps)
    
        diss_normalized_list = diss_list / D_lam
        avg_diss = jnp.mean(diss_list)
        return Re, diss_normalized_list, avg_diss
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_diss_vs_time, Re): Re for Re in Re_list}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda tup: Re_list.index(tup[0]))


    if plot_diss_vs_time:
        root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate", f"dvt_NDOF={NDOF}_dt={dt}_IC_seed={key_num}")
        os.makedirs(root, exist_ok=True)
        # --- Plot ---
        plt.figure(figsize=(6,4.8), dpi=120)
        t = np.linspace(0, T, nsteps)
        np.save(os.path.join(root, "t.npy"), t)
        for Re, diss_norm, _ in results:
            np.save(os.path.join(root, f"Re={Re}.npy"), np.array(diss_norm))
            plt.plot(t, diss_norm, label=f"Re={Re}", linewidth=2)

        plt.xlabel("Time")
        plt.ylabel("Dissipation / D_lam")
        plt.title(f"Dissipation Rate (N={NDOF}, dt={dt}, n={n})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()


        outpath = os.path.join(root, "diss_rate.png")
        plt.tight_layout()
        plt.savefig(outpath)
        print(f"Saved: {outpath}")
    else:
        root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate", f"avgd_NDOF={NDOF}_dt={dt}_IC_seed={key_num}")
        os.makedirs(root, exist_ok=True)
        Re_list = []
        avg_diss_list = []
        for Re, diss_norm, avg_diss in results:
            Re_list.append(Re)
            avg_diss_list.append(avg_diss)
        np.save(os.path.join(root, "Re_list.npy"), np.array(Re_list))
        np.save(os.path.join(root, "avg_diss_list.npy"), np.array(avg_diss_list))

def generate_sample_case_ani():
    NDOF = 128
    Re = 100
    beta = 0
    St = 0
    n  = 4
    dt = 1e-2
    T = 20
    nsteps = int(T/dt)
    n_particles = 100
    omega0_hat = generate_rand_IC(NDOF)
    stepper = KF_TP_Stepper(Re, n, NDOF, dt, St, beta, n_particles)
    u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
    u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)
    xp, yp, up, vp = init_particles_vector(n_particles, u, v, (0, stepper.NS.L), (0, stepper.NS.L), stepper.NS.L, rng=None)

    trj_gen_fn = create_vel_part_gen_fn(jax.jit(stepper), T)
    u_traj, v_traj, xp_traj, yp_traj = trj_gen_fn(omega0_hat, xp, yp, up, vp)
    root = os.path.join(create_results_dir(), "Trjs", "Animations", f"Re={Re}_St={St:.1e}_beta={beta:.1e}")
    os.makedirs(root, exist_ok=True)
    fig, anim = animate_particles_and_flow(u_traj, v_traj, xp_traj, yp_traj,
                    stepper.NS.L, NDOF,
                    interval=1, s=15, qskip=2,
                    repeat=True, blit=True, dpi=120, ax=None,
                    title="Particles + Velocity Field", skip=1
                )
    anim.save(os.path.join(root, "particles.mp4"), writer="ffmpeg", fps=60, dpi=150)




if __name__ == "__main__":
    generate_KF_diss_plots()