import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
#from Solver.KF_intergrators import KF_PS_RHS, Time_Stepper, make_divergence_monitor, make_incompressible_ic, KF_LPT_PS_RHS, init_particles_vector
from kf_da.solver.trj_animation import animate_particles_and_flow
from kf_da.solver.IC_gen import init_particles_vector
from kf_da.utils.utils import Specteral_Upsampling
from kf_da.utils.create_results_dir import create_results_dir
from kf_da.solver.ploting import plot_vorticity, plot_div, plot_D_vs_time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from kf_da.solver.solver import KF_Stepper, Omega_Integrator, KF_TP_Stepper, create_vel_part_gen_fn
from jax import config
from multiprocessing import Pool, cpu_count
config.update("jax_enable_x64", True)
import multiprocessing as mp

#jax.config.update("jax_default_device", jax.devices("cpu")[0])
# or for GPU:
# jax.config.update("jax_default_device", jax.devices("gpu")[0])

def generate_rand_IC(NDOF, key_num=0, sigma=3, kcut_frac=0.1):
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
    with open("../kf-da-configs/genConfig.yaml") as f:
        config = yaml.safe_load(f)["config"]
        
    NDOF = config["NDOF"]
    Re = config["Re"]
    n  = 4
    dt = float(config["dt"])
    T = float(config["T"])
    T_samp = float(config["T_samp"])
    nsteps = int(T / dt)
    sample_steps = int(T_samp / dt)
    chunk_size = float(config["chunk_size"])
    omega0_hat = generate_rand_IC(NDOF)

    stepper = jax.jit(KF_Stepper(Re, n, NDOF, dt))
    integrator = Omega_Integrator(stepper)

    root = os.path.join(
        create_results_dir(),
        "Trjs",
        "KF_datasets",
        f"Re={Re}_NDOF={NDOF}_dt={dt}_n={n}_sampT={int(T_samp)}_total_T={int(T)}"
    )
    os.makedirs(root, exist_ok=True)

    omega0_hat = integrator.fv_integrate(omega0_hat, sample_steps)
    integrator.integrate_scan_checkpoint(omega0_hat, nsteps, chunk_size, os.path.join(root, "dataset.npy"))
    #trj = integrator.integrate_scan(omega0_hat, nsteps)
    #np.save(os.path.join(root, "dataset.npy"), np.array(trj))


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
    generate_KF_dataset()
