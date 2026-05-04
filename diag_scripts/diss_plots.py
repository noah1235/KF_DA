import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from ..create_results_dir import..create_results_dir
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from SRC.Solver.solver import KF_Stepper, Omega_Integrator, KF_TP_Stepper,..create_vel_part_gen_fn
from jax import config
from multiprocessing import Pool, cpu_count
config.update("jax_enable_x64", True)
import multiprocessing as mp
from trj_generator import generate_rand_IC

def generate_KF_diss_plots():
    max_workers = 4
    #8, 22, 40, 100
    cases = [
        # (Re, NDOF, dt)
        #(8,  128, 1e-2),
        #(22,  128, 1e-2),
        #(40,  128, 1e-2),
        #(60,  128, 1e-2),
        #(80,  128, 1e-2),
        #(100,  128, 1e-2),
        #(120, 128, 1e-2),
        #(140, 128, 1e-2),
        #(160, 128, 1e-2),
        #(180, 128, 1e-2),
        #(200, 256, 2.5e-3),
        #(220, 256, 2.5e-3),
        (300, 512, 1e-3),
        (400, 512, 1e-3)
    ]

    n = 4
    T = 10000
    T_warmup = 100.0

    key_num = 1


    def run_case(Re, NDOF, dt):
        nsteps_warmup = int(T_warmup / dt)
        nsteps = int(T / dt)

        omega0_hat_base = generate_rand_IC(NDOF, key_num=key_num)

        # Create stepper/integrator OUTSIDE jit
        stepper = jax.jit(KF_Stepper(Re, n, NDOF, dt))
        integrator = Omega_Integrator(stepper)
        D_lam = Re / (2 * n**2)

        omega0_hat = integrator.fv_integrate(omega0_hat_base, nsteps_warmup)

        def scan_traj(omega_hat0):
            @jax.jit
            def body(omega_hat, _):
                omega_hat = stepper(omega_hat)

                u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega_hat)

                du_dx = jnp.fft.irfft2(stepper.NS.dxop * u_hat)
                du_dy = jnp.fft.irfft2(stepper.NS.dyop * u_hat)
                dv_dx = jnp.fft.irfft2(stepper.NS.dxop * v_hat)
                dv_dy = jnp.fft.irfft2(stepper.NS.dyop * v_hat)

                diss = (1.0 / Re) * jnp.mean(du_dx**2 + dv_dy**2 + du_dy**2 + dv_dx**2)
                return omega_hat, diss

            _, diss_list = jax.lax.scan(body, omega_hat0, xs=None, length=nsteps)
            return diss_list

        # optional debug print (outside jit, or use jax.debug inside scan_traj if needed)
        print(f"Running Re={Re}, NDOF={NDOF}, dt={dt}")

        diss_list = scan_traj(omega0_hat)
        diss_norm = diss_list / D_lam
        avg_diss = jnp.mean(diss_list)

        # move small scalars to host to make saving easy
        return (Re, NDOF, dt, np.array(diss_norm), float(avg_diss))

    # parallel over cases
    using_gpu = (jax.default_backend() == "gpu")
    print("JAX backend:", jax.default_backend())

    results = []

    if using_gpu:
        # GPU
        for (Re, NDOF, dt) in cases:
            results.append(run_case(Re, NDOF, dt))
    else:
        # CPU
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(run_case, Re, NDOF, dt): (Re, NDOF, dt)
                for (Re, NDOF, dt) in cases
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # Sort by Re
    results.sort(key=lambda x: x[0])

    root = os.path.join(
       ..create_results_dir(), "Trjs", "Dissipation_Rate",
        f"diss_v_time_IC_seed={key_num}"
    )
    os.makedirs(root, exist_ok=True)

    plt.figure(figsize=(6, 4.8), dpi=120)

    for Re, NDOF, dt, diss_norm, _ in results:
        nsteps = int(T / dt)
        t = np.linspace(0.0, T, nsteps, endpoint=False)
        np.save(os.path.join(root, f"t_Re={Re}_N={NDOF}_dt={dt}.npy"), t)
        np.save(os.path.join(root, f"dissnorm_Re={Re}_N={NDOF}_dt={dt}.npy"), diss_norm)
        plt.plot(t, diss_norm, label=f"Re={Re} (N={NDOF}, dt={dt})", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel(r"$D(t) / D_{\mathrm{lam}}$")
    plt.title(f"Dissipation Rate (n={n})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8)

    outpath = os.path.join(root, "diss_rate.png")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved: {outpath}")
    


if __name__ == "__main__":
    generate_KF_diss_plots()