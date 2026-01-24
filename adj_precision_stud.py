# =========================
# Imports (yours + minimal extras)
# =========================
from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.Solver.KF_intergrators import KF_LPT_PS_RHS, create_trj_generator
from SRC.DA_Comp.adjoint import get_loss_grad_vp_fn
from SRC.utils import load_data, build_div_free_proj
from SRC.Vel_init.CS_init import CS_init
from SRC.utils import build_div_free_proj
from SRC.vp_floats.vp_py_utils import choose_exponent_format, float_pos_range
from SRC.Solver.IC_gen import init_particles_vector
import jax
from SRC.Solver.solver import KF_Stepper, KF_TP_Stepper, create_omega_part_gen_fn
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import config
from create_results_dir import create_results_dir
from SRC.plotting_utils import save_svg
import matplotlib as mpl
import os
from SRC.parameterization.Fourier_Param import Fourier_Param
config.update("jax_enable_x64", True)


# =========================
# Core experiment
# =========================
def run_test(
    crit, target_trj, stepper, kf_stepper, IC_param,
    dt, T,
    mbits, exp_bits, exp_bias,
    attractor_snapshots,
    idxs,
):

    loss_fn = create_loss_fn(crit, stepper, target_trj, IC_param.inv_transform)
    loss_grad_fn_auto = jax.jit(jax.value_and_grad(loss_fn))

    loss_grad_fn_adj = get_loss_grad_vp_fn(
        crit, target_trj, kf_stepper, IC_param.inv_transform, (attractor_snapshots.shape[1], attractor_snapshots.shape[2]), dt, T,
        mbits=mbits, exp_bits=exp_bits, exp_bias=exp_bias,
    )

    loss_grad_fn_adj = jax.jit(loss_grad_fn_adj)

    loss_errs = []
    grad_errs = []
    cos_sims = []

    for idx in idxs:
        omega0_DA_hat = attractor_snapshots[idx]
        Z = IC_param.transform(omega0_DA_hat)

        loss_auto, grad_auto = loss_grad_fn_auto(Z)
        loss_adj,  grad_adj  = loss_grad_fn_adj(Z)

        loss_err = jnp.abs(loss_adj - loss_auto) / loss_auto * 100
        grad_err = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto) * 100
        cos_sim  = jnp.dot(grad_auto, grad_adj) / (
            jnp.linalg.norm(grad_auto) * jnp.linalg.norm(grad_adj)
        )
        print(f"[mbits={mbits}] Sample idx={idx}: Loss % err={loss_err:.2e}, Grad % err={grad_err:.2e}, Cos sim={cos_sim:.6f}")
        loss_errs.append(loss_err)
        grad_errs.append(grad_err)
        cos_sims.append(cos_sim)

    loss_errs = jnp.array(loss_errs)
    grad_errs = jnp.array(grad_errs)
    cos_sims  = jnp.array(cos_sims)

    return {
        "loss_mean": float(jnp.mean(loss_errs)),
        "loss_std":  float(jnp.std(loss_errs)),
        "grad_mean": float(jnp.mean(grad_errs)),
        "grad_std":  float(jnp.std(grad_errs)),
        "cos_mean":  float(jnp.mean(cos_sims)),
        "cos_std":   float(jnp.std(cos_sims)),
    }


# =========================
# Sweep mbits
# =========================
def collect_stats_over_mbits(
    mbits_list,
    *,
    crit, target_trj, stepper, kf_stepper, IC_param,
    dt, T, exp_bits, exp_bias,
    attractor_snapshots,
    nreps=20,
    seed=0,
):
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, attractor_snapshots.shape[0] - 1, size=nreps)

    rows = []
    for mbits in mbits_list:
        mint, maxt = float_pos_range(exp_bits, exp_bias, mbits)
        print(f"[mbits={mbits}] range = {mint:.2e} .. {maxt:.2e}")

        stats = run_test(
            crit, target_trj, stepper, kf_stepper, IC_param,
            dt, T,
            mbits, exp_bits, exp_bias,
            attractor_snapshots,
            idxs,
        )

        rows.append({
            "mbits": mbits,
            "min_pos": mint,
            "max_pos": maxt,
            "exp_bits": exp_bits,
            "exp_bias": exp_bias,
            **stats,
        })

    return pd.DataFrame(rows).sort_values("mbits")


# =========================
# Plotting
# =========================
def plot_metrics_vs_mbits(df, path):
    os.makedirs(path, exist_ok=True)
    mb = df["mbits"].to_numpy()

    # ---------------------------
    # Loss percent error
    # ---------------------------
    fig = plt.figure()
    plt.errorbar(mb, df["loss_mean"], yerr=df["loss_std"], fmt="o-", capsize=4)
    plt.yscale("log")
    plt.xlabel("mantissa bits (mbits)")
    plt.ylabel("Loss percent error (%)")
    plt.title("Loss error vs mantissa bits")
    #plt.grid(True, which="both", ls="--", alpha=0.4)

    save_svg(plt, fig, os.path.join(path, "loss_v_mbits.svg"))
    plt.close(fig)

    # ---------------------------
    # Gradient percent error
    # ---------------------------
    fig = plt.figure()
    plt.errorbar(mb, df["grad_mean"], yerr=df["grad_std"], fmt="o-", capsize=4)
    plt.yscale("log")
    plt.xlabel("mantissa bits (mbits)")
    plt.ylabel("Gradient percent error (%)")
    plt.title("Gradient error vs mantissa bits")
    #plt.grid(True, which="both", ls="--", alpha=0.4)

    save_svg(plt, fig, os.path.join(path, "grad_v_mbits.svg"))
    plt.close(fig)

    # ---------------------------
    # Gradient cosine similarity
    # ---------------------------
    fig = plt.figure()
    plt.errorbar(mb, df["cos_mean"], yerr=df["cos_std"], fmt="o-", capsize=4)
    plt.xlabel("mantissa bits (mbits)")
    plt.ylabel("Gradient cosine similarity")
    plt.title("Gradient cosine similarity vs mantissa bits")
    #plt.grid(True, ls="--", alpha=0.4)

    save_svg(plt, fig, os.path.join(path, "cos_v_mbits.svg"))
    plt.close(fig)

# =========================
# Main driver
# =========================
def adjoint_test():
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=1000,
        min_samp_T=50,
        t_skip=1e-1,
    )
    IC_param = Fourier_Param(kf_opts.NDOF, 64)
    npart = 25
    T = 4
    samp_period = .1
    mbits_list = np.arange(2, 14, 2)
    minv, maxv = 1, 5e4

    period_idx = int(samp_period/kf_opts.dt)
    idx = jnp.arange(int(T/kf_opts.dt)+1)
    t_mask = (idx % period_idx == 0)

    crit = MSE_Vel()
    attractor_snapshots = load_data(kf_opts)

    omega0_hat = attractor_snapshots[np.random.randint(0, attractor_snapshots.shape[0])]
    stepper = KF_TP_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt, 0, 0, npart)
    kf_stepper = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt)
    u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
    u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)
    xp, yp, up, vp = init_particles_vector(npart, u, v, (0, stepper.NS.L), (0, stepper.NS.L), stepper.NS.L, seed=1)

    trj_gen_fn = create_omega_part_gen_fn(jax.jit(stepper), T)
    #tuple (omega_traj, xp_traj, yp_traj, up_traj, vp_traj)
    target_trj = trj_gen_fn(omega0_hat, xp, yp, up, vp)

    crit.init_obj(t_mask, stepper.NS.L)
    exp_bits, exp_bias = choose_exponent_format(minv, maxv, max_E=8)
    df = collect_stats_over_mbits(
        mbits_list,
        crit=crit, target_trj=target_trj,
        stepper=stepper, kf_stepper=kf_stepper, IC_param=IC_param,
        dt=kf_opts.dt, T=T,
        exp_bits=exp_bits, exp_bias=exp_bias,
        attractor_snapshots=attractor_snapshots,
        nreps=10,
        seed=0,
    )

    print(df)
    path = os.path.join(os.path.join(create_results_dir(), "vpfloats", f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_T={T}"))
    os.makedirs(path, exist_ok=True)
    df.to_excel(os.path.join(path, "vpfloats_sweep.xlsx"), index=False)
    plot_metrics_vs_mbits(df, path)


if __name__ == "__main__":
    adjoint_test()
