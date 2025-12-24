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
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import config
from create_results_dir import create_results_dir
from SRC.plotting_utils import save_svg
import matplotlib as mpl
import os
config.update("jax_enable_x64", True)


# =========================
# Core experiment
# =========================
def run_test(
    pIC, crit, target_trj, stepper, adj_transform, vel_part_trans,
    dt, T,
    mbits, exp_bits, exp_bias,
    attractor_snapshots,
    idxs,
):
    loss_fn = create_loss_fn(crit, stepper, target_trj, pIC, vel_part_trans)
    loss_grad_fn_auto = jax.jit(jax.value_and_grad(loss_fn))

    loss_grad_fn_adj = get_loss_grad_vp_fn(
        pIC, crit, target_trj, stepper, adj_transform,
        vel_part_trans, dt, T,
        mbits=mbits, exp_bits=exp_bits, exp_bias=exp_bias,
    )
    loss_grad_fn_adj = jax.jit(loss_grad_fn_adj)

    loss_errs = []
    grad_errs = []
    cos_sims = []

    for idx in idxs:
        U_DA = attractor_snapshots[idx]
        U_DA_fourier = vel_part_trans.vel_flat_2_vel_Fourier(U_DA).astype(jnp.float64)

        loss_auto, grad_auto = loss_grad_fn_auto(U_DA_fourier)
        loss_adj,  grad_adj  = loss_grad_fn_adj(U_DA_fourier)

        loss_err = jnp.abs(loss_adj - loss_auto) / loss_auto * 100
        grad_err = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto) * 100
        cos_sim  = jnp.dot(grad_auto, grad_adj) / (
            jnp.linalg.norm(grad_auto) * jnp.linalg.norm(grad_adj)
        )

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
    pIC, crit, target_trj, stepper, adj_transform, vel_part_trans,
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
            pIC, crit, target_trj, stepper, adj_transform, vel_part_trans,
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

    npart = 25
    T = 4
    samp_period = .1
    mbits_list = np.arange(2, 14, 2)
    minv, maxv = 1e-3, 10.0

    period_idx = int(samp_period/kf_opts.dt)
    idx = jnp.arange(int(T/kf_opts.dt)+1)
    t_mask = (idx % period_idx == 0)

    crit = MSE_Vel()

    RHS = KF_LPT_PS_RHS(
        kf_opts.NDOF, kf_opts.Re, kf_opts.n,
        npart, beta=0, St=1e-2,
    )

    vel_part_trans = Vel_Part_Transformations(kf_opts.NDOF, npart)
    attractor_snapshots = load_data(kf_opts)

    U_true = attractor_snapshots[np.random.randint(0, attractor_snapshots.shape[0])]
    pIC = init_particles_vector(
        npart,
        vel_part_trans.reshape_flattened_vel(U_true),
        (0, RHS.KF_RHS.L), (0, RHS.KF_RHS.L),
        RHS.KF_RHS.L,
    )

    stepper = Particle_Stepper(RK4_Step(RHS, kf_opts.dt), npart)
    trj_gen_fn = create_trj_generator(RHS, kf_opts.dt, T, dtype=jnp.float64)
    target_trj = trj_gen_fn(pIC, U_true)

    crit.init_obj(t_mask, RHS.KF_RHS.L, vel_part_trans)
    adj_transform = build_div_free_proj(stepper, vel_part_trans)

    exp_bits, exp_bias = choose_exponent_format(minv, maxv, max_E=4)

    df = collect_stats_over_mbits(
        mbits_list,
        pIC=pIC, crit=crit, target_trj=target_trj,
        stepper=stepper, adj_transform=adj_transform,
        vel_part_trans=vel_part_trans,
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
