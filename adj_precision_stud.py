# =========================
# Imports (yours + minimal extras)
# =========================
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import config

from create_results_dir import create_results_dir
from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.DA_Comp.adjoint import get_loss_grad_vp_fn, get_loss_grad_fn
from SRC.parameterization.Fourier_Param import Fourier_Param
from SRC.plotting_utils import save_svg
from SRC.Solver.IC_gen import init_particles_vector
from SRC.Solver.solver import KF_Stepper, KF_TP_Stepper, create_omega_part_gen_fn
from SRC.utils import load_data
from SRC.vp_floats.vp_py_utils import choose_exponent_format, float_pos_range

config.update("jax_enable_x64", True)


# =========================
# Core experiment
# =========================
def run_test(
    crit,
    trj_gen_fn,
    pIC,
    stepper,
    kf_stepper,
    IC_param,
    dt,
    T,
    mbits,
    exp_bits,
    exp_bias,
    uniform,
    LLE,
    attractor_snapshots,
    double,
    idxs_DA,
    idxs_trg
):
    grad_errs = []
    cos_sims = []
    adj_rel_error_v_time = []  # list of (time,) arrays, one per DA sample (across all trg)

    for idx_trg in idxs_trg:
        omega0_hat = attractor_snapshots[idx_trg]
        if double:
            omega0_hat = omega0_hat.astype(jnp.complex128)
        else:
            omega0_hat = omega0_hat.astype(jnp.complex64)
        target_trj = trj_gen_fn(omega0_hat, *pIC)

        loss_fn = create_loss_fn(crit, stepper, target_trj, IC_param.inv_transform)
        loss_grad_fn_auto = jax.jit(jax.value_and_grad(loss_fn))  # kept (unused), do not change logic

        loss_grad_fn_adj_double = jax.jit(
            get_loss_grad_fn(
                crit,
                target_trj,
                kf_stepper,
                IC_param.inv_transform,
                (attractor_snapshots.shape[1], attractor_snapshots.shape[2]),
                int(T / dt),
            )
        )

        loss_grad_fn_adj = get_loss_grad_vp_fn(
            crit,
            target_trj,
            kf_stepper,
            IC_param.inv_transform,
            (attractor_snapshots.shape[1], attractor_snapshots.shape[2]),
            dt,
            T,
            mbits=mbits,
            exp_bits=exp_bits,
            exp_bias=exp_bias,
            uniform=uniform,
            LLE=LLE,
            return_lam_trj=True,
        )
        loss_grad_fn_adj = jax.jit(loss_grad_fn_adj)

        for idx in idxs_DA:
            omega0_DA_hat = attractor_snapshots[idx]
            Z = IC_param.transform(omega0_DA_hat)

            loss_true, grad_true, adj_trj_double = loss_grad_fn_adj_double(Z)
            loss_adj, grad_adj, adj_trj = loss_grad_fn_adj(Z)

            if not double:
                grad_true = grad_true.astype(jnp.float32)
                grad_adj = grad_adj.astype(jnp.float32)

                adj_trj_double = adj_trj_double.astype(jnp.float32)
                adj_trj = adj_trj.astype(jnp.float32)

            rel = (
                jnp.linalg.norm(adj_trj_double - adj_trj, axis=1)
                / jnp.linalg.norm(adj_trj_double, axis=1)
            )
            adj_rel_error_v_time.append(rel)

            grad_err = jnp.linalg.norm(grad_adj - grad_true) / jnp.linalg.norm(grad_true)
            cos_sim = jnp.dot(grad_true, grad_adj) / (
                jnp.linalg.norm(grad_true) * jnp.linalg.norm(grad_adj)
            )

            print(
                f"[mbits={mbits}] Sample idx={idx}, "
                f"Grad % err={grad_err:.2e}, Cos sim={cos_sim:.6f}"
            )

            grad_errs.append(grad_err)
            cos_sims.append(cos_sim)

    grad_errs = jnp.array(grad_errs)
    cos_sims = jnp.array(cos_sims)

    # Stack (nsamples, time)
    adj_rel_error_v_time = jnp.vstack(adj_rel_error_v_time)
    adj_rel_error_v_time_mean = jnp.mean(adj_rel_error_v_time, axis=0)

    # Convert to numpy for clean pickling / pandas storage
    grad_errs_np = np.asarray(grad_errs)
    cos_sims_np = np.asarray(cos_sims)
    adj_all_np = np.asarray(adj_rel_error_v_time)          # (nsamples, time)
    adj_mean_np = np.asarray(adj_rel_error_v_time_mean)    # (time,)

    return {
        "grad_mean": float(grad_errs_np.mean()),
        "grad_std": float(grad_errs_np.std()),
        "cos_mean": float(cos_sims_np.mean()),
        "cos_std": float(cos_sims_np.std()),
        "adj_rel_error_v_time_all": adj_all_np,
        "adj_rel_error_v_time_mean": adj_mean_np,
    }


# =========================
# Sweep mbits (DICT OUTPUT)
# =========================
def collect_stats_over_mbits(
    mbits_list,
    *,
    crit,
    trj_gen_fn,
    pIC,
    stepper,
    kf_stepper,
    IC_param,
    dt,
    T,
    exp_bits,
    exp_bias,
    uniform,
    LLE,
    attractor_snapshots,
    double,
    nreps_DA=20,
    nreps_trg=5
):
    IC_trg_seed = 0
    IC_DA_seed = 10

    rng = np.random.default_rng(IC_DA_seed)
    idxs_DA = rng.integers(0, attractor_snapshots.shape[0] - 1, size=nreps_DA)

    rng = np.random.default_rng(IC_trg_seed)
    idxs_trg = rng.integers(0, attractor_snapshots.shape[0] - 1, size=nreps_trg)

    results = {
        "meta": {
            "nreps_DA": int(nreps_DA),
            "nreps_trg": int(nreps_trg),
            "IC_trg_seed": int(IC_trg_seed),
            "IC_DA_seed": int(IC_DA_seed),
            "dt": float(dt),
            "T": float(T),
            "uniform": bool(uniform),
            "LLE": float(LLE),
            "exp_bits": int(exp_bits),
            "exp_bias": int(exp_bias),
            "idxs_DA": np.asarray(idxs_DA),
            "idxs_trg": np.asarray(idxs_trg),
        },
        "by_mbits": {},
    }

    for mbits in mbits_list:
        mint, maxt = float_pos_range(exp_bits, exp_bias, mbits)
        print(f"[mbits={mbits}] range = {mint:.2e} .. {maxt:.2e}")

        stats = run_test(
            crit,
            trj_gen_fn,
            pIC,
            stepper,
            kf_stepper,
            IC_param,
            dt,
            T,
            mbits,
            exp_bits,
            exp_bias,
            uniform,
            LLE,
            attractor_snapshots,
            double,
            idxs_DA,
            idxs_trg
        )

        results["by_mbits"][int(mbits)] = {
            "mbits": int(mbits),
            "min_pos": float(mint),
            "max_pos": float(maxt),
            "exp_bits": int(exp_bits),
            "exp_bias": int(exp_bias),
            **stats,
        }

    return results


def results_dict_to_df(results: dict) -> pd.DataFrame:
    rows = []
    for _, d in results["by_mbits"].items():
        rows.append(d)
    return pd.DataFrame(rows).sort_values("mbits")


# =========================
# Pickle + Excel helpers
# =========================
def save_results_pickle(results: dict, path: str, filename: str = "vpfloats_sweep.pkl") -> str:
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, filename)
    with open(fpath, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fpath


def load_results_pickle(path: str, filename: str = "vpfloats_sweep.pkl") -> dict:
    fpath = os.path.join(path, filename)
    with open(fpath, "rb") as f:
        return pickle.load(f)


def save_results_excel(results: dict, path: str, filename: str = "vpfloats_sweep.xlsx") -> str:
    """
    Writes an Excel workbook you can inspect easily.

    Sheet 1: 'summary'   -> one row per mbits, with scalars + ranges (+ fit params)
    Sheet 2: 'meta'      -> experiment metadata
    Sheet 3+: 'adj_err_m{mbits}' -> all rel-error-v-time trajectories for that mbits
                                  (columns are samples, rows are time index)
    Sheet 4+: 'adj_err_mean_m{mbits}' -> mean rel-error-v-time for that mbits
    """
    os.makedirs(path, exist_ok=True)
    xlsx_path = os.path.join(path, filename)

    # Summary sheet (scalars only; keep arrays out of the main table)
    df_summary = results_dict_to_df(results).copy()
    if "adj_rel_error_v_time_all" in df_summary.columns:
        df_summary = df_summary.drop(columns=["adj_rel_error_v_time_all"])
    if "adj_rel_error_v_time_mean" in df_summary.columns:
        df_summary = df_summary.drop(columns=["adj_rel_error_v_time_mean"])

    # Meta sheet
    meta = results.get("meta", {})
    df_meta = pd.DataFrame({"key": list(meta.keys()), "value": list(meta.values())})

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_meta.to_excel(writer, sheet_name="meta", index=False)

        # Per-mbits trajectories
        for mbits, d in sorted(results["by_mbits"].items(), key=lambda kv: kv[0]):
            all_err = np.asarray(d["adj_rel_error_v_time_all"])  # (nsamples, time)
            mean_err = np.asarray(d["adj_rel_error_v_time_mean"])  # (time,)

            # Put all trajectories as columns: sample_0, sample_1, ...
            df_all = pd.DataFrame(all_err.T, columns=[f"sample_{i}" for i in range(all_err.shape[0])])
            df_all.insert(0, "t_idx", np.arange(df_all.shape[0]))

            # Mean in its own sheet
            df_mean = pd.DataFrame({"t_idx": np.arange(mean_err.shape[0]), "mean": mean_err})

            sheet_all = f"adj_err_m{mbits}"
            sheet_mean = f"adj_err_mean_m{mbits}"

            # Excel sheet name limit is 31 chars—these are safe.
            df_all.to_excel(writer, sheet_name=sheet_all[:31], index=False)
            df_mean.to_excel(writer, sheet_name=sheet_mean[:31], index=False)

    return xlsx_path


# =========================
# Curve fit helpers (NEW)
# =========================
def fit_exp_on_mean(mean_err: np.ndarray, dt: float, *, skip: int = -40, eps: float = 1e-30):
    """
    Fit mean_err(t) ≈ A * exp(lam * t) on reverse-time, matching your existing slicing logic.
    Returns (A, lam, logA, t_fit, mean_fit, npts).
    """
    mean_err = np.asarray(mean_err)

    # time array
    t = np.arange(mean_err.shape[0]) * dt

    # match your slicing logic
    mean_err_fit = mean_err[:skip][::-1]
    t_fit = t[:skip]

    # log-linear regression: log(y) ≈ lam*t + logA
    y = np.log(mean_err_fit + eps)
    lam, logA = np.polyfit(t_fit, y, 1)
    A = np.exp(logA)

    return float(A), float(lam), float(logA), t_fit, mean_err_fit, int(t_fit.shape[0])


def write_fit_params_into_results(results: dict, df: pd.DataFrame) -> dict:
    """
    Copy exp-fit params from the df (one row per mbits) into results["by_mbits"][mbits].
    """
    for _, row in df.iterrows():
        mb = int(row["mbits"])
        d = results["by_mbits"][mb]
        d["expfit_A"] = float(row["expfit_A"])
        d["expfit_lam"] = float(row["expfit_lam"])
        d["expfit_logA"] = float(row["expfit_logA"])
        d["expfit_npts"] = int(row["expfit_npts"])
    return results


# =========================
# Plotting
# =========================
def plot_metrics_vs_mbits(df, path, dt):
    os.makedirs(path, exist_ok=True)

    # Ensure columns exist (NEW)
    for col in ["expfit_A", "expfit_lam", "expfit_logA", "expfit_npts"]:
        if col not in df.columns:
            df[col] = np.nan

    mb = df["mbits"].to_numpy()

    # ---------------------------
    # Gradient percent error
    # ---------------------------
    p = np.array(mb)
    err = np.array(df["grad_mean"])
    err_std = np.array(df["grad_std"])  # kept (unused), do not change logic

    p_ref = p[-1]
    err_ref = err[-1]
    theory = err_ref * 2.0 ** (-1.0 * (p - p_ref))

    fig = plt.figure()
    plt.scatter(p, err, label="Measured")
    plt.plot(p, theory, "--", label=r"$\propto 2^{-p}$")
    plt.yscale("log")
    plt.xlabel("mantissa bits (mbits)")
    plt.ylabel("Gradient relative error")
    plt.ylim(1e-5, 2e-1)
    plt.legend()
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
    save_svg(plt, fig, os.path.join(path, "cos_v_mbits.svg"))
    plt.close(fig)

    # ---------------------------
    # Rel error vs reverse-time + exp fit
    # Plot mean + all individual curves in lighter grey
    # Store fit params into df
    # ---------------------------
    for mbit, mbit_df in df.groupby("mbits"):
        fig = plt.figure()

        all_err = np.asarray(mbit_df["adj_rel_error_v_time_all"].iloc[0])     # (nsamples, time)
        mean_err = np.asarray(mbit_df["adj_rel_error_v_time_mean"].iloc[0])  # (time,)

        t = np.arange(mean_err.shape[0]) * dt

        # keep your original slicing logic (apply to both)
        skip = -40
        all_err_plot = all_err[:, :skip][:, ::-1]
        mean_err_plot = mean_err[:skip][::-1]
        t_plot = t[:skip]

        # individual curves (lighter grey)
        for k in range(all_err_plot.shape[0]):
            plt.plot(t_plot, all_err_plot[k], color="0.75", linewidth=1.0, alpha=0.7)

        # exponential fit on mean
        #A, lam, logA, t_fit, mean_fit, npts = fit_exp_on_mean(mean_err, dt, skip=skip)
        #fit_curve = A * np.exp(lam * t_fit)
        
        # ----- linear fit on mean -----
        # use same slicing logic as plot
        mean_fit = mean_err[:skip][::-1]
        t_fit = t[:skip]

        # fit: mean_fit ≈ a + b*t
        b, a = np.polyfit(t_fit, mean_fit, 1)
        fit_curve = a + b * t_fit

        # mean + fit
        plt.plot(t_plot, mean_err_plot, linewidth=2.0, label="mean")
        plt.plot(t_fit, fit_curve, "--", linewidth=2.0,
                label=f"fit: {a:.3e} + {b:.3e} t")

        plt.xlabel("reverse time")
        plt.ylabel("rel L2 error")
        #plt.yscale("log")
        plt.title(f"m = {mbit}")
        plt.legend()

        save_svg(plt, fig, os.path.join(path, f"rel_error_v_t_m={mbit}.svg"))
        plt.close(fig)

        # STORE PARAMETERS (NEW)
        mask = df["mbits"] == mbit
        df.loc[mask, "slope"] = b
        df.loc[mask, "y_int"] = a

    return df

def plot_fit_params_v_m(df, path):
    mbits = df["mbits"].to_numpy()
    y_int = df["y_int"].to_numpy()
    slope = df["slope"].to_numpy()
    eps = 1e-12
    slope = np.maximum(slope, eps)
    # log2 transform
    y = np.log2(y_int)

    # linear fit: y ≈ alpha*m + log2C
    alpha, log2C = np.polyfit(mbits, y, 1)

    C = 2 ** log2C

    print(f"Fit: A ≈ {C:.3e} * 2^({alpha:.3f} * mbits)")

    # fitted curve
    alpha = -1.0
    A_fit = C * 2 ** (alpha * mbits)

    # plot
    fig = plt.figure()
    plt.plot(mbits, y_int, "o-", label="measured")
    plt.plot(mbits, A_fit, "--", label=f"fit: {C:.2e}·2^({alpha:.2f} m)")
    plt.yscale("log")
    plt.xlabel("mbits")
    plt.ylabel("y int")
    plt.legend()
    save_svg(plt, fig, os.path.join(path, f"y_int_vs_mbits.svg"))
    plt.close(fig)


    fig = plt.figure()
    plt.plot(mbits, slope, "o-", label="measured")
    plt.xlabel("mbits")
    plt.ylabel("slope")
    plt.yscale("log")
    save_svg(plt, fig, os.path.join(path, f"slope_fit_vs_mbits.svg"))
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
    T_LLE = 3.3
    uniform = True
    LLE = 1 / (T_LLE)
    IC_param = Fourier_Param(kf_opts.NDOF, 64)
    double = True
    npart = 16
    NT = 1
    T = T_LLE * 1
    mbits_list = np.arange(2, 16, 2)
    # mbits_list = [10]

    minv, maxv = 1, 5e4
    seed = 10

    if NT == 1:
        t_mask = np.zeros(int(T / kf_opts.dt) + 1)
        t_mask[-1] = 1
        t_mask = jnp.array(t_mask)
    else:
        samp_period = T / (NT - 1)
        period_idx = int(samp_period / kf_opts.dt)
        idx = jnp.arange(int(T / kf_opts.dt) + 1)
        t_mask = idx % period_idx == 0

    crit = MSE_Vel()
    attractor_snapshots = load_data(kf_opts)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, attractor_snapshots.shape[0])
    omega0_hat = attractor_snapshots[idx]

    stepper = KF_TP_Stepper(
        kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt, 0, 0, npart, double=double
    )
    kf_stepper = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt)

    u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
    u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)

    pIC = init_particles_vector(
        npart,
        u,
        v,
        (0, stepper.NS.L),
        (0, stepper.NS.L),
        stepper.NS.L,
        seed=1,
    )

    trj_gen_fn = create_omega_part_gen_fn(jax.jit(stepper), T)
    crit.init_obj(t_mask, stepper.NS.L)

    exp_bits, exp_bias = choose_exponent_format(minv, maxv, max_E=8)

    path = os.path.join(
        create_results_dir(),
        "vpfloats",
        f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_T={T}_",
    )
    path += "uniform" if uniform else f"optimal_LLE={LLE:.3e}"
    if double:
        path += "_f64"
    else:
        path += "_f32"

    RUN_AND_SAVE = False

    if RUN_AND_SAVE:
        results = collect_stats_over_mbits(
            mbits_list,
            crit=crit,
            trj_gen_fn=trj_gen_fn,
            pIC=pIC,
            stepper=stepper,
            kf_stepper=kf_stepper,
            IC_param=IC_param,
            dt=kf_opts.dt,
            T=T,
            exp_bits=exp_bits,
            exp_bias=exp_bias,
            uniform=uniform,
            LLE=LLE,
            attractor_snapshots=attractor_snapshots,
            double=double,
            # nreps_DA=2,
            # nreps_trg=1,
            nreps_DA=20,
            nreps_trg=5
        )


        # Build df, plot, compute & store fit params
        df = results_dict_to_df(results)
        df = plot_metrics_vs_mbits(df, path, kf_opts.dt)

    else:
        results = load_results_pickle(path, "vpfloats_sweep.pkl")
    df = results_dict_to_df(results)
    df = plot_metrics_vs_mbits(df, path, kf_opts.dt)
    plot_fit_params_v_m(df, path)
    # Write parameters back into results + re-save (so dataset has the fit params)
    results = write_fit_params_into_results(results, df)

    pkl_path = save_results_pickle(results, path, "vpfloats_sweep.pkl")
    xlsx_path = save_results_excel(results, path, "vpfloats_sweep.xlsx")


if __name__ == "__main__":
    adjoint_test()
