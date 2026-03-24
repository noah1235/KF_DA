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
from SRC.DA_Comp.adjoint import get_loss_grad_vp_fn, get_loss_grad_fn, get_forced_adj_shooting_vp, get_forced_adj_shooting
from SRC.parameterization.Fourier_Param import Fourier_Param
from SRC.plotting_utils import save_svg
from SRC.Solver.IC_gen import init_particles_vector
from SRC.Solver.solver import KF_Stepper, KF_TP_Stepper, create_omega_part_gen_fn, Omega_Integrator
from SRC.utils import load_data
from SRC.vp_floats.vp_py_utils import choose_exponent_format, float_pos_range
from scipy.optimize import curve_fit
config.update("jax_enable_x64", True)

lyap_Re100 = np.array([
 3.39436830e-01,
 2.52602822e-01,
 2.08957505e-01,
 1.71225854e-01,
 1.28376072e-01,
 1.03627456e-01,
 6.49729618e-02,
 4.84835512e-02,
 2.72289807e-02,
 3.56469465e-03,
-3.25823890e-03,
-1.94173525e-02,
-3.38014049e-02,
-5.27306059e-02,
-6.91448927e-02,
-8.36226010e-02,
-9.54887895e-02,
-1.24953527e-01,
-1.50690809e-01,
-1.70231006e-01,
-1.88002303e-01,
-2.05407481e-01,
-2.20288769e-01,
-2.47525143e-01,
-2.60422973e-01,
-2.78445629e-01,
-2.98722845e-01,
-3.13331447e-01,
-3.36176138e-01,
-3.56947663e-01,
-3.65027149e-01,
-3.83934359e-01,
-4.01783225e-01,
-4.16175938e-01,
-4.35618804e-01,
-4.47886311e-01,
-4.55225585e-01,
-4.72889664e-01,
-4.95852446e-01,
-5.05242430e-01,
-5.16723717e-01,
-5.32855383e-01,
-5.43119337e-01,
-5.56389509e-01,
-5.77570964e-01,
-5.83656390e-01,
-5.95726338e-01,
-6.10354406e-01,
-6.24079719e-01,
-6.37177648e-01,
-6.49645668e-01,
-6.64505559e-01,
-6.70812526e-01,
-6.87311013e-01,
-6.98385101e-01,
-7.07292118e-01,
-7.23044196e-01,
-7.29724376e-01,
-7.50296889e-01,
-7.58083874e-01,
-7.70992780e-01,
-7.85133474e-01,
-7.96899882e-01,
-8.10659267e-01,
-8.22397869e-01,
-8.33366482e-01,
-8.39055538e-01,
-8.57483499e-01,
-8.74821021e-01,
-8.81815831e-01,
-8.90802395e-01,
-9.05559239e-01,
-9.12353204e-01,
-9.28375509e-01,
-9.38009938e-01,
-9.49347060e-01,
-9.60697809e-01,
-9.75383020e-01,
-9.82552791e-01,
-9.91862300e-01,
-1.00651484e+00,
-1.01392201e+00,
-1.02323876e+00,
-1.03773263e+00,
-1.05175703e+00,
-1.06145307e+00,])


def get_lam_N(key, AS_flat, n_samples):
    X = AS_flat
    Xc = X - jnp.mean(X, axis=0, keepdims=True)

    n_snap, n_feat = Xc.shape

    key, subkey = jax.random.split(key)
    Z = jax.random.normal(subkey, (n_snap, n_samples))

    # (n_feat, n_snap) @ (n_snap, n_samples) -> (n_feat, n_samples)
    samples = (Xc.T @ Z) / jnp.sqrt(n_snap - 1)

    # return shape (n_samples, n_feat)
    return samples.T

# =========================
# Core experiment
# =========================
def run_test(
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
    n_trials,
):
    grad_errs = []
    cos_sims = []
    adj_rel_error_v_time = []

    if double:
        dtype = jnp.complex128
    else:
        dtype = jnp.complex64

    AS_flat = attractor_snapshots.reshape((attractor_snapshots.shape[0], -1))

    key = jax.random.PRNGKey(0)

    loss_grad_fn_adj_double = jax.jit(
        get_forced_adj_shooting(
            kf_stepper,
            IC_param.inv_transform,
            (attractor_snapshots.shape[1], attractor_snapshots.shape[2]),
            int(T / dt),
        )
    )

    loss_grad_fn_adj = jax.jit(
        get_forced_adj_shooting_vp(
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
        )
    )

    # precompute one lam_N per trial
    key, subkey = jax.random.split(key)
    lam_N_list = get_lam_N(subkey, AS_flat, n_trials)

    n_snapshots = attractor_snapshots.shape[0]

    for i in range(n_trials):
        key, subkey = jax.random.split(key)
        idx = int(jax.random.randint(subkey, shape=(), minval=0, maxval=n_snapshots))

        lam_N = lam_N_list[i].astype(dtype)

        omega0_DA_hat = attractor_snapshots[idx]
        if not double:
            omega0_DA_hat = omega0_DA_hat.astype(jnp.complex64)

        Z = IC_param.transform(omega0_DA_hat)

        _, adj_trj_double = loss_grad_fn_adj_double(Z, lam_N)
        grad_true = adj_trj_double[-1]
        _, adj_trj = loss_grad_fn_adj(
            Z.astype(jnp.float64), lam_N.astype(jnp.complex128)
        )
        grad_adj = adj_trj[-1]

        adj_trj_error = (
            jnp.linalg.norm(adj_trj_double - adj_trj, axis=1)
            / jnp.linalg.norm(adj_trj_double, axis=1)
        )

        adj_rel_error_v_time.append(adj_trj_error)
        
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
    nsamples
):
    IC_trg_seed = 0
    IC_DA_seed = 10
    results = {
        "meta": {
            "IC_trg_seed": int(IC_trg_seed),
            "IC_DA_seed": int(IC_DA_seed),
            "dt": float(dt),
            "T": float(T),
            "uniform": bool(uniform),
            "LLE": float(LLE),
            "exp_bits": int(exp_bits),
            "exp_bias": int(exp_bias),
        },
        "by_mbits": {},
    }

    for mbits in mbits_list:
        mint, maxt = float_pos_range(exp_bits, exp_bias, mbits)
        print(f"[mbits={mbits}] range = {mint:.2e} .. {maxt:.2e}")

        stats = run_test(
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
            nsamples
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

    p_ref = p[-1]
    err_ref = err[-1]
    theory = err_ref * 2.0 ** (-1.0 * (p - p_ref))

    fig = plt.figure()
    plt.scatter(p, err, label="Measured")
    plt.plot(p, theory, "--", label=r"$\propto 2^{-p}$")
    plt.yscale("log")
    plt.xlabel("mantissa bits (mbits)")
    plt.ylabel("Gradient relative error")
    plt.ylim(1e-6, 2e-1)
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
    # Rel error vs reverse-time
    # Plot mean + all individual curves in lighter grey
    # Store fit params into df
    # ---------------------------
    for mbit, mbit_df in df.groupby("mbits"):
        fig = plt.figure()

        all_err = np.asarray(mbit_df["adj_rel_error_v_time_all"].iloc[0])     # (nsamples, time)
        mean_err = np.asarray(mbit_df["adj_rel_error_v_time_mean"].iloc[0])  # (time,)

        t = np.arange(mean_err.shape[0]) * dt

        # keep your original slicing logic (apply to both)
        skip = -int(int(.1/dt))
        skip = -1
        all_err_plot = all_err[:, :skip][:, ::-1]
        mean_err_plot = mean_err[:skip][::-1]
        t_plot = t[:skip]

        # individual curves (lighter grey)
        #for k in range(all_err_plot.shape[0]):
        #    plt.plot(t_plot, all_err_plot[k], color="0.75", linewidth=1.0, alpha=0.7)

        
        # ----- linear fit on mean -----
        # use same slicing logic as plot
        fit_skip = int(.5/dt)
        mean_fit = mean_err_plot[fit_skip:]
        t_fit = t_plot[fit_skip:]
        r = np.corrcoef(t_fit, mean_fit)[0, 1]

        print(f"Correlation coefficient r = {r:.4f} | mbit={mbit}")
        # fit: mean_fit ≈ a + b*t
        b, a = np.polyfit(t_fit, mean_fit, 1)
        fit_curve = a + b * (t_plot)
        plt.plot(t_plot, fit_curve, "--", linewidth=2.0,
                label=f"fit: {a:.3e} + {b:.3e} t")

        # mean + fit
        plt.plot(t_plot, mean_err_plot, linewidth=2.0, label="mean")


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

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fit_params_v_m(df, path):
    mbits = df["mbits"].to_numpy()
    y_int = df["y_int"].to_numpy()
    slope = df["slope"].to_numpy()

    eps = 1e-12
    y_int_clip = np.maximum(y_int, eps)
    slope_clip = np.maximum(slope, eps)

    # -------------------------
    # Fit #1: y_int ≈ C * 2^(alpha * m)
    # -------------------------
    y1 = np.log2(y_int_clip)
    alpha1, log2C1 = np.polyfit(mbits, y1, 1)
    C1 = 2 ** log2C1
    y_int_fit = C1 * 2 ** (alpha1 * mbits)

    fig = plt.figure()
    plt.plot(mbits, y_int, "o-", label="measured")
    plt.plot(mbits, y_int_fit, "--", label=f"fit: {C1:.2e}·2^({alpha1:.2f} m)")
    plt.yscale("log")
    plt.xlabel("mbits")
    plt.ylabel("y int")
    plt.legend()
    plt.ylim(1e-6, 1e-1)
    save_svg(plt, fig, os.path.join(path, "y_int_vs_mbits.svg"))
    plt.close(fig)

    # -------------------------
    # Fit #2: slope ≈ C_s * 2^(alpha_s * m)
    # -------------------------
    y2 = np.log2(slope_clip)
    alpha2, log2C2 = np.polyfit(mbits, y2, 1)
    C2 = 2 ** log2C2
    slope_fit = C2 * 2 ** (alpha2 * mbits)


    print(y_int/slope)

    fig = plt.figure()
    plt.plot(mbits, slope, "o-", label="measured")
    plt.plot(mbits, slope_fit, "--", label=f"fit: {C2:.2e}·2^({alpha2:.2f} m)")
    plt.xlabel("mbits")
    plt.ylabel("slope")
    plt.yscale("log")
    plt.legend()
    plt.ylim(1e-6, 1e-1)
    save_svg(plt, fig, os.path.join(path, "slope_fit_vs_mbits.svg"))
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
        total_T=int(1e3),
        min_samp_T=100,
        t_skip=1e-1,
    )
    T_LLE = 3.3
    uniform = True
    LLE = 1 / (T_LLE)
    IC_param = Fourier_Param(kf_opts.NDOF, 64, beta=0, Re=100)
    double = True

    T = T_LLE * 1
    mbits_list = np.arange(6, 14, 2)
    #mbits_list = [6]
    minv, maxv = 1, 5e4

    attractor_snapshots = load_data(kf_opts)
    kf_stepper = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt, double=double)

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

    RUN_AND_SAVE = True

    if RUN_AND_SAVE:
        results = collect_stats_over_mbits(
            mbits_list,
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
            nsamples=100
        )


        # Build df, plot, compute & store fit params
        df = results_dict_to_df(results)
        df = plot_metrics_vs_mbits(df, path, kf_opts.dt)

    else:
        results = load_results_pickle(path, "vpfloats_sweep.pkl")


    df = results_dict_to_df(results)
    df = df[df["mbits"] != 2]
    df = plot_metrics_vs_mbits(df, path, kf_opts.dt)
    plot_fit_params_v_m(df, path)
    # Write parameters back into results + re-save (so dataset has the fit params)
    #results = write_fit_params_into_results(results, df)

    pkl_path = save_results_pickle(results, path, "vpfloats_sweep.pkl")
    xlsx_path = save_results_excel(results, path, "vpfloats_sweep.xlsx")


def make_roundoff_trj(key, ref_trj, p):
    """
    Create synthetic roundoff error trajectory.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    ref_trj : jnp.ndarray
        Reference trajectory (used for shape and dtype).
        Shape: (T, state_dim), complex.
    p : int
        Number of mantissa bits (23 for float32, 52 for float64).

    Returns
    -------
    eps_trj : jnp.ndarray
        Synthetic roundoff error with same shape and dtype as ref_trj.
    """

    # roundoff amplitude
    delta = 2.0 ** (-p - 1)

    shape = ref_trj.shape
    real_dtype = jnp.float32 if ref_trj.dtype == jnp.complex64 else jnp.float64

    key_r, key_i = jax.random.split(key)

    eps_real = jax.random.uniform(
        key_r,
        shape,
        minval=-delta,
        maxval=delta,
        dtype=real_dtype,
    )

    eps_imag = jax.random.uniform(
        key_i,
        shape,
        minval=-delta,
        maxval=delta,
        dtype=real_dtype,
    )

    eps_trj = eps_real + 1j * eps_imag
    return eps_trj.astype(ref_trj.dtype)

def build_sn_fn(u_0, step, N, eps_key, mbits, char_size, round_off_model):
    integrator = Omega_Integrator(step)
    DA_trj = integrator.integrate_scan(u_0, N)
    if round_off_model == "indep_random":
        eps_trj = make_roundoff_trj(eps_key, DA_trj, p=mbits)
        eps_trj *= char_size
    elif round_off_model == "uniform_rand":
        eps_trj *= DA_trj
    def run(lam_N, s_N):
        @jax.jit
        def backward_body(k, carry):
            lam_n, s_n = carry

            # map forward loop index k -> backward i = N-k (so u_{i-1} = DA_trj[N-k-1])
            u_n = DA_trj[N - k - 1]
            eps = eps_trj[N-k-1]

            # VJP at base point u_n
            _, vjp_fun = jax.vjp(step, u_n)

            # VJP at perturbed point u_n + eps
            _, vjp_eps_fun = jax.vjp(step, u_n + eps)

            v = vjp_eps_fun(lam_n)[0] - vjp_fun(lam_n)[0] + s_n

            # update s_n and lam_n using the *base* vjp_fun
            s_n = vjp_fun(v)[0]
            lam_n = vjp_fun(lam_n)[0]

            return (lam_n, s_n)

        # Run N steps (corresponds to i = N, N-1, ..., 1)
        lam_n, s_n = lax.fori_loop(
            0, N,
            backward_body,
            (lam_N, s_N),
        )
        return lam_n, s_n
    return run

def indep_test():
    num_ic = 100
    num_forcing = 1
    T_mult = 1.0
    double = True
    k0 = 8.0
    p = 4.0
    #indep_rand, uniform_rand, det
    round_off_model = "indep_random"

    mbits = 8

    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=int(1e6),
        min_samp_T=100,
        t_skip=1e-1,
    )

    T_LLE = 3.3
    T = T_LLE * T_mult
    N = int(T / kf_opts.dt)

    kf_stepper = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt, double=double)
    attractor_snapshots = load_data(kf_opts)
    snap_shape = (attractor_snapshots.shape[1], attractor_snapshots.shape[2])

    char_size = jnp.mean(jnp.abs(attractor_snapshots), axis=0).reshape(-1) * 10**3

    @jax.jit
    def step(x):
        return kf_stepper(x.reshape(snap_shape)).reshape(-1)
    

    s_N = jnp.zeros((snap_shape[0]*snap_shape[1]), dtype=jnp.complex128)

    # amplitude scale for random field
    f_norm = jnp.mean(jnp.linalg.norm(attractor_snapshots, axis=(1, 2)))

    dtype = jnp.complex128 if double else jnp.complex64


    num_snaps = attractor_snapshots.shape[0]

    s_sum = None
    lam_sum = None
    s_sq_sum = None
    count = 0

    for ic_idx in range(num_ic):
        key_ic = jax.random.PRNGKey(ic_idx)
        rand_idx = jax.random.randint(key_ic, (), 0, num_snaps)

        u_0 = attractor_snapshots[rand_idx].reshape(-1)
        get_sn_fn = build_sn_fn(u_0, step, N, key_ic, mbits, char_size, round_off_model)
        for lam_N_idx in range(num_forcing):
            key_lam = jax.random.PRNGKey(lam_N_idx)
            lam_N = smooth_periodic_field_fft(
                key_lam,
                attractor_snapshots.shape[1],
                f_norm,
                dtype,
                p=p,
                k0=k0,
            ).reshape(-1)

            lam_0, s_0 = get_sn_fn(lam_N, s_N)
            if s_sum is None:
                s_sum = jnp.zeros_like(s_0)
                lam_sum = jnp.zeros_like(lam_0)
                s_sq_sum = jnp.zeros_like(s_0)

            s_sum += s_0
            lam_sum += lam_0
            s_sq_sum += jnp.abs(s_0) ** 2
            count += 1
            print(count)


    # mean
    s_mean = s_sum / count
    lam_mean = lam_sum / count

    # variance (per component)
    s_var = s_sq_sum / count - jnp.abs(s_mean) ** 2
    s_std = jnp.sqrt(jnp.maximum(s_var, 0.0))

    # standard error of the mean
    sem = s_std / jnp.sqrt(count)

    # 95% confidence bound
    bound = 1.96 * sem

    # test: is mean within CI?
    within_ci = jnp.abs(s_mean) < bound

    print("Max |mean|:", jnp.max(jnp.abs(s_mean) / char_size))
    print("Max CI bound:", jnp.max(bound))
    print("Max CI bound:", jnp.min(bound))
    print("Fraction within 95% CI:",
        jnp.mean(within_ci.astype(jnp.float32)))

    print(jnp.linalg.norm(s_mean)/jnp.linalg.norm(lam_mean))


def test():
    loss_grad_fn_adj_double = jax.jit(
    get_forced_adj_shooting(
        kf_stepper,
        lambda x: x,
        (attractor_snapshots.shape[1], attractor_snapshots.shape[2]),
        int(T / kf_opts.dt),
    )
)
    grad_true, adj_trj_double = loss_grad_fn_adj_double(u_0.reshape(snap_shape), lam_N)
    grad_true = grad_true.reshape(-1)
    print(jnp.linalg.norm(grad_true - lam_n)/jnp.linalg.norm(grad_true))


if __name__ == "__main__":
    adjoint_test()
