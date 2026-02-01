from create_results_dir import create_results_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from SRC.plotting_utils import save_svg
import matplotlib as mpl
from scipy.optimize import curve_fit
import pandas as pd
import re

def diss_v_time_plot():
    NDOF = 128
    dt = 1e-2
    seed_list = [1]
    Re_list = [8, 22, 40, 100]
    root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate")


    # --- create figure + axes explicitly ---
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    for seed in seed_list:
        data_path = os.path.join(root, f"diss_v_time_IC_seed={seed}")

        for Re in Re_list:
            D = np.load(os.path.join(data_path, f"dissnorm_Re={Re}_N={NDOF}_dt={dt}.npy"))
            t = np.load(os.path.join(data_path, f"t_Re={Re}_N={NDOF}_dt={dt}.npy"))
            ax.plot(
                t, D,
                lw=1.8,
                label=rf"$\mathrm{{Re}}={Re}$"
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Dissipation Rate")

    # Legend 1: colors = Re (deduplicate entries)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    leg1 = ax.legend(
        unique.values(), unique.keys(),
        title="Reynolds number",
        loc="upper right",
        frameon=True
    )
    ax.add_artist(leg1)

    save_svg(mpl, fig, os.path.join(root, f"dissipation_rate_comparison.svg"))

def avg_dist_plot():
    seed = 1
    n = 4
    cases = [
        # (Re, NDOF, dt)
        #(40,  128, 1e-2),
        #(60,  128, 1e-2),
        (80,  128, 1e-2),
        (100,  128, 1e-2),
        (120, 128, 1e-2),
        (140, 128, 1e-2),
        (160, 128, 1e-2),
        (180, 128, 1e-2),
        (200, 256, 2.5e-3),
        (220, 256, 2.5e-3),
        (300, 512, 1e-3),
        (400, 512, 1e-3)
    ]

    root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate")
    data_path = os.path.join(root, f"diss_v_time_IC_seed={seed}")

    avg_diss_list = []
    Re_list = []

    for Re, NDOF, dt in cases:
        D = np.load(os.path.join(data_path, f"dissnorm_Re={Re}_N={NDOF}_dt={dt}.npy"))
        print(Re, D.shape)
        D_lam = Re / (2 * n**2)
        mean_D = np.mean(D) * D_lam
        avg_diss_list.append(float(mean_D))
        Re_list.append(Re)

    Re_arr = np.array(Re_list, dtype=float)
    avg_arr = np.array(avg_diss_list, dtype=float)

    # ---- curve fit: b Re^{-1/2} + a ----
    def model(Re, a, b):
        return a + b * Re**(-0.5)

    # reasonable initial guess:
    a0 = avg_arr[-1]                  # high-Re asymptote
    b0 = (avg_arr[0] - a0) * np.sqrt(Re_arr[0])
    popt, pcov = curve_fit(model, Re_arr, avg_arr, p0=[a0, b0])

    a_fit, b_fit = popt
    print(f"Fit: a = {a_fit:.6g}, b = {b_fit:.6g}")

    # smooth fitted curve for plotting
    Re_fine = np.linspace(Re_arr.min(), Re_arr.max(), 400)
    fit_curve = model(Re_fine, a_fit, b_fit)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.plot(Re_arr, avg_arr, marker='o', linewidth=2, label="Data")
    ax.plot(Re_fine, fit_curve, linestyle="--", linewidth=2,
            label=rf"Fit: $a + b\,\mathrm{{Re}}^{{-1/2}}$" "\n"
                rf"$a={a_fit:.3g},\, b={b_fit:.3g}$")

    ax.set_xlabel("Reynolds Number")
    ax.set_ylabel("Average Dissipation Rate")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    save_svg(mpl, fig, os.path.join(root, "avg_diss_v__Re.svg"))

def vp_float_case_sum():
    Re = 100
    NDOF = 128
    dt = 1e-2
    T = 3.3
    St = 0
    beta = 0
    n = 4

    NT = 31
    n_part = 30

    data_path = os.path.join(
        create_results_dir(),
        f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}-St={St}_beta={beta}_AI_vp",
        "results.parquet",
    )
    df = pd.read_parquet(data_path)

    # Filters
    df = df[(df["T"] == T) & (df["n_part"] == n_part) & (df["NT"] == NT)].copy()

    # --- extract M from floatp strings like "M=12_E=5_bias=11"
    def parse_M(floatp: str):
        if str(floatp).lower() == "double":
            return None
        m = re.search(r"M=(\d+)", str(floatp))
        return int(m.group(1)) if m else None

    # --- pull per-run metrics
    df["final_loss"] = df["loss_record"].apply(lambda a: float(np.asarray(a)[-1]))
    df["final_cos"]  = df["trj_cos_sim"].astype(float)

    # --- baseline: float64 ("double")
    df64 = df[df["floatp"].astype(str).str.lower() == "double"]
    if len(df64) == 0:
        raise ValueError("No floatp == 'double' rows found for baseline.")

    base_loss_mean = float(df64["final_loss"].mean())
    base_cos_mean  = float(df64["final_cos"].mean())

    # guard against divide-by-zero baseline (esp. loss)
    eps = 1e-300
    loss_denom = max(abs(base_loss_mean), eps)
    cos_denom  = max(abs(base_cos_mean),  eps)

    # --- compute percent diffs PER TRIAL w.r.t. baseline MEAN
    df_vp = df[df["floatp"].astype(str).str.lower() != "double"].copy()
    df_vp["M"] = df_vp["floatp"].apply(parse_M)

    if df_vp["M"].isna().any():
        bad = df_vp[df_vp["M"].isna()]["floatp"].unique()
        raise ValueError(f"Couldn't parse M from these floatp values: {bad}")

    df_vp["loss_pctdiff"] = 100.0 * (df_vp["final_loss"] - base_loss_mean) / loss_denom
    df_vp["cos_pctdiff"]  = 100.0 * (df_vp["final_cos"]  - base_cos_mean)  / cos_denom

    # --- summarize by M (means only)
    summary = (
        df_vp.groupby("M", sort=True)
        .agg(
            n_trials=("M", "size"),
            loss_pctdiff_mean=("loss_pctdiff", "mean"),
            cos_pctdiff_mean=("cos_pctdiff", "mean"),
        )
        .reset_index()
        .sort_values("M")
    )

    x = summary["M"].to_numpy()
    loss_mean = summary["loss_pctdiff_mean"].to_numpy()
    cos_mean  = summary["cos_pctdiff_mean"].to_numpy()

    # ---- Single figure with twin y-axis ----
    fig, axL = plt.subplots(figsize=(9, 5), constrained_layout=True)

    axL.plot(x, loss_mean, marker="o", lw=2, label="Final loss % diff")
    axL.axhline(0.0, lw=1)
    axL.set_xlabel("Mantissa bits M")
    axL.set_ylabel("Final loss % diff vs float64")

    axR = axL.twinx()
    axR.axhline(0.0, lw=1)
    axR.plot(x, cos_mean, marker="o", color="red", lw=2, label="Final cos sim % diff")
    axR.set_ylabel("Final cos sim % diff vs float64", color="red")



    save_root = os.path.join(
        create_results_dir(),
        "vpfloats",
        f"Re={Re}_NDOF={NDOF}_T={T}",
    )
    os.makedirs(save_root, exist_ok=True)
    save_svg(mpl, fig, os.path.join(save_root, "pctdiff_loss_cos_v_M.svg"))

def avg_perf():
    Re = 100
    NDOF = 128
    dt = 1e-2
    T = 3.3
    St = 0
    beta = 0
    n = 4

    NT = 31
    n_part = 30
    optimizer = "L-BFGS_ArmBT-200"
    optimizer_pp = optimizer + "_PP"

    data_path = os.path.join(
        create_results_dir(),
        f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}-St={St}_beta={beta}_AI_pp",
        "results.parquet",
    )
    df = pd.read_parquet(data_path)
    df = df[(df["T"] == T) & (df["n_part"] == n_part) & (df["NT"] == NT)].copy()
    opt_df = df[df["optimizer"] == optimizer]
    opt_pp_df = df[df["optimizer"] == optimizer_pp]
    metrics = ["trj_cos_sim", "init_snap_cos_sim", "final_snap_cos_sim"]

    for metric in metrics:
        opt_mean = opt_df[metric].mean()
        opt_pp_mean = opt_pp_df[metric].mean()
        print(f"{metric}: {optimizer} mean = {opt_mean:.6g}, {optimizer_pp} mean = {opt_pp_mean:.6g}")


def optimal_m_plot():
    # parameters
    T = 3.3
    dt = 1e-2
    k = 0
    N = int(T / dt)

    LLE = 1 / (T)           
    m_list = [8]

    # time offsets T_{kj}
    j = np.arange(1, N + 1)
    T_kj = dt * ((j - 1) - k)

    # average time offset
    avg_T_term = np.mean(T_kj)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for m in m_list:
        p_j = m + (LLE / np.log(2)) * (T_kj - avg_T_term)
        p_j_r = np.round(p_j)
        print(np.mean(p_j_r))
        print(np.mean(p_j))
        #print(p_j[-1] - p_j[0])
        j = LLE * dt * j
        ax.plot(j, p_j_r, lw=2, label=f"$m={m}$")
        ax.plot(j, p_j, lw=2, label=f"$m={m}$")
    ax.set_xlabel("Time index $j$")
    ax.set_ylabel("Optimal mantissa bits $p_j$")
    ax.legend(title="Average mantissa budget")
    ax.grid(True, ls=":", alpha=0.5)

    plt.show()

if __name__ == "__main__":
    optimal_m_plot()