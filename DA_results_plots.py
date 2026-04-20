from create_results_dir import create_results_dir
import os
import pandas as pd
from SRC.global_post.global_post_main import loss_transformation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SRC.plotting_utils import save_svg
import matplotlib as mpl
import re

def m_dep_fig():
    # -----------------------
    # Configuration
    # -----------------------
    Re = 200
    n = 4
    dt = 2.5e-3
    NDOF = 256
    St = 0
    beta = 0
    m_dt = None

    m_targets = [800]
    metric = "final_snap_rel_error"
    for m_target in m_targets:
        # -----------------------
        # Load results
        # -----------------------
        root = os.path.join(
            create_results_dir(),
            "DA-no_noise",
            f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
        )
        save_root = os.path.join(root, "global_results", "mx_v_mt")
        os.makedirs(save_root, exist_ok=True)
        df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()

        # -----------------------
        # Find matching runs and plot
        # m = (NT - 1) * (2 * n_part)
        # -----------------------
        found = False
        fig = plt.figure()
        for (n_part, NT), g in df.groupby(["n_part", "NT"], sort=True):
            mx = 2 * n_part
            m = NT * mx
            if m != m_target:
                continue

            found = True

            perf = g[metric].to_numpy()

            # loss_record is assumed to be array-like per row with consistent length
            loss_traces = np.vstack(g["loss_record"].to_numpy())
            final_loss = loss_traces[:, -1]
            
            #loss_stand = loss_transformation(final_loss)

            perf, final_loss = remove_outliers_by_loss(g, metric)

            loss_stand = final_loss

            label = f"NT={NT}, n_part={n_part}, mean={np.mean(perf):.3f}"
            plt.scatter(loss_stand, perf, label=label)
        if not found:
            print(f"No runs found with m_target={m_target} (m = (NT-1)*(2*n_part)).")

        plt.legend()
        plt.title(f"m = {m_target} | metric = {metric}")
        plt.xlabel("loss")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.xscale("log")
        plt.ylim(0, 1)
        #plt.xlim(1e-8, 1e-3)
        save_svg(mpl, fig, os.path.join(save_root, f"m={m_target}.svg"))
        plt.close(fig)
        
def remove_outliers_by_loss(df, metric):
    loss_traces = np.vstack(df["loss_record"].to_numpy())
    final_loss = loss_traces[:, -1]

    median = np.median(final_loss)
    mad = np.median(np.abs(final_loss - median))

    modified_z = 0.6745 * (final_loss - median) / mad

    mask = np.abs(modified_z) < 3.5

    metric_arr = df.loc[mask, metric].to_numpy()
    final_loss = final_loss[mask]

    return metric_arr, final_loss

def recon_v_m_dt():
    Re = 100
    n = 4
    dt = 0.01
    NDOF = 128
    St = 0
    beta = 0
    NT = 4
    n_part = 80


    metric = "final_snap_rel_error"
    loss_crit = "PP_MSE"
    #loss_crit = "Vel_MSE"

    pattern = re.compile(
        rf"^DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt=(\d*\.?\d+)-St={St}_beta={beta}_AI$"
    )

    base_dir = create_results_dir()

    m_dt_vals = []
    mean_metric_vals = []
    mean_final_loss_vals = []

    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue

        match = pattern.match(name)
        if match is None:
            continue

        results_path = os.path.join(full_path, "results.parquet")
        if not os.path.exists(results_path):
            continue

        mdt = float(match.group(1))

        df = pd.read_parquet(results_path).dropna()
        df = df[
            (df["n_part"] == n_part)
            & (df["NT"] == NT)
            & (df["loss_crit"] == loss_crit)
        ]

        if df.empty:
            continue

        metric_arr, final_loss = remove_outliers_by_loss(df, metric)


        m_dt_vals.append(mdt)
        mean_metric_vals.append(np.mean(metric_arr))
        mean_final_loss_vals.append(np.mean(final_loss))

    if len(m_dt_vals) == 0:
        print("No matching runs found.")
        return

    m_dt_vals = np.array(m_dt_vals)
    mean_metric_vals = np.array(mean_metric_vals)
    mean_final_loss_vals = np.array(mean_final_loss_vals)

    sort_idx = np.argsort(m_dt_vals)
    m_dt_vals = m_dt_vals[sort_idx]
    mean_metric_vals = mean_metric_vals[sort_idx]
    mean_final_loss_vals = mean_final_loss_vals[sort_idx]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # ---- Primary axis (metric) ----
    color1 = "tab:blue"
    l1 = ax1.plot(
        m_dt_vals,
        mean_metric_vals,
        marker="o",
        color=color1,
        label=metric,
    )

    ax1.set_xlabel(r"$\Delta t_m$", fontsize=12)
    ax1.set_ylabel(metric, color=color1, fontsize=12)
    ax1.set_xlim(0.1, 0.8)


    # ---- Secondary axis (loss) ----
    ax2 = ax1.twinx()

    color2 = "tab:red"
    l2 = ax2.plot(
        m_dt_vals,
        mean_final_loss_vals,
        marker="s",
        color=color2,
        label="Mean Final Loss",
    )

    ax2.set_ylabel("Mean Final Loss", color=color2, fontsize=12)

    # ---- Combined legend ----
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=False)

    # ---- Title ----
    plt.title(rf"Loss Criterion: {loss_crit}", fontsize=13)

    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(base_dir, f"recon_v_m_dt.svg"))
    plt.close(fig)

def embedding_fig():
    # -----------------------
    # Configuration
    # -----------------------
    n = 4
    #dt = 0.01
    #NDOF = 128
    St = 0
    beta = 0
    m_dt = None

    noise_type = "DA-no_noise"

    #Re, NDOF, dt
    config_list = [
        (200, 256, 2.5e-3, 8),
        (100, 128, 0.01, 4),
        (40, 128, 0.01, 6),
        (60, 128, 0.01, 6)
    ]

    dM_dict = {
        200: 420,
        100: 300,
        60: 240,
        40: 160
    }

    metric = "final_snap_rel_error"
    loss_crit = "PP_MSE"

    Re_list = []
    dM_list = []
    m_list = []
    metric_list = []

    for Re, NDOF, dt, NT in config_list:
        print(Re)
        root = os.path.join(
            create_results_dir(),
            noise_type,
            f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
        )

        df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()

        df = df[
            (df["NT"] == NT)
            & (df["loss_crit"] == loss_crit)
        ]

        for n_part, df_npart in df.groupby("n_part", sort=True):
            print(n_part)
            metric_arr, final_loss = remove_outliers_by_loss(df_npart, metric)

            if len(metric_arr) == 0:
                continue

            mean_metric = np.mean(metric_arr)

            # compute m
            m = NT * n_part * 2

            Re_list.append(Re)
            dM_list.append(dM_dict[Re])
            m_list.append(m)
            metric_list.append(mean_metric)

    # -----------------------
    # Plot
    # -----------------------
    Re_arr = np.array(Re_list)
    dM_arr = np.array(dM_list)
    m_arr = np.array(m_list)
    metric_arr = np.array(metric_list)

    dM_x = np.linspace(np.min(dM_arr), np.max(dM_arr), 100)
    im_line = dM_x
    emb_line = 2*dM_x+1

    fig = plt.figure(figsize=(6, 5))

    sc = plt.scatter(
        dM_arr,
        m_arr,
        c=metric_arr,
        s=80,
        vmin=0.0,      # minimum color value
        vmax=0.5       # maximum color value
    )
    plt.plot(dM_x, im_line, label="immersion line")
    plt.plot(dM_x, emb_line, label="embedding line")

    cbar = plt.colorbar(sc)
    cbar.set_label("Mean Metric")

    plt.xlabel("IM dimension")
    plt.ylabel("m = NT * n_part * 2")

    plt.tight_layout()
    plt.legend()
    save_svg(mpl, fig, os.path.join(create_results_dir(), noise_type, "embedding_fig.svg"))

embedding_fig()