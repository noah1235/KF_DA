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
    Re = 100
    n = 4
    dt = 0.01
    NDOF = 128
    St = 0
    beta = 0
    m_dt = None

    m_targets = [640, 320, 160]
    metric = "final_snap_rel_error"
    for m_target in m_targets:
        # -----------------------
        # Load results
        # -----------------------
        root = os.path.join(
            create_results_dir(),
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
            loss_stand = final_loss
            #loss_stand = loss_transformation(final_loss)

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


    pattern = re.compile(
        rf"^DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt=(\d*\.?\d+)-St={St}_beta={beta}_AI$"
    )

    base_dir = create_results_dir()

    perf_list = []
    m_dt_list = []

    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)

        if os.path.isdir(full_path):
            match = pattern.match(name)
            if match:
                mdt = float(match.group(1))  # extract mdt
                m_dt_list.append(mdt)
                df = pd.read_parquet(os.path.join(base_dir, name, "results.parquet")).dropna()
                df = df[
                    (df["n_part"] == n_part) &
                    (df["NT"] == NT) &
                    (df["loss_crit"] == loss_crit)
                ]

                loss_traces = np.vstack(df["loss_record"].to_numpy())
                final_loss = loss_traces[:, -1]
                mask = final_loss <= 5e-4
                perf_arr = df[metric].to_numpy()[mask]
                print(perf_arr.shape)
                perf = np.mean(perf_arr)
                perf_list.append(perf)

    m_dt_list = np.array(m_dt_list)
    perf_list = np.array(perf_list)
    idx = np.argsort(m_dt_list)
    m_dt_list = m_dt_list[idx]
    perf_list = perf_list[idx]

    #m_dt_list = m_dt_list[1:]
    #perf_list = perf_list[1:]
    plt.scatter(m_dt_list, perf_list)
    plt.plot()
    #plt.ylim(0, 1.05)
    plt.xlim(.1, .8)
    plt.show()


recon_v_m_dt()