from create_results_dir import create_results_dir
import os
import pandas as pd
from SRC.global_post.global_post_main import loss_transformation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SRC.plotting_utils import save_svg
import matplotlib as mpl
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

    m_targets = [640, 320, 160]
    metric = "final_snap_rel_error"
    for m_target in m_targets:
        # -----------------------
        # Load results
        # -----------------------
        root = os.path.join(
            create_results_dir(),
            f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}-St={St}_beta={beta}_AI",
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
            m = (NT - 1) * mx
            if m != m_target:
                continue

            found = True

            perf = g[metric].to_numpy()

            # loss_record is assumed to be array-like per row with consistent length
            loss_traces = np.vstack(g["loss_record"].to_numpy())
            final_loss = loss_traces[:, -1]

            loss_stand = loss_transformation(final_loss)

            label = f"NT={NT}, n_part={n_part}, mean={np.mean(perf):.3f}"
            plt.scatter(loss_stand, perf, label=label)
        if not found:
            print(f"No runs found with m_target={m_target} (m = (NT-1)*(2*n_part)).")

        plt.legend()
        plt.title(f"m = {m_target} | metric = {metric}")
        plt.xlabel("loss_transformation(final_loss)")
        plt.ylabel(metric)
        plt.tight_layout()
        save_svg(mpl, fig, os.path.join(save_root, f"m={m_target}.svg"))
        plt.close(fig)
        
m_dep_fig()