from create_results_dir import create_results_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from SRC.plotting_utils import save_svg
import matplotlib as mpl

def diss_v_time_plot():
    NDOF = 128
    dt = 1e-2
    seed_list = [1, 10]
    Re_list = [8, 22, 40, 100]
    root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate")


    # --- create figure + axes explicitly ---
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    for seed in seed_list:
        data_path = os.path.join(root, f"NDOF={NDOF}_dt={dt}_IC_seed={seed}")
        t = np.load(os.path.join(data_path, "t.npy"))

        for Re in Re_list:
            D = np.load(os.path.join(data_path, f"Re={Re}.npy"))
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

    # Legend 2: linestyles = IC seeds
    seed_handles = [
        Line2D([0], [0], color="k", lw=2,
               label=f"IC seed = {seed_list[0]}"),
        Line2D([0], [0], color="k", lw=2,
               label=f"IC seed = {seed_list[1]}"),
    ]
    ax.legend(
        handles=seed_handles,
        title="Initial condition",
        loc="upper center",
        frameon=True
    )

    save_svg(mpl, fig, os.path.join(root, f"dissipation_rate_comparison.svg"))


def avg_dist_plot():
    NDOF = 256
    dt = 1e-2
    seed = 1
    root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate")
    data_path = os.path.join(root, f"avgd_NDOF={NDOF}_dt={dt}_IC_seed={seed}")
    Re_list = np.load(os.path.join(data_path, "Re_list.npy"))
    avg_diss_list = np.load(os.path.join(data_path, "avg_diss_list.npy"))
    print(avg_diss_list)
    plt.figure(figsize=(6,4.8), dpi=120)
    plt.plot(Re_list, avg_diss_list, marker='o', linewidth=2)
    plt.xlabel("Reynolds Number")
    plt.ylabel("Average Dissipation Rate")
    plt.show()

if __name__ == "__main__":
    avg_dist_plot()