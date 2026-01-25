from create_results_dir import create_results_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from SRC.plotting_utils import save_svg
import matplotlib as mpl
from scipy.optimize import curve_fit

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
    cases = [
        # (Re, NDOF, dt)
        #(40,  128, 1e-2),
        (60,  128, 1e-2),
        (80,  128, 1e-2),
        (100,  128, 1e-2),
        (120, 128, 1e-2),
        (140, 128, 1e-2),
        (160, 128, 1e-2),
        (180, 128, 1e-2),
        (200, 256, 2.5e-3),
        (220, 256, 2.5e-3),
    ]

    root = os.path.join(create_results_dir(), "Trjs", "Dissipation_Rate")
    data_path = os.path.join(root, f"diss_v_time_IC_seed={seed}")

    avg_diss_list = []
    Re_list = []

    for Re, NDOF, dt in cases:
        D = np.load(os.path.join(data_path, f"dissnorm_Re={Re}_N={NDOF}_dt={dt}.npy"))
        print(np.max(D))
        avg_diss_list.append(float(np.mean(D)))
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


if __name__ == "__main__":
    avg_dist_plot()