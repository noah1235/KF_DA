import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from SRC.plotting_utils import save_svg
import matplotlib as mpl
from create_results_dir import create_results_dir
from SRC.DA_Comp.case_post_proc import radial_spectral_error

def plot_opt_comp(optimizer, opt_df, root):
    # Figure for average optimizer performance (across all optimizers)
    fig_avg_v_its, ax_avg_v_its = plt.subplots()
    

    # Stack traces: shape (num_runs, num_iters)
    loss_traces = np.vstack(opt_df["loss_record"].to_numpy())

    final_loss = loss_traces[:, -1]

    # Average performance across runs for this optimizer
    avg_loss_trace = np.mean(loss_traces, axis=0)

    ax_avg_v_its.plot(avg_loss_trace,label=str(optimizer) )

    ax_avg_v_its.set_yscale("log")
    ax_avg_v_its.set_xlabel("cost")
    ax_avg_v_its.set_ylabel("loss")
    ax_avg_v_its.legend()
    fig_avg_v_its.tight_layout()
    fig_avg_v_its.savefig(os.path.join(root, "avg_opt_perf_v_its.png"))
    plt.close(fig_avg_v_its)

def save_histogram(
    data,
    save_path,
    bins=10,
    density=False,
    log_y=False,
    title="Histogram",
    xlabel="Value",
    ylabel=None,
    eps=1e-18,
):
    """
    Plot and save a histogram of a 1D array using log-spaced bins.
    Uses save_svg(mpl, fig, save_path).

    Notes:
      - Log-spaced bins require strictly positive data, so nonpositive values
        are dropped automatically.
      - x-axis is set to log scale automatically.
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]  # drop NaN/inf

    # log bins require positive values only
    data = data[data > 0]
    if data.size == 0:
        raise ValueError("Histogram requires positive values for log-spaced bins, but none were found.")

    if ylabel is None:
        ylabel = "Density" if density else "Count"

    dmin = max(data.min(), eps)
    dmax = data.max()

    # build log-spaced bin edges
    if dmax <= dmin:
        bin_edges = np.array([dmin, dmin * 1.01])
    else:
        bin_edges = np.logspace(np.log10(dmin), np.log10(dmax), bins + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bin_edges, density=density)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    fig.tight_layout()
    save_svg(mpl, fig, save_path)
    plt.close(fig)

def loss_transformation(loss, k=1.5, eps=0):
    loss = np.asarray(loss).ravel()
    loss = loss[np.isfinite(loss)]
    loss = loss[loss > 0]

    log_loss = np.log(loss + eps)

    q1, q3 = np.percentile(log_loss, [25, 75])
    iqr = (q3 - q1) + eps
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    inliers = (log_loss >= lo) & (log_loss <= hi)

    mu = np.mean(log_loss[inliers])
    sigma = np.std(log_loss[inliers]) + eps

    return (log_loss - mu) / sigma

def plot_recon_vs_loss(recon, loss, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(loss, recon)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    fig.tight_layout()
    save_svg(mpl, fig, save_path)
    plt.close(fig)

def plot_performance(fp_df, save_root):
    loss_traces = np.vstack(fp_df["loss_record"].to_numpy())
    final_loss = loss_traces[:, -1]
    save_histogram(final_loss, os.path.join(save_root, "loss_hist.svg"))
    loss_stand = loss_transformation(final_loss)
    plot_recon_vs_loss(fp_df["final_snap_cos_sim"], loss_stand, os.path.join(save_root, "final_snap_cos_sim_vs_loss.svg"), (0, 1))

def plot_feilds(
    fp_df,
    T, NP, NT,
    crit, optimizer, fp,
    save_root, base_root,
    max_k=15,
    nbins=8,
    log_bins=False,
):
    os.makedirs(save_root, exist_ok=True)

    # --- accumulators
    k_ref = None

    E_true_all   = []
    E_DA_all     = []
    E_guess_all  = []

    relerr_DA_all    = []
    relerr_guess_all = []

    for IC_trg_seed, IC_trg_seed_df in fp_df.groupby("true_IC_seed"):
        trg_path = os.path.join(
            base_root,
            f"IC_seed={IC_trg_seed}/T={T}/omega_trg_trj.npy",
        )
        omega_trg_trj = np.load(trg_path)

        for PIC_seed, PIC_seed_df in IC_trg_seed_df.groupby("PIC_seed"):
            for init_IC_seed, _ in PIC_seed_df.groupby("init_IC_seed"):

                data_root = os.path.join(
                    base_root,
                    f"IC_seed={IC_trg_seed}/T={T}/np={NP}/NT={NT}/PI/seed_{PIC_seed}"
                    f"/{crit}/{optimizer}/{fp}/Fourier_K=64/cases/{init_IC_seed}"
                )

                omega_DA_trj = np.load(os.path.join(data_root, "omega_DA_trj.npy"))
                omega_guess_trj = np.load(os.path.join(data_root, "omega_guess_trj.npy"))

                # --- use IC (change to [-1] for final condition)
                true_hat  = omega_trg_trj[0]
                DA_hat    = omega_DA_trj[0]
                guess_hat = omega_guess_trj[0]

                # --- DA vs true
                k_centers, rel_err_DA, E_true_k, E_DA_k = radial_spectral_error(
                    DA_hat,
                    true_hat,
                    log_bins=log_bins,
                    k_max=max_k,
                    nbins=nbins,
                    fft_input=True,
                )

                # --- guess vs true
                k_centers2, rel_err_guess, E_true_k2, E_guess_k = radial_spectral_error(
                    guess_hat,
                    true_hat,
                    log_bins=log_bins,
                    k_max=max_k,
                    nbins=nbins,
                    fft_input=True,
                )

                # --- enforce consistent bins
                if k_ref is None:
                    k_ref = np.asarray(k_centers)
                else:
                    if np.max(np.abs(np.asarray(k_centers) - k_ref)) > 1e-12:
                        raise ValueError("k_centers mismatch across cases")

                # --- store
                E_true_all.append(np.asarray(E_true_k))
                E_DA_all.append(np.asarray(E_DA_k))
                E_guess_all.append(np.asarray(E_guess_k))

                relerr_DA_all.append(np.asarray(rel_err_DA))
                relerr_guess_all.append(np.asarray(rel_err_guess))

    # --- stack + statistics
    E_true_all   = np.stack(E_true_all, axis=0)
    E_DA_all     = np.stack(E_DA_all, axis=0)
    E_guess_all  = np.stack(E_guess_all, axis=0)

    relerr_DA_all    = np.stack(relerr_DA_all, axis=0)
    relerr_guess_all = np.stack(relerr_guess_all, axis=0)

    E_true_mu  = E_true_all.mean(axis=0)
    E_DA_mu    = E_DA_all.mean(axis=0)
    E_guess_mu = E_guess_all.mean(axis=0)

    rel_DA_mu    = relerr_DA_all.mean(axis=0)
    rel_guess_mu = relerr_guess_all.mean(axis=0)

    # optional spread
    E_true_sd  = E_true_all.std(axis=0)
    E_DA_sd    = E_DA_all.std(axis=0)
    E_guess_sd = E_guess_all.std(axis=0)

    rel_DA_sd    = relerr_DA_all.std(axis=0)
    rel_guess_sd = relerr_guess_all.std(axis=0)

    # =========================
    # Plot 1: Energy spectra
    # =========================
    fig1, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(k_ref, E_true_mu,  label="True",  lw=2)
    ax1.plot(k_ref, E_DA_mu,    label="DA",    lw=2)
    ax1.plot(k_ref, E_guess_mu, label="Guess", lw=2)

    ax1.fill_between(k_ref, E_true_mu - E_true_sd,  E_true_mu + E_true_sd,  alpha=0.2)
    ax1.fill_between(k_ref, E_DA_mu   - E_DA_sd,    E_DA_mu   + E_DA_sd,    alpha=0.2)
    ax1.fill_between(k_ref, E_guess_mu - E_guess_sd, E_guess_mu + E_guess_sd, alpha=0.2)

    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$E(k)$")
    ax1.set_yscale("log")
    ax1.set_title("Average Energy Spectrum")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    fig1.savefig(os.path.join(save_root, "avg_energy_spectrum.png"), dpi=200)
    plt.close(fig1)

    # =========================
    # Plot 2: Relative error
    # =========================
    fig2, ax2 = plt.subplots(figsize=(9, 6))

    ax2.plot(k_ref, rel_DA_mu,    label="DA / True",    lw=2)
    ax2.plot(k_ref, rel_guess_mu, label="Guess / True", lw=2)

    ax2.fill_between(k_ref, rel_DA_mu - rel_DA_sd, rel_DA_mu + rel_DA_sd, alpha=0.2)
    ax2.fill_between(k_ref, rel_guess_mu - rel_guess_sd, rel_guess_mu + rel_guess_sd, alpha=0.2)

    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel("Relative Error")
    ax2.set_yscale("log")
    ax2.set_title("Average Relative Spectral Error")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    fig2.savefig(os.path.join(save_root, "avg_relative_error.png"), dpi=200)
    plt.close(fig2)

    
                

def global_post_main(df: pd.DataFrame, base_root: str) -> None:
    """
    Post-process global optimization results stored in a DataFrame and
    save summary plots grouped by sampling period, number of particles,
    loss criterion, and optimizer.

    Expected DataFrame columns:
        - "samp_period"
        - "n_part"
        - "loss_crit"
        - "optimizer"
        - "loss_record"
        - "loss_evals_record"
        - "loss_grad_evals_record"
        - "Hvp_evals_record"
        - "loss_avg_eval_time"
        - "loss_grad_avg_eval_time"
        - "Hvp_avg_eval_time"
    """
    root = os.path.join(base_root, "global_results")
    os.makedirs(root, exist_ok=True)

    for T, df_T in df.groupby("T"):
        T_root = os.path.join(root, f"T={T}")
        for NT, df_NT in df_T.groupby("NT"):
            NT_root = os.path.join(T_root, f"NT={NT}")
            for n_part, df_part in df_NT.groupby("n_part"):
                for loss_crit, crit_df in df_part.groupby("loss_crit"):
                    loss_crit_root = os.path.join(NT_root, f"np={n_part}", str(loss_crit))
                    os.makedirs(loss_crit_root, exist_ok=True)
                    for optimizer, opt_df in crit_df.groupby("optimizer"):
                        opt_root = os.path.join(loss_crit_root, optimizer)
                        for fp, fp_df in crit_df.groupby("floatp"):
                            fp_root = os.path.join(opt_root, fp)
                            for IC_param, IC_param_df in crit_df.groupby("IC_param"):
                                IC_param_root = os.path.join(fp_root, f"{IC_param}")
                                os.makedirs(IC_param_root, exist_ok=True)
                                plot_opt_comp(optimizer, IC_param_df, IC_param_root)
                                plot_performance(IC_param_df, IC_param_root)
                                plot_feilds(IC_param_df, T, n_part, NT, loss_crit, optimizer, fp, IC_param_root, base_root)


