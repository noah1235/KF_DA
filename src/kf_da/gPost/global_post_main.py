import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from SRC.plotting_utils import save_svg
import matplotlib as mpl
from ..create_results_dir import..create_results_dir
from SRC.DA_Comp.case_post_proc import radial_spectral_error
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.stats as stats

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
    plt.scatter(loss, recon, s=10, alpha=1.0, color="red")
    plt.hist2d(loss, recon, bins=20, cmap="viridis", alpha=0.6)
    plt.colorbar()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlim(-3, 3)
    fig.tight_layout()
    save_svg(mpl, fig, save_path)
    plt.close(fig)

def fd_bins(x):
    """Compute number of bins using Freedman–Diaconis rule."""
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return 1

    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.sqrt(n))  # fallback

    bin_width = 2 * iqr * n ** (-1/3)
    if bin_width == 0:
        return int(np.sqrt(n))

    return max(1, int(np.ceil((x.max() - x.min()) / bin_width)))


def plot_recon_conditional(recon, loss, save_path, per_list=(10, 20, 50, 100)):
    """
    Each row: histogram of recon conditioned on best x% of loss.
    Also saves a second figure of mean ± std vs percent.
    """
    recon = np.asarray(recon).ravel()
    loss  = np.asarray(loss).ravel()
    assert recon.shape == loss.shape, "recon and loss must have the same shape"

    n = len(loss)
    order = np.argsort(loss)  # ascending: best first

    # store stats for summary plot
    means = []
    stds = []
    counts = []

    # -----------------------
    # Histogram figure
    # -----------------------
    fig, axes = plt.subplots(
        nrows=len(per_list), ncols=1,
        figsize=(7, 2.2 * len(per_list)),
        sharex=True,
        constrained_layout=True
    )
    if len(per_list) == 1:
        axes = [axes]

    for ax, p in zip(axes, per_list):
        k = max(1, int(np.ceil(p / 100 * n)))
        idx = order[:k]
        recon_p = recon[idx]

        mu = np.mean(recon_p)
        sigma = np.std(recon_p)

        means.append(mu)
        stds.append(sigma)
        counts.append(k)

        # histogram
        ax.hist(recon_p, bins="fd", density=True)
        ax.axvline(mu, linestyle="--")

        ax.text(
            0.02, 0.95,
            f"μ={mu:.3f}, σ={sigma:.3f}",
            transform=ax.transAxes,
            verticalalignment="top"
        )

        ax.set_ylabel("density")
        ax.set_title(f"recon | loss in best {p}% (n={k})")

    axes[-1].set_xlabel("recon")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    # -----------------------
    # Mean vs percent figure
    # -----------------------
    per_array = np.array(per_list)
    means = np.array(means)
    stds = np.array(stds)

    fig2 = plt.figure(figsize=(6, 4))
    plt.errorbar(per_array, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("Best loss percentile (%)")
    plt.ylabel("recon mean ± std")
    plt.title("Conditional recon statistics")
    plt.grid(True, alpha=0.3)

    # save with modified filename
    base, ext = save_path.rsplit(".", 1)
    save_path_stats = f"{base}_stats.{ext}"
    fig2.savefig(save_path_stats, dpi=200)
    plt.close(fig2)


#metric="final_snap_cos_sim"
def plot_performance(fp_df, save_root, metric="final_snap_rel_error"):
    metric="final_snap_cos_sim"
    loss_traces = np.vstack(fp_df["loss_record"].to_numpy())
    final_loss = loss_traces[:, -1]
    save_histogram(final_loss, os.path.join(save_root, "loss_hist.svg"))
    loss_stand = loss_transformation(final_loss)

    fig = plt.figure()
    stats.probplot(loss_stand, dist="norm", plot=plt)

    fig.savefig(os.path.join(save_root, "qq_plot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_recon_vs_loss(fp_df[f"{metric}"], loss_stand, os.path.join(save_root, f"{metric}_vs_loss.svg"), (0, 1))
    plot_recon_conditional(fp_df[f"{metric}"], loss_stand, os.path.join(save_root, f"{metric}_loss_cond.svg"), per_list=(10, 20, 50, 100))


def plot_feilds(
    fp_df,
    T, NP, NT,
    crit, optimizer, fp,
    save_root, base_root,
    max_k=15,
    nbins=8,
    log_bins=False,
    n_workers=None,  # None -> default chosen by ThreadPoolExecutor
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(save_root, exist_ok=True)

    # -------------------------
    # Helper: plotters
    # -------------------------
    def plot_energy(k, E_true_mu, E_true_sd, E_DA_mu, E_DA_sd, E_guess_mu, E_guess_sd, title, outname):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(k, E_true_mu,  label="True",  lw=2)
        ax.plot(k, E_DA_mu,    label="DA",    lw=2)
        ax.plot(k, E_guess_mu, label="Guess", lw=2)
        ax.fill_between(k, E_true_mu - E_true_sd,   E_true_mu + E_true_sd,   alpha=0.2)
        ax.fill_between(k, E_DA_mu   - E_DA_sd,     E_DA_mu   + E_DA_sd,     alpha=0.2)
        ax.fill_between(k, E_guess_mu - E_guess_sd, E_guess_mu + E_guess_sd, alpha=0.2)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$E(k)$")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.savefig(os.path.join(save_root, outname), dpi=200)
        plt.close(fig)

    def plot_relerr(k, rel_DA_mu, rel_DA_sd, rel_guess_mu, rel_guess_sd, title, outname):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(k, rel_DA_mu,    label="DA / True",    lw=2)
        ax.plot(k, rel_guess_mu, label="Guess / True", lw=2)
        ax.fill_between(k, rel_DA_mu - rel_DA_sd,       rel_DA_mu + rel_DA_sd,       alpha=0.2)
        ax.fill_between(k, rel_guess_mu - rel_guess_sd, rel_guess_mu + rel_guess_sd, alpha=0.2)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel("Relative Error")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.savefig(os.path.join(save_root, outname), dpi=200)
        plt.close(fig)

    def stack_stats(arr_list):
        arr = np.stack(arr_list, axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    # -------------------------
    # Build list of cases to run
    # -------------------------
    cases = []
    for IC_trg_seed, IC_trg_seed_df in fp_df.groupby("true_IC_seed"):
        for PIC_seed, PIC_seed_df in IC_trg_seed_df.groupby("PIC_seed"):
            for init_IC_seed, _ in PIC_seed_df.groupby("init_IC_seed"):
                cases.append((IC_trg_seed, PIC_seed, init_IC_seed))

    # -------------------------
    # Per-case worker
    # -------------------------
    def run_case(case):
        IC_trg_seed, PIC_seed, init_IC_seed = case

        trg_path = os.path.join(
            base_root,
            f"IC_seed={IC_trg_seed}/T={T}/omega_trg_trj.npy",
        )
        omega_trg_trj = np.load(trg_path)

        data_root = os.path.join(
            base_root,
            f"IC_seed={IC_trg_seed}/T={T}/np={NP}/NT={NT}/PI/seed_{PIC_seed}"
            f"/{crit}/{optimizer}/{fp}/Fourier_K=64/cases/{init_IC_seed}"
        )
        omega_DA_trj = np.load(os.path.join(data_root, "omega_DA_trj.npy"))
        omega_guess_trj = np.load(os.path.join(data_root, "omega_guess_trj.npy"))

        # INITIAL
        true_hat_IC  = omega_trg_trj[0]
        DA_hat_IC    = omega_DA_trj[0]
        guess_hat_IC = omega_guess_trj[0]

        k_IC, rel_DA_IC, E_true_IC, E_DA_IC = radial_spectral_error(
            DA_hat_IC, true_hat_IC,
            log_bins=log_bins, k_max=max_k, nbins=nbins, fft_input=True
        )
        k2_IC, rel_guess_IC, E_true2_IC, E_guess_IC = radial_spectral_error(
            guess_hat_IC, true_hat_IC,
            log_bins=log_bins, k_max=max_k, nbins=nbins, fft_input=True
        )

        # FINAL
        true_hat_F  = omega_trg_trj[-1]
        DA_hat_F    = omega_DA_trj[-1]
        guess_hat_F = omega_guess_trj[-1]

        k_F, rel_DA_F, E_true_F, E_DA_F = radial_spectral_error(
            DA_hat_F, true_hat_F,
            log_bins=log_bins, k_max=max_k, nbins=nbins, fft_input=True
        )
        k2_F, rel_guess_F, E_true2_F, E_guess_F = radial_spectral_error(
            guess_hat_F, true_hat_F,
            log_bins=log_bins, k_max=max_k, nbins=nbins, fft_input=True
        )

        # Return everything needed; main thread will enforce k consistency
        return {
            "k_IC": np.asarray(k_IC),
            "k_F":  np.asarray(k_F),

            "E_true_IC":  np.asarray(E_true_IC),
            "E_DA_IC":    np.asarray(E_DA_IC),
            "E_guess_IC": np.asarray(E_guess_IC),
            "rel_DA_IC":  np.asarray(rel_DA_IC),
            "rel_guess_IC": np.asarray(rel_guess_IC),

            "E_true_F":  np.asarray(E_true_F),
            "E_DA_F":    np.asarray(E_DA_F),
            "E_guess_F": np.asarray(E_guess_F),
            "rel_DA_F":  np.asarray(rel_DA_F),
            "rel_guess_F": np.asarray(rel_guess_F),
        }

    # -------------------------
    # Run in parallel (threads)
    # -------------------------
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_case, c) for c in cases]
        for fut in as_completed(futures):
            results.append(fut.result())

    # -------------------------
    # Enforce consistent k bins
    # -------------------------
    k_ref = results[0]["k_IC"]
    for r in results[1:]:
        if np.max(np.abs(r["k_IC"] - k_ref)) > 1e-12:
            raise ValueError("k_centers mismatch across cases (IC)")
        if np.max(np.abs(r["k_F"] - k_ref)) > 1e-12:
            raise ValueError("k_centers mismatch across cases (Final vs ref)")

    # -------------------------
    # Gather arrays
    # -------------------------
    E_true_all_IC   = [r["E_true_IC"] for r in results]
    E_DA_all_IC     = [r["E_DA_IC"] for r in results]
    E_guess_all_IC  = [r["E_guess_IC"] for r in results]
    rel_DA_all_IC   = [r["rel_DA_IC"] for r in results]
    rel_guess_all_IC= [r["rel_guess_IC"] for r in results]

    E_true_all_F    = [r["E_true_F"] for r in results]
    E_DA_all_F      = [r["E_DA_F"] for r in results]
    E_guess_all_F   = [r["E_guess_F"] for r in results]
    rel_DA_all_F    = [r["rel_DA_F"] for r in results]
    rel_guess_all_F = [r["rel_guess_F"] for r in results]

    # -------------------------
    # Stats
    # -------------------------
    E_true_mu_IC,  E_true_sd_IC  = stack_stats(E_true_all_IC)
    E_DA_mu_IC,    E_DA_sd_IC    = stack_stats(E_DA_all_IC)
    E_guess_mu_IC, E_guess_sd_IC = stack_stats(E_guess_all_IC)
    rel_DA_mu_IC,  rel_DA_sd_IC  = stack_stats(rel_DA_all_IC)
    rel_guess_mu_IC, rel_guess_sd_IC = stack_stats(rel_guess_all_IC)

    E_true_mu_F,  E_true_sd_F  = stack_stats(E_true_all_F)
    E_DA_mu_F,    E_DA_sd_F    = stack_stats(E_DA_all_F)
    E_guess_mu_F, E_guess_sd_F = stack_stats(E_guess_all_F)
    rel_DA_mu_F,  rel_DA_sd_F  = stack_stats(rel_DA_all_F)
    rel_guess_mu_F, rel_guess_sd_F = stack_stats(rel_guess_all_F)

    # -------------------------
    # Plots: INITIAL
    # -------------------------
    plot_energy(
        k_ref,
        E_true_mu_IC, E_true_sd_IC,
        E_DA_mu_IC, E_DA_sd_IC,
        E_guess_mu_IC, E_guess_sd_IC,
        title="Average Energy Spectrum (Initial Condition)",
        outname="avg_energy_spectrum_IC.png",
    )
    plot_relerr(
        k_ref,
        rel_DA_mu_IC, rel_DA_sd_IC,
        rel_guess_mu_IC, rel_guess_sd_IC,
        title="Average Relative Spectral Error (Initial Condition)",
        outname="avg_relative_error_IC.png",
    )

    # -------------------------
    # Plots: FINAL
    # -------------------------
    plot_energy(
        k_ref,
        E_true_mu_F, E_true_sd_F,
        E_DA_mu_F, E_DA_sd_F,
        E_guess_mu_F, E_guess_sd_F,
        title="Average Energy Spectrum (Final Condition)",
        outname="avg_energy_spectrum_final.png",
    )
    plot_relerr(
        k_ref,
        rel_DA_mu_F, rel_DA_sd_F,
        rel_guess_mu_F, rel_guess_sd_F,
        title="Average Relative Spectral Error (Final Condition)",
        outname="avg_relative_error_final.png",
    )


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
                                #plot_feilds(IC_param_df, T, n_part, NT, loss_crit, optimizer, fp, IC_param_root, base_root)



