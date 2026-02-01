import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from SRC.plotting_utils import save_svg
import matplotlib as mpl

def plot_opt_comp(crit_df, loss_crit_root, avg_loss_grad_cost=2, avg_Hvp_cost=5):
    # Figure for average optimizer performance (across all optimizers)
    fig_avg, ax_avg = plt.subplots()
    fig_avg_v_its, ax_avg_v_its = plt.subplots()
    
    for optimizer, opt_df in crit_df.groupby("optimizer"):
        save_root = os.path.join(loss_crit_root, str(optimizer))
        os.makedirs(save_root, exist_ok=True)

        # Stack traces: shape (num_runs, num_iters)
        loss_traces = np.vstack(opt_df["loss_record"].to_numpy())
        loss_evals_record = np.vstack(opt_df["loss_evals_record"].to_numpy())
        loss_grad_evals_record = np.vstack(opt_df["loss_grad_evals_record"].to_numpy())
        Hvp_evals_record = np.vstack(opt_df["Hvp_evals_record"].to_numpy())

        final_loss = loss_traces[:, -1]

        # Broadcast costs over iterations
        total_cost = (
            loss_evals_record
            + avg_Hvp_cost * loss_grad_evals_record
            + avg_Hvp_cost * Hvp_evals_record
        )
        # Average performance across runs for this optimizer
        avg_loss_trace = np.mean(loss_traces, axis=0)
        avg_cost_trace = np.mean(total_cost, axis=0)

        # Loss vs cost for each run
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(total_cost.T, loss_traces.T)
        ax.set_yscale("log")
        ax.set_xlabel("cost")
        ax.set_ylabel("loss")
        plt.title(f"avg loss grad cost: {avg_loss_grad_cost:.2f} | avg Hvp cost: {avg_Hvp_cost:.2f}")
        fig.tight_layout()
        save_svg(mpl, fig, os.path.join(save_root, "loss_v_cost.svg"))
        plt.close(fig)

        # Cost vs iteration for each run
        if False:
            fig, ax = plt.subplots()
            ax.plot(total_cost.T)
            ax.set_ylabel("cost")
            ax.set_xlabel("iteration")
            fig.tight_layout()
            save_svg(mpl, fig, os.path.join(save_root, "cost_v_its.svg"))
            plt.close(fig)


        ax_avg.plot(avg_cost_trace, avg_loss_trace, label=str(optimizer))
        ax_avg_v_its.plot(avg_loss_trace,label=str(optimizer) )
        
    if False:
        # Finalize and save average performance plot
        ax_avg.set_yscale("log")
        ax_avg.set_xlabel("cost")
        ax_avg.set_ylabel("loss")
        ax_avg.legend()
        fig_avg.tight_layout()
        fig_avg.savefig(os.path.join(loss_crit_root, "avg_opt_perf.png"))
        plt.close(fig_avg)

    ax_avg_v_its.set_yscale("log")
    ax_avg_v_its.set_xlabel("cost")
    ax_avg_v_its.set_ylabel("loss")
    ax_avg_v_its.legend()
    fig_avg_v_its.tight_layout()
    fig_avg_v_its.savefig(os.path.join(loss_crit_root, "avg_opt_perf_v_its.png"))
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

def plot_performance(crit_df, loss_crit_root):
    for optimizer, opt_df in crit_df.groupby("optimizer"):
        save_root = os.path.join(loss_crit_root, str(optimizer))
        os.makedirs(save_root, exist_ok=True)
        loss_traces = np.vstack(opt_df["loss_record"].to_numpy())
        final_loss = loss_traces[:, -1]
        save_histogram(final_loss, os.path.join(save_root, "loss_hist.svg"))
        loss_stand = loss_transformation(final_loss)
        plot_recon_vs_loss(opt_df["final_snap_cos_sim"], loss_stand, os.path.join(save_root, "final_snap_cos_sim_vs_loss.svg"), (0, 1))



def global_post_main(df: pd.DataFrame, root: str) -> None:
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
    root = os.path.join(root, "global_results")
    os.makedirs(root, exist_ok=True)

    # Group hierarchically instead of repeatedly masking
    for NT, df_NT in df.groupby("NT"):
        NT_root = os.path.join(root, f"NT={NT}")

        for n_part, df_part in df_NT.groupby("n_part"):
            for loss_crit, crit_df in df_part.groupby("loss_crit"):
                loss_crit_root = os.path.join(NT_root, f"np={n_part}", str(loss_crit))
                os.makedirs(loss_crit_root, exist_ok=True)
                plot_opt_comp(crit_df, loss_crit_root)
                plot_performance(crit_df, loss_crit_root)


