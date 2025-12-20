import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

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
    for samp_period, df_samp in df.groupby("samp_period"):
        samp_root = os.path.join(root, f"SP={samp_period}")

        for n_part, df_part in df_samp.groupby("n_part"):
            for loss_crit, crit_df in df_part.groupby("loss_crit"):
                loss_crit_root = os.path.join(samp_root, f"np={n_part}", str(loss_crit))
                os.makedirs(loss_crit_root, exist_ok=True)

                # Figure for average optimizer performance (across all optimizers)
                fig_avg, ax_avg = plt.subplots()
                
                for optimizer, opt_df in crit_df.groupby("optimizer"):
                    save_root = os.path.join(loss_crit_root, str(optimizer))
                    os.makedirs(save_root, exist_ok=True)

                    # Stack traces: shape (num_runs, num_iters)
                    loss_traces = np.vstack(opt_df["loss_record"].to_numpy())
                    loss_evals_record = np.vstack(opt_df["loss_evals_record"].to_numpy())
                    loss_grad_evals_record = np.vstack(opt_df["loss_grad_evals_record"].to_numpy())
                    Hvp_evals_record = np.vstack(opt_df["Hvp_evals_record"].to_numpy())

                    # Per-run timings
                    #loss_eval_time = opt_df["loss_avg_eval_time"].to_numpy()
                    #loss_eval_time = np.where(
                    #    loss_eval_time == 0,
                    #    opt_df["loss_grad_avg_eval_time"].to_numpy() / 2,
                    #    loss_eval_time
                    #)

                    #loss_grad_cost = opt_df["loss_grad_avg_eval_time"].to_numpy() / loss_eval_time
                    #Hvp_cost = opt_df["Hvp_avg_eval_time"].to_numpy() / loss_eval_time
                    #loss_grad_cost = 2
                    #Hvp_cost = 5
                    
                    #avg_loss_grad_cost = np.mean(loss_grad_cost)
                    #avg_Hvp_cost = np.mean(Hvp_cost)
                    avg_loss_grad_cost = 2
                    avg_Hvp_cost = 5
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
                    fig.savefig(os.path.join(save_root, "loss_v_cost.png"))
                    plt.close(fig)

                    # Cost vs iteration for each run
                    fig, ax = plt.subplots()
                    ax.plot(total_cost.T)
                    ax.set_ylabel("cost")
                    ax.set_xlabel("iteration")
                    fig.tight_layout()
                    fig.savefig(os.path.join(save_root, "cost_v_its.png"))
                    plt.close(fig)


                    ax_avg.plot(avg_cost_trace, avg_loss_trace, label=str(optimizer))
                    

                # Finalize and save average performance plot
                ax_avg.set_yscale("log")
                ax_avg.set_xlabel("cost")
                ax_avg.set_ylabel("loss")
                ax_avg.legend()
                fig_avg.tight_layout()
                fig_avg.savefig(os.path.join(loss_crit_root, "avg_opt_perf.png"))
                plt.close(fig_avg)


