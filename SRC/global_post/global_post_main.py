import numpy as np
import matplotlib.pyplot as plt
import os

def global_post_main(df, opt, crit, root):
    df = df[df["optimizer"] == f"{opt}"]
    df = df[df["loss_crit"] == f"{crit}"]
    samp_period_list = df["samp_period"].unique()
    root = os.path.join(root, "global_results")
    for samp_period in samp_period_list:
        df_samp = df[df["samp_period"] == samp_period]
        samp_root = os.path.join(root, f"SP={samp_period}")
        for n_part in df_samp["n_part"].unique():
            save_root = os.path.join(samp_root, f"np={n_part}", f"{crit}", f"{opt}")
            os.makedirs(save_root, exist_ok=True)
            df_part = df_samp[df_samp["n_part"] == n_part]
            loss_traces = np.vstack(df_part["loss_record"].to_numpy())

            loss_evals_record = np.vstack(df_part["loss_evals_record"].to_numpy())
            loss_grad_evals_record = np.vstack(df_part["loss_grad_evals_record"].to_numpy())
            Hvp_evals_record = np.vstack(df_part["Hvp_evals_record"].to_numpy())

            loss_eval_time = df_part["loss_avg_eval_time"].to_numpy()
            loss_grad_cost = df_part["loss_grad_avg_eval_time"].to_numpy() / loss_eval_time
            Hvp_cost = df_part["Hvp_avg_eval_time"].to_numpy() / loss_eval_time

            print(loss_grad_cost, Hvp_cost)

            total_cost = loss_evals_record + loss_grad_cost*loss_grad_evals_record + Hvp_cost*Hvp_evals_record
            plt.figure(figsize=(10, 6))
            plt.plot(total_cost.T, loss_traces.T)
            plt.title(f"loss grad cost: {loss_grad_cost} | Hvp cost: {Hvp_cost}")
            plt.savefig(os.path.join(save_root, "loss_v_cost.png"))
            plt.close()

            plt.plot(total_cost.T)
            plt.ylabel("cost")
            plt.xlabel("its")
            plt.savefig(os.path.join(save_root, "cost_v_its.png"))
            plt.close()


