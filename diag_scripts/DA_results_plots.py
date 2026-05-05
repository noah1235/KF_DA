from __future__ import annotations

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from kf_da.utils.plotting_utils import save_svg 
from kf_da.utils.create_results_dir import create_results_dir


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def remove_outliers_by_loss(df, metric):
    loss_traces = np.vstack(df["loss_record"].to_numpy())
    final_loss = loss_traces[:, -1]

    median = np.median(final_loss)
    mad = np.median(np.abs(final_loss - median))

    if mad == 0:
        # All losses identical or single sample — keep everything
        return df[metric].to_numpy(), final_loss

    modified_z = 0.6745 * (final_loss - median) / mad
    mask = np.abs(modified_z) < 3.5
    return df.loc[mask, metric].to_numpy(), final_loss[mask]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def m_dep_fig(cfg: dict):
    Re       = cfg["Re"]
    n        = cfg["n"]
    dt       = cfg["dt"]
    NDOF     = cfg["NDOF"]
    St       = cfg.get("St", 0)
    beta     = cfg.get("beta", 0)
    m_dt     = cfg.get("m_dt", None)
    m_targets = cfg["m_targets"]
    metric   = cfg.get("metric", "final_snap_rel_error")

    root = os.path.join(
        create_results_dir(),
        "DA-no_noise",
        f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
    )
    save_root = os.path.join(root, "global_results", "mx_v_mt")
    os.makedirs(save_root, exist_ok=True)
    df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()

    for m_target in m_targets:
        found = False
        fig = plt.figure()

        for (n_part, NT), g in df.groupby(["n_part", "NT"], sort=True):
            m = NT * 2 * n_part
            if m != m_target:
                continue
            found = True

            perf, final_loss = remove_outliers_by_loss(g, metric)
            label = f"NT={NT}, n_part={n_part}, mean={np.mean(perf):.3f}"
            plt.scatter(final_loss, perf, label=label)

        if not found:
            print(f"No runs found with m_target={m_target}")

        plt.legend()
        plt.title(f"m = {m_target} | metric = {metric}")
        plt.xlabel("loss")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.xscale("log")
        plt.ylim(0, 1)
        save_svg(mpl, fig, os.path.join(save_root, f"m={m_target}.svg"))
        plt.close(fig)


def recon_v_m_dt(cfg: dict):
    Re        = cfg["Re"]
    n         = cfg["n"]
    dt        = cfg["dt"]
    NDOF      = cfg["NDOF"]
    St        = cfg.get("St", 0)
    beta      = cfg.get("beta", 0)
    NT        = cfg["NT"]
    n_part    = cfg["n_part"]
    metric    = cfg.get("metric", "final_snap_rel_error")
    loss_crit = cfg.get("loss_crit", "PP_MSE")

    pattern = re.compile(
        rf"^DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt=(\d*\.?\d+)-St={St}_beta={beta}_AI$"
    )

    base_dir = create_results_dir()
    m_dt_vals, mean_metric_vals, mean_final_loss_vals = [], [], []

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
        df = df[(df["n_part"] == n_part) & (df["NT"] == NT) & (df["loss_crit"] == loss_crit)]
        if df.empty:
            continue

        metric_arr, final_loss = remove_outliers_by_loss(df, metric)
        m_dt_vals.append(mdt)
        mean_metric_vals.append(np.mean(metric_arr))
        mean_final_loss_vals.append(np.mean(final_loss))

    if not m_dt_vals:
        print("No matching runs found.")
        return

    sort_idx = np.argsort(m_dt_vals)
    m_dt_vals          = np.array(m_dt_vals)[sort_idx]
    mean_metric_vals   = np.array(mean_metric_vals)[sort_idx]
    mean_final_loss_vals = np.array(mean_final_loss_vals)[sort_idx]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:blue"
    l1 = ax1.plot(m_dt_vals, mean_metric_vals, marker="o", color=color1, label=metric)
    ax1.set_xlabel(r"$\Delta t_m$", fontsize=12)
    ax1.set_ylabel(metric, color=color1, fontsize=12)
    ax1.set_xlim(0.1, 0.8)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    l2 = ax2.plot(m_dt_vals, mean_final_loss_vals, marker="s", color=color2, label="Mean Final Loss")
    ax2.set_ylabel("Mean Final Loss", color=color2, fontsize=12)

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", frameon=False)
    plt.title(rf"Loss Criterion: {loss_crit}", fontsize=13)
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(base_dir, "recon_v_m_dt.svg"))
    plt.close(fig)


def embedding_fig(cfg: dict):
    n           = cfg["n"]
    St          = cfg.get("St", 0)
    beta        = cfg.get("beta", 0)
    m_dt        = cfg.get("m_dt", None)
    noise_type  = cfg.get("noise_type", "DA-no_noise")
    config_list = cfg["config_list"]   # list of [Re, NDOF, dt, NT]
    dM_dict     = cfg["dM_dict"]       # {Re: dM} — inertial manifold dimension estimate per Re
    metric      = cfg.get("metric", "final_snap_rel_error")
    loss_crit   = cfg.get("loss_crit", "PP_MSE")

    Re_list, dM_list, m_list, metric_list = [], [], [], []

    for Re, NDOF, dt, NT in config_list:
        root = os.path.join(
            create_results_dir(),
            noise_type,
            f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
        )
        if os.path.isdir(root):
            df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()
            df = df[(df["NT"] == NT) & (df["loss_crit"] == loss_crit)]

            for n_part, df_npart in df.groupby("n_part", sort=True):
                metric_arr, _ = remove_outliers_by_loss(df_npart, metric)
                if len(metric_arr) == 0:
                    continue
                Re_list.append(Re)
                dM_list.append(dM_dict[Re])
                m_list.append(NT * n_part * 2)
                metric_list.append(np.mean(metric_arr))

    dM_arr     = np.array(dM_list)
    m_arr      = np.array(m_list)
    metric_arr = np.array(metric_list)

    if dM_arr.size == 0:
        print("embedding_fig: no matching data found — check config_list and NT values.")
        return

    dM_x = np.linspace(dM_arr.min(), dM_arr.max(), 100)

    fig = plt.figure(figsize=(6, 5))
    sc = plt.scatter(dM_arr, m_arr, c=metric_arr, s=80, vmin=0.0, vmax=0.5)
    plt.plot(dM_x, dM_x,       label="immersion line")
    plt.plot(dM_x, 2 * dM_x + 1, label="embedding line")
    plt.colorbar(sc).set_label("Mean Metric")
    plt.xlabel("IM dimension")
    plt.ylabel("m = NT * n_part * 2")
    plt.tight_layout()
    plt.legend()
    save_svg(mpl, fig, os.path.join(create_results_dir(), noise_type, "embedding_fig.svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config loader and registry
# ---------------------------------------------------------------------------

REGISTRY = {
    "m_dep_fig":    m_dep_fig,
    "recon_v_m_dt": recon_v_m_dt,
    "embedding_fig": embedding_fig,
}


def load_cfg(path: str) -> list[tuple[str, dict]]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    common = raw.get("common", {})
    run = raw.get("run", [])
    if isinstance(run, str):
        run = [run]

    tasks = []
    for fn_name in run:
        cfg = {**common, **raw.get(fn_name, {})}
        tasks.append((fn_name, cfg))
    return tasks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "kf-da-configs", "daPlotConfig.yaml")

if __name__ == "__main__":
    for fn_name, cfg in load_cfg(CONFIG_PATH):
        print(f"Running {fn_name}...")
        REGISTRY[fn_name](cfg)
