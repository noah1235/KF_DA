import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from SRC.plotting_utils import save_svg
import matplotlib as mpl
from SRC.Solver.ploting import plot_vorticity

def cos_sim(x, y):
    return jnp.dot(x, y)/(jnp.linalg.norm(x) * jnp.linalg.norm(y))

def rel_error(trg, pred):
    return jnp.linalg.norm(pred - trg) / jnp.linalg.norm(trg)

def post_proc_case_main(target_trj, DA_trj, init_guess_trj, opt_data, n_particles, save_dir, dt, omega_fn, t_mask, results_df):
    """
    Post-process a DA case:
      - compute per-timestep errors for velocity and particles
      - plot both vs time and save the figure.
    """
    # Split (particles | velocity)
    target_part = target_trj[:, : n_particles * 4]
    target_vel  = target_trj[:, n_particles * 4 :]

    DA_part = DA_trj[:, : n_particles * 4]
    DA_vel  = DA_trj[:, n_particles * 4 :]

    init_guess_vel = init_guess_trj[:, n_particles * 4 :]

    trj_cos_sim = cos_sim(target_vel.reshape(-1), DA_vel.reshape(-1))
    trj_rel_error = rel_error(target_vel.reshape(-1), DA_vel.reshape(-1))

    final_snap_cos_sim = cos_sim(target_vel[-1], DA_vel[-1])
    final_snap_rel_error = rel_error(target_vel[-1], DA_vel[-1])
    
    init_snap_cos_sim = cos_sim(target_vel[0], DA_vel[0])
    int_snap_rel_error = rel_error(target_vel[0], DA_vel[0])

    results_df["trj_cos_sim"] = [float(trj_cos_sim)]
    results_df["trj_rel_error"] = [float(trj_rel_error)]

    results_df["final_snap_cos_sim"] = [float(final_snap_cos_sim)]
    results_df["final_snap_rel_error"] = [float(final_snap_rel_error)]

    results_df["init_snap_cos_sim"] = [float(init_snap_cos_sim)]
    results_df["int_snap_rel_error"] = [float(int_snap_rel_error)]

    # Time axis length should match the number of timesteps
    nsteps = target_trj.shape[0]
    time_axis = jnp.arange(nsteps) * dt

    vel_error = compute_norm_vs_time(target_vel, DA_vel)
    plot_vel_error_vs_time(vel_error, time_axis, t_mask, save_dir)

    #Vorticity plot
    plot_vort_comp(
        init_guess_vel[0], target_vel[0], omega_fn,
        os.path.join(save_dir, "guess_vs_target_t0.svg"),
        l1="Guess vorticity (t0)", l2="Target vorticity (t0)"
    )

    plot_vort_comp(
        init_guess_vel[-1], target_vel[-1], omega_fn,
        os.path.join(save_dir, "guess_vs_target_tN.svg"),
        l1="Guess vorticity (tN)", l2="Target vorticity (tN)"
    )

    plot_vort_comp(
        DA_vel[-1], target_vel[-1], omega_fn,
        os.path.join(save_dir, "DA_vs_target_tN.svg"),
        l1="DA vorticity (tN)", l2="Target vorticity (tN)"
    )

    plot_vort_comp(
        DA_vel[0], target_vel[0], omega_fn,
        os.path.join(save_dir, "DA_vs_target_t0.svg"),
        l1="DA vorticity (t0)", l2="Target vorticity (t0)"
    )
    

    plot_convergence(opt_data, save_dir)


import jax.numpy as jnp

import jax.numpy as jnp

def radial_spectral_error(
    omega_pred: jnp.ndarray,
    omega_true: jnp.ndarray,
    Lx: float = 2.0 * jnp.pi,
    Ly: float = 2.0 * jnp.pi,
    nbins: int | None = None,
    bin_edges: jnp.ndarray | None = None,
    eps: float = 1e-30,
):
    if omega_pred.shape != omega_true.shape:
        raise ValueError(f"Shape mismatch: pred {omega_pred.shape}, true {omega_true.shape}")

    Ny, Nx = omega_true.shape

    pred_hat = jnp.fft.rfft2(omega_pred)
    true_hat = jnp.fft.rfft2(omega_true)
    err_hat  = pred_hat - true_hat

    err_pow  = jnp.abs(err_hat) ** 2
    true_pow = jnp.abs(true_hat) ** 2
    pred_pow = jnp.abs(pred_hat) ** 2

    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
    K = jnp.sqrt(KX**2 + KY**2)

    Kf = K.reshape(-1)
    Ef = err_pow.reshape(-1)
    Tf = true_pow.reshape(-1)
    Pf = pred_pow.reshape(-1)

    if bin_edges is None:
        if nbins is None:
            nbins = int(jnp.sqrt((Nx // 2) ** 2 + (Ny // 2) ** 2))
            nbins = max(nbins, 8)
        k_max = float(jnp.max(Kf))
        bin_edges = jnp.linspace(0.0, k_max, nbins + 1)

    nbins = bin_edges.size - 1
    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_ids = jnp.digitize(Kf, bin_edges) - 1
    valid = (bin_ids >= 0) & (bin_ids < nbins)

    bin_ids = bin_ids[valid]
    Ef = Ef[valid]
    Tf = Tf[valid]
    Pf = Pf[valid]

    err_sums  = jnp.zeros((nbins,), dtype=Ef.dtype).at[bin_ids].add(Ef)
    true_sums = jnp.zeros((nbins,), dtype=Tf.dtype).at[bin_ids].add(Tf)
    pred_sums = jnp.zeros((nbins,), dtype=Pf.dtype).at[bin_ids].add(Pf)

    # ---- GLOBAL normalization by max(true_sums) ----
    max_true = jnp.max(true_sums)
    denom = jnp.maximum(max_true, eps)

    err_norm_k  = err_sums / denom
    true_norm_k = true_sums / denom
    pred_norm_k = pred_sums / denom


    # Optional: if both err and true are basically zero, set err_norm to 0
    mask = (err_sums < eps) & (true_sums < eps)
    err_norm_k = jnp.where(mask, 0.0, err_norm_k)

    return k_centers, err_norm_k, true_norm_k, pred_norm_k



def plot_vort_comp(
    DA_vel, target_vel, omega_fn,
    save_path,
    l1, l2,
    err_label="Error (DA − target)",
    spec_label=r"Spectral diagnostics vs $k$",
):
    omega_T_DA     = omega_fn(DA_vel)       # "guess"
    omega_T_target = omega_fn(target_vel)   # true
    omega_err      = omega_T_DA - omega_T_target

    k_centers, err_k, E_true_norm, E_pred_norm = radial_spectral_error(omega_T_DA, omega_T_target)


    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    ax_DA, ax_target = axes[0, 0], axes[0, 1]
    ax_err, ax_spec  = axes[1, 0], axes[1, 1]

    # Top row: DA / Target
    plot_vorticity(omega_T_DA, ax=ax_DA)
    plot_vorticity(omega_T_target, ax=ax_target)
    ax_DA.set_title(l1)
    ax_target.set_title(l2)

    # Bottom-left: spatial error
    plot_vorticity(omega_err, ax=ax_err)
    ax_err.set_title(err_label)

    # Bottom-right: spectral diagnostics
    k_plot = k_centers

    ax_spec.plot(k_plot, err_k, marker="o", linewidth=1, label="Relative error")
    ax_spec.plot(k_plot, E_true_norm, marker="s", linewidth=1, label=r"True energy / max(true)")
    ax_spec.plot(k_plot, E_pred_norm, marker="^", linewidth=1, label=r"Guess energy / max(true)")

    ax_spec.set_xlabel(r"$k$")
    ax_spec.set_ylabel("Value")
    ax_spec.set_title(spec_label)
    ax_spec.grid(True, which="both", alpha=0.3)
    ax_spec.legend(loc="best")

    save_svg(mpl, fig, save_path)
    plt.close(fig)


def compute_norm_vs_time(target, pred):
    return jnp.linalg.norm(target - pred, axis=1) / jnp.linalg.norm(target)

def plot_vel_error_vs_time(vel_error, time_axis, t_mask, save_dir):
    """
    Plot velocity and particle errors vs. time and save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(time_axis, vel_error, label="Velocity error", marker="o")
    
    for t in time_axis[t_mask == 1]:
        ax.axvline(x=t, color="k", linestyle="--", alpha=0.3, linewidth=1.0)

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized L2 norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-.05, .2)
    fig.tight_layout()

    out_path = os.path.join(save_dir, "vel_part_error_vs_time.svg")
    save_svg(mpl, fig, out_path)
    plt.close(fig)
    return out_path

def plot_convergence(opt_data, save_dir, y_min=1e-18):
    """
    Plot loss, ||grad||, and -g^T p on the SAME axis with a shared y-scale (log).
    - Nonpositive/NaN values are masked (log axis cannot show 0 or negatives).
    - Keeps your saving logic exactly.
    Returns: (fig, ax)
    """
    # Pull records (supports either grad_norm_record or grad_norm attr)
    loss = np.asarray(opt_data.loss_record, dtype=float)
    grad_norm = np.asarray(
        getattr(opt_data, "grad_norm_record",
                getattr(opt_data, "grad_norm", opt_data.grad_norm_record)),
        dtype=float
    )
    alpha_gTp = np.asarray(opt_data.alpha_gTp_record, dtype=float)

    iters = np.arange(loss.size)

    # Mask to positives for log plotting
    def _pos_or_nan(y):
        y = y.astype(float, copy=True)
        y[~np.isfinite(y) | (y <= 0)] = y_min
        return y

    loss_plot    = _pos_or_nan(loss)
    grad_plot    = _pos_or_nan(grad_norm)
    descent_plot = _pos_or_nan(-alpha_gTp)   # -g^T p ≥ 0 for descent; mask nonpositive

    # Common y-limits (log): small positive floor to avoid 0
    candidates = [np.nanmax(loss_plot), np.nanmax(grad_plot), np.nanmax(descent_plot)]
    finite_cands = [c for c in candidates if np.isfinite(c)]
    ymax = max(finite_cands) if finite_cands else 1.0
    ymax = max(ymax, 1.0) * 1.05

    pos_mins = []
    for arr in (loss_plot, grad_plot, descent_plot):
        if np.isfinite(arr).any():
            pos_mins.append(np.nanmin(arr))

    # Single axes: overlay all curves
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(iters, loss_plot,    marker='o', ms=3, lw=1, label="Loss")
    ax.plot(iters, grad_plot,    marker='s', ms=3, lw=1, label=r"$\|\nabla f\|$")
    ax.plot(iters, descent_plot, marker='^', ms=3, lw=1, label=r"$-\,g^\top p$")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    #ax.set_ylim(y_min, ymax)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")

    # --- keep your saving logic exactly ---
    save_svg(mpl, fig, os.path.join(save_dir, "convergence.svg"))
    plt.close()

    return fig, ax
