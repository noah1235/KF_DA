import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from SRC.Solver.ploting import plot_vorticity

def post_proc_case_main(target_trj, DA_trj, opt_data, n_particles, save_dir, dt, omega_fn, t_mask, results_df):
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

    trj_cos_sim = jnp.vdot(target_vel.reshape(-1), DA_vel.reshape(-1)) / (jnp.linalg.norm(target_vel) * jnp.linalg.norm(DA_vel))
    results_df["trj_cos_sim"] = [float(trj_cos_sim)]

    # Time axis length should match the number of timesteps
    nsteps = target_trj.shape[0]
    time_axis = jnp.arange(nsteps) * dt

    vel_cos_sim = compute_cosine_vs_time(target_vel, DA_vel)
    part_cos_sim = compute_cosine_vs_time(target_part, DA_part)
    plot_vel_part_error_vs_time(vel_cos_sim, part_cos_sim, time_axis, t_mask, save_dir)

    #Vorticity plot
    plot_final_vort(DA_vel, target_vel, omega_fn, save_dir)
    plot_convergence(opt_data, save_dir)

def plot_final_vort(DA_vel, target_vel, omega_fn, save_dir):
    """
    Make a side-by-side vorticity plot at final time for DA vs. target.

    Parameters
    ----------
    DA_vel : array-like
        Velocity trajectory (time, ...). Last frame is used.
    target_vel : array-like
        Target velocity trajectory (time, ...). Last frame is used.
    omega_fn : callable
        Function mapping a velocity snapshot -> 2D vorticity array (Ny, Nx).
    save_dir : str
        Directory to save the figure.
    Lx, Ly : float or None
        Domain lengths for plotting extents. If None, use array shape (Nx, Ny).
    cmap : str
        Colormap name to forward to plot_vorticity (seaborn cmap).

    Returns
    -------
    out_path : str
        Path to the saved PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute final-time vorticity fields
    omega_T_DA     = omega_fn(DA_vel[-1])
    omega_T_target = omega_fn(target_vel[-1])



    # Create side-by-side axes
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    ax1, ax2 = axes

    # Draw both plots using your helper (assumed signature: (omega, Lx, Ly, cmap, ax))
    _, _, im1, _ = plot_vorticity(omega_T_DA, ax=ax1)
    _, _, im2, _ = plot_vorticity(omega_T_target, ax=ax2)

    # Match color scales
    vmin = float(jnp.minimum(jnp.min(omega_T_DA), jnp.min(omega_T_target)))
    vmax = float(jnp.maximum(jnp.max(omega_T_DA), jnp.max(omega_T_target)))
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)

    # Titles
    ax1.set_title("DA final vorticity")
    ax2.set_title("Target final vorticity")

    # Save
    out_path = os.path.join(save_dir, "final_vorticity_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def compute_cosine_vs_time(A, B, eps=1e-12):
    dot = jnp.sum(A * B, axis=1)
    na  = jnp.linalg.norm(A, axis=1)
    nb  = jnp.linalg.norm(B, axis=1)
    return dot / (na * nb + eps)

def plot_vel_part_error_vs_time(vel_cos_sim, part_cos_sim, time_axis, t_mask, save_dir):
    """
    Plot velocity and particle errors vs. time and save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(time_axis, vel_cos_sim, label="Velocity error", marker="o")
    ax.plot(time_axis, part_cos_sim, label="Particle error", marker="o")
    
    for t in time_axis[t_mask == 1]:
        ax.axvline(x=t, color="k", linestyle="--", alpha=0.3, linewidth=1.0)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cosine Similarity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()

    out_path = os.path.join(save_dir, "vel_part_error_vs_time.png")
    fig.savefig(out_path, dpi=150)
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
    ax.set_ylim(y_min, ymax)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")

    # --- keep your saving logic exactly ---
    fig.savefig(os.path.join(save_dir, "convergence.png"), dpi=200)

    return fig, ax
