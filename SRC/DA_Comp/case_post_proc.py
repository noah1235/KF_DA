import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from SRC.plotting_utils import save_svg
import matplotlib as mpl
from SRC.Solver.ploting import plot_vorticity

def cos_sim(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return jnp.dot(x, y)/(jnp.linalg.norm(x) * jnp.linalg.norm(y))

def rel_error(trg, pred, attractor_rad):
    return jnp.linalg.norm(pred - trg) / attractor_rad

def post_proc_case_main(target_trj, DA_trj, init_guess_trj, opt_data, save_dir, dt, t_mask, results_df, attractor_rad):
    """
    Post-process a DA case:
      - compute per-timestep errors for velocity and particles
      - plot both vs time and save the figure.
    """
    omega_trg_hat, xp_trg, yp_trg, up_trg, vp_trg = target_trj
    omega_DA_hat, xp_DA, yp_DA, up_DA, vp_DA = DA_trj
    omega_init_guess_hat = init_guess_trj[0]

    omega_trg = jnp.fft.irfft2(omega_trg_hat, axes=(-2, -1))
    omega_DA = jnp.fft.irfft2(omega_DA_hat, axes=(-2, -1))
    omega_init_guess = jnp.fft.irfft2(omega_init_guess_hat, axes=(-2, -1))

    trj_cos_sim = cos_sim(omega_trg, omega_DA)
    L2_error_v_t = jnp.linalg.norm(omega_DA_hat - omega_trg_hat, axis=(1, 2))
    rel_error_trj = L2_error_v_t/attractor_rad
    trj_rel_error = jnp.mean(rel_error_trj)


    final_snap_cos_sim = cos_sim(omega_trg[-1], omega_DA[-1])
    final_snap_rel_error = rel_error(omega_trg_hat[-1], omega_DA_hat[-1], attractor_rad)
    
    init_snap_cos_sim = cos_sim(omega_trg[0], omega_DA[0])
    int_snap_rel_error = rel_error(omega_trg_hat[0], omega_DA_hat[0], attractor_rad)

    results_df["trj_cos_sim"] = [float(trj_cos_sim)]
    results_df["trj_rel_error"] = [float(trj_rel_error)]
    results_df["rel_error_trj"] = [np.array(rel_error_trj)]

    results_df["final_snap_cos_sim"] = [float(final_snap_cos_sim)]
    results_df["final_snap_rel_error"] = [float(final_snap_rel_error)]

    results_df["init_snap_cos_sim"] = [float(init_snap_cos_sim)]
    results_df["int_snap_rel_error"] = [float(int_snap_rel_error)]

    # Time axis length should match the number of timesteps
    nsteps = omega_trg_hat.shape[0]
    time_axis = jnp.arange(nsteps) * dt

    vel_error = compute_norm_vs_time(omega_trg, omega_DA)
    plot_vel_error_vs_time(vel_error, time_axis, t_mask, save_dir)

    #particle tracks
    plot_particle_tracks(xp_trg, yp_trg, xp_DA, yp_DA, t_mask, os.path.join(save_dir, "particle_tracks.svg"))

    #Vorticity plot
    plot_vort_comp(
        omega_init_guess[0], omega_trg[0],
        os.path.join(save_dir, "guess_vs_target_t0.svg"),
        l1="Guess vorticity (t0)", l2="Target vorticity (t0)"
    )

    plot_vort_comp(
        omega_init_guess[-1], omega_trg[-1],
        os.path.join(save_dir, "guess_vs_target_tN.svg"),
        l1="Guess vorticity (tN)", l2="Target vorticity (tN)"
    )

    plot_vort_comp(
        omega_DA[-1], omega_trg[-1],
        os.path.join(save_dir, "DA_vs_target_tN.svg"),
        l1="DA vorticity (tN)", l2="Target vorticity (tN)"
    )

    plot_vort_comp(
        omega_DA[0], omega_trg[0],
        os.path.join(save_dir, "DA_vs_target_t0.svg"),
        l1="DA vorticity (t0)", l2="Target vorticity (t0)"
    )
    
    plot_convergence(opt_data, save_dir)

def _break_periodic_lines(x, y, Lx, Ly, jump_frac=0.5):
    """
    Insert NaNs where a periodic wrap likely occurred so matplotlib breaks the line.
    Assumes positions are in [0, L).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    dx = np.diff(x)
    dy = np.diff(y)

    # if jump is bigger than ~L/2, it's probably a wrap
    breaks = (np.abs(dx) > jump_frac * Lx) | (np.abs(dy) > jump_frac * Ly)

    xb = x.astype(float).copy()
    yb = y.astype(float).copy()

    # put NaN at the *point after* the jump to break the segment
    idx = np.where(breaks)[0] + 1
    xb[idx] = np.nan
    yb[idx] = np.nan
    return xb, yb

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_particle_tracks(
    xp_trg, yp_trg, xp_DA, yp_DA,
    t_mask, save_path,
    max_particles=5
):
    Lx, Ly = 2*jnp.pi, 2*jnp.pi
    xp_trg = np.asarray(xp_trg); yp_trg = np.asarray(yp_trg)
    xp_DA  = np.asarray(xp_DA);  yp_DA  = np.asarray(yp_DA)
    t_mask = np.asarray(t_mask)

    if xp_trg.shape != yp_trg.shape or xp_DA.shape != yp_DA.shape:
        raise ValueError("x/y shapes must match within trg and within DA.")
    if xp_trg.shape != xp_DA.shape:
        raise ValueError(f"trg shape {xp_trg.shape} must match DA shape {xp_DA.shape}.")

    T, n = xp_trg.shape

    # Allow t_mask to be (T,) or (T,n)
    if t_mask.shape == (T,):
        mask_mode = "global"
    elif t_mask.shape == (T, n):
        mask_mode = "per_particle"
    else:
        raise ValueError(f"t_mask must have shape (T,) or (T,n). Got {t_mask.shape}")

    m = n if max_particles is None else min(max_particles, n)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # marker styling for measurement markers
    ms = 14
    mew = 1.5

    for p in range(m):
        c = colors[p % len(colors)]

        # break lines for periodic wrap (NaNs inserted)
        x1, y1 = _break_periodic_lines(xp_trg[:, p], yp_trg[:, p], Lx, Ly)
        x2, y2 = _break_periodic_lines(xp_DA[:, p],  yp_DA[:, p],  Lx, Ly)

        # ---- 1) LINES first ----
        ax.plot(x1, y1, lw=1.5, color=c, label="Target" if p == 0 else None, zorder=1)
        ax.plot(x2, y2, lw=1.5, ls="--", color=c, label="DA" if p == 0 else None, zorder=1)

        # Measurement indices
        if mask_mode == "global":
            idx = np.flatnonzero(t_mask == 1)
        else:
            idx = np.flatnonzero(t_mask[:, p] == 1)

        if idx.size > 0:
            # ---- 2) OPEN markers (DA) ----
            ax.scatter(
                xp_DA[idx, p] % Lx, yp_DA[idx, p] % Ly,
                s=1.5 * ms,          # bigger
                marker="o",
                facecolors="white",
                edgecolors=c,
                linewidths=mew,
                zorder=3
            )

            # ---- 3) FILLED markers (Target) ----
            ax.scatter(
                xp_trg[idx, p] % Lx, yp_trg[idx, p] % Ly,
                s=ms,
                marker="o",
                color=c,
                edgecolors="none",
                label="Meas. times" if p == 0 else None,
                zorder=4
            )

        # ---- START marker LAST (big black hollow circle) ----
        ax.scatter(
            xp_DA[0, p] % Lx, yp_DA[0, p] % Ly,
            s=140,
            marker="o",
            facecolors="none",
            edgecolors="k",
            linewidths=2.5,
            alpha=1.0,
            zorder=10
        )

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")

    save_svg(mpl, fig, save_path)
    plt.close(fig)

def radial_spectral_error(
    omega_pred: jnp.ndarray,
    omega_true: jnp.ndarray,
    Lx: float = 2.0 * jnp.pi,
    Ly: float = 2.0 * jnp.pi,
    nbins: int | None = None,
    bin_edges: jnp.ndarray | None = None,
    k_max: float | None = None,          # NEW
    eps: float = 1e-16,
    log_bins: bool = True,
    fft_input=False
):

    if fft_input:
        pred_hat = omega_pred
        true_hat = omega_true
        Ny  = true_hat.shape[0]
        Nkx = true_hat.shape[1]
        Nx  = 2 * (Nkx - 1)

    else:
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

    # ---------------- NEW: optional spectral cutoff ----------------
    if k_max is not None:
        k_max = float(k_max)
        Kmask = Kf <= k_max
        Kf = Kf[Kmask]
        Ef = Ef[Kmask]
        Tf = Tf[Kmask]
        Pf = Pf[Kmask]

    # ---------------- choose / build bin edges ----------------
    if bin_edges is None:
        if nbins is None:
            nbins = int(jnp.sqrt((Nx // 2) ** 2 + (Ny // 2) ** 2))
            nbins = max(nbins, 8)

        # Use cutoff (if provided) as the bin upper limit; otherwise max(K)
        k_hi = float(k_max) if (k_max is not None) else float(jnp.max(Kf))

        if (not log_bins) or (k_hi <= 0.0):
            bin_edges = jnp.linspace(0.0, k_hi, nbins + 1)
        else:
            k_pos = Kf[Kf > 0.0]
            if k_pos.size == 0:
                bin_edges = jnp.linspace(0.0, k_hi, nbins + 1)
            else:
                k_min = float(jnp.min(k_pos))
                if not (k_hi > k_min):
                    bin_edges = jnp.linspace(0.0, k_hi, nbins + 1)
                else:
                    # first bin [0, k_min], then log-spaced up to k_hi
                    log_edges = jnp.logspace(
                        jnp.log10(k_min),
                        jnp.log10(k_hi),
                        nbins
                    )
                    bin_edges = jnp.concatenate(
                        [jnp.array([0.0], dtype=log_edges.dtype), log_edges]
                    )

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

    rel_err_k = err_sums / jnp.maximum(true_sums, eps)

    return k_centers, rel_err_k, true_sums, pred_sums

def plot_vort_comp(
    omega_T_DA, omega_T_target,
    save_path,
    l1, l2,
    err_label="Error (DA − target)",
    spec_label=r"Spectral diagnostics vs $k$",
    energy_eps=1e-30,
    log_k: bool = False,
    max_k: float | None = 15.0,
):
    omega_err      = omega_T_DA - omega_T_target

    k_centers, rel_err_k, E_true_k, E_pred_k = radial_spectral_error(
        omega_T_DA,
        omega_T_target,
        log_bins=log_k,
        k_max=max_k,
        nbins=8
    )

    E_true_max = jnp.max(E_true_k)
    denom = jnp.maximum(E_true_max, energy_eps)
    E_true_norm = E_true_k / denom
    E_pred_norm = E_pred_k / denom

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    ax_DA, ax_target = axes[0, 0], axes[0, 1]
    ax_err, ax_spec  = axes[1, 0], axes[1, 1]

    plot_vorticity(omega_T_DA, ax=ax_DA)
    plot_vorticity(omega_T_target, ax=ax_target)
    ax_DA.set_title(l1)
    ax_target.set_title(l2)

    plot_vorticity(omega_err, ax=ax_err)
    ax_err.set_title(err_label)

    ax_spec.plot(k_centers, rel_err_k, marker="o", linewidth=1, label="Relative error")
    ax_spec.plot(k_centers, E_true_norm, marker="s", linewidth=1, label=r"True energy / max(true)")
    ax_spec.plot(k_centers, E_pred_norm, marker="^", linewidth=1, label=r"Guess energy / max(true)")
    ax_spec.set_ylim(0, 1)

    ax_spec.set_xlabel(r"$k$")
    ax_spec.set_ylabel("Value")
    ax_spec.set_title(spec_label)

    if log_k:
        ax_spec.set_xscale("log")

    ax_spec.grid(True, which="both", alpha=0.3)
    ax_spec.legend(loc="best")

    save_svg(mpl, fig, save_path)
    plt.close(fig)

def compute_norm_vs_time(target, pred):
    num = jnp.linalg.norm(target - pred, axis=(1, 2))
    den = jnp.linalg.norm(target, axis=(1, 2))
    return num / den

def plot_vel_error_vs_time(vel_error, time_axis, t_mask, save_dir):
    """
    Plot velocity and particle errors vs. time and save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(time_axis, vel_error, label="Velocity error")
    
    for t in time_axis[t_mask == 1]:
        ax.axvline(x=t, color="k")

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized L2 norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
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
    descent_plot = _pos_or_nan(-alpha_gTp)

    # Common y-limits (log): small positive floor to avoid 0
    candidates = [np.nanmax(loss_plot), np.nanmax(grad_plot), np.nanmax(descent_plot)]
    finite_cands = [c for c in candidates if np.isfinite(c)]
    ymax = max(finite_cands) if finite_cands else 1.0
    ymax = max(ymax, 1.0) * 1.05

    pos_mins = []
    for arr in (loss_plot, grad_plot, descent_plot):
        if np.isfinite(arr).any():
            pos_mins.append(np.nanmin(arr))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    # left y-axis curves
    ax.plot(
        iters, loss_plot,
        marker='o', ms=6, lw=2.5,
        color='tab:blue', linestyle='-',
        label="Loss"
    )
    #ax.plot(iters, grad_plot,    marker='s', ms=3, lw=1, label=r"$\|\nabla f\|$")
    #ax.plot(iters, descent_plot, marker='^', ms=3, lw=1, label=r"$-\,\alpha\,g^\top p$")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss / Grad / Descent")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.5)

    # right y-axis
    ax2 = ax.twinx()
    ax2.plot(
        iters, opt_data.IC_error_record,
        marker='x', ms=7, lw=2.5,
        color='tab:red', linestyle='--',
        label="IC Error"
    )
    ax2.set_ylabel("IC Error")

    # combined legend (both axes)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # --- keep your saving logic exactly ---
    save_svg(mpl, fig, os.path.join(save_dir, "convergence.svg"))
    plt.close()

    return fig, ax
