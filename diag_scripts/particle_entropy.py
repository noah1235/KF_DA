from SRC.Solver.solver import KF_TP_Stepper,..create_vel_part_gen_fn
from SRC.DA_Comp.configs import KF_Opts
from SRC.utils import load_data, bilinear_sample_periodic
from ..create_results_dir import..create_results_dir
from SRC.plotting_utils import save_svg
import matplotlib as mpl

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.special import digamma
import pickle

jax.config.update("jax_enable_x64", True)
from dataclasses import dataclass

@dataclass
class Data:
    X_disp: any
    Y_disp: any
    X_vel: any
    Y_vel: any


def periodic_displacement(x, x0, L=2 * jnp.pi):
    # returns displacement in [-L/2, L/2)
    return jnp.mod((x - x0) + 0.5 * L, L) - 0.5 * L

def safe_corr(a, b, eps=1e-14):
    """
    Correlation Corr(a,b) = E[(a-mean(a))(b-mean(b))] / sqrt(Var(a)Var(b))
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.mean(a**2) * np.mean(b**2))
    if denom < eps:
        return np.nan
    return float(np.mean(a * b) / denom)

def safe_normalize(x, eps=1e-14):
    x = np.asarray(x)
    xmax = np.max(np.abs(x))
    if xmax < eps:
        return x
    return x / xmax

# ------------------------------------------------------------
# KSG mutual information estimator
# ------------------------------------------------------------
def knn_mutual_information(X, Y, k=5, eps=1e-15):
    """
    KSG (Kraskov-Stogbauer-Grassberger) mutual information estimator.

    Parameters
    ----------
    X : ndarray, shape (N, dx)
        Samples of first variable.
    Y : ndarray, shape (N, dy)
        Samples of second variable.
    k : int
        Number of nearest neighbors.
    eps : float
        Tiny offset to avoid boundary-counting issues.

    Returns
    -------
    mi : float
        Estimated mutual information in nats.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    N = X.shape[0]
    if k >= N:
        raise ValueError(f"k={k} must be smaller than number of samples N={N}.")

    XY = np.concatenate([X, Y], axis=1)

    # max norm KSG
    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    dists, _ = tree_xy.query(XY, k=k + 1, p=np.inf)
    radii = dists[:, -1] - eps

    nx = np.empty(N, dtype=int)
    ny = np.empty(N, dtype=int)

    for i in range(N):
        nx[i] = len(tree_x.query_ball_point(X[i], r=radii[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(Y[i], r=radii[i], p=np.inf)) - 1

    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi)

def knn_normalized_mi(X, Y, k=5, eps=1e-15):
    """
    KNN-based normalized mutual information (NMI) using
    KSG estimator + relative entropy normalization.

    Paper: Accurate estimation of the normalized mutual information of multidimensional data

    Parameters
    ----------
    X : ndarray, shape (N, dx)
    Y : ndarray, shape (N, dy)
    k : int
    eps : float

    Returns
    -------
    nmi : float
        Normalized mutual information in [0, 1]
    mi : float
        Raw mutual information (nats)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    N = X.shape[0]
    dx = X.shape[1]
    dy = Y.shape[1]

    if k >= N:
        raise ValueError(f"k={k} must be smaller than N={N}.")

    XY = np.concatenate([X, Y], axis=1)

    # --- KD trees ---
    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    # --- joint distances (max norm, KSG style) ---
    dists, _ = tree_xy.query(XY, k=k + 1, p=np.inf)
    eps_i = dists[:, -1] - eps  # radius

    # --- neighbor counts ---
    nx = np.empty(N, dtype=int)
    ny = np.empty(N, dtype=int)

    for i in range(N):
        nx[i] = len(tree_x.query_ball_point(X[i], r=eps_i[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(Y[i], r=eps_i[i], p=np.inf)) - 1

    # --- MI (standard KSG) ---
    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))

    # ============================================================
    # --- Relative entropy normalization (key addition) ---
    # ============================================================

    d = dx + dy

    # scale-invariant radius (paper Eq. 31–32)
    eps_mean_power = np.mean(eps_i ** d)
    eps_tilde = eps_i / (eps_mean_power ** (1.0 / d) + 1e-30)

    log_eps_tilde = np.log(eps_tilde + 1e-30)

    # marginal entropies (relative entropy estimator)
    Hx = -np.mean(digamma(nx + 1)) + digamma(N) + dx * np.mean(log_eps_tilde)
    Hy = -np.mean(digamma(ny + 1)) + digamma(N) + dy * np.mean(log_eps_tilde)

    # --- normalized MI ---
    denom = np.sqrt(max(Hx, 1e-30) * max(Hy, 1e-30))
    nmi = mi / denom if denom > 0 else 0.0

    return float(nmi), float(mi)


# ------------------------------------------------------------
# single m_dt experiment
# ------------------------------------------------------------
def compute_ensemble(
    omega0_hats,
    xp_init_all,
    yp_init_all,
    stepper,
    kf_opts,
    m_dt,
):
    """
    For a given m_dt, compute:
      X_disp = (dx_1, dy_1)
      Y_disp = (dx_2, dy_2)

      X_vel  = (u_1, v_1)
      Y_vel  = (u_2, v_2)

    where:
      first pair is over [0, m_dt],
      second pair is over [m_dt, 2*m_dt].

    Also compute:
      corr_u  = Corr(u_1, u_2)
      corr_v  = Corr(v_1, v_2)
      corr_dx = Corr(dx_1, dx_2)
      corr_dy = Corr(dy_1, dy_2)
    """
    T_f = 2.0 * m_dt
    trj_gen_fn =..create_vel_part_gen_fn(jax.jit(stepper), T_f)

    idx_1 = int(round(m_dt / kf_opts.dt))

    def extract_pair(a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)

        a_1 = a[idx_1]
        b_1 = b[idx_1]

        a_2 = a[-1]
        b_2 = b[-1]
        return a_1, b_1, a_2, b_2

    def get_part_stats_from_omega(omega0_hat, xp_init, yp_init):
        u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)

        up_init = bilinear_sample_periodic(
            u, xp_init, yp_init, stepper.NS.L, stepper.NS.L
        )
        vp_init = bilinear_sample_periodic(
            v, xp_init, yp_init, stepper.NS.L, stepper.NS.L
        )

        up_traj, vp_traj, xp_traj, yp_traj = trj_gen_fn(
            omega0_hat, xp_init, yp_init, up_init, vp_init
        )

        x_1, y_1, x_2, y_2 = extract_pair(xp_traj, yp_traj)
        u_1, v_1, u_2, v_2 = extract_pair(up_traj, vp_traj)

        x0 = xp_init.reshape(())
        y0 = yp_init.reshape(())

        # first displacement over [0, m_dt]
        dx_1 = periodic_displacement(x_1, x0, L=stepper.NS.L)
        dy_1 = periodic_displacement(y_1, y0, L=stepper.NS.L)

        # second displacement over [m_dt, 2*m_dt]
        dx_2 = periodic_displacement(x_2, x_1, L=stepper.NS.L)
        dy_2 = periodic_displacement(y_2, y_1, L=stepper.NS.L)

        return dx_1, dy_1, dx_2, dy_2, u_1, v_1, u_2, v_2

    batched_fn = jax.jit(
        jax.vmap(get_part_stats_from_omega, in_axes=(0, 0, 0))
    )

    dx_1_trj, dy_1_trj, dx_2_trj, dy_2_trj, u_1_trj, v_1_trj, u_2_trj, v_2_trj = batched_fn(
        omega0_hats,
        xp_init_all,
        yp_init_all,
    )

    dx_1 = np.asarray(dx_1_trj).reshape(-1)
    dy_1 = np.asarray(dy_1_trj).reshape(-1)
    dx_2 = np.asarray(dx_2_trj).reshape(-1)
    dy_2 = np.asarray(dy_2_trj).reshape(-1)

    u1 = np.asarray(u_1_trj).reshape(-1)
    v1 = np.asarray(v_1_trj).reshape(-1)
    u2 = np.asarray(u_2_trj).reshape(-1)
    v2 = np.asarray(v_2_trj).reshape(-1)

    X_disp = np.column_stack([dx_1, dy_1])
    Y_disp = np.column_stack([dx_2, dy_2])

    X_vel = np.column_stack([u1, v1])
    Y_vel = np.column_stack([u2, v2])

    data = Data(X_disp=X_disp, Y_disp=Y_disp, X_vel=X_vel, Y_vel=Y_vel)
    return data

    mi_disp = knn_mutual_information(X_disp, Y_disp, k=k_mi)
    mi_vel = knn_mutual_information(X_vel, Y_vel, k=k_mi)

    corr_u = safe_corr(u1, u2)
    corr_v = safe_corr(v1, v2)
    corr_dx = safe_corr(dx_1, dx_2)
    corr_dy = safe_corr(dy_1, dy_2)

    return mi_disp, mi_vel, corr_u, corr_v, corr_dx, corr_dy


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    kf_opts = KF_Opts(
        Re=60,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=int(1e4),
        min_samp_T=100,
        t_skip=10,
    )

    root = os.path.join..create_results_dir(), "MI", f"Re={kf_opts.Re}")
    os.makedirs(root, exist_ok=True)

    n_ICs = 200
    attractor_snapshots = load_data(kf_opts)

    St = 0
    beta = 0

    key = jax.random.PRNGKey(0)

    # Random particle initial positions in [0, 2π]
    key, k1, k2, k3 = jax.random.split(key, 4)

    xp_init_all = jax.random.uniform(
        k1, shape=(n_ICs, 1), minval=0.0, maxval=2 * jnp.pi
    )
    yp_init_all = jax.random.uniform(
        k2, shape=(n_ICs, 1), minval=0.0, maxval=2 * jnp.pi
    )

    # Random initial vorticity states
    IC_idxs = jax.random.randint(
        k3,
        shape=(n_ICs,),
        minval=0,
        maxval=attractor_snapshots.shape[0],
    )
    omega0_hats = attractor_snapshots[IC_idxs]

    # Build dynamics once
    stepper = KF_TP_Stepper(
        kf_opts.Re, kf_opts.n, kf_opts.NDOF,
        kf_opts.dt, St, beta, 1
    )

    # Range of m_dt values
    m_dt_vals = np.array([
        0.10,
        0.15, 0.20, 0.30, 0.40, 0.50,
        0.75, 1.00, 1.25, 1.5
    ])
    m_dt_vals = np.array([
        .1, .3, .5, .7, .8
    ])

    m_dt_vals = np.array([
        .2, .3, .5, .7, .9, 1.3, 1.5
    ])

    results_root = os.path.join..create_results_dir(), "MI", f"Re={kf_opts.Re}")
    data_path = os.path.join(results_root, "data_dict.pkl")
    recompute = True
    if recompute:
        data_dict = {}

        for m_dt in m_dt_vals:
            data = compute_ensemble(
                omega0_hats=omega0_hats,
                xp_init_all=xp_init_all,
                yp_init_all=yp_init_all,
                stepper=stepper,
                kf_opts=kf_opts,
                m_dt=float(m_dt),
            )
            data_dict[m_dt] = data
            print(m_dt)

        with open(data_path, "wb") as f:
            pickle.dump(data_dict, f)

    else:
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)

    k_mi = 10

    m_dt_vals = data_dict.keys()
    mi_disp_vals = []
    mi_vel_vals = []
    corr_u_vals = []
    corr_v_vals = []
    corr_dx_vals = []
    corr_dy_vals = []

    for m_dt in m_dt_vals:
        data = data_dict[m_dt]
        #X_vel = np.column_stack([u1, v1])
        #Y_vel = np.column_stack([u2, v2])

        #mi_disp = knn_mutual_information(data.X_disp, data.Y_disp, k=k_mi)
        #mi_vel = knn_mutual_information(data.X_vel, data.Y_vel, k=k_mi)
        
        nmi_disp, _ = knn_normalized_mi(data.X_disp, data.Y_disp, k=k_mi)
        nmi_vel, _ = knn_normalized_mi(data.X_vel, data.Y_vel, k=k_mi)
        #print(mi_disp)

        u1 = data.X_vel[:, 0]
        v1 = data.X_vel[:, 1]
        u2 = data.Y_vel[:, 0]
        v2 = data.Y_vel[:, 1]

        dx_1 = data.X_disp[:, 0]
        dy_1 = data.X_disp[:, 1]

        dx_2 = data.Y_disp[:, 0]
        dy_2 = data.Y_disp[:, 1]

        corr_u = safe_corr(u1, u2)
        corr_v = safe_corr(v1, v2)
        corr_dx = safe_corr(dx_1, dx_2)
        corr_dy = safe_corr(dy_1, dy_2)

        mi_disp_vals.append(nmi_disp)
        mi_vel_vals.append(nmi_vel)

        corr_u_vals.append(corr_u)
        corr_v_vals.append(corr_v)
        corr_dx_vals.append(corr_dx)
        corr_dy_vals.append(corr_dy)

        print(m_dt)

    # --------------------------------------------------------
    # Plot 1: normalized MI and velocity correlations
    # --------------------------------------------------------
    if False:
        plt.figure(figsize=(6, 4))
        plt.plot(
            m_dt_vals,
            safe_normalize(mi_disp_vals),
            marker="o",
            label="displacement MI",
        )
        plt.plot(
            m_dt_vals,
            safe_normalize(mi_vel_vals),
            marker="s",
            label="velocity MI",
        )
        plt.plot(
            m_dt_vals,
            corr_u_vals,
            marker="^",
            label=r"$\mathrm{Corr}(u_1,u_2)$",
        )
        plt.plot(
            m_dt_vals,
            corr_v_vals,
            marker="v",
            label=r"$\mathrm{Corr}(v_1,v_2)$",
        )
        plt.xlabel(r"$m_{\Delta t}$")
        plt.ylabel("Normalized value / correlation")
        plt.title("Normalized MI and velocity correlations vs time separation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # --------------------------------------------------------
    # Plot 2: MI with two y-axes
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(
        m_dt_vals,
        mi_disp_vals,
        marker="o",
        color="tab:blue",
        label=r"$I((dx_1,dy_1);(dx_2,dy_2))$",
    )
    ax1.set_xlabel(r"$m_{\Delta t}$")
    ax1.set_ylabel("Normalized MI")
    ax1.plot(
            m_dt_vals,
            mi_vel_vals,
            marker="s",
            color="tab:red",
            label=r"$I((u_1,v_1);(u_2,v_2))$",
        )
    
    ax1.legend(loc="best")

    plt.title("Mutual information vs time separation")
    plt.tight_layout()
    plt.ylim(0, 1.2)
    save_svg(mpl, fig, os.path.join(root, "MI_vs_m_dt.svg"))
    plt.close()
    return

    # --------------------------------------------------------
    # Plot 3: velocity correlations
    # --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(m_dt_vals, corr_u_vals, marker="o", label=r"$\mathrm{Corr}(u_1,u_2)$")
    plt.plot(m_dt_vals, corr_v_vals, marker="s", label=r"$\mathrm{Corr}(v_1,v_2)$")
    plt.xlabel(r"$m_{\Delta t}$")
    plt.ylabel("Correlation")
    plt.title("Velocity correlations vs time separation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Plot 4: x-displacement correlation
    # --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(m_dt_vals, corr_dx_vals, marker="o")
    plt.xlabel(r"$m_{\Delta t}$")
    plt.ylabel(r"$\mathrm{Corr}(dx_1, dx_2)$")
    plt.title("X-displacement correlation vs time separation")
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------------
    # Plot 5: y-displacement correlation
    # --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(m_dt_vals, corr_dy_vals, marker="o")
    plt.xlabel(r"$m_{\Delta t}$")
    plt.ylabel(r"$\mathrm{Corr}(dy_1, dy_2)$")
    plt.title("Y-displacement correlation vs time separation")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()