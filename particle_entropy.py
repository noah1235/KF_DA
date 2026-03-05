from SRC.Solver.solver import KF_Stepper, create_vel_part_gen_fn, KF_TP_Stepper
from SRC.DA_Comp.configs import KF_Opts
from SRC.utils import load_data
import jax.numpy as jnp
from SRC.Solver.IC_gen import init_particles_vector
import jax
from SRC.utils import bilinear_sample_periodic
import matplotlib.pyplot as plt
import numpy as np
jax.config.update("jax_enable_x64", True)


def periodic_displacement(x, x0, L=2*jnp.pi):
    # returns displacement in [-L/2, L/2)
    return jnp.mod((x - x0) + 0.5 * L, L) - 0.5 * L

jax.config.update("jax_enable_x64", True)


def particle_ftle():
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=int(1e6),
        min_samp_T=100,
        t_skip=1e-1,
    )
    T_f = 5
    St = 0
    beta = 0

    xp_init = jnp.array([jnp.pi])
    yp_init = jnp.array([jnp.pi])


    stepper = KF_TP_Stepper(
        kf_opts.Re, kf_opts.n, kf_opts.NDOF,
        kf_opts.dt, St, beta, 1
    )

    n_ICs = 25
    attractor_snapshots = load_data(kf_opts)
    trj_gen_fn = create_vel_part_gen_fn(jax.jit(stepper), T_f)

    def part_int(omega0_hat_flat):
        omega0_hat = omega0_hat_flat.reshape((attractor_snapshots.shape[1], attractor_snapshots.shape[2]))
        u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
        u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)

        up_init = bilinear_sample_periodic(
            u, xp_init, yp_init,
            stepper.NS.L, stepper.NS.L
        )
        vp_init = bilinear_sample_periodic(
            v, xp_init, yp_init,
            stepper.NS.L, stepper.NS.L
        )

        _, _, xp_traj, yp_traj = trj_gen_fn(
            omega0_hat, xp_init, yp_init,
            up_init, vp_init
        )

        return jnp.array([xp_traj[-1], yp_traj[-1]]).reshape(-1)
    
    jac_fn = jax.jacobian(part_int)

    def get_ftle(omega0_hat_flat):
        jac = jac_fn(omega0_hat_flat)
        _, S, _ = jnp.linalg.svd(jac)
        return jnp.log(S)/T_f
        return S


    batched_fn = jax.jit(
        jax.vmap(get_ftle, in_axes=0)
    )
    key = jax.random.PRNGKey(0)
    IC_idxs = jax.random.randint(
        key,
        shape=(n_ICs,),
        minval=0,
        maxval=attractor_snapshots.shape[0],
    )

    omega0_hats = attractor_snapshots[IC_idxs].reshape((IC_idxs.shape[0], -1))
    jac_batch = batched_fn(omega0_hats)
    ftle_1 = jac_batch[:, 0]
    ftle_2 = jac_batch[:, 1]
    print(ftle_1)
    print(ftle_2)

from scipy.spatial import cKDTree
from scipy.special import digamma

def knn_entropy_2d(x, y, k=3):
    """
    Kozachenko–Leonenko differential entropy estimator (2D).
    Returns entropy in nats.
    """
    pts = np.column_stack([x, y])
    N, dim = pts.shape

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k+1)  # first neighbor is itself
    eps = dists[:, -1]

    # volume of unit ball in 2D
    c_d = np.pi

    h = (
        digamma(N)
        - digamma(k)
        + np.log(c_d)
        + dim * np.mean(np.log(eps))
    )

    return h

# ------------------------------------------------------------
# entropy from 2D histogram
# ------------------------------------------------------------
def entropy_from_displacements(dx, dy, L, n_bins=64):
    edges = np.linspace(-L/2, L/2, n_bins + 1)

    counts, _, _ = np.histogram2d(dx, dy, bins=[edges, edges])

    p = counts / counts.sum()
    p = p[p > 0]

    return float(-np.sum(p * np.log(p)))  # entropy (nats)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=int(1e6),
        min_samp_T=100,
        t_skip=1e-1,
    )

    n_ICs = 300
    attractor_snapshots = load_data(kf_opts)

    St = 0
    beta = 0
    T_f = 1

    xp_init = jnp.array([jnp.pi])
    yp_init = jnp.array([jnp.pi])

    key = jax.random.PRNGKey(0)
    IC_idxs = jax.random.randint(
        key,
        shape=(n_ICs,),
        minval=0,
        maxval=attractor_snapshots.shape[0],
    )

    # ---- build dynamics once ----
    stepper = KF_TP_Stepper(
        kf_opts.Re, kf_opts.n, kf_opts.NDOF,
        kf_opts.dt, St, beta, 1
    )

    trj_gen_fn = create_vel_part_gen_fn(jax.jit(stepper), T_f)

    omega0_hats = attractor_snapshots[IC_idxs]

    # --------------------------------------------------------
    # displacement generator
    # --------------------------------------------------------
    def get_part_displace_from_omega(omega0_hat):

        u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
        u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)

        up_init = bilinear_sample_periodic(
            u, xp_init, yp_init,
            stepper.NS.L, stepper.NS.L
        )
        vp_init = bilinear_sample_periodic(
            v, xp_init, yp_init,
            stepper.NS.L, stepper.NS.L
        )

        _, _, xp_traj, yp_traj = trj_gen_fn(
            omega0_hat, xp_init, yp_init,
            up_init, vp_init
        )

        dx = periodic_displacement(xp_traj, xp_init, L=stepper.NS.L)
        dy = periodic_displacement(yp_traj, yp_init, L=stepper.NS.L)

        return dx.reshape(-1), dy.reshape(-1)

    batched_fn = jax.jit(
        jax.vmap(get_part_displace_from_omega, in_axes=0)
    )

    x_dis_trj, y_dis_trj = batched_fn(omega0_hats)

    # --------------------------------------------------------
    # ENTROPY vs TIME
    # --------------------------------------------------------
    L = float(stepper.NS.L)

    n_steps = x_dis_trj.shape[1]
    times = np.arange(n_steps) * kf_opts.dt

    # --------------------------------------------------------
    # PLOT 1: displacement scatter (added back)
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))

    for T_plot in [0.1, 0.5, 1.0]:
        idx = int(T_plot / kf_opts.dt)

        all_x_dis = np.asarray(x_dis_trj[:, idx]).reshape(-1)
        all_y_dis = np.asarray(y_dis_trj[:, idx]).reshape(-1)

        plt.scatter(
            all_x_dis,
            all_y_dis,
            s=10,
            label=f"T = {T_plot}"
        )

    plt.xlabel("x displacement")
    plt.ylabel("y displacement")
    plt.title("Particle displacements")
    plt.axis("equal")
    plt.ylim(-L/2, L/2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # PLOT 2: entropy vs time
    # --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    for k in [2, 4, 6, 8]:
        entropy_t = np.zeros(n_steps)

        for i in range(n_steps):
            dx = np.asarray(x_dis_trj[:, i]).reshape(-1)
            dy = np.asarray(y_dis_trj[:, i]).reshape(-1)
            #entropy_t[i] = entropy_from_displacements(dx, dy, L, n_bins)
            entropy_t[i] = knn_entropy_2d(dx, dy, k=k)

    
        plt.scatter(times, entropy_t, s=6, label=f"k = {k}")

    plt.xlabel("Time")
    plt.ylabel("Entropy (nats)")
    plt.title("Entropy of displacement distribution")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
     main()