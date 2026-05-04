from SRC.Solver.solver import KF_TP_Stepper, create_vel_part_gen_fn
from SRC.DA_Comp.configs import KF_Opts
from SRC.utils import load_data, bilinear_sample_periodic

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def get_int_pp_fn(stepper, T_f, omega0_hat):
    """
    Endpoint map for a fixed flow IC:
        pp_init = [xp0, yp0] -> [xp(T_f), yp(T_f)]
    """
    trj_gen_fn = create_vel_part_gen_fn(jax.jit(stepper), T_f)

    u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
    u = jnp.fft.irfft2(u_hat)
    v = jnp.fft.irfft2(v_hat)

    def int_pp_fn(pp_init):
        xp_init = pp_init[0][None, None]   # shape (1,1)
        yp_init = pp_init[1][None, None]   # shape (1,1)

        up_init = bilinear_sample_periodic(
            u, xp_init, yp_init, stepper.NS.L, stepper.NS.L
        )
        vp_init = bilinear_sample_periodic(
            v, xp_init, yp_init, stepper.NS.L, stepper.NS.L
        )

        up_traj, vp_traj, xp_traj, yp_traj = trj_gen_fn(
            omega0_hat, xp_init, yp_init, up_init, vp_init
        )

        return jnp.array([
            xp_traj[-1, 0],
            yp_traj[-1, 0],
        ])

    return int_pp_fn


def make_ftle_batch_fn(stepper, T_f, omega0_hat):
    """
    For one fixed omega0_hat, return a fast batched FTLE evaluator:
        pp_batch: shape (N, 2)
        returns: shape (N,)
    """
    int_pp_fn = get_int_pp_fn(stepper, T_f, omega0_hat)

    # Jacobian of endpoint map wrt pp_init: shape (2,2)
    jac_fn = jax.jit(jax.jacobian(int_pp_fn))

    # Batch jacobians over many pp_init: shape (N,2,2)
    jac_batch_fn = jax.jit(jax.vmap(jac_fn))

    @jax.jit
    def ftle_batch_fn(pp_batch):
        jac_batch = jac_batch_fn(pp_batch)              # (N, 2, 2)
        svals = jnp.linalg.svd(jac_batch, compute_uv=False)  # (N, 2)
        return jnp.log(svals[:, 0]) / T_f

    return ftle_batch_fn


def sample_pp_inits(key, num_pp, L):
    """
    Sample particle initial conditions uniformly in [0, L)^2.
    Returns shape (num_pp, 2).
    """
    k1, k2 = jax.random.split(key)
    xp = jax.random.uniform(k1, shape=(num_pp,), minval=0.0, maxval=L)
    yp = jax.random.uniform(k2, shape=(num_pp,), minval=0.0, maxval=L)
    return jnp.stack([xp, yp], axis=1)


def main():
    St = 0
    beta = 0
    T_f = .2

    num_flow_ics = 20
    num_pp_per_flow = 128

    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=int(1e3),
        min_samp_T=100,
        t_skip=10,
    )

    stepper = KF_TP_Stepper(
        kf_opts.Re, kf_opts.n, kf_opts.NDOF,
        kf_opts.dt, St, beta, 1
    )

    attractor_snapshots = load_data(kf_opts)
    num_snaps = attractor_snapshots.shape[0]

    key = jax.random.PRNGKey(0)

    # sample attractor snapshots
    key, k_flow = jax.random.split(key)
    flow_idx = jax.random.choice(
        k_flow,
        num_snaps,
        shape=(num_flow_ics,),
        replace=False
    )

    ftle_means_per_flow = []
    ftle_all = []

    for i in range(num_flow_ics):
        omega0_hat = jnp.asarray(attractor_snapshots[int(flow_idx[i]), :])

        key, k_pp = jax.random.split(key)
        pp_batch = sample_pp_inits(k_pp, num_pp_per_flow, stepper.NS.L)

        ftle_batch_fn = make_ftle_batch_fn(stepper, T_f, omega0_hat)
        ftles = ftle_batch_fn(pp_batch)   # shape (num_pp_per_flow,)

        ftle_means_per_flow.append(jnp.mean(ftles))
        ftle_all.append(ftles)

    ftle_means_per_flow = jnp.array(ftle_means_per_flow)
    ftle_all = jnp.concatenate(ftle_all)

    print("num flow ICs             =", num_flow_ics)
    print("num pp per flow          =", num_pp_per_flow)
    print("total FTLE samples       =", ftle_all.shape[0])
    print("mean FTLE (all samples)  =", jnp.mean(ftle_all))
    print("std FTLE  (all samples)  =", jnp.std(ftle_all))
    print("mean of per-flow means   =", jnp.mean(ftle_means_per_flow))
    print("std  of per-flow means   =", jnp.std(ftle_means_per_flow))


if __name__ == "__main__":
    main()