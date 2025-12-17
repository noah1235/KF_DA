import jax
import jax.numpy as jnp
from SRC.Solver.KF_intergrators import KF_PS_RHS, RK4_Step
from SRC.utils import load_data
from SRC.DA_Comp.configs import KF_Opts
import random
import multiprocessing as mp
#jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])

import jax.numpy as jnp

def kaplan_yorke_dimension(lyap: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Kaplan–Yorke (Lyapunov) dimension from a 1D array of Lyapunov exponents.

    - Accepts exponents in any order (will sort descending).
    - Returns a scalar jnp.ndarray (float).
    """
    lyap = jnp.asarray(lyap).reshape(-1)
    # sort λ1 >= λ2 >= ... >= λn
    lam = jnp.sort(lyap)[::-1]

    csum = jnp.cumsum(lam)
    pos = csum > 0.0
    k = jnp.sum(pos).astype(jnp.int32)  # k in {0,...,n}

    n = lam.shape[0]

    # Cases:
    # 1) k == 0 -> D_KY = 0
    # 2) 0 < k < n -> D_KY = k + S_k / |λ_{k+1}|
    # 3) k == n -> D_KY = n (all partial sums positive)
    def case_k0(_):
        return jnp.array(0.0, dtype=lam.dtype)

    def case_kn(_):
        return jnp.array(n, dtype=lam.dtype)

    def case_mid(_):
        Sk = csum[k - 1]          # sum_{i=1}^k λ_i  (positive)
        lam_next = lam[k]         # λ_{k+1}
        return k + Sk / jnp.abs(lam_next)

    return jax.lax.cond(
        k == 0,
        case_k0,
        lambda _: jax.lax.cond(k == n, case_kn, case_mid, operand=None),
        operand=None,
    )


def push_orthonormal_matrix_variation(stepper, u_0, Y_0, n, k: int):
    """
    Propagate orthonormal columns Y under the linearized dynamics of `stepper`,
    performing a QR re-orthonormalization every `k` steps.

    Args
    ----
    stepper : callable
        Nonlinear step: u_next = stepper(u). Must be JAX-traceable.
    u_0 : array
        Initial state.
    Y_0 : array, shape (dim, r)
        Initial basis (columns). Will be orthonormalized at start.
    n : int
        Number of steps to run.
    k : int
        Period for QR re-orthonormalization (k >= 1).

    Returns
    -------
    growth_trj : array, shape (n, r)
        Per-step growth factors. On QR steps, equals diag(R).
        On non-QR steps, filled with ones.
    """
    Q0 = Y_0

    @jax.jit
    def scan_fn(carry, t):
        u, Y = carry

        # Linearize the stepper around u, then push each column of Y
        u_next, jvp_fn = jax.linearize(stepper, u)
        Y_prop = jax.vmap(jvp_fn, in_axes=-1, out_axes=-1)(Y)  # same shape as Y

        # Every k steps, do QR; otherwise pass through.
        do_qr = (t + 1) % k == 0  # QR on steps k, 2k, 3k, ...
        #jax.debug.print("do_qr = {}", do_qr)
        def qr_branch(Yp):
            Q, R = jnp.linalg.qr(Yp, mode="reduced")
            growth = jnp.diag(R)
            return Q, growth

        def passthrough_branch(Yp):
            r = Yp.shape[-1]
            return Yp, jnp.ones((r,), dtype=Yp.dtype)

        Y_next, growth = jax.lax.cond(do_qr, qr_branch, passthrough_branch, Y_prop)

        return (u_next, Y_next), growth

    # Use explicit step indices so cond is on a static-like counter
    steps = jnp.arange(n, dtype=jnp.int32)

    (_, _Y_final), growth_trj = jax.lax.scan(
        scan_fn,
        (u_0, Q0),
        steps,
        length=n,
    )
    return growth_trj

def push_orthonormal_matrix(stepper, u_0, Y_0, n):
    @jax.jit
    def scan_fn(carry, _):
        u, Y = carry

        # More efficient approach
        u_next, jvp_fn = jax.linearize(stepper, u)
        Y_next = jax.vmap(jvp_fn, in_axes=-1, out_axes=-1)(Y)

        Q, R = jnp.linalg.qr(Y_next)
        Y_next = Q
        growth = jnp.diag(R)

        carry_next = (u_next, Y_next)

        return carry_next, growth
    
    Q, _ = jnp.linalg.qr(Y_0)

    initial_carry = (u_0, Q)

    _, growth_trj = jax.lax.scan(
        scan_fn,
        initial_carry,
        None,
        length=n,
    )

    return growth_trj


def _single_lyapunov_run(args):
    """
    One Monte Carlo run for the Lyapunov spectrum.
    This will be run in a separate process.
    """
    (seed, attractor_snapshots, kf_opts, r, T, T_skip) = args

    # Python RNG for picking U_0
    rng = random.Random(seed)

    # pick a random snapshot
    idx = rng.randint(0, attractor_snapshots.shape[0] - 1)
    U_0 = attractor_snapshots[idx, :]

    # build RHS & stepper locally in the worker
    rhs = KF_PS_RHS(kf_opts.NDOF, kf_opts.Re, kf_opts.n)
    stepper = RK4_Step(rhs, kf_opts.dt)

    n = U_0.shape[0]

    # JAX key per worker
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, r))
    Y_0, _ = jnp.linalg.qr(A)

    n_steps = int(T / kf_opts.dt)
    n_skip  = int(T_skip / kf_opts.dt)

    growth_trj = push_orthonormal_matrix_variation(
        stepper, U_0, Y_0, n_steps, n_skip
    )
    # return as a regular ndarray so multiprocessing pickles cleanly
    return jnp.array(growth_trj)


# ------------- main function -----------------

def ly_exp_main():
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=16,
        dt=1e-2,
        total_T=2000,
        min_samp_T=500,
        t_skip=1e-1,
    )

    r = 600
    T = 10
    T_skip = .1
    repeats = 4

    attractor_snapshots = load_data(kf_opts)

    # build argument list for workers
    # different seed per repeat
    worker_args = [
        (seed, attractor_snapshots, kf_opts, r, T, T_skip)
        for seed in range(repeats)
    ]

    # run in parallel
    # you can choose processes=min(repeats, mp.cpu_count()) if you like
    with mp.Pool(processes=repeats) as pool:
        results = pool.map(_single_lyapunov_run, worker_args)

    # stack results
    growth_trj_all = jnp.vstack([jnp.array(g) for g in results])

    lyapunov_spectrum = (
        jnp.sum(jnp.log(jnp.abs(growth_trj_all)), axis=0) / (T * repeats)
    )
    print(lyapunov_spectrum)
    KY_dim = kaplan_yorke_dimension(lyapunov_spectrum)
    print(KY_dim)

if __name__ == "__main__":
    ly_exp_main()