import jax
import jax.numpy as jnp
from SRC.Solver.KF_intergrators import KF_PS_RHS, RK4_Step
from SRC.utils import load_data
from SRC.DA_Comp.configs import KF_Opts
import random
import multiprocessing as mp
import os 
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_default_device", jax.devices("cpu")[0])
from create_results_dir import create_results_dir
import jax.numpy as jnp
from SRC.Solver.solver import KF_Stepper

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

# ------------- main function -----------------
import os
import random
import numpy as np
import jax
import jax.numpy as jnp

def ly_exp_main():
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=2e-2,
        total_T=1000,
        min_samp_T=50,
        t_skip=1e-1,
    )

    seed = 0
    r = 50
    T = 1e2
    T_skip = 1

    root = os.path.join(
        create_results_dir(),
        "Ly_Exps",
        f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_dt={kf_opts.dt}_T={T}"
    )
    os.makedirs(root, exist_ok=True)

    attractor_snapshots = load_data(kf_opts)

    # Python RNG for picking U_0
    rng = random.Random(seed)
    idx = rng.randint(0, attractor_snapshots.shape[0] - 1)
    U_0 = attractor_snapshots[idx, :]
    state_shape = U_0.shape
    U_0 = U_0.reshape(-1)
    # Build RHS & stepper
    stepper_raw = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt)
    stepper = lambda x: stepper_raw(x.reshape(*state_shape)).reshape(-1)
    n = U_0.shape[0]

    # JAX key
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, r), dtype=U_0.dtype)
    Y_0, _ = jnp.linalg.qr(A)

    n_steps = int(T / kf_opts.dt)
    n_skip  = int(T_skip / kf_opts.dt)

    growth_trj = push_orthonormal_matrix_variation(
        stepper, U_0, Y_0, n_steps, n_skip
    )

    # Lyapunov spectrum
    lyapunov_spectrum = (
        jnp.sum(jnp.log(jnp.abs(growth_trj)), axis=0) / T
    )

    # Move to host + sort
    lyap_np = np.asarray(lyapunov_spectrum)
    lyap_sorted = np.sort(lyap_np)[::-1]

    LLE = float(lyap_sorted[0])
    KY_dim = float(kaplan_yorke_dimension(lyap_sorted))

    # -------- Write to file --------
    out_file = os.path.join(root, "lyapunov_spectrum.txt")

    with open(out_file, "w") as f:
        f.write("# Lyapunov analysis\n")
        f.write(f"# Re        = {kf_opts.Re}\n")
        f.write(f"# NDOF      = {kf_opts.NDOF}\n")
        f.write(f"# dt        = {kf_opts.dt}\n")
        f.write(f"# T         = {T}\n")
        f.write(f"# r         = {r}\n")
        f.write("\n")

        f.write(f"LLE = {LLE:.8e}\n")
        f.write(f"KY_dim = {KY_dim:.8f}\n")
        f.write("\n")

        f.write("# Lyapunov exponents (sorted, descending)\n")
        for i, le in enumerate(lyap_sorted):
            f.write(f"{i:3d}  {le:.8e}\n")

    print(f"Saved Lyapunov results to: {out_file}")
    print(f"LLE = {LLE}")
    print(f"KY_dim = {KY_dim}")


if __name__ == "__main__":
    ly_exp_main()