# --- Project imports ---
from SRC.DA_Comp.configs import *  # provides KF_Opts, DA_Opts, etc.
from SRC.utils import load_data, real_concat_to_complex, complex_to_real_concat
from SRC.Solver.KF_intergrators import (
    KF_PS_RHS,
    KF_LPT_PS_RHS,
    create_trj_generator,
    Particle_Stepper,
    RK4_Step,
)
from SRC.DA_Comp.case_post_proc import post_proc_case_main
from SRC.Solver.IC_gen import init_particles_vector
from SRC.Solver.ploting import plot_particles
from SRC.DA_Comp.loss_funcs import create_loss_fn, project_divfree_rfft2_safe
from SRC.DA_Comp.optimization import Hessian_Opt, NCN, optax_opt, LBFGS, ADAM
from create_results_dir import create_results_dir

# --- Stdlib / third-party imports ---
import os
from pathlib import Path
import re
import random
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt




def get_max_seed_index(directory: str) -> int | None:
    """
    Find the maximum integer x among subfolders named exactly 'seed_idx=x' within `directory`.

    Parameters
    ----------
    directory : str
        Path to the root directory that may contain seed-indexed subfolders.

    Returns
    -------
    int | None
        The maximum seed index found, or None if there are no matching subfolders.
    """
    path = Path(directory)
    max_x = None
    for folder in path.iterdir():
        if folder.is_dir():
            m = re.match(r"seed_idx=(\d+)$", folder.name)
            if m:
                x = int(m.group(1))
                if max_x is None or x > max_x:
                    max_x = x
    return max_x


def calc_attractor_size(attractor_snapshots: jnp.ndarray) -> jnp.ndarray:
    """
    Estimate a characteristic size of the attractor as the mean distance
    of snapshots to their mean state (in the full-state Euclidean norm).

    Parameters
    ----------
    attractor_snapshots : jnp.ndarray
        Array of shape (Nsamples, Nstate) containing sampled states from the attractor.

    Returns
    -------
    jnp.ndarray
        Scalar mean distance (0-D array) representing an attractor "size" scale.
    """
    mean = jnp.mean(attractor_snapshots, axis=0)
    dist = jnp.linalg.norm(attractor_snapshots - mean.reshape((1, -1)), axis=1)
    return jnp.mean(dist)


def DA_exp_main(kf_opts: KF_Opts, DA_opts: DA_Opts) -> None:
    """
    Main entry point for running data assimilation (DA) experiments.

    The routine:
      1) Loads attractor snapshots and computes a characteristic scale.
      2) Builds a result directory tree organized by Re, n, dt, particle params, and seeds.
      3) For each seed:
         - Chooses/loads a target initial condition U_0.
         - For each horizon T and particle count:
             * Initializes particle ICs and saves a PNG.
             * Generates a target trajectory via LPT advection.
             * For each optimization initial distance:
                 - Picks a U_0_guess at the requested distance on the attractor.
                 - For each optimizer and loss criterion:
                     · Builds the loss function and runs the DA case.

    Parameters
    ----------
    kf_opts : KF_Opts
        Kolmogorov flow configuration.
    DA_opts : DA_Opts
        Data assimilation experiment configuration.
    """
    # Load attractor snapshots and compute attractor size scale
    attractor_snapshots = load_data(kf_opts)
    AS = calc_attractor_size(attractor_snapshots)

    # Draw initial-distance multipliers from a uniform distribution
    opt_init_dists = np.random.uniform(
        low=DA_opts.int_pert_range[0],
        high=DA_opts.int_pert_range[1],
        size=DA_opts.num_opt_inits,
    )

    # Create root results directory for this configuration
    root = os.path.join(
        create_results_dir(),
        (
            f"DA_Re={kf_opts.Re}_n={kf_opts.n}_dt={kf_opts.dt}"
            f"-St={DA_opts.part_opts.St}_beta={DA_opts.part_opts.beta}"
        ),
    )
    os.makedirs(root, exist_ok=True)

    # Continue seed indexing from the max existing seed, if present
    max_seed_idx = get_max_seed_index(root)
    if max_seed_idx is None:
        max_seed_idx = 0

    # Loop over experiment seeds
    for i in range(DA_opts.num_seeds):
        seed_idx = i + max_seed_idx
        seed_root = os.path.join(root, f"seed_idx={seed_idx}")
        os.makedirs(seed_root, exist_ok=True)

        # Load or create the target initial condition U_0 for this seed
        U_0_path = os.path.join(seed_root, "U_0.npy")
        if os.path.exists(U_0_path):
            U_0 = np.load(U_0_path)
        else:
            U_0 = attractor_snapshots[
                random.randint(0, attractor_snapshots.shape[0] - 1), :
            ]
            np.save(U_0_path, U_0)

        # Loop over time horizons
        for T in DA_opts.T_list:
            T_dir = os.path.join(seed_root, f"T={T}")

            # Loop over number of particles
            for npart in DA_opts.n_particles_list:
                npart_root = os.path.join(T_dir, f"np={npart}")

                # Build RHS and particle stepper for LPT
                RHS = KF_LPT_PS_RHS(
                    kf_opts.NDOF,
                    kf_opts.Re,
                    kf_opts.n,
                    npart,
                    beta=DA_opts.part_opts.beta,
                    St=DA_opts.part_opts.St,
                )
                stepper = Particle_Stepper(RK4_Step(RHS, kf_opts.dt), npart)

                # Loop over particle initializations
                for _ in range(DA_opts.num_particle_inits):
                    PI_root = os.path.join(npart_root, "PI")
                    os.makedirs(PI_root, exist_ok=True)

                    # Random particle ICs in the periodic domain
                    pIC = init_particles_vector(npart, (0, 2 * np.pi), (0, 2 * np.pi))
                    np.save(os.path.join(PI_root, "pIC.npy"), pIC)

                    # Quick visualization of particle ICs
                    fig, ax = plot_particles(pIC, RHS.KF_RHS.Lx, ax=None, s=20)
                    fig.savefig(os.path.join(PI_root, "particle_IC.png"))
                    plt.close()

                    # Generate target trajectory by advecting particles with the true U_0
                    trj_gen_fn = create_trj_generator(RHS, kf_opts.dt, T)
                    target_trj = trj_gen_fn(pIC, U_0)

                    # Split particle and velocity parts of the trajectory
                    target_part = target_trj[:, : RHS.n_particles * 4]
                    target_vel = target_trj[:, RHS.n_particles * 4 :]

                    # Loop over initial-guess distances (relative to AS)
                    for opt_init_dist in opt_init_dists:
                        opt_init_dir = os.path.join(PI_root, "cases", f"{opt_init_dist}")

                        # Skip if this case directory already exists
                        if os.path.isdir(opt_init_dir):
                            print("skipping")
                            continue

                        os.makedirs(opt_init_dir)

                        # Choose a guess U_0 at the requested distance from the true IC
                        state_dist = AS * opt_init_dist
                        True_IC_dist = jnp.linalg.norm(
                            attractor_snapshots - U_0.reshape((1, -1)), axis=1
                        )
                        U_0_guess = attractor_snapshots[
                            jnp.argmin(jnp.abs(True_IC_dist - state_dist)), :
                        ]
                        np.save(os.path.join(opt_init_dir, "U_0_guess.npy"), U_0_guess)

                        # For each optimizer and loss criterion, run a DA case
                        for optimizer in DA_opts.optimizer_list:
                            opt_method_dir = os.path.join(
                                opt_init_dir, f"{optimizer}"
                            )

                            for loss_crit in DA_opts.crit_list:
                                crit_dir = os.path.join(opt_method_dir, f"{loss_crit}")
                                os.makedirs(crit_dir, exist_ok=True)

                                # Build the loss function from criterion and stepper
                                loss_fn = create_loss_fn(
                                    loss_crit, stepper, target_part, pIC
                                )

                                def omega_fn(U):
                                    U = U.reshape((2, kf_opts.NDOF, kf_opts.NDOF//2+1))
                                    return RHS.KF_RHS.vorticity_real(U[0], U[1])
                                
                                def div_check(U_hat):
                                    U_hat = real_concat_to_complex(U_hat).reshape((2, RHS.KF_RHS.N, RHS.KF_RHS.N//2+1))
                                    div = RHS.KF_RHS.dxop * U_hat[0] + RHS.KF_RHS.dyop * U_hat[1]
                                    return jnp.mean(jnp.abs(div))
                                
                                def div_free_proj(U_real):
                                    U = real_concat_to_complex(U_real)
                                    U = U.reshape((2, RHS.KF_RHS.N, RHS.KF_RHS.N//2+1))
                                    U_proj = project_divfree_rfft2_safe(U, RHS.KF_RHS.KX, RHS.KF_RHS.KY, RHS.KF_RHS.K2)
                                    return (complex_to_real_concat(U_proj)).reshape(-1)

                                # Run the DA optimization for this setup
                                _run_DA_case(target_trj, U_0_guess, loss_fn, optimizer, trj_gen_fn, pIC, crit_dir, kf_opts.dt,
                                             omega_fn, div_check, div_free_proj)


def _run_DA_case(
    target_trj: np.ndarray | jnp.ndarray,
    U_0_guess: np.ndarray | jnp.ndarray,
    loss_fn,
    optimizer,
    trj_gen_fn,
    pIC,
    save_dir,
    dt,
    omega_fn,
    div_check,
    div_free_proj
) -> None:
    """
    Run a single DA case for a given optimizer and loss function.

    Parameters
    ----------
    target_vel : array-like
        Target velocity trajectory (unused here but kept for symmetry/extension).
    U_0_guess : array-like
        Complex initial condition guess for the flow field.
    loss_fn : callable
        Loss function mapping real-packed IC to scalar loss.
    optimizer : Hessian_Optimizer | LBFGS
        Optimizer configuration object (used to dispatch the optimization routine).
    """
    # Pack complex IC into real vector: [Re, Im]
    U_0_guess_real = complex_to_real_concat(U_0_guess)

    # Dispatch by optimizer type
    if isinstance(optimizer, NCN):
        grad_fn = jax.jit(jax.grad(loss_fn))
        Hess_fn = jax.jit(jax.hessian(loss_fn))
        Hessian_Opt(U_0_guess_real, loss_fn, grad_fn, Hess_fn, optimizer)
    elif isinstance(optimizer, LBFGS) or isinstance(optimizer, ADAM):
        U_0_real, opt_data = optax_opt(
            U_0_guess_real, loss_fn, jax.value_and_grad(loss_fn), optimizer, div_check, div_free_proj
        )

    
    U_0_DA = real_concat_to_complex(U_0_real)
    DA_trj = trj_gen_fn(pIC, U_0_DA)

    @jax.jit
    def H_eig_decomp(x):
        H = jax.hessian(loss_fn)(x)
        lam = jnp.linalg.eigvalsh(H)
        return lam
    
    lam = H_eig_decomp(U_0_real)
    
    print(lam)

    n_particles = pIC.shape[0]//4
    post_proc_case_main(target_trj, DA_trj, opt_data, n_particles, save_dir, dt, omega_fn)
    
