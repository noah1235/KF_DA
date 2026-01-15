# --- Project imports ---
from SRC.DA_Comp.configs import *  # provides KF_Opts, DA_Opts, etc.
from SRC.utils import load_data, Vel_Part_Transformations, build_hvp
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
from SRC.DA_Comp.loss_funcs import create_loss_fn, build_div_free_proj
from SRC.DA_Comp.optimization.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from SRC.DA_Comp.optimization.optax_logic import *
from create_results_dir import create_results_dir
from SRC.Solver.ploting import plot_vorticity
from SRC.Vel_init.CS_init import CS_init

# --- Stdlib / third-party imports ---
import os
from pathlib import Path
import re
import random
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import gc




def append_to_parquet(df, parquet_path):
    """
    Append a DataFrame to a Parquet file, or create it if it doesn't exist.

    Parameters
    ----------
    df : pd.DataFrame
        The new data to save.
    parquet_path : str
        Path to the Parquet file.
    """
    # If parquet doesn't exist, just write a new one
    if not os.path.exists(parquet_path):
        df.to_parquet(parquet_path, index=False)
        print(f"Created new Parquet file: {parquet_path}")
        return

    # Otherwise, load existing, concatenate, and overwrite
    existing_df = pd.read_parquet(parquet_path)
    combined = pd.concat([existing_df, df], ignore_index=True)
    combined.to_parquet(parquet_path, index=False)
    print(f"Appended data and updated {parquet_path}")

def DA_exp_main(kf_opts: KF_Opts, DA_opts: DA_Opts, root) -> None:
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
    DA_opts.ic_init.get_attractor_snaps(attractor_snapshots)

    os.makedirs(root, exist_ok=True)
    parquet_path = os.path.join(root, "results.parquet")


    total_cases = (
        len(DA_opts.TIC_seed_list)
        * len(DA_opts.T_list)
        * len(DA_opts.n_particles_list)
        * len(DA_opts.NT_list)
        * len(DA_opts.PIC_seed_list)
        * DA_opts.num_opt_inits
        * len(DA_opts.optimizer_list)
        * len(DA_opts.crit_list)
    )
    count = 0
    # Loop over experiment seeds
    for seed_idx in DA_opts.TIC_seed_list:
        seed_root = os.path.join(root, f"seed_idx={seed_idx}")
        os.makedirs(seed_root, exist_ok=True)

        # Load or create the target initial condition U_0 for this seed
        U_0_path = os.path.join(seed_root, "U_0.npy")
        if os.path.exists(U_0_path):
            U_0 = np.load(U_0_path)
        else:
            rng = np.random.default_rng(seed_idx)   # ← seed used HERE
            idx = rng.integers(
                low=0,
                high=attractor_snapshots.shape[0],
            )
            U_0 = attractor_snapshots[idx, :]
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
                vel_part_trans = Vel_Part_Transformations(kf_opts.NDOF, npart)
                if isinstance(DA_opts.ic_init, CS_init):
                    DA_opts.ic_init.set_transform(stepper, vel_part_trans)
                
                for NT in DA_opts.NT_list:
                    NT_root = os.path.join(npart_root, f"NT={NT}")
                    #t_mask = jnp.linspace(0, int(T/kf_opts.dt)+1)
                    samp_period = T/(NT-1)
                    period_idx = int(samp_period/kf_opts.dt)
                    idx = jnp.arange(int(T/kf_opts.dt)+1)
                    t_mask = (idx % period_idx == 0)

                    # Loop over particle initializations
                    
                    PI_root_base = os.path.join(NT_root, "PI")
                    for PIC_seed in DA_opts.PIC_seed_list:
                        # ------------------------------------------------
                        # 1) Choose a repeatable seed and folder name
                        # ------------------------------------------------
                        PI_root = os.path.join(PI_root_base, f"seed_{PIC_seed}")
                        os.makedirs(PI_root, exist_ok=True)

                        pIC_path = os.path.join(PI_root, "pIC.npy")
                        fig_path = os.path.join(PI_root, "particle_IC.png")

                        # ------------------------------------------------
                        # 2) If pIC already exists, just load it
                        # ------------------------------------------------
                        if os.path.exists(pIC_path):
                            print(f"[PI] Using existing pIC for seed {PIC_seed}")
                            pIC = np.load(pIC_path)

                        else:
                            print(f"[PI] Generating new pIC for seed {PIC_seed}")
                            rng = np.random.default_rng(PIC_seed)

                            # Random particle ICs in the periodic domain
                            pIC = init_particles_vector(
                                npart,
                                vel_part_trans.reshape_flattened_vel(U_0),
                                (0, RHS.KF_RHS.L),
                                (0, RHS.KF_RHS.L),
                                RHS.KF_RHS.L,
                                rng=rng,
                            )

                            # Save pIC for reuse
                            np.save(pIC_path, pIC)

                            # Quick visualization of particle ICs (only when newly generated)
                            fig, ax = plot_particles(pIC, RHS.KF_RHS.L, ax=None, s=20)
                            fig.savefig(fig_path)
                            plt.close(fig)

                        # Generate target trajectory by advecting particles with the true U_0
                        trj_gen_fn = create_trj_generator(RHS, kf_opts.dt, T)
                        target_trj = trj_gen_fn(pIC, U_0)

                        for loss_crit in DA_opts.crit_list:
                            loss_crit.init_obj(t_mask, RHS.KF_RHS.L, vel_part_trans)
                            crit_dir = os.path.join(PI_root, f"{loss_crit}")
                            loss_fn = create_loss_fn(
                                loss_crit, stepper, target_trj, pIC, vel_part_trans
                            )

                            # Loop over opt inits
                            for _ in range(DA_opts.num_opt_inits):
                                U_0_guess, actual_norm_dist = DA_opts.ic_init(U_0, pIC, loss_fn)
                                opt_init_dir = os.path.join(crit_dir, "cases", f"{actual_norm_dist}")

                                # Skip if this case directory already exists
                                if os.path.isdir(opt_init_dir):
                                    print("skipping")
                                    continue

                                os.makedirs(opt_init_dir)
                                np.save(os.path.join(opt_init_dir, "U_0_guess.npy"), U_0_guess)
                    
                                # For each optimizer and loss criterion, run a DA case
                                for optimizer in DA_opts.optimizer_list:
                                    opt_method_dir = os.path.join(
                                        opt_init_dir, f"{optimizer}"
                                )
                                    for vfloat in DA_opts.vp_list:
                                        if vfloat is None:
                                            vfloat_name = "double"
                                        else:
                                            vfloat_name = f"{vfloat}"
                                        vfloat_dir = os.path.join(opt_method_dir, vfloat_name)

                                        loss_fn_and_derivs = Loss_and_Deriv_fns(loss_crit, stepper, target_trj, pIC, vel_part_trans, kf_opts.dt, T, vfloat)
                                        os.makedirs(vfloat_dir, exist_ok=True)
                                        def omega_fn(U):
                                            U = vel_part_trans.reshape_flattened_vel(U)
                                            u_hat = jnp.fft.rfft2(U[0])
                                            v_hat = jnp.fft.rfft2(U[1])
                                            return RHS.KF_RHS.vorticity_real(u_hat, v_hat)
                                        
                                        def div_check(U_fourier):
                                            U_hat = vel_part_trans.vel_Fourier_2_vel_hat(U_fourier)
                                            div_field = jnp.fft.irfft2(RHS.KF_RHS.dxop * U_hat[0] + RHS.KF_RHS.dyop * U_hat[1])
                                            return jnp.mean(jnp.abs(div_field))
                                        
                                        results_df = pd.DataFrame({
                                                                    "true_IC_seed": [seed_idx],
                                                                    "PIC_seed" : [PIC_seed],
                                                                    "T": [T],
                                                                    "n_part": [npart],
                                                                    "NT": [NT],
                                                                    "init_IC_distance": [float(actual_norm_dist)],
                                                                    "optimizer": [f"{optimizer}"],
                                                                    "loss_crit": [f"{loss_crit}"],
                                                                    "floatp": [vfloat_name]


                                                                })
                                        div_free_proj = build_div_free_proj(stepper, vel_part_trans, return_type="Fourier_flat")
                

                                        _run_DA_case(target_trj, U_0_guess, loss_fn_and_derivs, optimizer, trj_gen_fn, pIC, vfloat_dir, kf_opts.dt,
                                                    omega_fn, div_check, div_free_proj, vel_part_trans, t_mask, results_df,
                                                    parquet_path)
                                        count += 1
                                        print(f"case: {count}/{total_cases}")


    return root

def _run_DA_case(
    target_trj: np.ndarray | jnp.ndarray,
    U_0_guess: np.ndarray | jnp.ndarray,
    loss_fn_and_derivs: Loss_and_Deriv_fns,
    optimizer: LS_TR_Opt,
    trj_gen_fn,
    pIC,
    save_dir,
    dt,
    omega_fn,
    div_check,
    div_free_proj,
    vel_part_trans: Vel_Part_Transformations,
    t_mask,
    results_df,
    parquet_path
    
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
    loss_fn_and_derivs.reset_cost_count()
    U_0_guess_fourier = vel_part_trans.vel_flat_2_vel_Fourier(U_0_guess)
    U_0_DA_fourier, opt_data = optimizer.opt_loop(U_0_guess_fourier, loss_fn_and_derivs, div_check, div_free_proj)

    U_0_DA_hat = vel_part_trans.vel_Fourier_2_vel_hat(U_0_DA_fourier)
    u = jnp.fft.irfft2(U_0_DA_hat[0])
    v = jnp.fft.irfft2(U_0_DA_hat[1])
    U_0_DA = jnp.stack([u, v], axis=0).reshape(-1)

    DA_trj = trj_gen_fn(pIC, U_0_DA)
    init_guess_trj = trj_gen_fn(pIC, vel_part_trans.reshape_flattened_vel(U_0_guess))

    if False:
        hvp = build_hvp(loss_fn, U_0_DA_fourier)
        print("eig decomp")
        A_op = LinearOperator((U_0_DA_fourier.shape[0], U_0_DA_fourier.shape[0]), matvec=hvp, dtype=np.float64)
        w, Q = eigsh(A_op, k=2, which='BE', tol=1e-6)
        print("done")
        
        max_H_eig = float(jnp.max(w))
        min_H_eig = float(jnp.min(w))

        with open(os.path.join(save_dir, "extreme_H_eigs.txt"), "w") as f:
            f.write(f"lambda_max={max_H_eig} \n lambda_min={min_H_eig}")

        results_df["max_H_eig"] = [max_H_eig]
        results_df["min_H_eig"] = [min_H_eig]
    
    results_df["loss_record"] = [opt_data.loss_record]
    results_df["loss_evals_record"] = [opt_data.loss_evals_record]
    results_df["loss_grad_evals_record"] = [opt_data.loss_grad_evals_record]
    results_df["Hvp_evals_record"] = [opt_data.Hvp_evals_record]

    #saving npy files
    np.save(os.path.join(save_dir, "DA_trj.npy"), np.array(DA_trj))
    opt_data.save_data(save_dir)

    n_particles = pIC.shape[0]//4
    post_proc_case_main(target_trj, DA_trj, init_guess_trj, opt_data, n_particles, save_dir, dt, omega_fn, t_mask, results_df)
    append_to_parquet(results_df, parquet_path)
    
    #cleanup
    del  loss_fn_and_derivs
    jax.clear_caches()
    gc.collect()


    
    
