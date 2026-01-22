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
from SRC.DA_Comp.loss_funcs import create_loss_fn
from SRC.DA_Comp.optimization.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from create_results_dir import create_results_dir
from SRC.Solver.ploting import plot_vorticity
from SRC.Vel_init.CS_init import CS_init
from SRC.Vel_init.AI import AI
from SRC.Solver.solver import KF_TP_Stepper, create_omega_part_gen_fn

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
         - Chooses/loads a target initial condition omega0_hat.
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
        seed_root = os.path.join(root, f"IC_seed={seed_idx}")
        os.makedirs(seed_root, exist_ok=True)

        # Load or create the target initial condition omega0_hat for this seed
        U_0_path = os.path.join(seed_root, "omega0_hat.npy")
        if os.path.exists(U_0_path):
            omega0_hat = np.load(U_0_path)
        else:
            rng = np.random.default_rng(seed_idx)   # ← seed used HERE
            idx = rng.integers(
                low=0,
                high=attractor_snapshots.shape[0],
            )
            omega0_hat = attractor_snapshots[idx, :]
            np.save(U_0_path, omega0_hat)

        # Loop over time horizons
        for T in DA_opts.T_list:
            T_dir = os.path.join(seed_root, f"T={T}")

            # Loop over number of particles
            for npart in DA_opts.n_particles_list:
                npart_root = os.path.join(T_dir, f"np={npart}")

                stepper = KF_TP_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt, DA_opts.part_opts.St, DA_opts.part_opts.beta, npart)
                if isinstance(DA_opts.ic_init, CS_init):
                    return
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
                        fig_path = os.path.join(PI_root, "particle_IC.png")

                        # Random particle ICs in the periodic domain
                        u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega0_hat)
                        u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)
                        xp, yp, up, vp = init_particles_vector(npart, u, v, (0, stepper.NS.L), (0, stepper.NS.L), stepper.NS.L, seed=PIC_seed)
                        particle_IC = (xp, yp, up, vp)
                        # Quick visualization of particle ICs (only when newly generated)
                        fig, _ = plot_particles(xp, yp, stepper.NS.L, ax=None, s=20)
                        fig.savefig(fig_path)
                        plt.close(fig)

                        trj_gen_fn = create_omega_part_gen_fn(jax.jit(stepper), T)
                        #tuple (omega_traj, xp_traj, yp_traj, up_traj, vp_traj)
                        target_trj = trj_gen_fn(omega0_hat, xp, yp, up, vp)

                        for loss_crit in DA_opts.crit_list:
                            loss_crit.init_obj(t_mask, stepper.NS.L)
                            crit_dir = os.path.join(PI_root, f"{loss_crit}")

                            # For each optimizer and loss criterion, run a DA case
                            for optimizer in DA_opts.optimizer_list:
                                opt_method_dir = os.path.join(
                                    crit_dir, f"{optimizer}"
                            )
                                for vfloat in DA_opts.vp_list:
                                    if vfloat is None:
                                        vfloat_name = "double"
                                    else:
                                        vfloat_name = f"{vfloat}"
                                    vfloat_dir = os.path.join(opt_method_dir, vfloat_name)
                                    for IC_param in DA_opts.IC_param_list:
                                        param_dir = os.path.join(vfloat_dir, f"{IC_param}")
                                        loss_fn_and_derivs = Loss_and_Deriv_fns(loss_crit, IC_param.inv_transform, stepper, target_trj, kf_opts.dt, T, vfloat)

                                        if isinstance(DA_opts.ic_init, AI):
                                            DA_opts.ic_init.set_unused_mask()
                                        for opt_init_key_num in range(DA_opts.num_opt_inits):
                                            omega0_guess_hat, actual_norm_dist = DA_opts.ic_init(omega0_hat, None, loss_fn_and_derivs.loss_fn_jit, opt_init_key_num)
                                            opt_init_dir = os.path.join(param_dir, "cases", f"{actual_norm_dist}")

                                            # Skip if this case directory already exists
                                            if os.path.isdir(opt_init_dir):
                                                print("skipping")
                                                continue

                                            os.makedirs(opt_init_dir)
                                            np.save(os.path.join(opt_init_dir, "omega0_guess_hat.npy"), omega0_guess_hat)

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

                                            _run_DA_case(target_trj, omega0_guess_hat, IC_param, loss_fn_and_derivs, optimizer, trj_gen_fn, particle_IC, opt_init_dir, kf_opts.dt,
                                                        t_mask, results_df,
                                                        parquet_path)
                                            count += 1
                                            print(f"case: {count}/{total_cases}")


    return root

def _run_DA_case(
    target_trj: jnp.ndarray,
    omega0_guess_hat:  jnp.ndarray,
    IC_param,
    loss_fn_and_derivs: Loss_and_Deriv_fns,
    optimizer: LS_TR_Opt,
    trj_gen_fn,
    pIC,
    save_dir,
    dt,
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
    Z0 = IC_param.transform(omega0_guess_hat)
    U_0_DA_hat, opt_data = optimizer.opt_loop(Z0, loss_fn_and_derivs)
    omega0_DA_hat = IC_param.inv_transform(U_0_DA_hat)
    DA_trj = trj_gen_fn(omega0_DA_hat, *pIC)
    init_guess_trj = trj_gen_fn(omega0_guess_hat, *pIC)

    
    results_df["loss_record"] = [opt_data.loss_record]
    results_df["loss_evals_record"] = [opt_data.loss_evals_record]
    results_df["loss_grad_evals_record"] = [opt_data.loss_grad_evals_record]
    results_df["Hvp_evals_record"] = [opt_data.Hvp_evals_record]

    #saving npy files
    np.save(os.path.join(save_dir, "omega_DA_trj.npy"), np.array(DA_trj[0]))
    opt_data.save_data(save_dir)

    post_proc_case_main(target_trj, DA_trj, init_guess_trj, opt_data, save_dir, dt, t_mask, results_df)
    append_to_parquet(results_df, parquet_path)
    
    #cleanup
    del  loss_fn_and_derivs
    jax.clear_caches()
    gc.collect()


    
    
