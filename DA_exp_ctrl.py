from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.Solver.KF_intergrators import KF_LPT_PS_RHS, create_trj_generator, create_trj_sens_generator
from SRC.DA_Comp.DA_engine import DA_exp_main
from SRC.DA_Comp.optimization.optimization import BFGS, DA_SR1
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch, Cubic_TR
from SRC.utils import load_data
import numpy as np
from SRC.Solver.IC_gen import init_particles_vector
from SRC.DA_Comp.loss_funcs import create_loss_fn
from SRC.DA_Comp.adjoint import build_adjoint_grad_fn, build_adjoint_Hess_fn
import jax
from SRC.function_perf_bench import bench
from SRC.utils import build_div_free_proj
import os
from create_results_dir import create_results_dir
import pandas as pd
from jax import config
config.update("jax_enable_x64", True)

def parquet_to_excel(parquet_path, excel_path=None):
    """
    Copy the contents of a Parquet file to an Excel (.xlsx) file.

    Parameters
    ----------
    parquet_path : str
        Path to the source Parquet file.
    excel_path : str, optional
        Path to the output Excel file. If None, uses same base name.

    Returns
    -------
    str
        Path to the saved Excel file.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Default Excel filename if not provided
    if excel_path is None:
        base = os.path.splitext(parquet_path)[0]
        excel_path = base + ".xlsx"

    # Read parquet and write to Excel
    df = pd.read_parquet(parquet_path)
    df.to_excel(excel_path, index=False)

    print(f"Saved Excel file: {excel_path} (rows={len(df)}, cols={len(df.columns)})")
    return excel_path

def main():
    kf_opts = KF_Opts(
        Re = 100,
        n = 4,
        NDOF = 32,
        dt = 1e-2,
        T = 1e3,
        min_samp_T=500,
        t_skip=1e-1

    )
    DA_opts = DA_Opts(
        n_particles_list=[40],
        sampling_period_list=[.1],
        part_opts=Particle_Opts(St=1e-2, beta=0),
        num_particle_inits=1,
        num_opt_inits=1,
        num_seeds=1,
        int_pert_range=(.8, .9),
        T_list=[1],
        optimizer_list=[
            #NCN(ls_method="BT", its=10, cond_num_cutoff=1e4)
            #LBFGS(its=20),
            #BFGS(ls=ArmijoLineSearch(alpha_init=1.0, rho=0.5, c=1e-4, max_iters=10), its=10, fallback_opt="eye", print_loss=True),
            DA_SR1(its=200, eps_H=1e-6, max_memory=50,
                   cubic_TR=Cubic_TR(rho=100, eta_min=1e-14, eta_0=1, eta_max=1e6),
                   grad_prob=0.9, neg_curve_prob=.125, num_hvp_iters=5, print_loss=True
                   )

            #ADAM(1e-4, its=100)
        ],
        crit_list=[
            #MSE_PP(),
            MSE_Vel()
        ]
    )

    root = os.path.join(
        create_results_dir(),
        (
            f"DA_Re={kf_opts.Re}_n={kf_opts.n}_dt={kf_opts.dt}_NDOF={kf_opts.NDOF}"
            f"-St={DA_opts.part_opts.St}_beta={DA_opts.part_opts.beta}"
        ),
    )

    DA_exp_main(kf_opts, DA_opts, root)
    parquet_to_excel(os.path.join(root, "results.parquet"), os.path.join(root, "results.xlsx"))

def adjoint_test():
    kf_opts = KF_Opts(
        Re = 10,
        n = 4,
        NDOF = 16,
        dt = 1e-2,
        T = 1e3,
        min_samp_T=500
    )
    transform = True
    grad_comp = True
    Hess_comp = False
    npart = 100
    T = .1

    attractor_snapshots = load_data(kf_opts)
    U_true = attractor_snapshots[np.random.randint(0, attractor_snapshots.shape[0]-1), :]
    U_DA = attractor_snapshots[np.random.randint(0, attractor_snapshots.shape[0]-1), :]


    pIC = init_particles_vector(npart, (0, 2 * np.pi), (0, 2 * np.pi))

    RHS = KF_LPT_PS_RHS(
        kf_opts.NDOF,
        kf_opts.Re,
        kf_opts.n,
        npart,
        beta=0,
        St=1e-2,
    )
    stepper = Particle_Stepper(RK4_Step(RHS, kf_opts.dt), npart)
    trj_gen_fn = create_trj_generator(RHS, kf_opts.dt, T)
    target_trj = trj_gen_fn(pIC, U_true)
    target_part = target_trj[:, : RHS.n_particles * 4]

    crit = MSE(jnp.ones(target_part.shape[0]))

    loss_fn = create_loss_fn(crit, stepper, target_part, pIC, transform=transform)
    loss_grad_fn_auto = jax.jit(jax.value_and_grad(loss_fn))
    hess_fn = jax.jit(jax.hessian(loss_fn))
    loss_auto, grad_auto = loss_grad_fn_auto(U_DA)

    if transform is None:
        adj_transform = lambda x: x
    else:
        adj_transform = build_div_free_proj(stepper)

    if grad_comp:
        loss_grad_fn_adj = build_adjoint_grad_fn(pIC, crit, target_part, trj_gen_fn, stepper, adj_transform)
        loss_adj, grad_adj = loss_grad_fn_adj(U_DA)

        loss_per_error = jnp.abs(loss_adj - loss_auto)/ loss_auto * 100
        grad_error = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto)
        print(f"Loss Percent Error: {loss_per_error}")
        print(f"grad error: {grad_error}")

        s1 = bench(loss_grad_fn_auto, U_DA, runs=1)
        s2 = bench(loss_grad_fn_adj, U_DA, runs=1)

        print("auto:", s1)
        print("adjoint:", s2)
    
    if Hess_comp:
        trj_sens_gen_fn = create_trj_sens_generator(RHS, kf_opts.dt, T)
        loss_grad_Hess_fn = build_adjoint_Hess_fn(pIC, crit, target_part, trj_sens_gen_fn, stepper, adj_transform)
        loss_adj, grad_adj, Hess_adj = loss_grad_Hess_fn(U_DA)

        loss_per_error = jnp.abs(loss_adj - loss_auto)/ loss_auto * 100
        grad_error = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto)
        print(f"Loss Percent Error: {loss_per_error}")
        print(f"grad error: {grad_error}")

        sym_err = jnp.linalg.norm(Hess_adj - Hess_adj.T) / jnp.linalg.norm(Hess_adj)
        print("symmetry rel. error:", sym_err)

        hess_auto = hess_fn(U_DA)
        hess_error = jnp.linalg.norm(hess_auto - Hess_adj) / jnp.linalg.norm(hess_auto)
        print(f"Hess error: {hess_error}")

        s1 = bench(hess_fn, U_DA, runs=1)
        s2 = bench(loss_grad_Hess_fn, U_DA, runs=1)

        print("auto:", s1)
        print("adjoint:", s2)


if __name__ == "__main__":
    main()