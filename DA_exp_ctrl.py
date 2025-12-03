from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.Solver.KF_intergrators import KF_LPT_PS_RHS, create_trj_generator, create_trj_sens_generator
from SRC.DA_Comp.DA_engine import DA_exp_main
from SRC.DA_Comp.optimization.optimization import BFGS, NCSR1, PCGBFGS, NCSR1_and_BFGS
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch, Cubic_TR
from SRC.utils import load_data
import numpy as np
from SRC.Solver.IC_gen import init_particles_vector
from SRC.DA_Comp.loss_funcs import create_loss_fn
from SRC.DA_Comp.adjoint import Adjoint_Solver
import jax
from SRC.function_perf_bench import bench
from SRC.utils import build_div_free_proj
import os
from create_results_dir import create_results_dir
import pandas as pd
from jax import config
from SRC.Vel_init.AI import AI
from SRC.Vel_init.CS_init import CS_init
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
        total_T=4000,
        min_samp_T=500,
        t_skip=1e-1
    )

    DA_opts = DA_Opts(
        n_particles_list=[50],
        sampling_period_list=[.1],
        part_opts=Particle_Opts(St=0, beta=0),
        num_particle_inits=2,
        num_opt_inits=1,
        num_seeds=5,
        #ic_init=AI(min_norm=.1, max_norm=1),
        ic_init=CS_init(l1_weight=1e-6, can_modes=jnp.arange(2, 16, 2)),
        T_list=[1.5],
        optimizer_list=[
            #NCN(ls_method="BT", its=10, cond_num_cutoff=1e4)
            #BFGS(ls=ArmijoLineSearch(alpha_init=1.0, rho=0.5, c=1e-4, max_iters=10), its=50, fallback_opt="eye", print_loss=True),
            
            #NCSR1(its=25, eps_H=1e-6, max_memory=20,
            #       cubic_TR=Cubic_TR(rho=30, eta_min=1e-14, eta_0=1e-2, eta_max=1e6),
            #       grad_prob=0.9, neg_curve_prob=0, num_hvp_iters=5, print_loss=True
            #       ),

            #NCSR1(its=5, eps_H=1e-6, max_memory=50,
            #cubic_TR=Cubic_TR(rho_trg=.8, eta_kp=0.7, eta_ki=.12, eta_kd=1, eta_min=1e-14, eta_0=1e-4, eta_max=1e6),
            #grad_prob=0.9, neg_curve_prob=0, num_hvp_iters=5, psd_stop=False,
            #print_loss=True
            #,

            NCSR1_and_BFGS(
                NCSR1(its=1, eps_H=1e-6, max_memory=50,
                cubic_TR=Cubic_TR(rho_trg=.8, eta_kp=0.7, eta_ki=.12, eta_kd=1, eta_min=1e-14, eta_0=1e-4, eta_max=1e6),
                num_batch_hvp=2,
                num_power_iters=1,
                print_loss=True
                ),
                BFGS(
                    #ls=ArmijoLineSearch(alpha_init=1.0, rho=0.5, c=1e-4, max_iters=10), 
                    ls=Cubic_TR(rho_trg=.8, eta_kp=0.7, eta_ki=.12, eta_kd=1, eta_min=1e-14, eta_0=1e-4, eta_max=1e6),
                    its=1, fallback_opt="eye", print_loss=True),
            ),

            #NCSR1(its=4, eps_H=1e-6, max_memory=50,
            #cubic_TR=Cubic_TR(rho_trg=.8, eta_kp=0.7, eta_ki=.12, eta_kd=1, eta_min=1e-14, eta_0=1e-4, eta_max=1e6),
            #num_batch_hvp=5,
            #num_power_iters=2,
            #print_loss=True
            #),
            #PCGBFGS(ls=ArmijoLineSearch(alpha_init=1.0, rho=0.5, c=1e-4, max_iters=10), its=2, n_hvp=5, fallback_opt="eye", print_loss=True),
            #BFGS(ls=ArmijoLineSearch(alpha_init=1.0, rho=0.5, c=1e-4, max_iters=10), its=200, fallback_opt="eye", print_loss=True),
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
            f"-St={DA_opts.part_opts.St}_beta={DA_opts.part_opts.beta}_{DA_opts.ic_init}"
        ),
    )

    DA_exp_main(kf_opts, DA_opts, root)
    parquet_to_excel(os.path.join(root, "results.parquet"), os.path.join(root, "results.xlsx"))

def adjoint_test():
    def hvp_many(loss_fn, x, V):
        """
        Compute k Hessian–vector products H(x) v_i for a scalar loss.

        Args
        ----
        loss_fn : callable
            Function loss_fn(x) -> scalar.
        x : array, shape (n,)
            Point at which to evaluate the Hessian.
        V : array, shape (n, k)
            Columns are direction vectors v_i.

        Returns
        -------
        loss : scalar
            loss_fn(x)
        grad : array, shape (n,)
            Gradient at x.
        HV : array, shape (n, k)
            Column i is H(x) @ V[:, i].
        """
        grad_fn = jax.grad(loss_fn)

        def loss_and_grad(x):
            return loss_fn(x), grad_fn(x)

        # One linearization: gives (loss, grad) and a linear map
        (loss, grad), lin = jax.linearize(loss_and_grad, x)

        # lin(v) returns (d_loss[v], H v); we only want H v
        def hvp_one(v):
            _, hv = lin(v)
            return hv

        HV = jax.vmap(hvp_one, in_axes=1, out_axes=1)(V)  # (n, k)
        return loss, grad, HV


    kf_opts = KF_Opts(
        Re = 100,
        n = 4,
        NDOF = 32,
        dt = 1e-2,
        total_T=2000,
        min_samp_T=500,
        t_skip=1e-1

    )
    transform = True
    grad_comp = False
    Hess_comp = True
    npart = 30
    T = 2

    attractor_snapshots = load_data(kf_opts)
    U_true = attractor_snapshots[np.random.randint(0, attractor_snapshots.shape[0]-1), :]
    U_DA = attractor_snapshots[np.random.randint(0, attractor_snapshots.shape[0]-1), :]


    RHS = KF_LPT_PS_RHS(
        kf_opts.NDOF,
        kf_opts.Re,
        kf_opts.n,
        npart,
        beta=0,
        St=1e-2,
    )
    vel_part_trans = Vel_Part_Transformations(kf_opts.NDOF, npart)
    U_DA_fourier = vel_part_trans.vel_flat_2_vel_Fourier(U_DA)
    pIC = init_particles_vector(npart,vel_part_trans.reshape_flattened_vel(U_true), (0, RHS.KF_RHS.L), (0, RHS.KF_RHS.L), RHS.KF_RHS.L)

    stepper = Particle_Stepper(RK4_Step(RHS, kf_opts.dt), npart)
    trj_gen_fn = create_trj_generator(RHS, kf_opts.dt, T)
    target_trj = trj_gen_fn(pIC, U_true)


    crit = MSE_Vel()
    #crit = MSE_PP()

    t_mask = np.ones(target_trj.shape[0])
    t_mask = jnp.array(t_mask)
    crit.init_obj(t_mask, RHS.KF_RHS.L, vel_part_trans)

    loss_fn = create_loss_fn(
        crit, stepper, target_trj, pIC, vel_part_trans
    )
    loss_grad_fn_auto = jax.jit(jax.value_and_grad(loss_fn))
    hess_fn = jax.jit(jax.hessian(loss_fn))
    loss_auto, grad_auto = loss_grad_fn_auto(U_DA_fourier)

    if transform is None:
        adj_transform = lambda x: x
    else:
        adj_transform = build_div_free_proj(stepper, vel_part_trans)

    if grad_comp:
        loss_grad_fn_adj = build_adjoint_grad_fn(pIC, crit, target_trj, trj_gen_fn, stepper, adj_transform, vel_part_trans)

        loss_adj, grad_adj = loss_grad_fn_adj(U_DA_fourier)

        loss_per_error = jnp.abs(loss_adj - loss_auto)/ loss_auto * 100
        grad_error = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto)
        print(f"Loss Percent Error: {loss_per_error}")
        print(f"grad error: {grad_error}")


        s1 = bench(loss_grad_fn_auto, U_DA_fourier, runs=1)
        s2 = bench(loss_grad_fn_adj, U_DA_fourier, runs=1)

        print("auto:", s1)
        print("adjoint:", s2)
    
    if Hess_comp:

        adj_solver = Adjoint_Solver(pIC, crit, target_trj, stepper, adj_transform,
                        vel_part_trans, trj_gen_fn)
        loss_adj, grad_adj = adj_solver.compute_grad(U_DA_fourier)
        loss_per_error = jnp.abs(loss_adj - loss_auto)/ loss_auto * 100
        grad_error = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto)
        print(f"Loss Percent Error: {loss_per_error}")
        print(f"grad error: {grad_error}")


        key = jax.random.PRNGKey(0)  # seed
        v = jax.random.normal(key, (U_DA_fourier.shape[0], 3))
        HV_adj = adj_solver.compute_Hvp(v)
        _, _, HV_auto = hvp_many(loss_fn, U_DA_fourier, v)
        hess_error = jnp.linalg.norm(HV_auto - HV_adj) / jnp.linalg.norm(HV_auto)
        print(f"Hess error: {hess_error}")
        return


        trj_sens_gen_fn = create_trj_sens_generator(RHS, kf_opts.dt, T)

        loss_grad_Hess_fn = build_adjoint_Hess_fn(pIC, crit, target_trj, stepper, adj_transform, vel_part_trans, trj_sens_gen_fn)
        loss_adj, grad_adj, HV_adj = loss_grad_Hess_fn(U_DA_fourier, v)




        #sym_err = jnp.linalg.norm(Hess_adj - Hess_adj.T) / jnp.linalg.norm(Hess_adj)
        #print("symmetry rel. error:", sym_err)

        _, _, HV_auto = hvp_many(loss_fn, U_DA_fourier, v)
        print(HV_auto.shape, HV_adj.shape)

        hess_error = jnp.linalg.norm(HV_auto - HV_adj) / jnp.linalg.norm(HV_auto)
        print(f"Hess error: {hess_error}")

        #s1 = bench(hess_fn, U_DA_fourier, runs=1)
        #s2 = bench(loss_grad_Hess_fn, U_DA_fourier, runs=1)

        #print("auto:", s1)
        #print("adjoint:", s2)


if __name__ == "__main__":
    main()
    #adjoint_test()