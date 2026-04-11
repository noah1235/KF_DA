from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.DA_Comp.DA_engine import DA_exp_main
from SRC.DA_Comp.optimization.optimization import BFGS, NCSR1, NCSR1_and_LBFGS
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch
from SRC.DA_Comp.optimization.parent_classes import Psuedo_Projection
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
from SRC.global_post.global_post_main import global_post_main
from SRC.parameterization.Fourier_Param import Fourier_Param
config.update("jax_enable_x64", True)
from pathlib import Path

def write_hierarchical_case_summary(
    folder,
    filename="results.xlsx",
    out_name="case_summary.txt",
):
    """
    Writes a hierarchical text summary:

    PIC_seed, T, n_part, NT
        └── true_IC_seed
                → number of unique init_IC_seed
    """

    folder = Path(folder)
    in_path = folder / filename
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find: {in_path}")

    # --- load file ---
    suffix = in_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_excel(in_path)


    # columns for hierarchy
    top_cols = ["PIC_seed", "T", "n_part", "NT"]
    required = top_cols + ["true_IC_seed", "init_IC_seed"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # normalize seed dtype
    df["init_IC_seed"] = df["init_IC_seed"].astype("Int64", errors="ignore")

    # count unique init seeds
    counts = (
        df.groupby(top_cols + ["true_IC_seed"], dropna=False)
          .agg(n_unique_init_IC_seed=("init_IC_seed", pd.Series.nunique))
          .reset_index()
          .sort_values(top_cols + ["true_IC_seed"])
    )

    # write text file
    out_path = folder / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"File: {in_path.name}\n")
        f.write(f"Total rows (after filtering): {len(df)}\n\n")

        for top_vals, top_df in counts.groupby(top_cols):
            PIC_seed, T, n_part, NT = top_vals

            f.write(
                f"PIC_seed={PIC_seed}, T={T}, n_part={n_part}, NT={NT}\n"
            )

            for _, row in top_df.iterrows():
                f.write(
                    f"    true_IC_seed={row['true_IC_seed']}: "
                    f"{int(row['n_unique_init_IC_seed'])} init_IC_seed\n"
                )

            f.write("\n")

    return out_path


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
        Re = 60,   
        n = 4,
        NDOF = 128,
        dt = 1e-2,
        total_T=int(1e4),
        min_samp_T=100,
        t_skip=1
    )

    #Re = 100 | T = 3,3
    #Re = 60 | T = 4.1
    #Re = 40 | T = 7.2

    BT_ls = ArmijoLineSearch(alpha_init=1.0, rho=0.25, c=1e-4, max_iters=5)

    DA_opts = DA_Opts(
        m_dt=None,
        n_particles_list=[45],
        NT_list=[6],
        part_opts=Particle_Opts(St=0, beta=0),
        PIC_seed_list=[0],
        num_opt_inits=10,
        TIC_seed_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ic_init=AI(min_norm=.1, max_norm=jnp.inf),
        #ic_init=CS_init(l1_weight=1e-6, can_modes=jnp.arange(2, 16, 2)),
        T_list=[7.2],
        optimizer_list=[
            BFGS(
                ls=BT_ls, 
                #Cubic_TR(rho_trg=1, eta_kp=1.0, eta_ki=0, eta_kd=0, eta_min=1e-14, eta_0=1-4, eta_max=1e0),
                #psuedo_proj=Psuedo_Projection(it_list=[24, 49, 74], T=.25),
                 its=150, max_mem=20, eps_H=1e-10, print_loss=True),
            #BFGS(
            #    ls=BT_ls, 
                #Cubic_TR(rho_trg=1, eta_kp=1.0, eta_ki=0, eta_kd=0, eta_min=1e-14, eta_0=1-4, eta_max=1e0),
            #its=200, max_mem=20, eps_H=1e-10, print_loss=True)
        ],
        vp_list=[None, 
                #VP_Float_Settings(mbits=4, minv=1e-3, maxv=10),
                 #VP_Float_Settings(mbits=8, minv=1e-3, maxv=10),
                 #VP_Float_Settings(mbits=12, minv=1e-3, maxv=10)
                 ],
        crit_list=[
            MSE_PP(),
            #MSE_Vel()
        ],
        IC_param_list=[Fourier_Param(kf_opts.NDOF, kf_opts.NDOF//2, beta=.1, Re=kf_opts.Re)]
    )

    root = os.path.join(
        create_results_dir(),
        (
            f"DA_Re={kf_opts.Re}_n={kf_opts.n}_dt={kf_opts.dt}_NDOF={kf_opts.NDOF}_mdt={DA_opts.m_dt}"
            f"-St={DA_opts.part_opts.St}_beta={DA_opts.part_opts.beta}_{DA_opts.ic_init}"
        ),
    )

    DA_exp_main(kf_opts, DA_opts, root)
    parquet_to_excel(os.path.join(root, "results.parquet"), os.path.join(root, "results.xlsx"))
    write_hierarchical_case_summary(root)
    df = pd.read_parquet(os.path.join(root, "results.parquet"))
    df = df.dropna()
    global_post_main(df, root)

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