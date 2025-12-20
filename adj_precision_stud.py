from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.Solver.KF_intergrators import KF_LPT_PS_RHS, create_trj_generator, create_trj_sens_generator
from SRC.DA_Comp.DA_engine import DA_exp_main
from SRC.DA_Comp.optimization.optimization import L_BFGS, NCSR1, NCSR1_and_LBFGS
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch, Cubic_TR, Armijo_TR
from SRC.utils import load_data
import numpy as np
from SRC.Solver.IC_gen import init_particles_vector
from SRC.DA_Comp.loss_funcs import create_loss_fn
from SRC.DA_Comp.adjoint import Adjoint_Solver, get_loss_grad_fn
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
config.update("jax_enable_x64", True)

def adjoint_test():
    kf_opts = KF_Opts(
        Re = 100,   
        n = 4,
        NDOF = 32,
        dt = 1e-2,
        total_T=1000,
        min_samp_T=50,
        t_skip=1e-1
    )

    npart = 30
    T = 2
    adj_dtype = jnp.float16

    attractor_snapshots = load_data(kf_opts)
    #print(jnp.mean(attractor_snapshots), jnp.max(attractor_snapshots), jnp.min(attractor_snapshots))

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
    U_DA_fourier = U_DA_fourier.astype(jnp.float64)
    pIC = init_particles_vector(npart,vel_part_trans.reshape_flattened_vel(U_true), (0, RHS.KF_RHS.L), (0, RHS.KF_RHS.L), RHS.KF_RHS.L)

    stepper = Particle_Stepper(RK4_Step(RHS, kf_opts.dt), npart)
    trj_gen_fn = create_trj_generator(RHS, kf_opts.dt, T, dtype=adj_dtype)
    trj_gen_fn_64 = create_trj_generator(RHS, kf_opts.dt, T, dtype=jnp.float64)
    target_trj = trj_gen_fn_64(pIC, U_true)

    crit = MSE_Vel()
    #crit = MSE_PP()

    t_mask = np.ones(target_trj.shape[0])
    t_mask = jnp.array(t_mask) 
    crit.init_obj(t_mask, RHS.KF_RHS.L, vel_part_trans)

    loss_fn = create_loss_fn(
        crit, stepper, target_trj, pIC, vel_part_trans
    )
    loss_grad_fn_auto = jax.jit(jax.value_and_grad(loss_fn))
    loss_auto, grad_auto = loss_grad_fn_auto(U_DA_fourier)


    adj_transform = build_div_free_proj(stepper, vel_part_trans)

    loss_grad_fn_adj = get_loss_grad_fn(pIC, crit, target_trj, stepper, adj_transform,
                    vel_part_trans, trj_gen_fn)
    loss_grad_fn_adj = jax.jit(loss_grad_fn_adj)
    loss_adj, grad_adj = loss_grad_fn_adj(U_DA_fourier)

    loss_per_error = jnp.abs(loss_adj - loss_auto)/ loss_auto * 100
    grad_error = jnp.linalg.norm(grad_adj - grad_auto) / jnp.linalg.norm(grad_auto)
    print(f"Loss Percent Error: {loss_per_error}")
    print(f"grad error: {grad_error}")


    s1 = bench(loss_grad_fn_auto, U_DA_fourier, runs=1)
    s2 = bench(loss_grad_fn_adj, U_DA_fourier, runs=1)

    print("auto:", s1)
    print("adjoint:", s2)



if __name__ == "__main__":
    adjoint_test()