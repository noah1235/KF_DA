# =========================
# Imports (yours + minimal extras)
# =========================
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import config

from create_results_dir import create_results_dir
from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.DA_Comp.adjoint import get_loss_grad_vp_fn, get_loss_grad_fn, get_forced_adj_shooting
from SRC.parameterization.Fourier_Param import Fourier_Param
from SRC.plotting_utils import save_svg
from SRC.Solver.IC_gen import init_particles_vector
from SRC.Solver.solver import KF_Stepper, KF_TP_Stepper, create_omega_part_gen_fn, Omega_Integrator
from SRC.utils import load_data
from SRC.vp_floats.vp_py_utils import choose_exponent_format, float_pos_range

config.update("jax_enable_x64", True)

# =========================
# Main
# =========================
def main():
    kf_opts = KF_Opts(
        Re=100,
        n=4,
        NDOF=128,
        dt=1e-2,
        total_T=1000,
        min_samp_T=50,
        t_skip=1e-1,
    )
    T_LLE = 3.3
    IC_param = Fourier_Param(kf_opts.NDOF, 64)
    T = T_LLE * 1
    n_steps = int(T/kf_opts.dt)
    seed = 10


    crit = MSE_Vel()
    attractor_snapshots = load_data(kf_opts)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, attractor_snapshots.shape[0])
    omega0_hat = attractor_snapshots[idx]

    kf_stepper = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt)
    omega_int = Omega_Integrator(kf_stepper)

    def f(z):
        x = IC_param.inv_transform(z)
        return omega_int.fv_integrate(x, n_steps).reshape(-1)

    J_fn = jax.jacfwd(f)
    #J = J_fn(omega0_hat)
    z0 = IC_param.transform(omega0_hat)
    print(z0.shape)
    J = J_fn(z0)
    print(J.shape)


if __name__ == "__main__":
    main()
