# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kolmogorov Flow Data Assimilation (KF_DA)** — a research project combining 2D pseudo-spectral fluid simulation, Lagrangian particle tracking, and inverse-problem optimization. The scientific goal is to infer a flow's initial condition from noisy particle position observations.

Key physical model: forced 2D Navier-Stokes (Kolmogorov flow) on a periodic domain, solved with a pseudo-spectral method + RK4 time stepping.

## Environment & Build

This project uses **UV** as the package manager. Python 3.14 is required (see `.python-version`).

```bash
uv venv && source .venv/bin/activate
uv sync --frozen          # Install locked dependencies
```

The `vp_floats` submodule is a C++ Pybind11 extension for variable-precision floats. VSCode has a build task configured in `.vscode/tasks.json`; the extension must be compiled before VP float experiments run.

Docker builds are published to GHCR via `.github/workflows/publish-ghcr.yaml`:
```bash
docker build -t kf-da:latest .
docker run --gpus all kf-da:latest python main_scripts/DA_exp_ctrl.py
```

## Running Experiments

Configuration is loaded from an external YAML file at `../kf-da-configs/daExpConfig.yaml` (outside this repo). The main entry points are:

```bash
python main_scripts/DA_exp_ctrl.py        # Main DA experiment loop
python main_scripts/trj_generator.py      # Trajectory generation / animation
```

Diagnostic scripts in `diag_scripts/` are standalone analyses (Lyapunov exponents, adjoint precision, particle entropy, etc.) and can be run directly.

## Architecture

### Core Data Flow

```
DA_exp_ctrl.py
  └─ DA_engine.py::DA_exp_main()
       ├─ Loads attractor snapshots (pre-computed)
       ├─ For each seed → true IC → true trajectory + particle positions
       └─ For each (T, n_particles, optimizer):
            ├─ Initial guess from attractor (velInit/)
            ├─ Parametrize IC via Fourier modes (icParam/Fourier_Param.py)
            └─ Optimize IC to match particle observations
                 ├─ Loss: MSE on particle positions (loss_funcs.py)
                 ├─ Gradients: adjoint solver (adjoint.py)
                 └─ Optimizer: BFGS / Quasi-Newton (opti/)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/kf_da/solver/solver.py` | Core physics: `Forced_2D_NS`, `KF_Stepper`, `KF_TP_Stepper` (flow + particles), `Omega_Integrator` |
| `src/kf_da/daComp/DA_engine.py` | Experiment orchestration; outer loops over seeds, parameters, optimizers |
| `src/kf_da/daComp/configs.py` | Config dataclasses: `KF_Opts`, `DA_Opts`, `Particle_Opts`, `VP_Float_Settings` |
| `src/kf_da/daComp/adjoint.py` | Reverse-mode AD for gradients and Hessian-vector products |
| `src/kf_da/daComp/loss_funcs.py` | `MSE_PP` (particle positions), `MSE_Vel` (velocity); measurement masks `t_mask` |
| `src/kf_da/opti/optimization.py` | `BFGS`, `Joint_Opt`, `Loss_and_Deriv_fns`; `ArmijoLineSearch` in `LS_TR.py` |
| `src/kf_da/icParam/Fourier_Param.py` | Fourier mode parametrization with log-sinh preconditioning |
| `src/kf_da/velInit/AI.py` | Attractor-based IC initialization (random sample from attractor snapshots) |
| `src/kf_da/vp_floats/` | C++ variable-precision float library (Pybind11); `vp_py_utils.py` for JAX integration |

### JAX Usage

The solver and adjoint code are JAX-first (JIT-compiled, GPU-capable). Computations operate on spectral (Fourier) coefficients of vorticity. `jax.grad` / `jax.vjp` underpin the adjoint solver; the custom VP float path bypasses JAX's autodiff for precision studies.

### Results Storage

Results are written as **Parquet files** (via pandas + PyArrow/fastparquet), organized hierarchically by seed, Re, and experiment parameters. `case_post_proc.py` and `gPost/global_post_main.py` handle post-processing and Excel export.
