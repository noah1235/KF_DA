import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import config
from sympy import gamma

from create_results_dir import create_results_dir
from SRC.DA_Comp.configs import *
from SRC.DA_Comp.loss_funcs import *
from SRC.DA_Comp.adjoint import get_loss_grad_vp_fn, get_loss_grad_fn, get_forced_adj_shooting_vp, get_forced_adj_shooting
from SRC.parameterization.Fourier_Param import Fourier_Param
from SRC.plotting_utils import save_svg
from SRC.Solver.IC_gen import init_particles_vector
from SRC.Solver.solver import KF_Stepper, KF_TP_Stepper, create_omega_part_gen_fn, Omega_Integrator
from SRC.utils import load_data
from SRC.vp_floats.vp_py_utils import choose_exponent_format, float_pos_range
from scipy.optimize import curve_fit
import matplotlib as mpl

config.update("jax_enable_x64", True)

def random_rotation_matrices(key, N, d, dtype=jnp.float32):

    keys = jax.random.split(key, N)

    def single(k):
        M = jax.random.normal(k, (d, d), dtype=dtype)
        Q, R = jnp.linalg.qr(M)

        # Fix QR sign ambiguity → Haar distribution
        s = jnp.sign(jnp.diag(R))
        s = jnp.where(s == 0, 1.0, s)
        Q = Q * s[None, :]
        return Q

    return jax.vmap(jax.jit(single))(keys)

def random_r_subspace_unit_vector(key, d, r, dtype=jnp.float64):
    # random Gaussian coefficients in r-dim subspace
    coeffs = jax.random.normal(key, (r,), dtype=dtype)

    # normalize → uniform on S^{r-1}
    coeffs = coeffs / jnp.linalg.norm(coeffs)

    # embed into R^d
    lam_N = jnp.zeros((d,), dtype=dtype).at[:r].set(coeffs)

    return lam_N

def plot_model_comp():
    root = os.path.join(create_results_dir(), "vpfloats", "model_comp")
    os.makedirs(root, exist_ok=True)

    # ----- spectrum -----
    ly_spec = jnp.linspace(0.3, -1.0, 100)

    dt = 0.1
    T = 50
    mbits = 8
    N = int(T / dt)

    lambdas = ly_spec * dt

    # ----- baseline model -----
    E_abs_alpha1_inv = 1.0
    tau_vals, f = expr_vs_tau_array(N, mbits, lambdas)
    f *= E_abs_alpha1_inv

    tau_plot = tau_vals * dt

    # ----- parameter sweeps -----
    rN_list = [2, 4]
    E_ratio_list = [1e-4, 1e-3, 1e-2]
    E_ratio_list = [0]

    fig = plt.figure(figsize=(8, 6))

    # baseline curve
    plt.plot(
        tau_plot,
        f,
        label="Baseline (r_N = 1)",
    )

    # loop over combinations
    for r_N in rN_list:
        for E_alpha_ratio in E_ratio_list:

            tau, g_k = bound_expr_tau_array(
                N,
                mbits,
                lambdas,
                r=lambdas.shape[0],
                r_N=r_N,
                E_abs_alpha1_inv=E_abs_alpha1_inv,
                E_alpha_ratio=E_alpha_ratio,
                clip_exp=True,
            )

            plt.plot(
                tau * dt,
                g_k,
                label=f"r_N={r_N}, E_ratio={E_alpha_ratio:.0e}",
            )

    plt.xlabel(r"$\tau$")
    plt.ylabel("Model value")
    plt.title("Linear Adjoint Error Model Comparison")
    plt.legend(fontsize=8)
    save_svg(mpl, fig, os.path.join(root, "param_sweep.svg"))
    return
    save_path = os.path.join(root, "model_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def linear_adj_prec():
    root = os.path.join(create_results_dir(), "vpfloats", "lin_adj_error")
    os.makedirs(root, exist_ok=True)

    num_E = 200
    num_lam_N = 1
    #gamma = .05
    gamma = 5e-2
    ly_spec = jnp.linspace(0.3, -1, 100)
    det_lam_N = True
    lam_N_sup = 4

    dt = 0.1
    A = jnp.exp(ly_spec * dt)  # (d,)

    T = 6
    mbits = 8
    N = int(T / dt)
    d = int(A.shape[0])

    E_abs_alpha1_inv = 1.0
    _, f = expr_vs_tau_array(N, mbits, ly_spec * dt)
    f *= E_abs_alpha1_inv

    # --------- core compute (same logic) ---------
    @jax.jit
    def _mean_norm_error(key):
        # lam_N samples
        if det_lam_N:
            lam0 = jnp.eye(d, dtype=A.dtype)[:, 0]
            lam_N_batch = lam0[None, :]  # (L=1, d)
            num_lam_eff = 1
        else:
            num_lam_eff = num_lam_N
            key_lam, key = jax.random.split(key)
            keys_lam = jax.random.split(key_lam, num_lam_eff)

            def sample_lamN(k):
                return random_r_subspace_unit_vector(k, d, lam_N_sup, dtype=jnp.float64)

            lam_N_batch = jax.vmap(sample_lamN)(keys_lam)  # (L, d)

        # E trajectories
        key_E, key = jax.random.split(key)
        keys_E = jax.random.split(key_E, num_E)

        def make_E_trj(k):
            return random_rotation_matrices(k, N, d, dtype=A.dtype) * (2.0 ** (-mbits))

        E_trj_batch = jax.vmap(make_E_trj)(keys_E)  # (E, N, d, d)

        # per-run lyap noise keys: one per (lam_N, E_trj) pair
        key_runs, _ = jax.random.split(key)
        keys_run = jax.random.split(key_runs, num_lam_eff * num_E).reshape(num_lam_eff, num_E, 2)

        # scan step (B fixed per run)
        def step(carry, E_i):
            lam, lam_e, B = carry
            lam_next = B * lam
            lam_e_next = B * lam_e + E_i @ lam_e
            return (lam_next, lam_e_next, B), (lam_next, lam_e_next)

        def run_one(lam_N, E_trj, key_run):
            # ---- per-run Lyapunov noise: mean 0, var gamma ----
            ly_noise = jnp.where(
                gamma > 0.0,
                jax.random.normal(key_run, (d,), dtype=ly_spec.dtype) * jnp.sqrt(gamma),
                jnp.zeros((d,), dtype=ly_spec.dtype),
            )

            # fixed growth operator for entire trajectory
            B = jnp.exp((ly_spec + ly_noise) * dt)

            (_, _, _), (lam_trj, lam_e_trj) = lax.scan(step, (lam_N, lam_N, B), E_trj)

            lam_trj = jnp.vstack([lam_N[None, :], lam_trj])
            lam_e_trj = jnp.vstack([lam_N[None, :], lam_e_trj])

            norm_error = jnp.linalg.norm(lam_trj - lam_e_trj, axis=1) / jnp.linalg.norm(lam_trj, axis=1)
            return norm_error  # (N,)

        # run all combinations (L, E, N), then mean over L and E
        errs = jax.vmap(
            lambda lam_N, ks: jax.vmap(lambda E_trj, k: run_one(lam_N, E_trj, k))(E_trj_batch, ks)
        )(lam_N_batch, keys_run)

        return errs.mean(axis=(0, 1))  # (N,)

    mean_norm_error = _mean_norm_error(jax.random.PRNGKey(0))

    # --------- curve fit + plot ---------
    y = np.array(mean_norm_error)
    x = np.arange(1, len(y) + 1) * dt

    def model(x, a, g, b):
        return a * x ** (-g) + b

    b0 = -0.8
    a0 = -0.8
    g0 = 0.023
    p0 = [a0, g0, b0]

    params, covariance = curve_fit(model, x, y, p0=p0, maxfev=10000)
    a_fit, g_fit, b_fit = params

    print("a =", a_fit)
    print("g =", g_fit)
    print("b =", b_fit)

    y_fit = model(x, a_fit, g_fit, b_fit)

    fig = plt.figure()
    plt.plot(x, y, label="data")
    plt.plot(x, np.asarray(f), label="model")
    plt.legend()
    save_svg(mpl, fig, os.path.join(root, "avg_norm_error.svg"))

def gen_err_model(
    N,
    p,
    lambdas,
    r,
    r_N,
    tau_0=None,
    q=None,
    a=None,
    E_abs_alpha1_inv=1.0,
    E_alpha_ratio=None,
    clip_exp=True,
    tau_scale=1.0,     # optional: use (tau_scale*tau)**a_gamma for nondimensionalization
    tau_eps=1e-12,     # avoid 0**negative
):
    lambdas = np.asarray(lambdas, dtype=float)
    d = len(lambdas)
    lambda1 = lambdas[0]

    lam_l = lambdas[:r]
    lam_i = lambdas[:r_N]
    rr_N = r * r_N

    # --- normalize E_alpha_ratio to array length d ---
    if E_alpha_ratio is None:
        E_ratio = np.ones(d, dtype=float)
    else:
        arr = np.asarray(E_alpha_ratio, dtype=float)
        if arr.ndim == 0:
            E_ratio = np.full(d, float(arr))
        else:
            if len(arr) != d:
                raise ValueError(f"E_alpha_ratio must be scalar or length d={d}, got {len(arr)}")
            E_ratio = arr

    tau_vals = np.arange(N + 1, dtype=float)
    y = np.zeros(N + 1, dtype=float)

    def _exp(x):
        if not clip_exp:
            return np.exp(x)
        return np.exp(np.clip(x, -700.0, 700.0))

    # -------- precompute braces (depends only on tau) --------
    # braces(tau) = E_abs_alpha1_inv*exp(-lambda1) + sum_{idx=1}^{d-1} E_ratio[idx]*exp(2*tau*lambdas[idx] - 2*tau*lambda1 - lambda1)
    base_brace = E_abs_alpha1_inv * _exp(-lambda1)
    if d > 1:
        idx_terms = lambdas[1:] - lambda1  # (d-1,)
        # (N+1, d-1): 2*tau*(lambda_i-lambda1)
        brace_mat = _exp(2.0 * tau_vals[:, None] * idx_terms[None, :] - lambda1)
        braces_all = base_brace + (brace_mat * E_ratio[1:][None, :]).sum(axis=1)
    else:
        braces_all = np.full(N + 1, base_brace, dtype=float)

    # -------- precompute gamma_factor (depends only on tau) --------
    tau_eff_all = np.maximum(tau_scale * tau_vals, tau_eps)
    if tau_0 is None or q is None or a is None:
        gamma_all = np.ones(N + 1, dtype=float)
    else:
        gamma_all = (1.0 + (tau_eff_all / tau_0) ** q) ** (a / q)

    # -------- compute y(tau) --------
    for t_idx, tau in enumerate(tau_vals):
        k = N - int(tau)

        prefactor_tau = (2.0 ** (-p)) * braces_all[t_idx] * gamma_all[t_idx]

        # j runs k+1,...,N  => let t = j-1-k runs 0,...,N-k-1
        M = N - k  # number of terms in j-sum
        if M <= 0:
            y[t_idx] = 0.0
            continue

        t = np.arange(M, dtype=float)  # t = j-1-k

        # outer = exp(-2*lambda1*(N + j - 1 - 2k))
        # with j = k+1+t => N + j - 1 - 2k = (N - k) + t
        outer = _exp(-2.0 * lambda1 * ((N - k) + t))  # (M,)

        # inner(t) = sum_l sum_i exp(2*lam_l[l]*t + 2*lam_i[i]*(N-k))
        # factorization: (sum_l exp(2*lam_l[l]*t)) * (sum_i exp(2*lam_i[i]*(N-k)))
        sum_l = _exp(2.0 * lam_l[:, None] * t[None, :]).sum(axis=0)          # (M,)
        sum_i = _exp(2.0 * lam_i * (N - k)).sum()                            # scalar
        inner = sum_l * sum_i                                                # (M,)

        S = np.sum(outer * inner)

        y[t_idx] = prefactor_tau * np.sqrt(S / rr_N)

    return tau_vals, y

def expr_vs_tau_array(N, p, lambdas):
    """
    Compute

        2^{-p} e^{-lambda_1} sqrt(
            (1/d) * ( tau
            + sum_{i=2}^d (1 - exp((lambda_i-lambda_1)*tau))
                            /(1 - exp(lambda_i-lambda_1)) )
        )

    using simple nested loops.
    """

    lambdas = np.asarray(lambdas, dtype=float)
    lambda1 = lambdas[0]
    d = len(lambdas)

    tau_vals = np.arange(N + 1, dtype=float)
    y = np.zeros(N + 1)

    prefactor = (2.0 ** (-p)) * np.exp(-lambda1)

    for t_idx, tau in enumerate(tau_vals):

        # start with i = 1 contribution
        total = tau

        # sum i = 2..d
        for i in range(1, d):
            delta = lambdas[i] - lambda1

            term = (1.0 - np.exp(2*delta * tau)) / (1.0 - np.exp(2*delta))

            total += term

        y[t_idx] = prefactor * np.sqrt((1.0 / d) * total)

    return tau_vals, y

if __name__ == "__main__":
    linear_adj_prec()
