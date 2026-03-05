import os
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from SRC.KS_Integrators import KS_RK4, run_solver
from create_results_dir import create_results_dir
from dataclasses import dataclass
from linear_adj_prec import random_rotation_matrices
import matplotlib.pyplot as plt
import numpy as np
from SRC.plotting_utils import save_svg
import matplotlib as mpl
import pickle
from linear_adj_prec import gen_err_model


lyapunov_L_11 = np.array([
    1.950497e-04,
    -8.794331e-04,
    -1.173343e-01,
    -2.698752e-01,
    -2.691901e-01,
    -5.620848e+00,
    -5.624045e+00,
    -2.028071e+01,
    -2.065392e+01,
    -2.095230e+01,
    -2.106287e+01,
    -2.111412e+01,
    -2.114699e+01,
    -2.117220e+01,
    -2.118891e+01,
    -2.120169e+01,
    -2.121266e+01,
    -2.122533e+01,
    -2.123835e+01,
    -2.124499e+01,
    -2.125057e+01,
    -2.125960e+01,
    -2.126792e+01,
    -2.127218e+01,
    -2.127782e+01,
    -2.128498e+01,
    -2.129426e+01,
    -2.129965e+01,
    -2.130967e+01,
    -2.131230e+01,
    -2.132089e+01,
    -2.133063e+01,
    -2.064285e+01,
    -2.067037e+01,
    -2.069085e+01,
    -2.071010e+01,
    -2.073129e+01,
    -2.075403e+01,
    -2.077623e+01,
    -2.080032e+01,
    -2.081850e+01,
    -2.084156e+01,
    -2.086603e+01,
    -2.088946e+01,
    -2.091322e+01,
    -2.093495e+01,
    -2.096531e+01,
    -2.099296e+01,
    -2.101714e+01,
    -2.105105e+01,
    -2.107896e+01,
    -2.111761e+01,
    -2.115020e+01,
    -2.118912e+01,
    -2.123036e+01,
    -2.127195e+01,
    -2.131927e+01,
    -2.137347e+01,
    -2.143181e+01,
    -2.150830e+01,
    -2.158787e+01,
    -2.170228e+01,
    -2.187474e+01,
    -2.223704e+01
])

lyapunov_L_22 = np.array([
    4.855346e-02,
    2.956982e-04,
   -2.087305e-05,
   -1.920932e-04,
   -3.892356e-03,
   -1.856883e-01,
   -2.564860e-01,
   -2.904235e-01,
   -3.112570e-01,
   -1.952488e+00,
   -1.957451e+00,
   -5.584101e+00,
   -5.584122e+00,
   -1.186628e+01,
   -1.186629e+01,
   -1.995243e+01,
   -2.009206e+01,
   -2.025250e+01,
   -2.040941e+01,
   -2.054534e+01,
   -2.064273e+01,
   -2.071207e+01,
   -2.075697e+01,
   -2.079967e+01,
   -2.082804e+01,
   -2.085130e+01,
   -2.087307e+01,
   -2.089170e+01,
   -2.090833e+01,
   -2.092310e+01,
   -2.093682e+01,
   -2.095017e+01,
   -2.043946e+01,
   -2.045879e+01,
   -2.047479e+01,
   -2.049560e+01,
   -2.051751e+01,
   -2.053413e+01,
   -2.055766e+01,
   -2.057740e+01,
   -2.059647e+01,
   -2.061986e+01,
   -2.064298e+01,
   -2.066450e+01,
   -2.068815e+01,
   -2.071850e+01,
   -2.074332e+01,
   -2.076966e+01,
   -2.079994e+01,
   -2.083018e+01,
   -2.086171e+01,
   -2.089578e+01,
   -2.093111e+01,
   -2.097181e+01,
   -2.101212e+01,
   -2.105726e+01,
   -2.110858e+01,
   -2.115851e+01,
   -2.121865e+01,
   -2.129312e+01,
   -2.137902e+01,
   -2.149062e+01,
   -2.165821e+01,
   -2.201684e+01
])

lyapunov_L_44 = np.array([
    8.337581e-02,
    5.794851e-02,
    3.527755e-02,
    1.355474e-02,
    8.981318e-05,
    1.491907e-04,
    -2.540422e-04,
    -1.636153e-02,
    -5.501936e-02,
    -1.204345e-01,
    -1.858500e-01,
    -2.462053e-01,
    -2.939605e-01,
    -3.325483e-01,
    -3.714066e-01,
    -4.236972e-01,
    -4.981945e-01,
    -6.871605e-01,
    -7.369600e-01,
    -1.914518e+00,
    -1.916628e+00,
    -3.473037e+00,
    -3.473108e+00,
    -5.560041e+00,
    -5.560071e+00,
    -8.306046e+00,
    -8.306056e+00,
    -1.183382e+01,
    -1.183382e+01,
    -1.626520e+01,
    -1.626521e+01,
    -1.965442e+01,
    -1.968816e+01,
    -1.975041e+01,
    -1.980514e+01,
    -1.986132e+01,
    -1.991037e+01,
    -1.995957e+01,
    -2.000508e+01,
    -2.004799e+01,
    -2.008950e+01,
    -2.012693e+01,
    -2.016601e+01,
    -2.020106e+01,
    -2.023799e+01,
    -2.027344e+01,
    -2.030807e+01,
    -2.034186e+01,
    -2.037995e+01,
    -2.041722e+01,
    -2.045698e+01,
    -2.049474e+01,
    -2.053802e+01,
    -2.058300e+01,
    -2.062627e+01,
    -2.067734e+01,
    -2.073084e+01,
    -2.079133e+01,
    -2.085498e+01,
    -2.093089e+01,
    -2.102485e+01,
    -2.114085e+01,
    -2.131588e+01,
    -2.168025e+01
])

lyapunov_L_66 = np.array([
    8.556927e-02,
    6.908166e-02,
    5.527508e-02,
    4.274269e-02,
    3.060744e-02,
    1.870754e-02,
    4.485381e-03,
    2.673822e-04,
    2.120855e-05,
    -2.600998e-04,
    -1.138555e-02,
    -3.103534e-02,
    -5.707341e-02,
    -8.999004e-02,
    -1.327945e-01,
    -1.756035e-01,
    -2.220679e-01,
    -2.594802e-01,
    -2.928949e-01,
    -3.203602e-01,
    -3.459882e-01,
    -3.730918e-01,
    -4.011099e-01,
    -4.318271e-01,
    -4.668787e-01,
    -5.085294e-01,
    -5.528572e-01,
    -1.019352e+00,
    -1.050063e+00,
    -1.871349e+00,
    -1.873410e+00,
    -2.833356e+00,
    -2.834282e+00,
    -4.005843e+00,
    -4.005907e+00,
    -5.446036e+00,
    -5.446016e+00,
    -7.199170e+00,
    -7.199200e+00,
    -9.289412e+00,
    -9.289412e+00,
    -1.175014e+01,
    -1.175014e+01,
    -1.462012e+01,
    -1.462012e+01,
    -1.793438e+01,
    -1.793461e+01,
    -1.983292e+01,
    -1.987923e+01,
    -1.992479e+01,
    -1.997170e+01,
    -2.001991e+01,
    -2.006836e+01,
    -2.011821e+01,
    -2.016962e+01,
    -2.022193e+01,
    -2.027732e+01,
    -2.033281e+01,
    -2.039094e+01,
    -2.045022e+01,
    -2.051322e+01,
    -2.057554e+01,
    -2.064132e+01,
    -2.071250e+01,
    -2.078368e+01,
    -2.086408e+01,
    -2.094806e+01,
    -2.103954e+01,
    -2.114518e+01,
    -2.127954e+01,
    -2.146713e+01,
    -2.185724e+01
])


lyapunov_L_88 = np.array([
    1.246013e-01,
    1.058539e-01,
    9.065441e-02,
    7.807779e-02,
    6.709771e-02,
    5.556247e-02,
    4.515043e-02,
    3.490747e-02,
    2.373449e-02,
    1.278871e-02,
    5.957775e-04,
    4.136248e-04,
    -2.495503e-04,
    -3.875414e-04,
    -1.241272e-02,
    -2.660873e-02,
    -4.241887e-02,
    -5.949412e-02,
    -8.191076e-02,
    -1.051619e-01,
    -1.328110e-01,
    -1.653039e-01,
    -1.984986e-01,
    -2.286312e-01,
    -2.605663e-01,
    -2.891508e-01,
    -3.172976e-01,
    -3.420799e-01,
    -3.669257e-01,
    -3.935991e-01,
    -4.180034e-01,
    -4.469062e-01,
    -4.768080e-01,
    -5.088469e-01,
    -5.505344e-01,
    -5.979803e-01,
    -6.606651e-01,
    -8.961318e-01,
    -9.482407e-01,
    -1.566706e+00,
    -1.578834e+00,
    -2.321053e+00,
    -2.323843e+00,
    -3.156700e+00,
    -3.157302e+00,
    -4.075031e+00,
    -4.075321e+00,
    -5.686100e+00,
    -5.686100e+00,
    -6.965725e+00,
    -6.965725e+00,
    -8.430123e+00,
    -8.430123e+00,
    -1.009520e+01,
    -1.009520e+01,
    -1.197748e+01,
    -1.197748e+01,
    -1.409413e+01,
    -1.409413e+01,
    -1.646291e+01,
    -1.646291e+01,
    -1.910095e+01,
    -1.913886e+01,
    -1.956044e+01,
    -1.964403e+01,
    -1.972648e+01,
    -1.981137e+01,
    -1.990480e+01,
    -2.001091e+01,
    -2.014044e+01,
    -2.032424e+01,
    -2.071515e+01
])

@dataclass
class Data_Options:
    N_DOF: int
    L: int
    T: float
    time_step_skip: float
    DT: float
    min_sampling_time: float
    solver_type: str

root = os.path.join(create_results_dir(), "KSE")

def load_data(data_options: Data_Options):
    min_sampling_idx = int(data_options.min_sampling_time/data_options.DT)
    skip = int(data_options.time_step_skip/data_options.DT)

    path = os.path.join("initial_trjs", f"domain_{data_options.L}")
    path = os.path.join(root, path, f"N_DOF={data_options.N_DOF}_DT={data_options.DT}_T={data_options.T}_solver={data_options.solver_type.name}", f"N_DOF={data_options.N_DOF}_DT={data_options.DT}_T={data_options.T}_solver={data_options.solver_type.name}.npy")
    data = np.load(path)[min_sampling_idx::skip, :]
    return data

def generate_init_trjs():
    trj_cases = [{"domain_size": 88, "N_DOF": 72, "DT": .05, "T": 1e4, "jax_seed": 1, "Solver_type": KS_RK4}]
    initial_trjs_dir = "initial_trjs"

    # Create "initial_trjs" folder if it doesn't exist
    os.makedirs(initial_trjs_dir, exist_ok=True)

    for trj_case in trj_cases:
        # Extract parameters
        DOMAIN_SIZE = trj_case["domain_size"]
        N_DOF = trj_case["N_DOF"]
        DT = trj_case["DT"]
        T = trj_case["T"]
        jax_seed = trj_case["jax_seed"]
        Solver_type = trj_case["Solver_type"]

        # Generate trajectory
        #key = jax.random.PRNGKey(jax_seed)
        #u_0 = jax.random.uniform(key, shape=(N_DOF,), minval=-1.0, maxval=1.0)
        mesh = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)
        u_0 = jnp.sin(2 * jnp.pi * mesh / DOMAIN_SIZE)

        time_steps = int(T / DT)
        solver = jax.jit(Solver_type(L=DOMAIN_SIZE, N=N_DOF, dt=DT))
        trj = run_solver(solver, u_0, time_steps)

        #fig = Solver_Plotting_Utils.plot_trj(trj, DT, DOMAIN_SIZE)
        
        # Create a subfolder based on domain size
        domain_dir = os.path.join(initial_trjs_dir, f"domain_{DOMAIN_SIZE}")


        # Define the filename based on the case parameters
        name = f"N_DOF={N_DOF}_DT={DT}_T={T}_solver={Solver_type.name}"
        folder_path = os.path.join(root, domain_dir, name)
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, name + ".npy")
        fig_path = os.path.join(folder_path, "plot.png")

        # Save the trajectory as a NumPy array
        np.save(file_path, np.array(trj))  # Save as a NumPy file
        #fig.savefig(fig_path)

        print(f"Saved trajectory: {file_path} with shape {trj.shape}")

import numpy as np

def eta_piecewise_linear(
    tau_pts,
    a,
    r,
    tau_0,
    p,
    lambdas,
    clip_exp=True,
    clip_val=700.0,
):
    tau_pts = np.asarray(tau_pts, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if r < 1:
        raise ValueError("r must be >= 1")
    if lambdas.shape[0] < r:
        raise ValueError(f"Need at least r={r} lambdas, got {lambdas.shape[0]}")

    lam1 = float(lambdas[0])
    lam = lambdas[:r]

    def _exp(x):
        if not clip_exp:
            return np.exp(x)
        return np.exp(np.clip(x, -clip_val, clip_val))


    const = a * (2.0 ** (-p)) * _exp(-lam1)

    if r == 1:
        # sum is empty
        def A(tau):
            return (1.0 / r) * tau

        def Aprime(tau):
            return (1.0 / r) * 1.0
    else:
        deltas = lam[1:] - lam1  # (r-1,)
        denom = 1.0 - _exp(2.0 * deltas)  # (r-1,)

        def A(tau):
            tau = np.asarray(tau, dtype=float)
            num = 1.0 - _exp(2.0 * deltas[None, :] * tau[..., None])  # (..., r-1)
            frac = num / denom[None, :]                                # (..., r-1)
            return (1.0 / r) * (tau + np.sum(frac, axis=-1))

        def Aprime(tau):
            tau = np.asarray(tau, dtype=float)
            exp_term = _exp(2.0 * deltas[None, :] * tau[..., None])     # (..., r-1)
            frac_prime = -(2.0 * deltas[None, :] * exp_term) / denom[None, :]
            return (1.0 / r) * (1.0 + np.sum(frac_prime, axis=-1))

    def eta1(tau):
        At = A(tau)
        # Guard against tiny negative due to roundoff
        At = np.maximum(At, 0.0)
        return const * np.sqrt(At)

    def eta1prime(tau):
        At = A(tau)
        At = np.maximum(At, 0.0)
        Apt = Aprime(tau)
        # If At==0, derivative would blow up; return 0 in that degenerate case
        denom_sqrt = np.sqrt(At)
        return np.where(
            denom_sqrt > 0.0,
            const * (0.5 * Apt / denom_sqrt),
            0.0,
        )

    # ----- piecewise evaluation -----
    eta0 = eta1(tau_0)
    slope0 = eta1prime(tau_0)

    eta = np.empty_like(tau_pts, dtype=float)
    mask = tau_pts <= tau_0
    eta[mask] = eta1(tau_pts[mask])
    eta[~mask] = eta0 + slope0 * (tau_pts[~mask] - tau_0)
    return eta

def lyp_var():
    save_folder = os.path.join(root, "ftle")
    os.makedirs(save_folder, exist_ok=True)
    IC_seed_key = jax.random.PRNGKey(0)
    num_IC = 100

    KS_opts = Data_Options(
        N_DOF=64, L=22, T=1e4, DT=0.1,
        time_step_skip=1,
        min_sampling_time=1000,
        solver_type=KS_RK4,
    )

    # ---- times to evaluate ----
    T_list = np.linspace(1, 200, 20)

    attractor_snapshots = load_data(KS_opts)
    step_fn = KS_opts.solver_type(
        L=KS_opts.L, N=KS_opts.N_DOF, dt=KS_opts.DT
    )

    # sample ICs
    IC_idx_list = jax.random.randint(
        IC_seed_key,
        (num_IC,),
        0,
        attractor_snapshots.shape[0],
    )
    u0_batch = attractor_snapshots[IC_idx_list, :]  # (num_IC, 64)

    mean_curve_dict = {}
    var_curve_dict = {}

    idx_2_plot = [0, 2, 4, 8]
    cov_time_indices = [1, 19]
    mean_LLE_dict = {}

    for ti, T in enumerate(T_list):
        N = int(T / KS_opts.DT)

        # ----- flow map (depends on N) -----
        def fv_fn(u0):
            trj = run_solver(step_fn, u0, N)
            return trj[-1]

        jac_fn = jax.jacobian(fv_fn)

        # ----- finite-time Lyapunov spectrum -----
        def ftle(u0):
            J = jac_fn(u0)
            s = jnp.linalg.svd(J, compute_uv=False)

            return jnp.log(s) / N
        
        ftle_batch = jax.jit(jax.vmap(ftle))(u0_batch)

        mean_spec = jnp.mean(ftle_batch, axis=0)
        var_spec = jnp.var(ftle_batch, axis=0)

        mean_LLE_dict[T] = mean_spec


        for idx in idx_2_plot:
            if idx not in mean_curve_dict:
                mean_curve_dict[idx] = [float(mean_spec[idx])]
            else:
                mean_curve_dict[idx].append(float(mean_spec[idx]))
            if idx not in var_curve_dict:
                var_curve_dict[idx] = [float(var_spec[idx])]
            else:
                var_curve_dict[idx].append(float(var_spec[idx]))


        print(f"T={T:.2f}: mean={float(mean_spec[0]):.6f}, var={float(var_spec[0]):.6e}")

        if ti in cov_time_indices:
            X = ftle_batch - mean_spec[None, :]
            cov = (X.T @ X) / (num_IC - 1)

            cov_np = np.array(cov)
            cov_abs = np.abs(cov_np)

            import matplotlib.colors as colors

            fig = plt.figure(figsize=(6.5, 5.5))

            im = plt.imshow(
                cov_abs,
                aspect="auto",
                norm=colors.LogNorm(
                    vmin=cov_abs[cov_abs > 0].min(),
                    vmax=cov_abs.max(),
                ),
            )

            plt.colorbar(im, label="|Covariance| (log scale)")
            plt.title(f"FTLE covariance matrix (T={float(T):.2f})")
            plt.xlabel("Exponent index")
            plt.ylabel("Exponent index")
            plt.tight_layout()
            save_svg(mpl, fig, os.path.join(save_folder, f"cov_matrix_T={float(T):.2f}.svg"))
            plt.close(fig)

    # ==========================
    # Plot mean/var vs time
    # ==========================
    fig = plt.figure(figsize=(7, 5))
    for idx in idx_2_plot:
        mean_curve = mean_curve_dict[idx]
        plt.plot(T_list, mean_curve, label=f"idx={idx}")
    plt.xlabel("Time horizon T")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(save_folder, "ftle_v_time.svg"))

    fig = plt.figure(figsize=(7, 5))
    for idx in idx_2_plot:
        var_curve = var_curve_dict[idx]
        plt.plot(T_list, var_curve, label=f"idx={idx}")
    plt.xlabel("Time horizon T")
    plt.ylabel("Value")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(save_folder, "ftle_var_v_time.svg"))

    with open(os.path.join(save_folder, "mean_LLE_dict.pkl"), "wb") as f:
        pickle.dump(mean_LLE_dict, f)

def plot_res(save_folder, mbits, NDOF, dt, lyapunov):
    r_N = 1

    fit_start_idx= int(4/dt)

    # ---- load data ----
    mean_err_t = np.load(os.path.join(save_folder, "mean_err_t.npy"))
    N = mean_err_t.shape[0]

    t = np.linspace(0, N * dt, N)
    # ---- plot data ----
    fig = plt.figure()
    plt.plot(t, mean_err_t, label="mean error")

    # =====================================================
    # Linear fit using only data after fit_start_idx
    # =====================================================
    if False:
        t_fit = t[fit_start_idx:]
        err_fit = mean_err_t[fit_start_idx:]

        # fit line: err ≈ a + b t
        b, a = np.polyfit(t_fit, err_fit, 1)

        fit_line = a + b * t

        plt.plot(
            t,
            fit_line,
            linewidth=2,
            label=f"linear fit: {a:.3e} + {b:.3e} t",
        )

        print(f"Linear fit (t ≥ {t_fit[0]:.3f})")
        print(f"  slope     = {b}")
        print(f"  intercept = {a}")

    # =====================================================
    # model curve
    # =====================================================
    #E_alpha_ratio = .0125
    #r = 8
    E_alpha_ratio = 1
    r = NDOF
    if True:
        r_N = 1
        tau, baseline_eta = gen_err_model(
            N,
            mbits,
            lyapunov * dt,
            r,
            r_N,
            E_abs_alpha1_inv=E_alpha_ratio,
            E_alpha_ratio=0,
            clip_exp=False,
        )
    if True:
        a = E_alpha_ratio
        tau_0 = int(1.0/dt)
        eta = eta_piecewise_linear(
            tau,
            a,
            r,
            tau_0,
            mbits,
            lyapunov * dt,
            clip_exp=True,
            clip_val=700.0,
        )
        plt.plot(tau*dt, eta, label="model2")

    plt.plot(tau*dt, baseline_eta, label="model1")
    #plt.ylim(0, .0007)
    # ---- formatting ----
    plt.xlabel("time")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(save_folder, "norm_error.svg"))

def rand_rot_noise():
    # ------------------------
    # Config
    # ------------------------
    num_IC = 50
    num_E = 50
    num_lam = 50
    mbits = 8
    compute_err = True
    det_lam_N = False
    error_type = "rand_rot"
    #error_type = "uni_round"

    KS_opts = Data_Options(
        N_DOF=64, L=22, T=1e4, DT=0.1, time_step_skip=1,
        min_sampling_time=1000, solver_type=KS_RK4
    )
    attractor_snapshots = load_data(KS_opts)              # (T, n)
    mean_mag = jnp.mean(jnp.abs(attractor_snapshots))
    n = attractor_snapshots.shape[1]

    T_horizon = 20
    N = int(T_horizon / KS_opts.DT)

    # One-step solver and Jacobian of one step
    step_fn = KS_opts.solver_type(L=KS_opts.L, N=KS_opts.N_DOF, dt=KS_opts.DT)
    step_fn = jax.jit(step_fn)
    jac_fn = jax.jit(jax.jacobian(step_fn))               # u -> (n, n)

    if det_lam_N:
        num_lam = 1
    save_folder = os.path.join(root, f"{error_type}", f"L={KS_opts.L}_T={T_horizon}_mbits={mbits}_num_IC={num_IC}_num_E={num_E}_num_lam={num_lam}")
    if compute_err:
        N = int(T_horizon / KS_opts.DT)

        # ------------------------
        # Build Gaussian sampler for lam_N ~ N(0, Sigma)
        # ------------------------
        mean = jnp.mean(attractor_snapshots, axis=0)          # (n,)
        X = attractor_snapshots - mean
        Sigma = (X.T @ X) / attractor_snapshots.shape[0]      # (n, n)

        eps = 1e-8
        L = jnp.linalg.cholesky(Sigma + eps * jnp.eye(n))      # (n, n)

        def sample_lam_batch(key):
            z = jax.random.normal(key, (num_lam, n))
            return z @ L.T    
            #return z                      

        # ------------------------
        # Adjoint sweep (scan)
        # ------------------------
        def adjoint_step_rand_rot(carry, inp):
            lam, lam_e = carry
            u, E_i = inp               

            JT = jac_fn(u).T       
            lam = JT @ lam
            lam_e = JT @ lam_e + (E_i @ lam_e)

            return (lam, lam_e), (lam, lam_e)
        def adjoint_step_uni_round(carry, inp):
            lam, lam_e = carry
            u, e_i = inp               

            JT = jac_fn(u).T
            E_i = jac_fn(u+e_i).T - JT
            lam = JT @ lam
            lam_e = JT @ lam_e + E_i @ lam_e

            return (lam, lam_e), (lam, lam_e)

        if error_type == "rand_rot":
            adjoint_step = adjoint_step_rand_rot
        if error_type == "uni_round":
            adjoint_step = adjoint_step_uni_round

        # JIT a full sweep over one (trj, E_trj) pair and one lam_N
        @jax.jit
        def sweep_one(trj, E_trj, lam_N):
            (lamT, lam_eT), (lam_trj, lam_e_trj) = jax.lax.scan(
                adjoint_step,
                (lam_N, lam_N),
                (trj, E_trj),
            )
            # Relative error over time: (N,)
            lam_trj = jnp.vstack([lam_N, lam_trj])
            lam_e_trj = jnp.vstack([lam_N, lam_e_trj])
            rel_err = jnp.linalg.norm(lam_trj - lam_e_trj, axis=1) / jnp.linalg.norm(lam_trj, axis=1)
            return rel_err

        sweep_over_lam = jax.jit(jax.vmap(sweep_one, in_axes=(None, None, 0)))
        sweep_over_E = jax.jit(jax.vmap(sweep_over_lam, in_axes=(None, 0, None)))


        key = jax.random.PRNGKey(0)
        norm_error_arr = []
        if True:
            def fv_fn(u):
                trj = run_solver(step_fn, u, N)
                return trj[-1]
            fv_jac_fn = jax.jit(jax.jacobian(fv_fn))

        for k1 in range(num_IC):
            key, k_ic, k_E, k_lam = jax.random.split(key, 4)

            # Random IC from attractor
            rand_idx = jax.random.randint(k_ic, (), 0, attractor_snapshots.shape[0])
            u0 = attractor_snapshots[rand_idx, :]

            trj = run_solver(step_fn, u0, N - 1)

            E_keys = jax.random.split(k_E, num_E)
            if error_type == "rand_rot":
                E_trj_batch = jax.vmap(
                    lambda kk: random_rotation_matrices(kk, N, n, dtype=u0.dtype) * (2.0 ** (-mbits))
                )(E_keys)
            elif error_type == "uni_round":
                E_trj_batch = jax.vmap(
                    lambda kk: jax.random.uniform(
                        kk,
                        shape=(N, n),
                        minval=-2**(-mbits),
                        maxval=2**(-mbits),
                    ) * mean_mag
                )(E_keys)

            if det_lam_N:
                jac = jac_fn(u0)
                U, S, Vh = jnp.linalg.svd(jac)
                lam_N = U[:, 0]
                lam_N_samples = lam_N.reshape(1, -1)

            else:
                lam_N_samples = sample_lam_batch(k_lam)
                print(lam_N_samples.shape)

            errs = sweep_over_E(trj, E_trj_batch, lam_N_samples)
            norm_error_arr.append(errs)

        norm_error_arr = jnp.stack(norm_error_arr, axis=0)
        
        mean_err_t = norm_error_arr.mean(axis=(0, 1, 2))
        os.makedirs(save_folder, exist_ok=True)
        np.save(os.path.join(save_folder, "mean_err_t.npy"), mean_err_t)
    if KS_opts.L == 11:
        lyap = lyapunov_L_11
    elif KS_opts.L == 22:
        lyap = lyapunov_L_22
    elif KS_opts.L == 44:
        lyap = lyapunov_L_44
    elif KS_opts.L == 66:
        lyap = lyapunov_L_66
    elif KS_opts.L == 88:
        lyap = lyapunov_L_88

    plot_res(save_folder, mbits, KS_opts.N_DOF, KS_opts.DT, lyap)





if __name__ == "__main__":
    rand_rot_noise()