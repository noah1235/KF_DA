import jax
import jax.numpy as jnp
from jax import lax
import functools
from SRC.Solver.KF_intergrators import create_trj_generator_vpfloat, create_trj_generator
from SRC.vp_floats.vp_py_utils import choose_exponent_format, calc_output_shape
import vpfloat
import numpy as np
from jax import ShapeDtypeStruct
# ---------- Small utilities ----------

def symmetric_error(A):
    """Host-only diagnostic for symmetry."""
    sym_err = jnp.linalg.norm(A - A.conj().T) / jnp.linalg.norm(A)
    print(f"sym error: {sym_err}")


def JT_times_vec(fn, x, lam_out):
    """
    Compute (J_fn(x))^T @ lam_out via VJP.
    """
    _, vjp = jax.vjp(fn, x)
    (res,) = vjp(lam_out)
    return res


def JT_times_matrix(f, u, S):
    """
    Compute (J_f(u))^H @ S for S with shape (n,) or (n, p).

    f : R^n -> R^n (or C^n -> C^n)
    u : (n,)
    S : (n,) or (n, p)

    Returns same shape as S.
    """
    _, vjp_fun = jax.vjp(f, u)

    def _col(c):
        (res,) = vjp_fun(c)
        return res

    # If S is a vector, just call once.
    if S.ndim == 1:
        return _col(S)

    # Otherwise treat S as (n, p) and apply column-wise.
    return jax.vmap(_col, in_axes=1, out_axes=1)(S)


# ---------- First-order adjoint stepper ----------

class Adjoint_Stepper_1:
    """
    One backward adjoint step:
        λ_n = J_f(u_n)^T λ_{n+1} + ∂g/∂u_n
    and accumulate per-step loss g.
    """
    def __init__(self, part_idx, crit, target_trj, f, vel_part_trans):
        self.crit = crit
        self.target_trj = target_trj
        self.vel_part_trans = vel_part_trans
        self.f = f

        # value_and_grad w.r.t. X (argnums=0)
        self.loss_dg__du_fn = jax.value_and_grad(self.g, argnums=0)
        self.part_idx = part_idx

        # Assuming this exists on f
        self.upsample_factor = f.step.rhs.r

    def g(self, X, i):
        target_snapshot = self.target_trj[i]

        particles, U_flat = self.vel_part_trans.split_part_and_vel(X)
        xp, yp, up, vp = self.vel_part_trans.get_part_pos_and_vel(particles)

        trg_particles, trg_U_flat = self.vel_part_trans.split_part_and_vel(
            target_snapshot
        )
        trg_xp, trg_yp, trg_up, trg_vp = self.vel_part_trans.get_part_pos_and_vel(
            trg_particles
        )

        return self.crit.g(
            xp,
            yp,
            trg_xp,
            trg_yp,
            U_flat,
            trg_U_flat,
            self.upsample_factor,
            i,
        )

    def df__du_v_fn(self, u_n, v):
        """
        Apply J_f(u_n)^T to v via VJP.
        """
        _, vjp_fun = jax.vjp(self.f, u_n)
        return vjp_fun(v)[0]

    def __call__(self, lam_n_1, u_n, i):
        """
        Perform one adjoint step at time index i.

        Inputs:
          lam_n_1: λ_{n+1}
          u_n:     state at time n
          i:       time index (int)

        Returns:
          lam_n, g_val
        """
        g_val, dg__du = self.loss_dg__du_fn(u_n, i)
        lam_n = self.df__du_v_fn(u_n, lam_n_1) + dg__du
        return lam_n, g_val


# ---------- Second-order (Hessian) adjoint stepper ----------

class Adjoint_Stepper_2:
    """
    Second-order adjoint stepper.

    Given:
      λ_{n+1}, dλ_{n+1}, u_n, du_n
    compute:
      λ_n, dλ_n, and add current loss contribution.
    """

    def __init__(self, adj_step_1: Adjoint_Stepper_1):
        self.adj_step_1 = adj_step_1
        self.part_idx = adj_step_1.part_idx
        self.vel_part_trans = adj_step_1.vel_part_trans

    def g_Hess_V(self, U, i, V):
        """
        Compute H_g(U)[vel, vel] @ V without explicitly forming the Hessian.

        U : (n,) full state [part, vel]
        i : time index (int)
        V : (n_vel, k) or (n_vel,) directions in the velocity subspace

        Returns: same shape as V (HVPs in the vel block).
        """
        # Split full state into non-vel "part" and "vel" block
        part, vel = self.vel_part_trans.split_part_and_vel(U)

        # Define scalar g as a function of vel only, with part and i closed over
        def gU_vel(vel_):
            U_full = jnp.concatenate([part, vel_])
            return self.adj_step_1.g(U_full, i)

        # Gradient wrt vel only
        grad_g = jax.grad(gU_vel)  # vel_ -> grad wrt vel_

        # Linearize grad_g at the current vel: lin : R^{n_vel} -> R^{n_vel}
        _, lin = jax.linearize(grad_g, vel)

        # Apply to one or many directions
        if V.ndim == 1:
            return lin(V)  # (n_vel,)
        else:
            # V: (n_vel, k) → apply column-wise
            return jax.vmap(lin, in_axes=1, out_axes=1)(V)  # (n_vel, k)

    def lambda_Hf_mat(self, f, vel, lam, V):
        """
        Compute columns w_j = (∂^2 f / ∂u^2)^T [lam] @ v_j, without forming f''.

        V: (n, k) or (n,)
        Returns: same shape as V
        """
        def pullback(vel_):
            _, vjp = jax.vjp(f, vel_)
            (g,) = vjp(lam)  # g(u) = J_f(u)^T lam
            return g

        # Linearize pullback once
        _, lin = jax.linearize(pullback, vel)

        if V.ndim == 1:
            return lin(V)
        return jax.vmap(lin, in_axes=1, out_axes=1)(V)

    def __call__(self, lam_n_1, dlam_n_1, u_n, d_u_n, i):
        """
        Perform one second-order adjoint step.

        Inputs:
          lam_n_1:  λ_{n+1}
          dlam_n_1: dλ_{n+1}
          u_n:      u_n
          d_u_n:    du_n (sensitivities)
          i:        time index

        Returns:
          lam_n, dlam_n, g_val
        """
        part, vel = self.vel_part_trans.split_part_and_vel(u_n)
        def f(vel_):
            return self.adj_step_1.f(jnp.concat([part, vel_]))
            

        loss_term   = self.g_Hess_V(u_n, i, d_u_n)
        solver_2_term = self.lambda_Hf_mat(f, vel, lam_n_1, d_u_n)

        def f_trunc(vel_):
            return self.vel_part_trans.split_part_and_vel(f(vel_))[1]
        dlam_term   = JT_times_matrix(f_trunc, vel, dlam_n_1)

        dlam_n = dlam_term + solver_2_term + loss_term
        return dlam_n



# ---------- Second-term construction (transform Hessian) ----------

def make_second_term_matvec(Jt_fn, x, grad_f_t):
    """
    Build a matvec mv(v) = S(x) v, where

      S(x) = Σ_i grad_f_t[i] * Hess_x t_i(x),

    without forming any third-order tensors.

    Jt_fn: x -> J(x) (Jacobian of transform_fn)
    x:     (n,)
    grad_f_t: gradient in transformed coords (m,)

    Returns:
      mv: function R^n -> R^n
    """
    grad_f_t = lax.stop_gradient(grad_f_t)

    def mv(v):
        # DJ[x][v] has shape (m, n)
        DJ = jax.jvp(Jt_fn, (x,), (v,))[1]
        return DJ.T @ grad_f_t  # (n,)

    return mv


def second_term_apply_to_matrix(Jt_fn, x, grad_f_t, V):
    """
    Apply S(x) to all columns of V without forming S.

    V: (n, k)  (typically identity for full Hessian)
    Returns: (n, k)
    """
    mv = make_second_term_matvec(Jt_fn, x, grad_f_t)
    return jax.vmap(mv, in_axes=1, out_axes=1)(V)


def join_f64_via_callback(sign, exp, mant, shape, exp_bits, exp_bias, mbits):
    """
    Calls vpfloat.join_f64(sign, exp, mant, shape, exp_bits, exp_bias, mbits)
    from inside JAX using a host pure_callback.

    Args
      sign, exp, mant: JAX arrays (will be transferred to host for callback)
      shape: tuple / sequence of ints (Python shape of the output)
      exp_bits, exp_bias, mbits: ints

    Returns
      U: JAX array of dtype float64 with `shape`.
    """
    shape = tuple(shape)

    # Host callback: receives NumPy arrays, returns NumPy array
    def _join_cb(sign, exp, mant):
        sign_np = np.asarray(sign)
        exp_np = np.asarray(exp)
        mant_np = np.asarray(mant)
        return vpfloat.join_f64(sign_np, exp_np, mant_np, shape, exp_bits, exp_bias, mbits)

    out_spec = ShapeDtypeStruct(shape=shape, dtype=jnp.float64)

    return jax.pure_callback(_join_cb, out_spec, sign, exp, mant)


def get_loss_grad_vp_fn(pIC, crit, target_trj, stepper, transform_fn,
                    vel_part_trans, dt, T, mbits, exp_bits, exp_bias):
    p_idx = pIC.shape[0]
    adj_step_1 = Adjoint_Stepper_1(
            p_idx, crit, target_trj, stepper, vel_part_trans
        )

    #trj_gen_fn = create_trj_generator(stepper.step.rhs, dt, T)
    trj_gen_fn = create_trj_generator_vpfloat(stepper.step.rhs, dt, T, exp_bits, exp_bias, mbits)
    def get_loss_grad_vp_fn(u_0_vel_raw):
        """
        Pure core: given u_0_vel_raw, compute (loss, grad_x, aux)
        without mutating self. aux is used later for HVPs.
        """
        # 1) Transform IC and get linearization
        u_0_vel, lin = jax.linearize(transform_fn, u_0_vel_raw)

        # 2) Forward trajectory
        sign_trj, exp_trj, mant_trj = trj_gen_fn(pIC, u_0_vel)   # (N, state_dim)

        N = sign_trj.shape[0]
        sign_N = sign_trj[-1]
        exp_N = exp_trj[-1]
        mant_N = mant_trj[-1]
        state_shape = (u_0_vel.shape[0]+pIC.shape[0],)
        U_N = join_f64_via_callback(
            sign_N, exp_N, mant_N,
            state_shape,
            exp_bits, exp_bias, mbits
        )

        # 3) First-order adjoint sweep
        loss0, lam_N = adj_step_1.loss_dg__du_fn(U_N, N - 1)
        xs = (sign_trj[:-1], exp_trj[:-1], mant_trj[:-1], jnp.arange(N - 1))

        def grad_step(carry, x):
            lam, loss_acc = carry
            sign_i, exp_i, mant_i, i = x        # U_i is DA_trj[i]
            U_i = join_f64_via_callback(
                    sign_i, exp_i, mant_i,
                    state_shape,
                    exp_bits, exp_bias, mbits
                )
            lam, g_val = adj_step_1(lam, U_i, i)
            return (lam, loss_acc + g_val), None

        (carry_final, _) = lax.scan(
            grad_step,
            (lam_N, loss0),
            xs,
            reverse=True,
        )

        lam_0, loss = carry_final
        # Gradient in transformed coordinates: only velocity block at time 0
        grad_t = lam_0[p_idx:]          # (n_vel,)

        # Transpose of lin: transformed vel-cotangent -> raw vel-cotangent
        lin_T = jax.linear_transpose(lin, u_0_vel_raw)
        (grad_x,) = lin_T(grad_t)            # (dim(u_0_vel_raw),)


        return loss, grad_x

    return get_loss_grad_vp_fn

def get_loss_grad_conditional_vp_fn(pIC, crit, target_trj, stepper, transform_fn,
                    vel_part_trans, dt, T, mbits, exp_bits, exp_bias):
    p_idx = pIC.shape[0]
    adj_step_1 = Adjoint_Stepper_1(
            p_idx, crit, target_trj, stepper, vel_part_trans
        )

    #trj_gen_fn = create_trj_generator(stepper.step.rhs, dt, T)
    trj_gen_fn = create_trj_generator_vpfloat(stepper.step.rhs, adj_step_1.g, dt, T, exp_bits, exp_bias, mbits)
    
    def get_loss_grad_vp_fn(loss_ub, u_0_vel_raw):
        # 1) Transform IC and linearization
        u_0_vel, lin = jax.linearize(transform_fn, u_0_vel_raw)

        # 2) Forward trajectory (must be fixed-length for JIT)
        sign_trj, exp_trj, mant_trj, loss = trj_gen_fn(pIC, u_0_vel)

        state_shape = (u_0_vel.shape[0] + pIC.shape[0],)

        def do_grad(_):
            N = sign_trj.shape[0]

            sign_N = sign_trj[-1]
            exp_N  = exp_trj[-1]
            mant_N = mant_trj[-1]

            U_N = join_f64_via_callback(
                sign_N, exp_N, mant_N,
                state_shape,
                exp_bits, exp_bias, mbits
            )

            _, lam_N = adj_step_1.loss_dg__du_fn(U_N, N - 1)

            xs = (sign_trj[:-1], exp_trj[:-1], mant_trj[:-1], jnp.arange(N - 1))

            def grad_step(lam, x):
                sign_i, exp_i, mant_i, i = x
                U_i = join_f64_via_callback(
                    sign_i, exp_i, mant_i,
                    state_shape,
                    exp_bits, exp_bias, mbits
                )
                lam, _ = adj_step_1(lam, U_i, i)
                return lam, None

            lam_0, _ = lax.scan(grad_step, lam_N, xs, reverse=True)

            grad_t = lam_0[p_idx:]
            lin_T = jax.linear_transpose(lin, u_0_vel_raw)
            (grad_x,) = lin_T(grad_t)

            return grad_x, jnp.bool_(True)

        def skip_grad(_):
            grad_x = jnp.zeros_like(u_0_vel_raw)
            return grad_x, jnp.bool_(False)

        grad_x, did_grad = lax.cond(
            loss <= loss_ub,
            do_grad,
            skip_grad,
            operand=None
        )

        return loss, grad_x, did_grad

    return get_loss_grad_vp_fn
                                                    

def get_loss_grad_fn(pIC, crit, target_trj, stepper, transform_fn,
                    vel_part_trans, trj_gen_fn):
    p_idx = pIC.shape[0]
    adj_step_1 = Adjoint_Stepper_1(
            p_idx, crit, target_trj, stepper, vel_part_trans
        )

    def get_loss_grad_fn(u_0_vel_raw):
        """
        Pure core: given u_0_vel_raw, compute (loss, grad_x, aux)
        without mutating self. aux is used later for HVPs.
        """
        dtype = u_0_vel_raw.dtype
        # 1) Transform IC and get linearization
        u_0_vel, lin = jax.linearize(transform_fn, u_0_vel_raw)

        # 2) Forward trajectory
        DA_trj = trj_gen_fn(pIC, u_0_vel)   # (N, state_dim)
        N = DA_trj.shape[0]
        # 3) First-order adjoint sweep
        U_N = DA_trj[-1].astype(dtype)
        loss0, lam_N = adj_step_1.loss_dg__du_fn(U_N, N - 1)
        xs = (DA_trj[:-1], jnp.arange(N - 1))

        def grad_step(carry, x):
            lam, loss_acc = carry
            U_i, i = x        # U_i is DA_trj[i]
            U_i = U_i.astype(dtype)
            lam, g_val = adj_step_1(lam, U_i, i)
            return (lam, loss_acc + g_val), None

        (carry_final, _) = lax.scan(
            grad_step,
            (lam_N, loss0),
            xs,
            reverse=True,
        )

        lam_0, loss = carry_final
        # Gradient in transformed coordinates: only velocity block at time 0
        grad_t = lam_0[p_idx:]          # (n_vel,)

        # Transpose of lin: transformed vel-cotangent -> raw vel-cotangent
        lin_T = jax.linear_transpose(lin, u_0_vel_raw)
        (grad_x,) = lin_T(grad_t)            # (dim(u_0_vel_raw),)


        return loss, grad_x

    return get_loss_grad_fn

class Adjoint_Solver:
    def __init__(self, pIC, crit, target_trj, stepper, transform_fn,
                 vel_part_trans, trj_gen_fn):
        self.pIC = pIC
        self.p_idx = pIC.shape[0]
        self.stepper = stepper
        self.transform_fn = transform_fn
        self.trj_gen_fn = trj_gen_fn
        self.vel_part_trans = vel_part_trans

        self.N = target_trj.shape[0]
        self.adj_step_1 = Adjoint_Stepper_1(
            self.p_idx, crit, target_trj, stepper, vel_part_trans
        )
        self.adj_step_2 = Adjoint_Stepper_2(self.adj_step_1)

        # cache for HVPs at the last point where we computed the gradient
        self._aux = None  # will hold (DA_trj, lam_trj, u_0_vel_raw)

    # =========================
    # Jitted core: loss + grad
    # =========================
    @functools.partial(jax.jit, static_argnums=(0,))
    def _loss_grad_core(self, u_0_vel_raw):
        """
        Pure core: given u_0_vel_raw, compute (loss, grad_x, aux)
        without mutating self. aux is used later for HVPs.
        """

        # 1) Transform IC and get linearization
        u_0_vel, lin = jax.linearize(self.transform_fn, u_0_vel_raw)

        # 2) Forward trajectory
        DA_trj = self.trj_gen_fn(self.pIC, u_0_vel)   # (N, state_dim)
        N = DA_trj.shape[0]

        # 3) First-order adjoint sweep
        loss0, lam_N = self.adj_step_1.loss_dg__du_fn(DA_trj[-1], N - 1)
        xs = (DA_trj[:-1], jnp.arange(N - 1, dtype=jnp.int32))

        def grad_step(carry, x):
            lam, loss_acc = carry
            U_i, i = x        # U_i is DA_trj[i]
            lam, g_val = self.adj_step_1(lam, U_i, i)
            return (lam, loss_acc + g_val), lam

        (lam_0, loss), lam_hist = lax.scan(
            grad_step,
            (lam_N, loss0),
            xs,
            reverse=True,
        )

        # full λ trajectory λ_0..λ_{N-1}
        lam_trj = jnp.concatenate([lam_hist, lam_N[None, :]], axis=0)

        # Gradient in transformed coordinates: only velocity block at time 0
        grad_t = lam_0[self.p_idx:]          # (n_vel,)

        # Transpose of lin: transformed vel-cotangent -> raw vel-cotangent
        lin_T = jax.linear_transpose(lin, u_0_vel_raw)
        (grad_x,) = lin_T(grad_t)            # (dim(u_0_vel_raw),)

        # aux for HVPs
        aux = (DA_trj, lam_trj, u_0_vel_raw)

        return loss, grad_x, aux

    def compute_grad(self, u_0_vel_raw):
        """
        Public API: compute loss and grad, and cache aux data for HVPs.
        """
        loss, grad_x, aux = self._loss_grad_core(u_0_vel_raw)
        self._aux = aux
        return loss, grad_x

    # =========================
    # Jitted core: HVP
    # =========================
    @functools.partial(jax.jit, static_argnums=(0,))
    def _hvp_core(self, V, aux):
        """
        Pure core: H @ V, given aux from _loss_grad_core.
        V has shape (dim(u_0_vel_raw), k).
        """

        DA_trj, lam_trj, u_0_vel_raw = aux
        N = DA_trj.shape[0]

        # Rebuild transform linearization at this u_0_vel_raw
        u_0_vel, lin = jax.linearize(self.transform_fn, u_0_vel_raw)

        # 1) Transform directions V into transformed vel coordinates
        # V: (dim(u_0_vel_raw), k)
        V_t = jax.vmap(lin, in_axes=1, out_axes=1)(V)   # (n_vel, k)

        # 2) Forward sensitivities in velocity space along trajectory
        def sens_step(J_k, u_i):
            part, vel = self.vel_part_trans.split_part_and_vel(u_i)

            def vel_step(vel_):
                u_full = jnp.concatenate([part, vel_], axis=0)
                # only velocity part at next step
                return self.vel_part_trans.split_part_and_vel(self.stepper(u_full))[1]

            # lin_step: δvel -> δ(next_vel)
            _, lin_step = jax.linearize(vel_step, vel)
            J_k1 = jax.vmap(lin_step, in_axes=1, out_axes=1)(J_k)  # (n_vel, k)
            return J_k1, J_k1

        # Scan over trajectory (N-1 steps), starting from J_0 = V_t
        _, sens_trj = lax.scan(sens_step, V_t, DA_trj[:-1])
        # Prepend J_0 so we have sensitivities at all N times
        sens_trj = jnp.concatenate([V_t[None, ...], sens_trj], axis=0)  # (N, n_vel, k)

        # 3) Terminal second-order adjoint: H_g(U_N) @ V_N (vel block)
        dlam_N = self.adj_step_2.g_Hess_V(DA_trj[-1], N - 1, sens_trj[-1])  # (n_vel, k)

        # 4) Backward second-order adjoint sweep
        xs2 = (
            DA_trj[:-1],
            lam_trj[1:],      # λ_i for i = 0..N-2
            sens_trj[:-1],
            jnp.arange(N - 1, dtype=jnp.int32),
        )

        def hess_step(carry, x):
            dlam = carry
            U_i, lam_i, sens_i, i = x
            dlam = self.adj_step_2(lam_i, dlam, U_i, sens_i, i)
            return dlam, None

        dlam_0, _ = lax.scan(
            hess_step,
            dlam_N,
            xs2,
            reverse=True,
        )  # dlam_0: (n_vel, k) in transformed-vel coordinates

        # 5) Map HVPs back to raw coordinates using lin_T
        lin_T = jax.linear_transpose(lin, u_0_vel_raw)

        def vjp_one(v):
            (res,) = lin_T(v)
            return res

        Hess_x = jax.vmap(vjp_one, in_axes=1, out_axes=1)(dlam_0)  # (dim(u_0_vel_raw), k)
        return Hess_x

    def compute_Hvp(self, V):
        """
        Public API: must call compute_grad first (to populate aux).
        """
        if self._aux is None:
            raise ValueError("Must call compute_grad before compute_Hvp")
        return self._hvp_core(V, self._aux)
