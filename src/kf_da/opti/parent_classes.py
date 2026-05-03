import jax.numpy as jnp
import jax
import numpy as np
import functools
from kf_da.daComp.adjoint import Adjoint_Solver, get_loss_grad_vp_fn, get_loss_grad_conditional_vp_fn
from kf_da.daComp.loss_funcs import create_loss_fn
from kf_da.utils.utils import build_div_free_proj
from kf_da.solver.solver import Omega_Integrator, KF_Stepper
import time
import os
from kf_da.opti.LS_TR import ArmijoLineSearch

class Loss_and_Deriv_fns_bu:
    def __init__(self, loss_crit, inv_transform, stepper, kf_stepper, target_trj, pp_sigma, meas_part_pos, dt, T, vfloat):
        xp_meas_traj, yp_meas_traj = meas_part_pos
        #num_parts = xp_meas_traj.shape[0] * xp_meas_traj.shape[1]
        
        gen_loss_fn = create_loss_fn(loss_crit, stepper, target_trj, pp_sigma, meas_part_pos, inv_transform)
        self.gen_loss_fn = gen_loss_fn
        PP_opt = jnp.concat([xp_meas_traj.copy().reshape(-1), yp_meas_traj.copy().reshape(-1)])
        loss_fn_base = lambda Z0: gen_loss_fn(Z0, PP_opt)
        self.hvp_fn_jit = jax.jit(self.make_hvp(loss_fn_base))

        if vfloat is None:
            loss_grad_fn_base = jax.value_and_grad(loss_fn_base)
            self.loss_grad_fn_jit = jax.jit(loss_grad_fn_base)
            self.conditional_loss_grad_fn_jit = jax.jit(lambda ub, x: self._conditional_loss_grad_fn_helper(ub, x, loss_fn_base))                
            
        else:
            snap_shape = target_trj[0][0].shape
            loss_grad_conditional_vp_fn_adj_jit = get_loss_grad_conditional_vp_fn(
                    loss_crit, target_trj, kf_stepper, inv_transform, snap_shape, dt, T,
                    mbits=vfloat.mbits, exp_bits=vfloat.exp_bits, exp_bias=vfloat.exp_bias,
                )
            self.conditional_loss_grad_fn_jit = jax.jit(loss_grad_conditional_vp_fn_adj_jit)
            self.loss_grad_fn_jit = lambda x: self.conditional_loss_grad_fn_jit(jnp.inf, x)[:-1]

        
        self.loss_fn_jit = jax.jit(loss_fn_base)
        self.reset_cost_count()

    @staticmethod
    def make_hvp(loss_fn):
        """
        Create a Hessian-vector product (HVP) function for a scalar loss.

        loss_fn(params, *args, **kwargs) -> scalar
        Returns:
            hvp_fn(params, v, *args, **kwargs) -> same shape as params
        """
        grad_fn = jax.grad(loss_fn)

        def hvp_fn(params, v):
            # H(params) @ v = d/dε grad(loss)(params + ε v) |_{ε=0}
            return jax.jvp(lambda p: grad_fn(p), (params,), (v,))[1]

        return hvp_fn

    def reset_cost_count(self):
        self.loss_evals = 0
        self.loss_grad_evals = 0
        self.Hvp_evals = 0

    def loss_fn(self, *args, **kwargs):
        self.loss_evals += 1
        out = self.loss_fn_jit(*args, **kwargs)

        return out
        
    def loss_grad_fn(self, *args, **kwargs):
        self.loss_grad_evals += 1
        out = self.loss_grad_fn_jit(*args, **kwargs)

        return out


    def HVP_fn(self, u, v):
        self.Hvp_evals += 1
        out = self.hvp_fn_jit(u, v)

        return out

    def Hvp_adj_fn(self, Q):
        raise ValueError("adjoint needs update")
        # If Q has k columns, one call = k HVPs
        k = Q.shape[1]
        self.Hvp_evals += k

        start = time.perf_counter()

        out = self.adj_solver.compute_Hvp(Q)
        out = jax.block_until_ready(out)

        end = time.perf_counter()
        dt = end - start
        self.Hvp_time_total += dt

        return out 

    def conditional_loss_grad_fn(self, loss_ub, U):
        loss, grad, active = self.conditional_loss_grad_fn_jit(loss_ub, U)
        if active:
            self.loss_grad_evals += 1
        else:
            self.loss_evals += 1
        
        return loss, grad, active
    
    @staticmethod
    def _conditional_loss_grad_fn_helper(loss_ub, U, loss_fn_base):
        loss, pullback = jax.vjp(loss_fn_base, U)  # forward happens here (always)
        def do_pullback(_):
            # pullback returns a tuple of cotangents (one per primal input)
            (g,) = pullback(1.0)  # assuming loss is scalar
            return g, True

        def no_grad(_):
            g0 = jnp.ones_like(U)
            return g0, False

        grad, active = jax.lax.cond(loss < loss_ub, do_pullback, no_grad, operand=None)
        return loss, grad, active

    def __repr__(self):
        return f"loss_evals: {self.loss_evals} | loss_grad_evals: {self.loss_grad_evals} | Hvp_evals: {self.Hvp_evals}"

import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp


class Loss_and_Deriv_fns:
    def __init__(
        self,
        loss_crit,
        inv_transform,
        stepper,
        kf_stepper,
        target_trj,
        pp_sigma,
        meas_part_pos,
        dt,
        T,
        vfloat,
        allow_dynamic_PP_opt=True,
        checkpoint=False,
    ):
        xp_meas_traj, yp_meas_traj = meas_part_pos

        self.allow_dynamic_PP_opt = allow_dynamic_PP_opt

        gen_loss_fn = create_loss_fn(
            loss_crit, stepper, target_trj, pp_sigma, meas_part_pos, inv_transform, checkpoint=checkpoint
        )
        self.gen_loss_fn = gen_loss_fn

        self.PP_opt_default = jnp.concatenate([
            xp_meas_traj.reshape(-1),
            yp_meas_traj.reshape(-1),
        ])

        def loss_fn_base(Z0, PP_opt):
            return gen_loss_fn(Z0, PP_opt)

        self._loss_fn_base = loss_fn_base

        self.loss_fn_jit = jax.jit(loss_fn_base)
        self.loss_grad_fn_jit = jax.jit(jax.value_and_grad(loss_fn_base, argnums=0))
        self.hvp_fn_jit = jax.jit(self.make_hvp(loss_fn_base, argnums=0))
        self.conditional_loss_grad_fn_jit = jax.jit(
            lambda loss_ub, Z0, PP_opt: self._conditional_loss_grad_fn_helper(
                loss_ub, Z0, PP_opt, loss_fn_base
            )
        )

        self.reset_cost_count()

    @staticmethod
    def make_hvp(loss_fn, argnums=0):
        grad_fn = jax.grad(loss_fn, argnums=argnums)

        def hvp_fn(params, v, *args):
            return jax.jvp(lambda p: grad_fn(p, *args), (params,), (v,))[1]

        return hvp_fn

    def _resolve_PP_opt(self, PP_opt):
        if PP_opt is None:
            return self.PP_opt_default

        if not self.allow_dynamic_PP_opt:
            raise ValueError(
                "PP_opt was passed, but allow_dynamic_PP_opt=False. "
                "Call this function with only Z0, or set allow_dynamic_PP_opt=True."
            )

        return PP_opt

    def set_PP_opt(self, PP_opt):
        self.PP_opt_default = PP_opt

    def reset_cost_count(self):
        self.loss_evals = 0
        self.loss_grad_evals = 0
        self.Hvp_evals = 0

    def loss_fn(self, Z0, PP_opt=None):
        self.loss_evals += 1
        PP_opt = self._resolve_PP_opt(PP_opt)
        return self.loss_fn_jit(Z0, PP_opt)

    def loss_grad_fn(self, Z0, PP_opt=None):
        self.loss_grad_evals += 1
        PP_opt = self._resolve_PP_opt(PP_opt)
        return self.loss_grad_fn_jit(Z0, PP_opt)

    def HVP_fn(self, Z0, v, PP_opt=None):
        self.Hvp_evals += 1
        PP_opt = self._resolve_PP_opt(PP_opt)
        return self.hvp_fn_jit(Z0, v, PP_opt)

    def conditional_loss_grad_fn(self, loss_ub, Z0, PP_opt=None):
        PP_opt = self._resolve_PP_opt(PP_opt)
        loss, grad, active = self.conditional_loss_grad_fn_jit(loss_ub, Z0, PP_opt)

        if active:
            self.loss_grad_evals += 1
        else:
            self.loss_evals += 1

        return loss, grad, active

    @staticmethod
    def _conditional_loss_grad_fn_helper(loss_ub, Z0, PP_opt, loss_fn_base):
        loss, pullback = jax.vjp(loss_fn_base, Z0, PP_opt)

        def do_pullback(_):
            g_Z0, _ = pullback(1.0)
            return g_Z0, True

        def no_grad(_):
            g0 = jnp.zeros_like(Z0)
            return g0, False

        grad, active = jax.lax.cond(
            loss < loss_ub,
            do_pullback,
            no_grad,
            operand=None,
        )
        return loss, grad, active

    def __repr__(self):
        return (
            f"loss_evals: {self.loss_evals} | "
            f"loss_grad_evals: {self.loss_grad_evals} | "
            f"Hvp_evals: {self.Hvp_evals}"
        )


class Opt_Data:
    def __init__(self, its):
        self.loss_record = np.zeros(its)
        self.grad_norm_record = np.zeros(its)
        self.alpha_gTp_record = np.zeros(its)
        self.IC_error_record = np.zeros(its)

        self.loss_evals_record = np.zeros(its)
        self.loss_grad_evals_record = np.zeros(its)
        self.Hvp_evals_record = np.zeros(its)
    
    def __call__(self, n, loss, grad, alpha_p, IC_error, loss_evals, loss_grad_evals, Hvp_evals):
        self.loss_record[n] = loss
        self.grad_norm_record[n] = jnp.linalg.norm(grad)
        self.alpha_gTp_record[n] = jnp.dot(alpha_p, grad)
        self.IC_error_record[n] = jnp.linalg.norm(IC_error)

        self.loss_evals_record[n] = loss_evals
        self.loss_grad_evals_record[n] = loss_grad_evals
        self.Hvp_evals_record[n] = Hvp_evals

    def early_stop_update(self, iters):
        raise NotImplementedError("early_stop_update needs update")
        self.loss_record = self.loss_record[:iters]
        self.grad_norm_record = self.grad_norm_record[:iters]
        self.alpha_gTp_record = self.alpha_gTp_record[:iters]

    def save_data(self, root):
        np.save(os.path.join(root, "loss_record.npy"), np.array(self.loss_record))
        np.save(os.path.join(root, "grad_norm_record.npy"), np.array(self.grad_norm_record))
        np.save(os.path.join(root, "alpha_gTp_record.npy"), np.array(self.alpha_gTp_record))
        np.save(os.path.join(root, "IC_error_record.npy"), np.array(self.IC_error_record))

    def __add__(self, other):
        if not isinstance(other, Opt_Data):
            return NotImplemented

        # Concatenate each record
        new_loss = np.concatenate([self.loss_record, other.loss_record])
        new_grad = np.concatenate([self.grad_norm_record, other.grad_norm_record])
        new_alpha = np.concatenate([self.alpha_gTp_record, other.alpha_gTp_record])

        new_loss_evals_record = np.concatenate([self.loss_evals_record, other.loss_evals_record])
        new_loss_grad_evals_record = np.concatenate([self.loss_grad_evals_record, other.loss_grad_evals_record])
        new_Hvp_evals_record = np.concatenate([self.Hvp_evals_record, other.Hvp_evals_record])

        # Create new object with correct size
        out = Opt_Data(len(new_loss))
        out.loss_record = new_loss
        out.grad_norm_record = new_grad
        out.alpha_gTp_record = new_alpha

        out.loss_evals_record = new_loss_evals_record
        out.loss_grad_evals_record = new_loss_grad_evals_record
        out.Hvp_evals_record = new_Hvp_evals_record

        return out

class Psuedo_Projection:
    def __init__(self, it_list, T):
        self.it_list = it_list
        self.T = T
    
    def attach_transform(self, transform, inv_transform):
        self.transform = transform
        self.inv_transform = inv_transform

    def attach_stepper(self, stepper: KF_Stepper):
        self.integrator = Omega_Integrator(stepper)
        self.nsteps = int(self.T / stepper.dt)
        print(self.nsteps)
    def __call__(self, z):
        omega_hat = self.inv_transform(z)
        omega_hat = self.integrator.fv_integrate(omega_hat, self.nsteps)
        return self.transform(omega_hat)

class LS_TR_Opt():
    def __init__(self, its, psuedo_proj, print_loss):
        self.its = its
        self.psuedo_proj = psuedo_proj
        self.print_loss = print_loss

        self.PP_opt = None

    def __repr__(self):
        name = f"{self.name}_{self.ls_method}-{self.its}"
        if self.psuedo_proj is not None:
            name += f"_PP"
        return name
    
    def inner_loop(self):
        pass


    def ls_choice_logic(self, i, loss, Z0, pk, grad, loss_grad_cond_fn, last_iteration):
        Z0, did_pp = self.apply_psuedo_proj(Z0, i)
        no_grad = last_iteration or did_pp
        if isinstance(self.ls, ArmijoLineSearch):
            alpha, Z0_next, loss_next, grad_next = self.ls(loss, Z0, pk, grad, loss_grad_cond_fn, no_grad)
            debug_str = ""
        if did_pp:
            print("Psuedo-Projection applied; recomputing loss and grad")
            loss_next, grad_next, _ = loss_grad_cond_fn(jnp.inf, Z0_next)
        return alpha, Z0_next, loss_next, grad_next, debug_str

    def it0_logic(self, init_loss, init_grad, init_Hg, loss_fn_and_derivs, Z0, do_Hg_int):
        if init_loss is not None and init_grad is not None:
            loss = init_loss
            grad = init_grad
        else:
            loss, grad = loss_fn_and_derivs.loss_grad_fn(Z0, PP_opt=self.PP_opt)

        if do_Hg_int:
            if init_Hg is not None:
                self.Hvp_init(grad, init_Hg)
            else:
                init_Hg = loss_fn_and_derivs.HVP_fn(Z0, grad, PP_opt=self.PP_opt)
                self.Hvp_init(grad, init_Hg)

        return loss, grad
    
    def apply_psuedo_proj(self, Z0, i):
        if self.psuedo_proj is not None and i in self.psuedo_proj.it_list:
            print("Applying Psuedo-Projection")
            Z0 = self.psuedo_proj(Z0)
            return Z0, True
        else:
            return Z0, False

    def opt_loop(self, Z0, loss_fn_and_derivs: Loss_and_Deriv_fns, inv_transform, omega0_hat_trg, attractor_rad, init_loss=None, init_grad=None, init_Hg=None, do_Hg_int=True, do_last_it_logic=True):
        opt_data = Opt_Data(self.its)
        self.init_opt_params(Z0.shape[0])
        for i in range(self.its):
            if i == 0:
                loss, grad = self.it0_logic(init_loss, init_grad, init_Hg, loss_fn_and_derivs, Z0, do_Hg_int)

            IC_error = (inv_transform(Z0) - omega0_hat_trg) / attractor_rad
            loss_prev = loss
            grad_prev = grad
            loss_grad_evals_prev = loss_fn_and_derivs.loss_grad_evals
            
            #last iteration
            if i == (self.its-1) and do_last_it_logic:
                Z0, _, _, alpha, alpha_pk, diag_str = self.inner_loop(Z0, grad, loss, loss_fn_and_derivs, i, last_iteration=True)
            else:
                Z0, loss, grad, alpha, alpha_pk, diag_str = self.inner_loop(Z0, grad, loss, loss_fn_and_derivs, i, last_iteration=False)

            if alpha == 0:
                if self.print_loss:
                    print(f"optimizer stalled | alpha={alpha}")

                opt_data.early_stop_update(i)
                break
            loss_evals_prev = loss_fn_and_derivs.loss_evals
            Hvp_evals_prev = loss_fn_and_derivs.Hvp_evals
            opt_data(i, loss_prev, grad_prev, alpha_pk, IC_error, loss_evals_prev, loss_grad_evals_prev, Hvp_evals_prev)

            if self.print_loss:
                print(f"i:{i} | loss: {loss_prev:.4e} | alpha: {alpha:.3e}" + "|" + diag_str)


        return Z0, opt_data
