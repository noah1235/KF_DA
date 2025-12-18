import jax.numpy as jnp
import jax
import numpy as np
import functools
from SRC.DA_Comp.adjoint import Adjoint_Solver
from SRC.DA_Comp.loss_funcs import create_loss_fn
from SRC.utils import build_div_free_proj
import time
from SRC.DA_Comp.optimization.LS_TR import ArmijoLineSearch, Cubic_TR, Armijo_TR
class Loss_and_Deriv_fns:
    def __init__(self, loss_crit, stepper, target_trj, pIC, vel_part_trans, trj_gen_fn):
        loss_fn_base = create_loss_fn(loss_crit, stepper, target_trj, pIC, vel_part_trans)
        #adj_transform = build_div_free_proj(stepper, vel_part_trans)

        loss_grad_fn_base = jax.value_and_grad(loss_fn_base)
        #self.adj_solver = Adjoint_Solver(pIC, loss_crit, target_trj, stepper, adj_transform,
        #                vel_part_trans, trj_gen_fn)

        
        self.loss_fn_jit = jax.jit(loss_fn_base)
        self.loss_grad_fn_jit = jax.jit(loss_grad_fn_base)
        self.hvp_fn_jit = jax.jit(self.make_hvp(loss_fn_base))

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

        self.loss_time_total = 0
        self.loss_grad_time_total = 0.0
        self.Hvp_time_total = 0.0


    def loss_fn(self, *args, **kwargs):
        self.loss_evals += 1

        start = time.perf_counter()
        out = self.loss_fn_jit(*args, **kwargs)
        out = jax.block_until_ready(out)
        dt = time.perf_counter() - start

        self.loss_time_total += dt

        return out
        
    def loss_grad_fn(self, *args, **kwargs):
        self.loss_grad_evals += 1

        start = time.perf_counter()
        out = self.loss_grad_fn_jit(*args, **kwargs)
        out = jax.block_until_ready(out)

        end = time.perf_counter()
        dt = end - start
        self.loss_grad_time_total += dt

        return out


    def loss_grad_adj_fn(self, *args, **kwargs):
        raise ValueError("Adjoint needs update")
        self.loss_grad_evals += 1

        start = time.perf_counter()

        out = self.adj_solver.compute_grad(*args, **kwargs)
        out = jax.block_until_ready(out)

        end = time.perf_counter()
        dt = end - start
        self.loss_grad_time_total += dt

        return out

    def HVP_fn(self, u, v):
        self.Hvp_evals += 1
        start = time.perf_counter()

        out = self.hvp_fn_jit(u, v)
        out = jax.block_until_ready(out)

        end = time.perf_counter()
        dt = end - start
        self.Hvp_time_total += dt

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

    def __repr__(self):
        return f"loss_evals: {self.loss_evals} | loss_grad_evals: {self.loss_grad_evals} | Hvp_evals: {self.Hvp_evals}"

class Opt_Data:
    def __init__(self, its):
        self.loss_record = np.zeros(its)
        self.grad_norm_record = np.zeros(its)
        self.alpha_gTp_record = np.zeros(its)

        self.loss_evals_record = np.zeros(its)
        self.loss_grad_evals_record = np.zeros(its)
        self.Hvp_evals_record = np.zeros(its)
    
    def __call__(self, n, loss, grad, alpha_p, loss_evals, loss_grad_evals, Hvp_evals):
        self.loss_record[n] = loss
        self.grad_norm_record[n] = jnp.linalg.norm(grad)
        self.alpha_gTp_record[n] = jnp.dot(alpha_p, grad)
        self.loss_evals_record[n] = loss_evals
        self.loss_grad_evals_record[n] = loss_grad_evals
        self.Hvp_evals_record[n] = Hvp_evals

    def early_stop_update(self, iters):
        self.loss_record = self.loss_record[:iters]
        self.grad_norm_record = self.grad_norm_record[:iters]
        self.alpha_gTp_record = self.alpha_gTp_record[:iters]

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
        
class LS_TR_Opt():
    def __init__(self, its, print_loss):
        self.its = its
        self.print_loss = print_loss

    def __repr__(self):
        return f"{self.name}_{self.ls_method}-{self.its}"
    
    def inner_loop(self):
        pass

    def ls_choice_logic(self, loss_fn, loss, U_0, pk, grad, loss_grad_fn, last_iteration):
        if isinstance(self.ls, ArmijoLineSearch):
            alpha, U_0_next, loss_next, grad_next = self.ls(loss_fn, loss, U_0, pk, grad, loss_grad_fn, last_iteration)
            debug_str = ""
        elif isinstance(self.ls, Cubic_TR):
            pTHp = jnp.dot(pk, -grad)
            alpha, U_0_next, loss_next, grad_next = self.ls.get_alpha(pk, grad, pTHp, loss_fn, U_0, loss, loss_grad_fn, last_iteration)
            debug_str = f"eta: {self.ls.eta}"
        elif isinstance(self.ls, Armijo_TR):
            alpha, U_0_next, loss_next, grad_next = self.ls(pk, grad, loss_fn, U_0, loss, loss_grad_fn, last_iteration)
            debug_str = ""
        return alpha, U_0_next, loss_next, grad_next, debug_str
    
    def opt_loop(self, U_0, loss_fn_and_derivs: Loss_and_Deriv_fns, div_check, div_free_proj):
        opt_data = Opt_Data(self.its)
        self.init_opt_params(U_0.shape[0])


        for i in range(self.its):
            if i == 0:
                loss, grad = loss_fn_and_derivs.loss_grad_fn(U_0)
            loss_prev = loss
            grad_prev = grad
            loss_grad_evals_prev = loss_fn_and_derivs.loss_grad_evals
            #last iteration
            if i == (self.its-1):
                U_0, _, _, alpha, alpha_pk, diag_str = self.inner_loop(U_0, grad, loss, loss_fn_and_derivs, div_free_proj, i, last_iteration=True)
            else:
                U_0, loss, grad, alpha, alpha_pk, diag_str = self.inner_loop(U_0, grad, loss, loss_fn_and_derivs, div_free_proj, i, last_iteration=False)

            if alpha == 0:
                if self.print_loss:
                    print(f"optimizer stalled | alpha={alpha}")

                opt_data.early_stop_update(i)
                break
            loss_evals_prev = loss_fn_and_derivs.loss_evals
            Hvp_evals_prev = loss_fn_and_derivs.Hvp_evals
            opt_data(i, loss_prev, grad_prev, alpha_pk, loss_evals_prev, loss_grad_evals_prev, Hvp_evals_prev)

            if self.print_loss:
                print(f"i:{i} | loss: {loss_prev:.4e} | Div: {div_check(U_0):.2e} | alpha: {alpha:.3e}" + "|" + diag_str)


        return U_0, opt_data
