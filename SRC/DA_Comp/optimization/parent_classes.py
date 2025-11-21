import jax.numpy as jnp
import jax
import numpy as np
import functools



class Loss_and_Deriv_fns:
    def __init__(self, loss_fn_base):
        loss_grad_fn_base = jax.value_and_grad(loss_fn_base)
        def Hvp_base(x, v):
            return jax.jvp(jax.grad(loss_fn_base), (x,), (v,))[1]
        
        self.loss_fn_jit = jax.jit(loss_fn_base)
        self.loss_grad_fn_jit = jax.jit(loss_grad_fn_base)
        self.Hvp_jit = jax.jit(Hvp_base)

        self.loss_evals = 0
        self.loss_grad_evals = 0
        self.Hvp_evals = 0

    def loss_fn(self, *args, **kwargs):
        self.loss_evals += 1
        return self.loss_fn_jit(*args, **kwargs)
    
    def loss_grad_fn(self, *args, **kwargs):
        self.loss_grad_evals += 1
        return self.loss_grad_fn_jit(*args, **kwargs)
    
    def Hvp_fn(self, *args, **kwargs):
        self.Hvp_evals += 1
        return self.Hvp_jit(*args, **kwargs)  

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

    def step(
        self,
        x: jnp.ndarray,      # current x, shape (m,)
        alpha: float,
        p: jnp.ndarray,       # search direction, shape (m,)
        div_free_proj
    ) -> jnp.ndarray:
        
        return div_free_proj(x + alpha * p)
    
    def opt_loop(self, U_0, loss_fn_and_derivs: Loss_and_Deriv_fns, div_check, div_free_proj):
        opt_data = Opt_Data(self.its)
        self.init_opt_params(U_0.shape[0])


        for i in range(self.its):
            if i == 0:
                loss, grad = loss_fn_and_derivs.loss_grad_fn(U_0)
            loss_prev = loss
            grad_prev = grad
            loss_grad_evals_prev = loss_fn_and_derivs.loss_grad_evals
            Hvp_evals_prev = loss_fn_and_derivs.Hvp_evals
            
            #last iteration
            if i == (self.its-1):
                U_0, _, _, alpha, alpha_pk, diag_str = self.inner_loop(U_0, grad, loss, loss_fn_and_derivs, div_free_proj, last_iteration=True)
            else:
                U_0, loss, grad, alpha, alpha_pk, diag_str = self.inner_loop(U_0, grad, loss, loss_fn_and_derivs, div_free_proj, last_iteration=False)

            if alpha == 0:
                if self.print_loss:
                    print(f"optimizer stalled | alpha={alpha}")

                opt_data.early_stop_update(i)
                break
            loss_evals_prev = loss_fn_and_derivs.loss_evals
            opt_data(i, loss_prev, grad_prev, alpha_pk, loss_evals_prev, loss_grad_evals_prev, Hvp_evals_prev)

            if self.print_loss:
                print(f"i:{i} | loss: {loss_prev:.4e} | Div: {div_check(U_0):.2e} | alpha: {alpha:.3e}" + "|" + diag_str)


        return U_0, opt_data
