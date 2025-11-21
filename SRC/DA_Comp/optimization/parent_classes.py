import jax.numpy as jnp
import jax
import numpy as np


class Opt_Data:
    def __init__(self, its):
        self.loss_record = np.zeros(its)
        self.grad_norm_record = np.zeros(its)
        self.alpha_gTp_record = np.zeros(its)
    
    def __call__(self, n, loss, grad, alpha_p):
        self.loss_record[n] = loss
        self.grad_norm_record[n] = jnp.linalg.norm(grad)
        self.alpha_gTp_record[n] = jnp.dot(alpha_p, grad)

    def early_stop_update(self, iters):
        self.loss_record = self.loss_record[:iters]
        self.grad_norm_record = self.grad_norm_record[:iters]
        self.alpha_gTp_record = self.alpha_gTp_record[:iters]


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
    
    def opt_loop(self, U_0, loss_fn_and_derivs, div_check, div_free_proj):
        opt_data = Opt_Data(self.its)
        self.init_opt_params(U_0.shape[0])
        for i in range(self.its):
            if i == 0:
                loss, grad = loss_fn_and_derivs["loss_grad_fn"](U_0)
            loss_prev = loss
            grad_prev = grad

            U_0, loss, grad, alpha, alpha_pk, diag_str = self.inner_loop(U_0, grad, loss, loss_fn_and_derivs, div_free_proj)
            if alpha == 0:
                if self.print_loss:
                    print(f"optimizer stalled | alpha={alpha}")

                opt_data.early_stop_update(i)
                break
            opt_data(i, loss_prev, grad_prev, alpha_pk)

            if self.print_loss:
                print(f"i:{i} | loss: {loss:.4e} | Div: {div_check(U_0):.2e} | alpha: {alpha:.3e}" + "|" + diag_str)


        return U_0, opt_data, i
