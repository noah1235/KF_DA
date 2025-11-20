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

class NCSR1_and_BFGS:
    def __init__(self, NCSR1_opt, BFGS_opt):
        self.NCSR1_opt = NCSR1_opt
        self.BFGS_opt = BFGS_opt

    def get_Bk_inv_for_BFGS(self):
        Bk_eigs, Bk_eig_vec = self.NCSR1_opt.Bk_eig_decomp(which="LM")
        n = self.NCSR1_opt.Bk.N        
        k = Bk_eig_vec.shape[1]
        min_eig = 1e-6
        Bk_eigs_clipped = np.maximum(Bk_eigs, min_eig)
        pad = np.full(n - k, min_eig, dtype=Bk_eigs.dtype)
        Bk_eigs_full = np.concatenate([Bk_eigs_clipped, pad], axis=0)
        U, _, _ = np.linalg.svd(Bk_eig_vec, full_matrices=True)
        Q_perp = U[:, k:]
        Bk_eig_vec_full = np.concatenate([Bk_eig_vec, Q_perp], axis=1)

        Bk_inv = (Bk_eig_vec_full * (1/Bk_eigs_full)) @ Bk_eig_vec_full.T
        Bk_inv = 0.5 * (Bk_inv + Bk_inv.T)
        Bk_inv = jnp.array(Bk_inv)
        return Bk_inv

    def opt_loop(self, U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj):
        min_eig = 1e-8
        U_0_DA_fourier, opt_data, its = self.NCSR1_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)

        Bk_inv = self.get_Bk_inv_for_BFGS()
        self.BFGS_opt.set_Bk_inv_init(Bk_inv)

        U_0_DA_fourier, opt_data, its = self.BFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)

        return U_0_DA_fourier, opt_data, its
    
    def __repr__(self):
        return f"{self.NCSR1_opt}__{self.BFGS_opt}"

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
                print(f"loss: {loss}")
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
