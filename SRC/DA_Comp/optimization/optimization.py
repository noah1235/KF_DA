import jax.numpy as jnp
import jax
import optax
from jax import lax
from SRC.iterative_methods import lanczos_eigs, pcg_curve_detection
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import eigsh
import random
from SRC.DA_Comp.optimization.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from SRC.DA_Comp.optimization.LS_TR import Cubic_TR
from SRC.DA_Comp.optimization.Quasi_Newton import L_SR1, HVP_Update, L_BK, BFGS_Update
import numpy as np



class BFGS(LS_TR_Opt, BFGS_Update):
    name = "BFGS"
    def __init__(self, ls, its, fallback_opt, print_loss=False):
        BFGS_Update.__init__(self, fallback_opt)
        self.ls = ls
        self.ls_method = ls.name
        self.its = its
        self.print_loss = print_loss


    def inner_loop(self, U_0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, div_free_proj, iter, last_iteration, eps=1e-12):
        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_fn = loss_fn_and_derivs.loss_grad_fn

        def jit_block(U_0_next):
            # Step and new loss/grad
            loss_next, grad_next = loss_grad_fn(U_0_next)

            # Quasi-Newton pair
            sk = U_0_next - U_0
            yk = grad_next - grad

            # Curvature
            ys = jnp.vdot(yk, sk).real
            return loss_next, grad_next, ys, sk, yk
        


        # Search direction and line search
        pk = -self.Bk_inv @ grad
        alpha = self.ls(loss_fn, loss, U_0, pk, grad)
        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        alpha_pk = pk * alpha  # return value
        if last_iteration:
            return U_0_next, jnp.nan, jnp.nan, alpha, alpha_pk, ""
        else:
            loss_next, grad_next, ys, sk, yk = jit_block(U_0_next)

            if ys > eps:
                self.Bk_inv_update(ys, sk, yk)
                return U_0_next, loss_next, grad_next, alpha, alpha_pk, f"Update: {True}"

            else:
                return U_0_next, loss_next, grad_next, alpha, alpha_pk, f"Update: {False}"


class PCGBFGS(BFGS):
    name = "PCGBFGS"
    def __init__(self, ls, its, n_hvp, fallback_opt, print_loss=False):
        super().__init__(ls, its, fallback_opt, print_loss)
        self.n_hvp = n_hvp
        self.pinv_cond = 1e-8

    def inner_loop(self, U_0, grad, loss, loss_fn_and_derivs: Loss_and_Deriv_fns, div_free_proj, iter, last_iteration, eps=1e-12):
        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_fn = loss_fn_and_derivs.loss_grad_fn
        hvp = loss_fn_and_derivs.Hvp_fn

        def jit_block(U_0_next):
            # Step and new loss/grad
            loss_next, grad_next = loss_grad_fn(U_0_next)

            # Quasi-Newton pair
            sk = U_0_next - U_0
            yk = grad_next - grad

            # Curvature
            ys = jnp.vdot(yk, sk).real
            return loss_next, grad_next, ys, sk, yk
        
        S = []
        Y = []

        def matvec(v):
            S.append(v)
            Hv = hvp(U_0, v)
            Y.append(Hv)
            return Hv

        pk, info = pcg(matvec, self.Bk_inv, -grad, x0=(-self.Bk_inv @ grad), maxiter=self.n_hvp)
        #print(jnp.linalg.norm(matvec(pk) + grad) / jnp.linalg.norm(grad))
        S = jnp.vstack(S).T
        Y = jnp.vstack(Y).T
        R = S - self.Bk_inv @ Y
    
        self.Bk_inv = self.Bk_inv + R @ jnp.linalg.pinv(Y.T @ R, rcond=self.pinv_cond) @ R.T

        if False:
            for i in range(5):
                v = S[:, i]
                Hv = Y[:, i]
                print(jnp.linalg.norm(self.Bk_inv @ Hv - v) / jnp.linalg.norm(v), jnp.linalg.norm(v))
                print("-----")

        alpha = self.ls(loss_fn, loss, U_0, pk, grad)
        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        alpha_pk = pk * alpha  # return value
        if last_iteration:
            return U_0_next, jnp.nan, jnp.nan, alpha, alpha_pk, ""
        else:
            loss_next, grad_next, ys, sk, yk = jit_block(U_0_next)

            if ys > eps:
                self.Bk_inv_update(ys, sk, yk)
                return U_0_next, loss_next, grad_next, alpha, alpha_pk, f"Update: {True}"

            else:
                return U_0_next, loss_next, grad_next, alpha, alpha_pk, f"Update: {False}"
        
class NCSR1(LS_TR_Opt, L_SR1, HVP_Update):
    ls_method = "TR"
    def __init__(self, its, eps_H, max_memory,
                cubic_TR: Cubic_TR,
                grad_prob=0.9, neg_curve_prob=.125, num_hvp_iters=5,
                SR1_type="conv", psd_crit=None,
                print_loss=False):
        LS_TR_Opt.__init__(self, its, print_loss)
        self.set_SR1_update_type(SR1_type)
        self.name = "NCSR1"
        if SR1_type == "mod":
            self.name += "M"

        self.eps_H = eps_H
        self._max_memory = max_memory
        self.cubic_TR = cubic_TR
        

        #step prob
        self.grad_prob = grad_prob
        self.neg_curve_prob = neg_curve_prob

        self.num_hvp_iters = num_hvp_iters
        

        if psd_crit is not None:
            #psd decision can be: early stopping, no negative curvature, nothing
            self.min_its, self.num_pos_eigs_crit, self.min_loss_crit, self.psd_decision = psd_crit
            self.num_pos_min_eigs = 0


    def second_order_logic_backup(self, grad, hvp, U_0, div_free_proj, iter):
        N = U_0.shape[0]
    
        computed_min_eig = False
        if (random.random() < self.neg_curve_prob) or (self.init_Bk == False):
            self.init_Bk = True
            Bk_min_eig, Bk_min_vec = self.Bk_eig_decomp(which="SA", num_eig=1)
            w, Q, vTAv_array, v_array  = lanczos_eigs(lambda v: hvp(U_0, v), N, v0=Bk_min_vec.reshape(-1), m=self.num_hvp_iters)
            self.HVP_Bk_update(vTAv_array, v_array)

            min_eig_idx = jnp.argmin(w)
            max_eig_idx = jnp.argmax(w)
            min_eig_vec = Q[:, min_eig_idx]
            min_eig = w[min_eig_idx]

            if jnp.dot(min_eig_vec, grad) > 0:
                    min_eig_vec = -min_eig_vec
            computed_min_eig = True
            
            if iter >= (self.min_its-1) and min_eig > 0:
                self.num_pos_min_eigs += 1
            else:
                self.num_pos_min_eigs = 0


            ###
            if False:
                print("-------")
                for i in range(4):
                    xi = v_array[:, i]
                    t = jnp.dot(xi, self.Bk @ xi)
                    print(t, vTAv_array[i])

                print("------")
            ###

        if computed_min_eig and min_eig < -self.eps_H:
            pk = min_eig_vec
            step_type = "neg_curve"
        
        elif computed_min_eig and (min_eig > 0):
            print("SR1 PCG")
            step_type = "PCG"
            v_array = []
            vTAv_array = []

            def matvec(v):
                v_array.append(v)
                Hv = hvp(U_0, v)
                vTAv_array.append(jnp.dot(v, Hv))
                return Hv
            M = self.Bk.build_Bk()
            M = jnp.linalg.pinv(M)
            pk, info = pcg(matvec, M, -grad, x0=(-M @ grad), maxiter=10)
            #print(jnp.linalg.norm(matvec(pk) + grad) / jnp.linalg.norm(grad))
            v_array = jnp.vstack(v_array).T
            vTAv_array = jnp.array(vTAv_array)
            print(v_array.shape, vTAv_array.shape)

            self.HVP_Bk_update(vTAv_array, v_array)

        else:
            NCN_min_eig = self.eps_H
            Bk_eigs, Bk_eig_vec = self.Bk_eig_decomp()

            Q = Bk_eig_vec
            Lam_reg = jnp.abs(Bk_eigs)
            idx = jnp.argsort(Lam_reg)[-self.current_memory:][::-1]
            Lam_reg = Lam_reg[idx]
            Q = Bk_eig_vec[:, idx]
            Lam_reg = jnp.where(Lam_reg > NCN_min_eig, Lam_reg, NCN_min_eig)

            null_space_comp = (-grad) - Q @ (Q.T @ -grad)
            step_type = "reg_Newton"
            pk = Q @ ((Q.T @ -grad) / Lam_reg) + (1.0 / self.eps_H) * null_space_comp

        pk = div_free_proj(pk)

        if jnp.any(jnp.isnan(pk)):
            pk = -grad

        return pk, step_type



    def second_order_logic(self, grad, hvp, U_0, div_free_proj, iter):
        N = U_0.shape[0]
        k = 4
        tried_PCG = False
        step_type = "SR1_reg"
        if iter % k == 0:
            tried_PCG = True
            v_array = []
            vTAv_array = []

            def matvec(v):
                v_array.append(v)
                Hv = hvp(U_0, v)
                vTAv_array.append(jnp.dot(v, Hv))
                return Hv
            if k == 0:
                M = jnp.eye(N)
            else:
                M = jnp.linalg.pinv(self.Bk.build_Bk())
            
            result, flag = pcg_curve_detection(matvec, M, -grad, max_iters=20)
            v_array = jnp.vstack(v_array).T
            vTAv_array = jnp.array(vTAv_array)
            self.HVP_Bk_update(vTAv_array, v_array)

            if flag == "indef":
                v_nc, vHv = result
                if vHv < -self.eps_H:
                    step_type = "neg_curve"
                    if jnp.dot(v_nc, grad) > 0:
                        pk = -v_nc
                    else:
                        pk = v_nc
            else:
                step_type = "PCG"
                pk = result
                #PCG error
                res = -grad - hvp(U_0, pk)
                print(jnp.linalg.norm(res) / jnp.linalg.norm(grad))
                


            ###
            if False:
                print("-------")
                for i in range(v_array.shape[1]):
                    xi = v_array[:, i]
                    t = jnp.dot(xi, self.Bk @ xi)
                    print(t, vTAv_array[i])

                print("------")
            ###

        if tried_PCG or (step_type == "SR1_reg"):
            NCN_min_eig = self.eps_H
            Bk_eigs = jnp.array([self.Bk.Bk_scalars[0]])
            Bk_eig_vec = self.Bk.Bk_vecs[:, 0].reshape((-1, 1))

            Q = Bk_eig_vec
            Lam_reg = jnp.abs(Bk_eigs)
            idx = jnp.argsort(Lam_reg)[-self.current_memory:][::-1]
            Lam_reg = Lam_reg[idx]
            Q = Bk_eig_vec[:, idx]
            Lam_reg = jnp.where(Lam_reg > NCN_min_eig, Lam_reg, NCN_min_eig)

            null_space_comp = (-grad) - Q @ (Q.T @ -grad)
            pk = Q @ ((Q.T @ -grad) / Lam_reg) + (1.0 / self.eps_H) * null_space_comp

        pk = div_free_proj(pk)

        if jnp.any(jnp.isnan(pk)):
            pk = -grad

        return pk, step_type


    @staticmethod
    def check_eig(hvp, x, v):
        vTHv = jnp.dot(v, hvp(x, v))
        return vTHv/jnp.linalg.norm(v)
    
    def init_opt_params(self, N):
        self.init_Bk = False
        self.Bk = L_BK(self._max_memory, N)
        self.current_memory = 0
        self.cubic_TR.init_opt()

    def inner_loop(self, U_0, grad, loss, loss_fn_and_derivs, div_free_proj, iter, last_iteration):
        N = U_0.shape[0]

        loss_fn = loss_fn_and_derivs.loss_fn
        loss_grad_fn = loss_fn_and_derivs.loss_grad_fn
        hvp = loss_fn_and_derivs.Hvp_fn

        if iter == 0:
            grad_norm = jnp.linalg.norm(grad)
            gTHg = jnp.dot(grad, hvp(U_0, grad))
            Rk = gTHg / grad_norm**2
            self.HVP_Bk_update(jnp.array([gTHg]), grad.reshape((-1, 1)))
            er = jnp.linalg.norm( jnp.dot(grad, self.Bk @ grad))
            print(f"er: {er}")

        if False and ((Rk < -self.eps_H) and (random.random() < self.grad_prob)):
            pk = -grad
            step_type = "grad"

        elif False and Rk >= -self.eps_H and Rk <= self.eps_H and (random.random() < self.grad_prob):
            pk = -grad
            step_type = "grad"

        #second order methods
        else:                
            pk, step_type = self.second_order_logic(grad, hvp, U_0, div_free_proj, iter)


        pTHp = jnp.dot(pk, self.Bk @ pk)
        alpha = self.cubic_TR.get_alpha(pk, grad, pTHp, loss_fn, U_0, loss)
        alpha_pk = pk * alpha

        # Step and new loss/grad
        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        diag_string = f"step type: {step_type} | eta: {self.cubic_TR.eta:.2e}"
        
        if last_iteration:
            return U_0_next, np.nan, np.nan, alpha, alpha_pk, diag_string
        else:
            loss_next, grad_next = loss_grad_fn(U_0_next)
            self.SR1_update(U_0_next, U_0, grad_next, grad, loss_next, loss)
            

            if (self.num_pos_min_eigs >= self.num_pos_eigs_crit) and (loss <= self.min_loss_crit):
                if self.psd_decision == "early_stop":
                    print("psd stopping")
                    alpha = 0
                elif self.psd_decision == "no_neg":
                    self.neg_curve_prob = 0
                    print("psd: no neg curv")

            return U_0_next, loss_next, grad_next, alpha, alpha_pk, diag_string
        
class NCSR1_and_BFGS_and_PCGBFGS:
    def __init__(self, NCSR1_opt: NCSR1, BFGS_opt: BFGS, PCGBFGS_opt = None):
        self.NCSR1_opt = NCSR1_opt
        self.BFGS_opt = BFGS_opt
        self.PCGBFGS_opt = PCGBFGS_opt

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
        U_0_DA_fourier, opt_data_1 = self.NCSR1_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)

        Bk_inv = self.get_Bk_inv_for_BFGS()
        self.BFGS_opt.set_Bk_inv_init(Bk_inv)
        U_0_DA_fourier, opt_data_2 = self.BFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)

        opt_data = opt_data_1 + opt_data_2
        if self.PCGBFGS_opt is not None:
            self.PCGBFGS_opt.set_Bk_inv_init(self.BFGS_opt.Bk_inv)
            U_0_DA_fourier, opt_data_3 = self.PCGBFGS_opt.opt_loop(U_0_DA_fourier, loss_fn_and_derivs, div_check, div_free_proj)
            opt_data = opt_data + opt_data_3

        return U_0_DA_fourier, opt_data
    
    def __repr__(self):
        name = f"{self.NCSR1_opt}__{self.BFGS_opt}"
        if self.PCGBFGS_opt is not None:
            name += f"__{self.PCGBFGS_opt}"
        return name

