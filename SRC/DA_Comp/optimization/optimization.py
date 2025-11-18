import jax.numpy as jnp
import jax
import optax
import numpy as np
from jax import lax
from SRC.iterative_methods import lanczos_tridiagonal, lanczos_2_eig_decomp, lanczos_eigs
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import eigsh
import random
from SRC.DA_Comp.optimization.parent_classes import LS_TR_Opt
from SRC.DA_Comp.optimization.LS_TR import Cubic_TR
from SRC.DA_Comp.optimization.Quasi_Newton import L_SR1, HVP_Update, L_BK



class BFGS(LS_TR_Opt):
    name = "BFGS"
    def __init__(self, ls, its, fallback_opt, print_loss=False):
        self.ls = ls
        self.ls_method = ls.name
        self.its = its
        self.print_loss = print_loss

        if fallback_opt == "eye":
            self.fallback = self.eye_fallback
    @staticmethod
    def eye_fallback(N):
        return jnp.eye(N)
    
    def init_opt_params(self, N):
        self.Bk_inv = self.fallback(N)


    def inner_loop(self, U_0, grad, loss, loss_fn_base, loss_grad_fn, div_free_proj, eps=1e-8):
        loss_fn = jax.jit(loss_fn_base)
        @jax.jit
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
        alpha = self.ls(loss_fn, U_0, pk, grad)
        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        alpha_pk = pk * alpha  # return value
        loss_next, grad_next, ys, sk, yk = jit_block(U_0_next)

        if ys > eps:
            self.Bk_inv_update(ys, sk, yk)
            return U_0_next, loss_next, grad_next, alpha, alpha_pk, f"Update: {True}"

        else:
            return U_0_next, loss_next, grad_next, alpha, alpha_pk, f"Update: {False}"

    @jax.jit
    def Bk_inv_update(self, ys, sk, yk):
        I = jnp.eye(yk.shape[0], dtype=self.Bk_inv.dtype)
        rho = 1.0 / ys
        Sy = jnp.outer(sk, yk)
        Ys = jnp.outer(yk, sk)
        Ss = jnp.outer(sk, sk)
        self.Bk_inv = (I - rho * Sy) @ self.Bk_inv @ (I - rho * Ys) + rho * Ss

class NCSR1(LS_TR_Opt, L_SR1, HVP_Update):
    name = "NCSR1"
    ls_method = "TR"
    def __init__(self, its, eps_H, max_memory,
                cubic_TR: Cubic_TR,
                grad_prob=0.9, neg_curve_prob=.125, num_hvp_iters=5,
                print_loss=False):
        LS_TR_Opt.__init__(self, its, print_loss)
        self.eps_H = eps_H
        self._max_memory = max_memory
        self.cubic_TR = cubic_TR
        

        #step prob
        self.grad_prob = grad_prob
        self.neg_curve_prob = neg_curve_prob

        self.num_hvp_iters = num_hvp_iters

    
    def second_order_logic(self, grad, hvp, U_0, div_free_proj):
        N = U_0.shape[0]
        computed_min_eig = False
        if (random.random() < self.neg_curve_prob) or (self.init_Bk == False):
            self.init_Bk = True
            w, Q, vTAv_array, v_array  = lanczos_eigs(lambda v: hvp(U_0, v), N, m=self.num_hvp_iters)
            self.HVP_Bk_update(vTAv_array, v_array)

            min_eig_idx = jnp.argmin(w)
            max_eig_idx = jnp.argmax(w)
            max_eig_vec = Q[:, max_eig_idx]
            min_eig_vec = Q[:, min_eig_idx]
            min_eig = w[min_eig_idx]
            max_eig = w[max_eig_idx]
            if jnp.dot(min_eig_vec, grad) > 0:
                    min_eig_vec = -min_eig_vec
            computed_min_eig = True

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

        else:
            NCN_min_eig = self.eps_H
            #reg newton
            #eigen decomp
            A_op = LinearOperator((N, N), matvec=lambda v: self.Bk @ v, dtype=np.float64)
            Bk_eigs, Bk_eig_vec = eigsh(A_op, k=len(self.Bk), which='LM')
            Bk_eigs = jnp.array(Bk_eigs)
            Bk_eig_vec = jnp.array(Bk_eig_vec)

            Q = Bk_eig_vec
            Lam_reg = jnp.abs(Bk_eigs)
            idx = jnp.argsort(Lam_reg)[-self.current_memory:][::-1]
            Lam_reg = Lam_reg[idx]
            Q = Bk_eig_vec[:, idx]
            Lam_reg = jnp.where(Lam_reg > NCN_min_eig, Lam_reg, NCN_min_eig)

            null_space_comp = (-grad) - Q @ (Q.T @ -grad)
            #proj_error = jnp.linalg.norm(null_space_comp)/jnp.linalg.norm(grad)
            #t = Q.T @ Q
            #ortho_error = jnp.linalg.norm(t - jnp.eye(t.shape[0]))
            #print(f"proj error: {proj_error:.4f} | ortho error: {ortho_error:.3e}")
            
            step_type = "reg_Newton"
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

    def inner_loop(self, U_0, grad, loss, loss_fn_base, loss_grad_fn_base, div_free_proj):
        N = U_0.shape[0]

        #Jit and define model functions
        loss_grad_fn = jax.jit(loss_grad_fn_base)
        loss_fn = jax.jit(loss_fn_base)
        @jax.jit
        def hvp(x, v):
            return jax.jvp(jax.grad(loss_fn), (x,), (v,))[1]


        # Search direction and line search
        grad_norm = jnp.linalg.norm(grad)
        gTHg = jnp.dot(grad, hvp(U_0, grad))
        Rk = gTHg / grad_norm**2
        self.HVP_Bk_update(jnp.array([gTHg]), grad.reshape((-1, 1)))
        step_type = "none"

        if ((Rk < -self.eps_H) and (random.random() < self.grad_prob)):
            pk = -grad
            step_type = "grad"

        elif Rk >= -self.eps_H and Rk <= self.eps_H and (random.random() < self.grad_prob):
            pk = -grad
            step_type = "grad"

        #second order methods
        else:                
            pk, step_type = self.second_order_logic(grad, hvp, U_0, div_free_proj)

        if step_type == "grad":
            pTHp = gTHg
        else:
            pTHp = jnp.dot(pk, hvp(U_0, pk))
            self.HVP_Bk_update(jnp.array([pTHp]), pk.reshape((-1, 1)))
            #pTHp_approx = jnp.dot(pk, self.Bk_mat_vec(Bk_vecs, Bk_scalars, pk, self.current_memory))
        
        alpha = self.cubic_TR.get_alpha(pk, grad, pTHp, loss_fn, U_0, loss)
        alpha_pk = pk * alpha

        # Step and new loss/grad
        U_0_next = self.step(U_0, alpha, pk, div_free_proj)
        loss_next, grad_next = loss_grad_fn(U_0_next)
        self.SR1_update(U_0_next, U_0, grad_next, grad, N)
        
        diag_string = f"step type: {step_type} | eta: {self.cubic_TR.eta:.2e}"

        return U_0_next, loss_next, grad_next, alpha, alpha_pk, diag_string
        

