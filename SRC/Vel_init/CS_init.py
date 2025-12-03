import numpy as np
import jax.numpy as jnp
from SRC.Vel_init.IC_init import IC_init
from SRC.utils import build_div_free_proj, Vel_Part_Transformations, Specteral_Upsampling, bilinear_sample_periodic
from SRC.DA_Comp.loss_funcs import MSE_Vel
import jax
from scipy.optimize import minimize
import jaxopt


class CS_init(IC_init):
    def __init__(self, l1_weight, can_modes):
        self.l1_weight = l1_weight

        self.can_modes = can_modes
        

    def get_attractor_snaps(self, attractor_snapshots):
        self.attractor_snapshots = attractor_snapshots
        self.attractor_rad = self.calc_attractor_size(attractor_snapshots)

    @staticmethod
    def make_first_k_mask(N, k, ord='xy'):
        """
        Create a mask for rfft2(U) that keeps only the first k modes in both directions.

        Parameters
        ----------
        N   : grid size (U is NxN)
        k   : number of lowest-index modes to keep (0, 1, ..., k-1)
        ord : meshgrid indexing ('ij' or 'xy')

        Returns
        -------
        M   : boolean mask of shape (N, N//2 + 1)
        """
        # Physical frequencies: -N/2,...,N/2-1 (fftfreq*N)
        mx_full = jnp.fft.fftfreq(N) * N
        my_full = jnp.fft.fftfreq(N) * N

        MX, MY = jnp.meshgrid(mx_full, my_full, indexing=ord)

        # Keep only frequencies with |kx| < k and |ky| < k
        # i.e., rectangular low-pass on indices
        M_full = (jnp.abs(MX) < k) & (jnp.abs(MY) < k)

        # Restrict to rfft2 domain (positive y-frequencies)
        return M_full[:, :N//2 + 1].astype(jnp.complex128)

    def set_transform(self, stepper, vel_part_trans):
        self.N = stepper.step.rhs.KF_RHS.N
        self.vel_part_trans = vel_part_trans
        self.transform_fn = build_div_free_proj(
                    stepper,
                    vel_part_trans,
    )
        self.mse_vel = MSE_Vel()
        self.mse_vel.init_obj(jnp.array([1]), stepper.step.rhs.KF_RHS.L, vel_part_trans)
        self.upsample_factor = stepper.step.rhs.r
    
    def g_wrapper(self, target_part_x, target_part_y, U_flat, trg_U_flat):
        return self.mse_vel.g(None, None, target_part_x, target_part_y, U_flat, trg_U_flat, self.upsample_factor, 0)

    def __repr__(self):
        return "CS"

    def __call__(self, U_0, pIC, DA_loss_fn_base):
        DA_loss_fn = lambda x: DA_loss_fn_base(self.vel_part_trans.vel_flat_2_vel_Fourier(x))
        # ----------------------------------------------------
        # Loss function 
        # ----------------------------------------------------
        target_part_x, target_part_y, _, _ = self.vel_part_trans.get_part_pos_and_vel(pIC)

        best_DA_loss = None
        for k in self.can_modes:
            mask = self.make_first_k_mask(self.N, k)
            
            @jax.jit
            def loss_fn(X):
                # X: u_0 in Fourier space
                U_guess = self.transform_fn(X, M=mask)
                l1_pen = jnp.mean(jnp.abs(X)) * self.l1_weight
                data_loss = self.g_wrapper(target_part_x, target_part_y, U_guess, U_0)
                return data_loss + l1_pen

            # ----------------------------------------------------
            # BFGS optimizer (JAXopt)
            # ----------------------------------------------------
            solver = jaxopt.LBFGS(
                fun=loss_fn,
                maxiter=200,           # tweak as needed
                tol=1e-12,              # stopping tolerance
                verbose=False
            )


            def optimize_u0_fourier(u0_init):
                # BFGS works directly on the same PyTree / array shape as u0_init
                params_opt, state = solver.run(u0_init)
                final_loss = state.value
                return params_opt, final_loss

            # ----------------------------------------------------
            # Fourier template / shape (as before)
            # ----------------------------------------------------
            fourier_template = self.vel_part_trans.vel_flat_2_vel_Fourier(U_0)
            fourier_shape = fourier_template.shape

            # ----------------------------------------------------
            # Multiple random inits, pick best
            # ----------------------------------------------------
            key = jax.random.PRNGKey(0)
            u_0_fourier_guess = jax.random.normal(key, fourier_shape)
            u_0_fourier_opt, final_loss = optimize_u0_fourier(u_0_fourier_guess)

            # Map back to physical-space initial condition
            u_0_opt = self.transform_fn(u_0_fourier_opt)
            DA_loss = DA_loss_fn(u_0_opt)
            print(DA_loss, final_loss)
            if best_DA_loss is None:
                best_DA_loss = DA_loss
                best_u_0 = u_0_opt
            elif DA_loss < best_DA_loss:
                best_DA_loss = DA_loss
                best_u_0 = u_0_opt
            else:
                break
            

        norm_dist = jnp.linalg.norm(best_u_0 - U_0)/self.attractor_rad
        return best_u_0, norm_dist
                        