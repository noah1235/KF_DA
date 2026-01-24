
import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
from SRC.utils import bilinear_sample_periodic
from dataclasses import dataclass
from SRC.vp_floats.vp_py_utils import calc_output_shape
from jax import ShapeDtypeStruct
import vpfloat
#for animation
def create_vel_part_gen_fn(stepper, T):
    nsteps = int(T/stepper.dt)
    def body(carry, _):
        omega_hat, xp, yp, up, vp = carry

        # advance one step
        omega_hat, xp, yp, up, vp = stepper(omega_hat, xp, yp, up, vp)

        # compute grid velocity from omega_hat
        u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega_hat)
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)

        new_carry = (omega_hat, xp, yp, up, vp)
        y = (u, v, xp, yp)  # saved outputs
        return new_carry, y
    def vel_part_trj_gen_fn(omega0_hat, xp, yp, up, vp):
        carry0 = (omega0_hat, xp, yp, up, vp)
        _, (u_traj, v_traj, xp_traj, yp_traj) = jax.lax.scan(
            body, carry0, xs=None, length=nsteps
        )
        return u_traj, v_traj, xp_traj, yp_traj
    
    return vel_part_trj_gen_fn


def create_omega_part_gen_fn(stepper, T):
    nsteps = int(T / stepper.dt)

    def body(carry, _):
        omega_hat, xp, yp, up, vp = carry

        # advance one step
        omega_hat, xp, yp, up, vp = stepper(omega_hat, xp, yp, up, vp)

        new_carry = (omega_hat, xp, yp, up, vp)
        y = (omega_hat, xp, yp, up, vp)  # saved outputs (after step)
        return new_carry, y

    def omega_part_trj_gen_fn(omega0_hat, xp0, yp0, up0, vp0):
        carry0 = (omega0_hat, xp0, yp0, up0, vp0)

        _, (omega_traj, xp_traj, yp_traj, up_traj, vp_traj) = jax.lax.scan(
            body, carry0, xs=None, length=nsteps
        )

        # Prepend initial conditions so trajectories are length nsteps+1
        omega_traj = jnp.concatenate([omega0_hat[None, ...], omega_traj], axis=0)
        xp_traj    = jnp.concatenate([xp0[None, ...],       xp_traj],    axis=0)
        yp_traj    = jnp.concatenate([yp0[None, ...],       yp_traj],    axis=0)
        up_traj    = jnp.concatenate([up0[None, ...],       up_traj],    axis=0)
        vp_traj    = jnp.concatenate([vp0[None, ...],       vp_traj],    axis=0)

        return omega_traj, xp_traj, yp_traj, up_traj, vp_traj

    return omega_part_trj_gen_fn


class Forced_2D_NS:
    L = 2 * jnp.pi

    def __init__(self, Re, n, N):
        self.Re = Re

        KX, KY = self.get_K(N)
        self.dxop = 1j * KX
        self.dyop = 1j * KY

        laplacian_op = (self.dxop**2 + self.dyop**2)
        self.diff_op = (1 / Re) * laplacian_op
        self.laplacian_op_safe = laplacian_op.at[0, 0].set(1)

        forcing = self.kolmogorov_vorticity_forcing(self.L, N, n)  
        self.forcing_hat = jnp.fft.rfft2(forcing)  

        self.M = self.get_dealias_mask(N)          

    @staticmethod
    def get_K(N):
        L = Forced_2D_NS.L
        dx = L / N
        kx = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=dx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
        KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
        return KX, KY
    
    @staticmethod
    def kolmogorov_vorticity_forcing(L, N, n):
        y = jnp.linspace(0.0, L, N, endpoint=False)
        x = jnp.linspace(0.0, L, N, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        return -n * jnp.cos(n * Y)

    @staticmethod
    def get_dealias_mask(N):
        mx_full = jnp.fft.fftfreq(N) * N
        my_full = jnp.fft.fftfreq(N) * N
        MX, MY = jnp.meshgrid(mx_full, my_full, indexing="xy")
        M_full = (jnp.abs(MX) <= N/3) & (jnp.abs(MY) <= N/3)
        return M_full[:, :N//2 + 1]

    def vort_hat_2_vel_hat(self, omega_hat):
        psi_hat = omega_hat / self.laplacian_op_safe   
        u_hat = self.dyop * psi_hat
        v_hat = -self.dxop * psi_hat
        return u_hat, v_hat

    def explicit_term(self, omega_hat):
        u_hat, v_hat = self.vort_hat_2_vel_hat(omega_hat)
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)

        dw_dx = jnp.fft.irfft2(self.dxop * omega_hat)
        dw_dy = jnp.fft.irfft2(self.dyop * omega_hat)

        adv = -(u * dw_dx + v * dw_dy)
        adv_hat = jnp.fft.rfft2(adv)
        adv_hat = adv_hat * self.M
        return adv_hat + self.forcing_hat, u, v
    
    def implicit_term(self, omega_hat):
        return self.diff_op * omega_hat
    
    def implicit_solve(self, omega_hat, mu):
        return omega_hat / (1 - mu * self.diff_op)
    
class KF_Stepper:
    alpha = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
    beta  = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
    gamma = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

    def __init__(self, Re, n, N, dt):
        self.NS = Forced_2D_NS(Re, n, N)
        self.dt = dt

    def calc_h(self, g, h, i):
        return g + self.beta[i] * h 

    def calc_mu(self, i):
        return 0.5 * self.dt * (self.alpha[i+1] - self.alpha[i])
    
    def calc_imp_rhs(self, h, mu, u, i):
        return u + self.gamma[i] * self.dt * h + mu * self.NS.implicit_term(u)

    def __call__(self, u_n): 
        u = u_n
        h = jnp.zeros_like(u_n)
        for i in range(5):
            g, _, _ = self.NS.explicit_term(u)   # g(u, t_k)
            h = self.calc_h(g, h, i)             # h <- g + beta*h

            mu = self.calc_mu(i)
            rhs = self.calc_imp_rhs(h, mu, u, i)
            u = self.NS.implicit_solve(rhs, mu)

        return u
    
class Tracer_Evolution:
    @staticmethod
    def part_pos_update(h_p, pos, fluid_vel, beta_coef, gamma_coef, dt):
        h_p = fluid_vel + beta_coef * h_p
        pos = pos + gamma_coef * dt * h_p
        return pos, h_p
    
    def __call__(
            self, 
            xp, yp, h_xp, h_yp, 
            up, vp, h_up, h_vp,
            u_fluid, v_fluid, 
            beta_coef, gamma_coef, dt
    ):
        xp, h_xp = self.part_pos_update(h_xp, xp, u_fluid, beta_coef, gamma_coef, dt)
        yp, h_yp = self.part_pos_update(h_yp, yp, v_fluid, beta_coef, gamma_coef, dt)
        return xp, yp, h_xp, h_yp, up, vp, h_up, h_vp

class Inertial_Evolution:
    def __init__(self, St):
        self.St = St

    def state_update(self, h_p, pos, h_vel, part_vel, fluid_vel, beta_coef, gamma_coef, dt):
        h_vel = (fluid_vel - part_vel) / self.St + beta_coef * h_vel
        h_p = part_vel + beta_coef * h_p

        part_vel = part_vel + gamma_coef * dt * h_vel
        pos = pos + gamma_coef * dt * h_p
        return pos, h_p, part_vel, h_vel
    
    def __call__(
            self, 
            xp, yp, h_xp, h_yp, 
            up, vp, h_up, h_vp,
            u_fluid, v_fluid, 
            beta_coef, gamma_coef, dt
    ):
        xp, h_xp, up, h_up = self.state_update(h_xp, xp, h_up, up, u_fluid, beta_coef, gamma_coef, dt)
        yp, h_yp, vp, h_vp = self.state_update(h_yp, yp, h_vp, vp, v_fluid, beta_coef, gamma_coef, dt)

        return xp, yp, h_xp, h_yp, up, vp, h_up, h_vp

#KF and Tracer Particles
class KF_TP_Stepper(KF_Stepper):
    def __init__(self, Re, n, N, dt, St, beta, npart):
        super().__init__(Re, n, N, dt)
        if St == 0 and beta == 0:
            self.p_ev = Tracer_Evolution()
        else:
            self.p_ev = Inertial_Evolution(St)
        
        self.h = jnp.zeros((N, N//2+1))
        self.h_xp = jnp.zeros(npart)
        self.h_yp = jnp.zeros(npart)
        self.h_up = jnp.zeros(npart)
        self.h_vp = jnp.zeros(npart)

    def part_pos_update(self, i, h_p, pos, fluid_vel):
        h_p = fluid_vel + self.beta[i] * h_p
        pos = pos + self.gamma[i] * self.dt * h_p
        return pos, h_p

    def __call__(self, omega_hat, xp, yp, up, vp):
        h = self.h * 0
        h_xp = self.h_xp * 0
        h_yp = self.h_yp * 0
        h_up = self.h_up * 0
        h_vp = self.h_vp * 0

        for i in range(5):
            g, u_grid, v_grid = self.NS.explicit_term(omega_hat)
            h = self.calc_h(g, h, i)

            mu = self.calc_mu(i)
            rhs = self.calc_imp_rhs(h, mu, omega_hat, i)
            omega_hat = self.NS.implicit_solve(rhs, mu)

            # sample fluid velocity at particle positions
            u = bilinear_sample_periodic(u_grid, xp, yp, self.NS.L, self.NS.L)
            v = bilinear_sample_periodic(v_grid, xp, yp, self.NS.L, self.NS.L)

            xp, yp, h_xp, h_yp, up, vp, h_up, h_vp = self.p_ev(
                xp, yp, h_xp, h_yp, 
                up, vp, h_up, h_vp,
                u, v, 
                self.beta[i], self.gamma[i], self.dt
            )

        xp = jnp.mod(xp, self.NS.L)
        yp = jnp.mod(yp, self.NS.L)
        return omega_hat, xp, yp, up, vp

class Omega_Integrator:
    def __init__(self, stepper):
        self.stepper = stepper

    def fv_integrate(self, u0, n):
        def body(u, _):
            u_next = self.stepper(u)
            return u_next, None

        u_final, _ = jax.lax.scan(body, u0, xs=None, length=n)
        return u_final
    
    def integrate_scan_checkpoint(self, u0, nsteps, chunk_size, path, dtype=np.complex128):
        u0 = jnp.asarray(u0)
        if nsteps % chunk_size != 0:
            raise ValueError("nsteps must be divisible by chunk size")

        nsteps = int(nsteps)
        chunk_size = int(chunk_size)

        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=(nsteps, *u0.shape)
        )

        @partial(jax.jit, static_argnums=(1,))
        def run_checkpoint(U, L):
            def body(carry, _):
                U_next = self.stepper(carry)
                return U_next, U_next
            U_end, trj = jax.lax.scan(body, U, xs=None, length=L)  # trj: (L, ndof)
            return U_end, trj

        remaining = nsteps
        U = u0
        write_idx = 0

        while remaining > 0:
            U, trj = run_checkpoint(U, chunk_size)

            trj_np = np.asarray(jax.device_get(trj), dtype=dtype)  # shape (L, ndof)
            print
            mm[write_idx:write_idx + chunk_size] = trj_np

            write_idx += chunk_size
            remaining -= chunk_size

            mm.flush()

        return path
    
    def integrate_scan(self, U0, nsteps):
        """
        Returns (U_final, traj) where
        traj has shape (nsteps, *U0.shape) and stores U at each step.
        """
        def body(U, _):
            U_next = self.stepper(U)
            
            return U_next, U_next  # carry, y

        U_f, trj = jax.lax.scan(body, U0, xs=None, length=nsteps)
        trj = jnp.concatenate([U0[None, ...], trj], axis=0)
        return trj
    