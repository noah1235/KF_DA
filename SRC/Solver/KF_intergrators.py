
import jax
import jax.numpy as jnp
from jax import lax
from SRC.utils import bilinear_sample_periodic, Specteral_Upsampling, Vel_Part_Transformations, Vel_Reshaper
from SRC.vp_floats.vp_py_utils import calc_output_shape
from dataclasses import dataclass
import numpy as np
from functools import partial
import vpfloat
from jax import ShapeDtypeStruct

def run_particle_advection_with_sens(U_hat_0, part_IC, rhs, dt, T, V):
    nsteps = int(T/dt)
    X0 = jnp.concat([part_IC, U_hat_0.reshape(-1)])
    integrator = Time_Stepper(rhs, dt, method="RK4", n_particles=rhs.n_particles)
    trj, sens_trj = integrator.integrate_scan_with_sens(X0, nsteps, V=V)
    return trj, sens_trj

def create_trj_sens_generator(rhs, dt, T):
    def trj_gen(pIC, U_hat_0, V):
        return run_particle_advection_with_sens(U_hat_0, pIC, rhs, dt, T, V)
    return trj_gen

def create_trj_generator(rhs, dt, T, dtype=jnp.float64):
    def trj_gen(pIC, U_hat_0):
        nsteps = int(T/dt)
        X0 = jnp.concat([pIC, U_hat_0.reshape(-1)])
        integrator = Time_Stepper(rhs, dt, method="RK4", n_particles=rhs.n_particles)
        trj = integrator.integrate_scan(X0, nsteps, dtype=dtype)
        return trj
    return trj_gen

def create_trj_generator_vpfloat(rhs, g, dt, T, exp_bits, exp_bias, mbits):
    def trj_gen(pIC, U_hat_0):
        nsteps = int(T/dt)
        X0 = jnp.concat([pIC, U_hat_0.reshape(-1)])
        integrator = Time_Stepper(rhs, dt, method="RK4", n_particles=rhs.n_particles)
        sign_trj, exp_trj, mant_trj, loss = integrator.integrate_scan_vp_save(X0, g, nsteps, exp_bits, exp_bias, mbits)
        return sign_trj, exp_trj, mant_trj, loss
    return trj_gen

def split_f64_cb(U, exp_bits, exp_bias, mbits):
    # runs on host (Python)
    sign, exp, mant, _ = vpfloat.split_f64(
        U, exp_bits, exp_bias, mbits
    )
    return sign, exp, mant

class RK4_Step:
    def __init__(self, rhs, dt):
        self.dt = dt
        self.rhs = rhs
    def __call__(self, X):
        dt = self.dt
        k1 = self.rhs(X)
        k2 = self.rhs(X + 0.5 * dt * k1)
        k3 = self.rhs(X + 0.5 * dt * k2)
        k4 = self.rhs(X + dt * k3)
        X_next = X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return X_next
    
class Particle_Stepper:
    def __init__(self, step, n_particles):
        self.step = step
        self.n_particles = n_particles
    
    def __call__(self, X):
        X_next = self.step(X)
        xp = X_next[:self.n_particles*4:4].real
        yp = X_next[1:self.n_particles*4:4].real

        xp = jnp.mod(xp, 2*jnp.pi)
        yp = jnp.mod(yp, 2*jnp.pi)
        n4 = self.n_particles * 4
        X_next = X_next.at[0:n4:4].set(xp)  # set x entries
        X_next = X_next.at[1:n4:4].set(yp)  # set y entries
        return X_next

class Time_Stepper:
    def __init__(self, rhs, dt, method="RK4", n_particles=None):
        """
        rhs : callable mapping U_hat -> RHS_hat (same shape)
        dt  : time step
        """
        self.rhs = rhs
        self.dt = dt
        self.n_particles = n_particles
        if method == "RK4":
            #self.step_jit = jax.jit(RK4_Step(rhs, dt))
            self.step = RK4_Step(rhs, dt)
        if n_particles is not None:
            self.step = Particle_Stepper(self.step, n_particles)
    
    
    def integrate(self, U_hat0, nsteps, callback=None):
        """
        Integrate for nsteps. Optionally pass a callback(n, U_hat) every step.
        """
        U = U_hat0
        for n in range(1, nsteps + 1):
            U = self.step(U)
            if callback is not None:
                callback(n, U)
        return U

    def integrate_scan(self, U0, nsteps, dtype=jnp.float64):
        """
        Returns (U_final, traj) where
        traj has shape (nsteps, *U0.shape) and stores U at each step.
        """
        def body(U, _):
            U_next = self.step(U)
            
            return U_next, U_next.astype(dtype)  # carry, y

        U_f, trj = lax.scan(body, U0, xs=None, length=nsteps)
        trj = jnp.concatenate([U0[None, ...].astype(dtype), trj], axis=0)
        return trj
    
    def integrate_scan_vp_save(self, U0, g, nsteps, exp_bits, exp_bias, mbits):
        sign_shape, exp_shape, m_shape = calc_output_shape(len(U0), mbits, exp_bits)
        # output specs (must match vpfloat output exactly)
        sign_spec = ShapeDtypeStruct(
            shape=sign_shape, dtype=jnp.uint8
        )
        exp_spec = ShapeDtypeStruct(
            shape=exp_shape, dtype=jnp.uint8
        )
        if mbits == 16:
            mant_spec = ShapeDtypeStruct(
                shape=m_shape, dtype=jnp.uint16
            )
        else:
            mant_spec = ShapeDtypeStruct(
                shape=m_shape, dtype=jnp.uint8
            )

        def body(carry, _):
            U, loss, i = carry
            i += 1
            U_next = self.step(U)
            loss += g(U_next, i)

            sign, exp, mant = jax.pure_callback(
                split_f64_cb,
                (sign_spec, exp_spec, mant_spec),
                U_next, exp_bits, exp_bias, mbits
            )
            return (U_next, loss, i), (sign, exp, mant)

        (U_f, loss, _), (sign_trj, exp_trj, mant_trj) = lax.scan(
            body, (U0, g(U0, 0), 0), xs=None, length=nsteps
        )

        # also save t = 0
        sign0, exp0, mant0 = jax.pure_callback(
            split_f64_cb,
            (sign_spec, exp_spec, mant_spec),
            U0, exp_bits, exp_bias, mbits
        )

        sign_trj = jnp.concatenate([sign0[None, ...], sign_trj], axis=0)
        exp_trj  = jnp.concatenate([exp0[None, ...],  exp_trj],  axis=0)
        mant_trj = jnp.concatenate([mant0[None, ...], mant_trj], axis=0)

        return sign_trj, exp_trj, mant_trj, loss
  
    def integrate_scan_checkpoint(self, U0, nsteps, chunk_size, path, dtype=np.float32):
        U0 = jnp.asarray(U0)
        if nsteps % chunk_size != 0:
            raise ValueError("nsteps must be divisible by chunk size")

        nsteps = int(nsteps)
        chunk_size = int(chunk_size)
        ndof = int(U0.shape[0])

        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=(nsteps, ndof)
        )

        @partial(jax.jit, static_argnums=(1,))
        def run_checkpoint(U, L):
            def body(carry, _):
                U_next = self.step(carry)
                return U_next, U_next
            U_end, trj = lax.scan(body, U, xs=None, length=L)  # trj: (L, ndof)
            return U_end, trj

        remaining = nsteps
        U = U0
        write_idx = 0

        while remaining > 0:
            U, trj = run_checkpoint(U, chunk_size)

            trj_np = np.asarray(jax.device_get(trj), dtype=dtype)  # shape (L, ndof)
            mm[write_idx:write_idx + chunk_size] = trj_np

            write_idx += chunk_size
            remaining -= chunk_size

            # Optional: flush less often for speed; keep as-is if you want safety
            mm.flush()

        return path

    
    def integrate_scan_with_sens(self, U0, nsteps, V=None):
        """
        Returns (traj, sens_trj) where:
          - traj has shape (nsteps+1, *U0.shape) and stores U_k for k=0..nsteps
          - sens_trj has shape (nsteps+1, Udim, Udim) and stores dU_k/dU_0
            (the cumulative sensitivity/Jacobian wrt the initial state)
        """
        Udim = U0.size
        if V is None:
            V = jnp.eye(Udim, dtype=U0.dtype)

        def apply_lin_to_matrix(lin_fn, M):
            # lin_fn: R^{Udim} -> R^{Udim}, M: (Udim, Udim) (columns = directions)
            # returns lin_fn(M) columnwise -> (Udim, Udim)
            return jax.vmap(lin_fn, in_axes=1, out_axes=1)(M)

        def body(carry, _):
            U_k, J_k = carry                            # J_k = dU_k/dU_0
            U_kp1, lin = jax.linearize(self.step, U_k)  # lin(v) = (dU_{k+1}/dU_k) @ v

            #A_k = apply_lin_to_matrix(lin, jnp.eye(Udim, dtype=U0.dtype))  # dU_{k+1}/dU_k
            #J_kp1 = A_k @ J_k                         # chain rule: dU_{k+1}/dU_0 = A_k @ J_k
            J_kp1 = apply_lin_to_matrix(lin, J_k)

            return (U_kp1, J_kp1), (U_kp1, J_kp1)      # carry, y

        (U_last, J_last), (traj_noinit, sens_noinit) = lax.scan(
            body, (U0, V), xs=None, length=nsteps
        )

        traj = jnp.concatenate([U0[None, ...], traj_noinit], axis=0)     # (nsteps+1, ...)
        sens_trj = jnp.concatenate([V[None, ...], sens_noinit], axis=0)  # (nsteps+1, Udim, Udim)

        return traj, sens_trj

class KF_PS_RHS:
    """
    Kolmogorov flow spectral RHS using rfft2/irfft2 (half-spectrum in x).
    Internal spectral arrays are shaped (Ny, Nx_r).
    Externally, we accept/return a flat vector of length 2 * Ny * Nx_r (u_hat_r, v_hat_r).
    """
    def __init__(self, NDOF, Re, n, calc_mat_deriv=False):
        dealias=True
        self.N = NDOF
        self.Re = Re
        self.n = int(n)
        self.L = 2 * jnp.pi

        self.vel_reshaper = Vel_Reshaper(NDOF)

        ord = "xy"

        # rfft wavenumbers & operators
        kx = jnp.fft.rfftfreq(self.N, d=self.L / (self.N * 2 * jnp.pi))
        ky = jnp.fft.fftfreq(self.N, d=self.L / (self.N * 2 * jnp.pi))
        self.KX, self.KY = jnp.meshgrid(kx, ky, indexing=ord)

        self.K2   = self.KX**2 + self.KY**2
        self.dxop = 1j * self.KX
        self.dyop = 1j * self.KY

        # forcing f = (sin(n y), 0)
        y = jnp.linspace(0.0, self.L, self.N, endpoint=False)
        x = jnp.linspace(0.0, self.L, self.N, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing=ord)
        fx = jnp.sin(self.n * Y)
        fy = jnp.zeros_like(fx)
        self.f_hat_x = jnp.fft.rfft2(fx)               # (Ny, Nx_r)
        self.f_hat_y = jnp.fft.rfft2(fy)
        self.fx = fx

        # 2/3 dealias mask in rfft layout
        if dealias:
            mx_full = jnp.fft.fftfreq(self.N) * self.N
            my_full = jnp.fft.fftfreq(self.N) * self.N
            MX, MY = jnp.meshgrid(mx_full, my_full, indexing=ord)
            M_full = (jnp.abs(MX) <= self.N/3) & (jnp.abs(MY) <= self.N/3)
            #self.M = M_full[:, :self.N//2 + 1].astype(jnp.complex128)  # (Ny, Nx_r)
            self.M = M_full[:, :self.N//2 + 1].astype(jnp.complex64)  # (Ny, Nx_r)
        else:
            #self.M = jnp.ones((self.N, self.N//2 + 1), dtype=jnp.complex128)
            self.M = jnp.ones((self.N, self.N//2 + 1), dtype=jnp.complex64)

        self.calc_mat_deriv = calc_mat_deriv

    # optional: vorticity in real space from rfft spectra
    def vorticity_real(self, u_hat_r, v_hat_r):
        vort_hat = self.dxop * v_hat_r - self.dyop * u_hat_r
        return jnp.fft.irfft2(vort_hat, s=(self.N, self.N)).real

    def get_pressure_hat(self, conv_x_hat, conv_y_hat):
        num = self.dxop * (conv_x_hat - self.f_hat_x) + self.dyop * (conv_y_hat - self.f_hat_y)
        denom = jnp.where(self.K2 == 0.0, 1.0, self.K2)
        p_hat = num / denom
        p_hat = p_hat.at[0, 0].set(0.0)  # zero-mean
        return p_hat

    def __call__(self, U_flat):
        U_hat, U = self.vel_reshaper.get_vel_hat_from_flat(U_flat)
        u_hat = U_hat[0]
        u = U[0]
        v_hat = U_hat[1]
        v = U[1]
        
        # gradients in spectral → real
        ux = jnp.fft.irfft2(self.dxop * u_hat, s=(self.N, self.N))
        uy = jnp.fft.irfft2(self.dyop * u_hat, s=(self.N, self.N))
        vx = jnp.fft.irfft2(self.dxop * v_hat, s=(self.N, self.N))
        vy = jnp.fft.irfft2(self.dyop * v_hat, s=(self.N, self.N))

        # convective terms in real
        conv_x = u*ux + v*uy
        conv_y = u*vx + v*vy

        # back to spectral (rfft) + dealias
        conv_x_hat = jnp.fft.rfft2(conv_x)
        conv_y_hat = jnp.fft.rfft2(conv_y)

        # pressure & viscosity
        p_hat = self.get_pressure_hat(conv_x_hat, conv_y_hat)
        visc  = (1.0 / self.Re) * self.K2

        RHS_u = jnp.fft.irfft2((-conv_x_hat - self.dxop * p_hat - visc * u_hat + self.f_hat_x) * self.M, s=(self.N, self.N))
        RHS_v = jnp.fft.irfft2((-conv_y_hat - self.dyop * p_hat - visc * v_hat + self.f_hat_y) * self.M, s=(self.N, self.N))

        RHS = self.vel_reshaper.flatten_from_comps(RHS_u, RHS_v)

        if self.calc_mat_deriv:
            u_mat = jnp.fft.irfft2((- self.dxop * p_hat - visc * u_hat + self.f_hat_x)*self.M, s=(self.N, self.N))
            v_mat = jnp.fft.irfft2((- self.dyop * p_hat - visc * v_hat + self.f_hat_y)*self.M, s=(self.N, self.N))
            return RHS, u_mat, v_mat
        else:
            return RHS

class Maxey_Riley_RHS:
    def __init__(self, beta, St, L):
        self.a = 1.0 / (1.0 + beta/2.0)
        self.alpha = -self.a / St
        self.St_inv = 1.0 / St
        self.beta = beta
        self.L = L

    def __call__(self, u, v, xp, yp, up, vp, u_t_field, v_t_field):
        u_mat = bilinear_sample_periodic(u_t_field, xp, yp, self.L, self.L)
        v_mat = bilinear_sample_periodic(v_t_field, xp, yp, self.L, self.L)

        hx = self.a * (self.St_inv * u + 1.5 * self.beta * u_mat)
        hy = self.a * (self.St_inv * v + 1.5 * self.beta * v_mat)

        up_dot_rhs = self.alpha * up + hx
        vp_dot_rhs = self.alpha * vp + hy

        return up_dot_rhs, vp_dot_rhs

class KF_LPT_PS_RHS:
    def __init__(self, NDOF, Re, n, n_particles, beta, St, vel_fine_NDOF=512):
        self.KF_RHS = KF_PS_RHS(NDOF, Re, n, calc_mat_deriv=True)
        self.vel_part_trans = Vel_Part_Transformations(NDOF, n_particles)
        if beta == 0 and St == 0:
            self.tracer_parts = True
        else:
            maxey_riley_rhs = Maxey_Riley_RHS(beta, St, self.KF_RHS.L)
            self.particle_GE = maxey_riley_rhs
            self.tracer_parts = False

        self.N = NDOF
        self.n_particles = n_particles
        self.r = int(round(vel_fine_NDOF / NDOF))

    

    def __call__(self, X):
        # unpack
        part, U_flat = self.vel_part_trans.split_part_and_vel(X)
        U = self.vel_part_trans.reshape_flattened_vel(U_flat)
        xp, yp, up, vp = self.vel_part_trans.get_part_pos_and_vel(part)

        xp = jnp.mod(xp, self.KF_RHS.L)
        yp = jnp.mod(yp, self.KF_RHS.L)

        # flow RHS & material derivative fields (real)
        KF_rhs_eval, u_t_field, v_t_field = self.KF_RHS(U_flat)

        # upsample to a finer grid for particle sampling
        u_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(U[0]), self.r)
        v_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(U[1]), self.r)

        # bilinear samples (periodic) at particle positions
        u = bilinear_sample_periodic(u_fine, xp, yp, self.KF_RHS.L, self.KF_RHS.L)
        v = bilinear_sample_periodic(v_fine, xp, yp, self.KF_RHS.L, self.KF_RHS.L)
        if self.tracer_parts:
            px = u
            py = v
            up_dot_rhs = (u - up) * 0
            vp_dot_rhs = (v - vp) * 0
            
        else:
            up_dot_rhs, vp_dot_rhs = self.particle_GE(u, v, xp, yp, up, vp, u_t_field, v_t_field)
            px = up
            py = vp

        # particle RHS: [dx/dt, dy/dt, du_p/dt, dv_p/dt]
        particle_rhs = jnp.stack([px, py, up_dot_rhs, vp_dot_rhs], axis=1).reshape(-1)

        return jnp.concatenate([particle_rhs, KF_rhs_eval])


