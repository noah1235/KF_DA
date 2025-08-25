
import jax
import jax.numpy as jnp
from jax import lax
from SRC.utils import bilinear_sample_periodic, Specteral_Upsampling

def run_particle_advection(U_hat_0, part_IC, rhs, dt, T):
    nsteps = int(T/dt)
    X0 = jnp.concat([part_IC, U_hat_0.reshape(-1)])
    integrator = Time_Stepper(rhs, dt, method="RK4", n_particles=rhs.n_particles)
    trj = integrator.integrate_scan(X0, nsteps)
    return trj

def run_particle_advection_with_sens(U_hat_0, part_IC, rhs, dt, T):
    nsteps = int(T/dt)
    X0 = jnp.concat([part_IC, U_hat_0.reshape(-1)])
    integrator = Time_Stepper(rhs, dt, method="RK4", n_particles=rhs.n_particles)
    trj, sens_trj = integrator.integrate_scan_with_sens(X0, nsteps)
    return trj, sens_trj

def create_trj_sens_generator(rhs, dt, T):
    def trj_gen(pIC, U_hat_0):
        return run_particle_advection_with_sens(U_hat_0, pIC, rhs, dt, T)
    return trj_gen

def create_trj_generator(rhs, dt, T):
    def trj_gen(pIC, U_hat_0):
        return run_particle_advection(U_hat_0, pIC, rhs, dt, T)
    return trj_gen


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

    
# ---------- RK4 Integrator ----------
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

    
    def integrate_scan(self, U0, nsteps):
        """
        Returns (U_final, traj) where
        traj has shape (nsteps, *U0.shape) and stores U at each step.
        """
        def body(U, _):
            U_next = self.step(U)
            
            return U_next, U_next  # carry, y

        _, trj = lax.scan(body, U0, xs=None, length=nsteps)
        trj = jnp.concatenate([U0[None, ...], trj], axis=0)
        return trj
    
    def integrate_scan_with_sens(self, U0, nsteps):
        """
        Returns (traj, sens_trj) where:
          - traj has shape (nsteps+1, *U0.shape) and stores U_k for k=0..nsteps
          - sens_trj has shape (nsteps+1, Udim, Udim) and stores dU_k/dU_0
            (the cumulative sensitivity/Jacobian wrt the initial state)
        """
        Udim = U0.size
        I = jnp.eye(Udim, dtype=U0.dtype)

        def apply_lin_to_matrix(lin_fn, M):
            # lin_fn: R^{Udim} -> R^{Udim}, M: (Udim, Udim) (columns = directions)
            # returns lin_fn(M) columnwise -> (Udim, Udim)
            return jax.vmap(lin_fn, in_axes=1, out_axes=1)(M)

        def body(carry, _):
            U_k, J_k = carry                            # J_k = dU_k/dU_0
            U_kp1, lin = jax.linearize(self.step, U_k)  # lin(v) = (dU_{k+1}/dU_k) @ v

            A_k = apply_lin_to_matrix(lin, jnp.eye(Udim, dtype=U0.dtype))  # dU_{k+1}/dU_k
            J_kp1 = A_k @ J_k                         # chain rule: dU_{k+1}/dU_0 = A_k @ J_k

            return (U_kp1, J_kp1), (U_kp1, J_kp1)      # carry, y

        (U_last, J_last), (traj_noinit, sens_noinit) = lax.scan(
            body, (U0, I), xs=None, length=nsteps
        )

        traj = jnp.concatenate([U0[None, ...], traj_noinit], axis=0)     # (nsteps+1, ...)
        sens_trj = jnp.concatenate([I[None, ...], sens_noinit], axis=0)  # (nsteps+1, Udim, Udim)

        return traj, sens_trj


class KF_PS_RHS:
    """
    Kolmogorov flow spectral RHS using rfft2/irfft2 (half-spectrum in x).
    Internal spectral arrays are shaped (Ny, Nx_r).
    Externally, we accept/return a flat vector of length 2 * Ny * Nx_r (u_hat_r, v_hat_r).
    """
    def __init__(self, NDOF, Re, n, dealias=True, calc_mat_deriv=False):
        self.N = NDOF
        Ny = Nx = NDOF
        self.Ny, self.Nx = Ny, Nx
        self.Re = Re
        self.n = int(n)
        self.Lx = self.Ly = 2 * jnp.pi
        dx, dy = self.Lx / Nx, self.Ly / Ny

        # rfft wavenumbers & operators
        kx = 2 * jnp.pi * jnp.fft.rfftfreq(Nx, d=dx)   # length Nx_r
        ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)    # length Ny
        self.KX, self.KY = jnp.meshgrid(kx, ky, indexing="xy")

        self.K2   = self.KX**2 + self.KY**2
        self.dxop = 1j * self.KX
        self.dyop = 1j * self.KY

        # forcing f = (sin(n y), 0)
        y = jnp.linspace(0.0, self.Ly, Ny, endpoint=False)
        x = jnp.linspace(0.0, self.Lx, Nx, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        fx = jnp.sin(self.n * Y)
        fy = jnp.zeros_like(fx)
        self.f_hat_x = jnp.fft.rfft2(fx)               # (Ny, Nx_r)
        self.f_hat_y = jnp.fft.rfft2(fy)
        self.fx = fx

        # 2/3 dealias mask in rfft layout
        if dealias:
            mx_full = jnp.fft.fftfreq(Nx) * Nx
            my_full = jnp.fft.fftfreq(Ny) * Ny
            MX, MY = jnp.meshgrid(mx_full, my_full, indexing="xy")
            M_full = (jnp.abs(MX) <= Nx/3) & (jnp.abs(MY) <= Ny/3)
            self.M = M_full[:, :Nx//2 + 1].astype(jnp.complex128)  # (Ny, Nx_r)
        else:
            self.M = jnp.ones((Ny, Nx//2 + 1), dtype=jnp.complex128)

        self.calc_mat_deriv = calc_mat_deriv

    # optional: vorticity in real space from rfft spectra
    def vorticity_real(self, u_hat_r, v_hat_r):
        Ny, Nx = self.Ny, self.Nx
        vort_hat = self.dxop * v_hat_r - self.dyop * u_hat_r
        return jnp.fft.irfft2(vort_hat, s=(Ny, Nx)).real

    def get_pressure_hat(self, conv_x_hat, conv_y_hat):
        num = self.dxop * (conv_x_hat - self.f_hat_x) + self.dyop * (conv_y_hat - self.f_hat_y)
        denom = jnp.where(self.K2 == 0.0, 1.0, self.K2)
        p_hat = num / denom
        p_hat = p_hat.at[0, 0].set(0.0)  # zero-mean
        return p_hat

    def __call__(self, U_flat):
        Ny, Nx = self.Ny, self.Nx
        U = U_flat.reshape((2, Ny, Nx))
        u_hat = jnp.fft.rfft2(U[0])
        u = U[0]
        v_hat = jnp.fft.rfft2(U[1])
        v = U[1]
        
        # gradients in spectral → real
        ux = jnp.fft.irfft2(self.dxop * u_hat, s=(Ny, Nx))
        uy = jnp.fft.irfft2(self.dyop * u_hat, s=(Ny, Nx))
        vx = jnp.fft.irfft2(self.dxop * v_hat, s=(Ny, Nx))
        vy = jnp.fft.irfft2(self.dyop * v_hat, s=(Ny, Nx))

        # convective terms in real
        conv_x = u*ux + v*uy
        conv_y = u*vx + v*vy

        # back to spectral (rfft) + dealias
        conv_x_hat = jnp.fft.rfft2(conv_x)
        conv_y_hat = jnp.fft.rfft2(conv_y)

        # pressure & viscosity
        p_hat = self.get_pressure_hat(conv_x_hat, conv_y_hat)
        visc  = (1.0 / self.Re) * self.K2

        RHS_u = jnp.fft.irfft2((-conv_x_hat - self.dxop * p_hat - visc * u_hat + self.f_hat_x) * self.M, s=(Ny, Nx))
        RHS_v = jnp.fft.irfft2((-conv_y_hat - self.dyop * p_hat - visc * v_hat + self.f_hat_y) * self.M, s=(Ny, Nx))

        RHS = jnp.stack([RHS_u, RHS_v], axis=0).reshape(-1)

        if self.calc_mat_deriv:
            u_mat = jnp.fft.irfft2((- self.dxop * p_hat - visc * u_hat + self.f_hat_x)*self.M, s=(Ny, Nx))
            v_mat = jnp.fft.irfft2((- self.dyop * p_hat - visc * v_hat + self.f_hat_y)*self.M, s=(Ny, Nx))
            return RHS, u_mat, v_mat
        else:
            return RHS

class KF_LPT_PS_RHS:
    def __init__(self, NDOF, Re, n, n_particles, beta, St, vel_fine_NDOF=256):
        self.KF_RHS = KF_PS_RHS(NDOF, Re, n, dealias=True, calc_mat_deriv=True)
        self.a = 1.0 / (1.0 + beta/2.0)
        self.alpha = -self.a / St
        self.St_inv = 1.0 / St
        self.beta = beta
        self.n_particles = n_particles
        self.N = NDOF

        self.r = int(round(vel_fine_NDOF / NDOF))

    def __call__(self, X):
        Ny = Nx = self.N
        # unpack
        part = X[:self.n_particles * 4]
        U_flat = X[self.n_particles * 4:]
        U = U_flat.reshape(2, Ny, Nx)

        xp = jnp.mod(part[0::4].real, self.KF_RHS.Lx)
        yp = jnp.mod(part[1::4].real, self.KF_RHS.Ly)
        up = part[2::4].real
        vp = part[3::4].real

        # flow RHS & material derivative fields (real)
        KF_rhs, u_t_field, v_t_field = self.KF_RHS(U_flat)

        # upsample to a finer grid for particle sampling
        u_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(U[0]), self.r)
        v_fine = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(jnp.fft.rfft2(U[1]), self.r)

        # bilinear samples (periodic) at particle positions
        u = bilinear_sample_periodic(u_fine, xp, yp, self.KF_RHS.Lx, self.KF_RHS.Ly)
        v = bilinear_sample_periodic(v_fine, xp, yp, self.KF_RHS.Lx, self.KF_RHS.Ly)

        # sample material derivs on the base grid (already real fields)
        u_mat = bilinear_sample_periodic(u_t_field, xp, yp, self.KF_RHS.Lx, self.KF_RHS.Ly)
        v_mat = bilinear_sample_periodic(v_t_field, xp, yp, self.KF_RHS.Lx, self.KF_RHS.Ly)

        hx = self.a * (self.St_inv * u + 1.5 * self.beta * u_mat)
        hy = self.a * (self.St_inv * v + 1.5 * self.beta * v_mat)

        up_dot_rhs = self.alpha * up + hx
        vp_dot_rhs = self.alpha * vp + hy

        # particle RHS: [dx/dt, dy/dt, du_p/dt, dv_p/dt]
        px = up
        py = vp
        particle_rhs = jnp.stack([px, py, up_dot_rhs, vp_dot_rhs], axis=1).reshape(-1)

        return jnp.concatenate([particle_rhs, KF_rhs])

def make_divergence_monitor_rfft(rhs, NDOF, n_particles=None, n_print=50):
    """
    Print RMS(∇·u) every n_print steps; RHS is rfft-based.
    """
    Ny = Nx = NDOF
    Nx_r = Nx//2 + 1

    def monitor(n, X):
        if n % n_print != 0:
            return
        if n_particles is not None:
            U_hat_r_flat = X[n_particles*4:]
        else:
            U_hat_r_flat = X

        U_hat_r = U_hat_r_flat.reshape(2, Ny, Nx_r)
        div_hat = rhs.dxop * U_hat_r[0] + rhs.dyop * U_hat_r[1]    # (Ny, Nx_r)
        div = jnp.fft.irfft2(div_hat, s=(Ny, Nx)).real
        div_rms = float(jnp.sqrt(jnp.mean(div**2)))
        print(f"[step {n:5d}] div(RMS) = {div_rms:.3e}")

    return monitor


