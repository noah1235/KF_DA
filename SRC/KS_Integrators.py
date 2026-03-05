import jax
import jax.numpy as jnp
import numpy as np


def run_solver(solver, u_0, time_steps):
    u_0 = u_0.astype(jnp.float64)
    def step_fn(u_current, _):
        u_next = solver(u_current)
        return u_next, u_next
    u_trajectory = jax.lax.scan(step_fn, u_0, None, length=time_steps)[1]

    u_trajectory = jnp.concatenate([jnp.expand_dims(u_0, 0), u_trajectory], axis=0)

    return u_trajectory


class KS_Integrator:
    def __init__(self, L, N, dt,use_double_precision, Euler_term_taylor_cutoff=0.0001):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N
        
        if use_double_precision:
            self.real_dtype = jnp.float64
            self.complex_dtype = jnp.complex128
        else:
            self.real_dtype = jnp.float32
            self.complex_dtype = jnp.complex64


        wavenumbers = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi)).astype(self.real_dtype)
        self.wavenumbers = wavenumbers
        self.derivative_operator = (1j * wavenumbers).astype(self.complex_dtype)

        self.linear_operator = (-self.derivative_operator**2 - self.derivative_operator**4).astype(self.complex_dtype)
        self.h_terms, self.c_terms = self.Pre_Compute_Taylor_Terms(dt, self.linear_operator)
   
        self.exp_term = jnp.exp(self.linear_operator * dt).astype(self.complex_dtype)
        Euler_coef_exact = (jnp.expm1(self.linear_operator * dt) / self.linear_operator).astype(self.complex_dtype)
        Euler_coef_Taylor = self.Euler_coef_Taylor_approx(self.linear_operator, dt)

        self.Euler_coef = jnp.where(jnp.abs(self.linear_operator * dt) <= Euler_term_taylor_cutoff, 
                                    Euler_coef_Taylor, Euler_coef_exact).astype(self.complex_dtype)

        self.alias_mask = (wavenumbers < 2/3 * jnp.max(wavenumbers)).astype(jnp.bool_)
    

    @staticmethod
    def Pre_Compute_Taylor_Terms(h, c):
        # Precompute powers of h to avoid redundant calculations
        h2 = h**2
        h3 = h**3
        h4 = h**4
        h5 = h**5
        h6 = h**6
        h7 = h**7
        h8 = h**8

        # Precompute powers of c for reuse
        c2 = c**2
        c3 = c**3
        c4 = c**4
        c5 = c**5
        c6 = c**6
        c7 = c**7

        h_terms = (h2, h3, h4, h5, h6, h7, h8)
        c_terms = (c2, c3, c4, c5, c6, c7)

        return h_terms, c_terms

    @staticmethod
    def kahan_sum(terms):
        sum_ = 0.0
        c = 0.0
        for term in terms:
            y = term - c
            t = sum_ + y
            c = (t - sum_) - y
            sum_ = t
        return sum_

    def Euler_coef_Taylor_approx(self, c, h):
        h2, h3, h4, h5, h6, h7, h8 = self.h_terms
        c2, c3, c4, c5, c6, c7 = self.c_terms
        
        Euler_coef_taylor = [
            h,
            c * h2 / 2,
            c2 * h3 / 6,
            c3 * h4 / 24,
            c4 * h5 / 120,
            c5 * h6 / 720,
            c6 * h7 / 5040,
            c7 * h8 / 40320
        ]

        return self.kahan_sum(Euler_coef_taylor).astype(self.complex_dtype)

    def irfft(self, x):
        return jnp.fft.irfft(x, n=self.N).astype(self.real_dtype)
    
    def rfft(self, x):
        return jnp.fft.rfft(x).astype(self.complex_dtype)

    def F_nonlinear_fourier_space(self, U):
        F = (-0.5 * U**2).astype(self.real_dtype)
        F_hat = self.rfft(F)
        F_hat *= self.alias_mask
        F_hat *= self.derivative_operator

        return F_hat

class KS_Foward_Euler(KS_Integrator):
    def __repr__(self) -> str:
        return "1st_Euler"

    def __init__(self, L, N, dt, use_double_precision=True):
        super().__init__(L, N, dt, use_double_precision)
    
    def __call__(self, u):
        u_nonlin = (-0.5 * u**2).astype(self.real_dtype)
        u_hat = self.rfft(u)
        u_nonlin_hat = self.rfft(u_nonlin)
        u_nonlin_hat = u_nonlin_hat * self.alias_mask
        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        u_next_hat = self.exp_term * u_hat + self.Euler_coef * u_nonlin_der_hat
        u_next = self.irfft(u_next_hat)
        return u_next

class ETD2(KS_Integrator):
    name = "ETD2"
    def __init__(self, L, N, dt, use_double_precision=True):
        super().__init__(L, N, dt, use_double_precision)
        self.coef_1 = jnp.where(
            self.linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / self.linear_operator,
        )
        self.coef_2 = jnp.where(
            self.linear_operator == 0.0,
            dt / 2,
            (self.exp_term - 1.0 - self.linear_operator * dt) / (self.linear_operator**2 * dt)
        )

    def __call__(self,u):
        u_nonlin = - 0.5 * u**2
        u_hat = jnp.fft.rfft(u)
        u_nonlin_hat = jnp.fft.rfft(u_nonlin)
        u_nonlin_hat = self.alias_mask * u_nonlin_hat
        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        u_stage_1_hat = self.exp_term * u_hat + self.coef_1 * u_nonlin_der_hat
        u_stage_1 = jnp.fft.irfft(u_stage_1_hat, n=self.N)

        u_stage_1_nonlin = - 0.5 * u_stage_1**2
        u_stage_1_nonlin_hat = jnp.fft.rfft(u_stage_1_nonlin)
        u_stage_1_nonlin_hat = self.alias_mask * u_stage_1_nonlin_hat
        u_stage_1_nonlin_der_hat = self.derivative_operator * u_stage_1_nonlin_hat

        u_next_hat = u_stage_1_hat + self.coef_2 * (u_stage_1_nonlin_der_hat - u_nonlin_der_hat)
        u_next = jnp.fft.irfft(u_next_hat, n=self.N)

        return u_next

class KS_RK3(KS_Integrator):
    name = "RK3"


    def __init__(self, L, N, dt, HOT_term_taylor_cutoff=0.01, use_double_precision=True):
        super().__init__(L, N, dt, use_double_precision)

        hc = (self.dt * self.linear_operator).astype(self.complex_dtype)

        self.exp_term_half_step = jnp.exp(self.linear_operator * (dt / 2)).astype(self.complex_dtype)
        self.Euler_half_step_coef = jnp.where(self.linear_operator == 0.0,dt / 2,
                                    jnp.expm1(self.linear_operator * (dt / 2)) / self.linear_operator).astype(self.complex_dtype)

        
        
        coef1_exact, coef2_exact, coef3_exact = self.compute_exact_coefs(hc, self.linear_operator, self.exp_term)

        coef1_taylor, coef2_taylor, coef3_taylor = self.compute_taylor_approx_coefs(self.linear_operator, self.dt)

        self.coef1 = jnp.where(jnp.abs(self.linear_operator * dt) <= HOT_term_taylor_cutoff, 
                                coef1_taylor, coef1_exact).astype(self.complex_dtype)
        self.coef2 = jnp.where(jnp.abs(self.linear_operator * dt) <= HOT_term_taylor_cutoff, 
                                coef2_taylor, coef2_exact).astype(self.complex_dtype)
        self.coef3 = jnp.where(jnp.abs(self.linear_operator * dt) <= HOT_term_taylor_cutoff, 
                                coef3_taylor, coef3_exact).astype(self.complex_dtype)


    def compute_exact_coefs(self, hc, c, exp_term):
        hc2 = (hc ** 2).astype(self.complex_dtype)
        denom = hc2 * c

        coef1_exact = (-4 - hc + exp_term * (4 - 3 * hc + hc2)) / (denom).astype(self.complex_dtype)
        coef2_exact = (2 + hc + exp_term * (-2 + hc)) / (denom).astype(self.complex_dtype)
        coef3_exact = (-4 - 3 * hc - hc2 + exp_term * (4 - hc)) / (denom).astype(self.complex_dtype)

        return coef1_exact, coef2_exact, coef3_exact

    def compute_taylor_approx_coefs(self, c, h):
        h2, h3, h4, h5, h6, h7, h8 = self.h_terms
        c2, c3, c4, c5, c6, c7 = self.c_terms
        
        #6 Order taylor approx
        coef1_taylor_terms = [
            c7 * h8 / 40320,
            c6 * h7 / 8064,
            c5 * h6 / 1120,
        5 * c4 * h5 / 1008,
            c3 * h4 / 45,
            3 * c2 * h3 / 40,
            c * h2 / 6,
            h / 6
        ]
        
        coef2_taylor_terms = [
            c6 * h7 / 40320,
            c5 * h6 / 6720,
            c4 * h5 / 1008,
            c3 * h4 / 180,
            c2 * h3 / 40,
            c * h2 / 12,
            h / 6
        ]
        
        coef3_taylor_terms = [
            -(c6 * h7 / 40320),
            -(c5 * h6 / 10080)
            -(c4 * h5 / 1680),
            -(c3 * h4 / 360),
            -(c2 * h3 / 120),
            h / 6
        ]

        coef1_taylor = self.kahan_sum(coef1_taylor_terms).astype(self.complex_dtype)
        coef2_taylor = self.kahan_sum(coef2_taylor_terms).astype(self.complex_dtype)
        coef3_taylor = self.kahan_sum(coef3_taylor_terms).astype(self.complex_dtype)

        return coef1_taylor, coef2_taylor, coef3_taylor  

    def __call__(self, U):
        U_n_hat = self.rfft(U)
        F_n_hat = self.F_nonlinear_fourier_space(U)

        linear_half_step_term = U_n_hat * self.exp_term_half_step
        linear_full_step_term = U_n_hat * self.exp_term

        an_hat = linear_half_step_term + self.Euler_half_step_coef * F_n_hat
        an = self.irfft(an_hat)
        F_an_hat = self.F_nonlinear_fourier_space(an)

        bn_hat = linear_full_step_term + self.Euler_coef * (2 * F_an_hat - F_n_hat)
        bn = self.irfft(bn_hat)
        F_bn_hat = self.F_nonlinear_fourier_space(bn)

        nonlin_integration = F_n_hat * self.coef1 + 4*F_an_hat * self.coef2 + F_bn_hat * self.coef3

        U_next_hat = linear_full_step_term + nonlin_integration
        
        U_next = self.irfft(U_next_hat)
        return U_next

class KS_RK4(KS_RK3):
    name = "RK4"

    @classmethod
    def __repr__(self) -> str:
        return "RK4"

    def __init__(self, L, N, dt, use_double_precision=True):
        super().__init__(L, N, dt, use_double_precision)

    def __call__(self, U):
        U_n_hat = self.rfft(U)
        F_n_hat = self.F_nonlinear_fourier_space(U)

        linear_half_step_term = U_n_hat * self.exp_term_half_step
        linear_full_step_term = U_n_hat * self.exp_term

        an_hat = linear_half_step_term + self.Euler_half_step_coef * F_n_hat
        F_an_hat = self.F_nonlinear_fourier_space(self.irfft(an_hat))

        bn_hat = linear_half_step_term + self.Euler_half_step_coef * F_an_hat
        F_bn_hat = self.F_nonlinear_fourier_space(self.irfft(bn_hat))

        cn_hat = linear_half_step_term + self.Euler_half_step_coef * (2*F_bn_hat-F_n_hat)
        F_cn_hat = self.F_nonlinear_fourier_space(self.irfft(cn_hat))

        nonlin_integration = F_n_hat * self.coef1 + 2*(F_an_hat + F_bn_hat) * self.coef2 + F_cn_hat * self.coef3
        
        U_next_hat = linear_full_step_term + nonlin_integration
        
        U_next = self.irfft(U_next_hat)
        return U_next
