import jax.numpy as jnp

class ArmijoLineSearch:
    name = "ArmBT"
    def __init__(
        self,
        alpha_init: float = 1.0,
        rho: float        = 0.5,
        c: float          = 1e-4,
        max_iters: int    = 10
    ):
        self.alpha_init = alpha_init
        self.rho        = rho
        self.c          = c
        self.max_iters  = max_iters
        self.min_alpha = rho**max_iters

    def init_opt(self):
        pass


    def __call__(
        self,
        f,
        f0,
        x: jnp.ndarray,
        p: jnp.ndarray,
        grad: jnp.ndarray,
        loss_grad_fn,
        last_iter
    ) -> float:
        alpha  = self.alpha_init
        g0 = jnp.dot(grad, p)

        for _ in range(self.max_iters):
            new_loss = f(x + alpha*p)
            max_loss = f0 + self.c*alpha*g0
            if new_loss <= max_loss:
                break
            alpha *= self.rho

        x_next = x + alpha * p
        if last_iter:
            return alpha, x_next, jnp.nan, jnp.nan
        else:
            loss_next, grad_next = loss_grad_fn(x_next)
            return alpha, x_next, loss_next, grad_next


class Armijo_TR:
    name = "ATR"
    def __init__(self):
        self.alpha_init = 1
        self.alpha_max = 1
        self.c = 1e-4
        self.p = 1

    def init_opt(self):
        self.alpha = self.alpha_init

    def PD_update(self, max_loss, loss_next):
        error = (max_loss - loss_next)/max_loss
        log_alpha = jnp.log(self.alpha)
        log_alpha += error * self.p
        self.alpha = jnp.exp(log_alpha)
        self.alpha = min(self.alpha_max, self.alpha)

    def __call__(self, pk, grad, loss_fn, x0, loss, loss_grad_fn, last_iter):
        g0 = jnp.dot(grad, pk)
        max_loss = loss + self.c*self.alpha*g0
        x_next = x0 + self.alpha * pk
        loss_next = loss_fn(x_next)
        self.PD_update(max_loss, loss_next)

        x_next = x0 + self.alpha * pk
        alpha_old = self.alpha
        loss_next, grad_next = loss_grad_fn(x_next)
        self.PD_update(max_loss, loss_next)



        if last_iter:
            return alpha_old, x_next, jnp.nan, jnp.nan
        else:
            return alpha_old, x_next, loss_next, grad_next

class Cubic_TR:
    name = "TR"
    def __init__(self, rho_trg, eta_kp, eta_ki, eta_kd, eta_min=1e-12, eta_0=1, eta_max=1e6):
        self.BT_ls = ArmijoLineSearch()
        self.eta_0 = eta_0
        self.eta_max = eta_max
        self.eta_min = eta_min

        self.rho_target = rho_trg
        self.eta_kp = eta_kp
        self.eta_ki = eta_ki
        self.eta_kd = eta_kd
        self.eta_int    = 0.0          # integral state for PI
        self.eta_int_max = 5.0        # anti-windup clamp
        self.eta_prev_err = 0.0

    def init_opt(self):
        self.eta = self.eta_0

    def solve_alpha(self, pk, g, loss_fn, loss, x0, pTHp):
        c = jnp.dot(pk, g)
        b = pTHp
        p_norm = jnp.linalg.norm(pk)
        a = 0.5 * self.eta * (p_norm ** 3)


        # Solve aα² + bα + c = 0 → α = (-b + sqrt(b² - 4ac)) / (2a)
        disc = b * b - 4 * a * c

        alpha = (-b + jnp.sqrt(disc)) / (2 * a)
        model = loss + (alpha * c) + (0.5 * alpha**2 * b) + ((a / 3) * alpha**3)
        return alpha, model

    def get_alpha(self, pk, g, pTHp, loss_fn, x0, loss, loss_grad_fn, last_iter):
        """
        Cubic regularized trust-region step solver.
        Solves for step length α in m(α) = loss + αc + ½α²b + (a/3)α³
        where a = (η/2)||p||³.
        """

        alpha, model = self.solve_alpha(pk, g, loss_fn, loss, x0, pTHp)

        x_next = x0 + alpha * pk
        if not last_iter:
            loss_next, grad_next = loss_grad_fn(x_next)

            # Compute trust-region ratio
            pred_red = loss - model
            act_red = loss - loss_next
            print(f"pred: {pred_red:.2e} | act: {act_red:.2e}")

            if act_red < 0:
                self.eta = self.eta_0
                alpha, model = self.solve_alpha(pk, g, loss_fn, loss, x0, pTHp)
                x_next = x0 + alpha * pk
                loss_next, grad_next = loss_grad_fn(x_next)
                pred_red = loss - model
                act_red = loss - loss_next
                self.eta_int = 0


            rho = act_red / (pred_red + 1e-12)
            print(f"rho: {rho}")

            # --- PI-style update for eta (no more if/else jumps) ---
            # Error: want rho ≈ rho_target
            e = float(rho - self.rho_target)

            # Update integral (with anti-windup)
            self.eta_int += e
            self.eta_int = float(jnp.clip(self.eta_int, -self.eta_int_max, self.eta_int_max))

            de = e - self.eta_prev_err
            self.eta_prev_err = e  # store for next iteration
            #print(f"rho: {rho:.2e} | e: {e:.2e} | e_int: {self.eta_int:.2e} | de: {de:.2e}")

            # Work in log(eta) space → multiplicative updates but smooth
            log_eta = jnp.log(self.eta)

            delta_log_eta = -(
                self.eta_kp * e +
                self.eta_ki * self.eta_int +
                self.eta_kd * de
            )

        
            log_eta_new = log_eta + delta_log_eta
            eta_new = jnp.exp(log_eta_new)

            # Clamp eta to [eta_min, eta_max]
            eta_new = jnp.clip(eta_new, self.eta_min, self.eta_max)
            self.eta = float(eta_new)

            return float(alpha), x_next, loss_next, grad_next
        else:
            return float(alpha), x_next, jnp.nan, jnp.nan