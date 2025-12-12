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
    ) -> float:
        alpha  = self.alpha_init
        g0 = jnp.dot(grad, p)

        for _ in range(self.max_iters):
            new_loss = f(x + alpha*p)
            max_loss = f0 + self.c*alpha*g0
            if new_loss <= max_loss:
                break
            alpha *= self.rho
        return alpha

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

    def get_alpha_dec(self, pk, g, pTHp, loss_fn, x0, loss, num_repeats=0):
        """
        Cubic regularized trust-region step solver.
        Solves for step length α in m(α) = loss + αc + ½α²b + (a/3)α³
        where a = (η/2)||p||³.
        """
        retry = False

        c = jnp.dot(pk, g)
        b = pTHp
        p_norm = jnp.linalg.norm(pk)
        a = 0.5 * self.eta * (p_norm ** 3)
        eps = 1e-12


        # Solve aα² + bα + c = 0 → α = (-b + sqrt(b² - 4ac)) / (2a)
        disc = b * b - 4 * a * c
        if disc < 0 or a == 0:
            alpha = self.BT_ls(loss_fn, x0, pk, g)
            return alpha

        alpha = (-b + jnp.sqrt(disc)) / (2 * a)
        if jnp.isnan(alpha) or jnp.isinf(alpha) or alpha <= 0:
            print(f"alpha invalid | a={a}, b={b}, c={c}")
            alpha = self.BT_ls(loss_fn, x0, pk, g)
            return alpha

        try:
            # Evaluate model and new loss
            model = loss + (alpha * c) + (0.5 * alpha**2 * b) + ((a / 3) * alpha**3)
            new_loss = loss_fn(x0 + alpha * pk)
            if jnp.isnan(new_loss) or jnp.isnan(model):
                raise ValueError("NaN encountered in model or new_loss")
        except:
            retry = True

        # Compute trust-region ratio
        pred_red = loss - model
        act_red = loss - new_loss
        rho = act_red / (pred_red + eps)
        if act_red < 0:
            retry = True

        if retry:
            if num_repeats == 2:
                return 0
            elif num_repeats == 1:
                self.eta = self.eta_max
                return self.get_alpha(pk, g, pTHp, loss_fn, x0, loss, num_repeats=num_repeats+1)
            elif num_repeats == 0:
                self.eta = self.eta_0
                return self.get_alpha(pk, g, pTHp, loss_fn, x0, loss, num_repeats=num_repeats+1)


        # Update eta
        if rho > 0.9:
            self.eta /= self.rho
        elif rho < 0.5:
            self.eta *= self.rho

        self.eta = float(max(self.eta, self.eta_min))
        return float(alpha)


    def get_alpha(self, pk, g, pTHp, loss_fn, x0, loss, num_repeats=0):
        """
        Cubic regularized trust-region step solver.
        Solves for step length α in m(α) = loss + αc + ½α²b + (a/3)α³
        where a = (η/2)||p||³.
        """
        retry = False

        c = jnp.dot(pk, g)
        b = pTHp
        p_norm = jnp.linalg.norm(pk)
        a = 0.5 * self.eta * (p_norm ** 3)
        eps = 1e-12

        # Solve aα² + bα + c = 0 → α = (-b + sqrt(b² - 4ac)) / (2a)
        disc = b * b - 4 * a * c
        if disc < 0 or a == 0:
            alpha = self.BT_ls(loss_fn, loss, x0, pk, g)
            return float(alpha)


        alpha = (-b + jnp.sqrt(disc)) / (2 * a)
        if jnp.isnan(alpha) or jnp.isinf(alpha) or alpha <= 0:
            print(f"alpha invalid | a={a}, b={b}, c={c}")
            alpha = self.BT_ls(loss_fn, loss, x0, pk, g)
            return float(alpha)

        try:
            # Evaluate model and new loss
            model = loss + (alpha * c) + (0.5 * alpha**2 * b) + ((a / 3) * alpha**3)
            new_loss = loss_fn(x0 + alpha * pk)
            if jnp.isnan(new_loss) or jnp.isnan(model):
                raise ValueError("NaN encountered in model or new_loss")
        except Exception as e:
            print("Exception in loss/model eval:", e)
            retry = True

        # Compute trust-region ratio
        pred_red = loss - model
        act_red = loss - new_loss
        rho = act_red / (pred_red + eps)
        if act_red < 0:
            retry = True

        # --- Retry / reset logic (unchanged, but also reset integral term) ---
        if retry:
            # integral reset on catastrophic mismatch
            self.eta_int = 0.0
            self.eta_prev_err = 0.0

            if num_repeats == 2:
                return 0.0
            elif num_repeats == 1:
                self.eta = self.eta_max
                return self.get_alpha(pk, g, pTHp, loss_fn, x0, loss,
                                    num_repeats=num_repeats+1)
            elif num_repeats == 0:
                self.eta = self.eta_0
                return self.get_alpha(pk, g, pTHp, loss_fn, x0, loss,
                                    num_repeats=num_repeats+1)

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

        return float(alpha)