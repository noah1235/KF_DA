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
        f0,
        x: jnp.ndarray,
        p: jnp.ndarray,
        grad: jnp.ndarray,
        loss_grad_cond_fn,
        compute_grad: bool,
    ) -> float:
        alpha  = self.alpha_init
        g0 = jnp.dot(grad, p)

        for i in range(self.max_iters):
            max_loss = f0 + self.c*alpha*g0
            x_next = x + alpha * p
            if i == self.max_iters-1:
                loss_next, grad_next, _ = loss_grad_cond_fn(jnp.inf, x_next)
                return alpha, x_next, loss_next, grad_next
            if compute_grad:
                loss_next, _, _ = loss_grad_cond_fn(-jnp.inf, x_next)
                if loss_next < max_loss:
                    return alpha, x_next, jnp.nan, jnp.nan
            else:
                loss_next, grad_next, active = loss_grad_cond_fn(max_loss, x_next)
                if active:
                    return alpha, x_next, loss_next, grad_next

            alpha *= self.rho

