class LBFGS():
    name = "L-BFGS"
    def __init__(self, its):
        self.its = its
        self.ls_method = "BT"
        self.alg = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=20))
    
class ADAM():
    name = "ADAM"
    def __init__(self, lr, its):
        self.its = its
        self.alg = optax.adam(learning_rate=lr)
        
    def __repr__(self):
        return f"{self.name}-{self.its}"

def optax_opt(U_0, loss_fn, loss_grad_fn, optimizer_config, div_check, div_free_proj):
    optimizer = optimizer_config.alg
    opt_state = optimizer.init(U_0)
    opt_data = Opt_Data(optimizer_config.its)

    @jax.jit
    def inner_loop(U_0, opt_state):
        loss, grad = loss_grad_fn(U_0)

        updates, opt_state = optimizer.update(
            grad, opt_state, U_0, value=loss, grad=grad, value_fn=loss_fn
        )

        div_updates = div_check(updates)
        U_0_next = optax.apply_updates(U_0, updates)
        U_0_next = div_free_proj(U_0_next)
        return U_0_next, opt_state, loss, grad, updates, div_updates

    for i in range(optimizer_config.its):
        U_0, opt_state, loss, grad, alpha_p, div_updates = inner_loop(U_0, opt_state)
        opt_data(i, loss, grad, alpha_p)
        print(f"i:{i} | loss: {loss} | Div: {div_check(U_0)} | div updates: {div_updates} | {jnp.linalg.norm(alpha_p)}")

    
    
    del optimizer

    return U_0, opt_data

