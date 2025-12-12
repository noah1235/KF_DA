import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, eigsh
import jax
class L_BK:
    def __init__(self, max_memory, N):
        self.N = N
        self.max_memory = max_memory
        self.cmem = 0
        self.Bk_vecs = jnp.zeros((N, self.max_memory))
        self.Bk_scalars = jnp.zeros(self.max_memory)

    def set_Bk(self, Bk_vecs, Bk_scalars):
        if Bk_vecs.shape[1] != Bk_scalars.shape[0]:
            raise ValueError("Need same number of vecs and scalars")
        self.Bk_vecs = Bk_vecs
        self.Bk_scalars = Bk_scalars
        self.cmem = Bk_scalars.shape[0]
        

    def build_Bk(self):
        result = jnp.zeros((self.N, self.N))
        for i in range(self.cmem):
            x = self.Bk_vecs[:, i]
            result += jnp.outer(x, x) * self.Bk_scalars[i]
        return result

    def eig_decomp(self, which="LM", num_eig=None):
        if num_eig is None or num_eig > len(self):
            num_eig = len(self)
        def matvec(v):
            v = jnp.asarray(v)
            return self @ v

        A_op = LinearOperator((self.N, self.N), matvec=matvec)
        Bk_eigs, Bk_eig_vec = eigsh(A_op, k=num_eig, which=which)
        return jnp.array(Bk_eigs), jnp.array(Bk_eig_vec)
    
    def __len__(self):
        return self.cmem

    def __matmul__(self, v: jnp.ndarray):
        X = self.Bk_vecs[:, :self.cmem]         # shape (n, cmem)
        alpha = self.Bk_scalars[:self.cmem]     # shape (cmem,)

        proj = X.T @ v                           # (cmem,)
        weights = alpha * proj                   # (cmem,)
        return X @ weights                       # (n,)
    
    def append(self, vec: jnp.ndarray, scalar: any):
        vec = vec.reshape((self.N, -1))
        if self.cmem >= self.max_memory:
            raise ValueError("Current memory = max memory. Can't append to Bk")
        
        num_new = vec.shape[1]
        if num_new > 1:
            #appending multiple R1 matrices
            if type(scalar) != jnp.ndarray:
                raise TypeError("scalar list should be jnp array")
            
            if num_new != len(scalar):
                raise ValueError("num vecs and num scalars must be the same")
            
        self.Bk_vecs = self.Bk_vecs.at[:, self.cmem:self.cmem+num_new].set(vec)
        self.Bk_scalars = self.Bk_scalars.at[self.cmem:self.cmem+num_new].set(scalar)
        self.cmem += num_new

    def pop(self, idx):
        if idx >= self.cmem:
            raise ValueError("No R1 Matrix defined at this index")
        
        removed_vec = self.Bk_vecs[:, idx]
        removed_scalar = self.Bk_scalars[idx]
        
        zeros_cols = jnp.zeros((self.N, 1), dtype=self.Bk_vecs.dtype)
        self.Bk_vecs = jnp.concat([jnp.delete(self.Bk_vecs, idx, axis=1), zeros_cols], axis=1)
        self.Bk_scalars = jnp.concat([jnp.delete(self.Bk_scalars, idx), jnp.zeros(1, dtype=self.Bk_scalars.dtype)])
        self.cmem -= 1

        return removed_vec, removed_scalar
    
    def evict_oldest(self, n_del):
        zeros_cols = jnp.zeros((self.N, n_del), dtype=self.Bk_vecs.dtype)
        self.Bk_vecs    = jnp.concatenate([self.Bk_vecs[:, n_del:], zeros_cols], axis=1)
        self.Bk_scalars = jnp.concatenate([self.Bk_scalars[n_del:], jnp.zeros((n_del,), dtype=self.Bk_scalars.dtype)])
        self.cmem = self.cmem - n_del       
    
    def insert(self, vec: jnp.ndarray, scalar: float, idx: int):
        vec = vec.reshape((self.N, 1))
        scalar = jnp.array([scalar])
        
        if idx >= self.cmem:
            raise ValueError("No R1 Matrix defined at this index")
        
        Bk_vecs_1 = self.Bk_vecs[:, :idx]
        Bk_vecs_2 = self.Bk_vecs[:, idx:]
        self.Bk_vecs = jnp.concat([Bk_vecs_1, vec, Bk_vecs_2], axis=1)

        Bk_scalar_1 = self.Bk_scalars[:idx]
        Bk_scalar_2 = self.Bk_scalars[idx:]
        self.Bk_scalars = jnp.concat([Bk_scalar_1, scalar, Bk_scalar_2])

    def __getitem__(self, i):
        return self.Bk_vecs[:, i], self.Bk_scalars[i]
    
    def get_num_open_slots(self):
        return self.max_memory - self.cmem

class L_SR1():
    def __init__(self):
        self.Bk = L_BK()
    
    def set_SR1_update_type(self, type):
        self.SR1_type = type

    def SR1_update(self, U_0_next, U_0, grad_next, grad, loss_next, loss):
        
        if self.SR1_type == "conv":
            self.SR1_update_conv(U_0_next, U_0, grad_next, grad)
        elif self.SR1_type == "mod":
            self.SR1_update_mod(U_0_next, U_0, grad_next, grad, loss_next, loss)

    def SR1_update_conv(self, U_0_next, U_0, grad_next, grad, eps=1e-12):
        s = U_0_next - U_0
        y = grad_next - grad

        maxed_mem = self.Bk.get_num_open_slots() == 0

        if maxed_mem:
            removed_vec, removed_scalar = self.Bk.pop(0)

        r = y - self.Bk @ s
        denom = jnp.vdot(r, s)

        if jnp.abs(denom) <= eps:
            # no change
            print("Skip SR1")
            if maxed_mem:
                self.Bk.insert(removed_vec, removed_scalar, 0)
            return
        
        self.Bk.append(r, 1/denom)

    def SR1_update_mod_dec(self, U_0_next, U_0, grad_next, grad, loss_next, loss):
        s = U_0_next - U_0
        y = grad_next - grad
        theta = 6 * (loss - loss_next) + 3 * jnp.dot(((grad_next - grad)), s)
        delta = jnp.dot(y, s) + theta - jnp.dot(s, self.Bk @ s)
        u = y - self.Bk @ s
        gamma = delta / (jnp.dot(u, s)**2)

        maxed_mem = self.Bk.get_num_open_slots() == 0
        if maxed_mem:
            removed_vec, removed_scalar = self.Bk.pop(0)

        self.Bk.append(u, gamma)

    def SR1_update_mod(self, U_0_next, U_0, grad_next, grad, loss_next, loss):
        print("BFGS update")
        s = U_0_next - U_0
        y = grad_next - grad
        gamma_1 = -1/jnp.dot(s, self.Bk@s)
        u1 = self.Bk @ s
        maxed_mem = self.Bk.get_num_open_slots() == 0
        if maxed_mem:
            removed_vec, removed_scalar = self.Bk.pop(0)
        self.Bk.append(u1, gamma_1)

        gamma_2 = 1/jnp.dot(y, s)
        u2 = y
        maxed_mem = self.Bk.get_num_open_slots() == 0
        if maxed_mem:
            removed_vec, removed_scalar = self.Bk.pop(0)
        self.Bk.append(u2, gamma_2)


        

    
class BFGS_Update():

    def Bk_inv_update(self, ys, sk, yk):
        I = jnp.eye(yk.shape[0], dtype=self.Bk_inv.dtype)
        rho = 1.0 / ys
        Sy = jnp.outer(sk, yk)
        Ys = jnp.outer(yk, sk)
        Ss = jnp.outer(sk, sk)
        self.Bk_inv = (I - rho * Sy) @ self.Bk_inv @ (I - rho * Ys) + rho * Ss

class HVP_Update():
    def __init__(self):
        self.Bk = L_BK()

    def linear_dep_check(self, v_array, vTAv_array, cos_tol=.99):
        n_new = vTAv_array.shape[0]

        for i in range(n_new):
            xi = v_array[:, i]
            xi_norm = jnp.linalg.norm(xi)
            xi = xi/xi_norm
            v_array = v_array.at[:, i].set(xi)
            vTAv_array = vTAv_array.at[i].set(vTAv_array[i]/xi_norm**2)

            for j in range(len(self.Bk)):
                xj, _ = self.Bk[j]

                cos_sim = jnp.dot(xj, xi) / (jnp.linalg.norm(xi) * jnp.linalg.norm(xj))
                if jnp.abs(cos_sim) > cos_tol:
                    self.Bk.pop(j)
        
        return v_array, vTAv_array


    def HVP_Bk_update_dec(self, vTAv_array, v_array):
        """
        Insert n_new_vecs columns (xi) with scalars (-mu) into limited-memory buffers.

        Args
        ----
        vTAv_array : (n_new_vecs,)   # v_i^T A v_i
        v_array    : (N, n_new_vecs) # columns are x_i to insert
        Bk_vecs    : (N, M)          # memory buffer of vectors (M = self.max_memory)
        Bk_scalars : (M,)            # memory buffer of scalars
        """
        N = self.Bk.N
        if v_array.shape[0] != N:
            raise ValueError("New vecs must have dimension N")

        n_new = vTAv_array.shape[0]
        if n_new > 1:
            VTV = v_array.T @ v_array
            ortho_error = jnp.linalg.norm(VTV - jnp.eye(VTV.shape[0]))
            if ortho_error > 1e-8:
                print("vecs not ortho")
                return


        #linear indep check
        self.linear_dep_check(v_array, vTAv_array)
        n_free = self.Bk.get_num_open_slots()

        # If not enough free slots, evict the oldest n_del pairs by shifting left.
        if n_free < n_new:
            n_del = n_new - n_free
            self.Bk.evict_oldest(n_del)


        for i in range(n_new):
            xi = v_array[:, i]
            mu = jnp.dot(xi, self.Bk @ xi) - vTAv_array[i]
            self.Bk.append(xi, -mu)


    def HVP_Bk_update(self, vTAv_array, v_array):
        """
        Insert n_new_vecs columns (x_i) with scalars (-mu_i) into limited-memory buffers,
        enforcing x_i^T H_new x_i = vTAv_array[i] in the Frobenius-minimal sense.

        Args
        ----
        vTAv_array : () or (m,)      # y_i = desired quadratic forms x_i^T A x_i
        v_array    : (N,) or (N, m)  # columns are x_i
        """
        # --- Normalize shapes so we always have (N, m) and (m,) ---

        # Ensure vTAv_array is 1D (m,)
        vTAv_array = jnp.atleast_1d(vTAv_array)

        # Ensure v_array is 2D with shape (N, m)
        if v_array.ndim == 1:
            # (N,) -> (N, 1)
            v_array = v_array[:, None]

        N = self.Bk.N
        if v_array.shape[0] != N:
            raise ValueError(f"New vecs must have dimension N={N}, got {v_array.shape[0]}")

        n_new = vTAv_array.shape[0]
        if v_array.shape[1] != n_new:
            raise ValueError(
                f"Mismatch: v_array has {v_array.shape[1]} columns but "
                f"vTAv_array has length {n_new}"
            )

        # Optional: linear independence / sanity check
        self.linear_dep_check(v_array, vTAv_array)

        # Handle limited-memory buffer
        n_free = self.Bk.get_num_open_slots()
        if n_free < n_new:
            n_del = n_new - n_free
            self.Bk.evict_oldest(n_del)

        # --- 1. Build Gram matrix G_{ij} = (x_i^T x_j)^2 ---
        # VTV[i,j] = x_i^T x_j
        VTV = v_array.T @ v_array           # shape (m, m)
        G = VTV * VTV                       # elementwise square → (x_i^T x_j)^2

        # --- 2. Build b_j = x_j^T H0 x_j - y_j ---

        def quad_form(x):
            Hx = self.Bk @ x                # H0 x
            return jnp.dot(x, Hx)           # x^T H0 x

        # Works for both m=1 and m>1
        b = jnp.stack(
            [quad_form(v_array[:, j]) - vTAv_array[j] for j in range(n_new)],
            axis=0
        )  # shape (m,)

        # --- 3. Solve G mu = b (small m×m system) ---
        if n_new == 1:
            # Scalar case: G is 1×1, b is scalar -> avoid full solve
            # G[0,0] = (x^T x)^2
            denom = G[0, 0]
            # Add tiny regularization if you’re paranoid:
            # denom = denom + 1e-12
            mu = jnp.array([b[0] / denom])
        else:
            mu = jnp.linalg.solve(G, b)   # shape (m,)

        # --- 4. Apply update: H* = H0 - sum_i mu_i x_i x_i^T ---
        for j in range(n_new):
            xj = v_array[:, j]
            self.Bk.append(xj, -mu[j])