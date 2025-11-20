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


    @jax.jit
    def build_Bk(self, Bk_vecs, Bk_scalars):
        n = Bk_vecs.shape[0]
        result = jnp.zeros((n, n))
        for i in range(self.cmem):
            x = Bk_vecs[:, i]
            result += jnp.outer(x, x) * Bk_scalars[i]
        return result
    
    def __len__(self):
        return self.cmem
    
    def __matmul___dec(self, v: jnp.ndarray):
        result = jnp.zeros(v.shape[0])
        for i in range(self.cmem):
            x = self.Bk_vecs[:, i]
            result += self.Bk_scalars[i] * x * jnp.dot(x, v)
        return result

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

    def SR1_update(self, U_0_next, U_0, grad_next, grad, N, eps=1e-12):
        s = U_0_next - U_0
        y = grad_next - grad

        #maxed_mem = len(self.Bk) >= self.Bk.max_memory
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


    
    def Bk_eig_decomp(self, which="LM"):
        def matvec(v):
            v = jnp.asarray(v)
            return self.Bk @ v

        A_op = LinearOperator((self.Bk.N, self.Bk.N), matvec=matvec)
        Bk_eigs, Bk_eig_vec = eigsh(A_op, k=len(self.Bk), which=which)
        return jnp.array(Bk_eigs), jnp.array(Bk_eig_vec)





class HVP_Update():
    def __init__(self):
        self.Bk = L_BK()

    def linear_dep_check(self, v_array, vTAv_array, cos_tol=.97):
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


    def HVP_Bk_update(self, vTAv_array, v_array):
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
