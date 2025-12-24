import numpy as np
import jax.numpy as jnp
import vpfloat
from SRC.vp_floats.vp_py_utils import choose_exponent_format

def test_vpfloat(mbits, exp_bits, exp_bias, minv, maxv, N, logspace=True):
    # generate random values
    if logspace:
        x = 10 ** np.random.uniform(np.log10(minv), np.log10(maxv), size=N)
    else:
        x = np.random.uniform(minv, maxv, size=N)

    s = np.random.choice([-1, 1], size=N)
    x = x * s
    #x = x.astype(np.float64)
    st_mem = x.nbytes
    print(exp_bits)
    sign, exp, mant, shape = vpfloat.split_f32(x, exp_bits, exp_bias, mbits)
    print(sign.shape, exp.shape, mant.shape)
    vp_mem = sign.nbytes + exp.nbytes + mant.nbytes
    y = vpfloat.join_f32(sign, exp, mant, shape, exp_bits, exp_bias, mbits)
    
    # relative error (safe)
    rel_pct_error = np.abs((x - y) / x) * 100

    print(vp_mem / st_mem)

    print("relative error statistics:")
    print("  mean :", np.mean(rel_pct_error))
    print("  std  :", np.std(rel_pct_error))
    print("  max  :", np.max(rel_pct_error))


def main():
    mbits = 6
    N = 64
    minv = 1e-3
    maxv = 1e2

    exp_bits, exp_bias = choose_exponent_format(minv, maxv)
    print(exp_bits, exp_bias)

    test_vpfloat(mbits, exp_bits, exp_bias, minv, maxv, N, logspace=True)


main()
