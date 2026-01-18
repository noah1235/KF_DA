import numpy as np
import jax.numpy as jnp
import vpfloat
from SRC.vp_floats.vp_py_utils import choose_exponent_format, calc_output_shape, float_pos_range

def test_vpfloat(mbits, exp_bits, exp_bias, minv, maxv, N, logspace=True):
    # generate random values
    if logspace:
        x = 10 ** np.random.uniform(np.log10(minv), np.log10(maxv), size=N)
    else:
        x = np.random.uniform(minv, maxv, size=N)

    s = np.random.choice([-1, 1], size=N)
    x = x * s
    x = x.astype(np.float64)
    st_mem = x.nbytes
    sign, exp, mant, shape = vpfloat.split_f64(x, exp_bits, exp_bias, mbits)
    print(sign.dtype, exp.dtype, mant.dtype)
    vp_mem = sign.nbytes + exp.nbytes + mant.nbytes
    y = vpfloat.join_f64(sign, exp, mant, shape, exp_bits, exp_bias, mbits)
    
    # relative error (safe)
    rel_pct_error = np.abs((x - y) / x) * 100

    print(vp_mem / st_mem)

    print("relative error statistics:")
    print("  mean :", np.mean(rel_pct_error))
    print("  std  :", np.std(rel_pct_error))
    print("  max  :", np.max(rel_pct_error))


def main():
    mbits = 12
    N = 10024324
    minv = 1e-3
    maxv = 10
    exp_bits, exp_bias = choose_exponent_format(minv, maxv, max_E=4)
    total_bits = 1+exp_bits+mbits
    print(f"total bits: {total_bits}")
    mint, maxt = float_pos_range(exp_bits, exp_bias, mbits)
    print(f"{mint:.2e} | {maxt:.2e}")




    print(calc_output_shape(N, mbits, exp_bits))
    test_vpfloat(mbits, exp_bits, exp_bias, minv, maxv, N, logspace=True)


main()
