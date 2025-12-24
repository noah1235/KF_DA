import math

import math

def choose_exponent_format(min_mag, max_mag, max_E=8):
    assert min_mag > 0 and max_mag > min_mag

    k_min = math.floor(math.log2(min_mag))
    k_max = math.ceil(math.log2(max_mag))

    # required exponent span
    span = k_max - k_min + 3

    # minimal exponent bits
    E_req = math.ceil(math.log2(span))

    # clamp
    E = min(max_E, E_req)

    # bias choice: anchor smallest value
    bias = 1 - k_min

    # feasibility check after clamping
    max_bias = 2**E - 2
    if not (0 <= bias <= max_bias):
        raise ValueError(
            f"Requested range [{min_mag:.2e}, {max_mag:.2e}] "
            f"requires E={E_req} exponent bits, but max_E={max_E}."
        )

    return E, bias


import math
from typing import Tuple

def float_pos_range(exp_bits: int, bias: int, mant_bits: int, *, subnormals: bool = False) -> Tuple[float, float]:
    """
    Returns (min_positive, max_finite) for an IEEE-like binary float format.

    Assumptions:
      - base-2
      - exponent field:
          0           -> subnormals/zero
          1..(2^E-2)   -> normals
          (2^E-1)      -> inf/nan (reserved)
      - mantissa has mant_bits fraction bits.
      - normals have implicit leading 1, i.e. significand in [1, 2-2^-mant_bits]
      - subnormals (if enabled) have leading 0, significand in (0, 1-2^-mant_bits] scaled by same emin.

    Parameters:
      exp_bits: number of exponent bits (E)
      bias: exponent bias
      mant_bits: number of fraction (mantissa) bits
      subnormals: if False, min_positive is the smallest *normal*.

    """
    if exp_bits <= 0:
        raise ValueError("exp_bits must be > 0")
    if mant_bits < 0:
        raise ValueError("mant_bits must be >= 0")

    e_max_field = (1 << exp_bits) - 2      # max finite exponent field value
    e_min_field = 1                         # smallest normal exponent field value

    emax = e_max_field - bias               # max unbiased exponent for finite normals
    emin = e_min_field - bias               # min unbiased exponent for normals (and subnormals share this scale)

    # Largest finite: (2 - 2^-mant_bits) * 2^emax
    max_finite = (2.0 - 2.0**(-mant_bits)) * (2.0**emax)

    if subnormals:
        # Smallest positive subnormal: 2^(emin - mant_bits)
        min_pos = 2.0**(emin - mant_bits)
    else:
        # Smallest positive normal: 1.0 * 2^emin
        min_pos = 2.0**emin

    return min_pos, max_finite


def calc_output_shape(N, mbits, exp_bits):
    total_mbits = N * mbits
    total_exp_bits = N*exp_bits
    if mbits == 16:
        m_shape = (math.ceil(total_mbits/16),)
    else:
        m_shape = (math.ceil(total_mbits/8),)
    exp_shape = (math.ceil(total_exp_bits/8),)
    sign_shape = (math.ceil(N/8),)

    return sign_shape, exp_shape, m_shape

