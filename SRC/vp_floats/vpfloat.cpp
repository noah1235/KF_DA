#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static inline ssize_t ceil_int_div(ssize_t a, ssize_t b) { return (a + b - 1) / b; }

// --------- Fast LSB-first bit writer/reader (byte-chunked) ----------
struct BitWriter {
    uint8_t* out = nullptr;
    size_t bitpos = 0;

    void write(uint32_t v, int nbits) {
        while (nbits > 0) {
            const size_t byte = bitpos >> 3;
            const int off = int(bitpos & 7);
            const int take = std::min(nbits, 8 - off);

            const uint8_t mask = uint8_t((1u << take) - 1u);
            const uint8_t bits = uint8_t((v & mask) << off);
            out[byte] = uint8_t(out[byte] | bits);

            v >>= take;
            nbits -= take;
            bitpos += take;
        }
    }
};

struct BitReader {
    const uint8_t* in = nullptr;
    size_t bitpos = 0;

    uint32_t read(int nbits) {
        uint32_t v = 0;
        int filled = 0;
        while (nbits > 0) {
            const size_t byte = bitpos >> 3;
            const int off = int(bitpos & 7);
            const int take = std::min(nbits, 8 - off);

            const uint32_t chunk = (uint32_t(in[byte]) >> off) & ((1u << take) - 1u);
            v |= (chunk << filled);

            filled += take;
            nbits -= take;
            bitpos += take;
        }
        return v;
    }
};

// --------- IEEE traits ----------
template <typename T> struct IEEETraits;

template <> struct IEEETraits<float> {
    using UInt = uint32_t;
    static constexpr int EXP_BITS   = 8;
    static constexpr int MANT_BITS  = 23;
    static constexpr int EXP_BIAS   = 127;
    static constexpr int SIGN_SHIFT = 31;
    static constexpr int EXP_SHIFT  = 23;
    static constexpr UInt EXP_MASK  = 0xFFu;
    static constexpr UInt MANT_MASK = 0x7FFFFFu;
};

template <> struct IEEETraits<double> {
    using UInt = uint64_t;
    static constexpr int EXP_BITS   = 11;
    static constexpr int MANT_BITS  = 52;
    static constexpr int EXP_BIAS   = 1023;
    static constexpr int SIGN_SHIFT = 63;
    static constexpr int EXP_SHIFT  = 52;
    static constexpr UInt EXP_MASK  = 0x7FFull;
    static constexpr UInt MANT_MASK = 0xFFFFFFFFFFFFFull;
};

static inline uint32_t mask_u32(int bits) {
    // bits in [0, 32]
    if (bits <= 0) return 0u;
    if (bits >= 32) return 0xFFFFFFFFu;
    return (1u << bits) - 1u;
}

// Round mantissa to mant_bits (top bits), ties-to-even, with carry out.
// Input: mant_ieee is fraction field (no implicit leading 1), width Tr::MANT_BITS.
// Output: m_small in [0, 2^mant_bits), and carry (0 or 1) indicating mantissa overflow.
template <typename UInt>
static inline uint32_t round_mantissa_top(UInt mant_ieee, int ieee_mant_bits, int mant_bits, uint32_t& carry_out) {
    carry_out = 0;
    const int shift = ieee_mant_bits - mant_bits;
    if (shift <= 0) {
        // mant_bits >= ieee_mant_bits (not expected here, but safe)
        return uint32_t(mant_ieee);
    }

    const UInt top = mant_ieee >> shift;
    const UInt rem = mant_ieee & ((UInt(1) << shift) - 1);

    const UInt halfway = UInt(1) << (shift - 1);
    const bool round_up = (rem > halfway) || (rem == halfway && (top & 1u));

    UInt rounded = top + (round_up ? 1u : 0u);

    const UInt limit = (UInt(1) << mant_bits);
    if (rounded >= limit) {
        // overflow of mantissa => carry into exponent and mantissa becomes 0
        carry_out = 1;
        rounded = 0;
    }
    return uint32_t(rounded);
}

// ---------------- split_impl ----------------
template <typename T>
py::tuple split_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    int exp_bits,
    int exp_bias,
    int mant_bits
) {
    using Tr = IEEETraits<T>;
    using UInt = typename Tr::UInt;

    if (exp_bits < 1 || exp_bits > 8) {
        throw std::runtime_error("split(): exp_bits must be in [1, 8]");
    }
    if (mant_bits < 1 || mant_bits > 16) {
        throw std::runtime_error("split(): mant_bits must be in [1, 16]");
    }

    auto xbuf = x.request();

    std::vector<ssize_t> shape(xbuf.ndim);
    ssize_t n = 1;
    for (ssize_t k = 0; k < xbuf.ndim; ++k) {
        shape[k] = xbuf.shape[k];
        n *= xbuf.shape[k];
    }

    const T* xp = static_cast<const T*>(xbuf.ptr);

    // sign: packed bits
    const ssize_t n_sign_bytes = ceil_int_div(n, 8);
    py::array_t<uint8_t> sign(n_sign_bytes);
    auto sbuf = sign.request();
    auto* sp = static_cast<uint8_t*>(sbuf.ptr);
    std::memset(sp, 0, static_cast<size_t>(n_sign_bytes));

    // exp: packed if exp_bits < 8 else dense uint8[n]
    py::array exp;
    uint8_t* exp_packed_ptr = nullptr;
    uint8_t* exp_u8_ptr = nullptr;
    if (exp_bits == 8) {
        py::array_t<uint8_t> exp8(n);
        exp = exp8;
        auto ebuf = exp8.request();
        exp_u8_ptr = static_cast<uint8_t*>(ebuf.ptr);
    } else {
        const ssize_t n_exp_bytes = ceil_int_div(n * exp_bits, 8);
        py::array_t<uint8_t> expb(n_exp_bytes);
        exp = expb;
        auto ebuf = expb.request();
        exp_packed_ptr = static_cast<uint8_t*>(ebuf.ptr);
        std::memset(exp_packed_ptr, 0, static_cast<size_t>(n_exp_bytes));
    }

    // mant: packed if mant_bits < 16 else dense uint16[n]
    py::array mant;
    uint8_t*  mant_packed_ptr = nullptr;
    uint16_t* mant_u16_ptr = nullptr;
    if (mant_bits == 16) {
        py::array_t<uint16_t> mant16(n);
        mant = mant16;
        auto mbuf = mant16.request();
        mant_u16_ptr = static_cast<uint16_t*>(mbuf.ptr);
    } else {
        const ssize_t n_mant_bytes = ceil_int_div(n * mant_bits, 8);
        py::array_t<uint8_t> mantb(n_mant_bytes);
        mant = mantb;
        auto mbuf = mantb.request();
        mant_packed_ptr = static_cast<uint8_t*>(mbuf.ptr);
        std::memset(mant_packed_ptr, 0, static_cast<size_t>(n_mant_bytes));
    }

    const uint32_t exp_mask_small  = mask_u32(exp_bits);
    const uint32_t mant_mask_small = (mant_bits == 16) ? 0xFFFFu : mask_u32(mant_bits);

    const uint32_t emax = (1u << exp_bits) - 1u; // reserved for inf/nan

    BitWriter ew;
    BitWriter mw;
    if (exp_bits != 8) { ew.out = exp_packed_ptr; ew.bitpos = 0; }
    if (mant_bits != 16) { mw.out = mant_packed_ptr; mw.bitpos = 0; }

    for (ssize_t i = 0; i < n; ++i) {
        UInt bits;
        std::memcpy(&bits, &xp[i], sizeof(T));

        const UInt s = (bits >> Tr::SIGN_SHIFT) & UInt(1);
        const UInt e_ieee = (bits >> Tr::EXP_SHIFT) & Tr::EXP_MASK;
        const UInt m_ieee = bits & Tr::MANT_MASK;

        // sign pack
        const ssize_t sign_idx = i >> 3;
        const uint8_t smask = uint8_t(1u << (i & 7));
        if (s) sp[sign_idx] |= smask;

        uint32_t e_small = 0;
        uint32_t m_small = 0;

        const bool is_zero_or_sub = (e_ieee == 0);
        const bool is_inf_or_nan  = (e_ieee == Tr::EXP_MASK);
        const bool is_zero = is_zero_or_sub && (m_ieee == 0);
        const bool is_sub  = is_zero_or_sub && (m_ieee != 0);
        const bool is_inf  = is_inf_or_nan && (m_ieee == 0);
        const bool is_nan  = is_inf_or_nan && (m_ieee != 0);

        if (is_zero || is_sub) {
            // Policy: flush subnormals to zero in the small format
            e_small = 0;
            m_small = 0;
        } else if (is_inf) {
            e_small = emax;
            m_small = 0;
        } else if (is_nan) {
            e_small = emax;
            m_small = 1; // minimal NaN payload
        } else {
            // Normal IEEE
            // Unbias exponent, then rebias to small format.
            int unbiased = int(e_ieee) - Tr::EXP_BIAS;
            int e = unbiased + exp_bias;

            // Round mantissa to mant_bits, with possible carry into exponent
            uint32_t carry = 0;
            m_small = round_mantissa_top<UInt>(m_ieee, Tr::MANT_BITS, mant_bits, carry) & mant_mask_small;
            e += int(carry);

            // Map into IEEE-like small exponent ranges:
            // 0 => zero/sub bucket
            // 1..emax-1 => normals
            // emax => inf/nan
            if (e <= 0) {
                // underflow to zero (no subnormals)
                e_small = 0;
                m_small = 0;
            } else if (uint32_t(e) >= emax) {
                // overflow to +inf in small format
                e_small = emax;
                m_small = 0;
            } else {
                e_small = uint32_t(e);
            }
        }

        e_small &= exp_mask_small;

        // write exp
        if (exp_bits == 8) {
            exp_u8_ptr[i] = uint8_t(e_small);
        } else {
            ew.write(e_small, exp_bits);
        }

        // write mant
        if (mant_bits == 16) {
            mant_u16_ptr[i] = uint16_t(m_small);
        } else {
            mw.write(m_small, mant_bits);
        }
    }

    // shape tuple
    py::tuple shape_t(static_cast<py::ssize_t>(shape.size()));
    for (py::ssize_t k = 0; k < (py::ssize_t)shape.size(); ++k) {
        shape_t[k] = py::int_(shape[(size_t)k]);
    }

    return py::make_tuple(sign, exp, mant, shape_t);
}

// ---------------- join_impl ----------------
template <typename T>
py::array_t<T> join_impl(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> sign,
    py::array exp_any,
    py::array mant_any,
    py::tuple shape_t,
    int exp_bits,
    int exp_bias,
    int mant_bits
) {
    using Tr = IEEETraits<T>;
    using UInt = typename Tr::UInt;

    if (exp_bits < 1 || exp_bits > 8) throw std::runtime_error("join(): exp_bits must be in [1, 8]");
    if (mant_bits < 1 || mant_bits > 16) throw std::runtime_error("join(): mant_bits must be in [1, 16]");

    // Parse shape
    std::vector<ssize_t> shape((size_t)shape_t.size());
    ssize_t n = 1;
    for (py::ssize_t k = 0; k < shape_t.size(); ++k) {
        shape[(size_t)k] = shape_t[k].cast<ssize_t>();
        n *= shape[(size_t)k];
    }

    // sign
    auto sbuf = sign.request();
    const ssize_t n_sign_bytes = ceil_int_div(n, 8);
    if (sbuf.ndim != 1 || sbuf.shape[0] != n_sign_bytes) {
        throw std::runtime_error("join(): sign length mismatch");
    }
    const uint8_t* sp = static_cast<const uint8_t*>(sbuf.ptr);

    // exp
    const uint8_t* exp_packed_ptr = nullptr;
    const uint8_t* exp_u8_ptr = nullptr;
    if (exp_bits == 8) {
        auto exp8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(exp_any);
        auto ebuf = exp8.request();
        if (ebuf.ndim != 1 || ebuf.shape[0] != n) throw std::runtime_error("join(): exp(u8) length mismatch");
        exp_u8_ptr = static_cast<const uint8_t*>(ebuf.ptr);
    } else {
        auto expb = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(exp_any);
        auto ebuf = expb.request();
        const ssize_t n_exp_bytes = ceil_int_div(n * exp_bits, 8);
        if (ebuf.ndim != 1 || ebuf.shape[0] != n_exp_bytes) throw std::runtime_error("join(): exp(packed) length mismatch");
        exp_packed_ptr = static_cast<const uint8_t*>(ebuf.ptr);
    }

    // mant
    const uint8_t*  mant_packed_ptr = nullptr;
    const uint16_t* mant_u16_ptr = nullptr;
    if (mant_bits == 16) {
        auto mant16 = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>(mant_any);
        auto mbuf = mant16.request();
        if (mbuf.ndim != 1 || mbuf.shape[0] != n) throw std::runtime_error("join(): mant(uint16) length mismatch");
        mant_u16_ptr = static_cast<const uint16_t*>(mbuf.ptr);
    } else {
        auto mantb = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(mant_any);
        auto mbuf = mantb.request();
        const ssize_t n_mant_bytes = ceil_int_div(n * mant_bits, 8);
        if (mbuf.ndim != 1 || mbuf.shape[0] != n_mant_bytes) throw std::runtime_error("join(): mant(packed) length mismatch");
        mant_packed_ptr = static_cast<const uint8_t*>(mbuf.ptr);
    }

    py::array_t<T> x(shape);
    auto xbuf = x.request();
    T* xp = static_cast<T*>(xbuf.ptr);

    const uint32_t exp_mask_small  = mask_u32(exp_bits);
    const uint32_t mant_mask_small = (mant_bits == 16) ? 0xFFFFu : mask_u32(mant_bits);
    const uint32_t emax_small = (1u << exp_bits) - 1u;

    BitReader er;
    BitReader mr;
    if (exp_bits != 8) { er.in = exp_packed_ptr; er.bitpos = 0; }
    if (mant_bits != 16) { mr.in = mant_packed_ptr; mr.bitpos = 0; }

    for (ssize_t i = 0; i < n; ++i) {
        // sign
        const ssize_t sign_idx = i >> 3;
        const uint8_t smask = uint8_t(1u << (i & 7));
        const UInt s = (sp[sign_idx] & smask) ? UInt(1) : UInt(0);

        // exponent
        uint32_t e_small = 0;
        if (exp_bits == 8) {
            e_small = uint32_t(exp_u8_ptr[i]) & exp_mask_small;
        } else {
            e_small = er.read(exp_bits) & exp_mask_small;
        }

        // mantissa
        uint32_t m_small = 0;
        if (mant_bits == 16) {
            m_small = uint32_t(mant_u16_ptr[i]) & mant_mask_small;
        } else {
            m_small = mr.read(mant_bits) & mant_mask_small;
        }

        UInt e_ieee = 0;
        UInt m_ieee = 0;

        if (e_small == 0) {
            // Policy: represent as zero (no IEEE subnormals reconstructed)
            e_ieee = 0;
            m_ieee = 0;
        } else if (e_small == emax_small) {
            // Inf/NaN
            e_ieee = Tr::EXP_MASK;
            m_ieee = (m_small == 0) ? UInt(0) : UInt(1); // minimal qNaN-ish payload
        } else {
            // Normal
            int unbiased = int(e_small) - exp_bias;
            int e = unbiased + Tr::EXP_BIAS;

            // If underflow/overflow in IEEE target type, map to 0 or Inf
            const int emax_ieee = (1 << Tr::EXP_BITS) - 1; // includes inf/nan code
            if (e <= 0) {
                // underflow to zero
                e_ieee = 0;
                m_ieee = 0;
            } else if (e >= emax_ieee) {
                // overflow to inf
                e_ieee = Tr::EXP_MASK;
                m_ieee = 0;
            } else {
                e_ieee = UInt(e) & Tr::EXP_MASK;

                // Expand mantissa into IEEE mantissa field (left-align)
                const int shift = Tr::MANT_BITS - mant_bits;
                if (shift >= 0) {
                    m_ieee = (UInt(m_small) << shift) & Tr::MANT_MASK;
                } else {
                    // mant_bits > Tr::MANT_BITS (shouldn't happen with mant_bits<=16)
                    m_ieee = UInt(m_small) & Tr::MANT_MASK;
                }
            }
        }

        const UInt bits = (s << Tr::SIGN_SHIFT) | (e_ieee << Tr::EXP_SHIFT) | (m_ieee & Tr::MANT_MASK);
        T out;
        std::memcpy(&out, &bits, sizeof(T));
        xp[i] = out;
    }

    return x;
}

// ---- Python-visible wrappers ----
py::tuple split_f32(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                    int exp_bits=4, int exp_bias=7, int mant_bits=16) {
    return split_impl<float>(x, exp_bits, exp_bias, mant_bits);
}

py::tuple split_f64(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                    int exp_bits=4, int exp_bias=7, int mant_bits=16) {
    return split_impl<double>(x, exp_bits, exp_bias, mant_bits);
}

py::array_t<float> join_f32(py::array_t<uint8_t> s, py::array e, py::array m, py::tuple sh,
                            int exp_bits=4, int exp_bias=7, int mant_bits=16) {
    return join_impl<float>(s, e, m, sh, exp_bits, exp_bias, mant_bits);
}

py::array_t<double> join_f64(py::array_t<uint8_t> s, py::array e, py::array m, py::tuple sh,
                             int exp_bits=4, int exp_bias=7, int mant_bits=16) {
    return join_impl<double>(s, e, m, sh, exp_bits, exp_bias, mant_bits);
}

PYBIND11_MODULE(vpfloat, m) {
    m.doc() = "Split/join float32/float64 into packed sign + packed exp (if exp_bits<8) + packed mant (if mant_bits<16)";

    m.def("split_f32", &split_f32,
          py::arg("x"),
          py::arg("exp_bits") = 4,
          py::arg("exp_bias") = 7,
          py::arg("mant_bits") = 16);

    m.def("join_f32", &join_f32,
          py::arg("sign"),
          py::arg("exp"),
          py::arg("mant"),
          py::arg("shape"),
          py::arg("exp_bits") = 4,
          py::arg("exp_bias") = 7,
          py::arg("mant_bits") = 16);

    m.def("split_f64", &split_f64,
          py::arg("x"),
          py::arg("exp_bits") = 4,
          py::arg("exp_bias") = 7,
          py::arg("mant_bits") = 16);

    m.def("join_f64", &join_f64,
          py::arg("sign"),
          py::arg("exp"),
          py::arg("mant"),
          py::arg("shape"),
          py::arg("exp_bits") = 4,
          py::arg("exp_bias") = 7,
          py::arg("mant_bits") = 16);
}
