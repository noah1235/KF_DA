#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static inline ssize_t ceil_int_div(ssize_t a, ssize_t b) { return (a + b - 1) / b; }

// Bit-pack/unpack nbits LSB-first into a uint8 byte stream
static inline void pack_bits_u8(uint8_t* out, uint32_t v, int nbits, size_t& bitpos) {
    for (int b = 0; b < nbits; ++b) {
        const size_t byte = bitpos >> 3;
        const int off = int(bitpos & 7);
        const uint8_t bit = uint8_t((v >> b) & 1u);
        out[byte] = uint8_t(out[byte] | (bit << off));
        ++bitpos;
    }
}

static inline uint32_t unpack_bits_u8(const uint8_t* in, int nbits, size_t& bitpos) {
    uint32_t v = 0;
    for (int b = 0; b < nbits; ++b) {
        const size_t byte = bitpos >> 3;
        const int off = int(bitpos & 7);
        const uint32_t bit = (uint32_t(in[byte]) >> off) & 1u;
        v |= (bit << b);
        ++bitpos;
    }
    return v;
}

// ---- Traits for IEEE layouts ----
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

// ---- split implementation (templated) ----
// Returns: (sign_packed_u8, exp (u8 packed if exp_bits<8 else u8[n]),
//           mant (u8 packed if mant_bits<16 else int16[n]), shape_tuple)
template <typename T>
py::tuple split_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    int exp_bits,
    int exp_bias,
    int mant_bits
) {
    using Tr = IEEETraits<T>;
    using UInt = typename Tr::UInt;

    if (exp_bits <= 0 || exp_bits > 8) {
        throw std::runtime_error("split(): exp_bits must be in [1, 8]");
    }
    if (mant_bits <= 0 || mant_bits > 16) {
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

    // mant: packed if mant_bits < 16 else dense int16[n]
    py::array mant;
    uint8_t* mant_packed_ptr = nullptr;
    int16_t* mant_i16_ptr = nullptr;
    if (mant_bits == 16) {
        py::array_t<int16_t> mant16(n);
        mant = mant16;
        auto mbuf = mant16.request();
        mant_i16_ptr = static_cast<int16_t*>(mbuf.ptr);
    } else {
        const ssize_t n_mant_bytes = ceil_int_div(n * mant_bits, 8);
        py::array_t<uint8_t> mantb(n_mant_bytes);
        mant = mantb;
        auto mbuf = mantb.request();
        mant_packed_ptr = static_cast<uint8_t*>(mbuf.ptr);
        std::memset(mant_packed_ptr, 0, static_cast<size_t>(n_mant_bytes));
    }

    const uint32_t exp_mask_small  = (1u << exp_bits) - 1u;
    const uint32_t mant_mask_small = (mant_bits == 16) ? 0xFFFFu : ((1u << mant_bits) - 1u);
    const int emax = (1 << exp_bits) - 1;

    size_t exp_bitpos = 0;
    size_t mant_bitpos = 0;

    for (ssize_t i = 0; i < n; ++i) {
        UInt bits;
        std::memcpy(&bits, &xp[i], sizeof(T));

        // sign bit pack
        const ssize_t sign_idx = i >> 3;
        const uint8_t smask = static_cast<uint8_t>(1u << (i & 7));
        if ((bits >> Tr::SIGN_SHIFT) & 1u) sp[sign_idx] |= smask;

        // exponent field extract + rebias to your custom exponent
        int e_ieee = int((bits >> Tr::EXP_SHIFT) & Tr::EXP_MASK);
        int e = e_ieee - Tr::EXP_BIAS + exp_bias;
        if (e < 0) e = 0;
        if (e > emax) e = emax;
        const uint32_t e_small = uint32_t(e) & exp_mask_small;

        if (exp_bits == 8) {
            exp_u8_ptr[i] = static_cast<uint8_t>(e_small);
        } else {
            pack_bits_u8(exp_packed_ptr, e_small, exp_bits, exp_bitpos);
        }

        // mantissa extract: take top mant_bits of IEEE mantissa
        UInt mant_ieee = bits & Tr::MANT_MASK;
        const int shift = Tr::MANT_BITS - mant_bits;  // 23-m or 52-m
        uint32_t m_small = (shift > 0) ? uint32_t(mant_ieee >> shift) : uint32_t(mant_ieee);
        m_small &= mant_mask_small;

        if (mant_bits == 16) {
            mant_i16_ptr[i] = static_cast<int16_t>(m_small);
        } else {
            pack_bits_u8(mant_packed_ptr, m_small, mant_bits, mant_bitpos);
        }
    }

    // return shape as a Python tuple so join can reconstruct
    py::tuple shape_t(static_cast<py::ssize_t>(shape.size()));
    for (py::ssize_t k = 0; k < (py::ssize_t)shape.size(); ++k) {
        shape_t[k] = py::int_(shape[(size_t)k]);
    }

    return py::make_tuple(sign, exp, mant, shape_t);
}

// ---- join implementation (templated) ----
// Accepts exp as uint8 packed if exp_bits<8 else uint8[n]
// Accepts mant as uint8 packed if mant_bits<16 else int16[n]
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

    if (exp_bits <= 0 || exp_bits > 8) throw std::runtime_error("join(): exp_bits must be in [1, 8]");
    if (mant_bits <= 0 || mant_bits > 16) throw std::runtime_error("join(): mant_bits must be in [1, 16]");

    auto sbuf = sign.request();

    // Parse shape
    std::vector<ssize_t> shape((size_t)shape_t.size());
    ssize_t n = 1;
    for (py::ssize_t k = 0; k < shape_t.size(); ++k) {
        shape[(size_t)k] = shape_t[k].cast<ssize_t>();
        n *= shape[(size_t)k];
    }

    // Validate sign length
    const ssize_t n_sign_bytes = ceil_int_div(n, 8);
    if (sbuf.ndim != 1 || sbuf.shape[0] != n_sign_bytes) {
        throw std::runtime_error("join(): sign length mismatch");
    }
    const uint8_t* sp = static_cast<const uint8_t*>(sbuf.ptr);

    // exp pointers based on exp_bits
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

    // mant pointers based on mant_bits
    const uint8_t* mant_packed_ptr = nullptr;
    const int16_t* mant_i16_ptr = nullptr;
    if (mant_bits == 16) {
        auto mant16 = py::array_t<int16_t, py::array::c_style | py::array::forcecast>(mant_any);
        auto mbuf = mant16.request();
        if (mbuf.ndim != 1 || mbuf.shape[0] != n) throw std::runtime_error("join(): mant(int16) length mismatch");
        mant_i16_ptr = static_cast<const int16_t*>(mbuf.ptr);
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

    const uint32_t exp_mask_small  = (1u << exp_bits) - 1u;
    const uint32_t mant_mask_small = (mant_bits == 16) ? 0xFFFFu : ((1u << mant_bits) - 1u);

    size_t exp_bitpos = 0;
    size_t mant_bitpos = 0;

    for (ssize_t i = 0; i < n; ++i) {
        // sign
        const ssize_t sign_idx = i >> 3;
        const uint8_t smask = static_cast<uint8_t>(1u << (i & 7));
        const UInt s = (sp[sign_idx] & smask) ? UInt(1) : UInt(0);

        // exponent small (packed or dense)
        uint32_t e_small;
        if (exp_bits == 8) {
            e_small = uint32_t(exp_u8_ptr[i]) & exp_mask_small;
        } else {
            e_small = unpack_bits_u8(exp_packed_ptr, exp_bits, exp_bitpos) & exp_mask_small;
        }

        // mantissa small (packed or dense)
        uint32_t m_small;
        if (mant_bits == 16) {
            m_small = uint32_t(uint16_t(mant_i16_ptr[i])) & mant_mask_small;
        } else {
            m_small = unpack_bits_u8(mant_packed_ptr, mant_bits, mant_bitpos) & mant_mask_small;
        }

        // Map back to IEEE exponent field
        UInt e_ieee;
        if (e_small == 0) {
            e_ieee = 0; // your policy
        } else {
            int e = int(e_small) - exp_bias + Tr::EXP_BIAS;
            if (e < 1) e = 1;
            const int emax_ieee = (1 << Tr::EXP_BITS) - 2; // 254 for f32, 2046 for f64
            if (e > emax_ieee) e = emax_ieee;
            e_ieee = UInt(e) & Tr::EXP_MASK;
        }

        // Expand mantissa back to IEEE mantissa width
        const int shift = Tr::MANT_BITS - mant_bits;
        UInt mant_ieee = (shift > 0) ? (UInt(m_small) << shift) : UInt(m_small);
        mant_ieee &= Tr::MANT_MASK;

        const UInt bits = (s << Tr::SIGN_SHIFT) | (e_ieee << Tr::EXP_SHIFT) | mant_ieee;

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
