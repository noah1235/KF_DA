#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static inline int ceil_int_div(int a, int b){
    return (a + b - 1) / b;
}


// ---- Python-facing functions ----

// split(x) -> (sign, exp, mant)
py::tuple split(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int exp_bits = 4,
    int exp_bias = 7
)
{
    const int float_32_bias = 127;
    auto xbuf = x.request();

    // copy shape into a safe container
    std::vector<ssize_t> shape(xbuf.ndim);
    ssize_t n = 1;
    for (ssize_t k = 0; k < xbuf.ndim; ++k) {
        shape[k] = xbuf.shape[k];
        n *= xbuf.shape[k];
    }

    const float* xp = static_cast<const float*>(xbuf.ptr);

    // allocate outputs
    const ssize_t n_sign_bytes = ceil_int_div(n, 8);
    const ssize_t n_exp_bytes  = ceil_int_div(n * exp_bits, 8);

    py::array_t<uint8_t>  sign(n_sign_bytes);   // packed bits
    py::array_t<uint8_t>  exp(n_exp_bytes);     // packed exp_bits per element
    py::array_t<uint16_t> mant(shape);          // fixed 16-bit mantissa per element

    auto sbuf = sign.request();
    auto ebuf = exp.request();
    auto mbuf = mant.request();

    auto* sp = static_cast<uint8_t*>(sbuf.ptr);
    auto* ep = static_cast<uint8_t*>(ebuf.ptr);
    auto* mp = static_cast<uint16_t*>(mbuf.ptr);

    // IMPORTANT: zero init packed outputs
    std::memset(sp, 0, static_cast<size_t>(n_sign_bytes));
    std::memset(ep, 0, static_cast<size_t>(n_exp_bytes));

    const uint32_t exp_mask = (exp_bits >= 32) ? 0xFFFFFFFFu : ((1u << exp_bits) - 1u);

    int exp_arr_idx = 0; // byte index into ep
    int exp_bit_idx = 0; // bit offset [0,7] within current byte

    for (ssize_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &xp[i], sizeof(float));

        // packed sign
        const ssize_t sign_idx = i >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (i & 7));
        if ((bits >> 31) & 1u) sp[sign_idx] |= mask;

        // exponent: signed math + clamp
        int e32 = int((bits >> 23) & 0xFFu);
        int e   = e32 - float_32_bias + exp_bias;
        const int emax = (1 << exp_bits) - 1;
        if (e < 0) e = 0;
        if (e > emax) e = emax;
        uint32_t e_u = static_cast<uint32_t>(e) & exp_mask;

        // pack exponent bits
        if (exp_bits + exp_bit_idx <= 8) {
            ep[exp_arr_idx] |= static_cast<uint8_t>(e_u << exp_bit_idx);
            exp_bit_idx += exp_bits;
            if (exp_bit_idx == 8) {
                exp_bit_idx = 0;
                exp_arr_idx += 1;
            }
        } else {
            const int lo = 8 - exp_bit_idx;
            const int hi = exp_bits - lo;
            ep[exp_arr_idx] |= static_cast<uint8_t>(e_u << exp_bit_idx);
            exp_arr_idx += 1;
            ep[exp_arr_idx] |= static_cast<uint8_t>(e_u >> lo);
            exp_bit_idx = hi;
        }

        // mantissa: top 16 bits of IEEE mantissa
        if (e == 0) {
            mp[i] = 0;
        } else {
            const uint32_t mant23 = bits & 0x7FFFFFu;
            mp[i] = static_cast<uint16_t>(mant23 >> 7);
        }
    }

    return py::make_tuple(sign, exp, mant);
}



// join(sign_packed, exp_packed, mant16_top) -> x
py::array_t<float> join(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> sign,
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> exp,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> mant,
    int exp_bits = 4,
    int exp_bias = 7
)
{
    const int float_32_bias = 127;

    auto sbuf = sign.request();
    auto ebuf = exp.request();
    auto mbuf = mant.request();

    // Build shape from mant
    std::vector<ssize_t> shape(mbuf.ndim);
    ssize_t n = 1;
    for (ssize_t k = 0; k < mbuf.ndim; ++k) {
        shape[k] = mbuf.shape[k];
        n *= mbuf.shape[k];
    }

    // sign shape check
    const ssize_t n_sign_bytes = (n + 7) / 8;
    if (sbuf.ndim != 1 || sbuf.shape[0] != n_sign_bytes) {
        throw std::runtime_error("join(): sign must be 1D with length ceil(n/8)");
    }

    // exp shape check
    const ssize_t n_exp_bytes = (n * exp_bits + 7) / 8;
    if (ebuf.ndim != 1 || ebuf.shape[0] != n_exp_bytes) {
        throw std::runtime_error("join(): exp must be 1D with length ceil(n*exp_bits/8)");
    }

    const uint8_t*  sp = static_cast<const uint8_t*>(sbuf.ptr);
    const uint8_t*  ep = static_cast<const uint8_t*>(ebuf.ptr);
    const uint16_t* mp = static_cast<const uint16_t*>(mbuf.ptr);

    py::array_t<float> x(shape);
    auto xbuf = x.request();
    float* xp = static_cast<float*>(xbuf.ptr);

    const uint32_t exp_mask = (exp_bits >= 32) ? 0xFFFFFFFFu : ((1u << exp_bits) - 1u);

    size_t bitpos = 0;

    for (ssize_t i = 0; i < n; ++i) {
        // sign
        const ssize_t sign_idx = i >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (i & 7));
        const uint32_t s = (sp[sign_idx] & mask) ? 1u : 0u;

        // unpack exponent
        const size_t byte = bitpos >> 3;
        const int off = static_cast<int>(bitpos & 7);

        uint32_t chunk = static_cast<uint32_t>(ep[byte]);
        if (off + exp_bits > 8) {
            chunk |= (static_cast<uint32_t>(ep[byte + 1]) << 8);
        }

        uint32_t e_small = (chunk >> off) & exp_mask;
        bitpos += static_cast<size_t>(exp_bits);

        // map back to IEEE exponent
        uint32_t e;
        if (e_small == 0) {
            e = 0;
        } else {
            int e32 = static_cast<int>(e_small) - exp_bias + float_32_bias;
            if (e32 < 1)   e32 = 1;
            if (e32 > 254) e32 = 254;
            e = static_cast<uint32_t>(e32) & 0xFFu;
        }

        // mantissa reconstruction
        const uint32_t mant16 = static_cast<uint32_t>(mp[i]);
        const uint32_t mant23 = (mant16 << 7) & 0x7FFFFFu;

        const uint32_t bits = (s << 31) | (e << 23) | mant23;

        float out;
        std::memcpy(&out, &bits, sizeof(float));
        xp[i] = out;
    }

    return x;
}

PYBIND11_MODULE(vpfloat, m) {
    m.doc() = "Split/join float32 into packed sign/exp/mant with variable exp_bits and mant_bits";

    m.def(
        "split",
        &split,
        py::arg("x"),
        py::arg("exp_bits") = 4,
        py::arg("exp_bias") = 7,
        py::arg("mant_bits") = 16
    );

    m.def(
        "join",
        &join,
        py::arg("sign"),
        py::arg("exp"),
        py::arg("mant"),
        py::arg("shape"),
        py::arg("exp_bits") = 4,
        py::arg("exp_bias") = 7,
        py::arg("mant_bits") = 16
    );
}
