// Verbatim CPU quotation of the Anderson algebra.  See XeAndersonReference.h
// for why this is a separate translation unit and why it must not include
// XeKernel.h.
//
// NOTHING IN THIS FILE MAY BE "TIDIED".  Every expression below is a character
// for character copy of the production loop it names, because the whole value
// of the file is that the host compiler treats it the same way it treats the
// original.  A helper extracted, a temporary named, an operand reordered:
// all of them are changes to what is being measured.

#include "XeAndersonReference.h"

#include <cmath>
#include <cstdint>

namespace xeref {

namespace {

/// splitmix64 -- a fixed, dependency-free stream, so the fixture is the same
/// sequence on every host and every run.
inline std::uint64_t nextBits(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// A double in [1, 2) scaled by 2^e, so the fixture spans decades rather than
/// clustering at one exponent (see the Fixture comment).
inline double nextScaled(std::uint64_t& s, int lo_exp, int hi_exp) {
    const std::uint64_t bits = nextBits(s);
    const double mant = 1.0 + static_cast<double>(bits >> 12) / 4503599627370496.0;
    const int    e    = lo_exp + static_cast<int>(bits % static_cast<std::uint64_t>(
                                                  hi_exp - lo_exp + 1));
    double scale = 1.0;
    for (int i = 0; i < (e < 0 ? -e : e); ++i) scale *= 2.0;
    return (e < 0) ? mant / scale : mant * scale;
}

} // namespace

Fixture buildFixture(int n) {
    Fixture f;
    f.n = n;
    const auto sz = static_cast<std::size_t>(n);
    f.a_i.resize(sz); f.a_x.resize(sz); f.a_m.resize(sz);
    f.b_i.resize(sz); f.b_x.resize(sz); f.b_m.resize(sz);
    f.f_i.resize(sz); f.f_x.resize(sz); f.f_m.resize(sz);
    f.d_i.resize(2 * sz); f.d_x.resize(2 * sz); f.d_m.resize(2 * sz);

    // EVERY ROW OVER THE SAME EXPONENT RANGE, and that is deliberate.  The
    // physical Xe-135m inventory is ~0.6 % of Xe-135, and a fixture that
    // reproduced that ratio made the third product of each inner-product term
    // some 2^-10 of the first two -- so its rounding fell below the ULP of the
    // running sum and BOTH forms of that site gave the same answer, on every
    // slice.  The site then mines as a don't-care, which is a statement about
    // the fixture and not about the compiler.  The mask records a CODEGEN
    // choice, and codegen does not know the magnitudes, so the fixture is
    // chosen to discriminate codegen: comparable rows, decades of spread within
    // each, signed because g and dG both change sign across the core.
    std::uint64_t s = 0x5EED13ull;
    for (std::size_t k = 0; k < sz; ++k) {
        f.a_i[k] = nextScaled(s, -50, -30) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.a_x[k] = nextScaled(s, -50, -30) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.a_m[k] = nextScaled(s, -50, -30) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.b_i[k] = nextScaled(s, -50, -30) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.b_x[k] = nextScaled(s, -50, -30) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.b_m[k] = nextScaled(s, -50, -30) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        // Inventories: positive, ~1e-9 atoms/barn-cm.
        f.f_i[k] = nextScaled(s, -32, -28);
        f.f_x[k] = nextScaled(s, -32, -28);
        f.f_m[k] = nextScaled(s, -32, -28);
    }
    for (std::size_t k = 0; k < 2 * sz; ++k) {
        f.d_i[k] = nextScaled(s, -45, -33) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.d_x[k] = nextScaled(s, -45, -33) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        f.d_m[k] = nextScaled(s, -45, -33) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
    }
    f.gamma[0] = 0.4371283945612;
    f.gamma[1] = -1.2094857361204;

    // WP7-C.  64 Gram cases, every one of them CONDITIONED: a and c positive
    // and decades apart, |b| < 0.86*sqrt(a*c) so det clears the 1e-8 relative
    // floor with room, p and q of the size <dG,g> actually takes and signed
    // because g changes sign across the core.  std::sqrt here is fixture
    // construction, not scored arithmetic -- nothing below the `refAlgebra`
    // quotation is measured.
    f.alg_cases = 64;
    f.alg.resize(static_cast<std::size_t>(6 * f.alg_cases));
    for (int ci = 0; ci < f.alg_cases; ++ci) {
        const double a    = nextScaled(s, -40, -20);
        const double c2   = nextScaled(s, -40, -20);
        const double root = std::sqrt(a * c2);
        const double frac = 0.05 + 0.85 * static_cast<double>(nextBits(s) >> 12) /
                                       4503599627370496.0;
        const double b  = root * frac * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        const double p  = nextScaled(s, -35, -18) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        const double q  = nextScaled(s, -35, -18) * ((nextBits(s) & 1ull) ? 1.0 : -1.0);
        const double gg = nextScaled(s, -30, -14);
        const auto   u  = static_cast<std::size_t>(6 * ci);
        f.alg[u + 0] = gg; // XE_DOT_GG
        f.alg[u + 1] = a;  // XE_DOT_A
        f.alg[u + 2] = b;  // XE_DOT_B
        f.alg[u + 3] = c2; // XE_DOT_C
        f.alg[u + 4] = p;  // XE_DOT_P
        f.alg[u + 5] = q;  // XE_DOT_Q
    }
    return f;
}

// Driver.h  static double XeDot(const XeTriple& a, const XeTriple& b), quoted.
// The host runs it over the whole 0..n-1 range; the range is a parameter here
// only so the mining has more than one word to score (XeFormMine.h).
double refDot(const Fixture& f, int i0, int i1) {
    const double* a_i135 = f.a_i.data();
    const double* a_xe135 = f.a_x.data();
    const double* a_xe135m = f.a_m.data();
    const double* b_i135 = f.b_i.data();
    const double* b_xe135 = f.b_x.data();
    const double* b_xe135m = f.b_m.data();
    double        sum      = 0.0;
    for (int i = i0; i < i1; ++i)
        sum += a_i135[i] * b_i135[i] + a_xe135[i] * b_xe135[i] +
               a_xe135m[i] * b_xe135m[i];
    return sum;
}

// Driver.h  TryAndersonXeStep, the "x_{k+1} = F_k - sum_j gamma_j dF_j" loop,
// quoted.  The host's df[j] is an array of structs; the flat [j*n + k] layout
// here is the device's, and the VALUES are the same -- the quotation is of the
// arithmetic, which is what the mask is about.
void refCandidate(const Fixture& f, int ncol, std::vector<double>& ci,
                  std::vector<double>& cx, std::vector<double>& cm) {
    const auto n = static_cast<std::size_t>(f.n);
    ci.resize(n);
    cx.resize(n);
    cm.resize(n);
    const double* gamma = f.gamma;
    for (std::size_t i = 0; i < n; ++i) {
        double vi = f.f_i[i];
        double vx = f.f_x[i];
        double vm = f.f_m[i];
        for (int j = 0; j < ncol; ++j) {
            vi -= gamma[j] * f.d_i[static_cast<std::size_t>(j) * n + i];
            vx -= gamma[j] * f.d_x[static_cast<std::size_t>(j) * n + i];
            vm -= gamma[j] * f.d_m[static_cast<std::size_t>(j) * n + i];
        }
        ci[i] = vi;
        cx[i] = vx;
        cm[i] = vm;
    }
}

// Driver.h  TryAndersonXeStepGpu, the normal-equations block, quoted.  The
// `dots[]` reads are spelled out as locals with the SAME NAMES the production
// arm uses, because what is being measured is what the compiler does to the
// four expressions below and to nothing else.
bool refAlgebra(const Fixture& f, int idx, int ncol, double min_gram, double gamma_out[2],
                double* proj_out, double* det_out) {
    const auto   u        = static_cast<std::size_t>(6 * idx);
    const double gg       = f.alg[u + 0];
    double       gamma[2] = {0.0, 0.0};
    double       proj     = 0.0;
    bool         solved   = false;
    *det_out              = 0.0;
    if (ncol == 2) {
        const double a   = f.alg[u + 1];
        const double b   = f.alg[u + 2];
        const double c   = f.alg[u + 3];
        const double p   = f.alg[u + 4];
        const double q   = f.alg[u + 5];
        const double det = a * c - b * b;
        *det_out         = det;
        if (a > 0.0 && c > 0.0 && std::isfinite(det) && std::isfinite(p) &&
            std::isfinite(q) && det > min_gram * a * c) {
            gamma[0] = (c * p - b * q) / det;
            gamma[1] = (a * q - b * p) / det;
            proj     = gamma[0] * p + gamma[1] * q;
            solved   = true;
        }
    }
    if (!solved) {
        const int    j = ncol - 1;
        const double a = (j == 1) ? f.alg[u + 3] : f.alg[u + 1];
        const double p = (j == 1) ? f.alg[u + 5] : f.alg[u + 4];
        if (a > 0.0 && std::isfinite(a) && std::isfinite(p) && a > min_gram * gg) {
            for (double& gj : gamma)
                gj = 0.0;
            gamma[j] = p / a;
            proj     = gamma[j] * p;
            solved   = true;
        }
    }
    gamma_out[0] = gamma[0];
    gamma_out[1] = gamma[1];
    *proj_out    = proj;
    return solved;
}

} // namespace xeref
