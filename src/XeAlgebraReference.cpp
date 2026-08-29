// Verbatim CPU quotation of the WP7-C normal equations, AND ITS OWN FIXTURE.
//
// WHY THIS IS A THIRD TRANSLATION UNIT AND NOT THREE MORE FUNCTIONS IN
// XeAndersonReference.cpp
// ---------------------------------------------------------------------------
//
// XeAndersonReference.cpp exists because a quotation and the shipped body must
// not share a translation unit: gcc common-subexpressions across them and
// changes what the reference measures.  The same argument applies ONE LEVEL
// DOWN, and WP7-C (71092e2) did not apply it.
//
// That commit put `refAlgebra`, `#include <cmath>` and thirty-three more lines
// of fixture construction -- `std::sqrt`, four more draws per case, a 64-case
// loop -- INTO XeAndersonReference.cpp, directly ahead of `refDot` and
// `refCandidate` and inside the very function that produces their operands.
// Those two quotations are what bits 0..4 of RASBERY_XE_FORMS are mined
// against, and the mined mask is not a test artefact: XeFormMiner.cpp resolves
// it at startup and hands it to kXeDotStage1 and kXeCandidate
// (CudaXsReconBackend.cu:2806, 2835), which are the LIVE device Xe Anderson
// path whenever RASBERY_GPU_XE=1.  A codegen shift in this file is therefore a
// physics change in the production arm, with no flag moved and no gate asked.
//
// So the WP7-C quotation gets the separation the WP7-A one already had.  What
// is in here is measured against nothing else, and what is in
// XeAndersonReference.cpp is, character for character, what it was before
// WP7-C -- which is the only form in which the FORMS receipt can be compared
// across the two binaries.
//
// NOTHING IN THIS FILE MAY BE "TIDIED", for the reason the sibling states.
//
// THE FIXTURE IS SEEDED HERE, NOT CONTINUED FROM buildFixture().  WP7-C drew
// its 64 Gram cases from the tail of buildFixture's splitmix stream, which is
// what forced the block into that function in the first place.  A separate
// constant seed is just as deterministic, is the same on every host and every
// run, and costs the WP7-C sites nothing -- they were never mined on the
// authoring host anyway (XeKernel.h: the shipped bits 5..12 are 0 because
// nothing was measured, not because 0 was measured).

#include "XeAndersonReference.h"

#include <cmath>
#include <cstdint>

namespace xeref {

namespace {

/// splitmix64, the same stream function XeAndersonReference.cpp uses.  A
/// private copy rather than a shared symbol: this file must be able to change
/// without relinking anything about the WP7-A quotation.
inline std::uint64_t algBits(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// A double in [1, 2) scaled by 2^e.
inline double algScaled(std::uint64_t& s, int lo_exp, int hi_exp) {
    const std::uint64_t bits = algBits(s);
    const double mant = 1.0 + static_cast<double>(bits >> 12) / 4503599627370496.0;
    const int    e    = lo_exp + static_cast<int>(bits % static_cast<std::uint64_t>(
                                                  hi_exp - lo_exp + 1));
    double scale = 1.0;
    for (int i = 0; i < (e < 0 ? -e : e); ++i) scale *= 2.0;
    return (e < 0) ? mant / scale : mant * scale;
}

} // namespace

void buildAlgebraFixture(Fixture& f) {
    // WP7-C.  64 Gram cases, every one of them CONDITIONED: a and c positive
    // and decades apart, |b| < 0.86*sqrt(a*c) so det clears the 1e-8 relative
    // floor with room, p and q of the size <dG,g> actually takes and signed
    // because g changes sign across the core.  std::sqrt here is fixture
    // construction, not scored arithmetic -- nothing below the `refAlgebra`
    // quotation is measured.
    std::uint64_t s = 0xA16E13ull;
    f.alg_cases     = 64;
    f.alg.resize(static_cast<std::size_t>(6 * f.alg_cases));
    for (int ci = 0; ci < f.alg_cases; ++ci) {
        const double a    = algScaled(s, -40, -20);
        const double c2   = algScaled(s, -40, -20);
        const double root = std::sqrt(a * c2);
        const double frac = 0.05 + 0.85 * static_cast<double>(algBits(s) >> 12) /
                                       4503599627370496.0;
        const double b  = root * frac * ((algBits(s) & 1ull) ? 1.0 : -1.0);
        const double p  = algScaled(s, -35, -18) * ((algBits(s) & 1ull) ? 1.0 : -1.0);
        const double q  = algScaled(s, -35, -18) * ((algBits(s) & 1ull) ? 1.0 : -1.0);
        const double gg = algScaled(s, -30, -14);
        const auto   u  = static_cast<std::size_t>(6 * ci);
        f.alg[u + 0] = gg; // XE_DOT_GG
        f.alg[u + 1] = a;  // XE_DOT_A
        f.alg[u + 2] = b;  // XE_DOT_B
        f.alg[u + 3] = c2; // XE_DOT_C
        f.alg[u + 4] = p;  // XE_DOT_P
        f.alg[u + 5] = q;  // XE_DOT_Q
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
