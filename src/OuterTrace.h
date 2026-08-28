#pragma once

// Per-outer state trace -- RASBERY_OUTER_TRACE, default OFF.
//
// WHAT IT IS FOR.  A trajectory difference between two arms that agree on the
// converged answer ("same k_eff, one extra outer") cannot be localised from the
// answer: by the time the two runs disagree visibly, every array has been
// rewritten hundreds of times.  What localises it is the FIRST outer at which
// any carried quantity differs, and that needs the same five numbers printed
// from the same place in both arms.
//
// WHY HASHES AND NOT VALUES.  psi is [nxyz], jnet and dhat are [nsurf*ng], flux
// is [nxyz*ng] -- at APR1400 size that is ~100k doubles per outer, and printing
// them would produce a file nobody can diff.  A 64-bit FNV-1a over the raw bytes
// is enough to say WHICH array moved first; once that is known the array itself
// can be dumped for the one outer that matters.
//
// IT HASHES THE HOST ARRAYS, deliberately, and that is what makes the two arms
// comparable.  The device arm mirrors psi and dhat back and bridges jnet, so the
// host copies are live in both arms and a hash of them is like for like.  A hash
// of device memory would only exist on one side.
//
// COST WHEN OFF: one cached bool test per outer.  The environment read is a
// function-local static, so the whole tracer folds to a predictable branch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace rasbery::outertrace {

[[nodiscard]] inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_OUTER_TRACE");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return on;
}

/// FNV-1a over the raw bytes of a double array.
///
/// Over BYTES rather than values so that -0.0 and +0.0 hash differently and a
/// NaN payload is not collapsed: the question this answers is "are these arrays
/// the same bits", and a value-based hash would answer a weaker one.
[[nodiscard]] inline std::uint64_t hashDoubles(const double* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ull;
    if (p == nullptr) return h;
    const auto* b = reinterpret_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n * sizeof(double); ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

/// One line per outer.  `tag` is the arm ("host" or "dev") so a single grep
/// separates them when both are in one log.
inline void emit(int statepoint, int outer, const char* tag, double eigv, double residual,
                 double prev_inner, std::uint64_t psi, std::uint64_t jnet,
                 std::uint64_t dhat, std::uint64_t flux) {
    std::fprintf(stderr,
                 "[RASBERY][OUTER_TRACE] sp=%d outer=%d arm=%s eigv=%.17g res=%.17g "
                 "prev=%.17g psi=%016llx jnet=%016llx dhat=%016llx flux=%016llx\n",
                 statepoint, outer, tag, eigv, residual, prev_inner,
                 static_cast<unsigned long long>(psi),
                 static_cast<unsigned long long>(jnet),
                 static_cast<unsigned long long>(dhat),
                 static_cast<unsigned long long>(flux));
}

} // namespace rasbery::outertrace
