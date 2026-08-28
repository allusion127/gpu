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

/// WHERE THE SEGMENT IS, so the device runner can label a step line without
/// being handed the Driver's loop counters through every signature between
/// them.  Set by Driver.h immediately before it delegates an outer, read by the
/// runner; meaningless (and unread) when the tracer is off.
///
/// A PROCESS GLOBAL AND NOT A PARAMETER, deliberately: the alternative is a
/// statepoint/outer pair threaded through OuterSegmentScalars, which is a
/// struct three contract tests pin, for a field only a debug mode reads.
struct Context {
    int statepoint = -1;
    int outer      = -1;
};

inline Context& context() {
    static Context c;
    return c;
}

inline void setContext(int statepoint, int outer) {
    Context& c   = context();
    c.statepoint = statepoint;
    c.outer      = outer;
}

/// One line per STEP of one outer -- the resolution the per-outer line cannot
/// give.
///
/// WHY IT EXISTS.  The per-outer line says WHICH outer first differs; on a body
/// whose five steps each rewrite a different array, that still leaves five
/// candidates, and the ON/OFF arms reach the same five arrays through different
/// code (host arithmetic vs a device kernel).  A per-step line names the step,
/// which is the difference between "the device outer is wrong" and "updpsi is
/// wrong".
///
/// TWO HASHES, BECAUSE TWO STEPS PRODUCE TWO ARRAYS.  The sweep produces flux
/// and an eigenvalue, and the nodal drive produces jnet and phis; the rest fill
/// only the first slot and pass a null second name.
///
/// THE DEVICE ARM MUST HASH DEVICE MEMORY.  The per-outer tracer hashes the
/// HOST arrays and says so, on the argument that the device arm mirrors psi and
/// dhat back and bridges jnet -- which stopped being true when the mirrors moved
/// to the segment exit and the bridge was dropped for the canonical binding.  A
/// host hash of the ON arm is now a hash of whatever the last mirror left there,
/// which is the one thing worse than no hash.  So the runner copies the device
/// buffer down for the hash; that is a synchronise and a transfer per step, and
/// it is affordable exactly because nothing but a debugging session turns it on.
inline void emitStep(const char* tag, const char* step, const char* name1,
                     std::uint64_t hash1, const char* name2, std::uint64_t hash2) {
    const Context& c = context();
    if (name2 == nullptr) {
        std::fprintf(stderr,
                     "[RASBERY][OUTER_TRACE][STEP] sp=%d outer=%d arm=%s step=%s "
                     "%s=%016llx\n",
                     c.statepoint, c.outer, tag, step, name1,
                     static_cast<unsigned long long>(hash1));
        return;
    }
    std::fprintf(stderr,
                 "[RASBERY][OUTER_TRACE][STEP] sp=%d outer=%d arm=%s step=%s "
                 "%s=%016llx %s=%016llx\n",
                 c.statepoint, c.outer, tag, step, name1,
                 static_cast<unsigned long long>(hash1), name2,
                 static_cast<unsigned long long>(hash2));
}

/// The sweep's step line also carries the eigenvalue, which is the one carried
/// scalar the flux hash cannot express.
inline void emitStepEigv(const char* tag, const char* step, std::uint64_t flux,
                         double eigv) {
    const Context& c = context();
    std::fprintf(stderr,
                 "[RASBERY][OUTER_TRACE][STEP] sp=%d outer=%d arm=%s step=%s "
                 "flux=%016llx eigv=%.17g\n",
                 c.statepoint, c.outer, tag, step,
                 static_cast<unsigned long long>(flux), eigv);
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
