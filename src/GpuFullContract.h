#pragma once

// WP1(b): GPU FULL IS FAIL-CLOSED.  Plan Sec 6.3.
//
// THE PROBLEM.  Every GPU arm in this binary fails OPEN: a CUDA error, a shape
// it does not serve, a refused batch slot, a graph capture the driver would not
// take -- each one returns `false`, and the caller runs the host body, which is
// always correct.  That is the right default: a benchmark that crashes teaches
// nothing, and the host body IS the reference.  What it is not is a contract.
// A run can declare `RASBERY_GPU_NODAL=1 RASBERY_GPU_FLATXS=1 RASBERY_GPU_PPR=1`
// and spend every statepoint on the CPU, and the only trace is a one-shot
// stderr line nobody keeps and a counter three subsystems do not have
// (`BackendCounters::cmfd_cpu_fallbacks` and `nodal_cpu_fallbacks` are printed
// and never incremented).  An A/B built on such a run measures nothing, and its
// number goes into a table as if it did.
//
// WHAT THIS ADDS, AND WHAT IT DOES NOT CHANGE.
//
//   * Every top-level host-fallback seam now says so: one counter per
//     subsystem, and `[RASBERY][GPU_FULL]` at the end of the run, printed
//     whether or not the gate is on -- so "the arm was on and never engaged"
//     cannot look like "the arm was off" (the same G0 rule the XSRECON/XE/NODAL
//     receipts already exist for).
//   * With RASBERY_GPU_FULL=1 (alias RASBERY_GPU_STRICT=1) a seam THROWS
//     instead of running the host body.  main.cpp already catches per case, in
//     both the batch and the serial branch, so the violation fails ONE case and
//     the rest of the batch continues (plan Sec 6.3 item 4).
//   * DEFAULT OFF.  With the gate unset `note()` is one relaxed increment on a
//     path that was already about to run a whole CPU physics body; the
//     trajectory, the HDF5 and the [TRAJECTORY] digest are untouched.
//
// WHY RASBERY_GPU_FULL IS **NOT** IN Driver.h's `kArmEnv`.  That list is the
// knobs that can MOVE A TRAJECTORY, and this one cannot: it never selects a
// different numerical path.  It converts a fallback into a FAILURE, so a run
// that completes under the gate took exactly the path it would have taken
// without it.  Same reasoning the list already gives for RASBERY_GPU_PPR, and
// the opposite of the one it gives for RASBERY_GPU_CRAM.
//
// WHERE THE GUARDS ARE.  One per top-level seam -- the place the host body is
// actually entered, not the dozens of inner `return false`s that feed it,
// because every inner refusal surfaces at one of these:
//
//   src/BICGCMFD.cpp   BICGCMFD::drive        device sweep loop -> host BiCGSTAB
//   src/Nodal.cpp      Nodal::drive           TryDriveGpu       -> driveBody()
//   src/XSSet.cpp      UpdateFlatXS           FlatXS arm        -> reference loop
//   src/XSSet.cpp      UpdateEquilibriumXenon split/fused Xe    -> host Xe loop
//   src/XSSet.cpp      Deplete                CRAM predictor    -> DepleteNode loop
//   src/XSSet.cpp      PredictorCorrectorStep CRAM corrector    -> host corrector
//   src/Driver.h       ReconvergeFlux         outer segment     -> host outer
//   src/Driver.h       SolveLoop              outer segment     -> host outer
//   src/Driver.h       statepoint loop        GPU PPR           -> host reset+drive
//
// tools/test_gpu_full_fail_closed.py holds that list against the source.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>

namespace rasbery {
namespace gpufull {

/// The subsystems the plan's Sec 6.3 receipt names, in receipt order.
enum class Subsystem : int {
    Cmfd = 0,
    Outer,
    Nodal,
    FlatXs,
    Xe,
    Ppr,
    Cram,
    Count
};

inline const char* subsystemName(Subsystem s) {
    switch (s) {
        case Subsystem::Cmfd:   return "cmfd";
        case Subsystem::Outer:  return "outer";
        case Subsystem::Nodal:  return "nodal";
        case Subsystem::FlatXs: return "flatxs";
        case Subsystem::Xe:     return "xe";
        case Subsystem::Ppr:    return "ppr";
        case Subsystem::Cram:   return "cram";
        case Subsystem::Count:  break;
    }
    return "unknown";
}

/// A case-fatal contract violation.  It names the SITE, not just the
/// subsystem: "the nodal arm fell back" is a symptom; "Nodal::drive: the device
/// nodal arm declined" is where to look.
class Violation : public std::runtime_error {
public:
    Violation(Subsystem which, const char* where, const char* why)
        : std::runtime_error(std::string("[RASBERY][GPU_FULL][VIOLATION] subsystem=") +
                             subsystemName(which) + " site=" + (where != nullptr ? where : "?") +
                             " reason=" + (why != nullptr ? why : "?") +
                             " -- RASBERY_GPU_FULL forbids the CPU fallback; this case fails")
        , subsystem(which)
        , site(where != nullptr ? where : "?")
        , reason(why != nullptr ? why : "?") {}

    Subsystem   subsystem;
    std::string site;
    std::string reason;
};

/// The project's usual truthiness: set and not one of the off words.
inline bool contractTruthy(const char* value) {
    if (value == nullptr || *value == '\0') return false;
    const std::string text(value);
    return !(text == "0" || text == "off" || text == "OFF" ||
             text == "false" || text == "FALSE");
}

/// RASBERY_GPU_FULL -- DEFAULT OFF, read once.  RASBERY_GPU_STRICT is an alias:
/// the plan (Sec 6.3) names the gate RASBERY_GPU_FULL and the campaign has been
/// calling it "GPU strict", and a contract whose name depends on who is talking
/// is a contract nobody sets correctly.
inline bool required() {
    static const bool on = [] {
        return contractTruthy(std::getenv("RASBERY_GPU_FULL")) ||
               contractTruthy(std::getenv("RASBERY_GPU_STRICT"));
    }();
    return on;
}

/// Was this arm ASKED for?  A seam reached with the arm off is not a violation
/// -- nothing was promised -- so every guard is conditioned on the same
/// predicate the caller used to decide whether to try the device at all.
inline bool armRequested(const char* env_name) {
    return contractTruthy(std::getenv(env_name));
}

namespace detail {
inline std::atomic<unsigned long long>& counter(Subsystem which) {
    // Function-local static: one instance across every translation unit that
    // includes this header, including the .cu ones, with no .cpp to add to the
    // per-test add_executable lists in CMakeLists.txt.
    static std::array<std::atomic<unsigned long long>,
                      static_cast<std::size_t>(Subsystem::Count)>
        counters{};
    return counters[static_cast<std::size_t>(which)];
}
} // namespace detail

inline unsigned long long fallbacks(Subsystem which) {
    return detail::counter(which).load(std::memory_order_relaxed);
}

inline unsigned long long totalFallbacks() {
    unsigned long long sum = 0;
    for (int i = 0; i < static_cast<int>(Subsystem::Count); ++i)
        sum += fallbacks(static_cast<Subsystem>(i));
    return sum;
}

/// COUNT ONLY.  For a seam that cannot safely unwind -- today exactly one, the
/// CMFD enqueue hook that runs INSIDE a live device outer segment, where a
/// throw would leave the segment's stream and any in-flight capture in a state
/// nothing is written to clean up.  It is recorded, and the receipt's
/// `contract_pass` goes false, but the case still runs.
inline void count(Subsystem which) {
    detail::counter(which).fetch_add(1, std::memory_order_relaxed);
}

/// COUNT, THEN REFUSE IF THE GATE IS ON.  Call this at the moment the host body
/// is about to run because the device would not.
inline void note(Subsystem which, const char* where, const char* why) {
    count(which);
    if (required()) throw Violation(which, where, why);
}

/// The same, when the arm's own enable predicate decides whether a fallback was
/// even possible.
inline void noteIf(bool arm_requested, Subsystem which, const char* where, const char* why) {
    if (arm_requested) note(which, where, why);
}

/// True when nothing fell back.  Under the gate this is the only value a
/// completed run can report, which is the point: a run that reports
/// `contract_pass:false` with the gate OFF is telling you what the gate would
/// have caught.
inline bool contractPass() { return totalFallbacks() == 0; }

/// Plan Sec 6.3's receipt, minus the two counters that are not this header's to
/// own: `graph_fallbacks` is already reported by [RASBERY][CUDA][BACKEND_COUNTERS],
/// [RASBERY][NODAL][GPU] and [RASBERY][NODAL][BATCH], and
/// `mid_iteration_materializations` has no producer until WP5's XS ownership
/// work lands.  Duplicating either here would create a second number for one
/// fact.
inline void appendReceiptFields(std::ostream& out) {
    out << "\"gpu_full\":" << (required() ? "true" : "false");
    for (int i = 0; i < static_cast<int>(Subsystem::Count); ++i) {
        const auto which = static_cast<Subsystem>(i);
        out << ",\"" << subsystemName(which) << "_fallbacks\":" << fallbacks(which);
    }
    out << ",\"contract_pass\":" << (contractPass() ? "true" : "false");
}

} // namespace gpufull
} // namespace rasbery

/// The guard, as a macro so `tools/test_gpu_full_fail_closed.py` has one token
/// to scan every fallback seam for.  A new seam without it is a new silent
/// fallback, and that is the thing the test exists to refuse.
#define RASBERY_GPU_FULL_GUARD(which, where, why) \
    ::rasbery::gpufull::note(::rasbery::gpufull::Subsystem::which, (where), (why))

#define RASBERY_GPU_FULL_GUARD_IF(cond, which, where, why) \
    ::rasbery::gpufull::noteIf((cond), ::rasbery::gpufull::Subsystem::which, (where), (why))

/// Count-only variant; see gpufull::count.  Every use needs a written reason
/// why it cannot throw, and the contract test holds the list of allowed sites.
#define RASBERY_GPU_FULL_COUNT(which) \
    ::rasbery::gpufull::count(::rasbery::gpufull::Subsystem::which)
