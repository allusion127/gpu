#pragma once

// WP20: RASBERY_GPU_FP32 -- THE DEVICE-WIDE SINGLE-PRECISION ARM.
//
// WHAT WAS ACTUALLY IN THE TREE BEFORE THIS FILE, because the campaign started
// from a wrong belief and the correction belongs at the top of the header that
// fixes it.  The GPU path was **all FP64**, not mixed: every device buffer, every
// kernel and every reduction in CMFD, nodal, flat-XS, Xe, CRAM and PPR carried
// `double`.  The single exception was RASBERY_GPU_CMFD_FP32 (src/CudaBICGBackend.cu
// Sec "Mixed-precision inner iteration"), an experiment that narrowed the CMFD
// INNER BiCGSTAB only, defaulted OFF, and measured **+2.6 %** on M64
// (docs/CAMPAIGN_ANDERSON_WIDTH_FP32_20260827_KO.md Sec 5).  So "the GPU is
// already mixed precision" was false, and "FP32 will be 64x faster because the
// card is 1:64 on FP64" was false too.
//
// WHY IT IS STILL WORTH DOING, stated as the mechanism rather than the hope.
// The RTX PRO 6000 Blackwell throttles FP64 to 1/64 of FP32, but every kernel in
// this solver is a stencil or a BLAS-1 sweep at ~0.13 FLOP/byte.  At that
// intensity the ALU ratio is irrelevant -- the kernels never get near the FP64
// issue limit -- and the ONLY thing single precision buys is **halved bytes**:
// half the DRAM traffic per element, half the L2 footprint (so a working set
// that missed may now hit), half the H2D/D2H when the transfer itself is
// narrowed, and half the VRAM for the resident blocks.  The ceiling is therefore
// ~2x on the memory-bound fraction of the time and 1.0x on everything else, and
// the honest expectation sits well under 2x.  Anybody quoting 64x from this
// header has read the FP64:FP32 ratio and not this paragraph.
//
// GATE CLASS: **A2**.  This arm MOVES THE TRAJECTORY on purpose.  FP32 carries
// ~7.2 decimal digits against FP64's ~15.9, and k_eff is judged at pcm (1e-5)
// precision, which is at the edge of what float resolves for a quantity near 1.0
// (float eps ~ 1.19e-7, so ~12 pcm per ULP of a bare float k_eff -- which is why
// the eigenvalue and every convergence scalar stay FP64 below).  It is therefore
// NOT validated by the bit-golden gate.  It is validated by:
//
//   Gate A   per-step against the same deck's FP64 trajectory (v6-FP64), which
//            answers "how far did the arm move it".
//   Gate B   against MASTER, which answers "is it still right".  The standing
//            budget is 1.905 pcm / 15.309 ppm / AO 0.013 / BOC pin 0.238 % RMS,
//            0.80 % max (tools/compare_master_rasbery.py, tools/gate_b_pin_rms.py).
//
// The acceptance decision is thus MEASURABLE rather than argued, and that is the
// whole reason this arm is a flag and not an edit.
//
// ---------------------------------------------------------------------------
// WHAT IS FLOAT AND WHAT DELIBERATELY IS NOT
// ---------------------------------------------------------------------------
//
// FLOAT (device hot state, behind the flag):
//   * the CMFD/BiCGSTAB working set -- the operator as consumed by the inner
//     solve (diag/cc float mirrors, the inverted diagonal blocks dinv), the
//     Krylov vectors r/r0/p/v/s/t/y/z, the colour Gauss-Seidel sweeps and the
//     A*y / A*z applications.
//
// DOUBLE, AND WHY -- this is the DECLARED DEVIATION from "everything FP32":
//
//   1. THE RESIDUAL-NORM ACCUMULATION.  The FP32 dot products load FLOAT
//      operands and fold them into a DOUBLE accumulator.  float x float widened
//      to double is EXACT, so stage 1 sums exact products and the reduction adds
//      no rounding beyond the operands' own.  This is not a hedge, it is
//      standard mixed-solver practice (it is what every mixed-precision Krylov
//      library does, and what LAPACK's iterative refinement has done since the
//      1970s): BiCGSTAB's breakdown modes are all scalar -- rho going to zero,
//      omega going to zero, a sign flip in alpha -- and a float accumulator over
//      ~8,451 nodes x 2 groups loses enough digits to manufacture those
//      breakdowns out of nothing.  The dots are memory bound at these sizes, so
//      the wide accumulator is FREE in time and buys back exactly the accuracy
//      the method is most sensitive to.
//   2. THE CONVERGENCE DECISION SCALARS.  rho, alpha, omega, the norms, the
//      relative test against the frozen reference r20, the Wielandt shift and
//      the eigenvalue stay FP64, so the OUTER ITERATION COUNTS stay comparable
//      between the FP64 and FP32 arms.  If they did not, Gate A would be
//      comparing two different amounts of work and could not attribute the
//      difference to precision at all.
//   3. THE STORED OPERATOR AND THE FLUX.  diag/cc/udiag/phi/src/psi remain the
//      FP64 authority; the arm narrows what the inner loop CONSUMES.  One outer
//      is FP64 residual -> FP32 inner solve for the correction -> FP64
//      correction accumulation, i.e. iterative refinement, which is what makes
//      the FP32 error a correction error rather than a solution error.
//
// RASBERY_GPU_FP32_STRICT=1 is the pure-FP32 reduction arm: it narrows (1) and
// (2) as well, and exists so the claim "the double accumulator is what keeps the
// outer counts comparable" is TESTABLE rather than asserted.  It is expected to
// be worse and it is not a production arm.
//
// RASBERY_GPU_FP32_CRAM=1 extends the arm to the CRAM depletion solve.  Held
// back from the main flag because CRAM evaluates matrix exponentials over
// nuclide fields spanning ~20 decades: the partial-fraction terms alternate in
// sign and cancel catastrophically, and float has no headroom for that
// cancellation.  Default: CRAM stays FP64 even with RASBERY_GPU_FP32=1, and the
// refusal is COUNTED as a demotion so the receipt says the arm was asked and
// declined rather than silently doing nothing.
//
// ---------------------------------------------------------------------------
// FEATURE-OFF BYTE IDENTITY
// ---------------------------------------------------------------------------
//
// With RASBERY_GPU_FP32 unset, `armed()` is false, `routes()` is false for every
// backend, and no float kernel is REACHABLE.  The FP64 enqueue paths are
// textually untouched, so the trajectory digest stays 1f36e75dc00ed2b4 / 4377
// outers.  The receipt line below prints unconditionally -- the same G0 rule the
// [RASBERY][GPU_FULL] receipt exists for: "the arm was on and never engaged"
// must not be able to look like "the arm was off".  The digest folds statepoints,
// outers and the bit patterns of efpd / k_eff / boron; it does not fold stdout,
// so an extra receipt line cannot move it.
//
// ---------------------------------------------------------------------------
// WHY THE THREE KNOBS **ARE** IN trajectory::kArmEnv
// ---------------------------------------------------------------------------
//
// The opposite of the argument RASBERY_GPU_PPR and RASBERY_GPU_XFER_ELIDE get.
// Those cannot move a trajectory; these three select the ROUNDING of the whole
// device iteration, which is the most trajectory-moving thing a knob in this
// binary can do.  Listing them is also what folds them into the WP10.1 case key,
// so an FP64 answer can never be served to an FP32 request.
//
// tools/test_gpu_fp32_contract.py holds every claim above against the source.

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>

namespace rasbery {
namespace fp32 {

/// The backends the receipt names, in receipt order.  One entry per device
/// subsystem that owns hot floating-point state, so the receipt can say "asked
/// and declined" for the ones this commit did not convert instead of leaving
/// them unmentioned.
enum class Backend : int {
    Cmfd = 0,
    Nodal,
    FlatXs,
    Xe,
    Cram,
    Ppr,
    Count
};

inline const char* backendName(Backend which) {
    switch (which) {
        case Backend::Cmfd:   return "cmfd";
        case Backend::Nodal:  return "nodal";
        case Backend::FlatXs: return "flatxs";
        case Backend::Xe:     return "xe";
        case Backend::Cram:   return "cram";
        case Backend::Ppr:    return "ppr";
        case Backend::Count:  break;
    }
    return "?";
}

/// HOW THE TWO ARMS ARE PARAMETERISED, since a reader will look here for a
/// `Real` typedef and not find one.
///
/// The precision is spelled as a TEMPLATE PARAMETER at the kernel that owns the
/// state, not as an alias in this header.  `flatxsSolveNodeCta<T, POL, WS>`
/// (src/FlatXsCtaKernel.cuh) takes the WORKSPACE TYPE, so the FP32 and FP64 arms
/// are two instantiations of one body: they cannot drift apart under
/// maintenance, and every structural property the contract test checks is
/// checked once and holds for both.  A bare `Real<narrow>` alias would not have
/// bought that -- the struct, not the scalar, is what the kernel names.
///
/// Where a kernel is precision MIXED at its boundary (reads a double operator,
/// writes a float working vector) even that does not work: one template would
/// need an `if constexpr` at exactly the site that matters and would still have
/// to be launched from a branch, because the pointer types differ.  Those
/// kernels stay duplicated -- see the "WHY A PARALLEL KERNEL SET" note in
/// src/CudaBICGBackend.cu, which is the reason CMFD has an `_f32` kernel set
/// rather than a templated one.

/// Opt-IN, and the same spelling every other arm in this tree uses: unset means
/// off, and "0"/"off"/"false" mean off so a launcher can pin the OFF arm
/// explicitly (which the case key needs -- an unset knob and an explicit "0" are
/// two different payloads, deliberately).
inline bool envFlagOn(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string s(value);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
             s == "FALSE");
}

/// THE ARM.  Read ONCE and cached, for the same reason every capture-relevant
/// gate in this tree is: the choice fixes the captured graph TOPOLOGY (the FP32
/// and FP64 kernel sets are different nodes), so it must not be able to change
/// between two outers of the same run.
inline bool armed() {
    static const bool on = envFlagOn("RASBERY_GPU_FP32");
    return on;
}

/// The pure-FP32 reduction arm.  Implies nothing on its own: it only narrows the
/// accumulators of an arm that is already on.
inline bool strict() {
    static const bool on = envFlagOn("RASBERY_GPU_FP32_STRICT");
    return on;
}

/// Extend the arm to CRAM depletion.  See the header note: default OFF even
/// under RASBERY_GPU_FP32, because the partial-fraction sum cancels.
inline bool cramExtended() {
    static const bool on = envFlagOn("RASBERY_GPU_FP32_CRAM");
    return on;
}

namespace detail {

/// Per-backend sticky latch, set by the non-finite fallback.  Process-wide,
/// relaxed, and never cleared -- the same scope and reasoning as
/// rasbery::xfer::Ledger and rasbery::xe::XeGpuTally.  A latch is per BACKEND
/// and not per case because a captured graph serves every slot of a launch: a
/// per-case precision would mean carrying both kernel sets in one graph and
/// masking one, which doubles the node count this campaign exists to reduce.
struct Tally {
    std::atomic<bool>               latched[static_cast<int>(Backend::Count)];
    std::atomic<unsigned long long> fallbacks[static_cast<int>(Backend::Count)];
    std::atomic<unsigned long long> demotions[static_cast<int>(Backend::Count)];
    std::atomic<unsigned long long> bytes_saved;

    Tally() : bytes_saved(0) {
        for (int i = 0; i < static_cast<int>(Backend::Count); ++i) {
            latched[i].store(false, std::memory_order_relaxed);
            fallbacks[i].store(0, std::memory_order_relaxed);
            demotions[i].store(0, std::memory_order_relaxed);
        }
    }
};

inline Tally& tally() {
    static Tally t;
    return t;
}

} // namespace detail

/// WHICH BACKENDS THIS TREE ACTUALLY NARROWS, as a table rather than as a
/// sentence in a doc.
///
/// WP20 landed the two BANDWIDTH CARRIERS and stopped there on purpose; the
/// rest are `false` here, and being `false` here is what makes the receipt say
/// `"deferred"` instead of `"fp32"` for them.  A receipt that claimed an arm it
/// does not have is worse than no receipt: every A/B built on it would be
/// attributing a wall-clock difference to a conversion that never happened.
///
///   cmfd    TRUE.  The whole inner BiCGSTAB -- operator as consumed, the
///           inverted diagonal blocks, all eight Krylov vectors, the colour
///           Gauss-Seidel sweeps and both matvecs.  src/CudaBICGBackend.cu.
///   flatxs  TRUE.  The per-node workspace the CTA kernel holds in __shared__
///           (919 elements: 7,352 -> 3,676 B/CTA).  src/FlatXsCtaKernel.cuh.
///   nodal   DEFERRED.  NodalView is ~25 double* fields consumed by a kernel
///           set that is CAPTURED INTO A GRAPH and cached under a key that does
///           not carry a precision; a float arm needs a parallel view, a
///           parallel device block and a precision in that key.  See
///           docs/WP20_GPU_FP32_20260831_KO.md Sec 7.
///   xe      DEFERRED.  Same shape, plus the Anderson normal equations, whose
///           conditioning is exactly what RASBERY_XE_HOST_FORMS already sweeps
///           -- narrowing them without re-running that sweep would confound two
///           variables.
///   cram    DEFERRED **AND FLAGGED** (RASBERY_GPU_FP32_CRAM).  See the header
///           note: the partial-fraction sum cancels catastrophically.
///   ppr     DEFERRED.  Strictly downstream of the iteration and therefore
///           worth nothing to the trajectory; it is a VRAM item, not a
///           throughput one.
inline bool converted(Backend which) {
    switch (which) {
        case Backend::Cmfd:   return true;
        case Backend::FlatXs: return true;
        case Backend::Nodal:  return false;
        case Backend::Xe:     return false;
        case Backend::Cram:   return false;
        case Backend::Ppr:    return false;
        case Backend::Count:  break;
    }
    return false;
}

/// Is this backend IN SCOPE for the arm at all?
///
/// Separate from `routes()` so the receipt can distinguish the states a reader
/// actually cares about: not asked (arm off), asked and deferred (no narrow
/// path in this tree), asked and declined (CRAM without its extension, or a
/// latch), asked and taken.
inline bool inScope(Backend which) {
    if (!converted(which)) return false;
    if (which == Backend::Cram) return cramExtended();
    return true;
}

inline bool latched(Backend which) {
    return detail::tally().latched[static_cast<int>(which)].load(
        std::memory_order_relaxed);
}

/// THE SINGLE ROUTING PREDICATE.  Every FP32 launch site in the tree asks this
/// and nothing else, so "which arm did this kernel run under" has one answer.
inline bool routes(Backend which) {
    return armed() && inScope(which) && !latched(which);
}

/// A site that was asked for FP32 and stayed FP64: a shape the narrow kernel
/// does not serve, a backend out of scope, a path that has not been converted
/// yet.  Counted rather than silent, because an arm that quietly does nothing is
/// the failure mode this whole receipt exists to make impossible.
inline void noteDemotion(Backend which) {
    detail::tally().demotions[static_cast<int>(which)].fetch_add(
        1, std::memory_order_relaxed);
}

/// Estimated bytes NOT moved because an element was four wide instead of eight.
/// Accumulated at the conversion sites, in the same units the [RASBERY][XFER]
/// ledger counts, so the two receipts can be read against each other.
inline void noteBytesSaved(std::size_t bytes) {
    detail::tally().bytes_saved.fetch_add(static_cast<unsigned long long>(bytes),
                                          std::memory_order_relaxed);
}

/// THE NON-FINITE FALLBACK.  Loud, counted, once per backend, ENV-INDEPENDENT.
///
/// Reuses the pattern src/CudaBICGBackend.cu's latchFp32Off() established: the
/// narrow kernels REFUSE to write a non-finite result, so the case comes back
/// holding the iterate it entered with, the failed attempt is DISCARDED rather
/// than accepted, and the backend moves to FP64 for the rest of the process.
/// The caller is responsible for dropping any cached graph captured under the
/// old precision -- this function cannot do it, because it does not own one.
///
/// Returns true the FIRST time it fires for a backend, so a caller can do its
/// own one-shot work (graph invalidation) without a second flag.
inline bool latchOff(Backend which, const char* reason) {
    const int idx = static_cast<int>(which);
    detail::tally().fallbacks[idx].fetch_add(1, std::memory_order_relaxed);
    bool expected = false;
    if (!detail::tally().latched[idx].compare_exchange_strong(
            expected, true, std::memory_order_relaxed)) {
        return false;
    }
    std::cerr << "[RASBERY][FP32][FALLBACK] {\"backend\":\"" << backendName(which)
              << "\",\"reason\":\"" << (reason != nullptr ? reason : "nonfinite")
              << "\",\"precision\":\"fp64\"}" << std::endl;
    return true;
}

/// Summed over the backends, because that is the shape the receipt states.  The
/// per-backend split is kept anyway and costs nothing: `backendState()` already
/// names WHICH backend stayed wide, so a reader who wants the attribution has
/// it, and a future receipt can print the split without touching a call site.
inline unsigned long long fallbackTotal() {
    unsigned long long total = 0;
    for (int i = 0; i < static_cast<int>(Backend::Count); ++i)
        total += detail::tally().fallbacks[i].load(std::memory_order_relaxed);
    return total;
}

inline unsigned long long demotions() {
    unsigned long long total = 0;
    for (int i = 0; i < static_cast<int>(Backend::Count); ++i)
        total += detail::tally().demotions[i].load(std::memory_order_relaxed);
    return total;
}

inline unsigned long long bytesSavedEst() {
    return detail::tally().bytes_saved.load(std::memory_order_relaxed);
}

/// What a backend RESOLVED to, as one word, for the receipt.
///
///   "fp64"     the arm is off; nothing was asked of anybody
///   "deferred" the arm is on but this tree has no narrow path for the backend
///   "declined" the arm is on and the backend has one, but it is not enabled
///              (CRAM without RASBERY_GPU_FP32_CRAM)
///   "latched"  the arm was on and taken, and a non-finite pushed it to FP64
///   "fp32"     the arm is on, in scope, and nothing latched it off
inline const char* backendState(Backend which) {
    if (!armed()) return "fp64";
    if (!converted(which)) return "deferred";
    if (latched(which)) return "latched";
    if (!inScope(which)) return "declined";
    return "fp32";
}

/// `[RASBERY][FP32] {...}` -- printed unconditionally from every branch of
/// main.cpp, next to the other end-of-run receipts.
inline void appendReceiptFields(std::ostream& out) {
    out << "\"arm\":\"" << (armed() ? "fp32" : "fp64") << "\"";
    out << ",\"strict\":" << (armed() && strict() ? "true" : "false");
    out << ",\"backends\":{";
    for (int i = 0; i < static_cast<int>(Backend::Count); ++i) {
        const auto which = static_cast<Backend>(i);
        if (i > 0) out << ",";
        out << "\"" << backendName(which) << "\":\"" << backendState(which) << "\"";
    }
    out << "}";
    out << ",\"demotions\":" << demotions();
    out << ",\"nonfinite_fallbacks\":" << fallbackTotal();
    out << ",\"bytes_saved_est\":" << bytesSavedEst();
}

} // namespace fp32
} // namespace rasbery
