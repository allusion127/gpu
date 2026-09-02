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
//   * the flat-XS CTA workspace (WP20) AND, since WP20.1, THE FOUR micx/lmpx
//     DEVICE BLOCKS THEMSELVES -- live and reference, 59.5 MB each at KNGR
//     size.  This is the measured BANDWIDTH carrier of the whole run: ~8.7 GB
//     of residual D2H on a single deck, ~2 GB per case in a batch, and a
//     reference re-upload on every branch rebuild.  Halving the element halves
//     every one of them.
//   * the nodal drive's own state (WP20.1): the nine updateConstant products
//     (3.9 GB of H2D over a KNGR run, the nodal per-outer carrier), chif, the
//     three macroscopic inputs as a narrowed per-drive COPY, and the twelve
//     private working arrays.
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
//   4. THE MACROSCOPIC CROSS-SECTIONS (WP20.1).  `xs` / `xs_ssm` / `iden` stay
//      double.  They are the answer the nodal drive, the CMFD operator, the Xe
//      commit and every host reader take as authoritative, they are 2 % of the
//      bytes the micx/lmpx blocks are, and narrowing them would put a
//      conversion on the ONE path in this tree that has a bit-golden gate.
//      The nodal drive takes a narrowed COPY of the three it reads instead.
//   5. THE CANONICAL STATE (WP20.1).  jnet / flux / phis stay double on the
//      nodal arm because in shared mode those pointers ARE the CMFD backend's
//      buffers and the host's Geometry arrays are the D2H destinations.
//      Narrowing them would turn a pointer swap into a type pun and would trip
//      HostPinRegistry's rule that one host base is pinned at one width.
//   6. CRAM's INPUT (WP20.1).  CRAM still consumes the four condensation slots
//      as double -- the partial-fraction sum cancels -- so the WP15.1 D2D
//      handover becomes a WIDENING KERNEL under the arm rather than a memcpy.
//      Half the DRAM read, none of the cancellation.
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

/// WP20.2 -- THE REFINEMENT ROUND CAP, and the one knob in this header whose
/// value is a NUMBER rather than a bit.
///
/// WHY IT EXISTS.  WP20 measured Gate A 0.017 pcm / Gate B 0.238 % PASS on the
/// FP32 arm and still lost time: outers went 4377 -> 4502 (+2.9 %).  That is the
/// classic mixed-precision symptom and it is not a bug in any kernel.  The FP32
/// inner BiCGSTAB's ATTAINABLE residual is ~1e-7 relative -- float's own eps --
/// and the outer Wielandt/convergence logic was written against an inner solve
/// that could go further, so the outer loop pays for the shortfall in extra
/// sweeps.  Halving the bytes and then buying the saving back in outers is a
/// wash, which is exactly what the +6.8 % wall said.
///
/// THE FIX IS THE TEXTBOOK ONE and it is already half-built: WP20's outer is
/// FP64 residual -> FP32 inner -> FP64 correction, i.e. ONE round of iterative
/// refinement.  WP20.2 makes the round count a LOOP: after each round the TRUE
/// FP64 residual r = b - A*x is recomputed from the FP64 operator and the FP64
/// flux (the arithmetic begin_outer_fused_f32 already performs), its FP64 norm
/// is taken, and it is tested against the SAME frozen reference r20 and the
/// SAME eps the FP64 path uses.  A slot that meets the test halts and costs
/// nothing further; a slot that does not gets another FP32 solve for the
/// correction.  The bandwidth-heavy sweeps stay FP32 and the ACCEPTANCE
/// CRITERION goes back to being the FP64 one, which is what the outer count is
/// a function of.
///
/// WHY THE VALUE IS A CAP AND NOT A `while`.  The inner loop is a CAPTURED
/// GRAPH.  Its depth is topology, so "repeat until converged" is spelled the
/// way this tree has always spelled it: capture a fixed number of rounds and
/// let the trailing ones find `halt` already raised and self-cancel on their
/// first instruction, exactly as the captured iterations past `1 + nmax` do.
///
///   unset (and the arm on)  2 -- the default, and the one WP20.2 measures
///   "1" / "on" / "true"     2 -- the same, spelled as a bit
///   an integer >= 2         that many rounds, clamped to kRefineRoundsMax
///   "0" / "off" / "false"   1 -- ONE round, which IS the WP20 topology, node
///                           for node.  This is how the arm is turned off, and
///                           it is why the OFF answer is 1 and never 0.
///
/// With RASBERY_GPU_FP32 unset this returns 1 whatever the variable says: a
/// refinement round is a round of an FP32 solve, and there is no FP32 solve.
constexpr int kRefineRoundsMax     = 8;
constexpr int kRefineRoundsDefault = 2;

inline int refineRounds() {
    static const int rounds = [] {
        const char* value = std::getenv("RASBERY_GPU_FP32_REFINE");
        if (value == nullptr) return kRefineRoundsDefault;
        const std::string s(value);
        if (s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
            s == "FALSE")
            return 1;
        const int n = std::atoi(value);
        if (n >= 2) return n < kRefineRoundsMax ? n : kRefineRoundsMax;
        return kRefineRoundsDefault;
    }();
    return armed() ? rounds : 1;
}

/// Is the refinement LOOP engaged, as opposed to WP20's single round?
inline bool refine() { return refineRounds() > 1; }

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
///           (919 elements: 7,352 -> 3,676 B/CTA) AND, since WP20.1, the four
///           micx/lmpx device blocks -- live and reference -- with the host
///           materialisation widening at ONE site (XSSet::EnsureMicxHost ->
///           XsReconBackend::downloadFlatXsMicx -> Impl::drainMicxWiden).
///           src/FlatXsCtaKernel.cuh, src/FlatXsKernel.h,
///           src/CudaXsReconBackend.cu.
///   nodal   TRUE since WP20.1: NodalViewT<ValueT> narrows the nine
///           updateConstant products, chif, a per-drive copy of the three
///           macroscopic inputs, and the twelve working arrays; PRECISION IS A
///           FIELD OF NodalGraphKey so the two graphs cannot alias.  TWO ARMS
///           STILL DEMOTE and are counted: the hybrid drive
///           (RASBERY_GPU_NODAL_FULL unset), which round-trips trlcff/matM to
///           FP64 host arrays mid-drive, and the multi-deck batch arena, whose
///           per-slot view table and bucket graphs are a second narrowing.
///   xe      DEFERRED, and the reason is a MEASUREMENT contract rather than a
///           numeric one.  Bits 0..4 of the shipped form mask
///           (XE_DOT_FIRST/XE_DOT_THIRD/XE_CAND1/XE_CAND2) are mined ON THE
///           HOST, in FP64, by comparing xe::xeDotChunk / xeCandidateOrdinal
///           against src/XeAndersonReference.cpp BIT FOR BIT
///           (src/XeFormMine.h).  Narrowing XeTripleConst changes the signature
///           of the body being mined, so the miner would go on reporting
///           `sound = true` for a computation the device no longer performs --
///           a silent false negative of exactly the class src/XeFormAudit.cpp
///           exists to catch.  Two more would have to be re-argued rather than
///           cast: xsrFma/xsrMul have no float overload, and the atomicMax
///           monotonicity trick on __double_as_longlong
///           (CudaXsReconBackend.cu) would become __float_as_int.  Xe's XS
///           INPUTS are nevertheless narrowed by the flat-XS item above -- the
///           Xe kernels read the same BatchView micx/lmpx blocks -- and that
///           saving is attributed to `flatxs`, where the conversion happens.
///   cram    DEFERRED **AND FLAGGED** (RASBERY_GPU_FP32_CRAM).  See the header
///           note: the partial-fraction sum cancels catastrophically.
///   ppr     DEFERRED.  Strictly downstream of the iteration and therefore
///           worth nothing to the trajectory; it is a VRAM item, not a
///           throughput one.
inline bool converted(Backend which) {
    switch (which) {
        case Backend::Cmfd:   return true;
        case Backend::FlatXs: return true;
        case Backend::Nodal:  return true;
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
/// WP20.1: THERE ARE NOW TWO RECOVERY SHAPES BEHIND ONE LATCH, and the
/// difference is not a policy choice, it is what each backend's precision IS.
///
///   cmfd   a KERNEL SET selected per solve.  The latch demotes the next solve
///          to the wide kernels in place; nothing is reallocated.
///   nodal  an ALLOCATION and a LAYOUT: `ndev_flt` holds the constants, chif
///          and the whole working set, and `ndev_dbl` was laid out WITHOUT
///          them.  There is no wide device arm to fall back to without
///          re-laying-out and re-uploading mid-run, from inside a drive whose
///          kernels are captured.  So the valve REFUSES the device nodal arm
///          for the rest of the process and hands the drive back to
///          Nodal::TryDriveGpu's CPU body -- which is the reference this arm
///          is scored against.  Slower, correct, and loud.
///   flatxs the same allocation-shaped precision, and it therefore has NO
///          latch at all: the block width is a stand-up decision and every
///          reader in the tree -- CRAM, xsrecon, Xe, the host materialisation
///          -- is bound to it.  Declared in
///          CudaXsReconBackend.cu::flatxsNarrowBlocks() rather than discovered.
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
    out << ",\"refine\":" << refineRounds();
    out << ",\"demotions\":" << demotions();
    out << ",\"nonfinite_fallbacks\":" << fallbackTotal();
    out << ",\"bytes_saved_est\":" << bytesSavedEst();
}

} // namespace fp32
} // namespace rasbery
