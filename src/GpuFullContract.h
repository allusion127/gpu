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
#include <cstring>
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
    // WP22.  The T/H and critical-search device arms.  They are subsystems on
    // the list's own terms: both write state a later phase of the SAME
    // statepoint reads -- T/H writes tful/tmod/dmod, which UpdateFlatXS
    // reconstructs every cross section from, and the search writes the boron
    // the same reconstruction applies -- so an arm that refuses every call and
    // an arm that was never set produce the same numbers and the same log.
    // That is exactly the condition the GPU_FULL counters exist to tell apart.
    Th,
    Search,
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
        case Subsystem::Th:     return "th";
        case Subsystem::Search: return "search";
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

// ===========================================================================
// WP1 FOLLOW-UP: THE OUTER SEAM'S ALLOWANCE LIST (plan Sec 6.3 item 5)
// ===========================================================================
//
// THE GAP THIS CLOSES.  On host 181 at 8919331, arm X + RASBERY_GPU_FULL=1
// printed `contract_pass:false` with `outer_fallbacks:71` and STILL EXITED 0.
// Two separate defects sat behind that one line:
//
//   (a) the outer seam inside the segment could only COUNT (see gpufull::count
//       below), so 71 host outer bodies ran and nothing failed; and
//   (b) `contract_pass:false` was a receipt field nobody consumed, so even the
//       throwing seams could have been silent had they all been count-only.
//
// (b) is fixed by enforceExitCode() at the bottom of this header.  (a) is fixed
// by handing the seam the REASON and deciding per reason -- which is what this
// list is, and what the plan means by "허용된 host boundary materialization은
// 목록으로 관리한다" (Sec 6.3 item 5).
//
// THE RULE FOR BEING ON THIS LIST.  Not "the fallback is common", not "the
// fallback is cheap".  A reason belongs here only when the host body is the
// DESIGNED path for that region -- there is no device implementation that
// declined, because none was ever written, and none can be asked for.  Every
// other reason is CPU numerics standing in for a GPU arm that was requested,
// which is exactly plan Sec 6.3 item 3.
//
// ---------------------------------------------------------------------------
// ADMITTED
// ---------------------------------------------------------------------------
//
//   wielandt_warmup -- BICGCMFD::EnqueueRefusal::WielandtWarmup.
//
//     WHAT IT IS.  The first WIELANDT_WARMUP_SWEEPS CMFD sweeps after every
//     resetIteration() (Driver.h:3439, once per statepoint) run the Rayleigh
//     quotient rather than the Wielandt extrapolation.  BICGCMFD::drive says
//     why the device does not serve them: "the warm-up and its Rayleigh
//     schedule stay on the host, so the device never needs the icy < 0 branch"
//     (BICGCMFD.cpp).  There is no device warm-up to decline.
//
//     THE DECIDING EVIDENCE is that the CMFD seam ALREADY excludes exactly this
//     region, and has since WP1(b) landed: the `RASBERY_GPU_FULL_GUARD(Cmfd,
//     ...)` in BICGCMFD::drive sits INSIDE `if (!cap && canEnqueueDrive())`, so
//     a warm-up drive never reaches it.  That is why the same 181 run reported
//     `cmfd_fallbacks:0` beside `outer_fallbacks:71`.  Two seams disagreeing
//     about one region is not a contract; the outer seam is brought into line
//     with the CMFD seam rather than the reverse, because the reverse would
//     make every deck fail the gate structurally -- 2 warm-up drives x 35
//     statepoints = the 70 of the 71 observed -- and a gate no run can pass
//     tells you nothing about any run.
//
// ---------------------------------------------------------------------------
// CONSIDERED AND REFUSED -- these fail the case, and the contract test pins it
// ---------------------------------------------------------------------------
//
//   batch_mode (OuterSegmentRefusal::BatchMode) -- REFUSED.  It reads like "the
//     batch rendezvous arm is the designed GPU path", and that WAS true when
//     the predicate was `batch_width > 1`.  Task 18-lite changed it: the arena
//     is stood up at the RUN's width and the predicate is now
//     `batch_width > arena_slots` (CudaOuterGraph.h, outerSegmentRefusal) --
//     i.e. a batch that does NOT fit the arena that was stood up, which a VRAM
//     admission shrank or which ran past kMaxDeviceSlots.  A batch that fits is
//     never refused for this reason at all.  So the modern meaning is "no seat
//     on the device for this deck", the pre-arm gate skips the arm, and
//     SolveLoop runs its host outer body for the whole solve.  That is the
//     purest case of item 3 in the plan, not an exemption from it.
//
//   sweep_arm_off / no_cuda_solver / not_two_group -- REFUSED.  Each one is an
//     arm that was ASKED for (the outer segment only reaches this seam with
//     RASBERY_GPU_OUTER on) and is not there.  Plan Sec 6.3 item 1 is exactly
//     "enabled 이면서 engaged"; a silent CPU solve is the failure it names.
//
//   stage_prep_failed -- REFUSED.  The gate said yes and the staging refused:
//     a CUDA-side decline, which is item 2 verbatim.
//
//   The hostfree (OuterHostFreeRefusal) and graph (OuterGraphRefusal) ladders
//   are NOT on this list and are not violations either, because they are not
//   fallbacks: they choose between two DEVICE arms (per-outer vs host-free, and
//   stream vs captured WHILE).  Where a hostfree refusal does lead to host
//   numerics -- `sweep_wont_enqueue` -- the host body is entered through the
//   enqueue seam below, which asks BICGCMFD for the finer reason and decides
//   there.  tools/test_gpu_full_fail_closed.py holds this split.

struct AllowedRefusal {
    const char* reason;    ///< the enum's own name(), so the two cannot drift
    const char* rationale; ///< why the host body is the DESIGNED path here
};

inline constexpr AllowedRefusal kGpuFullAllowedOuterRefusals[] = {
    {"wielandt_warmup",
     "the Rayleigh warm-up has no device implementation to decline -- the CMFD "
     "seam already excludes the same region (BICGCMFD::drive guards inside "
     "canEnqueueDrive()), and it recurs once per statepoint by construction"},
};

inline constexpr std::size_t kGpuFullAllowedOuterRefusalCount =
    sizeof(kGpuFullAllowedOuterRefusals) / sizeof(kGpuFullAllowedOuterRefusals[0]);

/// Index into the allowance list, or -1.  Matched on the REASON STRING because
/// that is what the receipt prints and what the enum's name() produces, so a
/// renamed enumerator shows up as a violation rather than as a silent pass.
inline int allowedOuterRefusalIndex(const char* why) {
    if (why == nullptr) return -1;
    for (std::size_t i = 0; i < kGpuFullAllowedOuterRefusalCount; ++i)
        if (std::strcmp(why, kGpuFullAllowedOuterRefusals[i].reason) == 0)
            return static_cast<int>(i);
    return -1;
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

/// One counter per ALLOWANCE entry.  Kept apart from the fallback counters on
/// purpose: an allowed refusal must not move `contract_pass`, and it must not
/// be invisible either -- an allowance that fires ten thousand times is a
/// design question, and a receipt that folded it into `outer_fallbacks` could
/// not raise it.
inline std::atomic<unsigned long long>& allowedCounter(std::size_t index) {
    static std::array<std::atomic<unsigned long long>,
                      kGpuFullAllowedOuterRefusalCount>
        counters{};
    return counters[index];
}

// ===========================================================================
// WP10.7: WHICH SEAM WAS FIRST, AND WHERE EACH SEAM WAS
// ===========================================================================
//
// THE DEFECT THIS CLOSES, in the receipt that showed it.  The 238 GPU1 20-gen
// soak at 0054838 (arm A, no ARENA_PERSIST) printed
//
//   {"outer_fallbacks":9,"flatxs_fallbacks":4,"contract_pass":false,
//    "first_violation":"subsystem=outer site=Driver: outer segment pre-arm
//                       reason=no_residency"}
//
// and NOTHING in that line can be checked.  Two separate things were wrong:
//
//   (a) `first_violation` was whichever thread won a CAS on ONE process-wide
//       string, and the CAS is taken where the TEXT is built -- not where the
//       fallback happened.  Sixteen lanes fall back inside microseconds of each
//       other; the thread that got descheduled between count() and the string
//       loses the latch even though its event was first.  So the field names
//       "a violation", never provably "the first violation", and a reader who
//       treats it as the first -- which is what it is called -- chases the
//       wrong subsystem.  Here the four flatxs deaths and the nine outer
//       refusals both start at generation 6, and the receipt gave no way to
//       tell which of the two came first in that generation.
//
//   (b) the counters say a subsystem fired N times and say NOTHING about
//       where.  `flatxs_fallbacks:4` sent the reader to the raw 16k-line log to
//       recover a site string the process already had in its hand.
//
// THE FIX IS ONE ORDINAL, TAKEN WHERE THE EVENT IS.  count() -- the raw
// increment every seam ends at -- stamps a monotonic ordinal, and the
// per-subsystem first-site record keeps it.  `first_violation` is then the
// record with the SMALLEST ordinal across subsystems, computed at receipt time
// (post-join, single-threaded, exactly as this header already required of
// every first-violation reader).  A total order over the events, not over the
// string builders.

/// A monotonic ordinal per fallback EVENT, 1-based so 0 can mean "none".
inline unsigned long long nextOrdinal() {
    static std::atomic<unsigned long long> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) + 1;
}

/// The ordinal of the fallback THIS thread is in the middle of reporting.
/// Written by count(), read by the naming step a few instructions later, so
/// the ordinal a site is filed under is the one its event took -- not the one
/// the string builder happened to reach.
inline unsigned long long& pendingOrdinal() {
    static thread_local unsigned long long ordinal = 0;
    return ordinal;
}

/// Consume this thread's stamp.  CONSUMING IS THE POINT: a stamp left behind by
/// an earlier event on the same thread would file a LATER site under an EARLIER
/// ordinal, which is the one way this table could lie about order.  A caller
/// that names without counting gets a fresh ordinal instead.
inline unsigned long long takePendingOrdinal() {
    unsigned long long& slot = pendingOrdinal();
    const unsigned long long ordinal = slot != 0 ? slot : nextOrdinal();
    slot = 0;
    return ordinal;
}

/// The first site each subsystem reported, and when.  One slot per subsystem;
/// written at most once each (the CAS), read post-join.
struct SubsystemFirst {
    std::atomic<bool>               armed{false};
    std::atomic<bool>               ready{false};
    std::atomic<unsigned long long> ordinal{0};
    std::string                     site;
    std::string                     reason;
};

inline SubsystemFirst& subsystemFirst(Subsystem which) {
    static std::array<SubsystemFirst, static_cast<std::size_t>(Subsystem::Count)> slots;
    return slots[static_cast<std::size_t>(which)];
}

/// File this subsystem's FIRST site, under the ordinal its event took.
inline void recordSubsystemFirst(Subsystem which, const char* where, const char* why,
                                 unsigned long long ordinal) {
    SubsystemFirst& slot     = subsystemFirst(which);
    bool            expected = false;
    if (!slot.armed.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return;
    slot.ordinal.store(ordinal, std::memory_order_relaxed);
    slot.site   = (where != nullptr ? where : "?");
    slot.reason = (why != nullptr ? why : "?");
    slot.ready.store(true, std::memory_order_release);
}

/// The first violation of the RUN, for the run-level exit code.  Written at
/// most once (the CAS), and READ ONLY AFTER EVERY DRIVER HAS JOINED -- main.cpp
/// prints it beside the receipt, past both the batch parallel region and the
/// evaluator's server.run().  A reader that raced the writer could see a
/// half-assigned string; nothing reads it before the join.
///
/// KEPT BESIDE THE ORDERED TABLE ABOVE, not replaced by it: this latch is the
/// cheap guard that keeps the steady path from building a string per fallback,
/// and it is the fallback answer when a caller named a violation without going
/// through a subsystem slot.  `firstViolation()` prefers the ordered table.
inline std::atomic<bool>& firstViolationArmed() {
    static std::atomic<bool> armed{false};
    return armed;
}
inline std::string& firstViolationText() {
    static std::string text;
    return text;
}
inline void recordFirstViolation(std::string text) {
    bool expected = false;
    if (firstViolationArmed().compare_exchange_strong(expected, true,
                                                      std::memory_order_acq_rel))
        firstViolationText() = std::move(text);
}

/// A violation that was DETECTED where it could not be thrown.  Thread-local,
/// so in `--batch-mode` the deck that violated is the deck that fails -- a
/// process-wide latch would fail whichever worker reached the next safe point
/// first, which is a different case with a clean record.
struct DeferredSlot {
    bool        armed = false;
    Subsystem   which = Subsystem::Count;
    std::string site;
    std::string reason;
};
inline DeferredSlot& deferred() {
    static thread_local DeferredSlot slot;
    return slot;
}
} // namespace detail

inline unsigned long long fallbacks(Subsystem which) {
    return detail::counter(which).load(std::memory_order_relaxed);
}

inline unsigned long long allowedRefusals(std::size_t index) {
    return detail::allowedCounter(index).load(std::memory_order_relaxed);
}

inline unsigned long long totalAllowedRefusals() {
    unsigned long long sum = 0;
    for (std::size_t i = 0; i < kGpuFullAllowedOuterRefusalCount; ++i)
        sum += allowedRefusals(i);
    return sum;
}

/// How many times this subsystem fell back, and where it first did.
struct SubsystemViolations {
    unsigned long long count   = 0;
    unsigned long long ordinal = 0; ///< 0 when this subsystem never fired
    const char*        site    = nullptr;
    const char*        reason  = nullptr;
};

/// WP10.7.  The per-subsystem half of the receipt: the counter that already
/// existed, joined to the site that produced it.  Post-join readers only, for
/// the same reason firstViolation() is.
inline SubsystemViolations violations(Subsystem which) {
    SubsystemViolations out;
    out.count = fallbacks(which);
    detail::SubsystemFirst& slot = detail::subsystemFirst(which);
    if (!slot.ready.load(std::memory_order_acquire)) return out;
    out.ordinal = slot.ordinal.load(std::memory_order_relaxed);
    out.site    = slot.site.c_str();
    out.reason  = slot.reason.c_str();
    return out;
}

/// WP10.7.  THE SUBSYSTEM THAT FELL BACK FIRST, by ordinal -- Subsystem::Count
/// when nothing did.  This is a real chronological answer: the ordinal is
/// stamped by count(), i.e. at the event, so a lane that was descheduled
/// between its fallback and its receipt string no longer loses the race to a
/// later event on another lane.
inline Subsystem firstViolationSubsystem() {
    Subsystem          winner = Subsystem::Count;
    unsigned long long best   = 0;
    for (int i = 0; i < static_cast<int>(Subsystem::Count); ++i) {
        const SubsystemViolations v = violations(static_cast<Subsystem>(i));
        if (v.ordinal == 0) continue;
        if (best == 0 || v.ordinal < best) {
            best   = v.ordinal;
            winner = static_cast<Subsystem>(i);
        }
    }
    return winner;
}

/// The ordinal of that first violation, or 0.
inline unsigned long long firstViolationOrdinal() {
    const Subsystem which = firstViolationSubsystem();
    return which == Subsystem::Count ? 0ULL : violations(which).ordinal;
}

/// The first violation this RUN recorded, or nullptr.  See the note on
/// detail::firstViolationText: post-join readers only.
///
/// WP10.7.  BUILT FROM THE ORDERED TABLE when there is one, so the string a
/// reader quotes as "the first violation" is the smallest ordinal across every
/// subsystem rather than whichever thread won the latch.  The one-shot latch is
/// the answer only when nothing filed a subsystem slot at all.
inline const char* firstViolation() {
    const Subsystem which = firstViolationSubsystem();
    if (which != Subsystem::Count) {
        static thread_local std::string rendered;
        const SubsystemViolations v = violations(which);
        rendered = std::string("subsystem=") + subsystemName(which) +
                   " site=" + (v.site != nullptr ? v.site : "?") +
                   " reason=" + (v.reason != nullptr ? v.reason : "?");
        return rendered.c_str();
    }
    return detail::firstViolationArmed().load(std::memory_order_acquire)
               ? detail::firstViolationText().c_str()
               : nullptr;
}

inline unsigned long long totalFallbacks() {
    unsigned long long sum = 0;
    for (int i = 0; i < static_cast<int>(Subsystem::Count); ++i)
        sum += fallbacks(static_cast<Subsystem>(i));
    return sum;
}

/// COUNT ONLY.  The raw increment behind everything below; no seam calls it
/// directly any more.  A seam that cannot throw where it stands uses
/// noteDeferred(), which counts HERE and raises at the next safe point --
/// "counted and the case still runs" was the WP1 gap, not the design.
///
/// WP10.7.  IT ALSO STAMPS THE EVENT'S ORDINAL.  The ordinal is what makes
/// `first_violation` a chronological claim rather than a race between string
/// builders, and it has to be taken HERE -- at the increment, which is the
/// moment the fallback happened -- because the naming step that files it runs
/// afterwards and can be descheduled in between.  One relaxed fetch_add and one
/// thread-local store, on a path that was already about to run a whole CPU
/// physics body.
inline void count(Subsystem which) {
    detail::pendingOrdinal() = detail::nextOrdinal();
    detail::counter(which).fetch_add(1, std::memory_order_relaxed);
}

/// NAME THE FIRST FALLBACK, GATE OR NO GATE.
///
/// WHY THIS IS NOT BEHIND `required()`.  `contract_pass` is computed from the
/// fallback counters and printed whether or not the gate is on -- so a gate-OFF
/// run reports `contract_pass:false` and, until this, `first_violation:null`
/// beside it.  That pairing is what sent a reader chasing
/// `ppr_fallbacks:35` with nothing to chase: the receipt said a seam had fired
/// and refused to say which, and the reason (RASBERY_PPR_MODE=master was never
/// ported to the device) sat one string away from being printed.
///
/// THE DEFAULT PATH IS STILL CHEAP.  The armed flag is read FIRST, so only the
/// very first fallback of a run builds a string; every one after it is one
/// acquire load on a path that was already about to run a whole CPU physics
/// body.
///
/// WP10.7.  IT ALSO FILES THE SUBSYSTEM'S OWN FIRST SITE, under the ordinal
/// count() stamped.  Same cheapness rule, one level down: the per-subsystem
/// slot is guarded by its own armed flag, so a subsystem that has already named
/// its first site pays one acquire load and builds nothing.  Seven slots, seven
/// strings, for the whole run.
inline void nameFirstFallback(Subsystem which, const char* where, const char* why) {
    const unsigned long long ordinal = detail::takePendingOrdinal();
    if (!detail::subsystemFirst(which).armed.load(std::memory_order_acquire))
        detail::recordSubsystemFirst(which, where, why, ordinal);
    if (detail::firstViolationArmed().load(std::memory_order_acquire)) return;
    detail::recordFirstViolation(std::string("subsystem=") + subsystemName(which) +
                                 " site=" + (where != nullptr ? where : "?") +
                                 " reason=" + (why != nullptr ? why : "?"));
}

/// COUNT, THEN REFUSE IF THE GATE IS ON.  Call this at the moment the host body
/// is about to run because the device would not.
inline void note(Subsystem which, const char* where, const char* why) {
    count(which);
    nameFirstFallback(which, where, why);
    if (!required()) return;
    throw Violation(which, where, why);
}

/// COUNT, THEN LATCH -- for a seam that has detected the violation somewhere it
/// cannot unwind from.  Today exactly one: the CMFD enqueue hook, which runs
/// INSIDE a live device outer segment where a throw would leave the segment's
/// stream and any in-flight graph capture with nothing written to clean them
/// up.  The case still fails; it fails at the next point the caller declares
/// safe, via raisePending().
inline void noteDeferred(Subsystem which, const char* where, const char* why) {
    count(which);
    nameFirstFallback(which, where, why);
    if (!required()) return;
    detail::DeferredSlot& slot = detail::deferred();
    // THE FIRST ONE NAMES THE CASE.  A segment can refuse several outers before
    // it returns, and the reason a reader wants is the one that started it.
    if (slot.armed) return;
    slot.armed  = true;
    slot.which  = which;
    slot.site   = (where != nullptr ? where : "?");
    slot.reason = (why != nullptr ? why : "?");
}

/// Is a deferred violation waiting on THIS thread?
inline bool violationPending() { return detail::deferred().armed; }

/// Throw it.  Call at a point where unwinding is safe -- for the outer seam,
/// immediately after runSegment() returns, when the segment's stream is drained
/// and any capture is closed.  A no-op when nothing is latched, which is every
/// call with the gate off.
inline void raisePending() {
    detail::DeferredSlot& slot = detail::deferred();
    if (!slot.armed) return;
    const Violation violation(slot.which, slot.site.c_str(), slot.reason.c_str());
    slot = detail::DeferredSlot{};
    throw violation;
}

/// The reason-aware guard.  For Subsystem::Outer, @p why is matched against
/// kGpuFullAllowedOuterRefusals first: an admitted reason is tallied and the
/// host body runs, anything else is note()'s throw.  Every other subsystem
/// behaves exactly as note() -- the allowance list is the OUTER seam's, and a
/// list shared by seams that never agreed on one would be a way to exempt a
/// seam by accident.
inline void noteAllowedOrFail(Subsystem which, const char* where, const char* why) {
    if (which == Subsystem::Outer) {
        const int index = allowedOuterRefusalIndex(why);
        if (index >= 0) {
            detail::allowedCounter(static_cast<std::size_t>(index))
                .fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    note(which, where, why);
}

/// The same decision, at a seam that cannot unwind: allowed reasons are tallied
/// and anything else is latched for raisePending().
inline void noteAllowedOrDefer(Subsystem which, const char* where, const char* why) {
    if (which == Subsystem::Outer) {
        const int index = allowedOuterRefusalIndex(why);
        if (index >= 0) {
            detail::allowedCounter(static_cast<std::size_t>(index))
                .fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    noteDeferred(which, where, why);
}

/// The same, when the arm's own enable predicate decides whether a fallback was
/// even possible.
inline void noteIf(bool arm_requested, Subsystem which, const char* where, const char* why) {
    if (arm_requested) note(which, where, why);
}

// ===========================================================================
// WP10.7: THE ADMISSION GATE -- residency, asked at the door
// ===========================================================================
//
// WHAT IT IS AND WHY IT IS NOT `note()`.  A seam guard fires where the HOST
// BODY IS ENTERED, after the device declined a call that was already underway.
// This one fires BEFORE any physics: at the admission door, where a case's
// device residency is either established or provably cannot be.  The
// difference matters for exactly one reason -- the REASON is still in hand
// here.  The 238 arm-A soak killed four cases with
//
//   subsystem=flatxs site=XSSet::UpdateFlatXS reason=the FlatXS device arm
//   declined; the reference reconstruction loop runs
//
// which is the seam describing itself.  The backend knew the actual reason
// (XsReconBackend::status(): "no CUDA device: ...", "stream: ...", "scalar
// buffer allocation failed") and printed it through a process-wide
// std::call_once warn -- so in a RESIDENT evaluator process the first case to
// hit it printed a reason and the other three printed nothing at all.
//
// THE COUNTER RULE, AND WHY IT IS NOT `note()`'s.  With the gate ON this
// counts and throws exactly once, at the door, and the case never reaches the
// seam -- so a run's per-subsystem counts are what they were.  With the gate
// OFF it counts NOTHING: the seam downstream is still going to fall back and
// still going to count, and a door that counted too would double every
// gate-off tally and break the feature-off identity the campaign reads these
// numbers under.  It still NAMES, because naming is free of that problem and
// is the whole point: the reason reaches the receipt either way.
inline void requireResidency(Subsystem which, const char* where, const char* why) {
    if (!required()) {
        nameFirstFallback(which, where, why);
        return;
    }
    note(which, where, why);
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
    // WP1 follow-up.  PRINTED WHETHER OR NOT ANY FIRED, and by name, so an
    // allowance is a thing a reader can see and argue with rather than a
    // subtraction they cannot.  These do NOT move contract_pass -- that is what
    // being on the list means -- and the rationale for each is in the header
    // comment above kGpuFullAllowedOuterRefusals.
    out << ",\"allowed_refusals\":{";
    for (std::size_t i = 0; i < kGpuFullAllowedOuterRefusalCount; ++i) {
        if (i > 0) out << ",";
        out << "\"" << kGpuFullAllowedOuterRefusals[i].reason << "\":"
            << allowedRefusals(i);
    }
    out << "}";
    out << ",\"contract_pass\":" << (contractPass() ? "true" : "false");
    // WHICH SEAM, IN THE RECEIPT ITSELF.  `contract_pass:false` used to be the
    // whole story and the run still exited 0; now the run fails, and the line
    // that says so has to name the site or the next reader repeats the 181
    // investigation from scratch.
    //
    // AND IT IS FILLED WITH THE GATE OFF TOO (nameFirstFallback).  A gate-off
    // run prints `contract_pass:false` for exactly the same reason a gate-on
    // one does; a null beside it made the commonest reading of this receipt --
    // "which arm did not engage?" -- unanswerable from the receipt.
    out << ",\"first_violation\":";
    if (firstViolation() != nullptr)
        out << "\"" << firstViolation() << "\"";
    else
        out << "null";
    // WP10.7.  THE ORDINAL, so `first_violation` is a checkable claim.  A
    // reader can now see that the named site really is the smallest ordinal in
    // `violations` below rather than taking the field's name on trust -- which
    // is what the 238 arm-A receipt asked them to do, with two subsystems
    // firing in the same generation and no way to order them.
    out << ",\"first_violation_seq\":" << firstViolationOrdinal();
    // WP10.7.  PER SUBSYSTEM: the count that already existed, joined to the
    // site that produced it and the ordinal it happened at.  Printed for every
    // subsystem, zeros included, for the same reason `allowed_refusals` is:
    // "this arm never fell back" and "this arm has no counter" must not look
    // the same.  `flatxs_fallbacks:4` used to send a reader to a 16k-line log
    // for a string the process already held.
    out << ",\"violations\":{";
    for (int i = 0; i < static_cast<int>(Subsystem::Count); ++i) {
        const auto                which = static_cast<Subsystem>(i);
        const SubsystemViolations v     = violations(which);
        if (i > 0) out << ",";
        out << "\"" << subsystemName(which) << "\":{\"count\":" << v.count
            << ",\"seq\":" << v.ordinal << ",\"site\":";
        if (v.site != nullptr) out << "\"" << v.site << "\""; else out << "null";
        out << ",\"reason\":";
        if (v.reason != nullptr) out << "\"" << v.reason << "\""; else out << "null";
        out << "}";
    }
    out << "}";
}

/// THE RUN-LEVEL HALF OF THE GATE.
///
/// `contract_pass:false` MUST imply a nonzero exit whenever the gate is on --
/// that is the whole point of a fail-closed contract, and on host 181 it was
/// not true: arm X printed `contract_pass:false, outer_fallbacks:71` and exited
/// 0, because the only seam that fired was count-only and no one read the
/// receipt back.  The per-case throw is still the primary mechanism (plan Sec
/// 6.3 item 4 keeps the rest of a batch running); this is the backstop that
/// makes the RUN fail even if a future seam can only count.
///
/// Call AFTER every Driver has joined and after appendReceiptFields, in each of
/// main.cpp's three branches.
inline int enforceExitCode(std::ostream& out, int exit_code) {
    if (!required() || contractPass()) return exit_code;
    out << "[RASBERY][FAIL] exit_code=1 what=[RASBERY][GPU_FULL][VIOLATION] "
        << (firstViolation() != nullptr
                ? firstViolation()
                : "a GPU arm fell back to CPU numerics under RASBERY_GPU_FULL; "
                  "see the per-subsystem counters in the receipt above")
        << std::endl;
    return exit_code != 0 ? exit_code : 1;
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

/// Count-only variant; see gpufull::count.  NO SEAM MAY USE THIS ANY MORE --
/// it is kept only so a stray use is a compile-visible token the contract test
/// can refuse by name.  A seam that cannot unwind uses RASBERY_GPU_FULL_DEFER,
/// which still fails the case, at the next point its caller declares safe.
#define RASBERY_GPU_FULL_COUNT(which) \
    ::rasbery::gpufull::count(::rasbery::gpufull::Subsystem::which)

/// THE REASON-AWARE GUARD.  @p why must be an enum's own name() string, so the
/// allowance list in this header and the ladder that produced the reason cannot
/// drift apart silently.  Allowed reasons are tallied; everything else throws.
#define RASBERY_GPU_FULL_GUARD_ALLOWED(which, where, why) \
    ::rasbery::gpufull::noteAllowedOrFail(::rasbery::gpufull::Subsystem::which, (where), (why))

/// The same, at a seam that cannot unwind where it stands: allowed reasons are
/// tallied, everything else is LATCHED and raised by
/// RASBERY_GPU_FULL_RAISE_PENDING at the caller's next safe point.
#define RASBERY_GPU_FULL_DEFER_ALLOWED(which, where, why) \
    ::rasbery::gpufull::noteAllowedOrDefer(::rasbery::gpufull::Subsystem::which, (where), (why))

/// Throw whatever a deferring seam latched on this thread.  A no-op when
/// nothing is latched, which is every call with the gate off.
#define RASBERY_GPU_FULL_RAISE_PENDING() ::rasbery::gpufull::raisePending()

/// WP10.7: THE ADMISSION GATE.  One token, so
/// tools/test_evaluator_residency_contract.py can scan for the door the way
/// tools/test_gpu_full_fail_closed.py scans for the seams.  @p why must be the
/// establishing layer's OWN reason string (XsReconBackend::status(),
/// CudaOuterSegment::status()) and not a restatement of the guard -- a door
/// that described itself would be the seam's defect moved one level up.
#define RASBERY_GPU_FULL_REQUIRE_RESIDENCY(which, where, why) \
    ::rasbery::gpufull::requireResidency(::rasbery::gpufull::Subsystem::which, (where), (why))
