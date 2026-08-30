#include "CudaBICGBackend.h"

#include "BatchRefill.h"
#include "CmfdAssemblyKernel.h"
#include "CudaTransferMirror.h"
#include "CudaXsReconBackend.h" // rasberyHostPinningEnabled(): header-only gate
#include "Geometry.h"
#include "GpuCaptureArbiter.h"
#include "GpuFullContract.h" // F9: the seam tally the three fallback fields read
#include "GpuGraphSplice.h"
#include "XferLedger.h"
#include "pch.h"

#include <cooperative_groups.h> // WP17: the persistent BiCG arm's grid barrier
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rasbery {
namespace {

constexpr int kDefaultBlockSize = 256;

enum ScalarSlot : int {
    kRhoNew = 0,
    kR0V,
    kPts,
    kPtt,
    kRho,
    kAlpha,
    kOmega,
    kInitialNorm,
    /// ||b - A*phi0|| captured once by reset(); the fixed reference of the
    /// relative inner-loop exit test, which now lives on the device.
    kR20,
    /// _epsbicg, pushed down from the host whenever setIterLim changes it.
    kEps,
    // ---- device-resident CMFD sweep state (RASBERY_GPU_CMFD_SWEEP) ----
    // Kept contiguous from kSweepFirst so the host stages/harvests the whole
    // block with one memcpy without touching the BiCG slots above.
    kEigv,        ///< current eigenvalue
    kReigv,       ///< 1/eigv
    kReigvs,      ///< shifted 1/(eigv+eshift), 0 when unshifted
    kErrl2,       ///< fission-source change of the last sweep
    kEpsl2,       ///< sweep convergence criterion
    kEshift,      ///< Wielandt shift
    kReigvdel,    ///< reigv - reigvs, refreshed at each sweep start
    kSweepBudget, ///< iout slots remaining for this drive()
    kSweepsDone,  ///< iout advances made on the device
    kIcmfdBudget, ///< 20*ncmfd, the negative-flux retry cap
    kIcmfdDone,   ///< sweep attempts including retries (host value at entry)
    kNegative,    ///< negative flux entries of the last sweep
    kSweepState,  ///< 0 running, 1 converged, 2 needs-host wiel, 3 budget spent
    kGammaD,      ///< wiel sums exported for the needs-host fallback
    kGammaN,
    kErrAcc,
    kNgxyz,       ///< ng*nxyz, for the all-negative reset rule
    // ---- Rev.7.1 Task 6 Step 3: the sweep-slot budget, off the graph key ----
    //
    // `unroll` used to be a CAPTURE-TIME loop bound and part of the graph cache
    // key.  It is the REMAINING sweep budget (`_ncmfd - iout`, BICGCMFD.cpp:394)
    // and _ncmfd is 5 (Driver.h:2114), so it walked 5,4,3,... and back to 5 on
    // the next drive() -- the sweep graph was destroyed and rebuilt continuously
    // and graph_reinstantiations climbed for the whole run.
    //
    // Moving it here makes the captured graph CAPACITY instead of
    // configuration: it may hold more slots than a launch may spend, and the
    // excess halt in cmfd_sweep_begin before touching anything.  That is what
    // makes a deeper capture bit-identical to an exact one rather than merely
    // similar.
    kSweepSlotBudget, ///< sweep slots this launch may spend (was `unroll`)
    kSweepSlots,      ///< slots spent so far in this launch
    kScalarCount
};

constexpr int kSweepFirst = kEigv;
constexpr int kSweepCount = kScalarCount - kSweepFirst;

// ---------------------------------------------------------------------------
// Rev.7.1 Task 10 part 3: the segment-scoped accumulator's layout.
// ---------------------------------------------------------------------------
//
// ONE FLAT BLOCK OF DOUBLES, and deliberately not the nested struct itself:
// a __global__ signature that names CudaBatchArena::CmfdSweepProbeSink::Accum
// drags a host header's nesting into device code for no benefit, while a
// `double*` plus these indices is the same bytes with none of it.  The
// static_assert below is what keeps the two spellings from drifting.
enum AccumSlot : int {
    kAccAttempts = 0, ///< sum of kIcmfdDone over the launches whose verdict ran
    kAccSweeps,       ///< sum of kSweepsDone
    kAccLaunches,     ///< how many those were
    kAccState,        ///< the last such launch's kSweepState
    kAccNegative,     ///< the last such launch's negative census, as 0/1
    kAccExceptional,  ///< 1 once a launch ended in state 0 or 2
    kAccSaved,        ///< [kSweepCount] the abandoned launch's whole block
    kAccCount = kAccSaved + kSweepCount
};
static_assert(sizeof(CudaBatchArena::CmfdSweepProbeSink::Accum) ==
                  static_cast<std::size_t>(kAccCount) * sizeof(double),
              "the accumulator's host struct and its device slot map disagree");
static_assert(sizeof(CudaBatchArena::CmfdSweepProbeSink::Accum::saved) ==
                  static_cast<std::size_t>(kSweepCount) * sizeof(double),
              "the saved scalar block is not one sweep block wide");

/// Device-side tallies harvested once per outer instead of once per iteration.
enum CounterSlot : int {
    kRestartCount = 0,
    kEarlyExitCount,
    kSolveCount,
    /// Captured iterations that found `halt` already raised and did nothing.
    /// Appended last so the existing slot indices are untouched.
    kOverrunCount,
    kCounterSlots
};

void cuda_check(cudaError_t value, const char* expression) {
    if (value == cudaSuccess) return;
    std::ostringstream message;
    message << expression << ": " << cudaGetErrorString(value);
    throw std::runtime_error(message.str());
}

void cublas_check(cublasStatus_t value, const char* expression) {
    if (value == CUBLAS_STATUS_SUCCESS) return;
    std::ostringstream message;
    message << expression << ": cuBLAS status " << static_cast<int>(value);
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr)

/// Rev.7.1 Task 18d: a capture window that CANNOT be left open.
///
/// THE HOLE IT CLOSES.  Both graph captures on the arena stream were written as
///
///     rc = cudaStreamBeginCapture(stream, ...);
///     if (rc == cudaSuccess) { enqueue_...(); rc = cudaStreamEndCapture(...); }
///
/// and `enqueue_outer` contains a CUDA_CHECK -- which THROWS.  When a sibling
/// deck's stand-up invalidated the capture, that CUDA_CHECK threw, the
/// EndCapture was skipped, and the arena stream stayed in capture mode for the
/// rest of the process: every later operation on it answered "operation not
/// permitted when stream is capturing" or "operation failed due to a previous
/// error during capture", and all M decks died.  ONE deck's transient became
/// the batch's death because of these two lines.
///
/// The guard makes the window exception-safe (the destructor ends a capture the
/// code did not end, discards the partial graph and clears the sticky error)
/// and pairs it with the process-wide CaptureWindow, which is what stops the
/// sibling's allocation from invalidating it in the first place.
class ScopedStreamCapture {
  public:
    ScopedStreamCapture(cudaStream_t stream, const char* tag)
        : _stream(stream), _window(stream, tag) {}

    /// WP19: GLOBAL IS GONE, AND ITS REMOVAL IS THE FIRST HALF OF THE FIX.
    ///
    /// It used to be reachable as RASBERY_GPU_CAPTURE_MODE=global, on the
    /// theory that "CUDA policing the rule" was a diagnostic.  It is not a
    /// diagnostic in a batch: cudaStreamCaptureModeGlobal makes every unsafe
    /// API in every UNRELATED thread fail with
    ///
    ///     operation not permitted when stream is capturing
    ///
    /// which is not a report of the defect, it IS the defect -- one lane's
    /// capture killing a sibling lane's stand-up, with the sibling holding the
    /// corpse.  The mode is chosen per site now and never globally: ThreadLocal
    /// where the capture is a leaf (only the capturing thread is constrained),
    /// Relaxed where a conditional-node build has to touch a second stream
    /// inside the window.  Neither can make another lane's call illegal.
    /// tools/test_capture_arbiter_contract.py holds the absence against the
    /// source, so this cannot come back as an env flag.
    static cudaStreamCaptureMode captureMode() {
        static const cudaStreamCaptureMode mode = [] {
            const char* v = std::getenv("RASBERY_GPU_CAPTURE_MODE");
            if (v != nullptr && std::string(v) == "relaxed")
                return cudaStreamCaptureModeRelaxed;
            return cudaStreamCaptureModeThreadLocal;
        }();
        return mode;
    }

    cudaError_t begin() {
        const cudaError_t rc = cudaStreamBeginCapture(_stream, captureMode());
        _open = (rc == cudaSuccess);
        // DIAGNOSTIC ONLY, and the reason the root cause is a measurement
        // rather than a reading of the CUDA manual.  RASBERY_GPU_CAPTURE_
        // STALL_US widens this window to a chosen number of microseconds, which
        // turns a 4-in-20 race into a certainty: an 8-deck batch with the
        // arbiter off and a 200 ms stall died 4 times in 5, and the
        // [RASBERY][CAPTURE] trace separates the five runs perfectly --
        // 73 sibling cudaHostRegister calls inside the window and NO
        // cudaDeviceSynchronize is the run that survived; one
        // cudaDeviceSynchronize inside the window is every run that died.
        // With the arbiter on and the SAME stall, 0 of 5.  Unset (the default)
        // this is one relaxed load per capture.
        if (_open) {
            static const long stall_us = [] {
                const char* v = std::getenv("RASBERY_GPU_CAPTURE_STALL_US");
                return v == nullptr ? 0L : std::strtol(v, nullptr, 10);
            }();
            if (stall_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(stall_us));
        }
        return rc;
    }

    cudaError_t end(cudaGraph_t* graph) {
        if (!_open) return cudaErrorStreamCaptureUnmatched;
        _open = false;
        return cudaStreamEndCapture(_stream, graph);
    }

    ~ScopedStreamCapture() {
        if (!_open) return;
        cudaGraph_t partial = nullptr;
        cudaStreamEndCapture(_stream, &partial);
        if (partial != nullptr) cudaGraphDestroy(partial);
        cudaGetLastError();
        rasbery::captureArbiterStats().captures_unwound.fetch_add(
            1, std::memory_order_relaxed);
    }

    ScopedStreamCapture(const ScopedStreamCapture&)            = delete;
    ScopedStreamCapture& operator=(const ScopedStreamCapture&) = delete;

  private:
    cudaStream_t            _stream = nullptr;
    rasbery::CaptureWindow  _window;
    bool                    _open   = false;
};

bool envFlagDisabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string s(value);
    return s.empty() || s == "0" || s == "off" || s == "OFF" ||
           s == "false" || s == "FALSE";
}

bool cmfdAssemblyEnabled() {
    static const bool enabled = !envFlagDisabled("RASBERY_GPU_CMFD_ASSEMBLY");
    return enabled;
}

bool cmfdScalarFusionEnabled() {
    static const bool enabled = !envFlagDisabled("RASBERY_GPU_CMFD_SCALAR_FUSION");
    return enabled;
}

// ---------------------------------------------------------------------------
// WP7 stage B -- RASBERY_GPU_CMFD_FUSE, a BITMASK, DEFAULT 15 (kFuseDefaultMask)
// since the v5 freeze (2026-08-30).  `RASBERY_GPU_CMFD_FUSE=0` is the off switch
// and still launches the reference kernels, unchanged.
//
// Every bit below merges two ADJACENT graph nodes into one kernel.  The
// reference kernels stay exactly where they are and are what mask 0 launches,
// so the fused path is always measured against a live reference rather than
// against a memory of one.
//
// THE RULE EVERY BIT OBEYS, and the reason each of them is B0 (bit-identical):
//
//     same operations, same order, same rounding.
//
// Concretely: a fused body is the two reference bodies CONCATENATED, with every
// expression copied character for character.  That matters for more than
// readability -- nvcc contracts `x + a * b` into an FMA under the default
// --fmad=true, so an expression that is retyped rather than copied can change
// its own rounding even when it is algebraically the same.  Where a fused
// kernel needs a spelling the reference did not have, it uses the explicit
// round-to-nearest intrinsics (__fmul_rn / __fadd_rn / __dmul_rn / __ddiv_rn)
// so the contraction question does not arise; of the four bits below only bit 3
// carries such an expression, and it inherits __ddiv_rn from its reference.
//
// WHAT IS DELIBERATELY NOT HERE, and why (long form in
// docs/WP7_CMFD_GRAPH_CENSUS_20260831_KO.md Sec 4):
//
//   * colored_block_sweep's colour passes -- 23.7% of single-run GPU time and
//     the largest single item in the Amdahl table.  Sweep k+1 reads x at
//     NEIGHBOURING nodes, so it depends on sweep k's writes ACROSS THE WHOLE
//     GRID; the kernel boundary IS that barrier and the colour order IS the
//     Gauss-Seidel semantics.  There is no "same loop, same i" concatenation
//     available: a merged kernel would need a device-wide barrier between the
//     colours (cooperative grid.sync), which is a different and far less local
//     argument than anything in this bitmask, and it would change the
//     occupancy the sweep runs at.  NOT FUSABLE, and it stays that way.
//   * the trailing reduce_dot_stage1 of an iteration, and the prologue one.
//     Their stage-2 partners (reduce_norm_accumulate_stage2 and
//     reduce_norm_store_reference_stage2) already carry a SECOND kernel's body
//     under the scalar-fusion arm, and they gate on `active` where stage 1
//     gates on `halt` -- the difference between those two guards is exactly
//     where the over-run telemetry lives.  Folding stage 1 in would collapse
//     the two guards into one and change what the counters mean.
//   * the FP32 twins (reduce_dot_stage1_f32 / reduce_dot2_stage1_f32).  The
//     mixed-precision inner loop is its own opt-in arm; keeping the two gates
//     independent means neither A/B has to carry the other's variance.
// ---------------------------------------------------------------------------
enum CmfdFuseBit : unsigned {
    kFuseDot      = 1u << 0, ///< reduce_dot_stage1  + reduce_dot_stage2
    kFuseDot2     = 1u << 1, ///< reduce_dot2_stage1 + reduce_dot2_stage2
    kFuseWiel     = 1u << 2, ///< cmfd_wiel_stage1   + cmfd_wiel_finalize_chunked
    kFuseSweepPre = 1u << 3, ///< cmfd_sweep_gate    + cmfd_sweep_patch
    kFuseAllBits  = kFuseDot | kFuseDot2 | kFuseWiel | kFuseSweepPre
};

/// THE DEFAULT, since the v5 freeze.  Mask 15 was measured B0 against mask 0 on
/// BOTH hosts -- 238 (all six masks digest 0d15abf29d222a02 / 4382, pricing
/// block 4) and 181 (all six masks digest 1d897e3f77204799, h5diff 0 lines vs
/// mask 0) -- and it is the only mask that cleared the 0.2 s adoption threshold
/// (-0.396 s interleaved, block 4b; mask 4 managed -0.183 s and was not
/// adopted).  A default is a claim, and this one is the strongest kind the
/// campaign has: the fused body is the two reference bodies concatenated
/// character for character, so "same operations, same order, same rounding"
/// is a property of the text, not of a measurement that could drift.
enum : unsigned { kFuseDefaultMask = kFuseAllBits };

/// Read ONCE, like every other RASBERY_* gate: the mask fixes the captured
/// graph topology, so it must not be able to change between two outers of the
/// same run.  Decimal or 0x-prefixed hex.
///
/// UNSET IS kFuseDefaultMask, not 0.  Explicit `=0` still parses to 0 and still
/// selects the reference kernels, so the off switch survives the flip -- that is
/// what keeps mask 0 a LIVE reference rather than a memory of one.  An
/// UNPARSEABLE value also falls back to the default rather than to 0: after the
/// flip, silently dropping a typo to the reference path would look like a
/// performance regression with no cause in any receipt, and the [CMFD][GRAPH]
/// receipt's `fuse_mask` is what a reader would have to catch it with.
unsigned cmfdFuseMask() {
    static const unsigned mask = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_FUSE");
        if (v == nullptr) return static_cast<unsigned>(kFuseDefaultMask);
        char*               end    = nullptr;
        const unsigned long parsed = std::strtoul(v, &end, 0);
        if (end == v) return static_cast<unsigned>(kFuseDefaultMask);
        return static_cast<unsigned>(parsed) & static_cast<unsigned>(kFuseAllBits);
    }();
    return mask;
}

/// Opt-IN counterpart of envFlagDisabled: unset means off.
bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string s(value);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
             s == "false" || s == "FALSE");
}

/// Mixed-precision inner BiCGSTAB (see the FP32 section below).  Read ONCE and
/// cached: the choice fixes the captured graph topology, so it must not be able
/// to change between two outers of the same run.
bool cmfdFp32InnerEnabled() {
    static const bool enabled = envFlagEnabled("RASBERY_GPU_CMFD_FP32");
    return enabled;
}

// ---------------------------------------------------------------------------
// Device-side inner-loop control.
//
// The inner BiCGSTAB loop used to be steered from the host: every iteration
// finished with a 64-byte status D2H plus a cudaStreamSynchronize so the CPU
// could evaluate `r2/r20 < eps` and decide whether to launch another one.  At
// an observed 2.16 inner iterations per outer that is 4.16 full pipeline
// drains per outer -- roughly 218 000 of them on CY02 -- and it makes the
// whole sequence uncapturable as a CUDA graph, because the graph topology
// would depend on a host decision taken mid-stream.
//
// The test is a comparison of two doubles that both already live on the
// device, so it belongs there.  `halt` is the device word that replaces the
// host's `break`: once set, every kernel of every remaining iteration returns
// immediately.  The trailing iterations are therefore launched unconditionally
// (a fixed topology the graph can capture) but execute nothing.  The decision
// sequence is bit-for-bit the one the host used to take -- same operands, same
// order, same comparison -- so the numerical trajectory is unchanged.
//
// With the batch axis in place `halt` is per slot, and it doubles as the
// inactive-slot mask: initialize_solver_state seeds it with 1 for every slot
// that is not taking part in this launch, so those slots' kernels retire on
// their first instruction while the participating slots run untouched.
//
// ---- On "batch K iterations per graph launch" -----------------------------
//
// That optimisation is what the paragraph above already describes, and the
// counters say so: `graph_launches` counts CMFD SWEEPS, not iterations, and
// `iter_batch` is the 1 + nmax iterations each of those launches carries (4 on
// KNGR).  There is no per-iteration launch or per-iteration status D2H left to
// remove -- status_d2h_calls == graph_launches, one per sweep.
//
// Going DEEPER than 1 + nmax is possible (RASBERY_GPU_ITER_BATCH) and is
// provably inert, because the last budgeted iteration raises the halt itself
// and the gating below makes the rest no-ops.  It is also measurably a loss:
// on KNGR, K = 8 executes 432 704 no-op iterations, returns a bit-identical
// h5, and costs +8.7 s of 84.9 s.  The batching crossover the literature
// points at was passed when the inner loop was first captured; past it, the
// extra graph nodes are dispatched whether or not they do anything.
//
// What is left at the sweep boundary is host work, not launch latency: the
// Wielandt update and updls run on the CPU between sweeps, so the stream must
// drain every sweep.  Removing that drain means moving those to the device --
// which is RASBERY_GPU_CMFD_SWEEP (enqueue_sweeps), today reachable only
// through the batch arena.
//
// ---- On the cost that IS left: per-node dispatch ---------------------------
//
// With one launch and one status D2H per sweep, what remains is the graph's
// own per-node dispatch: on KNGR the captured outer was 117 nodes (7 prologue
// + 4 iterations x 27 + finalize + the status copy) for a 17 000-unknown
// system whose arithmetic is ~10 us, and the launch measured ~449 us.  The
// lever is therefore the NODE COUNT, and the only safe way to lower it is to
// concatenate adjacent elementwise kernels over the same index domain: two
// loop bodies run back to back over the same i is bit-identical to two kernels
// run back to back over the same i, by construction.
//
// What was fused, and the count it bought (nmax = 3, rb_sweeps = 4):
//
//   prologue   7 -> 5   begin_outer_fused = invert + A*phi + initial residual
//   iteration 27 -> 22  prepare_p + block_jacobi  -> prepare_p_jacobi
//                       update_s  + block_jacobi  -> update_s_jacobi
//                       (s.t) and (t.t)           -> one dot2 pair
//                       the iter_flags memset node -> two existing scalar
//                       kernels (initialize_solver_state / accumulate_iteration)
//   outer    117 -> 95  (kernel+memset nodes 116 -> 94)
//
// What was deliberately NOT fused, and why:
//
//   * the colour sweeps.  Each reads x at NEIGHBOURING nodes, so it depends on
//     the previous sweep across the whole grid; the kernel boundary is that
//     barrier and the colour order is the Gauss-Seidel semantics.  Collapsing
//     them needs a cooperative grid.sync(), which is a different (and much
//     less local) argument than "same loop, same i" -- 8 of the remaining 22
//     nodes sit here if anyone wants to make it.
//   * anything through a reduction.  reduce_dot_stage1/2 keep their exact
//     two-stage structure, partition and fold order.  dot2 is not an exception
//     to that rule: it runs two INDEPENDENT reductions side by side, each with
//     the partition and tree it had alone.
//   * accumulate_iteration into the trailing stage2, and store_reference_norm
//     into the prologue stage2.  Both are one-thread scalar kernels with the
//     same launch geometry as their stage2, so they would fuse cleanly, but
//     they gate on `active` where stage2 gates on `halt` -- and the difference
//     between those two guards is exactly where the over-run telemetry lives.
// ---------------------------------------------------------------------------
#define HALT_GUARD(halt) \
    if (*(halt) != 0u) return

// ---------------------------------------------------------------------------
// ACTIVE-SLOT COMPACTION (RASBERY_GPU_CMFD_COMPACT, default OFF)
//
// The arena's grids are `slots`-wide on the batch axis whatever the arrival
// width, so a 64-slot arena serving a mean of 2.9-22 instances dispatches
// 90-95% padding blocks.  Each of those still costs a dispatch (W0 measured
// 0.78 us per node, and the block count is what a node's cost scales with),
// and there are ~25 kernels per inner iteration.
//
// The fix is the Task 8 nodal one, transplanted: grid.y becomes a LOGICAL lane
// over a bucket wide enough for the arrivals, and `slot_map[logical]` names the
// physical slot that lane drives.  Everything downstream of the map is
// unchanged -- every per-slot array (scalars, halt, active, sweep_halt,
// device_assembly_active, the vector strides) is still indexed by the PHYSICAL
// slot, because that is what the host's Slot table and the status D2H agree
// on.  The map is the only thing allowed to be logical.
//
// TWO INVARIANTS, both of which a test would otherwise miss because they only
// bite when compaction is ON:
//
//   * Nothing may read blockIdx.y as a slot index.  It is correct exactly when
//     the map is the identity, which is the compaction-OFF case -- so the bug
//     passes every OFF test and every ON test that happens to arrive full.
//     RASBERY_CMFD_SLOT is the only reader; the contract test enforces that.
//
//   * The guard must be the kernel's FIRST statement, before any __syncthreads
//     or shared write.  Unlike the nodal kernels these DO carry barriers
//     (cmfd_wiel_finalize).  That is safe here and only here because the guard
//     is BLOCK-UNIFORM: it reads blockIdx.y and nothing else, so either the
//     whole block returns or none of it does, and no barrier is ever reached
//     by a partial block.
//
// Compaction OFF is the FULL IDENTITY over physical slots (slot_map[i] == i
// for every declared slot, participant or not) with lanes == slots, so the OFF
// launch visits exactly the blocks it visited before and masks them exactly
// where it masked them before -- the halt guard, not the map.
// ---------------------------------------------------------------------------
#define RASBERY_CMFD_SLOT_ARGS const int* __restrict__ slot_map, const int lanes

#define RASBERY_CMFD_SLOT(m)                                            \
    int m = 0;                                                          \
    do {                                                                \
        const int rasbery_logical = static_cast<int>(blockIdx.y);       \
        if (rasbery_logical >= lanes) return;                           \
        m = slot_map[rasbery_logical];                                  \
        if (m < 0) return;                                              \
    } while (0)

/// The dispatch ladder, shared with the nodal arena (GpuPhaseScheduler.h's
/// kDispatchBuckets).  A graph bakes grid.y, so the dispatch width IS topology
/// and every distinct width is a separate instantiation; the coarse ladder is
/// what keeps that list at nine entries instead of sixty-four.
inline int cmfdBucketFor(int count, int slots) {
    static const int kBuckets[] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
    for (int b : kBuckets)
        if (count <= b) return b < slots ? b : slots;
    return slots;
}

inline int cmfdBucketIndex(int lanes) {
    static const int kBucketIndex[9] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
    for (int i = 0; i < 9; ++i)
        if (lanes <= kBucketIndex[i]) return i;
    return 8;
}

// ---------------------------------------------------------------------------
// WIELANDT FOLD MODE (RASBERY_GPU_WIEL_FOLD, default serial)
//
// Two ways to sum the three Wielandt addend arrays, and they are NOT the same
// number.  The distinction is a classification, not a preference:
//
//   serial (default)  One strict l-ascending chain per sum, which is
//                     BICGCMFD::wiel's own accumulation reproduced bit for
//                     bit.  Class B0: the frozen reference output is what it
//                     produces, so it is the only acceptance path.  It is also
//                     a hard serial dependency -- 8451 dependent DADDs -- and
//                     measures 258 us/call on sm_61 against a 202 us floor,
//                     which is 54% of GPU kernel time on the S2 arm.
//
//   chunked           reduce_dot_stage1/2's shape: a fixed contiguous chunk
//                     per block, a fixed per-thread traversal, a fixed binary
//                     tree, then a strict ascending fold over the partials.
//                     16.1 us/call at the KNGR mesh on sm_61 -- 19.8x against
//                     the serial 318.9 us in the same harness.  Class N1:
//                     DETERMINISTIC (the partition is a pure function of
//                     (nxyz, gridDim.x), the traversal and the tree are fixed,
//                     the stage-2 walk is strict, and there is not an atomic
//                     anywhere in it, so it is bit-identical run to run on a
//                     given arch -- the replay asserts that over 10 launches)
//                     but NOT bit-identical to serial.
//
//                     HOW FAR FROM SERIAL, measured, on addends with the
//                     12-decade magnitude spread the real ones have: 71 ULP
//                     worst case at nxyz = 8451, growing with the range (0 ULP
//                     below one chunk, 4 at 128, 37 at 4096).  A flat fixture
//                     reports 1-4 ULP and is not evidence -- reassociation
//                     only costs where the magnitudes differ.
//
//                     71 ULP is ~1.6e-14 relative, and note the SIGN of the
//                     argument: pairwise summation has an O(log n) error bound
//                     against serial's O(n), so the chunked sum is the more
//                     accurate of the two, not the less.  The delta is still
//                     disqualifying for acceptance -- the frozen output is the
//                     serial one and "closer to the exact sum" is not "equal
//                     to the reference" -- which is exactly why this is a
//                     campaign arm gated against the v2 golden and never a
//                     silent default.
//
// Which is why the mode is opt-in and the receipt records both the mode and
// where it came from: an N1 result that reached a report without a
// [RASBERY][CMFD][WIEL_FOLD] line saying "chunked" would be indistinguishable
// from a B0 one.
// ---------------------------------------------------------------------------
enum class WielFoldMode { SERIAL, CHUNKED };

/// Parsed once per process, trimmed and case-folded like Driver.h's xeMode().
/// A typo must not silently buy a mode: the two arms produce different bits.
inline WielFoldMode wielFoldMode() {
    static const WielFoldMode mode = [] {
        const char* value = std::getenv("RASBERY_GPU_WIEL_FOLD");
        if (value == nullptr) return WielFoldMode::SERIAL;
        std::string s(value);
        const auto  not_space = [](unsigned char c) { return std::isspace(c) == 0; };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s.empty() || s == "serial") return WielFoldMode::SERIAL;
        if (s == "chunked") return WielFoldMode::CHUNKED;
        std::cerr << "[RASBERY][WARN][wiel] RASBERY_GPU_WIEL_FOLD=\"" << value
                  << "\" is not a mode (serial|chunked); using serial\n";
        return WielFoldMode::SERIAL;
    }();
    return mode;
}

/// Was the mode chosen, or defaulted?  Provenance for the receipt: "chunked"
/// is only ever reachable through the environment, and the receipt has to be
/// able to say so.
inline bool wielFoldModeFromEnv() {
    static const bool from_env = [] {
        const char* value = std::getenv("RASBERY_GPU_WIEL_FOLD");
        if (value == nullptr) return false;
        std::string s(value);
        const auto  not_space = [](unsigned char c) { return std::isspace(c) == 0; };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return !s.empty();
    }();
    return from_env;
}

/// One line, the first time the sweep path enqueues.  `chunk` is the number of
/// contiguous chunks the fold range is cut into -- with nxyz it fixes every
/// bit the chunked mode produces -- and is 0 for serial, which has no
/// partition at all.
inline void reportWielFold(int chunks) {
    // enqueue_sweeps is launcher-only today, but "only one thread ever calls
    // this" is the kind of invariant that quietly stops being true; an
    // exchange costs nothing and cannot double-print.
    static std::atomic<bool> done{false};
    if (done.exchange(true, std::memory_order_relaxed)) return;
    const bool chunked = wielFoldMode() == WielFoldMode::CHUNKED;
    std::cout << "[RASBERY][CMFD][WIEL_FOLD] {\"mode\":\""
              << (chunked ? "chunked" : "serial") << "\",\"source\":\""
              << (wielFoldModeFromEnv() ? "env" : "default") << "\",\"chunk\":"
              << (chunked ? chunks : 0) << "}" << std::endl;
}

/// RASBERY_GPU_CMFD_COMPACT, default OFF.  Read once, like every other gate.
inline bool cmfdCompactEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_COMPACT");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

// ---------------------------------------------------------------------------
// WP17: THE CMFD BLOCK WIDTH (RASBERY_GPU_CMFD_BLOCK), default UNCHANGED.
//
// WHAT THE PROFILE SAYS.  238, RTX PRO 6000 Blackwell, 188 SMs, v6 single deck
// (KNGR, 8,451 nodes x 2 groups): every per-node CMFD kernel dispatches
// ceil(8451 / 256) = 34 blocks and every per-element one ceil(16902 / 256) =
// 67, against 188 SMs.  colored_block_sweep alone is 379,027 launches at
// 2.55 us -- 967.6 ms -- on 34 blocks, i.e. 18 % of the SM array; matvec is 34
// blocks at 2.75 us.  Even at 8 MPS clients the array is under a third
// occupied.  The kernels are LAUNCH-LATENCY BOUND, not compute bound, and the
// 2.5 us floor is what most of the CMFD wall is made of.
//
// THE LEVER, AND WHY IT IS B0.  All five per-node/per-element classes -- the
// colour sweep, the two-group matvec, prepare_p_jacobi, update_s_jacobi and
// update_solution -- are ELEMENTWISE in the index the launch geometry hands
// them:
//
//   * the index is `blockIdx.x * blockDim.x + threadIdx.x`, guarded by
//     `>= nxyz` (or `>= n`), so which block owns which node changes but the
//     SET of nodes touched, and the expression each one evaluates, do not;
//   * none of the five has a __shared__ array, a __syncthreads(), a warp
//     shuffle or any cross-thread reduction, so no operand pairing exists to
//     be re-associated by a different partition;
//   * the only cross-thread writes are `atomicOr(flags + m, ...)` -- bitwise
//     OR, commutative and idempotent, so the final mask is the same whatever
//     order the blocks retire -- and the `l == 0` / `i == 0` scalar stores,
//     which are keyed to the INDEX and not to the block, so exactly one thread
//     still performs them wherever it lands.
//
// THE ONE THING THIS IS NOT.  The block partition is NOT the Gauss-Seidel
// colouring.  colored_block_sweep takes `target_color` as an argument and
// filters on `colors[l] != target_color`, a per-NODE array built once by the
// greedy BFS colouring in init(); the launch covers every node and the mask
// selects the colour.  init() throws when two adjacent nodes share a colour,
// so within one colour no thread reads a node another thread of the same
// launch writes.  Splitting a colour across MORE blocks therefore keeps the
// sweep order exactly -- same colour, same one-shot update, same neighbour
// values from the previous colour's launch.  What would be N1 is changing
// WHICH colour a node belongs to, or folding two colours into one launch;
// neither is reachable from this knob.
//
// DEFAULT.  0 means "unchanged": the five classes keep `block_size`
// (kDefaultBlockSize = 256 unless RASBERY_GPU_BLOCK_SIZE says otherwise), so
// an unset environment reproduces the 34/67-block grids block for block.
// RASBERY_GPU_BLOCK_SIZE keeps its meaning -- the width of EVERY per-node
// kernel, the sweep-assembly family included -- and this one narrows the five
// that run once per BiCGSTAB iteration.  32 is reachable here and not there,
// because 32 is the width the 265-block arm needs.
inline int cmfdBlockThreads() {
    static const int threads = [] {
        const char* value = std::getenv("RASBERY_GPU_CMFD_BLOCK");
        if (value == nullptr) return 0;
        const int requested = std::atoi(value);
        if (requested == 0) return 0;
        constexpr int candidates[] = {32, 64, 128, 192, 256};
        if (std::find(std::begin(candidates), std::end(candidates), requested) ==
            std::end(candidates)) {
            std::cerr << "[RASBERY][WARN][cmfd] RASBERY_GPU_CMFD_BLOCK=\"" << value
                      << "\" is not one of 32|64|128|192|256; keeping the block width"
                      << " unchanged\n";
            return 0;
        }
        return requested;
    }();
    return threads;
}

/// RASBERY_GPU_CMFD_PERSISTENT, default OFF.  Arms the cooperative
/// single-launch BiCGSTAB iteration (plan Task 21 / W0 spike 2).  Read once,
/// like every other gate: the arm is part of the enqueue topology and may not
/// change between two outers of the same run.
inline bool cmfdPersistentRequested() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_PERSISTENT");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

// ---------------------------------------------------------------------------
// WHY THE PERSISTENT ARM REFUSES BY NAME.
//
// Modelled on BICGCMFD::EnqueueRefusal (src/BICGCMFD.h): a boolean "it did not
// run" is indistinguishable from "it ran and bought nothing", and the point of
// the spike is to attribute a measured wall to a named cause.  Every
// enumerator has a name, the switch carries no `default:` so a new enumerator
// fails to compile until it is named, and the name is what the
// [RASBERY][CMFD][OCCUPANCY] receipt prints.
// ---------------------------------------------------------------------------
enum class PersistentRefusal : int {
    None = 0,
    ArmOff,              ///< RASBERY_GPU_CMFD_PERSISTENT unset or 0
    OuterGraphActive,    ///< mutually exclusive with the captured outer graph
    CaptureActive,       ///< a capture is open on the stream at launch time
    BatchWidth,          ///< lanes != 1; a grid barrier cannot straddle a halted lane
    NoCooperativeLaunch, ///< cudaDeviceProp::cooperativeLaunch == 0
    OccupancyTooSmall,   ///< the co-resident grid cannot cover the fixed fold
    BlockWidthMismatch,  ///< the persistent block is not kReduceThreads wide
    Fp32Inner,           ///< the mixed-precision inner loop is out of scope
    LaunchFailed,        ///< cudaLaunchCooperativeKernel refused at run time
    Count
};

inline const char* persistentRefusalName(PersistentRefusal r) {
    switch (r) {
        case PersistentRefusal::None:                return "none";
        case PersistentRefusal::ArmOff:              return "arm_off";
        case PersistentRefusal::OuterGraphActive:    return "outer_graph_active";
        case PersistentRefusal::CaptureActive:       return "capture_active";
        case PersistentRefusal::BatchWidth:          return "batch_width";
        case PersistentRefusal::NoCooperativeLaunch: return "no_cooperative_launch";
        case PersistentRefusal::OccupancyTooSmall:   return "occupancy_too_small";
        case PersistentRefusal::BlockWidthMismatch:  return "block_width_mismatch";
        case PersistentRefusal::Fp32Inner:           return "fp32_inner";
        case PersistentRefusal::LaunchFailed:        return "launch_failed";
        case PersistentRefusal::Count:               break;
    }
    return "?";
}

/// One line, the first time a BiCGSTAB iteration is enqueued.  The occupancy
/// receipt: what the launch geometry actually is, how many dispatches one
/// iteration costs, and -- when the persistent arm is off -- the NAME of the
/// reason.  Emitted whatever the graph mode, because a graph-off run has no
/// [CMFD][GRAPH] line to carry the fields.
inline void reportCmfdOccupancy(int block_threads, int sweep_block_threads,
                                int node_blocks, int vector_blocks,
                                int reduce_blocks, int launches_per_iteration,
                                bool persistent_arm, int persistent_blocks,
                                bool cooperative_supported,
                                PersistentRefusal refusal) {
    static std::atomic<bool> done{false};
    if (done.exchange(true, std::memory_order_relaxed)) return;
    std::cout << "[RASBERY][CMFD][OCCUPANCY] {\"block_threads\":" << block_threads
              << ",\"sweep_block_threads\":" << sweep_block_threads
              << ",\"node_blocks\":" << node_blocks
              << ",\"vector_blocks\":" << vector_blocks
              << ",\"reduce_blocks\":" << reduce_blocks
              << ",\"scalar_blocks\":1"
              << ",\"launches_per_iteration\":" << launches_per_iteration
              << ",\"persistent_arm\":" << (persistent_arm ? 1 : 0)
              << ",\"persistent_blocks\":" << persistent_blocks
              << ",\"cooperative_supported\":" << (cooperative_supported ? 1 : 0)
              << ",\"persistent_refusal\":\"" << persistentRefusalName(refusal)
              << "\"}" << std::endl;
}

std::atomic<unsigned long long> g_cmfd_logical_drives{0};
std::atomic<unsigned long long> g_cmfd_physical_blocks{0};
std::atomic<unsigned long long> g_cmfd_padding_blocks{0};
std::atomic<unsigned long long> g_cmfd_bucket_graphs{0};
/// Nine buckets, indexed by the ladder position (1,2,4,8,16,24,32,48,64).
std::array<std::atomic<unsigned long long>, 9> g_cmfd_bucket_histogram{};

inline void cmfdBucketHistogramBump(int lanes) {
    g_cmfd_bucket_histogram[static_cast<std::size_t>(cmfdBucketIndex(lanes))].fetch_add(
        1, std::memory_order_relaxed);
}

/// Flux-mirror cost, kept OUTSIDE BackendCounters because the two places that
/// pay it -- stageSlot() and adoptFluxMirror() -- both run UNLOCKED on the
/// owning instance's thread, by design ("No CUDA call: safe to run
/// concurrently on every instance thread").  Every other counter in that
/// struct is bumped on the launcher, which is single-threaded by the
/// rendezvous; these are not, so they are atomics folded in by counters().
std::atomic<unsigned long long> g_cmfd_phi_mirror_ns{0};
std::atomic<unsigned long long> g_cmfd_phi_mirror_calls{0};
std::atomic<unsigned long long> g_cmfd_phi_mirror_bypassed{0};
std::atomic<unsigned long long> g_cmfd_phi_h2d_elided_bytes{0};

/// The one place the mirror atomics enter a BackendCounters snapshot.
inline BackendCounters withPhiMirrorCounters(BackendCounters c) {
    c.cmfd_phi_mirror_ns = g_cmfd_phi_mirror_ns.load(std::memory_order_relaxed);
    c.cmfd_phi_mirror_calls = g_cmfd_phi_mirror_calls.load(std::memory_order_relaxed);
    c.cmfd_phi_mirror_bypassed =
        g_cmfd_phi_mirror_bypassed.load(std::memory_order_relaxed);
    c.cmfd_phi_h2d_elided_bytes =
        g_cmfd_phi_h2d_elided_bytes.load(std::memory_order_relaxed);
    return c;
}

constexpr int kReduceThreads   = 256;
constexpr int kMaxReduceBlocks = 256;

__host__ __device__ inline int reduce_blocks_for(const int n) {
    const int per_block = kReduceThreads * 4;
    int blocks = (n + per_block - 1) / per_block;
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxReduceBlocks) blocks = kMaxReduceBlocks;
    return blocks;
}

// ---------------------------------------------------------------------------
// Deterministic dot product.
//
// cuBLAS level-1 reductions (cublasDdot / cublasDnrm2) do not guarantee a
// run-to-run reproducible summation order: the library is free to pick the
// grid shape and the partial-accumulation strategy from internal heuristics,
// so two identical invocations can differ in the last few ulps.  In a
// BiCGSTAB driven by an outer critical search those ulps are amplified into
// visible k_eff scatter.  The kernels below fix the partition (a pure
// function of n), the per-thread traversal order and the reduction tree, so
// the result is bit-identical for every run on the same device.
//
// The batch axis is gridDim.y.  `chunk` below reads gridDim.x only, so a
// batched launch splits every instance exactly as its single-instance launch
// did -- this one line is what the whole bit-identity argument rests on.
// ---------------------------------------------------------------------------

__global__ void reduce_dot_stage1(const int n,
                                  const long long vec_stride,
                                  const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double* __restrict__ partial,
                                  const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    __shared__ double shared[kReduceThreads];

    const double* am = a + m * vec_stride;
    const double* bm = b + m * vec_stride;
    double*       pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;

    // Fixed, contiguous chunk per block: depends only on (n, gridDim.x).
    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x))
        sum += am[i] * bm[i];

    shared[threadIdx.x] = sum;
    __syncthreads();

    // Fixed binary tree: identical operand pairing on every launch.
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) pm[blockIdx.x] = shared[0];
}

__global__ void reduce_dot_stage2(const int blocks,
                                  const double* __restrict__ partial,
                                  double* __restrict__ scalars,
                                  const int slot,
                                  const bool take_sqrt,
                                  const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const double* pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order
    scalars[static_cast<long long>(m) * kScalarCount + slot] = take_sqrt ? sqrt(sum) : sum;
}

// ---------------------------------------------------------------------------
// RASBERY_GPU_CMFD_FUSE bit 0 (kFuseDot): reduce_dot_stage1 + reduce_dot_stage2
// in ONE graph node.
//
// ORDER-PRESERVATION NOTE -- why this is bit-identical to the two-node pair.
//
//  1. STAGE 1 IS COPIED, NOT REWRITTEN.  The partition (`chunk`, a pure
//     function of n and gridDim.x), the strided per-thread walk, the shared
//     array, the fixed binary tree and the `sum += am[i] * bm[i]` expression
//     are character for character reduce_dot_stage1's.  Same expression, same
//     translation unit, same flags -- so it contracts to the same FMA and even
//     the rounding of the multiply-add is the same.
//  2. STAGE 2 IS COPIED, NOT REWRITTEN.  The fold is still
//     `for (i = 0; i < blocks; ++i) sum += pm[i]` -- strict ascending index
//     order over the SAME partial row -- run by ONE thread, and it still ends
//     in the same `take_sqrt ? sqrt(sum) : sum`.  WHICH block runs it is
//     irrelevant to the result: the fold reads pm[0..blocks-1], not "its own"
//     partials, so no operand pairing and no addition order moves.
//  3. THE BARRIER BETWEEN THEM IS STILL A BARRIER.  In the two-node form the
//     kernel boundary guaranteed every partial was written before the fold read
//     it.  Here that guarantee is the retire counter: thread 0 of each block
//     publishes its partial, issues __threadfence() so the write is visible
//     device-wide, and only then joins the count.  The fold runs in the block
//     that draws the LAST ticket, so by construction all gridDim.x partials are
//     written and fenced before it starts.  The read-back goes through a
//     volatile pointer so the compiler cannot serve pm[blockIdx.x] out of a
//     register this block still holds.
//  4. `blocks` IS gridDim.x.  The two-node form passed the host's `blocks` to
//     stage 2 and used that same variable as grid.x for stage 1, so reading
//     gridDim.x here is the identical number by construction.
//  5. THE GUARDS ARE THE SAME GUARDS, AND THEY CANNOT DISAGREE.  Both reference
//     kernels open with RASBERY_CMFD_SLOT + HALT_GUARD, and NO kernel runs
//     between them, so `halt[m]` cannot change across the pair: either both ran
//     or neither did.  Fused, a halted slot returns before any block touches
//     the counter, so the fold never runs and the scalar is never written --
//     which is exactly what stage 2's own HALT_GUARD did.
//  6. THE COUNTER SELF-REARMS.  atomicInc(retire + m, gridDim.x - 1) wraps the
//     last ticket back to 0, so the array is 0 on entry to every fused
//     reduction with no separate memset node.  It is zeroed once at allocation,
//     and a halted launch leaves it untouched at 0.  ONE array serves every
//     fused kernel because they are STREAM-ORDERED -- a captured graph replays
//     a linear chain, so no two of them are ever in flight.
//
// Saving: one node per dot().  Two per BiCGSTAB iteration (rho_new and r0.v).
// ---------------------------------------------------------------------------
__global__ void reduce_dot_fused(const int n,
                                 const long long vec_stride,
                                 const double* __restrict__ a,
                                 const double* __restrict__ b,
                                 double* partial,
                                 double* __restrict__ scalars,
                                 const int slot,
                                 const bool take_sqrt,
                                 unsigned int* __restrict__ retire,
                                 const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    __shared__ double shared[kReduceThreads];

    const double* am = a + m * vec_stride;
    const double* bm = b + m * vec_stride;
    double*       pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;

    // Fixed, contiguous chunk per block: depends only on (n, gridDim.x).
    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x))
        sum += am[i] * bm[i];

    shared[threadIdx.x] = sum;
    __syncthreads();

    // Fixed binary tree: identical operand pairing on every launch.
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x != 0) return;
    pm[blockIdx.x] = shared[0];
    __threadfence();
    if (atomicInc(retire + m, gridDim.x - 1u) != gridDim.x - 1u) return;

    // ---- the former reduce_dot_stage2 node, verbatim ----------------------
    const volatile double* vpm    = pm;
    const int              blocks = static_cast<int>(gridDim.x);
    double                 fold   = 0.0;
    for (int i = 0; i < blocks; ++i) fold += vpm[i];   // strict index order
    scalars[static_cast<long long>(m) * kScalarCount + slot] =
        take_sqrt ? sqrt(fold) : fold;
}

/// Strict stage-2 fold plus the immediately dependent r20 snapshot.
/// This removes one scalar graph node without changing the reduction tree or
/// the liveness/participation guard used by the former two-kernel sequence.
__global__ void reduce_norm_store_reference_stage2(
    const int blocks,
    const double* __restrict__ partial,
    double* scalars,
    const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const double* pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += pm[i];
    const double norm = sqrt(sum);
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kInitialNorm] = norm;
    sm[kR20] = norm;
}

// ---------------------------------------------------------------------------
// TWO reductions in one pair of nodes.
//
// This is NOT a change to any reduction: it is two INDEPENDENT dot products
// evaluated side by side in one kernel.  Each keeps its own accumulator, its
// own shared array, the same `chunk` partition (a pure function of n and
// gridDim.x, untouched below), the same per-thread traversal order, the same
// fixed binary tree and the same strict index-order stage-2 fold.  Nothing is
// re-associated and no operand pairing moves, so both results are bit-for-bit
// what the two separate launches produced -- the only thing that disappears
// is two graph nodes' worth of dispatch.
//
// Used for the (s.t, t.t) pair, the only two adjacent dots in a BiCGSTAB
// iteration with no kernel between them.
// ---------------------------------------------------------------------------
__global__ void reduce_dot2_stage1(const int n,
                                   const long long vec_stride,
                                   const double* __restrict__ a0,
                                   const double* __restrict__ b0,
                                   const double* __restrict__ a1,
                                   const double* __restrict__ b1,
                                   double* __restrict__ partial0,
                                   double* __restrict__ partial1,
                                   const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    __shared__ double shared0[kReduceThreads];
    __shared__ double shared1[kReduceThreads];

    const double* a0m = a0 + m * vec_stride;
    const double* b0m = b0 + m * vec_stride;
    const double* a1m = a1 + m * vec_stride;
    const double* b1m = b1 + m * vec_stride;
    double*       p0m = partial0 + static_cast<long long>(m) * kMaxReduceBlocks;
    double*       p1m = partial1 + static_cast<long long>(m) * kMaxReduceBlocks;

    // Identical to reduce_dot_stage1: depends only on (n, gridDim.x).
    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        sum0 += a0m[i] * b0m[i];
        sum1 += a1m[i] * b1m[i];
    }

    shared0[threadIdx.x] = sum0;
    shared1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared0[threadIdx.x] += shared0[threadIdx.x + stride];
            shared1[threadIdx.x] += shared1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        p0m[blockIdx.x] = shared0[0];
        p1m[blockIdx.x] = shared1[0];
    }
}

__global__ void reduce_dot2_stage2(const int blocks,
                                   const double* __restrict__ partial0,
                                   const double* __restrict__ partial1,
                                   double* __restrict__ scalars,
                                   const int slot0,
                                   const int slot1,
                                   const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const double* p0m = partial0 + static_cast<long long>(m) * kMaxReduceBlocks;
    const double* p1m = partial1 + static_cast<long long>(m) * kMaxReduceBlocks;
    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = 0; i < blocks; ++i) {   // strict index order, one per reduction
        sum0 += p0m[i];
        sum1 += p1m[i];
    }
    scalars[static_cast<long long>(m) * kScalarCount + slot0] = sum0;
    scalars[static_cast<long long>(m) * kScalarCount + slot1] = sum1;
}

// ---------------------------------------------------------------------------
// RASBERY_GPU_CMFD_FUSE bit 1 (kFuseDot2): reduce_dot2_stage1 +
// reduce_dot2_stage2 in ONE graph node.
//
// ORDER-PRESERVATION NOTE.  Points 1-6 of reduce_dot_fused apply unchanged and
// are not repeated; the only difference is that TWO independent reductions ride
// through the fused kernel exactly as they rode through the pair -- each with
// its own accumulator, its own shared array, its own partial row and its own
// strict ascending stage-2 fold.  Nothing is re-associated across the two, and
// the retire counter is shared because the two reductions retire together (they
// are the same blocks of the same launch), not because their sums are.
//
// Saving: one node per dot2().  One per BiCGSTAB iteration -- the (s.t, t.t)
// pair.
// ---------------------------------------------------------------------------
__global__ void reduce_dot2_fused(const int n,
                                  const long long vec_stride,
                                  const double* __restrict__ a0,
                                  const double* __restrict__ b0,
                                  const double* __restrict__ a1,
                                  const double* __restrict__ b1,
                                  double* partial0,
                                  double* partial1,
                                  double* __restrict__ scalars,
                                  const int slot0,
                                  const int slot1,
                                  unsigned int* __restrict__ retire,
                                  const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    __shared__ double shared0[kReduceThreads];
    __shared__ double shared1[kReduceThreads];

    const double* a0m = a0 + m * vec_stride;
    const double* b0m = b0 + m * vec_stride;
    const double* a1m = a1 + m * vec_stride;
    const double* b1m = b1 + m * vec_stride;
    double*       p0m = partial0 + static_cast<long long>(m) * kMaxReduceBlocks;
    double*       p1m = partial1 + static_cast<long long>(m) * kMaxReduceBlocks;

    // Identical to reduce_dot2_stage1: depends only on (n, gridDim.x).
    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        sum0 += a0m[i] * b0m[i];
        sum1 += a1m[i] * b1m[i];
    }

    shared0[threadIdx.x] = sum0;
    shared1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared0[threadIdx.x] += shared0[threadIdx.x + stride];
            shared1[threadIdx.x] += shared1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) return;
    p0m[blockIdx.x] = shared0[0];
    p1m[blockIdx.x] = shared1[0];
    __threadfence();
    if (atomicInc(retire + m, gridDim.x - 1u) != gridDim.x - 1u) return;

    // ---- the former reduce_dot2_stage2 node, verbatim ---------------------
    const volatile double* v0m    = p0m;
    const volatile double* v1m    = p1m;
    const int              blocks = static_cast<int>(gridDim.x);
    double                 fold0  = 0.0;
    double                 fold1  = 0.0;
    for (int i = 0; i < blocks; ++i) {   // strict index order, one per reduction
        fold0 += v0m[i];
        fold1 += v1m[i];
    }
    scalars[static_cast<long long>(m) * kScalarCount + slot0] = fold0;
    scalars[static_cast<long long>(m) * kScalarCount + slot1] = fold1;
}

__global__ void matvec_two_group(const int nxyz,
                                 const long long vec_stride,
                                 const long long mat_stride,
                                 const long long cpl_stride,
                                 const int* __restrict__ neighbors,
                                 const double* __restrict__ diag,
                                 const double* __restrict__ cc,
                                 const double* __restrict__ x,
                                 double* __restrict__ y,
                                 const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double* dm = diag + m * mat_stride;
    const double* cm = cc + m * cpl_stride;
    const double* xm = x + m * vec_stride;
    double*       ym = y + m * vec_stride;

    const double x0 = xm[2 * l + 0];
    const double x1 = xm[2 * l + 1];
    double       y0 = dm[4 * l + 0] * x0 + dm[4 * l + 1] * x1;
    double       y1 = dm[4 * l + 2] * x0 + dm[4 * l + 3] * x1;

#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = neighbors[6 * l + slot];
        if (neighbor >= 0) {
            y0 += cm[12 * l + slot] * xm[2 * neighbor + 0];
            y1 += cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    ym[2 * l + 0] = y0;
    ym[2 * l + 1] = y1;
}

// ---------------------------------------------------------------------------
// FUSED: invert_two_group_blocks + matvec_two_group(phi -> ax) + initial_residual.
//
// Three former nodes, one index domain.  The vector-domain step folds in
// because the two-group layout is node-major: element i = 2*l + ig belongs to
// node l, so thread l of the node grid owns exactly the two elements the
// vector grid gave two threads.  n == ng*nxyz == 2*nxyz is checked once in the
// constructor, so the coverage is exact.
//
// Why no intermediate has another consumer:
//   * dinv   -- written here, first read by the colour sweeps, i.e. after the
//               next kernel boundary.  The block inversion is INDEPENDENT of
//               the other two steps (it reads diag, writes dinv, and neither
//               of the others touches dinv), so concatenating it is a pure
//               node saving with no ordering question at all.
//   * ax     -- written by the matvec, read only by the residual, and read
//               only at the SAME node the writing thread owns.  No thread
//               reads another thread's ax, so no grid-wide ordering is needed
//               and the register value is the stored value.  ax is still
//               written to memory: the sweep path reuses it as scratch.
//
// Bit-identity: every expression below is copied verbatim from the three
// kernels it replaces, so nvcc makes the same contraction decisions on the
// same operands (this TU compiles with the default --fmad, which is exactly
// why the text must not drift).  `src - y0` substitutes the register for a
// load of the double just stored there -- the same bits by definition.
//
// Gating: one HALT_GUARD stands in for the three identical ones.
// ---------------------------------------------------------------------------
__global__ void begin_outer_fused(const int nxyz,
                                  const long long vec_stride,
                                  const long long mat_stride,
                                  const long long cpl_stride,
                                  const int* __restrict__ neighbors,
                                  const double* __restrict__ diag,
                                  const double* __restrict__ cc,
                                  const double* __restrict__ x,
                                  const double* __restrict__ src,
                                  double* __restrict__ dinv,
                                  double* __restrict__ ax,
                                  double* __restrict__ r,
                                  double* __restrict__ r0,
                                  double* __restrict__ p,
                                  double* __restrict__ v,
                                  const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double* dm = diag + m * mat_stride;

    // ---- invert_two_group_blocks ----
    {
        double* im = dinv + m * mat_stride;

        const double a00  = dm[4 * l + 0];
        const double a01  = dm[4 * l + 1];
        const double a10  = dm[4 * l + 2];
        const double a11  = dm[4 * l + 3];
        const double rdet = 1.0 / (a00 * a11 - a10 * a01);

        im[4 * l + 0] = rdet * a11;
        im[4 * l + 1] = -rdet * a01;
        im[4 * l + 2] = -rdet * a10;
        im[4 * l + 3] = rdet * a00;
    }

    // ---- matvec_two_group(x = phi -> y = ax) ----
    const double* cm = cc + m * cpl_stride;
    const double* xm = x + m * vec_stride;
    double*       ym = ax + m * vec_stride;

    const double x0 = xm[2 * l + 0];
    const double x1 = xm[2 * l + 1];
    double       y0 = dm[4 * l + 0] * x0 + dm[4 * l + 1] * x1;
    double       y1 = dm[4 * l + 2] * x0 + dm[4 * l + 3] * x1;

#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = neighbors[6 * l + slot];
        if (neighbor >= 0) {
            y0 += cm[12 * l + slot] * xm[2 * neighbor + 0];
            y1 += cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    ym[2 * l + 0] = y0;
    ym[2 * l + 1] = y1;

    // ---- initial_residual over this node's two elements ----
    const long long base   = m * vec_stride;
    const double    value0 = src[base + 2 * l + 0] - y0;
    const double    value1 = src[base + 2 * l + 1] - y1;
    r[base + 2 * l + 0]    = value0;
    r0[base + 2 * l + 0]   = value0;
    p[base + 2 * l + 0]    = 0.0;
    v[base + 2 * l + 0]    = 0.0;
    r[base + 2 * l + 1]    = value1;
    r0[base + 2 * l + 1]   = value1;
    p[base + 2 * l + 1]    = 0.0;
    v[base + 2 * l + 1]    = 0.0;
}

__global__ void initialize_solver_state(double* scalars,
                                        std::uint32_t* flags,
                                        std::uint32_t* halt,
                                        std::uint32_t* counters,
                                        std::uint32_t* iter_flags,
                                        const std::uint32_t* __restrict__ active,
                                        const std::uint32_t* __restrict__ sweep_halt) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);

    // Half of what the per-iteration cudaMemsetAsync(iter_flags) node used to
    // do; the other half is the re-zero at the end of accumulate_iteration.
    // Written for EVERY slot, participating or not, before the mask below --
    // so this is byte for byte the memset's effect at the top of the outer.
    iter_flags[m] = 0u;

    // The one place the participation mask enters the device: every later
    // kernel only ever consults `halt`, exactly as it did before batching.
    // A raised sweep_halt (device-resident CMFD sweeps that converged, ran
    // out of budget, or handed back to the host) masks the slot the same way
    // non-participation does; it is all-zero outside the sweep path.
    halt[m] = (active[m] != 0u && sweep_halt[m] == 0u) ? 0u : 1u;
    if (halt[m] != 0u) return;

    double*        sm = scalars + static_cast<long long>(m) * kScalarCount;
    std::uint32_t* cm = counters + static_cast<long long>(m) * kCounterSlots;
    sm[kRhoNew]      = 1.0;
    sm[kR0V]         = 0.0;
    sm[kPts]         = 0.0;
    sm[kPtt]         = 0.0;
    sm[kRho]         = 1.0;
    sm[kAlpha]       = 1.0;
    sm[kOmega]       = 1.0;
    sm[kInitialNorm] = 0.0;
    flags[m] = 0;
    for (int i = 0; i < kCounterSlots; ++i) cm[i] = 0;
}

/// Freeze the reference residual for this outer.  The host used to hold it in
/// a local `r20`; it is now the fixed denominator of the device-side test.
__global__ void store_reference_norm(double* scalars,
                                     const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kR20]   = sm[kInitialNorm];
}

// ---------------------------------------------------------------------------
// FUSED: prepare_p + block_jacobi(b = p, x = y).
//
// block_jacobi is the DIAGONAL solve that opens the preconditioner chain: it
// reads b only at its own node (bm[2l+0], bm[2l+1]) and dinv only at its own
// node.  It therefore has no cross-thread dependency on its producer, which is
// what makes this fusion legal where fusing the colour sweeps is not -- those
// read x at NEIGHBOURING nodes and need the grid-wide barrier that a kernel
// boundary provides.
//
// p has no other consumer between the two: the colour sweeps that follow read
// b = p at their own node only (again no neighbour read of b), and the next
// iteration's prepare_p reads the stored p, which this still writes.
//
// The arithmetic per element and per node is verbatim the two originals, so
// the contraction pattern nvcc picks is unchanged; b0/b1 are the registers
// holding the doubles just stored to p, i.e. the same bits block_jacobi used
// to load back.  The atomicOr is idempotent, so folding n element-threads into
// nxyz node-threads leaves `flags` at the same value.
// ---------------------------------------------------------------------------
__global__ void prepare_p_jacobi(const int nxyz,
                                 const long long vec_stride,
                                 const long long mat_stride,
                                 double* scalars,
                                 std::uint32_t* flags,
                                 const double* __restrict__ dinv,
                                 const double* __restrict__ r,
                                 const double* __restrict__ v,
                                 double* __restrict__ p,
                                 double* __restrict__ y,
                                 const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double*  sm   = scalars + static_cast<long long>(m) * kScalarCount;
    const long long base = m * vec_stride;
    const int       i0   = 2 * l + 0;
    const int       i1   = 2 * l + 1;

    const double rho_new = sm[kRhoNew];
    const double rho_old = sm[kRho];
    const double alpha   = sm[kAlpha];
    const double omega   = sm[kOmega];
    const double denom   = rho_old * omega;
    const bool breakdown = !isfinite(rho_new) || !isfinite(denom) ||
                           fabs(denom) < 1.0e-30;
    double b0, b1;
    if (breakdown) {
        // Restart convention (matches what the CPU reference effectively does
        // by carrying on with a degenerate beta): drop the Krylov direction
        // and restart from p = r.  update_s / update_solution then rebuild
        // rho, alpha and omega from scratch, so the state self-heals.  The
        // flag is reported for telemetry, it is no longer fatal.
        atomicOr(flags + m, static_cast<std::uint32_t>(BICGSTAB_BREAKDOWN));
        b0 = r[base + i0];
        b1 = r[base + i1];
    } else {
        const double beta = rho_new * alpha / denom;
        b0 = r[base + i0] + beta * (p[base + i0] - omega * v[base + i0]);
        b1 = r[base + i1] + beta * (p[base + i1] - omega * v[base + i1]);
    }
    p[base + i0] = b0;
    p[base + i1] = b1;

    // ---- block_jacobi(b = p, x = y) ----
    const double* im = dinv + m * mat_stride;
    double*       xm = y + m * vec_stride;

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

__global__ void colored_block_sweep(const int nxyz,
                                    const long long vec_stride,
                                    const long long mat_stride,
                                    const long long cpl_stride,
                                    const int target_color,
                                    const int* __restrict__ colors,
                                    const int* __restrict__ neighbors,
                                    const double* __restrict__ cc,
                                    const double* __restrict__ dinv,
                                    const double* __restrict__ b,
                                    double* __restrict__ x,
                                    const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz || colors[l] != target_color) return;

    const double* cm = cc + m * cpl_stride;
    const double* im = dinv + m * mat_stride;
    const double* bm = b + m * vec_stride;
    double*       xm = x + m * vec_stride;

    double b0 = bm[2 * l + 0];
    double b1 = bm[2 * l + 1];
#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = neighbors[6 * l + slot];
        if (neighbor >= 0) {
            b0 -= cm[12 * l + slot] * xm[2 * neighbor + 0];
            b1 -= cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

// ---------------------------------------------------------------------------
// FUSED: update_s + block_jacobi(b = s, x = z).  Same argument as
// prepare_p_jacobi: the diagonal solve reads only its own node's b.
//
// s has no other consumer before the next kernel boundary; afterwards the
// colour sweeps (own-node b), the (s.t) dot and update_solution all read the
// stored s, which this still writes.
//
// The three exits of the original are branches here rather than returns,
// because the diagonal solve ran unconditionally as its own kernel and must
// keep doing so.  Each branch computes the SAME expression the original ran
// for that element, and `i == 0` becomes `l == 0` restricted to the first of
// the node's two elements -- which is what element i = 0 was.
// ---------------------------------------------------------------------------
__global__ void update_s_jacobi(const int nxyz,
                                const long long vec_stride,
                                const long long mat_stride,
                                double* scalars,
                                std::uint32_t* flags,
                                const double* __restrict__ dinv,
                                const double* __restrict__ r,
                                const double* __restrict__ v,
                                double* __restrict__ s,
                                double* __restrict__ z,
                                const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    double*         sm   = scalars + static_cast<long long>(m) * kScalarCount;
    const long long base = m * vec_stride;
    const int       i0   = 2 * l + 0;
    const int       i1   = 2 * l + 1;

    // A BICGSTAB_BREAKDOWN flag means prepare_p restarted with p = r; the
    // rest of the step is still well defined and must run so that rho, alpha
    // and omega are re-established.  Bailing out here used to leave the
    // solver in the same degenerate state on every subsequent call.
    const double rho = sm[kRhoNew];
    const double r0v = sm[kR0V];
    double b0, b1;
    if (!isfinite(rho) || !isfinite(r0v)) {
        atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        b0 = r[base + i0];
        b1 = r[base + i1];
    } else {
        // rho is committed unconditionally, exactly like the CPU reference
        // (BICGSolver::solve assigns _crho before the |r0.v| test).  Leaving it
        // stale here poisons beta = rho_new*alpha/(rho_old*omega) on the next
        // call.  alpha stays untouched on the early exit, again as on the CPU.
        if (l == 0) sm[kRho] = rho;

        // The legacy CPU solver treats a finite, near-orthogonal r0.v as a
        // successful no-op. Preserve that convergence behavior explicitly
        // instead of misclassifying it as a fatal BiCGSTAB breakdown.
        if (fabs(r0v) < 1.0e-10) {
            atomicOr(flags + m, static_cast<std::uint32_t>(FLUX_CONVERGED));
            b0 = r[base + i0];
            b1 = r[base + i1];
        } else {
            const double alpha = rho / r0v;
            b0 = r[base + i0] - alpha * v[base + i0];
            b1 = r[base + i1] - alpha * v[base + i1];
            if (l == 0) sm[kAlpha] = alpha;
        }
    }
    s[base + i0] = b0;
    s[base + i1] = b1;

    // ---- block_jacobi(b = s, x = z) ----
    const double* im = dinv + m * mat_stride;
    double*       xm = z + m * vec_stride;

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

__global__ void update_solution(const int n,
                                const long long vec_stride,
                                double* scalars,
                                std::uint32_t* flags,
                                const double* __restrict__ y,
                                const double* __restrict__ z,
                                const double* __restrict__ s,
                                const double* __restrict__ t,
                                double* __restrict__ phi,
                                double* __restrict__ r,
                                const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if ((flags[m] & static_cast<std::uint32_t>(FLUX_CONVERGED)) != 0) return;

    double*         sm   = scalars + static_cast<long long>(m) * kScalarCount;
    const long long base = m * vec_stride;

    const double alpha = sm[kAlpha];
    const double pts   = sm[kPts];
    const double ptt   = sm[kPtt];
    // Non-finite scalars are a hard failure, not a restartable breakdown.
    if (!isfinite(alpha) || !isfinite(pts) || !isfinite(ptt)) {
        atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        return;
    }

    const double omega = (ptt != 0.0) ? pts / ptt : 0.0;
    const double next_phi = phi[base + i] + alpha * y[base + i] + omega * z[base + i];
    const double next_r   = s[base + i] - omega * t[base + i];
    if (!isfinite(next_phi) || !isfinite(next_r)) {
        atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        return;
    }
    if (next_phi < 0.0)
        atomicOr(flags + m, static_cast<std::uint32_t>(NEGATIVE_FLUX));
    phi[base + i] = next_phi;
    r[base + i]   = next_r;
    if (i == 0) sm[kOmega] = omega;
}

/// Shared end-of-iteration state transition. The caller has already verified
/// that the slot is active and not halted. Both the historical standalone
/// kernel and the fused norm-finalizer call this exact body.
__device__ inline void accumulate_iteration_active(
    const int m,
    const int allow_halt,
    const int force_halt,
    const double* __restrict__ scalars,
    std::uint32_t* iter_flags,
    std::uint32_t* sticky_flags,
    std::uint32_t* counters,
    std::uint32_t* halt) {
    const double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    std::uint32_t* cm = counters + static_cast<long long>(m) * kCounterSlots;

    const std::uint32_t flags = iter_flags[m];
    // Re-arm the scratch flags for the next captured iteration at the same
    // stream-ordered point as the retired cudaMemsetAsync node.
    iter_flags[m] = 0u;
    const double ptt = sm[kPtt];
    const double rnorm = sm[kInitialNorm];
    const bool corrupt = !isfinite(ptt) || ptt < 0.0 || !isfinite(rnorm);

    sticky_flags[m] |= flags;
    if (corrupt) sticky_flags[m] |= static_cast<std::uint32_t>(NONFINITE_DETECTED);
    if ((flags & static_cast<std::uint32_t>(BICGSTAB_BREAKDOWN)) != 0)
        ++cm[kRestartCount];
    if ((flags & static_cast<std::uint32_t>(FLUX_CONVERGED)) != 0)
        ++cm[kEarlyExitCount];
    ++cm[kSolveCount];

    if (corrupt || (sticky_flags[m] & static_cast<std::uint32_t>(NONFINITE_DETECTED)) != 0) {
        halt[m] = 1u;
        return;
    }
    if (allow_halt != 0) {
        const double r20 = sm[kR20];
        if (r20 <= 0.0 || rnorm / r20 < sm[kEps]) halt[m] = 1u;
    }
    if (force_halt != 0) halt[m] = 1u;
}

/// End-of-iteration bookkeeping retained as the runtime rollback path.
__global__ void accumulate_iteration(const int allow_halt,
                                     const int force_halt,
                                     const double* __restrict__ scalars,
                                     std::uint32_t* iter_flags,
                                     std::uint32_t* sticky_flags,
                                     std::uint32_t* counters,
                                     std::uint32_t* halt,
                                     const std::uint32_t* __restrict__ active,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    if (active[m] == 0u) return;

    std::uint32_t* cm = counters + static_cast<long long>(m) * kCounterSlots;
    if (halt[m] != 0u) {
        ++cm[kOverrunCount];
        return;
    }
    accumulate_iteration_active(m, allow_halt, force_halt, scalars, iter_flags,
                                sticky_flags, counters, halt);
}

/// Strict residual-norm stage 2 fused with accumulate_iteration. The early
/// active/halt test is intentionally before the partial fold: an overrun
/// iteration never wrote new partials and must only increment its telemetry.
__global__ void reduce_norm_accumulate_stage2(
    const int blocks,
    const int allow_halt,
    const int force_halt,
    const double* __restrict__ partial,
    double* scalars,
    std::uint32_t* iter_flags,
    std::uint32_t* sticky_flags,
    std::uint32_t* counters,
    std::uint32_t* halt,
    const std::uint32_t* __restrict__ active,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    if (active[m] == 0u) return;

    std::uint32_t* cm = counters + static_cast<long long>(m) * kCounterSlots;
    if (halt[m] != 0u) {
        ++cm[kOverrunCount];
        return;
    }

    const double* pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += pm[i];
    const double norm = sqrt(sum);
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kInitialNorm] = norm;

    accumulate_iteration_active(m, allow_halt, force_halt, scalars, iter_flags,
                                sticky_flags, counters, halt);
}

// ---------------------------------------------------------------------------
// WP17 / plan Task 21, W0 spike 2: THE PERSISTENT COOPERATIVE BiCG ITERATION
// (RASBERY_GPU_CMFD_PERSISTENT=1, default OFF).
//
// WHAT IT IS FOR.  One BiCGSTAB iteration is 18 dispatches at FUSE=15
// (2 dots + 1 dot2 + 2 x 4 colour sweeps + prepare_p + 2 matvecs + update_s +
// update_solution + the residual stage 1 + the fused scalar tail).  On 238 the
// dispatch floor is ~2.5 us and 74,508 iterations ran, so the pure launch floor
// is ~1.5 s in single and ~64x that summed over an 8-client batch -- paid on a
// GPU whose SM array these 34- and 67-block grids leave 82 % empty.  This
// kernel replaces the 18 dispatches with ONE cooperative launch and 17
// cg::grid_group::sync() barriers.  The gain is exactly
//     N_node * c_dispatch - N_barrier * c_barrier
// which is why tools/probe_gridsync_cost.cu (spike 2/5) has to produce
// c_barrier before this arm may be adopted, and why this is a SPIKE: it is
// written, gated, refused by name and receipted, and it is off.
//
// ORDER-PRESERVATION NOTE -- why the arm is B0.
//
//  1. EVERY STAGE BODY IS COPIED, NOT REWRITTEN.  The expressions below are
//     character for character those of prepare_p_jacobi, colored_block_sweep,
//     matvec_two_group, update_s_jacobi, update_solution, reduce_dot_stage1,
//     reduce_dot2_stage1 and the strict stage-2 folds.  Same translation unit,
//     same --fmad=false, so the same contraction on the same operands.
//  2. THE PER-NODE STAGES ARE GRID-STRIDED, WHICH IS THE SAME ARGUMENT THE
//     BLOCK-WIDTH KNOB ABOVE MAKES.  They are elementwise in the index: the
//     grid-stride loop changes WHICH thread owns node l and covers exactly the
//     same set of l, with no __shared__, no __syncthreads and no cross-thread
//     pairing.  The atomicOr flag writes are commutative; the `l == 0` and
//     `i == 0` scalar stores are keyed to the index, so exactly one thread
//     still performs them.
//  3. THE REDUCTION PARTITION IS PINNED TO reduce_blocks, NOT TO THE GRID.
//     Stage 1 runs on blocks [0, reduce_blocks) with
//     `chunk = (n + reduce_blocks - 1) / reduce_blocks` -- the host passes
//     reduce_blocks_for(n), the same 17 the standalone launch used as its
//     gridDim.x -- the same `begin = blockIdx.x * chunk`, the same per-thread
//     stride of blockDim.x (pinned to kReduceThreads, see the refusal
//     BlockWidthMismatch) and the same fixed 256-lane binary tree.  Every
//     partial is therefore the SAME double the standalone kernel wrote.
//  4. THE FOLD IS THE SAME SERIAL FOLD, RUN REDUNDANTLY.  After the barrier
//     every thread folds pm[0 .. reduce_blocks-1] in strict ascending index
//     order into a register.  Strict order over identical operands is a
//     deterministic function, so every thread gets the same bits -- the bits
//     reduce_dot_stage2's one thread produced -- and no thread has to wait for
//     another to publish them.  Block 0 thread 0 additionally STORES the value
//     into scalars[], which is what the next iteration and the telemetry read.
//     This is the only structural difference from the launch chain and it
//     costs one barrier per reduction rather than two; it moves no addition.
//  5. THE BARRIERS ARE THE KERNEL BOUNDARIES.  Each grid.sync() stands where a
//     kernel boundary stood, one for one, and grid.sync() carries a device-wide
//     memory fence, which is the guarantee the boundary gave.
//  6. EVERY EARLY RETURN IS GRID-UNIFORM.  The slot guard and HALT_GUARD are
//     the only returns before a barrier, and the arm refuses lanes != 1
//     (PersistentRefusal::BatchWidth) precisely so that both are the same
//     answer in every block; a per-lane return would strand the rest of the
//     grid at the next barrier for ever.  Everything after the first barrier
//     is a branch, never a return.
//
// MUTUALLY EXCLUSIVE WITH THE CAPTURED GRAPH.  A cooperative launch cannot be
// recorded into a stream capture, so PERSISTENT is refused whenever the outer
// graph is armed (OuterGraphActive) and again, defensively, whenever a capture
// is actually open on the stream at launch time (CaptureActive).  The 238
// runbook therefore pairs RASBERY_GPU_CMFD_PERSISTENT=1 with
// RASBERY_GPU_GRAPH=0 / OUTER_GRAPH=0; see
// docs/WP17_CMFD_OCCUPANCY_20260830_KO.md Sec 5.
// ---------------------------------------------------------------------------

/// Everything one persistent iteration needs, in one by-value argument so the
/// cudaLaunchCooperativeKernel argument vector is a single pointer.
struct PersistentBicgParams {
    int       nxyz;
    int       n;
    int       reduce_blocks;   ///< reduce_blocks_for(n): the FIXED fold partition
    int       rb_sweeps;
    int       ncolors;
    int       allow_halt;
    int       force_halt;
    long long vec_stride;
    long long mat_stride;
    long long cpl_stride;
    const int*    colors;
    const int*    neighbors;
    const double* cc;
    const double* diag;
    const double* dinv;
    double*       scalars;
    std::uint32_t* iter_flags;
    std::uint32_t* sticky_flags;
    std::uint32_t* counters;
    std::uint32_t* halt;
    const std::uint32_t* active;
    const double* r0;
    double*       r;
    double*       p;
    double*       v;
    double*       s;
    double*       t;
    double*       y;
    double*       z;
    double*       phi;
    double*       partials;
    double*       partials2;
};

/// reduce_dot_stage1's body, with gridDim.x replaced by the pinned partition.
/// Blocks outside the partition simply do not participate; they return to the
/// caller, which then joins the same grid barrier.
__device__ inline void persistentDotStage1(const int n, const int reduce_blocks,
                                           const double* __restrict__ am,
                                           const double* __restrict__ bm,
                                           double* __restrict__ pm,
                                           double* shared) {
    if (static_cast<int>(blockIdx.x) >= reduce_blocks) return;
    const int chunk = (n + reduce_blocks - 1) / reduce_blocks;
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x))
        sum += am[i] * bm[i];

    shared[threadIdx.x] = sum;
    __syncthreads();

    // Fixed binary tree: identical operand pairing on every launch.
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) pm[blockIdx.x] = shared[0];
}

/// reduce_dot2_stage1's body, same partition, two independent accumulators.
__device__ inline void persistentDot2Stage1(const int n, const int reduce_blocks,
                                            const double* __restrict__ a0m,
                                            const double* __restrict__ b0m,
                                            const double* __restrict__ a1m,
                                            const double* __restrict__ b1m,
                                            double* __restrict__ p0m,
                                            double* __restrict__ p1m,
                                            double* shared0, double* shared1) {
    if (static_cast<int>(blockIdx.x) >= reduce_blocks) return;
    const int chunk = (n + reduce_blocks - 1) / reduce_blocks;
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        sum0 += a0m[i] * b0m[i];
        sum1 += a1m[i] * b1m[i];
    }

    shared0[threadIdx.x] = sum0;
    shared1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared0[threadIdx.x] += shared0[threadIdx.x + stride];
            shared1[threadIdx.x] += shared1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        p0m[blockIdx.x] = shared0[0];
        p1m[blockIdx.x] = shared1[0];
    }
}

/// reduce_dot_stage2's fold, verbatim: strict ascending index order over the
/// same partial row.  Run by every thread after the barrier (see point 4).
__device__ inline double persistentFold(const int blocks, const double* pm) {
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order
    return sum;
}

/// colored_block_sweep's body for one node, verbatim.
__device__ inline void persistentColourSweepNode(const int l,
                                                 const PersistentBicgParams& a,
                                                 const int m, const int target_color,
                                                 const double* bm, double* xm) {
    if (a.colors[l] != target_color) return;
    const double* cm = a.cc + m * a.cpl_stride;
    const double* im = a.dinv + m * a.mat_stride;

    double b0 = bm[2 * l + 0];
    double b1 = bm[2 * l + 1];
#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = a.neighbors[6 * l + slot];
        if (neighbor >= 0) {
            b0 -= cm[12 * l + slot] * xm[2 * neighbor + 0];
            b1 -= cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

/// matvec_two_group's body for one node, verbatim.
__device__ inline void persistentMatvecNode(const int l,
                                            const PersistentBicgParams& a,
                                            const int m,
                                            const double* xm, double* ym) {
    const double* dm = a.diag + m * a.mat_stride;
    const double* cm = a.cc + m * a.cpl_stride;

    const double x0 = xm[2 * l + 0];
    const double x1 = xm[2 * l + 1];
    double       y0 = dm[4 * l + 0] * x0 + dm[4 * l + 1] * x1;
    double       y1 = dm[4 * l + 2] * x0 + dm[4 * l + 3] * x1;

#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = a.neighbors[6 * l + slot];
        if (neighbor >= 0) {
            y0 += cm[12 * l + slot] * xm[2 * neighbor + 0];
            y1 += cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    ym[2 * l + 0] = y0;
    ym[2 * l + 1] = y1;
}

/// ONE BiCGSTAB iteration, one cooperative launch, 17 grid barriers.
///
/// Launched ONLY through BatchCore::enqueuePersistentIteration(), which owns
/// every precondition the body assumes: lanes == 1, blockDim.x ==
/// kReduceThreads, gridDim.x >= reduce_blocks, gridDim.y == 1, FP64 inner, no
/// capture open, cooperativeLaunch supported and the grid co-resident.
__global__ void bicg_iteration_persistent(PersistentBicgParams a,
                                          RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    // The halted slot's ONE observable effect, kept because the launch chain
    // has it: every compute kernel HALT_GUARDs out, and the scalar tail --
    // reduce_norm_accumulate_stage2, or accumulate_iteration with the fusion
    // off -- still bumps kOverrunCount for an ACTIVE slot before returning.
    // Grid-uniform (lanes == 1), so no block is left at a barrier.
    if (a.halt[m] != 0u) {
        if (blockIdx.x == 0u && threadIdx.x == 0u && a.active[m] != 0u)
            ++a.counters[static_cast<long long>(m) * kCounterSlots + kOverrunCount];
        return;
    }
    __shared__ double shared0[kReduceThreads];
    __shared__ double shared1[kReduceThreads];

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const long long base   = m * a.vec_stride;
    double*         sm     = a.scalars + static_cast<long long>(m) * kScalarCount;
    const double*   im     = a.dinv + m * a.mat_stride;
    double*         pm0    = a.partials + static_cast<long long>(m) * kMaxReduceBlocks;
    double*         pm1    = a.partials2 + static_cast<long long>(m) * kMaxReduceBlocks;
    const int       stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    const int       first  = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                            static_cast<int>(threadIdx.x);
    const bool      writer = (blockIdx.x == 0u && threadIdx.x == 0u);

    // ---- dot(r0, r) -> kRhoNew --------------------------------------------
    persistentDotStage1(a.n, a.reduce_blocks, a.r0 + base, a.r + base, pm0, shared0);
    grid.sync();
    const double rho_new = persistentFold(a.reduce_blocks, pm0);
    if (writer) sm[kRhoNew] = rho_new;

    // ---- prepare_p_jacobi --------------------------------------------------
    {
        const double rho_old = sm[kRho];
        const double alpha   = sm[kAlpha];
        const double omega   = sm[kOmega];
        const double denom   = rho_old * omega;
        const bool breakdown = !isfinite(rho_new) || !isfinite(denom) ||
                               fabs(denom) < 1.0e-30;
        if (breakdown && writer)
            atomicOr(a.iter_flags + m, static_cast<std::uint32_t>(BICGSTAB_BREAKDOWN));
        for (int l = first; l < a.nxyz; l += stride) {
            const int i0 = 2 * l + 0;
            const int i1 = 2 * l + 1;
            double b0, b1;
            if (breakdown) {
                b0 = a.r[base + i0];
                b1 = a.r[base + i1];
            } else {
                const double beta = rho_new * alpha / denom;
                b0 = a.r[base + i0] + beta * (a.p[base + i0] - omega * a.v[base + i0]);
                b1 = a.r[base + i1] + beta * (a.p[base + i1] - omega * a.v[base + i1]);
            }
            a.p[base + i0] = b0;
            a.p[base + i1] = b1;
            a.y[base + 2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
            a.y[base + 2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
        }
    }
    grid.sync();

    // ---- precondition_sweeps(p, y) ----------------------------------------
    for (int sweep = 0; sweep < a.rb_sweeps; ++sweep) {
        const int target_color = sweep % a.ncolors;
        for (int l = first; l < a.nxyz; l += stride)
            persistentColourSweepNode(l, a, m, target_color, a.p + base, a.y + base);
        grid.sync();
    }

    // ---- matvec_two_group(y -> v) -----------------------------------------
    for (int l = first; l < a.nxyz; l += stride)
        persistentMatvecNode(l, a, m, a.y + base, a.v + base);
    grid.sync();

    // ---- dot(r0, v) -> kR0V ------------------------------------------------
    persistentDotStage1(a.n, a.reduce_blocks, a.r0 + base, a.v + base, pm0, shared0);
    grid.sync();
    const double r0v = persistentFold(a.reduce_blocks, pm0);
    if (writer) sm[kR0V] = r0v;

    // ---- update_s_jacobi ---------------------------------------------------
    {
        const bool nonfinite  = !isfinite(rho_new) || !isfinite(r0v);
        const bool orthogonal = !nonfinite && fabs(r0v) < 1.0e-10;
        if (writer) {
            if (nonfinite)
                atomicOr(a.iter_flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
            else if (orthogonal)
                atomicOr(a.iter_flags + m, static_cast<std::uint32_t>(FLUX_CONVERGED));
        }
        const double alpha = (!nonfinite && !orthogonal) ? rho_new / r0v : 0.0;
        if (writer && !nonfinite) {
            sm[kRho] = rho_new;
            if (!orthogonal) sm[kAlpha] = alpha;
        }
        for (int l = first; l < a.nxyz; l += stride) {
            const int i0 = 2 * l + 0;
            const int i1 = 2 * l + 1;
            double b0, b1;
            if (nonfinite || orthogonal) {
                b0 = a.r[base + i0];
                b1 = a.r[base + i1];
            } else {
                b0 = a.r[base + i0] - alpha * a.v[base + i0];
                b1 = a.r[base + i1] - alpha * a.v[base + i1];
            }
            a.s[base + i0] = b0;
            a.s[base + i1] = b1;
            a.z[base + 2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
            a.z[base + 2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
        }
    }
    grid.sync();

    // ---- precondition_sweeps(s, z) ----------------------------------------
    for (int sweep = 0; sweep < a.rb_sweeps; ++sweep) {
        const int target_color = sweep % a.ncolors;
        for (int l = first; l < a.nxyz; l += stride)
            persistentColourSweepNode(l, a, m, target_color, a.s + base, a.z + base);
        grid.sync();
    }

    // ---- matvec_two_group(z -> t) -----------------------------------------
    for (int l = first; l < a.nxyz; l += stride)
        persistentMatvecNode(l, a, m, a.z + base, a.t + base);
    grid.sync();

    // ---- dot2(s.t -> kPts, t.t -> kPtt) ------------------------------------
    persistentDot2Stage1(a.n, a.reduce_blocks, a.s + base, a.t + base,
                         a.t + base, a.t + base, pm0, pm1, shared0, shared1);
    grid.sync();
    const double pts = persistentFold(a.reduce_blocks, pm0);
    const double ptt = persistentFold(a.reduce_blocks, pm1);
    if (writer) {
        sm[kPts] = pts;
        sm[kPtt] = ptt;
    }

    // ---- update_solution ---------------------------------------------------
    {
        // The FLUX_CONVERGED test is FIRST, exactly as in update_solution: a
        // converged slot returns before the isfinite test, so a stale alpha
        // must not raise NONFINITE_DETECTED here either.
        const bool converged =
            (a.iter_flags[m] & static_cast<std::uint32_t>(FLUX_CONVERGED)) != 0;
        const double alpha = converged ? 0.0 : sm[kAlpha];
        const bool   scalars_bad =
            !converged && (!isfinite(alpha) || !isfinite(pts) || !isfinite(ptt));
        if (scalars_bad && writer)
            atomicOr(a.iter_flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        if (!converged && !scalars_bad) {
            const double omega = (ptt != 0.0) ? pts / ptt : 0.0;
            for (int i = first; i < a.n; i += stride) {
                const double next_phi =
                    a.phi[base + i] + alpha * a.y[base + i] + omega * a.z[base + i];
                const double next_r = a.s[base + i] - omega * a.t[base + i];
                // `continue` is the reference's per-element `return`: that
                // kernel's thread owned exactly one i, so skipping i is what it
                // did.  The kOmega store stays BEHIND this guard and behind the
                // two stores, exactly where `if (i == 0)` sat.
                if (!isfinite(next_phi) || !isfinite(next_r)) {
                    atomicOr(a.iter_flags + m,
                             static_cast<std::uint32_t>(NONFINITE_DETECTED));
                    continue;
                }
                if (next_phi < 0.0)
                    atomicOr(a.iter_flags + m,
                             static_cast<std::uint32_t>(NEGATIVE_FLUX));
                a.phi[base + i] = next_phi;
                a.r[base + i]   = next_r;
                if (i == 0) sm[kOmega] = omega;
            }
        }
    }
    grid.sync();

    // ---- residual norm stage 1 + the fused scalar tail ---------------------
    persistentDotStage1(a.n, a.reduce_blocks, a.r + base, a.r + base, pm0, shared0);
    grid.sync();
    if (!writer) return;
    if (a.active[m] == 0u) return;
    std::uint32_t* cm = a.counters + static_cast<long long>(m) * kCounterSlots;
    if (a.halt[m] != 0u) {
        ++cm[kOverrunCount];
        return;
    }
    sm[kInitialNorm] = sqrt(persistentFold(a.reduce_blocks, pm0));
    accumulate_iteration_active(m, a.allow_halt, a.force_halt, a.scalars, a.iter_flags,
                                a.sticky_flags, a.counters, a.halt);
}

__global__ void finalize_status(const double* scalars,
                                const std::uint32_t* flags,
                                const std::uint32_t* counters,
                                DeviceSolveStatus* status,
                                const std::uint32_t* __restrict__ active) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
    // Not halt-guarded: by this point `halt` is legitimately 1 for every slot
    // whose inner loop exited early, and those slots still need a status.
    if (active[m] == 0u) return;

    // MERGE RECONCILIATION with the CPU track (fix4, src/BICGSolver.cpp):
    // flux_l2 is the PLAIN ABSOLUTE residual norm ||r|| = ||s - omega*t|| of
    // the last iterate that actually ran.  The relative test against r20 is
    // now applied by accumulate_iteration on the device; before the batching
    // change it was applied by the host in BICGCMFD.cpp.  Either way there is
    // exactly one relative test, against a reference frozen at reset().
    const double*        sm     = scalars + static_cast<long long>(m) * kScalarCount;
    const std::uint32_t* cm     = counters + static_cast<long long>(m) * kCounterSlots;
    const std::uint32_t  sticky = flags[m];
    const double         rnorm  = sm[kInitialNorm];
    const double value =
        ((sticky & static_cast<std::uint32_t>(NONFINITE_DETECTED)) != 0 || !isfinite(rnorm))
            ? nan("")
            : rnorm;
    const double unavailable = nan("");
    status[m].keff            = unavailable;
    status[m].flux_l2         = value;
    status[m].dhat_defect_max = unavailable;
    status[m].dhat_update_max = unavailable;
    status[m].search_residual = unavailable;
    status[m].flags           = sticky;
    status[m].outer_iter      = cm[kOverrunCount];
    status[m].linear_iter     = cm[kSolveCount];
    // Reused as transport for the device-side tallies: the batched inner loop
    // no longer stops on the host, so these can only come back this way.
    status[m].material_gen    = cm[kRestartCount];
    status[m].operator_gen    = cm[kEarlyExitCount];
    status[m].flux_gen        = cm[kSolveCount];
}

// ===========================================================================
// Mixed-precision inner iteration (RASBERY_GPU_CMFD_FP32, default OFF).
//
// On the measured card FP64 is throttled to 1/64 of FP32, so the natural
// reading is "the inner loop is ALU-starved in double".  That is half the
// story: every kernel in this solver is a stencil or a BLAS-1 sweep with an
// arithmetic intensity of 0.1 - 0.3 FLOP/byte, so what FP32 actually buys is
// HALVED TRAFFIC on the operator and the Krylov vectors, with the FP64 issue
// pressure removed as a bonus.  Either way the lever is the same one.
//
// The structure is classic iterative refinement / mixed-precision Krylov, and
// it maps onto the two levels this solver already has:
//
//   FP64, unchanged   the stored operator (diag / cc / udiag), the flux phi,
//                     the source, psi, the Wielandt terms and the eigenvalue,
//                     the 2x2 block inversion, the TRUE residual r = b - A*phi
//                     recomputed at the top of every outer, the reference norm
//                     r20 taken from that FP64 residual, every entry of
//                     `scalars` (rho, alpha, omega, the norms), every
//                     convergence and non-finite test, and the correction
//                     accumulation phi += (double)dx.
//   FP32              the Krylov working vectors (r, r0, p, v, s, t, y, z),
//                     the colour Gauss-Seidel sweeps, the preconditioner's
//                     inverted diagonal blocks (dinv), and the operator
//                     application A*y / A*z inside the inner loop -- through
//                     per-slot FLOAT MIRRORS of diag/cc that feed nothing else.
//
// One outer is therefore: FP64 residual -> FP32 inner solve for the correction
// -> FP64 correction update, which is the refinement scheme exactly.  The
// inner loop's job is to take one or two orders off the residual (nmaxbicg = 3
// captured iterations against _epsbicg = 0.1), comfortably inside what FP32
// BiCGSTAB delivers before it stagnates; the accuracy comes from the outer
// Wielandt loop, which never leaves FP64.
//
// DETERMINISM.  The FP32 path keeps every rule the FP64 path rests on: `chunk`
// still depends only on (n, gridDim.x), the per-thread traversal, the fixed
// binary tree and the strict index-order stage-2 fold are unchanged, and the
// batch axis is still gridDim.y alone.  The one deliberate difference is the
// PAYLOAD / ACCUMULATOR SPLIT: stage 1 loads FLOAT operands and folds them into
// a DOUBLE accumulator -- float x float widened to double is EXACT, so stage 1
// sums exact products -- and stage 2 is the existing double kernel, reused
// unmodified.  A run is thus bit-reproducible run to run and independent of
// batch composition, exactly as before; it is simply not bit-equal to the FP64
// path, which is what the Gate A/B numeric gates (not the bit-golden gate)
// validate.  The dots are memory bound at these sizes, so the double
// accumulator is free and buys back the scalar accuracy BiCGSTAB is most
// sensitive to.
//
// WHY A PARALLEL KERNEL SET rather than template<typename T>.  Three of the
// kernels below are precision-MIXED at their boundary (the prologue reads a
// double operator and writes a float Krylov state; update_solution reads float
// vectors and accumulates into a double flux), so one template would need an
// `if constexpr` at precisely the sites that matter and would still have to be
// launched from a branch, because the pointer types differ.  Duplicating
// instead leaves every FP64 kernel and the whole FP64 enqueue path TEXTUALLY
// UNTOUCHED, which is the property the byte-identity gate on the OFF path
// actually needs, and it keeps the capture trivial: the two sets are in 1:1
// kernel correspondence, so the graph has the same shape either way (plus the
// one mirror-refresh node), and the choice is made once, before capture.
// ===========================================================================

/// Refresh the per-slot FLOAT MIRRORS of the operator from the authoritative
/// double arrays.
///
/// This runs at the top of every outer, right after initialize_solver_state and
/// before anything can read diag_f/cc_f.  That single site DOMINATES every
/// mutation of the double operator -- the H2D pushes in issueUploads and
/// issueSweepUploads, the device assembly in cmfd_assemble_operator_2g, and the
/// per-sweep Wielandt rewrite in cmfd_updls all complete before the next
/// enqueue_outer -- so the mirror cannot go stale by CONSTRUCTION rather than by
/// an audit of call sites.  It is the only writer of diag_f and cc_f.
__global__ void refresh_operator_mirror_f32(const int nxyz,
                                            const long long mat_stride,
                                            const long long cpl_stride,
                                            const double* __restrict__ diag,
                                            const double* __restrict__ cc,
                                            float* __restrict__ diag_f,
                                            float* __restrict__ cc_f,
                                            const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double* dm = diag + m * mat_stride;
    float*        df = diag_f + m * mat_stride;
#pragma unroll
    for (int k = 0; k < 4; ++k) df[4 * l + k] = static_cast<float>(dm[4 * l + k]);

    const double* cm = cc + m * cpl_stride;
    float*        cf = cc_f + m * cpl_stride;
#pragma unroll
    for (int k = 0; k < 12; ++k) cf[12 * l + k] = static_cast<float>(cm[12 * l + k]);
}

/// FP32 twin of begin_outer_fused, and the FP64 half of the refinement step.
///
/// Everything that decides accuracy stays in double: the 2x2 block inversion
/// (its determinant is the one cancellation-prone quantity in the whole inner
/// loop, and it is computed once per outer, not once per iteration), the A*phi
/// application and the residual b - A*phi.  Only the RESULT is narrowed -- the
/// inverted blocks into dinv_f and the residual into the FP32 Krylov state.
///
/// The double residual is still written to `r` because the reference norm r20,
/// the fixed denominator of the inner-loop exit test, must be the FP64 one; the
/// prologue reduction that follows is the unmodified FP64 pair.
__global__ void begin_outer_fused_f32(const int nxyz,
                                      const long long vec_stride,
                                      const long long mat_stride,
                                      const long long cpl_stride,
                                      const int* __restrict__ neighbors,
                                      const double* __restrict__ diag,
                                      const double* __restrict__ cc,
                                      const double* __restrict__ x,
                                      const double* __restrict__ src,
                                      float* __restrict__ dinv_f,
                                      double* __restrict__ ax,
                                      double* __restrict__ r,
                                      float* __restrict__ r_f,
                                      float* __restrict__ r0_f,
                                      float* __restrict__ p_f,
                                      float* __restrict__ v_f,
                                      const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double* dm = diag + m * mat_stride;

    // ---- invert_two_group_blocks, evaluated in FP64, stored narrowed ----
    {
        float* im = dinv_f + m * mat_stride;

        const double a00  = dm[4 * l + 0];
        const double a01  = dm[4 * l + 1];
        const double a10  = dm[4 * l + 2];
        const double a11  = dm[4 * l + 3];
        const double rdet = 1.0 / (a00 * a11 - a10 * a01);

        im[4 * l + 0] = static_cast<float>(rdet * a11);
        im[4 * l + 1] = static_cast<float>(-rdet * a01);
        im[4 * l + 2] = static_cast<float>(-rdet * a10);
        im[4 * l + 3] = static_cast<float>(rdet * a00);
    }

    // ---- matvec_two_group(x = phi -> y = ax), in FP64 ----
    const double* cm = cc + m * cpl_stride;
    const double* xm = x + m * vec_stride;
    double*       ym = ax + m * vec_stride;

    const double x0 = xm[2 * l + 0];
    const double x1 = xm[2 * l + 1];
    double       y0 = dm[4 * l + 0] * x0 + dm[4 * l + 1] * x1;
    double       y1 = dm[4 * l + 2] * x0 + dm[4 * l + 3] * x1;

#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = neighbors[6 * l + slot];
        if (neighbor >= 0) {
            y0 += cm[12 * l + slot] * xm[2 * neighbor + 0];
            y1 += cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    ym[2 * l + 0] = y0;
    ym[2 * l + 1] = y1;

    // ---- initial residual: FP64 for the reference norm, FP32 for the loop ----
    const long long base   = m * vec_stride;
    const double    value0 = src[base + 2 * l + 0] - y0;
    const double    value1 = src[base + 2 * l + 1] - y1;
    r[base + 2 * l + 0]    = value0;
    r[base + 2 * l + 1]    = value1;

    const float f0 = static_cast<float>(value0);
    const float f1 = static_cast<float>(value1);
    r_f[base + 2 * l + 0]  = f0;
    r0_f[base + 2 * l + 0] = f0;
    p_f[base + 2 * l + 0]  = 0.0f;
    v_f[base + 2 * l + 0]  = 0.0f;
    r_f[base + 2 * l + 1]  = f1;
    r0_f[base + 2 * l + 1] = f1;
    p_f[base + 2 * l + 1]  = 0.0f;
    v_f[base + 2 * l + 1]  = 0.0f;
}

/// FP32 payload, FP64 accumulator.  Partition, traversal order and reduction
/// tree are those of reduce_dot_stage1, verbatim; only the operand loads are
/// narrowed and the products are widened back before they are summed.  Stage 2
/// is the existing double kernel -- there is no _f32 stage 2.
__global__ void reduce_dot_stage1_f32(const int n,
                                      const long long vec_stride,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      double* __restrict__ partial,
                                      const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    __shared__ double shared[kReduceThreads];

    const float* am = a + m * vec_stride;
    const float* bm = b + m * vec_stride;
    double*      pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;

    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x))
        sum += static_cast<double>(am[i]) * static_cast<double>(bm[i]);

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) pm[blockIdx.x] = shared[0];
}

/// Two independent FP32-payload dots in one pair of nodes; same argument as
/// reduce_dot2_stage1, same double accumulators as reduce_dot_stage1_f32.
__global__ void reduce_dot2_stage1_f32(const int n,
                                       const long long vec_stride,
                                       const float* __restrict__ a0,
                                       const float* __restrict__ b0,
                                       const float* __restrict__ a1,
                                       const float* __restrict__ b1,
                                       double* __restrict__ partial0,
                                       double* __restrict__ partial1,
                                       const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    __shared__ double shared0[kReduceThreads];
    __shared__ double shared1[kReduceThreads];

    const float* a0m = a0 + m * vec_stride;
    const float* b0m = b0 + m * vec_stride;
    const float* a1m = a1 + m * vec_stride;
    const float* b1m = b1 + m * vec_stride;
    double*      p0m = partial0 + static_cast<long long>(m) * kMaxReduceBlocks;
    double*      p1m = partial1 + static_cast<long long>(m) * kMaxReduceBlocks;

    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        sum0 += static_cast<double>(a0m[i]) * static_cast<double>(b0m[i]);
        sum1 += static_cast<double>(a1m[i]) * static_cast<double>(b1m[i]);
    }

    shared0[threadIdx.x] = sum0;
    shared1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared0[threadIdx.x] += shared0[threadIdx.x + stride];
            shared1[threadIdx.x] += shared1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        p0m[blockIdx.x] = shared0[0];
        p1m[blockIdx.x] = shared1[0];
    }
}

/// FP32 twin of matvec_two_group, reading the float operator mirrors.
__global__ void matvec_two_group_f32(const int nxyz,
                                     const long long vec_stride,
                                     const long long mat_stride,
                                     const long long cpl_stride,
                                     const int* __restrict__ neighbors,
                                     const float* __restrict__ diag_f,
                                     const float* __restrict__ cc_f,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const float* dm = diag_f + m * mat_stride;
    const float* cm = cc_f + m * cpl_stride;
    const float* xm = x + m * vec_stride;
    float*       ym = y + m * vec_stride;

    const float x0 = xm[2 * l + 0];
    const float x1 = xm[2 * l + 1];
    float       y0 = dm[4 * l + 0] * x0 + dm[4 * l + 1] * x1;
    float       y1 = dm[4 * l + 2] * x0 + dm[4 * l + 3] * x1;

#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = neighbors[6 * l + slot];
        if (neighbor >= 0) {
            y0 += cm[12 * l + slot] * xm[2 * neighbor + 0];
            y1 += cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    ym[2 * l + 0] = y0;
    ym[2 * l + 1] = y1;
}

/// FP32 twin of colored_block_sweep.  The colour order IS the Gauss-Seidel
/// semantics and the kernel boundary IS the grid-wide barrier, so the sweep
/// structure is untouched; only the arithmetic narrows.
__global__ void colored_block_sweep_f32(const int nxyz,
                                        const long long vec_stride,
                                        const long long mat_stride,
                                        const long long cpl_stride,
                                        const int target_color,
                                        const int* __restrict__ colors,
                                        const int* __restrict__ neighbors,
                                        const float* __restrict__ cc_f,
                                        const float* __restrict__ dinv_f,
                                        const float* __restrict__ b,
                                        float* __restrict__ x,
                                        const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz || colors[l] != target_color) return;

    const float* cm = cc_f + m * cpl_stride;
    const float* im = dinv_f + m * mat_stride;
    const float* bm = b + m * vec_stride;
    float*       xm = x + m * vec_stride;

    float b0 = bm[2 * l + 0];
    float b1 = bm[2 * l + 1];
#pragma unroll
    for (int slot = 0; slot < 6; ++slot) {
        const int neighbor = neighbors[6 * l + slot];
        if (neighbor >= 0) {
            b0 -= cm[12 * l + slot] * xm[2 * neighbor + 0];
            b1 -= cm[12 * l + 6 + slot] * xm[2 * neighbor + 1];
        }
    }

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

/// FP32 twin of prepare_p_jacobi.  The breakdown test and beta itself are
/// evaluated in FP64 from the FP64 `scalars` -- they are the numbers the whole
/// iteration hangs on and they cost one thread's worth of work -- and only the
/// vector update runs narrowed.
__global__ void prepare_p_jacobi_f32(const int nxyz,
                                     const long long vec_stride,
                                     const long long mat_stride,
                                     double* scalars,
                                     std::uint32_t* flags,
                                     const float* __restrict__ dinv_f,
                                     const float* __restrict__ r_f,
                                     const float* __restrict__ v_f,
                                     float* __restrict__ p_f,
                                     float* __restrict__ y_f,
                                     const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double*   sm   = scalars + static_cast<long long>(m) * kScalarCount;
    const long long base = m * vec_stride;
    const int       i0   = 2 * l + 0;
    const int       i1   = 2 * l + 1;

    const double rho_new = sm[kRhoNew];
    const double rho_old = sm[kRho];
    const double alpha   = sm[kAlpha];
    const double omega   = sm[kOmega];
    const double denom   = rho_old * omega;
    const bool breakdown = !isfinite(rho_new) || !isfinite(denom) ||
                           fabs(denom) < 1.0e-30;
    float b0, b1;
    if (breakdown) {
        atomicOr(flags + m, static_cast<std::uint32_t>(BICGSTAB_BREAKDOWN));
        b0 = r_f[base + i0];
        b1 = r_f[base + i1];
    } else {
        const float beta   = static_cast<float>(rho_new * alpha / denom);
        const float omegaf = static_cast<float>(omega);
        b0 = r_f[base + i0] + beta * (p_f[base + i0] - omegaf * v_f[base + i0]);
        b1 = r_f[base + i1] + beta * (p_f[base + i1] - omegaf * v_f[base + i1]);
    }
    p_f[base + i0] = b0;
    p_f[base + i1] = b1;

    // ---- block_jacobi(b = p, x = y) ----
    const float* im = dinv_f + m * mat_stride;
    float*       xm = y_f + m * vec_stride;

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

/// FP32 twin of update_s_jacobi.  rho, r0.v and alpha are the FP64 dot results
/// and every test on them keeps its FP64 form and its FP64 threshold; alpha is
/// narrowed once, at the point it multiplies a vector.
__global__ void update_s_jacobi_f32(const int nxyz,
                                    const long long vec_stride,
                                    const long long mat_stride,
                                    double* scalars,
                                    std::uint32_t* flags,
                                    const float* __restrict__ dinv_f,
                                    const float* __restrict__ r_f,
                                    const float* __restrict__ v_f,
                                    float* __restrict__ s_f,
                                    float* __restrict__ z_f,
                                    const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    double*         sm   = scalars + static_cast<long long>(m) * kScalarCount;
    const long long base = m * vec_stride;
    const int       i0   = 2 * l + 0;
    const int       i1   = 2 * l + 1;

    const double rho = sm[kRhoNew];
    const double r0v = sm[kR0V];
    float b0, b1;
    if (!isfinite(rho) || !isfinite(r0v)) {
        atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        b0 = r_f[base + i0];
        b1 = r_f[base + i1];
    } else {
        if (l == 0) sm[kRho] = rho;

        if (fabs(r0v) < 1.0e-10) {
            atomicOr(flags + m, static_cast<std::uint32_t>(FLUX_CONVERGED));
            b0 = r_f[base + i0];
            b1 = r_f[base + i1];
        } else {
            const double alpha  = rho / r0v;
            const float  alphaf = static_cast<float>(alpha);
            b0 = r_f[base + i0] - alphaf * v_f[base + i0];
            b1 = r_f[base + i1] - alphaf * v_f[base + i1];
            if (l == 0) sm[kAlpha] = alpha;
        }
    }
    s_f[base + i0] = b0;
    s_f[base + i1] = b1;

    // ---- block_jacobi(b = s, x = z) ----
    const float* im = dinv_f + m * mat_stride;
    float*       xm = z_f + m * vec_stride;

    xm[2 * l + 0] = im[4 * l + 0] * b0 + im[4 * l + 1] * b1;
    xm[2 * l + 1] = im[4 * l + 2] * b0 + im[4 * l + 3] * b1;
}

/// THE REFINEMENT STEP.  The correction dx = alpha*y + omega*z is formed in
/// FP32 from the FP32 search directions and then WIDENED before it is added to
/// the FP64 flux, so the flux accumulates in double for the whole run and only
/// the increment ever lives in single.  The recursive residual stays FP32; the
/// true FP64 residual is re-established by begin_outer_fused_f32 at the next
/// outer, which is what makes this refinement rather than a plain FP32 solve.
///
/// The non-finite guards are the FP64 ones, and they still refuse to WRITE a
/// bad flux: on failure the element keeps its last finite value, so a slot that
/// trips this exits the inner loop with the iterate it entered with rather than
/// with garbage.  That is what lets the host absorb one FP32 failure and fall
/// back to the FP64 path instead of failing the deck (see BatchCore::drain).
__global__ void update_solution_f32(const int n,
                                    const long long vec_stride,
                                    double* scalars,
                                    std::uint32_t* flags,
                                    const float* __restrict__ y_f,
                                    const float* __restrict__ z_f,
                                    const float* __restrict__ s_f,
                                    const float* __restrict__ t_f,
                                    double* __restrict__ phi,
                                    float* __restrict__ r_f,
                                    const std::uint32_t* __restrict__ halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    HALT_GUARD(halt + m);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if ((flags[m] & static_cast<std::uint32_t>(FLUX_CONVERGED)) != 0) return;

    double*         sm   = scalars + static_cast<long long>(m) * kScalarCount;
    const long long base = m * vec_stride;

    const double alpha = sm[kAlpha];
    const double pts   = sm[kPts];
    const double ptt   = sm[kPtt];
    if (!isfinite(alpha) || !isfinite(pts) || !isfinite(ptt)) {
        atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        return;
    }

    const double omega  = (ptt != 0.0) ? pts / ptt : 0.0;
    const float  alphaf = static_cast<float>(alpha);
    const float  omegaf = static_cast<float>(omega);

    const float  dx       = alphaf * y_f[base + i] + omegaf * z_f[base + i];
    const double next_phi = phi[base + i] + static_cast<double>(dx);
    const float  next_r   = s_f[base + i] - omegaf * t_f[base + i];
    // The residual test is written on the widened value so the overload picked
    // here is the same isfinite(double) the FP64 kernel uses; widening a float
    // preserves inf and NaN exactly, so the two tests agree by construction.
    if (!isfinite(next_phi) || !isfinite(static_cast<double>(next_r))) {
        atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));
        return;
    }
    if (next_phi < 0.0)
        atomicOr(flags + m, static_cast<std::uint32_t>(NEGATIVE_FLUX));
    phi[base + i] = next_phi;
    r_f[base + i] = next_r;
    if (i == 0) sm[kOmega] = omega;
}

// ---------------------------------------------------------------------------
// Device-resident CMFD sweep (RASBERY_GPU_CMFD_SWEEP).
//
// One graph launch runs up to `unroll` Wielandt sweeps -- source rebuild,
// BiCGSTAB inner (the existing enqueue_outer sequence), the wiel eigenvalue
// update, updls and the negative-flux bookkeeping -- with a per-slot
// `sweep_halt` playing the role the host `break`/retry logic played, exactly
// as `halt` does for the BiCG inner loop.
//
// Every contraction-ambiguous expression uses the form MINED from the
// production host build by test/cmfd_form_probe.cpp (capture:
// RASBERY_CMFD_DUMP).  gcc's pattern on this code: in `a*b + c*d` it rounds
// the SECOND product and fuses the FIRST into the add; the wiel accumulations
// are unfused; the updls subtract is fused.  On the device, fma() forces the
// fused sites and __dmul_rn() pins the unfused ones regardless of this TU's
// -fmad setting, so the kernels below cannot drift when compiler flags do.
// ---------------------------------------------------------------------------

/// Build the unshifted two-group CMFD operator and its initial Wielandt
/// diagonal directly in the arena allocations consumed by BiCGSTAB. Geometry
/// is shared; every mutable input/output is slot-strided.
__global__ void cmfd_assemble_operator_2g(
    const int nxyz,
    const long long vec_stride,
    const long long mat_stride,
    const long long cpl_stride,
    const long long surface_stride,
    const int* __restrict__ node_surface,
    const double* __restrict__ face_area,
    const double* __restrict__ geometry_volume,
    const double* __restrict__ xsrf,
    const double* __restrict__ xssm,
    const double* __restrict__ chif,
    const double* __restrict__ xsnf,
    const double* __restrict__ dtil,
    const double* __restrict__ dhat,
    double* diag,
    double* cc,
    double* udiag,
    const double* __restrict__ scalars,
    const std::uint32_t* __restrict__ device_assembly_active,
    const std::uint32_t* __restrict__ sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (device_assembly_active[m] == 0u || sweep_halt[m] != 0u) return;
    const int l = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;

    const double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    cmfd_assembly::View view{
        nxyz,
        node_surface,
        face_area,
        geometry_volume,
        xsrf + m * vec_stride,
        xssm + m * mat_stride,
        chif + m * vec_stride,
        xsnf + m * vec_stride,
        dtil + m * surface_stride,
        dhat + m * surface_stride,
        sm[kReigvs],
        sm[kEshift],
        diag + m * mat_stride,
        cc + m * cpl_stride,
        udiag + m * mat_stride,
    };
    cmfd_assembly::assembleNode2G(view, l);
}

__global__ void cmfd_sweep_begin(double* scalars, std::uint32_t* sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;

    // Rev.7.1 Task 6 Step 3.  The captured graph is CAPACITY: it can hold more
    // sweep slots than this launch may spend, so the slot budget is a DEVICE
    // scalar and the excess halts HERE -- at the first kernel of the slot,
    // before anything is read or written.
    //
    // This is what makes over-capture bit-identical rather than merely similar.
    // A halted slot's every later kernel returns on its first instruction:
    // cmfd_src_build, cmfd_wiel_terms, cmfd_wiel_finalize, cmfd_updls,
    // cmfd_negative_scan and cmfd_sweep_end all test sweep_halt, and the whole
    // inner BiCGSTAB is masked too because initialize_solver_state folds
    // sweep_halt into `halt` and returns before it touches scalars, flags or
    // counters.  So a graph of depth D launched with a budget of U < D slots
    // executes exactly what a depth-U graph would have.
    //
    // Note this is the SLOT budget, not kSweepBudget.  kSweepBudget counts
    // ADVANCES (cmfd_sweep_end skips the increment on a negative-flux retry);
    // this counts attempts.  Conflating them would silently change how many
    // retries fit in one launch, which is the one observable the host's
    // `state == 0` loop reacts to.
    if (sm[kSweepSlots] >= sm[kSweepSlotBudget]) {
        sweep_halt[m] = 1u;
        return;
    }
    sm[kSweepSlots] += 1.0;

    sm[kReigvdel] = sm[kReigv] - sm[kReigvs];
    sm[kNegative] = 0.0;
    sm[kIcmfdDone] += 1.0; // host ++icmfd at the top of each pass
}

/// src(ig,l) = chif(ig,l) * (psi(l) * reigvdel) -- two bare multiplies, no
/// contraction ambiguity.  chif/xsnf are group-major [ig*nxyz+l], src/flux are
/// node-major [l*ng+ig], psi/vol are [l]; all strides mirror the host arrays.
__global__ void cmfd_src_build(const int nxyz,
                               const long long vec_stride,
                               const long long node_stride,
                               const double* __restrict__ chif,
                               const double* __restrict__ psi,
                               double* __restrict__ src,
                               const double* __restrict__ scalars,
                               const std::uint32_t* __restrict__ sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    const int l = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;
    const double fs =
        psi[m * node_stride + l] *
        scalars[static_cast<long long>(m) * kScalarCount + kReigvdel];
    const double* cm = chif + m * vec_stride;
    double*       sv = src + m * vec_stride;
    sv[l * 2 + 0]    = cm[l] * fs;
    sv[l * 2 + 1]    = cm[nxyz + l] * fs;
}

/// The wiel node work, split for speed WITHOUT changing a single rounding:
/// the mined accumulation forms are UNFUSED, so each addend the host folds in
/// (round(err1*err1), round(psid*pv), round(pv*pv)) is an ordinary rounded
/// double -- computing those addends in parallel and then folding the stored
/// values in the same l-ascending order is bit-for-bit the host's serial
/// loop, while turning a one-thread stride-gather walk into a coalesced
/// parallel pass plus a cache-friendly sequential sum.
__global__ void cmfd_wiel_terms(const int nxyz,
                                const long long vec_stride,
                                const long long node_stride,
                                const double* __restrict__ phi,
                                double* __restrict__ psi,
                                const double* __restrict__ xsnf,
                                const double* __restrict__ vol,
                                double* __restrict__ terms_ab, ///< [slot][2*nxyz]: err1^2, psid*pv
                                double* __restrict__ terms_c,  ///< [slot][>=nxyz]: pv*pv
                                const std::uint32_t* __restrict__ sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    const int l = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;
    const double* f    = phi + m * vec_stride;
    const double* x0   = xsnf + m * vec_stride;
    const double* x1   = x0 + nxyz;
    double*       ps   = psi + m * node_stride;
    const double  psid = ps[l];
    double        pv   = fma(x0[l], f[l * 2 + 0], __dmul_rn(x1[l], f[l * 2 + 1]));
    pv                 = pv * vol[m * node_stride + l];
    const double err1  = pv - psid;
    double*      ta    = terms_ab + m * vec_stride;
    ta[l]              = __dmul_rn(err1, err1);
    ta[nxyz + l]       = __dmul_rn(psid, pv);
    terms_c[m * vec_stride + l] = __dmul_rn(pv, pv);
    ps[l]              = pv;
}

/// How many addends one fold lane pulls into registers before it folds them.
///
/// This is a MEMORY-LEVEL-PARALLELISM knob and nothing else.  The fold below
/// is a hard serial dependency (see the kernel comment), so the only latency
/// a restructuring can remove is the load latency, and the only way a single
/// thread removes load latency is by having more than one load in flight.
///
/// Measured on GP102 (sm_61, nxyz = 8451 -- the KNGR mesh), interleaved
/// best-of-12 so the clock ramp cannot bias the ratio:
///
///     flat (what ptxas schedules on its own)   258.3 us   1.00x
///     batch 4                                  258.8 us   1.00x
///     batch 8 / 16 / 32                        230.6 us   1.12x
///     batch 64                                 216.8 us   1.19x
///     dependent DADD chain, no memory at all   202.3 us   1.28x
///
/// That last row is the ceiling: it is the same 8451-long chain with the
/// operands already in registers, so 1.28x is ALL that any bit-preserving
/// rewrite of this fold can ever be worth.  64 collects 93% of it and ptxas
/// reports 32 registers with zero spill, so there is nothing to trade back.
constexpr int kWielFoldBatch = 64;

/// The scalar tail of the Wielandt update, SHARED by both fold modes.
///
/// This is the frozen part: the export of the three sums, the Rayleigh
/// hand-back latch (sweep_state = 2 plus sweep_halt, taken whenever gamma is
/// degenerate), the fma/__dmul_rn choices in the eigenvalue update and the
/// shift.  Factoring it out is what makes "the two modes differ ONLY in how
/// the three sums were summed" a property of the source rather than a claim
/// about it -- there is exactly one copy, so the chunked arm cannot drift into
/// a different convergence or latch semantics while nobody is looking.
__device__ __forceinline__ void cmfd_wiel_apply(const double err, const double gammad,
                                                const double gamman, double* scalars,
                                                std::uint32_t* sweep_halt, const int m) {
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    sm[kErrAcc] = err;
    sm[kGammaD] = gammad;
    sm[kGammaN] = gamman;
    if (!((gammad > 0.0) && (gamman > 0.0))) {
        sm[kSweepState] = 2.0; // host finishes this sweep with the Rayleigh path
        sweep_halt[m]   = 1u;
        return;
    }
    const double gamma = gammad / gamman;
    const double den   = fma(sm[kReigv], gamma, __dmul_rn(1.0 - gamma, sm[kReigvs]));
    const double eigv  = 1.0 / den;
    sm[kEigv]          = eigv;
    sm[kReigv]         = 1.0 / eigv;
    const double err_scale = (gammad > 0.0) ? gammad : gamman;
    sm[kErrl2] = (err_scale > 0.0) ? sqrt(fabs(err / err_scale)) : 0.0;
    double eigvs = eigv, reigvs = 0.0;
    eigvs += sm[kEshift]; // icy >= 0 by the delegation contract
    if (sm[kEshift] != 0.0) reigvs = 1.0 / eigvs;
    sm[kReigvs] = reigvs;
}

/// The serial l-ascending fold of the stored addends, plus the eigenvalue
/// update.  One thread per slot; slots in parallel on gridDim.y.  The
/// warm-up (icy < 0) and Rayleigh branches never run here: the host only
/// delegates once the Wielandt regime is reached, and a degenerate gamma
/// hands the sweep back to the host with the sums exported.
///
/// WHY THIS FOLD IS SERIAL, AND WHY IT STAYS SERIAL.  `sum = sum + v[l]` over
/// l ascending is BICGCMFD::wiel's own accumulation (BICGCMFD.cpp, the
/// `err`/`gammad`/`gamman` loop), and a rounded floating-point add is not
/// associative: any partition of the range into chunks that are summed
/// separately and then folded changes the result by a few ULP.  The stage-1 /
/// stage-2 partition that reduce_dot_stage1/2 use is therefore NOT available
/// here -- those reductions define their own order and only have to be
/// reproducible, this one has to reproduce a specific serial order that is
/// already baked into the frozen reference output.  Nor is there a parallel
/// algorithm that reproduces a chosen sequential rounding sequence: each
/// rounding depends on the running sum, so the chain length nxyz IS the
/// critical path.
///
/// What CAN be removed is load latency, and only that; see kWielFoldBatch.
/// The addends are folded in exactly the same order, each one an ordinary
/// rounded double, so the batching is bit-preserving by construction -- only
/// the loads move, never an add.
__global__ void cmfd_wiel_finalize(const int nxyz,
                                   const long long vec_stride,
                                   const double* __restrict__ terms_ab,
                                   const double* __restrict__ terms_c,
                                   double* scalars,
                                   std::uint32_t* sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return; // uniform for the whole slot block
    const int lane = static_cast<int>(threadIdx.x);
    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    __shared__ double lane_sum[3];

    // Each independent sum retains the original l-ascending dependency chain.
    // Lanes 0, 1 and 2 therefore run concurrently without changing a sum's
    // operand pairing or deterministic double result.
    if (lane < 3) {
        const double* values = lane == 0 ? ta : (lane == 1 ? ta + nxyz : tc);
        double sum = 0.0;
        // RASBERY_CMFD_WIEL_FOLD_BEGIN -- mirrored verbatim by
        // test/cmfd_wiel_fold_replay.cu; tools/test_cmfd_wiel_fold_contract.py
        // fails if the two texts drift apart.
        int       l    = 0;
        const int tail = nxyz - (nxyz % kWielFoldBatch);
        for (; l < tail; l += kWielFoldBatch) {
            double batch[kWielFoldBatch];
#pragma unroll
            for (int j = 0; j < kWielFoldBatch; ++j) batch[j] = __ldg(values + l + j);
#pragma unroll
            for (int j = 0; j < kWielFoldBatch; ++j) sum = sum + batch[j];
        }
        for (; l < nxyz; ++l) sum = sum + values[l];
        // RASBERY_CMFD_WIEL_FOLD_END
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    cmfd_wiel_apply(lane_sum[0], lane_sum[1], lane_sum[2], scalars, sweep_halt, m);
}

// ---------------------------------------------------------------------------
// OPT-IN CHUNKED FOLD (RASBERY_GPU_WIEL_FOLD=chunked).  Class N1.
//
// Same three sums, a different summation ORDER, and therefore different bits
// by 1-4 ULP.  What it is NOT is nondeterministic: the partition below is a
// pure function of (nxyz, gridDim.x), the per-thread traversal is a fixed
// stride, the in-block tree pairs the same operands every launch, the stage-2
// fold walks the partials in strict ascending index order, and there is no
// atomic anywhere in either kernel.  Two runs of the same deck on the same
// arch produce the same doubles -- that is exactly the reduce_dot_stage1/2
// contract, and it is what makes this an N1 arm rather than a coin toss.
//
// gridDim.x is chosen by reduce_blocks_for(nxyz), the same function the BiCG
// reductions use, so a bucket of the compaction ladder cannot change it and a
// captured graph replays the identical partition.
// ---------------------------------------------------------------------------

/// Stage 1: three independent reductions riding in one kernel, each with its
/// own accumulator, its own shared array and its own partial row -- the
/// reduce_dot2_stage1 pattern widened to three, not a fused reduction.
__global__ void cmfd_wiel_stage1(const int nxyz,
                                 const long long vec_stride,
                                 const double* __restrict__ terms_ab,
                                 const double* __restrict__ terms_c,
                                 double* __restrict__ partial, ///< [slot][3][kMaxReduceBlocks]
                                 const std::uint32_t* __restrict__ sweep_halt,
                                 RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    __shared__ double sh_err[kReduceThreads];
    __shared__ double sh_gd[kReduceThreads];
    __shared__ double sh_gn[kReduceThreads];

    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    double*       pm = partial + static_cast<long long>(m) * (3 * kMaxReduceBlocks);

    // Fixed, contiguous chunk per block: depends only on (nxyz, gridDim.x).
    const int chunk = (nxyz + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, nxyz);

    double s_err = 0.0, s_gd = 0.0, s_gn = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        s_err += ta[i];
        s_gd += ta[nxyz + i];
        s_gn += tc[i];
    }
    sh_err[threadIdx.x] = s_err;
    sh_gd[threadIdx.x]  = s_gd;
    sh_gn[threadIdx.x]  = s_gn;
    __syncthreads();

    // Fixed binary tree: identical operand pairing on every launch.
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            sh_err[threadIdx.x] += sh_err[threadIdx.x + stride];
            sh_gd[threadIdx.x] += sh_gd[threadIdx.x + stride];
            sh_gn[threadIdx.x] += sh_gn[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pm[0 * kMaxReduceBlocks + blockIdx.x] = sh_err[0];
        pm[1 * kMaxReduceBlocks + blockIdx.x] = sh_gd[0];
        pm[2 * kMaxReduceBlocks + blockIdx.x] = sh_gn[0];
    }
}

/// Stage 2: the strict ascending fold over the partials, then the SAME scalar
/// tail the serial path runs.  Three lanes, one per sum, each keeping its own
/// ascending chain -- the fold order over `blocks` partials is the one thing
/// stage 2 is allowed to have, and it is fixed.
__global__ void cmfd_wiel_finalize_chunked(const int blocks,
                                           const double* __restrict__ partial,
                                           double* scalars,
                                           std::uint32_t* sweep_halt,
                                           RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return; // uniform for the whole slot block
    const int     lane = static_cast<int>(threadIdx.x);
    const double* pm   = partial + static_cast<long long>(m) * (3 * kMaxReduceBlocks);
    __shared__ double lane_sum[3];
    if (lane < 3) {
        const double* values = pm + lane * kMaxReduceBlocks;
        double        sum    = 0.0;
        for (int i = 0; i < blocks; ++i) sum = sum + values[i]; // strict index order
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    cmfd_wiel_apply(lane_sum[0], lane_sum[1], lane_sum[2], scalars, sweep_halt, m);
}

// ---------------------------------------------------------------------------
// RASBERY_GPU_CMFD_FUSE bit 2 (kFuseWiel): cmfd_wiel_stage1 +
// cmfd_wiel_finalize_chunked in ONE graph node.
//
// ORDER-PRESERVATION NOTE.  The same six-point argument as reduce_dot_fused,
// with three differences worth naming rather than leaving to the reader:
//
//  a. THREE reductions, not one, and they stay three.  Each keeps its own
//     accumulator, its own shared array, its own partial row and its own
//     strict ascending fold, exactly as in the pair -- this was never a fused
//     reduction and it is not one now.
//  b. THE TAIL IS NOT A ONE-THREAD TAIL.  cmfd_wiel_finalize_chunked folds on
//     three lanes and then calls cmfd_wiel_apply from lane 0, so the fused
//     kernel has to let the WHOLE last block through.  `sh_last` is written by
//     thread 0 and read after a __syncthreads every thread of every block
//     reaches, so the branch is BLOCK-UNIFORM -- which is the same property the
//     compaction guard relies on, and the reason a __syncthreads inside the
//     tail is safe.  The reference ran the tail in a 32-thread block and this
//     runs it in a kReduceThreads one; only lanes 0..2 do arithmetic in either,
//     and cmfd_wiel_apply is called by lane 0 in both.
//  c. THE TAIL WRITES sweep_halt, WHICH EVERY BLOCK READ AT ITS FIRST
//     INSTRUCTION.  That is not a race: a block reaches the atomicInc only
//     after it has read sweep_halt, and the tail runs only once every one of
//     the gridDim.x blocks has incremented.  So every read of sweep_halt in
//     this launch strictly precedes cmfd_wiel_apply's write of it -- the same
//     ordering the kernel boundary gave the pair.
//
// Saving: one node per CMFD sweep, under the chunked Wielandt fold only.  With
// RASBERY_GPU_WIEL_FOLD on the serial path there is no stage 1 and this bit is
// inert.
// ---------------------------------------------------------------------------
__global__ void cmfd_wiel_fused(const int nxyz,
                                const long long vec_stride,
                                const double* __restrict__ terms_ab,
                                const double* __restrict__ terms_c,
                                double* partial, ///< [slot][3][kMaxReduceBlocks]
                                double* scalars,
                                std::uint32_t* sweep_halt,
                                unsigned int* __restrict__ retire,
                                RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    __shared__ double sh_err[kReduceThreads];
    __shared__ double sh_gd[kReduceThreads];
    __shared__ double sh_gn[kReduceThreads];
    __shared__ bool   sh_last;
    // The finalize tail's three lanes.  Declared up here with the rest so no
    // __shared__ declaration sits behind a conditional return.
    __shared__ double lane_sum[3];

    const double* ta = terms_ab + m * vec_stride;
    const double* tc = terms_c + m * vec_stride;
    double*       pm = partial + static_cast<long long>(m) * (3 * kMaxReduceBlocks);

    // Fixed, contiguous chunk per block: depends only on (nxyz, gridDim.x).
    const int chunk = (nxyz + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, nxyz);

    double s_err = 0.0, s_gd = 0.0, s_gn = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x)) {
        s_err += ta[i];
        s_gd += ta[nxyz + i];
        s_gn += tc[i];
    }
    sh_err[threadIdx.x] = s_err;
    sh_gd[threadIdx.x]  = s_gd;
    sh_gn[threadIdx.x]  = s_gn;
    __syncthreads();

    // Fixed binary tree: identical operand pairing on every launch.
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            sh_err[threadIdx.x] += sh_err[threadIdx.x + stride];
            sh_gd[threadIdx.x] += sh_gd[threadIdx.x + stride];
            sh_gn[threadIdx.x] += sh_gn[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pm[0 * kMaxReduceBlocks + blockIdx.x] = sh_err[0];
        pm[1 * kMaxReduceBlocks + blockIdx.x] = sh_gd[0];
        pm[2 * kMaxReduceBlocks + blockIdx.x] = sh_gn[0];
        __threadfence();
        sh_last = (atomicInc(retire + m, gridDim.x - 1u) == gridDim.x - 1u);
    }
    __syncthreads();
    if (!sh_last) return;

    // ---- the former cmfd_wiel_finalize_chunked node, verbatim -------------
    const int              lane   = static_cast<int>(threadIdx.x);
    const int              blocks = static_cast<int>(gridDim.x);
    const volatile double* vpm    = pm;
    if (lane < 3) {
        const volatile double* values = vpm + lane * kMaxReduceBlocks;
        double                 sum    = 0.0;
        for (int i = 0; i < blocks; ++i) sum = sum + values[i]; // strict index order
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    cmfd_wiel_apply(lane_sum[0], lane_sum[1], lane_sum[2], scalars, sweep_halt, m);
}

/// diag = udiag - chif*xsnf*reigvs*vol, with the mined fused subtract.
__global__ void cmfd_updls(const int nxyz,
                           const long long vec_stride,
                           const long long node_stride,
                           const long long mat_stride,
                           const double* __restrict__ chif,
                           const double* __restrict__ xsnf,
                           const double* __restrict__ vol,
                           const double* __restrict__ udiag,
                           double* __restrict__ diag,
                           const double* __restrict__ scalars,
                           const std::uint32_t* __restrict__ sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    const double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    if (sm[kEshift] == 0.0) return;
    const int l = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (l >= nxyz) return;
    const double  reigvs = sm[kReigvs];
    const double  vl     = vol[m * node_stride + l];
    const double* cm     = chif + m * vec_stride;
    const double* xm     = xsnf + m * vec_stride;
    const double* um     = udiag + m * mat_stride;
    double*       dm     = diag + m * mat_stride;
    for (int ige = 0; ige < 2; ++ige)
        for (int igs = 0; igs < 2; ++igs) {
            const double c2 =
                __dmul_rn(__dmul_rn(cm[ige * nxyz + l], xm[igs * nxyz + l]), reigvs);
            const long long idx = static_cast<long long>(l) * 4 + ige * 2 + igs;
            dm[idx]             = fma(-c2, vl, um[idx]);
        }
}

/// Negative-flux census.  An integer-valued count in a double accumulates
/// exactly in any order, so a parallel atomic matches the host's serial scan.
__global__ void cmfd_negative_scan(const int n,
                                   const long long vec_stride,
                                   const double* __restrict__ phi,
                                   double* scalars,
                                   const std::uint32_t* __restrict__ sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (phi[m * vec_stride + i] < 0.0)
        atomicAdd(&scalars[static_cast<long long>(m) * kScalarCount + kNegative], 1.0);
}

/// The host loop's control tail: the all-negative reset, the retry rule that
/// refuses to count a negative-flux sweep against iout, and the exit tests.
__global__ void cmfd_sweep_end(double* scalars, std::uint32_t* sweep_halt,
                                   RASBERY_CMFD_SLOT_ARGS) {
    if (threadIdx.x != 0) return;
    RASBERY_CMFD_SLOT(m);
    if (sweep_halt[m] != 0u) return;
    double* sm  = scalars + static_cast<long long>(m) * kScalarCount;
    double  neg = sm[kNegative];
    if (neg == sm[kNgxyz]) neg = 0.0;
    const bool retry = (neg != 0.0) && (sm[kIcmfdDone] < sm[kIcmfdBudget]);
    if (!retry) sm[kSweepsDone] += 1.0;
    if (sm[kErrl2] < sm[kEpsl2]) {
        sm[kSweepState] = 1.0;
        sweep_halt[m]   = 1u;
        return;
    }
    if (sm[kSweepsDone] >= sm[kSweepBudget]) {
        sm[kSweepState] = 3.0;
        sweep_halt[m]   = 1u;
    }
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 10 part 2: the two kernels that let a sweep run inside a device
// outer segment without a host round trip.
// ---------------------------------------------------------------------------

/// Carry the outer segment's halt into the sweep's own mask.
///
/// A SEGMENT THAT HAS ALREADY EXITED MUST NOT RUN ANOTHER SWEEP.  The segment
/// stops enqueueing at its next observation, but the outer whose kernels were
/// already in flight when the previous outer's transition latched has to be a
/// no-op -- that is the halt-gate contract every other body kernel obeys
/// (CudaCmfdOuterKernels.h), and the sweep's own mask is `sweep_halt`.
///
/// It runs BETWEEN issueSweepUploads (which uploads the participation masks) and
/// launch_sweeps (whose every kernel tests the mask at its first instruction),
/// which is the only window in which the segment's halt is visible and the graph
/// has not started.  Raising sweep_halt here masks the whole captured graph
/// exactly the way an over-captured slot budget does in cmfd_sweep_begin.
__global__ void cmfd_sweep_gate(std::uint32_t* sweep_halt,
                                const std::uint32_t* __restrict__ outer_halt,
                                const int outer_slot, const int m) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (outer_halt != nullptr && outer_halt[outer_slot] != 0u) sweep_halt[m] = 1u;
}

/// Publish what the outer segment's transition has to rank, from the device.
///
/// THIS IS THE READBACK THAT ISN'T.  Without it the sweep hook has to drain the
/// stream, copy four scalars home and hand them back down as a probe -- the
/// per-outer round trip the segment exists to delete.  The four values are
/// already in device memory when the graph ends, and the segment's convergence
/// kernel reads them out of device memory, so the host was only ever a courier.
///
/// `negative` and `rayleigh` are the SAME two the host publishes today
/// (BICGCMFD::lastSweepNegativeFlux / lastSweepRayleigh, fed from
/// io.negative_last and io.state == 2), so the transition ranks the same signals
/// from the same numbers.
///
/// AN UNFINISHED DRIVE ENDS THE OUTER HERE.  States 0 (the launch's slot budget
/// ran out) and 2 (the Wielandt gamma degenerated) are the two the host `while`
/// loop in driveDeviceSweeps spins on: the drive is NOT over, so the flux this
/// outer's updjnet and upddhat would read is a half sweep.  Raising the
/// segment's halt stops the rest of the body dead, and the host finishes the
/// drive verbatim at the observation it was going to make anyway.
__global__ void cmfd_sweep_verdict(const double* __restrict__ scalars, const int m,
                                   double* eigv_out, double* residual_out,
                                   std::uint32_t* negative_out,
                                   std::uint32_t* rayleigh_out,
                                   std::uint32_t* nonfinite_out,
                                   std::uint32_t* outer_halt, const int outer_slot,
                                   double* acc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (outer_halt != nullptr && outer_halt[outer_slot] != 0u) return;
    const double* sm = scalars + static_cast<long long>(m) * kScalarCount;

    if (eigv_out != nullptr) *eigv_out = sm[kEigv];
    if (residual_out != nullptr) *residual_out = sm[kErrl2];
    if (negative_out != nullptr) *negative_out = (sm[kNegative] != 0.0) ? 1u : 0u;
    // Per outer, exactly as the host probe publish this replaces was.
    if (nonfinite_out != nullptr) *nonfinite_out = 0u;

    const int state = static_cast<int>(sm[kSweepState]);
    if (rayleigh_out != nullptr) *rayleigh_out = (state == 2) ? 1u : 0u;

    // =====================================================================
    // Rev.7.1 Task 10 part 3: THE OBSERVATION THE HOST NO LONGER HAS TO MAKE
    // =====================================================================
    //
    // BEHIND THE SAME EARLY RETURN AS EVERYTHING ELSE HERE, and that is what
    // makes the total mean what the host reconstruction needs it to mean: a
    // launch enqueued behind a halt that had already fired ran nothing, so it
    // contributes nothing.  `launches` is therefore the count of drives this
    // segment actually performed, not the count it submitted.
    //
    // ONE THREAD, NO ATOMICS.  Every launch of one slot is serialised by the
    // stream it rides, and a slot is never in two launches at once (the
    // enqueue path holds the arena's stream claim), so the read-modify-write
    // below has no second writer.
    if (acc != nullptr) {
        acc[kAccAttempts] += sm[kIcmfdDone];
        acc[kAccSweeps] += sm[kSweepsDone];
        acc[kAccLaunches] += 1.0;
        acc[kAccState]    = sm[kSweepState];
        acc[kAccNegative] = (sm[kNegative] != 0.0) ? 1.0 : 0.0;
        // THE ABANDONED LAUNCH, SAVED WHERE IT IS STILL READABLE.  States 0 and
        // 2 hand the drive back to the host, and the host finishes it from THIS
        // block -- eigv, reigv, reigvs, errl2, icmfd_done, and the three
        // exported wiel sums.  The outers enqueued behind this one will each
        // upload their own staged block over it before the host ever gets to
        // look, so a copy taken here is the only copy there is.
        if (state == 0 || state == 2) {
            acc[kAccExceptional] = 1.0;
            for (int i = 0; i < kSweepCount; ++i) acc[kAccSaved + i] = sm[kSweepFirst + i];
        }
    }

    if (outer_halt != nullptr && (state == 0 || state == 2)) outer_halt[outer_slot] = 1u;
}

/// Rev.7.1 Task 10 part 3: THE NEXT SWEEP'S EIGENVALUE, WHERE IT ALREADY IS.
///
/// THE ONLY HOST INPUT A DEFERRED OUTER STILL NEEDED.  BICGCMFD::setls is a
/// no-op on the device-assembly arm (it records `_device_assembly_pending` and
/// returns), so of everything stageSweepIO writes into the staged block only
/// eigv, its two reciprocals and the residual carry a value the PREVIOUS
/// outer's observation would have produced.  Everything else -- epsl2, eshift,
/// the budgets, the array pointers -- is constant across a segment.
///
/// So the block is staged and uploaded exactly as it always was, and this
/// kernel then overwrites those four from the device probe the previous outer's
/// verdict wrote.  Nothing about the launch changes; the host simply stops being
/// the courier.
///
/// THE ARITHMETIC IS THE HOST'S, SPELLED THE HOST'S WAY.  BICGCMFD::enqueueDrive
/// computes `1. / eigv` and `(_eshift != 0.0) ? 1. / (eigv + _eshift) : 0.0`;
/// __ddiv_rn is IEEE division with round-to-nearest, which is what the host's
/// `/` is, and the add is a plain add.  Same inputs, same operations, same bits.
///
/// NOT HALT-GATED, for k_outer_publish_reigv's reason (exactness invariant 7):
/// it advances nothing, and gating it would only leave a stale value standing
/// in a block whose H2D has already overwritten everything else anyway.
__global__ void cmfd_sweep_patch(double* scalars, const int m,
                                 const double* __restrict__ probe_eigv,
                                 const double* __restrict__ probe_residual) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    if (probe_eigv != nullptr) {
        const double eigv = *probe_eigv;
        sm[kEigv]         = eigv;
        sm[kReigv]        = __ddiv_rn(1.0, eigv);
        sm[kReigvs] =
            (sm[kEshift] != 0.0) ? __ddiv_rn(1.0, eigv + sm[kEshift]) : 0.0;
    }
    if (probe_residual != nullptr) sm[kErrl2] = *probe_residual;
}

// ---------------------------------------------------------------------------
// RASBERY_GPU_CMFD_FUSE bit 3 (kFuseSweepPre): cmfd_sweep_gate +
// cmfd_sweep_patch in ONE launch.
//
// ORDER-PRESERVATION NOTE.  Both references are <<<1, 1>>> one-thread kernels
// issued back to back in the same window (after issueSweepUploads, before
// launch_sweeps).  One thread running the gate body and then the patch body IS
// the stream order of the two nodes it replaces, so "same operations, same
// order" is immediate.  They also touch DISJOINT memory -- the gate writes
// sweep_halt[m], the patch writes this slot's scalar block -- so neither can
// observe the other and the concatenation cannot introduce a read-after-write
// the pair did not have.  The patch arithmetic is copied verbatim, __ddiv_rn
// included, so the rounding of the two reciprocals is unchanged.
//
// The guards are copied too, and that matters: the reference gate is a no-op
// when outer_halt is null, and the reference patch is not issued at all when
// patch_from_probe is false.  Passing null probe pointers reproduces the second
// exactly, and enqueueSweepPreamble declines to launch anything when NEITHER is
// wanted -- which is the only case where the pair enqueued zero nodes.
//
// Saving: one LAUNCH per device-outer sweep drive.  These two sit outside the
// captured graph, so this bit lowers launches_per_outer without changing any
// graph node count.
// ---------------------------------------------------------------------------
__global__ void cmfd_sweep_gate_patch(std::uint32_t* sweep_halt,
                                      const std::uint32_t* __restrict__ outer_halt,
                                      const int outer_slot, const int m,
                                      double* scalars,
                                      const double* __restrict__ probe_eigv,
                                      const double* __restrict__ probe_residual) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // ---- cmfd_sweep_gate, verbatim ----------------------------------------
    if (outer_halt != nullptr && outer_halt[outer_slot] != 0u) sweep_halt[m] = 1u;
    // ---- cmfd_sweep_patch, verbatim ---------------------------------------
    // NOT halt-gated in the reference either, so the concatenation does not
    // acquire a guard the pair lacked (exactness invariant 7).
    double* sm = scalars + static_cast<long long>(m) * kScalarCount;
    if (probe_eigv != nullptr) {
        const double eigv = *probe_eigv;
        sm[kEigv]         = eigv;
        sm[kReigv]        = __ddiv_rn(1.0, eigv);
        sm[kReigvs] =
            (sm[kEshift] != 0.0) ? __ddiv_rn(1.0, eigv + sm[kEshift]) : 0.0;
    }
    if (probe_residual != nullptr) sm[kErrl2] = *probe_residual;
}

/// WP7 stage B: the CMFD GRAPH NODE CENSUS, measured rather than counted by hand.
///
/// Emitted once per successful cudaGraphInstantiate on either CMFD graph, so a
/// FUSE=0 run and a FUSE=<bits> run of the same deck print directly comparable
/// lines and the node reduction is a fact instead of an estimate.  Graph
/// instantiations are rare by construction (a bounded bucket ladder, and
/// graph_reinstantiations is 0 on a normal run), so this is a handful of lines
/// per run, not a per-iteration log.
///
/// TRAJECTORY-NEUTRAL BY CONSTRUCTION.  cudaGraphGetNodes and
/// cudaGraphNodeGetType are host-side queries on an already-built cudaGraph_t:
/// nothing is enqueued, nothing is synchronised, no device memory is read and
/// the graph is not modified.  A query failure is swallowed (the error is
/// cleared) rather than turned into a solver failure.
///
/// FIELD MEANINGS (docs/WP7_CMFD_GRAPH_CENSUS_20260831_KO.md Sec 2):
///   nodes                total nodes in the instantiated graph.
///   sweeps               CMFD sweeps the capture carries (1 for the outer graph,
///                        which IS one sweep's inner solve).
///   nodes_per_sweep      "sweep": (nodes - 1) / sweeps -- the assemble node is
///                        once per LAUNCH, not per sweep.  "outer": nodes.
///   launches_per_outer   what one CMFD outer costs in dispatches.  One sweep
///                        carries exactly one outer, so for both graphs this is
///                        nodes_per_sweep.  It is the number stage B lowers.
///   block_threads        WP17: the width of the five per-iteration elementwise
///                        classes; equal to the arena's block_size unless
///                        RASBERY_GPU_CMFD_BLOCK narrowed it.
///   node_blocks          blocks_per_launch for the per-NODE classes.
///   vector_blocks        blocks_per_launch for the per-ELEMENT class.
///   persistent_arm       whether the cooperative single-launch iteration ran.
static void reportCmfdGraphCensus(const char* tag, cudaGraph_t graph, int sweeps,
                                  int iterations_per_outer, unsigned fuse_mask,
                                  int block_threads, int node_blocks,
                                  int vector_blocks, bool persistent_arm) {
    if (graph == nullptr) return;
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        return;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    unsigned long long kernels = 0, memcpies = 0, memsets = 0, other = 0;
    for (size_t i = 0; i < count; ++i) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) {
            cudaGetLastError();
            ++other;
        } else if (type == cudaGraphNodeTypeKernel) {
            ++kernels;
        } else if (type == cudaGraphNodeTypeMemcpy) {
            ++memcpies;
        } else if (type == cudaGraphNodeTypeMemset) {
            ++memsets;
        } else {
            ++other;
        }
    }
    // The sweep graph opens with ONE cmfd_assemble_operator_2g per launch; the
    // outer graph has no such prologue and is itself the per-sweep unit.
    const unsigned long long per =
        sweeps > 0 ? static_cast<unsigned long long>(count - 1) /
                         static_cast<unsigned long long>(sweeps)
                   : static_cast<unsigned long long>(count);
    std::ostringstream line;
    line << "[RASBERY][CMFD][GRAPH] {\"graph\":\"" << tag << "\""
         << ",\"nodes\":" << count
         << ",\"sweeps\":" << (sweeps > 0 ? sweeps : 1)
         << ",\"iterations_per_outer\":" << iterations_per_outer
         << ",\"nodes_per_sweep\":" << per
         << ",\"kernel_nodes\":" << kernels
         << ",\"memcpy_nodes\":" << memcpies
         << ",\"memset_nodes\":" << memsets
         << ",\"other_nodes\":" << other
         << ",\"launches_per_outer\":" << per
         << ",\"block_threads\":" << block_threads
         << ",\"node_blocks\":" << node_blocks
         << ",\"vector_blocks\":" << vector_blocks
         << ",\"persistent_arm\":" << (persistent_arm ? 1 : 0)
         << ",\"fuse_mask\":" << fuse_mask << "}";
    std::cout << line.str() << std::endl;
}

// ---------------------------------------------------------------------------
// BatchCore -- the device side of both execution modes.
//
// `slots` is 1 for a plain single-instance run and M for the batch mode; the
// kernels above do not know the difference, they simply see gridDim.y.
// ---------------------------------------------------------------------------
class BatchCore {
public:
    // -----------------------------------------------------------------------
    // Bulk-upload elision.
    //
    // reset() used to push diag, cc, phi and src across PCIe on every single
    // CMFD outer.  Two of the four are provably redundant in the steady state:
    //
    //   * cc (the 12-per-node coupling coefficients, 60 % of the traffic) is
    //     only rewritten by upddtil/upddhat, which run *between* drive() calls,
    //     never inside the outer loop.
    //   * phi is owned by the device from reset() until synchronize() copies it
    //     back; re-uploading the very bytes we just downloaded is a round trip
    //     for nothing.
    //
    // Rather than thread a generation counter through CMFD/Nodal/Driver (and
    // risk one forgotten call site silently feeding the GPU a stale operator),
    // the backend keeps a host-side shadow of what the device currently holds
    // and elides the copy only when the incoming buffer is *bit-identical* to
    // it.  A local memcmp over host DRAM is several times cheaper than the DMA
    // it replaces, and it cannot go wrong: a byte that differs anywhere forces
    // the upload.  This makes the optimisation numerically inert by
    // construction, which is what the h5 bit-equality gate demands.
    //
    // In batch mode each slot keeps its own shadow, so instance m's elision
    // decision is a pure function of instance m's own history.
    // -----------------------------------------------------------------------
    struct MirroredUpload {
        std::vector<double> shadow;
        bool                valid = false;
    };

    /// Everything the arena remembers about one instance between the moment it
    /// hands its buffers over and the moment its flux comes back.
    struct Slot {
        const double* host_diag = nullptr;
        const double* host_cc   = nullptr;
        double*       host_diag_out = nullptr;
        double*       host_cc_out   = nullptr;
        const double* host_phi  = nullptr;
        const double* host_src  = nullptr;
        bool          push_diag = true;
        bool          push_cc   = true;
        bool          push_phi  = true;
        MirroredUpload diag_mirror;
        MirroredUpload cc_mirror;
        MirroredUpload phi_mirror;
        double        eps           = std::numeric_limits<double>::quiet_NaN();
        double        eps_on_device = std::numeric_limits<double>::quiet_NaN();
        int           nmax          = -1;
        double*       out_phi       = nullptr;
        bool          in_use        = false;
        bool          nonfinite     = false; ///< device flagged THIS slot's flux

        // ---- sweep-mode staging (RASBERY_GPU_CMFD_SWEEP) ----
        const double* host_chif  = nullptr;
        const double* host_xsnf  = nullptr;
        const double* host_xsrf  = nullptr;
        const double* host_xssm  = nullptr;
        const double* host_dtil  = nullptr;
        const double* host_dhat  = nullptr;
        const double* host_vol   = nullptr;
        double*       host_udiag = nullptr;
        double*       host_psi   = nullptr; ///< in/out
        bool          push_psi   = true;    ///< CmfdSweepIO::psi_dirty
        /// Rev.7.1 Task 9 link 2: the device outer segment owns these in the
        /// arena, so their H2D is skipped -- see CmfdSweepIO.
        bool          dhat_resident = false;
        bool          psi_resident  = false;
        bool          psi_downloaded = false; ///< D2H issued for THIS launch
        bool          device_assembly = false;
        bool          pushed_xsrf = false;
        bool          pushed_xssm = false;
        bool          pushed_xsnf = false;
        bool          pushed_dtil = false;
        cuda_transfer::ByteExactMirror<double> xsrf_mirror;
        cuda_transfer::ByteExactMirror<double> xssm_mirror;
        cuda_transfer::ByteExactMirror<double> xsnf_mirror;
        cuda_transfer::ByteExactMirror<double> dtil_mirror;
        MirroredUpload chif_mirror;
        MirroredUpload vol_mirror;
        // ---- Rev.7.1 Task 10 part 4 precondition (a): PINNED, NOT INLINE ----
        //
        // These two used to be `double[kSweepCount]` VALUE arrays inside the
        // Slot, which put them in the std::vector<Slot>'s pageable storage --
        // and both are cudaMemcpyAsync endpoints on the sweep's hot path
        // (issueSweepUploads' H2D, issueSweepDownloads' D2H, one of each per
        // outer).  A pageable async copy is not asynchronous: the driver
        // stages it through its own bounce buffer and the call blocks.  It is
        // also NOT RECORDABLE -- a cudaMemcpyAsync from pageable memory issued
        // on a capturing stream is refused and invalidates the capture, which
        // is what stopped the outer body from being captured at all.
        //
        // They now point into BatchCore::host_sweep_scalars, one 2*kSweepCount
        // lane per slot, cudaMallocHost'd once at stand-up.  Every one of the
        // 22 use sites reads them exactly as before (an array name was already
        // a pointer at every one of them); what changed is where the bytes
        // live.  bindSweepLanes() is the single place that sets them, and
        // acquireSlot re-runs it after its whole-struct reset -- `Slot{}` sets
        // both to nullptr, and the reset audit (batchSlotIsReset) tests for
        // that rather than trusting it.
        double*       sweep_in               = nullptr;
        double*       sweep_out              = nullptr;
        int           sweep_unroll           = 0;
    };

    explicit BatchCore(Geometry& geometry, int slot_count)
        : slots(slot_count),
          nxyz(geometry.nxyz()),
          n(geometry.ngxyz()),
          surface_group_count(static_cast<size_t>(geometry.nsurf()) * geometry.ng()),
          matrix_count(static_cast<size_t>(geometry.ng2()) * geometry.nxyz()),
          coupling_count(static_cast<size_t>(geometry.ng()) * NDIRMAX * LR * geometry.nxyz()) {
        if (geometry.ng() != 2 || NDIRMAX * LR != 6) {
            status = "CUDA backend currently requires a two-group, six-neighbor CMFD system";
            return;
        }
        // The fused kernels walk the vector domain from the NODE grid: thread
        // l owns elements 2*l and 2*l+1 of a node-major two-group vector.  The
        // coverage is exact only when n is exactly twice nxyz, so make the
        // assumption a checked precondition rather than an implication of the
        // ng == 2 test above.
        if (n != 2 * nxyz) {
            status = "CUDA backend requires ngxyz == 2 * nxyz for the node-major two-group layout";
            return;
        }
        if (slots < 1) {
            status = "CUDA batch arena needs at least one slot";
            return;
        }

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            status = "no CUDA device is visible";
            cudaGetLastError();
            return;
        }

        // Rev.7.1 Task 18d.  Everything below allocates, page-locks or copies
        // synchronously, and in a batch it runs on a Driver thread while OTHER
        // Drivers may already be capturing the arena's graphs.  One window for
        // the whole stand-up rather than one per call: the arbiter is a
        // shared/exclusive lock, so this neither serialises the decks against
        // each other nor costs anything once the run is warm.
        rasbery::AllocWindow arena_standup("cmfd.arena.standup");
        try {
            int device = 0;
            CUDA_CHECK(cudaGetDevice(&device));
            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            status = properties.name;

            if (const char* block_env = std::getenv("RASBERY_GPU_BLOCK_SIZE")) {
                const int requested = std::atoi(block_env);
                constexpr int candidates[] = {64, 128, 192, 256};
                if (std::find(std::begin(candidates), std::end(candidates), requested) ==
                    std::end(candidates))
                    throw std::runtime_error(
                        "RASBERY_GPU_BLOCK_SIZE must be one of 64, 128, 192, 256");
                block_size = requested;
            }

            std::vector<int> host_neighbors(static_cast<size_t>(nxyz) * 6);
            for (int l = 0; l < nxyz; ++l)
                for (int idir = 0; idir < NDIRMAX; ++idir)
                    for (int lr = 0; lr < LR; ++lr)
                        host_neighbors[6 * l + idir * LR + lr] = geometry.neib(lr, idir, l);

            // Keep an immutable host copy of the topology used to build the
            // device neighbour list.  A batch slot may vary coefficients and
            // flux, but it must not silently reuse this list for a different
            // same-sized core map.
            topology_neighbors = host_neighbors;

            std::vector<int> host_node_surface(static_cast<size_t>(nxyz) * 6);
            std::vector<double> host_face_area(static_cast<size_t>(nxyz) * NDIRMAX);
            std::vector<double> host_geometry_volume(static_cast<size_t>(nxyz));
            for (int l = 0; l < nxyz; ++l) {
                const double hx = geometry.hmesh(XDIR, l);
                const double hy = geometry.hmesh(YDIR, l);
                const double hz = geometry.hmesh(ZDIR, l);
                host_face_area[static_cast<size_t>(l) * NDIRMAX + XDIR] = hy * hz;
                host_face_area[static_cast<size_t>(l) * NDIRMAX + YDIR] = hx * hz;
                host_face_area[static_cast<size_t>(l) * NDIRMAX + ZDIR] = hx * hy;
                host_geometry_volume[static_cast<size_t>(l)] = geometry.vol(l);
                for (int idir = 0; idir < NDIRMAX; ++idir)
                    for (int lr = 0; lr < LR; ++lr)
                        host_node_surface[static_cast<size_t>(l) * 6 + idir * LR + lr] =
                            geometry.lktosfc(lr, idir, l);
            }
            topology_node_surface = host_node_surface;
            topology_face_area = host_face_area;
            topology_volume = host_geometry_volume;

            // Greedy BFS graph colouring.  On a bipartite lattice the
            // smallest-available-colour rule reproduces the historical
            // red/black parity colouring exactly (same seeds, same queue
            // order), so existing cores keep a bit-identical sweep.  The
            // 90-degree rotational quarter-core fold stitches node (0,t) to
            // node (t,0) -- a same-parity edge that makes the graph
            // non-bipartite -- and there the greedy rule simply opens a
            // third (or further) colour instead of refusing the core.
            std::vector<int> host_colors(static_cast<size_t>(nxyz), -1);
            int              host_ncolors = 1;
            std::queue<int>  frontier;
            for (int seed = 0; seed < nxyz; ++seed) {
                if (host_colors[seed] >= 0) continue;
                host_colors[seed] = 0;
                frontier.push(seed);
                while (!frontier.empty()) {
                    const int l = frontier.front();
                    frontier.pop();
                    for (int slot = 0; slot < 6; ++slot) {
                        const int neighbor = host_neighbors[6 * l + slot];
                        if (neighbor < 0 || host_colors[neighbor] >= 0) continue;
                        unsigned used = 0;
                        for (int s2 = 0; s2 < 6; ++s2) {
                            const int nb2 = host_neighbors[6 * neighbor + s2];
                            // A self-edge (the centre node's rotational closure maps its
                            // west/north face onto itself) constrains nothing: the sweep's
                            // self term reads the node's own previous iterate.
                            if (nb2 >= 0 && nb2 != neighbor && host_colors[nb2] >= 0)
                                used |= 1u << host_colors[nb2];
                        }
                        int c = 0;
                        while (used & (1u << c)) ++c;
                        host_colors[neighbor] = c;
                        host_ncolors          = std::max(host_ncolors, c + 1);
                        frontier.push(neighbor);
                    }
                }
            }
            for (int l = 0; l < nxyz; ++l)
                for (int slot = 0; slot < 6; ++slot) {
                    const int nb = host_neighbors[6 * l + slot];
                    if (nb >= 0 && nb != l && host_colors[nb] == host_colors[l])
                        throw std::runtime_error("CMFD sweep colouring failed: adjacent nodes share a colour");
                }
            ncolors = std::max(host_ncolors, 2);

            if (const char* sweep_env = std::getenv("RASBERY_GPU_RB_SWEEPS"))
                rb_sweeps = std::max(0, std::atoi(sweep_env));
            if (const char* graph_env = std::getenv("RASBERY_GPU_GRAPH"))
                use_graph = std::string(graph_env) != "0";
            if (const char* batch_env = std::getenv("RASBERY_GPU_ITER_BATCH")) {
                const int requested = std::atoi(batch_env);
                if (requested > 0) iter_batch_request = requested;
            }
            scalar_fusion = cmfdScalarFusionEnabled();
            fuse_mask      = cmfdFuseMask();
            fuse_dot       = (fuse_mask & kFuseDot) != 0u;
            fuse_dot2      = (fuse_mask & kFuseDot2) != 0u;
            fuse_wiel      = (fuse_mask & kFuseWiel) != 0u;
            fuse_sweep_pre = (fuse_mask & kFuseSweepPre) != 0u;
            fp32_inner    = cmfdFp32InnerEnabled();
            telemetry.fp32_active = fp32_inner ? 1u : 0u;
            // WP17.  Both are latched here, once, for the same reason the fuse
            // mask is: the block width and the persistent arm are part of the
            // captured topology and may not change between two outers.
            cmfd_block = cmfdBlockThreads();
            armPersistent(properties);
            status += " (block=" + std::to_string(block_size) +
                      ", cmfd block=" + std::to_string(cmfd_block_threads()) +
                      ", persistent=" +
                      (persistent_armed ? std::string("on")
                                        : std::string("off:") +
                                              persistentRefusalName(persistent_refusal)) +
                      ", RB sweeps=" + std::to_string(rb_sweeps) +
                      ", graph=" + (use_graph ? "on" : "off") +
                      ", iter batch=" +
                      (iter_batch_request > 0 ? std::to_string(iter_batch_request)
                                              : std::string("auto")) +
                      ", assembly=" + (cmfdAssemblyEnabled() ? "on" : "off") +
                      ", scalar fusion=" + (scalar_fusion ? "on" : "off") +
                      ", fuse=" + std::to_string(fuse_mask) +
                      // The [PHYSICS_MODE] receipt for the inner-solve precision.
                      // fp64 = the historical all-double path; mixed = FP32 inner
                      // BiCGSTAB under an FP64 outer correction.
                      ", precision=" + (fp32_inner ? "mixed" : "fp64") +
                      ", slots=" + std::to_string(slots) + ")";

            const size_t S = static_cast<size_t>(slots);
            allocate(reinterpret_cast<void**>(&neighbors), host_neighbors.size() * sizeof(int));
            CUDA_CHECK(rasbery::xfer::memcpy("CudaBICGBackend.cu:BatchCore::init", "neighbors", neighbors,
                                  host_neighbors.data(),
                                  host_neighbors.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&colors), host_colors.size() * sizeof(int));
            CUDA_CHECK(rasbery::xfer::memcpy("CudaBICGBackend.cu:BatchCore::init", "colors", colors,
                                  host_colors.data(),
                                  host_colors.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&assembly_node_surface),
                     host_node_surface.size() * sizeof(int));
            CUDA_CHECK(rasbery::xfer::memcpy("CudaBICGBackend.cu:BatchCore::init",
                                  "assembly_node_surface", assembly_node_surface,
                                  host_node_surface.data(),
                                  host_node_surface.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&assembly_face_area),
                     host_face_area.size() * sizeof(double));
            CUDA_CHECK(rasbery::xfer::memcpy("CudaBICGBackend.cu:BatchCore::init",
                                  "assembly_face_area", assembly_face_area,
                                  host_face_area.data(),
                                  host_face_area.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&assembly_volume),
                     host_geometry_volume.size() * sizeof(double));
            CUDA_CHECK(rasbery::xfer::memcpy("CudaBICGBackend.cu:BatchCore::init",
                                  "assembly_volume", assembly_volume,
                                  host_geometry_volume.data(),
                                  host_geometry_volume.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&diag), S * matrix_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&dinv), S * matrix_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&cc), S * coupling_count * sizeof(double));
            const size_t vec_bytes = S * static_cast<size_t>(n) * sizeof(double);
            allocate(reinterpret_cast<void**>(&src), vec_bytes);
            allocate(reinterpret_cast<void**>(&phi), vec_bytes);
            allocate(reinterpret_cast<void**>(&r), vec_bytes);
            allocate(reinterpret_cast<void**>(&r0), vec_bytes);
            allocate(reinterpret_cast<void**>(&p), vec_bytes);
            allocate(reinterpret_cast<void**>(&v), vec_bytes);
            allocate(reinterpret_cast<void**>(&s), vec_bytes);
            allocate(reinterpret_cast<void**>(&t), vec_bytes);
            allocate(reinterpret_cast<void**>(&y), vec_bytes);
            allocate(reinterpret_cast<void**>(&z), vec_bytes);
            allocate(reinterpret_cast<void**>(&ax), vec_bytes);
            allocate(reinterpret_cast<void**>(&partials),
                     S * static_cast<size_t>(kMaxReduceBlocks) * sizeof(double));
            // Second landing pad so the paired (s.t, t.t) reduction can keep
            // one partial array per dot product -- same layout, same stride,
            // so each dot's stage-2 fold is the identical strict index walk.
            allocate(reinterpret_cast<void**>(&partials2),
                     S * static_cast<size_t>(kMaxReduceBlocks) * sizeof(double));
            // Landing pad for the OPT-IN chunked Wielandt fold: three rows per
            // slot, one per sum, so each keeps the partial array and the strict
            // stage-2 walk it would have had alone.  Its OWN buffer rather than
            // a borrowed corner of `partials`: that one is live across the
            // inner BiCGSTAB of the same sweep, and "dead by inspection right
            // now" is not a property worth betting a campaign arm on.  384 KB
            // at 64 slots, and it is allocated whatever the mode so that
            // flipping RASBERY_GPU_WIEL_FOLD never changes the arena's shape.
            allocate(reinterpret_cast<void**>(&wiel_partials),
                     S * 3u * static_cast<size_t>(kMaxReduceBlocks) * sizeof(double));
            // WP7 stage B.  Allocated and zeroed whatever RASBERY_GPU_CMFD_FUSE
            // says, so the arena's footprint is mask-independent; `slots`
            // uint32s.  The zero here is the ONLY one it ever needs -- see the
            // self-rearming argument at reduce_dot_fused point 6.
            allocate(reinterpret_cast<void**>(&fuse_retire), S * sizeof(unsigned int));
            CUDA_CHECK(cudaMemset(fuse_retire, 0, S * sizeof(unsigned int)));
            allocate(reinterpret_cast<void**>(&scalars), S * kScalarCount * sizeof(double));
            allocate(reinterpret_cast<void**>(&device_flags), S * sizeof(std::uint32_t));
            allocate(reinterpret_cast<void**>(&iter_flags), S * sizeof(std::uint32_t));
            allocate(reinterpret_cast<void**>(&device_halt), S * sizeof(std::uint32_t));
            allocate(reinterpret_cast<void**>(&device_active), S * sizeof(std::uint32_t));
            allocate(reinterpret_cast<void**>(&device_counters),
                     S * kCounterSlots * sizeof(std::uint32_t));
            allocate(reinterpret_cast<void**>(&device_status), S * sizeof(DeviceSolveStatus));
            allocate(reinterpret_cast<void**>(&xs_chif), vec_bytes);
            allocate(reinterpret_cast<void**>(&xs_xsnf), vec_bytes);
            allocate(reinterpret_cast<void**>(&xs_xsrf), vec_bytes);
            allocate(reinterpret_cast<void**>(&xs_xssm), S * matrix_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&dtil_dev), S * surface_group_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&dhat_dev), S * surface_group_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&node_vol), S * static_cast<size_t>(nxyz) * sizeof(double));
            allocate(reinterpret_cast<void**>(&udiag_dev), S * matrix_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&psi_dev), S * static_cast<size_t>(nxyz) * sizeof(double));
            // The FP32 working set exists only when the mixed-precision inner
            // loop is armed, so the default configuration pays neither the
            // allocation nor the footprint.
            if (fp32_inner) {
                const size_t vec_f_bytes = S * static_cast<size_t>(n) * sizeof(float);
                allocate(reinterpret_cast<void**>(&diag_f), S * matrix_count * sizeof(float));
                allocate(reinterpret_cast<void**>(&dinv_f), S * matrix_count * sizeof(float));
                allocate(reinterpret_cast<void**>(&cc_f), S * coupling_count * sizeof(float));
                allocate(reinterpret_cast<void**>(&r_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&r0_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&p_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&v_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&s_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&t_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&y_f), vec_f_bytes);
                allocate(reinterpret_cast<void**>(&z_f), vec_f_bytes);
            }
            allocate(reinterpret_cast<void**>(&sweep_halt), S * sizeof(std::uint32_t));
            allocate(reinterpret_cast<void**>(&device_assembly_active),
                     S * sizeof(std::uint32_t));
            CUDA_CHECK(cudaMemset(sweep_halt, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemset(device_assembly_active, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemset(device_halt, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemset(device_active, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_status),
                                      S * sizeof(DeviceSolveStatus)));
            // ---- Rev.7.1 Task 18: ONE STAGING LANE PER SLOT, PLUS ONE -----
            //
            // These four are the FLEET-WIDE masks, and every one of them is the
            // page-locked SOURCE of a cudaMemcpyAsync.  With one copy of each
            // there could only ever be one launch in flight: a second launcher
            // rewriting the mask while the first one's DMA was still reading it
            // is the hazard the sweep_halt snapshot buffer below already
            // documents, and it is why the stream-ordered enqueue path refused
            // batch mode outright.
            //
            // The device buffers do not need this and deliberately do not get
            // it: everything reaches the device through ONE stream, so the
            // uploads of two launches are strictly ordered and cannot overlap.
            // It is only the HOST side that has no such ordering, because the
            // host runs ahead of the stream.  Lane `slots` belongs to the
            // rendezvous launcher (one at a time, by the `launching` claim);
            // lane m belongs to slot m's own stream-ordered enqueue, which is
            // the only path that can have several launches outstanding.
            //
            // slots+1 lanes of slots words each: 65 x 64 x 4 bytes at the widest
            // configuration this tree supports.
            const std::size_t SL = (S + 1) * S;
            stage_lane           = slots;
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_active),
                                      SL * sizeof(std::uint32_t)));
            std::memset(host_active, 0, SL * sizeof(std::uint32_t));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_assembly_active),
                                      SL * sizeof(std::uint32_t)));
            std::memset(host_assembly_active, 0, SL * sizeof(std::uint32_t));
            // SNAPSHOT BUFFER for the sweep mask.  issueSweepUploads used to
            // build the participation mask in host_active, upload it, and then
            // INVERT THAT SAME BUFFER IN PLACE for the sweep_halt upload -- a
            // host write to the source of a cudaMemcpyAsync that had not been
            // synchronised.  host_active is cudaMallocHost'd, so that copy is a
            // real asynchronous DMA and the driver is entitled to read the
            // bytes after the inversion has already landed.  One buffer per
            // upload is the fix; see issueSweepUploads.
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_sweep_halt),
                                      SL * sizeof(std::uint32_t)));
            std::memset(host_sweep_halt, 0, SL * sizeof(std::uint32_t));

            // Rev.7.1 Task 10 part 4 precondition (a).  The sweep scalar block,
            // page-locked.  TWO lanes per slot -- in then out, adjacent -- not
            // one lane per LAUNCH like the masks above: unlike host_active this
            // block is not rewritten by a second launcher while a first one's
            // DMA reads it, because a slot has one tenant and a tenant's
            // launches are ordered by the arena's own stream.  The
            // per-launcher lanes exist for the FLEET-WIDE masks, which every
            // launcher writes; this is per-slot state and stays per-slot.
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_sweep_scalars),
                                      static_cast<std::size_t>(S) * 2u *
                                          static_cast<std::size_t>(kSweepCount) *
                                          sizeof(double)));
            std::memset(host_sweep_scalars, 0,
                        static_cast<std::size_t>(S) * 2u *
                            static_cast<std::size_t>(kSweepCount) * sizeof(double));

            // The lane -> slot map.  Allocated once at the FULL fleet width
            // whatever a launch's bucket turns out to be, so d_slot_map is a
            // fixed address a captured graph can bake, and seeded with the
            // identity so a launch that never calls buildSlotMap (a direct
            // enqueue, a test) behaves exactly as the pre-compaction code did.
            allocate(reinterpret_cast<void**>(&d_slot_map), S * sizeof(int));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_slot_map),
                                      SL * sizeof(int)));
            for (std::size_t i = 0; i < SL; ++i)
                h_slot_map[i] = static_cast<int>(i % S);
            CUDA_CHECK(rasbery::xfer::memcpy("CudaBICGBackend.cu:BatchCore::init",
                                  "d_slot_map seed", d_slot_map, stageSlotMap(),
                                  S * sizeof(int), cudaMemcpyHostToDevice));
            lanes = slots;

            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUBLAS_CHECK(cublasCreate(&handle));
            CUBLAS_CHECK(cublasSetStream(handle, stream));
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
            slot.resize(static_cast<size_t>(slots));
            for (int m = 0; m < slots; ++m) bindSweepLanes(m);
            available = true;
        } catch (const std::exception& error) {
            status = error.what();
            release();
        }
    }

    ~BatchCore() { release(); }

    void allocate(void** pointer, size_t bytes) {
        rasbery::AllocWindow window("cmfd.arena.malloc");
        CUDA_CHECK(cudaMalloc(pointer, bytes));
    }

    /// Point slot m's sweep scalar block at its pinned lane and zero it.
    ///
    /// THE ONLY WRITER OF THESE TWO POINTERS, and it is called from exactly
    /// two places: stand-up, and acquireSlot after the whole-struct reset.  A
    /// third caller would be a slot rebinding under a launch whose DMA reads
    /// the lane, which is why there is not one.
    void bindSweepLanes(int m) {
        if (host_sweep_scalars == nullptr) return;
        Slot& sl = slot[static_cast<std::size_t>(m)];
        sl.sweep_in =
            host_sweep_scalars + static_cast<std::size_t>(m) * 2u *
                                     static_cast<std::size_t>(kSweepCount);
        sl.sweep_out = sl.sweep_in + kSweepCount;
        // The bytes `double sweep_in[kSweepCount] = {}` used to give: the
        // reset audit reads them, and a stale lane would make a recycled slot
        // look like the previous tenant's.
        std::memset(sl.sweep_in, 0, 2u * static_cast<std::size_t>(kSweepCount) *
                                        sizeof(double));
    }

    void release() {
        // cudaFree is a synchronising API and the arena outlives some decks:
        // a teardown that overlaps a surviving deck's capture invalidates it.
        rasbery::AllocWindow window("cmfd.arena.release");
        // Every live instantiation is in the caches (a successful capture is
        // pushed there before it is used, a failed one leaves nothing), so
        // this is the single teardown -- destroying graph_exec separately
        // would be a double free of a cached entry.
        destroyGraphCaches();
        if (handle != nullptr) cublasDestroy(handle);
        handle = nullptr;
        if (stream != nullptr) cudaStreamDestroy(stream);
        stream = nullptr;
        cudaFree(neighbors);
        cudaFree(colors);
        cudaFree(d_slot_map);
        d_slot_map = nullptr;
        if (h_slot_map != nullptr) cudaFreeHost(h_slot_map);
        h_slot_map = nullptr;
        cudaFree(assembly_node_surface);
        cudaFree(assembly_face_area);
        cudaFree(assembly_volume);
        cudaFree(diag);
        cudaFree(dinv);
        cudaFree(cc);
        cudaFree(src);
        cudaFree(phi);
        cudaFree(r);
        cudaFree(r0);
        cudaFree(p);
        cudaFree(v);
        cudaFree(s);
        cudaFree(t);
        cudaFree(y);
        cudaFree(z);
        cudaFree(ax);
        cudaFree(partials);
        cudaFree(partials2);
        cudaFree(wiel_partials);
        cudaFree(fuse_retire);
        cudaFree(scalars);
        cudaFree(device_flags);
        cudaFree(iter_flags);
        cudaFree(device_halt);
        cudaFree(device_active);
        cudaFree(device_counters);
        cudaFree(device_status);

        cudaFree(xs_chif);
        cudaFree(xs_xsnf);
        cudaFree(xs_xsrf);
        cudaFree(xs_xssm);
        cudaFree(dtil_dev);
        cudaFree(dhat_dev);
        cudaFree(node_vol);
        cudaFree(udiag_dev);
        cudaFree(psi_dev);
        cudaFree(sweep_halt);
        cudaFree(device_assembly_active);
        cudaFree(diag_f);
        cudaFree(dinv_f);
        cudaFree(cc_f);
        cudaFree(r_f);
        cudaFree(r0_f);
        cudaFree(p_f);
        cudaFree(v_f);
        cudaFree(s_f);
        cudaFree(t_f);
        cudaFree(y_f);
        cudaFree(z_f);
        diag_f = dinv_f = cc_f = nullptr;
        r_f = r0_f = p_f = v_f = s_f = t_f = y_f = z_f = nullptr;
        xs_chif = xs_xsnf = xs_xsrf = xs_xssm = dtil_dev = dhat_dev = nullptr;
        node_vol = udiag_dev = psi_dev = nullptr;
        sweep_halt = device_assembly_active = nullptr;
        if (host_status != nullptr) cudaFreeHost(host_status);
        host_status = nullptr;
        if (host_active != nullptr) cudaFreeHost(host_active);
        host_active = nullptr;
        if (host_sweep_halt != nullptr) cudaFreeHost(host_sweep_halt);
        host_sweep_halt = nullptr;
        // The slots alias this block, so it is freed only after every slot has
        // stopped pointing at it -- and they are cleared here rather than in
        // ~Slot, because release() can run while the vector is still alive.
        for (auto& sl : slot) { sl.sweep_in = nullptr; sl.sweep_out = nullptr; }
        if (host_sweep_scalars != nullptr) cudaFreeHost(host_sweep_scalars);
        host_sweep_scalars = nullptr;
        if (host_assembly_active != nullptr) cudaFreeHost(host_assembly_active);
        host_assembly_active = nullptr;
        neighbors = nullptr;
        colors = nullptr;
        assembly_node_surface = nullptr;
        assembly_face_area = assembly_volume = nullptr;
        diag = dinv = cc = src = phi = r = r0 = p = v = s = t = y = z = ax = nullptr;
        partials = nullptr;
        partials2 = nullptr;
        wiel_partials = nullptr;
        fuse_retire = nullptr;
        scalars = nullptr;
        device_flags = nullptr;
        iter_flags = nullptr;
        device_halt = nullptr;
        device_active = nullptr;
        device_counters = nullptr;
        device_status = nullptr;
        available = false;
    }

    [[nodiscard]] int node_blocks() const { return (nxyz + block_size - 1) / block_size; }
    [[nodiscard]] int vector_blocks() const { return (n + block_size - 1) / block_size; }

    /// WP17: the width of the FIVE per-iteration elementwise classes.  0 means
    /// "unchanged", so with RASBERY_GPU_CMFD_BLOCK unset this IS block_size and
    /// every grid below is the grid the profile measured.
    [[nodiscard]] int cmfd_block_threads() const {
        return cmfd_block > 0 ? cmfd_block : block_size;
    }
    [[nodiscard]] int cmfd_node_blocks() const {
        const int w = cmfd_block_threads();
        return (nxyz + w - 1) / w;
    }
    [[nodiscard]] int cmfd_vector_blocks() const {
        const int w = cmfd_block_threads();
        return (n + w - 1) / w;
    }
    [[nodiscard]] long long vec_stride() const { return static_cast<long long>(n); }
    [[nodiscard]] long long mat_stride() const { return static_cast<long long>(matrix_count); }
    [[nodiscard]] long long cpl_stride() const { return static_cast<long long>(coupling_count); }
    [[nodiscard]] long long surface_stride() const {
        return static_cast<long long>(surface_group_count);
    }

    /// Exact compatibility check for the immutable CMFD topology.  Shape
    /// counts alone are insufficient: two loading maps can have identical
    /// nxyz/ngxyz yet different neighbours, and sharing the first map would
    /// produce physically wrong CMFD results without an allocation error.
    [[nodiscard]] bool compatibleGeometry(Geometry& geometry) const {
        if (geometry.ng() != 2 || geometry.nxyz() != nxyz ||
            geometry.ngxyz() != n ||
            static_cast<size_t>(geometry.ng2()) * geometry.nxyz() != matrix_count)
            return false;
        for (int l = 0; l < nxyz; ++l) {
            for (int idir = 0; idir < NDIRMAX; ++idir) {
                for (int lr = 0; lr < LR; ++lr) {
                    const size_t idx = static_cast<size_t>(6 * l + idir * LR + lr);
                    if (topology_neighbors[idx] != geometry.neib(lr, idir, l) ||
                        topology_node_surface[idx] != geometry.lktosfc(lr, idir, l))
                        return false;
                }
            }
            const double hx = geometry.hmesh(XDIR, l);
            const double hy = geometry.hmesh(YDIR, l);
            const double hz = geometry.hmesh(ZDIR, l);
            const double area[NDIRMAX] = {hy * hz, hx * hz, hx * hy};
            for (int idir = 0; idir < NDIRMAX; ++idir)
                if (topology_face_area[static_cast<size_t>(l) * NDIRMAX + idir] !=
                    area[idir])
                    return false;
            if (topology_volume[static_cast<size_t>(l)] != geometry.vol(l)) return false;
        }
        return true;
    }

    /// Grid for a per-node/per-element kernel: x is the *single-instance* grid,
    /// y is the batch axis.  Never fold the two.
    [[nodiscard]] dim3 node_grid() const {
        return dim3(static_cast<unsigned>(node_blocks()), static_cast<unsigned>(lanes));
    }
    [[nodiscard]] dim3 vector_grid() const {
        return dim3(static_cast<unsigned>(vector_blocks()), static_cast<unsigned>(lanes));
    }
    /// WP17: the same two grids at the per-iteration width.  Identical to
    /// node_grid()/vector_grid() when RASBERY_GPU_CMFD_BLOCK is unset.
    [[nodiscard]] dim3 cmfd_node_grid() const {
        return dim3(static_cast<unsigned>(cmfd_node_blocks()), static_cast<unsigned>(lanes));
    }
    [[nodiscard]] dim3 cmfd_vector_grid() const {
        return dim3(static_cast<unsigned>(cmfd_vector_blocks()), static_cast<unsigned>(lanes));
    }

    /// How many DISPATCHES one BiCGSTAB iteration costs on the launch-chain
    /// arm.  The same structural model tools/test_cmfd_fuse_contract.py builds
    /// the graph census from, so the receipt and the census cannot disagree:
    /// two dots + one dot2 + 2 x rb_sweeps colour sweeps + the five elementwise
    /// kernels + the residual stage 1 + the scalar tail.
    [[nodiscard]] int launchesPerIteration() const {
        const int dot_nodes  = fuse_dot ? 1 : 2;
        const int dot2_nodes = fuse_dot2 ? 1 : 2;
        const int tail       = scalar_fusion ? 1 : 2;
        return 2 * dot_nodes + dot2_nodes + 2 * rb_sweeps + 6 + tail;
    }

    // -----------------------------------------------------------------------
    // WP17: THE PERSISTENT ARM'S ADMISSION TEST.
    //
    // Every precondition bicg_iteration_persistent's body assumes is decided
    // HERE, once, at stand-up, and the first one that fails is what the receipt
    // names.  The ladder order is deliberate: the cheapest and most common
    // reasons first, so a run that simply did not ask says `arm_off` rather
    // than a device fact that is true but irrelevant.
    // -----------------------------------------------------------------------
    void armPersistent(const cudaDeviceProp& properties) {
        persistent_request     = cmfdPersistentRequested();
        cooperative_supported  = properties.cooperativeLaunch != 0;
        persistent_armed       = false;
        persistent_blocks      = 0;
        if (!persistent_request) {
            persistent_refusal = PersistentRefusal::ArmOff;
            return;
        }
        // A cooperative launch cannot be recorded into a stream capture, so the
        // captured outer graph and this arm are mutually exclusive BY
        // CONSTRUCTION, not by preference.  Refusing at stand-up rather than at
        // launch keeps the capture path exactly as it is.
        if (use_graph) {
            persistent_refusal = PersistentRefusal::OuterGraphActive;
            return;
        }
        // One grid barrier spans the whole grid, batch axis included, so a lane
        // that halts while its neighbours do not would strand the grid for
        // ever.  The spike serves the single-deck shape only.
        if (slots != 1) {
            persistent_refusal = PersistentRefusal::BatchWidth;
            return;
        }
        if (fp32_inner) {
            persistent_refusal = PersistentRefusal::Fp32Inner;
            return;
        }
        if (!cooperative_supported) {
            persistent_refusal = PersistentRefusal::NoCooperativeLaunch;
            return;
        }
        // The reduction's binary tree is kReduceThreads lanes wide at compile
        // time, so the persistent block is kReduceThreads and nothing else --
        // which is only legal if the device will take a block that wide.  A
        // runtime test rather than a static_assert, because what can fail here
        // is the DEVICE, not the constant.
        if (kReduceThreads > properties.maxThreadsPerBlock) {
            persistent_refusal = PersistentRefusal::BlockWidthMismatch;
            return;
        }
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, reinterpret_cast<const void*>(&bicg_iteration_persistent),
                kReduceThreads, 0) != cudaSuccess) {
            cudaGetLastError();
            persistent_refusal = PersistentRefusal::OccupancyTooSmall;
            return;
        }
        const int resident = per_sm * properties.multiProcessorCount;
        const int fold     = reduce_blocks_for(n);
        // Blocks beyond the node coverage do nothing but wait at barriers, so
        // the grid is the SMALLER of "what fits resident" and "what the work
        // needs"; it may never be smaller than the fixed fold partition,
        // because blocks [0, fold) are the only ones that write partials.
        const int wanted = std::max(fold, (nxyz + kReduceThreads - 1) / kReduceThreads);
        const int blocks = std::min(resident, wanted);
        if (blocks < fold) {
            persistent_refusal = PersistentRefusal::OccupancyTooSmall;
            return;
        }
        persistent_blocks  = blocks;
        persistent_armed   = true;
        persistent_refusal = PersistentRefusal::None;
    }

    /// The persistent arm's enqueue.  Returns false -- having changed nothing
    /// on the stream -- when the iteration must fall through to the launch
    /// chain, so the caller's next statement is the unmodified reference path.
    bool enqueuePersistentIteration(int allow_halt, int force_halt) {
        if (!persistent_armed) return false;
        // Defensive second half of the graph exclusion: `use_graph` was false
        // at stand-up, but launch_sweeps' RESIDENT_SINGLE segment can still
        // open a capture on this stream, and a cooperative launch recorded into
        // one is an error, not a slow path.
        if (rasbery::graphCaptureActive(stream)) {
            persistent_refusal = PersistentRefusal::CaptureActive;
            return false;
        }
        if (fp32Active()) {
            persistent_refusal = PersistentRefusal::Fp32Inner;
            return false;
        }
        PersistentBicgParams args{};
        args.nxyz          = nxyz;
        args.n             = n;
        args.reduce_blocks = reduce_blocks_for(n);
        args.rb_sweeps     = rb_sweeps;
        args.ncolors       = ncolors;
        args.allow_halt    = allow_halt;
        args.force_halt    = force_halt;
        args.vec_stride    = vec_stride();
        args.mat_stride    = mat_stride();
        args.cpl_stride    = cpl_stride();
        args.colors        = colors;
        args.neighbors     = neighbors;
        args.cc            = cc;
        args.diag          = diag;
        args.dinv          = dinv;
        args.scalars       = scalars;
        args.iter_flags    = iter_flags;
        args.sticky_flags  = device_flags;
        args.counters      = device_counters;
        args.halt          = device_halt;
        args.active        = device_active;
        args.r0            = r0;
        args.r             = r;
        args.p             = p;
        args.v             = v;
        args.s             = s;
        args.t             = t;
        args.y             = y;
        args.z             = z;
        args.phi           = phi;
        args.partials      = partials;
        args.partials2     = partials2;

        int*      map_arg   = d_slot_map;
        int       lanes_arg = lanes;
        void*     argv[]    = {&args, &map_arg, &lanes_arg};
        const cudaError_t rc = cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(&bicg_iteration_persistent),
            dim3(static_cast<unsigned>(persistent_blocks), 1u, 1u),
            dim3(static_cast<unsigned>(kReduceThreads), 1u, 1u), argv, 0, stream);
        if (rc != cudaSuccess) {
            // A refused cooperative launch enqueues nothing, so the reference
            // chain below is the first and only execution of this iteration --
            // the same argument launch_outer's capture fallback rests on.  The
            // arm is latched off so the refusal is paid once, not per iteration.
            cudaGetLastError();
            persistent_armed   = false;
            persistent_refusal = PersistentRefusal::LaunchFailed;
            return false;
        }
        return true;
    }
    [[nodiscard]] dim3 scalar_grid() const {
        return dim3(1u, static_cast<unsigned>(lanes));
    }

    /// The two kernels that are deliberately NOT compacted.
    ///
    /// initialize_solver_state writes iter_flags[m] and halt[m] for EVERY
    /// declared slot before it masks anything -- that is byte for byte the
    /// per-iteration cudaMemsetAsync(iter_flags) it replaced, and `halt` is
    /// what every later kernel consults, so a slot that never gets written
    /// keeps a stale mask.  finalize_status likewise has to cover every slot
    /// because the status D2H copies all of them.  Both are one thread per
    /// slot, so keeping them full width costs a handful of empty blocks and
    /// buys the whole full-width argument for free -- no separate reset
    /// kernel, no extra graph node, and the OFF path is untouched.
    [[nodiscard]] dim3 full_scalar_grid() const {
        return dim3(1u, static_cast<unsigned>(slots));
    }

    /// Record the incoming buffers and decide, per array, whether the device
    /// copy is already the same bytes.  No CUDA call: safe to run concurrently
    /// on every instance thread.
    void stageSlot(int m, const double* host_diag, const double* host_cc,
                   const double* host_phi, const double* host_src) {
        Slot& sl     = slot[static_cast<size_t>(m)];
        sl.host_diag = host_diag;
        sl.host_cc   = host_cc;
        sl.host_diag_out = const_cast<double*>(host_diag);
        sl.host_cc_out   = const_cast<double*>(host_cc);
        sl.host_phi  = host_phi;
        sl.host_src  = host_src;
        // diag is rewritten every outer (Wielandt) and every CMFD sweep
        // (updls), so its mirror can never match: skip the 4n-double memcmp
        // here and the 4n-double shadow copy on the launcher's critical path,
        // and just upload it every time -- which is what happened anyway.
        sl.push_diag = true;
        sl.push_cc   = !mirrorMatches(sl.cc_mirror, host_cc, coupling_count);
        sl.push_phi  = !phiMirrorMatches(sl, host_phi, static_cast<size_t>(n));
    }

    static bool mirrorMatches(const MirroredUpload& mirror, const double* host, size_t count) {
        return mirror.valid &&
               std::memcmp(mirror.shadow.data(), host, count * sizeof(double)) == 0;
    }

    /// Is the FLUX mirror worth its keep?  MEASURED: yes, at every width.
    ///
    /// The audit item this answers (C5) proposed bypassing the mirror in
    /// single-slot mode, on the theory that its two host passes -- the shadow
    /// copy in adoptFluxMirror and the memcmp before the next upload, 2 x n
    /// doubles -- cost more than the one async H2D of n doubles they elide,
    /// with an idle copy engine and no batch-mates to amortise over.  The
    /// measurement says the opposite, at the KNGR mesh (n = 16 902, 132 KB):
    ///
    ///     adoptFluxMirror shadow copy               1.74 us
    ///     mirrorMatches memcmp                      1.95 us
    ///     -> mirror, per launch                     3.69 us
    ///     pinned cudaMemcpyAsync H2D, ISSUE only   19.52 us
    ///     pinned H2D + stream sync                 50.11 us
    ///
    /// and the elision is not occasional: inside the sweep loop the device
    /// produced the flux and the host only read it, so the shadow matches on
    /// every continuation launch.  The mirror is therefore ~3.7 us spent to
    /// avoid >=19.5 us of launcher time, single slot included.  It stays on.
    ///
    /// The gate remains so the claim stays falsifiable on the real device:
    /// RASBERY_GPU_PHI_MIRROR=0 turns it off, and cmfd_phi_mirror_ns /
    /// _calls / cmfd_phi_h2d_elided_bytes are the two sides of the trade.
    /// Turning it off is byte-exact by construction -- the mirror only ever
    /// ELIDES an upload of bytes the device already holds, so declining to
    /// elide uploads the same bytes again.
    [[nodiscard]] bool phiMirrorEnabled() const {
        static const bool on = [] {
            const char* v = std::getenv("RASBERY_GPU_PHI_MIRROR");
            return v == nullptr || std::string(v) != "0";
        }();
        return on;
    }

    /// mirrorMatches for the flux, with the width gate and the cost clock.
    bool phiMirrorMatches(Slot& sl, const double* host_phi, size_t count) {
        if (!phiMirrorEnabled()) {
            g_cmfd_phi_mirror_bypassed.fetch_add(1, std::memory_order_relaxed);
            return false; // "does not match" == upload, which is always correct
        }
        const auto t0 = std::chrono::steady_clock::now();
        const bool hit = mirrorMatches(sl.phi_mirror, host_phi, count);
        g_cmfd_phi_mirror_ns.fetch_add(
            static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count()),
            std::memory_order_relaxed);
        g_cmfd_phi_mirror_calls.fetch_add(1, std::memory_order_relaxed);
        if (hit)
            g_cmfd_phi_h2d_elided_bytes.fetch_add(
                static_cast<unsigned long long>(count) * sizeof(double),
                std::memory_order_relaxed);
        return hit;
    }

    // -----------------------------------------------------------------------
    // WP13: the three per-outer masks that never change on a single deck
    // -----------------------------------------------------------------------
    //
    // `d_slot_map`, `device_active` and `device_assembly_active` are four bytes
    // each on a width-1 arena and they were re-sent on EVERY launch: three of
    // the ~110,000 cudaMemcpyAsync calls nsys counted per run come from here
    // and carry 12 bytes between them.  Bytes are not the cost -- the API call
    // is (2.27 s of cudaMemcpyAsync over a 14.4 s wall, dominated by call
    // COUNT, not payload).
    //
    // WHY THESE THREE AND NOT sweep_halt.  All three are declared
    // `const std::uint32_t* __restrict__` / `const int* __restrict__` in every
    // kernel signature that takes them, so no device code writes them and a
    // host shadow of the last upload is a true statement about device content.
    // `sweep_halt` is the counter-example and is deliberately NOT here:
    // initialize_solver_state RAISES it for a masked-off slot and
    // issueSweepDownloads memsets it back, so the device value between two
    // launches is not the value the host last sent, and a shadow would elide a
    // copy that is doing real work.
    //
    // AND WHY THE SHADOW IS NOT THE STAGING BUFFER.  stageActive()/stageSlotMap()
    // rotate through per-lane pinned buffers precisely so that no host write can
    // land in the source range of an in-flight copy; comparing against one of
    // them would be comparing against a buffer some other lane is about to
    // rewrite.  These shadows are ordinary heap vectors owned by the core.
    template <class T>
    void pushDeviceReadOnly(const char* leaf, T* dst, const T* src, size_t count,
                            std::vector<T>& shadow) {
        const size_t bytes = count * sizeof(T);
        if (rasbery::xfer::elideEnabled()) {
            const bool hit = shadow.size() == count &&
                             std::memcmp(shadow.data(), src, bytes) == 0;
            rasbery::xfer::countElisionTest(hit, bytes);
            if (hit) return;
        }
        CUDA_CHECK(rasbery::xfer::memcpyAsync("CudaBICGBackend.cu:pushDeviceReadOnly", leaf,
                                       dst, src, bytes, cudaMemcpyHostToDevice, stream));
        if (rasbery::xfer::elideEnabled()) {
            shadow.assign(src, src + count);
        }
    }

    /// Decide this launch's dispatch width and publish the lane -> slot map.
    ///
    /// Called from issueUploads / issueSweepUploads, i.e. on the launcher's
    /// stream BEFORE the graph launch and OUTSIDE any capture, so the map the
    /// replayed graph reads is the one this launch wrote.  `active_slots` is
    /// ascending (the rendezvous builds it that way), and the compacted map
    /// preserves that order: every per-slot mirror, generation counter and
    /// status row is keyed by the PHYSICAL slot, so lane order is the only
    /// thing allowed to be logical.
    ///
    /// Compaction OFF is the FULL IDENTITY, not "the participants only": the
    /// OFF launch must visit the same blocks it visited before compaction
    /// existed and be masked in the same place (the halt guard), or the two
    /// paths stop being the same program.
    void buildSlotMap(const int* active_slots, int count) {
        int* const map = stageSlotMap();
        if (!compact) {
            lanes = slots;
            for (int i = 0; i < slots; ++i) map[i] = i;
        } else {
            lanes = cmfdBucketFor(count, slots);
            for (int i = 0; i < slots; ++i) map[i] = -1;
            for (int i = 0; i < count && i < lanes; ++i) map[i] = active_slots[i];
        }
        // Always the FULL fleet width, never `lanes`: a stale entry from a
        // wider previous launch must never be reachable by a later, deeper
        // graph replay.
        //
        // WP13 (RASBERY_GPU_XFER_ELIDE): and on a single deck it is the SAME
        // full width every launch -- `compact` is false, so the map is the
        // identity 0..slots-1 and this 4-byte copy has re-sent it once per
        // outer for the whole run.  d_slot_map is `const int* __restrict__
        // slot_map` at every one of the ~50 kernel signatures that take it
        // (RASBERY_CMFD_SLOT_ARGS), so nothing on the device writes it and a
        // shadow of what the host last sent IS what the device holds.
        pushDeviceReadOnly("buildSlotMap:d_slot_map", d_slot_map, map,
                           static_cast<size_t>(slots), shadow_slot_map);
        g_cmfd_logical_drives.fetch_add(static_cast<unsigned long long>(count),
                                        std::memory_order_relaxed);
        g_cmfd_physical_blocks.fetch_add(static_cast<unsigned long long>(count),
                                         std::memory_order_relaxed);
        g_cmfd_padding_blocks.fetch_add(
            static_cast<unsigned long long>(lanes > count ? lanes - count : 0),
            std::memory_order_relaxed);
        cmfdBucketHistogramBump(lanes);
    }

    /// H2D for the participating slots, plus the participation mask itself.
    void issueUploads(const int* active_slots, int count) {
        buildSlotMap(active_slots, count);
        std::uint32_t* const active = stageActive();
        std::memset(active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        for (int i = 0; i < count; ++i) active[active_slots[i]] = 1u;
        // THROUGH THE SAME SHADOW as issueSweepUploads', and that is not an
        // optimisation here but the correctness condition for the one there:
        // two writers of device_active and one shadow means the shadow has to
        // see both, or a rendezvous launch would leave it describing bytes the
        // device no longer holds and the next sweep launch would elide against
        // it.  (This arm is dead under RASBERY_GPU_CMFD_SWEEP, which is why it
        // would never have shown up in a PROD A/B.)
        pushDeviceReadOnly("issueUploads:device_active", device_active, active,
                           static_cast<size_t>(slots), shadow_active);

        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];

            // diag really does change every outer (the Wielandt shift rewrites
            // it), so it is not mirrored at all (see stageSlot); cc and phi are
            // the ones whose mirrors drop uploads.
            {
                const size_t bytes = matrix_count * sizeof(double);
                CUDA_CHECK(rasbery::xfer::memcpyAsync(
                    "CudaBICGBackend.cu:issueUploads", "diag", diag + m * mat_stride(),
                    sl.host_diag, bytes, cudaMemcpyHostToDevice, stream));
                ++telemetry.bulk_h2d_calls_during_iteration;
                telemetry.bulk_h2d_bytes_during_iteration += bytes;
            }
            pushOrSkip("issueUploads:cc", cc + m * cpl_stride(), sl.host_cc,
                       coupling_count, sl.push_cc, sl.cc_mirror);
            pushOrSkip("issueUploads:phi", phi + m * vec_stride(), sl.host_phi,
                       static_cast<size_t>(n), sl.push_phi, sl.phi_mirror);

            // src is rebuilt from psi on the host at the top of every outer; it
            // is the one buffer that is genuinely new each time.
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueUploads", "src", src + m * vec_stride(),
                sl.host_src, static_cast<size_t>(n) * sizeof(double),
                cudaMemcpyHostToDevice, stream));
            ++telemetry.bulk_h2d_calls_during_iteration;
            telemetry.bulk_h2d_bytes_during_iteration +=
                static_cast<size_t>(n) * sizeof(double);

            // The exit tolerance is a kernel *input*, not a kernel argument:
            // keeping it in device memory is what lets one captured graph serve
            // every outer of every instance.
            if (!(sl.eps == sl.eps_on_device)) {
                CUDA_CHECK(rasbery::xfer::memcpyAsync(
                    "CudaBICGBackend.cu:issueUploads", "eps",
                    scalars + static_cast<long long>(m) * kScalarCount + kEps, &sl.eps,
                    sizeof(double), cudaMemcpyHostToDevice, stream));
                // Graph/direct kernels are submitted to this same stream, so
                // stream order publishes eps without draining the pipeline.
                sl.eps_on_device = sl.eps;
            }

            // update_solution advances the device flux, so the host mirror no
            // longer describes device memory.  fetchFlux re-establishes it.
            sl.phi_mirror.valid = false;
        }
    }

    void pushOrSkip(const char* leaf, double* device_buffer, const double* host_buffer,
                    size_t count, bool push, MirroredUpload& mirror) {
        if (!push) {
            ++telemetry.bulk_h2d_skipped_during_iteration;
            return;
        }
        const size_t bytes = count * sizeof(double);
        CUDA_CHECK(rasbery::xfer::memcpyAsync("CudaBICGBackend.cu:pushOrSkip", leaf,
                                       device_buffer, host_buffer, bytes,
                                       cudaMemcpyHostToDevice, stream));
        mirror.shadow.assign(host_buffer, host_buffer + count);
        mirror.valid = true;
        ++telemetry.bulk_h2d_calls_during_iteration;
        telemetry.bulk_h2d_bytes_during_iteration += bytes;
    }

    /// Bit-reproducible replacement for cublasDdot / cublasDnrm2.
    void dot(const double* a, const double* b, int scalar_slot, bool take_sqrt = false) {
        const int blocks = reduce_blocks_for(n);
        // FUSE bit 0.  Same partition, same tree, same strict stage-2 fold, one
        // node instead of two -- see reduce_dot_fused for the six-point
        // bit-identity argument.  The reference pair below is what mask 0 runs
        // and it is never deleted.
        if (fuse_dot) {
            reduce_dot_fused<<<dim3(static_cast<unsigned>(blocks),
                                    static_cast<unsigned>(lanes)),
                               kReduceThreads, 0, stream>>>(
                n, vec_stride(), a, b, partials, scalars, scalar_slot, take_sqrt,
                fuse_retire, device_halt, d_slot_map, lanes);
            return;
        }
        reduce_dot_stage1<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(lanes)),
                            kReduceThreads, 0, stream>>>(
            n, vec_stride(), a, b, partials, device_halt, d_slot_map, lanes);
        reduce_dot_stage2<<<scalar_grid(), 1, 0, stream>>>(
            blocks, partials, scalars, scalar_slot, take_sqrt, device_halt, d_slot_map, lanes);
    }

    /// Two independent dots over the same n in ONE pair of nodes.  See
    /// reduce_dot2_stage1: this is not a fused reduction, it is two reductions
    /// riding in one kernel with their own accumulators, their own partial
    /// arrays and the same partition each had alone.
    void dot2(const double* a0, const double* b0, int scalar_slot0,
              const double* a1, const double* b1, int scalar_slot1) {
        const int blocks = reduce_blocks_for(n);
        // FUSE bit 1.  See reduce_dot2_fused: two independent reductions, each
        // with the partition, tree and strict fold it had in the pair.
        if (fuse_dot2) {
            reduce_dot2_fused<<<dim3(static_cast<unsigned>(blocks),
                                     static_cast<unsigned>(lanes)),
                                kReduceThreads, 0, stream>>>(
                n, vec_stride(), a0, b0, a1, b1, partials, partials2, scalars,
                scalar_slot0, scalar_slot1, fuse_retire, device_halt, d_slot_map, lanes);
            return;
        }
        reduce_dot2_stage1<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(lanes)),
                             kReduceThreads, 0, stream>>>(
            n, vec_stride(), a0, b0, a1, b1, partials, partials2, device_halt, d_slot_map, lanes);
        reduce_dot2_stage2<<<scalar_grid(), 1, 0, stream>>>(
            blocks, partials, partials2, scalars, scalar_slot0, scalar_slot1, device_halt, d_slot_map, lanes);
    }

    /// Is the mixed-precision inner loop armed right now?  The env gate minus
    /// the sticky safety latch (see latchFp32Off).  Every enqueue and every
    /// graph-validity test asks this one question, so the captured topology and
    /// the kernels inside it can never disagree.
    [[nodiscard]] bool fp32Active() const { return fp32_inner && !fp32_latched_off; }

    /// FP32-payload counterpart of dot(): _f32 stage 1, unmodified double
    /// stage 2.  The partial array, its stride and the fold order are shared
    /// with the FP64 path, which is why no _f32 stage 2 exists.
    void dot_f32(const float* a, const float* b, int scalar_slot) {
        const int blocks = reduce_blocks_for(n);
        reduce_dot_stage1_f32<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(lanes)),
                                kReduceThreads, 0, stream>>>(
            n, vec_stride(), a, b, partials, device_halt, d_slot_map, lanes);
        reduce_dot_stage2<<<scalar_grid(), 1, 0, stream>>>(
            blocks, partials, scalars, scalar_slot, false, device_halt, d_slot_map, lanes);
    }

    void dot2_f32(const float* a0, const float* b0, int scalar_slot0,
                  const float* a1, const float* b1, int scalar_slot1) {
        const int blocks = reduce_blocks_for(n);
        reduce_dot2_stage1_f32<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(lanes)),
                                 kReduceThreads, 0, stream>>>(
            n, vec_stride(), a0, b0, a1, b1, partials, partials2, device_halt, d_slot_map, lanes);
        reduce_dot2_stage2<<<scalar_grid(), 1, 0, stream>>>(
            blocks, partials, partials2, scalars, scalar_slot0, scalar_slot1, device_halt, d_slot_map, lanes);
    }

    void precondition_sweeps_f32(const float* b, float* x) {
        for (int sweep = 0; sweep < rb_sweeps; ++sweep)
            colored_block_sweep_f32<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
                nxyz, vec_stride(), mat_stride(), cpl_stride(), sweep % ncolors, colors,
                neighbors, cc_f, dinv_f, b, x, device_halt, d_slot_map, lanes);
    }

    /// One mixed-precision BiCGSTAB iteration.  Kernel for kernel, node for
    /// node, this is enqueue_iteration with the FP32 twins substituted: the two
    /// scalar stage-2 kernels, the halt/telemetry bookkeeping and the fusion
    /// switch are the SAME kernels the FP64 path launches, so the captured
    /// topology is identical and the counters mean the same thing.
    void enqueue_iteration_f32(int allow_halt, int force_halt = 0) {
        dot_f32(r0_f, r_f, kRhoNew);
        prepare_p_jacobi_f32<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv_f, r_f, v_f,
            p_f, y_f, device_halt, d_slot_map, lanes);
        precondition_sweeps_f32(p_f, y_f);
        matvec_two_group_f32<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag_f, cc_f,
            y_f, v_f, device_halt, d_slot_map, lanes);

        dot_f32(r0_f, v_f, kR0V);
        update_s_jacobi_f32<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv_f, r_f, v_f,
            s_f, z_f, device_halt, d_slot_map, lanes);
        precondition_sweeps_f32(s_f, z_f);
        matvec_two_group_f32<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag_f, cc_f,
            z_f, t_f, device_halt, d_slot_map, lanes);

        dot2_f32(s_f, t_f, kPts, t_f, t_f, kPtt);

        update_solution_f32<<<cmfd_vector_grid(), cmfd_block_threads(), 0, stream>>>(
            n, vec_stride(), scalars, iter_flags, y_f, z_f, s_f, t_f, phi, r_f,
            device_halt, d_slot_map, lanes);
        const int norm_blocks = reduce_blocks_for(n);
        reduce_dot_stage1_f32<<<
            dim3(static_cast<unsigned>(norm_blocks), static_cast<unsigned>(lanes)),
            kReduceThreads, 0, stream>>>(
            n, vec_stride(), r_f, r_f, partials, device_halt, d_slot_map, lanes);
        if (scalar_fusion) {
            reduce_norm_accumulate_stage2<<<scalar_grid(), 1, 0, stream>>>(
                norm_blocks, allow_halt, force_halt, partials, scalars, iter_flags,
                device_flags, device_counters, device_halt, device_active, d_slot_map, lanes);
        } else {
            reduce_dot_stage2<<<scalar_grid(), 1, 0, stream>>>(
                norm_blocks, partials, scalars, kInitialNorm, true, device_halt, d_slot_map, lanes);
            accumulate_iteration<<<scalar_grid(), 1, 0, stream>>>(
                allow_halt, force_halt, scalars, iter_flags, device_flags,
                device_counters, device_halt, device_active, d_slot_map, lanes);
        }
    }

    /// The COLOUR SWEEPS of the block-Jacobi preconditioner.
    ///
    /// The diagonal solve that used to open this chain (block_jacobi) is now
    /// fused into whichever kernel produces `b` -- prepare_p_jacobi for
    /// (p -> y), update_s_jacobi for (s -> z).  It could move because it reads
    /// b only at its own node; the sweeps below cannot, because each reads x
    /// at NEIGHBOURING nodes and therefore depends on the previous sweep's
    /// writes across the whole grid.  The kernel boundary IS that barrier, and
    /// the colour order is the Gauss-Seidel semantics, so this loop stays
    /// exactly as it was.
    void precondition_sweeps(const double* b, double* x) {
        for (int sweep = 0; sweep < rb_sweeps; ++sweep)
            colored_block_sweep<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
                nxyz, vec_stride(), mat_stride(), cpl_stride(), sweep % ncolors, colors, neighbors,
                cc, dinv, b, x, device_halt, d_slot_map, lanes);
    }

    /// One BiCGSTAB iteration.  `allow_halt` is 0 only for the first one,
    /// which the CPU reference also runs before testing anything.
    /// `force_halt` is 1 only on the last iteration of the algorithmic budget
    /// when the capture is deeper than that budget (see enqueue_outer).
    void enqueue_iteration(int allow_halt, int force_halt = 0) {
        // The per-iteration cudaMemsetAsync(iter_flags) node is gone: its two
        // halves now live in initialize_solver_state (arm at the top of the
        // outer) and at the end of accumulate_iteration (re-arm for the next
        // captured iteration).  Both are one-thread-per-slot writes of the
        // same word at the same points in stream order, so the scratch flags
        // every reader sees are the ones the memset produced -- see the
        // argument at the re-arm site.
        //
        // The persistent arm is ONE cooperative launch for this whole function.
        // It returns false having enqueued nothing when it cannot run, so the
        // reference chain below is untouched on every refusal path.
        if (enqueuePersistentIteration(allow_halt, force_halt)) return;
        dot(r0, r, kRhoNew);
        prepare_p_jacobi<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv, r, v, p, y,
            device_halt, d_slot_map, lanes);
        precondition_sweeps(p, y);
        matvec_two_group<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, y, v, device_halt, d_slot_map, lanes);

        dot(r0, v, kR0V);
        update_s_jacobi<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv, r, v, s, z,
            device_halt, d_slot_map, lanes);
        precondition_sweeps(s, z);
        matvec_two_group<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, z, t, device_halt, d_slot_map, lanes);

        // s.t and t.t are the only two adjacent dots with no kernel between
        // them; one stage-1/stage-2 pair carries both, each with its own
        // accumulator and partial array.
        dot2(s, t, kPts, t, t, kPtt);

        update_solution<<<cmfd_vector_grid(), cmfd_block_threads(), 0, stream>>>(
            n, vec_stride(), scalars, iter_flags, y, z, s, t, phi, r, device_halt, d_slot_map, lanes);
        // Absolute residual of the iterate that update_solution just wrote.
        // The stage-1 partition is unchanged; only the scalar stage-2 node is
        // optionally fused with its immediately dependent bookkeeping node.
        const int norm_blocks = reduce_blocks_for(n);
        reduce_dot_stage1<<<
            dim3(static_cast<unsigned>(norm_blocks), static_cast<unsigned>(lanes)),
            kReduceThreads, 0, stream>>>(
            n, vec_stride(), r, r, partials, device_halt, d_slot_map, lanes);
        if (scalar_fusion) {
            reduce_norm_accumulate_stage2<<<scalar_grid(), 1, 0, stream>>>(
                norm_blocks, allow_halt, force_halt, partials, scalars, iter_flags,
                device_flags, device_counters, device_halt, device_active, d_slot_map, lanes);
        } else {
            reduce_dot_stage2<<<scalar_grid(), 1, 0, stream>>>(
                norm_blocks, partials, scalars, kInitialNorm, true, device_halt, d_slot_map, lanes);
            accumulate_iteration<<<scalar_grid(), 1, 0, stream>>>(
                allow_halt, force_halt, scalars, iter_flags, device_flags,
                device_counters, device_halt, device_active, d_slot_map, lanes);
        }
    }

    /// How many BiCGSTAB iterations one graph launch carries.
    ///
    /// The default is the algorithmic budget itself: the inner loop is ALREADY
    /// a single graph of `1 + nmax` iterations, so the per-iteration launch and
    /// sync cost the batching literature targets is already amortised here --
    /// `graph_launches` counts CMFD sweeps, not iterations.  K only ever
    /// *raises* the capture depth, never lowers it: a shallower capture would
    /// under-iterate, which the correctness contract forbids outright, so a
    /// too-small K is clamped rather than honoured.
    [[nodiscard]] int captured_iterations(int nmax) const {
        const int algorithmic = 1 + nmax;
        return iter_batch_request > algorithmic ? iter_batch_request : algorithmic;
    }

    /// The whole outer: initial residual, then 1 + nmax BiCGSTAB iterations of
    /// which the trailing ones self-cancel once `halt` is raised.  Exactly the
    /// sequence BICGCMFD::drive used to drive from the host, with the same
    /// operands in the same order.
    void enqueue_outer(int nmax) {
        initialize_solver_state<<<full_scalar_grid(), 1, 0, stream>>>(
            scalars, device_flags, device_halt, device_counters, iter_flags,
            device_active, sweep_halt);
        if (fp32Active()) {
            // The only added node of the mixed-precision topology, and the one
            // that makes a stale operator mirror impossible: it dominates every
            // write to the double diag/cc -- the H2D pushes, the device
            // assembly and the per-sweep cmfd_updls all precede this point.
            refresh_operator_mirror_f32<<<node_grid(), block_size, 0, stream>>>(
                nxyz, mat_stride(), cpl_stride(), diag, cc, diag_f, cc_f, device_halt, d_slot_map, lanes);
            // Same three fused steps as below, but only the RESULTS narrow: the
            // block inversion, A*phi and b - A*phi are FP64, and `r` still
            // receives the FP64 residual so the reference norm harvested by the
            // unmodified reduction below is the FP64 one.
            begin_outer_fused_f32<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc,
                phi, src, dinv_f, ax, r, r_f, r0_f, p_f, v_f, device_halt, d_slot_map, lanes);
        } else {
            // One node for what used to be three: the block inversion (independent
            // of the other two), the A*phi matvec and the residual it feeds.
            begin_outer_fused<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, phi,
                src, dinv, ax, r, r0, p, v, device_halt, d_slot_map, lanes);
        }
        const int reference_blocks = reduce_blocks_for(n);
        reduce_dot_stage1<<<
            dim3(static_cast<unsigned>(reference_blocks), static_cast<unsigned>(lanes)),
            kReduceThreads, 0, stream>>>(
            n, vec_stride(), r, r, partials, device_halt, d_slot_map, lanes);
        if (scalar_fusion) {
            reduce_norm_store_reference_stage2<<<scalar_grid(), 1, 0, stream>>>(
                reference_blocks, partials, scalars, device_halt, d_slot_map, lanes);
        } else {
            reduce_dot_stage2<<<scalar_grid(), 1, 0, stream>>>(
                reference_blocks, partials, scalars, kInitialNorm, true, device_halt, d_slot_map, lanes);
            store_reference_norm<<<scalar_grid(), 1, 0, stream>>>(scalars, device_halt, d_slot_map, lanes);
        }

        // The algorithmic budget is `1 + nmax`; the capture may be deeper.
        // Iteration `algorithmic - 1` then raises the halt itself, so every
        // captured iteration past the budget finds halt set, returns on its
        // first instruction in every kernel, and is counted as an over-run.
        const int algorithmic = 1 + nmax;
        const int captured    = captured_iterations(nmax);
        // WP17: the occupancy receipt, once, at the ONE place both precision
        // arms pass through -- a graph-off run has no [CMFD][GRAPH] line to
        // carry these fields, and the FP32 arm has no separate receipt.
        reportCmfdOccupancy(cmfd_block_threads(), block_size, cmfd_node_blocks(),
                            cmfd_vector_blocks(), reduce_blocks_for(n),
                            persistent_armed ? 1 : launchesPerIteration(),
                            persistent_armed, persistent_blocks,
                            cooperative_supported, persistent_refusal);
        for (int i = 0; i < captured; ++i) {
            const int allow_halt = i == 0 ? 0 : 1;
            const int force_halt =
                (i == algorithmic - 1 && captured > algorithmic) ? 1 : 0;
            if (fp32Active())
                enqueue_iteration_f32(allow_halt, force_halt);
            else
                enqueue_iteration(allow_halt, force_halt);
        }
        // A property of the capture, not a tally: assigned, never accumulated.
        // Set here rather than at launch so it is right on the graph-off path
        // too, where there is no launch to hang it off.
        iter_batch_used      = captured;
        telemetry.iter_batch = static_cast<std::uint64_t>(captured);

        finalize_status<<<full_scalar_grid(), 1, 0, stream>>>(
            scalars, device_flags, device_counters, device_status, device_active);
        // WP13.1: launch_outer CAPTURES this body when use_graph, so on that
        // arm the ledger sees one call per capture and the replays are
        // invisible to it.  The scope says so rather than leaving a reader
        // to reconcile the row against nsys and conclude the site is cold.
        CUDA_CHECK(rasbery::xfer::memcpyAsync(
            "CudaBICGBackend.cu:enqueue_outer(captured)", "host_status", host_status,
            device_status, static_cast<size_t>(slots) * sizeof(DeviceSolveStatus),
            cudaMemcpyDeviceToHost, stream));
    }

    /// Capture enqueue_outer once and replay it.  Every kernel argument is a
    /// fixed pointer or a compile-time-stable integer -- r20, eps, the
    /// iteration counter and now the participation mask all live in device
    /// memory precisely so that this holds -- so one instantiation serves every
    /// outer of the whole run, whichever subset of instances is riding along.
    void launch_outer(int nmax) {
        if (!use_graph) {
            enqueue_outer(nmax);
            return;
        }
        // The precision mode is part of the captured topology (different
        // kernels, one extra node), so a latched fallback invalidates the graph
        // exactly the way a changed nmax does.
        // grid.y is baked into a graph, so the dispatch width is TOPOLOGY:
        // a bucket change invalidates the instantiation exactly the way a
        // changed nmax or a latched FP32 fallback does.  With compaction off
        // `lanes` is the constant `slots` and this term never fires.
        if (graph_exec == nullptr || graph_nmax != nmax || graph_lanes != lanes ||
            graph_precision != precisionTag()) {
            // A bucket the arena has already served has an instantiation
            // waiting: switch to it instead of paying a capture again.  Without
            // this the arrival width oscillating between two buckets would
            // re-instantiate on every launch, which costs far more than the
            // padding blocks compaction removes.  The key space is bounded --
            // nine buckets x two precisions x one nmax -- so the list is short
            // by construction and needs no eviction.
            graph_exec = nullptr;
            graph_src  = nullptr;
            for (const OuterGraph& e : outer_graphs)
                if (e.nmax == nmax && e.lanes == lanes && e.precision == precisionTag()) {
                    graph_exec = e.exec;
                    graph_src  = e.src;
                    break;
                }
            if (graph_exec != nullptr) {
                graph_nmax      = nmax;
                graph_lanes     = lanes;
                graph_precision = precisionTag();
                CUDA_CHECK(rasbery::graphLaunchOrSplice(graph_exec, graph_src, stream));
                ++telemetry.graph_launches;
                if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
                return;
            }
            // Rev.7.1 Task 10 part 4: THE SAME RULE AS launch_sweeps', AND FOR
            // THE SAME REASON THIS SITE ALREADY GOT THE SPLICE (part 4 §5.4).
            // The BiCG outer graph is not on the RESIDENT_SINGLE segment path
            // TODAY -- but "today" is a premise nobody has to break on purpose,
            // and the failure if it is broken is a nested capture, which is
            // fatal rather than slow.  The direct enqueue is the same kernels.
            if (rasbery::graphCaptureActive(stream)) {
                rasbery::graphWarmupMiss();
                enqueue_outer(nmax);
                return;
            }
            cudaGraph_t graph = nullptr;
            cudaError_t rc    = cudaSuccess;
            {
                ScopedStreamCapture capture(stream, "cmfd.outer");
                rc = capture.begin();
                if (rc == cudaSuccess) {
                    // enqueue_outer CUDA_CHECKs its status D2H, so it can throw
                    // -- and a throw here used to leave the arena stream in
                    // capture mode for the rest of the process.  It is caught
                    // and demoted to the SAME fallback a refused capture takes,
                    // which is exact for the reason stated below: work
                    // submitted to a capturing stream is recorded, not run.
                    try {
                        enqueue_outer(nmax);
                        rc = capture.end(&graph);
                    } catch (const std::exception&) {
                        rc = cudaErrorStreamCaptureInvalidated;
                    }
                }
            }
            if (rc == cudaSuccess)
                // 3-argument form: the legacy (errorNode, logBuffer, size)
                // overload is gone in CUDA 13, which the 238 server builds with.
                rc = cudaGraphInstantiate(&graph_exec, graph, 0ull);
            // The number this whole fusion exercise is about, measured rather
            // than counted by hand.  Off by default (one stderr line would
            // otherwise land in every log that parses this one); capture is
            // rare enough -- graph_reinstantiations is 0 on a normal run --
            // that the query costs nothing when it is on.
            if (rc == cudaSuccess && graph != nullptr &&
                std::getenv("RASBERY_GPU_GRAPH_NODES") != nullptr) {
                size_t node_count = 0;
                if (cudaGraphGetNodes(graph, nullptr, &node_count) == cudaSuccess)
                    std::cerr << "[RASBERY][CUDA][GRAPH_NODES] {\"nmax\":" << nmax
                              << ",\"captured_iterations\":" << iter_batch_used
                              << ",\"nodes\":" << node_count << "}" << std::endl;
            }
            if (rc != cudaSuccess) {
                if (graph != nullptr) cudaGraphDestroy(graph);
                // Capture is a pure optimisation; a driver that refuses it
                // must not take the solver down with it.
                //
                // The fallback re-enqueues the same work, which is only correct
                // under a CUDA semantic this code depends on and never stated:
                // work submitted to a stream *in capture mode is not executed*,
                // it is only recorded, and a failed BeginCapture/EndCapture pair
                // leaves the stream out of capture mode with nothing pending.
                // So there is no partially-executed outer to undo here and no
                // double-application -- enqueue_outer(nmax) below is the first
                // and only execution of this outer.
                cudaGetLastError();
                graph_exec = nullptr;
                graph_src  = nullptr;
                destroyGraphCaches();
                use_graph  = false;
                ++telemetry.graph_fallbacks;
                enqueue_outer(nmax);
                return;
            }
            graph_src       = graph;
            graph_nmax      = nmax;
            graph_lanes     = lanes;
            graph_precision = precisionTag();
            outer_graphs.push_back(
                OuterGraph{graph_exec, nmax, lanes, precisionTag(), graph_src});
            g_cmfd_bucket_graphs.fetch_add(1, std::memory_order_relaxed);
            // WP7 stage B: the node census, at the one place where a graph is
            // built.  `sweeps` is 0 because the outer graph IS the per-sweep
            // unit -- it carries no assemble prologue and no Wielandt tail.
            reportCmfdGraphCensus("outer", graph_src, 0, iter_batch_used, fuse_mask,
                                  cmfd_block_threads(), cmfd_node_blocks(),
                                  cmfd_vector_blocks(), persistent_armed);
            // The capture itself enqueued nothing: replay it now.
        }
        CUDA_CHECK(rasbery::graphLaunchOrSplice(graph_exec, graph_src, stream));
        ++telemetry.graph_launches;
        // Counted only alongside a real graph launch, so the invariant
        // `batched_graph_launches <= graph_launches` holds on every path --
        // including graph-off and post-fallback, where there is no launch.
        if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
    }

    // -----------------------------------------------------------------------
    // Device-resident CMFD sweeps (RASBERY_GPU_CMFD_SWEEP)
    // -----------------------------------------------------------------------

    /// One launch = up to `unroll` full Wielandt sweeps.  Each sweep is the
    /// exact host sequence -- source rebuild, the whole BiCGSTAB inner
    /// (enqueue_outer, whose state reset doubles as the per-sweep halt
    /// refresh), wiel, updls, the negative census and the control tail --
    /// with sweep_halt carrying the host loop's break/retry decisions.
    void enqueue_sweeps(int nmax, int unroll) {
        // One operator build per drive. The following sweep loop only updates
        // the shifted diagonal as reigvs changes; cc and udiag stay resident.
        cmfd_assemble_operator_2g<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), surface_stride(),
            assembly_node_surface, assembly_face_area, assembly_volume,
            xs_xsrf, xs_xssm, xs_chif, xs_xsnf, dtil_dev, dhat_dev,
            diag, cc, udiag_dev, scalars, device_assembly_active, sweep_halt, d_slot_map, lanes);
        for (int sweep = 0; sweep < unroll; ++sweep) {
            cmfd_sweep_begin<<<scalar_grid(), 1, 0, stream>>>(scalars, sweep_halt, d_slot_map, lanes);
            cmfd_src_build<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), node_stride(), xs_chif, psi_dev, src, scalars,
                sweep_halt, d_slot_map, lanes);
            enqueue_outer(nmax);
            // ax/s are BiCG scratch, dead between the inner loop and the next
            // sweep's initial residual; they carry the wiel addends here.
            cmfd_wiel_terms<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), node_stride(), phi, psi_dev, xs_xsnf, node_vol,
                ax, s, sweep_halt, d_slot_map, lanes);
            // The one fork between the two fold modes.  Resolved from a
            // process-constant gate, so the captured graph's topology is fixed
            // for the run and no cache key has to carry the mode.
            if (wielFoldMode() == WielFoldMode::CHUNKED) {
                const int wiel_blocks = reduce_blocks_for(nxyz);
                reportWielFold(wiel_blocks);
                // FUSE bit 2.  Three reductions in, three reductions out, the
                // same strict ascending fold and the same cmfd_wiel_apply tail
                // -- see cmfd_wiel_fused.  Inert under the serial fold, which
                // has no stage 1 to fuse.
                if (fuse_wiel) {
                    cmfd_wiel_fused<<<dim3(static_cast<unsigned>(wiel_blocks),
                                           static_cast<unsigned>(lanes)),
                                      kReduceThreads, 0, stream>>>(
                        nxyz, vec_stride(), ax, s, wiel_partials, scalars, sweep_halt,
                        fuse_retire, d_slot_map, lanes);
                } else {
                    cmfd_wiel_stage1<<<dim3(static_cast<unsigned>(wiel_blocks),
                                            static_cast<unsigned>(lanes)),
                                       kReduceThreads, 0, stream>>>(
                        nxyz, vec_stride(), ax, s, wiel_partials, sweep_halt,
                        d_slot_map, lanes);
                    cmfd_wiel_finalize_chunked<<<scalar_grid(), 32, 0, stream>>>(
                        wiel_blocks, wiel_partials, scalars, sweep_halt, d_slot_map,
                        lanes);
                }
            } else {
                reportWielFold(0);
                cmfd_wiel_finalize<<<scalar_grid(), 32, 0, stream>>>(
                    nxyz, vec_stride(), ax, s, scalars, sweep_halt, d_slot_map, lanes);
            }
            cmfd_updls<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), node_stride(), mat_stride(), xs_chif, xs_xsnf,
                node_vol, udiag_dev, diag, scalars, sweep_halt, d_slot_map, lanes);
            cmfd_negative_scan<<<vector_grid(), block_size, 0, stream>>>(
                n, vec_stride(), phi, scalars, sweep_halt, d_slot_map, lanes);
            cmfd_sweep_end<<<scalar_grid(), 1, 0, stream>>>(scalars, sweep_halt, d_slot_map, lanes);
        }
    }

    /// Graph-cached counterpart of launch_outer for the sweep sequence.
    ///
    /// `unroll` is now a CAPACITY request, not a configuration: the launch may
    /// spend at most that many sweep slots, and the device enforces it through
    /// kSweepSlotBudget (issueSweepUploads stamps it).  A captured graph that is
    /// deeper serves the launch unchanged, so the cache only ever grows -- see
    /// SweepGraphCapacity in CudaBICGBackend.h for why that is exact and not an
    /// approximation.
    void launch_sweeps(int nmax, int unroll) {
        if (!use_graph) {
            enqueue_sweeps(nmax, unroll);
            return;
        }
        if (!sweep_graph.serves(nmax, unroll, precisionTag(), lanes)) {
            // Per-bucket cache, exactly as launch_outer.  The capacity
            // ratchet on `unroll` stays PER BUCKET: a deeper capture serves a
            // shallower launch only at the same grid.y, because grid.y is
            // baked and a wider one would dispatch padding blocks again.
            sweep_graph_exec = nullptr;
            sweep_graph_src  = nullptr;
            for (const SweepGraph& e : sweep_graphs)
                if (e.key.serves(nmax, unroll, precisionTag(), lanes)) {
                    sweep_graph_exec = e.exec;
                    sweep_graph_src  = e.src;
                    sweep_graph      = e.key;
                    break;
                }
            if (sweep_graph_exec != nullptr) {
                CUDA_CHECK(rasbery::graphLaunchOrSplice(sweep_graph_exec, sweep_graph_src,
                                                        stream));
                ++telemetry.graph_launches;
                if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
                return;
            }
            // Rev.7.1 Task 10 part 4: A MISS INSIDE A CAPTURED OUTER BODY IS
            // ANSWERED WITH THE ENQUEUE, NOT WITH A NESTED CAPTURE.
            //
            // `stream` is the segment's, and inside a WHILE body capture it is
            // recording.  cudaStreamBeginCapture on a stream already capturing
            // is refused -- and the existing failure path answers a refusal by
            // destroying the graph caches and setting use_graph=false for the
            // rest of the RUN, which is a silent performance cliff whose cause
            // would be four frames away.  The direct enqueue below is the same
            // kernels in the same order as the replay it stands in for, so the
            // body records exactly what a warm cache would have spliced.
            //
            // COUNTED, because it means the warm-up rule did not hold: the
            // segment's outer 0 runs eagerly precisely so this cache is warm by
            // the time outer 1 is captured.  The gate is graph_warmup_misses 0.
            if (rasbery::graphCaptureActive(stream)) {
                rasbery::graphWarmupMiss();
                enqueue_sweeps(nmax, unroll);
                return;
            }
            // The capacity ratchet, now PER BUCKET: start from the deepest
            // capture this grid.y already has, so a recapture is never
            // shallower than what exists and the depth settles instead of
            // oscillating -- exactly the pre-compaction property, restricted
            // to the entries a launch at this width could have used.
            sweep_graph = SweepGraphCapacity{};
            for (const SweepGraph& e : sweep_graphs)
                if (e.key.lanes == lanes && e.key.nmax == nmax &&
                    e.key.precision == precisionTag() && e.key.slots > sweep_graph.slots)
                    sweep_graph = e.key;
            const int depth = sweep_graph.captureDepth(unroll);
            cudaGraph_t graph = nullptr;
            cudaError_t rc    = cudaSuccess;
            {
                ScopedStreamCapture capture(stream, "cmfd.sweep");
                rc = capture.begin();
                if (rc == cudaSuccess) {
                    // enqueue_sweeps calls enqueue_outer, which CUDA_CHECKs --
                    // same throw, same exception-safe demotion to the fallback.
                    try {
                        enqueue_sweeps(nmax, depth);
                        rc = capture.end(&graph);
                    } catch (const std::exception&) {
                        rc = cudaErrorStreamCaptureInvalidated;
                    }
                }
            }
            if (rc == cudaSuccess)
                rc = cudaGraphInstantiate(&sweep_graph_exec, graph, 0ull);
            if (rc != cudaSuccess) {
                // Same fallback contract as launch_outer: nothing ran during a
                // failed capture, so the direct enqueue below is the first and
                // only execution.
                if (graph != nullptr) cudaGraphDestroy(graph);
                cudaGetLastError();
                sweep_graph_exec = nullptr;
                sweep_graph_src  = nullptr;
                sweep_graph      = SweepGraphCapacity{};
                destroyGraphCaches();
                use_graph        = false;
                ++telemetry.graph_fallbacks;
                enqueue_sweeps(nmax, unroll);
                return;
            }
            // Rev.7.1 Task 10: THE cudaGraphDestroy THAT USED TO BE HERE IS GONE.
            //
            // It stood on the line after the instantiate and it was right until
            // the outer body became capturable: a cudaGraphLaunch into a
            // capturing stream is refused AND invalidates the capture, so the
            // only way this sweep enters a captured outer body is as a child
            // graph node -- and cudaGraphAddChildGraphNode needs the graph, which
            // an exec cannot be turned back into.  Ownership moves into the cache
            // entry below; destroyGraphCaches() destroys the pair.
            sweep_graph_src = graph;
            sweep_graph = SweepGraphCapacity{nmax, depth, precisionTag(), lanes};
            sweep_graphs.push_back(SweepGraph{sweep_graph_exec, sweep_graph, sweep_graph_src});
            g_cmfd_bucket_graphs.fetch_add(1, std::memory_order_relaxed);
            // WP7 stage B: the node census.  `depth` sweeps ride in this
            // capture, and exactly one cmfd_assemble_operator_2g node sits in
            // front of them.
            reportCmfdGraphCensus("sweep", sweep_graph_src, depth, iter_batch_used,
                                  fuse_mask, cmfd_block_threads(), cmfd_node_blocks(),
                                  cmfd_vector_blocks(), persistent_armed);
        }
        CUDA_CHECK(rasbery::graphLaunchOrSplice(sweep_graph_exec, sweep_graph_src, stream));
        ++telemetry.graph_launches;
        if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
    }

    /// Rev.7.1 Task 10 part 2: mask this slot's sweep when the segment has halted.
    ///
    /// Issued AFTER issueSweepUploads, which is what uploads the participation
    /// masks this kernel then overrides for the one slot, and BEFORE
    /// launch_sweeps, because every kernel of the graph tests what it writes.
    void enqueueSweepGate(int m, const std::uint32_t* outer_halt, int outer_slot) {
        if (outer_halt == nullptr) return;
        cmfd_sweep_gate<<<1, 1, 0, stream>>>(sweep_halt, outer_halt, outer_slot, m);
        CUDA_CHECK(cudaGetLastError());
    }

    /// Publish the outer segment's probe from device memory (no readback).
    void enqueueSweepVerdict(int m, const CudaBatchArena::CmfdSweepProbeSink& p) {
        cmfd_sweep_verdict<<<1, 1, 0, stream>>>(scalars, m, p.eigv, p.residual, p.negative,
                                                p.rayleigh, p.nonfinite, p.halt,
                                                p.halt_slot,
                                                reinterpret_cast<double*>(p.accum));
        CUDA_CHECK(cudaGetLastError());
    }

    /// Rev.7.1 Task 10 part 3: take this launch's eigenvalue from the probe.
    ///
    /// Issued in the same window cmfd_sweep_gate is -- AFTER issueSweepUploads,
    /// whose H2D of the staged block is what it overwrites, and BEFORE
    /// launch_sweeps, whose first kernel reads it.
    void enqueueSweepPatch(int m, const CudaBatchArena::CmfdSweepProbeSink& p) {
        if (!p.patch_from_probe) return;
        cmfd_sweep_patch<<<1, 1, 0, stream>>>(scalars, m, p.eigv, p.residual);
        CUDA_CHECK(cudaGetLastError());
    }

    /// The gate and the patch, in that order -- as two launches (the reference)
    /// or as one (FUSE bit 3).
    ///
    /// The one-launch form reproduces the reference's ISSUE decisions as well as
    /// its arithmetic: the gate is a no-op when `outer_halt` is null, the patch
    /// is not issued at all when `patch_from_probe` is false (null probe
    /// pointers say the same thing inside the kernel), and when NEITHER is
    /// wanted the pair enqueued nothing, so neither does this.
    void enqueueSweepPreamble(int m, const std::uint32_t* outer_halt, int outer_slot,
                              const CudaBatchArena::CmfdSweepProbeSink& p) {
        if (fuse_sweep_pre) {
            if (outer_halt == nullptr && !p.patch_from_probe) return;
            cmfd_sweep_gate_patch<<<1, 1, 0, stream>>>(
                sweep_halt, outer_halt, outer_slot, m, scalars,
                p.patch_from_probe ? p.eigv : nullptr,
                p.patch_from_probe ? p.residual : nullptr);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        enqueueSweepGate(m, outer_halt, outer_slot);
        enqueueSweepPatch(m, p);
    }

    /// H2D for one sweep batch.  chif/vol mirror away their (rare/never)
    /// changes; xsnf, udiag, psi and the sweep scalars are new every drive.
    /// The per-slot sweep_halt starts at 0 for participants, 1 for everyone
    /// else -- and is restored to all-zero by finishSweeps so the plain solve
    /// path never sees a stale mask.
    /// `slot_budget` is how many sweep slots THIS launch may spend -- the value
    /// that used to be the graph's capture depth.  It is stamped into every
    /// participant's staged scalar block here, right before the H2D, because it
    /// is a property of the launch and not of any one slot: the pre-Task-6 code
    /// gave every participant the batch-wide max, and preserving that exactly is
    /// what keeps the retry packing (and therefore `state == 0`) unchanged.
    void issueSweepUploads(const int* active_slots, int count, int slot_budget) {
        buildSlotMap(active_slots, count);
        for (int i = 0; i < count; ++i) {
            Slot& sl = slot[static_cast<size_t>(active_slots[i])];
            sl.sweep_in[kSweepSlotBudget - kSweepFirst] = static_cast<double>(slot_budget);
            sl.sweep_in[kSweepSlots - kSweepFirst]      = 0.0;
        }
        std::uint32_t* const active   = stageActive();
        std::uint32_t* const assembly = stageAssemblyActive();
        std::uint32_t* const halt     = stageSweepHalt();
        std::memset(active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        std::memset(assembly, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            active[m]   = 1u;
            assembly[static_cast<size_t>(m)] =
                slot[static_cast<size_t>(m)].device_assembly ? 1u : 0u;
        }
        // participants: sweep_halt = 0; everyone else: 1 (masks their slots
        // inside every sweep kernel AND the inner reset).
        //
        // BUILT BEFORE EITHER UPLOAD, AND IN ITS OWN BUFFER.  This used to be
        // `for (m) host_active[m] = host_active[m] ? 0 : 1;` placed BETWEEN the
        // device_active copy and the sweep_halt copy, i.e. a host write to the
        // source range of a cudaMemcpyAsync that nothing had synchronised.
        // host_active is cudaMallocHost'd, so that copy is a real DMA with no
        // guarantee about WHEN it reads: the driver may stage it inline at
        // call time (in which case the inversion is invisible and the run is
        // correct) or defer it (in which case device_active receives the
        // INVERTED mask -- every participant reads active == 0).  Which one it
        // does depends on the copy engine's queue state, so the same binary
        // flips between the two: initialize_solver_state then computes
        // halt[m] = 1 for the participant, the whole BiCGSTAB inner loop of
        // that sweep is masked off while the Wielandt tail still advances psi
        // and the eigenvalue from the un-updated flux, and the drive converges
        // to a neighbouring iterate.  That is the 1e-14..1e-13 run-to-run
        // drift, and the retry/negative-flux path it occasionally steers into
        // is the non-finite abort.
        //
        // The rule this restores: NO HOST BUFFER THAT IS THE SOURCE OF AN
        // IN-FLIGHT cudaMemcpyAsync MAY BE WRITTEN BEFORE THAT COPY IS KNOWN
        // TO HAVE COMPLETED.  One buffer per upload is the cheap way to obey
        // it (slots uint32s, once per launch); an event or a sync here would
        // cost the pipeline this path exists to keep full.
        // tools/test_cmfd_async_h2d_snapshot_contract.py pins it.
        for (int m = 0; m < slots; ++m) halt[m] = active[m] ? 0u : 1u;

        // WP13: the two participation masks are device-read-only and shadowed;
        // sweep_halt is NOT (see pushDeviceReadOnly's header) and keeps its
        // unconditional copy.
        pushDeviceReadOnly("issueSweepUploads:device_active", device_active, active,
                           static_cast<size_t>(slots), shadow_active);
        pushDeviceReadOnly("issueSweepUploads:device_assembly_active",
                           device_assembly_active, assembly,
                           static_cast<size_t>(slots), shadow_assembly_active);
        CUDA_CHECK(rasbery::xfer::memcpyAsync(
            "CudaBICGBackend.cu:issueSweepUploads", "sweep_halt", sweep_halt, halt,
            static_cast<size_t>(slots) * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
            stream));

        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];

            pushOrSkip("issueSweepUploads:chif", xs_chif + m * vec_stride(), sl.host_chif,
                       static_cast<size_t>(n), sl.chif_mirror.valid
                           ? !mirrorMatches(sl.chif_mirror, sl.host_chif,
                                            static_cast<size_t>(n))
                           : true,
                       sl.chif_mirror);
            pushOrSkip("issueSweepUploads:vol", node_vol + m * node_stride(), sl.host_vol,
                       static_cast<size_t>(nxyz), sl.vol_mirror.valid
                           ? !mirrorMatches(sl.vol_mirror, sl.host_vol,
                                            static_cast<size_t>(nxyz))
                           : true,
                       sl.vol_mirror);

            auto push = [&](const char* leaf, double* dst, const double* src_host,
                            size_t cnt) {
                if (src_host == nullptr)
                    throw std::invalid_argument("CMFD sweep upload received a null host buffer");
                CUDA_CHECK(rasbery::xfer::memcpyAsync(
                    "CudaBICGBackend.cu:issueSweepUploads:push", leaf, dst, src_host,
                    cnt * sizeof(double), cudaMemcpyHostToDevice, stream));
                ++telemetry.bulk_h2d_calls_during_iteration;
                telemetry.bulk_h2d_bytes_during_iteration += cnt * sizeof(double);
            };
            auto push_pending = [&](const char* leaf, double* dst, const double* src_host,
                                    size_t cnt, bool& pushed,
                                    cuda_transfer::ByteExactMirror<double>& mirror) {
                if (src_host == nullptr)
                    throw std::invalid_argument("CMFD assembly upload received a null host buffer");
                pushed = !mirror.matches(src_host, cnt);
                if (!pushed) {
                    ++telemetry.bulk_h2d_skipped_during_iteration;
                    return;
                }
                push(leaf, dst, src_host, cnt);
                // ==========================================================
                // Rev.7.1 Task 10 part 3: COMMITTED AT THE ISSUE, NOT AT THE
                // OBSERVATION
                // ==========================================================
                //
                // WHAT A SHADOW IS SUPPOSED TO MEAN: `this is what the DEVICE
                // holds`.  It used to be written at the OBSERVATION instead, from
                // inside absorb() -- i.e. from the host bytes AS THEY ARE WHEN
                // THE LAUNCH IS OBSERVED, which is a different moment from when
                // the copy was handed its source.  While the observation
                // followed every launch immediately that distinction had no
                // room to matter.
                //
                // A HOST-FREE SEGMENT GIVES IT ROOM.  There the observation is
                // deferred to the segment exit, so `pushed` survives from a
                // launch inside the segment until an absorb outside it -- and
                // whatever rewrote the host array in between (the boron trial
                // commit is the one that bit) is then recorded as if it had been
                // uploaded.  The very next launch compares equal and skips the
                // upload it needed most.
                //
                // MEASURED, kngr3, budget 2: statepoint 2's second boron trial.
                // The host xsrf moved to 0c514dd11a55b1b7, the shadow claimed
                // it, the device kept 95425148870c3384, and the sweep solved a
                // reactor with the previous trial's removal cross sections --
                // k_eff 1.0000507 where the host gets 0.9999139, 41 of 644
                // datasets and four extra outers.
                //
                // COMMITTING HERE IS WHAT pushOrSkip HAS ALWAYS DONE (three
                // lines up, `mirror.shadow.assign(host_buffer, ...)`) and what
                // the segment's own stageXsnf does with the same justification:
                // the shadow records the bytes the copy was HANDED, so a writer
                // that arrives afterwards is caught by the next comparison
                // instead of being absorbed into it.
                mirror.commit(src_host, cnt);
            };

            if (sl.device_assembly) {
                push_pending("xsnf", xs_xsnf + m * vec_stride(), sl.host_xsnf,
                             static_cast<size_t>(n), sl.pushed_xsnf,
                             sl.xsnf_mirror);
                push_pending("xsrf", xs_xsrf + m * vec_stride(), sl.host_xsrf,
                             static_cast<size_t>(n), sl.pushed_xsrf,
                             sl.xsrf_mirror);
                push_pending("xssm", xs_xssm + m * mat_stride(), sl.host_xssm,
                             matrix_count, sl.pushed_xssm, sl.xssm_mirror);
                push_pending("dtil", dtil_dev + m * surface_stride(), sl.host_dtil,
                             surface_group_count, sl.pushed_dtil,
                             sl.dtil_mirror);
                // dhat changes after every nodal correction, so comparing it
                // on the launcher's critical path is wasted work.
                //
                // UNLESS THE DEVICE OUTER SEGMENT ALREADY WROTE IT (Task 9 link
                // 2).  Its upddhat kernel writes THIS buffer for THIS slot, so
                // the host array is one outer behind and copying it would undo
                // the segment's work.  Skipping is the only correct action here,
                // not an optimisation.
                if (sl.dhat_resident) {
                    ++telemetry.bulk_h2d_skipped_during_iteration;
                    telemetry.cmfd_dhat_h2d_elided_bytes +=
                        surface_group_count * sizeof(double);
                } else {
                    push("dhat", dhat_dev + m * surface_stride(), sl.host_dhat,
                         surface_group_count);
                }

                ++telemetry.cmfd_assembly_gpu_calls;
                telemetry.cmfd_diag_h2d_elided_bytes += matrix_count * sizeof(double);
                telemetry.cmfd_cc_h2d_elided_bytes += coupling_count * sizeof(double);
                telemetry.bulk_h2d_skipped_during_iteration += 3; // diag/cc/udiag

                // The assembly kernel overwrites cc. A former host mirror no
                // longer describes device memory and must not elide a later
                // rollback upload.
                sl.cc_mirror.valid = false;
            } else {
                sl.pushed_xsnf = sl.pushed_xsrf = sl.pushed_xssm = sl.pushed_dtil = false;
                // This branch overwrites xs_xsnf without going through the
                // mirror; a stale shadow must not elide a later assembly-path
                // upload if the slot toggles back.
                sl.xsnf_mirror.invalidate();
                push("xsnf (host assembly)", xs_xsnf + m * vec_stride(), sl.host_xsnf,
                     static_cast<size_t>(n));
                push("udiag (host assembly)", udiag_dev + m * mat_stride(),
                     sl.host_udiag, matrix_count);
                push("diag (host assembly)", diag + m * mat_stride(), sl.host_diag,
                     matrix_count);
                pushOrSkip("issueSweepUploads:cc", cc + m * cpl_stride(), sl.host_cc,
                           coupling_count, sl.push_cc, sl.cc_mirror);
                ++telemetry.cmfd_assembly_cpu_fallbacks;
            }

            // psi round trip, removed.  See CmfdSweepIO::psi_dirty: only the
            // first launch of a drive carries host-written psi; a later launch
            // in the same drive would be re-uploading the bytes the device
            // itself produced, so leaving the device copy alone is the same
            // state by a shorter path.
            //
            // THE RESIDENCY TEST OUTRANKS psi_dirty (Task 9 link 2).  psi_dirty
            // means 'the host wrote psi since the last launch', which is true at
            // every drive boundary -- but when the device outer segment owns psi
            // the host never wrote it at all, and uploading the host twin would
            // overwrite the fission source the segment's updpsi just produced.
            if (sl.psi_resident) {
                ++telemetry.bulk_h2d_skipped_during_iteration;
                telemetry.cmfd_resident_psi_h2d_elided_bytes +=
                    static_cast<std::uint64_t>(nxyz) * sizeof(double);
            } else if (sl.push_psi) {
                push("psi", psi_dev + m * node_stride(), sl.host_psi,
                     static_cast<size_t>(nxyz));
            } else {
                ++telemetry.bulk_h2d_skipped_during_iteration;
                telemetry.cmfd_psi_h2d_elided_bytes +=
                    static_cast<std::uint64_t>(nxyz) * sizeof(double);
            }
            pushOrSkip("issueSweepUploads:phi", phi + m * vec_stride(), sl.host_phi,
                       static_cast<size_t>(n), sl.push_phi, sl.phi_mirror);
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueSweepUploads", "sweep_in",
                scalars + static_cast<long long>(m) * kScalarCount + kSweepFirst,
                sl.sweep_in, kSweepCount * sizeof(double), cudaMemcpyHostToDevice,
                stream));
            if (!(sl.eps == sl.eps_on_device)) {
                CUDA_CHECK(rasbery::xfer::memcpyAsync(
                    "CudaBICGBackend.cu:issueSweepUploads", "eps",
                    scalars + static_cast<long long>(m) * kScalarCount + kEps, &sl.eps,
                    sizeof(double), cudaMemcpyHostToDevice, stream));
                sl.eps_on_device = sl.eps;
            }
            sl.phi_mirror.valid = false;
        }
    }

    /// D2H after the sweep graph: flux (issueFluxDownloads) and the sweep
    /// scalar block per participant, then the sweep_halt restore.
    ///
    /// psi is NOT here any more.  It used to come back after every launch, and
    /// the only reader of those bytes was the degenerate-gamma (state == 2)
    /// hand-back inside BICGCMFD::driveDeviceSweeps -- one launch in many.  On
    /// every other path the host overwrites psi wholesale (CMFD::updpsi from
    /// the flux) before it is next read, so the download was writing bytes
    /// nobody looks at.  The exceptional launch pulls it in
    /// issueExceptionalOperatorDownloads, where the state is already known.
    void issueSweepDownloads(const int* active_slots, int count) {
        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];
            sl.psi_downloaded = false;
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueSweepDownloads", "sweep_out", sl.sweep_out,
                scalars + static_cast<long long>(m) * kScalarCount + kSweepFirst,
                kSweepCount * sizeof(double), cudaMemcpyDeviceToHost, stream));
            ++telemetry.bulk_d2h_calls_during_iteration;
            telemetry.bulk_d2h_bytes_during_iteration +=
                static_cast<std::uint64_t>(kSweepCount) * sizeof(double);
            telemetry.cmfd_psi_d2h_elided_bytes +=
                static_cast<std::uint64_t>(nxyz) * sizeof(double);
        }
        CUDA_CHECK(cudaMemsetAsync(sweep_halt, 0,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   stream));
    }

    /// Queue the flux D2H for every participant.
    void issueFluxDownloads(const int* active_slots, int count) {
        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueFluxDownloads", "out_phi", sl.out_phi,
                phi + m * vec_stride(), static_cast<size_t>(n) * sizeof(double),
                cudaMemcpyDeviceToHost, stream));
            ++telemetry.bulk_d2h_calls_during_iteration;
            telemetry.bulk_d2h_bytes_during_iteration +=
                static_cast<std::uint64_t>(n) * sizeof(double);
            ++telemetry.status_d2h_calls_during_iteration;
        }
    }

    // THE DEFERRED MIRROR COMMIT IS GONE.  Rev.7.1 Task 10 part 3 moved all four
    // of the assembly shadows (xsnf, xsrf, xssm, dtil) to their own issue site,
    // in issueSweepUploads' push_pending, where a shadow can only ever record
    // bytes a copy was actually handed.  Written at the OBSERVATION instead --
    // which is where they used to be, inside absorb() -- they record whatever the
    // host array holds THEN, and a deferred observation gives a boron trial
    // commit a whole segment in which to slip inside that window.
    // The `pushed_*` flags stay: they are what slotIsPristine tests, and they
    // still say `this launch uploaded this array`.

    /// A degenerate Wielandt gamma hands control to the host Rayleigh branch.
    /// Only those exceptional slots need a host copy of the operator that the
    /// assembly kernel produced; the normal path never downloads diag/cc/udiag.
    /// @param forced when true the caller has established `state == 2` from
    ///        somewhere other than the staging block.  A deferred segment's
    ///        block belongs to whichever launch was enqueued LAST, which is not
    ///        the launch that handed back -- the accumulator holds that one.
    void issueExceptionalOperatorDownloads(const int* active_slots, int count,
                                           bool forced = false) {
        bool queued = false;
        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            Slot& sl = slot[static_cast<size_t>(m)];
            const int state = static_cast<int>(sl.sweep_out[kSweepState - kSweepFirst]);
            if (!forced && state != 2) continue;
            // The Rayleigh hand-back in BICGCMFD reads psi(l) for sumf/summ,
            // and this is the one launch on which it does.  Unconditional on
            // device_assembly: the host arm needs the new fission source too.
            if (sl.host_psi != nullptr && !sl.psi_downloaded) {
                CUDA_CHECK(rasbery::xfer::memcpyAsync(
                    "CudaBICGBackend.cu:issueExceptionalOperatorDownloads", "psi",
                    sl.host_psi, psi_dev + m * node_stride(),
                    static_cast<size_t>(nxyz) * sizeof(double), cudaMemcpyDeviceToHost,
                    stream));
                sl.psi_downloaded = true;
                ++telemetry.bulk_d2h_calls_during_iteration;
                telemetry.bulk_d2h_bytes_during_iteration +=
                    static_cast<std::uint64_t>(nxyz) * sizeof(double);
                telemetry.cmfd_psi_d2h_elided_bytes -=
                    static_cast<std::uint64_t>(nxyz) * sizeof(double);
                queued = true;
            }
            if (!sl.device_assembly) continue;
            if (sl.host_diag_out == nullptr || sl.host_cc_out == nullptr ||
                sl.host_udiag == nullptr)
                throw std::runtime_error(
                    "CMFD device assembly fallback has no writable host operator buffers");
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueExceptionalOperatorDownloads", "diag",
                sl.host_diag_out, diag + m * mat_stride(), matrix_count * sizeof(double),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueExceptionalOperatorDownloads", "cc",
                sl.host_cc_out, cc + m * cpl_stride(), coupling_count * sizeof(double),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(rasbery::xfer::memcpyAsync(
                "CudaBICGBackend.cu:issueExceptionalOperatorDownloads", "udiag",
                sl.host_udiag, udiag_dev + m * mat_stride(),
                matrix_count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            telemetry.bulk_d2h_calls_during_iteration += 3;
            telemetry.bulk_d2h_bytes_during_iteration +=
                (2 * static_cast<std::uint64_t>(matrix_count) +
                 static_cast<std::uint64_t>(coupling_count)) * sizeof(double);
            queued = true;
        }
        if (!queued) return;
        ++telemetry.stream_sync_calls_during_iteration;
        CUDA_CHECK(rasbery::xfer::streamSync(
            "CudaBICGBackend.cu:issueExceptionalOperatorDownloads", "drain", stream));
        CUDA_CHECK(cudaGetLastError());
    }

    /// The one drain per launch.  It covers the flux copies *and* the status
    /// packets the graph already queued, which is why the inner loop needs none.
    void drain(const int* active_slots, int count) {
        ++telemetry.stream_sync_calls_during_iteration;
        CUDA_CHECK(rasbery::xfer::streamSync("CudaBICGBackend.cu:BatchCore::drain", "launch",
                                      stream));
        CUDA_CHECK(cudaGetLastError());
        absorb(active_slots, count);
    }

    /// The half of drain() that is NOT the synchronise.
    ///
    /// Split out for the Rev.7.1 Task 10 part 2 enqueue path, which owns its own
    /// stream and synchronises once for the whole outer -- a second sync here
    /// would be the round trip that path exists to remove.  drain() is this plus
    /// the sync, so there is still one body and the two cannot drift.
    void absorb(const int* active_slots, int count) {
        const bool fp32_was_active = fp32Active();
        bool       fp32_failed     = false;

        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            // The flux mirror is NOT recorded here: count*n double copies on
            // the launcher's critical path kept `launching` set while the next
            // batch starved.  Each participant adopts its own mirror on its own
            // thread on the way out of solve() -- see adoptFluxMirror().
            telemetry.cmfd_gpu_calls += host_status[m].flux_gen;
            telemetry.bicg_restarts += host_status[m].material_gen;
            telemetry.bicg_early_convergence_exits += host_status[m].operator_gen;
            telemetry.overrun_iterations += host_status[m].outer_iter;
            // A non-finite flux is THAT instance's failure, not the batch's:
            // recorded per slot here, thrown from the owning thread on its way
            // out of solve().  The old batch-fatal throw took every batch-mate
            // down with the diverging deck, which turned one bad candidate in
            // a GA screen into a build-dependent set of collateral failures.
            const bool nonfinite = (host_status[m].flags & NONFINITE_DETECTED) != 0;
            if (nonfinite && fp32_was_active) {
                // MIXED-PRECISION SAFETY VALVE, deliberately env-independent.
                //
                // The FP32 kernels refuse to write a non-finite flux, so the
                // slot comes back holding the iterate it entered the outer
                // with: the failed FP32 attempt is DISCARDED, never accepted.
                // Absorb it once, move the whole arena back to FP64 and let the
                // outer Wielandt loop carry on from that last good iterate --
                // which is the same self-healing the BiCGSTAB restart
                // convention already relies on.  If the flux is genuinely
                // diverging, the FP64 path hits it on the very next outer and
                // raises it the historical way, because fp32 is off by then.
                ++telemetry.fp32_fallbacks;
                fp32_failed                            = true;
                slot[static_cast<size_t>(m)].nonfinite = false;
            } else {
                slot[static_cast<size_t>(m)].nonfinite = nonfinite;
            }
        }

        if (fp32_failed) latchFp32Off();
    }

    /// Retire the mixed-precision path for the rest of the process.
    ///
    /// Arena-wide rather than per slot, and that is a property of the design
    /// rather than an omission: one captured graph serves every slot of a
    /// launch, so a per-slot precision would mean carrying BOTH kernel sets in
    /// the graph and masking one of them -- doubling the node count, which is
    /// the cost this whole campaign is trying to remove.  Dropping the cached
    /// graphs is what makes the switch take effect; the next launch re-captures
    /// the FP64 topology.
    ///
    /// Rev.7.1 Task 18 CHANGED WHAT MAKES THAT SAFE, not whether it is.  The
    /// old reason was "called from drain(), i.e. with the stream already
    /// synchronised" -- true while the rendezvous elected one launcher that
    /// drained its own batch before absorbing it, and no longer true now that
    /// absorb() is also reached from the stream-ordered path, where another
    /// deck's launch of one of these executables may still be in flight on the
    /// arena stream.  The reason it is still safe is cudaGraphExecDestroy's own
    /// contract: a graph that is executing is destroyed when it finishes, not
    /// under the launch.  What DOES need the claim is the cache itself, and
    /// every caller of this function holds it.
    void latchFp32Off() {
        if (fp32_latched_off) return;
        fp32_latched_off = true;
        // Precision is part of every cache key, so a latched fallback could in
        // principle just miss; dropping the FP32 instantiations outright is
        // what makes "the run FINISHED in fp64" a property of the process
        // rather than of which bucket arrives next.
        destroyGraphCaches();
        std::cerr << "[RASBERY][CUDA][FP32_FALLBACK] {\"reason\":\"nonfinite\","
                  << "\"fp32_fallbacks\":" << telemetry.fp32_fallbacks
                  << ",\"precision\":\"fp64\"}" << std::endl;
    }

    /// Host and device agree on slot m's flux once its batch drained; record
    /// it so the next stage() can skip the upload.  Runs on the OWNING
    /// instance's thread after batch completion, never on the launcher's
    /// critical path, and never on a failed batch (the flux is undefined).
    void adoptFluxMirror(int m) {
        Slot& sl = slot[static_cast<size_t>(m)];
        if (!phiMirrorEnabled()) {
            // Nothing will consult it, and a stale shadow must never be able
            // to elide an upload if the gate is flipped mid-run.
            sl.phi_mirror.valid = false;
            g_cmfd_phi_mirror_bypassed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        sl.phi_mirror.shadow.assign(sl.out_phi, sl.out_phi + n);
        sl.phi_mirror.valid = true;
        g_cmfd_phi_mirror_ns.fetch_add(
            static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count()),
            std::memory_order_relaxed);
        g_cmfd_phi_mirror_calls.fetch_add(1, std::memory_order_relaxed);
    }

    int           slots;
    int           nxyz;
    int           n;
    size_t        surface_group_count;
    size_t        matrix_count;
    size_t        coupling_count;
    std::vector<int> topology_neighbors;
    std::vector<int> topology_node_surface;
    std::vector<double> topology_face_area;
    std::vector<double> topology_volume;
    bool          available = false;
    std::string   status;
    cublasHandle_t handle = nullptr;
    cudaStream_t  stream = nullptr;
    int*          neighbors = nullptr;
    int*          colors = nullptr;
    int*          assembly_node_surface = nullptr;
    double*       assembly_face_area = nullptr;
    double*       assembly_volume = nullptr;
    int           block_size = kDefaultBlockSize;
    /// WP17 RASBERY_GPU_CMFD_BLOCK.  0 = unchanged; see cmfd_block_threads().
    int           cmfd_block = 0;
    /// WP17 persistent arm (RASBERY_GPU_CMFD_PERSISTENT).  `persistent_armed`
    /// is the only thing enqueue_iteration consults; the other three exist so
    /// the receipt can say WHY, by name, rather than only whether.
    bool              persistent_request    = false;
    bool              persistent_armed      = false;
    bool              cooperative_supported = false;
    int               persistent_blocks     = 0;
    PersistentRefusal persistent_refusal    = PersistentRefusal::ArmOff;
    int           rb_sweeps = 4;
    int           ncolors   = 2; // sweep colour count (2 = historical red/black; >2 under the rotational fold)
    double *diag = nullptr, *dinv = nullptr, *cc = nullptr, *src = nullptr, *phi = nullptr;
    double *r = nullptr, *r0 = nullptr, *p = nullptr, *v = nullptr, *s = nullptr, *t = nullptr;
    double *y = nullptr, *z = nullptr, *ax = nullptr;
    double*        partials = nullptr;
    double*        partials2 = nullptr;
    /// [slot][3][kMaxReduceBlocks] partials for the opt-in chunked Wielandt fold.
    double*        wiel_partials = nullptr;
    /// WP7 stage B: per-slot retire counter for the FUSE bitmask's fused
    /// reductions.  Zeroed once at allocation and SELF-REARMING thereafter
    /// (atomicInc wraps the last ticket back to 0), so it costs no memset node
    /// and a halted launch leaves it at 0.  One array serves every fused kernel
    /// because those kernels are stream-ordered and never overlap.  Allocated
    /// whatever the mask, so flipping RASBERY_GPU_CMFD_FUSE never changes the
    /// arena's shape.
    unsigned int*  fuse_retire = nullptr;
    double*        scalars = nullptr;
    std::uint32_t* device_flags = nullptr;
    std::uint32_t* iter_flags = nullptr;
    std::uint32_t* device_halt = nullptr;
    std::uint32_t* device_active = nullptr;
    std::uint32_t* device_counters = nullptr;
    DeviceSolveStatus* device_status = nullptr;
    DeviceSolveStatus* host_status = nullptr;
    /// Rev.7.1 Task 18: (slots+1) x slots, one staging LANE per slot plus one
    /// for the rendezvous launcher.  See the allocation for why the host side
    /// needs lanes and the device side does not.
    std::uint32_t*     host_active = nullptr;
    /// Staging for the sweep_halt H2D.  Separate from host_active on purpose:
    /// both are page-locked and both are memcpyAsync SOURCES in the same
    /// launcher window, so neither may be rewritten to serve the other.
    std::uint32_t*     host_sweep_halt = nullptr;
    /// Rev.7.1 Task 10 part 4 precondition (a): the page-locked home of every
    /// slot's sweep scalar block, `slots` x 2 x kSweepCount doubles.  See
    /// Slot::sweep_in and bindSweepLanes().
    double*            host_sweep_scalars = nullptr;
    /// Which lane the launch being staged writes into.  Set by the caller
    /// under `stream_mutex` and read by buildSlotMap / issueUploads /
    /// issueSweepUploads; `slots` is the rendezvous lane.
    int                stage_lane = 0;
    [[nodiscard]] std::uint32_t* stageActive() const {
        return host_active + static_cast<std::size_t>(stage_lane) * slots;
    }
    [[nodiscard]] std::uint32_t* stageSweepHalt() const {
        return host_sweep_halt + static_cast<std::size_t>(stage_lane) * slots;
    }
    [[nodiscard]] std::uint32_t* stageAssemblyActive() const {
        return host_assembly_active + static_cast<std::size_t>(stage_lane) * slots;
    }
    [[nodiscard]] int* stageSlotMap() const {
        return h_slot_map + static_cast<std::size_t>(stage_lane) * slots;
    }
    BackendCounters telemetry{};
    /// WP13 (RASBERY_GPU_XFER_ELIDE): host shadows of the three device-read-only
    /// masks.  Empty means "nothing known about the device copy", which is the
    /// state every one of them is in until its first upload -- including after
    /// the constructor's cudaMemset, so the first launch always copies.
    /// Written ONLY by pushDeviceReadOnly, and only when the flag is on, so the
    /// OFF arm allocates nothing and compares nothing.
    std::vector<int>           shadow_slot_map;
    std::vector<std::uint32_t> shadow_active;
    std::vector<std::uint32_t> shadow_assembly_active;
    std::vector<Slot> slot;
    std::uint32_t* host_assembly_active = nullptr; // pinned, (slots+1) lanes
    bool          use_graph = true;
    bool          scalar_fusion = true;
    /// WP7 stage B: RASBERY_GPU_CMFD_FUSE, latched once at construction so the
    /// captured topology cannot change between two outers of the same run.  The
    /// four booleans are the mask, unpacked at the enqueue sites.
    unsigned      fuse_mask      = 0u;
    bool          fuse_dot       = false;
    bool          fuse_dot2      = false;
    bool          fuse_wiel      = false;
    bool          fuse_sweep_pre = false;
    /// RASBERY_GPU_ITER_BATCH: requested iterations per graph launch.  0 =
    /// unset = follow the algorithmic budget (today's behaviour, and the only
    /// setting for which the capture depth is exactly `1 + nmax`).
    int           iter_batch_request = 0;
    /// What the last capture actually carried.
    int           iter_batch_used = 0;
    cudaGraphExec_t graph_exec = nullptr;
    /// The cudaGraph_t `graph_exec` was instantiated from.  Owned by the
    /// matching outer_graphs entry, not by this pointer.
    cudaGraph_t   graph_src  = nullptr;
    int           graph_nmax = -1;
    /// One instantiation per (nmax, grid.y, precision) for the outer, and per
    /// (nmax, capture depth, precision, grid.y) for the sweep sequence.  Both
    /// key spaces are bounded by the nine-entry bucket ladder, so these lists
    /// saturate at a handful of entries and are scanned linearly.
    struct OuterGraph {
        cudaGraphExec_t exec;
        int             nmax;
        int             lanes;
        int             precision;
        /// The graph the exec was made from, kept for the same reason
        /// SweepGraph::src is: a cudaGraphLaunch cannot enter a capture, and
        /// cudaGraphAddChildGraphNode cannot take an exec.  This one is not on
        /// the outer-segment path today -- launch_outer is reached through the
        /// non-arena solveInner and the batch rendezvous -- and it is kept
        /// anyway, because "not reachable today" is the kind of premise that
        /// stops being true without anybody editing this line.
        cudaGraph_t     src = nullptr;
    };
    struct SweepGraph {
        cudaGraphExec_t    exec;
        SweepGraphCapacity key;
        /// Rev.7.1 Task 10: THE GRAPH THE EXEC WAS MADE FROM, KEPT.
        ///
        /// Every other cache in this file destroys it on the next line, which is
        /// idiomatic and was right until the outer body became something that
        /// gets CAPTURED.  A cudaGraphLaunch into a capturing stream is refused
        /// -- cudaErrorStreamCaptureUnsupported -- and the refusal invalidates
        /// the whole capture, so the sweep can only enter a captured outer body
        /// as a CHILD GRAPH NODE, and cudaGraphAddChildGraphNode takes a
        /// cudaGraph_t.  An exec cannot be turned back into one.  Measured:
        /// tools/probe_while_body_capture.cu, records `graph_launch_in_capture`
        /// (false) and `child_graph_node` (true).
        ///
        /// It costs the node descriptors -- kilobytes per bucket entry, and the
        /// ladder saturates at a handful -- for the life of the run, and it must
        /// be destroyed WITH its exec or the cache leaks two objects per entry
        /// instead of one.
        cudaGraph_t        src = nullptr;
    };
    std::vector<OuterGraph> outer_graphs;
    std::vector<SweepGraph> sweep_graphs;

    void destroyGraphCaches() {
        for (const OuterGraph& e : outer_graphs) {
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
            if (e.src != nullptr) cudaGraphDestroy(e.src);
        }
        for (const SweepGraph& e : sweep_graphs) {
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
            if (e.src != nullptr) cudaGraphDestroy(e.src);
        }
        if (!outer_graphs.empty() || !sweep_graphs.empty())
            telemetry.graph_reinstantiations +=
                outer_graphs.size() + sweep_graphs.size();
        outer_graphs.clear();
        sweep_graphs.clear();
        graph_exec       = nullptr;
        graph_src        = nullptr;
        graph_nmax       = -1;
        graph_lanes      = -1;
        graph_precision  = -1;
        sweep_graph_exec = nullptr;
        sweep_graph_src  = nullptr;
        sweep_graph      = SweepGraphCapacity{};
    }

    /// grid.y the cached outer graph was captured at (a graph bakes it).
    int           graph_lanes = -1;

    // ---- active-slot compaction (RASBERY_GPU_CMFD_COMPACT, default OFF) ----
    /// Logical dispatch lane -> physical slot; -1 is a padding lane.  Both
    /// pointers are allocated once and NEVER move: d_slot_map is a kernel
    /// argument baked into every captured graph, so only its contents change.
    int*          d_slot_map = nullptr;
    int*          h_slot_map = nullptr; // pinned
    /// grid.y of the launch being enqueued.  Equals `slots` with compaction
    /// off, the bucket for the arrival width with it on.
    int           lanes      = 0;
    const bool    compact    = cmfdCompactEnabled();

    // ---- mixed-precision inner loop (RASBERY_GPU_CMFD_FP32) ----
    /// The env gate, resolved once in the constructor.
    bool          fp32_inner = false;
    /// Sticky safety fallback: set by drain() when an FP32 launch reported a
    /// non-finite, never cleared.
    bool          fp32_latched_off = false;
    /// Which kernel set the cached graphs were captured with.
    [[nodiscard]] int precisionTag() const { return fp32Active() ? 1 : 0; }
    int           graph_precision = -1;
    // (the sweep graph's precision now lives in SweepGraphCapacity::precision)
    /// Float mirrors of the operator.  Written ONLY by
    /// refresh_operator_mirror_f32; the double diag/cc stay authoritative.
    float*        diag_f = nullptr;
    float*        cc_f   = nullptr;
    /// Narrowed inverted diagonal blocks, written by begin_outer_fused_f32.
    float*        dinv_f = nullptr;
    /// The FP32 Krylov working set.  The flux, the source and `r` (the FP64
    /// residual that fixes the reference norm) are NOT here: they stay double.
    float *r_f = nullptr, *r0_f = nullptr, *p_f = nullptr, *v_f = nullptr;
    float *s_f = nullptr, *t_f = nullptr, *y_f = nullptr, *z_f = nullptr;

    // ---- device-resident CMFD sweep state (RASBERY_GPU_CMFD_SWEEP) ----
    double*        xs_chif    = nullptr; ///< [slot][ig*nxyz+l]
    double*        xs_xsnf    = nullptr; ///< [slot][ig*nxyz+l]
    double*        xs_xsrf    = nullptr; ///< [slot][ig*nxyz+l]
    double*        xs_xssm    = nullptr; ///< [slot][(igs*ng+ige)*nxyz+l]
    double*        dtil_dev   = nullptr; ///< [slot][surface*ng+ig]
    double*        dhat_dev   = nullptr; ///< [slot][surface*ng+ig]
    double*        node_vol   = nullptr; ///< [slot][l]
    double*        udiag_dev  = nullptr; ///< [slot][l*ng2+ige*ng+igs]
    double*        psi_dev    = nullptr; ///< [slot][l]
    std::uint32_t* sweep_halt = nullptr; ///< all-zero outside the sweep path
    std::uint32_t* device_assembly_active = nullptr;
    cudaGraphExec_t sweep_graph_exec = nullptr;
    /// The cudaGraph_t `sweep_graph_exec` was instantiated from, kept so the
    /// sweep can be SPLICED into a capturing stream instead of launched into
    /// one.  Owned by the matching sweep_graphs entry, not by this pointer.
    cudaGraph_t     sweep_graph_src  = nullptr;
    /// Replaces the old (nmax, unroll, precision) triple.  `unroll` is gone from
    /// the key entirely -- it lives in kSweepSlotBudget now -- and what is left
    /// is a capacity that only grows.  See SweepGraphCapacity in the header.
    SweepGraphCapacity sweep_graph{};

    long long node_stride() const { return static_cast<long long>(nxyz); }
};

} // namespace

// ===========================================================================
// Single-instance backend -- a one-slot BatchCore with an immediate launch.
// ===========================================================================

class CudaBICGBackend::Impl {
public:
    explicit Impl(Geometry& geometry) : core(geometry, 1) {}
    BatchCore core;
    int       zero = 0;
};

CudaBICGBackend::CudaBICGBackend(Geometry& geometry) : _impl(std::make_unique<Impl>(geometry)) {}
CudaBICGBackend::~CudaBICGBackend() = default;

bool CudaBICGBackend::available() const { return _impl->core.available; }
const std::string& CudaBICGBackend::status() const { return _impl->core.status; }
BackendCounters CudaBICGBackend::counters() const {
    return withPhiMirrorCounters(_impl->core.telemetry);
}
DeviceSolveStatus CudaBICGBackend::lastSolveStatus() const {
    return _impl->core.host_status != nullptr ? _impl->core.host_status[0] : DeviceSolveStatus{};
}

void CudaBICGBackend::reset(const double* host_diag,
                            const double* host_cc,
                            const double* host_phi,
                            const double* host_src) {
    if (!_impl->core.available) throw std::runtime_error(_impl->core.status);
    _impl->core.stageSlot(0, host_diag, host_cc, host_phi, host_src);
}

void CudaBICGBackend::solveInner(int nmax, double eps) {
    if (!_impl->core.available) throw std::runtime_error(_impl->core.status);
    auto& d = _impl->core;
    if (nmax < 0) nmax = 0;
    d.slot[0].eps  = eps;
    d.slot[0].nmax = nmax;
    d.issueUploads(&_impl->zero, 1);
    d.launch_outer(nmax);
}

void CudaBICGBackend::synchronize(double* host_phi) {
    if (!_impl->core.available) throw std::runtime_error(_impl->core.status);
    if (host_phi == nullptr)
        throw std::invalid_argument("CUDA BiCGSTAB synchronize requires a host flux buffer");
    auto& d = _impl->core;
    d.slot[0].out_phi = host_phi;
    d.issueFluxDownloads(&_impl->zero, 1);
    d.drain(&_impl->zero, 1);
}

// ===========================================================================
// Multi-instance arena.
// ===========================================================================

namespace {

/// Rev.7.1 Task 18: WHAT THE STREAM CLAIM COSTS, measured rather than argued.
///
/// The claim below is the one piece of the batch's CMFD path that is a HOST
/// serialisation, and the first regression it produced (238, M64: 224.5 c/h ->
/// under half) was diagnosed from source alone because nothing counted it.  It
/// is counted now: how long a thread waited to take it, and how long it held
/// it, split by the four sites that take it -- because the sites are not
/// interchangeable.  `enqueue` is bounded host work; `sync` drains a stream and
/// is therefore expected to hold for as long as the device takes; `rendezvous`
/// covers a whole batched launch.  A wait total that approaches the run's wall
/// says the claim is the bottleneck; a hold total dominated by `sync` says the
/// drain is.
struct ClaimStats {
    unsigned long long events  = 0;
    double             wait_us = 0.0;
    double             hold_us = 0.0;
};

/// A `lock_guard` that leaves a receipt.  Every field it touches is written
/// with the mutex held (`wait_us` after the acquire, the other two before the
/// release), so the counters need no atomicity of their own.
class TimedStreamClaim {
public:
    TimedStreamClaim(std::mutex& m, ClaimStats& stats) : _m(m), _stats(stats) {
        const auto before = std::chrono::steady_clock::now();
        _m.lock();
        _held = std::chrono::steady_clock::now();
        _stats.wait_us +=
            std::chrono::duration<double, std::micro>(_held - before).count();
    }
    ~TimedStreamClaim() {
        _stats.hold_us += std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - _held)
                              .count();
        ++_stats.events;
        _m.unlock();
    }
    TimedStreamClaim(const TimedStreamClaim&)            = delete;
    TimedStreamClaim& operator=(const TimedStreamClaim&) = delete;

private:
    std::mutex&                           _m;
    ClaimStats&                           _stats;
    std::chrono::steady_clock::time_point _held{};
};

} // namespace

class CudaBatchArena::Impl {
public:
    Impl(Geometry& geometry, int slots) : core(geometry, slots) {
        if (const char* wait_env = std::getenv("RASBERY_BATCH_WAIT_US")) {
            // Keep the historical numeric interface intact, and add an
            // explicit "auto" mode for batch rendezvous.  Auto never waits
            // longer than wait_max_us; callers can opt in without risking
            // the unbounded linger values used by old experiments.
            const std::string requested(wait_env);
            if (requested == "auto" || requested == "AUTO" || requested == "adaptive") {
                wait_auto = true;
            } else {
                const long parsed = std::atol(wait_env);
                if (parsed >= 0) wait_us = parsed;
            }
        }
        if (wait_auto) {
            if (const char* max_env = std::getenv("RASBERY_BATCH_WAIT_MAX_US")) {
                const long parsed = std::atol(max_env);
                if (parsed >= 0) wait_max_us = std::min(parsed, 2000L);
            }
        }
    }

    ~Impl() {
        // The arena outlives every deck, so this only runs at process teardown;
        // it is here so the pair is owned by the object that created it rather
        // than leaked on the strength of that.
        //
        // WP19 takes the window anyway rather than claiming an exemption: "the
        // arena outlives every deck" is a fact about THIS tree, and the arbiter
        // costs an uncontended shared lock once.  The immortal mutex in
        // GpuCaptureArbiter.h is what makes this safe during static
        // destruction.
        rasbery::AllocWindow _alloc_window("cmfd.arena.events.release");
        for (cudaEvent_t e : seg_ev_in)
            if (e != nullptr) cudaEventDestroy(e);
        for (cudaEvent_t e : seg_ev_out)
            if (e != nullptr) cudaEventDestroy(e);
        cudaGetLastError();
    }

    BatchCore core;

    // ---------------------------------------------------------------------
    // Rendezvous.
    //
    // Instances are not in lock step and must not be forced into it: they run
    // independent host code between CMFD solves, take different numbers of
    // outer iterations per burnup step, and hit their I/O at different times.
    // A batch therefore forms *opportunistically* -- it is simply everybody
    // who showed up while the previous batch was on the GPU.  That is sound
    // only because a slot's answer does not depend on which other slots rode
    // along; see the bit-identity rule in the header.
    //
    // The one thing this must not do is hold the arena lock across the launch:
    // that was the first version's mistake.  Threads waiting to *register*
    // would then be stuck behind the running batch, so instead of joining the
    // next one they arrived one at a time after it and each launched a batch
    // of one (measured mean width 2.1 of 8).  Registration is now lock-free
    // with respect to the GPU: `launching` is what serialises the device, and
    // `pending` fills up underneath it.
    //
    // `wait_us` optionally lingers for a fuller batch before launching. It
    // defaults to 0: when the GPU is not the bottleneck (the hybrid CMFD-only
    // configuration), lingering costs host time and buys nothing.  It earns
    // its keep once more of the step is device-resident.
    // ---------------------------------------------------------------------
    std::mutex              mutex;
    // ---------------------------------------------------------------------
    // Rev.7.1 Task 18: THE STREAM CLAIM
    // ---------------------------------------------------------------------
    //
    // `mutex` above guards the RENDEZVOUS -- the pending lists, the launcher
    // election, the occupancy counters -- and it is deliberately DROPPED across
    // the device work so the next batch can fill up underneath it.  That left
    // one thing unguarded: the arena's single stream and the state that is only
    // meaningful while somebody is enqueueing on it (the staging lanes, `lanes`,
    // the graph caches, the capture itself).  Under the rendezvous that was
    // safe by construction, because `launching` elects exactly one thread.
    //
    // The stream-ordered sweep has no election -- that is its whole point, and
    // it is why it refused batch mode.  A capture swallows everything a stream
    // receives while it is open, so worker A enqueueing its sweep while worker
    // B captured one killed four decks in 1.2 s with `operation failed due to a
    // previous error during capture`.
    //
    // So the two paths now share ONE claim on the stream, held only for the
    // ENQUEUE.  Two launches never interleave on the host; on the device they
    // are serialised by the stream itself, which is what makes the per-slot
    // device masks unnecessary.  Never taken while `mutex` is held (the
    // rendezvous drops its lock first), so the two cannot deadlock.
    std::mutex              stream_mutex;
    std::condition_variable cv;
    // Two rendezvous domains sharing one launcher election: kind 0 is the
    // plain per-sweep solve, kind 1 the device-resident multi-sweep launch
    // (RASBERY_GPU_CMFD_SWEEP).  A batch never mixes kinds -- they need
    // different graphs -- but both kinds' passengers sleep on the one cv and
    // either kind's thread may be elected to launch its own kind next.
    std::vector<int>        pending[2];      // registered, not yet launched
    unsigned long long      open_batch[2] = {0, 0};
    unsigned long long      completed[2]  = {0, 0};
    bool                    launching  = false;
    /// True only while a launcher sits in the linger wait below.  The
    /// per-arrival broadcast exists solely for that launcher; every other
    /// waiter blocks on completed/launching/open_batch, none of which an
    /// arrival changes.  Gating the notify on this flag turns 200k+ broadcasts
    /// per M64 run (each waking ~40 pinned threads into one mutex convoy)
    /// into zero when no linger budget is set.
    bool                    lingering  = false;
    long                    wait_us    = 0;
    bool                    wait_auto  = false;
    long                    wait_max_us = 2000;
    std::chrono::steady_clock::time_point last_arrival{};
    bool                    have_arrival = false;
    double                  arrival_gap_ewma_us = 0.0;
    long                    last_wait_budget_us = 0;
    unsigned long long      wait_events = 0;
    double                  idle_wait_us_total = 0.0;
    std::vector<char>       taken;
    /// Rev.7.1 Task 18: the cross-stream handover for a segment that does NOT
    /// share the arena stream.  One pair per slot, created on first use under
    /// `stream_mutex` and destroyed with the arena.
    ///
    /// A SINGLE deck (the resident-single case) still binds the arena stream
    /// itself and gets neither event: same stream, same order, and the enqueue
    /// path is byte-identical to what it was.  A BATCH cannot do that -- M
    /// segments on one stream would capture each other's kernels -- so each
    /// keeps its own stream and the two are joined here instead: `in` says the
    /// segment's updpsi has landed before the sweep reads psi, `out` says the
    /// sweep's flux and scalars have landed before the segment reads them.
    std::vector<cudaEvent_t> seg_ev_in, seg_ev_out;
    unsigned long long      launches       = 0;
    unsigned long long      batched_solves = 0;
    std::vector<unsigned long long> width_histogram;
    /// Rev.7.1 Task 18: the stream-ordered path's own width, which is ONE by
    /// construction -- `enqueueSweeps` stages a single slot.  Counted beside
    /// the rendezvous launches because the two are alternatives, and the
    /// question the M64 regression asked is precisely "which of the two ran,
    /// and how wide was it".  A batch whose sweeps all come through here has
    /// traded one launch of width M for M launches of width 1.
    unsigned long long      enqueue_launches = 0;
    ClaimStats              claim_enqueue, claim_finish, claim_sync, claim_rendezvous;
    /// Batch index whose launch threw, so its passengers report the same fatal
    /// condition the launcher did instead of carrying on with a stale flux.
    unsigned long long      failed_batch[2] = {~0ull, ~0ull};
    std::string             failed_message[2];

    [[nodiscard]] int inUseCount() const {
        int c = 0;
        for (char t : taken)
            if (t) ++c;
        return c;
    }
};

/// Rev.7.1 Task 20: the post-condition of a tenant reset, spelled out.
///
/// Every field a refilled slot must NOT inherit, listed once.  Three classes,
/// and each of them is a way one deck's numbers become another's:
///
///   - the borrowed host pointers.  A dead Driver's Geometry/XSSet are freed
///     memory; staging from them is a read of whatever the allocator put there.
///   - the residency and upload flags.  Left set, they ELIDE an upload the new
///     tenant needs, so the device keeps computing on the previous deck's
///     operator with no transfer and no error.
///   - the mirrors and the sweep scalars.  A valid mirror is a claim that the
///     device already holds these bytes, which is false for a new tenant.
[[nodiscard]] bool batchSlotIsReset(const BatchCore::Slot& sl) {
    const bool pointers_clear =
        sl.host_diag == nullptr && sl.host_cc == nullptr && sl.host_diag_out == nullptr &&
        sl.host_cc_out == nullptr && sl.host_phi == nullptr && sl.host_src == nullptr &&
        sl.out_phi == nullptr && sl.host_chif == nullptr && sl.host_xsnf == nullptr &&
        sl.host_xsrf == nullptr && sl.host_xssm == nullptr && sl.host_dtil == nullptr &&
        sl.host_dhat == nullptr && sl.host_vol == nullptr && sl.host_udiag == nullptr &&
        sl.host_psi == nullptr;
    const bool flags_default =
        sl.push_diag && sl.push_cc && sl.push_phi && sl.push_psi &&
        !sl.in_use && !sl.nonfinite && !sl.dhat_resident && !sl.psi_resident &&
        !sl.psi_downloaded && !sl.device_assembly && !sl.pushed_xsrf && !sl.pushed_xssm &&
        !sl.pushed_xsnf && !sl.pushed_dtil && sl.nmax == -1 && sl.sweep_unroll == 0;
    const bool mirrors_clear =
        !sl.diag_mirror.valid && !sl.cc_mirror.valid && !sl.phi_mirror.valid &&
        !sl.chif_mirror.valid && !sl.vol_mirror.valid && !sl.xsrf_mirror.valid() &&
        !sl.xssm_mirror.valid() && !sl.xsnf_mirror.valid() && !sl.dtil_mirror.valid();
    if (!pointers_clear || !flags_default || !mirrors_clear) return false;
    // Rev.7.1 Task 10 part 4 precondition (a).  The scalar block is a pinned
    // lane now, so "reset" is two facts and not one: the slot points at its
    // lane (Slot{} nulls it, bindSweepLanes puts it back) and the lane is
    // zero.  Order matters -- the null test is what stops the loop below from
    // reading through a pointer the reset just cleared.
    if (sl.sweep_in == nullptr || sl.sweep_out == nullptr) return false;
    for (int i = 0; i < kSweepCount; ++i)
        if (sl.sweep_in[i] != 0.0 || sl.sweep_out[i] != 0.0) return false;
    return true;
}

CudaBatchArena::CudaBatchArena(Geometry& geometry, int slots)
    : _impl(std::make_unique<Impl>(geometry, slots)) {
    _impl->taken.assign(static_cast<size_t>(std::max(slots, 1)), 0);
    _impl->width_histogram.assign(static_cast<size_t>(std::max(slots, 1)) + 1, 0);
}
CudaBatchArena::~CudaBatchArena() = default;

bool CudaBatchArena::available() const { return _impl->core.available; }
const std::string& CudaBatchArena::status() const { return _impl->core.status; }
int CudaBatchArena::slots() const { return _impl->core.slots; }
BackendCounters CudaBatchArena::counters() const {
    return withPhiMirrorCounters(_impl->core.telemetry);
}

bool CudaBatchArena::compatible(Geometry& geometry) const {
    return _impl->core.compatibleGeometry(geometry);
}

int CudaBatchArena::acquireSlot() {
    std::lock_guard<std::mutex> lock(_impl->mutex);
    for (int m = 0; m < _impl->core.slots; ++m) {
        if (_impl->taken[static_cast<size_t>(m)]) continue;
        _impl->taken[static_cast<size_t>(m)] = 1;
        // A fresh instance inherits nothing: the upload shadows of the previous
        // tenant would otherwise elide an upload the new tenant needs.
        //
        // Reset the WHOLE slot, not the four fields the plain solve happens to
        // read.  The partial reset predates the sweep path, which added
        // chif_mirror/vol_mirror and a second set of host_* pointers; those were
        // left carrying the previous tenant's state, and every host_* pointer a
        // dead Driver left behind dangles into freed memory.  `Slot{}` is exactly
        // the state slot.resize() gives a slot on the first acquire, so on the
        // validated one-worker-per-deck path (a single acquire per slot) this
        // assignment is a no-op -- same reset the NodalArena already does.
        BatchCore::Slot& sl = _impl->core.slot[static_cast<size_t>(m)];
        sl        = BatchCore::Slot{};
        // ...and the one field the whole-struct reset CANNOT restore, because
        // its correct value is not a constant: sweep_in/sweep_out point into
        // the arena's pinned block (Task 10 part 4 precondition (a)), and
        // Slot{} nulls them.  Re-bound here, before the audit below, which
        // tests for exactly this.
        _impl->core.bindSweepLanes(m);
        // Rev.7.1 Task 20 (plan Sec 3.2 "재활용 감사", Sec 8.2).  The reset above
        // is a whole-struct assignment, so it cannot MISS a field -- but it can
        // stop being one.  This checks the post-condition rather than trusting
        // the statement: if a future change makes any per-slot state survive an
        // admission, the next tenant computes its physics from the previous
        // deck's residency flags and mirrors, every value finite and plausible,
        // and nothing else in the process would ever say so.  Counted rather
        // than thrown: the state IS correct by the time this runs (the reset
        // just happened), so aborting a batch here would trade a receipt for a
        // lost run.  The gate is `stale_tenants: 0`.
        if (!batchSlotIsReset(sl))
            rasbery::refill::tenancy().stale_tenants.fetch_add(1, std::memory_order_relaxed);
        sl.in_use = true;
        rasbery::refill::tenancy().admissions.fetch_add(1, std::memory_order_relaxed);
        return m;
    }
    return -1;
}

void CudaBatchArena::releaseSlot(int m) {
    if (m < 0) return;
    std::lock_guard<std::mutex> lock(_impl->mutex);
    // Rev.7.1 Task 20.  A release of a slot nobody holds means a Driver
    // lifetime is not what the arena thinks it is, and the very next acquire
    // could hand the same slot to two tenants -- the same-case-concurrency
    // error of plan Sec 9.3, arriving through the host door.
    if (_impl->taken[static_cast<size_t>(m)] == 0)
        rasbery::refill::tenancy().double_releases.fetch_add(1, std::memory_order_relaxed);
    _impl->taken[static_cast<size_t>(m)]            = 0;
    _impl->core.slot[static_cast<size_t>(m)].in_use = false;
    // A lingering launcher may be waiting for this slot to show up; it never
    // will, and inUseCount() has just dropped, so wake it.
    _impl->cv.notify_all();
}

void CudaBatchArena::stage(int m, const double* diag, const double* cc,
                           const double* phi, const double* src) {
    if (!_impl->core.available) throw std::runtime_error(_impl->core.status);
    _impl->core.stageSlot(m, diag, cc, phi, src);
}

void CudaBatchArena::setInner(int m, int nmax, double eps) {
    if (nmax < 0) nmax = 0;
    BatchCore::Slot& sl = _impl->core.slot[static_cast<size_t>(m)];
    sl.nmax = nmax;
    sl.eps  = eps;
}

void CudaBatchArena::solve(int m, double* out_phi) { solveCommon(m, out_phi, 0); }

namespace {

/// Same two hooks CudaXsReconBackend.cu installs; both TUs install because
/// either can be the first to reach a pinHost call, and the install is
/// idempotent.  An anonymous namespace in each TU keeps them ODR-private.
int cudaHostPinRegister(void* address, std::size_t bytes) {
    // Rev.7.1 Task 18d: THE FIRST-TOUCH PIN IS THE RACE'S OTHER HALF.
    // BICGCMFD.cpp's `pinHost` block runs once per deck on that deck's own
    // Driver thread, which in a batch is while the earlier decks are capturing.
    rasbery::AllocWindow window("pin.register");
    const cudaError_t rc = cudaHostRegister(address, bytes, cudaHostRegisterDefault);
    if (rc != cudaSuccess) {
        // Named, not just swallowed.  Under the diagnostic global capture mode
        // this is the line that says "the sibling's first-touch pin is what was
        // running inside the capture", which is how Task 18d's root cause was
        // identified rather than guessed.
        rasbery::captureTrace("register-refused", cudaGetErrorString(rc), address, 0);
        cudaGetLastError(); // already registered / exotic host
    }
    return static_cast<int>(rc);
}

int cudaHostPinUnregister(void* address) {
    rasbery::AllocWindow window("pin.unregister");
    const cudaError_t rc = cudaHostUnregister(address);
    if (rc != cudaSuccess) cudaGetLastError();
    return static_cast<int>(rc);
}

void installHostPinHooks() {
    static const bool installed = [] {
        rasberyInstallHostPinHooks(&cudaHostPinRegister, &cudaHostPinUnregister);
        return true;
    }();
    (void)installed;
}

} // namespace

bool CudaBatchArena::pinHost(const void* p, size_t bytes, const char* tag) const {
    // Leased registration; the buffer's owner releases it in its destructor.
    // See HostPinRegistry.h and XsReconBackend::pinHost.
    installHostPinHooks();
    return rasberyPinHost(p, bytes, tag);
}

CudaBatchArena::CmfdResidentView CudaBatchArena::residentView(int m) const {
    CmfdResidentView v;
    const auto& c = _impl->core;
    if (!c.available || m < 0 || m >= c.slots) return v;
    if (c.phi == nullptr || c.psi_dev == nullptr || c.dtil_dev == nullptr ||
        c.dhat_dev == nullptr || c.xs_xsnf == nullptr)
        return v;
    v.phi   = c.phi + m * c.vec_stride();
    v.psi   = c.psi_dev + m * c.node_stride();
    v.dtil  = c.dtil_dev + m * c.surface_stride();
    v.dhat  = c.dhat_dev + m * c.surface_stride();
    v.xsnf  = c.xs_xsnf + m * c.vec_stride();
    v.nxyz  = c.nxyz;
    v.ngxyz = c.n;
    v.nsurf = static_cast<int>(c.surface_group_count /
                               static_cast<size_t>(c.n / c.nxyz));
    v.valid = true;
    return v;
}

void CudaBatchArena::stageSweeps(int m, const CmfdSweepIO& io) {
    auto& sl      = _impl->core.slot[static_cast<size_t>(m)];
    sl.host_chif  = io.chif;
    sl.host_xsnf  = io.xsnf;
    sl.host_xsrf  = io.xsrf;
    sl.host_xssm  = io.xssm;
    sl.host_dtil  = io.dtil;
    sl.host_dhat  = io.dhat;
    sl.host_vol   = io.vol;
    sl.host_udiag = io.udiag;
    sl.host_psi   = io.psi;
    sl.push_psi   = io.psi_dirty;
    sl.dhat_resident = io.dhat_device_resident;
    sl.psi_resident  = io.psi_device_resident;
    sl.device_assembly = io.device_assembly && cmfdAssemblyEnabled();
    if (sl.device_assembly &&
        (sl.host_xsrf == nullptr || sl.host_xssm == nullptr ||
         sl.host_dtil == nullptr || sl.host_dhat == nullptr))
        throw std::invalid_argument(
            "CMFD device assembly requires xsrf/xssm/dtil/dhat inputs");
    double* in    = sl.sweep_in;
    in[kEigv - kSweepFirst]        = io.eigv;
    in[kReigv - kSweepFirst]       = io.reigv;
    in[kReigvs - kSweepFirst]      = io.reigvs;
    in[kErrl2 - kSweepFirst]       = io.errl2;
    in[kEpsl2 - kSweepFirst]       = io.epsl2;
    in[kEshift - kSweepFirst]      = io.eshift;
    in[kReigvdel - kSweepFirst]    = 0.0;
    in[kSweepBudget - kSweepFirst] = static_cast<double>(io.sweep_budget);
    in[kSweepsDone - kSweepFirst]  = 0.0;
    in[kIcmfdBudget - kSweepFirst] = static_cast<double>(io.icmfd_budget);
    in[kIcmfdDone - kSweepFirst]   = static_cast<double>(io.icmfd_done);
    in[kNegative - kSweepFirst]    = 0.0;
    in[kSweepState - kSweepFirst]  = 0.0;
    in[kGammaD - kSweepFirst]      = 0.0;
    in[kGammaN - kSweepFirst]      = 0.0;
    in[kErrAcc - kSweepFirst]      = 0.0;
    in[kNgxyz - kSweepFirst]       = static_cast<double>(io.ngxyz);
    // The slot budget is a LAUNCH property (the batch-wide max), so it is
    // stamped by issueSweepUploads once the participant set is known.  Seeded
    // here only so a staged-but-never-launched block is not left holding a
    // stale budget from the previous drive.
    in[kSweepSlotBudget - kSweepFirst] = static_cast<double>(io.sweep_budget);
    in[kSweepSlots - kSweepFirst]      = 0.0;
    sl.sweep_unroll                    = io.sweep_budget;
}

void* CudaBatchArena::sweepStream() const {
    return _impl->core.available ? static_cast<void*>(_impl->core.stream) : nullptr;
}

bool CudaBatchArena::enqueueSweeps(int m, double* out_phi, const CmfdSweepIO& io,
                                   const CmfdSweepProbeSink& probe, void* caller_stream) {
    Impl& a = *_impl;
    if (!a.core.available || out_phi == nullptr) return false;
    if (m < 0 || m >= a.core.slots) return false;

    // ---- Rev.7.1 Task 18: ONE PARTICIPANT, AND NOW ANY NUMBER OF THEM ------
    //
    // WHAT THIS USED TO REFUSE AND WHY.  `if (a.inUseCount() > 1) return false;`
    // -- because this path takes no lock and the arena has ONE stream, so a
    // second in-flight instance would be a second launcher on it, which is
    // exactly the failure the rendezvous `launching` claim exists to prevent.
    // The device outer segment therefore ran in batch mode with the BLOCKING
    // sweep hook, which forces `sweep_synchronizes` and a segment budget of one.
    //
    // WHAT REPLACED IT.  `stream_mutex` is the claim this path was missing.  It
    // is held for the ENQUEUE only -- no drain, no rendezvous, no linger -- so
    // two decks' sweeps never interleave on the host, and on the device they are
    // ordered by the stream they share.  Everything a launch stages into is
    // either per-slot already (the physics arrays, the scalar block) or now has
    // one staging lane per slot (the fleet masks and the slot map), so the
    // second launcher's host writes cannot reach the first one's in-flight DMA.
    //
    // THE STAGE IS INSIDE THE CLAIM, and it has to be: stageSweeps writes
    // slot m's own scalar block, but issueSweepUploads reads `stage_lane`,
    // `lanes` and the graph caches, all of which the claim guards.
    TimedStreamClaim stream_claim(a.stream_mutex, a.claim_enqueue);

    // ---- join the caller's stream to the arena's ---------------------------
    //
    // Skipped entirely when they are the same stream, which is the single-deck
    // arrangement Task 10 chose and this leaves untouched.
    cudaStream_t caller = static_cast<cudaStream_t>(caller_stream);
    const bool   join   = caller != nullptr && caller != a.core.stream;
    if (join) {
        if (a.seg_ev_in.empty()) {
            a.seg_ev_in.assign(static_cast<size_t>(a.core.slots), nullptr);
            a.seg_ev_out.assign(static_cast<size_t>(a.core.slots), nullptr);
        }
        cudaEvent_t& ein  = a.seg_ev_in[static_cast<size_t>(m)];
        cudaEvent_t& eout = a.seg_ev_out[static_cast<size_t>(m)];
        // WP19.  Created lazily, per slot, on the slot's FIRST segment join --
        // which is a lane's stand-up by another name, and was the last
        // unguarded pair on this path.
        if (ein == nullptr || eout == nullptr) {
            rasbery::AllocWindow _alloc_window("cmfd.segment.events");
            if (ein == nullptr &&
                cudaEventCreateWithFlags(&ein, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            if (eout == nullptr &&
                cudaEventCreateWithFlags(&eout, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
        }
        // RECORDED BEFORE THE WAIT IS ENQUEUED, which is what makes the join
        // acyclic: the event describes work the caller has ALREADY submitted, so
        // the arena stream's wait can only ever be satisfied by segment work
        // that itself depends on strictly earlier arena-stream entries.
        if (cudaEventRecord(ein, caller) != cudaSuccess ||
            cudaStreamWaitEvent(a.core.stream, ein, 0) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }

    a.core.stage_lane = m; // this slot's own staging lane
    stageSweeps(m, io);
    auto& sl   = a.core.slot[static_cast<size_t>(m)];
    sl.out_phi = out_phi;

    const int nmax   = sl.nmax;
    const int unroll = sl.sweep_unroll;
    a.core.issueSweepUploads(&m, 1, unroll);
    // Rev.7.1 Task 10 part 3: the patch goes AFTER the gate, so the two
    // orderings this launch depends on are both `between the upload and the
    // graph`, and the patch reads the kEshift the upload has just landed.  WP7
    // stage B bit 3 may serve both from one launch; the order is the same.
    a.core.enqueueSweepPreamble(m, probe.halt, probe.halt_slot, probe);
    a.core.launch_sweeps(nmax, unroll);
    ++a.enqueue_launches; // one slot, therefore width one -- see the field's note
    // The verdict BEFORE the downloads: it reads the scalar block the graph just
    // wrote, and issueSweepDownloads ends by clearing sweep_halt for the next
    // launch, which would erase the very state the verdict is reading if the
    // order were reversed.
    a.core.enqueueSweepVerdict(m, probe);
    a.core.issueFluxDownloads(&m, 1);
    a.core.issueSweepDownloads(&m, 1);
    if (join &&
        (cudaEventRecord(a.seg_ev_out[static_cast<size_t>(m)], a.core.stream) != cudaSuccess ||
         cudaStreamWaitEvent(caller, a.seg_ev_out[static_cast<size_t>(m)], 0) != cudaSuccess)) {
        // The launch is already in flight and its results are correct; only the
        // ORDERING against the caller's stream failed.  Refusing here would let
        // the caller run a blocking drive over buffers this launch is writing,
        // so the honest recovery is to make the ordering by hand.
        cudaGetLastError();
        if (rasbery::xfer::streamSync("CudaBICGBackend.cu:enqueueSweep", "event fallback",
                               a.core.stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

bool CudaBatchArena::finishSweeps(int m, CmfdSweepIO& io) {
    Impl& a = *_impl;
    // THE OBSERVATION TOUCHES ARENA-WIDE STATE, so it takes the same claim the
    // enqueue does: absorb() folds into the shared telemetry and can latch the
    // FP32 fallback (which destroys the graph caches another deck may be about
    // to launch from), and the exceptional download below enqueues on the arena
    // stream and synchronises it.  No device wait is held across the claim on
    // the normal path -- the caller has already synchronised its own stream,
    // which the enqueue joined to the arena's.
    TimedStreamClaim stream_claim(a.stream_mutex, a.claim_finish);
    a.core.absorb(&m, 1);
    readSweepObservation(m, io);
    // The Rayleigh hand-back is the one launch whose host branch reads psi and
    // the assembled operator, so it is the one launch that pulls them.  This
    // call synchronises internally -- exceptionally, and only on the path that
    // is about to run host arithmetic anyway.
    if (io.state == 2) a.core.issueExceptionalOperatorDownloads(&m, 1);
    if (a.core.slot[static_cast<size_t>(m)].nonfinite) return false;
    a.core.adoptFluxMirror(m);
    return true;
}

bool CudaBatchArena::finishSweepsDeferred(int m, int state) {
    Impl& a = *_impl;
    // THE SAME CLAIM finishSweeps TAKES, for the same reasons: absorb() folds
    // into shared telemetry and can latch the FP32 fallback, and the
    // exceptional download enqueues on the arena stream and drains it.
    TimedStreamClaim stream_claim(a.stream_mutex, a.claim_finish);
    // ONE ABSORB FOR A WHOLE SEGMENT, and what that costs is TELEMETRY, not
    // correctness.  `host_status` is refreshed by the D2H at the end of every
    // launch, so the flag this reads -- NONFINITE_DETECTED -- is the LAST
    // launch's and is the one that matters; the four cumulative tallies beside
    // it are folded once instead of once per outer, which under-reports them on
    // this arm.  The receipt says so (hostfree_segments), and nothing numerical
    // reads them.
    a.core.absorb(&m, 1);
    // NO readSweepObservation.  The staging block belongs to whichever launch
    // was enqueued last -- on a segment that halted, one whose every kernel was
    // masked -- so reading it would hand the caller a drive that never ran.
    // Everything the caller needs comes from the accumulator instead.
    if (state == 2) a.core.issueExceptionalOperatorDownloads(&m, 1, /*forced=*/true);
    if (a.core.slot[static_cast<size_t>(m)].nonfinite) return false;
    a.core.adoptFluxMirror(m);
    return true;
}

void CudaBatchArena::syncSweepStream() {
    if (!_impl->core.available) return;
    // Rev.7.1 Task 18: UNDER THE STREAM CLAIM, and this is not bookkeeping.
    //
    // A stream that another thread has open for capture may not be
    // synchronised at all -- `operation not permitted when stream is capturing`
    // -- and once that error is raised the capture is poisoned, so the next
    // three decks die on `operation failed due to a previous error during
    // capture` from wherever they happened to be.  Measured on the 4-deck local
    // batch, four decks dead in 0.97 s, with the drain named as the first
    // failure and the other three as its fallout.
    //
    // This drain is a HOST-side wait, so holding the claim across it does stall
    // the other decks' enqueues -- for exactly as long as the arena stream was
    // going to take anyway, on the path where the caller is about to run a
    // blocking host CMFD drive.
    TimedStreamClaim stream_claim(_impl->stream_mutex, _impl->claim_sync);
    CUDA_CHECK(rasbery::xfer::streamSync("CudaBICGBackend.cu:syncSweepStream", "segment",
                                  _impl->core.stream));
    CUDA_CHECK(cudaGetLastError());
}

void CudaBatchArena::unpackSavedSweepBlock(const CmfdSweepProbeSink::Accum& acc,
                                          CmfdSweepIO& io) {
    const double* out = acc.saved;
    io.eigv          = out[kEigv - kSweepFirst];
    io.reigv         = out[kReigv - kSweepFirst];
    io.reigvs        = out[kReigvs - kSweepFirst];
    io.errl2         = out[kErrl2 - kSweepFirst];
    io.sweeps_done   = static_cast<int>(out[kSweepsDone - kSweepFirst]);
    io.icmfd_done    = static_cast<int>(out[kIcmfdDone - kSweepFirst]);
    io.state         = static_cast<int>(out[kSweepState - kSweepFirst]);
    io.negative_last = static_cast<int>(out[kNegative - kSweepFirst]);
    io.gammad        = out[kGammaD - kSweepFirst];
    io.gamman        = out[kGammaN - kSweepFirst];
    io.err_acc       = out[kErrAcc - kSweepFirst];
}

void CudaBatchArena::readSweepObservation(int m, CmfdSweepIO& io) const {
    const auto&   sl  = _impl->core.slot[static_cast<size_t>(m)];
    const double* out = sl.sweep_out;
    io.eigv          = out[kEigv - kSweepFirst];
    io.reigv         = out[kReigv - kSweepFirst];
    io.reigvs        = out[kReigvs - kSweepFirst];
    io.errl2         = out[kErrl2 - kSweepFirst];
    io.sweeps_done   = static_cast<int>(out[kSweepsDone - kSweepFirst]);
    io.icmfd_done    = static_cast<int>(out[kIcmfdDone - kSweepFirst]);
    io.state         = static_cast<int>(out[kSweepState - kSweepFirst]);
    io.negative_last = static_cast<int>(out[kNegative - kSweepFirst]);
    io.gammad        = out[kGammaD - kSweepFirst];
    io.gamman        = out[kGammaN - kSweepFirst];
    io.err_acc       = out[kErrAcc - kSweepFirst];
}

void CudaBatchArena::solveSweeps(int m, double* out_phi, CmfdSweepIO& io) {
    solveCommon(m, out_phi, 1);
    const auto&   sl  = _impl->core.slot[static_cast<size_t>(m)];
    const double* out = sl.sweep_out;
    io.eigv          = out[kEigv - kSweepFirst];
    io.reigv         = out[kReigv - kSweepFirst];
    io.reigvs        = out[kReigvs - kSweepFirst];
    io.errl2         = out[kErrl2 - kSweepFirst];
    io.sweeps_done   = static_cast<int>(out[kSweepsDone - kSweepFirst]);
    io.icmfd_done    = static_cast<int>(out[kIcmfdDone - kSweepFirst]);
    io.state         = static_cast<int>(out[kSweepState - kSweepFirst]);
    io.negative_last = static_cast<int>(out[kNegative - kSweepFirst]);
    io.gammad        = out[kGammaD - kSweepFirst];
    io.gamman        = out[kGammaN - kSweepFirst];
    io.err_acc       = out[kErrAcc - kSweepFirst];
}

void CudaBatchArena::solveCommon(int m, double* out_phi, int kind) {
    if (!_impl->core.available) throw std::runtime_error(_impl->core.status);
    if (out_phi == nullptr)
        throw std::invalid_argument("CUDA BiCGSTAB solve requires a host flux buffer");
    Impl& a = *_impl;
    a.core.slot[static_cast<size_t>(m)].out_phi = out_phi;

    std::unique_lock<std::mutex> lock(a.mutex);
    // Record inter-arrival gaps while the rendezvous lock is held.  The EWMA
    // is deliberately process-local and cheap: it estimates how long the
    // next sibling usually takes to reach this same CMFD solve without
    // coupling physics progress between instances.
    const auto arrival_now = std::chrono::steady_clock::now();
    if (a.have_arrival) {
        const double gap_us = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(arrival_now - a.last_arrival).count());
        if (gap_us >= 0.0 && gap_us <= 1.0e6) {
            a.arrival_gap_ewma_us = a.arrival_gap_ewma_us == 0.0
                                        ? gap_us
                                        : 0.8 * a.arrival_gap_ewma_us + 0.2 * gap_us;
        }
    }
    a.last_arrival  = arrival_now;
    a.have_arrival  = true;
    const unsigned long long my_batch = a.open_batch[kind];
    // Rev.7.1 Task 20 (plan Sec 5.2 ownership rule, host arm).  A physical slot
    // that is already queued for THIS batch must never be inserted again: the
    // launcher stages every participant in turn, so a second entry would stage
    // the same slot twice and the second stage would overwrite the operator the
    // first one uploaded.  It cannot happen while one Driver owns one slot,
    // which is exactly why it is worth counting -- this is the witness that the
    // one-owner rule still holds after a refill.  The scan is over at most
    // `slots` (<= 64) ints under a lock we already hold.
    if (std::find(a.pending[kind].begin(), a.pending[kind].end(), m) !=
        a.pending[kind].end()) {
        rasbery::refill::tenancy().queue_duplicates.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error(
            "CUDA batch arena: slot " + std::to_string(m) +
            " arrived twice in one rendezvous batch (two tenants share a slot)");
    }
    a.pending[kind].push_back(m);
    if (a.lingering)
        a.cv.notify_all();   // a lingering launcher is waiting for arrivals

    while (true) {
        if (a.completed[kind] > my_batch) {
            // Somebody else ran our batch; out_phi already holds the answer.
            if (a.failed_batch[kind] == my_batch)
                throw std::runtime_error(a.failed_message[kind]);
            lock.unlock();
            if (a.core.slot[static_cast<size_t>(m)].nonfinite)
                throw std::runtime_error("CUDA BiCGSTAB detected a non-finite value");
            a.core.adoptFluxMirror(m); // own slot, own thread -- no lock needed
            return;
        }
        if (a.launching || a.open_batch[kind] != my_batch) {
            a.cv.wait(lock);
            continue;
        }

        // ---- nobody is on the device: we launch this batch -----------------
        //
        // Claim it *before* doing anything that releases the lock.  The linger
        // below waits on the condition variable, which unlocks; without this
        // flag set first, a thread arriving during the linger would find
        // `launching == false` and `open_batch == my_batch` and elect itself a
        // second launcher of the same batch.  Two launchers then drove the same
        // stream at once -- NaN flux, invalidated graph captures and heap
        // corruption, all of it invisible until batches got wide enough for the
        // linger to matter.
        a.launching = true;

        // Cost of the linger, stated plainly: `launching` is already set, so the
        // GPU sits idle while missing participants catch up.  Auto mode uses a
        // bounded 2x arrival-gap estimate (100 us bootstrap) and therefore
        // adapts to the actual host skew instead of imposing a fixed 100 ms
        // penalty on every batch.
        long linger_us = a.wait_us;
        if (a.wait_auto) {
            if (a.wait_max_us <= 0) {
                linger_us = 0;
            } else {
                const double estimate = a.arrival_gap_ewma_us > 0.0 ? 2.0 * a.arrival_gap_ewma_us : 100.0;
                linger_us = static_cast<long>(std::clamp(estimate, 25.0,
                                                          static_cast<double>(a.wait_max_us)));
            }
        }
        a.last_wait_budget_us = linger_us;
        const auto wait_start = std::chrono::steady_clock::now();
        if (linger_us > 0 && static_cast<int>(a.pending[kind].size()) < a.inUseCount()) {
            const auto deadline =
                wait_start + std::chrono::microseconds(linger_us);
            a.lingering = true;
            while (static_cast<int>(a.pending[kind].size()) < a.inUseCount() &&
                   a.cv.wait_until(lock, deadline) != std::cv_status::timeout) {
            }
            a.lingering = false;
        }
        if (linger_us > 0) {
            const double waited_us = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start)
                    .count()) /
                1.0e3;
            a.idle_wait_us_total += waited_us;
            ++a.wait_events;
        }

        std::vector<int> participants;
        participants.swap(a.pending[kind]);
        ++a.open_batch[kind];      // arrivals from here on join the next batch
        std::sort(participants.begin(), participants.end());

        int         nmax    = -1;
        int         unroll  = 0;
        bool        failed  = false;
        std::string message;
        for (int slot_id : participants) {
            const auto& sl        = a.core.slot[static_cast<size_t>(slot_id)];
            const int   slot_nmax = sl.nmax;
            if (nmax < 0) {
                nmax = slot_nmax;
            } else if (nmax != slot_nmax) {
                // Reported, not thrown from here: the batch is already claimed
                // and its passengers must be released whatever happens.
                failed  = true;
                message = "batch mode requires a uniform inner BiCGSTAB budget across "
                          "instances (RASBERY_BICG_NMAX must not differ between decks)";
            }
            if (kind == 1) unroll = std::max(unroll, sl.sweep_unroll);
        }

        // The device work runs *unlocked* on purpose: this is the window in
        // which the next batch fills up.
        lock.unlock();
        if (!failed && !participants.empty()) {
            // The rendezvous claim (`launching`) keeps other RENDEZVOUS
            // launchers out; the stream claim keeps the stream-ordered enqueue
            // path out, which has no election of its own.
            TimedStreamClaim stream_claim(a.stream_mutex, a.claim_rendezvous);
            a.core.stage_lane = a.core.slots; // the rendezvous lane
            try {
                if (kind == 0) {
                    a.core.issueUploads(participants.data(),
                                        static_cast<int>(participants.size()));
                    a.core.launch_outer(nmax);
                    a.core.issueFluxDownloads(participants.data(),
                                              static_cast<int>(participants.size()));
                    a.core.drain(participants.data(),
                                 static_cast<int>(participants.size()));
                } else {
                    a.core.issueSweepUploads(participants.data(),
                                             static_cast<int>(participants.size()), unroll);
                    a.core.launch_sweeps(nmax, unroll);
                    a.core.issueFluxDownloads(participants.data(),
                                              static_cast<int>(participants.size()));
                    a.core.issueSweepDownloads(participants.data(),
                                               static_cast<int>(participants.size()));
                    a.core.drain(participants.data(),
                                 static_cast<int>(participants.size()));
                    a.core.issueExceptionalOperatorDownloads(
                        participants.data(), static_cast<int>(participants.size()));
                }
            } catch (const std::exception& error) {
                failed  = true;
                message = error.what();
            }
        }
        lock.lock();
        a.launching = false;
        a.completed[kind] = my_batch + 1;
        if (failed) {
            a.failed_batch[kind]   = my_batch;
            a.failed_message[kind] = message;
        } else if (!participants.empty()) {
            // Counted here, not before the launch.  These three feed
            // BATCH_OCCUPANCY, whose whole job is to answer "how wide are the
            // batches that actually ran?"  Incrementing them ahead of the gate
            // folded in the batches that never reached the device -- the
            // non-uniform-nmax abort and the empty participant list -- and an
            // aborted run therefore reported a mean width it never achieved,
            // which is exactly the number an operator uses to decide whether
            // RASBERY_BATCH_WAIT_US is worth setting.
            ++a.launches;
            a.batched_solves += participants.size();
            a.width_histogram[participants.size()] += 1;
        }
        lock.unlock();
        a.cv.notify_all();

        if (failed) throw std::runtime_error(message);
        if (a.core.slot[static_cast<size_t>(m)].nonfinite)
            throw std::runtime_error("CUDA BiCGSTAB detected a non-finite value");
        a.core.adoptFluxMirror(m); // the launcher is a participant too
        return;
    }
}

void CudaBatchArena::reportBatchOccupancy(const char* tag) const {
    const Impl& a = *_impl;
    // The stream-ordered path is counted too, and that is why the gate is an
    // OR: a batch that took it exclusively has zero rendezvous launches, and
    // returning early there is exactly how the M64 regression hid -- the one
    // receipt that answers "how wide were the sweeps" printed nothing at all.
    if (a.launches == 0 && a.enqueue_launches == 0) return;
    const double all_launches =
        static_cast<double>(a.launches) + static_cast<double>(a.enqueue_launches);
    std::ostringstream line;
    line << "[RASBERY][CUDA][BATCH_OCCUPANCY] {\"tag\":\"" << (tag ? tag : "") << "\","
         << "\"slots\":" << a.core.slots << ','
         << "\"launches\":" << a.launches << ','
         << "\"instance_solves\":" << a.batched_solves << ','
         << "\"mean_width\":"
         << (a.launches ? static_cast<double>(a.batched_solves) / static_cast<double>(a.launches)
                        : 0.0)
         << ','
         // The stream-ordered sweep enqueue: one slot per launch, always.
         << "\"enqueue_launches\":" << a.enqueue_launches << ','
         << "\"effective_mean_width\":"
         << (all_launches > 0.0
                 ? (static_cast<double>(a.batched_solves) +
                    static_cast<double>(a.enqueue_launches)) /
                       all_launches
                 : 0.0)
         << ','
         << "\"claim_enqueue\":{\"n\":" << a.claim_enqueue.events
         << ",\"wait_ms\":" << (a.claim_enqueue.wait_us / 1.0e3)
         << ",\"hold_ms\":" << (a.claim_enqueue.hold_us / 1.0e3) << "},"
         << "\"claim_finish\":{\"n\":" << a.claim_finish.events
         << ",\"wait_ms\":" << (a.claim_finish.wait_us / 1.0e3)
         << ",\"hold_ms\":" << (a.claim_finish.hold_us / 1.0e3) << "},"
         << "\"claim_sync\":{\"n\":" << a.claim_sync.events
         << ",\"wait_ms\":" << (a.claim_sync.wait_us / 1.0e3)
         << ",\"hold_ms\":" << (a.claim_sync.hold_us / 1.0e3) << "},"
         << "\"claim_rendezvous\":{\"n\":" << a.claim_rendezvous.events
         << ",\"wait_ms\":" << (a.claim_rendezvous.wait_us / 1.0e3)
         << ",\"hold_ms\":" << (a.claim_rendezvous.hold_us / 1.0e3) << "},"
         << "\"wait_us\":" << (a.wait_auto ? -1 : a.wait_us) << ','
         << "\"wait_mode\":\"" << (a.wait_auto ? "auto" : "fixed") << "\","
         << "\"wait_budget_us\":" << a.last_wait_budget_us << ','
         << "\"wait_mean_us\":"
         << (a.wait_events ? a.idle_wait_us_total / static_cast<double>(a.wait_events) : 0.0) << ','
         << "\"idle_wait_us_total\":" << a.idle_wait_us_total << ','
         << "\"arrival_gap_ewma_us\":" << a.arrival_gap_ewma_us << ','
         << "\"width_histogram\":[";
    for (size_t w = 1; w < a.width_histogram.size(); ++w) {
        if (w > 1) line << ',';
        line << a.width_histogram[w];
    }
    line << "]}";
    std::cout << line.str() << std::endl;
}

// ===========================================================================
// Process-wide batch plumbing.
// ===========================================================================

namespace {
std::mutex                      g_arena_mutex;
std::unique_ptr<CudaBatchArena> g_arena;
int                             g_batch_width = 0;
} // namespace

void rasberySetBatchWidth(int slots) { g_batch_width = slots > 0 ? slots : 0; }

int rasberyBatchWidth() { return g_batch_width; }

/// Rev.7.1 Task 6.  Read once, like every other RASBERY_* gate, so the flag
/// cannot change meaning halfway through a run.
bool rasberyResidentSingleCmfd() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_RESIDENT_SINGLE");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

CudaBatchArena* rasberyBatchArena(Geometry& geometry) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (!g_arena) {
        // Width 1 is the resident-single case (Task 6): no --batch-mode, one
        // physical slot, and the SAME BatchCore kernels.  There is deliberately
        // no separate single-instance kernel set -- a second path would be a
        // second thing to keep bit-identical, which is the cost the whole task
        // exists to avoid.
        const int width = g_batch_width > 0 ? g_batch_width : 1;
        g_arena = std::make_unique<CudaBatchArena>(geometry, width);
        if (!g_arena->available()) {
            const std::string message = g_arena->status();
            g_arena.reset();
            throw std::runtime_error("RASBERY batch mode requested but unavailable: " + message);
        }
        std::cout << "[RASBERY][CUDA][BATCH] arena: " << g_arena->status() << std::endl;
    }
    if (!g_arena->compatible(geometry))
        throw std::runtime_error(
            "batch mode requires every instance to share one geometry "
            "(a deck with a different CMFD shape was submitted)");
    return g_arena.get();
}

/// The active-slot compaction receipt.
///
/// Printed whether or not compaction is on, so two runs of the same deck are
/// directly comparable and `padding_fraction` states the size of the prize
/// before anyone pays for it.  Its own tag, so nothing that parses
/// [RASBERY][CUDA][BACKEND_COUNTERS] or [BATCH_OCCUPANCY] has to change.
static void reportCmfdCompaction() {
    const unsigned long long phys = g_cmfd_physical_blocks.load(std::memory_order_relaxed);
    const unsigned long long pad  = g_cmfd_padding_blocks.load(std::memory_order_relaxed);
    std::ostringstream line;
    line << "[RASBERY][CMFD][COMPACT] {\"enabled\":" << (cmfdCompactEnabled() ? 1 : 0)
         << ",\"logical_drives\":"
         << g_cmfd_logical_drives.load(std::memory_order_relaxed)
         << ",\"physical_slot_blocks\":" << phys
         << ",\"padding_blocks\":" << pad
         << ",\"padding_fraction\":"
         << ((phys + pad) ? static_cast<double>(pad) / static_cast<double>(phys + pad)
                          : 0.0)
         << ",\"bucket_graphs\":"
         << g_cmfd_bucket_graphs.load(std::memory_order_relaxed)
         << ",\"bucket_histogram\":[";
    for (std::size_t i = 0; i < g_cmfd_bucket_histogram.size(); ++i) {
        if (i) line << ',';
        line << g_cmfd_bucket_histogram[i].load(std::memory_order_relaxed);
    }
    line << "]}";
    std::cout << line.str() << std::endl;
}

void rasberyReleaseBatchArena() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    // Before the arena test, and gated on evidence rather than on batch mode:
    // the single-instance backend dispatches through the same buildSlotMap, so
    // a non-batch run has a padding fraction worth reading too (it is 0 there,
    // which is the point).  A run that never reached the device stays silent.
    if (g_cmfd_logical_drives.load(std::memory_order_relaxed) != 0)
        reportCmfdCompaction();
    if (!g_arena) return;
    g_arena->reportBatchOccupancy("run");
    // Printed beside the occupancy because it answers the question the
    // occupancy raises: the batch is this wide, and this is what it cost to
    // keep its captures away from its allocations.
    std::cout << rasbery::captureArbiterReceipt("run") << std::endl;
    BackendCounters c = g_arena->counters();
    // F9 (review doc Sec 3).  The arena spelling of this receipt never carried
    // the three host-fallback fields at all, and the non-arena spelling in
    // src/BICGSolver.cpp carried them as a hard 0.  A batch run is the arm the
    // campaign actually measures, so the fields have to be HERE or the fix is
    // invisible where it matters.  One source: the WP1 seam tally, not a second
    // set of counters that could disagree with it (F13).
    {
        namespace gf = ::rasbery::gpufull;
        c.cmfd_cpu_fallbacks  = gf::fallbacks(gf::Subsystem::Cmfd);
        c.nodal_cpu_fallbacks = gf::fallbacks(gf::Subsystem::Nodal);
        c.xs_cpu_fallbacks    = gf::fallbacks(gf::Subsystem::FlatXs) +
                                gf::fallbacks(gf::Subsystem::Xe) +
                                gf::fallbacks(gf::Subsystem::Cram);
    }
    std::cout << "[RASBERY][CUDA][BACKEND_COUNTERS] {"
              << "\"cmfd_gpu_calls\":" << c.cmfd_gpu_calls << ','
              << "\"cmfd_cpu_fallbacks\":" << c.cmfd_cpu_fallbacks << ','
              << "\"nodal_cpu_fallbacks\":" << c.nodal_cpu_fallbacks << ','
              << "\"xs_cpu_fallbacks\":" << c.xs_cpu_fallbacks << ','
              << "\"cmfd_assembly_gpu_calls\":" << c.cmfd_assembly_gpu_calls << ','
              << "\"cmfd_assembly_cpu_fallbacks\":"
              << c.cmfd_assembly_cpu_fallbacks << ','
              << "\"cmfd_diag_h2d_elided_bytes\":"
              << c.cmfd_diag_h2d_elided_bytes << ','
              << "\"cmfd_cc_h2d_elided_bytes\":"
              << c.cmfd_cc_h2d_elided_bytes << ','
              << "\"cmfd_psi_h2d_elided_bytes\":"
              << c.cmfd_psi_h2d_elided_bytes << ','
              << "\"cmfd_psi_d2h_elided_bytes\":"
              << c.cmfd_psi_d2h_elided_bytes << ','
              << "\"cmfd_dhat_h2d_elided_bytes\":"
              << c.cmfd_dhat_h2d_elided_bytes << ','
              << "\"cmfd_resident_psi_h2d_elided_bytes\":"
              << c.cmfd_resident_psi_h2d_elided_bytes << ','
              << "\"cmfd_phi_mirror_ns\":" << c.cmfd_phi_mirror_ns << ','
              << "\"cmfd_phi_mirror_calls\":" << c.cmfd_phi_mirror_calls << ','
              << "\"cmfd_phi_mirror_bypassed\":" << c.cmfd_phi_mirror_bypassed
              << ','
              << "\"cmfd_phi_h2d_elided_bytes\":"
              << c.cmfd_phi_h2d_elided_bytes << ','
              << "\"bicg_early_convergence_exits\":" << c.bicg_early_convergence_exits << ','
              << "\"bicg_restarts\":" << c.bicg_restarts << ','
              << "\"bulk_h2d_calls_during_iteration\":" << c.bulk_h2d_calls_during_iteration << ','
              << "\"bulk_h2d_skipped_during_iteration\":" << c.bulk_h2d_skipped_during_iteration
              << ','
              << "\"bulk_h2d_bytes_during_iteration\":" << c.bulk_h2d_bytes_during_iteration << ','
              << "\"bulk_d2h_calls_during_iteration\":" << c.bulk_d2h_calls_during_iteration << ','
              << "\"bulk_d2h_bytes_during_iteration\":" << c.bulk_d2h_bytes_during_iteration << ','
              << "\"stream_sync_calls_during_iteration\":" << c.stream_sync_calls_during_iteration
              << ','
              << "\"graph_launches\":" << c.graph_launches << ','
              << "\"graph_reinstantiations\":" << c.graph_reinstantiations << ','
              << "\"graph_fallbacks\":" << c.graph_fallbacks << ','
              << "\"iter_batch\":" << c.iter_batch << ','
              << "\"batched_graph_launches\":" << c.batched_graph_launches << ','
              << "\"overrun_iterations\":" << c.overrun_iterations << ','
              << "\"fp32_active\":" << c.fp32_active << ','
              << "\"fp32_fallbacks\":" << c.fp32_fallbacks << '}' << std::endl;
    g_arena.reset();
}

} // namespace rasbery
