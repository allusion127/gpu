#include "CudaBICGBackend.h"

#include "CmfdAssemblyKernel.h"
#include "CudaTransferMirror.h"
#include "CudaXsReconBackend.h" // rasberyHostPinningEnabled(): header-only gate
#include "Geometry.h"
#include "pch.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
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

/// RASBERY_GPU_CMFD_COMPACT, default OFF.  Read once, like every other gate.
inline bool cmfdCompactEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_COMPACT");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
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

    const double err    = lane_sum[0];
    const double gammad = lane_sum[1];
    const double gamman = lane_sum[2];
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
        double        sweep_in[kSweepCount]  = {};
        double        sweep_out[kSweepCount] = {};
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
            fp32_inner    = cmfdFp32InnerEnabled();
            telemetry.fp32_active = fp32_inner ? 1u : 0u;
            status += " (block=" + std::to_string(block_size) +
                      ", RB sweeps=" + std::to_string(rb_sweeps) +
                      ", graph=" + (use_graph ? "on" : "off") +
                      ", iter batch=" +
                      (iter_batch_request > 0 ? std::to_string(iter_batch_request)
                                              : std::string("auto")) +
                      ", assembly=" + (cmfdAssemblyEnabled() ? "on" : "off") +
                      ", scalar fusion=" + (scalar_fusion ? "on" : "off") +
                      // The [PHYSICS_MODE] receipt for the inner-solve precision.
                      // fp64 = the historical all-double path; mixed = FP32 inner
                      // BiCGSTAB under an FP64 outer correction.
                      ", precision=" + (fp32_inner ? "mixed" : "fp64") +
                      ", slots=" + std::to_string(slots) + ")";

            const size_t S = static_cast<size_t>(slots);
            allocate(reinterpret_cast<void**>(&neighbors), host_neighbors.size() * sizeof(int));
            CUDA_CHECK(cudaMemcpy(neighbors,
                                  host_neighbors.data(),
                                  host_neighbors.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&colors), host_colors.size() * sizeof(int));
            CUDA_CHECK(cudaMemcpy(colors,
                                  host_colors.data(),
                                  host_colors.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&assembly_node_surface),
                     host_node_surface.size() * sizeof(int));
            CUDA_CHECK(cudaMemcpy(assembly_node_surface, host_node_surface.data(),
                                  host_node_surface.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&assembly_face_area),
                     host_face_area.size() * sizeof(double));
            CUDA_CHECK(cudaMemcpy(assembly_face_area, host_face_area.data(),
                                  host_face_area.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
            allocate(reinterpret_cast<void**>(&assembly_volume),
                     host_geometry_volume.size() * sizeof(double));
            CUDA_CHECK(cudaMemcpy(assembly_volume, host_geometry_volume.data(),
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
            host_assembly_active.assign(S, 0u);
            CUDA_CHECK(cudaMemset(device_halt, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemset(device_active, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_status),
                                      S * sizeof(DeviceSolveStatus)));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_active),
                                      S * sizeof(std::uint32_t)));
            std::memset(host_active, 0, S * sizeof(std::uint32_t));
            // SNAPSHOT BUFFER for the sweep mask.  issueSweepUploads used to
            // build the participation mask in host_active, upload it, and then
            // INVERT THAT SAME BUFFER IN PLACE for the sweep_halt upload -- a
            // host write to the source of a cudaMemcpyAsync that had not been
            // synchronised.  host_active is cudaMallocHost'd, so that copy is a
            // real asynchronous DMA and the driver is entitled to read the
            // bytes after the inversion has already landed.  One buffer per
            // upload is the fix; see issueSweepUploads.
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_sweep_halt),
                                      S * sizeof(std::uint32_t)));
            std::memset(host_sweep_halt, 0, S * sizeof(std::uint32_t));

            // The lane -> slot map.  Allocated once at the FULL fleet width
            // whatever a launch's bucket turns out to be, so d_slot_map is a
            // fixed address a captured graph can bake, and seeded with the
            // identity so a launch that never calls buildSlotMap (a direct
            // enqueue, a test) behaves exactly as the pre-compaction code did.
            allocate(reinterpret_cast<void**>(&d_slot_map), S * sizeof(int));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_slot_map),
                                      S * sizeof(int)));
            for (std::size_t i = 0; i < S; ++i) h_slot_map[i] = static_cast<int>(i);
            CUDA_CHECK(cudaMemcpy(d_slot_map, h_slot_map, S * sizeof(int),
                                  cudaMemcpyHostToDevice));
            lanes = slots;

            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUBLAS_CHECK(cublasCreate(&handle));
            CUBLAS_CHECK(cublasSetStream(handle, stream));
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
            slot.resize(static_cast<size_t>(slots));
            available = true;
        } catch (const std::exception& error) {
            status = error.what();
            release();
        }
    }

    ~BatchCore() { release(); }

    void allocate(void** pointer, size_t bytes) { CUDA_CHECK(cudaMalloc(pointer, bytes)); }

    void release() {
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
        neighbors = nullptr;
        colors = nullptr;
        assembly_node_surface = nullptr;
        assembly_face_area = assembly_volume = nullptr;
        diag = dinv = cc = src = phi = r = r0 = p = v = s = t = y = z = ax = nullptr;
        partials = nullptr;
        partials2 = nullptr;
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
        if (!compact) {
            lanes = slots;
            for (int i = 0; i < slots; ++i) h_slot_map[i] = i;
        } else {
            lanes = cmfdBucketFor(count, slots);
            for (int i = 0; i < slots; ++i) h_slot_map[i] = -1;
            for (int i = 0; i < count && i < lanes; ++i) h_slot_map[i] = active_slots[i];
        }
        // Always the FULL fleet width, never `lanes`: a stale entry from a
        // wider previous launch must never be reachable by a later, deeper
        // graph replay.
        CUDA_CHECK(cudaMemcpyAsync(d_slot_map, h_slot_map,
                                   static_cast<size_t>(slots) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
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
        std::memset(host_active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        for (int i = 0; i < count; ++i) host_active[active_slots[i]] = 1u;
        CUDA_CHECK(cudaMemcpyAsync(device_active,
                                   host_active,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice,
                                   stream));

        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];

            // diag really does change every outer (the Wielandt shift rewrites
            // it), so it is not mirrored at all (see stageSlot); cc and phi are
            // the ones whose mirrors drop uploads.
            {
                const size_t bytes = matrix_count * sizeof(double);
                CUDA_CHECK(cudaMemcpyAsync(diag + m * mat_stride(), sl.host_diag,
                                           bytes, cudaMemcpyHostToDevice, stream));
                ++telemetry.bulk_h2d_calls_during_iteration;
                telemetry.bulk_h2d_bytes_during_iteration += bytes;
            }
            pushOrSkip(cc + m * cpl_stride(), sl.host_cc, coupling_count,
                       sl.push_cc, sl.cc_mirror);
            pushOrSkip(phi + m * vec_stride(), sl.host_phi, static_cast<size_t>(n),
                       sl.push_phi, sl.phi_mirror);

            // src is rebuilt from psi on the host at the top of every outer; it
            // is the one buffer that is genuinely new each time.
            CUDA_CHECK(cudaMemcpyAsync(src + m * vec_stride(),
                                       sl.host_src,
                                       static_cast<size_t>(n) * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
            ++telemetry.bulk_h2d_calls_during_iteration;
            telemetry.bulk_h2d_bytes_during_iteration +=
                static_cast<size_t>(n) * sizeof(double);

            // The exit tolerance is a kernel *input*, not a kernel argument:
            // keeping it in device memory is what lets one captured graph serve
            // every outer of every instance.
            if (!(sl.eps == sl.eps_on_device)) {
                CUDA_CHECK(cudaMemcpyAsync(scalars + static_cast<long long>(m) * kScalarCount + kEps,
                                           &sl.eps,
                                           sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           stream));
                // Graph/direct kernels are submitted to this same stream, so
                // stream order publishes eps without draining the pipeline.
                sl.eps_on_device = sl.eps;
            }

            // update_solution advances the device flux, so the host mirror no
            // longer describes device memory.  fetchFlux re-establishes it.
            sl.phi_mirror.valid = false;
        }
    }

    void pushOrSkip(double* device_buffer, const double* host_buffer, size_t count,
                    bool push, MirroredUpload& mirror) {
        if (!push) {
            ++telemetry.bulk_h2d_skipped_during_iteration;
            return;
        }
        const size_t bytes = count * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(
            device_buffer, host_buffer, bytes, cudaMemcpyHostToDevice, stream));
        mirror.shadow.assign(host_buffer, host_buffer + count);
        mirror.valid = true;
        ++telemetry.bulk_h2d_calls_during_iteration;
        telemetry.bulk_h2d_bytes_during_iteration += bytes;
    }

    /// Bit-reproducible replacement for cublasDdot / cublasDnrm2.
    void dot(const double* a, const double* b, int scalar_slot, bool take_sqrt = false) {
        const int blocks = reduce_blocks_for(n);
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
            colored_block_sweep_f32<<<node_grid(), block_size, 0, stream>>>(
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
        prepare_p_jacobi_f32<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv_f, r_f, v_f,
            p_f, y_f, device_halt, d_slot_map, lanes);
        precondition_sweeps_f32(p_f, y_f);
        matvec_two_group_f32<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag_f, cc_f,
            y_f, v_f, device_halt, d_slot_map, lanes);

        dot_f32(r0_f, v_f, kR0V);
        update_s_jacobi_f32<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv_f, r_f, v_f,
            s_f, z_f, device_halt, d_slot_map, lanes);
        precondition_sweeps_f32(s_f, z_f);
        matvec_two_group_f32<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag_f, cc_f,
            z_f, t_f, device_halt, d_slot_map, lanes);

        dot2_f32(s_f, t_f, kPts, t_f, t_f, kPtt);

        update_solution_f32<<<vector_grid(), block_size, 0, stream>>>(
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
            colored_block_sweep<<<node_grid(), block_size, 0, stream>>>(
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
        dot(r0, r, kRhoNew);
        prepare_p_jacobi<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv, r, v, p, y,
            device_halt, d_slot_map, lanes);
        precondition_sweeps(p, y);
        matvec_two_group<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, y, v, device_halt, d_slot_map, lanes);

        dot(r0, v, kR0V);
        update_s_jacobi<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv, r, v, s, z,
            device_halt, d_slot_map, lanes);
        precondition_sweeps(s, z);
        matvec_two_group<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, z, t, device_halt, d_slot_map, lanes);

        // s.t and t.t are the only two adjacent dots with no kernel between
        // them; one stage-1/stage-2 pair carries both, each with its own
        // accumulator and partial array.
        dot2(s, t, kPts, t, t, kPtt);

        update_solution<<<vector_grid(), block_size, 0, stream>>>(
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
        CUDA_CHECK(cudaMemcpyAsync(host_status,
                                   device_status,
                                   static_cast<size_t>(slots) * sizeof(DeviceSolveStatus),
                                   cudaMemcpyDeviceToHost,
                                   stream));
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
            for (const OuterGraph& e : outer_graphs)
                if (e.nmax == nmax && e.lanes == lanes && e.precision == precisionTag()) {
                    graph_exec = e.exec;
                    break;
                }
            if (graph_exec != nullptr) {
                graph_nmax      = nmax;
                graph_lanes     = lanes;
                graph_precision = precisionTag();
                CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
                ++telemetry.graph_launches;
                if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
                return;
            }
            cudaGraph_t graph = nullptr;
            cudaError_t rc = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (rc == cudaSuccess) {
                enqueue_outer(nmax);
                rc = cudaStreamEndCapture(stream, &graph);
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
            if (graph != nullptr) cudaGraphDestroy(graph);
            if (rc != cudaSuccess) {
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
                destroyGraphCaches();
                use_graph  = false;
                ++telemetry.graph_fallbacks;
                enqueue_outer(nmax);
                return;
            }
            graph_nmax      = nmax;
            graph_lanes     = lanes;
            graph_precision = precisionTag();
            outer_graphs.push_back(OuterGraph{graph_exec, nmax, lanes, precisionTag()});
            g_cmfd_bucket_graphs.fetch_add(1, std::memory_order_relaxed);
            // The capture itself enqueued nothing: replay it now.
        }
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
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
            cmfd_wiel_finalize<<<scalar_grid(), 32, 0, stream>>>(
                nxyz, vec_stride(), ax, s, scalars, sweep_halt, d_slot_map, lanes);
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
            for (const SweepGraph& e : sweep_graphs)
                if (e.key.serves(nmax, unroll, precisionTag(), lanes)) {
                    sweep_graph_exec = e.exec;
                    sweep_graph      = e.key;
                    break;
                }
            if (sweep_graph_exec != nullptr) {
                CUDA_CHECK(cudaGraphLaunch(sweep_graph_exec, stream));
                ++telemetry.graph_launches;
                if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
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
            cudaError_t rc = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (rc == cudaSuccess) {
                enqueue_sweeps(nmax, depth);
                rc = cudaStreamEndCapture(stream, &graph);
            }
            if (rc == cudaSuccess)
                rc = cudaGraphInstantiate(&sweep_graph_exec, graph, 0ull);
            if (graph != nullptr) cudaGraphDestroy(graph);
            if (rc != cudaSuccess) {
                // Same fallback contract as launch_outer: nothing ran during a
                // failed capture, so the direct enqueue below is the first and
                // only execution.
                cudaGetLastError();
                sweep_graph_exec = nullptr;
                sweep_graph      = SweepGraphCapacity{};
                destroyGraphCaches();
                use_graph        = false;
                ++telemetry.graph_fallbacks;
                enqueue_sweeps(nmax, unroll);
                return;
            }
            sweep_graph = SweepGraphCapacity{nmax, depth, precisionTag(), lanes};
            sweep_graphs.push_back(SweepGraph{sweep_graph_exec, sweep_graph});
            g_cmfd_bucket_graphs.fetch_add(1, std::memory_order_relaxed);
        }
        CUDA_CHECK(cudaGraphLaunch(sweep_graph_exec, stream));
        ++telemetry.graph_launches;
        if (iter_batch_used >= 2) ++telemetry.batched_graph_launches;
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
        std::memset(host_active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        std::fill(host_assembly_active.begin(), host_assembly_active.end(), 0u);
        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            host_active[m] = 1u;
            host_assembly_active[static_cast<size_t>(m)] =
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
        for (int m = 0; m < slots; ++m) host_sweep_halt[m] = host_active[m] ? 0u : 1u;

        CUDA_CHECK(cudaMemcpyAsync(device_active, host_active,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_assembly_active, host_assembly_active.data(),
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sweep_halt, host_sweep_halt,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];

            pushOrSkip(xs_chif + m * vec_stride(), sl.host_chif,
                       static_cast<size_t>(n), sl.chif_mirror.valid
                           ? !mirrorMatches(sl.chif_mirror, sl.host_chif,
                                            static_cast<size_t>(n))
                           : true,
                       sl.chif_mirror);
            pushOrSkip(node_vol + m * node_stride(), sl.host_vol,
                       static_cast<size_t>(nxyz), sl.vol_mirror.valid
                           ? !mirrorMatches(sl.vol_mirror, sl.host_vol,
                                            static_cast<size_t>(nxyz))
                           : true,
                       sl.vol_mirror);

            auto push = [&](double* dst, const double* src_host, size_t cnt) {
                if (src_host == nullptr)
                    throw std::invalid_argument("CMFD sweep upload received a null host buffer");
                CUDA_CHECK(cudaMemcpyAsync(dst, src_host, cnt * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
                ++telemetry.bulk_h2d_calls_during_iteration;
                telemetry.bulk_h2d_bytes_during_iteration += cnt * sizeof(double);
            };
            auto push_pending = [&](double* dst, const double* src_host, size_t cnt,
                                    bool& pushed,
                                    cuda_transfer::ByteExactMirror<double>& mirror) {
                if (src_host == nullptr)
                    throw std::invalid_argument("CMFD assembly upload received a null host buffer");
                pushed = !mirror.matches(src_host, cnt);
                if (!pushed) {
                    ++telemetry.bulk_h2d_skipped_during_iteration;
                    return;
                }
                push(dst, src_host, cnt);
            };

            if (sl.device_assembly) {
                push_pending(xs_xsnf + m * vec_stride(), sl.host_xsnf,
                             static_cast<size_t>(n), sl.pushed_xsnf,
                             sl.xsnf_mirror);
                push_pending(xs_xsrf + m * vec_stride(), sl.host_xsrf,
                             static_cast<size_t>(n), sl.pushed_xsrf,
                             sl.xsrf_mirror);
                push_pending(xs_xssm + m * mat_stride(), sl.host_xssm,
                             matrix_count, sl.pushed_xssm, sl.xssm_mirror);
                push_pending(dtil_dev + m * surface_stride(), sl.host_dtil,
                             surface_group_count, sl.pushed_dtil,
                             sl.dtil_mirror);
                // dhat changes after every nodal correction, so comparing it
                // on the launcher's critical path is wasted work.
                push(dhat_dev + m * surface_stride(), sl.host_dhat,
                     surface_group_count);

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
                push(xs_xsnf + m * vec_stride(), sl.host_xsnf, static_cast<size_t>(n));
                push(udiag_dev + m * mat_stride(), sl.host_udiag, matrix_count);
                push(diag + m * mat_stride(), sl.host_diag, matrix_count);
                pushOrSkip(cc + m * cpl_stride(), sl.host_cc, coupling_count,
                           sl.push_cc, sl.cc_mirror);
                ++telemetry.cmfd_assembly_cpu_fallbacks;
            }

            // psi round trip, removed.  See CmfdSweepIO::psi_dirty: only the
            // first launch of a drive carries host-written psi; a later launch
            // in the same drive would be re-uploading the bytes the device
            // itself produced, so leaving the device copy alone is the same
            // state by a shorter path.
            if (sl.push_psi) {
                push(psi_dev + m * node_stride(), sl.host_psi, static_cast<size_t>(nxyz));
            } else {
                ++telemetry.bulk_h2d_skipped_during_iteration;
                telemetry.cmfd_psi_h2d_elided_bytes +=
                    static_cast<std::uint64_t>(nxyz) * sizeof(double);
            }
            pushOrSkip(phi + m * vec_stride(), sl.host_phi, static_cast<size_t>(n),
                       sl.push_phi, sl.phi_mirror);
            CUDA_CHECK(cudaMemcpyAsync(
                scalars + static_cast<long long>(m) * kScalarCount + kSweepFirst,
                sl.sweep_in, kSweepCount * sizeof(double), cudaMemcpyHostToDevice,
                stream));
            if (!(sl.eps == sl.eps_on_device)) {
                CUDA_CHECK(cudaMemcpyAsync(
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
            CUDA_CHECK(cudaMemcpyAsync(
                sl.sweep_out,
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
            CUDA_CHECK(cudaMemcpyAsync(sl.out_phi,
                                       phi + m * vec_stride(),
                                       static_cast<size_t>(n) * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            ++telemetry.bulk_d2h_calls_during_iteration;
            telemetry.bulk_d2h_bytes_during_iteration +=
                static_cast<std::uint64_t>(n) * sizeof(double);
            ++telemetry.status_d2h_calls_during_iteration;
        }
    }

    void commitAssemblyMirrors(const int* active_slots, int count) {
        for (int i = 0; i < count; ++i) {
            Slot& sl = slot[static_cast<size_t>(active_slots[i])];
            if (!sl.device_assembly) continue;
            if (sl.pushed_xsrf) sl.xsrf_mirror.commit(sl.host_xsrf, static_cast<size_t>(n));
            if (sl.pushed_xssm) sl.xssm_mirror.commit(sl.host_xssm, matrix_count);
            if (sl.pushed_xsnf) sl.xsnf_mirror.commit(sl.host_xsnf, static_cast<size_t>(n));
            if (sl.pushed_dtil) sl.dtil_mirror.commit(sl.host_dtil, surface_group_count);
        }
    }

    /// A degenerate Wielandt gamma hands control to the host Rayleigh branch.
    /// Only those exceptional slots need a host copy of the operator that the
    /// assembly kernel produced; the normal path never downloads diag/cc/udiag.
    void issueExceptionalOperatorDownloads(const int* active_slots, int count) {
        bool queued = false;
        for (int i = 0; i < count; ++i) {
            const int m = active_slots[i];
            Slot& sl = slot[static_cast<size_t>(m)];
            const int state = static_cast<int>(sl.sweep_out[kSweepState - kSweepFirst]);
            if (state != 2) continue;
            // The Rayleigh hand-back in BICGCMFD reads psi(l) for sumf/summ,
            // and this is the one launch on which it does.  Unconditional on
            // device_assembly: the host arm needs the new fission source too.
            if (sl.host_psi != nullptr && !sl.psi_downloaded) {
                CUDA_CHECK(cudaMemcpyAsync(sl.host_psi, psi_dev + m * node_stride(),
                                           static_cast<size_t>(nxyz) * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
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
            CUDA_CHECK(cudaMemcpyAsync(sl.host_diag_out, diag + m * mat_stride(),
                                       matrix_count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(sl.host_cc_out, cc + m * cpl_stride(),
                                       coupling_count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(sl.host_udiag, udiag_dev + m * mat_stride(),
                                       matrix_count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            telemetry.bulk_d2h_calls_during_iteration += 3;
            telemetry.bulk_d2h_bytes_during_iteration +=
                (2 * static_cast<std::uint64_t>(matrix_count) +
                 static_cast<std::uint64_t>(coupling_count)) * sizeof(double);
            queued = true;
        }
        if (!queued) return;
        ++telemetry.stream_sync_calls_during_iteration;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
    }

    /// The one drain per launch.  It covers the flux copies *and* the status
    /// packets the graph already queued, which is why the inner loop needs none.
    void drain(const int* active_slots, int count) {
        ++telemetry.stream_sync_calls_during_iteration;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        commitAssemblyMirrors(active_slots, count);

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
    /// the FP64 topology.  Called from drain(), i.e. with the stream already
    /// synchronised, so destroying the executables here is safe.
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
    int           rb_sweeps = 4;
    int           ncolors   = 2; // sweep colour count (2 = historical red/black; >2 under the rotational fold)
    double *diag = nullptr, *dinv = nullptr, *cc = nullptr, *src = nullptr, *phi = nullptr;
    double *r = nullptr, *r0 = nullptr, *p = nullptr, *v = nullptr, *s = nullptr, *t = nullptr;
    double *y = nullptr, *z = nullptr, *ax = nullptr;
    double*        partials = nullptr;
    double*        partials2 = nullptr;
    double*        scalars = nullptr;
    std::uint32_t* device_flags = nullptr;
    std::uint32_t* iter_flags = nullptr;
    std::uint32_t* device_halt = nullptr;
    std::uint32_t* device_active = nullptr;
    std::uint32_t* device_counters = nullptr;
    DeviceSolveStatus* device_status = nullptr;
    DeviceSolveStatus* host_status = nullptr;
    std::uint32_t*     host_active = nullptr;
    /// Staging for the sweep_halt H2D.  Separate from host_active on purpose:
    /// both are page-locked and both are memcpyAsync SOURCES in the same
    /// launcher window, so neither may be rewritten to serve the other.
    std::uint32_t*     host_sweep_halt = nullptr;
    BackendCounters telemetry{};
    std::vector<Slot> slot;
    std::vector<std::uint32_t> host_assembly_active;
    bool          use_graph = true;
    bool          scalar_fusion = true;
    /// RASBERY_GPU_ITER_BATCH: requested iterations per graph launch.  0 =
    /// unset = follow the algorithmic budget (today's behaviour, and the only
    /// setting for which the capture depth is exactly `1 + nmax`).
    int           iter_batch_request = 0;
    /// What the last capture actually carried.
    int           iter_batch_used = 0;
    cudaGraphExec_t graph_exec = nullptr;
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
    };
    struct SweepGraph {
        cudaGraphExec_t    exec;
        SweepGraphCapacity key;
    };
    std::vector<OuterGraph> outer_graphs;
    std::vector<SweepGraph> sweep_graphs;

    void destroyGraphCaches() {
        for (const OuterGraph& e : outer_graphs)
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
        for (const SweepGraph& e : sweep_graphs)
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
        if (!outer_graphs.empty() || !sweep_graphs.empty())
            telemetry.graph_reinstantiations +=
                outer_graphs.size() + sweep_graphs.size();
        outer_graphs.clear();
        sweep_graphs.clear();
        graph_exec       = nullptr;
        graph_nmax       = -1;
        graph_lanes      = -1;
        graph_precision  = -1;
        sweep_graph_exec = nullptr;
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
    unsigned long long      launches       = 0;
    unsigned long long      batched_solves = 0;
    std::vector<unsigned long long> width_histogram;
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
        sl.in_use = true;
        return m;
    }
    return -1;
}

void CudaBatchArena::releaseSlot(int m) {
    if (m < 0) return;
    std::lock_guard<std::mutex> lock(_impl->mutex);
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
    const cudaError_t rc = cudaHostRegister(address, bytes, cudaHostRegisterDefault);
    if (rc != cudaSuccess) cudaGetLastError(); // already registered / exotic host
    return static_cast<int>(rc);
}

int cudaHostPinUnregister(void* address) {
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
    if (a.launches == 0) return;
    std::ostringstream line;
    line << "[RASBERY][CUDA][BATCH_OCCUPANCY] {\"tag\":\"" << (tag ? tag : "") << "\","
         << "\"slots\":" << a.core.slots << ','
         << "\"launches\":" << a.launches << ','
         << "\"instance_solves\":" << a.batched_solves << ','
         << "\"mean_width\":"
         << (static_cast<double>(a.batched_solves) / static_cast<double>(a.launches)) << ','
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
    const BackendCounters c = g_arena->counters();
    std::cout << "[RASBERY][CUDA][BACKEND_COUNTERS] {"
              << "\"cmfd_gpu_calls\":" << c.cmfd_gpu_calls << ','
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
