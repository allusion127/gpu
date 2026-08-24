#include "CudaBICGBackend.h"

#include "CudaXsReconBackend.h" // rasberyHostPinningEnabled(): header-only gate
#include "Geometry.h"
#include "pch.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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
                                  const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                  const std::uint32_t* __restrict__ halt) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
    HALT_GUARD(halt + m);
    const double* pm = partial + static_cast<long long>(m) * kMaxReduceBlocks;
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order
    scalars[static_cast<long long>(m) * kScalarCount + slot] = take_sqrt ? sqrt(sum) : sum;
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
                                   const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                   const std::uint32_t* __restrict__ halt) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
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
                                 const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                  const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                     const std::uint32_t* __restrict__ halt) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
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
                                 const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                    const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                const std::uint32_t* __restrict__ halt) {
    const int m = static_cast<int>(blockIdx.y);
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

/// End-of-iteration bookkeeping that the host used to perform after every
/// status D2H: fold the per-iteration flags into the sticky set, tally the
/// telemetry, and -- for every iteration except the first, which the CPU
/// reference also runs unconditionally -- apply the relative exit test.
__global__ void accumulate_iteration(const int allow_halt,
                                     const int force_halt,
                                     const double* __restrict__ scalars,
                                     std::uint32_t* iter_flags,
                                     std::uint32_t* sticky_flags,
                                     std::uint32_t* counters,
                                     std::uint32_t* halt,
                                     const std::uint32_t* __restrict__ active) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
    // A slot that is not taking part in this launch never had an iteration to
    // over-run; it is masked exactly as the HALT_GUARD used to mask it.
    if (active[m] == 0u) return;

    const double*  sm  = scalars + static_cast<long long>(m) * kScalarCount;
    std::uint32_t* cm  = counters + static_cast<long long>(m) * kCounterSlots;

    // This is where the HALT_GUARD used to sit.  Splitting it out is what
    // makes the over-run visible: every kernel of this iteration returned on
    // its first instruction, so nothing in the solver state moved, and the
    // only thing left to do is say so.  The counter is telemetry -- no other
    // kernel and no host decision reads it -- so the numerical trajectory is
    // byte for byte the one the plain HALT_GUARD produced.
    if (halt[m] != 0u) {
        ++cm[kOverrunCount];
        return;
    }

    const std::uint32_t flags = iter_flags[m];
    // The other half of the retired per-iteration cudaMemsetAsync node: arm
    // the scratch word for the NEXT captured iteration, here rather than in a
    // memset node of its own.  Placed immediately after the read, so it holds
    // on every exit below.
    //
    // The paths that skip it are exactly the paths on which nobody reads it
    // again: `active == 0` (the slot never runs a kernel of this launch) and
    // `halt != 0` (halt is monotone within an outer -- only this kernel sets
    // it and only initialize_solver_state clears it -- so every later
    // iteration returns here before the read, and the only writers of
    // iter_flags are HALT_GUARDed).  initialize_solver_state zeroes the word
    // for every slot at the top of each outer, which covers the first
    // iteration and re-arms a slot whose previous outer halted.
    iter_flags[m] = 0u;
    const double        ptt   = sm[kPtt];
    const double        rnorm = sm[kInitialNorm];

    // Same corrupt-state test finalize_status applied: a non-finite or
    // negative t.t means the reported residual is meaningless.
    const bool corrupt = !isfinite(ptt) || ptt < 0.0 || !isfinite(rnorm);

    sticky_flags[m] |= flags;
    if (corrupt) sticky_flags[m] |= static_cast<std::uint32_t>(NONFINITE_DETECTED);
    if ((flags & static_cast<std::uint32_t>(BICGSTAB_BREAKDOWN)) != 0)
        ++cm[kRestartCount];
    if ((flags & static_cast<std::uint32_t>(FLUX_CONVERGED)) != 0)
        ++cm[kEarlyExitCount];
    ++cm[kSolveCount];

    if (corrupt || (sticky_flags[m] & static_cast<std::uint32_t>(NONFINITE_DETECTED)) != 0) {
        // The host threw as soon as it saw this; stop the batch so the state
        // handed back is the first bad one rather than three more of them.
        halt[m] = 1u;
        return;
    }
    if (allow_halt != 0) {
        const double r20 = sm[kR20];
        if (r20 <= 0.0 || rnorm / r20 < sm[kEps]) halt[m] = 1u;
    }
    // The algorithmic budget is `1 + nmax` iterations -- the host loop stopped
    // there whether or not the residual test had fired.  When the graph
    // captures MORE than that (RASBERY_GPU_ITER_BATCH > 1 + nmax), the last
    // budgeted iteration raises the halt itself, which is what makes every
    // extra captured iteration provably inert instead of silently deepening
    // the inner solve.  With the default batch this argument is always 0 and
    // the kernel behaves exactly as before.
    if (force_halt != 0) halt[m] = 1u;
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

__global__ void cmfd_sweep_begin(double* scalars, std::uint32_t* sweep_halt) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
    if (sweep_halt[m] != 0u) return;
    double* sm    = scalars + static_cast<long long>(m) * kScalarCount;
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
                               const std::uint32_t* __restrict__ sweep_halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                const std::uint32_t* __restrict__ sweep_halt) {
    const int m = static_cast<int>(blockIdx.y);
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

/// The serial l-ascending fold of the stored addends, plus the eigenvalue
/// update.  One thread per slot; slots in parallel on gridDim.y.  The
/// warm-up (icy < 0) and Rayleigh branches never run here: the host only
/// delegates once the Wielandt regime is reached, and a degenerate gamma
/// hands the sweep back to the host with the sums exported.
__global__ void cmfd_wiel_finalize(const int nxyz,
                                   const long long vec_stride,
                                   const double* __restrict__ terms_ab,
                                   const double* __restrict__ terms_c,
                                   double* scalars,
                                   std::uint32_t* sweep_halt) {
    const int m = static_cast<int>(blockIdx.y);
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
        for (int l = 0; l < nxyz; ++l) sum = sum + values[l];
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
                           const std::uint32_t* __restrict__ sweep_halt) {
    const int m = static_cast<int>(blockIdx.y);
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
                                   const std::uint32_t* __restrict__ sweep_halt) {
    const int m = static_cast<int>(blockIdx.y);
    if (sweep_halt[m] != 0u) return;
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (phi[m * vec_stride + i] < 0.0)
        atomicAdd(&scalars[static_cast<long long>(m) * kScalarCount + kNegative], 1.0);
}

/// The host loop's control tail: the all-negative reset, the retry rule that
/// refuses to count a negative-flux sweep against iout, and the exit tests.
__global__ void cmfd_sweep_end(double* scalars, std::uint32_t* sweep_halt) {
    if (threadIdx.x != 0) return;
    const int m = static_cast<int>(blockIdx.y);
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
        const double* host_vol   = nullptr;
        const double* host_udiag = nullptr;
        double*       host_psi   = nullptr; ///< in/out
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
            status += " (block=" + std::to_string(block_size) +
                      ", RB sweeps=" + std::to_string(rb_sweeps) +
                      ", graph=" + (use_graph ? "on" : "off") +
                      ", iter batch=" +
                      (iter_batch_request > 0 ? std::to_string(iter_batch_request)
                                              : std::string("auto")) +
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
            allocate(reinterpret_cast<void**>(&node_vol), S * static_cast<size_t>(nxyz) * sizeof(double));
            allocate(reinterpret_cast<void**>(&udiag_dev), S * matrix_count * sizeof(double));
            allocate(reinterpret_cast<void**>(&psi_dev), S * static_cast<size_t>(nxyz) * sizeof(double));
            allocate(reinterpret_cast<void**>(&sweep_halt), S * sizeof(std::uint32_t));
            CUDA_CHECK(cudaMemset(sweep_halt, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemset(device_halt, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemset(device_active, 0, S * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_status),
                                      S * sizeof(DeviceSolveStatus)));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_active),
                                      S * sizeof(std::uint32_t)));
            std::memset(host_active, 0, S * sizeof(std::uint32_t));
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
        if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
        if (handle != nullptr) cublasDestroy(handle);
        handle = nullptr;
        if (stream != nullptr) cudaStreamDestroy(stream);
        stream = nullptr;
        cudaFree(neighbors);
        cudaFree(colors);
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
        if (sweep_graph_exec != nullptr) cudaGraphExecDestroy(sweep_graph_exec);
        sweep_graph_exec = nullptr;
        cudaFree(xs_chif);
        cudaFree(xs_xsnf);
        cudaFree(node_vol);
        cudaFree(udiag_dev);
        cudaFree(psi_dev);
        cudaFree(sweep_halt);
        xs_chif = xs_xsnf = node_vol = udiag_dev = psi_dev = nullptr;
        sweep_halt = nullptr;
        if (host_status != nullptr) cudaFreeHost(host_status);
        host_status = nullptr;
        if (host_active != nullptr) cudaFreeHost(host_active);
        host_active = nullptr;
        neighbors = nullptr;
        colors = nullptr;
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

    /// Exact compatibility check for the immutable CMFD topology.  Shape
    /// counts alone are insufficient: two loading maps can have identical
    /// nxyz/ngxyz yet different neighbours, and sharing the first map would
    /// produce physically wrong CMFD results without an allocation error.
    [[nodiscard]] bool compatibleGeometry(Geometry& geometry) const {
        if (geometry.ng() != 2 || geometry.nxyz() != nxyz ||
            geometry.ngxyz() != n ||
            static_cast<size_t>(geometry.ng2()) * geometry.nxyz() != matrix_count)
            return false;
        for (int l = 0; l < nxyz; ++l)
            for (int idir = 0; idir < NDIRMAX; ++idir)
                for (int lr = 0; lr < LR; ++lr) {
                    const size_t idx = static_cast<size_t>(6 * l + idir * LR + lr);
                    if (topology_neighbors[idx] != geometry.neib(lr, idir, l))
                        return false;
                }
        return true;
    }

    /// Grid for a per-node/per-element kernel: x is the *single-instance* grid,
    /// y is the batch axis.  Never fold the two.
    [[nodiscard]] dim3 node_grid() const {
        return dim3(static_cast<unsigned>(node_blocks()), static_cast<unsigned>(slots));
    }
    [[nodiscard]] dim3 vector_grid() const {
        return dim3(static_cast<unsigned>(vector_blocks()), static_cast<unsigned>(slots));
    }
    [[nodiscard]] dim3 scalar_grid() const {
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
        sl.host_phi  = host_phi;
        sl.host_src  = host_src;
        // diag is rewritten every outer (Wielandt) and every CMFD sweep
        // (updls), so its mirror can never match: skip the 4n-double memcmp
        // here and the 4n-double shadow copy on the launcher's critical path,
        // and just upload it every time -- which is what happened anyway.
        sl.push_diag = true;
        sl.push_cc   = !mirrorMatches(sl.cc_mirror, host_cc, coupling_count);
        sl.push_phi  = !mirrorMatches(sl.phi_mirror, host_phi, static_cast<size_t>(n));
    }

    static bool mirrorMatches(const MirroredUpload& mirror, const double* host, size_t count) {
        return mirror.valid &&
               std::memcmp(mirror.shadow.data(), host, count * sizeof(double)) == 0;
    }

    /// H2D for the participating slots, plus the participation mask itself.
    void issueUploads(const int* active_slots, int count) {
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
        reduce_dot_stage1<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(slots)),
                            kReduceThreads, 0, stream>>>(
            n, vec_stride(), a, b, partials, device_halt);
        reduce_dot_stage2<<<scalar_grid(), 1, 0, stream>>>(
            blocks, partials, scalars, scalar_slot, take_sqrt, device_halt);
    }

    /// Two independent dots over the same n in ONE pair of nodes.  See
    /// reduce_dot2_stage1: this is not a fused reduction, it is two reductions
    /// riding in one kernel with their own accumulators, their own partial
    /// arrays and the same partition each had alone.
    void dot2(const double* a0, const double* b0, int scalar_slot0,
              const double* a1, const double* b1, int scalar_slot1) {
        const int blocks = reduce_blocks_for(n);
        reduce_dot2_stage1<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(slots)),
                             kReduceThreads, 0, stream>>>(
            n, vec_stride(), a0, b0, a1, b1, partials, partials2, device_halt);
        reduce_dot2_stage2<<<scalar_grid(), 1, 0, stream>>>(
            blocks, partials, partials2, scalars, scalar_slot0, scalar_slot1, device_halt);
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
                cc, dinv, b, x, device_halt);
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
            device_halt);
        precondition_sweeps(p, y);
        matvec_two_group<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, y, v, device_halt);

        dot(r0, v, kR0V);
        update_s_jacobi<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), scalars, iter_flags, dinv, r, v, s, z,
            device_halt);
        precondition_sweeps(s, z);
        matvec_two_group<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, z, t, device_halt);

        // s.t and t.t are the only two adjacent dots with no kernel between
        // them; one stage-1/stage-2 pair carries both, each with its own
        // accumulator and partial array.
        dot2(s, t, kPts, t, t, kPtt);

        update_solution<<<vector_grid(), block_size, 0, stream>>>(
            n, vec_stride(), scalars, iter_flags, y, z, s, t, phi, r, device_halt);
        // Absolute residual of the iterate that update_solution just wrote (or
        // of the unchanged iterate on the breakdown / early-exit paths, which
        // is what the CPU publishes there too).
        dot(r, r, kInitialNorm, /*take_sqrt=*/true);
        accumulate_iteration<<<scalar_grid(), 1, 0, stream>>>(
            allow_halt, force_halt, scalars, iter_flags, device_flags, device_counters,
            device_halt, device_active);
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
        initialize_solver_state<<<scalar_grid(), 1, 0, stream>>>(
            scalars, device_flags, device_halt, device_counters, iter_flags,
            device_active, sweep_halt);
        // One node for what used to be three: the block inversion (independent
        // of the other two), the A*phi matvec and the residual it feeds.
        begin_outer_fused<<<node_grid(), block_size, 0, stream>>>(
            nxyz, vec_stride(), mat_stride(), cpl_stride(), neighbors, diag, cc, phi,
            src, dinv, ax, r, r0, p, v, device_halt);
        dot(r, r, kInitialNorm, /*take_sqrt=*/true);
        store_reference_norm<<<scalar_grid(), 1, 0, stream>>>(scalars, device_halt);

        // The algorithmic budget is `1 + nmax`; the capture may be deeper.
        // Iteration `algorithmic - 1` then raises the halt itself, so every
        // captured iteration past the budget finds halt set, returns on its
        // first instruction in every kernel, and is counted as an over-run.
        const int algorithmic = 1 + nmax;
        const int captured    = captured_iterations(nmax);
        for (int i = 0; i < captured; ++i)
            enqueue_iteration(/*allow_halt=*/i == 0 ? 0 : 1,
                              /*force_halt=*/(i == algorithmic - 1 && captured > algorithmic)
                                  ? 1
                                  : 0);
        // A property of the capture, not a tally: assigned, never accumulated.
        // Set here rather than at launch so it is right on the graph-off path
        // too, where there is no launch to hang it off.
        iter_batch_used      = captured;
        telemetry.iter_batch = static_cast<std::uint64_t>(captured);

        finalize_status<<<scalar_grid(), 1, 0, stream>>>(
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
        if (graph_exec == nullptr || graph_nmax != nmax) {
            if (graph_exec != nullptr) {
                CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
                graph_exec = nullptr;
                ++telemetry.graph_reinstantiations;
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
                use_graph  = false;
                ++telemetry.graph_fallbacks;
                enqueue_outer(nmax);
                return;
            }
            graph_nmax = nmax;
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
        for (int sweep = 0; sweep < unroll; ++sweep) {
            cmfd_sweep_begin<<<scalar_grid(), 1, 0, stream>>>(scalars, sweep_halt);
            cmfd_src_build<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), node_stride(), xs_chif, psi_dev, src, scalars,
                sweep_halt);
            enqueue_outer(nmax);
            // ax/s are BiCG scratch, dead between the inner loop and the next
            // sweep's initial residual; they carry the wiel addends here.
            cmfd_wiel_terms<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), node_stride(), phi, psi_dev, xs_xsnf, node_vol,
                ax, s, sweep_halt);
            cmfd_wiel_finalize<<<scalar_grid(), 32, 0, stream>>>(
                nxyz, vec_stride(), ax, s, scalars, sweep_halt);
            cmfd_updls<<<node_grid(), block_size, 0, stream>>>(
                nxyz, vec_stride(), node_stride(), mat_stride(), xs_chif, xs_xsnf,
                node_vol, udiag_dev, diag, scalars, sweep_halt);
            cmfd_negative_scan<<<vector_grid(), block_size, 0, stream>>>(
                n, vec_stride(), phi, scalars, sweep_halt);
            cmfd_sweep_end<<<scalar_grid(), 1, 0, stream>>>(scalars, sweep_halt);
        }
    }

    /// Graph-cached counterpart of launch_outer for the sweep sequence.
    void launch_sweeps(int nmax, int unroll) {
        if (!use_graph) {
            enqueue_sweeps(nmax, unroll);
            return;
        }
        if (sweep_graph_exec == nullptr || sweep_graph_nmax != nmax ||
            sweep_graph_unroll != unroll) {
            if (sweep_graph_exec != nullptr) {
                CUDA_CHECK(cudaGraphExecDestroy(sweep_graph_exec));
                sweep_graph_exec = nullptr;
                ++telemetry.graph_reinstantiations;
            }
            cudaGraph_t graph = nullptr;
            cudaError_t rc = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (rc == cudaSuccess) {
                enqueue_sweeps(nmax, unroll);
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
                use_graph        = false;
                ++telemetry.graph_fallbacks;
                enqueue_sweeps(nmax, unroll);
                return;
            }
            sweep_graph_nmax   = nmax;
            sweep_graph_unroll = unroll;
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
    void issueSweepUploads(const int* active_slots, int count) {
        std::memset(host_active, 0, static_cast<size_t>(slots) * sizeof(std::uint32_t));
        for (int i = 0; i < count; ++i) host_active[active_slots[i]] = 1u;
        CUDA_CHECK(cudaMemcpyAsync(device_active, host_active,
                                   static_cast<size_t>(slots) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        // participants: sweep_halt = 0; everyone else: 1 (masks their slots
        // inside every sweep kernel AND the inner reset).
        for (int m = 0; m < slots; ++m) host_active[m] = host_active[m] ? 0u : 1u;
        CUDA_CHECK(cudaMemcpyAsync(sweep_halt, host_active,
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
                CUDA_CHECK(cudaMemcpyAsync(dst, src_host, cnt * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
                ++telemetry.bulk_h2d_calls_during_iteration;
                telemetry.bulk_h2d_bytes_during_iteration += cnt * sizeof(double);
            };
            push(xs_xsnf + m * vec_stride(), sl.host_xsnf, static_cast<size_t>(n));
            push(udiag_dev + m * mat_stride(), sl.host_udiag, matrix_count);
            push(psi_dev + m * node_stride(), sl.host_psi, static_cast<size_t>(nxyz));
            push(diag + m * mat_stride(), sl.host_diag, matrix_count);
            pushOrSkip(cc + m * cpl_stride(), sl.host_cc, coupling_count, sl.push_cc,
                       sl.cc_mirror);
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

    /// D2H after the sweep graph: flux (issueFluxDownloads), psi and the sweep
    /// scalar block per participant, then the sweep_halt restore.
    void issueSweepDownloads(const int* active_slots, int count) {
        for (int i = 0; i < count; ++i) {
            const int m  = active_slots[i];
            Slot&     sl = slot[static_cast<size_t>(m)];
            CUDA_CHECK(cudaMemcpyAsync(sl.host_psi, psi_dev + m * node_stride(),
                                       static_cast<size_t>(nxyz) * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                sl.sweep_out,
                scalars + static_cast<long long>(m) * kScalarCount + kSweepFirst,
                kSweepCount * sizeof(double), cudaMemcpyDeviceToHost, stream));
            ++telemetry.bulk_d2h_calls_during_iteration;
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
            ++telemetry.status_d2h_calls_during_iteration;
        }
    }

    /// The one drain per launch.  It covers the flux copies *and* the status
    /// packets the graph already queued, which is why the inner loop needs none.
    void drain(const int* active_slots, int count) {
        ++telemetry.stream_sync_calls_during_iteration;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

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
            slot[static_cast<size_t>(m)].nonfinite =
                (host_status[m].flags & NONFINITE_DETECTED) != 0;
        }
    }

    /// Host and device agree on slot m's flux once its batch drained; record
    /// it so the next stage() can skip the upload.  Runs on the OWNING
    /// instance's thread after batch completion, never on the launcher's
    /// critical path, and never on a failed batch (the flux is undefined).
    void adoptFluxMirror(int m) {
        Slot& sl = slot[static_cast<size_t>(m)];
        sl.phi_mirror.shadow.assign(sl.out_phi, sl.out_phi + n);
        sl.phi_mirror.valid = true;
    }

    int           slots;
    int           nxyz;
    int           n;
    size_t        matrix_count;
    size_t        coupling_count;
    std::vector<int> topology_neighbors;
    bool          available = false;
    std::string   status;
    cublasHandle_t handle = nullptr;
    cudaStream_t  stream = nullptr;
    int*          neighbors = nullptr;
    int*          colors = nullptr;
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
    BackendCounters telemetry{};
    std::vector<Slot> slot;
    bool          use_graph = true;
    /// RASBERY_GPU_ITER_BATCH: requested iterations per graph launch.  0 =
    /// unset = follow the algorithmic budget (today's behaviour, and the only
    /// setting for which the capture depth is exactly `1 + nmax`).
    int           iter_batch_request = 0;
    /// What the last capture actually carried.
    int           iter_batch_used = 0;
    cudaGraphExec_t graph_exec = nullptr;
    int           graph_nmax = -1;

    // ---- device-resident CMFD sweep state (RASBERY_GPU_CMFD_SWEEP) ----
    double*        xs_chif    = nullptr; ///< [slot][ig*nxyz+l]
    double*        xs_xsnf    = nullptr; ///< [slot][ig*nxyz+l]
    double*        node_vol   = nullptr; ///< [slot][l]
    double*        udiag_dev  = nullptr; ///< [slot][l*ng2+ige*ng+igs]
    double*        psi_dev    = nullptr; ///< [slot][l]
    std::uint32_t* sweep_halt = nullptr; ///< all-zero outside the sweep path
    cudaGraphExec_t sweep_graph_exec = nullptr;
    int             sweep_graph_nmax = -1;
    int             sweep_graph_unroll = -1;

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
BackendCounters CudaBICGBackend::counters() const { return _impl->core.telemetry; }
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
BackendCounters CudaBatchArena::counters() const { return _impl->core.telemetry; }

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

void CudaBatchArena::pinHost(const void* p, size_t bytes) const {
    if (p == nullptr || bytes == 0) return;
    // Permanent registration, so only legal while the caller's buffer outlives
    // the process.  A recycled Driver worker breaks that: see
    // rasberySetHostPinningEnabled() in CudaXsReconBackend.h.
    if (!rasberyHostPinningEnabled()) return;
    const cudaError_t rc =
        cudaHostRegister(const_cast<void*>(p), bytes, cudaHostRegisterDefault);
    if (rc != cudaSuccess) cudaGetLastError(); // already registered / exotic host
}

void CudaBatchArena::stageSweeps(int m, const CmfdSweepIO& io) {
    auto& sl      = _impl->core.slot[static_cast<size_t>(m)];
    sl.host_chif  = io.chif;
    sl.host_xsnf  = io.xsnf;
    sl.host_vol   = io.vol;
    sl.host_udiag = io.udiag;
    sl.host_psi   = io.psi;
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
    sl.sweep_unroll                = io.sweep_budget;
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
                                             static_cast<int>(participants.size()));
                    a.core.launch_sweeps(nmax, unroll);
                    a.core.issueFluxDownloads(participants.data(),
                                              static_cast<int>(participants.size()));
                    a.core.issueSweepDownloads(participants.data(),
                                               static_cast<int>(participants.size()));
                    a.core.drain(participants.data(),
                                 static_cast<int>(participants.size()));
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

CudaBatchArena* rasberyBatchArena(Geometry& geometry) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (!g_arena) {
        g_arena = std::make_unique<CudaBatchArena>(geometry, g_batch_width);
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

void rasberyReleaseBatchArena() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (!g_arena) return;
    g_arena->reportBatchOccupancy("run");
    const BackendCounters c = g_arena->counters();
    std::cout << "[RASBERY][CUDA][BACKEND_COUNTERS] {"
              << "\"cmfd_gpu_calls\":" << c.cmfd_gpu_calls << ','
              << "\"bicg_early_convergence_exits\":" << c.bicg_early_convergence_exits << ','
              << "\"bicg_restarts\":" << c.bicg_restarts << ','
              << "\"bulk_h2d_calls_during_iteration\":" << c.bulk_h2d_calls_during_iteration << ','
              << "\"bulk_h2d_skipped_during_iteration\":" << c.bulk_h2d_skipped_during_iteration
              << ','
              << "\"bulk_h2d_bytes_during_iteration\":" << c.bulk_h2d_bytes_during_iteration << ','
              << "\"bulk_d2h_calls_during_iteration\":" << c.bulk_d2h_calls_during_iteration << ','
              << "\"stream_sync_calls_during_iteration\":" << c.stream_sync_calls_during_iteration
              << ','
              << "\"graph_launches\":" << c.graph_launches << ','
              << "\"graph_reinstantiations\":" << c.graph_reinstantiations << ','
              << "\"graph_fallbacks\":" << c.graph_fallbacks << ','
              << "\"iter_batch\":" << c.iter_batch << ','
              << "\"batched_graph_launches\":" << c.batched_graph_launches << ','
              << "\"overrun_iterations\":" << c.overrun_iterations << '}' << std::endl;
    g_arena.reset();
}

} // namespace rasbery
