#include "CudaXsReconBackend.h"

#include "BatchRefill.h"
#include "CudaTransferMirror.h"
#include "FlatXsKernel.h"
#include "GpuCanonicalState.h"
#include "GpuCaptureArbiter.h"
#include "GpuGraphSplice.h"
#include "NodalKernel.h"
#include "XeFormMask.h"
#include "XeGpuReceipt.h"
#include "XeKernel.h"
#include "XsReconKernel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// This translation unit must be compiled with --fmad=false (set in
// CMakeLists.txt).  The kernel's contract is bit-identical reproduction of the
// host loop; letting nvcc contract a*b+c would make the comparison depend on
// which expressions each compiler chose to fuse.  See XsReconKernel.h.

namespace rasbery {

namespace xsr = rasbery::xsrecon;

namespace {

std::atomic<unsigned long long> g_nodes_solved{0};

// Rev.7.1 Task 13 receipts.  PROCESS-WIDE and therefore summed over every
// Driver in a --batch-mode run, exactly like g_nodes_solved: they answer "did
// the arm fire at all" (G0), which is a question about the process.  Nothing
// per-deck lives here -- the Anderson history is a per-backend allocation and
// there is no static holding deck state anywhere in this arm.
std::atomic<unsigned long long> g_xe_evaluations{0};
std::atomic<unsigned long long> g_xe_commits{0};

bool envFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE");
}

/// Default-ON counterpart: true only when the variable is present AND falsy.
bool envFlagDisabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE";
}

// RASBERY_NODAL_REFACTOR_V1
/// Mat and Even are node-local and have an exact producer/consumer relation:
/// calculateEven(lk) reads only the matrices and coefficients produced by
/// updateMatrix(lk). Running both bodies in one thread removes one graph node
/// without changing any floating-point expression or cross-node ordering.
bool nodalFuseMatEvenEnabled() {
    static const bool on = !envFlagDisabled("RASBERY_GPU_NODAL_FUSE_MAT_EVEN");
    return on;
}

/// The arena has a separate XS allocation, so hoststateGeneration is not an
/// honest residency key. A byte shadow is: an upload is skipped only when the
/// complete incoming xsrf/xsnf/xssm bytes equal the last successfully drained
/// upload for that same slot and destination.
bool nodalXsMirrorEnabled() {
    static const bool on = !envFlagDisabled("RASBERY_GPU_NODAL_XS_MIRROR");
    return on;
}

#define RASBERY_CUDA_TRY(expr, sink)                                         \
    do {                                                                     \
        const cudaError_t _e = (expr);                                       \
        if (_e != cudaSuccess) {                                             \
            (sink) = std::string(#expr) + " -> " + cudaGetErrorString(_e);   \
            return false;                                                    \
        }                                                                    \
    } while (0)

/// Rev.7.1 Task 18d: the ALLOCATING twin of RASBERY_CUDA_TRY.
///
/// Identical in every respect except that it holds a rasbery::AllocWindow for
/// the duration of the call.  Allocation, page-locking and device-wide
/// synchronisation are exactly the APIs CUDA calls "potentially unsafe" while a
/// graph capture is open, and `--batch-mode` runs them on one deck's Driver
/// thread while another deck is capturing the shared arena's graph.  The window
/// is what keeps the two apart; see GpuCaptureArbiter.h for the failure it
/// prevents, and tools/test_gpu_capture_arbiter_contract.py for the source rule
/// that keeps this file using it.
#define RASBERY_CUDA_TRY_ALLOC(expr, sink)                                   \
    do {                                                                     \
        rasbery::AllocWindow _alloc_window(#expr);                           \
        RASBERY_CUDA_TRY(expr, sink);                                        \
    } while (0)

// max_change >= 0 always (|dXe| / max(|Xe|, 1e-30)), and for non-negative
// IEEE doubles the unsigned-integer order of the bit patterns is the value
// order, so a 64-bit atomicMax is an exact, order-insensitive max reduction.
__global__ void kernelXsRecon(xsr::BatchView v, unsigned long long* max_bits,
                              unsigned long long* solved) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.n_fuel) return;
    const int l = v.fuel[i];

    double mc = 0.0;
    if (xsreconSolveNode(v, l, &mc)) {
        atomicMax(max_bits, static_cast<unsigned long long>(__double_as_longlong(mc)));
        atomicAdd(solved, 1ULL);
    }
}

namespace xek = rasbery::xe;

/// One preloaded dot layout: 2*XE_DOT_COUNT pair ids followed by XE_DOT_COUNT
/// destination slots, which is what kXeDotStage1/2 read.
constexpr int XE_TXN_LAYOUT_INTS = 3 * xek::XE_DOT_COUNT;

// ---------------------------------------------------------------------------
// Rev.7.1 Task 13 -- the split Xe arm's three kernels
// ---------------------------------------------------------------------------
//
// THE HISTORY BLOCK is one allocation laid out [row][triple][ordinal]:
//
//     base + (row * XE_TRIPLE_COUNT + triple) * n_fuel
//
// row-major over the TRIPLE and not over the row, which looks backwards until
// the candidate kernel is read: it wants the same row of consecutive window
// columns (df[0].i135 and df[1].i135) n_fuel apart, so the column index is a
// plain stride.  A triple's three rows being 10*n_fuel apart costs nothing --
// XeTripleConst carries three pointers either way.

__device__ __host__ inline const double* xeRow(const double* base, int row, int triple,
                                               int n_fuel) {
    return base + (static_cast<long long>(row) * xek::XE_TRIPLE_COUNT + triple) * n_fuel;
}

__device__ __host__ inline xek::XeTriple xeTripleAt(double* base, int triple, int n_fuel) {
    xek::XeTriple t;
    t.i135   = const_cast<double*>(xeRow(base, 0, triple, n_fuel));
    t.xe135  = const_cast<double*>(xeRow(base, 1, triple, n_fuel));
    t.xe135m = const_cast<double*>(xeRow(base, 2, triple, n_fuel));
    return t;
}

__device__ __host__ inline xek::XeTripleConst xeTripleConstAt(const double* base,
                                                              int triple, int n_fuel) {
    xek::XeTripleConst t;
    t.i135   = xeRow(base, 0, triple, n_fuel);
    t.xe135  = xeRow(base, 1, triple, n_fuel);
    t.xe135m = xeRow(base, 2, triple, n_fuel);
    return t;
}

/// KERNEL 1 of 3: the Xe rate/dot evaluation.  One thread per fuel ordinal;
/// x, F(x) and g land in the device history and nothing else is written.
///
/// The residual reduces through the same 64-bit atomicMax kernelXsRecon uses:
/// the metric is |dXe| / max(|Xe|, 1e-30) and therefore never negative, and for
/// non-negative IEEE doubles the unsigned order of the bit patterns IS the value
/// order -- so the reduction is exact and order-insensitive, which is what makes
/// it say the same thing whatever order the blocks retire in.
__global__ void kXeEvaluate(xsr::BatchView v, double* hist, unsigned char* processed,
                            unsigned long long* max_bits) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= v.n_fuel) return;

    const xek::XeTriple x = xeTripleAt(hist, xek::XE_T_X, v.n_fuel);
    const xek::XeTriple f = xeTripleAt(hist, xek::XE_T_F, v.n_fuel);
    const xek::XeTriple g = xeTripleAt(hist, xek::XE_T_G, v.n_fuel);

    double change = 0.0;
    xek::xeEvaluateOrdinal(v, k, x, f, g, processed, &change);
    atomicMax(max_bits, static_cast<unsigned long long>(__double_as_longlong(change)));
}

/// out = a - b over the three rows: the difference columns of the window, and
/// the f_prev/g_prev save, expressed once.
__global__ void kXeSub(const double* hist, int ta, int tb, int tout, int n_fuel) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_fuel) return;
    xek::xeSubOrdinal(xeTripleConstAt(hist, ta, n_fuel), xeTripleConstAt(hist, tb, n_fuel),
                      xeTripleAt(const_cast<double*>(hist), tout, n_fuel), k);
}

/// KERNEL 3 of 3, stage 1: k_xe_dot_reduce.  One thread owns ONE PARTITION of
/// one pair, start to finish, so the accumulation inside a partition is serial
/// and in ascending ordinal order -- the host's own order.  The partition
/// boundaries come from xeDotPartitionRange, which reads only (n_fuel, parts):
/// no launch shape, no occupancy, no arrival order.
///
/// This is NOT a warp-per-slot serial fold.  Thirty-two lanes splitting ~15,000
/// terms would put the association in the hands of whoever wrote the lane
/// mapping and would still not be the host's; a fixed partition at least makes
/// the difference a stated, reproducible one.
__global__ void kXeDotStage1(const double* hist, int n_fuel, int parts, int npairs,
                             const int* pairs, double* partials,
                             unsigned long long forms) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npairs * parts) return;
    const int pair = t / parts;
    const int part = t - pair * parts;

    const xek::XeTripleConst a = xeTripleConstAt(hist, pairs[2 * pair], n_fuel);
    const xek::XeTripleConst b = xeTripleConstAt(hist, pairs[2 * pair + 1], n_fuel);

    int i0 = 0, i1 = 0;
    xek::xeDotPartitionRange(n_fuel, parts, part, &i0, &i1);
    partials[static_cast<long long>(pair) * parts + part] =
        xek::xeDotChunk(a, b, i0, i1, forms);
}

/// Stage 2: one thread per pair, strict serial fold in ascending partition
/// order.  A tree here would hand the association back to the scheduler, which
/// is the whole thing the fixed partition takes out of its hands.
__global__ void kXeDotStage2(int parts, int npairs, const double* partials,
                             const int* slots, double* out) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= npairs) return;
    out[slots[pair]] =
        xek::xeDotFold(partials + static_cast<long long>(pair) * parts, parts);
}

/// KERNEL 2a of 3: the Anderson candidate.  One thread per fuel ordinal.
__global__ void kXeCandidate(double* hist, int n_fuel, int ncol, double g0, double g1,
                             unsigned long long forms, int* physics_bad,
                             unsigned long long* step_bits) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_fuel) return;

    const double gamma[2] = {g0, g1};
    int          bad      = 0;
    double       step     = 0.0;
    xek::xeCandidateOrdinal(xeTripleConstAt(hist, xek::XE_T_F, n_fuel),
                            xeTripleConstAt(hist, xek::XE_T_X, n_fuel),
                            xeRow(hist, 0, xek::XE_T_DF0, n_fuel),
                            xeRow(hist, 1, xek::XE_T_DF0, n_fuel),
                            xeRow(hist, 2, xek::XE_T_DF0, n_fuel), ncol, gamma,
                            xeTripleAt(hist, xek::XE_T_CAND, n_fuel), k, n_fuel, forms,
                            &bad, &step);
    if (bad) atomicOr(physics_bad, 1);
    atomicMax(step_bits, static_cast<unsigned long long>(__double_as_longlong(step)));
}

/// KERNEL 2b of 3: the Picard/Anderson update -- write the three Xe-chain rows
/// and reconstruct the node.
///
/// `picard_skip` is the difference between the two host functions this one
/// stands in for.  UpdateEquilibriumXenon's loop `continue`s on a node whose
/// normalized flux is not positive: it writes nothing and reconstructs nothing.
/// CommitXenon reconstructs every fuel node it is handed, because the caller
/// chose that inventory for all of them.  Getting this wrong is not a rounding
/// -- it is a node reconstructed that the host left alone.
__global__ void kXeCommit(xsr::BatchView v, const double* hist, int triple, double relax,
                          int picard_skip, const unsigned char* processed,
                          unsigned long long* solved) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= v.n_fuel) return;
    if (picard_skip && processed[k] == 0u) return;

    double vi = 0.0, vx = 0.0, vm = 0.0;
    if (picard_skip) {
        xek::xeBlendOrdinal(xeTripleConstAt(hist, triple, v.n_fuel),
                            xeTripleConstAt(hist, xek::XE_T_X, v.n_fuel), relax, k, &vi,
                            &vx, &vm);
    } else {
        const xek::XeTripleConst t = xeTripleConstAt(hist, triple, v.n_fuel);
        vi                         = t.i135[k];
        vx                         = t.xe135[k];
        vm                         = t.xe135m[k];
    }
    xek::xeCommitOrdinal(v, k, vi, vx, vm);
    atomicAdd(solved, 1ULL);
}

// ---------------------------------------------------------------------------
// WP7 stage C -- the Xe device transaction (RASBERY_GPU_XE_TXN, default off)
// ---------------------------------------------------------------------------
//
// The five kernels above stay exactly as they are and are what TXN=0 launches,
// so the B0 replay compares against LIVE CODE and not a memory of it -- the same
// rule WP7-B's fusion bits follow.  What is added here is the same work with
// the three host decisions moved inside.

/// FUSED HISTORY MAINTENANCE.  Replaces, in one launch, what was two kXeSub
/// launches and twelve device-to-device copies: xeRotateHistory (6 copies),
/// xeRecordColumn (2 kernels) and xeSaveEvaluation (6 copies).
///
/// ORDER-PRESERVATION NOTE (WP7-B's rule, applied here).  Every operation is
/// ELEMENTWISE at one fuel ordinal and one row: no reduction, no neighbour, no
/// accumulation, so ordinal k's outputs depend on ordinal k's inputs alone and
/// the grid may retire in any order whatever.  The sequence that DOES matter is
/// the one inside a single ordinal -- the record must read F_prev before the
/// save overwrites it, and the rotate must read df[1] before the record
/// overwrites it -- and xeHistoryOrdinal loads all four operands into registers
/// before it stores anything.  No arithmetic is added, removed or reassociated:
/// two unfused subtractions and four copies, which is what the twelve
/// cudaMemcpyAsync and two kXeSub did.
__global__ void kXeHistory(double* hist, int n_fuel, int col, int rotate) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_fuel) return;
    for (int row = 0; row < 3; ++row) {
        xek::xeHistoryOrdinal(
            xeRow(hist, row, xek::XE_T_F, n_fuel),
            xeRow(hist, row, xek::XE_T_F_PREV, n_fuel),
            xeRow(hist, row, xek::XE_T_G, n_fuel),
            xeRow(hist, row, xek::XE_T_G_PREV, n_fuel),
            const_cast<double*>(xeRow(hist, row, xek::XE_T_DF0, n_fuel)),
            const_cast<double*>(xeRow(hist, row, xek::XE_T_DF1, n_fuel)),
            const_cast<double*>(xeRow(hist, row, xek::XE_T_DG0, n_fuel)),
            const_cast<double*>(xeRow(hist, row, xek::XE_T_DG1, n_fuel)),
            const_cast<double*>(xeRow(hist, row, xek::XE_T_F_PREV, n_fuel)),
            const_cast<double*>(xeRow(hist, row, xek::XE_T_G_PREV, n_fuel)), k, col,
            rotate);
    }
}

/// ONE THREAD, AND THAT IS THE POINT.  Eight doubles of arithmetic that used to
/// cost a stream synchronisation now cost a launch.  It consumes `dots` in
/// XeDotSlot order -- the same order kXeDotStage2 wrote them and the same order
/// the host read them -- and the constants arrive as arguments because Driver.h
/// owns them.
///
/// It also clears the candidate kernel's two reduction cells, which is what
/// removes the two cudaMemsetAsync nodes the host arm needed there.
__global__ void kXeAndersonSolve(const double* dots, int ncol,
                                 const unsigned long long* picard_bits, double eq_tol,
                                 double min_gram, unsigned long long forms,
                                 xek::XeTxnControl* ctl, int* physics_bad,
                                 unsigned long long* step_bits) {
    double picard = __longlong_as_double(static_cast<long long>(*picard_bits));
    xek::xeAndersonSolveControl(dots, ncol, picard, eq_tol, min_gram, forms, ctl);
    *physics_bad = 0;
    *step_bits   = 0ull;
}

/// kXeCandidate with gamma read from device memory and the launch predicated on
/// the fit having conditioned.  The body is xeCandidateOrdinal, unchanged: a
/// candidate built here is bit-for-bit the candidate TXN=0 builds, because the
/// gammas it is built from are bit-for-bit the same gammas.
__global__ void kXeCandidateTxn(double* hist, int n_fuel, const xek::XeTxnControl* ctl,
                                unsigned long long forms, int* physics_bad,
                                unsigned long long* step_bits) {
    if (!ctl->solved) return;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_fuel) return;

    const double gamma[2] = {ctl->gamma[0], ctl->gamma[1]};
    int          bad      = 0;
    double       step     = 0.0;
    xek::xeCandidateOrdinal(xeTripleConstAt(hist, xek::XE_T_F, n_fuel),
                            xeTripleConstAt(hist, xek::XE_T_X, n_fuel),
                            xeRow(hist, 0, xek::XE_T_DF0, n_fuel),
                            xeRow(hist, 1, xek::XE_T_DF0, n_fuel),
                            xeRow(hist, 2, xek::XE_T_DF0, n_fuel), ctl->ncol, gamma,
                            xeTripleAt(hist, xek::XE_T_CAND, n_fuel), k, n_fuel, forms,
                            &bad, &step);
    if (bad) atomicOr(physics_bad, 1);
    atomicMax(step_bits, static_cast<unsigned long long>(__double_as_longlong(step)));
}

/// SAFEGUARDS 3/4 and 4/4, one thread, after the candidate grid has retired.
/// A separate kernel and not a prologue inside kXeCommitTxn because a grid-wide
/// OR and MAX are not readable until the grid that wrote them is done, and the
/// kernel boundary is the only thing that says so.
__global__ void kXeAndersonGate(xek::XeTxnControl* ctl, const int* physics_bad,
                                const unsigned long long* step_bits, double max_step) {
    const double step =
        __longlong_as_double(static_cast<long long>(*step_bits));
    xek::xeAndersonGateControl(*physics_bad, step, max_step, ctl);
}

/// kXeCommit with `triple` and `picard_skip` read from the control block rather
/// than passed from the host.
///
///   accept == 1   commit XE_T_CAND over EVERY fuel node, exactly as the host
///                 arm's accepted commit does (CommitXenon's contract).
///   accept == 0   commit x + relax*(F - x) and SKIP the zero-flux nodes,
///                 exactly as the Picard fallback does
///                 (UpdateEquilibriumXenon's `continue`).
///
/// relax is 1.0 on this path and the host asserts it; the parameter is here so
/// the kernel does not silently bake an invariant it does not own.
__global__ void kXeCommitTxn(xsr::BatchView v, const double* hist,
                             const xek::XeTxnControl* ctl, double relax,
                             const unsigned char* processed,
                             unsigned long long* solved) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= v.n_fuel) return;
    const int accept = ctl->accept;
    if (!accept && processed[k] == 0u) return;

    double vi = 0.0, vx = 0.0, vm = 0.0;
    if (accept) {
        const xek::XeTripleConst c = xeTripleConstAt(hist, xek::XE_T_CAND, v.n_fuel);
        vi                         = c.i135[k];
        vx                         = c.xe135[k];
        vm                         = c.xe135m[k];
    } else {
        xek::xeBlendOrdinal(xeTripleConstAt(hist, xek::XE_T_F, v.n_fuel),
                            xeTripleConstAt(hist, xek::XE_T_X, v.n_fuel), relax, k, &vi,
                            &vx, &vm);
    }
    xek::xeCommitOrdinal(v, k, vi, vx, vm);
    atomicAdd(solved, 1ULL);
}

namespace fxs = rasbery::flatxs;

std::atomic<unsigned long long> g_flatxs_nodes_solved{0};

// One thread per target node; the shared body does the rest.  StaticForms
// folds the mined mask at compile time (this TU builds with --fmad=false, so
// only the explicit fma() arms fuse).
__global__ void kernelFlatXs(fxs::FlatXsView v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.n_nodes) return;
    fxs::flatxsSolveNode(v, i, fxs::StaticForms{});
}

namespace ndl = rasbery::nodal;

std::atomic<unsigned long long> g_nodal_drives{0};
/// Rev.7.1 Task 18-lite receipt: bytes the canonical binding kept off the bus.
///
/// PROCESS-WIDE AND NOT PER-INSTANCE, for the same reason g_nodal_drives is: the
/// number a reader wants is what the RUN did, and the receipt is printed by
/// main() long after any particular backend went away.
std::atomic<unsigned long long> g_canon_up_bytes{0};
std::atomic<unsigned long long> g_canon_down_bytes{0};
std::atomic<unsigned long long> g_nodal_graph_launches{0};
std::atomic<unsigned long long> g_nodal_graph_fallbacks{0};
/// Rev.7.1 Task 10 part 4: how many times the drive had to CAPTURE, as opposed
/// to launch what it had captured before.
///
/// COUNTED BECAUSE IT WAS INVISIBLE.  Every re-capture costs a
/// cudaStreamSynchronize on the backend's own stream (the drain that has to
/// leave an idle stream for cudaStreamBeginCapture), a capture, and an
/// instantiate -- and NONE of that lands in the segment's
/// `in_body_host_syncs`, which counts the runner's own drains.  A host-free
/// segment can therefore read `in_body_host_syncs: 0` and still rendezvous once
/// per segment inside the nodal backend.  `graph_launches - graph_captures` is
/// the number the caching is about; on a run where the key is stable it should
/// be `launches - 1`.
std::atomic<unsigned long long> g_nodal_graph_captures{0};
std::atomic<unsigned long long> g_nodal_d2h_bytes{0}; // per drive, last shape seen
/// Rev.7.1 W3 item 2: drives whose terminal drain became an event on the
/// segment's stream instead of a host block.  A deferral that never happened is
/// the same receipt as a feature that is not there, which is why it is counted.
std::atomic<unsigned long long> g_nodal_drains_deferred{0};
/// Rev.7.1 W3 item 3: drives that read 1/eigv out of a device slot somebody else
/// wrote, instead of uploading the host's copy.  Counted for the same reason:
/// an elision that never engages is indistinguishable from a feature that is not
/// wired, and this one is wired through four files.
std::atomic<unsigned long long> g_nodal_reigv_device{0};

// --- batch arena receipts (see NodalArena below).  Kept as TU-scope atomics,
// not arena members, so the static-destruction receipt never has to reason
// about whether the arena object is still alive.
std::atomic<int>                g_nodal_batch_slots{0};
std::atomic<unsigned long long> g_nodal_batches{0};        // batches launched
std::atomic<unsigned long long> g_nodal_batch_drives{0};   // sum of participants
std::atomic<unsigned long long> g_nodal_batch_graph_launches{0};
std::atomic<unsigned long long> g_nodal_batch_graph_fallbacks{0};

// --- Rev.7.1 Task 8 compaction receipt (Sec 9.3) --------------------------
//
// The point of compaction is the RATIO, so all three are counted: how much work
// there was (logical_drives / physical_slot_blocks) and how much grid was
// launched to do nothing (padding_blocks).  Reporting drives alone would hide
// the effect entirely -- the drive count is identical either way.
std::atomic<unsigned long long> g_nodal_logical_drives{0};
std::atomic<unsigned long long> g_nodal_physical_blocks{0};
std::atomic<unsigned long long> g_nodal_padding_blocks{0};
/// Distinct (bucket, fusion, geometry) graph instantiations.  With the
/// coarse ladder this saturates at a handful; a number that keeps climbing
/// means the key is churning, which is the Task 10 instantiation gate.
std::atomic<unsigned long long> g_nodal_bucket_graphs{0};
/// Nine buckets, indexed by the ladder position (1,2,4,8,16,24,32,48,64).
std::array<std::atomic<unsigned long long>, 9> g_nodal_bucket_histogram{};

inline int nodalBucketIndex(int lanes) {
    static const int kBuckets[9] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
    for (int i = 0; i < 9; ++i)
        if (lanes <= kBuckets[i]) return i;
    return 8;
}
inline void nodalBucketHistogramBump(int lanes) {
    g_nodal_bucket_histogram[static_cast<std::size_t>(nodalBucketIndex(lanes))].fetch_add(
        1, std::memory_order_relaxed);
}

/// RASBERY_GPU_NODAL_COMPACT, default OFF.  Read once, like every other gate.
inline bool nodalCompactEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_NODAL_COMPACT");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}
std::atomic<unsigned long long> g_nodal_batch_fallbacks{0}; // -> per-instance arm
std::atomic<unsigned long long> g_nodal_batch_refused{0};   // slot/geometry refusals
std::atomic<unsigned long long> g_nodal_batch_hist[65]     = {}; // width -> count
std::atomic<unsigned long long> g_nodal_batch_linger_us{0};
std::atomic<unsigned long long> g_nodal_batch_xs_h2d_bytes{0};
std::atomic<unsigned long long> g_nodal_batch_xs_h2d_skipped_bytes{0};

// The two group-SEPARABLE phases run one thread per (node, group): trlcff0 and
// trlcff12 carry nothing across the ig loop (see NodalKernel.h), so lk*NG+ig is
// the smallest real work unit and the launch gets 2x the threads at NG=2.
// Adjacent threads then differ only in ig, and every array they touch is
// [...*NG + ig], so the doubled thread count also coalesces better than the
// thread-per-node form did.  updateMatrix/calculateEven stay thread-per-node:
// their 2x2 group coupling (matM/matMI inversion, the a[2][2] solve) is
// inherent and cannot be split without changing the arithmetic.
//
// BATCH AXIS.  Every one of the five is templated on BATCHED:
//
//  - BATCHED=false is the per-instance launch: grid.y is 1, `active` is null,
//    and the body is textually what it always was -- `base` is used verbatim,
//    nodalSlotView is not even instantiated on that path.  Adding the batch
//    axis therefore cannot perturb the single-instance arm at all, not even by
//    an extra integer add.
//
//  - BATCHED=true is the arena launch: grid.y is the DISPATCH WIDTH and
//    blockIdx.y is a LOGICAL lane, not a slot.  `slot_map` turns the lane into a
//    physical slot; the FIRST instruction is the padding guard -- the W1 rule
//    (gpuDispatchIsPadding), one uniform predicate per block, evaluated before
//    anything is read or written.  gridDim.x chunking is untouched, so lk / ig /
//    ls are computed exactly as in the per-instance launch; only the base
//    pointers move (see nodal::nodalSlotView).
//
// WHY A MAP AND NOT blockIdx.y DIRECTLY (Rev.7.1 Task 8).  Before compaction,
// grid.y was the FLEET width and every non-participating slot still launched a
// block that read the participation mask and returned.  At 64 slots with 3 in
// the batch that is 61/64 of every grid doing nothing but a load and a branch --
// and the nodal phases launch six of those per drive.  With the map, grid.y is
// the smallest bucket covering the participants, so the blocks that are launched
// are (nearly) all real work.
//
// THE MAP IS THE ONLY WAY A LANE LEARNS ITS SLOT.  Nothing below may use
// blockIdx.y as a slot index: it is correct only when the map happens to be the
// identity, which is exactly the compaction-off case, so the bug would pass
// every test that did not turn compaction on.
//
// NO __syncthreads ANYWHERE IN THESE KERNELS, which is what makes an early
// `return` from a subset of blocks safe.  The contract test asserts it rather
// than trusting it, because adding one later would turn this guard into a
// deadlock.
//
// ---------------------------------------------------------------------------
// PER-SLOT POINTER TABLE (Rev.7.1 pre-W3), replacing the dense rebase
// ---------------------------------------------------------------------------
//
// A slot's view used to be COMPUTED: nodalSlotView(base, m) advanced each array
// pointer by that array's own dense per-slot count (`jnet += m*nsurf*NG`).  That
// works only while every array is its own contiguous slot-major block, which is
// true of the arena's private allocation and false of GpuPhysicsArena, where a
// slot is one 29.4-million-double stride covering all of its arrays.  Feeding a
// canonical pointer to the dense rebase would have indexed it with the wrong
// stride -- slot 1 reading inside slot 0 -- and every value would still have
// been finite and plausible.
//
// So the view is now LOOKED UP: `views[m]` is slot m's fully-resolved
// NodalView, built once on the host.  The kernel does not know or care how the
// pointers were laid out, which is exactly what lets one slot borrow canonical
// buffers while another keeps the arena's dense ones (mixed mode).
//
// THIS CHANGES NO ARITHMETIC.  nodalSlotView is still the builder for a legacy
// slot, so a default table holds byte-for-byte the addresses the rebase
// computed; the phase bodies are untouched and the replay gates score the same
// functions on the same values.  Indirection only.
#define RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v)              \
    int _m = 0;                                                                \
    if (BATCHED) {                                                             \
        const int _logical = static_cast<int>(blockIdx.y);                     \
        if (_logical >= (lanes)) return;                                       \
        _m = (slot_map)[_logical];                                             \
        if (_m < 0) return;                                                    \
    }                                                                          \
    const ndl::NodalView v =                                                   \
        !BATCHED                 ? (base)                                      \
        : ((views) != nullptr)   ? (views)[_m]                                 \
                                 : ndl::nodalSlotView(base, _m);               \
    /* Rev.7.1 Task 10 part 3: the segment's halt, in the one place every    */ \
    /* nodal phase already passes through.  Null on every arm but a          */ \
    /* host-free device outer segment, so this is a predicted branch over a  */ \
    /* baked null for the whole rest of the tree.  See NodalView::halt for   */ \
    /* why the drive may not simply be re-run past a decided exit.           */ \
    if ((v).halt != nullptr && (v).halt[(v).halt_slot] != 0u) return

template <bool BATCHED>
__global__ void kNodalTrl0(ndl::NodalView base, const int* __restrict__ slot_map,
                             int lanes, const ndl::NodalView* __restrict__ views) {
    RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v);
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lk = i / ndl::NG;
    const int ig = i - lk * ndl::NG;
    if (lk < v.nxyz) ndl::nodalTrlcff0Group(v, lk, ig);
}
template <bool BATCHED>
__global__ void kNodalTrl12(ndl::NodalView base, const int* __restrict__ slot_map,
                             int lanes, const ndl::NodalView* __restrict__ views) {
    RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v);
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lk = i / ndl::NG;
    const int ig = i - lk * ndl::NG;
    if (lk < v.nxyz) ndl::nodalTrlcff12Group(v, lk, ig, ndl::StaticForms{});
}
template <bool BATCHED>
__global__ void kNodalMat(ndl::NodalView base, const int* __restrict__ slot_map,
                             int lanes, const ndl::NodalView* __restrict__ views) {
    RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalUpdateMatrix(v, lk, ndl::StaticForms{});
}
template <bool BATCHED>
__global__ void kNodalEven(ndl::NodalView base, const int* __restrict__ slot_map,
                             int lanes, const ndl::NodalView* __restrict__ views) {
    RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) ndl::nodalCalculateEven(v, lk, ndl::StaticForms{});
}

template <bool BATCHED>
__global__ void kNodalMatEven(ndl::NodalView base, const int* __restrict__ slot_map,
                              int lanes, const ndl::NodalView* __restrict__ views) {
    RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v);
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk >= v.nxyz) return;
    const ndl::StaticForms forms{};
    ndl::nodalUpdateMatrix(v, lk, forms);
    ndl::nodalCalculateEven(v, lk, forms);
}
template <bool BATCHED>
__global__ void kNodalJnet(ndl::NodalView base, const int* __restrict__ slot_map,
                             int lanes, const ndl::NodalView* __restrict__ views) {
    RASBERY_NODAL_SLOT_GUARD(base, slot_map, lanes, views, v);
    const int ls = blockIdx.x * blockDim.x + threadIdx.x;
    if (ls < v.nsurf) ndl::nodalCalculateJnet(v, ls, ndl::StaticForms{});
}

// The 9 ACTIVE_XT slots of XSSet.cpp, as Chiffon::XSTYPE values.  The enum
// order is pinned by the static_assert-style constants in XsReconKernel.h
// (T_XSTF..T_XS3N); XSDF and XSRF are derived and skipped.
constexpr int ACTIVE_XT9[fxs::N_ACTIVE] = {
    xsr::T_XSTF, xsr::T_XSAF, xsr::T_XSFF, xsr::T_XSNF, xsr::T_XSKF,
    xsr::T_XSSF, xsr::T_FYLD, xsr::T_XS2N, xsr::T_XS3N};

std::uint64_t fnvMix(const void* p, std::size_t bytes, std::uint64_t h) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < bytes; ++i)
        h = (h ^ c[i]) * 1099511628211ULL;
    return h;
}

/// Process-wide device copies of the flat coefficient tables, shared across
/// the batch instances (the tables are immutable after library load, and the
/// 64 instances of one benchmark load the same library).  Keyed by a full
/// FNV-1a over sizes and bytes; the uploader copies synchronously while
/// holding the mutex, so a concurrent instance can never launch against a
/// half-uploaded table.
struct FlatXsLibDevice {
    std::uint64_t       hash = 0;
    double*             block = nullptr; // [9 lmp | lsm | 9 mic | msm | knots]
    fxs::DeltaMeta*     deltas = nullptr;
    std::size_t         off_lmp[fxs::N_ACTIVE] = {};
    std::size_t         off_lsm = 0;
    std::size_t         off_mic[fxs::N_ACTIVE] = {};
    std::size_t         off_msm = 0;
    std::size_t         off_knots = 0;
};

std::mutex g_flatxs_lib_mutex;
/// std::deque, NOT std::vector: `Impl::lib` caches the address of an element
/// and is dereferenced later WITHOUT the mutex (solveFlatXs builds the kernel
/// view from it).  A vector's push_back can reallocate, which would dangle
/// every already-cached pointer the moment a second distinct library appeared.
/// deque never invalidates references to existing elements on push_back, so the
/// cached address stays valid for the process lifetime.  Single-library runs --
/// every run today -- behave identically.
std::deque<FlatXsLibDevice>* g_flatxs_libs = nullptr; // leaked on purpose (process lifetime)

// ===========================================================================
// NodalArena -- the multi-instance nodal arena.
//
// WHY.  In M64 batch mode the per-instance nodal device arm is correct but far
// too light to matter: 8451 nodes x 2 groups is 67-209 blocks, 3-9% occupancy
// on a 188-SM device, and the 64 instances take turns at it.  Host nodal work
// was measured at 36.5% of all thread-time (18 ms/outer/instance) and it is
// what manufactures the ~6 ms arrival skew that holds the CMFD rendezvous down
// to mean width 17/64.  Aggregating the SAME five kernels across instances --
// 64 x 8451 x 2 = 1.08M work items, ~4 full waves -- is the only way this arm
// fills the device, which is why it is an arena and not a wider block size.
//
// STRUCTURE.  Deliberately the same recipe as the CMFD CudaBatchArena
// (CudaBICGBackend.cu:1297-2530), because that one is proven in production:
//
//   * slot-strided allocations, one block per array, stride = the array's
//     single-instance element count (nodal::nodalSlotView is the rebase);
//   * gridDim.y == slots with blockIdx.y == m, and a per-slot participation
//     mask read as the kernels' first instruction (the HALT_GUARD shape);
//   * ONE fixed-topology graph that serves every subset -- participation lives
//     in device memory, not in the launch shape, so the graph is captured once
//     and replayed for every batch whatever its width;
//   * opportunistic rendezvous: a batch is whoever arrived while the last one
//     was on the device, with a bounded adaptive linger;
//   * a single elected launcher (`_launching`), because two launchers on one
//     stream corrupt captures -- see the comment at the election site;
//   * fail-open everywhere: any refusal or failure drops that instance back on
//     the per-instance FULL path, which drops back to the CPU body.
//
// WHAT STAYS PER INSTANCE.  Everything above the five phases: updateConstant
// (host, shadow-checked, the only transcendental), convergence tests, the
// critical search, TH.  Instances arrive, hand over five phases' worth of
// arithmetic, and go back to their own control flow.  A slot's answer does not
// depend on who else rode along -- that is what makes the opportunistic batch
// legitimate, and it is the same argument the CMFD arena rests on.
// ===========================================================================

class NodalArena {
public:
    NodalArena(const ndl::NodalView& proto, int slots) : _slots(slots > 0 ? slots : 1) {
        if (const char* w = std::getenv("RASBERY_NODAL_BATCH_WAIT_US")) {
            const std::string requested(w);
            if (requested == "auto" || requested == "AUTO" || requested == "adaptive")
                _wait_auto = true;
            else {
                const long parsed = std::atol(w);
                if (parsed >= 0) { _wait_us = parsed; _wait_auto = false; }
            }
        }
        if (const char* mx = std::getenv("RASBERY_NODAL_BATCH_WAIT_MAX_US")) {
            const long parsed = std::atol(mx);
            if (parsed >= 0) _wait_max_us = std::min(parsed, 20000L);
        }
        _use_graph     = !envFlagDisabled("RASBERY_GPU_NODAL_GRAPH");
        _fuse_mat_even = nodalFuseMatEvenEnabled();
        _mirror_xs     = nodalXsMirrorEnabled();
        init(proto);
    }

    // Never destroyed in practice: the arena is a process-lifetime singleton
    // and tearing its device allocations down during static destruction would
    // race the CUDA runtime's own teardown.  Same policy as g_flatxs_libs.
    NodalArena(const NodalArena&)            = delete;
    NodalArena& operator=(const NodalArena&) = delete;

    [[nodiscard]] bool               available() const { return _available; }
    [[nodiscard]] const std::string& status() const { return _status; }
    [[nodiscard]] int                slots() const { return _slots; }

    /// Byte-exact compatibility of one instance's view with the arena's SHARED
    /// immutable geometry.  Shape counts alone are not enough -- two decks can
    /// agree on nxyz/nsurf and still have different neighbour maps, and sharing
    /// the first map would produce physically wrong currents with no error --
    /// so this is the full memcmp, exactly as CudaBatchArena::compatibleGeometry
    /// walks the CMFD neighbour table.
    [[nodiscard]] bool compatible(const ndl::NodalView& p) const {
        if (p.nxyz != _nxyz || p.nsurf != _nsurf || p.chif_empty != _chif_empty)
            return false;
        const std::size_t nx = static_cast<std::size_t>(_nxyz);
        const std::size_t ns = static_cast<std::size_t>(_nsurf);
        return std::memcmp(_ref_hmesh.data(), p.hmesh, nx * ndl::NDIR * sizeof(double)) == 0 &&
               std::memcmp(_ref_albedo.data(), p.albedo,
                           ndl::NDIR * ndl::NLR * sizeof(double)) == 0 &&
               std::memcmp(_ref_lktosfc.data(), p.lktosfc,
                           nx * ndl::NDIR * ndl::NLR * sizeof(int)) == 0 &&
               std::memcmp(_ref_neib.data(), p.neib, nx * ndl::NEWSB * sizeof(int)) == 0 &&
               std::memcmp(_ref_lklr.data(), p.lklr, ns * ndl::NLR * sizeof(int)) == 0 &&
               std::memcmp(_ref_idirlr.data(), p.idirlr, ns * ndl::NLR * sizeof(int)) == 0 &&
               std::memcmp(_ref_sgnlr.data(), p.sgnlr, ns * ndl::NLR * sizeof(int)) == 0;
    }

    int acquireSlot(const ndl::NodalView& p) {
        if (!_available || !compatible(p)) return -1;
        std::lock_guard<std::mutex> lock(_mutex);
        for (int m = 0; m < _slots; ++m) {
            if (_slot[static_cast<std::size_t>(m)].in_use) continue;
            Slot& sl = _slot[static_cast<std::size_t>(m)];
            // A fresh tenant inherits nothing: the previous tenant's residency
            // flags would elide uploads the new one needs.
            sl              = Slot{};
            // Rev.7.1 Task 18: AND NOT THE PREVIOUS TENANT'S CANONICAL BINDING
            // EITHER.  _canon lives outside Slot because the view table is
            // indexed by physical slot, so the `sl = Slot{}` above does not
            // reach it -- and a borrowed pointer into a dead deck's physics
            // arena is the worst thing this slot could be left holding: the
            // kernels would read and write another deck's jnet, every value
            // finite and plausible.  The new tenant re-adopts below, in
            // solveNodal, at the moment it learns which slot it got.
            _canon[static_cast<std::size_t>(m)] = gpu::CanonicalSlotBuffers{};
            _views_dirty                        = true;
            // Rev.7.1 Task 20 (plan Sec 3.2 / 8.2).  The post-condition of the
            // two resets above, checked rather than assumed.  This arena is the
            // one that has ALREADY been bitten by out-of-struct per-slot state:
            // `_canon` lives outside Slot because the view table is indexed by
            // physical slot, so `sl = Slot{}` does not reach it, and a borrowed
            // pointer into a dead deck's physics arena left there would have the
            // kernels read and write another deck's jnet with every value
            // finite.  The audit is what notices if a third such field appears.
            if (!nodalSlotIsReset(sl, _canon[static_cast<std::size_t>(m)]))
                rasbery::refill::tenancy().stale_tenants.fetch_add(1, std::memory_order_relaxed);
            sl.in_use = true;
            rasbery::refill::tenancy().admissions.fetch_add(1, std::memory_order_relaxed);
            return m;
        }
        return -1;
    }

    void releaseSlot(int m) {
        if (m < 0) return;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            Slot& sl  = _slot[static_cast<std::size_t>(m)];
            // Rev.7.1 Task 20: see CudaBatchArena::releaseSlot.  A release of a
            // slot nobody held means the next acquire could hand it to two.
            if (!sl.in_use)
                rasbery::refill::tenancy().double_releases.fetch_add(1, std::memory_order_relaxed);
            sl.in_use = false;
            // Drop the pin memo with the tenant.  These are NOT leases -- the
            // lease belongs to the buffer's owner (Geometry/XSSet), which
            // releases it in its own destructor -- they are only "have I
            // already asked pinHost about this pointer".  A memo that outlived
            // the tenant would let the next one skip a pinHost call for an
            // address that merely happens to match a dead deck's, so it is
            // cleared here as well as in acquireSlot's full Slot{} reset.
            for (int i = 0; i < 6; ++i) sl.pin_bulk[i] = nullptr;
            for (int i = 0; i < 9; ++i) sl.pin_const[i] = nullptr;
            sl.pin_chif = nullptr;
            // The borrowed canonical pointers belong to the departing tenant's
            // physics arena; nothing here may keep them (see acquireSlot).
            _canon[static_cast<std::size_t>(m)] = gpu::CanonicalSlotBuffers{};
            _views_dirty                        = true;
        }
        // A lingering launcher may still be waiting for this slot to show up.
        // It never will, and inUseCount() has just dropped, so wake it.
        _cv.notify_all();
    }

    /// One batched nodal drive for slot `m`.  Blocks until the batch that
    /// carried this slot has finished and its jnet/phis have landed in the
    /// caller's host arrays.  false = the caller must run its own arm.
    /// No `state_generation`: see the xs upload in launchBatch for why the
    /// arena cannot use it as a residency key.
    bool drive(int m, const ndl::NodalView& host, unsigned long long const_gen,
               unsigned long long ref_gen, const gpu::CanonicalSlotState& canon,
               std::uint32_t materialize) {
        if (!_available) return false;

        // ---- stage.  No CUDA call here: every instance thread runs this
        // concurrently, and the launcher is the only one allowed near the
        // stream (CudaBatchArena::stageSlot has the same contract).
        {
            Slot& sl     = _slot[static_cast<std::size_t>(m)];
            // Rev.7.1 Task 18: THE OWNERSHIP IS STAGED, NOT READ AT LAUNCH.
            //
            // The launcher is one of the M instance threads and it decides the
            // elisions for every participant, so it would otherwise be reading
            // another thread's XsReconBackend::Impl while that thread sits in
            // the rendezvous.  Copying the two words each participant needs --
            // the three canonical owners and the materialize mask -- under the
            // same "no CUDA call here" staging contract keeps the launcher's
            // reads inside the arena.
            for (int r = 0; r < gpu::kCanonicalNodalRegionCount; ++r)
                sl.canon_owner[r] =
                    canon.ownerOf(gpu::kCanonicalNodalRegions[r]);
            sl.canon_materialize = materialize;
            sl.h_jnet    = host.jnet;
            sl.h_flux    = host.flux;
            sl.h_phis    = host.phis;
            sl.h_xsrf    = host.xsrf;
            sl.h_xsnf    = host.xsnf;
            sl.h_xssm    = host.xssm;
            sl.h_chif    = host.chif;
            sl.h_const[0] = host.eta1;   sl.h_const[1] = host.eta2;
            sl.h_const[2] = host.m260;   sl.h_const[3] = host.m251;
            sl.h_const[4] = host.m253;   sl.h_const[5] = host.m262;
            sl.h_const[6] = host.m264;   sl.h_const[7] = host.diagD;
            sl.h_const[8] = host.diagDI;
            sl.reigv     = host.reigv;
            sl.const_gen = const_gen;
            sl.ref_gen   = ref_gen;
        }

        std::unique_lock<std::mutex> lock(_mutex);
        const auto now = std::chrono::steady_clock::now();
        if (_have_arrival) {
            const double gap_us = static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(now - _last_arrival).count());
            if (gap_us >= 0.0 && gap_us <= 1.0e6)
                _arrival_gap_ewma_us = _arrival_gap_ewma_us == 0.0
                                           ? gap_us
                                           : 0.8 * _arrival_gap_ewma_us + 0.2 * gap_us;
        }
        _last_arrival = now;
        _have_arrival = true;

        const unsigned long long my_batch = _open_batch;
        // Rev.7.1 Task 20: see CudaBatchArena::solveCommon.  The nodal arena has
        // the same one-owner rule and the same failure if it breaks -- the
        // launcher would stage this slot twice and the second stage would
        // overwrite the first participant's xs.
        if (std::find(_pending.begin(), _pending.end(), m) != _pending.end()) {
            rasbery::refill::tenancy().queue_duplicates.fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error(
                "nodal batch arena: slot " + std::to_string(m) +
                " arrived twice in one rendezvous batch (two tenants share a slot)");
        }
        _pending.push_back(m);
        if (_lingering) _cv.notify_all();

        while (true) {
            if (_completed > my_batch) {
                // Somebody else ran our batch; our jnet/phis are already home.
                //
                // The test is `>=`, not `==`: one launch failure disables the
                // arena for good (fail() clears _available and launchBatch
                // refuses immediately after that), so every batch from the
                // first failure onwards failed.  Comparing against a
                // last-failure marker would let a straggler from the failing
                // batch wake up after a later failure moved the marker and
                // report success it never got.
                return my_batch < _first_failed_batch;
            }
            if (_launching || _open_batch != my_batch) {
                _cv.wait(lock);
                continue;
            }

            // ---- nobody is on the device: WE launch this batch --------------
            //
            // Claim it BEFORE anything that releases the lock.  The linger
            // below waits on the condition variable, which unlocks; without
            // this flag set first, a thread arriving during the linger would
            // see `_launching == false` and `_open_batch == my_batch` and elect
            // itself a SECOND launcher of the same batch.  Two launchers then
            // drive one stream at once, which is exactly the failure the CMFD
            // arena documents at CudaBICGBackend.cu:2345-2355: NaN results,
            // corrupted graph captures and heap corruption, invisible until the
            // batches get wide enough for the linger to matter.  One stream,
            // one elected launcher, no exceptions.
            _launching = true;

            long linger_us = _wait_us;
            if (_wait_auto) {
                if (_wait_max_us <= 0) {
                    linger_us = 0;
                } else {
                    const double estimate =
                        _arrival_gap_ewma_us > 0.0 ? 2.0 * _arrival_gap_ewma_us : 100.0;
                    linger_us = static_cast<long>(
                        std::clamp(estimate, 25.0, static_cast<double>(_wait_max_us)));
                }
            }
            const auto wait_start = std::chrono::steady_clock::now();
            if (linger_us > 0 && static_cast<int>(_pending.size()) < inUseCount()) {
                const auto deadline = wait_start + std::chrono::microseconds(linger_us);
                _lingering          = true;
                while (static_cast<int>(_pending.size()) < inUseCount() &&
                       _cv.wait_until(lock, deadline) != std::cv_status::timeout) {
                }
                _lingering = false;
            }
            g_nodal_batch_linger_us.store(static_cast<unsigned long long>(linger_us),
                                          std::memory_order_relaxed);

            std::vector<int> participants;
            participants.swap(_pending);
            ++_open_batch; // arrivals from here on join the NEXT batch
            std::sort(participants.begin(), participants.end());

            // The device work runs UNLOCKED on purpose: this is the window in
            // which the next batch fills up.
            lock.unlock();
            bool ok = true;
            if (!participants.empty()) ok = launchBatch(participants);
            lock.lock();
            _launching = false;
            _completed = my_batch + 1;
            if (!ok) {
                if (my_batch < _first_failed_batch) _first_failed_batch = my_batch;
            } else if (!participants.empty()) {
                g_nodal_batches.fetch_add(1, std::memory_order_relaxed);
                g_nodal_batch_drives.fetch_add(participants.size(), std::memory_order_relaxed);
                const std::size_t w = std::min<std::size_t>(participants.size(), 64);
                g_nodal_batch_hist[w].fetch_add(1, std::memory_order_relaxed);
            }
            lock.unlock();
            _cv.notify_all();
            return ok;
        }
    }

private:
    struct Slot {
        bool in_use = false;
        // staged host pointers for the drive in flight
        double*       h_jnet     = nullptr;
        const double* h_flux     = nullptr;
        double*       h_phis     = nullptr;
        const double* h_xsrf     = nullptr;
        const double* h_xsnf     = nullptr;
        const double* h_xssm     = nullptr;
        const double* h_chif     = nullptr;
        const double* h_const[9] = {};
        double        reigv      = 0.0;
        unsigned long long const_gen = 0, ref_gen = 0;
        // device residency, per array.  Explicit booleans rather than a
        // "generation 0 means nothing" sentinel: ref/state generations may
        // legitimately BE zero.
        bool               have_const = false, have_chif = false;
        unsigned long long res_const_gen = 0, res_ref_gen = 0;
        cuda_transfer::ByteExactMirror<double> xsrf_mirror;
        cuda_transfer::ByteExactMirror<double> xsnf_mirror;
        cuda_transfer::ByteExactMirror<double> xssm_mirror;
        // RASBERY_NODAL_XS_MIRROR_NO_BATCH_ALLOCATION
        bool pushed_xsrf = false, pushed_xsnf = false, pushed_xssm = false;
        // Page-locking is idempotent but it is a synchronising call, so it runs
        // only when one of these pointers actually changed (i.e. once).
        const void* pin_bulk[6]  = {}; // jnet, phis, flux, xsrf, xsnf, xssm
        const void* pin_const[9] = {};
        const void* pin_chif     = nullptr;
        // Rev.7.1 Task 18: the canonical ownership this drive was staged with,
        // in kCanonicalNodalRegions order (flux, jnet, phis).  Host for every
        // region until a segment says otherwise, which is what makes a legacy
        // slot's transfers byte-identical to the pre-Task-18 ones.
        gpu::CanonicalOwner canon_owner[gpu::kCanonicalNodalRegionCount] = {
            gpu::CanonicalOwner::Host, gpu::CanonicalOwner::Host,
            gpu::CanonicalOwner::Host};
        std::uint32_t canon_materialize =
            gpu::canonicalBit(gpu::CanonicalRegion::Jnet) |
            gpu::canonicalBit(gpu::CanonicalRegion::Phis);

        /// The staged owner of one nodal region.  A region outside
        /// kCanonicalNodalRegions is not shared with this arena at all, so Host
        /// -- the answer that transfers -- is the only safe reply.
        [[nodiscard]] gpu::CanonicalOwner ownerOf(gpu::CanonicalRegion r) const {
            for (int i = 0; i < gpu::kCanonicalNodalRegionCount; ++i)
                if (gpu::kCanonicalNodalRegions[i] == r) return canon_owner[i];
            return gpu::CanonicalOwner::Host;
        }
    };

    [[nodiscard]] int inUseCount() const {
        int c = 0;
        for (const Slot& s : _slot)
            if (s.in_use) ++c;
        return c;
    }

    /// Rev.7.1 Task 20: the post-condition of a nodal tenant reset.
    ///
    /// Slot fields AND the out-of-struct canonical binding, because the second
    /// is the one that has actually gone wrong here before (Task 18) and the
    /// audit is worthless if it only covers the half that never broke.
    [[nodiscard]] static bool nodalSlotIsReset(const Slot&                     sl,
                                               const gpu::CanonicalSlotBuffers& canon) {
        if (sl.in_use || sl.h_jnet != nullptr || sl.h_flux != nullptr ||
            sl.h_phis != nullptr || sl.h_xsrf != nullptr || sl.h_xsnf != nullptr ||
            sl.h_xssm != nullptr || sl.h_chif != nullptr || sl.pin_chif != nullptr)
            return false;
        if (sl.have_const || sl.have_chif || sl.pushed_xsrf || sl.pushed_xsnf ||
            sl.pushed_xssm || sl.reigv != 0.0)
            return false;
        if (sl.xsrf_mirror.valid() || sl.xsnf_mirror.valid() || sl.xssm_mirror.valid())
            return false;
        for (int i = 0; i < 6; ++i)
            if (sl.pin_bulk[i] != nullptr) return false;
        for (int i = 0; i < 9; ++i)
            if (sl.h_const[i] != nullptr || sl.pin_const[i] != nullptr) return false;
        for (int i = 0; i < gpu::kCanonicalNodalRegionCount; ++i)
            if (sl.canon_owner[i] != gpu::CanonicalOwner::Host) return false;
        return canon.flux == nullptr && canon.jnet == nullptr && canon.phis == nullptr &&
               canon.dtil == nullptr && canon.dhat == nullptr && canon.live_xs == nullptr;
    }

    bool fail(const char* what, cudaError_t e) {
        _status    = std::string(what) + " -> " + cudaGetErrorString(e);
        _available = false;
        return false;
    }

    void init(const ndl::NodalView& p) {
        // Rev.7.1 Task 18d: the nodal arena stands up on whichever deck reaches
        // it first, while the other decks may already be capturing.
        rasbery::AllocWindow window("nodal.arena.standup");
        _nxyz       = p.nxyz;
        _nsurf      = p.nsurf;
        _chif_empty = p.chif_empty;

        const std::size_t S   = static_cast<std::size_t>(_slots);
        const std::size_t nx  = static_cast<std::size_t>(_nxyz);
        const std::size_t ns  = static_cast<std::size_t>(_nsurf);
        const std::size_t ndg = nx * ndl::NDIR * ndl::NG;
        const std::size_t dg2 = nx * ndl::NDIR * ndl::NG2;
        const std::size_t ng1 = nx * ndl::NG;
        const std::size_t ng2 = nx * ndl::NG2;
        const std::size_t sg  = ns * ndl::NG;

        // Host shadows of the SHARED immutable tables, for compatible().
        _ref_hmesh.assign(p.hmesh, p.hmesh + nx * ndl::NDIR);
        _ref_albedo.assign(p.albedo, p.albedo + ndl::NDIR * ndl::NLR);
        _ref_lktosfc.assign(p.lktosfc, p.lktosfc + nx * ndl::NDIR * ndl::NLR);
        _ref_neib.assign(p.neib, p.neib + nx * ndl::NEWSB);
        _ref_lklr.assign(p.lklr, p.lklr + ns * ndl::NLR);
        _ref_idirlr.assign(p.idirlr, p.idirlr + ns * ndl::NLR);
        _ref_sgnlr.assign(p.sgnlr, p.sgnlr + ns * ndl::NLR);

        std::size_t off = 0;
        auto        take_shared = [&](std::size_t n) { const std::size_t o = off; off += n; return o; };
        auto        take_slot   = [&](std::size_t n) { const std::size_t o = off; off += S * n; return o; };

        const std::size_t o_hmesh  = take_shared(nx * ndl::NDIR);
        const std::size_t o_albedo = take_shared(ndl::NDIR * ndl::NLR);
        std::size_t       o_const[9];
        for (int i = 0; i < 9; ++i) o_const[i] = take_slot(ndg);
        const std::size_t o_chif = take_slot(ng1);
        const std::size_t o_xsrf = take_slot(ng1);
        const std::size_t o_xsnf = take_slot(ng1);
        const std::size_t o_xssm = take_slot(ng2);
        const std::size_t o_jnet = take_slot(sg);
        const std::size_t o_flux = take_slot(ng1);
        const std::size_t o_phis = take_slot(sg);
        const std::size_t o_reig = take_slot(1);
        const std::size_t o_tr0  = take_slot(ndg);
        const std::size_t o_tr1  = take_slot(ndg);
        const std::size_t o_tr2  = take_slot(ndg);
        const std::size_t o_mu   = take_slot(dg2);
        const std::size_t o_tau  = take_slot(dg2);
        const std::size_t o_mM   = take_slot(ng2);
        const std::size_t o_mMI  = take_slot(ng2);
        const std::size_t o_mMs  = take_slot(ng2);
        const std::size_t o_mMf  = take_slot(ng2);
        const std::size_t o_ds2  = take_slot(ndg);
        const std::size_t o_ds4  = take_slot(ndg);
        const std::size_t o_ds6  = take_slot(ndg);

        std::size_t ioff = 0;
        auto        take_int = [&](std::size_t n) { const std::size_t o = ioff; ioff += n; return o; };
        const std::size_t o_lktosfc = take_int(nx * ndl::NDIR * ndl::NLR);
        const std::size_t o_neib    = take_int(nx * ndl::NEWSB);
        const std::size_t o_lklr    = take_int(ns * ndl::NLR);
        const std::size_t o_idirlr  = take_int(ns * ndl::NLR);
        const std::size_t o_sgnlr   = take_int(ns * ndl::NLR);

        cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&_dbl), off * sizeof(double));
        if (rc != cudaSuccess) { fail("cudaMalloc(nodal arena doubles)", rc); return; }
        rc = cudaMalloc(reinterpret_cast<void**>(&_idx), ioff * sizeof(int));
        if (rc != cudaSuccess) { fail("cudaMalloc(nodal arena ints)", rc); return; }
        rc = cudaMalloc(reinterpret_cast<void**>(&_d_active), S * sizeof(std::uint32_t));
        if (rc != cudaSuccess) { fail("cudaMalloc(nodal arena mask)", rc); return; }
        rc = cudaMallocHost(reinterpret_cast<void**>(&_h_active), S * sizeof(std::uint32_t));
        if (rc != cudaSuccess) { fail("cudaMallocHost(nodal arena mask)", rc); return; }
        rc = cudaMallocHost(reinterpret_cast<void**>(&_h_reigv), S * sizeof(double));
        if (rc != cudaSuccess) { fail("cudaMallocHost(nodal arena reigv)", rc); return; }
        std::memset(_h_active, 0, S * sizeof(std::uint32_t));
        std::memset(_h_reigv, 0, S * sizeof(double));
        rc = cudaMemset(_d_active, 0, S * sizeof(std::uint32_t));
        if (rc != cudaSuccess) { fail("cudaMemset(nodal arena mask)", rc); return; }
        rc = cudaMalloc(reinterpret_cast<void**>(&_d_slot_map), S * sizeof(int));
        if (rc != cudaSuccess) { fail("cudaMalloc(nodal arena slot map)", rc); return; }
        rc = cudaMallocHost(reinterpret_cast<void**>(&_h_slot_map), S * sizeof(int));
        if (rc != cudaSuccess) { fail("cudaMallocHost(nodal arena slot map)", rc); return; }
        for (std::size_t i = 0; i < S; ++i) _h_slot_map[i] = -1;
        rc = cudaMemcpy(_d_slot_map, _h_slot_map, S * sizeof(int), cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) { fail("cudaMemcpy(nodal arena slot map)", rc); return; }
        rc = cudaMalloc(reinterpret_cast<void**>(&_d_views), S * sizeof(ndl::NodalView));
        if (rc != cudaSuccess) { fail("cudaMalloc(nodal arena view table)", rc); return; }
        rc = cudaMallocHost(reinterpret_cast<void**>(&_h_views), S * sizeof(ndl::NodalView));
        if (rc != cudaSuccess) { fail("cudaMallocHost(nodal arena view table)", rc); return; }
        _canon.assign(S, gpu::CanonicalSlotBuffers{});
        rc = cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
        if (rc != cudaSuccess) { fail("cudaStreamCreateWithFlags(nodal arena)", rc); return; }

        // SHARED geometry, uploaded once, synchronously, before any slot exists.
        struct { const void* src; std::size_t off; std::size_t bytes; bool is_int; } geo[] = {
            {p.hmesh, o_hmesh, nx * ndl::NDIR * sizeof(double), false},
            {p.albedo, o_albedo, ndl::NDIR * ndl::NLR * sizeof(double), false},
            {p.lktosfc, o_lktosfc, nx * ndl::NDIR * ndl::NLR * sizeof(int), true},
            {p.neib, o_neib, nx * ndl::NEWSB * sizeof(int), true},
            {p.lklr, o_lklr, ns * ndl::NLR * sizeof(int), true},
            {p.idirlr, o_idirlr, ns * ndl::NLR * sizeof(int), true},
            {p.sgnlr, o_sgnlr, ns * ndl::NLR * sizeof(int), true},
        };
        for (const auto& g : geo) {
            void* dst = g.is_int ? static_cast<void*>(_idx + g.off)
                                 : static_cast<void*>(_dbl + g.off);
            rc = cudaMemcpy(dst, g.src, g.bytes, cudaMemcpyHostToDevice);
            if (rc != cudaSuccess) { fail("cudaMemcpy(nodal arena geometry)", rc); return; }
        }

        // The SLOT-0 base view.  Everything the graph bakes lives here and none
        // of it ever moves: one allocation, no realloc path, no per-batch
        // rebinding.  nodalSlotView(base, m) is the only thing that turns this
        // into slot m's view, and it runs inside the kernel.
        _base            = p;              // shape scalars + chif_empty
        _base.hmesh      = _dbl + o_hmesh;
        _base.albedo     = _dbl + o_albedo;
        _base.lktosfc    = _idx + o_lktosfc;
        _base.neib       = _idx + o_neib;
        _base.lklr       = _idx + o_lklr;
        _base.idirlr     = _idx + o_idirlr;
        _base.sgnlr      = _idx + o_sgnlr;
        _base.eta1       = _dbl + o_const[0];
        _base.eta2       = _dbl + o_const[1];
        _base.m260       = _dbl + o_const[2];
        _base.m251       = _dbl + o_const[3];
        _base.m253       = _dbl + o_const[4];
        _base.m262       = _dbl + o_const[5];
        _base.m264       = _dbl + o_const[6];
        _base.diagD      = _dbl + o_const[7];
        _base.diagDI     = _dbl + o_const[8];
        _base.chif       = _dbl + o_chif;
        _base.xsrf       = _dbl + o_xsrf;
        _base.xsnf       = _dbl + o_xsnf;
        _base.xssm       = _dbl + o_xssm;
        _base.jnet       = _dbl + o_jnet;
        _base.flux       = _dbl + o_flux;
        _base.phis       = _dbl + o_phis;
        _base.trlcff0    = _dbl + o_tr0;
        _base.trlcff1    = _dbl + o_tr1;
        _base.trlcff2    = _dbl + o_tr2;
        _base.mu         = _dbl + o_mu;
        _base.tau        = _dbl + o_tau;
        _base.matM       = _dbl + o_mM;
        _base.matMI      = _dbl + o_mMI;
        _base.matMs      = _dbl + o_mMs;
        _base.matMf      = _dbl + o_mMf;
        _base.dsncff2    = _dbl + o_ds2;
        _base.dsncff4    = _dbl + o_ds4;
        _base.dsncff6    = _dbl + o_ds6;
        _base.reigv      = 0.0;              // superseded by reigv_dev, per slot
        _base.reigv_dev  = _dbl + o_reig;

        _cnt_ndg = ndg; _cnt_ng1 = ng1; _cnt_ng2 = ng2; _cnt_sg = sg;

        _slot.resize(S);
        _available = true;
        _status    = "ready";
        g_nodal_batch_slots.store(_slots, std::memory_order_relaxed);
    }

    /// Page-lock the host buffers this slot will be DMA'd from/to.  Launcher
    /// thread only, outside any capture, and only when a pointer actually
    /// changed -- cudaHostRegister is a synchronising call, not a per-drive one.
    /// The leases these take are owned by the buffers' owners (Geometry's
    /// jnet/phis/flux, XSSet's xs blocks, Nodal's nine constants) and released
    /// in their destructors; the pin_* fields here are only the per-tenant memo
    /// that keeps this from calling pinHost again on every batch.
    void pinSlot(Slot& sl) {
        const void* const  bulk[6]  = {sl.h_jnet, sl.h_phis, sl.h_flux,
                                       sl.h_xsrf, sl.h_xsnf, sl.h_xssm};
        const std::size_t  bytes[6] = {_cnt_sg * sizeof(double),  _cnt_sg * sizeof(double),
                                       _cnt_ng1 * sizeof(double), _cnt_ng1 * sizeof(double),
                                       _cnt_ng1 * sizeof(double), _cnt_ng2 * sizeof(double)};
        // The byte counts match the per-instance arm below and the XSSet/CMFD
        // arms exactly, so a buffer already leased by one of them deduplicates
        // here instead of arriving as a wider request the registry must refuse.
        const char* const tags[6] = {"geom.jnet@arena", "geom.phis@arena", "geom.phif@arena",
                                     "xs.xsrf@arena",   "xs.xsnf@arena",   "xs.xssm@arena"};
        for (int i = 0; i < 6; ++i) {
            if (bulk[i] == nullptr || bulk[i] == sl.pin_bulk[i]) continue;
            XsReconBackend::pinHost(bulk[i], bytes[i], tags[i]);
            sl.pin_bulk[i] = bulk[i];
        }
        for (int i = 0; i < 9; ++i) {
            if (sl.h_const[i] == nullptr || sl.h_const[i] == sl.pin_const[i]) continue;
            XsReconBackend::pinHost(sl.h_const[i], _cnt_ndg * sizeof(double),
                                    "nodal.const@arena");
            sl.pin_const[i] = sl.h_const[i];
        }
        if (sl.h_chif != nullptr && sl.h_chif != sl.pin_chif) {
            XsReconBackend::pinHost(sl.h_chif, _cnt_ng1 * sizeof(double), "xs.chif@arena");
            sl.pin_chif = sl.h_chif;
        }
    }

    bool memcpyAsyncOrFail(void* dst, const void* src, std::size_t bytes,
                           cudaMemcpyKind kind, const char* what) {
        const cudaError_t rc = cudaMemcpyAsync(dst, src, bytes, kind, _stream);
        if (rc != cudaSuccess) { fail(what, rc); return false; }
        return true;
    }

    /// The fixed-topology graph: five kernels, grid.y == slots, participation
    /// read from device memory.  Because nothing about the launch shape depends
    /// on WHO is in the batch, this single capture serves every subset for the
    /// rest of the run.  Failure is not fatal -- the plain launches below are
    /// numerically the same thing.
    /// ONE CHILD GRAPH PER BUCKET (Rev.7.1 Task 8).
    ///
    /// A graph bakes its launch dimensions, so grid.y -- the dispatch width --
    /// is topology.  Before compaction there was one width (the fleet) and
    /// therefore one graph; with compaction there is one per bucket, and the
    /// KEY is (bucket, MatEven fusion, geometry):
    ///
    ///   bucket    grid.y, baked
    ///   fusion    a different kernel set (MatEven vs Mat + Even)
    ///   geometry  grid.x comes from nxyz / nsurf
    ///
    /// The bucket ladder is deliberately coarse (nine steps to 64) so the cache
    /// saturates after a handful of distinct batch widths instead of growing
    /// with every count -- which is the same reason the case-phase scheduler
    /// buckets at all (Sec 5.5).
    cudaGraphExec_t ensureGraph(int lanes) {
        if (!_use_graph) return nullptr;
        for (const auto& e : _graphs)
            if (e.lanes == lanes && e.fuse == _fuse_mat_even && e.nxyz == _nxyz &&
                e.nsurf == _nsurf)
                return e.exec;

        const cudaError_t drc = cudaStreamSynchronize(_stream);
        if (drc != cudaSuccess) { _use_graph = false; return nullptr; }
        cudaGraph_t graph = nullptr;
        // Rev.7.1 Task 18d: exclusive of every allocation in the process for
        // the length of the window.  A sibling deck's first-touch pin or lazy
        // cudaMalloc landing here invalidates the capture, and a nodal capture
        // invalidated mid-flight is the same class of failure as the CMFD one.
        rasbery::CaptureWindow _capture_window(_stream, "nodal.bucket");
        cudaError_t rc =
            cudaStreamBeginCapture(_stream, cudaStreamCaptureModeThreadLocal);
        if (rc == cudaSuccess) {
            enqueueKernels(lanes);
            // Must run even if the enqueue faulted: this is what takes the
            // stream back OUT of capture mode.
            rc = cudaStreamEndCapture(_stream, &graph);
        }
        cudaGraphExec_t exec = nullptr;
        if (rc == cudaSuccess) rc = cudaGraphInstantiate(&exec, graph, 0ull);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (rc != cudaSuccess) {
            // Work submitted to a capturing stream is RECORDED, not executed,
            // so a failed capture leaves nothing pending and the direct
            // launches below are this batch's first and only execution.
            cudaGetLastError();
            _use_graph = false;
            g_nodal_batch_graph_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        _graphs.push_back(BucketGraph{exec, lanes, _fuse_mat_even, _nxyz, _nsurf});
        g_nodal_bucket_graphs.fetch_add(1, std::memory_order_relaxed);
        return exec;
    }

    /// Rev.7.1 Task 8.  `lanes` is the DISPATCH WIDTH: the fleet width when
    /// compaction is off (identity map, exactly the pre-Task-8 shape), and the
    /// smallest bucket covering the participants when it is on.
    void enqueueKernels(int lanes) {
        const unsigned L  = static_cast<unsigned>(lanes);
        const int      B  = 128;
        const int      gn = (_nxyz + B - 1) / B;
        const int      gs = (_nsurf + B - 1) / B;
        const int      gg = (_nxyz * ndl::NG + B - 1) / B;
        kNodalTrl0<true><<<dim3(static_cast<unsigned>(gg), L), B, 0, _stream>>>(
            _base, _d_slot_map, lanes, _d_views);
        kNodalTrl12<true><<<dim3(static_cast<unsigned>(gg), L), B, 0, _stream>>>(
            _base, _d_slot_map, lanes, _d_views);
        if (_fuse_mat_even) {
            kNodalMatEven<true><<<dim3(static_cast<unsigned>(gn), L), B, 0, _stream>>>(
                _base, _d_slot_map, lanes, _d_views);
        } else {
            kNodalMat<true><<<dim3(static_cast<unsigned>(gn), L), B, 0, _stream>>>(
                _base, _d_slot_map, lanes, _d_views);
            kNodalEven<true><<<dim3(static_cast<unsigned>(gn), L), B, 0, _stream>>>(
                _base, _d_slot_map, lanes, _d_views);
        }
        kNodalJnet<true><<<dim3(static_cast<unsigned>(gs), L), B, 0, _stream>>>(
            _base, _d_slot_map, lanes, _d_views);
    }

    /// Rebuild the per-slot view table and push it.
    ///
    /// A legacy slot's entry is exactly nodalSlotView(_base, m) -- the addresses
    /// the dense rebase used to compute in the kernel.  An ADOPTED slot keeps
    /// every one of those except the three canonical regions, because only
    /// flux/jnet/phis are shared with the CMFD side; the working arrays, the
    /// nine constants, the geometry and the XS block stay the arena's own.  That
    /// is what makes the stride question disappear: the canonical pointer is
    /// absolute, so its layout no longer has to match the arena's.
    bool refreshViews() {
        if (!_views_dirty) return true;
        for (int m = 0; m < _slots; ++m) {
            ndl::NodalView v = ndl::nodalSlotView(_base, m);
            const gpu::CanonicalSlotBuffers& c = _canon[static_cast<std::size_t>(m)];
            if (c.flux != nullptr) v.flux = c.flux;
            if (c.jnet != nullptr) v.jnet = c.jnet;
            if (c.phis != nullptr) v.phis = c.phis;
            _h_views[m] = v;
        }
        const cudaError_t rc =
            cudaMemcpy(_d_views, _h_views,
                       static_cast<std::size_t>(_slots) * sizeof(ndl::NodalView),
                       cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) { fail("cudaMemcpy(nodal arena view table)", rc); return false; }
        _views_dirty = false;
        return true;
    }

public:
    /// Adopt canonical buffers for ONE slot.  All-null reverts it to legacy.
    /// The table is a kernel argument baked into every captured graph, but the
    /// TABLE POINTER never moves -- only its contents -- so adoption does not
    /// invalidate a graph.  It does mark the table dirty.
    void adoptCanonical(int slot, const gpu::CanonicalSlotBuffers& buffers) {
        if (slot < 0 || slot >= _slots) return;
        if (!gpu::canonicalNodalSetIsCoherent(buffers)) {
            // All three or none.  A partial set would pair the canonical jnet
            // with the arena's own flux -- two different outer iterations,
            // silently blended.
            std::cerr << "[RASBERY][WARN][nodal] refusing a partial canonical set for slot "
                      << slot << " (flux/jnet/phis must be adopted together)"
                      << std::endl;
            return;
        }
        _canon[static_cast<std::size_t>(slot)] = buffers;
        _views_dirty                           = true;
    }

    [[nodiscard]] gpu::CanonicalSlotBuffers canonicalOf(int slot) const {
        if (slot < 0 || slot >= _slots) return gpu::CanonicalSlotBuffers{};
        return _canon[static_cast<std::size_t>(slot)];
    }

private:
    /// Smallest configured bucket covering `count`, mirroring the scheduler's
    /// ladder (GpuPhaseScheduler.h) so the nodal phase and the case-phase
    /// scheduler cannot disagree about what a bucket is.  Kept as its own
    /// function so the contract test can check the two ladders agree.
    static int nodalBucketFor(int count, int slots) {
        static const int kBuckets[] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
        for (int b : kBuckets)
            if (count <= b) return b < slots ? b : slots;
        return slots;
    }

    /// Launcher-thread only.  Uploads for the participants, one graph launch
    /// (or five plain launches), downloads for the participants, one drain.
    bool launchBatch(const std::vector<int>& part) {
        if (!_available) return false; // a previous batch already took us down
        const std::size_t S    = static_cast<std::size_t>(_slots);
        const std::size_t sgb  = _cnt_sg * sizeof(double);
        const std::size_t ng1b = _cnt_ng1 * sizeof(double);
        const std::size_t ng2b = _cnt_ng2 * sizeof(double);
        const std::size_t ndgb = _cnt_ndg * sizeof(double);

        unsigned long long xs_h2d_bytes = 0;
        unsigned long long xs_h2d_skipped_bytes = 0;

        for (int m : part) pinSlot(_slot[static_cast<std::size_t>(m)]);

        // ---- Rev.7.1 Task 8: the slot map -----------------------------------
        //
        // COMPACTION OFF (default) is the identity over the whole fleet, so the
        // grid, the block count and the work each block does are exactly what
        // they were before Task 8; only the participation test moved from a
        // mask lookup to a map lookup.  ON, the map holds just the participants
        // and `lanes` is the bucket, so the blocks that launch are the blocks
        // that work.
        //
        // Ascending physical order in both cases.  That is not cosmetic: the
        // per-slot XS mirrors and generation counters are indexed by PHYSICAL
        // slot, so the map is the only thing that may be logical -- and a map
        // whose order depended on the participant vector's arrival order would
        // make the device access pattern differ run to run for the same batch.
        const int lanes = _compact ? nodalBucketFor(static_cast<int>(part.size()), _slots)
                                   : _slots;
        for (int i = 0; i < _slots; ++i) _h_slot_map[i] = -1;
        if (_compact) {
            int n = 0;
            for (int m : part) _h_slot_map[n++] = m; // `part` is already sorted
        } else {
            for (int m : part) _h_slot_map[m] = m;   // identity, holes stay -1
        }
        if (!memcpyAsyncOrFail(_d_slot_map, _h_slot_map,
                               static_cast<std::size_t>(_slots) * sizeof(int),
                               cudaMemcpyHostToDevice, "nodal arena slot map H2D"))
            return drained();

        if (!refreshViews()) return drained();

        cudaGraphExec_t exec = ensureGraph(lanes);

        // ---- compaction receipt (Sec 9.3) -----------------------------------
        // Counted per PHASE-BLOCK-COLUMN, which is what the compaction actually
        // removes: `lanes` grid.y columns are launched, of which part.size() do
        // work.  Reporting only "drives" would hide the whole effect.
        g_nodal_logical_drives.fetch_add(part.size(), std::memory_order_relaxed);
        g_nodal_physical_blocks.fetch_add(static_cast<unsigned long long>(part.size()),
                                          std::memory_order_relaxed);
        g_nodal_padding_blocks.fetch_add(
            static_cast<unsigned long long>(lanes) - part.size(), std::memory_order_relaxed);
        nodalBucketHistogramBump(lanes);

        // ---- per-slot uploads: the SAME set the per-instance FULL path
        // uploads, with the same residency rules, in the same order ----------
        for (int m : part) {
            Slot&             sl = _slot[static_cast<std::size_t>(m)];
            const std::size_t s  = static_cast<std::size_t>(m);
            _h_reigv[s]          = sl.reigv;

            if (!sl.have_const || sl.const_gen != sl.res_const_gen) {
                for (int i = 0; i < 9; ++i)
                    if (!memcpyAsyncOrFail(_dbl_const(i) + s * _cnt_ndg, sl.h_const[i], ndgb,
                                           cudaMemcpyHostToDevice, "nodal arena consts H2D"))
                        return drained();
                sl.res_const_gen = sl.const_gen;
                sl.have_const    = true;
            }
            if (!_chif_empty && sl.h_chif != nullptr &&
                (!sl.have_chif || sl.ref_gen != sl.res_ref_gen)) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.chif) + s * _cnt_ng1, sl.h_chif,
                                       ng1b, cudaMemcpyHostToDevice, "nodal arena chif H2D"))
                    return drained();
                sl.res_ref_gen = sl.ref_gen;
                sl.have_chif   = true;
            }
            // The arena cannot use hoststateGeneration because its XS
            // allocation is separate from the per-instance backend. Instead,
            // compare every byte with the last upload that completed
            // successfully for this slot. This is conservative for all NaN
            // payloads and signed zeroes, and never assumes a physics
            // generation implies residency.
            const bool push_xsrf =
                !_mirror_xs || !sl.xsrf_mirror.matches(sl.h_xsrf, _cnt_ng1);
            const bool push_xsnf =
                !_mirror_xs || !sl.xsnf_mirror.matches(sl.h_xsnf, _cnt_ng1);
            const bool push_xssm =
                !_mirror_xs || !sl.xssm_mirror.matches(sl.h_xssm, _cnt_ng2);

            if (push_xsrf) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.xsrf) + s * _cnt_ng1,
                                       sl.h_xsrf, ng1b, cudaMemcpyHostToDevice,
                                       "nodal arena xsrf H2D"))
                    return drained();
                xs_h2d_bytes += ng1b;
            } else {
                xs_h2d_skipped_bytes += ng1b;
            }
            if (push_xsnf) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.xsnf) + s * _cnt_ng1,
                                       sl.h_xsnf, ng1b, cudaMemcpyHostToDevice,
                                       "nodal arena xsnf H2D"))
                    return drained();
                xs_h2d_bytes += ng1b;
            } else {
                xs_h2d_skipped_bytes += ng1b;
            }
            if (push_xssm) {
                if (!memcpyAsyncOrFail(const_cast<double*>(_base.xssm) + s * _cnt_ng2,
                                       sl.h_xssm, ng2b, cudaMemcpyHostToDevice,
                                       "nodal arena xssm H2D"))
                    return drained();
                xs_h2d_bytes += ng2b;
            } else {
                xs_h2d_skipped_bytes += ng2b;
            }
            sl.pushed_xsrf = _mirror_xs && push_xsrf;
            sl.pushed_xsnf = _mirror_xs && push_xsnf;
            sl.pushed_xssm = _mirror_xs && push_xssm;
            // ---- Rev.7.1 Task 18: THE ARENA HONOURS THE CANONICAL BINDING ---
            //
            // WHAT IT USED TO DO AND WHY THAT WAS WRONG.  adoptCanonical put the
            // segment's jnet/flux/phis into the view table -- so the KERNELS
            // read and wrote them -- and then these two lines uploaded
            // Geometry::Jnet and Geometry::Phif into the ARENA'S OWN dense
            // block, which the kernels no longer touched, while the downloads
            // below read that same untouched block back over the caller's host
            // arrays.  Adoption was therefore accepted and ignored twice: the
            // uploads went nowhere and the downloads carried a stale copy home.
            // The segment, seeing the binding live, had stopped filling
            // Geometry::Jnet, so a batch that dropped its bridge converged
            // somewhere else entirely (kngr3 statepoint 1: 800.33 ppm in 290
            // outers against 770.15 in 263).  Task 18-lite worked around it by
            // asking a static `is it honoured` predicate and keeping the bridge; this is
            // the fix that makes the question unnecessary.
            //
            // THE ADDRESS COMES FROM THE VIEW TABLE, WHICH IS THE ONLY PLACE
            // THAT KNOWS.  `v` is what refreshViews() built for this slot a few
            // lines up -- the arena's dense pointers for a legacy slot, the
            // borrowed canonical ones for an adopted slot -- so a legacy slot
            // gets byte-identical transfers to the pre-Task-18 code and an
            // adopted slot transfers to and from the buffer the kernel uses.
            //
            // THE ELISION IS THE SAME PREDICATE THE PER-INSTANCE ARM USES,
            // consulted with the ownership this slot STAGED (see drive()): a
            // canonical region whose last writer was a device side needs no
            // upload, and one no host consumer has asked for needs no download.
            const ndl::NodalView&            v     = _h_views[s];
            const gpu::CanonicalSlotBuffers& canon = _canon[s];
            if (gpu::canonicalElidesUpload(canon, gpu::CanonicalRegion::Jnet,
                                           sl.ownerOf(gpu::CanonicalRegion::Jnet),
                                           gpu::CanonicalOwner::Nodal)) {
                g_canon_up_bytes.fetch_add(sgb, std::memory_order_relaxed);
            } else if (!memcpyAsyncOrFail(v.jnet, sl.h_jnet, sgb, cudaMemcpyHostToDevice,
                                          "nodal arena jnet H2D")) {
                return drained();
            }
            if (gpu::canonicalElidesUpload(canon, gpu::CanonicalRegion::Flux,
                                           sl.ownerOf(gpu::CanonicalRegion::Flux),
                                           gpu::CanonicalOwner::Nodal)) {
                g_canon_up_bytes.fetch_add(ng1b, std::memory_order_relaxed);
            } else if (!memcpyAsyncOrFail(const_cast<double*>(v.flux), sl.h_flux, ng1b,
                                          cudaMemcpyHostToDevice, "nodal arena flux H2D")) {
                return drained();
            }
        }
        // One copy of the whole reigv array: 8 bytes a slot, and it keeps the
        // per-drive eigenvalue out of the kernel arguments the graph baked.
        if (!memcpyAsyncOrFail(const_cast<double*>(_base.reigv_dev), _h_reigv, S * sizeof(double),
                               cudaMemcpyHostToDevice, "nodal arena reigv H2D"))
            return drained();

        // ---- the five phases ----------------------------------------------
        if (exec != nullptr) {
            const cudaError_t rc = cudaGraphLaunch(exec, _stream);
            if (rc != cudaSuccess) { fail("cudaGraphLaunch(nodal arena)", rc); return drained(); }
            g_nodal_batch_graph_launches.fetch_add(1, std::memory_order_relaxed);
        } else {
            enqueueKernels(lanes);
        }
        // Judged BEFORE the downloads are queued: a launch that never ran must
        // not have stale device jnet copied over the caller's host array, which
        // is the very buffer its CPU fallback then reads as an input.
        const cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess) { fail("nodal arena launch", lerr); return drained(); }

        // ---- the mined minimal download set, per participant ---------------
        //
        // Rev.7.1 Task 18: read from the VIEW, and only when a host consumer has
        // asked.  Inside a device outer segment nothing on the host reads
        // Geometry::Jnet or Geometry::Phis between two outers -- the segment
        // mirrors both itself at its exit -- so an adopted slot's two D2Hs
        // disappear along with the two H2Ds above.  A legacy slot's mask has
        // both bits set and its transfers are unchanged.
        for (int m : part) {
            Slot&             sl = _slot[static_cast<std::size_t>(m)];
            const std::size_t s  = static_cast<std::size_t>(m);
            const ndl::NodalView&            v     = _h_views[s];
            const gpu::CanonicalSlotBuffers& canon = _canon[s];
            if (gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Jnet,
                                             sl.canon_materialize)) {
                g_canon_down_bytes.fetch_add(sgb, std::memory_order_relaxed);
            } else if (!memcpyAsyncOrFail(sl.h_jnet, v.jnet, sgb, cudaMemcpyDeviceToHost,
                                          "nodal arena jnet D2H")) {
                return drained();
            }
            if (gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Phis,
                                             sl.canon_materialize)) {
                g_canon_down_bytes.fetch_add(sgb, std::memory_order_relaxed);
            } else if (!memcpyAsyncOrFail(sl.h_phis, v.phis, sgb, cudaMemcpyDeviceToHost,
                                          "nodal arena phis D2H")) {
                return drained();
            }
        }

        const cudaError_t src = cudaStreamSynchronize(_stream);
        if (src != cudaSuccess) { fail("nodal arena drain", src); return false; }

        // Commit only after the drain proved the queued H2D reached the device.
        // A failed batch leaves the previous mirrors untouched, so they never
        // describe bytes that may not be resident.
        for (int m : part) {
            Slot& sl = _slot[static_cast<std::size_t>(m)];
            if (sl.pushed_xsrf) sl.xsrf_mirror.commit(sl.h_xsrf, _cnt_ng1);
            if (sl.pushed_xsnf) sl.xsnf_mirror.commit(sl.h_xsnf, _cnt_ng1);
            if (sl.pushed_xssm) sl.xssm_mirror.commit(sl.h_xssm, _cnt_ng2);
            sl.pushed_xsrf = sl.pushed_xsnf = sl.pushed_xssm = false;
        }
        g_nodal_batch_xs_h2d_bytes.fetch_add(xs_h2d_bytes, std::memory_order_relaxed);
        g_nodal_batch_xs_h2d_skipped_bytes.fetch_add(
            xs_h2d_skipped_bytes, std::memory_order_relaxed);
        g_nodal_d2h_bytes.store(2 * sgb, std::memory_order_relaxed);
        g_nodal_drives.fetch_add(part.size(), std::memory_order_relaxed);
        return true;
    }

    /// Failing out of a half-enqueued batch must not leave a D2H in flight:
    /// every participant is about to run its own CPU body over the very
    /// host.jnet this stream might still be writing.  Drain first.
    bool drained() {
        cudaStreamSynchronize(_stream);
        cudaGetLastError();
        return false;
    }

    [[nodiscard]] double* _dbl_const(int i) const {
        const double* base[9] = {_base.eta1, _base.eta2, _base.m260, _base.m251, _base.m253,
                                 _base.m262, _base.m264, _base.diagD, _base.diagDI};
        return const_cast<double*>(base[i]);
    }

    int          _slots      = 0;
    bool         _available  = false;
    std::string  _status     = "not initialised";
    int          _nxyz = 0, _nsurf = 0, _chif_empty = 0;
    cudaStream_t _stream = nullptr;

    double*        _dbl         = nullptr;
    int*           _idx         = nullptr;
    std::uint32_t* _d_active    = nullptr;
    std::uint32_t* _h_active    = nullptr; // pinned
    // Rev.7.1 Task 8: logical lane -> physical slot.  -1 is a padding lane.
    int*           _d_slot_map  = nullptr;
    int*           _h_slot_map  = nullptr; // pinned
    // Rev.7.1 pre-W3: physical slot -> fully-resolved NodalView.
    //
    // Built on the host so the kernels never compute a slot address.  Default
    // entries come from nodalSlotView(_base, m), so a table nobody has adopted
    // into holds byte-for-byte the addresses the dense rebase produced -- which
    // is what makes the conversion pure indirection.
    ndl::NodalView* _d_views = nullptr;
    ndl::NodalView* _h_views = nullptr; // pinned
    /// Per-slot canonical overrides.  All-null entries mean "legacy slot".
    std::vector<gpu::CanonicalSlotBuffers> _canon;
    bool                                   _views_dirty = true;
    /// RASBERY_GPU_NODAL_COMPACT, default OFF.  Off, the map is the identity
    /// over the fleet and the launch shape is exactly the pre-Task-8 one.
    bool           _compact     = nodalCompactEnabled();
    double*        _h_reigv     = nullptr; // pinned

    ndl::NodalView  _base{};
    /// One instantiation per (bucket, fusion, geometry).  A graph bakes grid.y,
    /// so the dispatch width IS topology; the coarse bucket ladder is what keeps
    /// this list short instead of growing with every distinct batch count.
    struct BucketGraph {
        cudaGraphExec_t exec;
        int             lanes;
        bool            fuse;
        int             nxyz;
        int             nsurf;
    };
    std::vector<BucketGraph> _graphs;
    bool                     _use_graph = true;
    bool            _fuse_mat_even = true;
    bool            _mirror_xs = true;

    std::size_t _cnt_ndg = 0, _cnt_ng1 = 0, _cnt_ng2 = 0, _cnt_sg = 0;

    std::vector<double> _ref_hmesh, _ref_albedo;
    std::vector<int>    _ref_lktosfc, _ref_neib, _ref_lklr, _ref_idirlr, _ref_sgnlr;

    std::vector<Slot>       _slot;
    std::mutex              _mutex;
    std::condition_variable _cv;
    std::vector<int>        _pending;
    unsigned long long      _open_batch         = 0;
    unsigned long long      _completed          = 0;
    unsigned long long      _first_failed_batch = ~0ull;
    bool                    _launching    = false;
    bool                    _lingering    = false;
    // Default: NO linger -- a batch is purely "whoever arrived while the last
    // one ran".  Same default as the CMFD arena, and for the same reason: a
    // slot that is somewhere else entirely (CMFD, TH, I/O) is not coming, so a
    // launcher that waits for a full house pays the budget on every single
    // drive and buys nothing.  RASBERY_NODAL_BATCH_WAIT_US=auto turns on the
    // bounded 2x-arrival-gap estimate; a number sets a fixed budget.
    long                    _wait_us      = 0;
    bool                    _wait_auto    = false;
    long                    _wait_max_us  = 2000;
    std::chrono::steady_clock::time_point _last_arrival{};
    bool                                  _have_arrival        = false;
    double                                _arrival_gap_ewma_us = 0.0;
};

std::mutex  g_nodal_arena_mutex;
NodalArena* g_nodal_arena        = nullptr; // leaked on purpose (process lifetime)
bool        g_nodal_arena_failed = false;

/// The arena is engaged ONLY for a real multi-instance batch running the FULL
/// device pipeline.  Width 1, hybrid mode, or the arm switched off all keep the
/// existing per-instance paths exactly as they are.
bool nodalArenaWanted() {
    static const bool on = [] {
        return rasberyGpuNodalEnabled() && rasberyGpuNodalFullEnabled() &&
               !envFlagDisabled("RASBERY_GPU_NODAL_BATCH");
    }();
    return on && rasberyNodalBatchWidth() > 1;
}

NodalArena* nodalArenaFor(const ndl::NodalView& proto) {
    std::lock_guard<std::mutex> lock(g_nodal_arena_mutex);
    if (g_nodal_arena_failed) return nullptr;
    if (g_nodal_arena == nullptr) {
        NodalArena* a = new NodalArena(proto, rasberyNodalBatchWidth());
        if (!a->available()) {
            std::cerr << "[RASBERY][WARN][nodal] batch arena unavailable (" << a->status()
                      << ") -- per-instance nodal arm\n";
            g_nodal_arena_failed = true;
            return nullptr; // deliberately leaked: it may still own device memory
        }
        g_nodal_arena = a;
        std::cout << "[RASBERY][NODAL][BATCH] arena: slots=" << a->slots()
                  << " nxyz=" << proto.nxyz << " nsurf=" << proto.nsurf << " (" << a->status()
                  << ")" << std::endl;
    }
    return g_nodal_arena;
}

/// Extended [RASBERY][NODAL][GPU] receipt.
///
/// main.cpp already prints `{"drives_solved":N}` from rasberyGpuNodalDrives(),
/// and main.cpp is not this member's file to edit -- so the arm's own counters
/// (which only exist in this TU) are emitted from here as a strict SUPERSET of
/// that line, at static destruction, i.e. AFTER main returns and after main's
/// line.  A consumer taking the LAST [RASBERY][NODAL][GPU] match gets every
/// field; one taking the first still gets valid, if shorter, JSON.
///
/// <iostream> is included at the top of this TU, so its ios_base::Init object
/// is constructed before this one and destroyed after it: std::cout is alive
/// here.
struct NodalReceipt {
    ~NodalReceipt() {
        if (!rasbery::rasberyGpuNodalEnabled()) return;
        std::cout << "[RASBERY][NODAL][GPU] {\"drives_solved\":"
                  << g_nodal_drives.load(std::memory_order_relaxed)
                  << ",\"full_mode\":"
                  << (rasbery::rasberyGpuNodalFullEnabled() ? 1 : 0)
                  << ",\"mat_even_fused\":"
                  << (nodalFuseMatEvenEnabled() ? 1 : 0)
                  << ",\"graph_launches\":"
                  << g_nodal_graph_launches.load(std::memory_order_relaxed)
                  << ",\"graph_captures\":"
                  << g_nodal_graph_captures.load(std::memory_order_relaxed)
                  << ",\"graph_fallbacks\":"
                  << g_nodal_graph_fallbacks.load(std::memory_order_relaxed)
                  << ",\"d2h_bytes_per_drive\":"
                  << g_nodal_d2h_bytes.load(std::memory_order_relaxed)
                  << ",\"drains_deferred\":"
                  << g_nodal_drains_deferred.load(std::memory_order_relaxed)
                  << ",\"reigv_device_drives\":"
                  << g_nodal_reigv_device.load(std::memory_order_relaxed) << "}"
                  << std::endl;

        // The arena's own receipt, on its own tag so nothing consuming the line
        // above has to change.  Printed whenever the arena was reachable at all
        // (slots>0), so "engaged but never launched" is distinguishable from
        // "never built".
        // Rev.7.1 Task 8 compaction receipt, on its own tag.  The number that
        // matters is padding_blocks / (physical + padding): with compaction off
        // it is (slots - width)/slots per drive, which is what the compaction
        // removes.  Printed whether or not compaction is on, so the two are
        // directly comparable from two runs of the same deck.
        std::ostringstream comp;
        const unsigned long long phys = g_nodal_physical_blocks.load(std::memory_order_relaxed);
        const unsigned long long pad  = g_nodal_padding_blocks.load(std::memory_order_relaxed);
        comp << "[RASBERY][NODAL][COMPACT] {\"enabled\":" << (nodalCompactEnabled() ? 1 : 0)
             << ",\"logical_drives\":"
             << g_nodal_logical_drives.load(std::memory_order_relaxed)
             << ",\"physical_slot_blocks\":" << phys
             << ",\"padding_blocks\":" << pad
             << ",\"padding_fraction\":"
             << ((phys + pad) ? static_cast<double>(pad) / static_cast<double>(phys + pad)
                              : 0.0)
             << ",\"bucket_graphs\":"
             << g_nodal_bucket_graphs.load(std::memory_order_relaxed)
             << ",\"bucket_histogram\":[";
        for (std::size_t i = 0; i < g_nodal_bucket_histogram.size(); ++i) {
            if (i) comp << ',';
            comp << g_nodal_bucket_histogram[i].load(std::memory_order_relaxed);
        }
        comp << "]}";
        std::cout << comp.str() << std::endl;

        const int slots = g_nodal_batch_slots.load(std::memory_order_relaxed);
        if (slots <= 0) return;
        const unsigned long long b = g_nodal_batches.load(std::memory_order_relaxed);
        const unsigned long long p = g_nodal_batch_drives.load(std::memory_order_relaxed);
        std::ostringstream line;
        line << "[RASBERY][NODAL][BATCH] {\"slots\":" << slots
             << ",\"batches_launched\":" << b
             << ",\"instance_drives\":" << p
             << ",\"mean_width\":"
             << (b ? static_cast<double>(p) / static_cast<double>(b) : 0.0)
             << ",\"graph_launches\":"
             << g_nodal_batch_graph_launches.load(std::memory_order_relaxed)
             << ",\"graph_fallbacks\":"
             << g_nodal_batch_graph_fallbacks.load(std::memory_order_relaxed)
             << ",\"drive_fallbacks\":"
             << g_nodal_batch_fallbacks.load(std::memory_order_relaxed)
             << ",\"slot_refusals\":"
             << g_nodal_batch_refused.load(std::memory_order_relaxed)
             << ",\"last_linger_us\":"
             << g_nodal_batch_linger_us.load(std::memory_order_relaxed)
             << ",\"xs_mirror\":" << (nodalXsMirrorEnabled() ? 1 : 0)
             << ",\"xs_h2d_bytes\":"
             << g_nodal_batch_xs_h2d_bytes.load(std::memory_order_relaxed)
             << ",\"xs_h2d_skipped_bytes\":"
             << g_nodal_batch_xs_h2d_skipped_bytes.load(std::memory_order_relaxed)
             << ",\"width_histogram\":[";
        const int wmax = slots < 64 ? slots : 64;
        for (int w = 1; w <= wmax; ++w) {
            if (w > 1) line << ',';
            line << g_nodal_batch_hist[w].load(std::memory_order_relaxed);
        }
        line << "]}";
        std::cout << line.str() << std::endl;

    }
};
NodalReceipt g_nodal_receipt;

} // namespace

struct XsReconBackend::Impl {
    bool          available = false;
    std::string   status    = "not initialised";
    cudaStream_t  stream    = nullptr;

    int nxyz   = 0;
    int n_fuel = 0;

    double*             dev_block   = nullptr;
    std::size_t         block_doubles = 0;
    int*                dev_fuel    = nullptr;
    unsigned long long* dev_scalars = nullptr; // [0]=max bits, [1]=solved
    double*             dev_dep     = nullptr; // depTrans rows: [0..38]=I135, [39..77]=Xe135

    unsigned long long resident_micx_generation  = 0; // 0 = nothing resident
    unsigned long long resident_state_generation = 0; // _xs/_iden host==device
    bool               fuel_uploaded             = false;

    // --- flat-XS extension (RASBERY_GPU_FLATXS) ---------------------------
    double*            dev_ref = nullptr; // [9 mic | msm | 9 lmp | lsm]
    std::size_t        off_ref_mic[fxs::N_ACTIVE] = {};
    std::size_t        off_ref_msm = 0;
    std::size_t        off_ref_lmp[fxs::N_ACTIVE] = {};
    std::size_t        off_ref_lsm = 0;
    unsigned long long resident_ref_generation = 0;

    double*     dev_pernode = nullptr; // [wvfr | dmod | bppm], nx each
    int*        dev_nodes   = nullptr; // node list + off + cnt, grow-only
    int*        dev_off     = nullptr;
    int*        dev_cnt     = nullptr;
    std::size_t nodes_cap   = 0;
    int*        dev_sdid    = nullptr; // delta stream, grow-only
    double*     dev_sx      = nullptr;
    double*     dev_sscale  = nullptr;
    std::size_t stream_cap  = 0;

    // --- nodal extension (RASBERY_GPU_NODAL) ------------------------------
    double*            ndev_dbl   = nullptr; // hmesh|albedo|consts x9|chif|jnet|flux|phis|work
    int*               ndev_int   = nullptr; // lktosfc|neib|lklr|idirlr|sgnlr
    bool               nodal_geom_uploaded = false;
    unsigned long long resident_const_generation = 0;
    unsigned long long resident_chif_generation  = 0;
    int                nodal_nsurf = 0;
    std::size_t n_off_hmesh = 0, n_off_albedo = 0, n_off_consts = 0,
                n_off_chif = 0, n_off_jnet = 0, n_off_flux = 0, n_off_phis = 0,
                n_off_reigv = 0, n_off_work = 0;
    std::size_t n_ioff_lktosfc = 0, n_ioff_neib = 0, n_ioff_lklr = 0,
                n_ioff_idirlr = 0, n_ioff_sgnlr = 0;

    // --- FULL-mode CUDA graph (RASBERY_GPU_NODAL_FULL) --------------------
    // The FULL drive is the same sequence of fixed-address operations every
    // time, so it is captured once and replayed.  `nodal_h_reigv` is the
    // pinned host slot the captured H2D reads from: a graph bakes memcpy
    // source addresses, not their contents, so writing the new eigenvalue
    // there before each launch is what keeps the replay current.
    // RASBERY_GPU_NODAL_GRAPH=0 forces the plain per-drive launches.  It is
    // the A/B knob that proves the capture (and the reigv indirection it
    // needs) changes nothing numerically: graph on vs off must be bit-equal.
    cudaGraphExec_t nodal_graph      = nullptr;
    /// Rev.7.1 Task 10: THE GRAPH THE EXEC WAS MADE FROM, KEPT.
    ///
    /// The drive below used to destroy it on the line after the instantiate.
    /// That was right until the outer body became something a conditional WHILE
    /// captures: a cudaGraphLaunch into a capturing stream is REFUSED
    /// (cudaErrorStreamCaptureUnsupported) and the refusal invalidates the whole
    /// capture, so the only way this drive enters a captured body is as a child
    /// graph node -- and cudaGraphAddChildGraphNode takes a cudaGraph_t, which an
    /// exec cannot be turned back into.  Measured in
    /// tools/probe_while_body_capture.cu: `graph_launch_in_capture` false,
    /// `child_graph_node` true, both on the local 12.6/sm_61 box.
    ///
    /// It is the EXEC'S TWIN and dropNodalGraph destroys the pair: a key change
    /// that dropped one and kept the other would splice a body that no longer
    /// matches the exec the stream path launches, which is the one failure this
    /// mechanism must not have.
    cudaGraph_t     nodal_graph_src  = nullptr;
    bool            nodal_use_graph  = !envFlagDisabled("RASBERY_GPU_NODAL_GRAPH");
    double*         nodal_h_reigv    = nullptr; // pinned, 1 double
    /// Rev.7.1 W3 item 3: the device outer segment writes the reigv slot itself.
    ///
    /// NOT A GRAPH KEY.  It flips per DRIVE -- on for an in-segment canonical
    /// outer, off for the Wielandt warm-up outer beside it -- so keying the
    /// capture on it would drop and re-instantiate the graph at every flip.  The
    /// upload it suppresses therefore lives OUTSIDE the capture (issued on
    /// d.stream immediately before the launch, which is the same stream order
    /// being the graph's first node gave it), and the capture is identical
    /// either way.
    bool            nodal_reigv_device = false;
    /// Rev.7.1 Task 10 part 3: the device outer segment's per-slot halt table,
    /// or null.  UNLIKE nodal_reigv_device this IS a graph key: it is a kernel
    /// ARGUMENT (NodalView::halt), and a graph captured with it null cannot be
    /// replayed gated.  It settles once per run, so it costs one instantiation.
    const void*     nodal_halt      = nullptr;
    int             nodal_halt_slot = 0;
    // Everything the capture baked in.  Any change invalidates the graph FOR
    // THAT KEY -- which, since Rev.7.1 Task 10 part 4, is not the same thing as
    // invalidating the graph.  See NodalGraphKey below.

    // --- Rev.7.1 Task 7: canonical CMFD-Nodal device state ------------------
    //
    // `canonical.buffers` is BORROWED from GpuPhysicsArena (fixed addresses, so
    // baking them into the capture is safe).  All-null means legacy, and every
    // predicate is then inert -- which is how one slot shares while another does
    // not, in one process, with no third code path.
    gpu::CanonicalSlotState canonical{};
    /// Which regions a host consumer has asked to SEE this drive.  0 = nobody
    /// is looking, so the downloads are skipped and the Geometry arrays go
    /// deliberately stale.
    std::uint32_t canonical_materialize = 0u;
    unsigned long long canonical_uploads_elided   = 0;
    unsigned long long canonical_downloads_elided = 0;

    // --- Rev.7.1 W3 item 2: the DEFERRED DRAIN -----------------------------
    //
    // WHAT THE TERMINAL SYNCHRONISE IS FOR.  solveNodal ends in
    // cudaStreamSynchronize(d.stream) because the device outer enqueues
    // upddhat on the SEGMENT's stream on the very next line and that kernel
    // reads the jnet this drive produced.  Two streams, one handover: it has to
    // be ordered by a synchronise or an event, and it was the synchronise
    // (tools/test_device_outer_exactness_contract.py invariant 4).
    //
    // WHY IT CAN BECOME AN EVENT, BUT ONLY SOMETIMES.  The drain is ALSO what
    // makes the two downloads land before a host reader looks at Geometry::Jnet
    // and Geometry::Phis.  When the drive elided both of them -- canonical
    // buffers, materialize mask 0, i.e. inside a device outer segment -- nothing
    // came back and no host reader can be waiting.  The only consumer left is a
    // device kernel on another stream, and cudaStreamWaitEvent orders that
    // strictly and without a host round trip.  When either download ran, the
    // synchronise stays: an event would leave a D2H in flight into a page-locked
    // Geometry array the host is about to read, which is invariant 6's failure
    // one level down.
    //
    // ONE EVENT, CREATED ONCE, WITH DISABLE_TIMING.  A timing event forces a
    // clock read on both sides; nothing here measures anything.  It is recorded
    // at most once per drive and waited on at most once, so it can never be
    // recorded twice before it is consumed.
    cudaEvent_t nodal_done_event   = nullptr;
    bool        nodal_drain_deferred = false; ///< true = the last drive left the event pending

    [[nodiscard]] bool ensureNodalEvent() {
        if (nodal_done_event != nullptr) return true;
        if (cudaEventCreateWithFlags(&nodal_done_event, cudaEventDisableTiming) !=
            cudaSuccess) {
            cudaGetLastError();
            nodal_done_event = nullptr;
            return false;
        }
        return true;
    }

    // Both of these are BAKED INTO THE CAPTURE: the borrowed pointers are memcpy
    // operands, and the materialize mask decides which memcpy NODES exist at
    // all.  A graph captured under one and replayed under the other would move
    // the wrong bytes, or none, so they are part of the key rather than assumed
    // constant.
    /// Everything a captured nodal drive baked in, in one comparable object.
    ///
    /// It used to be sixteen `g_key_*` scalars beside ONE graph, and the pair
    /// behaved as "the graph, plus what it was for": a key that no longer
    /// matched meant the graph was destroyed.  For fifteen of these that is
    /// right -- they move when the geometry or the bound buffers move, and the
    /// old graph is then worthless.
    ///
    /// `materialize` is the sixteenth and it does not behave like the others.
    /// It takes exactly two values -- 0 inside a device outer segment, where
    /// nobody is looking and both downloads are elided, and `Jnet|Phis` outside
    /// it, where a host reader is about to touch both -- and it ALTERNATES,
    /// twice per segment, for the whole run.  With one slot that is a destroy
    /// and a re-capture at every segment boundary, and a re-capture is not free:
    /// it drains the backend's stream (a host rendezvous that no receipt counts,
    /// least of all the segment's own `in_body_host_syncs`), captures, and
    /// instantiates.  Measured on kngr_238 before this change: 3,282 captures
    /// against 3,214 segments and 12,041 launches -- one hidden rendezvous per
    /// segment, on the arm whose entire claim is that the host never looks.
    ///
    /// So the key stops being a validity test on one graph and becomes what it
    /// always described: an INDEX.  Two alternating values now cost two entries
    /// instead of two captures per segment, and the safety argument is
    /// unchanged, because it was never "the key matched" -- it was "this graph
    /// was captured under exactly these conditions", which is what a lookup
    /// establishes and an equality test only approximated.
    struct NodalGraphKey {
        const void*   ndev = nullptr;
        const void*   dblk = nullptr;
        const void*   jnet = nullptr;
        const void*   phis = nullptr;
        const void*   flux = nullptr;
        int           nxyz = 0;
        int           nsurf = 0;
        int           chif_empty = -1;
        const void*   canon_jnet = nullptr;
        const void*   canon_flux = nullptr;
        const void*   canon_phis = nullptr;
        std::uint32_t materialize = 0xFFFFFFFFu;
        int           owner_jnet = -1;
        int           owner_flux = -1;
        /// Rev.7.1 Task 10 part 3: the halt gate is a kernel argument, so it is
        /// a key.
        const void*   halt = reinterpret_cast<const void*>(~static_cast<std::uintptr_t>(0));
        int           halt_slot = -1;

        /// Written out rather than defaulted: this TU compiles as C++17 (no
        /// `= default` for operator==), and every field here is one a captured
        /// graph would silently move the wrong bytes without.
        bool operator==(const NodalGraphKey& o) const {
            return ndev == o.ndev && dblk == o.dblk && jnet == o.jnet &&
                   phis == o.phis && flux == o.flux && nxyz == o.nxyz &&
                   nsurf == o.nsurf && chif_empty == o.chif_empty &&
                   canon_jnet == o.canon_jnet && canon_flux == o.canon_flux &&
                   canon_phis == o.canon_phis && materialize == o.materialize &&
                   owner_jnet == o.owner_jnet && owner_flux == o.owner_flux &&
                   halt == o.halt && halt_slot == o.halt_slot;
        }
    };

    struct NodalGraph {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t     src  = nullptr;  ///< kept so the drive can be spliced
        NodalGraphKey   key;
    };

    /// SMALL BY CONSTRUCTION, AND CAPPED ANYWAY.  In a run whose geometry does
    /// not move, the only key field that changes is `materialize`, so the cache
    /// holds two entries.  The cap exists for the case that assumption is wrong:
    /// an unbounded cache of graphs nobody will ask for again is a leak with a
    /// polite name, and the fallback -- throw the lot away and start over -- is
    /// exactly the behaviour this replaced, so it cannot be worse than before.
    static constexpr std::size_t kNodalGraphCacheMax = 8;
    std::vector<NodalGraph> nodal_graphs;

    /// Point `nodal_graph` / `nodal_graph_src` at the entry captured under
    /// `want`, or at nothing.
    bool selectNodalGraph(const NodalGraphKey& want) {
        for (const NodalGraph& e : nodal_graphs)
            if (e.key == want) {
                nodal_graph     = e.exec;
                nodal_graph_src = e.src;
                return true;
            }
        nodal_graph     = nullptr;
        nodal_graph_src = nullptr;
        return false;
    }

    /// EVERY entry, not the selected one.
    ///
    /// All four callers are topology changes that invalidate the whole cache --
    /// ndev/dev_block reallocated, the canonical buffers re-adopted -- so
    /// "destroy the current graph" was only ever right because there was one.
    void dropNodalGraph() {
        for (const NodalGraph& e : nodal_graphs) {
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
            if (e.src != nullptr) cudaGraphDestroy(e.src);
        }
        nodal_graphs.clear();
        nodal_graph     = nullptr;
        nodal_graph_src = nullptr;
    }

    // --- batch arena (multi-instance nodal) -------------------------------
    // The arena is process-wide; this instance only holds the slot it was
    // handed.  -1 with `nodal_slot_refused` set means "asked once, refused"
    // (geometry mismatch or more instances than slots): never ask again, just
    // run the per-instance arm below.
    int  nodal_slot          = -1;
    bool nodal_slot_refused  = false;

    const FlatXsLibDevice* lib = nullptr;      // shared, process lifetime
    std::uint64_t          lib_hash_cached = 0; // host tables are immutable;
    const void*            lib_hash_key    = nullptr; // hash once per source

    // --- Rev.7.1 Task 13: the split Xe arm (RASBERY_GPU_XE) ---------------
    //
    // PER BACKEND, therefore per XSSet, therefore per Driver, therefore per
    // deck in a --batch-mode run.  Not one byte of this is static.
    double*             xe_hist      = nullptr; // [3][XE_TRIPLE_COUNT][n_fuel]
    unsigned char*      xe_processed = nullptr; // [n_fuel], the zero-flux skip
    double*             xe_partials  = nullptr; // [XE_DOT_COUNT * xe_parts]
    double*             xe_dots      = nullptr; // [XE_DOT_COUNT]
    int*                xe_pairs     = nullptr; // [2*XE_DOT_COUNT] then [XE_DOT_COUNT]
    int*                xe_flags     = nullptr; // [1] physics_bad
    unsigned long long* xe_bits      = nullptr; // [0] max/step, [1] step/nodes, [2] txn nodes
    int                 xe_parts     = 0;       // the FIXED partition count
    int                 xe_hist_fuel = 0;       // n_fuel the block was sized for
    // WP7-C.  The step's decision block, DEVICE-RESIDENT and PER-Impl -- which
    // is per XSSet, per Driver, per deck.  A file-scope one would be the batch
    // bug this tree already paid for once, with M decks writing one another's
    // acceptance.
    xek::XeTxnControl*  xe_ctl       = nullptr;
    xek::XeTxnControl   xe_ctl_host{}; // the single per-step download's landing pad

    // Offsets into dev_block, in doubles.
    std::size_t off_mic[xsr::NXS] = {};
    std::size_t off_mic_ssm       = 0;
    std::size_t off_lmp[xsr::NXS] = {};
    std::size_t off_lmp_ssm       = 0;
    std::size_t off_iden          = 0;
    std::size_t off_xs[xsr::NXS]  = {};
    std::size_t off_xs_ssm        = 0;
    std::size_t off_phif          = 0;

    ~Impl() {
        // Rev.7.1 Task 18d: a deck's teardown frees while the surviving decks
        // may be capturing, and cudaFree is a synchronising API.
        rasbery::AllocWindow _alloc_window("xsrecon.instance.release");
        // Hand the arena slot back BEFORE anything else: a lingering launcher
        // may be counting this instance among the participants it is waiting
        // for, and releaseSlot is what wakes it.
        if (nodal_slot >= 0 && g_nodal_arena != nullptr) {
            g_nodal_arena->releaseSlot(nodal_slot);
            nodal_slot = -1;
        }
        xeRelease();
        if (dev_block) cudaFree(dev_block);
        if (dev_fuel) cudaFree(dev_fuel);
        if (dev_scalars) cudaFree(dev_scalars);
        if (dev_dep) cudaFree(dev_dep);
        if (dev_ref) cudaFree(dev_ref);
        if (dev_pernode) cudaFree(dev_pernode);
        if (dev_nodes) cudaFree(dev_nodes);
        if (dev_off) cudaFree(dev_off);
        if (dev_cnt) cudaFree(dev_cnt);
        if (dev_sdid) cudaFree(dev_sdid);
        if (dev_sx) cudaFree(dev_sx);
        if (dev_sscale) cudaFree(dev_sscale);
        // THE CACHE, NOT THE SELECTION.  `nodal_graph` / `nodal_graph_src` are
        // aliases into nodal_graphs since Rev.7.1 Task 10 part 4, so destroying
        // them here would free one entry twice and leak the rest.
        dropNodalGraph();
        if (nodal_h_reigv) cudaFreeHost(nodal_h_reigv);
        // W3 item 2: one event per backend, created lazily on the first
        // deferred drain.  Destroyed beside the other lazily-created nodal
        // resources for the same reason they are: this destructor is the only
        // place that knows the backend is going away.
        if (nodal_done_event != nullptr) cudaEventDestroy(nodal_done_event);
        if (ndev_dbl) cudaFree(ndev_dbl);
        if (ndev_int) cudaFree(ndev_int);
        if (stream) cudaStreamDestroy(stream);
    }

    bool ensure(int want_nxyz, int want_fuel) {
        if (want_nxyz == nxyz && dev_block != nullptr && want_fuel <= n_fuel) {
            n_fuel = want_fuel;
            return true;
        }
        // The nodal graph baked dev_block-relative xs pointers into its kernel
        // nodes; this realloc moves them.
        dropNodalGraph();
        // The Xe history is sized on n_fuel and its addresses are handed to
        // kernels; a geometry change invalidates both.
        xeRelease();
        rasbery::AllocWindow _alloc_window("xsrecon.instance.regrow");
        if (dev_block) { cudaFree(dev_block); dev_block = nullptr; }
        if (dev_fuel) { cudaFree(dev_fuel); dev_fuel = nullptr; }
        if (dev_ref) { cudaFree(dev_ref); dev_ref = nullptr; }
        if (dev_pernode) { cudaFree(dev_pernode); dev_pernode = nullptr; }
        resident_micx_generation  = 0;
        resident_state_generation = 0;
        resident_ref_generation   = 0;
        fuel_uploaded             = false;

        nxyz   = want_nxyz;
        n_fuel = want_fuel;

        const std::size_t nx  = static_cast<std::size_t>(nxyz);
        const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
        const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
        const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
        const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

        std::size_t off = 0;
        for (int xt = 0; xt < xsr::NXS; ++xt) { off_mic[xt] = off; off += mic; }
        off_mic_ssm = off; off += msm;
        for (int xt = 0; xt < xsr::NXS; ++xt) { off_lmp[xt] = off; off += lmp; }
        off_lmp_ssm = off; off += ssm;
        off_iden = off; off += static_cast<std::size_t>(xsr::NISO) * nx;
        for (int xt = 0; xt < xsr::NXS; ++xt) { off_xs[xt] = off; off += lmp; }
        off_xs_ssm = off; off += ssm;
        off_phif = off; off += static_cast<std::size_t>(xsr::NG) * nx;
        block_doubles = off;

        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&dev_block),
                                          block_doubles * sizeof(double)), status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&dev_fuel),
                                          static_cast<std::size_t>(nxyz) * sizeof(int)), status);
        return true;
    }

    void xeRelease() {
        rasbery::AllocWindow _alloc_window("xsrecon.xe.release");
        if (xe_hist) { cudaFree(xe_hist); xe_hist = nullptr; }
        if (xe_processed) { cudaFree(xe_processed); xe_processed = nullptr; }
        if (xe_partials) { cudaFree(xe_partials); xe_partials = nullptr; }
        if (xe_dots) { cudaFree(xe_dots); xe_dots = nullptr; }
        if (xe_pairs) { cudaFree(xe_pairs); xe_pairs = nullptr; }
        if (xe_flags) { cudaFree(xe_flags); xe_flags = nullptr; }
        if (xe_bits) { cudaFree(xe_bits); xe_bits = nullptr; }
        if (xe_ctl) { cudaFree(xe_ctl); xe_ctl = nullptr; }
        xe_hist_fuel = 0;
        xe_parts     = 0;
    }

    /// Allocate the Xe arm's blocks for the current n_fuel.  Called from every
    /// entry point rather than from a setup hook, so the arm has no ordering
    /// requirement on the caller: the first thing that needs it makes it.
    bool xeEnsure() {
        if (xe_hist != nullptr && xe_hist_fuel == n_fuel) return true;
        xeRelease();
        if (n_fuel <= 0) return false;

        const std::size_t nf = static_cast<std::size_t>(n_fuel);
        xe_parts             = rasberyGpuXeDotPartitions();
        if (xe_parts > n_fuel) xe_parts = n_fuel; // empty partitions are pure waste
        if (xe_parts < 1) xe_parts = 1;

        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_hist),
                                          3 * static_cast<std::size_t>(xek::XE_TRIPLE_COUNT) *
                                              nf * sizeof(double)),
                               status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_processed), nf), status);
        RASBERY_CUDA_TRY_ALLOC(
            cudaMalloc(reinterpret_cast<void**>(&xe_partials),
                       static_cast<std::size_t>(xek::XE_DOT_COUNT) *
                           static_cast<std::size_t>(xe_parts) * sizeof(double)),
            status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_dots),
                                          xek::XE_DOT_COUNT * sizeof(double)),
                               status);
        // 3*XE_DOT_COUNT for the legacy arm's per-call upload, then TWO more
        // fixed layouts -- one per window width -- that the transaction uploads
        // ONCE here instead of twice per step.  Separate storage on purpose: the
        // legacy arm still writes its own region every call, and a shared one
        // would make the transaction's tables depend on who ran last.
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_pairs),
                                          (3 * xek::XE_DOT_COUNT +
                                           2 * XE_TXN_LAYOUT_INTS) * sizeof(int)),
                               status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_flags), sizeof(int)),
                               status);
        // THREE, not two.  The transaction needs its own commit counter: slot 0
        // still carries the residual the control kernel reads and slot 1 the
        // candidate's trust-region max, and both are live when the commit runs.
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_bits),
                                          3 * sizeof(unsigned long long)),
                               status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&xe_ctl),
                                          sizeof(xek::XeTxnControl)),
                               status);
        // Only the transaction reads them, and only the transaction should pay
        // the one-time synchronisation the upload needs -- so with TXN off the
        // round-tripping arm's allocation path is byte-for-byte what it was.
        if (rasberyGpuXeTxnEnabled() && !uploadXeTxnLayouts()) return false;
        // The history is READ before it is written on exactly one path: a
        // window column the arm never recorded.  The host guards that with
        // ncol, and so does this arm -- but a NaN sitting in an unwritten
        // column would turn a guard bug into a silent poisoning instead of an
        // obvious zero, so the block starts at zero.
        RASBERY_CUDA_TRY(cudaMemsetAsync(xe_hist, 0,
                                         3 * static_cast<std::size_t>(
                                                 xek::XE_TRIPLE_COUNT) *
                                             nf * sizeof(double),
                                         stream),
                         status);
        xe_hist_fuel = n_fuel;
        return true;
    }

    /// The uploads and the view repointing every kernel entry point needs:
    /// _micx/_lmpx on micx_generation, _xs/_iden on state_generation, phif per
    /// call, depTrans once per instance.
    ///
    /// FACTORED OUT OF solve(), which now calls it, so the residency contract
    /// is written once.  Two copies of this would be two opinions about when a
    /// 70 MB block is stale, and the arm that guessed wrong would compute
    /// physics against yesterday's cross sections.
    bool stage(const xsr::BatchView& host, unsigned long long micx_generation,
               unsigned long long state_generation, bool upload_phif,
               xsr::BatchView& v) {
        const std::size_t nx  = static_cast<std::size_t>(nxyz);
        const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
        const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
        const std::size_t msm =
            static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
        const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

        if (!fuel_uploaded) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(dev_fuel, host.fuel,
                                             static_cast<std::size_t>(host.n_fuel) *
                                                 sizeof(int),
                                             cudaMemcpyHostToDevice, stream),
                             status);
            fuel_uploaded = true;
        }

        // _micx and _lmpx move together (both are outputs of the same host-side
        // rebuild paths), so one generation covers both.
        if (micx_generation != resident_micx_generation) {
            for (int xt = 0; xt < xsr::NXS; ++xt)
                if (!upload(host.mic[xt], off_mic[xt], mic)) return false;
            if (!upload(host.mic_ssm, off_mic_ssm, msm)) return false;
            for (int xt = 0; xt < xsr::NXS; ++xt)
                if (!upload(host.lmp[xt], off_lmp[xt], lmp)) return false;
            if (!upload(host.lmp_ssm, off_lmp_ssm, ssm)) return false;
            resident_micx_generation = micx_generation;
        }

        // Per-call state.  _iden and _xs are uploaded whole so the kernel's
        // fuel-only writes round-trip the non-fuel entries unchanged, keeping
        // the host arrays authoritative for every node after the download.
        // While the host-state generation matches the resident copy, the host
        // has not written _xs/_iden since our last download, so both are
        // already bit-identical on the device and the ~4.4 MB re-upload is
        // skipped.
        if (state_generation != resident_state_generation) {
            if (!upload(host.iden, off_iden, static_cast<std::size_t>(xsr::NISO) * nx))
                return false;
            for (int xt = 0; xt < xsr::NXS; ++xt)
                if (!upload(host.xs[xt], off_xs[xt], lmp)) return false;
            if (!upload(host.xs_ssm, off_xs_ssm, ssm)) return false;
        }
        if (upload_phif &&
            !upload(host.phif, off_phif, static_cast<std::size_t>(xsr::NG) * nx))
            return false;

        v = host;
        for (int xt = 0; xt < xsr::NXS; ++xt) {
            v.mic[xt] = dev_block + off_mic[xt];
            v.lmp[xt] = dev_block + off_lmp[xt];
            v.xs[xt]  = dev_block + off_xs[xt];
        }
        v.mic_ssm = dev_block + off_mic_ssm;
        v.lmp_ssm = dev_block + off_lmp_ssm;
        v.iden    = dev_block + off_iden;
        v.xs_ssm  = dev_block + off_xs_ssm;
        v.phif    = dev_block + off_phif;
        v.fuel    = dev_fuel;

        // depTrans rows: 39 doubles each, constant for the process, so they are
        // uploaded exactly once per instance.  With 100 Xe calls per case the
        // two per-call copies were pure API-call overhead (nsys: memcpy CALL
        // COUNT, not payload, dominates the timeline).
        if (dev_dep == nullptr) {
            RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&dev_dep),
                                              2 * xsr::NISO * sizeof(double)),
                                   status);
            RASBERY_CUDA_TRY(cudaMemcpyAsync(dev_dep, host.dep_i135,
                                             xsr::NISO * sizeof(double),
                                             cudaMemcpyHostToDevice, stream),
                             status);
            RASBERY_CUDA_TRY(cudaMemcpyAsync(dev_dep + xsr::NISO, host.dep_xe135,
                                             xsr::NISO * sizeof(double),
                                             cudaMemcpyHostToDevice, stream),
                             status);
        }
        v.dep_i135  = dev_dep;
        v.dep_xe135 = dev_dep + xsr::NISO;
        return true;
    }

    /// The xs + Xe-chain-iden download every committing path owes the host.
    /// Every stream synchronisation the Xe path takes, counted where it is
    /// taken.  The WP7-C census is a RECEIPT and not a paragraph: `host_syncs`
    /// divided by `xe_device_steps` is the number the doc's before/after table
    /// quotes, measured by the run that quotes it.
    cudaError_t xeSync() {
        xe::xeGpuTally().host_syncs.fetch_add(1, std::memory_order_relaxed);
        return cudaStreamSynchronize(stream);
    }

    static void countXeD2H(std::size_t bytes) {
        xe::xeGpuTally().d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    /// The two fixed dot layouts, uploaded once per allocation.  WHICH PRODUCTS
    /// THE HOST ACTUALLY READS is decided exactly as XsReconBackend::xeDots
    /// decides it -- and the two must stay the same decision, so the table is
    /// built by the same `add` sequence, spelled once here and read from there.
    bool uploadXeTxnLayouts() {
        int host[2 * XE_TXN_LAYOUT_INTS] = {};
        for (int ncol = 1; ncol <= xek::XE_DEPTH; ++ncol) {
            int* pairs  = host + (ncol - 1) * XE_TXN_LAYOUT_INTS;
            int* slots  = pairs + 2 * xek::XE_DOT_COUNT;
            int  npairs = 0;
            auto add    = [&](int slot, int left, int right) {
                pairs[2 * npairs]     = left;
                pairs[2 * npairs + 1] = right;
                slots[npairs]         = slot;
                ++npairs;
            };
            add(xek::XE_DOT_GG, xek::XE_T_G, xek::XE_T_G);
            add(xek::XE_DOT_A, xek::XE_T_DG0, xek::XE_T_DG0);
            add(xek::XE_DOT_P, xek::XE_T_DG0, xek::XE_T_G);
            if (ncol == xek::XE_DEPTH) {
                add(xek::XE_DOT_B, xek::XE_T_DG0, xek::XE_T_DG1);
                add(xek::XE_DOT_C, xek::XE_T_DG1, xek::XE_T_DG1);
                add(xek::XE_DOT_Q, xek::XE_T_DG1, xek::XE_T_G);
            }
        }
        RASBERY_CUDA_TRY(cudaMemcpyAsync(xe_pairs + 3 * xek::XE_DOT_COUNT, host,
                                         sizeof(host), cudaMemcpyHostToDevice, stream),
                         status);
        // The layouts are read by kernels this call does not order against, so
        // the upload is completed here rather than left in flight.  Once per
        // allocation, never per step.
        RASBERY_CUDA_TRY(cudaStreamSynchronize(stream), status);
        return true;
    }

    static int xeTxnPairCount(int ncol) { return (ncol == xek::XE_DEPTH) ? 6 : 3; }

    bool drainXeCommit(const xsr::BatchView& host) {
        const std::size_t nx  = static_cast<std::size_t>(nxyz);
        const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
        const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!download(host.xs[xt], off_xs[xt], lmp)) return false;
        if (!download(host.xs_ssm, off_xs_ssm, ssm)) return false;
        if (!download(host.iden + static_cast<std::size_t>(xsr::I135) * nx,
                      off_iden + static_cast<std::size_t>(xsr::I135) * nx, 3 * nx))
            return false;
        countXeD2H((static_cast<std::size_t>(xsr::NXS) * lmp + ssm + 3 * nx) *
                   sizeof(double));
        return true;
    }

    bool upload(const double* src, std::size_t off, std::size_t count) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(dev_block + off, src, count * sizeof(double),
                                         cudaMemcpyHostToDevice, stream), status);
        return true;
    }

    bool download(double* dst, std::size_t off, std::size_t count) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(dst, dev_block + off, count * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream), status);
        return true;
    }
};

XsReconBackend::XsReconBackend() : _impl(std::make_unique<Impl>()) {
    if (!rasberyGpuXsReconEnabled() && !rasberyGpuFlatXsEnabled() &&
        !rasberyGpuNodalEnabled() && !rasberyGpuXeEnabled()) {
        _impl->status =
            "disabled (RASBERY_GPU_XSRECON/RASBERY_GPU_FLATXS/RASBERY_GPU_XE unset)";
        return;
    }
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        _impl->status = std::string("no CUDA device: ") +
                        (e == cudaSuccess ? "count 0" : cudaGetErrorString(e));
        return;
    }
    const cudaError_t se = cudaStreamCreateWithFlags(&_impl->stream, cudaStreamNonBlocking);
    if (se != cudaSuccess) {
        _impl->status = std::string("stream: ") + cudaGetErrorString(se);
        return;
    }
    rasbery::AllocWindow _scalars_window("xsrecon.scalars");
    if (cudaMalloc(reinterpret_cast<void**>(&_impl->dev_scalars),
                   2 * sizeof(unsigned long long)) != cudaSuccess) {
        _impl->status = "scalar buffer allocation failed";
        return;
    }
    _impl->available = true;
    _impl->status    = "ready";
}

XsReconBackend::~XsReconBackend() = default;

bool XsReconBackend::available() const { return _impl->available; }
const std::string& XsReconBackend::status() const { return _impl->status; }

bool XsReconBackend::solve(const xsr::BatchView& host, unsigned long long micx_generation,
                           unsigned long long state_generation, double* max_change_out) {
    Impl& d = *_impl;
    if (!d.available || host.n_fuel <= 0 || host.nxyz <= 0) return false;
    if (!d.ensure(host.nxyz, host.n_fuel)) {
        d.available = false; // one allocation failure disables the instance
        return false;
    }

    // Uploads and view repointing: Impl::stage, which is this function's own
    // former body -- the Task 13 arm needs the identical residency contract and
    // two copies of it would be two opinions about when a 70 MB block is stale.
    xsr::BatchView v{};
    if (!d.stage(host, micx_generation, state_generation, true, v)) return false;

    RASBERY_CUDA_TRY(cudaMemsetAsync(d.dev_scalars, 0, 2 * sizeof(unsigned long long),
                                     d.stream), d.status);

    const int block = 128;
    const int grid  = (host.n_fuel + block - 1) / block;
    kernelXsRecon<<<grid, block, 0, d.stream>>>(v, d.dev_scalars, d.dev_scalars + 1);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    // Results the host needs back: the reconstructed xs, the three Xe-chain
    // density rows (contiguous, iso-major), and the two scalars.
    if (!d.drainXeCommit(host)) return false;

    unsigned long long scalars[2] = {0, 0};
    RASBERY_CUDA_TRY(cudaMemcpyAsync(scalars, d.dev_scalars,
                                     2 * sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    double max_change;
    static_assert(sizeof(max_change) == sizeof(scalars[0]), "bit width");
    std::memcpy(&max_change, &scalars[0], sizeof(max_change));
    *max_change_out = max_change;

    // The downloads above just made host _xs and the Xe-chain _iden rows
    // equal to the device copy again; nothing else on the host wrote since
    // state_generation was read.
    d.resident_state_generation = state_generation;

    g_nodes_solved.fetch_add(scalars[1], std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 13 -- the split Xe arm
// ---------------------------------------------------------------------------
//
// Six entry points, and between them the Anderson history NEVER LEAVES THE
// DEVICE.  That is the whole reason the split is worth having: the host arm
// carries ten triples of 3*n_fuel doubles (about 2 MB on kngr_238) and touches
// all of them every step, and moving that across the bus twice per step would
// cost more than the algebra it is accelerating.  What crosses instead is
// eight doubles and a flag.
//
// EVERY ONE FAILS OPEN.  A false return means "the host path must run", and it
// is returned before anything the host can observe has been touched -- the
// uploads copy host->device only, and the one function that writes host memory
// (xeCommit) does so after its kernel has already succeeded.

bool XsReconBackend::xeEvaluate(const xsr::BatchView& host,
                                unsigned long long micx_generation,
                                unsigned long long state_generation,
                                double* picard_out) {
    Impl& d = *_impl;
    if (!d.available || host.n_fuel <= 0 || host.nxyz <= 0) return false;
    if (!d.ensure(host.nxyz, host.n_fuel)) {
        d.available = false;
        return false;
    }
    if (!d.xeEnsure()) {
        d.available = false;
        return false;
    }

    xsr::BatchView v{};
    if (!d.stage(host, micx_generation, state_generation, true, v)) return false;

    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_bits, 0, 2 * sizeof(unsigned long long),
                                     d.stream),
                     d.status);

    const int block = 128;
    const int grid  = (host.n_fuel + block - 1) / block;
    kXeEvaluate<<<grid, block, 0, d.stream>>>(v, d.xe_hist, d.xe_processed, d.xe_bits);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    unsigned long long bits = 0;
    RASBERY_CUDA_TRY(cudaMemcpyAsync(&bits, d.xe_bits, sizeof(bits),
                                     cudaMemcpyDeviceToHost, d.stream),
                     d.status);
    Impl::countXeD2H(sizeof(bits));
    RASBERY_CUDA_TRY(d.xeSync(), d.status);

    double picard;
    static_assert(sizeof(picard) == sizeof(bits), "bit width");
    std::memcpy(&picard, &bits, sizeof(picard));
    *picard_out = picard;

    // NOTHING WAS WRITTEN on the host, so the host arrays and the device copy
    // still agree exactly as the staging left them -- and the staging is what
    // brought them into agreement, so the residency advances here even though
    // no download ran.  Missing this would re-upload _xs and _iden on every
    // step of a cascade that never dirtied them.
    d.resident_state_generation = state_generation;

    g_xe_evaluations.fetch_add(static_cast<unsigned long long>(host.n_fuel),
                               std::memory_order_relaxed);
    return true;
}

bool XsReconBackend::xeRotateHistory() {
    Impl& d = *_impl;
    if (!d.available || d.xe_hist == nullptr || d.xe_hist_fuel <= 0) return false;
    // df[0] <- df[1] and dg[0] <- dg[1], one row at a time: the block is
    // [row][triple][ordinal], so a triple's three rows are not contiguous and a
    // single copy would drag nine other triples along with them.
    const std::size_t bytes = static_cast<std::size_t>(d.xe_hist_fuel) * sizeof(double);
    const int         pairs[2][2] = {{xek::XE_T_DF1, xek::XE_T_DF0},
                                     {xek::XE_T_DG1, xek::XE_T_DG0}};
    for (const auto& pr : pairs)
        for (int row = 0; row < 3; ++row) {
            double* dst = const_cast<double*>(xeRow(d.xe_hist, row, pr[1], d.xe_hist_fuel));
            const double* src = xeRow(d.xe_hist, row, pr[0], d.xe_hist_fuel);
            RASBERY_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                             d.stream),
                             d.status);
        }
    return true;
}

bool XsReconBackend::xeRecordColumn(int col) {
    Impl& d = *_impl;
    if (!d.available || d.xe_hist == nullptr || d.xe_hist_fuel <= 0) return false;
    if (col < 0 || col >= xek::XE_DEPTH) return false;

    const int block = 256;
    const int grid  = (d.xe_hist_fuel + block - 1) / block;
    kXeSub<<<grid, block, 0, d.stream>>>(d.xe_hist, xek::XE_T_F, xek::XE_T_F_PREV,
                                         xek::XE_T_DF0 + col, d.xe_hist_fuel);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    kXeSub<<<grid, block, 0, d.stream>>>(d.xe_hist, xek::XE_T_G, xek::XE_T_G_PREV,
                                         xek::XE_T_DG0 + col, d.xe_hist_fuel);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    return true;
}

bool XsReconBackend::xeSaveEvaluation() {
    Impl& d = *_impl;
    if (!d.available || d.xe_hist == nullptr || d.xe_hist_fuel <= 0) return false;
    const std::size_t bytes = static_cast<std::size_t>(d.xe_hist_fuel) * sizeof(double);
    const int         pairs[2][2] = {{xek::XE_T_F, xek::XE_T_F_PREV},
                                     {xek::XE_T_G, xek::XE_T_G_PREV}};
    for (const auto& pr : pairs)
        for (int row = 0; row < 3; ++row) {
            double* dst = const_cast<double*>(xeRow(d.xe_hist, row, pr[1], d.xe_hist_fuel));
            const double* src = xeRow(d.xe_hist, row, pr[0], d.xe_hist_fuel);
            RASBERY_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                             d.stream),
                             d.status);
        }
    return true;
}

bool XsReconBackend::xeDots(int ncol, double* out_six) {
    Impl& d = *_impl;
    if (!d.available || d.xe_hist == nullptr || d.xe_hist_fuel <= 0) return false;
    if (ncol < 1 || ncol > xek::XE_DEPTH) return false;

    // WHICH PRODUCTS THE HOST ACTUALLY READS, and no others.  With a full
    // window it reads all six.  With one column it reads <g,g>, <dg0,dg0> and
    // <dg0,g> -- and its one-column fallback takes the NEWEST column, which at
    // ncol == 1 is dg0, so those three cover both branches.  Computing the rest
    // would read triples no evaluation has written; leaving them at zero says
    // "not measured" instead of manufacturing a number.
    int host_pairs[2 * xek::XE_DOT_COUNT] = {};
    int host_slots[xek::XE_DOT_COUNT]     = {};
    int npairs                            = 0;
    auto add = [&](int slot, int left, int right) {
        host_pairs[2 * npairs]     = left;
        host_pairs[2 * npairs + 1] = right;
        host_slots[npairs]         = slot;
        ++npairs;
    };
    add(xek::XE_DOT_GG, xek::XE_T_G, xek::XE_T_G);
    add(xek::XE_DOT_A, xek::XE_T_DG0, xek::XE_T_DG0);
    add(xek::XE_DOT_P, xek::XE_T_DG0, xek::XE_T_G);
    if (ncol == xek::XE_DEPTH) {
        add(xek::XE_DOT_B, xek::XE_T_DG0, xek::XE_T_DG1);
        add(xek::XE_DOT_C, xek::XE_T_DG1, xek::XE_T_DG1);
        add(xek::XE_DOT_Q, xek::XE_T_DG1, xek::XE_T_G);
    }

    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.xe_pairs, host_pairs,
                                     static_cast<std::size_t>(2 * npairs) * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream),
                     d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.xe_pairs + 2 * xek::XE_DOT_COUNT, host_slots,
                                     static_cast<std::size_t>(npairs) * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream),
                     d.status);
    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_dots, 0, xek::XE_DOT_COUNT * sizeof(double),
                                     d.stream),
                     d.status);

    const int block1 = 128;
    const int total  = npairs * d.xe_parts;
    kXeDotStage1<<<(total + block1 - 1) / block1, block1, 0, d.stream>>>(
        d.xe_hist, d.xe_hist_fuel, d.xe_parts, npairs, d.xe_pairs, d.xe_partials,
        xe::xeFormMask());
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    kXeDotStage2<<<1, xek::XE_DOT_COUNT, 0, d.stream>>>(
        d.xe_parts, npairs, d.xe_partials, d.xe_pairs + 2 * xek::XE_DOT_COUNT,
        d.xe_dots);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    RASBERY_CUDA_TRY(cudaMemcpyAsync(out_six, d.xe_dots,
                                     xek::XE_DOT_COUNT * sizeof(double),
                                     cudaMemcpyDeviceToHost, d.stream),
                     d.status);
    Impl::countXeD2H(xek::XE_DOT_COUNT * sizeof(double));
    RASBERY_CUDA_TRY(d.xeSync(), d.status);
    return true;
}

bool XsReconBackend::xeCandidate(const double* gamma, int ncol, double* step_out,
                                 bool* physics_ok) {
    Impl& d = *_impl;
    if (!d.available || d.xe_hist == nullptr || d.xe_hist_fuel <= 0) return false;
    if (ncol < 1 || ncol > xek::XE_DEPTH) return false;

    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_flags, 0, sizeof(int), d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_bits, 0, sizeof(unsigned long long), d.stream),
                     d.status);

    const int block = 256;
    const int grid  = (d.xe_hist_fuel + block - 1) / block;
    kXeCandidate<<<grid, block, 0, d.stream>>>(d.xe_hist, d.xe_hist_fuel, ncol, gamma[0],
                                               gamma[1], xe::xeFormMask(), d.xe_flags,
                                               d.xe_bits);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    int                bad  = 0;
    unsigned long long bits = 0;
    RASBERY_CUDA_TRY(cudaMemcpyAsync(&bad, d.xe_flags, sizeof(int),
                                     cudaMemcpyDeviceToHost, d.stream),
                     d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(&bits, d.xe_bits, sizeof(bits),
                                     cudaMemcpyDeviceToHost, d.stream),
                     d.status);
    Impl::countXeD2H(sizeof(bad) + sizeof(bits));
    RASBERY_CUDA_TRY(d.xeSync(), d.status);

    double step;
    std::memcpy(&step, &bits, sizeof(step));
    *step_out   = step;
    *physics_ok = (bad == 0);
    return true;
}

bool XsReconBackend::xeCommit(const xsr::BatchView& host, int triple, double relax,
                              bool picard_skip, unsigned long long state_generation) {
    Impl& d = *_impl;
    if (!d.available || host.n_fuel <= 0 || host.nxyz <= 0) return false;
    if (d.xe_hist == nullptr || d.xe_hist_fuel != host.n_fuel) return false;

    // A commit is only ever reached through an evaluate that staged _micx and
    // _lmpx; if that has not happened, the resident generation is still 0 and
    // passing it to stage() would read "already resident" and run the kernel
    // against an empty block.  Refuse instead, and let the host path run.
    if (d.resident_micx_generation == 0) return false;

    // The evaluate that produced this image already staged everything; passing
    // state_generation again is what makes that explicit rather than assumed --
    // if the host DID write in between, the re-upload happens here and the
    // commit runs against what the host actually holds.  phif is not uploaded:
    // a commit reads no flux.
    xsr::BatchView v{};
    if (!d.stage(host, d.resident_micx_generation, state_generation, false, v))
        return false;

    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_bits + 1, 0, sizeof(unsigned long long),
                                     d.stream),
                     d.status);

    const int block = 128;
    const int grid  = (host.n_fuel + block - 1) / block;
    kXeCommit<<<grid, block, 0, d.stream>>>(v, d.xe_hist, triple, relax,
                                            picard_skip ? 1 : 0, d.xe_processed,
                                            d.xe_bits + 1);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    if (!d.drainXeCommit(host)) return false;

    unsigned long long solved = 0;
    RASBERY_CUDA_TRY(cudaMemcpyAsync(&solved, d.xe_bits + 1, sizeof(solved),
                                     cudaMemcpyDeviceToHost, d.stream),
                     d.status);
    Impl::countXeD2H(sizeof(solved));
    RASBERY_CUDA_TRY(d.xeSync(), d.status);
    xe::xeGpuTally().xe_device_steps.fetch_add(1, std::memory_order_relaxed);

    // Same contract solve() has: the downloads just made the host arrays equal
    // to the device copy, so the caller must NOT bump its host-state generation
    // -- the two agree, and a bump would buy a 4.4 MB re-upload for nothing.
    d.resident_state_generation = state_generation;

    g_xe_commits.fetch_add(solved, std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------------------
// WP7 stage C -- one Xe step as one device transaction
// ---------------------------------------------------------------------------
//
// THE CENSUS THIS REPLACES.  A step on the round-tripping arm is
//
//     xeEvaluate  -> kernel, D2H 8 B,  SYNC        (is it armed?)
//     rotate/record/save -> 12 D2D copies + 2 kernels
//     xeDots      -> 2 H2D, kernel, kernel, D2H 48 B, SYNC   (did it condition?)
//     xeCandidate -> 2 memset, kernel, D2H 12 B, SYNC        (did it pass?)
//     xeCommit    -> kernel, drain D2H, D2H 8 B, SYNC
//
// -- four synchronisations, three of them paying for a decision that is eight
// doubles wide.  This is the same work with those three decisions inside:
//
//     kXeEvaluate -> kXeHistory -> kXeDotStage1/2 -> kXeAndersonSolve
//                 -> kXeCandidateTxn -> kXeAndersonGate -> kXeCommitTxn
//     -> drain D2H + one XeTxnControl, ONE SYNC
//
// The remaining sync is the drain's, and it is not the transaction's to remove:
// the host arrays are authoritative for everything downstream of an Xe step on
// this tree, so `_xs` and the three `_iden` rows have to be back before the
// caller reads them.  Plan Sec WP7-C makes that conditional on device residency
// -- "state arrays가 device resident이면 중간 D2H를 생략한다" -- and residency
// is not this work package.  What IS removed is every sync that was not paying
// for a materialisation.
//
// FAILS OPEN, LIKE THE SIX ENTRY POINTS ABOVE, and it fails open EARLY: every
// refusal below happens before a kernel that writes host-visible state has been
// launched, so the caller can run the round-tripping arm on an untouched
// solver.  A refusal after the commit kernel would not be recoverable, and
// there is none.
bool XsReconBackend::xeTransaction(const xsr::BatchView& host,
                                   unsigned long long micx_generation,
                                   unsigned long long state_generation,
                                   const XeTxnRequest& req, xe::XeTxnControl* out) {
    Impl& d = *_impl;
    if (!d.available || host.n_fuel <= 0 || host.nxyz <= 0) return false;
    if (req.ncol < 0 || req.ncol > xek::XE_DEPTH) return false;
    if (req.hist_col >= xek::XE_DEPTH) return false;
    // relax is 1.0 on the Anderson path by construction (Driver.h arms the
    // attempt with `xe_relax == 1.0`).  Refusing anything else is not
    // defensiveness: the rejected branch commits the damped Picard image, and
    // if the caller ever damps here the two arms would commit different images
    // and the B0 claim would be silently false.
    if (!(req.relax == 1.0)) return false;
    if (!d.ensure(host.nxyz, host.n_fuel)) {
        d.available = false;
        return false;
    }
    if (!d.xeEnsure()) {
        d.available = false;
        return false;
    }
    if (d.xe_ctl == nullptr) return false;

    xsr::BatchView v{};
    if (!d.stage(host, micx_generation, state_generation, true, v)) return false;

    const unsigned long long forms = xe::xeFormMask();

    // 1. Evaluate.  x, F(x), g into the device history; the residual reduces
    //    through the same exact atomicMax it always did.
    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_bits, 0, 3 * sizeof(unsigned long long),
                                     d.stream),
                     d.status);
    {
        const int block = 128;
        const int grid  = (host.n_fuel + block - 1) / block;
        kXeEvaluate<<<grid, block, 0, d.stream>>>(v, d.xe_hist, d.xe_processed,
                                                  d.xe_bits);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    }
    g_xe_evaluations.fetch_add(static_cast<unsigned long long>(host.n_fuel),
                               std::memory_order_relaxed);

    // 2. History: rotate, record, save -- one launch, the same numbers.
    {
        const int block = 256;
        const int grid  = (d.xe_hist_fuel + block - 1) / block;
        kXeHistory<<<grid, block, 0, d.stream>>>(d.xe_hist, d.xe_hist_fuel,
                                                 req.hist_col, req.hist_rotate ? 1 : 0);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    }

    // 3. The six inner products, fixed partition, unchanged.  The layouts were
    //    uploaded once at xeEnsure; nothing about them is per step.
    const int ncol   = (req.ncol < 1) ? 1 : req.ncol;
    const int npairs = Impl::xeTxnPairCount(ncol);
    const int* layout = d.xe_pairs + 3 * xek::XE_DOT_COUNT +
                        (ncol - 1) * XE_TXN_LAYOUT_INTS;
    RASBERY_CUDA_TRY(cudaMemsetAsync(d.xe_dots, 0, xek::XE_DOT_COUNT * sizeof(double),
                                     d.stream),
                     d.status);
    {
        const int block1 = 128;
        const int total  = npairs * d.xe_parts;
        kXeDotStage1<<<(total + block1 - 1) / block1, block1, 0, d.stream>>>(
            d.xe_hist, d.xe_hist_fuel, d.xe_parts, npairs, layout, d.xe_partials, forms);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
        kXeDotStage2<<<1, xek::XE_DOT_COUNT, 0, d.stream>>>(
            d.xe_parts, npairs, d.xe_partials, layout + 2 * xek::XE_DOT_COUNT,
            d.xe_dots);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    }

    // 4. The decision the host used to make, one thread, same order.
    //    `req.ncol` and NOT the clamped `ncol`: a zero window is the arming
    //    test's business, and the clamp above exists only so the dot layout
    //    index is in range.
    kXeAndersonSolve<<<1, 1, 0, d.stream>>>(d.xe_dots, req.ncol, d.xe_bits, req.eq_tol,
                                            req.min_gram, forms, d.xe_ctl, d.xe_flags,
                                            d.xe_bits + 1);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    // 5. The candidate, plus SAFEGUARD 3/4 and the trust-region metric.
    {
        const int block = 256;
        const int grid  = (d.xe_hist_fuel + block - 1) / block;
        kXeCandidateTxn<<<grid, block, 0, d.stream>>>(d.xe_hist, d.xe_hist_fuel,
                                                      d.xe_ctl, forms, d.xe_flags,
                                                      d.xe_bits + 1);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    }
    kXeAndersonGate<<<1, 1, 0, d.stream>>>(d.xe_ctl, d.xe_flags, d.xe_bits + 1,
                                           req.max_step);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    // 6. Commit -- the candidate if it survived, the Picard image if it did
    //    not.  The kernel reads which from the control block; the host does not
    //    learn which until the download below, and does not need to.
    {
        const int block = 128;
        const int grid  = (host.n_fuel + block - 1) / block;
        kXeCommitTxn<<<grid, block, 0, d.stream>>>(v, d.xe_hist, d.xe_ctl, req.relax,
                                                   d.xe_processed, d.xe_bits + 2);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);
    }

    // 7. THE ONE HOST OBSERVATION.  The drain the caller needs anyway, and the
    //    control block riding on the same transfer, cleared by the same sync.
    //
    //    THE FAIL-OPEN PROMISE ENDS HERE, AND THAT HAS TO BE SAID OUT LOUD.
    //    Every refusal above happens before a kernel that writes host-visible
    //    state; a failure below is a CUDA error AFTER the commit kernel, so the
    //    device inventory has already moved and letting the round-tripping arm
    //    take the step again would commit it TWICE.  The instance is retired
    //    instead: `available = false` makes every later device call decline, so
    //    the run continues on the host loop, which is the only arm that cannot
    //    double-commit.  (xeCommit has carried the same exposure since it
    //    shipped, and did not say so.)
    unsigned long long solved = 0;
    if (!d.drainXeCommit(host) ||
        cudaMemcpyAsync(&d.xe_ctl_host, d.xe_ctl, sizeof(xek::XeTxnControl),
                        cudaMemcpyDeviceToHost, d.stream) != cudaSuccess ||
        cudaMemcpyAsync(&solved, d.xe_bits + 2, sizeof(solved), cudaMemcpyDeviceToHost,
                        d.stream) != cudaSuccess ||
        d.xeSync() != cudaSuccess) {
        d.status    = "xeTransaction: download failed after the commit kernel";
        d.available = false;
        return false;
    }
    Impl::countXeD2H(sizeof(xek::XeTxnControl) + sizeof(solved));

    // Same contract xeCommit keeps: the downloads made the host arrays equal to
    // the device copy, so the caller must NOT bump its host-state generation.
    d.resident_state_generation = state_generation;
    *out                        = d.xe_ctl_host;

    g_xe_commits.fetch_add(solved, std::memory_order_relaxed);
    {
        xe::XeGpuTally& tally = xe::xeGpuTally();
        tally.xe_device_steps.fetch_add(1, std::memory_order_relaxed);
        tally.txn_steps.fetch_add(1, std::memory_order_relaxed);
        if (out->accept) tally.txn_accepted.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

unsigned long long XsReconBackend::xeEvaluations() {
    return g_xe_evaluations.load(std::memory_order_relaxed);
}

unsigned long long XsReconBackend::xeCommits() {
    return g_xe_commits.load(std::memory_order_relaxed);
}

bool XsReconBackend::solveFlatXs(const fxs::FlatXsView& host,
                                 const FlatXsLibShape& shape,
                                 unsigned long long micx_generation,
                                 unsigned long long micx_generation_next,
                                 unsigned long long ref_generation,
                                 unsigned long long state_generation,
                                 bool mark_micx_resident) {
    Impl& d = *_impl;
    if (!d.available || host.n_nodes <= 0 || host.nxyz <= 0) return false;
    // Reuse ensure() so the mic/lmp/xs/iden regions exist; keep the resident
    // fuel-count contract of the xsrecon solve intact by passing its current
    // value (or the node count on first contact).
    if (!d.ensure(host.nxyz, d.n_fuel > 0 ? d.n_fuel : host.n_nodes)) {
        d.available = false;
        return false;
    }

    const std::size_t nx  = static_cast<std::size_t>(d.nxyz);
    const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

    // --- shared library tables (upload once per distinct content) ---------
    if (d.lib == nullptr || d.lib_hash_key != static_cast<const void*>(host.coeff_lsm)) {
        std::uint64_t h = 1469598103934665603ULL;
        h = fnvMix(&shape, sizeof shape, h);
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            h = fnvMix(host.coeff_lmp[t], shape.lmp_slot * sizeof(double), h);
        h = fnvMix(host.coeff_lsm, shape.lsm * sizeof(double), h);
        if (host.has_coeff_micx) {
            for (int t = 0; t < fxs::N_ACTIVE; ++t)
                h = fnvMix(host.coeff_mic[t], shape.mic_slot * sizeof(double), h);
            h = fnvMix(host.coeff_msm, shape.msm * sizeof(double), h);
        }
        h = fnvMix(host.knots, shape.n_knots * sizeof(double), h);
        h = fnvMix(host.deltas, shape.n_deltas * sizeof(fxs::DeltaMeta), h);
        d.lib_hash_cached = h;
        d.lib_hash_key    = host.coeff_lsm;

        std::lock_guard<std::mutex> lock(g_flatxs_lib_mutex);
        if (g_flatxs_libs == nullptr) g_flatxs_libs = new std::deque<FlatXsLibDevice>;
        const FlatXsLibDevice* found = nullptr;
        for (const auto& e : *g_flatxs_libs)
            if (e.hash == h) { found = &e; break; }
        if (found == nullptr) {
            FlatXsLibDevice e;
            e.hash = h;
            std::size_t off = 0;
            for (int t = 0; t < fxs::N_ACTIVE; ++t) { e.off_lmp[t] = off; off += shape.lmp_slot; }
            e.off_lsm = off; off += shape.lsm;
            for (int t = 0; t < fxs::N_ACTIVE; ++t) { e.off_mic[t] = off; off += shape.mic_slot; }
            e.off_msm = off; off += shape.msm;
            e.off_knots = off; off += shape.n_knots;
            RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&e.block),
                                              off * sizeof(double)), d.status);
            RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&e.deltas),
                                              shape.n_deltas * sizeof(fxs::DeltaMeta)), d.status);
            // Synchronous copies under the mutex: nothing can race a
            // half-uploaded table, and this happens once per process.
            for (int t = 0; t < fxs::N_ACTIVE; ++t)
                RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_lmp[t], host.coeff_lmp[t],
                                            shape.lmp_slot * sizeof(double),
                                            cudaMemcpyHostToDevice), d.status);
            RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_lsm, host.coeff_lsm,
                                        shape.lsm * sizeof(double),
                                        cudaMemcpyHostToDevice), d.status);
            if (host.has_coeff_micx) {
                for (int t = 0; t < fxs::N_ACTIVE; ++t)
                    RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_mic[t], host.coeff_mic[t],
                                                shape.mic_slot * sizeof(double),
                                                cudaMemcpyHostToDevice), d.status);
                RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_msm, host.coeff_msm,
                                            shape.msm * sizeof(double),
                                            cudaMemcpyHostToDevice), d.status);
            }
            if (shape.n_knots > 0)
                RASBERY_CUDA_TRY(cudaMemcpy(e.block + e.off_knots, host.knots,
                                            shape.n_knots * sizeof(double),
                                            cudaMemcpyHostToDevice), d.status);
            RASBERY_CUDA_TRY(cudaMemcpy(e.deltas, host.deltas,
                                        shape.n_deltas * sizeof(fxs::DeltaMeta),
                                        cudaMemcpyHostToDevice), d.status);
            g_flatxs_libs->push_back(e);
            found = &g_flatxs_libs->back();
        }
        d.lib = found;
    }

    // --- per-instance reference block (re-upload on ref generation) -------
    if (d.dev_ref == nullptr) {
        std::size_t off = 0;
        for (int t = 0; t < fxs::N_ACTIVE; ++t) { d.off_ref_mic[t] = off; off += mic; }
        d.off_ref_msm = off; off += msm;
        for (int t = 0; t < fxs::N_ACTIVE; ++t) { d.off_ref_lmp[t] = off; off += lmp; }
        d.off_ref_lsm = off; off += ssm;
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_ref),
                                          off * sizeof(double)), d.status);
        d.resident_ref_generation = 0;
    }
    if (ref_generation != d.resident_ref_generation) {
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_mic[t], host.ref_mic[t],
                                             mic * sizeof(double),
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_msm, host.ref_msm,
                                         msm * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_lmp[t], host.ref_lmp[t],
                                             lmp * sizeof(double),
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_ref + d.off_ref_lsm, host.ref_lsm,
                                         ssm * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        d.resident_ref_generation = ref_generation;
    }

    // --- live micx/lmpx blocks: same residency contract as the xsrecon
    // solve.  On mismatch upload ALL 11 slots (the kernel writes only the
    // ACTIVE 9, but the xsrecon condense loop reads every slot later).
    if (micx_generation != d.resident_micx_generation) {
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.mic_all[xt], d.off_mic[xt], mic)) return false;
        if (!d.upload(host.msm, d.off_mic_ssm, msm)) return false;
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.lmp_all[xt], d.off_lmp[xt], lmp)) return false;
        if (!d.upload(host.lsm, d.off_lmp_ssm, ssm)) return false;
        d.resident_micx_generation = micx_generation;
    }

    // --- per-call state: xs and iden whole (target-only writes round-trip
    // every other column unchanged), skipped while the host-state generation
    // says the resident copies are still bit-identical.
    if (state_generation != d.resident_state_generation) {
        for (int xt = 0; xt < xsr::NXS; ++xt)
            if (!d.upload(host.xs[xt], d.off_xs[xt], lmp)) return false;
        if (!d.upload(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
        if (!d.upload(host.iden, d.off_iden, static_cast<std::size_t>(xsr::NISO) * nx)) return false;
    }

    if (d.dev_pernode == nullptr)
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_pernode),
                                          3 * nx * sizeof(double)), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_pernode, host.wvfr, nx * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_pernode + nx, host.dmod, nx * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_pernode + 2 * nx, host.bppm, nx * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);

    const std::size_t n_nodes = static_cast<std::size_t>(host.n_nodes);
    if (n_nodes > d.nodes_cap) {
        rasbery::AllocWindow _alloc_window("xsrecon.nodes.regrow");
        if (d.dev_nodes) cudaFree(d.dev_nodes);
        if (d.dev_off) cudaFree(d.dev_off);
        if (d.dev_cnt) cudaFree(d.dev_cnt);
        d.dev_nodes = d.dev_off = d.dev_cnt = nullptr;
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_nodes),
                                          n_nodes * sizeof(int)), d.status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_off),
                                          n_nodes * sizeof(int)), d.status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_cnt),
                                          n_nodes * sizeof(int)), d.status);
        d.nodes_cap = n_nodes;
    }
    std::size_t stream_len = 0;
    if (host.n_nodes > 0)
        stream_len = static_cast<std::size_t>(host.node_off[host.n_nodes - 1]) +
                     static_cast<std::size_t>(host.node_cnt[host.n_nodes - 1]);
    if (stream_len > d.stream_cap) {
        rasbery::AllocWindow _alloc_window("xsrecon.stream.regrow");
        if (d.dev_sdid) cudaFree(d.dev_sdid);
        if (d.dev_sx) cudaFree(d.dev_sx);
        if (d.dev_sscale) cudaFree(d.dev_sscale);
        d.dev_sdid = nullptr; d.dev_sx = d.dev_sscale = nullptr;
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_sdid),
                                          stream_len * sizeof(int)), d.status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_sx),
                                          stream_len * sizeof(double)), d.status);
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.dev_sscale),
                                          stream_len * sizeof(double)), d.status);
        d.stream_cap = stream_len;
    }
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_nodes, host.nodes, n_nodes * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_off, host.node_off, n_nodes * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_cnt, host.node_cnt, n_nodes * sizeof(int),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    if (stream_len > 0) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_sdid, host.stream_did,
                                         stream_len * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_sx, host.stream_x,
                                         stream_len * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.dev_sscale, host.stream_scale,
                                         stream_len * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
    }

    // --- repoint the view at the device copies ----------------------------
    fxs::FlatXsView v = host;
    for (int t = 0; t < fxs::N_ACTIVE; ++t) {
        v.coeff_lmp[t] = d.lib->block + d.lib->off_lmp[t];
        v.coeff_mic[t] = d.lib->block + d.lib->off_mic[t];
        v.ref_lmp[t]   = d.dev_ref + d.off_ref_lmp[t];
        v.ref_mic[t]   = d.dev_ref + d.off_ref_mic[t];
        v.lmp[t]       = d.dev_block + d.off_lmp[ACTIVE_XT9[t]];
        v.mic[t]       = d.dev_block + d.off_mic[ACTIVE_XT9[t]];
    }
    v.coeff_lsm = d.lib->block + d.lib->off_lsm;
    v.coeff_msm = d.lib->block + d.lib->off_msm;
    v.knots     = d.lib->block + d.lib->off_knots;
    v.deltas    = d.lib->deltas;
    v.ref_lsm   = d.dev_ref + d.off_ref_lsm;
    v.ref_msm   = d.dev_ref + d.off_ref_msm;
    v.lsm       = d.dev_block + d.off_lmp_ssm;
    v.msm       = d.dev_block + d.off_mic_ssm;
    for (int xt = 0; xt < xsr::NXS; ++xt)
        v.xs[xt] = d.dev_block + d.off_xs[xt];
    v.xs_ssm       = d.dev_block + d.off_xs_ssm;
    v.iden         = d.dev_block + d.off_iden;
    v.wvfr         = d.dev_pernode;
    v.dmod         = d.dev_pernode + nx;
    v.bppm         = d.dev_pernode + 2 * nx;
    v.stream_did   = d.dev_sdid;
    v.stream_x     = d.dev_sx;
    v.stream_scale = d.dev_sscale;
    v.node_off     = d.dev_off;
    v.node_cnt     = d.dev_cnt;
    v.nodes        = d.dev_nodes;

    const int block = 128;
    const int grid  = (host.n_nodes + block - 1) / block;
    kernelFlatXs<<<grid, block, 0, d.stream>>>(v);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    // --- results the host needs back --------------------------------------
    // EXPERIMENT (RASBERY_FLATXS_SKIP_MICX_DL=1): leave micx/lmpx device-only
    // -- 57 of the 61 MB/call downloads.  UNSAFE whenever the host reads
    // _micx/_lmpx afterwards (depletion, rodded NodeSpectralIndex fallback,
    // full micx export, CPU fallback arms); the full-deck A/B is the gate for
    // any deck class this is enabled on.  Default off.
    static const bool skip_micx_dl = envFlagEnabled("RASBERY_FLATXS_SKIP_MICX_DL");
    if (!skip_micx_dl) {
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            if (!d.download(host.lmp[t], d.off_lmp[ACTIVE_XT9[t]], lmp)) return false;
            if (!d.download(host.mic[t], d.off_mic[ACTIVE_XT9[t]], mic)) return false;
        }
        if (!d.download(host.lsm, d.off_lmp_ssm, ssm)) return false;
        if (!d.download(host.msm, d.off_mic_ssm, msm)) return false;
    }
    for (int xt = 0; xt < xsr::NXS; ++xt)
        if (!d.download(host.xs[xt], d.off_xs[xt], lmp)) return false;
    if (!d.download(host.xs_ssm, d.off_xs_ssm, ssm)) return false;
    // Light-isotope rows H-1/B-10/O-16 are 0..2 -- contiguous by registry design.
    if (!d.download(host.iden, d.off_iden, 3 * nx)) return false;

    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    // After the download the host and device copies are bit-identical, so the
    // caller's post-call generation bump can be marked already-resident and
    // the next xsrecon call skips its ~70 MB re-upload.  A rodded CPU pass
    // after this call invalidates that (mark_micx_resident=false).
    d.resident_micx_generation  = mark_micx_resident ? micx_generation_next : 0;
    // A rodded CPU pass right after this call rewrites _xs/_iden columns on
    // the host, so the caller only lets us keep the state residency when the
    // whole call was device-side.
    d.resident_state_generation = mark_micx_resident ? state_generation : 0;

    g_flatxs_nodes_solved.fetch_add(static_cast<unsigned long long>(host.n_nodes),
                                    std::memory_order_relaxed);
    return true;
}

bool XsReconBackend::solveNodal(const ndl::NodalView& host,
                                unsigned long long const_generation,
                                unsigned long long ref_generation,
                                unsigned long long state_generation) {
    // calculateEven stays on the host until its 1-ULP residual class is
    // mined out (RASBERY_GPU_NODAL_FULL=1 forces the all-device path).
    // Shared with Nodal::TryDriveGpu through the one inline flag reader, so
    // the two halves of a drive cannot disagree about which mode this is.
    const bool hybrid_even = !rasberyGpuNodalFullEnabled();
    Impl& d = *_impl;
    // W3 item 2: the pending event describes ONE drive.  Cleared HERE, before
    // any early return, so a drive that refuses -- the batch arena took it, the
    // allocation failed, the enqueue failed -- cannot leave the previous drive's
    // event standing for the caller to wait on.  It is set again only at the
    // very end, and only when this drive actually deferred.
    d.nodal_drain_deferred = false;
    if (!d.available || host.nxyz <= 0 || host.nsurf <= 0) return false;

    // ---- multi-instance batch arena ---------------------------------------
    // Engaged only for --batch-mode M>1 with the FULL pipeline; see
    // NodalArena.  Every refusal and every failure below falls through to the
    // per-instance code that follows, which is untouched.
    if (nodalArenaWanted()) {
        NodalArena* arena = nodalArenaFor(host);
        if (arena != nullptr) {
            if (d.nodal_slot < 0 && !d.nodal_slot_refused) {
                d.nodal_slot = arena->acquireSlot(host);
                if (d.nodal_slot < 0) {
                    d.nodal_slot_refused = true;
                    g_nodal_batch_refused.fetch_add(1, std::memory_order_relaxed);
                    std::cerr << "[RASBERY][WARN][nodal] batch arena refused a slot "
                                 "(geometry mismatch or width exhausted) -- per-instance arm\n";
                } else if (d.canonical.buffers.shared()) {
                    // Rev.7.1 Task 18: THE ADOPTION HAS TO BE REPLAYED HERE, and
                    // getting this wrong is what a batch looks like when it is
                    // one outer stale.  adoptCanonicalBuffers forwards to the
                    // arena, but the segment arms BEFORE the first drive and the
                    // slot is acquired lazily INSIDE it -- so at arm time
                    // nodal_slot is -1 and the forward is a no-op.  Statepoint 1
                    // then ran with an unadopted arena slot: the kernels wrote
                    // the arena's own dense jnet while the segment, believing
                    // the binding, had stopped filling Geometry::Jnet.
                    //
                    // Replaying it at the moment the slot is learned costs one
                    // view-table rebuild per tenancy and closes the window.
                    arena->adoptCanonical(d.nodal_slot, d.canonical.buffers);
                }
            }
            if (d.nodal_slot >= 0) {
                if (arena->drive(d.nodal_slot, host, const_generation, ref_generation,
                                 d.canonical, d.canonical_materialize)) {
                    // Rev.7.1 Task 18: the batch drive just wrote jnet and phis
                    // on the device, exactly as the per-instance FULL path does
                    // at the bottom of this function -- so the ownership moves
                    // here too, or the NEXT drive would upload the host arrays
                    // back over them and the arena would honour the binding for
                    // one outer only.
                    d.canonical.setOwner(gpu::CanonicalRegion::Jnet,
                                         gpu::CanonicalOwner::Nodal);
                    d.canonical.setOwner(gpu::CanonicalRegion::Phis,
                                         gpu::CanonicalOwner::Nodal);
                    return true; // jnet/phis are home; counters bumped in the arena
                }
                g_nodal_batch_fallbacks.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    if (!d.ensure(host.nxyz, d.n_fuel > 0 ? d.n_fuel : host.nxyz)) {
        d.available = false;
        return false;
    }
    const std::size_t nx = static_cast<std::size_t>(host.nxyz);
    const std::size_t ns = static_cast<std::size_t>(host.nsurf);

    if (d.ndev_dbl == nullptr || d.nodal_nsurf != host.nsurf) {
        d.dropNodalGraph(); // baked ndev_dbl-relative pointers
        rasbery::AllocWindow _alloc_window("nodal.instance.regrow");
        if (d.ndev_dbl) { cudaFree(d.ndev_dbl); d.ndev_dbl = nullptr; }
        if (d.ndev_int) { cudaFree(d.ndev_int); d.ndev_int = nullptr; }
        d.nodal_geom_uploaded       = false;
        d.resident_const_generation = 0;
        d.resident_chif_generation  = 0;
        d.nodal_nsurf               = host.nsurf;
        const std::size_t ndg0 = nx * ndl::NDIR * ndl::NG;
        std::size_t off = 0;
        d.n_off_hmesh = off; off += nx * ndl::NDIR;
        d.n_off_albedo = off; off += ndl::NDIR * ndl::NLR;
        d.n_off_consts = off; off += 9 * ndg0;
        d.n_off_chif = off; off += ndl::NG * nx;
        d.n_off_jnet = off; off += ns * ndl::NG;
        d.n_off_flux = off; off += nx * ndl::NG;
        d.n_off_phis = off; off += ns * ndl::NG;
        d.n_off_reigv = off; off += 1;
        d.n_off_work = off;
        off += 3 * ndg0 + 2 * nx * ndl::NDIR * ndl::NG2 + 4 * nx * ndl::NG2 +
               3 * ndg0;
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.ndev_dbl),
                                          off * sizeof(double)), d.status);
        std::size_t ioff = 0;
        d.n_ioff_lktosfc = ioff; ioff += nx * ndl::NDIR * ndl::NLR;
        d.n_ioff_neib = ioff; ioff += nx * ndl::NEWSB;
        d.n_ioff_lklr = ioff; ioff += ns * ndl::NLR;
        d.n_ioff_idirlr = ioff; ioff += ns * ndl::NLR;
        d.n_ioff_sgnlr = ioff; ioff += ns * ndl::NLR;
        RASBERY_CUDA_TRY_ALLOC(cudaMalloc(reinterpret_cast<void**>(&d.ndev_int),
                                          ioff * sizeof(int)), d.status);
    }
    const std::size_t ndg = nx * ndl::NDIR * ndl::NG;

    if (!d.nodal_geom_uploaded) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_hmesh, host.hmesh,
                                         nx * ndl::NDIR * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_albedo, host.albedo,
                                         ndl::NDIR * ndl::NLR * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_lktosfc, host.lktosfc,
                                         nx * ndl::NDIR * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_neib, host.neib,
                                         nx * ndl::NEWSB * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_lklr, host.lklr,
                                         ns * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_idirlr, host.idirlr,
                                         ns * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_int + d.n_ioff_sgnlr, host.sgnlr,
                                         ns * ndl::NLR * sizeof(int),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        // The three per-drive host buffers are Geometry-owned; page-locking
        // them once makes the per-drive copies truly async and keeps them
        // capturable as graph memcpy nodes.  pinHost is idempotent, and the
        // lease it takes is released by ~Geometry, not here.
        pinHost(host.jnet, ns * ndl::NG * sizeof(double), "geom.jnet@nodal");
        pinHost(host.phis, ns * ndl::NG * sizeof(double), "geom.phis@nodal");
        pinHost(host.flux, nx * ndl::NG * sizeof(double), "geom.phif@nodal");
        d.nodal_geom_uploaded = true;
    }

    if (const_generation != d.resident_const_generation) {
        const double* consts[9] = {host.eta1, host.eta2, host.m260,
                                   host.m251, host.m253, host.m262,
                                   host.m264, host.diagD, host.diagDI};
        for (int i = 0; i < 9; ++i)
            RASBERY_CUDA_TRY(
                cudaMemcpyAsync(d.ndev_dbl + d.n_off_consts + i * ndg, consts[i],
                                ndg * sizeof(double), cudaMemcpyHostToDevice,
                                d.stream), d.status);
        d.resident_const_generation = const_generation;
    }

    if (!host.chif_empty && ref_generation != d.resident_chif_generation) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_chif, host.chif,
                                         ndl::NG * nx * sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
        d.resident_chif_generation = ref_generation;
    }

    // xs inputs: resident when the host-state generation matches; upload the
    // three rows for this call otherwise (residency untouched -- iden may be
    // stale, and the xs arms own that contract).
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;
    if (state_generation != d.resident_state_generation) {
        if (!d.upload(host.xsrf, d.off_xs[xsr::T_XSRF], lmp)) return false;
        if (!d.upload(host.xsnf, d.off_xs[xsr::T_XSNF], lmp)) return false;
        if (!d.upload(host.xssm, d.off_xs_ssm, ssm)) return false;
    }

    ndl::NodalView v = host;
    v.hmesh   = d.ndev_dbl + d.n_off_hmesh;
    v.albedo  = d.ndev_dbl + d.n_off_albedo;
    v.lktosfc = d.ndev_int + d.n_ioff_lktosfc;
    v.neib    = d.ndev_int + d.n_ioff_neib;
    v.lklr    = d.ndev_int + d.n_ioff_lklr;
    v.idirlr  = d.ndev_int + d.n_ioff_idirlr;
    v.sgnlr   = d.ndev_int + d.n_ioff_sgnlr;
    v.eta1   = d.ndev_dbl + d.n_off_consts + 0 * ndg;
    v.eta2   = d.ndev_dbl + d.n_off_consts + 1 * ndg;
    v.m260   = d.ndev_dbl + d.n_off_consts + 2 * ndg;
    v.m251   = d.ndev_dbl + d.n_off_consts + 3 * ndg;
    v.m253   = d.ndev_dbl + d.n_off_consts + 4 * ndg;
    v.m262   = d.ndev_dbl + d.n_off_consts + 5 * ndg;
    v.m264   = d.ndev_dbl + d.n_off_consts + 6 * ndg;
    v.diagD  = d.ndev_dbl + d.n_off_consts + 7 * ndg;
    v.diagDI = d.ndev_dbl + d.n_off_consts + 8 * ndg;
    v.chif   = d.ndev_dbl + d.n_off_chif;
    v.xsrf = d.dev_block + d.off_xs[xsr::T_XSRF];
    v.xsnf = d.dev_block + d.off_xs[xsr::T_XSNF];
    v.xssm = d.dev_block + d.off_xs_ssm;
    // Rev.7.1 Task 7: CANONICAL STATE.  In shared mode jnet/flux/phis are the
    // arena's buffers, which the CMFD backend also holds -- so the three
    // per-drive uploads and the two downloads below are copies of data that is
    // already where it needs to be.  A null borrowed pointer means this slot is
    // legacy and the private block is used, which is what lets one slot share
    // while another does not (mixed mode) with no third code path.
    const gpu::CanonicalSlotBuffers& canon = d.canonical.buffers;
    v.jnet = canon.jnet != nullptr ? canon.jnet : d.ndev_dbl + d.n_off_jnet;
    v.flux = canon.flux != nullptr ? canon.flux : d.ndev_dbl + d.n_off_flux;
    v.phis = canon.phis != nullptr ? canon.phis : d.ndev_dbl + d.n_off_phis;
    double* wk = d.ndev_dbl + d.n_off_work;
    v.trlcff0 = wk; wk += ndg;
    v.trlcff1 = wk; wk += ndg;
    v.trlcff2 = wk; wk += ndg;
    v.mu = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.tau = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.matM = wk; wk += nx * ndl::NG2;
    v.matMI = wk; wk += nx * ndl::NG2;
    v.matMs = wk; wk += nx * ndl::NG2;
    v.matMf = wk; wk += nx * ndl::NG2;
    v.dsncff2 = wk; wk += ndg;
    v.dsncff4 = wk; wk += ndg;
    v.dsncff6 = wk; wk += ndg;

    const int B  = 128;
    const int gn = (host.nxyz + B - 1) / B;
    const int gs = (host.nsurf + B - 1) / B;
    // trlcff0/trlcff12 are launched over (node, group) pairs; the rest stay
    // per node / per surface.
    const int gng = (host.nxyz * ndl::NG + B - 1) / B;
    const std::size_t surf_bytes = ns * ndl::NG * sizeof(double);

    if (hybrid_even) {
        // ---- HYBRID (validated fallback): unchanged, and deliberately NOT
        // graphed -- its mid-drive host hop makes the sequence non-static.
        // reigv_dev stays null so updateMatrix reads the by-value scalar
        // exactly as before.
        v.reigv_dev = nullptr;
        // Task 7: the same elision as the FULL path, and NOT optional here.
        // v.jnet/v.flux are bound to the canonical buffers above, so an
        // unconditional upload would overwrite what the CMFD backend just
        // produced with the host's stale copy -- the sharing would silently
        // become a slower, wronger version of the old path.
        if (!gpu::canonicalElidesUpload(canon, gpu::CanonicalRegion::Jnet,
                                        d.canonical.ownerOf(gpu::CanonicalRegion::Jnet),
                                        gpu::CanonicalOwner::Nodal)) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(v.jnet, host.jnet, surf_bytes,
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        } else {
            ++d.canonical_uploads_elided;
            g_canon_up_bytes.fetch_add(
                surf_bytes, std::memory_order_relaxed);
        }
        if (!gpu::canonicalElidesUpload(canon, gpu::CanonicalRegion::Flux,
                                        d.canonical.ownerOf(gpu::CanonicalRegion::Flux),
                                        gpu::CanonicalOwner::Nodal)) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(const_cast<double*>(v.flux), host.flux,
                                             nx * ndl::NG * sizeof(double),
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        } else {
            ++d.canonical_uploads_elided;
            g_canon_up_bytes.fetch_add(
                nx * ndl::NG * sizeof(double), std::memory_order_relaxed);
        }
        kNodalTrl0<false><<<gng, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        kNodalTrl12<false><<<gng, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        kNodalMat<false><<<gn, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

        // calculateEven runs on the HOST (the production member function is
        // its own bit-exact reference; its mined mask still has a 1-ULP
        // residual class).  Ship the inputs it needs, let the caller run it,
        // and finish in solveNodalPost.
        const std::size_t ndg2 = nx * ndl::NDIR * ndl::NG;
        double* wk0 = d.ndev_dbl + d.n_off_work;
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.trlcff0, wk0,
                                         ndg2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.trlcff2, wk0 + 2 * ndg2,
                                         ndg2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
        double* wmat = wk0 + 3 * ndg2 + 2 * nx * ndl::NDIR * ndl::NG2;
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.matM, wmat,
                                         nx * ndl::NG2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
        RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);
        // Hybrid D2H per drive = trlcff0 + trlcff2 + matM here, jnet + phis in
        // solveNodalPost.
        g_nodal_d2h_bytes.store((2 * ndg2 + nx * ndl::NG2) * sizeof(double) +
                                    2 * surf_bytes,
                                std::memory_order_relaxed);
        return true; // caller: run host even, then solveNodalPost
    }

    // ---- FULL device path -------------------------------------------------
    // The whole drive is one device pipeline: nothing comes back mid-drive,
    // and the ONLY arrays copied out are the two the host actually consumes
    // afterwards (jnet, read by CMFD::upddhat on the very next line of
    // Driver::SolveLoop; phis, read by PPR/NormalizeFluxSign at the
    // statepoint).  Everything the hybrid used to round-trip -- trlcff0,
    // trlcff2, matM down and dsncff2/4/6 back up, 5 transfers and a mid-drive
    // stream sync -- is gone, because calculateEven now runs on the device
    // beside the data.  trlcff1/2/0 stay device-only, which is exactly what
    // the fractional-rod guard in Nodal::TryDriveGpu already covers: rod
    // cusping is the only host reader of trlcff, and it never runs when that
    // guard let us get here.
    v.reigv_dev = d.ndev_dbl + d.n_off_reigv;
    // Rev.7.1 Task 10 part 3: the halt gate, on the FULL arm only.
    //
    // THE HYBRID ARM IS DELIBERATELY LEFT UNGATED.  It runs calculateEven on
    // the host in the middle of the drive, so half of it could not be gated by
    // a device word anyway -- and a host-free segment refuses that arm for
    // exactly that reason (outerCanonicalNodalEligibleHook requires FULL).
    // Setting the pointer only here means a hybrid run cannot inherit a gate
    // some earlier arm installed.
    v.halt      = static_cast<const unsigned int*>(d.nodal_halt);
    v.halt_slot = d.nodal_halt_slot;
    if (d.nodal_h_reigv == nullptr) {
        rasbery::AllocWindow _alloc_window("nodal.reigv.hostalloc");
        if (cudaHostAlloc(reinterpret_cast<void**>(&d.nodal_h_reigv),
                          sizeof(double), cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            d.nodal_h_reigv   = nullptr;
            d.nodal_use_graph = false; // no stable staging slot -> no capture
        }
    }
    // Rev.7.1 W3 item 3: WHO PUT THE RECIPROCAL IN THAT SLOT.
    //
    // Normally the host: it holds `eigv`, it divides, and the copy below carries
    // the quotient down.  Inside a device outer segment the eigenvalue is the
    // SWEEP's and it never left the device, so the segment's own one-thread
    // kernel has already written 1/eigv into this exact address, stream-ordered
    // ahead of this drive.  Uploading then would overwrite the device's answer
    // with a host copy of an older one.
    const bool reigv_device = d.nodal_reigv_device;
    // The staged memcpy reads this address at every launch, so the current
    // eigenvalue has to be in it BEFORE the launch.
    if (!reigv_device && d.nodal_h_reigv != nullptr)
        *d.nodal_h_reigv = host.reigv;

    const double* reigv_src =
        d.nodal_h_reigv != nullptr ? d.nodal_h_reigv : &host.reigv;

    // One drive's worth of device work, all fixed addresses -- this is what
    // gets captured.  No host callbacks, no allocation, no synchronisation
    // inside: every one of those is illegal or capture-breaking.
    auto enqueue_full = [&]() -> bool {
        // Task 7: elided when the region is canonical AND a device side wrote it
        // last.  When the HOST wrote last -- a perturbation, a rod move, a
        // restart -- the upload is still required, which is why the predicate
        // consults ownership instead of just the sharing flag.
        if (!gpu::canonicalElidesUpload(canon, gpu::CanonicalRegion::Jnet,
                                        d.canonical.ownerOf(gpu::CanonicalRegion::Jnet),
                                        gpu::CanonicalOwner::Nodal)) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(v.jnet, host.jnet, surf_bytes,
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        } else {
            ++d.canonical_uploads_elided;
            g_canon_up_bytes.fetch_add(
                surf_bytes, std::memory_order_relaxed);
        }
        if (!gpu::canonicalElidesUpload(canon, gpu::CanonicalRegion::Flux,
                                        d.canonical.ownerOf(gpu::CanonicalRegion::Flux),
                                        gpu::CanonicalOwner::Nodal)) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(const_cast<double*>(v.flux), host.flux,
                                             nx * ndl::NG * sizeof(double),
                                             cudaMemcpyHostToDevice, d.stream), d.status);
        } else {
            ++d.canonical_uploads_elided;
            g_canon_up_bytes.fetch_add(
                nx * ndl::NG * sizeof(double), std::memory_order_relaxed);
        }
        kNodalTrl0<false><<<gng, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        kNodalTrl12<false><<<gng, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        if (nodalFuseMatEvenEnabled()) {
            kNodalMatEven<false><<<gn, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        } else {
            kNodalMat<false><<<gn, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
            kNodalEven<false><<<gn, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        }
        kNodalJnet<false><<<gs, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
        // The drive just wrote jnet and phis on the device, so Nodal owns them.
        // The download back to the Geometry arrays happens only when a host
        // consumer has ASKED (materialize); otherwise the CMFD backend reads
        // them straight out of the same buffers.
        if (!gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Jnet,
                                          d.canonical_materialize)) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(host.jnet, v.jnet, surf_bytes,
                                             cudaMemcpyDeviceToHost, d.stream), d.status);
        } else {
            ++d.canonical_downloads_elided;
            g_canon_down_bytes.fetch_add(
                surf_bytes, std::memory_order_relaxed);
        }
        if (!gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Phis,
                                          d.canonical_materialize)) {
            RASBERY_CUDA_TRY(cudaMemcpyAsync(host.phis, v.phis, surf_bytes,
                                             cudaMemcpyDeviceToHost, d.stream), d.status);
        } else {
            ++d.canonical_downloads_elided;
            g_canon_down_bytes.fetch_add(
                surf_bytes, std::memory_order_relaxed);
        }
        return true;
    };

    // Anything the capture baked in that could have moved since.  host.jnet /
    // host.phis / host.flux are Geometry-owned and stable per backend
    // instance, but a re-bound Nodal::reset would silently orphan the graph,
    // so they are checked rather than assumed.
    //
    // Task 7 adds four more.  The borrowed canonical pointers are memcpy
    // OPERANDS in the capture, and the materialize mask plus the two ownerships
    // decide which memcpy NODES exist at all -- a graph captured with the
    // downloads elided and replayed when a consumer is looking would leave the
    // host arrays stale, which is exactly the failure the observation API is
    // there to prevent.  All four are constant in the steady state (ownership
    // settles on the device side, and nobody materialises mid-segment), so this
    // costs one instantiation at the mode change and none after.
    // Rev.7.1 Task 10 part 4: A LOOKUP, NOT AN EQUALITY TEST AND A DESTROY.
    //
    // The key is built here and asked for; a miss leaves the selection empty and
    // the capture below fills it, and -- the whole point -- it leaves the OTHER
    // entries alone.  The alternating materialize mask therefore costs two
    // entries for the run instead of two captures per segment.
    XsReconBackend::Impl::NodalGraphKey want;
    want.ndev        = d.ndev_dbl;
    want.dblk        = d.dev_block;
    want.jnet        = host.jnet;
    want.phis        = host.phis;
    want.flux        = host.flux;
    want.nxyz        = host.nxyz;
    want.nsurf       = host.nsurf;
    want.chif_empty  = host.chif_empty;
    want.canon_jnet  = canon.jnet;
    want.canon_flux  = canon.flux;
    want.canon_phis  = canon.phis;
    want.materialize = d.canonical_materialize;
    want.owner_jnet  = static_cast<int>(d.canonical.ownerOf(gpu::CanonicalRegion::Jnet));
    want.owner_flux  = static_cast<int>(d.canonical.ownerOf(gpu::CanonicalRegion::Flux));
    want.halt        = d.nodal_halt;
    want.halt_slot   = d.nodal_halt_slot;
    d.selectNodalGraph(want);

    // Failing out of a half-enqueued drive must not leave a D2H in flight:
    // TryDriveGpu answers false and the CPU body then writes the very same
    // host.jnet/host.phis this stream is still copying into.  Drain first.
    auto fail_drained = [&]() {
        cudaStreamSynchronize(d.stream);
        cudaGetLastError();
        return false;
    };

    // ======================================================================
    // Rev.7.1 W3 item 3: THE reigv UPLOAD, OUTSIDE THE CAPTURE
    // ======================================================================
    //
    // IT USED TO BE THE GRAPH'S FIRST NODE, and it is issued here instead for
    // one reason: whether it should happen at all now varies per DRIVE.  Inside
    // a device outer segment the eigenvalue never left the device and the
    // segment wrote 1/eigv into d.ndev_dbl + n_off_reigv itself; on the Wielandt
    // warm-up outer beside it the host owns the eigenvalue and the upload is
    // required.  A memcpy NODE cannot be conditional, so keeping it in the
    // capture would mean keying the graph on the flag and re-instantiating at
    // every flip -- for an eight-byte copy.
    //
    // THE ORDERING IS EXACTLY WHAT IT WAS.  This is d.stream, and every consumer
    // of the slot is a kernel launched on d.stream below (directly, or as the
    // replay of a graph launched on d.stream).  Being the first node of that
    // graph and being the operation immediately before it are the same position
    // in the same stream order.  On the capture pass it also lands before
    // cudaStreamBeginCapture -- there is a drain between the two -- so the
    // recorded graph never contains it.
    if (!reigv_device) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(d.ndev_dbl + d.n_off_reigv, reigv_src,
                                         sizeof(double),
                                         cudaMemcpyHostToDevice, d.stream), d.status);
    } else {
        g_nodal_reigv_device.fetch_add(1, std::memory_order_relaxed);
    }

    bool ok = true;
    if (!d.nodal_use_graph) {
        ok = enqueue_full();
    } else if (d.nodal_graph != nullptr) {
        // Rev.7.1 Task 10: LAUNCHED, OR SPLICED IF d.stream IS RECORDING.
        //
        // d.stream is pulled into the outer body's capture by the handover event
        // (CudaOuterGraph.cu records it on the segment stream and
        // waitOnSegmentEvent makes this one wait), so inside a captured body this
        // stream is capturing without ever having been begun -- and a graph
        // LAUNCH there is refused and takes the capture down with it.  See
        // src/GpuGraphSplice.h for the measurement and the three-call splice.
        const cudaError_t lrc =
            rasbery::graphLaunchOrSplice(d.nodal_graph, d.nodal_graph_src, d.stream);
        if (lrc != cudaSuccess) {
            d.status = std::string("cudaGraphLaunch -> ") + cudaGetErrorString(lrc);
            ok = false;
        } else {
            g_nodal_graph_launches.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (rasbery::graphCaptureActive(d.stream)) {
        // Rev.7.1 Task 10 part 4: A MISS INSIDE A CAPTURED OUTER BODY.
        //
        // d.stream is pulled into the outer body's capture by the handover
        // event, so the drain and the cudaStreamBeginCapture below would be a
        // host rendezvous and a NESTED capture on a recording stream -- the
        // second of which is refused and invalidates the outer capture.  The
        // direct enqueue is the same kernels the graph would have replayed, so
        // the body records the right work; what it does not do is populate the
        // cache, which is the segment's outer 0's job.
        //
        // COUNTED rather than silent: graph_warmup_misses is a gate at 0.
        rasbery::graphWarmupMiss();
        ok = enqueue_full();
    } else {
        // Capture once, then replay for the rest of the run.  The conditional
        // uploads above are already queued on this stream; drain them so the
        // capture starts on an idle stream (once per run, so free).
        //
        // Rev.7.1 Task 10 part 4: "ONCE PER RUN, SO FREE" IS THE CLAIM THIS
        // COUNTER EXISTS TO CHECK.  The drain is a host rendezvous on the
        // backend's stream and it is not in the segment's in_body_host_syncs,
        // so if the graph key moves the cost is real and nothing reports it.
        g_nodal_graph_captures.fetch_add(1, std::memory_order_relaxed);
        RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);
        cudaGraph_t graph  = nullptr;
        bool        enq_ok = true;
        rasbery::CaptureWindow _capture_window(d.stream, "nodal.instance");
        cudaError_t rc =
            cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal);
        if (rc == cudaSuccess) {
            enq_ok = enqueue_full();
            // Must be called even when the enqueue failed: it is what takes
            // the stream back out of capture mode.
            rc = cudaStreamEndCapture(d.stream, &graph);
        }
        if (rc == cudaSuccess && enq_ok)
            // 3-argument form: the legacy (errorNode, logBuffer, size) overload
            // is gone in CUDA 13, which the 238 server builds with.
            rc = cudaGraphInstantiate(&d.nodal_graph, graph, 0ull);
        // Rev.7.1 Task 10: the graph is KEPT on success (see nodal_graph_src) and
        // destroyed only when the instantiate did not take it.  dropNodalGraph
        // owns the pair from here.
        if (rc == cudaSuccess && enq_ok)
            d.nodal_graph_src = graph;
        else if (graph != nullptr)
            cudaGraphDestroy(graph);
        if (rc != cudaSuccess || !enq_ok) {
            // Capture is a pure optimisation; a driver that refuses it must
            // not take the solver down.  Work submitted to a stream in capture
            // mode is RECORDED, not executed, and a failed capture leaves
            // nothing pending -- so the re-enqueue below is this drive's first
            // and only execution, not a double application.
            cudaGetLastError();
            d.nodal_graph     = nullptr;
            d.nodal_graph_src = nullptr;
            d.nodal_use_graph = false;
            g_nodal_graph_fallbacks.fetch_add(1, std::memory_order_relaxed);
            ok = enqueue_full();
        } else {
            // INTO THE CACHE UNDER THE KEY IT WAS ASKED FOR.  `want` was built
            // before the capture and nothing between here and there can have
            // moved it -- the capture reads those values, it does not set them
            // -- so this is the graph for exactly these conditions.
            //
            // The cap is a leak guard, not a policy: if the key space turns out
            // to be bigger than the two values `materialize` alternates between,
            // throwing the cache away is precisely what this code did before,
            // so the degenerate case cannot be worse than the status quo.
            if (d.nodal_graphs.size() >= XsReconBackend::Impl::kNodalGraphCacheMax) {
                cudaGraphExec_t keep_exec = d.nodal_graph;
                cudaGraph_t     keep_src  = d.nodal_graph_src;
                d.dropNodalGraph();
                d.nodal_graph     = keep_exec;
                d.nodal_graph_src = keep_src;
            }
            d.nodal_graphs.push_back(
                XsReconBackend::Impl::NodalGraph{d.nodal_graph, d.nodal_graph_src, want});
            const cudaError_t lrc =
                rasbery::graphLaunchOrSplice(d.nodal_graph, d.nodal_graph_src, d.stream);
            if (lrc != cudaSuccess) {
                d.status = std::string("cudaGraphLaunch -> ") + cudaGetErrorString(lrc);
                ok = false;
            } else {
                g_nodal_graph_launches.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (!ok) return fail_drained();

    // ==================================================================
    // Rev.7.1 W3 item 2: DRAIN, OR HAND THE SEGMENT AN EVENT
    // ==================================================================
    //
    // NOTHING CAME BACK, SO NOBODY ON THIS SIDE IS WAITING.  Both downloads
    // above were elided exactly when the canonical buffers are shared and the
    // materialize mask is 0 -- which is what setCanonicalNodalSegmentMode(true)
    // establishes and what only the device outer segment ever asks for.  In that
    // state the drive's only consumer is a kernel on the SEGMENT's stream
    // (upddhat, enqueued on the very next line of runSegment), and
    // cudaStreamWaitEvent orders that strictly, with no host round trip.
    //
    // THE ORDERING CONTRACT IS UNCHANGED, ONLY ITS MECHANISM.  The exactness
    // contract's invariant 4 requires the two cross-stream handovers around the
    // nodal drive to be ordered by a synchronise OR an event; this is the
    // second.  What must not change is that upddhat still runs after this
    // drive, and the event is what says so.
    //
    // WHEN EITHER DOWNLOAD RAN, THE SYNCHRONISE STAYS.  The D2H targets
    // page-locked Geometry arrays that a host reader is about to touch --
    // CMFD::upddhat on the host outer body takes Geometry::Jnet on the very next
    // line -- and an event orders devices, not hosts.
    const bool drain_deferrable =
        !hybrid_even &&
        gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Jnet,
                                     d.canonical_materialize) &&
        gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Phis,
                                     d.canonical_materialize) &&
        d.ensureNodalEvent();

    d.nodal_drain_deferred = false;
    // ASKED BEFORE EITHER BRANCH.  cudaGetLastError() reports launch-time
    // failures without waiting for anything, so it is the half of the error
    // check the deferred path keeps; an execution fault surfaces at the
    // segment's own next synchronise, which is where every other enqueued phase
    // of the outer already reports one.
    const cudaError_t lasterr = cudaGetLastError();
    if (lasterr != cudaSuccess) {
        d.status = std::string("nodal FULL -> ") + cudaGetErrorString(lasterr);
        cudaStreamSynchronize(d.stream);
        cudaGetLastError();
        return false;
    }
    if (drain_deferrable) {
        const cudaError_t erc = cudaEventRecord(d.nodal_done_event, d.stream);
        if (erc != cudaSuccess) {
            d.status = std::string("nodal FULL event -> ") + cudaGetErrorString(erc);
            cudaStreamSynchronize(d.stream);
            cudaGetLastError();
            return false;
        }
        d.nodal_drain_deferred = true;
        g_nodal_drains_deferred.fetch_add(1, std::memory_order_relaxed);
    } else if (const cudaError_t syncrc = cudaStreamSynchronize(d.stream);
               syncrc != cudaSuccess) {
        d.status = std::string("nodal FULL -> ") + cudaGetErrorString(syncrc);
        cudaGetLastError();
        return false;
    }

    // Task 7: the FULL drive wrote jnet and phis on the device.  Recording it
    // here rather than at the enqueue is deliberate -- the ownership must not
    // move until the work has actually landed, or a failed drive would leave the
    // next upload elided against bytes that were never produced.
    if (!hybrid_even) {
        d.canonical.setOwner(gpu::CanonicalRegion::Jnet, gpu::CanonicalOwner::Nodal);
        d.canonical.setOwner(gpu::CanonicalRegion::Phis, gpu::CanonicalOwner::Nodal);
    }

    g_nodal_d2h_bytes.store(2 * surf_bytes, std::memory_order_relaxed);
    g_nodal_drives.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool XsReconBackend::solveNodalPost(const ndl::NodalView& host) {
    Impl& d = *_impl;
    if (!d.available || d.ndev_dbl == nullptr) return false;
    const std::size_t nx   = static_cast<std::size_t>(host.nxyz);
    const std::size_t ns   = static_cast<std::size_t>(host.nsurf);
    const std::size_t ndg2 = nx * ndl::NDIR * ndl::NG;

    // Upload the host-computed dsncff blocks into the device work region.
    double* wk0 = d.ndev_dbl + d.n_off_work;
    double* wds = wk0 + 3 * ndg2 + 2 * nx * ndl::NDIR * ndl::NG2 +
                  4 * nx * ndl::NG2;
    RASBERY_CUDA_TRY(cudaMemcpyAsync(wds, host.dsncff2, ndg2 * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(wds + ndg2, host.dsncff4,
                                     ndg2 * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);
    RASBERY_CUDA_TRY(cudaMemcpyAsync(wds + 2 * ndg2, host.dsncff6,
                                     ndg2 * sizeof(double),
                                     cudaMemcpyHostToDevice, d.stream), d.status);

    ndl::NodalView v = host;
    v.lklr    = d.ndev_int + d.n_ioff_lklr;
    v.idirlr  = d.ndev_int + d.n_ioff_idirlr;
    v.sgnlr   = d.ndev_int + d.n_ioff_sgnlr;
    v.lktosfc = d.ndev_int + d.n_ioff_lktosfc;
    v.neib    = d.ndev_int + d.n_ioff_neib;
    v.hmesh   = d.ndev_dbl + d.n_off_hmesh;
    v.albedo  = d.ndev_dbl + d.n_off_albedo;
    v.eta1   = d.ndev_dbl + d.n_off_consts + 0 * ndg2;
    v.eta2   = d.ndev_dbl + d.n_off_consts + 1 * ndg2;
    v.m260   = d.ndev_dbl + d.n_off_consts + 2 * ndg2;
    v.m251   = d.ndev_dbl + d.n_off_consts + 3 * ndg2;
    v.m253   = d.ndev_dbl + d.n_off_consts + 4 * ndg2;
    v.m262   = d.ndev_dbl + d.n_off_consts + 5 * ndg2;
    v.m264   = d.ndev_dbl + d.n_off_consts + 6 * ndg2;
    v.diagD  = d.ndev_dbl + d.n_off_consts + 7 * ndg2;
    v.diagDI = d.ndev_dbl + d.n_off_consts + 8 * ndg2;
    // Task 7: same canonical binding as solveNodal -- solveNodalPost finishes
    // the SAME drive, so it must address the same buffers or the jnet phase
    // writes somewhere the CMFD backend will never look.
    const gpu::CanonicalSlotBuffers& canon = d.canonical.buffers;
    v.jnet = canon.jnet != nullptr ? canon.jnet : d.ndev_dbl + d.n_off_jnet;
    v.flux = canon.flux != nullptr ? canon.flux : d.ndev_dbl + d.n_off_flux;
    v.phis = canon.phis != nullptr ? canon.phis : d.ndev_dbl + d.n_off_phis;
    double* wk = wk0;
    v.trlcff0 = wk; wk += ndg2;
    v.trlcff1 = wk; wk += ndg2;
    v.trlcff2 = wk; wk += ndg2;
    v.mu = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.tau = wk; wk += nx * ndl::NDIR * ndl::NG2;
    v.matM = wk; wk += nx * ndl::NG2;
    v.matMI = wk; wk += nx * ndl::NG2;
    v.matMs = wk; wk += nx * ndl::NG2;
    v.matMf = wk; wk += nx * ndl::NG2;
    v.dsncff2 = wk; wk += ndg2;
    v.dsncff4 = wk; wk += ndg2;
    v.dsncff6 = wk; wk += ndg2;

    const int B  = 128;
    const int gs = (host.nsurf + B - 1) / B;
    kNodalJnet<false><<<gs, B, 0, d.stream>>>(v, nullptr, 0, nullptr);
    RASBERY_CUDA_TRY(cudaGetLastError(), d.status);

    if (!gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Jnet,
                                      d.canonical_materialize)) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.jnet, v.jnet,
                                         ns * ndl::NG * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
    } else {
        ++d.canonical_downloads_elided;
        g_canon_down_bytes.fetch_add(ns * ndl::NG * sizeof(double),
                                     std::memory_order_relaxed);
    }
    if (!gpu::canonicalElidesDownload(canon, gpu::CanonicalRegion::Phis,
                                      d.canonical_materialize)) {
        RASBERY_CUDA_TRY(cudaMemcpyAsync(host.phis, v.phis,
                                         ns * ndl::NG * sizeof(double),
                                         cudaMemcpyDeviceToHost, d.stream), d.status);
    } else {
        ++d.canonical_downloads_elided;
        g_canon_down_bytes.fetch_add(ns * ndl::NG * sizeof(double),
                                     std::memory_order_relaxed);
    }
    RASBERY_CUDA_TRY(cudaStreamSynchronize(d.stream), d.status);

    // The drive produced jnet and phis on the device; record that so the next
    // drive's upload predicate knows it does not have to push them back.
    d.canonical.setOwner(gpu::CanonicalRegion::Jnet, gpu::CanonicalOwner::Nodal);
    d.canonical.setOwner(gpu::CanonicalRegion::Phis, gpu::CanonicalOwner::Nodal);

    g_nodal_drives.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 7: canonical CMFD-Nodal device state
// ---------------------------------------------------------------------------

void XsReconBackend::adoptCanonicalBuffers(const gpu::CanonicalSlotBuffers& buffers) {
    Impl& d = *_impl;
    // Rev.7.1 pre-W3: the adoption reaches the ARENA too, not just the
    // per-instance path.  Before the pointer table this was impossible -- the
    // arena computed a slot address by a dense stride the canonical block does
    // not have -- so the adoption stopped at the instance and the arena went on
    // using its own buffers.  Now the arena stores the override in its view
    // table and the stride question is gone.
    if (d.nodal_slot >= 0 && g_nodal_arena != nullptr)
        g_nodal_arena->adoptCanonical(d.nodal_slot, buffers);
    // Adopting is a TOPOLOGY change for the captured nodal graph: the borrowed
    // pointers are memcpy operands baked into it.  The key check in solveNodal
    // catches that on its own, but dropping here makes the invalidation happen
    // at the moment of the decision rather than one drive later.
    if (d.canonical.buffers.jnet != buffers.jnet ||
        d.canonical.buffers.flux != buffers.flux ||
        d.canonical.buffers.phis != buffers.phis)
        d.dropNodalGraph();

    d.canonical.buffers = buffers;
    // A freshly adopted buffer holds whatever the arena last put there, which
    // this backend did not produce.  Start every region HOST-owned so the first
    // drive uploads rather than trusting bytes nobody has vouched for.
    for (int r = 0; r < gpu::kCanonicalRegionCount; ++r)
        d.canonical.setOwner(static_cast<gpu::CanonicalRegion>(r), gpu::CanonicalOwner::Host);
}

gpu::CanonicalSlotBuffers XsReconBackend::canonicalBuffers() const {
    return _impl->canonical.buffers;
}

void XsReconBackend::setMaterializeMask(std::uint32_t mask) {
    Impl& d = *_impl;
    // THE MASK DECIDES WHICH DOWNLOAD NODES EXIST IN THE CAPTURE, so a graph
    // captured while nobody was looking must never replay when a consumer is:
    // that is a stale Geometry array, the exact failure this API exists to
    // prevent.  It used to be enforced by DESTROYING the graph here.
    //
    // Rev.7.1 Task 10 part 4: it is enforced by the key instead, and the graph
    // stays.  `materialize` is a field of NodalGraphKey, so a drive under this
    // mask can only ever select a graph captured under this mask -- the same
    // guarantee, from a lookup rather than from the absence of an alternative.
    // What that buys is the re-capture this line used to force at every segment
    // boundary: 3,282 of them on kngr_238, each one draining the backend's
    // stream.  See NodalGraphKey.
    d.canonical_materialize = mask;
}

std::uint32_t XsReconBackend::materializeMask() const {
    return _impl->canonical_materialize;
}

void XsReconBackend::setCanonicalNodalSegmentMode(bool in_segment, bool device_owns_flux) {
    Impl& d = *_impl;
    // A legacy instance borrows nothing, so there is no ownership to declare and
    // no download to suppress; answering here rather than at the call site is
    // what lets the segment call this unconditionally.
    if (d.canonical.buffers.jnet == nullptr) return;

    if (in_segment) {
        // WHY Nodal AND NOT Cmfd FOR jnet, when it is the CMFD updjnet that
        // wrote it.  canonicalElidesUpload only asks `did a DEVICE side write
        // this`, so both answers elide -- but the captured nodal graph's key
        // carries the ownership, and solveNodal leaves Jnet on Nodal when the
        // drive finishes.  Claiming Cmfd here would flip the key back on every
        // outer and drop the graph on every outer with it.  The value that
        // matches what the drive leaves behind is the one that keeps the graph.
        d.canonical.setOwner(gpu::CanonicalRegion::Jnet, gpu::CanonicalOwner::Nodal);
        d.canonical.setOwner(gpu::CanonicalRegion::Phis, gpu::CanonicalOwner::Nodal);
        // FLUX IS THE ONE THAT CAN LEGITIMATELY BE THE HOST'S.  A drive that
        // fell back to the host CMFD loop -- the Wielandt warm-up, a declined
        // enqueue -- left Geometry::Phif ahead of the device phi, and the upload
        // is then required rather than wasteful.  The caller is the only place
        // that knows, which is why it is an argument and not a guess.
        d.canonical.setOwner(gpu::CanonicalRegion::Flux,
                             device_owns_flux ? gpu::CanonicalOwner::Cmfd
                                              : gpu::CanonicalOwner::Host);
        setMaterializeMask(0u);
        return;
    }

    // Out of the segment: back to the pre-Task-7 transfers exactly.  Every
    // region host-owned means every upload happens; jnet and phis materialised
    // means both downloads happen.  That is what a host outer body reads --
    // CMFD::upddhat takes Geometry::Jnet on the very next line -- and what every
    // statepoint consumer of Geometry::Phis reads.
    for (int r = 0; r < gpu::kCanonicalRegionCount; ++r)
        d.canonical.setOwner(static_cast<gpu::CanonicalRegion>(r), gpu::CanonicalOwner::Host);
    setMaterializeMask(gpu::canonicalBit(gpu::CanonicalRegion::Jnet) |
                       gpu::canonicalBit(gpu::CanonicalRegion::Phis));
}

void* XsReconBackend::nodalCompletionEvent() {
    Impl& d = *_impl;
    if (!d.nodal_drain_deferred) return nullptr;
    // CONSUME-ONCE.  The event describes the drive that just ran, and the
    // caller's wait is what discharges it.  Leaving the flag up would hand the
    // NEXT outer -- whose drive may have taken the CPU body and enqueued
    // nothing at all -- an event from an older drive to wait on: harmless
    // today, because that event is long since complete, and exactly the kind of
    // stale handover that stops being harmless the moment anything overlaps.
    d.nodal_drain_deferred = false;
    return static_cast<void*>(d.nodal_done_event);
}

void* XsReconBackend::nodalReigvDeviceSlot() const {
    const Impl& d = *_impl;
    // NULL UNTIL THERE IS A BLOCK.  The nodal device arena is allocated inside
    // the first solveNodal, and re-laid-out whenever nsurf changes, so this is
    // the only honest answer before that and the reason the caller must ask
    // again every outer rather than cache it.  The hybrid arm never sets
    // v.reigv_dev, so a caller that got an address there would be writing a slot
    // updateMatrix does not read -- hence the FULL gate.
    if (!rasberyGpuNodalFullEnabled() || d.ndev_dbl == nullptr) return nullptr;
    return static_cast<void*>(d.ndev_dbl + d.n_off_reigv);
}

void XsReconBackend::setNodalReigvDeviceResident(bool resident) {
    // A PLAIN FLAG, WITH NO GRAPH DROP.  See the header: the upload it governs
    // was moved out of the capture precisely so that this can flip per drive
    // without costing an instantiation.
    _impl->nodal_reigv_device = resident;
}

bool XsReconBackend::waitOnSegmentEvent(void* event) {
    Impl& d = *_impl;
    if (!d.available || event == nullptr) return true;
    // NOT RECORDED INTO ANYTHING.  This is a stream-order dependency issued on
    // d.stream at the moment the caller asks, exactly like the reigv upload two
    // functions down -- and for the same reason it may not become a graph node:
    // the event describes ONE outer's updjnet, and a captured wait would make
    // every later replay depend on that outer's copy of it.
    const cudaError_t rc =
        cudaStreamWaitEvent(d.stream, static_cast<cudaEvent_t>(event), 0);
    if (rc != cudaSuccess) {
        d.status = std::string("nodal segment wait -> ") + cudaGetErrorString(rc);
        cudaGetLastError();
        return false;
    }
    return true;
}

void XsReconBackend::setNodalHaltGate(const void* halt, int slot) {
    // NO EXPLICIT GRAPH DROP HERE, and that is not an omission: the pair is in
    // NodalGraphKey, so the next drive simply selects a different entry (or
    // captures one).  Dropping here would destroy a graph exec while the
    // previous drive may still be in flight on d.stream -- the whole point of
    // the deferred drain is that the host does not wait for it.
    _impl->nodal_halt      = halt;
    _impl->nodal_halt_slot = slot;
}

unsigned long long XsReconBackend::canonicalUploadsElided() const {
    return _impl->canonical_uploads_elided;
}

unsigned long long XsReconBackend::canonicalDownloadsElided() const {
    return _impl->canonical_downloads_elided;
}

unsigned long long XsReconBackend::nodalDrivesSolved() {
    return g_nodal_drives.load(std::memory_order_relaxed);
}

unsigned long long XsReconBackend::nodesSolved() {
    return g_nodes_solved.load(std::memory_order_relaxed);
}

unsigned long long XsReconBackend::flatXsNodesSolved() {
    return g_flatxs_nodes_solved.load(std::memory_order_relaxed);
}

namespace {

/// The two hooks HostPinRegistry.h calls; installed by installHostPinHooks()
/// below at first use, so a stub build (which links neither .cu) keeps the
/// lease bookkeeping and makes no device call.
int cudaHostPinRegister(void* address, std::size_t bytes) {
    // Rev.7.1 Task 18d -- see the twin in CudaBICGBackend.cu: a first-touch pin
    // taken on one deck's thread must not overlap another deck's capture.
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

bool XsReconBackend::pinHost(const void* p, size_t bytes, const char* tag) {
    // Registration is LEASED, not permanent: the registry hands out one lease
    // per distinct base address, the buffer's owner releases it in its
    // destructor, and cudaHostUnregister runs at the address cudaHostRegister
    // saw.  That is what makes a recycled Driver worker safe -- the next deck's
    // allocation arrives at an empty registry instead of aliasing a dead
    // tenant's registration.  See HostPinRegistry.h.
    installHostPinHooks();
    return rasberyPinHost(p, bytes, tag);
}

bool rasberyGpuXsReconEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_XSRECON");
    return on;
}

bool rasberyGpuFlatXsEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_FLATXS");
    return on;
}

bool rasberyGpuNodalEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_NODAL");
    return on;
}

bool rasberyGpuXeEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_XE");
    return on;
}

bool rasberyGpuXeTxnEnabled() {
    // WP7-C.  envFlagEnabled, i.e. ABSENT MEANS OFF.  The transaction is a
    // performance change with a bit-identity claim attached, and a default-on
    // performance change is a claim nobody was asked to check.
    static const bool on = envFlagEnabled("RASBERY_GPU_XE_TXN");
    return on;
}

int rasberyGpuXeDotPartitions() {
    // Read once, clamped once.  A partition count that changed between calls
    // would change the association between calls, and then a run would not even
    // be reproducible against itself -- which is the one property the fixed
    // partition exists to give.  Nonsense (zero, negative, unparseable) falls
    // back to the default rather than to 1: silently switching to the exact but
    // slow fold would look like a performance regression with no cause.
    static const int parts = [] {
        const char* v = std::getenv("RASBERY_GPU_XE_DOT_PARTITIONS");
        if (v == nullptr) return xe::XE_DOT_PARTITIONS_DEFAULT;
        char*           end = nullptr;
        const long long n   = std::strtoll(v, &end, 10);
        if (end == nullptr || *end != '\0' || n < 1) {
            std::cerr << "[RASBERY][WARN][xe] RASBERY_GPU_XE_DOT_PARTITIONS=\"" << v
                      << "\" is not a positive count; using the default "
                      << xe::XE_DOT_PARTITIONS_DEFAULT << " instead.\n";
            return xe::XE_DOT_PARTITIONS_DEFAULT;
        }
        return static_cast<int>(n > xe::XE_DOT_PARTITIONS_MAX
                                    ? xe::XE_DOT_PARTITIONS_MAX
                                    : n);
    }();
    return parts;
}

unsigned long long rasberyGpuXeEvaluations() {
    return g_xe_evaluations.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuXeCommits() {
    return g_xe_commits.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuNodalDrives() {
    return g_nodal_drives.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuNodalCanonicalElidedUploadBytes() {
    return g_canon_up_bytes.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuNodalCanonicalElidedDownloadBytes() {
    return g_canon_down_bytes.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuXsReconNodes() {
    return g_nodes_solved.load(std::memory_order_relaxed);
}

unsigned long long rasberyGpuFlatXsNodes() {
    return g_flatxs_nodes_solved.load(std::memory_order_relaxed);
}

} // namespace rasbery
