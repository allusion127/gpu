#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace rasbery {

class Geometry;

enum SolveFlag : std::uint32_t {
    FLUX_CONVERGED     = 1u << 0,
    KEFF_CONVERGED     = 1u << 1,
    DHAT_CONVERGED     = 1u << 2,
    SEARCH_CONVERGED   = 1u << 3,
    NONFINITE_DETECTED = 1u << 4,
    NEGATIVE_FLUX      = 1u << 5,
    BICGSTAB_BREAKDOWN = 1u << 6,
    STALE_GENERATION   = 1u << 7,
    MAX_ITER_REACHED   = 1u << 8
};

/**
 * One cache-line status packet copied from device to host at a solver
 * observation boundary. Fields not owned by the linear solver are NaN/zero.
 */
struct alignas(64) DeviceSolveStatus {
    double keff;
    double flux_l2;
    double dhat_defect_max;
    double dhat_update_max;
    double search_residual;

    std::uint32_t flags;
    std::uint32_t outer_iter;
    std::uint32_t linear_iter;
    std::uint32_t material_gen;
    std::uint32_t operator_gen;
    std::uint32_t flux_gen;
};
static_assert(sizeof(DeviceSolveStatus) == 64);
static_assert(alignof(DeviceSolveStatus) == 64);

/**
 * Runtime evidence for strict GPU residency. The fields cover the common
 * backend contract even though this class currently owns only CMFD.
 */
struct BackendCounters {
    std::uint64_t xs_gpu_calls                         = 0;
    std::uint64_t xs_cpu_fallbacks                     = 0;
    std::uint64_t cmfd_gpu_calls                       = 0;
    std::uint64_t cmfd_cpu_fallbacks                   = 0;
    std::uint64_t bicg_early_convergence_exits          = 0;
    std::uint64_t bicg_restarts                        = 0;
    std::uint64_t nodal_gpu_calls                      = 0;
    std::uint64_t nodal_cpu_fallbacks                  = 0;
    std::uint64_t th_gpu_calls                         = 0;
    std::uint64_t depletion_gpu_calls                  = 0;
    std::uint64_t bulk_h2d_calls_during_iteration      = 0;
    /// Bulk H2D uploads elided because the host buffer was bit-identical to
    /// what the device already holds (see CudaBICGBackend::reset).
    std::uint64_t bulk_h2d_skipped_during_iteration    = 0;
    /// Bytes actually pushed across PCIe by the bulk uploads above.
    std::uint64_t bulk_h2d_bytes_during_iteration      = 0;
    std::uint64_t bulk_d2h_calls_during_iteration      = 0;
    std::uint64_t status_d2h_calls_during_iteration    = 0;
    std::uint64_t stream_sync_calls_during_iteration   = 0;
    std::uint64_t graph_launches                       = 0;
    std::uint64_t graph_reinstantiations               = 0;
    std::uint64_t graph_fallbacks                      = 0;
};

/**
 * CUDA-resident two-group block-Jacobi BiCGSTAB backend.
 *
 * The CMFD matrix, Krylov vectors, and flux remain on the GPU for every
 * inner iteration. The converged/current flux is copied back explicitly at
 * the CMFD/nodal observation boundary.
 *
 * Since the multi-instance batch mode landed this is a *thin wrapper over a
 * one-slot batch arena*: there is exactly one kernel implementation, launched
 * with gridDim.y = 1 here and gridDim.y = M there.  Keeping a single body is
 * what makes "M = 1 batch mode is bit-identical to a plain single run" true by
 * construction rather than by manual synchronisation of two copies.
 */
class CudaBICGBackend {
private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    explicit CudaBICGBackend(Geometry& geometry);
    ~CudaBICGBackend();

    CudaBICGBackend(const CudaBICGBackend&)            = delete;
    CudaBICGBackend& operator=(const CudaBICGBackend&) = delete;

    [[nodiscard]] bool               available() const;
    [[nodiscard]] const std::string& status() const;
    [[nodiscard]] BackendCounters    counters() const;
    [[nodiscard]] DeviceSolveStatus  lastSolveStatus() const;

    /// Stage the operator, flux and source for one CMFD outer.  Uploads only:
    /// no kernel runs and no synchronisation happens here.
    void reset(const double* diag, const double* cc, const double* phi, const double* src);

    /// Enqueue the entire inner BiCGSTAB loop -- initial residual plus one
    /// unconditional iteration and `nmax` conditional ones -- as a single
    /// replayed CUDA graph.  The relative exit test `||r||/||r0|| < eps` runs
    /// on the device, so the host neither observes nor steers the loop.
    void solveInner(int nmax, double eps);

    /// Drain the stream once per outer: fetch the flux and the status packet
    /// the graph queued, and raise the accumulated fatal condition if any.
    void synchronize(double* phi);
};

// ---------------------------------------------------------------------------
// Multi-instance (multi-state) batch arena.
//
// M independent RASBERY instances share one geometry, so they share the CMFD
// sparsity pattern (`neighbors`, `colors`) and differ only in the coefficients
// (diag / cc / src) and the flux.  The arena holds M *slots* of those
// coefficients back to back and runs the whole inner BiCGSTAB loop for every
// slot in one grid: block (b, m) of the batched launch does exactly the work
// that block b of instance m's single-instance launch would have done.
//
//   *** The bit-identity rule: gridDim.x is per-instance, never re-chunked. ***
//
// The only order-sensitive operation in the solver is the dot product, whose
// partition is `chunk = ceil(n / gridDim.x)`.  Adding the batch axis as
// gridDim.y leaves gridDim.x -- and therefore the chunking, the operand
// pairing and the summation order -- exactly as it was.  Batching is a grid
// *re-layout*, not a re-association, so every slot reproduces the single-run
// answer bit for bit no matter which other slots ride along.
//
// A corollary the host side depends on: because the answer does not depend on
// the batch composition, the arena is free to launch with whichever instances
// happen to be waiting.  Instances never have to march in lock step.
// ---------------------------------------------------------------------------
class CudaBatchArena {
private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    CudaBatchArena(Geometry& geometry, int slots);
    ~CudaBatchArena();

    CudaBatchArena(const CudaBatchArena&)            = delete;
    CudaBatchArena& operator=(const CudaBatchArena&) = delete;

    [[nodiscard]] bool               available() const;
    [[nodiscard]] const std::string& status() const;
    [[nodiscard]] int                slots() const;
    [[nodiscard]] BackendCounters    counters() const;

    /// True when @p geometry has the same CMFD shape the arena was built for.
    /// The batch mode is restricted to a single geometry on purpose (the DB
    /// use case varies state, not the core).
    [[nodiscard]] bool compatible(Geometry& geometry) const;

    /// Claim/return one instance slot.  Thread-safe; -1 when the arena is full.
    int  acquireSlot();
    void releaseSlot(int slot);

    /// Record this outer's host buffers for @p slot.  No CUDA call and no lock:
    /// the owning thread only compares against its private upload shadow.
    void stage(int slot, const double* diag, const double* cc,
               const double* phi, const double* src);

    /// Inner-loop budget for @p slot.  `nmax` fixes the graph topology and must
    /// therefore agree across the batch; `eps` is per-slot device state.
    void setInner(int slot, int nmax, double eps);

    /// Rendezvous + launch + drain.  Returns once @p phi holds this slot's
    /// updated flux.  Whichever thread completes the waiting set launches for
    /// all of them; the rest sleep on a condition variable.
    void solve(int slot, double* phi);

    /// Host<->device contract of one device-resident CMFD sweep run
    /// (RASBERY_GPU_CMFD_SWEEP).  Inputs describe the state at delegation;
    /// outputs return the sweep loop's results.  `state`: 1 converged, 2 the
    /// wiel gamma degenerated and the host must finish the current sweep with
    /// the Rayleigh branch (gammad/gamman/err_acc carry the exported sums),
    /// 3 the sweep budget was spent, 0 only the launch unroll ran out.
    struct CmfdSweepIO {
        const double* chif  = nullptr; ///< [ig*nxyz+l]
        const double* xsnf  = nullptr; ///< [ig*nxyz+l]
        const double* vol   = nullptr; ///< [l]
        const double* udiag = nullptr; ///< [l*ng2+ige*ng+igs]
        double*       psi   = nullptr; ///< [l], in/out
        double eigv = 0, reigv = 0, reigvs = 0, errl2 = 0;
        double epsl2 = 0, eshift = 0;
        int    sweep_budget = 0, icmfd_budget = 0, icmfd_done = 0, ngxyz = 0;
        // outputs
        int    sweeps_done = 0, state = 0, negative_last = 0;
        double gammad = 0, gamman = 0, err_acc = 0;
    };

    /// Record one drive()'s sweep inputs for @p slot.  No CUDA call, no lock.
    void stageSweeps(int slot, const CmfdSweepIO& io);

    /// Rendezvous + one multi-sweep graph launch + drain; same batching
    /// contract as solve() but in its own rendezvous domain (a batch never
    /// mixes plain solves with sweep runs -- the graphs differ).
    void solveSweeps(int slot, double* phi, CmfdSweepIO& io);

    /// One JSON line of how full the batches actually were.  The batch mode is
    /// only worth its complexity when the mean width is close to the slot
    /// count, so this is the number to read first when the speed-up disappoints.
    void reportBatchOccupancy(const char* tag) const;

private:
    /// Shared rendezvous body of solve()/solveSweeps(); kind selects the
    /// batch domain and the launch sequence.
    void solveCommon(int slot, double* phi, int kind);

public:
};

// ---------------------------------------------------------------------------
// Process-wide batch-mode plumbing.  main() declares the batch width once,
// and the first BICGSolver that is constructed afterwards builds the arena
// from its own Geometry (the shape is not known before a deck is read).
// ---------------------------------------------------------------------------

/// Declare the batch width for this process.  0 disables the batch path.
void rasberySetBatchWidth(int slots);

/// The width declared above (0 when batch mode is off).
[[nodiscard]] int rasberyBatchWidth();

/// The process arena, created on first call from @p geometry.  Throws when a
/// later instance presents an incompatible geometry.
CudaBatchArena* rasberyBatchArena(Geometry& geometry);

/// Tear the arena down and report its aggregated counters (end of run).
void rasberyReleaseBatchArena();

} // namespace rasbery
