#pragma once

#include "Geometry.h"
#include "CudaBICGBackend.h"
#include "milk.h"

namespace rasbery {

/**
 * @brief BICGStab solver class
 * see : https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
 *
 */
class BICGSolver {
private:
    /// @brief Geometry object
    Geometry& _g;

    /// @brief alpha in BICGStab
    double _calpha;

    /// @brief beta in BICGStab
    double _cbeta;

    /// @brief rho in BICGStab
    double _crho;

    /// @brief omega in BICGStab
    double _comega;

    /// y in BICGStab
    milk::Vector<double> _vy;

    /// z in BICGStab
    milk::Vector<double> _vz;

    /// @brief r in BICGStab
    milk::Vector<double> _vr;

    /// @brief r0 in BICGStab
    milk::Vector<double> _vr0;

    /// @brief p in BICGStab
    milk::Vector<double> _vp;

    /// @brief v in BICGStab
    milk::Vector<double> _vv;

    /// @brief s in BICGStab
    milk::Vector<double> _vs;

    /// @brief t in BICGStab
    milk::Vector<double> _vt;

    /// @brief SSOR: inverse of 2x2 diagonal blocks [nxyz * ng2]
    milk::Vector<double> _dinv;

    /// @brief Compressed per-node neighbor lists for the matvec; geometry-static,
    /// built lazily on the first axb call.
    std::vector<int> _neib_count;
    std::vector<int> _neib_node;
    std::vector<int> _neib_slot;

    /// @brief SSOR: forward sweep workspace [nxyz * ng]
    milk::Vector<double> _ssor_tmp;

    /// @brief Pointer to current diagonal (saved from facilu)
    double* _diag_ptr;

    /// Optional CUDA-resident block-Jacobi BiCGSTAB backend.
    std::unique_ptr<CudaBICGBackend> _cuda;
    bool                             _use_cuda;

    /// Multi-instance batch mode: instead of owning a private backend, this
    /// solver holds one slot of the process-wide arena and its CMFD solves
    /// ride along with the other instances' solves in one grid.  `_arena` and
    /// `_cuda` are mutually exclusive.
    CudaBatchArena* _arena;
    int             _batch_slot;

public:
    /// @brief Construct a new BICGStab object
    BICGSolver(Geometry& g);

    /// @brief  Destroy the BICGStab object
    virtual ~BICGSolver();

    /// @brief reset the BICGStab calculation
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    /// @param src the source term
    /// @param r20 the initial residual
    void reset(double* diag, double* cc, double* phi, double* src, double& r20);

    /// @brief reset the BICGStab calculation for each group and node
    /// @param ig group index
    /// @param l node index
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    /// @param src the source term
    /// @return the initial residual
    double reset(const int& ig, const int& l, double* diag, double* cc, double* phi, double* src);

    /// @brief Apply SSOR preconditioner: sol = M^{-1} * b
    /// @param cc the coupling coefficient
    /// @param b the right-hand side
    /// @param x the solution
    void minv(double* cc, double* b, double* x);

    /// @brief Compute SSOR diagonal block inverses
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    void facilu(double* diag, double* cc);

    /// @brief solve the BICGStab calculation
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param r20 the initial residual
    /// @param phi the flux
    /// @param r2 the final residual
    void solve(double* diag, double* cc, double& r20, double* phi, double& r2);

    /// @brief Run the whole inner BiCGSTAB loop on the device (CUDA path only)
    ///
    /// Enqueues 1 + @p nmax iterations whose relative exit test
    /// ||r||/||r0|| < @p eps is evaluated on the GPU, so the loop costs one
    /// graph launch and no host round trip. The CPU path keeps driving the
    /// same loop from solve() in the caller.
    /// @param nmax the extra iteration budget (_nmaxbicg)
    /// @param eps the relative residual tolerance (_epsbicg)
    void solveInner(int nmax, double eps);

    /// Copy the resident CUDA flux to the host at a CMFD observation boundary.
    void synchronizeCudaFlux(double* phi);

    /// Run a device-resident CMFD sweep batch (RASBERY_GPU_CMFD_SWEEP).
    /// Requires the batch arena; returns false when unavailable so the caller
    /// keeps the host sweep loop.  reset()/solveInner() must have staged this
    /// outer's operator and inner budget first, exactly as for solve().
    bool driveSweepsCuda(double* phi, CudaBatchArena::CmfdSweepIO& io);

    /// Rev.7.1 Task 10 part 2: the same sweep run, ENQUEUED and not drained.
    ///
    /// Returns false when the arena cannot serve a single-participant launch, in
    /// which case the caller must take driveSweepsCuda.  On true, nothing has
    /// been observed yet: the caller synchronises sweepStream() when it is ready
    /// and calls finishSweepsCuda() to read the outcome.
    /// @p caller_stream: see CudaBatchArena::enqueueSweeps.
    bool enqueueSweepsCuda(double* phi, const CudaBatchArena::CmfdSweepIO& io,
                           const CudaBatchArena::CmfdSweepProbeSink& probe,
                           void* caller_stream);

    /// The post-synchronise half of enqueueSweepsCuda.  False = non-finite flux.
    bool finishSweepsCuda(CudaBatchArena::CmfdSweepIO& io);

    /// The stream enqueueSweepsCuda issues on, so the caller can order its own
    /// work against it.  Null when there is no arena.
    [[nodiscard]] void* sweepStream() const;

    /// Drain that stream.  See CudaBatchArena::syncSweepStream.
    void syncSweepStream();

    /// @brief calculate Axb in the BICGStab calculation
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    /// @param aphi the result of Axb
    void axb(double* diag, double* cc, double* phi, double* aphi);


    /// @brief calculate an element of Axb calculation in the BICGStab calculation
    /// @param ig group index
    /// @param l node index
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    double axb(const int& ig, const int& l, double* diag, double* cc, double* phi);

    [[nodiscard]] bool usingCuda() const { return _use_cuda; }
    [[nodiscard]] CudaBatchArena* arena() const { return _arena; }
    /// Batch-arena slot, or -1 when this instance is not in the arena.
    [[nodiscard]] int batchSlot() const { return _batch_slot; }
    [[nodiscard]] BackendCounters cudaCounters() const {
        return _cuda ? _cuda->counters() : BackendCounters{};
    }
};
}
