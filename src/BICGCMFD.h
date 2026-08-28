#pragma once
#include "BICGSolver.h"
#include "CMFD.h"
#include "HostPinRegistry.h"
#include "Nodal.h"
#include <memory>
#include <vector>

namespace rasbery {
/**
 * @brief Class for implementing combination of BICGCMFD + Nodal method
 *
 */

class BICGCMFD : public CMFD {
protected:
    /// @brief BICG solver object
    std::unique_ptr<BICGSolver> _ls;

    /// Rev.7.1 Task 9 link 2: the device outer segment owns dhat and psi in
    /// the sweep arena, so their H2D must be skipped.
    bool _outer_segment_resident = false;
    /// What the last device sweep observed, for the segment's probe.
    int  _last_sweep_negative = 0;
    int  _last_sweep_state    = 0;

    /// @brief Nodal object
    std::unique_ptr<Nodal> _nodal;

    /// @brief total number of CMFD sweeps since the last resetIteration(); diagnostic only
    int iter;

    /// @brief inner BiCGSTAB iterations issued since the last resetIteration();
    /// diagnostic only, and the `bicg_iters` field of the Sec 8 statepoint
    /// telemetry.
    ///
    /// Host path: exact -- one per BICGSolver::solve() call, so the early
    /// relative exit is reflected.  CUDA path: the captured graph budget,
    /// 1 + _nmaxbicg per sweep, because that is the number of iterations the
    /// device actually executes; the ones that run after an early halt are
    /// no-ops and are reported separately as BackendCounters::overrun_iterations.
    long long _bicg_iters;

    /// @brief Number of CMFD sweeps since the last resetIteration(), used ONLY to decide
    /// when the Wielandt extrapolation may take over from the Rayleigh-quotient warm-up.
    ///
    /// This used to be `iter` itself, which made a diagnostic counter silently steer the
    /// eigenvalue update: any caller that read, reset, or reused `iter` for reporting
    /// would have changed the Wielandt activation point. The two are now separate
    /// variables with the same reset point, so the numerical schedule is unchanged but
    /// no longer coupled to the reporting counter.
    int _wiel_sweep;

    /// @brief number of warm-up sweeps solved with the Rayleigh quotient before the
    /// Wielandt extrapolation is allowed to drive the eigenvalue
    static constexpr int WIELANDT_WARMUP_SWEEPS = 5;

    /// @brief the maximum number of bicg iteration (inner iteration)
    int _nmaxbicg;

    /// @brief the convergence criterion of bicg iteration (inner iteration)
    double _epsbicg;

    /// @brief the Wielandt shift value in CMFD acceleration
    double _eshift;

    /// @brief the unshifted diagonal matrix
    ///
    /// Page-exclusive storage (HostPinRegistry.h): this and the three staging
    /// vectors below are the four vector-backed ranges driveDeviceSweeps
    /// page-locks, and std::vector's default allocator packs them adjacently,
    /// which makes every one after the first unregisterable.
    PageExclusiveVector<double> _udiag;

    /// @brief group-major chif/xsnf and node volumes, materialized once per
    /// drive() for the device-resident sweep path (RASBERY_GPU_CMFD_SWEEP)
    PageExclusiveVector<double> _sweep_chif, _sweep_xsnf, _sweep_vol;

    /// @brief one-shot page-locking of the FIXED-address buffers the sweep path
    /// uploads (the raw arrays owned by Geometry/CMFD/XSSet)
    bool _sweep_pinned = false;

    /// @brief the addresses the four vector-backed host pin leases were taken
    /// on, so a resize that reallocates releases the old lease and acquires a
    /// new one instead of leaving a registration on freed memory (plan Sec 6.5).
    /// ~BICGCMFD releases exactly these.
    const void* _pin_udiag      = nullptr;
    const void* _pin_sweep_chif = nullptr;
    const void* _pin_sweep_xsnf = nullptr;
    const void* _pin_sweep_vol  = nullptr;

    /// The preceding setls() intentionally left diag/cc/udiag to the arena.
    bool _device_assembly_pending = false;

    [[nodiscard]] bool canUseDeviceAssembly() const;
    void assembleHostLinearSystem(const double& eigv);

    /// The three aliased host arrays every sweep launch of one drive reads.
    ///
    /// Resolved once by prepareDeviceSweeps and carried, rather than re-derived
    /// per launch, because the resolution includes the page-lock leases and the
    /// no-fission-spectrum chif fallback -- work that is per DRIVE, not per
    /// launch, and that the stream-ordered arm must not repeat between the
    /// enqueue and the observation.
    struct SweepPrep {
        const double* xsnf = nullptr;
        const double* chif = nullptr;
        const double* vol  = nullptr;
        bool          ok   = false;
    };

    /// Alias + page-lock this drive's sweep inputs.  False = no device sweep.
    bool prepareDeviceSweeps(double* flux, SweepPrep& p);
    /// Fill one launch's IO block.  `device_assembly` is the caller's to set.
    void stageSweepIO(CudaBatchArena::CmfdSweepIO& io, const SweepPrep& p, double eigv,
                      double reigv, double reigvs, double errl2, int iout, int icmfd,
                      bool psi_dirty);
    /// Absorb one launch; true when the DRIVE is over.
    bool absorbSweepLaunch(CudaBatchArena::CmfdSweepIO& io, double& eigv, double* flux,
                           double& errl2, double& reigv, double& reigvs, int& iout,
                           int& icmfd);

    /// @brief the device-resident sweep loop; true when it owned the whole
    /// drive, false when the caller must run the host loop from scratch
    bool driveDeviceSweeps(double& eigv, double* flux, double& errl2);

    /// What a stream-ordered drive has to carry from its enqueue to the
    /// observation that finishes it.  Exactly the locals driveDeviceSweeps keeps
    /// across one launch, and nothing else.
    struct EnqueuedDrive {
        bool      active              = false;
        bool      use_device_assembly = false;
        SweepPrep prep{};
        CudaBatchArena::CmfdSweepIO io{};
        double*   flux   = nullptr;
        double    reigv  = 0.0;
        double    reigvs = 0.0;
    };
    EnqueuedDrive _enqueued{};

public:
    // -------------------------------------------------------------------
    // Rev.7.1 Task 9 link 2: handing the CMFD arena to the device outer
    // -------------------------------------------------------------------
    //
    // These expose WHERE the sweep keeps its state, not what is in it.  The
    // segment needs the arena and the slot to ask for device addresses, and
    // it needs to know the resident sweep is actually the path being taken --
    // with RASBERY_GPU_CMFD_SWEEP off the host loop runs and the device flux
    // the segment reads is never written.
    [[nodiscard]] CudaBatchArena* residentArena() const {
        return _ls ? _ls->arena() : nullptr;
    }
    [[nodiscard]] int residentSlot() const { return _ls ? _ls->batchSlot() : -1; }

    /// Is the device-resident sweep the path drive() will take?
    ///
    /// It is the SAME predicate drive() uses (BICGCMFD.cpp:558-563) minus the
    /// per-call Wielandt warm-up test, plus the device assembly -- because the
    /// assembly is what reads the dhat the segment writes.  A second spelling
    /// of the gate would be free to disagree with the one that decides.
    [[nodiscard]] bool deviceSweepResident() const;

    /// Tell the sweep that dhat and psi are the segment's now.
    void setOuterSegmentResident(bool on) { _outer_segment_resident = on; }
    [[nodiscard]] bool outerSegmentResident() const { return _outer_segment_resident; }

    // -------------------------------------------------------------------
    // Rev.7.1 Task 10 part 2: the drive as an ENQUEUE
    // -------------------------------------------------------------------
    //
    // WHAT THIS REPLACES, AND WHAT IT DOES NOT.  drive() rendezvouses: it takes
    // the arena mutex, joins a batch, lingers, launches, DRAINS the stream, runs
    // the per-slot absorb and copies the flux mirror -- all of it per outer, on
    // the segment's critical path, and all of it before the segment can enqueue
    // anything else.  That is why OuterSegmentHooks::sweep_synchronizes was true
    // and the segment budget was forced to one.
    //
    // enqueueDrive is the same launch with the rendezvous and the drain removed.
    // It does NOT remove the segment's own synchronise: the nodal drive that
    // follows in the same outer is host arithmetic over Geometry::Jnet, so the
    // segment has to observe once per outer whatever this does.  What it removes
    // is the SECOND observation and the rendezvous around it, and it moves the
    // sweep's verdict (eigv, residual, the negative-flux and Rayleigh signals)
    // into a device kernel so the normal path publishes the segment's probe
    // without a readback at all.

    /// Would drive() take the device-resident sweep right now?  Same predicate,
    /// asked without running anything, so the segment can choose its arm.
    [[nodiscard]] bool canEnqueueDrive() const;

    /// Enqueue one drive on the arena's stream and return.  Nothing is drained
    /// and nothing is observed; @p probe names the device addresses the sweep
    /// verdict kernel publishes into.  False = nothing was enqueued and the
    /// caller must take the blocking drive().
    bool enqueueDrive(double& eigv, double* flux, double& errl2,
                      const CudaBatchArena::CmfdSweepProbeSink& probe);

    /// Finish the drive enqueueDrive started.  THE CALLER MUST HAVE
    /// SYNCHRONISED sweepStream().  Sets @p host_continued when the drive was
    /// not over at that observation and this call ran the remaining (blocking)
    /// launches -- in which case the device probe is stale and the caller must
    /// republish it from lastSweep*() and clear its halt.
    bool finishDrive(double& eigv, double* flux, double& errl2, bool& host_continued);

    /// The stream enqueueDrive issues on.  Null when there is no arena.
    [[nodiscard]] void* sweepStream() const { return _ls ? _ls->sweepStream() : nullptr; }

    /// Drain that stream, for a caller that must fall back to a blocking drive
    /// with stream-ordered work of its own still in flight.
    void syncSweepStream() { if (_ls) _ls->syncSweepStream(); }

    /// The two device-only signals the segment's transition ranks.
    [[nodiscard]] bool lastSweepNegativeFlux() const { return _last_sweep_negative != 0; }
    [[nodiscard]] bool lastSweepRayleigh() const { return _last_sweep_state == 2; }

    BICGCMFD(Geometry& g, XSSet& x);

    ~BICGCMFD() override;

    [[nodiscard]] int innerIterations() const { return iter; }

    /// @brief inner BiCGSTAB iterations since the last resetIteration()
    [[nodiscard]] long long bicgIterations() const { return _bicg_iters; }

    /// @brief the CUDA backend's cumulative counters, whichever backend this
    /// instance actually uses.  Zeroed when there is no CUDA backend at all.
    ///
    /// In batch mode the arena is process-wide, so the numbers cover EVERY slot
    /// -- see batchSlot() -- and the copy is taken without the arena lock, so a
    /// concurrent launcher can be mid-update.  Both are acceptable only because
    /// the sole consumer is diagnostic (the Sec 8 telemetry, which publishes
    /// `counters_shared` alongside the deltas); nothing numerical may read this.
    [[nodiscard]] BackendCounters backendCounters() const;

    /// @brief this instance's batch-arena slot, or -1 when it is not in the
    /// arena (single-instance CUDA or CPU).  >= 0 therefore also means
    /// backendCounters() is shared with the other decks in this process.
    [[nodiscard]] int batchSlot() const;

    /// @brief set the iteration limit for BICGCMFD
    /// @param maxls maximum number of iteration in BICG calculation
    /// @param epsls convergence criterion in BICG calculation
    void setIterLim(int maxls, double epsls);

    /// @brief update d_tilde for CMFD
    void upddtil() override;

    /// @brief update d_hat for CMFD
    void upddhat(double* flux, double* jnet) override;

    /// @brief setup the linear system for CMFD
    /// @param eigv the eigenvalue
    void setls(const double& eigv) override;

    /// @brief update the net current for CMFD
    /// @param flux the flux
    /// @param jnet the net current
    void updjnet(double* flux, double* jnet) override;

    /// @brief update the fission source term for CMFD
    /// @param flux the flux
    void updpsi(const double* flux) override;

    /// @brief drive the CMFD calculation
    /// @param eigv the eigenvalue
    /// @param flux the flux
    /// @param errl2 the L2 norm of the error
    void drive(double& eigv, double* flux, double& errl2) override;


    /// @brief reset the iteration count
    void resetIteration();

    /// @brief set the Wielandt shift value
    void setEshift(double eshift0);

    /// @brief update linear system of CMFD with Wielandt shift value
    /// @param reigvs the reciprocal of shifted eigenvalue
    void updls(const double& reigvs);

    /// @brief setup the element of linear system for CMFD
    /// @param l the node index
    void setls(const int& l);

    /// @brief update the element of the linear system for CMFD
    /// @param l the node index
    /// @param reigvs the reciprocal of shifted eigenvalue
    void updls(const int& l, const double& reigvs);



    /// @brief perform the Wielandt shift to accelerate CMFD calculation
    /// @param icy the iteration count
    /// @param flux the flux
    /// @param reigvs the reciprocal of shifted eigenvalue
    /// @param eigv the eigenvalue
    /// @param reigv the reciprocal of eigenvalue
    /// @param errl2 the L2 norm of the error
    void wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, double& errl2);

    /// @brief the unshifted diagonal matrix
    /// @param igs from-group index
    /// @param ige to-group index
    /// @param l the node index
    double& udiag(const int& igs, const int& ige, const int& l) {
        return _udiag[l * _g.ng2() + ige * _g.ng() + igs];
    };

    /// @brief the Wielandt shift value
    double& eshift() {
        return _eshift;
    };
};
} // namespace rasbery
