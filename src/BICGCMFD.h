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
public:
    // -------------------------------------------------------------------
    // WP1 follow-up (plan Sec 6.3): WHY a drive could not be enqueued
    // -------------------------------------------------------------------
    //
    // canEnqueueDrive() answers `will it`, and for the fail-closed gate that is
    // not enough: the outer segment's enqueue hook falls back to the BLOCKING
    // drive() when it is false, and drive() then runs the pristine host
    // BiCGSTAB loop -- CPU numerics.  Under RASBERY_GPU_FULL that must fail the
    // case, EXCEPT for the one reason that is host BY DESIGN and has no device
    // implementation to decline: the Rayleigh warm-up.  A single boolean cannot
    // tell those apart, so the gate is handed the reason instead.
    //
    // WHY THE LADDER LIVES HERE AND canEnqueueDrive() DELEGATES TO IT.  Two
    // spellings of one predicate is how a segment arms an arm drive() would not
    // have taken; the enum IS the predicate now, and `will it` is `== None`.
    enum class EnqueueRefusal : int {
        None = 0,
        SweepArmOff,     ///< RASBERY_GPU_CMFD_SWEEP unset or 0
        NoCudaSolver,    ///< no BICGSolver, or it is not the CUDA one
        NotTwoGroup,     ///< _g.ng() != 2; the device sweep serves 2 groups
        WielandtWarmup,  ///< _wiel_sweep < WIELANDT_WARMUP_SWEEPS
        StagePrepFailed, ///< the gate said yes and prepareDeviceSweeps refused
        Count
    };

    static const char* enqueueRefusalName(EnqueueRefusal r) {
        switch (r) {
            case EnqueueRefusal::None:            return "none";
            case EnqueueRefusal::SweepArmOff:     return "sweep_arm_off";
            case EnqueueRefusal::NoCudaSolver:    return "no_cuda_solver";
            case EnqueueRefusal::NotTwoGroup:     return "not_two_group";
            case EnqueueRefusal::WielandtWarmup:  return "wielandt_warmup";
            case EnqueueRefusal::StagePrepFailed: return "stage_prep_failed";
            case EnqueueRefusal::Count:           break;
        }
        return "?";
    }


protected:
    /// @brief BICG solver object
    std::unique_ptr<BICGSolver> _ls;

    /// Rev.7.1 Task 9 link 2: the device outer segment owns dhat and psi in
    /// the sweep arena, so their H2D must be skipped.
    bool _outer_segment_resident = false;
    /// What the last device sweep observed, for the segment's probe.
    int  _last_sweep_negative = 0;
    int  _last_sweep_state    = 0;
    /// Did the drive that just returned leave the DEVICE flux equal to the host
    /// one?  True only when the device sweep finished and issueFluxDownloads
    /// wrote Geometry::Phif FROM the device phi.  False for the Wielandt
    /// warm-up, a declined enqueue, and the pristine host loop.
    ///
    /// NOT SUFFICIENT ON ITS OWN for an upload elision -- it says the device
    /// downloaded the flux, not that the host has left it alone since -- which
    /// is why its consumer pairs it with Geometry::fluxGeneration().
    bool _last_drive_device_flux = false;
    /// Bumped by upddtil(), the only writer of _dtil.
    unsigned long long _dtil_generation = 1;
    /// WP1 follow-up: why the last enqueueDrive() refused.  See
    /// EnqueueRefusal; `None` until the first refusal, which is also what a run
    /// that never called enqueueDrive() reports.
    EnqueueRefusal _enqueue_refusal = EnqueueRefusal::None;

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

public:
    /// A2 S0 (docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md Sec 1.5-1.6, Sec 3.1
    /// S0, Sec 5 items 4-5).  WHY ONE DRIVE ENDED, and how many of its sweeps
    /// nothing charged for.
    ///
    /// The campaign has `cmfd_sweeps / outers` = 4.256 against a budget of 5 and
    /// no way to read what that ratio MEANS.  If most drives stop because
    /// `errl2 < _epsl2` then relaxing `_epsl2` buys sweeps (design S3); if most
    /// stop because the budget ran out, a drive that cannot reach 1e-6 in five
    /// sweeps will not reach 5e-6 either and S3 is dead before it is written.
    /// That single number is S3's K0 gate and there was no counter behind it.
    ///
    /// `negative_retry_sweeps` is the design's Sec 1.6 loss: a sweep that
    /// produced a negative flux is re-run without consuming the budget (up to
    /// `20 * _ncmfd`), so it is a real launch that appears in NO outer-count
    /// receipt.  `icmfd_done - sweeps_done` has been carried in CmfdSweepIO all
    /// along and simply never read.
    ///
    /// `deferred_drives` / `deferred_sweeps` are the honest hole.  On the device
    /// outer-segment arm the verdict kernel sums a whole SEGMENT of drives into
    /// one Accum whose `state` is only the LAST launch's, so the per-drive exit
    /// of the others cannot be recovered without a device-side histogram -- and
    /// S0 opens no .cu file.  They are reported separately rather than folded in
    /// so the exit ratio's denominator is on the face of the receipt, and their
    /// `sweeps / drive` is the usable proxy: at the budget it IS exhaustion.
    struct DriveExits {
        long long drives                = 0; ///< drives whose exit the host classified
        long long converged             = 0; ///< ended on tolerance: errl2 < _epsl2
        long long budget                = 0; ///< ended on the sweep budget instead
        long long aborted               = 0; ///< a launch refused mid-drive; caller fell back
        long long deferred_drives       = 0; ///< segment-summed drives, exit unattributed
        long long deferred_sweeps       = 0; ///< their charged sweeps, for sweeps/drive
        long long sweeps_charged        = 0; ///< sweeps that consumed the drive budget
        long long negative_retry_sweeps = 0; ///< sweeps re-run for a negative flux
    };

protected:
    /// See DriveExits.  Zeroed by resetIteration() with the other diagnostics,
    /// so what it holds is one SolveLoop's own total.
    DriveExits _drive_exits{};

    /// Charge one finished drive to `converged` or `budget`.
    ///
    /// ONE TEST, EVERY PATH, AND IT IS THE DRIVE'S OWN.  `errl2 < _epsl2` is the
    /// exit condition the host loop breaks on, the condition the device kernel
    /// raises sweep state 1 for (cmfd_sweep_end, and state 1 takes precedence
    /// over state 3), and the condition the Rayleigh hand-back returns true on.
    /// Reading it once at the exit therefore classifies a device drive, a host
    /// drive and a hand-back by the same rule, without the caller having to know
    /// which one it took -- and it is exactly the question S3's K0 gate asks.
    void chargeDriveExit(double errl2);

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

    /// The ladder, ranked so the first reason a reader would ask about is the
    /// one reported.  Pure: it runs nothing and consumes nothing.
    [[nodiscard]] EnqueueRefusal enqueueRefusal() const;

    /// Would drive() take the device-resident sweep right now?  Same predicate,
    /// asked without running anything, so the segment can choose its arm.
    [[nodiscard]] bool canEnqueueDrive() const {
        return enqueueRefusal() == EnqueueRefusal::None;
    }

    /// Why the LAST enqueueDrive() returned false.  Set at both of its refusal
    /// points, so a caller that already paid for the call does not re-ask the
    /// ladder and get a different answer than the one that decided.
    [[nodiscard]] EnqueueRefusal lastEnqueueRefusal() const { return _enqueue_refusal; }

    /// Enqueue one drive on the arena's stream and return.  Nothing is drained
    /// and nothing is observed; @p probe names the device addresses the sweep
    /// verdict kernel publishes into.  False = nothing was enqueued and the
    /// caller must take the blocking drive().
    /// @p caller_stream is the stream the caller's own work rides; see
    /// CudaBatchArena::enqueueSweeps for what passing a different one buys.
    bool enqueueDrive(double& eigv, double* flux, double& errl2,
                      const CudaBatchArena::CmfdSweepProbeSink& probe,
                      void* caller_stream = nullptr);

    /// Finish the drive enqueueDrive started.  THE CALLER MUST HAVE
    /// SYNCHRONISED sweepStream().  Sets @p host_continued when the drive was
    /// not over at that observation and this call ran the remaining (blocking)
    /// launches -- in which case the device probe is stale and the caller must
    /// republish it from lastSweep*() and clear its halt.
    bool finishDrive(double& eigv, double* flux, double& errl2, bool& host_continued);

    // -------------------------------------------------------------------
    // Rev.7.1 Task 10 part 3: the observation, once per SEGMENT
    // -------------------------------------------------------------------
    //
    // WHY finishDrive COULD NOT SIMPLY BE MOVED.  It reads the arena's staging
    // block, and a segment that enqueues N drives back to back overwrites that
    // block N times -- the last writer being, on a segment that halted, a
    // launch whose every kernel was masked.  So what the host needs at the exit
    // is not the last block, it is a SUMMARY the device kept as it went:
    // CudaBatchArena::CmfdSweepProbeSink::Accum.
    //
    // WHAT THIS RECONSTRUCTS, AND FROM WHERE.
    //   iter / _wiel_sweep / _bicg_iters   the accumulator's `attempts` sum.
    //                     _wiel_sweep is the one with teeth: canEnqueueDrive()
    //                     gates on it and resetIteration() zeroes it per
    //                     statepoint, so it must end the segment holding exactly
    //                     what N per-outer observations would have left.
    //   _last_sweep_state / _last_sweep_negative   the accumulator's record of
    //                     the last launch whose verdict actually ran.
    //   _last_drive_device_flux   true on states 1 and 3, which is
    //                     absorbSweepLaunch's own rule -- issueFluxDownloads
    //                     wrote Geometry::Phif from the device phi at the end of
    //                     every launch, so the two agree byte for byte.
    //   eigv / errl2      NOT from here.  The segment reconstructs them from
    //                     DeviceSlotState at its exit observation, which is the
    //                     same numbers by a shorter path; they move only on the
    //                     exceptional branch below, where the host finishes the
    //                     drive and therefore owns them again.
    //
    // @p acc must be the host copy of the device accumulator the segment zeroed
    //        at its entry and handed to every enqueueDrive of the segment.
    /// Sets @p host_continued when a launch ended in sweep state 0 or 2 and this
    /// call ran the remaining blocking launches -- same contract as finishDrive,
    /// including that the caller must then republish the segment's probe and
    /// re-issue the steps the verdict's halt swallowed.
    bool finishDeferredDrives(const CudaBatchArena::CmfdSweepProbeSink::Accum& acc,
                              double& eigv, double* flux, double& errl2,
                              bool& host_continued);

    /// The stream enqueueDrive issues on.  Null when there is no arena.
    [[nodiscard]] void* sweepStream() const { return _ls ? _ls->sweepStream() : nullptr; }

    /// Drain that stream, for a caller that must fall back to a blocking drive
    /// with stream-ordered work of its own still in flight.
    void syncSweepStream() { if (_ls) _ls->syncSweepStream(); }

    /// The two device-only signals the segment's transition ranks.
    [[nodiscard]] bool lastSweepNegativeFlux() const { return _last_sweep_negative != 0; }
    [[nodiscard]] bool lastSweepRayleigh() const { return _last_sweep_state == 2; }

    /// See _last_drive_device_flux.
    [[nodiscard]] bool lastDriveLeftDeviceFlux() const { return _last_drive_device_flux; }
    /// See _dtil_generation.
    [[nodiscard]] unsigned long long dtilGeneration() const { return _dtil_generation; }

    BICGCMFD(Geometry& g, XSSet& x);

    ~BICGCMFD() override;

    [[nodiscard]] int innerIterations() const { return iter; }

    /// @brief inner BiCGSTAB iterations since the last resetIteration()
    [[nodiscard]] long long bicgIterations() const { return _bicg_iters; }

    /// @brief why this SolveLoop's drives ended.  See DriveExits.
    [[nodiscard]] const DriveExits& driveExits() const { return _drive_exits; }

    /// @brief the inner BiCGSTAB budget this instance resolved (A2 S2 scoring:
    /// the only honest way to price nmax is `outers * cmfd_sweeps * (1 + nmax)`,
    /// because `bicg_iters` is DERIVED from it on the device arm).
    [[nodiscard]] int innerBudget() const { return _nmaxbicg; }
    /// @brief the CMFD sweep budget per drive (`_ncmfd`), the denominator the
    /// budget-exhausted fraction is read against.
    [[nodiscard]] int sweepBudget() const { return _ncmfd; }

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
    void upddhat(const double* flux, double* jnet) override;

    /// @brief setup the linear system for CMFD
    /// @param eigv the eigenvalue
    void setls(const double& eigv) override;

    /// @brief update the net current for CMFD
    /// @param flux the flux
    /// @param jnet the net current
    void updjnet(const double* flux, double* jnet) override;

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
