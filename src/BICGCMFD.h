#pragma once
#include "BICGSolver.h"
#include "CMFD.h"
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
    std::vector<double> _udiag;

    /// @brief group-major chif/xsnf and node volumes, materialized once per
    /// drive() for the device-resident sweep path (RASBERY_GPU_CMFD_SWEEP)
    std::vector<double> _sweep_chif, _sweep_xsnf, _sweep_vol;

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

    /// @brief the device-resident sweep loop; true when it owned the whole
    /// drive, false when the caller must run the host loop from scratch
    bool driveDeviceSweeps(double& eigv, double* flux, double& errl2);

public:
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
