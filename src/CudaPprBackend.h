#pragma once

// Device backend for pin-power reconstruction -- GA evaluator plan Sec 6.3
// Task 10 (promoted from Rev.7.1 Task 19b), behind RASBERY_GPU_PPR (default
// off).
//
// WHAT IT COVERS, AND WHY EXACTLY THAT.  The plan attributes the 0.474 s/
// statepoint host floor to "PPR / CRAM depletion / FlatXS / T-H / result
// packing" from reading the code.  The floor_wall receipt (Driver.h, sptelem
// PH_PPR_* / PH_DEPL_*) measured it instead, on the v3 arm, kngr_238, 35
// statepoints, 1080 Ti:
//
//     ppr_reset        0.560 s     15.99 ms/sp    10.1 % of PPR
//     ppr_drive        4.676 s    133.60 ms/sp    84.5 % of PPR
//     ppr_recon        0.297 s      8.48 ms/sp     5.4 % of PPR
//     ---------------------------------------------------------
//     PPR total        5.532 s    158.07 ms/sp    ~50 % of the floor
//
// So this backend ports `reset` and `drive` -- 94.6 % of PPR -- and leaves
// `reconstructPinPower` on the host.  That is not a stopping point chosen for
// convenience:
//
//   * reconstructPinPower costs 8.5 ms/statepoint, i.e. 1.0 % of the case.
//   * Its per-assembly-plane inputs are the burnup-interpolated Chiffon form
//     functions (fmap/gmap), which live in std::map-keyed depletion points and
//     change every statepoint.  Feeding them to a kernel means either ~38 MB of
//     H2D per statepoint (interpolated maps) or a device-resident registry of
//     every depletion point plus an on-device interpolation.  Both cost more
//     engineering, and the first costs more wall, than the 0.3 s/run on offer.
//   * The host reconstruction already runs under OpenMP over (plane, assembly)
//     and is the only part of PPR that does.
//
// The coefficients the host reconstruction reads (_p, _a, _bt, _c) come back
// D2H once per statepoint -- 5.8 MB, ~1 ms -- which is the entire per-statepoint
// device->host traffic this arm adds.  Fq / FdH / the pin map stay exactly where
// they were computed before.
//
// WHAT THE 100-ITERATION DRIVE IS.  Neither a fixed sweep nor a per-assembly
// batched solve.  `updateCorner` reads the fitting coefficients of the 3x3 node
// stencil, which crosses assembly boundaries, so a per-assembly kernel would
// need a halo exchange every iteration and would not be independent.  It is a
// GLOBAL Picard iteration on the corner fluxes with a global convergence test,
// and it is driven from the host exactly as before: seven kernels per iteration
// on one stream, then one small D2H of the corner-sum partials so the host
// applies the same RelativeChange test against the same 1e-5 tolerance and
// breaks on the same condition.  The iteration count is therefore an observable,
// not an assumption -- pprIterations() reports it.
//
// CLASS N1, NOT B0.  The per-node arithmetic is written in the host's order and
// the file is compiled with --fmad=false, but `exp` is the device libm's, and
// the corner-sum reduction is the fixed 256-chunk deterministic partition
// (Geometry.h rasbery_det_chunks) rather than the host's single sequential
// accumulation.  Run-to-run the arm is deterministic -- one thread per output,
// no atomics, a fixed partition -- but it is not bit-identical to the host, and
// the gate is Gate A (pin power max relative difference, Fq/FdH deltas), not
// h5diff.
//
// PER SLOT, NOT PER PROCESS.  One backend instance is owned by one PPR object,
// which is a local of Driver::Drive().  --batch-mode gives every deck its own
// Driver, hence its own PPR, hence its own backend, stream and device buffers.
// There is no process-wide state here at all: that is the slot-0 bug class this
// design refuses to re-enter.
//
// FAIL OPEN.  Any CUDA failure, an unsupported deck (ng != 2), or
// RASBERY_PPR_MODE=master makes the entry point return false and the caller runs
// the untouched host reset+drive.  The receipt counts those as host_fallbacks.

#include <cstddef>
#include <memory>
#include <string>

namespace rasbery {

namespace ppr {

/// Geometry that never changes for a deck.  Uploaded once, on first call.
struct GeomView {
    int                  ng     = 0;
    int                  nxyz   = 0;
    int                  nxy    = 0;
    int                  nsurf  = 0;
    const double*        hmesh  = nullptr; ///< [nxyz * NDIRMAX]
    const int*           lktosfc = nullptr; ///< [(nxyz * NDIRMAX) * LR]
    const int*           neibrb = nullptr;  ///< [nxy * NEWS]
    const unsigned char* is_fuel = nullptr; ///< [nxyz], 1 = fuel
};

/// Everything that changes at a statepoint, plus the host destinations the
/// coefficients are copied back into.  All XS arrays are the [ig * nxyz + l]
/// layout XSSet publishes; xssm is [(igs * ng + ige) * nxyz + l].
struct StepView {
    double        reigv = 0.0;
    const double* phif  = nullptr; ///< [nxyz * ng]
    const double* phis  = nullptr; ///< [nsurf * ng]
    const double* jnet  = nullptr; ///< [nsurf * ng]
    const double* xsdf  = nullptr; ///< [ng * nxyz]
    const double* xsrf  = nullptr;
    const double* xsnf  = nullptr;
    const double* xssm  = nullptr; ///< [ng * ng * nxyz]
    const double* chif  = nullptr; ///< [ng * nxyz]
    const double* crdf  = nullptr; ///< [nxyz * ng], all 1.0 unless RASBERY_PPR_CRDF

    // Host coefficient arrays (Geometry-owned).  Filled on success.
    double* phic = nullptr; ///< [nxyz * 4 * ng]
    double* p    = nullptr; ///< [nxyz * 15 * ng]
    double* a    = nullptr; ///< [nxyz * 8 * ng]
    double* c    = nullptr; ///< [nxyz * 15 * ng]
    double* q    = nullptr; ///< [nxyz * 15 * ng]
    double* l    = nullptr; ///< [nxyz * 9 * ng]
    double* bt   = nullptr; ///< [nxyz * ng]
};

} // namespace ppr

/// One instance per PPR object (per Driver, per batch slot).
class PprBackend {
public:
    PprBackend();
    ~PprBackend();

    PprBackend(const PprBackend&)            = delete;
    PprBackend& operator=(const PprBackend&) = delete;

    /// RASBERY_GPU_PPR is set to a truthy value AND a device exists.
    [[nodiscard]] bool available() const;

    /// Human-readable reason, for the one-line receipt.
    [[nodiscard]] const std::string& status() const;

    /// PPR::reset + PPR::drive(niter) on the device.  Returns false and leaves
    /// every host array untouched if anything at all goes wrong, so the caller
    /// can run the host path.  On success `*iters` gets the number of
    /// corner-balance iterations actually executed.
    bool resetAndDrive(const ppr::GeomView& geom, const ppr::StepView& step,
                       int niter, int* iters);

    /// Receipt counters.
    [[nodiscard]] unsigned long long statepoints() const;
    [[nodiscard]] unsigned long long iterations() const;
    [[nodiscard]] double             wallMs() const;
    [[nodiscard]] int                deviceOrdinal() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rasbery
