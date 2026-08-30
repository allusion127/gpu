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
//
// ===========================================================================
// WP6 (bottleneck plan 20260830 Sec WP6) -- WHAT STAGES B/C/E CHANGED
// ===========================================================================
//
// STAGE B, THE STOPPING TEST MOVED TO THE DEVICE.  The loop above paid one
// `cudaStreamSynchronize` PER PICARD ITERATION so the host could fold 4x256
// partials and apply RelativeChange.  On kngr_238 that is ~50 syncs per
// statepoint and the drive is serialised behind every one of them.  The fold
// and the test are now a kernel (`kCornerFoldAndCheck<<<1,4>>>`), and the loop
// is driven one of three ways -- RASBERY_GPU_PPR_DEVICE_LOOP / _GRAPH:
//
//     host_sync      the c502856 loop, kept verbatim for the A/B
//     device_stream  niter bodies enqueued; every body kernel returns
//                    immediately once the device flag is raised  (default)
//     device_graph   ONE body captured into a conditional WHILE, armed and
//                    re-armed by the same predicate kernel     (opt-in)
//
// THE ITERATION COUNT IS IDENTICAL TO THE HOST'S, and that is a property of
// the fold, not a hope:
//
//   * the fold is the SAME association -- one thread per corner, chunk 0 to
//     nchunk-1 ascending, which is byte for byte the loop the host ran over
//     the D2H'd partials;
//   * `iters` is incremented by the fold kernel and only when the flag is
//     down, so it counts exactly the rounds the host's `citer` would have;
//   * a `device_stream` body enqueued AFTER convergence writes nothing -- every
//     kernel in it returns on the flag -- so the extra launches are no-ops and
//     the final state is the state at the converged round.  The batch does not
//     "run extra iterations that change the result": it runs empty ones.  That
//     is what buys exactness at the price of launches, and it is why K-batched
//     checking (which does run real extra rounds) was NOT chosen.
//
// The WHILE's predicate is `iters < niter && converged == 0`, evaluated by the
// same kernel in the arm node and as the body's last node -- the do-while the
// host loop is.
//
// STAGE C, THE INPUTS STOPPED BEING RE-UPLOADED.  Three separate reductions,
// and only the first is conditional:
//
//   * chif and crdf are gated on GENERATIONS (XSSet::refGeneration and a PPR
//     local counter).  chif is the burnup-interpolated fission spectrum, which
//     moves when the library reference blocks are rebuilt, not per statepoint;
//     crdf is all-ones unless RASBERY_PPR_CRDF.  Both are unconditional wins.
//   * phif / phis / jnet are BORROWED from the canonical nodal set when the
//     outer segment holds one (RASBERY_GPU_PPR_CANONICAL).  That is the whole
//     surface traffic of the arm.
//   * RASBERY_GPU_PPR_CANONICAL=verify borrows AND uploads the host copy into
//     a scratch buffer and compares elementwise on the device, reporting
//     `canonical_mismatch`.  The borrow is only sound if the device buffers
//     hold what the host arrays hold at PPR time; `verify` is how 238 answers
//     that instead of the header asserting it.
//
// The receipt carries `h2d_bytes` and `h2d_bytes_elided` so the reduction is a
// measurement rather than a claim, per slot.
//
// STAGE E, THE BATCH ARENA IS STILL CONDITIONAL, AND THE RECEIPT IS WHY.  The
// plan makes `CudaPprArena` conditional on PPR exceeding 10 % of the M64
// profile.  What a per-slot backend has to prove in the meantime is that it
// does not allocate per STATEPOINT -- `allocations` counts every cudaMalloc
// this instance has ever made, so `allocations == 21 + k` for any number of
// statepoints is the observable, and a slot that re-shaped mid-run shows up as
// `reallocations > 0` rather than as a wall-time mystery.

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

namespace rasbery {

namespace ppr {

/// Stage C.  How much of the borrow the caller is asking for.
enum class CanonicalMode : int {
    Off = 0,  ///< upload everything, as before (default)
    Borrow,   ///< use the device-resident nodal set, skip the three uploads
    Verify    ///< borrow AND upload-and-compare, reporting the mismatch count
};

/// RASBERY_GPU_PPR_CANONICAL, read once.  DEFAULT OFF, because the borrow's
/// premise -- that the device buffers hold at PPR time what the host arrays
/// hold -- is a property of the outer segment's exit, not of this arm, and
/// `verify` is how a deck establishes it.  Anything else truthy is `Borrow`.
inline CanonicalMode canonicalModeFromEnv() {
    static const CanonicalMode mode = [] {
        const char* v = std::getenv("RASBERY_GPU_PPR_CANONICAL");
        if (v == nullptr) return CanonicalMode::Off;
        const std::string s(v);
        if (s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
            s == "FALSE")
            return CanonicalMode::Off;
        if (s == "verify" || s == "VERIFY" || s == "2") return CanonicalMode::Verify;
        return CanonicalMode::Borrow;
    }();
    return mode;
}

inline const char* canonicalModeName(CanonicalMode m) {
    switch (m) {
        case CanonicalMode::Off:    return "off";
        case CanonicalMode::Borrow: return "borrow";
        case CanonicalMode::Verify: return "verify";
    }
    return "?";
}

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

    // --- WP6 stage C -------------------------------------------------------
    //
    // BORROWED DEVICE POINTERS, all-or-nothing.  A partial set would pair the
    // segment's device jnet with a freshly uploaded host phif -- two different
    // outer iterations, silently blended -- which is the shape
    // gpu::canonicalNodalSetIsCoherent refuses one layer down.  The backend
    // checks the same rule and declines the borrow (not the statepoint) if the
    // three are not all present.
    CanonicalMode canonical = CanonicalMode::Off;
    const double* dev_phif  = nullptr; ///< canonical Flux,  [l*ng + ig]
    const double* dev_phis  = nullptr; ///< canonical Phis,  [ls*ng + ig]
    const double* dev_jnet  = nullptr; ///< canonical Jnet,  [ls*ng + ig]

    /// Generations for the two inputs that do NOT move every statepoint.  The
    /// upload is issued when the value differs from the one the device copy was
    /// built from; 0 means "unknown", which always uploads.
    unsigned long long chif_generation = 0;
    unsigned long long crdf_generation = 0;
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

    // --- WP6 receipt counters ----------------------------------------------

    /// Which loop drove the Picard iteration: "host_sync", "device_stream" or
    /// "device_graph".  Never null.
    [[nodiscard]] const char* loopArm() const;
    /// Every cudaStreamSynchronize this arm has issued.
    [[nodiscard]] unsigned long long hostSyncs() const;
    /// hostSyncs() / statepoints(), 0 when nothing ran.  The stage B number.
    [[nodiscard]] double hostSyncsPerStatepoint() const;
    /// Conditional-WHILE graph launches (0 on the other two arms), and the
    /// instantiations behind them.
    [[nodiscard]] unsigned long long graphLaunches() const;
    [[nodiscard]] unsigned long long graphBuilds() const;
    /// Why the graph arm was refused, "" when it was not asked for or not
    /// refused.  A refusal falls back to device_stream, never to the host.
    [[nodiscard]] const std::string& graphRefusal() const;

    /// Bytes this arm actually pushed H2D, and the bytes stage C did not have
    /// to push (borrowed canonical buffers plus generation-held uploads).
    [[nodiscard]] unsigned long long h2dBytes() const;
    [[nodiscard]] unsigned long long h2dBytesElided() const;
    [[nodiscard]] unsigned long long d2hBytes() const;

    /// Statepoints whose nodal inputs were borrowed rather than uploaded, and
    /// -- under CanonicalMode::Verify -- the number of elements at which a
    /// borrowed buffer disagreed with the host array it replaced.  A non-zero
    /// mismatch means the borrow is NOT sound for this deck and the arm says so
    /// rather than shipping the difference as physics.
    [[nodiscard]] unsigned long long canonicalStatepoints() const;
    [[nodiscard]] unsigned long long canonicalMismatch() const;

    /// Every cudaMalloc this instance has made, and how many of them were a
    /// RE-shape (the second and later ensureShape).  Stage E's observable:
    /// a per-slot backend must allocate once per slot, not once per statepoint.
    [[nodiscard]] unsigned long long allocations() const;
    [[nodiscard]] unsigned long long reallocations() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rasbery
