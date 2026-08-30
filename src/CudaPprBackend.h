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
// FAIL OPEN, AND SAY WHY.  Any CUDA failure or an unsupported deck (ng != 2)
// makes the entry point return false and the caller runs the untouched host
// reset+drive.  The receipt counts those as host_fallbacks AND names them:
// `ppr::Refusal` below is the ladder, mirroring BICGCMFD's EnqueueRefusal, and
// the [RASBERY][PPR_GPU] line carries `refusal` (the last one) plus `refusals`
// (the whole tally).  A fallback with no stated reason is the defect that let
// the master-mode refusal below run unnoticed for a whole campaign.
//
// ===========================================================================
// WP6 STAGE F -- RASBERY_PPR_MODE=master RUNS ON THE DEVICE
// ===========================================================================
//
// WHAT WAS WRONG.  Until this stage `PPR::resetAndDriveGpu` opened with
// `if (_mode_master) return false;`.  MASTER mode is what the production
// campaign runs (it is what gives Gate B pin RMS 0.238 % against the default
// mode's 0.522 %), so the arm declined on 35 statepoints out of 35 and the
// receipt said only `host_fallbacks:35` -- a number that looks the same whether
// the arm is refusing a deck or refusing the entire campaign.
//
// WHAT THE TWO SCHEMES ARE.  They are genuinely different reconstructions, not
// two spellings of one:
//
//   SENM   (default)  a semi-analytic expansion.  `drive` is a Picard iteration
//                     -- three source sweeps (updateFused: particular p,
//                     homogeneous a, projected c) then updateCorner -- with a
//                     global break on the four fuel-only corner-flux SUMS.
//   MASTER (this)     MASTER 4.0 MM Sec 6.1.  `driveMaster` writes a 13-term
//                     Legendre interpolant straight into `c`: nine even-parity
//                     terms from the surface fluxes, the reduced outward
//                     currents (2D/h) and the node average (Eq. 6.6), and four
//                     cross terms from corner fluxes that solve the
//                     corner-point-balance system (Eq. 6.7/6.8).  `p`, `a` and
//                     `bt` are never read; there is no source iteration.
//
// So the port is three new kernels plus a different loop body, not a flag.
//
// CLASS N1, AND EXACTLY WHERE.  The host's CPB solve is GAUSS-SEIDEL: it writes
// `phic` in place while later nodes read it, which is a serial dependence over
// every node of the mesh and cannot be a kernel.  The device runs the same
// balance as JACOBI (read the previous iterate, write the next) with the same
// 1e-5 relative-change break and the same iteration cap.  Both iterations are
// contractions on the SAME diagonally dominant system -- each corner's balance
// has diagonal 4w against off-diagonal 2w per node, so the Jacobi spectral
// radius is bounded by 1/2 and the Gauss-Seidel one by 1/4 -- and both converge
// to the SAME fixed point.  What differs is where each stops relative to it:
// ~1e-5 relative on the corner fluxes, which is four orders below the Gate B
// pin-power envelope.  Everything else in the master arm is the host's
// expression in the host's order on the host's operands: kMasterEven and
// kMasterCross are elementwise, the max-fold is over the same deterministic
// partition, and `max` does not depend on association.
//
// THE MASTER RECONSTRUCTION, BY CONTRAST, IS B0.  MASTER mode's pin expansion
// is `sum_t c[t] * leg[t]` -- a 15-term dot product with no `exp` at all -- so
// PprReconstructionKernel.cuh's master branch is the host's loop, in the host's
// order, with no transcendental and with --fmad=false on the TU.  The only
// reason a master pin map differs from the host's is the N1 corner fluxes it
// was built from.
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

/// WHY A STATEPOINT RAN ON THE HOST.  The BICGCMFD::EnqueueRefusal ladder, for
/// the PPR seam, and for the same reason that one exists: `host_fallbacks:35`
/// is a symptom, and a seam that only counts cannot tell "this build has no
/// CUDA" from "this deck is not two-group" from "the production reconstruction
/// mode was never ported".  The campaign spent a release on the third of those
/// while the receipt printed the first two's number.
///
/// EXHAUSTIVE BY CONSTRUCTION.  Every `return false` on the way into the device
/// -- in PPR::resetAndDriveGpu and in PprBackend::resetAndDrive -- sets one of
/// these before it returns, and tools/test_ppr_gpu_master_mode_contract.py
/// holds that against the source.  `None` is the value of a run that never
/// fell back.
enum class Refusal : int {
    None = 0,        ///< no fallback has happened
    ArmOff,          ///< RASBERY_GPU_PPR unset, no CUDA device, or a stub build
    BackendDisabled, ///< an earlier CUDA failure released this instance
    NotTwoGroup,     ///< ng != 2; the source body serves two groups
    NonPositiveIter, ///< the caller passed niter <= 0
    ShapeAllocFail,  ///< ensureShape could not stand the buffers up
    CudaFailure,     ///< any cudaError_t on the statepoint's own path
    /// WP19.  A graph build refused with one of CUDA's capture-concurrency
    /// codes and was rebuilt once with the capture arbiter held.  It is the
    /// one rung that is NOT necessarily a fallback: the retry usually works,
    /// and then only `refusals[CaptureRaceRetry]` moves while `refusal` stays
    /// `none`.  A run whose count is non-zero saw the race and survived it; a
    /// run whose count is non-zero AND whose refusal is `capture_race_retry`
    /// lost the retry too and fell back to the stream arm.
    CaptureRaceRetry,
    Count
};

inline const char* refusalName(Refusal r) {
    switch (r) {
        case Refusal::None:            return "none";
        case Refusal::ArmOff:          return "arm_off";
        case Refusal::BackendDisabled: return "backend_disabled";
        case Refusal::NotTwoGroup:     return "not_two_group";
        case Refusal::NonPositiveIter: return "non_positive_iter";
        case Refusal::ShapeAllocFail:  return "shape_alloc_fail";
        case Refusal::CudaFailure:     return "cuda_failure";
        case Refusal::CaptureRaceRetry: return "capture_race_retry";
        case Refusal::Count:           break;
    }
    return "?";
}

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

    /// WP6 stage D.  THE SEVEN COEFFICIENT ARRAYS HAVE ONE CONSUMER, and when
    /// that consumer is about to run on the device they have no reason to come
    /// back at all -- 9,059,472 B/statepoint at KNGR size, which is larger than
    /// everything this arm uploads put together.
    ///
    /// THE CALLER OWNS THE REPAIR.  Setting this is a promise that the host
    /// arrays will not be read before something rewrites them.  PPR keeps that
    /// promise by re-running reset() + drive() on the host if the device
    /// reconstruction it planned does not happen, and the receipt counts those
    /// as `recon_repairs`.  A caller that sets this and then reads _p is not
    /// wrong by a little.
    bool coefficients_stay_on_device = false;

    /// Generations for the two inputs that do NOT move every statepoint.  The
    /// upload is issued when the value differs from the one the device copy was
    /// built from; 0 means "unknown", which always uploads.
    unsigned long long chif_generation = 0;
    unsigned long long crdf_generation = 0;

    /// WP6 stage F.  RASBERY_PPR_MODE=master -- run MASTER MM Sec 6.1's
    /// interpolant + corner-point-balance solve instead of the SENM Picard
    /// iteration.  It selects a DIFFERENT loop body, not a flag inside one, and
    /// the `reset()` half (buckling, corner init, SENM fit, axial leakage,
    /// source expansion) is launched identically in both -- because the host
    /// runs reset() identically in both, and driveMaster consumes the corner
    /// flux reset() seeded.
    bool mode_master = false;
};

// ---------------------------------------------------------------------------
// Stage D: reconstructPinPower on the device
// ---------------------------------------------------------------------------

/// RASBERY_GPU_PPR_RECON, read once.  DEFAULT OFF.
inline bool reconEnabledFromEnv() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_PPR_RECON");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
                 s == "FALSE");
    }();
    return on;
}

/// The reconstruction's geometry and its quadrature table, flattened.
///
/// UPLOADED ONCE.  Every field is a pure function of (ndivxy, npins) and the
/// loading map -- `buildPinQuadratureTable` reads nothing else (PprQuadrature.h),
/// and `latol` / `vol` / `hz` are fixed after Geometry stand-up.  A deck whose
/// shape changed would re-shape the backend, which is counted.
struct ReconGeomView {
    int nxya = 0;
    int nz   = 0;
    int kbc  = 0;
    int kec  = 0;
    int ndiv = 0;   ///< Geometry::ndivxy
    int npins = 0;

    const int*    latol = nullptr; ///< [nxya * ndiv*ndiv], index la*ndiv2 + li
    const double* vol   = nullptr; ///< [nxyz]
    const double* hz    = nullptr; ///< [nz]

    int           n_overlaps = 0;
    const int*    pin_off    = nullptr; ///< [npins*npins + 1]
    const int*    ovl_di     = nullptr; ///< [n_overlaps]
    const int*    ovl_dj     = nullptr;
    const double* ovl_dxh    = nullptr;
    const double* ovl_dyh    = nullptr;
    const double* q_xq       = nullptr; ///< [n_overlaps * 9]
    const double* q_yq       = nullptr;
    const double* q_wt       = nullptr;
    const double* q_leg      = nullptr; ///< [n_overlaps * 9 * 15]
};

/// What changes at a statepoint, plus the host destinations.
///
/// THE FORM FUNCTIONS ARE A FIXED REGISTRY.  `gmap`/`fmap` hold EVERY reference
/// depletion point of every model, uploaded once, and what moves per statepoint
/// is the per-(plane, assembly) bracketing triple.  `plane_lo < 0` means the
/// host's "no fuel node in this (k, la)", i.e. the plane it skips entirely.
struct ReconStepView {
    const double* xskf = nullptr; ///< [ng * nxyz]

    int           n_form_slots = 0;
    const double* gmap         = nullptr; ///< [n_form_slots * npina]
    const double* fmap         = nullptr; ///< [n_form_slots * ng * npina], may be null

    const int*    plane_lo    = nullptr; ///< [nz * nxya]
    const int*    plane_hi    = nullptr;
    const double* plane_alpha = nullptr;

    /// WP6 stage F.  MASTER mode's pin expansion is a 15-term dot product of
    /// `c` with the pre-computed Legendre products -- no `p`, no `a`, no `bt`,
    /// no `exp`.  It is the same branch PPR.cpp takes inside the overlap loop.
    bool mode_master = false;

    bool reconstruct_flux = false;
    /// Will the HOST read the pin map after this call?  False (a statepoint
    /// whose `pin_info` is off) means the map stays on the device and only the
    /// two peaking factors come back -- which is the point of the port, because
    /// the map is 5.4 MB and Fq/FdH are 16 B.
    bool materialize_pin = true;

    double* pin_power = nullptr; ///< [nxya * nz * npina], host
    double* pin_flux  = nullptr; ///< [nxya * nz * ng * npina], host, may be null
    double* frp       = nullptr; ///< Geometry::frp()
    double* fqp       = nullptr; ///< Geometry::fqp()
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

    /// PPR::reconstructPinPower on the device (WP6 stage D, RASBERY_GPU_PPR_RECON).
    ///
    /// Only legal immediately after a resetAndDrive() that returned true: it
    /// reads the coefficient arrays THAT call left on the device.  Returns false
    /// -- having written nothing -- when the arm is off, the previous drive did
    /// not run on the device, the shapes do not line up, or any CUDA call
    /// fails; the caller then runs the untouched host loop.
    bool reconstructPinPower(const ppr::ReconGeomView& geom, const ppr::ReconStepView& step);

    /// The caller could not honour `coefficients_stay_on_device` -- it is
    /// about to rebuild the host coefficients itself.  Counts the repair and
    /// clears the promise.
    void noteReconRepair();

    /// THE CALLER DECLINED BEFORE THE BACKEND WAS ASKED.  PPR::resetAndDriveGpu
    /// can refuse on its own (the arm is off, the deck is not two-group) and
    /// those statepoints never reach resetAndDrive(); recording the reason here
    /// is what keeps the ladder in the receipt exhaustive rather than "the
    /// reasons the backend happened to see".  Safe on a disabled or stub
    /// backend: it moves counters and nothing else.
    void noteHostFallback(ppr::Refusal reason);

    /// The reason the LAST fallback gave, and the whole tally.  `refusalJson`
    /// is a `{"name":count,...}` object with an entry for every reason that has
    /// fired at least once, i.e. `{}` on a run that never fell back.
    [[nodiscard]] ppr::Refusal       lastRefusal() const;
    [[nodiscard]] const char*        lastRefusalName() const;
    [[nodiscard]] unsigned long long refusalCount(ppr::Refusal reason) const;
    [[nodiscard]] std::string        refusalJson() const;

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

    /// Statepoints where the coefficient D2H was elided and the device
    /// reconstruction then did NOT run, so the host had to recompute the
    /// coefficients before its own loop could read them.  Non-zero is not a
    /// wrong answer -- the repair is exact -- but it is a wasted drive, and a
    /// run that reports many of them elided a transfer it should not have.
    [[nodiscard]] unsigned long long reconRepairs() const;

    /// Statepoints whose reconstruction ran on the device, and how many of
    /// those had to send the pin map back.  `recon_statepoints -
    /// pin_materializations` is the number of times 5.4 MB stayed where it was.
    [[nodiscard]] unsigned long long reconStatepoints() const;
    [[nodiscard]] unsigned long long pinMaterializations() const;
    /// Why the reconstruction arm declined, "" when it did not.
    [[nodiscard]] const std::string& reconRefusal() const;

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
