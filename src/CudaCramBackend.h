#pragma once

// Device backend for the CRAM Bateman depletion -- GA evaluator plan Sec 6.3
// Task 16 (Rev.7.1 amendment 1), behind RASBERY_GPU_CRAM (default off).
//
// WHAT IT COVERS, AND WHY EXACTLY THAT.  Task 10 measured the per-statepoint
// host floor and moved the largest half of it (PPR reset+drive, 158 ms/sp of
// 343 ms/sp) onto the device.  The `floor_wall` receipt's other two buckets --
// `depl_predictor` and `depl_corrector` -- are what is left, and they are one
// piece of arithmetic repeated 8,451 times:
//
//     XSSet::Deplete        -> DepleteNode  -> condense 11x39 micro XS,
//                              BuildTransitionMatrix, solveBatemanCRAM,
//                              ApplyXeEquilibrium                       (all nxyz)
//     XSSet::CorrectorStep  -> the same body on BOS/EOS-averaged rates,
//                              plus the Eq. (6.20) density average and the
//                              burnup-key accumulation                 (all nxyz)
//
// So this backend ports those two node loops and NOTHING else.  In particular
// it does NOT port:
//
//   * DepleteRodMaterials -- it is a fluence accumulation over the fine rod
//     mesh, not a CRAM solve, and it never enters solveBatemanCRAM.
//   * XSSet::DecayIsotopeDensityFlat -- a restart-cooling CRAM chain that runs
//     ONCE per run, not once per statepoint.
//   * RASBERY_PC_SUBSTEPS > 1 -- the Isotalo substep chain is k CRAM solves per
//     node with interpolated rates.  It is a straight extension of the k == 1
//     body, but it is off by default, it has never been in a campaign arm, and
//     an untested second device path is a worse trade than a host fallback.
//     The backend DECLINES (returns false) and the host loop runs.
//
// WHAT THE HOST SOLVE ACTUALLY IS.  Not a dense LU.  milk.h's
// `solveBatemanCRAM` at order 8 is a SPARSE GAUSS-SEIDEL over 4 CRAM poles:
//
//     alpha0     = 1.1722341374385704e-08
//     poles      = 4          (order 8, matrix_sgn = -1)
//     max_iter   = 64
//     rel_tol    = 1.0e-13    abs_tol = 1.0e-28    diag_tol = 1.0e-30
//
// per pole: seed x = rhs/diag, then up to 64 sweeps of
// `x[row] = (rhs[row] - sum_offdiag vals*x[col]) / pole_diag[row]` followed by
// an explicit residual sweep and a max-norm break test.  The system is
// 39 x 39 with `first = iI135 = 3` (H/B/O are copied through), so 36 unknowns,
// and the sparsity is FIXED BY THE ISOTOPE CHAIN, not by the node: over rows and
// columns >= 3 the union of nonzeros(depDecay), the diagonal, nonzeros(depTrans)
// and the two (n,2n) specials is 160 entries -- 124 off-diagonal, 36 diagonal,
// at most 19 in any one row.  That is the whole reason this is a GPU problem at
// all: 8,451 independent 36-unknown sparse solves.
//
// THE SWEEP IS SERIAL AND STAYS SERIAL; THE NODE IS NOT, AND NO LONGER IS.
// Gauss-Seidel is sequential in `row` (row i reads the x[j] this sweep already
// updated for j < i) and the inner `sum -= vals[i] * x[cols[i]]` is sequential
// in i.  A kernel that spread EITHER of those across lanes would reassociate,
// which turns a reproduction into a different method, so neither is spread.
//
// But the POLES are not one of those two.  Order 8 is four independent complex
// linear solves against the same real matrix; they share `mdiag` and the
// compressed off-diagonals and nothing else, and the only thing that couples
// them is milk.h's closing `accum[row] += x_pole[row]`, a fixed left-to-right
// walk over pole = 0, 1, 2, 3.  WP21-D therefore maps ONE NODE TO FOUR LANES,
// lane == pole, and re-forms that walk with __shfl_sync IN POLE ORDER: the same
// four addends into the same zero-initialised accumulator in the same order, so
// the sums are bit-identical additions and not merely equivalent ones.  The
// per-node state (~4.7 KB: cond4, the 160 matrix values, x, accum, iden) still
// lives in per-thread LOCAL memory -- shared memory for the pole-invariant half
// would cap a 64-thread block at eight nodes and cost more occupancy than the
// four-fold rebuild costs instructions -- and the win is that the 4-pole serial
// latency chain that made kPredictor/kCorrector 4.4 ms per launch at 4.15 %
// warps active now runs four-wide, 33,804 lanes instead of 8,451 threads.
//
// WHICH MAKES THE POLE-PARALLEL ARM B0 AGAINST THE SERIAL ONE.  That is a
// DIFFERENT claim from the N1 below, and both are true at once: the device is
// N1 against the HOST because of complex division, and the pole-parallel kernel
// is B0 against the SERIAL KERNEL because it performs the same operations in
// the same order.  RASBERY_GPU_CRAM_PARALLEL=serial keeps the pre-WP21-D body
// in the same binary so that equality is a measurement anyone can repeat.
// There is no `jacobi` arm: it would only be needed if lane-per-pole could not
// preserve the pole-sum order, and it can, so nothing N1 is offered here.
//
// THE MINED PATTERN IS A SUPERSET, AND THE ZERO TEST IS STILL RUN.  The host
// compresses each row by scanning `A(row, col)` and skipping `value == 0.0`.
// The uploaded pattern is the node-independent union, so a node whose value
// happens to be exactly zero would carry an entry the host would not have.  The
// kernel therefore RE-RUNS the host's `value == 0.0` test per node per entry
// rather than trusting the pattern -- the compressed row is then the host's
// row, entry for entry, in the host's ascending-column order, and the
// arithmetic is the same arithmetic and not merely an equivalent one.
//
// CLASS N1, NOT B0, AND THE REASON IS COMPLEX DIVISION.  Every real operation
// here is transcribed in the host's statement order into a --fmad=false TU, and
// the complex multiply is written out as the naive (ac-bd, ad+bc) that gcc
// emits inline for finite operands.  The complex DIVISION is the one place the
// device cannot simply copy the host: libstdc++ lowers `complex<double> /
// complex<double>` to libgcc's `__divdc3`, which rescales by `logb`/`scalbn`
// before dividing.  The kernel transcribes __divdc3's algorithm, but `logb`,
// `scalbn` and the underlying divides are the device libm's, so equality is a
// measurement and not a guarantee.  The gate is therefore Gate A -- nuclide
// number densities max relative difference against the host, plus keff/ppm/AO
// at the NEXT statepoint -- not h5diff.  Run to run the arm is deterministic:
// one thread per output, no atomics, no reduction, a fixed pattern.
//
// RASBERY_GPU_CRAM IS AN ARM KNOB, AND THAT IS THE DIFFERENCE FROM TASK 10.
// PPR is strictly downstream of the statepoint: it reads the converged flux and
// writes only pin powers, so RASBERY_GPU_PPR is deliberately ABSENT from
// trajectory::kArmEnv.  Depletion is the opposite.  Its output IS the next
// statepoint's isotope inventory; the next XS reconstruction, the next boron
// search and every eigenvalue after it are downstream of these densities.  A
// receipt that held RASBERY_GPU_CRAM outside the arm list would let two runs
// with different trajectories compare as the same arm.  So the knob is IN
// kArmEnv, and tools/test_cram_gpu_contract.py asserts its presence -- the
// mirror image of the assertion that keeps RASBERY_GPU_PPR out.
//
// PER SLOT, NOT PER PROCESS.  One backend instance is owned by one XSSet, which
// is owned by one Driver, and --batch-mode gives every deck its own Driver.
// Nothing in the .cu is `static` except the CRAM constants: the slot-0 bug class
// (deck 7 driving deck 0's buffers) is refused by construction.
//
// FAIL OPEN, AND THE FAILURE SET IS BIGGER THAN CUDA.  The entry points return
// false -- having written NOTHING to any host array -- for a CUDA error, an
// unsupported deck (ng != 2, CRAM order != 8, no depletion data), an unsupported
// mode (RASBERY_PC_SUBSTEPS > 1), a corrector whose BOS snapshot did not come
// from the matching predictor, and -- the one that matters physically -- ANY
// node whose solve hit the host's two throw conditions (zero Gauss-Seidel
// diagonal, or 64 sweeps without convergence) or produced a non-finite density.
// The kernel writes its results to a device staging block and a per-node status
// byte; the status is reduced first, and the D2H happens only if the reduction
// is clean.  The host loop then runs from an untouched inventory and throws the
// same std::runtime_error at the same node it always would.  See the NaN-mask
// note in docs/TASK16_CRAM_GPU_20260831_KO.md.

#include <cstddef>
#include <memory>
#include <string>

namespace rasbery {

namespace cram {

/// The CRAM order this backend reproduces.  XSSet.cpp static_asserts its own
/// CRAM_ORDER against this, so a change to the host solver cannot leave the
/// device quietly running the old pole set.
inline constexpr int kOrder = 8;

/// Pole count / iteration bound of that order, restated so the contract test can
/// check them against milk.h's literals from outside the build.
inline constexpr int kPoles   = 4;
inline constexpr int kMaxIter = 64;

/// Node-invariant tables.  Uploaded once per deck (they are library and
/// geometry properties), re-uploaded when `generation` changes.
struct LibView {
    unsigned long long generation = 0;

    int niso     = 0; ///< 39
    int nxs      = 0; ///< 11 (N_XS_SCALAR)
    int first    = 0; ///< iI135; rows below this are copied through
    int ac_first = 0;
    int ac_last  = 0;
    int i135     = 0;
    int xe135    = 0;
    int xe135m   = 0;
    int u234     = 0;
    int u235     = 0;
    int u238     = 0;
    int np237    = 0;

    const double* dep_decay = nullptr; ///< [niso * niso], row-major (milk::Matrix)
    const double* dep_trans = nullptr; ///< [niso * niso], row-major

    /// Per-node burnup-key normalisation, precomputed HOST-SIDE with exactly the
    /// host's expression `8.64e7 * (vol(l) / lib_model_volu[comp[l]]) *
    /// lib_model_hmas[comp[l]]`, so the device reads one double instead of
    /// re-deriving it from three tables in a different order.
    const double* dfac = nullptr; ///< [nxyz]
    const double* vol  = nullptr; ///< [nxyz]
};

/// XSSet::Deplete over every node.  `iden` is read AND written in place; the
/// backend writes rows [first, niso) back exactly as DepleteNode does.
struct PredictorView {
    int    ng          = 0;
    int    nxyz        = 0;
    double dt          = 0.0;
    double norm_factor = 0.0;
    int    xe_transient = 0; ///< 1 = skip the Xe equilibrium overwrite

    /// Micro XS residency key.  Matches XSSet::_micx_generation; when it equals
    /// the resident copy's the 11-block upload is skipped.
    unsigned long long micx_generation = 0;

    const double* phif   = nullptr; ///< [nxyz * ng], node-major (l * ng + ig)
    const double* mic[11] = {};     ///< [(iso * ng + ig) * nxyz + l], N_XS_SCALAR order
    double*       iden    = nullptr; ///< [niso * nxyz], in and out

    /// WP15.1: the SAME eleven blocks as `mic`, as DEVICE addresses inside the
    /// flat-XS backend's resident block, plus the event that orders this
    /// backend's stream behind the solve that wrote them.
    ///
    /// WHY THIS EXISTS.  The `mic` pointers above are host arrays that the
    /// flat-XS device solve downloaded from exactly this block; uploading them
    /// again is a 21 MB round trip to fetch bytes the device already has.  When
    /// `mic_device[0]` is non-null AND `mic_device_ready` is non-null the
    /// backend takes the four slots it reads D2D instead.
    ///
    /// ALL OR NOTHING, AND THE CALLER OWNS THE GENERATION CHECK: XSSet fills
    /// these only when the resident block's generation equals `micx_generation`
    /// below.  A null pair is not an error, it is "use the host copy" -- which
    /// is what a stub build, a declined solve, or a host-rebuilt _micx gives.
    const void*   mic_device[11]  = {};
    void*         mic_device_ready = nullptr; ///< cudaEvent_t, or null
    /// WP20.1: element width of what `mic_device` points at -- 8 on the FP64
    /// arm, 4 under RASBERY_GPU_FP32.  CRAM's own state stays FP64 (the
    /// partial-fraction sum cancels catastrophically), so a narrow source is
    /// WIDENED on the device rather than memcpy'd.
    int           mic_device_elem_bytes = static_cast<int>(sizeof(double));
};

/// XSSet::CorrectorStep over every node, pcSubsteps == 1.
struct CorrectorView {
    int    ng           = 0;
    int    nxyz         = 0;
    double dt           = 0.0;
    double bos_norm     = 0.0;
    double eos_norm     = 0.0;
    int    xe_transient = 0;
    int    density_average    = 0; ///< RASBERY_PC_MODE=decart
    int    xe_equilibrium_fix = 0; ///< RASBERY_PC_XE_EQUILIBRIUM_FIX

    unsigned long long micx_generation = 0;

    /// The predictor call that produced the BOS snapshot this corrector must
    /// read.  A corrector whose token does not match the resident one declines:
    /// the device BOS block would be some other statepoint's.
    unsigned long long bos_token = 0;

    const double* flux_bos = nullptr; ///< [nxyz * ng]
    const double* flux_eos = nullptr; ///< [nxyz * ng] (Geometry::Phif)
    const double* xskf_bos = nullptr; ///< [ng * nxyz]
    const double* xskf_eos = nullptr; ///< [ng * nxyz]
    const double* mic[11]  = {};      ///< EOS micro XS; BOS comes from the device snapshot
    /// WP15.1: see PredictorView::mic_device -- same contract, same all-or-
    /// nothing rule, same caller-owned generation check.
    const void*   mic_device[11]  = {};
    void*         mic_device_ready = nullptr; ///< cudaEvent_t, or null
    /// WP20.1: see PredictorView::mic_device_elem_bytes.
    int           mic_device_elem_bytes = static_cast<int>(sizeof(double));
    const double* iden_bos = nullptr; ///< [niso * nxyz]
    double*       iden     = nullptr; ///< [niso * nxyz]: predictor inventory in, corrected out
    const int*    burn_bos = nullptr; ///< [nxyz]
    int*          burn     = nullptr; ///< [nxyz], out
};

} // namespace cram

/// One instance per XSSet, i.e. per Driver, i.e. per batch slot.
class CramBackend {
public:
    CramBackend();
    ~CramBackend();

    CramBackend(const CramBackend&)            = delete;
    CramBackend& operator=(const CramBackend&) = delete;

    /// RASBERY_GPU_CRAM is set to a truthy value AND a device exists.
    [[nodiscard]] bool available() const;

    /// Human-readable reason, for the one-line receipt.
    [[nodiscard]] const std::string& status() const;

    /// XSSet::Deplete on the device.  Returns false with every host array
    /// untouched if anything at all goes wrong, so the caller runs the host
    /// loop.  On success `*token_out` receives the BOS token the matching
    /// corrector must present.
    bool predictor(const cram::LibView& lib, const cram::PredictorView& view,
                   unsigned long long* token_out);

    /// XSSet::CorrectorStep on the device.  Same fail-open contract.
    bool corrector(const cram::LibView& lib, const cram::CorrectorView& view);

    // --- receipt counters --------------------------------------------------

    /// Device calls that succeeded, counted separately so the receipt can show
    /// a predictor/corrector imbalance (which is what a mid-statepoint decline
    /// looks like).
    [[nodiscard]] unsigned long long predictorCalls() const;
    [[nodiscard]] unsigned long long correctorCalls() const;
    /// predictorCalls() + correctorCalls(), the receipt's `statepoints` field
    /// counts the pairs.
    [[nodiscard]] unsigned long long nodesSolved() const;
    /// Sum over (node, pole) of Gauss-Seidel sweeps actually executed, and the
    /// count of (node, pole) pairs, so the receipt can print a mean that is
    /// directly comparable to a host instrumentation of the same loop.
    [[nodiscard]] unsigned long long gsIterations() const;
    [[nodiscard]] unsigned long long gsSolves() const;
    [[nodiscard]] double             wallMs() const;
    [[nodiscard]] int                deviceOrdinal() const;
    /// Bytes of micro-XS H2D actually paid (the residency check's own receipt).
    [[nodiscard]] unsigned long long micxH2dBytes() const;

    /// WP15.1: micro-XS bytes taken DEVICE-TO-DEVICE out of the flat-XS backend's
    /// resident block instead of re-uploaded from the host.  The pair
    /// (micxH2dBytes, micxD2dBytes) is the whole receipt: with the arm on and
    /// the generations agreeing the first should be ~0 and the second should
    /// carry what it used to.
    [[nodiscard]] unsigned long long micxD2dBytes() const;
    /// Statepoints whose corrector reused the device BOS snapshot instead of a
    /// second 11-block upload.
    [[nodiscard]] unsigned long long bosReuses() const;

    // --- WP21-D: which node mapping ran, and what it cost ------------------

    /// "pole4" (default: kLanesPerNode lanes per node, lane == pole) or
    /// "serial" (the pre-WP21-D body).  RASBERY_GPU_CRAM_PARALLEL selects it.
    /// The receipt has to carry this: two runs of the same binary with the same
    /// arm knobs but different mappings are the same TRAJECTORY (the default arm
    /// is B0 against serial) and a different KERNEL, and a per-launch time
    /// quoted without it is unattributable.
    [[nodiscard]] const std::string& kernelVariant() const;

    /// Lanes co-operating on one node: 4 on the pole-parallel arm, 1 on serial.
    [[nodiscard]] int lanesPerNode() const;

    /// kPredictor/kCorrector launches, counted on the decline path too -- a
    /// launch that produced a bad node still ran and still cost its time.
    [[nodiscard]] unsigned long long launches() const;

    /// Mean cudaEvent-measured microseconds per KERNEL launch, or -1.0 when
    /// RASBERY_GPU_CRAM_TIMING is unset.  `wallMs()` is the whole call --
    /// transfers, status drain and all -- which is not the number ncu prints
    /// for block 39; this one is, and -1 is "never measured" rather than 0.
    [[nodiscard]] double launchUsMean() const;
    /// WP20.2.  "fp32" when the four-pole partial-fraction sum accumulated
    /// in float with a Neumaier compensation, "fp64" otherwise.  A WORD and
    /// not a bool because that is what the receipt prints, and because a
    /// receipt that said `true` would leave a reader to guess which of this
    /// backend's many doubles it was true ABOUT.  It is exactly one of them:
    /// the accumulator `accr`.  Everything else -- the Gauss-Seidel solve,
    /// its 1.0e-13 break test, the matrix split and the alpha_0 term -- stays
    /// FP64 with a reason each, stated above cramFoldPole in the .cu.
    [[nodiscard]] const char* poleSumPrecision() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rasbery
