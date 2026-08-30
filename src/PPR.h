#pragma once

#include "CudaPprBackend.h"
#include "Geometry.h"
#include "PprQuadrature.h"
#include "XSSet.h"
#include "pch.h"
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// PPR coefficient macros
// All underlying arrays are owned by Geometry; PPR holds raw pointers to them.

/// @brief Particular coefficients of flux
#define p(i, j) _p[(lk * 15 * _ng) + (g * 15) + (5 * i - (i * (i - 1)) / 2 + j)]

/// @brief Homogeneous coefficients of flux
#define a(i) _a[(lk * 8 * _ng) + (g * 8) + (i - 1)]

/// @brief Polynomial fitting coefficient of flux
#define c(i, j) _c[(lk * 15 * _ng) + (g * 15) + (5 * i - (i * (i - 1)) / 2 + j)]

/// @brief Expansion coefficients of source term
#define q(lk, g, i, j) _q[(lk * 15 * _ng) + (g * 15) + (5 * i - (i * (i - 1)) / 2 + j)]

/// @brief Expansion coefficients of axial leakage
#define l(i, j) _l[(lk * 9 * _ng) + (g * 9) + i * 3 + j]

/// @brief Buckling value (B_t)
#define Bt _bt[lk * _ng + g]

/// @brief Node-averaged flux from nodal calculation
#define aflux (_phif[lk * _ng + g])

/// @brief Corner flux
#define phic(dir) _phic[(lk * 4 * _ng) + (g * 4) + dir]

/// @brief Surface flux from nodal calculation
#define phis(dir, lr) (_phis[_g.lktosfc(lr, dir, lk) * _ng + g])

/// @brief Surface averaged current from nodal calculation
#define jnet(dir, lr) (_jnet[_g.lktosfc(lr, dir, lk) * _ng + g])

/// @brief Axial leakage from nodal calculation
#define zleak(lk) (_jnet[_g.lktosfc(RIGHT, ZDIR, lk) * _ng + g] - _jnet[_g.lktosfc(LEFT, ZDIR, lk) * _ng + g]) / _g.hmesh(ZDIR, lk)

namespace rasbery {

// QuadPoint / PinOverlap / PinQuadInfo moved to PprQuadrature.h (WP8 stage 2)
// so CohortContext.h can hold the table without dragging Geometry, XSSet and
// the CUDA backend header in behind it.

/// WP6 stage C.  The device-resident nodal arrays the CALLER is offering PPR in
/// place of the host copies it would otherwise upload.
///
/// BORROWED AND ALL-OR-NOTHING.  These are GpuPhysicsArena addresses reached
/// through CudaOuterSegment::canonicalNodalSet(), which already answers "not
/// yet" with an empty set rather than with a partial one; PPR checks the rule a
/// second time because the cost of getting it wrong is a reconstruction that
/// blends two different outer iterations and looks entirely plausible.
///
/// The MODE is the caller's, not this class's: `Verify` uploads the host copies
/// as well and has the device compare them elementwise, which is how 238
/// establishes that the borrow is sound for a deck instead of the header
/// asserting that it is.
struct PprCanonicalInputs {
    ppr::CanonicalMode mode = ppr::CanonicalMode::Off;
    const double*      phif = nullptr;
    const double*      phis = nullptr;
    const double*      jnet = nullptr;

    [[nodiscard]] bool complete() const {
        return phif != nullptr && phis != nullptr && jnet != nullptr;
    }
};

class PPR {
private:
    Geometry& _g;
    XSSet&    _xs;

    int _ng;   // cached energy groups
    int _nxyz; // cached total 3D nodes

    double _hmesh; // scratch: current node mesh size
    double _reigv; // reciprocal of eigenvalue (1/k_eff)

    // Pointers to Geometry-owned nodal arrays
    const double* _phif; // -> g.Phif(), READ-ONLY (Geometry.h)
    double* _phis; // -> g.Phis()
    double* _jnet; // -> g.Jnet()

    // Pointers to Geometry-owned PPR arrays
    double* _phic; // -> g.Phic()   corner flux
    double* _p;    // -> g.PprP()   particular coefficients
    double* _a;    // -> g.PprA()   homogeneous coefficients
    double* _c;    // -> g.PprC()   polynomial fitting coefficients
    double* _q;    // -> g.PprQ()   source expansion coefficients
    double* _l;    // -> g.PprL()   axial leakage expansion
    double* _bt;   // -> g.PprBt()  transverse buckling

    // The pre-computed quadrature table, indexed by pin.
    //
    // WP8 stage 2: BORROWED, not owned.  It is a pure function of
    // (ndivxy, npins) -- see PprQuadrature.h -- so it belongs to the cohort and
    // not to this PPR object.  It used to be a per-object `vector`, which meant
    // a 64-case wave built 64 bit-identical copies of it, one per Driver.  The
    // null pointer is the "not built yet" flag the old `bool` was.
    std::shared_ptr<const PinQuadTable> _pin_quad_table;

    // Corner-DF consistency ratio sdfa/pdfa per (node, group), refreshed in
    // reset().  The corner-balance phic is a heterogeneous corner-flux estimate;
    // each node's expansion works in its own surface-DF-folded (SET) space and
    // must consume rc * phic instead (MASTER: CRADF corner factors).  OFF by
    // default: the MASTER runs benchmarked against carry no CRADF.INP, so the
    // correction moves the comparison away from them.  RASBERY_PPR_CRDF=1
    // enables it for physical-accuracy studies against transport references.
    std::vector<double> _crdf;
    bool                _crdf_on = false;

    inline double crdf(int lk, int g) const { return _crdf[static_cast<size_t>(lk) * _ng + g]; }

    // Reconstruction mode.  SENM (historical) iterates a semi-analytic expansion
    // with source sweeps; MASTER mode is the MASTER 4.0 MM section 6.1 scheme:
    // a 13-term Legendre interpolant of the nodal solution whose cross terms
    // come from a corner-point-balance (CPB) linear system (Eq. 6.7/6.8), no
    // source iteration.  RASBERY_PPR_MODE=master selects it.
    bool _mode_master = false;

    // GA evaluator plan Task 10: the device arm for reset()+drive().  ONE per
    // PPR object, and a PPR object is a local of Driver::Drive(), so this is
    // per-Driver -- per batch slot -- with no process-wide state to mix two
    // decks' buffers.  Null is impossible; unavailable is the normal state
    // (RASBERY_GPU_PPR unset), and then resetAndDriveGpu() returns false and
    // the host path runs untouched.
    std::unique_ptr<PprBackend> _gpu;

    /// Staging for the one macroscopic block XSSet does not publish as a raw
    /// SoA pointer.  Sized once, refilled per statepoint (ng*nxyz doubles).
    std::vector<double> _xsdf_stage;
    /// Fuel flags as bytes -- Geometry keeps `bool*`, whose object
    /// representation is not something a device upload may assume.
    std::vector<unsigned char> _isfuel_stage;

    /// Corner-balance iterations spent by the host drive() over this run.
    unsigned long long _host_iters = 0;

    /// WP6 stage C.  What the caller most recently offered, and how much of it
    /// PPR is allowed to use.  Re-offered per statepoint (the arena addresses
    /// are fixed, but whether the binding is LIVE is not), so a stale offer
    /// cannot outlive the segment that made it.
    PprCanonicalInputs _canonical;

    /// The generation of `_crdf`.  It moves only when the corner-DF correction
    /// is enabled AND recomputed; with RASBERY_PPR_CRDF off the array is all
    /// ones for the life of the run, which is exactly the case the device arm
    /// should upload once and never again.
    unsigned long long _crdf_generation = 1;

    /// Did the LAST resetAndDriveGpu() actually drive on the device?  The
    /// device reconstruction reads the coefficients that call left there, so a
    /// statepoint that fell back to the host must not reconstruct from them.
    bool _gpu_drove = false;

    /// WP6 stage D.  The last drive left the seven coefficient arrays on the
    /// device only, because a device reconstruction was planned.  If that
    /// reconstruction does not happen, the host loop cannot read _p/_a/_c/_bt
    /// -- they are one statepoint stale -- so reconstructPinPower rebuilds them
    /// first.  The flag is the whole difference between an elided transfer and
    /// a wrong pin map.
    bool _coeffs_device_only = false;
    /// The niter the last drive was given, so the repair can repeat it.
    int  _last_niter = 0;

    // --- WP6 stage D: what the device reconstruction needs staged ----------
    //
    // BUILT ONCE PER PPR OBJECT, and that is the whole reason the port is
    // affordable.  The quadrature table is a pure function of (ndivxy, npins)
    // and the form functions are LIBRARY data: every reference depletion point
    // of every model goes up once, and what travels per statepoint is the
    // per-(plane, assembly) bracketing triple -- 24 B * nz * nxya -- instead of
    // the interpolated maps, which would be nz * nxya * (1 + ng) * npina
    // doubles and cost more than the loop they replace.
    struct ReconStaging {
        bool built    = false;
        bool declined = false; ///< a shape or a library this arm cannot serve
        /// Why, for the receipt.  Set once, on the decision.
        std::string reason;

        // the flattened quadrature table
        std::vector<int>    pin_off; ///< [npina + 1]
        std::vector<int>    ovl_di, ovl_dj;
        std::vector<double> ovl_dxh, ovl_dyh;
        std::vector<double> q_xq, q_yq, q_wt; ///< [n_overlaps * 9]
        std::vector<double> q_leg;            ///< [n_overlaps * 9 * 15]
        int                 n_overlaps = 0;

        // the form-function registry
        int                 slots = 0;
        std::vector<double> gmap; ///< [slots * npina]
        std::vector<double> fmap; ///< [slots * ng * npina]
        /// (model index, flat depletion-point index) -> slot.  A std::map and
        /// not a hash: it is built once, read nz*nxya times per statepoint, and
        /// an ordering is easier to audit than a hash of a pair.
        std::map<std::pair<size_t, size_t>, int> slot_of;

        // refilled every statepoint
        std::vector<int>    plane_lo, plane_hi; ///< [nz * nxya], -1 = no fuel
        std::vector<double> plane_alpha;
        std::vector<double> xskf; ///< [ng * nxyz]
    };
    ReconStaging _recon;

    /// Flatten the quadrature table and register every reference depletion
    /// point.  Returns false -- once, permanently, with a reason -- when the
    /// library and the geometry disagree about npins/ng, which is the one shape
    /// the device kernels cannot be given.
    bool buildReconStaging();

    /// Will this statepoint's reconstruction run on the device?  Asked BEFORE
    /// the drive, because that is when the coefficient D2H is decided.  Every
    /// term is a property of the run or of the library, not of the statepoint,
    /// so the answer cannot change between here and the reconstruction -- with
    /// one exception, a CUDA failure inside it, which is what the repair path
    /// in reconstructPinPower() exists for.
    bool reconPlanned();

    /// @brief WP6 stage D.  reconstructPinPower on the device.
    ///
    /// Returns false having written nothing when the arm is off
    /// (RASBERY_GPU_PPR_RECON), the drive did not run on the device, the mode
    /// is pointwise, or any CUDA call fails.  MASTER mode IS served: its
    /// expansion is the same quadrature over a 15-term dot product of `c`.
    bool reconstructPinPowerGpu(bool use_quadrature, bool reconstruct_flux,
                                bool materialize_pin_map);

    /// @brief MASTER MM 6.1 reconstruction: fill _c with the 13-term Legendre
    /// coefficients (even terms from surfaces/currents, cross terms from the
    /// CPB corner solve).
    void driveMaster(int niter);

    /// @brief Outward net current at surface lr of node lk, reduced by 2D/h
    /// (the current unit of MM Eq. 6.6/6.8).
    inline double getJoutRed(int lr, int dir, int lk, int g) {
        if (lk < 0 || lk >= _nxyz) return 0.0;
        const double D = _xs.xsdf(g, lk);
        if (D <= 0.0) return 0.0;
        const double h   = _g.hmesh(dir, lk);
        const double jn  = _jnet[_g.lktosfc(lr, dir, lk) * _ng + g];
        const double sgn = (lr == RIGHT) ? 1.0 : -1.0;
        return sgn * jn * h / (2.0 * D);
    }

    /// Take this cohort's table (building it if the process has not yet).
    void acquireQuadratureTable();

    /// @brief Fused update: Particular + Homogeneous + ProjectFlux in single pass
    void updateFused(int lk, int g);

    /// @brief Flux from general solution at (x, y)
    double phig(int lk, int g, double x, double y);

    double jnetDir(int dir, int lk, int g, double x, double y, bool xrev, bool yrev);

    double jnetX(int lk, int g, double x, double y, bool xrev, bool yrev);

    double jnetY(int lk, int g, double x, double y, bool xrev, bool yrev);

    /// @brief Build 3×3 neighbor stencil and reflection flags from neibrb.
    void buildStencil(int lk, int idx[3][3], bool xrev[3][3], bool yrev[3][3]);

    inline double getLeakage(int lk, int g) { return (lk >= 0 && lk < _nxyz) ? zleak(lk) : 0.0; }

public:
    PPR(Geometry& g, XSSet& xs);

    ~PPR() = default;

    /// @brief Reset pointers and recompute buckling / corner flux / fitting coefficients
    void reset(const double reigv, double* jnet, const double* phif, double* phis);

    double getPhis(int side, int dir, int lk, int g);

    /// @brief Run the pin power reconstruction iteration
    void drive(int niter);

    /// @brief reset() + drive(niter) on the device (RASBERY_GPU_PPR).
    ///
    /// Returns false -- having touched nothing -- when the arm is off, the
    /// build has no CUDA, the deck is not 2-group, or any CUDA call fails.
    /// EVERY ONE OF THOSE NAMES ITSELF on the way out, through
    /// PprBackend::noteHostFallback / the backend's own ladder, so the
    /// [RASBERY][PPR_GPU] receipt carries `refusal` and `refusals` beside
    /// `host_fallbacks`.  The caller must then run reset() and drive() exactly
    /// as before.  On success the host coefficient arrays
    /// reconstructPinPower() reads (_p, _a, _c, _bt, and _phic/_q/_l for
    /// completeness) hold the device result and the nodal pointers are set the
    /// same way reset() sets them.
    ///
    /// WP6 STAGE F: RASBERY_PPR_MODE=master IS SERVED.  It used to be the first
    /// refusal in the function, which meant the arm declined every statepoint
    /// of the production configuration -- master mode is what gives Gate B pin
    /// RMS 0.238 % against the default mode's 0.522 % -- while the receipt said
    /// only `host_fallbacks:35`.  The mode now travels in ppr::StepView and the
    /// backend selects the MASTER MM 6.1 body (interpolant + corner-point
    /// balance) instead of the SENM Picard one.  The device CPB solve is
    /// JACOBI where the host's is Gauss-Seidel -- the host writes _phic in
    /// place while later nodes read it, which is a serial dependence over the
    /// whole mesh -- so the arm is class N1 there and nowhere else in the
    /// master path; see CudaPprBackend.h for the contraction argument.
    bool resetAndDriveGpu(double reigv, double* jnet, const double* phif,
                          double* phis, int niter);

    /// @brief WP6 stage C.  Offer (or withdraw) the device-resident nodal set.
    ///
    /// Called once per statepoint by the Driver, immediately before
    /// resetAndDriveGpu.  A default-constructed argument withdraws the offer,
    /// which is what a statepoint whose outer segment was not resident must
    /// pass -- silence would leave the previous statepoint's offer standing.
    void adoptCanonicalDeviceInputs(const PprCanonicalInputs& in) { _canonical = in; }

    /// @brief The device arm's receipt source (never null).
    [[nodiscard]] const PprBackend& gpu() const { return *_gpu; }

    /// @brief Corner-balance iterations the HOST drive() has spent this run.
    ///
    /// The device arm reports its own (PprBackend::iterations()).  Both are
    /// printed so "did the break test move?" is a comparison and not a claim:
    /// the two arms apply the same RelativeChange test to the same tolerance,
    /// but the device sums the corner fluxes on a 256-chunk partition, so the
    /// iteration counts are allowed to differ and have to be seen.
    [[nodiscard]] unsigned long long hostIterations() const { return _host_iters; }

    /// @brief Update axial leakage expansion
    void updateAxialLeakage();

    /// @brief Update fission source shape
    void updateSource();

    /// @brief Update corner flux from current continuity
    void updateCorner();

    /// @brief Reconstruct pin-wise flux and power from PPR coefficients
    /// @param use_quadrature true = 3x3 Gauss integration per pin, false = single center-point evaluation
    /// @param reconstruct_flux true = rebuild pin-wise flux with fmap/pphif, false = power only
    /// @param materialize_pin_map will the HOST read Geometry::PinPower() after
    ///        this call?  It has exactly one reader (IO.cpp, under
    ///        print_opt.pin_info), so a statepoint that does not print may leave
    ///        the 5.4 MB map on the device -- WP6 stage D's whole saving.  The
    ///        host path ignores this: it writes the array either way, because
    ///        it computes in place.
    void reconstructPinPower(bool use_quadrature = true, bool reconstruct_flux = false,
                             bool materialize_pin_map = true);
};
} // namespace rasbery
