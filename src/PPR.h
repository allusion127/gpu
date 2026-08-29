#pragma once

#include "CudaPprBackend.h"
#include "Geometry.h"
#include "PprQuadrature.h"
#include "XSSet.h"
#include "pch.h"
#include <cmath>
#include <memory>
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
    /// build has no CUDA, the deck is not 2-group, RASBERY_PPR_MODE=master is
    /// selected, or any CUDA call fails.  The caller must then run reset() and
    /// drive() exactly as before.  On success the host coefficient arrays
    /// reconstructPinPower() reads (_p, _a, _c, _bt, and _phic/_q/_l for
    /// completeness) hold the device result and the nodal pointers are set the
    /// same way reset() sets them.
    bool resetAndDriveGpu(double reigv, double* jnet, const double* phif,
                          double* phis, int niter);

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
    void reconstructPinPower(bool use_quadrature = true, bool reconstruct_flux = false);
};
} // namespace rasbery
