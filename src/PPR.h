#pragma once

#include "Geometry.h"
#include "XSSet.h"
#include "pch.h"
#include <cmath>

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

// Pre-computed quadrature point data for pin power reconstruction
struct QuadPoint {
    double xq, yq;  // local [-1,1] coords within sub-node
    double leg[15]; // Lx[i]*Ly[j] products (upper-triangular ordering)
    double wt;      // wi3[qi] * wi3[qj]
};

struct PinOverlap {
    int       di, dj;     // sub-node indices within assembly
    double    dx_h, dy_h; // half-widths for jacobian
    QuadPoint qpts[9];    // 3x3 Gauss-Legendre points
};

struct PinQuadInfo {
    std::vector<PinOverlap> overlaps;
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
    double* _phif; // -> g.Phif()
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

    // Pre-computed quadrature table (built once, indexed by pin)
    std::vector<PinQuadInfo> _pin_quad_table;
    bool                     _quad_table_built = false;

    void buildQuadratureTable();

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
    void reset(const double reigv, double* jnet, double* phif, double* phis);

    double getPhis(int side, int dir, int lk, int g);

    /// @brief Run the pin power reconstruction iteration
    void drive(int niter);

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
