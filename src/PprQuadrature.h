#pragma once

// The PPR pin-power quadrature table, split out of PPR.h.
//
// WHY IT MOVED.  It is the cleanest cohort-shareable object in the tree
// (WP8 stage 2): `buildPinQuadratureTable` reads NOTHING but `ndivxy` and
// `npins` -- no flux, no cross sections, no loading pattern -- and every value
// it writes (the sub-node overlap indices, the half-widths, the 3x3
// Gauss-Legendre coordinates and weights, and the fifteen precomputed Legendre
// products) is a function of that pair alone.  It was rebuilt once per `PPR`
// object, i.e. once per Driver, i.e. once per CASE; a 64-case wave built 64
// bit-identical copies of it.
//
// It lives in its own header so `CohortContext.h` can hold the table without
// including PPR.h, which drags in Geometry, XSSet and the CUDA backend header.
// The types are plain aggregates with no invariant, which is what makes them
// safe to publish as `shared_ptr<const>` and read from many threads at once.

#include <vector>

namespace rasbery {

/// One 3x3 Gauss-Legendre point inside a sub-node.
struct QuadPoint {
    double xq, yq;  // local [-1,1] coords within sub-node
    double leg[15]; // Lx[i]*Ly[j] products (upper-triangular ordering)
    double wt;      // wi3[qi] * wi3[qj]
};

/// The part of one sub-node a pin cell overlaps.
struct PinOverlap {
    int       di, dj;     // sub-node indices within assembly
    double    dx_h, dy_h; // half-widths for jacobian
    QuadPoint qpts[9];    // 3x3 Gauss-Legendre points
};

struct PinQuadInfo {
    std::vector<PinOverlap> overlaps;
};

/// `npins * npins` entries, indexed `py * npins + px`.
using PinQuadTable = std::vector<PinQuadInfo>;

/// Build the table for one (ndivxy, npins) pair.
///
/// Defined in PPR.cpp, unchanged from the member function it was: the values
/// have to be bit-identical to the ones a per-case build produced, because the
/// gate for this whole work package is that the digest does not move.
PinQuadTable buildPinQuadratureTable(int ndivxy, int npins);

} // namespace rasbery
