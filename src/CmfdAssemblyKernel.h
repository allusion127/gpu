#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define RASBERY_CMFD_ASSEMBLY_HD __host__ __device__
#else
#define RASBERY_CMFD_ASSEMBLY_HD
#endif

// ---------------------------------------------------------------------------
// WP21-A: WHERE THE NODE-INDEXED CMFD ARRAYS LIVE.
//
// Historically every node-indexed CMFD array was packed NODE-MAJOR: the four
// diagonal entries of node l at diag[4*l + k], the twelve couplings at
// cc[12*l + j], the six neighbours at neighbors[6*l + s], the three face areas
// at face_area[3*l + d].  One CMFD thread owns one node (element i = 2*l + ig,
// see docs/WP17_CMFD_OCCUPANCY_20260830_KO.md), so consecutive threads of a
// warp read addresses 32, 96, 24 and 24 bytes apart and each one lands in its
// own 32-byte sector.  Block 39's ncu sweep on 238 measured 6.55-9.92 sectors
// per request across the five CMFD/BiCG kernels against an ideal of 2.
//
// The layout below stores the SAME values COMPONENT-MAJOR instead: component k
// of node l at k*nxyz + l, so consecutive nodes are adjacent doubles, which is
// what a warp wants.  It is a pure permutation of storage:
//
//   * every kernel reads the same value for the same (l, k);
//   * every kernel evaluates the same expression on the same operands in the
//     same order -- not one arithmetic site below or in CudaBICGBackend.cu
//     changes -- so nvcc makes the same contraction decisions;
//   * no reduction partition moves.  The dot products walk the ELEMENT index
//     i in [0, n) over the Krylov vectors, and those vectors are deliberately
//     LEFT node-major (see the inventory in
//     docs/WP21_A_CMFD_COALESCING_20260831_KO.md): permuting them would
//     re-chunk the fixed `chunk = ceil(n/gridDim.x)` partition and move
//     additions, which is exactly the bit-identity the arena rests on.
//
// So the run is bit-identical (Class B0) and the digest gate is the proof.
//
// kLayoutVersion is part of the graph capture key and of the
// [RASBERY][CMFD][GRAPH] receipt, so a graph instantiated under one layout can
// never be replayed under the other.
//   1 = node-major AoS (every build before WP21-A)
//   2 = component-major SoA (this build)
// ---------------------------------------------------------------------------
namespace rasbery::cmfd_layout {

/// The one switch.  A compile-time constant on purpose: both branches below
/// fold away, so neither layout pays a runtime test per access, and a build
/// carries exactly one layout end to end.
constexpr bool kNodeInnermost = true;
constexpr int  kLayoutVersion = kNodeInnermost ? 2 : 1;

inline const char* layoutName() { return kNodeInnermost ? "soa" : "aos"; }

/// Component @p k of node @p l in an array carrying @p per_node components per
/// node over @p nxyz nodes.
RASBERY_CMFD_ASSEMBLY_HD inline long long component(int nxyz, int l, int k,
                                                    int per_node) {
    return kNodeInnermost ? static_cast<long long>(k) * nxyz + l
                          : static_cast<long long>(l) * per_node + k;
}

/// diag / dinv / udiag / xssm and their float mirrors: ng*ng = 4 per node.
RASBERY_CMFD_ASSEMBLY_HD inline long long mat(int nxyz, int l, int k) {
    return component(nxyz, l, k, 4);
}

/// cc / cc_f: ng*NDIRMAX*LR = 12 couplings per node.
RASBERY_CMFD_ASSEMBLY_HD inline long long cpl(int nxyz, int l, int j) {
    return component(nxyz, l, j, 12);
}

/// neighbors / node_surface: NDIRMAX*LR = 6 faces per node.
RASBERY_CMFD_ASSEMBLY_HD inline long long face(int nxyz, int l, int f) {
    return component(nxyz, l, f, 6);
}

/// face_area: NDIRMAX = 3 directions per node.
RASBERY_CMFD_ASSEMBLY_HD inline long long dir(int nxyz, int l, int d) {
    return component(nxyz, l, d, 3);
}

} // namespace rasbery::cmfd_layout

namespace rasbery::cmfd_assembly {

constexpr int kGroups = 2;
constexpr int kDirections = 3;
constexpr int kSides = 2;
constexpr int kFacesPerNode = kDirections * kSides;
constexpr int kMatrixPerNode = kGroups * kGroups;
constexpr int kCouplingPerNode = kGroups * kFacesPerNode;

struct View {
    int nxyz;

    // Geometry-static, shared by every arena slot.  WP21-A: node_surface and
    // face_area are addressed through cmfd_layout::face / cmfd_layout::dir, so
    // they are [direction][left/right][node] and [direction][node] under the
    // SoA layout and the historical node-major packings under AoS.
    const int*    node_surface; ///< cmfd_layout::face(nxyz, node, dir*LR+lr)
    const double* face_area;    ///< cmfd_layout::dir(nxyz, node, direction)
    const double* volume;       ///< [node]

    // Slot-local physics state. XS arrays are group-major SoA; dtil/dhat are
    // surface-major with the two groups adjacent.
    const double* xsrf; ///< [group][node]
    const double* xssm; ///< [from_group][to_group][node]
    const double* chif; ///< [to_group][node]
    const double* xsnf; ///< [from_group][node]
    const double* dtil; ///< [surface][group]
    const double* dhat; ///< [surface][group]

    double reigvs;
    double eshift;

    // Slot-local outputs, in the layout consumed by BiCGSTAB.  WP21-A: written
    // through cmfd_layout::mat / cmfd_layout::cpl, i.e. component-major
    // ([to_group][from_group][node], [group][direction][left/right][node])
    // under the SoA layout and node-major under AoS.  The component ORDER
    // within a node is untouched either way, so the k of diag[k] still means
    // ige*ng+igs and the j of cc[j] still means ig*NDIRMAX*LR + idir*LR + lr.
    double* diag;  ///< cmfd_layout::mat(nxyz, node, ige*ng+igs)
    double* cc;    ///< cmfd_layout::cpl(nxyz, node, ig*NDIRMAX*LR+idir*LR+lr)
    double* udiag; ///< unshifted diagonal, same layout as diag
};

// The four `diagonal +=` accumulation sites below are an INFERRED contraction:
// unlike the Wielandt shift (mined from live gcc codegen via the form probe),
// they were never captured by RASBERY_CMFD_DUMP.  Empirically gcc 13 fuses all
// of them at -O3 -march=native (the production Release flags) and does NOT at
// -O2 or below or with -ffp-contract=off.  If the golden gate ever fails on a
// non-Release or non-gcc host build, define RASBERY_CMFD_ASSEMBLY_NO_CONTRACT
// to bisect the accumulation form without rolling back the whole feature.
RASBERY_CMFD_ASSEMBLY_HD inline double mulAdd(double a, double b, double c) {
#if defined(RASBERY_CMFD_ASSEMBLY_NO_CONTRACT)
    return a * b + c;
#elif defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

RASBERY_CMFD_ASSEMBLY_HD inline double roundedMul(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}

/// Assemble one two-group CMFD node in the exact loop and accumulation order
/// used by CMFD::setls + BICGCMFD::updls.
///
/// Each invocation owns one node, so all writes are disjoint. The function is
/// shared by the CUDA kernel and the host-only arithmetic contract test.
RASBERY_CMFD_ASSEMBLY_HD inline void assembleNode2G(const View& v, int l) {
    const double vol = v.volume[l];
    // WP21-A: the per-node base pointers are gone -- the component index now
    // carries the node through cmfd_layout, so nothing but the ADDRESS of each
    // store moves.  Every expression, every operand and every accumulation
    // order below is character for character what it was.
    const int nxyz = v.nxyz;

    for (int ige = 0; ige < kGroups; ++ige) {
        for (int igs = 0; igs < kGroups; ++igs) {
            const double scatter = v.xssm[(igs * kGroups + ige) * v.nxyz + l];
            v.diag[cmfd_layout::mat(nxyz, l, ige * kGroups + igs)] =
                -roundedMul(scatter, vol);
        }

        double& diagonal = v.diag[cmfd_layout::mat(nxyz, l, ige * kGroups + ige)];
        diagonal = mulAdd(v.xsrf[ige * v.nxyz + l], vol, diagonal);

        // Preserve the historical left-face Z,Y,X accumulation order.
        for (int idir = kDirections - 1; idir >= 0; --idir) {
            const int face = idir * kSides;
            const int ls = v.node_surface[cmfd_layout::face(nxyz, l, face)];
            const double area = v.face_area[cmfd_layout::dir(nxyz, l, idir)];
            const double dt = v.dtil[ls * kGroups + ige];
            const double dh = v.dhat[ls * kGroups + ige];
            v.cc[cmfd_layout::cpl(nxyz, l, ige * kFacesPerNode + face)] =
                roundedMul(-dt + dh, area);
            diagonal = mulAdd(dt + dh, area, diagonal);
        }

        // Preserve the historical right-face X,Y,Z accumulation order.
        for (int idir = 0; idir < kDirections; ++idir) {
            const int face = idir * kSides + 1;
            const int ls = v.node_surface[cmfd_layout::face(nxyz, l, face)];
            const double area = v.face_area[cmfd_layout::dir(nxyz, l, idir)];
            const double dt = v.dtil[ls * kGroups + ige];
            const double dh = v.dhat[ls * kGroups + ige];
            v.cc[cmfd_layout::cpl(nxyz, l, ige * kFacesPerNode + face)] =
                roundedMul(-dt - dh, area);
            diagonal = mulAdd(dt - dh, area, diagonal);
        }
    }

    for (int ige = 0; ige < kGroups; ++ige) {
        for (int igs = 0; igs < kGroups; ++igs) {
            const int idx = ige * kGroups + igs;
            const long long at = cmfd_layout::mat(nxyz, l, idx);
            const double unshifted = v.diag[at];
            v.udiag[at] = unshifted;
            if (v.eshift != 0.0) {
                // Match the mined host form already used by cmfd_updls:
                // round(round(chif*xsnf)*reigvs), then fused subtract * volume.
                const double c2 = roundedMul(
                    roundedMul(v.chif[ige * v.nxyz + l],
                               v.xsnf[igs * v.nxyz + l]),
                    v.reigvs);
                v.diag[at] = mulAdd(-c2, vol, unshifted);
            }
        }
    }
}

} // namespace rasbery::cmfd_assembly

#undef RASBERY_CMFD_ASSEMBLY_HD
