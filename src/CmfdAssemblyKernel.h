#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define RASBERY_CMFD_ASSEMBLY_HD __host__ __device__
#else
#define RASBERY_CMFD_ASSEMBLY_HD
#endif

namespace rasbery::cmfd_assembly {

constexpr int kGroups = 2;
constexpr int kDirections = 3;
constexpr int kSides = 2;
constexpr int kFacesPerNode = kDirections * kSides;
constexpr int kMatrixPerNode = kGroups * kGroups;
constexpr int kCouplingPerNode = kGroups * kFacesPerNode;

struct View {
    int nxyz;

    // Geometry-static, shared by every arena slot.
    const int*    node_surface; ///< [node][direction][left/right]
    const double* face_area;    ///< [node][direction]
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

    // Slot-local outputs, node-major in the layout consumed by BiCGSTAB.
    double* diag;  ///< [node][to_group][from_group]
    double* cc;    ///< [node][group][direction][left/right]
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
    double* const diag_l = v.diag + l * kMatrixPerNode;
    double* const cc_l = v.cc + l * kCouplingPerNode;
    double* const udiag_l = v.udiag + l * kMatrixPerNode;

    for (int ige = 0; ige < kGroups; ++ige) {
        for (int igs = 0; igs < kGroups; ++igs) {
            const double scatter = v.xssm[(igs * kGroups + ige) * v.nxyz + l];
            diag_l[ige * kGroups + igs] = -roundedMul(scatter, vol);
        }

        double& diagonal = diag_l[ige * kGroups + ige];
        diagonal = mulAdd(v.xsrf[ige * v.nxyz + l], vol, diagonal);

        // Preserve the historical left-face Z,Y,X accumulation order.
        for (int idir = kDirections - 1; idir >= 0; --idir) {
            const int face = idir * kSides;
            const int ls = v.node_surface[l * kFacesPerNode + face];
            const double area = v.face_area[l * kDirections + idir];
            const double dt = v.dtil[ls * kGroups + ige];
            const double dh = v.dhat[ls * kGroups + ige];
            cc_l[ige * kFacesPerNode + face] = roundedMul(-dt + dh, area);
            diagonal = mulAdd(dt + dh, area, diagonal);
        }

        // Preserve the historical right-face X,Y,Z accumulation order.
        for (int idir = 0; idir < kDirections; ++idir) {
            const int face = idir * kSides + 1;
            const int ls = v.node_surface[l * kFacesPerNode + face];
            const double area = v.face_area[l * kDirections + idir];
            const double dt = v.dtil[ls * kGroups + ige];
            const double dh = v.dhat[ls * kGroups + ige];
            cc_l[ige * kFacesPerNode + face] = roundedMul(-dt - dh, area);
            diagonal = mulAdd(dt - dh, area, diagonal);
        }
    }

    for (int ige = 0; ige < kGroups; ++ige) {
        for (int igs = 0; igs < kGroups; ++igs) {
            const int idx = ige * kGroups + igs;
            const double unshifted = diag_l[idx];
            udiag_l[idx] = unshifted;
            if (v.eshift != 0.0) {
                // Match the mined host form already used by cmfd_updls:
                // round(round(chif*xsnf)*reigvs), then fused subtract * volume.
                const double c2 = roundedMul(
                    roundedMul(v.chif[ige * v.nxyz + l],
                               v.xsnf[igs * v.nxyz + l]),
                    v.reigvs);
                diag_l[idx] = mulAdd(-c2, vol, unshifted);
            }
        }
    }
}

} // namespace rasbery::cmfd_assembly

#undef RASBERY_CMFD_ASSEMBLY_HD
