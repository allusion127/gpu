#pragma once

// The CPU reference for the Task 5 CMFD outer bodies, and the fixture that
// drives it.
//
// WHY THE REFERENCE IS A SEPARATE TRANSLATION UNIT.  Task 4 paid for this
// lesson: with a verbatim CPU quotation and the shipped body in one TU and one
// loop, gcc common-subexpressions across them and changes the QUOTATION's
// contraction, so the harness fails against a reference that is no longer the
// production one.  Reaching the quotation across a plain call boundary changes
// the choices too, and letting the operands be compile-time visible folds the
// whole comparison away.  So: the quotations live in CmfdOuterReference.cpp,
// alone, inlined into batch loops over arrays this TU cannot see through --
// which is exactly how Nodal/CMFD evaluate them in production.
//
// The fixture builder below is NOT part of the reference: it only produces
// input, so it is free to be inline and shared.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cmfdref {

constexpr int NG   = 2;
constexpr int NDIR = 3;
constexpr int NLR  = 2;

/// Everything the four bodies read.  Pointers, so the reference TU cannot see
/// the values.
struct Mesh {
    int nxyz  = 0;
    int nsurf = 0;
    int ng    = NG;

    const int*    surface_node    = nullptr; ///< [ls*LR + side], -1 = boundary
    const int*    surface_dir     = nullptr; ///< [ls*LR + side]
    const double* node_hmesh      = nullptr; ///< [l*NDIRMAX + dir]
    const double* node_volume     = nullptr; ///< [l]
    const double* boundary_albedo = nullptr; ///< [dir*LR + side]

    const double* xsdf = nullptr; ///< [ig*nxyz + l]
    const double* xsnf = nullptr; ///< [ig*nxyz + l]
    const double* flux = nullptr; ///< [l*ng + ig]
    const double* jnet = nullptr; ///< [ls*ng + ig]
    const double* dtil = nullptr; ///< [ls*ng + ig]
    const double* dhat = nullptr; ///< [ls*ng + ig]
};

/// CMFD's four dhat diagnostics, accumulated in the host's order.
struct DhatCounters {
    long long total      = 0;
    long long fsum_guard = 0;
    long long clamped    = 0;
    double    ratio_max  = 0.0;
};

// --- the reference bodies (defined in CmfdOuterReference.cpp) --------------

void refUpdDtil(const Mesh& m, double* dtil_out);
void refUpdPsi(const Mesh& m, double* psi_out);
void refUpdJnet(const Mesh& m, double* jnet_out);
void refUpdDhat(const Mesh& m, bool clamp_enabled, double* dhat_out, DhatCounters* counters);

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Which upddhat branch a (surface, group) is built to exercise.  The probe and
/// the replay both assert every one of these is actually hit -- a guard that is
/// never taken is a guard that was never tested, and upddhat's guards are where
/// the CNCC correction is allowed to throw the answer away.
enum class DhatBranch : int {
    Normal = 0,       ///< finite dh, |dhat| <= |dtil|
    FsumBelowFloor,   ///< |fsum| <= 1e-12*max(1,|dtil|)
    FsumNonFinite,    ///< fsum is inf or nan
    QuotientNonFinite,///< fsum finite and above the floor, dh still non-finite
    ZeroDtil,         ///< |dtil| == 0, so the ratio branch is skipped entirely
    OverEnvelope,     ///< |dhat| > |dtil|: counted, and clamped when enabled
    Count
};

/// A synthetic 1-D chain: surface ls joins node ls-1 (left) and node ls (right),
/// so surface 0 has no left node, surface nxyz has no right node, and everything
/// between is internal.  That covers updjnet's and upddhat's three topological
/// branches without a real deck.
///
/// On top of that, specific surfaces are poisoned to reach each DhatBranch.
struct Fixture {
    int                 nxyz = 0;
    int                 nsurf = 0;
    std::vector<int>    surface_node, surface_dir;
    std::vector<double> node_hmesh, node_volume, boundary_albedo;
    std::vector<double> xsdf, xsnf, flux, jnet, dtil, dhat;
    std::vector<int>    branch; ///< [ls*ng + ig], the DhatBranch each site targets

    [[nodiscard]] Mesh mesh() const {
        Mesh m;
        m.nxyz            = nxyz;
        m.nsurf           = nsurf;
        m.ng              = NG;
        m.surface_node    = surface_node.data();
        m.surface_dir     = surface_dir.data();
        m.node_hmesh      = node_hmesh.data();
        m.node_volume     = node_volume.data();
        m.boundary_albedo = boundary_albedo.data();
        m.xsdf            = xsdf.data();
        m.xsnf            = xsnf.data();
        m.flux            = flux.data();
        m.jnet            = jnet.data();
        m.dtil            = dtil.data();
        m.dhat            = dhat.data();
        return m;
    }
};

inline Fixture buildFixture(int nxyz) {
    if (nxyz < 32) nxyz = 32;
    Fixture f;
    f.nxyz  = nxyz;
    f.nsurf = nxyz + 1;

    f.surface_node.assign(static_cast<size_t>(f.nsurf) * NLR, -1);
    f.surface_dir.assign(static_cast<size_t>(f.nsurf) * NLR, 0);
    f.node_hmesh.assign(static_cast<size_t>(nxyz) * NDIR, 0.0);
    f.node_volume.assign(static_cast<size_t>(nxyz), 0.0);
    f.boundary_albedo.assign(static_cast<size_t>(NDIR) * NLR, 0.0);
    f.xsdf.assign(static_cast<size_t>(NG) * nxyz, 0.0);
    f.xsnf.assign(static_cast<size_t>(NG) * nxyz, 0.0);
    f.flux.assign(static_cast<size_t>(nxyz) * NG, 0.0);
    f.jnet.assign(static_cast<size_t>(f.nsurf) * NG, 0.0);
    f.dtil.assign(static_cast<size_t>(f.nsurf) * NG, 0.0);
    f.dhat.assign(static_cast<size_t>(f.nsurf) * NG, 0.0);
    f.branch.assign(static_cast<size_t>(f.nsurf) * NG, static_cast<int>(DhatBranch::Normal));

    for (int d = 0; d < NDIR; ++d) {
        f.boundary_albedo[static_cast<size_t>(d) * NLR + 0] = 0.4692 + 0.01 * d;
        f.boundary_albedo[static_cast<size_t>(d) * NLR + 1] = 0.5 - 0.013 * d;
    }

    const double hm[] = {1.26, 3.81, 5.355, 10.71, 15.24, 21.42, 30.48};
    for (int l = 0; l < nxyz; ++l) {
        for (int d = 0; d < NDIR; ++d)
            f.node_hmesh[static_cast<size_t>(l) * NDIR + d] = hm[(l + d) % 7];
        f.node_volume[static_cast<size_t>(l)] =
            f.node_hmesh[static_cast<size_t>(l) * NDIR + 0] *
            f.node_hmesh[static_cast<size_t>(l) * NDIR + 1] *
            f.node_hmesh[static_cast<size_t>(l) * NDIR + 2];

        const double xsdf_f[] = {1.4326, 1.2887, 1.1204};
        const double xsdf_t[] = {0.3721, 0.4015, 0.2687};
        const double xsnf_f[] = {0.0056, 0.0071, 0.0034};
        const double xsnf_t[] = {0.0921, 0.1153, 0.0607};
        f.xsdf[static_cast<size_t>(0) * nxyz + l] = xsdf_f[l % 3];
        f.xsdf[static_cast<size_t>(1) * nxyz + l] = xsdf_t[(l / 3) % 3];
        f.xsnf[static_cast<size_t>(0) * nxyz + l] = xsnf_f[l % 3];
        f.xsnf[static_cast<size_t>(1) * nxyz + l] = xsnf_t[(l / 2) % 3];
        // A flux profile with a real shape, so fr - fl is not systematically
        // tiny and the internal updjnet site is genuinely exercised.
        f.flux[static_cast<size_t>(l) * NG + 0] = 1.0 + 0.37 * std::sin(0.11 * l);
        f.flux[static_cast<size_t>(l) * NG + 1] = 0.21 + 0.09 * std::cos(0.17 * l);
    }

    for (int ls = 0; ls < f.nsurf; ++ls) {
        const int idir = ls % NDIR;
        f.surface_dir[static_cast<size_t>(ls) * NLR + 0] = idir;
        f.surface_dir[static_cast<size_t>(ls) * NLR + 1] = idir;
        f.surface_node[static_cast<size_t>(ls) * NLR + 0] = (ls == 0) ? -1 : ls - 1;
        f.surface_node[static_cast<size_t>(ls) * NLR + 1] = (ls == f.nsurf - 1) ? -1 : ls;
        for (int ig = 0; ig < NG; ++ig) {
            f.dtil[static_cast<size_t>(ls) * NG + ig] = 0.031 + 0.004 * ((ls + ig) % 5);
            f.dhat[static_cast<size_t>(ls) * NG + ig] = 0.0011 * ((ls % 7) - 3);
            f.jnet[static_cast<size_t>(ls) * NG + ig] =
                0.0043 * std::sin(0.23 * ls + 0.7 * ig);
        }
    }

    // --- poison specific sites so every guard branch is reached --------------
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf  = std::numeric_limits<double>::infinity();
    auto mark = [&](int ls, int ig, DhatBranch b) {
        f.branch[static_cast<size_t>(ls) * NG + ig] = static_cast<int>(b);
    };

    // fsum below the floor: two internal nodes whose fluxes cancel exactly.
    {
        const int ls = 5, ll = ls - 1, lr = ls;
        for (int ig = 0; ig < NG; ++ig) {
            f.flux[static_cast<size_t>(ll) * NG + ig] = 0.25;
            f.flux[static_cast<size_t>(lr) * NG + ig] = -0.25;
            mark(ls, ig, DhatBranch::FsumBelowFloor);
        }
    }
    // fsum non-finite.
    {
        const int ls = 9, lr = ls;
        f.flux[static_cast<size_t>(lr) * NG + 0] = inf;
        f.flux[static_cast<size_t>(lr) * NG + 1] = qnan;
        for (int ig = 0; ig < NG; ++ig) mark(ls, ig, DhatBranch::FsumNonFinite);
    }
    // Quotient non-finite while fsum is fine: an infinite jnet makes the
    // numerator infinite, so dh is inf even though fsum passed both tests.
    {
        const int ls = 13;
        for (int ig = 0; ig < NG; ++ig) {
            f.jnet[static_cast<size_t>(ls) * NG + ig] = inf;
            mark(ls, ig, DhatBranch::QuotientNonFinite);
        }
    }
    // |dtil| == 0: the ratio block is skipped entirely, and the floor collapses
    // to 1e-12*1.0 rather than 1e-12*|dtil|.
    {
        const int ls = 17;
        for (int ig = 0; ig < NG; ++ig) {
            f.dtil[static_cast<size_t>(ls) * NG + ig] = 0.0;
            mark(ls, ig, DhatBranch::ZeroDtil);
        }
    }
    // |dhat| > |dtil|: a large jnet against a small dtil.
    {
        const int ls = 21;
        for (int ig = 0; ig < NG; ++ig) {
            f.dtil[static_cast<size_t>(ls) * NG + ig] = 1.0e-4;
            f.jnet[static_cast<size_t>(ls) * NG + ig] = 3.7;
            mark(ls, ig, DhatBranch::OverEnvelope);
        }
    }
    // A second over-envelope site with the opposite sign, so the clamp's
    // `dh > 0 ? cap : -cap` is exercised both ways.
    {
        const int ls = 25;
        for (int ig = 0; ig < NG; ++ig) {
            f.dtil[static_cast<size_t>(ls) * NG + ig] = 1.0e-4;
            f.jnet[static_cast<size_t>(ls) * NG + ig] = -4.1;
            mark(ls, ig, DhatBranch::OverEnvelope);
        }
    }
    return f;
}

} // namespace cmfdref
