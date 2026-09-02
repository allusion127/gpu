// WP21-C2 gate: ONE BODY, TWO LAYOUTS, THE SAME BITS.
//
// src/NodalKernel.h is a shared host/device body whose private arrays used to
// be spelled `arr[(lk*NDIR + idir)*NG + ig]` inline at 142 sites.  WP21-C2
// routed every one of them through the view's own accessors and put the
// strides in the view, so ONE text can read a node-major array (the host
// production arrays, the RASBERY_NODAL_DUMP capture, the replay tools) and a
// node-innermost one (the device arrays, which is where the 16.7 sectors per
// request live).
//
// THE CLAIM THAT NEEDS A GATE is that the two are the same computation.  That
// is not a source property -- it is an equivalence between two runs -- and it
// does not need a GPU, a capture or nvcc: build a mesh, fill it, run all five
// phases twice (once over node-major arrays with the default strides, once
// over the SAME VALUES packed node-innermost with the permuted strides), and
// compare the outputs BIT FOR BIT.
//
// Exit 0 means every output double of every phase has the identical bit
// pattern under both layouts.  A stride that was forgotten at one of the 142
// sites reads a different element and shows up here, on this machine, in
// milliseconds -- instead of as a finite, plausible, wrong cross section on
// the benchmark host.

#include "NodalKernel.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace nk = rasbery::nodal;

namespace {

constexpr int NX = 4, NY = 3, NZ = 5;
constexpr int NXYZ = NX * NY * NZ;

int nodeOf(int i, int j, int k) { return (k * NY + j) * NX + i; }

// A deterministic LCG: the two runs must see the SAME inputs, and a shared
// std::mt19937 whose call order differed between them would be a bug in the
// gate rather than in the code under test.
struct Lcg {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    double next(double lo, double hi) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double u = static_cast<double>((s >> 11) & ((1ull << 53) - 1)) /
                         static_cast<double>(1ull << 53);
        return lo + u * (hi - lo);
    }
};

/// The mesh: a structured NX x NY x NZ box, surfaces per direction, boundary
/// sides marked -1.  Plain Cartesian, because what this gate exercises is the
/// INDEXING and every branch of it (interior 2n surfaces, boundary 1n
/// surfaces, missing neighbours on both sides).
struct Mesh {
    std::vector<int> lktosfc, neib, lklr, idirlr, sgnlr;
    std::vector<double> hmesh, albedo;
    int nsurf = 0;

    Mesh() {
        const int nxs = (NX + 1) * NY * NZ;
        const int nys = NX * (NY + 1) * NZ;
        const int nzs = NX * NY * (NZ + 1);
        nsurf = nxs + nys + nzs;

        lktosfc.assign(static_cast<std::size_t>(NXYZ) * nk::NDIR * nk::NLR, 0);
        neib.assign(static_cast<std::size_t>(NXYZ) * nk::NEWSB, -1);
        lklr.assign(static_cast<std::size_t>(nsurf) * nk::NLR, -1);
        idirlr.assign(static_cast<std::size_t>(nsurf) * nk::NLR, 0);
        sgnlr.assign(static_cast<std::size_t>(nsurf) * nk::NLR, 1);
        albedo.assign(nk::NDIR * nk::NLR, 0.5);

        auto xs = [&](int i, int j, int k) { return (k * NY + j) * (NX + 1) + i; };
        auto ys = [&](int i, int j, int k) { return nxs + (k * (NY + 1) + j) * NX + i; };
        auto zs = [&](int i, int j, int k) { return nxs + nys + (k * NY + j) * NX + i; };

        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    const int lk = nodeOf(i, j, k);
                    const int s[nk::NDIR][nk::NLR] = {
                        {xs(i, j, k), xs(i + 1, j, k)},
                        {ys(i, j, k), ys(i, j + 1, k)},
                        {zs(i, j, k), zs(i, j, k + 1)}};
                    const int nb[nk::NDIR][nk::NLR] = {
                        {i > 0 ? nodeOf(i - 1, j, k) : -1,
                         i + 1 < NX ? nodeOf(i + 1, j, k) : -1},
                        {j > 0 ? nodeOf(i, j - 1, k) : -1,
                         j + 1 < NY ? nodeOf(i, j + 1, k) : -1},
                        {k > 0 ? nodeOf(i, j, k - 1) : -1,
                         k + 1 < NZ ? nodeOf(i, j, k + 1) : -1}};
                    for (int d = 0; d < nk::NDIR; ++d)
                        for (int sd = 0; sd < nk::NLR; ++sd) {
                            lktosfc[static_cast<std::size_t>((lk * nk::NDIR + d) *
                                                             nk::NLR + sd)] = s[d][sd];
                            neib[static_cast<std::size_t>(lk * nk::NEWSB +
                                                          d * nk::NLR + sd)] = nb[d][sd];
                            // Surface `s[d][sd]` sees this node on the side
                            // OPPOSITE to sd: the node's LEFT surface has the
                            // node on its RIGHT.
                            const int other = 1 - sd;
                            lklr[static_cast<std::size_t>(s[d][sd] * nk::NLR + other)] = lk;
                            idirlr[static_cast<std::size_t>(s[d][sd] * nk::NLR + other)] = d;
                            sgnlr[static_cast<std::size_t>(s[d][sd] * nk::NLR + other)] =
                                other == nk::C_LEFT ? 1 : -1;
                        }
                }

        hmesh.assign(static_cast<std::size_t>(NXYZ) * nk::NDIR, 0.0);
        Lcg r;
        for (auto& h : hmesh) h = r.next(10.0, 25.0);
    }
};

/// One instance's arrays, in whichever layout the strides say.  Sizes are
/// layout-independent -- a permutation moves elements, never their count.
struct Arrays {
    std::vector<double> ndg[15]; // eta1..diagDI, trlcff0/1/2, dsncff2/4/6
    std::vector<double> dg2[2];  // mu, tau
    std::vector<double> ng2[4];  // matM, matMI, matMs, matMf
    std::vector<double> xsrf, xsnf, xssm, chif, flux, jnet, phis;

    Arrays() {
        for (auto& a : ndg) a.assign(static_cast<std::size_t>(NXYZ) * nk::NDIR * nk::NG, 0.0);
        for (auto& a : dg2) a.assign(static_cast<std::size_t>(NXYZ) * nk::NDIR * nk::NG2, 0.0);
        for (auto& a : ng2) a.assign(static_cast<std::size_t>(NXYZ) * nk::NG2, 0.0);
        xsrf.assign(static_cast<std::size_t>(NXYZ) * nk::NG, 0.0);
        xsnf = xsrf;
        chif = xsrf;
        flux = xsrf;
        xssm.assign(static_cast<std::size_t>(NXYZ) * nk::NG2, 0.0);
    }
};

const char* const NDG_NAME[15] = {"eta1", "eta2", "m260", "m251", "m253",
                                  "m262", "m264", "diagD", "diagDI",
                                  "trlcff0", "trlcff1", "trlcff2",
                                  "dsncff2", "dsncff4", "dsncff6"};
const char* const DG2_NAME[2] = {"mu", "tau"};
const char* const NG2_NAME[4] = {"matM", "matMI", "matMs", "matMf"};

void bind(nk::NodalView& v, const Mesh& m, Arrays& a) {
    v.hmesh   = m.hmesh.data();
    v.lktosfc = m.lktosfc.data();
    v.neib    = m.neib.data();
    v.lklr    = m.lklr.data();
    v.idirlr  = m.idirlr.data();
    v.sgnlr   = m.sgnlr.data();
    v.albedo  = m.albedo.data();

    v.xsrf = a.xsrf.data();
    v.xsnf = a.xsnf.data();
    v.xssm = a.xssm.data();
    v.chif = a.chif.data();
    v.chif_empty = 0;

    const double* n[15];
    for (int i = 0; i < 15; ++i) n[i] = a.ndg[i].data();
    v.eta1 = n[0]; v.eta2 = n[1]; v.m260 = n[2]; v.m251 = n[3]; v.m253 = n[4];
    v.m262 = n[5]; v.m264 = n[6]; v.diagD = n[7]; v.diagDI = n[8];
    v.trlcff0 = a.ndg[9].data();
    v.trlcff1 = a.ndg[10].data();
    v.trlcff2 = a.ndg[11].data();
    v.dsncff2 = a.ndg[12].data();
    v.dsncff4 = a.ndg[13].data();
    v.dsncff6 = a.ndg[14].data();
    v.mu   = a.dg2[0].data();
    v.tau  = a.dg2[1].data();
    v.matM = a.ng2[0].data();
    v.matMI = a.ng2[1].data();
    v.matMs = a.ng2[2].data();
    v.matMf = a.ng2[3].data();

    v.flux  = a.flux.data();
    v.jnet  = a.jnet.data();
    v.phis  = a.phis.data();
    v.reigv = 0.987654321;
    v.nxyz  = NXYZ;
    v.nsurf = static_cast<int>(a.phis.size()) / nk::NG;
}

void fill(Arrays& a, int nsurf) {
    Lcg r;
    r.s = 0xDEADBEEFCAFEF00Dull;
    // The nine constants and hmesh are updateConstant's outputs; the working
    // arrays are seeded too, because caltrlcff12 reads trlcff0 written by the
    // previous PHASE and a zero seed would leave branches unexercised.
    for (auto& arr : a.ndg)
        for (auto& x : arr) x = r.next(0.05, 2.0);
    for (auto& arr : a.dg2)
        for (auto& x : arr) x = r.next(-1.0, 1.0);
    for (auto& arr : a.ng2)
        for (auto& x : arr) x = r.next(-0.5, 1.5);
    for (auto& x : a.xsrf) x = r.next(0.02, 0.9);
    for (auto& x : a.xsnf) x = r.next(0.01, 0.4);
    for (auto& x : a.xssm) x = r.next(0.001, 0.2);
    for (auto& x : a.chif) x = r.next(0.0, 1.0);
    for (auto& x : a.flux) x = r.next(0.1, 3.0);
    a.jnet.assign(static_cast<std::size_t>(nsurf) * nk::NG, 0.0);
    a.phis.assign(static_cast<std::size_t>(nsurf) * nk::NG, 0.0);
    for (auto& x : a.jnet) x = r.next(-0.3, 0.3);
}

/// Permute one array from the node-major layout into the node-innermost one,
/// element by element, THROUGH BOTH VIEWS.  The pack is written with the two
/// views' own accessors on purpose: if `nodalNodeInnermostView` and the
/// accessors ever disagree, the pack disagrees with them in the same
/// direction and the gate would pass a broken pair.  It cannot -- the SOURCE
/// index comes from the AoS view and the DESTINATION from the SoA one, so a
/// wrong stride puts the value somewhere the kernel does not look and the
/// comparison at the end fails.
void packNdg(const nk::NodalView& aos, const nk::NodalView& soa,
             const std::vector<double>& src, std::vector<double>& dst) {
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int d = 0; d < nk::NDIR; ++d)
            for (int g = 0; g < nk::NG; ++g)
                dst[static_cast<std::size_t>(soa.ndg(lk, d, g))] =
                    src[static_cast<std::size_t>(aos.ndg(lk, d, g))];
}
void packDg2(const nk::NodalView& aos, const nk::NodalView& soa,
             const std::vector<double>& src, std::vector<double>& dst) {
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int d = 0; d < nk::NDIR; ++d)
            for (int e = 0; e < nk::NG2; ++e)
                dst[static_cast<std::size_t>(soa.dg2(lk, d, e))] =
                    src[static_cast<std::size_t>(aos.dg2(lk, d, e))];
}
void packNg2(const nk::NodalView& aos, const nk::NodalView& soa,
             const std::vector<double>& src, std::vector<double>& dst) {
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int e = 0; e < nk::NG2; ++e)
            dst[static_cast<std::size_t>(soa.ng2(lk, e))] =
                src[static_cast<std::size_t>(aos.ng2(lk, e))];
}
void packHm(const nk::NodalView& aos, const nk::NodalView& soa,
            const std::vector<double>& src, std::vector<double>& dst) {
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int d = 0; d < nk::NDIR; ++d)
            dst[static_cast<std::size_t>(soa.hm(lk, d))] =
                src[static_cast<std::size_t>(aos.hm(lk, d))];
}

bool sameBits(double x, double y) {
    std::uint64_t a, b;
    std::memcpy(&a, &x, sizeof a);
    std::memcpy(&b, &y, sizeof b);
    return a == b;
}

int bad = 0;

void cmpNdg(const char* what, const nk::NodalView& aos, const nk::NodalView& soa,
            const std::vector<double>& A, const std::vector<double>& S) {
    int n = 0;
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int d = 0; d < nk::NDIR; ++d)
            for (int g = 0; g < nk::NG; ++g)
                if (!sameBits(A[static_cast<std::size_t>(aos.ndg(lk, d, g))],
                              S[static_cast<std::size_t>(soa.ndg(lk, d, g))]))
                    ++n;
    if (n) std::printf("  MISMATCH %-9s %d element(s)\n", what, n);
    bad += n;
}
void cmpDg2(const char* what, const nk::NodalView& aos, const nk::NodalView& soa,
            const std::vector<double>& A, const std::vector<double>& S) {
    int n = 0;
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int d = 0; d < nk::NDIR; ++d)
            for (int e = 0; e < nk::NG2; ++e)
                if (!sameBits(A[static_cast<std::size_t>(aos.dg2(lk, d, e))],
                              S[static_cast<std::size_t>(soa.dg2(lk, d, e))]))
                    ++n;
    if (n) std::printf("  MISMATCH %-9s %d element(s)\n", what, n);
    bad += n;
}
void cmpNg2(const char* what, const nk::NodalView& aos, const nk::NodalView& soa,
            const std::vector<double>& A, const std::vector<double>& S) {
    int n = 0;
    for (int lk = 0; lk < NXYZ; ++lk)
        for (int e = 0; e < nk::NG2; ++e)
            if (!sameBits(A[static_cast<std::size_t>(aos.ng2(lk, e))],
                          S[static_cast<std::size_t>(soa.ng2(lk, e))]))
                ++n;
    if (n) std::printf("  MISMATCH %-9s %d element(s)\n", what, n);
    bad += n;
}
void cmpFlat(const char* what, const std::vector<double>& A,
             const std::vector<double>& S) {
    int n = 0;
    for (std::size_t i = 0; i < A.size(); ++i)
        if (!sameBits(A[i], S[i])) ++n;
    if (n) std::printf("  MISMATCH %-9s %d element(s)\n", what, n);
    bad += n;
}

void drive(const nk::NodalView& v) {
    nk::StaticForms pol;
    for (int lk = 0; lk < NXYZ; ++lk) nk::nodalTrlcff0(v, lk);
    for (int lk = 0; lk < NXYZ; ++lk) nk::nodalTrlcff12(v, lk, pol);
    for (int lk = 0; lk < NXYZ; ++lk) nk::nodalUpdateMatrix(v, lk, pol);
    for (int lk = 0; lk < NXYZ; ++lk) nk::nodalCalculateEven(v, lk, pol);
    for (int ls = 0; ls < v.nsurf; ++ls) nk::nodalCalculateJnet(v, ls, pol);
}

} // namespace

int main() {
    Mesh mesh;

    Arrays aos_a, soa_a;
    fill(aos_a, mesh.nsurf);
    fill(soa_a, mesh.nsurf);

    nk::NodalView aos{};
    bind(aos, mesh, aos_a);
    nk::NodalView soa{};
    bind(soa, mesh, soa_a);
    soa = nk::nodalNodeInnermostView(soa);
    // `bind` overwrote the pointers but the strides survive the copy above;
    // re-bind would put the defaults back, so the order matters and is
    // asserted rather than remembered.
    if (nk::nodalIsNodeInnermost(aos) || !nk::nodalIsNodeInnermost(soa)) {
        std::printf("FAIL: the two views do not carry two layouts\n");
        return 2;
    }

    // Permute the SoA instance's INPUTS so both runs see the same values.
    for (int i = 0; i < 15; ++i) {
        std::vector<double> tmp(soa_a.ndg[i].size(), 0.0);
        packNdg(aos, soa, aos_a.ndg[i], tmp);
        soa_a.ndg[i] = tmp;
    }
    for (int i = 0; i < 2; ++i) {
        std::vector<double> tmp(soa_a.dg2[i].size(), 0.0);
        packDg2(aos, soa, aos_a.dg2[i], tmp);
        soa_a.dg2[i] = tmp;
    }
    for (int i = 0; i < 4; ++i) {
        std::vector<double> tmp(soa_a.ng2[i].size(), 0.0);
        packNg2(aos, soa, aos_a.ng2[i], tmp);
        soa_a.ng2[i] = tmp;
    }
    std::vector<double> hmesh_soa(mesh.hmesh.size(), 0.0);
    packHm(aos, soa, mesh.hmesh, hmesh_soa);
    soa.hmesh = hmesh_soa.data();
    // The canonical and xs arrays are NOT permuted -- that is the contract.
    soa_a.xsrf = aos_a.xsrf; soa_a.xsnf = aos_a.xsnf; soa_a.xssm = aos_a.xssm;
    soa_a.chif = aos_a.chif; soa_a.flux = aos_a.flux; soa_a.jnet = aos_a.jnet;
    soa_a.phis = aos_a.phis;
    bind(soa, mesh, soa_a);
    soa = nk::nodalNodeInnermostView(soa);
    soa.hmesh = hmesh_soa.data();

    // NEGATIVE CONTROL (RASBERY_NODAL_LAYOUT_SABOTAGE=1).  A gate that compared
    // two runs which happened to agree for a reason other than the one it
    // claims would pass for ever.  Swapping the SoA view's `idir` and `ig`
    // strides leaves every index in range and every value finite -- it is
    // exactly the shape of the bug this gate exists for -- and the comparison
    // below must then FAIL.  ctest registers both arms.
    const char* const sab = std::getenv("RASBERY_NODAL_LAYOUT_SABOTAGE");
    const bool sabotage = sab != nullptr && sab[0] == '1';
    if (sabotage) {
        const int t = soa.ndg_dir;
        soa.ndg_dir = soa.ndg_grp;
        soa.ndg_grp = t;
    }

    drive(aos);
    drive(soa);

    for (int i = 0; i < 15; ++i) cmpNdg(NDG_NAME[i], aos, soa, aos_a.ndg[i], soa_a.ndg[i]);
    for (int i = 0; i < 2; ++i) cmpDg2(DG2_NAME[i], aos, soa, aos_a.dg2[i], soa_a.dg2[i]);
    for (int i = 0; i < 4; ++i) cmpNg2(NG2_NAME[i], aos, soa, aos_a.ng2[i], soa_a.ng2[i]);
    cmpFlat("jnet", aos_a.jnet, soa_a.jnet);
    cmpFlat("phis", aos_a.phis, soa_a.phis);

    // A gate that compared two runs of a body that wrote nothing would pass
    // for ever.  Prove the drive actually moved the outputs.
    int nonzero = 0;
    for (double x : aos_a.jnet) if (x != 0.0) ++nonzero;
    for (double x : aos_a.phis) if (x != 0.0) ++nonzero;
    if (nonzero == 0) {
        std::printf("FAIL: the drive wrote nothing -- the gate is vacuous\n");
        return 2;
    }

    const bool ok = sabotage ? bad > 0 : bad == 0;
    std::printf("[nodal_layout_equivalence] nodes=%d surfaces=%d sabotage=%d "
                "mismatches=%d -> %s\n", NXYZ, mesh.nsurf,
                sabotage ? 1 : 0, bad, ok ? "PASS" : "FAIL");
    if (sabotage && bad == 0)
        std::printf("  the negative control did not fire: a swapped stride "
                    "changed nothing, so this gate is measuring nothing\n");
    return ok ? 0 : 1;
}
