#pragma once

// Shared host/device body of the per-outer SENM nodal phases:
//   caltrlcff0 -> caltrlcff12 -> updateMatrix -> calculateEven -> calculateJnet
// (Nodal.cpp), one function per phase, one thread per node (or surface).
//
// updateConstant is NOT here by design: it is the only transcendental site
// (exp/sqrt -- glibc exp is not correctly rounded and does not match CUDA)
// and it already recomputes only when xsrf/xsdf change, so it stays on the
// host and its nine coefficient arrays ride a generation counter.  Same
// resolve/apply split as FlatXsKernel.h.
//
// Determinism contract (same rules as XsReconKernel.h / FlatXsKernel.h):
//  - statement-for-statement transcription of Nodal.cpp, same evaluation
//    order, no reassociation;
//  - every a*b+c-shaped site goes through the policy's ma(); two-product
//    sums a*b (+|-) c*d follow the CMFD form finding (gcc evaluates the
//    SECOND product first, rounded, and fuses the FIRST into an fma), spelled
//    out below via ma(bit, a, b, +-xsrMul(c, d));
//  - the production mask is mined from a RASBERY_NODAL_DUMP capture by
//    test/nodal_replay.cpp, per phase.
//
// Must stay compilable by g++ and nvcc: no STL, no allocation.

#include "XsReconKernel.h" // xsrFma / xsrMul / RASBERY_XSR_HD

namespace rasbery::nodal {

using xsrecon::xsrFma;
using xsrecon::xsrMul;

constexpr int NG    = 2;
constexpr int NG2   = 4;
constexpr int NDIR  = 3; // NDIRMAX
constexpr int NLR   = 2; // LR
constexpr int NEWSB = 6; // NEWSBT (dir*LR+side stride of Geometry::_neib)
constexpr int C_LEFT = 0, C_RIGHT = 1, C_CENTER = 2;
constexpr int C_XDIR = 0, C_YDIR = 1, C_ZDIR = 2;

// Verbatim from Nodal.h's #defines.
constexpr double M011  = 0.666666667;
constexpr double M022  = 0.4;
constexpr double M033  = 0.285714286;
constexpr double M044  = 0.222222222;
constexpr double M220  = 6.;
constexpr double RM220 = 0.166666667;
constexpr double M240  = 20.;
constexpr double M231  = 10.;
constexpr double M242  = 14.;

/// Per-phase contraction masks.  Bit=1 means the site is a single-rounding
/// fma.  Phases: 0=trlcff12, 1=updateMatrix, 2=calculateEven, 3=jnet1n,
/// 4=jnet2n.  (caltrlcff0 has no multiply-add sites: products and divides
/// only.)  Production values mined by test/nodal_replay.cpp --sweep.
/// Mined masks (test/nodal_replay.cpp --sweep against the APR1400 profile
/// capture).  1-bit sites: 1 = single-rounding fma.  2-bit (ma2) sites over
/// A*B + C*D: 0 = both products rounded; 1 = fma(A,B, round(C*D));
/// 2 = fma(C,D, round(A*B)).  trlcff12 and updateMatrix mined exact on the
/// first pass (all their sites fused).
constexpr unsigned long long NODAL_FORMS[5] = {
    0x3Full, 0xFull, 0x355ADull, 0x5DD445Bull, 0xFBAAB56F79ull};

/// Device code cannot address the namespace-scope array's storage (same
/// finding as xsrecon's ACTIVE_XT); this constexpr function is the mask
/// source both compilers can fold.  Keep it identical to NODAL_FORMS.
RASBERY_XSR_HD constexpr unsigned long long nodalFormsOf(int phase) {
    return phase == 0   ? 0x3Full
           : phase == 1 ? 0xFull
           : phase == 2 ? 0x355ADull
           : phase == 3 ? 0x5DD445Bull
                        : 0xFBAAB56F79ull;
}

RASBERY_XSR_HD inline double nodalMa1(unsigned long long m, int bit, double a,
                                      double b, double c) {
#if defined(__CUDA_ARCH__)
    return ((m >> bit) & 1ull) ? fma(a, b, c) : a * b + c;
#else
    return ((m >> bit) & 1ull) ? xsrFma(a, b, c) : xsrMul(a, b) + c;
#endif
}

RASBERY_XSR_HD inline double nodalMa2(unsigned long long m, int bit, double a,
                                      double b, double c, double d) {
    const unsigned st = static_cast<unsigned>((m >> bit) & 3ull);
#if defined(__CUDA_ARCH__)
    if (st == 1) return fma(a, b, c * d);
    if (st == 2) return fma(c, d, a * b);
    return a * b + c * d;
#else
    if (st == 1) return xsrFma(a, b, xsrMul(c, d));
    if (st == 2) return xsrFma(c, d, xsrMul(a, b));
    return xsrMul(a, b) + xsrMul(c, d);
#endif
}

struct StaticForms {
    RASBERY_XSR_HD double ma(int phase, int bit, double a, double b, double c) const {
        return nodalMa1(nodalFormsOf(phase), bit, a, b, c);
    }
    RASBERY_XSR_HD double ma2(int phase, int bit, double a, double b, double c,
                              double d) const {
        return nodalMa2(nodalFormsOf(phase), bit, a, b, c, d);
    }
};

/// Runtime-mask policy for the sweep (host-only test tool).
struct RuntimeForms {
    unsigned long long mask[5] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
    double ma(int phase, int bit, double a, double b, double c) const {
        return nodalMa1(mask[phase], bit, a, b, c);
    }
    double ma2(int phase, int bit, double a, double b, double c, double d) const {
        return nodalMa2(mask[phase], bit, a, b, c, d);
    }
};

/// Pointer view of one instance's nodal state.  Indexing mirrors Nodal.cpp's
/// macros exactly:
///   per-(node,dir,group):  arr[(lk*NDIR + idir)*NG + ig]
///   per-(node,dir) matrix: arr[(lk*NDIR + idir)*NG2 + j*NG + i]
///   per-node matrix:       arr[lk*NG2 + j*NG + i]
///   flux:                  flux[lk*NG + ig]
///   jnet/phis:             arr[ls*NG + ig]
///   xs arrays:             arr[ig*nxyz + lk]  (SoA, shared with the xs arms)
///   xssm:                  arr[(igs*NG + ige)*nxyz + lk]
struct NodalView {
    // geometry (immutable; device copy uploaded once)
    const double* hmesh;   // [lk*NDIR + dir]
    const int*    lktosfc; // [(lk*NDIR + dir)*NLR + side]
    const int*    neib;    // [lk*NEWSB + dir*NLR + side]
    const int*    lklr;    // [ls*NLR + side]
    const int*    idirlr;  // [ls*NLR + side]
    const int*    sgnlr;   // [ls*NLR + side]
    const double* albedo;  // [dir*NLR + side]

    // xs inputs (device-resident via the shared backend block)
    const double* xsrf; // [ig*nxyz + lk]
    const double* xsnf;
    const double* xssm; // [(igs*NG+ige)*nxyz + lk]
    const double* chif; // [ig*nxyz + lk]; ignored when chif_empty
    int           chif_empty;

    // updateConstant products (host-computed, uploaded on generation change)
    const double* eta1;
    const double* eta2;
    const double* m260;
    const double* m251;
    const double* m253;
    const double* m262;
    const double* m264;
    const double* diagD;
    const double* diagDI;

    // device-resident working arrays
    double* trlcff0;
    double* trlcff1;
    double* trlcff2;
    double* mu;
    double* tau;
    double* matM;
    double* matMI;
    double* matMs;
    double* matMf;
    double* dsncff2;
    double* dsncff4;
    double* dsncff6;

    // per-call inputs/outputs
    const double* flux; // phif
    double*       jnet;
    double*       phis;
    double        reigv;
    int           nxyz;
    int           nsurf;
};

// Accessor helpers mirroring the Nodal.cpp macros.
RASBERY_XSR_HD inline double nvTrl0(const NodalView& v, int ig, int lkd) { return v.trlcff0[lkd * NG + ig]; }
RASBERY_XSR_HD inline double nvMatM(const NodalView& v, int i, int j, int lk) { return v.matM[lk * NG2 + j * NG + i]; }
RASBERY_XSR_HD inline double nvMatMI(const NodalView& v, int i, int j, int lk) { return v.matMI[lk * NG2 + j * NG + i]; }
RASBERY_XSR_HD inline double nvMu(const NodalView& v, int i, int j, int lkd) { return v.mu[lkd * NG2 + j * NG + i]; }
RASBERY_XSR_HD inline double nvTau(const NodalView& v, int i, int j, int lkd) { return v.tau[lkd * NG2 + j * NG + i]; }

// ---------------------------------------------------------------------------
// Phase caltrlcff0 (per node): products, divides and adds only -- no
// contraction sites, so no policy involvement.
// ---------------------------------------------------------------------------
RASBERY_XSR_HD inline void nodalTrlcff0(const NodalView& v, int lk) {
    const int lkd0 = lk * NDIR;
    double    avgjnet[NDIR];

    for (int ig = 0; ig < NG; ig++) {
        for (int idir = 0; idir < NDIR; idir++) {
            const int lsl = v.lktosfc[(lk * NDIR + idir) * NLR + C_LEFT];
            const int lsr = v.lktosfc[(lk * NDIR + idir) * NLR + C_RIGHT];
            avgjnet[idir] =
                (v.jnet[lsr * NG + ig] - v.jnet[lsl * NG + ig]) / v.hmesh[lk * NDIR + idir];
        }
        v.trlcff0[(lkd0 + C_XDIR) * NG + ig] = avgjnet[C_YDIR] + avgjnet[C_ZDIR];
        v.trlcff0[(lkd0 + C_YDIR) * NG + ig] = avgjnet[C_XDIR] + avgjnet[C_ZDIR];
        v.trlcff0[(lkd0 + C_ZDIR) * NG + ig] = avgjnet[C_XDIR] + avgjnet[C_YDIR];
    }
}

// ---------------------------------------------------------------------------
// Phase caltrlcff12 (per node).  trlcffbyintg inlined verbatim.
// Sites (phase 0): 0..3 = the four rh-product accumulations of the interior
// branch (each `x*y + z*w`-shaped via the second-product-rounded form).
// ---------------------------------------------------------------------------
template <class POL>
RASBERY_XSR_HD inline void nodalTrlcff12(const NodalView& v, int lk, const POL& pol) {
    const int lkd0 = lk * NDIR;

    for (int idir = 0; idir < NDIR; idir++) {
        const int lkd = lkd0 + idir;

        const int lkl = v.neib[lk * NEWSB + idir * NLR + C_LEFT];
        const int lkr = v.neib[lk * NEWSB + idir * NLR + C_RIGHT];

        for (int ig = 0; ig < NG; ig++) {
            double avgtrl3[3] = {};
            double hmesh3[3]  = {};
            hmesh3[C_CENTER]  = v.hmesh[lk * NDIR + idir];
            avgtrl3[C_CENTER] = v.trlcff0[lkd * NG + ig];

            if (lkl > -1) {
                const int lsl    = v.lktosfc[(lk * NDIR + idir) * NLR + C_LEFT];
                const int idirl  = v.idirlr[lsl * NLR + C_LEFT];
                hmesh3[C_LEFT]   = v.hmesh[lkl * NDIR + idirl];
                avgtrl3[C_LEFT]  = v.trlcff0[(lkl * NDIR + idirl) * NG + ig];
            } else if (v.albedo[idir * NLR + C_LEFT] < 1.0E-10) {
                hmesh3[C_LEFT]  = hmesh3[C_CENTER];
                avgtrl3[C_LEFT] = avgtrl3[C_CENTER];
            }

            if (lkr > -1) {
                const int lsr    = v.lktosfc[(lk * NDIR + idir) * NLR + C_RIGHT];
                const int idirr  = v.idirlr[lsr * NLR + C_RIGHT];
                hmesh3[C_RIGHT]  = v.hmesh[lkr * NDIR + idirr];
                avgtrl3[C_RIGHT] = v.trlcff0[(lkr * NDIR + idirr) * NG + ig];
            } else if (v.albedo[idir * NLR + C_RIGHT] < 1.0E-10) {
                hmesh3[C_RIGHT]  = hmesh3[C_CENTER];
                avgtrl3[C_RIGHT] = avgtrl3[C_CENTER];
            }

            // --- trlcffbyintg, inlined verbatim ---
            double& out1 = v.trlcff1[lkd * NG + ig];
            double& out2 = v.trlcff2[lkd * NG + ig];
            double  sh[4];
            const double rh =
                (1 / ((hmesh3[C_LEFT] + hmesh3[C_CENTER] + hmesh3[C_RIGHT]) *
                      (hmesh3[C_LEFT] + hmesh3[C_CENTER]) *
                      (hmesh3[C_CENTER] + hmesh3[C_RIGHT])));
            sh[0] = (2 * hmesh3[C_LEFT] + hmesh3[C_CENTER]) *
                    (hmesh3[C_LEFT] + hmesh3[C_CENTER]);
            sh[1] = hmesh3[C_LEFT] + hmesh3[C_CENTER];
            sh[2] = (hmesh3[C_CENTER] + 2 * hmesh3[C_RIGHT]) *
                    (hmesh3[C_CENTER] + hmesh3[C_RIGHT]);
            sh[3] = hmesh3[C_CENTER] + hmesh3[C_RIGHT];

            if (hmesh3[C_LEFT] == 0.0) {
                out1 = 0.125 * pol.ma(0, 2, 5., avgtrl3[C_CENTER], avgtrl3[C_RIGHT]);
                out2 = 0.125 * pol.ma(0, 3, -3., avgtrl3[C_CENTER], avgtrl3[C_RIGHT]);
            } else if (hmesh3[C_RIGHT] == 0.0) {
                out1 = -0.125 * pol.ma(0, 4, 5., avgtrl3[C_CENTER], avgtrl3[C_LEFT]);
                out2 = 0.125 * pol.ma(0, 5, -3., avgtrl3[C_CENTER], avgtrl3[C_LEFT]);
            } else {
                // (a-b)*s2 + (c-a)*s0, second product rounded first per the
                // CMFD form finding; sites 0/1.
                out1 = 0.5 * rh * hmesh3[C_CENTER] *
                       pol.ma(0, 0, (avgtrl3[C_CENTER] - avgtrl3[C_LEFT]), sh[2],
                              xsrMul((avgtrl3[C_RIGHT] - avgtrl3[C_CENTER]), sh[0]));
                out2 = 0.5 * rh * (hmesh3[C_CENTER] * hmesh3[C_CENTER]) *
                       pol.ma(0, 1, (avgtrl3[C_LEFT] - avgtrl3[C_CENTER]), sh[3],
                              xsrMul((avgtrl3[C_RIGHT] - avgtrl3[C_CENTER]), sh[1]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase updateMatrix (per node).  Sites (phase 1):
//  0: matM = matMs - reigv*matMf         (fnma-shaped: -reigv*f + s)
//  1: det  = M00*M11 - M10*M01
//  2: tempz accumulate  += m251*tau
//  3: mu row contraction MI0*t0 + MI1*t1
// ---------------------------------------------------------------------------
template <class POL>
RASBERY_XSR_HD inline void nodalUpdateMatrix(const NodalView& v, int lk, const POL& pol) {
    const int lkd0 = lk * NDIR;
    const int nxyz = v.nxyz;

    for (int ige = 0; ige < NG; ige++) {
        for (int igs = 0; igs < NG; igs++) {
            v.matMs[lk * NG2 + ige * NG + igs] = -v.xssm[(igs * NG + ige) * nxyz + lk];
            const double chif_v = v.chif_empty ? (ige == 0 ? 1.0 : 0.0)
                                               : v.chif[ige * nxyz + lk];
            v.matMf[lk * NG2 + ige * NG + igs] = chif_v * v.xsnf[igs * nxyz + lk];
        }
        v.matMs[lk * NG2 + ige * NG + ige] += v.xsrf[ige * nxyz + lk];

        for (int igs = 0; igs < NG; igs++) {
            v.matM[lk * NG2 + ige * NG + igs] =
                pol.ma(1, 0, -v.reigv, v.matMf[lk * NG2 + ige * NG + igs],
                       v.matMs[lk * NG2 + ige * NG + igs]);
        }
    }

    const double det =
        pol.ma(1, 1, nvMatM(v, 0, 0, lk), nvMatM(v, 1, 1, lk),
               -xsrMul(nvMatM(v, 1, 0, lk), nvMatM(v, 0, 1, lk)));

    if (fabs(det) > 1.E-10) {
        const double rdet = 1 / det;
        v.matMI[lk * NG2 + 0 * NG + 0] = rdet * nvMatM(v, 1, 1, lk);
        v.matMI[lk * NG2 + 0 * NG + 1] = -rdet * nvMatM(v, 1, 0, lk);
        v.matMI[lk * NG2 + 1 * NG + 0] = -rdet * nvMatM(v, 0, 1, lk);
        v.matMI[lk * NG2 + 1 * NG + 1] = rdet * nvMatM(v, 0, 0, lk);
    } else {
        v.matMI[lk * NG2 + 0 * NG + 0] = 0;
        v.matMI[lk * NG2 + 0 * NG + 1] = 0;
        v.matMI[lk * NG2 + 1 * NG + 0] = 0;
        v.matMI[lk * NG2 + 1 * NG + 1] = 0;
    }

    const double rm011 = 1. / M011;

    for (int idir = 0; idir < NDIR; idir++) {
        const int lkd = lkd0 + idir;

        double tempz[2][2] = {};

        for (int igd = 0; igd < NG; igd++) {
            const double tau1 =
                M033 * (v.diagDI[lkd * NG + igd] / v.m253[lkd * NG + igd]);

            tempz[igd][igd] = tempz[igd][igd] + M231;

            for (int igs = 0; igs < NG; igs++) {
                v.tau[lkd * NG2 + igd * NG + igs] =
                    tau1 * nvMatM(v, igs, igd, lk);

                tempz[igs][igd] =
                    pol.ma(1, 2, v.m251[lkd * NG + igd],
                           v.tau[lkd * NG2 + igd * NG + igs], tempz[igs][igd]);

                tempz[igs][igd] *= v.diagD[lkd * NG + igd];
            }
        }

        v.mu[lkd * NG2 + 0 * NG + 0] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 0, lk), tempz[0][0],
                           xsrMul(nvMatMI(v, 1, 0, lk), tempz[0][1]));
        v.mu[lkd * NG2 + 0 * NG + 1] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 0, lk), tempz[1][0],
                           xsrMul(nvMatMI(v, 1, 0, lk), tempz[1][1]));
        v.mu[lkd * NG2 + 1 * NG + 0] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 1, lk), tempz[0][0],
                           xsrMul(nvMatMI(v, 1, 1, lk), tempz[0][1]));
        v.mu[lkd * NG2 + 1 * NG + 1] =
            rm011 * pol.ma(1, 3, nvMatMI(v, 0, 1, lk), tempz[1][0],
                           xsrMul(nvMatMI(v, 1, 1, lk), tempz[1][1]));
    }
}

// ---------------------------------------------------------------------------
// Phase calculateEven (per node).  Sites (phase 2):
//  0: at2 = m022*rm220*mu2*matM       (pure products -- no site; kept literal)
//  0: a = mu1*matM + M0*at2_0 + M1*at2_1   (3-term chain: two ma sites 0,1)
//  2: bt2 inner  M0*flux0 + M1*flux1 (+trl0)  -- sites 2,3
//  4: b = m022*trl2 + (M0*bt1_0 + M1*bt1_1)  -- sites 4,5
//  6: rdet = a00*a11 - a10*a01
//  7: dsncff4 row solves a11*b0 - a10*b1
//  8: dsncff6 contraction M0*c4_0 + M1*c4_1
//  9: dsncff2 tail  -m240*c4 - m260*c6 (+DI*bt2)  -- sites 9,10
// ---------------------------------------------------------------------------
template <class POL>
RASBERY_XSR_HD inline void nodalCalculateEven(const NodalView& v, int lk, const POL& pol) {
    const int lkd0 = lk * NDIR;

    for (int idir = 0; idir < NDIR; idir++) {
        const int lkd = lkd0 + idir;
        double    at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2], b[2];

        for (int igd = 0; igd < NG; igd++) {
            rm4464[igd] = 0.0;
            if (fabs(v.m264[lkd * NG + igd]) > 1.0E-10)
                rm4464[igd] = M044 / v.m264[lkd * NG + igd];

            const double mu2 =
                rm4464[igd] * v.m260[lkd * NG + igd] * v.diagDI[lkd * NG + igd];

            for (int igs = 0; igs < NG; igs++) {
                at2[igs][igd] = M022 * RM220 * mu2 * nvMatM(v, igs, igd, lk);
            }
            at2[igd][igd] += M022 * RM220 * M240;
        }

        for (int igd = 0; igd < NG; igd++) {
            const double mu1 = rm4464[igd] * v.m262[lkd * NG + igd];
            for (int igs = 0; igs < NG; igs++) {
                // ((mu1*matM + matM0*at2_0) + matM1*at2_1)
                const double t1 =
                    pol.ma2(2, 0, mu1, nvMatM(v, igs, igd, lk),
                            nvMatM(v, 0, igd, lk), at2[igs][0]);
                a[igs][igd] =
                    pol.ma(2, 2, nvMatM(v, 1, igd, lk), at2[igs][1], t1);
            }
            a[igd][igd] =
                pol.ma(2, 17, v.diagD[lkd * NG + igd], M242, a[igd][igd]);
            bt2[igd] =
                2 * (pol.ma2(2, 3, nvMatM(v, 0, igd, lk), v.flux[lk * NG + 0],
                             nvMatM(v, 1, igd, lk), v.flux[lk * NG + 1]) +
                     v.trlcff0[lkd * NG + igd]);
            bt1[igd] = M022 * RM220 * v.diagDI[lkd * NG + igd] * bt2[igd];
        }

        for (int ig = 0; ig < NG; ig++) {
            const double inner =
                pol.ma2(2, 5, nvMatM(v, 0, ig, lk), bt1[0],
                        nvMatM(v, 1, ig, lk), bt1[1]);
            b[ig] = pol.ma(2, 7, M022, v.trlcff2[lkd * NG + ig], inner);
        }

        double rdet = pol.ma2(2, 8, a[0][0], a[1][1], -a[1][0], a[0][1]);

        if (rdet != 0.0) {
            rdet = 1. / rdet;
            v.dsncff4[lkd * NG + 0] =
                rdet * pol.ma2(2, 10, a[1][1], b[0], -a[1][0], b[1]);
            v.dsncff4[lkd * NG + 1] =
                rdet * pol.ma2(2, 10, a[0][0], b[1], -a[0][1], b[0]);
        } else {
            v.dsncff4[lkd * NG + 0] = 0.0;
            v.dsncff4[lkd * NG + 1] = 0.0;
        }

        for (int ig = 0; ig < NG; ig++) {
            v.dsncff6[lkd * NG + ig] =
                v.diagDI[lkd * NG + ig] * rm4464[ig] *
                pol.ma2(2, 12, nvMatM(v, 0, ig, lk), v.dsncff4[lkd * NG + 0],
                        nvMatM(v, 1, ig, lk), v.dsncff4[lkd * NG + 1]);
            const double t2 =
                pol.ma2(2, 14, v.diagDI[lkd * NG + ig], bt2[ig], -M240,
                        v.dsncff4[lkd * NG + ig]);
            v.dsncff2[lkd * NG + ig] =
                RM220 * pol.ma(2, 16, -v.m260[lkd * NG + ig],
                               v.dsncff6[lkd * NG + ig], t2);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase calculateJnet (per surface): 1n boundary or 2n interior.
// jnet1n sites (phase 3), jnet2n sites (phase 4) -- enumerated inline.
// ---------------------------------------------------------------------------
template <class POL>
RASBERY_XSR_HD inline void nodalJnet1n(const NodalView& v, int ls, int lr,
                                       double alb, const POL& pol) {
    const int lk   = v.lklr[ls * NLR + lr];
    const int idir = v.idirlr[ls * NLR + lr];
    const int lkd  = lk * NDIR + idir;
    int       sgn  = 1;
    if (lr == C_RIGHT)
        sgn = -1;

    double diagDj[2]{};
    double a11[2][2], a12[2], a13[2], a22[2][2], a23[2], a31[2], a32[2], a33[2];
    double b1[2], b2[2];

    for (int ige = 0; ige < NG; ige++)
        for (int igs = 0; igs < NG; igs++)
            a11[igs][ige] = nvMatM(v, igs, ige, lk) * M011;

    for (int ig = 0; ig < NG; ig++)
        a12[ig] = -v.diagD[lkd * NG + ig] * M231;

    for (int ig = 0; ig < NG; ig++)
        a13[ig] = -v.diagD[lkd * NG + ig] * v.m251[lkd * NG + ig];

    for (int ige = 0; ige < NG; ige++)
        for (int igs = 0; igs < NG; igs++)
            a22[igs][ige] = nvMatM(v, igs, ige, lk) * M033;

    for (int ig = 0; ig < NG; ig++)
        a23[ig] = -v.diagD[lkd * NG + ig] * v.m253[lkd * NG + ig];

    for (int ig = 0; ig < NG; ig++)
        diagDj[ig] = 0.5 * v.hmesh[lk * NDIR + idir] * v.diagD[lkd * NG + ig];

    for (int ig = 0; ig < NG; ig++)
        a31[ig] = diagDj[ig] + alb;
    for (int ig = 0; ig < NG; ig++)
        a32[ig] = pol.ma(3, 26, 6., diagDj[ig], alb);
    for (int ig = 0; ig < NG; ig++)
        a33[ig] = pol.ma(3, 0, diagDj[ig], v.eta1[lkd * NG + ig], alb);

    for (int ig = 0; ig < NG; ig++)
        b1[ig] = -M011 * v.trlcff1[lkd * NG + ig];
    for (int ig = 0; ig < NG; ig++) {
        const double t1 = pol.ma(
            3, 3, v.eta2[lkd * NG + ig], v.dsncff6[lkd * NG + ig],
            pol.ma2(3, 1, 3, v.dsncff2[lkd * NG + ig], 10,
                    v.dsncff4[lkd * NG + ig]));
        const double t2 = v.flux[lk * NG + ig] + v.dsncff2[lkd * NG + ig] +
                          v.dsncff4[lkd * NG + ig] + v.dsncff6[lkd * NG + ig];
        b2[ig] = -sgn * pol.ma2(3, 4, diagDj[ig], t1, alb, t2);
    }

    for (int ige = 0; ige < NG; ige++) {
        for (int igs = 0; igs < NG; igs++) {
            a22[igs][ige] = -a22[igs][ige] / a23[ige];
            a11[igs][ige] = a11[igs][ige] / a31[igs];
        }
    }

    double a[2][2] = {};
    for (int ige = 0; ige < NG; ige++)
        for (int igs = 0; igs < NG; igs++)
            a[igs][ige] = pol.ma2(3, 6, a13[ige], a22[igs][ige],
                                  -a11[igs][ige], a32[igs]);

    for (int ige = 0; ige < NG; ige++)
        a[ige][ige] = a[ige][ige] + a12[ige];

    for (int ige = 0; ige < NG; ige++) {
        for (int igs = 0; igs < NG; igs++) {
            const double S =
                pol.ma2(3, 8, xsrMul(a11[0][ige], a33[0]), a22[igs][0],
                        xsrMul(a11[1][ige], a33[1]), a22[igs][1]);
            a[igs][ige] = a[igs][ige] - S;
        }
        b1[ige] =
            b1[ige] - pol.ma2(3, 10, a11[0][ige], b2[0], a11[1][ige], b2[1]);
    }

    double oddcff[3][2];

    const double det1 = pol.ma2(3, 12, a[0][0], a[1][1], -a[1][0], a[0][1]);
    const double rdet = 1.0 / det1;
    a11[0][0] = rdet * a[1][1];
    a11[1][0] = -rdet * a[1][0];
    a11[0][1] = -rdet * a[0][1];
    a11[1][1] = rdet * a[0][0];

    for (int ig = 0; ig < NG; ig++)
        oddcff[1][ig] = pol.ma2(3, 14, a11[0][ig], b1[0], a11[1][ig], b1[1]);

    for (int ig = 0; ig < NG; ig++)
        oddcff[2][ig] =
            pol.ma2(3, 16, a22[0][ig], oddcff[1][0], a22[1][ig], oddcff[1][1]);

    for (int ig = 0; ig < NG; ig++)
        oddcff[0][ig] =
            pol.ma(3, 19, -a33[ig], oddcff[2][ig],
                   pol.ma(3, 18, -a32[ig], oddcff[1][ig], b2[ig])) /
            a31[ig];

    for (int ig = 0; ig < NG; ig++) {
        const double even_t = pol.ma(
            3, 22, v.eta2[lkd * NG + ig], v.dsncff6[lkd * NG + ig],
            pol.ma2(3, 20, 3, v.dsncff2[lkd * NG + ig], 10,
                    v.dsncff4[lkd * NG + ig]));
        const double odd_t = pol.ma(
            3, 24, v.eta1[lkd * NG + ig], oddcff[2][ig],
            pol.ma(3, 23, 6, oddcff[1][ig], oddcff[0][ig]));
        v.jnet[ls * NG + ig] = -v.hmesh[lk * NDIR + idir] * 0.5 *
                               v.diagD[lkd * NG + ig] *
                               pol.ma(3, 25, static_cast<double>(sgn), even_t, odd_t);

        v.phis[ls * NG + ig] =
            v.flux[lk * NG + ig] + v.dsncff2[lkd * NG + ig] +
            v.dsncff4[lkd * NG + ig] + v.dsncff6[lkd * NG + ig] +
            sgn * (oddcff[0][ig] + oddcff[1][ig] + oddcff[2][ig]);
    }
}

template <class POL>
RASBERY_XSR_HD inline void nodalJnet2n(const NodalView& v, int ls, const POL& pol) {
    const int lkl   = v.lklr[ls * NLR + C_LEFT];
    const int lkr   = v.lklr[ls * NLR + C_RIGHT];
    const int idirl = v.idirlr[ls * NLR + C_LEFT];
    const int idirr = v.idirlr[ls * NLR + C_RIGHT];
    const int sgnl  = v.sgnlr[ls * NLR + C_LEFT];
    const int sgnr  = v.sgnlr[ls * NLR + C_RIGHT];
    const int lkdl  = lkl * NDIR + idirl;
    const int lkdr  = lkr * NDIR + idirr;

    double diagDj[2][2], tempz[2][2], tempzI[2][2], zeta1[2][2], zeta2[2],
        bfc[2], mat1g[2][2];

    for (int ig = 0; ig < NG; ig++) {
        diagDj[ig][C_LEFT] =
            0.5 * v.hmesh[lkl * NDIR + idirl] * v.diagD[lkdl * NG + ig];
        diagDj[ig][C_RIGHT] =
            0.5 * v.hmesh[lkr * NDIR + idirr] * v.diagD[lkdr * NG + ig];
    }

    tempz[0][0] = nvMu(v, 0, 0, lkdr) + nvTau(v, 0, 0, lkdr) + 1;
    tempz[1][0] = nvMu(v, 1, 0, lkdr) + nvTau(v, 1, 0, lkdr);
    tempz[0][1] = nvMu(v, 0, 1, lkdr) + nvTau(v, 0, 1, lkdr);
    tempz[1][1] = nvMu(v, 1, 1, lkdr) + nvTau(v, 1, 1, lkdr) + 1;

    double rdet =
        1 / pol.ma2(4, 0, tempz[0][0], tempz[1][1], -tempz[1][0], tempz[0][1]);
    tempzI[0][0] = rdet * tempz[1][1];
    tempzI[1][0] = -rdet * tempz[1][0];
    tempzI[0][1] = -rdet * tempz[0][1];
    tempzI[1][1] = rdet * tempz[0][0];

    tempz[0][0] = nvMu(v, 0, 0, lkdl) + nvTau(v, 0, 0, lkdl) + 1;
    tempz[1][0] = nvMu(v, 1, 0, lkdl) + nvTau(v, 1, 0, lkdl);
    tempz[0][1] = nvMu(v, 0, 1, lkdl) + nvTau(v, 0, 1, lkdl);
    tempz[1][1] = nvMu(v, 1, 1, lkdl) + nvTau(v, 1, 1, lkdl) + 1;

    zeta1[0][0] = pol.ma2(4, 2, tempzI[0][0], tempz[0][0], tempzI[1][0], tempz[0][1]);
    zeta1[1][0] = pol.ma2(4, 2, tempzI[0][0], tempz[1][0], tempzI[1][0], tempz[1][1]);
    zeta1[0][1] = pol.ma2(4, 2, tempzI[0][1], tempz[0][0], tempzI[1][1], tempz[0][1]);
    zeta1[1][1] = pol.ma2(4, 2, tempzI[0][1], tempz[1][0], tempzI[1][1], tempz[1][1]);

    for (int ig = 0; ig < NG; ig++) {
        const double right_part = pol.ma(
            4, 5, nvMatMI(v, 1, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 1],
            pol.ma(4, 4, nvMatMI(v, 0, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 0],
                   v.dsncff2[lkdr * NG + ig] + v.dsncff4[lkdr * NG + ig] +
                       v.dsncff6[lkdr * NG + ig] + v.flux[lkr * NG + ig]));
        const double left_part = pol.ma(
            4, 5, nvMatMI(v, 1, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 1],
            pol.ma(4, 4, nvMatMI(v, 0, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 0],
                   -v.dsncff2[lkdl * NG + ig] - v.dsncff4[lkdl * NG + ig] -
                       v.dsncff6[lkdl * NG + ig] - v.flux[lkl * NG + ig]));
        bfc[ig] = right_part + left_part;
    }

    for (int ig = 0; ig < NG; ig++)
        zeta2[ig] = pol.ma2(4, 6, tempzI[0][ig], bfc[0], tempzI[1][ig], bfc[1]);

    tempz[0][0] = diagDj[0][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 0], nvTau(v, 0, 0, lkdr),
                         nvMu(v, 0, 0, lkdr) + 6);
    tempz[1][0] = diagDj[0][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 0], nvTau(v, 1, 0, lkdr),
                         nvMu(v, 1, 0, lkdr));
    tempz[0][1] = diagDj[1][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 1], nvTau(v, 0, 1, lkdr),
                         nvMu(v, 0, 1, lkdr));
    tempz[1][1] = diagDj[1][C_RIGHT] *
                  pol.ma(4, 8, v.eta1[lkdr * NG + 1], nvTau(v, 1, 1, lkdr),
                         nvMu(v, 1, 1, lkdr) + 6);

    mat1g[0][0] = pol.ma(4, 10, -tempz[1][0], zeta1[0][1],
                         pol.ma(4, 9, -tempz[0][0], zeta1[0][0],
                                -diagDj[0][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 0],
                                           nvTau(v, 0, 0, lkdl),
                                           nvMu(v, 0, 0, lkdl) + 6)));
    mat1g[1][0] = pol.ma(4, 10, -tempz[1][0], zeta1[1][1],
                         pol.ma(4, 9, -tempz[0][0], zeta1[1][0],
                                -diagDj[0][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 0],
                                           nvTau(v, 1, 0, lkdl),
                                           nvMu(v, 1, 0, lkdl))));
    mat1g[0][1] = pol.ma(4, 10, -tempz[1][1], zeta1[0][1],
                         pol.ma(4, 9, -tempz[0][1], zeta1[0][0],
                                -diagDj[1][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 1],
                                           nvTau(v, 0, 1, lkdl),
                                           nvMu(v, 0, 1, lkdl))));
    mat1g[1][1] = pol.ma(4, 10, -tempz[1][1], zeta1[1][1],
                         pol.ma(4, 9, -tempz[0][1], zeta1[1][0],
                                -diagDj[1][C_LEFT] *
                                    pol.ma(4, 8, v.eta1[lkdl * NG + 1],
                                           nvTau(v, 1, 1, lkdl),
                                           nvMu(v, 1, 1, lkdl) + 6)));

    double bcc[2], vec1g[2];

    for (int ig = 0; ig < NG; ig++) {
        const double even_l =
            pol.ma(4, 13, v.eta2[lkdl * NG + ig], v.dsncff6[lkdl * NG + ig],
                   pol.ma2(4, 11, 3, v.dsncff2[lkdl * NG + ig], 10,
                           v.dsncff4[lkdl * NG + ig]));
        const double even_r =
            pol.ma(4, 13, v.eta2[lkdr * NG + ig], v.dsncff6[lkdr * NG + ig],
                   pol.ma2(4, 11, 3, v.dsncff2[lkdr * NG + ig], 10,
                           v.dsncff4[lkdr * NG + ig]));
        bcc[ig] = pol.ma2(4, 14, diagDj[ig][C_LEFT], even_l,
                          diagDj[ig][C_RIGHT], even_r);
        const double trl_l =
            pol.ma2(4, 16, nvMatMI(v, 0, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 0],
                    nvMatMI(v, 1, ig, lkl), sgnl * v.trlcff1[lkdl * NG + 1]);
        const double trl_r =
            pol.ma2(4, 16, nvMatMI(v, 0, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 0],
                    nvMatMI(v, 1, ig, lkr), sgnr * v.trlcff1[lkdr * NG + 1]);
        const double tz =
            pol.ma2(4, 18, tempz[0][ig], zeta2[0], tempz[1][ig], zeta2[1]);
        vec1g[ig] = pol.ma(4, 21, diagDj[ig][C_RIGHT], trl_r,
                           pol.ma(4, 20, -diagDj[ig][C_LEFT], trl_l, bcc[ig])) -
                    tz;
    }

    rdet = 1 / pol.ma2(4, 22, mat1g[0][0], mat1g[1][1], -mat1g[1][0], mat1g[0][1]);
    const double tmp = mat1g[0][0];
    mat1g[0][0] = rdet * mat1g[1][1];
    mat1g[1][0] = -rdet * mat1g[1][0];
    mat1g[0][1] = -rdet * mat1g[0][1];
    mat1g[1][1] = rdet * tmp;

    double oddcff[3][2];

    const double s0 = pol.ma2(4, 24, mat1g[0][0], vec1g[0], mat1g[1][0], vec1g[1]);
    const double s1 = pol.ma2(4, 24, mat1g[0][1], vec1g[0], mat1g[1][1], vec1g[1]);
    oddcff[1][0] = zeta2[0] - pol.ma2(4, 26, zeta1[0][0], s0, zeta1[1][0], s1);
    oddcff[1][1] = zeta2[1] - pol.ma2(4, 26, zeta1[0][1], s0, zeta1[1][1], s1);

    oddcff[2][0] = pol.ma2(4, 28, nvTau(v, 0, 0, lkdr), oddcff[1][0],
                           nvTau(v, 1, 0, lkdr), oddcff[1][1]);
    oddcff[2][1] = pol.ma2(4, 28, nvTau(v, 0, 1, lkdr), oddcff[1][0],
                           nvTau(v, 1, 1, lkdr), oddcff[1][1]);

    for (int ig = 0; ig < NG; ig++) {
        double t = pol.ma2(4, 30, nvMu(v, 0, ig, lkdr), oddcff[1][0],
                           -nvMatMI(v, 0, ig, lkr),
                           sgnr * v.trlcff1[lkdr * NG + 0]);
        t = pol.ma(4, 32, nvMu(v, 1, ig, lkdr), oddcff[1][1], t);
        t = pol.ma(4, 33, -nvMatMI(v, 1, ig, lkr),
                   sgnr * v.trlcff1[lkdr * NG + 1], t);
        oddcff[0][ig] = t;
    }

    for (int ig = 0; ig < NG; ig++) {
        double chain = pol.ma2(4, 34, -1.0, oddcff[0][ig], 3,
                               v.dsncff2[lkdr * NG + ig]);
        chain = pol.ma(4, 36, -6, oddcff[1][ig], chain);
        chain = pol.ma(4, 37, 10, v.dsncff4[lkdr * NG + ig], chain);
        chain = pol.ma(4, 38, -v.eta1[lkdr * NG + ig], oddcff[2][ig], chain);
        chain = pol.ma(4, 39, v.eta2[lkdr * NG + ig], v.dsncff6[lkdr * NG + ig], chain);
        v.jnet[ls * NG + ig] = sgnr * v.hmesh[lkr * NDIR + idirr] * 0.5 *
                               v.diagD[lkdr * NG + ig] * chain;

        v.phis[ls * NG + ig] =
            v.flux[lkr * NG + ig] -
            (oddcff[0][ig] + oddcff[1][ig] + oddcff[2][ig]) +
            v.dsncff2[lkdr * NG + ig] + v.dsncff4[lkdr * NG + ig] +
            v.dsncff6[lkdr * NG + ig];
    }
}

template <class POL>
RASBERY_XSR_HD inline void nodalCalculateJnet(const NodalView& v, int ls,
                                              const POL& pol) {
    const int lkl = v.lklr[ls * NLR + C_LEFT];
    const int lkr = v.lklr[ls * NLR + C_RIGHT];

    if (lkl < 0) {
        const int idirr = v.idirlr[ls * NLR + C_RIGHT];
        nodalJnet1n(v, ls, C_RIGHT, v.albedo[idirr * NLR + C_LEFT], pol);
    } else if (lkr < 0) {
        const int idirl = v.idirlr[ls * NLR + C_LEFT];
        nodalJnet1n(v, ls, C_LEFT, v.albedo[idirl * NLR + C_RIGHT], pol);
    } else {
        nodalJnet2n(v, ls, pol);
    }
}

} // namespace rasbery::nodal
