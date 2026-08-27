#pragma once

// Pure coefficient body for Nodal::updateConstant.
//
// Shared host/device as of Rev.7.1 Task 4: src/CudaNodalConstantKernel.h calls
// this body from k_nodal_update_constant, and the CUDA TU is built with
// --fmad=false so nvcc compiles the written forms literally.  Nothing about the
// arithmetic changed -- the statement order and association below are still
// Nodal::updateConstant's, verbatim.
//
// CLASSIFICATION: N1 (deterministic GPU baseline transition), Rev.7.1 Sec 9.1.
// NOT B0.  The Task 4 Step 0 B0-rescue spike (Sec 6.1, test/nodal_constant_exp_probe.cu)
// measured the two transcendental calls below against glibc 2.39 on sm_61:
//
//     sqrt   4,000,000 / 4,000,000 bit-identical      (0 mismatches)
//     exp      133,547 / 4,000,000 differ  (3.34%), max 1 ulp
//              in the physical band kp2 in [1e-3,1e2]: 5.34% differ
//
// sqrt is IEEE-754 correctly rounded on both sides, exactly as NVIDIA documents,
// so it would have been B0 on its own.  exp is not: CUDA's is <=1 ulp and glibc's
// is a different (near-correctly-rounded) algorithm, and on 1 in 19 of the
// arguments this body actually evaluates they round to different doubles.  The
// rescue therefore FAILED and Task 4 keeps the Rev.7 path: run-to-run bit
// determinism is required, Gate A/B and the v3 freeze are deferred to Task 22.
//
// What that means for anyone editing this file: the GPU arm is allowed to differ
// from the CPU arm at the 1-ulp level in exp AND NOWHERE ELSE.  Every other
// operation here is IEEE basic arithmetic, so a deviation that is not traceable
// to exp is a bug, not a classification.  test/nodal_constant_gpu_replay.cpp
// enforces exactly that split.

// ---------------------------------------------------------------------------
// CONTRACTION IS EXPLICIT HERE, and it is not optional
// ---------------------------------------------------------------------------
//
// gcc at -O3 (-ffp-contract=fast, the C++ default) fuses TWENTY multiply-adds
// when it inlines this body into Nodal::updateConstant -- measured, by counting
// vfmadd/vfmsub/vfnmadd in the -march=native asm for
// _ZN7rasbery5Nodal14updateConstantERKi.  nvcc compiling the same source with
// --fmad=false fuses NONE of them.  The two arms therefore do not merely differ
// by the 1-ulp exp of the N1 classification: at authoring time the unpinned
// device build differed from the host on 95% of eta1 and by up to 3e15 ulp on
// m264, because the bfcff2/bfcff4/m264 terms are cancellation-heavy and a
// single unfused rounding in the numerator is amplified.
//
// So every a*b+c and a*b+c*d site below goes through ncMa1/ncMa2 under a MINED
// mask, exactly as NodalKernel.h and XsReconKernel.h already do.  The mask is
// the set of choices gcc actually made; with it, the host build is bit-identical
// to the unpinned body it replaced (test/nodal_constant_gpu_replay.cpp --mine
// re-derives it, and tools/test_nodal_constant_kernel.py scores this file
// against an independent verbatim copy of the original).
//
// EDITING RULE.  Changing a form bit changes the CPU baseline.  Do not "tidy"
// one; re-mine and re-record.

#include "XsReconKernel.h" // xsrFma / xsrMul, the two rounding primitives

#include <cmath>

#if defined(__CUDACC__)
    #define RASBERY_NODAL_CONST_HD __host__ __device__
#else
    #define RASBERY_NODAL_CONST_HD
#endif

namespace rasbery::nodal {

/// Mined contraction mask for nodalConstantCoefficients.
///
/// Provenance: test/nodal_constant_gpu_replay.cpp --mine, coordinate descent to
/// ZERO mismatches against a verbatim quotation of the pre-policy body over a
/// 2-group PWR/SMR sweep, g++ 13.3.0 -O3 -march=native -DNDEBUG (Ubuntu 24.04).
/// Cross-checked against the 20 fma instructions gcc emits for
/// Nodal::updateConstant.
///
/// 1-bit sites: 1 = single-rounding fma.  2-bit sites over A*B + C*D:
/// 0 = both products rounded, 1 = fma(A, B, round(C*D)), 2 = fma(C, D,
/// round(A*B)).
inline constexpr unsigned long long NODAL_CONST_FORMS = 0x55545FFFull;

/// Device code cannot address a namespace-scope constant's storage in every
/// context (same finding as xsrecon's ACTIVE_XT), and a default argument reads
/// better than a policy object for a body with one call site.  Keep identical
/// to NODAL_CONST_FORMS.
RASBERY_NODAL_CONST_HD constexpr unsigned long long nodalConstForms() {
    return 0x55545FFFull;
}

// Site bit offsets.  Named so a diff of the mask is readable and so the miner
// and the body cannot disagree about which bit is which.
enum : int {
    // --- 1-bit sites: c +- a*b ---
    NC_B2_KP2SINH = 0,  ///< bfcff2:  ... + kp2*sinhkp
    NC_B4_105SINH,      ///< bfcff4:  ... + 105*sinhkp
    NC_B4_45KP2SINH,    ///< bfcff4:  ... + 45*kp2*sinhkp
    NC_B4_KP4SINH,      ///< bfcff4:  ... + kp4*sinhkp
    NC_B1_KPCOSH,       ///< bfcff1:  kp*coshkp - sinhkp
    NC_B3_15SINH,       ///< bfcff3:  ... - 15*sinhkp
    NC_B3_6KP2SINH,     ///< bfcff3:  ... - 6*kp2*sinhkp
    NC_EVEN_B0,         ///< eventemp: coshkp + bfcff0   (bfcff0 = -sinhkp*rkp)
    NC_ETA1_A,          ///< eta1:    kp*coshkp + bfcff1
    NC_ETA1_B,          ///< eta1:    ... + 6*bfcff3
    NC_ETA2_B,          ///< eta2:    ... + 10*bfcff4
    NC_M251_A,          ///< m251:    kp*coshkp - sinhkp
    NC_M251_B,          ///< m251:    ... + 5*bfcff3
    NC_M253_W,          ///< m253:    5 + 2*kp2
    NC_M262_B,          ///< m262:    ... + 7*kp*bfcff4
    NC_M264_P,          ///< m264:    21 + 2*kp2
    NC_M264_Q,          ///< m264:    105 + 45*kp2
    NC_M264_Q4,         ///< m264:    (105 + 45*kp2) + kp4
    NC_ONE_BIT_COUNT,

    // --- 2-bit sites: a*b + c*d ---
    NC2_B2_HEAD = NC_ONE_BIT_COUNT, ///< bfcff2: -3*kp*coshkp + 3*sinhkp
    NC2_B4_HEAD = NC2_B2_HEAD + 2,  ///< bfcff4: -105*kp*coshkp - 10*kp3*coshkp
    NC2_B3_HEAD = NC2_B4_HEAD + 2,  ///< bfcff3: 15*kp*coshkp + kp3*coshkp
    NC2_ETA2_HEAD = NC2_B3_HEAD + 2,///< eta2:   kp*sinhkp + 3*bfcff2
    NC2_M253_HEAD = NC2_ETA2_HEAD + 2, ///< m253: kp*(15+kp2)*coshkp - 3*w*sinhkp
    NC2_M262_HEAD = NC2_M253_HEAD + 2, ///< m262: -3*kp*coshkp + (3+kp2)*sinhkp
    NC2_M264_HEAD = NC2_M262_HEAD + 2, ///< m264: -5*kp*p*coshkp + q*sinhkp
    NC_BIT_COUNT  = NC2_M264_HEAD + 2
};

static_assert(NC_BIT_COUNT <= 64, "the form mask is one uint64");

/// c + a*b, fused or with the product rounded separately.
RASBERY_NODAL_CONST_HD inline double ncMa1(unsigned long long m, int bit, double a,
                                           double b, double c) {
#if defined(__CUDA_ARCH__)
    return ((m >> bit) & 1ull) ? fma(a, b, c) : a * b + c;
#else
    return ((m >> bit) & 1ull) ? xsrecon::xsrFma(a, b, c) : xsrecon::xsrMul(a, b) + c;
#endif
}

/// a*b + c*d.  gcc picks ONE of the two products to fuse and rounds the other;
/// which one is not guessable from the source, which is why it is mined.
RASBERY_NODAL_CONST_HD inline double ncMa2(unsigned long long m, int bit, double a,
                                           double b, double c, double d) {
    const unsigned st = static_cast<unsigned>((m >> bit) & 3ull);
#if defined(__CUDA_ARCH__)
    if (st == 1) return fma(a, b, c * d);
    if (st == 2) return fma(c, d, a * b);
    return a * b + c * d;
#else
    if (st == 1) return xsrecon::xsrFma(a, b, xsrecon::xsrMul(c, d));
    if (st == 2) return xsrecon::xsrFma(c, d, xsrecon::xsrMul(a, b));
    return xsrecon::xsrMul(a, b) + xsrecon::xsrMul(c, d);
#endif
}

struct NodalConstantCoefficients {
    double eta1;
    double eta2;
    double m260;
    double m251;
    double m253;
    double m262;
    double m264;
    double diagD;
    double diagDI;
};

/// `forms` defaults to the mined mask, so production reads exactly as it did.
/// The parameter exists for the miner, which sweeps candidate masks against a
/// verbatim quotation of the pre-policy body; a runtime mask keeps this ONE
/// function with ONE instantiation, which a template policy would not (gcc
/// contracts un-policied expressions differently per instantiation -- the
/// finding recorded in test/nodal_device_replay.cu).
[[nodiscard]] RASBERY_NODAL_CONST_HD inline NodalConstantCoefficients
nodalConstantCoefficients(double xsrf, double xsdf, double hmesh,
                          unsigned long long forms = nodalConstForms()) {
    // Keep the statement order and association of Nodal::updateConstant.
    // Non-multiply-add arithmetic is left verbatim: products, quotients and the
    // two library calls cannot be contracted into anything, so they need no
    // policy and reading them against Nodal.cpp must stay trivial.
    double kp2    = xsrf * hmesh * hmesh / (4 * xsdf);
    double kp     = std::sqrt(kp2);
    double kp3    = kp2 * kp;
    double kp4    = kp2 * kp2;
    double rkp    = 1 / kp;
    double rkp2   = rkp * rkp;
    double rkp3   = rkp2 * rkp;
    double rkp4   = rkp2 * rkp2;
    double rkp5   = rkp2 * rkp3;
    double ekp    = std::exp(kp);
    double iekp   = 1.0 / ekp;
    double sinhkp = 0.5 * (ekp - iekp);
    double coshkp = 0.5 * (ekp + iekp);

    // bfcff2 = -5 * (-3*kp*coshkp + 3*sinhkp + kp2*sinhkp) * rkp3
    double b2 = ncMa2(forms, NC2_B2_HEAD, -3 * kp, coshkp, 3, sinhkp);
    b2        = ncMa1(forms, NC_B2_KP2SINH, kp2, sinhkp, b2);
    double bfcff2 = -5 * b2 * rkp3;

    // bfcff4 = -9 * (-105*kp*coshkp - 10*kp3*coshkp + 105*sinhkp
    //                + 45*kp2*sinhkp + kp4*sinhkp) * rkp5
    double b4 = ncMa2(forms, NC2_B4_HEAD, -105 * kp, coshkp, -10 * kp3, coshkp);
    b4        = ncMa1(forms, NC_B4_105SINH, 105, sinhkp, b4);
    b4        = ncMa1(forms, NC_B4_45KP2SINH, 45 * kp2, sinhkp, b4);
    b4        = ncMa1(forms, NC_B4_KP4SINH, kp4, sinhkp, b4);
    double bfcff4 = -9. * b4 * rkp5;

    // bfcff1 = -3 * (kp*coshkp - sinhkp) * rkp2
    double bfcff1 = -3 * ncMa1(forms, NC_B1_KPCOSH, kp, coshkp, -sinhkp) * rkp2;

    // bfcff3 = -7 * (15*kp*coshkp + kp3*coshkp - 15*sinhkp - 6*kp2*sinhkp) * rkp4
    double b3 = ncMa2(forms, NC2_B3_HEAD, 15 * kp, coshkp, kp3, coshkp);
    b3        = ncMa1(forms, NC_B3_15SINH, -15, sinhkp, b3);
    b3        = ncMa1(forms, NC_B3_6KP2SINH, -6 * kp2, sinhkp, b3);
    double bfcff3 = -7 * b3 * rkp4;

    // bfcff0 = -sinhkp*rkp is single-use, so gcc may fold it into the sum below.
    double oddtemp  = 1 / (sinhkp + bfcff1 + bfcff3);
    double eventemp =
        1 / (ncMa1(forms, NC_EVEN_B0, -sinhkp, rkp, coshkp) + bfcff2 + bfcff4);

    NodalConstantCoefficients out{};
    double e1 = ncMa1(forms, NC_ETA1_A, kp, coshkp, bfcff1);
    e1        = ncMa1(forms, NC_ETA1_B, 6, bfcff3, e1);
    out.eta1  = e1 * oddtemp;

    double e2 = ncMa2(forms, NC2_ETA2_HEAD, kp, sinhkp, 3, bfcff2);
    e2        = ncMa1(forms, NC_ETA2_B, 10, bfcff4, e2);
    out.eta2  = e2 * eventemp;

    out.m260 = 2 * out.eta2;

    double t51 = ncMa1(forms, NC_M251_A, kp, coshkp, -sinhkp);
    t51        = ncMa1(forms, NC_M251_B, 5, bfcff3, t51);
    out.m251   = 2 * t51 * oddtemp;

    // m253 = 2*(kp*(15+kp2)*coshkp - 3*(5+2*kp2)*sinhkp) * oddtemp * rkp2
    double w53 = ncMa1(forms, NC_M253_W, 2, kp2, 5);
    out.m253   = 2 *
               ncMa2(forms, NC2_M253_HEAD, kp * (15 + kp2), coshkp, -3 * w53, sinhkp) *
               oddtemp * rkp2;

    // m262 = 2*(-3*kp*coshkp + (3+kp2)*sinhkp + 7*kp*bfcff4) * eventemp * rkp
    double t62 = ncMa2(forms, NC2_M262_HEAD, -3 * kp, coshkp, 3 + kp2, sinhkp);
    t62        = ncMa1(forms, NC_M262_B, 7 * kp, bfcff4, t62);
    out.m262   = 2 * t62 * eventemp * rkp;

    // m264 = 2*(-5*kp*(21+2*kp2)*coshkp + (105+45*kp2+kp4)*sinhkp) * eventemp * rkp3
    double p64 = ncMa1(forms, NC_M264_P, 2, kp2, 21);
    double q64 = ncMa1(forms, NC_M264_Q, 45, kp2, 105);
    q64        = ncMa1(forms, NC_M264_Q4, kp2, kp2, q64); // kp2*kp2 == kp4
    out.m264   = 2 *
               ncMa2(forms, NC2_M264_HEAD, -5 * kp * p64, coshkp, q64, sinhkp) *
               eventemp * rkp3;

    out.diagD  = 4 * xsdf / (hmesh * hmesh);
    out.diagDI = 1.0 / out.diagD;
    return out;
}

} // namespace rasbery::nodal
