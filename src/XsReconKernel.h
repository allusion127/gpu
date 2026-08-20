#pragma once

// Shared host/device body of the fused equilibrium-Xe node step:
// micro-XS condense -> Xe-135 equilibrium overwrite -> damped _iden write ->
// ReconstructNode.  One call of xsreconSolveNode() reproduces one iteration of
// the fuel-node loop in XSSet::UpdateEquilibriumXenon (src/XSSet.cpp) for one
// node, in the same arithmetic order, so the host build of this header is the
// reference the device build is scored against.
//
// Determinism contract (mirrors the guards already written into XSSet.cpp):
//  - every isotope accumulation runs iso = 0..NISO-1 ascending, exactly like
//    ReconstructNode; no reassociation, no tree reduction over iso;
//  - the `relax < 1.0` guard is kept rather than fused into
//    1.0*(a-b)+b, per the comment at the damped update in XSSet.cpp;
//  - contraction is EXPLICIT: gcc at -O3 (-ffp-contract=fast) fuses a*b+c
//    into fma at every multiply-add in this body, so those spots are written
//    as xsrFma() -- a hardware FMA on both compilers -- and the device TU is
//    compiled with --fmad=false so nvcc cannot fuse anything else on its own.
//    The host harness (vs the verbatim -O3 reference) verifies each choice:
//    if gcc did NOT contract a spot written as xsrFma, the harness reports
//    the exact elementwise deviation instead of hiding it.
//
// This header must stay compilable by both g++ and nvcc: no STL containers,
// no exceptions, no allocation.  (Same rule and same reason as
// DepletionKernel.h on the depletion probe branch.)

#include <cmath>

#if defined(__CUDACC__)
    #define RASBERY_XSR_HD __host__ __device__
#else
    #define RASBERY_XSR_HD
#endif

namespace rasbery::xsrecon {

// Problem-fixed dimensions.  NISO/NG are compile-time by design: the isotope
// registry is a constexpr array of 39 (include/chiffon/Model.h) and every deck
// this solver accepts is 2-group.  Fixing them lets nvcc unroll every loop and
// keep cond[] and iden[] in registers.
constexpr int NISO     = 39;
constexpr int NG       = 2;
constexpr int NXS      = 11; // == N_XS_SCALAR
constexpr int AC_FIRST = 17; // == Chiffon::Isotope::iAcFirst
constexpr int AC_LAST  = 38; // == Chiffon::Isotope::iAcLast
constexpr int I135     = 3;  // == Chiffon::Isotope::iI135
constexpr int XE135    = 4;  // == Chiffon::Isotope::iXe135
constexpr int XE135M   = 5;  // == Chiffon::Isotope::iXe135m

// Chiffon::XSTYPE scalar slots (enum order asserted in CudaXsReconBackend.cu).
constexpr int T_XSTF = 0, T_XSDF = 1, T_XSAF = 2, T_XSFF = 3, T_XSNF = 4,
              T_XSKF = 5, T_XSSF = 6, T_XSRF = 7, T_FYLD = 8, T_XS2N = 9,
              T_XS3N = 10;

// ReconstructNode's scalar pass covers XSTF..XS3N minus the derived XSDF/XSRF.
// Host-side only (harness): device code cannot take the address of a
// namespace-scope constexpr array, so xsreconSolveNode carries its own local
// copy -- keep the two lists identical.
constexpr int ACTIVE_XT[9] = {T_XSTF, T_XSAF, T_XSFF, T_XSNF, T_XSKF,
                              T_XSSF, T_FYLD, T_XS2N, T_XS3N};

// Xe/I chain constants, verbatim from ApplyXeEquilibrium (src/XSSet.cpp).
constexpr double LAMBDA_I       = 2.930607e-05;
constexpr double LAMBDA_XE      = 2.106574e-05;
constexpr double LAMBDA_XEM     = 7.555561e-04;
constexpr double BR_I_TO_XE135M = 1.650900e-01;

/// Single-rounding a*b+c on both compilers: vfmadd on the host (gcc emits it
/// for std::fma when the ISA has FMA), DFMA on the device.
RASBERY_XSR_HD inline double xsrFma(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

/// Pointer view of one instance's SoA state.  The kernel sees only this view:
/// when any of these arrays becomes device-resident (or device-produced), the
/// pointers are repointed and the body below does not change.  Scalar slot i
/// of mic/lmp/xs is the array for Chiffon::XSTYPE i; every scalar array is
/// indexed [ig*nxyz + l], mic scalars [(iso*NG+ig)*nxyz + l], the scatter
/// blocks [(igs*NG+ige)*nxyz + l] and [(iso*NG*NG + igs*NG + ige)*nxyz + l],
/// iden [iso*nxyz + l], and phif -- the one AoS array -- [l*NG + ig].
struct BatchView {
    const double* mic[NXS];
    const double* mic_ssm;
    const double* lmp[NXS];
    const double* lmp_ssm;
    double*       iden;
    double*       xs[NXS];
    double*       xs_ssm;
    const double* phif;
    const int*    fuel;        ///< fuel-node indices, built once at setup
    int           n_fuel;
    int           nxyz;
    double        norm_factor; ///< NormFactor(power), computed on the host
    double        relax;
    const double* dep_i135;    ///< depTrans(iI135, j),  j = 0..NISO-1
    const double* dep_xe135;   ///< depTrans(iXe135, j), j = 0..NISO-1
};

/// One fuel node.  Returns 1 if the node was processed, 0 if it was skipped by
/// the same zero-flux guard the CPU loop uses.  max_change_out receives the
/// RAW (pre-damping) relative Xe-135 step for this node; the caller reduces
/// with max, which is order-insensitive and therefore deterministic.
RASBERY_XSR_HD inline int xsreconSolveNode(const BatchView& v, int l,
                                           double* max_change_out) {
    const int nxyz = v.nxyz;

    // (a) absolute flux -- same accumulation order as the CPU loop.
    double absflux[NG];
    double raw_sumflux = 0.0;
    for (int ig = 0; ig < NG; ++ig) {
        absflux[ig] = v.phif[l * NG + ig] * v.norm_factor;
        raw_sumflux += absflux[ig];
    }
    if (raw_sumflux <= 0.0)
        return 0;

    // (b) condense the micro XS to one group; snapshot the densities.
    const double invflux = 1.0 / raw_sumflux;
    double       cond[NISO * NXS];
    double       iden[NISO];
    for (int iso = 0; iso < NISO; ++iso) {
        for (int xt = 0; xt < NXS; ++xt) {
            double sum = 0.0;
            for (int ig = 0; ig < NG; ++ig)
                sum = xsrFma(v.mic[xt][(iso * NG + ig) * nxyz + l], absflux[ig], sum);
            cond[iso * NXS + xt] = sum * invflux;
        }
        iden[iso] = v.iden[iso * nxyz + l];
    }

    // FluxScale(abs_flux, ng) recomputes the same ig-ascending sum as (a), so
    // raw_sumflux * 1.0e-24 is bit-identical to it.
    const double sumflux = raw_sumflux * 1.0e-24;

    // (c) ApplyXeEquilibrium, verbatim.
    const double old_i   = iden[I135];
    const double old_xe  = iden[XE135];
    const double old_xem = iden[XE135M];
    {
        double fissSourceI = 0.0, fissSourceXe = 0.0;
        for (int j = AC_FIRST; j <= AC_LAST; ++j) {
            double xsff  = cond[j * NXS + T_XSFF];
            double fRate = iden[j] * xsff * sumflux;
            fissSourceI += fRate * v.dep_i135[j];
            fissSourceXe += fRate * v.dep_xe135[j];
        }
        double sigaXe = cond[XE135 * NXS + T_XSAF] * sumflux;
        double Ieq    = fissSourceI / LAMBDA_I;
        double Xeeq   = xsrFma(LAMBDA_I, Ieq, fissSourceXe) / (LAMBDA_XE + sigaXe);

        iden[I135]   = Ieq;
        iden[XE135]  = Xeeq;
        iden[XE135M] = BR_I_TO_XE135M * LAMBDA_I * Ieq / LAMBDA_XEM;
    }

    // (d) raw relative step.  std::max(a, b) is (a < b) ? b : a; spelled out
    // so host and device pick the identical operand.
    const double new_xe    = iden[XE135];
    const double abs_newxe = fabs(new_xe);
    const double scale     = (abs_newxe < 1.0e-30) ? 1.0e-30 : abs_newxe;
    *max_change_out        = fabs(new_xe - old_xe) / scale;

    // x <- x + relax*(F(x) - x).  The guard keeps the undamped case
    // arithmetically identical instead of relying on 1.0*(a-b)+b == a.
    if (v.relax < 1.0) {
        iden[I135]   = xsrFma(v.relax, iden[I135] - old_i, old_i);
        iden[XE135]  = xsrFma(v.relax, iden[XE135] - old_xe, old_xe);
        iden[XE135M] = xsrFma(v.relax, iden[XE135M] - old_xem, old_xem);
    }

    // (e) only the Xe-chain rows change in the global density array.
    v.iden[I135 * nxyz + l]   = iden[I135];
    v.iden[XE135 * nxyz + l]  = iden[XE135];
    v.iden[XE135M * nxyz + l] = iden[XE135M];

    // (f) ReconstructNode, verbatim: iso-ascending accumulation per element.
    // The CPU version re-reads _iden from memory here; rows 3..5 of the local
    // copy hold exactly the values just stored, so the operands are the same.
    // Local mirror of ACTIVE_XT: device code cannot reference the
    // namespace-scope array's storage.
    const int active_xt[9] = {T_XSTF, T_XSAF, T_XSFF, T_XSNF, T_XSKF,
                              T_XSSF, T_FYLD, T_XS2N, T_XS3N};
    for (int a = 0; a < 9; ++a) {
        const int     xt  = active_xt[a];
        const double* lmp = v.lmp[xt];
        const double* mic = v.mic[xt];
        double*       dst = v.xs[xt];
        for (int ig = 0; ig < NG; ++ig) {
            double val = lmp[ig * nxyz + l];
            for (int iso = 0; iso < NISO; ++iso)
                val = xsrFma(mic[(iso * NG + ig) * nxyz + l], iden[iso], val);
            dst[ig * nxyz + l] = val;
        }
    }

    for (int igs = 0; igs < NG; ++igs) {
        for (int ige = 0; ige < NG; ++ige) {
            double val = v.lmp_ssm[(igs * NG + ige) * nxyz + l];
            for (int iso = 0; iso < NISO; ++iso)
                val = xsrFma(v.mic_ssm[(iso * NG * NG + igs * NG + ige) * nxyz + l],
                             iden[iso], val);
            v.xs_ssm[(igs * NG + ige) * nxyz + l] = val;
        }
    }

    for (int ig = 0; ig < NG; ++ig) {
        double tr                   = v.xs[T_XSTF][ig * nxyz + l];
        v.xs[T_XSDF][ig * nxyz + l] = (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;
    }

    for (int igs = 0; igs < NG; ++igs) {
        double rf = v.xs[T_XSAF][igs * nxyz + l];
        for (int ige = 0; ige < NG; ++ige)
            rf += v.xs_ssm[(igs * NG + ige) * nxyz + l];
        v.xs[T_XSRF][igs * nxyz + l] = rf;
    }

    return 1;
}

} // namespace rasbery::xsrecon
