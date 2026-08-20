#pragma once

// Shared pieces of the xsrecon consistency harnesses: deterministic input
// generation, the verbatim CPU reference loop (quoted from src/XSSet.cpp),
// and the ULP comparator.  Included by the host harness and the device
// harness so both score against the identical reference.

#include "XsReconKernel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace xsr = rasbery::xsrecon;

namespace {

// Deterministic 64-bit LCG (Knuth): the harness must not depend on run time.
std::uint64_t g_state = 0x9E3779B97F4A7C15ULL;
double urand() {
    g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((g_state >> 11) & ((1ULL << 53) - 1)) /
           static_cast<double>(1ULL << 53);
}

struct Arrays {
    int                 nxyz = 0;
    std::vector<double> mic[xsr::NXS], lmp[xsr::NXS], xs[xsr::NXS];
    std::vector<double> mic_ssm, lmp_ssm, xs_ssm, iden, phif;
    std::vector<int>    fuel;
    std::vector<char>   is_fuel;
    std::vector<double> dep_i135, dep_xe135;

    xsr::BatchView view(double norm_factor, double relax) {
        xsr::BatchView v{};
        for (int xt = 0; xt < xsr::NXS; ++xt) {
            v.mic[xt] = mic[xt].data();
            v.lmp[xt] = lmp[xt].data();
            v.xs[xt]  = xs[xt].data();
        }
        v.mic_ssm     = mic_ssm.data();
        v.lmp_ssm     = lmp_ssm.data();
        v.xs_ssm      = xs_ssm.data();
        v.iden        = iden.data();
        v.phif        = phif.data();
        v.fuel        = fuel.data();
        v.n_fuel      = static_cast<int>(fuel.size());
        v.nxyz        = nxyz;
        v.norm_factor = norm_factor;
        v.relax       = relax;
        v.dep_i135    = dep_i135.data();
        v.dep_xe135   = dep_xe135.data();
        return v;
    }
};

Arrays makeArrays(int nxyz) {
    Arrays a;
    a.nxyz               = nxyz;
    const std::size_t nx = static_cast<std::size_t>(nxyz);
    for (int xt = 0; xt < xsr::NXS; ++xt) {
        a.mic[xt].resize(static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx);
        a.lmp[xt].resize(static_cast<std::size_t>(xsr::NG) * nx);
        a.xs[xt].resize(static_cast<std::size_t>(xsr::NG) * nx);
        for (auto& x : a.mic[xt]) x = urand() * 1.0e-22;
        for (auto& x : a.lmp[xt]) x = urand() * 1.0e-2;
        for (auto& x : a.xs[xt]) x = urand(); // stale values the kernel must overwrite
    }
    a.mic_ssm.resize(static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx);
    a.lmp_ssm.resize(static_cast<std::size_t>(xsr::NG) * xsr::NG * nx);
    a.xs_ssm.resize(static_cast<std::size_t>(xsr::NG) * xsr::NG * nx);
    a.iden.resize(static_cast<std::size_t>(xsr::NISO) * nx);
    a.phif.resize(static_cast<std::size_t>(xsr::NG) * nx);
    for (auto& x : a.mic_ssm) x = urand() * 1.0e-23;
    for (auto& x : a.lmp_ssm) x = urand() * 1.0e-3;
    for (auto& x : a.xs_ssm) x = urand();
    for (auto& x : a.iden) x = urand() * 1.0e-4;
    for (auto& x : a.phif) x = urand() * 1.0e14;
    // The synthetic equilibrium Xe/I densities land near 1e-45; seed the
    // Xe-chain rows at the same magnitude, or the damped update's
    // old + relax*(new - old) would be dominated by `old` and its low bits
    // could mask a wrong contraction choice in that expression.
    for (int iso : {xsr::I135, xsr::XE135, xsr::XE135M})
        for (std::size_t l = 0; l < nx; ++l)
            a.iden[static_cast<std::size_t>(iso) * nx + l] *= 1.0e-41;

    a.is_fuel.assign(static_cast<std::size_t>(nxyz), 0);
    for (int l = 0; l < nxyz; ++l) {
        const double r = urand();
        if (r < 0.65) {
            a.fuel.push_back(l); // fuel node
            a.is_fuel[static_cast<std::size_t>(l)] = 1;
            if (r < 0.02)        // a few fuel nodes with zero flux (skip guard)
                for (int ig = 0; ig < xsr::NG; ++ig)
                    a.phif[static_cast<std::size_t>(l) * xsr::NG + ig] = 0.0;
        }
    }
    a.dep_i135.assign(xsr::NISO, 0.0);
    a.dep_xe135.assign(xsr::NISO, 0.0);
    for (int j = xsr::AC_FIRST; j <= xsr::AC_LAST; ++j) {
        a.dep_i135[static_cast<std::size_t>(j)]  = 0.05 + 0.02 * urand();
        a.dep_xe135[static_cast<std::size_t>(j)] = 0.001 * urand();
    }
    return a;
}

// refApplyXeEquilibrium reads these like production reads the global depTrans.
inline const double* g_dep_i135_ref  = nullptr;
inline const double* g_dep_xe135_ref = nullptr;

// --------------------------------------------------------------------------
// Reference: quoted from src/XSSet.cpp with its OpenMP structure, and with
// every contraction-sensitive expression PINNED to the form production gcc
// actually emitted -- mined from a real capture by xsrecon_form_probe
// (condense and fissSource unfused; both sides of the Xeeq quotient fused,
// the denominator across the sigaXe temporary; reconstruct fused).  Quoting
// the source alone is NOT enough: gcc contracts the same statements
// differently in different translation units, which this harness proved the
// hard way.  refMulR pins an unfused rounding; std::fma pins a fused one.
// --------------------------------------------------------------------------

// Separately-rounded product the harness compiler cannot re-fuse.
inline double refMulR(double a, double b) {
#if defined(__GNUC__)
    double p = a * b;
    asm volatile("" : "+x"(p));
    return p;
#else
    volatile double p = a * b;
    return p;
#endif
}

// FluxScale, quoted.
static double refFluxScale(const double* flux, int ng) {
    double sum = 0.0;
    for (int ig = 0; ig < ng; ++ig)
        sum += flux[ig];
    return sum * 1.0e-24;
}

// ApplyXeEquilibrium, quoted (milk::Vector -> std::vector, depTrans -> the
// globals above are the only edits).
static void refApplyXeEquilibrium(std::vector<double>& iden,
                                  const std::vector<double>& cond, double sumflux) {
    constexpr double lambdaI     = 2.930607e-05;
    constexpr double lambdaXe    = 2.106574e-05;
    constexpr double lambdaXem   = 7.555561e-04;
    constexpr double brItoXe135m = 1.650900e-01;

    double fissSourceI = 0.0, fissSourceXe = 0.0;
    for (size_t j = xsr::AC_FIRST; j <= xsr::AC_LAST; ++j) {
        double xsff  = cond[j * xsr::NXS + xsr::T_XSFF];
        double fRate = iden[j] * xsff * sumflux;
        fissSourceI += refMulR(fRate, g_dep_i135_ref[j]);
        fissSourceXe += refMulR(fRate, g_dep_xe135_ref[j]);
    }

    double Ieq  = fissSourceI / lambdaI;
    double Xeeq = std::fma(lambdaI, Ieq, fissSourceXe) /
                  std::fma(cond[xsr::XE135 * xsr::NXS + xsr::T_XSAF], sumflux, lambdaXe);

    iden[xsr::I135]   = Ieq;
    iden[xsr::XE135]  = Xeeq;
    iden[xsr::XE135M] = brItoXe135m * lambdaI * Ieq / lambdaXem;
}

// ReconstructNode, quoted (member access -> BatchView is the only edit).
static void refReconstructNode(const xsr::BatchView& v, size_t l) {
    const int    ng   = xsr::NG;
    const int    nxyz = v.nxyz;
    const size_t niso = xsr::NISO;

    for (int xt = xsr::T_XSTF; xt <= xsr::T_XS3N; ++xt) {
        if (xt == xsr::T_XSDF || xt == xsr::T_XSRF) continue;
        double*       dst = v.xs[xt];
        const double* lmp = v.lmp[xt];
        const double* mic = v.mic[xt];

        for (int ig = 0; ig < ng; ++ig) {
            double val = lmp[ig * nxyz + l];
            for (size_t iso = 0; iso < niso; ++iso)
                val = std::fma(mic[(iso * ng + ig) * nxyz + l],
                               v.iden[iso * nxyz + l], val);
            dst[ig * nxyz + l] = val;
        }
    }

    for (int igs = 0; igs < ng; ++igs) {
        for (int ige = 0; ige < ng; ++ige) {
            double val = v.lmp_ssm[(igs * ng + ige) * nxyz + l];
            for (size_t iso = 0; iso < niso; ++iso)
                val = std::fma(v.mic_ssm[(iso * ng * ng + igs * ng + ige) * nxyz + l],
                               v.iden[iso * nxyz + l], val);
            v.xs_ssm[(igs * ng + ige) * nxyz + l] = val;
        }
    }

    for (int ig = 0; ig < ng; ++ig) {
        double tr = v.xs[xsr::T_XSTF][ig * nxyz + l];
        v.xs[xsr::T_XSDF][ig * nxyz + l] =
            (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;
    }

    for (int igs = 0; igs < ng; ++igs) {
        double rf = v.xs[xsr::T_XSAF][igs * nxyz + l];
        for (int ige = 0; ige < ng; ++ige)
            rf += v.xs_ssm[(igs * ng + ige) * nxyz + l];
        v.xs[xsr::T_XSRF][igs * nxyz + l] = rf;
    }
}

double referenceLoop(const xsr::BatchView& v, const std::vector<char>& is_fuel) {
    const int    ng   = xsr::NG;
    const int    nxyz = v.nxyz;
    const size_t niso = xsr::NISO;
    double max_change = 0.0;

    g_dep_i135_ref  = v.dep_i135;
    g_dep_xe135_ref = v.dep_xe135;

#pragma omp parallel if (nxyz > 64) reduction(max : max_change)
    {
        static thread_local std::vector<double> ws_condensed;
        static thread_local std::vector<double> ws_iden;
        static thread_local std::vector<double> abs_flux_tls;

        if (ws_iden.size() != niso)
            ws_iden.resize(niso);
        if (abs_flux_tls.size() != static_cast<size_t>(ng))
            abs_flux_tls.resize(static_cast<size_t>(ng));
        if (ws_condensed.size() < niso * xsr::NXS)
            ws_condensed.resize(niso * xsr::NXS, 0.0);

        const double* mic_ptrs[xsr::NXS] = {
            v.mic[0], v.mic[1], v.mic[2], v.mic[3], v.mic[4], v.mic[5],
            v.mic[6], v.mic[7], v.mic[8], v.mic[9], v.mic[10]};

#pragma omp for schedule(dynamic, 8)
        for (int l = 0; l < nxyz; ++l) {
            if (!is_fuel[static_cast<size_t>(l)])
                continue;

            double raw_sumflux = 0.0;
            for (int ig = 0; ig < ng; ++ig) {
                abs_flux_tls[static_cast<size_t>(ig)] =
                    v.phif[l * ng + ig] * v.norm_factor;
                raw_sumflux += abs_flux_tls[static_cast<size_t>(ig)];
            }
            if (raw_sumflux <= 0.0)
                continue;

            const double invflux = 1.0 / raw_sumflux;
            for (size_t iso = 0; iso < niso; ++iso) {
                double* dst = ws_condensed.data() + iso * xsr::NXS;
                for (size_t xt = 0; xt < static_cast<size_t>(xsr::NXS); ++xt) {
                    double sum = 0.0;
                    for (int ig = 0; ig < ng; ++ig) {
                        const size_t off =
                            (iso * static_cast<size_t>(ng) + static_cast<size_t>(ig)) *
                                static_cast<size_t>(nxyz) +
                            static_cast<size_t>(l);
                        sum += refMulR(mic_ptrs[xt][off],
                                       abs_flux_tls[static_cast<size_t>(ig)]);
                    }
                    dst[xt] = sum * invflux;
                }
                ws_iden[iso] = v.iden[iso * static_cast<size_t>(nxyz) +
                                      static_cast<size_t>(l)];
            }

            const double old_i   = ws_iden[xsr::I135];
            const double old_xe  = ws_iden[xsr::XE135];
            const double old_xem = ws_iden[xsr::XE135M];
            refApplyXeEquilibrium(ws_iden, ws_condensed,
                                  refFluxScale(abs_flux_tls.data(), ng));
            const double new_xe = ws_iden[xsr::XE135];
            const double scale  = std::max(std::abs(new_xe), 1.0e-30);
            max_change          = std::max(max_change, std::abs(new_xe - old_xe) / scale);

            if (v.relax < 1.0) {
                ws_iden[xsr::I135]   = std::fma(v.relax, ws_iden[xsr::I135] - old_i, old_i);
                ws_iden[xsr::XE135]  = std::fma(v.relax, ws_iden[xsr::XE135] - old_xe, old_xe);
                ws_iden[xsr::XE135M] = std::fma(v.relax, ws_iden[xsr::XE135M] - old_xem, old_xem);
            }

            v.iden[xsr::I135 * static_cast<size_t>(nxyz) + static_cast<size_t>(l)] =
                ws_iden[xsr::I135];
            v.iden[xsr::XE135 * static_cast<size_t>(nxyz) + static_cast<size_t>(l)] =
                ws_iden[xsr::XE135];
            v.iden[xsr::XE135M * static_cast<size_t>(nxyz) + static_cast<size_t>(l)] =
                ws_iden[xsr::XE135M];

            refReconstructNode(v, static_cast<size_t>(l));
        }
    }
    return max_change;
}

int ulpDiffCount(const char* name, const std::vector<double>& a,
                 const std::vector<double>& b, int report_limit, int& reported) {
    int diffs = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
            ++diffs;
            if (reported < report_limit) {
                std::int64_t ba, bb;
                std::memcpy(&ba, &a[i], 8);
                std::memcpy(&bb, &b[i], 8);
                std::printf("  DIFF %s[%zu]: ref=%.17g got=%.17g ulp=%lld\n", name, i,
                            a[i], b[i], static_cast<long long>(bb - ba));
                ++reported;
            }
        }
    }
    return diffs;
}

} // namespace

