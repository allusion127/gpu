// Which contraction FORM did production gcc pick, per expression?  Recompute
// the captured first Xe call under every form combination and count exact bit
// matches against production's own outputs.  Plain forms are pinned with an
// inline-asm barrier so the probe's own compiler cannot re-fuse them.
//
// Usage: rasbery_xsrecon_form_probe <dump-prefix>

#include "XsReconKernel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace xsr = rasbery::xsrecon;

namespace {

inline double opaque(double x) {
    asm volatile("" : "+x"(x));
    return x;
}

// a*b + c under the two possible roundings.
inline double mac(bool use_fma, double a, double b, double c) {
    if (use_fma) return std::fma(a, b, c);
    return opaque(a * b) + c;
}

std::vector<double> readDoubles(std::FILE* f, size_t n) {
    std::vector<double> v(n);
    if (std::fread(v.data(), sizeof(double), n, f) != n) {
        std::fprintf(stderr, "short read\n");
        std::exit(2);
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::string prefix = argv[1];
    std::FILE*        fi     = std::fopen((prefix + ".in").c_str(), "rb");
    std::FILE*        fo     = std::fopen((prefix + ".out").c_str(), "rb");
    if (!fi || !fo) return 2;

    std::int64_t hdr[2];
    if (std::fread(hdr, sizeof hdr[0], 2, fi) != 2) return 2;
    const int ng = static_cast<int>(hdr[0]), nxyz = static_cast<int>(hdr[1]);
    double norm_factor, relax;
    std::fread(&norm_factor, sizeof(double), 1, fi);
    std::fread(&relax, sizeof(double), 1, fi);
    std::vector<char> isf(static_cast<size_t>(nxyz));
    std::fread(isf.data(), 1, isf.size(), fi);
    std::vector<double> dep = readDoubles(fi, 2 * static_cast<size_t>(xsr::NISO));

    const size_t nx  = static_cast<size_t>(nxyz);
    const size_t ngn = static_cast<size_t>(ng) * nx;
    const size_t ssn = static_cast<size_t>(ng) * ng * nx;

    std::vector<double> mic[xsr::NXS], lmp[xsr::NXS], xs_in[xsr::NXS];
    for (int xt = 0; xt < xsr::NXS; ++xt) mic[xt] = readDoubles(fi, xsr::NISO * ngn);
    std::vector<double> mic_ssm = readDoubles(fi, xsr::NISO * ssn);
    for (int xt = 0; xt < xsr::NXS; ++xt) lmp[xt] = readDoubles(fi, ngn);
    std::vector<double> lmp_ssm = readDoubles(fi, ssn);
    for (int xt = 0; xt < xsr::NXS; ++xt) xs_in[xt] = readDoubles(fi, ngn);
    std::vector<double> xs_ssm_in = readDoubles(fi, ssn);
    std::vector<double> iden_in   = readDoubles(fi, xsr::NISO * nx);
    std::vector<double> phif      = readDoubles(fi, nx * static_cast<size_t>(ng));
    std::fclose(fi);

    std::vector<double> ref_xs[xsr::NXS];
    for (int xt = 0; xt < xsr::NXS; ++xt) ref_xs[xt] = readDoubles(fo, ngn);
    std::vector<double> ref_ssm  = readDoubles(fo, ssn);
    std::vector<double> ref_iden = readDoubles(fo, xsr::NISO * nx);
    std::fclose(fo);

    const double* dep_i  = dep.data();
    const double* dep_xe = dep.data() + xsr::NISO;

    // ---- Stage 1: the Ieq/Xeeq chain.  Forms: condense (c), fissSource (f),
    // Xeeq numerator (n).  Score each combo by exact matches on the three
    // written iden rows across fuel nodes.
    std::printf("stage1: iden rows, combos (c,f) 0=plain 1=fma; n bit0=num-fma bit1=den-fma\n");
    for (int c = 0; c < 2; ++c)
        for (int f = 0; f < 2; ++f)
            for (int n = 0; n < 4; ++n) {
                size_t okI = 0, okXe = 0, okXem = 0, fuel_n = 0;
                for (int l = 0; l < nxyz; ++l) {
                    if (!isf[static_cast<size_t>(l)]) continue;
                    double absflux[2];
                    double raw = 0.0;
                    for (int ig = 0; ig < ng; ++ig) {
                        absflux[ig] = phif[static_cast<size_t>(l) * ng + ig] * norm_factor;
                        raw += absflux[ig];
                    }
                    if (raw <= 0.0) continue;
                    ++fuel_n;
                    const double invflux = 1.0 / raw;
                    const double sumflux = raw * 1.0e-24;

                    auto condense = [&](int iso, int xt) {
                        double sum = 0.0;
                        for (int ig = 0; ig < ng; ++ig)
                            sum = mac(c, mic[xt][(static_cast<size_t>(iso) * ng + ig) * nx + l],
                                      absflux[ig], sum);
                        return sum * invflux;
                    };

                    double fissI = 0.0, fissXe = 0.0;
                    for (int j = xsr::AC_FIRST; j <= xsr::AC_LAST; ++j) {
                        double xsff  = condense(j, xsr::T_XSFF);
                        double fRate = iden_in[static_cast<size_t>(j) * nx + l] * xsff * sumflux;
                        fissI  = mac(f, fRate, dep_i[j], fissI);
                        fissXe = mac(f, fRate, dep_xe[j], fissXe);
                    }
                    // n bit0: numerator lambdaI*Ieq+fissXe as fma; n bit1:
                    // denominator lambdaXe+sigaXe fused across the sigaXe
                    // temporary as fma(cond, sumflux, lambdaXe).
                    const double condXe = condense(xsr::XE135, xsr::T_XSAF);
                    double sigaXe = opaque(condXe * sumflux);
                    double Ieq    = fissI / xsr::LAMBDA_I;
                    double num    = (n & 1) ? std::fma(xsr::LAMBDA_I, Ieq, fissXe)
                                            : opaque(xsr::LAMBDA_I * Ieq) + fissXe;
                    double den    = (n & 2) ? std::fma(condXe, sumflux, xsr::LAMBDA_XE)
                                            : xsr::LAMBDA_XE + sigaXe;
                    double Xeeq   = num / den;
                    double Xem    = xsr::BR_I_TO_XE135M * xsr::LAMBDA_I * Ieq / xsr::LAMBDA_XEM;

                    if (!std::memcmp(&Ieq, &ref_iden[static_cast<size_t>(xsr::I135) * nx + l], 8)) ++okI;
                    if (!std::memcmp(&Xeeq, &ref_iden[static_cast<size_t>(xsr::XE135) * nx + l], 8)) ++okXe;
                    if (!std::memcmp(&Xem, &ref_iden[static_cast<size_t>(xsr::XE135M) * nx + l], 8)) ++okXem;
                }
                std::printf("  c=%d f=%d n=%d : I135 %zu/%zu  Xe135 %zu/%zu  Xe135m %zu/%zu\n",
                            c, f, n, okI, fuel_n, okXe, fuel_n, okXem, fuel_n);
            }

    // ---- Stage 2: the reconstruct accumulation, decoupled from stage 1 by
    // feeding PRODUCTION's own iden output.  Score on xs8 (FYLD, the widest
    // stage-1-independent... it does depend on iden, hence the decoupling)
    // and xs0/xs2.
    std::printf("stage2: recon val accumulation, using production iden\n");
    for (int r = 0; r < 2; ++r) {
        size_t ok0 = 0, ok2 = 0, ok8 = 0, tot = 0;
        for (int l = 0; l < nxyz; ++l) {
            if (!isf[static_cast<size_t>(l)]) continue;
            double raw = 0.0;
            for (int ig = 0; ig < ng; ++ig)
                raw += phif[static_cast<size_t>(l) * ng + ig] * norm_factor;
            if (raw <= 0.0) continue;
            for (int ig = 0; ig < ng; ++ig) {
                ++tot;
                for (int xt : {xsr::T_XSTF, xsr::T_XSAF, xsr::T_FYLD}) {
                    double val = lmp[xt][static_cast<size_t>(ig) * nx + l];
                    for (int iso = 0; iso < xsr::NISO; ++iso)
                        val = mac(r, mic[xt][(static_cast<size_t>(iso) * ng + ig) * nx + l],
                                  ref_iden[static_cast<size_t>(iso) * nx + l], val);
                    const double* ref = &ref_xs[xt][static_cast<size_t>(ig) * nx + l];
                    if (!std::memcmp(&val, ref, 8)) {
                        if (xt == xsr::T_XSTF) ++ok0;
                        else if (xt == xsr::T_XSAF) ++ok2;
                        else ++ok8;
                    }
                }
            }
        }
        std::printf("  r=%d : xstf %zu/%zu  xsaf %zu/%zu  fyld %zu/%zu\n", r, ok0,
                    tot, ok2, tot, ok8, tot);
    }
    return 0;
}
