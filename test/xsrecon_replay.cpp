// Replay the shared kernel body over a production capture
// (RASBERY_XSRECON_DUMP) and score it elementwise against the production CPU
// loop's own outputs.  Production data, production codegen on the reference
// side -- this closes the synthetic-coverage gap the harnesses cannot.
//
// Usage: rasbery_xsrecon_replay <dump-prefix>   (reads <p>.in and <p>.out)

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

std::vector<double> readDoubles(std::FILE* f, size_t n) {
    std::vector<double> v(n);
    if (std::fread(v.data(), sizeof(double), n, f) != n) {
        std::fprintf(stderr, "short read\n");
        std::exit(2);
    }
    return v;
}

struct Diff {
    const char* name;
    size_t      idx;
    double      ref, got;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <dump-prefix>\n", argv[0]);
        return 2;
    }
    const std::string prefix = argv[1];
    std::FILE*        fi     = std::fopen((prefix + ".in").c_str(), "rb");
    std::FILE*        fo     = std::fopen((prefix + ".out").c_str(), "rb");
    if (!fi || !fo) {
        std::printf("cannot open capture\n");
        return 2;
    }

    std::int64_t hdr[2];
    if (std::fread(hdr, sizeof hdr[0], 2, fi) != 2) return 2;
    const int ng = static_cast<int>(hdr[0]), nxyz = static_cast<int>(hdr[1]);
    if (ng != xsr::NG) {
        std::printf("ng mismatch\n");
        return 2;
    }
    double norm_factor, relax;
    if (std::fread(&norm_factor, sizeof(double), 1, fi) != 1) return 2;
    if (std::fread(&relax, sizeof(double), 1, fi) != 1) return 2;
    std::vector<char> isf(static_cast<size_t>(nxyz));
    if (std::fread(isf.data(), 1, isf.size(), fi) != isf.size()) return 2;
    std::vector<double> dep = readDoubles(fi, 2 * static_cast<size_t>(xsr::NISO));

    const size_t nx  = static_cast<size_t>(nxyz);
    const size_t ngn = static_cast<size_t>(ng) * nx;
    const size_t ssn = static_cast<size_t>(ng) * ng * nx;

    std::vector<double> mic[xsr::NXS], lmp[xsr::NXS], xs[xsr::NXS];
    for (int xt = 0; xt < xsr::NXS; ++xt) mic[xt] = readDoubles(fi, xsr::NISO * ngn);
    std::vector<double> mic_ssm = readDoubles(fi, xsr::NISO * ssn);
    for (int xt = 0; xt < xsr::NXS; ++xt) lmp[xt] = readDoubles(fi, ngn);
    std::vector<double> lmp_ssm = readDoubles(fi, ssn);
    for (int xt = 0; xt < xsr::NXS; ++xt) xs[xt] = readDoubles(fi, ngn);
    std::vector<double> xs_ssm = readDoubles(fi, ssn);
    std::vector<double> iden   = readDoubles(fi, xsr::NISO * nx);
    std::vector<double> phif   = readDoubles(fi, nx * static_cast<size_t>(ng));
    std::fclose(fi);

    std::vector<double> ref_xs[xsr::NXS];
    for (int xt = 0; xt < xsr::NXS; ++xt) ref_xs[xt] = readDoubles(fo, ngn);
    std::vector<double> ref_ssm  = readDoubles(fo, ssn);
    std::vector<double> ref_iden = readDoubles(fo, xsr::NISO * nx);
    double              ref_max;
    if (std::fread(&ref_max, sizeof(double), 1, fo) != 1) return 2;
    std::fclose(fo);

    std::vector<int> fuel;
    for (int l = 0; l < nxyz; ++l)
        if (isf[static_cast<size_t>(l)]) fuel.push_back(l);

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
    v.dep_i135    = dep.data();
    v.dep_xe135   = dep.data() + xsr::NISO;

    double max_change = 0.0;
    for (int i = 0; i < v.n_fuel; ++i) {
        double mc = 0.0;
        if (xsreconSolveNode(v, v.fuel[i], &mc))
            max_change = std::max(max_change, mc);
    }

    auto cmp = [&](const char* name, const std::vector<double>& got,
                   const std::vector<double>& ref, int& shown) {
        size_t n = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i], &ref[i], sizeof(double)) != 0) {
                ++n;
                if (shown < 12) {
                    std::int64_t bg, br;
                    std::memcpy(&bg, &got[i], 8);
                    std::memcpy(&br, &ref[i], 8);
                    const size_t l  = i % nx;
                    const size_t ig = i / nx;
                    std::printf("  DIFF %s[%zu] (l=%zu row=%zu): prod=%.17g "
                                "replay=%.17g ulp=%lld\n",
                                name, i, l, ig, ref[i], got[i],
                                static_cast<long long>(bg - br));
                    ++shown;
                }
            }
        }
        if (n) std::printf("%s: %zu/%zu differ\n", name, n, got.size());
        return n;
    };

    size_t total = 0;
    int    shown = 0;
    for (int xt = 0; xt < xsr::NXS; ++xt) {
        char nm[8];
        std::snprintf(nm, sizeof nm, "xs%d", xt);
        total += cmp(nm, xs[xt], ref_xs[xt], shown);
    }
    total += cmp("ssm", xs_ssm, ref_ssm, shown);
    total += cmp("iden", iden, ref_iden, shown);
    if (std::memcmp(&max_change, &ref_max, sizeof(double)) != 0) {
        std::printf("  DIFF max: prod=%.17g replay=%.17g\n", ref_max, max_change);
        ++total;
    }
    if (total)
        std::printf("REPLAY FAIL: %zu elements differ (fuel=%d relax=%g)\n", total,
                    v.n_fuel, relax);
    else
        std::printf("REPLAY PASS bit-identical (fuel=%d relax=%g)\n", v.n_fuel, relax);
    return total ? 1 : 0;
}
