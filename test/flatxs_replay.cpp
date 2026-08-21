// Replay gate + contraction-form miner for the flat-XS device arm.
//
//   ./flatxs_replay <capture-base>            replay with the production forms
//                                             (FLATXS_FORMS) and score ULP
//   ./flatxs_replay <capture-base> --sweep    try all 1024 contraction masks,
//                                             print every exact-match mask
//
// <capture-base>.in / .out come from a production run with
// RASBERY_FLATXS_DUMP=<capture-base> (see XSSet.cpp flatxsDumpState).  The
// capture carries the resolved delta stream, so this tool scores the APPLY
// side of the split: the shared body of FlatXsKernel.h against the bytes the
// production gcc loop actually produced.  PASS means 0 ULP everywhere.

#include "../src/FlatXsKernel.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fxs = rasbery::flatxs;
namespace xsr = rasbery::xsrecon;

namespace {

struct Capture {
    std::int64_t ng = 0, nxyz = 0, niso = 0, n_nodes = 0, stream_len = 0;
    std::int64_t n_deltas = 0, n_knots = 0, lmp_slot = 0, lsm_n = 0,
                 mic_slot = 0, msm_n = 0, has_micx = 0, use_avg = 0;
    double boron_avg = 0.0;

    std::vector<int>            nodes, off, cnt, sdid;
    std::vector<double>         sx, sscale, knots;
    std::vector<fxs::DeltaMeta> deltas;
    std::vector<double> coeff_lmp[fxs::N_ACTIVE], coeff_lsm;
    std::vector<double> coeff_mic[fxs::N_ACTIVE], coeff_msm;
    std::vector<double> ref_mic[fxs::N_ACTIVE], ref_msm;
    std::vector<double> ref_lmp[fxs::N_ACTIVE], ref_lsm;
    std::vector<double> wvfr, dmod, bppm, iden;
    std::vector<double> lmp[fxs::N_ACTIVE], lsm, mic[fxs::N_ACTIVE], msm;
    std::vector<double> xs[xsr::NXS], xs_ssm;

    // .out payload
    std::vector<double> out_lmp[fxs::N_ACTIVE], out_lsm;
    std::vector<double> out_mic[fxs::N_ACTIVE], out_msm;
    std::vector<double> out_xs[xsr::NXS], out_xs_ssm;
    std::vector<double> out_iden3;
};

bool readBlock(std::FILE* f, std::vector<double>& v, std::size_t n) {
    v.resize(n);
    return std::fread(v.data(), sizeof(double), n, f) == n;
}
bool readBlock(std::FILE* f, std::vector<int>& v, std::size_t n) {
    v.resize(n);
    return std::fread(v.data(), sizeof(int), n, f) == n;
}

bool loadCapture(const std::string& base, Capture& c) {
    std::FILE* f = std::fopen((base + ".in").c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s.in\n", base.c_str()); return false; }
    std::int64_t hdr[16];
    double       bavg;
    if (std::fread(hdr, sizeof hdr[0], 16, f) != 16 ||
        std::fread(&bavg, sizeof bavg, 1, f) != 1) { std::fclose(f); return false; }
    c.ng = hdr[0]; c.nxyz = hdr[1]; c.niso = hdr[2]; c.n_nodes = hdr[3];
    c.stream_len = hdr[4]; c.n_deltas = hdr[5]; c.n_knots = hdr[6];
    c.lmp_slot = hdr[7]; c.lsm_n = hdr[8]; c.mic_slot = hdr[9]; c.msm_n = hdr[10];
    c.has_micx = hdr[11]; c.use_avg = hdr[12];
    c.boron_avg = bavg;
    if (c.ng != xsr::NG || c.niso != xsr::NISO || hdr[13] != 1) {
        std::fprintf(stderr, "capture dimensions unsupported (ng=%" PRId64
                             " niso=%" PRId64 " full=%" PRId64 ")\n",
                     c.ng, c.niso, hdr[13]);
        std::fclose(f);
        return false;
    }
    const std::size_t nx  = static_cast<std::size_t>(c.nxyz);
    const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

    bool ok = readBlock(f, c.nodes, c.n_nodes) && readBlock(f, c.off, c.n_nodes) &&
              readBlock(f, c.cnt, c.n_nodes) && readBlock(f, c.sdid, c.stream_len) &&
              readBlock(f, c.sx, c.stream_len) && readBlock(f, c.sscale, c.stream_len);
    c.deltas.resize(static_cast<std::size_t>(c.n_deltas));
    ok = ok && std::fread(c.deltas.data(), sizeof(fxs::DeltaMeta),
                          c.deltas.size(), f) == c.deltas.size();
    ok = ok && readBlock(f, c.knots, c.n_knots);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t)
        ok = readBlock(f, c.coeff_lmp[t], c.lmp_slot);
    ok = ok && readBlock(f, c.coeff_lsm, c.lsm_n);
    if (c.has_micx) {
        for (int t = 0; ok && t < fxs::N_ACTIVE; ++t)
            ok = readBlock(f, c.coeff_mic[t], c.mic_slot);
        ok = ok && readBlock(f, c.coeff_msm, c.msm_n);
    }
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = readBlock(f, c.ref_mic[t], mic);
    ok = ok && readBlock(f, c.ref_msm, msm);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = readBlock(f, c.ref_lmp[t], lmp);
    ok = ok && readBlock(f, c.ref_lsm, ssm);
    ok = ok && readBlock(f, c.wvfr, nx) && readBlock(f, c.dmod, nx) &&
         readBlock(f, c.bppm, nx) &&
         readBlock(f, c.iden, static_cast<std::size_t>(xsr::NISO) * nx);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = readBlock(f, c.lmp[t], lmp);
    ok = ok && readBlock(f, c.lsm, ssm);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = readBlock(f, c.mic[t], mic);
    ok = ok && readBlock(f, c.msm, msm);
    for (int xt = 0; ok && xt < xsr::NXS; ++xt) ok = readBlock(f, c.xs[xt], lmp);
    ok = ok && readBlock(f, c.xs_ssm, ssm);
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "truncated %s.in\n", base.c_str()); return false; }

    f = std::fopen((base + ".out").c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s.out\n", base.c_str()); return false; }
    std::int64_t ohdr[16];
    double       obavg;
    ok = std::fread(ohdr, sizeof ohdr[0], 16, f) == 16 &&
         std::fread(&obavg, sizeof obavg, 1, f) == 1;
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = readBlock(f, c.out_lmp[t], lmp);
    ok = ok && readBlock(f, c.out_lsm, ssm);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = readBlock(f, c.out_mic[t], mic);
    ok = ok && readBlock(f, c.out_msm, msm);
    for (int xt = 0; ok && xt < xsr::NXS; ++xt) ok = readBlock(f, c.out_xs[xt], lmp);
    ok = ok && readBlock(f, c.out_xs_ssm, ssm);
    ok = ok && readBlock(f, c.out_iden3, 3 * nx);
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "truncated %s.out\n", base.c_str()); return false; }
    return true;
}

struct Work {
    std::vector<double> lmp[fxs::N_ACTIVE], lsm, mic[fxs::N_ACTIVE], msm;
    std::vector<double> xs[xsr::NXS], xs_ssm, iden;
};

fxs::FlatXsView makeView(Capture& c, Work& w) {
    fxs::FlatXsView v{};
    for (int t = 0; t < fxs::N_ACTIVE; ++t) {
        v.coeff_lmp[t] = c.coeff_lmp[t].data();
        v.coeff_mic[t] = c.has_micx ? c.coeff_mic[t].data() : nullptr;
        v.ref_lmp[t]   = c.ref_lmp[t].data();
        v.ref_mic[t]   = c.ref_mic[t].data();
        v.lmp[t]       = w.lmp[t].data();
        v.mic[t]       = w.mic[t].data();
    }
    v.coeff_lsm = c.coeff_lsm.data();
    v.coeff_msm = c.has_micx ? c.coeff_msm.data() : nullptr;
    v.knots     = c.knots.data();
    v.deltas    = c.deltas.data();
    v.has_coeff_micx = static_cast<int>(c.has_micx);
    v.ref_lsm  = c.ref_lsm.data();
    v.ref_msm  = c.ref_msm.data();
    v.lsm      = w.lsm.data();
    v.msm      = w.msm.data();
    for (int xt = 0; xt < xsr::NXS; ++xt) v.xs[xt] = w.xs[xt].data();
    v.xs_ssm = w.xs_ssm.data();
    v.iden   = w.iden.data();
    v.wvfr   = c.wvfr.data();
    v.dmod   = c.dmod.data();
    v.bppm   = c.bppm.data();
    v.stream_did   = c.sdid.data();
    v.stream_x     = c.sx.data();
    v.stream_scale = c.sscale.data();
    v.node_off     = c.off.data();
    v.node_cnt     = c.cnt.data();
    v.nodes        = c.nodes.data();
    v.n_nodes      = static_cast<int>(c.n_nodes);
    v.nxyz         = static_cast<int>(c.nxyz);
    v.boron_dmod_average = c.boron_avg;
    v.use_average_dmod   = static_cast<int>(c.use_avg);
    return v;
}

void resetWork(const Capture& c, Work& w) {
    for (int t = 0; t < fxs::N_ACTIVE; ++t) {
        w.lmp[t] = c.lmp[t];
        w.mic[t] = c.mic[t];
    }
    w.lsm = c.lsm;
    w.msm = c.msm;
    for (int xt = 0; xt < xsr::NXS; ++xt) w.xs[xt] = c.xs[xt];
    w.xs_ssm = c.xs_ssm;
    w.iden   = c.iden;
}

std::uint64_t ulpDiff(double a, double b) {
    if (a == b) return 0;
    std::int64_t ia, ib;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    if (ia < 0) ia = INT64_MIN - ia;
    if (ib < 0) ib = INT64_MIN - ib;
    const std::int64_t d = ia - ib;
    return static_cast<std::uint64_t>(d < 0 ? -d : d);
}

std::uint64_t scoreColumn(const std::vector<double>& got,
                          const std::vector<double>& want, std::size_t stride,
                          std::size_t nxyz, int l, std::uint64_t& worst) {
    std::uint64_t bad = 0;
    for (std::size_t r = 0; r < got.size() / nxyz; ++r) {
        const std::uint64_t u = ulpDiff(got[r * nxyz + l], want[r * nxyz + l]);
        if (u) { ++bad; if (u > worst) worst = u; }
    }
    (void)stride;
    return bad;
}

/// Score one mask: elementwise vs the .out capture, target columns only
/// (non-target columns are untouched copies of the .in state on both sides).
template <class POL>
std::uint64_t runAndScore(Capture& c, Work& w, const POL& pol,
                          std::uint64_t* worst_out, bool verbose) {
    resetWork(c, w);
    fxs::FlatXsView v = makeView(c, w);
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < v.n_nodes; ++i)
        fxs::flatxsSolveNode(v, i, pol);

    const std::size_t nx    = static_cast<std::size_t>(c.nxyz);
    std::uint64_t     bad   = 0;
    std::uint64_t     worst = 0;
    for (int i = 0; i < v.n_nodes; ++i) {
        const int l = c.nodes[static_cast<std::size_t>(i)];
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            bad += scoreColumn(w.lmp[t], c.out_lmp[t], 0, nx, l, worst);
            bad += scoreColumn(w.mic[t], c.out_mic[t], 0, nx, l, worst);
        }
        bad += scoreColumn(w.lsm, c.out_lsm, 0, nx, l, worst);
        bad += scoreColumn(w.msm, c.out_msm, 0, nx, l, worst);
        for (int xt = 0; xt < xsr::NXS; ++xt)
            bad += scoreColumn(w.xs[xt], c.out_xs[xt], 0, nx, l, worst);
        bad += scoreColumn(w.xs_ssm, c.out_xs_ssm, 0, nx, l, worst);
        for (int iso = 0; iso < 3; ++iso) {
            const std::uint64_t u =
                ulpDiff(w.iden[iso * nx + l], c.out_iden3[iso * nx + l]);
            if (u) { ++bad; if (u > worst) worst = u; }
        }
    }
    if (worst_out) *worst_out = worst;
    if (verbose)
        std::printf("mismatched elements: %" PRIu64 "  worst ULP: %" PRIu64 "\n",
                    bad, worst);
    return bad;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <capture-base> [--sweep]\n", argv[0]);
        return 2;
    }
    Capture c;
    if (!loadCapture(argv[1], c)) return 2;
    std::printf("capture: nxyz=%" PRId64 " nodes=%" PRId64 " stream=%" PRId64
                " deltas=%" PRId64 " micx=%" PRId64 "\n",
                c.nxyz, c.n_nodes, c.stream_len, c.n_deltas, c.has_micx);

    Work w;
    const bool sweep = argc > 2 && std::string(argv[2]) == "--sweep";
    if (!sweep) {
        std::uint64_t worst = 0;
        const std::uint64_t bad =
            runAndScore(c, w, fxs::StaticForms{}, &worst, true);
        std::printf("[flatxs_replay] production mask 0x%03x: %s\n",
                    fxs::FLATXS_FORMS, bad == 0 ? "PASS" : "FAIL");
        return bad == 0 ? 0 : 1;
    }

    // Sweep every contraction mask; a capture with real branch activity
    // constrains each exercised bit.  Bits that never fire stay ambiguous and
    // every mask differing only there ties -- report all winners.
    std::vector<unsigned> winners;
    for (unsigned mask = 0; mask < 1u << 10; ++mask) {
        fxs::RuntimeForms pol;
        pol.mask = mask;
        if (runAndScore(c, w, pol, nullptr, false) == 0)
            winners.push_back(mask);
        if ((mask & 63u) == 63u)
            std::fprintf(stderr, "\rsweep %u/1024, winners so far %zu",
                         mask + 1, winners.size());
    }
    std::fprintf(stderr, "\n");
    if (winners.empty()) {
        std::printf("[flatxs_replay] sweep: NO exact mask -- the body itself "
                    "diverges from production; fix the algorithm first\n");
        return 1;
    }
    unsigned fixed_and = ~0u, fixed_or = 0;
    for (unsigned m : winners) { fixed_and &= m; fixed_or |= m; }
    std::printf("[flatxs_replay] sweep winners: %zu masks; constrained bits: "
                "set=0x%03x clear=0x%03x ambiguous=0x%03x\n",
                winners.size(), fixed_and, ~fixed_or & 0x3FF,
                (fixed_or ^ fixed_and) & 0x3FF);
    for (unsigned m : winners)
        std::printf("  winner mask 0x%03x\n", m);
    return 0;
}
