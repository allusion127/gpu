// Device arm of the flat-XS replay gate: run the CUDA build of the shared
// body (FlatXsKernel.h, --fmad=false, StaticForms) over a RASBERY_FLATXS_DUMP
// capture and score elementwise ULP against the captured CPU outputs.
//
//   CUDA_VISIBLE_DEVICES=<uuid> ./flatxs_device_replay <capture-base>
//   CUDA_VISIBLE_DEVICES=<uuid> ./flatxs_device_replay <capture-base> --cta [T]
//   CUDA_VISIBLE_DEVICES=<uuid> ./flatxs_device_replay <capture-base> --one <i> [n]
//
// PASS (exit 0) means every element of every array the kernel writes is
// bit-identical to what the production gcc loop wrote.  This is the same
// verdict the host replay gives; running both separates "the body is wrong"
// from "nvcc rounds differently".
//
// --cta [T] adds WP5 stage B's CTA-per-node arm on its OWN copies of every
// mutable array and reports two extra verdicts: CTA vs the reference kernel
// (the B0 claim -- exactly 0 mismatches required, at every block size on the
// ladder) and CTA vs the capture.  T defaults to flatxs::CTA_THREADS_DEFAULT;
// run it at 64, 128 and 256, because a block size that changed the bytes would
// mean the lane-ownership invariant (P1 in FlatXsCtaKernel.cuh) is broken.
//
// WP21-B2: RASBERY_GPU_FLATXS_CTA_TILE=<n> selects the same tile the backend
// selects, THROUGH THE SAME LADDER HELPER, so this gate can certify the tiled
// arm.  The tile permutes lane ownership exactly as the block size does, so
// the pass condition is unchanged and stronger to read: run --cta at
// tile = 1, 2, 4 (and 8 on the FP32 workspace) and every one of them must
// report cta_vs_ref_mismatches = 0.

#include "../src/FlatXsCtaKernel.cuh"
#include "../src/FlatXsKernel.h"

#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fxs = rasbery::flatxs;
namespace xsr = rasbery::xsrecon;

#define TRY(expr)                                                          \
    do {                                                                   \
        const cudaError_t _e = (expr);                                     \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s -> %s\n", #expr, cudaGetErrorString(_e)); \
            return 3;                                                      \
        }                                                                  \
    } while (0)

namespace {

__global__ void kernelReplay(fxs::FlatXsView v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.n_nodes) return;
    fxs::flatxsSolveNode(v, i, fxs::StaticForms{});
}

// Debug: process exactly one node index with one thread, and (optionally)
// stop after `max_deltas` stream entries by shrinking node_cnt on the host.
__global__ void kernelReplayOne(fxs::FlatXsView v, int i) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        fxs::flatxsSolveNode(v, i, fxs::StaticForms{});
}

template <class T>
bool rd(std::FILE* f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    return std::fread(v.data(), sizeof(T), n, f) == n;
}

template <class T>
T* up(const std::vector<T>& v) {
    T* p = nullptr;
    if (v.empty()) return nullptr;
    cudaError_t e = cudaMalloc(&p, v.size() * sizeof(T));
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%zu bytes) -> %s\n",
                     v.size() * sizeof(T), cudaGetErrorString(e));
        std::abort();
    }
    e = cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "upload(%zu bytes) -> %s\n", v.size() * sizeof(T),
                     cudaGetErrorString(e));
        std::abort();
    }
    return p;
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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <capture-base> [--cta [T] | --one <i> [n]]\n",
                     argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    std::FILE*        f    = std::fopen((base + ".in").c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s.in\n", base.c_str()); return 2; }
    std::int64_t hdr[16];
    double       bavg;
    if (std::fread(hdr, 8, 16, f) != 16 || std::fread(&bavg, 8, 1, f) != 1)
        return 2;
    const std::int64_t ng = hdr[0], nxyz = hdr[1], niso = hdr[2], n_nodes = hdr[3],
                       stream_len = hdr[4], n_deltas = hdr[5], n_knots = hdr[6],
                       lmp_slot = hdr[7], lsm_n = hdr[8], mic_slot = hdr[9],
                       msm_n = hdr[10], has_micx = hdr[11], use_avg = hdr[12];
    if (ng != xsr::NG || niso != xsr::NISO || hdr[13] != 1) {
        std::fprintf(stderr, "unsupported capture\n");
        return 2;
    }
    const std::size_t nx  = static_cast<std::size_t>(nxyz);
    const std::size_t mic = static_cast<std::size_t>(xsr::NISO) * xsr::NG * nx;
    const std::size_t lmp = static_cast<std::size_t>(xsr::NG) * nx;
    const std::size_t msm = static_cast<std::size_t>(xsr::NISO) * xsr::NG * xsr::NG * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsr::NG) * xsr::NG * nx;

    std::vector<int>            nodes, off, cnt, sdid;
    std::vector<double>         sx, sscale, knots;
    std::vector<fxs::DeltaMeta> deltas;
    std::vector<double> coeff_lmp[fxs::N_ACTIVE], coeff_lsm,
        coeff_mic[fxs::N_ACTIVE], coeff_msm;
    std::vector<double> ref_mic[fxs::N_ACTIVE], ref_msm, ref_lmp[fxs::N_ACTIVE],
        ref_lsm;
    std::vector<double> wvfr, dmod, bppm, iden;
    std::vector<double> lmpx[fxs::N_ACTIVE], lsm, micx[fxs::N_ACTIVE], msmx;
    std::vector<double> xsa[xsr::NXS], xs_ssm;

    bool ok = rd(f, nodes, n_nodes) && rd(f, off, n_nodes) && rd(f, cnt, n_nodes) &&
              rd(f, sdid, stream_len) && rd(f, sx, stream_len) &&
              rd(f, sscale, stream_len) && rd(f, deltas, n_deltas) &&
              rd(f, knots, n_knots);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, coeff_lmp[t], lmp_slot);
    ok = ok && rd(f, coeff_lsm, lsm_n);
    if (has_micx) {
        for (int t = 0; ok && t < fxs::N_ACTIVE; ++t)
            ok = rd(f, coeff_mic[t], mic_slot);
        ok = ok && rd(f, coeff_msm, msm_n);
    }
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, ref_mic[t], mic);
    ok = ok && rd(f, ref_msm, msm);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, ref_lmp[t], lmp);
    ok = ok && rd(f, ref_lsm, ssm);
    ok = ok && rd(f, wvfr, nx) && rd(f, dmod, nx) && rd(f, bppm, nx) &&
         rd(f, iden, static_cast<std::size_t>(xsr::NISO) * nx);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, lmpx[t], lmp);
    ok = ok && rd(f, lsm, ssm);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, micx[t], mic);
    ok = ok && rd(f, msmx, msm);
    for (int xt = 0; ok && xt < xsr::NXS; ++xt) ok = rd(f, xsa[xt], lmp);
    ok = ok && rd(f, xs_ssm, ssm);
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "truncated .in\n"); return 2; }

    std::vector<double> out_lmp[fxs::N_ACTIVE], out_lsm, out_mic[fxs::N_ACTIVE],
        out_msm, out_xs[xsr::NXS], out_xs_ssm, out_iden3;
    f = std::fopen((base + ".out").c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s.out\n", base.c_str()); return 2; }
    std::int64_t ohdr[16];
    double       obavg;
    ok = std::fread(ohdr, 8, 16, f) == 16 && std::fread(&obavg, 8, 1, f) == 1;
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, out_lmp[t], lmp);
    ok = ok && rd(f, out_lsm, ssm);
    for (int t = 0; ok && t < fxs::N_ACTIVE; ++t) ok = rd(f, out_mic[t], mic);
    ok = ok && rd(f, out_msm, msm);
    for (int xt = 0; ok && xt < xsr::NXS; ++xt) ok = rd(f, out_xs[xt], lmp);
    ok = ok && rd(f, out_xs_ssm, ssm);
    ok = ok && rd(f, out_iden3, 3 * nx);
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "truncated .out\n"); return 2; }

    std::printf("shape: lmp_slot=%" PRId64 " lsm=%" PRId64 " mic_slot=%" PRId64
                " msm=%" PRId64 " knots=%" PRId64 " deltas=%" PRId64 "\n",
                lmp_slot, lsm_n, mic_slot, msm_n, n_knots, n_deltas);

    // --one <idx> [max_deltas]: single node, single thread, host-vs-device on
    // an optionally truncated stream -- bisects the first diverging entry.
    const bool one_mode = argc > 3 && std::string(argv[2]) == "--one";
    const int  one_idx  = one_mode ? std::atoi(argv[3]) : -1;
    // --cta [T]: run WP5's CTA-per-node arm alongside the reference kernel.
    const bool cta_mode = argc > 2 && std::string(argv[2]) == "--cta";
    const int  cta_threads =
        (cta_mode && argc > 3) ? std::atoi(argv[3]) : fxs::CTA_THREADS_DEFAULT;
    // WP21-B2.  Read from the environment rather than argv so the existing
    // positional grammar (`--cta [T]`) is untouched and every caller that
    // predates the tile keeps launching tile 1.  Resolved through the SAME
    // helper the backend calls, so the gate cannot certify a tile production
    // never runs.
    const int cta_tile = [&] {
        const char* e = std::getenv("RASBERY_GPU_FLATXS_CTA_TILE");
        const int   n = (e == nullptr) ? 1 : std::atoi(e);
        return fxs::flatxsCtaTileResolved(cta_threads, false, n > 0 ? n : 1);
    }();
    if (one_mode && argc > 4) {
        const int maxd = std::atoi(argv[4]);
        if (cnt[static_cast<std::size_t>(one_idx)] > maxd)
            cnt[static_cast<std::size_t>(one_idx)] = maxd;
    }

    // Host-tool arm for --one: same body, host build, on private copies.
    std::vector<double> h_lmp[fxs::N_ACTIVE], h_lsm, h_mic[fxs::N_ACTIVE], h_msm,
        h_xs[xsr::NXS], h_xs_ssm, h_iden;
    if (one_mode) {
        for (int t = 0; t < fxs::N_ACTIVE; ++t) { h_lmp[t] = lmpx[t]; h_mic[t] = micx[t]; }
        h_lsm = lsm; h_msm = msmx;
        for (int xt = 0; xt < xsr::NXS; ++xt) h_xs[xt] = xsa[xt];
        h_xs_ssm = xs_ssm; h_iden = iden;
        fxs::FlatXsView hv{};
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            hv.coeff_lmp[t] = coeff_lmp[t].data();
            hv.coeff_mic[t] = has_micx ? coeff_mic[t].data() : nullptr;
            hv.ref_lmp[t]   = ref_lmp[t].data();
            hv.ref_mic[t]   = ref_mic[t].data();
            hv.lmp[t]       = h_lmp[t].data();
            hv.mic[t]       = h_mic[t].data();
        }
        hv.coeff_lsm = coeff_lsm.data();
        hv.coeff_msm = has_micx ? coeff_msm.data() : nullptr;
        hv.knots     = knots.data();
        hv.deltas    = deltas.data();
        hv.has_coeff_micx = static_cast<int>(has_micx);
        hv.ref_lsm   = ref_lsm.data();
        hv.ref_msm   = ref_msm.data();
        hv.lsm       = h_lsm.data();
        hv.msm       = h_msm.data();
        for (int xt = 0; xt < xsr::NXS; ++xt) hv.xs[xt] = h_xs[xt].data();
        hv.xs_ssm = h_xs_ssm.data();
        hv.iden   = h_iden.data();
        hv.wvfr   = wvfr.data();
        hv.dmod   = dmod.data();
        hv.bppm   = bppm.data();
        hv.stream_did   = sdid.data();
        hv.stream_x     = sx.data();
        hv.stream_scale = sscale.data();
        hv.node_off     = off.data();
        hv.node_cnt     = cnt.data();
        hv.nodes        = nodes.data();
        hv.n_nodes      = static_cast<int>(n_nodes);
        hv.nxyz         = static_cast<int>(nxyz);
        hv.boron_dmod_average = bavg;
        hv.use_average_dmod   = static_cast<int>(use_avg);
        fxs::flatxsSolveNode(hv, one_idx, fxs::StaticForms{});
    }

    fxs::FlatXsView v{};
    for (int t = 0; t < fxs::N_ACTIVE; ++t) {
        v.coeff_lmp[t] = up(coeff_lmp[t]);
        v.coeff_mic[t] = has_micx ? up(coeff_mic[t]) : nullptr;
        v.ref_lmp[t]   = up(ref_lmp[t]);
        v.ref_mic[t]   = up(ref_mic[t]);
        v.lmp[t]       = up(lmpx[t]);
        v.mic[t]       = up(micx[t]);
    }
    v.coeff_lsm = up(coeff_lsm);
    v.coeff_msm = has_micx ? up(coeff_msm) : nullptr;
    v.knots     = up(knots);
    v.deltas    = up(deltas);
    v.has_coeff_micx = static_cast<int>(has_micx);
    v.ref_lsm   = up(ref_lsm);
    v.ref_msm   = up(ref_msm);
    v.lsm       = up(lsm);
    v.msm       = up(msmx);
    for (int xt = 0; xt < xsr::NXS; ++xt) v.xs[xt] = up(xsa[xt]);
    v.xs_ssm       = up(xs_ssm);
    v.iden         = up(iden);
    v.wvfr         = up(wvfr);
    v.dmod         = up(dmod);
    v.bppm         = up(bppm);
    v.stream_did   = up(sdid);
    v.stream_x     = up(sx);
    v.stream_scale = up(sscale);
    v.node_off     = up(off);
    v.node_cnt     = up(cnt);
    v.nodes        = up(nodes);
    v.n_nodes      = static_cast<int>(n_nodes);
    v.nxyz         = static_cast<int>(nxyz);
    v.boron_dmod_average = bavg;
    v.use_average_dmod   = static_cast<int>(use_avg);

    if (one_mode) {
        kernelReplayOne<<<1, 1>>>(v, one_idx);
        TRY(cudaGetLastError());
        TRY(cudaDeviceSynchronize());
        const int     l   = nodes[static_cast<std::size_t>(one_idx)];
        std::uint64_t bad = 0;
        std::vector<double> g;
        auto cmp1 = [&](const double* dv, const std::vector<double>& hv_,
                        std::size_t total, const char* tag) {
            g.resize(total);
            cudaMemcpy(g.data(), dv, total * sizeof(double), cudaMemcpyDeviceToHost);
            for (std::size_t r = 0; r < total / nx; ++r) {
                const std::uint64_t u = ulpDiff(g[r * nx + l], hv_[r * nx + l]);
                if (u) {
                    ++bad;
                    if (bad <= 6)
                        std::printf("  ONE %-6s row=%zu dev=%.17g host=%.17g ulp=%" PRIu64 "\n",
                                    tag, r, g[r * nx + l], hv_[r * nx + l], u);
                }
            }
        };
        char t2[16];
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            std::snprintf(t2, sizeof t2, "mic%d", t);
            cmp1(v.mic[t], h_mic[t], mic, t2);
        }
        cmp1(v.msm, h_msm, msm, "msm");
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            std::snprintf(t2, sizeof t2, "lmp%d", t);
            cmp1(v.lmp[t], h_lmp[t], lmp, t2);
        }
        cmp1(v.lsm, h_lsm, ssm, "lsm");
        std::printf("[flatxs_device_replay --one] idx=%d l=%d cnt=%d dev-vs-host "
                    "mismatches=%" PRIu64 "\n",
                    one_idx, l, cnt[static_cast<std::size_t>(one_idx)], bad);
        return bad == 0 ? 0 : 1;
    }

    const int block = 128;
    const int grid  = (v.n_nodes + block - 1) / block;
    kernelReplay<<<grid, block>>>(v);
    TRY(cudaGetLastError());
    TRY(cudaDeviceSynchronize());

    // --- WP5 stage C: the CTA arm, on its OWN copies of every mutable array.
    //
    // The point of a separate copy set is that the two arms must not be able
    // to help each other: `c` shares only the immutable library/reference
    // tables and the per-node inputs, and every array either kernel WRITES is
    // re-uploaded from the same captured pre-state.  So a three-way verdict is
    // possible from one run --
    //   (1) CTA vs REFERENCE KERNEL, per node, element for element -- the B0
    //       claim of FlatXsCtaKernel.cuh, and the one this file exists for;
    //   (2) CTA vs the captured gcc OUTPUT -- the same 0-ULP bar the
    //       thread-per-node arm already clears.
    // (1) is the sharper test: it fails on a lane-ownership or barrier bug
    // even on a deck where (2) happens to be insensitive.
    fxs::FlatXsView c = v;
    if (cta_mode) {
        for (int t = 0; t < fxs::N_ACTIVE; ++t) {
            c.lmp[t] = up(lmpx[t]);
            c.mic[t] = up(micx[t]);
        }
        c.lsm = up(lsm);
        c.msm = up(msmx);
        for (int xt = 0; xt < xsr::NXS; ++xt) c.xs[xt] = up(xsa[xt]);
        c.xs_ssm = up(xs_ssm);
        c.iden   = up(iden);
        fxs::flatxsCtaLaunch(c, cta_threads, nullptr, false, cta_tile);
        TRY(cudaGetLastError());
        TRY(cudaDeviceSynchronize());
    }

    auto pull = [&](const double* dv, std::vector<double>& hv_, std::size_t n) {
        hv_.resize(n);
        return cudaMemcpy(hv_.data(), dv, n * sizeof(double),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    std::uint64_t bad = 0, worst = 0;      // arm-vs-capture
    std::uint64_t cta_bad = 0, cta_worst = 0; // CTA-vs-capture
    std::uint64_t ab_bad = 0;              // CTA-vs-reference-kernel (must be 0)
    auto score = [&](const std::vector<double>& got,
                     const std::vector<double>& want, const char* tag,
                     std::uint64_t& bad_out, std::uint64_t& worst_out,
                     const char* what) {
        std::uint64_t abad = 0, aworst = 0;
        int           first_l = -1, first_r = -1;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const int l = nodes[i];
            for (std::size_t r = 0; r < got.size() / nx; ++r) {
                const std::uint64_t u = ulpDiff(got[r * nx + l], want[r * nx + l]);
                if (u) {
                    ++abad;
                    if (u > aworst) aworst = u;
                    if (first_l < 0) { first_l = l; first_r = static_cast<int>(r); }
                }
            }
        }
        if (abad) {
            std::printf("  [%s] %-8s mismatches=%-10" PRIu64 " worst_ulp=%" PRIu64
                        " first(l=%d,row=%d) got=%.17g want=%.17g\n",
                        what, tag, abad, aworst, first_l, first_r,
                        got[static_cast<std::size_t>(first_r) * nx + first_l],
                        want[static_cast<std::size_t>(first_r) * nx + first_l]);
        }
        bad_out += abad;
        if (aworst > worst_out) worst_out = aworst;
    };
    std::vector<double> h, hc;
    char tag[32];
    // One walk over every array either kernel writes.  `ref` is the reference
    // kernel's device pointer, `cta` the CTA arm's, `want` the capture.
    auto arm = [&](const double* ref_dv, const double* cta_dv,
                   const std::vector<double>& want, std::size_t n,
                   const char* tag_) -> bool {
        if (!pull(ref_dv, h, n)) return false;
        score(h, want, tag_, bad, worst, "ref-vs-capture");
        if (!cta_mode) return true;
        if (!pull(cta_dv, hc, n)) return false;
        std::uint64_t sink = 0;
        score(hc, h, tag_, ab_bad, sink, "cta-vs-ref");
        score(hc, want, tag_, cta_bad, cta_worst, "cta-vs-capture");
        return true;
    };
    for (int t = 0; t < fxs::N_ACTIVE; ++t) {
        std::snprintf(tag, sizeof tag, "lmp%d", t);
        if (!arm(v.lmp[t], c.lmp[t], out_lmp[t], lmp, tag)) return 3;
        std::snprintf(tag, sizeof tag, "mic%d", t);
        if (!arm(v.mic[t], c.mic[t], out_mic[t], mic, tag)) return 3;
    }
    if (!arm(v.lsm, c.lsm, out_lsm, ssm, "lsm")) return 3;
    if (!arm(v.msm, c.msm, out_msm, msm, "msm")) return 3;
    for (int xt = 0; xt < xsr::NXS; ++xt) {
        std::snprintf(tag, sizeof tag, "xs%d", xt);
        if (!arm(v.xs[xt], c.xs[xt], out_xs[xt], lmp, tag)) return 3;
    }
    if (!arm(v.xs_ssm, c.xs_ssm, out_xs_ssm, ssm, "xs_ssm")) return 3;
    if (!arm(v.iden, c.iden, out_iden3, 3 * nx, "iden3")) return 3;

    std::printf("[flatxs_device_replay] nodes=%" PRId64 " mismatches=%" PRIu64
                " worst_ulp=%" PRIu64 " -> %s\n",
                n_nodes, bad, worst, bad == 0 ? "PASS" : "FAIL");
    if (cta_mode) {
        std::printf("[flatxs_device_replay --cta threads=%d tile=%d] "
                    "cta_vs_ref_mismatches=%" PRIu64 "  "
                    "cta_vs_capture_mismatches=%" PRIu64 " worst_ulp=%" PRIu64
                    " -> %s\n",
                    cta_threads, cta_tile, ab_bad, cta_bad, cta_worst,
                    (ab_bad == 0 && cta_bad == 0) ? "PASS" : "FAIL");
        return (bad == 0 && ab_bad == 0 && cta_bad == 0) ? 0 : 1;
    }
    return bad == 0 ? 0 : 1;
}
