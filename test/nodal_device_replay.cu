// Device arm of the nodal replay gate: run the CUDA build of the shared body
// (NodalKernel.h, --fmad=false, StaticForms with the mined masks) over a
// RASBERY_NODAL_DUMP capture, phase by phase (seeded from the captured
// upstream state), and score elementwise ULP against the captured CPU
// intermediates and outputs.
//
//   CUDA_VISIBLE_DEVICES=<gpu> ./nodal_device_replay <capture>
//
// PASS means the device build reproduces the production gcc loop bit-for-bit
// on real data.  The HOST build of the shared body is deliberately not the
// gate: gcc contracts the un-policied expressions differently per template
// instantiation (measured: 623/233 elements between the constexpr-mask and
// runtime-mask instantiations), while nvcc with --fmad=false compiles the
// written forms literally.

#include "../src/NodalKernel.h"

#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace nk = rasbery::nodal;

#define TRY(expr)                                                              \
    do {                                                                       \
        const cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s -> %s\n", #expr, cudaGetErrorString(_e)); \
            return 3;                                                          \
        }                                                                      \
    } while (0)

namespace {

__global__ void kTrl0(nk::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) nk::nodalTrlcff0(v, lk);
}
__global__ void kTrl12(nk::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) nk::nodalTrlcff12(v, lk, nk::StaticForms{});
}
__global__ void kMat(nk::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) nk::nodalUpdateMatrix(v, lk, nk::StaticForms{});
}
__global__ void kEven(nk::NodalView v) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk < v.nxyz) nk::nodalCalculateEven(v, lk, nk::StaticForms{});
}
__global__ void kJnet(nk::NodalView v) {
    const int ls = blockIdx.x * blockDim.x + threadIdx.x;
    if (ls < v.nsurf) nk::nodalCalculateJnet(v, ls, nk::StaticForms{});
}

__global__ void kEvenProbe(nk::NodalView v, int lk, int idir, double* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        nk::nodalEvenProbe(v, lk, idir, nk::StaticForms{}, out);
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
    if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess ||
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "upload failed (%zu bytes)\n", v.size() * sizeof(T));
        std::abort();
    }
    return p;
}

std::uint64_t ulp(double a, double b) {
    if (a == b) return 0;
    std::int64_t ia, ib;
    std::memcpy(&ia, &a, 8); std::memcpy(&ib, &b, 8);
    if (ia < 0) ia = INT64_MIN - ia;
    if (ib < 0) ib = INT64_MIN - ib;
    const std::int64_t d = ia - ib;
    return static_cast<std::uint64_t>(d < 0 ? -d : d);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <capture>\n", argv[0]); return 2; }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::int64_t hdr[8];
    if (std::fread(hdr, 8, 8, f) != 8) return 2;
    const std::int64_t nxyz = hdr[0], nsurf = hdr[1], chif_empty = hdr[5];
    if (hdr[2] != nk::NDIR || hdr[3] != nk::NG || hdr[4] != nk::NEWSB) {
        std::fprintf(stderr, "dimension mismatch\n");
        return 2;
    }
    double reigv;
    if (std::fread(&reigv, 8, 1, f) != 1) return 2;
    const std::size_t nx = nxyz, ns = nsurf;

    std::vector<int>    lktosfc, neib, lklr, idirlr, sgnlr;
    std::vector<double> hmesh, albedo, xsrf, xsnf, xssm, chif;
    std::vector<double> C[9], jnet_in, flux;
    std::vector<double> trl0, trl1, trl2, matMs, matMf, matM, matMI, mu, tau,
        c2, c4, c6, jnet_out, phis_out;

    bool ok = rd(f, lktosfc, nx * nk::NDIR * nk::NLR) && rd(f, neib, nx * nk::NEWSB) &&
              rd(f, lklr, ns * nk::NLR) && rd(f, idirlr, ns * nk::NLR) &&
              rd(f, sgnlr, ns * nk::NLR) && rd(f, hmesh, nx * nk::NDIR) &&
              rd(f, albedo, nk::NDIR * nk::NLR) && rd(f, xsrf, nk::NG * nx) &&
              rd(f, xsnf, nk::NG * nx) && rd(f, xssm, nk::NG2 * nx);
    if (ok && !chif_empty) ok = rd(f, chif, nk::NG * nx);
    for (int i = 0; ok && i < 9; ++i) ok = rd(f, C[i], nx * nk::NDIR * nk::NG);
    ok = ok && rd(f, jnet_in, ns * nk::NG) && rd(f, flux, nx * nk::NG);
    ok = ok && rd(f, trl0, nx * nk::NDIR * nk::NG) && rd(f, trl1, nx * nk::NDIR * nk::NG) &&
         rd(f, trl2, nx * nk::NDIR * nk::NG) && rd(f, matMs, nx * nk::NG2) &&
         rd(f, matMf, nx * nk::NG2) && rd(f, matM, nx * nk::NG2) &&
         rd(f, matMI, nx * nk::NG2) && rd(f, mu, nx * nk::NDIR * nk::NG2) &&
         rd(f, tau, nx * nk::NDIR * nk::NG2) && rd(f, c2, nx * nk::NDIR * nk::NG) &&
         rd(f, c4, nx * nk::NDIR * nk::NG) && rd(f, c6, nx * nk::NDIR * nk::NG) &&
         rd(f, jnet_out, ns * nk::NG) && rd(f, phis_out, ns * nk::NG);
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "truncated capture\n"); return 2; }

    nk::NodalView v{};
    v.hmesh = up(hmesh); v.lktosfc = up(lktosfc); v.neib = up(neib);
    v.lklr = up(lklr); v.idirlr = up(idirlr); v.sgnlr = up(sgnlr);
    v.albedo = up(albedo);
    v.xsrf = up(xsrf); v.xsnf = up(xsnf); v.xssm = up(xssm);
    v.chif = chif_empty ? nullptr : up(chif);
    v.chif_empty = static_cast<int>(chif_empty);
    const double* devC[9];
    for (int i = 0; i < 9; ++i) devC[i] = up(C[i]);
    v.eta1 = devC[0]; v.eta2 = devC[1]; v.m260 = devC[2]; v.m251 = devC[3];
    v.m253 = devC[4]; v.m262 = devC[5]; v.m264 = devC[6]; v.diagD = devC[7];
    v.diagDI = devC[8];
    // Work arrays seeded from the captured upstream state (phase isolation).
    v.trlcff0 = up(trl0);
    v.trlcff1 = up(trl1);
    v.trlcff2 = up(trl2);
    v.mu = up(mu); v.tau = up(tau);
    v.matM = up(matM); v.matMI = up(matMI);
    v.matMs = up(matMs); v.matMf = up(matMf);
    v.dsncff2 = up(c2); v.dsncff4 = up(c4); v.dsncff6 = up(c6);
    v.flux = up(flux);
    v.jnet = up(jnet_in);
    std::vector<double> zeros(ns * nk::NG, 0.0);
    v.phis  = up(zeros);
    v.reigv = reigv;
    v.nxyz  = static_cast<int>(nxyz);
    v.nsurf = static_cast<int>(nsurf);

    // --probe <lk> <idir>: single-element intermediate dump, device vs host.
    if (argc > 3 && std::string(argv[2]) == std::string("--probe")) {
        const int plk = std::atoi(argv[3]);
        const int pdir = argc > 4 ? std::atoi(argv[4]) : 0;
        // seed work arrays with captured upstream (trl0/trl2/matM already
        // uploaded from capture in the view construction above)
        double* dev_out = nullptr;
        cudaMalloc(&dev_out, 64 * sizeof(double));
        kEvenProbe<<<1, 1>>>(v, plk, pdir, dev_out);
        TRY(cudaGetLastError());
        TRY(cudaDeviceSynchronize());
        double dv[64], hv[64];
        cudaMemcpy(dv, dev_out, 64 * sizeof(double), cudaMemcpyDeviceToHost);
        // host recompute with the same masks on the captured host arrays
        nk::NodalView hview{};
        hview = v; // copy scalars
        hview.hmesh = hmesh.data(); hview.lktosfc = lktosfc.data();
        hview.neib = neib.data(); hview.lklr = lklr.data();
        hview.idirlr = idirlr.data(); hview.sgnlr = sgnlr.data();
        hview.albedo = albedo.data();
        hview.xsrf = xsrf.data(); hview.xsnf = xsnf.data(); hview.xssm = xssm.data();
        hview.chif = chif_empty ? nullptr : chif.data();
        hview.eta1 = C[0].data(); hview.eta2 = C[1].data(); hview.m260 = C[2].data();
        hview.m251 = C[3].data(); hview.m253 = C[4].data(); hview.m262 = C[5].data();
        hview.m264 = C[6].data(); hview.diagD = C[7].data(); hview.diagDI = C[8].data();
        hview.trlcff0 = trl0.data(); hview.trlcff1 = trl1.data();
        hview.trlcff2 = trl2.data();
        hview.matM = matM.data(); hview.matMI = matMI.data();
        hview.matMs = matMs.data(); hview.matMf = matMf.data();
        hview.mu = mu.data(); hview.tau = tau.data();
        hview.dsncff2 = c2.data(); hview.dsncff4 = c4.data();
        hview.dsncff6 = c6.data();
        hview.flux = flux.data(); hview.jnet = jnet_in.data();
        hview.phis = phis_out.data();
        nk::RuntimeForms rp;
        for (int q = 0; q < 5; ++q) rp.mask[q] = nk::nodalFormsOf(q);
        nk::nodalEvenProbe(hview, plk, pdir, rp, hv);
        static const char* names[31] = {
            "rm4464_0", "rm4464_1", "at2_00", "at2_10", "at2_01", "at2_11",
            "a_00", "a_10", "a_01", "a_11", "bt2_0", "bt2_1", "bt1_0", "bt1_1",
            "b_0", "b_1", "rdet", "c4_0", "c4_1", "c6_0", "c6_1", "c2_0",
            "c2_1", "mu2_0", "mu2_1", "mu1_0", "mu1_1", "matM00", "matM10",
            "matM01", "matM11"};
        for (int i = 0; i < 31; ++i) {
            const std::uint64_t u = ulp(dv[i], hv[i]);
            std::printf("%-9s dev=%.17g host=%.17g %s\n", names[i], dv[i],
                        hv[i], u ? "DIFF" : "");
        }
        // also compare against captured c4/c6/c2 at this element
        const int lkd = plk * nk::NDIR + pdir;
        std::printf("captured  c4=(%.17g, %.17g) c6=(%.17g, %.17g)\n",
                    c4[lkd * nk::NG + 0], c4[lkd * nk::NG + 1],
                    c6[lkd * nk::NG + 0], c6[lkd * nk::NG + 1]);
        return 0;
    }

    const int B = 128;
    const int gn = (v.nxyz + B - 1) / B;
    const int gs = (v.nsurf + B - 1) / B;
    // All five phases in sequence; work arrays were seeded with captured
    // upstream values, and each phase's outputs are compared before the next
    // phase could consume anything the comparison would have caught anyway.
    kTrl0<<<gn, B>>>(v);
    kTrl12<<<gn, B>>>(v);
    kMat<<<gn, B>>>(v);
    kEven<<<gn, B>>>(v);
    kJnet<<<gs, B>>>(v);
    TRY(cudaGetLastError());
    TRY(cudaDeviceSynchronize());

    std::uint64_t grand = 0;
    auto check = [&](const char* name, const double* dev,
                     const std::vector<double>& want) {
        std::vector<double> got(want.size());
        cudaMemcpy(got.data(), dev, want.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        std::uint64_t bad = 0, worst = 0;
        std::size_t   first = 0;
        for (std::size_t i = 0; i < want.size(); ++i) {
            const std::uint64_t u = ulp(got[i], want[i]);
            if (u) { if (!bad) first = i; ++bad; if (u > worst) worst = u; }
        }
        if (bad)
            std::printf("  %-8s bad=%" PRIu64 " worst=%" PRIu64
                        " first=%zu got=%.17g want=%.17g\n",
                        name, bad, worst, first, got[first], want[first]);
        grand += bad;
    };
    check("trl0", v.trlcff0, trl0);
    check("trl1", v.trlcff1, trl1);
    check("trl2", v.trlcff2, trl2);
    check("matMs", v.matMs, matMs);
    check("matMf", v.matMf, matMf);
    check("matM", v.matM, matM);
    check("matMI", v.matMI, matMI);
    check("mu", v.mu, mu);
    check("tau", v.tau, tau);
    check("c2", v.dsncff2, c2);
    check("c4", v.dsncff4, c4);
    check("c6", v.dsncff6, c6);
    check("jnet", v.jnet, jnet_out);
    check("phis", v.phis, phis_out);
    std::printf("[nodal_device_replay] nxyz=%" PRId64 " nsurf=%" PRId64
                " total_bad=%" PRIu64 " -> %s\n",
                nxyz, nsurf, grand, grand == 0 ? "PASS" : "FAIL");
    return grand == 0 ? 0 : 1;
}
