// Task 4 N1 gate: run the REAL phase -- k_nodal_update_constant plus its
// publish kernel, through a real DevicePhaseQueue -- against the CPU reference
// dump that test/nodal_constant_gpu_replay.cpp writes.
//
//   ./rasbery_nodal_constant_device_replay <dump>
//
// WHAT AN N1 PHASE OWES (Sec 9.1).  Not bit-identity with the CPU -- that is
// B0, and Task 4 Step 0 measured that it is unreachable here (device exp
// differs from glibc exp on ~5% of the arguments this body evaluates, by 1
// ulp).  What N1 owes is:
//
//   (1) RUN-TO-RUN BIT DETERMINISM on the device.  Two launches of the same
//       phase over the same inputs must produce byte-identical arrays.  This is
//       the property the whole campaign's A/B comparisons rest on, and it is
//       the one a reduction, an atomic or a race would break.  Checked by
//       running the phase twice into two separate buffers.
//   (2) THE DEVIATION IS THE TRANSCENDENTAL AND NOTHING ELSE.  Two checks, and
//       the second is the sharp one:
//         (a) diagD and diagDI are 4*xsdf/(h*h) and its reciprocal -- neither
//             touches sqrt or exp, so both must be bit-identical to the host;
//         (b) ATTRIBUTION.  This file recomputes exp(sqrt(kp2)) on the device
//             for every (node, dir, group) and compares it to glibc's, giving
//             the exact set of sites where the library disagrees.  EVERY
//             differing coefficient element must lie in that set.  A
//             coefficient that differs where exp AGREED is not the N1 library
//             deviation -- it is an unpinned contraction, and that is a bug,
//             not a classification.
//       Check (b) is what caught the real defect during Task 4: before the
//       contraction mask was mined into NodalConstantKernel.h, 95% of eta1
//       differed while exp only disagreed on 5.7% of sites.
//   (3) THE AMPLIFICATION IS REPORTED, NOT BOUNDED BY A GUESS.  The nodal
//       coefficients are cancellation-heavy (bfcff4's numerator cancels to
//       O(kp^9)), so a 1-ulp exp does NOT stay a 1-ulp output: measured up to
//       ~3e-2 RELATIVE on the worst-conditioned nodes of the synthetic deck.
//       That is a property of the algorithm, identical on both arms, and it is
//       exactly why Sec 6.1 classifies this phase as trajectory-changing.  The
//       number is printed so Task 22's Gate A/B knows what it is looking at.
//
// The MAPPING (packing order, indexing, early-out scope) is not re-tested here:
// test/nodal_constant_gpu_replay.cpp settles it on the host, without a GPU, so
// a failure in this file is unambiguously about the device.
//
// Build: --fmad=false, exactly like the production TU.  Without it (2) fails,
// which is the point of (2).

#include "../src/CudaNodalConstantKernel.h"

#include <cuda_runtime.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ng_ = rasbery::gpu;

#define TRY(expr)                                                              \
    do {                                                                       \
        const cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s -> %s\n", #expr, cudaGetErrorString(_e)); \
            return 3;                                                          \
        }                                                                      \
    } while (0)

namespace {

/// Probe kernel: the two library calls, on the device, for every
/// (node, dir, group).  Deliberately NOT the coefficient body -- it exists only
/// to name the sites where the library disagrees, so it must not be able to
/// disagree with the body about anything else.  kp2 is basic arithmetic with no
/// add in it, so host and device compute it bit-identically.
__global__ void k_probe_exp(const double* __restrict__ xs, const double* __restrict__ hmesh,
                            double* __restrict__ out_exp, int nxyz, int ng) {
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nxyz * 3 * ng;
    if (tid >= total) return;
    const int ig   = tid % ng;
    const int idir = (tid / ng) % 3;
    const int lk   = tid / (ng * 3);
    const double xsrf = xs[rasbery::gpu::macroXsIndex(rasbery::gpu::kXtXsrf, ig, lk, ng, nxyz)];
    const double xsdf = xs[rasbery::gpu::macroXsIndex(rasbery::gpu::kXtXsdf, ig, lk, ng, nxyz)];
    const double h    = hmesh[lk * 3 + idir];
    out_exp[tid]      = exp(sqrt(xsrf * h * h / (4 * xsdf)));
}

std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

long long ulpDistance(double a, double b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return -1;
    long long x = static_cast<long long>(bits(a));
    long long y = static_cast<long long>(bits(b));
    if (x < 0) x = static_cast<long long>(0x8000000000000000ull) - x;
    if (y < 0) y = static_cast<long long>(0x8000000000000000ull) - y;
    const long long d = x - y;
    return d < 0 ? -d : d;
}

const char* kArrayName[ng_::kNcCount] = {"eta1", "eta2",   "m260",   "m251", "m253",
                                         "m262", "m264",   "diagDI", "diagD"};

struct Dump {
    int                 nxyz = 0, ndir = 0, ng = 0;
    std::vector<double> xs, hmesh, ref_const, ref_cache;
};

bool load(const char* path, Dump& d) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::int64_t hdr[4];
    if (std::fread(hdr, sizeof hdr[0], 4, f) != 4) { std::fclose(f); return false; }
    d.nxyz = static_cast<int>(hdr[0]);
    d.ndir = static_cast<int>(hdr[1]);
    d.ng   = static_cast<int>(hdr[2]);
    const int narr = static_cast<int>(hdr[3]);
    if (d.nxyz <= 0 || d.ndir != 3 || d.ng != 2 || narr != ng_::kNcCount) {
        std::fclose(f);
        return false;
    }
    auto rd = [&](std::vector<double>& v, size_t n) {
        v.resize(n);
        return std::fread(v.data(), sizeof(double), n, f) == n;
    };
    const size_t nx = static_cast<size_t>(d.nxyz);
    const bool   ok = rd(d.xs, static_cast<size_t>(ng_::kDevNxs) * d.ng * nx) &&
                    rd(d.hmesh, nx * d.ndir) &&
                    rd(d.ref_const, static_cast<size_t>(narr) * nx * d.ndir * d.ng) &&
                    rd(d.ref_cache, 2 * nx * d.ng);
    std::fclose(f);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <dump>   (written by rasbery_nodal_constant_gpu_replay "
                     "--dump)\n",
                     argv[0]);
        return 2;
    }
    Dump d;
    if (!load(argv[1], d)) {
        std::fprintf(stderr, "cannot read dump %s\n", argv[1]);
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("nodal constant device replay: SKIP (no CUDA device)\n");
        return 0;
    }

    const size_t n_const = d.ref_const.size();
    const size_t n_cache = d.ref_cache.size();

    // --- device state -------------------------------------------------------
    // The slot's arrays plus the arena/queue plumbing the kernel resolves
    // through.  Two independent output sets, so run-to-run determinism is a
    // comparison and not a re-read of the same memory.
    double *d_xs = nullptr, *d_hmesh = nullptr;
    double *d_const_a = nullptr, *d_const_b = nullptr;
    double *d_cache_a = nullptr, *d_cache_b = nullptr;
    TRY(cudaMalloc(&d_xs, sizeof(double) * d.xs.size()));
    TRY(cudaMalloc(&d_hmesh, sizeof(double) * d.hmesh.size()));
    TRY(cudaMalloc(&d_const_a, sizeof(double) * n_const));
    TRY(cudaMalloc(&d_const_b, sizeof(double) * n_const));
    TRY(cudaMalloc(&d_cache_a, sizeof(double) * n_cache));
    TRY(cudaMalloc(&d_cache_b, sizeof(double) * n_cache));
    TRY(cudaMemcpy(d_xs, d.xs.data(), sizeof(double) * d.xs.size(), cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(d_hmesh, d.hmesh.data(), sizeof(double) * d.hmesh.size(),
                   cudaMemcpyHostToDevice));

    // The cache starts at quiet NaN, exactly as Nodal's does (Nodal.cpp:69-70),
    // so the first pass recomputes every node.
    std::vector<double> nan_cache(n_cache, std::nan(""));
    std::vector<double> zero_const(n_const, 0.0);

    ng_::DeviceSlotView h_view{};
    h_view.nxyz = d.nxyz;
    h_view.ng   = d.ng;
    h_view.xs   = d_xs;

    ng_::DeviceGeometryView h_geom{};
    h_geom.hmesh = d_hmesh;
    h_geom.nxyz  = d.nxyz;
    h_geom.ng    = d.ng;

    // Slot 3 of a 4-slot fleet, queued as the only entry: exercising a non-zero
    // slot id is the cheap way to catch a kernel that used blockIdx.y as a slot
    // instead of an index into the queue.
    constexpr int kSlots     = 4;
    constexpr int kUsedSlot  = 3;
    ng_::DevicePhaseQueue h_queue{};
    for (int i = 0; i < ng_::kMaxSchedulerSlots; ++i) h_queue.slots[i] = ng_::kQueueEmptySlot;
    h_queue.slots[0] = kUsedSlot;
    h_queue.count    = 1;
    h_queue.bucket   = ng_::gpuSelectBucket(1);

    ng_::DeviceSlotView* d_views = nullptr;
    ng_::DeviceSlotState* d_states = nullptr;
    TRY(cudaMalloc(&d_views, sizeof(ng_::DeviceSlotView) * kSlots));
    TRY(cudaMalloc(&d_states, sizeof(ng_::DeviceSlotState) * kSlots));

    ng_::DeviceArenaView arena{};
    arena.slot_views = d_views;
    arena.states     = d_states;
    arena.slot_count = kSlots;

    auto runPass = [&](double* d_const, double* d_cache) -> cudaError_t {
        // Point the used slot's view at this pass's buffers; every other slot's
        // view stays null, so a kernel that ignored the queue would fault.
        std::vector<ng_::DeviceSlotView> views(kSlots);
        views[kUsedSlot]             = h_view;
        views[kUsedSlot].nodal_const = d_const;
        views[kUsedSlot].constant_xs = d_cache;
        std::vector<ng_::DeviceSlotState> states(kSlots);
        for (auto& st : states) ng_::deviceSlotStateReset(st);
        states[kUsedSlot].material_generation       = 7;
        states[kUsedSlot].nodal_constant_generation = 0; // stale -> the gate opens

        cudaError_t rc;
        if ((rc = cudaMemcpy(d_views, views.data(), sizeof(ng_::DeviceSlotView) * kSlots,
                             cudaMemcpyHostToDevice)) != cudaSuccess) return rc;
        if ((rc = cudaMemcpy(d_states, states.data(), sizeof(ng_::DeviceSlotState) * kSlots,
                             cudaMemcpyHostToDevice)) != cudaSuccess) return rc;
        if ((rc = cudaMemcpy(d_const, zero_const.data(), sizeof(double) * n_const,
                             cudaMemcpyHostToDevice)) != cudaSuccess) return rc;
        if ((rc = cudaMemcpy(d_cache, nan_cache.data(), sizeof(double) * n_cache,
                             cudaMemcpyHostToDevice)) != cudaSuccess) return rc;
        if ((rc = ng_::enqueueNodalUpdateConstant(arena, h_queue, h_geom, d.nxyz, d.ng,
                                                  nullptr)) != cudaSuccess) return rc;
        return cudaDeviceSynchronize();
    };

    TRY(runPass(d_const_a, d_cache_a));
    TRY(runPass(d_const_b, d_cache_b));

    std::vector<double> a(n_const), b(n_const), ca(n_cache), cb(n_cache);
    TRY(cudaMemcpy(a.data(), d_const_a, sizeof(double) * n_const, cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(b.data(), d_const_b, sizeof(double) * n_const, cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(ca.data(), d_cache_a, sizeof(double) * n_cache, cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(cb.data(), d_cache_b, sizeof(double) * n_cache, cudaMemcpyDeviceToHost));

    // Also read back the published generation.
    std::vector<ng_::DeviceSlotState> states_out(kSlots);
    TRY(cudaMemcpy(states_out.data(), d_states, sizeof(ng_::DeviceSlotState) * kSlots,
                   cudaMemcpyDeviceToHost));

    int failures = 0;

    // --- (1) run-to-run bit determinism ------------------------------------
    long long nondet = 0;
    for (size_t i = 0; i < n_const; ++i)
        if (bits(a[i]) != bits(b[i])) ++nondet;
    for (size_t i = 0; i < n_cache; ++i)
        if (bits(ca[i]) != bits(cb[i])) ++nondet;
    std::printf("run-to-run determinism: %lld differing elements of %zu\n", nondet,
                n_const + n_cache);
    if (nondet) {
        std::fprintf(stderr,
                     "  FAIL an N1 phase must be bit-deterministic run to run; %lld "
                     "elements moved between two identical launches\n",
                     nondet);
        ++failures;
    }

    // --- (2) diagD / diagDI are transcendental-free, so they must be exact ---
    const long long stride = ng_::nodalConstStride(d.nxyz, d.ng);
    for (const int slot : {ng_::kNcDiagD, ng_::kNcDiagDI}) {
        long long bad = 0;
        for (long long i = 0; i < stride; ++i) {
            const size_t k = static_cast<size_t>(slot * stride + i);
            if (bits(a[k]) != bits(d.ref_const[k])) ++bad;
        }
        std::printf("%-6s (no sqrt/exp): %lld / %lld differ from the CPU reference\n",
                    kArrayName[slot], bad, stride);
        if (bad) {
            std::fprintf(stderr,
                         "  FAIL %s contains no transcendental, so a difference is a "
                         "CONTRACTION difference, not the N1 library deviation -- check "
                         "that this TU is compiled with --fmad=false\n",
                         kArrayName[slot]);
            ++failures;
        }
    }

    // --- (2b) attribution: which sites does the LIBRARY actually disagree on? -
    std::vector<double> dev_exp(static_cast<size_t>(stride));
    {
        double* d_exp = nullptr;
        TRY(cudaMalloc(&d_exp, sizeof(double) * stride));
        const int block = 256;
        k_probe_exp<<<(static_cast<int>(stride) + block - 1) / block, block>>>(
            d_xs, d_hmesh, d_exp, d.nxyz, d.ng);
        TRY(cudaGetLastError());
        TRY(cudaDeviceSynchronize());
        TRY(cudaMemcpy(dev_exp.data(), d_exp, sizeof(double) * stride, cudaMemcpyDeviceToHost));
        cudaFree(d_exp);
    }
    std::vector<char> exp_differs(static_cast<size_t>(stride), 0);
    long long         exp_bad = 0;
    for (int lk = 0; lk < d.nxyz; ++lk)
        for (int idir = 0; idir < 3; ++idir)
            for (int ig = 0; ig < d.ng; ++ig) {
                const size_t k =
                    static_cast<size_t>(ng_::nodalConstIndex(lk, idir, ig, d.ng));
                const double xsrf =
                    d.xs[static_cast<size_t>(ng_::macroXsIndex(ng_::kXtXsrf, ig, lk, d.ng, d.nxyz))];
                const double xsdf =
                    d.xs[static_cast<size_t>(ng_::macroXsIndex(ng_::kXtXsdf, ig, lk, d.ng, d.nxyz))];
                const double h    = d.hmesh[static_cast<size_t>(lk) * 3 + idir];
                const double host = std::exp(std::sqrt(xsrf * h * h / (4 * xsdf)));
                // The probe kernel's linear index is (lk*3 + idir)*ng + ig,
                // which is exactly nodalConstIndex.
                if (bits(host) != bits(dev_exp[k])) {
                    exp_differs[k] = 1;
                    ++exp_bad;
                }
            }
    std::printf("library disagreement: exp differs at %lld / %lld sites (%.4f%%)\n", exp_bad,
                stride, 100.0 * static_cast<double>(exp_bad) / static_cast<double>(stride));

    // --- (3) the seven exp/sqrt-dependent arrays -----------------------------
    long long  worst_ulp    = 0;
    double     worst_rel    = 0.0;
    long long  unattributed = 0;
    for (int slot = 0; slot < ng_::kNcCount; ++slot) {
        if (slot == ng_::kNcDiagD || slot == ng_::kNcDiagDI) continue;
        long long bad = 0, mx = 0, slot_unattributed = 0;
        double    rel = 0.0;
        for (long long i = 0; i < stride; ++i) {
            const size_t k = static_cast<size_t>(slot * stride + i);
            if (bits(a[k]) == bits(d.ref_const[k])) continue;
            ++bad;
            if (!exp_differs[static_cast<size_t>(i)]) ++slot_unattributed;
            const long long u = ulpDistance(a[k], d.ref_const[k]);
            if (u > mx) mx = u;
            const double den = std::fabs(d.ref_const[k]);
            if (den > 0.0) {
                const double r = std::fabs(a[k] - d.ref_const[k]) / den;
                if (r > rel) rel = r;
            }
        }
        if (mx > worst_ulp) worst_ulp = mx;
        if (rel > worst_rel) worst_rel = rel;
        unattributed += slot_unattributed;
        std::printf("%-6s: %lld / %lld differ, max %lld ulp, max rel %.3e, unattributed %lld\n",
                    kArrayName[slot], bad, stride, mx, rel, slot_unattributed);
    }
    if (unattributed) {
        std::fprintf(stderr,
                     "  FAIL %lld coefficient elements differ at sites where the device "
                     "and host exp AGREE.  That is not the N1 library deviation -- it is "
                     "an unpinned multiply-add.  Re-mine NODAL_CONST_FORMS and check that "
                     "this TU is built with --fmad=false.\n",
                     unattributed);
        ++failures;
    }

    // --- the cache is exact: it is a copy, not a computation ----------------
    long long cache_bad = 0;
    for (size_t i = 0; i < n_cache; ++i)
        if (bits(ca[i]) != bits(d.ref_cache[i])) ++cache_bad;
    std::printf("constant_xs cache: %lld / %zu differ\n", cache_bad, n_cache);
    if (cache_bad) {
        std::fprintf(stderr, "  FAIL the xsrf/xsdf cache is copied, not computed\n");
        ++failures;
    }

    // --- the publish kernel ran, and only for the queued slot ---------------
    if (states_out[kUsedSlot].nodal_constant_generation !=
        states_out[kUsedSlot].material_generation) {
        std::fprintf(stderr, "  FAIL publish kernel did not advance the generation\n");
        ++failures;
    }
    for (int s = 0; s < kSlots; ++s) {
        if (s == kUsedSlot) continue;
        if (states_out[s].nodal_constant_generation != 0) {
            std::fprintf(stderr,
                         "  FAIL slot %d was not queued but its generation moved -- the "
                         "kernel is using a block index as a slot id\n",
                         s);
            ++failures;
        }
    }

    cudaFree(d_xs);
    cudaFree(d_hmesh);
    cudaFree(d_const_a);
    cudaFree(d_const_b);
    cudaFree(d_cache_a);
    cudaFree(d_cache_b);
    cudaFree(d_views);
    cudaFree(d_states);

    if (failures) {
        std::printf("nodal constant device replay: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("nodal constant device replay: PASS (N1: every deviation attributable to "
                "exp; max %lld ulp = %.3e relative)\n",
                worst_ulp, worst_rel);
    return 0;
}
