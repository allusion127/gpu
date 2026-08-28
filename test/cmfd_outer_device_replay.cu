// Task 5 device arm: run the five real CMFD outer kernels through a real
// DevicePhaseQueue and score them against the CPU reference.
//
//   ./rasbery_cmfd_outer_device_replay [nxyz]
//
// WHY THIS FILE EXISTS SEPARATELY FROM cmfd_outer_replay.cpp.  That one proves
// the shared BODIES are bit-identical to the CPU loops, on the host, with no
// device.  Two things are not covered there and cannot be:
//
//   1. THE Sec 6.12 COUNTER REDUCTION.  A warp reduction, one atomicAdd per
//      block for each of the three integer counts, and an atomicMax over the
//      IEEE bit pattern for the ratio maximum.  None of that has a host
//      equivalent, and all of it is exactly the kind of code that gets the
//      right answer on a fixture with one block and the wrong one on a real
//      mesh.  So the fixture here is deliberately many blocks wide.
//   2. THE QUEUE PLUMBING.  Every kernel takes its slot from
//      queue.slots[logical], not from blockIdx.  The queue below puts the only
//      live slot at index 3 of a 4-slot fleet and leaves the other three views
//      NULL, so a kernel that used a block index as a slot id faults instead of
//      quietly driving the wrong tenant.
//
// The arithmetic itself is Class B0, so the comparison is bit equality --
// --fmad=false plus the mined CMFD_OUTER_FORMS is what makes that reachable.

#include "CmfdOuterKernel.h"
#include "../src/CudaCmfdOuterKernels.h"

#include "CmfdOuterReference.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace co  = rasbery::cmfd;
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

int failures = 0;

std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

long compare(const char* what, const std::vector<double>& got,
             const std::vector<double>& want) {
    long bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (bits(got[i]) != bits(want[i])) ++bad;
    std::printf("  %-8s %ld / %zu differ\n", what, bad, want.size());
    if (bad) ++failures;
    return bad;
}

template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    if (cudaMalloc(&d, sizeof(T) * h.size()) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        return nullptr;
    return d;
}

} // namespace

int main(int argc, char** argv) {
    const int              nxyz = argc > 1 ? std::atoi(argv[1]) : 4096;
    const cmfdref::Fixture f    = cmfdref::buildFixture(nxyz);
    const cmfdref::Mesh    m    = f.mesh();

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("cmfd outer device replay: SKIP (no CUDA device)\n");
        return 0;
    }

    // --- host reference -----------------------------------------------------
    std::vector<double>   ref_dtil(f.dtil.size()), ref_jnet(f.jnet.size()),
        ref_dhat(f.dhat.size()), ref_psi(static_cast<size_t>(f.nxyz));
    cmfdref::DhatCounters ref_counters;
    cmfdref::refUpdDtil(m, ref_dtil.data());
    cmfdref::refUpdPsi(m, ref_psi.data());
    cmfdref::refUpdJnet(m, ref_jnet.data());
    cmfdref::refUpdDhat(m, /*clamp_enabled=*/false, ref_dhat.data(), &ref_counters);

    // --- device state -------------------------------------------------------
    // Every kernel reads the SAME input state the host reference read, so the
    // four outputs go to four separate buffers rather than being chained.
    const double* d_xsdf = upload(f.xsdf);
    const double* d_xsnf = upload(f.xsnf);
    const double* d_flux = upload(f.flux);
    const double* d_jnet_in = upload(f.jnet);
    const double* d_dtil_in = upload(f.dtil);
    const double* d_dhat_in = upload(f.dhat);
    const int*    d_snode = upload(f.surface_node);
    const int*    d_sdir  = upload(f.surface_dir);
    const double* d_hmesh = upload(f.node_hmesh);
    const double* d_vol   = upload(f.node_volume);
    const double* d_alb   = upload(f.boundary_albedo);
    if (!d_xsdf || !d_flux || !d_snode || !d_alb) {
        std::fprintf(stderr, "upload failed\n");
        return 3;
    }

    double *d_dtil_out = nullptr, *d_jnet_out = nullptr, *d_dhat_out = nullptr,
           *d_psi_out = nullptr;
    TRY(cudaMalloc(&d_dtil_out, sizeof(double) * f.dtil.size()));
    TRY(cudaMalloc(&d_jnet_out, sizeof(double) * f.jnet.size()));
    TRY(cudaMalloc(&d_dhat_out, sizeof(double) * f.dhat.size()));
    TRY(cudaMalloc(&d_psi_out, sizeof(double) * static_cast<size_t>(f.nxyz)));

    co::CmfdGeometryView geom{};
    geom.surface_node    = d_snode;
    geom.surface_dir     = d_sdir;
    geom.node_hmesh      = d_hmesh;
    geom.node_volume     = d_vol;
    geom.boundary_albedo = d_alb;
    geom.nxyz            = m.nxyz;
    geom.nsurf           = m.nsurf;
    geom.ng              = m.ng;

    // Slot 3 of a 4-slot fleet; the other three views stay null so a kernel that
    // used blockIdx.y as a slot id would fault rather than silently misbehave.
    constexpr int kSlots    = 4;
    constexpr int kUsedSlot = 3;

    std::vector<co::CmfdOuterView> views(kSlots);
    views[kUsedSlot].xsdf = d_xsdf;
    views[kUsedSlot].xsnf = d_xsnf;
    views[kUsedSlot].flux = d_flux;
    views[kUsedSlot].jnet = const_cast<double*>(d_jnet_in);
    views[kUsedSlot].dtil = const_cast<double*>(d_dtil_in);
    views[kUsedSlot].dhat = const_cast<double*>(d_dhat_in);
    views[kUsedSlot].psi  = d_psi_out;

    ng_::DevicePhaseQueue queue{};
    for (int i = 0; i < ng_::kMaxSchedulerSlots; ++i) queue.slots[i] = ng_::kQueueEmptySlot;
    queue.slots[0] = kUsedSlot;
    queue.count    = 1;
    queue.bucket   = ng_::gpuSelectBucket(1);

    ng_::DeviceArenaView arena{};
    ng_::DeviceSlotState* d_states = nullptr;
    TRY(cudaMalloc(&d_states, sizeof(ng_::DeviceSlotState) * kSlots));
    arena.states     = d_states;
    arena.slot_count = kSlots;

    ng_::CmfdOuterCounters* d_counters = nullptr;
    TRY(cudaMalloc(&d_counters, sizeof(ng_::CmfdOuterCounters) * kSlots));
    TRY(cudaMemset(d_counters, 0, sizeof(ng_::CmfdOuterCounters) * kSlots));

    const unsigned long long forms = co::cmfdOuterForms();

    // Each kernel writes to its own output buffer, so the view is re-pointed
    // between launches and re-uploaded; the input arrays never move.
    auto runOne = [&](double* out, int which) -> cudaError_t {
        std::vector<co::CmfdOuterView> v = views;
        if (which == 0) v[kUsedSlot].dtil = out;
        if (which == 1) v[kUsedSlot].jnet = out;
        if (which == 2) v[kUsedSlot].dhat = out;
        if (which == 3) v[kUsedSlot].psi = out;
        co::CmfdOuterView* d_views = nullptr;
        cudaError_t        rc      = cudaMalloc(&d_views, sizeof(co::CmfdOuterView) * kSlots);
        if (rc != cudaSuccess) return rc;
        rc = cudaMemcpy(d_views, v.data(), sizeof(co::CmfdOuterView) * kSlots,
                        cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) return rc;
        ng_::CmfdOuterSlotTable table{d_views, kSlots};
        switch (which) {
            case 0: rc = ng_::enqueueUpdDtil(arena, queue, geom, table, forms, nullptr); break;
            case 1: rc = ng_::enqueueUpdJnet(arena, queue, geom, table, forms, nullptr); break;
            case 2:
                rc = ng_::enqueueUpdDhat(arena, queue, geom, table, forms, false, d_counters,
                                         nullptr);
                break;
            default: rc = ng_::enqueueUpdPsi(arena, queue, geom, table, forms, nullptr); break;
        }
        if (rc != cudaSuccess) return rc;
        rc = cudaDeviceSynchronize();
        cudaFree(d_views);
        return rc;
    };

    TRY(runOne(d_dtil_out, 0));
    TRY(runOne(d_jnet_out, 1));
    TRY(runOne(d_dhat_out, 2));
    TRY(runOne(d_psi_out, 3));

    std::vector<double> got_dtil(f.dtil.size()), got_jnet(f.jnet.size()),
        got_dhat(f.dhat.size()), got_psi(static_cast<size_t>(f.nxyz));
    TRY(cudaMemcpy(got_dtil.data(), d_dtil_out, sizeof(double) * got_dtil.size(),
                   cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(got_jnet.data(), d_jnet_out, sizeof(double) * got_jnet.size(),
                   cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(got_dhat.data(), d_dhat_out, sizeof(double) * got_dhat.size(),
                   cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(got_psi.data(), d_psi_out, sizeof(double) * got_psi.size(),
                   cudaMemcpyDeviceToHost));

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("cmfd outer device replay: %s sm_%d%d  nxyz=%d nsurf=%d mask=0x%llX\n",
                prop.name, prop.major, prop.minor, m.nxyz, m.nsurf, forms);
    std::printf(" [1] Class B0 arithmetic\n");
    compare("dtil", got_dtil, ref_dtil);
    compare("jnet", got_jnet, ref_jnet);
    compare("dhat", got_dhat, ref_dhat);
    compare("psi", got_psi, ref_psi);

    // --- the Sec 6.12 reduction --------------------------------------------
    std::vector<ng_::CmfdOuterCounters> counters(kSlots);
    TRY(cudaMemcpy(counters.data(), d_counters, sizeof(ng_::CmfdOuterCounters) * kSlots,
                   cudaMemcpyDeviceToHost));
    const ng_::CmfdOuterCounters& c = counters[kUsedSlot];
    const double dev_ratio          = ng_::cmfdRatioFromBits(c.ratio_max_bits);
    std::printf(" [2] Sec 6.12 counters (block reduce + one atomic per block, atomicMax "
                "over bit patterns)\n");
    std::printf("  total=%llu/%lld  fsum_guard=%llu/%lld  clamped=%llu/%lld  ratio_max=%a/%a\n",
                c.total, ref_counters.total, c.fsum_guard, ref_counters.fsum_guard, c.clamped,
                ref_counters.clamped, dev_ratio, ref_counters.ratio_max);
    if (c.total != static_cast<unsigned long long>(ref_counters.total)) {
        std::fprintf(stderr, "  FAIL _dhat_total\n");
        ++failures;
    }
    if (c.fsum_guard != static_cast<unsigned long long>(ref_counters.fsum_guard)) {
        std::fprintf(stderr, "  FAIL _dhat_fsum_guard\n");
        ++failures;
    }
    if (c.clamped != static_cast<unsigned long long>(ref_counters.clamped)) {
        std::fprintf(stderr, "  FAIL _dhat_clamped\n");
        ++failures;
    }
    if (bits(dev_ratio) != bits(ref_counters.ratio_max)) {
        std::fprintf(stderr,
                     "  FAIL max|dhat/dtil|: the atomicMax over bit patterns did not "
                     "reproduce the host's serial maximum\n");
        ++failures;
    }
    // Every slot that was NOT queued must be untouched -- the cheapest way to
    // catch a kernel indexing counters by block instead of by slot.
    for (int s = 0; s < kSlots; ++s) {
        if (s == kUsedSlot) continue;
        if (counters[s].total || counters[s].fsum_guard || counters[s].clamped ||
            counters[s].ratio_max_bits) {
            std::fprintf(stderr, "  FAIL slot %d was not queued but its counters moved\n", s);
            ++failures;
        }
    }

    if (failures) {
        std::printf("cmfd outer device replay: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("cmfd outer device replay: PASS\n");
    return 0;
}
