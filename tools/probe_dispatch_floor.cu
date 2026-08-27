// ---------------------------------------------------------------------------
// W0 decision spike 1/5 -- graph-node dispatch floor on sm_120.
//
// WHAT THIS DECIDES.  docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md 0 (conclusion
// 3) pins the current per-graph-node dispatch cost at
//
//     c_dispatch = 8.7 s / (432,704 no-op iterations x 22 nodes) = 0.914 us
//
// inferred from the RASBERY_GPU_ITER_BATCH K=8 experiment.  That number is the
// entire numerator of the Phase 5 / Rev.7.1 persistent-residency argument: 10.28 M
// node executions x 0.914 us = 9.4 s of the 55.5 s drive wall.  It was INFERRED,
// never measured directly, and it was inferred on a run whose kernels were not
// empty.  This spike measures the same quantity directly on an empty-ish node so
// the program's headline "dispatch is 17% of drive" claim rests on a measurement
// instead of a subtraction.
//
// METHOD.  N=110 trivial kernels in one linear dependency chain (the production
// sweep graph is 95 nodes -- CudaBICGBackend.cu:2629 -- so 110 brackets it),
// captured into ONE cudaGraph, instantiated once, replayed R=1000 times under
// cudaEvent timing.  The same 110 kernels are then launched one-by-one on a
// plain stream for the graph-vs-stream delta.  Reported as ns per node per
// replay, which is directly comparable to c_dispatch above.
//
// The trivial kernel does one global read-modify-write from a single thread of
// a single block.  All other blocks return at once: that is deliberate -- the
// quantity of interest is the node boundary, not the body -- but the write
// keeps the whole launch from being elided.
//
// GRID SWEEP.  Anchored where anchors exist:
//     34   node_blocks()   = ceil(8451/256)    CudaBICGBackend.cu:2205
//     67   vector_blocks() = ceil(16902/256)   CudaBICGBackend.cu:2206
//     209  XS-recon grid   (8451 nodes x 2 groups) CudaXsReconBackend.cu:256
//     1188 / 4224          campaign-supplied batched shapes bracketing the
//                          M22 and M64 widths.  These two are NOT derived from
//                          a source anchor in this tree; do not quote them as
//                          if they were.
// 188 SMs means 34 and 67 leave two thirds of the device idle; 1188 and 4224
// are the first shapes that saturate it.  If ns/node is flat across the sweep
// the cost is pure dispatch; if it climbs with blocks it is not, and the
// persistent-kernel argument has to be re-derived.
//
// BUILD AND RUN (server 238: gcc13 + CUDA 13, sm_120, RTX PRO 6000 Blackwell):
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/probe_dispatch_floor \
//        tools/probe_dispatch_floor.cu
//   CUDA_VISIBLE_DEVICES=0 /tmp/probe_dispatch_floor
//
// Runtime: about 30 s.  Output: JSON Lines on stdout, one object per line.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The five real kernel shapes.  Kept as a named constant because the contract
// test (tools/test_w0_spikes.py) asserts the sweep did not silently shrink.
static const int kGrids[] = {34, 67, 209, 1188, 4224};
static const int kNumGrids = static_cast<int>(sizeof(kGrids) / sizeof(kGrids[0]));

static const int kThreads = 256;   // kDefaultBlockSize, CudaBICGBackend.cu:31
static const int kNodes   = 110;   // production sweep graph is 95 nodes
static const int kReplays = 1000;
static const int kWarmup  = 20;

// The frozen anchor this spike exists to confirm or refute.
static const double kInferredDispatchUs = 0.914;

#define TRY(expr)                                                              \
    do {                                                                       \
        const cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s -> %s\n", #expr, cudaGetErrorString(_e)); \
            return 3;                                                          \
        }                                                                      \
    } while (0)

__global__ void k_touch(double* sink) {
    // One RMW, one thread, one block.  Cannot be elided; costs one L2 round
    // trip per node, not per block.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sink[0] = sink[0] + 1.0;
    }
}

// Capture kNodes launches on one stream.  Same-stream capture yields exactly
// the linear dependency chain this wants, and it is the capture path the
// production backend already uses (CudaBICGBackend.cu:2653), so nothing here
// exercises an API the server build has not already accepted.
static int build_chain_graph(int blocks, double* d_sink, cudaStream_t stream,
                             cudaGraphExec_t* out_exec, size_t* out_nodes) {
    cudaGraph_t graph = nullptr;
    TRY(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < kNodes; ++i) {
        k_touch<<<blocks, kThreads, 0, stream>>>(d_sink);
    }
    TRY(cudaStreamEndCapture(stream, &graph));

    // Verify the chain is the length claimed: a capture that silently dropped
    // or fused nodes would make ns/node a fiction.
    size_t count = 0;
    TRY(cudaGraphGetNodes(graph, nullptr, &count));
    *out_nodes = count;

    cudaGraphExec_t exec = nullptr;
    TRY(cudaGraphInstantiate(&exec, graph, 0ull));
    TRY(cudaGraphDestroy(graph));
    *out_exec = exec;
    return 0;
}

static int measure_graph(int blocks, double* d_sink, cudaStream_t stream,
                         double* out_ns_per_node, double* out_ms,
                         double* out_instantiate_ms, size_t* out_nodes) {
    // Capture + instantiate is HOST work: it enqueues nothing, so a cudaEvent
    // pair around it would measure an empty device timeline and report ~0.  It
    // gets the host clock; only the replay below gets events.
    const auto t0 = std::chrono::steady_clock::now();
    cudaGraphExec_t exec = nullptr;
    const int rc = build_chain_graph(blocks, d_sink, stream, &exec, out_nodes);
    if (rc != 0) return rc;
    const auto t1 = std::chrono::steady_clock::now();
    *out_instantiate_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    TRY(cudaEventCreate(&e0));
    TRY(cudaEventCreate(&e1));

    for (int i = 0; i < kWarmup; ++i) TRY(cudaGraphLaunch(exec, stream));
    TRY(cudaStreamSynchronize(stream));

    TRY(cudaEventRecord(e0, stream));
    for (int i = 0; i < kReplays; ++i) TRY(cudaGraphLaunch(exec, stream));
    TRY(cudaEventRecord(e1, stream));
    TRY(cudaEventSynchronize(e1));

    float ms = 0.0f;
    TRY(cudaEventElapsedTime(&ms, e0, e1));
    *out_ms = static_cast<double>(ms);
    *out_ns_per_node =
        static_cast<double>(ms) * 1.0e6 /
        (static_cast<double>(kReplays) * static_cast<double>(kNodes));

    TRY(cudaGraphExecDestroy(exec));
    TRY(cudaEventDestroy(e0));
    TRY(cudaEventDestroy(e1));
    return 0;
}

static int measure_stream(int blocks, double* d_sink, cudaStream_t stream,
                          double* out_ns_per_node, double* out_ms) {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    TRY(cudaEventCreate(&e0));
    TRY(cudaEventCreate(&e1));

    for (int i = 0; i < kWarmup; ++i) {
        for (int n = 0; n < kNodes; ++n) k_touch<<<blocks, kThreads, 0, stream>>>(d_sink);
    }
    TRY(cudaStreamSynchronize(stream));

    TRY(cudaEventRecord(e0, stream));
    for (int i = 0; i < kReplays; ++i) {
        for (int n = 0; n < kNodes; ++n) k_touch<<<blocks, kThreads, 0, stream>>>(d_sink);
    }
    TRY(cudaEventRecord(e1, stream));
    TRY(cudaEventSynchronize(e1));

    float ms = 0.0f;
    TRY(cudaEventElapsedTime(&ms, e0, e1));
    *out_ms = static_cast<double>(ms);
    *out_ns_per_node =
        static_cast<double>(ms) * 1.0e6 /
        (static_cast<double>(kReplays) * static_cast<double>(kNodes));

    TRY(cudaEventDestroy(e0));
    TRY(cudaEventDestroy(e1));
    return 0;
}

int main() {
    int device = 0;
    TRY(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    TRY(cudaGetDeviceProperties(&prop, device));

    int runtime_version = 0, driver_version = 0;
    TRY(cudaRuntimeGetVersion(&runtime_version));
    TRY(cudaDriverGetVersion(&driver_version));

    std::printf("{\"probe\":\"dispatch_floor\",\"record\":\"device\","
                "\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                "\"runtime\":%d,\"driver\":%d,\"nodes\":%d,\"replays\":%d,"
                "\"threads\":%d,\"inferred_c_dispatch_us\":%.4f}\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                runtime_version, driver_version, kNodes, kReplays, kThreads,
                kInferredDispatchUs);
    std::fflush(stdout);

    double* d_sink = nullptr;
    TRY(cudaMalloc(&d_sink, sizeof(double)));
    TRY(cudaMemset(d_sink, 0, sizeof(double)));

    cudaStream_t stream = nullptr;
    TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    double graph_ns_at_34 = 0.0, graph_ns_at_67 = 0.0, stream_ns_at_34 = 0.0;

    for (int i = 0; i < kNumGrids; ++i) {
        const int blocks = kGrids[i];

        double g_ns = 0.0, g_ms = 0.0, g_inst_ms = 0.0;
        size_t g_nodes = 0;
        const int grc = measure_graph(blocks, d_sink, stream, &g_ns, &g_ms,
                                      &g_inst_ms, &g_nodes);
        if (grc != 0) return grc;

        double s_ns = 0.0, s_ms = 0.0;
        const int src = measure_stream(blocks, d_sink, stream, &s_ns, &s_ms);
        if (src != 0) return src;

        if (blocks == 34) { graph_ns_at_34 = g_ns; stream_ns_at_34 = s_ns; }
        if (blocks == 67) { graph_ns_at_67 = g_ns; }

        std::printf("{\"probe\":\"dispatch_floor\",\"record\":\"sweep\","
                    "\"blocks\":%d,\"threads\":%d,\"graph_nodes\":%zu,"
                    "\"graph_ns_per_node\":%.3f,\"graph_ms_total\":%.3f,"
                    "\"graph_instantiate_ms\":%.3f,"
                    "\"stream_ns_per_node\":%.3f,\"stream_ms_total\":%.3f,"
                    "\"stream_over_graph\":%.4f}\n",
                    blocks, kThreads, g_nodes, g_ns, g_ms, g_inst_ms, s_ns,
                    s_ms, (g_ns > 0.0) ? s_ns / g_ns : 0.0);
        std::fflush(stdout);
    }

    // The single-deck CMFD shapes are 34 and 67; the persistent argument is
    // built on those two, so the summary quotes the worse of the pair.
    const double c_dispatch_us =
        ((graph_ns_at_34 > graph_ns_at_67) ? graph_ns_at_34 : graph_ns_at_67) / 1000.0;

    std::printf("{\"probe\":\"dispatch_floor\",\"record\":\"summary\","
                "\"c_dispatch_us_measured\":%.4f,"
                "\"c_dispatch_us_inferred\":%.4f,"
                "\"measured_over_inferred\":%.4f,"
                "\"graph_ns_per_node_at_34\":%.3f,"
                "\"graph_ns_per_node_at_67\":%.3f,"
                "\"stream_ns_per_node_at_34\":%.3f}\n",
                c_dispatch_us, kInferredDispatchUs,
                (kInferredDispatchUs > 0.0) ? c_dispatch_us / kInferredDispatchUs : 0.0,
                graph_ns_at_34, graph_ns_at_67, stream_ns_at_34);
    std::fflush(stdout);

    TRY(cudaStreamDestroy(stream));
    TRY(cudaFree(d_sink));
    return 0;
}
