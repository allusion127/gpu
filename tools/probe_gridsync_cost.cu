// ---------------------------------------------------------------------------
// W0 decision spike 2/5 -- cooperative grid.sync() cost on sm_120.
//
// WHAT THIS DECIDES.  Everything about the persistent/cooperative CMFD kernel.
// docs/PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md 0 (conclusion 4) is blunt about
// it: a persistent kernel replaces graph node boundaries with grid barriers
// almost 1:1 (the exhaustive survey in 3.4.3 found 21 of 21 node boundaries
// need a global barrier, so N_barrier/N_node = 0.955 is a design constant that
// cannot be lowered), and the entire gain is the single term
//
//     removable = N_node * c_dispatch  -  N_barrier * c_barrier
//               = 10.28e6 * c_dispatch  -  9.82e6 * c_barrier
//
// So Phase 5 Stage 1b is GO or NO-GO on one number: c_barrier(sm_120, G blocks).
// This spike is that number.  Nothing in the persistent track may be written
// before it exists.
//
// GATE.  Rev.7.1 program threshold: c_barrier <= 0.384 us at the single-deck
// CMFD shapes (34 and 67 blocks).  The Rev.4-era Phase 5 doc carried 0.45 us,
// derived against the then-current 94.6 s end-to-end; both are printed, and so
// is the removable-seconds arithmetic at the measured value, so the verdict can
// be re-derived against whatever end-to-end wall is current rather than taken
// on faith.
//
// METHOD.  Two arms, because they answer different questions and mixing them
// would corrupt the gate:
//
//   barrier_only  register accumulate + grid.sync(), K=1000 iterations.  This
//                 is c_barrier proper and reproduces the Stage 0-E microbench
//                 sketched in the Phase 5 doc 2.6 line-for-line.
//   barrier_rmw   one global read-modify-write by one thread + grid.sync().
//                 The realistic shape: a persistent kernel's barriers always
//                 straddle memory traffic.  The gap between the arms IS the
//                 L2 round trip, and it is reported rather than buried.
//
// Per-launch cooperative-launch overhead is removed by differencing K=1000
// against K=1 launches, so the reported c_barrier is barrier cost and not
// launch cost.  The raw (undifferenced) number is printed too.
//
// CO-RESIDENCY.  A cooperative launch whose grid exceeds what fits resident
// fails with cudaErrorCooperativeLaunchTooLarge.  This checks
// cudaOccupancyMaxActiveBlocksPerMultiprocessor BEFORE every launch and reports
// {"supported": false} for shapes that do not fit -- 4224 blocks very likely
// does not, and that is itself a finding: it bounds the batched width a
// persistent kernel could ever cover.
//
// BUILD AND RUN (server 238: gcc13 + CUDA 13, sm_120, RTX PRO 6000 Blackwell):
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true \
//        -o /tmp/probe_gridsync_cost tools/probe_gridsync_cost.cu -lcudadevrt
//   CUDA_VISIBLE_DEVICES=0 /tmp/probe_gridsync_cost
//
// -rdc=true is required for cooperative groups grid.sync(); this project
// already builds with CUDA_SEPARABLE_COMPILATION ON (CMakeLists.txt:89), so it
// is not a new constraint.  -lcudadevrt comes AFTER the source: GNU ld resolves
// left to right and would otherwise leave the device-runtime symbols undefined.
// (nvcc links cudadevrt itself under -rdc=true; naming it is belt and braces.)
//
// Runtime: about 20 s.  Output: JSON Lines on stdout, one object per line.
// ---------------------------------------------------------------------------

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

// Same five shapes as probe 1, for the same reasons; see
// tools/probe_dispatch_floor.cu for the anchor of each.
static const int kGrids[] = {34, 67, 209, 1188, 4224};
static const int kNumGrids = static_cast<int>(sizeof(kGrids) / sizeof(kGrids[0]));

static const int kThreads = 256;
static const int kIters   = 1000;  // K
static const int kReps    = 5;     // timed launches per arm, averaged

// THE GATE.  Rev.7.1 persistent-CMFD threshold, in microseconds.
static const double kBarrierGateUs = 0.384;
// The Rev.4-era value from PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md 0, kept so a
// reader can see which threshold a historical verdict was quoted against.
static const double kBarrierGateUsRev4 = 0.45;

// Frozen inputs to the removable-seconds arithmetic (Phase 5 doc 0).
static const double kNodeExecutions   = 10.28e6;
static const double kBarrierRatio     = 0.955;
static const double kDispatchUs       = 0.914;

#define TRY(expr)                                                              \
    do {                                                                       \
        const cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s -> %s\n", #expr, cudaGetErrorString(_e)); \
            return 3;                                                          \
        }                                                                      \
    } while (0)

// Arm 1: the barrier and nothing else.  The accumulate exists only so the loop
// body is not empty; grid.sync() has side effects so the loop cannot be
// removed regardless.
__global__ void k_barrier_only(int iters, double* sink) {
    cg::grid_group grid = cg::this_grid();
    double acc = 0.0;
    for (int i = 0; i < iters; ++i) {
        acc += 1.0;
        grid.sync();
    }
    if (grid.thread_rank() == 0) sink[0] = acc;
}

// Arm 2: barrier straddling one global read-modify-write.  grid.sync() carries
// a memory fence, so the value cannot be held in a register across iterations.
__global__ void k_barrier_rmw(int iters, double* sink) {
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < iters; ++i) {
        if (grid.thread_rank() == 0) {
            const double v = sink[0];
            sink[0] = v + 1.0;
        }
        grid.sync();
    }
}

// One timed cooperative launch series.  Returns total ms for kReps launches.
// Returns a cudaError_t, NOT an exit code.  A cooperative launch can be refused
// for reasons that are specific to one grid shape -- cudaErrorCooperativeLaunchTooLarge
// above all -- and a refusal at 4224 blocks says nothing about 34.  Exiting the
// process there would throw away the measurements the gate actually depends on,
// so a refusal is data: the caller records {"supported": false} with the driver's
// own reason and the sweep continues.
#define TRY_E(expr)                                                            \
    do {                                                                       \
        const cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                               \
            cudaGetLastError();  /* clear so the next shape starts clean */    \
            return _e;                                                         \
        }                                                                      \
    } while (0)

static cudaError_t time_arm(void* func, int blocks, int iters, double* d_sink,
                            cudaStream_t stream, double* out_ms) {
    void* args[2];
    int iters_local = iters;
    args[0] = &iters_local;
    args[1] = &d_sink;

    const dim3 grid(static_cast<unsigned>(blocks), 1, 1);
    const dim3 block(static_cast<unsigned>(kThreads), 1, 1);

    // Warm: first cooperative launch pays module/JIT costs the gate must not see.
    // This is also where a too-large grid is refused, before any event exists.
    TRY_E(cudaLaunchCooperativeKernel(func, grid, block, args, 0, stream));
    TRY_E(cudaStreamSynchronize(stream));

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    TRY_E(cudaEventCreate(&e0));
    TRY_E(cudaEventCreate(&e1));
    TRY_E(cudaEventRecord(e0, stream));
    for (int r = 0; r < kReps; ++r) {
        TRY_E(cudaLaunchCooperativeKernel(func, grid, block, args, 0, stream));
    }
    TRY_E(cudaEventRecord(e1, stream));
    TRY_E(cudaEventSynchronize(e1));

    float ms = 0.0f;
    TRY_E(cudaEventElapsedTime(&ms, e0, e1));
    *out_ms = static_cast<double>(ms);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return cudaSuccess;
}

// c_barrier in us, with per-launch overhead differenced out.
static cudaError_t measure(void* func, int blocks, double* d_sink, cudaStream_t stream,
                           double* out_us_corrected, double* out_us_raw) {
    double ms_k = 0.0, ms_1 = 0.0;
    cudaError_t rc = time_arm(func, blocks, kIters, d_sink, stream, &ms_k);
    if (rc != cudaSuccess) return rc;
    rc = time_arm(func, blocks, 1, d_sink, stream, &ms_1);
    if (rc != cudaSuccess) return rc;

    const double per_launch_k = ms_k / static_cast<double>(kReps);
    const double per_launch_1 = ms_1 / static_cast<double>(kReps);
    *out_us_raw = per_launch_k * 1000.0 / static_cast<double>(kIters);
    *out_us_corrected =
        (per_launch_k - per_launch_1) * 1000.0 / static_cast<double>(kIters - 1);
    return cudaSuccess;
}

int main() {
    int device = 0;
    TRY(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    TRY(cudaGetDeviceProperties(&prop, device));

    std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"device\","
                "\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                "\"cooperative_launch\":%d,\"iters\":%d,\"threads\":%d,"
                "\"gate_c_barrier_us\":%.4f,\"gate_c_barrier_us_rev4\":%.4f}\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.cooperativeLaunch, kIters, kThreads, kBarrierGateUs,
                kBarrierGateUsRev4);
    std::fflush(stdout);

    if (!prop.cooperativeLaunch) {
        std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"summary\","
                    "\"supported\":false,"
                    "\"error\":\"device reports cooperativeLaunch=0\","
                    "\"verdict\":\"NO-GO\"}\n");
        return 0;
    }

    // Occupancy first, launch second -- for BOTH arms, because register
    // pressure differs between them and so may blocks/SM.
    int per_sm_only = 0, per_sm_rmw = 0;
    TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_only, k_barrier_only,
                                                      kThreads, 0));
    TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_rmw, k_barrier_rmw,
                                                      kThreads, 0));
    const int resident_only = per_sm_only * prop.multiProcessorCount;
    const int resident_rmw  = per_sm_rmw * prop.multiProcessorCount;

    std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"occupancy\","
                "\"blocks_per_sm_barrier_only\":%d,"
                "\"blocks_per_sm_barrier_rmw\":%d,"
                "\"max_coresident_blocks_barrier_only\":%d,"
                "\"max_coresident_blocks_barrier_rmw\":%d}\n",
                per_sm_only, per_sm_rmw, resident_only, resident_rmw);
    std::fflush(stdout);

    double* d_sink = nullptr;
    TRY(cudaMalloc(&d_sink, sizeof(double)));
    TRY(cudaMemset(d_sink, 0, sizeof(double)));

    cudaStream_t stream = nullptr;
    TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    double only_at_34 = -1.0, only_at_67 = -1.0;

    for (int i = 0; i < kNumGrids; ++i) {
        const int blocks = kGrids[i];
        const bool fits_only = (blocks <= resident_only);
        const bool fits_rmw  = (blocks <= resident_rmw);

        if (!fits_only && !fits_rmw) {
            std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"sweep\","
                        "\"blocks\":%d,\"supported\":false,"
                        "\"reason\":\"grid exceeds co-resident capacity\","
                        "\"max_coresident_blocks\":%d}\n",
                        blocks, resident_only);
            std::fflush(stdout);
            continue;
        }

        double only_us = -1.0, only_raw = -1.0, rmw_us = -1.0, rmw_raw = -1.0;
        cudaError_t only_err = cudaSuccess, rmw_err = cudaSuccess;
        if (fits_only) {
            // C-style cast on a __global__ function is the CUDA-sample idiom
            // for the cudaLaunchCooperativeKernel void* parameter.
            only_err = measure((void*)k_barrier_only, blocks,
                               d_sink, stream, &only_us, &only_raw);
            if (only_err != cudaSuccess) only_us = only_raw = -1.0;
        }
        if (fits_rmw) {
            rmw_err = measure((void*)k_barrier_rmw, blocks,
                              d_sink, stream, &rmw_us, &rmw_raw);
            if (rmw_err != cudaSuccess) rmw_us = rmw_raw = -1.0;
        }

        // The occupancy query said it fits and the driver disagreed.  That is a
        // finding about this shape, not a reason to abandon the sweep: record
        // the driver's own reason and carry on to the next grid.
        if (only_err != cudaSuccess && rmw_err != cudaSuccess) {
            std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"sweep\","
                        "\"blocks\":%d,\"supported\":false,"
                        "\"reason\":\"cooperative launch refused: %s (%d): %s\","
                        "\"max_coresident_blocks\":%d}\n",
                        blocks, cudaGetErrorName(only_err),
                        static_cast<int>(only_err), cudaGetErrorString(only_err),
                        resident_only);
            std::fflush(stdout);
            continue;
        }

        if (blocks == 34) only_at_34 = only_us;
        if (blocks == 67) only_at_67 = only_us;

        // "unknown", not "false": an arm that never ran did not fail the gate.
        const char* passes = (only_us < 0.0) ? "\"unknown\""
                           : (only_us <= kBarrierGateUs) ? "true" : "false";

        std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"sweep\","
                    "\"blocks\":%d,\"supported\":true,"
                    "\"c_barrier_us\":%.4f,\"c_barrier_us_raw\":%.4f,"
                    "\"c_barrier_rmw_us\":%.4f,\"c_barrier_rmw_us_raw\":%.4f,"
                    "\"rmw_minus_barrier_us\":%.4f,"
                    "\"barrier_only_error\":\"%s\",\"barrier_rmw_error\":\"%s\","
                    "\"passes_gate\":%s}\n",
                    blocks, only_us, only_raw, rmw_us, rmw_raw,
                    (rmw_us >= 0.0 && only_us >= 0.0) ? rmw_us - only_us : -1.0,
                    (only_err == cudaSuccess) ? "" : cudaGetErrorName(only_err),
                    (rmw_err == cudaSuccess) ? "" : cudaGetErrorName(rmw_err),
                    passes);
        std::fflush(stdout);
    }

    // The gate is quoted at the single-deck CMFD shapes: the worse of 34 / 67.
    double gate_value = -1.0;
    if (only_at_34 >= 0.0 && only_at_67 >= 0.0) {
        gate_value = (only_at_34 > only_at_67) ? only_at_34 : only_at_67;
    } else if (only_at_34 >= 0.0) {
        gate_value = only_at_34;
    } else if (only_at_67 >= 0.0) {
        gate_value = only_at_67;
    }

    // Removable seconds at the measured barrier cost, so the verdict can be
    // re-derived against any end-to-end wall rather than trusted blind.
    //
    // -1 when unmeasured, NOT 0.  Substituting 0 for an unmeasured barrier would
    // print the full 9.4 s of dispatch as "removable" -- the most optimistic
    // number the arithmetic can produce -- on a run that measured nothing.
    const double removable_s =
        (gate_value >= 0.0)
            ? (kNodeExecutions * kDispatchUs -
               kNodeExecutions * kBarrierRatio * gate_value) * 1.0e-6
            : -1.0;

    const char* verdict = "UNKNOWN";
    if (gate_value >= 0.0) {
        verdict = (gate_value <= kBarrierGateUs) ? "GO" : "NO-GO";
    }

    std::printf("{\"probe\":\"gridsync_cost\",\"record\":\"summary\","
                "\"supported\":%s,"
                "\"c_barrier_us_at_34\":%.4f,\"c_barrier_us_at_67\":%.4f,"
                "\"c_barrier_us_gate_value\":%.4f,"
                "\"gate_c_barrier_us\":%.4f,\"gate_c_barrier_us_rev4\":%.4f,"
                "\"removable_seconds_at_measured\":%.3f,"
                "\"verdict\":\"%s\"}\n",
                (gate_value >= 0.0) ? "true" : "false",
                only_at_34, only_at_67, gate_value, kBarrierGateUs,
                kBarrierGateUsRev4, removable_s, verdict);
    std::fflush(stdout);

    TRY(cudaStreamDestroy(stream));
    TRY(cudaFree(d_sink));
    return 0;
}
