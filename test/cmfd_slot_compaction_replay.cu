// Device replay gate for the CMFD active-slot compaction.
//
// The claim compaction rests on is narrow and testable: changing which BLOCK
// drives a slot changes nothing that slot computes.  Everything downstream of
// slot_map is indexed by the PHYSICAL slot, so a launch that dispatches only
// the arrivals -- in a bucket, in ascending physical order, with the rest of
// the fleet not dispatched at all -- must leave every participant's outputs
// bit-identical to what the full-width launch wrote, and must leave every
// NON-participant's memory untouched.
//
// This harness carries the two kernel shapes that between them exercise every
// hazard in the real arena:
//
//   * a reduction whose per-block chunk is a function of gridDim.x (the thing
//     compaction must not disturb, since it only ever touches grid.y), with a
//     __syncthreads the padding guard has to precede;
//   * a scalar per-slot tail with a shared-memory fold, which is the
//     cmfd_wiel_finalize shape -- the one kernel where a block that returns
//     early would strand its warp-mates at a barrier if the guard were not
//     block-uniform.
//
// The guard is a verbatim copy of RASBERY_CMFD_SLOT from
// src/CudaBICGBackend.cu; tools/test_cmfd_slot_compaction_contract.py compares
// the two texts.
//
// Build (no CMake target; the contract test drives it with --run):
//   nvcc -O3 -std=c++17 -arch=sm_61 --fmad=false \
//        test/cmfd_slot_compaction_replay.cu -o cmfd_slot_compaction_replay
//
// Exit 0 = every arrangement bit-identical and every bystander untouched.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(call)                                                            \
    do {                                                                         \
        const cudaError_t rc_ = (call);                                          \
        if (rc_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(rc_),  \
                         __FILE__, __LINE__);                                    \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

#define RASBERY_CMFD_SLOT_ARGS const int* __restrict__ slot_map, const int lanes

#define RASBERY_CMFD_SLOT(m)                                            \
    int m = 0;                                                          \
    do {                                                                \
        const int rasbery_logical = static_cast<int>(blockIdx.y);       \
        if (rasbery_logical >= lanes) return;                           \
        m = slot_map[rasbery_logical];                                  \
        if (m < 0) return;                                              \
    } while (0)

constexpr int kReduceThreads = 256;

/// reduce_dot_stage1's shape: fixed contiguous chunk from gridDim.x, fixed
/// binary tree, physical-slot strides, halt mask keyed by the mapped slot.
__global__ void stage1(const int n, const long long vec_stride,
                       const double* __restrict__ a, const double* __restrict__ b,
                       double* __restrict__ partial,
                       const std::uint32_t* __restrict__ halt,
                       RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (halt[m] != 0u) return;
    __shared__ double shared[kReduceThreads];

    const double* am = a + m * vec_stride;
    const double* bm = b + m * vec_stride;
    double*       pm = partial + static_cast<long long>(m) * 256;

    const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int begin = static_cast<int>(blockIdx.x) * chunk;
    const int end   = min(begin + chunk, n);

    double sum = 0.0;
    for (int i = begin + static_cast<int>(threadIdx.x); i < end;
         i += static_cast<int>(blockDim.x))
        sum += am[i] * bm[i];

    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) pm[blockIdx.x] = shared[0];
}

/// cmfd_wiel_finalize's shape: a barrier and a shared fold behind the guard.
__global__ void scalar_tail(const int blocks, const double* __restrict__ partial,
                            double* out, const std::uint32_t* __restrict__ halt,
                            RASBERY_CMFD_SLOT_ARGS) {
    RASBERY_CMFD_SLOT(m);
    if (halt[m] != 0u) return;
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ double lane_sum[3];
    const double* pm = partial + static_cast<long long>(m) * 256;
    if (lane < 3) {
        double sum = 0.0;
        for (int i = lane; i < blocks; i += 3) sum = sum + pm[i];
        lane_sum[lane] = sum;
    }
    __syncthreads();
    if (lane != 0) return;
    out[m] = (lane_sum[0] + lane_sum[1]) + lane_sum[2];
}

namespace {

constexpr int kSlots  = 64;
constexpr int kN      = 4099;   // not a multiple of anything convenient
constexpr int kBlocks = 5;
constexpr double kPoison = -12345.678;

int g_failures = 0;
int g_cases    = 0;

int bucketFor(int count, int slots) {
    static const int kBuckets[] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
    for (int b : kBuckets)
        if (count <= b) return b < slots ? b : slots;
    return slots;
}

struct Fixture {
    double *a = nullptr, *b = nullptr, *partial = nullptr, *out = nullptr;
    std::uint32_t* halt = nullptr;
    int*           map  = nullptr;
};

void allocate(Fixture& f) {
    const size_t vec = static_cast<size_t>(kSlots) * 2 * kN;
    CUDA_OK(cudaMalloc(&f.a, vec * sizeof(double)));
    CUDA_OK(cudaMalloc(&f.b, vec * sizeof(double)));
    CUDA_OK(cudaMalloc(&f.partial, static_cast<size_t>(kSlots) * 256 * sizeof(double)));
    CUDA_OK(cudaMalloc(&f.out, static_cast<size_t>(kSlots) * sizeof(double)));
    CUDA_OK(cudaMalloc(&f.halt, static_cast<size_t>(kSlots) * sizeof(std::uint32_t)));
    CUDA_OK(cudaMalloc(&f.map, static_cast<size_t>(kSlots) * sizeof(int)));

    std::vector<double> ha(vec), hb(vec);
    for (size_t i = 0; i < vec; ++i) {
        ha[i] = 1e-3 * std::sin(0.37 * static_cast<double>(i)) + 1.0;
        hb[i] = 1e-3 * std::cos(0.11 * static_cast<double>(i)) + 2.0;
    }
    CUDA_OK(cudaMemcpy(f.a, ha.data(), vec * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(f.b, hb.data(), vec * sizeof(double), cudaMemcpyHostToDevice));
}

/// One launch pair. `lanes`/`map` decide the dispatch; `halt` the masking.
void run(const Fixture& f, const std::vector<int>& map, int lanes,
         const std::vector<std::uint32_t>& halt, std::vector<double>& out) {
    // Poison everything a launch is allowed to write, so "untouched" is
    // observable rather than inferred.
    std::vector<double> poison(static_cast<size_t>(kSlots), kPoison);
    CUDA_OK(cudaMemcpy(f.out, poison.data(), poison.size() * sizeof(double),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f.partial, 0, static_cast<size_t>(kSlots) * 256 * sizeof(double)));
    CUDA_OK(cudaMemcpy(f.map, map.data(), map.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(f.halt, halt.data(), halt.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice));

    stage1<<<dim3(kBlocks, static_cast<unsigned>(lanes)), kReduceThreads>>>(
        kN, 2LL * kN, f.a, f.b, f.partial, f.halt, f.map, lanes);
    scalar_tail<<<dim3(1u, static_cast<unsigned>(lanes)), 32>>>(
        kBlocks, f.partial, f.out, f.halt, f.map, lanes);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    out.assign(static_cast<size_t>(kSlots), 0.0);
    CUDA_OK(cudaMemcpy(out.data(), f.out, out.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
}

std::uint64_t bits(double v) {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

void check(const Fixture& f, const std::vector<int>& participants, const char* tag) {
    ++g_cases;
    // Reference: full width, identity map, halt masking the non-participants --
    // exactly the pre-compaction launch.
    std::vector<int>           full(kSlots);
    std::vector<std::uint32_t> halt(static_cast<size_t>(kSlots), 1u);
    for (int i = 0; i < kSlots; ++i) full[static_cast<size_t>(i)] = i;
    for (int m : participants) halt[static_cast<size_t>(m)] = 0u;
    std::vector<double> reference;
    run(f, full, kSlots, halt, reference);

    // Compacted: participants at lanes [0, count), -1 beyond, grid.y = bucket.
    const int        lanes = bucketFor(static_cast<int>(participants.size()), kSlots);
    std::vector<int> compact(static_cast<size_t>(kSlots), -1);
    for (size_t i = 0; i < participants.size(); ++i)
        compact[i] = participants[i];
    std::vector<double> compacted;
    run(f, compact, lanes, halt, compacted);

    for (int m = 0; m < kSlots; ++m) {
        const bool participates = halt[static_cast<size_t>(m)] == 0u;
        const double r = reference[static_cast<size_t>(m)];
        const double c = compacted[static_cast<size_t>(m)];
        if (participates) {
            if (bits(r) != bits(c)) {
                std::fprintf(stderr,
                             "FAIL %s slot %d: full-width %016llx != compacted %016llx\n",
                             tag, m, static_cast<unsigned long long>(bits(r)),
                             static_cast<unsigned long long>(bits(c)));
                ++g_failures;
            }
            if (bits(r) == bits(kPoison)) {
                std::fprintf(stderr, "FAIL %s slot %d: participant never written\n", tag, m);
                ++g_failures;
            }
        } else if (bits(c) != bits(kPoison)) {
            std::fprintf(stderr,
                         "FAIL %s slot %d: non-participant was written by the compacted "
                         "launch (%016llx)\n", tag, m,
                         static_cast<unsigned long long>(bits(c)));
            ++g_failures;
        }
    }
}

} // namespace

int main() {
    Fixture f;
    allocate(f);

    // Arrival patterns that matter: one instance, the exact bucket steps, a
    // scattered set (so a lane index used as a slot index cannot accidentally
    // be right), and the full fleet.
    const std::vector<std::pair<const char*, std::vector<int>>> cases = {
        {"single", {0}},
        {"single_high", {63}},
        {"pair_far", {0, 63}},
        {"three", {5, 17, 40}},
        {"scattered_9", {1, 3, 7, 11, 19, 28, 33, 51, 62}},
        {"exactly_24", {}},
        {"exactly_25", {}},
        {"full", {}},
    };
    std::vector<std::pair<const char*, std::vector<int>>> built = cases;
    for (int i = 0; i < 24; ++i) built[5].second.push_back(i);
    for (int i = 0; i < 25; ++i) built[6].second.push_back(i);
    for (int i = 0; i < kSlots; ++i) built[7].second.push_back(i);

    for (const auto& entry : built) check(f, entry.second, entry.first);

    if (g_failures != 0) {
        std::fprintf(stderr, "cmfd slot compaction replay: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("cmfd slot compaction replay: PASS (%d arrival patterns, participants "
                "bit-identical, non-participants untouched)\n", g_cases);
    return 0;
}
