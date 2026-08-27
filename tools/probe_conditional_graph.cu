// ---------------------------------------------------------------------------
// W0 decision spike 3/5 -- conditional WHILE/SWITCH legality and cost.
//
// WHAT THIS DECIDES.  Whether the Rev.7 5.5 "CUDA conditional case-phase
// scheduler backend" (Task 10, Task 18) is buildable at all on the installed
// driver/runtime, what the control flow costs per iteration, and -- the
// question that actually forks the architecture -- whether a cooperative
// (grid.sync) kernel may live inside a conditional body.
//
// That last one is load-bearing.  Rev.7 offers two device-resident designs:
//   5.5  conditional graph scheduler  (WHILE over outers, SWITCH over phases)
//   5.7  persistent cooperative kernel
// If a cooperative launch is REFUSED inside a conditional body then the two are
// mutually exclusive and the program must choose one, not sequence them.  The
// exact refusal string is the evidence, so it is captured verbatim rather than
// collapsed into a boolean.
//
// FOUR SUB-PROBES.
//   (a) legality   WHILE(h){ SWITCH(hs){ 3 bodies x ~30 kernel nodes } } via the
//                  explicit cudaGraphAddNode conditional API.  SWITCH needs
//                  CUDA 12.8+; the nested-IF fallback -- WHILE{ IF(h0) IF(h1)
//                  IF(h2) } with exactly one handle hot per iteration -- is
//                  built and timed too, ALWAYS, not only when SWITCH fails.
//                  Both are reported so the portability cost of the fallback is
//                  a measured number.
//   (b) instantiate  wall for total node counts ~100 / 500 / 1500, on the host
//                  clock (instantiation is host work; a cudaEvent would not see
//                  it).  Rev.7 Task 10 rebuilds the scheduler graph on shape
//                  changes, so this is the cost of every re-plan.
//   (c) control    us per iteration of the WHILE evaluation with empty bodies,
//                  10k iterations, and separately of WHILE+SWITCH, so the
//                  SWITCH evaluation cost is the difference of two measurements
//                  rather than an attribution.
//   (d) coop       cooperative kernel node inside a conditional body: set
//                  cudaLaunchAttributeCooperative on it, instantiate, launch.
//                  Whichever step refuses, its error name and string are kept.
//
// HANDLE SCOPE.  The API documents the conditional handle as belonging to "the
// graph which will contain the conditional node using this handle".  For a
// SWITCH nested inside a WHILE body it is not obvious whether that means the
// body graph or the root.  This tries the body graph first and falls back to
// the root, and reports which one the runtime accepted -- that fact is needed
// by Task 10 and is not written down anywhere.
//
// BUILD AND RUN (server 238: gcc13 + CUDA 13, sm_120, RTX PRO 6000 Blackwell):
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -rdc=true \
//        -o /tmp/probe_conditional_graph tools/probe_conditional_graph.cu -lcudadevrt
//   CUDA_VISIBLE_DEVICES=0 /tmp/probe_conditional_graph
//
// -rdc=true is needed for the device-side cudaGraphSetConditional() and for the
// cooperative kernel of sub-probe (d).  -lcudadevrt comes AFTER the source:
// GNU ld resolves left to right and would otherwise leave the device-runtime
// symbols undefined.
//
// Runtime: about 40 s.  Output: JSON Lines on stdout; the last line is the
// {"record":"summary"} object the runner lifts into the receipt.
// ---------------------------------------------------------------------------

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>

namespace cg = cooperative_groups;

// Conditional nodes: CUDA 12.3.  SWITCH and IF/ELSE: CUDA 12.8.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define RASBERY_HAS_COND_NODES 1
#else
#define RASBERY_HAS_COND_NODES 0
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#define RASBERY_HAS_COND_SWITCH 1
#else
#define RASBERY_HAS_COND_SWITCH 0
#endif

static const int kCases          = 3;
static const int kBodyNodes      = 30;    // kernel nodes per SWITCH case body
static const int kControlIters   = 10000; // sub-probe (c)
static const int kBlocks         = 34;    // node_blocks(), CudaBICGBackend.cu:2205
static const int kThreads        = 256;
static const int kInstNodeCounts[] = {100, 500, 1500};   // sub-probe (b)
static const int kNumInstCounts  = 3;

static double* g_sink    = nullptr;
static int*    g_counter = nullptr;

// --------------------------------------------------------------------------
// Error bookkeeping.  Nothing here is allowed to swallow a CUDA error: every
// failed step lands in a named slot that is printed in the summary.
// --------------------------------------------------------------------------
struct ErrSlot {
    char text[320];
};
static ErrSlot g_err[10];

static const char* const kSlotName[10] = {
    "while_build", "switch_build", "nested_if_build", "coop_attr",
    "coop_instantiate", "coop_launch", "instantiate", "control",
    "handle_scope", "launch"
};

static void note(int slot, const char* what, cudaError_t e) {
    std::snprintf(g_err[slot].text, sizeof(g_err[slot].text),
                  "%s -> %s (%d): %s", what, cudaGetErrorName(e), static_cast<int>(e),
                  cudaGetErrorString(e));
    cudaGetLastError();  // clear, so the next sub-probe starts clean
}

static void note_text(int slot, const char* text) {
    std::snprintf(g_err[slot].text, sizeof(g_err[slot].text), "%s", text);
}

// JSON string escaping.  CUDA error strings have never contained a quote, but a
// receipt that fails to parse is worse than one that is slightly verbose.
static void json_escape(const char* in, char* out, size_t out_cap) {
    size_t j = 0;
    for (size_t i = 0; in[i] != '\0' && j + 7 < out_cap; ++i) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (c == '"' || c == '\\') {
            out[j++] = '\\';
            out[j++] = static_cast<char>(c);
        } else if (c < 0x20) {
            j += static_cast<size_t>(std::snprintf(out + j, out_cap - j, "\\u%04x", c));
        } else {
            out[j++] = static_cast<char>(c);
        }
    }
    out[j] = '\0';
}

static void print_errors() {
    char esc[640];
    for (int i = 0; i < 10; ++i) {
        if (g_err[i].text[0] == '\0') continue;
        json_escape(g_err[i].text, esc, sizeof(esc));
        std::printf("{\"probe\":\"conditional_graph\",\"record\":\"error\","
                    "\"slot\":\"%s\",\"error\":\"%s\"}\n", kSlotName[i], esc);
    }
    std::fflush(stdout);
}

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------
__global__ void k_body(double* sink) {
    if (blockIdx.x == 0 && threadIdx.x == 0) sink[0] = sink[0] + 1.0;
}

// A real cooperative kernel: it must actually grid.sync(), or sub-probe (d)
// would be testing the attribute and not the capability.
__global__ void k_coop_body(double* sink) {
    cg::grid_group grid = cg::this_grid();
    grid.sync();
    if (grid.thread_rank() == 0) sink[0] = sink[0] + 1.0;
}

#if RASBERY_HAS_COND_NODES
__global__ void k_ctl_while(cudaGraphConditionalHandle h_while, int* counter, int limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int c = counter[0] + 1;
        counter[0] = c;
        cudaGraphSetConditional(h_while, (c < limit) ? 1u : 0u);
    }
}

__global__ void k_ctl_while_switch(cudaGraphConditionalHandle h_while,
                                   cudaGraphConditionalHandle h_switch,
                                   int* counter, int limit, int cases) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int c = counter[0] + 1;
        counter[0] = c;
        cudaGraphSetConditional(h_while, (c < limit) ? 1u : 0u);
        cudaGraphSetConditional(h_switch, static_cast<unsigned>(c % cases));
    }
}

__global__ void k_ctl_while_if3(cudaGraphConditionalHandle h_while,
                                cudaGraphConditionalHandle h0,
                                cudaGraphConditionalHandle h1,
                                cudaGraphConditionalHandle h2,
                                int* counter, int limit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int c = counter[0] + 1;
        counter[0] = c;
        cudaGraphSetConditional(h_while, (c < limit) ? 1u : 0u);
        const int sel = c % 3;
        cudaGraphSetConditional(h0, (sel == 0) ? 1u : 0u);
        cudaGraphSetConditional(h1, (sel == 1) ? 1u : 0u);
        cudaGraphSetConditional(h2, (sel == 2) ? 1u : 0u);
    }
}
#endif  // RASBERY_HAS_COND_NODES

#if RASBERY_HAS_COND_NODES

// --------------------------------------------------------------------------
// Graph construction helpers
// --------------------------------------------------------------------------

// A linear chain of `count` trivial kernel nodes appended after `deps`.
static cudaError_t add_kernel_chain(cudaGraph_t g, const cudaGraphNode_t* deps,
                                    size_t ndeps, int count,
                                    cudaGraphNode_t* out_last) {
    void* args[1];
    args[0] = &g_sink;

    cudaKernelNodeParams p;
    std::memset(&p, 0, sizeof(p));
    p.func           = reinterpret_cast<void*>(k_body);
    p.gridDim        = dim3(kBlocks, 1, 1);
    p.blockDim       = dim3(kThreads, 1, 1);
    p.sharedMemBytes = 0;
    p.kernelParams   = args;
    p.extra          = nullptr;

    cudaGraphNode_t prev = nullptr;
    for (int i = 0; i < count; ++i) {
        cudaGraphNode_t n = nullptr;
        const cudaError_t e =
            (i == 0) ? cudaGraphAddKernelNode(&n, g, deps, ndeps, &p)
                     : cudaGraphAddKernelNode(&n, g, &prev, 1, &p);
        if (e != cudaSuccess) return e;
        prev = n;
    }
    if (out_last) *out_last = prev;
    return cudaSuccess;
}

// Create a handle, preferring the body graph and falling back to the root.
// Records which scope the runtime accepted the first time it is asked.
static int g_handle_scope = -1;  // 0 = body graph, 1 = root graph, -1 = unknown

static cudaError_t make_handle(cudaGraph_t preferred, cudaGraph_t root,
                               unsigned default_value,
                               cudaGraphConditionalHandle* out) {
    cudaError_t e = cudaGraphConditionalHandleCreate(out, preferred, default_value,
                                                     cudaGraphCondAssignDefault);
    if (e == cudaSuccess) {
        if (g_handle_scope < 0) g_handle_scope = (preferred == root) ? 1 : 0;
        return e;
    }
    note(8, "cudaGraphConditionalHandleCreate(body graph)", e);
    e = cudaGraphConditionalHandleCreate(out, root, default_value,
                                         cudaGraphCondAssignDefault);
    if (e == cudaSuccess && g_handle_scope < 0) g_handle_scope = 1;
    return e;
}

static cudaError_t add_conditional(cudaGraph_t parent, const cudaGraphNode_t* deps,
                                   size_t ndeps, cudaGraphConditionalHandle handle,
                                   cudaGraphConditionalNodeType type, unsigned size,
                                   cudaGraphNode_t* out_node, cudaGraph_t* out_bodies) {
    cudaGraphNodeParams np;
    std::memset(&np, 0, sizeof(np));
    np.type                = cudaGraphNodeTypeConditional;
    np.conditional.handle  = handle;
    np.conditional.type    = type;
    np.conditional.size    = size;

    const cudaError_t e = cudaGraphAddNode(out_node, parent, deps, ndeps, &np);
    if (e != cudaSuccess) return e;
    for (unsigned i = 0; i < size; ++i) out_bodies[i] = np.conditional.phGraph_out[i];
    return cudaSuccess;
}

struct Built {
    cudaGraph_t     root  = nullptr;
    cudaGraphExec_t exec  = nullptr;
    size_t          nodes = 0;
    double          instantiate_ms = 0.0;
};

static void destroy(Built* b) {
    if (b->exec) { cudaGraphExecDestroy(b->exec); b->exec = nullptr; }
    if (b->root) { cudaGraphDestroy(b->root); b->root = nullptr; }
}

// WHILE { ctl -> SWITCH(kCases) { body_nodes each } }.  body_nodes == 0 builds
// the empty-body form used by sub-probe (c).
static cudaError_t build_while_switch(int body_nodes, int limit, Built* out,
                                      int err_slot) {
#if !RASBERY_HAS_COND_SWITCH
    (void)body_nodes; (void)limit; (void)out;
    note_text(err_slot, "cudaGraphCondTypeSwitch requires CUDA 12.8+; this build has an older CUDART");
    return cudaErrorNotSupported;
#else
    cudaError_t e = cudaGraphCreate(&out->root, 0);
    if (e != cudaSuccess) { note(err_slot, "cudaGraphCreate(root)", e); return e; }

    cudaGraphConditionalHandle h_while;
    e = cudaGraphConditionalHandleCreate(&h_while, out->root, 1, cudaGraphCondAssignDefault);
    if (e != cudaSuccess) { note(err_slot, "handleCreate(while)", e); return e; }

    cudaGraphNode_t while_node = nullptr;
    cudaGraph_t     while_body = nullptr;
    e = add_conditional(out->root, nullptr, 0, h_while, cudaGraphCondTypeWhile, 1,
                        &while_node, &while_body);
    if (e != cudaSuccess) { note(err_slot, "addNode(WHILE)", e); return e; }

    cudaGraphConditionalHandle h_switch;
    e = make_handle(while_body, out->root, 0, &h_switch);
    if (e != cudaSuccess) { note(err_slot, "handleCreate(switch)", e); return e; }

    // ctl node first: it must set the SWITCH selector before the SWITCH runs.
    int   limit_local = limit;
    int   cases_local = kCases;
    void* ctl_args[5];
    ctl_args[0] = &h_while;
    ctl_args[1] = &h_switch;
    ctl_args[2] = &g_counter;
    ctl_args[3] = &limit_local;
    ctl_args[4] = &cases_local;

    cudaKernelNodeParams cp;
    std::memset(&cp, 0, sizeof(cp));
    cp.func         = reinterpret_cast<void*>(k_ctl_while_switch);
    cp.gridDim      = dim3(1, 1, 1);
    cp.blockDim     = dim3(1, 1, 1);
    cp.kernelParams = ctl_args;

    cudaGraphNode_t ctl = nullptr;
    e = cudaGraphAddKernelNode(&ctl, while_body, nullptr, 0, &cp);
    if (e != cudaSuccess) { note(err_slot, "addKernelNode(ctl)", e); return e; }

    cudaGraphNode_t switch_node = nullptr;
    cudaGraph_t     cases[8];
    e = add_conditional(while_body, &ctl, 1, h_switch, cudaGraphCondTypeSwitch,
                        static_cast<unsigned>(kCases), &switch_node, cases);
    if (e != cudaSuccess) { note(err_slot, "addNode(SWITCH)", e); return e; }

    for (int c = 0; c < kCases && body_nodes > 0; ++c) {
        e = add_kernel_chain(cases[c], nullptr, 0, body_nodes, nullptr);
        if (e != cudaSuccess) { note(err_slot, "addKernelChain(case body)", e); return e; }
    }

    const auto t0 = std::chrono::steady_clock::now();
    e = cudaGraphInstantiate(&out->exec, out->root, 0ull);
    const auto t1 = std::chrono::steady_clock::now();
    if (e != cudaSuccess) { note(err_slot, "cudaGraphInstantiate(while+switch)", e); return e; }
    out->instantiate_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    out->nodes = static_cast<size_t>(2 + kCases * body_nodes);
    return cudaSuccess;
#endif
}

// WHILE { ctl -> IF(h0){b0} -> IF(h1){b1} -> IF(h2){b2} }: the portable
// fallback for runtimes without SWITCH.  Exactly one handle is hot per
// iteration, so it is semantically a 3-way switch with 3 node evaluations
// instead of 1.
static cudaError_t build_while_nested_if(int body_nodes, int limit, Built* out,
                                         int err_slot) {
    cudaError_t e = cudaGraphCreate(&out->root, 0);
    if (e != cudaSuccess) { note(err_slot, "cudaGraphCreate(root)", e); return e; }

    cudaGraphConditionalHandle h_while;
    e = cudaGraphConditionalHandleCreate(&h_while, out->root, 1, cudaGraphCondAssignDefault);
    if (e != cudaSuccess) { note(err_slot, "handleCreate(while)", e); return e; }

    cudaGraphNode_t while_node = nullptr;
    cudaGraph_t     while_body = nullptr;
    e = add_conditional(out->root, nullptr, 0, h_while, cudaGraphCondTypeWhile, 1,
                        &while_node, &while_body);
    if (e != cudaSuccess) { note(err_slot, "addNode(WHILE)", e); return e; }

    cudaGraphConditionalHandle h[3];
    for (int i = 0; i < 3; ++i) {
        e = make_handle(while_body, out->root, 0, &h[i]);
        if (e != cudaSuccess) { note(err_slot, "handleCreate(if)", e); return e; }
    }

    int   limit_local = limit;
    void* ctl_args[6];
    ctl_args[0] = &h_while;
    ctl_args[1] = &h[0];
    ctl_args[2] = &h[1];
    ctl_args[3] = &h[2];
    ctl_args[4] = &g_counter;
    ctl_args[5] = &limit_local;

    cudaKernelNodeParams cp;
    std::memset(&cp, 0, sizeof(cp));
    cp.func         = reinterpret_cast<void*>(k_ctl_while_if3);
    cp.gridDim      = dim3(1, 1, 1);
    cp.blockDim     = dim3(1, 1, 1);
    cp.kernelParams = ctl_args;

    cudaGraphNode_t ctl = nullptr;
    e = cudaGraphAddKernelNode(&ctl, while_body, nullptr, 0, &cp);
    if (e != cudaSuccess) { note(err_slot, "addKernelNode(ctl)", e); return e; }

    cudaGraphNode_t prev = ctl;
    for (int i = 0; i < 3; ++i) {
        cudaGraphNode_t if_node = nullptr;
        cudaGraph_t     if_body = nullptr;
        e = add_conditional(while_body, &prev, 1, h[i], cudaGraphCondTypeIf, 1,
                            &if_node, &if_body);
        if (e != cudaSuccess) { note(err_slot, "addNode(IF)", e); return e; }
        if (body_nodes > 0) {
            e = add_kernel_chain(if_body, nullptr, 0, body_nodes, nullptr);
            if (e != cudaSuccess) { note(err_slot, "addKernelChain(if body)", e); return e; }
        }
        prev = if_node;
    }

    const auto t0 = std::chrono::steady_clock::now();
    e = cudaGraphInstantiate(&out->exec, out->root, 0ull);
    const auto t1 = std::chrono::steady_clock::now();
    if (e != cudaSuccess) { note(err_slot, "cudaGraphInstantiate(while+nested if)", e); return e; }
    out->instantiate_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    out->nodes = static_cast<size_t>(1 + 3 + 3 * body_nodes);
    return cudaSuccess;
}

// Reset the trip counter and replay, event-timed.
static cudaError_t run_timed(Built* b, cudaStream_t stream, double* out_ms) {
    cudaError_t e = cudaMemsetAsync(g_counter, 0, sizeof(int), stream);
    if (e != cudaSuccess) return e;
    e = cudaGraphLaunch(b->exec, stream);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return e;

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if ((e = cudaEventCreate(&e0)) != cudaSuccess) return e;
    if ((e = cudaEventCreate(&e1)) != cudaSuccess) return e;

    if ((e = cudaMemsetAsync(g_counter, 0, sizeof(int), stream)) != cudaSuccess) return e;
    if ((e = cudaEventRecord(e0, stream)) != cudaSuccess) return e;
    if ((e = cudaGraphLaunch(b->exec, stream)) != cudaSuccess) return e;
    if ((e = cudaEventRecord(e1, stream)) != cudaSuccess) return e;
    if ((e = cudaEventSynchronize(e1)) != cudaSuccess) return e;

    float ms = 0.0f;
    if ((e = cudaEventElapsedTime(&ms, e0, e1)) != cudaSuccess) return e;
    *out_ms = static_cast<double>(ms);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return cudaSuccess;
}

// --------------------------------------------------------------------------
// Sub-probe (d): a cooperative kernel node inside a conditional body.
// Returns 1 if it ran, 0 if refused.  The refusing step's error is kept.
// --------------------------------------------------------------------------
static int probe_coop_in_conditional(cudaStream_t stream) {
    Built b;
    cudaError_t e = cudaGraphCreate(&b.root, 0);
    if (e != cudaSuccess) { note(3, "cudaGraphCreate(coop root)", e); return 0; }

    cudaGraphConditionalHandle h_while;
    e = cudaGraphConditionalHandleCreate(&h_while, b.root, 1, cudaGraphCondAssignDefault);
    if (e != cudaSuccess) { note(3, "handleCreate(coop while)", e); destroy(&b); return 0; }

    cudaGraphNode_t while_node = nullptr;
    cudaGraph_t     while_body = nullptr;
    e = add_conditional(b.root, nullptr, 0, h_while, cudaGraphCondTypeWhile, 1,
                        &while_node, &while_body);
    if (e != cudaSuccess) { note(3, "addNode(coop WHILE)", e); destroy(&b); return 0; }

    int   limit_local = 4;
    void* ctl_args[3];
    ctl_args[0] = &h_while;
    ctl_args[1] = &g_counter;
    ctl_args[2] = &limit_local;

    cudaKernelNodeParams cp;
    std::memset(&cp, 0, sizeof(cp));
    cp.func         = reinterpret_cast<void*>(k_ctl_while);
    cp.gridDim      = dim3(1, 1, 1);
    cp.blockDim     = dim3(1, 1, 1);
    cp.kernelParams = ctl_args;

    cudaGraphNode_t ctl = nullptr;
    e = cudaGraphAddKernelNode(&ctl, while_body, nullptr, 0, &cp);
    if (e != cudaSuccess) { note(3, "addKernelNode(coop ctl)", e); destroy(&b); return 0; }

    void* body_args[1];
    body_args[0] = &g_sink;
    cudaKernelNodeParams bp;
    std::memset(&bp, 0, sizeof(bp));
    bp.func         = reinterpret_cast<void*>(k_coop_body);
    bp.gridDim      = dim3(kBlocks, 1, 1);
    bp.blockDim     = dim3(kThreads, 1, 1);
    bp.kernelParams = body_args;

    cudaGraphNode_t coop = nullptr;
    e = cudaGraphAddKernelNode(&coop, while_body, &ctl, 1, &bp);
    if (e != cudaSuccess) { note(3, "addKernelNode(coop body)", e); destroy(&b); return 0; }

    // The attribute is what makes the node a cooperative launch.  Without it
    // the grid.sync() in k_coop_body is undefined, so a "pass" that skipped
    // this step would be a false pass.
    cudaLaunchAttributeValue attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.cooperative = 1;
    e = cudaGraphKernelNodeSetAttribute(coop, cudaLaunchAttributeCooperative, &attr);
    if (e != cudaSuccess) {
        note(3, "cudaGraphKernelNodeSetAttribute(cooperative) on a node inside a conditional body", e);
        destroy(&b);
        return 0;
    }

    e = cudaGraphInstantiate(&b.exec, b.root, 0ull);
    if (e != cudaSuccess) {
        note(4, "cudaGraphInstantiate with a cooperative node inside a conditional body", e);
        destroy(&b);
        return 0;
    }

    e = cudaMemsetAsync(g_counter, 0, sizeof(int), stream);
    if (e == cudaSuccess) e = cudaGraphLaunch(b.exec, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        note(5, "launch of a cooperative node inside a conditional body", e);
        destroy(&b);
        return 0;
    }

    destroy(&b);
    return 1;
}

#endif  // RASBERY_HAS_COND_NODES

int main() {
    for (int i = 0; i < 10; ++i) g_err[i].text[0] = '\0';

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device\n");
        return 3;
    }
    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, device);

    int runtime_version = 0, driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    std::printf("{\"probe\":\"conditional_graph\",\"record\":\"device\","
                "\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                "\"runtime\":%d,\"driver\":%d,\"cudart_compiled\":%d,"
                "\"has_cond_nodes_compiled\":%s,\"has_switch_compiled\":%s}\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                runtime_version, driver_version,
#ifdef CUDART_VERSION
                CUDART_VERSION,
#else
                0,
#endif
                RASBERY_HAS_COND_NODES ? "true" : "false",
                RASBERY_HAS_COND_SWITCH ? "true" : "false");
    std::fflush(stdout);

#if !RASBERY_HAS_COND_NODES
    std::printf("{\"probe\":\"conditional_graph\",\"record\":\"summary\","
                "\"while_ok\":false,\"switch_ok\":false,"
                "\"nested_if_fallback\":false,\"coop_in_conditional\":false,"
                "\"instantiate_ms\":{},\"control_overhead_us_per_iter\":{},"
                "\"handle_scope\":\"unknown\","
                "\"error\":\"built against CUDART < 12.3: no conditional graph nodes\"}\n");
    return 0;
#else
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 3;
    if (cudaMalloc(&g_sink, sizeof(double)) != cudaSuccess) return 3;
    if (cudaMemset(g_sink, 0, sizeof(double)) != cudaSuccess) return 3;
    if (cudaMalloc(&g_counter, sizeof(int)) != cudaSuccess) return 3;
    if (cudaMemset(g_counter, 0, sizeof(int)) != cudaSuccess) return 3;

    // ---- (a) legality -----------------------------------------------------
    bool switch_ok = false, nested_ok = false, while_ok = false;

    {
        Built b;
        const cudaError_t e = build_while_switch(kBodyNodes, 16, &b, 1);
        if (e == cudaSuccess) {
            double ms = 0.0;
            const cudaError_t r = run_timed(&b, stream, &ms);
            if (r == cudaSuccess) {
                switch_ok = true;
                while_ok  = true;
                std::printf("{\"probe\":\"conditional_graph\",\"record\":\"legality\","
                            "\"form\":\"while_switch\",\"ok\":true,\"nodes\":%zu,"
                            "\"instantiate_ms\":%.3f,\"replay_ms\":%.3f}\n",
                            b.nodes, b.instantiate_ms, ms);
            } else {
                note(9, "run(while+switch)", r);
            }
        }
        destroy(&b);
    }

    {
        Built b;
        const cudaError_t e = build_while_nested_if(kBodyNodes, 16, &b, 2);
        if (e == cudaSuccess) {
            double ms = 0.0;
            const cudaError_t r = run_timed(&b, stream, &ms);
            if (r == cudaSuccess) {
                nested_ok = true;
                while_ok  = true;
                std::printf("{\"probe\":\"conditional_graph\",\"record\":\"legality\","
                            "\"form\":\"while_nested_if\",\"ok\":true,\"nodes\":%zu,"
                            "\"instantiate_ms\":%.3f,\"replay_ms\":%.3f}\n",
                            b.nodes, b.instantiate_ms, ms);
            } else {
                note(9, "run(while+nested if)", r);
            }
        }
        destroy(&b);
    }
    std::fflush(stdout);

    // ---- (b) instantiation wall vs node count -----------------------------
    // Reported for whichever form built; the fallback is measured too, because
    // if the program ships the fallback this is the number it pays.
    double inst_ms[3] = {-1.0, -1.0, -1.0};
    double inst_if_ms[3] = {-1.0, -1.0, -1.0};
    for (int i = 0; i < kNumInstCounts; ++i) {
        const int per_case = kInstNodeCounts[i] / kCases;
        if (switch_ok) {
            Built b;
            if (build_while_switch(per_case, 4, &b, 6) == cudaSuccess) {
                inst_ms[i] = b.instantiate_ms;
                std::printf("{\"probe\":\"conditional_graph\",\"record\":\"instantiate\","
                            "\"form\":\"while_switch\",\"target_nodes\":%d,"
                            "\"actual_nodes\":%zu,\"instantiate_ms\":%.3f}\n",
                            kInstNodeCounts[i], b.nodes, b.instantiate_ms);
            }
            destroy(&b);
        }
        if (nested_ok) {
            Built b;
            if (build_while_nested_if(per_case, 4, &b, 6) == cudaSuccess) {
                inst_if_ms[i] = b.instantiate_ms;
                std::printf("{\"probe\":\"conditional_graph\",\"record\":\"instantiate\","
                            "\"form\":\"while_nested_if\",\"target_nodes\":%d,"
                            "\"actual_nodes\":%zu,\"instantiate_ms\":%.3f}\n",
                            kInstNodeCounts[i], b.nodes, b.instantiate_ms);
            }
            destroy(&b);
        }
        std::fflush(stdout);
    }

    // ---- (c) per-iteration control overhead, empty bodies ------------------
    double us_while = -1.0, us_while_switch = -1.0, us_while_if = -1.0;
    {
        // WHILE alone (ctl node only): the floor.
        Built b;
        cudaError_t e = cudaGraphCreate(&b.root, 0);
        cudaGraphConditionalHandle h_while = 0;
        if (e == cudaSuccess)
            e = cudaGraphConditionalHandleCreate(&h_while, b.root, 1, cudaGraphCondAssignDefault);
        cudaGraphNode_t wn = nullptr;
        cudaGraph_t wb = nullptr;
        if (e == cudaSuccess)
            e = add_conditional(b.root, nullptr, 0, h_while, cudaGraphCondTypeWhile, 1, &wn, &wb);
        int   limit_local = kControlIters;
        void* ctl_args[3];
        ctl_args[0] = &h_while;
        ctl_args[1] = &g_counter;
        ctl_args[2] = &limit_local;
        cudaKernelNodeParams cp;
        std::memset(&cp, 0, sizeof(cp));
        cp.func         = reinterpret_cast<void*>(k_ctl_while);
        cp.gridDim      = dim3(1, 1, 1);
        cp.blockDim     = dim3(1, 1, 1);
        cp.kernelParams = ctl_args;
        cudaGraphNode_t ctl = nullptr;
        if (e == cudaSuccess) e = cudaGraphAddKernelNode(&ctl, wb, nullptr, 0, &cp);
        if (e == cudaSuccess) e = cudaGraphInstantiate(&b.exec, b.root, 0ull);
        if (e != cudaSuccess) {
            note(7, "build(while only, empty body)", e);
        } else {
            double ms = 0.0;
            const cudaError_t r = run_timed(&b, stream, &ms);
            if (r == cudaSuccess) us_while = ms * 1000.0 / kControlIters;
            else note(7, "run(while only)", r);
        }
        destroy(&b);
    }
    if (switch_ok) {
        Built b;
        if (build_while_switch(0, kControlIters, &b, 7) == cudaSuccess) {
            double ms = 0.0;
            if (run_timed(&b, stream, &ms) == cudaSuccess)
                us_while_switch = ms * 1000.0 / kControlIters;
        }
        destroy(&b);
    }
    if (nested_ok) {
        Built b;
        if (build_while_nested_if(0, kControlIters, &b, 7) == cudaSuccess) {
            double ms = 0.0;
            if (run_timed(&b, stream, &ms) == cudaSuccess)
                us_while_if = ms * 1000.0 / kControlIters;
        }
        destroy(&b);
    }

    std::printf("{\"probe\":\"conditional_graph\",\"record\":\"control\","
                "\"iters\":%d,\"while_us_per_iter\":%.4f,"
                "\"while_switch_us_per_iter\":%.4f,"
                "\"while_nested_if_us_per_iter\":%.4f,"
                "\"switch_eval_us\":%.4f,\"nested_if_eval_us\":%.4f}\n",
                kControlIters, us_while, us_while_switch, us_while_if,
                (us_while_switch >= 0.0 && us_while >= 0.0) ? us_while_switch - us_while : -1.0,
                (us_while_if >= 0.0 && us_while >= 0.0) ? us_while_if - us_while : -1.0);
    std::fflush(stdout);

    // ---- (d) cooperative launch inside a conditional body ------------------
    const int coop_ok = probe_coop_in_conditional(stream);
    std::printf("{\"probe\":\"conditional_graph\",\"record\":\"coop\","
                "\"coop_in_conditional\":%s}\n", coop_ok ? "true" : "false");
    std::fflush(stdout);

    print_errors();

    const char* scope = (g_handle_scope == 0) ? "body_graph"
                      : (g_handle_scope == 1) ? "root_graph" : "unknown";

    std::printf("{\"probe\":\"conditional_graph\",\"record\":\"summary\","
                "\"while_ok\":%s,\"switch_ok\":%s,\"nested_if_fallback\":%s,"
                "\"coop_in_conditional\":%s,"
                "\"instantiate_ms\":{\"100\":%.3f,\"500\":%.3f,\"1500\":%.3f},"
                "\"instantiate_ms_nested_if\":{\"100\":%.3f,\"500\":%.3f,\"1500\":%.3f},"
                "\"control_overhead_us_per_iter\":{\"while\":%.4f,"
                "\"while_switch\":%.4f,\"while_nested_if\":%.4f},"
                "\"handle_scope\":\"%s\"}\n",
                while_ok ? "true" : "false",
                switch_ok ? "true" : "false",
                nested_ok ? "true" : "false",
                coop_ok ? "true" : "false",
                inst_ms[0], inst_ms[1], inst_ms[2],
                inst_if_ms[0], inst_if_ms[1], inst_if_ms[2],
                us_while, us_while_switch, us_while_if, scope);
    std::fflush(stdout);

    cudaFree(g_counter);
    cudaFree(g_sink);
    cudaStreamDestroy(stream);
    return 0;
#endif  // RASBERY_HAS_COND_NODES
}
