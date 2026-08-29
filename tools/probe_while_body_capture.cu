// ---------------------------------------------------------------------------
// Task 10 Step 1 spike -- HOW THE OUTER BODY GETS INTO A WHILE BODY GRAPH.
//
// `docs/TASK10_HOSTFREE_OUTER_20260830_KO.md` §5 leaves exactly one hole open:
// the nodal drive inside the outer body is a HOST call that ends in
// `cudaGraphLaunch(d.nodal_graph, d.stream)` (CudaXsReconBackend.cu:3330,3388).
// A stream in capture mode RECORDS work; the question nobody had measured is
// whether a graph LAUNCH is work that records.
//
// The earlier spike (tools/probe_conditional_graph.cu) settled the control flow
// -- WHILE is legal on 12.6/sm_61, the handle scope the runtime accepts is the
// body graph, SWITCH needs 12.8, 4.55 us per iteration.  It built its bodies
// with the explicit node API.  The real body is ~30 enqueue helpers across two
// streams and will be built by CAPTURE, so this spike asks the capture-shaped
// questions instead.  Six of them, in the order they decide the design:
//
//   (1) capture_while       Can the WHILE itself be built while the ROOT graph
//                           is being stream-captured?  (cudaStreamGetCaptureInfo
//                           -> cudaGraphConditionalHandleCreate -> cudaGraphAddNode
//                           -> cudaStreamUpdateCaptureDependencies), with the
//                           body populated by cudaStreamBeginCaptureToGraph on a
//                           SECOND stream.  Trip count is checked against a
//                           device counter, so a WHILE that runs zero or
//                           `for ever` times fails rather than passes quietly.
//
//   (2) graph_launch_in_capture   THE HOLE.  cudaGraphLaunch(exec, s) with `s`
//                           capturing.  Reported three ways -- the error code,
//                           whether the node count grew, and whether the work
//                           actually ran on replay -- because "returned success"
//                           and "recorded a child node" are not the same claim
//                           and only the third one is the one Task 10 needs.
//
//   (3) child_graph_node    The documented alternative: keep the nodal backend's
//                           un-instantiated cudaGraph_t and hang it off the body
//                           with cudaGraphAddChildGraphNode.  Needs the capture
//                           dependencies to be re-pointed by hand, which is why
//                           it is a different measurement from (2).
//
//   (4) fork_join_in_body   The alternative that needs no nodal API change at
//                           all: the nodal work is ENQUEUED (not launched) on
//                           its own stream, joined to the body capture by the
//                           event pair the runner already records
//                           (m.nodal_handover_ev / nodal_completion_event).
//                           Conditional bodies forbid host nodes and event
//                           nodes; capture turns an event pair into EDGES, so
//                           this should be legal -- should be, until measured.
//
//   (5) device_launch_in_body   cudaGraphInstantiateFlagDeviceLaunch + a kernel
//                           that calls cudaGraphLaunch() from device code.  The
//                           CUDA >= 12.3 escape hatch if (2), (3) and (4) all
//                           refuse.
//
//   (6) memcpy_in_body      One H2D memcpy node inside the body (the segment's
//                           fixed {flux} upload), because the conditional-body
//                           node whitelist is kernel/empty/child/memset/memcpy
//                           and a body that cannot carry the flux upload is a
//                           body that cannot carry outer 1..N.
//
// EVERY SUB-PROBE IS RUN, whatever the ones before it answered -- sub-probe (5)
// excepted, which is opt-in behind RASBERY_PROBE_DEVICE_LAUNCH=1 because on the
// local box it HANGS rather than refusing and takes the rest of the run with it
// (see its comment).  The point is a table for the 238 box (CUDA 13.0, sm_120)
// to be compared against, not an early exit: a local refusal that is a 12.6
// limitation and a local refusal that is a design error look identical from one
// row.
//
// BUILD AND RUN
//   local  (CUDA 12.6, GTX 1080 Ti):
//     nvcc -O3 -std=c++17 -arch=sm_61 -rdc=true \
//          -o /tmp/probe_while_body_capture tools/probe_while_body_capture.cu -lcudadevrt
//   238    (CUDA 13.0, RTX PRO 6000):  -arch=sm_120
//   CUDA_VISIBLE_DEVICES=0 /tmp/probe_while_body_capture
//
// -rdc=true is required: cudaGraphSetConditional() and the device-side
// cudaGraphLaunch() are device-runtime calls.  -lcudadevrt comes AFTER the
// source (GNU ld resolves left to right).
//
// Output: JSON Lines; the last line is {"record":"summary"}.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define RASBERY_HAS_COND_NODES 1
#else
#define RASBERY_HAS_COND_NODES 0
#endif

// cudaStreamBeginCaptureToGraph: CUDA 12.3.  Device-side graph launch: 12.0,
// but the instantiate flag pairs with the conditional API so it is gated the
// same way.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define RASBERY_HAS_CAPTURE_TO_GRAPH 1
#else
#define RASBERY_HAS_CAPTURE_TO_GRAPH 0
#endif

namespace {

int* g_counter = nullptr;   // the body's trip counter
int* g_marks   = nullptr;   // one slot per sub-probe, bumped by the work under test

constexpr int kMarkGraphLaunch = 0;
constexpr int kMarkChildGraph  = 1;
constexpr int kMarkForkJoin    = 2;
constexpr int kMarkDeviceLaunch= 3;
constexpr int kMarkMemcpy      = 4;
constexpr int kMarkSlots       = 5;

char g_err[16][256];
int  g_nerr = 0;

void note(const char* what, cudaError_t rc) {
    if (g_nerr >= 16) return;
    std::snprintf(g_err[g_nerr], sizeof(g_err[0]), "%s -> %s: %s", what,
                  cudaGetErrorName(rc), cudaGetErrorString(rc));
    ++g_nerr;
    cudaGetLastError();
}

}  // namespace

__global__ void k_bump(int* c) { atomicAdd(c, 1); }
__global__ void k_mark(int* m, int slot) { atomicAdd(m + slot, 1); }
__global__ void k_zero(int* c) { *c = 0; }

#if RASBERY_HAS_COND_NODES
/// The verdict kernel's shape: read a device word, decide whether the WHILE
/// runs again.  In the real segment this is the outer transition's exit word.
__global__ void k_set_cond(cudaGraphConditionalHandle h, const int* c, int limit) {
    cudaGraphSetConditional(h, (*c < limit) ? 1u : 0u);
}
/// Sets the handle from the host-visible default before the WHILE is entered.
__global__ void k_arm_cond(cudaGraphConditionalHandle h, unsigned int v) {
    cudaGraphSetConditional(h, v);
}
#endif

#if RASBERY_HAS_COND_NODES
/// Device-side graph launch (sub-probe 5).  The exec handle must come from an
/// instantiate with cudaGraphInstantiateFlagDeviceLaunch and must have been
/// uploaded; `cudaGetCurrentGraphExec()` is not used -- this launches a
/// DIFFERENT graph, which is what the nodal drive would be.
__global__ void k_device_launch(cudaGraphExec_t exec) {
    cudaGraphLaunch(exec, cudaStreamGraphFireAndForget);
}
#endif

namespace {

/// A tiny graph standing in for the nodal backend's instantiated drive: one
/// kernel that bumps a mark so a replay is distinguishable from a no-op.
cudaError_t buildMarkGraph(int slot, cudaStream_t s, cudaGraph_t* graph_out,
                           cudaGraphExec_t* exec_out, unsigned long long inst_flags) {
    cudaGraph_t g = nullptr;
    cudaError_t rc = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (rc != cudaSuccess) return rc;
    k_mark<<<1, 1, 0, s>>>(g_marks, slot);
    rc = cudaStreamEndCapture(s, &g);
    if (rc != cudaSuccess) return rc;
    if (exec_out != nullptr) {
        rc = cudaGraphInstantiate(exec_out, g, inst_flags);
        if (rc != cudaSuccess) { cudaGraphDestroy(g); return rc; }
    }
    if (graph_out != nullptr) *graph_out = g;
    else cudaGraphDestroy(g);
    return cudaSuccess;
}

/// What every sub-probe builds: a root graph whose only node is a WHILE, whose
/// body bumps the trip counter, runs `inject` (the thing under test), and ends
/// with the verdict kernel.  `inject` is handed the body stream and the body
/// graph, because sub-probe 3 needs the graph and the rest need the stream.
///
/// Returns the error of the FIRST step that refused, and reports through
/// `stage` which one that was, so a summary row can say "AddNode" rather than
/// "somewhere in the build".
struct WhileBuild {
    cudaGraph_t     root = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t     body = nullptr;
    cudaGraphNode_t cond = nullptr;
    size_t          root_nodes = 0;
    size_t          body_nodes_before = 0;
    size_t          body_nodes_after  = 0;
};

template <typename Inject>
cudaError_t buildWhile(cudaStream_t s, cudaStream_t bs, int limit, WhileBuild* out,
                       Inject inject, const char** stage) {
#if !RASBERY_HAS_COND_NODES || !RASBERY_HAS_CAPTURE_TO_GRAPH
    (void)s; (void)bs; (void)limit; (void)out; (void)inject;
    *stage = "compiled without conditional nodes";
    return cudaErrorNotSupported;
#else
    *stage = "BeginCapture(root)";
    cudaError_t rc = cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) return rc;

    cudaStreamCaptureStatus st{};
    unsigned long long      id = 0;
    cudaGraph_t             g  = nullptr;
    const cudaGraphNode_t*  deps = nullptr;
    size_t                  ndeps = 0;

    *stage = "GetCaptureInfo(root)";
    rc = cudaStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ndeps);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    *stage = "ConditionalHandleCreate";
    cudaGraphConditionalHandle h{};
    // Default 0 and NO cudaGraphCondAssignDefault: the handle is armed by a
    // kernel node of the root graph, which is the shape the segment needs --
    // the first iteration must be entered only if the segment has not already
    // exited, and only a device word knows that.
    rc = cudaGraphConditionalHandleCreate(&h, g, 0, 0);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    // Arm it, then re-read the capture dependencies so the WHILE hangs off the
    // arming kernel rather than off nothing.
    k_arm_cond<<<1, 1, 0, s>>>(h, 1u);
    *stage = "GetCaptureInfo(after arm)";
    rc = cudaStreamGetCaptureInfo(s, &st, &id, &g, &deps, &ndeps);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    cudaGraphNodeParams p{};
    p.type                 = cudaGraphNodeTypeConditional;
    p.conditional.handle   = h;
    p.conditional.type     = cudaGraphCondTypeWhile;
    p.conditional.size     = 1;

    *stage = "AddNode(conditional)";
    cudaGraphNode_t cond = nullptr;
    rc = cudaGraphAddNode(&cond, g, deps, ndeps, &p);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    *stage = "UpdateCaptureDependencies";
    rc = cudaStreamUpdateCaptureDependencies(s, &cond, 1, cudaStreamSetCaptureDependencies);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    cudaGraph_t body = p.conditional.phGraph_out[0];

    *stage = "BeginCaptureToGraph(body)";
    rc = cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0,
                                       cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    k_bump<<<1, 1, 0, bs>>>(g_counter);

    size_t before = 0;
    cudaGraphGetNodes(body, nullptr, &before);

    *stage = "inject";
    const cudaError_t irc = inject(bs, body);

    size_t after = 0;
    cudaGraphGetNodes(body, nullptr, &after);

    k_set_cond<<<1, 1, 0, bs>>>(h, g_counter, limit);

    *stage = "EndCapture(body)";
    cudaGraph_t body_out = nullptr;
    rc = cudaStreamEndCapture(bs, &body_out);
    if (rc != cudaSuccess) { cudaStreamEndCapture(s, &g); return rc; }

    *stage = "EndCapture(root)";
    rc = cudaStreamEndCapture(s, &g);
    if (rc != cudaSuccess) return rc;

    if (irc != cudaSuccess) { *stage = "inject"; cudaGraphDestroy(g); return irc; }

    *stage = "Instantiate(root)";
    cudaGraphExec_t exec = nullptr;
    rc = cudaGraphInstantiate(&exec, g, 0ull);
    if (rc != cudaSuccess) { cudaGraphDestroy(g); return rc; }

    size_t rn = 0;
    cudaGraphGetNodes(g, nullptr, &rn);

    out->root = g;
    out->exec = exec;
    out->body = body;
    out->cond = cond;
    out->root_nodes = rn;
    out->body_nodes_before = before;
    out->body_nodes_after  = after;
    *stage = "ok";
    return cudaSuccess;
#endif
}

void destroyWhile(WhileBuild* b) {
    if (b->exec != nullptr) cudaGraphExecDestroy(b->exec);
    if (b->root != nullptr) cudaGraphDestroy(b->root);
    b->exec = nullptr;
    b->root = nullptr;
}

/// Runs a built WHILE once and returns the trip count the device observed.
cudaError_t runWhile(WhileBuild* b, cudaStream_t s, int* trips, int* marks_out,
                     int mark_slot) {
    k_zero<<<1, 1, 0, s>>>(g_counter);
    if (mark_slot >= 0) cudaMemsetAsync(g_marks + mark_slot, 0, sizeof(int), s);
    cudaError_t rc = cudaStreamSynchronize(s);
    if (rc != cudaSuccess) return rc;
    rc = cudaGraphLaunch(b->exec, s);
    if (rc != cudaSuccess) return rc;
    rc = cudaStreamSynchronize(s);
    if (rc != cudaSuccess) return rc;
    cudaMemcpy(trips, g_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (mark_slot >= 0 && marks_out != nullptr)
        cudaMemcpy(marks_out, g_marks + mark_slot, sizeof(int), cudaMemcpyDeviceToHost);
    return cudaSuccess;
}

}  // namespace

int main() {
#if !RASBERY_HAS_COND_NODES
    std::printf("{\"probe\":\"while_body_capture\",\"record\":\"summary\","
                "\"supported\":false,\"cudart\":%d}\n", CUDART_VERSION);
    return 0;
#else
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    std::printf("{\"probe\":\"while_body_capture\",\"record\":\"env\",\"gpu\":\"%s\","
                "\"cc\":\"%d.%d\",\"driver\":%d,\"runtime\":%d,\"cudart_header\":%d}\n",
                prop.name, prop.major, prop.minor, drv, rt, CUDART_VERSION);
    std::fflush(stdout);

    cudaMalloc(&g_counter, sizeof(int));
    cudaMalloc(&g_marks, kMarkSlots * sizeof(int));
    cudaMemset(g_counter, 0, sizeof(int));
    cudaMemset(g_marks, 0, kMarkSlots * sizeof(int));

    cudaStream_t s = nullptr, bs = nullptr, ns = nullptr;
    cudaStreamCreate(&s);
    cudaStreamCreate(&bs);
    cudaStreamCreate(&ns);

    double* h_pinned = nullptr;
    double* d_dst    = nullptr;
    cudaHostAlloc(&h_pinned, 1024 * sizeof(double), cudaHostAllocDefault);
    cudaMalloc(&d_dst, 1024 * sizeof(double));
    for (int i = 0; i < 1024; ++i) h_pinned[i] = static_cast<double>(i);

    cudaEvent_t ev_out = nullptr, ev_back = nullptr;
    cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_back, cudaEventDisableTiming);

    const int kLimit = 7;

    // ---- (1) the WHILE itself, built under capture -------------------------
    bool while_ok = false;
    int  while_trips = -1;
    {
        WhileBuild b;
        const char* stage = "";
        const cudaError_t rc = buildWhile(s, bs, kLimit, &b,
            [](cudaStream_t, cudaGraph_t) { return cudaSuccess; }, &stage);
        if (rc != cudaSuccess) {
            note("capture_while", rc);
            std::printf("{\"probe\":\"while_body_capture\",\"record\":\"capture_while\","
                        "\"ok\":false,\"stage\":\"%s\",\"err\":\"%s\"}\n", stage,
                        cudaGetErrorName(rc));
        } else {
            const cudaError_t r = runWhile(&b, s, &while_trips, nullptr, -1);
            if (r != cudaSuccess) note("capture_while.run", r);
            while_ok = (r == cudaSuccess) && (while_trips == kLimit);
            std::printf("{\"probe\":\"while_body_capture\",\"record\":\"capture_while\","
                        "\"ok\":%s,\"trips\":%d,\"expect\":%d,\"root_nodes\":%zu}\n",
                        while_ok ? "true" : "false", while_trips, kLimit, b.root_nodes);
        }
        destroyWhile(&b);
        std::fflush(stdout);
    }

    // ---- (2) THE HOLE: cudaGraphLaunch into a capturing stream -------------
    //
    // Three independent facts, because they can disagree: the API's return
    // code, whether the body graph GREW a node, and whether the launched work
    // actually ran when the WHILE replayed.  Task 10 needs the third.
    bool  gl_api_ok = false, gl_recorded = false, gl_ran = false;
    int   gl_marks = -1, gl_trips = -1;
    {
        cudaGraphExec_t child_exec = nullptr;
        cudaError_t brc = buildMarkGraph(kMarkGraphLaunch, s, nullptr, &child_exec, 0ull);
        if (brc != cudaSuccess) note("graph_launch.child_build", brc);

        WhileBuild b;
        const char* stage = "";
        cudaError_t inject_rc = cudaSuccess;
        const cudaError_t rc = buildWhile(s, bs, kLimit, &b,
            [&](cudaStream_t body_stream, cudaGraph_t) {
                inject_rc = cudaGraphLaunch(child_exec, body_stream);
                // NOT propagated: a refusal here is the answer, not a build
                // failure, and the WHILE must still be finishable so the node
                // count and the replay can be read.
                if (inject_rc != cudaSuccess) cudaGetLastError();
                return cudaSuccess;
            }, &stage);
        gl_api_ok = (inject_rc == cudaSuccess);
        if (rc != cudaSuccess) {
            note("graph_launch.build", rc);
        } else {
            gl_recorded = b.body_nodes_after > b.body_nodes_before;
            const cudaError_t r = runWhile(&b, s, &gl_trips, &gl_marks, kMarkGraphLaunch);
            if (r != cudaSuccess) note("graph_launch.run", r);
            gl_ran = (r == cudaSuccess) && gl_marks == gl_trips && gl_trips == kLimit;
        }
        std::printf("{\"probe\":\"while_body_capture\",\"record\":\"graph_launch_in_capture\","
                    "\"api_ok\":%s,\"api_err\":\"%s\",\"node_recorded\":%s,"
                    "\"body_nodes\":[%zu,%zu],\"replayed\":%s,\"marks\":%d,\"trips\":%d,"
                    "\"stage\":\"%s\"}\n",
                    gl_api_ok ? "true" : "false",
                    gl_api_ok ? "none" : cudaGetErrorName(inject_rc),
                    gl_recorded ? "true" : "false",
                    b.body_nodes_before, b.body_nodes_after,
                    gl_ran ? "true" : "false", gl_marks, gl_trips, stage);
        destroyWhile(&b);
        if (child_exec != nullptr) cudaGraphExecDestroy(child_exec);
        std::fflush(stdout);
    }

    // ---- (3) child graph node, hung off the body by hand -------------------
    //
    // The body is being stream-captured, so the child cannot simply be added:
    // it has to depend on what the capture has recorded so far, and what the
    // capture records next has to depend on IT.  cudaStreamGetCaptureInfo /
    // cudaStreamUpdateCaptureDependencies on the BODY stream is the splice.
    bool cg_ok = false;
    int  cg_marks = -1, cg_trips = -1;
    cudaError_t cg_err = cudaSuccess;
    {
        cudaGraph_t child = nullptr;
        cudaError_t brc = buildMarkGraph(kMarkChildGraph, s, &child, nullptr, 0ull);
        if (brc != cudaSuccess) note("child_graph.build", brc);

        WhileBuild b;
        const char* stage = "";
        const cudaError_t rc = buildWhile(s, bs, kLimit, &b,
            [&](cudaStream_t body_stream, cudaGraph_t body_graph) {
                cudaStreamCaptureStatus st{};
                unsigned long long id = 0;
                cudaGraph_t cur = nullptr;
                const cudaGraphNode_t* deps = nullptr;
                size_t nd = 0;
                cudaError_t r = cudaStreamGetCaptureInfo(body_stream, &st, &id, &cur,
                                                         &deps, &nd);
                if (r != cudaSuccess) { cg_err = r; cudaGetLastError(); return cudaSuccess; }
                cudaGraphNode_t child_node = nullptr;
                r = cudaGraphAddChildGraphNode(&child_node, cur, deps, nd, child);
                if (r != cudaSuccess) { cg_err = r; cudaGetLastError(); return cudaSuccess; }
                r = cudaStreamUpdateCaptureDependencies(body_stream, &child_node, 1,
                                                        cudaStreamSetCaptureDependencies);
                if (r != cudaSuccess) { cg_err = r; cudaGetLastError(); return cudaSuccess; }
                (void)body_graph;
                return cudaSuccess;
            }, &stage);
        if (rc != cudaSuccess) {
            note("child_graph.while", rc);
        } else if (cg_err == cudaSuccess) {
            const cudaError_t r = runWhile(&b, s, &cg_trips, &cg_marks, kMarkChildGraph);
            if (r != cudaSuccess) note("child_graph.run", r);
            cg_ok = (r == cudaSuccess) && cg_marks == cg_trips && cg_trips == kLimit;
        }
        std::printf("{\"probe\":\"while_body_capture\",\"record\":\"child_graph_node\","
                    "\"ok\":%s,\"err\":\"%s\",\"body_nodes\":[%zu,%zu],"
                    "\"marks\":%d,\"trips\":%d,\"stage\":\"%s\"}\n",
                    cg_ok ? "true" : "false",
                    cg_err == cudaSuccess ? "none" : cudaGetErrorName(cg_err),
                    b.body_nodes_before, b.body_nodes_after, cg_marks, cg_trips, stage);
        destroyWhile(&b);
        if (child != nullptr) cudaGraphDestroy(child);
        std::fflush(stdout);
    }

    // ---- (4) cross-stream fork/join inside the body ------------------------
    //
    // The shape the runner already has: record an event on the body stream,
    // have the nodal stream wait on it, ENQUEUE the drive there, record a
    // completion event, have the body stream wait back.  Under capture the
    // event pair is a pair of EDGES, not nodes -- which matters, because the
    // conditional-body whitelist has no event nodes in it.
    bool fj_ok = false;
    int  fj_marks = -1, fj_trips = -1;
    cudaError_t fj_err = cudaSuccess;
    {
        WhileBuild b;
        const char* stage = "";
        const cudaError_t rc = buildWhile(s, bs, kLimit, &b,
            [&](cudaStream_t body_stream, cudaGraph_t) {
                cudaError_t r = cudaEventRecord(ev_out, body_stream);
                if (r != cudaSuccess) { fj_err = r; cudaGetLastError(); return cudaSuccess; }
                r = cudaStreamWaitEvent(ns, ev_out, 0);
                if (r != cudaSuccess) { fj_err = r; cudaGetLastError(); return cudaSuccess; }
                k_mark<<<1, 1, 0, ns>>>(g_marks, kMarkForkJoin);
                r = cudaEventRecord(ev_back, ns);
                if (r != cudaSuccess) { fj_err = r; cudaGetLastError(); return cudaSuccess; }
                r = cudaStreamWaitEvent(body_stream, ev_back, 0);
                if (r != cudaSuccess) { fj_err = r; cudaGetLastError(); return cudaSuccess; }
                return cudaSuccess;
            }, &stage);
        if (rc != cudaSuccess) {
            note("fork_join.while", rc);
        } else if (fj_err == cudaSuccess) {
            const cudaError_t r = runWhile(&b, s, &fj_trips, &fj_marks, kMarkForkJoin);
            if (r != cudaSuccess) note("fork_join.run", r);
            fj_ok = (r == cudaSuccess) && fj_marks == fj_trips && fj_trips == kLimit;
        }
        std::printf("{\"probe\":\"while_body_capture\",\"record\":\"fork_join_in_body\","
                    "\"ok\":%s,\"err\":\"%s\",\"body_nodes\":[%zu,%zu],"
                    "\"marks\":%d,\"trips\":%d,\"stage\":\"%s\"}\n",
                    fj_ok ? "true" : "false",
                    fj_err == cudaSuccess ? "none" : cudaGetErrorName(fj_err),
                    b.body_nodes_before, b.body_nodes_after, fj_marks, fj_trips, stage);
        destroyWhile(&b);
        std::fflush(stdout);
    }

    // ---- (5) device-side graph launch inside the body ----------------------
    //
    // OPT-IN, AND THE REASON IS A MEASUREMENT.  On the local box (CUDA 12.6,
    // sm_61, driver 560.94) this sub-probe does not refuse and does not return:
    // it wedges somewhere in instantiate-for-device-launch / upload / replay and
    // takes the whole process with it -- the 300 s `timeout` killed it, and the
    // sub-probes after it never ran.  A hang is worse than a refusal, because it
    // costs the five answers Task 10 actually acts on.
    //
    // So it is asked only when RASBERY_PROBE_DEVICE_LAUNCH=1.  On 238 (CUDA
    // 13.0, sm_120) the answer may well differ and the question is worth
    // asking -- under `timeout`, and expecting to lose the run if it wedges
    // there too.
    bool dl_ok = false, dl_asked = false;
    int  dl_marks = -1, dl_trips = -1;
    cudaError_t dl_err = cudaSuccess;
    {
        const char* dl_env = std::getenv("RASBERY_PROBE_DEVICE_LAUNCH");
        dl_asked = dl_env != nullptr && std::strcmp(dl_env, "0") != 0;
    }
    if (dl_asked) {
        cudaGraphExec_t dev_exec = nullptr;
        cudaError_t brc = buildMarkGraph(kMarkDeviceLaunch, s, nullptr, &dev_exec,
                                         cudaGraphInstantiateFlagDeviceLaunch);
        if (brc != cudaSuccess) { dl_err = brc; note("device_launch.instantiate", brc); }
        if (dev_exec != nullptr) {
            const cudaError_t urc = cudaGraphUpload(dev_exec, s);
            if (urc != cudaSuccess) { dl_err = urc; note("device_launch.upload", urc); }
            cudaStreamSynchronize(s);
        }
        if (dl_err == cudaSuccess && dev_exec != nullptr) {
            WhileBuild b;
            const char* stage = "";
            const cudaError_t rc = buildWhile(s, bs, kLimit, &b,
                [&](cudaStream_t body_stream, cudaGraph_t) {
                    k_device_launch<<<1, 1, 0, body_stream>>>(dev_exec);
                    const cudaError_t r = cudaGetLastError();
                    if (r != cudaSuccess) dl_err = r;
                    return cudaSuccess;
                }, &stage);
            if (rc != cudaSuccess) {
                note("device_launch.while", rc);
                if (dl_err == cudaSuccess) dl_err = rc;
            } else if (dl_err == cudaSuccess) {
                const cudaError_t r = runWhile(&b, s, &dl_trips, &dl_marks, kMarkDeviceLaunch);
                if (r != cudaSuccess) { dl_err = r; note("device_launch.run", r); }
                dl_ok = (r == cudaSuccess) && dl_marks == dl_trips && dl_trips == kLimit;
            }
            destroyWhile(&b);
        }
        if (dev_exec != nullptr) cudaGraphExecDestroy(dev_exec);
    }
    std::printf("{\"probe\":\"while_body_capture\",\"record\":\"device_launch_in_body\","
                "\"asked\":%s,\"ok\":%s,\"err\":\"%s\",\"marks\":%d,\"trips\":%d}\n",
                dl_asked ? "true" : "false", dl_ok ? "true" : "false",
                dl_err == cudaSuccess ? "none" : cudaGetErrorName(dl_err),
                dl_marks, dl_trips);
    std::fflush(stdout);

    // ---- (6) an H2D memcpy node inside the body ----------------------------
    bool mc_ok = false;
    int  mc_trips = -1;
    cudaError_t mc_err = cudaSuccess;
    {
        WhileBuild b;
        const char* stage = "";
        const cudaError_t rc = buildWhile(s, bs, kLimit, &b,
            [&](cudaStream_t body_stream, cudaGraph_t) {
                const cudaError_t r =
                    cudaMemcpyAsync(d_dst, h_pinned, 1024 * sizeof(double),
                                    cudaMemcpyHostToDevice, body_stream);
                if (r != cudaSuccess) { mc_err = r; cudaGetLastError(); }
                return cudaSuccess;
            }, &stage);
        if (rc != cudaSuccess) {
            note("memcpy.while", rc);
            if (mc_err == cudaSuccess) mc_err = rc;
        } else if (mc_err == cudaSuccess) {
            const cudaError_t r = runWhile(&b, s, &mc_trips, nullptr, -1);
            if (r != cudaSuccess) { mc_err = r; note("memcpy.run", r); }
            mc_ok = (r == cudaSuccess) && mc_trips == kLimit;
        }
        std::printf("{\"probe\":\"while_body_capture\",\"record\":\"memcpy_in_body\","
                    "\"ok\":%s,\"err\":\"%s\",\"trips\":%d,\"stage\":\"%s\"}\n",
                    mc_ok ? "true" : "false",
                    mc_err == cudaSuccess ? "none" : cudaGetErrorName(mc_err),
                    mc_trips, stage);
        destroyWhile(&b);
        std::fflush(stdout);
    }

    for (int i = 0; i < g_nerr; ++i)
        std::printf("{\"probe\":\"while_body_capture\",\"record\":\"error\",\"what\":\"%s\"}\n",
                    g_err[i]);

    std::printf("{\"probe\":\"while_body_capture\",\"record\":\"summary\","
                "\"supported\":true,\"cudart\":%d,\"cc\":\"%d.%d\","
                "\"capture_while\":%s,"
                "\"graph_launch_in_capture\":{\"api\":%s,\"recorded\":%s,\"replayed\":%s},"
                "\"child_graph_node\":%s,\"fork_join_in_body\":%s,"
                "\"device_launch_in_body\":{\"asked\":%s,\"ok\":%s},"
                "\"memcpy_in_body\":%s}\n",
                CUDART_VERSION, prop.major, prop.minor,
                while_ok ? "true" : "false",
                gl_api_ok ? "true" : "false",
                gl_recorded ? "true" : "false",
                gl_ran ? "true" : "false",
                cg_ok ? "true" : "false",
                fj_ok ? "true" : "false",
                dl_asked ? "true" : "false", dl_ok ? "true" : "false",
                mc_ok ? "true" : "false");
    std::fflush(stdout);

    cudaEventDestroy(ev_out);
    cudaEventDestroy(ev_back);
    cudaFreeHost(h_pinned);
    cudaFree(d_dst);
    cudaFree(g_counter);
    cudaFree(g_marks);
    cudaStreamDestroy(s);
    cudaStreamDestroy(bs);
    cudaStreamDestroy(ns);
    return 0;
#endif
}
