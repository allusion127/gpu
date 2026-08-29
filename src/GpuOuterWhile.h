#pragma once
// ---------------------------------------------------------------------------
// Rev.7.1 Task 10 part 4: THE OUTER SEGMENT AS ONE DEVICE-SIDE WHILE.
//
// WHAT IT REPLACES.  A host-free segment already runs `budget` outers without
// reading a device word -- but the HOST still walks the loop.  Every pass
// re-issues ~30 enqueue helpers across two streams, and on the default arm it
// also pays a cudaStreamSynchronize at the top so it can read the exit word and
// stop (`sync_exit_observation`, 11,433 of kngr_238's 11,937 in-body syncs at
// b8).  RASBERY_GPU_OUTER_HOSTFREE_FULL=1 removes that sync and pays for it in
// overrun instead: 13,639 no-op outers on the same deck.  Neither is free, and
// they are the same loop written two ways.
//
// The WHILE is the third way.  ONE outer is captured into a conditional body,
// and the device evaluates the stop rule for itself:
//
//     cudaGraphSetConditional(h, (seg.exit == 0 && halt[slot] == 0 &&
//                                 seg.outer_in_segment < seg.budget) ? 1 : 0)
//
// THE BIT-EXACTNESS ARGUMENT IS THAT PREDICATE.  The stream loop runs outer i
// only when it has observed `exit == 0` (the top-of-pass synchronise) and
// `i < budget`; the transition kernel sets `outer_in_segment = i + 1`, so
// `outer_in_segment < budget` IS `i + 1 < budget` and the two spellings agree
// term for term.  The halt term is not an extra rule -- it is what the stream
// arm gets for free by being bounded by `budget`: an outer whose sweep the
// device abandoned raises the halt, its transition is a no-op, and
// `outer_in_segment` STOPS MOVING.  Without the halt term this WHILE would not
// be a different answer, it would be an infinite one.  With it, the loop stops
// exactly where the stream arm's remaining passes become no-ops, and the
// segment exit finds the same `sweep_host_continued` and runs the same repair.
//
// WHY OUTER 0 IS OUTSIDE IT.  Three separate facts, and each one alone is
// enough:
//
//   * The sweep and nodal graph caches must be WARM.  A miss opens
//     cudaStreamBeginCapture on the very stream the body is being captured on
//     (CudaBICGBackend.cu launch_sweeps, CudaXsReconBackend.cu solveNodal) --
//     a nested capture, which is fatal.  Outer 0 runs eagerly and warms both;
//     the backends additionally refuse to open a nested capture and enqueue
//     directly instead, counted as `graph_warmup_misses`.
//   * `cmfd_sweep_patch` does not patch a segment's FIRST outer (part 3 §1a),
//     so outer 0 is a different body from outers 1..N by construction.
//   * The flux H2D fires on the first outer of a segment and is elided
//     afterwards (the generation matches).  A body captured at outer 1 has no
//     flux node in it, and the graph arm REFUSES if outer 1 would still need
//     one -- a captured upload whose source generation moved is the one shape
//     this mechanism must never have.
//
// WHAT THE HOST STILL DOES.  Exactly what a host-free segment already did at
// its exit: one synchronise, the deferred sweep observation, the repair pass if
// the device abandoned a drive, the exit mirrors and the single observation.
// The WHILE removes passes, not the exit.
//
// MEASURED BEFORE IT WAS WRITTEN (tools/probe_while_body_capture.cu, local
// CUDA 12.6 / sm_61 / driver 560.94):
//
//     capture_while true (trips 7/7)      child_graph_node true (marks 7)
//     fork_join_in_body true (marks 7)    memcpy_in_body true
//     graph_launch_in_capture FALSE       -- hence src/GpuGraphSplice.h
//     exec_update_conditional {"api":true,"applied":true,"result":"success"}
//
// The last row is why this file caches EXECS by key and does not
// cudaGraphExecUpdate: the update is legal and does take effect, so it was a
// real option -- and a cache of at most a handful of instantiations per deck is
// the simpler of the two, because it never has to prove that the graph it is
// updating FROM is the graph the stream arm would have launched.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "CudaOuterGraph.h"

namespace rasbery {
namespace gpu {

// Conditional graph nodes: CUDA 12.3.  cudaStreamBeginCaptureToGraph: 12.3.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define RASBERY_HAS_OUTER_WHILE 1
#else
#define RASBERY_HAS_OUTER_WHILE 0
#endif

/// RASBERY_GPU_OUTER_GRAPH -- DEFAULT OFF, and that is a gate decision.
///
/// The host-free arm shipped default-ON because its exactness argument is an
/// identity (the host reads nothing it did not already know).  This one's is a
/// CAPTURE: the body's node set is frozen at outer 1 and replayed, so it is
/// exact only while every input the body bakes is frozen for the segment too.
/// That is argued term by term at the arm below and it is true -- and it is
/// still a different KIND of claim, so it goes in behind a flag and the whole
/// gate table is run with the flag both ways.
inline bool outerGraphEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_OUTER_GRAPH");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

// OuterGraphRefusal lives in CudaOuterGraph.h, beside OuterHostFreeRefusal and
// for the same reason: OuterSegmentCounters has an array indexed by it, and the
// counters struct must stay usable by a translation unit that never sees a
// conditional-node API.

/// What makes two segments the same BODY.
///
/// Everything the capture bakes that a later segment could legally change.  The
/// budget is here even though the stop rule reads `seg.budget` from device
/// memory -- the condition kernel does not bake it, but the arm's own
/// `budget > 1` test and the eager outer 0 do -- and the mesh dimensions are
/// here because every kernel's grid is derived from them at enqueue time.
///
/// NOT a std::map key: this cache has single digits of entries and a linear
/// scan over a vector is both faster and easier to audit than an ordering.
struct OuterWhileKey {
    unsigned int budget    = 0;
    int          slot      = -1;
    int          nxyz      = 0;
    int          ng        = 0;
    int          nsurf     = 0;
    unsigned int canonical = 0;  ///< the canonical-nodal binding was live
    unsigned int hostfree_full = 0;
    /// The nodal backend's reigv slot -- the ONE device address in the body that
    /// is not the arena's and can therefore move under a re-layout.
    const void*  reigv_slot = nullptr;

    [[nodiscard]] bool operator==(const OuterWhileKey& o) const {
        return budget == o.budget && slot == o.slot && nxyz == o.nxyz && ng == o.ng &&
               nsurf == o.nsurf && canonical == o.canonical &&
               hostfree_full == o.hostfree_full && reigv_slot == o.reigv_slot;
    }
};

/// One instantiated WHILE, and the root graph it came from.
///
/// THE ROOT IS KEPT for the same reason GpuGraphSplice.h keeps the backends'
/// sources: an exec cannot be turned back into a graph, and a future revision
/// that wants cudaGraphExecUpdate (measured legal; see the header) needs one to
/// update FROM.  Destroyed as a pair, always.
struct OuterWhileGraph {
    OuterWhileKey   key{};
    cudaGraph_t     root = nullptr;
    cudaGraphExec_t exec = nullptr;
};

/// The per-runner cache.  Capped, like the nodal one, because a key space
/// bigger than expected must degrade to "re-instantiate" rather than to a leak.
struct OuterWhileCache {
    static constexpr std::size_t kMax = 8;
    std::vector<OuterWhileGraph> entries;
    /// The scratch stream the ROOT graph is captured on.  The BODY is captured
    /// on the segment's own stream (that is where every enqueue helper writes),
    /// so the root needs a second one -- it carries exactly two nodes, the arm
    /// kernel and the conditional, and never executes on this stream.
    cudaStream_t root_stream = nullptr;

    [[nodiscard]] cudaGraphExec_t find(const OuterWhileKey& k) const {
        for (const OuterWhileGraph& e : entries)
            if (e.key == k) return e.exec;
        return nullptr;
    }

    void clear() {
        for (OuterWhileGraph& e : entries) {
            if (e.exec != nullptr) cudaGraphExecDestroy(e.exec);
            if (e.root != nullptr) cudaGraphDestroy(e.root);
        }
        entries.clear();
    }

    void release() {
        clear();
        if (root_stream != nullptr) cudaStreamDestroy(root_stream);
        root_stream = nullptr;
    }
};

#if defined(__CUDACC__) && RASBERY_HAS_OUTER_WHILE

// ---------------------------------------------------------------------------
// The stop rule, in one kernel, used TWICE
// ---------------------------------------------------------------------------
//
// Once as the ARM node of the root graph -- deciding whether the WHILE is
// entered at all -- and once as the last node of the BODY, deciding whether it
// runs again.  ONE kernel and not two, because the two questions are the same
// question asked at the same point in the segment's life: outer 0's transition
// has just run when the arm reads it, and outer i's transition has just run
// when the body's copy does.  Two spellings of a stop rule are two chances to
// spell it differently.
__global__ void k_outer_graph_cond(cudaGraphConditionalHandle handle,
                                   const DeviceOuterSegmentState* segments,
                                   const std::uint32_t* halt, int slot) {
    const DeviceOuterSegmentState s = segments[slot];
    const unsigned int halted = (halt == nullptr) ? 0u : halt[slot];
    cudaGraphSetConditional(handle, (s.exit == 0u && halted == 0u &&
                                     s.outer_in_segment < s.budget)
                                        ? 1u
                                        : 0u);
}

/// cudaGraphAddNode's signature changed in CUDA 13.0 (an edge-data pointer was
/// inserted before numDependencies).  The ONLY call site in this tree, exactly
/// as tools/probe_conditional_graph.cu documents it: do not "fix" the
/// cudaGraphAddKernelNode calls to match, they are a different entry point.
inline cudaError_t addWhileNode(cudaGraphNode_t* out_node, cudaGraph_t parent,
                                const cudaGraphNode_t* deps, std::size_t ndeps,
                                cudaGraphConditionalHandle handle,
                                cudaGraph_t* body_out) {
    // cudaGraphNodeParams holds a union whose members have user-provided
    // constructors, so `np;` does not compile and `np{}` is required; the
    // memset then guarantees the padding the conditional API reads is zero.
    cudaGraphNodeParams np{};
    std::memset(&np, 0, sizeof(np));
    np.type               = cudaGraphNodeTypeConditional;
    np.conditional.handle = handle;
    np.conditional.type   = cudaGraphCondTypeWhile;
    np.conditional.size   = 1;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaError_t e = cudaGraphAddNode(out_node, parent, deps, nullptr, ndeps, &np);
#else
    const cudaError_t e = cudaGraphAddNode(out_node, parent, deps, ndeps, &np);
#endif
    if (e != cudaSuccess) return e;
    *body_out = np.conditional.phGraph_out[0];
    return cudaSuccess;
}

/// Build the WHILE.  `record` is handed the body stream and must enqueue
/// EXACTLY ONE OUTER on it, returning false to abandon the build.
///
/// THE ORDER OF THE SEVEN CALLS IS THE CONTRACT, and it is the order
/// tools/probe_while_body_capture.cu measured:
///
///     BeginCapture(root_stream)               the root, on a scratch stream
///     GetCaptureInfo(root_stream)             which graph is being built
///     ConditionalHandleCreate(handle, root)   the handle belongs to the ROOT
///     k_outer_graph_cond on root_stream       the arm node
///     GetCaptureInfo(root_stream)             the cursor, now the arm
///     addWhileNode(cond, deps = arm)          the WHILE hangs off the arm
///     UpdateCaptureDependencies(root_stream)  the cursor, now the WHILE
///     BeginCaptureToGraph(body_stream, body)  the body, on the SEGMENT stream
///
/// A failure after BeginCapture leaves the stream in capture mode, which only
/// EndCapture can leave -- so every early return below ends both captures
/// before it returns, and the caller's `stage` says which call refused.
template <typename RecordBody>
inline cudaError_t buildOuterWhile(cudaStream_t root_stream, cudaStream_t body_stream,
                                   const DeviceOuterSegmentState* d_segments,
                                   const std::uint32_t* d_halt, int slot,
                                   const char** stage, RecordBody record,
                                   cudaGraph_t* root_out, cudaGraphExec_t* exec_out) {
    *stage = "BeginCapture(root)";
    cudaError_t rc = cudaStreamBeginCapture(root_stream, cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) return rc;

    auto abandon_root = [&](cudaError_t err) {
        cudaGraph_t dead = nullptr;
        cudaStreamEndCapture(root_stream, &dead);
        if (dead != nullptr) cudaGraphDestroy(dead);
        cudaGetLastError();
        return err;
    };

    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long      id = 0;
    cudaGraph_t             g  = nullptr;
    const cudaGraphNode_t*  deps  = nullptr;
    std::size_t             ndeps = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edge_data = nullptr;
#endif

    *stage = "GetCaptureInfo(root)";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &edge_data, &ndeps);
#else
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &ndeps);
#endif
    if (rc != cudaSuccess) return abandon_root(rc);
    if (g == nullptr) return abandon_root(cudaErrorStreamCaptureUnsupported);

    *stage = "ConditionalHandleCreate";
    cudaGraphConditionalHandle handle{};
    // Default 0 and NO cudaGraphCondAssignDefault: the arm kernel below is the
    // only thing entitled to open the loop, and a default of 1 would run one
    // body iteration on a segment that had already exited.
    rc = cudaGraphConditionalHandleCreate(&handle, g, 0, 0);
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "arm";
    k_outer_graph_cond<<<1, 1, 0, root_stream>>>(handle, d_segments, d_halt, slot);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "GetCaptureInfo(arm)";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &edge_data, &ndeps);
#else
    rc = cudaStreamGetCaptureInfo(root_stream, &st, &id, &g, &deps, &ndeps);
#endif
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "AddNode(while)";
    cudaGraphNode_t cond = nullptr;
    cudaGraph_t     body = nullptr;
    rc = addWhileNode(&cond, g, deps, ndeps, handle, &body);
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "UpdateCaptureDependencies";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    rc = cudaStreamUpdateCaptureDependencies(root_stream, &cond, nullptr, 1,
                                             cudaStreamSetCaptureDependencies);
#else
    rc = cudaStreamUpdateCaptureDependencies(root_stream, &cond, 1,
                                             cudaStreamSetCaptureDependencies);
#endif
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "BeginCaptureToGraph(body)";
    rc = cudaStreamBeginCaptureToGraph(body_stream, body, nullptr, nullptr, 0,
                                       cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) return abandon_root(rc);

    *stage = "record(body)";
    const bool recorded = record(body_stream);

    // The body's LAST node is the stop rule, and it is the same kernel the arm
    // is.  Issued even when `recorded` is false, because the body capture must
    // be ended either way and a body without it would be an infinite loop if it
    // ever reached an exec.
    k_outer_graph_cond<<<1, 1, 0, body_stream>>>(handle, d_segments, d_halt, slot);
    const cudaError_t cond_rc = cudaGetLastError();

    *stage = "EndCapture(body)";
    cudaGraph_t body_out = nullptr;
    const cudaError_t body_rc = cudaStreamEndCapture(body_stream, &body_out);

    *stage = "EndCapture(root)";
    cudaGraph_t root = nullptr;
    const cudaError_t root_rc = cudaStreamEndCapture(root_stream, &root);

    if (!recorded || cond_rc != cudaSuccess || body_rc != cudaSuccess ||
        root_rc != cudaSuccess) {
        if (root != nullptr) cudaGraphDestroy(root);
        cudaGetLastError();
        if (body_rc != cudaSuccess) { *stage = "EndCapture(body)"; return body_rc; }
        if (root_rc != cudaSuccess) { *stage = "EndCapture(root)"; return root_rc; }
        if (cond_rc != cudaSuccess) { *stage = "cond"; return cond_rc; }
        *stage = "record(body)";
        return cudaErrorStreamCaptureInvalidated;
    }

    *stage = "Instantiate";
    cudaGraphExec_t exec = nullptr;
    // 3-argument form: the legacy (errorNode, logBuffer, size) overload is gone
    // in CUDA 13, which the 238 server builds with.
    rc = cudaGraphInstantiate(&exec, root, 0ull);
    if (rc != cudaSuccess) {
        cudaGraphDestroy(root);
        cudaGetLastError();
        return rc;
    }
    *root_out = root;
    *exec_out = exec;
    *stage    = "ok";
    return cudaSuccess;
}

#endif // __CUDACC__ && RASBERY_HAS_OUTER_WHILE

} // namespace gpu
} // namespace rasbery
