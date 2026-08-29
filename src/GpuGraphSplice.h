#pragma once
// ---------------------------------------------------------------------------
// Rev.7.1 Task 10 (device outer WHILE): LAUNCH A CACHED GRAPH, OR SPLICE IT.
//
// THE FACT THIS HEADER EXISTS FOR.  A stream in capture mode RECORDS work.  A
// cudaGraphLaunch is not work it records -- it is refused, and the refusal is
// not local: the capture is invalidated and every node recorded before it is
// lost.  Measured, on the local box, by tools/probe_while_body_capture.cu:
//
//     {"record":"graph_launch_in_capture","api_ok":false,
//      "api_err":"cudaErrorStreamCaptureUnsupported","node_recorded":false,
//      "replayed":false}
//     {"record":"error","what":"graph_launch.build ->
//      cudaErrorStreamCaptureInvalidated: operation failed due to a previous
//      error during capture"}
//
// That is the hole docs/TASK10_HOSTFREE_OUTER_20260830_KO.md §5 left open, and
// it is a hole in TWO places, not the one §5 named: the nodal drive
// (CudaXsReconBackend.cu) and the CMFD sweep (CudaBICGBackend.cu) both end in a
// cudaGraphLaunch of a graph they captured once and replay for the rest of the
// run.  Either of them inside a captured outer body kills the capture.
//
// WHAT WORKS INSTEAD, from the same probe:
//
//     {"record":"child_graph_node","ok":true,"body_nodes":[1,2],
//      "marks":7,"trips":7}
//
// -- the SOURCE cudaGraph_t hung off the capture as a child graph node.  The
// splice is three calls and they have to be made in this order, because a
// capture is a moving cursor and not a set:
//
//     cudaStreamGetCaptureInfo          where the cursor is now
//     cudaGraphAddChildGraphNode        the child, depending on that
//     cudaStreamUpdateCaptureDependencies   the cursor, now the child
//
// Skip the third and the next thing captured on that stream depends on what the
// child depended on rather than on the child -- a graph that runs the sweep and
// updjnet CONCURRENTLY, which is not a slower answer, it is a different one.
//
// WHY A CHILD GRAPH AND NOT A RE-ENQUEUE.  Both backends could simply call
// their own enqueue path again under capture and let the nodes record
// directly.  The child graph is preferred because it is the SAME graph object
// the stream path launches: the node set, the parameters and the topology are
// not re-derived from host state that has moved on, so "the graph arm ran what
// the stream arm ran" is true by construction rather than by review.  It also
// keeps the backends' capacity/key logic exactly where it is -- the caller
// hands us the exec it WOULD have launched and the graph it was instantiated
// from, and this decides only how it gets into the stream.
//
// THE COST OF ADMISSION.  A backend that wants to be spliceable must keep the
// cudaGraph_t it instantiated from.  Every one of them currently destroys it on
// the next line (`cudaGraphDestroy(graph)` after `cudaGraphInstantiate`), which
// is correct and idiomatic and exactly wrong for this: an instantiated exec
// cannot be turned back into a graph.  Retaining it costs the graph's node
// descriptors -- kilobytes, once per cache entry, for the life of the run --
// and it must be destroyed with its exec, or the cache leaks two objects
// instead of one.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>

namespace rasbery {

/// HAS ANYTHING IN THIS PROCESS EVER OPENED A CAPTURE A CACHED LAUNCH COULD LAND
/// IN?  Raised once, by the outer-body capture, and never lowered.
///
/// WHY A FLAG AND NOT JUST cudaStreamIsCapturing.  The splice below sits on the
/// hottest launch site in the tree -- the CMFD sweep graph, once per outer, plus
/// the nodal drive -- and every arm of the run pays for whatever it does, not
/// just the one arm that captures.  cudaStreamIsCapturing is a cheap runtime
/// call, but "cheap" times 24,000 on a 60 s run is a number somebody would have
/// to defend, and the honest answer is that a run which never captures does not
/// need to be asked.  With the flag down this is a relaxed load of a cache-hot
/// word and then the launch that was always there.
///
/// NEVER LOWERED, and that is deliberate: it is not "a capture is open now" (the
/// runtime owns that question and answers it per stream) but "this process is
/// one where the question is worth asking".  A flag that could go back down
/// would have to be exactly synchronised with the capture window, and getting
/// that wrong means a cudaGraphLaunch inside a capture -- which is not a slow
/// answer, it is a destroyed capture.
inline std::atomic<bool> g_graph_capture_possible{false};

inline void graphCapturePossible() {
    g_graph_capture_possible.store(true, std::memory_order_relaxed);
}

/// Is this stream recording rather than executing?
///
/// Answered by cudaStreamIsCapturing, which does NOT clear a sticky error and
/// does not care whether the capture was started on this stream or joined into
/// from another one -- both matter, because the nodal backend's stream joins
/// the outer body's capture through an event pair and is `Active` without ever
/// having been begun.
inline bool graphCaptureActive(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return st == cudaStreamCaptureStatusActive;
}

/// Launch `exec` on `stream`, or -- if `stream` is capturing -- record `graph`
/// into the capture as a child graph node in the stream's position.
///
/// `graph` may be null on the non-capturing path; it is only consulted when a
/// splice is actually required, and a null there is the honest refusal
/// (cudaErrorStreamCaptureUnsupported) rather than a silently skipped step.
///
/// RETURNS THE FIRST ERROR AND LEAVES THE CAPTURE'S FATE TO THE CALLER.  A
/// splice that fails part way has already moved the capture into a state only
/// cudaStreamEndCapture can leave, so this does not try to unwind: the caller's
/// existing capture-failure path (both backends have one, and both demote to a
/// direct enqueue) is the recovery.
inline cudaError_t graphLaunchOrSplice(cudaGraphExec_t exec, cudaGraph_t graph,
                                       cudaStream_t stream) {
    if (!g_graph_capture_possible.load(std::memory_order_relaxed))
        return cudaGraphLaunch(exec, stream);
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaError_t rc = cudaStreamIsCapturing(stream, &st);
    if (rc != cudaSuccess) return rc;
    if (st != cudaStreamCaptureStatusActive) return cudaGraphLaunch(exec, stream);
    if (graph == nullptr) return cudaErrorStreamCaptureUnsupported;

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    unsigned long long      id  = 0;
    cudaGraph_t             cur = nullptr;
    const cudaGraphNode_t*  deps = nullptr;
    std::size_t             ndeps = 0;
    rc = cudaStreamGetCaptureInfo(stream, &cap, &id, &cur, &deps, &ndeps);
    if (rc != cudaSuccess) return rc;
    if (cur == nullptr) return cudaErrorStreamCaptureUnsupported;

    cudaGraphNode_t child = nullptr;
    rc = cudaGraphAddChildGraphNode(&child, cur, deps, ndeps, graph);
    if (rc != cudaSuccess) return rc;

    return cudaStreamUpdateCaptureDependencies(stream, &child, 1,
                                               cudaStreamSetCaptureDependencies);
}

}  // namespace rasbery
