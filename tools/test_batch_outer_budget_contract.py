#!/usr/bin/env python3
"""Contract gate for the device outer segment's BUDGET IN `--batch-mode`
(Rev.7.1 Task 18).

WHY THIS IS ITS OWN GATE.  A batch that has silently fallen back to a segment
budget of one produces exactly the same numbers as one that has not.  It is
slower, it pays a host rendezvous per outer, and the only place it shows is a
receipt field nobody diffs -- `segment_launches` equal to `device_outers`
instead of a fraction of it.  Every invariant below is therefore a source-level
one, because at result level there is nothing to see.

WHAT TASK 18-LITE MEASURED, AND WHAT TASK 18 CHANGED.  Task 18-lite made the
segment serve one arena slot per deck, and then found two structural reasons a
batch could not run it at more than budget 1:

  * `CudaBatchArena::enqueueSweeps` refused any second live instance, because it
    took no claim on the arena stream and a graph capture swallows every enqueue
    that stream receives while it is open.  A batch therefore took the BLOCKING
    sweep hook (`sweep_synchronizes`), which the runner answers by forcing the
    budget to one -- correctly, because under a hook that drains, outer i+1
    cannot be enqueued before outer i has been observed.

  * the batched nodal arena took an adopted canonical set into its view table
    and then uploaded Geometry::Jnet over it on every drive, so the segment had
    to keep the jnet bridge it exists to remove.

Both are fixed, and this file pins the fixes in the four files that carry them.

THE EIGHT CHECKS

  1. THE ARM DOES NOT DECIDE THE SWEEP ARM FROM THE BATCH WIDTH.  `stream_sweep`
     must not be gated on the run being solo; only the choice of WHOSE stream
     the segment binds may be.

  2. A BATCH DOES NOT BIND THE ARENA STREAM.  M segments on one stream capture
     each other's kernels.  `useStream` may be called with the arena's stream
     only under the solo test.

  3. THE HOOK HANDS ITS STREAM TO THE DRIVE.  Without it the drive cannot know
     whether it has to join two streams or is already on one.

  4. THE FALLBACK SETTLES BOTH STREAMS.  A blocking drive reads host arrays that
     the arena stream (sweep staging) and the runner's stream (the psi and dhat
     mirrors of this outer) are both filling.

  5. THE ENQUEUE PATH CLAIMS THE STREAM AND STAGES INTO ITS OWN LANE.  Checked
     in full by test_outer_stream_sweep_contract.py; asserted here as the reason
     check 1 is safe, so a reader of either file finds the other.

  6. THE FLEET MASKS HAVE ONE LANE PER SLOT.  They are page-locked memcpyAsync
     SOURCES; one buffer means one launch in flight.

  7. THE BATCHED NODAL ARENA HONOURS THE CANONICAL BINDING.  Pinned in full by
     test_segment_canonical_nodal_contract.py; asserted here because a segment
     that has to keep its jnet bridge pays 2 x nsurf*ng doubles an OUTER, which
     is the cost a wider budget is supposed to be amortising.

  8. THE RUNNER STILL FORCES BUDGET 1 UNDER A SYNCHRONISING HOOK.  The two arms
     have not merged; what changed is which arm a batch gets.

Run:  python tools/test_batch_outer_budget_contract.py
"""

from __future__ import annotations

import os
import py_compile
import re
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def body_after(code: str, marker: str, end: str) -> str:
    at = code.find(marker)
    if at < 0:
        return ""
    stop = code.find(end, at + len(marker))
    return code[at:stop if stop > 0 else len(code)]


def main() -> int:
    problems: list[str] = []

    driver = strip_comments(read("src", "Driver.h"))
    backend = strip_comments(read("src", "CudaBICGBackend.cu"))
    nodal = strip_comments(read("src", "CudaXsReconBackend.cu"))

    arm = body_after(driver, "const bool solo = rasberyBatchWidth() <= 1;",
                     "gpu::OuterSegmentHooks hooks;")
    if not arm:
        problems.append("Driver.h: the segment arm no longer spells `solo` from "
                        "rasberyBatchWidth(); the batch/solo stream decision has moved or "
                        "gone, and this gate can no longer see which arm a batch gets")
    else:
        # 1. the sweep arm is about the STREAM's existence, not the batch width.
        if "const bool stream_sweep = have_sweep_stream;" not in arm:
            problems.append("Driver.h: `stream_sweep` is not `have_sweep_stream`.  Gating it "
                            "on `solo` puts a batch back on the blocking sweep hook, which "
                            "the runner answers by forcing the segment budget to 1 -- and "
                            "the only visible trace is segment_launches == device_outers")
        # 2. ...but the arena stream is still adopted only when solo.
        adopt = body_after(arm, "const bool shared_stream", "if (!shared_stream)")
        if "solo &&" not in adopt:
            problems.append("Driver.h: the segment adopts the arena stream without the solo "
                            "test.  M segments on one stream capture each other's kernels: "
                            "`operation failed due to a previous error during capture`, four "
                            "decks dead in 1.2 s")
        if "useStream(nullptr)" not in arm:
            problems.append("Driver.h: the arm no longer restores the runner's own stream.  "
                            "The runner is process-lived and arms many times; a previous "
                            "arm's arena stream would outlive the run that chose it")

    # 3/4. the hook carries its stream into the drive, and settles both on the
    # fallback.
    hook = body_after(driver, "static bool outerSweepEnqueueHook(",
                      "static bool outerSweepFinishHook(")
    if "gpu::OuterSegmentStream stream" not in hook:
        problems.append("Driver.h: outerSweepEnqueueHook drops the runner's stream.  The "
                        "drive cannot then tell whether it is already on the arena's stream "
                        "(solo) or has to join two (batch)")
    if "sink, stream)" not in hook:
        problems.append("Driver.h: outerSweepEnqueueHook does not pass its stream to "
                        "enqueueDrive; the join would silently not happen and nothing would "
                        "order the segment's updpsi against the sweep that reads its psi")
    if "rasberySyncSegmentStream(stream)" not in hook:
        problems.append("Driver.h: the enqueue hook's fallback settles only the arena "
                        "stream.  The blocking drive it takes reads _dhat and _psi, which "
                        "this outer's mirrors are filling on the RUNNER's stream")

    # 5/6. the enqueue path's three enabling properties.
    enqueue = body_after(backend, "bool CudaBatchArena::enqueueSweeps(",
                         "bool CudaBatchArena::finishSweeps(")
    for needle, why in (
            ("stream_mutex", "the claim that makes the capture exclusive"),
            ("stage_lane", "the per-slot staging lane"),
            ("cudaEventRecord", "the join to the caller's stream")):
        if needle not in enqueue:
            problems.append("CudaBICGBackend.cu enqueueSweeps: missing %s (%s) -- without it "
                            "a batch cannot run the stream-ordered sweep at all" %
                            (needle, why))
    for accessor in ("stageActive()", "stageSweepHalt()", "stageAssemblyActive()",
                     "stageSlotMap()"):
        if accessor not in backend:
            problems.append("CudaBICGBackend.cu: %s is gone.  The four fleet masks are "
                            "page-locked memcpyAsync SOURCES; sharing one buffer between "
                            "launches means the second launcher rewrites the first one's "
                            "in-flight DMA" % accessor)

    # 7. the nodal arena honours the binding, so the bridge is not back.
    arena = body_after(nodal, "class NodalArena {", "std::mutex  g_nodal_arena_mutex;")
    if "canonicalElidesUpload" not in arena or "canonicalElidesDownload" not in arena:
        problems.append("CudaXsReconBackend.cu: the batched nodal arena no longer consults "
                        "the canonical elision predicate, so a batch segment has to keep the "
                        "jnet bridge -- 2 x nsurf*ng doubles per OUTER, which is exactly the "
                        "cost a wider budget exists to amortise")
    if "canonicalNodalIsHonoured" in driver:
        problems.append("Driver.h: the `is it honoured` predicate is back in the arm; it "
                        "keeps the bridge for every batch run")

    # 8. the two arms have not merged.
    graph = strip_comments(read("src", "CudaOuterGraph.cu"))
    if "stream_sweep ? outerSegmentBudget() : 1u" not in graph:
        problems.append("CudaOuterGraph.cu: the runner no longer forces budget 1 under a "
                        "synchronising sweep hook.  That arm still exists -- no arena, the "
                        "Wielandt warm-up, a refused enqueue -- and under it outer i+1 would "
                        "be enqueued before the transition could publish outer i's halt")

    if problems:
        for problem in problems:
            print("batch outer budget contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("batch outer budget contract: PASS (stream arm decided by the stream, arena stream "
          "solo-only, hook carries it, both streams settled on the fallback, claim + lanes + "
          "join in place, nodal binding honoured, budget-1 arm intact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
