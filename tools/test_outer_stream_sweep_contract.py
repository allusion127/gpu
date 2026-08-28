#!/usr/bin/env python3
"""Contract gate for the stream-ordered CMFD sweep inside the device outer
segment  (Rev.7.1 Task 10 part 2, step 1).

WHAT THIS EXISTS TO CATCH.  The whole point of the change is a NEGATIVE: the
sweep no longer rendezvouses, no longer drains, and no longer carries its verdict
home through the host.  Every one of those is invisible in a result file -- an
enqueue path that quietly grew a `cudaStreamSynchronize` back produces exactly
the same numbers, one rendezvous per outer more slowly, and the receipt still
says `device_outers: 1690`.  So the invariants have to be asserted at source
level or they are not asserted at all.

The nine checks, and the failure each one is written against:

  1. enqueueSweeps NEVER SYNCHRONISES OR DRAINS.  A sync inside it puts the
     per-outer round trip straight back and `sweep_synchronizes = false` becomes
     a lie the receipt cannot see.

  2. enqueueSweeps NEVER READS sweep_out.  The scalar block is filled by a D2H
     that rides the same stream; reading it before the caller's synchronise is
     a use of uninitialised host memory that happens to hold last outer's answer.

  3. enqueueSweeps REFUSES A SECOND PARTICIPANT.  It takes no lock, so two
     arrivals would be two launchers on one stream -- the exact failure the
     rendezvous in solveCommon documents ("NaN flux, invalidated graph captures
     and heap corruption, all of it invisible until batches got wide enough").

  4. drain() IS sync + absorb().  The enqueue arm has to do the same per-slot
     bookkeeping the blocking arm does -- the mirror commits, the non-finite
     latch, the fp32 valve -- and a second copy of it would be free to drift.

  5. THE GATE RUNS BETWEEN THE UPLOADS AND THE LAUNCH.  cmfd_sweep_gate is what
     carries the segment's halt into `sweep_halt`; issued after launch_sweeps it
     would mask nothing, and issued before issueSweepUploads it would be
     overwritten by the mask upload.

  6. THE VERDICT LATCHES ONLY AN UNFINISHED DRIVE.  Sweep states 0 and 2 mean
     the drive is NOT over, so the outer's remaining body would run on a half
     sweep.  States 1 and 3 must NOT latch -- that would end every outer.

  7. THE VERDICT CLEARS nonfinite.  The host probe publish it replaces wrote a
     zeroed struct every outer; a verdict that only wrote the live fields would
     let one non-finite outer latch the flag for the process.

  8. A HOST CONTINUATION REPUBLISHES AND UNLATCHES.  When the host finishes a
     drive the device could not, the probe the verdict kernel published is a
     half drive's and the halt it set is wrong -- both have to be replaced.

  9. THE SEGMENT'S dhat/psi OWNERSHIP FOLLOWS THE ARMED DECISION.  This is the
     bug that killed i-SMR CY02: the residency flags were set by the bind, which
     only asks whether the RUN is configured for the resident sweep, while
     whether a segment ever runs is also a DECK question answered afterwards.  A
     refused deck then elided the dhat and psi H2D of every sweep with nothing
     writing them.

Run:  python tools/test_outer_stream_sweep_contract.py
"""

from __future__ import annotations

import os
import re
import sys

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
    stop = code.find(end, at)
    return code[at:stop if stop > at else len(code)]


def main() -> int:
    problems: list[str] = []

    backend = strip_comments(read("src", "CudaBICGBackend.cu"))
    backend_raw = read("src", "CudaBICGBackend.cu")
    graph_cu = strip_comments(read("src", "CudaOuterGraph.cu"))
    graph_h = strip_comments(read("src", "CudaOuterGraph.h"))
    cmfd = strip_comments(read("src", "BICGCMFD.cpp"))
    driver = strip_comments(read("src", "Driver.h"))

    # ---- 1/2/3: what the enqueue may not do -------------------------------
    enqueue = body_after(backend, "bool CudaBatchArena::enqueueSweeps(",
                         "void CudaBatchArena::syncSweepStream(")
    if not enqueue:
        problems.append("CudaBatchArena::enqueueSweeps is gone; the segment's sweep is back "
                        "to the rendezvous path and sweep_synchronizes must go back to true")
    else:
        for banned, why in (
                ("cudaStreamSynchronize", "a synchronise here IS the per-outer round trip"),
                ("cudaDeviceSynchronize", "a device-wide wait is a rendezvous with "
                                          "everything else on the device too"),
                (".drain(", "drain() synchronises; the enqueue path's caller owns when "
                            "that happens"),
                ("std::unique_lock", "the rendezvous is what this path exists to remove"),
                ("sweep_out", "the scalar block is filled by a D2H on the same stream and "
                              "holds the PREVIOUS outer's answer until the caller syncs")):
            if banned in enqueue:
                problems.append(f"CudaBatchArena::enqueueSweeps uses {banned!r} -- {why}")
        if "inUseCount() > 1" not in enqueue:
            problems.append("CudaBatchArena::enqueueSweeps does not refuse a second "
                            "participant.  It takes no lock, so two arrivals are two "
                            "launchers on one stream -- see solveCommon's `launching` claim")
        order = [enqueue.find(step) for step in
                 ("issueSweepUploads", "enqueueSweepGate", "launch_sweeps",
                  "enqueueSweepVerdict", "issueSweepDownloads")]
        if any(p < 0 for p in order):
            problems.append("CudaBatchArena::enqueueSweeps is missing one of uploads / gate "
                            "/ launch / verdict / downloads")
        elif order != sorted(order):
            problems.append(
                "CudaBatchArena::enqueueSweeps issues its five steps out of order.  The gate "
                "must follow the mask upload that would overwrite it and precede the graph "
                "whose kernels test it; the verdict must precede issueSweepDownloads, which "
                "ends by clearing sweep_halt for the next launch")

    # ---- 4: one absorb, two arms ------------------------------------------
    drain = body_after(backend, "void drain(const int* active_slots, int count) {",
                       "void absorb(")
    if "absorb(active_slots, count);" not in drain:
        problems.append("BatchCore::drain no longer delegates to absorb(); the enqueue arm "
                        "and the blocking arm would then have two copies of the per-slot "
                        "bookkeeping, free to disagree about the non-finite latch")
    finish = body_after(backend, "bool CudaBatchArena::finishSweeps(",
                        "void CudaBatchArena::syncSweepStream(")
    if "absorb(&m, 1)" not in finish:
        problems.append("CudaBatchArena::finishSweeps does not run BatchCore::absorb, so an "
                        "enqueued launch never commits its upload mirrors or latches a "
                        "non-finite flux")
    if "cudaStreamSynchronize" in finish:
        problems.append("CudaBatchArena::finishSweeps synchronises directly.  Its contract is "
                        "that the CALLER already did -- the segment's one observation per "
                        "outer -- and a sync here would make that two.  The one exception is "
                        "issueExceptionalOperatorDownloads, which syncs INSIDE itself and "
                        "only on the Rayleigh hand-back, i.e. only when host arithmetic is "
                        "about to read the operator anyway")
    if "issueExceptionalOperatorDownloads" not in finish or "io.state == 2" not in finish:
        problems.append("CudaBatchArena::finishSweeps does not pull the operator and psi on "
                        "the Rayleigh hand-back; the host branch that finishes that sweep "
                        "would read arrays the assembly kernel wrote and nobody downloaded")

    # ---- 5/6/7: the two kernels -------------------------------------------
    gate = body_after(backend, "__global__ void cmfd_sweep_gate(", "__global__ void "
                      "cmfd_sweep_verdict(")
    if "sweep_halt[m] = 1u" not in gate:
        problems.append("cmfd_sweep_gate does not raise sweep_halt; a halted segment's next "
                        "outer would run a real sweep")
    verdict = body_after(backend, "__global__ void cmfd_sweep_verdict(",
                         "\n// ---------------------------------------------------------")
    if not verdict:
        problems.append("cmfd_sweep_verdict is gone; the segment's probe is back to a host "
                        "publish and the sweep is observing on the host again")
    else:
        if "state == 0 || state == 2" not in verdict:
            problems.append(
                "cmfd_sweep_verdict does not latch the segment halt on sweep state 0 or 2.  "
                "Those are the two the host `while` loop spins on -- the drive is NOT over, "
                "so updjnet and upddhat would read a half sweep")
        if re.search(r"state\s*==\s*[13]", verdict):
            problems.append("cmfd_sweep_verdict branches on sweep state 1 or 3.  Those are "
                            "the FINISHED drive; latching there would end every outer")
        if "nonfinite_out" not in verdict or "*nonfinite_out = 0u" not in verdict:
            problems.append(
                "cmfd_sweep_verdict does not clear the probe's nonfinite flag.  The host "
                "publish it replaces wrote a zeroed struct every outer, so without this one "
                "non-finite outer latches the flag and the next armed segment fails its "
                "first outer on a signal from an abandoned solve")
        if "(state == 2) ? 1u : 0u" not in verdict:
            problems.append("cmfd_sweep_verdict does not publish the Rayleigh signal as "
                            "`state == 2`; that is the latch cmfd_wiel_finalize sets and the "
                            "transition ranks")

    # ---- 8: the host continuation overrules the verdict -------------------
    finish_drive = body_after(cmfd, "bool BICGCMFD::finishDrive(", "\n}\n")
    if "host_continued = true;" not in finish_drive:
        problems.append("BICGCMFD::finishDrive never reports a host continuation, so the "
                        "caller cannot know the device probe describes a half drive")
    republish = body_after(graph_cu, "bool CudaOuterSegment::republishAfterHostSweep(",
                           "bool rasberyBindOuterResidency(")
    if "publishProbe(" not in republish or "d_halt" not in republish:
        problems.append("CudaOuterSegment::republishAfterHostSweep must do BOTH: replace the "
                        "probe the verdict kernel published for the half drive, and release "
                        "the halt it latched on it")
    if republish.find("publishProbe(") > republish.find("d_halt"):
        problems.append("republishAfterHostSweep clears the halt BEFORE the probe is "
                        "replaced; an outer resuming in between computes its convergence "
                        "from the half drive")
    if "outerSweepFinishHook" not in driver or "republishAfterHostSweep" not in driver:
        problems.append("Driver.h has no finish hook that republishes after a host "
                        "continuation")

    # ---- the budget only opens with BOTH halves ---------------------------
    if "finish_cmfd_sweep" not in graph_h:
        problems.append("OuterSegmentHooks has no finish_cmfd_sweep; an enqueue without an "
                        "observation leaves the nodal drive reading last outer's flux")
    if "m.hooks.finish_cmfd_sweep != nullptr" not in graph_cu:
        problems.append("CudaOuterGraph.cu opens the budget without requiring the "
                        "observation half of the sweep")

    # ---- 9: the residency ownership follows the ARMED decision ------------
    #
    # Checked structurally rather than by counting call sites: what matters is
    # that no site sets it to a literal `true`, and that the armed decision is
    # what feeds it.
    for m in re.finditer(r"setOuterSegmentResident\(([^)]*)\)", driver):
        arg = m.group(1).strip()
        if arg == "true":
            problems.append(
                "Driver.h: setOuterSegmentResident(true) -- the flags must follow the ARMED "
                "decision, not the bind.  A deck refused for fractional rods or a critical "
                "search then elides the dhat and psi H2D of every sweep with nothing writing "
                "them, which is how i-SMR CY02 died at statepoint 1 with `device_outers: 0`")
    if "setOuterSegmentOwnership(ctx, gpu_outer_armed)" not in driver:
        problems.append("Driver.h never hands the residency decision the armed flag")
    if driver.count("setOuterSegmentOwnership(ctx, false)") < 3:
        problems.append(
            "Driver.h: a mid-loop disarm does not hand dhat and psi back to the host sweep.  "
            "The same BICGCMFD serves the rest of the loop and the next SolveLoop entry, so "
            "a stale `true` is the same bug one outer later")

    if problems:
        print("outer stream sweep contract: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("outer stream sweep contract: PASS (enqueue is drain-free and single-participant, "
          "one absorb for both arms, gate before the graph, verdict latches only an "
          "unfinished drive, residency follows the armed decision)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
