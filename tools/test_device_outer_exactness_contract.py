#!/usr/bin/env python3
"""Contract gate: the ways RASBERY_GPU_OUTER=1 stopped being the host outer.

All three were found on the FULL kngr_238 deck (35 statepoints) after the
trimmed 3-statepoint gate had passed, and all three are invisible to a gate that
only compares converged k_eff.  Each one gets a mechanical rule here, so the
next person to move this code is told which invariant they broke rather than
being handed a 400-dataset h5diff.

-------------------------------------------------------------------------------
1. THE CONTRACTION MASK MUST BE MINED, NOT ASSUMED
-------------------------------------------------------------------------------

CmfdOuterKernel.h's CMFD_OUTER_FORMS records which multiply-adds THE HOST
COMPILER fused, so the device build can reproduce them.  Its own header says the
value is a property of the build machine (0x6 on the authoring box, 0x7 on 238),
and the TESTS were taught to mine the host's value first -- but the PRODUCTION
resolver still returned the baked constant.  On both machines this campaign runs
on, the mined value is 0x7 and the baked one is 0x6, so the device updpsi
rounded `psi += flux*xsnf` in two steps where the host loop fused it.  Every
node, every outer, from the first device outer: 421 of 644 datasets on
kngr_238, and 51 of 644 on the trimmed deck the gate did run.

RULE: cmfdOuterFormsRuntime() mines, and the env override still wins over the
mined value (a binary built on one host and validated against a reference from
another has to be able to say so).

-------------------------------------------------------------------------------
2. THE HOST LADDER'S COUNTERS COUNT OUTERS, NOT PASSES
-------------------------------------------------------------------------------

One pass of SolveLoop's `for (iout...)` is ONE outer on the host arm and up to
`budget` outers on the delegated arm.  `flux_stall` -- "outers since the flux
last converged", the counter the limit-cycle test reads -- was advanced by one
per PASS, so a budget-8 segment ran up to eight times as far past a stalling
boron trial point as the host ever would: kngr_238 statepoint 12 took 3619
outers against the host's 680, the search then sampled a different iterate, and
the whole depletion diverged.  The loop bound `max_iter` had the same shape;
ReconvergeFlux had charged it since Task 9 and SolveLoop had not.

RULE: both are charged by `seg.device_outers`, and the device's own copy of
flux_stall is SEEDED from the host's so the segment ends at the outer the host's
test would have ended it at.

-------------------------------------------------------------------------------
3. A STEP THE HALT SWALLOWED MUST BE RE-ISSUED
-------------------------------------------------------------------------------

cmfd_sweep_verdict raises the segment's halt when the drive did not finish on
the device (sweep state 0 or 2) so the rest of the body does not read a half
sweep.  updjnet is enqueued BEHIND that verdict on the same stream, so it is
already in flight as a no-op by the time the host finishes the drive, and
republishAfterHostSweep taking the halt off cannot un-skip it.  upddhat then
corrected the current against the PREVIOUS outer's jnet.  Three outers out of
11,993 on kngr_238 -- and those three were the entire remaining ON-vs-OFF
divergence once (1) and (2) were fixed.

RULE: republishAfterHostSweep records that the host finished the drive, and the
body re-enqueues updjnet (and re-takes the jnet bridge) under that flag, with a
synchronise before the nodal drive reads it.

-------------------------------------------------------------------------------
4. CROSS-STREAM HANDOVERS ARE ORDERED, NOT HOPED FOR
-------------------------------------------------------------------------------

The nodal drive runs on XsReconBackend's OWN stream while the segment body runs
on the sweep arena's.  Nothing but a host synchronise (or an event) orders the
two, in either direction: the drive reads the jnet updjnet wrote, and upddhat
reads the jnet the drive wrote.  This is currently a pair of host synchronises
and it must stay one of the two.

-------------------------------------------------------------------------------
5. AN UPLOAD ELISION IS DECIDED FROM SOMETHING THAT CANNOT MISS A WRITER
-------------------------------------------------------------------------------

The segment skips the per-outer xsnf H2D when it believes the device copy is
already the host's bytes.  It used to believe that on the strength of
XSSet::hoststateGeneration(), and that generation does not mean what the gate
needed: it means "the device mirror of _xs is stale", NOT "the host bytes of _xs
changed".  XSSet deliberately leaves it alone when a DEVICE arm rewrites host
_xs -- UpdateFlatXS's GPU branch bumps only `if (!gpu_ok || !rodded.empty())`,
and UpdateEquilibriumXenon's GPU branch returns before the bump -- because after
such a write the host and that particular device mirror agree.

The segment's device xsnf is a DIFFERENT buffer (the CMFD arena's), so those are
precisely the writes it must not miss.  CudaBICGBackend.cu has always known
this: the sweep's own xsnf upload is gated on Slot::xsnf_mirror, a byte-exact
shadow, and tools/test_cmfd_sweep_transfer_contract.py forbids the generation
there by name.  The segment used the generation anyway, and updpsi is the reader
that makes it fatal: psi = flux . xsnf . vol runs BEFORE the sweep's own upload
can correct the buffer, so the outer computes its fission source from the
previous cross sections and everything after it from the new ones.

MEASURED on kngr_238 with RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1 (which is
how the 238 host runs it; the authoring box had them unset, which is why b1 was
exact there and not there).  Statepoint 1, outer 28 -- the first boron trial
commit -- ON b1's updpsi produced a psi hash EQUAL to its own outer 27 while the
OFF arm's moved; the run finished 434 of 644 datasets apart, at 12642 outers
against 12017.  With the byte-exact gate: 0 of 644, at 12017.

RULE: the xsnf elision compares BYTES (cuda_transfer::ByteExactMirror), never a
generation, and commits the shadow only from the bytes it actually uploaded.

-------------------------------------------------------------------------------
6. A MIRROR ISSUED FOR A HOST READER IS SYNCHRONISED BEFORE THAT READER RUNS
-------------------------------------------------------------------------------

On the outers whose drive takes the HOST CMFD loop the segment mirrors psi and
dhat back with cudaMemcpyAsync, because that loop reads them.  Both arrays are
page-locked (BICGCMFD::prepareDeviceSweeps leases them), so those are real
asynchronous transfers whose timing belongs to the driver.

The first thing the host path then did was setls(eigv) -- and on the warm-up
that is assembleHostLinearSystem, which reads _dhat for every node
(CMFD::setls: `(-dtil(ige, ls) + dhat(ige, ls)) * area`).  It ran BEFORE
outerSweepEnqueueHook's syncSweepStream(), i.e. with the D2H still in flight
into the buffer it was reading.  The operator came out built from the new
d-hat, the old one, or a mix, and that decided how many Wielandt sweeps the
warm-up took.

INVISIBLE AT A BUDGET OF ONE, which is why it survived every b1 gate: there the
previous segment had already exited, and its exit mirror plus the observation's
synchronise had put that same d-hat in _dhat, so the in-body copy rewrote
identical bytes.  From the second outer of a wider segment there has been no
exit since and the copy is the first arrival of the current d-hat.  kngr_238
statepoint 35 is where it surfaced -- its first outer's warm-up drive stops in
four sweeps, so outer 1 is still inside WIELANDT_WARMUP_SWEEPS and still takes
the host loop.  Two b8 runs of one binary took 245 and 253 outers there with
statepoints 1..34 bit-identical; with the synchronise, 12017 outers and 0 of
644 datasets against OFF, twice.

RULE: an async D2H issued because a host reader is next is synchronised before
that reader runs, and the condition is the same `host_reader_next` the mirrors
themselves are gated on.

Run:  python tools/test_device_outer_exactness_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8", errors="replace") as handle:
        return handle.read()


def body_of(code: str, signature: str) -> str:
    """The brace-balanced body that follows `signature`."""
    start = code.index(signature)
    open_brace = code.index("{", start)
    depth = 0
    for i in range(open_brace, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_brace : i + 1]
    raise AssertionError("unbalanced braces after " + signature)


def strip_comments(code: str) -> str:
    """Code only.  A rule that can be satisfied by the COMMENT explaining it is
    not a rule -- and every one of these invariants has a long comment next to
    it naming the very identifiers the check looks for."""
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def check_form_mask_is_mined(problems: list[str]) -> None:
    kernel = read("src", "CmfdOuterKernel.h")
    mask = read("src", "GpuFormMask.h")

    runtime = body_of(kernel, "inline unsigned long long cmfdOuterFormsRuntime()")
    if "mineCmfdOuterFormsOnThisHost" not in runtime:
        problems.append(
            "cmfdOuterFormsRuntime() does not mine this host's contraction. "
            "A baked mask is a record of the machine it was measured on; when it "
            "disagrees with the build host the device bodies stop reproducing the "
            "host loop and every device outer is wrong in the last bits."
        )
    if "resolveCalibratedFormMask" not in runtime:
        problems.append(
            "cmfdOuterFormsRuntime() must resolve through resolveCalibratedFormMask, "
            "which is where the mined/default/override precedence and the receipt live"
        )

    if "unsigned long long mineCmfdOuterFormsOnThisHost(bool& sound);" not in kernel:
        problems.append(
            "CmfdOuterKernel.h must DECLARE mineCmfdOuterFormsOnThisHost; it may not "
            "define it, because the definition needs the verbatim CPU reference and "
            "that must never share a translation unit with the shipped bodies"
        )

    resolver = body_of(
        mask, "inline unsigned long long resolveCalibratedFormMask"
    )
    if "mined_sound" not in resolver or "value  = mined;" not in resolver:
        problems.append(
            "resolveCalibratedFormMask must return the MINED mask when the mining is "
            "sound -- a measurement of this binary on this machine beats a constant "
            "measured elsewhere"
        )
    # The env override has to be applied AFTER the mined value, or a human who
    # typed the variable is silently overruled by the miner.
    if resolver.index("value  = mined;") > resolver.index("source = \"env\";"):
        problems.append(
            "the environment override must be applied AFTER the mined value, so an "
            "explicit RASBERY_CMFD_OUTER_FORMS still wins"
        )

    miner = read("src", "CmfdOuterFormMiner.cpp")
    if re.search(r'^\s*#\s*include\s+"CmfdOuterKernel\.h"', read("src", "CmfdOuterReference.cpp"), re.M):
        problems.append(
            "CmfdOuterReference.cpp includes the SHIPPED bodies.  With both in one "
            "translation unit gcc common-subexpressions across them and changes the "
            "quotation's own contraction, so the mining scores a reference that is no "
            "longer the production one"
        )
    if "mineStable" not in miner:
        problems.append(
            "CmfdOuterFormMiner.cpp must use the multi-seed mineStable, not a single "
            "descent: a seed that cannot reach zero mismatches is how an "
            "under-determined site announces itself"
        )
    if not os.path.exists(os.path.join(ROOT, "src", "CmfdOuterFormMiner.cpp")):
        problems.append("CmfdOuterFormMiner.cpp must live in src/ so RASBERY links it")


def check_host_ladder_counts_outers(problems: list[str]) -> None:
    driver = read("src", "Driver.h")

    if re.search(r"\+\+flux_stall\s*<=\s*schedule\.max_outer_iter", driver):
        problems.append(
            "SolveLoop advances flux_stall by ONE per pass.  A delegated pass is up to "
            "`budget` outers, so the limit-cycle test fires up to `budget` times too "
            "late and the search samples a different iterate than the host would"
        )
    if not re.search(r"flux_stall\s*\+=\s*outers_this_pass", driver):
        problems.append(
            "SolveLoop must advance flux_stall by the number of OUTERS the pass ran "
            "(`outers_this_pass`), which is 1 on the host body and seg.device_outers "
            "on the delegated arm"
        )
    if not re.search(r"iout\s*\+=\s*static_cast<int>\(seg\.device_outers\)\s*-\s*1", driver):
        problems.append(
            "SolveLoop must charge its own loop bound in outers, the way "
            "ReconvergeFlux has since Task 9 -- otherwise `max_iter` means `max_iter "
            "SEGMENTS` on the ON arm"
        )
    if not re.search(r"s\.flux_stall\s*=\s*static_cast<unsigned int>\(flux_stall\)", driver):
        problems.append(
            "SolveLoop must seed the device machine's flux_stall from its own, or the "
            "segment cannot end at the outer the host's limit-cycle test would"
        )

    runner = read("src", "CudaOuterGraph.cu")
    if "&m.arena.states[slot].flux_stall" not in runner:
        problems.append(
            "runSegment must upload scalars.flux_stall into DeviceSlotState::flux_stall, "
            "exactly as it uploads prev_inner: both are SolveLoop locals the device "
            "carries inside a segment and the host owns between them"
        )


def check_swallowed_step_is_reissued(problems: list[str]) -> None:
    runner = read("src", "CudaOuterGraph.cu")
    backend = read("src", "CudaBICGBackend.cu")

    verdict = body_of(backend, "__global__ void cmfd_sweep_verdict")
    raises_halt = re.search(r"outer_halt\[outer_slot\]\s*=\s*1u", verdict) is not None
    if not raises_halt:
        problems.append(
            "cmfd_sweep_verdict no longer raises the segment halt on an unfinished "
            "drive.  If that is deliberate, this gate's premise has changed and the "
            "re-issue below is dead code -- say so here rather than leaving both"
        )

    republish = body_of(runner, "bool CudaOuterSegment::republishAfterHostSweep")
    if "sweep_host_continued = true" not in republish:
        problems.append(
            "republishAfterHostSweep must record that the HOST finished this drive.  "
            "Taking the halt off does not un-skip the kernels that were enqueued "
            "behind the verdict while it was raised"
        )

    run = body_of(runner, "bool CudaOuterSegment::runSegment")
    if "m.sweep_host_continued = false" not in run:
        problems.append(
            "the flag must be cleared at the top of every outer, or it describes an "
            "older one"
        )
    reissue = re.search(
        r"if\s*\(\s*m\.sweep_host_continued\s*\)\s*\{(.*?)\n        \}", run, re.S
    )
    if reissue is None:
        problems.append(
            "runSegment must re-issue the body step the verdict's halt swallowed"
        )
    else:
        block = reissue.group(1)
        for needed, why in (
            ("enqueueUpdJnet", "updjnet is the step the halt swallows"),
            (
                "cudaStreamSynchronize",
                "the nodal drive reads what it wrote, from another stream and (on the "
                "bridged arm) from a host array -- an enqueue is not enough",
            ),
            ("updjnet_reissued", "the repair has to be countable in the receipt"),
        ):
            if needed not in block:
                problems.append(f"the re-issue block is missing {needed}: {why}")

    # The re-issue has to happen AFTER the observation that finished the drive and
    # BEFORE the nodal drive that consumes it.
    finish = run.index("finish_cmfd_sweep")
    nodal = run.index("enqueue_nodal_drive")
    if reissue is not None and not (finish < reissue.start() < nodal):
        problems.append(
            "the re-issue must sit between the sweep observation (which is what "
            "finished the drive) and the nodal drive (which reads the jnet)"
        )


def check_cross_stream_handovers_are_ordered(problems: list[str]) -> None:
    runner = read("src", "CudaOuterGraph.cu")
    xsrecon = read("src", "CudaXsReconBackend.cu")

    run = body_of(runner, "bool CudaOuterSegment::runSegment")
    # segment stream -> nodal stream: the drive reads what updjnet wrote.
    jnet = run.index("enqueueUpdJnet")
    nodal = run.index("enqueue_nodal_drive")
    between = run[jnet:nodal]
    if "cudaStreamSynchronize" not in between and "cudaStreamWaitEvent" not in between:
        problems.append(
            "nothing orders the segment stream against the nodal backend's stream "
            "between updjnet and the nodal drive.  The drive reads the jnet updjnet "
            "wrote; without a synchronise or a recorded event that is a data race"
        )

    # nodal stream -> segment stream: upddhat reads what the drive wrote.  The
    # drive's own stream has to be drained before it returns.
    solve = body_of(xsrecon, "bool XsReconBackend::solveNodal")
    if "cudaStreamSynchronize(d.stream)" not in solve:
        problems.append(
            "XsReconBackend::solveNodal must drain its own stream before returning: "
            "upddhat is enqueued on the SEGMENT's stream immediately afterwards and "
            "reads the jnet this drive produced"
        )


def check_xsnf_elision_is_byte_exact(problems: list[str]) -> None:
    runner = read("src", "CudaOuterGraph.cu")
    backend = read("src", "CudaBICGBackend.cu")

    run = body_of(runner, "bool CudaOuterSegment::runSegment")
    try:
        gate_at = run.index("const bool xsnf_current")
    except ValueError:
        problems.append(
            "CudaOuterGraph.cu: runSegment has no xsnf_current gate at all; the "
            "per-outer xsnf upload is either unconditional or gone"
        )
        return
    gate = run[gate_at : run.index(";", run.index("bump(counters().xsnf_uploads_elided)"))]

    if "resident_xsnf.matches" not in gate:
        problems.append(
            "CudaOuterGraph.cu: the xsnf elision is not decided from a byte-exact "
            "shadow.  hoststateGeneration() cannot see a DEVICE arm rewriting host "
            "_xs (XSSet.cpp's UpdateFlatXS / UpdateEquilibriumXenon GPU branches), "
            "and updpsi reads the device xsnf before the sweep's own byte-exact "
            "upload can correct it -- kngr_238 sp1 outer 28 with RASBERY_GPU_XSRECON=1"
        )
    if "xs_generation" in gate:
        problems.append(
            "CudaOuterGraph.cu: the xsnf elision still reads a generation.  A "
            "generation records what the host DECLARED; only the bytes record what "
            "the host WROTE, and the device arms write without declaring"
        )
    if "ByteExactMirror<double> resident_xsnf" not in runner:
        problems.append(
            "CudaOuterGraph.cu: the segment keeps no byte-exact shadow of the xsnf it "
            "uploaded, so its gate has nothing sound to compare against"
        )
    if "resident_xsnf.commit(" not in runner:
        problems.append(
            "CudaOuterGraph.cu: the xsnf shadow is never committed, so the elision can "
            "only ever say `upload` and the counter is dead"
        )
    if "resident_xsnf.invalidate()" not in body_of(
        runner, "bool CudaOuterSegment::bindResidency"
    ):
        problems.append(
            "CudaOuterGraph.cu: bindResidency does not invalidate the xsnf shadow.  A "
            "rebind may hand over different device memory, and a shadow describing the "
            "old buffer would elide the upload that fills the new one"
        )

    # The two arms upload the SAME device buffer (the arena's xs_xsnf); they must
    # not disagree about what makes it stale.
    if "sl.xsnf_mirror" not in backend:
        problems.append(
            "CudaBICGBackend.cu: the sweep's xsnf upload no longer uses a byte-exact "
            "mirror.  Both arms write the arena's xs_xsnf; a generation on either "
            "side is the sp1-outer-28 divergence again"
        )


def check_host_reader_mirror_is_synchronised(problems: list[str]) -> None:
    runner = read("src", "CudaOuterGraph.cu")
    run = strip_comments(body_of(runner, "bool CudaOuterSegment::runSegment"))

    try:
        loop_at = run.index("for (unsigned int i = 0")
        # The LAST of the two in-loop mirrors, so the guard slice below holds
        # only the synchronise's own condition -- anchoring on the dhat one puts
        # the psi mirror's own `if (host_reader_next ...)` inside it and any
        # condition at all would then look right.
        mirror_at = run.index('launchFailed("mirror psi to the host", rc)', loop_at)
        hook_at = run.index("enqueue_cmfd_sweep(m.hooks.ctx", mirror_at)
    except ValueError:
        problems.append(
            "CudaOuterGraph.cu: cannot find the in-loop psi/dhat mirror followed by the "
            "sweep hook; invariant 6 has nothing to check and the ordering is unpinned"
        )
        return

    between = run[mirror_at:hook_at]
    if "cudaStreamSynchronize" not in between:
        problems.append(
            "CudaOuterGraph.cu: nothing synchronises the stream between the in-loop "
            "psi/dhat mirror and the sweep hook.  Those are async D2Hs into PAGE-LOCKED "
            "host buffers, and the hook's very first host call -- setls -> "
            "assembleHostLinearSystem -- reads _dhat.  That is the kngr_238 sp35 b8 "
            "run-to-run divergence"
        )
        return

    guard = between[: between.index("cudaStreamSynchronize")]
    if "host_reader_next" not in guard:
        problems.append(
            "CudaOuterGraph.cu: the synchronise before the sweep hook is not conditioned "
            "on host_reader_next.  It has to fire for exactly the outers that issued a "
            "mirror: weaker and the host loop reads a buffer with a DMA in flight, "
            "stronger and it drains the stream on the 99% of outers whose drive is "
            "enqueued and issues no mirror -- the round trip the stream-ordered sweep "
            "exists to remove"
        )

    # The host side must not be relied on to sync for itself: outerSweepEnqueueHook's
    # own syncSweepStream() comes AFTER setls, which is the read that raced.
    hook = strip_comments(body_of(read("src", "Driver.h"), "static bool outerSweepEnqueueHook"))
    if "setls" in hook and "syncSweepStream" in hook and             hook.index("setls") < hook.index("syncSweepStream") and             "host_reader_next" not in guard:
        problems.append(
            "Driver.h: outerSweepEnqueueHook calls setls before syncSweepStream, and "
            "nothing upstream has drained the mirror setls reads"
        )


def main() -> int:
    problems: list[str] = []
    check_form_mask_is_mined(problems)
    check_host_ladder_counts_outers(problems)
    check_swallowed_step_is_reissued(problems)
    check_cross_stream_handovers_are_ordered(problems)
    check_xsnf_elision_is_byte_exact(problems)
    check_host_reader_mirror_is_synchronised(problems)

    if problems:
        print("FAIL: device outer exactness contract")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("PASS: device outer exactness contract")
    print("  1. the CMFD outer contraction mask is mined on this host, not baked")
    print("  2. SolveLoop's flux_stall and loop bound are charged in outers")
    print("  3. the step the sweep verdict's halt swallows is re-issued")
    print("  4. both cross-stream handovers around the nodal drive are ordered")
    print("  5. the xsnf upload elision compares bytes, not a generation")
    print("  6. the psi/dhat mirror lands before the host loop that reads it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
