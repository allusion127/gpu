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
reads the jnet the drive wrote.

The FIRST handover (segment -> nodal) is still a host synchronise and cannot yet
be anything else: the sweep's OBSERVATION hook (BICGCMFD::finishDrive) is a host
read of device memory inside the body, so the host has to see the stream.  W3
item 3 removed the OTHER reason it was there -- the nodal drive's `1.0 / eigv` --
but one host read is as blocking as two, so the synchronise stays until the
observation moves (Task 10 part 3).  See invariant 7.

The SECOND (nodal -> segment) became an event in W3 item 2.  XsReconBackend::solveNodal
records cudaEventRecord(d.nodal_done_event) instead of draining -- but ONLY when
the drive left nothing on the host, i.e. when both canonical downloads were
elided, which is what setCanonicalNodalSegmentMode(true) establishes and nothing
outside a device outer segment asks for.  On the Wielandt warm-up, the CPU body
and any materialised drive it still blocks, because the D2H targets page-locked
Geometry arrays a host reader is about to touch -- invariant 6 one level down.

RULE: each handover is ordered by a synchronise or an event, and the two halves
of the event form -- the record in solveNodal and the cudaStreamWaitEvent in
runSegment, between the drive and the upddhat that reads it -- are pinned
together.  Either half alone is not a slower ordering, it is no ordering.

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

W3 item 4 MOVED WHERE THAT GATE RUNS, not what it compares.  A conditional WHILE
cannot capture a body whose set of memcpy NODES is decided by a host memcmp each
iteration, so the xsnf and dtil stages were hoisted to the segment entry -- on a
proof, not a hope, about which host inputs can move inside a segment:

  Geometry::Phif  moves EVERY outer (BICGCMFD::drive's host loop writes it, and
                  absorbSweepLaunch adopts the device phi into it).  Its gate did
                  NOT move and must not: that is the i-SMR CY02 shape.
  XSSet::_xs      moves inside a segment only through XSSet::ApplyRodCusping.
                  The device arms that write host _xs -- UpdateFlatXS,
                  UpdateEquilibriumXenon -- belong to the Xe and Search steps of
                  the host ladder, and reaching either means the segment has
                  already exited (neither is an Outer -> Outer transition).
  CMFD::_dtil     same: upddtil() is its only writer, called from SolveLoop's
                  entry and from the cusping hook.

So the cusping branch owes BOTH re-stages, and it used to owe only the d-tilde
one -- the cross-section blend was caught by accident, by the next outer's
memcmp, which is exactly the mechanism that was hoisted away.

RULE (item 4): the xsnf and dtil stages are issued in the arm block, not in the
body; the cusping branch re-issues both; the flux stage stays in the body.

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

-------------------------------------------------------------------------------
7. A DEVICE-WRITTEN SCALAR AND ITS UPLOAD ARE MUTUALLY EXCLUSIVE
-------------------------------------------------------------------------------

W3 item 3 took the eigenvalue's round trip out of the body.  The sweep produces
eigv on the device; the FULL nodal drive consumes 1/eigv from a device double
(NodalView::reigv_dev).  Between two device facts sat the host -- the observation
copied eigv up, Driver.h's nodal hook divided, and a staged H2D carried the
quotient back down from a pinned slot the host rewrote before every launch.
k_outer_publish_reigv now does the divide in place.

That makes TWO writers of one 8-byte device slot, and every failure mode here is
one of them being wrong about the other:

  * BOTH WRITE.  The upload lands after the kernel and the drive solves at the
    host's copy of an older eigenvalue.  Harmless while the two agree, which is
    every outer until they do not.
  * NEITHER WRITES.  The declaration is sticky on the backend, so a segment that
    claimed it and returned without clearing it hands every subsequent HOST
    nodal drive a slot nobody updates -- the same shape as the dhat sticky-flag
    bug that ran 169 kngr_238 host outers off a frozen operator.
  * THE KERNEL WRITES THE WRONG OUTER'S.  On the exceptional launches (sweep
    state 0 or 2) the verdict kernel's probe describes a HALF drive; the host
    finishes it and republishAfterHostSweep overwrites the probe.  A publish
    issued only behind the sweep would leave the half drive's reciprocal in the
    slot -- invariant 3's failure, in the scalar.
  * THE KERNEL IS HALT-GATED.  It looks like every other body kernel and it must
    not be one: the halt stops steps that ADVANCE state past a decided exit, and
    this one advances nothing -- but its consumer is a HOST call the halt cannot
    stop, so gating it leaves the previous outer's reciprocal standing on exactly
    the outers the host has to finish.

RULE: the publish kernel is issued behind the sweep and re-issued under
`sweep_host_continued`; it is not halt-gated; the claim is made only on the arm
that reads reigv_dev (`canonical_now`, i.e. Nodal::TryDriveGpu's own predicate)
and is cleared by releaseCanonicalNodal on every path out; and the backend's
upload is gated on the negation of the claim, OUTSIDE the captured graph so the
per-drive flip costs no instantiation.  The divide is correctly rounded
(__ddiv_rn), because the host's `1.0 / eigv` is and the two have to agree bit for
bit.

-------------------------------------------------------------------------------
8. A REFUSAL THE CALLER CAN SEE COMING IS NOT ARMED FOR
-------------------------------------------------------------------------------

The first six were found with the segment RUNNING.  This one is the opposite
failure: the segment never ran at all and the answers moved anyway.

Both call sites armed on the bare RASBERY_GPU_OUTER flag and asked the refusal
ladder afterwards.  Arming is not a query -- armOuterSegment binds residency
(a kernel that patches the shared slot table, a device synchronise, and two H2D
seeds over live arena memory) and then adopts the segment's jnet/phis/flux as
the XS-recon backend's CANONICAL nodal set.  Those three pointers come from ONE
process-wide arena slot, so in --batch-mode every concurrent Driver adopted the
SAME three device buffers and the batched nodal drive stopped having a per-deck
jnet, phis and flux.

MEASURED on the 238 host, M64 manifest, arm X + OUTER=1 b8: ~622 of 708 datasets
differed against OUTER unset, boron by ~1.3 ppm of 1285, k_eff by 7-9e-6, one
dataset by 27.7 absolute -- under a receipt that read segment_launches 0,
device_outers 0, refusals {"batch_mode": 6483}.  Reproduced on a 4-deck local
batch as 184 of 184 accounted datasets, and 0 after the gate.

RULE: the half of the ladder that is a property of the RUN rather than of the
arming (gpu::outerSegmentPreArmRefusal) is asked BEFORE the arm, through the
same outerSegmentRefusal ladder, with this run's batch width.  The post-arm
refusal() still runs and still reports, so the receipt is unchanged.

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
    # drive either drains its own stream before returning, or records an event
    # the segment stream waits on -- W3 item 2 made the second legal, and the
    # rule below pins BOTH halves of it, because either half alone is a race.
    solve = body_of(xsrecon, "bool XsReconBackend::solveNodal")
    if "cudaStreamSynchronize(d.stream)" not in solve:
        problems.append(
            "XsReconBackend::solveNodal must still be able to drain its own stream: "
            "the deferred path is only legal when the drive left NOTHING on the host, "
            "and every other outer (the Wielandt warm-up, the CPU body, a "
            "materialised drive) needs the block"
        )
    if "cudaEventRecord(d.nodal_done_event" in solve:
        # The deferral is on: its condition must be the one that means "no host
        # reader is waiting", and the runner must actually wait on the event.
        defer = solve[solve.index("const bool drain_deferrable") : solve.index(
            "cudaEventRecord(d.nodal_done_event"
        )]
        for needed, why in (
            (
                "canonicalElidesDownload",
                "the deferral may only fire when BOTH canonical downloads were "
                "elided; with either one live, a D2H is in flight into a "
                "page-locked Geometry array a host reader is about to touch, "
                "which is invariant 6 one level down",
            ),
            (
                "hybrid_even",
                "the hybrid arm returns to the host for calculateEven and finishes "
                "in solveNodalPost, so its stream has a host consumer",
            ),
        ):
            if needed not in defer:
                problems.append(
                    f"solveNodal's deferred-drain condition is missing {needed}: {why}"
                )
        if "cudaStreamWaitEvent" not in run:
            problems.append(
                "solveNodal can defer its drain but runSegment never waits on the "
                "event.  That is not a slower ordering, it is no ordering: upddhat "
                "would read the jnet while the drive is still writing it"
            )
        else:
            nodal_at = run.index("enqueue_nodal_drive(m.hooks.ctx")
            dhat_at = run.index("enqueueUpdDhat")
            if not (nodal_at < run.index("cudaStreamWaitEvent") < dhat_at):
                problems.append(
                    "the wait on the nodal completion event must sit between the "
                    "drive that records it and the upddhat that reads what the drive "
                    "wrote"
                )
        driver = read("src", "Driver.h")
        if "hooks.nodal_completion_event" not in driver:
            problems.append(
                "Driver.h installs no nodal_completion_event hook, so the runner can "
                "never see the event solveNodal recorded"
            )


def check_xsnf_elision_is_byte_exact(problems: list[str]) -> None:
    runner = read("src", "CudaOuterGraph.cu")
    backend = read("src", "CudaBICGBackend.cu")

    run = body_of(runner, "bool CudaOuterSegment::runSegment")
    # W3 item 4 moved the gate out of the body and into a lambda the arm block
    # calls once; the RULE is unchanged, so the check follows the code rather
    # than pinning the code to where it used to live.
    try:
        gate = body_of(run, "auto stageXsnf = [&]() -> bool")
    except ValueError:
        problems.append(
            "CudaOuterGraph.cu: runSegment has no stageXsnf gate at all; the per-segment "
            "xsnf upload is either unconditional or gone"
        )
        return

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



def check_device_reigv_has_one_writer(problems: list[str]) -> None:
    """Invariant 7.  Two writers of one 8-byte device slot, and the whole
    correctness argument is that exactly one of them is armed per drive."""
    runner_src = read("src", "CudaOuterGraph.cu")
    header = read("src", "CudaOuterGraph.h")
    xsrecon = read("src", "CudaXsReconBackend.cu")
    driver = read("src", "Driver.h")

    kernel_at = header.find("__global__ void k_outer_publish_reigv")
    if kernel_at < 0:
        problems.append(
            "CudaOuterGraph.h: no k_outer_publish_reigv.  Without it the segment has "
            "no way to put 1/eigv where the nodal drive reads it, and the eigenvalue "
            "goes back to travelling device -> host -> pinned slot -> device"
        )
        return
    kernel = strip_comments(body_of(header, "__global__ void k_outer_publish_reigv"))
    if "__ddiv_rn" not in kernel:
        problems.append(
            "k_outer_publish_reigv does not spell the divide __ddiv_rn.  The host's "
            "1.0 / eigv is correctly rounded; an approximate reciprocal, or a plain "
            "`/` under a build that ever acquires -use_fast_math, is a different "
            "double and the ON arm stops being the OFF arm"
        )
    if "halt" in kernel:
        problems.append(
            "k_outer_publish_reigv is halt-gated.  The halt stops steps that advance "
            "state past a decided exit; this one advances nothing, and its consumer "
            "is a HOST call the halt cannot stop -- so gating it leaves the PREVIOUS "
            "outer's reciprocal standing on exactly the outers (sweep state 0 or 2) "
            "the host has to finish by hand"
        )

    run = strip_comments(body_of(runner_src, "bool CudaOuterSegment::runSegment"))
    try:
        sweep_at = run.index("enqueue_cmfd_sweep(m.hooks.ctx")
        publish_at = run.index("enqueueOuterPublishReigv", sweep_at)
        sync_at = run.index('launchFailed("synchronize before the nodal drive"', publish_at)
    except ValueError:
        problems.append(
            "CudaOuterGraph.cu: the reigv publish is not issued between the sweep that "
            "produces the eigenvalue and the synchronise that precedes the nodal drive "
            "that consumes it.  Earlier and it reads the previous outer's probe; later "
            "and the drive is launched on the other stream before the value has landed"
        )
        return
    if not sweep_at < publish_at < sync_at:
        problems.append(
            "CudaOuterGraph.cu: the reigv publish is out of order against the sweep "
            "and the pre-nodal synchronise"
        )

    reissue = run[run.index("if (m.sweep_host_continued)") :]
    reissue = reissue[: reissue.index('launchFailed("synchronize after the re-issued updjnet"')]
    if "enqueueOuterPublishReigv" not in reissue:
        problems.append(
            "CudaOuterGraph.cu: the reigv publish is not re-issued under "
            "sweep_host_continued.  On sweep state 0 or 2 the verdict kernel's probe "
            "is a HALF drive's; republishAfterHostSweep overwrites it with the numbers "
            "the host finished on, and a slot written only behind the sweep still holds "
            "the half drive's reciprocal when the nodal solve reads it"
        )

    release = strip_comments(body_of(runner_src, "auto releaseCanonicalNodal = [&](bool stream_ordered)"))
    if "setReigvDevice(false)" not in release:
        problems.append(
            "CudaOuterGraph.cu: releaseCanonicalNodal does not clear the reigv claim.  "
            "The claim suppresses an upload on the backend and is sticky, so a segment "
            "that returns without clearing it hands every later HOST nodal drive a slot "
            "nobody writes"
        )
    elif release.index("setReigvDevice(false)") > release.index("if (!m.canonical_nodal_live) return;"):
        problems.append(
            "CudaOuterGraph.cu: releaseCanonicalNodal clears the reigv claim AFTER its "
            "early return.  The claim is a second, independent declaration on the same "
            "backend; a release that skips it whenever the canonical binding was not "
            "live is a release that will one day skip it when it was"
        )
    if "if (canonical_now && m.hooks.canonical_nodal_mode != nullptr)" not in run or            "setReigvDevice(true)" not in run[run.index("if (canonical_now && m.hooks.canonical_nodal_mode != nullptr)") :]:
        problems.append(
            "CudaOuterGraph.cu: the reigv claim is not made under canonical_now.  That "
            "predicate is Nodal::TryDriveGpu's own refusal test and it is the only "
            "thing that says the drive about to run is the FULL device pipeline -- the "
            "one arm that reads reigv_dev at all.  Claiming for a CPU-body or hybrid "
            "drive suppresses an upload the drive still needs"
        )

    solve = body_of(xsrecon, "bool XsReconBackend::solveNodal")
    enqueue_full_at = solve.index("auto enqueue_full = [&]() -> bool")
    enqueue_full = body_of(solve[enqueue_full_at:], "auto enqueue_full = [&]() -> bool")
    if "n_off_reigv" in strip_comments(enqueue_full):
        problems.append(
            "XsReconBackend::solveNodal: the reigv upload is back inside enqueue_full, "
            "i.e. inside the captured graph.  A memcpy NODE cannot be conditional, so "
            "the per-drive flip would have to become a graph key and drop the capture "
            "at every Wielandt warm-up outer -- for eight bytes"
        )
    if "if (!reigv_device)" not in strip_comments(solve):
        problems.append(
            "XsReconBackend::solveNodal: the reigv upload is not gated on the negation "
            "of the device-resident claim, so both writers of the slot are armed and "
            "the drive solves at whichever landed last"
        )
    for hook in ("hooks.nodal_reigv_slot", "hooks.nodal_reigv_mode"):
        if hook not in driver:
            problems.append(
                "Driver.h installs no " + hook + ".  The address and the declaration "
                "are two halves of one handover: the runner writing a slot the backend "
                "still uploads over is waste, and a backend that stops uploading for a "
                "runner that is not writing is wrong"
            )



def check_staged_uploads_have_one_in_body_writer(problems: list[str]) -> None:
    """Invariant 5, item-4 half.  Two of the three operator uploads are staged
    once per segment; the proof that they may be is that exactly one host call in
    the body can move them, and that call re-stages."""
    runner_src = read("src", "CudaOuterGraph.cu")
    run = strip_comments(body_of(runner_src, "bool CudaOuterSegment::runSegment"))

    try:
        loop_at = run.index("for (unsigned int i = 0")
    except ValueError:
        problems.append("CudaOuterGraph.cu: runSegment has no body loop to check")
        return
    arm, body = run[:loop_at], run[loop_at:]

    for call, what in (("stageXsnf()", "xsnf"), ("stageDtil(", "dtil")):
        if call not in arm:
            problems.append(
                "CudaOuterGraph.cu: the " + what + " stage is not issued in the arm "
                "block.  Left in the body it is a memcpy NODE whose existence a host "
                "memcmp decides per iteration, which is a body a conditional WHILE "
                "cannot capture"
            )

    # The flux is the one that CANNOT be staged, and saying so mechanically is
    # what stops the next tidy-up from moving it too.
    if "flux_current" not in body or "flux_current" in arm:
        problems.append(
            "CudaOuterGraph.cu: the flux elision is no longer decided in the body.  "
            "BICGCMFD::drive writes Geometry::Phif on every host-loop outer and "
            "absorbSweepLaunch adopts the device phi into it on the rest, so a "
            "segment-entry answer is stale from the second outer -- the i-SMR CY02 "
            "failure at b8 and b16 that passed at b1"
        )

    # The cusping hook is the only in-body writer of the two staged arrays, so it
    # owes both re-stages.  It already owed the d-tilde one.
    try:
        cusp_at = body.index("m.hooks.apply_cusping(m.hooks.ctx, slot, i)")
        cusp_end = body.index("if (bridge_jnet)", cusp_at)
    except ValueError:
        problems.append(
            "CudaOuterGraph.cu: cannot find the cusping branch; the re-stage rule has "
            "nothing to check"
        )
        return
    cusp = body[cusp_at:cusp_end]
    if "stageXsnf()" not in cusp:
        problems.append(
            "CudaOuterGraph.cu: a cusping that fired does not re-stage xsnf.  "
            "ApplyRodCusping BLENDS the cross sections, and with the gate hoisted to "
            "the segment entry nothing downstream will notice: the next outer's "
            "updpsi builds psi = flux . xsnf . vol from the pre-blend cross sections "
            "while the rest of that outer uses the post-blend ones"
        )
    if "device_dtil" not in cusp:
        problems.append(
            "CudaOuterGraph.cu: a cusping that fired does not re-upload dtil.  It "
            "rebuilt the host d-tilde and the device upddhat reads the device one"
        )

    # The receipt that says how many host rendezvous are left, which is the number
    # Task 10 exists to drive to zero.
    if "in_body_host_syncs" not in runner_src:
        problems.append(
            "CudaOuterGraph.cu: the body's host synchronises are not counted.  Every "
            "other receipt says what the device did; without this one the remaining "
            "round trip is an assertion instead of a number"
        )


def check_refused_segment_is_not_armed(problems: list[str]) -> None:
    """Invariant 8: a refusal the caller can see coming is not armed for.

    ARMING IS NOT A QUERY.  armOuterSegment binds residency -- which patches the
    shared slot table with a kernel, synchronises the device and seeds dhat and
    psi H2D over live arena memory -- and then adopts the segment's
    jnet/phis/flux as the XS-recon backend's CANONICAL nodal set.  Those three
    pointers come from ONE process-wide arena slot (CudaOuterGraph.cu's
    stand-up: `outerArena().slotView(0).jnet` / `.phis`).

    Both call sites armed on the bare `RASBERY_GPU_OUTER` flag and asked the
    refusal ladder AFTERWARDS.  In `--batch-mode` the ladder answered
    `batch_mode` every single time and the segment never ran an outer -- and
    every concurrent Driver had already adopted the SAME three device buffers,
    so the batched nodal drive stopped having a per-deck jnet, phis and flux.
    MEASURED on the 238 host, M64 manifest, arm X + OUTER=1 b8: ~622 of 708
    datasets differed against OUTER unset, boron by ~1.3 ppm of 1285, k_eff by
    7-9e-6, one dataset by 27.7 absolute -- under a receipt that read
    `{"segment_launches":0,"device_outers":0,
      "refusals":{"batch_mode":6483},"idle_reason":"batch_mode"}`.

    RULE: the half of the ladder that is a property of the RUN rather than of
    the arming is asked BEFORE the arm (gpu::outerSegmentPreArmRefusal), it is
    the same ladder rather than a second spelling of it, and it is asked with
    this run's batch width.  The post-arm refusal() still runs and still
    reports, so the receipt is unchanged.
    """
    header = read("src", "CudaOuterGraph.h")
    pre_arm = "outerSegmentPreArmRefusal"
    if pre_arm not in header:
        problems.append(
            "CudaOuterGraph.h does not define outerSegmentPreArmRefusal: there is no "
            "way for a caller to ask whether arming is pointless before it arms, and "
            "arming is what adopts the process-wide canonical nodal set"
        )
        return

    pre_body = strip_comments(body_of(header, "outerSegmentPreArmRefusal"))
    if "outerSegmentRefusal(e)" not in pre_body:
        problems.append(
            "outerSegmentPreArmRefusal must answer through outerSegmentRefusal, not "
            "restate the ladder: two spellings of the same gate are free to disagree, "
            "and the receipt is printed from the other one"
        )
    if "e.batch_width" not in pre_body:
        problems.append(
            "outerSegmentPreArmRefusal must carry the batch width -- batch_mode is the "
            "refusal that armed 6483 times and moved 622 of 708 datasets"
        )
    for satisfied in ("e.residency_bound", "e.have_sweep_hook", "e.have_nodal_hook"):
        if f"{satisfied}   = 1" not in pre_body and f"{satisfied}  = 1" not in pre_body:
            problems.append(
                f"outerSegmentPreArmRefusal must answer {satisfied} with 1: those are "
                "exactly the fields arming exists to satisfy, and refusing on them "
                "before the arm would refuse every run"
            )

    # The receipt has to keep naming batch_mode, which it only does while the
    # ladder ranks it ABOVE the fields a skipped arm leaves false.
    ladder = strip_comments(body_of(header, "outerSegmentRefusal(const OuterSegmentEligibility& e)"))
    if "BatchMode" in ladder and "NoResidency" in ladder and             ladder.index("BatchMode") > ladder.index("NoResidency"):
        problems.append(
            "outerSegmentRefusal ranks NoResidency above BatchMode.  With the arm "
            "skipped there is no residency, so the batch receipt would report "
            "no_residency and stop naming the reason the segment was never eligible"
        )

    driver = read("src", "Driver.h")
    for fn, sig in (
        ("ReconvergeFlux", "static void ReconvergeFlux(SolverContext& ctx"),
        ("SolveLoop", "static void SolveLoop(SolverContext& ctx"),
    ):
        try:
            body = strip_comments(body_of(driver, sig))
        except ValueError:
            problems.append(f"Driver.h: could not find {fn}'s body ({sig!r})")
            continue
        if "armOuterSegment(ctx" not in body:
            continue
        if re.search(r"if\s*\(\s*gpu_outer_enabled\s*\)\s*armOuterSegment", body):
            problems.append(
                f"Driver.h: {fn} arms the outer segment on the bare RASBERY_GPU_OUTER "
                "flag.  Arming adopts the process-wide canonical nodal set, so a run "
                "the ladder is going to refuse -- every batch run -- is perturbed by a "
                "segment that never executes an outer"
            )
        if pre_arm not in body:
            problems.append(
                f"Driver.h: {fn} does not ask gpu::{pre_arm} before arming"
            )
            continue
        if body.index(pre_arm) > body.index("armOuterSegment(ctx"):
            problems.append(
                f"Driver.h: {fn} asks {pre_arm} AFTER armOuterSegment, which is the "
                "ordering that caused the batch perturbation in the first place"
            )
        if not re.search(pre_arm + r"\s*\(\s*rasberyBatchWidth\(\)", body):
            problems.append(
                f"Driver.h: {fn} must pass rasberyBatchWidth() to {pre_arm}; a pre-arm "
                "gate that cannot see the batch is not a gate"
            )


def main() -> int:
    problems: list[str] = []
    check_form_mask_is_mined(problems)
    check_host_ladder_counts_outers(problems)
    check_swallowed_step_is_reissued(problems)
    check_cross_stream_handovers_are_ordered(problems)
    check_xsnf_elision_is_byte_exact(problems)
    check_host_reader_mirror_is_synchronised(problems)
    check_device_reigv_has_one_writer(problems)
    check_staged_uploads_have_one_in_body_writer(problems)
    check_refused_segment_is_not_armed(problems)

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
    print("  5. the xsnf upload elision compares bytes, not a generation,")
    print("     is staged once per segment, and is re-staged by the cusping that")
    print("     is its only in-body writer")
    print("  6. the psi/dhat mirror lands before the host loop that reads it")
    print("  7. the nodal reigv slot has exactly one writer per drive")
    print("  8. a refusal the caller can see coming is not armed for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
