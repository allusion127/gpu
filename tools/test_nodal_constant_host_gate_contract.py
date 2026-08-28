#!/usr/bin/env python3
"""Contract gate: the host nodal-constants sweep is skipped only when it CANNOT
have work to do.

-------------------------------------------------------------------------------
WHAT THIS GUARDS
-------------------------------------------------------------------------------

Rev.7.1 W3 item 1 removed the last per-outer host body from the device outer
segment.  The [RASBERY][OUTER_GPU] receipt on kngr_238 read

    "host_body_calls":{...,"upddtil":69,"nodal_constants":12017}

with device_outers 12017 -- i.e. every single device outer still ran
Nodal::updateConstant's sweep over all nxyz*ng nodes on the CPU, inside a body
that is otherwise entirely enqueued.

It was NOT removed by porting the sweep to the device.  The coefficient body is
class N1 (CUDA exp differs from glibc by 1 ulp on 3.34% of arguments,
src/NodalConstantKernel.h), so a device producer changes the trajectory of every
outer that genuinely recomputes and ON could no longer equal OFF.  It was removed
by proving the sweep is a NO-OP and skipping it:

    updateConstant's entire input is xs.xsrf, xs.xsdf and Geometry::hmesh
    (immutable after stand-up).  When none moved, every node takes the
    node-scoped early-out, the function returns false everywhere, not one
    coefficient is written and _const_generation does not advance.

That argument is only as good as the answer to "did xsrf/xsdf move", and the
whole point of this file is that the answer is not allowed to be guessed.

-------------------------------------------------------------------------------
1. THE GATE READS macroXsGeneration(), NEVER hoststateGeneration()
-------------------------------------------------------------------------------

hoststateGeneration() means "the device MIRROR of _xs is stale", which is a
different question, and XSSet deliberately does NOT bump it when a device arm
rewrites the host array -- UpdateFlatXS's GPU branch bumps only
`if (!gpu_ok || !rodded.empty())` and UpdateEquilibriumXenon's GPU branch returns
before its bump, because after such a write host and mirror agree.  A constants
gate on that generation would skip a sweep that had work to do on exactly the
runs the campaign measures with RASBERY_GPU_XSRECON=1 RASBERY_GPU_FLATXS=1.
tools/test_device_outer_exactness_contract.py invariant 5 is the record of what
believing otherwise cost the xsnf elision.

-------------------------------------------------------------------------------
2. EVERY WRITER OF _xs.xsrf / _xs.xsdf ANNOUNCES ITSELF
-------------------------------------------------------------------------------

The bump lives AT THE WRITE, inside the function that assigns, not at its
callers -- so no caller's policy about which arm ran can change the answer.  This
check finds every function body in XSSet.cpp that assigns to _xs.xsrf, _xs.xsdf
or the generic _xs[xtype] form (ResetCuspingNodesToBase restores every scalar
column through that spelling, XSDF and XSRF among them) and requires
noteMacroXsWrite() in the same body.  Add a writer without the bump and this
fails by name.

The two device arms are named explicitly as well: they hand the backend the host
_xs column pointers and the backend downloads whole arrays into them, so they are
macro-XS writers that no assignment in XSSet.cpp can reveal.

-------------------------------------------------------------------------------
3. THE PREMISE IS MEASURABLE, NOT ONLY ARGUED
-------------------------------------------------------------------------------

RASBERY_NODAL_CONST_VERIFY=1 runs the sweep even when the gate says skip and
counts every node set that turns out to have moved.  A gate whose soundness
cannot be checked on a real deck is a comment.

-------------------------------------------------------------------------------
4. THE RECEIPT STATES THE CLAIM THE SEGMENT ACTUALLY MAKES
-------------------------------------------------------------------------------

`host_body_calls` is run-wide and can never be zero on a real deck: SolveLoop
builds d-tilde before it delegates (Driver.h:1288, Driver.h:1988 -- that is the
whole of the 69), and a material change rebuilds the constants before the next
segment starts.  The assertion a device outer makes is about the inside of a
segment, so the receipt carries `host_body_calls_in_segment` beside it, measured
as a delta around runSegment by an RAII scope (every early return counted).

Run:  python tools/test_nodal_constant_host_gate_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8", errors="replace") as handle:
        return handle.read()


def strip_comments(code: str) -> str:
    """A rule that the COMMENT explaining it can satisfy is not a rule."""
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def function_bodies(code: str) -> list[tuple[str, str]]:
    """(name, body) for every `Type Class::name(...) { ... }` at file scope."""
    out: list[tuple[str, str]] = []
    for match in re.finditer(r"\n[A-Za-z_][\w:<>,\s*&]*?\bXSSet::(\w+)\s*\(", code):
        name = match.group(1)
        brace = code.find("{", match.end())
        if brace < 0:
            continue
        depth = 0
        for i in range(brace, len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append((name, code[brace : i + 1]))
                    break
    return out


# Assignment to one of the macroscopic arrays the nodal constants read.  The
# generic `_xs[xtype][idx] =` form is included because ResetCuspingNodesToBase
# restores XSDF/XSRF through it without ever naming them.
WRITE_RE = re.compile(
    r"_xs\.(?:xsrf|xsdf)\s*\[[^;]*?\]\s*=(?!=)"
    r"|_xs\.(?:xsrf|xsdf)\s*\.data\s*\("
    r"|_xs\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=(?!=)"
)


def check_every_writer_announces(problems: list[str]) -> None:
    code = strip_comments(read("src", "XSSet.cpp"))
    found_any = False
    for name, body in function_bodies(code):
        if not WRITE_RE.search(body):
            continue
        found_any = True
        if "noteMacroXsWrite()" not in body:
            problems.append(
                f"XSSet::{name} assigns a macroscopic XS column but does not call "
                "noteMacroXsWrite().  Nodal::updateConstantsIfMoved skips its whole "
                "sweep while that generation holds still, so a silent writer here "
                "leaves the nine SENM coefficient arrays stale for the rest of the "
                "statepoint"
            )
    if not found_any:
        problems.append(
            "no function in XSSet.cpp appears to assign _xs.xsrf / _xs.xsdf / "
            "_xs[xt][..] at all -- this check has stopped matching the code it "
            "guards, which is worse than failing"
        )

    # The two device arms write the host columns by DOWNLOAD; no assignment in
    # XSSet.cpp can reveal them, so they are named.
    for fn in ("TryUpdateFlatXSGpu", "TryUpdateEquilibriumXenonGpu"):
        body = dict(function_bodies(code)).get(fn)
        if body is None:
            problems.append(f"XSSet::{fn} has gone; this gate names it by hand")
        elif "noteMacroXsWrite()" not in body:
            problems.append(
                f"XSSet::{fn} must call noteMacroXsWrite().  It hands the backend the "
                "host _xs column pointers and the backend downloads whole arrays into "
                "them; hoststateGeneration() is deliberately NOT bumped for that write "
                "(see UpdateFlatXS's `if (!gpu_ok || !rodded.empty())`), so this "
                "counter is the only announcement the constants gate can see"
            )


def check_gate_reads_the_right_generation(problems: list[str]) -> None:
    header = read("src", "XSSet.h")
    if "macroXsGeneration()" not in header or "_macroxs_generation" not in header:
        problems.append(
            "XSSet no longer publishes macroXsGeneration()/_macroxs_generation, the "
            "only counter that means `the host bytes of xsrf/xsdf moved`"
        )

    nodal = strip_comments(read("src", "Nodal.cpp"))
    if "void Nodal::updateConstantsIfMoved()" not in nodal:
        problems.append(
            "Nodal::updateConstantsIfMoved has gone.  One implementation for both "
            "drives is the point: a gate spelled twice is a gate that can be answered "
            "twice differently"
        )
        return

    start = nodal.index("void Nodal::updateConstantsIfMoved()")
    brace = nodal.index("{", start)
    depth = 0
    body = ""
    for i in range(brace, len(nodal)):
        if nodal[i] == "{":
            depth += 1
        elif nodal[i] == "}":
            depth -= 1
            if depth == 0:
                body = nodal[brace : i + 1]
                break

    if "xs.macroXsGeneration()" not in body:
        problems.append(
            "the constants gate does not read XSSet::macroXsGeneration()"
        )
    if "hoststateGeneration" in body:
        problems.append(
            "the constants gate reads hoststateGeneration().  That counter means `the "
            "device mirror of _xs is stale` and is deliberately NOT bumped when a "
            "device arm rewrites the host array, which is precisely the write the "
            "gate must not miss"
        )
    if "_constants_macroxs_generation" not in body:
        problems.append(
            "the gate does not remember which generation the constants were built "
            "for, so it cannot tell `unchanged` from `never built`"
        )
    if "RASBERY_NODAL_CONST_VERIFY" not in nodal:
        problems.append(
            "RASBERY_NODAL_CONST_VERIFY has gone.  The gate's premise -- every node "
            "would have taken the early-out -- must stay measurable on a real deck"
        )
    if "nodalConstantGateViolations" not in nodal:
        problems.append(
            "the verify mode does not COUNT its violations.  A complaint that "
            "scrolled past in a 60-second run is not evidence"
        )

    # Both drives must go through the one gate; neither may keep a raw sweep.
    raw = re.findall(r"constants_changed\s*\|=\s*updateConstant\(", nodal)
    if len(raw) != 1:
        problems.append(
            f"found {len(raw)} raw updateConstant sweeps in Nodal.cpp; there must be "
            "exactly one, inside updateConstantsIfMoved.  TryDriveGpu and driveBody "
            "both call the gate"
        )
    for caller in ("bool Nodal::TryDriveGpu()", "void Nodal::driveBody()"):
        at = nodal.index(caller)
        nxt = nodal.find("\n}", at)
        if "updateConstantsIfMoved()" not in nodal[at:nxt]:
            problems.append(f"{caller} does not go through the constants gate")


def check_receipt_states_the_segment_claim(problems: list[str]) -> None:
    counters = read("src", "HostOuterBodyCounters.h")
    for needed, why in (
        ("segmentCounters", "the in-segment counters have to exist to be printed"),
        (
            "class SegmentScope",
            "the delta must be taken by an RAII guard: runSegment has a dozen early "
            "returns and a claim measured only on the success path is not a claim",
        ),
    ):
        if needed not in counters:
            problems.append(f"HostOuterBodyCounters.h is missing {needed}: {why}")

    header = read("src", "CudaOuterGraph.h")
    if "host_body_calls_in_segment" not in header:
        problems.append(
            "the receipt no longer prints host_body_calls_in_segment.  The run-wide "
            "number cannot be zero on a real deck (SolveLoop builds d-tilde before it "
            "delegates: Driver.h:1288 and Driver.h:1988 are the whole of kngr_238's "
            "upddtil:69), so a gate held to it would be held to something unreachable"
        )

    runner = strip_comments(read("src", "CudaOuterGraph.cu"))
    if "hostouter::SegmentScope" not in runner:
        problems.append(
            "CudaOuterSegment::runSegment does not open a hostouter::SegmentScope, so "
            "nothing attributes a host body call to the segment it happened in"
        )


def check_arena_prologue_stays_inert(problems: list[str]) -> None:
    """The device nodal-constants phase must not be wired to the physics arena.

    Two independent blockers, both silent if crossed: the arena's Xs/ConstantXs
    slot regions are never written on the production path (stand-up imports five
    GEOMETRY regions and nothing else), and the arena's packing puts diagDI at
    slot 7 while the only reader -- the nodal backend -- puts diagD there.
    """
    kernel = read("src", "CudaNodalConstantKernel.h")
    backend = read("src", "CudaXsReconBackend.cu")

    m = re.search(r"enum NodalConstSlot\s*:\s*int\s*\{(.*?)\}", kernel, re.S)
    if m is None:
        problems.append("NodalConstSlot has gone; the arena's packing is unnamed")
        return
    order = [x.strip().split("=")[0].strip() for x in m.group(1).split(",")]
    order = [x for x in order if x and x != "kNcCount"]
    if order[7:9] != ["kNcDiagDI", "kNcDiagD"]:
        problems.append(
            "NodalConstSlot's diagDI/diagD order moved.  It is pinned by "
            "test/nodal_constant_gpu_replay.cpp and "
            "tools/test_nodal_constant_gpu_contract.py; if the intent is to make it "
            "agree with the nodal backend, change BOTH ends and say so here"
        )
    if not re.search(r"v\.diagD\s*=\s*d\.ndev_dbl \+ d\.n_off_consts \+ 7", backend):
        problems.append(
            "the nodal backend's consts packing moved.  It put diagD at slot 7 and "
            "diagDI at slot 8 -- the OPPOSITE of NodalConstSlot -- and that mismatch "
            "is the reason the arena's nodal-constants phase may not be bound to it"
        )

    runner = read("src", "CudaOuterGraph.cu")
    if "run_nodal_constants" in runner and "INERT ON THE PRODUCTION PATH" not in runner:
        problems.append(
            "CudaOuterGraph.cu keeps the run_nodal_constants prologue without the note "
            "that says why it is inert.  Turning it on reads uninitialised arena Xs / "
            "ConstantXs and writes diagD where the backend reads diagDI; both failures "
            "are finite, plausible and silent"
        )
    if re.search(r"\.run_nodal_constants\s*=\s*true", read("src", "Driver.h")):
        problems.append(
            "Driver.h now sets run_nodal_constants.  The physics arena's Xs and "
            "ConstantXs slot regions are never written on the production path "
            "(rasberyStandUpOuterSegment imports Lklr, Idirlr, Hmesh, Vol, Albedo and "
            "nothing else), so the phase would build the nine coefficient arrays out "
            "of an uninitialised pool block"
        )


def main() -> int:
    problems: list[str] = []
    check_every_writer_announces(problems)
    check_gate_reads_the_right_generation(problems)
    check_receipt_states_the_segment_claim(problems)
    check_arena_prologue_stays_inert(problems)

    if problems:
        print("FAIL: nodal constants host gate contract")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("PASS: nodal constants host gate contract")
    print("  1. every writer of _xs.xsrf/_xs.xsdf calls noteMacroXsWrite()")
    print("  2. the gate reads macroXsGeneration(), never hoststateGeneration()")
    print("  3. RASBERY_NODAL_CONST_VERIFY can measure the gate's premise")
    print("  4. the receipt prints the in-segment claim, not only the run-wide count")
    print("  5. the arena's nodal-constants prologue stays inert and says why")
    return 0


if __name__ == "__main__":
    sys.exit(main())
