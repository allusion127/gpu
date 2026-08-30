#!/usr/bin/env python3
"""WP14 -- the outer segment's exit-reason census and the V2 rendezvous elisions.

WHAT THIS DEFENDS, AND WHY EACH HALF NEEDS DEFENDING.

  1. THE CENSUS.  `escapes{}` could not say why a segment stopped: three of the
     five phase-changing CmfdOuterActions -- Xenon, ThermalHydraulics, Search --
     carry `escape = None` (src/CmfdOuterKernel.h), so the largest bucket in
     every receipt was "none" and it named nothing.  `exit_reasons{}` splits it
     by DeviceOuterSegmentState::next_phase.  The contract is that the histogram
     is bumped from the SAME struct the escape histogram is bumped from, at the
     same site, and that the receipt prints it.

  2. THE ELISIONS.  RASBERY_GPU_OUTER_SEGMENT_V2 removes two host calls at the
     segment exit -- the second cudaStreamSynchronize and a 32-byte D2H -- and
     both are argued from ONE fact: the loop broke at the top of a pass, on a
     cudaStreamSynchronize of that stream that returned success, with nothing
     enqueued since.  The flag that carries that fact is `observed_exit`, and
     the contract is that it is (a) set ONLY at that break, (b) RETIRED where a
     repair pass puts a transition back on the stream, and (c) the only thing
     besides the env flag that either elision consults.

  3. THE OFF PATH.  With the flag unset both elisions must be textually the code
     that was there before: an async accumulator copy plus a `host-free exit`
     synchronise, and an `exit segment state` D2H.  A refactor that "simplified"
     the else-branch away would make the OFF arm a different program from the
     one the digest was measured on.

Every check is re-run against a MUTATED copy of the source that breaks the
property it defends; a check that cannot fail is not a check.

Run:  python tools/test_outer_segment_v2_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

CU = os.path.join(SRC, "CudaOuterGraph.cu")
HDR = os.path.join(SRC, "CudaOuterGraph.h")
STUB = os.path.join(SRC, "CudaOuterGraphStub.cpp")
DRIVER = os.path.join(SRC, "Driver.h")

FLAG = "RASBERY_GPU_OUTER_SEGMENT_V2"


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# lexing: comments are NOT code
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    """Drop // and /* */ comments and KEEP everything else, string literals
    included -- the properties below are about statements AND about the tags
    those statements pass to the ledger, and a tag lives inside a literal.  A
    property that could be satisfied by a comment is one this file would be
    lying about, which is the only thing the stripping is for."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find(chr(10), i)
            i = n if j < 0 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                if text[i] == "\\":
                    out.append(text[i:i + 2])
                    i += 2
                    continue
                out.append(text[i])
                if text[i] == q:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# the checks -- each takes the sources dict and returns True when it PASSES
# ---------------------------------------------------------------------------

Sources = dict


def load() -> Sources:
    return {
        "cu": read(CU),
        "hdr": read(HDR),
        "stub": read(STUB),
        "driver": read(DRIVER),
    }


def c_exit_reason_histogram(s: Sources) -> bool:
    """exit_reasons is bumped from next_phase, bounds-tested, beside escapes."""
    code = strip_comments(s["cu"])
    if "counters().exit_reasons[seg_out.next_phase]" not in code:
        return False
    if "seg_out.next_phase < static_cast<std::uint32_t>(kDevicePhaseCount)" not in code:
        return False
    # Same site as the escape histogram: within a few lines of it, so the two
    # can never be bumped from different observations.
    esc = code.find("counters().escapes[seg_out.escape]")
    phs = code.find("counters().exit_reasons[seg_out.next_phase]")
    return esc >= 0 and phs > esc and (phs - esc) < 600


def c_exit_reason_printed(s: Sources) -> bool:
    """The receipt prints exit_reasons{} with the phase NAMES, non-zero only."""
    code = strip_comments(s["cu"])
    if "exit_reasons" not in code or "outerExitPhaseName(static_cast<DevicePhase>(i))" not in code:
        return False
    # The map opens in the receipt builder and the name function renders its
    # keys: the two must be within one loop of each other, not merely both
    # present somewhere in a 3,600-line file.
    key = code.find('exit_reasons\\":{')
    nam = code.find("outerExitPhaseName(static_cast<DevicePhase>(i))")
    return key >= 0 and nam > key and (nam - key) < 900


def c_phase_name_total(s: Sources) -> bool:
    """outerExitPhaseName covers every DevicePhase, so a new phase breaks the
    build rather than silently rendering as '?'."""
    hdr = strip_comments(s["hdr"])
    m = re.search(r"inline const char\* outerExitPhaseName\(DevicePhase p\)\s*\{(.*?)\n\}",
                  hdr, re.S)
    if not m:
        return False
    body = m.group(1)
    slot = read(os.path.join(SRC, "GpuSlotControl.h"))
    e = re.search(r"enum class DevicePhase\s*:\s*std::uint32_t\s*\{(.*?)\}", slot, re.S)
    if not e:
        return False
    names = []
    for line in e.group(1).splitlines():
        line = strip_comments(line).strip()
        if not line:
            continue
        nm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", line)
        if nm:
            names.append(nm.group(1))
    return bool(names) and all(("DevicePhase::%s:" % nm) in body for nm in names)


def c_observed_exit_set_once(s: Sources) -> bool:
    """`observed_exit = true` appears EXACTLY once, and inside the exit
    observation's `m.h_seg->exit != 0u` break -- next to the discovery-pass
    bump, which is the only other statement that break makes."""
    code = strip_comments(s["cu"])
    if code.count("observed_exit = true") != 1:
        return False
    i = code.find("observed_exit = true")
    window = code[max(0, i - 500):i]
    return (
        "m.h_seg->exit != 0u" in window
        and "bump(counters().discovery_passes)" in window
    )


def c_observed_exit_retired_by_repair(s: Sources) -> bool:
    """The repair pass puts a transition back on the stream, so it must retire
    the flag BEFORE it enqueues the tail -- otherwise the exit observation would
    read a pinned word the repair's own transition has since overwritten."""
    code = strip_comments(s["cu"])
    # The DECLARATION is also `observed_exit = false`; the one that matters is
    # the assignment inside the repair block, so the search starts there.
    i = code.find("if (m.sweep_host_continued) {")
    if i < 0:
        return False
    j = code.find("observed_exit = false", i)
    k = code.find("runOuterTail(budget,", i)
    return 0 <= i < j < k


def c_v2_gate_terms(s: Sources) -> bool:
    """Both elisions are gated on the env flag AND observed_exit; the sync
    elision additionally refuses a batch (a blocking cudaMemcpy is issued on the
    legacy stream)."""
    code = strip_comments(s["cu"])
    drained = re.search(r"const bool v2_exit_drained\s*=\s*([^;]+);", code)
    frompin = re.search(r"const bool v2_seg_from_pin\s*=\s*([^;]+);", code)
    if not drained or not frompin:
        return False
    d, p = drained.group(1), frompin.group(1)
    return (
        "segment_v2" in d and "observed_exit" in d and "batch_width <= 1" in d
        and "segment_v2" in p and "observed_exit" in p and "m.h_seg != nullptr" in p
    )


def c_flag_off_path_intact(s: Sources) -> bool:
    """With the flag off, the exit is the pre-WP14 sequence: an async
    accumulator copy, the `host-free exit` synchronise, and the
    `exit segment state` D2H off d_segments."""
    code = strip_comments(s["cu"])
    return (
        '"sweep accumulator"' in code
        and '"host-free exit"' in code
        and '"exit segment state"' in code
        and "m.d_segments + slot, sizeof(seg_out)" in code
    )


def c_flag_reader_shape(s: Sources) -> bool:
    """One cached reader per process, opt-in, spelled the way every other
    RASBERY_GPU_* reader in the tree is spelled -- and defined in BOTH the CUDA
    arm and the CPU-only stub, because the header declares it."""
    cu = strip_comments(s["cu"])
    stub = strip_comments(s["stub"])
    hdr = strip_comments(s["hdr"])
    if "bool outerSegmentV2Enabled()" not in hdr:
        return False
    for text in (cu, stub):
        m = re.search(r"bool outerSegmentV2Enabled\(\)\s*\{(.*?)\n\}", text, re.S)
        if not m or FLAG not in m.group(1) or "static const bool" not in m.group(1):
            return False
    return True


def c_flag_not_in_arm_env(s: Sources) -> bool:
    """B0 knobs must NOT be in trajectory::kArmEnv -- listing one says it can
    move a trajectory and forks the evaluator's cache for two runs that are the
    same run (src/XferLedger.h's argument, verbatim)."""
    driver = strip_comments(s["driver"])
    m = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", driver, re.S)
    return bool(m) and FLAG not in m.group(1)


def c_pass_census(s: Sources) -> bool:
    """segment_passes is bumped at the TOP of the loop (so a WHILE-consumed pass
    counts) and discovery_passes only at the observation break."""
    code = strip_comments(s["cu"])
    if code.count("counters().segment_passes") != 1:
        return False
    if code.count("counters().discovery_passes") != 1:
        return False
    loop = code.find("for (unsigned int i = 0; i < budget; ++i) {")
    sp = code.find("bump(counters().segment_passes)")
    graph = code.find("if (graph_arm && i == 1u)")
    dp = code.find("bump(counters().discovery_passes)")
    obs = code.find('"exit observation"')
    return 0 <= loop < sp < graph and 0 <= obs < dp


def c_no_host_clock(s: Sources) -> bool:
    """WP14 wanted an enqueue/observe split and did NOT take one.
    tools/test_device_outer_state_machine.py bans host clocks from this file --
    a timer on a per-outer path is a tax on the thing it measures -- and the
    number already exists as a ledger row
    (`CudaOuterGraph.cu:runSegment:exit observation`, calls AND ns, under
    RASBERY_XFER_LEDGER).  Pinned here too, because the temptation recurs and
    the receipt that made it unnecessary is this feature's own."""
    code = strip_comments(s["cu"])
    banned = ("std::chrono", "clock_gettime", "cudaEventElapsedTime", "steady_clock")
    if any(b in code for b in banned):
        return False
    # ...and the row it defers to must actually be tagged.
    return '"exit observation"' in code and "xfer::streamSync" in code


CHECKS = [
    ("exit_reasons bumped from next_phase beside escapes", c_exit_reason_histogram),
    ("exit_reasons printed with phase names", c_exit_reason_printed),
    ("outerExitPhaseName is total over DevicePhase", c_phase_name_total),
    ("observed_exit set once, at the observation break", c_observed_exit_set_once),
    ("observed_exit retired by the repair pass", c_observed_exit_retired_by_repair),
    ("both V2 elisions gate on flag + observed_exit", c_v2_gate_terms),
    ("flag-off exit path unchanged", c_flag_off_path_intact),
    ("V2 flag reader: cached, opt-in, CUDA + stub", c_flag_reader_shape),
    ("V2 flag is NOT in trajectory::kArmEnv", c_flag_not_in_arm_env),
    ("pass census: top of loop / observation break", c_pass_census),
    ("no host clock in the segment loop; the ledger row is the number", c_no_host_clock),
]


# ---------------------------------------------------------------------------
# negative controls -- break the property, the check must notice
# ---------------------------------------------------------------------------

def controls(s: Sources):
    def mut(key: str, old: str, new: str) -> Sources:
        m = dict(s)
        assert old in m[key], "control anchor missing: %r" % old
        m[key] = m[key].replace(old, new, 1)
        return m

    return [
        (
            "exit_reasons bumped from a different observation",
            c_exit_reason_histogram,
            mut("cu", "counters().exit_reasons[seg_out.next_phase]",
                "counters().exit_reasons[0]"),
        ),
        (
            "the observe-once predicate loses its observed_exit term",
            c_v2_gate_terms,
            mut("cu", "const bool v2_seg_from_pin = segment_v2 && observed_exit &&",
                "const bool v2_seg_from_pin = segment_v2 &&"),
        ),
        (
            "the sync elision stops refusing a batch",
            c_v2_gate_terms,
            mut("cu", "segment_v2 && observed_exit && batch_width <= 1",
                "segment_v2 && observed_exit"),
        ),
        (
            "observed_exit set somewhere that is not the break",
            c_observed_exit_set_once,
            mut("cu", "bool observed_exit = false;",
                "bool observed_exit = false;\n    observed_exit = true;"),
        ),
        (
            "the repair pass stops retiring observed_exit",
            c_observed_exit_retired_by_repair,
            mut("cu", "observed_exit = false;\n            double* const repair_reigv_slot",
                "double* const repair_reigv_slot"),
        ),
        (
            "the flag-off exit path loses its synchronise",
            c_flag_off_path_intact,
            mut("cu", '"host-free exit"', '"host-free exit (removed)"'),
        ),
        (
            "the V2 flag is listed in kArmEnv",
            c_flag_not_in_arm_env,
            mut("driver", "\"RASBERY_GPU_OUTER_SEGMENT_MAX\",",
                "\"RASBERY_GPU_OUTER_SEGMENT_MAX\",\n    \"%s\"," % FLAG),
        ),
        (
            "a per-outer host timer is reintroduced",
            c_no_host_clock,
            mut("cu", "        if (!runOneOuter(i)) return false;",
                "        const auto t0 = std::chrono::steady_clock::now();" + chr(10) +
                "        if (!runOneOuter(i)) return false;"),
        ),
        (
            "discovery_passes bumped outside the observation",
            c_pass_census,
            mut("cu", "bump(counters().discovery_passes);",
                "bump(counters().discovery_passes);\n                bump(counters().segment_passes);"),
        ),
        (
            "the stub loses its flag reader",
            c_flag_reader_shape,
            mut("stub", "bool outerSegmentV2Enabled() {",
                "bool outerSegmentV2EnabledXX() {"),
        ),
    ]


def main() -> int:
    s = load()
    failures = 0

    print("WP14 outer segment V2 -- source contract")
    print()
    for label, fn in CHECKS:
        ok = fn(s)
        print("%-4s %s" % ("ok" if ok else "FAIL", label))
        if not ok:
            failures += 1

    print()
    print("negative controls")
    for label, fn, mutated in controls(s):
        if fn(mutated):
            failures += 1
            print("FAIL  control NOT caught: %s" % label)
        else:
            print("ok    control caught: %s" % label)

    print()
    print("FAILED (%d)" % failures if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
