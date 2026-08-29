#!/usr/bin/env python3
"""Contract gate: no host buffer is rewritten while a copy FROM it is in flight.

THE BUG THIS PINS.  CudaBatchArena::issueSweepUploads built the participation
mask in `host_active`, uploaded it to `device_active` with cudaMemcpyAsync, and
then INVERTED THAT SAME BUFFER IN PLACE to make the `sweep_halt` mask -- with no
event, no stream sync and no second buffer between the two.  `host_active` is
cudaMallocHost'd, so the first copy is a real asynchronous DMA: the driver is
free to stage it inline at call time (the inversion is then invisible and the
run is correct) or to defer it (device_active then receives the INVERTED mask
and every participant reads active == 0).  Which one it does is a property of
the copy engine's queue state, not of the program, so ONE BINARY FLIPS BETWEEN
THE TWO FROM RUN TO RUN.

Downstream that is not a crash, which is why it hid: initialize_solver_state
computes `halt[m] = (active[m] && !sweep_halt[m]) ? 0 : 1`, so a zeroed
`active` masks the whole BiCGSTAB inner loop of that sweep while the Wielandt
tail (cmfd_wiel_terms / _finalize / cmfd_updls) still advances psi and the
eigenvalue from the un-updated flux.  The drive converges to a NEIGHBOURING
iterate: 1e-14..1e-13 relative drift on every downstream dataset, run to run,
from the first statepoint -- and occasionally the retry / negative-flux branch
it steers into ends in the non-finite abort.

THE RULE.  A host range that is the SOURCE of a cudaMemcpyAsync may not be
written again until that copy is known to have completed.  Inside the arena's
launcher the next completion point is the drain() at the end of the launch, so
in practice: nothing a launcher function hands to cudaMemcpyAsync as a source
may be assigned later in that same function.  One staging buffer per upload is
how the code obeys it; the cost is `slots` uint32s per launch.

WHAT IS CHECKED
  1. The mechanical rule above, over every H2D in the three launcher functions
     that stage per-slot control words (buildSlotMap, issueUploads,
     issueSweepUploads).  Each function's H2D source identifiers are collected,
     and any later assignment to one of them in the same function is a failure.
  2. The specific shape of the fix: sweep_halt is uploaded from its OWN
     page-locked buffer, that buffer is allocated with cudaMallocHost and freed
     with cudaFreeHost, and host_active is never inverted in place again.

Run:  python tools/test_cmfd_async_h2d_snapshot_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The launcher functions that stage per-slot control words for a launch.  These
# are the ones whose bodies run entirely between two stream synchronisations,
# so "assigned later in this function" is exactly "written while in flight".
LAUNCHER_FUNCTIONS = ("buildSlotMap", "issueUploads", "issueSweepUploads")

# Sources that are not host memory the launcher owns, or are provably written
# only before their copy: `sl.*` fields are per-slot staging filled by
# stageSweeps/stageSlot before the launch, and `&sl.eps` is a scalar the launch
# does not touch again.
SOURCE_EXEMPT = re.compile(r"^(&?sl\.|slot\[)")


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def function_body(code: str, name: str) -> str:
    """The brace-matched body of `name`, or "" when it is absent."""
    start = code.find(" " + name + "(")
    if start < 0:
        start = code.find("\n" + name + "(")
    if start < 0:
        return ""
    open_brace = code.find("{", start)
    if open_brace < 0:
        return ""
    depth = 0
    for i in range(open_brace, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_brace : i + 1]
    return ""


def split_args(call: str) -> list[str]:
    """Top-level comma split of an argument list (handles nested parens)."""
    args: list[str] = []
    depth = 0
    current = ""
    for ch in call:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


def h2d_sources(body: str) -> list[tuple[int, str]]:
    """(end offset, source expression) for each HostToDevice cudaMemcpyAsync."""
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"cudaMemcpyAsync\s*\(", body):
        depth = 0
        for i in range(match.end() - 1, len(body)):
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
                if depth == 0:
                    inner = body[match.end() : i]
                    if "cudaMemcpyHostToDevice" not in inner:
                        break
                    args = split_args(inner)
                    if len(args) >= 2:
                        found.append((i, args[1]))
                    break
    return found


def base_identifier(expression: str) -> str:
    """`host_active` from `host_active`, `h.data()`, `&h[0]` ..."""
    cleaned = expression.strip().lstrip("&")
    match = re.match(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", cleaned)
    return match.group(0) if match else cleaned


def main() -> int:
    problems: list[str] = []
    code = strip_comments(read("src", "CudaBICGBackend.cu"))

    # ---- 1. the mechanical rule ------------------------------------------
    for name in LAUNCHER_FUNCTIONS:
        body = function_body(code, name)
        if not body:
            problems.append(
                f"{name}: not found in CudaBICGBackend.cu -- the gate cannot "
                f"check a launcher it cannot locate (renamed? then rename it here)"
            )
            continue
        for end, source in h2d_sources(body):
            identifier = base_identifier(source)
            if SOURCE_EXEMPT.match(source.strip()) or SOURCE_EXEMPT.match(identifier):
                continue
            tail = body[end:]
            # An assignment to the buffer, an element of it, or a bulk rewrite.
            writes = [
                rf"\b{re.escape(identifier)}\s*\[[^\]]*\]\s*=[^=]",
                rf"\b{re.escape(identifier)}\s*=[^=]",
                rf"memset\s*\(\s*{re.escape(identifier)}\b",
                rf"fill\s*\(\s*{re.escape(identifier)}\b",
                rf"\b{re.escape(identifier)}\s*\.\s*(assign|clear|resize)\s*\(",
            ]
            for pattern in writes:
                if re.search(pattern, tail):
                    problems.append(
                        f"{name}: `{identifier}` is the SOURCE of an async H2D and is "
                        f"written again later in the same function, before any "
                        f"synchronisation.  The DMA may read either value.  Stage the "
                        f"second value in its own buffer (see host_sweep_halt)."
                    )
                    break

    # ---- 2. the shape of the fix -----------------------------------------
    sweep = function_body(code, "issueSweepUploads")
    if sweep:
        # Rev.7.1 Task 18: the two buffers are now two STAGING LANES of two
        # page-locked blocks -- `halt` is stageSweepHalt(), `active` is
        # stageActive() -- because the stream-ordered enqueue path can have
        # several launches outstanding and one buffer per arena meant one launch
        # in flight.  The property is unchanged and so is the reason for it: the
        # halt mask must never be the participation mask rewritten in place.
        if not re.search(
            r"cudaMemcpyAsync\s*\(\s*sweep_halt\s*,\s*halt\b", sweep
        ):
            problems.append(
                "issueSweepUploads: the sweep_halt upload must come from its own "
                "page-locked buffer (stageSweepHalt()), not from a rewritten "
                "participation mask"
            )
        if "stageSweepHalt()" not in sweep or "stageActive()" not in sweep:
            problems.append(
                "issueSweepUploads: the two masks must come from this launch's own "
                "staging lanes; sharing one buffer between launches puts the "
                "write-during-DMA back one launcher over"
            )
        if re.search(r"\bactive\s*\[[^\]]*\]\s*=\s*active\s*\[", sweep):
            problems.append(
                "issueSweepUploads: the participation mask is inverted IN PLACE again -- "
                "that is the exact write-during-DMA this gate exists to stop"
            )

    if not re.search(
        r"cudaMallocHost\s*\(\s*reinterpret_cast<void\*\*>\(&host_sweep_halt\)", code
    ):
        problems.append(
            "host_sweep_halt must be page-locked with cudaMallocHost: it is a "
            "per-launch H2D source and a pageable one would put the staging copy "
            "back on the launcher's critical path"
        )
    if not re.search(r"cudaFreeHost\s*\(\s*host_sweep_halt\s*\)", code):
        problems.append("host_sweep_halt is allocated but never cudaFreeHost'd")

    if problems:
        print("FAIL: async H2D snapshot contract")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("PASS: async H2D snapshot contract")
    print(f"  launchers checked: {', '.join(LAUNCHER_FUNCTIONS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
