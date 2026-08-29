#!/usr/bin/env python3
"""WP1(b): RASBERY_GPU_FULL=1 MUST FAIL THE CASE, NOT THE MEASUREMENT.

Plan Sec 6.3, priority P0, risk ledger Sec 10 row 2 ("GPU flag가 켜졌지만
fallback").

WHAT WAS TRUE BEFORE THIS.  Every GPU arm fails OPEN.  A CUDA error, a shape the
kernel does not serve, a refused batch slot, a graph capture the driver would
not take -- each returns `false` and the caller runs the host body.  That is the
right DEFAULT: the host body is the reference and a benchmark that crashes
teaches nothing.  What it is not is a contract.  A run can declare five GPU arms
and spend every statepoint on the CPU; the only trace is a one-shot stderr line
nobody keeps and, for three of the subsystems, a receipt field that is printed
and never incremented (`BackendCounters::xs_cpu_fallbacks`,
`cmfd_cpu_fallbacks`, `nodal_cpu_fallbacks` -- declared in
src/CudaBICGBackend.h, printed in src/BICGSolver.cpp, written nowhere).  An A/B
built on such a run measures the CPU and reports it as the GPU.

WHAT THIS TEST HOLDS.

  1. THE GATE EXISTS AND IS OFF BY DEFAULT.  A fail-closed contract that
     defaulted ON would turn every legitimate decline -- a fractional rod, a
     non-2-group deck, the MASTER PPR scheme -- into a failed case.

  2. EVERY TOP-LEVEL HOST-FALLBACK SEAM CARRIES THE GUARD.  Not the hundreds of
     inner `return false`s, which all surface at one of these; the seams are the
     places the host body is actually entered.  A new seam without a guard is a
     new silent fallback, and that is what this refuses.  Each anchor below is
     checked with a NEGATIVE CONTROL: the same check run against the same text
     with the guard deleted must fire.

  3. THE VIOLATION IS PER CASE.  main.cpp's batch branch and its serial branch
     both wrap the Driver in try/catch and turn an exception into that job's
     exit code.  A thrown Violation must therefore fail ONE deck; if either
     catch disappeared, it would take the whole batch.

  4. THE RECEIPT IS PRINTED WHETHER OR NOT THE GATE IS ON.  "The arm was on and
     never engaged" must not look like "the arm was off" -- the same G0 rule the
     XSRECON/XE/NODAL/OUTER_GPU receipts already exist for.

  5. THE GATE IS NOT IN Driver.h's kArmEnv.  It cannot move a trajectory: it
     converts a fallback into a failure, so a run that COMPLETES under it took
     the path it would have taken without it.  Same reasoning the list already
     records for RASBERY_GPU_PPR.  Listing it would say it could.

Run:  python tools/test_gpu_full_fail_closed.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        failures.append(f"{rel} does not exist")
        return ""
    return path.read_text(encoding="utf-8-sig")


GUARD = "RASBERY_GPU_FULL_GUARD"

# ------------------------------------------------------------- the header ----
header = read("src/GpuFullContract.h")
check("enum class Subsystem" in header,
      "src/GpuFullContract.h does not declare the subsystem enum")
for member in ("Cmfd", "Outer", "Nodal", "FlatXs", "Xe", "Ppr", "Cram"):
    check(re.search(rf"\b{member}\b", header) is not None,
          f"src/GpuFullContract.h: Subsystem is missing {member} (plan Sec 6.3)")
check("class Violation" in header, "src/GpuFullContract.h declares no Violation type")
check("public std::runtime_error" in header,
      "gpufull::Violation does not derive from std::runtime_error; main.cpp's per-case "
      "`catch (const std::exception&)` would not see it and the batch would abort")
check('"site="' in header or "site=" in header,
      "gpufull::Violation does not name the SITE; 'the nodal arm fell back' is a symptom, "
      "not a place to look")
check("#define RASBERY_GPU_FULL_GUARD(" in header,
      "src/GpuFullContract.h defines no RASBERY_GPU_FULL_GUARD macro; the seam scan below "
      "has no token to look for")

# DEFAULT OFF, and read once.
required_body = header[header.find("inline bool required()"):]
required_body = required_body[:required_body.find("\n}")]
check('std::getenv("RASBERY_GPU_FULL")' in required_body,
      "gpufull::required() does not read RASBERY_GPU_FULL (the plan Sec 6.3 name)")
check('std::getenv("RASBERY_GPU_STRICT")' in required_body,
      "gpufull::required() does not accept the RASBERY_GPU_STRICT alias")
check("static const bool" in required_body,
      "gpufull::required() re-reads the environment on every call; a gate whose answer can "
      "change mid-run is not a contract")
check(re.search(r"required\(\)\s*\{[^}]*return\s+true\s*;", header, re.S) is None,
      "gpufull::required() is hard-wired on; the default must be the current fail-open "
      "behaviour (plan Sec 6.3 keeps the host body as the reference path)")

# The throw is what makes it fail-CLOSED.
note_body = header[header.find("inline void note("):]
note_body = note_body[:note_body.find("\n}")]
check("throw Violation" in note_body,
      "gpufull::note() does not throw under the gate; it would only count")
check("if (required())" in note_body,
      "gpufull::note() throws unconditionally; the default must stay fail-open")

# --------------------------------------------------------------- the seams ---
# (file, anchor, expected subsystem).  The anchor is a line that exists in the
# source today and marks the moment the host body is entered.
SEAMS = [
    ("src/BICGCMFD.cpp",
     "if (driveDeviceSweeps(eigv, flux, errl2)) return;",
     "Cmfd",
     "the CMFD device sweep loop declined and the pristine host BiCGSTAB loop runs"),
    ("src/Nodal.cpp",
     "if (!TryDriveGpu())",
     "Nodal",
     "TryDriveGpu declined and driveBody() runs the whole CPU nodal body"),
    ("src/XSSet.cpp",
     "// Device arm declined: fall through to the reference loop.",
     "FlatXs",
     "the FlatXS device arm declined and the reference reconstruction loop runs"),
    ("src/XSSet.cpp",
     "tally.host_fallbacks.fetch_add(1, std::memory_order_relaxed);",
     "Xe",
     "the split Xe arm declined"),
    ("src/XSSet.cpp",
     "// any failure falls through to the unchanged CPU loop",
     "Xe",
     "the fused Xe arm declined -- the one seam that had no counter at all"),
    ("src/Driver.h",
     "if (!ppr_on_device) {",
     "Ppr",
     "GPU PPR declined and the host reset+drive runs"),
]

# Every one of these tokens, wherever it appears, is a fallback: the CRAM
# predictor/corrector seams and the three places the outer segment stops
# delegating.  A count is not an allowlist -- each occurrence is checked.
SEAM_TOKENS = [
    ("src/XSSet.cpp", "++_cram_host_fallbacks;", "Cram", 2),
    ("src/Driver.h", "gpu_outer_armed = false;", "Outer", 3),
    ("src/Driver.h", "gpu::noteOuterSegmentRefusal(gpu_outer_why);", "Outer", 2),
]

WINDOW = 6  # lines either side of the anchor the guard may sit in


def guard_names(window: str, subsystem: str) -> bool:
    """Does *window* invoke the guard macro FOR *subsystem*?

    Both spellings are accepted, and only for the subsystem asked about:
        RASBERY_GPU_FULL_GUARD(Nodal, ...)
        RASBERY_GPU_FULL_GUARD_IF(<predicate>, Nodal, ...)
    """
    for match in re.finditer(rf"{GUARD}(_IF)?\(", window):
        tail = window[match.end():]
        if match.group(1):  # _IF: the subsystem is the second argument
            if re.match(rf"[^;]*?,\s*{subsystem}\s*,", tail, re.S):
                return True
        elif re.match(rf"\s*{subsystem}\s*,", tail):
            return True
    return False


def guarded(text: str, anchor: str, subsystem: str, occurrence: int = 0) -> bool:
    """Is occurrence #*occurrence* of *anchor* within WINDOW lines of its guard?"""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if anchor in line]
    if occurrence >= len(hits):
        return False
    at = hits[occurrence]
    window = "\n".join(lines[max(0, at - WINDOW):at + WINDOW + 1])
    return guard_names(window, subsystem)


def occurrences(text: str, anchor: str) -> int:
    return sum(1 for line in text.splitlines() if anchor in line)


sources: dict[str, str] = {}
for rel in ("src/BICGCMFD.cpp", "src/Nodal.cpp", "src/XSSet.cpp", "src/Driver.h",
            "src/main.cpp"):
    sources[rel] = read(rel)

for rel, anchor, subsystem, why in SEAMS:
    text = sources.get(rel, "")
    check(occurrences(text, anchor) >= 1,
          f"{rel}: the fallback seam anchor {anchor!r} is gone; this test can no longer "
          f"prove that {why}")
    check(guarded(text, anchor, subsystem),
          f"{rel}: {anchor!r} has no RASBERY_GPU_FULL_GUARD(...Subsystem::{subsystem}...) "
          f"within {WINDOW} lines -- {why}, silently, even under RASBERY_GPU_FULL=1")

for rel, anchor, subsystem, expected in SEAM_TOKENS:
    text = sources.get(rel, "")
    found = occurrences(text, anchor)
    check(found == expected,
          f"{rel}: expected {expected} occurrences of the fallback token {anchor!r}, found "
          f"{found}; a new one is a new seam and needs its own guard, a vanished one means "
          f"this scan no longer covers what it claims to")
    for i in range(found):
        check(guarded(text, anchor, subsystem, i),
              f"{rel}: occurrence {i + 1} of {anchor!r} has no "
              f"RASBERY_GPU_FULL_GUARD(...Subsystem::{subsystem}...) within {WINDOW} lines")

# NEGATIVE CONTROLS.  Strip the guard out of a copy of each file and prove the
# same check fires.  A seam scan that cannot fail is decoration.
for rel, anchor, subsystem, _why in SEAMS:
    stripped = "\n".join(line for line in sources.get(rel, "").splitlines()
                         if GUARD not in line)
    check(not guarded(stripped, anchor, subsystem),
          f"negative control failed: {rel} {anchor!r} still reads as guarded after every "
          f"{GUARD} line was deleted")

for rel, anchor, subsystem, _expected in SEAM_TOKENS:
    stripped = "\n".join(line for line in sources.get(rel, "").splitlines()
                         if GUARD not in line)
    check(not guarded(stripped, anchor, subsystem, 0),
          f"negative control failed: {rel} {anchor!r} still reads as guarded after every "
          f"{GUARD} line was deleted")

# A guard naming the WRONG subsystem must not satisfy an anchor.
check(not guarded("  if (!TryDriveGpu())\n"
                  "      RASBERY_GPU_FULL_GUARD(Ppr, \"x\", \"y\");\n"
                  "  driveBody();\n",
                  "if (!TryDriveGpu())", "Nodal"),
      "negative control failed: a guard naming the wrong subsystem satisfies the nodal seam")

# ---------------------------------------------------- the count-only escape ---
# gpufull::count() does not throw.  Exactly one site is allowed to use it, and
# it needs a written reason; anything else is a seam quietly opting out.
COUNT_ONLY_ALLOWED = {
    ("src/Driver.h", "Outer"):
        "the CMFD enqueue hook runs INSIDE a live device outer segment, where a throw "
        "would unwind past a stream (and possibly a graph capture) that nothing is "
        "written to clean up",
}
for rel, text in sources.items():
    for i, line in enumerate(text.splitlines(), 1):
        if "RASBERY_GPU_FULL_COUNT(" in line:
            which = re.search(r"RASBERY_GPU_FULL_COUNT\(\s*(\w+)", line)
            key = (rel, which.group(1) if which else "?")
            check(key in COUNT_ONLY_ALLOWED,
                  f"{rel}:{i} uses the count-only escape for {key[1]}, which is not on the "
                  f"allowlist; a seam that cannot throw needs a written reason here")

# ---------------------------------------------------------- per-case failure --
main = sources.get("src/main.cpp", "")
check(main.count("catch (const std::exception& error) {") >= 2,
      "src/main.cpp no longer has a per-case catch in BOTH the batch branch and the serial "
      "loop; a GpuFullContract violation would abort the whole run instead of one deck "
      "(plan Sec 6.3 item 4)")
check("job_status[static_cast<std::size_t>(i)] = 1;" in main,
      "src/main.cpp's batch branch no longer turns a caught exception into that job's "
      "status")
check("[RASBERY][FAIL]" in main,
      "src/main.cpp no longer names the failed deck; a violation would be silent")

# ---------------------------------------------------------------- receipt ----
check("[RASBERY][GPU_FULL]" in main,
      "src/main.cpp does not emit the [RASBERY][GPU_FULL] receipt (plan Sec 6.3)")
check(main.count("[RASBERY][GPU_FULL]") >= 2,
      "the [RASBERY][GPU_FULL] receipt is emitted by only one branch; single and batch runs "
      "must both report it, the way [RASBERY][BATCH_HOST][PIN] does")
check("appendGpuFullReceiptFields" in main or "gpufull::appendReceiptFields" in main,
      "src/main.cpp does not use the header's receipt emitter; a second formatting of the "
      "same counters is a second number for one fact")
receipt = header[header.find("inline void appendReceiptFields"):]
for field in ("gpu_full", "contract_pass", "_fallbacks"):
    check(field in receipt, f"the GPU_FULL receipt omits {field}")
check("subsystemName" in receipt,
      "the GPU_FULL receipt does not name its counters from the subsystem table; a "
      "hand-written list would drift from the enum")
check("required() ?" in receipt,
      "the GPU_FULL receipt does not state whether the gate was on; a run that fell back "
      "with the gate off and one that could not have are different runs")

# ------------------------------------------------ not a trajectory knob -------
driver = sources.get("src/Driver.h", "")
arm_env = driver[driver.find("inline constexpr const char* kArmEnv[]"):]
arm_env = arm_env[:arm_env.find("};") + 2]
check("RASBERY_GPU_FULL" not in arm_env and "RASBERY_GPU_STRICT" not in arm_env,
      "RASBERY_GPU_FULL is in Driver.h's kArmEnv.  That list is the knobs that can MOVE a "
      "trajectory, and this one cannot: it turns a fallback into a case failure, so a run "
      "that COMPLETES under it took the path it would have taken without it (the same "
      "reasoning the list already records for RASBERY_GPU_PPR)")
check("RASBERY_GPU_CRAM" in arm_env,
      "RASBERY_GPU_CRAM left kArmEnv; the counterexample that makes the rule above mean "
      "something is gone")

# ------------------------------------------- CMFD stand-up is already closed --
# BICGSolver refuses at construction when RASBERY_GPU was requested and the
# backend is unavailable.  That is the ONE arm that was already fail-closed, and
# turning it into a fallback would be a regression the seam scan cannot see
# (there would be no seam).
solver = read("src/BICGSolver.cpp")
check('throw std::runtime_error("RASBERY_GPU requested but unavailable: "' in solver,
      "src/BICGSolver.cpp no longer refuses to construct when RASBERY_GPU is set and the "
      "CUDA backend is unavailable; the one already-fail-closed arm became fail-open")

if failures:
    for problem in failures:
        print("gpu-full fail-closed contract: FAIL " + problem, file=sys.stderr)
    raise SystemExit(1)
print("gpu-full fail-closed contract: PASS")
