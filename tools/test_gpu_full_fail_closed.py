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

  5. THE ALLOWANCE LIST IS EXHAUSTIVE AND ARGUED.  Every refusal reason that can
     reach an outer seam is EITHER on kGpuFullAllowedOuterRefusals with a
     written rationale, OR reaches a guard that fails the case.  There is no
     third state.  The reasons that were considered and refused -- `batch_mode`
     above all -- are pinned as NOT allowed, because that is the entry a future
     reader is most likely to add for the wrong reason.

  6. NO SEAM IS COUNT-ONLY.  On host 181 at 8919331 the outer seam counted 71
     host outer bodies, printed `contract_pass:false`, and exited 0.  A seam
     that cannot unwind where it stands now DEFERS -- it latches and the caller
     raises at a point it declares safe -- so "cannot throw here" no longer
     means "does not fail".  Every DEFER site must have a RAISE_PENDING that
     can reach it.

  7. `contract_pass:false` IMPLIES A NONZERO EXIT whenever the gate is on.  The
     per-case throw is the primary mechanism and keeps the rest of a batch
     running; gpufull::enforceExitCode is the backstop that makes the RUN fail,
     and all three of main.cpp's branches must apply it.

  8. THE GATE IS NOT IN Driver.h's kArmEnv.  It cannot move a trajectory: it
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
for member in ("Cmfd", "Outer", "Nodal", "FlatXs", "Xe", "Ppr", "Cram",
               "Th", "Search", "FlatXsStream", "NodalConsts"):
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
check("if (!required()) return;" in note_body,
      "gpufull::note() throws unconditionally; the default must stay fail-open")
# ... and nothing may allocate PER FALLBACK.  `first_violation` is named with
# the gate OFF as well as on -- `contract_pass:false` is printed either way, and
# a null beside it is a receipt that says a seam fired and refuses to say which
# -- so the no-allocation rule moved from "behind the gate" to "behind the
# armed flag": the FIRST fallback of a run builds one string, every one after it
# reads one atomic and returns.
check("nameFirstFallback(which, where, why);" in note_body,
      "gpufull::note() does not name the first fallback; with the gate off the receipt "
      "prints contract_pass:false beside first_violation:null, which is the reading that "
      "cost the PPR campaign a release")
check("recordFirstViolation" not in note_body,
      "gpufull::note() builds the violation text itself; there must be ONE writer of the "
      "first-fallback slot (nameFirstFallback) or the gate-on and gate-off paths can "
      "record different things")
name_body = header[header.find("inline void nameFirstFallback("):]
name_body = name_body[:name_body.find(chr(10) + "}")]
check("firstViolationArmed().load" in name_body and
      name_body.index("firstViolationArmed().load") < name_body.index("recordFirstViolation"),
      "gpufull::nameFirstFallback() builds the text before testing the armed flag; the "
      "default fallback path would then allocate a string per fallback")

# The DEFERRING seam: counts, latches, and does NOT throw where it stands.
defer_body = header[header.find("inline void noteDeferred("):]
defer_body = defer_body[:defer_body.find(chr(10) + "}")]
check("count(which)" in defer_body,
      "gpufull::noteDeferred() does not count; a deferred violation still has to move "
      "the receipt in case the raise point is never reached")
check("throw" not in defer_body,
      "gpufull::noteDeferred() throws.  It exists for the ONE seam that cannot unwind "
      "where it stands -- the CMFD enqueue hook inside a live segment and a possibly "
      "open graph capture")
check("slot.armed" in defer_body,
      "gpufull::noteDeferred() latches nothing; raisePending() would have nothing to "
      "raise and the seam would be count-only again")
check("thread_local" in header,
      "the deferred-violation slot is not thread-local.  In --batch-mode a process-wide "
      "latch fails whichever worker reaches the next safe point first, which is a "
      "different deck with a clean record (plan Sec 6.3 item 4)")

raise_body = header[header.find("inline void raisePending()"):]
raise_body = raise_body[:raise_body.find(chr(10) + "}")]
check("if (!slot.armed) return;" in raise_body,
      "gpufull::raisePending() does not short-circuit when nothing is latched; it is "
      "called per segment on a path that is almost always clean")
check("throw violation" in raise_body,
      "gpufull::raisePending() does not throw; the deferral would never become a failure")

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
    # WP23.  The device nodal-CONSTANTS arm declines by declining the whole
    # drive, so without its own guard at the same seam it is reported as "the
    # nodal arm fell back" and the nine coefficient uploads quietly return.
    ("src/Nodal.cpp",
     "if (!TryDriveGpu())",
     "NodalConsts",
     "the device nodal-constants arm declined and the nine arrays are uploaded again"),
]

# Every one of these tokens, wherever it appears, is a fallback: the CRAM
# predictor/corrector seams and the three places the outer segment stops
# delegating.  A count is not an allowlist -- each occurrence is checked.
SEAM_TOKENS = [
    # WP23.  Both host-resolver calls under the stream arm: the deck the arm
    # cannot serve, and the call whose device phase refused.  Either one is
    # 1.0 s of host work that the arm's whole claim is about not doing.
    ("src/XSSet.cpp", "BuildFlatXsStream(unrodded);", "FlatXsStream", 2),
    ("src/XSSet.cpp", "++_cram_host_fallbacks;", "Cram", 2),
    ("src/Driver.h", "gpu_outer_armed = false;", "Outer", 3),
    ("src/Driver.h", "gpu::noteOuterSegmentRefusal(gpu_outer_why);", "Outer", 2),
]

WINDOW = 8  # lines either side of the anchor the guard may sit in
# 8 and not 6: every guard now carries a written reason -- which allowance the
# refusal was matched against, or why it defers -- and the comment that carries
# it is longer than the three lines the first version needed.  The rule this
# encodes is `the guard is adjacent to the seam`, not `within six lines`; what
# it refuses is a guard that drifted to another branch, and eight lines of
# comment does not reach one.


# The spellings that FAIL A CASE for a fallback.  RASBERY_GPU_FULL_COUNT is
# deliberately NOT here: it only counts, which is the WP1 gap host 181 found.
FAILING_MACROS = ("RASBERY_GPU_FULL_GUARD", "RASBERY_GPU_FULL_GUARD_IF",
                  "RASBERY_GPU_FULL_GUARD_ALLOWED", "RASBERY_GPU_FULL_DEFER_ALLOWED")
_MACRO_RE = re.compile(r"RASBERY_GPU_FULL_(GUARD_ALLOWED|DEFER_ALLOWED|GUARD_IF|GUARD)\(")


def guard_names(window: str, subsystem: str) -> bool:
    """Does *window* invoke a case-failing guard macro FOR *subsystem*?

    Every spelling is accepted, and only for the subsystem asked about:
        RASBERY_GPU_FULL_GUARD(Nodal, ...)
        RASBERY_GPU_FULL_GUARD_IF(<predicate>, Nodal, ...)
        RASBERY_GPU_FULL_GUARD_ALLOWED(Outer, ...)   -- reason-aware, throws
        RASBERY_GPU_FULL_DEFER_ALLOWED(Outer, ...)   -- reason-aware, latches

    The last two consult kGpuFullAllowedOuterRefusals and fail for every reason
    that is not on it; the allowance section below proves that fall-through, so
    accepting them here does not weaken the seam scan.
    """
    for match in _MACRO_RE.finditer(window):
        tail = window[match.end():]
        if match.group(1) == "GUARD_IF":  # the subsystem is the second argument
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

# ------------------------------------------------- NO SEAM IS COUNT-ONLY -----
# THIS IS THE HOST-181 DEFECT, AS A RULE.  At 8919331 exactly one seam used the
# count-only escape -- the CMFD enqueue hook, which genuinely cannot throw where
# it stands -- and the consequence was 71 host outer bodies, `contract_pass:false`
# and exit 0.  "Cannot unwind here" is a real constraint; "therefore the case
# survives" was never a consequence of it.  The seam DEFERS now: it latches and
# the caller raises at a point it declares safe.  So the escape has no legitimate
# user left, and any reappearance is a seam quietly opting out again.
for rel, text in sources.items():
    for i, line in enumerate(text.splitlines(), 1):
        if "RASBERY_GPU_FULL_COUNT(" in line:
            check(False,
                  f"{rel}:{i} uses the count-only escape.  No seam may: a seam that "
                  f"cannot unwind where it stands uses RASBERY_GPU_FULL_DEFER_ALLOWED "
                  f"and its caller raises at the next safe point, so the case still "
                  f"fails (this is exactly the gap host 181 found at 8919331)")

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


# ===========================================================================
# THE ALLOWANCE LIST (plan Sec 6.3 item 5)
# ===========================================================================
#
# EVERY REFUSAL REASON THAT CAN REACH AN OUTER SEAM IS EITHER ON THE LIST WITH A
# WRITTEN RATIONALE, OR FAILS THE CASE.  There is no third state, and the third
# state is exactly what host 181 measured at 8919331: 71 refusals that were
# neither argued for nor fatal.
bicg_h  = read("src/BICGCMFD.h")
outer_h = read("src/CudaOuterGraph.h")

NL = chr(10)


def enum_reasons(text: str, fn: str) -> list:
    """The reason STRINGS a name() function can return, in declaration order."""
    at = text.find(fn)
    if at < 0:
        return []
    body = text[at:]
    end = body.find(NL + "}")
    body = body[:end] if end > 0 else body[:4000]
    return re.findall(r'return\s+"([a-z0-9_]+)"\s*;', body)


check("kGpuFullAllowedOuterRefusals" in header,
      "src/GpuFullContract.h declares no kGpuFullAllowedOuterRefusals; the outer seam "
      "would be back to `every refusal is fatal` or `no refusal is`, and neither is true "
      "(plan Sec 6.3 item 5 asks for the list to be MANAGED, not assumed)")

list_block = header[header.find("inline constexpr AllowedRefusal kGpuFullAllowedOuterRefusals"):]
list_block = list_block[:list_block.find("};") + 2]
ALLOWED = re.findall(r'\{"([a-z0-9_]+)",', list_block)

check(ALLOWED == ["wielandt_warmup"],
      "the outer allowance list is " + repr(ALLOWED) + ".  Adding or removing an entry "
      "is a contract change and must be argued in the header comment AND here: the only "
      "reason that qualifies today is the Rayleigh warm-up, which has no device "
      "implementation to decline and which the CMFD seam already excludes by "
      "construction (the Cmfd guard sits inside `if (!cap && canEnqueueDrive())`)")

# Each entry carries a rationale, and one that says something.
for reason in ALLOWED:
    tail = list_block[list_block.find('"' + reason + '"') + len(reason) + 2:]
    rationale = "".join(re.findall(r'"([^"]*)"', tail[:tail.find("}")]))
    check(len(rationale) >= 80,
          "the allowance for " + repr(reason) + " carries no real rationale (" +
          repr(rationale) + ").  An entry on this list is a documented exception to plan "
          "Sec 6.3 item 3; if it cannot be argued in a sentence it is not an exception, "
          "it is a hole")

# THE REASONS CONSIDERED AND REFUSED.  Pinned by name, because these are the
# entries a future reader is most likely to add for the wrong reason.
REFUSED_CANDIDATES = {
    "batch_mode":
        "since Task 18-lite the predicate is `batch_width > arena_slots`, i.e. no seat "
        "on the device for this deck -- the pre-arm gate skips the arm and SolveLoop "
        "runs its HOST outer body for the whole solve.  That is plan Sec 6.3 item 3, "
        "not an exemption from it",
    "sweep_arm_off":     "an arm that was asked for and is not there (item 1)",
    "no_cuda_solver":    "an arm that was asked for and is not there (item 1)",
    "not_two_group":     "an arm that was asked for and is not there (item 1)",
    "stage_prep_failed": "a CUDA-side decline (item 2)",
    "geometry_mismatch": "the arena holds another deck's shape; the host outer runs",
    "launch_failed":     "a CUDA call inside the segment failed (item 2)",
    "no_residency":      "the arm was configured and never handed its buffers (item 1)",
}
for reason, why in REFUSED_CANDIDATES.items():
    check(reason not in ALLOWED,
          repr(reason) + " was added to kGpuFullAllowedOuterRefusals.  It does not "
          "qualify: " + why)

# EXHAUSTIVE OVER BOTH LADDERS THAT REACH A SEAM.  The hostfree and graph
# ladders are deliberately NOT here: they choose between two DEVICE arms, and
# where one of them does lead to host numerics (`sweep_wont_enqueue`) the host
# body is entered through the enqueue seam, which asks BICGCMFD for the finer
# reason and decides there.
ENQUEUE_REASONS = enum_reasons(bicg_h, "enqueueRefusalName(EnqueueRefusal r)")
SEGMENT_REASONS = enum_reasons(outer_h, "outerRefusalName(OuterSegmentRefusal r)")
check(len(ENQUEUE_REASONS) >= 6,
      "could not read BICGCMFD::enqueueRefusalName's reasons out of src/BICGCMFD.h (got "
      + repr(ENQUEUE_REASONS) + "); this scan cannot claim to be exhaustive")
check(len(SEGMENT_REASONS) >= 13,
      "could not read outerRefusalName's reasons out of src/CudaOuterGraph.h (got "
      + repr(SEGMENT_REASONS) + "); this scan cannot claim to be exhaustive")
check("wielandt_warmup" in ENQUEUE_REASONS,
      "BICGCMFD::EnqueueRefusal no longer spells `wielandt_warmup`, but the allowance "
      "list still matches on that string -- the allowance would silently stop applying "
      "and every warm-up would fail the case")
for reason in ALLOWED:
    check(reason in ENQUEUE_REASONS or reason in SEGMENT_REASONS,
          "the allowance list admits " + repr(reason) + ", which no outer refusal ladder "
          "can produce.  A dead allowance is a hole waiting for a rename to fall into")

# The DISPATCH is what makes "not on the list" mean "fails the case".  Read it,
# so the exhaustiveness above is a property of the code and not of this file.
fail_body = header[header.find("inline void noteAllowedOrFail("):]
fail_body = fail_body[:fail_body.find(NL + "}")]
check("allowedOuterRefusalIndex(why)" in fail_body and "note(which, where, why)" in fail_body,
      "gpufull::noteAllowedOrFail does not fall through to note() for a reason that is "
      "not on the allowance list; `not allowed` would stop meaning `fails`")
defer_dispatch = header[header.find("inline void noteAllowedOrDefer("):]
defer_dispatch = defer_dispatch[:defer_dispatch.find(NL + "}")]
check("allowedOuterRefusalIndex(why)" in defer_dispatch and
      "noteDeferred(which, where, why)" in defer_dispatch,
      "gpufull::noteAllowedOrDefer does not fall through to noteDeferred() for a reason "
      "that is not on the allowance list")
check("which == Subsystem::Outer" in fail_body and
      "which == Subsystem::Outer" in defer_dispatch,
      "the allowance list is consulted for subsystems other than Outer.  It is the OUTER "
      "seam's list; one shared by seams that never agreed on it is a way to exempt a "
      "seam by accident")
check("count(Subsystem::Outer)" not in fail_body,
      "gpufull::noteAllowedOrFail counts directly instead of delegating to note(); a "
      "later edit could then drop the throw and this scan would not notice")

# NEGATIVE CONTROLS for the list reader itself: it must SEE an entry that is
# there.  A parser that returns [] for everything would pass every check above.
_probe = re.findall(r'\{"([a-z0-9_]+)",',
                    '{"wielandt_warmup", "r"},\n    {"batch_mode", "r"},')
check(_probe == ["wielandt_warmup", "batch_mode"],
      "negative control failed: the allowance-list reader does not see a two-entry list, "
      "so `batch_mode is not on the list` above proves nothing")

# --------------------------------------- the deferred seam has a raiser -------
# A DEFER WITH NO RAISE IS A COUNT, which is the 181 defect wearing a new name.
defers = occurrences(driver, "RASBERY_GPU_FULL_DEFER_ALLOWED(")
raises = occurrences(driver, "RASBERY_GPU_FULL_RAISE_PENDING();")
check(defers >= 1,
      "src/Driver.h no longer defers anywhere.  If the enqueue seam learned to throw in "
      "place, say so here; if it went back to counting, that is the 181 defect")
check(raises >= 4,
      "src/Driver.h has " + str(defers) + " deferring seam(s) and only " + str(raises) +
      " RASBERY_GPU_FULL_RAISE_PENDING() call(s).  Both loops (ReconvergeFlux and "
      "SolveLoop) must raise on BOTH runSegment paths -- the delegated branch and the "
      "refusal branch -- or a latched violation rides to the end of the run and the case "
      "that caused it finishes clean")
_rf_at = driver.find("void ReconvergeFlux")
_sl_at = driver.find("void SolveLoop")
check(_rf_at >= 0 and _sl_at > _rf_at,
      "src/Driver.h: ReconvergeFlux/SolveLoop are not both present in that order; the "
      "per-loop raise-site scan cannot run")
if _rf_at >= 0 and _sl_at > _rf_at:
    for name, body in (("ReconvergeFlux", driver[_rf_at:_sl_at]),
                       ("SolveLoop", driver[_sl_at:])):
        check(body.count("RASBERY_GPU_FULL_RAISE_PENDING();") >= 2,
              "src/Driver.h: " + name + " raises the deferred violation fewer than "
              "twice.  The delegated branch and the refusal branch are two different "
              "returns from runSegment and both are safe points -- the segment is over, "
              "its stream is drained and any capture is closed")

# ------------------------------- contract_pass:false => nonzero exit ----------
# THE POINT OF THE WHOLE GATE.  On host 181 the receipt said contract_pass:false
# and the process exited 0.
check("inline int enforceExitCode(" in header,
      "src/GpuFullContract.h has no enforceExitCode; `contract_pass:false` would go back "
      "to being a receipt field nobody consumes")
enforce = header[header.find("inline int enforceExitCode("):]
enforce = enforce[:enforce.find(NL + "}")]
check("if (!required() || contractPass()) return exit_code;" in enforce,
      "gpufull::enforceExitCode does not short-circuit on `gate off OR contract passed`; "
      "with the gate off a fallback must stay a fallback -- the host body is the "
      "reference path and a benchmark that crashes teaches nothing")
check("[RASBERY][FAIL]" in enforce,
      "gpufull::enforceExitCode fails the run without emitting [RASBERY][FAIL]; the "
      "harness greps for that tag, and a silent nonzero exit is worse than none")
check("firstViolation()" in enforce,
      "gpufull::enforceExitCode does not name the first violation; the whole 181 "
      "investigation existed because `contract_pass:false` named no site")
check(main.count("gpufull::enforceExitCode(std::cout, exit_code)") == 3,
      "src/main.cpp applies the run-level gate " +
      str(main.count("gpufull::enforceExitCode(std::cout, exit_code)")) +
      " times; all THREE branches -- evaluator, batch and serial -- must, the way each "
      "already prints the [RASBERY][GPU_FULL] receipt")
check(main.count("exit_code = rasbery::gpufull::enforceExitCode") == 3,
      "src/main.cpp calls enforceExitCode without assigning its result; the run would "
      "print the failure and still exit 0, which is the defect verbatim")

# The receipt gained the two fields a reader needs in order to act on it.
for _field in ("allowed_refusals", "first_violation"):
    check(_field in receipt,
          "the GPU_FULL receipt omits " + _field + ".  `allowed_refusals` is how an "
          "allowance that fires ten thousand times becomes visible instead of being "
          "subtracted silently; `first_violation` is the site the 181 receipt could not "
          "name")

if failures:
    for problem in failures:
        print("gpu-full fail-closed contract: FAIL " + problem, file=sys.stderr)
    raise SystemExit(1)
print("gpu-full fail-closed contract: PASS")
