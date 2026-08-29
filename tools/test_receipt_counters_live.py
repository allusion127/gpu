#!/usr/bin/env python3
"""F9/F10/F11: A RECEIPT FIELD THAT IS NEVER WRITTEN IS WORSE THAN NO FIELD.

Review doc `docs/WP_PLAN_REVIEW_AND_TRACKER_20260831_KO.md` Sec 3, findings F9,
F10 and F11.  All three are the same defect wearing three hats: a receipt that
reports a CONSTANT, and a reader who cannot tell the constant from a
measurement.

  F9.  `BackendCounters::xs_cpu_fallbacks`, `cmfd_cpu_fallbacks` and
       `nodal_cpu_fallbacks` were declared in src/CudaBICGBackend.h and printed
       from src/BICGSolver.cpp, and nothing in the tree ever incremented one of
       them.  They read 0 on a run that spent every statepoint on the CPU.  WP2
       proposed to judge "the feature engaged" from exactly these fields.

  F10. The FUSED Xe arm (RASBERY_GPU_XSRECON) had no tally at all.  The split
       arm (RASBERY_GPU_XE) charges xe_updates / device_updates /
       host_fallbacks; the fused arm could decline every step of every
       statepoint and `[XE_GPU]` would say nothing.

  F11. `Driver::armOuterSegment` had two bare `return false`s -- an invalid
       resident view, and a failed residency bind.  Neither called
       `noteOuterSegmentRefusal`, so `[OUTER_GPU].refusals` stayed empty and a
       run that lost the outer THERE looked identical to a run that never tried.

WHAT THIS TEST HOLDS, AND THE NEGATIVE CONTROL FOR EACH.

  1. Every `*fallbacks*` field of BackendCounters has at least one WRITE site in
     src/ -- an increment, a `+=`, or an assignment from a live tally.  The
     scan's own negative control: an invented field name must come back with
     zero sites, and deleting the F9 block must make the three fields fail.
     The `*_gpu_calls` fields that are still dead are listed explicitly, so the
     list shrinks visibly when WP2 wires them rather than the test drifting.

  2. The three F9 fields are filled from ONE tally (gpufull::fallbacks) in BOTH
     spellings of the [CUDA][BACKEND_COUNTERS] receipt -- the arena one in
     CudaBICGBackend.cu and the non-arena one in BICGSolver.cpp.  A batch run
     prints the arena spelling, which is the arm the campaign measures, so a fix
     present only in the other one is invisible where it matters.  Two
     independent tallies of one event would be the F13 defect, so the test also
     refuses a second set of counters at the seams.

  3. The fused Xe arm charges a sum-to-whole triple of its own, and charges the
     "asked for" counter BEFORE the attempt.  Negative control: with the
     `tally.fused_*` lines deleted the check must fire.

  4. Every `return` inside armOuterSegment that happens BEFORE the segment is
     armed notes a refusal.  The one documented exception is the feature-off
     line, where nothing was promised and the caller records FeatureOff anyway.
     Negative control: with the noteOuterSegmentRefusal lines deleted, every
     one of those returns must be reported.

Run:  python tools/test_receipt_counters_live.py
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


SOURCES = {p.as_posix()[len(ROOT.as_posix()) + 1:]: p.read_text(encoding="utf-8-sig",
                                                                errors="replace")
           for p in sorted((ROOT / "src").iterdir())
           if p.suffix in (".h", ".hpp", ".cpp", ".cu")}

# ===========================================================================
# F9.  Every fallback field of BackendCounters must have a write site.
# ===========================================================================

backend_h = read("src/CudaBICGBackend.h")
struct_at = backend_h.find("struct BackendCounters")
struct_end = backend_h.find("};", struct_at)
check(struct_at >= 0 and struct_end > struct_at,
      "src/CudaBICGBackend.h: could not find the BackendCounters declaration")
struct_body = backend_h[struct_at:struct_end] if struct_at >= 0 else ""
DECL_LINES = set(struct_body.splitlines())

FIELDS = re.findall(r"std::uint64_t\s+(\w+)\s*=", struct_body)
check(len(FIELDS) > 20,
      f"only {len(FIELDS)} BackendCounters fields parsed; the declaration shape moved and "
      "this scan is no longer looking at the receipt it claims to")


def write_sites(field: str, sources: dict[str, str] | None = None) -> list[str]:
    """Places that WRITE *field*.

    A write is `++x.field`, `field +=` or `field =` (an assignment from a live
    process-wide tally, which is how the phi-mirror and the F9 fields are
    filled).  Lines that only stream the field into a receipt (`<<`) are reads,
    and so is the declaration itself -- the whole point of F9 is that a field
    can be declared with an initialiser, printed, and written nowhere.
    """
    src = SOURCES if sources is None else sources
    patterns = (re.compile(r"\+\+\s*[\w:.>-]*\b" + field + r"\b"),
                re.compile(r"\b" + field + r"\s*(\+=|=\s*($|[^=]))"))
    found = []
    for rel, text in src.items():
        for i, line in enumerate(text.splitlines(), 1):
            if "<<" in line or line in DECL_LINES or field not in line:
                continue
            if any(p.search(line) for p in patterns):
                found.append(f"{rel}:{i}")
    return found


# The positive-engagement counters WP2 still has to wire.  Listed, not ignored:
# when one is wired the list must shrink here, which is the whole difference
# between a known gap and a silent one.
KNOWN_DEAD = {
    "xs_gpu_calls": "F9 residue (WP2): the XS arm counts its fallbacks now, not its calls",
    "nodal_gpu_calls": "F9 residue (WP2): Nodal::drive counts fallbacks, not device drives",
    "th_gpu_calls": "no device T/H arm exists yet (WP9 TH is not-started)",
    "depletion_gpu_calls": "GPU CRAM reports through [CRAM_GPU], not through this field",
}

for field in FIELDS:
    sites = write_sites(field)
    if field in KNOWN_DEAD:
        check(not sites,
              f"BackendCounters::{field} is on the KNOWN_DEAD list but now has a write site "
              f"({', '.join(sites)}); take it off the list so the list keeps meaning "
              f"'nobody writes this'")
        continue
    check(bool(sites),
          f"BackendCounters::{field} is printed in a receipt and written nowhere in src/. "
          f"A field that always reports the same number cannot be read as evidence "
          f"(review doc F9). Wire it, delete it, or record it in KNOWN_DEAD with a reason")

# NEGATIVE CONTROL for the scanner itself.
check(write_sites("unicorn_cpu_fallbacks") == [],
      "the write-site scanner finds writes for a field that does not exist; it cannot "
      "detect the thing it exists to detect")
check(write_sites("graph_fallbacks") != [],
      "the write-site scanner finds no write for graph_fallbacks, which is incremented at "
      "src/CudaBICGBackend.cu; the scanner is broken, not the tree")

# THE THREE F9 FIELDS, specifically.
F9_FIELDS = ("xs_cpu_fallbacks", "cmfd_cpu_fallbacks", "nodal_cpu_fallbacks")
for field in F9_FIELDS:
    sites = write_sites(field)
    check(bool(sites),
          f"BackendCounters::{field} is still never written (review doc F9): it has been "
          f"printed since the backend was written and reports 0 on a run that never "
          f"touched the device")
    check(any("CudaBICGBackend.cu" in s for s in sites),
          f"{field} is not filled in the ARENA spelling of [CUDA][BACKEND_COUNTERS] "
          f"(src/CudaBICGBackend.cu). --batch-mode is the arm the campaign measures, and "
          f"BICGSolver's destructor returns early for an arena run, so a fix only there is "
          f"invisible in every M64 receipt")
    check(any("BICGSolver.cpp" in s for s in sites),
          f"{field} is not filled in the non-arena spelling of [CUDA][BACKEND_COUNTERS] "
          f"(src/BICGSolver.cpp); the two spellings of one receipt would disagree")

# NEGATIVE CONTROL: strip the fill and the three fields must go dead again.
stripped_sources = {rel: "\n".join(line for line in text.splitlines()
                                   if "gf::fallbacks(" not in line)
                    for rel, text in SOURCES.items()}
for field in F9_FIELDS:
    check(write_sites(field, stripped_sources) == [],
          f"negative control failed: {field} still reads as written after every "
          f"`gf::fallbacks(` line was deleted, so the scan is matching something else")

# ONE SOURCE OF TRUTH.  The fields read the WP1 seam tally rather than a second
# set of counters bumped at the seams -- two tallies of one event can disagree,
# and neither receipt can show the disagreement (review doc F13).
for rel in ("src/BICGSolver.cpp", "src/CudaBICGBackend.cu"):
    text = SOURCES.get(rel, "")
    check("gpufull" in text,
          f"{rel} does not read the gpufull seam tally; the three fallback fields are "
          f"filled from something else, or not at all")
    for subsystem in ("Cmfd", "Nodal", "FlatXs", "Xe", "Cram"):
        check(f"Subsystem::{subsystem}" in text,
              f"{rel} does not fold gpufull::Subsystem::{subsystem} into the fallback "
              f"fields; that subsystem's seam declines would be invisible here")
check("#include \"GpuFullContract.h\"" in SOURCES.get("src/BICGSolver.cpp", ""),
      "src/BICGSolver.cpp does not include GpuFullContract.h")
check("GpuFullContract.h" in SOURCES.get("src/CudaBICGBackend.cu", ""),
      "src/CudaBICGBackend.cu does not include GpuFullContract.h")

# Whatever the source, the value has to actually reach the printed receipt.
for rel in ("src/BICGSolver.cpp", "src/CudaBICGBackend.cu"):
    text = SOURCES.get(rel, "")
    for field in F9_FIELDS:
        check(f'\\"{field}\\":' in text or f'"\\"{field}\\":"' in text
              or f'"{field}"' in text.replace('\\"', '"'),
              f"{rel} fills {field} but never prints it")

# ===========================================================================
# F10.  The fused Xe arm gets a tally.
# ===========================================================================

xe_receipt = read("src/XeGpuReceipt.h")
xsset = SOURCES.get("src/XSSet.cpp", "")

FUSED = ("fused_updates", "fused_device_updates", "fused_host_fallbacks")
for field in FUSED:
    check(f"std::atomic<unsigned long long> {field}" in xe_receipt,
          f"src/XeGpuReceipt.h: XeGpuTally has no {field}; the fused RASBERY_GPU_XSRECON "
          f"Xe arm still reports nothing (review doc F10)")
    check(f'\\"{field}\\":' in xe_receipt,
          f"src/XeGpuReceipt.h: {field} is declared but not printed in [RASBERY][XE_GPU]")
    check(f"tally.{field}.fetch_add(1, std::memory_order_relaxed);" in xsset,
          f"src/XSSet.cpp: the fused Xe arm never charges {field}")

# The split arm's own triple must survive untouched: folding the fused arm into
# host_fallbacks would double-charge one Xe step and break
# `xe_updates == device_updates + host_fallbacks`.
for field in ("xe_updates", "device_updates", "host_fallbacks"):
    check(f"tally.{field}.fetch_add(1, std::memory_order_relaxed);" in xsset,
          f"src/XSSet.cpp: the split Xe arm no longer charges {field}")

# ASKED-FOR IS CHARGED BEFORE THE ATTEMPT, or the triple is not an identity.
lines = xsset.splitlines()


def first_line_with(token: str, text_lines: list[str]) -> int:
    for i, line in enumerate(text_lines):
        if token in line:
            return i
    return -1


fused_charge = first_line_with("tally.fused_updates.fetch_add", lines)
fused_try = first_line_with("if (TryUpdateEquilibriumXenonGpu(power, relax, gpu_max))", lines)
fused_fall = first_line_with("tally.fused_host_fallbacks.fetch_add", lines)
check(0 <= fused_charge < fused_try,
      "src/XSSet.cpp: fused_updates is not charged BEFORE TryUpdateEquilibriumXenonGpu; "
      "a receipt whose parts do not sum to its whole cannot be used to find the fallbacks")
check(fused_try < fused_fall,
      "src/XSSet.cpp: fused_host_fallbacks is charged before the attempt, so it counts "
      "steps that succeeded on the device")

# NEGATIVE CONTROL: delete the fused charges and the checks must fire.
stripped_xs = "\n".join(line for line in lines if "tally.fused_" not in line)
for field in FUSED:
    check(f"tally.{field}.fetch_add(1, std::memory_order_relaxed);" not in stripped_xs,
          f"negative control failed: {field} still reads as charged after every "
          f"`tally.fused_` line was deleted")

# ===========================================================================
# F11.  No nameless refusal inside armOuterSegment.
# ===========================================================================

driver = SOURCES.get("src/Driver.h", "")
# Comments are stripped, line for line, before anything below looks for a
# `return`.  The block this finding added QUOTES the line it replaced ("this
# used to be a bare `return false`") so the defect cannot come back by accident,
# and a scanner that reads prose would count that quotation as a fourth silent
# refusal.  Blanking the comment text keeps every line number honest.
dlines = [re.sub(r"//.*$", "", line) for line in driver.splitlines()]
arm_at = first_line_with("static bool armOuterSegment(", dlines)
check(arm_at >= 0, "src/Driver.h: armOuterSegment is gone; this scan has nothing to hold")

arm_end = -1
if arm_at >= 0:
    depth = 0
    opened = False
    for i in range(arm_at, len(dlines)):
        depth += dlines[i].count("{") - dlines[i].count("}")
        if "{" in dlines[i]:
            opened = True
        if opened and depth == 0:
            arm_end = i
            break
check(arm_end > arm_at,
      "src/Driver.h: could not find the end of armOuterSegment by brace balance")

# The one documented exception.  `outerGpuEnabled()` false is not a refusal: the
# arm was never asked for, and the caller records FeatureOff on its own.
FEATURE_OFF = "if (!gpu::outerGpuEnabled()) return false;"
NOTE = "gpu::noteOuterSegmentRefusal("


def unnamed_returns(text_lines: list[str], start: int, stop: int) -> list[int]:
    """Lines in [start, stop] that decline without naming a reason.

    The span searched for the note is everything since the PREVIOUS `return` --
    not a fixed window.  A fixed window is worth nothing here: the two F11
    returns sit a few lines below OTHER refusals that do call the note, so a
    window wide enough to hold a real guard is also wide enough to be satisfied
    by the neighbour's.  That is exactly how these two stayed nameless while a
    reader skimming the function saw notes everywhere.
    """
    out = []
    previous_return = start
    for i in range(start, stop + 1):
        line = text_lines[i]
        if "return" not in line:
            continue
        if "return false" in line and FEATURE_OFF not in line:
            span = "\n".join(text_lines[previous_return + 1:i + 1])
            if NOTE not in span:
                out.append(i + 1)
        previous_return = i
    return out


if arm_end > arm_at:
    silent = unnamed_returns(dlines, arm_at, arm_end)
    check(not silent,
          "src/Driver.h: armOuterSegment declines without calling "
          "noteOuterSegmentRefusal at line(s) " + ", ".join(str(n) for n in silent) +
          ". [OUTER_GPU].refusals stays empty and a run that lost the outer there looks "
          "exactly like a run that never tried (review doc F11)")

    # The two F11 sites, by the reason each must give.
    arm_body = "\n".join(dlines[arm_at:arm_end + 1])
    check("gpu::OuterSegmentRefusal::NoResidency" in arm_body,
          "src/Driver.h: the invalid-resident-view return does not report `no_residency`")
    check("gpu::OuterSegmentRefusal::Unbound" in arm_body,
          "src/Driver.h: the failed rasberyBindOuterResidency return does not report "
          "`unbound`")

    # NEGATIVE CONTROL: delete every note and every non-feature-off return must
    # be reported, including the two this finding added.
    blinded = [("" if NOTE in line else line) for line in dlines]
    silent_blind = unnamed_returns(blinded, arm_at, arm_end)
    check(len(silent_blind) >= 3,
          "negative control failed: with every noteOuterSegmentRefusal line deleted the "
          f"scan still finds only {len(silent_blind)} nameless refusals in "
          "armOuterSegment; it is not looking at the returns it claims to")

# Both reasons must be spellable by the receipt, or they print as `?`.
outer_graph = read("src/CudaOuterGraph.h")
for name in ("no_residency", "unbound"):
    check(f'return "{name}";' in outer_graph,
          f"src/CudaOuterGraph.h: outerRefusalName has no `{name}` case; the refusal this "
          f"finding added would print as `?`")

if failures:
    for problem in failures:
        print("receipt-counters-live contract: FAIL " + problem, file=sys.stderr)
    raise SystemExit(1)
print("receipt-counters-live contract: PASS")
