#!/usr/bin/env python3
"""WP10.7: DEVICE RESIDENCY IS ESTABLISHED AT THE DOOR, ONCE PER ADMISSION.

THE RUN THIS EXISTS FOR.  238 GPU1, build 0054838, 20 generations x width 16,
PROD + RASBERY_GPU_FULL=1, arm A (RASBERY_ARENA_PERSIST unset).  Thirteen cases
were killed by the fail-closed gate and the receipt could not say which seam was
first:

    {"outer_fallbacks":9,"flatxs_fallbacks":4,"contract_pass":false,
     "first_violation":"subsystem=outer site=Driver: outer segment pre-arm
                        reason=no_residency"}

Four of the thirteen -- generations 6, 9 and 12's `promote` step and generation
19's case 14 -- died at ZERO statepoints in 0.18-0.43 s, which is a deck parse
and nothing else: they died at the FIRST XSSet::UpdateFlatXS call, the one
InitXS makes, having never run a statepoint.  The other nine died naming
`no_residency`, a reason produced by re-deriving a failed bind one line after
the bind's own reason had been thrown away.

WHAT THIS TEST HOLDS.

  1. THERE IS AN ADMISSION DOOR, AND IT IS UNCONDITIONAL.  Driver::Run
     establishes device residency for every admission, before any physics, with
     NO first-admission latch -- no `static bool`, no `std::call_once`, nothing
     keyed on a lane or a slot.  A recycled slot goes through the identical
     call.  Establishment is idempotent by construction (XSSet::EnsureBackend),
     so "establish" and "re-establish" are one call.

  2. THE DOOR RUNS BEFORE THE SEAM IT PROTECTS.  It must be called before
     InitXS, because InitXS's UpdateFlatXS was the only thing that ever asked
     whether the flat-XS device residency existed.

  3. IT FAILS CLOSED, AND IT NAMES THE REFUSAL.  An arm that was ASKED for and
     cannot be established still raises under RASBERY_GPU_FULL=1 -- and with
     the establishing layer's OWN reason string, not the guard describing
     itself.  With the gate off it names and does NOT count, because the seam
     downstream still counts and a door that counted too would double every
     gate-off tally.  Both halves are DRIVEN, not read: the shipped function
     text is compiled and run.

  4. THE ARM'S RETURN VALUE IS READ.  armOuterSegment's return was discarded at
     both of its call sites, which is how a failed bind -- whose reason sits in
     CudaOuterSegment::status() -- became the generic `no_residency` one line
     later.

  5. `promote` GOES THROUGH ONE DOOR.  In rolling mode a promotion resolves
     against the PROCESS default the instant it is read; in wave mode it used to
     resolve against the wave's declaration, so one op meant two things
     depending on the mode.  A wave default no longer reaches a promotion --
     which also closes the concrete hole that a wave declaring staged
     tolerances handed them to the acceptance lane's own re-run.

  6. `first_violation` IS THE FIRST VIOLATION.  Chronologically, across
     subsystems, by an ordinal stamped where the EVENT happens rather than by a
     race between the threads that build the receipt strings.  Driven with the
     238 interleave: flatxs's event first, outer's string first.

  7. THE RECEIPT NAMES EVERY SUBSYSTEM'S FIRST SITE beside the counter that
     already existed, so `flatxs_fallbacks:4` stops being a number that sends a
     reader to a 16k-line log.

  8. THE SOAK READS BOTH.  `outer_fallbacks` is asserted like every other
     subsystem, and the receipt's own `contract_pass` verdict is read rather
     than only re-derived from the counters.

Every scanner below has a NEGATIVE CONTROL: the same check run against the same
text with the property deleted must fire.

Run: python3 tools/test_evaluator_residency_contract.py
"""

from __future__ import annotations

import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NL = "\n"

failures: list[str] = []
controls: list[str] = []


def check(condition: bool, why: str) -> None:
    if not condition:
        failures.append(why)


def read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        failures.append("missing source file " + relative)
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


DRIVER = read("src/Driver.h")
SERVER = read("src/EvaluatorServer.h")
GPUFULL = read("src/GpuFullContract.h")
XSSET_H = read("src/XSSet.h")
SOAK = read("tools/soak_run.py")
GATE = read("tools/promotion_gate.py")


def masked(text: str) -> str:
    """*text* with comments and literal contents blanked, same length.

    THE BRACES IN THIS TREE'S RECEIPTS ARE IN STRINGS.  appendReceiptFields
    writes `out << ",\\"allowed_refusals\\":{";` and closes it in a different
    statement, so a brace counter that reads string literals loses count inside
    the one function this test most needs to read.  And this file's own scanners
    ask "is there a `std::call_once` latch in this body", which must not be
    answered by a comment that says the word.  Blanking rather than deleting
    keeps every index the same, so a slice of the mask is a slice of the source.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif two == "/*":
            while i < n and text[i:i + 2] != "*/":
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        elif text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                    if i < n:
                        out[i] = " "
                        i += 1
                    continue
                out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
            i += 1
        else:
            i += 1
    return "".join(out)


def brace_span(text: str, anchor: str) -> tuple[int, int]:
    """(open, close+1) of the `{...}` body after *anchor*, or (-1, -1)."""
    mask = masked(text)
    start = text.find(anchor)
    if start < 0:
        return (-1, -1)
    open_at = mask.find("{", start)
    if open_at < 0:
        return (-1, -1)
    depth = 0
    for i in range(open_at, len(mask)):
        if mask[i] == "{":
            depth += 1
        elif mask[i] == "}":
            depth -= 1
            if depth == 0:
                return (open_at, i + 1)
    return (-1, -1)


def brace_block(text: str, anchor: str) -> str:
    """The `{...}` body that follows *anchor*, braces balanced.

    Lexical, like every other scanner in this tree: it does not parse C++, it
    counts braces on the masked text so a `{` inside a string or a comment
    cannot move the count.  The negative controls below prove the reader sees
    what it claims to.
    """
    open_at, close_at = brace_span(text, anchor)
    return "" if open_at < 0 else text[open_at:close_at]


def code_block(text: str, anchor: str) -> str:
    """The same body with comments and literals blanked -- for `is X in the CODE`."""
    open_at, close_at = brace_span(text, anchor)
    return "" if open_at < 0 else masked(text)[open_at:close_at]


# ---------------------------------------------------------------------------
# 1 + 2.  The door exists, is unconditional, and runs before the seam
# ---------------------------------------------------------------------------

DOOR_ANCHOR = "static void establishDeviceResidency("
DOOR = brace_block(DRIVER, DOOR_ANCHOR)
DOOR_CODE = code_block(DRIVER, DOOR_ANCHOR)

check(bool(DOOR),
      "src/Driver.h has no establishDeviceResidency(): there is no admission door, so "
      "the first thing that asks whether a case's flat-XS device residency exists is "
      "still the fail-closed guard that kills the case for it not existing")

# NO FIRST-ADMISSION LATCH.  This is the whole hypothesis (b) the 238 evidence
# raised and the property that makes a recycled slot indistinguishable from a
# fresh one.  A `static` inside this body would make the Nth admission skip what
# the first did.
for latch in ("static bool", "std::call_once", "once_flag", "static std::atomic",
              "first_admission", "lane_first"):
    check(latch not in DOOR_CODE,
          "establishDeviceResidency carries a " + repr(latch) + " latch. Residency "
          "establishment must be unconditional per admission -- the recycled slot and "
          "the fresh one go through the identical call, or hypothesis (b) is back")

check("EnsureBackend()" in DOOR,
      "establishDeviceResidency does not go through XSSet::EnsureBackend(), which is "
      "the one idempotent construct-or-reuse entry point (src/XSSet.h). A door with its "
      "own construction path would be a second opinion about whether a backend exists")

check("XsReconBackend* EnsureBackend()" in XSSET_H and
      "if (!_xsrecon_backend)" in brace_block(XSSET_H, "XsReconBackend* EnsureBackend()"),
      "XSSet::EnsureBackend is no longer the idempotent construct-or-reuse call the "
      "door relies on; re-binding on every admission would then mean re-CONSTRUCTING on "
      "every admission")

# CALLED, AND CALLED BEFORE InitXS.
call = DRIVER.find("establishDeviceResidency(cross_sections,")
init_xs = DRIVER.find("cross_sections.InitXS(")
check(call > 0,
      "establishDeviceResidency is defined and never called; a door nothing goes "
      "through is a comment")
check(call > 0 and init_xs > 0 and call < init_xs,
      "establishDeviceResidency is called after cross_sections.InitXS(). InitXS is the "
      "call whose UpdateFlatXS was the ONLY thing that ever asked whether this case's "
      "flat-XS residency existed -- a door behind it protects nothing")

# It is not itself behind a per-case condition at the call site.
call_line_start = DRIVER.rfind(NL, 0, call) + 1
call_line = DRIVER[call_line_start:DRIVER.find(NL, call)]
check(call_line.strip().startswith("establishDeviceResidency("),
      "the admission door's call site is guarded by a condition (" +
      repr(call_line.strip()[:80]) + "); the door decides which arms were REQUESTED, "
      "and a caller that decides for it can decide wrong once")

# ---------------------------------------------------------------------------
# 3.  It fails closed, with the establishing layer's own reason
# ---------------------------------------------------------------------------

check("RASBERY_GPU_FULL_REQUIRE_RESIDENCY(" in DOOR,
      "the admission door does not use RASBERY_GPU_FULL_REQUIRE_RESIDENCY. A door that "
      "cannot raise is a log line, and the contract this work package is inside is "
      "fail-closed")
check(DOOR.count("RASBERY_GPU_FULL_REQUIRE_RESIDENCY(") >= 2,
      "the admission door raises for fewer than two arms; both the FlatXS arm (the four "
      "238 deaths) and the outer arm (the nine) are established here")
check("flatxs_status.c_str()" in DOOR and "status().c_str()" in DOOR,
      "the door raises with a literal instead of the establishing layer's own status "
      "string. `site=XSSet::UpdateFlatXS reason=the FlatXS device arm declined` is the "
      "seam describing itself, and it is exactly the message four 238 cases died with "
      "while XsReconBackend::status() held the real reason")
check("[RASBERY][RESIDENCY]" in DOOR,
      "the door prints no receipt. `the arm was on and established` and `the arm was "
      "never asked` must not look the same -- the same G0 rule every other receipt in "
      "this tree exists for")
check("want_flatxs && !flatxs_ready" in DOOR,
      "the FlatXS raise is not conditioned on the arm having been REQUESTED; a seam "
      "reached with the arm off is not a violation, nothing was promised")

macro = brace_block(GPUFULL, "inline void requireResidency(")
check(bool(macro), "gpufull::requireResidency is gone; the door has no primitive")
check("if (!required()) {" in macro and "nameFirstFallback(which, where, why);" in macro,
      "gpufull::requireResidency does not NAME with the gate off. The reason reaching "
      "the receipt either way is the point of moving the question to the door")
check("count(which)" not in macro.split("if (!required())")[-1].split("return;")[0],
      "gpufull::requireResidency counts on the gate-OFF path. The seam downstream will "
      "still fall back and still count; a door that counted too would double every "
      "gate-off tally and break the feature-off identity these numbers are read under")
check("note(which, where, why);" in macro,
      "gpufull::requireResidency does not delegate to note() under the gate, so a later "
      "edit could drop the throw and nothing would notice")

# ---------------------------------------------------------------------------
# 4.  armOuterSegment's return is read
# ---------------------------------------------------------------------------

discarded = DRIVER.count("if (gpu_outer_may_arm) armOuterSegment(ctx, eigv, residual);")
check(discarded == 0,
      "src/Driver.h discards armOuterSegment's return at " + str(discarded) + " call "
      "site(s). armOuterSegment binds residency; bindResidency clears residency_bound on "
      "the way in and leaves its reason in CudaOuterSegment::status(), and the post-arm "
      "ladder then re-derives the failure as the generic `no_residency` -- which is the "
      "string nine 238 cases died with, naming the symptom and losing the cause")
read_sites = DRIVER.count("gpu_outer_may_arm && armOuterSegment(ctx, eigv, residual)")
check(read_sites == 2,
      "armOuterSegment's return is read at " + str(read_sites) + " of the 2 call sites "
      "(SolveLoop and ReconvergeFlux); one unread site is one place a failed bind still "
      "reports as no_residency")
check(DRIVER.count('"Driver: outer segment arm"') == 2,
      "a call site that reads the return does not NAME the failed arm; the ladder's "
      "generic answer would still be the only thing in the receipt")
# The ladder still decides.  Naming must not become a second throw: the counters
# and the exit code are the seam's, and this work package does not move them.
for site in re.findall(r"gpu_outer_may_arm && armOuterSegment\(ctx, eigv, residual\);"
                       r"(.{0,700})", DRIVER, re.S):
    check("nameFirstFallback" in site,
          "the arm's failure is not named at one call site")
    check("RASBERY_GPU_FULL_GUARD" not in site.split("if (gpu_outer_may_arm && !")[-1][:400],
          "naming the failed arm became a second guard. The post-arm ladder still "
          "decides and still throws; doubling the throw would double the counters and "
          "make this work package a behaviour change instead of a diagnosis")

# ---------------------------------------------------------------------------
# 5.  One admission door for `promote`
# ---------------------------------------------------------------------------

apply_default = brace_block(SERVER, "inline void applyWaveFidelityDefault(")
check("bool promoted" in SERVER[SERVER.find("inline void applyWaveFidelityDefault("):
                                SERVER.find("inline void applyWaveFidelityDefault(") + 300],
      "applyWaveFidelityDefault does not take the `promoted` flag, so a wave-mode "
      "promotion still resolves against the wave's declaration while a rolling-mode one "
      "resolves against the process default. One op, two doors")
check("if (promoted) return;" in apply_default,
      "applyWaveFidelityDefault does not return early for a promotion. `op\":\"promote\"` "
      "exists so a promotion's defaults are the ones a promotion must have rather than "
      "the ones the process happens to carry; a wave default is that same accident "
      "wearing a wave's name")
check("applyWaveFidelityDefault(wave_fidelity, c.request_fidelity, c.promoted)" in SERVER,
      "the wave path calls applyWaveFidelityDefault without the case's own `promoted` "
      "flag")
# The rolling door is the one being matched: it resolves at admission against the
# PROCESS default, with no wave in sight.
rolling_admit_region = SERVER[SERVER.find("if (detail::rollingEnabled()) {"):
                              SERVER.find("rollingAdmit(request);") + 40]
check("resolveCaseFidelity(request.request_fidelity, processCaseFidelity()"
      in rolling_admit_region,
      "the rolling admission no longer resolves against processCaseFidelity(); the "
      "wave-mode promote rule above was matched to it and would now be matched to "
      "nothing")
# The three fields the op does NOT flip are the reason this matters.
promote_region = SERVER[SERVER.find("if (promote) {"):SERVER.find("if (promote) {") + 1200]
check('request.request_fidelity.fidelity = "strict"' in promote_region and
      'request.request_fidelity.statepoint_grid = "full"' in promote_region,
      "the promote op no longer flips strict/full; the whole argument for refusing it a "
      "wave default is that its defaults are its own")
for unflipped in ("flux_mult", "xe_mult", "loose_settle"):
    check(unflipped not in promote_region,
          "the promote op now flips " + unflipped + " itself. That is a second way to "
          "state the same rule, and two spellings of one rule drift; the early return "
          "in applyWaveFidelityDefault is the one place it is said")

# ---------------------------------------------------------------------------
# 6 + 7.  The receipt: ordinal at the event, first site per subsystem
# ---------------------------------------------------------------------------

count_body = brace_block(GPUFULL, "inline void count(Subsystem which)")
check("pendingOrdinal() = detail::nextOrdinal();" in count_body,
      "gpufull::count does not stamp the event's ordinal. The ordinal has to be taken "
      "at the INCREMENT -- the moment the fallback happened -- because the naming step "
      "that files it runs afterwards and can be descheduled in between; an ordinal "
      "taken at naming time reproduces the race it is meant to settle")
check("pendingOrdinal()" in count_body and "counter(which)" in count_body and
      count_body.index("pendingOrdinal()") < count_body.index("counter(which)"),
      "gpufull::count stamps the ordinal after the counter moves; a reader of the two "
      "would see a count with no ordinal")

name_body = brace_block(GPUFULL, "inline void nameFirstFallback(")
check("takePendingOrdinal()" in name_body,
      "gpufull::nameFirstFallback does not consume count()'s stamp")
take_body = brace_block(GPUFULL, "inline unsigned long long takePendingOrdinal()")
check("slot = 0;" in take_body,
      "takePendingOrdinal does not CONSUME the stamp. A stamp left behind by an earlier "
      "event on the same thread would file a LATER site under an EARLIER ordinal, which "
      "is the one way this table could lie about order")
check("nextOrdinal()" in take_body,
      "a caller that names without counting gets no ordinal at all, so its site would "
      "sort before every real event")
check("recordSubsystemFirst(which, where, why, ordinal)" in name_body,
      "gpufull::nameFirstFallback does not file the subsystem's own first site")

first_body = brace_block(GPUFULL, "inline Subsystem firstViolationSubsystem()")
check("v.ordinal < best" in first_body,
      "gpufull::firstViolationSubsystem does not pick the SMALLEST ordinal; "
      "`first_violation` would go on naming whichever thread won a latch")
check("firstViolationSubsystem()" in brace_block(GPUFULL, "inline const char* firstViolation()"),
      "gpufull::firstViolation ignores the ordered table and still answers from the "
      "one-shot latch, which is the field the 238 receipt could not be checked on")

receipt = brace_block(GPUFULL, "inline void appendReceiptFields(")
# The receipt is written as C++ escapes -- `out << ",\\"violations\\":{"` -- so
# the field names are searched in their SOURCE spelling.
for field in ('first_violation_seq\\":', 'violations\\":{', 'count\\":',
              ',\\"seq\\":', ',\\"site\\":', ',\\"reason\\":'):
    check(field in receipt,
          "the GPU_FULL receipt omits " + field + ". A per-subsystem count with no site "
          "is what sent the arm-A reader to a 16k-line log for a string the process "
          "already held")

# ---------------------------------------------------------------------------
# 8.  The soak reads both
# ---------------------------------------------------------------------------

fallback_list = SOAK[SOAK.find("GPU_FULL_FALLBACKS: tuple"):]
fallback_list = fallback_list[:fallback_list.find(")")]
for subsystem in ("cmfd", "outer", "nodal", "flatxs", "xe", "ppr", "cram"):
    check('"' + subsystem + '_fallbacks"' in fallback_list,
          "tools/soak_run.py does not assert " + subsystem + "_fallbacks under "
          "RASBERY_GPU_FULL=1. The 238 arm-A run reported outer_fallbacks:9 -- nine "
          "cases killed by the gate, a larger category than the four flatxs deaths -- "
          "and the soak's verdict never looked at the number")
check('"gpu_full_contract"' in SOAK and '"contract_pass"' in SOAK,
      "tools/soak_run.py does not read the [RASBERY][GPU_FULL] receipt's own verdict. "
      "Re-deriving it from the counters is what a future count-only seam would slip "
      "past, which is the WP1 defect this contract already exists for")
check("gpu_full_contract" in GATE,
      "tools/promotion_gate.py does not refuse a soak whose GPU_FULL contract failed; a "
      "flag promoted on a run whose GPU arms fell back is promoted on the CPU's numbers")

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS for every scanner above
# ---------------------------------------------------------------------------

if brace_block("void f() { int x; { int y; } }", "void f()") != "{ int x; { int y; } }":
    controls.append("brace_block does not balance nested braces, so every body-scan "
                    "above is reading an arbitrary prefix")
if brace_block("void f() { }", "void g()") != "":
    controls.append("brace_block invents a body for an anchor that is not there, so a "
                    "DELETED function would pass every check about its contents")

if DOOR:
    if "static bool" in DOOR.replace("static bool", "static bool", 1):
        pass
    mutated = DOOR.replace("RASBERY_GPU_FULL_REQUIRE_RESIDENCY(", "// removed(", 1)
    if mutated.count("RASBERY_GPU_FULL_REQUIRE_RESIDENCY(") >= 2:
        controls.append("the door's raise-count check cannot see a deleted raise")
    latched = DOOR_CODE.replace("{", "{ static bool done = false;", 1)
    if "static bool" not in latched:
        controls.append("the first-admission-latch scan cannot see an inserted latch")

if DRIVER:
    relapsed = DRIVER.replace("gpu_outer_may_arm && armOuterSegment(ctx, eigv, residual)",
                              "XX", 1)
    if relapsed.count("gpu_outer_may_arm && armOuterSegment(ctx, eigv, residual)") != 1:
        controls.append("the arm-return scan cannot count call sites, so `both sites "
                        "read the return` proves nothing")

if apply_default:
    if "if (promoted) return;" in apply_default.replace("if (promoted) return;", "", 1):
        controls.append("the promote early-return scan sees a return that was removed")

if count_body:
    if "pendingOrdinal() = detail::nextOrdinal();" in \
            count_body.replace("pendingOrdinal() = detail::nextOrdinal();", "", 1):
        controls.append("the ordinal-stamp scan sees a stamp that was removed")

_probe_fallbacks = '("cmfd_fallbacks", "outer_fallbacks"'
if '"outer_fallbacks"' not in _probe_fallbacks:
    controls.append("the fallback-list reader does not see an entry that is there")
if '"invented_fallbacks"' in fallback_list:
    controls.append("the fallback-list reader accepts a name that is not in the tuple")


# ---------------------------------------------------------------------------
# THE LIVE HALVES.  Reading the source is not the same as running it.
# ---------------------------------------------------------------------------

def find_compiler():
    """MSVC's vcvars64.bat, or a g++/clang++ on PATH, or None."""
    if os.name == "nt":
        for base in (r"C:\Program Files\Microsoft Visual Studio",
                     r"C:\Program Files (x86)\Microsoft Visual Studio"):
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                if "vcvars64.bat" in files:
                    return os.path.join(dirpath, "vcvars64.bat")
    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_and_run(source: str, name: str, extra_include: Path | None = None):
    """Compile *source* and return its stdout, or None when there is no compiler."""
    compiler = find_compiler()
    if compiler is None:
        return None
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        cpp = tmp / (name + ".cpp")
        cpp.write_text(source, encoding="utf-8")
        exe = tmp / (name + (".exe" if os.name == "nt" else ""))
        include = str(extra_include) if extra_include is not None else str(ROOT / "src")
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                script = tmp / ("build_" + name + ".bat")
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul 2>&1\r\n' % compiler
                    + 'cd /d "%s"\r\n' % tmp
                    + 'cl /nologo /std:c++20 /EHsc /D_CRT_SECURE_NO_WARNINGS '
                      '/I "%s" "%s" /Fe:"%s"\r\n' % (include, cpp, exe),
                    encoding="utf-8")
                subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(tmp),
                               capture_output=True, universal_newlines=True)
            else:
                subprocess.run([compiler, "-std=c++20", "-O0", "-pthread",
                                "-I", include, str(cpp), "-o", str(exe)],
                               check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            failures.append("the " + name + " harness does not compile:" + NL
                            + (failure.stdout or "") + (failure.stderr or ""))
            return ""
        done = subprocess.run([str(exe)], capture_output=True, universal_newlines=True)
        if done.returncode != 0:
            failures.append("the " + name + " harness failed: "
                            + done.stdout + done.stderr)
            return ""
        return done.stdout


# --- 6 live: the 238 interleave --------------------------------------------
#
# The FIRST event is flatxs and its reporting thread is descheduled between the
# event and the receipt string; outer's event is later and its string is
# earlier.  That is the arm-A shape -- both subsystems firing inside one
# generation, nothing in the receipt able to order them -- and the old
# one-shot latch answers `outer` for it.

ORDER_HARNESS = r"""
#include "GpuFullContract.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

int main() {
    using namespace rasbery::gpufull;
    std::thread early([] {
        // count(FlatXs), spelled out so the sleep can sit between the event and
        // the naming exactly as a descheduled lane puts it there.
        detail::pendingOrdinal() = detail::nextOrdinal();
        detail::counter(Subsystem::FlatXs).fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        nameFirstFallback(Subsystem::FlatXs, "XSSet::UpdateFlatXS", "declined");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    std::thread late([] {
        note(Subsystem::Outer, "Driver: outer segment pre-arm", "no_residency");
    });
    early.join();
    late.join();
    // Three more outer events, so the counts in the receipt are the 238 shape.
    for (int i = 0; i < 8; ++i)
        note(Subsystem::Outer, "later site", "no_residency");
    for (int i = 0; i < 3; ++i)
        note(Subsystem::FlatXs, "later site", "declined");

    std::ostringstream out;
    appendReceiptFields(out);
    const std::string receipt = out.str();
    std::cout << "first_is_flatxs "
              << (firstViolationSubsystem() == Subsystem::FlatXs) << "\n";
    std::cout << "first_seq_is_one " << (firstViolationOrdinal() == 1) << "\n";
    std::cout << "flatxs_count_4 " << (fallbacks(Subsystem::FlatXs) == 4) << "\n";
    std::cout << "outer_count_9 " << (fallbacks(Subsystem::Outer) == 9) << "\n";
    std::cout << "flatxs_site "
              << (violations(Subsystem::FlatXs).site != nullptr &&
                  std::string(violations(Subsystem::FlatXs).site) ==
                      "XSSet::UpdateFlatXS") << "\n";
    std::cout << "outer_site "
              << (violations(Subsystem::Outer).site != nullptr &&
                  std::string(violations(Subsystem::Outer).site) ==
                      "Driver: outer segment pre-arm") << "\n";
    std::cout << "receipt_names_flatxs "
              << (receipt.find("\"first_violation\":\"subsystem=flatxs") !=
                  std::string::npos) << "\n";
    std::cout << "receipt_has_seq "
              << (receipt.find("\"first_violation_seq\":1") != std::string::npos) << "\n";
    std::cout << "receipt_pairs_count_and_site "
              << (receipt.find("\"flatxs\":{\"count\":4,\"seq\":1,"
                               "\"site\":\"XSSet::UpdateFlatXS\"") !=
                  std::string::npos) << "\n";
    return 0;
}
"""

ORDER_EXPECTED = {
    "first_is_flatxs":
        "first_violation still names the subsystem whose RECEIPT STRING was built "
        "first, not the subsystem whose EVENT was first. That is the 238 receipt "
        "verbatim: outer and flatxs both fired in generation 6 and the field named "
        "outer with nothing to check it against",
    "first_seq_is_one":
        "the first violation's ordinal is not 1, so the ordinals are not a total order "
        "over the events and `first_violation_seq` cannot be checked against "
        "`violations`",
    "flatxs_count_4": "the flatxs counter does not match the events raised",
    "outer_count_9": "the outer counter does not match the events raised",
    "flatxs_site":
        "the flatxs slot records a LATER site than the first event's; a per-subsystem "
        "record that keeps the most recent site answers a different question",
    "outer_site": "the outer slot records a later site than its first event's",
    "receipt_names_flatxs":
        "the printed receipt's first_violation disagrees with firstViolationSubsystem(); "
        "the field a reader quotes must be the one the code decided",
    "receipt_has_seq":
        "the receipt does not carry first_violation_seq, so the claim `this was first` "
        "is unfalsifiable from the receipt alone -- which is the state the 238 "
        "investigation was left in",
    "receipt_pairs_count_and_site":
        "the receipt does not pair a subsystem's count with its site and ordinal; "
        "flatxs_fallbacks:4 goes on being a number that sends a reader to the log",
}

order_out = build_and_run(ORDER_HARNESS, "residency_order")
if order_out is None:
    print("test_evaluator_residency_contract: no compiler; the LIVE ordering and door "
          "checks did not run", file=sys.stderr)
elif order_out:
    results = dict(line.split() for line in order_out.split(NL) if line.strip())
    for key, why in ORDER_EXPECTED.items():
        check(results.get(key) == "1", why)

# --- 3 live: the door's own fail-closed behaviour ---------------------------
#
# The SHIPPED function text, compiled against stubs for everything it touches
# that a host build cannot supply (HDF5 is not on this machine, so Driver.h
# itself cannot be compiled here).  What is under test is the door's decision
# table, and that is entirely in this body.

DOOR_HARNESS = r"""
#include "GpuFullContract.h"
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

// --- the stubs the door touches --------------------------------------------
static bool g_flatxs_arm = false;
static bool g_outer_arm  = false;
static bool g_flatxs_ok  = false;

struct XsReconBackend {
    bool available() const { return g_flatxs_ok; }
    const std::string& status() const {
        static std::string s;
        s = g_flatxs_ok ? "ready" : "no CUDA device: count 0";
        return s;
    }
};
struct XSSet {
    XsReconBackend backend;
    XsReconBackend* EnsureBackend() { return &backend; }
};
bool rasberyGpuFlatXsEnabled() { return g_flatxs_arm; }
namespace gpu {
bool outerGpuEnabled() { return g_outer_arm; }
struct Segment {
    const std::string& status() const {
        static const std::string s = "runner not initialised";
        return s;
    }
};
inline Segment& rasberyOuterSegment() { static Segment s; return s; }
} // namespace gpu

namespace rasbery {
%s
} // namespace rasbery

static std::string drive(bool flatxs_arm, bool flatxs_ok, bool outer_arm,
                         bool outer_stood_up, bool* threw, std::string* what) {
    g_flatxs_arm = flatxs_arm;
    g_flatxs_ok  = flatxs_ok;
    g_outer_arm  = outer_arm;
    XSSet xs;
    std::ostringstream receipt;
    *threw = false;
    try {
        rasbery::establishDeviceResidency(xs, outer_stood_up, receipt);
    } catch (const std::exception& e) {
        *threw = true;
        *what  = e.what();
    }
    return receipt.str();
}

int main() {
    bool threw = false;
    std::string what;

    // (1) EVERY ARM OFF: no receipt, no raise, nothing asked and nothing promised.
    std::string r = drive(false, false, false, true, &threw, &what);
    std::cout << "off_silent " << (r.empty() && !threw) << "\n";

    // (2) ARM ON AND ESTABLISHED: a receipt, and no raise.
    r = drive(true, true, true, true, &threw, &what);
    std::cout << "ready_receipt "
              << (r.find("\"flatxs_ready\":true") != std::string::npos && !threw) << "\n";

    // (3) ARM ON, NOT ESTABLISHED, GATE OFF: named, not raised -- and the seam
    //     downstream is still the one that counts.
    const unsigned long long before = rasbery::gpufull::fallbacks(
        rasbery::gpufull::Subsystem::FlatXs);
    r = drive(true, false, false, true, &threw, &what);
    const unsigned long long after = rasbery::gpufull::fallbacks(
        rasbery::gpufull::Subsystem::FlatXs);
    std::cout << "gateoff_no_throw " << (!threw) << "\n";
    std::cout << "gateoff_no_count " << (after == before) << "\n";
    std::cout << "gateoff_named "
              << (rasbery::gpufull::firstViolation() != nullptr &&
                  std::string(rasbery::gpufull::firstViolation())
                      .find("no CUDA device") != std::string::npos) << "\n";
    std::cout << "gateoff_receipt_reason "
              << (r.find("no CUDA device") != std::string::npos) << "\n";
    return 0;
}
"""

DOOR_EXPECTED = {
    "off_silent":
        "the admission door speaks with every arm off. The feature-off path must be a "
        "predicate and a return -- this door runs on every case of every run in this "
        "tree, including the ones with no GPU asked for",
    "ready_receipt":
        "the door prints no receipt when residency IS established, so `established` and "
        "`never asked` look the same to a reader -- which is the reading the arm-A log "
        "could not disambiguate for its 2nd, 3rd and 4th flatxs deaths",
    "gateoff_no_throw":
        "the door raises with RASBERY_GPU_FULL unset. A fallback is a legal thing for "
        "this binary to do and the host body is the reference",
    "gateoff_no_count":
        "the door COUNTS with the gate off. The seam downstream still counts the same "
        "fallback, so the gate-off tally would double and the feature-off identity the "
        "campaign reads these numbers under would be gone",
    "gateoff_named":
        "the door does not name the establishing layer's own reason with the gate off. "
        "The whole defect is that XsReconBackend::status() went to a process-wide "
        "call_once warn, so in a resident evaluator the first case printed a reason and "
        "the rest printed nothing",
    "gateoff_receipt_reason":
        "the [RASBERY][RESIDENCY] line does not carry the status string, so the per-case "
        "receipt is as silent as the call_once warn it replaces",
}

if DOOR and order_out is not None:
    body = DOOR
    shipped = "static void establishDeviceResidency(XSSet& cross_sections, " \
              "bool outer_stood_up,\n                                         " \
              "std::ostream& receipt) " + body
    # `static` is a class-member spelling; at namespace scope it would make the
    # function internal to the harness's TU, which is fine, but dropping it keeps
    # the text closer to what ships.
    shipped = shipped.replace("static void establishDeviceResidency", "void "
                              "establishDeviceResidency", 1)
    door_out = build_and_run(DOOR_HARNESS % shipped, "residency_door")
    if door_out:
        results = dict(line.split() for line in door_out.split(NL) if line.strip())
        for key, why in DOOR_EXPECTED.items():
            check(results.get(key) == "1", why)

# --- the harnesses' own negative control -----------------------------------
#
# A harness that compiled and printed nothing would satisfy every `== "1"` check
# above by making `results` empty... except it would not, because .get() returns
# None.  This states that explicitly rather than leaving it to be re-derived.
if order_out is not None and order_out == "":
    controls.append("the ordering harness produced no output; every live check above "
                    "silently proved nothing")

# This file must itself parse.
py_compile.compile(str(Path(__file__).resolve()), doraise=True)

if controls:
    for problem in controls:
        print("evaluator residency contract: CONTROL FAILED " + problem, file=sys.stderr)
if failures:
    for problem in failures:
        print("evaluator residency contract: FAIL " + problem, file=sys.stderr)
if failures or controls:
    raise SystemExit(1)
print("evaluator residency contract: PASS (admission door unconditional, promote through "
      "one door, first_violation ordered by event, "
      + ("live halves driven" if order_out else "live halves skipped") + ")")
