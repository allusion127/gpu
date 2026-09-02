#!/usr/bin/env python3
"""Static contract for the Phase 2 statepoint telemetry (plan Rev.4 Sec 8).

The receipt this guards is emitted only by a GPU run, so the properties that
make it trustworthy have to be checked in the source: that it is env-gated, that
nothing it added formats a string inside a solver loop, that the print sites sit
on the statepoint boundary, that every outer lands in exactly one cause bucket,
and that the schema carries every field Sec 8 asks for.
"""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
BICG_H = (ROOT / "src" / "BICGCMFD.h").read_text(encoding="utf-8-sig")
BICG = (ROOT / "src" / "BICGCMFD.cpp").read_text(encoding="utf-8-sig")
SOLVER_H = (ROOT / "src" / "BICGSolver.h").read_text(encoding="utf-8-sig")
SOLVER = (ROOT / "src" / "BICGSolver.cpp").read_text(encoding="utf-8-sig")
CUDA_H = (ROOT / "src" / "CudaBICGBackend.h").read_text(encoding="utf-8-sig")
CUDA = (ROOT / "src" / "CudaBICGBackend.cu").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"statepoint telemetry contract: FAIL: {message}")


def region(text: str, start: str, end: str, what: str) -> str:
    """The source between two anchors, so a check can be scoped to one function."""
    i = text.find(start)
    if i < 0:
        fail(f"{what}: anchor not found: {start!r}")
    j = text.find(end, i)
    if j < 0:
        fail(f"{what}: closing anchor not found: {end!r}")
    return text[i:j]


# ---------------------------------------------------------------------------
# 1. Env gate.  Off by default, and the gate is a cached static bool, not a
#    getenv() the hot path could reach.
# ---------------------------------------------------------------------------
if "RASBERY_STATEPOINT_TELEMETRY" not in DRIVER:
    fail("no RASBERY_STATEPOINT_TELEMETRY gate in Driver.h")
if DRIVER.count('std::getenv("RASBERY_STATEPOINT_TELEMETRY")') != 1:
    fail("the telemetry env var is read in more than one place; it must be one cached gate")

gate = region(DRIVER, "inline bool enabled() {", "} // namespace sptelem", "sptelem::enabled")
if "static const bool on" not in gate:
    fail("sptelem::enabled() does not cache the env lookup in a static")
if "const bool sp_telem = sptelem::enabled();" not in DRIVER:
    fail("Drive() does not hoist the gate into a local bool")

# ---------------------------------------------------------------------------
# 2. Counter discipline: plain members, no atomics, no shared state.
# ---------------------------------------------------------------------------
counters = region(DRIVER, "struct Counters {", "} // namespace sptelem", "sptelem::Counters")
if "atomic" in counters:
    fail("sptelem::Counters uses an atomic; Sec 8.1 requires plain per-Driver counters")
if "sptelem::Counters telemetry{};" not in DRIVER:
    fail("SolverContext does not carry the per-Driver Counters")

# ---------------------------------------------------------------------------
# 3. No formatting anywhere in the instrumented loop bodies.  Everything the
#    telemetry does inside SolveLoop/ReconvergeFlux must be a plain increment or
#    compound assignment on ctx.telemetry / sp_cause.
# ---------------------------------------------------------------------------
solve_loop = region(DRIVER, "    static void SolveLoop(", "    /// Where this run's restart_", "SolveLoop")
if "SPTELEM" in solve_loop:
    fail("SolveLoop formats a telemetry receipt; Sec 8.1 allows it only at the statepoint end")

TOUCH = re.compile(r"^\s*(if \([A-Za-z_0-9]+\) )?"
                   r"(\+\+ctx\.telemetry\.[A-Za-z_0-9\[\]:.]+;"
                   r"|ctx\.telemetry\.[A-Za-z_0-9\[\]:.]+\s*\+=[^;]+;"
                   r"|sp_cause\s*=\s*sptelem::CAUSE_[A-Z]+;)\s*$")
for line in solve_loop.splitlines():
    if "ctx.telemetry" not in line and "sp_cause" not in line:
        continue
    if line.lstrip().startswith("//"):
        continue
    stripped = line.rstrip()
    if stripped.endswith(("+=", "+")) or stripped.endswith("outers_by_cause[sptelem::CAUSE_FALLBACK] +="):
        continue  # a wrapped compound assignment; its tail is checked below
    if "sptelem::Cause sp_cause = sptelem::CAUSE_INITIAL;" in stripped:
        continue
    if not TOUCH.match(stripped):
        fail(f"SolveLoop telemetry line is not a plain counter update: {stripped.strip()!r}")

for banned in ("std::format", "std::cout", "std::cerr", "std::string", "new ", "push_back"):
    for line in solve_loop.splitlines():
        if "telemetry" in line and banned in line:
            fail(f"telemetry line in SolveLoop uses {banned!r}: {line.strip()!r}")

# ---------------------------------------------------------------------------
# 4. Attribution: exactly one bucket per outer, charged right where the outer is
#    counted, and every cause the comment block documents is actually assigned.
# ---------------------------------------------------------------------------
if solve_loop.count("++ctx.telemetry.outers_by_cause[sp_cause];") != 1:
    fail("the per-outer cause charge must appear exactly once in SolveLoop")
charge = solve_loop.index("++ctx.telemetry.outers_by_cause[sp_cause];")
bump = solve_loop.index("++total_outer;")
between = solve_loop[bump:charge]
if bump > charge or "for (" in between or "if (" in between:
    fail("the cause charge is not adjacent to ++total_outer; it could miss or double-count outers")

for cause in ("CAUSE_XE", "CAUSE_TH", "CAUSE_SEARCH", "CAUSE_SETTLE"):
    if not re.search(rf"sp_cause\s*=\s*sptelem::{cause};", solve_loop):
        fail(f"no segment boundary opens {cause}; the attribution rules are not implemented")
if "outers_by_cause[sptelem::CAUSE_FALLBACK] +=" not in solve_loop:
    fail("ReconvergeFlux outers are not charged to the FALLBACK bucket")
if "const int fallback_outer0 = total_outer;" not in solve_loop:
    fail("the FALLBACK bucket is not measured as a total_outer delta across ReconvergeFlux")

# The perturbation counters must sit on the sites that actually move state.
for anchor, counter, span in (
    ("UpdateEquilibriumXenon(schedule.thermalPower(), xe_relax)", "xe_updates", 700),
    ("th_dop = ctx.cross_sections.UpdateTH(power_fraction);", "th_updates", 200),
    # The search commit is followed by the RASBERY_SEARCH_TRACE block before the
    # trial point is applied, so its window has to clear that.
    ("schedule.CommitSearchPoint(eigv, next_x, ctx.search_memory);", "search_trials", 2000),
    ("++stall_events;", "flux_limit_retries", 100),
):
    i = solve_loop.find(anchor)
    if i < 0:
        fail(f"instrumentation anchor vanished: {anchor!r}")
    if f"ctx.telemetry.{counter}" not in solve_loop[i:i + span]:
        fail(f"{counter} is not incremented at its perturbation site ({anchor!r})")

# Cascade resolution.  A Xe cascade starts on entry and again at every committed
# perturbation, so xe_cascades is charged at exactly those two sites; the budget
# re-arm that changes the iteration path stays behind its own cached env gate.
if solve_loop.count("++ctx.telemetry.xe_cascades;") != 2:
    fail("xe_cascades must be charged exactly twice: SolveLoop entry and the "
         "cascade-restart site after a committed perturbation")
if DRIVER.count('std::getenv("RASBERY_XE_CASCADE_BUDGET")') != 1:
    fail("the per-cascade Xe budget must be read through exactly one cached gate")
if "static const bool xe_cascade_budget" not in solve_loop:
    fail("the per-cascade Xe budget gate is not cached in a static bool")
if not re.search(r"if \(xe_cascade_budget\) \{\s*\n\s*xe_count\s*=\s*0;", solve_loop):
    fail("the per-cascade Xe counter re-arm is not gated on xe_cascade_budget; "
         "feature-off would not be byte-golden")
# One definition of "out of budget", shared by the interim probe, the Xe step and
# the starvation charge, so no gated ceiling can block one of them but not another.
if solve_loop.count("const bool xe_starved =") != 1:
    fail("the Xe budget test is not defined once as xe_starved")
if solve_loop.count("!xe_starved") != 2:
    fail("xe_starved must gate exactly the interim probe and the Xe step")
exhausted = solve_loop.find("++ctx.telemetry.xe_budget_exhausted;")
if exhausted < 0 or "xe_starved &&" not in solve_loop[max(0, exhausted - 400):exhausted]:
    fail("xe_budget_exhausted is not charged at the budget-starvation test")

# ---------------------------------------------------------------------------
# 5. Print sites: one per statepoint at the statepoint boundary, one per run.
# ---------------------------------------------------------------------------
if DRIVER.count('"[RASBERY][SPTELEM] {{') != 1:
    fail("the per-statepoint receipt must be emitted from exactly one site")
if DRIVER.count('"[RASBERY][SPTELEM][SUMMARY] {{') != 1:
    fail("the run summary must be emitted from exactly one site")

drive = region(DRIVER, "    int Drive() {", "\n};\n", "Drive")
io_close = drive.find("total_io_seconds += step_io_seconds;")
step_print = drive.find('"[RASBERY][SPTELEM] {{')
eoc = drive.find("// Natural EOC (MASTER %EXE_DEP tgobj boron)")
if not (0 <= io_close < step_print < eoc):
    fail("the per-statepoint receipt is not emitted at the statepoint boundary "
         "(after the step's IO accounting, before the natural-EOC block)")
if "ctx.telemetry.begin(" not in drive or "ctx.telemetry.end(" not in drive:
    fail("Drive() does not arm/close the per-statepoint baselines")
arm = drive.find("ctx.telemetry.begin(")
if not (0 <= arm < step_print):
    fail("the statepoint baselines are armed after the receipt is printed")

summary = drive.find('"[RASBERY][SPTELEM][SUMMARY] {{')
if not (0 <= step_print < summary):
    fail("the run summary is not emitted after the statepoint loop")
if 'TOTAL DRIVER TIME' not in drive[:summary]:
    fail("the run summary must be emitted after the run wall is known")

# Both receipts stay inside the gate.
for site in (step_print, summary):
    guard = drive.rfind("if (sp_telem) {", 0, site)
    if guard < 0:
        fail("a telemetry receipt is printed outside the env gate")

# ---------------------------------------------------------------------------
# 6. Schema.  Sec 8.1 names these fields; schema_version rides along so a
#    consumer can tell versions apart.
# ---------------------------------------------------------------------------
REQUIRED = (
    "schema_version", "job_id", "slot", "statepoint", "efpd", "outers",
    "xe_updates", "xe_outers", "search_trials", "search_outers",
    "th_updates", "th_outers", "settle_outers", "flux_limit_retries",
    "cmfd_sweeps", "bicg_iters", "graph_launches_delta", "h2d_bytes_delta",
    "d2h_bytes_delta", "phase_wall",
    # Cascade resolution of the Xe budget.  Additive, so schema_version stays 1:
    # a consumer that predates them still parses every field it knows.
    "xe_cascades", "xe_steps_per_cascade", "xe_budget_exhausted",
)
step_block = drive[step_print:drive.index(");", step_print)]
missing = [f for f in REQUIRED if f'"{f}"' not in step_block.replace('\\"', '"')]
if missing:
    fail(f"per-statepoint schema missing {missing}")
if '"schema_version\\":1' not in step_block:
    fail("per-statepoint receipt does not pin schema_version")

summary_block = drive[summary:drive.index(");", summary)]
for field in ("schema_version", "job_id", "statepoints", "outers", "xe_updates",
              "search_trials", "th_updates", "cmfd_sweeps", "bicg_iters",
              "t_fixed", "library_seconds", "phase_wall",
              "xe_cascades", "xe_steps_per_cascade", "xe_budget_exhausted"):
    if f'"{field}"' not in summary_block.replace('\\"', '"'):
        fail(f"run summary missing {field!r} (Sec 8.1 wants totals + a T_fixed estimate)")

for phase in ("updpsi", "setls", "drive", "updjnet", "nodal", "cusping", "upddhat"):
    if f'"{phase}"' not in step_block.replace('\\"', '"'):
        fail(f"phase_wall missing bucket {phase!r}")

# ---------------------------------------------------------------------------
# 6b. WP9-A: the host floor, re-decomposed.  THREE non-overlapping wall objects
#     plus one that is explicitly NOT additive, and the boundary transfer bill.
#
#     The load-bearing property is the SEPARATION, not the presence.
#     tools/scheduler_trace_replay.py derives a statepoint's boundary work as
#     `wall - sum(phase_wall)` and is calibrated on that; folding a new bucket
#     into phase_wall would silently move a number a model depends on.  So
#     phase_wall must still hold EXACTLY the seven outer-body phases, and
#     everything WP9-A named must live in loop_wall / floor_wall / nested_wall.
# ---------------------------------------------------------------------------
LOOP_BUCKETS = ("th_update", "xe_step", "search_propose", "search_apply")
FLOOR_BUCKETS = ("ppr_reset", "ppr_drive", "ppr_recon", "depl_predictor",
                 "depl_corrector", "result_add", "result_write")


def json_object(block: str, key: str) -> str:
    """The text of the `"key":{...}` object inside a receipt format string."""
    plain = block.replace('\\"', '"')
    i = plain.find(f'"{key}":{{{{')
    if i < 0:
        fail(f"receipt has no {key!r} object")
    j = plain.find("}}", i)
    if j < 0:
        fail(f"receipt's {key!r} object is not closed")
    return plain[i:j]


for name, block in (("per-statepoint", step_block), ("summary", summary_block)):
    phase_obj = json_object(block, "phase_wall")
    got = tuple(re.findall(r'"([a-z_0-9]+)":', phase_obj))[1:]  # drop the key itself
    if got != ("updpsi", "setls", "drive", "updjnet", "nodal", "cusping", "upddhat"):
        fail(f"{name} phase_wall is no longer exactly the seven outer-body phases "
             f"({list(got)}); tools/scheduler_trace_replay.py is calibrated on that set")
    loop_obj = json_object(block, "loop_wall")
    for bucket in LOOP_BUCKETS:
        if f'"{bucket}":' not in loop_obj:
            fail(f"{name} loop_wall missing bucket {bucket!r}")
    floor_obj = json_object(block, "floor_wall")
    for bucket in FLOOR_BUCKETS:
        if f'"{bucket}":' not in floor_obj:
            fail(f"{name} floor_wall missing bucket {bucket!r}")
    nested_obj = json_object(block, "nested_wall")
    for bucket in ("flatxs", "flatxs_calls"):
        if f'"{bucket}":' not in nested_obj:
            fail(f"{name} nested_wall missing bucket {bucket!r}")
    transfer_obj = json_object(block, "floor_transfer")
    for bucket in ("h2d_bytes", "h2d_calls", "d2h_bytes", "d2h_calls"):
        if f'"{bucket}":' not in transfer_obj:
            fail(f"{name} floor_transfer missing {bucket!r}")
    # Negative control: a bucket may appear in exactly ONE of the three additive
    # objects.  A name in two of them is a double count, and the residual the
    # ledger prints would silently absorb it.
    seen = {}
    for obj_name, obj in (("phase_wall", phase_obj), ("loop_wall", loop_obj),
                          ("floor_wall", floor_obj)):
        for bucket in re.findall(r'"([a-z_0-9]+)":', obj)[1:]:
            if bucket in seen:
                fail(f"{name}: bucket {bucket!r} is in both {seen[bucket]!r} and "
                     f"{obj_name!r}; the three wall objects must not overlap")
            seen[bucket] = obj_name

# The scopes that fill the new buckets sit on the calls they name.  Each is
# checked at its site, so a bucket cannot quietly become a bucket of nothing.
for anchor, phase in (
    ("th_dop = ctx.cross_sections.UpdateTH(power_fraction);", "PH_TH_UPDATE"),
    ("schedule.CommitSearchPoint(eigv, next_x, ctx.search_memory);", "PH_SEARCH_PROPOSE"),
    ("input_output.WriteStepToResult(geometry, cross_sections, step_index);",
     "PH_RESULT_WRITE"),
):
    i = DRIVER.find(anchor)
    if i < 0:
        fail(f"WP9-A instrumentation anchor vanished: {anchor!r}")
    # THE WINDOW IS A PROXIMITY HEURISTIC, NOT A SEMANTIC BOUND: what is being
    # checked is that a scope of the right phase is OPEN over the call, and the
    # character count is only how this static reader approximates "open".  It is
    # widened when the block between the two legitimately grows -- WP24 added
    # one field to the light-result Fidelity fill inside PH_RESULT_WRITE's scope
    # and put the anchor 1,543 characters out.  Widening it is not a weakening
    # as long as no OTHER scope opens in between, which the nesting rules
    # checked above already forbid.
    if f"sptelem::{phase}" not in DRIVER[max(0, i - 1800):i]:
        fail(f"{phase} has no scope opened before {anchor!r}")
# xe_step is charged over the WHOLE Xe step region -- the Anderson attempt, the
# production Picard step and the trust-region bookkeeping -- so it is pinned by
# position rather than by proximity: the scope opens inside the Xe step block
# and before the production call, which is the only placement with no hole in it
# and no edit to the call site tools/test_xe_anderson.py pins.
xe_block = DRIVER.find("(flux_converged || xe_interim || stall_sample)) {")
xe_scope = DRIVER.find("outer_timing::Scope xe_step_scope(sptelem::PH_XE_STEP);")
xe_call = DRIVER.find(
    "ctx.cross_sections.UpdateEquilibriumXenon(schedule.thermalPower(), xe_relax);")
if not (0 <= xe_block < xe_scope < xe_call):
    fail("PH_XE_STEP is not opened inside the Xe step block, before the production "
         "UpdateEquilibriumXenon call")
if DRIVER.count("outer_timing::Scope apply_scope(sptelem::PH_SEARCH_APPLY);") != 3:
    fail("search_apply must be charged at all three SetBoron/SetRod sites "
         "(first trial, committed trial, best-point fallback)")

# The XS phase mirror: armed from the one gate, never gated on its own, and
# differenced across the statepoint like every other bucket.
if "xsphase::armLocalWall(sp_telem);" not in DRIVER:
    fail("the XS phase mirror is not armed from Drive()'s telemetry gate")
XSTIMING = (ROOT / "src" / "XSTiming.h").read_text(encoding="utf-8-sig")
if "LB_FLATXS" not in XSTIMING or "inline LocalWall& localWall()" not in XSTIMING:
    fail("XSTiming.h has no per-statepoint local bucket")
if "static thread_local LocalWall wall;" not in XSTIMING:
    fail("the XS phase mirror is not thread-local; it would mix batch decks")
XSSET = (ROOT / "src" / "XSSet.cpp").read_text(encoding="utf-8-sig")
if "xsphase::LB_FLATXS" not in XSSET:
    fail("UpdateFlatXS does not feed the nested_wall bucket")

# The boundary transfer baseline is armed AFTER the last SolveLoop and BEFORE
# PPR.  Armed anywhere else, `floor_transfer` would be measuring something else.
arm_floor = drive.find("ctx.telemetry.armFloor(")
last_solve = drive.rfind("SolveLoop(ctx, eigv, schedule, total_outer", 0, arm_floor)
ppr = drive.find("resetAndDriveGpu(", arm_floor)
if not (0 <= last_solve < arm_floor < ppr):
    fail("the floor transfer baseline is not armed between the last SolveLoop and PPR")
if drive.count("ctx.telemetry.armFloor(") != 1:
    fail("the floor transfer baseline is armed more than once")

# ---------------------------------------------------------------------------
# 6d. WP9-D: the critical search's convergence history.
#
#     INSTRUMENT ONLY.  The plan's outer census charges 15.3 % of a case to the
#     boron search over 137 trials, and says the cost is CANDIDATE DEPENDENT.
#     Before any trial-reduction lever is built, the runner has to be able to
#     say what those trials were -- a bootstrap probe, a carried slope, a secant
#     and a bisection inside a collapsing bracket are four different stories
#     about one trial count, with four different fixes.  What this section
#     protects is that the story is complete and that telling it does not change
#     the search.
# ---------------------------------------------------------------------------
SCHEDULER = (ROOT / "src" / "Scheduler.h").read_text(encoding="utf-8-sig")
SEARCH_FIELDS = ("trials", "proposals", "refused", "secant", "carry_secant", "probe",
                 "bisect", "extrap", "iterations")
for name, block in (("per-statepoint", step_block), ("summary", summary_block)):
    search_obj = json_object(block, "search")
    for field in SEARCH_FIELDS:
        if f'"{field}":' not in search_obj:
            fail(f"{name} search object missing {field!r}")
# The per-statepoint receipt additionally carries the convergence history that
# does not SUM over a run, so a run summary deliberately does not repeat it.
for field in ("exit", "tol", "dk", "x_first", "x_final", "dx_last"):
    if f'"{field}":' not in json_object(step_block, "search"):
        fail(f"per-statepoint search object missing {field!r}")

# The classification happens exactly once, where `method` is decided.  Two call
# sites would be two classifications that can disagree.
if SCHEDULER.count("TallyProposal(") != 2:  # the definition and its one call
    fail("Schedule::TallyProposal is not defined once and called once; the search "
         "classification must have a single site")
if "const bool proposed = AdvanceSecantSearch(" not in SCHEDULER:
    fail("ProposeNextSearchPoint no longer folds both search arms into one call, so "
         "the proposal classification would have to be repeated per arm")

# TRAJECTORY NEUTRALITY, mechanically.  Every WP9-D field may be WRITTEN and
# may be READ BY A RECEIPT.  None of them may be read by anything that decides
# what the search does next -- no condition, no arithmetic feeding next_x.
WP9D_FIELDS = ("search_n_proposals", "search_n_refused", "search_n_secant",
               "search_n_carry", "search_n_probe", "search_n_bisect",
               "search_n_extrap",
               "search_first_x", "search_last_dx")
CONDITION = re.compile(r"\b(if|while|for|return|assert)\b[^;]*")


def split_guard(stripped: str) -> tuple[str, str]:
    """Split a leading `[else ]if (...)` / `else` off a statement.

    Paren-counted rather than regex-matched, because a condition can contain
    parentheses of its own (`method.rfind("bisection", 0) == 0`) and a regex
    that stopped at the first `)` would report the tail of the CONDITION as the
    statement -- which is how a scan like this quietly starts passing.
    """
    text = stripped
    if text.startswith("else "):
        text = text[5:].lstrip()
    elif text == "else":
        return "", ""
    if not text.startswith("if ("):
        return "", text
    depth, i = 0, text.index("(")
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], text[j + 1:].strip()
    return text, ""
for line in SCHEDULER.splitlines():
    body = line.split("//")[0]
    if not any(f in body for f in WP9D_FIELDS):
        continue
    stripped = body.strip()
    if stripped.startswith(("long long ", "double ")):
        continue  # the declarations
    # A leading `if (cond)` is allowed -- the classification reads `method`, the
    # string the proposal already built -- but the CONDITION may not mention a
    # counter, because that would be the search reading its own instrument.
    cond, rest = split_guard(stripped)
    if any(f in cond for f in WP9D_FIELDS):
        fail(f"a WP9-D search counter is read by a CONDITION: {stripped!r}")
        continue
    rest = rest.strip()
    if rest.startswith("++") and rest.endswith(";"):
        continue
    if re.match(r"^[a-z_0-9]+\s*=\s*[^=].*;$", rest):
        continue
    if CONDITION.search(rest) and any(f in rest for f in WP9D_FIELDS):
        fail(f"a WP9-D search counter reaches control flow: {stripped!r}")
        continue
    if any(f in rest for f in WP9D_FIELDS):
        fail(f"a WP9-D search counter is not a plain write in Scheduler.h: {stripped!r}")
for field in WP9D_FIELDS:
    for m in re.finditer(re.escape(field), DRIVER):
        line = DRIVER[DRIVER.rfind("\n", 0, m.start()) + 1:
                      DRIVER.find("\n", m.start())]
        if line.lstrip().startswith("//") or "///" in line:
            continue
        if "ResetSearchTelemetry" in line:
            continue
        # Every other reader must be inside the receipt's argument list.
        if "ctx.telemetry.search_" not in line and "schedule.search_" not in line:
            fail(f"Driver.h reads {field} outside the receipt: {line.strip()!r}")

# ---------------------------------------------------------------------------
# 6c. Negative control for a defect no reviewer can see and no compiler here can
#     catch: a receipt whose std::format placeholder count and argument count
#     have drifted apart.  Both receipts are re-counted from the source.
# ---------------------------------------------------------------------------
def format_call(text: str, anchor: str) -> tuple[str, list[str]]:
    i = text.index(anchor)
    start = text.rindex("std::format(", 0, i) + len("std::format(")
    j, depth = start, 1
    while depth:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
        j += 1
    body = text[start:j - 1]
    literal = re.compile(r'"(?:[^"\\]|\\.)*"')
    pos, parts = 0, []
    while True:
        m = literal.match(body, pos)
        if not m:
            skip = re.match(r"\s*(//[^\n]*\n\s*)*", body[pos:])
            if skip and skip.end():
                pos += skip.end()
                m = literal.match(body, pos)
            if not m:
                break
        parts.append(m.group(0)[1:-1])
        pos = m.end()
    rest = body[pos:].lstrip()
    if not rest.startswith(","):
        fail(f"cannot split the format string from its arguments at {anchor!r}")
    args, depth, cur = [], 0, ""
    for ch in rest[1:]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
        else:
            cur += ch
    args.append(cur)
    args = [a for a in (re.sub(r"//[^\n]*", "", a).strip() for a in args) if a]
    return "".join(parts), args


for anchor in ('"[RASBERY][SPTELEM] {{', '"[RASBERY][SPTELEM][SUMMARY] {{'):
    fmt, args = format_call(DRIVER, anchor)
    holes = re.findall(r"\{[^{}]*\}", fmt.replace("{{", "\x01").replace("}}", "\x02"))
    if len(holes) != len(args):
        fail(f"{anchor}: {len(holes)} format placeholders but {len(args)} arguments")

# ---------------------------------------------------------------------------
# 7. The phase wall must be per-thread, and must still feed the pre-existing
#    process-wide RASBERY_OUTER_TIMING receipt unchanged.
# ---------------------------------------------------------------------------
if "static thread_local double wall[PH_COUNT]" not in DRIVER:
    fail("the per-statepoint phase wall is not thread-local")
scope = region(DRIVER, "class Scope {", "inline void report(std::ostream& out)", "outer_timing::Scope")
if "if (!_global && !_local) return;" not in scope:
    fail("outer_timing::Scope reads the clock with both receipts disabled")
if "sptelem::phaseWall()[_phase] += dt;" not in scope:
    fail("outer_timing::Scope does not feed the per-statepoint phase wall")
report = region(DRIVER, "inline void report(std::ostream& out)", "} // namespace outer_timing",
                "outer_timing::report")
if '"[RASBERY][OUTER][PHASE] {\\"outers\\":"' not in report:
    fail("the RASBERY_OUTER_TIMING receipt changed shape")

# ---------------------------------------------------------------------------
# 8. Solver-side counters the receipt reads.
# ---------------------------------------------------------------------------
if "long long bicgIterations() const" not in BICG_H:
    fail("BICGCMFD exposes no BiCGSTAB iteration count")
if "BackendCounters backendCounters() const" not in BICG_H:
    fail("BICGCMFD exposes no backend counters for the per-statepoint deltas")
if "int batchSlot() const" not in BICG_H or "int batchSlot() const" not in SOLVER_H:
    fail("the arena slot is not reachable for the receipt's `slot` key")
if "_bicg_iters              = 0;" not in BICG:
    fail("resetIteration() does not zero the BiCGSTAB iteration count")
if BICG.count("_bicg_iters +=") + BICG.count("++_bicg_iters;") < 4:
    fail("the BiCGSTAB iteration count is not maintained on every solve path "
         "(host unconditional + host loop + CUDA solveInner + device sweeps)")
if "ctx.telemetry.cmfd_sweeps += ctx.cmfd_solver.innerIterations();" not in DRIVER:
    fail("CMFD sweeps are not folded into the statepoint totals")

# ---------------------------------------------------------------------------
# 9. D2H bytes: Sec 8 asks for both transfer directions, so the backend has to
#    tally the bytes it pulls back, not just the calls.
# ---------------------------------------------------------------------------
if "bulk_d2h_bytes_during_iteration" not in CUDA_H:
    fail("BackendCounters has no D2H byte tally")
if CUDA.count("telemetry.bulk_d2h_bytes_during_iteration") != CUDA.count(
        "telemetry.bulk_d2h_calls_during_iteration"):
    fail("D2H byte tally is not maintained at every site that counts a D2H call")
if "bulk_d2h_bytes_during_iteration" not in SOLVER:
    fail("the single-instance BACKEND_COUNTERS dump omits the D2H byte tally")

py_compile.compile(str(Path(__file__).resolve()), doraise=True)
print("statepoint telemetry contract: PASS")
