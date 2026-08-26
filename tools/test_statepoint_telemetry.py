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
