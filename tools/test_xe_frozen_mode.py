#!/usr/bin/env python3
"""Static contract for the in-core equilibrium-Xe modes (RASBERY_XE_MODE).

The modes are a runtime switch on a GPU solver, so the properties that make them
safe cannot be observed from a Python test run -- they have to be checked in the
source.  Seven things are guarded here:

  1. BYTE-GOLDENNESS.  With RASBERY_XE_MODE unset (or `equilibrium`) the solver
     path must be what it was: the env var is read exactly once, cached in a
     function-local static, and the only things it feeds in the solve path are
     one extra `&& !xeFrozen()` term on `has_eq_xe` and one cached `xeOnce()`
     read whose every consumer short-circuits to the old expression when it is
     false.  No getenv, no mode lookup, anywhere inside the outer loop.
  2. ONE SHORT-CIRCUIT POINT (frozen).  `has_eq_xe` is the single gate the whole
     in-core Xe machinery hangs off, so clearing it must be enough: the cascade
     counters, the pending/interim probe, the starvation charge, the
     UpdateEquilibriumXenon call and the cascade re-arm are ALL gated on it, and
     there is no second, ungated call site.
  3. TELEMETRY AT ZERO.  In frozen mode xe_updates and xe_cascades are
     identically zero, so the derived steps-per-cascade must not divide by them.
  4. RECEIPTS.  Both the [RASBERY][PHYSICS_MODE] line and the two
     [RASBERY][SPTELEM] receipts publish `xe_mode`, fed from xeModeName(), with
     format placeholders and arguments still in step -- and xeModeName() can
     produce every mode the parser accepts, no more and no fewer.
  5. DEPLETION AND SAMARIUM UNTOUCHED.  The modes stop the in-SolveLoop refresh
     only.  XSSet's Bateman/CRAM advance and its own equilibrium overwrite at
     the depletion rates stay gated on the deck's `xe_transient` alone, and
     nothing in XSSet knows the modes exist.  Sm-149 has no in-core equilibrium
     to freeze -- if one is ever added, this test fails on purpose.
  6. ONCE = ONE STEP PER CASCADE.  `xe_once_fired` is set by the Xe step, gates
     the Xe step, and is cleared at exactly the three cascade starts the
     cascade counter itself opens a segment at (SolveLoop entry, T/H commit,
     search commit).  The cap carries no test on the Xe residual: one Picard
     step per segment, accepted as it stands.
  7. ONCE HAS NO DAMPER AND NO INTERIM PATH.  A single undamped step cannot
     limit-cycle, so both the oscillation damper and the RASBERY_XE_INTERIM_*
     probe -- including the RASBERY_XE_INTERIM_DAMP primer that would start the
     step at relax=0.5 -- must be unreachable in the mode.
"""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
MAIN = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8-sig")
XSSET = (ROOT / "src" / "XSSet.cpp").read_text(encoding="utf-8-sig")
XSSET_H = (ROOT / "src" / "XSSet.h").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"xe mode contract: FAIL: {message}")


def region(text: str, start: str, end: str, what: str) -> str:
    """The source between two anchors, so a check can be scoped to one function."""
    i = text.find(start)
    if i < 0:
        fail(f"{what}: anchor not found: {start!r}")
    j = text.find(end, i)
    if j < 0:
        fail(f"{what}: closing anchor not found: {end!r}")
    return text[i:j]


def code_lines(text: str) -> str:
    """`text` with whole-line // comments dropped, so prose cannot satisfy a check."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))


SOLVE_LOOP = region(DRIVER, "    static void SolveLoop(",
                    "    /// Where this run's restart_", "SolveLoop")
SOLVE_CODE = code_lines(SOLVE_LOOP)
DRIVE = region(DRIVER, "    int Drive() {", "\n};\n", "Drive")

# ---------------------------------------------------------------------------
# 1. The env gate: one read, cached, default = the old behaviour.
# ---------------------------------------------------------------------------
if DRIVER.count('std::getenv("RASBERY_XE_MODE")') != 1:
    fail("RASBERY_XE_MODE must be read through exactly one cached gate")
for other in (ROOT / "src").rglob("*"):
    if other.suffix not in (".h", ".cpp", ".cu") or other.name == "Driver.h":
        continue
    if 'getenv("RASBERY_XE_MODE")' in other.read_text(encoding="utf-8-sig", errors="ignore"):
        fail(f"{other.name} reads RASBERY_XE_MODE; the mode has one home (Driver.h)")

gate = region(DRIVER, "inline XeMode xeMode() {", "inline bool xeFrozen()", "xeMode")
if "static const XeMode mode = [] {" not in gate:
    fail("xeMode() does not cache the env lookup in a function-local static")
null_at = gate.find("if (value == nullptr)")
if null_at < 0 or "return XeMode::EQUILIBRIUM;" not in gate[null_at:null_at + 120]:
    fail("an unset RASBERY_XE_MODE does not return EQUILIBRIUM before anything else; "
         "the default path would not be byte-golden")
MODES = ("equilibrium", "frozen", "once")
for token in tuple(f's == "{m}"' for m in MODES) + ("std::tolower",):
    if token not in gate:
        fail(f"xeMode() is missing {token!r} (every mode, case-insensitively)")
if "[RASBERY][WARN][xe]" not in gate:
    fail("xeMode() silently swallows an unrecognised value; a typo must be reported")
for m in MODES:
    if m not in gate[gate.index("[RASBERY][WARN][xe]"):]:
        fail(f"the unrecognised-value warning does not name the {m!r} mode; a typo must be "
             "told what the alternatives are")
if "inline bool xeFrozen() { return xeMode() == XeMode::FROZEN; }" not in DRIVER:
    fail("xeFrozen() is not the single predicate over xeMode()")
if "inline bool xeOnce() { return xeMode() == XeMode::ONCE; }" not in DRIVER:
    fail("xeOnce() is not the single predicate over xeMode()")
# once must NOT be folded into xeFrozen(): frozen clears has_eq_xe, and once
# needs every arm of the machinery that gate feeds.
if "XeMode::ONCE" in region(DRIVER, "inline bool xeFrozen()", "inline bool xeOnce()",
                            "xeFrozen"):
    fail("xeFrozen() answers true for once mode; the frozen short-circuit would then "
         "bypass the very machinery once mode drives")

name_fn = region(DRIVER, "inline const char* xeModeName() {", "\nclass Driver", "xeModeName")
if "switch (xeMode())" not in name_fn:
    fail("xeModeName() does not dispatch on xeMode(); it must publish every mode")
published = set(re.findall(r'return "([a-z]+)";', name_fn))
if published != set(MODES):
    fail(f"xeModeName() publishes {sorted(published)}; the parser accepts {sorted(MODES)}")
enum_decl = "enum class XeMode { EQUILIBRIUM, FROZEN, ONCE };"
if enum_decl not in DRIVER:
    fail(f"the mode enum is not {enum_decl!r}")

# Nothing may look the mode up inside the iteration: SolveLoop sees it only
# through xeFrozen()/xeOnce(), whose static is resolved once per process.
for banned in ('getenv("RASBERY_XE_MODE")', "xeMode()", "xeModeName()"):
    if banned in SOLVE_CODE:
        fail(f"SolveLoop reaches {banned!r}; the mode must be one cached read outside the loop")
if SOLVE_CODE.count("xeFrozen()") != 1:
    fail("xeFrozen() must be consulted exactly once in SolveLoop (the has_eq_xe term)")
if SOLVE_CODE.count("xeOnce()") != 1:
    fail("xeOnce() must be consulted exactly once in SolveLoop (the xe_once_mode term)")

# ---------------------------------------------------------------------------
# 2. The short-circuit point: has_eq_xe, and nothing else.
# ---------------------------------------------------------------------------
HAS_EQ_XE = "const bool   has_eq_xe      = !schedule.xenon_transient && !xeFrozen();"
if HAS_EQ_XE not in SOLVE_LOOP:
    fail("has_eq_xe is not `!schedule.xenon_transient && !xeFrozen()`; the frozen branch "
         "must be one term on the gate the Xe machinery already hangs off")

decl_at = SOLVE_LOOP.index(HAS_EQ_XE)
for after in ("ctx.cmfd_solver.resetIteration();", "++ctx.telemetry.solve_loops;",
              "ctx.cross_sections.SetBoron(schedule.search_current_x);"):
    j = SOLVE_LOOP.find(after)
    if j < 0:
        fail(f"ordering anchor vanished from SolveLoop: {after!r}")
    if j < decl_at:
        fail(f"the frozen branch is decided after {after!r}; it must be checked before "
             "any behaviour change")

# Every arm of the machinery is gated on it, so clearing it bypasses all of them.
GATED = (
    ("if (has_eq_xe) ++ctx.telemetry.xe_cascades;", "the SolveLoop-entry cascade charge"),
    ("const bool xe_pending = has_eq_xe && !xe_starved &&", "the pending/interim probe"),
    ("if (has_eq_xe && !xe_cap_charged && xe_starved &&", "the starvation charge"),
    ("if (has_eq_xe && !xe_starved &&", "the equilibrium-Xe step"),
    ("if (xe_restart && has_eq_xe) {", "the cascade re-arm after a committed perturbation"),
)
for token, what in GATED:
    if token not in SOLVE_LOOP:
        fail(f"{what} is no longer gated on has_eq_xe: {token!r}")

# The Xe step is the only reachable equilibrium call, and it sits inside that gate.
if DRIVER.count("UpdateEquilibriumXenon(") != 1:
    fail("Driver.h has more than one UpdateEquilibriumXenon call site; frozen mode can "
         "only guarantee the one it gates")
call_at = SOLVE_LOOP.index("UpdateEquilibriumXenon(")
guard_at = SOLVE_LOOP.rfind("if (has_eq_xe && !xe_starved &&", 0, call_at)
if guard_at < 0 or "for (" in SOLVE_LOOP[guard_at:call_at]:
    fail("the UpdateEquilibriumXenon call is not directly inside the has_eq_xe gate; "
         "frozen mode could still reach it")
# Nothing between the gate and the call may re-open the path.
if SOLVE_LOOP[guard_at:call_at].count("}") > SOLVE_LOOP[guard_at:call_at].count("{"):
    fail("the has_eq_xe gate closes before the UpdateEquilibriumXenon call")

# ---------------------------------------------------------------------------
# 3. Telemetry at xe_cascades == 0 / xe_updates == 0.
# ---------------------------------------------------------------------------
if DRIVE.count("c.xe_cascades > 0") != 2:
    fail("the steps-per-cascade ratio is not guarded against xe_cascades == 0 at both "
         "receipt sites; frozen mode publishes zeros there")
for m in re.finditer(r"static_cast<double>\(c\.xe_updates\)\s*/\s*"
                     r"static_cast<double>\(c\.xe_cascades\)", DRIVE):
    head = DRIVE[max(0, m.start() - 200):m.start()]
    if "c.xe_cascades > 0" not in head:
        fail("an xe_updates/xe_cascades division is not behind the zero guard")
# No other consumer divides by a Xe counter.
for m in re.finditer(r"/\s*static_cast<double>\(c\.xe_(updates|cascades)\)", DRIVE):
    if "c.xe_cascades > 0" not in DRIVE[max(0, m.start() - 200):m.start()]:
        fail("an unguarded division by a Xe counter would trap/NaN in frozen mode")

# ---------------------------------------------------------------------------
# 4. Receipts.
# ---------------------------------------------------------------------------
receipt_at = MAIN.find("[RASBERY][PHYSICS_MODE]")
if receipt_at < 0:
    fail("src/main.cpp: the PHYSICS_MODE receipt vanished")
receipt = MAIN[receipt_at:MAIN.index("std::endl;", receipt_at)]
if '\\"xe_mode\\":\\"' not in receipt:
    fail("src/main.cpp: the PHYSICS_MODE receipt does not publish xe_mode")
if "rasbery::xeModeName()" not in receipt:
    fail("src/main.cpp: xe_mode is not fed from xeModeName()")

for anchor, what in (('"[RASBERY][SPTELEM] {{', "per-statepoint receipt"),
                     ('"[RASBERY][SPTELEM][SUMMARY] {{', "run summary")):
    i = DRIVE.find(anchor)
    if i < 0:
        fail(f"{what}: print site not found")
    block = DRIVE[i:DRIVE.index(");", i)]
    if '\\"xe_mode\\":\\"{}\\"' not in block:
        fail(f"{what} does not publish xe_mode")
    if "xeModeName()" not in block:
        fail(f"{what}: xe_mode is not fed from xeModeName()")
    # The format string is the leading run of adjacent string literals; the rest
    # is the argument list.  Keeping the two in step is what a hand-edited
    # 40-argument std::format most easily breaks.
    literals: list[str] = []
    end = 0
    for lit in re.finditer(r'"((?:[^"\\]|\\.)*)"', block):
        if literals and block[end:lit.start()].strip():
            break
        literals.append(lit.group(1))
        end = lit.end()
    # Retire the escaped braces first, then every remaining {...} is a hole.
    fmt = "".join(literals).replace("{{", "\x00").replace("}}", "\x01")
    holes = len(re.findall(r"\{[^{}]*\}", fmt))
    args, depth = 1, 0
    for ch in block[end:].lstrip().lstrip(","):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            args += 1
    if holes != args:
        fail(f"{what}: {holes} format placeholders but {args} arguments")

# ---------------------------------------------------------------------------
# 5. The startup guard: warns, does not block, fires once, FROZEN only, and
#    reads the incoming inventory at the one moment that phrase means anything.
#
#    The first version measured `max|Xe| == 0.0` from inside SolveLoop and
#    latched on whichever call came first.  Both halves failed on a real deck:
#    the library's fresh-fuel point hands over a trace, so a bit-exact zero is
#    never seen, and on a depletion-first deck the first SolveLoop runs after
#    PredictorStep has already written the Xe rows at the depletion rates.  Both
#    regressions are pinned below.
# ---------------------------------------------------------------------------
warn = region(DRIVER, "    static void WarnFrozenXeIfEmpty(",
              "    // Single-loop eigenvalue solve", "WarnFrozenXeIfEmpty")
if "if (!xeFrozen() || schedule.xenon_transient || ctx.xe_frozen_checked)" not in warn:
    fail("the startup guard is not short-circuited on mode/deck/one-shot latch")
if "xeOnce()" in warn:
    fail("the startup guard consults xeOnce(); once mode computes its own Xe and has "
         "nothing to warn about, so the guard is scoped to frozen by xeFrozen() alone")
if "ctx.xe_frozen_checked = true;" not in warn:
    fail("the startup guard does not latch; it would warn at every statepoint")
if "bool   xe_frozen_checked = false;" not in DRIVER:
    fail("SolverContext does not carry the one-shot latch")
if "Chiffon::Isotope::iXe135" not in warn:
    fail("the startup guard does not inspect the incoming Xe-135 inventory")
if "ctx.geometry.IsFuel(l)" not in warn:
    fail("the startup guard does not restrict itself to fuel nodes")
if "peak_xe == 0.0" not in warn:
    fail("the startup guard does not test for an identically-zero Xe inventory")
# REGRESSION (1): a bit-exact zero cannot be the only way in, or a single trace
# density anywhere in the core suppresses the warning for the whole core.
if "inherited_inventory" not in warn:
    fail("the startup guard has no arm for a run that never inherited an inventory; "
         "`peak_xe == 0.0` alone is defeated by the library's fresh-fuel trace")
if "!empty_inventory && inherited_inventory" not in warn:
    fail("the startup guard does not fire on `no inherited inventory OR bit-zero`; "
         "one of the two ways to hold no in-core Xe would go unreported")
if "{:.3e}" not in warn:
    fail("the startup guard does not print the measured peak; neither arm would be "
         "diagnosable from the warning alone")
if "[RASBERY][WARN][xe]" not in warn:
    fail("the startup guard does not emit a [RASBERY][WARN][xe] line")
for banned in ("throw ", "std::exit", "return 2;", "runtime_error"):
    if banned in warn:
        fail(f"the startup guard uses {banned!r}; a zero-Xe frozen run must warn, not block")

# REGRESSION (2): the hook point.  Not SolveLoop -- Drive(), straight after
# InitXS, before any schedule entry (a depletion step's PredictorStep above all)
# can overwrite the Xe rows the guard is supposed to be reading.
if "WarnFrozenXeIfEmpty(" in SOLVE_CODE:
    fail("SolveLoop calls the startup guard again; on a depletion-first deck its first "
         "call lands after PredictorStep's own equilibrium overwrite and the one-shot "
         "latch is spent on a full inventory")
CALL = "WarnFrozenXeIfEmpty(ctx, initial_schedule, is_restart_run || is_shuffle_run);"
if CALL not in DRIVE:
    fail(f"Drive() does not call the startup guard as {CALL!r}")
if DRIVER.count("WarnFrozenXeIfEmpty(ctx,") != 1:
    fail("the startup guard has more than one call site; the latch makes the extra ones "
         "silent, so the surviving one must be the right one")
init_at = DRIVE.index("cross_sections.InitXS(")
call_at = DRIVE.index(CALL)
if call_at < init_at:
    fail("the startup guard runs before InitXS; there is no incoming inventory to read yet")
for later in ("cross_sections.PredictorStep(", "SolveLoop(ctx, eigv, schedule",
              "cross_sections.SetPowerRate(power_fraction);"):
    j = DRIVE.find(later)
    if j < 0:
        fail(f"ordering anchor vanished from Drive: {later!r}")
    if j < call_at:
        fail(f"the startup guard runs after {later!r}; the Xe rows it reads are no longer "
             "the ones the deck handed over")

# ---------------------------------------------------------------------------
# 6. Depletion and samarium: not this mode's business.
# ---------------------------------------------------------------------------
for name in ("xeFrozen()", "xeMode()", 'getenv("RASBERY_XE_MODE")'):
    if name in XSSET or name in XSSET_H:
        fail(f"XSSet knows about {name!r}; frozen mode must stop the in-SolveLoop refresh "
             "only, never depletion's Bateman/CRAM bookkeeping")
if XSSET.count("if (!xe_transient)") < 3:
    fail("the depletion-side equilibrium overwrites are no longer gated on the deck's "
         "xe_transient alone (Deplete + both CorrectorStep arms)")
for call in ("cross_sections.PredictorStep(sub_dt, thermal_power, schedule.xenon_transient);",
             "cross_sections.CorrectorStep(sub_dt, thermal_power, schedule.xenon_transient);"):
    if call not in DRIVER:
        fail(f"depletion no longer takes the deck's xenon flag straight through: {call!r}")

# Samarium: there is nothing symmetric to freeze.  ApplyXeEquilibrium writes the
# three Xe-chain rows and nothing else, and no Sm index appears in the solver at
# all -- Sm-149/Pm-149 move only through the depletion CRAM solve.  Should an
# in-core Sm equilibrium ever be added, frozen mode has to freeze it too, and
# this check is the reminder.
eq = region(XSSET, "static void ApplyXeEquilibrium(", "\n// Divergence probe",
            "ApplyXeEquilibrium")
written = set(re.findall(r"iden\[(i[A-Za-z0-9]+)\]\s*=", eq))
if written != {"iI135", "iXe135", "iXe135m"}:
    fail(f"ApplyXeEquilibrium writes {sorted(written)}; frozen mode is documented as "
         "holding exactly the I-135/Xe-135/Xe-135m chain")
for sm in ("iSm149", "iPm149"):
    if sm in code_lines(XSSET) or sm in code_lines(DRIVER):
        fail(f"{sm} now appears in the solver: RASBERY has grown an in-core samarium "
             "treatment, and RASBERY_XE_MODE=frozen must be extended to hold it")

# ---------------------------------------------------------------------------
# 7. ONCE mode: one undamped step per cascade, cleared at the three cascade
#    starts, with no damper and no interim path reachable.
# ---------------------------------------------------------------------------
ONCE_GATE = "const bool   xe_once_mode   = xeOnce();"
if ONCE_GATE not in SOLVE_LOOP:
    fail(f"the once gate is not read once into a local: {ONCE_GATE!r}")
once_at = SOLVE_LOOP.index(ONCE_GATE)
# Decided before anything moves, exactly like the frozen term it sits beside.
for after in ("ctx.cmfd_solver.resetIteration();", "++ctx.telemetry.solve_loops;",
              "for (int iout = 0;"):
    j = SOLVE_LOOP.find(after)
    if j < 0:
        fail(f"ordering anchor vanished from SolveLoop: {after!r}")
    if j < once_at:
        fail(f"the once gate is decided after {after!r}; it must be checked before any "
             "behaviour change")

# The cap: one term on the gate the step already hangs off, and no test on the
# Xe residual anywhere in it -- one Picard step per segment, accepted as-is.
STEP_GATE = ("if (has_eq_xe && !xe_starved && (!xe_once_mode || !xe_once_fired) &&\n"
             "                (flux_converged || xe_interim || stall_sample)) {")
if STEP_GATE not in SOLVE_LOOP:
    fail("the Xe step gate does not carry `(!xe_once_mode || !xe_once_fired)`; the cap "
         "must be one term on the existing gate, not a second branch")
if "XE_EQUILIBRIUM_TOLERANCE" in STEP_GATE:
    fail("the once cap tests the Xe residual; the single step is accepted as it stands")

# The flag: declared false at SolveLoop entry (cascade start 1), set by the
# step, cleared at the commit re-arm (cascade starts 2 and 3 -- T/H and search
# are one restart, the tie-break the sptelem comment describes).
if "bool   xe_once_fired  = false;" not in SOLVE_LOOP:
    fail("xe_once_fired is not initialised false at SolveLoop entry, which IS the first "
         "cascade start")
if SOLVE_CODE.count("xe_once_fired = true;") != 1:
    fail("xe_once_fired must be set in exactly one place: the Xe step that spends it")
set_at = SOLVE_LOOP.index("xe_once_fired = true;")
step_at = SOLVE_LOOP.index("UpdateEquilibriumXenon(")
if set_at < step_at:
    fail("xe_once_fired is set before the Xe step it accounts for")
if SOLVE_LOOP.rfind(STEP_GATE, 0, set_at) < 0:
    fail("xe_once_fired is set outside the gated Xe step")
if SOLVE_CODE.count("xe_once_fired = false;") != 1:
    fail("xe_once_fired must be cleared in exactly one place inside the loop: the cascade "
         "re-arm.  SolveLoop entry is the declaration; T/H and search commits share the "
         "re-arm because they are one restart, not two")
rearm = region(SOLVE_LOOP, "if (xe_restart && has_eq_xe) {",
               "\n        }\n", "cascade re-arm")
if "xe_once_fired = false;" not in rearm:
    fail("the cascade re-arm does not clear xe_once_fired; the T/H and search commits "
         "would not start a new once segment")
if "if (xe_cascade_budget)" not in rearm:
    fail("the cascade re-arm lost its budget branch")
budget_at = rearm.index("if (xe_cascade_budget)")
if rearm.index("xe_once_fired = false;") > budget_at:
    fail("xe_once_fired is cleared inside the RASBERY_XE_CASCADE_BUDGET branch; the once "
         "cap must not depend on an unrelated experiment gate")

# No damper, no interim, in the mode.
damper = "if (!xe_once_mode && xe_relax == 1.0 && xe_no_progress >= xe_streak_limit) {"
if damper not in SOLVE_LOOP:
    fail("the oscillation damper is reachable in once mode; a streak assembled from "
         "single steps against different macro-XS states is not a limit cycle")
relax0 = "double    xe_relax        = (xe_interim_damp && !xe_once_mode)"
if relax0 not in SOLVE_LOOP:
    fail("RASBERY_XE_INTERIM_DAMP can still start once mode's single step at relax=0.5; "
         "the mode is undamped by definition")
if "const bool xe_interim = xe_interim_l2 > 0.0 && !xe_once_mode && xe_pending &&" \
        not in SOLVE_LOOP:
    fail("the interim-flux probe is reachable in once mode; the single step has to be "
         "taken on the converged flux the segment publishes")

# Byte-goldenness of the once terms: every one of them short-circuits to the
# pre-once expression when xe_once_mode is false.  `xe_once_fired` is a dead
# store on that path, so it may only ever be READ next to xe_once_mode.
for line in SOLVE_CODE.splitlines():
    if "xe_once_fired" not in line:
        continue
    if re.search(r"xe_once_fired\s*=[^=]", line):
        continue    # a write; a dead store on the default path by construction
    if "xe_once_mode" not in line:
        fail(f"xe_once_fired is read without xe_once_mode guarding it: {line.strip()!r}")

py_compile.compile(str(Path(__file__).resolve()), doraise=True)
print("xe mode contract: PASS")
