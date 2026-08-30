#!/usr/bin/env python3
"""Static contract for the safeguarded Anderson Xe arm (RASBERY_XE_ANDERSON).

Plan Rev.4 Sec 10.  The arm is a runtime switch on a GPU solver, so the
properties that make it safe cannot be observed from a Python test run -- they
have to be checked in the source.  Ten things are guarded here:

  1. BYTE-GOLDENNESS OF THE OFF PATH, AND THE MODE-DEPENDENT DEFAULT.  Whenever
     the gate resolves OFF -- the batch default, or an explicit
     RASBERY_XE_ANDERSON=0 in either mode -- the solve path must be what it was:
     the env var is read exactly once, cached in a function-local static,
     hoisted into ONE local at SolveLoop entry, and the single call it feeds
     short-circuits on that local before anything is evaluated.  No getenv
     inside the outer loop, and the production UpdateEquilibriumXenon call still
     sits where it always sat.  As of the 2026-08-27 adoption the DEFAULT is
     mode-dependent (single ON, --batch-mode OFF) with an explicit env
     overriding it in BOTH directions, so this also pins how the execution mode
     reaches the decision and that the receipt publishes both the effective
     state and its provenance.
  2. THE SPLIT IS A SPLIT, NOT A COPY.  ApplyXeEquilibrium keeps every caller
     and every write it had; the arithmetic moved wholesale into a compute-only
     ComputeXeEquilibrium that writes NOTHING, and ApplyXeEquilibrium is the
     wrapper that applies its result.  The fused GPU arm is untouched.
  3. EVALUATE WRITES NOTHING.  XSSet::EvaluateEquilibriumXenon must not assign
     an _iden row, must not reconstruct a node, must not bump the host-state
     generation and must not touch _xs.  That property is the whole reason a
     refused Anderson candidate can fall back to the production Picard step:
     the step finds the solver exactly as it expects it.
  4. COMMIT WRITES EXACTLY THREE ROWS.  I-135, Xe-135, Xe-135m and no other
     isotope; then the per-node reconstruction and the generation bump the
     device arms re-upload on; and a length check, because a short vector would
     publish a half-updated core.
  5. ONE FUEL-NODE LIST.  Evaluate/Commit index by fuel-node ordinal, so the
     xsrecon device arm and they must read the SAME list or the layout forks.
  6. THE FORMULA.  The two-column normal equations, the one-column secant
     fallback, and the candidate x_{k+1} = F_k - sum_j gamma_j dF_j, all
     written out; plus the static_assert that stops a depth bump from silently
     mis-solving.
  7. FOUR SAFEGUARDS, ALL OF THEM.  Conditioning, predicted residual decrease,
     physics (finite and non-negative), and the trust cap against the Picard
     step -- each on the accept path, each with a rejection that returns false
     without committing.
  8. THE DAMPER WINS.  Anderson is armed only on an undamped relax, and damper
     activation discards the history: the damper is a root selector, and an
     extrapolation off the pre-damping map would fight the choice.
  9. HISTORY RESETS.  At SolveLoop entry (the declaration, which IS cascade
     start 1), at the cascade re-arm after a committed T/H or search
     perturbation, and at damper activation.  The state is a SolveLoop local,
     which is what makes a depletion step structurally unable to carry history.
 10. TELEMETRY.  Four additive counters, charged only inside the gated arm,
     with proposed == accepted + rejected by construction, folded into the run
     totals and published by both SPTELEM receipts.  There are now THREE arms
     (host, split device, device transaction), chained by guarded tail calls so
     that exactly one body runs per call; the per-proposal counters are charged
     ONCE IN EACH, which is one charge per proposal and is why the WP7-C
     receipt can claim anderson_proposed/anderson_accepted do not move under
     RASBERY_GPU_XE_TXN.

It also pins the two DELIBERATE DEVIATIONS from Sec 10 -- no full-exact
true-residual acceptance, no coupled snapshot/rollback, and therefore no
runtime axial-branch guard -- as documented deviations rather than omissions.
If either is ever implemented, the note goes and this check goes with it.
"""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
XSSET = (ROOT / "src" / "XSSet.cpp").read_text(encoding="utf-8-sig")
XSSET_H = (ROOT / "src" / "XSSet.h").read_text(encoding="utf-8-sig")
MAIN = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"xe anderson contract: FAIL: {message}")


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


def squash(text: str) -> str:
    """All runs of whitespace collapsed to one space.

    Anchors that span a line break or sit behind alignment padding are checked
    through this, so a clang-format pass or a rename that shifts a column does
    not fail a contract about SEMANTICS.  Never used for checks where the
    whitespace itself is the property being pinned.
    """
    return re.sub(r"\s+", " ", text)


SOLVE_LOOP = region(DRIVER, "    static void SolveLoop(",
                    "    /// Where this run's restart_", "SolveLoop")
SOLVE_CODE = code_lines(SOLVE_LOOP)
DRIVE = region(DRIVER, "    int Drive() {", "\n};\n", "Drive")
AA = region(DRIVER, "    static bool TryAndersonXeStep(",
            "\n    // Frozen-Xe startup guard", "TryAndersonXeStep")
AA_CODE = code_lines(AA)
# Rev.7.1 Task 13: the device sibling (RASBERY_GPU_XE).  Scoped separately so a
# check can say WHICH arm it is talking about; every check above this line is
# about the host arm and stays that way.
GPU_AA = region(DRIVER, "    static bool TryAndersonXeStepGpu(",
                "\n    /// One safeguarded Anderson step", "TryAndersonXeStepGpu")
# WP7-C: the device TRANSACTION sibling (RASBERY_GPU_XE_TXN), which takes the
# whole step -- safeguards, decision and commit -- inside one device call and
# reads the outcome back as a reason code.
# The signature carries RASBERY_NEVER_INLINE (see the header there and
# docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md Sec 7), so the anchor starts
# at the return type rather than at `static`.
TXN_AA = region(DRIVER, "bool TryAndersonXeStepGpuTxn(SolverContext",
                "\n    /// The DEVICE arm of the safeguarded Anderson step",
                "TryAndersonXeStepGpuTxn")

# THE ARMS, ENUMERATED.  Three function bodies can take an Anderson step and
# EXACTLY ONE OF THEM TAKES ANY GIVEN ONE: the dispatch is a chain of tail
# calls, each guarded by a cached flag and each returning the callee's answer
# without falling through --
#
#     SolveLoop -> TryAndersonXeStep            (host)
#                    RASBERY_GPU_XE     -> TryAndersonXeStepGpu       (split device)
#                      RASBERY_GPU_XE_TXN -> TryAndersonXeStepGpuTxn  (device txn)
#
# so a per-proposal counter must be charged ONCE IN EACH ARM.  The totals are
# then independent of which arm ran, which is exactly what the WP7-C receipt
# claims: anderson_proposed/anderson_accepted are "the same event, the same
# place" under TXN=0 and TXN=1.  The regions are disjoint, so summing them is
# a partition and not a double count.
ARMS = (("host (TryAndersonXeStep)", AA),
        ("split device (TryAndersonXeStepGpu)", GPU_AA),
        ("device transaction (TryAndersonXeStepGpuTxn)", TXN_AA))
for _name, _arm in ARMS:
    for _other_name, _other in ARMS:
        if _name is not _other_name and _other in _arm:
            fail(f"the {_other_name} arm's body is nested inside the {_name} arm's region; "
                 "the per-arm charge counts would double-count it")
# ...and the chain is a CHAIN.  Each hand-off is a guarded call whose result is
# returned immediately, so the caller's own body cannot run as well; and each
# callee has exactly one call site.  Together that is what turns "one charge
# per arm" into "one charge per proposal".
for _caller, _what, _call in (
    (AA, "host -> split device",
     "if (rasberyGpuXeEnabled()) return TryAndersonXeStepGpu(ctx, aa, power, max_step, "
     "xe_change);"),
    (GPU_AA, "split device -> device transaction",
     "if (txn && TryAndersonXeStepGpuTxn(ctx, aa, power, max_step, xe_change)) return "
     "true;"),
):
    if _call not in squash(_caller):
        fail(f"the {_what} hand-off is not a guarded tail call: {_call!r} not found; "
             "without the immediate return the caller's body would run too and charge "
             "the same proposal a second time")
for _callee in ("TryAndersonXeStepGpu(ctx,", "TryAndersonXeStepGpuTxn(ctx,"):
    if DRIVER.count(_callee) != 1:
        fail(f"{_callee} has {DRIVER.count(_callee)} call sites; an arm entered from two "
             "places is an arm whose charges no longer partition the proposals")
if squash(GPU_AA).index("TryAndersonXeStepGpuTxn(ctx,") > \
        squash(GPU_AA).index("++ctx.telemetry.xe_aa_proposed;"):
    fail("the split device arm charges a proposal BEFORE handing off to the transaction; "
         "a transaction step would then be charged in both arms")

# ---------------------------------------------------------------------------
# 1. The gate: one read, cached, MODE-DEPENDENT default, env overrides both ways.
#
# ADOPTION 2026-08-27.  Default ON for a single run (1.69x with the MASTER
# agreement IMPROVING, 1.970 -> 1.905 pcm -- adopted as a correctness fix that
# happens to be faster), default OFF under --batch-mode (net-negative there:
# 202 vs 216 cases/h, arrival-width starvation).  RASBERY_XE_ANDERSON=1/0
# overrides in BOTH modes, so batch A/B experiments stay possible and a single
# deck can still reproduce a legacy Picard trajectory.
# ---------------------------------------------------------------------------
if DRIVER.count('std::getenv("RASBERY_XE_ANDERSON")') != 1:
    fail("RASBERY_XE_ANDERSON must be read through exactly one cached gate")
for other in (ROOT / "src").rglob("*"):
    if other.suffix not in (".h", ".cpp", ".cu") or other.name == "Driver.h":
        continue
    body = other.read_text(encoding="utf-8-sig", errors="ignore")
    for var in ('getenv("RASBERY_XE_ANDERSON")', 'getenv("RASBERY_XE_ANDERSON_MAX_STEP")',
                'getenv("RASBERY_XE_ANDERSON_DEBUG")'):
        if var in body:
            fail(f"{other.name} reads {var}; the arm has one home (Driver.h)")

gate = region(DRIVER, "inline const XeAndersonGate& xeAndersonGate() {",
              "\n/// The effective state", "xeAndersonGate")
GATE_FLAT = squash(gate)
if "static const XeAndersonGate resolved = [] {" not in gate:
    fail("xeAndersonGate() does not cache the resolution in a function-local static")

# TRIMMED AND CASE-FOLDED before anything reads it.  A CRLF-terminated env file
# is how this tree has been bitten before, and a trailing '\r' would turn "0"
# into an unrecognised word -- i.e. into the OPPOSITE state in single mode.
if "find_first_not_of(kBlank)" not in GATE_FLAT or "find_last_not_of(kBlank)" not in GATE_FLAT:
    fail("the value is not trimmed on BOTH ends before it is parsed; a trailing CR would "
         "buy the opposite state of the one that was written")
if "\\r" not in GATE_FLAT:
    fail("the trim set does not include the carriage return, which is the whole point")
if "std::tolower" not in GATE_FLAT:
    fail("the value is not case-folded; xeMode() sets the precedent for every RASBERY "
         "mode switch")

# The two accepted vocabularies, and the fact that the env decides ALONE when it
# is present -- an explicit request must beat the mode default in BOTH
# directions, or a batch A/B is impossible and a single deck cannot go legacy.
OFF_WORDS = ('requested == "0"', 'requested == "off"', 'requested == "false"',
             'requested == "no"')
ON_WORDS = ('requested == "1"', 'requested == "on"', 'requested == "true"',
            'requested == "yes"')
for word in OFF_WORDS + ON_WORDS:
    if word not in GATE_FLAT:
        fail(f"the env override does not accept {word!r}; both vocabularies must be complete")
off_at = GATE_FLAT.index(OFF_WORDS[0])
on_at = GATE_FLAT.index(ON_WORDS[0])
for arm, at, state in (("off", off_at, "want = false;"), ("on", on_at, "want = true;")):
    window = GATE_FLAT[at:at + 260]
    if state not in window:
        fail(f"the {arm!r} arm does not set {state!r}")
    if "source = XeAndersonSource::Env;" not in window:
        fail(f"the {arm!r} arm does not record Env provenance; an explicit request would be "
             "reported as a default")

# A value in neither vocabulary is NOT a state: it warns, names itself, and
# falls through to the mode default rather than being guessed at.
if 'is not a state (0|off|false|no or 1|on|true|yes)' not in GATE_FLAT:
    fail("an unrecognised RASBERY_XE_ANDERSON value is accepted silently; the warning must "
         "also name the accepted vocabularies")
if '<< value' not in GATE_FLAT:
    fail("the warning does not name the rejected value; without it a typo is invisible")
typo_at = GATE_FLAT.index("is not a state (0|off")
if "source = XeAndersonSource::Env;" in GATE_FLAT[typo_at:]:
    fail("an unrecognised value still claims Env provenance; it must fall through to the "
         "mode default")

# The mode default is the fallback for EVERY non-request, and the execution mode
# is consulted at exactly one place.
DEFAULT_ARM = ("if (source == XeAndersonSource::Default) { "
               "want = (executionMode() == ExecutionMode::Single); }")
if DEFAULT_ARM not in squash(code_lines(gate)):
    fail("the default is not 'ON for a single run, OFF under --batch-mode', applied to every "
         f"path that did not come from the env: {DEFAULT_ARM!r}")
if gate.count("executionMode()") != 1:
    fail("the execution mode is consulted more than once in the gate; the default has one "
         "decision point")

# EQUILIBRIUM ONLY, and said out loud when it was ASKED for.  frozen has no
# cascade to accelerate and once is deliberately not converging one.
if "xeMode() != XeMode::EQUILIBRIUM" not in gate:
    fail("the gate does not refuse the non-equilibrium modes; Anderson accelerates "
         "the equilibrium cascade and there is nothing to accelerate in frozen/once")
if "[RASBERY][WARN][xe]" not in gate:
    fail("the gate silently swallows a mode conflict; a misconfiguration must be "
         "reported, not quietly honoured as 'off'")
conflict = GATE_FLAT[GATE_FLAT.index("xeMode() != XeMode::EQUILIBRIUM"):]
if "if (source == XeAndersonSource::Env)" not in conflict.split("return")[0]:
    fail("the mode-conflict warning fires for the DEFAULT too; nobody asked for Anderson in "
         "that run, so there is no misconfiguration to report")
if "return XeAndersonGate{false, source};" not in conflict:
    fail("the mode conflict warns but still enables the arm")

# PROVENANCE.  With a mode-dependent default, the state alone does not identify
# a run: `on` plus "who decided it" is the fact a measurement needs.
if "enum class XeAndersonSource { Default, Env };" not in DRIVER:
    fail("there is no provenance type; default-ON and env-ON would be indistinguishable")
if not re.search(r"struct XeAndersonGate\s*\{\s*bool\s+on;\s*XeAndersonSource\s+source;\s*\};",
                 DRIVER):
    fail("XeAndersonGate does not carry BOTH the effective state and its provenance")
eff = region(DRIVER, "inline bool xeAnderson() {", "\n/// \"default\" or \"env\"", "xeAnderson")
if "return xeAndersonGate().on;" not in eff:
    fail("xeAnderson() does not read through the single cached gate; two statics could "
         "disagree about a run's state")
src_name = region(DRIVER, "inline const char* xeAndersonSourceName() {",
                  "\n/// Per-rejection reason trace", "xeAndersonSourceName")
for token in ('"env"', '"default"'):
    if token not in src_name:
        fail(f"xeAndersonSourceName() does not publish {token}")

# ---------------------------------------------------------------------------
# 1b. How the decision point learns it is in batch mode.
#
# It cannot ask the CUDA backend (rasberyBatchWidth() is absent from a stub
# build) and it cannot re-parse argv, so main() DECLARES the mode -- once, from
# the same predicate that selects the batch branch, before the first receipt.
# ---------------------------------------------------------------------------
if "enum class ExecutionMode { Single, Batch };" not in DRIVER:
    fail("Driver.h has no ExecutionMode; the mode-dependent default has nothing to read")
exec_read = region(DRIVER, "inline ExecutionMode executionMode() {", "\n/// True under",
                   "executionMode")
if "ExecutionMode::Single" not in region(DRIVER, "inline std::atomic<ExecutionMode>& modeCell()",
                                         "\n/// Latched by the first READ", "modeCell"):
    fail("the execution-mode latch does not default to Single; a unit test or a tool that "
         "never declares a mode would be treated as a batch run")
if "observed().store(true" not in exec_read:
    fail("reading the execution mode does not latch 'observed'; a late declaration could not "
         "be detected")
declare = region(DRIVER, "inline void declareExecutionMode(ExecutionMode requested) {",
                 "\n// Safeguarded Anderson", "declareExecutionMode")
if "[RASBERY][WARN][exec]" not in declare:
    fail("a declaration that arrives after the mode was already read is silent; the gate it "
         "was supposed to feed has already cached the wrong answer")
if "observed().load" not in declare:
    fail("declareExecutionMode does not check whether the mode was already read")
# ...and it REFUSES the late change instead of recording it.  Storing a value a
# cached gate has already resolved against would leave the cell -- which is what
# the receipt reads -- describing a run that is not the one executing.
warn_at = declare.index("[RASBERY][WARN][exec]")
store_at = declare.index("modeCell().store(")
if store_at < warn_at:
    fail("the late-declare guard runs after the store it is supposed to prevent")
if "return;" not in declare[warn_at:store_at]:
    fail("the late-declare path warns and then stores anyway; it must return first, or the "
         "cell disagrees with the state the cached gate resolved against")
if declare.count("modeCell().store(") != 1:
    fail("declareExecutionMode has more than one store; the refusal path could be bypassed")

MAIN_CODE = code_lines(MAIN)
MAIN_FLAT = squash(MAIN_CODE)
# THREE sites, named individually: the definition, the declaration and the
# branch.  (A bare occurrence count would also be satisfied by three mentions in
# the wrong places, and would break on an unrelated fourth use.)
# WP8 added a SECOND PRODUCER of the same predicate, not a second predicate:
# `--evaluator` is a batch process whose job list arrives on a stream, so it has
# to resolve the same mode-dependent Anderson default as the `--jobs
# --batch-mode M` run whose per-case digests it must reproduce.  The rule this
# block enforces is unchanged -- ONE expression, declared once, and the same
# name selects the branch that runs.
DEFINITION = ("const bool batch_execution = (batch_width > 0 && !rasbery_inputs.empty()) "
              "|| evaluator_mode;")
DECLARATION = ("rasbery::declareExecutionMode(batch_execution ? rasbery::ExecutionMode::Batch "
               ": rasbery::ExecutionMode::Single);")
BRANCH = "if (batch_execution) {"
if DEFINITION not in MAIN_FLAT:
    fail(f"main() does not compute the batch predicate once: {DEFINITION!r}")
if DECLARATION not in MAIN_FLAT:
    fail("main() does not declare the execution mode from the batch predicate: "
         f"{DECLARATION!r}")
if BRANCH not in MAIN_FLAT:
    fail("the batch branch is not selected by the SAME predicate that was declared; the run "
         "could do one thing and report another")
# ...and no SECOND predicate that could disagree with the one that was declared.
if "batch_width > 0 && !rasbery_inputs.empty()" in MAIN_FLAT.replace(DEFINITION, "", 1):
    fail("the batch predicate is spelled out a second time; the declared mode and the branch "
         "that runs must come from ONE expression")
declare_at = MAIN_FLAT.index("rasbery::declareExecutionMode(")
receipt_at = MAIN_FLAT.index("[RASBERY][PHYSICS_MODE]")
branch_at = MAIN_FLAT.index(BRANCH)
if not declare_at < receipt_at < branch_at:
    fail("the execution mode is declared after the [PHYSICS_MODE] receipt; the receipt is the "
         "first read of the mode-dependent default, so it would cache the wrong answer")
# The receipt publishes the EFFECTIVE state and its provenance, and the mode
# that chose it.  Without all three, two runs of the same binary are not
# distinguishable in a log.
for field, expr in (('\\"exec_mode\\":', "rasbery::executionModeName()"),
                    ('\\"xe_anderson\\":', "rasbery::xeAnderson()"),
                    ('\\"xe_anderson_source\\":', "rasbery::xeAndersonSourceName()")):
    if field not in MAIN_CODE:
        fail(f"the [RASBERY][PHYSICS_MODE] receipt omits {field}")
    if expr not in MAIN_CODE:
        fail(f"the {field} receipt field is not the resolved value ({expr})")

debug = region(DRIVER, "inline bool xeAndersonDebug() {", "\nclass Driver", "xeAndersonDebug")
if "static const bool on" not in debug:
    fail("xeAndersonDebug() does not cache its env lookup")

# The trust-region override: one cached read, resolved before the outer loop.
if DRIVER.count('std::getenv("RASBERY_XE_ANDERSON_MAX_STEP")') != 1:
    fail("RASBERY_XE_ANDERSON_MAX_STEP must be read through exactly one cached gate")
if "static const double xe_aa_max_step = [] {" not in SOLVE_LOOP:
    fail("the Anderson trust region is not cached in a function-local static; the loop "
         "would reach getenv on every step")
max_env = region(SOLVE_LOOP, "static const double xe_aa_max_step = [] {", "}();",
                 "xe_aa_max_step")
if "XE_ANDERSON_MAX_STEP" not in max_env:
    fail("an unset RASBERY_XE_ANDERSON_MAX_STEP does not fall back to XE_ANDERSON_MAX_STEP")
if "(t > 0.0) ? t" not in max_env:
    fail("a non-positive/unparsable RASBERY_XE_ANDERSON_MAX_STEP is not rejected; it would "
         "refuse every candidate and turn the feature off without saying so")
loop_at = SOLVE_LOOP.index("for (int iout = 0;")
if SOLVE_LOOP.index("static const double xe_aa_max_step = [] {") > loop_at:
    fail("the Anderson trust region is resolved inside the outer loop")

# Nothing may look the gate up inside the iteration: SolveLoop sees it through
# ONE local, hoisted before anything in the solve moves.
GATE_LOCAL = "const bool   xe_anderson    = xeAnderson();"
if GATE_LOCAL not in SOLVE_LOOP:
    fail(f"the Anderson gate is not read once into a local: {GATE_LOCAL!r}")
if SOLVE_CODE.count("xeAnderson()") != 1:
    fail("xeAnderson() must be consulted exactly once in SolveLoop (the xe_anderson term)")
for banned in ('getenv("RASBERY_XE_ANDERSON")', "xeAndersonDebug()"):
    if banned in SOLVE_CODE:
        fail(f"SolveLoop reaches {banned!r}; the gate must be one cached read outside the loop")
gate_at = SOLVE_LOOP.index(GATE_LOCAL)
for after in ("ctx.cmfd_solver.resetIteration();", "++ctx.telemetry.solve_loops;",
              "for (int iout = 0;"):
    j = SOLVE_LOOP.find(after)
    if j < 0:
        fail(f"ordering anchor vanished from SolveLoop: {after!r}")
    if j < gate_at:
        fail(f"the Anderson gate is decided after {after!r}; it must be checked before any "
             "behaviour change")

# The step site: the attempt short-circuits on the gate, and the production call
# is still there, unchanged, as the fallback.
ATTEMPT = ("const bool xe_aa_taken = xe_anderson && xe_relax == 1.0 && flux_converged &&\n"
           "                                         TryAndersonXeStep(ctx, xe_aa, "
           "schedule.thermalPower(),\n"
           "                                                           xe_aa_max_step, "
           "xe_change);")
if ATTEMPT not in SOLVE_LOOP:
    fail("the Anderson attempt is not `xe_anderson && xe_relax == 1.0 && flux_converged && "
         "TryAndersonXeStep(...)`; all three terms must short-circuit BEFORE the call, or "
         "the feature-off path evaluates the map")
FALLBACK = ("                if (!xe_aa_taken)\n"
            "                    xe_change =\n"
            "                        ctx.cross_sections.UpdateEquilibriumXenon("
            "schedule.thermalPower(), xe_relax);")
if FALLBACK not in SOLVE_LOOP:
    fail("a refused Anderson step does not fall back to the unchanged production "
         "UpdateEquilibriumXenon call")
if DRIVER.count("UpdateEquilibriumXenon(") != 1:
    fail("the Anderson arm grew a second UpdateEquilibriumXenon call site; the fallback "
         "must BE the production step, not a copy of it")
if "UpdateEquilibriumXenon" in AA:
    fail("TryAndersonXeStep calls the fused update; it must use the Evaluate/Commit split "
         "so that a refusal writes nothing")
# xe_change must stay a plain double the step writes once on each arm.
if "double     xe_change   = 0.0;" not in SOLVE_LOOP:
    fail("xe_change is no longer a single local written by exactly one of the two arms")

# ---------------------------------------------------------------------------
# 2. ApplyXeEquilibrium: split, not duplicated.
# ---------------------------------------------------------------------------
compute = region(XSSET, "static XeEquilibriumImage ComputeXeEquilibrium(",
                 "\n/// Apply Xe-135 equilibrium overwrite", "ComputeXeEquilibrium")
if re.search(r"iden\[[^\]]+\]\s*=[^=]", compute):
    fail("ComputeXeEquilibrium writes an iden entry; it is the COMPUTE half and must have "
         "no side effect at all")
if "const milk::Vector<double>& iden" not in compute:
    fail("ComputeXeEquilibrium takes iden by mutable reference; a const reference is what "
         "makes 'writes nothing' a compiler-checked property, not a convention")
for term in ("lambdaI", "lambdaXe", "lambdaXem", "brItoXe135m", "fissSourceI",
             "fissSourceXe", "sigaXe"):
    if term not in compute:
        fail(f"ComputeXeEquilibrium lost {term!r}; the split must move the arithmetic "
             "wholesale, not paraphrase it")

apply_fn = region(XSSET, "static void ApplyXeEquilibrium(", "\n// Divergence probe",
                  "ApplyXeEquilibrium")
if "ComputeXeEquilibrium(iden, cond, sumflux)" not in apply_fn:
    fail("ApplyXeEquilibrium does not delegate to ComputeXeEquilibrium; the two halves "
         "would be free to drift")
written = set(re.findall(r"iden\[(i[A-Za-z0-9]+)\]\s*=", apply_fn))
if written != {"iI135", "iXe135", "iXe135m"}:
    fail(f"ApplyXeEquilibrium writes {sorted(written)}; the wrapper must apply exactly the "
         "three Xe-chain rows it always did")
# Every pre-existing caller still calls the wrapper, so the depletion paths and
# the settled-flux loop are bit-for-bit what they were.
if XSSET.count("ApplyXeEquilibrium(") < 6:
    fail("ApplyXeEquilibrium lost call sites; the split must not move any caller onto the "
         "compute half")

# The fused device arm is untouched: it has no evaluate-only entry point, which
# is precisely why Evaluate is documented as host-only.
if "ComputeXeEquilibrium" in (ROOT / "src" / "XsReconKernel.h").read_text(
        encoding="utf-8-sig"):
    fail("the device kernel was split too; the campaign's bit-exact xsrecon A/B is gated "
         "on that kernel staying one fused body")

# ---------------------------------------------------------------------------
# 3. Evaluate writes nothing.
# ---------------------------------------------------------------------------
evaluate = region(XSSET, "double XSSet::EvaluateEquilibriumXenon(",
                  "\nvoid XSSet::CommitXenon(", "EvaluateEquilibriumXenon")
ev_code = code_lines(evaluate)
if re.search(r"_iden\[[^\]]+\]\s*=[^=]", ev_code):
    fail("EvaluateEquilibriumXenon assigns an _iden entry; it must be side-effect free or a "
         "refused Anderson candidate has already changed the solver")
if "ReconstructNode" in ev_code:
    fail("EvaluateEquilibriumXenon reconstructs a node; that belongs to Commit")
if "_hoststate_generation" in ev_code:
    fail("EvaluateEquilibriumXenon bumps the host-state generation; it wrote nothing, so "
         "there is nothing for the device arms to re-upload")
if re.search(r"_xs[.\[][^;]*=[^=]", ev_code):
    fail("EvaluateEquilibriumXenon writes a macroscopic XS entry")
if "ComputeXeEquilibrium(" not in ev_code:
    fail("EvaluateEquilibriumXenon does not go through ComputeXeEquilibrium; it would be a "
         "second copy of the closed form, free to drift from the applied one")
if "TryUpdateEquilibriumXenonGpu" in ev_code or "rasberyGpuXsReconEnabled" in ev_code:
    fail("EvaluateEquilibriumXenon reaches for the device arm; the fused kernel has no "
         "evaluate-only entry point, so this path is host-only by construction")
# The residual it returns has to be the SAME measurement the production step
# returns, or the trust region compares two different numbers.
if 'std::max(std::abs(img.xe135), 1.0e-30)' not in ev_code:
    fail("EvaluateEquilibriumXenon does not use the production |F(x)-x|/max(|F(x)|,1e-30) "
         "scale; the trust region would compare unlike quantities")
if "max_change = std::max(max_change, std::abs(img.xe135 - old_xe) / scale);" not in ev_code:
    fail("EvaluateEquilibriumXenon does not measure the RAW pre-damping change the way "
         "UpdateEquilibriumXenon does")
# The skipped nodes are the identity, expressed as data, so Commit of the image
# is a no-op there rather than a zero.
if "SnapshotXenon(iodine_out, xenon_out, xe135m_out);" not in ev_code:
    fail("EvaluateEquilibriumXenon does not seed its outputs with the current inventory; a "
         "node the loop skips would be committed as whatever the buffer held")

# ---------------------------------------------------------------------------
# 4. Commit writes exactly the three Xe-chain rows.
# ---------------------------------------------------------------------------
commit = region(XSSET, "void XSSet::CommitXenon(", "\nvoid XSSet::DepleteNode(", "CommitXenon")
cm_code = code_lines(commit)
rows = set(re.findall(r"_iden\[(i[A-Za-z0-9]+)\s*\*", cm_code))
if rows != {"iI135", "iXe135", "iXe135m"}:
    fail(f"CommitXenon writes {sorted(rows)}; exactly the three Xe-chain rows, and nothing "
         "of the rest of the isotope vector, which belongs to depletion")
if len(re.findall(r"_iden\[[^\]]+\]\s*=[^=]", cm_code)) != 3:
    fail("CommitXenon makes more (or fewer) than three _iden assignments")
if "ReconstructNode(l);" not in cm_code:
    fail("CommitXenon does not reconstruct the nodes it wrote; the flux solve would run on "
         "macro-XS built from the previous inventory")
if "++_hoststate_generation;" not in cm_code:
    fail("CommitXenon does not bump the host-state generation; the device arms would keep "
         "serving a stale resident copy of _iden/_xs")
if "throw std::runtime_error(" not in cm_code:
    fail("CommitXenon accepts a vector of the wrong length; a short vector commits a "
         "partial inventory and leaves the rest of the core on the previous iterate")
if "iodine.size() != nf || xenon.size() != nf || xe135m.size() != nf" not in cm_code:
    fail("the CommitXenon length check does not test all three vectors")

# ---------------------------------------------------------------------------
# 5. One fuel-node list, shared with the device arm.
# ---------------------------------------------------------------------------
if XSSET.count("_fuel_nodes.push_back(l);") != 1:
    fail("the fuel-node list is built in more than one place; Evaluate/Commit index by "
         "ordinal, so a second list is a silent layout fork")
fuel_fn = region(XSSET, "const std::vector<int>& XSSet::fuel_nodes() {",
                 "\nvoid XSSet::SnapshotXenon(", "fuel_nodes")
if "_g.IsFuel(l)" not in fuel_fn or "if (_fuel_nodes.empty())" not in fuel_fn:
    fail("fuel_nodes() is not the build-once, ascending-node-index list the ordinals mean")
# Rev.7.1 Task 13 moved the shared setup -- dimension check, backend creation,
# pinning, the ordinal list and the pointer view -- into PrepareXeDeviceCall, so
# the fused arm and the three split entry points cannot disagree about any of
# them.  The region therefore starts THERE: what this check has always been
# about is that the device side indexes through the ONE fuel_nodes() list rather
# than building a second one, and the single owner of that call is now the
# preparation function every device arm goes through.
gpu_arm = region(XSSET, "bool XSSet::PrepareXeDeviceCall(",
                 "\ndouble XSSet::UpdateEquilibriumXenon(", "PrepareXeDeviceCall")
if "fuel_nodes()" not in gpu_arm:
    fail("the xsrecon device arm no longer reads the shared fuel_nodes() list")
if gpu_arm.count("fuel_nodes()") != 1 or "_g.IsFuel" in gpu_arm:
    fail("a device Xe arm builds its own fuel-node list; there must be exactly one "
         "fuel_nodes() call, in PrepareXeDeviceCall, and no independent IsFuel scan")
for fn in ("SnapshotXenon", "EvaluateEquilibriumXenon", "CommitXenon"):
    body = region(XSSET, f"XSSet::{fn}(", "\n}\n", fn)
    if "fuel_nodes()" not in body:
        fail(f"XSSet::{fn} does not index through fuel_nodes(); the halves could disagree "
             "about the ordinal layout")
for decl in ("void SnapshotXenon(", "double EvaluateEquilibriumXenon(", "void CommitXenon(",
             "const std::vector<int>& fuel_nodes();"):
    if decl not in XSSET_H:
        fail(f"XSSet.h does not declare {decl!r}")

# ---------------------------------------------------------------------------
# 6. The formula, written out and pinned to the window depth.
# ---------------------------------------------------------------------------
DEPTH = re.search(r"static constexpr int\s+XE_ANDERSON_DEPTH\s*=\s*(\d+);", DRIVER)
if DEPTH is None:
    fail("XE_ANDERSON_DEPTH is not a static constexpr int in Driver.h")
if int(DEPTH.group(1)) != 2:
    fail(f"XE_ANDERSON_DEPTH = {DEPTH.group(1)}; Sec 10.5 starts at 2 and only tries 3 "
         "after the Gate -- and the normal equations below are written for 2")
if "static_assert(XE_ANDERSON_DEPTH == 2," not in AA:
    fail("nothing stops a depth bump from silently mis-solving: the two-column normal "
         "equations are written out by hand and need a static_assert guarding them")
for token, what in (
    ("const double det = a * c - b * b;", "the 2x2 normal-equation determinant"),
    ("gamma[0] = (c * p - b * q) / det;", "gamma_0 = (c*p - b*q)/det"),
    ("gamma[1] = (a * q - b * p) / det;", "gamma_1 = (a*q - b*p)/det"),
    ("gamma[j] = p / a;", "the one-column secant fallback gamma = <d,g>/<d,d>"),
    ("const double pred2 = gg - proj;", "the predicted squared residual"),
):
    if token not in AA:
        fail(f"{what} is missing from TryAndersonXeStep: {token!r}")
# The candidate is F_k minus the dF combination -- NOT x_k plus something, which
# is a different (and wrong) update.
cand = region(AA, "aa.cand.resize(n);", "// SAFEGUARD 3/4", "candidate assembly")
for token in ("double vi = aa.f.i135[i];", "vi -= gamma[j] * aa.df[j].i135[i];",
              "vx -= gamma[j] * aa.df[j].xe135[i];",
              "vm -= gamma[j] * aa.df[j].xe135m[i];"):
    if token not in cand:
        fail(f"the candidate is not x_(k+1) = F_k - sum_j gamma_j dF_j: {token!r} missing")
# The history columns are consecutive-evaluation differences of the RAW pairs.
if "XeSub(aa.f, aa.f_prev, aa.df[aa.ncol]);" not in AA:
    fail("dF is not F_(j+1) - F_j between consecutive evaluations")
if "XeSub(aa.g, aa.g_prev, aa.dg[aa.ncol]);" not in AA:
    fail("dG is not g_(j+1) - g_j between consecutive evaluations")
if "XeSub(aa.f, aa.x, aa.g);" not in AA:
    fail("the residual is not g_k = F(x_k) - x_k")
# Sec 10.5: raw undamped pairs only.  Nothing in the arm may apply a relaxation.
for banned in ("XE_DAMPED_RELAX", "xe_relax", "relax"):
    if banned in AA_CODE:
        fail(f"TryAndersonXeStep mentions {banned!r}; Sec 10.5 forbids damping an Anderson "
             "candidate, and the history stores raw undamped pairs only")

# ---------------------------------------------------------------------------
# 7. Four safeguards, each on the accept path, each refusing without committing.
# ---------------------------------------------------------------------------
commit_at = AA.index("xs.CommitXenon(")
head = AA[:commit_at]
SAFEGUARDS = (
    ("condition", "det > XE_ANDERSON_MIN_GRAM * a * c",
     "the least-squares conditioning guard (2-column Gram determinant)"),
    ("condition", "a > XE_ANDERSON_MIN_GRAM * gg",
     "the one-column conditioning guard (<d,d> against <g,g>)"),
    ("residual", "if (!(std::isfinite(pred2) && pred2 >= 0.0 && pred2 < gg)) {",
     "the predicted-residual-decrease guard"),
    ("physics", "vi < 0.0 || vx < 0.0 || vm < 0.0",
     "the non-negative-density guard"),
    ("physics", "std::isfinite(vi) && std::isfinite(vx) && std::isfinite(vm)",
     "the finite-density guard"),
    ("step", "if (!(step <= max_step * picard)) {",
     "the trust-region guard against the Picard step"),
)
for reason, token, what in SAFEGUARDS:
    if token not in head:
        fail(f"{what} is not on the accept path: {token!r}")
for reason in ("condition", "residual", "physics", "step"):
    call = f'RejectXeAnderson(ctx, "{reason}",'
    if call not in head:
        fail(f"the {reason!r} safeguard has no rejection: {call!r}")
    at = head.index(call)
    if "return false;" not in head[at:at + 200]:
        fail(f"the {reason!r} rejection does not return false; execution would fall through "
             "to the commit")
# Every rejection is BEFORE the commit, so a refused candidate never reaches it.
if AA.count("RejectXeAnderson(ctx,") != 4:
    fail("there must be exactly four rejection sites, one per safeguard")
if AA.count("xs.CommitXenon(") != 1:
    fail("TryAndersonXeStep has more than one commit site")
# The trust cap is a MULTIPLE of the Picard step, in the Picard step's own metric.
MAXSTEP = re.search(r"static constexpr double\s+XE_ANDERSON_MAX_STEP\s*=\s*([0-9.]+);", DRIVER)
if MAXSTEP is None:
    fail("XE_ANDERSON_MAX_STEP is not a static constexpr double in Driver.h")
if not 1.0 <= float(MAXSTEP.group(1)) <= 2.0:
    fail(f"XE_ANDERSON_MAX_STEP = {MAXSTEP.group(1)}; below 1 the cap refuses steps no "
         "larger than the Picard step it replaces, and far above it the bound stops "
         "standing in for the deferred true-residual check")
rel = region(DRIVER, "    static double XeRelativeChange(", "\n    /// Anderson history",
             "XeRelativeChange")
if "std::max(std::abs(cand.xe135[i]), 1.0e-30)" not in rel:
    fail("XeRelativeChange does not use the production |new-old|/max(|new|,1e-30) metric; "
         "the trust region must compare the candidate and the Picard step in ONE measure")
GRAM = re.search(r"static constexpr double\s+XE_ANDERSON_MIN_GRAM\s*=\s*([0-9.e-]+);", DRIVER)
if GRAM is None or not 0.0 < float(GRAM.group(1)) < 1.0:
    fail("XE_ANDERSON_MIN_GRAM is not a static constexpr double in (0,1); outside that it "
         "either rejects every fit or guards nothing")

# Arming is not rejection: no history, or a residual already inside the
# convergence tolerance, returns false WITHOUT charging a counter -- and the
# tolerance term is what keeps an unmeasured state out of the published
# inventory.
ARM = "if (aa.ncol == 0 || picard < XE_EQUILIBRIUM_TOLERANCE)"
if ARM not in AA:
    fail(f"the arming test is not {ARM!r}; without the tolerance term a cascade could "
         "publish an extrapolated inventory the convergence test never measured")
arm_at = AA.index(ARM)
if "++ctx.telemetry.xe_aa_proposed;" not in AA[arm_at:arm_at + 700]:
    fail("xe_aa_proposed is not charged immediately after arming; proposed would then "
         "count steps where no candidate was ever built")
if AA.index("++ctx.telemetry.xe_aa_proposed;") < arm_at:
    fail("xe_aa_proposed is charged before the arming test")

# ---------------------------------------------------------------------------
# 8. The damper wins.
# ---------------------------------------------------------------------------
if "xe_anderson && xe_relax == 1.0 &&" not in SOLVE_LOOP:
    fail("Anderson is not disabled while the oscillation damper is engaged; the damper is "
         "a ROOT SELECTOR and an extrapolation off the pre-damping map would fight it")
# And only the converged-flux evaluations are points of the composite map, so
# the interim-flux probe and the flux-limit-cycle fall-through take the plain
# step and contribute no history.
if "xe_relax == 1.0 && flux_converged &&" not in SOLVE_LOOP:
    fail("Anderson is attempted on a loose flux (the RASBERY_XE_INTERIM_L2 probe or the "
         "flux-limit-cycle fall-through); those evaluate a DIFFERENT map, so extrapolating "
         "across them fits one map's curvature onto another")
damper = region(SOLVE_LOOP,
                "if (!xe_once_mode && xe_relax == 1.0 && xe_no_progress >= xe_streak_limit) {",
                "\n                }\n", "damper activation")
if "xe_relax = XE_DAMPED_RELAX;" not in damper:
    fail("the damper block no longer engages the damping; the anchor moved")
if "ResetXeAndersonHistory(ctx, xe_aa);" not in damper:
    fail("damper activation does not discard the Anderson history (Sec 10.5: a relaxation "
         "change resets it)")

# ---------------------------------------------------------------------------
# 9. History resets: three sites, and a state that cannot outlive a solve.
# ---------------------------------------------------------------------------
DECL = "XeAndersonState xe_aa{};"
if DECL not in SOLVE_LOOP:
    fail(f"the Anderson state is not declared at SolveLoop entry as {DECL!r}, which IS "
         "cascade start 1")
if SOLVE_LOOP.index(DECL) > loop_at:
    fail("the Anderson state is declared inside the outer loop; it would reset every "
         "iteration and never accumulate a history")
# A SolveLoop local is what makes a depletion step structurally unable to carry
# history: Drive() runs PredictorStep/CorrectorStep BETWEEN SolveLoop calls.
if re.search(r"static\s+\w*\s*XeAndersonState", DRIVER):
    fail("the Anderson state is static; history would leak across statepoints, depletion "
         "steps and (in batch mode) decks")
if "XeAndersonState" in region(DRIVER, "    struct SolverContext {", "\n    };",
                               "SolverContext"):
    fail("SolverContext carries the Anderson state; it would then survive the depletion "
         "steps Drive() runs between SolveLoop calls")
if DRIVER.count("XeAndersonState xe_aa") != 1:
    fail("more than one Anderson state exists in the solve path")
rearm = region(SOLVE_LOOP, "if (xe_restart && has_eq_xe) {", "\n        }\n",
               "cascade re-arm")
if "ResetXeAndersonHistory(ctx, xe_aa);" not in rearm:
    fail("the cascade re-arm after a committed T/H or search perturbation does not discard "
         "the history; the fit would carry the OLD map's curvature onto the new one")
if rearm.index("ResetXeAndersonHistory(ctx, xe_aa);") > rearm.index("if (xe_cascade_budget)"):
    fail("the history reset is inside the RASBERY_XE_CASCADE_BUDGET branch; the Anderson "
         "reset must not depend on an unrelated experiment gate")
if SOLVE_CODE.count("ResetXeAndersonHistory(ctx, xe_aa);") != 2:
    fail("the history must be reset at exactly two sites inside SolveLoop -- the cascade "
         "re-arm and damper activation (SolveLoop entry is the declaration)")
reset = region(DRIVER, "    static void ResetXeAndersonHistory(",
               "\n    /// Charge one rejection", "ResetXeAndersonHistory")
if "if (!aa.holds_history())" not in reset:
    fail("ResetXeAndersonHistory charges a reset even with nothing stored; the counter "
         "would count outers instead of discards")
if "aa.forget();" not in reset:
    fail("ResetXeAndersonHistory does not actually drop the history")
forget = region(DRIVER, "        void forget() {", "\n    };", "forget")
if "have_prev = false;" not in forget or "ncol      = 0;" not in forget:
    fail("forget() does not clear BOTH the previous-evaluation flag and the column count; "
         "a stale column would be differenced against a new map")

# ---------------------------------------------------------------------------
# 10. Telemetry: four additive counters, charged only inside the arm.
# ---------------------------------------------------------------------------
COUNTERS = ("xe_aa_proposed", "xe_aa_accepted", "xe_aa_rejected", "xe_aa_history_resets")
# ONE CHARGE PER ARM, AND THE ARMS ARE ENUMERATED (ARMS, above).
#
# Rev.7.1 Task 13 made the device arm a SIBLING of the host one rather than a
# shared body -- Driver.h says why -- and WP7-C added a third sibling, the
# device transaction.  The price of that decision is that a per-proposal
# counter is written out once per arm.  That is not a double count: the
# dispatch chain runs exactly one arm's body per call, so a proposal is charged
# exactly once however the run is flagged, which is what lets the WP7-C receipt
# claim anderson_proposed/anderson_accepted are unchanged by TXN.
#
# What WOULD break the totals is a second charge INSIDE one arm (the same
# proposal counted twice) or a charge site outside every arm (a step Anderson
# never took).  Both are failures, and the two conditions together are what
# `charge_sites_ok` tests: once in each arm, and no charges left over.
#
# "rejected" and "history_resets" stay at ONE site each: all three arms reach
# them through RejectXeAnderson / ResetXeAndersonHistory, which is what makes
# the arms' rejection distributions and reset counts comparable at all.
PER_ARM = ("xe_aa_proposed", "xe_aa_accepted")


def charge_sites_ok(counter: str, arms, driver: str) -> bool:
    """Exactly one charge of `counter` in every arm, and none anywhere else."""
    charge = f"++ctx.telemetry.{counter};"
    return (all(arm.count(charge) == 1 for _, arm in arms)
            and driver.count(charge) == len(arms))


# NEGATIVE CONTROL for that rule, because a rule that cannot fail is a comment.
# A double charge inside one arm and a site outside every arm must each be
# caught, and a correct layout must pass.
_CHARGE = "        ++ctx.telemetry.xe_aa_proposed;\n"
if not charge_sites_ok("xe_aa_proposed", (("a", _CHARGE), ("b", _CHARGE)), _CHARGE * 2):
    fail("negative control: charge_sites_ok rejects a correct one-per-arm layout")
if charge_sites_ok("xe_aa_proposed", (("a", _CHARGE), ("b", _CHARGE * 2)), _CHARGE * 3):
    fail("negative control: a second charge inside one arm is not detected; the same "
         "proposal would be counted twice")
if charge_sites_ok("xe_aa_proposed", (("a", _CHARGE), ("b", _CHARGE)), _CHARGE * 3):
    fail("negative control: a charge site outside every enumerated arm is not detected; "
         "a fourth arm (or a charge in SolveLoop) would pass unseen")

for counter in COUNTERS:
    if f"long long {counter}" not in DRIVER:
        fail(f"sptelem::Counters has no {counter} field")
    if not re.search(rf"^\s*{counter}\s*\+= step\.{counter};", DRIVER, re.M):
        fail(f"{counter} is not folded into the run totals by Counters::accumulate")
    if DRIVER.count(f'\\"{counter}\\":{{}}') != 2:
        fail(f"{counter} is not published by both SPTELEM receipts")
    if DRIVER.count(f"c.{counter}") != 2:
        fail(f"{counter} is not passed as an argument at both receipt sites")
    charge = f"++ctx.telemetry.{counter};"
    if counter in PER_ARM:
        for name, arm in ARMS:
            n = arm.count(charge)
            if n != 1:
                fail(f"{counter} is charged {n} time(s) in the {name} arm; exactly one arm "
                     "body runs per call, so each must charge a proposal exactly once -- "
                     "twice in one arm counts the same proposal twice, never charging it "
                     "leaves that arm's receipt claiming steps it did not take")
        if not charge_sites_ok(counter, ARMS, DRIVER):
            fail(f"{counter} is charged {DRIVER.count(charge)} time(s) in Driver.h but "
                 f"there are {len(ARMS)} enumerated arms, one charge each; the extra site "
                 "is outside every arm -- either a FOURTH arm nobody added to ARMS, or a "
                 "charge on a step Anderson never took")
    elif DRIVER.count(charge) != 1:
        fail(f"{counter} must be charged in exactly 1 place; all three arms reach it "
             "through RejectXeAnderson / ResetXeAndersonHistory, which is what makes "
             "their rejection distributions and reset counts comparable")
    # ...and that place is inside the gated arm, never in the loop body, where it
    # would be charged on a step Anderson never took.
    if f"ctx.telemetry.{counter}" in SOLVE_CODE:
        fail(f"{counter} is charged from SolveLoop; every charge belongs inside "
             "TryAndersonXeStep / ResetXeAndersonHistory, which only run under the gate")
# proposed == accepted + rejected, structurally: one proposal, then exactly one
# of the two outcomes on every path out.
if AA.count("++ctx.telemetry.xe_aa_accepted;") != 1:
    fail("the accept counter is charged more than once")
acc_at = AA.index("++ctx.telemetry.xe_aa_accepted;")
if acc_at < commit_at:
    fail("xe_aa_accepted is charged before the commit; a throw in Commit would leave the "
         "receipt claiming a step that never landed")
if "return true;" not in AA[acc_at:]:
    fail("the accept path does not return true")
if AA_CODE.count("return true;") != 1:
    fail("TryAndersonXeStep has more than one success exit")

# ---------------------------------------------------------------------------
# 11. The two deviations from Sec 10 are DOCUMENTED, not silent.
# ---------------------------------------------------------------------------
notes = region(DRIVER, "    // TWO DEVIATIONS FROM SEC 10", "    struct XeTriple {",
               "deviation notes")
for token, what in (
    ("NO FULL-EXACT TRUE-RESIDUAL ACCEPTANCE", "the deferred Sec 10.4 true-residual accept"),
    ("NO AXIAL BRANCH GUARD", "the deferred Sec 10.4 axial branch guard"),
    ("CoupledStateSnapshot", "the deferred Sec 10.2/10.3 snapshot + rollback"),
):
    if token not in notes:
        fail(f"{what} is not documented as a deviation: {token!r} missing")
# If the deferred machinery ever lands, this contract has to be revisited --
# so a snapshot/rollback appearing in the solver fails on purpose.
for symbol in ("EvaluateCoupledXeCandidate", "CoupledStateSnapshot ", "TrialResult "):
    if symbol in code_lines(DRIVER):
        fail(f"{symbol!r} now exists in Driver.h: Sec 10.2/10.3 landed, so the trust cap is "
             "no longer standing in for the true-residual accept and this contract's "
             "deviation notes are stale")

py_compile.compile(str(Path(__file__).resolve()), doraise=True)
print("xe anderson contract: PASS")
