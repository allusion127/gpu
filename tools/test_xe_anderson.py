#!/usr/bin/env python3
"""Static contract for the safeguarded Anderson Xe arm (RASBERY_XE_ANDERSON).

Plan Rev.4 Sec 10.  The arm is a runtime switch on a GPU solver, so the
properties that make it safe cannot be observed from a Python test run -- they
have to be checked in the source.  Ten things are guarded here:

  1. BYTE-GOLDENNESS.  With RASBERY_XE_ANDERSON unset the solve path must be
     what it was: the env var is read exactly once, cached in a function-local
     static, hoisted into ONE local at SolveLoop entry, and the single call it
     feeds short-circuits on that local before anything is evaluated.  No
     getenv inside the outer loop, and the production UpdateEquilibriumXenon
     call still sits where it always sat.
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
     totals and published by both SPTELEM receipts.

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


SOLVE_LOOP = region(DRIVER, "    static void SolveLoop(",
                    "    /// Where this run's restart_", "SolveLoop")
SOLVE_CODE = code_lines(SOLVE_LOOP)
DRIVE = region(DRIVER, "    int Drive() {", "\n};\n", "Drive")
AA = region(DRIVER, "    static bool TryAndersonXeStep(",
            "\n    // Frozen-Xe startup guard", "TryAndersonXeStep")
AA_CODE = code_lines(AA)

# ---------------------------------------------------------------------------
# 1. The env gate: one read, cached, default = the old behaviour.
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

gate = region(DRIVER, "inline bool xeAnderson() {", "\n/// Per-rejection reason trace",
              "xeAnderson")
if "static const bool on = [] {" not in gate:
    fail("xeAnderson() does not cache the env lookup in a function-local static")
null_at = gate.find("if (value == nullptr)")
if null_at < 0 or "return false;" not in gate[null_at:null_at + 80]:
    fail("an unset RASBERY_XE_ANDERSON does not return false before anything else; "
         "the default path would not be byte-golden")
for off in ('s == "0"', 's == "off"', 's == "false"', "s.empty()"):
    if off not in gate:
        fail(f"xeAnderson() does not treat {off!r} as off; the other Xe gates all do")
# EQUILIBRIUM ONLY, and said out loud.  frozen has no cascade to accelerate and
# once is deliberately not converging one.
if "xeMode() != XeMode::EQUILIBRIUM" not in gate:
    fail("xeAnderson() does not refuse the non-equilibrium modes; Anderson accelerates "
         "the equilibrium cascade and there is nothing to accelerate in frozen/once")
if "[RASBERY][WARN][xe]" not in gate:
    fail("xeAnderson() silently swallows a mode conflict; a misconfiguration must be "
         "reported, not quietly honoured as 'off'")
if "return false;" not in gate[gate.index("xeMode() != XeMode::EQUILIBRIUM"):]:
    fail("the mode conflict warns but still enables the arm")

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
gpu_arm = region(XSSET, "bool XSSet::TryUpdateEquilibriumXenonGpu(",
                 "\ndouble XSSet::UpdateEquilibriumXenon(", "TryUpdateEquilibriumXenonGpu")
if "fuel_nodes()" not in gpu_arm:
    fail("the xsrecon device arm no longer reads the shared fuel_nodes() list")
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
for counter in COUNTERS:
    if f"long long {counter}" not in DRIVER:
        fail(f"sptelem::Counters has no {counter} field")
    if not re.search(rf"^\s*{counter}\s*\+= step\.{counter};", DRIVER, re.M):
        fail(f"{counter} is not folded into the run totals by Counters::accumulate")
    if DRIVER.count(f'\\"{counter}\\":{{}}') != 2:
        fail(f"{counter} is not published by both SPTELEM receipts")
    if DRIVER.count(f"c.{counter}") != 2:
        fail(f"{counter} is not passed as an argument at both receipt sites")
    if DRIVER.count(f"++ctx.telemetry.{counter};") != 1:
        fail(f"{counter} must be charged in exactly one place")
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
