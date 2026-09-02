#!/usr/bin/env python3
"""Static contract for WP20's device-wide single-precision arm.

RASBERY_GPU_FP32 narrows DEVICE HOT STATE to float.  It is gate class A2 -- it
moves the trajectory on purpose -- so the bit-golden gate cannot validate it and
the numbers are settled by Gate A (against the FP64 trajectory) and Gate B
(against MASTER).  What a static gate CAN settle is everything a reviewer would
otherwise have to take on trust, and that is this file:

  1. ONE ARM, ONE PREDICATE.  Three env knobs, each read once into a cached
     static; every narrow launch site in the tree asks `routes()` and nothing
     else, so the captured graph topology and the kernels inside it can never
     disagree about which precision this run is.
  2. FEATURE-OFF IS UNREACHABLE, NOT MERELY UNTAKEN.  With the flag unset no
     float kernel is reachable: the narrow launch parameter defaults to false,
     every FP64 kernel and struct is textually intact, and the FP64 arm of every
     dispatch still launches what it launched before.  That is what the
     byte-identity claim (digest 1f36e75dc00ed2b4 / 4377) rests on.
  3. THE RECEIPT CANNOT LIE.  A backend the tree does not actually narrow
     reports "deferred", not "fp32" -- so a wall-clock A/B can always be
     attributed to a conversion that really happened.
  4. THE FP64 DEVIATION IS DECLARED.  The residual-norm accumulation and the
     convergence scalars stay double on purpose; the header says so, the
     reduction kernels prove it, and RASBERY_GPU_FP32_STRICT is the arm that
     makes the claim testable rather than asserted.
  5. THE FALLBACK IS LOUD, COUNTED, STICKY AND ENV-INDEPENDENT, and the CMFD
     backend bridges its own latch into it.

Pure python, no build, no device.

Run:  python tools/test_gpu_fp32_contract.py
"""

from __future__ import annotations

import os
import py_compile
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []
checks = 0


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8-sig") as fh:
        return fh.read()


def check(ok: bool, what: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(what)


def strip_comments(text: str) -> str:
    """Rules are about CODE.  These headers describe the arm in prose and name
    every construct they forbid, so a rule that searched the raw text would fire
    on its own documentation."""
    return re.sub(r"//[^\n]*", "", text)


def body_after(text: str, anchor: str) -> str:
    """The brace-matched block that opens at the first '{' after `anchor`."""
    start = text.find(anchor)
    if start < 0:
        raise LookupError(anchor)
    open_at = text.find("{", start)
    if open_at < 0:
        raise LookupError(anchor)
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
    raise LookupError(anchor)


ARM = read("src", "GpuFp32Arm.h")
ARM_CODE = strip_comments(ARM)
CTA = read("src", "FlatXsCtaKernel.cuh")
CTA_CODE = strip_comments(CTA)
BACKEND = read("src", "CudaXsReconBackend.cu")
BICG = read("src", "CudaBICGBackend.cu")
DRIVER = read("src", "Driver.h")
MAIN = read("src", "main.cpp")
DOC = read("docs", "WP20_GPU_FP32_20260831_KO.md")

BACKENDS = ("cmfd", "nodal", "flatxs", "xe", "cram", "ppr")


# ---------------------------------------------------------------------------
# 1. THE THREE KNOBS: OPT-IN, CACHED ONCE, DECLARED WHERE ARMS ARE DECLARED.
#
# Cached because the choice fixes the captured graph TOPOLOGY (the narrow and
# wide kernel sets are different nodes), so it must not be able to change
# between two outers of one run.  Declared in kArmEnv because it selects the
# rounding of the whole device iteration -- the most trajectory-moving thing a
# knob in this binary can do -- and because being on that list is what folds it
# into the WP10.1 case key, so an FP64 answer can never be served to an FP32
# request.
# ---------------------------------------------------------------------------
KNOBS = {
    "RASBERY_GPU_FP32": "armed",
    "RASBERY_GPU_FP32_STRICT": "strict",
    "RASBERY_GPU_FP32_CRAM": "cramExtended",
}


def rule_gate_is_cached_opt_in(arm: str, knob: str, fn: str) -> bool:
    try:
        gate = body_after(arm, "inline bool %s()" % fn)
    except LookupError:
        return False
    return ('envFlagOn("%s")' % knob) in gate and "static const bool on" in gate


for knob, fn in KNOBS.items():
    check(rule_gate_is_cached_opt_in(ARM, knob, fn),
          "%s is read once into a cached static by %s(), opt-IN" % (knob, fn))
    check('"%s",' % knob in DRIVER,
          "%s is listed in Driver.h's kArmEnv -- a knob that selects the "
          "rounding of every device kernel must be in the case key" % knob)
    check(ARM_CODE.count('"%s"' % knob) == 1,
          "%s is spelled exactly once in the arm header; a second reader is a "
          "second policy" % knob)
    check(knob not in strip_comments(BICG) and knob not in strip_comments(BACKEND),
          "%s is never read outside src/GpuFp32Arm.h -- the backends ask "
          "routes(), not the environment" % knob)

check("std::getenv" in ARM and 'std::getenv("RASBERY_GPU_FP32")' not in ARM,
      "the arm header reads the environment through ONE helper (envFlagOn) "
      "rather than calling getenv per knob")


# ---------------------------------------------------------------------------
# 2. ONE ROUTING PREDICATE, AND EVERY BACKEND IS NAMED IN IT.
#
# `routes()` is the only thing a launch site is allowed to ask.  It is
# (armed AND in scope AND not latched), and `inScope` is (converted AND, for
# CRAM, its own extension flag).  The Backend enum names all six subsystems that
# own device floating-point state so the receipt cannot quietly omit one.
# ---------------------------------------------------------------------------
try:
    ENUM = body_after(ARM, "enum class Backend : int")
except LookupError:
    ENUM = ""
for name in ("Cmfd", "Nodal", "FlatXs", "Xe", "Cram", "Ppr"):
    check(name in ENUM, "Backend names %s" % name)
check("Count" in ENUM, "Backend carries a Count terminator so the receipt loop "
                       "cannot fall out of step with the enum")

try:
    ROUTES = body_after(ARM, "inline bool routes(Backend which)")
except LookupError:
    ROUTES = ""
check("armed()" in ROUTES and "inScope(which)" in ROUTES and "!latched(which)" in ROUTES,
      "routes() is (armed AND in scope AND NOT latched) and nothing else")

try:
    SCOPE = body_after(ARM, "inline bool inScope(Backend which)")
except LookupError:
    SCOPE = ""
check("converted(which)" in SCOPE and "cramExtended()" in SCOPE,
      "inScope() refuses a backend the tree does not narrow, and refuses CRAM "
      "unless RASBERY_GPU_FP32_CRAM extends the arm to it")

try:
    CONVERTED = body_after(ARM, "inline bool converted(Backend which)")
except LookupError:
    CONVERTED = ""
for name in ("Cmfd", "Nodal", "FlatXs", "Xe", "Cram", "Ppr"):
    check("Backend::%s:" % name in CONVERTED,
          "converted() answers for %s explicitly -- a backend that falls "
          "through to a default is a backend nobody decided about" % name)
check("case Backend::Cmfd:   return true;" in CONVERTED
      and "case Backend::FlatXs: return true;" in CONVERTED,
      "CMFD and flat-XS are the two backends WP20 actually narrows")
check("case Backend::Nodal:  return false;" in CONVERTED
      and "case Backend::Xe:     return false;" in CONVERTED
      and "case Backend::Cram:   return false;" in CONVERTED
      and "case Backend::Ppr:    return false;" in CONVERTED,
      "nodal / xe / cram / ppr are declared DEFERRED in the table rather than "
      "left to look converted")


# ---------------------------------------------------------------------------
# 3. THE RECEIPT: `[RASBERY][FP32] {arm, backends:{...}, demotions,
#    nonfinite_fallbacks, bytes_saved_est}`, printed from every branch.
#
# Printed WHETHER OR NOT the arm is on, the same G0 rule [RASBERY][GPU_FULL]
# exists for: "the arm was on and never engaged" must not be able to look like
# "the arm was off".
# ---------------------------------------------------------------------------
try:
    RECEIPT = body_after(ARM, "inline void appendReceiptFields(std::ostream& out)")
except LookupError:
    RECEIPT = ""
for field in ('\\"arm\\":', ',\\"backends\\":{', ',\\"demotions\\":',
              ',\\"nonfinite_fallbacks\\":', ',\\"bytes_saved_est\\":'):
    check(field.replace("\\", "") in RECEIPT.replace("\\", ""),
          "the receipt carries %s" % field.replace("\\", ""))
check("backendName(which)" in RECEIPT and "backendState(which)" in RECEIPT,
      "the backends object is generated from the enum, so it cannot omit one")

try:
    STATE = body_after(ARM, "inline const char* backendState(Backend which)")
except LookupError:
    STATE = ""
for word in ('"fp64"', '"deferred"', '"declined"', '"latched"', '"fp32"'):
    check(word in STATE, "backendState distinguishes %s" % word)
check(STATE.index('return "deferred"') < STATE.index('return "fp32"'),
      "a backend the tree does not narrow reports deferred BEFORE anything can "
      "report fp32 -- the receipt may never claim a conversion that is absent")

check(MAIN.count('"[RASBERY][FP32] {"') == 3,
      "the receipt is printed from all three of main.cpp's exit branches")
check(MAIN.count("rasbery::fp32::appendReceiptFields(std::cout);") == 3,
      "each of those three prints the same fields")
check('#include "GpuFp32Arm.h"' in MAIN,
      "main.cpp includes the arm header it prints from")
check(re.search(r"if\s*\([^)]*fp32[^)]*\)\s*\{?\s*std::cout << \"\[RASBERY\]\[FP32\]",
                MAIN) is None,
      "the receipt is UNCONDITIONAL -- an arm-off run must say so out loud "
      "rather than by omission")


# ---------------------------------------------------------------------------
# 4. CMFD: THE WHOLE INNER SOLVE, THROUGH THE EXISTING KERNEL SET.
#
# The arm does not add a second CMFD precision path; it OR-s into the one that
# already exists, at the one cached gate, so there is still one captured
# topology decision.  tools/test_cmfd_fp32_contract.py owns the kernel-level
# properties of that path; what belongs HERE is the join.
# ---------------------------------------------------------------------------
try:
    CMFD_GATE = body_after(BICG, "bool cmfdFp32InnerEnabled()")
except LookupError:
    CMFD_GATE = ""
check("rasbery::fp32::routes(rasbery::fp32::Backend::Cmfd)" in CMFD_GATE,
      "RASBERY_GPU_FP32 routes CMFD through the existing cached inner-solve gate")
check('envFlagEnabled("RASBERY_GPU_CMFD_FP32")' in CMFD_GATE,
      "the historical per-backend knob still reaches the same path")
check("static const bool enabled" in CMFD_GATE,
      "the joined gate is still resolved once and cached")
check(strip_comments(BICG).count("rasbery::fp32::routes(") == 1,
      "CMFD asks the arm exactly once; a per-launch predicate could disagree "
      "with the graph it is inside")
check('#include "GpuFp32Arm.h"' in BICG,
      "src/CudaBICGBackend.cu includes the arm header")


# ---------------------------------------------------------------------------
# 5. FLAT-XS: A NARROWED WORKSPACE, ONE BODY, AND AN OFF PATH THAT CANNOT MOVE.
# ---------------------------------------------------------------------------
try:
    WS64 = body_after(CTA, "struct CtaWorkspace {")
    WS32 = body_after(CTA, "struct CtaWorkspaceF32 {")
except LookupError:
    WS64 = WS32 = ""
EXPECTED_DIMS = {
    "bl": "N_ACTIVE * NG",
    "bls": "NLSM",
    "bm": "N_ACTIVE * NMIC",
    "bms": "NMSM",
    "iden": "NISO",
}
for name, dim in EXPECTED_DIMS.items():
    check(re.search(r"float\s+%s\[%s\]" % (name, re.escape(dim)), WS32) is not None,
          "CtaWorkspaceF32::%s is float and sized [%s] -- the SAME expression "
          "the FP64 twin uses, so the isotope registry cannot move under one "
          "struct and not the other" % (name, dim))
    check(re.search(r"double\s+%s\[%s\]" % (name, re.escape(dim)), WS64) is not None,
          "CtaWorkspace::%s is still double and still sized [%s]" % (name, dim))
check(re.search(r"double\s+\w+\[", WS32) is None,
      "no field of CtaWorkspaceF32 stayed double")
check(re.search(r"float\s+\w+\[\s*\d+\s*\]", WS32) is None,
      "no array bound in CtaWorkspaceF32 is a literal")


def rule_one_body_for_both_arms(cta: str) -> bool:
    """The narrow and wide arms must be TWO INSTANTIATIONS OF ONE TEMPLATE, not
    two bodies.  Two bodies drift; a template cannot."""
    return ("template <int T, class POL, class WS>" in cta
            and "const POL& pol, WS& w) {" in cta
            and cta.count("flatxsSolveNodeCta(const FlatXsView& v, int i,") == 1)


check(rule_one_body_for_both_arms(CTA),
      "the CTA body is ONE template parameterised on the workspace type, so the "
      "FP32 arm cannot drift away from the FP64 reference")
check("const double* mt = w.bm" not in CTA_CODE and "const auto* mt = w.bm" in CTA_CODE,
      "the one place the body named the workspace element type now deduces it")

check("__shared__ CtaWorkspace w;" in CTA_CODE
      and "__shared__ CtaWorkspaceF32 w;" in CTA_CODE,
      "each kernel declares its own static __shared__ workspace: the shared "
      "allocation is a compile-time property, so a runtime branch would have to "
      "reserve the wide one and would save nothing")
for kernel in ("kernelFlatXsCta", "kernelFlatXsCtaF32"):
    check("__global__ void __launch_bounds__(T) %s(FlatXsView v)" % kernel in CTA,
          "%s is a __launch_bounds__(T) kernel like its twin" % kernel)


def rule_narrow_defaults_off(cta: str) -> bool:
    """Every caller that predates the arm -- the replay gate included -- must
    keep launching exactly the kernel it launched before."""
    return "bool narrow = false)" in cta


check(rule_narrow_defaults_off(CTA),
      "flatxsCtaLaunch's precision parameter DEFAULTS TO FALSE, which is what "
      "the feature-off byte-identity claim rests on")

try:
    LAUNCH = body_after(CTA, "inline void flatxsCtaLaunch(")
except LookupError:
    LAUNCH = ""
try:
    NARROW_ARM = body_after(LAUNCH, "if (narrow) {")
except LookupError:
    NARROW_ARM = ""
check(LAUNCH.count("kernelFlatXsCtaF32<") == 3,
      "the narrow arm spells the same three-entry block-size ladder")
check(NARROW_ARM.count("kernelFlatXsCtaF32<") == 3
      and "kernelFlatXsCta<" not in NARROW_ARM,
      "every F32 launch sits inside the `narrow` arm, and no FP64 launch leaked "
      "into it")
WIDE_ARM = LAUNCH[LAUNCH.index("return;", LAUNCH.index("if (narrow) {")):]
check("kernelFlatXsCtaF32<" not in WIDE_ARM,
      "no F32 launch leaked into the wide arm")
check(CTA.count("kernelFlatXsCtaF32<") == 3,
      "kernelFlatXsCtaF32 is launched from exactly one place")

try:
    SOLVE = body_after(BACKEND, "bool XsReconBackend::solveFlatXs(")
except LookupError:
    SOLVE = ""
check("static const bool narrow = rasbery::fp32::routes(rasbery::fp32::Backend::FlatXs);"
      in SOLVE,
      "solveFlatXs resolves the arm ONCE into a cached bool, for the same "
      "reason it caches the CTA flag")
check(strip_comments(SOLVE).count("rasbery::fp32::routes(") == 1,
      "there is exactly ONE precision dispatch point in solveFlatXs")
check("fxs::flatxsCtaLaunch(v, rasberyGpuFlatXsCtaThreads(), d.stream, narrow);" in SOLVE,
      "the precision travels with the launch rather than through a global")
check("kernelFlatXs<<<grid, block, 0, d.stream>>>(v);" in SOLVE,
      "the thread-per-node reference arm is untouched and still reachable")
check("rasbery::fp32::noteDemotion(rasbery::fp32::Backend::FlatXs);" in SOLVE,
      "an FP32 request that lands on the reference arm (CTA=0) is COUNTED as a "
      "demotion rather than silently ignored -- the narrow workspace exists "
      "only on the CTA arm")


# ---------------------------------------------------------------------------
# 6. THE DECLARED FP64 DEVIATION, AND THE ARM THAT MAKES IT TESTABLE.
#
# The residual-norm ACCUMULATION and the convergence decision scalars stay FP64
# so the outer iteration counts stay comparable between the two arms -- if they
# did not, Gate A would be comparing two different amounts of work and could not
# attribute the difference to precision at all.  float x float widened to double
# is EXACT, so the wide accumulator adds no rounding of its own; and the dots
# are memory bound at these sizes, so it costs no time.
# ---------------------------------------------------------------------------
for stage1 in ("reduce_dot_stage1_f32", "reduce_dot2_stage1_f32"):
    block = body_after(BICG, "__global__ void %s(" % stage1)
    check("__shared__ double" in block and "__shared__ float" not in block,
          "%s still folds a float payload into a DOUBLE accumulator" % stage1)
check("reduce_dot_stage2<<<" in body_after(BICG, "void dot_f32(const float* a"),
      "the FP32 dot reuses the unmodified double stage-2 fold")

for phrase in ("RESIDUAL-NORM ACCUMULATION", "CONVERGENCE DECISION SCALARS",
               "iterative refinement", "RASBERY_GPU_FP32_STRICT"):
    check(phrase in ARM,
          "src/GpuFp32Arm.h declares the FP64 deviation in the words %r" % phrase)
check("A2" in ARM and "Gate A" in ARM and "Gate B" in ARM,
      "the header states the gate class and names the two numeric gates that "
      "decide acceptance, since the bit-golden gate cannot")


# ---------------------------------------------------------------------------
# 7. THE NON-FINITE FALLBACK: loud, counted, sticky, env-independent.
# ---------------------------------------------------------------------------
try:
    LATCH = body_after(ARM, "inline bool latchOff(Backend which, const char* reason)")
except LookupError:
    LATCH = ""
check("[RASBERY][FP32][FALLBACK]" in LATCH,
      "the fallback prints a loud, greppable line")
check("std::cerr" in LATCH, "it prints to stderr, like every other fallback here")
check("fallbacks[idx].fetch_add(1" in LATCH, "the fallback is counted")
check("compare_exchange_strong" in LATCH,
      "the latch is set exactly once, so a storm of non-finites is one line and "
      "not one line per case")
check("getenv" not in LATCH and "envFlagOn" not in LATCH,
      "the fallback is ENV-INDEPENDENT: it is a safety valve, not an arm")
check("rasbery::fp32::latchOff(rasbery::fp32::Backend::Cmfd, \"nonfinite\");"
      in body_after(BICG, "void latchFp32Off()"),
      "the CMFD backend bridges its own sticky latch into the arm, so the "
      "receipt reports cmfd:\"latched\" instead of cmfd:\"fp32\"")
check("fp32_latched_off = true;" in body_after(BICG, "void latchFp32Off()")
      and "destroyGraphCaches();" in body_after(BICG, "void latchFp32Off()"),
      "the CMFD latch still drops the graphs captured under the old precision")


# ---------------------------------------------------------------------------
# 8. BYTES SAVED IS ACCOUNTED WHERE IT IS TRUE, AND NOWHERE ELSE.
# ---------------------------------------------------------------------------
check(strip_comments(BICG).count("rasbery::fp32::noteBytesSaved(") == 1
      and strip_comments(BACKEND).count("rasbery::fp32::noteBytesSaved(") == 1,
      "each converted backend accounts its own footprint delta exactly once")
check("sizeof(fxs::CtaWorkspaceF32)" in SOLVE,
      "the flat-XS estimate is derived from the struct, not from a copied 3,676")
try:
    ALLOC = body_after(BICG, "if (fp32_inner) {")
except LookupError:
    ALLOC = ""
check("noteBytesSaved" in ALLOC,
      "the CMFD estimate is taken inside the block that allocates the narrow "
      "working set, so it cannot be counted for an arm that never allocated one")


# ---------------------------------------------------------------------------
# 9. THE DOC CARRIES THE 238 RUNBOOK THIS FILE DEFERS TO.
# ---------------------------------------------------------------------------
for token in ("RASBERY_GPU_FP32", "RASBERY_GPU_FP32_STRICT", "RASBERY_GPU_FP32_CRAM",
              "1.905", "15.309", "0.013", "0.238", "0.80",
              "compare_master_rasbery.py", "gate_b_pin_rms.py",
              "1f36e75dc00ed2b4", "4377", "8×M16",
              "dram__throughput.avg.pct_of_peak_sustained_elapsed",
              "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld",
              "sm__warps_active.avg.pct_of_peak"):
    check(token in DOC, "docs/WP20_GPU_FP32_20260831_KO.md names %s" % token)
check("A2" in DOC and "Gate A" in DOC and "Gate B" in DOC,
      "the doc states the gate class and the two gates that decide acceptance")
for deferred in ("nodal", "xe", "cram", "ppr"):
    check(deferred in DOC.lower(),
          "the doc says what happened to %s" % deferred)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each probe is the rule applied to a mutation that MUST
# make it fail; a rule that cannot fail is not a rule.
# ---------------------------------------------------------------------------
NEGATIVES: list[tuple[str, object]] = [
    ("the arm becomes opt-OUT",
     lambda: rule_gate_is_cached_opt_in(
         "inline bool armed() { static const bool on = "
         '!envFlagOff("RASBERY_GPU_FP32"); return on; }',
         "RASBERY_GPU_FP32", "armed")),
    ("the arm is re-read per call instead of cached",
     lambda: rule_gate_is_cached_opt_in(
         'inline bool armed() { return envFlagOn("RASBERY_GPU_FP32"); }',
         "RASBERY_GPU_FP32", "armed")),
    ("the narrow launch parameter stops defaulting off",
     lambda: rule_narrow_defaults_off(
         "inline void flatxsCtaLaunch(const FlatXsView& v, int threads,\n"
         "                            cudaStream_t stream, bool narrow) {")),
    ("the two arms become two bodies instead of two instantiations",
     lambda: rule_one_body_for_both_arms(
         CTA.replace("template <int T, class POL, class WS>",
                     "template <int T, class POL>"))),
    ("a second copy of the CTA body appears",
     lambda: rule_one_body_for_both_arms(
         CTA + "\nflatxsSolveNodeCta(const FlatXsView& v, int i,\n")),
]
for label, probe in NEGATIVES:
    checks += 1
    try:
        fired = not probe()  # type: ignore[operator]
    except Exception:
        fired = True
    if not fired:
        failures.append("negative control did not fire: " + label)


py_compile.compile(os.path.abspath(__file__), doraise=True)

# ---------------------------------------------------------------------------
if failures:
    print("FAIL (%d of %d checks)" % (len(failures), checks))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("PASS  tools/test_gpu_fp32_contract.py  (%d checks, %d negative controls)"
      % (checks, len(NEGATIVES)))
print("  the NUMBERS are a 238 gate, not a source property: this arm is A2 and")
print("  is decided by Gate A against the FP64 trajectory and Gate B against")
print("  MASTER.  See section 6 of docs/WP20_GPU_FP32_20260831_KO.md.")
