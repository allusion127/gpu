#!/usr/bin/env python3
"""Static contract for WP20's device-wide single-precision arm.

RASBERY_GPU_FP32 narrows DEVICE HOT STATE to float.  It is gate class A2 -- it
moves the trajectory on purpose -- so the bit-golden gate cannot validate it and
the numbers are settled by Gate A (against the FP64 trajectory) and Gate B
(against MASTER).  What a static gate CAN settle is everything a reviewer would
otherwise have to take on trust, and that is this file:

  1. ONE ARM, ONE PREDICATE.  Four env knobs -- the three device-wide ones and
     CMFD's historical per-backend spelling -- each read once into a cached
     static IN THE ARM HEADER; every narrow launch site in the tree asks
     `routes()` and nothing else, so the captured graph topology, the kernels
     inside it and the receipt can never disagree about which precision this run
     is.  A knob spelled at a launch site instead is invisible to the receipt,
     which is exactly how `"cmfd":"fp64"` came to be printed over a run whose
     whole inner BiCGSTAB was float.
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
  6. THE LATCH HAS A CASE BOUNDARY, AND IT IS ON THE RECEIPT.  The latch is
     process-lifetime and the evaluator is one process for N cases, so "this
     worker ran FP32" is a claim about a PREFIX.  Every per-case receipt says
     which side of the latch its case fell on, a latched case does not publish
     a case_key an FP32 request could be served from, and the wave receipt
     names the boundary before a killed worker can lose it.

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
BACKEND_CODE = strip_comments(BACKEND)
FXK = read("src", "FlatXsKernel.h")
FXK_CODE = strip_comments(FXK)
XSRK = read("src", "XsReconKernel.h")
XSRK_CODE = strip_comments(XSRK)
NODK = read("src", "NodalKernel.h")
NODK_CODE = strip_comments(NODK)
CRAM = read("src", "CudaCramBackend.cu")
CRAM_H = read("src", "CudaCramBackend.h")
XSR_H = read("src", "CudaXsReconBackend.h")
XSSET = read("src", "XSSet.cpp")
XSSET_H = read("src", "XSSet.h")
BICG = read("src", "CudaBICGBackend.cu")
DRIVER = read("src", "Driver.h")
EVAL = read("src", "EvaluatorServer.h")
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
#
# RASBERY_GPU_CMFD_FP32 is on this list even though it is not one of "the three
# device-wide knobs".  It is the HISTORICAL per-backend spelling of the CMFD arm,
# it still arms the FP32 inner BiCGSTAB by itself, and while it was spelled in
# src/CudaBICGBackend.cu -- OR-ed against routes(Cmfd) at the gate -- nothing in
# the arm header could see it.  A run with RASBERY_GPU_CMFD_FP32=1 and
# RASBERY_GPU_FP32 unset therefore ran every kernel of the inner solve in float
# and printed `"arm":"fp64","backends":{"cmfd":"fp64",...}`.  Same rule, same
# reason: one reader, in the header that owns the policy.
KNOBS = {
    "RASBERY_GPU_FP32": "armed",
    "RASBERY_GPU_FP32_STRICT": "strict",
    "RASBERY_GPU_FP32_CRAM": "cramExtended",
    "RASBERY_GPU_CMFD_FP32": "cmfdLegacyArm",
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

# WP20.2 -- THE PPR EXTENSION IS A KNOB OF A DIFFERENT SHAPE, AND IT IS READ.
#
# RASBERY_GPU_FP32_PPR is deliberately NOT in kArmEnv (section 3: PPR runs after
# the statepoint's last SolveLoop and feeds nothing back, so two runs that
# differ only in it are the same arm), which is why it is not in KNOBS above --
# that loop demands case-key membership.  It is still a GATE, and it was once a
# gate in prose only: the flag was named by the commit that added the narrowing,
# by the doc's flag row, backend table and arm-D runbook, and by four comments
# in this tree -- and by no getenv anywhere.  `inScope()` special-cased CRAM and
# let every other converted backend through, so the narrowing engaged on
# RASBERY_GPU_FP32 alone, arm B and arm D were the same binary, and the receipt
# printed "ppr":"fp32" for an extension nobody had asked for.  So this block
# asserts the two halves the old rules did not: the READ, and the SCOPE test.
check(rule_gate_is_cached_opt_in(ARM, "RASBERY_GPU_FP32_PPR", "pprExtended"),
      "RASBERY_GPU_FP32_PPR is read once into a cached static by pprExtended(), "
      "opt-IN -- the same shape cramExtended() has, and for the same reason: "
      "the answer fixes a device allocation and may not move mid-run")
check(ARM_CODE.count('"RASBERY_GPU_FP32_PPR"') == 1,
      "and is spelled exactly once in the arm header -- a second reader is a "
      "second policy")


# EVERY FLAG THE DOC'S FLAG ROW NAMES IS READ BY THIS BINARY.
#
# The general form of the same defect, and the reason this rule is written over
# the DOC ROW rather than over a list kept here: a knob that reaches the doc,
# the runbook and the receipt without reaching a getenv cannot be caught by any
# rule that only asks "is the name absent from kArmEnv" or "is it mentioned in
# the doc" -- both of which passed while the gate was dead.  Add a flag to the
# row and it must be read, or this test fails before the campaign does.
FLAG_ROW = [ln for ln in DOC.splitlines() if ln.startswith("| 플래그 |")]
check(len(FLAG_ROW) == 1,
      "docs/WP20_GPU_FP32_20260831_KO.md carries exactly one flag row, which is "
      "the list this rule is stated over")
DOC_FLAGS = sorted(set(re.findall(r"RASBERY_[A-Z0-9_]+",
                                  FLAG_ROW[0] if FLAG_ROW else "")))
check(len(DOC_FLAGS) == 5,
      "the flag row names the arm and its four extensions, five in all -- got "
      "%s" % ", ".join(DOC_FLAGS))


def rule_flag_is_read(text: str, flag: str) -> bool:
    """A flag is READ when a gate passes its NAME to envFlagOn or getenv.  Called
    on STRIPPED source: a flag named only in a comment is exactly the defect."""
    return bool(re.search(r'(?:envFlagOn|std::getenv)\(\s*"%s"\s*\)' % re.escape(flag),
                          text))


for flag in DOC_FLAGS:
    check(rule_flag_is_read(ARM_CODE, flag),
          "%s is READ by src/GpuFp32Arm.h and not merely documented -- a knob "
          "the doc names, the runbook sets and no getenv reads is a flag every "
          "A/B silently attributes a number to for nothing" % flag)


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
check("armedFor(which)" in ROUTES and "inScope(which)" in ROUTES
      and "!latched(which)" in ROUTES,
      "routes() is (armed FOR THIS BACKEND AND in scope AND NOT latched) and "
      "nothing else")
# NEGATIVE CONTROL for the defect this rule exists for.  `armed()` alone is the
# DEVICE-WIDE knob; CMFD has a second, historical spelling.  A routing predicate
# that asks armed() cannot see it, which is how the CMFD launch site and the
# receipt came to disagree in the first place.
check("armed()" not in ROUTES,
      "routes() does NOT ask armed() directly -- a backend whose arm has a "
      "second spelling would route on one predicate and be reported on another")

try:
    ARMED_FOR = body_after(ARM, "inline bool armedFor(Backend which)")
except LookupError:
    ARMED_FOR = ""
check("armed()" in ARMED_FOR and "cmfdLegacyArm()" in ARMED_FOR
      and "Backend::Cmfd" in ARMED_FOR,
      "armedFor() folds BOTH spellings of the CMFD arm -- the device-wide "
      "RASBERY_GPU_FP32 and the historical RASBERY_GPU_CMFD_FP32 -- so the one "
      "predicate the launch sites and the receipt share knows about both")

try:
    ANY_ARMED = body_after(ARM, "inline bool anyArmed()")
except LookupError:
    ANY_ARMED = ""
check("armedFor(" in ANY_ARMED and "Backend::Count" in ANY_ARMED,
      "anyArmed() sweeps the whole enum through armedFor(), so a per-backend "
      "spelling added later cannot be omitted from the receipt's `arm` field")

try:
    SCOPE = body_after(ARM, "inline bool inScope(Backend which)")
except LookupError:
    SCOPE = ""


def rule_scope_gates_extensions(scope: str) -> bool:
    """Both FLAGGED backends are tested BY NAME in inScope().  A converted
    backend that falls through to `return true` is one the arm takes without
    being asked -- which is what `ppr` did until its flag was wired."""
    return ("converted(which)" in scope and "cramExtended()" in scope
            and "pprExtended()" in scope)


check(rule_scope_gates_extensions(SCOPE),
      "inScope() refuses a backend the tree does not narrow, and refuses CRAM "
      "and PPR unless RASBERY_GPU_FP32_CRAM / RASBERY_GPU_FP32_PPR extend the "
      "arm to them by name")

try:
    CONVERTED = body_after(ARM, "inline bool converted(Backend which)")
except LookupError:
    CONVERTED = ""
for name in ("Cmfd", "Nodal", "FlatXs", "Xe", "Cram", "Ppr"):
    check("Backend::%s:" % name in CONVERTED,
          "converted() answers for %s explicitly -- a backend that falls "
          "through to a default is a backend nobody decided about" % name)
check("case Backend::Cmfd:   return true;" in CONVERTED
      and "case Backend::FlatXs: return true;" in CONVERTED
      and "case Backend::Nodal:  return true;" in CONVERTED,
      "CMFD, flat-XS and nodal are the three backends the tree narrows "
      "(WP20 landed the first two, WP20.1 the blocks and nodal)")
check("case Backend::Cram:   return true;" in CONVERTED,
      "WP20.2 converted CRAM: the four-pole partial-fraction sum accumulates "
      "in float with a Neumaier compensation.  It is still gated behind "
      "RASBERY_GPU_FP32_CRAM by inScope(), so the arm reports `declined` and "
      "not `fp32` unless the extension is asked for explicitly")
check("case Backend::Ppr:    return true;" in CONVERTED,
      "WP20.2 converted PPR, and converted it AS A VRAM ITEM -- which is what "
      "the WP20 deferral sentence already said it would be")


# ---------------------------------------------------------------------------
# 2b. THE SECOND GATE INSIDE inScope() MUST BE LIVE, NOT SHADOWED.
#
# `inScope()` answers `!converted(which)` FIRST.  So a backend it gates a SECOND
# time -- today CRAM, behind RASBERY_GPU_FP32_CRAM -- is only really gated if
# converted() says TRUE for it.  Say `false` there instead and the extension
# test becomes dead text: inScope is false via the converted line, so
# `backendState()` returns `deferred` before it can ever return `declined` --
# for CRAM and therefore, CRAM being the only backend gated twice, for every
# backend -- while the knob goes on forking the WP10.1 case key and changing
# nothing the binary does.  Two runs that behave identically then cannot share a
# cached answer, and a reader looking for the documented `declined` never sees
# it in any run of any build.
#
# The rule is written on the STRUCTURE and not on today's backend list, so it
# holds for whichever backend is gated next.
# ---------------------------------------------------------------------------
def scope_gated_backends(arm: str) -> list[str]:
    """The backends inScope() tests a SECOND time, by name."""
    try:
        scope = body_after(arm, "inline bool inScope(Backend which)")
    except LookupError:
        return []
    return re.findall(r"which\s*==\s*Backend::(\w+)", scope)


def rule_scope_gate_is_live(arm: str) -> bool:
    """Every specially gated backend must be converted(), or its gate is dead
    code and `declined` is a word no run can print."""
    names = scope_gated_backends(arm)
    if not names:
        return False          # a gate that gates nothing can decline nothing
    try:
        conv = body_after(arm, "inline bool converted(Backend which)")
    except LookupError:
        return False
    for name in names:
        hit = re.search(r"case Backend::%s:\s*return\s+(true|false)\s*;" % name,
                        conv)
        if hit is None or hit.group(1) != "true":
            return False
    return True


check(rule_scope_gate_is_live(ARM),
      "every backend inScope() gates a second time is converted() == true -- "
      "otherwise the `!converted(which)` line above shadows the gate, the "
      "extension flag becomes unobservable, and `declined` becomes unreachable "
      "for all six backends")
check(scope_gated_backends(ARM) == ["Cram", "Ppr"],
      "and CRAM and PPR are the backends with a second gate, in that order -- "
      "which is what the receipt's `declined` word, the header's CRAM paragraph "
      "and the `ppr` row of the backend table all describe.  A third one added "
      "without a line here is a knob nobody pinned")
check(rule_scope_gate_is_live(ARM)
      and "rasbery::fp32::routes(rasbery::fp32::Backend::Cram)" in CRAM
      and "if (s.fp32_pole) kPredictor<true><<<" in CRAM,
      "RASBERY_GPU_FP32_CRAM is on kArmEnv, so it MUST be able to move the "
      "binary: the gate is live, the backend asks routes() once, and a "
      "different kernel instantiation launches because of it.  A knob that "
      "forks the case key without changing what runs makes two identical "
      "answers uncacheable and buys nothing for it")


# ---------------------------------------------------------------------------
# 2c. THE DEMOTION COUNTER IS A LIST OF CALL SITES, NOT A CLAIM IN PROSE.
#
# `demotions` counts SITES that asked for FP32 and ran FP64 anyway.  A backend
# that inScope() withholds by POLICY is NOT one of those: it is reported once,
# per backend, by backendState() as `declined`, and folding it into the counter
# would put `"demotions"` above zero on every correct default run -- the one
# number Sec 9.1 of the doc reads as G0.  The header used to claim the CRAM
# refusal was counted while `git grep noteDemotion` showed sites for FlatXs and
# Nodal only, so the header now carries the LIST and this rule compares it to
# the tree in BOTH directions: a claimed site that does not exist fails, and an
# existing site nobody declared fails too.
# ---------------------------------------------------------------------------
def src_corpus() -> str:
    """Every C/C++/CUDA source in src/, comments stripped.  A rule about where a
    counter is incremented has to look at the whole tree: one call site anywhere
    makes the claim true and its absence everywhere makes it false."""
    parts = []
    src_dir = os.path.join(ROOT, "src")
    for name in sorted(os.listdir(src_dir)):
        if name.endswith((".h", ".hh", ".hpp", ".cuh", ".c", ".cc", ".cpp", ".cu")):
            with open(os.path.join(src_dir, name), encoding="utf-8-sig",
                      errors="replace") as fh:
                parts.append(fh.read())
    return strip_comments("\n".join(parts))


SRC_ALL = src_corpus()


def declared_demotion_sites(arm: str) -> set:
    """The backends the header CLAIMS are counted, read off the list beside
    noteDemotion() rather than out of its prose."""
    at = arm.find("/// COUNTED AT")
    if at < 0:
        return set()
    end = arm.find("inline void noteDemotion(", at)
    if end < 0:
        return set()
    # Only the list ITEMS -- a `Backend::` that appears mid-sentence (in
    # `XsReconBackend::solveFlatXs`, say) is prose and not an entry.
    return set(re.findall(r"^///\s+Backend::(\w+)", arm[at:end], re.M))


def counted_demotion_sites(sources: str) -> set:
    return set(re.findall(r"noteDemotion\(rasbery::fp32::Backend::(\w+)\)",
                          sources))


def rule_demotion_claim_matches_call_sites(arm: str, sources: str) -> bool:
    declared = declared_demotion_sites(arm)
    return bool(declared) and declared == counted_demotion_sites(sources)


check(rule_demotion_claim_matches_call_sites(ARM, SRC_ALL),
      "the header's COUNTED AT list and the tree's noteDemotion() call sites "
      "are the same set -- a demotion claimed in prose but never incremented "
      "makes `demotions` unreadable as a G0, and an increment nobody declared "
      "makes a correct run look demoted")
check(declared_demotion_sites(ARM) == {"FlatXs", "Nodal", "Cram"},
      "and that set is {FlatXs, Nodal, Cram}: the flat-XS reference arm, the "
      "hybrid nodal drive and the batch arena, plus the CRAM policy refusal -- "
      "which IS counted, once per process, so that RASBERY_GPU_FP32_CRAM=1 and "
      "not asking are distinguishable in the receipt as well as in the case key")
check("Cram" in counted_demotion_sites(SRC_ALL),
      "and the CRAM policy refusal really is one of the call sites, which is "
      "what makes Sec 9.1's G0 baseline a function of RASBERY_GPU_CRAM rather "
      "than a flat zero")


# ---------------------------------------------------------------------------
# WP20.2 -- THE XE TRIPWIRE.
#
# Xe is the last `deferred` backend and its blocker is a MEASUREMENT CONTRACT
# rather than a numeric one.  Bits 0..4 of the shipped form mask are mined ON
# THE HOST, IN FP64, by comparing xe::xeDotChunk / xeCandidateOrdinal against
# src/XeAndersonReference.cpp BIT FOR BIT.  If somebody converts Xe and leaves
# the miner alone, `mineStable` goes on setting `sound = true` for a body the
# device no longer runs -- a silent false negative of exactly the class
# src/XeFormAudit.cpp exists to catch, and one that no amount of Gate A/Gate B
# would surface, because both compare ANSWERS and this is a claim about CODE.
#
# So the rule is written CONDITIONALLY on the table, not on today's answer.  As
# long as Xe is deferred it holds the deferral to stating its own reason; the
# moment the table says otherwise it demands that the miner have been told, in
# one of the two ways that are honest:
#
#   (a) mine and score the FP32 forms SEPARATELY, so `sound` is a verdict on
#       the instantiation the device runs; or
#   (b) report `sound = false` with the reason `fp32_arm` for that arm and skip
#       the mask-dependence, which is what `algebra_sound` already does for the
#       WP7-C channel: its own verdict, its own line, no effect on the value.
#
# What is NOT acceptable is the third option, and the rule exists to make it
# fail loudly: leave the FP64 verdict in place and let it be read as covering
# the device.  XE_HOST_FORMS -- the host Anderson algebra -- stays FP64 either
# way, so its verdict is untouched by any of this.
MINER = read("src", "XeFormMiner.cpp")
MINE  = read("src", "XeFormMine.h")
XE_CONVERTED = "case Backend::Xe:     return true;" in CONVERTED
if XE_CONVERTED:
    armed_separately = ("scoreShippedMaskF32" in MINE or "scoreShippedMask32" in MINE
                        or "fp32" in strip_comments(MINE).lower())
    armed_declared = ('"fp32_arm"' in MINER
                      and "rasbery::fp32::routes(rasbery::fp32::Backend::Xe)" in MINER)
    check(armed_separately or armed_declared,
          "Xe is CONVERTED and the miner was not told.  Bits 0..4 are mined "
          "against the FP64 bodies bit-for-bit, so `sound = true` would now "
          "describe arithmetic the device does not perform.  Either mine the "
          "FP32 forms separately, or report sound=false with the reason "
          '"fp32_arm" for that arm and skip the mask-dependence -- the way '
          "algebra_sound already carries its own verdict without touching the "
          "shipped value.  Do not leave the FP64 verdict to be read as "
          "covering the device.")
    check("XE_HOST_FORMS" in MINER or "xeHostFormMask" in MINER,
          "and XE_HOST_FORMS -- the host Anderson algebra -- stays FP64 and "
          "keeps its own verdict whatever the device arm does")
else:
    check("case Backend::Xe:     return false;" in CONVERTED,
          "xe is declared DEFERRED in the table rather than left to look "
          "converted")
    check("XeTripleConst" in ARM and "sound = true" in ARM,
          "and the deferral states its own reason IN THE TABLE: that the "
          "blocker is the miner's bit-for-bit contract against the FP64 "
          "bodies, not a numeric objection.  A deferral without its reason is "
          "the one a later package deletes as an oversight")
    check("bits 0..4 of the shipped form mask" in ARM.lower(),
          "and names the bits it is about")

# WP20.2 -- PPR: THE TWO ARRAYS MASTER MODE OWNS, AND NOT ONE MORE.
#
# PPR is downstream of the statepoint, so this arm cannot buy throughput and
# does not claim to.  What it can get wrong is SCOPE: `phic` and `partials` are
# shared with the SENM arm, which is B0 against the host, and `partials` carries
# a host-pinned chunk association -- narrowing either would put a bit-exactness
# claim behind a rounding flag.  So the rules here are about which arrays the
# arm may touch, and about the one place a corner value crosses into memory.
PPR = read("src", "CudaPprBackend.cu")
PPR_CODE = strip_comments(PPR)
check(PPR_CODE.count("rasbery::fp32::routes(") == 1
      and "rasbery::fp32::routes(rasbery::fp32::Backend::Ppr)" in PPR,
      "the PPR backend asks the arm exactly once")
GATE = body_after(PPR, "inline bool pprNarrowCorner()")
check("static const bool on" in GATE,
      "and caches it: the answer fixes a device allocation and a field of the "
      "captured graph key, so it may not move between two statepoints")
for name in ("float*  phic_next_f;", "float*  mrel_f;", "int     narrow_corner;"):
    check(name in PPR, "DevCtx declares %s" % name)
check("x.narrow_corner = (s.d_phic_next_f != nullptr) ? 1 : 0;" in PPR,
      "the ctx stamps the flag from the BLOCK IT BOUND, so it can never "
      "describe a width its pointers do not have")
check("narrow_corner ? (void**)&d_phic_next_f : (void**)&d_phic_next" in PPR,
      "ONE of the two rows is allocated, never both -- the wide pointer stays "
      "null on the narrow arm so a body that skipped the accessors faults "
      "instead of reading some other allocation and computing something "
      "finite and plausible")
check("phicNextPtr" not in PPR_CODE,
      "the raw phic_next pointer accessor is GONE rather than kept-and-unused: "
      "one that returned the wide array on the narrow arm is a null "
      "dereference waiting for one more caller")
for narrowed, why in (
        ("phic_next", "the CPB Jacobi's next iterate"),
        ("mrel", "the per-(node, group) relative-change scratch")):
    check("x.%s_f[i]" % narrowed in PPR,
          "%s (%s) has a narrow twin" % (narrowed, why))
for wide, why in (
        ("double* phic;", "SHARED with the SENM arm, which is B0 against the host"),
        ("double* partials;", "SHARED, and its 256-chunk association is host-pinned"),
        ("double* c;", "the interpolant the reconstruction consumes")):
    check(wide in PPR, "%s stays FP64 -- %s" % (wide, why))
check("double* pin_power" in read("src", "PprReconstructionKernel.cuh"),
      "pin_power is the answer Gate B measures and leaves the device as double")
STORE = body_after(PPR, "inline void pprPhicNextStore(")
check("static_cast<float>(value)" in STORE,
      "the rounding to the arm's width happens at the ONE place a corner value "
      "crosses into memory")
check('"RASBERY_GPU_FP32_PPR"' not in DRIVER,
      "RASBERY_GPU_FP32_PPR is NOT in kArmEnv, on the same footing "
      "RASBERY_GPU_PPR is not: PPR runs after the statepoint's last SolveLoop "
      "and feeds nothing back, so two runs that differ only in it are the same "
      "arm and must compare as one")
check("RASBERY_GPU_FP32_PPR is deliberately ABSENT" in DRIVER,
      "and the absence is EXPLAINED where the list is, because an unexplained "
      "absence reads as an oversight and gets fixed")

# WP20.2 -- CRAM: EXACTLY ONE ACCUMULATOR NARROWS, AND EVERY OTHER DOUBLE IN
# THAT FILE HAS A NUMBER BEHIND IT.
#
# WP20 deferred CRAM with a reason -- "the partial-fraction terms alternate in
# sign and cancel catastrophically" -- and an arm that narrowed the SOLVE would
# not be testing that reason, it would be replacing it with a worse one.  So
# what this section holds is the SCOPE: the pole sum and nothing else, with the
# compensation that makes the cancellation question a fair one.
CRAM_CODE = strip_comments(CRAM)
check(CRAM_CODE.count("rasbery::fp32::routes(") == 1
      and "rasbery::fp32::routes(rasbery::fp32::Backend::Cram)" in CRAM,
      "the CRAM backend asks the arm exactly once, through routes(), which is "
      "what folds RASBERY_GPU_FP32 AND RASBERY_GPU_FP32_CRAM into one answer")
check('#include "GpuFp32Arm.h"' in CRAM, "src/CudaCramBackend.cu includes the arm header")
check("_impl->fp32_pole = " in CRAM,
      "and stores it on the Impl rather than in a function-local static -- that "
      "TU has no process state by contract, and a width that could change "
      "between a statepoint's predictor and its corrector would be two "
      "arithmetics inside one depletion step")
FOLD = body_after(CRAM, "__device__ inline void cramFoldPole(")
check("*comp +=" in FOLD and "*acc = t;" in FOLD,
      "cramFoldPole carries the bits the add could not hold instead of "
      "discarding them")
FINISH = body_after(CRAM, "__device__ inline unsigned int cramFinish(")
check("kAlpha0 * iden[row]" in FINISH and "static_cast<float>" not in FINISH,
      "the alpha_0 term stays DOUBLE on both arms: at 1.17e-08 x iden[row] it "
      "is ~8 decades below its addend and would round to nothing in float, "
      "which makes it the term with the least headroom in the expression")
check("1.0e-13" in CRAM,
      "kRelTol is still 1.0e-13 -- six decades below float eps -- which is the "
      "proof that the Gauss-Seidel solve may not narrow: a float solve could "
      "never satisfy the break test and would fail open on every node")
for kern in ("kPredictor", "kCorrector", "kPredictorP4", "kCorrectorP4"):
    check("template <bool FP32>\n__global__ void %s(DevCtx x)" % kern in CRAM,
          "%s is templated on the width, so the two precisions are two "
          "instantiations of one text" % kern)
check(CRAM_CODE.count("<true><<<") == 4 and CRAM_CODE.count("<false><<<") == 4,
      "four launches per width and no more: the mapping (serial/pole4) and the "
      "width are independent, and neither may be read inside a kernel")
check("const char* CramBackend::poleSumPrecision() const" in CRAM
      and '"fp32" : "fp64"' in CRAM,
      "the backend reports the width as a WORD, because a receipt that said "
      "`true` would leave a reader to guess which of that file's many doubles "
      "it was true about")
check('\\"pole_sum\\":\\"{}\\"' in DRIVER and "c.poleSumPrecision()" in DRIVER,
      "and the [RASBERY][CRAM_GPU] receipt prints it")


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

# THE RECEIPT ASKS THE PREDICATE THE LAUNCH SITE ASKS.  This is the rule the
# cmfd:"fp64"-over-a-float-solve defect violated: backendState() short-circuited
# on the device-wide knob while the CMFD gate was (that knob OR the historical
# RASBERY_GPU_CMFD_FP32), so the receipt was wrong in exactly the configuration
# the header cites as worth +2.6 % on M64 -- and could print
# "nonfinite_fallbacks":1 beside an arm it said was never engaged.
check("armedFor(which)" in STATE,
      "backendState() gates on armedFor(which) -- the same predicate routes() "
      "and therefore every launch site asks")
check("if (!armed())" not in STATE,
      "NEGATIVE CONTROL: backendState() must NOT short-circuit on the "
      "device-wide armed() knob; that is what reported cmfd:\"fp64\" over a "
      "run whose whole inner BiCGSTAB was float")
check('(armed() ? "fp32" : "fp64")' not in RECEIPT,
      "NEGATIVE CONTROL: the receipt's `arm` field is not a bare armed() "
      "ternary -- with RASBERY_GPU_CMFD_FP32 set and RASBERY_GPU_FP32 unset "
      "that printed \"fp64\" over a float CMFD solve")
check("anyArmed()" in RECEIPT and '"partial"' in RECEIPT,
      "`arm` is three-valued: fp32 (device-wide), partial (a per-backend "
      "spelling only), fp64 (nothing) -- so it can never contradict `backends`")

# ---------------------------------------------------------------------------
# 3b. `declined` IS A STATE A RUN CAN REACH, AND THE REFUSAL IS COUNTED.
#
# Two failures hide behind one word here, and both make RASBERY_GPU_FP32_CRAM=1
# INDISTINGUISHABLE from not asking -- while the two runs already carry two
# different WP10.1 case keys and therefore cannot share a cache.
#
#   (a) REACHABILITY.  backendState() tests `!converted` -> "deferred" before it
#       tests `!inScope` -> "declined".  If NO backend is converted-but-gated,
#       the second test can never fire: `declined` becomes dead text and the
#       inScope() branch that spells the gate becomes dead code.  So the rule is
#       not "Cram is gated" -- it is the STRUCTURE: some backend converted()
#       answers TRUE for is then gated by inScope() on a knob of its own.
#   (b) THE COUNT.  src/GpuFp32Arm.h promises that the default refusal is
#       COUNTED as a demotion "so the receipt says the arm was asked and
#       declined rather than silently doing nothing".  A promise with no
#       noteDemotion() behind it is the exact failure mode the receipt exists
#       to prevent, one level up: the operator reads `demotions` and cannot
#       tell an arm that declined from an arm nobody asked.
#
# The count is taken ONCE, in the constructor that resolves the width, and only
# when the arm was actually armed -- a run without RASBERY_GPU_FP32 must not
# count anything, which is the same feature-off rule the rest of this file
# holds.
# ---------------------------------------------------------------------------
def rule_declined_is_reachable(arm: str) -> bool:
    """Some backend converted() answers TRUE for is gated again by inScope(),
    and backendState() still asks !inScope after !converted."""
    try:
        conv  = body_after(arm, "inline bool converted(Backend which)")
        scope = body_after(arm, "inline bool inScope(Backend which)")
        state = body_after(arm, "inline const char* backendState(Backend which)")
    except LookupError:
        return False
    gated = re.findall(r"which == Backend::(\w+)\)\s*return\s+\w+\(\);", scope)
    if not gated:
        return False
    if not any(re.search(r"case Backend::%s:\s*return true;" % g, conv)
               for g in gated):
        return False
    return '!inScope(which)' in state and 'return "declined"' in state


def rule_declined_is_counted(cram: str) -> bool:
    """The site that resolves the CRAM width counts the refusal, gated on the
    ARM and not on the extension: armed-and-not-routed is a demotion."""
    try:
        ctor = body_after(cram, "CramBackend::CramBackend()")
    except LookupError:
        return False
    if "rasbery::fp32::noteDemotion(rasbery::fp32::Backend::Cram);" not in ctor:
        return False
    return re.search(r"rasbery::fp32::armed\(\)", ctor) is not None


check(rule_declined_is_reachable(ARM),
      "`declined` is a state a run can actually reach: converted() says TRUE "
      "for a backend that inScope() then gates on its own knob, and "
      "backendState() still asks !inScope after !converted.  A word no run can "
      "print is not a receipt state")
check(rule_declined_is_counted(CRAM),
      "and the refusal is COUNTED: CramBackend::CramBackend() calls "
      "noteDemotion(Backend::Cram) when the arm is armed and routes() came "
      "back false, which is what src/GpuFp32Arm.h promises and what makes "
      "RASBERY_GPU_FP32_CRAM=1 distinguishable from not asking")
check("RASBERY_GPU_FP32_CRAM" not in strip_comments(CRAM),
      "the count reads the ARM through the header and never the extension "
      "environment variable itself -- one reader per knob, which is what keeps "
      "a run with RASBERY_GPU_FP32 unset counting nothing at all")
check("COUNTED as a demotion" in ARM and "CramBackend::CramBackend()" in ARM,
      "src/GpuFp32Arm.h names the site its promise is kept at, so the promise "
      "and the call cannot drift apart without this rule seeing it")

def rule_declined_is_reachable_ordered(state: str, arm: str) -> bool:
    """`declined` is the receipt's word for "this tree HAS the narrow path and
    withheld it".  Naming the word is not enough: it has to be TESTED, on
    !inScope, between `deferred` and `fp32`, AND some backend has to be able to
    reach it -- which is only true while inScope()'s second gate is live."""
    for word in ('"deferred"', '"declined"', '"fp32"'):
        if ("return %s" % word) not in state:
            return False
    if "!inScope(which)" not in state:
        return False
    if not (state.index('return "deferred"') < state.index('return "declined"')
            < state.index('return "fp32"')):
        return False
    return rule_scope_gate_is_live(arm)


check(rule_declined_is_reachable_ordered(STATE, ARM),
      "`declined` is REACHABLE and not merely spelled: it is tested on "
      "!inScope(which) after `deferred` and before `fp32`, and the backend that "
      "second gate names is converted().  A run with RASBERY_GPU_FP32=1 and no "
      "RASBERY_GPU_FP32_CRAM must therefore print `\"cram\":\"declined\"` and "
      "not `\"cram\":\"deferred\"`")

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
# The arm does not add a second CMFD precision path; it routes into the one that
# already exists, at the one cached gate, so there is still one captured
# topology decision.  The historical RASBERY_GPU_CMFD_FP32 spelling is folded in
# at armedFor() in the ARM HEADER and not OR-ed here, because a gate the receipt
# cannot see is a gate that lets `"cmfd":"fp64"` be printed over a float solve.
# tools/test_cmfd_fp32_contract.py owns the kernel-level properties of that
# path; what belongs HERE is the join.
# ---------------------------------------------------------------------------
try:
    CMFD_GATE = body_after(BICG, "bool cmfdFp32InnerEnabled()")
except LookupError:
    CMFD_GATE = ""
check("rasbery::fp32::routes(rasbery::fp32::Backend::Cmfd)" in CMFD_GATE,
      "RASBERY_GPU_FP32 routes CMFD through the existing cached inner-solve gate")
check("static const bool enabled" in CMFD_GATE,
      "the joined gate is still resolved once and cached")
# NEGATIVE CONTROL: no second predicate at the gate.  The OR that used to sit
# here is the defect -- it armed the FP32 kernel set from a knob `armed()` never
# saw, so the receipt reported an FP64 device run that did not happen.
check("envFlagEnabled(" not in CMFD_GATE and "getenv" not in CMFD_GATE
      and "||" not in CMFD_GATE,
      "the CMFD gate reads NO environment of its own and ORs nothing: it is "
      "routes(Cmfd) and nothing else, so the receipt reads the same predicate")
check(BACKEND_CODE.count("rasbery::fp32::routes(") == 2,
      "the xsrecon/nodal TU asks the arm exactly twice -- once in "
      "flatxsNarrowBlocks(), once in nodalNarrowState() -- and both cache it")
for fn, knob in (("flatxsNarrowBlocks", "Backend::FlatXs"),
                 ("nodalNarrowState", "Backend::Nodal")):
    body = body_after(BACKEND, "inline bool %s()" % fn)
    check("static const bool on" in body and knob in body,
          "%s() resolves routes(%s) ONCE into a cached static: the width fixes "
          "a device allocation and a captured graph topology, so it may not "
          "move between two calls of a run" % (fn, knob))
check("rasberyGpuNodalFullEnabled()" in body_after(BACKEND, "inline bool nodalNarrowState()"),
      "the narrow nodal drive requires the FULL arm -- the hybrid drive ships "
      "trlcff/matM back to FP64 host arrays mid-drive, so it stays wide and is "
      "counted as a demotion")

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


# 5b. THE WORKSPACE IS THE STATE; `v.iden` IS THE ANSWER.  THE ANSWER IS NEVER
#     WRITTEN FROM THE STATE.
#
# The arm's whole claim is that it narrows the shared-memory WORKSPACE and
# leaves the outputs alone -- section 4 of the doc says `xs`/`xs_ssm`/`iden`
# stay double, and the struct rules above are what hold the workspace half of
# it.  This is the other half, and it is not implied by them: `w` and `v.iden`
# are BOTH in scope inside RefreshLightIsotopes, so `v.iden[..] = w.iden[..]`
# type-checks, reads as a copy, and is a float round-trip on this arm.
#
# It is worth saying WHICH rounding, because the arm already declares one.  The
# A2 analysis in src/FlatXsCtaKernel.cuh is about the delta-stream accumulator
# `dst[e] = scale*val + dst[e]` re-rounding to float per entry.  These three
# rows are pure products with no delta stream at all, they are not in that
# analysis, and the array they land in is the FP64 authority CRAM reads as
# `x.iden_in` and XSSet materialises on the host.  The B-10 row is the quantity
# the critical-boron search drives and the quantity CRAM depletes.  So the
# shape is fixed: form the row in a NAMED DOUBLE, store that double to the
# authority, then convert the same local into the workspace for the macro fold.
# Both refresh sites obey it -- the untiled flatxsSolveNodeCta and the WP21-B2
# tiled flatxsSolveTileCta -- which is what the counts of 2 are.
IDEN_ROWS = (("IH1", "nden_h1"), ("IB10", "nden_b10"), ("IO16", "nden_o16"))


def rule_iden_authority_is_published_wide(cta: str) -> bool:
    flat = re.sub(r"\s+", " ", strip_comments(cta))
    # (a) no row of the authority may be written FROM the workspace, in either
    #     body's spelling (`w.iden[..]` untiled, `w[j].iden[..]` tiled).
    if re.search(r"v\.iden\s*\[[^\]]*\]\s*=\s*w\s*(\[[^\]]*\]\s*)?\.\s*iden\s*\[",
                 flat) is not None:
        return False
    for row, local in IDEN_ROWS:
        # (b) the row is formed in a double local, once per body.
        if flat.count("const double %s =" % local) != 2:
            return False
        # (c) and THAT local is what the authority gets, once per body.
        if flat.count("v.iden[%s * nxyz + l] = %s;" % (row, local)) != 2:
            return False
        # (d) and the workspace gets the SAME local, so the two cannot drift
        #     into two different expressions of the same density.
        if flat.count("w.iden[%s] = %s;" % (row, local)) != 1:
            return False
        if flat.count("w[j].iden[%s] = %s;" % (row, local)) != 1:
            return False
    return True


check(rule_iden_authority_is_published_wide(CTA),
      "both RefreshLightIsotopes sites in src/FlatXsCtaKernel.cuh publish H-1, "
      "B-10 and O-16 to `v.iden` from a DOUBLE local and only then convert into "
      "the workspace.  `v.iden[..] = w.iden[..]` would route the authoritative "
      "FP64 densities through the float workspace -- silently, with no receipt "
      "field, no finite check and no fallback -- and the B-10 row is what the "
      "critical-boron search drives and what CRAM depletes")
check(re.search(r"double\s+iden\[NISO\];", FXK) is not None
      and re.search(r"v\.iden\[IB10 \* nxyz \+ l\]\s*=\s*iden\[IB10\];", FXK)
          is not None,
      "and the thread-per-node reference still publishes the same three rows "
      "from its own `double iden[NISO]` -- if IT narrowed, the CTA arm would be "
      "matching the wrong thing")


def rule_narrow_defaults_off(cta: str) -> bool:
    """Every caller that predates the arm -- the replay gate included -- must
    keep launching exactly the kernel it launched before.

    WP21-B2 appended a second optional parameter (the node tile) on the same
    terms, so the defaults are checked TOGETHER: `narrow = false` selects the
    FP64 workspace and `tile = 1` selects the untiled body, which between them
    are the pre-WP20 / pre-WP21-B2 launch exactly."""
    return "bool narrow = false, int tile = 1)" in cta


check(rule_narrow_defaults_off(CTA),
      "flatxsCtaLaunch's precision parameter DEFAULTS TO FALSE and its tile "
      "parameter DEFAULTS TO 1, which is what the feature-off byte-identity "
      "claim rests on")

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
try:
    SOLVE_NODAL_BINDS = body_after(BACKEND, "bool XsReconBackend::solveNodal(")
except LookupError:
    SOLVE_NODAL_BINDS = ""
check(strip_comments(SOLVE).count("rasbery::fp32::routes(") == 0
      and "const bool narrow = narrow_blocks;" in SOLVE,
      "solveFlatXs no longer asks the arm itself: WP20.1 moved the question to "
      "flatxsNarrowBlocks(), because the answer now fixes an ALLOCATION made "
      "in ensure() and a per-call read could disagree with the block that "
      "exists")
check("fxs::flatxsCtaLaunch(v, cta_threads, d.stream, narrow, tile);" in SOLVE
      and "const bool narrow = narrow_blocks;" in SOLVE,
      "the precision travels with the launch rather than through a global "
      "(WP21-B2 put the node tile on the same argument list for the same "
      "reason: both are compile-time properties of the kernel that is about to "
      "run, so both have to be decided before the launch and neither may wobble "
      "between two calls of one run)")
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
check(strip_comments(BICG).count("rasbery::fp32::noteBytesSaved(") == 1,
      "CMFD accounts its footprint delta exactly once")
check(BACKEND_CODE.count("rasbery::fp32::noteBytesSaved(") == 6,
      "the xsrecon/nodal TU accounts SIX deltas and no more: the CTA "
      "workspace, the live micx H2D, the micx D2H, the reference-block H2D, "
      "the nodal consts H2D and the nodal chif H2D.  One per conversion site, "
      "so a byte is counted once and only where it was really not moved")
for site, what in (
        ("uploadMicx", "the live micx/lmpx H2D"),
        ("downloadMicx", "the micx/lmpx D2H"),
):
    body = body_after(BACKEND, "bool %s(const char* leaf" % site)
    check("noteBytesSaved(count * (sizeof(double) - sizeof(float)))" in body,
          "%s counts the delta from the SAME count it copied, so the receipt "
          "and the ledger cannot disagree" % what)
    check("count * sizeof(float)" in body,
          "%s hands the ledger the FLOAT byte count -- a copy quoted in "
          "doubles would hide exactly the saving this arm exists for" % what)
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
# 8b. WP20.2 -- THE REFINEMENT LOOP AND THE ARM THAT MEASURES THE ACCUMULATOR.
#
# WP20 left two claims resting on prose.  The first was that the FP32 arm's
# extra outers were a CONVERGENCE-CRITERION effect and not a kernel defect; the
# second was that the double accumulator in the FP32 dots is what keeps the
# outer counts comparable.  WP20.2 turns both into arms, so what belongs here
# is that each arm is (a) one cached answer, (b) OFF-identical, and (c) able to
# be read off a receipt afterwards.
#
# tools/test_cmfd_fp32_contract.py owns the STRUCTURE of the refinement loop
# inside enqueue_outer; what belongs HERE is the knob and the receipt.
# ---------------------------------------------------------------------------
try:
    ROUNDS = body_after(ARM, "inline int refineRounds()")
except LookupError:
    ROUNDS = ""
check('std::getenv("RASBERY_GPU_FP32_REFINE")' in ROUNDS,
      "the refinement cap is read from RASBERY_GPU_FP32_REFINE")
check("static const int rounds" in ROUNDS,
      "and read ONCE into a cached static: the cap fixes the captured graph "
      "DEPTH, so it may not move between two outers of a run")
check("return armed() ? rounds : 1;" in ROUNDS,
      "with the arm off the cap is 1 whatever the variable says -- a "
      "refinement round is a round of an FP32 solve and there is no FP32 solve")
check("kRefineRoundsDefault" in ROUNDS and "kRefineRoundsMax" in ROUNDS,
      "the default and the clamp are named constants, not literals buried in "
      "the parser")
check(ARM_CODE.count('"RASBERY_GPU_FP32_REFINE"') == 1,
      "RASBERY_GPU_FP32_REFINE is spelled exactly once in the arm header")
check('"RASBERY_GPU_FP32_REFINE",' in DRIVER,
      "RASBERY_GPU_FP32_REFINE is in kArmEnv -- it changes HOW MANY TIMES the "
      "inner solve runs and what the outer accepts, which is a sharper claim "
      "on that list than a rounding knob")
check('return 1;' in ROUNDS,
      "the OFF answer is 1 and never 0: one round IS the WP20 topology")
check("refine" in ARM_CODE and 'out << ",\\"refine\\":" << refineRounds();' in ARM,
      "the receipt carries the cap, so an A/B can be attributed to a round "
      "count that was really captured")

try:
    STRICT = body_after(ARM, "inline bool strictActive()")
except LookupError:
    STRICT = ""
check("armed()" in STRICT and "strict()" in STRICT,
      "strictActive() is (the arm AND the knob): narrowing the accumulator of "
      "an FP64 solve would just be an FP64 solve with a worse dot product")
check("strictActive()" in RECEIPT.replace(" ", "") or "strictActive()" in ARM,
      "the receipt reports the ARM and not the bare knob, so the receipt and "
      "the launch site cannot disagree about which width was measured")

BICG_CODE = strip_comments(BICG)
for kernel in ("reduce_dot_stage1_f32_strict", "reduce_dot2_stage1_f32_strict",
               "reduce_dot_stage2_strict", "reduce_dot2_stage2_strict"):
    body = body_after(BICG, "__global__ void %s(" % kernel)
    check("__shared__ double" not in body and "double sum" not in body,
          "%s has NO wide accumulator anywhere in it -- that is the whole "
          "point of the arm" % kernel)
for stage1 in ("reduce_dot_stage1_f32_strict", "reduce_dot2_stage1_f32_strict"):
    check("__shared__ float" in body_after(BICG, "__global__ void %s(" % stage1),
          "%s accumulates in float shared memory" % stage1)
STRICT_S2 = body_after(BICG, "__global__ void reduce_dot_stage2_strict(")
check("float sum = 0.0f;" in STRICT_S2 and "sqrtf(sum)" in STRICT_S2,
      "the strict stage-2 fold and its square root are float too; a float "
      "stage 1 folded in double would leave half the accumulation wide")
check("for (int i = 0; i < blocks; ++i) sum += pm[i];" in STRICT_S2,
      "and the fold keeps the strict index order the wide one has, so the arm "
      "changes the WIDTH of the reduction and nothing about its shape")

check(BICG_CODE.count("rasbery::fp32::strictActive()") == 1,
      "the CMFD backend asks the strict arm exactly once and caches it")
STRICT_GATE = "fp32_strict     = fp32_inner && rasbery::fp32::strictActive();"
check(STRICT_GATE in BICG,
      "and it is ANDed with the inner-solve gate rather than read as a second "
      "arm of its own")
PREC = body_after(BICG, "int precisionTag() const")
check("fp32_strict" in PREC and "refineRoundsActive()" in PREC,
      "precisionTag() folds BOTH the round count and the strict kernel set: a "
      "capture that differs in either is a different topology and may not "
      "share an instantiation")
for name in ("partials_f", "partials2_f"):
    check("float*        %s" % name in BICG,
          "%s is a float buffer" % name)
check("if (fp32_strict) {" in body_after(BICG, "if (fp32_inner) {"),
      "the narrow partials are allocated inside the FP32 block and gated on "
      "the strict arm again, so the default configuration pays neither")

for scalar, site in (("beta", "prepare_p_jacobi_f32"),
                     ("alpha", "update_s_jacobi_f32"),
                     ("omega", "update_solution_f32")):
    body = body_after(BICG, "__global__ void %s(" % site)
    check("strict_acc != 0" in body,
          "%s narrows %s at the ONE site that forms it" % (site, scalar))
    at = BICG.index("__global__ void %s(" % site)
    signature = BICG[at:BICG.index("{", at)]
    check("const int strict_acc," in signature,
          "%s takes the arm as a launch PARAMETER rather than reading a "
          "global: it is a capture-time constant of the graph it is in" % site)

check("fabs(denom) < 1.0e-30" in body_after(BICG, "__global__ void prepare_p_jacobi_f32("),
      "the breakdown test keeps its FP64 form and its FP64 threshold on BOTH "
      "arms -- a test that moved with the arm would be measuring the test")
check("fabs(r0v) < 1.0e-10" in body_after(BICG, "__global__ void update_s_jacobi_f32("),
      "and so does the r0.v guard")

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

# WP20.1: the doc carries the NUMBERS, because "it should be faster" is not a
# runbook.  Each of these is a figure a reader is expected to check a 238 run
# against, so a doc that lost one has lost the ability to falsify the arm.
for token in ("8.7 GB", "4.35 GB", "1 GB/케이스", "3.9 GB", "1.96 GB",
              "70.3 MB", "35.2 MB", "59.5 MB", "29.7 MB", "64.9 MB", "66.8 MB",
              "8,789,040", "7,436,880", "50,706",
              "RASBERY_GPU_MICX_RESIDENT", "kCramWidenMic", "NodalGraphKey",
              "drainMicxWiden", "narrow_blocks", "NodalViewT",
              "RASBERY_XFER_LEDGER", "d2h_bytes", "nodal_const_mb",
              "bytes_saved_est", "vram_mb", "HostPinRegistry"):
    check(token in DOC,
          "docs/WP20_GPU_FP32_20260831_KO.md names %s -- WP20.1's savings and "
          "runbook are numbers a reader can check, not a claim" % token)
check('"nodal":"fp32"' in DOC,
      "the doc shows the receipt shape WP20.1 actually produces")
check('"demotions":0' in DOC,
      "and says that a NON-zero demotion count is the G0 failure, not a detail")
for deferred in ("nodal", "xe", "cram", "ppr"):
    check(deferred in DOC.lower(),
          "the doc says what happened to %s" % deferred)

# WP20.2: the doc carries the DIAGNOSIS, the DESIGN and a runbook PER ARM.
#
# The arms are now five and they do not share a runbook: a single procedure
# would leave a reader unable to attribute a number to a flag, which is the same
# failure the per-backend receipt exists to prevent one level down.
for token, what in (
        ("RASBERY_GPU_FP32_REFINE", "the refinement knob"),
        ("RASBERY_GPU_FP32_PPR", "the PPR knob"),
        ("4,502", "the outer count the FP32 arm actually produced"),
        ("10.40", "the wall it produced"),
        ("9.75", "the FP64 wall it has to beat"),
        ("1e-7", "the FP32 inner solve's attainable residual, which is the "
                 "diagnosis"),
        ("refine_round_open", "the round opener"),
        ("refine_round_test", "the round test"),
        ("fp32_refine_rounds", "the receipt field the mean is computed from"),
        ("fp32_refine_solves", "and its denominator"),
        ("search_residual", "the transport, named rather than left to be found"),
        ("kCornerFluxTolerance", "the PPR break test that PERMITS its arm"),
        ("kRelTol", "the CRAM break test that FORBIDS its solve"),
        ("Neumaier", "the compensation, named -- plain Kahan would be a "
                     "different claim"),
        ("kAlpha0", "the CRAM term that stays double"),
        ("pole_sum", "the CRAM receipt field"),
        ("phic_next", "the PPR array that narrows"),
        ("reduce_dot_stage1_f32_strict", "the STRICT kernel"),
        ("bicg_restarts", "the counter STRICT is read on"),
        ("8×M16", "the batch reference"),
):
    check(token in DOC,
          "docs/WP20_GPU_FP32_20260831_KO.md names %s (%s)" % (token, what))
for arm in ("9.1 arm A", "9.2 arm B", "9.3 arm C", "9.4 arm D", "9.5 arm E"):
    check(arm in DOC,
          "the doc carries a SEPARATE 238 runbook for %s -- one procedure for "
          "five arms cannot attribute a number to a flag" % arm)
check("채택하지 않는다" in DOC,
      "and says outright that STRICT is not an arm to adopt: it is measured to "
      "settle a claim, not to be shipped")



# ---------------------------------------------------------------------------
# 10. WP20.1 -- THE micx/lmpx BLOCKS: ONE SPELLING PER READ, ONE PER WRITE.
#
# The blocks have four device readers (the two flat-XS kernels, the xsrecon
# condense/reconstruct arm, the Xe evaluate/commit pair) and one device writer.
# If ANY of them names `v.mic` / `v.lmp` / `v.ref_mic` / `v.msm` directly, it
# reads the WIDE pointer -- which on the narrow arm is null, and in the shared
# `dev_block` layout the wide offsets point at the MACROSCOPIC region.  Finite,
# plausible, and not the cross-sections.  So the rule is that the accessors are
# the only spelling, and it is checked on the stripped source of all three
# bodies.
# ---------------------------------------------------------------------------
FXS_LOADS  = ("fxsRefLmp", "fxsRefLsm", "fxsRefMic", "fxsRefMsm")
FXS_STORES = ("fxsStoreLmp", "fxsStoreLsm", "fxsStoreMic", "fxsStoreMsm")
for name in FXS_LOADS + FXS_STORES:
    check(("%s(const FlatXsView& v" % name) in FXK,
          "src/FlatXsKernel.h defines the %s accessor" % name)
for name in FXS_LOADS:
    check("v.narrow_blocks ?" in body_after(FXK, "double %s(const FlatXsView& v" % name),
          "%s branches on the view's OWN narrow_blocks flag rather than on a "
          "global, so the width travels with the pointers it describes" % name)
for name in FXS_STORES:
    check("if (v.narrow_blocks)" in body_after(FXK, "void %s(const FlatXsView& v" % name),
          "%s rounds to the block's width at the ONE place the value crosses "
          "into memory" % name)
for name in ("xsrMic", "xsrLmp", "xsrMicSsm", "xsrLmpSsm"):
    check("v.narrow_blocks ?" in body_after(XSRK, "double %s(const BatchView& v" % name),
          "%s asks the same flag, carried in the same view" % name)


def rule_no_bare_block_read(text: str) -> bool:
    """No body outside the accessors may name a micx/lmpx pointer directly."""
    stripped = strip_comments(text)
    # The accessor bodies are the only legal spellings; drop them, then look.
    for probe in ("v.ref_lmp[t][e]", "v.ref_lsm[e]", "v.ref_mic[t][e]",
                  "v.ref_msm[e]", "v.lmp[t][e]", "v.lsm[e]", "v.mic[t][e]",
                  "v.msm[e]", "v.mic[xt][e]", "v.lmp[xt][e]", "v.mic_ssm[e]",
                  "v.lmp_ssm[e]"):
        stripped = stripped.replace(probe, "")
    return not re.search(
        r"v\.(ref_lmp|ref_lsm|ref_mic|ref_msm|lmp|lsm|mic|msm|mic_ssm|lmp_ssm)\s*\[",
        stripped)


check(rule_no_bare_block_read(FXK),
      "src/FlatXsKernel.h reaches the four blocks ONLY through the accessors")
check(rule_no_bare_block_read(CTA),
      "src/FlatXsCtaKernel.cuh reaches the four blocks ONLY through the "
      "accessors -- both arms of the CTA kernel share that one text")
check(rule_no_bare_block_read(XSRK),
      "src/XsReconKernel.h (and therefore src/XeKernel.h, which consumes the "
      "same BatchView) reaches the blocks ONLY through xsrMic/xsrLmp/"
      "xsrMicSsm/xsrLmpSsm")

for field in ("const float* ref_lmp_f[N_ACTIVE]", "const float* ref_mic_f[N_ACTIVE]",
              "float*       lmp_f[N_ACTIVE]", "float*       mic_f[N_ACTIVE]",
              "int narrow_blocks"):
    check(field in FXK, "FlatXsView declares %s" % field)
for field in ("const float* mic_f[NXS]", "const float* lmp_f[NXS]",
              "const float* mic_ssm_f", "const float* lmp_ssm_f",
              "int narrow_blocks"):
    check(field in XSRK, "BatchView declares %s" % field)

check("v.narrow_blocks = narrow_blocks ? 1 : 0;" in SOLVE
      and "v.narrow_blocks = (dev_block_f != nullptr) ? 1 : 0;"
          in body_after(BACKEND, "bool stage(const xsr::BatchView& host"),
      "BOTH view builders stamp the flag from the block they actually bound, "
      "so a view can never describe a width its pointers do not have")
check("v.mic[t]       = nullptr;" in SOLVE and "v.mic[xt]   = nullptr;"
      in body_after(BACKEND, "bool stage(const xsr::BatchView& host"),
      "the wide pointers are NULLED on the narrow arm: a reader that skipped "
      "the accessors faults instead of computing something plausible")


# ---------------------------------------------------------------------------
# 11. WP20.1 -- THE HOST MATERIALISATION IS ONE SITE, AND IT IS PAID AFTER A
#     DRAIN.
#
# `_micx` / `_lmpx` are milk::Vector<double> and every host reader in the tree
# takes them as double, so the narrow arm owes a widening.  The rule is that
# the widening exists in exactly ONE function, that it is QUEUED at the
# asynchronous D2H rather than performed there, and that both callers pay it
# immediately after their stream drain -- a drain that forgot it would leave
# the host array holding the previous flat-XS epoch, which is the WP15 failure
# this arm may not reintroduce.
# ---------------------------------------------------------------------------
check(BACKEND_CODE.count("static_cast<double>(src[i])") == 1,
      "the float -> double widening is spelled ONCE in the whole backend")
WIDEN = body_after(BACKEND, "void drainMicxWiden()")
check("micx_widen.clear();" in WIDEN,
      "drainMicxWiden clears the debt it just paid, so a second call is a "
      "no-op rather than a second widening of stale staging")
check(BACKEND_CODE.count("d.drainMicxWiden();") == 2,
      "exactly two callers pay the widening: solveFlatXs's eager download and "
      "downloadFlatXsMicx's lazy one -- the same two call sites the copy list "
      "itself has")
for scope in ("CudaXsReconBackend.cu:solveFlatXs", "CudaXsReconBackend.cu:downloadFlatXsMicx"):
    idx = BACKEND.find('xfer::streamSync("%s"' % scope)
    check(idx > 0 and 0 < BACKEND.find("d.drainMicxWiden();", idx) - idx < 900,
          "the widening in %s runs AFTER that scope's streamSync -- the D2H is "
          "asynchronous, so widening at the issue would read staging bytes the "
          "device has not written" % scope)
check("micx_widen.push_back" in body_after(BACKEND, "bool downloadMicx(const char* leaf"),
      "downloadMicx QUEUES the widening instead of performing it")
check("EnsureMicxHost" in XSSET and "downloadFlatXsMicx" in XSSET,
      "XSSet::EnsureMicxHost is still the single host-side payer")


# ---------------------------------------------------------------------------
# 12. WP20.1 -- THE BYTE COUNTS AND THE MICX RECEIPT STAY TRUTHFUL.
#
# `micxBytesSaved()` is (deferred - paid).  Both halves must be quoted at the
# arm's width or the WP15 receipt reports a saving the arm did not make -- and
# WP15's number is the one a batch A/B is read against.
# ---------------------------------------------------------------------------
check("inline std::size_t flatxsMicxElemBytes()" in BACKEND,
      "the block's element width has a name, so it can be used rather than "
      "re-derived at each byte count")
check("flatxsNarrowBlocks() ? sizeof(float) : sizeof(double)"
      in body_after(BACKEND, "inline std::size_t flatxsMicxElemBytes()"),
      "and that name resolves through the SAME cached predicate the "
      "allocation used")
check("flatxsMicxBlockElems(nx) * flatxsMicxElemBytes()"
      in body_after(BACKEND, "inline unsigned long long flatxsMicxBlockBytes(std::size_t nx)"),
      "the deferred half of the MICX receipt is elements x the arm's width")
check("const std::size_t   ebytes = flatxsMicxElemBytes();" in
      body_after(BACKEND, "bool XsReconBackend::downloadFlatXsMicx("),
      "the paid half uses the same width, so the difference is a saving and "
      "not an artefact of two units")
check("d.ndev_flt != nullptr ? sizeof(float) : sizeof(double)" in BACKEND,
      "the nodal const-upload receipt is quoted at the arm's width too")


# ---------------------------------------------------------------------------
# 13. WP20.1 -- CRAM IS TOLD THE WIDTH; IT DOES NOT ASSUME ONE.
#
# The WP15.1 handover gives CRAM a device address inside the flat-XS block.
# Under the arm that block is float and CRAM's own state is not (the
# partial-fraction sum cancels).  A memcpy of `nmic * sizeof(double)` out of a
# float block is half the next slot's data: finite, plausible, wrong.
# ---------------------------------------------------------------------------
check("const void* micxDeviceSlot(int xt) const" in XSR_H,
      "micxDeviceSlot hands out an untyped address -- a typed one would let a "
      "consumer pick the wrong element width without saying so")
check("int micxDeviceElemBytes() const" in XSR_H,
      "and the width is a separate, explicit question")
check("int           mic_device_elem_bytes" in CRAM_H,
      "the CRAM views carry the width beside the addresses")
check(CRAM_H.count("const void*   mic_device[11]") == 2,
      "both the predictor and the corrector view carry untyped addresses")


def rule_offer_follows_the_view(cram_h: str, xsset_h: str, xsset_cpp: str) -> bool:
    """The PRODUCER of `mic_device` is spelled in the view's own pointee type.

    WP20.1 widened `PredictorView::mic_device` and `CorrectorView::mic_device`
    from `const double*[11]` to `const void*[11]`.  The one producer that fills
    them is XSSet::FillCramMicDevice, and C++ gives `const void**` NO implicit
    conversion to `const double**` -- a pointer-to-pointer conversion needs
    identical pointee types modulo top-level qualification.  So a widening that
    stops at the header is not a latent bug, it is a hard compile error at both
    call sites, and every gate the commit claims becomes unreachable because no
    binary can be produced.  That is exactly what shipped in WP21-D
    (5b237bd..d6b51a7) and had to be repaired out of band in 3e36a10.

    The expected signature is DERIVED FROM THE VIEW rather than hard-coded, so
    the next width change has to move both sides or fail here.  The width is
    checked too: a call site that takes `elem_bytes` and drops it leaves the
    view's FP64 default standing over a float block, which is the silent half
    of the same mistake.
    """
    view = strip_comments(cram_h)
    pointee = re.search(r"const\s+(\w+)\s*\*\s+mic_device\[11\]", view)
    if not pointee:
        return False
    want = ("FillCramMicDevice(const %s** dev, void*& ready, int& elem_bytes)"
            % pointee.group(1))
    if want not in strip_comments(xsset_h):
        return False
    body = strip_comments(xsset_cpp)
    if ("bool XSSet::" + want) not in body:
        return False
    if body.count("FillCramMicDevice(v.mic_device, mic_ready, mic_bytes)") != 2:
        return False
    return body.count("v.mic_device_elem_bytes = mic_bytes;") == 2


check(rule_offer_follows_the_view(CRAM_H, XSSET_H, XSSET),
      "XSSet::FillCramMicDevice -- the ONLY producer of mic_device -- is "
      "declared and defined in the pointee type the CRAM views declare, and "
      "both call sites publish the width it hands back into "
      "v.mic_device_elem_bytes.  A view widened without the producer does not "
      "compile at all (const void** does not convert to const double**), and a "
      "producer whose elem_bytes is dropped lets CRAM memcpy 8 bytes per "
      "element out of a 4-byte block")
FILLMIC = body_after(CRAM, "bool fillMic(const double* const* host_mic")
check("elem_bytes == static_cast<int>(sizeof(float))" in FILLMIC,
      "fillMic decides from the caller's word, not from a guess")
check("kCramWidenMic<<<" in FILLMIC,
      "a narrow source is WIDENED on the device rather than memcpy'd -- CRAM "
      "stays FP64 because its partial-fraction sum cancels")
check("nmic * sizeof(float)" in FILLMIC,
      "and the D2D byte tally is the float count that actually moved")


# ---------------------------------------------------------------------------
# 13.1 WP20.1 -- THE WIDTH HAS A PRODUCER, AND THE WIDEN ARM IS ON THE LEDGER.
#
# Two ways the section above can be true of the TEXT and false of the BINARY,
# and both have happened:
#
#   * `mic_device_elem_bytes` defaults to sizeof(double) in both views.  If no
#     caller ever writes it, `narrow` is always false, `kCramWidenMic` is dead
#     code, and the header's "the handover is a WIDENING KERNEL rather than a
#     memcpy" describes behaviour RASBERY_GPU_FP32=1 cannot reach.  A field
#     with a default and no producer is a comment, not a contract.
#   * The wide arm is a rasbery::xfer::memcpyAsync and is therefore in the
#     [RASBERY][XFER] ledger by construction; the narrow arm is a bare kernel
#     launch that no memcpy wrapper can see.  If it does not declare its own
#     row, the ledger's D2D total silently drops by the arm's whole payload the
#     moment the arm goes live, and the A/B that reads the FP32 and FP64
#     ledgers against each other charges the missing bytes to the arm.
# ---------------------------------------------------------------------------
def rule_cram_width_has_a_producer(xsset_text: str) -> bool:
    """The width is ASKED FOR and then HANDED ON, at both view-fill sites."""
    xs = strip_comments(xsset_text)
    offer = body_after(xs, "bool XSSet::FillCramMicDevice(")
    if "elem_bytes = _xsrecon_backend->micxDeviceElemBytes();" not in offer:
        return False
    # Both the predictor's and the corrector's view: one of the two left on the
    # default is one arm silently memcpying out of a float block.
    return (xs.count("FillCramMicDevice(v.mic_device, mic_ready, mic_bytes)") == 2
            and xs.count("v.mic_device_elem_bytes = mic_bytes;") == 2)


def rule_widen_arm_is_on_the_xfer_ledger(cram_text: str) -> bool:
    """The kernel arm declares the SAME ledger row the memcpy arm gets free,
    for the SAME bytes its own tally counts."""
    body = strip_comments(body_after(cram_text,
                                     "bool fillMic(const double* const* host_mic"))
    wide = re.search(r'xfer::memcpyAsync\(\s*"([^"]*)"\s*,\s*"([^"]*)"', body)
    if wide is None:
        return False
    narrow = re.search(r'xfer::note\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*'
                       r'rasbery::xfer::XferDir::kD2D\s*,\s*moved\s*\)', body)
    if narrow is None:
        return False
    if narrow.groups() != wide.groups():
        return False
    # ...and `moved` is what the backend-local tally adds, so the two receipts
    # cannot drift apart on either arm.
    return "n_micx_d2d_bytes += moved;" in body


check(rule_cram_width_has_a_producer(XSSET),
      "the CRAM view's element width has a PRODUCER: FillCramMicDevice asks "
      "the flat-XS backend and both view-fill sites hand the answer on.  With "
      "no writer the field keeps its sizeof(double) default, `narrow` is never "
      "true and kCramWidenMic is unreachable")
check(rule_widen_arm_is_on_the_xfer_ledger(CRAM),
      "the widening arm records its D2D bytes in the [RASBERY][XFER] ledger "
      "under the SAME scope and leaf the memcpy arm uses -- a kernel cannot go "
      "through the memcpy wrapper, so it declares the row with xfer::note or "
      "the ledger stops agreeing with itself the moment the arm goes live")


# ---------------------------------------------------------------------------
# 14. WP20.1 -- NODAL: ONE BODY, TWO INSTANTIATIONS, AND A KEY THAT KNOWS.
# ---------------------------------------------------------------------------
check("template <class ValueT>\nstruct NodalViewT {" in NODK,
      "the nodal view is ONE template, so the narrow and wide drives are two "
      "instantiations of one text and cannot drift")
check("using NodalView    = NodalViewT<double>;" in NODK
      and "using NodalViewF32 = NodalViewT<float>;" in NODK,
      "the FP64 view keeps its name, which is what makes every host caller "
      "and every replay tool textually unchanged")
for narrowed in ("const ValueT* xsrf", "const ValueT* eta1", "const ValueT* diagDI",
                 "ValueT* trlcff0", "ValueT* mu", "ValueT* matM", "ValueT* dsncff6"):
    check(narrowed in NODK, "NodalViewT narrows %s" % narrowed)
for wide in ("const double* hmesh", "const double* albedo", "const double* flux",
             "double*       jnet", "double*       phis", "double        reigv"):
    check(wide in NODK,
          "NodalViewT keeps %s FP64 -- geometry, the canonical state shared "
          "with CMFD, and the eigenvalue" % wide)

SHELL = body_after(NODK, "inline NodalViewT<ValueT> nodalWideShell(const NodalView& h)")
for f in ("hmesh", "albedo", "lktosfc", "neib", "lklr", "idirlr", "sgnlr",
          "chif_empty", "flux", "jnet", "phis", "reigv", "reigv_dev",
          "nxyz", "nsurf", "halt", "halt_slot"):
    check(("v.%s " % f) in SHELL or ("v.%s  " % f) in SHELL or ("v.%s=" % f) in SHELL
          or ("v.%s " % f) in SHELL.replace("=", " "),
          "nodalWideShell copies %s -- a field left null here is a kernel "
          "reading address zero" % f)
check(not re.search(r"v\.(eta1|xsrf|trlcff0|matM|mu|tau|dsncff2|chif)\s*=", SHELL),
      "and it copies NONE of the narrowed fields: those are bound from the "
      "float block by the caller, and a copy here would leave the narrow view "
      "pointing at the wide block")

for kern in ("kNodalTrl0", "kNodalTrl12", "kNodalMat", "kNodalEven",
             "kNodalMatEven", "kNodalJnet"):
    check("template <bool BATCHED, class VT>\n__global__ void %s(ndl::NodalViewT<VT> base"
          % kern in BACKEND,
          "%s is templated on the view's element type" % kern)
check("const ndl::NodalViewT<VT> v =" in BACKEND,
      "the slot guard resolves the view at the arm's precision")

KEY = body_after(BACKEND, "struct NodalGraphKey {")
check("int           precision = 0;" in KEY,
      "NodalGraphKey carries the PRECISION: a narrow drive has three more "
      "kernel nodes (the xs boundary conversion) and half-size memcpy nodes, "
      "so the two graphs may never alias")
check("precision == o.precision" in KEY,
      "and the longhand operator== compares it -- a field the key declares but "
      "does not compare is not a key field at all")
check("want.precision   = nnarrow ? 1 : 0;" in BACKEND,
      "the lookup asks for the precision this drive is about to enqueue")

check("auto enqueue_full = [&](auto& v) -> bool {" in BACKEND,
      "the drive body is a GENERIC lambda: not one statement moved, and the "
      "two arms are two instantiations of one text")
check("return nnarrow ? enqueue_full(v32) : enqueue_full(v);" in BACKEND,
      "and there is exactly ONE place that chooses between them")
check(BACKEND_CODE.count("enqueueDrive();") == 4,
      "all four enqueue sites go through that one chooser")
check("kNodalNarrowXs<<<" in BACKEND
      and BACKEND_CODE.count("kNodalNarrowXs<<<") == 3,
      "the three macroscopic inputs are narrowed by a captured kernel, once "
      "per drive -- they are rewritten every outer, so a residency-gated copy "
      "would hand the drive the previous outer's cross-sections")


# ---------------------------------------------------------------------------
# 15. WP20.1 -- THE NON-FINITE VALVE REACHES NODAL, ON THE ARM THE ARM IS
#     MEASURED ON, AND ITS RECOVERY IS THE ONE THE ALLOCATION ALLOWS.
#
# The valve used to read `host.jnet` under a gate that asked whether the D2H
# had been elided.  On the production segment arm
# (RASBERY_GPU_OUTER_SEGMENT_V2, part of V6_ENV) it is ALWAYS elided --
# setCanonicalNodalSegmentMode(true) sets the materialize mask to 0 -- so the
# valve could not fire at all, while `upddhat` consumed the drive's answer
# anyway and CMFD's own NONFINITE_DETECTED took the blame for it.  The rules
# below are that defect, each one stated so it fails on the old shape.
# ---------------------------------------------------------------------------


def rule_valve_is_on_device(backend: str) -> bool:
    """The verdict is produced by a kernel over the DEVICE jnet, launched from
    inside the drive body, after the kernel that wrote it.  A valve that reads a
    host array cannot run on an arm that never fills one."""
    code = strip_comments(backend)
    if "__global__ void kNodalJnetFinite(" not in code:
        return False
    kern = body_after(code, "__global__ void kNodalJnetFinite(")
    if "isfinite(jnet[i])" not in kern or "*nonfinite = 1u;" not in kern:
        return False
    drive = body_after(code, "auto enqueue_full = [&](auto& v) -> bool {")
    launch = drive.find("kNodalJnetFinite<<<")
    writer = drive.find("kNodalJnet<false><<<")
    return launch > 0 and 0 < writer < launch


def rule_valve_survives_the_elision(backend: str) -> bool:
    """Nothing between the arm test and the read may ask whether a download
    happened.  This is the rule the shipped defect broke."""
    code = strip_comments(backend)
    at = code.find("d.narrowNonFinite(!d.nodal_drain_deferred)")
    if at < 0:
        return False
    open_at = code.rfind("if (nnarrow", 0, at)
    if open_at < 0:
        return False
    window = code[open_at:at]
    return ("canonicalElidesDownload" not in window
            and "canonical_materialize" not in window)


def rule_valve_word_is_mapped(backend: str) -> bool:
    """One page-locked word with a device alias: no D2H to keep in step with
    the graph key, and -- the point -- nothing about the verdict's route to the
    host that the download elision can switch off.  It is written once, at
    allocation, and never reset: the word is STICKY, which is what lets a host
    that must read it LATE read it correctly."""
    code = strip_comments(backend)
    return ("cudaHostAllocMapped" in code
            and "cudaHostGetDevicePointer(reinterpret_cast<void**>(&d.nodal_nfin_dev)"
                in code
            and code.count("*d.nodal_nfin_host = 0u;") == 1)


def rule_deferred_read_never_blocks(backend: str) -> bool:
    """A deferred drive handed the segment an event precisely so the host would
    not wait; the valve may QUERY that event but never wait on it."""
    code = strip_comments(backend)
    fn = body_after(code, "[[nodiscard]] bool narrowNonFinite(bool settled)")
    return "cudaEventQuery(nodal_done_event)" in fn and "cudaEventSynchronize" not in fn


check(rule_valve_is_on_device(BACKEND),
      "the narrow drive's answer is judged ON THE DEVICE, by a node of the "
      "drive that produced it -- jnet is the one array a consumer reads next, "
      "and on the segment arm it is never copied to the host at all")
check(rule_valve_survives_the_elision(BACKEND),
      "and the valve is INDEPENDENT OF THE DOWNLOAD ELISION: a gate on "
      "canonicalElidesDownload(Jnet) is exactly the defect where a NaN jnet "
      "reached upddhat, CMFD latched, and the receipt blamed CMFD for a "
      "failure the nodal arm caused")
check(rule_valve_word_is_mapped(BACKEND),
      "the verdict travels in a mapped page-locked word, written only when "
      "there is something to say and never reset")
check(rule_deferred_read_never_blocks(BACKEND),
      "reading it costs the deferred drain nothing: the drive's event is "
      "queried, never waited on")
check("d.nodal_nfin_pending = true;" in BACKEND_CODE
      and "if (d.nodal_nfin_pending && d.narrowNonFinite(false)) {" in BACKEND_CODE,
      "a drive that could not read its own word leaves the question standing, "
      "and the NEXT drive asks it before enqueueing anything")
check("if (!in_segment) d.resolveNarrowVerdict();"
      in body_after(BACKEND_CODE, "void XsReconBackend::setCanonicalNodalSegmentMode("),
      "and the segment exit is the latency bound: every drive of a host-free "
      "segment defers, so without this a run whose LAST segment went "
      "non-finite would end with the arm never refused")
check('d.status = "WP20.1: the narrow nodal non-finite valve could not be armed";'
      in BACKEND_CODE,
      "a narrow drive that cannot arm the valve refuses itself to the CPU body "
      "-- running the narrow arm unvalved is the state this section exists to "
      "make unreachable")
check('rasbery::fp32::latchOff(rasbery::fp32::Backend::Nodal, "nonfinite jnet");'
      in BACKEND,
      "the narrow nodal drive latches on a non-finite jnet -- the one array a "
      "consumer reads on the very next line")
check("nodal_device_refused = true;" in body_after(BACKEND, "void latchNarrowNodalOff()")
      and "if (d.nodal_device_refused) return false;" in BACKEND,
      "and the recovery is a REFUSAL of the device arm, sticky for the "
      "process: nodal's precision is an allocation, so there is no wide device "
      "arm to demote to without re-laying-out mid-run")
check("dropNodalGraph();" in body_after(BACKEND, "void latchNarrowNodalOff()"),
      "the latch drops the graphs captured under the old precision")
check(BACKEND_CODE.count("latchNarrowNodalOff()") == 4,
      "and there is ONE latch body with exactly three callers -- the drive, "
      "the next drive's poll, and the segment exit -- so a caller cannot "
      "half-refuse the arm")


# ---------------------------------------------------------------------------
# 16. WP20.1 -- FEATURE-OFF IS STILL UNREACHABLE, NOT MERELY UNTAKEN.
# ---------------------------------------------------------------------------
ENSURE = body_after(BACKEND, "bool ensure(int want_nxyz, int want_fuel)")
check("if (narrow) off = 0;" in ENSURE,
      "the FP64 arm's dev_block layout is the one that shipped: micx/lmpx are "
      "laid out first and `off` continues past them, so off_iden / off_xs land "
      "exactly where they always did")
check("if (nnarrow) { d.n_off_consts = foff; foff += 9 * ndg0; }" in BACKEND
      and "else         { d.n_off_consts = off;  off  += 9 * ndg0; }" in BACKEND,
      "and the nodal block splits the same way, so with the flag off its "
      "sequence is literally the layout that shipped")
check("if (!nnarrow) {" in SOLVE_NODAL_BINDS,
      "the FP64 nodal bind block is intact and guarded, not rewritten")
check("int narrow_blocks;" in FXK and "int narrow_blocks;" in XSRK,
      "both narrow flags are plain ints that the `View v{}` every builder in "
      "the tree uses zero-initialises -- which is what the feature-off claim "
      "rests on")

# ---------------------------------------------------------------------------
# 12. THE LATCH HAS A CASE BOUNDARY (defect: src/CudaBICGBackend.cu
#     latchFp32Off -- process-wide, sticky, and invisible per case).
#
# WHAT WENT WRONG.  Driver.h's kArmEnv comment claims that listing
# RASBERY_GPU_FP32 there folds it into the WP10.1 case key "so an FP64 answer
# can never be served to an FP32 request".  That is true of the KEY and false
# of the ANSWER in the only mode the dispatcher and the soak use.
# `latchFp32Off()` retires the narrow CMFD path for the rest of the PROCESS
# (`fp32_latched_off` is a member of BatchCore, and BatchCore is owned by the
# process-global arena), while the evaluator is one process for N cases.  So a
# diverging candidate at case 12 of 64 sends cases 13..64 to FP64 with an
# unchanged FP32 case_key on every receipt, and the only trace was a
# [RASBERY][FP32][FALLBACK] line on stderr plus a process-total
# [RASBERY][FP32] line at close() -- which names no boundary and is not
# printed at all if the worker is killed.
#
# WHAT THESE RULES PIN.  Not that the latch is per case: it CANNOT be, one
# captured graph serves every slot of a launch and that is why the latch is
# arena-wide.  They pin that the RESOLVED precision is a per-case fact, is
# bracketed around the whole case (a case that throws is the case that
# latches), reaches the per-case receipt, is counted per wave, and -- the one
# behavioural rule -- that a case which did not run under the arm its key names
# does not publish that key as a cacheable identity.
# ---------------------------------------------------------------------------
CMFD_STATE_CALL = "rasbery::fp32::backendState(rasbery::fp32::Backend::Cmfd)"


def rule_receipt_carries_precision(driver: str) -> bool:
    """Driver::CaseReceipt carries the resolved precision, both ends."""
    try:
        rec = body_after(driver, "struct CaseReceipt")
    except LookupError:
        return False
    return (re.search(r"\bstd::string\s+fp32_entry;", rec) is not None
            and re.search(r"\bstd::string\s+fp32_exit;", rec) is not None)


def rule_stamp_brackets_the_whole_case(evaluator: str) -> bool:
    """Read before the try, stamped after the LAST catch block.

    The case that latches is the case that throws -- a diverging candidate
    comes back non-finite, the valve absorbs it, and the FP64 path raises it
    on the next outer.  A stamp inside the try is therefore missing on exactly
    the receipt that carries the boundary.
    """
    try:
        body = body_after(evaluator, "static void runOneCase(")
    except LookupError:
        return False
    if body.count(CMFD_STATE_CALL) != 2:
        return False
    try_at = body.find("try {")
    if try_at < 0 or not 0 <= body.find(CMFD_STATE_CALL) < try_at:
        return False
    try:
        catch_all = body_after(body, "catch (...)")
    except LookupError:
        return False
    after = body[body.rindex(catch_all) + len(catch_all):]
    return "receipt.fp32_entry" in after and "receipt.fp32_exit" in after


def rule_latched_case_withholds_its_key(evaluator: str) -> bool:
    """A latched case does not publish a key a cache could serve FP32 from."""
    try:
        body = body_after(evaluator, "void reportCase(long long wave_id")
    except LookupError:
        return False
    return ('receipt.fp32_exit == "latched"' in body
            and "receipt.case_key.empty() || fp32_latched" in body
            and r'\"case_key_withheld\":' in body)


def rule_case_line_reports_both_ends(evaluator: str) -> bool:
    try:
        body = body_after(evaluator, "void reportCase(long long wave_id")
    except LookupError:
        return False
    return all(field in body for field in (r'\"fp32_entry\":', r'\"fp32_exit\":',
                                           r'\"fp32_latch_boundary\":'))


def rule_every_wave_receipt_names_the_boundary(evaluator: str) -> bool:
    """BOTH wave receipts -- the wave path's and the rolling path's."""
    waves = evaluator.count(r'[RASBERY][EVALUATOR][WAVE] {\"wave_id\":')
    if waves != 2:
        return False
    return (evaluator.count(r'\"fp32_latched_cases\":') == waves
            and evaluator.count(r'\"fp32_boundary_cases\":') == waves
            and evaluator.count(r'\"fp32_first_latched_case\":') == waves)


def rule_tally_is_reset_per_wave(evaluator: str) -> bool:
    """Once per stand-up, both paths: a wave must not inherit a stale count."""
    return len(re.findall(r"_fp32\s*=\s*Fp32WaveTally\{\};", evaluator)) == 2


check('#include "GpuFp32Arm.h"' in EVAL,
      "the evaluator includes the arm header -- it has to ASK the arm what "
      "precision answered a case, not infer it from the environment")
check(rule_receipt_carries_precision(DRIVER),
      "Driver::CaseReceipt carries fp32_entry/fp32_exit: the resolved "
      "precision is a per-case fact because the latch has a case boundary")
check(rule_stamp_brackets_the_whole_case(EVAL),
      "runOneCase reads the arm's state either side of the case, both reads "
      "outside the try/catch, so the case that threw AND latched still reports")
check(rule_latched_case_withholds_its_key(EVAL),
      "a case whose CMFD arm resolved to \"latched\" emits case_key null with "
      "case_key_withheld -- it did not run under the arm its key names, so a "
      "cache must not be able to serve it to an FP32 request")
check(rule_case_line_reports_both_ends(EVAL),
      "[EVALUATOR][CASE] reports fp32_entry, fp32_exit and the derived "
      "fp32_latch_boundary, so a Gate A/B comparison cannot silently mix arms")
check(rule_every_wave_receipt_names_the_boundary(EVAL),
      "BOTH [EVALUATOR][WAVE] receipts count the latched cases and name the "
      "first one -- the process-total [RASBERY][FP32] line is printed at "
      "close() and a SIGKILLed worker never prints it")
check(rule_tally_is_reset_per_wave(EVAL),
      "the per-wave tally is reset at both stand-ups, so a wave never reports "
      "the previous wave's boundary as its own")
check("fp32_latched_off" in body_after(BICG, "void latchFp32Off()"),
      "the latch itself stays ARENA-WIDE -- one captured graph serves every "
      "slot, so a per-slot precision would double the node count.  The fix is "
      "that the boundary is REPORTED, not that the latch is per case")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each probe is the rule applied to a mutation that MUST
# make it fail; a rule that cannot fail is not a rule.
# ---------------------------------------------------------------------------
#: The exact three lines runOneCase stamps with, so the "moved inside the try"
#: probe below is a real MOVE and not a deletion dressed up as one.
STAMP_BLOCK = (
    "        receipt.fp32_entry = fp32_entry;\n"
    "        receipt.fp32_exit =\n"
    "            rasbery::fp32::backendState(rasbery::fp32::Backend::Cmfd);\n")

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
    ("a kernel body names a micx block pointer directly again",
     lambda: rule_no_bare_block_read(
         CTA + "\n  double z = v.ref_mic[t][q * nxyz + l];\n")),
    ("the xsrecon reader goes back to the wide pointer",
     lambda: rule_no_bare_block_read(
         XSRK + "\n  double z = v.mic[xt][(iso * NG + ig) * nxyz + l];\n")),
    # The shipped defect, restored four ways.
    ("the valve is gated on the jnet D2H again",
     lambda: rule_valve_survives_the_elision(
         BACKEND.replace(
             "if (nnarrow) {\n        // SET BEFORE THE READ",
             "if (nnarrow && !gpu::canonicalElidesDownload(\n"
             "                       canon, gpu::CanonicalRegion::Jnet,\n"
             "                       d.canonical_materialize)) {\n"
             "        // SET BEFORE THE READ"))),
    ("the verdict goes back to a scan of the host array",
     lambda: rule_valve_is_on_device(
         BACKEND.replace("kNodalJnetFinite<<<", "kNodalJnetSkipped<<<"))),
    ("the device word stops being mapped and needs a D2H again",
     lambda: rule_valve_word_is_mapped(
         BACKEND.replace("cudaHostAllocMapped", "cudaHostAllocDefault"))),
    ("the deferred read starts blocking the drain it exists to preserve",
     lambda: rule_deferred_read_never_blocks(
         BACKEND.replace("const cudaError_t q = cudaEventQuery(nodal_done_event);",
                         "const cudaError_t q = cudaEventSynchronize(nodal_done_event);"))),
    ("the PPR extension goes back to being declared everywhere and read nowhere",
     lambda: rule_flag_is_read(
         ARM_CODE.replace('envFlagOn("RASBERY_GPU_FP32_PPR")', "false"),
         "RASBERY_GPU_FP32_PPR")),
    ("a flag row knob is 'read' by a COMMENT rather than by a gate",
     lambda: rule_flag_is_read(
         strip_comments('/// envFlagOn("RASBERY_GPU_FP32_PPR") -- one day\n'),
         "RASBERY_GPU_FP32_PPR")),
    ("inScope() stops testing PPR and the narrowing engages on the main flag alone",
     lambda: rule_scope_gates_extensions(
         SCOPE.replace("if (which == Backend::Ppr)  return pprExtended();", ""))),
    # 3b (a): the state goes dead by the table rather than by the ordering --
    # exactly how it was dead before WP20.2 converted CRAM.
    # BOTH gated backends, because there are two of them now: a mutation that
    # defers only CRAM leaves PPR converted-and-gated, so `declined` is still
    # reachable and the control would be reporting on nothing.
    ("no gated backend is converted any more, so `declined` is dead text",
     lambda: rule_declined_is_reachable(
         ARM.replace("case Backend::Cram:   return true;",
                     "case Backend::Cram:   return false;")
            .replace("case Backend::Ppr:    return true;",
                     "case Backend::Ppr:    return false;"))),
    # 3b (a) again, the other way in: the ordering answers before scope is asked.
    ("backendState stops asking !inScope at all",
     lambda: rule_declined_is_reachable(
         ARM.replace('if (!inScope(which)) return "declined";', ""))),
    # 3b (a), the third way: the gate leaves inScope, so every converted backend
    # routes and the receipt can only ever say fp32.
    ("inScope loses the per-backend gates that make `declined` possible",
     lambda: rule_declined_is_reachable(
         ARM.replace("if (which == Backend::Cram) return cramExtended();", "")
            .replace("if (which == Backend::Ppr)  return pprExtended();", ""))),
    # 3b (b): the promise goes back to being prose with nothing behind it.
    ("the CRAM refusal stops being counted",
     lambda: rule_declined_is_counted(
         CRAM.replace("rasbery::fp32::noteDemotion(rasbery::fp32::Backend::Cram);",
                      ""))),
    # 3b (b), the subtle one: counting only when the EXTENSION was given inverts
    # the meaning -- a demotion would then mark the arm that was TAKEN.
    ("the count moves off the arm and onto the extension",
     lambda: rule_declined_is_counted(
         CRAM.replace("rasbery::fp32::armed()", "_impl->fp32_pole"))),
    ("the untiled body publishes an iden row FROM the float workspace again",
     lambda: rule_iden_authority_is_published_wide(
         CTA + "\n  v.iden[IB10 * nxyz + l] = w.iden[IB10];\n")),
    ("the TILED body publishes an iden row from the float workspace",
     lambda: rule_iden_authority_is_published_wide(
         CTA + "\n  v.iden[IH1 * nxyz + l] = w[j].iden[IH1];\n")),
    ("the density local stops being double",
     lambda: rule_iden_authority_is_published_wide(
         CTA.replace("const double nden_b10", "const float nden_b10"))),
    ("the workspace copy drifts to its own expression of the same density",
     lambda: rule_iden_authority_is_published_wide(
         CTA.replace("w.iden[IH1]             = nden_h1;",
                     "w.iden[IH1]             = 2.0 * nH2O;"))),
    # WP21-D shipped exactly this: the view widened, the producer did not, and
    # the tree stopped compiling at src/XSSet.cpp:5374 and :5434.
    ("the view is widened but the producer stays on const double**",
     lambda: rule_offer_follows_the_view(
         CRAM_H,
         XSSET_H.replace("FillCramMicDevice(const void**",
                         "FillCramMicDevice(const double**"),
         XSSET.replace("FillCramMicDevice(const void**",
                       "FillCramMicDevice(const double**"))),
    ("a call site stops publishing the width it was handed",
     lambda: rule_offer_follows_the_view(
         CRAM_H, XSSET_H,
         XSSET.replace("    v.mic_device_elem_bytes = mic_bytes;\n", "", 1))),
    ("the offer stops asking the backend for the block's width",
     lambda: rule_cram_width_has_a_producer(
         XSSET.replace("elem_bytes = _xsrecon_backend->micxDeviceElemBytes();",
                       "elem_bytes = static_cast<int>(sizeof(double));"))),
    ("one of the two view-fill sites drops the width and keeps the default",
     lambda: rule_cram_width_has_a_producer(
         XSSET.replace("    v.mic_device_elem_bytes = mic_bytes;\n", "", 1))),
    ("the widening arm's ledger row drifts off the memcpy arm's",
     lambda: rule_widen_arm_is_on_the_xfer_ledger(
         CRAM.replace('rasbery::xfer::note("CudaCramBackend.cu:fillMic"',
                      'rasbery::xfer::note("CudaCramBackend.cu:fillMicWiden"'))),
    ("the widening arm stops declaring a ledger row at all",
     lambda: rule_widen_arm_is_on_the_xfer_ledger(
         CRAM.replace("rasbery::xfer::note(", "(void)sizeof("))),
    ("the ledger is told a different number from the one the tally adds",
     lambda: rule_widen_arm_is_on_the_xfer_ledger(
         CRAM.replace("rasbery::xfer::XferDir::kD2D, moved);",
                      "rasbery::xfer::XferDir::kD2D, bytes);"))),
    ("converted() defers CRAM again, so inScope()'s cramExtended() test is "
     "shadowed by the !converted line above it and `declined` goes dead",
     lambda: rule_scope_gate_is_live(
         ARM.replace("case Backend::Cram:   return true;",
                     "case Backend::Cram:   return false;"))),
    ("inScope() stops gating anything at all, so the extension flags fork the "
     "case key without being able to withhold anything",
     lambda: rule_scope_gate_is_live(
         ARM.replace("    if (which == Backend::Cram) return cramExtended();\n", "")
            .replace("    if (which == Backend::Ppr)  return pprExtended();\n", ""))),
    # And the new gate's own liveness: PPR is gated, so PPR must be converted.
    ("converted() defers PPR again, so its brand-new gate is dead text",
     lambda: rule_scope_gate_is_live(
         ARM.replace("case Backend::Ppr:    return true;",
                     "case Backend::Ppr:    return false;"))),
    ("backendState() returns fp32 before it tests !inScope, so a withheld "
     "backend reports the conversion it did not get",
     lambda: rule_declined_is_reachable_ordered(
         '{ if (!armed()) return "fp64";\n'
         '  if (!converted(which)) return "deferred";\n'
         '  if (latched(which)) return "latched";\n'
         '  return "fp32";\n'
         '  if (!inScope(which)) return "declined"; }', ARM)),
    ("the header claims a demotion the tree never counts",
     lambda: rule_demotion_claim_matches_call_sites(
         ARM.replace("///   Backend::FlatXs",
                     "///   Backend::Xe       the refusal, allegedly\n"
                     "///   Backend::FlatXs"),
         SRC_ALL)),
    ("a demotion is counted for a backend the header never declared",
     lambda: rule_demotion_claim_matches_call_sites(
         ARM,
         SRC_ALL + "\nrasbery::fp32::noteDemotion(rasbery::fp32::Backend::Xe);\n")),
    ("the receipt stops carrying the resolved precision",
     lambda: rule_receipt_carries_precision(
         DRIVER.replace("std::string        fp32_entry;", ""))),
    ("the precision stamp moves INSIDE the try, so a case that threw while "
     "latching reports nothing",
     lambda: rule_stamp_brackets_the_whole_case(
         EVAL.replace(STAMP_BLOCK, "").replace(
             "                receipt = driver.caseReceipt();\n",
             "                receipt = driver.caseReceipt();\n" + STAMP_BLOCK))),
    ("a latched case goes back to publishing its FP32 case_key",
     lambda: rule_latched_case_withholds_its_key(
         EVAL.replace("(receipt.case_key.empty() || fp32_latched)",
                      "(receipt.case_key.empty())"))),
    ("the per-case line drops the resolved precision again",
     lambda: rule_case_line_reports_both_ends(
         EVAL.replace(r'\"fp32_exit\":', r'\"unused\":'))),
    ("only ONE of the two wave receipts names the boundary",
     lambda: rule_every_wave_receipt_names_the_boundary(
         EVAL.replace(r'<< ",\"fp32_latched_cases\":" << _fp32.latched_cases' + "\n",
                      "", 1))),
    ("a wave stops resetting the tally and inherits the last wave's boundary",
     lambda: rule_tally_is_reset_per_wave(
         EVAL.replace("_fp32          = Fp32WaveTally{};\n", "", 1))),
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
