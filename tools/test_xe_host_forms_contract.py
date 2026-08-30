#!/usr/bin/env python3
"""Contract: the split Xe arm's host arithmetic is BARRIERED, so inlining cannot
move it -- RASBERY_XE_HOST_FORMS (238, 2026-08-31).

===========================================================================
WHAT HAPPENED
===========================================================================

`048c6c1` put RASBERY_NEVER_INLINE on the default-off transaction arm so that
-finline-functions-called-once could not splice it into SolveLoop.  238 then ran
the A/B, arm X with RASBERY_GPU_XE_TXN unset, and returned the INVERSE of what
the fix predicted:

    048c6c1                                  c1a5d9116df9edb3 / 4601 outers
    the SAME tree, that one token removed    22b9a3187bfb4beb / 4566 outers

and 4566 is the `7cfe3a4` baseline.  Every earlier point agrees on the shape of
the thing: 47161ed clean, 47161ed+71092e2 drifted, d7b81af / 8919331 / 32ac308
drifted, the kernel-side hunk drifted.  The flag-off trajectory flips with ANY
change of gcc's inlining and codegen context around the split device Xe arm.

The class in docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md section 7 was
therefore right and its REMEDY was a coin toss.  `d85984e` did not fix
anything -- it happened to land on the 7cfe3a4 form.

===========================================================================
THE ROOT CAUSE, AND WHY AN ATTRIBUTE CANNOT BE THE FIX
===========================================================================

src/Driver.h is one implicitly-inline class in ONE translation unit
(src/main.cpp), built -O3 -ffp-contract=fast with no LTO.  The four normal
equations of TryAndersonXeStepGpu were raw C:

    det      = a * c - b * b;
    gamma[0] = (c * p - b * q) / det;
    gamma[1] = (a * q - b * p) / det;
    proj     = gamma[0] * p + gamma[1] * q;

Each is an add fed by two multiplies, and at -ffp-contract=fast gcc's
convert_mult_to_fma pass decides WHICH multiply to fold into the add -- per
call site, as a function of the inlining context it finds itself in.  So the
flag-off trajectory was a function of the call graph, and every edit near
SolveLoop re-rolled it.  An attribute is one of the things that re-rolls it;
that is why 048c6c1 moved the number it was written to hold still.

The fix removes the decision instead of steering it.  Every one of the four
sites now goes through xe::xeSiteSub / xe::xeSiteAdd -- the same bodies the
device arm runs -- whose three states are all written with xsr::xsrFma and
xsr::xsrMul, and xsrMul carries an `asm volatile` barrier on gcc.  Once every
FP expression on the path is barriered or unfusable, INLINING CANNOT CHANGE THE
RESULT, and the arm's trajectory becomes a property of one written-down
constant (XE_HOST_FORMS_DEFAULT) instead of a property of the call graph.

===========================================================================
WHAT THIS FILE PINS
===========================================================================

  S.  THE SCAN.  Every binary `*`, `+` and `-` on a double inside
      TryAndersonXeStepGpu and TryAndersonXeStepGpuTxn is enumerated, and each
      one must be a member of an ALLOWED multiset -- with its reason -- or the
      file fails.  This is the rule that makes "the path is barriered" a check
      rather than a claim, and it is what a future edit near SolveLoop trips
      over before 238 has to measure anything.
  M.  THE MASK MACHINERY.  XE_HOST_FORMS_DEFAULT is four two-bit fields in the
      algebra channel; xeHostFormMask() is declared in XeFormMask.h, defined in
      XeFormMiner.cpp, NEVER mines, reads its env once into a cached static,
      trims to XE_ALGEBRA_FORMS, and prints one receipt naming all four sites.
  A.  THE AUDIT sees BOTH masks, so `forms_audit_mismatch: 0` has a readable
      precondition and a mismatch names which knob closes it.
  N.  THE NEGATIVE CONTROL THAT IS NOT A MUTATION: the PURE host arm
      (TryAndersonXeStep's own body, RASBERY_GPU_XE unset) must stay
      UNBARRIERED.  It is the frozen MASTER reference the whole campaign is
      measured against; barriering it would move the baseline rather than hold
      it, and its arithmetic is not a trajectory-deciding CHOICE -- it IS the
      definition.

Pure source reading: no compiler, no device, no run.

    tools/test_xe_host_forms_contract.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILURES = []
CHECKS = 0


def read(*parts):
    with open(os.path.join(ROOT, *parts), "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


DRIVER = read("src", "Driver.h")
KERNEL = read("src", "XeKernel.h")
MASK_H = read("src", "XeFormMask.h")
MINER = read("src", "XeFormMiner.cpp")
AUDIT_CPP = read("src", "XeFormAudit.cpp")
RECEIPT = read("src", "XeGpuReceipt.h")


def check(ok, label, detail=""):
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(label + ((" -- " + detail) if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# Source helpers.  Comments AND string literals go: a rule a comment can satisfy
# is a rule about prose, and a static_assert message that says "two-column" must
# not be read as a subtraction.
# ---------------------------------------------------------------------------

def strip_comments_and_strings(text):
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif text[i] in "\"'":
            q = text[i]
            j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            out.append(" ")  # the literal becomes whitespace, not tokens
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def strip_comments(text):
    """Comments only -- STRING LITERALS SURVIVE.

    The scan above must not see a static_assert message as arithmetic, so it
    drops literals.  The environment-knob rules need the opposite: the knob IS
    a literal, and a stripper that ate it would count zero readers and call
    that a pass."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif text[i] in "\"'":
            q = text[i]
            j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:min(j + 1, n)])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def body_of(text, anchor):
    at = text.find(anchor)
    if at < 0:
        return ""
    open_at = text.find("{", at)
    if open_at < 0:
        return ""
    depth, i, n = 0, open_at, len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
        i += 1
    return ""


def squash(text):
    return re.sub(r"\s+", "", text)


GPU_SIG = "static bool TryAndersonXeStepGpu(SolverContext& ctx"
TXN_SIG = "bool TryAndersonXeStepGpuTxn(SolverContext& ctx"
HOST_SIG = "static bool TryAndersonXeStep(SolverContext& ctx, XeAndersonState& aa, double power"


# ===========================================================================
# S.  The scan -- every binary FP operator on the split arm's path
# ===========================================================================
#
# The names that carry a double in these two bodies.  A scan that guessed types
# would either miss `aa.ncol - 1` (an int, and harmless) or drown in it; naming
# the doubles is both the cheaper implementation and the more honest one --
# adding a double to either function means adding it here, which is the moment
# to ask whether its arithmetic needs a barrier.
DOUBLE_NAMES = {
    # TryAndersonXeStepGpu
    "picard", "gg", "a", "b", "c", "p", "q", "det", "proj", "gamma", "dots",
    "pred2", "step", "max_step", "power", "gj", "xe_change",
    "XE_ANDERSON_MIN_GRAM", "XE_EQUILIBRIUM_TOLERANCE",
    # TryAndersonXeStepGpuTxn -- `out` and `req` are the two structs that carry
    # doubles across the device boundary.
    "out", "req", "value",
}

TERM = (r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*(?:\[[^\]\[]*\])?"
        r"|[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?")
LEFT = re.compile(r"(?:%s)\s*$" % TERM)
RIGHT = re.compile(r"\s*(?:%s)" % TERM)


def base(term):
    m = re.match(r"[A-Za-z_][A-Za-z_0-9]*", term)
    return m.group(0) if m else ""


def fp_binops(body):
    """Every binary `*`, `+`, `-` whose left or right operand is a double.

    Operator-by-operator rather than expression-by-expression, so a chain like
    `MIN_GRAM * a * c` yields BOTH of its multiplies: a scan that reported one
    of them could be defeated by writing the offending product second.
    """
    found = []
    for i, ch in enumerate(body):
        if ch not in "*+-":
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        prv = body[i - 1] if i else ""
        # ++, --, +=, -=, *=, ->, and the trailing half of a doubled token
        if nxt == "=" or ch == nxt or (ch == "-" and nxt == ">"):
            continue
        if prv == ch:
            continue
        # a unary sign, a pointer declarator or a dereference: the character
        # before is an operator or an opener, not the end of an operand
        if prv in "=+-*/<>(,[&|!?:;{}%^~":
            continue
        lm = LEFT.search(body[:i])
        rm = RIGHT.match(body[i + 1:])
        if not lm or not rm:
            continue
        lhs, rhs = lm.group(0).strip(), rm.group(0).strip()
        if base(lhs) in DOUBLE_NAMES or base(rhs) in DOUBLE_NAMES:
            found.append(squash(lhs + " " + ch + " " + rhs))
    return found


# THE ALLOWED MULTISET.  Every entry carries the reason it cannot be moved by an
# inlining decision.  Counts are exact: `a*c` is allowed ONCE, so re-introducing
# `const double det = a * c - b * b;` is caught by the count even though one of
# its three operators is on this list.
ALLOWED_GPU = {
    # SAFEGUARD 1/4's conditioning floor, `det > XE_ANDERSON_MIN_GRAM * a * c`.
    # A multiply CHAIN feeding a comparison: there is no add for gcc to fuse
    # into, on either compiler, and the device arm spells it identically.
    "XE_ANDERSON_MIN_GRAM*a": 1,
    "a*c": 1,
    # The one-column form of the same floor, `a > XE_ANDERSON_MIN_GRAM * gg`.
    "XE_ANDERSON_MIN_GRAM*gg": 1,
    # SAFEGUARD 2/4, `pred2 = gg - proj`.  ONE subtraction, and it is safe only
    # BECAUSE `proj` is a barrier output on both paths (xeSiteAdd's fma/add of
    # xsrMul results, or xsrMul directly) -- an unbarriered product here is
    # exactly what -ffp-contract=fast folds into an fnma in some inlining
    # contexts and not others.  Rule S3 is what keeps that true.
    "gg-proj": 1,
    # SAFEGUARD 4/4's trust region, `step <= max_step * picard`.  One multiply
    # into a comparison; no add is reachable from it.
    "max_step*picard": 1,
}

ALLOWED_TXN = {
    # The rejection's reported value.  Both operands were DOWNLOADED from the
    # device -- the transaction did the algebra there -- so this subtraction
    # decides nothing about a trajectory; it formats a log line.
    "out.gg-out.proj": 1,
}


def multiset(items):
    counts = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return counts


def rule_scan(driver_text, sig, allowed):
    code = strip_comments_and_strings(driver_text)
    body = body_of(code, sig)
    if not body:
        return False, "the function is not findable"
    got = multiset(fp_binops(body))
    extra = {k: v for k, v in got.items()
             if k not in allowed or v > allowed[k]}
    missing = {k: v for k, v in allowed.items()
               if got.get(k, 0) < v}
    if extra:
        return False, "unbarriered/unaccounted FP arithmetic: %s" % sorted(extra)
    if missing:
        return False, "an ALLOWED site vanished (%s); the list is stale, not the code" % sorted(missing)
    return True, ""


ok, why = rule_scan(DRIVER, GPU_SIG, ALLOWED_GPU)
check(ok,
      "S1 TryAndersonXeStepGpu contains no double `*`/`+`/`-` outside the "
      "enumerated unfusable sites", why)

ok, why = rule_scan(DRIVER, TXN_SIG, ALLOWED_TXN)
check(ok,
      "S2 TryAndersonXeStepGpuTxn's host-side algebra is one subtraction of two "
      "downloaded scalars", why)

GPU_BODY = body_of(strip_comments_and_strings(DRIVER), GPU_SIG)
GPU_SQ = squash(GPU_BODY)

FOUR_SITES = [
    ("det", "constdoubledet=xe::xeSiteSub(a,c,b,b,"
            "xe::xeSiteState(host_forms,xe::XE_TXN_DET_BIT));"),
    ("gamma[0]", "gamma[0]=xe::xeSiteSub(c,p,b,q,"
                 "xe::xeSiteState(host_forms,xe::XE_TXN_G0_BIT))/det;"),
    ("gamma[1]", "gamma[1]=xe::xeSiteSub(a,q,b,p,"
                 "xe::xeSiteState(host_forms,xe::XE_TXN_G1_BIT))/det;"),
    ("proj", "proj=xe::xeSiteAdd(gamma[0],p,gamma[1],q,"
             "xe::xeSiteState(host_forms,xe::XE_TXN_PROJ_BIT));"),
]

for name, spelling in FOUR_SITES:
    check(spelling in GPU_SQ,
          "S3a the `%s` site is spelled through xeSite* under host_forms" % name,
          "the scan alone cannot tell a barriered site from a site that was "
          "deleted; this is what says the four are still there")

check("proj=xsrecon::xsrMul(gamma[j],p);" in GPU_SQ,
      "S3b the one-column projection is barriered (`xsrecon::xsrMul(gamma[j], p)`)",
      "one multiply is not a mined SITE, but it feeds `gg - proj` and an "
      "unbarriered product there is a contraction waiting for the next inline")

check("staticconstunsignedlonghosthost_forms=xe::xeHostFormMask();" in
      GPU_SQ.replace("unsignedlonglong", "unsignedlonghost"),
      "S4 the host mask is resolved once into a cached function-local static",
      "a getenv per Anderson step is a cost and a second opinion; the audit "
      "hook's `static const bool audit` is the shape proven neutral on 238")

check(GPU_BODY.count("xe::xeHostFormMask()") == 1,
      "S5 the split arm resolves the host mask exactly once",
      "found %d call(s)" % GPU_BODY.count("xe::xeHostFormMask()"))


# ===========================================================================
# M.  The mask machinery
# ===========================================================================

KERNEL_CODE = strip_comments_and_strings(KERNEL)

DEFAULT_DECL = re.search(
    r"constexpr\s+unsigned\s+long\s+long\s+XE_HOST_FORMS_DEFAULT\s*=(.*?);",
    KERNEL_CODE, re.S)
check(DEFAULT_DECL is not None,
      "M1 XE_HOST_FORMS_DEFAULT is declared in src/XeKernel.h")

FIELDS = re.findall(r"XE_SITE_([A-Z0-9]+)\s*\)?\s*<<\s*XE_TXN_([A-Z0-9]+)_BIT",
                    DEFAULT_DECL.group(1) if DEFAULT_DECL else "")
check(sorted(site for _, site in FIELDS) == ["DET", "G0", "G1", "PROJ"],
      "M2 the default names all four sites -- DET, G0, G1, PROJ -- once each",
      "got %s" % sorted(site for _, site in FIELDS))
check(all(state in ("NONE", "P1", "P2") for state, _ in FIELDS) and len(FIELDS) == 4,
      "M3 every field is one of the three XE_SITE_* states",
      "got %s" % FIELDS)

# --- The pin itself. -------------------------------------------------------
# M2/M3 say the default is WELL FORMED; they would pass just as happily on a
# default that had silently drifted back to a guess.  What follows says WHICH
# well-formed value it is, and it is the one 238 measured on 2026-08-30 (81
# points, one build, env-only sweep, arm X with RASBERY_GPU_XE_TXN unset):
#
#     0xaa0  [1,1,1,1]  e1660d1f4652b49a / 4576 outers   <- the prediction
#     0xac0  [2,1,1,1]  22b9a3187bfb4beb / 4566 outers   <- 7cfe3a4, point 55/81
#
# The det site is the odd one out because its fusable multiply is the
# SUBTRAHEND of `a * c - b * b`: gcc takes the fnma on b*b, not the fma on a*c.
PINNED_SITES = {"DET": "P2", "G0": "P1", "G1": "P1", "PROJ": "P1"}
PINNED_MASK = 0xac0

STATE_VALUE = {"NONE": 0, "P1": 1, "P2": 2}


def default_sites(kernel_code):
    """{site: state} out of the XE_HOST_FORMS_DEFAULT initializer, or {}."""
    decl = re.search(
        r"constexpr\s+unsigned\s+long\s+long\s+XE_HOST_FORMS_DEFAULT\s*=(.*?);",
        kernel_code, re.S)
    if decl is None:
        return {}
    return {site: state for state, site in re.findall(
        r"XE_SITE_([A-Z0-9]+)\s*\)?\s*<<\s*XE_TXN_([A-Z0-9]+)_BIT", decl.group(1))}


def default_mask(kernel_code):
    """The initializer folded into a number, with the SHIFTS read from the
    header too -- so a moved XE_TXN_*_BIT is caught here and not only by the
    device-side contract."""
    sites = default_sites(kernel_code)
    if sorted(sites) != ["DET", "G0", "G1", "PROJ"]:
        return None
    value = 0
    for site, state in sites.items():
        bit = re.search(r"XE_TXN_%s_BIT\s*=\s*(\d+)" % site, kernel_code)
        if bit is None or state not in STATE_VALUE:
            return None
        value |= STATE_VALUE[state] << int(bit.group(1))
    return value


SITES_NOW = default_sites(KERNEL_CODE)
check(SITES_NOW == PINNED_SITES,
      "M3a the default is the 238-MEASURED pin: det=P2, g0=P1, g1=P1, proj=P1",
      "got %s; if 238 re-measured, re-run the 81-point sweep and move BOTH "
      "this pin and src/XeKernel.h together -- a default that moves alone is "
      "the regression this whole file exists to stop" % SITES_NOW)

MASK_NOW = default_mask(KERNEL_CODE)
check(MASK_NOW == PINNED_MASK,
      "M3b the initializer folds to 0x%x -- the value the receipt must print "
      "as build_default" % PINNED_MASK,
      "got %s; the states and the BIT positions are both read from the header, "
      "so this also fails if XE_TXN_*_BIT moved under a correct-looking "
      "initializer" % (MASK_NOW if MASK_NOW is None else "0x%x" % MASK_NOW))

check(squash("static_assert((XE_HOST_FORMS_DEFAULT & ~XE_ALGEBRA_FORMS) == 0ull,")
      in squash(KERNEL_CODE),
      "M4 a static_assert keeps the default inside the algebra channel",
      "a bit below 5 would decide a DEVICE dot/candidate site from a knob "
      "whose name says host")

MASK_CODE = strip_comments_and_strings(MASK_H)
check(re.search(r"unsigned\s+long\s+long\s+xeHostFormMask\s*\(\s*\)\s*;", MASK_CODE)
      is not None,
      "M5 xeHostFormMask is DECLARED in XeFormMask.h")
check(re.search(r"xeHostFormMask\s*\(\s*\)\s*\{", MASK_CODE) is None,
      "M6 XeFormMask.h does not define it",
      "the resolver reads the environment; a header body would put getenv in "
      "every TU that includes it, nvcc's included")

MINER_CODE = strip_comments_and_strings(MINER)
HOST_MASK_BODY = body_of(MINER_CODE, "unsigned long long xeHostFormMask()")
# The same body with its string literals intact, for the rules that are ABOUT a
# literal (the knob's name, the receipt's keys).
HOST_MASK_TEXT = body_of(strip_comments(MINER), "unsigned long long xeHostFormMask()")
check(HOST_MASK_BODY != "",
      "M7 xeHostFormMask is DEFINED in src/XeFormMiner.cpp")

check("mineXeFormsOnThisHost" not in HOST_MASK_BODY and
      "xeFormMask()" not in HOST_MASK_BODY,
      "M8 xeHostFormMask NEVER mines",
      "there is no fixture that can measure Driver.h's own call site "
      "(src/XeFormAudit.h); mining here would answer a different question and "
      "would print a [FORMS] line in a run whose device Xe arm never ran")

check(squash("static const unsigned long long mask = [] {") in squash(HOST_MASK_BODY),
      "M9 the resolution is a cached function-local static",
      "read once per process, or two halves of one solve can disagree after a "
      "mid-run setenv")

check(HOST_MASK_TEXT.count('std::getenv("RASBERY_XE_HOST_FORMS")') == 1,
      "M10 the knob is read from exactly one site in the resolver")

ENV_READS = 0
SRC_DIR = os.path.join(ROOT, "src")
for name in sorted(os.listdir(SRC_DIR)):
    path = os.path.join(SRC_DIR, name)
    if not os.path.isfile(path):
        continue
    ENV_READS += len(re.findall(
        r"getenv\s*\(\s*\"RASBERY_XE_HOST_FORMS\"",
        strip_comments(read("src", name))))
check(ENV_READS == 1,
      "M11 RASBERY_XE_HOST_FORMS has exactly ONE reader in src/",
      "found %d; a second reader is a second contraction contract" % ENV_READS)

check("XE_ALGEBRA_FORMS" in HOST_MASK_BODY,
      "M12 the resolved value is trimmed to the algebra channel",
      "an override with a stray low bit must not reach the shipped "
      "dot/candidate kernels")

check('"mask\\":\\"XE_HOST_FORMS' in MINER.replace('\\"', '\\"') or
      '\\"mask\\":\\"XE_HOST_FORMS\\"' in MINER,
      "M13 the resolver prints one [RASBERY][FORMS] receipt named XE_HOST_FORMS")

for field in ("det", "g0", "g1", "proj"):
    check('\\"%s\\":' % field in MINER,
          "M14 the receipt spells the `%s` site's state as its own field" % field,
          "an 81-point sweep reads the four digits, not the hex")

ARM_ENV = DRIVER[DRIVER.find("inline constexpr const char* kArmEnv[] = {"):]
ARM_ENV = ARM_ENV[:ARM_ENV.find("};")]
check('"RASBERY_XE_HOST_FORMS"' in ARM_ENV,
      "M15 RASBERY_XE_HOST_FORMS is in kArmEnv",
      "it selects the rounding of four expressions on the production arm's own "
      "path, so it moves a trajectory and must fold into the WP10.1 case key -- "
      "otherwise one sweep point's cached answer is served to another's request")


# ===========================================================================
# A.  The audit sees both masks
# ===========================================================================

AUDIT_CODE = strip_comments_and_strings(AUDIT_CPP)
check("xeHostFormMask()" in AUDIT_CODE and "xeFormMask()" in AUDIT_CODE,
      "A1 auditAndersonFit resolves BOTH the device mask and the host mask",
      "B0 for RASBERY_GPU_XE_TXN=1 is now a stateable precondition -- the two "
      "algebra channels equal AND zero mismatches -- and it could not be "
      "stated while the host's spelling was whatever gcc chose")

check("forms_audit_host_mask" in AUDIT_CODE,
      "A2 the audit records the host mask in the tally")
check("forms_audit_host_mask" in RECEIPT,
      "A3 the receipt prints forms_audit_host_mask")
check("XE_ALGEBRA_FORMS" in AUDIT_CODE,
      "A4 the audit compares the two ALGEBRA CHANNELS, not the whole masks",
      "bits 0..4 are the device's dot and candidate and have nothing to say "
      "about the normal equations")


# ===========================================================================
# N.  The pure host arm is the reference and stays raw
# ===========================================================================

HOST_BODY = body_of(strip_comments_and_strings(DRIVER), HOST_SIG)
check(HOST_BODY != "", "N0 TryAndersonXeStep is findable")

RAW_FOUR = [
    "constdoubledet=a*c-b*b;",
    "gamma[0]=(c*p-b*q)/det;",
    "gamma[1]=(a*q-b*p)/det;",
    "proj=gamma[0]*p+gamma[1]*q;",
]
HOST_SQ = squash(HOST_BODY)
for spelling in RAW_FOUR:
    check(spelling in HOST_SQ,
          "N1 the PURE host arm still spells `%s` raw" % spelling,
          "this arm is the frozen MASTER reference, not a choice: barriering it "
          "would MOVE the arm-off baseline the whole campaign is measured "
          "against.  The standing rule -- every host FP expression on a "
          "trajectory-deciding path must be barriered -- applies to paths that "
          "have to REPRODUCE a reference, not to the reference itself")

check("xeHostFormMask" not in HOST_SQ,
      "N2 the pure host arm never reads the host form mask")


# ===========================================================================
# Negative controls.  Each mutation must break the rule it aims at.
# ===========================================================================

GPU_DEF_AT = DRIVER.index(GPU_SIG)


TXN_DEF_AT = DRIVER.index(TXN_SIG)


def mutate(old, new, count=1):
    """Edit only from TryAndersonXeStepGpu's signature on: the pure host arm
    below spells the same names, and a mutation that landed there would make a
    control pass for the wrong reason."""
    return DRIVER[:GPU_DEF_AT] + DRIVER[GPU_DEF_AT:].replace(old, new, count)


def mutate_txn(old, new, count=1):
    """The transaction arm is DEFINED ABOVE TryAndersonXeStepGpu, so `mutate`
    cannot reach it -- a control aimed at it and edited with `mutate` would
    change nothing and pass, which is how a negative control quietly stops
    being one."""
    return DRIVER[:TXN_DEF_AT] + DRIVER[TXN_DEF_AT:].replace(old, new, count)


def mutate_after(text, anchor, old, new, count=1):
    """Replace inside `text` but only from `anchor` on.  XeFormMiner.cpp holds
    two resolvers that spell their cached static identically; a plain replace
    would land on xeFormMask's and leave xeHostFormMask's alone."""
    at = text.index(anchor)
    return text[:at] + text[at:].replace(old, new, count)


NEGATIVES = [
    ("the det site is written back out as raw C",
     lambda: rule_scan(mutate(
         "const double det = xe::xeSiteSub(\n                a, c, b, b, "
         "xe::xeSiteState(host_forms, xe::XE_TXN_DET_BIT));",
         "const double det = a * c - b * b;"), GPU_SIG, ALLOWED_GPU)[0]),

    ("the one-column projection loses its barrier",
     lambda: rule_scan(mutate("proj     = xsrecon::xsrMul(gamma[j], p);",
                              "proj     = gamma[j] * p;"),
                       GPU_SIG, ALLOWED_GPU)[0]),

    ("SAFEGUARD 2/4 grows an unbarriered product",
     lambda: rule_scan(mutate("const double pred2 = gg - proj;",
                              "const double pred2 = gg - gamma[0] * p;"),
                       GPU_SIG, ALLOWED_GPU)[0]),

    ("a new damped blend appears on the accept path",
     lambda: rule_scan(mutate("xe_change = picard;",
                              "xe_change = picard * 0.5 + step;"),
                       GPU_SIG, ALLOWED_GPU)[0]),

    ("the transaction arm starts doing algebra on the host",
     lambda: rule_scan(mutate_txn("value  = out.step;",
                                  "value  = out.step * out.picard + out.gg;"),
                       TXN_SIG, ALLOWED_TXN)[0]),

    ("the mask is resolved per step instead of once",
     lambda: squash("static const unsigned long long mask = [] {") in
     squash(body_of(strip_comments_and_strings(mutate_after(
         MINER, "unsigned long long xeHostFormMask()",
         "static const unsigned long long mask = [] {",
         "const unsigned long long mask = [] {")),
         "unsigned long long xeHostFormMask()"))),

    ("the host resolver starts mining",
     lambda: (lambda b: "mineXeFormsOnThisHost" not in b and "xeFormMask()" not in b)(
         body_of(strip_comments_and_strings(mutate_after(
             MINER, "unsigned long long xeHostFormMask()",
             "unsigned long long value  = XE_HOST_FORMS_DEFAULT;",
             "unsigned long long value  = xeFormMask();")),
             "unsigned long long xeHostFormMask()"))),

    ("the override stops being trimmed to the algebra channel",
     lambda: "XE_ALGEBRA_FORMS" in body_of(strip_comments_and_strings(
         MINER.replace("const unsigned long long kept = value & XE_ALGEBRA_FORMS;",
                       "const unsigned long long kept = value;", 1)
              .replace("outside the algebra channel 0x\"\n                      << XE_ALGEBRA_FORMS",
                       "outside the channel 0x\"\n                      << 0", 1)),
         "unsigned long long xeHostFormMask()")),

    ("the knob is dropped from kArmEnv",
     lambda: (lambda d: '"RASBERY_XE_HOST_FORMS"' in
              d[d.find("inline constexpr const char* kArmEnv[] = {"):
                d.find("};", d.find("inline constexpr const char* kArmEnv[] = {"))])(
         DRIVER.replace('    "RASBERY_XE_HOST_FORMS",\n', "", 1))),

    ("the audit forgets the host mask",
     lambda: "xeHostFormMask()" in strip_comments_and_strings(
         AUDIT_CPP.replace("xeHostFormMask()", "xeFormMask()"))),

    ("the default silently drifts back to the 0xaa0 prediction",
     lambda: default_sites(strip_comments_and_strings(KERNEL.replace(
         "XE_SITE_P2) << XE_TXN_DET_BIT", "XE_SITE_P1) << XE_TXN_DET_BIT", 1)))
     == PINNED_SITES),

    ("the default keeps its four states but a site's BIT moves",
     lambda: default_mask(strip_comments_and_strings(KERNEL.replace(
         "constexpr int XE_TXN_DET_BIT   = 5;",
         "constexpr int XE_TXN_DET_BIT   = 6;", 1))) == PINNED_MASK),

    ("the pure host arm gets barriered too",
     lambda: (lambda d: all(s in squash(body_of(strip_comments_and_strings(d), HOST_SIG))
                            for s in RAW_FOUR))(
         DRIVER[:DRIVER.index(HOST_SIG)] +
         DRIVER[DRIVER.index(HOST_SIG):].replace(
             "const double det = a * c - b * b;",
             "const double det = xe::xeSiteSub(a, c, b, b, 1u);", 1))),
]

for label, probe in NEGATIVES:
    CHECKS += 1
    try:
        still_passes = probe()
    except Exception as exc:  # a counterexample that crashes the rule is not a pass
        still_passes = False
        label += " (rule raised %s)" % type(exc).__name__
    if still_passes:
        FAILURES.append("NEGATIVE CONTROL survived: " + label)


# ---------------------------------------------------------------------------

if FAILURES:
    print("FAIL  tools/test_xe_host_forms_contract.py")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)

print("PASS  tools/test_xe_host_forms_contract.py  (%d checks, %d negative controls)"
      % (CHECKS, len(NEGATIVES)))
print("  every FP expression on the split Xe arm's host path is barriered or")
print("  unfusable, so gcc's inlining context cannot choose its rounding; which")
print("  rounding it IS is XE_HOST_FORMS_DEFAULT, pinned by the 238 sweep.")
