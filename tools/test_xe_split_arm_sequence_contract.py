#!/usr/bin/env python3
"""The split Xe arm's shape, pinned against its pre-71092e2 form.

WHY THIS FILE EXISTS.  `71092e2` (WP7-C) moved the flag-off trajectory --
digest 22b9a3187bfb4beb / 4566 outers became c1a5d9116df9edb3 / 4601 with
RASBERY_GPU_XE_TXN unset -- and it did so without changing one expression the
split arm evaluates.  What it changed was WHERE the split arm is compiled.

src/Driver.h is an implicitly-inline class in ONE translation unit
(src/main.cpp is the only .cpp that includes it; the build has no LTO), and the
Xe step is a chain of static member functions with EXACTLY ONE CALL SITE EACH:

    SolveLoop -> TryAndersonXeStep -> TryAndersonXeStepGpu

so -finline-functions-called-once folds all of them into SolveLoop.  A new
default-off sibling with one call site is therefore not a neighbour of the
production arm; its body is spliced into the hottest function in the tree and
every inlining and -ffp-contract=fast decision already there is re-made around
it.  The split arm's four normal-equation expressions -- `det = a * c - b * b`
and the three below it -- are the unbarriered host arithmetic that feeds the
device gammas, and they are exactly what such a re-make can move.

The shape that is PROVEN neutral on the same host is the audit hook:
`static const bool audit = ...; if (audit) xe::auditAndersonFit(...);`, whose
callee is defined in src/XeFormAudit.cpp -- another translation unit, so the
call is opaque and nothing joins the caller.  `8919331` added it and the
trajectory did not move.  A default-off arm that cannot leave this header buys
the same opacity with RASBERY_NEVER_INLINE, and this file is what keeps it
bought.

WHAT IS PINNED
  A. the never-inline contract, and that the neutral precedent is really
     out of line;
  B. the dispatch shape -- two statements, at the top, and nothing else before
     the split arm's first line;
  C. the split arm's device-call SEQUENCE, in order, once each;
  D. the window bookkeeping -- which column is rotated, which is recorded, and
     when the save happens;
  E. that none of the transaction's machinery (the fused history kernel, the
     device solve, the device gate) is reachable from the split arm;
  F. the four normal-equation expressions, character for character.

Pure source reading: no compiler, no device, no run.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

FAILURES = []
CHECKS = 0


def read(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


DRIVER = read("src/Driver.h")
AUDIT_H = read("src/XeFormAudit.h")
AUDIT_CPP = read("src/XeFormAudit.cpp")
BACKEND_CU = read("src/CudaXsReconBackend.cu")


# ---------------------------------------------------------------------------
# Source helpers.  Comments are stripped before anything is matched: a rule
# that a comment can satisfy is a rule about prose.
# ---------------------------------------------------------------------------

def strip_comments(text):
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
    """The brace-matched body of the definition whose signature contains `anchor`."""
    at = text.find(anchor)
    if at < 0:
        return None
    open_at = text.find("{", at)
    if open_at < 0:
        return None
    depth, i, n = 0, open_at, len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
        i += 1
    return None


def squash(text):
    return re.sub(r"\s+", "", text)


def check(ok, label, detail=""):
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(label + ((" -- " + detail) if detail else ""))
    return bool(ok)


DRIVER_CODE = strip_comments(DRIVER)
BACKEND_CODE = strip_comments(BACKEND_CU)

TXN_SIG = "bool TryAndersonXeStepGpuTxn(SolverContext& ctx"
GPU_SIG = "static bool TryAndersonXeStepGpu(SolverContext& ctx"


# ---------------------------------------------------------------------------
# A.  The never-inline contract
# ---------------------------------------------------------------------------

def rule_macro_defined(driver_code):
    """RASBERY_NEVER_INLINE must actually suppress inlining on the build
    compiler.  An empty definition on gcc would be a comment with a name."""
    m = re.search(r"#\s*if\s+defined\(__GNUC__\)\s*"
                  r"#\s*define\s+RASBERY_NEVER_INLINE\s+([^\n]+)", driver_code)
    if not m:
        return False
    body = squash(m.group(1))
    return "noinline" in body and "__attribute__" in body


def rule_txn_is_never_inline(driver_code):
    """The definition of TryAndersonXeStepGpuTxn carries the attribute."""
    m = re.search(r"static\s+([A-Za-z_][A-Za-z_0-9]*\s+)?bool\s+TryAndersonXeStepGpuTxn\s*\(",
                  driver_code)
    return bool(m) and m.group(1) is not None and m.group(1).strip() == "RASBERY_NEVER_INLINE"


def rule_audit_precedent_is_out_of_line(audit_h, audit_cpp):
    """The proven-neutral hook is only neutral because its body is in another
    TU.  If auditAndersonFit ever gained a body in the header, this file's
    whole argument would have lost its control."""
    head = strip_comments(audit_h)
    declared = re.search(r"\bauditAndersonFit\s*\([^;{]*\)\s*;", head) is not None
    defined_in_header = re.search(r"\bauditAndersonFit\s*\([^;{]*\)\s*\{", head) is not None
    defined_in_cpp = re.search(r"\bauditAndersonFit\s*\([^;{]*\)\s*\{",
                               strip_comments(audit_cpp)) is not None
    return declared and not defined_in_header and defined_in_cpp


check(rule_macro_defined(DRIVER_CODE),
      "A1 RASBERY_NEVER_INLINE expands to __attribute__((noinline, ...)) under __GNUC__",
      "an empty macro is a comment with a name, and gcc would inline the arm anyway")

check(rule_txn_is_never_inline(DRIVER_CODE),
      "A2 TryAndersonXeStepGpuTxn is defined RASBERY_NEVER_INLINE",
      "without it -finline-functions-called-once splices ~110 lines of a "
      "default-off arm into SolveLoop (71092e2)")

check(rule_audit_precedent_is_out_of_line(AUDIT_H, AUDIT_CPP),
      "A3 xe::auditAndersonFit is declared in XeFormAudit.h and DEFINED in "
      "XeFormAudit.cpp",
      "the neutral control for A2 is 'the callee is in another TU'; a header "
      "body would destroy the control")


# ---------------------------------------------------------------------------
# B.  The dispatch shape
# ---------------------------------------------------------------------------

GPU_BODY = body_of(DRIVER_CODE, GPU_SIG)
check(GPU_BODY is not None, "B0 TryAndersonXeStepGpu is findable")
GPU_BODY = GPU_BODY or ""

PROLOGUE, _, SPLIT_BODY = GPU_BODY.partition("XSSet& xs = ctx.cross_sections;")

check(squash(PROLOGUE) ==
      squash("static const bool txn = rasberyGpuXeTxnEnabled();"
             "if (txn && TryAndersonXeStepGpuTxn(ctx, aa, power, max_step, xe_change))"
             "return true;"),
      "B1 the dispatch is exactly two statements at the top of TryAndersonXeStepGpu",
      "anything else before `XSSet& xs = ctx.cross_sections;` is code the "
      "flag-off arm did not have")

check(SPLIT_BODY != "",
      "B2 the split arm still begins at `XSSet& xs = ctx.cross_sections;`")

check(len(re.findall(r"\bTryAndersonXeStepGpuTxn\s*\(", DRIVER_CODE)) == 2,
      "B3 TryAndersonXeStepGpuTxn has exactly one call site (plus its definition)",
      "found %d occurrences" % len(re.findall(r"\bTryAndersonXeStepGpuTxn\s*\(", DRIVER_CODE)))


# ---------------------------------------------------------------------------
# C.  The split arm's device-call sequence
# ---------------------------------------------------------------------------

SEQUENCE = [
    "XeGpuEvaluate",
    "XeGpuRotateHistory",
    "XeGpuRecordColumn",
    "XeGpuSaveEvaluation",
    "XeGpuDots",
    "XeGpuCandidate",
    "XeGpuCommitCandidate",
]


def rule_sequence(split_body):
    positions = []
    for name in SEQUENCE:
        hits = [m.start() for m in re.finditer(r"\bxs\.%s\s*\(" % name, split_body)]
        if len(hits) != 1:
            return False, "%s appears %d time(s), expected 1" % (name, len(hits))
        positions.append(hits[0])
    if positions != sorted(positions):
        order = [SEQUENCE[i] for i in sorted(range(len(positions)), key=lambda k: positions[k])]
        return False, "order is " + " -> ".join(order)
    return True, ""


ok, why = rule_sequence(SPLIT_BODY)
check(ok,
      "C1 the split arm calls evaluate -> rotate -> record -> save -> dots -> "
      "candidate -> commit, once each, in that order", why)

check("xs.XeGpuCommitPicard" not in SPLIT_BODY,
      "C2 the split arm never commits the Picard image itself",
      "a refusal returns false and the CALLER runs UpdateEquilibriumXenon; "
      "committing here is the transaction's semantics, not this arm's")


# ---------------------------------------------------------------------------
# D.  The window bookkeeping -- which column, and when
# ---------------------------------------------------------------------------

def rule_rotate_guard(split_body):
    """A rotation happens only at a FULL window and drops exactly one column."""
    m = re.search(r"if\s*\(\s*aa\.ncol\s*==\s*XE_ANDERSON_DEPTH\s*\)\s*\{"
                  r"(.*?)\}", split_body, re.S)
    if not m:
        return False, "no `if (aa.ncol == XE_ANDERSON_DEPTH)` guard"
    inner = squash(m.group(1))
    if "xs.XeGpuRotateHistory()" not in inner:
        return False, "the guard does not rotate"
    if "--aa.ncol;" not in inner:
        return False, "the rotation does not drop the oldest column (`--aa.ncol;`)"
    return True, ""


ok, why = rule_rotate_guard(SPLIT_BODY)
check(ok, "D1 the rotate is guarded by a FULL window and is followed by --aa.ncol", why)

check(re.search(r"xs\.XeGpuRecordColumn\s*\(\s*aa\.ncol\s*\)", SPLIT_BODY) is not None,
      "D2 the recorded column is `aa.ncol` -- the width BEFORE the increment",
      "a one-column shift in the Anderson history changes accept/reject decisions")

RECORD_AT = SPLIT_BODY.find("xs.XeGpuRecordColumn")
BUMP_AT = SPLIT_BODY.find("++aa.ncol;")
SAVE_AT = SPLIT_BODY.find("xs.XeGpuSaveEvaluation")
HAVE_PREV_AT = SPLIT_BODY.find("aa.have_prev = true;")

check(0 <= RECORD_AT < BUMP_AT < SAVE_AT,
      "D3 `++aa.ncol;` sits between the record and the save",
      "record=%d bump=%d save=%d" % (RECORD_AT, BUMP_AT, SAVE_AT))

check(0 <= SAVE_AT < HAVE_PREV_AT,
      "D4 `aa.have_prev = true;` is set after the save and never before it")


def rule_save_unconditional(split_body):
    """The save must NOT be inside the `if (aa.have_prev)` block: the first
    evaluation of a window has no column to record and still has to be saved,
    or the next step's difference column is built from nothing."""
    m = re.search(r"if\s*\(\s*aa\.have_prev\s*\)\s*\{", split_body)
    if not m:
        return False, "no `if (aa.have_prev)` block"
    depth, i = 0, m.end() - 1
    while i < len(split_body):
        if split_body[i] == "{":
            depth += 1
        elif split_body[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return split_body.find("xs.XeGpuSaveEvaluation") > i, "the save is inside the guard"


ok, why = rule_save_unconditional(SPLIT_BODY)
check(ok, "D5 the save is UNCONDITIONAL -- outside `if (aa.have_prev)`", why)

check(re.search(r"xs\.XeGpuDots\s*\(\s*aa\.ncol\s*,\s*dots\s*\)", SPLIT_BODY) is not None,
      "D6 the dots are asked for `aa.ncol` -- the width AFTER the record")

check(re.search(r"xs\.XeGpuCandidate\s*\(\s*gamma\s*,\s*aa\.ncol\s*,", SPLIT_BODY) is not None,
      "D7 the candidate is built at `aa.ncol`, the same width the dots used")


# ---------------------------------------------------------------------------
# E.  None of the transaction's machinery is reachable from the split arm
# ---------------------------------------------------------------------------

TXN_ONLY_SYMBOLS = [
    "XeGpuTransaction",
    "XeTxnRequest",
    "XeTxnControl",
    "xeAndersonFit",
    "xeAndersonSolveControl",
    "xeAndersonGateControl",
    "xeHistoryOrdinal",
]

for sym in TXN_ONLY_SYMBOLS:
    check(sym not in SPLIT_BODY,
          "E1 the split arm does not mention `%s`" % sym,
          "the fused history kernel and the device solve are TXN-only; the "
          "B0 replay compares the split arm against LIVE code")

TXN_ONLY_KERNELS = ["kXeHistory", "kXeAndersonSolve", "kXeCandidateTxn",
                    "kXeAndersonGate", "kXeCommitTxn"]

TXN_BACKEND_BODY = body_of(BACKEND_CODE, "bool XsReconBackend::xeTransaction(") or ""
check(TXN_BACKEND_BODY != "", "E0 XsReconBackend::xeTransaction is findable")

for kern in TXN_ONLY_KERNELS:
    launches = re.findall(r"\b%s\s*<<<" % kern, BACKEND_CODE)
    inside = re.findall(r"\b%s\s*<<<" % kern, TXN_BACKEND_BODY)
    check(len(launches) == 1 and len(inside) == 1,
          "E2 `%s` is launched only inside XsReconBackend::xeTransaction" % kern,
          "%d launch(es) in the TU, %d of them inside the transaction"
          % (len(launches), len(inside)))

for entry in ["xeEvaluate", "xeDots", "xeCandidate", "xeCommit"]:
    body = body_of(BACKEND_CODE, "bool XsReconBackend::%s(" % entry) or ""
    check(body != "", "E3 XsReconBackend::%s is findable" % entry)
    check(not any(re.search(r"\b%s\s*<<<" % k, body) for k in TXN_ONLY_KERNELS),
          "E4 XsReconBackend::%s launches no transaction kernel" % entry)


# ---------------------------------------------------------------------------
# F.  The four normal-equation expressions, character for character
# ---------------------------------------------------------------------------
#
# These four are the split arm's UNBARRIERED host arithmetic -- no xsrMul, no
# xsrFma, no asm barrier -- so what g++ contracts in them is decided by the
# function they end up inlined into.  That is why B1/A2 exist; pinning the text
# is what makes "the body below is untouched" a check rather than a claim.

FOUR = [
    "const double det = a * c - b * b;",
    "gamma[0] = (c * p - b * q) / det;",
    "gamma[1] = (a * q - b * p) / det;",
    "proj     = gamma[0] * p + gamma[1] * q;",
]

SPLIT_SQUASHED = squash(SPLIT_BODY)
for expr in FOUR:
    check(squash(expr) in SPLIT_SQUASHED,
          "F1 the split arm still spells `%s` verbatim" % expr.strip(),
          "re-spelling it with std::fma or xsrMul pins a different rounding and "
          "moves the baseline the B0 gate is measured against")

check("XE_ANDERSON_MIN_GRAM * a * c" in SPLIT_BODY,
      "F2 the conditioning floor is still `XE_ANDERSON_MIN_GRAM * a * c`",
      "a multiply chain with no add: not a site, and none is mined for it")


# ---------------------------------------------------------------------------
# Negative controls.  Each mutation must break the rule it aims at; a rule that
# survives its own counterexample is not a rule.
# ---------------------------------------------------------------------------

def gpu_body_of(driver_text):
    return body_of(strip_comments(driver_text), GPU_SIG) or ""


def split_of(driver_text):
    _, _, tail = gpu_body_of(driver_text).partition("XSSet& xs = ctx.cross_sections;")
    return tail


def prologue_of(driver_text):
    head, _, _ = gpu_body_of(driver_text).partition("XSSet& xs = ctx.cross_sections;")
    return head


# The split arm's own lines are not unique in this header -- the transaction arm
# above it and the host arm below it spell the same window bookkeeping -- so a
# counterexample edits only the text FROM TryAndersonXeStepGpu's signature on.
# A mutation that landed in the wrong function would make the control pass for
# the wrong reason, which is how a negative control quietly stops being one.
GPU_DEF_AT = DRIVER.index("static bool TryAndersonXeStepGpu(SolverContext")


def mutate(old, new, count=1):
    return DRIVER[:GPU_DEF_AT] + DRIVER[GPU_DEF_AT:].replace(old, new, count)


NEGATIVES = [
    ("the attribute is dropped from the definition",
     lambda: rule_txn_is_never_inline(strip_comments(DRIVER.replace(
         "static RASBERY_NEVER_INLINE bool TryAndersonXeStepGpuTxn",
         "static bool TryAndersonXeStepGpuTxn", 1)))),

    ("the macro is defined empty on gcc",
     lambda: rule_macro_defined(strip_comments(DRIVER.replace(
         "#    define RASBERY_NEVER_INLINE __attribute__((noinline, cold))",
         "#    define RASBERY_NEVER_INLINE", 1)))),

    ("the audit hook grows a body in the header",
     lambda: rule_audit_precedent_is_out_of_line(
         AUDIT_H.replace("auditAndersonFit(", "auditAndersonFit(") +
         "\nnamespace rasbery::xe { void auditAndersonFit(int) { } }\n",
         AUDIT_CPP)),

    ("a statement is inserted before the dispatch",
     lambda: squash(prologue_of(mutate(
         "        static const bool txn = rasberyGpuXeTxnEnabled();",
         "        ctx.telemetry.xe_aa_proposed += 0;\n"
         "        static const bool txn = rasberyGpuXeTxnEnabled();"))) ==
     squash("static const bool txn = rasberyGpuXeTxnEnabled();"
            "if (txn && TryAndersonXeStepGpuTxn(ctx, aa, power, max_step, xe_change))"
            "return true;")),

    ("a second call site of the transaction arm appears",
     lambda: len(re.findall(r"\bTryAndersonXeStepGpuTxn\s*\(", strip_comments(
         mutate("        XSSet& xs = ctx.cross_sections;",
                "        if (false) TryAndersonXeStepGpuTxn(ctx, aa, power, "
                "max_step, xe_change);\n"
                "        XSSet& xs = ctx.cross_sections;")))) == 2),

    ("the recorded column shifts by one",
     lambda: re.search(r"xs\.XeGpuRecordColumn\s*\(\s*aa\.ncol\s*\)",
                       split_of(mutate("xs.XeGpuRecordColumn(aa.ncol)",
                                       "xs.XeGpuRecordColumn(aa.ncol - 1)")))
     is not None),

    ("the dots are asked for the pre-record width",
     lambda: re.search(r"xs\.XeGpuDots\s*\(\s*aa\.ncol\s*,\s*dots\s*\)",
                       split_of(mutate("xs.XeGpuDots(aa.ncol, dots)",
                                       "xs.XeGpuDots(aa.ncol - 1, dots)")))
     is not None),

    ("the rotation stops dropping the oldest column",
     lambda: rule_rotate_guard(split_of(mutate(
         "                --aa.ncol; // the oldest column falls out of the window",
         "")))[0]),

    ("the rotation loses its full-window guard",
     lambda: rule_rotate_guard(split_of(mutate(
         "            if (aa.ncol == XE_ANDERSON_DEPTH) {",
         "            if (true) {")))[0]),

    ("a second save appears ahead of the record",
     lambda: rule_sequence(split_of(mutate(
         "        if (aa.have_prev) {",
         "        if (!xs.XeGpuSaveEvaluation())\n            return false;\n"
         "        if (aa.have_prev) {")))[0]),

    ("the split arm reaches for the transaction",
     lambda: "XeGpuTransaction" not in split_of(mutate(
         "        XSSet& xs = ctx.cross_sections;",
         "        XSSet& xs = ctx.cross_sections;\n"
         "        xe::XeTxnControl out{};\n"
         "        if (xs.XeGpuTransaction(power, req, out)) return true;"))),

    ("one of the four expressions is re-spelled",
     lambda: squash("const double det = a * c - b * b;") in
     squash(split_of(mutate("const double det = a * c - b * b;",
                            "const double det = std::fma(a, c, -(b * b));")))),

    ("the fused history kernel escapes the transaction",
     lambda: (lambda code: len(re.findall(r"\bkXeHistory\s*<<<", code)) == 1 and
              len(re.findall(r"\bkXeHistory\s*<<<",
                             body_of(code, "bool XsReconBackend::xeTransaction(") or "")) == 1)(
         strip_comments(BACKEND_CU.replace(
             "bool XsReconBackend::xeCommit(",
             "void rasberyXeHistoryEscape(double* h, int n) {\n"
             "    kXeHistory<<<1, 1>>>(h, n, 0, 0);\n"
             "}\n\nbool XsReconBackend::xeCommit(", 1)))),
]

NEG_FAILURES = []
for label, probe in NEGATIVES:
    CHECKS += 1
    try:
        still_passes = probe()
    except Exception as exc:  # a counterexample that crashes the rule is not a pass
        still_passes = False
        label += " (rule raised %s)" % type(exc).__name__
    if still_passes:
        NEG_FAILURES.append(label)

for label in NEG_FAILURES:
    FAILURES.append("NEGATIVE CONTROL survived: " + label)


# ---------------------------------------------------------------------------

if FAILURES:
    print("FAIL  tools/test_xe_split_arm_sequence_contract.py")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)

print("PASS  tools/test_xe_split_arm_sequence_contract.py  (%d checks, %d negative controls)"
      % (CHECKS, len(NEGATIVES)))
print("  the split arm is compiled where it was compiled: the transaction arm is")
print("  RASBERY_NEVER_INLINE, the dispatch is two statements, and the history")
print("  sequence and column indices are the pre-71092e2 ones.")
