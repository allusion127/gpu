#!/usr/bin/env python3
"""Source contract for the Rev.7.1 Task 13 device Xe arm (RASBERY_GPU_XE).

WHAT A SOURCE CONTRACT IS FOR HERE.  Most of what can go wrong with this arm
does not show up in a converged answer.  A default that flipped to ON, a
process-wide buffer that two Drivers shared, a safeguard that the device arm
spelled with a different constant than the host arm, a history reset edge that
stopped charging its counter: every one of those either produces a plausible
number or produces one only on a deck nobody happens to run.  So they are
asserted against the SOURCE, where they are decidable.

The numerical claims live elsewhere and are not restated here:

    test/xe_form_probe.cpp        the contraction mask, mined and scored, plus
                                  the fixed partition's cover and its cost
    the campaign gate runs        feature-off byte identity, the Picard path's
                                  bit-exactness, Gate A/B for Anderson
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

failures = []


def check(ok, what):
    if not ok:
        failures.append(what)


def read(name):
    return (SRC / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. The flag is OFF by default, and it is spelled the same everywhere.
# ---------------------------------------------------------------------------
backend = read("CudaXsReconBackend.cu")
stub = read("CudaXsReconBackendStub.cpp")

check(
    'envFlagEnabled("RASBERY_GPU_XE")' in backend,
    "RASBERY_GPU_XE is resolved with envFlagEnabled, i.e. absent means OFF "
    "(envFlagDisabled would make it default-ON)",
)
check(
    "bool rasberyGpuXeEnabled() { return false; }" in stub,
    "the no-CUDA stub reports the arm disabled",
)
for sym in (
    "xeEvaluate",
    "xeRotateHistory",
    "xeRecordColumn",
    "xeSaveEvaluation",
    "xeDots",
    "xeCandidate",
    "xeCommit",
):
    check(
        f"XsReconBackend::{sym}(" in stub,
        f"the no-CUDA stub defines {sym} (a call site must never need an #ifdef)",
    )


# ---------------------------------------------------------------------------
# 2. NOTHING PER-DECK IS PROCESS-WIDE.
#
# The batch bug this tree just fixed was a process-wide slot-0 buffer that every
# Driver adopted.  The Anderson history is exactly the kind of thing that would
# be tempting to hoist into a file-scope pointer, and doing so would silently
# make M decks share one core's inventory.  The history members must therefore
# live in Impl, which is per backend, per XSSet, per Driver, per deck.
# ---------------------------------------------------------------------------
impl_start = backend.index("struct XsReconBackend::Impl {")
impl_end = backend.index("XsReconBackend::XsReconBackend()")
impl = backend[impl_start:impl_end]
for member in ("xe_hist", "xe_processed", "xe_partials", "xe_dots", "xe_hist_fuel"):
    check(
        re.search(r"^\s+\S.*\b%s\b" % member, impl, re.M) is not None,
        f"the Xe arm's {member} is a member of Impl (per Driver), not a global",
    )
    # A file-scope definition of the same name would be the bug in question.
    check(
        re.search(r"^(static\s+)?\w[\w:<>* ]*\s+%s\s*(=|\[)" % member, backend, re.M)
        is None,
        f"{member} has no file-scope definition",
    )

# The only process-wide state the arm may hold is the receipt counters, and
# those are counters rather than buffers -- see XeGpuReceipt.h's header.
for counter in ("g_xe_evaluations", "g_xe_commits"):
    check(
        f"std::atomic<unsigned long long> {counter}" in backend,
        f"{counter} is an atomic counter (the only process-wide state allowed here)",
    )


# ---------------------------------------------------------------------------
# 3. The two Anderson arms carry the SAME safeguards, in the SAME order, with
#    the SAME constants.
#
# The device arm is a sibling of the host arm rather than a shared body, and the
# reason is written out in Driver.h: sharing would move the host's arithmetic
# into a different function and a bit-identity claim does not survive that.  The
# price of that decision is that the eight decisions appear twice, and this is
# what keeps the second copy honest.
# ---------------------------------------------------------------------------
driver = read("Driver.h")


def body(fn):
    start = driver.index(f"static bool {fn}(SolverContext&")
    depth, i, seen = 0, driver.index("{", start), False
    while i < len(driver):
        if driver[i] == "{":
            depth += 1
            seen = True
        elif driver[i] == "}":
            depth -= 1
            if seen and depth == 0:
                return driver[start : i + 1]
        i += 1
    raise AssertionError(fn)


host_arm = body("TryAndersonXeStep")
gpu_arm = body("TryAndersonXeStepGpu")

check("TryAndersonXeStepGpu(ctx, aa, power, max_step, xe_change)" in host_arm,
      "the host arm dispatches to the device arm at its top, in one line, so the "
      "host body below it is untouched")

for reason in ("condition", "residual", "physics", "step"):
    check(
        f'RejectXeAnderson(ctx, "{reason}"' in host_arm
        and f'RejectXeAnderson(ctx, "{reason}"' in gpu_arm,
        f'both Anderson arms charge the "{reason}" rejection',
    )

# The order the four safeguards fire in is part of the contract: a candidate
# that fails two of them must be charged to the same one on both arms, or the
# rejection distribution stops being comparable.
def reject_order(text):
    return re.findall(r'RejectXeAnderson\(ctx, "(\w+)"', text)


check(
    reject_order(host_arm) == reject_order(gpu_arm),
    "the two arms fire their safeguards in the same order "
    f"(host {reject_order(host_arm)} vs device {reject_order(gpu_arm)})",
)

for const in (
    "XE_ANDERSON_MIN_GRAM",
    "XE_EQUILIBRIUM_TOLERANCE",
    "XE_ANDERSON_DEPTH",
):
    check(
        const in host_arm and const in gpu_arm,
        f"both arms use {const} rather than a restated literal",
    )

# The normal equations.  These four expressions ARE the algebra; a
# transcription slip in the copy is not something a converged answer would
# show, because a wrong gamma still produces a candidate that the trust region
# will usually accept.
#
# THE TWO ARMS NO LONGER SPELL THEM THE SAME WAY, AND THAT IS THE FIX RATHER
# THAN A DRIFT (238, 2026-08-31).  The device arm's copy used to be raw C, and
# because src/Driver.h is one translation unit built -O3 -ffp-contract=fast
# with no LTO, which multiply gcc folded into each add was re-decided for every
# inlining context that block landed in.  So the flag-off trajectory was a
# function of the call graph: 048c6c1's `RASBERY_NEVER_INLINE` -- a token that
# touched no expression -- moved it from 22b9a3187bfb4beb / 4566 outers to
# c1a5d9116df9edb3 / 4601.  The device arm's four sites now go through
# xe::xeSiteSub / xe::xeSiteAdd under RASBERY_XE_HOST_FORMS, which are
# barriered on gcc, so no inlining decision can reach them.
#
# The PURE HOST arm keeps the raw spelling: it is the frozen MASTER reference,
# not a choice, and barriering it would move the baseline instead of holding
# it.  What must still be equal is the VALUE, and that is what
# XE_HOST_FORMS_DEFAULT is pinned to deliver -- by the 238 sweep, because no
# fixture can reach a production call site (src/XeFormAudit.h).  The sweep ran
# on 2026-08-30 and the answer is 0xac0 (det=P2, g0=g1=proj=P1): 22b9a3187bfb4beb
# / 4566 outers, the 7cfe3a4 value.  See
# docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md sections 8.4.2-8.4.3, and
# tools/test_xe_host_forms_contract.py M3a/M3b, which own the pin.
for expr in (
    "det = a * c - b * b",
    "gamma[0] = (c * p - b * q) / det",
    "gamma[1] = (a * q - b * p) / det",
):
    check(expr in host_arm,
          f"the pure host arm still computes `{expr}` raw -- it is the reference")

for bit in ("XE_TXN_DET_BIT", "XE_TXN_G0_BIT", "XE_TXN_G1_BIT", "XE_TXN_PROJ_BIT"):
    check(f"xe::xeSiteState(host_forms, xe::{bit})" in gpu_arm,
          f"the device arm's `{bit}` site is selected by the host form mask")

check(gpu_arm.count("xe::xeSiteSub(") == 3 and gpu_arm.count("xe::xeSiteAdd(") == 1,
      "the device arm has exactly three subtraction sites and one addition site")

check("pred2 = gg - proj" in host_arm and "pred2 = gg - proj" in gpu_arm,
      "both arms compute `pred2 = gg - proj` identically -- one subtraction, "
      "and on the device arm `proj` is a barrier output so nothing can fuse it")

# Arming, and the counters that hang off it.
check(
    "if (aa.ncol == 0 || picard < XE_EQUILIBRIUM_TOLERANCE)" in host_arm
    and "if (aa.ncol == 0 || picard < XE_EQUILIBRIUM_TOLERANCE)" in gpu_arm,
    "both arms arm on the same two terms",
)
check(
    "++ctx.telemetry.xe_aa_proposed" in gpu_arm
    and "++ctx.telemetry.xe_aa_accepted" in gpu_arm,
    "the device arm feeds the same per-statepoint telemetry the host arm does",
)

# The history roll happens BEFORE any decision on both arms: a refused step
# still contributes its difference column, or the next one is meaningless.
check(
    gpu_arm.index("aa.have_prev = true") < gpu_arm.index("xe_aa_proposed"),
    "the device arm rolls the history before it arms",
)
check(
    host_arm.index("aa.have_prev = true") < host_arm.index("xe_aa_proposed"),
    "the host arm rolls the history before it arms",
)


# ---------------------------------------------------------------------------
# 4. The history reset edges are charged ONCE, in ONE place, for both arms.
#
# The plan requires the device arm's reset count to match the host's exactly:
# a missed edge leaves the previous map's difference columns alive and fits its
# curvature onto the new map.  What makes that structurally true here is that
# ncol/have_prev live on the host for BOTH arms, so ResetXeAndersonHistory is
# the single decision point -- there is no separate device reset to forget.
# ---------------------------------------------------------------------------
reset_fn = driver[
    driver.index("static void ResetXeAndersonHistory(") : driver.index(
        "/// Charge one rejection"
    )
]
check(
    "++ctx.telemetry.xe_aa_history_resets" in reset_fn
    and "reset_edges.fetch_add(1" in reset_fn,
    "the reset edge charges both the per-statepoint counter and the run receipt, "
    "in the same place",
)
check(
    driver.count("ResetXeAndersonHistory(ctx, xe_aa)") >= 2,
    "the reset is still wired to its edges (damper engagement and cascade re-arm)",
)
check(
    "reset_edges.fetch_add" not in gpu_arm,
    "the device arm does not charge reset edges of its own -- one place decides, "
    "which is what makes the two arms' counts identical by construction",
)


# ---------------------------------------------------------------------------
# 5. The receipt exists, carries the five required fields, and is emitted on
#    BOTH the batch and the serial shutdown path.
# ---------------------------------------------------------------------------
receipt = read("XeGpuReceipt.h")
for field in (
    "xe_updates",
    "device_updates",
    "host_fallbacks",
    "anderson_accept_rate",
    "reset_edges",
):
    check(f'\\"{field}\\":' in receipt or f'"{field}\\":' in receipt or f'\\"{field}' in receipt
          or f'"{field}"' in receipt,
          f"the [XE_GPU] receipt carries {field}")

main = read("main.cpp")
check(
    # Three shutdown paths since WP8: batch, serial, and the long-lived
    # evaluator.  "on and never engaged" must not look like "off" in ANY of them.
    main.count('"[RASBERY][XE_GPU] {"') == 3,
    "the receipt is emitted on every main() shutdown path (batch, serial, evaluator)",
)
check(
    "xe_updates.fetch_add" in read("XSSet.cpp"),
    "xe_updates is charged where the arm is asked for a step, so device_updates + "
    "host_fallbacks sums to it",
)


# ---------------------------------------------------------------------------
# 6. The quotation still quotes.
#
# XeAndersonReference.cpp exists to be scored against the shipped bodies, and it
# is only worth anything while it still says what Driver.h says.  Compare the
# expressions, whitespace-normalised.
# ---------------------------------------------------------------------------
ref = read("XeAndersonReference.cpp")


def squash(text):
    return re.sub(r"\s+", "", text)


dot_expr = ("sum += a_i135[i] * b_i135[i] + a_xe135[i] * b_xe135[i] + "
            "a_xe135m[i] * b_xe135m[i];")
host_dot = ("sum += a.i135[i] * b.i135[i] + a.xe135[i] * b.xe135[i] + "
            "a.xe135m[i] * b.xe135m[i];")
check(squash(dot_expr) in squash(ref),
      "the reference still quotes XeDot's accumulation")
check(squash(host_dot) in squash(driver),
      "Driver.h's XeDot still has the accumulation the reference quotes")

for row in ("i", "x", "m"):
    check(
        squash(f"v{row} -= gamma[j] * f.d_{row}[static_cast<std::size_t>(j) * n + i];")
        in squash(ref),
        f"the reference still quotes the candidate loop's {row} row",
    )
check(
    squash("vi -= gamma[j] * aa.df[j].i135[i];") in squash(driver),
    "Driver.h's candidate loop still has the form the reference quotes",
)


# ---------------------------------------------------------------------------
# 7. The shared body is not a second copy of the fused one.
#
# XeKernel.h must reach the node arithmetic through XsReconKernel.h's own
# functions.  A copy would drift, and the drift would be invisible: both arms
# would still converge, to slightly different answers.
# ---------------------------------------------------------------------------
kernel = read("XeKernel.h")
check(
    "xsr::xsreconImageNode(" in kernel,
    "the device evaluate reaches the fused body's (a)-(d) rather than restating it",
)
check(
    "xsr::xsreconReconstructNode(" in kernel,
    "the device commit reaches the fused body's ReconstructNode rather than "
    "restating it",
)
recon = read("XsReconKernel.h")
check(
    recon.count("0.333333333333333 / tr") == 1,
    "there is exactly one copy of the reconstruct body in the tree",
)
check(
    "xsreconImageNode(v, l, iden, &img, max_change_out)" in recon,
    "the fused node body is composed from the same halves the split arm uses, so "
    "the two cannot diverge",
)


# ---------------------------------------------------------------------------
# 8. The partition is FIXED, and the reduction is not a warp-per-slot fold.
# ---------------------------------------------------------------------------
check(
    "XE_DOT_PARTITIONS_DEFAULT" in kernel and "xeDotPartitionRange" in kernel,
    "the partition count is a named constant and the range comes from one function",
)
check(
    re.search(r"xeDotPartitionRange\(int n, int parts, int p", kernel) is not None,
    "the partition range depends only on (n, parts) -- no launch shape, no "
    "occupancy, no arrival order",
)
for banned in ("__shfl_down_sync", "__shfl_xor_sync", "warpSize"):
    check(
        banned not in backend[backend.index("kXeDotStage1") : backend.index("kXeCommit")],
        f"the Xe reduction uses no {banned}: a warp-level fold would put the "
        "association back in the lane mapping's hands",
    )

print("\n".join("FAIL: " + f for f in failures) if failures else "PASS")
sys.exit(1 if failures else 0)
