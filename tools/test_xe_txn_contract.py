#!/usr/bin/env python3
"""Source contract for WP7 stage C -- the Xe device transaction.

WHAT THIS FILE CAN AND CANNOT DECIDE.

It can decide the SHAPE of the change, and the shape is where this work package
can go wrong quietly.  A flag that flipped to default-on; a per-deck decision
block hoisted to file scope so M batch decks overwrite one another's
acceptance; a constant respelled on the device so the two arms condition
differently; a contraction site added to the algebra but not to the mining, so
the mask that is supposed to make the device agree with g++ never learns about
it; the fused history kernel losing the order-preservation note that is the
only reason a reader can check it.  Every one of those either produces a
plausible number or produces one only on a deck nobody runs, and every one is
decidable from the source.

IT CANNOT DECIDE THE ONE CLAIM THAT MATTERS.  "TXN=1 reproduces TXN=0's digest
on the RASBERY_GPU_XE arm" is a 238 GATE, not a source property: it needs nvcc,
the reference deck, and h5diff.  docs/WP7C_XE_TXN_20260831_KO.md carries the
runbook.  What this file does is make sure that when the gate runs, it is
running against the code the doc describes.

NEGATIVE CONTROLS.  Every structural rule is also run against a synthetic
snippet that violates it, so a rule that has quietly stopped matching anything
fails here rather than passing forever.

Pure python, no build, no device.

Run:  python tools/test_xe_txn_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []
checks = 0


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def check(ok: bool, what: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(what)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


KERNEL = read(os.path.join("src", "XeKernel.h"))
BACKEND = read(os.path.join("src", "CudaXsReconBackend.cu"))
BACKEND_H = read(os.path.join("src", "CudaXsReconBackend.h"))
STUB = read(os.path.join("src", "CudaXsReconBackendStub.cpp"))
DRIVER = read(os.path.join("src", "Driver.h"))
RECEIPT = read(os.path.join("src", "XeGpuReceipt.h"))
REFERENCE = read(os.path.join("src", "XeAndersonReference.cpp"))
MINE = read(os.path.join("src", "XeFormMine.h"))
XSSET_H = read(os.path.join("src", "XSSet.h"))
XSSET = read(os.path.join("src", "XSSet.cpp"))
DOC = read(os.path.join("docs", "WP7C_XE_TXN_20260831_KO.md"))


def body_of(text: str, start_marker: str, end_marker: str) -> str:
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    return text[a:b]


# ---------------------------------------------------------------------------
# 1. THE FLAG IS OFF BY DEFAULT AND DECLARED WHERE ARMS ARE DECLARED.
#
# A performance change carrying a bit-identity claim must not be on for anyone
# who did not ask for it: default-on would mean the claim ships unchecked.
# ---------------------------------------------------------------------------
def rule_flag_default_off(backend: str) -> bool:
    return 'envFlagEnabled("RASBERY_GPU_XE_TXN")' in backend


check(
    rule_flag_default_off(BACKEND),
    "RASBERY_GPU_XE_TXN is resolved with envFlagEnabled, i.e. absent means OFF "
    "(envFlagDisabled would make it default-ON)",
)
check(
    "bool rasberyGpuXeTxnEnabled() { return false; }" in STUB,
    "the no-CUDA stub reports the transaction disabled",
)
check(
    "XsReconBackend::xeTransaction(" in STUB,
    "the no-CUDA stub defines xeTransaction (a call site must never need an #ifdef)",
)
check(
    '"RASBERY_GPU_XE_TXN",' in DRIVER,
    "RASBERY_GPU_XE_TXN is listed in Driver.h's kArmEnv -- a knob that can move a "
    "trajectory has to be declared where the receipt reads the arms from",
)


# ---------------------------------------------------------------------------
# 2. THE REFERENCE ARM SURVIVES.
#
# WP7-B's rule, and the reason a B0 replay is worth anything: the code TXN=0
# runs must still be present and still be reached, so the comparison is against
# LIVE CODE and not against a memory of it.
# ---------------------------------------------------------------------------
LEGACY_KERNELS = ("kXeEvaluate", "kXeSub", "kXeDotStage1", "kXeDotStage2",
                  "kXeCandidate", "kXeCommit")
for k in LEGACY_KERNELS:
    check(
        "__global__ void %s(" % k in BACKEND,
        "the reference kernel %s is still defined (TXN=0 launches it)" % k,
    )
for entry in ("xeEvaluate", "xeRotateHistory", "xeRecordColumn", "xeSaveEvaluation",
              "xeDots", "xeCandidate", "xeCommit"):
    check(
        "XsReconBackend::%s(" % entry in BACKEND,
        "the round-tripping entry point %s survives" % entry,
    )

TXN_KERNELS = ("kXeHistory", "kXeAndersonSolve", "kXeCandidateTxn", "kXeAndersonGate",
               "kXeCommitTxn")
for k in TXN_KERNELS:
    check("__global__ void %s(" % k in BACKEND, "the transaction kernel %s exists" % k)


# ---------------------------------------------------------------------------
# 3. THE DISPATCH IS ONE LINE AT THE TOP, AND THE HOST ARM IS UNTOUCHED.
# ---------------------------------------------------------------------------
GPU_ARM = body_of(
    DRIVER,
    "static bool TryAndersonXeStepGpu(SolverContext& ctx, XeAndersonState& aa,",
    "/// One safeguarded Anderson step on the Xe fixed point.",
)
HOST_ARM = body_of(
    DRIVER,
    "static bool TryAndersonXeStep(SolverContext& ctx, XeAndersonState& aa, double power,",
    "static void ResetXeAnderson",
) if "static void ResetXeAnderson" in DRIVER[DRIVER.index(
    "static bool TryAndersonXeStep(SolverContext& ctx, XeAndersonState& aa, double power,"):] else DRIVER[
    DRIVER.index("static bool TryAndersonXeStep(SolverContext& ctx, XeAndersonState& aa, double power,"):]


def rule_dispatch_is_guarded(arm: str) -> bool:
    return "if (txn && TryAndersonXeStepGpuTxn(" in arm


check(
    rule_dispatch_is_guarded(GPU_ARM),
    "the transaction is dispatched from the top of TryAndersonXeStepGpu, guarded by "
    "the cached flag, and the round-tripping body follows it",
)
check(
    "static const bool txn = rasberyGpuXeTxnEnabled();" in GPU_ARM,
    "the flag is read ONCE into a cached bool, not per step",
)
# The PURE-HOST arm's first gate is byte identity with the pre-GPU baseline, so
# it must not have learned about the transaction at all.
HOST_BODY = HOST_ARM[HOST_ARM.index("const double gg = XeDot(aa.g, aa.g);"):][:8000]
check(
    "XeGpuTransaction" not in HOST_BODY and "XE_TXN" not in HOST_BODY,
    "the pure-host Anderson arm knows nothing about the transaction",
)


# ---------------------------------------------------------------------------
# 4. NOTHING PER-DECK IS PROCESS-WIDE.
#
# The transaction's decision block is per-STEP state for one deck.  A file-scope
# one would let M batch decks write one another's accept flag -- the same shape
# as the process-wide slot-0 buffer this tree already paid for once.
# ---------------------------------------------------------------------------
IMPL = body_of(BACKEND, "struct XsReconBackend::Impl {", "XsReconBackend::XsReconBackend()")


def rule_ctl_is_per_impl(impl: str, backend: str) -> bool:
    in_impl = re.search(r"^\s+\S.*\bxe_ctl\b", impl, re.M) is not None
    file_scope = re.search(r"^(static\s+)?\w[\w:<>* ]*\s+xe_ctl\s*(=|\[)", backend, re.M)
    return in_impl and file_scope is None


check(
    rule_ctl_is_per_impl(IMPL, BACKEND),
    "the transaction's control block is a member of Impl (per Driver, per deck) and "
    "has no file-scope definition",
)
check(
    re.search(r"^\s+\S.*\bxe_ctl_host\b", IMPL, re.M) is not None,
    "the download's landing pad is per-Impl too -- a shared one would be M threads "
    "writing one buffer",
)


# ---------------------------------------------------------------------------
# 5. CAPTURE SAFETY (GpuCaptureArbiter.h's rule).
#
# Every allocation the transaction adds must go through RASBERY_CUDA_TRY_ALLOC,
# which opens an AllocWindow.  A bare cudaMalloc on a batch thread while a
# sibling deck is capturing its CMFD graph invalidates that capture -- measured,
# not theorised (GpuCaptureArbiter.h).
# ---------------------------------------------------------------------------
XE_ENSURE = body_of(BACKEND, "    bool xeEnsure() {", "    /// The uploads and the view")


def rule_all_allocs_windowed(section: str) -> bool:
    # Every cudaMalloc must be immediately preceded by the ALLOC macro's open
    # paren.  An unwrapped one is a bare device allocation on a batch thread.
    # Squashed first: the macro and its argument are line-wrapped in places,
    # and a rule that only saw the one-line spelling would pass a bare
    # allocation someone happened to wrap.
    return len(re.findall(r"(?<!TRY_ALLOC\()\bcudaMalloc\(", squash(section))) == 0


check(
    rule_all_allocs_windowed(XE_ENSURE),
    "every cudaMalloc in xeEnsure is wrapped in RASBERY_CUDA_TRY_ALLOC, i.e. in an "
    "AllocWindow the capture arbiter can serialise against",
)
check(
    "cudaDeviceSynchronize" not in body_of(
        BACKEND, "bool XsReconBackend::xeTransaction(", "unsigned long long XsReconBackend::xeEvaluations()"),
    "the transaction takes no DEVICE-wide synchronisation -- that is the call the "
    "arbiter measured killing sibling captures",
)
check(
    "d.stream" in body_of(
        BACKEND, "bool XsReconBackend::xeTransaction(",
        "unsigned long long XsReconBackend::xeEvaluations()"),
    "the transaction runs on the backend's own per-Impl stream, so a batch slot's "
    "work is ordered against its own work and nobody else's",
)


# ---------------------------------------------------------------------------
# 6. THE PER-STEP SYNC CENSUS, COUNTED FROM THE SOURCE.
#
# The doc's before/after table is the reason this work package exists, so it is
# asserted here rather than written down and left to age.  One synchronisation
# per entry point on the round-tripping arm; ONE for the whole transaction.
# ---------------------------------------------------------------------------
TXN_BODY = body_of(BACKEND, "bool XsReconBackend::xeTransaction(",
                   "unsigned long long XsReconBackend::xeEvaluations()")
LEGACY_ENTRIES = {
    "xeEvaluate": body_of(BACKEND, "bool XsReconBackend::xeEvaluate(",
                          "bool XsReconBackend::xeRotateHistory()"),
    "xeDots": body_of(BACKEND, "bool XsReconBackend::xeDots(",
                      "bool XsReconBackend::xeCandidate("),
    "xeCandidate": body_of(BACKEND, "bool XsReconBackend::xeCandidate(",
                           "bool XsReconBackend::xeCommit("),
    "xeCommit": body_of(BACKEND, "bool XsReconBackend::xeCommit(",
                        "// WP7 stage C -- one Xe step as one device transaction"),
}


def syncs(section: str) -> int:
    return section.count("xeSync()") + section.count("cudaStreamSynchronize(")


for name, section in LEGACY_ENTRIES.items():
    check(
        syncs(section) == 1,
        "the round-tripping %s takes exactly one host synchronisation "
        "(the census says four per step across the four entry points)" % name,
    )
check(
    syncs(TXN_BODY) == 1,
    "the transaction takes exactly ONE host synchronisation -- the drain's, which the "
    "caller was paying for anyway",
)
check(
    "4 -> 1" in DOC or "4 → 1" in DOC,
    "the doc states the census this file just counted",
)


# ---------------------------------------------------------------------------
# 7. THE CONSTANTS ARE PASSED, NEVER RESPELLED.
#
# Driver.h owns XE_EQUILIBRIUM_TOLERANCE, XE_ANDERSON_MIN_GRAM and the trust
# region.  A second literal on the device is a second opinion, and it drifts
# silently: the two arms would then condition differently on the same Gram
# matrix and the digest would move for a reason nobody could name.
# ---------------------------------------------------------------------------
def rule_no_respelled_constants(kernel: str) -> bool:
    return ("1.0e-8" not in kernel and "1e-8" not in kernel
            and "1.0e-6" not in kernel and "1e-6" not in kernel)


check(
    rule_no_respelled_constants(KERNEL),
    "XeKernel.h declares neither tolerance nor Gram floor -- they arrive as "
    "parameters from Driver.h",
)
for name in ("eq_tol", "min_gram", "max_step"):
    check("%s" % name in BACKEND_H, "XeTxnRequest carries %s from the host" % name)
check(
    "req.eq_tol   = XE_EQUILIBRIUM_TOLERANCE;" in DRIVER
    and "req.min_gram = XE_ANDERSON_MIN_GRAM;" in DRIVER,
    "the request is filled from Driver.h's own constants",
)
# XeFormMine.h needs the floor at mining time and cannot include Driver.h.  The
# two literals must agree; this is the guard that says so.
mine_floor = re.search(r"XE_MIN_GRAM\s*=\s*([0-9.e+-]+);", MINE)
driver_floor = re.search(r"XE_ANDERSON_MIN_GRAM\s*=\s*([0-9.e+-]+);", DRIVER)
check(mine_floor is not None and driver_floor is not None,
      "both Gram floors are findable")
if mine_floor and driver_floor:
    check(
        float(mine_floor.group(1)) == float(driver_floor.group(1)),
        "XeFormMine.h's XE_MIN_GRAM equals Driver.h's XE_ANDERSON_MIN_GRAM "
        "(%s vs %s)" % (mine_floor.group(1), driver_floor.group(1)),
    )


# ---------------------------------------------------------------------------
# 8. THE FOUR NEW CONTRACTION SITES ARE MINED.
#
# This is the trap WP7-C is most exposed to.  The device TU is built
# --fmad=false; g++ at -O3 with -ffp-contract=fast is not.  Four expressions
# moved from Driver.h onto the device is four chances for the two to round
# differently, and the mask is the only mechanism that closes them.  A site
# added to XeKernel.h but not to the descent's site table would ship as an
# unmeasured guess -- CmfdOuterFormMiner.cpp records what that costs.
# ---------------------------------------------------------------------------
SITE_BITS = ("XE_TXN_DET_BIT", "XE_TXN_G0_BIT", "XE_TXN_G1_BIT", "XE_TXN_PROJ_BIT")
for bit in SITE_BITS:
    check("constexpr int %s" % bit in KERNEL, "%s is declared" % bit)
    check("xe::%s" % bit in MINE, "%s is in the mining descent's site table" % bit)


def rule_bit_count_covers_sites(kernel: str) -> bool:
    m = re.search(r"XE_BIT_COUNT\s*=\s*(\d+)", kernel)
    if not m:
        return False
    top = 0
    for name in SITE_BITS:
        b = re.search(r"constexpr int %s\s*=\s*(\d+)" % name, kernel)
        if not b:
            return False
        top = max(top, int(b.group(1)) + 2)  # two bits per three-state site
    return int(m.group(1)) >= top


check(
    rule_bit_count_covers_sites(KERNEL),
    "XE_BIT_COUNT covers the four new two-bit sites -- the mining's all-ones seed is "
    "built from it, and a stale count would never explore them",
)
check(
    "refAlgebra" in REFERENCE and "refAlgebra" in MINE,
    "the algebra has a verbatim quotation and the mining scores against it",
)


# ---------------------------------------------------------------------------
# 9. THE QUOTATION STILL QUOTES.
#
# XeAndersonReference.cpp is worth exactly nothing once it stops saying what
# Driver.h says.  Compare the four expressions, whitespace-normalised.
# ---------------------------------------------------------------------------
ALGEBRA_EXPRESSIONS = (
    "const double det = a * c - b * b;",
    "gamma[0] = (c * p - b * q) / det;",
    "gamma[1] = (a * q - b * p) / det;",
    "proj     = gamma[0] * p + gamma[1] * q;",
    "gamma[j] = p / a;",
    "proj     = gamma[j] * p;",
)
for expr in ALGEBRA_EXPRESSIONS:
    check(squash(expr) in squash(REFERENCE),
          "the reference still quotes `%s`" % expr.strip())
    check(squash(expr) in squash(GPU_ARM),
          "the production device arm still contains `%s`" % expr.strip())

# The conditioning tests, which the mask can also move through `det`.
for expr in ("det > XE_ANDERSON_MIN_GRAM * a * c", "a > XE_ANDERSON_MIN_GRAM * gg"):
    check(squash(expr) in squash(GPU_ARM),
          "the production device arm still spells the conditioning test `%s`" % expr)


# ---------------------------------------------------------------------------
# 10. THE ORDER-PRESERVATION NOTE (WP7-B's rule, carried forward).
#
# The fused history kernel replaces twelve device-to-device copies and two
# kernels.  A fusion without a written argument for why the order it removed did
# not matter is a fusion nobody can review.
# ---------------------------------------------------------------------------
def rule_history_has_order_note(backend: str) -> bool:
    head = backend[max(0, backend.index("__global__ void kXeHistory(") - 2000):
                   backend.index("__global__ void kXeHistory(")]
    return "ORDER-PRESERVATION NOTE" in head


check(
    rule_history_has_order_note(BACKEND),
    "kXeHistory carries an ORDER-PRESERVATION NOTE above it",
)
check(
    "ORDER-PRESERVATION NOTE" in KERNEL or "ORDER-PRESERVATION" in KERNEL,
    "the shared history body carries the same note where the body itself lives",
)
check(
    "grid-wide" in BACKEND or "grid.sync" in DOC or "grid-wide" in DOC,
    "the doc or the source says why kXeEvaluate and the dot stage-1 are NOT fused",
)


# ---------------------------------------------------------------------------
# 11. THE REJECTED-STEP INVARIANT.
#
# The transaction commits the Picard image itself on a rejection, which is only
# the same image as the fallback's while relax is 1.0.  Two things guarantee it:
# SolveLoop's arming guard, and the backend's refusal of anything else.  Both
# must be present -- the guard alone is an invariant nobody rechecks.
# ---------------------------------------------------------------------------
check(
    "xe_anderson && xe_relax == 1.0 && flux_converged" in DRIVER,
    "SolveLoop still arms the Anderson attempt only at relax == 1.0",
)
check(
    "if (!(req.relax == 1.0)) return false;" in BACKEND,
    "the backend refuses a transaction at any other relax rather than assuming it",
)
check(
    "picard_skip" in BACKEND and "processed[k] == 0u" in BACKEND,
    "the rejected commit still honours UpdateEquilibriumXenon's zero-flux skip",
)


# ---------------------------------------------------------------------------
# 12. THE RECEIPT SAYS WHICH ARM RAN AND WHAT IT COST.
#
# device_updates == 0 with the flag set is the G0 failure the Task 13 receipt
# exists to prevent; txn_steps == 0 with RASBERY_GPU_XE_TXN set is the same
# failure one arm to the left, and txn_declined > 0 means the run is a MIXTURE
# and its census is not a census.
# ---------------------------------------------------------------------------
for field in ("txn_steps", "txn_accepted", "txn_declined", "host_syncs",
              "host_syncs_per_step", "d2h_bytes", "d2h_bytes_per_step",
              "xe_device_steps"):
    check('"%s\\":' % field in RECEIPT,
          "the [XE_GPU] receipt carries %s" % field)
    check("%s" % field in RECEIPT, "%s is declared in the tally" % field)

# Every atomic declared in the tally must be printed: an unprinted counter is a
# measurement nobody can read.
# Two counters are printed under the names the gate scripts already grep for;
# the rest print under their own.  The alias is spelled out rather than
# excused, so a THIRD renamed counter fails here instead of disappearing.
PRINTED_AS = {"aa_proposed": "anderson_proposed", "aa_accepted": "anderson_accepted"}
declared = set(re.findall(r"std::atomic<unsigned long long>\s+(\w+)\{0\}", RECEIPT))
for name in sorted(declared):
    printed = PRINTED_AS.get(name, name)
    check('"%s\\":' % printed in RECEIPT,
          "XeGpuReceipt.h declares %s and prints it as %s" % (name, printed))

check(
    "xe_device_steps.fetch_add" in BACKEND,
    "xe_device_steps is charged in the backend, once per committed step, on BOTH arms "
    "-- the per-step ratios are meaningless if only one arm counts",
)
check(
    BACKEND.count("xe_device_steps.fetch_add") == 2,
    "exactly two charge sites: the round-tripping commit and the transaction",
)


# ---------------------------------------------------------------------------
# 13. THE COUNTERS THE HOST TELEMETRY READS ARE THE SAME COUNTERS.
#
# A flag claiming bit-identity whose receipt moved would be the first thing to
# disbelieve.  On a rejected step TXN=0 charges aa_proposed here and
# xe_updates/device_updates in UpdateEquilibriumXenon; the transaction makes one
# call instead of two, so it charges both from the downloaded reason.
# ---------------------------------------------------------------------------
TXN_ARM = body_of(DRIVER, "static bool TryAndersonXeStepGpuTxn(",
                  "/// The DEVICE arm of the safeguarded Anderson step")
for term in ("xe_updates.fetch_add", "device_updates.fetch_add",
             "aa_proposed.fetch_add", "aa_accepted.fetch_add",
             "ctx.telemetry.xe_aa_proposed", "ctx.telemetry.xe_aa_accepted",
             "RejectXeAnderson(ctx, reason"):
    check(term in TXN_ARM, "the transaction arm charges %s" % term)
check(
    "reset_edges.fetch_add" not in TXN_ARM,
    "the transaction arm charges no reset edges of its own -- one place decides, "
    "which is what makes the two arms' counts identical by construction",
)
for reason in ("condition", "residual", "physics", "step"):
    check('"%s"' % reason in TXN_ARM,
          "the transaction reports the rejection reason `%s`, same word as the "
          "round-tripping arm" % reason)


# ---------------------------------------------------------------------------
# 14. THE XSSet SEAM IS ONE FUNCTION, IN THE Xe SECTION.
# ---------------------------------------------------------------------------
check("XeGpuTransaction" in XSSET_H and "XSSet::XeGpuTransaction" in XSSET,
      "the XSSet seam exists")
XSSET_TXN = body_of(XSSET, "bool XSSet::XeGpuTransaction(", "bool XSSet::XeGpuCommitPicard(")
check("noteMacroXsWrite();" in XSSET_TXN,
      "the transaction announces its macro-XS write, exactly as both commit entry "
      "points do -- it always commits")
check("PrepareXeDeviceCall(power, 1.0," in XSSET_TXN,
      "the view is prepared at relax 1.0, the only relax this path can see")


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each rule, fed a snippet that breaks it, must fail.
# ---------------------------------------------------------------------------
NEGATIVES = [
    ("flag default-on",
     lambda: rule_flag_default_off('static const bool on = envFlagDisabled("RASBERY_GPU_XE_TXN");')),
    ("dispatch not guarded",
     lambda: rule_dispatch_is_guarded("        TryAndersonXeStepGpuTxn(ctx, aa, power);")),
    ("control block hoisted to file scope",
     lambda: rule_ctl_is_per_impl("struct Impl { int nothing; };",
                                  "xek::XeTxnControl* xe_ctl = nullptr;\n")),
    ("bare cudaMalloc",
     lambda: rule_all_allocs_windowed("cudaMalloc(&p, 8);")),
    ("constant respelled on the device",
     lambda: rule_no_respelled_constants("constexpr double MIN_GRAM = 1.0e-8;")),
    ("bit count does not cover the sites",
     lambda: rule_bit_count_covers_sites(
         "constexpr int XE_TXN_DET_BIT = 5;\nconstexpr int XE_TXN_G0_BIT = 7;\n"
         "constexpr int XE_TXN_G1_BIT = 9;\nconstexpr int XE_TXN_PROJ_BIT = 11;\n"
         "constexpr int XE_BIT_COUNT = 5;\n")),
    ("history fusion without an order note",
     lambda: rule_history_has_order_note("__global__ void kXeHistory(int a) {}")),
    ("two syncs in the transaction",
     lambda: syncs("cudaStreamSynchronize(s); cudaStreamSynchronize(s);") == 1),
]
for label, probe in NEGATIVES:
    checks += 1
    try:
        fired = not probe()
    except Exception:
        fired = True
    if not fired:
        failures.append("negative control did not fire: " + label)


# ---------------------------------------------------------------------------
if failures:
    print("FAIL (%d of %d checks)" % (len(failures), checks))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("PASS  tools/test_xe_txn_contract.py  (%d checks, %d negative controls)"
      % (checks, len(NEGATIVES)))
print("  the digest-identity claim (TXN=0 vs TXN=1 on RASBERY_GPU_XE=1) is a 238 gate,")
print("  not a source property: see docs/WP7C_XE_TXN_20260831_KO.md section 6.")
