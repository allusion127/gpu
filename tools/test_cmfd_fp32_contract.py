#!/usr/bin/env python3
"""Static contract for the mixed-precision CMFD inner solve.

RASBERY_GPU_CMFD_FP32 runs the inner BiCGSTAB in FP32 under an FP64 outer
correction (iterative refinement).  It changes the numerical trajectory on
purpose, so it is validated by the Gate A/B numeric gates rather than by the
bit-golden gate -- which makes these five properties the ones a reviewer cannot
check by diffing an h5:

  1. the env gate is read ONCE and cached, so the captured graph topology and
     the kernels inside it can never disagree;
  2. with the gate off, no FP32 code is REACHABLE -- every _f32 launch sits
     behind fp32Active(), and the FP64 enqueue path still launches the FP64
     kernels;
  3. the float operator mirrors are refreshed downstream of every mutation of
     the double operator (host upload, device assembly, per-sweep updls);
  4. the FP32 reductions use a float payload with a DOUBLE accumulator and reuse
     the unmodified double stage-2 folds;
  5. the non-finite fallback to FP64 exists, is env-independent, and invalidates
     the cached graphs;
  6. the fallback ROLLS THE FLUX BACK.  update_solution_f32's non-finite guard is
     per ELEMENT while the flag, the halt and the host valve are per SLOT, so
     without an explicit rollback the flux that survives the valve is a mixture
     of advanced and frozen nodes -- and absorb() clears the flag on it, adopts
     the mirror and lets the run report success with one stale node.  Section 6
     is the guard against that, with negative controls, because it is exactly
     the kind of claim a comment can assert and no h5 diff can check.
"""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "src" / "CudaBICGBackend.cu").read_text(encoding="utf-8-sig")
CUDA_H = (ROOT / "src" / "CudaBICGBackend.h").read_text(encoding="utf-8-sig")
SOLVER = (ROOT / "src" / "BICGSolver.cpp").read_text(encoding="utf-8-sig")
ARM = (ROOT / "src" / "GpuFp32Arm.h").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"cmfd fp32 contract: FAIL: {message}")


def body_after(anchor: str, *, text: str = CUDA) -> str:
    """The brace-matched block that opens at the first '{' after `anchor`."""
    start = text.find(anchor)
    if start < 0:
        fail(f"anchor not found: {anchor!r}")
    open_at = text.find("{", start)
    if open_at < 0:
        fail(f"no block after {anchor!r}")
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    fail(f"unbalanced block after {anchor!r}")
    return ""


def signature(kernel: str) -> str:
    """The parameter list of `__global__ void <kernel>(...)`."""
    anchor = f"__global__ void {kernel}("
    start = CUDA.find(anchor)
    if start < 0:
        fail(f"kernel not found: {kernel}")
    open_at = start + len(anchor) - 1
    depth = 0
    for i in range(open_at, len(CUDA)):
        if CUDA[i] == "(":
            depth += 1
        elif CUDA[i] == ")":
            depth -= 1
            if depth == 0:
                return CUDA[open_at + 1 : i]
    fail(f"unbalanced parameter list for {kernel}")
    return ""


# ---------------------------------------------------------------------------
# 1. The env gate: opt-in, cached once, never consulted per launch.
# ---------------------------------------------------------------------------
#
# WHERE THE KNOB LIVES, AND WHY IT MOVED.  RASBERY_GPU_CMFD_FP32 used to be
# spelled in this .cu and OR-ed against rasbery::fp32::routes(Cmfd) at the gate
# below.  That OR was invisible to src/GpuFp32Arm.h, whose receipt asked
# `armed()` -- the DEVICE-WIDE knob -- and nothing else, so a run with
# RASBERY_GPU_CMFD_FP32=1 and RASBERY_GPU_FP32 unset ran every kernel of the
# inner BiCGSTAB in float and printed `"arm":"fp64","backends":{"cmfd":"fp64"}`.
# The knob is now read once, in the arm header, by cmfdLegacyArm(), and folded
# into armedFor(Cmfd); this gate is a plain routes() call, so the launch site and
# the receipt read THE SAME predicate.
gate = body_after("bool cmfdFp32InnerEnabled()")
if "rasbery::fp32::routes(rasbery::fp32::Backend::Cmfd)" not in gate:
    fail("cmfdFp32InnerEnabled does not route through the arm header")
if "static const bool enabled" not in gate:
    fail("the FP32 env gate is not cached in a function-local static")
# NEGATIVE CONTROL: the gate reads no environment of its own and ORs nothing in.
if "getenv" in gate or "envFlagEnabled(" in gate or "||" in gate:
    fail("the FP32 gate arms from something routes() cannot see -- that is the "
         "defect that made the receipt report cmfd:\"fp64\" over a float solve")
arm_gate = body_after("inline bool cmfdLegacyArm()", text=ARM)
if 'envFlagOn("RASBERY_GPU_CMFD_FP32")' not in arm_gate:
    fail("the FP32 gate must be opt-IN (envFlagOn), not opt-out")
if "static const bool on" not in arm_gate:
    fail("RASBERY_GPU_CMFD_FP32 is not cached in a function-local static")
if ARM.count('"RASBERY_GPU_CMFD_FP32"') != 1:
    fail("RASBERY_GPU_CMFD_FP32 is read more than once in the arm header")
if "cmfdLegacyArm()" not in body_after("inline bool armedFor(Backend which)",
                                       text=ARM):
    fail("armedFor(Cmfd) does not fold the historical RASBERY_GPU_CMFD_FP32 arm")
for spelling in ('getenv("RASBERY_GPU_CMFD_FP32")',
                 'envFlagEnabled("RASBERY_GPU_CMFD_FP32")'):
    if spelling in CUDA:
        fail(f"RASBERY_GPU_CMFD_FP32 is read in the CMFD TU as {spelling} -- "
             "the knob belongs to the arm header, which is what the receipt "
             "reads")
if CUDA.count("cmfdFp32InnerEnabled()") != 2:  # the definition and one call
    fail("the cached FP32 gate is consulted more than once")
if "fp32_inner    = cmfdFp32InnerEnabled();" not in CUDA:
    fail("BatchCore does not resolve the FP32 gate once at construction")
if "bool fp32Active() const { return fp32_inner && !fp32_latched_off; }" not in CUDA:
    fail("fp32Active() is not the single (gate AND NOT latched) predicate")

# ---------------------------------------------------------------------------
# 2. OFF path: every FP32 kernel is reachable only through fp32Active().
# ---------------------------------------------------------------------------
FP32_KERNELS = (
    "refresh_operator_mirror_f32",
    "begin_outer_fused_f32",
    "reduce_dot_stage1_f32",
    "reduce_dot2_stage1_f32",
    "matvec_two_group_f32",
    "colored_block_sweep_f32",
    "prepare_p_jacobi_f32",
    "update_s_jacobi_f32",
    "update_solution_f32",
    "snapshot_flux_f32",
    "restore_flux_f32",
)
for kernel in FP32_KERNELS:
    if f"__global__ void {kernel}(" not in CUDA:
        fail(f"missing FP32 kernel {kernel}")

FP32_ENQUEUE = ("dot_f32", "dot2_f32", "precondition_sweeps_f32", "enqueue_iteration_f32")
guarded = "".join(body_after(f"void {name}(") for name in FP32_ENQUEUE)

outer = body_after("void enqueue_outer(int nmax)")


def fp32_arms(text: str) -> str:
    """EVERY `if (fp32Active()) {` block of the body, concatenated.

    WP20.2 gave enqueue_outer TWO of them and the split is structural rather
    than cosmetic: the operator mirror is refreshed ONCE per outer (the
    refinement loop moves phi, not diag/cc), while the FP32 prologue that
    recomputes the FP64 residual and narrows it runs ONCE PER ROUND.  A rule
    that took only the first block would have stopped seeing the prologue, and
    would have called that a pass."""
    out = []
    at = 0
    while True:
        at = text.find("if (fp32Active()) {", at)
        if at < 0:
            return "".join(out)
        out.append(body_after("if (fp32Active()) {", text=text[at:]))
        at += 1


fp32_arm = fp32_arms(outer)
fp64_arm = body_after("} else {", text=outer)
if "refresh_operator_mirror_f32<<<" not in fp32_arm or "begin_outer_fused_f32<<<" not in fp32_arm:
    fail("the FP32 prologue is not inside the fp32Active() arm of enqueue_outer")
if "begin_outer_fused<<<" not in fp64_arm:
    fail("the FP64 prologue arm no longer launches begin_outer_fused")
if "_f32" in fp64_arm:
    fail("FP32 code leaked into the FP64 arm of enqueue_outer")
if "if (fp32Active())\n                    enqueue_iteration_f32(" not in outer:
    fail("the iteration loop does not select the FP32 body through fp32Active()")
if "enqueue_iteration(allow_halt, force_halt);" not in outer:
    fail("the FP64 iteration body is no longer reachable")

# ---------------------------------------------------------------------------
# WP20.2: THE REFINEMENT LOOP.
#
# The round count is a CAP and not a `while`, because the inner solve is a
# captured graph and its depth is topology.  What this file can settle is the
# structure: the loop exists, it is bounded by the arm's own cached answer, its
# trip count is 1 on every path that is not the refinement arm, the reference
# norm is frozen at round 0, and the round boundary re-establishes the TRUE
# FP64 residual rather than reusing the FP32 recursive one.
# ---------------------------------------------------------------------------
if "for (int round = 0; round < rounds; ++round) {" not in outer:
    fail("enqueue_outer has no refinement round loop")
if "const int rounds = refineRoundsActive();" not in outer:
    fail("the round count does not come from refineRoundsActive()")
if "refineRounds()" in outer:
    fail("enqueue_outer asks the arm header directly; the cap is a cached "
         "member because it fixes the captured graph depth")
if "if (round == 0) {" not in outer:
    fail("round 0 is not distinguished; the reference norm r20 must be frozen "
         "there and never restored by a later round")
store_ref_at = outer.find("store_reference_norm<<<")
round_test_at = outer.find("refine_round_test<<<")
if store_ref_at < 0 or round_test_at < 0 or store_ref_at > round_test_at:
    fail("store_reference_norm must belong to round 0 and precede the round "
         "test that reads the r20 it froze")
if outer.count("reduce_norm_store_reference_stage2<<<") != 1:
    fail("the store-reference stage 2 is launched more than once; a round > 0 "
         "that restored r20 would test the refinement against the residual the "
         "refinement itself produced")
if "refine_round_open<<<full_scalar_grid()" not in outer:
    fail("refine_round_open is not launched at full width; it writes the halt "
         "every kernel of the round consults")

rr_active = body_after("int refineRoundsActive() const")
if "fp32Active()" not in rr_active or "fp32_refine_cap" not in rr_active:
    fail("refineRoundsActive() is not (the FP32 arm AND the cached cap); a "
         "latched arm must collapse to the single-round FP64 topology")
prec = body_after("int precisionTag() const")
if "refineRoundsActive()" not in prec:
    fail("precisionTag() does not fold the round count; two captures that "
         "differ by four nodes may not share one graph instantiation")

open_body = body_after("__global__ void refine_round_open(")
for token, why in (
        ("active[m] == 0u || sweep_halt[m] != 0u",
         "refine_round_open must reproduce initialize_solver_state's "
         "participation test rather than trust counters it never zeroed"),
        ("cm[kRefineDone] != 0u",
         "a slot whose FP64 residual already met eps must not be re-opened"),
        ("NONFINITE_DETECTED",
         "a slot the FP32 attempt broke must not get another FP32 round"),
        ("sm[kRho]         = 1.0;",
         "a new round is a new Krylov space and rho must return to its seed"),
        ("halt[m]       = 0u;",
         "the round has to CLEAR the halt its predecessor raised, or nothing "
         "below it runs")):
    if token not in open_body:
        fail("refine_round_open: " + why)

test_body = body_after("__global__ void refine_round_test(")
if "rnorm / r20 < sm[kEps]" not in test_body:
    fail("the round test is not the FP64 path's own relative test against the "
         "frozen r20 and eps")
if "cm[kRefineDone] = 1u;" not in test_body or "halt[m]         = 1u;" not in test_body:
    fail("a converged round does not halt the slot and mark it done")

# ---------------------------------------------------------------------------
# 4b. The ROUND-COUNT TRANSPORT, and the slot that enters no round at all.
#
# ffd172f states the receipt contract: fp32_refine_cap / _rounds / _solves are
# the two sums and the denominator, UNREDUCED -- "0/0 is a missing measurement,
# not a mean of one" -- and docs/WP20 Sec 9 makes the derived round MEAN the
# decision variable ("Arm B's round mean at 1.0 falsifies Sec 2.4's diagnosis
# and the instruction is to go back to the diagnosis rather than raise the
# cap").  A slot that ran zero refinement rounds must therefore be incapable of
# contributing a round of 1; if it can, the decision variable is read off a
# denominator that includes non-participants.
#
# The three links that make that possible, and why only a SENTINEL closes them:
#   * initialize_solver_state's participation mask (active && !sweep_halt)
#     returns ABOVE the `for (i < kCounterSlots) cm[i] = 0` block, so a masked
#     slot -- e.g. a batch participant whose device-resident CMFD sweep already
#     converged, active[m]=1 and sweep_halt[m]=1 -- carries the PREVIOUS
#     outer's kRefineRounds into this one and enters no round;
#   * finalize_status is deliberately NOT halt-guarded (a masked slot still
#     needs a status) and transports `count + 1`, which is >= 1.0 even from a
#     freshly zeroed counter -- so merely zeroing the two slots would still
#     fabricate exactly the 1.0 that Sec 9.2 reads as falsifying;
#   * BatchCore::absorb folds in every finite positive value it sees.
# ---------------------------------------------------------------------------
SENTINEL = "kRefineRoundsAbsent"


def code_only(text: str) -> str:
    """The block with its `//` comments stripped.

    Every check below runs on this: these kernels carry long rationale comments
    that name the very identifiers being asserted, and a rule a COMMENT can
    satisfy is not a rule.
    """
    return "\n".join(
        line if line.find("//") < 0 else line[: line.find("//")]
        for line in text.split("\n"))


def masked_return(init_body: str) -> str:
    """initialize_solver_state's masked early-return, brace-matched or not."""
    head = init_body.find("if (halt[m] != 0u)")
    tail = init_body.find("double*        sm", head)
    if head < 0 or tail <= head:
        return ""
    return init_body[head:tail]


def refine_transport_problems(text: str) -> list[str]:
    """Every way the round-count transport can lie about a slot that ran none."""
    bad: list[str] = []
    if f"constexpr std::uint32_t {SENTINEL} = 0xFFFFFFFFu;" not in text:
        bad.append(
            "no kRefineRoundsAbsent sentinel is defined next to CounterSlot; "
            "the transport is `count + 1` so every in-band value is >= 1 and "
            "there is no way to say 'this slot entered no round'")

    init_body = code_only(
        body_after("__global__ void initialize_solver_state(", text=text))
    masked = masked_return(init_body)
    if not masked:
        bad.append("initialize_solver_state's participation mask no longer has "
                   "an early return this rule can inspect")
    elif SENTINEL not in masked or "kRefineRounds" not in masked:
        bad.append(
            "initialize_solver_state's masked early return does not stamp "
            "kRefineRounds with kRefineRoundsAbsent; the slot keeps the "
            "previous outer's round index and finalize_status reports it + 1")
    zero_at = init_body.find("for (int i = 0; i < kCounterSlots; ++i)")
    stamp_at = init_body.find(SENTINEL)
    if zero_at >= 0 and stamp_at >= 0 and stamp_at > zero_at:
        bad.append("the sentinel is stamped BELOW the counter zero block, which "
                   "the masked slot never reaches")

    fin_body = code_only(body_after("__global__ void finalize_status(", text=text))
    if "search_residual" not in fin_body:
        bad.append("finalize_status no longer writes search_residual")
    elif SENTINEL not in fin_body:
        bad.append(
            "finalize_status transports the round count without testing "
            "kRefineRoundsAbsent, so a slot that entered no round still "
            "reports count + 1 >= 1.0 instead of NaN")

    absorb_body = code_only(
        body_after("void absorb(const int* active_slots, int count)", text=text))
    if "rr == rr && rr > 0.0" not in absorb_body:
        bad.append("absorb no longer filters the round transport on finite and "
                   "positive; that filter is what turns the NaN back into a "
                   "missing measurement")
    return bad


problems = refine_transport_problems(CUDA)
if problems:
    fail("round-count transport: " + "; ".join(problems))

# NEGATIVE CONTROLS.  A rule that still passes on the code it was written
# against is not a rule, so both limbs are regressed to their pre-fix shape --
# separately, and with the sentinel constant left in place so nothing is
# rejected merely for the constant being gone -- and each must be caught.
_masked_now = masked_return(body_after("__global__ void initialize_solver_state(",
                                       text=CUDA))
_pre_fix_mask = "if (halt[m] != 0u) return;" + chr(10) * 2 + "    "
_regressed_init = CUDA.replace(_masked_now, _pre_fix_mask, 1)
if _regressed_init == CUDA or not refine_transport_problems(_regressed_init):
    fail("the round-count rule does not reject an initialize_solver_state that "
         "returns on the participation mask without stamping the sentinel")

_fin_now = body_after("__global__ void finalize_status(", text=CUDA)
_fin_pre = _fin_now.replace(
    "(refine_rounds > 1 && rounds_index != kRefineRoundsAbsent)",
    "(refine_rounds > 1)",
    1)
_regressed_fin = CUDA.replace(_fin_now, _fin_pre, 1)
if _regressed_fin == CUDA or not refine_transport_problems(_regressed_fin):
    fail("the round-count rule does not reject a finalize_status that reports "
         "count + 1 without testing the absent sentinel")

reachable = guarded + fp32_arm
for kernel in FP32_KERNELS:
    total = CUDA.count(f"{kernel}<<<")
    if total == 0:
        fail(f"{kernel} is never launched")
    if reachable.count(f"{kernel}<<<") != total:
        fail(f"{kernel} is launched outside an fp32Active()-guarded region")

# The FP64 inner loop must still launch the FP64 kernels, untouched.
iteration = body_after("void enqueue_iteration(int allow_halt")
for kernel in ("prepare_p_jacobi<<<", "matvec_two_group<<<", "update_s_jacobi<<<",
               "update_solution<<<", "reduce_dot_stage1<<<"):
    if kernel not in iteration:
        fail(f"the FP64 iteration no longer launches {kernel}")
if "_f32" in iteration:
    fail("FP32 code leaked into the FP64 iteration body")
if "colored_block_sweep<<<" not in body_after("void precondition_sweeps(const double* b"):
    fail("the FP64 colour sweeps were disturbed")

# The FP32 working set must not be allocated when the gate is off.
if "if (fp32_inner) {" not in CUDA:
    fail("the FP32 device allocations are not gated on the env flag")
alloc = body_after("if (fp32_inner) {")
for name in ("diag_f", "cc_f", "dinv_f", "r_f", "r0_f", "p_f", "v_f", "s_f", "t_f",
             "y_f", "z_f"):
    if f"&{name})" not in alloc:
        fail(f"{name} is not allocated inside the gated FP32 block")

# ---------------------------------------------------------------------------
# 3. Mirror freshness: one writer, and it dominates every mutation site.
# ---------------------------------------------------------------------------
refresh = body_after("__global__ void refresh_operator_mirror_f32(")
if "static_cast<float>(dm[" not in refresh or "static_cast<float>(cm[" not in refresh:
    fail("the mirror kernel does not narrow both diag and cc")
# Only the refresh kernel may WRITE the mirrors: every other appearance of
# diag_f/cc_f is a const pointer parameter, an allocation, a free or an argument.
for mirror in ("diag_f", "cc_f"):
    writers = [line.strip() for line in CUDA.splitlines()
               if f"{mirror}[" in line and "const float" not in line]
    if [line for line in writers if not line.startswith(("df[", "cf[", "float*"))]:
        fail(f"{mirror} is written outside refresh_operator_mirror_f32: {writers}")

refresh_at = outer.find("refresh_operator_mirror_f32<<<")
consume_at = outer.find("begin_outer_fused_f32<<<")
if not 0 <= refresh_at < consume_at:
    fail("the mirror refresh does not precede the first FP32 consumer of the outer")

sweeps = body_after("void enqueue_sweeps(int nmax, int unroll)")
assembly_at = sweeps.find("cmfd_assemble_operator_2g<<<")
outer_at = sweeps.find("enqueue_outer(nmax);")
updls_at = sweeps.find("cmfd_updls<<<")
if not 0 <= assembly_at < outer_at:
    fail("the device assembly does not precede the outer that refreshes the mirror")
if not 0 <= outer_at < updls_at:
    fail("cmfd_updls is not inside the sweep loop that re-enters enqueue_outer")
if "for (int sweep = 0; sweep < unroll; ++sweep)" not in sweeps:
    fail("the sweep loop no longer re-enters enqueue_outer after each updls")
# The host-side uploads of diag/cc are likewise upstream of the launch that
# replays enqueue_outer; assert both upload sites still exist so the dominance
# argument keeps naming real code.
# WP20 note: issueSweepUploads grew a `slot_budget` parameter after this file was
# written, and the check went stale silently -- it named a signature no longer in
# the tree, so the dominance argument stopped naming real code.  Both sites are
# now matched on the NAME plus its opening parenthesis, which is what the
# argument actually needs (the site exists and is upstream), and cannot rot again
# when a parameter is added.
for site in ("void issueUploads(const int* active_slots, int count)",
             "void issueSweepUploads(const int* active_slots, int count,"):
    if site not in CUDA:
        fail(f"missing operator upload site {site}")
if "_f32" in body_after("void issueUploads(const int* active_slots, int count)"):
    fail("the FP32 mirrors must not be refreshed from the upload path "
         "(the in-graph refresh is the single site)")

# ---------------------------------------------------------------------------
# 4. Reductions: float payload, double accumulator, unmodified double stage 2.
# ---------------------------------------------------------------------------
for stage1 in ("reduce_dot_stage1_f32", "reduce_dot2_stage1_f32"):
    block = body_after(f"__global__ void {stage1}(")
    if "__shared__ double" not in block:
        fail(f"{stage1} does not fold through a double shared accumulator")
    if "float sum" in block or "__shared__ float" in block:
        fail(f"{stage1} accumulates in float")
    if "static_cast<double>(" not in block:
        fail(f"{stage1} does not widen its float payload before multiplying")
    partition = ("const int chunk = (n + static_cast<int>(gridDim.x) - 1) / "
                 "static_cast<int>(gridDim.x);")
    if partition not in block:
        fail(f"{stage1} changed the deterministic partition")
    if "for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1)" not in block:
        fail(f"{stage1} changed the fixed reduction tree")
for duplicated in ("reduce_dot_stage2_f32", "reduce_dot2_stage2_f32"):
    if f"__global__ void {duplicated}" in CUDA:
        fail("a duplicated FP32 stage 2 exists; the double stage 2 must be reused")
if "reduce_dot_stage2<<<" not in body_after("void dot_f32(const float* a"):
    fail("dot_f32 does not reuse the double stage-2 fold")
if "reduce_dot2_stage2<<<" not in body_after("void dot2_f32(const float* a0"):
    fail("dot2_f32 does not reuse the double stage-2 fold")
fp32_iter = body_after("void enqueue_iteration_f32(int allow_halt")
for scalar_kernel in ("reduce_norm_accumulate_stage2<<<", "accumulate_iteration<<<",
                      "reduce_dot_stage2<<<"):
    if scalar_kernel not in fp32_iter:
        fail(f"the FP32 iteration does not reuse the double scalar kernel {scalar_kernel}")

# The FP64 boundary of the refinement step: the flux stays double and only the
# correction is narrowed.
step = body_after("__global__ void update_solution_f32(")
if "double* __restrict__ phi" not in signature("update_solution_f32"):
    fail("update_solution_f32 does not keep the flux in FP64")
if "const float  dx" not in step:
    fail("the correction is not formed in FP32")
if "phi[base + i] + static_cast<double>(dx)" not in step:
    fail("the correction is not widened before it is accumulated into the flux")
prologue = signature("begin_outer_fused_f32")
if "double* __restrict__ r," not in prologue:
    fail("the FP64 residual is no longer written for the reference norm")
for double_input in ("const double* __restrict__ diag",
                     "const double* __restrict__ cc",
                     "const double* __restrict__ x",
                     "const double* __restrict__ src"):
    if double_input not in prologue:
        fail(f"the FP32 prologue no longer reads the FP64 operand {double_input}")
if "float* __restrict__ dinv_f" not in prologue:
    fail("the FP32 prologue does not narrow the inverted diagonal blocks")
if "reduce_dot_stage1<<<" not in outer:
    fail("the reference norm no longer uses the FP64 reduction over the FP64 residual")

# ---------------------------------------------------------------------------
# 5. The non-finite fallback: env-independent, counted, graph-invalidating.
# ---------------------------------------------------------------------------
# WP20 note: the fallback body moved out of drain() into absorb() when the
# stream-ordered enqueue path stopped synchronising here (Rev.7.1 Task 10 part
# 2).  drain() is now "absorb() plus the sync", so the checks follow the body and
# a separate check holds the delegation -- otherwise this file would be asserting
# against a function that no longer contains the code it names.
drain = body_after("void drain(const int* active_slots, int count)")
if "absorb(active_slots, count);" not in drain:
    fail("drain() no longer delegates the absorb half; the fallback checks below "
         "would be scanning the wrong body")
absorb = body_after("void absorb(const int* active_slots, int count)")
if "++telemetry.fp32_fallbacks;" not in absorb:
    fail("absorb() does not count an FP32 fallback")
if "if (fp32_failed) latchFp32Off();" not in absorb:
    fail("absorb() does not latch the arena back to FP64 after an FP32 failure")
if "nonfinite && fp32_was_active" not in absorb:
    fail("the fallback is not conditioned on the FP32 path having been active")
latch = body_after("void latchFp32Off()")
if "fp32_latched_off = true;" not in latch:
    fail("latchFp32Off does not set the sticky latch")
# WP20 note: the two inline cudaGraphExecDestroy calls became destroyGraphCaches(),
# which drops the plain-solve AND the sweep graph caches (and every keyed
# instantiation in them, which the old pair could not).  The property is "both
# caches are dropped", so it is checked through the function that does it.
if "destroyGraphCaches();" not in latch:
    fail("latchFp32Off does not drop the cached graphs")
caches = body_after("void destroyGraphCaches()")
if caches.count("cudaGraphExecDestroy") != 2 or "outer_graphs.clear();" not in caches \
        or "sweep_graphs.clear();" not in caches:
    fail("destroyGraphCaches does not drop BOTH cached graph sets")
if "std::getenv" in latch or "envFlag" in latch:
    fail("the fallback must be env-independent")
if "atomicOr(flags + m, static_cast<std::uint32_t>(NONFINITE_DETECTED));" not in step:
    fail("update_solution_f32 dropped the non-finite guard")
if step.count("return;") < 2:
    fail("update_solution_f32 must refuse to write on a non-finite result")
launch_outer = body_after("void launch_outer(int nmax)")
if "graph_precision != precisionTag()" not in launch_outer:
    fail("the plain-solve graph is not invalidated when the precision changes")
# WP20 note: the sweep cache became per-bucket and the scalar
# `sweep_graph_precision` became a FIELD OF THE KEY (SweepGraphCapacity::precision),
# consulted through serves().  Same property, better enforced -- a keyed cache
# cannot serve a mismatched entry at all, where the scalar could only invalidate
# the one entry it remembered -- so the check follows the key.
launch_sweeps = body_after("void launch_sweeps(int nmax, int unroll)")
if "serves(nmax, unroll, precisionTag(), lanes)" not in launch_sweeps:
    fail("the sweep graph cache is not keyed on the precision")
if "int precision" not in body_after("struct SweepGraphCapacity", text=CUDA_H):
    fail("SweepGraphCapacity does not carry the precision in its key")
if "precision == want_precision" not in CUDA_H:
    fail("SweepGraphCapacity::serves does not compare the precision exactly")

# ---------------------------------------------------------------------------
# 6. Telemetry and the [PHYSICS_MODE] receipt.
# ---------------------------------------------------------------------------
for field in ("std::uint64_t fp32_active", "std::uint64_t fp32_fallbacks"):
    if field not in CUDA_H:
        fail(f"BackendCounters is missing {field}")
for emitter, name in ((CUDA, "arena"), (SOLVER, "single-instance")):
    for key in ("fp32_active", "fp32_fallbacks"):
        if f'\\"{key}\\":' not in emitter or f"c.{key}" not in emitter:
            fail(f"the {name} BACKEND_COUNTERS dump omits {key}")
if 'telemetry.fp32_active = fp32_inner ? 1u : 0u;' not in CUDA:
    fail("fp32_active reports something other than the declared env gate")
if '", precision=" + (fp32_inner ? "mixed" : "fp64")' not in CUDA:
    fail("the [RASBERY][CUDA] banner does not carry the cmfd precision receipt")
if "BATCH_OCCUPANCY" not in CUDA:
    fail("the batch occupancy line disappeared")
if "_f32" in body_after("void CudaBatchArena::reportBatchOccupancy"):
    fail("BATCH_OCCUPANCY must be unchanged by the precision mode")

# ---------------------------------------------------------------------------
# 6. THE ROLLBACK -- what makes absorb()'s "the failed FP32 attempt is
#    DISCARDED, never accepted" a fact instead of a sentence.
#
# THE DEFECT THIS GUARDS AGAINST.  update_solution_f32 guards per ELEMENT: its
# `return` is one thread's return, so on the failing inner iteration every node
# whose own alpha*y + omega*z stayed finite is still written and only the node
# that overflowed keeps the previous value -- and every earlier iteration of the
# same outer had already been accumulated into phi.  The flag, the halt and the
# host valve are all per SLOT.  So the vector that reached absorb() was a
# MIXTURE; absorb() cleared `nonfinite`, no exception was thrown, adoptFluxMirror
# published it, and the Wielandt loop carried on in FP64 from a flux with one
# stale node.  The run reported success.  Nothing in an h5 diff can see this,
# which is why it is a static rule.
#
# THE RULES BELOW ARE WRITTEN AGAINST A TEXT PARAMETER, not against the module's
# CUDA, so the negative controls at the end can feed them mutations that put the
# defect back and prove each rule actually fires.
# ---------------------------------------------------------------------------
NONFINITE_TEST = "if ((flags[m] & static_cast<std::uint32_t>(NONFINITE_DETECTED)) == 0u) return;"
PARTICIPATION_TEST = "active[m] == 0u || sweep_halt[m] != 0u"


def _block(text: str, anchor: str):
    """body_after() that returns None instead of exiting -- controls need that."""
    start = text.find(anchor)
    if start < 0:
        return None
    open_at = text.find("{", start)
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    return None


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def rollback_problems(cuda: str) -> list:
    problems = []
    snap    = _block(cuda, "__global__ void snapshot_flux_f32(")
    restore = _block(cuda, "__global__ void restore_flux_f32(")
    outer   = _block(cuda, "void enqueue_outer(int nmax)")
    absorb  = _block(cuda, "void absorb(const int* active_slots, int count)")
    alloc   = _block(cuda, "if (fp32_inner) {")

    # -- the snapshot ------------------------------------------------------
    if snap is None:
        problems.append(
            "snapshot_flux_f32 is gone: nothing records the iterate the outer "
            "began with, so there is nothing for the valve to fall back TO")
    else:
        if "phi_entry[base + i] = phi[base + i];" not in _squash(snap):
            problems.append("snapshot_flux_f32 does not copy phi into phi_entry")
        if "HALT_GUARD(halt + m)" not in snap:
            problems.append(
                "snapshot_flux_f32 is not halt-guarded; it must cover exactly "
                "the slots initialize_solver_state let run, or restore_flux_f32 "
                "is handed a lane that never took a snapshot")

    # -- the rollback ------------------------------------------------------
    if restore is None:
        problems.append(
            "restore_flux_f32 is gone: a latched slot's flux is once again the "
            "MIXTURE update_solution_f32's per-element guard leaves behind, and "
            "absorb() adopts it")
    else:
        if "HALT_GUARD" in restore:
            problems.append(
                "restore_flux_f32 is halt-guarded.  The slot it exists for is "
                "PRECISELY the one whose halt the non-finite latch raised, so a "
                "halt guard here skips every slot it is meant to repair and the "
                "kernel becomes a no-op that still looks like a fix")
        if "NONFINITE_DETECTED" not in restore:
            problems.append(
                "restore_flux_f32 does not select on NONFINITE_DETECTED; it must "
                "touch the latched slots and only those, or it would throw away "
                "every healthy slot's outer as well")
        if PARTICIPATION_TEST not in restore:
            problems.append(
                "restore_flux_f32 does not reproduce initialize_solver_state's "
                "participation test.  `flags` is zeroed only for slots that "
                "participate, so a non-participant can still be carrying a stale "
                "NONFINITE_DETECTED -- and it has no snapshot to be restored from")
        if "phi[base + i] = phi_entry[base + i];" not in _squash(restore):
            problems.append("restore_flux_f32 does not write phi back from phi_entry")

    # -- where the two sit in the outer -----------------------------------
    if outer is None:
        problems.append("enqueue_outer not found; the ordering rules are vacuous")
    else:
        snap_at  = outer.find("snapshot_flux_f32<<<")
        rest_at  = outer.find("restore_flux_f32<<<")
        move_at  = outer.find("begin_outer_fused_f32<<<")
        final_at = outer.find("finalize_status<<<")
        if snap_at < 0:
            problems.append("enqueue_outer never takes the entry snapshot")
        if rest_at < 0:
            problems.append("enqueue_outer never rolls the flux back")
        if 0 <= move_at < snap_at:
            problems.append(
                "the snapshot is taken after the first FP32 kernel of the round; "
                "it must precede every kernel that can move phi or it records an "
                "iterate the failed attempt already touched")
        if rest_at >= 0 and 0 <= final_at < rest_at:
            problems.append(
                "the rollback runs after finalize_status.  It has to precede the "
                "status packet, the flux D2H, the Wielandt update and absorb(), "
                "because every one of those is a reader of the failed flux")
        if 0 <= rest_at < snap_at:
            problems.append("the rollback is enqueued before the snapshot it reads")
        arms = fp32_arms(outer)
        for launch in ("snapshot_flux_f32<<<", "restore_flux_f32<<<"):
            if outer.count(launch) != arms.count(launch):
                problems.append(
                    launch + " is enqueued outside an fp32Active() arm, so it "
                    "would change the FP64 topology and break feature-off byte "
                    "identity")

    # -- the buffer --------------------------------------------------------
    if alloc is None or "&phi_entry)" not in alloc:
        problems.append(
            "phi_entry is not allocated inside the gated FP32 block; the FP64 "
            "arm must not pay for it, and the FP32 arm must not run without it")
    if "cudaFree(phi_entry);" not in cuda:
        problems.append("phi_entry is never freed")
    writers = re.findall(r"phi_entry\s*\[[^\]]*\]\s*=[^=]", cuda)
    if len(writers) != 1:
        problems.append(
            "phi_entry has %d writers; snapshot_flux_f32 must be the only one, "
            "or the vector the rollback restores is not the entry iterate"
            % len(writers))

    # -- the sentence that started this ------------------------------------
    if absorb is None:
        problems.append("absorb() not found")
    elif "restore_flux_f32" not in absorb:
        problems.append(
            "absorb()'s safety-valve note no longer names restore_flux_f32 as "
            "what makes 'the failed FP32 attempt is DISCARDED' true.  That "
            "sentence was ornamental once and the cost was a silent wrong answer")
    return problems


ROLLBACK = rollback_problems(CUDA)

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Each mutation of the real source puts one link of the
# rollback back the way the defect had it; the rule that owns that link must
# fire.  A control whose anchor no longer exists is itself a failure, so these
# cannot rot into no-ops.
# ---------------------------------------------------------------------------
SNAP_LAUNCH = "snapshot_flux_f32<<<"
REST_LAUNCH = "restore_flux_f32<<<"
SENTINEL = "@@swapped@@"


def _halt_guard_the_rollback(text: str) -> str:
    """Put a HALT_GUARD back into restore_flux_f32 -- and only there."""
    body = _block(text, "__global__ void restore_flux_f32(")
    if body is None:
        return text
    return text.replace(
        body,
        body.replace("RASBERY_CMFD_SLOT(m);",
                     "RASBERY_CMFD_SLOT(m);\n    HALT_GUARD(halt + m);", 1),
        1)


def _swap_launches(text: str) -> str:
    return (text.replace(SNAP_LAUNCH, SENTINEL)
                .replace(REST_LAUNCH, SNAP_LAUNCH)
                .replace(SENTINEL, REST_LAUNCH))


CONTROLS = (
    ("the rollback launch deleted -- the defect exactly as reported",
     lambda t: t.replace(REST_LAUNCH, "restore_flux_f32_never_launched<<<")),
    ("restore_flux_f32 halt-guarded, so it skips the only slot it exists for",
     _halt_guard_the_rollback),
    ("the latch test dropped, so the rollback would discard healthy outers",
     lambda t: t.replace(NONFINITE_TEST, "")),
    ("snapshot and rollback swapped, so the outer restores before it records",
     _swap_launches),
    ("a second writer of phi_entry",
     lambda t: t.replace("        iter_batch_used      = captured;",
                         "        phi_entry[0] = 0.0;\n"
                         "        iter_batch_used      = captured;")),
    ("phi_entry no longer allocated on the FP32 arm",
     lambda t: t.replace("allocate(reinterpret_cast<void**>(&phi_entry), vec_bytes);", "")),
    ("absorb()'s claim back to a bare assertion with no mechanism named",
     lambda t: t.replace("That is restore_flux_f32's doing",
                         "That is the element guards' doing")),
)

for label, mutate in CONTROLS:
    mutant = mutate(CUDA)
    if mutant == CUDA:
        fail("negative control no longer applies (its anchor is gone): " + label)
    if not rollback_problems(mutant):
        fail("negative control did not fire: " + label)

if ROLLBACK:
    fail("non-finite rollback: " + "; ".join(ROLLBACK))

py_compile.compile(str(Path(__file__).resolve()), doraise=True)
print("cmfd fp32 contract: PASS (%d rollback negative controls, all fired)"
      % len(CONTROLS))
