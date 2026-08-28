#!/usr/bin/env python3
"""Device outer segment contract (Rev.7.1 Task 9, Sec 3.1/6.13/6.21).

Ten properties, none of which a numerical test or a converged answer can see:

  1. THE ARITHMETIC IS NOT DUPLICATED.  src/CudaOuterGraph.h composes the W2
     enqueue functions and restates none of the CMFD bodies.
  2. THE SEGMENT BODY IS Driver.h's ORDER.  kOuterSegmentPlan and
     kOuterQuantumSteps are the same list, step for step -- checked here on the
     SOURCE (the replay checks it again at runtime), because a reordered device
     outer is a different fixed point and no timing shows it.
  3. EVERY EDGE THE TRANSITION CAN EMIT EXISTS in kPhaseTransitions.  The table
     is read from GpuPhaseScheduler.h, not restated here.
  4. THE TRANSITION KERNEL NEVER WRITES queued_phase / queued_epoch (Sec 5.2),
     and it is the ONLY kernel in the segment that writes DeviceSlotPhase::phase.
  5. THE ESCAPE RANKING IS THE ONE THE PLAN NAMES, in the order it names, with
     the budget LAST -- a converged outer must publish FluxConverged even when it
     is the budget-th one.
  6. THE HALT GATE IS ON EVERY BODY KERNEL, after the slot is resolved and never
     before, and it defaults to nullptr so the pre-Task-9 path is untouched.
  7. THE SEGMENT DOES NOT OBSERVE BETWEEN OUTERS: exactly one stream
     synchronisation, and no D2H inside the per-outer loop.
  8. THE GATE IS OPT-IN: RASBERY_GPU_OUTER, default OFF, and
     RASBERY_GPU_OUTER_SEGMENT_MAX defaults to 8.
  9. STUB PARITY: every symbol the CUDA arm defines has a no-CUDA definition.
 10. THE RECEIPT EXISTS AND IS EMITTED, including on a run that refused every
     segment -- "on and never engaged" must not look like "off".
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST = ROOT / "test"

GRAPH_H = SRC / "CudaOuterGraph.h"
GRAPH_CU = SRC / "CudaOuterGraph.cu"
SCHED = SRC / "GpuPhaseScheduler.h"
KERNEL = SRC / "CudaCmfdOuterKernels.h"
STUB = SRC / "CudaOuterGraphStub.cpp"
DRIVER = SRC / "Driver.h"
MAIN = SRC / "main.cpp"
CMAKE = ROOT / "CMakeLists.txt"
REPLAY = TEST / "outer_state_replay.cpp"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


GRAPH_H_TEXT = read(GRAPH_H)
GRAPH_H_CODE = strip_comments(GRAPH_H_TEXT)
GRAPH_CU_TEXT = read(GRAPH_CU)
GRAPH_CU_CODE = strip_comments(GRAPH_CU_TEXT)
SCHED_CODE = strip_comments(read(SCHED))
KERNEL_CODE = strip_comments(read(KERNEL))
STUB_CODE = strip_comments(read(STUB))
DRIVER_TEXT = read(DRIVER)
DRIVER_CODE = strip_comments(DRIVER_TEXT)
MAIN_CODE = strip_comments(read(MAIN))
CMAKE_TEXT = read(CMAKE)
REPLAY_TEXT = read(REPLAY)


def want(text: str, needle: str, where: str, why: str) -> None:
    if needle not in text:
        problems.append(f"{where}: missing {needle!r} -- {why}")


def body_of(code: str, marker: str, stop: str | None = None) -> str:
    start = code.find(marker)
    if start < 0:
        return ""
    rest = code[start:]
    if stop is not None:
        end = rest.find(stop, len(marker))
        if end > 0:
            return rest[:end]
    return rest


# --- 1. the arithmetic lives in ONE place ------------------------------------
for fn in ("enqueueUpdPsi", "enqueueUpdJnet", "enqueueUpdDhat", "enqueueOuterConvergence",
           "enqueueNodalUpdateConstant"):
    want(GRAPH_CU_CODE, fn + "(", "CudaOuterGraph.cu",
         "the segment body must COMPOSE the W2 enqueue rather than reimplement it")
# Distinctive operands of the CMFD bodies.  Their appearance here would mean the
# arithmetic was copied instead of called.
for token in ("betal", "betar", "fdiff", "jnet_fdm", "cmfdUpdPsiNode", "cmfdUpdDhatSurface"):
    for name, code in (("CudaOuterGraph.h", GRAPH_H_CODE), ("CudaOuterGraph.cu", GRAPH_CU_CODE)):
        if token in code:
            problems.append(f"{name}: contains {token!r} -- the CMFD arithmetic belongs to "
                            "CmfdOuterKernel.h; a second copy will diverge")
# The convergence DECISION must come from the shared body, not be re-derived.
want(GRAPH_CU_CODE, "enqueueOuterConvergence", "CudaOuterGraph.cu",
     "the outer decision must come from cmfdOuterConvergence via the W2 kernel")
# CARRYING the decision's flux_converged is not RE-DERIVING it, and Task 10

# needs the carry: SolveLoop's ladder takes that one bit from the device.  What

# must never appear is the EXPRESSION -- a second copy of

# |prev_inner - eigv| < keff_tol && residual < flux_tol, which is the divergence

# Task 5 exists to prevent.  So the ban is on the arithmetic, not the name.

for _fc_code, _fc_where in ((GRAPH_H_CODE, "CudaOuterGraph.h"),

                            (GRAPH_CU_CODE, "CudaOuterGraph.cu")):

    _fc_hits = (re.search(r"keff_tol\s*&&", _fc_code) is not None or
                re.search(r"fabs\([^)]*prev_inner", _fc_code) is not None or
                re.search(r"abs\([^)]*prev_inner", _fc_code) is not None)
    if _fc_hits:
        problems.append(f"{_fc_where}: re-derives flux_converged.  Sec 6.13's decision is "

                        "cmfdOuterConvergence's; a second copy of that test is the exact "

                        "divergence Task 5 exists to prevent -- carry the decision's bit "

                        "instead")

want(GRAPH_CU_CODE, "decision_out.flux_converged", "CudaOuterGraph.cu",

     "SolveLoop's ladder needs the decision's own flux_converged carried back, or it "

     "would have to recompute it from scalars the device already judged")

# --- 2. the segment body is Driver.h's order ---------------------------------
# Scoped to the kOuterSegmentPlan array itself: the prologue table below it has
# the same shape, and folding the two together would let a step move between them
# without the comparison noticing.
PLAN_BLOCK = body_of(GRAPH_H_CODE, "kOuterSegmentPlan[] = {", "kOuterSegmentPlanCount")
PLAN = re.findall(r'\{"([a-z_]+)",\s*"[^"]*",\s*\d,\s*\d\}', PLAN_BLOCK)
QUANTUM_BLOCK = body_of(SCHED_CODE, "kOuterQuantumSteps[] = {", "kOuterQuantumStepCount")
QUANTUM = re.findall(r'\{"([a-z_]+)",\s*"[^"]*"\}', QUANTUM_BLOCK)
if not QUANTUM:
    problems.append("GpuPhaseScheduler.h: kOuterQuantumSteps could not be parsed")
elif PLAN != QUANTUM:
    problems.append("CudaOuterGraph.h: kOuterSegmentPlan %s does not match "
                    "kOuterQuantumSteps %s -- a device outer that reorders Driver.h's "
                    "steps is a different fixed point, not a faster one" % (PLAN, QUANTUM))
want(GRAPH_H_CODE, "kOuterSegmentPlanCount == kOuterQuantumStepCount", "CudaOuterGraph.h",
     "the static_assert that ties the two lists together")

# The enqueue sequence in the .cu must appear in the plan's order.
SEQ_ORDER = ["enqueueUpdPsi", "enqueue_cmfd_sweep", "enqueueOuterRefreshInputs",
             "enqueueUpdJnet", "enqueue_nodal_drive", "enqueueUpdDhat",
             "enqueueOuterConvergence", "enqueueOuterTransition"]
positions = []
for name in SEQ_ORDER:
    idx = GRAPH_CU_CODE.find(name + "(", GRAPH_CU_CODE.find("for (unsigned int i = 0"))
    if idx < 0:
        problems.append(f"CudaOuterGraph.cu: the segment loop does not issue {name}")
    positions.append(idx)
if all(p >= 0 for p in positions) and positions != sorted(positions):
    problems.append("CudaOuterGraph.cu: the segment loop issues its steps out of order %s -- "
                    "the Sec 6.21 order is the physics" % SEQ_ORDER)

# --- 3. every emitted edge exists in the W1 table ----------------------------
EDGE = re.compile(r"\{DevicePhase::(\w+),\s*DevicePhase::(\w+),")
OUTER_EDGES = {m.group(2) for m in EDGE.finditer(SCHED_CODE) if m.group(1) == "Outer"}
EMITTED = set(re.findall(r"case \d+: return DevicePhase::(\w+);", GRAPH_H_CODE))
if not EMITTED:
    problems.append("CudaOuterGraph.h: outerEmittedPhaseAt has no mapping")
for phase in sorted(EMITTED):
    if phase not in OUTER_EDGES:
        problems.append("outerEmittedPhaseAt lists Outer -> %s, which kPhaseTransitions does "
                        "not contain; the scheduler cannot execute that trajectory (Sec 5.4)"
                        % phase)
# And every phase deviceOuterTransition can WRITE must be listed.
TRANSITION = body_of(GRAPH_H_CODE, "OuterTransition deviceOuterTransition",
                     "RASBERY_GPU_HD inline void outerApplyTransition")
WRITTEN = set(re.findall(r"t\.next_phase\s*=\s*static_cast<unsigned int>\(DevicePhase::(\w+)\)",
                         TRANSITION))
for phase in sorted(WRITTEN):
    if phase not in EMITTED:
        problems.append("deviceOuterTransition writes DevicePhase::%s but outerEmittedPhaseAt "
                        "does not list it, so the replay's edge check never sees it" % phase)

# --- 4. the transition owns the phase word, and nothing else -----------------
for field in ("queued_phase", "queued_epoch"):
    if re.search(r"\." + field + r"\s*=", GRAPH_H_CODE) or \
            re.search(r"\." + field + r"\s*=", GRAPH_CU_CODE):
        problems.append(f"CudaOuterGraph: writes {field} -- classify captures it (Sec 5.2); a "
                        "phase kernel that stamps it re-validates the queue entry it is being "
                        "run from and the slot looks queued forever")
APPLY = body_of(GRAPH_H_CODE, "void outerApplyTransition", "outerEmittedPhaseAt")
want(APPLY, "++p.state_epoch", "outerApplyTransition",
     "a transition must bump state_epoch, INCLUDING on Outer -> Outer: without it "
     "slotAlreadyQueued() stays true and classify refuses to re-queue the slot")
if re.search(r"\.error_code\s*=", APPLY):
    problems.append("outerApplyTransition writes error_code, which carries the SCHEDULER "
                    "fault bits (kSchedFault*, 0x1..0x8).  A DeviceEscape is a small ordinal "
                    "from the same range, so gpuSchedulerFaultName would decode a physics "
                    "escape as a queue fault")
# The W2 convergence kernel must still NOT write the phase; Task 9 is what made
# that split pay off, and a regression there would give the slot two owners.
if re.search(r"\bphases\[[^\]]+\]\.phase\s*=", KERNEL_CODE) or \
        re.search(r"\.phase\s*=\s*static_cast<std::uint8_t>", KERNEL_CODE):
    problems.append("CudaCmfdOuterKernels.h: writes DeviceSlotPhase::phase -- the transition "
                    "belongs to k_outer_transition alone (Sec 5.2)")
if GRAPH_H_CODE.count("outerApplyTransition(") < 2:
    problems.append("CudaOuterGraph.h: k_outer_transition does not call outerApplyTransition, "
                    "so the phase write is not going through the single audited site")

# --- 5. the escape ranking ---------------------------------------------------
RANK = re.findall(r"DeviceEscape::(\w+)\)", TRANSITION)
EXPECTED_RANK = ["NonFinite", "NegativeFlux", "RayleighFallback", "MaterialChanged",
                 "SegmentBudget", "None"]
if RANK != EXPECTED_RANK:
    problems.append("deviceOuterTransition ranks its escapes %s; the contract is %s.  The "
                    "budget MUST be last: a converged outer that happens to be the budget-th "
                    "one has to publish FluxConverged, or the host re-enters, runs one more "
                    "outer and converges again -- a different outer count for the same "
                    "answer, which is the Class-B0-on-trajectory failure this task is gated "
                    "on." % (RANK, EXPECTED_RANK))
# The plan's eight escape reasons must all be reachable somewhere in the design.
for escape in ("FluxConverged", "FluxLimitCycleSample", "FluxStallFatal", "SegmentBudget",
               "NegativeFlux", "RayleighFallback", "NonFinite", "MaterialChanged"):
    if escape not in GRAPH_H_TEXT:
        problems.append(f"CudaOuterGraph.h: the plan names {escape} as a Task 9 escape and it "
                        "does not appear at all")
# The three the CMFD body owns are carried through verbatim, not re-labelled.
want(TRANSITION, "t.escape       = d.escape;", "deviceOuterTransition",
     "FluxConverged / FluxLimitCycleSample / FluxStallFatal are the CPU's own words for "
     "what happened and the search consumes them; re-labelling loses the observation")
# The Rayleigh escape must be mined from the existing latch, not invented.
want(GRAPH_H_TEXT, "kSweepState", "CudaOuterGraph.h",
     "the Rayleigh escape must name the cmfd_wiel_finalize latch it is mined from")
want(GRAPH_H_TEXT, "CmfdSweepIO", "CudaOuterGraph.h",
     "the probe's signals must name the sweep contract they come from")

# --- 6. the halt gate --------------------------------------------------------
GLOBALS = re.findall(r"__global__[^{]*?void\s+(\w+)\s*\(", KERNEL_CODE)
for name in GLOBALS:
    start = KERNEL_CODE.index("void " + name)
    rest = KERNEL_CODE[start:]
    nxt = rest.find("__global__", 1)
    kbody = rest[:nxt] if nxt > 0 else rest
    if "halt" not in kbody:
        problems.append(f"{name}: has no halt gate.  A segment enqueues its whole budget up "
                        "front, so without the gate an outer past the exit still runs and "
                        "moves the trajectory")
        continue
    pad = kbody.find("gpuDispatchIsPadding(")
    slot = kbody.find("queue.slots[logical]")
    halt = kbody.find("halt[slot]")
    if not (0 <= pad < slot < halt):
        problems.append(f"{name}: the halt test must come AFTER gpuDispatchIsPadding and "
                        "AFTER the slot is resolved -- halt is indexed by physical slot, so "
                        "testing it on `logical` reads the wrong tenant at any width below "
                        "the bucket")
    if "halt != nullptr" not in kbody:
        problems.append(f"{name}: does not null-check halt, so every pre-Task-9 call site "
                        "would dereference nothing")
for enq in ("enqueueUpdPsi", "enqueueUpdDtil", "enqueueUpdJnet", "enqueueUpdDhat",
            "enqueueOuterConvergence"):
    if not re.search(re.escape(enq) + r"\([^)]*const std::uint32_t\* halt = nullptr",
                     KERNEL_CODE, re.S):
        problems.append(f"{enq}: the halt parameter must DEFAULT to nullptr, so the "
                        "feature-off path is byte-identical by construction rather than by "
                        "comparison")
want(KERNEL_CODE, "halt", "CudaCmfdOuterKernels.h", "the halt gate")
DHAT = body_of(KERNEL_CODE, "void k_cmfd_upd_dhat", "__global__ void k_cmfd_outer_convergence")
if DHAT and "blockIdx.y" not in DHAT:
    problems.append("k_cmfd_upd_dhat: the halt/padding returns must be UNIFORM over the "
                    "block (they test blockIdx.y), or the counter reduction's __syncthreads "
                    "deadlocks the block")

# --- 7. the segment observes only where a HOST hook forces it -------------
#
# THIS RULE CHANGED IN LINK 2, and the change is the honest description of what
# the segment can do today rather than a relaxation.  The original ban -- no
# sync, no D2H anywhere in the per-outer loop -- encoded "a segment never
# returns to the host between outers".  That is still the goal, and it is still
# what Task 10 delivers.  What link 2 established is that TWO of the eight steps
# are host calls: BICGCMFD::drive rendezvouses and Nodal::drive is host
# arithmetic over host arrays.  A segment containing them cannot enqueue outer
# i+1 before observing outer i, whatever the loop says.
#
# So the contract is now: the segment may synchronise ONLY where a hook that
# declares itself synchronising requires it, and declaring that MUST force the
# budget to 1.  A budget above 1 with a synchronising hook would enqueue outers
# whose halt state the transition has not been able to publish yet -- a
# different trajectory, not a faster one, and exactly the failure the original
# ban was written against.
LOOP = body_of(GRAPH_CU_CODE, "for (unsigned int i = 0", "DeviceOuterSegmentState seg_out")
if not LOOP:
    problems.append("CudaOuterGraph.cu: the per-outer loop could not be located")
for banned in ("cudaDeviceSynchronize", "cudaStreamQuery", "cudaEventSynchronize"):
    if banned in LOOP:
        problems.append("CudaOuterGraph.cu: the per-outer loop calls %s.  Only the two host "
                        "hooks may force an observation, and they do it with a stream "
                        "synchronise on the segment's own stream -- a device-wide or "
                        "polling wait is a rendezvous with everything else too." % banned)
want(GRAPH_H_CODE, "sweep_synchronizes", "CudaOuterGraph.h",
     "a hook that rendezvouses has to SAY so, or the budget silently enqueues outers the "
     "transition cannot have halted yet")
if "m_hooks_synchronize ? 1u : outerSegmentBudget()" not in GRAPH_CU_CODE:
    problems.append("CudaOuterGraph.cu: a synchronising sweep hook does not force the budget "
                    "to 1.  With a hook that drains the stream, outers past the first are "
                    "enqueued against a halt word the transition has not published -- the "
                    "Class-B0-on-trajectory failure the no-observation rule exists to stop")
# Every D2H inside the loop must be one of the three NAMED bridges, and each one
# has to carry its byte counter -- an unnamed copy is the rendezvous creeping
# back in.
for bridge, counter in (("mirror psi to the host", "host_mirror_bytes"),
                        ("mirror dhat to the host", "host_mirror_bytes"),
                        ("download jnet for the nodal drive", "jnet_bridge_bytes")):
    if bridge not in GRAPH_CU_TEXT:
        problems.append(f"CudaOuterGraph.cu: the {bridge!r} bridge is gone; if the copy is "
                        "no longer needed the comment explaining why it was must go too")
    if counter not in GRAPH_CU_CODE:
        problems.append(f"CudaOuterGraph.cu: {bridge!r} is not counted by {counter} -- an "
                        "uncounted transfer is one nobody can argue about")
if GRAPH_CU_CODE.count("cudaMemcpyDeviceToHost") > 7:
    problems.append("CudaOuterGraph.cu: more D2H sites than the three named bridges plus "
                    "the segment's three-copy observation (%d).  Each one is a rendezvous "
                    "and needs a name" % GRAPH_CU_CODE.count("cudaMemcpyDeviceToHost"))
if "cudaGraph" in GRAPH_CU_CODE or "cudaGraph" in GRAPH_H_CODE:
    problems.append("CudaOuterGraph: uses the graph API.  The conditional WHILE wrapper is "
                    "Task 10; Task 9 is the stream-ordered sequence it will capture.")

# --- 8. the gate is opt-in ---------------------------------------------------
want(GRAPH_CU_CODE, '"RASBERY_GPU_OUTER"', "CudaOuterGraph.cu", "the Task 9 feature gate")
want(GRAPH_CU_CODE, '"RASBERY_GPU_OUTER_SEGMENT_MAX"', "CudaOuterGraph.cu",
     "the segment budget flag")
m = re.search(r"kOuterSegmentBudgetDefault\s*=\s*(\d+)u", GRAPH_H_CODE)
if not m:
    problems.append("CudaOuterGraph.h: kOuterSegmentBudgetDefault not found")
elif m.group(1) != "8":
    problems.append("CudaOuterGraph.h: the W0-informed segment budget default is 8, found %s"
                    % m.group(1))
for arm, code in (("CudaOuterGraph.cu", GRAPH_CU_CODE),
                  ("CudaOuterGraphStub.cpp", STUB_CODE)):
    fn = body_of(code, "bool outerGpuEnabled()", "unsigned int outerSegmentBudget()")
    if fn and "return false" not in fn and "EnvFlagOn" not in fn and "envFlagOn" not in fn:
        problems.append(f"{arm}: outerGpuEnabled does not resolve through the shared "
                        "truthiness rule")
# Driver.h's branch has to fold away with the feature off.
want(DRIVER_CODE, "gpu::outerGpuEnabled()", "Driver.h",
     "the delegation branch must be gated on the feature")
want(DRIVER_CODE, "gpu::rasberyOuterSegment().runSegment(", "Driver.h",
     "the delegation branch must actually call the runner")
want(DRIVER_CODE, "gpu::outerDeckHasFractionalRods(", "Driver.h",
     "Stage A eligibility must use the mined cusping predicate")
want(DRIVER_CODE, "gpu::noteOuterSegmentRefusal(", "Driver.h",
     "a hoisted eligibility decision must still record its refusal, or the receipt cannot "
     "tell 'off' from 'on and refused every time'")
# --- 8b. the delegation sits on the loop whose escape set has a host resume ---
#
# ReconvergeFlux holds every feedback fixed, so a segment there exits only on
# outcomes Driver.h can resume from exactly.  SolveLoop cannot: a segment ends on
# the outer whose decision was NOT a requeue, that outer's BODY has already run,
# and resuming means re-entering SolveLoop PAST its control ladder -- an entry
# point that does not exist.  The plan's Task 9 Step 5b (host_numeric_calls == 0
# across SolveLoop) is therefore a Task 10 gate in practice, and this check makes
# the deviation a stated one instead of a silent one.
RECONVERGE = body_of(DRIVER_CODE, "static void ReconvergeFlux(", "static void SolveLoop(")
SOLVELOOP = DRIVER_CODE[DRIVER_CODE.find("static void SolveLoop("):]
if "rasberyOuterSegment().runSegment(" not in RECONVERGE:
    problems.append("Driver.h: ReconvergeFlux does not delegate.  It is the only loop in "
                    "Driver.h whose escape set is closed -- every feedback is held fixed -- "
                    "so it is the only Stage A call site that has an exact host resume")
# TASK 10 INVERTED THIS.  Task 9 refused to delegate in SolveLoop because the
# outer whose decision ends the segment has already had its body run and
# SolveLoop had no entry point past the body.  That was a statement about the
# LOOP, not the physics: the only thing the ladder needs from the body is
# flux_converged and the three scalars behind it, so hoisting one declaration is
# the entry point.  The delegation is now required, and what has to be guarded
# is that it stays a hoist rather than a rewrite.
if "rasberyOuterSegment().runSegment(" not in SOLVELOOP:
    problems.append("Driver.h: SolveLoop does not delegate.  Task 10's whole M1 claim is "
                    "device_outers == outer count, and ReconvergeFlux is the search "
                    "FALLBACK -- a deck that never falls back would run zero device outers")
want(DRIVER_TEXT, "WHY THE CALL SITE IS HERE AND NOT IN SolveLoop", "Driver.h",
     "the deviation from the plan's Task 9 Step 5b has to be named where the call site is, "
     "or a reader finds Step 5b unmet and no reason for it")
# The loop bound is charged for EVERY device outer before any exit is taken.
_charge = RECONVERGE.find("i += static_cast<int>(seg.device_outers) - 1;")
_converged = RECONVERGE.find("DeviceEscape::FluxConverged")
if _charge < 0:
    problems.append("Driver.h: the delegation does not advance the loop counter by the "
                    "outers the device took")
elif 0 <= _converged < _charge:
    problems.append("Driver.h: an exit is taken before the loop counter is charged for the "
                    "device outers.  max_iter is a hard bound on OUTERS, so an arm that "
                    "skipped the charge would carry a larger remaining budget than the OFF "
                    "arm at the same point in the solve -- a trajectory difference with no "
                    "iterate moving")

# The host loop must remain the reference path: the delegation is an ADDITION,
# never a replacement, so the host outer body still has to be there verbatim.
for anchor in ("ctx.cmfd_solver.updpsi(ctx.geometry.Phif());",
               "ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);",
               "ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());"):
    if DRIVER_CODE.count(anchor) < 2:
        problems.append("Driver.h: the host outer body no longer appears in both "
                        "ReconvergeFlux and SolveLoop -- the host loop is the reference path "
                        "and the delegation is an addition to it, never a replacement")

# --- 8c. the SolveLoop delegation is a HOIST, not a rewrite (Task 10) -------
#
# The ladder is the most B0-sensitive code in the tree.  The delegation is
# allowed to skip the BODY and nothing else, so the body has to still be there,
# guarded, and the ladder has to be untouched.
if "if (!outer_on_device) {" not in SOLVELOOP:
    problems.append("Driver.h: the host outer body is not guarded by outer_on_device.  The "
                    "delegation must SKIP the body, not replace it -- the body is the "
                    "reference every B0 comparison is against")
for anchor_line in ("ctx.cmfd_solver.updpsi(ctx.geometry.Phif());",
                    "ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);",
                    "ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());"):
    if anchor_line not in SOLVELOOP:
        problems.append(f"Driver.h: SolveLoop no longer contains {anchor_line!r}; the host "
                        "body is the fallback and the reference, and it must survive verbatim")
# The ladder must NOT consume the device escape: the device machine advances its
# own flux_stall/stall_events/clean_iters inside cmfdOuterConvergence and the
# ladder advances the host's.  Consuming both double-counts.
_deleg = body_of(SOLVELOOP, "if (gpu_outer_armed) {", "if (!outer_on_device) {")
if _deleg and "seg.escape" in _deleg:
    problems.append("Driver.h: SolveLoop consumes seg.escape.  In this mode the device "
                    "decision is advisory -- the host ladder owns flux_stall, stall_events "
                    "and clean_iters, and the device copies drift -- so only flux_converged "
                    "and the three carried scalars may cross")
if _deleg and "seg.flux_converged" not in _deleg:
    problems.append("Driver.h: the delegation does not adopt the device flux_converged, so "
                    "the ladder would run on the hoisted false")
# A SEARCH DECK IS NO LONGER REFUSED, and the two places that ask have to agree.
#
# Task 9 refused it because the DEVICE decision stood in for Driver.h's search
# terms.  SolveLoop stopped consuming that decision in Task 10 -- it takes
# flux_converged, which is computed before any search term is read -- so the
# reason cannot reach the answer there.  This matters because the production
# workload is a boron-search deck from statepoint 1: with the refusal in place
# APR1400/kngr_238 never saw a single device outer.
#
# BOTH CALL SITES OR NEITHER.  The hoisted eligibility check and runSegment's
# own re-check ask the same predicate; giving them different answers arms the
# loop and then refuses every individual segment, which is the worst of the two
# behaviours and shows up only as device_outers == 0 with a full refusal count.
_SL_REFUSAL = body_of(SOLVELOOP, "gpu::rasberyOuterSegment().refusal(", ");")
_SL_RUN = body_of(SOLVELOOP, "gpu::rasberyOuterSegment().runSegment(", ") {")
if _SL_REFUSAL and "has_search" in _SL_REFUSAL:
    problems.append("Driver.h: SolveLoop still refuses a critical-search deck.  The device "
                    "decision is advisory there -- only flux_converged crosses, and it is "
                    "computed before any search term is read -- while the production deck "
                    "searches boron from statepoint 1, so the refusal costs every device "
                    "outer on the workload M1 is measured against")
if _SL_RUN and "has_search" in _SL_RUN:
    problems.append("Driver.h: SolveLoop's runSegment call still passes has_search while "
                    "the hoisted check does not.  runSegment re-asks the same predicate, so "
                    "disagreeing arms the loop and then refuses every segment inside it")
# The macro-XS a search commit moves must be covered by the per-outer syncs,
# which is what replaced the refusal.
for _need in ("upload xsnf", "upload dtil"):
    want(GRAPH_CU_CODE, _need, "CudaOuterGraph.cu",
         "a search commit moves the macro-XS between outers; the per-outer sync is what "
         "makes the segment safe on a search deck now that it is not refused")

# --- 8d. every device buffer the segment READS is synced first (Task 10) ----
#
# TWO BUGS FOUND THIS WAY, both the same shape: the segment reads a device
# buffer whose refresh belongs to the sweep and happens mid-outer.  dhat is
# written by the segment at step 8 and read by the sweep at step 2, so the first
# outer after arming read a dhat nobody had written (k_eff = -0.034501,
# negative_flux on 447 of 516 outers).  xsnf is refreshed inside drive(), which
# is step 2, but updpsi is step 1 -- so the fission source was built from the
# previous outer's cross sections.
want(GRAPH_CU_CODE, "seed residency", "CudaOuterGraph.cu",
     "the device dhat/psi must be seeded from the host at arm time, or the first sweep "
     "after arming reads a buffer the segment has not written yet")
want(GRAPH_CU_CODE, "upload xsnf", "CudaOuterGraph.cu",
     "updpsi runs BEFORE the sweep refreshes xs_xsnf, so the segment must sync it or "
     "build the fission source from the previous outer's cross sections")
want(GRAPH_CU_CODE, "upload flux", "CudaOuterGraph.cu",
     "drive() takes the host loop during the Wielandt warm-up, after which the device phi "
     "is behind Geometry::Phif")
# And the control packet has to be reset, not just stamped.
if "deviceSlotStateReset" not in GRAPH_H_CODE:
    problems.append("CudaOuterGraph.h: k_outer_seed_slot does not reset DeviceSlotState.  The "
                    "arena block comes back from cudaMallocFromPoolAsync with whatever was "
                    "in those pages, clearSlotAsync cannot reach the control structs, and "
                    "cmfdOuterConvergence BRANCHES on flux_stall/stall_events/clean_iters -- "
                    "garbage there publishes prev_inner = eigv + 1.0 and SolveLoop adopts it")

# --- 9. stub parity ----------------------------------------------------------
CU_SYMBOLS = ["outerGpuEnabled", "outerSegmentBudget", "outerSegmentCounters",
              "outerSegmentReceiptJson", "reportOuterSegment", "noteOuterSegmentRefusal",
              "CudaOuterSegment::CudaOuterSegment", "CudaOuterSegment::~CudaOuterSegment",
              "CudaOuterSegment::initialize", "CudaOuterSegment::release",
              "CudaOuterSegment::available", "CudaOuterSegment::status",
              "CudaOuterSegment::bind", "CudaOuterSegment::bound",
              "CudaOuterSegment::setHooks", "CudaOuterSegment::hooks",
              "CudaOuterSegment::refusal", "CudaOuterSegment::runSegment"]
for sym in CU_SYMBOLS:
    if sym not in GRAPH_CU_CODE:
        problems.append(f"CudaOuterGraph.cu: does not define {sym}")
    if sym not in STUB_CODE:
        problems.append(f"CudaOuterGraphStub.cpp: does not define {sym} -- a CPU-only "
                        "build would fail to link, which is the stub-parity contract")
if "no CUDA in this build" not in STUB_CODE:
    problems.append("CudaOuterGraphStub.cpp: the runner stub must say WHY it refuses")

# --- 10. the receipt ---------------------------------------------------------
for field in ("segment_launches", "device_outers", "host_outer_observations", "budget_exits",
              "halted_outer_launches", "escapes", "refusals"):
    want(GRAPH_CU_CODE, f'"{field}', "CudaOuterGraph.cu",
         "the Sec 9.3 receipt field named by the task")
want(GRAPH_CU_CODE, "[RASBERY][OUTER_GPU]", "CudaOuterGraph.cu", "the receipt tag")
want(MAIN_CODE, "rasbery::gpu::reportOuterSegment(std::cout)", "main.cpp",
     "the receipt has to be emitted or it is not a receipt")
if MAIN_CODE.count("rasbery::gpu::reportOuterSegment(std::cout)") < 2:
    problems.append("main.cpp: the receipt is emitted on only one of the two exit paths "
                    "(batch and single); a batch run would print nothing")
REPORT = body_of(GRAPH_CU_CODE, "void reportOuterSegment", "void noteOuterSegmentRefusal")
if "outerGpuEnabled()" not in REPORT:
    problems.append("reportOuterSegment: must print whenever the feature was ENABLED, "
                    "including when every segment was refused -- a receipt that only appears "
                    "on success cannot tell 'off' from 'on but never engaged'")
# Telemetry stays cheap: counters only, no timers on a per-outer path.
for banned in ("std::chrono", "clock_gettime", "cudaEventElapsedTime"):
    if banned in GRAPH_CU_CODE:
        problems.append(f"CudaOuterGraph.cu: uses {banned}.  Task 9's telemetry budget is one "
                        "increment per event; a segment is at most 8 outers and a timer per "
                        "outer would be a measurable tax on the thing being measured")

# --- 11. the host bodies are COUNTED, so "on" and "engaged" are different ----
#
# An audit of 8be6bee found the Task 4/5/7 device bodies had ZERO production
# callers: a run with every device feature on still executed
# BICGCMFD::updpsi/updjnet/upddhat and Nodal::updateConstant on the CPU, and no
# receipt in the tree could have said so.  A device outer's claim is
# `host_body_calls` flat across the segment, and a claim with no counter behind
# it is a review comment rather than a gate.
COUNTERS_H = read(SRC / "HostOuterBodyCounters.h")
BICG = strip_comments(read(SRC / "BICGCMFD.cpp"))
NODAL = strip_comments(read(SRC / "Nodal.cpp"))
for field in ("updpsi", "updjnet", "upddhat", "upddtil", "nodal_constants"):
    want(COUNTERS_H, field, "HostOuterBodyCounters.h", "one counter per host outer body")
# Each host body must bump its own counter, and the bump must be INSIDE the
# function rather than at some call site that a second caller could bypass.
for fn, counter, stop in (
        ("void BICGCMFD::updpsi(", "updpsi", "void BICGCMFD::"),
        ("void BICGCMFD::updjnet(", "updjnet", "void BICGCMFD::"),
        ("void BICGCMFD::upddhat(", "upddhat", "bool BICGCMFD::"),
        ("void BICGCMFD::upddtil(", "upddtil", "void BICGCMFD::")):
    body = body_of(BICG, fn, stop)
    if "hostouter::counters()." + counter not in body:
        problems.append("BICGCMFD.cpp: %s does not bump hostouter::counters().%s -- without "
                        "it a device outer's 'the host did not run this' is unfalsifiable"
                        % (fn.strip("(") , counter))
if NODAL.count("hostouter::counters().nodal_constants") < 2:
    problems.append("Nodal.cpp: both updateConstant sweeps (the GPU arm's and the CPU "
                    "drive's) must be counted; counting one leaves the other invisible")
# Counted per SWEEP, not per node: a per-node atomic inside the OpenMP loop
# would be nxyz increments per drive on the hottest loop in the nodal solver.
if "updateConstant(lk) ? 1 : 0;\n            hostouter::" in NODAL or \
        "hostouter::bumpHostBody" in body_of(NODAL, "bool Nodal::updateConstant(", "\n}"):
    problems.append("Nodal.cpp: the constants counter is bumped per NODE.  It must be "
                    "bumped once around the sweep -- nxyz atomics per drive is a "
                    "measurable tax on the thing being measured")
# And the receipt has to carry them, in one shared formatter so the two arms
# cannot disagree.
want(GRAPH_H_CODE, "outerHostBodyJson", "CudaOuterGraph.h",
     "the host-body block must be formatted in ONE place for both arms")
for arm, code in (("CudaOuterGraph.cu", GRAPH_CU_CODE),
                  ("CudaOuterGraphStub.cpp", STUB_CODE)):
    if "outerHostBodyJson()" not in code:
        problems.append(f"{arm}: the receipt does not report host_body_calls, so a run "
                        "cannot tell 'the device outer is on' from 'the host still did the "
                        "arithmetic'")
want(GRAPH_H_CODE, '\\"host_body_calls\\"', "CudaOuterGraph.h", "the receipt field name")
# And an idle run must say WHY.  `refusals{}` is only filled by a call site that
# reached the delegation, and ReconvergeFlux is the search FALLBACK -- a deck
# that never falls back never enters it, so an ordinary run with the feature on
# printed `device_outers:0, refusals:{}` and not one word about the reason.
want(GRAPH_H_CODE, "outerIdleReasonJson", "CudaOuterGraph.h",
     "an idle run must name its refusal, asked at print time rather than remembered "
     "from a call site that may never have been reached")
for arm, code in (("CudaOuterGraph.cu", GRAPH_CU_CODE),
                  ("CudaOuterGraphStub.cpp", STUB_CODE)):
    if "outerIdleReasonJson(" not in code:
        problems.append(f"{arm}: the receipt does not name the idle reason, so "
                        "'on and never engaged' and 'on and every segment did nothing' "
                        "print the same line")
    if "segment_launches == 0" not in code:
        problems.append(f"{arm}: the idle reason must be printed only when nothing ran -- "
                        "on a healthy run it is 'none' and would be noise")

# --- 12. the arena is stood up in production (link 1) ------------------------
#
# GpuPhysicsArena::reserve() had ZERO callers at 8be6bee, so no DeviceSlotView
# was ever built, no DeviceArenaView was ever published, and the runner refused
# every segment with `no_runner` -- which made the Task 4/5/7 kernels dead code
# with a passing replay behind them.
ARENA_H = read(SRC / "GpuPhysicsArena.h")
SLOTCTL = read(SRC / "GpuSlotControl.h")
for arm, code, name in ((GRAPH_CU_CODE, GRAPH_CU_CODE, "CudaOuterGraph.cu"),
                        (STUB_CODE, STUB_CODE, "CudaOuterGraphStub.cpp")):
    if "rasberyStandUpOuterSegment" not in code:
        problems.append(f"{name}: does not define rasberyStandUpOuterSegment; the stand-up "
                        "must have both arms or a CPU-only build cannot link the call site")
want(GRAPH_CU_CODE, "outerArena().reserve(dims)", "CudaOuterGraph.cu",
     "the production reserve() -- without it nothing builds a DeviceSlotView and the "
     "runner refuses with no_runner forever")
want(GRAPH_CU_CODE, "emitReceipt(receipt)", "CudaOuterGraph.cu",
     "the Sec 4.4 VRAM admission receipt, on BOTH paths: a refusal that printed nothing "
     "is the silent-shrink failure Sec 4.4 was written against")
# The receipt must be emitted whether or not the reservation succeeded.
_res = body_of(GRAPH_CU_CODE, "const bool reserved = outerArena().reserve", "if (!reserved)")
if _res and "emitReceipt" not in _res:
    problems.append("CudaOuterGraph.cu: the arena receipt is emitted only on success; a "
                    "refused reservation is exactly the case a reader needs the numbers for")
want(GRAPH_CU_CODE, "arena.slot_views = t.slot_views", "CudaOuterGraph.cu",
     "the DeviceArenaView must carry an uploaded slot-view table")
for field in ("arena.phases", "arena.states", "arena.searches", "arena.params"):
    want(GRAPH_CU_CODE, field, "CudaOuterGraph.cu",
         "the four dense control arrays are what DeviceArenaView is; a missing one is a "
         "null dereference in the first kernel that touches it")
want(DRIVER_CODE, "gpu::rasberyStandUpOuterSegment(", "Driver.h",
     "the stand-up needs a production call site or reserve() is still uncalled")
# It must be gated and it must run once, before any solve.
_drv = body_of(DRIVER_CODE, "if (gpu::outerGpuEnabled()) {", "const bool is_restart_run")
if _drv and "rasberyStandUpOuterSegment" not in _drv:
    problems.append("Driver.h: the stand-up is not inside the RASBERY_GPU_OUTER gate, so an "
                    "OFF run would pay a VRAM query and an allocation it never uses")
if DRIVER_CODE.count("gpu::rasberyStandUpOuterSegment(") != 1:
    problems.append("Driver.h: the arena must be stood up EXACTLY once -- reserve() is "
                    "documented as an error to call twice, and every address it hands out "
                    "has to be fixed before a graph could exist")

# --- 13. one slot table, DERIVED from the arena (links 3 and 4) --------------
#
# A hand-built CmfdOuterSlotTable beside a reserved arena is a second set of
# pointers to the same slot, free to disagree with the first.  It fails nowhere
# and is wrong everywhere.
want(GRAPH_H_CODE, "k_cmfd_build_slot_table", "CudaOuterGraph.h",
     "the CMFD slot table must be DERIVED from arena.slotView(), not built beside it")
TABLE = body_of(GRAPH_H_CODE, "void k_cmfd_build_slot_table", "__global__ void k_outer_seed_slot")
if "arena.slotView(slot)" not in TABLE:
    problems.append("k_cmfd_build_slot_table: does not read arena.slotView(slot), so the "
                    "table is not derived from the arena and the two can disagree about "
                    "which bytes a slot owns")
# LINK 4: psi is cmfd_psi, and this is the ONLY place it is decided.
if "o.psi  = v.cmfd_psi;" not in TABLE and "o.psi = v.cmfd_psi;" not in TABLE:
    problems.append("k_cmfd_build_slot_table: CmfdOuterView::psi must bind to "
                    "DeviceSlotView::cmfd_psi, not psi.  DeviceSlotView carries BOTH -- "
                    "`psi` (SlotRegion::Psi) and `cmfd_psi` (SlotRegion::CmfdPsi, 'the CMFD "
                    "fission source, distinct from psi') -- and CmfdOuterView::psi is the "
                    "CMFD fission source.  Both are [nxyz] writable doubles, so binding the "
                    "wrong one is silent: updpsi fills the array the nodal path reads")
if re.search(r"o\.psi\s*=\s*v\.psi\b", TABLE):
    problems.append("k_cmfd_build_slot_table: binds CmfdOuterView::psi to DeviceSlotView::psi")
# LINK 3: pointer REBASE, never a transpose -- the arena adopted host order.
for banned in ("transpose", "Transpose"):
    if banned in GRAPH_H_CODE or banned in GRAPH_CU_CODE:
        problems.append("CudaOuterGraph: contains a transpose.  DeviceSlotView::phif is "
                        "[l*ng+ig] ('AoS, matching Geometry::Phif') and dtil/dhat/jnet are "
                        "the host's [ls*ng+ig] by deliberate design, so every CmfdOuterView "
                        "field is a pointer rebase.  The group-major mismatch is against "
                        "BatchCore::phi, which belongs to the sweep")
for rebase in ("o.flux = v.phif;", "o.jnet = v.jnet;", "o.dtil = v.dtil;", "o.dhat = v.dhat;"):
    if rebase not in TABLE:
        problems.append(f"k_cmfd_build_slot_table: missing the {rebase!r} rebase")
if "macroXsIndex(kXtXsdf" not in TABLE or "macroXsIndex(kXtXsnf" not in TABLE:
    problems.append("k_cmfd_build_slot_table: xsdf/xsnf must be addressed through the shared "
                    "macroXsIndex, not a hand-written stride into the packed xs block")

# --- 14. material_generation has a host counter now (link 5) -----------------
#
# nodalConstantSlotIsCurrent gates the whole constants phase on
# `nodal_constant_generation == material_generation`.  With nothing bumping
# material_generation both sat at zero, the gate read "current", and the phase
# returned without computing anything for the entire run.
want(GRAPH_H_CODE, "k_outer_seed_slot", "CudaOuterGraph.h",
     "the slot's control packet must be seeded or the constants gate reads 'current' from "
     "a pair of zeros")
SEED = body_of(GRAPH_H_CODE, "void k_outer_seed_slot", "inline cudaError_t enqueueBuildCmfdSlotTable")
if "st.material_generation" not in SEED:
    problems.append("k_outer_seed_slot: does not stamp material_generation")
if "st.nodal_constant_generation" not in SEED:
    problems.append("k_outer_seed_slot: does not stamp nodal_constant_generation.  Leaving "
                    "it equal to material_generation makes nodalConstantSlotIsCurrent read "
                    "'current' before the constants have ever been built on the device")
want(DRIVER_CODE, "hoststateGeneration()", "Driver.h",
     "material_generation must be fed from XSSet's EXISTING host counter; a second counter "
     "beside a correct one is how two counters disagree")
want(GRAPH_CU_CODE, "seed_generation", "CudaOuterGraph.cu",
     "the seed must be clamped away from zero -- `gen - 1` on a zero wraps to UINT64_MAX "
     "and leaves the constants permanently stale")
# The speculative list has to lose the entry, or the next reader is told not to
# gate on the counter this task just made truthful.
# Scoped to the LIST, which ends where the note explaining the removal begins;
# spanning further would match the note itself and pass on any list there is.
_spec = body_of(SLOTCTL, "SPECULATIVE (no host counter yet", "material_generation LEFT")
if "material_generation" in _spec:
    problems.append("GpuSlotControl.h: material_generation is still listed as SPECULATIVE "
                    "('do not gate an upload on these'), but link 5 gave it a host counter "
                    "and nodalConstantSlotIsCurrent now gates on it")
want(SLOTCTL, "hoststateGeneration", "GpuSlotControl.h",
     "the counter that backs material_generation has to be named where the field is")
# The plan's own name for the constants phase must exist.
want(GRAPH_H_CODE, "enqueueNodalConstants", "CudaOuterGraph.h",
     "the plan calls the constants phase enqueueNodalConstants; a reader who greps the "
     "plan's name found nothing")

# --- 15. the dhat/psi handoff (link 2) ---------------------------------------
#
# The 416 KiB/outer dhat H2D was the ONE sweep input pushed unconditionally
# every outer, because dhat changes after every nodal correction and comparing a
# mirror would cost more than the copy.  Link 2 removes it by making the
# segment's upddhat write the buffer the sweep reads -- so "skip the upload" is
# not an optimisation that could be wrong, it is the only correct action once
# the handoff is bound.
BICG_CU = strip_comments(read(SRC / "CudaBICGBackend.cu"))
BICG_H = read(SRC / "CudaBICGBackend.h")
BICG_CPP = strip_comments(read(SRC / "BICGCMFD.cpp"))
BICG_HDR = strip_comments(read(SRC / "BICGCMFD.h"))

for field in ("dhat_device_resident", "psi_device_resident"):
    want(BICG_H, field, "CudaBICGBackend.h",
         "the sweep has to be TOLD the segment owns this, or it uploads the host twin "
         "over the bytes the segment just produced")
    want(BICG_CPP, "io." + field, "BICGCMFD.cpp",
         "the flag has to be published from the drive that stages the sweep")
for flag in ("dhat_resident", "psi_resident"):
    want(BICG_CU, "sl." + flag, "CudaBICGBackend.cu",
         "stageSweeps must carry the residency into the slot the upload loop reads")
# The dhat push must be GUARDED, and the guard must be the residency.
DHAT = body_of(BICG_CU, "if (sl.dhat_resident)", "++telemetry.cmfd_assembly_gpu_calls")
if not DHAT or "push(dhat_dev" not in DHAT:
    problems.append("CudaBICGBackend.cu: the dhat H2D is not guarded by sl.dhat_resident. "
                    "It is the one sweep input pushed unconditionally every outer, so it "
                    "is the whole 416 KiB/outer link 2 exists to remove")
if "cmfd_dhat_h2d_elided_bytes" not in BICG_CU:
    problems.append("CudaBICGBackend.cu: the elided dhat bytes are not counted, so the "
                    "claim 'the H2D is gone' has no number behind it")
# psi residency must outrank psi_dirty: dirty means 'the host wrote it', and
# when the segment owns psi the host never wrote it at all.
PSI = body_of(BICG_CU, "if (sl.psi_resident)", "pushOrSkip(phi + m")
if PSI and PSI.find("else if (sl.push_psi)") < 0:
    problems.append("CudaBICGBackend.cu: the psi residency test does not outrank psi_dirty. "
                    "psi_dirty is true at every drive boundary, so ranking it first would "
                    "upload the host twin over the fission source the segment's updpsi "
                    "just produced")

# The segment must bind to the sweep's buffers, and the binding must be a
# POINTER REBASE -- every layout already matches (see the residentView note).
want(GRAPH_H_CODE, "k_cmfd_bind_resident", "CudaOuterGraph.h",
     "the slot table has to be repointed at the sweep's buffers")
BIND = body_of(GRAPH_H_CODE, "void k_cmfd_bind_resident", "inline cudaError_t enqueueBuildCmfdSlotTable")
for field in ("v.flux = flux;", "v.psi  = psi;", "v.dtil = dtil;", "v.dhat = dhat;"):
    if field not in BIND:
        problems.append(f"k_cmfd_bind_resident: missing {field!r}")
if "v.jnet" in BIND:
    problems.append("k_cmfd_bind_resident: rebinds jnet.  The sweep arena has no jnet -- it "
                    "is not a CMFD input -- so jnet must keep the physics-arena pointer "
                    "k_cmfd_build_slot_table gave it, or the two kernels own one field each "
                    "and are free to disagree about it")
want(BICG_H, "residentView", "CudaBICGBackend.h",
     "the segment needs the sweep's device addresses to bind to")
want(read(SRC / "CudaBICGBackendStub.cpp"), "CudaBatchArena::residentView",
     "CudaBICGBackendStub.cpp", "stub parity for the residency accessor")

# THE HOST TWINS MUST STILL BE WRITTEN.  drive() takes the host loop for the
# Wielandt warm-up and whenever the device sweep declines, and that loop reads
# _dhat and _psi.  A segment that wrote only the device copies would hand the
# fallback path stale arrays -- silently, and only on decks that warm up
# differently.
for mirror in ("host_dhat", "host_psi"):
    want(GRAPH_CU_CODE, "bound_." + mirror, "CudaOuterGraph.cu",
         "the host twin has to be refreshed or the host drive path reads an array one "
         "outer stale")
want(GRAPH_H_CODE, "host_mirror_bytes", "CudaOuterGraph.h",
     "the mirror is the honest remaining cost and it has to be counted")
want(strip_comments(read(SRC / "CMFD.h")), "dhatData", "CMFD.h",
     "the segment needs the host twin's address")
# The arming gate must NOT reuse canUseDeviceAssembly: that carries the
# per-drive Wielandt warm-up, which is false at every SolveLoop entry, so an
# arming site built on it would arm nothing ever.
RESIDENT = body_of(BICG_CPP, "bool BICGCMFD::deviceSweepResident() const", "bool BICGCMFD::canUseDeviceAssembly")
if RESIDENT and "canUseDeviceAssembly" in RESIDENT:
    problems.append("BICGCMFD::deviceSweepResident reuses canUseDeviceAssembly, which "
                    "carries `_wiel_sweep >= WIELANDT_WARMUP_SWEEPS`.  That is per-drive "
                    "state and it is false at the top of every SolveLoop entry, so the "
                    "segment would never arm")
if RESIDENT and "_wiel_sweep" in RESIDENT:
    problems.append("BICGCMFD::deviceSweepResident tests the Wielandt warm-up, which is "
                    "per-drive state the arming site cannot use")

# --- 16. the residency cannot outlive the segment (the kngr_238 crash) ------
#
# THE BUG.  setOuterSegmentResident(true) was latched at arm time and never
# cleared, so the moment the segment stopped running -- an escape, a launch
# failure, or a deck like kngr_238 where the critical search makes SolveLoop
# refuse and only ReconvergeFlux ever delegates -- the sweep went on eliding
# the dhat and psi H2D for the rest of the run while the HOST body wrote the
# host arrays.  Every subsequent host outer then assembled the CMFD operator
# from a device dhat nobody was updating: one device outer, an escape, 169 host
# outers, and a non-finite abort on 238 (k_eff 1.135 -> 0.678 locally).
#
# The flag now means "the caller of THIS drive owns dhat and psi on the
# device", which is the only reading under which it cannot outlive its caller.
_HOOK = body_of(DRIVER_CODE, "static bool outerSweepHook(", "static bool outerNodalHook(")
if _HOOK:
    _on = _HOOK.find("setOuterSegmentResident(true)")
    _drive = _HOOK.find("cmfd_solver.drive(")
    _off = _HOOK.find("setOuterSegmentResident(false)")
    if not (0 <= _on < _drive < _off):
        problems.append("Driver.h: outerSweepHook must raise the residency BEFORE its drive() "
                        "and lower it AFTER.  A flag set anywhere else outlives the segment, "
                        "and the sweep then elides the dhat/psi H2D for every host outer that "
                        "follows -- which is a stale CMFD operator, not a slow one")
_ARM = body_of(DRIVER_CODE, "static bool armOuterSegment(", "static void ReconvergeFlux(")
if _ARM and "setOuterSegmentResident(true)" in _ARM:
    problems.append("Driver.h: armOuterSegment latches the residency true.  Arming happens "
                    "once per SolveLoop/ReconvergeFlux entry and the segment may never run "
                    "there at all (a critical-search deck refuses); the flag belongs to the "
                    "segment's own drive()")

# --- 17. a host-driven drive publishes no device-only signals ---------------
#
# driveDeviceSweeps owns the negative-flux retry and finishes the
# degenerate-gamma sweep on the Rayleigh branch itself, then keeps looping --
# it returns only when neither is outstanding.  So after a HOST-driven drive
# those signals describe nothing live, and worse, they describe something
# STALE: the latches are written only when a device sweep actually ran, and
# drive() takes the host loop for the whole Wielandt warm-up.  That is the
# negative_flux escape kngr_238 raised on its very first device outer.
if _HOOK and "rasberyPublishOuterProbe(slot, *h.eigv, *h.residual, false, false)" not in _HOOK:
    problems.append("Driver.h: outerSweepHook publishes device-only signals from a "
                    "HOST-driven drive.  driveDeviceSweeps resolves the negative-flux retry "
                    "and the Rayleigh hand-back before it returns, so the probe would report "
                    "a condition that no longer exists -- or a stale one from an earlier "
                    "drive.  A bad iterate is still caught by k_outer_refresh_inputs' "
                    "finiteness test; the signals come back when the sweep is stream-ordered")
# And the latches must describe the CURRENT drive, so a reader during the
# warm-up window cannot pick up an older one.
_DDS = body_of(BICG_CPP, "bool BICGCMFD::driveDeviceSweeps", "while (iout < _ncmfd)")
for latch in ("_last_sweep_negative", "_last_sweep_state"):
    if _DDS and not re.search(re.escape(latch) + r"\s*=\s*0;", _DDS):
        problems.append(f"BICGCMFD.cpp: {latch} is not cleared on entry to "
                        "driveDeviceSweeps, so a drive that takes the host loop leaves the "
                        "previous drive's signal readable")

# --- 18. a Failed escape does not carry the device's prev_inner -------------
_FAILBR = body_of(RECONVERGE, "DevicePhase::Failed", "gpu_outer_armed = false;")
if _FAILBR and "prev_inner" not in _FAILBR:
    problems.append("Driver.h: a Failed escape keeps the device's prev_inner.  eigv and "
                    "residual came back through the sweep hook, which wrote them in place, "
                    "but prev_inner came out of a DeviceSlotState whose decision reached "
                    "Failed -- on a non-finite that is not a number this loop should carry. "
                    "The host body's own rule is prev_inner = eigv")

# --- 19. EVERY device buffer the segment reads is synced by the segment -----
#
# THE RULE, learned three times.  A buffer the segment READS but the SWEEP
# refreshes is only current if the sweep's refresh is unconditional and happens
# before the segment's first reader.  Three were not:
#
#   dhat  written by the segment at step 8, read by the sweep at step 2, and its
#         H2D is the one link 2 elides -> uninitialised on the first outer
#         (k_eff -0.034501, negative_flux on 447 of 516 outers).
#   xsnf  refreshed inside drive(), which is step 2, but updpsi is step 1 -> the
#         fission source came from the previous outer's cross sections.
#   dtil  pushed ONLY inside issueSweepUploads' `if (sl.device_assembly)` branch,
#         and device assembly is off for the whole Wielandt warm-up -> the device
#         updjnet read a dtil the host had already recomputed.  Cost: one extra
#         outer per statepoint on i-SMR CY01 with an identical converged k_eff,
#         which is a trajectory-B0 violation that no answer-level check sees.
#
# So the segment syncs all of them itself.  A new device input added to a body
# without a sync here is the same bug a fourth time.
for _sync, _why in (
        ("upload flux", "drive() takes the host loop during the Wielandt warm-up, after "
                        "which the device phi is behind Geometry::Phif"),
        ("upload xsnf", "updpsi runs BEFORE the sweep refreshes xs_xsnf"),
        ("upload dtil", "issueSweepUploads pushes dtil only on the device-assembly path, "
                        "which is off for the whole Wielandt warm-up"),
        ("seed residency", "the segment writes dhat at step 8 and the sweep reads it at "
                           "step 2, so the first outer after arming has nothing to read")):
    want(GRAPH_CU_CODE, _sync, "CudaOuterGraph.cu", _why)
# The bodies read exactly these; if one grows an input the sync list must grow.
_BODY_INPUTS = ("v.flux", "v.dtil", "v.dhat", "v.jnet", "v.psi", "v.xsnf", "v.xsdf")
_KERNEL_SRC = strip_comments(read(SRC / "CmfdOuterKernel.h"))
for _fld in re.findall(r"v\.(\w+)\[", _KERNEL_SRC):
    if "v." + _fld not in _BODY_INPUTS:
        problems.append(f"CmfdOuterKernel.h: a body reads v.{_fld}, which is not in the "
                        "segment's sync list.  Every device buffer a body reads has to be "
                        "either written by the segment itself or synced by it -- a buffer "
                        "whose refresh belongs to the sweep is only current if that refresh "
                        "is unconditional and happens before the first reader, and three of "
                        "them were not")
want(strip_comments(read(SRC / "CMFD.h")), "dtilData", "CMFD.h",
     "the segment needs the host d-tilde to sync from")

# --- 20. the per-outer tracer stays off by default ---------------------------
#
# It is what localised the dtil bug -- it prints eigv/residual/prev_inner and a
# byte hash of psi/jnet/dhat/flux per outer, so the ON and OFF arms can be
# diffed to the FIRST differing quantity instead of to the converged answer.
# Kept because the next trajectory difference will need it, gated because it is
# a per-outer fprintf over ~100k doubles.
_TRACE = read(SRC / "OuterTrace.h")
want(_TRACE, "RASBERY_OUTER_TRACE", "OuterTrace.h", "the tracer's gate")
if "enabled()" not in DRIVER_CODE or "outertrace::emit" not in DRIVER_CODE:
    problems.append("Driver.h: the per-outer tracer is not wired, so a future trajectory "
                    "difference has to be localised by re-deriving the instrument first")
_TR_CALL = body_of(DRIVER_CODE, "if (outertrace::enabled()) {", "outertrace::emit")
if not _TR_CALL:
    problems.append("Driver.h: outertrace::emit is not guarded by outertrace::enabled(), so "
                    "every run would pay a hash of psi/jnet/dhat/flux per outer")

# --- the gates exist and are built -------------------------------------------
want(REPLAY_TEXT, "kPhaseTransitions", "test/outer_state_replay.cpp",
     "the emitted edges must be cross-checked against the W1 table")
want(REPLAY_TEXT, "hostSegmentReference", "test/outer_state_replay.cpp",
     "the reference must be a QUOTATION of Driver.h; a test that drove the shipped body "
     "against itself would pass on any body there is")
if "cmfdOuterConvergence" not in REPLAY_TEXT:
    problems.append("test/outer_state_replay.cpp: does not drive the shipped body")
want(REPLAY_TEXT, "budget", "test/outer_state_replay.cpp",
     "the budget-neutrality property is the Class-B0-on-trajectory gate")
if "rasbery_outer_state_replay" not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: does not build rasbery_outer_state_replay")
if "add_test(NAME outer_state_replay" not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: outer_state_replay is built but not registered with ctest")
if "src/CudaOuterGraph.cu" not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: does not build src/CudaOuterGraph.cu")
# The stub must be DROPPED from a CUDA build, exactly like the other three, or
# both arms define the same symbols and the link fails.
remove_block = CMAKE_TEXT[CMAKE_TEXT.find("list(REMOVE_ITEM RASBERY_SOURCES"):]
remove_block = remove_block[:remove_block.find(")")]
if "src/CudaOuterGraphStub.cpp" not in remove_block:
    problems.append("CMakeLists.txt: CudaOuterGraphStub.cpp is not removed from the sources "
                    "in a CUDA build, so both arms would define the same symbols")
block = CMAKE_TEXT[CMAKE_TEXT.find("src/CudaOuterGraph.cu"):]
if block and "--fmad=false" not in block[:1200]:
    problems.append("CMakeLists.txt: CudaOuterGraph.cu is not compiled with --fmad=false.  It "
                    "composes the Task 4/5 phase kernels, whose contract is bit-identical "
                    "reproduction of the host loops; Class B0 is unreachable without it")


def main() -> int:
    if problems:
        for problem in problems:
            print("device outer state machine: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("device outer state machine: PASS (%d plan steps, %d emitted phases, %d Outer "
          "edges, budget default %s)" % (len(PLAN), len(EMITTED), len(OUTER_EDGES),
                                         m.group(1) if m else "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
