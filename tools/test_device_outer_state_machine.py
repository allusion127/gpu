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
if re.search(r"flux_converged\s*=", GRAPH_H_CODE) or re.search(r"flux_converged\s*=",
                                                               GRAPH_CU_CODE):
    problems.append("CudaOuterGraph: re-derives flux_converged.  Sec 6.13's decision is "
                    "cmfdOuterConvergence's; a second copy of the stall ladder is the "
                    "exact divergence Task 5 exists to prevent")

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

# --- 7. no observation between outers ---------------------------------------
LOOP = body_of(GRAPH_CU_CODE, "for (unsigned int i = 0", "DeviceOuterSegmentState seg_out")
if not LOOP:
    problems.append("CudaOuterGraph.cu: the per-outer loop could not be located")
for banned in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyDeviceToHost",
               "cudaEventSynchronize", "cudaStreamQuery", "cudaMemcpy("):
    if banned in LOOP:
        problems.append("CudaOuterGraph.cu: the per-outer loop calls %s.  A segment that "
                        "observes between outers reinstates the host rendezvous this task "
                        "exists to remove -- the M64 campaign measured that rendezvous, not "
                        "the kernels, as the wall." % banned)
if GRAPH_CU_CODE.count("cudaStreamSynchronize") != 1:
    problems.append("CudaOuterGraph.cu: expected exactly ONE cudaStreamSynchronize in the "
                    "whole runner (found %d)" % GRAPH_CU_CODE.count("cudaStreamSynchronize"))
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
if "runSegment(" in SOLVELOOP:
    problems.append("Driver.h: SolveLoop delegates a segment.  A segment there exits on the "
                    "outer whose decision was not a requeue, and that outer's body has "
                    "already advanced flux/jnet/d-hat -- so the host would have to re-enter "
                    "PAST the control ladder, which is not an entry point SolveLoop has.  "
                    "Task 10's conditional WHILE is what makes that resume a no-op")
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
