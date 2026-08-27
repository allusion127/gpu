#!/usr/bin/env python3
"""Static contract for the W0 decision-spike toolkit (Rev.7.1 programme).

These five probes gate a multi-month programme, and four of the five cannot run
on the authoring machine: three need nvcc and a GPU, one needs the production
binary and Nsight Compute.  A probe that silently measured the wrong thing would
not fail anywhere -- it would come back with a plausible number and the number
would go into a decision.  So the properties that make each probe trustworthy
are asserted in its SOURCE:

  * that the grid sweep is still the five real kernel shapes,
  * that the co-residency check precedes the cooperative launch rather than
    following it,
  * that every conditional-graph error is captured instead of swallowed,
  * that the ncu-absent fallback exists and is labelled as a proxy,
  * that the gate constants are the same number in every file that quotes them.

The two pure-python probes CAN be checked dynamically, so they are: their
self-tests run as subprocesses here.

Run:  python3 tools/test_w0_spikes.py
"""
from __future__ import annotations

import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def read(name: str) -> str:
    path = TOOLS / name
    if not path.is_file():
        fail(f"missing file: tools/{name}")
    return path.read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"w0 spikes contract: FAIL: {message}")


def require(text: str, needle: str, what: str) -> None:
    if needle not in text:
        fail(f"{what}: expected to find {needle!r}")


def strip_comments(text: str) -> str:
    """C/C++ source with comments blanked out.

    Checks that forbid a CODE pattern have to run on code.  Several of the
    comments in these probes quote the very construct they warn against -- e.g.
    "`cudaGraphNodeParams np;` does not compile" -- and a naive whole-file regex
    fires on the warning instead of on a real defect.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


DISPATCH = read("probe_dispatch_floor.cu")
GRIDSYNC = read("probe_gridsync_cost.cu")
CONDGRAPH = read("probe_conditional_graph.cu")
L2SH = read("probe_l2_width.sh")
L2PY = read("parse_ncu_l2.py")
SCHED = read("scheduler_trace_replay.py")
RUNNER = read("run_w0_spikes.sh")

# Comment-stripped views, for checks that forbid a code construct which the
# comments deliberately quote.
CONDGRAPH_CODE = strip_comments(CONDGRAPH)
GRIDSYNC_CODE = strip_comments(GRIDSYNC)

# The five real kernel shapes.  Probes 1 and 2 must sweep the same list or their
# numbers are not comparable, which is the whole point of running them together.
GRID_SWEEP = [34, 67, 209, 1188, 4224]

# Gate constants, each owned by one probe and quoted by the runner.  A gate that
# drifts between files is the failure mode this section exists to prevent.
C_BARRIER_GATE_US = 0.384
DRAM_GATE_PCT = 60
IDLE_REDUCTION_GATE_PCT = 20

# ---------------------------------------------------------------------------
# 0. Every probe must carry its own build+run line.  The person running these on
#    238 is not the person who wrote them.
# ---------------------------------------------------------------------------
for name, text in (("probe_dispatch_floor.cu", DISPATCH),
                   ("probe_gridsync_cost.cu", GRIDSYNC),
                   ("probe_conditional_graph.cu", CONDGRAPH)):
    header = text[:text.find("#include")] if "#include" in text else text
    if "nvcc" not in header:
        fail(f"{name}: no nvcc build line in the header comment")
    if "-arch=sm_120" not in header:
        fail(f"{name}: the header build line does not name the target arch")
    if "CUDA_VISIBLE_DEVICES" not in header:
        fail(f"{name}: the header does not show how to run it")

# ---------------------------------------------------------------------------
# 1. probe_dispatch_floor.cu -- sweep list, events, JSON lines.
# ---------------------------------------------------------------------------
sweep = re.search(r"kGrids\[\]\s*=\s*\{([^}]*)\}", DISPATCH)
if not sweep:
    fail("probe_dispatch_floor: no kGrids sweep array")
got = [int(x) for x in re.findall(r"\d+", sweep.group(1))]
if got != GRID_SWEEP:
    fail(f"probe_dispatch_floor: grid sweep is {got}, expected {GRID_SWEEP}")

for needle in ("cudaEventCreate", "cudaEventRecord", "cudaEventElapsedTime",
               "cudaEventSynchronize"):
    require(DISPATCH, needle, "probe_dispatch_floor")
# The two clocks must be on the right side of the line: the replay is device
# work and gets cudaEvent; capture+instantiate is host work that enqueues
# nothing, so an event pair around it would measure an empty device timeline and
# report ~0.
measure = DISPATCH[DISPATCH.find("static int measure_graph("):
                   DISPATCH.find("static int measure_stream(")]
if not measure:
    fail("probe_dispatch_floor: measure_graph not found")
if "cudaEventElapsedTime" not in measure:
    fail("probe_dispatch_floor: the graph replay is not timed with cudaEvent")
if "steady_clock" not in measure:
    fail("probe_dispatch_floor: instantiation is host work and must use a host clock")
# The ASSIGNMENT, not the parameter declaration -- the declaration is in the
# signature and would sit before everything, making this check vacuous.
inst_at = measure.find("*out_instantiate_ms =")
event_at = measure.find("cudaEventCreate")
if inst_at < 0 or event_at < 0 or inst_at > event_at:
    fail("probe_dispatch_floor: instantiation is timed inside the event region")
if not re.search(r"kReplays\)\s*\*\s*static_cast<double>\(kNodes\)", DISPATCH):
    fail("probe_dispatch_floor: ns/node is not divided by replays * nodes")

# Both arms, and the node count / replay count the header claims.
require(DISPATCH, "cudaGraphLaunch", "probe_dispatch_floor (graph arm)")
require(DISPATCH, "k_touch<<<", "probe_dispatch_floor (stream arm)")
if not re.search(r"kNodes\s*=\s*110\b", DISPATCH):
    fail("probe_dispatch_floor: kNodes is not 110")
if not re.search(r"kReplays\s*=\s*1000\b", DISPATCH):
    fail("probe_dispatch_floor: kReplays is not 1000")
if not re.search(r"kThreads\s*=\s*256\b", DISPATCH):
    fail("probe_dispatch_floor: kThreads is not 256")

# A capture that dropped nodes would silently divide by the wrong N.
require(DISPATCH, "cudaGraphGetNodes", "probe_dispatch_floor: node count is unverified")

# JSON Lines: one object per printf, each tagged, each flushed.
if DISPATCH.count('\\"probe\\":\\"dispatch_floor\\"') < 3:
    fail("probe_dispatch_floor: fewer than three tagged JSON line records")
for record in ("\\\"record\\\":\\\"device\\\"", "\\\"record\\\":\\\"sweep\\\"",
               "\\\"record\\\":\\\"summary\\\""):
    require(DISPATCH, record, "probe_dispatch_floor")
if "std::fflush(stdout)" not in DISPATCH:
    fail("probe_dispatch_floor: stdout is never flushed; a crash would lose the records")

# ---------------------------------------------------------------------------
# 2. probe_gridsync_cost.cu -- occupancy BEFORE launch, threshold constant.
# ---------------------------------------------------------------------------
sweep = re.search(r"kGrids\[\]\s*=\s*\{([^}]*)\}", GRIDSYNC)
if not sweep:
    fail("probe_gridsync_cost: no kGrids sweep array")
got = [int(x) for x in re.findall(r"\d+", sweep.group(1))]
if got != GRID_SWEEP:
    fail(f"probe_gridsync_cost: grid sweep is {got}, expected {GRID_SWEEP}")

require(GRIDSYNC, "grid.sync()", "probe_gridsync_cost")
require(GRIDSYNC, "cudaLaunchCooperativeKernel", "probe_gridsync_cost")
require(GRIDSYNC, "cooperative_groups.h", "probe_gridsync_cost")

occ = GRIDSYNC.find("cudaOccupancyMaxActiveBlocksPerMultiprocessor")
launch = GRIDSYNC.find("cudaLaunchCooperativeKernel(func")
if occ < 0:
    fail("probe_gridsync_cost: no occupancy query")
if launch < 0:
    fail("probe_gridsync_cost: no cooperative launch site")
if occ > launch:
    # Textual order is the check that matters here: the occupancy query must be
    # reachable before any launch, and the file is written top-down.
    fail("probe_gridsync_cost: the occupancy query does not precede the cooperative launch")

# An oversized grid must be REPORTED, not crashed into.
require(GRIDSYNC, "max_coresident_blocks", "probe_gridsync_cost")
require(GRIDSYNC, '\\"supported\\":false', "probe_gridsync_cost: no unsupported-shape path")
if "resident_only" not in GRIDSYNC or "fits_only" not in GRIDSYNC:
    fail("probe_gridsync_cost: no residency guard around the launch")
require(GRIDSYNC, "prop.cooperativeLaunch", "probe_gridsync_cost: device capability unchecked")

# FIX 4 -- a cooperative-launch refusal must not end the process.  The occupancy
# query can say a shape fits and the driver still refuse it
# (cudaErrorCooperativeLaunchTooLarge); exiting there throws away the 34/67
# measurements the gate is actually quoted at.
if "static int measure(" in GRIDSYNC_CODE or "static int time_arm(" in GRIDSYNC_CODE:
    fail("probe_gridsync_cost: measure/time_arm still return an exit code; a launch "
         "refusal at one grid would abort the whole sweep")
require(GRIDSYNC, "static cudaError_t time_arm(",
        "probe_gridsync_cost: time_arm must return cudaError_t")
require(GRIDSYNC, "static cudaError_t measure(",
        "probe_gridsync_cost: measure must return cudaError_t")
sweep_loop = GRIDSYNC[GRIDSYNC.find("for (int i = 0; i < kNumGrids; ++i)"):
                      GRIDSYNC.find("// The gate is quoted at")]
if not sweep_loop:
    fail("probe_gridsync_cost: sweep loop not found")
if re.search(r"return\s+(rc|only_err|rmw_err)\s*;", sweep_loop):
    fail("probe_gridsync_cost: the sweep loop returns on a launch error instead of "
         "recording it and continuing")
require(sweep_loop, "cooperative launch refused",
        "probe_gridsync_cost: a driver refusal is not reported as unsupported")
require(sweep_loop, "continue;",
        "probe_gridsync_cost: the sweep does not continue past a refused shape")
# An arm that never ran did not fail the gate.
require(GRIDSYNC, '\\"unknown\\"',
        "probe_gridsync_cost: an unmeasured arm still reports passes_gate:false")
if not re.search(r"removable_s\s*=\s*\n?\s*\(gate_value >= 0\.0\)", GRIDSYNC):
    fail("probe_gridsync_cost: removable_seconds must be -1 when unmeasured, not "
         "computed with a substituted 0 barrier (which prints the most optimistic "
         "number the arithmetic can produce)")

# The gate.
m = re.search(r"kBarrierGateUs\s*=\s*([0-9.]+)", GRIDSYNC)
if not m:
    fail("probe_gridsync_cost: no kBarrierGateUs threshold constant")
if abs(float(m.group(1)) - C_BARRIER_GATE_US) > 1e-9:
    fail(f"probe_gridsync_cost: barrier gate is {m.group(1)}, expected {C_BARRIER_GATE_US}")
require(GRIDSYNC, "gate_c_barrier_us", "probe_gridsync_cost: the gate is never printed")
require(GRIDSYNC, '\\"verdict\\"', "probe_gridsync_cost: no verdict in the summary")

# Both arms: the pure barrier is the gate, the RMW arm is the realistic shape.
require(GRIDSYNC, "k_barrier_only", "probe_gridsync_cost")
require(GRIDSYNC, "k_barrier_rmw", "probe_gridsync_cost")

# ---------------------------------------------------------------------------
# 3. probe_conditional_graph.cu -- four sub-probes, errors captured.
# ---------------------------------------------------------------------------
# (a) legality: WHILE, SWITCH, and the nested-IF fallback all built.
for needle in ("cudaGraphCondTypeWhile", "cudaGraphCondTypeSwitch", "cudaGraphCondTypeIf",
               "cudaGraphNodeTypeConditional", "cudaGraphConditionalHandleCreate",
               "cudaGraphAddNode", "cudaGraphSetConditional"):
    require(CONDGRAPH, needle, "probe_conditional_graph (sub-probe a)")
require(CONDGRAPH, "build_while_switch", "probe_conditional_graph (sub-probe a)")
require(CONDGRAPH, "build_while_nested_if", "probe_conditional_graph (sub-probe a)")

# (b) instantiation wall at ~100 / 500 / 1500 nodes, on a HOST clock.
m = re.search(r"kInstNodeCounts\[\]\s*=\s*\{([^}]*)\}", CONDGRAPH)
if not m:
    fail("probe_conditional_graph: no instantiation node-count sweep")
counts = [int(x) for x in re.findall(r"\d+", m.group(1))]
if counts != [100, 500, 1500]:
    fail(f"probe_conditional_graph: instantiate sweep is {counts}, expected [100, 500, 1500]")
if "std::chrono::steady_clock" not in CONDGRAPH:
    fail("probe_conditional_graph: instantiation is host work and must be timed on a host clock")

# (c) control overhead over 10k iterations, with the SWITCH cost as a difference.
if not re.search(r"kControlIters\s*=\s*10000\b", CONDGRAPH):
    fail("probe_conditional_graph: control-overhead iteration count is not 10000")
require(CONDGRAPH, "switch_eval_us", "probe_conditional_graph (sub-probe c)")
require(CONDGRAPH, "while_us_per_iter", "probe_conditional_graph (sub-probe c)")

# (d) cooperative launch inside a conditional body.
require(CONDGRAPH, "probe_coop_in_conditional", "probe_conditional_graph (sub-probe d)")
require(CONDGRAPH, "cudaLaunchAttributeCooperative", "probe_conditional_graph (sub-probe d)")
require(CONDGRAPH, "cudaGraphKernelNodeSetAttribute", "probe_conditional_graph (sub-probe d)")
if "k_coop_body" not in CONDGRAPH or "grid.sync()" not in CONDGRAPH:
    fail("probe_conditional_graph: sub-probe d must launch a kernel that really grid.syncs")

# Errors captured, not swallowed.  Every failure path has to reach note()/note_text(),
# and note() has to record the driver's own strings.
note_body = CONDGRAPH[CONDGRAPH.find("static void note(int slot"):]
note_body = note_body[:note_body.find("static void note_text")]
for needle in ("cudaGetErrorName", "cudaGetErrorString"):
    require(note_body, needle, "probe_conditional_graph: note() drops the driver error text")
if CONDGRAPH.count("note(") < 12:
    fail("probe_conditional_graph: too few error-capture sites; some failure path is silent")
require(CONDGRAPH, '\\"record\\":\\"error\\"', "probe_conditional_graph: errors are never printed")
require(CONDGRAPH, "json_escape", "probe_conditional_graph: error text is not JSON-escaped")
if re.search(r"\(void\)\s*cuda[A-Za-z]+\(", CONDGRAPH_CODE):
    fail("probe_conditional_graph: a CUDA call result is cast to void, i.e. swallowed")

# The summary must carry every key the task asked the probe to answer.
for key in ("while_ok", "switch_ok", "nested_if_fallback", "coop_in_conditional",
            "instantiate_ms", "control_overhead_us_per_iter"):
    require(CONDGRAPH, f'\\"{key}\\"', f"probe_conditional_graph: summary lacks {key}")

# Version guards, so an older CUDART reports rather than fails to compile.
require(CONDGRAPH, "CUDART_VERSION", "probe_conditional_graph: no runtime-version guard")
require(CONDGRAPH, "RASBERY_HAS_COND_SWITCH", "probe_conditional_graph: no SWITCH guard")

# FIX 1 -- aggregate initialisation is mandatory, not stylistic.
# cudaGraphNodeParams and cudaLaunchAttributeValue both contain a union with a
# dim3 member; dim3 has a user-provided constructor, which deletes the union's
# implicit default constructor and therefore the enclosing struct's.  Plain
# `T x;` does not compile -- this was the 238 build failure.
for typ in ("cudaGraphNodeParams", "cudaLaunchAttributeValue"):
    if re.search(rf"\b{typ}\s+\w+\s*;", CONDGRAPH_CODE):
        fail(f"probe_conditional_graph: `{typ} x;` is default-initialised and will "
             f"not compile (deleted default ctor via a dim3 union member); use "
             f"`{typ} x{{}};`")
    if not re.search(rf"\b{typ}\s+\w+\{{\}}\s*;", CONDGRAPH_CODE):
        fail(f"probe_conditional_graph: no aggregate-initialised {typ}")

# FIX 6 -- cudaGraphAddNode's signature changed in CUDA 13.0: an edge-data
# pointer was inserted BEFORE numDependencies, so the 12.x 5-argument form does
# not compile there.  Both forms must be present behind a CUDART_VERSION guard,
# because the authoring box is on 12.6 and the target on 13.0.
add_node_calls = re.findall(r"cudaGraphAddNode\s*\(([^;]*?)\)\s*;", CONDGRAPH_CODE,
                            re.DOTALL)
if len(add_node_calls) != 2:
    fail(f"probe_conditional_graph: expected exactly 2 cudaGraphAddNode call sites "
         f"(the CUDA 13 form and the 12.x form behind a version guard), found "
         f"{len(add_node_calls)}")
if not re.search(r"CUDART_VERSION\s*>=\s*13000", CONDGRAPH_CODE):
    fail("probe_conditional_graph: no CUDART_VERSION >= 13000 guard around "
         "cudaGraphAddNode")
guarded = CONDGRAPH_CODE[CONDGRAPH_CODE.find("CUDART_VERSION >= 13000"):]
guarded = guarded[:guarded.find("#endif")]
if "deps, nullptr, ndeps" not in guarded:
    fail("probe_conditional_graph: the CUDA 13 branch does not pass the edge-data "
         "argument (deps, nullptr, ndeps)")
# The unchanged sibling API must not be "fixed" to match.
if re.search(r"cudaGraphAddKernelNode\s*\([^;]*?nullptr\s*,\s*\w*ndeps", CONDGRAPH_CODE):
    fail("probe_conditional_graph: cudaGraphAddKernelNode was given an edge-data "
         "argument; its signature did NOT change in CUDA 13")

# FIX 2a -- a conditional body graph may never be empty; an empty body wedges
# the runtime and the probe hangs holding the GPU.
require(CONDGRAPH, "effective_body_nodes",
        "probe_conditional_graph: no guard forcing a non-empty conditional body")
eff = CONDGRAPH[CONDGRAPH.find("static int effective_body_nodes("):]
eff = eff[:eff.find("}") + 1]
if "> 0) ? requested : 1" not in eff:
    fail("probe_conditional_graph: effective_body_nodes does not floor the body at 1")
# Every body-population site must go through it, not through a raw body_nodes.
for site in ("add_kernel_chain(cases[c], nullptr, 0, body, nullptr)",
             "add_kernel_chain(if_body, nullptr, 0, body, nullptr)"):
    require(CONDGRAPH, site,
            "probe_conditional_graph: a conditional body is populated without the "
            "non-empty guarantee")
if "kCases && body_nodes > 0" in CONDGRAPH_CODE or \
        "if (body_nodes > 0) {" in CONDGRAPH_CODE:
    fail("probe_conditional_graph: a case body is still skipped when body_nodes == 0")

# FIX 2b -- the host-side watchdog, so a wedged graph exits with an error record
# rather than hanging the runner.
require(CONDGRAPH, "SIGALRM", "probe_conditional_graph: no watchdog signal")
require(CONDGRAPH, "alarm(", "probe_conditional_graph: no watchdog timer")
require(CONDGRAPH_CODE, "static void arm_watchdog()",
        "probe_conditional_graph: no arm_watchdog definition")
watchdog = CONDGRAPH[CONDGRAPH.find("extern \"C\" void w0_watchdog("):]
watchdog = watchdog[:watchdog.find("static void arm_watchdog")]
if "write(STDOUT_FILENO" not in watchdog:
    fail("probe_conditional_graph: the watchdog handler must use write(2); printf is "
         "not async-signal-safe and can deadlock on a lock the interrupted code holds")
if "_exit(" not in watchdog:
    fail("probe_conditional_graph: the watchdog does not exit non-zero")
if '\\"record\\":\\"summary\\"' not in watchdog:
    fail("probe_conditional_graph: a watchdog kill emits no summary record, so the "
         "runner would see a probe with no verdict at all")
# Called from main, in CODE -- a commented-out call arms nothing.
main_at = CONDGRAPH_CODE.find("int main()")
if main_at < 0 or CONDGRAPH_CODE.find("arm_watchdog();", main_at) < 0:
    fail("probe_conditional_graph: arm_watchdog() is not called from main")
if CONDGRAPH_CODE.count("kick_watchdog();") < 3:
    fail("probe_conditional_graph: the watchdog deadline is not re-armed between "
         "stages; a slow-but-progressing run would be shot at 120 s")

# FIX 3 -- node counts include the WHILE and ctl nodes.
require(CONDGRAPH, "3 + kCases * body",
        "probe_conditional_graph: while+switch node count is wrong "
        "(WHILE + ctl + SWITCH = 3)")
require(CONDGRAPH, "2 + 3 + 3 * body",
        "probe_conditional_graph: while+nested-if node count is wrong "
        "(WHILE + ctl = 2, plus 3 IF nodes)")

# FIX 5 -- one error slot per failure site, and no discarded run results.
if re.search(r"\bnote\(\s*\d+\s*,", CONDGRAPH_CODE):
    fail("probe_conditional_graph: a numeric error-slot literal remains; two sites "
         "sharing a slot lose one of the two failures")
require(CONDGRAPH, "ERR_SLOT_COUNT", "probe_conditional_graph: no error-slot enum")
control = CONDGRAPH[CONDGRAPH.find("// ---- (c) per-iteration control overhead"):
                    CONDGRAPH.find("// ---- (d) cooperative launch")]
if not control:
    fail("probe_conditional_graph: control-overhead section not found")
if re.search(r"if\s*\(run_timed\([^)]*\)\s*==\s*cudaSuccess\)", control):
    fail("probe_conditional_graph: a control arm discards run_timed's error instead "
         "of note()ing it")
for slot in ("ERR_CONTROL_WHILE_SWITCH", "ERR_CONTROL_WHILE_NESTED_IF"):
    require(control, slot, f"probe_conditional_graph: control arm lacks {slot}")

# ---------------------------------------------------------------------------
# 4. probe_l2_width.sh + parse_ncu_l2.py -- widths, fallback, gate.
# ---------------------------------------------------------------------------
m = re.search(r'WIDTHS="([^"]*)"', L2SH)
if not m:
    fail("probe_l2_width.sh: no WIDTHS default")
widths = [int(x) for x in m.group(1).split()]
if widths != [1, 8, 22, 64]:
    fail(f"probe_l2_width.sh: width sweep is {widths}, expected [1, 8, 22, 64]")

require(L2SH, "lts__t_sector_hit_rate.pct", "probe_l2_width.sh")
require(L2SH, "dram__throughput.avg.pct_of_peak_sustained_elapsed", "probe_l2_width.sh")
require(L2SH, "--target-processes all", "probe_l2_width.sh")

# The ncu-absent fallback.
if "nvidia-smi dmon" not in L2SH:
    fail("probe_l2_width.sh: no nvidia-smi fallback for an absent ncu")
require(L2SH, 'command -v ncu', "probe_l2_width.sh: ncu presence is never tested")
require(L2SH, 'TOOL=', "probe_l2_width.sh: the measurement path is not recorded")
require(L2SH, '"tool":"%s"', "probe_l2_width.sh: the tool used is not written to the receipt")

# Uniformity across the sweep: only the width may vary.
require(L2SH, "RASBERY_XE_ANDERSON=0", "probe_l2_width.sh")
require(L2SH, "unset RASBERY_STATEPOINT_TELEMETRY", "probe_l2_width.sh: SPTELEM not disabled")
require(L2SH, "--batch-mode", "probe_l2_width.sh")

# The W2 gate, printed by the script and enforced by the parser.
m = re.search(r'DRAM_GATE_PCT="?([0-9.]+)"?', L2SH)
if not m or abs(float(m.group(1)) - DRAM_GATE_PCT) > 1e-9:
    fail(f"probe_l2_width.sh: DRAM gate is {m.group(1) if m else None}, expected {DRAM_GATE_PCT}")
if "W2 gate" not in L2SH:
    fail("probe_l2_width.sh: the W2 gate is never printed")

m = re.search(r"DRAM_GATE_PCT_OF_PEAK\s*=\s*([0-9.]+)", L2PY)
if not m or abs(float(m.group(1)) - DRAM_GATE_PCT) > 1e-9:
    fail(f"parse_ncu_l2.py: DRAM gate is {m.group(1) if m else None}, expected {DRAM_GATE_PCT}")

# The kernel filter must reach the BiCGSTAB inner kernels.  A filter of just
# "cmfd|nodal" matches NONE of them -- they are spelled matvec_two_group,
# colored_block_sweep, update_solution, reduce_dot* -- and would make the whole
# sweep vacuous while still returning green numbers.
#
# This reads the REGEX VALUE, not the file: the stems are also named in the
# comment above the regex, so a whole-file search would pass on a gutted filter.
m = re.search(r"NCU_KERNEL_REGEX='([^']*)'", L2SH)
if not m:
    fail("probe_l2_width.sh: could not read the NCU_KERNEL_REGEX value")
kernel_regex = m.group(1)
if not kernel_regex.startswith("regex:"):
    fail(f"probe_l2_width.sh: kernel filter is not an ncu regex: {kernel_regex!r}")
classifier = L2PY[L2PY.find("KERNEL_CLASSES"):L2PY.find("_CLASS_RE")]
for stem in ("matvec_two_group", "colored_block_sweep", "update_solution", "reduce_dot",
             "update_s_jacobi", "prepare_p_jacobi"):
    if stem not in kernel_regex:
        fail(f"probe_l2_width.sh: the ncu kernel filter misses {stem}, a CMFD inner "
             f"kernel; the sweep would profile everything except the bottleneck")
    if stem not in classifier:
        fail(f"parse_ncu_l2.py: the kernel classifier misses {stem}; it would be "
             f"filed under 'other' instead of 'cmfd'")
# Nodal kernels are spelled kNodal*, and ncu's regex: filter is case sensitive.
if "[Nn]odal" not in kernel_regex and "kNodal" not in kernel_regex:
    fail("probe_l2_width.sh: the kernel filter cannot match kNodal* (case-sensitive regex)")

# The fallback must not be quotable against the gate.
require(L2PY, "dram_proxy_pct_utilisation",
        "parse_ncu_l2.py: the fallback reuses the ncu key name")
require(L2PY, '"PROXY"', "parse_ncu_l2.py: the fallback does not mark its verdict as a proxy")

# ---------------------------------------------------------------------------
# 5. scheduler_trace_replay.py -- five policies, gate, assumptions, self-test.
# ---------------------------------------------------------------------------
policy_keys = re.findall(r'Policy\(\s*"([a-z0-9_]+)"', SCHED)
if len(policy_keys) != 5:
    fail(f"scheduler_trace_replay: {len(policy_keys)} policies, expected 5: {policy_keys}")
expected_policies = ["p1_rendezvous", "p2_fixed_cohort_conditional",
                     "p3_track_slot_refill", "p4_largest_ready_first",
                     "p5_phase_queue_max_age"]
if policy_keys != expected_policies:
    fail(f"scheduler_trace_replay: policy set changed: {policy_keys}")

m = re.search(r"IDLE_REDUCTION_GATE_PCT\s*=\s*([0-9.]+)", SCHED)
if not m or abs(float(m.group(1)) - IDLE_REDUCTION_GATE_PCT) > 1e-9:
    fail(f"scheduler_trace_replay: idle gate is {m.group(1) if m else None}, "
         f"expected {IDLE_REDUCTION_GATE_PCT}")

# Every metric the task asks each policy to report.
for key in ("makespan_s", "gpu_busy_fraction", "tail_s", "mean_active_width",
            "scheduler_ops", "tail_efficiency"):
    require(SCHED, f'"{key}"', f"scheduler_trace_replay: no {key} metric")

# The assumptions docstring: present, and actually enumerated rather than a
# one-line promise.  This is the difference between a ranking tool and a lie.
doc_match = re.search(r'"""(.*?)"""', SCHED, re.DOTALL)
if not doc_match:
    fail("scheduler_trace_replay: no module docstring")
doc = doc_match.group(1)
if "ASSUMPTIONS" not in doc:
    fail("scheduler_trace_replay: the docstring has no ASSUMPTIONS section")
assumption_ids = re.findall(r"^A(\d+)\.", doc, re.MULTILINE)
if len(assumption_ids) < 5:
    fail(f"scheduler_trace_replay: only {len(assumption_ids)} enumerated assumptions")
if "RANKING TOOL, NOT A PREDICTOR" not in doc:
    fail("scheduler_trace_replay: the docstring does not disclaim absolute prediction")

# Rev.7 5.8: one heavy phase at a time.
require(SCHED, "single_heavy_phase_at_a_time",
        "scheduler_trace_replay: the one-heavy-phase constraint is not recorded")
require(SCHED, "COST_CLASS", "scheduler_trace_replay: no Rev.7 8.5 cost classes")
require(SCHED, "--selftest", "scheduler_trace_replay: no self-test entry point")

# ---------------------------------------------------------------------------
# 6. run_w0_spikes.sh -- builds all five, and the receipt schema.
# ---------------------------------------------------------------------------
for src in ("probe_dispatch_floor.cu", "probe_gridsync_cost.cu",
            "probe_conditional_graph.cu"):
    require(RUNNER, src, "run_w0_spikes.sh: a probe is never built")
for helper in ("probe_l2_width.sh", "parse_ncu_l2.py", "scheduler_trace_replay.py"):
    require(RUNNER, helper, "run_w0_spikes.sh: a probe is never run")

# The two probes that need device-side runtime calls must get -rdc=true, or they
# will not link.
for name in ("probe_gridsync_cost.cu", "probe_conditional_graph.cu"):
    line = next((ln for ln in RUNNER.splitlines()
                 if "build_and_run" in ln and name in ln), None)
    if line is None:
        fail(f"run_w0_spikes.sh: no build_and_run line for {name}")
    if "-rdc=true" not in line:
        fail(f"run_w0_spikes.sh: {name} is built without -rdc=true")

# Receipt schema.
for key in ("\"schema\"", "\"schema_version\"", "\"generated_utc\"", "\"probes\"",
            "\"gates\"", "\"arch\"", "\"nvcc\"", "\"driver\"", "\"git_sha\""):
    require(RUNNER, key, "run_w0_spikes.sh: receipt schema key missing")
require(RUNNER, '"w0_spikes"', "run_w0_spikes.sh: the receipt is not tagged")

for probe in ("probe_dispatch_floor", "probe_gridsync_cost", "probe_conditional_graph",
              "l2_width", "scheduler_replay"):
    if f'"{probe}"' not in RUNNER:
        fail(f"run_w0_spikes.sh: the receipt has no entry for {probe}")

for gate in ("c_dispatch_us", "c_barrier_us", "conditional_scheduler",
             "dram_pct_of_peak", "idle_reduction_pct"):
    if f'"{gate}"' not in RUNNER:
        fail(f"run_w0_spikes.sh: the receipt has no gate entry for {gate}")

# A skipped probe must say why, and a failed build must not read as a pass.
for status in ("build_failed", "run_failed", "skipped", "timeout"):
    require(RUNNER, status, f"run_w0_spikes.sh: no {status} status")
require(RUNNER, "reason_", "run_w0_spikes.sh: a skipped probe never records a reason")

# FIX 2c -- the runner must bound every probe itself.  Probe 3's own SIGALRM is
# the first line of defence; this is the second, for the case where the probe is
# wedged somewhere its own handler cannot run.
if not re.search(r"timeout\s+--kill-after=\d+\s+180", RUNNER):
    fail("run_w0_spikes.sh: probe invocations are not wrapped in `timeout 180`; a "
         "hung probe would hang the whole W0 run")
if "124" not in RUNNER or "137" not in RUNNER:
    fail("run_w0_spikes.sh: a timeout kill (rc 124/137) is not distinguished from a "
         "normal run failure")

# Gate constants, one number per gate across every file that quotes it.
m = re.search(r"C_BARRIER_GATE_US\s*=\s*([0-9.]+)", RUNNER)
if not m or abs(float(m.group(1)) - C_BARRIER_GATE_US) > 1e-9:
    fail(f"run_w0_spikes.sh: barrier gate {m.group(1) if m else None} "
         f"disagrees with probe_gridsync_cost.cu ({C_BARRIER_GATE_US})")
m = re.search(r"DRAM_GATE_PCT\s*=\s*([0-9.]+)", RUNNER)
if not m or abs(float(m.group(1)) - DRAM_GATE_PCT) > 1e-9:
    fail(f"run_w0_spikes.sh: dram gate {m.group(1) if m else None} "
         f"disagrees with parse_ncu_l2.py ({DRAM_GATE_PCT})")
m = re.search(r"IDLE_REDUCTION_GATE_PCT\s*=\s*([0-9.]+)", RUNNER)
if not m or abs(float(m.group(1)) - IDLE_REDUCTION_GATE_PCT) > 1e-9:
    fail(f"run_w0_spikes.sh: idle gate {m.group(1) if m else None} "
         f"disagrees with scheduler_trace_replay.py ({IDLE_REDUCTION_GATE_PCT})")

# ---------------------------------------------------------------------------
# 7. No probe may touch production source.  These are measurement tools; a spike
#    that edited src/ would invalidate the baseline it is measuring against.
# ---------------------------------------------------------------------------
for name, text in (("probe_dispatch_floor.cu", DISPATCH),
                   ("probe_gridsync_cost.cu", GRIDSYNC),
                   ("probe_conditional_graph.cu", CONDGRAPH)):
    if re.search(r'#include\s+"\.\./src/', text):
        fail(f"{name}: includes production source; the probes must be standalone")

# ---------------------------------------------------------------------------
# 8. The two python probes can be checked for real, so they are.
# ---------------------------------------------------------------------------
PY_FILES = ["parse_ncu_l2.py", "scheduler_trace_replay.py", "test_w0_spikes.py"]
for name in PY_FILES:
    py_compile.compile(str(TOOLS / name), doraise=True)

for name in ("parse_ncu_l2.py", "scheduler_trace_replay.py"):
    proc = subprocess.run([sys.executable, str(TOOLS / name), "--selftest"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"{name} --selftest failed:\n{proc.stdout}\n{proc.stderr}")
    if "PASS" not in proc.stdout:
        fail(f"{name} --selftest did not report PASS: {proc.stdout!r}")

# The replay must produce a parseable receipt with all five policies on a real
# (synthetic) fleet, not only inside its own self-test.
sys.path.insert(0, str(TOOLS))
import scheduler_trace_replay as replay_mod  # noqa: E402

fleet = replay_mod.synthetic_fleet(n_jobs=128, statepoints=5, outers_mean=20,
                                   seed=7, coarsen=4)
out = replay_mod.replay(fleet, width=32, dispatch_share=0.6, ref_width=64)
json.dumps(out)  # the receipt must be JSON-serialisable
if len(out["policies"]) != 5:
    fail("scheduler_trace_replay: replay() did not return five policies")
if out["gate"]["threshold_pct"] != float(IDLE_REDUCTION_GATE_PCT):
    fail("scheduler_trace_replay: replay() reports a different gate threshold")
if any(p["jobs_completed"] != len(fleet) for p in out["policies"]):
    fail("scheduler_trace_replay: a policy left jobs unfinished")

print("w0 spikes contract: PASS")
