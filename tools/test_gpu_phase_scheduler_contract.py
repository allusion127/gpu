#!/usr/bin/env python3
"""Case-phase scheduler core contract (plan Rev.7.1 Task 3, W0-scoped).

Source-level, because these are properties no timing and no numerical A/B can
see:

  1. LEVEL-1 READS ONLY DeviceSlotPhase.  k_classify_case_phases must not touch
     DeviceSlotState, DeviceSearchState, DeviceScheduleParams or any bulk array.
     The whole reason the hot struct is 32 bytes is that 64 of them stay in L1
     across ~68k epochs; one stray dereference of the cold struct silently
     restores the traffic the split was built to remove, and nothing downstream
     would report it.
  2. NO FLOAT ATOMICS anywhere in the scheduler.  The queue order is fixed by a
     ballot-and-prefix, and the counters are integer sums; a float atomic would
     mean somebody added an order-dependent reduction to the control path.
  3. NO PERSISTENT / COOPERATIVE SCAFFOLDING.  W0 measured c_barrier = 0.78 us
     against the 0.384 us kill threshold, so Sec 5.7 is permanently closed.
  4. FIXED-ORDER POLICY, WITH THE EXTENSION POINT MARKED.  The W0 replay failed
     at fleet = 64, so fairness is deferred: no max-age, no cost class, no
     starvation credit.  The marker has to be there so the next person knows
     where the decision was made and what it was.
  5. CMFD QUANTUM = ONE OUTER, with the Sec 5.1(2) fallback clause documented
     and both modes retained (constraint 15 rollback), and the escalation
     arithmetic citing the W0 switch_eval measurement of 5.13 us.
  6. Compaction gate compiles and passes with no CUDA at all.
"""
from __future__ import annotations

import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST = ROOT / "test"

HEADER = (SRC / "GpuPhaseScheduler.h").read_text(encoding="utf-8-sig")
KERNELS = (SRC / "GpuPhaseScheduler.cu").read_text(encoding="utf-8-sig")
STUB = (SRC / "GpuPhysicsBackendStub.cpp").read_text(encoding="utf-8-sig")
COMPACT_TEST = (TEST / "gpu_phase_compaction.cpp").read_text(encoding="utf-8-sig")
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def strip_literals(text: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


HEADER_CODE = strip_comments(HEADER)
KERNEL_CODE = strip_comments(KERNELS)
KERNEL_BARE = strip_literals(KERNEL_CODE)
STUB_CODE = strip_comments(STUB)

problems: list[str] = []


def need(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def body_after(anchor: str, text: str) -> str:
    start = text.find(anchor)
    if start < 0:
        problems.append(f"anchor not found: {anchor!r}")
        return ""
    open_at = text.find("{", start)
    if open_at < 0:
        problems.append(f"no block after {anchor!r}")
        return ""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    problems.append(f"unbalanced block after {anchor!r}")
    return ""


# ---------------------------------------------------------------------------
# 1. Level-1 reads only DeviceSlotPhase.
# ---------------------------------------------------------------------------
classify = body_after("void k_classify_case_phases(", KERNEL_CODE)
need(classify != "", "k_classify_case_phases is missing")
for forbidden in ("DeviceSlotState", "DeviceSearchState", "DeviceScheduleParams"):
    need(forbidden not in classify,
         f"k_classify_case_phases touches {forbidden}; Level-1 reads DeviceSlotPhase only")
need("DeviceSlotPhase" in KERNEL_CODE, "the classify kernel does not read DeviceSlotPhase at all")

# Its signature must not even offer the cold structs.
classify_sig = KERNEL_CODE[KERNEL_CODE.find("void k_classify_case_phases("):]
classify_sig = classify_sig[: classify_sig.find(")")]
for forbidden in ("DeviceSlotState", "DeviceSearchState", "DeviceScheduleParams"):
    need(forbidden not in classify_sig,
         f"k_classify_case_phases takes a {forbidden} parameter; it must not be able to read one")

# One CTA of 128 threads, one thread per slot.
need("__launch_bounds__(kClassifyBlock)" in KERNEL_CODE,
     "the classify kernel does not declare its launch bounds")
need(re.search(r"kClassifyBlock\s*=\s*128", KERNEL_CODE) is not None,
     "the classify block is not 128 threads")
launch = body_after("bool gpuLaunchClassify(", KERNEL_CODE)
need("<<<1," in launch.replace(" ", ""),
     "gpuLaunchClassify does not launch exactly ONE CTA (the cross-warp prefix needs it)")

# The Sec 5.2 predicates come from the shared header, not from a local copy.
for predicate in ("slotAlreadyQueued", "slotInFlight", "slotActive"):
    need(predicate in classify, f"the classify kernel does not use {predicate}")
need("queued_epoch" in classify and "state_epoch" in classify,
     "the classify kernel does not capture the epoch on insertion (Sec 5.2)")
need("kSchedFaultInFlightRequeue" in classify and "kSchedFaultDuplicateQueue" in classify,
     "the classify kernel does not raise both Sec 5.2 fatal faults")

# THE FAULT MUST HAVE A CONSUMER.  Bits written to a struct nobody reads are not
# a check: the offending slot would keep running with two bodies driving it.
need("gpuSchedulerFaultIsFatal" in HEADER_CODE, "there is no fatal-fault predicate")
need("gpuMarkSlotFailed" in HEADER_CODE, "there is no consumer for a fatal fault")
need("gpuSchedulerFaultIsFatal" in classify and "gpuMarkSlotFailed" in classify,
     "the classify kernel raises faults but never acts on them")
mark = body_after("inline void gpuMarkSlotFailed(", HEADER_CODE)
need("error_code" in mark, "gpuMarkSlotFailed does not record the reason")
need("kSlotFlagFatal" in mark, "gpuMarkSlotFailed does not set the fatal flag")
need("DevicePhase::Failed" in mark, "gpuMarkSlotFailed does not fail the slot")
need("state_epoch" in mark,
     "gpuMarkSlotFailed does not bump the epoch; the stale queue entry would survive")
serial = body_after("inline void gpuClassifySerial(", HEADER_CODE)
need("gpuMarkSlotFailed" in serial,
     "the serial reference does not consume faults, so the CPU gate cannot check it")

# Deterministic compaction: ballot + prefix, never an atomic bump for the index.
need("__ballot_sync" in classify, "the classify kernel does not use a warp ballot")
need("__popc" in classify, "the classify kernel does not use a population count for the rank")
queue_writes = re.findall(r"queue\[[^\]]*\]\.slots\[[^\]]*\]\s*=\s*([^;]+);", classify)
need(all("atomic" not in w for w in queue_writes),
     "a queue slot index comes from an atomic; the queue order would depend on warp scheduling")

# ---------------------------------------------------------------------------
# 2. No float atomics.
# ---------------------------------------------------------------------------
for atomic in re.finditer(r"\batomic\w*\s*\(\s*&?([A-Za-z_][\w\.\[\]>-]*)", KERNEL_BARE):
    target = atomic.group(1)
    need(not any(f in target for f in ("residual", "eigv", "flux", "relax", "tol")),
         f"an atomic writes what looks like a floating-point quantity: {target}")
need("atomicAdd(&s_" in KERNEL_CODE or "atomicAdd(&counters->" in KERNEL_CODE,
     "the counters are not accumulated atomically")
need(re.search(r"atomic\w*\s*\(\s*\(?\s*(float|double)", KERNEL_BARE) is None,
     "a float/double atomic appears in the scheduler")

# ---------------------------------------------------------------------------
# 3. No persistent / cooperative scaffolding.
# ---------------------------------------------------------------------------
FORBIDDEN_PERSISTENT = ("cooperative_groups", "grid_group", "this_grid", "grid.sync",
                        "cudaLaunchCooperativeKernel", "PersistentKernel", "persistent_kernel",
                        "cudaOccupancyMaxActiveBlocksPerMultiprocessor")
for path, text in (("GpuPhaseScheduler.h", HEADER_CODE), ("GpuPhaseScheduler.cu", KERNEL_CODE),
                   ("gpu_phase_compaction.cpp", strip_comments(COMPACT_TEST))):
    for token in FORBIDDEN_PERSISTENT:
        need(token not in text,
             f"{path} carries persistent/cooperative scaffolding ({token}); W0 closed that track")
need("0.78" in HEADER and "0.384" in HEADER,
     "the header does not record the W0 c_barrier measurement that closed the persistent track")

# The header is a shared pure body: constraint 35 applies to it.
need("cuda_runtime.h" not in HEADER_CODE and re.search(r"\bcuda[A-Z]\w*\s*\(", HEADER_CODE) is None,
     "GpuPhaseScheduler.h reaches for CUDA; it is a shared pure body (constraint 35)")
for intrinsic in ("__ballot", "__shfl", "__popc", "__activemask", "__syncwarp"):
    need(intrinsic not in HEADER_CODE,
         f"GpuPhaseScheduler.h uses the CUDA-only intrinsic {intrinsic} (constraint 35)")

# ---------------------------------------------------------------------------
# 4. Fixed-order policy present, fairness deferred and MARKED.
# ---------------------------------------------------------------------------
need("kPhaseScanOrder" in HEADER_CODE, "the fixed phase scan order is missing")
need("FAIRNESS EXTENSION POINT" in HEADER,
     "the deferred-fairness extension point is not marked")
need("W0" in HEADER and "fleet" in HEADER,
     "the header does not say WHY fairness is deferred (the W0 fleet=64 replay verdict)")
select = body_after("inline int gpuSelectPhase(", HEADER_CODE)
for deferred in ("phase_age", "max_age", "cost_class", "credit", "starv"):
    need(deferred not in select,
         f"gpuSelectPhase already implements fairness ({deferred}); Task 3 defers it")
need("kPhaseScanOrderCount" in select and "phase_count" in select,
     "gpuSelectPhase is not the fixed-order scan over the phase counts")

# Sec 5.5 buckets, and the padding helper the graph path needs.
need("kDispatchBuckets[]   = {1, 2, 4, 8, 16, 24, 32, 48, 64}" in HEADER_CODE
     or re.search(r"kDispatchBuckets\[\]\s*=\s*\{1,\s*2,\s*4,\s*8,\s*16,\s*24,\s*32,\s*48,\s*64\}",
                  HEADER_CODE) is not None,
     "the Sec 5.5 bucket set is not 1/2/4/8/16/24/32/48/64")
padding = body_after("inline bool gpuDispatchIsPadding(", HEADER_CODE)
need("logical >= active_count" in padding,
     "the graph-path padding helper is not `logical >= active_count`")
need("kQueueEmptySlot = -1" in HEADER_CODE, "queue padding is not -1")

# ---------------------------------------------------------------------------
# 5. CMFD quantum: one outer, fallback documented, both modes retained.
# ---------------------------------------------------------------------------
need("CmfdQuantumMode" in HEADER_CODE, "the CMFD quantum mode enum is missing")
need("OneOuter" in HEADER_CODE and "RemainingBudget" in HEADER_CODE,
     "both CMFD quantum modes must exist (constraint 15 rollback preservation)")
need("RASBERY_GPU_PHASE_QUANTUM_CMFD" in HEADER,
     "the Sec 5.1(2) flag that selects the CMFD quantum mode is not named")
quantum = body_after("inline PhaseQuantum gpuPhaseQuantum(", HEADER_CODE)
need("cmfd == CmfdQuantumMode::RemainingBudget ? PhaseQuantum::RemainingBudget"
     in re.sub(r"\s+", " ", quantum),
     "the default CMFD quantum is not one outer with the escalation as the alternative")
need("5.13" in HEADER, "the W0 switch_eval measurement (5.13 us) is not cited")
need("0.783" in HEADER, "the W0 c_dispatch measurement (0.783 us) is not cited")
need("68" in HEADER and "0.03" in HEADER_CODE,
     "the Sec 5.1(2) escalation arithmetic (68k epochs vs the 3% ceiling) is not written down")
need("tail_efficiency" in HEADER,
     "the escalation clause omits the tail_efficiency <= 3pp condition")
need("gpuCmfdQuantumShouldEscalate" in HEADER_CODE,
     "the escalation predicate is prose only; it must be code the test can call")

# The quantum table's other rows (Task 3 Step 6).
for phase, quantum_name in (("Xenon", "OneStep"), ("ThermalHydraulics", "OneStep"),
                            ("Search", "OneTrial"), ("DepletionPredictor", "OneStage"),
                            ("DepletionCorrector", "OneStage"), ("Ppr", "OneCornerIteration")):
    need(re.search(r"DevicePhase::" + phase + r":\s*return PhaseQuantum::" + quantum_name,
                   re.sub(r"\s+", " ", quantum)) is not None,
         f"the quantum table does not map {phase} to {quantum_name}")

# ---------------------------------------------------------------------------
# 6. Sec 8.2 refill.
# ---------------------------------------------------------------------------
refill = body_after("void k_refill_free_slots(", KERNEL_CODE)
need(refill != "", "k_refill_free_slots is missing")
for helper in ("deviceSlotPhaseReset", "deviceSlotStateReset", "deviceSearchStateReset",
               "deviceScheduleParamsReset"):
    need(helper in refill, f"the refill kernel does not reset via {helper} (Sec 8.2 is all four)")
need("state_epoch + 1" in refill.replace("  ", " "),
     "the refill kernel does not bump state_epoch")
need("atomicAdd(next_input" in refill.replace(" ", "").replace("atomicAdd(next_input,1)",
                                                               "atomicAdd(next_input")
     or "atomicAdd(next_input" in re.sub(r"\s+", "", refill),
     "the refill kernel does not claim its input with an atomic cursor")
need("claim >= input_count" in refill,
     "the refill kernel does not guard the claim against the input count")
need("claim < 0" in refill,
     "the refill kernel does not guard against a wrapped cursor before indexing")
need("*next_input >= input_count" in refill,
     "the refill kernel bumps the cursor even when the inputs are exhausted; the "
     "cursor would grow without bound and `exhausted` would count passes, not slots")
need("if (finished) {" in refill,
     "the refill kernel re-resets already-Empty slots every pass")
launch_refill = body_after("bool gpuLaunchRefill(", KERNEL_CODE)
need("a.inputs == nullptr" in launch_refill,
     "gpuLaunchRefill does not reject a null descriptor array with a positive count")
# Reset must precede the claim: a slot may never hold half of two decks.
need(refill.find("deviceScheduleParamsReset") < refill.find("atomicAdd(next_input"),
     "the refill kernel claims an input before resetting the tenant (Sec 8.2 order)")
need("kSlotFlagActive | kSlotFlagInputReady" in refill,
     "a refilled slot is not stamped active and input-ready")
need("DevicePhase::Import" in refill, "a refilled slot does not enter the Import phase")

# The reset helpers must clear PADDING, not just fields: the Task 20 audit is
# byte-level, and so is any snapshot hash.
slot_control = (SRC / "GpuSlotControl.h").read_text(encoding="utf-8-sig")
need("deviceZeroBytes" in slot_control,
     "the reset helpers do not zero struct padding; a byte-level tenant audit would fail")
for helper in ("deviceSlotPhaseReset", "deviceSlotStateReset", "deviceSearchStateReset",
               "deviceScheduleParamsReset"):
    body = body_after(f"inline void {helper}(", strip_comments(slot_control))
    need("deviceZeroBytes" in body, f"{helper} does not zero the struct before setting defaults")

# ---------------------------------------------------------------------------
# 7. Transition table anchors resolve, and their ORDER matches Driver.h.
#
# This is the half test/gpu_phase_compaction.cpp cannot do: it checks the map
# against an expected-edge list, and this checks the map against the SOURCE.
# ---------------------------------------------------------------------------
driver_lines = DRIVER.splitlines()


def anchor_line(anchor: str, lo: int = 1, hi: int | None = None) -> int | None:
    end = len(driver_lines) if hi is None else min(hi, len(driver_lines))
    for i in range(lo, end + 1):
        if anchor in driver_lines[i - 1]:
            return i
    return None


DRIVER_SCAN = strip_literals(strip_comments(DRIVER))


def function_range(signature_pattern: str) -> tuple[int, int]:
    """1-based [first, last] line of the brace-matched body of `signature`.

    The Outer-quantum anchors must be located INSIDE SolveLoop, not just
    anywhere in Driver.h: ReconvergeFlux (Driver.h:829-851) replays the same
    call sequence for the flux-only path, so a whole-file search would find the
    helper's copy and never see SolveLoop's convergence test at all.
    """
    # Braces are counted over source with comments AND string literals removed:
    # a `{` inside either would close the range early or never, and the range is
    # what makes the anchor search find SolveLoop's copy of the call sequence
    # rather than ReconvergeFlux's (Driver.h:829-851 replays the same calls).
    match = re.search(signature_pattern, DRIVER_SCAN)
    if match is None:
        problems.append(f"Driver.h no longer has a function matching {signature_pattern!r}")
        return (1, len(driver_lines))
    start   = match.start()
    open_at = DRIVER_SCAN.find("{", match.end() - 1)
    if open_at < 0:
        problems.append("SolveLoop has no body")
        return (1, len(driver_lines))
    depth  = 0
    end_at = len(DRIVER_SCAN)
    for i in range(open_at, len(DRIVER_SCAN)):
        if DRIVER_SCAN[i] == "{":
            depth += 1
        elif DRIVER_SCAN[i] == "}":
            depth -= 1
            if depth == 0:
                end_at = i
                break
    # Stripping preserves newlines, so line numbers still line up with the file.
    return (DRIVER_SCAN.count("\n", 0, start) + 1, DRIVER_SCAN.count("\n", 0, end_at) + 1)


SOLVE_LO, SOLVE_HI = function_range(r"\bstatic\s+void\s+SolveLoop\s*\(")


outer_block = HEADER[HEADER.find("kOuterQuantumSteps[] = "):]
outer_block = outer_block[: outer_block.find("};")]
outer_anchors = re.findall(r'\{"([a-z_]+)",\s*"((?:[^"\\]|\\.)*)"\}', outer_block)
need(len(outer_anchors) == 8,
     f"kOuterQuantumSteps has {len(outer_anchors)} entries; the Outer quantum is 8 steps")

previous_line = 0
previous_name = ""
for name, anchor in outer_anchors:
    line = anchor_line(anchor.replace('\\"', '"'), SOLVE_LO, SOLVE_HI)
    if line is None:
        problems.append(
            f"Outer-quantum anchor for {name!r} is not inside SolveLoop "
            f"(lines {SOLVE_LO}-{SOLVE_HI}): {anchor!r}")
        continue
    if line <= previous_line:
        problems.append(
            f"Driver.h runs {name!r} (line {line}) before {previous_name!r} (line {previous_line}); "
            "the Outer quantum order in GpuPhaseScheduler.h no longer matches the host")
    previous_line = line
    previous_name = name

# Every non-empty transition anchor has to exist in Driver.h -- an anchor that
# has drifted away is a table nobody can check any more.
edges_block = HEADER[HEADER.find("kPhaseTransitions[] = "):]
edges_block = edges_block[: edges_block.find("\n};")]
edge_anchors = [a for a in re.findall(r'PhaseEdgeGuard::\w+,\s*\n?\s*"((?:[^"\\]|\\.)*)"',
                                      edges_block) if a]
need(len(edge_anchors) >= 12,
     f"only {len(edge_anchors)} transition edges carry a Driver.h anchor; Sec 6.21 needs more")
for anchor in edge_anchors:
    need(anchor_line(anchor.replace('\\"', '"')) is not None,
         f"transition anchor is not in Driver.h: {anchor!r}")

# Sec 5.4: the PERTURBATION order inside a converged outer is Xe -> TH -> Search.
# Driver.h checks search before TH but perturbs TH first, and it is the
# perturbation that moves the physics -- so this is the order that must hold.
xe_line = anchor_line("// 3. Equilibrium xenon feedback.", SOLVE_LO, SOLVE_HI)
th_line = anchor_line("if (has_th && !th_converged) {", SOLVE_LO, SOLVE_HI)
search_line = anchor_line("if (has_search && !search_converged) {", SOLVE_LO, SOLVE_HI)
need(xe_line is not None and th_line is not None and search_line is not None,
     "one of the Xe / TH / Search perturbation sites is no longer findable in Driver.h")
if xe_line and th_line and search_line:
    need(xe_line < th_line < search_line,
         f"Driver.h perturbs in the order Xe={xe_line}, TH={th_line}, Search={search_line}; "
         "Sec 5.4 requires Xe -> TH -> Search")

# ---------------------------------------------------------------------------
# 8. Stub parity and build wiring.
# ---------------------------------------------------------------------------
for launcher in ("gpuLaunchClassify", "gpuLaunchRefill"):
    need(f"bool {launcher}(" in HEADER_CODE, f"{launcher} is not declared")
    need(re.search(r"\bbool\s+" + launcher + r"\s*\(", KERNEL_CODE) is not None,
         f"GpuPhaseScheduler.cu does not define {launcher}")
    need(re.search(r"\bbool\s+" + launcher + r"\s*\(", STUB_CODE) is not None,
         f"GpuPhysicsBackendStub.cpp does not define {launcher}")
need("cuda_runtime.h" not in STUB_CODE and re.search(r"\bcuda[A-Z]\w*\s*\(", STUB_CODE) is None,
     "the stub translation unit reaches for CUDA")

# Stream ordering: classify must observe the refill's writes.
need("gpuLaunchRefillThenClassify" in HEADER_CODE,
     "there is no ordered entry point; refill and classify could be issued on "
     "different streams and race on the same phases array")
ordered = body_after("bool gpuLaunchRefillThenClassify(", KERNEL_CODE)
need("gpuLaunchRefill(a, stream)" in ordered and "queues, stream" in ordered,
     "the ordered launcher does not put both kernels on the SAME stream")
need("same stream or separated by an event" in HEADER or "same stream" in HEADER,
     "the classify/refill ordering requirement is not documented")

# gpuQueueRank is gone: it was untested and its semantics did not match the
# kernel's (it counted inactive slots and ignored the ownership predicates).
need("gpuQueueRank" not in HEADER_CODE, "gpuQueueRank is still present")

need("GpuPhaseScheduler.cu" in CMAKE, "CMakeLists.txt does not compile the scheduler kernels")
need("rasbery_gpu_phase_compaction" in CMAKE, "CMakeLists.txt does not build the compaction gate")
need("add_test(NAME gpu_phase_compaction" in CMAKE,
     "the compaction gate is not registered with ctest")

# ---------------------------------------------------------------------------
# 9. Compile and run the compaction gate.
# ---------------------------------------------------------------------------


def msvc_vcvars() -> str | None:
    if os.name != "nt":
        return None
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    done = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, universal_newlines=True)
    root = done.stdout.strip().splitlines()
    if done.returncode != 0 or not root:
        return None
    bat = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return str(bat) if bat.is_file() else None


def build_and_run(compiler: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        exe = tmp_path / ("phase_compaction.exe" if os.name == "nt" else "phase_compaction")
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                script = tmp_path / "build_phase_compaction.bat"
                # /wd4324: the alignas tail padding on DeviceSlotState, which
                # Task 1 Step 4 requires and which is suppressed at the source.
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul\r\n' % compiler
                    + 'cl /nologo /std:c++20 /EHsc /W4 /WX /wd4324 "%s" /I "%s" /Fe:"%s"\r\n'
                      % (TEST / "gpu_phase_compaction.cpp", SRC, exe),
                    encoding="utf-8")
                subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(tmp_path),
                               capture_output=True, universal_newlines=True)
            else:
                subprocess.run(
                    [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                     "-I", str(SRC), str(TEST / "gpu_phase_compaction.cpp"), "-o", str(exe)],
                    check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            output = (failure.stdout or "") + (failure.stderr or "")
            return ["compaction gate did not compile: " + output.strip()[-2000:]]
        done = subprocess.run([str(exe)], capture_output=True, universal_newlines=True)
        if done.returncode != 0:
            return ["compaction gate failed: " + (done.stderr or done.stdout).strip()[-2000:]]
    return []


def main() -> int:
    compiler = (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
                or msvc_vcvars())
    if compiler is not None:
        problems.extend(build_and_run(compiler))
    if problems:
        for problem in problems:
            print("gpu phase scheduler: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if compiler is None:
        print("gpu phase scheduler: static contract PASS "
              "(no C++ compiler here -- the compaction gate was skipped)")
    else:
        print("gpu phase scheduler: PASS (static contract + compaction gate, %s)"
              % Path(compiler).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
