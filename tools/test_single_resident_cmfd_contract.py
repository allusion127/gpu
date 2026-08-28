#!/usr/bin/env python3
"""Single-instance resident CMFD contract (Rev.7.1 Task 6).

Two things are being pinned, and they fail in different ways.

(A) THE GRAPH KEY.  Task 6 Step 3 is marked completion-mandatory in the plan
    because `sweep_graph_unroll` was still part of the sweep graph's cache key.
    `unroll` is the REMAINING sweep budget (`_ncmfd - iout`, BICGCMFD.cpp) and
    _ncmfd is 5 (Driver.h), so it walks 5,4,3,... and back to 5 at the next
    drive(): the topology was destroyed and rebuilt continuously and
    `graph_reinstantiations` climbed for the whole run, which Task 10's
    instantiation gate cannot pass.  The fix moves `unroll` to a DEVICE scalar
    (kSweepSlotBudget) and makes the cached graph a CAPACITY that only grows.
    This file checks the key is gone, the scalar exists, the device honours it,
    and -- with a compiler -- exercises the capacity policy exhaustively.

(B) THE FEATURE GATE.  Resident-single must be reachable without --batch-mode
    and must be OFF by default, because it changes which solver path a normal
    single run takes.  It must also not weaken canUseDeviceAssembly's five
    conditions: the warm-up sweeps stay on the host, and the resident path
    activates only after them.

WHY THE CAPACITY IS EXACT AND NOT AN APPROXIMATION.  A graph deeper than the
budget executes the extra sweep slots, so this is only sound if those slots are
true no-ops.  They are, and the chain is checked here rather than asserted:
cmfd_sweep_begin raises sweep_halt BEFORE it touches any scalar, every later
sweep kernel returns on `sweep_halt`, and the whole inner BiCGSTAB is masked
because initialize_solver_state folds sweep_halt into `halt` and returns before
it writes scalars, flags or counters.
"""
from __future__ import annotations

import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

BACKEND_H = SRC / "CudaBICGBackend.h"
BACKEND_CU = SRC / "CudaBICGBackend.cu"
SOLVER_CPP = SRC / "BICGSolver.cpp"
CMFD_CPP = SRC / "BICGCMFD.cpp"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


H_TEXT = read(BACKEND_H)
H_CODE = strip_comments(H_TEXT)
CU_TEXT = read(BACKEND_CU)
CU_CODE = strip_comments(CU_TEXT)
SOLVER_CODE = strip_comments(read(SOLVER_CPP))
CMFD_CODE = strip_comments(read(CMFD_CPP))


def want(text: str, needle: str, where: str, why: str) -> None:
    if needle not in text:
        problems.append(f"{where}: missing {needle!r} -- {why}")


# ---------------------------------------------------------------------------
# (A1) the unroll key is GONE
# ---------------------------------------------------------------------------
if "sweep_graph_unroll" in CU_CODE:
    problems.append(
        "CudaBICGBackend.cu: sweep_graph_unroll still exists.  Task 6 Step 3 is "
        "completion-mandatory: while unroll keys the graph, every change of the "
        "remaining sweep budget re-instantiates the topology")
if re.search(r"sweep_graph_nmax|sweep_graph_precision", CU_CODE):
    problems.append("CudaBICGBackend.cu: the old sweep graph key fields are still "
                    "declared; they are replaced by SweepGraphCapacity")

want(H_CODE, "struct SweepGraphCapacity", "CudaBICGBackend.h",
     "the capacity policy must be a named, testable thing rather than three ints "
     "compared inline in the launcher")
want(H_CODE, "captureDepth", "CudaBICGBackend.h", "the ratchet must be explicit")
want(CU_CODE, "sweep_graph.serves(", "CudaBICGBackend.cu",
     "launch_sweeps must consult the capacity policy")
want(CU_CODE, "sweep_graph.captureDepth(", "CudaBICGBackend.cu",
     "a recapture must never be shallower than what is already captured, or the "
     "capacity oscillates instead of settling")

# nmax stays an EXACT key, deliberately -- a deeper nmax capture over-iterates
# because force_halt is placed at capture time.  Check the reasoning survives.
if "nmax == want_nmax" not in H_CODE:
    problems.append("CudaBICGBackend.h: SweepGraphCapacity::serves must compare nmax "
                    "EXACTLY -- a deeper nmax capture over-iterates, because "
                    "force_halt is placed from `1 + nmax` at capture time")

# ---------------------------------------------------------------------------
# (A2) unroll became a device scalar, and the device honours it
# ---------------------------------------------------------------------------
for slot in ("kSweepSlotBudget", "kSweepSlots"):
    want(CU_CODE, slot, "CudaBICGBackend.cu", "the sweep-slot budget must be device state")

begin = CU_CODE[CU_CODE.find("void cmfd_sweep_begin"):]
begin = begin[:begin.find("\n__global__", 1)] if "\n__global__" in begin[1:] else begin[:2000]
if "kSweepSlots" not in begin or "kSweepSlotBudget" not in begin:
    problems.append("cmfd_sweep_begin: does not enforce the slot budget; an over-captured "
                    "graph would then run extra sweeps for real")
if not re.search(r"if\s*\(\s*sm\[kSweepSlots\]\s*>=\s*sm\[kSweepSlotBudget\]\s*\)", begin):
    problems.append("cmfd_sweep_begin: the slot-budget test is not the expected "
                    "`sm[kSweepSlots] >= sm[kSweepSlotBudget]`")
# The halt must be raised BEFORE anything is read or written, or the excess slot
# is not a no-op.
budget_at = begin.find("kSweepSlotBudget")
for touched in ("kReigvdel", "kIcmfdDone", "kNegative"):
    at = begin.find(touched)
    if at >= 0 and at < budget_at:
        problems.append(f"cmfd_sweep_begin: touches {touched} before the slot-budget "
                        "test; an over-captured slot must halt before it changes "
                        "anything at all")

# The launch stamps the budget for every participant.
want(CU_CODE, "issueSweepUploads(const int* active_slots, int count, int slot_budget)",
     "CudaBICGBackend.cu",
     "the slot budget is a LAUNCH property (the batch-wide max), so it is stamped "
     "when the participant set is known -- staging it per slot would change how "
     "many retries fit in one launch")

# ---------------------------------------------------------------------------
# (A3) the halt chain that makes over-capture a no-op
# ---------------------------------------------------------------------------
for kernel in ("cmfd_src_build", "cmfd_wiel_terms", "cmfd_wiel_finalize", "cmfd_updls",
               "cmfd_negative_scan", "cmfd_sweep_end"):
    at = CU_CODE.find("void " + kernel)
    if at < 0:
        problems.append(f"CudaBICGBackend.cu: {kernel} not found")
        continue
    body = CU_CODE[at:at + 1200]
    if "sweep_halt[m] != 0u" not in body:
        problems.append(f"{kernel}: does not return on sweep_halt, so an over-captured "
                        "sweep slot would not be a no-op and the capacity graph would "
                        "change the answer")
init = CU_CODE[CU_CODE.find("void initialize_solver_state"):]
init = init[:2000]
if "sweep_halt[m] == 0u" not in init or "if (halt[m] != 0u) return;" not in init:
    problems.append("initialize_solver_state: must fold sweep_halt into `halt` and return "
                    "before touching scalars/flags/counters -- that is what masks the "
                    "whole inner BiCGSTAB of an over-captured sweep")

# ---------------------------------------------------------------------------
# (B) the feature gate
# ---------------------------------------------------------------------------
want(H_CODE, "rasberyResidentSingleCmfd", "CudaBICGBackend.h",
     "resident-single needs a declared gate")
want(CU_CODE, "RASBERY_GPU_CMFD_RESIDENT_SINGLE", "CudaBICGBackend.cu",
     "the env flag is the gate")
if 'std::string(v) != "0"' not in CU_CODE[CU_CODE.find("rasberyResidentSingleCmfd"):][:600]:
    problems.append("rasberyResidentSingleCmfd: must be OFF unless explicitly set, and "
                    "must treat '0' as off, like every other RASBERY_* gate")
want(SOLVER_CODE, "resident_single", "BICGSolver.cpp", "the constructor must take the arena "
     "path for a single instance when the flag is on")
if "rasberyBatchWidth() == 0 && rasberyResidentSingleCmfd()" not in SOLVER_CODE:
    problems.append("BICGSolver.cpp: resident-single must apply only when --batch-mode is "
                    "absent; the two must not both try to own the arena")
if "gpu_requested" not in SOLVER_CODE:
    problems.append("BICGSolver.cpp: resident-single must still require RASBERY_GPU -- a "
                    "flag that silently switched the GPU on is a worse surprise than one "
                    "that does nothing")
# Width 1, and the SAME kernels.
if "g_batch_width > 0 ? g_batch_width : 1" not in CU_CODE:
    problems.append("rasberyBatchArena: resident-single must open the arena at width 1, "
                    "reusing BatchCore rather than adding a single-instance kernel set")

# ---------------------------------------------------------------------------
# canUseDeviceAssembly's five conditions are untouched
# ---------------------------------------------------------------------------
gate = CMFD_CODE[CMFD_CODE.find("bool BICGCMFD::canUseDeviceAssembly"):]
gate = gate[:gate.find("}") + 1]
for condition, why in (
        ('envFlagEnabledDefaultOn("RASBERY_GPU_CMFD_ASSEMBLY")', "the assembly gate"),
        ('envFlagEnabled("RASBERY_GPU_CMFD_SWEEP")', "assembly is coupled to the resident "
                                                     "sweep path on purpose"),
        ('std::getenv("RASBERY_CMFD_DUMP") == nullptr', "the form-probe capture must see "
                                                        "the host operator"),
        ("_g.ng() == 2", "the assembly kernel is 2-group"),
        ("_ls->arena() != nullptr", "the resident path needs the arena -- this is the "
                                    "condition Task 6 makes reachable in single mode"),
        ("_wiel_sweep >= WIELANDT_WARMUP_SWEEPS", "the warm-up and its Rayleigh schedule "
                                                  "stay on the host")):
    if condition not in gate:
        problems.append(f"canUseDeviceAssembly: lost the condition {condition!r} ({why})")

# ---------------------------------------------------------------------------
# Compiled gate: the capacity policy, exhaustively
# ---------------------------------------------------------------------------
HARNESS = r'''
#include "CudaBICGBackend.h"

#include <cstdio>

using rasbery::SweepGraphCapacity;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL %s\n", what); ++failures; }
}

int main() {
    // 1. A fresh cache serves nothing.
    check(!SweepGraphCapacity{}.serves(3, 1, 0), "empty cache must not serve");

    // 2. THE TASK 6 PROPERTY: walking the budget down never re-captures.
    //    _ncmfd = 5 (Driver.h) so the real sequence is 5,4,3,2,1 and back to 5.
    SweepGraphCapacity cap{};
    int captures = 0;
    for (int repeat = 0; repeat < 50; ++repeat) {
        for (int want = 5; want >= 1; --want) {
            if (!cap.serves(3, want, 0)) {
                cap = SweepGraphCapacity{3, cap.captureDepth(want), 0};
                ++captures;
            }
        }
    }
    std::printf("  budget walk 5..1 x50: captures=%d (depth=%d)\n", captures, cap.slots);
    check(captures == 1, "the remaining-budget walk must capture exactly once");
    check(cap.slots == 5, "the capacity must settle at the largest budget seen");

    // 3. Growth re-captures once, then settles.
    captures = 0;
    for (int want : {5, 7, 7, 6, 7, 3}) {
        if (!cap.serves(3, want, 0)) {
            cap = SweepGraphCapacity{3, cap.captureDepth(want), 0};
            ++captures;
        }
    }
    check(captures == 1, "growing past the capacity must capture once and only once");
    check(cap.slots == 7, "capacity must ratchet up to the new maximum");

    // 4. captureDepth never shrinks -- otherwise the capacity oscillates.
    for (int have = 0; have <= 8; ++have)
        for (int want = 0; want <= 8; ++want) {
            const SweepGraphCapacity c{3, have, 0};
            const int d = c.captureDepth(want);
            check(d >= have && d >= want, "captureDepth must cover both");
        }

    // 5. nmax and precision are EXACT keys.
    check(!cap.serves(4, 1, 0), "a different nmax must re-capture (force_halt is baked)");
    check(!cap.serves(3, 1, 1), "a precision change must re-capture (different kernels)");
    check(cap.serves(3, 1, 0), "same nmax/precision with spare capacity must be served");

    if (failures) { std::printf("sweep graph capacity: FAIL (%d)\n", failures); return 1; }
    std::printf("sweep graph capacity: PASS\n");
    return 0;
}
'''


def build_and_run(compiler: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="resident-cmfd-") as td:
        td_path = Path(td)
        src = td_path / "cap.cpp"
        exe = td_path / "cap"
        src.write_text(HARNESS, encoding="utf-8")
        cmd = [compiler, "-std=c++20", "-O2", f"-I{SRC}", str(src), "-o", str(exe)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            return ["capacity harness did not compile: " + done.stderr[-800:]]
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        if run.returncode != 0:
            return ["capacity harness failed"]
    return []


def main() -> int:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is not None:
        problems.extend(build_and_run(compiler))
    if problems:
        for problem in problems:
            print("single resident cmfd contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if compiler is None:
        print("single resident cmfd contract: static contract PASS "
              "(no C++ compiler here -- the capacity gate was skipped)")
    else:
        print("single resident cmfd contract: PASS (static contract + capacity gate, %s)"
              % Path(compiler).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
