#!/usr/bin/env python3
"""Contract gate for the CMFD arena's active-slot compaction.

The bugs this exists to catch all share one property: they are invisible while
RASBERY_GPU_CMFD_COMPACT is off, because compaction-off is the identity map and
a kernel that reads blockIdx.y as a slot index is then perfectly correct.  A
test that only runs the OFF path proves nothing about the ON path, and the ON
path is the one the batch lever rides on.

What is asserted, statically, against src/CudaBICGBackend.cu:

  1. NO direct blockIdx.y slot read survives.  RASBERY_CMFD_SLOT is the only
     reader, and the only exceptions are the two kernels that are deliberately
     full width (initialize_solver_state, finalize_status) -- for those,
     blockIdx.y IS the physical slot and must stay that way.
  2. The guard is the kernel's first statement, and in particular it precedes
     every __syncthreads and every shared-memory write.  Unlike the nodal
     kernels these DO carry barriers; the guard is safe before one only because
     it is block-uniform, and only if it actually comes first.
  3. The four per-slot masks (active, halt, sweep_halt, device_assembly_active)
     are indexed by the MAPPED slot, never by a logical lane.
  4. gridDim.x expressions and the reduction chunking are untouched: grid.x is
     the single-instance domain and compaction must not have reached it.
  5. Compaction is OFF by default and, when off, is the FULL identity over
     physical slots with lanes == slots.
  6. The bucket ladder is the scheduler's ladder, not a second one.
  7. The receipt exists with all six fields.

Run:  python tools/test_cmfd_slot_compaction_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FULL_WIDTH_KERNELS = ("initialize_solver_state", "finalize_status")


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def kernel_bodies(code: str) -> list[tuple[str, str]]:
    """(name, body) for every __global__ in the translation unit."""
    out: list[tuple[str, str]] = []
    for match in re.finditer(r"__global__ (?:__launch_bounds__\([^)]*\)\s*)?void (\w+)\(",
                             code):
        name = match.group(1)
        open_brace = code.find("{", match.end())
        depth = 0
        i = open_brace
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append((name, code[open_brace + 1:i]))
    return out


def main() -> int:
    problems: list[str] = []
    raw = read("src", "CudaBICGBackend.cu")
    code = strip_comments(raw)
    sched = strip_comments(read("src", "GpuPhaseScheduler.h"))

    macro = code[code.find("#define RASBERY_CMFD_SLOT(m)"):]
    macro = macro[: macro.find("\n\n")]
    if not macro:
        problems.append("RASBERY_CMFD_SLOT is not defined")
    else:
        lane_at = macro.find("rasbery_logical >= lanes")
        map_at = macro.find("slot_map[")
        if lane_at < 0:
            problems.append(
                "RASBERY_CMFD_SLOT has no padding test (`rasbery_logical >= lanes`); a "
                "graph replayed wider than the map describes would index it out of range")
        elif 0 <= map_at < lane_at:
            problems.append("RASBERY_CMFD_SLOT reads the map BEFORE the padding test")
        if "m < 0" not in macro:
            problems.append(
                "RASBERY_CMFD_SLOT does not drop a -1 lane; padding lanes would drive "
                "physical slot -1")

    kernels = kernel_bodies(code)
    if len(kernels) < 30:
        problems.append(f"only {len(kernels)} __global__ kernels parsed; the sweep is "
                        "not covering the file")

    for name, body in kernels:
        if name in FULL_WIDTH_KERNELS:
            # These must KEEP the direct read: blockIdx.y is the physical slot
            # and the kernel has to cover every declared slot.
            if "const int m = static_cast<int>(blockIdx.y);" not in body:
                problems.append(
                    f"{name} no longer reads blockIdx.y directly. It is deliberately full "
                    "width -- it writes iter_flags/halt (or the status row) for EVERY "
                    "declared slot -- so compacting it leaves non-participants holding a "
                    "stale mask.")
            if "RASBERY_CMFD_SLOT(" in body:
                problems.append(f"{name} was compacted; it must stay full width")
            continue

        if "RASBERY_CMFD_SLOT(m);" not in body:
            problems.append(f"{name} does not resolve its slot through RASBERY_CMFD_SLOT")
            continue
        if re.search(r"\bblockIdx\.y\b", body):
            problems.append(
                f"{name} reads blockIdx.y directly. That is correct exactly when the map "
                "is the identity -- i.e. with compaction OFF -- so the bug passes every "
                "test that does not turn it on.")

        # The guard must precede any barrier or shared write.  The only thing
        # allowed in front of it is the thread-0 filter of a scalar kernel,
        # which is barrier-free by construction.
        guard_at = body.find("RASBERY_CMFD_SLOT(m);")
        prologue = body[:guard_at].strip()
        if prologue and prologue != "if (threadIdx.x != 0) return;":
            problems.append(
                f"{name}: `{prologue[:60]}` precedes the slot guard. Only the thread-0 "
                "filter may.")
        for hazard in ("__syncthreads", "__shared__", "__syncwarp"):
            at = body.find(hazard)
            if 0 <= at < guard_at:
                problems.append(
                    f"{name}: {hazard} appears before the slot guard. The guard is safe "
                    "before a barrier only because it is block-uniform; after one, a "
                    "returning block strands the rest at the barrier.")

        # 3. the four per-slot masks are indexed by the mapped slot.
        for mask in ("active", "halt", "sweep_halt", "device_assembly_active"):
            for hit in re.finditer(rf"\b{mask}\[([^\]]*)\]", body):
                index = hit.group(1).strip()
                if index not in ("m", "m]"):
                    problems.append(
                        f"{name}: {mask}[{index}] is not indexed by the mapped slot `m`. "
                        "Every per-slot mask is keyed by the PHYSICAL slot; a logical "
                        "lane there silently masks the wrong instance.")

    # 4. grid.x is untouched.
    for needle, why in (
        ("const int chunk = (n + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);",
         "reduce_dot_stage1/2's partition is a pure function of (n, gridDim.x)"),
        ("for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order",
         "reduce_dot_stage2's strict fold order"),
    ):
        if needle not in raw:
            problems.append(f"lost `{needle[:60]}...` -- {why}")
    if code.count("static_cast<int>(blockIdx.x) * chunk") < 1:
        problems.append("the per-block chunk origin no longer comes from blockIdx.x")

    # 5. default off, identity when off.
    gate = code[code.find("bool cmfdCompactEnabled"):][:400]
    if "RASBERY_GPU_CMFD_COMPACT" not in gate:
        problems.append("cmfdCompactEnabled does not read RASBERY_GPU_CMFD_COMPACT")
    if 'std::string(v) != "0"' not in gate or "v != nullptr" not in gate:
        problems.append(
            "cmfdCompactEnabled must be OFF unless explicitly set and treat '0' as off, "
            "like every other RASBERY_* gate")
    build = code[code.find("void buildSlotMap"):][:1400]
    if "if (!compact) {" not in build:
        problems.append("buildSlotMap has no compaction-off arm")
    off_arm = build[build.find("if (!compact) {"):build.find("} else {")]
    if "lanes = slots;" not in off_arm or "h_slot_map[i] = i;" not in off_arm:
        problems.append(
            "buildSlotMap: with compaction OFF the map must be the FULL IDENTITY over "
            "physical slots with lanes == slots. Anything narrower changes which blocks "
            "the OFF launch visits and where they are masked.")
    if "static_cast<size_t>(slots) * sizeof(int)" not in build:
        problems.append(
            "buildSlotMap uploads less than the full fleet width; a stale entry from a "
            "wider previous launch would stay reachable")

    # 6. one ladder, shared with the scheduler.
    want = re.search(r"kDispatchBuckets\[\]\s*=\s*\{([^}]*)\}", sched)
    have = re.search(r"kBuckets\[\]\s*=\s*\{([^}]*)\}", code)
    if want is None or have is None:
        problems.append("cannot compare the CMFD bucket ladder with kDispatchBuckets")
    else:
        a = [t.strip() for t in want.group(1).split(",") if t.strip()]
        b = [t.strip() for t in have.group(1).split(",") if t.strip()]
        if a != b:
            problems.append(f"the CMFD bucket ladder {b} is not kDispatchBuckets {a}")

    # 7. graph keys and the receipt.
    if "graph_lanes != lanes" not in code:
        problems.append(
            "the outer graph cache does not key on grid.y. A graph BAKES grid.y, so a "
            "bucket change is a topology change.")
    serves = code[code.find("bool serves("):][:400] or read("src", "CudaBICGBackend.h")
    if "lanes == want_lanes" not in strip_comments(read("src", "CudaBICGBackend.h")):
        problems.append("SweepGraphCapacity::serves does not key on grid.y")
    for field in ("logical_drives", "physical_slot_blocks", "padding_blocks",
                  "padding_fraction", "bucket_graphs", "bucket_histogram"):
        if field not in raw:
            problems.append(f"the compaction receipt has no {field}")
    if "RASBERY][CMFD][COMPACT" not in raw:
        problems.append("no [RASBERY][CMFD][COMPACT] receipt line")

    # 8. the replay's copy of the guard is the shipped guard.
    replay_raw = read("test", "cmfd_slot_compaction_replay.cu")
    def guard_text(text: str) -> str:
        start = text.find("#define RASBERY_CMFD_SLOT(m)")
        body = text[start:]
        body = body[: body.find("\n\n")]
        return "\n".join(ln.rstrip().rstrip("\\").rstrip() for ln in body.splitlines())
    if guard_text(raw) != guard_text(replay_raw):
        problems.append(
            "test/cmfd_slot_compaction_replay.cu's copy of RASBERY_CMFD_SLOT has drifted "
            "from the shipped one; the replay would be proving a guard that does not ship")

    if problems:
        for problem in problems:
            print(f"cmfd slot compaction contract: FAIL {problem}", file=sys.stderr)
        return 1

    if "--run" in sys.argv:
        import shutil
        import subprocess
        import tempfile
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            print("cmfd slot compaction contract: replay SKIPPED (nvcc not on PATH)")
        else:
            arch = os.environ.get("RASBERY_TEST_ARCH", "sm_61")
            source = os.path.join(ROOT, "test", "cmfd_slot_compaction_replay.cu")
            with tempfile.TemporaryDirectory() as tmp:
                exe = os.path.join(tmp, "replay")
                build = subprocess.run(
                    [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "--fmad=false", source,
                     "-o", exe], capture_output=True, text=True)
                if build.returncode != 0:
                    print(f"cmfd slot compaction contract: FAIL replay build\n"
                          f"{build.stderr}", file=sys.stderr)
                    return 1
                run = subprocess.run([exe], capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                sys.stderr.write(run.stderr)
                if run.returncode != 0:
                    return 1

    print(f"cmfd slot compaction contract: PASS ({len(kernels)} kernels, "
          "map-indexed, guard before every barrier, OFF is the identity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
