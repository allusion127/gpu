#!/usr/bin/env python3
"""GpuPhysicsArena fixed-address layout contract (plan Rev.7.1 Task 2).

The arena's whole value is a promise no test of the numbers can check: that
every device address is decided once, before any graph is captured, and never
moves.  A CUDA graph bakes its kernel arguments, so a pointer that can move
silently invalidates a captured graph -- and the symptom is a wrong answer or a
launch failure many phases later, nowhere near the allocation.  So:

  1. LAYOUT INVARIANTS, compiled and run with no CUDA at all
     (test/gpu_physics_arena_layout.cpp): 256 B alignment in every slot,
     non-overlap inside a slot, slot isolation, immutable blocks out of every
     slot's reach, the Sec 4.2 per-slot-per-phase aliasing rule, and the Sec 3.6
     budget (200-225 MiB/slot, 12.8-14.4 GiB at 64 slots);
  2. NO HOT-PATH ALLOCATION: reserve() is the only function in the CUDA arm that
     allocates, and nothing else calls cudaMalloc/cudaFree/cudaMallocManaged;
  3. the Sec 4.1 allocation policy is what the plan says -- a memory pool, ONE
     cudaMallocFromPoolAsync, a synchronisation before any pointer is published,
     and a release threshold so the driver keeps the pages;
  4. Sec 4.4 admission exists, is loud, and does NOT reduce the slot count;
  5. the [RASBERY][GPU_ARENA] receipt carries the keys Sec 9.3 needs;
  6. stub parity: every arena symbol is defined in the no-CUDA translation unit,
     and the layout half still answers truthfully there;
  7. no compression tier and no persistent/cooperative scaffolding: Sec 3.6
     removed the first, W0 (c_barrier = 0.78 us vs the 0.384 us kill threshold)
     closed the second.
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

LAYOUT = (SRC / "GpuPhysicsArenaLayout.h").read_text(encoding="utf-8-sig")
ARENA_H = (SRC / "GpuPhysicsArena.h").read_text(encoding="utf-8-sig")
ARENA_CU = (SRC / "GpuPhysicsArenaCuda.cu").read_text(encoding="utf-8-sig")
STUB = (SRC / "GpuPhysicsBackendStub.cpp").read_text(encoding="utf-8-sig")
LAYOUT_TEST = (TEST / "gpu_physics_arena_layout.cpp").read_text(encoding="utf-8-sig")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def strip_literals(text: str) -> str:
    """Blank out string literals, so a name that only appears inside a message
    (`cudaWhy("cudaMallocFromPoolAsync", rc)`) cannot count as a call site."""
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


LAYOUT_CODE = strip_comments(LAYOUT)
ARENA_H_CODE = strip_comments(ARENA_H)
CU_CODE = strip_comments(ARENA_CU)
CU_BARE = strip_literals(CU_CODE)
STUB_CODE = strip_comments(STUB)

problems: list[str] = []


def need(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def body_after(anchor: str, text: str) -> str:
    """The brace-matched block that opens at the first '{' after `anchor`."""
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
# 1. The layout calculator is pure and testable without CUDA.
# ---------------------------------------------------------------------------
need("cuda_runtime.h" not in LAYOUT_CODE and re.search(r"\bcuda[A-Z]\w*\s*\(", LAYOUT_CODE) is None,
     "GpuPhysicsArenaLayout.h touches CUDA; it must be pure host arithmetic")
need("cuda_runtime.h" not in LAYOUT_TEST and "#include <cuda" not in LAYOUT_TEST,
     "test/gpu_physics_arena_layout.cpp includes a CUDA header; it must build CPU-only")
need('#include "GpuPhysicsArenaLayout.h"' in LAYOUT_TEST,
     "the layout test does not include the layout header")

need("kArenaAlignment = 256" in LAYOUT_CODE, "the arena alignment is not 256 B")
for symbol in ("ArenaDims", "ArenaOffsets", "ArenaRegion", "ArenaAdmission",
               "arenaComputeLayout", "arenaAdmit", "arenaAlignUp",
               "arenaScratchPhaseAllowed", "arenaScratchBand", "kScratchSpecs"):
    need(symbol in LAYOUT_CODE, f"GpuPhysicsArenaLayout.h lacks {symbol}")

need("inline constexpr ArenaDims arenaDims(int nxyz, int nsurf, int nxy, int n_fuel, int slots)"
     in LAYOUT_CODE,
     "the (nxyz, nsurf, nxy, n_fuel, slots) layout entry point is missing or renamed")

# The scratch table is DATA, keyed per slot per phase (Sec 4.2), not a comment.
need("owner_phases" in LAYOUT_CODE,
     "the scratch table has no owner-phase mask; the debug trap has nothing to check")
for band in ("KrylovOuter", "NodalDepletion", "ThOutputPpr", "Cram"):
    need(band in LAYOUT_CODE, f"the Sec 4.2 alias band {band} is missing")
for user in ("BicgAx", "BicgS", "WielandtTerms", "OuterPartials", "NodalTrlScratch",
             "DepletionTemp", "ThChannel", "OutputPackScratch", "PprCorner",
             "CramPredictor", "CramCorrector"):
    need(user in LAYOUT_CODE, f"the Sec 4.2 scratch user {user} is missing")
need("DepletionPredictor" in LAYOUT_CODE and "DepletionCorrector" in LAYOUT_CODE,
     "the scratch table does not key the CRAM band on the depletion phases")

# Sec 3.6: full-size arrays.  A compression tier would show up as a scaling
# factor or a "compress" spelling; neither may exist.
for token in ("compress", "Compress", "COMPRESS"):
    need(token not in LAYOUT_CODE, f"GpuPhysicsArenaLayout.h mentions {token}; Sec 3.6 removed the tier")
# Raised from 225 to 256 MiB in Task 3: the measured layout is 224.06 MiB, so
# the old ceiling left 0.4% of headroom and any W2 array would have tripped
# admission on arithmetic alone.  The band the LAYOUT must stay in (200-225 MiB,
# checked below by the layout gate) is unchanged -- only the refusal threshold
# moved, and it must stay above the band or the check would be vacuous.
need(re.search(r"kArenaPerSlotByteCeiling\s*=\s*256", LAYOUT_CODE) is not None,
     "the Sec 4.4 per-slot ceiling is not 256 MiB (Task 3 raised it, reserving W2 growth)")
need("256" in LAYOUT_CODE and "W2" in LAYOUT,
     "the raised ceiling is not documented as W2 growth headroom")
need(re.search(r"kArenaDriverReserveFraction\s*=\s*0\.10", LAYOUT_CODE) is not None
     and re.search(r"kArenaFragmentationReserveFraction\s*=\s*0\.10", LAYOUT_CODE) is not None,
     "the Sec 4.4 10% + 10% reserves are missing")

# The Rev.7.1 arrays Rev.7's table omitted.
for region in ("RefMicx", "RefLmpx", "XeAaHistory", "BosMicx"):
    need(region in LAYOUT_CODE, f"the slot region {region} is missing (Sec 3.5/3.6)")
need("kBosMicroXtCount * niso * ng * nxyz" in LAYOUT_CODE,
     "the BOS microscopic snapshot is not sized at the four Sec 6.18 slots")

# ---------------------------------------------------------------------------
# 2. No hot-path allocation.
# ---------------------------------------------------------------------------
ALLOCATORS = ("cudaMalloc(", "cudaMallocManaged(", "cudaMallocHost(", "cudaMallocPitch(",
              "cudaHostAlloc(")
for allocator in ALLOCATORS:
    need(allocator not in CU_BARE,
         f"GpuPhysicsArenaCuda.cu calls {allocator}; the arena takes ONE pooled allocation")

reserve_body = body_after("bool GpuPhysicsArena::reserve(const ArenaDims& dims)", CU_CODE)
need("cudaMallocFromPoolAsync" in reserve_body,
     "reserve() does not take the allocation from the pool")
need(len(re.findall(r"cudaMallocFromPoolAsync\s*\(", CU_BARE)) == 1,
     "cudaMallocFromPoolAsync is called more than once; the arena allocates exactly once")
release_body = body_after("void GpuPhysicsArena::release()", CU_CODE)
need("cudaFreeAsync" in release_body, "release() does not free the block")
need(len(re.findall(r"cudaFreeAsync\s*\(", CU_BARE)) == 1,
     "cudaFreeAsync is called outside release(); the block is freed exactly once")

# Every accessor and every transfer is an offset add plus at most a memcpy.
for accessor in ("geometryRegion", "libraryRegion", "slotRegion", "scratch", "slotView"):
    body = body_after(f"GpuPhysicsArena::{accessor}(", CU_CODE)
    for allocator in ALLOCATORS + ("cudaMallocFromPoolAsync", "cudaFree"):
        need(allocator not in body, f"{accessor}() allocates; it must be an offset add")

# ---------------------------------------------------------------------------
# 3. Sec 4.1 allocation policy.
# ---------------------------------------------------------------------------
need("cudaMemPoolCreate" in reserve_body or "cudaDeviceGetDefaultMemPool" in reserve_body,
     "reserve() does not obtain a memory pool")
need("cudaMemPoolAttrReleaseThreshold" in reserve_body,
     "reserve() does not pin the pool release threshold; the driver would return the pages")
need("cudaStreamSynchronize" in reserve_body,
     "reserve() never synchronises; the base address would not be final before graph work")
# The synchronisation has to come AFTER the allocation, or it proves nothing.
need(reserve_body.find("cudaMallocFromPoolAsync") < reserve_body.find("cudaStreamSynchronize"),
     "reserve() synchronises before it allocates")

# ---------------------------------------------------------------------------
# 4. Sec 4.4 admission: present, loud, and not a silent narrowing.
# ---------------------------------------------------------------------------
need("cudaMemGetInfo" in reserve_body, "reserve() does not ask the driver what is free")
need("arenaAdmit" in reserve_body, "reserve() does not run Sec 4.4 admission")
need("[RASBERY][GPU_ARENA][FAIL]" in ARENA_CU,
     "an admission refusal is not announced; Sec 4.4 requires it to fail loud")
# Nothing in the CUDA arm may rewrite the slot count: the only write is the
# whole-layout assignment in reserve(), which comes FROM arenaComputeLayout.
need(re.search(r"slot_count\s*=(?!=)", CU_BARE) is None,
     "the CUDA arm assigns slot_count; admission must refuse, not narrow")
need(re.search(r"\bslots\s*(?:=(?!=)|--|-=)", CU_BARE) is None,
     "the CUDA arm mutates a slot count; admission must refuse, not narrow")

admit_body = body_after("inline constexpr ArenaAdmission arenaAdmit(", LAYOUT_CODE)
need("per_slot_over_ceiling" in admit_body, "admission does not check the Sec 3.6 per-slot ceiling")
need("driver_reserve_bytes" in admit_body and "fragmentation_reserve_bytes" in admit_body,
     "admission does not compute both Sec 4.4 reserves")

# ---------------------------------------------------------------------------
# 5. Receipt schema (Sec 9.3).
# ---------------------------------------------------------------------------
RECEIPT_KEYS = ("shared_geometry_bytes", "shared_library_bytes", "per_slot_bytes", "slot_count",
                "scratch_bytes", "total_bytes", "alignment", "driver_reserve_bytes",
                "fragmentation_reserve_bytes", "per_slot_ceiling_bytes", "admitted")
for key in RECEIPT_KEYS:
    need(f'\\"{key}\\"' in ARENA_CU or f'"{key}"' in ARENA_CU,
         f"the CUDA arena receipt lacks {key}")
    need(f'\\"{key}\\"' in STUB or f'"{key}"' in STUB,
         f"the stub arena receipt lacks {key}")
need("[RASBERY][GPU_ARENA] " in ARENA_CU and "[RASBERY][GPU_ARENA] " in STUB,
     "the [RASBERY][GPU_ARENA] receipt tag is missing from one of the arms")

# ---------------------------------------------------------------------------
# 6. Stub parity.
# ---------------------------------------------------------------------------
arena_class = ARENA_H_CODE[ARENA_H_CODE.find("class GpuPhysicsArena"):]
arena_class = arena_class[: arena_class.find("\n};")]
# Declarations only: a name followed by (...) and a SEMICOLON.  Inline bodies
# are dropped first, or `{ return arenaScratchPhaseAllowed(...); }` would
# contribute the free function it calls, which no arm defines as a member.
# A declaration can wrap across lines, so the class text is squashed to one
# line and split on the statement separator rather than on newlines.
arena_flat = re.sub(r"\{[^{}]*\}", "", re.sub(r"\s+", " ", arena_class))
declared = set()
for statement in arena_flat.split(";"):
    match = re.search(r"\b(\w+)\s*\([^)]*\)\s*(?:const\s*)?$", statement.strip())
    if match is None:
        continue
    name = match.group(1)
    if name in ("GpuPhysicsArena", "delete"):
        continue
    declared.add(name)
need("reserve" in declared and "slotView" in declared and "importSlotAsync" in declared,
     "the arena facade lost one of its core methods")
for method in sorted(declared):
    if method == "scratchPhaseAllowed":
        continue  # defined inline in the header, both arms share it
    need(re.search(r"GpuPhysicsArena::" + method + r"\s*\(", CU_CODE) is not None,
         f"GpuPhysicsArenaCuda.cu does not define GpuPhysicsArena::{method}")
    need(re.search(r"GpuPhysicsArena::" + method + r"\s*\(", STUB_CODE) is not None,
         f"GpuPhysicsBackendStub.cpp does not define GpuPhysicsArena::{method}")

need("arenaComputeLayout" in STUB_CODE,
     "the stub arena does not compute the layout; a CPU-only build cannot answer the budget")
need(re.search(r"bool GpuPhysicsArena::available\(\)\s*const\s*\{\s*return false;\s*\}", STUB_CODE)
     is not None,
     "the stub arena's available() does not return false")
need("cuda_runtime.h" not in STUB_CODE and re.search(r"\bcuda[A-Z]\w*\s*\(", STUB_CODE) is None,
     "the stub translation unit reaches for CUDA")

# ---------------------------------------------------------------------------
# 7. No persistent / cooperative scaffolding (W0 closed that track).
# ---------------------------------------------------------------------------
FORBIDDEN_PERSISTENT = ("cooperative_groups", "grid_group", "this_grid", "grid.sync",
                        "cudaLaunchCooperativeKernel", "PersistentKernel", "persistent_kernel")
for path, text in (("GpuPhysicsArenaLayout.h", LAYOUT_CODE), ("GpuPhysicsArena.h", ARENA_H_CODE),
                   ("GpuPhysicsArenaCuda.cu", CU_CODE)):
    for token in FORBIDDEN_PERSISTENT:
        need(token not in text,
             f"{path} carries persistent/cooperative scaffolding ({token}); W0 closed that track")

# ---------------------------------------------------------------------------
# 8. Build wiring.
# ---------------------------------------------------------------------------
need("GpuPhysicsArenaCuda.cu" in CMAKE, "CMakeLists.txt does not compile the CUDA arena")
need("rasbery_gpu_arena_layout" in CMAKE, "CMakeLists.txt does not build the layout test")
need("add_test(NAME gpu_arena_layout" in CMAKE, "the layout test is not registered with ctest")

# ---------------------------------------------------------------------------
# 9. Compile and run the layout gate.
# ---------------------------------------------------------------------------
LAYOUT_BAND_MIB = (200.0, 225.0)
LAYOUT_TOTAL_GIB = (12.8, 14.4)


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
        exe = tmp_path / ("arena_layout.exe" if os.name == "nt" else "arena_layout")
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                script = tmp_path / "build_arena_layout.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul\r\n' % compiler
                    # C4324 is the alignas tail padding on DeviceSlotState, which
                    # Task 1 Step 4 requires; it is suppressed at the source but
                    # restated here so a toolchain that ignores the pragma does
                    # not turn an intended layout into a build failure.
                    + 'cl /nologo /std:c++20 /EHsc /W4 /WX /wd4324 "%s" /I "%s" /Fe:"%s"\r\n'
                      % (TEST / "gpu_physics_arena_layout.cpp", SRC, exe),
                    encoding="utf-8")
                subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(tmp_path),
                               capture_output=True, universal_newlines=True)
            else:
                subprocess.run(
                    [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                     "-I", str(SRC), str(TEST / "gpu_physics_arena_layout.cpp"), "-o", str(exe)],
                    check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            output = (failure.stdout or "") + (failure.stderr or "")
            return ["layout gate did not compile: " + output.strip()[-2000:]]

        done = subprocess.run([str(exe), "--print"], capture_output=True, universal_newlines=True)
        if done.returncode != 0:
            return ["layout gate failed: " + (done.stderr or done.stdout).strip()[-2000:]]

        found: list[str] = []
        per_slot = re.findall(r"per_slot=([0-9.]+) MiB", done.stdout)
        total = re.findall(r"total=([0-9.]+) GiB", done.stdout)
        if len(per_slot) != 3 or len(total) != 3:
            return ["the layout gate did not report all three slot counts: " + done.stdout]
        for value in per_slot:
            mib = float(value)
            if not (LAYOUT_BAND_MIB[0] <= mib <= LAYOUT_BAND_MIB[1]):
                found.append(f"per-slot footprint {mib} MiB is outside the Sec 3.6 band "
                             f"{LAYOUT_BAND_MIB}")
        gib64 = float(total[2])
        if not (LAYOUT_TOTAL_GIB[0] <= gib64 <= LAYOUT_TOTAL_GIB[1]):
            found.append(f"64-slot total {gib64} GiB is outside the Sec 3.6 range {LAYOUT_TOTAL_GIB}")
        print("gpu arena layout budget: " + done.stdout.strip().splitlines()[2])
        return found


def main() -> int:
    compiler = (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
                or msvc_vcvars())
    if compiler is not None:
        problems.extend(build_and_run(compiler))
    if problems:
        for problem in problems:
            print("gpu arena contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if compiler is None:
        print("gpu arena contract: static contract PASS "
              "(no C++ compiler here -- the layout gate was skipped)")
    else:
        print("gpu arena contract: PASS (static contract + layout gate, %s)"
              % Path(compiler).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
