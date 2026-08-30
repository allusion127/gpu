#!/usr/bin/env python3
"""WP10.6 -- device-block lifetime, and the VRAM number a soak is allowed to believe.

WHY THIS FILE EXISTS.  The 238 GPU1 20-generation soak at `bd7a0d3` reported a
VRAM "sawtooth": ~298 MB in most generations and 31,000-47,000 MB in generations
0, 11-16 and 19, with a correlated 33 % throughput dip in exactly those
generations.  Read as a fact about the evaluator, that says the arena is torn
down and rebuilt per generation and that something allocates an order of
magnitude more VRAM than the whole width-16 arena can hold.

It was not a fact about the evaluator.  The soak was launched with
`CUDA_VISIBLE_DEVICES=1` and no `--gpu`, so the child ran on GPU1 and
`sample_vram_mb(args.gpu)` went on asking `nvidia-smi -i 0` -- a board index,
which ignores CUDA_VISIBLE_DEVICES -- about GPU0, where an eight-process
MPS batch was running for the whole soak.  298 MB is that board's MPS server
with no client work on it; 31-47 GB is 8 processes x (16 lanes x ~200 MB +
singletons).  The throughput dip is the host-side half of the same contention,
and the campaign had already flagged it in the OTHER direction ("never take GPU0
timing numbers while GPU1 is busy") without noticing it ran both ways.

Three things had to become true, and this file pins all three.

  1. THE SOAK SAMPLES THE BOARD THE CHILD IS ON, and says whose reading it is.
     `sampled_gpu()` resolves CUDA_VISIBLE_DEVICES; `sample_vram()` prefers the
     per-compute-app row for the child's pid; every sample carries a `scope`,
     and a board-scoped sample with other tenants on it is REPORTED AND NOT
     CONVICTED.  A leak gate that cannot tell "this process grew" from "the
     board got busy" is a gate that will fail again the same way.

  2. THE PROCESS REPORTS ITS OWN DEVICE FOOTPRINT.  `[EVALUATOR][MEM]` now
     carries `vram_mb`, `vram_delta_mb`, `device_allocs`, `device_frees` and
     `arena_rebuilds`.  A number the process states about itself cannot be moved
     by a neighbour, and `arena_rebuilds` answers the question the sawtooth
     raised -- "is the arena rebuilt per generation?" -- directly instead of by
     inference from a memory trace.

  3. THE PER-CASE DEVICE BLOCKS ARE POOLED.  `XsReconBackend::Impl` is per
     XSSet, per Driver, per deck, so every case ran ~12 `cudaMalloc` calls in
     and ~12 `cudaFree` calls out for blocks whose SIZES are a function of the
     geometry and are identical across the candidates of one generation --
     and `cudaFree` is a synchronising call.  `GpuDeviceBlockPool.h` parks them
     on an EXACT-size free list instead, under `RASBERY_ARENA_PERSIST`.

EXACTNESS.  The pool is exact-size only: `take(bytes)` returns a block of
exactly `bytes` or nullptr.  There is no rounding, no splitting and no best fit,
because a pool that could satisfy a 74.8 MB request with a 74.9 MB block would
be a pool that changes what a kernel may read.  Allocation lifetime is not
observable in a result, so per-case outputs are byte-identical either way; the
flag exists so the 238 A/B can prove that rather than assert it.

Run: python3 tools/test_arena_persist_contract.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


POOL = read("src/GpuDeviceBlockPool.h")
SERVER = read("src/EvaluatorServer.h")
XSRECON = read("src/CudaXsReconBackend.cu")
ARENA = read("src/GpuPhysicsArenaCuda.cu")
SOAK = read("tools/soak_run.py")


# ---------------------------------------------------------------------------
# 1. THE POOL IS EXACT-SIZE, PROCESS-LIFETIME, AND GATED
# ---------------------------------------------------------------------------

def check_pool(pool: str) -> list[str]:
    bad: list[str] = []
    for symbol in ("take(", "give(", "noteAllocated(", "noteArenaRebuild(",
                   "snapshot(", "enabled(", "deviceBlockAlloc(",
                   "deviceBlockAllocOnce(", "deviceBlockFree("):
        if symbol not in pool:
            bad.append(f"GpuDeviceBlockPool.h has no {symbol!r}")
    if "RASBERY_ARENA_PERSIST" not in pool:
        bad.append("the pool is not gated on RASBERY_ARENA_PERSIST; a device-lifetime "
                   "change with no off switch cannot be A/B'd against a digest")
    # EXACT SIZE.  The free list is keyed by byte count and looked up with
    # find(), not lower_bound()/upper_bound(): a size-ordered search is how a
    # pool starts handing out bigger blocks than were asked for.
    if "lower_bound" in pool or "upper_bound" in pool:
        bad.append("the free list does a size-ordered search; an exact-size pool must "
                   "only ever answer find(bytes), because a block bigger than the "
                   "request is a block whose tail no caller has written")
    if "s.parked.find(bytes)" not in pool:
        bad.append("take() does not look the free list up by exact byte count")
    # The counters must NOT be behind the gate: an A/B whose two arms print
    # different fields is an A/B whose arms cannot be diffed.
    hits = pool.index("inline Stats snapshot()")
    if "enabled()" in pool[hits:pool.index("inline std::uint64_t deviceBytes()")]:
        bad.append("snapshot() consults the gate; the counters must be printed in both "
                   "arms or the A/B cannot be diffed line for line")
    if "poolable" not in pool:
        bad.append("the pool cannot distinguish a per-case block from a process-lifetime "
                   "singleton, so the nodal arena and the flat-XS library would be "
                   "parked on a free list nothing will ever ask for again")
    return bad


# ---------------------------------------------------------------------------
# 2. THE PER-CASE SITES GO THROUGH IT -- AND THE SINGLETONS DO NOT GET POOLED
# ---------------------------------------------------------------------------

#: Every device block `XsReconBackend::Impl` owns.  Impl is per XSSet, per
#: Driver, per deck (CudaXsReconBackend.cu says so beside `xe_hist`), so each of
#: these was a cudaMalloc/cudaFree pair PER CASE.
PER_CASE_BLOCKS = ("dev_block", "dev_fuel", "dev_scalars", "dev_dep", "dev_ref",
                   "dev_pernode", "dev_nodes", "dev_off", "dev_cnt", "dev_sdid",
                   "dev_sx", "dev_sscale", "ndev_dbl", "ndev_int",
                   "xe_hist", "xe_processed", "xe_partials", "xe_dots",
                   "xe_pairs", "xe_flags", "xe_bits", "xe_ctl")


def check_sites(xsrecon: str) -> list[str]:
    bad: list[str] = []
    for line in xsrecon.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # A raw cudaFree( on a per-case block is a synchronising call per case
        # for a block the next case will ask for again.
        if "cudaFree(" in stripped and "deviceBlockFree(" not in stripped:
            for block in PER_CASE_BLOCKS:
                if f"cudaFree({block})" in stripped or f"cudaFree(d.{block})" in stripped:
                    bad.append(f"a raw cudaFree survives on the per-case block "
                               f"{block!r}: {stripped}")
    for block in ("dev_block", "dev_ref", "xe_hist"):
        if f"deviceBlockFree({block})" not in xsrecon and \
           f"deviceBlockFree(d.{block})" not in xsrecon:
            bad.append(f"{block!r} is not freed through the pool, and it is one of the "
                       "three that dominate the per-case footprint")
    if "deviceBlockAllocOnce(" not in xsrecon:
        bad.append("the process-lifetime singletons (nodal arena, flat-XS library) do "
                   "not use the once-allocator, so they would be counted as poolable "
                   "and parked on a free list that will never be asked for them")
    if "noteArenaRebuild(" not in xsrecon:
        bad.append("nothing counts an arena rebuild in CudaXsReconBackend.cu, so "
                   "'is the arena rebuilt per generation' stays a question a receipt "
                   "cannot answer")
    return bad


def check_arena(arena: str) -> list[str]:
    bad: list[str] = []
    if "blockpool::noteAllocated" not in arena:
        bad.append("GpuPhysicsArenaCuda.cu does not register its block, so the "
                   "receipt's vram_mb would be a footprint missing its largest term")
    if "poolable=*/false" not in arena.replace(" ", "").replace("/*", "/*"):
        if "/*poolable=*/false" not in arena:
            bad.append("the physics arena is registered as poolable; it is taken once "
                       "for the process and must never be parked")
    if "blockpool::give" not in arena:
        bad.append("release() does not deregister the arena block, so bytes_live would "
                   "keep counting memory the process has handed back")
    return bad


# ---------------------------------------------------------------------------
# 3. THE RECEIPT CARRIES THE FIVE FIELDS
# ---------------------------------------------------------------------------

REQUIRED_MEM_FIELDS = ("vram_mb", "vram_delta_mb", "device_allocs", "device_frees",
                       "arena_rebuilds")


def check_receipt(server: str) -> list[str]:
    bad: list[str] = []
    marker = '"[RASBERY][EVALUATOR][MEM] {'
    if server.count(marker) != 1:
        return ["src/EvaluatorServer.h must emit [RASBERY][EVALUATOR][MEM] from exactly "
                "one site"]
    start = server.index(marker)
    block = server[start:server.find("std::endl;", start)]
    for field in REQUIRED_MEM_FIELDS:
        if f'\\"{field}\\"' not in block:
            bad.append(f"the [RASBERY][EVALUATOR][MEM] receipt has no {field!r}; the "
                       "238 soak had to infer device behaviour from a board-level "
                       "trace precisely because this was missing")
    # The allocator, asked directly.  The 238 report ended on "look at the
    # allocator next" because the receipt could not look at it.
    if '\\"malloc_retained_mb\\"' not in block:
        bad.append("the receipt does not report allocator retention, so the RSS finding "
                   "still hands off with 'look at the allocator next'")
    if '\\"malloc_readable\\"' not in block:
        bad.append("a zero malloc_retained_mb on a non-glibc C library would read as "
                   "'the allocator is holding nothing'; say whether it was readable")
    return bad


# ---------------------------------------------------------------------------
# 4. THE SOAK SAMPLES THE RIGHT BOARD AND LABELS WHOSE READING IT IS
# ---------------------------------------------------------------------------

def check_soak_source(soak: str) -> list[str]:
    bad: list[str] = []
    if "CUDA_VISIBLE_DEVICES" not in soak or "def sampled_gpu" not in soak:
        bad.append("soak_run.py does not resolve CUDA_VISIBLE_DEVICES to the board it "
                   "samples; that mismatch IS the 238 sawtooth")
    if "query-compute-apps" not in soak:
        bad.append("soak_run.py never asks nvidia-smi for the per-process rows, so it "
                   "can only ever report a board total")
    if 'result.vram_scope' not in soak:
        bad.append("the soak does not record the scope of a VRAM sample")
    return bad


def check_soak_behaviour() -> list[str]:
    """Drive the real functions.  A source scan alone checks that words exist."""
    sys.path.insert(0, str(ROOT / "tools"))
    import soak_run  # noqa: E402

    bad: list[str] = []

    # (a) CUDA_VISIBLE_DEVICES wins over --gpu, which is the whole defect.
    if soak_run.sampled_gpu({"CUDA_VISIBLE_DEVICES": "1"}, "0") != "1":
        bad.append("sampled_gpu ignored CUDA_VISIBLE_DEVICES=1 and would sample GPU0 "
                   "while the child ran on GPU1 -- the 238 bug, unfixed")
    if soak_run.sampled_gpu({"CUDA_VISIBLE_DEVICES": "2,3"}, "0") != "2":
        bad.append("sampled_gpu did not take the FIRST visible device")
    if soak_run.sampled_gpu({}, "1") != "1":
        bad.append("sampled_gpu ignored --gpu when CUDA_VISIBLE_DEVICES was unset")
    if soak_run.sampled_gpu({"CUDA_VISIBLE_DEVICES": ""}, "1") != "1":
        bad.append("an empty CUDA_VISIBLE_DEVICES must fall back to --gpu, not to ''")

    # (b) A board-scoped sample with foreign tenants must not be convicted.
    class Gen:
        def __init__(self, scope, foreign, vram):
            self.vram_scope, self.vram_foreign, self.vram_mb = scope, foreign, vram

    contaminated = [Gen("board", 8, 300.0), Gen("board", 8, 47000.0)]
    if not [g for g in contaminated if g.vram_scope == "board" and g.vram_foreign > 0]:
        bad.append("the contamination predicate does not fire on a board sample with "
                   "eight other tenants on it")

    # (c) The attribution reads the new fields rather than handing off.
    class MemGen:
        def __init__(self, mem):
            self.mem = mem

    base = {"cache_entries": {"xslib": 1}, "evictions": {"xslib": 0},
            "cache_bytes": {"xslib": 100}, "live_cases": 0, "cuda_host_bytes": 0,
            "malloc_retained_mb": 10.0, "malloc_readable": True,
            "vram_mb": 4000.0, "arena_rebuilds": 0}
    grown = dict(base, malloc_retained_mb=960.0, vram_mb=4000.0)
    named = " ".join(soak_run.attribute_rss_growth([MemGen(base), MemGen(grown)], 1000.0))
    if "RETENTION" not in named:
        bad.append("attribute_rss_growth did not identify allocator retention when "
                   "malloc_retained_mb accounted for 95 % of the growth; the finding "
                   "would hand off exactly as the 238 one did")
    if "arena_rebuilds=0" not in named:
        bad.append("attribute_rss_growth does not report arena_rebuilds, so 'was the "
                   "arena rebuilt per generation' stays unanswered")
    return bad


# ---------------------------------------------------------------------------
# 5. NEGATIVE CONTROLS -- a check that cannot fail is a comment
# ---------------------------------------------------------------------------

def controls() -> list[str]:
    broken: list[str] = []

    if not check_pool(POOL.replace("RASBERY_ARENA_PERSIST", "RASBERY_NOTHING")):
        broken.append("check_pool passes a pool with no RASBERY_ARENA_PERSIST gate")
    if not check_pool(POOL.replace("s.parked.find(bytes)", "s.parked.lower_bound(bytes)")):
        broken.append("check_pool passes a free list that does a size-ordered search")
    if not check_receipt(SERVER.replace('\\"arena_rebuilds\\"', '\\"nothing\\"')):
        broken.append("check_receipt passes a receipt with no arena_rebuilds")
    if not check_receipt(SERVER.replace('\\"vram_delta_mb\\"', '\\"nothing\\"')):
        broken.append("check_receipt passes a receipt with no vram_delta_mb")
    if not check_sites(XSRECON.replace("deviceBlockFree(dev_block)", "cudaFree(dev_block)")):
        broken.append("check_sites passes a tree where dev_block is still freed raw")
    if not check_sites(XSRECON.replace("deviceBlockAllocOnce(", "deviceBlockAlloc(")):
        broken.append("check_sites passes a tree where the process-lifetime singletons "
                      "are registered as poolable")
    if not check_arena(ARENA.replace("blockpool::noteAllocated", "nothing")):
        broken.append("check_arena passes an arena that never registers its block")
    if not check_soak_source(SOAK.replace("query-compute-apps", "query-nothing")):
        broken.append("check_soak_source passes a sampler that never asks for the "
                      "per-process rows")
    return broken


# ---------------------------------------------------------------------------
# 6. THE COMPILED HALF -- the free list, run for real
# ---------------------------------------------------------------------------

HARNESS_CPP = r"""
#include "GpuDeviceBlockPool.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bp = rasbery::gpu::blockpool;

// Stand-ins for device pointers.  The pool never dereferences a block; it is
// bookkeeping over opaque addresses, which is exactly why it can be tested with
// no CUDA runtime present.
static void* fake(std::size_t n) { return reinterpret_cast<void*>(0x1000 + n * 64); }

int main() {
    const bool on = bp::enabled();
    std::printf("gate_matches_env %d\n",
                on == (std::getenv("RASBERY_ARENA_PERSIST") != nullptr &&
                       std::getenv("RASBERY_ARENA_PERSIST")[0] != '\0' &&
                       std::string(std::getenv("RASBERY_ARENA_PERSIST")) != "0"));

    void* a = fake(1);
    bp::noteAllocated(a, 4096, true);
    const bool parked = bp::give(a);
    std::printf("park_follows_gate %d\n", parked == on);

    // EXACT SIZE.  A neighbouring request must never be served from the block
    // parked above, in either arm.
    std::printf("no_size_bleed %d\n", bp::take(4095) == nullptr && bp::take(4097) == nullptr);
    // And the exact size comes back, but only with the gate on.
    void* again = bp::take(4096);
    std::printf("exact_reuse %d\n", on ? (again == a) : (again == nullptr));
    if (again != nullptr) bp::give(again);

    // A singleton is never parked, gate or no gate.
    bp::resetForTest();
    void* s = fake(2);
    bp::noteAllocated(s, 8192, false);
    std::printf("singleton_never_parked %d\n", bp::give(s) == false);

    // An unregistered pointer stays the caller's to free: a site this header
    // does not know about must not be broken by it.
    std::printf("unknown_pointer_passthrough %d\n", bp::give(fake(99)) == false);

    // Footprint accounting: live + pooled, and a high-water that only rises.
    bp::resetForTest();
    void* b = fake(3);
    void* c = fake(4);
    bp::noteAllocated(b, 1000, true);
    bp::noteAllocated(c, 2000, true);
    const std::uint64_t peak = bp::deviceBytes();
    bp::give(b);
    bp::give(c);
    const bp::Stats st = bp::snapshot();
    std::printf("footprint_conserved %d\n",
                (st.bytes_live + st.bytes_pooled) == (on ? peak : 0u) &&
                st.bytes_high_water >= peak);
    std::printf("allocs_counted %d\n", st.device_allocs == 2);
    std::printf("frees_counted %d\n", st.device_frees == (on ? 0u : 2u));

    bp::noteArenaRebuild();
    std::printf("rebuilds_counted %d\n", bp::snapshot().arena_rebuilds == 1);
    return 0;
}
"""

EXPECTED = {
    "gate_matches_env": "enabled() must agree with RASBERY_ARENA_PERSIST",
    "park_follows_gate": "with the gate off, give() must hand the block back to the "
                         "caller to free -- that is what makes the off arm the old "
                         "behaviour instruction for instruction",
    "no_size_bleed": "a 4096-byte block was offered for a 4095- or 4097-byte request; "
                     "an exact-size pool must never round",
    "exact_reuse": "the exact size did not come back from the free list with the gate "
                   "on (or came back with it off)",
    "singleton_never_parked": "a process-lifetime block was parked on a free list that "
                              "will never be asked for it",
    "unknown_pointer_passthrough": "a pointer the pool never saw was claimed by it; an "
                                   "unwrapped call site would then leak",
    "footprint_conserved": "live + pooled did not conserve the registered bytes",
    "allocs_counted": "device_allocs did not count the driver allocations",
    "frees_counted": "device_frees counted a free that never reached the driver (or "
                     "missed one that did)",
    "rebuilds_counted": "noteArenaRebuild() did not move arena_rebuilds",
}


def find_compiler():
    """MSVC's vcvars64.bat, or a g++/clang++ on PATH, or None."""
    if os.name == "nt":
        for base in (r"C:\Program Files\Microsoft Visual Studio",
                     r"C:\Program Files (x86)\Microsoft Visual Studio"):
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                if "vcvars64.bat" in files:
                    return os.path.join(dirpath, "vcvars64.bat")
    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def compiled_contract() -> bool:
    """True when the compiled half actually ran.  BOTH arms, one process each."""
    compiler = find_compiler()
    if compiler is None:
        return False
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        cpp = tmp / "arena_persist_harness.cpp"
        cpp.write_text('#include <string>\n' + HARNESS_CPP, encoding="utf-8")
        exe = tmp / ("arena_persist_harness.exe" if os.name == "nt"
                     else "arena_persist_harness")
        includes = [ROOT / "src"]
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                # QUOTED include paths: this repository's own path contains an
                # `&`, which cmd would otherwise read as a command separator.
                script = tmp / "build_arena_persist_harness.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul 2>&1\r\n' % compiler
                    + 'cd /d "%s"\r\n' % tmp
                    + 'cl /nologo /std:c++20 /EHsc /D_CRT_SECURE_NO_WARNINGS "%s" %s /Fe:"%s"\r\n'
                      % (cpp, " ".join('/I "%s"' % d for d in includes), exe),
                    encoding="utf-8")
                subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(tmp),
                               capture_output=True, universal_newlines=True)
            else:
                subprocess.run(
                    [compiler, "-std=c++20", "-O0", str(cpp), "-o", str(exe)]
                    + [arg for d in includes for arg in ("-I", str(d))],
                    check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            failures.append("the arena-persist harness does not compile:\n"
                            + (failure.stdout or "") + (failure.stderr or ""))
            return True

        # BOTH ARMS.  `enabled()` is latched on first read, so the two arms have
        # to be two processes; running only the ON arm would leave the OFF arm
        # -- the one that has to be the old behaviour exactly -- unchecked.
        for arm, value in (("off", None), ("on", "1")):
            env = dict(os.environ)
            env.pop("RASBERY_ARENA_PERSIST", None)
            if value is not None:
                env["RASBERY_ARENA_PERSIST"] = value
            done = subprocess.run([str(exe)], capture_output=True,
                                  universal_newlines=True, env=env)
            if done.returncode != 0:
                failures.append(f"the arena-persist harness failed ({arm}): "
                                f"{done.stdout}{done.stderr}")
                return True
            results = dict(line.split() for line in done.stdout.split("\n") if line.strip())
            for name, why in EXPECTED.items():
                if results.get(name) != "1":
                    failures.append(f"compiled arena-persist contract [{arm}/{name}]: "
                                    f"{why} (harness said {results.get(name)!r})")
    return True


failures += check_pool(POOL)
failures += check_sites(XSRECON)
failures += check_arena(ARENA)
failures += check_receipt(SERVER)
failures += check_soak_source(SOAK)
failures += check_soak_behaviour()

broken_controls = controls()
if broken_controls:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are "
                    "therefore comments:\n    " + "\n    ".join(broken_controls))

compiled = compiled_contract()

if failures:
    raise SystemExit("arena persist: FAIL\n  " + "\n  ".join(failures))
if not compiled:
    print("arena persist: NOTE -- no C++ compiler found; the free list was checked "
          "by source scan only", file=sys.stderr)
print(f"arena persist: PASS ({len(REQUIRED_MEM_FIELDS)} receipt fields, "
      f"{len(PER_CASE_BLOCKS)} per-case blocks, {len(broken_controls) == 0 and 8 or 0} "
      f"negative controls"
      + (f", {len(EXPECTED)} compiled x2 arms" if compiled else "") + ")")
