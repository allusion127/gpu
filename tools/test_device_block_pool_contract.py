#!/usr/bin/env python3
"""Contract: WP10.8 -- the device free list is BOUNDED, KEYED, and says so.

WHAT WENT WRONG, AND WHY A SOURCE SCAN WOULD NOT HAVE CAUGHT IT.  WP10.6 shipped
`src/GpuDeviceBlockPool.h` as "not a cache with a policy": an exact-size map from
byte count to a list of blocks, `take()` or `nullptr`, nothing else.  That is a
correct allocator and an unbounded container.  A block parked at generation 0 for
a geometry no later case asks for again was held to process exit; a size key that
ever carried something per case would have grown a class per case; and NOTHING IN
ANY RECEIPT COULD SAY EITHER HAD HAPPENED.  The 238 block-38 arm-B soak
(`RASBERY_ARENA_PERSIST=1`, 20 generations, width 16) reported RSS climbing
115.97 MB/generation and settled at 6039 MB of device footprint, and the pool was
the obvious suspect that nothing in the report could weigh or exonerate.

So this test asserts four things the WP10.6 pool could not have satisfied:

  1. BOUNDED.  Parking is capped by bytes, by blocks, and by depth per size
     class, and the caps are operator-visible environment variables.  Past a cap
     the pool either EVICTS (least recently parked first, across size classes)
     or REFUSES TO PARK.  It never grows past the bound, and with no device free
     installed it refuses rather than exceeding it -- a bound that depends on a
     reclaimer being present is not a bound.
  2. KEYED, STILL EXACTLY.  Bounding must not have introduced rounding.  A
     4095-byte request must never be served from a 4096-byte parked block, in
     either arm, with or without eviction pressure.
  3. EVICTION IS A RECEIPT, not a silence.  `pool_evictions`, `pool_evicted_mb`,
     `pool_park_refusals`, `pool_size_classes` and `pool_bookkeeping_bytes` reach
     `[RASBERY][EVALUATOR][MEM]`, so the next soak can price the accused instead
     of suspecting it.
  4. THE COUNTERS MEAN WHAT THEY ARE CALLED.  `block_reshapes` is the per-case
     device-block re-layout the old `arena_rebuilds` field actually counted;
     `arena_rebuilds` is now process-lifetime regions handed back, derived from
     the registration flag with no call site cooperating.

NEGATIVE CONTROLS.  Every check is re-run against a deliberately broken copy of
the header or the receipt and must fail there, because a check that cannot fail
is a comment.

USE

    python tools/test_device_block_pool_contract.py
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


# ---------------------------------------------------------------------------
# 1. THE SOURCE: BOUNDED BY CONSTRUCTION, AND THE BOUND IS OPERATOR-VISIBLE
# ---------------------------------------------------------------------------

#: Every cap has to be a named environment variable.  A cap compiled in is a cap
#: that cannot be moved on the host where it turns out to be wrong, and this
#: campaign's hosts are ones nobody can rebuild on at 3 a.m.
CAP_ENVS = ("RASBERY_ARENA_PERSIST_CAP_MB",
            "RASBERY_ARENA_PERSIST_CAP_BLOCKS",
            "RASBERY_ARENA_PERSIST_CLASS_DEPTH")


def check_bounded(pool: str) -> list[str]:
    bad: list[str] = []
    for name in CAP_ENVS:
        if name not in pool:
            bad.append(f"the pool has no {name}: a free list whose bound cannot be "
                       "moved without a rebuild is a bound nobody can act on")
    for symbol in ("capBytes(", "capBlocks(", "capClassDepth(", "withinCapsLocked(",
                   "evictOneLocked(", "setReclaimer(", "park_refusals",
                   "pool_evictions", "bytes_evicted", "bookkeeping_bytes"):
        if symbol not in pool:
            bad.append(f"GpuDeviceBlockPool.h has no {symbol!r}; the free list is the "
                       "WP10.6 one, which could only grow")
    # THE LRU STAMP.  Eviction across size classes needs an order, and a free
    # list with no stamp can only evict within one class -- which is the class
    # the steady state is asking for.
    if "seq" not in pool or "struct Parked" not in pool:
        bad.append("parked blocks carry no recency stamp, so eviction cannot be "
                   "least-recently-parked-first ACROSS size classes and would take "
                   "the working set instead of the stale class")
    # THE RECLAIMER IS NOT ASSUMED.  A pool that parks past its cap whenever no
    # device free is installed is unbounded on exactly the builds that have no
    # CUDA -- which is every host build of EvaluatorServer.h.
    if "reclaim == nullptr" not in pool:
        bad.append("give() does not handle the no-reclaimer case, so the cap holds "
                   "only when a device free happens to have been installed")
    # And the driver call must not happen under the pool's own mutex: cudaFree
    # synchronises the device, and holding a lock across it serialises the lanes.
    give = pool[pool.index("inline bool give("):pool.index("inline void noteBlockReshape(")]
    if "for (void* victim : victims) reclaim(victim);" not in give:
        bad.append("give() calls the reclaimer inside the locked region; cudaFree is a "
                   "synchronising call and every lane would queue behind it")
    return bad


def check_exact_still(pool: str) -> list[str]:
    """Bounding must not have introduced rounding."""
    bad: list[str] = []
    if "lower_bound" in pool or "upper_bound" in pool:
        bad.append("the free list does a size-ordered search; a block bigger than the "
                   "request is a block whose tail no caller has written")
    if "s.parked.find(bytes)" not in pool:
        bad.append("take() does not look the free list up by exact byte count")
    return bad


# ---------------------------------------------------------------------------
# 2. THE RECEIPT: AN EVICTION NOBODY CAN SEE IS A SILENCE
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("pool_cap_mb", "pool_cap_blocks", "pool_class_depth",
                   "pool_size_classes", "pool_evictions", "pool_evicted_mb",
                   "pool_park_refusals", "pool_bookkeeping_bytes",
                   "block_reshapes", "arena_standups", "arena_rebuilds")


def check_receipt(server: str) -> list[str]:
    bad: list[str] = []
    marker = '"[RASBERY][EVALUATOR][MEM] {'
    if server.count(marker) != 1:
        return ["src/EvaluatorServer.h must emit [RASBERY][EVALUATOR][MEM] from exactly "
                "one site"]
    block = server[server.index(marker):server.find("std::endl;", server.index(marker))]
    for field in REQUIRED_FIELDS:
        if f'\\"{field}\\"' not in block:
            bad.append(f"the MEM receipt has no {field!r}; the 238 arm-B RSS finding "
                       "named the pool as a suspect and no field could price it")
    # `arena_rebuilds` must be the DERIVED number now, not the raw counter the
    # three XsRecon sites move.  A field that kept its old value under its old
    # name would have fixed nothing -- the 238 report's wrong conclusion came
    # from reading exactly that.
    if "blockpool::arenaRebuilds(dev)" not in block:
        bad.append("the MEM receipt still prints a raw counter as `arena_rebuilds`; "
                   "WP10.8 makes that field process-lifetime regions handed back, "
                   "which is what its name always promised")
    return bad


# ---------------------------------------------------------------------------
# 3. THE COMPILED HALF -- the bound, the eviction order, and the exactness
# ---------------------------------------------------------------------------

HARNESS_CPP = r"""
#include "GpuDeviceBlockPool.h"

#include <cstdio>
#include <string>
#include <vector>

namespace bp = rasbery::gpu::blockpool;

// Opaque stand-ins.  The pool never dereferences a block, which is why the whole
// policy can be exercised with no CUDA runtime present.
static void* fake(std::size_t n) { return reinterpret_cast<void*>(0x100000 + n * 256); }

static std::vector<void*> g_reclaimed;
static void reclaim(void* p) { g_reclaimed.push_back(p); }

// ---- THE EVICTION ARM -----------------------------------------------------
// RASBERY_ARENA_PERSIST_CAP_MB=0 disables the byte cap and CLASS_DEPTH is left
// wide, so the BLOCK cap is the one under test: CAP_BLOCKS=3.  Park one block of
// each of four DIFFERENT sizes, oldest first; the fourth park must evict the
// FIRST, not one of its own class -- evicting within the class being asked for
// throws away the working set and keeps the stale one.
static int evict_arm() {
    bp::resetForTest();
    bp::setReclaimer(&reclaim);
    g_reclaimed.clear();
    void* a = fake(100); bp::noteAllocated(a, 11, true);
    void* b = fake(101); bp::noteAllocated(b, 22, true);
    void* c = fake(102); bp::noteAllocated(c, 33, true);
    void* d = fake(103); bp::noteAllocated(d, 44, true);
    bp::give(a); bp::give(b); bp::give(c);
    bp::Stats s = bp::snapshot();
    const bool three_classes = (s.blocks_pooled == 3 && s.size_classes == 3 &&
                                s.pool_evictions == 0);
    bp::give(d);
    s = bp::snapshot();
    std::printf("block_cap_evicts %d\n",
                three_classes && s.blocks_pooled == 3 && s.pool_evictions == 1 &&
                s.bytes_evicted == 11);
    std::printf("evicts_least_recently_parked %d\n",
                g_reclaimed.size() == 1 && g_reclaimed[0] == a);
    // The evicted class is gone entirely, so `size_classes` tracks what is
    // actually held rather than what was ever held.
    std::printf("evicted_class_forgotten %d\n",
                s.size_classes == 3 && bp::take(11) == nullptr);
    // And the eviction reached the driver, so device_frees counts it.
    std::printf("evictions_are_driver_frees %d\n", s.device_frees == 1);
    return 0;
}

// TWO PROCESSES, TWO CAP SETS.  `capBytes()`/`capBlocks()`/`capClassDepth()` are
// latched on first read, exactly as `enabled()` is and for the same reason -- a
// cap that could change under a running process is a cap that can park a block
// the free list will never be allowed to hand back.  So one image cannot
// exercise "depth binds" and "block count binds" both, and the runner launches
// this twice with an argument saying which.
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "bounded";
    if (!bp::enabled()) { std::printf("gate_on 0\n"); return 0; }
    std::printf("gate_on 1\n");
    if (mode == "evict") return evict_arm();

    // ---- 1. THE DEPTH CAP HOLDS WITH NO RECLAIMER -------------------------
    // RASBERY_ARENA_PERSIST_CLASS_DEPTH=4 for this process.  Eight blocks of
    // one size: four park, four are refused and go back to the caller.
    bp::resetForTest();
    std::vector<void*> ones;
    for (int i = 0; i < 8; ++i) {
        void* p = fake(i);
        ones.push_back(p);
        bp::noteAllocated(p, 1024, true);
    }
    int parked = 0;
    for (void* p : ones) parked += bp::give(p) ? 1 : 0;
    bp::Stats s = bp::snapshot();
    std::printf("depth_cap_holds %d\n",
                parked == 4 && s.blocks_pooled == 4 && s.park_refusals == 4);
    // A refused park is a free that REACHED THE DRIVER, and device_frees has to
    // say so or the receipt's alloc/free balance is a lie.
    std::printf("refusals_are_driver_frees %d\n", s.device_frees == 4);

    // ---- 2. EXACTNESS SURVIVES THE CAP ------------------------------------
    std::printf("no_size_bleed %d\n",
                bp::take(1023) == nullptr && bp::take(1025) == nullptr);
    void* exact = bp::take(1024);
    std::printf("exact_reuse %d\n", exact != nullptr);
    if (exact != nullptr) bp::give(exact);

    // ---- 4. THE BOOKKEEPING IS WEIGHED ------------------------------------
    // The whole point: "is the pool the RSS growth?" must be arithmetic.
    bp::resetForTest();
    void* one = fake(500);
    bp::noteAllocated(one, 64, true);
    bp::give(one);
    const std::uint64_t small = bp::snapshot().bookkeeping_bytes;
    bp::resetForTest();
    for (int i = 0; i < 200; ++i) {
        void* p = fake(1000 + i);
        bp::noteAllocated(p, 4096 + static_cast<std::size_t>(i), true);
        bp::give(p);
    }
    const std::uint64_t big = bp::snapshot().bookkeeping_bytes;
    std::printf("bookkeeping_grows_with_the_pool %d\n", big > small && small > 0);
    // 200 blocks of bookkeeping is kilobytes, not megabytes.  This is the
    // arithmetic that exonerated the pool for the 238 arm-B RSS finding, and it
    // is asserted rather than asserted-about.
    std::printf("bookkeeping_is_small %d\n", big < 1024u * 1024u);

    // ---- 5. THE COUNTERS MEAN WHAT THEY ARE CALLED ------------------------
    bp::resetForTest();
    bp::noteBlockReshape();
    bp::noteArenaRebuild();  // deprecated alias -- same counter
    s = bp::snapshot();
    std::printf("reshape_alias %d\n",
                s.block_reshapes == 2 && bp::arenaRebuilds(s) == 0);
    void* once = fake(9000);
    bp::noteAllocated(once, 4096, false);
    std::printf("standup_counted %d\n",
                bp::snapshot().arena_standups == 1 &&
                bp::arenaRebuilds(bp::snapshot()) == 0);
    const bool singleton_parked = bp::give(once);
    std::printf("singleton_never_parked %d\n", singleton_parked == false);
    std::printf("teardown_is_a_rebuild %d\n",
                bp::arenaRebuilds(bp::snapshot()) == 1);
    return 0;
}
"""

GATE = {
    "gate_on": "the harness ran with RASBERY_ARENA_PERSIST unset; the pool arm was "
               "never exercised at all",
}

EXPECTED_BOUNDED = {
    "depth_cap_holds":"the free list parked more than RASBERY_ARENA_PERSIST_CLASS_DEPTH "
                       "blocks of one size, or did not count the refusals. A pool that "
                       "can only grow is the WP10.6 pool",
    "refusals_are_driver_frees": "a refused park did not count as a driver free, so the "
                                 "receipt's allocs and frees no longer balance",
    "no_size_bleed": "a 1024-byte block was offered for a 1023- or 1025-byte request; "
                     "bounding the pool must not have introduced rounding",
    "exact_reuse": "the exact size did not come back out of the bounded free list",
    "bookkeeping_grows_with_the_pool":"pool_bookkeeping_bytes does not move with the "
                                       "free list, so it cannot price it",
    "bookkeeping_is_small": "200 parked blocks weigh more than a megabyte of "
                            "bookkeeping; either the accounting is wrong or the pool "
                            "really is an RSS suspect",
    "reshape_alias": "noteArenaRebuild() no longer lands in block_reshapes, so the two "
                     ".cu files that still call it by that name are counting into "
                     "nothing -- or it is still moving arena_rebuilds, which is the "
                     "mislabel WP10.8 exists to close",
    "standup_counted": "a process-lifetime registration did not move arena_standups, or "
                       "moved arena_rebuilds before anything was handed back",
    "singleton_never_parked": "a process-lifetime block was parked on a free list that "
                              "will never be asked for it",
    "teardown_is_a_rebuild": "a process-lifetime region was handed back and "
                             "arena_rebuilds did not count it; that event IS the arena "
                             "teardown the 238 VRAM sawtooth raised",
}

EXPECTED_EVICT = {
    "block_cap_evicts": "RASBERY_ARENA_PERSIST_CAP_BLOCKS did not bound the free list, "
                        "or the eviction was not counted and weighed",
    "evicts_least_recently_parked": "eviction did not take the OLDEST parked block "
                                    "across size classes; evicting within the class "
                                    "being asked for throws away the working set and "
                                    "keeps the stale one",
    "evicted_class_forgotten": "an emptied size class stayed in the map, so "
                               "pool_size_classes reports what was ever held instead of "
                               "what is held -- and that is the field a reader watches "
                               "for a size key that carries something per case",
    "evictions_are_driver_frees": "an eviction did not count as a driver free",
}

#: (argv word, env overrides, expectations).  Two images because the caps latch.
ARMS = (
    ("bounded", {"RASBERY_ARENA_PERSIST_CAP_MB": "0",
                 "RASBERY_ARENA_PERSIST_CAP_BLOCKS": "0",
                 "RASBERY_ARENA_PERSIST_CLASS_DEPTH": "4"}, EXPECTED_BOUNDED),
    ("evict", {"RASBERY_ARENA_PERSIST_CAP_MB": "0",
               "RASBERY_ARENA_PERSIST_CAP_BLOCKS": "3",
               "RASBERY_ARENA_PERSIST_CLASS_DEPTH": "64"}, EXPECTED_EVICT),
)


def find_compiler() -> "str | None":
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


def run_compiled() -> bool:
    """True when the compiled half ran.  False means no compiler, and SAYS so."""
    compiler = find_compiler()
    if compiler is None:
        failures.append(
            "NO C++ COMPILER: the free list's bound, its eviction order and its "
            "exactness are POLICY, and policy that was only source-scanned has not "
            "been run. This is a failure rather than a skip because a green report "
            "from a host that compiled nothing is the report that let the WP10.6 "
            "pool ship unbounded.")
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cpp = tmp / "device_block_pool_harness.cpp"
        cpp.write_text(HARNESS_CPP, encoding="utf-8", newline="\n")
        exe = tmp / ("device_block_pool_harness.exe" if os.name == "nt"
                     else "device_block_pool_harness")
        includes = [ROOT / "src"]
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                # QUOTED include paths: this repository's own path contains an
                # `&`, which cmd would otherwise read as a command separator.
                script = tmp / "build_device_block_pool_harness.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul 2>&1\r\n' % compiler
                    + 'cd /d "%s"\r\n' % tmp
                    + 'cl /nologo /std:c++20 /EHsc /D_CRT_SECURE_NO_WARNINGS "%s" %s '
                      '/Fe:"%s"\r\n'
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
            failures.append("the device-block-pool harness does not compile:\n"
                            + (failure.stdout or "") + (failure.stderr or ""))
            return True

        for arm, caps, expected in ARMS:
            env = dict(os.environ)
            env["RASBERY_ARENA_PERSIST"] = "1"
            # Caps small enough to REACH.  The defaults are 8 GiB / 4096 blocks
            # / depth 64, and a harness that could only exercise the defaults is
            # a harness that never saw the bound hold.
            env.update(caps)
            done = subprocess.run([str(exe), arm], capture_output=True,
                                  universal_newlines=True, env=env)
            if done.returncode != 0:
                failures.append(f"the device-block-pool harness failed ({arm}): "
                                f"{done.stdout}{done.stderr}")
                return True
            results = dict(line.split() for line in done.stdout.split("\n")
                           if line.strip())
            for name, why in dict(GATE, **expected).items():
                if results.get(name) != "1":
                    failures.append(f"compiled device-block-pool contract "
                                    f"[{arm}/{name}]: {why} "
                                    f"(harness said {results.get(name)!r})")
    return True


# ---------------------------------------------------------------------------
# 4. NEGATIVE CONTROLS -- a check that cannot fail is a comment
# ---------------------------------------------------------------------------

def controls() -> list[str]:
    broken: list[str] = []
    if not check_bounded(POOL.replace("RASBERY_ARENA_PERSIST_CLASS_DEPTH", "NOTHING")):
        broken.append("check_bounded passes a pool with no per-class depth cap")
    if not check_bounded(POOL.replace("evictOneLocked(", "nothingLocked(")):
        broken.append("check_bounded passes a pool that cannot evict")
    if not check_bounded(POOL.replace("reclaim == nullptr", "false")):
        broken.append("check_bounded passes a pool that parks past its cap when no "
                      "device free is installed")
    if not check_bounded(POOL.replace(
            "for (void* victim : victims) reclaim(victim);", "// nothing")):
        broken.append("check_bounded passes a pool that never calls the reclaimer "
                      "outside the lock")
    if not check_exact_still(POOL.replace("s.parked.find(bytes)",
                                          "s.parked.lower_bound(bytes)")):
        broken.append("check_exact_still passes a size-ordered search")
    if not check_receipt(SERVER.replace('\\"pool_evictions\\"', '\\"nothing\\"')):
        broken.append("check_receipt passes a receipt with no pool_evictions")
    if not check_receipt(SERVER.replace('\\"pool_bookkeeping_bytes\\"',
                                        '\\"nothing\\"')):
        broken.append("check_receipt passes a receipt that cannot weigh the pool")
    if not check_receipt(SERVER.replace("blockpool::arenaRebuilds(dev)",
                                        "dev.block_reshapes")):
        broken.append("check_receipt passes a receipt that prints the per-case reshape "
                      "counter under the name `arena_rebuilds` -- the exact mislabel "
                      "the 238 block-38 report read and drew a wrong conclusion from")
    return broken


failures += check_bounded(POOL)
failures += check_exact_still(POOL)
failures += check_receipt(SERVER)
run_compiled()

broken_controls = controls()
if broken_controls:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are "
                    "therefore comments:\n    " + "\n    ".join(broken_controls))

if failures:
    print("device block pool: FAIL")
    for problem in failures:
        print("  " + problem)
    raise SystemExit(1)

print(f"device block pool: PASS ({len(CAP_ENVS)} operator caps, "
      f"{len(REQUIRED_FIELDS)} receipt fields, {len(EXPECTED_BOUNDED) + len(EXPECTED_EVICT)} compiled assertions x 2 arms, "
      f"8 negative controls)")
