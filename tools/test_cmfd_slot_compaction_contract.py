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

WP3 HARDENING (plan Sec WP3 "테스트 우선 절차" 1, review doc R2).  Two things
were missing, and both are about the word EVERY.

  8. THE SCAN DID NOT COVER EVERY CMFD KERNEL THAT INDEXES BY SLOT.  It read
     one translation unit.  The CMFD OUTER kernels live in
     src/CudaCmfdOuterKernels.h and src/CudaOuterGraph.h and resolve their slot
     through the OTHER logical->physical map in this codebase -- the phase
     queue's `slot = queue.slots[logical]` after `gpuDispatchIsPadding` -- and
     nothing held them to it.  A `halt[logical]` there is exactly the bug this
     file exists to catch, in a file this file did not open: correct at full
     width, wrong at every width below the bucket, and invisible with
     compaction off.  Sec 8 below scans that family and requires, per kernel,
     one of three sanctioned shapes: queue-mapped, caller-named slot, or
     declared full width.
  9. EVERY CHECK IS EXERCISED AGAINST A MUTATED SOURCE.  A scanner that passes
     on the real tree proves nothing until it FAILS on a tree with the bug in
     it; a pattern that stopped matching (a rename, a reformat) would otherwise
     read as PASS forever.  Sec 9 introduces each defect into an in-memory copy
     of the source and requires the scanner to name it.

THE BUCKET LADDER IS NINE RUNGS, AND THE SOURCE IS THE TRUTH (review R2).  The
plan wrote seven (`1,2,4,8,16,32,64`); the tree has
`{1,2,4,8,16,24,32,48,64}` in GpuPhaseScheduler.h, CudaBICGBackend.cu,
CudaXsReconBackend.cu and test/cmfd_slot_compaction_replay.cu, with a
`bucket_histogram` one entry per rung.  24 and 48 are the rungs that cover the
238 M64 arrival width of 14.5, so dropping them would make the padding WP3 is
pricing worse.  This file holds every copy of the ladder to the scheduler's and
holds the histogram widths to its LENGTH rather than to a literal, and
tools/cmfd_compact_receipt.py parses the receipt the same way -- see Sec 6.

Run:  python tools/test_cmfd_slot_compaction_contract.py
      python tools/test_cmfd_slot_compaction_contract.py --run   (+ nvcc replay)
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from cmfd_compact_receipt import (  # noqa: E402
    bucket_ladder,
    label_histogram,
    parse_compact_receipts,
)

# WP20.2 adds the third, refine_round_open, for initialize_solver_state's
# reason and no other: it WRITES `halt[m]`, and `halt` is what every kernel of
# the round it opens consults, so a slot it skipped would keep the previous
# round's mask.  It also reproduces initialize_solver_state's participation
# test (`active && !sweep_halt`) rather than trusting the counters, because a
# slot that test masked never had its counters zeroed.
FULL_WIDTH_KERNELS = ("initialize_solver_state", "finalize_status",
                      "refine_round_open")

# Rev.7.1 Task 10 part 2/3: the THREE kernels of the STREAM-ORDERED sweep enqueue.
#
# THEY TAKE THE PHYSICAL SLOT AS AN ARGUMENT, and that is not a compaction
# escape -- it is the absence of a batch.  CudaBatchArena::enqueueSweeps serves
# ONE participant by construction (it refuses when inUseCount() > 1, because it
# takes no rendezvous and two launchers on one stream is the failure the
# rendezvous exists to prevent), so there is no arrival width to compact and no
# slot map to consult: both are launched <<<1, 1>>> for a slot the CALLER named.
# Reading blockIdx.y here would be reading a dimension that is always 0, and
# resolving through the map would be resolving a map with one entry.
#
# The rule they must obey instead is the thread-0/block-0 filter, which is
# checked below, because a launch geometry that ever widened would otherwise
# have every thread write the same scalars.
# Rev.7.1 Task 10 part 3 adds the third, cmfd_sweep_patch, on exactly the same
# footing: <<<1, 1>>>, a slot the caller named, and a thread-0/block-0 filter.
# WP7 stage B adds the fourth, cmfd_sweep_gate_patch, which IS the gate and the
# patch concatenated in one thread (RASBERY_GPU_CMFD_FUSE bit 3) -- same launch
# geometry, same caller-named slot, same filter, so it inherits the exemption
# for the same reason and not a new one.
EXPLICIT_SLOT_KERNELS = ("cmfd_sweep_gate", "cmfd_sweep_verdict", "cmfd_sweep_patch",
                         "cmfd_sweep_gate_patch")

# WP3 Sec 8: the CMFD OUTER family.  Same question as CudaBICGBackend.cu -- does
# every kernel that indexes by slot go through a logical->physical map -- asked
# of the files where the map is the phase queue's rather than RASBERY_CMFD_SLOT.
OUTER_FAMILY = (
    ("src", "CudaCmfdOuterKernels.h"),
    ("src", "CudaOuterGraph.h"),
    ("src", "GpuOuterWhile.h"),
)

# CmfdOuterKernel.h holds the shared host/device BODIES and must contain no
# __global__ at all.  If it ever grows one, OUTER_FAMILY has to grow with it --
# checked, rather than assumed, because a new kernel in an unscanned file is
# precisely the hole Sec 8 was opened to close.
BODY_ONLY_HEADERS = (("src", "CmfdOuterKernel.h"),)


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


def kernel_signatures(code: str) -> list[tuple[str, str, str]]:
    """(name, signature, body) for every __global__ -- the signature too.

    The outer family is classified by what a kernel is GIVEN (a slot argument,
    a phase queue, neither), so the parameter list is part of the evidence.
    """
    out: list[tuple[str, str, str]] = []
    for match in re.finditer(r"__global__ (?:__launch_bounds__\([^)]*\)\s*)?void (\w+)\(",
                             code):
        name = match.group(1)
        open_brace = code.find("{", match.end())
        signature = code[match.end():open_brace]
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
        out.append((name, signature, code[open_brace + 1:i]))
    return out


# ---------------------------------------------------------------------------
# Sec 1-7: src/CudaBICGBackend.cu
# ---------------------------------------------------------------------------


def scan_backend(raw: str, sched: str, header: str) -> list[str]:
    """Every compaction rule that is about the BiCG/CMFD sweep kernels."""
    problems: list[str] = []
    code = strip_comments(raw)

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

        if name in EXPLICIT_SLOT_KERNELS:
            if re.search(r"blockIdx\.y", body):
                problems.append(f"{name} reads blockIdx.y; it is launched for ONE named "
                                "slot and has no lane dimension to read")
            if "if (threadIdx.x != 0 || blockIdx.x != 0) return;" not in body:
                problems.append(f"{name} has no single-thread filter.  It writes per-slot "
                                "scalars, so a launch geometry wider than <<<1, 1>>> would "
                                "have every thread write them")
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
    # Rev.7.1 Task 18 moved the map into a per-launch STAGING LANE (`map` is
    # stageSlotMap(), one lane per slot plus one for the rendezvous), because
    # the host buffer is a page-locked memcpyAsync source and the stream-ordered
    # enqueue path can have several launches outstanding.  The identity is what
    # is being checked, not which buffer holds it.
    if "lanes = slots;" not in off_arm or "map[i] = i;" not in off_arm:
        problems.append(
            "buildSlotMap: with compaction OFF the map must be the FULL IDENTITY over "
            "physical slots with lanes == slots. Anything narrower changes which blocks "
            "the OFF launch visits and where they are masked.")
    if "stageSlotMap()" not in build:
        problems.append(
            "buildSlotMap writes the shared h_slot_map rather than this launch's staging "
            "lane; a second launcher would rewrite the first one's in-flight H2D source")
    if "static_cast<size_t>(slots) * sizeof(int)" not in build:
        problems.append(
            "buildSlotMap uploads less than the full fleet width; a stale entry from a "
            "wider previous launch would stay reachable")

    # 6. one ladder, shared with the scheduler.  (The rung VALUES are compared
    # here; their COUNT is what the histogram widths are held to, in Sec 6
    # below, so neither the plan's seven nor the tree's nine is written down
    # twice.)
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
    if "lanes == want_lanes" not in strip_comments(header):
        problems.append("SweepGraphCapacity::serves does not key on grid.y")
    for field in ("logical_drives", "physical_slot_blocks", "padding_blocks",
                  "padding_fraction", "bucket_graphs", "bucket_histogram"):
        if field not in raw:
            problems.append(f"the compaction receipt has no {field}")
    if "RASBERY][CMFD][COMPACT" not in raw:
        problems.append("no [RASBERY][CMFD][COMPACT] receipt line")
    return problems


# ---------------------------------------------------------------------------
# Sec 8: the CMFD OUTER family -- the OTHER logical->physical map
# ---------------------------------------------------------------------------
#
# There are exactly two ways a CMFD kernel in this tree is allowed to learn a
# physical slot, and a kernel must be visibly one of them:
#
#   QUEUE-MAPPED.  `logical = blockIdx.*`, then `gpuDispatchIsPadding(logical,
#   queue.count)`, then `slot = queue.slots[logical]`.  After that line the
#   LOGICAL index is dead: every per-slot array -- halt, probes, inputs,
#   decisions, segments, counters, the view table, arena.states -- is keyed by
#   the physical slot.  `halt[logical]` is the canonical bug: right at full
#   width, wrong at every width below the bucket, invisible with compaction off.
#
#   CALLER-NAMED.  The slot arrives as an `int slot` parameter because the
#   launch serves exactly one participant (<<<1,1>>>).  Such a kernel has no
#   lane dimension, so it must not read blockIdx.y at all, and if it WRITES
#   per-slot state it needs the single-thread filter, or a geometry that ever
#   widened would have every thread write the same words.
#
#   DECLARED FULL WIDTH.  It derives an index from blockIdx and bounds it
#   against the fleet width (`>= slot_count`), covering every declared slot on
#   purpose -- the table builder does this, and it must, because a view table
#   with holes is a nullptr dereference the first time a slot is admitted.


def scan_outer_family(sources: dict[str, str]) -> list[str]:
    problems: list[str] = []
    seen = 0
    for path, raw in sources.items():
        code = strip_comments(raw)
        for name, signature, body in kernel_signatures(code):
            seen += 1
            slot_arg = re.search(r"\bint\s+slot\b", signature) is not None
            # Classified by the QUEUE ARGUMENT, not by the presence of the
            # padding test: a kernel that lost the test is still a queue kernel
            # and must be told so by name, rather than falling through to the
            # generic "cannot be classified" arm and hiding which rule broke.
            queued = ("DevicePhaseQueue queue" in signature
                      or "queue.count" in body or "queue.slots[" in body)
            single_thread = re.search(
                r"if\s*\(\s*(threadIdx\.x|blockIdx\.x)\s*!=\s*0u?\s*\|\|\s*"
                r"(threadIdx\.x|blockIdx\.x)\s*!=\s*0u?\s*\)\s*return;", body) is not None

            if queued:
                pad_at = body.find("gpuDispatchIsPadding(logical, queue.count)")
                map_at = body.find("queue.slots[logical]")
                if not re.search(r"const int\s+logical\s*=\s*static_cast<int>\(blockIdx\.",
                                 body):
                    problems.append(
                        f"{path}:{name} reads the phase queue but does not name its lane "
                        "`logical` from blockIdx: the queue's contract (GpuPhaseScheduler.h "
                        "Sec 8.3) is that the index into it is a LANE, not a slot")
                if pad_at < 0:
                    problems.append(
                        f"{path}:{name} indexes the phase queue without "
                        "`gpuDispatchIsPadding(logical, queue.count)`. A padding lane must "
                        "not read the queue at all -- the value there is kQueueEmptySlot")
                if map_at < 0:
                    problems.append(
                        f"{path}:{name} never resolves `slot = queue.slots[logical]`: it is "
                        "using the LOGICAL lane as a physical slot, which is correct only "
                        "when the arrival width equals the bucket")
                elif 0 <= map_at < pad_at:
                    problems.append(
                        f"{path}:{name} reads queue.slots[logical] BEFORE the padding test")
                logical_uses = len(re.findall(r"\[\s*logical\s*\]", body))
                mapped_uses = len(re.findall(r"queue\.slots\[\s*logical\s*\]", body))
                if logical_uses != mapped_uses:
                    offenders = [
                        m.group(0) for m in re.finditer(r"\w+(?:\.\w+)*\[\s*logical\s*\]", body)
                        if not m.group(0).startswith("queue.slots")
                    ]
                    problems.append(
                        f"{path}:{name} indexes {offenders or ['something']} by the LOGICAL "
                        "lane after the map. Every per-slot array here is keyed by the "
                        "PHYSICAL slot; a lane there addresses the wrong tenant at any "
                        "arrival width below the bucket, and is right at full width -- so "
                        "it passes every test that does not narrow the queue.")
                for call in re.finditer(r"\bslotView\(\s*logical\s*\)", body):
                    problems.append(
                        f"{path}:{name} calls arena.slotView(logical): the arena is "
                        "addressed by physical slot")
                continue

            if slot_arg:
                if re.search(r"\bblockIdx\.y\b", body):
                    problems.append(
                        f"{path}:{name} takes the slot as an argument but also reads "
                        "blockIdx.y. One kernel cannot have two ideas about where its slot "
                        "comes from")
                if "queue.slots[" in body:
                    problems.append(
                        f"{path}:{name} takes a named slot AND reads the queue's map")
                writes_slot = (re.search(r"\w+\[\s*slot\s*\]\s*=(?!=)", body)
                               or "arena.states[slot]" in body
                               or "arena.slotView(slot)" in body)
                if writes_slot and not single_thread:
                    problems.append(
                        f"{path}:{name} writes per-slot state for a caller-named slot with "
                        "no single-thread filter: it is launched <<<1,1>>>, and a geometry "
                        "that ever widened would have every thread write the same words")
                continue

            if re.search(r"\bblockIdx\b", body):
                bounded = re.search(r"\bslot\s*>=\s*slot_count\b", body) is not None
                if not (bounded or single_thread):
                    problems.append(
                        f"{path}:{name} derives an index from blockIdx with neither the "
                        "queue's map, a caller-named slot, nor a fleet-width bound "
                        "(`slot >= slot_count`). It is none of the three sanctioned "
                        "shapes, so what it indexes cannot be told from the source")
    if seen < 10:
        problems.append(
            f"only {seen} __global__ kernel(s) found across the CMFD outer family; the "
            "scan is not reaching the files it names")
    for parts in BODY_ONLY_HEADERS:
        if "__global__" in strip_comments(read(*parts)):
            problems.append(
                f"{'/'.join(parts)} has grown a __global__ kernel. It holds the shared "
                "host/device bodies and had none, so OUTER_FAMILY has to grow with it -- "
                "an unscanned file with a slot-indexing kernel is the hole this section "
                "was opened to close.")
    return problems


# ---------------------------------------------------------------------------
# Sec 6 (continued): the ladder's LENGTH, and the parser that reads it
# ---------------------------------------------------------------------------


def scan_ladder(backend: str, nodal: str, replay: str, sched: str) -> list[str]:
    """Every copy of the ladder is the scheduler's, and nothing counts rungs.

    Review R2 asked which of the plan's seven rungs and the tree's nine is
    right.  ANSWER: the tree's, and this function is where that answer is
    enforced -- not by writing nine down, but by deriving every width from the
    scheduler's list so that a future change to the ladder moves every copy and
    every histogram together or fails here.
    """
    problems: list[str] = []
    ladder = bucket_ladder()
    if not ladder:
        return ["cannot read the dispatch ladder out of src/GpuPhaseScheduler.h"]
    rungs = len(ladder)

    want = re.search(r"kDispatchBuckets\[\]\s*=\s*\{([^}]*)\}", sched)
    if want is None:
        return ["src/GpuPhaseScheduler.h has no kDispatchBuckets"]
    reference = [int(t.strip()) for t in want.group(1).split(",") if t.strip()]
    if reference != ladder:
        problems.append(f"cmfd_compact_receipt.bucket_ladder() {ladder} is not "
                        f"kDispatchBuckets {reference}")
    if f"kDispatchBucketCount = {rungs}" not in sched:
        problems.append(
            f"kDispatchBucketCount does not say {rungs}: the ladder and its count are two "
            "statements of one fact and they have drifted")

    for name, text in (("src/CudaBICGBackend.cu", backend),
                       ("src/CudaXsReconBackend.cu", nodal),
                       ("test/cmfd_slot_compaction_replay.cu", replay)):
        found = re.findall(r"kBuckets\[\d*\]\s*=\s*\{([^}]*)\}", strip_comments(text))
        if not found:
            problems.append(f"{name} has no kBuckets ladder to compare")
            continue
        for copy in found:
            rung_values = [int(t.strip()) for t in copy.split(",") if t.strip()]
            if rung_values != ladder:
                problems.append(
                    f"{name}: kBuckets {rung_values} is not the scheduler's ladder "
                    f"{ladder}. One ladder, or the graph cache and the dispatcher "
                    "disagree about what width a captured body was baked for")

    # The histograms are sized from the ladder's LENGTH.  Writing 9 here would
    # be the same mistake as writing 7 in the plan.
    for name, text in (("src/CudaBICGBackend.cu", backend),
                       ("src/CudaXsReconBackend.cu", nodal)):
        for match in re.finditer(
                r"std::array<std::atomic<unsigned long long>,\s*(\d+)>\s*g_\w*bucket_histogram",
                strip_comments(text)):
            if int(match.group(1)) != rungs:
                problems.append(
                    f"{name}: the bucket histogram has {match.group(1)} entries but the "
                    f"ladder has {rungs} rungs. cmfdBucketIndex() can return "
                    f"{rungs - 1}, so the last rung would write past the array")

    # ...and the parser the 238 runbook uses agrees with the source rather than
    # carrying a rung count of its own.
    sample = ('[RASBERY][CMFD][COMPACT] {"enabled":1,"logical_drives":10,'
              '"physical_slot_blocks":300,"padding_blocks":100,"padding_fraction":0.25,'
              '"bucket_graphs":3,"bucket_histogram":[' + ",".join("1" * rungs) + "]}")
    parsed = parse_compact_receipts(sample, ladder)
    if len(parsed) != 1:
        problems.append("cmfd_compact_receipt cannot parse the receipt this tree prints")
    else:
        record = parsed[0]
        if record["ladder_mismatch"]:
            problems.append(
                "cmfd_compact_receipt calls this tree's own histogram a mismatch: "
                + record["ladder_mismatch"])
        if [name for name, _n in record["buckets"]] != [f"<={r}" for r in ladder]:
            problems.append("cmfd_compact_receipt labels the rungs with something other "
                            "than the source ladder")
        if abs(record["padding_fraction_recomputed"] - 0.25) > 1e-9:
            problems.append("cmfd_compact_receipt does not recompute padding_fraction from "
                            "the block counts, so a receipt whose three numbers disagree "
                            "would read as consistent")
    return problems


# ---------------------------------------------------------------------------
# Sec 9: negative controls.  Every rule above, broken on purpose.
# ---------------------------------------------------------------------------


def mutate_in_kernel(raw: str, kernel: str, old: str, new: str) -> str:
    """Replace *old* with *new* inside ONE kernel body of *raw*.

    Whole-file replacement would not do: the point of a negative control is to
    introduce the defect exactly where the real bug would be, in one kernel,
    and see whether the scanner names THAT kernel.
    """
    at = raw.find(f"void {kernel}(")
    if at < 0:
        raise AssertionError(f"negative control: no kernel named {kernel}")
    open_brace = raw.find("{", at)
    depth, i = 0, open_brace
    while i < len(raw):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = raw[open_brace:i]
    if old not in body:
        raise AssertionError(f"negative control: {kernel} does not contain {old!r}")
    return raw[:open_brace] + body.replace(old, new, 1) + raw[i:]


def negative_controls(backend: str, sched: str, header: str, nodal: str, replay: str,
                      outer: dict[str, str]) -> list[str]:
    """A scanner that cannot fail is not a test.  Each entry breaks one rule."""
    problems: list[str] = []

    def backend_must_fail(label: str, mutated: str, expect: str) -> None:
        found = scan_backend(mutated, sched, header)
        if not any(expect in p for p in found):
            problems.append(
                f"negative control `{label}` was not caught: expected a complaint "
                f"mentioning {expect!r}, got {found or '[no problems at all]'}")

    def outer_must_fail(label: str, sources: dict[str, str], expect: str) -> None:
        found = scan_outer_family(sources)
        if not any(expect in p for p in found):
            problems.append(
                f"negative control `{label}` was not caught: expected a complaint "
                f"mentioning {expect!r}, got {found or '[no problems at all]'}")

    # --- Sec 1: a mapped kernel goes back to reading blockIdx.y -------------
    backend_must_fail(
        "cmfd_updls reads blockIdx.y",
        mutate_in_kernel(backend, "cmfd_updls", "RASBERY_CMFD_SLOT(m);",
                         "const int m = static_cast<int>(blockIdx.y);"),
        "cmfd_updls")

    # --- Sec 2: a barrier before the guard ----------------------------------
    backend_must_fail(
        "__syncthreads before the slot guard",
        mutate_in_kernel(backend, "cmfd_updls", "RASBERY_CMFD_SLOT(m);",
                         "__syncthreads(); RASBERY_CMFD_SLOT(m);"),
        "before the slot guard")

    # --- Sec 3: a per-slot mask indexed by the lane -------------------------
    mask_kernel, mask_name = None, None
    for name, body in kernel_bodies(strip_comments(backend)):
        if name in FULL_WIDTH_KERNELS or name in EXPLICIT_SLOT_KERNELS:
            continue
        hit = re.search(r"\b(active|halt|sweep_halt|device_assembly_active)\[m\]", body)
        if hit:
            mask_kernel, mask_name = name, hit.group(1)
            break
    if mask_kernel is None:
        problems.append("negative control: no kernel indexes a per-slot mask by `m`, so "
                        "rule 3 has nothing to break -- has the mask family been renamed?")
    else:
        backend_must_fail(
            f"{mask_name}[lane] in {mask_kernel}",
            mutate_in_kernel(backend, mask_kernel, f"{mask_name}[m]",
                             f"{mask_name}[rasbery_logical]"),
            mask_kernel)

    # --- Sec 5: compaction OFF stops being the identity ---------------------
    backend_must_fail("OFF map is no longer the identity",
                      backend.replace("map[i] = i;", "map[i] = i % 2;", 1),
                      "FULL IDENTITY")
    backend_must_fail("the gate stops reading its env var",
                      backend.replace("RASBERY_GPU_CMFD_COMPACT", "RASBERY_ALWAYS_ON"),
                      "cmfdCompactEnabled")

    # --- Sec 1 again: a full-width kernel gets compacted --------------------
    backend_must_fail(
        "initialize_solver_state compacted",
        mutate_in_kernel(backend, "initialize_solver_state",
                         "const int m = static_cast<int>(blockIdx.y);",
                         "RASBERY_CMFD_SLOT(m);"),
        "must stay full width")

    # --- Sec 6: a second ladder ---------------------------------------------
    ladder_broken = backend.replace("{1, 2, 4, 8, 16, 24, 32, 48, 64}",
                                    "{1, 2, 4, 8, 16, 32, 64}", 1)
    if ladder_broken == backend:
        problems.append("negative control: could not find the kBuckets literal to break")
    else:
        backend_must_fail("the plan's seven-rung ladder in the .cu", ladder_broken,
                          "bucket ladder")
        found = scan_ladder(ladder_broken, nodal, replay, sched)
        if not any("scheduler's ladder" in p or "is not kDispatchBuckets" in p
                   for p in found):
            problems.append(
                "negative control: shortening the .cu ladder to the plan's seven rungs "
                f"was not caught by scan_ladder, got {found or '[no problems]'}")
    hist_broken = backend.replace(
        "std::array<std::atomic<unsigned long long>, 9> g_cmfd_bucket_histogram",
        "std::array<std::atomic<unsigned long long>, 7> g_cmfd_bucket_histogram", 1)
    if hist_broken != backend:
        found = scan_ladder(hist_broken, nodal, replay, sched)
        if not any("histogram" in p for p in found):
            problems.append(
                "negative control: a histogram narrower than the ladder (the exact shape "
                "of an out-of-bounds atomic increment) was not caught")

    # --- the tolerant parser stays tolerant, and still reports -------------
    labelled, mismatch = label_histogram([1, 2, 3, 4, 5, 6, 7], [1, 2, 4, 8, 16, 24, 32, 48, 64])
    if not mismatch or len(labelled) != 7:
        problems.append(
            "negative control: a 7-entry histogram against the 9-rung ladder must be "
            "REPORTED as a mismatch and returned whole -- truncating or padding it to fit "
            "is how a parser and a receipt drift apart for a whole campaign")
    _labelled, mismatch = label_histogram([1, 2, 3], [])
    if not mismatch:
        problems.append("negative control: with no ladder available the parser must say so")

    # --- Sec 8: the outer family -------------------------------------------
    key = "src/CudaCmfdOuterKernels.h"
    graph = "src/CudaOuterGraph.h"

    broken = dict(outer)
    broken[key] = mutate_in_kernel(outer[key], "k_cmfd_upd_dhat",
                                   "const int                  slot = queue.slots[logical];",
                                   "const int                  slot = logical;")
    outer_must_fail("k_cmfd_upd_dhat uses the lane as the slot", broken, "k_cmfd_upd_dhat")

    broken = dict(outer)
    broken[key] = mutate_in_kernel(outer[key], "k_cmfd_upd_psi",
                                   "if (halt != nullptr && halt[slot] != 0u) return;",
                                   "if (halt != nullptr && halt[logical] != 0u) return;")
    outer_must_fail("halt[logical] in k_cmfd_upd_psi", broken, "LOGICAL")

    broken = dict(outer)
    broken[key] = mutate_in_kernel(outer[key], "k_cmfd_upd_dtil",
                                   "if (gpuDispatchIsPadding(logical, queue.count)) return;",
                                   "")
    outer_must_fail("no padding test in k_cmfd_upd_dtil", broken, "gpuDispatchIsPadding")

    broken = dict(outer)
    broken[key] = mutate_in_kernel(outer[key], "k_cmfd_upd_dhat",
                                   "detail::cmfdReduceCounters(n_total, n_guard, n_clamp, "
                                   "ratio_bits, &counters[slot]);",
                                   "detail::cmfdReduceCounters(n_total, n_guard, n_clamp, "
                                   "ratio_bits, &counters[logical]);")
    outer_must_fail("counters[logical] in k_cmfd_upd_dhat", broken, "counters[logical]")

    broken = dict(outer)
    broken[graph] = mutate_in_kernel(outer[graph], "k_cmfd_bind_resident",
                                     "if (threadIdx.x != 0 || blockIdx.x != 0) return;", "")
    outer_must_fail("k_cmfd_bind_resident loses its single-thread filter", broken,
                    "k_cmfd_bind_resident")

    broken = dict(outer)
    broken[graph] = mutate_in_kernel(outer[graph], "k_cmfd_build_slot_table",
                                     "if (slot >= slot_count) return;", "")
    outer_must_fail("k_cmfd_build_slot_table loses its fleet-width bound", broken,
                    "k_cmfd_build_slot_table")

    # A kernel the scan cannot classify at all must be reported, not skipped.
    broken = dict(outer)
    broken[graph] = outer[graph] + (
        "\n__global__ void k_wp3_negative_control(double* out) {\n"
        "    const int m = static_cast<int>(blockIdx.y);\n"
        "    out[m] = 1.0;\n"
        "}\n")
    outer_must_fail("a new kernel that indexes by a raw blockIdx.y", broken,
                    "k_wp3_negative_control")

    return problems


def main() -> int:
    raw = read("src", "CudaBICGBackend.cu")
    header = read("src", "CudaBICGBackend.h")
    sched = strip_comments(read("src", "GpuPhaseScheduler.h"))
    nodal = read("src", "CudaXsReconBackend.cu")
    replay_raw = read("test", "cmfd_slot_compaction_replay.cu")
    outer = {"/".join(parts): read(*parts) for parts in OUTER_FAMILY}

    problems = scan_backend(raw, sched, header)
    problems += scan_outer_family(outer)
    problems += scan_ladder(raw, nodal, replay_raw, sched)

    # 8. the replay's copy of the guard is the shipped guard.
    def guard_text(text: str) -> str:
        start = text.find("#define RASBERY_CMFD_SLOT(m)")
        body = text[start:]
        body = body[: body.find("\n\n")]
        return "\n".join(ln.rstrip().rstrip("\\").rstrip() for ln in body.splitlines())
    if guard_text(raw) != guard_text(replay_raw):
        problems.append(
            "test/cmfd_slot_compaction_replay.cu's copy of RASBERY_CMFD_SLOT has drifted "
            "from the shipped one; the replay would be proving a guard that does not ship")

    problems += negative_controls(raw, sched, header, nodal, replay_raw, outer)

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

    kernels = len(kernel_bodies(strip_comments(raw)))
    outer_kernels = sum(len(kernel_bodies(strip_comments(t))) for t in outer.values())
    print(f"cmfd slot compaction contract: PASS ({kernels} sweep kernels + {outer_kernels} "
          f"outer-family kernels, every slot through a logical->physical map, guard before "
          f"every barrier, OFF is the identity, {len(bucket_ladder())}-rung ladder shared "
          "by every copy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
