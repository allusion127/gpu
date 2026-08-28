#!/usr/bin/env python3
"""Nodal phase graph + active-slot compaction contract (Rev.7.1 Task 8).

Before compaction, the batched nodal launch used grid.y == the FLEET width and
every non-participating slot still launched a block that read a participation
mask and returned.  At 64 slots with 3 in the batch that is 61/64 of every grid
doing nothing, six times per drive.  Task 8 makes grid.y the smallest BUCKET
covering the participants and routes each logical lane to its physical slot
through a map.

Six properties, none of which shows up in a converged answer:

  1. THE SLOT COMES FROM THE MAP.  No nodal kernel may use blockIdx.y as a slot
     index.  It is correct only when the map happens to be the identity -- which
     is exactly the compaction-OFF case -- so the bug would pass every test that
     did not turn compaction on.
  2. THE PADDING GUARD IS THE FIRST STATEMENT (the W1 rule), and there are NO
     __syncthreads in these kernels.  An early return from a subset of blocks is
     safe only because of that; adding a barrier later would turn the guard into
     a deadlock, so the absence is asserted rather than assumed.
  3. THE GRAPH IS KEYED (bucket, MatEven fusion, geometry).  A graph bakes its
     launch dimensions, so grid.y IS topology -- one graph for all widths would
     replay the wrong width.
  4. THE PER-SLOT XS MIRRORS AND GENERATIONS STAY PHYSICAL-SLOT-INDEXED.  Only
     the map may be logical; a mirror indexed by lane would follow the batch
     composition and describe the wrong deck's bytes.
  5. THE RECEIPT REPORTS THE RATIO.  logical_drives, physical_slot_blocks,
     padding_blocks and the bucket histogram -- the drive count alone is
     identical with and without compaction and would hide the whole effect.
  6. DEFAULT OFF (RASBERY_GPU_NODAL_COMPACT), identity map when off, so the
     feature-off launch shape is exactly the pre-Task-8 one.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

BACKEND = SRC / "CudaXsReconBackend.cu"
SCHED = SRC / "GpuPhaseScheduler.h"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


CU_TEXT = read(BACKEND)
CU_CODE = strip_comments(CU_TEXT)
SCHED_CODE = strip_comments(read(SCHED))

NODAL_KERNELS = ("kNodalTrl0", "kNodalTrl12", "kNodalMat", "kNodalEven", "kNodalMatEven",
                 "kNodalJnet")


def kernel_body(name: str) -> str:
    at = CU_CODE.find("void " + name + "(")
    if at < 0:
        return ""
    rest = CU_CODE[at:]
    nxt = rest.find("__global__", 1)
    return rest[:nxt] if nxt > 0 else rest[:1500]


# --- 1 + 2. the guard, the map, and no barriers ------------------------------
guard = CU_CODE[CU_CODE.find("#define RASBERY_NODAL_SLOT_GUARD"):]
guard = guard[:guard.find("\n\ntemplate")] if "\n\ntemplate" in guard else guard[:900]
if not guard:
    problems.append("CudaXsReconBackend.cu: RASBERY_NODAL_SLOT_GUARD not found")
else:
    if "views" not in guard:
        problems.append("RASBERY_NODAL_SLOT_GUARD: does not consult the per-slot view "
                        "table.  Computing a slot address by a dense stride cannot reach "
                        "canonical buffers, whose slot stride is the whole slot block")
    if "slot_map" not in guard:
        problems.append("RASBERY_NODAL_SLOT_GUARD: does not consult a slot map -- with "
                        "compaction the lane index is NOT the slot index")
    # The padding test must come before the map read, and both before any use.
    lane_at = guard.find("_logical >=")
    map_at = guard.find("(slot_map)[")
    if lane_at < 0:
        problems.append("RASBERY_NODAL_SLOT_GUARD: no padding guard (`_logical >= lanes`); "
                        "a padding lane would index the map out of range")
    elif 0 <= map_at < lane_at:
        problems.append("RASBERY_NODAL_SLOT_GUARD: reads the map BEFORE the padding test")

for name in NODAL_KERNELS:
    body = kernel_body(name)
    if not body:
        problems.append(f"CudaXsReconBackend.cu: {name} not found")
        continue
    if "RASBERY_NODAL_SLOT_GUARD" not in body:
        problems.append(f"{name}: does not use the slot guard")
        continue
    # The guard must be the first statement of the kernel body.
    opening = body.find("{")
    first = body[opening + 1:body.find("RASBERY_NODAL_SLOT_GUARD", opening)]
    if first.strip():
        problems.append(f"{name}: statements precede the slot guard ({first.strip()[:60]!r}); "
                        "the W1 rule is that the padding guard is FIRST, before anything is "
                        "read or written")
    if "__syncthreads" in body:
        problems.append(f"{name}: contains __syncthreads.  These kernels must have none: a "
                        "padding lane returns early, so a barrier would be reached by only "
                        "part of the block set and deadlock")
    if re.search(r"\bblockIdx\.y\b", body.replace("RASBERY_NODAL_SLOT_GUARD", "")):
        problems.append(f"{name}: uses blockIdx.y directly.  With compaction the lane is not "
                        "the slot; this is correct only when the map is the identity, which "
                        "is exactly the compaction-OFF case")

# The whole nodal kernel region must be barrier-free, not just the six bodies.
region = CU_CODE[CU_CODE.find("#define RASBERY_NODAL_SLOT_GUARD"):CU_CODE.find("kNodalJnet") + 900]
if "__syncthreads" in region:
    problems.append("the nodal kernel region contains __syncthreads; see above")

# --- 3. the bucket graph key -------------------------------------------------
if "struct BucketGraph" not in CU_CODE:
    problems.append("CudaXsReconBackend.cu: no per-bucket graph cache; one graph cannot "
                    "serve two dispatch widths, because a graph bakes grid.y")
else:
    key = CU_CODE[CU_CODE.find("for (const auto& e : _graphs)"):][:400]
    for field, why in (("lanes", "grid.y is baked into the graph"),
                       ("fuse", "MatEven fusion is a different kernel set"),
                       ("nxyz", "grid.x comes from the geometry"),
                       ("nsurf", "the jnet phase's grid.x comes from nsurf")):
        if f"e.{field}" not in key:
            problems.append(f"the bucket graph lookup does not compare {field} -- {why}")

want_ladder = re.search(r"kDispatchBuckets\[\]\s*=\s*\{([^}]*)\}", SCHED_CODE)
have_ladder = re.search(r"kBuckets\[\]\s*=\s*\{([^}]*)\}", CU_CODE)
if want_ladder and have_ladder:
    a = [x.strip() for x in want_ladder.group(1).split(",") if x.strip()]
    b = [x.strip() for x in have_ladder.group(1).split(",") if x.strip()]
    if a != b:
        problems.append("the nodal bucket ladder %s differs from the scheduler's %s; the "
                        "nodal phase and the case-phase scheduler must agree about what a "
                        "bucket is, or the two compactions fight" % (b, a))

# --- 4. mirrors and generations stay physical --------------------------------
for mirror in ("xsrf_mirror", "xsnf_mirror", "xssm_mirror"):
    for m in re.finditer(re.escape(mirror), CU_CODE):
        line_start = CU_CODE.rfind("\n", 0, m.start()) + 1
        line = CU_CODE[line_start:CU_CODE.find("\n", m.start())]
        if "logical" in line or "lane" in line:
            problems.append(f"{mirror} is indexed by a LOGICAL lane ({line.strip()[:70]!r}); "
                            "the mirrors and generations follow the physical slot, or they "
                            "describe the wrong deck's bytes when the batch changes")
if "_slot[static_cast<std::size_t>(m)]" not in CU_CODE:
    problems.append("CudaXsReconBackend.cu: slots are no longer addressed by physical index")

# --- 5. the receipt ----------------------------------------------------------
for field in ("logical_drives", "physical_slot_blocks", "padding_blocks",
              "bucket_histogram"):
    if field not in CU_TEXT:
        problems.append(f"the compaction receipt has no {field!r}; the drive count alone is "
                        "identical with and without compaction and would hide the effect")
if "RASBERY][NODAL][COMPACT" not in CU_TEXT:
    problems.append("no [RASBERY][NODAL][COMPACT] receipt line")

# --- 6. default off, identity when off ---------------------------------------
want_gate = CU_CODE[CU_CODE.find("nodalCompactEnabled"):][:500]
if "RASBERY_GPU_NODAL_COMPACT" not in want_gate:
    problems.append("nodalCompactEnabled: does not read RASBERY_GPU_NODAL_COMPACT")
if 'std::string(v) != "0"' not in want_gate or "v != nullptr" not in want_gate:
    problems.append("nodalCompactEnabled: must be OFF unless explicitly set and treat '0' "
                    "as off, like every other RASBERY_* gate")
build = CU_CODE[CU_CODE.find("const int lanes = _compact"):][:900]
if not build:
    problems.append("launchBatch: the dispatch width does not depend on _compact")
else:
    if ": _slots" not in build:
        problems.append("launchBatch: with compaction OFF the dispatch width must be the "
                        "fleet width, or the feature-off launch shape is not the old one")
    if "_h_slot_map[m] = m" not in build:
        problems.append("launchBatch: with compaction OFF the map must be the IDENTITY over "
                        "physical slots -- anything else changes which slot a lane drives")


def main() -> int:
    if problems:
        for problem in problems:
            print("nodal phase graph contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("nodal phase graph contract: PASS (%d kernels, map-indexed, barrier-free)"
          % len(NODAL_KERNELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
