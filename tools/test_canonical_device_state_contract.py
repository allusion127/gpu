#!/usr/bin/env python3
"""Canonical CMFD-Nodal device state contract (Rev.7.1 Task 7).

CMFD and Nodal each kept their own device copy of the same physics, so every
outer paid a round trip that existed only because neither side knew the other
had already put the bytes on the device.  Task 7 makes one buffer canonical and
has both backends borrow the pointer.

What that breaks, if it is done carelessly, is not arithmetic -- it is
observability and ownership.  This file pins the six ways:

  1. THE GATE.  RASBERY_GPU_SHARED_STATE, default OFF, and feature-off must be
     byte-identical: all-null borrowed pointers, every elision predicate false,
     both backends on the branches they took before.
  2. BORROWED, NOT OWNED.  The adapter must never free or reallocate what it
     was handed, and the pointers must come from the arena -- whose fixed-address
     contract is the only reason it is legal to bake them into a captured graph.
  3. THE GRAPH KEY.  The borrowed pointers are memcpy OPERANDS in the nodal
     capture and the materialize mask decides which memcpy NODES exist at all.
     Both must be in the key, or a graph captured while nobody was looking
     replays with the downloads still missing.
  4. THE ELISION IS OWNERSHIP-BASED, not sharing-based.  Skipping an upload
     because a region is shared -- without asking who wrote it last -- loses
     every host-side perturbation (a rod move, a restart, a boron step).
  5. THE OBSERVATION API IS EXPLICIT.  Every host consumer of the Geometry
     arrays is named, and every shared region is reachable by some consumer.
  6. ONLY THE FOUR REAL GENERATION COUNTERS.  DeviceSlotState carries eight
     more that nothing on the host bumps; an upload gated on one of those would
     be suppressed forever.
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
TEST = ROOT / "test"

CANON = SRC / "GpuCanonicalState.h"
NODAL_CU = SRC / "CudaXsReconBackend.cu"
NODAL_H = SRC / "CudaXsReconBackend.h"
STUB = SRC / "CudaXsReconBackendStub.cpp"
CMAKE = ROOT / "CMakeLists.txt"
GATE = TEST / "canonical_state.cpp"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


CANON_TEXT = read(CANON)
CANON_CODE = strip_comments(CANON_TEXT)
CU_TEXT = read(NODAL_CU)
CU_CODE = strip_comments(CU_TEXT)
H_CODE = strip_comments(read(NODAL_H))
STUB_CODE = strip_comments(read(STUB))
CMAKE_TEXT = read(CMAKE)
GATE_TEXT = read(GATE)


def want(text: str, needle: str, where: str, why: str) -> None:
    if needle not in text:
        problems.append(f"{where}: missing {needle!r} -- {why}")


# --- 1. the gate -------------------------------------------------------------
want(CANON_CODE, "RASBERY_GPU_SHARED_STATE", "GpuCanonicalState.h", "the feature gate")
gate_fn = CANON_CODE[CANON_CODE.find("canonicalSharedStateEnabled"):][:500]
if 'std::string(v) != "0"' not in gate_fn or "v != nullptr" not in gate_fn:
    problems.append("canonicalSharedStateEnabled: must be OFF unless explicitly set and "
                    "treat '0' as off, like every other RASBERY_* gate")

# Feature-off must be inert BY CONSTRUCTION: both predicates return false the
# moment the borrowed pointer is null, before consulting anything else.
for fn in ("canonicalElidesUpload", "canonicalElidesDownload"):
    body = CANON_CODE[CANON_CODE.find(fn + "("):]
    body = body[:body.find("\n}") + 2]
    if "b.get(region) == nullptr) return false" not in body:
        problems.append(f"{fn}: must return false immediately when the region is not "
                        "borrowed -- that is what makes feature-off byte-identical and "
                        "mixed mode possible without a third code path")

# --- 2. borrowed, not owned --------------------------------------------------
for bad in ("cudaMalloc", "cudaFree", "new double", "delete[]"):
    if bad in CANON_CODE:
        problems.append(f"GpuCanonicalState.h: contains {bad!r} -- the canonical buffers "
                        "are BORROWED from the arena; allocating here would defeat the "
                        "fixed-address contract that makes graph capture legal")
want(CANON_CODE, "canonicalFromSlotView", "GpuCanonicalState.h",
     "the borrowed set must be derived from the arena's slot view, not assembled by hand")
# The nodal arena now really borrows; the old hard refusal must be gone.
if "canonicalIsSlotDense" in CANON_CODE:
    problems.append("GpuCanonicalState.h: canonicalIsSlotDense is still here.  The nodal "
                    "per-slot pointer table replaced the stride refusal with real wiring; "
                    "leaving the predicate invites someone to re-add the refusal")
want(CANON_CODE, "canonicalNodalSetIsCoherent", "GpuCanonicalState.h",
     "flux/jnet/phis are adopted together or not at all -- a partial set would pair the "
     "canonical jnet with the arena's own flux, blending two outer iterations")
want(CU_CODE, "canonicalNodalSetIsCoherent", "CudaXsReconBackend.cu",
     "the arena must refuse a partial set at the adoption boundary")
want(CU_CODE, "g_nodal_arena->adoptCanonical", "CudaXsReconBackend.cu",
     "adoptCanonicalBuffers must reach the ARENA, not stop at the per-instance path")
want(CU_CODE, "refreshViews", "CudaXsReconBackend.cu",
     "the view table has to be rebuilt when an adoption changes it")
adopt = CU_CODE[CU_CODE.find("XsReconBackend::adoptCanonicalBuffers"):][:1200]
if adopt and re.search(r"cudaFree|cudaMalloc", adopt):
    problems.append("adoptCanonicalBuffers: must not allocate or free -- it borrows")

# --- 3. the graph key --------------------------------------------------------
#
# THE KEY IS A STRUCT NOW, AND THE MASK IS AN INDEX RATHER THAN A REASON TO
# DESTROY.  This section used to read four `g_key_*` globals and a `const bool
# key_ok` conjunction, and to require that BOTH adoptCanonicalBuffers and
# setMaterializeMask dropped the captured graph.  Task 10 part 4 §6 replaced the
# single remembered key with NodalGraphKey + a small keyed cache, because the
# materialize mask alternates between two values at every device-outer segment
# boundary: enforcing it by destruction meant re-capturing twice per segment --
# 3,282 captures on kngr_238, each one draining the backend's stream inside a
# window nothing counted.
#
# THE SAFETY ARGUMENT DID NOT CHANGE, and that is why the rules below are the
# same rules re-aimed rather than weaker ones.  It was never "the key matched";
# it was "this graph was captured under exactly these conditions".  A LOOKUP
# establishes that; an equality test against one remembered key only
# approximated it.  So what has to be pinned is that every condition is a FIELD
# and that operator== compares it -- a field that exists and is not compared is
# the stale capture this section is about.
KEY_STRUCT = CU_CODE[CU_CODE.find("struct NodalGraphKey"):]
KEY_STRUCT = KEY_STRUCT[:KEY_STRUCT.find("struct NodalGraph ")] if KEY_STRUCT else ""
if not KEY_STRUCT:
    problems.append("CudaXsReconBackend.cu: no NodalGraphKey; the nodal capture has no key "
                    "and a graph captured under one set of conditions can replay under "
                    "another")
# THE DECLARATIONS AND THE COMPARISON ARE READ SEPARATELY, because a struct that
# still MENTIONS a field in its operator== reads as "has the field" to a scan of
# the whole block -- which is how "delete the declaration" passed this rule's own
# negative control once.
KEY_EQ = KEY_STRUCT[KEY_STRUCT.find("bool operator=="):] if KEY_STRUCT else ""
KEY_FIELDS = KEY_STRUCT[:KEY_STRUCT.find("bool operator==")] if KEY_STRUCT else ""
for field, why in (
        ("canon_jnet", "the borrowed jnet is baked into the capture"),
        ("canon_flux", "the borrowed flux is baked into the capture"),
        ("canon_phis", "the borrowed phis is baked into the capture"),
        ("materialize", "the mask decides which download NODES exist in the capture"),
        ("halt", "the halt gate is a kernel ARGUMENT; a graph captured ungated cannot be "
                 "replayed gated"),
        ("halt_slot", "the gated slot is an argument too")):
    if not re.search(r"\b%s\s*=" % re.escape(field), KEY_FIELDS):
        problems.append(f"NodalGraphKey has no {field} field -- {why}")
    elif not re.search(r"\b%s\s*==" % re.escape(field), KEY_EQ):
        problems.append(f"NodalGraphKey::operator== does not compare {field} -- {why}, and "
                        "a field nobody compares is a field that is not in the key")
# And every one of them has to be FILLED at the lookup, or the cache answers with
# whatever the default was.
for field in ("canon_jnet", "canon_flux", "canon_phis", "materialize", "halt", "halt_slot"):
    # WITH A WORD BOUNDARY, because `want.halt` is a prefix of `want.halt_slot`
    # -- a substring test says the halt is filled as long as the SLOT is, which
    # is the pair this rule exists to keep together.
    if not re.search(r"want\.%s\b" % re.escape(field), CU_CODE):
        problems.append(f"CudaXsReconBackend.cu: the drive does not set want.{field} before "
                        "the cache lookup, so the key it searches with is a default")

# adoptCanonicalBuffers STILL DESTROYS, and setMaterializeMask still must not.
# The asymmetry is the point: an adoption changes which BUFFERS every entry in
# the cache points at, so no entry survives it; a mask change selects among
# entries that are all still valid.
adopt_body = CU_CODE[CU_CODE.find("XsReconBackend::adoptCanonicalBuffers"):][:1200]
if "dropNodalGraph" not in adopt_body:
    problems.append("adoptCanonicalBuffers: must drop the captured nodal graphs -- an "
                    "adoption re-points buffers that EVERY cache entry has baked, so none "
                    "of them survives it")
mask_body = CU_CODE[CU_CODE.find("XsReconBackend::setMaterializeMask"):][:1400]
if "dropNodalGraph" in mask_body:
    problems.append("setMaterializeMask: drops the captured nodal graph.  The mask "
                    "alternates twice per device-outer segment, so that is a re-capture "
                    "and a host drain per segment (3,282 on kngr_238); it is a "
                    "NodalGraphKey field instead, and the lookup is what makes a drive "
                    "under this mask select a graph captured under it")

# --- 4. ownership-based elision ----------------------------------------------
want(CANON_CODE, "last_writer != CanonicalOwner::Host", "GpuCanonicalState.h",
     "an upload may be skipped only when a DEVICE side wrote the bytes; skipping "
     "because the region is merely shared loses every host-side perturbation")
# Every elided transfer in the backend must go through the predicate.
for region in ("Jnet", "Flux"):
    if f"canonicalElidesUpload(canon, gpu::CanonicalRegion::{region}" not in CU_CODE:
        problems.append(f"CudaXsReconBackend.cu: the {region} upload does not consult "
                        "canonicalElidesUpload")
for region in ("Jnet", "Phis"):
    if f"canonicalElidesDownload(canon, gpu::CanonicalRegion::{region}" not in CU_CODE:
        problems.append(f"CudaXsReconBackend.cu: the {region} download does not consult "
                        "canonicalElidesDownload")
# Both drive paths must bind the canonical pointers.  The HYBRID path is the one
# that is easy to forget, and forgetting it is not a missed optimisation: it
# uploads the host's stale copy over what CMFD just produced.
binds = len(re.findall(r"canon\.jnet != nullptr \? canon\.jnet", CU_CODE))
if binds < 2:
    problems.append("CudaXsReconBackend.cu: only %d of the 2 drive paths (solveNodal, "
                    "solveNodalPost) bind the canonical jnet -- the other would write "
                    "where the CMFD backend never looks" % binds)
if "canonicalElidesUpload" not in CU_CODE[CU_CODE.find("if (hybrid_even)"):][:3000]:
    problems.append("the HYBRID drive path uploads unconditionally; with v.jnet bound to "
                    "the canonical buffer that overwrites what CMFD just produced")
# Ownership must be published only after the work landed.
if "setOwner(gpu::CanonicalRegion::Jnet, gpu::CanonicalOwner::Nodal)" not in CU_CODE:
    problems.append("CudaXsReconBackend.cu: the drive never claims ownership of jnet, so "
                    "the next upload can never be elided")

# --- 5. the observation API --------------------------------------------------
for name in ("CanonicalConsumer", "canonicalConsumerMask", "canonicalAllConsumerMask"):
    want(CANON_CODE, name, "GpuCanonicalState.h", "host consumers must be named, not implied")
for consumer in ("PinPowerReconstruction", "ResultOutput", "HostCmfdOuter", "Diagnostics"):
    want(CANON_CODE, consumer, "GpuCanonicalState.h",
         "every host reader of the Geometry arrays has to be served deliberately once "
         "the routine downloads are gone")
want(H_CODE, "setMaterializeMask", "CudaXsReconBackend.h", "the observation API")

# --- 6. only the four real generation counters -------------------------------
want(CANON_CODE, "enum class CanonicalGeneration", "GpuCanonicalState.h",
     "the generation set must be explicit")
for real in ("micx_generation", "ref_generation", "hoststate_generation",
             "nodal_constant_generation"):
    want(CANON_CODE, real, "GpuCanonicalState.h", "one of the four host-backed counters")
gen_fn = CANON_CODE[CANON_CODE.find("canonicalGenerationOf"):][:900]
for speculative in ("geometry_generation", "material_generation", "operator_generation",
                    "flux_generation", "current_generation", "dhat_generation",
                    "isotope_generation", "th_generation"):
    if speculative in gen_fn:
        problems.append(f"canonicalGenerationOf reads {speculative}, which is SPECULATIVE "
                        "(GpuSlotControl.h): nothing on the host bumps it, so an upload "
                        "gated on it would be suppressed for the rest of the run")

# --- stub parity -------------------------------------------------------------
for sym in ("adoptCanonicalBuffers", "canonicalBuffers", "setMaterializeMask",
            "materializeMask", "canonicalUploadsElided", "canonicalDownloadsElided"):
    if f"XsReconBackend::{sym}" not in STUB_CODE:
        problems.append(f"CudaXsReconBackendStub.cpp: {sym} is not defined; the no-CUDA "
                        "build would not link and the call sites would need an #ifdef")

# --- the gate is built -------------------------------------------------------
if "rasbery_canonical_state" not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: does not build test/canonical_state.cpp")
if "add_test(NAME canonical_state" not in CMAKE_TEXT:
    problems.append("CMakeLists.txt: canonical_state is not registered with ctest")
for needle, why in (("pointer identity", "the identity check"),
                    ("mixed mode", "one slot shared, one legacy"),
                    ("speculative", "the speculative-counter check"),
                    ("feature-off", "the inert-when-off check"),
                    ("mixed stride", "canonical slot 0 beside legacy slot 1 -- the "
                                     "configuration the dense rebase could not represent"),
                    ("dense-equivalent", "a default table entry must equal the rebase, "
                                         "which is what makes the conversion indirection "
                                         "only")):
    if needle not in GATE_TEXT:
        problems.append(f"test/canonical_state.cpp: no {why}")


def main() -> int:
    if problems:
        for problem in problems:
            print("canonical device state contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("canonical device state contract: PASS (%d regions, %d consumers, 4 real "
          "generations)" % (CANON_CODE.count("case CanonicalRegion::") // 2,
                            CANON_CODE.count("case CanonicalConsumer::") - 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
