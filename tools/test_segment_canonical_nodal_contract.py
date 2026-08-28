#!/usr/bin/env python3
"""Canonical nodal binding inside the device outer segment (Rev.7.1 Task 18-lite).

The segment used to bridge jnet around its nodal hook: the device jnet went
down to Geometry::Jnet, the host called Nodal::drive, and the result came back
up -- 2 x nsurf*ng doubles per outer, and on kngr_238 the largest single item
left in the segment's transfer budget.  The bridge existed for one reason only,
that the nodal drive addressed a DIFFERENT device buffer than updjnet and
upddhat did.  Task 18-lite makes it the same buffer.

Everything that can go wrong with that is an ownership or an observability
question, never an arithmetic one, and this file pins each of them.

  1. THE ADOPTION IS ON THE ARMING PATH.  armOuterSegment must hand the
     segment's own flux/jnet/phis to XsReconBackend::adoptCanonicalBuffers.
     Without a production caller the whole mechanism is dead code -- which is
     exactly what it was before this task.

  2. ALL THREE OR NONE.  canonicalNodalSet() answers with a complete set or an
     empty one.  A partial set pairs the segment's jnet with the nodal arena's
     own flux: two different outer iterations, silently blended.

  3. THE BINDING IS REFUSED PER OUTER WHERE THE DRIVE CAN FALL BACK.
     Nodal::TryDriveGpu drops to the CPU body on any deck with a fractional rod,
     and the CPU body reads Geometry::Jnet.  With the bridge gone that array is
     several outers stale, so such an outer must keep its bridge.  Asked once
     per ARM this is wrong, and i-SMR CY02 is the deck that proves it: a rod
     search moves the bank inside SolveLoop, so a loop that armed with every rod
     integral meets fractional ones later and converges somewhere else.  The
     question is therefore asked every outer, through TryDriveGpu's own
     predicate rather than a copy of it.

  4. NO PER-OUTER TRANSFER WHEN THE BINDING IS LIVE.  The bridge is gated on
     `!canonical_nodal`, and the backend's per-drive uploads and downloads are
     gated on the ownership/materialize pair the segment declares.

  5. THE HOST READERS ARE COVERED.  Geometry::Jnet and Geometry::Phis are
     mirrored at the segment EXIT, once, and the ownership is released there so
     the next drive outside the segment transfers exactly as it did before.
     The reader list is enumerated here and each entry must still exist.

  6. THE BATCH PATH IS UNTOUCHED.  NodalArena::drive and its rendezvous take no
     part in any of this; the segment refuses batch mode outright.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

OUTER_H = SRC / "CudaOuterGraph.h"
OUTER_CU = SRC / "CudaOuterGraph.cu"
OUTER_STUB = SRC / "CudaOuterGraphStub.cpp"
DRIVER = SRC / "Driver.h"
NODAL_H = SRC / "CudaXsReconBackend.h"
NODAL_CU = SRC / "CudaXsReconBackend.cu"
NODAL_STUB = SRC / "CudaXsReconBackendStub.cpp"
CANON = SRC / "GpuCanonicalState.h"
GEOM = SRC / "Geometry.h"
XSSET = SRC / "XSSet.cpp"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.name}")
        return ""
    return path.read_text(encoding="utf-8-sig")


OUTER_H_CODE = read(OUTER_H)
OUTER_CU_CODE = read(OUTER_CU)
OUTER_STUB_CODE = read(OUTER_STUB)
DRIVER_CODE = read(DRIVER)
NODAL_H_CODE = read(NODAL_H)
NODAL_CU_CODE = read(NODAL_CU)
NODAL_STUB_CODE = read(NODAL_STUB)
CANON_CODE = read(CANON)
GEOM_CODE = read(GEOM)
XSSET_CODE = read(XSSET)


def want(code: str, needle: str, where: str, why: str) -> None:
    if needle not in code:
        problems.append(f"{where}: {why} (looked for {needle!r})")


def between(code: str, start: str, end: str) -> str:
    """The slice from `start` up to the next `end`, or '' when start is absent."""
    i = code.find(start)
    if i < 0:
        return ""
    j = code.find(end, i + len(start))
    return code[i:j if j > 0 else len(code)]


# ---------------------------------------------------------------------------
# 1. The adoption has a production caller, and it is the arming path
# ---------------------------------------------------------------------------

ARM = between(DRIVER_CODE, "static bool armOuterSegment(", "\n    /// WHY THERE IS NO LOOP-SCOPE")
if not ARM:
    problems.append("Driver.h: armOuterSegment not found -- the arming path is where the "
                    "adoption belongs, because it is the one place that has both the "
                    "segment and the XSSet")

want(ARM, "adoptCanonicalBuffers", "Driver.h armOuterSegment",
     "the adoption must happen on the arming path; without a production caller the "
     "canonical nodal set is dead code, which is what it was before Task 18-lite")
want(ARM, "canonicalNodalSet()", "Driver.h armOuterSegment",
     "the three pointers must come from the SEGMENT, not be rebuilt at the call site -- "
     "a second source of truth is a second chance to bind the wrong flux")
want(ARM, "setCanonicalNodalBound", "Driver.h armOuterSegment",
     "the runner has to be told whether a backend actually took the set; it decides "
     "whether to bridge, and inferring it from the pointers would answer `yes` for every "
     "run that merely has an arena")

# The set must be ARMED out of segment: arming is not running, and a host outer
# may still run between the arm and the first device outer.
want(ARM, "setCanonicalNodalSegmentMode(false", "Driver.h armOuterSegment",
     "the mode must start OUT of segment -- a host outer between the arm and the first "
     "device outer must transfer exactly as it always did")

# ---------------------------------------------------------------------------
# 2. All three or none
# ---------------------------------------------------------------------------

SET_FN = between(OUTER_CU_CODE, "CanonicalSlotBuffers CudaOuterSegment::canonicalNodalSet()",
                 "\n}\n")
if not SET_FN:
    problems.append("CudaOuterGraph.cu: canonicalNodalSet() not found")
for member in ("device_flux", "device_jnet", "device_phis"):
    want(SET_FN, member + " == nullptr", "CudaOuterGraph.cu canonicalNodalSet",
         f"{member} must be checked; a partial set pairs one outer's jnet with another's "
         "flux and blends them silently")
want(SET_FN, "residency_bound", "CudaOuterGraph.cu canonicalNodalSet",
     "the flux half of the set is the SWEEP's phi, so the set cannot exist before the "
     "residency is bound")
want(CANON_CODE, "canonicalNodalSetIsCoherent", "GpuCanonicalState.h",
     "the all-three-or-none rule must stay where both backends can ask it")

# ---------------------------------------------------------------------------
# 3. Refused where the drive can fall back to the CPU body
# ---------------------------------------------------------------------------

want(DRIVER_CODE, "static int outerCanonicalNodalEligibleHook(void* raw)", "Driver.h",
     "the per-outer eligibility hook must exist: an outer whose drive falls back to the "
     "CPU body reads Geometry::Jnet, and with the bridge gone that array is stale")
want(DRIVER_CODE, "h.ctx->nodal_solver.DeviceDriveEligible()", "Driver.h",
     "and it must ask TryDriveGpu's OWN predicate -- a second spelling of `does this deck "
     "have a fractional rod` is a second chance to answer it differently")
want(DRIVER_CODE, "hooks.canonical_nodal_eligible = &outerCanonicalNodalEligibleHook;",
     "Driver.h", "the hook must actually be installed")
want(read(SRC / "Nodal.cpp"), "if (!DeviceDriveEligible())", "Nodal.cpp",
     "TryDriveGpu must CALL the predicate rather than keep its own copy, or the segment "
     "and the drive can disagree about one outer")
want(OUTER_CU_CODE, "m.hooks.canonical_nodal_eligible(m.hooks.ctx) != 0",
     "CudaOuterGraph.cu",
     "the runner must ask per outer, BEFORE it decides the bridge -- the answer is what "
     "the bridge decision is")
want(ARM, "rasberyGpuNodalFullEnabled()", "Driver.h armOuterSegment",
     "the binding is only safe when the drive inside the segment is the all-device one; "
     "the hybrid arm runs calculateEven on the host over arrays it downloads")
want(ARM, "rasberyGpuNodalEnabled()", "Driver.h armOuterSegment",
     "and only when the device nodal is armed at all -- a host drive reads and writes "
     "the Geometry arrays the binding stops maintaining")
# The receipt.  A binding that did not engage must be distinguishable from one
# that was never asked for.
want(ARM, "canonical_nodal=%d", "Driver.h armOuterSegment",
     "the decision must be receipted; `off` and `on and refused` are different bugs")

# ---------------------------------------------------------------------------
# 4. No per-outer transfer while the binding is live
# ---------------------------------------------------------------------------

want(OUTER_CU_CODE, "!canonical_now && bound_.host_jnet != nullptr",
     "CudaOuterGraph.cu",
     "the jnet bridge must be gated on the binding -- that is the transfer this task "
     "removes, and leaving it in would make the adoption a pure cost")
want(OUTER_CU_CODE, "canonical_nodal_mode(m.hooks.ctx, 1)", "CudaOuterGraph.cu",
     "the device claim must be made before every in-segment drive: the FLUX answer "
     "changes per outer, because a drive that fell back to the host loop left "
     "Geometry::Phif ahead of the device phi")
want(OUTER_CU_CODE, "canonical_nodal_outers", "CudaOuterGraph.cu",
     "the outers that ran under the binding must be counted, or jnet_bridge_bytes==0 "
     "cannot be told apart from a segment that never ran")

MODE_FN = between(NODAL_CU_CODE, "void XsReconBackend::setCanonicalNodalSegmentMode(",
                  "\r\n}\r\n")
if not MODE_FN:
    MODE_FN = between(NODAL_CU_CODE, "void XsReconBackend::setCanonicalNodalSegmentMode(",
                      "\n}\n")
if not MODE_FN:
    problems.append("CudaXsReconBackend.cu: setCanonicalNodalSegmentMode not found")
want(MODE_FN, "setMaterializeMask(0u)", "CudaXsReconBackend.cu segment mode",
     "in segment nothing may be materialised per drive; the segment mirrors both arrays "
     "itself at its exit")
want(MODE_FN, "CanonicalOwner::Nodal", "CudaXsReconBackend.cu segment mode",
     "jnet/phis must be declared device-owned so the per-drive uploads are elided -- and "
     "with the value the drive itself leaves behind, or the captured graph's key flips "
     "on every outer")
want(MODE_FN, "device_owns_flux ? gpu::CanonicalOwner::Cmfd", "CudaXsReconBackend.cu "
     "segment mode",
     "flux is the one region that can legitimately still be the host's, so its ownership "
     "must follow the caller's answer rather than a constant")
want(MODE_FN, "d.canonical.buffers.jnet == nullptr", "CudaXsReconBackend.cu segment mode",
     "a legacy instance borrows nothing: the call must be a no-op there so the segment "
     "can make it unconditionally")
# Out of segment: exactly the pre-Task-7 transfers.
want(MODE_FN, "CanonicalOwner::Host", "CudaXsReconBackend.cu segment mode",
     "leaving the segment must put every region back on the host, or a host outer body's "
     "drive elides an upload against an array CMFD::updjnet has just rewritten")
want(MODE_FN, "canonicalBit(gpu::CanonicalRegion::Jnet)", "CudaXsReconBackend.cu "
     "segment mode",
     "and must restore the downloads, because CMFD::upddhat reads Geometry::Jnet on the "
     "very next line of the host outer")

# The elision predicates the backend actually uses must still be the shared ones.
for region in ("Jnet", "Flux"):
    want(NODAL_CU_CODE, f"canonicalElidesUpload(canon, gpu::CanonicalRegion::{region}",
         "CudaXsReconBackend.cu",
         "the per-drive upload must stay behind the shared predicate rather than a local "
         "flag, so the two backends cannot disagree about one buffer")
for region in ("Jnet", "Phis"):
    want(NODAL_CU_CODE, f"canonicalElidesDownload(canon, gpu::CanonicalRegion::{region}",
         "CudaXsReconBackend.cu",
         "same for the download half, which is the half that can leave a host array "
         "silently stale")

# ---------------------------------------------------------------------------
# 5. Every host reader of Jnet / Phis is covered by the exit mirror
# ---------------------------------------------------------------------------

want(OUTER_CU_CODE, "mirror jnet to the host at the segment exit", "CudaOuterGraph.cu",
     "the exit owes the bytes: PPR and XSSet::NormalizeFluxSign read Geometry::Jnet "
     "outside every segment")
want(OUTER_CU_CODE, "mirror phis to the host at the segment exit", "CudaOuterGraph.cu",
     "and Geometry::Phis, which only the nodal drive writes and only the statepoint "
     "reads")
want(OUTER_CU_CODE, "releaseCanonicalNodal(true)", "CudaOuterGraph.cu",
     "the ownership release must ride the segment's final synchronise -- the mirror above "
     "it is an ENQUEUE, and a release published earlier lets a consumer read the arrays "
     "before the copies land")
want(OUTER_CU_CODE, "releaseCanonicalNodal(false)", "CudaOuterGraph.cu",
     "the failure paths must release too, and with a BLOCKING mirror: they abandon a "
     "stream whose state the segment can no longer reason about")
want(OUTER_CU_CODE, "if (m.canonical_nodal_live) {", "CudaOuterGraph.cu",
     "the exit mirror must be gated on the LIVE latch and not on the binding: if the last "
     "outer fell back to the CPU body the host arrays are the newer copy, and mirroring "
     "the device over them replaces new values with old")

# THE READER LIST.  Each of these is a host consumer of Jnet and/or Phis; if one
# is renamed or moved the mirror above has to be re-argued, not silently kept.
READERS = (
    (DRIVER_CODE,
     "pin_power_reconstruction.reset(1.0 / eigv, geometry.Jnet(), geometry.Phif(), "
     "geometry.Phis());",
     "Driver.h: PPR at every statepoint -- reads Jnet AND Phis, and runs after SolveLoop "
     "with no intervening write"),
    (XSSET_CODE, "_g.Jnet()[i] = -_g.Jnet()[i];",
     "XSSet::NormalizeFluxSign -- reads and writes Jnet in place at statepoint level"),
    (XSSET_CODE, "_g.Phis()[i] = -_g.Phis()[i];",
     "XSSet::NormalizeFluxSign -- the same for Phis"),
    (DRIVER_CODE, "outertrace::hashDoubles(ctx.geometry.Jnet(), nsg)",
     "Driver.h: the RASBERY_OUTER_TRACE hash of Jnet on the host outer body"),
    (DRIVER_CODE, "ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());",
     "Driver.h: the host outer body WRITES the whole of Jnet before reading it, so that "
     "reader is covered by the write -- but only because the exit released the ownership "
     "and the drive uploads again"),
    (DRIVER_CODE, "ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());",
     "Driver.h: the host upddhat reads Jnet on the line after the host nodal drive"),
)
for code, needle, why in READERS:
    if needle not in code:
        problems.append("the Jnet/Phis reader list is out of date: " + why +
                        f" (looked for {needle!r})")

want(GEOM_CODE, "inline double* Phis()", "Geometry.h",
     "Phis must still be the array the segment mirrors into")
want(CANON_CODE, "PinPowerReconstruction", "GpuCanonicalState.h",
     "the consumer table is the safety argument for eliding the downloads; PPR is the "
     "consumer that needs all three regions")

# ---------------------------------------------------------------------------
# 6. Feature-off and the batch path
# ---------------------------------------------------------------------------

want(OUTER_H_CODE, "bool                   canonical_nodal = false;", "CudaOuterGraph.h",
     "the binding must default to OFF in the binding struct, so a caller that never "
     "adopts keeps the bridge with no second code path")
want(OUTER_STUB_CODE, "CudaOuterSegment::canonicalNodalSet", "CudaOuterGraphStub.cpp",
     "the no-CUDA stub must define the new entry points or the stub build stops linking")
want(OUTER_STUB_CODE, "setCanonicalNodalBound(bool) { _impl->binding.canonical_nodal = false; }",
     "CudaOuterGraphStub.cpp",
     "and must never let the binding go live without a device")
want(NODAL_STUB_CODE, "setCanonicalNodalSegmentMode", "CudaXsReconBackendStub.cpp",
     "same for the backend stub")
want(NODAL_H_CODE, "void setCanonicalNodalSegmentMode(bool in_segment, bool device_owns_flux);",
     "CudaXsReconBackend.h", "the declaration must carry both halves in one call, so the "
     "ownership and the materialize mask cannot be set to disagreeing values")

# The batch rendezvous must be byte-identical: nothing in NodalArena may consult
# the segment mode.
ARENA = between(NODAL_CU_CODE, "class NodalArena {", "std::mutex  g_nodal_arena_mutex;")
if "setCanonicalNodalSegmentMode" in ARENA or "canonical_nodal" in ARENA:
    problems.append("CudaXsReconBackend.cu: the batch arena must not consult the segment "
                    "mode -- the segment refuses batch mode outright and the rendezvous "
                    "path has to stay byte-identical")

# And the segment itself must still refuse a batch.
want(OUTER_H_CODE, "BatchMode", "CudaOuterGraph.h",
     "the segment's refusal of batch width must still exist by name")


def main() -> int:
    if problems:
        for problem in problems:
            print("segment canonical nodal contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("segment canonical nodal contract: PASS (%d host readers pinned, 3 regions, "
          "exit mirror + ownership release)" % len(READERS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
