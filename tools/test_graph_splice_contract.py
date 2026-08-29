#!/usr/bin/env python3
"""Cached-graph splice contract (Rev.7.1 Task 10 part 4).

A stream in capture mode RECORDS work.  A cudaGraphLaunch is not work it
records: it is refused with cudaErrorStreamCaptureUnsupported, and the refusal
invalidates the whole capture, taking every node recorded before it.  Measured
by tools/probe_while_body_capture.cu on the local 12.6/sm_61 box; the supported
route is cudaGraphAddChildGraphNode with the SOURCE cudaGraph_t, spliced into
the capture's current position.

That fact turns two idiomatic lines into landmines, and this file pins the five
ways they can grow back.

  1. EVERY CACHED LAUNCH GOES THROUGH THE SPLICE.  A bare
     `cudaGraphLaunch(<cached exec>, <stream>)` in either backend is a
     destroyed capture the moment the outer body is captured -- and it will not
     look like one, because the arms that do not capture keep working.

  2. THE SOURCE GRAPH IS KEPT, AND IN A PAIR.  cudaGraphExec_t cannot be turned
     back into cudaGraph_t, so a cache that destroys the graph on the line after
     the instantiate cannot ever be spliced.  Both caches keep it.

  3. THE PAIR IS DESTROYED AS A PAIR.  A key change that dropped the exec and
     kept the graph would splice a body that no longer matches the exec the
     stream arm launches -- the one failure this mechanism must not have -- and
     the reverse leaks.

  4. A RUN THAT NEVER CAPTURES PAYS NOTHING.  The splice sits on the hottest
     launch site in the tree (the sweep graph, once per outer, plus the nodal
     drive).  The process-wide flag must be checked FIRST, before any runtime
     call, and it must be raise-only: a flag that can go back down has to be
     exactly synchronised with the capture window, and getting that wrong is a
     destroyed capture rather than a slow one.

  5. THE THREE-CALL SPLICE IS THREE CALLS, IN ORDER.  GetCaptureInfo ->
     AddChildGraphNode -> UpdateCaptureDependencies.  Drop the third and the
     next thing captured on that stream depends on what the child depended on
     rather than on the child: the sweep and updjnet run CONCURRENTLY, which is
     not a slower answer, it is a different one.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SPLICE = SRC / "GpuGraphSplice.h"
BICG = SRC / "CudaBICGBackend.cu"
NODAL = SRC / "CudaXsReconBackend.cu"
PROBE = ROOT / "tools" / "probe_while_body_capture.cu"

problems: list[str] = []


def read(path: Path) -> str:
    if not path.is_file():
        problems.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8-sig")


SPLICE_TEXT = read(SPLICE)
BICG_TEXT = read(BICG)
NODAL_TEXT = read(NODAL)
PROBE_TEXT = read(PROBE)


def strip_comments(text: str) -> str:
    """Comments AND string literals out, so no rule is satisfied -- or broken --
    by prose.

    Both directions bite.  The house style carries the argument in the comments,
    and every forbidden spelling below appears there on purpose:
    GpuGraphSplice.h quotes verbatim the refusal it exists for.  And the
    backends name their own calls in error strings -- fail("cudaGraphLaunch(nodal
    arena)", rc) -- which a scanner that reads string literals reports as a bare
    launch at a line where there is no launch at all.  Newlines survive both
    substitutions so the reported line numbers stay the file's.
    """

    def blank(match: "re.Match[str]") -> str:
        return "\n" * match.group(0).count("\n")

    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)


BICG_CODE = strip_comments(BICG_TEXT)
NODAL_CODE = strip_comments(NODAL_TEXT)
SPLICE_CODE = strip_comments(SPLICE_TEXT)

# ---------------------------------------------------------------------------
# 1. every cached launch goes through the splice
# ---------------------------------------------------------------------------
#
# The nodal ARENA graph (CudaXsReconBackend's multi-instance batch path) is the
# one deliberate exception and it is named here rather than pattern-matched
# away: the graph arm refuses a batch by name, so that launch can never be
# inside an outer-body capture.  If the arm ever admits a batch, this line is
# what has to change with it.
ALLOWED_BARE_LAUNCH = {
    "cudaGraphLaunch(exec, _stream)",  # nodal arena, batch-only (see above)
}

for path, code in ((BICG, BICG_CODE), (NODAL, NODAL_CODE)):
    for match in re.finditer(r"cudaGraphLaunch\(([^;]*?)\)", code, flags=re.S):
        call = "cudaGraphLaunch(%s)" % " ".join(match.group(1).split())
        if call in ALLOWED_BARE_LAUNCH:
            continue
        line = code[: match.start()].count("\n") + 1
        problems.append(
            f"{path.name}:{line}: bare {call} -- a cached launch must go through "
            "rasbery::graphLaunchOrSplice, or it destroys the capture it lands in")

for path, code, text in ((BICG, BICG_CODE, BICG_TEXT), (NODAL, NODAL_CODE, NODAL_TEXT)):
    if "rasbery::graphLaunchOrSplice(" not in code:
        problems.append(f"{path.name}: does not use rasbery::graphLaunchOrSplice")
    # Against the RAW text: an #include is a string literal, and the stripper
    # above has already blanked it out of `code` by design.
    if '#include "GpuGraphSplice.h"' not in text:
        problems.append(f"{path.name}: does not include GpuGraphSplice.h")

# ---------------------------------------------------------------------------
# 2. the source graph is kept
# ---------------------------------------------------------------------------
if "cudaGraph_t        src" not in BICG_CODE and "cudaGraph_t src" not in BICG_CODE:
    problems.append("CudaBICGBackend.cu: SweepGraph does not retain the source cudaGraph_t")
if "sweep_graph_src" not in BICG_CODE:
    problems.append("CudaBICGBackend.cu: no sweep_graph_src for the current cache entry")
if "nodal_graph_src" not in NODAL_CODE:
    problems.append("CudaXsReconBackend.cu: Impl does not retain the nodal source cudaGraph_t")

# The instantiate must NOT be followed by an unconditional destroy of the graph
# it consumed.  Both sites now destroy only on the path where the instantiate
# did not take it.
for path, code, exec_name in ((BICG, BICG_CODE, "sweep_graph_exec"),
                              (NODAL, NODAL_CODE, "d.nodal_graph")):
    for match in re.finditer(
            r"cudaGraphInstantiate\(&%s,\s*graph[^;]*;\s*(.{0,80})" % re.escape(exec_name),
            code, flags=re.S):
        tail = " ".join(match.group(1).split())
        if tail.startswith("if (graph != nullptr) cudaGraphDestroy(graph);"):
            line = code[: match.start()].count("\n") + 1
            problems.append(
                f"{path.name}:{line}: the source graph is destroyed unconditionally after "
                "instantiate -- it can never be spliced")

# ---------------------------------------------------------------------------
# 3. the pair is destroyed as a pair
# ---------------------------------------------------------------------------
DESTROY_PAIRS = (
    (BICG, BICG_CODE, "destroyGraphCaches",
     ("cudaGraphExecDestroy(e.exec)", "cudaGraphDestroy(e.src)")),
    # Task 10 part 4 turned the nodal side's single slot into a keyed cache, so
    # the pair to destroy is every ENTRY's -- and `nodal_graph` /
    # `nodal_graph_src` became non-owning aliases into it.
    (NODAL, NODAL_CODE, "dropNodalGraph",
     ("cudaGraphExecDestroy(e.exec)", "cudaGraphDestroy(e.src)", "nodal_graphs.clear()")),
)
for path, code, func, needles in DESTROY_PAIRS:
    # The DEFINITION, not the first mention: both of these are called from above
    # their own body, and a window opened at a call site checks nothing.
    match = re.search(r"void\s+%s\s*\(\s*\)\s*\{" % re.escape(func), code)
    body = code[match.end():match.end() + 900] if match else ""
    if not match:
        problems.append(f"{path.name}: no {func} definition")
        continue
    for needle in needles:
        if needle.replace(" ", "") not in body.replace(" ", ""):
            problems.append(
                f"{path.name}: {func} does not {needle} -- the exec and its source graph "
                "must come down together")

# THE SELECTION IS AN ALIAS, SO NOBODY MAY DESTROY IT.
#
# Since the nodal side became a keyed cache, `nodal_graph` and `nodal_graph_src`
# point INTO nodal_graphs.  A destroy through either one frees an entry the
# vector still holds -- a double free at release, and a live cache entry pointing
# at a dead exec until then.  dropNodalGraph is the only owner; the backend's
# release path has to call it rather than reach past it.
for spelling in ("cudaGraphExecDestroy(nodal_graph)", "cudaGraphDestroy(nodal_graph_src)"):
    if spelling.replace(" ", "") in NODAL_CODE.replace(" ", ""):
        problems.append(
            f"CudaXsReconBackend.cu: {spelling} -- that handle is a non-owning alias into "
            "nodal_graphs; destroy through dropNodalGraph()")
if "dropNodalGraph();" not in NODAL_CODE:
    problems.append("CudaXsReconBackend.cu: nothing calls dropNodalGraph()")

# The cache is capped: an unbounded one is a leak with a polite name.
if "kNodalGraphCacheMax" not in NODAL_CODE:
    problems.append("CudaXsReconBackend.cu: the nodal graph cache has no cap")

# materialize must be a KEY field, not a reason to destroy.  This is the whole
# point of part 4's second half: the mask alternates twice per segment, and
# destroying on it cost a hidden host rendezvous per segment.
mask_fn = re.search(r"void\s+XsReconBackend::setMaterializeMask[^{]*\{", NODAL_CODE)
mask_body = NODAL_CODE[mask_fn.end():mask_fn.end() + 400] if mask_fn else ""
if not mask_fn:
    problems.append("CudaXsReconBackend.cu: no setMaterializeMask definition")
elif "dropNodalGraph" in mask_body:
    problems.append(
        "CudaXsReconBackend.cu: setMaterializeMask drops the nodal graph -- the mask "
        "alternates twice per device outer segment, so that is a re-capture (and a host "
        "drain nothing counts) per segment; it belongs in NodalGraphKey instead")
if "materialize" not in NODAL_CODE[NODAL_CODE.find("struct NodalGraphKey"):
                                   NODAL_CODE.find("struct NodalGraphKey") + 1600]:
    problems.append("CudaXsReconBackend.cu: NodalGraphKey does not carry the materialize mask")

# ---------------------------------------------------------------------------
# 4. a run that never captures pays nothing, and the flag is raise-only
# ---------------------------------------------------------------------------
launch_body = SPLICE_CODE[SPLICE_CODE.find("graphLaunchOrSplice"):]
flag_at = launch_body.find("g_graph_capture_possible")
capturing_at = launch_body.find("cudaStreamIsCapturing")
if flag_at < 0 or capturing_at < 0 or flag_at > capturing_at:
    problems.append(
        "GpuGraphSplice.h: graphLaunchOrSplice must test g_graph_capture_possible BEFORE "
        "cudaStreamIsCapturing -- an arm that never captures must not pay a runtime call "
        "per launch")
if re.search(r"g_graph_capture_possible\.store\(\s*false", SPLICE_CODE):
    problems.append(
        "GpuGraphSplice.h: g_graph_capture_possible is lowered somewhere -- it is "
        "raise-only by design; a flag that can go down must be exactly synchronised with "
        "the capture window, and being wrong there is a destroyed capture")

# ---------------------------------------------------------------------------
# 5. the three-call splice, in order
# ---------------------------------------------------------------------------
ORDER = ("cudaStreamGetCaptureInfo",
         "cudaGraphAddChildGraphNode",
         "cudaStreamUpdateCaptureDependencies")
pos = [SPLICE_CODE.find(name) for name in ORDER]
if any(p < 0 for p in pos):
    problems.append("GpuGraphSplice.h: the splice is not the three documented calls: "
                    + ", ".join(ORDER))
elif pos != sorted(pos):
    problems.append("GpuGraphSplice.h: the three splice calls are out of order -- "
                    "GetCaptureInfo, AddChildGraphNode, UpdateCaptureDependencies")

# The probe is the evidence for all of the above and must stay reachable.
for needle, why in (("graph_launch_in_capture", "the refusal this header exists for"),
                    ("child_graph_node", "the route it takes instead"),
                    ("fork_join_in_body", "the nodal stream joining the body capture"),
                    ("RASBERY_PROBE_DEVICE_LAUNCH",
                     "the opt-in that keeps the hanging sub-probe from taking the run")):
    if needle not in PROBE_TEXT:
        problems.append(f"tools/probe_while_body_capture.cu: no {why} ({needle})")


def main() -> int:
    if problems:
        for problem in problems:
            print("graph splice contract: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("graph splice contract: PASS (%d spliced launch sites, %d bare launches allowed)"
          % (BICG_CODE.count("rasbery::graphLaunchOrSplice(")
             + NODAL_CODE.count("rasbery::graphLaunchOrSplice("),
             len(ALLOWED_BARE_LAUNCH)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
