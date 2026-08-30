#!/usr/bin/env python3
"""WP19 -- the capture arbiter's contract, and the negative controls for it.

WHAT THIS DEFENDS.  `--batch-mode M` runs M host Drivers on M threads against
one device.  CUDA's stream-capture rules make that arrangement fragile in a way
no amount of care at one call site can fix:

  * under `cudaStreamCaptureModeGlobal` a capture makes every "potentially
    unsafe" API in every UNRELATED thread fail outright;
  * under ThreadLocal or Relaxed it does not fail them -- it lets them
    INVALIDATE the capture instead, and the capturing lane dies somewhere else
    entirely, with a message about a stream it never touched.

Neither is survivable by accident, so the tree's rule is mechanical:

  1. NO capture anywhere asks for cudaStreamCaptureModeGlobal.  ThreadLocal
     where the capture is a leaf, Relaxed where a conditional-node build has to
     touch a second stream inside the window.  Nothing else.
  2. EVERY BeginCapture/EndCapture pair happens inside a rasbery::CaptureWindow
     -- directly, as a member of the RAII class that owns the pair, or in every
     caller of the helper that contains the pair.
  3. EVERY allocation, page-lock, stream/event creation-or-destruction and
     device-wide drain in src/ happens inside a rasbery::AllocWindow, which is
     the shared side of the same lock.  This is the half that was open: the
     capture sites were guarded and CudaPprBackend.cu's ENTIRE stand-up was not
     (see docs/WP19_CAPTURE_RACE_20260831_KO.md).
  4. A capture-illegal error is RETRIED ONCE with the arbiter held, counted,
     and -- if the retry loses too -- said out loud.
  5. A case that dies says so on its own line, and the dispatcher lifts that
     line into the run's FAIL text instead of printing a deck name.

WHY THIS IS A SECOND FILE.  tools/test_gpu_capture_arbiter_contract.py (Rev.7.1
Task 18d) checks the SHAPE of the four capture sites it knows about, from a
hard-coded list of four sources.  A hard-coded list cannot notice a fifth
backend, and that is exactly what happened: CudaPprBackend.cu and
CudaCramBackend.cu were never in it and were never checked.  So this file
enumerates the sites instead of listing them -- every BeginCapture and every
unsafe API in src/, found by scanning -- and a new backend is covered on the day
it is added.  Both files are expected to pass; neither replaces the other.

Every check is re-run against a MUTATED copy of the source that breaks the
property it defends (the negative controls at the bottom); a check that cannot
fail is not a check.

Run:  python tools/test_capture_arbiter_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TOOLS = os.path.join(ROOT, "tools")

EXTS = (".cu", ".cuh", ".h", ".cpp")

# --- rule 1 ---------------------------------------------------------------
FORBIDDEN_MODE = "cudaStreamCaptureModeGlobal"
ALLOWED_MODES = ("cudaStreamCaptureModeThreadLocal", "cudaStreamCaptureModeRelaxed")

# --- rule 2 ---------------------------------------------------------------
BEGIN_CAPTURE = re.compile(r"(?<![\w:])(cudaStreamBeginCaptureToGraph|cudaStreamBeginCapture)\s*\(")
CAPTURE_WINDOW = re.compile(r"rasbery::CaptureWindow\s")

#: Capture pairs that live inside a helper rather than at their guarded call
#: site.  Each entry is (file, helper name): the helper's own body needs no
#: window, and EVERY call to it elsewhere in src/ must have one.  This is how a
#: template that cannot name rasbery:: types is still covered.
CAPTURE_HELPERS = (
    ("GpuOuterWhile.h", "buildOuterWhile"),
    ("CudaPprBackend.cu", "buildPprWhile"),
)

#: Classes that hold the window as a MEMBER, so the pair is guarded for the
#: object's whole lifetime and no block-scope declaration will be found.
CAPTURE_MEMBER_CLASSES = (
    ("CudaBICGBackend.cu", "ScopedStreamCapture"),
)

# --- rule 3 ---------------------------------------------------------------
#: The APIs CUDA documents as "potentially unsafe" during capture, plus the
#: stream/event lifetime calls that behave the same way.  cudaMemcpy* and
#: cudaStreamSynchronize/cudaDeviceSynchronize are NOT here: they are owned by
#: src/XferLedger.h and enforced by tools/test_xfer_ledger_contract.py, whose
#: wrappers are themselves inside the windows this file checks.
UNSAFE_ALLOC = (
    "cudaMalloc",
    "cudaMallocHost",
    "cudaMallocAsync",
    "cudaHostAlloc",
    "cudaHostRegister",
    "cudaHostUnregister",
    "cudaFree",
    "cudaFreeHost",
    "cudaFreeAsync",
    "cudaStreamCreate",
    "cudaStreamCreateWithFlags",
    "cudaStreamDestroy",
    "cudaEventCreate",
    "cudaEventCreateWithFlags",
    "cudaEventDestroy",
)
ALLOC_CALL = re.compile(
    r"(?<![\w:])(" + "|".join(sorted(UNSAFE_ALLOC, key=len, reverse=True)) + r")\s*\("
)
ALLOC_WINDOW = re.compile(r"rasbery::AllocWindow\s|RASBERY_CUDA_TRY_ALLOC\s*\(")

#: PROVABLY SINGLE-THREADED INIT.  Empty, and that is the correct state.
#:
#: The format exists so a future exemption has to be argued in writing rather
#: than taken silently, and "argued" means naming the thread: an entry is only
#: honest if the call CANNOT run while another lane holds a capture open.
#: "It only runs at teardown" is not such an argument in --batch-mode, where
#: sixteen decks tear down while sixteen others stand up -- src/CudaBICGBackend.cu's
#: arena destructor takes the window rather than claim it.
#:   (relative path, substring that must appear on the line, reason)
ALLOC_ALLOW: list[tuple[str, str, str]] = []

#: Files the rules do not apply to.  The arbiter's own header defines the
#: windows; the stub backends have no CUDA in them at all.
EXEMPT_FILES = ("GpuCaptureArbiter.h",)


# ---------------------------------------------------------------------------
# lexing: comments and string literals are NOT code
# ---------------------------------------------------------------------------

def strip_noncode(text: str) -> list[str]:
    """Return the file's lines with comments and string/char literals blanked.

    A design comment that NAMES cudaStreamCaptureModeGlobal in order to explain
    why it was removed is documentation, not a call.  A scanner that cannot tell
    the two apart forces the explanation out of the source, which is the exact
    opposite of what this file is for.
    """
    out: list[str] = []
    in_block = False
    for line in text.split("\n"):
        buf: list[str] = []
        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if in_block:
                if c == "*" and i + 1 < n and line[i + 1] == "/":
                    in_block = False
                    buf.append("  ")
                    i += 2
                    continue
                buf.append(" ")
                i += 1
                continue
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                buf.append(" " * (n - i))
                break
            if c == "/" and i + 1 < n and line[i + 1] == "*":
                in_block = True
                buf.append("  ")
                i += 2
                continue
            if c in "\"'":
                quote = c
                buf.append(" ")
                i += 1
                while i < n:
                    if line[i] == "\\":
                        buf.append("  ")
                        i += 2
                        continue
                    if line[i] == quote:
                        buf.append(" ")
                        i += 1
                        break
                    buf.append(" ")
                    i += 1
                continue
            buf.append(c)
            i += 1
        out.append("".join(buf))
    return out


def read(path: str) -> str:
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace").replace("\r\n", "\n")


def sources() -> list[str]:
    found = []
    for base, _dirs, files in os.walk(SRC):
        for name in sorted(files):
            if name.endswith(EXTS):
                found.append(os.path.join(base, name))
    return sorted(found)


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


# ---------------------------------------------------------------------------
# brace bookkeeping: "is this line inside a block that still has X open"
# ---------------------------------------------------------------------------

def depths(code: list[str]) -> list[int]:
    """Brace depth at the START of each line."""
    out = []
    d = 0
    for line in code:
        out.append(d)
        d += line.count("{") - line.count("}")
    return out


def guarded(code: list[str], depth: list[int], i: int, guard: re.Pattern) -> bool:
    """Is line `i` inside a block where `guard` was declared and is still live?

    Walks OUTWARD.  `md` is the depth of the innermost block still open at the
    site; a candidate declaration counts only when it sits at exactly that
    depth, which is what excludes a sibling block's already-destroyed guard.
    """
    md = depth[i]
    for j in range(i - 1, -1, -1):
        if depth[j] < md:
            md = depth[j]
        if md <= 0:
            return False
        if depth[j] == md and guard.search(code[j]):
            return True
    return False


def enclosing_openers(code: list[str], depth: list[int], i: int) -> str:
    """Text of the lines that OPEN each block enclosing line `i`.

    A signature can be wrapped over several lines, so each opener contributes a
    small window of lines ending at the brace, which is where the name is.
    """
    chunks: list[str] = []
    md = depth[i]
    for j in range(i - 1, -1, -1):
        if depth[j] < md:
            md = depth[j]
            chunks.append(" ".join(code[max(0, j - 8): j + 1]))
        if md <= 0:
            break
    return " || ".join(chunks)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def check_no_global_mode(files: dict[str, str]) -> list[str]:
    """Rule 1: cudaStreamCaptureModeGlobal appears in no CODE line in src/."""
    bad = []
    for path, text in sorted(files.items()):
        for n, code in enumerate(strip_noncode(text), start=1):
            if FORBIDDEN_MODE in code:
                bad.append(
                    "%s:%d asks for %s -- it makes an UNRELATED lane's stand-up "
                    "fail with 'operation not permitted when stream is capturing', "
                    "which is the WP19 defect and not a diagnostic for it"
                    % (rel(path), n, FORBIDDEN_MODE)
                )
    return bad


def check_capture_mode_named(files: dict[str, str]) -> list[str]:
    """Rule 1b: every BeginCapture names an allowed mode (or captureMode())."""
    bad = []
    for path, text in sorted(files.items()):
        code = strip_noncode(text)
        for n, line in enumerate(code, start=1):
            if not BEGIN_CAPTURE.search(line):
                continue
            window = " ".join(code[n - 1: n + 3])
            if any(mode in window for mode in ALLOWED_MODES):
                continue
            if "captureMode()" in window:
                continue
            bad.append(
                "%s:%d BeginCapture names no capture mode within 4 lines -- "
                "ThreadLocal or Relaxed has to be spelled at the site"
                % (rel(path), n)
            )
    return bad


def _helper_defines(text: str, helper: str) -> bool:
    return re.search(r"\b%s\s*\(" % re.escape(helper), text) is not None


def check_capture_inside_arbiter(files: dict[str, str]) -> list[str]:
    """Rule 2: every BeginCapture is under a live rasbery::CaptureWindow."""
    helper_names = tuple(name for _f, name in CAPTURE_HELPERS)
    class_names = tuple(name for _f, name in CAPTURE_MEMBER_CLASSES)
    bad = []
    for path, text in sorted(files.items()):
        base = os.path.basename(path)
        if base in EXEMPT_FILES:
            continue
        code = strip_noncode(text)
        depth = depths(code)
        for i, line in enumerate(code):
            if not BEGIN_CAPTURE.search(line):
                continue
            if guarded(code, depth, i, CAPTURE_WINDOW):
                continue
            openers = enclosing_openers(code, depth, i)
            if any(
                re.search(r"\b%s\b" % re.escape(h), openers)
                for h, (f, _n) in zip(helper_names, CAPTURE_HELPERS)
                if f == base
            ):
                continue  # covered by the helper rule below
            if any(
                re.search(r"\b(class|struct)\s+%s\b" % re.escape(c), openers)
                for c, (f, _n) in zip(class_names, CAPTURE_MEMBER_CLASSES)
                if f == base
            ):
                continue  # the window is a member of the RAII class
            bad.append(
                "%s:%d opens a capture with no rasbery::CaptureWindow held -- a "
                "sibling lane's stand-up can invalidate it"
                % (rel(path), i + 1)
            )
    return bad


def check_capture_helper_callers(files: dict[str, str]) -> list[str]:
    """Rule 2b: every CALL to a capture helper holds the window itself."""
    bad = []
    for helper_file, helper in CAPTURE_HELPERS:
        definition_seen = False
        called = 0
        for path, text in sorted(files.items()):
            base = os.path.basename(path)
            code = strip_noncode(text)
            depth = depths(code)
            for i, line in enumerate(code):
                if not re.search(r"(?<![\w:])%s\s*\(" % re.escape(helper), line):
                    continue
                openers = enclosing_openers(code, depth, i)
                # The definition itself, and the recursion-free body inside it.
                if base == helper_file and (
                    re.search(r"\b%s\b" % re.escape(helper), openers)
                    or re.search(r"\binline\s+cudaError_t\s+%s\b" % re.escape(helper), line)
                ):
                    definition_seen = True
                    continue
                called += 1
                if not guarded(code, depth, i, CAPTURE_WINDOW):
                    bad.append(
                        "%s:%d calls %s() with no rasbery::CaptureWindow held -- "
                        "the helper opens two captures and holds nothing itself"
                        % (rel(path), i + 1, helper)
                    )
        if not definition_seen:
            bad.append(
                "%s: the capture helper %s() named in CAPTURE_HELPERS is gone -- "
                "the allow-list is now a lie" % (helper_file, helper)
            )
        if called == 0:
            bad.append(
                "%s: %s() is never called under a window; if the helper is dead, "
                "take it out of CAPTURE_HELPERS" % (helper_file, helper)
            )
    return bad


def _allowed_alloc(path: str, line: str) -> bool:
    base = rel(path)
    for allow_path, needle, _reason in ALLOC_ALLOW:
        if base.endswith(allow_path) and needle in line:
            return True
    return False


def check_allocs_inside_arbiter(files: dict[str, str]) -> list[str]:
    """Rule 3: no unguarded allocation / page-lock / stream+event lifetime."""
    bad = []
    for path, text in sorted(files.items()):
        base = os.path.basename(path)
        if base in EXEMPT_FILES:
            continue
        raw = text.split("\n")
        code = strip_noncode(text)
        depth = depths(code)
        for i, line in enumerate(code):
            m = ALLOC_CALL.search(line)
            if not m:
                continue
            # The alloc macro takes the window itself; a call wrapped over
            # several lines still counts as inside it.
            back = " ".join(code[max(0, i - 4): i + 1])
            if "RASBERY_CUDA_TRY_ALLOC" in back and back.count("(") > back.count(")"):
                continue
            if guarded(code, depth, i, ALLOC_WINDOW):
                continue
            if _allowed_alloc(path, raw[i]):
                continue
            bad.append(
                "%s:%d %s() runs with no rasbery::AllocWindow held -- in "
                "--batch-mode this is a sibling lane's capture being invalidated "
                "by this thread: %s"
                % (rel(path), i + 1, m.group(1), raw[i].strip()[:90])
            )
    return bad


def check_retry_path(files: dict[str, str]) -> list[str]:
    """Rule 4: the capture-illegal retry exists, is counted, and is loud."""
    bad = []
    arbiter = files.get(os.path.join(SRC, "GpuCaptureArbiter.h"), "")
    for token in ("capture_race_retry", "capture_race_unrecovered",
                  "inline bool captureIllegal(",
                  "inline void noteCaptureRaceRetry()",
                  "inline void noteCaptureRaceUnrecovered("):
        if token not in arbiter:
            bad.append("src/GpuCaptureArbiter.h has no %s -- the retry has no "
                       "process-wide term" % token)
    if '\\"capture_race_retry\\":' not in arbiter:
        bad.append("src/GpuCaptureArbiter.h: captureArbiterReceipt() does not print "
                   "capture_race_retry, so a run cannot report the race it survived")

    for name in ("CudaPprBackend.cu", "CudaOuterGraph.cu"):
        text = files.get(os.path.join(SRC, name), "")
        code = "\n".join(strip_noncode(text))
        if "rasbery::captureIllegal(" not in code:
            bad.append("src/%s: a graph build with no capture-illegal retry -- one "
                       "lost race is one dead case" % name)
        if "rasbery::noteCaptureRaceRetry()" not in code:
            bad.append("src/%s: a retry nobody counts" % name)
        if "rasbery::noteCaptureRaceUnrecovered(" not in code:
            bad.append("src/%s: a retry that can lose in silence" % name)

    ladder = files.get(os.path.join(SRC, "CudaPprBackend.h"), "")
    if "CaptureRaceRetry," not in ladder or '"capture_race_retry"' not in ladder:
        bad.append("src/CudaPprBackend.h: the refusal ladder has no "
                   "capture_race_retry rung")
    return bad


def check_error_propagation(_files: dict[str, str]) -> list[str]:
    """Rule 5: a dead case names itself, and the dispatcher repeats it."""
    bad = []
    evaluator = read(os.path.join(SRC, "EvaluatorServer.h"))
    if "[RASBERY][EVALUATOR][ERROR]" not in evaluator:
        bad.append("src/EvaluatorServer.h prints no [RASBERY][EVALUATOR][ERROR] line "
                   "-- a case death is a field of a receipt nobody reads")
    if "std::cerr << err.str()" not in evaluator:
        bad.append("src/EvaluatorServer.h does not put the case death on stderr, so a "
                   "controller that owns the protocol stream swallows it")

    harness = read(os.path.join(TOOLS, "run_multi_gpu_batch.py"))
    for token in ("EVALUATOR_CASE_ERROR", "def collect_case_errors(",
                  "failed_case_errors", "case died: "):
        if token not in harness:
            bad.append("tools/run_multi_gpu_batch.py has no %s -- the child's error "
                       "text does not reach the FAIL line" % token)
    if harness.count("collect_case_errors(result,") < 3:
        bad.append("tools/run_multi_gpu_batch.py collects case errors on fewer than "
                   "the three paths that count [FAIL] lines (wave, rolling, epilogue)")

    # LIVE, not textual.  The reader is run against the exact line the fix makes
    # the evaluator print, including the message that started WP19 -- a regex
    # that has drifted from the emitter passes every substring check and finds
    # nothing at 03:00 on the 238.
    sys.path.insert(0, TOOLS)
    try:
        import run_multi_gpu_batch as harness_mod  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover -- import failure IS the finding
        bad.append("tools/run_multi_gpu_batch.py does not import: %s" % exc)
        return bad
    line = (
        '[RASBERY][EVALUATOR][ERROR] {"wave_id":1,"case":3,"deck":"d.inp",'
        '"output":"o.h5","lane":2,"slot":-1,"exit_code":1,'
        '"error":"cudaGetLastError(): operation not permitted when stream is capturing"}'
    )
    probe = harness_mod.WorkerResult(gpu="0", proc=0)
    harness_mod.collect_case_errors(probe, line)
    harness_mod.collect_case_errors(probe, line)  # the same text is scanned twice
    if len(probe.failed_case_errors) != 1:
        bad.append("collect_case_errors recorded %d entries for one death (want 1) -- "
                   "a de-duplication or a regex fault"
                   % len(probe.failed_case_errors))
    elif "operation not permitted when stream is capturing" not in probe.failed_case_errors[0]:
        bad.append("collect_case_errors dropped the message: %r"
                   % probe.failed_case_errors[0])
    elif "d.inp" not in probe.failed_case_errors[0]:
        bad.append("collect_case_errors dropped the deck: %r"
                   % probe.failed_case_errors[0])
    return bad


def check_workdir_is_per_run(_files: dict[str, str]) -> list[str]:
    """Rule 5b: consecutive runs do not clobber each other's logs."""
    bad = []
    harness = read(os.path.join(TOOLS, "run_multi_gpu_batch.py"))
    if 'p.add_argument("--workdir", default="multi_gpu_run"' in harness:
        bad.append("tools/run_multi_gpu_batch.py: --workdir still defaults to a FIXED "
                   "multi_gpu_run/, so rerun N+1 overwrites rerun N's evidence")
    if "run_%Y%m%dT%H%M%SZ" not in harness:
        bad.append("tools/run_multi_gpu_batch.py builds no timestamped workdir")
    if "[RASBERY][MULTI_GPU][WORKDIR]" not in harness:
        bad.append("tools/run_multi_gpu_batch.py does not print the workdir it chose")
    return bad


CHECKS = (
    ("no capture asks for the Global mode", check_no_global_mode),
    ("every BeginCapture names ThreadLocal or Relaxed", check_capture_mode_named),
    ("every capture pair is inside the arbiter", check_capture_inside_arbiter),
    ("every capture helper's callers hold the window", check_capture_helper_callers),
    ("every allocation is inside the arbiter", check_allocs_inside_arbiter),
    ("the capture-illegal retry exists and is counted", check_retry_path),
    ("a dead case propagates its error text", check_error_propagation),
    ("the batch workdir is per-run", check_workdir_is_per_run),
)


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------

def mutate(files: dict[str, str], name: str, old: str, new: str) -> dict[str, str]:
    out = dict(files)
    hits = [p for p in out if os.path.basename(p) == name]
    if not hits:
        raise SystemExit("control target not found: %s" % name)
    path = hits[0]
    if old not in out[path]:
        raise SystemExit("control anchor not found in %s: %r" % (name, old[:70]))
    out[path] = out[path].replace(old, new, 1)
    return out


def controls(files: dict[str, str]):
    return [
        ("a capture is switched back to the Global mode",
         check_no_global_mode,
         mutate(files, "CudaBICGBackend.cu",
                "            return cudaStreamCaptureModeThreadLocal;",
                "            return cudaStreamCaptureModeGlobal;")),
        ("a BeginCapture stops naming its mode",
         check_capture_mode_named,
         mutate(files, "CudaPprBackend.cu",
                "cudaStreamBeginCapture(root_stream, cudaStreamCaptureModeRelaxed)",
                "cudaStreamBeginCapture(root_stream, kMode)")),
        ("a capture site loses its window",
         check_capture_inside_arbiter,
         mutate(files, "CudaXsReconBackend.cu",
                'rasbery::CaptureWindow _capture_window(_stream, "nodal.bucket");',
                "")),
        ("the PPR WHILE build loses its window again",
         check_capture_helper_callers,
         mutate(files, "CudaPprBackend.cu",
                'rasbery::CaptureWindow _capture_window(s.stream, "ppr.while");',
                "")),
        ("a stand-up allocation loses its window",
         check_allocs_inside_arbiter,
         mutate(files, "CudaPprBackend.cu",
                'rasbery::AllocWindow _alloc_window("ppr.shape.standup");',
                "")),
        ("the CRAM stand-up loses its window",
         check_allocs_inside_arbiter,
         mutate(files, "CudaCramBackend.cu",
                'rasbery::AllocWindow _alloc_window("cram.shape.standup");',
                "")),
        ("the retry stops being counted",
         check_retry_path,
         mutate(files, "GpuCaptureArbiter.h",
                "inline void noteCaptureRaceRetry() {",
                "inline void noteCaptureRaceRetryX() {")),
    ]


# ---------------------------------------------------------------------------

def main() -> int:
    files = {p: read(p) for p in sources()}

    failures = 0
    for label, fn in CHECKS:
        bad = fn(files)
        if bad:
            failures += 1
            print("FAIL  %s" % label)
            for line in bad[:40]:
                print("        %s" % line)
            if len(bad) > 40:
                print("        ... and %d more" % (len(bad) - 40))
        else:
            print("ok    %s" % label)

    print()
    for label, fn, mutated in controls(files):
        if fn(mutated):
            print("ok    control caught: %s" % label)
        else:
            failures += 1
            print("FAIL  control NOT caught: %s" % label)

    # A census, so the receipt's shape can be checked against the source
    # without running anything.
    print()
    captures: list[str] = []
    windows = 0
    allocs = 0
    for path, text in sorted(files.items()):
        code = strip_noncode(text)
        for n, line in enumerate(code, start=1):
            if BEGIN_CAPTURE.search(line):
                mode = "?"
                joined = " ".join(code[n - 1: n + 3])
                for candidate in ALLOWED_MODES:
                    if candidate in joined:
                        mode = candidate.replace("cudaStreamCaptureMode", "")
                if "captureMode()" in joined:
                    mode = "captureMode()"
                captures.append("%s:%d  %s" % (rel(path), n, mode))
            if CAPTURE_WINDOW.search(line):
                windows += 1
            if ALLOC_WINDOW.search(line):
                allocs += 1
    print("capture sites: %d" % len(captures))
    for entry in captures:
        print("    %s" % entry)
    print("CaptureWindow declarations: %d" % windows)
    print("AllocWindow declarations:   %d" % allocs)

    print()
    print("FAILED (%d)" % failures if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
