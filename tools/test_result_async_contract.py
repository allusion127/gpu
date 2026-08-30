#!/usr/bin/env python3
"""The result-I/O writer must not be able to see the solver's state.

WHY THIS EXISTS.  WP12 moves the pin-power CSV serialisation -- ~119 MB of
`ostream <<` per `--result full` case, and essentially all of the v4 candidate's
io_wall -- off the solver thread and onto the HDF5 writer thread, behind
RASBERY_RESULT_ASYNC=1.  That is a safe move for exactly one reason: the closure
the writer runs owns every number it touches.  The moment someone captures a
`Geometry&`, an `XSSet&`, a `Schedule&` or an `IO` member into it, the emitter
reads state the solver is concurrently rewriting -- and the failure mode is not
a crash, it is a CSV whose numbers came from a later statepoint, which no h5diff
and no digest would catch (neither covers the CSV).

The same class of mistake has three other spellings, and all four are checked
here:

  1. THE SNAPSHOT LEAKS.  PinPowerCsvRecord holds a reference or a pointer, or
     names a solver type; or Emit() reaches through `g.`, `d.`, `xs.` or
     `_pin_power_csv_*` instead of its own fields.
  2. THE QUEUE IS UNBOUNDED.  A side task's payload does not count against the
     writer queue's byte bound, so 35 statepoints x 15 MB accumulate instead of
     back-pressuring the solver.
  3. THE JOIN IS MISSING.  CloseResult() resets the session (and the file
     handle) without fencing the queue first, so a task can outlive its file.
  4. FEATURE-OFF DRIFTED.  The default is no longer sync, or the sync branch no
     longer calls the same emitter, so `RASBERY_RESULT_ASYNC` unset stops being
     the pre-WP12 path and the B0 claim is unverifiable.

NEGATIVE CONTROLS.  Every scan is run against a synthetic source that violates
it.  A scan that cannot fail its own control is a scan that proves nothing, and
this file reports that as a FAILURE rather than a pass.

USAGE
    tools/test_result_async_contract.py
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IO_CPP = ROOT / "src" / "IO.cpp"
IO_WRITER = ROOT / "src" / "IoWriter.h"

# Types the emitter must never name: all four are mutated by the solver thread
# the instant WriteStepToResult returns.
SOLVER_TYPES = ("Geometry", "XSSet", "Schedule", "HighFive::")
# Member-access spellings that can only come from solver-owned objects.
SOLVER_REACHES = (
    re.compile(r"\bg\s*\."),
    re.compile(r"\bd\s*\."),
    re.compile(r"\bxs\s*\."),
    re.compile(r"\b_pin_power_csv_"),
    re.compile(r"\b_result_session\b"),
)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def block(text: str, opener: str) -> str:
    """The brace-balanced body that starts at the first `{` after `opener`."""
    start = text.find(opener)
    if start < 0:
        return ""
    brace = text.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : i + 1]
    return ""


# ---------------------------------------------------------------------------
# 1. The snapshot owns everything it touches.
# ---------------------------------------------------------------------------
def scan_snapshot(io_cpp: str) -> list[str]:
    bad: list[str] = []
    body = block(io_cpp, "struct PinPowerCsvRecord")
    if not body:
        return ["PinPowerCsvRecord is gone -- the snapshot type is the contract"]
    # Field declarations only: everything up to the first member function.
    fields = body.split("[[nodiscard]]")[0]
    for line in fields.splitlines():
        code = line.strip()
        if not code or code in ("{", "}"):
            continue
        if "&" in code or re.search(r"\w\s*\*\s*\w", code):
            bad.append("PinPowerCsvRecord holds a reference/pointer: " + code)
        for solver in SOLVER_TYPES:
            if solver in code:
                bad.append("PinPowerCsvRecord names a solver type (%s): %s" % (solver, code))

    emit = block(io_cpp, "void PinPowerCsvRecord::Emit()")
    if not emit:
        return bad + ["PinPowerCsvRecord::Emit is gone -- nothing to check"]
    for pattern in SOLVER_REACHES:
        hit = pattern.search(emit)
        if hit:
            bad.append("Emit() reaches into solver state: %r" % hit.group(0))
    for solver in SOLVER_TYPES:
        if solver in emit:
            bad.append("Emit() names a solver type: %s" % solver)
    return bad


# ---------------------------------------------------------------------------
# 2. The queue stays bounded, and side tasks run outside the HDF5 guard.
# ---------------------------------------------------------------------------
def scan_queue(io_writer: str) -> list[str]:
    bad: list[str] = []
    push = block(io_writer, "void pushSideTask")
    if not push:
        return ["pushSideTask is gone -- there is no hand-off to bound"]
    if "_batch.bytes += bytes" not in push:
        bad.append("pushSideTask does not charge its payload to the batch byte bound")
    if "_inline" not in push:
        bad.append("pushSideTask has no inline branch -- inline mode would drop the task")

    submit = block(io_writer, "void submit(Batch&& batch)")
    if not submit:
        bad.append("Queue::submit is gone")
    else:
        if "queueDepthLimit()" not in submit or "queueByteLimit()" not in submit:
            bad.append("Queue::submit no longer consults both queue bounds")
        if "_not_full.wait" not in submit:
            bad.append("Queue::submit no longer blocks on a full queue (unbounded growth)")
        if "block_ns" not in submit:
            bad.append("Queue::submit no longer accounts solver block time")

    replay = block(io_writer, "inline void replay(Batch& batch)")
    if not replay:
        bad.append("replay() is gone")
    else:
        tasks = replay.find("batch.tasks")
        guard = replay.find("Chiffon::Hdf5Guard")
        if tasks < 0:
            bad.append("replay() never runs batch.tasks -- side tasks would be dropped")
        elif guard >= 0 and tasks > guard:
            bad.append("replay() runs side tasks INSIDE Hdf5Guard -- the CSV format "
                       "would hold the process-global HDF5 lock")
    if "_batch.ops.empty() && _batch.tasks.empty()" not in io_writer:
        bad.append("Recorder::submit can drop a batch that carries only side tasks")
    return bad


# ---------------------------------------------------------------------------
# 3. The join happens before the file closes.
# ---------------------------------------------------------------------------
def scan_join(io_cpp: str) -> list[str]:
    bad: list[str] = []
    close = block(io_cpp, "void IO::CloseResult()")
    if not close:
        return ["IO::CloseResult is gone"]
    fence = close.find("FenceJobWrites()")
    reset = close.find("_result_session.reset()")
    if fence < 0:
        bad.append("CloseResult does not fence the writer queue")
    elif reset >= 0 and fence > reset:
        bad.append("CloseResult drops the session BEFORE the fence -- a queued task "
                   "would outlive the file it writes beside")
    fj = block(io_cpp, "void IO::FenceJobWrites()")
    if fj and "iowriter::fence(_result_session)" not in fj:
        bad.append("FenceJobWrites no longer waits on the result session")
    return bad


# ---------------------------------------------------------------------------
# 4. Feature-off is the old path.
# ---------------------------------------------------------------------------
def scan_feature_off(io_cpp: str, io_writer: str) -> list[str]:
    bad: list[str] = []
    req = block(io_writer, "inline bool resultAsyncRequested()")
    if not req:
        return ["resultAsyncRequested is gone -- the gate has no default"]
    if "RASBERY_RESULT_ASYNC" not in req:
        bad.append("the gate no longer reads RASBERY_RESULT_ASYNC")
    if "return false" not in req:
        bad.append("the gate has no unset->false default; feature-off is not the default")
    if "mode() == Mode::Thread" not in io_writer:
        bad.append("async is not conditioned on the writer thread existing")

    site = block(io_cpp, "void IO::WriteStepToResult")
    if not site:
        return bad + ["IO::WriteStepToResult is gone"]
    if "iowriter::resultAsyncEnabled()" not in site:
        bad.append("the call site does not consult the gate")
    if "csv_record.Emit()" not in site:
        bad.append("the sync branch no longer calls the shared emitter -- feature-off "
                   "would be a second implementation, not the old one")
    if "record.Emit()" not in site:
        bad.append("the async branch no longer calls the shared emitter")
    if "std::ofstream" in site:
        bad.append("WriteStepToResult still opens the CSV itself -- the serialisation "
                   "did not actually move into the record")
    return bad


# ---------------------------------------------------------------------------
# Negative controls.  Each is the same scan run over a source that breaks it.
# ---------------------------------------------------------------------------
CONTROL_SNAPSHOT = """
struct PinPowerCsvRecord {
    Geometry&           g;
    std::vector<double> hz;
    [[nodiscard]] std::size_t bytes() const { return 0; }
    void Emit() const;
};
void PinPowerCsvRecord::Emit() const {
    for (int k = 0; k < nz; ++k) csv << g.hz(k);
}
"""

CONTROL_QUEUE = """
    template <class F>
    void pushSideTask(F&& task, std::size_t bytes = 0) {
        _batch.tasks.emplace_back(std::forward<F>(task));
    }
    void submit(Batch&& batch) {
        std::unique_lock<std::mutex> lock(_mtx);
        _queue.push_back(std::move(batch));
    }
inline void replay(Batch& batch) {
    Chiffon::Hdf5Guard hdf5_guard;
    for (std::function<void()>& task : batch.tasks) task();
}
"""

CONTROL_JOIN = """
void IO::CloseResult() {
    _result_session.reset();
    FenceJobWrites();
}
void IO::FenceJobWrites() const {
}
"""

CONTROL_FEATURE_OFF_WRITER = """
inline bool resultAsyncRequested() {
    return true;
}
"""

CONTROL_FEATURE_OFF_IO = """
void IO::WriteStepToResult(Geometry& g, const XSSet& xs, int schedule_index) {
    std::ofstream csv(_pin_power_csv_path);
    csv << 1.0;
}
"""


def main() -> int:
    failures: list[str] = []
    io_cpp = strip_comments(IO_CPP.read_text(encoding="utf-8"))
    io_writer = strip_comments(IO_WRITER.read_text(encoding="utf-8"))

    failures += scan_snapshot(io_cpp)
    failures += scan_queue(io_writer)
    failures += scan_join(io_cpp)
    failures += scan_feature_off(io_cpp, io_writer)

    controls = [
        ("snapshot leaks a solver reference", scan_snapshot(CONTROL_SNAPSHOT)),
        ("queue unbounded / task under the HDF5 guard", scan_queue(CONTROL_QUEUE)),
        ("join after the session is dropped", scan_join(CONTROL_JOIN)),
        ("gate defaults to async / CSV still written inline",
         scan_feature_off(CONTROL_FEATURE_OFF_IO, CONTROL_FEATURE_OFF_WRITER)),
    ]
    for name, hits in controls:
        if not hits:
            failures.append("negative control PASSED -- the scan is vacuous for: " + name)

    if failures:
        print("result async contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("result async contract: PASS (4 scans, %d negative controls)" % len(controls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
