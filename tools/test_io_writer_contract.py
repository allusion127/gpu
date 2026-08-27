#!/usr/bin/env python3
"""Static contract for the dedicated HDF5 writer thread (RASBERY_IO_WRITER).

The writer thread is validated end-to-end by the bit-golden gates -- the whole
claim is that `thread` and `inline` produce the same file -- so this contract
covers the properties an h5diff CANNOT see:

  1. the env gate has exactly three outcomes (unset/inline -> inline, thread ->
     thread, anything else -> warn + inline), is read ONCE, and is the only
     reader of the variable;
  2. the writer thread OWNS the write path: no HighFive write call site and no
     DataSpace construction survives outside IoWriter.h, every remaining
     HighFive::File in IO.cpp is ReadOnly, and the single replay site holds
     Hdf5Guard (the reads still run on Driver threads);
  3. order and payload ownership -- the two things byte-identity rests on: ops
     replay in record order onto push_back'd handle slots, dataset creation and
     write_raw are separate ops, and every payload is copied at record time;
  4. the queue is bounded on BOTH count and bytes, blocks instead of dropping,
     and charges the block to enqueue_block_ms;
  5. shutdown drains before it joins, a writer-thread failure names its job and
     reaches the exit code, and both main() branches emit both receipts.
"""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = (ROOT / "src" / "IoWriter.h").read_text(encoding="utf-8-sig")
IO_CPP = (ROOT / "src" / "IO.cpp").read_text(encoding="utf-8-sig")
IO_H = (ROOT / "src" / "IO.h").read_text(encoding="utf-8-sig")
MAIN = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8-sig")
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")


def fail(message: str) -> None:
    raise SystemExit(f"io writer contract: FAIL: {message}")


def body_after(anchor: str, *, text: str = WRITER) -> str:
    """The brace-matched block that opens at the first '{' after `anchor`."""
    start = text.find(anchor)
    if start < 0:
        fail(f"anchor not found: {anchor!r}")
    open_at = text.find("{", start)
    if open_at < 0:
        fail(f"no block after {anchor!r}")
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    fail(f"unbalanced block after {anchor!r}")
    return ""


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments so prose can never satisfy a code assertion."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


WRITER_CODE = strip_comments(WRITER)
IO_CPP_CODE = strip_comments(IO_CPP)

# ---------------------------------------------------------------------------
# 1. The env gate: three states, cached once, single reader.
# ---------------------------------------------------------------------------
gate = body_after("inline Mode mode()")
if 'std::getenv("RASBERY_IO_WRITER")' not in gate:
    fail("mode() does not read RASBERY_IO_WRITER")
if "static const Mode resolved" not in gate:
    fail("the RASBERY_IO_WRITER gate is not cached in a function-local static")
# State 1: unset (and empty) is the production default.
if "if (value == nullptr || *value == '\\0') return Mode::Inline;" not in gate:
    fail("an unset RASBERY_IO_WRITER does not default to inline")
# State 2: the opt-in.
if 'if (requested == "thread") return Mode::Thread;' not in gate:
    fail("RASBERY_IO_WRITER=thread does not select the writer thread")
# State 3: the explicit legacy value, and the typo guard that shares its answer.
if 'if (requested != "inline")' not in gate:
    fail("RASBERY_IO_WRITER=inline is not an accepted explicit state")
if "[RASBERY][WARN][IO_WRITER] unknown RASBERY_IO_WRITER=" not in gate:
    fail("an unknown RASBERY_IO_WRITER value is accepted silently")
if gate.count("return Mode::Inline;") != 2 or gate.count("return Mode::Thread;") != 1:
    fail("mode() has outcomes other than the three declared states")
if "std::tolower" not in gate:
    fail("the mode string is not case-folded (RASBERY_HOST_PINNING sets the precedent)")
if WRITER_CODE.count('getenv("RASBERY_IO_WRITER")') != 1:
    fail("RASBERY_IO_WRITER is read outside the single cached gate")
for other in (IO_CPP_CODE, strip_comments(MAIN), strip_comments(DRIVER), strip_comments(IO_H)):
    if "RASBERY_IO_WRITER" in other and "RASBERY_IO_WRITER_" not in other:
        fail("RASBERY_IO_WRITER is read outside IoWriter.h")

# The two queue bounds are gated the same way: cached, opt-in, with a default.
for env, fn, default in (("RASBERY_IO_WRITER_QUEUE", "queueDepthLimit", "64"),
                         ("RASBERY_IO_WRITER_QUEUE_MB", "queueByteLimit", "512")):
    block = body_after(f"inline std::size_t {fn}()")
    if f'std::getenv("{env}")' not in block:
        fail(f"{fn}() does not read {env}")
    if "static const std::size_t limit" not in block:
        fail(f"{env} is not cached in a function-local static")
    if default not in block:
        fail(f"{fn}() lost its documented default of {default}")

# ---------------------------------------------------------------------------
# 2. The writer thread owns every HDF5 WRITE call.
# ---------------------------------------------------------------------------
# 2a. IO.cpp keeps no HighFive write handle at all.  Everything left is a read.
for banned in ("HighFive::Group", "HighFive::DataSet", "HighFive::DataSpace",
               "_result_file"):
    if banned in IO_CPP_CODE:
        fail(f"{banned} still appears in IO.cpp: a write path bypasses the recorder")
if "HighFive::File" in strip_comments(IO_H):
    fail("IO.h still owns a HighFive::File; the session owns it now")
for match in re.finditer(r"HighFive::File[^;]*;", IO_CPP_CODE, flags=re.S):
    if "ReadOnly" not in match.group(0):
        fail(f"a non-ReadOnly HighFive::File survives in IO.cpp: {match.group(0)!r}")

# 2b. Every HDF5 write call in the process lives in IoWriter.h, and inside it
#     only in the inline arms (guarded by the recorder) or in a recorded op.
if "write_raw" in IO_CPP_CODE and "createDataSet" not in IO_CPP_CODE:
    fail("write_raw without createDataSet in IO.cpp")
if "->write_raw(" in IO_CPP_CODE:
    fail("IO.cpp writes through a raw HighFive handle")

# 2c. The single replay site, and it holds the guard for the whole batch: the
#     Driver threads still READ HDF5, so one thread inside the runtime is still
#     the invariant -- the writer just holds the lock on the solver's behalf.
replay = body_after("inline void replay(Batch& batch)")
if "Chiffon::Hdf5Guard hdf5_guard;" not in replay:
    fail("replay() does not take the HDF5 guard; concurrent reads would race it")
guard_at = replay.find("Chiffon::Hdf5Guard hdf5_guard;")
ctx_at = replay.find("ReplayCtx          ctx{")
loop_at = replay.find("for (Op& op : batch.ops) op(ctx);")
if not 0 <= guard_at < ctx_at < loop_at:
    fail("the replay context is not constructed and destroyed inside the guard scope "
         "(Group/DataSet destructors re-enter HDF5 too)")
if WRITER_CODE.count("for (Op& op : batch.ops) op(ctx);") != 1:
    fail("recorded ops are executed from more than one site")
if WRITER_CODE.count("replay(batch)") != 2:  # the drain loop and the post-shutdown fallback
    fail("replay() is called from an unexpected number of sites")

# 2d. The recorder is the ONLY thing that takes the guard on a Driver thread,
#     and only in inline mode -- a thread-mode Driver must hold no HDF5 lock
#     while it blocks on a full queue (that is the no-deadlock argument).
recorder_ctor = body_after("explicit Recorder(std::shared_ptr<FileSession> session)")
if "if (_inline) _guard.emplace();" not in recorder_ctor:
    fail("the recorder does not hold Hdf5Guard for the whole inline scope")
if "std::optional<Chiffon::Hdf5Guard> _guard;" not in WRITER_CODE:
    fail("the inline guard is not the optional member the constructor arms")
for fn in ("void IO::WriteStepToResult", "void IO::OpenResult", "void IO::CloseResult",
           "void IO::SaveRestart"):
    block = body_after(fn, text=IO_CPP_CODE)
    if "Chiffon::Hdf5Guard" in block:
        fail(f"{fn} still takes the HDF5 guard directly; the recorder owns it now")
    if "iowriter::Recorder rec(" not in block:
        fail(f"{fn} does not open a recorder scope")
    if "rec.submit();" not in block:
        fail(f"{fn} never submits its batch")

# ---------------------------------------------------------------------------
# 3. Order and payload ownership -- what byte-identity actually rests on.
# ---------------------------------------------------------------------------
# 3a. Handles are addressed by SLOT, and a slot is filled by push_back in replay
#     order, so slot k is always the k-th create of that kind in the batch.
create_group = body_after("inline Node Node::createGroup(const std::string& name) const")
if "ctx.groups.push_back(" not in create_group:
    fail("createGroup does not append its handle in replay order")
if "_recorder->nextGroupSlot()" not in create_group:
    fail("createGroup does not reserve its slot at record time")
raw_create = body_after(
    "inline RawDataSet Node::createDataSet(const std::string& name, const Dims& dims) const")
if "ctx.datasets.push_back(" not in raw_create:
    fail("the raw createDataSet does not append its handle in replay order")
if "HighFive::DataSpace space(extent);" not in raw_create:
    fail("the DataSpace is not built on the replay side from the recorded extents")
# The whole reason Dims exists: H5Screate_simple must not run on a Driver thread.
dims_doc = WRITER[WRITER.find("struct Dims") - 1400 : WRITER.find("struct Dims")]
if "H5Screate_simple" not in dims_doc:
    fail("Dims lost the note explaining why a DataSpace may not be built while recording")

# 3b. Creation and write_raw are SEPARATE ops, so a dataset created but never
#     written still gets its link -- exactly as the inline path does.
write_raw = body_after("inline void RawDataSet::write_raw(const T* buffer) const")
if "std::vector<T> owned(buffer, buffer + _count);" not in write_raw:
    fail("write_raw does not copy the caller's block into an owned buffer")
if "ctx.datasets[static_cast<std::size_t>(slot)].write_raw(payload.data());" not in write_raw:
    fail("the recorded write_raw does not target the slot its creation reserved")
if "_recorder->nextDataSetSlot()" in write_raw:
    fail("write_raw reserves its own slot; it must reuse the creation's")

# 3c. Value payloads are copied at record time (`payload = value`), never
#     captured by reference -- the solver mutates Geometry/XSSet immediately.
value_create = body_after(
    "inline void Node::createDataSet(const std::string& name, const T& value) const")
if "payload = value" not in value_create:
    fail("the value payload is not copied into the recorded op")
for capture in ("[&]", "[&,", "[=, &"):
    if capture in value_create or capture in raw_create or capture in write_raw:
        fail("a recorded op captures by reference; the payload must be owned")

# 3d. One FIFO queue, and a file's batches ride it in submission order.
if "std::deque<Batch>       _queue;" not in WRITER_CODE:
    fail("the queue is not a FIFO deque")
run = body_after("void run()")
if "_queue.front()" not in run or "_queue.pop_front()" not in run:
    fail("the writer does not drain the queue front-first")
if "_queue.push_back(std::move(batch));" not in WRITER_CODE:
    fail("submit() does not append to the queue tail")

# ---------------------------------------------------------------------------
# 4. Bounded queue and block accounting.
# ---------------------------------------------------------------------------
submit = body_after("void submit(Batch&& batch)")
if "queueDepthLimit()" not in submit or "queueByteLimit()" not in submit:
    fail("submit() does not enforce BOTH the count and the byte bound")
if "_not_full.wait(lock," not in submit:
    fail("a full queue does not block the enqueuing Driver")
if "counters().block_ns.fetch_add(" not in submit:
    fail("the enqueue block is not accounted")
block_at = submit.find("const auto blocked = std::chrono::steady_clock::now();")
wait_at = submit.find("_not_full.wait(lock,")
charge_at = submit.find("counters().block_ns.fetch_add(")
if not 0 <= block_at < wait_at < charge_at:
    fail("enqueue_block_ms is not measured across the wait it is supposed to measure")
if "|| _queue.empty()" not in submit:
    fail("a single batch larger than the byte bound would deadlock the queue")
if "bumpMax(counters().max_depth" not in submit or "bumpMax(counters().max_bytes" not in submit:
    fail("the queue high-water marks are not recorded")
if "_not_full.notify_all();" not in run:
    fail("the writer does not release blocked enqueuers after it dequeues")

# The bound must be checked BEFORE the batch is added, never after.
if submit.find("const bool full =") > submit.find("_bytes += batch.bytes;"):
    fail("submit() admits the batch before testing the bound")

# ---------------------------------------------------------------------------
# 5. Shutdown, failure reporting, receipts.
# ---------------------------------------------------------------------------
shutdown = body_after("void shutdown()")
if "_stopped = true;" not in shutdown or "worker.join()" not in shutdown:
    fail("shutdown() does not stop and join the writer")
if "if (_queue.empty()) return;" not in run:
    fail("the writer exits before the queue is drained")
if "!_queue.empty() || _stopped" not in run:
    fail("the drain loop does not wake on both work and shutdown")

top_shutdown = body_after("inline std::uint64_t shutdown()")
if "queue().shutdown();" not in top_shutdown or "flushLines();" not in top_shutdown:
    fail("the top-level shutdown does not drain the queue and flush the line sink")
if "return counters().failures.load();" not in top_shutdown:
    fail("shutdown() does not hand the failure count back to main()")

# A failure names its job and its file, and it is never swallowed.
if replay.count("[RASBERY][IO_WRITER][FAIL]") != 2:  # std::exception and (...)
    fail("a writer-thread failure is not reported on both exception paths")
if replay.count("batch.session->job") != 2 or replay.count("batch.session->failed = true;") != 2:
    fail("a writer-thread failure does not name its job / mark its session")
if "counters().failures.fetch_add(" not in replay:
    fail("writer-thread failures are not counted")

# ...and it lands on the job, through the fence, and on the process exit code.
fence = body_after("inline void fence(const std::shared_ptr<FileSession>& session)")
if "mode() == Mode::Inline) return;" not in fence:
    fail("fence() is not a no-op in inline mode")
if "session->cv.wait(lock" not in fence or "session->pending == 0" not in fence:
    fail("fence() does not wait for the session's batches to drain")
job_fence = body_after("void IO::FenceJobWrites() const", text=IO_CPP_CODE)
if "iowriter::fence(_result_session);" not in job_fence:
    fail("FenceJobWrites does not fence the result file")
if "_restart_sessions" not in job_fence:
    fail("FenceJobWrites ignores the restart snapshots")
if "throw std::runtime_error" not in job_fence:
    fail("a writer failure does not fail its own job")
close_result = body_after("void IO::CloseResult()", text=IO_CPP_CODE)
if "FenceJobWrites();" not in close_result:
    fail("CloseResult does not fence the job's writes before returning")
if "rec.closeFile();" not in close_result:
    fail("CloseResult does not close the file through the recorder")
dtor = body_after("IO::~IO()", text=IO_CPP_CODE)
if "iowriter::fence(_result_session);" not in dtor:
    fail("~IO drops the file handle without fencing the queued batches")

main_code = strip_comments(MAIN)
if main_code.count("rasbery::iowriter::reportConfig(std::cout);") != 2:
    fail("the [IO_WRITER] configuration receipt is not emitted from both main() branches")
if main_code.count("rasbery::iowriter::reportSummary(std::cout);") != 2:
    fail("the [IO_WRITER][SUMMARY] receipt is not emitted from both main() branches")
if main_code.count("rasbery::iowriter::shutdown()") != 2:
    fail("the writer is not drained from both main() branches")
if "if (io_writer_failures > 0 && exit_code == 0) exit_code = 1;" not in main_code:
    fail("a lost write does not reach the batch branch's exit code")
if "if (rasbery::iowriter::shutdown() > 0 && exit_code == 0) exit_code = 1;" not in main_code:
    fail("a lost write does not reach the serial branch's exit code")
# The drain must precede the verdicts it can change.
batch_shutdown = main_code.find("const std::uint64_t io_writer_failures")
verdicts = main_code.find('"[RASBERY][FAIL] exit_code="')
if not 0 <= batch_shutdown < verdicts:
    fail("the batch branch reads its per-job verdicts before the writer has drained")

config = body_after("inline void reportConfig(std::ostream& os)")
if '"[RASBERY][IO_WRITER] {\\"mode\\":\\""' not in config:
    fail("the configuration receipt does not publish the mode")
summary = body_after("inline void reportSummary(std::ostream& os)")
for field in ("requests", "bytes", "max_queue_depth", "enqueue_block_ms",
              "writer_busy_ms", "failures", "ops", "max_queue_bytes"):
    if f'\\"{field}\\":' not in summary:
        fail(f"the summary receipt omits {field}")
if '"[RASBERY][IO_WRITER][SUMMARY]' not in summary:
    fail("the summary receipt does not use the [SUMMARY] tag of the receipt family")

# ---------------------------------------------------------------------------
# 6. Inline is the default path, and it is the OLD path.
# ---------------------------------------------------------------------------
recorder_submit = body_after("void submit()", text=WRITER_CODE)
if "if (_inline || _batch.ops.empty()) {" not in recorder_submit:
    fail("submit() queues something in inline mode; inline must execute in place")
open_overwrite = body_after("void openOverwrite(const std::string& path)")
if "if (_inline) {" not in open_overwrite:
    fail("openOverwrite has no inline arm")
if open_overwrite.count("HighFive::File::Overwrite") != 2:
    fail("the inline and recorded opens no longer make the same HighFive call")
for arm in ("inline Node Node::createGroup", "inline void Node::createDataSet",
            "inline RawDataSet Node::createDataSet"):
    block = body_after(arm)
    if "_recorder->isInline()" not in block:
        fail(f"{arm} does not branch on the mode")

# ---------------------------------------------------------------------------
# 7. SPTELEM through the buffered, line-atomic sink (not std::cout).
# ---------------------------------------------------------------------------
driver_code = strip_comments(DRIVER)
if driver_code.count("iowriter::appendLine(std::format(") != 2:
    fail("the two SPTELEM receipts do not go through the line sink")
if 'std::cout << std::format(\n' in driver_code and "[RASBERY][SPTELEM]" in driver_code:
    site = driver_code.find("[RASBERY][SPTELEM]")
    if "std::cout" in driver_code[max(0, site - 200):site]:
        fail("an SPTELEM receipt still writes straight to std::cout")
append = body_after("inline void appendLine(const std::string& line)")
if "std::lock_guard<std::mutex> lock(sink.mtx);" not in append:
    fail("appendLine is not mutex-protected")
if "std::cout.write(" not in append:
    fail("appendLine does not write the buffer out")
if "mode() != Mode::Inline && sink.buffer.size() < kLineSinkFlushBytes" not in append:
    fail("inline mode does not flush every line (it must stay byte- and order-identical)")
# The write happens under the same lock as the append: that is the atomicity.
if append.find("std::lock_guard") > append.find("std::cout.write("):
    fail("appendLine writes outside the lock; a line could be split")
flush = body_after("inline void flushLines()")
if "sink.buffer.clear();" not in flush:
    fail("flushLines does not clear what it wrote")

py_compile.compile(str(Path(__file__).resolve()), doraise=True)
print("io writer contract: PASS")
