#!/usr/bin/env python3
"""Static contract for the dedicated HDF5 writer thread (RASBERY_IO_WRITER).

The writer thread is validated end-to-end by the bit-golden gates -- the whole
claim is that `thread` and `inline` produce the same file -- so this contract
covers the properties an h5diff CANNOT see:

  1. the env gate resolves to exactly four outcomes over two modes and two
     provenances (unset/garbage -> thread(default), thread -> thread(env),
     inline -> inline(env)), is read ONCE, is the only reader of the variable,
     and publishes its PROVENANCE in the receipt;
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
# 1. The env gate: two modes x two provenances, cached once, single reader.
#
# ADOPTION 2026-08-27: `thread` is the DEFAULT in every mode of execution
# (byte-identical in all validated configurations, +0.6 % M64), and `inline` is
# the legacy path an explicit env still selects.  The receipt must be able to
# tell thread(default) from thread(env) from inline(env), or an adopted default
# and a deliberate A/B arm are indistinguishable in a log.
# ---------------------------------------------------------------------------
gate = body_after("inline const Resolution& resolution()")
if 'std::getenv("RASBERY_IO_WRITER")' not in gate:
    fail("resolution() does not read RASBERY_IO_WRITER")
if "static const Resolution resolved" not in gate:
    fail("the RASBERY_IO_WRITER gate is not cached in a function-local static")
# State 1: unset (and empty) is the ADOPTED DEFAULT -- the writer thread.
if ("if (value == nullptr || *value == '\\0')\n            return Resolution{Mode::Thread, "
        "ModeSource::Default};") not in gate:
    fail("an unset/empty RASBERY_IO_WRITER does not default to thread(default); the writer "
         "thread was adopted as the default on 2026-08-27")
# State 2: the explicit thread request -- same mode, different provenance.
if 'if (requested == "thread") return Resolution{Mode::Thread, ModeSource::Env};' not in gate:
    fail("RASBERY_IO_WRITER=thread does not resolve to thread(env)")
# State 3: the legacy path, reachable only on purpose.
if 'if (requested == "inline") return Resolution{Mode::Inline, ModeSource::Env};' not in gate:
    fail("RASBERY_IO_WRITER=inline is not an accepted explicit state; the legacy path must "
         "stay reachable for a bisect or an A/B arm")
# State 4: a typo falls back to the DEFAULT (not to inline), and says so.
if "[RASBERY][WARN][IO_WRITER] unknown RASBERY_IO_WRITER=" not in gate:
    fail("an unknown RASBERY_IO_WRITER value is accepted silently")
warn_at = gate.find("[RASBERY][WARN][IO_WRITER] unknown")
if "Resolution{Mode::Thread, ModeSource::Default}" not in gate[warn_at:]:
    fail("an unknown RASBERY_IO_WRITER value does not fall back to the default; a typo must "
         "buy the path the goldens were frozen on, not the legacy one")
if gate.count("return Resolution{Mode::Inline,") != 1:
    fail("inline is reachable from more than the one explicit state")
if gate.count("return Resolution{Mode::Thread,") != 3:
    fail("resolution() has thread outcomes other than the three declared states "
         "(unset, explicit, typo-fallback)")
if "ModeSource::Env" not in gate or "ModeSource::Default" not in gate:
    fail("resolution() does not record where the mode came from")
if "std::tolower" not in gate:
    fail("the mode string is not case-folded (RASBERY_HOST_PINNING sets the precedent)")
if WRITER_CODE.count('getenv("RASBERY_IO_WRITER")') != 1:
    fail("RASBERY_IO_WRITER is read outside the single cached gate")
for other in (IO_CPP_CODE, strip_comments(MAIN), strip_comments(DRIVER), strip_comments(IO_H)):
    if "RASBERY_IO_WRITER" in other and "RASBERY_IO_WRITER_" not in other:
        fail("RASBERY_IO_WRITER is read outside IoWriter.h")

# mode()/modeSource() are thin views on ONE cached answer -- two statics could
# disagree after an env change between them.
for accessor, expected in (("inline Mode       mode()", "return resolution().mode;"),
                           ("inline ModeSource modeSource()", "return resolution().source;")):
    if expected not in body_after(accessor):
        fail(f"{accessor} does not read through the single cached resolution()")
source_name = body_after("inline const char* modeSourceName()")
for token in ('"env"', '"default"'):
    if token not in source_name:
        fail(f"modeSourceName() does not publish {token}")

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

# 2b. Every write_raw in the write path goes through RawDataSet, never through a
#     bare HighFive handle -- and IO.cpp keeps none, so every write_raw call site
#     it has is a proxy call.  (2a already proved no HighFive handle types
#     survive there; this pins the call FORM too.)
if "->write_raw(" in IO_CPP_CODE:
    fail("IO.cpp writes through a raw HighFive handle")
raw_writes = re.findall(r"(\w+)\s*\.\s*write_raw\s*\(", IO_CPP_CODE)
if not raw_writes:
    fail("IO.cpp has no write_raw call sites at all; the write path moved somewhere unexpected")
for receiver in set(raw_writes):
    # Every receiver must be a RawDataSet: either the temporary returned by
    # createDataSet<T>(name, dims), or a local bound from one.
    if not re.search(rf"createDataSet<[^>]+>\([^;]*\)\s*\.\s*write_raw\(", IO_CPP_CODE) and \
       not re.search(rf"\b{re.escape(receiver)}\s*=\s*\w+\.createDataSet<", IO_CPP_CODE):
        fail(f"write_raw receiver {receiver!r} in IO.cpp is not a RawDataSet from createDataSet")

# 2c. The single replay site, and it holds the guard for the whole batch: the
#     Driver threads still READ HDF5, so one thread inside the runtime is still
#     the invariant -- the writer just holds the lock on the solver's behalf.
replay = body_after("inline void replay(Batch& batch)")
if "Chiffon::Hdf5Guard hdf5_guard;" not in replay:
    fail("replay() does not take the HDF5 guard; concurrent reads would race it")
guard_at = replay.find("Chiffon::Hdf5Guard hdf5_guard;")
ctx_m = re.search(r"ReplayCtx\s+ctx\{", replay)
loop_m = re.search(r"for\s*\(Op&\s*op\s*:\s*batch\.ops\)\s*op\(ctx\);", replay)
if ctx_m is None:
    fail("replay() no longer builds a ReplayCtx")
if loop_m is None:
    fail("replay() no longer runs the recorded ops")
if not 0 <= guard_at < ctx_m.start() < loop_m.start():
    fail("the replay context is not constructed and destroyed inside the guard scope "
         "(Group/DataSet destructors re-enter HDF5 too)")
if len(re.findall(r"for\s*\(Op&\s*op\s*:\s*batch\.ops\)\s*op\(ctx\);", WRITER_CODE)) != 1:
    fail("recorded ops are executed from more than one site")
# replay() is entered from exactly two places: the drain loop, and the
# post-shutdown in-place fallback.  Anything else is a second writer.
if len(re.findall(r"(?<!inline void )\breplay\(batch\)", WRITER_CODE)) != 2:
    fail("replay() is called from an unexpected number of sites")

# 2e. FAILURE ISOLATION.  A session whose file failed to open must be ABSORBING:
#     later batches are skipped, never replayed, or their ops walk a null file
#     and the writer thread segfaults -- taking all 64 decks down, not one.
poison_at = replay.find("batch.session->failed")
if poison_at < 0:
    fail("replay() does not consult session->failed before replaying")
if not 0 <= poison_at < guard_at:
    fail("the poisoned-session check runs AFTER the ops; it must gate them")
if "counters().skipped.fetch_add(" not in replay:
    fail("skipped batches are not counted; that is silent data loss")
if "batch.ops.clear();" not in replay:
    fail("a poisoned batch's ops are not dropped")
null_guard = re.search(r"if\s*\(!batch\.opens_file\s*&&\s*!ctx\.session\.file\)", replay)
if null_guard is None:
    fail("replay() has no null-file guard for batches that do not open the file")
if not null_guard.start() < loop_m.start():
    fail("the null-file guard runs after the ops it is supposed to protect")
if not re.search(r"bool\s+opens_file\s*=\s*false;", WRITER_CODE):
    fail("Batch does not carry the opens_file flag the null guard reads")
if not re.search(r"_batch\.opens_file\s*=\s*true;", WRITER_CODE):
    fail("openOverwrite does not mark its batch as the one that opens the file")

# Slot lookups are bounds-checked, and nothing indexes the vectors raw.
for signature, helper in (("inline HighFive::Group& groupAt(", "groupAt"),
                          ("inline HighFive::DataSet& dataSetAt(", "dataSetAt")):
    block = body_after(signature)
    if "out of range" not in block:
        fail(f"{helper}() does not bounds-check the slot")
    if "throw std::runtime_error" not in block:
        fail(f"{helper}() does not throw on an out-of-range slot")
# The accessors themselves are the one place allowed to index raw -- that is
# what they are.  Everywhere else must go through them.
outside_accessors = WRITER_CODE
for signature in ("inline HighFive::Group& groupAt(", "inline HighFive::DataSet& dataSetAt("):
    outside_accessors = outside_accessors.replace(body_after(signature), "")
if re.search(r"ctx\.(groups|datasets)\[", outside_accessors):
    fail("a replay op indexes the handle vectors directly instead of via the checked accessor")
file_of = body_after("inline HighFive::File& fileOf(ReplayCtx& ctx)")
if "if (!ctx.session.file)" not in file_of or "throw std::runtime_error" not in file_of:
    fail("fileOf() does not refuse a closed/never-opened file")
if re.search(r"ctx\.session\.file->", WRITER_CODE):
    fail("a replay op dereferences session.file directly instead of via fileOf()")

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
if not re.search(r"dataSetAt\(ctx,\s*slot\)\s*\.\s*write_raw\(payload\.data\(\)\);", write_raw):
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
if not re.search(r"std::deque<Batch>\s+_queue;", WRITER_CODE):
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

# A teardown while an enqueuer is parked must WAKE it, or it waits on a queue
# that will never drain again.
wait_pred = submit[submit.find("_not_full.wait(lock,"):]
wait_pred = wait_pred[: wait_pred.find("});") + 3]
if "_stopped" not in wait_pred:
    fail("the _not_full predicate ignores _stopped; a teardown would strand a blocked Driver")
if not re.search(r"if\s*\(_stopped\)\s*\{[^}]*replayHere\(batch\);", submit, flags=re.S):
    fail("a Driver woken by shutdown does not fall back to running its batch in place")

# The writer thread is CONSTRUCTED before the started flag is latched: a throwing
# std::thread ctor with the flag already set leaves a queue nobody services and
# every fence() on it hangs forever.
start_block = body_after("if (!_started)", text=submit)
ctor_at = start_block.find("std::thread([this] { run(); })")
latch_at = start_block.find("_started = true;")
if ctor_at < 0 or latch_at < 0:
    fail("submit() no longer starts the writer thread the way the contract describes")
if not ctor_at < latch_at:
    fail("_started is latched before the std::thread ctor can throw")

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

# ~Queue() runs during STATIC DESTRUCTION.  It must not replay (that would enter
# HDF5, std::cout and counters() when none of them is guaranteed to exist).
dtor = body_after("~Queue()")
for banned in ("replay(", "replayHere(", "counters()", "std::cout", "std::cerr"):
    if banned in dtor:
        fail(f"~Queue() touches {banned} -- no late I/O is allowed during static destruction")
if "std::fprintf(stderr" not in dtor:
    fail("~Queue() abandons batches without warning on stderr")
if "worker.join()" not in dtor:
    fail("~Queue() does not join the writer; a joinable thread would std::terminate")
if "abandoned.swap(_queue);" not in dtor and "_queue.clear();" not in dtor:
    fail("~Queue() does not clear the undrained queue")
if "--batch.session->pending;" not in dtor or "cv.notify_all();" not in dtor:
    fail("~Queue() leaves sessions with pending batches; a fence() would hang")

# A failure names its job and its file, is counted, and is never swallowed.
# One emitter, poison(), so the two catch arms cannot drift apart.
poison_fn = body_after("inline void poison(FileSession& session, const std::string& what)")
if "[RASBERY][IO_WRITER][FAIL]" not in poison_fn:
    fail("poison() does not emit the FAIL receipt")
if "session.job" not in poison_fn or "session.path" not in poison_fn:
    fail("the FAIL receipt does not name its job and its file")
if "session.failed = true;" not in poison_fn:
    fail("poison() does not mark the session failed")
if "counters().failures.fetch_add(" not in poison_fn:
    fail("writer-thread failures are not counted")
if WRITER_CODE.count("[RASBERY][IO_WRITER][FAIL]") != 2:  # poison() and ~Queue()'s stderr note
    fail("the FAIL receipt is emitted from an unexpected number of sites")
# Both exception arms of replay() must reach it -- std::exception AND (...).
if "catch (const std::exception& error)" not in replay or "catch (...)" not in replay:
    fail("replay() does not catch both exception forms")
if replay.count("poison(*batch.session,") != 2:
    fail("a writer-thread failure is not reported on both exception paths")

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
if "ThrowIfWritesFailed();" not in job_fence:
    fail("FenceJobWrites drains but never checks the result")

# The failure lands on the job that owns it, and it is checked at BOTH the
# blocking fence and the cheap per-statepoint probe.
job_check = body_after("void IO::ThrowIfWritesFailed() const", text=IO_CPP_CODE)
if "throw std::runtime_error" not in job_check:
    fail("a writer failure does not fail its own job")
if "_restart_sessions" not in job_check:
    fail("the failure probe ignores the restart snapshots")
if "iowriter::fence(" in job_check:
    fail("ThrowIfWritesFailed blocks; it must be the NON-waiting probe")
if "std::lock_guard<std::mutex> lock(session->mtx);" not in job_check:
    fail("the failure probe reads session->failed without its mutex")
# ...and the probe is what stops a doomed deck from computing a whole run.
write_step = body_after("void IO::WriteStepToResult", text=IO_CPP_CODE)
if "ThrowIfWritesFailed();" not in write_step:
    fail("a deck whose output file could not be created keeps computing statepoints")
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

# BOTH deck loops isolate a failing deck.  The serial loop used to let the
# exception escape main(), which aborted the process and killed every deck after
# the bad one -- the exact collateral the batch loop already refuses.
if main_code.count("driver.Drive()") != 2:
    fail("main() no longer drives decks from exactly the two expected loops")
for loop, anchor in (("batch", "job_status[static_cast<std::size_t>(i)] = driver.Drive();"),
                     ("serial", "driver_exit_code = driver.Drive();")):
    at = main_code.find(anchor)
    if at < 0:
        fail(f"the {loop} deck loop does not call Drive() the way the contract describes")
    window = main_code[max(0, at - 400):at]
    if "try {" not in window:
        fail(f"the {loop} deck loop does not guard Drive(); one bad deck would abort the process")
    after = main_code[at:at + 500]
    if "catch (const std::exception& error)" not in after or "catch (...)" not in after:
        fail(f"the {loop} deck loop does not catch both exception forms")
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
              "writer_busy_ms", "failures", "ops", "max_queue_bytes", "skipped"):
    if f'\\"{field}\\":' not in summary:
        fail(f"the summary receipt omits {field}")
if '"[RASBERY][IO_WRITER][SUMMARY]' not in summary:
    fail("the summary receipt does not use the [SUMMARY] tag of the receipt family")
# PROVENANCE, on BOTH receipts.  With thread as the default, `mode":"thread"`
# alone no longer says whether a run was configured or merely defaulted -- and
# an A/B whose arms cannot be told apart from the log is void.
for receipt, block in (("configuration", config), ("summary", summary)):
    if '\\"mode_source\\":\\""' not in block:
        fail(f"the {receipt} receipt does not publish mode_source; thread(default), "
             "thread(env) and inline(env) would be indistinguishable")
    if "modeSourceName()" not in block:
        fail(f"the {receipt} receipt's mode_source is not the resolved provenance")
    if "modeName()" not in block:
        fail(f"the {receipt} receipt's mode is not the resolved mode")

# ---------------------------------------------------------------------------
# 6. Inline is no longer the DEFAULT, but it is still the OLD path.
#
# The adoption moved the default; it must not have moved the legacy arm.  A
# bisect that sets RASBERY_IO_WRITER=inline has to get literally the
# pre-writer-thread code, so every property that made inline trivially
# byte-golden is still asserted here.
# ---------------------------------------------------------------------------
recorder_submit = body_after("void submit()", text=WRITER_CODE)
if "if (_inline || _batch.ops.empty()) {" not in recorder_submit:
    fail("submit() queues something in inline mode; inline must execute in place")
# Slots index the PER-BATCH replay context, so they must reset with the batch --
# otherwise a reused recorder hands its second batch indices from the first.
if recorder_submit.count("resetBatch();") < 2:
    fail("submit() does not reset the batch on both return paths")
reset = body_after("void resetBatch()")
for field in ("_group_slots", "_dataset_slots"):
    if not re.search(rf"{field}\s*=\s*0;", reset):
        fail(f"resetBatch() does not reset {field}")
if not re.search(r"_batch\s*=\s*Batch\{\};", reset):
    fail("resetBatch() does not start a fresh batch")
# A failed hand-off must undo the pending it already counted, or fence() hangs.
if "queue().submit(std::move(_batch));" not in recorder_submit:
    fail("submit() does not hand the batch to the queue")
if "catch (const std::exception& error)" not in recorder_submit:
    fail("submit() does not guard the hand-off; a throwing thread ctor would strand pending")
if "--_session->pending;" not in recorder_submit:
    fail("a failed hand-off does not release the pending count it took")
if "poison(*_session," not in recorder_submit:
    fail("a failed hand-off does not fail its job")
if "throw;" not in recorder_submit:
    fail("a failed hand-off is swallowed instead of reaching the Driver")
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
# A finished deck's telemetry must be durable: flush at the run-summary site so
# an abnormal exit loses only the lines of decks still running.
summary_at = driver_code.find("[RASBERY][SPTELEM][SUMMARY]")
if summary_at < 0:
    fail("the SPTELEM summary receipt disappeared")
if "iowriter::flushLines();" not in driver_code[summary_at:summary_at + 3000]:
    fail("the SPTELEM run summary is not flushed; an abnormal exit would lose finished decks")

py_compile.compile(str(Path(__file__).resolve()), doraise=True)
print("io writer contract: PASS")
