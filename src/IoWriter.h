#pragma once

// Dedicated HDF5 writer thread -- batch I/O de-serialisation (RASBERY_IO_WRITER).
//
// THE HOLE THIS CLOSES.  The HDF5 1.10.x builds RASBERY links are not
// thread-safe, so Chiffon::Hdf5Guard serialises every entry into the runtime
// behind one process-global recursive mutex.  That is correct and it is what
// makes --batch-mode legal at all, but it charges the whole cost to the SOLVER
// threads: at M64 the receipts show ~734 ms per lock acquisition and
// multi-million-ms cumulative waits, and the SPTELEM phase split puts io_wall at
// ~40% of a case's wall (vs ~3% for a single deck).  Sixty-four GPU-fed solvers
// spend nearly half their time queued behind an lseek.
//
// THE FIX IS OWNERSHIP, NOT A BIGGER LOCK.  A lock lets 64 threads take turns
// inside a non-thread-safe library; a dedicated thread means only ONE thread
// ever enters it for writing.  Driver threads no longer wait for the runtime --
// they record what they wanted written into an owned buffer, hand it to a
// bounded MPSC queue and return to the solve immediately.  The HDF5 work is not
// made smaller (it is the same calls in the same order), it is made CONCURRENT
// with the compute of the other 63 decks.
//
// HOW BYTE-IDENTITY IS GUARANTEED.  The recorder is not a serialiser and does
// not reimplement any HDF5 concept.  Every call the write path used to make
// through a HighFive handle is captured VERBATIM as a closure over an OWNED
// copy of its arguments, and the writer thread replays those closures:
//
//   * in order.  Ops replay in the order they were recorded, and a file's
//     batches replay in submission order (one FIFO queue, and one Driver thread
//     per file records in program order).  Link-insertion order in every group
//     is therefore exactly what the inline path produced.
//   * on the same handles.  A recorded createGroup/createDataSet pushes its
//     result onto the replay context's handle vector, so the k-th group op
//     always lands in slot k.  A child op names its parent by that slot, which
//     reproduces `parent.createDataSet(...)` -- not a re-derived path.
//   * with the same values.  The payload is copied at record time; the solver
//     may mutate Geometry/XSSet the instant it returns without touching it.
//
// So the replay IS the inline call sequence, executed later on another thread.
// The bit-golden gates (single-deck 500/500 datasets, batch 708/708) are the
// acceptance test for that claim, and RASBERY_IO_WRITER=inline (the default)
// keeps today's path literally unchanged for the runs that must not move.
//
// READS ARE NOT MOVED.  Deck parse, XSLIB load, restart load and shuffle still
// read HDF5 on the Driver thread under Hdf5Guard, so the writer thread takes
// the SAME guard once per batch.  The runtime therefore still has exactly one
// thread inside it; what changed is that the writer holds the guard on the
// solver's behalf instead of the solver holding it itself.  Per-batch (rather
// than per-op) acquisition also collapses the acquisition count.
//
// NO DEADLOCK.  The only blocking edge is solver -> full queue, and the writer
// never waits on a solver: it waits on the queue and on Hdf5Guard, both of
// which a solver releases without ever needing the queue.  A solver blocked on
// a full queue holds no HDF5 lock (in thread mode the recorder takes none).
//
// LAYERING.  Header-only, CUDA-free, and MSVC-clean (plain std::thread, no
// std::jthread, no gcc-only constructs), matching HostPinRegistry.h.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Hdf5Guard.h"
#include "highfive/highfive.hpp"

namespace rasbery::iowriter {

// ---------------------------------------------------------------------------
// The env gate.  Three states and nothing else: unset and "inline" are the
// production path (byte-identical to the pre-writer-thread code because it IS
// that code), "thread" opts into the dedicated writer.  Anything else warns and
// falls back to inline -- a typo must never silently change how a deck is
// written.  Read ONCE into a function-local static so the recorder and the
// queue can never disagree about which mode a run is in.
// ---------------------------------------------------------------------------
enum class Mode { Inline, Thread };

inline Mode mode() {
    static const Mode resolved = [] {
        const char* value = std::getenv("RASBERY_IO_WRITER");
        if (value == nullptr || *value == '\0') return Mode::Inline;
        std::string requested(value);
        std::transform(requested.begin(), requested.end(), requested.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (requested == "thread") return Mode::Thread;
        if (requested != "inline")
            std::cerr << "[RASBERY][WARN][IO_WRITER] unknown RASBERY_IO_WRITER=" << value
                      << " -- using inline.\n";
        return Mode::Inline;
    }();
    return resolved;
}

inline const char* modeName() { return mode() == Mode::Thread ? "thread" : "inline"; }

/// Bounded queue, batch count.  RASBERY_IO_WRITER_QUEUE overrides.
inline std::size_t queueDepthLimit() {
    static const std::size_t limit = [] {
        const char* value = std::getenv("RASBERY_IO_WRITER_QUEUE");
        const int   requested = (value != nullptr) ? std::atoi(value) : 0;
        return requested > 0 ? static_cast<std::size_t>(requested) : std::size_t{64};
    }();
    return limit;
}

/// Bounded queue, resident payload bytes.  The count bound alone is not enough:
/// one statepoint carries pin_power/pin_flux blocks of several MB, so 64 queued
/// batches is a memory decision as much as a latency one.  Whichever bound
/// trips first blocks the enqueuing Driver.  RASBERY_IO_WRITER_QUEUE_MB.
inline std::size_t queueByteLimit() {
    static const std::size_t limit = [] {
        const char* value = std::getenv("RASBERY_IO_WRITER_QUEUE_MB");
        const int   requested = (value != nullptr) ? std::atoi(value) : 0;
        const std::size_t mb = requested > 0 ? static_cast<std::size_t>(requested) : std::size_t{512};
        return mb * 1024u * 1024u;
    }();
    return limit;
}

// ---------------------------------------------------------------------------
// Receipt counters.  Same shape as HostPinCounters: plain atomics, published
// once at teardown, never read on a hot path.
// ---------------------------------------------------------------------------
struct Counters {
    std::atomic<std::uint64_t> requests{0};    ///< batches handed to the writer
    std::atomic<std::uint64_t> ops{0};         ///< recorded HDF5 calls replayed
    std::atomic<std::uint64_t> bytes{0};       ///< payload bytes moved through the queue
    std::atomic<std::uint64_t> max_depth{0};   ///< high-water queued batch count
    std::atomic<std::uint64_t> max_bytes{0};   ///< high-water queued payload bytes
    std::atomic<std::uint64_t> block_ns{0};    ///< total time Drivers spent blocked on a full queue
    std::atomic<std::uint64_t> writer_ns{0};   ///< total time the writer spent inside HDF5
    std::atomic<std::uint64_t> failures{0};    ///< batches that threw on the writer thread
};

inline Counters& counters() {
    static Counters c;
    return c;
}

inline void bumpMax(std::atomic<std::uint64_t>& slot, std::uint64_t value) {
    std::uint64_t current = slot.load(std::memory_order_relaxed);
    while (current < value && !slot.compare_exchange_weak(current, value)) {}
}

// ---------------------------------------------------------------------------
// One output file's lifetime, shared between the Driver that records for it and
// the writer that owns its handle.  `file` is touched ONLY by whichever thread
// is allowed to be inside HDF5 for it: the writer during replay, or the Driver
// itself in inline mode (and after a fence).
// ---------------------------------------------------------------------------
struct FileSession {
    std::unique_ptr<HighFive::File> file;
    std::string                     path;  ///< for the FAIL receipt
    std::string                     job;   ///< result_stem(), i.e. the job id

    std::mutex              mtx;
    std::condition_variable cv;
    std::size_t             pending = 0;   ///< submitted batches not yet replayed
    bool                    failed  = false;
    std::string             error;
};

/// Replay-side scratch.  `groups[k]` is the handle produced by the k-th group op
/// of this batch, `datasets[k]` likewise -- which is how a recorded op names its
/// parent without re-deriving a path.
struct ReplayCtx {
    FileSession&                   session;
    std::vector<HighFive::Group>   groups;
    std::vector<HighFive::DataSet> datasets;
};

using Op = std::function<void(ReplayCtx&)>;

struct Batch {
    std::shared_ptr<FileSession> session;
    std::vector<Op>              ops;
    std::size_t                  bytes = 0;
};

/// Execute one batch's ops against its file.  The ONLY place HDF5 write calls
/// happen on the thread path; `ctx` is declared inside the guard scope because
/// Group/DataSet destructors re-enter the runtime too.
inline void replay(Batch& batch) {
    const auto started = std::chrono::steady_clock::now();
    try {
        Chiffon::Hdf5Guard hdf5_guard;
        ReplayCtx          ctx{*batch.session, {}, {}};
        for (Op& op : batch.ops) op(ctx);
    } catch (const std::exception& error) {
        counters().failures.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(batch.session->mtx);
        batch.session->failed = true;
        if (batch.session->error.empty()) batch.session->error = error.what();
        std::cout << "[RASBERY][IO_WRITER][FAIL] {\"job\":\"" << batch.session->job
                  << "\",\"path\":\"" << batch.session->path << "\",\"what\":\""
                  << error.what() << "\"}" << std::endl;
    } catch (...) {
        counters().failures.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(batch.session->mtx);
        batch.session->failed = true;
        if (batch.session->error.empty()) batch.session->error = "unknown exception";
        std::cout << "[RASBERY][IO_WRITER][FAIL] {\"job\":\"" << batch.session->job
                  << "\",\"path\":\"" << batch.session->path
                  << "\",\"what\":\"unknown exception\"}" << std::endl;
    }
    counters().writer_ns.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - started)
                                       .count()),
        std::memory_order_relaxed);
    counters().ops.fetch_add(batch.ops.size(), std::memory_order_relaxed);

    std::shared_ptr<FileSession> session = batch.session;
    {
        std::lock_guard<std::mutex> lock(session->mtx);
        if (session->pending > 0) --session->pending;
    }
    session->cv.notify_all();
}

// ---------------------------------------------------------------------------
// The bounded MPSC queue and the one thread that drains it.
// ---------------------------------------------------------------------------
class Queue {
public:
    Queue() = default;

    /// Belt and braces only.  main() drains this explicitly in both branches
    /// (that is the contract), but a joinable std::thread destroyed anyway calls
    /// std::terminate -- so an early `return` that never reached the teardown
    /// must not turn into a crash on the way out.
    ~Queue() { shutdown(); }

    Queue(const Queue&)            = delete;
    Queue& operator=(const Queue&) = delete;

    /// Hand a batch over.  Blocks (and accounts the block) while the queue is at
    /// either bound -- backpressure is correct and simple; dropping or growing
    /// without limit is neither.
    void submit(Batch&& batch) {
        std::unique_lock<std::mutex> lock(_mtx);
        if (_stopped) {
            // Post-shutdown submit (a late destructor, a serial-branch tail).
            // Nothing is dropped: run it here, on this thread, under the guard.
            lock.unlock();
            replay(batch);
            counters().requests.fetch_add(1, std::memory_order_relaxed);
            counters().bytes.fetch_add(batch.bytes, std::memory_order_relaxed);
            return;
        }
        if (!_started) {
            _started = true;
            _worker  = std::thread([this] { run(); });
        }
        const bool full = _queue.size() >= queueDepthLimit() ||
                          (_bytes + batch.bytes > queueByteLimit() && !_queue.empty());
        if (full) {
            const auto blocked = std::chrono::steady_clock::now();
            _not_full.wait(lock, [this, &batch] {
                return _queue.size() < queueDepthLimit() &&
                       (_bytes + batch.bytes <= queueByteLimit() || _queue.empty());
            });
            counters().block_ns.fetch_add(
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - blocked)
                                               .count()),
                std::memory_order_relaxed);
        }
        _bytes += batch.bytes;
        counters().requests.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(batch.bytes, std::memory_order_relaxed);
        _queue.push_back(std::move(batch));
        bumpMax(counters().max_depth, static_cast<std::uint64_t>(_queue.size()));
        bumpMax(counters().max_bytes, static_cast<std::uint64_t>(_bytes));
        lock.unlock();
        _not_empty.notify_one();
    }

    /// Drain everything queued, then join.  Explicit, called from main() next to
    /// the other teardown steps -- never left to a static destructor.
    void shutdown() {
        std::thread worker;
        {
            std::lock_guard<std::mutex> lock(_mtx);
            if (!_started || _stopped) {
                _stopped = true;
                return;
            }
            _stopped = true;
            worker   = std::move(_worker);
        }
        _not_empty.notify_all();
        if (worker.joinable()) worker.join();
    }

private:
    void run() {
        for (;;) {
            Batch batch;
            {
                std::unique_lock<std::mutex> lock(_mtx);
                _not_empty.wait(lock, [this] { return !_queue.empty() || _stopped; });
                if (_queue.empty()) return;  // stopped AND drained
                batch = std::move(_queue.front());
                _queue.pop_front();
                _bytes -= batch.bytes;
            }
            _not_full.notify_all();
            replay(batch);
        }
    }

    std::mutex              _mtx;
    std::condition_variable _not_empty;
    std::condition_variable _not_full;
    std::deque<Batch>       _queue;
    std::size_t             _bytes   = 0;
    bool                    _started = false;
    bool                    _stopped = false;
    std::thread             _worker;
};

inline Queue& queue() {
    static Queue q;
    return q;
}

/// Wait until every batch submitted for `session` has been replayed.  No-op in
/// inline mode (there is nothing in flight, ever).
inline void fence(const std::shared_ptr<FileSession>& session) {
    if (!session || mode() == Mode::Inline) return;
    std::unique_lock<std::mutex> lock(session->mtx);
    session->cv.wait(lock, [&session] { return session->pending == 0; });
}

// ---------------------------------------------------------------------------
// Payload sizing for the byte bound and the receipt.  Approximate by design --
// it bounds memory, it does not describe the file.
// ---------------------------------------------------------------------------
template <class T>
inline std::size_t payloadBytes(const T&) {
    return sizeof(T);
}
template <class T>
inline std::size_t payloadBytes(const std::vector<T>& v) {
    return v.size() * sizeof(T);
}
inline std::size_t payloadBytes(const std::vector<std::string>& v) {
    std::size_t total = 0;
    for (const std::string& s : v) total += s.size() + 1;
    return total;
}

/// Dataset extents, carried as plain numbers.
///
/// Constructing a HighFive::DataSpace calls H5Screate_simple, so the write path
/// must NOT build one on a Driver thread in thread mode -- that would be an
/// unguarded entry into the runtime, which is the whole thing this file exists
/// to prevent.  Call sites hand over the dims and the DataSpace is built where
/// the dataset is: inline under the recorder's guard, or on the writer thread.
/// A distinct type (not a vector alias) so it can never collide with the
/// value-taking createDataSet overload.
struct Dims {
    std::vector<std::size_t> extent;

    Dims(std::initializer_list<std::size_t> dims) : extent(dims) {}
    explicit Dims(std::vector<std::size_t> dims) : extent(std::move(dims)) {}

    [[nodiscard]] std::size_t count() const {
        std::size_t n = 1;
        for (std::size_t d : extent) n *= d;
        return n;
    }
};

class Recorder;

/// A dataset created with an explicit extent, awaiting its raw block.
///
/// Creation and write_raw are SEPARATE ops on purpose: inline,
/// `createDataSet<T>(name, space)` creates the link immediately whether or not
/// write_raw follows, so the recorded form has to do the same or a caller that
/// never writes would produce a different file.
class RawDataSet {
public:
    RawDataSet(HighFive::DataSet dataset) : _dataset(std::move(dataset)), _live(true) {}
    RawDataSet(Recorder* recorder, int slot, std::size_t count)
        : _recorder(recorder), _slot(slot), _count(count) {}

    template <class T>
    void write_raw(const T* buffer) const;

private:
    // The HighFive handles are `mutable` because HighFive's create/write calls
    // are non-const member functions while a Node/RawDataSet is a HANDLE, not a
    // value: passing one by const reference (writeTermContrib) must still be
    // able to write through it, exactly as `HighFive::Group&` used to.
    mutable HighFive::DataSet _dataset;
    bool                      _live     = false;
    Recorder*                 _recorder = nullptr;
    int                       _slot     = -1;
    std::size_t               _count    = 0;
};

/// Stands in for `HighFive::File&` / `HighFive::Group`.
///
/// One type for both modes so the write path reads the same either way: inline
/// it carries the live handle and every call goes straight through; on the
/// thread path it carries the recorder plus the slot its handle will occupy at
/// replay.  Call sites keep using `auto`, so the substitution is invisible to
/// them.
class Node {
public:
    Node() = default;
    Node(Recorder* recorder, HighFive::File* file) : _recorder(recorder), _file(file) {}
    Node(Recorder* recorder, HighFive::Group group)
        : _recorder(recorder), _group(std::move(group)), _has_group(true) {}
    Node(Recorder* recorder, int slot) : _recorder(recorder), _slot(slot) {}

    Node createGroup(const std::string& name) const;

    template <class T>
    void createDataSet(const std::string& name, const T& value) const;

    template <class T>
    RawDataSet createDataSet(const std::string& name, const Dims& dims) const;

private:
    Recorder*               _recorder  = nullptr;
    HighFive::File*         _file      = nullptr;  ///< inline root
    mutable HighFive::Group _group;                ///< inline group (see RawDataSet)
    bool                    _has_group = false;
    int                     _slot      = -1;       ///< thread mode: -1 == file root
    friend class Recorder;
};

/// Scope that owns one batch.
///
/// Inline mode: holds Chiffon::Hdf5Guard for its whole lifetime and executes
/// everything immediately -- literally the pre-existing `Chiffon::Hdf5Guard
/// hdf5_guard;` at the top of each write function.
/// Thread mode: takes no lock, records, and hands the batch over on submit().
///
/// submit() is EXPLICIT and the destructor discards an unsubmitted batch: the
/// write functions have early returns (and can throw), and a half-recorded
/// batch must not reach the file.
class Recorder {
public:
    explicit Recorder(std::shared_ptr<FileSession> session)
        : _session(std::move(session)), _inline(mode() == Mode::Inline) {
        if (_inline) _guard.emplace();
        _batch.session = _session;
    }

    Recorder(const Recorder&)            = delete;
    Recorder& operator=(const Recorder&) = delete;

    ~Recorder() = default;  // an unsubmitted batch is discarded, by design

    [[nodiscard]] bool isInline() const { return _inline; }

    /// The file root, i.e. what `HighFive::File&` used to be at the call site.
    Node root() {
        if (_inline) return Node(this, _session->file.get());
        return Node(this, -1);
    }

    /// Create (or truncate) the output file.  Same HighFive call in both modes;
    /// only the thread it runs on differs.
    void openOverwrite(const std::string& path) {
        _session->path = path;
        if (_inline) {
            _session->file = std::make_unique<HighFive::File>(path, HighFive::File::Overwrite);
            return;
        }
        push([path](ReplayCtx& ctx) {
            ctx.session.file =
                std::make_unique<HighFive::File>(path, HighFive::File::Overwrite);
        });
    }

    /// Close the file (HighFive::File's destructor is an HDF5 call too).
    void closeFile() {
        if (_inline) {
            _session->file.reset();
            return;
        }
        push([](ReplayCtx& ctx) { ctx.session.file.reset(); });
    }

    void submit() {
        if (_inline || _batch.ops.empty()) {
            _batch.ops.clear();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(_session->mtx);
            ++_session->pending;
        }
        queue().submit(std::move(_batch));
        _batch = Batch{};
        _batch.session = _session;
    }

    // -- recording plumbing, used by Node/RawDataSet --------------------------
    template <class F>
    void push(F&& op, std::size_t bytes = 0) {
        _batch.ops.emplace_back(std::forward<F>(op));
        _batch.bytes += bytes;
    }
    int nextGroupSlot() { return _group_slots++; }
    int nextDataSetSlot() { return _dataset_slots++; }

private:
    std::shared_ptr<FileSession>      _session;
    bool                              _inline = true;
    std::optional<Chiffon::Hdf5Guard> _guard;
    Batch                             _batch;
    int                               _group_slots   = 0;
    int                               _dataset_slots = 0;
};

// ---------------------------------------------------------------------------
// Node / RawDataSet definitions (Recorder must be complete first).
// ---------------------------------------------------------------------------
inline Node Node::createGroup(const std::string& name) const {
    if (_recorder->isInline())
        return Node(_recorder, _has_group ? _group.createGroup(name) : _file->createGroup(name));

    const int slot = _recorder->nextGroupSlot();
    const int parent = _slot;
    _recorder->push([parent, name](ReplayCtx& ctx) {
        ctx.groups.push_back(parent < 0 ? ctx.session.file->createGroup(name)
                                        : ctx.groups[static_cast<std::size_t>(parent)]
                                              .createGroup(name));
    });
    return Node(_recorder, slot);
}

template <class T>
inline void Node::createDataSet(const std::string& name, const T& value) const {
    if (_recorder->isInline()) {
        if (_has_group)
            _group.createDataSet(name, value);
        else
            _file->createDataSet(name, value);
        return;
    }
    const int parent = _slot;
    _recorder->push(
        [parent, name, payload = value](ReplayCtx& ctx) {
            if (parent < 0)
                ctx.session.file->createDataSet(name, payload);
            else
                ctx.groups[static_cast<std::size_t>(parent)].createDataSet(name, payload);
        },
        payloadBytes(value));
}

template <class T>
inline RawDataSet Node::createDataSet(const std::string& name, const Dims& dims) const {
    if (_recorder->isInline()) {
        HighFive::DataSpace space(dims.extent);
        return RawDataSet(_has_group ? _group.createDataSet<T>(name, space)
                                     : _file->createDataSet<T>(name, space));
    }
    const int slot   = _recorder->nextDataSetSlot();
    const int parent = _slot;
    _recorder->push([parent, name, extent = dims.extent](ReplayCtx& ctx) {
        HighFive::DataSpace space(extent);
        ctx.datasets.push_back(
            parent < 0
                ? ctx.session.file->createDataSet<T>(name, space)
                : ctx.groups[static_cast<std::size_t>(parent)].createDataSet<T>(name, space));
    });
    return RawDataSet(_recorder, slot, dims.count());
}

template <class T>
inline void RawDataSet::write_raw(const T* buffer) const {
    if (_live) {
        _dataset.write_raw(buffer);
        return;
    }
    std::vector<T> owned(buffer, buffer + _count);
    const std::size_t bytes = owned.size() * sizeof(T);
    const int         slot  = _slot;
    _recorder->push(
        [slot, payload = std::move(owned)](ReplayCtx& ctx) {
            ctx.datasets[static_cast<std::size_t>(slot)].write_raw(payload.data());
        },
        bytes);
}

// ---------------------------------------------------------------------------
// Statepoint telemetry line sink (plan follow-up, req 6).
//
// SPTELEM emits one JSON line per statepoint per deck -- 35-51 lines x 64 decks
// through a std::cout that is sync_with_stdio(true), i.e. a stdio lock (and
// often a write(2)) per line, on the same threads the writer thread was just
// freed from.  These are not HDF5 and do not need the writer thread; they need
// to stop being one syscall each.
//
// So: the Driver formats its line and appends it here under one small mutex.
// Inline mode flushes every line, which is byte-for-byte and order-for-order
// what `std::cout << std::format(...)` did.  Thread mode accumulates and flushes
// in chunks -- same bytes, same per-job order (SPTELEM lines carry job_id and
// are parsed one line at a time), far fewer stream/stdio round trips.
// ---------------------------------------------------------------------------
inline constexpr std::size_t kLineSinkFlushBytes = 64u * 1024u;

struct LineSink {
    std::mutex  mtx;
    std::string buffer;
};

inline LineSink& lineSink() {
    static LineSink sink;
    return sink;
}

inline void flushLines() {
    LineSink&                   sink = lineSink();
    std::lock_guard<std::mutex> lock(sink.mtx);
    if (sink.buffer.empty()) return;
    std::cout.write(sink.buffer.data(), static_cast<std::streamsize>(sink.buffer.size()));
    sink.buffer.clear();
}

/// Line-atomic append.  `line` must already end in '\n'.  The write happens
/// under the same mutex as the append, so a line can never be split by another
/// thread's flush -- that atomicity is the point, not the buffering.
inline void appendLine(const std::string& line) {
    LineSink&                   sink = lineSink();
    std::lock_guard<std::mutex> lock(sink.mtx);
    sink.buffer.append(line);
    if (mode() != Mode::Inline && sink.buffer.size() < kLineSinkFlushBytes) return;
    std::cout.write(sink.buffer.data(), static_cast<std::streamsize>(sink.buffer.size()));
    sink.buffer.clear();
}

// ---------------------------------------------------------------------------
// Receipts.  Same family/style as [RASBERY][BATCH_HOST] (configuration, before
// the first deck) and [RASBERY][BATCH_HOST][PIN] (final counters, at teardown).
// ---------------------------------------------------------------------------
inline void reportConfig(std::ostream& os) {
    os << "[RASBERY][IO_WRITER] {\"mode\":\"" << modeName() << "\",\"queue_limit\":"
       << queueDepthLimit() << ",\"queue_bytes\":" << queueByteLimit() << "}" << std::endl;
}

inline void reportSummary(std::ostream& os) {
    const Counters& c = counters();
    os << "[RASBERY][IO_WRITER][SUMMARY] {\"mode\":\"" << modeName()
       << "\",\"requests\":" << c.requests.load()
       << ",\"ops\":" << c.ops.load()
       << ",\"bytes\":" << c.bytes.load()
       << ",\"max_queue_depth\":" << c.max_depth.load()
       << ",\"max_queue_bytes\":" << c.max_bytes.load()
       << ",\"enqueue_block_ms\":" << static_cast<double>(c.block_ns.load()) / 1.0e6
       << ",\"writer_busy_ms\":" << static_cast<double>(c.writer_ns.load()) / 1.0e6
       << ",\"failures\":" << c.failures.load() << "}" << std::endl;
}

/// Drain the queue, join the writer, flush the line sink.  Called explicitly
/// from main() alongside the other teardown steps.  Returns the failure count so
/// the caller can turn a lost write into a nonzero exit code.
inline std::uint64_t shutdown() {
    queue().shutdown();
    flushLines();
    return counters().failures.load();
}

} // namespace rasbery::iowriter
