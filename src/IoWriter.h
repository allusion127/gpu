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
// The bit-golden gates (single-deck 500/500 datasets, batch 45,312/45,312) are
// the acceptance test for that claim; they passed in every configuration, which
// is why `thread` is the default as of 2026-08-27.  RASBERY_IO_WRITER=inline
// still keeps the pre-writer-thread path literally unchanged, for a bisect or
// an A/B arm that needs it.
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
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Hdf5Guard.h"
#include "highfive/highfive.hpp"

namespace rasbery::iowriter {

// ---------------------------------------------------------------------------
// The env gate.  Two modes, and a provenance for the receipt.
//
// ADOPTION (2026-08-27).  `thread` is now the DEFAULT for every mode of
// execution.  The 238 validation ran the writer thread byte-identical in every
// configuration it was tried in -- single deck 500/500 datasets, M64
// 45,312/45,312 datasets, restart snapshots included -- and measured +0.6 % on
// M64 throughput, so `inline` stopped being the safe choice and became the
// legacy one.  It is still reachable, because a bisect, an A/B and a
// "is the writer thread doing this?" question all need it:
//
//   RASBERY_IO_WRITER unset      -> thread, source=default   (production)
//   RASBERY_IO_WRITER=thread     -> thread, source=env       (explicit)
//   RASBERY_IO_WRITER=inline     -> inline, source=env       (legacy path)
//   RASBERY_IO_WRITER=<garbage>  -> thread, source=default + a warning
//
// An empty value is not a request; it reads as unset, the same rule the other
// RASBERY gates use.  A typo resolves to the DEFAULT rather than to inline: the
// operator asked for something this build does not have, and the answer to that
// is the path the goldens were frozen on, said out loud.
//
// Read ONCE into a function-local static so the recorder, the queue and the
// receipt can never disagree about which mode a run is in.
// ---------------------------------------------------------------------------
enum class Mode { Inline, Thread };

/// Where the mode came from.  Published next to the mode so `thread(default)`,
/// `thread(env)` and `inline(env)` are three distinguishable runs in a log --
/// without it, an adopted default and a deliberate A/B arm look identical.
enum class ModeSource { Default, Env };

struct Resolution {
    Mode       mode;
    ModeSource source;
};

inline const Resolution& resolution() {
    static const Resolution resolved = [] {
        const char* value = std::getenv("RASBERY_IO_WRITER");
        if (value == nullptr || *value == '\0')
            return Resolution{Mode::Thread, ModeSource::Default};
        std::string requested(value);
        std::transform(requested.begin(), requested.end(), requested.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (requested == "thread") return Resolution{Mode::Thread, ModeSource::Env};
        if (requested == "inline") return Resolution{Mode::Inline, ModeSource::Env};
        std::cerr << "[RASBERY][WARN][IO_WRITER] unknown RASBERY_IO_WRITER=" << value
                  << " -- using thread (the default).\n";
        return Resolution{Mode::Thread, ModeSource::Default};
    }();
    return resolved;
}

inline Mode       mode() { return resolution().mode; }
inline ModeSource modeSource() { return resolution().source; }

inline const char* modeName() { return mode() == Mode::Thread ? "thread" : "inline"; }

/// "default" or "env" -- the second half of the receipt's mode field.
inline const char* modeSourceName() {
    return modeSource() == ModeSource::Env ? "env" : "default";
}

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
    /// Batches dropped because their session was ALREADY failed.  Separate from
    /// `failures` on purpose: one failure plus N skips says "this job's output
    /// is gone from here on", which one failure alone would not.
    std::atomic<std::uint64_t> skipped{0};
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

/// Bounds-checked slot lookup.  A recorded op names its parent by slot, and the
/// slot is only correct if creates and replays stayed in lockstep.  If they ever
/// did not, this must be a thrown error -- which poisons the session and fails
/// the job -- and never an out-of-range read on the writer thread.
inline HighFive::Group& groupAt(ReplayCtx& ctx, int slot) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= ctx.groups.size())
        throw std::runtime_error("IO writer: group slot " + std::to_string(slot) +
                                 " out of range (" + std::to_string(ctx.groups.size()) + ")");
    return ctx.groups[static_cast<std::size_t>(slot)];
}

inline HighFive::DataSet& dataSetAt(ReplayCtx& ctx, int slot) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= ctx.datasets.size())
        throw std::runtime_error("IO writer: dataset slot " + std::to_string(slot) +
                                 " out of range (" + std::to_string(ctx.datasets.size()) + ")");
    return ctx.datasets[static_cast<std::size_t>(slot)];
}

/// The open file every non-opening op dereferences.  Never a raw `->` on the
/// unique_ptr: a batch whose file failed to open would otherwise walk a null
/// pointer on the writer thread and take all 64 decks down with it.
inline HighFive::File& fileOf(ReplayCtx& ctx) {
    if (!ctx.session.file)
        throw std::runtime_error("IO writer: '" + ctx.session.path + "' is not open");
    return *ctx.session.file;
}

struct Batch {
    std::shared_ptr<FileSession> session;
    std::vector<Op>              ops;
    std::size_t                  bytes = 0;
    /// True for the batch that carries the file's open op.  Every other batch
    /// requires the file to be open already, and says so before it touches it.
    bool                         opens_file = false;
};

/// Mark a session failed, publish it against its job id, and count it.  ONE
/// emitter of the FAIL receipt, so the paths that can fail cannot drift apart.
inline void poison(FileSession& session, const std::string& what) {
    counters().failures.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(session.mtx);
        session.failed = true;
        if (session.error.empty()) session.error = what;
    }
    std::cout << "[RASBERY][IO_WRITER][FAIL] {\"job\":\"" << session.job << "\",\"path\":\""
              << session.path << "\",\"what\":\"" << what << "\"}" << std::endl;
}

/// Execute one batch's ops against its file.  The ONLY place HDF5 write calls
/// happen on the thread path; `ctx` is declared inside the guard scope because
/// Group/DataSet destructors re-enter the runtime too.
inline void replay(Batch& batch) {
    const auto started = std::chrono::steady_clock::now();

    // A POISONED SESSION IS ABSORBING.  Once a file's open -- or any earlier
    // batch for it -- has failed, every later batch is SKIPPED rather than
    // replayed: its ops would dereference a file that was never opened, and a
    // segfault on the writer thread takes all 64 decks down instead of one.
    // This is a skip, not a swallow: the failure was published when it
    // happened, the receipt counts the skips, and CloseResult()'s fence still
    // turns it into THAT job's exception.
    bool poisoned = false;
    {
        std::lock_guard<std::mutex> lock(batch.session->mtx);
        poisoned = batch.session->failed;
    }

    if (poisoned) {
        counters().skipped.fetch_add(1, std::memory_order_relaxed);
        batch.ops.clear();
    } else {
        try {
            Chiffon::Hdf5Guard hdf5_guard;
            ReplayCtx          ctx{*batch.session, {}, {}};
            // The file must already be open unless THIS batch is the one that
            // opens it.  Checked before any op runs, so even a poisoning this
            // thread somehow missed cannot reach a null dereference.
            if (!batch.opens_file && !ctx.session.file)
                throw std::runtime_error("IO writer: '" + ctx.session.path +
                                         "' is not open (its create must have failed)");
            for (Op& op : batch.ops) op(ctx);
        } catch (const std::exception& error) {
            poison(*batch.session, error.what());
        } catch (...) {
            poison(*batch.session, "unknown exception");
        }
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

    /// LAST RESORT, AND IT DOES NO I/O.  main() drains this explicitly in both
    /// branches; that is the contract and the only supported path.  This runs
    /// only when an early `return` skipped the teardown, and by then we are in
    /// static destruction: replaying here would enter HDF5, std::cout and the
    /// counters at a point where none of them is guaranteed to still exist.
    /// So it ABANDONS whatever is queued, says so on stderr, and joins -- a
    /// joinable std::thread destroyed instead would call std::terminate.
    ~Queue() {
        std::thread worker;
        std::deque<Batch> abandoned;
        {
            std::lock_guard<std::mutex> lock(_mtx);
            _stopped = true;
            abandoned.swap(_queue);
            _bytes   = 0;
            worker   = std::move(_worker);
            _started = false;
        }
        _not_empty.notify_all();
        _not_full.notify_all();
        if (worker.joinable()) worker.join();

        if (abandoned.empty()) return;
        // Release anyone waiting in fence() and record the loss on the sessions
        // (they are heap objects held by the batches, so they are still alive).
        for (Batch& batch : abandoned) {
            if (!batch.session) continue;
            {
                std::lock_guard<std::mutex> lock(batch.session->mtx);
                batch.session->failed = true;
                if (batch.session->error.empty())
                    batch.session->error = "writer queue torn down before this batch ran";
                if (batch.session->pending > 0) --batch.session->pending;
            }
            batch.session->cv.notify_all();
        }
        std::fprintf(stderr,
                     "[RASBERY][IO_WRITER][FAIL] {\"what\":\"writer torn down with %zu batch(es) "
                     "undrained -- iowriter::shutdown() was not called\"}\n",
                     abandoned.size());
    }

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
            replayHere(batch);
            return;
        }
        if (!_started) {
            // Construct FIRST, then latch.  A throwing std::thread ctor with the
            // flag already set would leave a queue nobody ever services, and
            // every fence() on it would hang forever.
            _worker  = std::thread([this] { run(); });
            _started = true;
        }
        const bool full = _queue.size() >= queueDepthLimit() ||
                          (_bytes + batch.bytes > queueByteLimit() && !_queue.empty());
        if (full) {
            const auto blocked = std::chrono::steady_clock::now();
            // `|| _stopped` so a teardown while we are blocked wakes us instead
            // of leaving us parked on a queue that will never drain again.
            _not_full.wait(lock, [this, &batch] {
                return _stopped ||
                       (_queue.size() < queueDepthLimit() &&
                        (_bytes + batch.bytes <= queueByteLimit() || _queue.empty()));
            });
            counters().block_ns.fetch_add(
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - blocked)
                                               .count()),
                std::memory_order_relaxed);
            if (_stopped) {
                lock.unlock();
                replayHere(batch);
                return;
            }
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
            _started = false;
        }
        _not_empty.notify_all();
        _not_full.notify_all();
        if (worker.joinable()) worker.join();
    }

private:
    /// Run a batch on the CALLING thread, with the same accounting the writer
    /// would have done.  Used only when the queue is already stopped.
    static void replayHere(Batch& batch) {
        const std::size_t bytes = batch.bytes;
        replay(batch);
        counters().requests.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

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
        _batch.opens_file = true;
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
            resetBatch();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(_session->mtx);
            ++_session->pending;
        }
        try {
            queue().submit(std::move(_batch));
        } catch (const std::exception& error) {
            // The hand-off itself failed (a std::thread that could not start is
            // the realistic case).  `pending` is already counted, so undo it
            // here or every fence() on this session waits for a batch that will
            // never run; then fail the job loudly rather than silently.
            {
                std::lock_guard<std::mutex> lock(_session->mtx);
                if (_session->pending > 0) --_session->pending;
            }
            _session->cv.notify_all();
            resetBatch();
            poison(*_session, std::string("writer hand-off failed: ") + error.what());
            throw;
        }
        resetBatch();
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
    /// Start a fresh batch.  The SLOT COUNTERS RESET WITH IT: slots index the
    /// replay context, which is per batch, so a recorder reused across two
    /// submits would otherwise hand the second batch indices that only existed
    /// in the first.
    void resetBatch() {
        _batch         = Batch{};
        _batch.session = _session;
        _group_slots   = 0;
        _dataset_slots = 0;
    }

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
        ctx.groups.push_back(parent < 0 ? fileOf(ctx).createGroup(name)
                                        : groupAt(ctx, parent).createGroup(name));
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
                fileOf(ctx).createDataSet(name, payload);
            else
                groupAt(ctx, parent).createDataSet(name, payload);
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
        ctx.datasets.push_back(parent < 0 ? fileOf(ctx).createDataSet<T>(name, space)
                                          : groupAt(ctx, parent).createDataSet<T>(name, space));
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
            dataSetAt(ctx, slot).write_raw(payload.data());
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
    os << "[RASBERY][IO_WRITER] {\"mode\":\"" << modeName() << "\",\"mode_source\":\""
       << modeSourceName() << "\",\"queue_limit\":" << queueDepthLimit()
       << ",\"queue_bytes\":" << queueByteLimit() << "}" << std::endl;
}

inline void reportSummary(std::ostream& os) {
    const Counters& c = counters();
    os << "[RASBERY][IO_WRITER][SUMMARY] {\"mode\":\"" << modeName()
       << "\",\"mode_source\":\"" << modeSourceName()
       << "\",\"requests\":" << c.requests.load()
       << ",\"ops\":" << c.ops.load()
       << ",\"bytes\":" << c.bytes.load()
       << ",\"max_queue_depth\":" << c.max_depth.load()
       << ",\"max_queue_bytes\":" << c.max_bytes.load()
       << ",\"enqueue_block_ms\":" << static_cast<double>(c.block_ns.load()) / 1.0e6
       << ",\"writer_busy_ms\":" << static_cast<double>(c.writer_ns.load()) / 1.0e6
       << ",\"failures\":" << c.failures.load()
       << ",\"skipped\":" << c.skipped.load() << "}" << std::endl;
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
