#pragma once

// GpuCaptureArbiter -- the process-wide rule that a batch's graph captures and
// a batch's allocations never overlap.
//
// THE DEFECT THIS CLOSES (Rev.7.1 Task 18d).  `--batch-mode M` starts M host
// Drivers on M threads against ONE arena stream.  The first decks to reach a
// CMFD solve capture their outer/sweep graph on that stream while the later
// decks are still standing up: page-locking their host buffers
// (BICGCMFD.cpp's first-touch `pinHost` block -> cudaHostRegister),
// cudaMalloc'ing their device-outer segment, cudaMallocHost'ing their staging
// lanes, and -- in CudaOuterSegment::bindResidency -- draining the device with
// cudaDeviceSynchronize.
//
// Every one of those is a "potentially unsafe" API in CUDA's stream-capture
// vocabulary.  The captures ask for cudaStreamCaptureModeThreadLocal, which
// only forbids the CAPTURING thread from making such calls; a SIBLING thread
// is not stopped, and its call invalidates the capture instead.  The capturing
// thread then finds its next enqueue answered
//
//   cudaMemcpyAsync(host_status, ...): operation failed due to a previous
//   error during capture                            (cudaErrorStreamCaptureInvalidated)
//
// -- measured at 4acff55 on the 8-deck local batch, 4 runs in 20, 1.8 s in.
//
// WHICH OF THEM ACTUALLY DOES IT, measured rather than assumed.  Widening the
// window with RASBERY_GPU_CAPTURE_STALL_US and reading the [RASBERY][CAPTURE]
// trace separates five runs perfectly: the ONE that survived had 73 sibling
// cudaHostRegister calls inside the window and no cudaDeviceSynchronize; the
// four that died each had exactly one cudaDeviceSynchronize.  The device-wide
// drain is the trigger; the first-touch pins are the accomplice that puts a
// sibling thread on the device API at that moment.  The rule below covers both,
// because the next unsafe call added to a stand-up path will not announce which
// kind it is.
//
// THE RULE.  A capture window is exclusive of every allocation, registration
// and device-wide synchronisation in the process; allocations are free to run
// concurrently with each other.  That is a shared/exclusive lock with the
// polarity inverted from the obvious one: the RARE operation (capture) is the
// writer, the COMMON one (allocation) is the reader, because it is the capture
// that must see a quiet process and not the other way round.
//
// WHAT IT COSTS.  Nothing on the steady path: captures happen a handful of
// times per run (one per graph bucket) and allocations happen at stand-up and
// at teardown.  A batch's rendezvous width is untouched -- the arbiter is not
// in the solve path at all, and `mean_width` is unchanged by construction.
//
// DEADLOCK, AND WHY THERE IS NONE.  std::shared_mutex is not recursive, so the
// two ways to hang are (a) an allocation made from inside a capture window by
// the capturing thread and (b) a capture opened from inside an allocation
// window.  Both are answered by thread-local depth counters rather than by a
// lock: case (a) does not lock (it is already exclusive, and it is ALSO a
// contract violation -- counted and traced), case (b) does not lock either.
// So the arbiter can only ever wait for a DIFFERENT thread.
//
// WP19, AND THE GAP THAT WAS STILL OPEN.  The rule above was enforced at four
// of the five capture sites in this tree.  src/CudaPprBackend.cu's WHILE build
// -- the arm `RASBERY_GPU_PPR=1 RASBERY_GPU_PPR_GRAPH=1` turns on, i.e. the
// production v6 arm -- took NO window at all: that TU did not even include this
// header, and its stand-up (cudaStreamCreate, cudaEventCreate, 25 cudaMalloc,
// 3 cudaMallocHost, and the reconstruction's 23 more) ran unguarded on the
// FIRST statepoint of every deck.  With 16 host Drivers per process the first
// deck to reach its PPR capture and the first deck to reach its PPR stand-up
// are different threads a few milliseconds apart, which is why the death was
// intermittent, non-positional, and always on a lane's FIRST case.
//
// The measured signature: 2-5 of 128 cases dead in ~15.7 s with
//
//   "error":"cudaGetLastError(): operation not permitted when stream is
//    capturing"                                (cudaErrorStreamCaptureUnsupported)
//
// and `"slot":-1` -- no arena slot yet, i.e. stand-up, not solve.
//
// LAYERING.  Header-only and CUDA-free, exactly like HostPinRegistry.h: the
// plain C++ TUs that allocate through the pin hooks must keep compiling in the
// no-CUDA stub build, and the counters must be one process-wide set no matter
// which backend TU is linked.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <sstream>
#include <string>

namespace rasbery {

/// Process-wide counters.  Every one of these is a receipt line term; see
/// captureArbiterReceipt().
struct CaptureArbiterStats {
    std::atomic<unsigned long long> capture_windows{0};
    std::atomic<unsigned long long> capture_wait_us{0};
    std::atomic<unsigned long long> alloc_windows{0};
    std::atomic<unsigned long long> alloc_wait_us{0};
    /// An allocation that actually had to wait for a capture to close.  This
    /// is the number that says the race was real: it is 0 in single mode and
    /// non-zero in a batch.
    std::atomic<unsigned long long> alloc_blocked{0};
    /// An allocation attempted by a thread that has a capture open.  Illegal
    /// by the same CUDA rule; the arbiter cannot serialise it (it IS the
    /// capture) so it names it instead.
    std::atomic<unsigned long long> alloc_in_capture{0};
    /// A capture window left open by an exception rather than by an
    /// EndCapture.  The RAII guard closes it; this counts how often it had to.
    std::atomic<unsigned long long> captures_unwound{0};
    /// Allocations observed while at least one capture was open ANYWHERE --
    /// counted before the arbiter serialises them, so it stays non-zero after
    /// the fix and keeps the receipt honest about the race still existing.
    std::atomic<unsigned long long> alloc_overlapped{0};
    /// WP19.  A graph build that came back with one of the capture-illegal
    /// codes ANYWAY, and was rebuilt once with the arbiter held.  It is the
    /// receipt term that says "the race happened and the case survived it";
    /// 0 is the number a fixed tree prints.
    std::atomic<unsigned long long> capture_race_retry{0};
    /// The retry that ALSO came back capture-illegal.  A case only ever gets
    /// one retry, so this is the count of cases that fell back to the
    /// non-capturing arm rather than dying -- and it must be loud.
    std::atomic<unsigned long long> capture_race_unrecovered{0};
    /// WP19.1.  A capture-illegal refusal that was NOT retried, because the
    /// build had already run the body's host-side enqueue helpers and a second
    /// run of them would elide uploads whose bytes never left the host.  See
    /// noteCaptureRaceAbandoned() -- this is the counter that says "the race
    /// happened, and the honest answer was to stop".
    std::atomic<unsigned long long> capture_race_abandoned{0};
};

inline CaptureArbiterStats& captureArbiterStats() {
    static CaptureArbiterStats s;
    return s;
}

/// IMMORTAL on purpose.  Instance destructors take an AllocWindow (a deck's
/// teardown frees device memory), and some of them run during static
/// destruction; a function-local std::shared_mutex destroyed earlier in that
/// order would make the last teardown undefined.  The stats block below needs
/// no such treatment -- std::atomic has a trivial destructor.
inline std::shared_mutex& captureArbiterMutex() {
    static std::shared_mutex* m = new std::shared_mutex();
    return *m;
}

inline std::atomic<int>& captureArbiterOpen() {
    static std::atomic<int> n{0};
    return n;
}

/// The arbiter's LOCKING is switchable, its accounting is not.
///
/// RASBERY_GPU_CAPTURE_ARBITER=0 keeps every counter and every trace line and
/// removes only the serialisation -- which is exactly the A/B the defect needs:
/// the same binary, the same receipts, with and without the fix.
inline bool captureArbiterEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_CAPTURE_ARBITER");
        return v == nullptr || std::string(v) != "0";
    }();
    return on;
}

inline bool captureTraceEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_CAPTURE_TRACE");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

/// A small dense thread ordinal, because a std::thread::id hash is unreadable
/// in a log and the only question asked of it is "same thread or not".
inline int captureThreadOrdinal() {
    static std::atomic<int>  next{0};
    static thread_local int  id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

inline long long captureNowUs() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

inline void captureTrace(const char* event, const char* tag, const void* stream,
                         long long extra) {
    if (!captureTraceEnabled()) return;
    std::ostringstream line;
    line << "[RASBERY][CAPTURE] {\"ev\":\"" << event << "\",\"tag\":\"" << (tag ? tag : "")
         << "\",\"tid\":" << captureThreadOrdinal() << ",\"stream\":\"" << stream
         << "\",\"open\":" << captureArbiterOpen().load(std::memory_order_relaxed)
         << ",\"us\":" << extra << ",\"t_us\":" << captureNowUs() << '}';
    static std::mutex* m = new std::mutex(); // immortal, as above
    std::lock_guard<std::mutex> g(*m);
    std::cerr << line.str() << std::endl;
}

/// thread-local depths.  Both are "how many windows of this kind does THIS
/// thread have open", which is what makes the two re-entrancy cases decidable
/// without touching the lock.
inline int& captureDepthRef() {
    static thread_local int d = 0;
    return d;
}
inline int& allocDepthRef() {
    static thread_local int d = 0;
    return d;
}

inline bool threadIsCapturing() { return captureDepthRef() > 0; }

// --- WP19: the capture-illegal error class, spelled without CUDA ------------
//
// This header is deliberately CUDA-free (see LAYERING above), so the codes are
// written as numbers.  They are the contiguous cudaError_t block CUDA 11.0
// introduced for stream capture and has not moved since:
//
//   900 cudaErrorStreamCaptureUnsupported  "operation not permitted when
//                                           stream is capturing"   <-- WP19
//   901 cudaErrorStreamCaptureInvalidated  "operation failed due to a previous
//                                           error during capture"
//   902 cudaErrorStreamCaptureMerge
//   903 cudaErrorStreamCaptureUnmatched
//   904 cudaErrorStreamCaptureUnjoined
//   905 cudaErrorStreamCaptureIsolation
//   906 cudaErrorStreamCaptureImplicit
//   907 cudaErrorCapturedEvent
//   908 cudaErrorStreamCaptureWrongThread
//
// Every one of them says "somebody else's capture, or this one, was in the way"
// and NONE of them says "the arithmetic is wrong" -- which is exactly the
// property a retry is allowed to lean on.  static_assert in the CUDA TUs pins
// the numbers to the enum; see CudaPprBackend.cu.
inline constexpr int kCaptureErrFirst = 900;
inline constexpr int kCaptureErrLast  = 908;

/// Is this cudaError_t a capture-concurrency refusal rather than a real fault?
inline bool captureIllegal(int cuda_error) {
    return cuda_error >= kCaptureErrFirst && cuda_error <= kCaptureErrLast;
}

/// One capture-illegal build was retried under the arbiter.  Counted here so
/// the term is process-wide and every capture site reports into one number.
///
/// WP19.1: AND SAID OUT LOUD, WITH THE SITE AND THE STAGE ON IT.
///
/// WP19 spelled this `noteCaptureRaceRetry()` -- no arguments, no line, one
/// process-wide integer printed at teardown between ten others.  The 238
/// evidence run (2026-08-30, 8 x M16 v6) therefore printed
/// `"capture_race_retry":1` in BOTH the process that lost a case and the
/// process that did not, and nothing anywhere said which capture site it was,
/// which stage refused, or which slot was building -- which is exactly the
/// three facts that separate a harmless retry (`ppr.while`, a body of pure
/// kernel launches) from a corrupting one (`outer.while` past record(body)).
/// The dispatcher does not lift the arbiter receipt either, so the term never
/// reached the harness log at all.  One line per event, on stderr, fixes both.
inline void noteCaptureRaceRetry(const char* tag, const char* stage, int cuda_error,
                                 int slot) {
    captureArbiterStats().capture_race_retry.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream line;
    line << "[RASBERY][CUDA][CAPTURE_RACE][RETRY] {\"tag\":\"" << (tag ? tag : "")
         << "\",\"stage\":\"" << (stage ? stage : "") << "\",\"cuda_error\":"
         << cuda_error << ",\"slot\":" << slot
         << ",\"tid\":" << captureThreadOrdinal()
         << ",\"open\":" << captureArbiterOpen().load(std::memory_order_relaxed)
         << ",\"retried\":1}";
    std::cerr << line.str() << std::endl;
}

/// WP19.1.  A capture-illegal refusal that MUST NOT be retried.
///
/// THE DEFECT THIS NAMES.  A graph build that refuses BEFORE its body is
/// recorded has moved nothing: the same build under a quiet process is a clean
/// second experiment, which is what WP19's retry leans on.  A build that
/// refuses AT OR AFTER `record(body)` has already run the body's host-side
/// enqueue helpers, and those helpers keep byte-exact upload shadows
/// (CudaTransferMirror.h) and residency generations that they commit AT THE
/// ISSUE, not at the landing.  The capture that was to carry those copies is
/// then discarded, so the bytes never reach the device -- while the host now
/// believes they did.  A second run of the same helpers ELIDES them, and the
/// graph that gets instantiated has no upload node for data the device does
/// not hold.  On a lane's first case that device memory is uninitialised, and
/// the replay produces a non-finite flux with no CUDA error anywhere.
///
/// So this rung exists to be taken instead: loud, named, counted, and followed
/// by the caller's ordinary "the host state moved" hard stop.
inline void noteCaptureRaceAbandoned(const char* tag, const char* stage, int cuda_error,
                                     int slot, const char* what) {
    captureArbiterStats().capture_race_abandoned.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream line;
    line << "[RASBERY][CUDA][CAPTURE_RACE][ERROR] {\"tag\":\"" << (tag ? tag : "")
         << "\",\"stage\":\"" << (stage ? stage : "") << "\",\"cuda_error\":"
         << cuda_error << ",\"slot\":" << slot << ",\"what\":\""
         << (what ? what : "") << "\",\"retried\":0,\"recovered\":0,"
         << "\"why\":\"host state moved: a second body record would elide uploads "
            "that never left the host\"}";
    std::cerr << line.str() << std::endl;
}

/// WP19.2.  HOW MANY CAPTURE-RACE EVENTS HAS THIS PROCESS SEEN, EVER.
///
/// WHY A SINGLE NUMBER AND NOT A PREDICATE.  The evaluator's belt (WP19.1)
/// asked "is this a lane's first case", because the graph caches it was
/// reasoning about are per-slot and process-lived.  The PPR WHILE's cache is
/// NOT: `s.graph_valid` lives on the PprBackend, which lives on the Driver,
/// which lives for exactly one case -- the PPR_GPU receipts say `graph_builds:1`
/// for EVERY case, not for a lane's first.  So a build window, and with it a
/// capture race, opens on every case, and a first-case-only belt structurally
/// cannot see the ones that do not.  It did not: 238 run3 proc5's
/// candidate_0060 was case 8 on lane 5, preceded by a second `ppr.while`
/// EndCapture(root)/901 retry, and was never retried.
///
/// The honest replacement is a DELTA, not a heuristic: snapshot this counter
/// before a case and after it, and a case that spans a capture-race event is
/// the case that may have been corrupted by one.  It cannot launder a real
/// physics failure, because in a quiet process the delta is zero and the belt
/// never fires.
inline unsigned long long captureRaceEvents() {
    const auto& s = captureArbiterStats();
    return s.capture_race_retry.load(std::memory_order_relaxed) +
           s.capture_race_abandoned.load(std::memory_order_relaxed) +
           s.capture_race_unrecovered.load(std::memory_order_relaxed);
}

/// The arbiter's standing-up provenance, as one compact JSON object body.
///
/// WP19.1.  Printed beside a first-case death so the receipt that says WHAT
/// died sits next to the numbers that say whether a capture window was open
/// while this lane was standing up.  Deliberately the same terms as the
/// teardown receipt, so a reader does not have to learn two vocabularies.
inline std::string captureArbiterProvenance() {
    const auto&        s = captureArbiterStats();
    std::ostringstream line;
    line << "\"capture_windows\":" << s.capture_windows.load() << ','
         << "\"alloc_overlapped\":" << s.alloc_overlapped.load() << ','
         << "\"alloc_blocked\":" << s.alloc_blocked.load() << ','
         << "\"alloc_in_capture\":" << s.alloc_in_capture.load() << ','
         << "\"captures_unwound\":" << s.captures_unwound.load() << ','
         << "\"capture_race_retry\":" << s.capture_race_retry.load() << ','
         << "\"capture_race_abandoned\":" << s.capture_race_abandoned.load() << ','
         << "\"capture_race_unrecovered\":" << s.capture_race_unrecovered.load();
    return line.str();
}

/// The retry lost too.  LOUD, on stderr, with the site that lost -- a silent
/// fallback is what WP19 exists to remove.
inline void noteCaptureRaceUnrecovered(const char* tag, int cuda_error,
                                       const char* what) {
    captureArbiterStats().capture_race_unrecovered.fetch_add(1,
                                                             std::memory_order_relaxed);
    std::ostringstream line;
    line << "[RASBERY][CUDA][CAPTURE_RACE][ERROR] {\"tag\":\"" << (tag ? tag : "")
         << "\",\"cuda_error\":" << cuda_error << ",\"what\":\""
         << (what ? what : "") << "\",\"retried\":1,\"recovered\":0}";
    std::cerr << line.str() << std::endl;
}

/// Exclusive window: while this is alive, no thread may enter an AllocWindow.
class CaptureWindow {
  public:
    CaptureWindow(const void* stream, const char* tag) : _tag(tag), _stream(stream) {
        const bool nested   = captureDepthRef()++ > 0;
        // Opening a capture from inside an allocation window would be a
        // self-upgrade on a non-recursive lock.  It does not happen in this
        // tree (captures are taken from solve paths, allocations from stand-up
        // and teardown), and if it ever does, the arbiter declines rather than
        // hangs.
        const bool in_alloc = allocDepthRef() > 0;
        if (nested || in_alloc || !captureArbiterEnabled()) {
            captureArbiterOpen().fetch_add(1, std::memory_order_relaxed);
            captureArbiterStats().capture_windows.fetch_add(1, std::memory_order_relaxed);
            captureTrace("begin", _tag, _stream, 0);
            return;
        }
        _owned              = true;
        const long long t0  = captureNowUs();
        captureArbiterMutex().lock();
        const long long dt  = captureNowUs() - t0;
        captureArbiterStats().capture_wait_us.fetch_add(
            static_cast<unsigned long long>(dt < 0 ? 0 : dt), std::memory_order_relaxed);
        captureArbiterOpen().fetch_add(1, std::memory_order_relaxed);
        captureArbiterStats().capture_windows.fetch_add(1, std::memory_order_relaxed);
        captureTrace("begin", _tag, _stream, dt);
    }

    ~CaptureWindow() {
        --captureDepthRef();
        captureArbiterOpen().fetch_sub(1, std::memory_order_relaxed);
        captureTrace("end", _tag, _stream, 0);
        if (_owned) captureArbiterMutex().unlock();
    }

    CaptureWindow(const CaptureWindow&)            = delete;
    CaptureWindow& operator=(const CaptureWindow&) = delete;

  private:
    const char* _tag    = nullptr;
    const void* _stream = nullptr;
    bool        _owned  = false;
};

/// Shared window: many may be open at once, none may overlap a CaptureWindow.
///
/// Wrap every cudaMalloc / cudaFree / cudaMallocHost / cudaFreeHost /
/// cudaHostAlloc / cudaHostRegister / cudaHostUnregister / cudaDeviceSynchronize
/// that a batch can reach while another deck is capturing -- which, with M host
/// threads and one arena, is all of them.
class AllocWindow {
  public:
    explicit AllocWindow(const char* tag) : _tag(tag) {
        auto& st = captureArbiterStats();
        st.alloc_windows.fetch_add(1, std::memory_order_relaxed);
        if (captureArbiterOpen().load(std::memory_order_relaxed) > 0)
            st.alloc_overlapped.fetch_add(1, std::memory_order_relaxed);
        if (threadIsCapturing()) {
            // The one case the arbiter cannot fix by waiting: this thread IS
            // the capture.  Name it; the contract test forbids the source
            // shape that produces it.
            st.alloc_in_capture.fetch_add(1, std::memory_order_relaxed);
            captureTrace("alloc-in-capture", _tag, nullptr, 0);
            return;
        }
        if (!captureArbiterEnabled()) {
            captureTrace("alloc", _tag, nullptr, -1);
            return;
        }
        if (allocDepthRef()++ > 0) { _state = kNested; return; }
        _state             = kOwned;
        const long long t0 = captureNowUs();
        captureArbiterMutex().lock_shared();
        const long long dt = captureNowUs() - t0;
        if (dt > 0) {
            st.alloc_wait_us.fetch_add(static_cast<unsigned long long>(dt),
                                       std::memory_order_relaxed);
            if (dt >= 50) st.alloc_blocked.fetch_add(1, std::memory_order_relaxed);
        }
        captureTrace("alloc", _tag, nullptr, dt);
    }

    ~AllocWindow() {
        if (_state == kNone) return;
        --allocDepthRef();
        if (_state == kOwned) captureArbiterMutex().unlock_shared();
    }

    AllocWindow(const AllocWindow&)            = delete;
    AllocWindow& operator=(const AllocWindow&) = delete;

  private:
    enum State { kNone, kOwned, kNested };
    const char* _tag   = nullptr;
    State       _state = kNone;
};

/// One line, printed beside BATCH_OCCUPANCY, that answers "did the arbiter have
/// anything to do".
inline std::string captureArbiterReceipt(const char* tag) {
    const auto&        s = captureArbiterStats();
    std::ostringstream line;
    line << "[RASBERY][CUDA][CAPTURE_ARBITER] {\"tag\":\"" << (tag ? tag : "") << "\","
         << "\"enabled\":" << (captureArbiterEnabled() ? 1 : 0) << ','
         << "\"capture_windows\":" << s.capture_windows.load() << ','
         << "\"capture_wait_us\":" << s.capture_wait_us.load() << ','
         << "\"alloc_windows\":" << s.alloc_windows.load() << ','
         << "\"alloc_overlapped\":" << s.alloc_overlapped.load() << ','
         << "\"alloc_blocked\":" << s.alloc_blocked.load() << ','
         << "\"alloc_wait_us\":" << s.alloc_wait_us.load() << ','
         << "\"alloc_in_capture\":" << s.alloc_in_capture.load() << ','
         << "\"captures_unwound\":" << s.captures_unwound.load() << ','
         << "\"capture_race_retry\":" << s.capture_race_retry.load() << ','
         << "\"capture_race_abandoned\":" << s.capture_race_abandoned.load() << ','
         << "\"capture_race_unrecovered\":" << s.capture_race_unrecovered.load() << '}';
    return line.str();
}

} // namespace rasbery
