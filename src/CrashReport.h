#pragma once

// CrashReport -- WP19.2: a SIGSEGV that says which case, which lane, which
// slot and which phase, on a host where no core can be written.
//
// THE DEFECT THIS CLOSES.  The 238 block-38 campaign (0054838, 8 x M16 + MPS,
// 128-job manifest, six runs) took two worker SIGSEGVs -- run2 proc 7 and run3
// proc 1.  Both were recovered by the dispatcher's restart/requeue, and both
// were forensically EMPTY:
//
//   {"gpu":"0","proc":7,"chunk":1,"attempt":1,"wave_id":101,"returncode":-11,
//    "completed":0,"unfinished":[...16...],"requeued":true,"restarts":0}
//
// `completed:0` is the whole record.  The child had emitted no [EVALUATOR][CASE]
// receipt yet, so nothing upstream of the FATAL line named a deck; `ulimit -c`
// is 0 on that host, /proc/sys/kernel/core_pattern routes to systemd-coredump,
// and `coredumpctl` is not readable by the run account -- so there was no core
// either.  Two crashes, zero evidence, and the block's own write-up had to
// close with "unexplained ... out of scope".  That is the gap: not the crash,
// the SILENCE around it.
//
// WHAT IT PRINTS, AND WHY IT IS ENOUGH.  One line per live lane plus one frame
// list, on stderr -- which the dispatcher already merges into the worker log
// (subprocess.STDOUT), so the record reaches the harness with no new plumbing:
//
//   [RASBERY][CRASH] {"signal":11,"name":"SIGSEGV","pid":41213,"tid":6,
//    "capture_open":1,"thread_capturing":0,"capture_race_events":2,
//    "lanes":[{"lane":5,"case":8,"slot":-1,"phase":"drive",
//              "deck":".../candidate_0060.json"}]}
//   [RASBERY][CRASH][FRAME] ...backtrace_symbols_fd output...
//   [RASBERY][CRASH][END] {"signal":11,"frames":23}
//
// The four facts the block-38 forensics wanted and could not get -- case, lane,
// slot, phase -- are the first four fields, and `capture_open` /
// `capture_race_events` answer the standing question of whether a capture
// window was open somewhere in the process when the thread died.
//
// ASYNC-SIGNAL SAFETY, KEPT RATHER THAN CLAIMED.  Inside the handler this file
// calls exactly write(2), backtrace(), backtrace_symbols_fd(), sigaction(),
// getpid(), and raise() -- every one of them on the POSIX
// async-signal-safe list, except backtrace()/backtrace_symbols_fd(), which
// glibc documents as the malloc-free spelling of the pair and which are the
// only way to get a frame list without a core.  No std::ostream, no
// std::string, no snprintf, no locale.  Integers are rendered by a local
// reverse-digit loop into a stack buffer.  `backtrace()` is called ONCE at
// install time so the dynamic loader has already resolved it and populated its
// internal state before a signal can arrive.
//
// THE BREADCRUMBS ARE LOCK-FREE AND LANE-LOCAL.  `Breadcrumb` is a fixed table
// indexed by host thread ordinal; a lane writes only its own row and only with
// relaxed atomics, so the steady path costs a handful of stores per case and
// the handler can read every row without taking anything.  A row is published
// last-field-first (`active` is the last store to go true and the first to go
// false) so a half-written row reads as inactive rather than as a lie.
//
// RE-RAISE, NOT EXIT.  The handler restores the default disposition and
// re-raises, so the process still dies with signal 11: the dispatcher's
// `returncode:-11` stays exactly what it was, WCOREDUMP semantics are
// unchanged, and a host that CAN write a core still writes one.  This adds
// evidence; it changes no outcome.
//
// LAYERING.  Header-only, CUDA-free, POSIX-optional: on a platform without
// <csignal>/<execinfo.h> every entry point compiles to a no-op so the MSVC stub
// build keeps building.

#include <atomic>
#include <cstddef>
#include <cstdlib>

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    #define RASBERY_HAS_CRASH_REPORT 1
#else
    #define RASBERY_HAS_CRASH_REPORT 0
#endif

#if RASBERY_HAS_CRASH_REPORT
    #include <csignal>
    #include <cstring>
    #include <unistd.h>
    #if defined(__has_include)
        #if __has_include(<execinfo.h>)
            #define RASBERY_HAS_BACKTRACE 1
        #else
            #define RASBERY_HAS_BACKTRACE 0
        #endif
    #else
        #define RASBERY_HAS_BACKTRACE 0
    #endif
    #if RASBERY_HAS_BACKTRACE
        #include <execinfo.h>
    #endif
#else
    #define RASBERY_HAS_BACKTRACE 0
#endif

#include "GpuCaptureArbiter.h"

namespace rasbery {
namespace crash {

/// The widest lane table this build can describe.  A worker runs one host
/// thread per lane and the dispatcher's widest declared shape is 64; 128 is a
/// power of two above that with room to spare, and the table is static so the
/// cost is bytes, not allocations.
inline constexpr int kMaxLanes = 128;
/// A deck path is copied, not pointed at, because the std::string that owns it
/// can be destroyed while the handler is reading.  260 is longer than every
/// manifest path this tree has produced; a longer one is truncated, not
/// refused.
inline constexpr int kDeckChars = 260;

/// One lane's row.  Written by that lane alone; read by the handler, on
/// whichever thread died.
struct Breadcrumb {
    std::atomic<int>  active{0};
    std::atomic<int>  case_index{-1};
    std::atomic<int>  slot{-1};
    /// A string LITERAL only -- the phase names below are all static storage,
    /// so the pointer stays valid for the life of the process and the handler
    /// can print it without copying.
    std::atomic<const char*> phase{nullptr};
    /// Written character by character; `deck_len` is published after the bytes,
    /// so a reader that sees a length sees bytes that are already there.
    char                     deck[kDeckChars]{};
    std::atomic<int>         deck_len{0};
};

inline Breadcrumb* breadcrumbs() {
    static Breadcrumb table[kMaxLanes];
    return table;
}

/// Phase names.  Deliberately literals in one place: the handler prints the
/// pointer, and a phase spelled at its call site would be a pointer into a
/// temporary.
inline constexpr const char* kPhaseAdmit    = "admit";
inline constexpr const char* kPhaseDrive    = "drive";
inline constexpr const char* kPhaseTeardown = "teardown";
inline constexpr const char* kPhaseRetry    = "capture_race_retry";
inline constexpr const char* kPhaseReport   = "report";

/// WHICH ROW IS THIS THREAD'S.  Set by Scope, read by the deep code that knows
/// a fact the evaluator does not (the arena slot) and has no business learning
/// the OpenMP thread number to say so.  -1 until a case opens on this thread.
inline int& currentLane() {
    static thread_local int lane = -1;
    return lane;
}

/// Open a lane's row.  `lane` outside the table is dropped rather than folded,
/// because folding two lanes onto one row would make the crash record lie.
inline void enter(int lane, int case_index, const char* deck, const char* phase) {
    currentLane() = lane;
    if (lane < 0 || lane >= kMaxLanes) return;
    Breadcrumb& b = breadcrumbs()[lane];
    b.active.store(0, std::memory_order_relaxed);
    b.case_index.store(case_index, std::memory_order_relaxed);
    b.phase.store(phase, std::memory_order_relaxed);
    int n = 0;
    if (deck != nullptr)
        for (; n < kDeckChars - 1 && deck[n] != '\0'; ++n) b.deck[n] = deck[n];
    b.deck[n] = '\0';
    b.deck_len.store(n, std::memory_order_release);
    b.active.store(1, std::memory_order_release);
}

/// Move a lane between phases without disturbing the rest of the row.
inline void phase(int lane, const char* name) {
    if (lane < 0 || lane >= kMaxLanes) return;
    breadcrumbs()[lane].phase.store(name, std::memory_order_relaxed);
}

/// The arena slot this lane got, once something knows it.  The evaluator does
/// not (its [ERROR] receipts print `slot:-1` for exactly this reason); the
/// outer segment does, and calls this at initialize().
inline void noteSlot(int lane, int slot) {
    if (lane < 0 || lane >= kMaxLanes) return;
    breadcrumbs()[lane].slot.store(slot, std::memory_order_relaxed);
}

/// Same, for a caller that knows the slot and not the lane -- which is every
/// backend.  A call from a thread with no open case is a no-op.
inline void noteSlot(int slot) { noteSlot(currentLane(), slot); }

/// Close a lane's row.  `active` falls FIRST so a handler can never read a row
/// whose case is already gone.
inline void leave(int lane) {
    if (lane < 0 || lane >= kMaxLanes) return;
    Breadcrumb& b = breadcrumbs()[lane];
    b.active.store(0, std::memory_order_release);
    b.case_index.store(-1, std::memory_order_relaxed);
    b.slot.store(-1, std::memory_order_relaxed);
    b.phase.store(nullptr, std::memory_order_relaxed);
}

/// RAII around enter()/leave(), so an exception out of a case -- which is how
/// a fail-closed refusal leaves runOneCase -- cannot leave a stale row behind.
class Scope {
  public:
    Scope(int lane, int case_index, const char* deck, const char* phase_name)
        : _lane(lane) {
        enter(_lane, case_index, deck, phase_name);
    }
    ~Scope() { leave(_lane); }
    void phase(const char* name) const { crash::phase(_lane, name); }
    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

  private:
    int _lane;
};

#if RASBERY_HAS_CRASH_REPORT

namespace detail {

/// write(2) with the short-write loop the man page requires.  `ssize_t` and no
/// error reporting: inside a handler there is nowhere to report to.
inline void rawWrite(const char* text, std::size_t n) {
    while (n > 0) {
        const ssize_t wrote = ::write(2, text, n);
        if (wrote <= 0) return;
        text += wrote;
        n -= static_cast<std::size_t>(wrote);
    }
}

inline void rawStr(const char* text) {
    if (text == nullptr) { rawWrite("null", 4); return; }
    std::size_t n = 0;
    while (text[n] != '\0') ++n;
    rawWrite(text, n);
}

/// Signed decimal, rendered without snprintf.  22 characters is longer than
/// any 64-bit value plus its sign and terminator.
inline void rawInt(long long value) {
    char        buf[24];
    int         i        = 0;
    const bool  negative = value < 0;
    unsigned long long v =
        negative ? (0ull - static_cast<unsigned long long>(value))
                 : static_cast<unsigned long long>(value);
    if (v == 0) buf[i++] = '0';
    while (v > 0 && i < 20) { buf[i++] = static_cast<char>('0' + (v % 10)); v /= 10; }
    if (negative && i < 23) buf[i++] = '-';
    char out[24];
    int  n = 0;
    while (i > 0) out[n++] = buf[--i];
    rawWrite(out, static_cast<std::size_t>(n));
}

/// A JSON string body with the two escapes a filesystem path can contain.
/// Everything else is passed through: this is a crash record, not a parser
/// conformance test, and a path with a control character in it is a bigger
/// problem than its quoting.
inline void rawQuoted(const char* text, int len) {
    rawWrite("\"", 1);
    for (int i = 0; i < len && text[i] != '\0'; ++i) {
        if (text[i] == '"' || text[i] == '\\') rawWrite("\\", 1);
        rawWrite(text + i, 1);
    }
    rawWrite("\"", 1);
}

inline const char* signalName(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGBUS:  return "SIGBUS";
        case SIGFPE:  return "SIGFPE";
        case SIGILL:  return "SIGILL";
        case SIGABRT: return "SIGABRT";
        default:      return "?";
    }
}

/// Set once, so two threads faulting together do not interleave two reports.
/// A second thread that loses the race re-raises immediately: one complete
/// record is worth more than two shredded ones.
inline std::atomic<int>& reporting() {
    static std::atomic<int> flag{0};
    return flag;
}

inline void reraise(int sig) {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_DFL;
    ::sigaction(sig, &sa, nullptr);
    ::raise(sig);
}

/// NOT `extern "C"`.  A C-linkage function inside a namespace still gets the
/// unqualified symbol `handler` at link time, which is a collision waiting for
/// the next TU that wants that name; sigaction takes a plain `void(int)` and
/// every platform this builds on accepts a C++ one.
inline void handler(int sig) {
    int expected = 0;
    if (!reporting().compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        reraise(sig);
        return;
    }
    rawStr("[RASBERY][CRASH] {\"signal\":");
    rawInt(sig);
    rawStr(",\"name\":\"");
    rawStr(signalName(sig));
    rawStr("\",\"pid\":");
    rawInt(static_cast<long long>(::getpid()));
    rawStr(",\"tid\":");
    rawInt(captureThreadOrdinal());
    // The three numbers that say whether a capture was in flight anywhere in
    // the process when this thread died -- the standing question WP19 left and
    // block 38 could not answer for either SIGSEGV.
    rawStr(",\"capture_open\":");
    rawInt(static_cast<long long>(captureArbiterOpen().load(std::memory_order_relaxed)));
    rawStr(",\"thread_capturing\":");
    rawInt(threadIsCapturing() ? 1 : 0);
    rawStr(",\"capture_race_events\":");
    rawInt(static_cast<long long>(captureRaceEvents()));
    rawStr(",\"lanes\":[");
    const Breadcrumb* table = breadcrumbs();
    int               shown = 0;
    for (int lane = 0; lane < kMaxLanes; ++lane) {
        if (table[lane].active.load(std::memory_order_acquire) == 0) continue;
        if (shown++ > 0) rawWrite(",", 1);
        rawStr("{\"lane\":");
        rawInt(lane);
        rawStr(",\"case\":");
        rawInt(table[lane].case_index.load(std::memory_order_relaxed));
        rawStr(",\"slot\":");
        rawInt(table[lane].slot.load(std::memory_order_relaxed));
        rawStr(",\"phase\":\"");
        rawStr(table[lane].phase.load(std::memory_order_relaxed));
        rawStr("\",\"deck\":");
        rawQuoted(table[lane].deck,
                  table[lane].deck_len.load(std::memory_order_acquire));
        rawWrite("}", 1);
    }
    rawStr("]}\n");

    int frames = 0;
#if RASBERY_HAS_BACKTRACE
    // 64 frames: deep enough for the solve stack (evaluator -> Driver -> CMFD
    // -> backend -> CUDA) and small enough to sit on the signal stack.
    void* stack[64];
    frames = ::backtrace(stack, 64);
    rawStr("[RASBERY][CRASH][FRAME] begin\n");
    ::backtrace_symbols_fd(stack, frames, 2);
#else
    rawStr("[RASBERY][CRASH][FRAME] unavailable: no <execinfo.h> in this build\n");
#endif
    rawStr("[RASBERY][CRASH][END] {\"signal\":");
    rawInt(sig);
    rawStr(",\"frames\":");
    rawInt(frames);
    rawStr("}\n");
    reraise(sig);
}

} // namespace detail

/// Install the handler for the fatal signals a solver can actually take.
///
/// IDEMPOTENT, because the evaluator's entry is not the only thing that may
/// want it and installing twice must not chain two handlers.
///
/// RASBERY_CRASH_REPORT=0 removes it, which is the A/B a host with working
/// cores wants: the same binary, the same run, with and without the report.
inline void install() {
    static std::atomic<int> installed{0};
    int                     expected = 0;
    if (!installed.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) return;
    const char* off = std::getenv("RASBERY_CRASH_REPORT");
    if (off != nullptr && off[0] == '0' && off[1] == '\0') return;
#if RASBERY_HAS_BACKTRACE
    // Warm the loader BEFORE a signal can arrive: the first backtrace() in a
    // process resolves symbols and may allocate, and a handler is the wrong
    // place to find that out.
    void* warm[4];
    (void)::backtrace(warm, 4);
#endif
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &detail::handler;
    ::sigemptyset(&sa.sa_mask);
    // No SA_ONSTACK: this tree does not install an alternate signal stack, and
    // asking for one that does not exist is worse than the default.
    sa.sa_flags = SA_RESTART;
    const int fatal[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT};
    for (int sig : fatal) ::sigaction(sig, &sa, nullptr);
}

#else // !RASBERY_HAS_CRASH_REPORT

inline void install() {}

#endif

} // namespace crash
} // namespace rasbery
