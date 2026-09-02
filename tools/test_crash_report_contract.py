#!/usr/bin/env python3
"""WP19.2 -- the crash-report contract, and the controls for it.

WHAT THIS DEFENDS.

Block 38 on 238 (0054838, 8 processes x M16 + MPS, 128-job manifest, six runs)
took two worker SIGSEGVs -- run2 proc 7 and run3 proc 1.  Both were recovered by
the dispatcher's restart/requeue, 128/128 and 127/128, and both were forensically
EMPTY.  This is the whole of what either one left behind:

    [RASBERY][MULTI_GPU][EVALUATOR][FATAL] {"gpu":"0","proc":7,"chunk":1,
     "attempt":1,"wave_id":101,"returncode":-11,"completed":0,
     "unfinished":[...16 jobs...],"requeued":true,"restarts":0}

`completed:0` means the child had not emitted its first [EVALUATOR][CASE]
receipt, so nothing upstream named a deck.  `ulimit -c` is 0 on that host,
/proc/sys/kernel/core_pattern routes to systemd-coredump, and `coredumpctl` is
not readable by the run account -- no core for either crash.  The block's own
write-up had to close with "unexplained ... out of scope".

The gap is not the crash.  It is that a crash on this host produces NO evidence
at all, so the next one will be exactly as unreadable as these two.  Four rules
close it, and every one is re-run against a mutated copy that breaks it:

  1. THE HANDLER EXISTS AND IS INSTALLED IN THE WORKER.  src/CrashReport.h
     installs on SIGSEGV/SIGBUS/SIGFPE/SIGILL/SIGABRT, and the evaluator's
     entry point -- which is what every dispatcher-started process runs --
     calls install() before it reads its first request.

  2. IT IS ASYNC-SIGNAL-SAFE.  Inside the handler: write(2), backtrace(),
     backtrace_symbols_fd(), sigaction(), getpid(), raise().  No std::ostream,
     no std::string, no printf family, no new/malloc.  A handler that allocates
     on a heap the fault may have corrupted prints nothing at all, which is the
     failure mode this is replacing.

  3. IT SAYS WHICH CASE, LANE, SLOT AND PHASE, AND IT RE-RAISES.  The four
     facts block 38 wanted; the breadcrumbs that carry them are written by both
     lane loops in EvaluatorServer.h; the slot -- which the evaluator layer
     genuinely does not know, hence its `"slot":-1` receipts -- is stamped by
     CudaOuterGraph.cu, which does.  Re-raise and not exit, so the dispatcher
     still sees returncode -11 and a host that CAN write a core still writes
     one.

  4. THE HARNESS KEEPS IT.  tools/run_multi_gpu_batch.py puts the dead child's
     stderr tail, its parsed crash records and its backtrace ON the FATAL
     record, because the record that says a worker crashed is the one a reader
     greps and the one block 38 found empty.

  5. THE HANDLER READS; IT DOES NOT MINT.  Rule 2 is a BAN-LIST, and a ban-list
     only ever catches the unsafe calls somebody already thought of.  It did
     not catch this one: the first cut of the handler printed `"tid"` by
     calling captureThreadOrdinal(), whose storage was

         static thread_local int id = next.fetch_add(1, ...);

     -- a function-local thread_local with a DYNAMIC initialiser, i.e. a
     guarded lazy init, whose initialiser is an atomic read-modify-write on a
     process-global counter.  With tracing off and no capture race (every
     healthy run, and BOTH block-38 SIGSEGVs) nothing had ever called it, so
     the handler was the first caller in the process: it ran the guarded init
     inside the signal, and the number it printed was a freshly minted 0 that
     cross-referenced against nothing -- while the header's own line 38-47
     claimed the handler "calls exactly write(2), backtrace(),
     backtrace_symbols_fd(), sigaction(), getpid(), and raise()".

     So this rule is the COMPLEMENT of rule 2: an allowlist over every
     identifier the handler names, plus the three structural facts that keep
     the allowed reads readable -- the ordinal's storage is CONSTANT-initialised
     (no guard variable), the reader is pure, and the minting happens on the
     ordinary path in crash::enter() and crash::install() so the handler has a
     real, cross-referenceable ordinal to load.  A newly added call is a finding
     until somebody justifies it in ALLOWED_HANDLER_CALLS.

Run:  python tools/test_crash_report_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TOOLS = os.path.join(ROOT, "tools")

TARGETS = ("CrashReport.h", "EvaluatorServer.h", "CudaOuterGraph.cu",
           "GpuCaptureArbiter.h")

#: The signals a solver can actually take and must not take silently.
FATAL_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL", "SIGABRT")

#: What the handler may call.  Everything on the POSIX async-signal-safe list
#: plus the two glibc backtrace entry points, which are the only way to a frame
#: list without a core.
BANNED_IN_HANDLER = (
    "std::cerr", "std::cout", "std::ostringstream", "std::string",
    "std::to_string", "snprintf", "sprintf", "fprintf", "printf",
    "new ", "malloc(", "std::vector",
)

#: Rule 5.  Every identifier the handler is allowed to name.  An ALLOWLIST on
#: purpose: BANNED_IN_HANDLER above is a ban-list, and captureThreadOrdinal()
#: -- a guarded lazy init plus an atomic RMW, run inside the signal -- was on
#: none of the ban-list's spellings.
ALLOWED_HANDLER_CALLS = frozenset((
    # this file's own renderers.  write(2) all the way down; see rawWrite().
    "rawWrite", "rawStr", "rawInt", "rawQuoted", "signalName", "reraise",
    # POSIX async-signal-safe.
    "write", "getpid", "raise", "sigaction",
    # glibc's documented malloc-free frame list -- the only stack that exists
    # on a host with ulimit -c 0 and coredumpctl unreadable.
    "backtrace", "backtrace_symbols_fd",
    # Pure loads of CONSTANT-initialised storage: a lock-free CAS on a
    # function-local std::atomic, the breadcrumb table, and thread_locals that
    # crash::enter() / crash::install() have already touched on the ordinary
    # path.  Nothing here allocates, locks, or lazily initialises.
    "reporting", "compare_exchange_strong", "load",
    "breadcrumbs", "currentLane", "captureArbiterOpen", "threadIsCapturing",
    "captureRaceEvents", "captureThreadOrdinalIfKnown",
))

#: Keywords the identifier scan would otherwise read as calls.
NOT_A_CALL = frozenset((
    "if", "for", "while", "switch", "return", "sizeof", "catch",
    "static_cast", "reinterpret_cast", "const_cast",
))

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

CRASH_TAG = "[RASBERY][CRASH]"


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def sources() -> dict[str, str]:
    out = {}
    for name in TARGETS:
        path = os.path.join(SRC, name)
        if not os.path.exists(path):
            raise SystemExit("missing source this contract is about: src/%s" % name)
        out[name] = read(path)
    return out


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        out.append(line.split("//")[0] if "//" in line and '"' not in line else line)
    return "\n".join(out)


def between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:] if j < 0 else text[i:j]


# ---------------------------------------------------------------------------
# rule 1 -- the handler exists and the worker installs it
# ---------------------------------------------------------------------------

def check_installed(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    crash = files["CrashReport.h"]
    install = between(crash, "inline void install()", "\n}\n")
    if not install:
        bad.append("src/CrashReport.h has no install()")
        return bad
    for sig in FATAL_SIGNALS:
        if sig not in install:
            bad.append("src/CrashReport.h: install() does not arm %s -- a worker "
                       "that dies of it leaves the block-38 record: completed:0 "
                       "and nothing else" % sig)
    if "sigaction(" not in install:
        bad.append("src/CrashReport.h: install() does not use sigaction(); signal() "
                   "has no portable disposition and no mask")

    ev = strip_comments(files["EvaluatorServer.h"])
    if "rasbery::crash::install()" not in ev:
        bad.append("src/EvaluatorServer.h never installs the handler -- the "
                   "evaluator IS the worker every dispatcher process runs, so a "
                   "handler it does not install protects nothing")
    else:
        # BEFORE the first request is read, or a crash during stand-up -- which
        # is exactly where run3 proc1 died, ~10 s in -- is still silent.
        run_body = between(ev, "int run() {", "\n        std::vector<CaseRequest> pending;")
        if "rasbery::crash::install()" not in run_body:
            bad.append("src/EvaluatorServer.h installs the handler somewhere other "
                       "than the top of run() -- a stand-up crash would beat it")
    return bad


# ---------------------------------------------------------------------------
# rule 2 -- async-signal safety
# ---------------------------------------------------------------------------

def check_handler_is_signal_safe(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    crash = files["CrashReport.h"]
    body = between(crash, "inline void handler(int sig)", "\n} // namespace detail")
    if not body:
        bad.append("src/CrashReport.h has no handler(int) -- this rule has no "
                   "subject")
        return bad
    code = strip_comments(body)
    for banned in BANNED_IN_HANDLER:
        if banned in code:
            bad.append("src/CrashReport.h: the handler uses %r.  A signal handler "
                       "that allocates or takes a stream lock on a heap the fault "
                       "may have corrupted prints nothing, which is the silence "
                       "this file exists to end" % banned)
    if "::write(" not in strip_comments(crash):
        bad.append("src/CrashReport.h never calls write(2) -- there is no "
                   "signal-safe way to emit anything else")
    # backtrace() warmed at install, not first called inside the handler.
    install = between(crash, "inline void install()", "\n}\n")
    if "backtrace(" in strip_comments(crash) and "backtrace(" not in install:
        bad.append("src/CrashReport.h: backtrace() is never called at install "
                   "time.  The FIRST call in a process resolves symbols and may "
                   "allocate; a handler is the wrong place to find that out")
    return bad


# ---------------------------------------------------------------------------
# rule 3 -- the four facts, the breadcrumbs, and the re-raise
# ---------------------------------------------------------------------------

def check_says_which_case(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    crash = files["CrashReport.h"]
    body = between(crash, "inline void handler(int sig)", "\n} // namespace detail")
    for field in ('\\"lane\\":', '\\"case\\":', '\\"slot\\":', '\\"phase\\":',
                  '\\"deck\\":'):
        if field not in body:
            bad.append("src/CrashReport.h: the crash record does not print %s -- "
                       "block 38's two SIGSEGVs were unreadable for exactly this "
                       "reason" % field.replace("\\", ""))
    if CRASH_TAG not in body:
        bad.append("src/CrashReport.h: the record carries no %s tag, so nothing "
                   "downstream can find it" % CRASH_TAG)
    if "capture_open" not in body or "capture_race_events" not in body:
        bad.append("src/CrashReport.h: the record does not say whether a capture "
                   "was open anywhere when the thread died -- the standing "
                   "question WP19 left and block 38 could not answer")
    if "backtrace_symbols_fd" not in body:
        bad.append("src/CrashReport.h: no frame list.  With ulimit -c 0 and "
                   "coredumpctl unreadable, this is the only stack that exists")
    reraise = between(crash, "inline void reraise(int sig)", "\n}\n")
    if "SIG_DFL" not in reraise or "raise(" not in reraise:
        bad.append("src/CrashReport.h: reraise() does not restore the default "
                   "disposition and re-raise.  Exiting instead would change the "
                   "dispatcher's returncode from -11 to something else and lose "
                   "the core on a host that can write one")
    if "_exit(" in strip_comments(crash) or "exit(" in strip_comments(body):
        bad.append("src/CrashReport.h: the handler exits rather than re-raising")

    ev = strip_comments(files["EvaluatorServer.h"])
    if ev.count("rasbery::crash::Scope") < 2:
        bad.append("src/EvaluatorServer.h opens a crash Scope in %d lane loop(s); "
                   "the wave path and the rolling path both need one, or half the "
                   "worker's crashes are still anonymous"
                   % ev.count("rasbery::crash::Scope"))
    outer = strip_comments(files["CudaOuterGraph.cu"])
    if "rasbery::crash::noteSlot(" not in outer:
        bad.append("src/CudaOuterGraph.cu does not stamp the arena slot.  The "
                   "evaluator layer genuinely does not know it -- its [ERROR] "
                   "receipts print slot:-1 -- so if the segment does not say it, "
                   "the crash record cannot either")
    return bad


# ---------------------------------------------------------------------------
# rule 5 -- the handler READS; it does not mint, allocate or lazily initialise
# ---------------------------------------------------------------------------

def check_handler_only_reads(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    crash = files["CrashReport.h"]
    arb = files["GpuCaptureArbiter.h"]
    body = between(crash, "inline void handler(int sig)", "\n} // namespace detail")
    if not body:
        return ["src/CrashReport.h has no handler(int) -- this rule has no subject"]
    code = strip_comments(body)

    # (a) the allowlist.  Anything the handler names that is not on it is a
    #     finding, whether or not anyone has thought about why.
    named = {m.group(1) for m in CALL_RE.finditer(code)} - NOT_A_CALL
    named.discard("handler")
    for name in sorted(named - ALLOWED_HANDLER_CALLS):
        bad.append("src/CrashReport.h: the handler calls %s(), which is not on "
                   "ALLOWED_HANDLER_CALLS.  Either it is async-signal-safe -- a "
                   "pure load of constant-initialised storage, or a call on the "
                   "POSIX list -- and belongs there with a reason, or it is the "
                   "next captureThreadOrdinal(): a lazy initialiser run inside "
                   "the signal, on a heap the fault may have corrupted" % name)

    # (b) the ordinal is read, never minted, in the handler.
    if "captureThreadOrdinalIfKnown(" not in code:
        bad.append("src/CrashReport.h: the handler does not read the thread "
                   "ordinal through captureThreadOrdinalIfKnown().  The minting "
                   "spelling runs a guarded lazy init and an atomic RMW inside "
                   "the signal, and on a healthy run the handler is its FIRST "
                   "caller in the process -- so the tid it prints is a fresh 0 "
                   "that cross-references against nothing in the rest of the log")
    if re.search(r"\bcaptureThreadOrdinal\s*\(", code):
        bad.append("src/CrashReport.h: the handler calls captureThreadOrdinal() "
                   "-- the minting spelling.  Use captureThreadOrdinalIfKnown()")

    # (c) the storage the reader loads is CONSTANT-initialised, so no guard
    #     variable and no dynamic initialiser exist for the handler to run.
    slot = between(arb, "inline int& captureThreadOrdinalSlot()", "\n}\n")
    if not slot:
        bad.append("src/GpuCaptureArbiter.h has no captureThreadOrdinalSlot() -- "
                   "the ordinal's storage and its minting are one function again, "
                   "so there is no spelling a signal handler may use")
    else:
        slot_code = strip_comments(slot)
        if "thread_local" not in slot_code:
            bad.append("src/GpuCaptureArbiter.h: captureThreadOrdinalSlot() is not "
                       "thread_local -- the ordinal would no longer identify a "
                       "thread")
        if "fetch_add" in slot_code or "fetch_sub" in slot_code:
            bad.append("src/GpuCaptureArbiter.h: the ordinal's storage has a "
                       "DYNAMIC initialiser again.  A function-local thread_local "
                       "initialised by a fetch_add is a guarded lazy init, which "
                       "is precisely what the crash handler must not run")

    peek = between(arb, "inline int captureThreadOrdinalIfKnown()", "\n")
    if not peek:
        bad.append("src/GpuCaptureArbiter.h has no captureThreadOrdinalIfKnown() "
                   "-- the handler has nothing signal-safe to call")
    else:
        for token in ("fetch_add", "fetch_sub", "getenv", "lock", "new "):
            if token in peek:
                bad.append("src/GpuCaptureArbiter.h: captureThreadOrdinalIfKnown() "
                           "uses %r.  It is called from a signal handler; it must "
                           "be a load and nothing else" % token)

    # (d) somebody warms it on the ordinary path, or the honest read is always
    #     -1 and the record is no more use than the one block 38 got.
    enter_body = strip_comments(between(crash, "inline void enter(int lane,", "\n}\n"))
    if "captureThreadOrdinal()" not in enter_body:
        bad.append("src/CrashReport.h: crash::enter() does not warm the thread "
                   "ordinal.  Every lane passes through it at the top of every "
                   "case; without the warm the handler's honest read is -1 for "
                   "every crash and the tid says nothing at all")
    install = strip_comments(between(crash, "inline void install()", "\n}\n"))
    if "captureThreadOrdinal()" not in install:
        bad.append("src/CrashReport.h: install() does not warm the installing "
                   "thread's ordinal.  It runs on the thread that never opens a "
                   "lane Scope -- which is where block 38's run3 proc1 died, ~10 s "
                   "in, during stand-up")

    # (e) and the record says which lane faulted, not just which were open.
    if "currentLane()" not in code:
        bad.append("src/CrashReport.h: the record does not name the FAULTING "
                   "lane.  `lanes[]` lists every case open anywhere in the "
                   "process -- up to sixteen rows on an M16 worker -- and marks "
                   "none of them, so the record names the suspects and not the "
                   "corpse")
    return bad


# ---------------------------------------------------------------------------
# rule 4 -- the harness keeps the dead child's last words
# ---------------------------------------------------------------------------

def check_harness_keeps_the_tail(_files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    harness_src = read(os.path.join(TOOLS, "run_multi_gpu_batch.py"))
    if "crash_evidence(" not in harness_src:
        return ["tools/run_multi_gpu_batch.py has no crash_evidence() -- the FATAL "
                "record still throws the dead child's output away, which is what "
                "made block 38's two SIGSEGVs unreconstructable"]
    fatal = between(harness_src, 'record = {\n            "gpu": gpu', "fatal_waves.append")
    if "crash_evidence(" not in fatal:
        bad.append("tools/run_multi_gpu_batch.py: crash_evidence() is not applied "
                   "to the FATAL record itself.  The record that says a worker "
                   "crashed is the one a reader greps")

    sys.path.insert(0, TOOLS)
    try:
        import run_multi_gpu_batch as harness  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover -- an import failure IS the finding
        return bad + ["tools/run_multi_gpu_batch.py does not import: %s" % exc]

    # The exact three lines src/CrashReport.h emits, against the parser that has
    # to read them.  A contract that only greps for a token cannot tell a
    # working regex from a typo.
    text = "\n".join((
        '  NO.=  8  EFPD=   665.000  K-EFF=0.999999  PPM= 1236.81  outer= 87',
        '[RASBERY][CRASH] {"signal":11,"name":"SIGSEGV","pid":41213,"tid":6,'
        '"capture_open":1,"thread_capturing":0,"capture_race_events":2,'
        '"lanes":[{"lane":5,"case":8,"slot":3,"phase":"drive",'
        '"deck":"/home/x/candidate_0060.json"}]}',
        '[RASBERY][CRASH][FRAME] begin',
        'RASBERY(_ZN7rasbery3gpu16CudaOuterSegment5driveEv+0x1f4)[0x55d1a2]',
        '/lib/x86_64-linux-gnu/libc.so.6(+0x29d90)[0x7f2c1d]',
        '[RASBERY][CRASH][END] {"signal":11,"frames":2}',
    ))
    evidence = harness.crash_evidence(text)
    if "stderr_tail" not in evidence:
        bad.append("crash_evidence() returns no stderr_tail -- a crash with no "
                   "[CRASH] line at all (an OOM kill, a glibc abort message) is "
                   "exactly the case that most needs the raw tail kept")
    records = evidence.get("crash") or []
    if len(records) != 1:
        bad.append("crash_evidence() parsed %d crash record(s) from one crash; "
                   "the [END] and [FRAME] lines must not be mistaken for records"
                   % len(records))
    else:
        lanes = records[0].get("lanes") or []
        if not lanes or lanes[0].get("case") != 8 or lanes[0].get("lane") != 5:
            bad.append("crash_evidence() lost the case/lane out of the record: %r"
                       % (records[0],))
    frames = evidence.get("backtrace") or []
    if len(frames) != 2:
        bad.append("crash_evidence() kept %d frame line(s) of 2 -- a backtrace "
                   "clipped by its own delimiters is worse than none" % len(frames))
    if any(CRASH_TAG in f for f in frames):
        bad.append("crash_evidence() folded a delimiter line into the backtrace")
    return bad


CHECKS = (
    ("the handler exists and the worker installs it", check_installed),
    ("the handler is async-signal-safe", check_handler_is_signal_safe),
    ("the record names case, lane, slot, phase -- and re-raises",
     check_says_which_case),
    ("the harness keeps the dead child's last words", check_harness_keeps_the_tail),
    ("the handler reads and does not mint", check_handler_only_reads),
)


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------

def mutate(files: dict[str, str], name: str, old: str, new: str) -> dict[str, str]:
    out = dict(files)
    if name not in out:
        raise SystemExit("control target not found: %s" % name)
    if old not in out[name]:
        raise SystemExit("control anchor not found in %s: %r" % (name, old[:80]))
    out[name] = out[name].replace(old, new, 1)
    return out


def controls(files: dict[str, str]):
    return [
        ("SIGSEGV is dropped from the armed set",
         check_installed,
         mutate(files, "CrashReport.h",
                "const int fatal[] = {SIGSEGV,",
                "const int fatal[] = {")),
        ("the worker stops installing it",
         check_installed,
         mutate(files, "EvaluatorServer.h",
                "rasbery::crash::install();",
                "// rasbery::crash::install();")),
        ("the handler formats with an ostringstream",
         check_handler_is_signal_safe,
         mutate(files, "CrashReport.h",
                "    rawStr(\"[RASBERY][CRASH] {\\\"signal\\\":\");",
                "    std::ostringstream line; rawStr(\"[RASBERY][CRASH] "
                "{\\\"signal\\\":\");")),
        ("the record stops naming the case",
         check_says_which_case,
         mutate(files, "CrashReport.h",
                'rawStr(",\\"case\\":");',
                'rawStr(",\\"unknown\\":");')),
        ("the handler exits instead of re-raising",
         check_says_which_case,
         mutate(files, "CrashReport.h",
                "    sa.sa_handler = SIG_DFL;\n    ::sigaction(sig, &sa, nullptr);\n"
                "    ::raise(sig);",
                "    (void)sa;\n    ::_exit(139);")),
        ("only one lane loop leaves breadcrumbs",
         check_says_which_case,
         mutate(files, "EvaluatorServer.h",
                "const rasbery::crash::Scope _crumb(\n                lane, i,",
                "const int _crumb_off = lane; (void)_crumb_off; //(\n                lane, i,")),
        # --- rule 5 -------------------------------------------------------
        ("the handler mints a thread ordinal inside the signal again",
         check_handler_only_reads,
         mutate(files, "CrashReport.h",
                "rawInt(captureThreadOrdinalIfKnown());",
                "rawInt(captureThreadOrdinal());")),
        ("the ordinal's storage goes back to a guarded lazy initialiser",
         check_handler_only_reads,
         mutate(files, "GpuCaptureArbiter.h",
                "static thread_local int id = -1;",
                "static thread_local int id = next.fetch_add(1, "
                "std::memory_order_relaxed);")),
        ("nobody warms the ordinal on the ordinary path",
         check_handler_only_reads,
         mutate(files, "CrashReport.h",
                "    (void)captureThreadOrdinal();\n    (void)threadIsCapturing();",
                "    // (void)captureThreadOrdinal();\n    (void)threadIsCapturing();")),
        ("the reader starts minting after all",
         check_handler_only_reads,
         mutate(files, "GpuCaptureArbiter.h",
                "inline int captureThreadOrdinalIfKnown() { return "
                "captureThreadOrdinalSlot(); }",
                "inline int captureThreadOrdinalIfKnown() { return "
                "next.fetch_add(1); }")),
        ("the handler picks up a call nobody vetted",
         check_handler_only_reads,
         mutate(files, "CrashReport.h",
                '    rawStr(",\\"capture_open\\":");',
                '    captureArbiterProvenance();\n'
                '    rawStr(",\\"capture_open\\":");')),
        ("the record stops saying which lane faulted",
         check_handler_only_reads,
         mutate(files, "CrashReport.h",
                "    rawInt(currentLane());",
                "    rawInt(-1);")),
    ]


def harness_controls():
    """Rule 4's controls run against MUTATED TEXT rather than mutated source:
    the parser is imported, so the honest way to break it is to feed it a log
    that a broken parser would mis-read."""
    sys.path.insert(0, TOOLS)
    import run_multi_gpu_batch as harness  # noqa: PLC0415

    out = []
    # A crash with no [CRASH] record at all -- an OOM kill, say.  The tail must
    # still survive, or the "unexplained" verdict comes straight back.
    silent = "\n".join("  NO.=  %d  EFPD= 620.0" % i for i in range(300))
    evidence = harness.crash_evidence(silent)
    out.append(("a silent death still keeps a tail",
                bool(evidence.get("stderr_tail")) and "crash" not in evidence))
    # And the tail is bounded: a 300-line preamble must not become the record.
    out.append(("the tail is bounded",
                len(evidence["stderr_tail"]) <= harness.CRASH_TAIL_LINES))
    return out


def main() -> int:
    files = sources()

    failures = 0
    for label, fn in CHECKS:
        bad = fn(files)
        if bad:
            failures += 1
            print("FAIL  %s" % label)
            for line in bad[:20]:
                print("        %s" % line)
        else:
            print("ok    %s" % label)

    print()
    for label, fn, mutated in controls(files):
        if fn(mutated):
            print("ok    control caught: %s" % label)
        else:
            failures += 1
            print("FAIL  control NOT caught: %s" % label)

    print()
    for label, held in harness_controls():
        if held:
            print("ok    %s" % label)
        else:
            failures += 1
            print("FAIL  %s" % label)

    print()
    crash = files["CrashReport.h"]
    print("signals armed:         %d"
          % sum(1 for s in FATAL_SIGNALS if s in crash))
    print("breadcrumb sites:      %d"
          % strip_comments(files["EvaluatorServer.h"]).count("rasbery::crash::Scope"))
    print("slot stamps:           %d"
          % strip_comments(files["CudaOuterGraph.cu"]).count("rasbery::crash::noteSlot("))

    print()
    print("FAILED (%d)" % failures if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
