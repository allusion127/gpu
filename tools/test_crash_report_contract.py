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

Run:  python tools/test_crash_report_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TOOLS = os.path.join(ROOT, "tools")

TARGETS = ("CrashReport.h", "EvaluatorServer.h", "CudaOuterGraph.cu")

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
