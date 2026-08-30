#!/usr/bin/env python3
"""WP19.1 -- the stand-up/capture isolation contract, and the controls for it.

WHAT THIS DEFENDS, AND WHY IT IS NOT WP19's FILE.

tools/test_capture_arbiter_contract.py defends the LOUD face of the stand-up
race: a lane's allocation invalidating a sibling lane's capture, which CUDA
answers with a code in the 900 block and which WP19 (c4656c6) closed by putting
every capture and every allocation in src/ inside the arbiter.

The 238 run of 2026-08-30 (8 processes x M16, v6 env, c4656c6) then produced the
SAME shape with no CUDA error anywhere:

    "error":"CUDA BiCGSTAB detected a non-finite value"
    "slot":-1, "lane":1, "wall_s":18.414, 1 case of 128, ~2 runs in 5,
    always the FIRST case a lane took, a different deck every rerun,
    every failing deck bit-identical single-shot.

and both evaluator logs printed `"capture_race_retry":1` -- the process that
lost a case and the process that did not.

THE MECHANISM, which is what this file holds the source to.

WP19's fix (c) retries a capture-illegal graph build ONCE with the arbiter held.
That is sound exactly when re-running the build's body recorder moves no host
state.  CudaOuterGraph.cu's recorder is `runOneOuter(1u)`, and it does move host
state: its enqueue helpers commit byte-exact upload shadows (CudaTransferMirror.h
commits AT THE ISSUE, not at the landing) and bump residency generations, for
copies that were only RECORDED into a capture the failure is about to discard.
The second record then finds every shadow already committed, ELIDES the uploads,
and instantiates a WHILE body with no H2D node for data the device never
received.  On a lane's FIRST case that device memory is uninitialised, so the
replay produces a non-finite flux -- with no CUDA error, four frames and one
graph replay away from the cause.  Which is why the same function has, since
Task 10, refused to fall back to the stream arm past `record(body)` for exactly
this reason ("a plausible wrong answer rather than a slow one") -- and why the
retry WP19 added beside that refusal had to ask the same question and did not.

So the rules:

  1. ONE PREDICATE for "has the build moved host state", asked by both the
     retry gate and the fallback gate, and its list of pre-body stages agrees
     with the stage strings GpuOuterWhile.h actually assigns.
  2. NO capture-race retry is reached without that gate, and a refusal it
     forbids is ABANDONED loudly instead (counted, named, with the stage on it).
     A recorder that is exempt has to say why in its own source.
  3. THE RETRY RECEIPT PRINTS.  WP19's noteCaptureRaceRetry() took no
     arguments and printed no line: one cumulative integer in a teardown
     receipt the dispatcher did not read.  Every retry and every abandonment
     now prints its own line with the site, the stage and the slot, both
     counters are in the arbiter receipt, and the dispatcher lifts all of it.
  4. THE LOUD PATH.  A lane's FIRST case dying non-finite prints the arbiter's
     stand-up provenance and is re-run EXACTLY ONCE from a clean slot; a lane's
     later case is not, because a lane's later case replays a graph the first
     one built and retrying it would launder physics into luck.
  5. WORKSPACES PRIVATE OR FENCED.  The one stream in this tree that is shared
     by every lane is the CMFD arena's, and the one path that can hand it to a
     capturing body is CudaOuterSegment::useStream().  In a batch it must stay
     private: Driver.h adopts the arena stream only on the SOLO arm.
  6. THE STRING THE LOUD PATH MATCHES is the string the tree throws.

Every rule is re-run against a MUTATED copy that breaks it (the negative
controls at the bottom); a check that cannot fail is not a check.

Run:  python tools/test_capture_standup_isolation_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TOOLS = os.path.join(ROOT, "tools")

TARGETS = (
    "GpuCaptureArbiter.h",
    "GpuOuterWhile.h",
    "CudaOuterGraph.cu",
    "CudaPprBackend.cu",
    "CudaBICGBackend.cu",
    "EvaluatorServer.h",
    "Driver.h",
)

#: The predicate both gates ask, by name.  Spelled once here so a rename shows
#: up as one failure and not as six.
GATE = "outerWhileStageMovedHostState"

#: The per-event lines WP19.1 requires.  A counter without a line is what made
#: the 2026-08-30 evidence unreadable.
RETRY_TAG = "[RASBERY][CUDA][CAPTURE_RACE][RETRY]"
ABANDON_TAG = "[RASBERY][CUDA][CAPTURE_RACE][ERROR]"
LOUD_TAG = "[RASBERY][EVALUATOR][CAPTURE_RACE]"

#: The message CudaBICGBackend.cu / BICGCMFD.cpp actually throw.
NONFINITE_TEXT = "CUDA BiCGSTAB detected a non-finite value"


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
    """Line comments removed; block comments left alone (this tree has none in
    the regions checked).  Enough to keep a rule from being satisfied by a
    sentence ABOUT the rule."""
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        out.append(line.split("//")[0] if "//" in line and '"' not in line else line)
    return "\n".join(out)


def between(text: str, start: str, end: str) -> str:
    """The slice from the first `start` to the first `end` after it -- how a
    check asks about ONE function without parsing C++."""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:] if j < 0 else text[i:j]


# ---------------------------------------------------------------------------
# rule 1 -- one predicate, and its stage list is the builder's
# ---------------------------------------------------------------------------

STAGE_ASSIGN = re.compile(r'\*stage\s*=\s*"([^"]+)"')


def check_stage_predicate(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    outer = files["CudaOuterGraph.cu"]
    if ("bool %s(" % GATE) not in outer:
        bad.append("src/CudaOuterGraph.cu defines no %s() -- the retry and the "
                   "fallback have no shared answer to 'did the body run?'" % GATE)
        return bad

    body = between(outer, "bool %s(" % GATE, "\n}\n")
    named = set(re.findall(r'"([^"]+)"', body))

    builder = files["GpuOuterWhile.h"]
    assigned = [m.group(1) for m in STAGE_ASSIGN.finditer(builder)]
    if not assigned:
        bad.append("src/GpuOuterWhile.h assigns no *stage -- the predicate has "
                   "nothing to agree with")
        return bad
    # Everything up to and including BeginCaptureToGraph(body) is plumbing; the
    # first stage AFTER it is the first one that has run the recorder.
    if "record(body)" not in assigned:
        bad.append("src/GpuOuterWhile.h no longer has a record(body) stage -- the "
                   "predicate's cut point does not exist")
        return bad
    plumbing = assigned[: assigned.index("record(body)")]
    missing = [s for s in plumbing if s not in named]
    if missing:
        bad.append("%s() does not name %s -- a refusal there would be treated as "
                   "'the body ran' and the retry would be skipped for a build that "
                   "moved nothing" % (GATE, ", ".join(repr(s) for s in missing)))
    extra = [s for s in named if s not in plumbing]
    if extra:
        bad.append("%s() names %s, which GpuOuterWhile.h assigns at or after "
                   "record(body) -- that build DID move host state and must not be "
                   "retried" % (GATE, ", ".join(repr(s) for s in extra)))
    if "return true" not in body:
        bad.append("%s() has no conservative default -- an unknown cursor must "
                   "read as 'moved'" % GATE)

    # asked TWICE: once by the retry gate, once by the fallback gate.
    if outer.count("%s(stage)" % GATE) < 2:
        bad.append("%s(stage) is asked %d time(s); the retry gate and the "
                   "host_state_moved fallback must both ask it, or they will "
                   "disagree" % (GATE, outer.count("%s(stage)" % GATE)))
    return bad


# ---------------------------------------------------------------------------
# rule 2 -- no ungated retry, and an abandonment is loud
# ---------------------------------------------------------------------------

def check_retry_is_gated(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    outer = strip_comments(files["CudaOuterGraph.cu"])
    if "noteCaptureRaceAbandoned(" not in outer:
        bad.append("src/CudaOuterGraph.cu never abandons a capture-illegal build "
                   "whose body already ran -- it can only retry, which is the "
                   "defect WP19.1 closes")
    # The gate has to come BEFORE the retry, and the retry has to be its else.
    gate_at = outer.find("%s(stage)" % GATE)
    retry_at = outer.find("noteCaptureRaceRetry(")
    if gate_at < 0 or retry_at < 0 or gate_at > retry_at:
        bad.append("src/CudaOuterGraph.cu: the capture-race retry is reached "
                   "without %s(stage) in front of it" % GATE)
    else:
        window = outer[gate_at:retry_at]
        if "else if" not in window:
            bad.append("src/CudaOuterGraph.cu: %s(stage) and the retry are not one "
                       "if/else -- a build that moved host state can still fall "
                       "into the retry" % GATE)
    # Every retry call names its site and its stage; a bare call is the WP19
    # shape that printed nothing.
    for name in ("CudaOuterGraph.cu", "CudaPprBackend.cu"):
        code = strip_comments(files[name])
        for m in re.finditer(r"noteCaptureRaceRetry\(([^;]*)\)", code):
            if m.group(1).count(",") < 3:
                bad.append("src/%s: noteCaptureRaceRetry(%s) -- a retry with no "
                           "site, stage and slot on it is a retry nobody can "
                           "attribute" % (name, m.group(1).strip()[:60]))
    # A recorder that is EXEMT from the gate has to say so where it is written.
    ppr = files["CudaPprBackend.cu"]
    if "noteCaptureRaceRetry(" in strip_comments(ppr) and "WP19.1" not in ppr:
        bad.append("src/CudaPprBackend.cu retries without the gate and without a "
                   "WP19.1 note saying why its recorder moves no host state")
    return bad


# ---------------------------------------------------------------------------
# rule 3 -- the receipt actually prints
# ---------------------------------------------------------------------------

def check_receipt_is_visible(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    arbiter = files["GpuCaptureArbiter.h"]

    retry_body = between(arbiter, "inline void noteCaptureRaceRetry(", "\n}\n")
    if RETRY_TAG not in retry_body or "std::cerr" not in retry_body:
        bad.append("GpuCaptureArbiter.h: noteCaptureRaceRetry prints no %s line -- "
                   "the 2026-08-30 evidence had the counter and nothing else"
                   % RETRY_TAG)
    abandon_body = between(arbiter, "inline void noteCaptureRaceAbandoned(", "\n}\n")
    if not abandon_body:
        bad.append("GpuCaptureArbiter.h has no noteCaptureRaceAbandoned()")
    elif ABANDON_TAG not in abandon_body or "std::cerr" not in abandon_body:
        bad.append("GpuCaptureArbiter.h: noteCaptureRaceAbandoned prints no %s line"
                   % ABANDON_TAG)

    for term in ("capture_race_retry", "capture_race_abandoned",
                 "capture_race_unrecovered"):
        if ('\\"%s\\":' % term) not in arbiter:
            bad.append("GpuCaptureArbiter.h: captureArbiterReceipt() does not print "
                       "%s" % term)
    if "captureArbiterProvenance" not in arbiter:
        bad.append("GpuCaptureArbiter.h has no captureArbiterProvenance() -- a "
                   "first-case death cannot say what the arbiter saw")

    harness = read(os.path.join(TOOLS, "run_multi_gpu_batch.py"))
    for token in ("CAPTURE_ARBITER", "CAPTURE_RACE", "collect_capture_race",
                  "capture_race_retry"):
        if token not in harness:
            bad.append("tools/run_multi_gpu_batch.py does not lift %s -- the term "
                       "stays in a per-worker file nobody opens" % token)
    return bad


def check_harness_reads_the_line(_files: dict[str, str]) -> list[str]:
    """Rule 3b, live: the dispatcher's regexes against the exact lines emitted."""
    bad: list[str] = []
    sys.path.insert(0, TOOLS)
    try:
        import run_multi_gpu_batch as harness  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover -- an import failure IS the finding
        return ["tools/run_multi_gpu_batch.py does not import: %s" % exc]

    arbiter_line = (
        '[RASBERY][CUDA][CAPTURE_ARBITER] {"tag":"run","enabled":1,'
        '"capture_windows":20,"capture_wait_us":252482,"alloc_windows":4820,'
        '"alloc_overlapped":2,"alloc_blocked":5,"alloc_wait_us":89931,'
        '"alloc_in_capture":0,"captures_unwound":0,"capture_race_retry":1,'
        '"capture_race_abandoned":1,"capture_race_unrecovered":0}'
    )
    event_line = (
        '[RASBERY][CUDA][CAPTURE_RACE][RETRY] {"tag":"outer.while",'
        '"stage":"BeginCapture(root)","cuda_error":901,"slot":3,"tid":7,'
        '"open":1,"retried":1}'
    )
    loud_line = (
        '[RASBERY][EVALUATOR][CAPTURE_RACE] {"wave_id":101,"case":0,"lane":1,'
        '"deck":"candidate_0048.json","lane_first_case":true,'
        '"error":"%s","action":"retry_once_clean_slot",'
        '"capture_windows":20,"alloc_overlapped":2}' % NONFINITE_TEXT
    )
    probe = harness.WorkerResult(gpu="0", proc=0)
    text = "\n".join((arbiter_line, event_line, loud_line))
    harness.collect_capture_race(probe, text)
    harness.collect_capture_race(probe, text)  # a wave's text is scanned twice
    if probe.capture_arbiter is None:
        bad.append("collect_capture_race did not parse the arbiter receipt")
    elif probe.capture_arbiter.get("capture_race_retry") != 1:
        bad.append("collect_capture_race lost capture_race_retry: %r"
                   % probe.capture_arbiter)
    if len(probe.capture_race) != 2:
        bad.append("collect_capture_race recorded %d event line(s) for two events "
                   "scanned twice (want 2)" % len(probe.capture_race))
    return bad


# ---------------------------------------------------------------------------
# rule 4 -- the loud path, once, first case only
# ---------------------------------------------------------------------------

def check_first_case_loud_path(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    ev = files["EvaluatorServer.h"]
    if "retryAfterCaptureRace" not in ev:
        return ["src/EvaluatorServer.h has no retryAfterCaptureRace() -- a lane's "
                "first case still dies silently"]

    body = between(ev, "void retryAfterCaptureRace(", "\n    }\n")
    # The GUARD, not the parameter: a signature that still takes the flag while
    # the early return has stopped reading it is exactly the regression this
    # control mutates into existence.
    guard = between(body, "if (status == 0", "return;")
    if "!lane_first_case" not in guard:
        bad.append("retryAfterCaptureRace's early return does not read "
                   "lane_first_case -- a lane's LATER case replays a graph the "
                   "first one built, so retrying it launders physics into luck")
    if "captureRaceCorruptionSuspect(" not in body:
        bad.append("retryAfterCaptureRace retries on any failure, not on the "
                   "non-finite signature")
    if body.count("runOneCase(") != 1:
        bad.append("retryAfterCaptureRace calls runOneCase %d times -- the contract "
                   "is exactly one retry" % body.count("runOneCase("))
    if "captureArbiterProvenance()" not in body:
        bad.append("retryAfterCaptureRace prints no stand-up provenance, which is "
                   "the only thing that says whether a capture window was open")
    if LOUD_TAG not in body:
        bad.append("retryAfterCaptureRace prints no %s line" % LOUD_TAG)
    if "std::cerr" not in body:
        bad.append("retryAfterCaptureRace does not reach stderr, so a race that "
                   "fired is invisible to a reader of the harness log")

    # BOTH lane loops call it: the wave path and the rolling path.
    if ev.count("retryAfterCaptureRace(") < 3:  # the definition plus two callers
        bad.append("retryAfterCaptureRace is wired into %d lane loop(s); the wave "
                   "path and the rolling path both need it"
                   % (ev.count("retryAfterCaptureRace(") - 1))
    for term in ("capture_race_case_retries", "capture_race_case_recovered"):
        if ('\\"%s\\":' % term) not in ev:
            bad.append("src/EvaluatorServer.h: the process receipt does not report "
                       "%s" % term)
    return bad


# ---------------------------------------------------------------------------
# rule 5 -- the shared stream stays out of a captured body in a batch
# ---------------------------------------------------------------------------

def check_shared_stream_not_captured(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    driver = strip_comments(files["Driver.h"])
    if "useStream(" not in driver:
        bad.append("src/Driver.h no longer arms CudaOuterSegment::useStream -- this "
                   "check has lost its subject")
        return bad
    arm = between(driver, "const bool shared_stream", ";")
    if not arm:
        bad.append("src/Driver.h: the shared-stream arm is not a named decision "
                   "any more, so nothing here can hold it to the solo case")
        return bad
    if "solo" not in arm:
        bad.append("src/Driver.h adopts the CMFD arena stream as the outer "
                   "segment's without the solo gate.  In --batch-mode that stream "
                   "is shared by every lane AND is the body stream of the outer "
                   "WHILE capture -- a sibling lane's enqueue on it would be "
                   "swallowed into the capture instead of executed")
    outer = strip_comments(files["CudaOuterGraph.cu"])
    if "m.while_cache.root_stream" not in outer:
        bad.append("src/CudaOuterGraph.cu: the WHILE root is no longer captured on "
                   "its own scratch stream")
    return bad


# ---------------------------------------------------------------------------
# rule 6 -- the loud path matches the string the tree throws
# ---------------------------------------------------------------------------

def check_nonfinite_spelling(files: dict[str, str]) -> list[str]:
    bad: list[str] = []
    thrown = NONFINITE_TEXT in files["CudaBICGBackend.cu"]
    if not thrown:
        bad.append("src/CudaBICGBackend.cu no longer throws %r -- the loud path "
                   "matches a message nothing produces" % NONFINITE_TEXT)
    ev = files["EvaluatorServer.h"]
    body = between(ev, "inline bool captureRaceCorruptionSuspect(", "\n}\n")
    if not body:
        bad.append("src/EvaluatorServer.h has no captureRaceCorruptionSuspect()")
        return bad
    needles = re.findall(r'"([^"]+)"', body)
    if not needles:
        bad.append("captureRaceCorruptionSuspect matches nothing")
    for needle in needles:
        if needle not in NONFINITE_TEXT:
            bad.append("captureRaceCorruptionSuspect looks for %r, which is not in "
                       "the message this tree throws (%r)" % (needle, NONFINITE_TEXT))
    return bad


CHECKS = (
    ("one host-state predicate, agreeing with the builder", check_stage_predicate),
    ("no capture-race retry without the gate", check_retry_is_gated),
    ("the retry receipt prints and is lifted", check_receipt_is_visible),
    ("the dispatcher reads the lines that are emitted", check_harness_reads_the_line),
    ("a lane's first case is loud and retried once", check_first_case_loud_path),
    ("the shared arena stream stays out of a captured body", check_shared_stream_not_captured),
    ("the loud path matches the thrown message", check_nonfinite_spelling),
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
        # THE DEFECT ITSELF, put back: the retry no longer asks whether the body
        # ran.  This is c4656c6's shape, and it is the mutation that must fail.
        ("the retry stops asking whether the body ran",
         check_retry_is_gated,
         mutate(files, "CudaOuterGraph.cu",
                "if (rasbery::captureIllegal(static_cast<int>(rc)) &&\n"
                "                outerWhileStageMovedHostState(stage)) {",
                "if (false) {")),
        ("the predicate treats a post-body stage as plumbing",
         check_stage_predicate,
         mutate(files, "CudaOuterGraph.cu",
                '"UpdateCaptureDependencies", "BeginCaptureToGraph(body)"};',
                '"UpdateCaptureDependencies", "BeginCaptureToGraph(body)",\n'
                '        "record(body)"};')),
        ("the retry goes back to counting without printing",
         check_receipt_is_visible,
         mutate(files, "GpuCaptureArbiter.h",
                '"[RASBERY][CUDA][CAPTURE_RACE][RETRY] {\\"tag\\":\\""',
                '"[RASBERY][CUDA][CAPTURE_RACE][RETRYX] {\\"tag\\":\\""')),
        ("the loud path drops its first-case gate",
         check_first_case_loud_path,
         mutate(files, "EvaluatorServer.h",
                "if (status == 0 || !lane_first_case || "
                "!captureRaceCorruptionSuspect(failure))",
                "if (status == 0 || !captureRaceCorruptionSuspect(failure))")),
        ("the loud path retries in a loop",
         check_first_case_loud_path,
         mutate(files, "EvaluatorServer.h",
                "        if (retry_status == 0)\n"
                "            _capture_race_case_recovered.fetch_add(1, "
                "std::memory_order_relaxed);",
                "        if (retry_status != 0)\n"
                "            runOneCase(deck, output, mode, warm_from, warm_save, "
                "fidelity, retry_status,\n"
                "                       retry_failure, retry_receipt, retry_seconds, "
                "retry_teardown);")),
        ("a batch adopts the shared arena stream again",
         check_shared_stream_not_captured,
         mutate(files, "Driver.h",
                "        const bool shared_stream =\n            solo && have_sweep_stream &&",
                "        const bool shared_stream =\n            have_sweep_stream &&")),
        ("the loud path matches a message nothing throws",
         check_nonfinite_spelling,
         mutate(files, "EvaluatorServer.h",
                'return failure.find("non-finite") != std::string::npos;',
                'return failure.find("not-finite") != std::string::npos;')),
    ]


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
    outer = files["CudaOuterGraph.cu"]
    print("gate asked:            %d time(s) in CudaOuterGraph.cu"
          % outer.count("%s(stage)" % GATE))
    print("retry sites:           %d"
          % sum(strip_comments(t).count("noteCaptureRaceRetry(")
                for t in files.values()))
    print("abandon sites:         %d"
          % sum(strip_comments(t).count("noteCaptureRaceAbandoned(")
                for t in files.values()))
    print("loud-path callers:     %d"
          % (files["EvaluatorServer.h"].count("retryAfterCaptureRace(") - 1))

    print()
    print("FAILED (%d)" % failures if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
