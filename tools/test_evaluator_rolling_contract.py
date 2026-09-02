#!/usr/bin/env python3
"""Static contract for WP18 -- per-slot refill (RASBERY_EVALUATOR_ROLLING).

WHAT THIS MODE CAN GET WRONG WITHOUT LOOKING WRONG.  Every defect it can
introduce produces a run that finishes, prints finite numbers and exits zero.

  * A lane that admits a case without the arena slot having been reset computes
    the new deck's flux from the previous deck's residency flags and mirrors --
    every value finite, every receipt plausible, one keff quietly wrong.
  * Two lanes holding one `--raso` write into one HDF5 file and one restart
    namespace.  The file is still a file afterwards.
  * A `wave` line whose fidelity declaration arrives after its cases have
    already started is a screening generation wearing an acceptance label.
  * A flag that is OFF but not INERT makes every wave-mode number taken before
    this work package incomparable to every number taken after it, and nothing
    in any receipt says so.
  * A dispatcher that claims a job without a free slot to put it in has moved
    the queue's tail into its own outstanding set: the arena still drains, the
    receipt still says `rolling`, and the arm reports the control's throughput
    under the treatment's name.

So the contract is not "does it work" -- it is a set of properties that cannot
be checked by looking at the output, pinned here in the source.

SEVEN PARTS.

  1. THE GATE.  The environment variable is read in one place, the wave path
     names none of the rolling state, and every line the mode prints lives in a
     function only the mode calls or behind an explicit `rollingEnabled()`
     branch.  Flag off is not "unused"; it is untouched.

  2. ADMISSION IS A FULL RESET.  A rolling lane admits a case the only way any
     case is admitted -- by constructing a Driver, which acquires an arena slot,
     which whole-struct resets it and audits the post-condition.  The lane may
     not have a door of its own.

  3. EPOCHS ARE MONOTONIC AND PAIRED.  Every admission takes `++_epoch` under
     the ledger lock and every finish retires exactly the epoch its admission
     took; a finish that retires an epoch nobody holds is counted, not ignored.

  4. ONE TENANT PER OUTPUT.  The queue admits a case only when no live tenancy
     holds its output, and it WAITS rather than refusing or dropping -- the
     serialisation a wave boundary used to provide by accident.

  5. NO BARRIER.  A lane loops on the queue; an empty-but-open queue is a wait,
     not an end.  The distinction between "the client has not sent the next one"
     and "the run is over" is the entire mode.

  6. THE REFUSALS, BY NAME.  Each one names the property it protects.

  7. THE HARNESS.  `--claim rolling` claims ONE job at a time, only while a slot
     is free, re-claims from inside the read loop, and refuses to run against a
     binary that does not report the mode in its [READY] receipt.

NEGATIVE CONTROLS.  Each check is a function of source text, and the bottom of
this file runs every one of them against a deliberately broken copy and fails if
the check PASSES.  A contract test that cannot fail is a comment.

Run: python tools/test_evaluator_rolling_contract.py
"""
from __future__ import annotations

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


SERVER = read("src/EvaluatorServer.h")
REFILL = read("src/BatchRefill.h")
BICG = read("src/CudaBICGBackend.cu")
DISPATCH = read("tools/run_multi_gpu_batch.py")


# ---------------------------------------------------------------------------
# Slicing helpers -- a member function's body, by brace depth
# ---------------------------------------------------------------------------
def member_body(src: str, name: str) -> str:
    """The body of member function *name*, or "" when it is not there.

    Brace counting rather than a regex: the bodies here contain string literals
    full of braces (`{\\"session\\":`), so a lazy `\\{.*?\\}` would stop at the
    first receipt and every check downstream would silently look at nothing.
    """
    match = re.search(r"\n    (?:void|bool|int|std::uint64_t) " + re.escape(name) + r"\(", src)
    if match is None:
        return ""
    open_brace = src.find("{", match.end())
    if open_brace < 0:
        return ""
    depth = 0
    i = open_brace
    in_string = False
    while i < len(src):
        ch = src[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace:i + 1]
        i += 1
    return ""


ROLLING_ONLY = ("rollingAdmit", "rollingOpen", "rollingLane", "rollingWaveLine",
                "rollingBarrier")


# ---------------------------------------------------------------------------
# 1. THE GATE
# ---------------------------------------------------------------------------
def check_gate(server: str) -> list[str]:
    bad: list[str] = []
    if server.count('std::getenv("RASBERY_EVALUATOR_ROLLING")') != 1:
        bad.append("RASBERY_EVALUATOR_ROLLING is not read in exactly one place; two "
                   "readers can disagree about which arm a run took, and the receipt "
                   "would name only one of them")
    if "static const bool on = [] {" not in server:
        bad.append("the rolling gate is not latched in a function-local static; a gate "
                   "re-read per case can change arm mid-run")

    # THE WAVE PATH MUST NOT NAME THE ROLLING STATE.  This is the feature-off
    # identity argument in its checkable form: if runWave never mentions the
    # queue, the ledger or the flag, then no wave-mode byte can depend on them.
    wave = member_body(server, "runWave")
    if not wave:
        bad.append("runWave could not be sliced; the flag-off check below is vacuous")
    for token in ("_roll", "_queue", "rolling", "Rolling"):
        if token in wave:
            bad.append(f"runWave mentions {token!r}: the wave path is supposed to be "
                       "untouched by WP18, and a shared statement is a shared output")

    # Every line the mode prints must be unreachable with the flag off.
    for tag in ("[RASBERY][EVALUATOR][ROLLING_START]",):
        for owner in ROLLING_ONLY:
            if tag in member_body(server, owner):
                break
        else:
            bad.append(f"{tag} is printed from outside the rolling-only functions, so "
                       "it can reach a wave-mode run's stdout")
    # ...and the two conditional fields on the READY receipt are the exception
    # that has to be explicit.
    if 'if (detail::rollingEnabled())\n            _out << ",\\"rolling\\":true' not in server:
        bad.append("the READY receipt's rolling fields are not guarded by "
                   "rollingEnabled(); with the flag off they would change a line every "
                   "existing measurement was taken beside")
    return bad


def check_wave_receipt_unchanged(refill: str) -> list[str]:
    """`[RASBERY][REFILL]` must keep exactly the keys its parsers know.

    tools/run_multi_gpu_batch.py (REFILL_RECEIPT), the campaign tables and every
    number already published read this line.  WP18 adds a SECOND line rather
    than keys to this one, and this is the check that keeps it that way.
    """
    bad: list[str] = []
    start = refill.find('out << "[RASBERY][REFILL] {')
    if start < 0:
        return ["the wave REFILL receipt is gone"]
    block = refill[start:refill.find("std::endl;", start)]
    expected = ["jobs", "slots", "lanes", "lanes_used", "lanes_never_admitted", "refills",
                "wall_s", "tail_idle_s", "slot_busy_fraction", "refill_latency_p50_ms",
                "refill_latency_max_ms", "admissions", "duplicates", "stale_tenants",
                "double_releases"]
    found = re.findall(r'\\"([a-z_0-9]+)\\":', block)
    if found != expected:
        bad.append("the wave [RASBERY][REFILL] receipt's keys changed from %r to %r; "
                   "every measurement taken before WP18 was parsed off that line"
                   % (expected, found))
    return bad


# ---------------------------------------------------------------------------
# 2. ADMISSION IS A FULL RESET
# ---------------------------------------------------------------------------
def check_admission_reset(server: str, bicg: str) -> list[str]:
    bad: list[str] = []
    lane = member_body(server, "rollingLane")
    if not lane:
        return ["rollingLane is missing: there is no per-slot refill to check"]
    if "runOneCase(" not in lane:
        bad.append("a rolling lane does not go through runOneCase; the Driver scope in "
                   "that function is what releases the arena slot between tenancies")
    if "Driver driver" in lane or "Driver(" in lane:
        bad.append("a rolling lane constructs its own Driver: the admission path must "
                   "have exactly one definition, or a refill can skip what the other "
                   "one does")

    # The reset itself, in the arena.  It is not WP18's code and that is the
    # point -- rolling admission reuses it, so a change to it has to break this.
    acquire = bicg[bicg.find("int CudaBatchArena::acquireSlot()"):]
    acquire = acquire[:acquire.find("\nvoid CudaBatchArena::releaseSlot")]
    if not acquire:
        bad.append("CudaBatchArena::acquireSlot could not be found")
        return bad
    if "sl        = BatchCore::Slot{};" not in acquire:
        bad.append("acquireSlot no longer whole-struct resets the slot; a refilled slot "
                   "inherits the previous tenant's mirrors and residency flags and "
                   "computes finite, plausible, wrong physics")
    if "batchSlotIsReset(sl)" not in acquire or "stale_tenants.fetch_add" not in acquire:
        bad.append("acquireSlot no longer AUDITS the reset post-condition; the reset "
                   "cannot miss a field, but it can stop being a whole-struct one")
    if "admissions.fetch_add" not in acquire:
        bad.append("acquireSlot no longer counts admissions, so the rolling receipt's "
                   "admit count has no independent witness")
    return bad


# ---------------------------------------------------------------------------
# 3. EPOCHS
# ---------------------------------------------------------------------------
def check_epoch(refill: str, server: str) -> list[str]:
    bad: list[str] = []
    admit = refill[refill.find("std::uint64_t admit("):]
    admit = admit[:admit.find("\n    void finish(")]
    if "const std::uint64_t epoch = ++_epoch;" not in admit:
        bad.append("an admission does not take a monotonically increasing epoch; two "
                   "tenancies that share an epoch cannot be told apart by the pairing "
                   "check below")
    if "_live_epochs.push_back(epoch)" not in admit:
        bad.append("an admission does not record its epoch as live, so nothing can "
                   "notice a completion attributed to the wrong tenancy")
    if "std::lock_guard<std::mutex> lock(_mutex);" not in admit:
        bad.append("admit() does not take the ledger lock; the lanes call it "
                   "concurrently and `++_epoch` is not atomic")

    finish = refill[refill.find("void finish(int lane, std::uint64_t epoch"):]
    finish = finish[:finish.find("\n    /// One request batch")]
    if "std::find(_live_epochs.begin(), _live_epochs.end(), epoch)" not in finish:
        bad.append("a finish does not retire the epoch its admission took")
    if "++_epoch_regressions" not in finish:
        bad.append("a finish that retires an epoch nobody holds is not counted; that is "
                   "a completion attributed to another tenancy, and it would show up "
                   "only as a width the run never had")

    # The lane has to hand the ledger back what the ledger gave it.
    lane = member_body(server, "rollingLane")
    if "rollingLedger().finish(lane, epoch," not in lane:
        bad.append("the lane does not return the admission's own epoch to finish(); the "
                   "pairing check is then a check of nothing")
    if 'stale_tenant_refusals' not in refill:
        bad.append("the rolling receipt does not report stale_tenant_refusals")
    return bad


# ---------------------------------------------------------------------------
# 4. ONE TENANT PER OUTPUT
# ---------------------------------------------------------------------------
def check_one_tenant_per_output(server: str) -> list[str]:
    bad: list[str] = []
    admissible = server[server.find("firstAdmissible() {"):]
    admissible = admissible[:admissible.find("return _queue.end();") + 24]
    if "_inflight" not in admissible:
        bad.append("the queue admits a job without checking whether its --raso is "
                   "already held; two Drivers would then write one HDF5 file and share "
                   "one restart namespace, which is what the wave-scoped namespace rule "
                   "exists to prevent")
    if "_inflight.push_back(out.output)" not in server:
        bad.append("a popped job's output is not registered as in flight")
    if "_inflight.erase(it)" not in server:
        bad.append("a finished job's output is never released, so any later case naming "
                   "it would wait forever")
    # It must WAIT, not drop.  The `_closed && _queue.empty()` conjunction is
    # what makes a closed queue holding a blocked job keep waiting.
    if "if (_closed && _queue.empty()) return false;" not in server:
        bad.append("a closed queue returns before its blocked jobs have run: a case "
                   "whose output was in flight at close would be silently dropped")
    return bad


# ---------------------------------------------------------------------------
# 5. NO BARRIER
# ---------------------------------------------------------------------------
def check_no_barrier(server: str) -> list[str]:
    bad: list[str] = []
    lane = member_body(server, "rollingLane")
    if "while (_queue.pop(job, immediate, waited_ms))" not in lane:
        bad.append("a rolling lane is not a pop loop; anything else re-introduces a "
                   "fixed job list and with it the wave barrier")
    if "#pragma omp for" in lane or "#pragma omp barrier" in lane:
        bad.append("a rolling lane contains a worksharing construct or a barrier, which "
                   "is exactly the wave semantics this mode replaces")
    pop = server[server.find("bool pop(RollingJob& out"):]
    pop = pop[:pop.find("\n    /// A tenancy ended")]
    if "_work.wait(lock);" not in pop:
        bad.append("an empty queue does not WAIT; a lane that exits on an empty queue "
                   "turns every gap in the client's stream into the end of the run")
    if "immediate = false;" not in pop:
        bad.append("the queue does not distinguish an admit that waited from one that "
                   "did not; without it, a harness that fell behind and an arena that "
                   "was full look the same in the receipt")
    return bad


# ---------------------------------------------------------------------------
# 6. THE REFUSALS
# ---------------------------------------------------------------------------
REFUSAL_NAMES = ("rolling_batch_width_latched", "rolling_wave_fidelity_after_admit",
                 "rolling_queue_closed")


def check_refusals(server: str) -> list[str]:
    bad: list[str] = []
    for name in REFUSAL_NAMES:
        if name not in server:
            bad.append(f"the refusal {name!r} is gone; the property it names is then "
                       "either unenforced or enforced silently")
    wave_line = member_body(server, "rollingWaveLine")
    if "wave_fidelity.empty()" not in wave_line:
        bad.append("a rolling `wave` line does not check for a fidelity declaration "
                   "arriving after its cases started; wave mode applies such a "
                   "declaration as a default, and a case that is already running cannot "
                   "be told what it should have been")
    if "if (_exit_code == 0) _exit_code = 2;" not in server:
        bad.append("a refusal no longer moves the exit code, so a generation that lost "
                   "candidates exits like one that did not")
    # A refusal can now be raised from the reader thread while lanes print
    # receipts; without the lock two lines interleave and neither parses.
    refuse = server[server.find("void refuse(const std::string& why"):]
    refuse = refuse[:refuse.find("\n    /// Resolve a wave")]
    if "std::lock_guard<std::mutex> lock(_out_mutex);" not in refuse:
        bad.append("refuse() does not serialise on the output mutex, and in rolling "
                   "mode it is called from the reader thread while lanes are writing")
    return bad


def check_receipt(refill: str) -> list[str]:
    bad: list[str] = []
    start = refill.find('out << "[RASBERY][REFILL][ROLLING] {')
    if start < 0:
        return ["the [RASBERY][REFILL][ROLLING] receipt is missing"]
    block = refill[start:refill.find("std::endl;", start)]
    for key in ("admits", "immediate_admits", "wave_barriers_avoided",
                "slot_idle_ms_total", "width_history", "stale_tenant_refusals",
                "epoch_regressions", "tail_idle_ms", "width_fill",
                "live_tenancies_at_close"):
        if f'\\"{key}\\"' not in block:
            bad.append(f"the rolling receipt has no {key!r} field")
    for pct in ("p10", "p50", "p90"):
        if f'\\"{pct}\\"' not in block:
            bad.append(f"width_history has no {pct!r} percentile")
    return bad


# ---------------------------------------------------------------------------
# 7. THE HARNESS
# ---------------------------------------------------------------------------
def check_harness(dispatch: str) -> list[str]:
    bad: list[str] = []
    if 'extra_env["RASBERY_EVALUATOR_ROLLING"] = "1"' not in dispatch:
        bad.append("--claim rolling does not set RASBERY_EVALUATOR_ROLLING in the child "
                   "environment, so the dispatcher would stream cases at an evaluator "
                   "that collects them and runs nothing")
    if '--claim rolling needs the persistent evaluator' not in dispatch:
        bad.append("--claim rolling is not refused without --evaluator; the chunked path "
                   "has no streaming form and would silently run as `auto`")
    if 'if not ready.get("rolling")' not in dispatch:
        bad.append("the dispatcher does not verify the [READY] receipt says the binary "
                   "is in rolling mode; against an older binary it would send one case "
                   "and wait forever for a receipt that needs a wave line")

    worker = dispatch[dispatch.find("def _run_rolling_worker("):]
    worker = worker[:worker.find("\ndef _run_wave_chunk(")]
    if not worker:
        return bad + ["_run_rolling_worker is missing"]
    # ONE AT A TIME, AND ONLY WHEN A SLOT IS FREE.  Claiming more than the arena
    # can hold moves the queue's tail into this worker's outstanding set: the
    # jobs stop being steal-able, the slow worker keeps them, and the arm
    # reports the control's tail under the treatment's name.
    if "queue.claim(1, index)" not in worker:
        bad.append("the rolling worker does not claim one job at a time; a chunked "
                   "claim is the tail this mode exists to remove, moved one layer out")
    if "len(outstanding) < target" not in worker:
        bad.append("the rolling worker tops up without checking that a slot is free; it "
                   "would claim the whole queue into its own outstanding set")
    if "target = max(1, batch_width + max(0, prefetch))" not in worker:
        bad.append("the outstanding target is not width + prefetch")
    # THE REFILL, FROM THIS SIDE: the next claim happens inside the read loop,
    # on a completion.  A top-up that only ran between waves would be a wave.
    on_line = worker[worker.find("def on_line("):worker.find("text, died = session.pump_until")]
    if "top_up()" not in on_line:
        bad.append("the rolling worker does not re-claim from inside the completion "
                   "callback, so the arena drains exactly as it does under `auto`")
    if "outstanding[output]" not in on_line and "del outstanding[output]" not in on_line:
        bad.append("a completion does not free an outstanding slot, so the top-up above "
                   "can never fire")
    if "evaluator_max_restarts=0" not in dispatch:
        bad.append("the rolling worker allows a restart; a rolling session's outstanding "
                   "set is spread across the arena, so a restart re-queues up to "
                   "width + prefetch cases into a process that just proved it can die")
    return bad


def check_harness_receipts(dispatch: str) -> list[str]:
    bad: list[str] = []
    if "ROLLING_RECEIPT = re.compile" not in dispatch:
        bad.append("the dispatcher has no parser for the rolling receipt")
    rolling_line = '[RASBERY][REFILL][ROLLING] {"admits":1}'
    wave_line = '[RASBERY][REFILL] {"jobs":1}'
    refill_re = re.search(r"^REFILL_RECEIPT = re\.compile\((r\".*\")\)$", dispatch, re.M)
    rolling_re = re.search(r"^ROLLING_RECEIPT = re\.compile\((r\".*\")\)$", dispatch, re.M)
    if refill_re is None or rolling_re is None:
        return bad + ["the two receipt regexes are not both present"]
    refill_pat = re.compile(eval(refill_re.group(1)))  # noqa: S307 - our own source
    rolling_pat = re.compile(eval(rolling_re.group(1)))  # noqa: S307
    if refill_pat.search(rolling_line):
        bad.append("REFILL_RECEIPT matches the ROLLING line: a rolling session's numbers "
                   "would be folded into tail_idle_s and slot_busy_fraction, which are "
                   "wave-mode quantities")
    if not rolling_pat.search(rolling_line):
        bad.append("ROLLING_RECEIPT does not match the rolling line")
    if rolling_pat.search(wave_line):
        bad.append("ROLLING_RECEIPT matches the wave line")
    if not refill_pat.search(wave_line):
        bad.append("REFILL_RECEIPT no longer matches the wave line")
    if "rolling tenancy audit" not in dispatch:
        bad.append("stale_tenant_refusals / epoch_regressions / live_tenancies_at_close "
                   "are not gated across the campaign; a nonzero value would be a "
                   "number in a log nobody reads")
    return bad


# ---------------------------------------------------------------------------
# 8. THE PROTOCOL, LIVE
# ---------------------------------------------------------------------------
#
# WHY A FAKE EVALUATOR AND NOT ANOTHER SOURCE SCAN.  Everything above is a
# statement about text.  The two properties that matter most here are
# statements about a CONVERSATION -- "the dispatcher never has more than
# width + prefetch cases outstanding" and "it re-claims on a completion, not on
# a barrier" -- and no amount of grep can tell whether the loop that implements
# them actually terminates, claims every job exactly once, or trips over a name.
# So the protocol is exercised against a stand-in that answers like the real
# evaluator and RECORDS what it was asked: the high-water mark of unanswered
# cases IS the arena occupancy this mode is buying, measured from the other end
# of the pipe.
FAKE_EVALUATOR = r"""
import json, sys, threading, time

WIDTH, PREFETCH = 4, 2
lock = threading.Lock()
queued = []
inflight = 0
answered = 0
high_water = 0
closed = False
work = threading.Condition(lock)


def emit(text):
    with lock:
        print(text, flush=True)


def lane():
    # A LANE, not a responder: it takes a case, holds it for a while, reports it
    # and takes the next.  The hold is what makes `outstanding` on the
    # dispatcher's side a real high-water mark instead of always 1, which is the
    # only reason this stand-in exists.
    global inflight, answered, high_water
    while True:
        with work:
            while not queued and not closed:
                work.wait()
            if not queued:
                return
            job = queued.pop(0)
            inflight += 1
            high_water = max(high_water, inflight + len(queued))
        # LONG ENOUGH that the dispatcher's opening burst is still outstanding
        # when the last of it arrives: that burst IS the property under test, and
        # a stand-in that answers faster than the pipe would report a high water
        # of 3 and call a working dispatcher broken.  Real cases are 30-90 s.
        time.sleep(0.25)
        with work:
            inflight -= 1
            answered += 1
            index = answered
        emit("[RASBERY][EVALUATOR][CASE] " + json.dumps(
            {"wave_id": 1, "case": index, "deck": job["deck"],
             "output": job["output"], "status": "ok",
             "digest": "%016x" % index, "isolation_check": False}))


# WP24: the stand-in prints `fidelity_preset` because a CURRENT binary does, and
# check_run_receipts() now requires the child to report back the preset it was
# handed.  A receipt without the field means "this binary predates WP24 and
# ignored RASBERY_FIDELITY", which is a refusal -- so a stand-in that omitted it
# would be standing in for the wrong binary.
emit('[RASBERY][PHYSICS_MODE] {"policy":"strict","physics_fidelity":"full",'
     '"acceptance_eligible":true,"screening":false,"result_mode":"light",'
     '"feedback_pass_limit":0,"fidelity_preset":"none"}')
emit('[RASBERY][EVALUATOR][READY] {"batch_width":4,"rolling":true,'
     '"rolling_prefetch":2,"rolling_target_inflight":6}')

lanes = [threading.Thread(target=lane, daemon=True) for _ in range(WIDTH)]
for t in lanes:
    t.start()

while True:
    # readline(), NOT `for line in sys.stdin`: the iterator protocol reads
    # AHEAD, so a stand-in written that way blocks until its input buffer fills
    # and the exercise deadlocks against a dispatcher that is waiting for it.
    line = sys.stdin.readline()
    if line == "":
        break
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    op = request.get("op")
    if op == "case":
        with work:
            queued.append(request)
            # SAMPLED HERE, on arrival, and not where a lane pops: what is under
            # test is how many cases the DISPATCHER keeps outstanding, and a
            # sample taken only at pop time never sees the ones still waiting.
            high_water = max(high_water, inflight + len(queued))
            work.notify()
    elif op == "wave":
        # The barrier: drain, then print the wave-shaped receipts.
        while True:
            with work:
                if not queued and inflight == 0:
                    break
            time.sleep(0.005)
        with work:
            total, water = answered, high_water
        emit('[RASBERY][BATCH_HOST] {"jobs":%d,"arena_width":4,"host_threads":4,'
             '"visible_cpus":4,"host_pinning":false,"pin_lease":true,'
             '"legacy_pinning_criterion":false,"wave_id":1}' % total)
        emit('[RASBERY][REFILL] {"jobs":%d,"slots":4,"lanes":4,"tail_idle_s":0.0}'
             % total)
        emit('[RASBERY][REFILL][ROLLING] {"session":1,"arena_width":4,"lanes":4,'
             '"admits":%d,"immediate_admits":%d,"wave_barriers_avoided":%d,'
             '"slot_idle_ms_total":0.0,"tail_idle_ms":0.0,"width_fill":0.9,'
             '"stale_tenant_refusals":0,"epoch_regressions":0,'
             '"live_tenancies_at_close":0,"high_water":%d}'
             % (total, total, max(0, total - 1), water))
        emit('[RASBERY][EVALUATOR][WAVE] {"wave_id":1,"jobs":%d,"ok":%d,"failed":0,'
             '"wall_s":1.0,"rolling":true}' % (total, total))
    elif op == "shutdown":
        with work:
            closed = True
            work.notify_all()
            total = answered
        emit('[RASBERY][EVALUATOR] {"cases":%d,"ok":%d,"failed":0,"refused":0,'
             '"stop_reason":"shutdown"}' % (total, total))
        break
"""


def check_protocol_live() -> list[str]:
    import os
    import sys
    import tempfile
    import types

    sys.path.insert(0, str(root / "tools"))
    import run_multi_gpu_batch as mg

    bad: list[str] = []
    jobs_n, width, prefetch = 20, 4, 2
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "fake_evaluator.py").write_text(FAKE_EVALUATOR, encoding="utf-8")
        jobs = [(f"deck{i:03d}.rasi", f"out{i:03d}.h5", "light") for i in range(jobs_n)]
        queue = mg.Queue(workdir / "queue.json", jobs_n, processes=1)
        result = mg.WorkerResult(gpu="0", proc=0, index=0)
        budget = types.SimpleNamespace(driver_workers=width, visible_cpus=4)
        session = mg._run_rolling_worker(
            result=result, queue=queue, jobs=jobs, index=0, budget=budget,
            batch_width=width, prefetch=prefetch,
            evaluator_command=[sys.executable, str(workdir / "fake_evaluator.py")],
            env=dict(os.environ), cwd=None, workdir=workdir, stem="gpu0.p0", gpu="0", proc=0,
            result_mode="light", declared_fidelity="strict", evaluator_max_restarts=0,
        )
        if session is not None:
            session.close()
        remaining = queue.remaining()

    if result.jobs != jobs_n:
        bad.append("the rolling worker sent %d of %d jobs" % (result.jobs, jobs_n))
    if remaining != 0:
        bad.append("the rolling worker left %d job(s) unclaimed" % remaining)
    if result.problems:
        bad.append("the rolling worker reported problems against a healthy stand-in: %r"
                   % (result.problems,))
    if result.failed_cases:
        bad.append("cases went unaccounted for: %r" % (result.failed_cases,))
    if not result.rolling_receipts:
        bad.append("the ROLLING receipt was not parsed off the barrier")
    else:
        high_water = int(result.rolling_receipts[0].get("high_water", -1))
        target = width + prefetch
        if high_water > target:
            bad.append(
                "the dispatcher had %d cases outstanding against a target of %d: it "
                "claims faster than the arena can hold, which moves the queue's tail "
                "into one worker's hands" % (high_water, target))
        if high_water < target:
            bad.append(
                "the dispatcher never filled the arena (high water %d of %d); a "
                "prefetch that never arrives is the wave arm with more round trips"
                % (high_water, target))
    if not result.refill_receipts:
        bad.append("the wave REFILL receipt was not parsed off the barrier")
    if result.waves != 1:
        bad.append("a rolling worker paid %d barriers; the whole point is one"
                   % result.waves)
    return bad


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
failures: list[str] = []
failures += check_gate(SERVER)
failures += check_wave_receipt_unchanged(REFILL)
failures += check_admission_reset(SERVER, BICG)
failures += check_epoch(REFILL, SERVER)
failures += check_one_tenant_per_output(SERVER)
failures += check_no_barrier(SERVER)
failures += check_refusals(SERVER)
failures += check_receipt(REFILL)
failures += check_harness(DISPATCH)
failures += check_harness_receipts(DISPATCH)
failures += check_protocol_live()

# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------
negative: list[str] = []


def control(name: str, checker, *sources) -> None:
    if not checker(*sources):
        negative.append(name)


control("check_gate misses the wave path reaching into the rolling queue",
        check_gate,
        SERVER.replace("        const int width = _summary.latched_width;",
                       "        const int width = _summary.latched_width; // _roll", 1))
control("check_gate misses a second reader of the environment variable",
        check_gate,
        SERVER.replace('inline bool rollingEnabled() {',
                       'inline bool rollingAgain() {\n'
                       '    return std::getenv("RASBERY_EVALUATOR_ROLLING") != nullptr;\n'
                       '}\n'
                       'inline bool rollingEnabled() {', 1))
control("check_gate misses the READY fields escaping their guard",
        check_gate,
        SERVER.replace('if (detail::rollingEnabled())\n            _out << ",\\"rolling\\":true',
                       '_out << ",\\"rolling\\":true', 1))
control("check_wave_receipt_unchanged misses a key added to the wave REFILL line",
        check_wave_receipt_unchanged,
        REFILL.replace('<< ",\\"admissions\\":" << c.admissions.load()',
                       '<< ",\\"width_fill\\":0'
                       '<< ",\\"admissions\\":" << c.admissions.load()', 1))
control("check_admission_reset misses acquireSlot losing its whole-struct reset",
        check_admission_reset, SERVER,
        BICG.replace("sl        = BatchCore::Slot{};", "sl.in_use = false;", 1))
control("check_admission_reset misses acquireSlot losing its audit",
        check_admission_reset, SERVER,
        BICG.replace("if (!batchSlotIsReset(sl))", "if (false)", 1))
control("check_admission_reset misses a lane that builds its own Driver",
        check_admission_reset,
        SERVER.replace("            runOneCase(job.deck, job.output, job.mode",
                       "            Driver driver(job.deck, job.output, job.mode);\n"
                       "            runOneCase(job.deck, job.output, job.mode", 1),
        BICG)
control("check_epoch misses an epoch that stops advancing",
        check_epoch,
        REFILL.replace("const std::uint64_t epoch = ++_epoch;",
                       "const std::uint64_t epoch = _epoch;", 1), SERVER)
control("check_epoch misses a finish that stops pairing",
        check_epoch,
        REFILL.replace("        const auto it = std::find(_live_epochs.begin(), "
                       "_live_epochs.end(), epoch);\n"
                       "        if (it == _live_epochs.end()) ++_epoch_regressions;\n"
                       "        else _live_epochs.erase(it);\n", "", 1), SERVER)
control("check_one_tenant_per_output misses a queue that ignores the in-flight set",
        check_one_tenant_per_output,
        SERVER.replace(
            "        for (auto it = _queue.begin(); it != _queue.end(); ++it)\n"
            "            if (std::find(_inflight.begin(), _inflight.end(), it->output) "
            "== _inflight.end())\n"
            "                return it;\n"
            "        return _queue.end();",
            "        return _queue.begin();", 1))
control("check_one_tenant_per_output misses a closed queue dropping a blocked job",
        check_one_tenant_per_output,
        SERVER.replace("if (_closed && _queue.empty()) return false;",
                       "if (_closed) return false;", 1))
control("check_no_barrier misses a lane that exits on an empty queue",
        check_no_barrier,
        SERVER.replace("            _work.wait(lock);", "            return false;", 1))
control("check_no_barrier misses a lane that stops being a pop loop",
        check_no_barrier,
        SERVER.replace("while (_queue.pop(job, immediate, waited_ms))",
                       "for (int i = 0; i < 8 && _queue.pop(job, immediate, waited_ms); ++i)",
                       1))
control("check_refusals misses a dropped refusal name",
        check_refusals,
        SERVER.replace("rolling_wave_fidelity_after_admit", "see_the_log", 1))
control("check_refusals misses a fidelity declaration applied retroactively",
        check_refusals,
        SERVER.replace("!wave_fidelity.empty()", "false", 1))
control("check_refusals misses an unserialised refusal",
        check_refusals,
        SERVER.replace("        std::lock_guard<std::mutex> lock(_out_mutex);\n"
                       "        ++_summary.refused;", "        ++_summary.refused;", 1))
control("check_receipt misses a dropped receipt field",
        check_receipt,
        REFILL.replace('<< ",\\"immediate_admits\\":" << _immediate\n', "", 1))
control("check_harness misses a rolling worker that claims in chunks",
        check_harness,
        DISPATCH.replace("queue.claim(1, index)", "queue.claim(batch_width, index)"))
control("check_harness misses a top-up that ignores whether a slot is free",
        check_harness,
        DISPATCH.replace("while pipe_ok and not exhausted and len(outstanding) < target:",
                         "while pipe_ok and not exhausted:", 1))
control("check_harness misses a top-up that no longer runs on a completion",
        check_harness,
        DISPATCH.replace("                    # THE REFILL, from this side: a completion "
                         "is a claim.\n                    top_up()\n", "", 1))
control("check_harness misses the dispatcher accepting a binary without the mode",
        check_harness,
        DISPATCH.replace('if not ready.get("rolling")', "if False", 1))
control("check_harness_receipts misses the wave regex swallowing the rolling line",
        check_harness_receipts,
        DISPATCH.replace(
            'REFILL_RECEIPT = re.compile(r"\\[RASBERY\\]\\[REFILL\\]\\s*(\\{.*\\})")',
            'REFILL_RECEIPT = re.compile(r"\\[RASBERY\\]\\[REFILL\\].*?(\\{.*\\})")', 1))
control("check_harness_receipts misses the rolling tenancy gate being dropped",
        check_harness_receipts,
        DISPATCH.replace("rolling tenancy audit", "rolling note", 1))

if negative:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are "
                    "therefore comments:\n    " + "\n    ".join(negative))

if failures:
    raise SystemExit("evaluator rolling contract: FAIL\n  " + "\n  ".join(failures))
print("evaluator rolling contract: PASS "
      f"({len(ROLLING_ONLY)} rolling-only functions, {len(REFUSAL_NAMES)} named refusals, "
      "23 negative controls, 1 live protocol exercise)")
