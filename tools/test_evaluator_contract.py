#!/usr/bin/env python3
"""Static contract for WP8 stage 1 -- the long-lived GA evaluator.

WHAT AN EVALUATOR CAN GET WRONG WITHOUT LOOKING WRONG.  Every failure mode this
mode introduces produces a run that finishes, prints finite numbers and exits
zero.  A case that inherited a value from the case before it does not crash; it
returns a slightly different keff.  A wave that quietly re-parsed the library
does not fail; it is just slow.  A request whose fidelity nobody honoured does
not warn; it lands in an acceptance table.  So the contract is not "does it
work" -- it is a set of properties that CANNOT be checked by looking at the
output, pinned here in the source.

FIVE PARTS.

  1. THE MODE.  The evaluator flags are parsed before the generic
     value-consuming loop (which would reject an optional-value and a valueless
     flag), the mode requires --batch-mode M and refuses argv decks, and it
     resolves the SAME execution mode as the `--jobs --batch-mode M` run whose
     per-case digests it has to reproduce.

  2. THE LIFETIMES.  The three teardown steps the single-shot batch branch runs
     per process are each either done per wave or explicitly deferred to
     shutdown -- and the deferred ones are asserted, not assumed.  The arena is
     released exactly once, after the wave loop.

  3. THE REFUSALS.  Two request fields cannot be honoured in stage 1
     (`batch_width` after the first wave, `fidelity` at all), and a stage-1
     evaluator that IGNORED either would be the campaign's fastest route from a
     screening run to an acceptance table.  Both refuse by name, and a refusal
     moves the exit code.

  4. THE RECEIPTS.  Every field the 238 runbook reads, plus the anti-drift rule
     that the evaluator's shutdown block emits every receipt tag the batch
     branch's does -- the two blocks are deliberately separate code (feature-off
     identity beat DRY here) and nothing else would notice them diverging.

  5. CROSS-CASE ISOLATION.  A scan of the case path for process-lifetime mutable
     state, against an inventory that has to be updated by hand.  A new
     unclassified static is the exact shape of a cross-case leak, and it is
     invisible in every output this program produces.

NEGATIVE CONTROLS.  Each check is a function of source text, and the bottom of
this file runs every one of them against a deliberately broken copy and fails if
the check PASSES.  A contract test that cannot fail is a comment.
"""
from __future__ import annotations

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


MAIN = read("src/main.cpp")
SERVER = read("src/EvaluatorServer.h")
DRIVER = read("src/Driver.h")

failures: list[str] = []


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(re.sub(r"//.*", "", line) for line in text.split("\n"))


# ===========================================================================
# 1. THE MODE
# ===========================================================================
def check_mode(main: str) -> list[str]:
    bad: list[str] = []
    flat = squash(strip_comments(main))

    # The three flags exist and are handled BEFORE the generic loop.  That loop
    # requires every option to be followed by a non-`--` value, so
    # `--evaluator-isolation-check` (no value) and a bare `--evaluator-jsonl`
    # (optional value) would both die there with "Missing value after option".
    for flag in ('"--evaluator"', '"--evaluator-jsonl"', '"--evaluator-idle-timeout"',
                 '"--evaluator-isolation-check"'):
        if flag not in main:
            bad.append(f"main.cpp does not accept {flag}")
    handled = main.find('option == "--evaluator" || option == "--evaluator-jsonl"')
    generic = main.find('std::cerr << "Missing value after option: "')
    if not (0 < handled < generic):
        bad.append(
            "the evaluator flags must be handled BEFORE the generic value-consuming loop "
            f"(handled={handled} generic={generic}); a valueless flag would otherwise be "
            "rejected with a misleading 'Missing value after option'")

    # An evaluator is a BATCH process whose job list arrives later.  If it
    # declared Single, the mode-dependent Anderson default (Driver.h) would flip
    # and every per-case digest would differ from the `--jobs --batch-mode M`
    # run this mode has to reproduce bit for bit -- and the B0 gate would fail
    # for a reason unrelated to the lifetime change under test.
    if ("const bool batch_execution = (batch_width > 0 && !rasbery_inputs.empty()) "
            "|| evaluator_mode;") not in flat:
        bad.append("the evaluator does not feed the ONE batch_execution predicate, so it "
                   "would resolve the mode-dependent Anderson default differently from the "
                   "batch run whose digests it must reproduce")

    # ...and it returns before the batch branch, so one predicate still selects
    # exactly one path.
    evaluator_branch = flat.find("if (evaluator_mode) {")
    batch_branch = flat.find("if (batch_execution) {")
    if not (0 < evaluator_branch < batch_branch):
        bad.append("the evaluator branch must come before `if (batch_execution)` and return, "
                   "or one predicate would select two paths")

    # The width is not optional: the arena is one allocation sized at the first
    # admission, so it must be known before the first request is read.
    if "--evaluator requires --batch-mode M" not in main:
        bad.append("--evaluator does not refuse a missing --batch-mode M")
    if "--evaluator takes its jobs from the request stream" not in main:
        bad.append("--evaluator does not refuse decks given on argv, which would run under a "
                   "different job-namespace rule than the waves")

    # FEATURE-OFF IDENTITY.  Everything that makes a run acceptable runs before
    # the evaluator branch, in the order it always did.
    receipt = flat.find("[RASBERY][PHYSICS_MODE]")
    exact_only = flat.find("[RASBERY][EXACT_ONLY][FAIL]")
    if not (0 < exact_only < receipt < evaluator_branch):
        bad.append("the evaluator branch runs before the exact-only gate or the "
                   "[PHYSICS_MODE] receipt; an evaluator that skipped the fidelity contract "
                   "is the fastest route yet from a screening run to an acceptance table")

    # One manifest grammar, not two.
    if "options.read_manifest = rasberyReadJobManifest;" not in main:
        bad.append("the evaluator does not reuse main.cpp's --jobs manifest reader; a second "
                   "implementation would be a second answer to 'what is a job'")
    return bad


# ===========================================================================
# 2. THE LIFETIMES
# ===========================================================================
def check_lifetimes(main: str, server: str) -> list[str]:
    bad: list[str] = []
    code = strip_comments(main)

    # The arena release is THE deferred step -- it is the entire lever -- so it
    # happens exactly once, in main(), after the wave loop has returned.
    if "rasberyReleaseBatchArena" in strip_comments(server):
        bad.append("EvaluatorServer.h releases the batch arena; the whole point of the mode "
                   "is that the arena outlives every wave, and only main() tears it down")
    run_at = code.find("exit_code         = server.run();")
    release_at = code.find("rasbery::rasberyReleaseBatchArena();\n        server.stampArenaRelease();")
    if run_at < 0 or release_at < 0 or not (run_at < release_at):
        bad.append("the evaluator branch does not release the arena exactly once, after the "
                   "wave loop, with the release stamped into the receipt")

    # The writer: joined once, at shutdown; flushed per wave.  Per-wave joining
    # is NOT needed because every case fences its own sessions in IO::~IO, and
    # per-wave joining would restart the thread on the next wave for nothing.
    if "iowriter::shutdown()" in strip_comments(server):
        bad.append("EvaluatorServer.h joins the writer thread; that belongs at process "
                   "shutdown, once")
    if "iowriter::flushLines();" not in strip_comments(server):
        bad.append("the wave loop does not flush the line sink, so a wave's telemetry could "
                   "sit in a buffer until the process ends")

    # The pin registry is the step that is ASSERTED rather than run.  Every
    # registration is leased and released by its owner's destructor, so between
    # waves there must be none live; a nonzero count is a lease that outlived
    # its Driver, which is the defect the lease was introduced to kill.
    if "rasberyHostPinLiveRanges()" not in server:
        bad.append("the wave loop does not assert that no host pin lease outlived its "
                   "Driver; that is the between-wave state nothing else would notice")
    if "pin_live_ranges" not in server:
        bad.append("the wave/process receipt does not carry the live pin-lease count")

    # Stage 1 means a FRESH case object every time.  A Driver hoisted out of the
    # loop would be stage 3 arriving without its reset coverage.
    if "Driver driver(deck, output, mode);" not in server:
        bad.append("a case does not construct its own Driver; stage 1 rebuilds the case "
                   "object every time and stage 3 is where that stops being true")
    return bad


# ===========================================================================
# 3. THE REFUSALS
# ===========================================================================
def check_refusals(server: str) -> list[str]:
    bad: list[str] = []
    code = strip_comments(server)

    if "latched batch_width=" not in server:
        bad.append("a wave asking for a different batch_width is not refused; the arena is "
                   "one allocation sized at the first admission and a second width would "
                   "silently run at the first one's")
    if "effectivePhysicsFidelity()" not in code or "fidelityAgrees" not in code:
        bad.append("a request's declared fidelity is not checked against the fidelity this "
                   "process actually resolved; a stage-1 evaluator that ignored the field "
                   "would answer a screening request with a strict receipt")
    if "cannot change fidelity per case" not in server:
        bad.append("the fidelity refusal does not say why it cannot be honoured")
    if "unknown op " not in server:
        bad.append("an unrecognised op is not refused by name")
    if "--raso paths must be distinct within a wave" not in server:
        bad.append("a wave does not apply the output-namespace rule; two Drivers on one "
                   "--raso race inside one HDF5 file and share a restart namespace")
    # A refusal that did not move the exit code is a refusal nothing downstream
    # can see: a generation that silently lost four candidates is not a
    # generation.
    refuse_body = code[code.find("void refuse("):]
    refuse_body = refuse_body[:refuse_body.find("\n    }")]
    if "_exit_code" not in refuse_body:
        bad.append("a refused request does not move the exit code")
    return bad


# ===========================================================================
# 4. FAILURE ISOLATION
# ===========================================================================
def check_failure_isolation(server: str) -> list[str]:
    bad: list[str] = []
    code = strip_comments(server)
    body = code[code.find("static void runOneCase("):]
    body = body[:body.find("\n    void reportCase(")] if "\n    void reportCase(" in body else body
    if "catch (const std::exception& error)" not in body or "catch (...)" not in body:
        bad.append("runOneCase does not catch every exception; an escaping one terminates "
                   "the OpenMP region and takes the other cases' partial results with it, "
                   "and a RASBERY_GPU_FULL=1 fail-closed refusal arrives as exactly that")
    if "throw" in body.replace("throws", ""):
        bad.append("runOneCase rethrows; a bad case must fail alone")
    # The Driver is scoped so its destructor -- the slot release -- runs before
    # the teardown stamp closes.  Same rule main.cpp's batch branch follows, and
    # for the same reason: teardown IS the refill latency being measured.
    if not re.search(r"\{\s*Driver driver\(deck, output, mode\);", body):
        bad.append("the Driver is not scoped inside the try, so the slot release would not "
                   "be inside the measured teardown")
    return bad


# ===========================================================================
# 5. THE RECEIPTS
# ===========================================================================
PROCESS_FIELDS = (
    "cases", "generations", "process_uptime_s", "cuda_context_reuse", "xslib_hits",
    "xslib_loads", "case_teardown_ms", "case_seconds", "library_loads", "geometry_builds",
    "arena_standups", "arena_releases", "slot_admissions", "slot_duplicates",
    "slot_stale_tenants", "slot_double_releases", "isolation_checks",
    "isolation_mismatches", "stop_reason", "refused", "batch_width",
)
CASE_FIELDS = ("wave_id", "case", "key", "deck", "output", "result_mode", "status",
               "exit_code", "digest", "statepoints", "outers", "slot", "lane", "wall_s",
               "teardown_ms", "isolation_check", "error")
WAVE_FIELDS = ("wave_id", "jobs", "ok", "failed", "wall_s", "cases_per_hour",
               "process_reused", "xslib_loads", "xslib_hits", "pin_live_ranges")


def check_receipts(server: str) -> list[str]:
    bad: list[str] = []
    for tag, fields in (("[RASBERY][EVALUATOR] ", PROCESS_FIELDS),
                        ("[RASBERY][EVALUATOR][CASE] ", CASE_FIELDS),
                        ("[RASBERY][EVALUATOR][WAVE] ", WAVE_FIELDS)):
        start = server.find(tag)
        if start < 0:
            bad.append(f"EvaluatorServer.h never emits {tag.strip()}")
            continue
        block = server[start:start + 4000]
        for field in fields:
            if f'\\"{field}\\":' not in block:
                bad.append(f"the {tag.strip()} receipt is missing {field!r}")
    # The digest is the identity of the case, and a case that never folded one
    # must say so rather than report zero.
    if "receipt.complete" not in server:
        bad.append("the case receipt does not distinguish 'no digest' from 'digest zero'")
    return bad


# The evaluator's shutdown block is a deliberate near-copy of the batch
# branch's: keeping the batch branch byte-identical beat sharing the code.  This
# is what notices them diverging.
SHUTDOWN_TAGS = (
    "[RASBERY][BATCH_HOST][PIN] {",
    "[RASBERY][GPU_FULL] {",
    "[RASBERY][XSRECON][GPU] ",
    "[RASBERY][FLATXS][GPU] ",
    "[RASBERY][XE_GPU] {",
    "[RASBERY][NODAL][GPU] ",
    "[RASBERY][NODAL][CANON] ",
    "[RASBERY][HDF5][LOCK] ",
)


def check_receipt_parity(main: str) -> list[str]:
    bad: list[str] = []
    evaluator = main.find("if (evaluator_mode) {")
    batch = main.find("if (batch_execution) {")
    if evaluator < 0 or batch < 0 or evaluator >= batch:
        return ["cannot locate the evaluator and batch branches to compare their receipts"]
    block = main[evaluator:batch]
    for tag in SHUTDOWN_TAGS:
        if tag not in block:
            bad.append(f"the evaluator shutdown block omits {tag.strip()!r}, which the batch "
                       "branch emits; the two blocks are separate code and nothing else "
                       "would notice them diverging")
    for call in ("rasbery::PrintXsLibraryCacheReceipt(std::cout);",
                 "rasbery::gpu::reportOuterSegment(std::cout);",
                 "rasbery::xsphase::report(std::cout);",
                 "rasbery::outer_timing::report(std::cout);",
                 "rasbery::iowriter::reportSummary(std::cout);",
                 "rasberyPrintProcessLedger(std::cout,"):
        if call not in block:
            bad.append(f"the evaluator shutdown block omits {call!r}")
    return bad


# ===========================================================================
# 6. CROSS-CASE ISOLATION -- the scan
# ===========================================================================
#
# WHAT IS SCANNED, AND WHY NOT MORE.  The .cu arenas are deliberately NOT here:
# their per-slot state is audited at RUNTIME by the tenancy counters
# (duplicates / stale_tenants / double_releases, BatchRefill.h), which is a
# stronger witness than any text scan and is already gated at 0.  What has no
# runtime witness is the HOST solver's process-lifetime state, and that is this.
#
# AUTO-CLASSIFIED AS SAFE, without an entry below:
#   * const / constexpr                 immutable after stand-up by definition
#   * std::atomic                       receipt counters; nothing reads them back
#   * mutex / condition_variable / once_flag   synchronisation, not values
#   * `static T x; return x;`           a singleton accessor -- the container is
#                                       process state, and what it CONTAINS is
#                                       audited where it is written
#
# EVERYTHING ELSE NEEDS A LINE HERE.  The key is (file, the declaration with its
# whitespace squashed); line numbers are deliberately absent because they move
# for unrelated reasons and a churning test gets disabled.
CASE_PATH_SOURCES = (
    "src/Driver.h", "src/XSSet.cpp", "src/Geometry.cpp", "src/PPR.cpp", "src/Nodal.cpp",
    "src/IO.cpp", "src/CMFD.cpp", "src/BICGCMFD.cpp", "src/BICGSolver.cpp",
)

# Reason codes.
#
# SCRATCH  A per-worker workspace, resized and filled inside the call that uses
#          it.  It ALREADY survives a case boundary today: `--batch-mode M` with
#          more decks than lanes recycles an OpenMP worker onto a second deck,
#          and that path is bit-identity gated.  So the evaluator adds no new
#          exposure here -- but a stale-value read would be invisible, and WP8
#          stage 3's generation poison is where these get a positive guarantee
#          rather than an argument.  Any NEW one has to be justified the same
#          way, deliberately, which is why they are listed one by one.
#
# DIAG     A one-shot latch or counter behind an off-by-default RASBERY_*_DUMP
#          environment variable.  It changes NOTHING the solver reads.  It does
#          change behaviour across cases in an evaluator -- only the FIRST case
#          of the process dumps -- and that is named here so nobody later reads
#          a one-case capture as a per-case one.
CLASSIFIED: dict[tuple[str, str], str] = {
    ("src/XSSet.cpp", "static thread_local CrossSection tls_xs, tls_delta;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local milk::Vector<double> tls_iden;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<DeltaApplication> deltas;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local CrossSection tls_xs, tls_delta, tls_xs2;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local milk::Vector<double> tls_iden, tls_iden2;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local CrossSection tls_xsp;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local milk::Vector<double> tls_idenp;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<double> buf;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<DeltaApplication> history;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<double> micprobe;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<int> p_did;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<double> p_x, p_scale;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<DeltaApplication> hist;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local DepletionWorkspace ws_tls;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<double> abs_flux_tls;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local DepletionWorkspace workspace;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<double> corrected_flux_tls;"): "SCRATCH",
    ("src/XSSet.cpp", "static thread_local std::vector<double> sub_flux_tls;"): "SCRATCH",
    ("src/XSSet.cpp", "static bool dump_done = false;"): "DIAG",
    ("src/XSSet.cpp", "static int call = 0;"): "DIAG",
    ("src/PPR.cpp", "static thread_local std::vector<double> gmap_interp;"): "SCRATCH",
    ("src/PPR.cpp", "static thread_local std::vector<double> fmap_interp;"): "SCRATCH",
}

STATIC_DECL = re.compile(
    r"^[ \t]*(?:inline\s+)?static\s+(?P<qual>(?:thread_local\s+)?)(?P<rest>[^;{()=]*?)"
    r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=[^;]*|\{[^;]*\})?;",
    re.M)
AUTO_SAFE = ("const", "constexpr", "std::atomic", "std::mutex", "std::condition_variable",
             "std::once_flag")


def scan_process_state(rel: str, text: str) -> list[tuple[str, str]]:
    """Every process-lifetime mutable declaration this scanner cannot auto-clear."""
    found: list[tuple[str, str]] = []
    for match in STATIC_DECL.finditer(text):
        decl = (match.group("qual") + match.group("rest")).strip()
        if any(token in decl for token in AUTO_SAFE):
            continue
        tail = text[match.end():match.end() + 240].split("\n")
        following = next((line.strip() for line in tail if line.strip()), "")
        if following.startswith("return " + match.group("name")):
            continue  # singleton accessor
        found.append((rel, squash(match.group(0)).strip()))
    return found


def check_process_state() -> list[str]:
    bad: list[str] = []
    seen: set[tuple[str, str]] = set()
    for rel in CASE_PATH_SOURCES:
        path = root / rel
        if not path.exists():
            bad.append(f"{rel} named in CASE_PATH_SOURCES does not exist")
            continue
        for entry in scan_process_state(rel, path.read_text(encoding="utf-8", errors="replace")):
            seen.add(entry)
            if entry not in CLASSIFIED:
                bad.append(
                    f"{entry[0]}: unclassified process-lifetime mutable state "
                    f"{entry[1]!r}. A case that writes it is a case that can change the "
                    "NEXT case's answer, and nothing this program prints would show it. "
                    "Classify it in tools/test_evaluator_contract.py (SCRATCH or DIAG) "
                    "with the argument for why it is safe, or move it into CaseContext.")
    for entry in CLASSIFIED:
        if entry not in seen:
            bad.append(f"{entry[0]}: the inventory still lists {entry[1]!r}, which the scan "
                       "no longer finds -- delete the stale entry so the list stays a "
                       "description of the tree")
    return bad


# ===========================================================================
# THE RUNTIME HALF, pinned here so the two cannot drift
# ===========================================================================
def check_isolation_check(server: str) -> list[str]:
    bad: list[str] = []
    if "--evaluator-isolation-check" not in MAIN:
        bad.append("main.cpp does not offer the runtime isolation check")
    if "_options.isolation_check" not in server:
        bad.append("EvaluatorServer.h never acts on the isolation-check option")
    # NON-ADJACENT is the property.  A back-to-back repeat would mostly
    # re-exercise ONE worker's thread_local buffers; the first-then-last
    # ordering puts the recheck on whichever lane the queue gives it, after
    # every other deck has been through.
    if "cases_between" not in server or "adjacent" not in server:
        bad.append("the isolation receipt does not report how many cases separated the two "
                   "runs; an adjacent repeat is a much weaker test and must not be reported "
                   "as the same thing")
    if "isolationOutput(" not in server:
        bad.append("the isolation re-run does not get its own output path; sharing one would "
                   "have the check overwrite the thing it is checking (the restart namespace "
                   "is derived from the output path)")
    if "recheck.digest != receipts[u0].digest" not in server:
        bad.append("the isolation check does not compare DIGESTS; comparing anything a "
                   "receipt prints would miss a trajectory change too small to print")
    return bad


# ===========================================================================
# RUN
# ===========================================================================
failures += check_mode(MAIN)
failures += check_lifetimes(MAIN, SERVER)
failures += check_refusals(SERVER)
failures += check_failure_isolation(SERVER)
failures += check_receipts(SERVER)
failures += check_receipt_parity(MAIN)
failures += check_isolation_check(SERVER)
failures += check_process_state()

# The Driver side of the per-case receipt: it must be stamped from the same
# variables the trajectory line prints, or the evaluator's per-case digest and
# the log's could disagree about the same case.
if "_case_receipt.digest      = sp_traj.h;" not in DRIVER:
    failures.append("Driver does not stamp the case receipt from the trajectory fold")
if "struct CaseReceipt" not in DRIVER or "caseReceipt() const" not in DRIVER:
    failures.append("Driver does not expose the per-case receipt the evaluator reports")

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Every check above, run against a copy broken in the exact
# way the check exists to catch.  A check that still passes here is a comment.
# ---------------------------------------------------------------------------
negative: list[str] = []


def control(name: str, checker, *args) -> None:
    if not checker(*args):
        negative.append(name)


control("check_mode misses a lost --batch-mode refusal",
        check_mode, MAIN.replace("--evaluator requires --batch-mode M", "whatever"))
control("check_mode misses the evaluator declaring Single",
        check_mode, MAIN.replace(
            "(batch_width > 0 && !rasbery_inputs.empty()) || evaluator_mode;",
            "(batch_width > 0 && !rasbery_inputs.empty());"))
control("check_lifetimes misses the arena being released per wave",
        check_lifetimes, MAIN, SERVER.replace(
            "iowriter::flushLines();", "rasberyReleaseBatchArena();"))
control("check_lifetimes misses a dropped pin-lease assertion",
        check_lifetimes, MAIN, SERVER.replace("rasberyHostPinLiveRanges()", "0u"))
control("check_refusals misses an ignored fidelity field",
        check_refusals, SERVER.replace("effectivePhysicsFidelity()", "PhysicsFidelity::FullExact"))
control("check_refusals misses a lost width latch",
        check_refusals, SERVER.replace("latched batch_width=", "used width "))
control("check_failure_isolation misses a case that takes the process down",
        check_failure_isolation, SERVER.replace("catch (...) {", "catch (int) {"))
control("check_receipts misses a dropped receipt field",
        check_receipts, SERVER.replace('\\"cuda_context_reuse\\":', '\\"ctx\\":'))
control("check_receipt_parity misses the two shutdown blocks diverging",
        check_receipt_parity, MAIN.replace(
            '        std::cout << "[RASBERY][GPU_FULL] {";\n'
            "        rasbery::gpufull::appendReceiptFields(std::cout);\n"
            '        std::cout << "}" << std::endl;\n'
            "        if (rasbery::rasberyGpuXsReconEnabled())", "        if (rasbery::rasberyGpuXsReconEnabled())", 1))
control("check_isolation_check misses a digest comparison replaced by a keff one",
        check_isolation_check,
        SERVER.replace("recheck.digest != receipts[u0].digest", "false"))

# The state scanner's negative control needs a source, not the tree: inject one
# unclassified mutable static and require the scan to surface it.
injected = scan_process_state(
    "src/Driver.h", "static double leaked_from_the_last_case = 0.0;\n")
if not injected or injected[0] in CLASSIFIED:
    negative.append("scan_process_state does not surface a new mutable process-lifetime static")
if scan_process_state("src/Driver.h", "static const double frozen = 0.0;\n"):
    negative.append("scan_process_state flags a `static const`, which cannot leak")
if scan_process_state("src/Driver.h",
                      "static std::atomic<int> counter{0};\n"):
    negative.append("scan_process_state flags an atomic receipt counter")
if scan_process_state("src/Driver.h", "    static Ledger instance;\n    return instance;\n"):
    negative.append("scan_process_state flags a singleton accessor")

if negative:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are therefore "
                    "comments:\n    " + "\n    ".join(negative))

if failures:
    raise SystemExit("evaluator contract: FAIL\n  " + "\n  ".join(failures))
print(f"evaluator contract: PASS ({len(CLASSIFIED)} classified process-lifetime statics)")
