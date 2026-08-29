#!/usr/bin/env python3
"""Contract: RASBERY_STATEPOINT_TELEMETRY may observe the solve, never move it.

WHAT THIS PROTECTS.  Instrumentation that changes the iteration is worse than no
instrumentation, because every number it reports is then a number about a
different run.  The A2 campaign hit exactly that doubt: on 238 a candidate was
reported at 3,114 outers plain and 4,393 with the telemetry on, and the two logs
could not settle whether the telemetry had moved the solve or the two runs had
been given different environments -- the per-statepoint line carries a wall time,
so a plain `diff` is useless, and each feature prints a receipt only when it is
already on, so "which arm was this" had no single answer in the log.

So there are two halves here.

SOURCE HALF (default).  Pin the property in the source: the telemetry flag has
exactly one reader, its two consumers are a clock scope and a print gate, and
nothing inside either writes solver state.  A new `if (sp_telem)` that touched an
iteration variable would be caught here, at the edit, rather than after a
campaign of measurements taken on the wrong trajectory.

RUN HALF (`--compare A.log B.log`).  The binary now prints one
`[RASBERY][TRAJECTORY]` line per run, always, carrying a fold of the published
per-statepoint scalars plus the raw value of every arm knob -- and the telemetry
flag BESIDE the fold rather than inside it.  Two runs are telemetry-neutral when
their `env` and `digest` agree and their `telemetry` differs.  That is a gate the
238 runner can run in one command, and it distinguishes the two explanations the
238 observation could not.

USAGE
    tools/test_telemetry_neutrality.py                        # source contract
    tools/test_telemetry_neutrality.py --compare OFF.log TEL.log
"""

import argparse
import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DRIVER = (SRC / "Driver.h").read_text(errors="replace")

FAILED = []

TELEM_ENV = "RASBERY_STATEPOINT_TELEMETRY"
RECEIPT = "[RASBERY][TRAJECTORY]"


def fail(msg):
    FAILED.append(msg)


def region(text, start, end, what):
    i = text.find(start)
    if i < 0:
        fail(f"cannot find the {what} region (looking for {start!r})")
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        fail(f"the {what} region is not closed (looking for {end!r})")
        return ""
    return text[i:j + len(end)]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ===========================================================================
# SOURCE HALF
# ===========================================================================
def source_contract():
    code = strip_comments(DRIVER)

    # ---------------------------------------------------------------- 1. one
    # reader.  The flag is resolved once, in sptelem::enabled(); anything else
    # reading the variable could resolve it differently (or later, after a
    # setenv) and give two halves of one run two different answers.
    if DRIVER.count(f'std::getenv("{TELEM_ENV}")') != 1:
        fail(f"{TELEM_ENV} must be read from exactly one site (sptelem::enabled)")
    for path in sorted(SRC.glob("*")):
        if not path.is_file() or path.name == "Driver.h":
            continue
        if TELEM_ENV in path.read_text(errors="replace"):
            fail(f"{path.name} reads {TELEM_ENV}; the flag has one owner, Driver.h")

    # ------------------------------------------------- 2. two consumers, named
    # A third consumer is not forbidden because it is wrong -- it is forbidden
    # because it has not been argued.  Adding one means adding it here with the
    # reason it cannot move the iteration.
    calls = [m.start() for m in re.finditer(r"sptelem::enabled\(\)", code)]
    definition = code.find("inline bool enabled() {")
    calls = [c for c in calls if c != definition]
    if len(calls) != 2:
        fail(f"sptelem::enabled() has {len(calls)} consumers, expected 2 "
             "(outer_timing::Scope's clock, Drive()'s print gate)")
    if "_local  = sptelem::enabled();" not in code:
        fail("outer_timing::Scope no longer takes its per-thread arm from sptelem::enabled()")
    if "const bool sp_telem = sptelem::enabled();" not in code:
        fail("Drive() no longer resolves the telemetry gate once into sp_telem")

    # ------------------------------------------ 3. the clock scope stays a clock
    # Scope is the one telemetry consumer that runs INSIDE the outer.  It may
    # read the clock and add to two accumulators; anything else it touched would
    # be in the iteration's path.
    scope = region(code, "class Scope {", "};", "outer_timing::Scope")
    for banned in ("ctx.", "eigv", "residual", "prev_inner", "clean_iters",
                   "flux_stall", "getenv", "cuda", "std::cout"):
        if banned in scope:
            fail(f"outer_timing::Scope touches {banned!r}; it may only read the "
                 "clock and add to the two phase accumulators")
    for required in ("_global = enabled();", "_local  = sptelem::enabled();",
                     "if (!_global && !_local) return;",
                     "if (_local) sptelem::phaseWall()[_phase] += dt;"):
        if required not in scope:
            fail(f"outer_timing::Scope lost {required!r}")

    # ------------------------------- 4. every sp_telem block is write-free
    # The gate may build receipt keys, snapshot backend counters and format
    # lines.  It may not assign to anything the solve reads back.
    drive = region(code, "int Drive() {", "\n};\n", "Drive")
    allowed_lhs = re.compile(
        r"^(ctx\.telemetry\.[A-Za-z_0-9\[\].]*|sp_job_id|sp_slot|sp_run\.[A-Za-z_0-9]*)$")
    for m in re.finditer(r"if \(sp_telem\) \{", drive):
        depth = 0
        i = m.end() - 1
        j = i
        while j < len(drive):
            if drive[j] == "{":
                depth += 1
            elif drive[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = drive[i:j]
        for line in block.splitlines():
            stripped = line.strip()
            assign = re.match(r"^([A-Za-z_][A-Za-z_0-9\[\].]*)\s*=[^=]", stripped)
            if assign is None:
                continue
            lhs = assign.group(1)
            # A declaration inside the block owns its own name and dies with it.
            if re.match(r"^(const |auto |double |int |bool |std::)", stripped):
                continue
            if not allowed_lhs.match(lhs):
                fail(f"an `if (sp_telem)` block assigns to {lhs!r}; the telemetry "
                     "gate must not write anything the solve reads back")

    # --------------------------------------- 5. the receipt is not itself gated
    # A receipt that only appears when the telemetry is on cannot be the thing
    # that proves the telemetry changed nothing.  Pinned by indentation: eight
    # spaces is Drive()'s own body, so the statement is inside no block.
    if DRIVER.count(RECEIPT) != 1:
        fail("the trajectory receipt must be emitted from exactly one site")
    if f'        std::cout << std::format(\n            "{RECEIPT}' not in DRIVER:
        fail("the trajectory receipt is indented into a block; it must be "
             "unconditional at Drive()'s body level")
    if "trajectory::Digest sp_traj;" not in code:
        fail("Drive() does not carry the trajectory fold")
    if code.count("sp_traj.step(") != 1:
        fail("the trajectory fold must be advanced from exactly one site")
    step_line = [ln for ln in DRIVER.splitlines() if "sp_traj.step(" in ln][0]
    if not step_line.startswith(" " * 12) or step_line.strip().startswith("if"):
        fail("the trajectory fold is advanced conditionally; every statepoint counts")

    # ------------------------------------- 6. the digest excludes the flag itself
    digest = region(code, "struct Digest {", "};", "trajectory::Digest")
    if digest.count("mix(h,") != 6:
        fail("trajectory::Digest folds a different number of scalars than the "
             "six the receipt documents (step, outers, th, efpd, eigv, ppm)")
    for banned in ("sp_telem", "telemetry", "seconds", "wall", "clock"):
        if banned in digest:
            fail(f"trajectory::Digest folds {banned!r}; a digest that moved with "
                 "the instrumentation or the clock could never prove neutrality")
    arm = region(code, "inline constexpr const char* kArmEnv[] = {", "};", "kArmEnv")
    if TELEM_ENV in arm:
        fail(f"{TELEM_ENV} is in kArmEnv; it must stay in its own field so an "
             "arm comparison can hold every other knob equal")
    for knob in ("RASBERY_STAGED_FLUX_TOL", "RASBERY_STAGED_XE_TOL",
                 "RASBERY_STAGED_LOOSE_SETTLE", "RASBERY_GPU_XE",
                 "RASBERY_GPU_OUTER", "RASBERY_XE_ANDERSON"):
        if knob not in arm:
            fail(f"{knob} moves a trajectory but is not in the arm receipt")
    # Raw, unparsed: a receipt that re-derived a multiplier could disagree with
    # SolveLoop about what the run was asked for.
    env_json = region(code, "inline std::string armEnvJson() {", "\n}", "armEnvJson")
    for banned in ("atof", "atoi", "stod", "stoi"):
        if banned in env_json:
            fail(f"armEnvJson parses the environment ({banned}); it must report "
                 "the raw string so it cannot drift from the solver's reading")


# ===========================================================================
# RUN HALF
# ===========================================================================
RECEIPT_RE = re.compile(r"\[RASBERY\]\[TRAJECTORY\]\s+(\{.*\})\s*$")


def read_receipt(path):
    """The one trajectory receipt in a run log, as a dict."""
    found = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = RECEIPT_RE.search(line)
        if m:
            found.append(json.loads(m.group(1)))
    if not found:
        raise SystemExit(f"error: {path} carries no {RECEIPT} line -- it was "
                         "produced by a binary older than this contract")
    if len(found) > 1:
        raise SystemExit(f"error: {path} carries {len(found)} trajectory receipts; "
                         "this comparison is for single-deck runs (batch logs "
                         "carry one per slot and must be split by slot first)")
    return found[0]


def compare(off_log, tel_log):
    a = read_receipt(off_log)
    b = read_receipt(tel_log)
    problems = []

    if a["env"] != b["env"]:
        diff = sorted(k for k in set(a["env"]) | set(b["env"])
                      if a["env"].get(k) != b["env"].get(k))
        for k in diff:
            problems.append(f"arm knob {k}: {a['env'].get(k)!r} vs {b['env'].get(k)!r}")
        problems.append("the two runs were given DIFFERENT ARMS -- this is not a "
                        "telemetry result, it is an environment difference")
    if a["telemetry"] == b["telemetry"]:
        problems.append(f"both runs have telemetry={a['telemetry']}; the comparison "
                        "needs one with it on and one with it off")
    for field in ("statepoints", "outers", "th_updates", "digest"):
        if a[field] != b[field]:
            problems.append(f"{field}: {a[field]} vs {b[field]}")

    print(f"  {Path(off_log).name}: telemetry={a['telemetry']} "
          f"statepoints={a['statepoints']} outers={a['outers']} digest={a['digest']}")
    print(f"  {Path(tel_log).name}: telemetry={b['telemetry']} "
          f"statepoints={b['statepoints']} outers={b['outers']} digest={b['digest']}")
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        print("telemetry neutrality: FAIL")
        return 1
    print("telemetry neutrality: PASS (same arm, same trajectory, telemetry differs)")
    return 0


def self_test():
    """The comparator must fail on each way the property can break."""
    import tempfile

    def log(env, telem, digest, outers=4614):
        payload = {"schema_version": 1, "slot": -1, "statepoints": 36,
                   "outers": outers, "th_updates": 100, "digest": digest,
                   "telemetry": telem, "env": env}
        fd = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        fd.write("  NO.=   1 ...\n")
        fd.write(f"{RECEIPT} {json.dumps(payload)}\n")
        fd.close()
        return fd.name

    base_env = {"RASBERY_GPU_XE": "1", "RASBERY_STAGED_FLUX_TOL": "50"}
    good = (log(base_env, 0, "0123456789abcdef"),
            log(base_env, 1, "0123456789abcdef"))
    if compare(*good) != 0:
        fail("the comparator rejects a genuinely neutral pair")
    moved = (log(base_env, 0, "0123456789abcdef", outers=3114),
             log(base_env, 1, "fedcba9876543210", outers=4393))
    if compare(*moved) == 0:
        fail("the comparator accepts a pair whose trajectory moved")
    drifted_env = dict(base_env)
    drifted_env["RASBERY_GPU_XE"] = None
    if compare(log(base_env, 0, "0123456789abcdef"),
               log(drifted_env, 1, "fedcba9876543210", outers=4393)) == 0:
        fail("the comparator accepts a pair that ran different arms")
    if compare(log(base_env, 1, "0123456789abcdef"),
               log(base_env, 1, "0123456789abcdef")) == 0:
        fail("the comparator accepts two runs that both had telemetry on")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", nargs=2, metavar=("OFF_LOG", "TELEMETRY_LOG"),
                    help="two run logs to compare through their trajectory receipts")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)

    source_contract()
    print("-- comparator self-test --")
    self_test()
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if FAILED:
        for f in FAILED:
            print(f"FAIL: {f}")
        print(f"telemetry neutrality contract: {len(FAILED)} FAILURE(S)")
        return 1
    print("telemetry neutrality contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
