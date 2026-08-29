#!/usr/bin/env python3
"""Contract: the warm start is off by default, N1 when on, and never wrong (WP10.2).

WHAT IS BEING PROTECTED, IN ORDER OF HOW BADLY IT COULD GO.

  1. FEATURE-OFF IDENTITY.  With neither `--warm-start-from` nor
     `--save-warm-state`, nothing in the warm-start path may run -- no file is
     opened, no flux is written, and not even a receipt line is printed.  A run
     of this build with the flags unset has to be the run of a build without
     them.
  2. EVERY REFUSAL IS A COLD START.  A missing file, a foreign magic, a version
     bump, a geometry that does not match, an implausible seed: each one leaves
     the cold flux in place and says which in the receipt.  A warm start that
     cannot be honoured must never become a WRONG one, and the way that defect
     would arrive is an exception or a partial application, so both are checked.
  3. THE GATE IS DECLARED N1, NOT B0.  A different starting point can select a
     different root where the Xe<->flux map has more than one (A2_OUTER_REDUCTION
     Sec 5's i-SMR CY02 precedent).  So the digest is ALLOWED to move, the
     acceptance gate is keff/CBC/Fq/FdH, and the warm-start provenance is folded
     into the case key so a warm answer never lands in a cold cache entry.
  4. IT IS SEEDED WHERE IT CAN HELP AND SAVED WHERE IT IS CHEAP.  Loaded after
     the cold reset and before the first solve; saved at the FIRST statepoint
     only, because every later one already starts from the previous statepoint's
     converged flux.

THREE MODES.

    tools/test_warm_start_contract.py                     source contract
    tools/test_warm_start_contract.py --inspect FILE.warm read a saved state
    tools/test_warm_start_contract.py --compare COLD.log WARM.log
                                                          the 238 A/B verdict
"""
from __future__ import annotations

import argparse
import json
import py_compile
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
WARM_H = (ROOT / "src" / "WarmState.h").read_text(encoding="utf-8-sig")
MAIN = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8-sig")
EVAL = (ROOT / "src" / "EvaluatorServer.h").read_text(encoding="utf-8-sig")

FAILED: list[str] = []

MAGIC = b"RASBWRM1"
# src/WarmState.h save(): magic, version, ng, nxyz, nx, ny, nz, keff, boron, efpd
HEADER = struct.Struct("<8s6I3d")


def fail(message: str) -> None:
    FAILED.append(message)


def strip_comments(text: str) -> str:
    """Code only.  A comment that NAMES a thing is the opposite of doing it --
    WarmState.h's header explains at length why it does not carry burnup or the
    isotope inventory, and a scan that read the explanation as the deed would
    make careful documentation into a test failure."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


WARM_CODE = strip_comments(WARM_H)


def region(text: str, start: str, end: str, what: str) -> str:
    i = text.find(start)
    if i < 0:
        fail(f"{what}: anchor not found: {start!r}")
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        fail(f"{what}: closing anchor not found: {end!r}")
        return ""
    return text[i:j]


def source_contract() -> None:
    # ---------------------------------------------------------------- 1. off
    # Both halves hang off an empty string, and both guards are the ONLY way in.
    load_block = region(DRIVER, "        std::string      warm_provenance;",
                        "        if (is_restart_run)", "the warm-start load block")
    if "if (!_warm_start_from.empty()) {" not in load_block:
        fail("the warm-start load is not gated on a non-empty _warm_start_from")
    for call in ("warmstate::load(", "warmstate::digest("):
        if load_block.count(call) != 1:
            fail(f"{call} appears {load_block.count(call)} times in the load block; "
                 "it must be reachable only inside the arm's own guard")
    if DRIVER.count("warmstate::save(") != 1:
        fail("warmstate::save must be called from exactly one site")
    save_site = DRIVER.index("warmstate::save(")
    guard = DRIVER.rfind("if (!_warm_state_out.empty() && !warm_saved) {", 0, save_site)
    if guard < 0 or DRIVER.find("}", guard) > save_site + 4000:
        fail("the warm-state save is not gated on a non-empty _warm_state_out "
             "plus the once-per-run latch")
    receipt = DRIVER.find('"  [RASBERY][WARMSTART] {{')
    if receipt < 0:
        fail("no [RASBERY][WARMSTART] receipt")
    else:
        pre = DRIVER.rfind('if (warm_status != "off" || warm_save_status != "off") {', 0,
                           receipt)
        if pre < 0 or receipt - pre > 2000:
            fail("the [RASBERY][WARMSTART] receipt is not gated on the arm being on; "
                 "feature-off must leave the log unchanged too")
    # The warm start touches the flux through ONE writer, after the cold reset.
    reset = DRIVER.find("cross_sections.ResetFluxAndCurrents(1.0);")
    copy = DRIVER.find("geometry.PhifMutable());")
    solve = DRIVER.find("SolveLoop(ctx, eigv, schedule, total_outer")
    if not (0 <= reset < copy < solve):
        fail("the warm flux is not written between the cold reset and the first solve")

    # -------------------------------------------------- 2. refusals are cold
    for reason in ("not a warm-state file", "truncated header", "truncated flux",
                   "implausible flux size"):
        if reason not in WARM_H:
            fail(f"WarmState.h does not refuse {reason!r}")
    if "throw" in WARM_CODE:
        fail("WarmState.h throws; a warm start that cannot be honoured must degrade "
             "to a cold start, not fail the case")
    for reason in ("geometry mismatch", "implausible seed"):
        if reason not in load_block:
            fail(f"the load block does not refuse on {reason!r}")
    if 'warm_status = "cold_fallback";' not in load_block:
        fail("a refused warm start does not report cold_fallback")
    if 'if (warm_status != "applied") warm_provenance.clear();' not in load_block:
        fail("a refused warm start still contributes provenance to the case key; a "
             "cold run must key like a cold run")

    # ------------------------------------------------------------ 3. N1 gate
    if '"gate\\":\\"N1' not in DRIVER.replace('\\"', '\\"'):
        if '\\"gate\\":\\"N1\\"' not in DRIVER:
            fail("the warm-start receipt does not declare its gate class")
    if "p.warm_start = warm_provenance;" not in DRIVER:
        fail("warm-start provenance does not reach the case key")
    if "warm_start" not in (ROOT / "src" / "CaseKey.h").read_text(encoding="utf-8-sig"):
        fail("the case key payload has no warm-start field")
    # The receipt carries the number the A/B is decided on, and it is tallied
    # OUTSIDE the telemetry gate -- a lever measurable only on a telemetry run
    # cannot be measured on the wall-timing run it is meant to shorten.
    tally = DRIVER.find(
        "warm_initial_outers += ctx.telemetry.outers_by_cause[sptelem::CAUSE_INITIAL];")
    if tally < 0:
        fail("initial_outers is not tallied for the warm-start receipt")
    else:
        gate = DRIVER.rfind("if (sp_telem) {", 0, tally)
        close = DRIVER.rfind("\n            }", 0, tally)
        if gate > close:
            fail("initial_outers is tallied inside the telemetry gate; the A/B has to "
                 "be readable on a wall-timing run")

    # --------------------------------------------- 4. seeded once, at the BOC
    if "// FIRST STATEPOINT ONLY" not in DRIVER and "!warm_saved" not in DRIVER:
        fail("the warm state is not saved once per run")
    for field in ("keff", "boron", "efpd", "flux"):
        if f"out.{field}" not in DRIVER and f"{field};" not in WARM_CODE:
            fail(f"the saved warm state has no {field}")
    # It carries a GUESS and never an input.  Burnup and the isotope inventory
    # belong to the fuel, and for a different loading pattern they are the wrong
    # fuel, not a warm start.
    for banned in ("burnup", "isotope", "_iden"):
        if banned in WARM_CODE.lower():
            fail(f"WarmState carries {banned!r}; it must carry only quantities the "
                 "solve overwrites")

    # ------------------------------------------------------- 5. the API surface
    for flag in ("--warm-start-from", "--save-warm-state"):
        if MAIN.count(f'option == "{flag}"') != 1:
            fail(f"{flag} is not parsed exactly once")
        if flag not in region(MAIN, 'std::cout << "Usage:', "return 0;", "the help text"):
            fail(f"{flag} is not documented in --help")
    if "--save-warm-state with an explicit path takes exactly one deck" not in MAIN:
        fail("an explicit --save-warm-state path with several decks is not refused; "
             "N decks would write one file and the child would seed from a stranger")
    if MAIN.count("driver.setWarmStart(") != 2:
        fail("setWarmStart is not applied on both main() paths (batch and single)")
    for field in ("warm_start_from", "save_warm_state"):
        if f'"{field}"' not in EVAL:
            fail(f"the evaluator does not accept a per-case {field}")
    blr = (ROOT / "include" / "chiffon" / "BatchLightResult.h").read_text(
        encoding="utf-8-sig")
    if '"warm_state"' not in blr:
        fail("the light JSONL does not name the warm state this case saved; a GA "
             "would have to grep a log to seed the next generation")
    if "warm_saved ? _warm_state_out : std::string()" not in DRIVER:
        fail("the light JSONL names a warm state that may not have been written")
    if "driver.setWarmStart(warm_from, warm_save);" not in EVAL:
        fail("the evaluator parses the warm-start fields but does not apply them")
    # `save_warm_state` empty on the recheck.  WP10.3 put the case's resolved
    # fidelity between it and the status out-parameter -- the recheck must run
    # at the SAME fidelity or its digest has no reason to match -- so the match
    # is on the empty save argument itself rather than on its neighbour.
    if not re.search(r"std::string\(\),\s*jobs\.fidelity\[u0\],\s*recheck_status,", EVAL):
        fail("the isolation recheck saves a warm state; it would overwrite the parent "
             "state the wave's own first case just produced")


# ---------------------------------------------------------------------------
# --inspect: read a saved warm state, so a runner can see what a parent handed on
# ---------------------------------------------------------------------------
def inspect(path: Path) -> int:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        print(f"  FAIL {path}: shorter than a header")
        return 1
    magic, version, ng, nxyz, nx, ny, nz, keff, boron, efpd = HEADER.unpack_from(raw)
    if magic != MAGIC:
        print(f"  FAIL {path}: magic {magic!r} is not {MAGIC!r}")
        return 1
    want = HEADER.size + 8 * ng * nxyz
    print(json.dumps({
        "path": str(path), "version": version, "ng": ng, "nxyz": nxyz,
        "shape": [nx, ny, nz], "keff": keff, "boron_ppm": boron, "efpd": efpd,
        "bytes": len(raw), "bytes_expected": want,
        "complete": len(raw) == want,
    }, indent=2))
    return 0 if len(raw) == want else 1


# ---------------------------------------------------------------------------
# --compare: the 238 A/B verdict
# ---------------------------------------------------------------------------
def receipts(text: str, tag: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        i = line.find(tag)
        if i < 0:
            continue
        try:
            out.append(json.loads(line[line.index("{", i):]))
        except (ValueError, json.JSONDecodeError):
            continue
    return out


STEP = re.compile(r"NO\.=\s*(\d+)\s+EFPD=\s*([0-9.eE+-]+)\s+K-EFF=([0-9.]+)"
                  r"\s+PPM=\s*([0-9.eE+-]+)\s+outer=\s*(\d+)")


def compare(cold: Path, warm: Path) -> int:
    problems = []
    rows = {}
    for name, path in (("cold", cold), ("warm", warm)):
        text = path.read_text(errors="replace")
        ws = receipts(text, "[RASBERY][WARMSTART]")
        traj = receipts(text, "[RASBERY][TRAJECTORY]")
        steps = STEP.findall(text)
        rows[name] = {
            "warmstart": ws[-1] if ws else None,
            "trajectory": traj[-1] if traj else None,
            "steps": [(int(a), float(b), float(c), float(d), int(e))
                      for a, b, c, d, e in steps],
        }
    cold_ws, warm_ws = rows["cold"]["warmstart"], rows["warm"]["warmstart"]
    if warm_ws is None:
        problems.append("the WARM log has no [RASBERY][WARMSTART] receipt")
    elif warm_ws.get("load") != "applied":
        problems.append(f"the warm arm did not apply its seed: load="
                        f"{warm_ws.get('load')!r} reason={warm_ws.get('reason')!r}")
    if cold_ws is not None and cold_ws.get("load") not in (None, "off"):
        problems.append("the COLD arm carries a warm start; it is not a cold arm")

    def initial(name):
        ws = rows[name]["warmstart"]
        return None if ws is None else ws.get("initial_outers")

    c_init, w_init = initial("cold"), initial("warm")
    print(f"  initial outers   cold {c_init}   warm {w_init}")
    if c_init and w_init is not None:
        print(f"  initial delta    {w_init - c_init:+d}  "
              f"({100.0 * (w_init - c_init) / c_init:+.1f} %)")
    for name in ("cold", "warm"):
        traj = rows[name]["trajectory"]
        print(f"  {name:>4}  outers {traj.get('outers') if traj else '?'}"
              f"  statepoints {traj.get('statepoints') if traj else '?'}"
              f"  digest {traj.get('digest') if traj else '?'}")
    # N1: the digest MAY move.  What must not move is the answer, so the
    # per-statepoint keff and boron are compared against acceptance thresholds
    # rather than for equality.
    cs, wss = rows["cold"]["steps"], rows["warm"]["steps"]
    if len(cs) != len(wss):
        problems.append(f"statepoint counts differ: cold {len(cs)} vs warm {len(wss)}")
    else:
        dk = max((abs(c[2] - w[2]) for c, w in zip(cs, wss)), default=0.0)
        dppm = max((abs(c[3] - w[3]) for c, w in zip(cs, wss)), default=0.0)
        print(f"  max |dk| {dk:.3e} ({dk * 1e5:.2f} pcm)   max |dppm| {dppm:.3f} ppm")
        if dk > 2.0e-5:
            problems.append(f"max |dk| {dk:.3e} exceeds the 2 pcm-scale search "
                            "tolerance; the warm start moved the answer, not just "
                            "the path")
        if dppm > 1.0:
            problems.append(f"max |dppm| {dppm:.3f} exceeds 1 ppm")
    for line in problems:
        print(f"  FAIL {line}")
    print("warm start A/B:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", type=Path, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("COLD.log", "WARM.log"), default=None)
    args = ap.parse_args(argv)
    if args.inspect is not None:
        return inspect(args.inspect)
    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    source_contract()
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if FAILED:
        for message in FAILED:
            print(f"warm start contract: FAIL: {message}")
        return 1
    print("warm start contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
