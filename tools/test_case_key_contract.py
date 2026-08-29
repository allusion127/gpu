#!/usr/bin/env python3
"""Contract: the canonical case key means one thing (WP10.1).

There are two implementations of this key -- src/CaseKey.h, which the solver
prints, and tools/case_key.py, which a GA controller uses to decide NOT to run
the solver.  Two implementations of one key is how a cache starts serving the
wrong answer, so this holds them to one definition.

THREE HALVES, because only two of them can run here.

SOURCE HALF (default).  The parts that must be identical are compared as
source: the arm-environment list and its ORDER, the fidelity table, the payload
field order, and the canonical token grammar.  There is no C++ compiler on the
authoring host, so these are static cross-checks -- but they are the checks that
actually catch drift, because drift happens when someone adds a knob to one list.

BEHAVIOUR HALF (default).  The python implementation is exercised on fixtures
for the properties the key is FOR: symmetric-equivalent loading patterns hash
equal; a genuinely different pattern does not; a half core is not folded; and
every provenance field is load-bearing (change it, and the key changes).  Each
one is a negative control as much as a check.

LIVE HALF (`--compare RUN.log DECK.json`).  The one comparison that closes the
loop between the two implementations, and it needs a real run: the binary prints
`[RASBERY][CASE]` with the key it used, and this asserts tools/case_key.py
computes the same 64 hex digits from the same deck under the same environment.
That is a 238 gate, listed in the WP9/WP10 runbook.

USAGE
    tools/test_case_key_contract.py
    tools/test_case_key_contract.py --compare run.log deck.json
"""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import case_key  # noqa: E402

CASEKEY_H = (ROOT / "src" / "CaseKey.h").read_text(encoding="utf-8-sig")
DRIVER_H = (ROOT / "src" / "Driver.h").read_text(encoding="utf-8-sig")
RUNCONTRACT_H = (ROOT / "src" / "RunContract.h").read_text(encoding="utf-8-sig")
IO_CPP = (ROOT / "src" / "IO.cpp").read_text(encoding="utf-8-sig")
BLR_H = (ROOT / "include" / "chiffon" / "BatchLightResult.h").read_text(encoding="utf-8-sig")
EVAL_H = (ROOT / "src" / "EvaluatorServer.h").read_text(encoding="utf-8-sig")

FAILED: list[str] = []


def fail(message: str) -> None:
    FAILED.append(message)


# ---------------------------------------------------------------------------
# SOURCE HALF
# ---------------------------------------------------------------------------
def source_contract() -> None:
    # 1. The arm-environment list, and its ORDER, which is part of the payload.
    #    There is exactly ONE list -- Driver.h's -- and case_key.py parses it
    #    rather than copying it, so this checks the PARSE rather than a copy: it
    #    has to find the real array, get every entry, and fail closed if the
    #    array ever moves or is renamed.  A tool that silently fell back to a
    #    stale list would be a tool that keys a cache on last month's physics.
    block = DRIVER_H[DRIVER_H.index("inline constexpr const char* kArmEnv[] = {"):]
    block = block[:block.index("};")]
    cpp_env = re.findall(r'"([A-Z_0-9]+)"', block)
    if cpp_env != case_key.ARM_ENV:
        fail("case_key.py did not parse Driver.h kArmEnv exactly "
             f"(C++ {len(cpp_env)} entries, tool {len(case_key.ARM_ENV)})")
    if len(cpp_env) < 20 or "RASBERY_GPU" not in cpp_env:
        fail(f"the kArmEnv parse looks wrong: {cpp_env[:4]}...")
    if "ARM_ENV = _read_arm_env()" not in (ROOT / "tools" / "case_key.py").read_text(
            encoding="utf-8"):
        fail("case_key.py carries its own copy of the arm-knob list again")
    with tempfile.TemporaryDirectory() as raw:
        empty = Path(raw) / "Driver.h"
        empty.write_text("nothing here", encoding="utf-8")
        try:
            case_key._read_arm_env(empty)
            fail("case_key._read_arm_env accepted a Driver.h with no kArmEnv; it "
                 "must fail closed rather than key on an empty knob list")
        except SystemExit:
            pass

    # 2. The fidelity table, in rank order: the key spells the fidelity, and a
    #    receipt that spelled it differently would key a different cache entry.
    traits = RUNCONTRACT_H[RUNCONTRACT_H.index("kFidelityTraits[4] = {"):]
    traits = traits[:traits.index("};")]
    cpp_traits = [(m[0], m[1]) for m in re.findall(r'\{"([a-zA-Z0-9_]+)",\s*"([a-z0-9_]+)"',
                                                   traits)]
    if cpp_traits != case_key.FIDELITY_TRAITS:
        fail(f"fidelity table drift: C++ {cpp_traits} vs python {case_key.FIDELITY_TRAITS}")

    # 3. The payload's field order.  Both sides build a line-oriented payload;
    #    the same fields in a different order is a different digest.
    payload_fn = CASEKEY_H[CASEKEY_H.index("inline std::string payloadOf("):]
    payload_fn = payload_fn[:payload_fn.index("\ninline std::string keyOf")]
    cpp_fields = re.findall(r'out \+= "\\n([a-z_]+)\\t"', payload_fn)
    py_fields = [line.split("\t")[0] for line in
                 case_key.payload_of("d", "full_exact", "strict", {}, "x", "")
                 .splitlines()[1:]]
    py_order, seen = [], set()
    for name in py_fields:
        if name not in seen:
            seen.add(name)
            py_order.append(name)
    if cpp_fields != py_order:
        fail(f"payload field order drift: C++ {cpp_fields} vs python {py_order}")

    # 4. The schema string, which is the payload's first line on both sides.
    schema = re.search(r'kSchema = "([^"]+)"', CASEKEY_H)
    if schema is None or schema.group(1) != case_key.SCHEMA:
        fail("kSchema in CaseKey.h does not match SCHEMA in case_key.py")
    if not case_key.payload_of("d", "f", "p", {}, "x", "").startswith(case_key.SCHEMA):
        fail("the payload does not start with its schema line")

    # 5. The token grammar.  One spelling per value on both sides; the float
    #    spelling in particular, which is the one that silently differs between
    #    any two JSON serialisers.
    if '"{:.17g}"' not in CASEKEY_H:
        fail("CaseKey.h no longer spells floats through {:.17g}; python uses %.17g "
             "and the two would drift on the first non-integer deck value")
    for token in ("out += '~';", "out += value.get<bool>() ? 'T' : 'F';",
                  "std::sort(keys.begin(), keys.end());"):
        if token not in CASEKEY_H:
            fail(f"CaseKey.h lost the canonical token rule {token!r}")

    # 6. The legal symmetry operations.  A fold that has not been argued can
    #    merge two DIFFERENT cores into one cache entry, so the two orbit
    #    definitions have to name the same operations.
    orbit = CASEKEY_H[CASEKEY_H.index("inline CoreCanon canonicalCore("):]
    orbit = orbit[:orbit.index("\n// ---")]
    cpp_ops = set(re.findall(r'orbit\.emplace_back\("([a-z0-9_]+)"', orbit))
    py_ops = set()
    for symang in (90, 360):
        for rows in (3, 4):
            core = [[f"{r}{c}" for c in range(4)] for r in range(rows)]
            py_ops.add(case_key.canonical_core(core, symang)[0])
    py_src = (ROOT / "tools" / "case_key.py").read_text(encoding="utf-8")
    py_declared = set(re.findall(r'orbit\.append\(\("([a-z0-9_]+)"', py_src))
    py_declared.add("identity")
    if cpp_ops != py_declared:
        fail(f"symmetry orbit drift: C++ {sorted(cpp_ops)} vs python {sorted(py_declared)}")

    # 7. The key is computed and published UNCONDITIONALLY -- a key only some
    #    runs carry is a key no cache can trust -- and it is folded where the
    #    parsed deck is live.
    if "casekey::deckPayload(config, &_deck_key_core_op)" not in IO_CPP:
        fail("IO::ReadInput does not fold the deck half of the case key")
    if DRIVER_H.count('"  [RASBERY][CASE] {{') != 1:
        fail("the [RASBERY][CASE] receipt must be emitted from exactly one site")
    case_site = DRIVER_H.index('"  [RASBERY][CASE] {{')
    guard = DRIVER_H.rfind("if (sp_telem) {", 0, case_site)
    close = DRIVER_H.rfind("\n        }", 0, case_site)
    if guard > close:
        fail("the [RASBERY][CASE] receipt is inside a telemetry gate; it must be "
             "unconditional")
    if '"case_key"' not in BLR_H:
        fail("the light JSONL does not carry case_key")
    if '",\\"case_key\\":"' not in EVAL_H.replace('\\"', '\\"'):
        if '\\"case_key\\":' not in EVAL_H:
            fail("the evaluator per-case report does not echo case_key")
    if 'object.contains("key")' not in EVAL_H:
        fail("the evaluator no longer accepts the client's optional opaque `key`")

    # 8. ONE hash implementation.  Two copies of SHA-256 is how a cache key ends
    #    up disagreeing with the receipt that named it.
    if "0x428a2f98u" in BLR_H:
        fail("BatchLightResult.h carries its own SHA-256 transform again; there "
             "must be exactly one (include/chiffon/Sha256.h)")
    sha_h = (ROOT / "include" / "chiffon" / "Sha256.h").read_text(encoding="utf-8-sig")
    if sha_h.count("0x428a2f98u") != 1:
        fail("Sha256.h does not carry exactly one K table")


# ---------------------------------------------------------------------------
# BEHAVIOUR HALF
# ---------------------------------------------------------------------------
# NOT diagonally symmetric -- deliberately.  A pattern that equals its own
# transpose would make the "symmetric-equivalent decks hash equal" check pass
# without the fold doing anything, which is the way this test could lie.
QUARTER = [
    ["A0", "B2", "C1", "C1"],
    ["B1", "A0", "B1", "C1"],
    ["A0", "B1", "C1", "R1"],
    ["C1", "C1", "R1", "XX"],
]
assert QUARTER != [list(r) for r in zip(*QUARTER)], "the fixture is self-transpose"
# The transpose of QUARTER: the SAME core, reflected in its diagonal.
QUARTER_T = [list(row) for row in zip(*QUARTER)]
# One assembly swapped: a genuinely different core.
QUARTER_DIFF = [list(r) for r in QUARTER]
QUARTER_DIFF[0][1] = "C1"


def deck(core, symang=90, mirror=True, extra=None):
    d = {
        "data": {"cross-section": "lib.h5"},
        "geometry": {
            "dimensions": {"ng": 2, "xydivision": 2, "npins": 17},
            "size": {"hx": 21.5, "hy": 21.5, "hz": [{"height": 30.0, "node": 4}]},
            "symmetry": {"angle": symang, "mirror": mirror,
                         "center assembly divided": True},
            "albedo": {"west": 0.0, "east": 0.5, "north": 0.0, "south": 0.5,
                       "bottom": 0.5, "up": 0.5},
        },
        "core": core,
        "batch": {"A0": [{"id": "RT", "count": 1}, {"id": "A0", "count": 25}],
                  "B1": [{"id": "B1", "count": 27}]},
        "schedule": [{"type": "depletion", "time": 1.0e-3, "power": 100.0}],
    }
    if extra:
        d.update(extra)
    return d


def write(tmp: Path, name: str, config) -> Path:
    path = tmp / name
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    return path


CLEAN_ENV = {"PATH": ""}


def key_of(path: Path, env=None, **kw) -> str:
    return case_key.case_key(path, env=dict(env or CLEAN_ENV), xslib=False, **kw)["case_key"]


def behaviour_contract() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        base = write(tmp, "base.json", deck(QUARTER))
        mirrored = write(tmp, "mirror.json", deck(QUARTER_T))
        different = write(tmp, "diff.json", deck(QUARTER_DIFF))

        # THE PROPERTY THE KEY EXISTS FOR.
        if key_of(base) != key_of(mirrored):
            fail("a quarter map and its transpose hash differently; the symmetry "
                 "fold is not working and the cache would miss on every mirror")
        # ... AND ITS NEGATIVE CONTROL, which matters more.
        if key_of(base) == key_of(different):
            fail("two different loading patterns hash equal; the key would serve "
                 "one core's answer for another")
        if case_key.case_key(base, env=CLEAN_ENV, xslib=False)["core_op"] not in (
                "identity", "transpose"):
            fail("a quarter map reported an operation outside its legal orbit")

        # A half core is NOT folded: the fold has not been argued for it, and an
        # unargued fold is exactly how two different cores share a cache entry.
        half = write(tmp, "half.json", deck(QUARTER, symang=180))
        half_t = write(tmp, "half_t.json", deck(QUARTER_T, symang=180))
        if key_of(half) == key_of(half_t):
            fail("a 180-degree map was folded by the transpose; only 90 and 360 "
                 "have an argued orbit")
        if case_key.case_key(half, env=CLEAN_ENV, xslib=False)["core_op"] != "identity":
            fail("a 180-degree map did not report core_op=identity")

        # Everything else in the deck is load-bearing.
        for name, mutation in (
            ("schedule", {"schedule": [{"type": "depletion", "time": 2.0e-3,
                                        "power": 100.0}]}),
            ("batch", {"batch": {"A0": [{"id": "RT", "count": 2}]}}),
            ("albedo", {"geometry": deck(QUARTER)["geometry"] | {
                "albedo": {"west": 0.0, "east": 0.25, "north": 0.0, "south": 0.5,
                           "bottom": 0.5, "up": 0.5}}}),
        ):
            other = write(tmp, f"m_{name}.json", deck(QUARTER, extra=mutation))
            if key_of(base) == key_of(other):
                fail(f"changing the deck's {name} did not change the case key")

        # Key order and whitespace in the deck file must NOT change the key.
        shuffled = deck(QUARTER)
        shuffled = {k: shuffled[k] for k in reversed(list(shuffled.keys()))}
        reordered = tmp / "reordered.json"
        reordered.write_text(json.dumps(shuffled, separators=(",", ":")), encoding="utf-8")
        if key_of(base) != key_of(reordered):
            fail("re-ordering the deck's top-level keys changed the case key; the "
                 "canonical form is not canonical")

        # Provenance beyond the deck.
        if key_of(base) == key_of(base, env=CLEAN_ENV | {"RASBERY_GPU_XE": "1"}):
            fail("an arm knob does not reach the key; two runs with different "
                 "physics would share one cache entry")
        if key_of(base) == key_of(base, env=CLEAN_ENV | {"RASBERY_STAGED_FLUX_TOL": "4"}):
            fail("the staged-tolerance A2 fidelity does not reach the key")
        if key_of(base) == key_of(base, warm_start="parent:abc"):
            fail("warm-start provenance does not reach the key; a warm start can "
                 "pick a root (GA evaluator plan Sec 5.4)")
        if key_of(base) == key_of(base, env=CLEAN_ENV | {"RASBERY_CODE_SHA": "deadbeef"}):
            fail("the declared code identity does not reach the key")
        # A knob that is NOT trajectory-affecting must not.
        if key_of(base) != key_of(base, env=CLEAN_ENV | {"RASBERY_STATEPOINT_TELEMETRY": "1"}):
            fail("the telemetry flag reached the key; instrumentation is not physics")

        # The library digest is CONTENT, not path.
        lib_a = tmp / "lib.h5"
        lib_a.write_bytes(b"library-one")
        with_lib = case_key.case_key(base, env=dict(CLEAN_ENV), xslib=True)
        lib_a.write_bytes(b"library-two")
        with_lib2 = case_key.case_key(base, env=dict(CLEAN_ENV), xslib=True)
        if with_lib["case_key"] == with_lib2["case_key"]:
            fail("rewriting the cross-section library did not change the key")
        if with_lib["case_key"] == key_of(base):
            fail("the library digest is not part of the key")


# ---------------------------------------------------------------------------
# LIVE HALF
# ---------------------------------------------------------------------------
def compare(log: Path, deck_path: Path) -> int:
    text = log.read_text(errors="replace")
    found = None
    for line in text.splitlines():
        i = line.find("[RASBERY][CASE]")
        if i < 0:
            continue
        found = json.loads(line[line.index("{", i):])
    if found is None:
        print(f"  FAIL no [RASBERY][CASE] receipt in {log}")
        return 1
    mine = case_key.case_key(deck_path)
    problems = []
    for field in ("case_key", "key_schema", "core_op", "deck_digest"):
        want = mine[field] if field != "deck_digest" else mine["deck_digest"]
        if found.get(field) != want:
            problems.append(f"{field}: solver {found.get(field)!r} vs tool {want!r}")
    for line in problems:
        print(f"  FAIL {line}")
    print(f"  solver case_key {found.get('case_key')}")
    print(f"  tool   case_key {mine['case_key']}")
    print("case key live contract:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("RUN.log", "DECK.json"), default=None)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    source_contract()
    behaviour_contract()
    py_compile.compile(str(ROOT / "tools" / "case_key.py"), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if FAILED:
        for message in FAILED:
            print(f"case key contract: FAIL: {message}")
        return 1
    print("case key contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
