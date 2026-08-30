#!/usr/bin/env python3
"""Contract: the canonical case key means one thing (WP10.1).

There are two implementations of this key -- src/CaseKey.h, which the solver
prints, and tools/case_key.py, which a GA controller uses to decide NOT to run
the solver.  Two implementations of one key is how a cache starts serving the
wrong answer, so this holds them to one definition.

FOUR HALVES, and how far each one gets.

SOURCE (always).  The parts that must be identical are compared as source: the
arm-environment list and its ORDER, the fidelity table, the payload field order,
the canonical token grammar, and the legal symmetry orbit.  These are the checks
that catch DRIFT, because drift happens when someone adds a knob to one list.

BEHAVIOUR (always).  The python implementation is exercised on fixtures for the
properties the key is FOR: symmetric-equivalent loading patterns hash equal; a
genuinely different pattern does not; a half core is not folded; and every
provenance field is load-bearing (change it, and the key changes) while the
telemetry flag is not.  Each one is a negative control as much as a check.

COMPILED (when a C++ compiler is present; SKIPPED, and said so, otherwise).  The
half that actually closes the loop between the two implementations without
needing a GPU: src/CaseKey.h is compiled into a small harness and BOTH payloads
are compared BYTE FOR BYTE against tools/case_key.py's -- the canonical deck
payload on decks seeded with every float spelling the two languages could
disagree about, and the FULL payload (the env lines, the library digest, the
warm-start token and the build identity) under two environments.  It also runs
three FIPS 180-4 SHA-256 vectors.

LIVE (`--compare RUN.log DECK.json`).  What is left after the compiled half: the
VALUES the payload takes from the environment and from the cross-section library
on a real host.  It compares every component the receipt publishes, in payload
order, so a mismatch names itself.

WHAT 181 COST, AND WHY THE SHAPE CHANGED (2026-08-30).  The live gate failed on
kngr_238.json and passed on short_rev71.json and s1.json, and the only thing it
could print was that two 64-hex digests differed.  Two causes were structural:
the receipt published one component (the deck half), and the library component
-- the only one that varies with the deck once the environment is fixed -- was
not exercised by any fixture here, because every fixture passed `xslib=False`.
Both are closed above: `receipt_component_contract` holds the receipt to the
tool's component list, and `xslib_contract` runs a deck that really does name an
external file.

USAGE
    tools/test_case_key_contract.py
    tools/test_case_key_contract.py --compare run.log deck.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
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
MAIN_CPP = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8-sig")
CASEFIDELITY_H = (ROOT / "src" / "CaseFidelity.h").read_text(encoding="utf-8-sig")

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
    # `value.template get<bool>()`, not `value.get<bool>()`: `value` has a
    # DEPENDENT type inside `template <class Json> appendValue`, and GCC 14.3
    # rejects the undisambiguated spelling that MSVC accepts.  The token asserted
    # here is the one that compiles on 238; see test_dependent_template_contract.py.
    for token in ("out += '~';", "out += value.template get<bool>() ? 'T' : 'F';",
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

    # 9. THE LIBRARY DIGEST MUST BE ABLE TO EXPIRE.  One hash transform, two
    #    caching policies: BatchLightResult::Sha256FileCached memoises by PATH
    #    ALONE and never expires, which was harmless when a process was one
    #    deck and is not now that the evaluator worker (WP8.1.5) is a process
    #    that outlives a library rebuild -- it would keep stamping the digest
    #    of a file that is no longer there.  The key bytes are the same either
    #    way while the file is unchanged; what differs is whether "unchanged"
    #    is ever re-checked.  XsLibraryContentDigest memoises by
    #    (path, size, mtime) and is the digest the library cache itself keys on.
    i = DRIVER_H.find("casekey::Provenance caseKeyProvenance(")
    if i < 0:
        fail("Driver.h no longer has caseKeyProvenance; the case key's provenance "
             "must be assembled in exactly one place")
    else:
        body = DRIVER_H[i:DRIVER_H.find("\n    }", i)]
        # Comments stripped: this function's comment says WHY it is not
        # Sha256FileCached, and a rule that read the reason as the offence
        # would forbid the tree from explaining itself.
        code = re.sub(r"//[^\n]*", "", body)
        if "XsLibraryContentDigest(" not in code:
            fail("caseKeyProvenance does not take xslib_digest from "
                 "XsLibraryContentDigest ((path, size, mtime)-memoised); a digest "
                 "memoised by path alone cannot notice a library rebuild")
        if "Sha256FileCached" in code:
            fail("caseKeyProvenance still reaches for Sha256FileCached, which "
                 "memoises by path alone and never expires")


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


def xslib_contract() -> None:
    """The library half, on a fixture that ACTUALLY REFERENCES AN EXTERNAL FILE.

    THE GAP THIS CLOSES.  Every fixture above passes `xslib=False`, so until now
    the library component was never exercised at all -- and it is the only
    component of the key that varies with the deck once the environment is
    fixed.  That is exactly the shape of the 181 failure (2026-08-30):
    kngr_238.json, which names a 34 MB CHIFFON library, disagreed while
    short_rev71.json and s1.json, which do not, agreed.  Two of three passing is
    not "mostly right", it is a component that only fires on one of the three.

    Four things are checked and each has its negative control: that the digest
    is the file's CONTENT, that the three path spellings src/IO.cpp normalises
    all reach it, that `RASBERY_XSLIB_DIGEST=off` empties it on this side
    exactly as it empties it in src/XSSet.cpp, and that the tool never silently
    substitutes one outcome for another.
    """
    # The policy parse, against the C++ that owns it.
    xsset = (ROOT / "src" / "XSSet.cpp").read_text(encoding="utf-8-sig")
    mode_fn = xsset[xsset.index("XsLibraryDigestPolicy XsLibraryDigestMode()"):]
    mode_fn = mode_fn[:mode_fn.index("\n}")]
    for literal, want in (("0", "off"), ("off", "off"), ("always", "always"),
                          ("cached", "cached"), ("", "cached"), ("nonsense", "cached")):
        if case_key.xslib_digest_policy({"RASBERY_XSLIB_DIGEST": literal}) != want:
            fail(f"case_key.xslib_digest_policy({literal!r}) is not {want!r}")
    for token in ('value == "0" || value == "off"', 'value == "always"'):
        if token not in mode_fn:
            fail(f"XsLibraryDigestMode() no longer decides on {token!r}; "
                 "case_key.xslib_digest_policy mirrors it and would now lie")
    if "XsLibraryDigestPolicyName" not in (ROOT / "src" / "XsLibrary.h").read_text(
            encoding="utf-8-sig"):
        fail("XsLibraryDigestPolicyName is not exported; the [RASBERY][CASE] "
             "receipt cannot say which policy emptied the library digest")

    # The WSL prefix is PARSED, not copied, and the parse fails closed.
    if 'WSL_UNC_PREFIX = _read_wsl_unc_prefix()' not in (
            ROOT / "tools" / "case_key.py").read_text(encoding="utf-8"):
        fail("case_key.py carries its own copy of kWslUncPrefix again")
    with tempfile.TemporaryDirectory() as raw:
        stub = Path(raw) / "IO.cpp"
        stub.write_text("nothing here", encoding="utf-8")
        try:
            case_key._read_wsl_unc_prefix(stub)
            fail("case_key._read_wsl_unc_prefix accepted an IO.cpp with no "
                 "kWslUncPrefix; it must fail closed rather than normalise nothing")
        except SystemExit:
            pass

    # THE NORMALISER ITSELF, not through the filesystem.  On Windows a
    # backslash is already a separator, so a fixture that only checked "both
    # spellings find the file" would pass on the authoring host with the
    # normaliser deleted -- and fail on 181, which is the host that matters.
    prefix = case_key.WSL_UNC_PREFIX
    for raw_text, want in (
            ("xs\\chiffon.h5", "xs/chiffon.h5"),
            ("..\\xs\\chiffon.h5", "../xs/chiffon.h5"),
            (prefix + "home/x/lib.h5", "/home/x/lib.h5"),
            ("/already/posix.h5", "/already/posix.h5"),
            ("", "")):
        got = case_key.normalize_input_path(raw_text)
        if got != want:
            fail(f"normalize_input_path({raw_text!r}) = {got!r}, want {want!r} "
                 "-- src/IO.cpp NormalizeInputPath does exactly this and the two "
                 "must agree on every host, not just on one whose separator is "
                 "already the backslash")
    # NEGATIVE CONTROL: the prefix rewrite is not the identity, so the check
    # above is testing the rewrite rather than a fixture that needed none.
    if case_key.normalize_input_path(prefix + "a") == prefix + "a":
        fail("the WSL UNC rewrite is a no-op")

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        libdir = tmp / "xs"
        libdir.mkdir()
        lib = libdir / "chiffon.h5"
        lib.write_bytes(b"\x89HDF\r\n\x1a\n" + b"not really a library, but real bytes")
        want = hashlib.sha256(lib.read_bytes()).hexdigest()

        def keyed(reference, env=None, xslib=True):
            path = write(tmp, "ext.json",
                         deck(QUARTER, extra={"data": {"cross-section": reference}}))
            return case_key.case_key(path, env=dict(env or CLEAN_ENV), xslib=xslib)

        posix = keyed("xs/chiffon.h5")
        if posix["xslib_digest"] != posix["xslib_digest_raw"]:
            fail("a present digest must token to itself; only the empty case "
                 "becomes a tilde")
        if posix["xslib_digest"] != want:
            fail("the tool does not digest the CONTENT of an external library "
                 f"named relative to the deck ({posix['xslib_source']})")
        if posix["xslib_source"] != "content":
            fail(f"xslib_source for a present library is {posix['xslib_source']!r}")

        # src/IO.cpp NormalizeInputPath: a backslash spelling and the WSL UNC
        # spelling are the SAME file to the solver.  A tool that opened the raw
        # string finds neither.
        backslash = keyed("xs\\chiffon.h5")
        if backslash["xslib_digest"] != want:
            fail("a backslash-spelled library path does not reach the same file; "
                 "src/IO.cpp NormalizeInputPath turns it into one and the tool "
                 "must too, or a Windows-authored deck keys differently")
        unc = keyed(case_key.WSL_UNC_PREFIX + str(lib).replace("\\", "/").lstrip("/"))
        # The rewritten prefix names a POSIX path that only exists on the WSL
        # side, so on this host the file may or may not be there; what is
        # asserted is the REWRITE, not the lookup.
        if not unc["xslib_path"].startswith("/"):
            fail("the WSL UNC prefix was not rewritten to an absolute POSIX path")
        if case_key.WSL_UNC_PREFIX in unc["xslib_path"]:
            fail("the WSL UNC prefix survived normalisation")
        # NEGATIVE CONTROL for the two above: the RAW strings differ, so the
        # agreement is the normaliser's doing and not the fixture's.
        if "xs\\chiffon.h5" == "xs/chiffon.h5":
            fail("the backslash fixture is not a backslash fixture")

        # LEXICAL, like lexically_normal(): `..` is folded without touching the
        # filesystem, so a symlinked data mount cannot send the two sides to two
        # different files.
        dotted = keyed("xs/../xs/chiffon.h5")
        if dotted["xslib_digest"] != want:
            fail("a path with `..` in it does not fold to the same library")
        if ".." in dotted["xslib_path"]:
            fail("resolve_xs_path left `..` unfolded")

        # RASBERY_XSLIB_DIGEST=off: EMPTY on both sides, and the tool says so.
        off = keyed("xs/chiffon.h5", env={"PATH": "", "RASBERY_XSLIB_DIGEST": "off"})
        if off["xslib_digest_raw"] != "" or off["xslib_digest"] != "~":
            fail("RASBERY_XSLIB_DIGEST=off must empty the library digest here "
                 "exactly as it empties it in src/XSSet.cpp; otherwise the tool "
                 "publishes a key no run can produce -- and only for decks that "
                 "name a library, which is how the 181 mismatch hid")
        if off["xslib_policy"] != "off" or "off" not in off["xslib_source"]:
            fail("the tool does not report that the policy emptied the digest")
        if off["case_key"] == posix["case_key"]:
            fail("the library digest is not load-bearing: `off` and `cached` "
                 "produced the same key")

        # And the content itself is load-bearing.
        lib.write_bytes(b"different bytes entirely")
        if keyed("xs/chiffon.h5")["xslib_digest"] == want:
            fail("rewriting the library did not change the digest")

        # `--no-xslib` is the tool's own shortcut and must never masquerade as a
        # measured empty digest.
        skipped = keyed("xs/chiffon.h5", xslib=False)
        if skipped["xslib_source"] != "skipped":
            fail("--no-xslib does not announce itself in xslib_source")


def one_builder_contract() -> None:
    """WP10.1 FOLLOW-UP: THE THREE PATHS MUST NOT BE THREE BUILDERS.

    The 181 gate (2026-08-30) found the plain single-run CLI's `case_key`
    disagreeing with tools/case_key.py on kngr_238.json while the evaluator's
    key for the same deck matched the tool exactly.  Both C++ paths already
    reach ONE emitter -- Driver::Drive computes caseKeyProvenance and prints
    [RASBERY][CASE] from a single site, pinned above -- so no COMPONENT is
    computed by different code.  What the two call sites can still differ in is
    the INPUTS they hand the Driver, and of the four (deck path, warm start,
    CaseFidelity, process env) exactly one had two spellings:

        main.cpp   run_fidelity = processCaseFidelity();
                   run_fidelity.statepoint_grid = isFullGrid(g) ? "" : g;

        EvaluatorServer.h   resolveCaseFidelity(request, processCaseFidelity(), ...)

    The two agreed line for line, which is not the property to rely on for a
    key whose entire job is that two paths describing the same run produce the
    same digest.  This holds the single-builder rule.
    """
    if "resolveCaseFidelity(" not in MAIN_CPP:
        fail("src/main.cpp no longer resolves its CaseFidelity through "
             "resolveCaseFidelity().  The single-shot CLI and the evaluator must "
             "reach Driver::setCaseFidelity through ONE function, or a change to "
             "how a fidelity resolves can reach one path and miss the other -- "
             "which is exactly the class of defect the 181 case_key gate found")
    if "resolveCaseFidelity(" not in EVAL_H:
        fail("src/EvaluatorServer.h no longer resolves its CaseFidelity through "
             "resolveCaseFidelity()")
    # The hand-assembly, by its shape: a CaseFidelity whose statepoint_grid is
    # written into after construction is a second spelling of the resolver's own
    # grid clause.  Comments are stripped first, because the fix's own comment
    # QUOTES the two lines it replaced; and the FidelityRequest's fields carry
    # the same names, which is the point -- they are the input to the one
    # builder, not a second one.
    def _hand_assembly(text: str) -> list:
        code = chr(10).join(line.split("//")[0] for line in text.splitlines())
        return [t for t in (".statepoint_grid =", ".staged_flux_mult =",
                            ".staged_xe_mult =", ".loose_settle =")
                if re.search(r"(?<!_request)" + re.escape(t), code)]

    for token in _hand_assembly(MAIN_CPP):
        fail(f"src/main.cpp writes {token.strip()} into a CaseFidelity by hand.  "
             f"Every fidelity input goes through a FidelityRequest and "
             f"resolveCaseFidelity, so the CLI cannot resolve a grid (or a staged "
             f"multiplier) by a rule the evaluator does not use")
    # ...and the request it builds must actually carry the grid, or the flag
    # would parse, warn, and then be dropped.
    if "cli_fidelity_request.statepoint_grid = statepoint_grid;" not in MAIN_CPP:
        fail("src/main.cpp does not put --statepoint-grid into the FidelityRequest; "
             "the flag would be validated and then ignored, and the receipt would "
             "report `full` for a coarse run")
    if "cli_fidelity_request.has_grid" not in MAIN_CPP:
        fail("src/main.cpp's FidelityRequest never sets has_grid, so "
             "resolveCaseFidelity skips the grid clause entirely")
    # THE CLI DECLARES NO FIDELITY WORD.  There is no --fidelity flag by design:
    # a word the CLI could declare and not honour is the mixing plan Sec 6.2
    # forbids.  If one is ever added it must go through the same request.
    if "cli_fidelity_request.fidelity" in MAIN_CPP and "--fidelity" not in MAIN_CPP:
        fail("src/main.cpp sets a fidelity WORD on the CLI request with no flag to "
             "have produced it; the request must describe what the operator asked for")
    # A refusal from the resolver must fail the run, not be logged and ignored.
    at = MAIN_CPP.find("resolveCaseFidelity(")
    tail = MAIN_CPP[at:at + 700] if at >= 0 else ""
    if "return 2;" not in tail:
        fail("src/main.cpp does not exit on a resolveCaseFidelity refusal.  The "
             "alternative is a case that runs at one fidelity while its receipt -- and "
             "its case_key -- claim another, which is what the resolver refuses FOR")

    # ONE EMITTER, still.  Restated here because the single-builder rule above is
    # only worth anything while the components are computed in one place.
    if DRIVER_H.count("casekey::Provenance caseKeyProvenance(") != 1:
        fail("Driver.h has more than one caseKeyProvenance; the two call sites could "
             "then assemble different provenances from the same inputs")
    for path, name in ((MAIN_CPP, "src/main.cpp"), (EVAL_H, "src/EvaluatorServer.h")):
        if "caseKeyProvenance" in path or "casekey::keyOf" in path:
            fail(f"{name} computes a case key of its own instead of reading the one "
                 f"Driver::Drive published; two producers of one fact is the defect "
                 f"this whole contract exists to prevent")

    # env_set comes from the header's own spelling, like env_digest.
    if "casekey::envSetToken(" not in DRIVER_H:
        fail("Driver.h does not take env_set from casekey::envSetToken; a second "
             "spelling of `which knobs are set` is a second answer")
    if "inline std::string envSetToken(" not in CASEKEY_H:
        fail("CaseKey.h lost envSetToken; the live gate cannot tell an env divergence "
             "that is CONFIGURATION from one that is a folded VALUE, which is the "
             "reading the 181 finding was left open on")
    # It must be NAMES only.  A receipt that published the values would leak
    # whatever a launcher exported and would duplicate env_digest's job.
    fn = CASEKEY_H[CASEKEY_H.index("inline std::string envSetToken("):]
    fn = fn[:fn.index(chr(10) + "}")]
    if "out += value" in fn or "+= tokenOrTilde(value)" in fn:
        fail("envSetToken publishes the env VALUES.  Names only: the values are what "
             "env_digest is for, and a receipt is not the place to print them")
    if "if (value.empty()) continue;" not in fn:
        fail("envSetToken does not skip unset knobs, so it says nothing about WHICH "
             "are set and cannot separate a configuration difference from a value one")

    # THE TOOL PUBLISHES WHAT IT CLAIMS TO.  COMPONENT_FIELDS is the list the
    # live gate walks; a name on it that case_key() does not return makes the
    # comparison raise mid-diff instead of reporting, which is how a component
    # goes unread again.
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        probe = write(tmp, "probe.json", deck([[1, 2], [2, 1]]))
        produced = case_key.case_key(probe, env={}, xslib=False)
        for field in case_key.COMPONENT_FIELDS:
            if field not in produced:
                fail(f"tools/case_key.py lists {field!r} in COMPONENT_FIELDS and does "
                     f"not return it; the live --compare would raise on the first deck "
                     f"instead of naming the component")

    # ---------------------------------------------------------------- controls
    # Every check above, run against text broken the way it exists to catch.
    controls = []
    if "resolveCaseFidelity(" in MAIN_CPP.replace("resolveCaseFidelity(", "xxx("):
        controls.append("the single-builder check reads a stripped main.cpp as still "
                        "calling resolveCaseFidelity")
    broken = MAIN_CPP.replace("cli_fidelity_request.statepoint_grid = statepoint_grid;",
                              "run_fidelity.statepoint_grid = statepoint_grid;")
    if ".statepoint_grid =" not in _hand_assembly(broken):
        controls.append("the hand-assembly check cannot see a CaseFidelity written "
                        "into by hand")
    probe_fn = ('inline std::string envSetToken(const Provenance& p) {\n'
                '    std::string out;\n'
                '    for (const auto& [name, value] : p.env) { out += tokenOrTilde(value); }\n'
                '    return out;\n}')
    body = probe_fn[probe_fn.index("{"):]
    if "+= tokenOrTilde(value)" not in body:
        controls.append("the names-only check cannot see a value-publishing envSetToken")
    for message in controls:
        fail("NEGATIVE CONTROL FAILED: " + message)


def receipt_component_contract() -> None:
    """The solver publishes every component the tool publishes.

    Without this the live gate can only say "the keys differ", which is what it
    said on 181 and why the diagnosis was a guess.
    """
    site = DRIVER_H.index('"  [RASBERY][CASE] {{')
    block = DRIVER_H[site:DRIVER_H.index(");", site)]
    if '\\"schema_version\\":5' not in block:
        fail("the [RASBERY][CASE] receipt did not bump schema_version when it "
             "gained the component fields; a reader cannot tell the two apart")
    # WP10.4.  The Sec 6.2 spelling of the fidelity, BESIDE the campaign one.
    # tools/exact_audit.py audits `physics_fidelity` on both case tags and this
    # line carried only `fidelity`; see the Driver.h comment at the receipt.
    if '\\"physics_fidelity\\"' not in block:
        fail("[RASBERY][CASE] does not publish 'physics_fidelity'; "
             "exact_audit.CASE_REQUIRED_FIELDS audits this tag for it and every "
             "Driver-side receipt is refused as pre-WP10.3 without it")
    if '\\"fidelity\\"' not in block:
        fail("[RASBERY][CASE] dropped 'fidelity'; tools/case_key.py keys the case "
             "on that name and every manifest on disk carries it")
    for field in case_key.COMPONENT_FIELDS:
        if field in ("case_key", "key_schema"):
            continue
        if f'\\"{field}\\"' not in block:
            fail(f"[RASBERY][CASE] does not publish {field!r}; tools/case_key.py "
                 f"--components prints it and the live gate compares it")
    if "casekey::envDigest(" not in DRIVER_H:
        fail("Driver.h does not take env_digest from casekey::envDigest; a second "
             "spelling of the env half is a second answer")
    for token in ("inline std::string envPayload(", "inline std::string envDigest(",
                  "inline std::string codeShaToken()"):
        if token not in CASEKEY_H:
            fail(f"CaseKey.h lost {token!r}")
    # envPayload must use payloadOf's OWN line spelling, or the component digest
    # is a digest of a paraphrase.
    env_fn = CASEKEY_H[CASEKEY_H.index("inline std::string envPayload("):]
    env_fn = env_fn[:env_fn.index("\n}")]
    for token in ('out += "env\\t";', "out += tokenOrTilde(value);"):
        if token not in env_fn:
            fail(f"envPayload does not spell an env line the way payloadOf does "
                 f"({token!r} missing)")
    if "codeShaToken()" not in CASEKEY_H[CASEKEY_H.index("inline std::string payloadOf("):]:
        fail("payloadOf reads RASBERY_CODE_SHA through its own getenv again; the "
             "receipt and the payload would then be able to disagree")


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
# COMPILED HALF -- the one that actually closes the loop, when a compiler is here
#
# The source half holds the two implementations to the same FIELD SET.  What it
# cannot check is the thing most likely to differ: how each language SPELLS a
# value.  `{:.17g}` versus `%.17g`, the exponent's digit count, integer versus
# float typing, object key order.  So when a C++ compiler is available,
# src/CaseKey.h is compiled into a small harness and its canonical deck payload
# is compared to tools/case_key.py's, BYTE FOR BYTE, on the same decks.
#
# Only `deckPayload` is compared, deliberately: it is the half that walks
# arbitrary JSON and formats floats.  `payloadOf` above it is concatenation of
# fields the source half already compares one by one.
#
# The harness also runs three FIPS 180-4 SHA-256 vectors, because the transform
# moved between headers in this change and "it still compiles" is not the same
# claim as "it still hashes".
#
# No compiler is not a failure -- it is a SKIP, reported as one, because the
# authoring host for this campaign often has none and a test that failed there
# would be turned off.
# ---------------------------------------------------------------------------
HARNESS_CPP = r'''
#include "CaseKey.h"
#include "Sha256.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    struct Vector { const char* in; const char* want; };
    const Vector vectors[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    for (const Vector& v : vectors) {
        if (rasbery::Sha256::hexOf(v.in) != v.want) {
            std::cerr << "sha256 vector mismatch for \"" << v.in << "\"\n";
            return 1;
        }
    }
    if (argc < 3) { std::cerr << "usage: harness <deck.json|--payload names.txt> <out.bin>\n"; return 2; }

    // WP10.1 component mode.  The deck half below is the half that walks
    // arbitrary JSON; THIS is the half the 181 gate actually tripped over --
    // the env lines, the library digest, the warm-start token and the build
    // identity, concatenated by payloadOf().  It was never compared because it
    // "is concatenation of fields the source half already compares one by one",
    // and a field list compared name by name is not a byte comparison: it does
    // not see a separator, a tilde rule or a code_sha read through a second
    // expression.  So it is compared as bytes now, on a provenance whose env
    // half is read from THIS PROCESS's environment through the same getenv the
    // solver uses.
    if (std::string(argv[1]) == "--payload") {
        if (argc < 4) { std::cerr << "usage: harness --payload <names.txt> <out.bin>\n"; return 2; }
        std::ifstream names(argv[2]);
        if (!names) { std::cerr << "cannot open " << argv[2] << "\n"; return 2; }
        rasbery::casekey::Provenance p;
        p.deck_digest  = "deadbeef";
        p.fidelity     = "full_exact";
        p.policy       = "strict";
        p.xslib_digest = "cafebabe";
        p.warm_start   = "";
        std::string name;
        while (std::getline(names, name)) {
            if (name.empty()) continue;
            const char* value = std::getenv(name.c_str());
            p.env.emplace_back(name, value != nullptr ? std::string(value) : std::string());
        }
        const std::string payload = rasbery::casekey::payloadOf(p);
        std::ofstream out(argv[3], std::ios::binary);
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        std::cout << rasbery::casekey::envDigest(p) << "\n"
                  << rasbery::casekey::keyOf(p) << "\n"
                  << rasbery::casekey::codeShaToken() << "\n";
        return 0;
    }

    std::ifstream in(argv[1]);
    if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
    nlohmann::ordered_json config;
    in >> config;
    std::string core_op;
    const std::string payload = rasbery::casekey::deckPayload(config, &core_op);
    std::ofstream out(argv[2], std::ios::binary);
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    std::cout << core_op << "\n";
    return 0;
}
'''

# Values chosen to exercise every spelling the two languages could disagree on:
# an integer, a plain decimal, a tiny exponent, a huge one, a negative, a value
# with no exact binary representation, a bool and a null.
FLOAT_TRAPS = {
    "traps": {"int": 7, "one": 1.0, "tenth": 0.1, "tiny": 1.0e-6, "tinier": 2.5e-17,
              "huge": 1.0e+21, "neg": -3.25e-4, "third": 1.0 / 3.0,
              "yes": True, "no": False, "nothing": None,
              "list": [1, 2.0, "three", None]},
}


def deck_with_floats(core, symang):
    return deck(core, symang=symang, extra=dict(FLOAT_TRAPS))


def find_compiler():
    """MSVC's vcvars64.bat, or a g++/clang++ on PATH, or None."""
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.is_file():
            done = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, universal_newlines=True)
            root = done.stdout.strip().splitlines()
            if done.returncode == 0 and root:
                bat = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if bat.is_file():
                    return str(bat)
    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def compiled_contract() -> bool:
    """True when the compiled half actually ran."""
    compiler = find_compiler()
    if compiler is None:
        return False
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        cpp = tmp / "case_key_harness.cpp"
        cpp.write_text(HARNESS_CPP, encoding="utf-8")
        exe = tmp / ("case_key_harness.exe" if os.name == "nt" else "case_key_harness")
        includes = [ROOT / "src", ROOT / "include", ROOT / "include" / "chiffon"]
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                # QUOTED include paths: this repository's own path contains an
                # `&`, which cmd would otherwise read as a command separator.
                script = tmp / "build_case_key_harness.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul\r\n' % compiler
                    + 'cd /d "%s"\r\n' % tmp
                    + 'cl /nologo /std:c++20 /EHsc /D_CRT_SECURE_NO_WARNINGS "%s" %s /Fe:"%s"\r\n'
                      % (cpp, " ".join('/I "%s"' % d for d in includes), exe),
                    encoding="utf-8")
                subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(tmp),
                               capture_output=True, universal_newlines=True)
            else:
                subprocess.run(
                    [compiler, "-std=c++20", "-O0", str(cpp), "-o", str(exe)]
                    + [arg for d in includes for arg in ("-I", str(d))],
                    check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            fail("the case-key harness does not compile:\n"
                 + (failure.stdout or "") + (failure.stderr or ""))
            return True

        payloads = {}
        for name, core, symang in (("base", QUARTER, 90), ("transpose", QUARTER_T, 90),
                                   ("half", QUARTER, 180)):
            path = write(tmp, f"{name}.json", deck_with_floats(core, symang))
            out = tmp / f"{name}.bin"
            done = subprocess.run([str(exe), str(path), str(out)],
                                  capture_output=True, universal_newlines=True)
            if done.returncode != 0:
                fail(f"the compiled case-key harness failed on {name}: "
                     f"{done.stdout}{done.stderr}")
                return True
            payloads[name] = out.read_bytes()
            py_text, py_op = case_key.deck_payload(json.loads(path.read_text()))
            py_bytes = py_text.encode("utf-8")
            if done.stdout.strip() != py_op:
                fail(f"{name}: C++ core_op {done.stdout.strip()!r} != python {py_op!r}")
            if payloads[name] != py_bytes:
                where = next((i for i, (a, b) in enumerate(zip(payloads[name], py_bytes))
                              if a != b), min(len(payloads[name]), len(py_bytes)))
                fail(f"{name}: the canonical deck payloads differ at byte {where}\n"
                     f"  C++    {payloads[name][max(0, where - 70):where + 70]!r}\n"
                     f"  python {py_bytes[max(0, where - 70):where + 70]!r}")
        # ---- the WHOLE payload, not just the deck half --------------------
        #
        # Two environments, and the second one is the negative control: the
        # first leaves every knob unset (so every env line is the tilde rule)
        # and the second sets a handful including RASBERY_CODE_SHA, so a
        # divergence in the separator, the tilde rule or the code_sha read
        # cannot hide behind "everything was empty anyway".
        names_file = tmp / "arm_env.txt"
        names_file.write_text("\n".join(case_key.ARM_ENV) + "\n", encoding="utf-8")
        base_env = {k: v for k, v in os.environ.items()
                    if not k.startswith("RASBERY_")}
        loaded = dict(base_env)
        loaded.update({"RASBERY_GPU": "1", "RASBERY_GPU_XE": "1",
                       "RASBERY_XE_FORMS": "0xd3d",
                       "RASBERY_STAGED_FLUX_TOL": "4.0",
                       "RASBERY_CODE_SHA": "a1b2c3d4"})
        for label, environ in (("unset", base_env), ("loaded", loaded)):
            out = tmp / f"payload_{label}.bin"
            done = subprocess.run([str(exe), "--payload", str(names_file), str(out)],
                                  capture_output=True, universal_newlines=True,
                                  env=environ)
            if done.returncode != 0:
                fail(f"the compiled case-key harness failed in payload mode "
                     f"({label}): {done.stdout}{done.stderr}")
                return True
            cpp_env_digest, cpp_key, cpp_code_sha = \
                (done.stdout.splitlines() + ["", "", ""])[:3]
            py_payload = case_key.payload_of("deadbeef", "full_exact", "strict",
                                             environ, "cafebabe", "")
            cpp_payload = out.read_bytes()
            if cpp_payload != py_payload.encode("utf-8"):
                where = next((i for i, (a, b) in enumerate(
                    zip(cpp_payload, py_payload.encode("utf-8"))) if a != b),
                    min(len(cpp_payload), len(py_payload)))
                fail(f"payload({label}): the two implementations differ at byte "
                     f"{where}\n  C++    {cpp_payload[max(0, where - 70):where + 70]!r}"
                     f"\n  python {py_payload.encode('utf-8')[max(0, where - 70):where + 70]!r}")
            if cpp_env_digest != case_key.env_digest(environ):
                fail(f"payload({label}): env_digest C++ {cpp_env_digest!r} != "
                     f"python {case_key.env_digest(environ)!r}")
            want_key = hashlib.sha256(py_payload.encode("utf-8")).hexdigest()
            if cpp_key != want_key:
                fail(f"payload({label}): case_key C++ {cpp_key!r} != python {want_key!r}")
            if cpp_code_sha != (environ.get("RASBERY_CODE_SHA") or "~"):
                fail(f"payload({label}): codeShaToken C++ {cpp_code_sha!r} != "
                     f"{environ.get('RASBERY_CODE_SHA') or '~'!r}")

        # And the property, proved on the COMPILED side rather than inferred
        # from the python one.
        if payloads["base"] != payloads["transpose"]:
            fail("C++ does not fold a quarter map and its transpose to one payload")
        if payloads["base"] == payloads["half"]:
            fail("C++ folded a 180-degree map; only 90 and 360 have an argued orbit")
    return True


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
    # EVERY COMPONENT, NOT JUST THE KEY.  The 181 gate (2026-08-30) failed here
    # on kngr_238.json and passed on two smaller decks, and all it could say was
    # that two 64-hex digests differed -- so the diagnosis was a guess about
    # float notation when the components had never been compared.  The receipt
    # now publishes them (schema_version 3) and this walks them in payload
    # order, so the FIRST differing component names itself.
    problems = []
    missing = []
    for field in case_key.COMPONENT_FIELDS:
        if field not in found:
            missing.append(field)
            continue
        if found.get(field) != mine[field]:
            problems.append(f"{field}: solver {found.get(field)!r} vs tool {mine[field]!r}")
    print(f"  receipt schema_version {found.get('schema_version')}")
    for field in case_key.COMPONENT_FIELDS:
        print(f"  {field:<18} solver {str(found.get(field, '(absent)')):<66} "
              f"tool {mine[field]}")
    print(f"  {'xslib_path':<18} tool   {mine['xslib_path']}  ({mine['xslib_source']})")
    if missing:
        print("  NOTE the receipt predates schema_version 3 and carries no "
              + ", ".join(missing)
              + " -- rebuild before reading a component verdict from it")
    for line in problems:
        print(f"  FAIL {line}")

    # WHAT KIND OF DIVERGENCE THIS IS.  `env_digest` differing has two very
    # different causes and they need opposite responses, which is why host 181's
    # WP10.1 finding could not be closed from the numbers it had: the solver
    # folds a value the tool does not model (read code), or the two ran under
    # different environments (fix the harness).  env_set separates them by name.
    if any(line.startswith("env_digest:") for line in problems):
        solver_set = set(filter(None, str(found.get("env_set", "")).split(",")))
        tool_set = set(filter(None, mine["env_set"].split(",")))
        if "env_set" not in found:
            print("  NOTE this receipt predates env_set, so which env vars differ "
                  "cannot be read from it -- rebuild and re-run before concluding "
                  "anything about env_digest")
        elif solver_set != tool_set:
            print("  ENV DIVERGENCE IS CONFIGURATION, NOT CODE.  The two runs did not "
                  "have the same knobs set, so the keys SHOULD differ:")
            for name in sorted(solver_set - tool_set):
                print(f"    set in the run, unset for the tool: {name}")
            for name in sorted(tool_set - solver_set):
                print(f"    set for the tool, unset in the run: {name}")
            print("  Re-run tools/case_key.py in the run's own environment before "
                  "reading this as a solver defect.")
        else:
            print("  ENV DIVERGENCE IS A VALUE, NOT A CONFIGURATION.  The same knobs "
                  "are set on both sides and the folded values differ -- this is the "
                  "case that IS a code question: src/Driver.h armEnvValue() re-spells "
                  "the three RASBERY_STAGED_* knobs from the case's CaseFidelity "
                  "whenever it differs from processCaseFidelity(), and "
                  "tools/case_key.py reads them raw.")

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
    one_builder_contract()
    receipt_component_contract()
    behaviour_contract()
    xslib_contract()
    compiled = compiled_contract()
    py_compile.compile(str(ROOT / "tools" / "case_key.py"), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if FAILED:
        for message in FAILED:
            print(f"case key contract: FAIL: {message}")
        return 1
    print("case key contract: PASS (source + receipt components + behaviour + "
          "external library"
          + (" + compiled byte-for-byte)" if compiled
             else "; NO C++ COMPILER -- the byte-for-byte half did not run)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
