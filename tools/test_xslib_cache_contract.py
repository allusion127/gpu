#!/usr/bin/env python3
"""Static contract for the process-wide host XSLIB cache (plan Rev.4 Sec 14).

WHAT THIS PROTECTS.  `Importer::LoadHDF` is a pure function of a filename and
the flatten that follows it is a pure function of (models, ng, niso), so M decks
naming one library used to produce M identical copies of the same ~34 MB parse
-- serialised, because `IO::ReadInput` holds the process-global HDF5 guard
across the whole call.  The measured shape was a staircase: eight concurrent
workers reporting Init+IO of 4.108 .. 14.506 s, ~1.49 s per case (GA evaluator
plan Sec 3.1).  The cache removes it.

The physics gate is the digest and h5diff; those cost 40 s a run.  This test
holds the three structural properties that make the shared parse SOUND, each of
which is cheap to break and silent when broken:

  1. THE PARSE IS SHARED AND CONST.  XSSet holds `shared_ptr<const XsLibrary>`,
     not its own copy, and hands out no mutable alias into it.  A single
     `double&` into a shared depletion point is a cross-deck data race.
  2. THE LOOKUP IS OUTSIDE THE HDF5 GUARD.  A cache whose hit path queues behind
     another worker's parse leaves the staircase exactly where it was.
  3. ONE ENTRY POINT.  LoadHDF is reachable only through BuildXsLibrary, so
     there is no second, uncached door into the parse.

Every check runs against a deliberately broken copy of the same text as a
negative control, so a rule that has stopped discriminating fails loudly instead
of passing vacuously.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def _still_in_scope(text: str, declared: int, used: int) -> bool:
    """True when the scope a declaration sits in has not closed by `used`.

    Brace depth relative to the declaration: the scope ends the first time the
    depth goes negative.  That is exactly "is this lock still held there".
    """
    depth = 0
    for i in range(declared, used):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth < 0:
                return False
    return True


def body(text: str, signature: str) -> str:
    """The brace-balanced body that follows `signature`."""
    start = text.find(signature)
    if start < 0:
        raise AssertionError(f"missing signature {signature!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise AssertionError(f"no body for {signature!r}")
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace:i + 1]
    raise AssertionError(f"unterminated body for {signature!r}")


# ---------------------------------------------------------------------------
# The rules.  Each takes the four sources and raises AssertionError on failure.
# ---------------------------------------------------------------------------


def rule_header_declares_the_cache(src: dict[str, str]) -> None:
    header = src["src/XsLibrary.h"]
    for token in (
        "struct XsLibrary",
        "struct XsLibraryCacheStats",
        "std::shared_ptr<const XsLibrary> BuildXsLibrary",
        "std::shared_ptr<const XsLibrary> AcquireXsLibrary",
        "XsLibraryCacheStats XsLibraryCacheSnapshot",
        "void PrintXsLibraryCacheReceipt",
    ):
        if token not in header:
            raise AssertionError(f"XsLibrary.h does not declare {token!r}")
    # The key is (canonical path, size, mtime, ng) -- a cache keyed by path
    # alone would serve a stale parse after the library file is regenerated,
    # which is the one way this can silently change physics.
    for field in ("path", "file_size", "mtime", "ng", "niso"):
        if not re.search(rf"^\s*\S.*\b{field}\b\s*(=|;)", header, re.M):
            raise AssertionError(f"XsLibrary carries no {field!r} provenance field")


def rule_xsset_shares_a_const_parse(src: dict[str, str]) -> None:
    header = src["src/XSSet.h"]
    if "std::shared_ptr<const XsLibrary> _lib;" not in header:
        raise AssertionError("XSSet does not hold a shared_ptr<const XsLibrary> _lib")
    # No private copy of the library may survive beside the shared one.
    strays = [
        name
        for name in re.findall(r"^\s+[\w:<>,\s*&]+?\s(_lib_\w+|_refr_\w+|_brch_\w+|_models)\s*[;=]",
                               header, re.M)
    ]
    if strays:
        raise AssertionError(f"XSSet still owns library-derived members: {sorted(set(strays))}")


def rule_no_mutable_alias_escapes(src: dict[str, str]) -> None:
    header = src["src/XSSet.h"]
    # models() must have no non-const overload...
    for match in re.finditer(r"std::vector<Chiffon::Model>&\s+models\s*\(\s*\)([^;{]*)", header):
        if "const" not in match.group(1):
            raise AssertionError("XSSet::models() has a non-const overload into the shared parse")
    # ...and fmap()/gmap(), which returned `double&` into _models[...], are gone.
    for dead in ("double& fmap(", "double& gmap("):
        if dead in header:
            raise AssertionError(f"{dead!r} still hands out a mutable alias into the shared parse")
    # PPR is the only reader that used to bind them mutably.
    ppr = src["src/PPR.cpp"]
    if re.search(r"(?<!const )auto&\s+model\s*=\s*_xs\.models\(\)", ppr):
        raise AssertionError("PPR binds a mutable reference to a shared model")
    if re.search(r"model\._refr_dpts\s*\[", ppr):
        raise AssertionError("PPR uses the inserting _refr_dpts[...] on a shared model")


def rule_lookup_is_outside_the_hdf5_guard(src: dict[str, str]) -> None:
    source = src["src/XSSet.cpp"]
    acquire = body(source, "std::shared_ptr<const XsLibrary> AcquireXsLibrary")
    if "Hdf5Guard" in acquire:
        raise AssertionError("AcquireXsLibrary takes the HDF5 guard; a hit would queue behind a parse")
    if "XsLibraryCacheMutex" not in acquire:
        raise AssertionError("AcquireXsLibrary does not take the cache's own mutex")
    # The miss must build OUTSIDE the cache lock too, or the queue simply moves
    # one level down: a HIT would wait behind somebody else's 34 MB parse and
    # the Init+IO staircase would be unchanged.  Walk the function in order and
    # ask, at every BuildXsLibrary call, whether the cache mutex is held.
    held = False
    events = re.finditer(
        r"std::unique_lock<std::mutex>\s+\w+\(XsLibraryCacheMutex\(\)\)"
        r"|std::lock_guard<std::mutex>\s+\w+\(XsLibraryCacheMutex\(\)\)"
        r"|\w+\.unlock\(\)"
        r"|\w+\.lock\(\)"
        r"|BuildXsLibrary\s*\(",
        acquire)
    builds = 0
    for event in events:
        token = event.group(0)
        if token.startswith("BuildXsLibrary"):
            builds += 1
            if held:
                raise AssertionError(
                    "AcquireXsLibrary calls BuildXsLibrary while holding the cache lock")
        elif "unlock" in token:
            held = False
        else:
            held = True
    if builds == 0:
        raise AssertionError("AcquireXsLibrary never builds a miss")

    # And exactly one worker may build a given key: the others wait on the key,
    # not on the mutex.  Without this, a cold width-M wave produces M identical
    # 34 MB parses at once (measured: loads=8 at width 8).
    if "XsLibraryCacheReady().wait(" not in acquire:
        raise AssertionError("AcquireXsLibrary does not single-flight: concurrent misses all parse")


def rule_one_entry_point_to_the_parse(src: dict[str, str]) -> None:
    source = src["src/XSSet.cpp"]
    build = body(source, "std::shared_ptr<const XsLibrary> BuildXsLibrary")
    outside = source.replace(build, "")
    if re.search(r"\bLoadHDF\s*\(", outside):
        raise AssertionError("LoadHDF is reachable outside BuildXsLibrary")
    if "LoadHDF" not in build:
        raise AssertionError("BuildXsLibrary does not parse the library")
    init = body(source, "void XSSet::Initialize(")
    if "AcquireXsLibrary" not in init:
        raise AssertionError("XSSet::Initialize does not go through the cache")


def rule_receipt_is_emitted(src: dict[str, str]) -> None:
    source = src["src/XSSet.cpp"]
    receipt = body(source, "void PrintXsLibraryCacheReceipt")
    for key in ("loads", "hits", "bytes", "lock_wait_ms"):
        if f'\\"{key}\\"' not in receipt and f'"{key}"' not in receipt:
            raise AssertionError(f"[XSLIB_CACHE] receipt omits {key!r}")
    if "[RASBERY][XSLIB_CACHE]" not in receipt:
        raise AssertionError("receipt does not carry the [RASBERY][XSLIB_CACHE] tag")
    main = src["src/main.cpp"]
    if main.count("PrintXsLibraryCacheReceipt") < 2:
        raise AssertionError("main.cpp does not print the receipt on both the batch and single paths")


RULES = [
    ("header declares the cache", rule_header_declares_the_cache,
     "src/XsLibrary.h", ("struct XsLibraryCacheStats", "struct XsLibCacheStatsRenamed")),
    ("XSSet shares a const parse", rule_xsset_shares_a_const_parse,
     "src/XSSet.h", ("std::shared_ptr<const XsLibrary> _lib;",
                     "std::shared_ptr<const XsLibrary> _lib;\n    std::vector<double> _lib_burn;")),
    ("no mutable alias escapes", rule_no_mutable_alias_escapes,
     "src/XSSet.h", ("const std::vector<Chiffon::Model>& models() const",
                     "std::vector<Chiffon::Model>& models() { return _lib->models; }\n"
                     "    const std::vector<Chiffon::Model>& models() const")),
    ("lookup is outside the HDF5 guard", rule_lookup_is_outside_the_hdf5_guard,
     "src/XSSet.cpp", ("const XsLibraryKeyFields key = XsLibraryKeyOf(xs_path);",
                       "Chiffon::Hdf5Guard hdf5_guard;\n"
                       "    const XsLibraryKeyFields key = XsLibraryKeyOf(xs_path);")),
    ("one entry point to the parse", rule_one_entry_point_to_the_parse,
     "src/XSSet.cpp", ("void XSSet::Initialize(const std::string& xs_path) {",
                       "void XSSet::Initialize(const std::string& xs_path) {\n"
                       "    Importer stray; stray.LoadHDF(xs_path);")),
    ("receipt is emitted", rule_receipt_is_emitted,
     "src/XSSet.cpp", ('\\"lock_wait_ms\\":', '\\"lock_wait_zz\\":')),
]

SOURCES = ["src/XsLibrary.h", "src/XSSet.h", "src/XSSet.cpp", "src/PPR.cpp", "src/main.cpp"]


def main() -> int:
    src = {rel: read(rel) for rel in SOURCES}
    failures: list[str] = []

    for name, rule, _f, _m in RULES:
        try:
            rule(src)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")

    # Negative control: every rule must REJECT its own break.  A presence check
    # that no longer discriminates is worse than no check -- it reports PASS.
    for name, rule, target, (needle, replacement) in RULES:
        broken = dict(src)
        if needle not in broken[target]:
            failures.append(f"{name}: negative control is stale, {needle!r} not in {target}")
            continue
        broken[target] = broken[target].replace(needle, replacement, 1)
        try:
            rule(broken)
        except AssertionError:
            continue
        failures.append(f"{name}: negative control PASSED the rule -- the rule is vacuous")

    if failures:
        for f in failures:
            print(f"xslib cache contract: FAIL {f}")
        return 1
    print(f"xslib cache contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
