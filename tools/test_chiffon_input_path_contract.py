#!/usr/bin/env python3
"""Every HGC path taken from a --chiffoni JSON must honour an absolute path.

WHY THIS EXISTS.  `Chiffon::Importer::ReadInput` and its HGC-appending helpers
(`AppendHGCPoints`, `AppendRodDepletionHGC`, `AppendRodDepletionPairHGC`) used
to resolve every "main"/"extra"/"rod depletion"/"reflectors.files" entry with
the unconditional `baseDir.string() + "/" + fileName`. On 238 this produced a
malformed doubled path -- e.g.
  /home/kmk/gates/ismr//home/kmk/ismr_bench_20260810/input/higa/hgc/FA_A1.HGC
-- whenever the json's "hgc" entries were already absolute, and the run
aborted (rc=134 / rc=1). Neither docs/CHIFFON_manual.md nor
docs/chiffon-rasbery-interface.md documents a path-resolution rule for these
fields -- the contract is enforced here in code, not in a schema doc.

WHAT THIS CHECKS.  include/chiffon/Importer.h must define a path resolver
(`ResolveHGCPath` calling `IsAbsoluteHGCPath`, or equivalent) that:
  1. treats a POSIX-absolute path (leading '/'), a leading '\\', or a
     Windows drive-letter path ("C:\\..." / "C:/...") as absolute and
     returns it unchanged;
  2. otherwise joins it onto the json's own directory (baseDir), the prior
     (buggy) behaviour;
and that every call site which used to build an HGC path via the raw
`baseDir.string() + "/" + <name>` concatenation now routes through that
resolver instead -- so no call site can silently regress back to the
unconditional-prepend bug this file guards against.

An unevaluable/renamed resolver, or any leftover raw `baseDir.string() +
"/"` concatenation for an HGC path, is a FAILURE, not a skip: this is a
static presence/absence check over a small, known file, so there is nothing
here that legitimately can't be evaluated.

USAGE
    tools/test_chiffon_input_path_contract.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "include" / "chiffon" / "Importer.h"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ---------------------------------------------------------------------------
# The guard itself: a resolver function that (a) exists, (b) checks for a
# POSIX '/' lead, a '\\' lead, and a drive-letter lead, and (c) falls back to
# baseDir / fileName. This is intentionally a structural regex over the
# resolver's body, not a full parse -- see MUST_REJECT/MUST_ACCEPT below for
# what it does and does not catch.
# ---------------------------------------------------------------------------
RESOLVER_SIG = re.compile(
    r"std::string\s+ResolveHGCPath\s*\([^)]*\)\s*\{(?P<body>.*?)\n\s*\}",
    re.S,
)
ABS_CHECK_FN = re.compile(
    r"bool\s+IsAbsoluteHGCPath\s*\([^)]*\)\s*\{(?P<body>.*?)\n\s*\}",
    re.S,
)
POSIX_LEAD = re.compile(r"fileName\[0\]\s*==\s*'/'")
BACKSLASH_LEAD = re.compile(r"fileName\[0\]\s*==\s*'\\\\\\\\'|fileName\[0\]\s*==\s*'\\\\'")
DRIVE_LEAD = re.compile(r"fileName\[1\]\s*==\s*':'")
FALLBACK_JOIN = re.compile(r"baseDir\s*/\s*fileName")

# A raw, unconditional "baseDir string + separator + name" concatenation --
# the exact shape of the bug this guards against -- anywhere in the file.
RAW_PREPEND = re.compile(r'baseDir\.string\(\)\s*\+\s*"/"\s*\+')


def find_resolver(text: str) -> tuple[bool, list[str]]:
    """(guard present and structurally correct, reasons if not)."""
    reasons: list[str] = []

    abs_match = ABS_CHECK_FN.search(text)
    if abs_match is None:
        reasons.append("no IsAbsoluteHGCPath(...) function found")
        abs_body = ""
    else:
        abs_body = abs_match.group("body")
        if not POSIX_LEAD.search(abs_body):
            reasons.append("IsAbsoluteHGCPath does not check a POSIX leading '/'")
        if not BACKSLASH_LEAD.search(abs_body):
            reasons.append("IsAbsoluteHGCPath does not check a leading '\\\\'")
        if not DRIVE_LEAD.search(abs_body):
            reasons.append("IsAbsoluteHGCPath does not check a Windows drive-letter lead (':')")

    res_match = RESOLVER_SIG.search(text)
    if res_match is None:
        reasons.append("no ResolveHGCPath(...) function found")
    else:
        res_body = res_match.group("body")
        if "IsAbsoluteHGCPath" not in res_body:
            reasons.append("ResolveHGCPath does not call IsAbsoluteHGCPath")
        if not FALLBACK_JOIN.search(res_body):
            reasons.append("ResolveHGCPath does not fall back to joining baseDir / fileName")

    return (not reasons), reasons


def find_raw_prepends(text: str) -> list[int]:
    """Line numbers of any leftover unconditional baseDir-prepend concatenation."""
    lines = text.split("\n")
    hits = []
    for i, line in enumerate(lines, start=1):
        if RAW_PREPEND.search(line):
            hits.append(i)
    return hits


# ---------------------------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------------------------
MUST_REJECT = (
    ("the literal bug: unconditional prepend, no absolute-path guard at all",
     'void ReadHGC(Model& model, const std::string& fileName);\n'
     'void AppendHGCPoints(Model& targetModel, const std::filesystem::path& baseDir,\n'
     '                     const std::string& fileName) {\n'
     '    ReadHGC(hgcModel, baseDir.string() + "/" + fileName);\n'
     '}\n'),
    ("a resolver that exists but never checks the POSIX '/' case",
     'static bool IsAbsoluteHGCPath(const std::string& fileName) {\n'
     '    if (fileName.size() >= 2 && std::isalpha((unsigned char)fileName[0]) && fileName[1] == \':\')\n'
     '        return true;\n'
     '    return false;\n'
     '}\n'
     'static std::string ResolveHGCPath(const std::filesystem::path& baseDir,\n'
     '                                   const std::string& fileName) {\n'
     '    if (IsAbsoluteHGCPath(fileName)) return fileName;\n'
     '    return (baseDir / fileName).string();\n'
     '}\n'),
    ("a resolver that checks absoluteness but never falls back to baseDir / fileName",
     'static bool IsAbsoluteHGCPath(const std::string& fileName) {\n'
     '    if (fileName.empty()) return false;\n'
     "    if (fileName[0] == '/' || fileName[0] == '\\\\\\\\') return true;\n"
     '    if (fileName.size() >= 2 && std::isalpha((unsigned char)fileName[0]) && fileName[1] == \':\')\n'
     '        return true;\n'
     '    return false;\n'
     '}\n'
     'static std::string ResolveHGCPath(const std::filesystem::path& baseDir,\n'
     '                                   const std::string& fileName) {\n'
     '    if (IsAbsoluteHGCPath(fileName)) return fileName;\n'
     '    return fileName;\n'
     '}\n'),
)

MUST_ACCEPT = (
    ("the fix: full guard with POSIX, backslash and drive-letter checks, "
     "falling back to baseDir / fileName",
     'static bool IsAbsoluteHGCPath(const std::string& fileName) {\n'
     '    if (fileName.empty())\n'
     '        return false;\n'
     "    if (fileName[0] == '/' || fileName[0] == '\\\\\\\\')\n"
     '        return true;\n'
     '    if (fileName.size() >= 2 && std::isalpha(static_cast<unsigned char>(fileName[0])) &&\n'
     '        fileName[1] == \':\')\n'
     '        return true;\n'
     '    return false;\n'
     '}\n'
     '\n'
     'static std::string ResolveHGCPath(const std::filesystem::path& baseDir,\n'
     '                                   const std::string& fileName) {\n'
     '    if (IsAbsoluteHGCPath(fileName))\n'
     '        return fileName;\n'
     '    return (baseDir / fileName).string();\n'
     '}\n'),
)


def main() -> int:
    failures: list[str] = []

    if not IMPORTER.is_file():
        print("chiffon input path contract: FAIL")
        print(f"  - {IMPORTER.relative_to(ROOT)} not found")
        return 1

    raw_text = IMPORTER.read_text(encoding="utf-8-sig")
    text = strip_comments(raw_text)

    ok, reasons = find_resolver(text)
    if not ok:
        for r in reasons:
            failures.append(f"{IMPORTER.relative_to(ROOT)}: {r}")

    raw_hits = find_raw_prepends(text)
    for line in raw_hits:
        failures.append(
            f"{IMPORTER.relative_to(ROOT)}:{line}: unconditional "
            '`baseDir.string() + "/" + ...` concatenation for an HGC path -- '
            "this is exactly the doubled-path bug (rc=134/rc=1 on 238); route "
            "through ResolveHGCPath instead")

    for name, source in MUST_REJECT:
        pass_ok, _ = find_resolver(source)
        if pass_ok and not find_raw_prepends(source):
            failures.append(f"negative control PASSED -- the scan is vacuous for: {name}")

    for name, source in MUST_ACCEPT:
        pass_ok, reasons = find_resolver(source)
        raw = find_raw_prepends(source)
        if not pass_ok or raw:
            failures.append(
                f"positive control REJECTED -- the scan cries wolf on: {name} "
                f"({'; '.join(reasons) if reasons else raw})")

    if failures:
        print("chiffon input path contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1

    print("chiffon input path contract: PASS (%s, %d negative + %d positive controls)"
          % (IMPORTER.relative_to(ROOT), len(MUST_REJECT), len(MUST_ACCEPT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
