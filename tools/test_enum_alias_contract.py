#!/usr/bin/env python3
"""No enumerator may be given another enumerator's value by ARITHMETIC.

WHY THIS EXISTS.  `5883023` inserted four phases into `Driver.h`'s `enum Phase`
between `PH_UPDDHAT` and a line that read `PH_PPR_RESET = PH_UPDDHAT + 1`.  That
initialiser had been correct when it was written and was now 7 -- the value
`PH_TH_UPDATE` already had -- so the six floor phases aliased the six loop
phases one for one, `phaseName`'s switch acquired four duplicate case VALUES,
and every translation unit that includes Driver.h stopped compiling.  It took
six commits and someone compiling `main.cpp` for an unrelated reason to notice
(`2a1161b`).

Two things made it invisible.  The labels are DISTINCT IDENTIFIERS, so no text
search for a repeated `case` finds it -- only evaluating the enum does.  And the
only compiler on the authoring machine is MSVC, which sees a header at all only
when some .cpp in the tree includes it; a phase enum consumed from a .cu is
never compiled here.

WHAT THIS CHECKS.  Every enum under src/, test/ and include/chiffon/ whose
enumerators are all evaluable (a bare name, an integer literal, or arithmetic
over earlier enumerators and literals) is evaluated, and two enumerators sharing
a value are reported WHEN at least one of them got its value from arithmetic.

That last clause is the whole difference between a guard and a nuisance.
`PH_LOOP_FIRST = PH_TH_UPDATE` is a deliberate range marker and `CO_ONE_BIT_COUNT`
is a deliberate trailing sentinel; both alias on purpose, both are spelled as an
identity or a bare successor, and neither has ever broken anything.  `= X + 1`
is the spelling that goes stale when someone inserts a member, and it is the
only spelling flagged.

An enum this cannot evaluate is REPORTED as skipped rather than assumed clean.

USAGE
    tools/test_enum_alias_contract.py
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STRING = re.compile(r'"(?:\\.|[^"\\])*"')
CHARLIT = re.compile(r"'(?:\\.|[^'\\])*'")
ENUM = re.compile(r"\benum\s+(?:class\s+|struct\s+)?(\w+)?\s*(?::\s*[\w:\s]+)?\s*\{")
# Only characters that can appear in an integer constant expression over names.
# Anything else (a call, a cast, a string) makes the enum unevaluable.
SAFE = re.compile(r"^[\w\s+\-*/()<>|&~^]*$")
ARITHMETIC = "+-*/"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = STRING.sub('""', text)
    return CHARLIT.sub("'x'", text)


def split_top_level(body: str) -> list[str]:
    """Split an enum body on commas that are not inside brackets."""
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def brace_body(text: str, open_at: int) -> str:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
    return ""


def scan(text: str, label: str) -> tuple[list[str], int, int]:
    """(findings, enums evaluated, enums skipped) for one source text."""
    code = strip_comments(text)
    findings: list[str] = []
    evaluated = skipped = 0
    for m in ENUM.finditer(code):
        body = brace_body(code, code.index("{", m.start()))
        values: dict[str, int] = {}
        from_arithmetic: dict[str, bool] = {}
        nxt, ok = 0, True
        for item in split_top_level(body):
            item = item.strip()
            if not item:
                continue
            name, _, expr = item.partition("=")
            name, expr = name.strip(), expr.strip()
            if not re.fullmatch(r"\w+", name) or (expr and not SAFE.match(expr)):
                ok = False
                break
            if expr:
                try:
                    # No builtins, and the only names in scope are the
                    # enumerators already seen -- this evaluates an integer
                    # constant expression, it does not run the tree's code.
                    value = eval(expr, {"__builtins__": {}}, dict(values))  # noqa: S307
                except Exception:
                    ok = False
                    break
                if not isinstance(value, int) or isinstance(value, bool):
                    ok = False
                    break
            else:
                value = nxt
            values[name] = value
            from_arithmetic[name] = bool(expr) and any(o in expr for o in ARITHMETIC)
            nxt = value + 1
        if not ok or len(values) < 2:
            skipped += 1
            continue
        evaluated += 1
        by_value: dict[int, list[str]] = {}
        for name, value in values.items():
            by_value.setdefault(value, []).append(name)
        for value, names in sorted(by_value.items()):
            if len(names) > 1 and any(from_arithmetic[n] for n in names):
                line = code[:m.start()].count("\n") + 1
                findings.append(
                    f"{label}:{line}: enum {m.group(1) or '<anonymous>'} gives "
                    f"{', '.join(names)} the same value ({value}), and at least one "
                    f"of them got it by arithmetic over another enumerator -- a "
                    f"switch over this enum has duplicate case VALUES and will not "
                    f"compile")
    return findings, evaluated, skipped


# ---------------------------------------------------------------------------
# CONTROLS.  The first is the literal shape of the break this guards against.
# ---------------------------------------------------------------------------
MUST_REJECT = (
    ("2a1161b: an initialiser that went stale when a member was inserted",
     "enum Phase : int {\n"
     "    PH_UPDPSI = 0,\n"
     "    PH_UPDDHAT,\n"
     "    PH_TH_UPDATE,\n"
     "    PH_PPR_RESET = PH_UPDDHAT + 1,\n"
     "};\n"),
    ("two arithmetic initialisers that meet",
     "enum E { A = 0, B = A + 3, C = 1, D = C + 2 };\n"),
)

MUST_ACCEPT = (
    ("the fix: no initialiser at all",
     "enum Phase : int {\n"
     "    PH_UPDPSI = 0,\n"
     "    PH_UPDDHAT,\n"
     "    PH_TH_UPDATE,\n"
     "    PH_PPR_RESET,\n"
     "};\n"),
    ("a deliberate identity range marker",
     "enum Phase : int { PH_A = 0, PH_B, PH_LOOP_FIRST = PH_A, PH_LOOP_LAST = PH_B };\n"),
    ("a trailing count sentinel, which equals nothing before it",
     "enum { CO_PSI_ACC = 0, CO_DHAT_NUM, CO_ONE_BIT_COUNT };\n"),
    ("distinct arithmetic values",
     "enum E { A = 0, B = A + 1, C = B + 1 };\n"),
)


def main() -> int:
    failures: list[str] = []
    files = []
    for folder in ("src", "test", "include/chiffon"):
        for ext in ("*.h", "*.cpp", "*.cu", "*.cuh"):
            files += list((ROOT / folder).rglob(ext))
    files = sorted(set(files))

    evaluated = skipped = 0
    for path in files:
        found, ev, sk = scan(path.read_text(encoding="utf-8-sig"),
                             str(path.relative_to(ROOT)).replace("\\", "/"))
        failures.extend(found)
        evaluated += ev
        skipped += sk

    for name, source in MUST_REJECT:
        if not scan(source, "<control>")[0]:
            failures.append(f"negative control PASSED -- the scan is vacuous for: {name}")
    for name, source in MUST_ACCEPT:
        hits = scan(source, "<control>")[0]
        if hits:
            failures.append(f"positive control REJECTED -- the scan cries wolf on: "
                            f"{name} ({hits[0]})")

    if failures:
        print("enum alias contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("enum alias contract: PASS (%d files, %d enums evaluated, %d not evaluable, "
          "%d negative + %d positive controls)"
          % (len(files), evaluated, skipped, len(MUST_REJECT), len(MUST_ACCEPT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
