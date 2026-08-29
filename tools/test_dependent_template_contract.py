#!/usr/bin/env python3
"""Dependent member-template calls must carry the `template` keyword.

WHY THIS EXISTS.  `eab1a43` did not compile on 238 (GCC 14.3) and did compile
here.  `src/CaseKey.h`'s `appendValue` is `template <class Json>`, so inside it
`value` has a DEPENDENT type and `value.get<bool>()` is ambiguous to a
standard-conforming parser: `<` is read as less-than, and the error surfaces as
`expected primary-expression before 'bool'`.  The fix is the `template`
disambiguator -- `value.template get<bool>()`.  MSVC's permissive parser accepts
the undisambiguated form, so the LOCAL compile gate cannot see this class of
break at all; six call sites shipped and the tree was uncompilable on the only
machine that matters.

WHAT THIS CHECKS.  Every function/method body under a `template <...>` header
that names at least one type parameter, for a member call `x.f<...>` / `x->f<...>`
where `f` is one of the known member templates of the JSON / stream types the
tree uses and `x` is rooted in a dependent name, WITHOUT a preceding `template`.

WHAT IT DOES NOT CHECK.  It is a lexical scan, not a compiler.  It cannot see a
dependent type reached through an alias, and it deliberately only flags a small
list of member-template names (`get`, `at`, `value`), because those are the ones
nlohmann::json exposes and the ones this tree calls.  A wider net on a lexical
scan is a net full of false positives, and a contract test that cries wolf gets
switched off.  The real gate is still GCC on 238; this is the gate that runs
before the push.

USAGE
    tools/test_dependent_template_contract.py
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The member templates this tree actually calls on JSON-ish values.  `value` is
# here because nlohmann::json has BOTH a two-argument non-template `value(key,
# default)` and a `value<T>()`; only the angle-bracket spelling is a member
# template, and only that spelling is matched below.
MEMBER_TEMPLATES = ("get", "at", "value", "get_to", "get_ptr")

MEMBER_CALL = re.compile(
    r"(?:\.|->)\s*(" + "|".join(MEMBER_TEMPLATES) + r")\s*<")

TEMPLATE_HEADER = re.compile(r"\btemplate\s*<([^<>]*(?:<[^<>]*>[^<>]*)*)>")
TYPE_PARAM = re.compile(r"(?:class|typename)\s+(?:\.\.\.\s*)?(\w+)")


def strip_comments(text: str) -> str:
    """Prose must never satisfy -- or trip -- a code assertion."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def brace_body(text: str, start: int):
    """The brace-matched body that follows `start`, or None if a `;` comes first.

    A `template <...> void f(...);` declaration has no body and cannot contain a
    call, so it is skipped rather than mis-parsed as the next function's.
    """
    open_at = text.find("{", start)
    semi = text.find(";", start)
    if open_at < 0 or (0 <= semi < open_at):
        return None
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def dependent_roots(header_params: str, body: str) -> set[str]:
    """Names in `body` whose type depends on a template parameter.

    Seeded from the parameter list (a parameter whose declared type mentions a
    type parameter) and grown through `for (auto x : <root>)` and
    `auto& x = <root>...`, which is how this tree reaches into a JSON tree.
    """
    tparams = set(TYPE_PARAM.findall(header_params))
    if not tparams:
        return set()
    roots: set[str] = set()
    args = body[body.find("(") + 1:body.find(")")] if "(" in body else ""
    for decl in args.split(","):
        name = re.search(r"(\w+)\s*(?:=[^,]*)?$", decl.strip())
        if name is None:
            continue
        if any(re.search(r"\b" + re.escape(t) + r"\b", decl) for t in tparams):
            roots.add(name.group(1))
    # Two rounds is enough for this tree (parameter -> item -> nothing deeper),
    # and a fixed bound keeps a pathological file from looping.
    for _ in range(2):
        for m in re.finditer(r"\bfor\s*\(\s*(?:const\s+)?auto\s*&{0,2}\s*(\w+)\s*:\s*(\w+)",
                             body):
            if m.group(2) in roots:
                roots.add(m.group(1))
        for m in re.finditer(r"\b(?:const\s+)?auto\s*&{0,2}\s*(\w+)\s*=\s*(\w+)\b", body):
            if m.group(2) in roots:
                roots.add(m.group(1))
    return roots


def offenders(text: str, label: str) -> list[str]:
    """Every undisambiguated dependent member-template call in `text`."""
    code = strip_comments(text)
    found: list[str] = []
    for header in TEMPLATE_HEADER.finditer(code):
        body = brace_body(code, header.end())
        if body is None:
            continue
        roots = dependent_roots(header.group(1), body)
        if not roots:
            continue
        for call in MEMBER_CALL.finditer(body):
            before = body[:call.start()]
            if re.search(r"\btemplate\s*$", before):
                continue
            # The receiver's ROOT identifier: walk back over a `a.b(...).c` chain
            # to the leading name.
            chain = re.search(r"([A-Za-z_]\w*)((?:\s*(?:\.|->)\s*\w+\s*\([^()]*\))*)\s*$",
                              before)
            if chain is None or chain.group(1) not in roots:
                continue
            line = code[:header.end() + call.start()].count("\n") + 1
            found.append(f"{label}:{line}: `{chain.group(1)}...{call.group(1)}<` on a "
                         f"dependent name without the `template` keyword -- GCC "
                         f"rejects this, MSVC accepts it")
    return found


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  A scan that cannot fail is a scan that reports PASS.
# ---------------------------------------------------------------------------
MUST_REJECT = (
    ("the CaseKey.h break itself",
     "template <class Json>\n"
     "inline void appendValue(std::string& out, const Json& value) {\n"
     "    out += value.get<bool>() ? 'T' : 'F';\n"
     "}\n"),
    ("a dependent call reached through a chain",
     "template <class Json>\n"
     "inline void f(const Json& config) {\n"
     '    auto x = config.at("core").get<CoreMap>();\n'
     "}\n"),
    ("a dependent call on a range-for binding",
     "template <class Json>\n"
     "inline void f(const Json& value) {\n"
     "    for (const auto& item : value) { g(item.get<double>()); }\n"
     "}\n"),
)

MUST_ACCEPT = (
    ("the disambiguated form",
     "template <class Json>\n"
     "inline void appendValue(std::string& out, const Json& value) {\n"
     "    out += value.template get<bool>() ? 'T' : 'F';\n"
     "}\n"),
    ("a non-dependent receiver inside a template",
     "template <class T>\n"
     "inline void f(const nlohmann::json& object, T& sink) {\n"
     '    sink = object["op"].get<std::string>();\n'
     "}\n"),
    ("nlohmann's two-argument value(), which is not a member template",
     "template <class Json>\n"
     "inline void f(const Json& sym) {\n"
     '    int angle = sym.value("angle", 0);\n'
     "}\n"),
    ("a template DECLARATION with no body",
     "template <class Json>\n"
     "void appendValue(std::string& out, const Json& value);\n"
     "inline void g(const nlohmann::json& j) { j.get<int>(); }\n"),
)


def main() -> int:
    failures: list[str] = []

    files = sorted(
        list((ROOT / "src").rglob("*.h")) + list((ROOT / "src").rglob("*.cpp"))
        + list((ROOT / "src").rglob("*.cu")) + list((ROOT / "src").rglob("*.cuh"))
        + list((ROOT / "include" / "chiffon").rglob("*.h"))
        + list((ROOT / "test").rglob("*.cpp")) + list((ROOT / "test").rglob("*.cu"))
    )
    for path in files:
        failures.extend(
            offenders(path.read_text(encoding="utf-8-sig"),
                      str(path.relative_to(ROOT)).replace("\\", "/")))

    for name, source in MUST_REJECT:
        if not offenders(source, "<control>"):
            failures.append(f"negative control PASSED the scan -- the scan is "
                            f"vacuous for: {name}")
    for name, source in MUST_ACCEPT:
        hits = offenders(source, "<control>")
        if hits:
            failures.append(f"positive control REJECTED -- the scan cries wolf on: "
                            f"{name} ({hits[0]})")

    if failures:
        print("dependent template contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("dependent template contract: PASS (%d files, %d negative + %d positive "
          "controls)" % (len(files), len(MUST_REJECT), len(MUST_ACCEPT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
