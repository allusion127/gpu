#!/usr/bin/env python3
"""Static contract for the WP8 stage 2 cohort: the middle lifetime.

WHAT THIS PROTECTS.  Stage 1 gave a RASBERY process two lifetimes -- the process
and the case -- and `geometry_builds == cases` in the evaluator receipt was the
machine-readable statement that there was nothing in between.  Stage 2 adds the
middle one: state that is a pure function of the geometry/topology declaration
and the cross-section library, built once per process and shared read-only by
every case of that cohort.

Three ways that is silently wrong, and one check apiece:

  1. THE STATE IS NOT ACTUALLY IMMUTABLE.  A `cohort::Context` is read by up to
     M Driver threads at once with no lock between them.  One non-const member
     is a data race, and the symptom is a digest that moves on a schedule
     nobody can reproduce.  So every member is `const` and this asserts it.

  2. THE KEY COVERS THE WRONG THING.  Too wide (the loading pattern's assembly
     types, the `batch` inventory, the schedule) and every candidate is its own
     cohort: the lever is exactly zero and nothing says so.  Too narrow (the
     dimensions alone, or the deck JSON instead of the built GeometryInput) and
     two different cores share one set of maps: a wrong answer rather than a
     slow one.

  3. THE SHARED THING IS NOT PURE.  `buildPinQuadratureTable` must read nothing
     but its two arguments.  If it ever reaches back into Geometry, the table
     one case built is a table shaped by that case.

Every check runs against a deliberately broken copy of the same text as a
negative control, so a rule that has stopped discriminating fails loudly rather
than passing vacuously.  Pure python: no GPU, no compiler, no RASBERY.

Run:  python tools/test_cohort_context_contract.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def body(text: str, signature: str) -> str:
    """The brace-balanced body that follows `signature`."""
    start = text.find(signature)
    if start < 0:
        raise AssertionError(f"missing signature {signature!r}")
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace:i + 1]
    raise AssertionError(f"unterminated body for {signature!r}")


# ===========================================================================
# 1.  EVERY MEMBER OF Context IS const
# ===========================================================================
def check_immutable(context_h: str) -> list[str]:
    bad: list[str] = []
    try:
        struct = body(strip_comments(context_h), "struct Context")
    except AssertionError as exc:
        return [str(exc)]
    members = [
        line.strip().rstrip(";")
        for line in struct.strip("{}").splitlines()
        if line.strip() and not line.strip().startswith("}")
    ]
    for member in members:
        if not member or member.endswith(")") or "(" in member:
            continue  # a method, not a field
        if not member.startswith("const "):
            bad.append(
                f"cohort::Context member {member!r} is not const. It is read by up to M "
                "Driver threads with no lock between them; a mutable member is a data "
                "race whose only symptom is a digest that moves. If it needs to be "
                "lazily filled, build it eagerly in the builder instead -- a lazily "
                "filled member IS a mutable member")
    if not members:
        bad.append("cohort::Context has no members at all, so this check proves nothing")
    return bad


# ===========================================================================
# 2.  NO LAZY / MUTABLE ESCAPE HATCH IN THE CONTEXT
# ===========================================================================
def check_no_mutable(context_h: str) -> list[str]:
    stripped = strip_comments(context_h)
    bad: list[str] = []
    if re.search(r"\bmutable\b", stripped):
        bad.append("CohortContext.h uses `mutable`: the immutability of the shared state "
                   "is the entire safety argument for sharing it without a lock")
    if re.search(r"\bonce_flag\b|\bcall_once\b", stripped):
        bad.append("CohortContext.h uses call_once: a lazily-filled member of a shared "
                   "const object is a mutable member wearing a hat. Build eagerly")
    return bad


# ===========================================================================
# 3.  THE KEY'S FIELD SET
# ===========================================================================
#
# IN: everything Geometry::Initialize is sized or shaped by.
# OUT: everything a candidate moves.  `batch` is the sharp one -- it is the
# assembly inventory, and what Geometry derives from it (`_is_fuel`, `_kbc`,
# `_kec`, `_hzcore`) is case state a candidate refills, not topology a cohort
# shares.  Putting it in would split a cohort for a reason nothing depends on.
KEY_MUST_COVER = ("ng", "nz", "ndivxy", "npins", "hx", "hy", "hz",
                  "symang", "symopt", "symdiv", "albedo", "occupancyMask")
KEY_MUST_NOT_COVER = ("gin.batch", "config.at(\"batch\")", "config[\"batch\"]")


def check_key_fields(key_h: str) -> list[str]:
    bad: list[str] = []
    try:
        payload = body(key_h, "inline std::string geometryPayload")
    except AssertionError as exc:
        return [str(exc)]
    for field in KEY_MUST_COVER:
        if field not in payload:
            bad.append(
                f"cohort::geometryPayload does not cover {field!r}. Geometry::Initialize "
                "is shaped by it, so two cases that differ in it would share maps built "
                "for the other one -- a wrong answer, not a slow one")
    for field in KEY_MUST_NOT_COVER:
        if field in payload:
            bad.append(
                f"cohort::geometryPayload covers {field!r}. `batch` is what a candidate "
                "moves; keying on it gives every candidate its own cohort and the lever "
                "is exactly zero, with nothing in any receipt saying so")
    # The mask, not the map.  A payload that hashed the core VALUES would make a
    # GA generation N cohorts instead of one.
    if "occupancyMask" not in payload:
        bad.append("the payload does not use the core OCCUPANCY MASK")
    if re.search(r"canonical\(\s*.*core", payload):
        bad.append(
            "cohort::geometryPayload hashes the core MAP, not its occupancy mask: every "
            "candidate of a GA generation would then be its own cohort")
    return bad


def check_key_source(key_h: str, io_cpp: str) -> list[str]:
    """The key is taken from the BUILT GeometryInput, not from the deck JSON."""
    bad: list[str] = []
    if "const GeometryInput& gin" not in key_h:
        bad.append(
            "cohort::geometryPayload does not take a GeometryInput. The deck JSON is the "
            "wrong source: IO's shuffle resolver rewrites `core` in place before "
            "Geometry::Initialize sees it, and a restart-driven case has no geometry "
            "block in its deck at all. The argument of the build is the key for the build")
    call = "cohort::geometryPayload(geometry_input)"
    if call not in io_cpp:
        bad.append(f"IO::ReadInput does not build the cohort key from {call!r}")
    else:
        # ...and it must happen AFTER the shuffle has been applied.
        shuffle = io_cpp.find("geometry_input.core[row][col] = assembly_name;")
        acquire = io_cpp.find(call)
        if shuffle >= 0 and acquire < shuffle:
            bad.append(
                "the cohort key is computed BEFORE the shuffle resolver rewrites "
                "geometry_input.core: two cases whose topology differs because a shuffle "
                "moved an assembly would key into one cohort")
    if "cohort::acquire(" not in io_cpp:
        bad.append("IO::ReadInput never acquires a cohort, so no case is attached to one")
    return bad


# ===========================================================================
# 4.  THE SHARED TABLE IS A PURE FUNCTION OF ITS ARGUMENTS
# ===========================================================================
def check_quadrature_pure(ppr_cpp: str, ppr_h: str) -> list[str]:
    bad: list[str] = []
    try:
        build = body(ppr_cpp, "PinQuadTable rasbery::buildPinQuadratureTable")
    except AssertionError as exc:
        return [str(exc)]
    stripped = strip_comments(build)
    if "_g." in stripped or "_xs." in stripped:
        bad.append(
            "buildPinQuadratureTable reaches into Geometry or XSSet. It is shared across "
            "every case of a cohort, so a table shaped by one case's state would be that "
            "case's table handed to all the others")
    for reached in re.findall(r"\b_[a-z][A-Za-z0-9_]*\b", stripped):
        bad.append(
            f"buildPinQuadratureTable reads the member {reached!r}: a free function of "
            "(ndivxy, npins) may not have members, and one that does is not the pure "
            "function the sharing argument rests on")
    # The consumer must BORROW the table, not own a copy of it.
    if "std::shared_ptr<const PinQuadTable> _pin_quad_table;" not in ppr_h:
        bad.append(
            "PPR still owns its quadrature table by value: a 64-case wave then builds 64 "
            "bit-identical copies of it, which is the cost stage 2 exists to remove")
    if "cohort::acquirePinQuadrature" not in ppr_cpp:
        bad.append("PPR does not take its quadrature table from the cohort")
    return bad


# ===========================================================================
# 5.  CROSS-COHORT ISOLATION: a hit must re-check the shape it hands back
# ===========================================================================
def check_isolation(context_h: str) -> list[str]:
    bad: list[str] = []
    try:
        acquire = body(context_h, "inline std::shared_ptr<const Context> acquire")
    except AssertionError as exc:
        return [str(exc)]
    if "throw" not in acquire:
        bad.append(
            "cohort::acquire never throws. A key that collided with a cohort of another "
            "SHAPE would hand back maps sized for a different core, and the run would "
            "produce numbers that no receipt would mark as wrong")
    for field in ("ng", "ndivxy", "npins"):
        if f"c.{field} != d.{field}" not in acquire:
            bad.append(f"cohort::acquire does not re-check {field!r} on a hit")
    if "e.key == key" not in acquire:
        bad.append("cohort::acquire does not look entries up by the cohort key")
    return bad


# ===========================================================================
# 6.  THE COUNTERS ARE REPORTED, IN EVERY BRANCH
# ===========================================================================
def check_receipts(context_h: str, main_cpp: str, server_h: str) -> list[str]:
    bad: list[str] = []
    for field in ("builds", "hits", "cohorts", "quadrature_builds"):
        if f'\\"{field}\\":' not in context_h:
            bad.append(f"the [RASBERY][COHORT] receipt is missing {field!r}")
    # main.cpp has three teardown blocks -- evaluator, batch, serial -- and a
    # receipt that only one of them prints is a receipt an A/B cannot use.
    printed = main_cpp.count("rasbery::cohort::printReceipt(std::cout);")
    if printed < 3:
        bad.append(
            f"[RASBERY][COHORT] is printed in {printed} of main.cpp's three teardown "
            "blocks (evaluator, batch, serial). An A/B whose two arms print different "
            "receipts is an A/B that cannot be read")
    for field in ("cohort_builds", "cohort_hits"):
        if f'\\"{field}\\":' not in server_h:
            bad.append(f"the [RASBERY][EVALUATOR] process receipt is missing {field!r}")
    # geometry_builds must SURVIVE: it is the honest statement that the Geometry
    # object itself is still per case, and folding it into cohort_builds would
    # let one number stand for two different facts.
    if '\\"geometry_builds\\":' not in server_h:
        bad.append(
            "the process receipt dropped `geometry_builds`. It is not the same claim as "
            "`cohort_builds`: the Geometry object is still built per case, and a receipt "
            "that says otherwise is claiming a lever that was not pulled")
    return bad


# ===========================================================================
# RUN
# ===========================================================================
CONTEXT = read("src/CohortContext.h")
KEY = read("src/CohortKey.h")
IO_CPP = read("src/IO.cpp")
PPR_CPP = read("src/PPR.cpp")
PPR_H = read("src/PPR.h")
MAIN = read("src/main.cpp")
SERVER = read("src/EvaluatorServer.h")

failures += check_immutable(CONTEXT)
failures += check_no_mutable(CONTEXT)
failures += check_key_fields(KEY)
failures += check_key_source(KEY, IO_CPP)
failures += check_quadrature_pure(PPR_CPP, PPR_H)
failures += check_isolation(CONTEXT)
failures += check_receipts(CONTEXT, MAIN, SERVER)

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Every check above, run against a copy broken in the exact
# way the check exists to catch.  A check that still passes here is a comment.
# ---------------------------------------------------------------------------
negative: list[str] = []


def control(name: str, checker, *args) -> None:
    if not checker(*args):
        negative.append(name)


control("check_immutable misses a non-const member",
        check_immutable,
        CONTEXT.replace("    const int         ng;", "    int               ng;"))
control("check_no_mutable misses a mutable member",
        check_no_mutable,
        CONTEXT.replace("struct Context {", "struct Context {\n    mutable int cache;"))
ALBEDO_BLOCK = '''    out += "\\nalbedo";
    for (const double a : gin.albedo) {
        out += '\\t';
        out += std::format("{:.17g}", a);
    }
'''
assert KEY.count(ALBEDO_BLOCK) == 1, "the albedo negative control no longer matches the source"
control("check_key_fields misses the boundary condition dropped from the key",
        check_key_fields, KEY.replace(ALBEDO_BLOCK, ""))
control("check_key_fields misses the batch inventory leaking into the key",
        check_key_fields,
        KEY.replace('out += "\\nmask\\t";', 'out += canonical(gin.batch);\n    out += "\\nmask\\t";'))
control("check_key_source misses a key read off the deck JSON",
        check_key_source, KEY.replace("const GeometryInput& gin", "const Json& config"),
        IO_CPP)
control("check_key_source misses a key computed before the shuffle",
        check_key_source, KEY,
        IO_CPP.replace("cohort::geometryPayload(geometry_input)", "X")
              .replace("_deck_key_digest =", "cohort::geometryPayload(geometry_input);\n    _deck_key_digest ="))
control("check_quadrature_pure misses a table that reaches into Geometry",
        check_quadrature_pure,
        PPR_CPP.replace("PinQuadTable            table;",
                        "PinQuadTable            table;\n    const int z = _g.nz();"),
        PPR_H)
control("check_quadrature_pure misses PPR owning its own copy again",
        check_quadrature_pure, PPR_CPP,
        PPR_H.replace("std::shared_ptr<const PinQuadTable> _pin_quad_table;",
                      "PinQuadTable _pin_quad_table;"))
control("check_isolation misses a hit that does not re-check its shape",
        check_isolation, CONTEXT.replace("c.ndivxy != d.ndivxy", "false"))
control("check_receipts misses the receipt vanishing from a branch",
        check_receipts, CONTEXT,
        MAIN.replace("rasbery::cohort::printReceipt(std::cout);", "", 1), SERVER)
control("check_receipts misses geometry_builds being folded away",
        check_receipts, CONTEXT, MAIN,
        SERVER.replace('\\"geometry_builds\\":', '\\"gb\\":'))

# ===========================================================================
# THE COMPILED HALF -- the one that closes the loop, when a compiler is here
# ===========================================================================
#
# Everything above is a source scan, and a source scan cannot answer the only
# question that matters: DOES THE KEY PUT THE RIGHT CASES IN THE SAME COHORT?
# So when a C++ compiler is available, CohortKey.h and CohortContext.h are
# compiled and driven directly, and six properties are asserted on real digests:
#
#   permuted    two candidates that permute assembly TYPES over one footprint
#               are ONE cohort.  This is the lever; without it stage 2 buys
#               nothing and no other number would say so.
#   holed       a candidate that EMPTIES a lattice position is a DIFFERENT
#               cohort.  This is the correctness half: the neighbour and index
#               maps are a function of which positions are occupied.
#   albedo      a different boundary condition is a different cohort.
#   library     a different cross-section library is a different cohort.
#   ragged      a row written short and the same row padded with "XX" are one
#               cohort -- a whitespace difference must not split one.
#   batch       the assembly inventory does not enter the key.
#
# ...plus three on the registry: the same descriptor returns the SAME object,
# two cohorts are isolated from each other, and two cohorts that agree on
# (ndivxy, npins) SHARE one quadrature table rather than building two.
#
# GeometryInput IS STUBBED, deliberately.  The real src/Geometry.h reaches
# pch.h -> highfive -> the HDF5 C headers, which are a build-system dependency
# and not a property of this key.  The stub is the struct verbatim; a drift
# between it and the real one is caught by the source check below, which reads
# the real declaration and compares the field list.
#
# No compiler is not a failure -- it is a SKIP, reported as one, because a test
# that silently degrades to nothing is worse than a test that says it did.
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

STUB_GEOMETRY_H = """#pragma once
#include <array>
#include <map>
#include <string>
#include <vector>
namespace rasbery {
struct GeometryInput {
    int ng; int nz; int ndivxy; int npins;
    double hx, hy;
    std::vector<double> hz;
    int symang; bool symopt; bool symdiv;
    std::array<double, 6> albedo;
    std::vector<std::vector<std::string>> core;
    std::map<std::string, std::vector<std::string>> batch;
};
}
"""

# The fields the stub must mirror.  Read off the real declaration so a field
# added to GeometryInput and NOT added to the cohort key fails here rather than
# silently leaving the key blind to it.
GEOMETRY_INPUT_FIELDS = ("ng", "nz", "ndivxy", "npins", "hx", "hy", "hz",
                         "symang", "symopt", "symdiv", "albedo", "core", "batch")

HARNESS_CPP = r"""
#include "CohortKey.h"
#include "CohortContext.h"
#include <iostream>
// The real one is in PPR.cpp; its arithmetic is not what this harness checks.
namespace rasbery { PinQuadTable buildPinQuadratureTable(int, int) { return {}; } }
static rasbery::GeometryInput base() {
    rasbery::GeometryInput g{};
    g.ng = 2; g.nz = 3; g.ndivxy = 2; g.npins = 17;
    g.hx = 21.5; g.hy = 21.5; g.hz = {10.0, 20.0, 10.0};
    g.symang = 90; g.symopt = true; g.symdiv = false;
    g.albedo = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    g.core = {{"A1", "B2"}, {"C3", "XX"}};
    return g;
}
static std::string k(const rasbery::GeometryInput& g, const char* lib) {
    return rasbery::cohort::keyOf(g, lib);
}
int main() {
    using namespace rasbery;
    const GeometryInput a = base();
    GeometryInput permuted = base();
    permuted.core = {{"C3", "A1"}, {"B2", "XX"}};
    GeometryInput holed = base();
    holed.core = {{"A1", "XX"}, {"C3", "XX"}};
    GeometryInput albedo = base();
    albedo.albedo[0] = 0.4;
    GeometryInput ragged = base();
    ragged.core = {{"A1", "B2"}, {"C3"}};
    GeometryInput inventory = base();
    inventory.batch["A1"] = {"F1", "F1", "F1"};
    GeometryInput mesh = base();
    mesh.hz = {10.0, 20.0, 10.5};

    std::cout << "permuted " << (k(a,"L") == k(permuted,"L")) << "\n";
    std::cout << "holed " << (k(a,"L") != k(holed,"L")) << "\n";
    std::cout << "albedo " << (k(a,"L") != k(albedo,"L")) << "\n";
    std::cout << "library " << (k(a,"L") != k(a,"M")) << "\n";
    std::cout << "mesh " << (k(a,"L") != k(mesh,"L")) << "\n";
    std::cout << "ragged " << (k(a,"L") == k(ragged,"L")) << "\n";
    std::cout << "batch " << (k(a,"L") == k(inventory,"L")) << "\n";
    cohort::Descriptor d{Sha256::hexOf(cohort::geometryPayload(a)), "L",
                         a.ng, a.ndivxy, a.npins};
    auto c1 = cohort::acquire(d);
    auto c2 = cohort::acquire(d);
    cohort::Descriptor e{Sha256::hexOf(cohort::geometryPayload(holed)), "L", 2, 2, 17};
    auto c3 = cohort::acquire(e);
    std::cout << "same_object " << (c1.get() == c2.get()) << "\n";
    std::cout << "isolated " << (c1.get() != c3.get()) << "\n";
    std::cout << "quadrature_shared " << (c1->ppr_quadrature == c3->ppr_quadrature) << "\n";
    const cohort::Stats s = cohort::snapshot();
    std::cout << "builds " << (s.builds == 2) << "\n";
    std::cout << "hits " << (s.hits == 1) << "\n";
    std::cout << "quad_builds " << (s.quadrature_builds == 1) << "\n";
    return 0;
}
"""


def find_compiler():
    """MSVC's vcvars64.bat, or a g++/clang++ on PATH, or None."""
    if os.name == "nt":
        for base in (r"C:\Program Files\Microsoft Visual Studio",
                     r"C:\Program Files (x86)\Microsoft Visual Studio"):
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                if "vcvars64.bat" in files:
                    return os.path.join(dirpath, "vcvars64.bat")
    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def compiled_contract() -> bool:
    """True when the compiled half actually ran."""
    # The stub must still describe the real struct.
    declaration = body(read("src/Geometry.h"), "struct GeometryInput")
    for field in GEOMETRY_INPUT_FIELDS:
        if field not in declaration:
            failures.append(
                f"GeometryInput no longer declares {field!r}: the compiled harness's stub "
                "has drifted from the real struct, and a harness built on a stale shape "
                "proves things about a type that does not exist")
    for name in re.findall(r"^\s+(?:int|double|bool|std::[^;]+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*([A-Za-z_][A-Za-z0-9_]*)\s*)?;",
                           declaration, re.M):
        for field in name:
            if field and field not in GEOMETRY_INPUT_FIELDS:
                failures.append(
                    f"GeometryInput gained the field {field!r} and the cohort key has not "
                    "been told about it. Either it shapes the topology -- in which case "
                    "cohort::geometryPayload must cover it, or two different cores will "
                    "share one set of maps -- or it does not, in which case say so here")

    compiler = find_compiler()
    if compiler is None:
        return False
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        stub = tmp / "stub"
        stub.mkdir()
        (stub / "Geometry.h").write_text(STUB_GEOMETRY_H, encoding="utf-8")
        # The three headers under test are COPIED beside the stub, byte for
        # byte, at test time.  They have to be: a quoted `#include "Geometry.h"`
        # resolves relative to the INCLUDING file first, so a CohortKey.h left
        # in src/ would reach the real Geometry.h -- and pch.h -> highfive ->
        # the HDF5 C headers -- whatever /I order this passes.  Everything else
        # they include (CaseKey.h, Sha256.h) is still the repository's own,
        # because those resolve through /I and carry no build dependency.
        for name in ("CohortKey.h", "CohortContext.h", "PprQuadrature.h"):
            (stub / name).write_text(read("src/" + name), encoding="utf-8")
        cpp = tmp / "cohort_harness.cpp"
        cpp.write_text(HARNESS_CPP, encoding="utf-8")
        exe = tmp / ("cohort_harness.exe" if os.name == "nt" else "cohort_harness")
        includes = [stub, ROOT / "src", ROOT / "include", ROOT / "include" / "chiffon"]
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                # QUOTED include paths: this repository's own path contains an
                # `&`, which cmd would otherwise read as a command separator.
                script = tmp / "build_cohort_harness.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul 2>&1\r\n' % compiler
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
            failures.append("the cohort harness does not compile:\n"
                            + (failure.stdout or "") + (failure.stderr or ""))
            return True

        done = subprocess.run([str(exe)], capture_output=True, universal_newlines=True)
        if done.returncode != 0:
            failures.append(f"the cohort harness failed: {done.stdout}{done.stderr}")
            return True
        results = dict(line.split() for line in done.stdout.split("\n") if line.strip())
        expected = {
            "permuted": "two candidates that permute assembly TYPES over one footprint "
                        "must be ONE cohort -- this IS the lever, and without it stage 2 "
                        "buys nothing while every other number stays right",
            "holed": "a candidate that EMPTIES a lattice position must be a DIFFERENT "
                     "cohort: the index and neighbour maps are a function of which "
                     "positions are occupied, so sharing here is a wrong answer",
            "albedo": "a different boundary condition must be a different cohort",
            "library": "a different cross-section library must be a different cohort",
            "mesh": "a different axial mesh must be a different cohort",
            "ragged": "a ragged row and the same row padded with \"XX\" are the same core, "
                      "so they must not be two cohorts for a whitespace reason",
            "batch": "the assembly inventory must NOT enter the key: it is what a "
                     "candidate moves, and keying on it makes every candidate its own "
                     "cohort",
            "same_object": "one descriptor must return one Context, not two",
            "isolated": "two cohorts must not share a Context",
            "quadrature_shared": "two cohorts with the same (ndivxy, npins) must SHARE one "
                                 "quadrature table: it is a pure function of that pair, and "
                                 "building it twice is the cost stage 2 removes",
            "builds": "cohort_builds must count the distinct cohorts, here 2",
            "hits": "cohort_hits must count the acquisitions served, here 1",
            "quad_builds": "the quadrature must be built ONCE for both cohorts",
        }
        for name, why in expected.items():
            if results.get(name) != "1":
                failures.append(f"compiled cohort contract [{name}]: {why} "
                                f"(harness said {results.get(name)!r})")
    return True


if not compiled_contract():
    print("cohort context: NOTE -- no C++ compiler found; the compiled half was SKIPPED "
          "and only the source scan ran", file=sys.stderr)

if negative:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are therefore "
                    "comments:\n    " + "\n    ".join(negative))

if failures:
    raise SystemExit("cohort context: FAIL\n  " + "\n  ".join(failures))
print("cohort context: PASS")
