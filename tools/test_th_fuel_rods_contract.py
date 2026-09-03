#!/usr/bin/env python3
"""Contract: the fuel-temperature divisor is ONE value, named, with one 62.0.

THE DEFECT THIS CLOSES.  XSSet::SolveTH indexes the Tfuel table with a linear
power density

    lpd = 1000 * P_node / (RODS * hz[k])

and RODS was the literal `62.0`, written three times -- the host body
(src/XSSet.cpp), the ported body (src/ThKernel.h) and the verbatim quotation the
form mask is mined against (src/ThReference.cpp) -- and inherited unexamined
from the CPU baseline (e76d40d).  It is wrong for both campaign decks:

    APR1400  16x16, 236 rods/assembly, ndivxy=2 -> 59 rods/node
    i-SMR    17x17, 260 rods/assembly, ndivxy=2 -> 65 rods/node

The measured i-SMR RASBERY/MASTER fuel-temperature-RISE ratio is 1.047-1.062,
which is 65/62 = 1.048; the bias is about +9 K on tfavg, roughly -25 pcm.

WHAT IS ASSERTED, AND WHY EACH ONE IS HERE.

  1. ONE 62.0.  A literal that appears twice is a fix that gets half-applied.
     The constant survives in src/ThFuelRods.h and nowhere else in the T/H path.
  2. THREE BODIES, ONE FIELD.  Host, device and quotation must divide by the
     SAME resolved value, or the mined form mask scores a quotation that is not
     the shipped arithmetic and the B0 claim evaporates.
  3. EVERY VIEW FILLER SETS IT.  ThView has no default (0.0 divides loudly);
     XSSet, the CUDA backend and the mining view must each carry it across.
  4. THE FIXTURE STAYS LEGACY, so the mask a host mines does not move when a
     deck declares a rod count.
  5. THE DECK KEY, parsed under geometry.dimensions, with the aliases the rest
     of IO.cpp uses, and folded by ndivxy^2 -- assembly count in, node count out.
  6. THE ENVIRONMENT, resolved by the real C++ function when a compiler is
     available: legacy | deck | <number>, and a REFUSAL rather than a guess when
     `deck` is asked for and the deck says nothing.
  7. THE CASE KEY, which must fork on the VALUE and not on the SOURCE.

USAGE
    tools/test_th_fuel_rods_contract.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import _cxx_toolchain  # noqa: E402
import case_key  # noqa: E402

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


TH_PATH_FILES = ("src/XSSet.cpp", "src/ThKernel.h", "src/ThReference.cpp",
                 "src/ThReference.h", "src/ThFormMine.h", "src/CudaThBackend.h",
                 "src/CudaThBackend.cu", "src/Geometry.cpp", "src/Geometry.h")


def code_lines(text: str) -> str:
    """The file with its `//` comment lines dropped -- prose may say 62, code may not."""
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("//"))


def one_literal() -> None:
    fuel_rods = code_lines(read("src/ThFuelRods.h"))
    if fuel_rods.count("62.0") != 1:
        fail(f"src/ThFuelRods.h spells 62.0 {fuel_rods.count('62.0')} times in code; "
             "the legacy divisor is one named constant or it is a literal again")
    if "inline constexpr double kLegacyFuelRodsPerNode = 62.0;" not in fuel_rods:
        fail("kLegacyFuelRodsPerNode is not the constant that holds the 62.0")
    for rel in TH_PATH_FILES:
        if "62.0" in code_lines(read(rel)):
            fail(f"{rel} contains the literal 62.0; the divisor must be read from "
                 "Geometry::fuel_rods_per_node() / the view field, and the only "
                 "62.0 in the tree is ThFuelRods.h's named fallback")


def three_bodies_one_field() -> None:
    sites = {
        "src/XSSet.cpp":
            "const double lpd   = 1000.0 * P_node / (_g.fuel_rods_per_node() * _g.hz(k));",
        "src/ThKernel.h":
            "const double lpd = 1000.0 * P_node / (v.fuel_rods_per_node * v.hz[k]);",
        "src/ThReference.cpp":
            "const double lpd = 1000.0 * P_node / (f.fuel_rods_per_node * "
            "f.hz[static_cast<std::size_t>(k)]);",
    }
    for rel, expression in sites.items():
        if expression not in read(rel):
            fail(f"{rel} no longer divides the linear power density by the resolved "
                 f"rods-per-node; expected:\n    {expression}")


def every_filler_sets_it() -> None:
    if "double fuel_rods_per_node   = 0.0;" not in read("src/ThKernel.h"):
        fail("th::ThView::fuel_rods_per_node gained a default; a default here is the "
             "wrong literal coming back through a filler somebody forgot")
    fillers = {
        "src/XSSet.cpp": "v.fuel_rods_per_node   = _g.fuel_rods_per_node();",
        "src/CudaThBackend.cu": "k.fuel_rods_per_node   = v.fuel_rods_per_node;",
        "src/ThFormMine.h": "v.fuel_rods_per_node   = f.fuel_rods_per_node;",
    }
    for rel, line in fillers.items():
        if line not in read(rel):
            fail(f"{rel} does not carry the divisor into the view; expected:\n    {line}")
    if "double fuel_rods_per_node   = 0.0;" not in read("src/CudaThBackend.h"):
        fail("thgpu::UpdateView lost its fuel_rods_per_node field")


def fixture_stays_legacy() -> None:
    ref = read("src/ThReference.cpp")
    if "f.fuel_rods_per_node   = rasbery::th::kLegacyFuelRodsPerNode;" not in ref:
        fail("the mining fixture does not pin the LEGACY divisor; the mined T/H form "
             "mask would move the day a deck declares a rod count, and the mask is "
             "supposed to be a property of the host, not of the deck")
    if '#include "ThKernel.h"' in ref:
        fail("ThReference.cpp includes the shipped bodies -- the quotation's whole "
             "contract (see tools/test_th_gpu_contract.py) is that it does not")


def deck_key() -> None:
    io_cpp = read("src/IO.cpp")
    for alias in ('"nfrod"', '"fuel rods per assembly"', '"fuel_rods_per_assembly"'):
        if alias not in io_cpp:
            fail(f"IO.cpp does not accept the deck key {alias}")
    if "geometry_input.nfrod = v->get<int>();" not in io_cpp:
        fail("IO.cpp does not parse the fuel-rod count into GeometryInput::nfrod")
    geom = read("src/Geometry.cpp")
    if "th::resolveFuelRodsPerNode(deck_rods)" not in geom:
        fail("Geometry::Initialize does not resolve the divisor through "
             "th::resolveFuelRodsPerNode; a second resolution is a second answer")
    if "static_cast<double>(_ndivxy2)" not in geom:
        fail("Geometry::Initialize does not fold the deck's per-ASSEMBLY count by "
             "ndivxy^2; the bodies divide per NODE")
    if "[RASBERY][TH][NFROD]" not in geom:
        fail("no [RASBERY][TH][NFROD] receipt; a divisor nobody can see in a log is "
             "how 62 survived from e76d40d to this campaign")

    # The python mirror, on the arithmetic itself.
    ismr = {"geometry": {"dimensions": {"ng": 2, "xydivision": 2, "npins": 17,
                                        "nfrod": 260}}}
    apr = {"geometry": {"dimensions": {"ng": 2, "xydivision": 2, "npins": 16,
                                       "nfrod": 236}}}
    silent = {"geometry": {"dimensions": {"ng": 2, "xydivision": 2, "npins": 17}}}
    for label, config, want in (("i-SMR", ismr, 65.0), ("APR1400", apr, 59.0),
                                ("no key", silent, 0.0)):
        got = case_key.deck_fuel_rods_per_node(config)
        if got != want:
            fail(f"case_key.deck_fuel_rods_per_node({label}) = {got}, want {want}")
    alias_deck = {"geometry": {"dimensions": {"xydivision": 2,
                                              "fuel rods per assembly": 260}}}
    if case_key.deck_fuel_rods_per_node(alias_deck) != 65.0:
        fail("the tool does not accept the long spelling of the deck key")

    for env, want in ((None, ("62", "legacy_62")),
                      ("legacy", ("62", "legacy_62")),
                      ("deck", ("65", "deck")),
                      ("59", ("59", "env"))):
        environ = {} if env is None else {"RASBERY_TH_FUEL_RODS": env}
        got = case_key.th_fuel_rods(ismr, environ)
        if got != want:
            fail(f"case_key.th_fuel_rods({env!r}) = {got}, want {want}")
    # NEGATIVE CONTROL: `deck` against a deck that declares nothing must refuse,
    # never fall back and never guess npins^2 minus guide tubes.
    try:
        case_key.th_fuel_rods(silent, {"RASBERY_TH_FUEL_RODS": "deck"})
        fail("NEGATIVE CONTROL FAILED: the tool accepted RASBERY_TH_FUEL_RODS=deck "
             "on a deck with no fuel-rod count")
    except SystemExit:
        pass


HARNESS_CPP = r'''
// Resolve the divisor with the REAL function, and print what it decided.
#include "ThFuelRods.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    const double deck = argc > 1 ? std::atof(argv[1]) : 0.0;
    try {
        const rasbery::th::FuelRods rods = rasbery::th::resolveFuelRodsPerNode(deck);
        std::printf("%.17g %s\n", rods.value, rods.source.c_str());
    } catch (const std::exception&) {
        std::printf("REFUSED\n");
    }
    return 0;
}
'''


def compiled_resolver() -> bool:
    toolchain, reason = _cxx_toolchain.discover(ROOT)
    if toolchain is None:
        print(f"th fuel rods compiled contract: SKIP -- {reason}", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        cpp = tmp / "th_fuel_rods_harness.cpp"
        cpp.write_text(HARNESS_CPP, encoding="utf-8")
        exe = tmp / ("harness.exe" if os.name == "nt" else "harness")
        includes = [ROOT / "src"]
        try:
            if toolchain.is_msvc:
                script = tmp / "build.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul\r\n' % toolchain.compiler
                    + 'cd /d "%s"\r\n' % tmp
                    + 'cl /nologo %s /EHsc /D_CRT_SECURE_NO_WARNINGS "%s" %s /Fe:"%s"\r\n'
                      % (toolchain.std_flag, cpp,
                         " ".join('/I "%s"' % d for d in includes), exe),
                    encoding="utf-8")
                subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(tmp),
                               capture_output=True, universal_newlines=True)
            else:
                subprocess.run(
                    [toolchain.compiler, toolchain.std_flag, "-O0", str(cpp), "-o", str(exe)]
                    + [arg for d in includes for arg in ("-I", str(d))],
                    check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            fail("the ThFuelRods harness does not compile:\n"
                 + (failure.stdout or "") + (failure.stderr or ""))
            return True

        base = {k: v for k, v in os.environ.items() if k != "RASBERY_TH_FUEL_RODS"}

        def run(value, deck="65"):
            environ = dict(base)
            if value is not None:
                environ["RASBERY_TH_FUEL_RODS"] = value
            done = subprocess.run([str(exe), deck], capture_output=True,
                                  universal_newlines=True, env=environ)
            return done.stdout.strip()

        cases = (
            (None, "65", "62 legacy_62", "unset must be the legacy divisor"),
            ("", "65", "62 legacy_62", "empty must be the legacy divisor"),
            ("legacy", "65", "62 legacy_62", "'legacy' must be the legacy divisor"),
            (" legacy ", "65", "62 legacy_62", "the value is trimmed"),
            ("deck", "65", "65 deck", "'deck' must take the deck's count"),
            ("deck", "0", "REFUSED", "'deck' with no deck count must REFUSE, not guess"),
            ("59", "0", "59 env", "a bare number is the rods-per-node override"),
            ("0", "65", "REFUSED", "zero is not a divisor"),
            ("-4", "65", "REFUSED", "a negative count is not a divisor"),
            ("sixty-two", "65", "REFUSED", "a word that is not legacy|deck must refuse"),
            ("62x", "65", "REFUSED", "trailing junk must refuse, not parse a prefix"),
        )
        for value, deck, want, why in cases:
            got = run(value, deck)
            if got != want:
                fail(f"RASBERY_TH_FUEL_RODS={value!r} (deck={deck}) -> {got!r}, "
                     f"want {want!r}: {why}")
    return True


def case_key_folds_value_not_source() -> None:
    """The key must fork on the VALUE and be blind to where it came from."""
    def deck_json(nfrod=None):
        core = [["A1", "A1"], ["A1", "A1"]]
        dims = {"ng": 2, "xydivision": 2, "npins": 17}
        if nfrod is not None:
            dims["nfrod"] = nfrod
        return {
            "geometry": {
                "dimensions": dims,
                "size": {"hx": 20.0, "hy": 20.0, "hz": [10.0, 10.0]},
                "symmetry": {"angle": 90, "mirror": True,
                             "center assembly divided": False},
                "albedo": {"west": 0.0, "east": 0.5, "north": 0.0, "south": 0.5,
                           "bottom": 0.5, "top": 0.5},
                "core": core,
            },
            "batch": {"A1": ["F1", "F1"]},
        }

    clean = {k: v for k, v in os.environ.items() if not k.startswith("RASBERY_")}
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        ismr = tmp / "ismr.json"
        ismr.write_text(json.dumps(deck_json(260)), encoding="utf-8")
        legacy_deck = tmp / "legacy.json"
        legacy_deck.write_text(json.dumps(deck_json(248)), encoding="utf-8")

        def key(path, env=None):
            return case_key.case_key(path, env=dict(clean, **(env or {})),
                                     xslib=False)["case_key"]

        default_key = key(ismr)
        deck_key_ = key(ismr, {"RASBERY_TH_FUEL_RODS": "deck"})
        if default_key == deck_key_:
            fail("the case key does not fork when the divisor moves from 62 to 65; "
                 "two different fuel temperatures would share one cache entry")
        # 248/4 = 62: a deck that DECLARES the legacy divisor is the legacy
        # arithmetic and must key identically to the legacy default.
        if key(legacy_deck, {"RASBERY_TH_FUEL_RODS": "deck"}) != key(legacy_deck):
            fail("a deck that declares 62 rods/node keys differently from the legacy "
                 "default; the SOURCE is not the arithmetic and must not be folded")
        # And the default is LEGACY for every deck, declared or not: a deck that
        # states 65 rods/node still keys -- and still runs -- at 62 until somebody
        # asks for the deck value.  (The deck key itself is part of the deck
        # digest, as every deck value is; what must not move on its own is the
        # ARITHMETIC.)
        payload = case_key.case_key(ismr, env=dict(clean), xslib=False)["payload"]
        if "\nth_fuel_rods\t62\n" not in payload:
            fail("a deck that declares nfrod=260 resolved away from the legacy "
                 "divisor with no override asked for; B0 requires the default to "
                 "stay 62 until somebody says otherwise")
        if key(ismr, {"RASBERY_TH_FUEL_RODS": "65"}) != deck_key_:
            fail("the numeric override and the deck value do not key alike at the "
                 "same number; the key folds the value, not the request")


def main() -> int:
    one_literal()
    three_bodies_one_field()
    every_filler_sets_it()
    fixture_stays_legacy()
    deck_key()
    case_key_folds_value_not_source()
    compiled = compiled_resolver()
    if FAILURES:
        for message in FAILURES:
            print("FAIL: " + message, file=sys.stderr)
        print(f"th fuel rods contract: FAIL ({len(FAILURES)})", file=sys.stderr)
        return 1
    tail = "source + deck key + case key" + (" + compiled resolver" if compiled else "")
    print(f"th fuel rods contract: PASS ({tail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
