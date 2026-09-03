#!/usr/bin/env python3
"""Contract: the fuel-temperature TABLE is ONE selected identity, not an assumption.

THE FINDING THIS CLOSES.  include/Database/tf.csv -- the dT(LPD, burnup) grid
XSSet::GetTfuel (src/XSSet.h:516-526) interpolates at src/XSSet.cpp:6378-6381,
mirrored on the device by thGetTfuel (src/ThKernel.h) and in the ThReference
quotation -- is MASTER's BUILT-IN WH-type table, `isolth = 11`; its top-left
corner is the %DEF_TFT example in the MASTER-3.0 manual.  The APR1400 / KNGR
decks are `isolth = 12`, the ABB-CE table, whose burnup slope of the rise is
about five times steeper: over one cycle MASTER's rise falls 29.6 % where this
grid falls 5.8 %, so KNGR tfavg lands about -14.6 K at BOC and +71 K at EOC --
absorbed by the boron search as the -15.3 ppm Gate B residual.

WHAT IS ASSERTED, AND WHY EACH ONE IS HERE.

  1. ONE LOADER.  The CPU path, the GPU upload and the reference all read the
     table that th::loadTfTable returned; a second ParseFromCSV of tf.csv
     anywhere is a second answer to "what was interpolated".
  2. THE DEFAULT IS LEGACY, and legacy is `wh` -- so the B0 baseline
     (1f36e75dc00ed2b4 / 4377 outers) is untouched, and a deck that NAMES `wh`
     keys identically to one that says nothing.
  3. THE DECK KEY, in both accepted places, refused when declared in both.
  4. THE ENVIRONMENT, resolved by the real C++ function when a compiler is
     available: legacy | deck | wh | ce | <path>, with a REFUSAL rather than a
     guess when `deck` is asked for and the deck says nothing.
  5. THE CE REFUSAL IS STILL WIRED.  include/Database/tf_ce.csv now EXISTS --
     regressed 2026-09-04 from the MASTER KNGR run; its content is
     tools/test_tf_ce_table_contract.py's business.  What is asserted here is
     that the refusal PATH survives: a MISSING CE table must still throw and
     still name tools/fit_tf_table.py, because the failure it prevents (falling
     back to the WH grid) is silent.
  6. THE CASE KEY folds the IDENTITY (name + content digest) and not the source.
  7. NEGATIVE CONTROLS: the fixture is table-independent, and a moved table
     moves the key.

USAGE
    tools/test_th_tf_table_contract.py
"""
from __future__ import annotations

import hashlib
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

# The WH grid's bytes.  `legacy` IS this file, and the whole published campaign
# was produced with it, so it is pinned here rather than merely described.
WH_TABLE_SHA256 = "cb72254367cc84c299991f1becb229cdf61acf3106a9b3a8dc19e0a15552cdf3"

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def code_lines(text: str) -> str:
    """The file with its `//` comment lines dropped -- prose may say tf.csv."""
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("//"))


# ---------------------------------------------------------------------------
# 1. One loader.
# ---------------------------------------------------------------------------
def one_loader() -> None:
    header = read("src/ThTfTable.h")
    for token in ("kTfTableEnv", "resolveTfTable", "loadTfTable", "TfTableData",
                  "tfInlinePayload"):
        if token not in header:
            fail(f"src/ThTfTable.h lost {token!r}; it is the one place the table is named")

    # The only `tf.csv` in the T/H path's CODE is the loader's built-in name.
    for rel in ("src/XSSet.cpp", "src/ThKernel.h", "src/ThReference.cpp",
                "src/CudaThBackend.cu", "src/Geometry.cpp"):
        if '"tf.csv"' in code_lines(read(rel)):
            fail(f"{rel} still names tf.csv in code; the table comes from "
                 "th::loadTfTable, and a second literal is a second table")
    if code_lines(read("src/ThTfTable.h")).count('"tf.csv"') != 1:
        fail("src/ThTfTable.h does not hold exactly one \"tf.csv\"; the built-in "
             "name lives there and nowhere else")

    xsset = read("src/XSSet.cpp")
    if "th::loadTfTable(_g.tf_table_choice())" not in xsset:
        fail("XSSet::LoadTHTables does not take the fuel-temperature table from "
             "Geometry's resolved choice")
    if "loadTfTable" not in read("src/Driver.h"):
        fail("Driver.h does not read the loaded table for the case key; the key "
             "would then describe a table nobody proved was the one interpolated")

    # The GPU and the reference are downstream of _tf_table BY CONSTRUCTION:
    # CudaThBackend's TableView is filled from it, so pin that it still is.
    if "_tf_table.x_axis.data()" not in xsset or "tables.tf_x" not in xsset:
        fail("the CUDA TableView is no longer filled from XSSet::_tf_table; the "
             "device could then interpolate a different grid from the host")


# ---------------------------------------------------------------------------
# 2. The default is legacy, and legacy is `wh`.
# ---------------------------------------------------------------------------
def default_is_legacy() -> None:
    header = read("src/ThTfTable.h")
    body = header[header.index("inline TfChoice resolveTfTable("):]
    body = body[:body.index("\n}")]
    if 'want.empty() || want == "legacy"' not in body:
        fail("resolveTfTable no longer defaults to legacy on an unset environment")
    if 'c.name   = "wh";' not in body:
        fail("the legacy default does not resolve to the `wh` IDENTITY; legacy and "
             "an explicit wh would then key differently for one arithmetic")

    # include/Database/tf_ce.csv NOW EXISTS -- regressed 2026-09-04 from the
    # MASTER KNGR run, see docs/TH_TF_TABLE_SELECTION_20260904_KO.md and
    # tools/test_tf_ce_table_contract.py, which owns everything about its
    # content.  What still belongs HERE is only that its arrival did not move
    # the default: legacy must still be the WH grid, byte for byte.
    wh = (ROOT / "include" / "Database" / "tf.csv").read_bytes()
    if hashlib.sha256(wh).hexdigest() != WH_TABLE_SHA256:
        fail("include/Database/tf.csv changed.  The B0 baseline (trajectory digest "
             "1f36e75dc00ed2b4 at 4377 outers) was produced with THESE bytes, and "
             "`legacy` resolves to them; regressing the CE table must not have "
             "touched the WH one.")


# ---------------------------------------------------------------------------
# 3. The deck key.
# ---------------------------------------------------------------------------
def deck_key() -> None:
    io_cpp = read("src/IO.cpp")
    if '"tf table"' not in io_cpp:
        fail('src/IO.cpp does not parse a "tf table" deck key')
    for alias in ("tf_table", "tfuel table", "tfuel_table"):
        if f'"{alias}"' not in io_cpp:
            fail(f"src/IO.cpp does not accept the {alias!r} alias")
    if "declared BOTH under" not in io_cpp:
        fail("src/IO.cpp does not refuse a deck that declares the table in both "
             "geometry.dimensions and \"default parameters\"; the loser would be "
             "invisible")
    if "geometry_input.tf_table" not in io_cpp:
        fail("the parsed spec never reaches GeometryInput")
    if "tf_table_name" not in io_cpp:
        fail("IO::SaveRestart does not carry the table choice; a restarted run "
             "would silently fall back to the WH grid")

    # The python mirror must read the same two places and the same aliases.
    for where in ("geometry.dimensions", "default parameters"):
        if where not in read("tools/case_key.py"):
            fail(f"tools/case_key.py does not look for the deck key under {where}")


# ---------------------------------------------------------------------------
# 4/5. The environment and the CE refusal, through the real C++ when possible.
# ---------------------------------------------------------------------------
PROBE = r"""
#include "ThTfTable.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace rasbery::th;

int main(int argc, char** argv) {
    TfTableSpec deck;
    if (argc > 1 && std::strcmp(argv[1], "none") != 0) {
        if (std::strcmp(argv[1], "inline") == 0) {
            deck.name = "inline";
            deck.lpd  = {50.0, 100.0};
            deck.bu   = {0.0, 60.0};
            deck.dt   = {80.0, 160.0, 55.0, 110.0};
        } else {
            deck = TfTableSpec{};
            const TfChoice c = tfChoiceOfString(argv[1], "deck");
            deck.name = c.name;
            deck.path = c.path;
        }
    }
    try {
        const TfChoice c = resolveTfTable(deck);
        std::printf("%s %s %s\n", c.name.c_str(), c.source.c_str(),
                    c.path.empty() ? "-" : c.path.c_str());
    } catch (const std::exception&) {
        std::printf("REFUSED\n");
    }
    return 0;
}
"""


def compiled_resolver() -> bool:
    toolchain, reason = _cxx_toolchain.discover(ROOT)
    if toolchain is None:
        print(f"th tf table compiled contract: SKIP -- {reason}", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        cpp = tmp / "th_tf_table_harness.cpp"
        cpp.write_text(PROBE, encoding="utf-8")
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
                    [toolchain.compiler, toolchain.std_flag, "-O0", str(cpp),
                     "-o", str(exe)]
                    + [arg for d in includes for arg in ("-I", str(d))],
                    check=True, capture_output=True, universal_newlines=True)
        except subprocess.CalledProcessError as failure:
            fail("the ThTfTable resolver harness does not compile:\n"
                 + (failure.stdout or "") + (failure.stderr or ""))
            return True

        base = {k: v for k, v in os.environ.items() if k != "RASBERY_TH_TF_TABLE"}

        def run(value, deck="none"):
            environ = dict(base)
            if value is not None:
                environ["RASBERY_TH_TF_TABLE"] = value
            done = subprocess.run([str(exe), deck], capture_output=True,
                                  universal_newlines=True, env=environ)
            return done.stdout.strip()

        cases = (
            (None, "none", "wh legacy -", "unset must be the shipped WH table"),
            ("", "none", "wh legacy -", "empty must be the shipped WH table"),
            ("legacy", "ce", "wh legacy -", "legacy ignores the deck, by design"),
            (" legacy ", "none", "wh legacy -", "the value is trimmed"),
            ("wh", "none", "wh env -", "an explicit wh is the same identity"),
            ("ce", "none", "ce env -", "ce names the ABB-CE table"),
            ("deck", "ce", "ce deck -", "deck takes the deck's table"),
            ("deck", "none", "REFUSED", "deck with no deck table must REFUSE"),
            ("deck", "inline", "inline deck -", "an inline deck table survives"),
            ("/tmp/x.csv", "none", "file env /tmp/x.csv", "a bare path is a file"),
            ("file:/tmp/x.csv", "none", "file env /tmp/x.csv", "file: is stripped"),
            (None, "none", "wh legacy -", "and back to the default"),
        )
        for value, deck, want, why in cases:
            got = run(value, deck)
            if got != want:
                fail(f"RASBERY_TH_TF_TABLE={value!r} (deck={deck}) -> {got!r}, "
                     f"want {want!r}: {why}")
    return True


def ce_refuses_loudly() -> None:
    cpp = read("src/ThTfTable.cpp")
    if "fit_tf_table.py" not in cpp:
        fail("the CE refusal does not name tools/fit_tf_table.py; the operator is "
             "left with an error and no next step")
    if "NOT YET REGRESSED" not in cpp:
        fail("the CE refusal does not say the table is unregressed")
    marker = cpp[cpp.index('if (choice.name == "ce")'):]
    if "throw" not in marker[:600]:
        fail("a missing CE table does not THROW; a silent fallback to the WH grid "
             "is exactly the defect this closes")

    # The python mirror refuses on a MISSING table too.  tf_ce.csv now exists,
    # so the refusal is exercised against a database directory that does not
    # hold it -- the path that matters is "asked for a table that is not there",
    # not "asked for `ce`".
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RASBERY_")}
    with tempfile.TemporaryDirectory() as raw:
        deck = Path(raw) / "d.json"
        deck.write_text(json.dumps(_deck_json()), encoding="utf-8")
        missing = Path(raw) / "no_such_table.csv"
        try:
            case_key.case_key(deck,
                              env=dict(clean, RASBERY_TH_TF_TABLE=str(missing)),
                              xslib=False)
        except SystemExit:
            pass
        else:
            fail("tools/case_key.py keyed a fuel-temperature table that does not "
                 "exist; the mirror must refuse where the solver refuses")

    # And with the file present, `ce` must now RESOLVE -- a refusal that outlived
    # the regression would send the operator back to a tool that already ran.
    with tempfile.TemporaryDirectory() as raw:
        deck = Path(raw) / "d.json"
        deck.write_text(json.dumps(_deck_json()), encoding="utf-8")
        try:
            case_key.case_key(deck, env=dict(clean, RASBERY_TH_TF_TABLE="ce"),
                              xslib=False)
        except SystemExit as exc:
            fail(f"tools/case_key.py still refuses `ce` although "
                 f"include/Database/tf_ce.csv exists: {exc}")


# ---------------------------------------------------------------------------
# 6/7. The case key, and the negative controls.
# ---------------------------------------------------------------------------
def _deck_json(tf=None, where="geometry.dimensions"):
    dims = {"ng": 2, "xydivision": 2, "npins": 16, "nfrod": 236}
    config = {
        "geometry": {
            "dimensions": dims,
            "size": {"hx": 20.0, "hy": 20.0, "hz": [10.0, 10.0]},
            "symmetry": {"angle": 90, "mirror": True, "center assembly divided": False},
            "albedo": {"west": 0.0, "east": 0.5, "north": 0.0, "south": 0.5,
                       "bottom": 0.5, "top": 0.5},
            "core": [["A1", "A1"], ["A1", "A1"]],
        },
        "batch": {"A1": ["F1", "F1"]},
    }
    if tf is not None:
        if where == "geometry.dimensions":
            dims["tf table"] = tf
        else:
            config["default parameters"] = {"tf table": tf}
    return config


def case_key_folds_identity_not_source() -> None:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RASBERY_")}
    tf_csv = ROOT / "include" / "Database" / "tf.csv"
    want = "wh:" + hashlib.sha256(tf_csv.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        plain = tmp / "plain.json"
        plain.write_text(json.dumps(_deck_json()), encoding="utf-8")
        says_wh = tmp / "wh.json"
        says_wh.write_text(json.dumps(_deck_json("wh")), encoding="utf-8")
        in_defaults = tmp / "defaults.json"
        in_defaults.write_text(
            json.dumps(_deck_json("wh", where="default parameters")), encoding="utf-8")
        both = tmp / "both.json"
        cfg = _deck_json("wh")
        cfg["default parameters"] = {"tf table": "wh"}
        both.write_text(json.dumps(cfg), encoding="utf-8")
        moved = tmp / "moved.csv"
        moved.write_bytes(tf_csv.read_bytes())
        bumped = tmp / "bumped.csv"
        bumped.write_bytes(tf_csv.read_bytes().replace(b"79.78", b"80.78"))
        uses_file = tmp / "file.json"
        uses_file.write_text(json.dumps(_deck_json(f"file:{moved}")), encoding="utf-8")

        def components(path, env=None):
            return case_key.case_key(path, env=dict(clean, **(env or {})), xslib=False)

        default = components(plain)
        if default["th_tf_table"] != want:
            fail(f"the default table identity is {default['th_tf_table']!r}, not the "
                 f"digest of the shipped tf.csv")
        if default["th_tf_table_source"] != "legacy":
            fail("the default source is not 'legacy'")
        if "\nth_tf_table\t" not in default["payload"]:
            fail("the case-key payload has no th_tf_table line; two runs on two "
                 "fuel-temperature tables would share a cache entry")

        # A deck that NAMES wh, taken from the deck, is the same arithmetic.
        for path in (says_wh, in_defaults):
            got = components(path, {"RASBERY_TH_TF_TABLE": "deck"})
            if got["th_tf_table"] != want:
                fail(f"{path.name}: naming wh in the deck did not resolve to the "
                     "shipped table")
            if got["th_tf_table_source"] != "deck":
                fail(f"{path.name}: the source was not reported as 'deck'")

        # ... and keys identically to the legacy default, because only the SOURCE
        # differs.  (The deck DIGEST differs, so compare the folded field.)
        if components(says_wh)["th_tf_table"] != components(says_wh, {
                "RASBERY_TH_TF_TABLE": "deck"})["th_tf_table"]:
            fail("a deck that declares wh folds a different identity from the "
                 "legacy default; the SOURCE is not the arithmetic")

        # DECLARED IN BOTH PLACES IS REFUSED.
        try:
            components(both, {"RASBERY_TH_TF_TABLE": "deck"})
        except SystemExit:
            pass
        else:
            fail("a deck declaring \"tf table\" in both places was accepted")

        # A path to the SAME BYTES is the same case; different bytes are not.
        by_path = components(uses_file, {"RASBERY_TH_TF_TABLE": "deck"})
        if not by_path["th_tf_table"].startswith("file:"):
            fail("a file: table did not key under the 'file' name")
        if by_path["th_tf_table"].split(":", 1)[1] != want.split(":", 1)[1]:
            fail("the same table bytes read from another path folded a different "
                 "digest; the key digests CONTENT")
        by_bumped = components(plain, {"RASBERY_TH_TF_TABLE": str(bumped)})
        if by_bumped["th_tf_table"] == by_path["th_tf_table"]:
            fail("a one-cell change to the table did not move the case key; two "
                 "fuel temperatures would share one cache entry")

        # The inline form: a different table is a different key, and the payload
        # is the one the C++ digests.
        inline = tmp / "inline.json"
        inline.write_text(json.dumps(_deck_json({"lpd": [50, 100], "bu": [0, 60],
                                                 "dt": [[80, 160], [55, 110]]})),
                          encoding="utf-8")
        got = components(inline, {"RASBERY_TH_TF_TABLE": "deck"})
        if not got["th_tf_table"].startswith("inline:"):
            fail("an inline deck table did not key under the 'inline' name")
        expect = hashlib.sha256(case_key._tf_inline_payload(
            [50, 100], [0, 60], [80, 160, 55, 110]).encode()).hexdigest()
        if got["th_tf_table"] != "inline:" + expect:
            fail("the inline digest is not the canonical payload's digest")


def fixture_is_table_independent() -> None:
    """NEGATIVE CONTROL.  The mined T/H form mask must not move with a deck."""
    reference = read("src/ThReference.cpp")
    if "f.tf      = buildTable(" not in reference:
        fail("the ThReference fixture no longer builds its own synthetic tf table; "
             "if it started reading the shipped CSV, the mined form mask would move "
             "the day somebody selects another table")
    if "loadTfTable" in read("src/ThFormMine.h"):
        fail("the T/H mining fixture reaches for the selected table; the mask is a "
             "property of the ARITHMETIC and must be table-independent")


def inline_payload_mirrors() -> None:
    """The C++ canonical form and the python one must be the same bytes."""
    cpp = read("src/ThTfTable.cpp")
    if "rasbery-tf-inline/v1" not in cpp:
        fail("src/ThTfTable.cpp lost the inline payload version line")
    if "rasbery-tf-inline/v1" not in read("tools/case_key.py"):
        fail("tools/case_key.py does not mirror the inline payload version line")
    got = case_key._tf_inline_payload([1.0, 2.5], [0.0], [3.0, 4.0])
    if got != "rasbery-tf-inline/v1\nlpd\t1\t2.5\nbu\t0\ndt\t3\t4\n":
        fail(f"the python inline payload changed shape: {got!r}")


def receipt_exists() -> None:
    cpp = read("src/ThTfTable.cpp")
    fields = ("source", "name", "path", "sha256", "nlpd", "nbu")
    site = cpp[cpp.index("[RASBERY][TH][TFTABLE]"):]
    site = site[:site.index(";")]
    for field in fields:
        if f'\\"{field}\\"' not in site:
            fail(f"the [RASBERY][TH][TFTABLE] receipt does not publish {field!r}")
    driver = read("src/Driver.h")
    case_site = driver[driver.index('"  [RASBERY][CASE] {{'):]
    if 'schema_version' not in case_site[:200] or ':10,' not in case_site[:200]:
        fail("the [RASBERY][CASE] receipt did not reach schema_version 10 for the "
             "table fields")


def main() -> int:
    one_loader()
    default_is_legacy()
    deck_key()
    ce_refuses_loudly()
    case_key_folds_identity_not_source()
    fixture_is_table_independent()
    inline_payload_mirrors()
    receipt_exists()
    compiled = compiled_resolver()
    if FAILURES:
        for message in FAILURES:
            print("FAIL: " + message, file=sys.stderr)
        print(f"th tf table contract: FAIL ({len(FAILURES)})", file=sys.stderr)
        return 1
    tail = "loader + deck key + CE refusal + case key + controls"
    tail += " + compiled resolver" if compiled else ""
    print(f"th tf table contract: PASS ({tail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
