#!/usr/bin/env python3
"""Contract for `--result full|pin-off|light` and the per-job result mode.

WHAT THE FLAG IS.  An output-shape switch, not a fidelity one.  All three modes
run the same solve, the same PPR and the same feedback loops; the campaign
measured full, pin-off and light to one trajectory digest (814201df0583e1d2,
single and batch, GA evaluator plan Sec 2.1).  What differs is what leaves the
process: ~301.6 MB, ~12.0 MB, ~25.1 kB per case.

WHY IT IS PER JOB.  A GA wave wants its population light and its promoted elites
full, in ONE --batch-mode process.  `RASBERY_BATCH_LIGHT_RESULT` is process-wide
and cannot say that, so the manifest gets an optional third field and the Driver
gets the mode as a constructor argument instead of reading the environment.

THE TWO THINGS THAT WOULD BE SILENT IF BROKEN, and are what this test holds:

  1. THE RESULT MODE MUST NOT REACH THE FIDELITY.  Until WP1 it did:
     `RASBERY_BATCH_LIGHT_RESULT=1` and `--result light` were both refused
     without RASBERY_ALLOW_SCREENING=1, because main.cpp folded the output mode
     into `screening`.  That is now fixed (BOTTLENECK plan Sec 6.2,
     src/RunContract.h) and the coupling must not come back: `screening` is a
     function of the FIDELITY, and `full_hdf5` is the one field the writer still
     decides.  The union `light_result = Enabled() || any_light` survives
     precisely because `full_hdf5` still has to be true of the whole run --
     including a job that asked for light through the manifest's third field.
     tools/test_result_fidelity_contract.py holds the other half.
  2. THE MANIFEST'S THIRD FIELD MUST SURVIVE THE DISPATCHER.  The multi-GPU
     launcher REWRITES chunk manifests.  A third field it silently dropped would
     turn every promoted elite back into a scalar receipt, and the only symptom
     would be a missing HDF5 nobody looks for until the campaign ends.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def body(text: str, signature: str) -> str:
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


failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# ---------------------------------------------------------------- the enum ---
light = read("include/chiffon/BatchLightResult.h")
check("enum class ResultMode { Full, PinOff, Light }" in light,
      "BatchLightResult.h does not declare ResultMode{Full,PinOff,Light}")
parse = body(light, "inline bool ParseResultMode")
for word in ('"full"', '"light"', '"pin-off"'):
    check(word in parse, f"ParseResultMode does not accept {word}")
check("return false" in parse, "ParseResultMode never rejects an unknown word")
write = body(light, "static void Write(")
check("Enabled()" not in write,
      "BatchLightResult::Write re-reads RASBERY_BATCH_LIGHT_RESULT; --result light "
      "would compute every scalar and write nothing")

default = body(light, "static ResultMode DefaultMode")
check("Enabled() ? ResultMode::Light : ResultMode::Full" in default,
      "DefaultMode does not mirror RASBERY_BATCH_LIGHT_RESULT")

# --------------------------------------------------------------- the driver --
driver = read("src/Driver.h")
check("ResultMode result_mode = BatchLightResult::DefaultMode()" in driver,
      "Driver's constructor does not take a per-case ResultMode")
check("ResultMode  _result_mode" in driver or "ResultMode _result_mode" in driver,
      "Driver does not hold a per-case _result_mode")
drive = body(driver, "int Drive()")
check("BatchLightResult::Enabled()" not in drive,
      "Driver::Drive still reads the process-wide environment instead of its own mode")
check("(_result_mode == ResultMode::Light)" in drive,
      "Driver::Drive does not derive light_result from its own mode")
check("(_result_mode == ResultMode::PinOff)" in drive,
      "Driver::Drive does not implement pin-off")
check(re.search(r"pin_off\s*\)\s*\{[^}]*print_opt\.pin_info\s*=\s*false", drive, re.S) is not None,
      "pin-off does not clear print_opt.pin_info on the schedule")

# ------------------------------------------------------------------- main ----
main = read("src/main.cpp")
check('option != "--result"' in main, "main.cpp does not accept --result as an option")
check('if (option == "--result")' in main or 'option == "--result"' in main,
      "main.cpp never dispatches the --result value")
check("full|pin-off|light" in main, "--help does not document the three result modes")
check("--result" in main and "manifest" in main, "--help does not mention the manifest override")

# 1. the result mode reaches full_hdf5 -- and NOTHING else
guard_region = main[main.find("Exact-only hard contract"):main.find("[RASBERY][PHYSICS_MODE]")]
check(len(guard_region) > 200, "could not locate the exact-only guard region in main.cpp")
check("any_light" in guard_region,
      "main.cpp reads only the environment for the light result; a job that asked for "
      "light through the manifest's third field would still be reported full_hdf5:true")
check(re.search(r"light_result\s*=\s*rasbery::BatchLightResult::Enabled\(\)\s*\|\|\s*any_light",
                guard_region) is not None,
      "light_result is not the union of the environment and the per-job modes")
# WP1: and it must NOT reach the screening verdict.  The comment block quotes
# the old line on purpose, so this scans the code.
guard_code = re.sub(r"//[^\n]*", "", guard_region)
check(re.search(r"screening\s*=[^;]*light_result", guard_code) is None,
      "the result mode is back in the screening verdict: `--result light` would once "
      "again declare the SOLVE approximate (BOTTLENECK plan Sec 6.2)")
check("light is a screening mode" not in main,
      "--help still calls light a screening mode; it is an output mode and a light run "
      "is acceptance-eligible")

# per-job modes reach both Driver construction sites
sites = re.findall(r"rasbery::Driver driver\((.*?)\);", main, re.S)
check(len(sites) == 2, f"expected two Driver construction sites, found {len(sites)}")
for i, site in enumerate(sites):
    check("rasbery_result_modes" in site,
          f"Driver construction site {i + 1} does not pass a per-job result mode")

# 2. the manifest's third field
manifest_reader = body(main, "bool rasberyReadJobManifest")
check("ParseResultMode" in manifest_reader,
      "the manifest reader does not parse a third field as a result mode")
check("found a fourth" in manifest_reader,
      "the manifest reader no longer rejects a fourth field")

# ------------------------------------------------------------- the harnesses -
single = read("tools/run_single_gpu_batch.py")
check('"--result"' in single, "run_single_gpu_batch.py has no --result")
check('command.extend(["--result", args.result])' in single,
      "run_single_gpu_batch.py does not forward --result to RASBERY")

multi = read("tools/run_multi_gpu_batch.py")
check('"--result"' in multi, "run_multi_gpu_batch.py has no --result")
check('["--result", result_mode]' in multi,
      "run_multi_gpu_batch.py does not forward --result to each chunk")

# The dispatcher rewrites chunk manifests -- prove the third field survives that
# rewrite, rather than trusting that it does.
spec = importlib.util.spec_from_file_location("rmgb", ROOT / "tools" / "run_multi_gpu_batch.py")
rmgb = importlib.util.module_from_spec(spec)
sys.modules["rmgb"] = rmgb  # @dataclass resolves annotations through sys.modules
spec.loader.exec_module(rmgb)

with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "jobs.txt"
    src.write_text(
        "# a wave: light population, one full elite\n"
        "a.json out/a.h5 light\n"
        'b.json "out/b.h5" full\n'
        "c.json out/c.h5\n",
        encoding="utf-8",
    )
    jobs = rmgb.read_manifest(src)
    check(jobs == [("a.json", "out/a.h5", "light"),
                   ("b.json", "out/b.h5", "full"),
                   ("c.json", "out/c.h5", "")],
          f"read_manifest lost the per-job result mode: {jobs}")

    rewritten = "".join(
        f'"{i}" "{o}"' + (f" {m}" if m else "") + "\n" for i, o, m in jobs
    )
    back = Path(tmp) / "chunk.txt"
    back.write_text(rewritten, encoding="utf-8")
    check(rmgb.read_manifest(back) == jobs,
          "a chunk manifest does not round-trip the per-job result mode")

    bad = Path(tmp) / "bad.txt"
    bad.write_text("a.json out/a.h5 sloppy\n", encoding="utf-8")
    try:
        rmgb.read_manifest(bad)
        failures.append("read_manifest accepted an unknown result mode")
    except ValueError:
        pass

if failures:
    for f in failures:
        print(f"result mode contract: FAIL {f}")
    raise SystemExit(1)
print("result mode contract: PASS")
