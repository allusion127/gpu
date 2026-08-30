#!/usr/bin/env python3
"""WP10.4 -- the evaluator's memory receipt, and the caches it reports on.

WHY THIS FILE EXISTS.  The host 181 soak at `91004f7` (5 generations x width 16,
persistent evaluator, PROD env + RASBERY_GPU_FULL=1) failed its stability gate
on RSS: +17.41 MB per generation over the second half against an 8.0 MB budget,
and +37.64 MB/generation on the run before it.  Every mechanism receipt the soak
asserts on came back ZERO -- no slot duplicates, no stale tenants, no double
releases, no pinned ranges left between waves -- so the process was growing and
NOTHING IN ANY RECEIPT COULD NAME WHAT WAS GROWING.  For an evaluator that has
to survive 10k generations, "we will profile it next time" is not a plan: the
host cannot be attached to and the growth is 170 GB by the end.

Two things had to be true, and this file pins both.

  1. THE PROCESS REPORTS ITS OWN SIZE, BESIDE THE CONTAINERS THAT COULD EXPLAIN
     IT.  `[RASBERY][EVALUATOR][MEM]`, once per generation, carries RSS and the
     entry count of every process-lifetime cache -- so the next soak reads a
     generation where RSS moved and a cache moved with it, instead of a number
     with no suspects.

  2. THE UNBOUNDED CACHES ARE BOUNDED.  Three tables were keyed on something a
     case can vary and trimmed by nothing:
       * `XsLibraryCacheEntries` (src/XSSet.cpp) -- ~34 MB per entry, keyed on
         the library's CONTENT, so a library rewritten under a running
         evaluator adds an entry and keeps the old parse forever;
       * `cohort::detail::entries` (src/CohortContext.h) -- keyed on the
         geometry payload digest, and a GA candidate IS a different geometry
         payload, so the table grows with the campaign;
       * `BatchLightResult`'s two digest memos -- keyed by PATH ALONE, and every
         GA candidate is its own deck file.
     Each is now capped, evicts least-recently-used (or, for the memos, drops
     whole), and COUNTS what it evicted.

The compiled half drives the cohort registry for real at a cap of 2 and asserts
the bound holds, the victim is the least recently used one, and the survivor is
still a hit rather than a silent rebuild.  Without it this file would only be
checking that some words are present in some files.

Run: python3 tools/test_evaluator_mem_receipt_contract.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


SERVER = read("src/EvaluatorServer.h")
XSSET = read("src/XSSet.cpp")
COHORT = read("src/CohortContext.h")
LIGHT = read("include/chiffon/BatchLightResult.h")
PINREG = read("src/HostPinRegistry.h")


# ---------------------------------------------------------------------------
# 1. THE RECEIPT
# ---------------------------------------------------------------------------
#: Every field the next soak has to be able to read off one line.  A field
#: dropped from the receipt is a container that goes back to being anonymous.
MEM_FIELDS = ("wave_id", "rss_mb", "rss_delta_mb", "rss_peak_mb", "live_cases",
              "cache_entries", "evictions", "cuda_host_bytes")

#: The containers the receipt has to size.  Each of these is process-lifetime
#: state that a case can add to; one missing is one suspect the next soak cannot
#: rule in or out.
MEM_CACHES = ("xslib", "xslib_digest", "cohorts", "quadratures", "pin_records",
              "digest_memo", "case_samples")


def check_receipt(server: str) -> list[str]:
    bad: list[str] = []
    if server.count('"[RASBERY][EVALUATOR][MEM] {') != 1:
        bad.append("src/EvaluatorServer.h must emit [RASBERY][EVALUATOR][MEM] from "
                   "exactly one site; two sites are two answers to how big the "
                   "process is")
        return bad
    start = server.index('"[RASBERY][EVALUATOR][MEM] {')
    block = server[start:server.find("std::endl;", start)]
    for field in MEM_FIELDS:
        if f'\\"{field}\\"' not in block:
            bad.append(f"the [RASBERY][EVALUATOR][MEM] receipt has no {field!r} field")
    for cache in MEM_CACHES:
        if f'\\"{cache}\\"' not in block:
            bad.append(f"the memory receipt does not size the {cache!r} container; a "
                       "cache the receipt cannot see is a cache the next soak cannot "
                       "accuse or clear")
    if "reportMemory(" not in server:
        bad.append("the memory receipt is not a named function called from the wave")
    # ONCE PER GENERATION, from runWave, after the wave's own receipt.  A
    # mid-wave sample measures where the wave happened to be; the question is
    # what is left behind when it is over.
    if server.count("reportMemory(wave.wave_id);") != 1:
        bad.append("reportMemory is not called exactly once per wave from runWave")
    if "/proc/self/statm" not in server:
        bad.append("the receipt does not read RSS from /proc/self/statm, which is the "
                   "same quantity tools/soak_run.py samples from outside; a second "
                   "source is a second number to disagree")
    return bad


def check_live_cases(server: str) -> list[str]:
    """`live_cases` must count CONSTRUCTED Drivers and fall on destruction."""
    bad: list[str] = []
    if "liveCases()" not in server:
        bad.append("there is no live-case counter; a Driver that outlived its case "
                   "leaks everything a case owns and nothing reports it")
        return bad
    if "fetch_add(1, std::memory_order_relaxed)" not in server:
        bad.append("the live-case counter is never incremented")
    if "fetch_sub(1, std::memory_order_relaxed)" not in server:
        bad.append("the live-case counter is never decremented")
    # The decrement has to be in a destructor, or a throwing case never
    # decrements and every later generation reports a leak that is not there.
    if "~LiveGuard()" not in server:
        bad.append("the live-case decrement is not in a destructor; a case that "
                   "throws -- which RASBERY_GPU_FULL=1 fail-closed does by design -- "
                   "would leave the counter high forever")
    start = server.find("static void runOneCase(")
    body = server[start:server.find("void reportCase(", start)]
    if "LiveGuard" not in body:
        bad.append("the live-case guard is not in runOneCase, so it does not bracket "
                   "the Driver at all")
    if not re.search(r"\{\s*Driver driver\(deck, output, mode\);", body):
        bad.append("the guard has broken the Driver's own scope; the slot release in "
                   "its destructor must still land inside the measured teardown")
    return bad


# ---------------------------------------------------------------------------
# 2. THE BOUNDS
# ---------------------------------------------------------------------------
def check_xslib_bound(xsset: str) -> list[str]:
    bad: list[str] = []
    if "XsLibraryCacheLimit()" not in xsset:
        bad.append("src/XSSet.cpp does not bound XsLibraryCacheEntries; one entry is a "
                   "whole ~34 MB parse and the key discriminates on CONTENT, so a "
                   "library rewritten under a running evaluator keeps both forever")
        return bad
    if "RASBERY_XSLIB_CACHE_ENTRIES" not in xsset:
        bad.append("the xslib cache bound has no environment override; a bound nobody "
                   "can raise is a bound somebody will delete")
    if "g_xslib_evictions" not in xsset:
        bad.append("xslib evictions are not counted; a cache that silently re-parses "
                   "34 MB every case looks exactly like one that is working")
    if "last_use" not in xsset:
        bad.append("the xslib cache evicts without a use stamp, so the victim is "
                   "whichever came first -- which in a long-lived evaluator is the "
                   "library every case is still asking for")
    # THE ONE THING THAT MUST NOT BE EVICTED.  An entry with value == nullptr is
    # a key another worker is mid-parse on, and workers wait on it BY KEY; erase
    # it and they wait for a publication that will never come.
    start = xsset.find("while (resident > XsLibraryCacheLimit())")
    if start < 0:
        bad.append("the xslib eviction loop is not driven by the resident count")
    else:
        loop = xsset[start:start + 900]
        if "it->value == nullptr" not in loop or "it->building" not in loop:
            bad.append("the xslib eviction can take a placeholder entry; workers wait "
                       "on that key and would wait forever for a parse nobody "
                       "publishes")
    return bad


def check_cohort_bound(cohort: str) -> list[str]:
    bad: list[str] = []
    if "RASBERY_COHORT_CACHE_ENTRIES" not in cohort:
        bad.append("the cohort registry is unbounded; the key covers the geometry "
                   "payload and a GA candidate IS a different payload, so it grows "
                   "with the campaign and `cohorts` stops proving anything")
    if "counter(4)" not in cohort:
        bad.append("cohort evictions are not counted")
    if "last_use" not in cohort:
        bad.append("the cohort registry evicts without a use stamp")
    if "evictions" not in cohort or '\\"evictions\\"' not in cohort:
        bad.append("the [RASBERY][COHORT] receipt does not publish evictions")
    return bad


def check_memo_bound(light: str) -> list[str]:
    bad: list[str] = []
    if "TrimLocked" not in light:
        bad.append("BatchLightResult's digest memos are unbounded; both are keyed by "
                   "PATH ALONE and never expire, and every GA candidate is its own "
                   "deck file")
    if "HashCacheEntries" not in light:
        bad.append("nothing can read the digest memo size, so the memory receipt "
                   "cannot report it")
    if "HashCacheClears" not in light:
        bad.append("a memo that drops itself does not count the drop, so re-reads are "
                   "paid for silently")
    return bad


def check_pin_bytes(pinreg: str) -> list[str]:
    bad: list[str] = []
    if "rasberyHostPinLiveBytes" not in pinreg:
        bad.append("HostPinRegistry reports live RANGES but not live BYTES; 40 records "
                   "of a megabyte and 40 of a kilobyte are the same count and are not "
                   "the same process")
    if "rasberyHostPinLiveBytes()" not in SERVER:
        bad.append("the memory receipt does not report the page-locked byte count")
    return bad


# ---------------------------------------------------------------------------
# 3. THE READER -- soak_run must turn the receipt into an attribution
# ---------------------------------------------------------------------------
def check_soak_attribution() -> list[str]:
    """A receipt nobody reads is a receipt that will silently stop being printed."""
    sys.path.insert(0, str(ROOT / "tools"))
    import soak_run  # noqa: E402  (imported here so a failure is a FAIL, not a crash)

    bad: list[str] = []

    class Gen:
        def __init__(self, mem):
            self.mem = mem

    flat = {"cache_entries": {"xslib": 1, "cohorts": 1}, "evictions": {"xslib": 0},
            "cache_bytes": {"xslib": 100}, "live_cases": 0, "cuda_host_bytes": 4096}
    grew = {"cache_entries": {"xslib": 1, "cohorts": 900}, "evictions": {"xslib": 0},
            "cache_bytes": {"xslib": 100}, "live_cases": 0, "cuda_host_bytes": 4096}

    named = " ".join(soak_run.attribute_rss_growth([Gen(flat), Gen(grew)]))
    if "cohorts" not in named:
        bad.append("soak_run.attribute_rss_growth did not name the container that grew; "
                   "the finding would read exactly as unhelpfully as the 181 one did")

    clean = " ".join(soak_run.attribute_rss_growth([Gen(flat), Gen(dict(flat))]))
    if "FLAT" not in clean:
        bad.append("a run where every container was flat is not reported as such; "
                   "'not in these caches' is an answer and has to be printed as one")

    leaked = dict(flat)
    leaked["live_cases"] = 2
    live = " ".join(soak_run.attribute_rss_growth([Gen(flat), Gen(leaked)]))
    if "live_cases" not in live:
        bad.append("a Driver that outlived its case is not called out")

    old_binary = " ".join(soak_run.attribute_rss_growth([Gen(None), Gen(None)]))
    if "no [RASBERY][EVALUATOR][MEM]" not in old_binary:
        bad.append("a binary that predates the receipt is not reported as unattributable; "
                   "silence there reads as 'nothing grew'")

    if "attribute_rss_growth(generations)" not in (ROOT / "tools" / "soak_run.py").read_text(
            encoding="utf-8-sig"):
        bad.append("soak_run never calls attribute_rss_growth on an RSS finding")
    return bad


# ---------------------------------------------------------------------------
# 4. NEGATIVE CONTROLS -- every check above must be able to fail
# ---------------------------------------------------------------------------
def controls() -> list[str]:
    broken: list[str] = []

    if not check_receipt(SERVER.replace('\\"rss_delta_mb\\"', '\\"nothing\\"')):
        broken.append("the receipt check passes a receipt with no rss_delta_mb")
    # `quadratures` appears only inside the MEM block, so this control cannot be
    # satisfied by the identically-named field in the process receipt.
    if not check_receipt(SERVER.replace('\\"quadratures\\"', '\\"nothing\\"')):
        broken.append("the receipt check passes a receipt that does not size one of "
                      "the containers it exists to name")
    if not check_live_cases(SERVER.replace("~LiveGuard()", "~NotAGuard()")):
        broken.append("the live-case check passes a decrement outside a destructor")
    if not check_xslib_bound(XSSET.replace("XsLibraryCacheLimit()", "9999")):
        broken.append("the xslib bound check passes a cache with no limit")
    if not check_xslib_bound(XSSET.replace("it->value == nullptr", "false")):
        broken.append("the xslib bound check passes an eviction that can take a "
                      "placeholder")
    if not check_cohort_bound(COHORT.replace("RASBERY_COHORT_CACHE_ENTRIES", "NOPE")):
        broken.append("the cohort bound check passes an unbounded registry")
    if not check_memo_bound(LIGHT.replace("TrimLocked", "NotATrim")):
        broken.append("the memo bound check passes an unbounded memo")
    if not check_pin_bytes(PINREG.replace("rasberyHostPinLiveBytes", "nope")):
        broken.append("the pinned-bytes check passes a registry with no byte accessor")
    return broken


# ---------------------------------------------------------------------------
# 5. THE COMPILED HALF -- the bound, driven for real
# ---------------------------------------------------------------------------
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

HARNESS_CPP = r"""
#include "CohortKey.h"
#include "CohortContext.h"
#include <cstdlib>
#include <iostream>
#include <string>
namespace rasbery { PinQuadTable buildPinQuadratureTable(int, int) { return {}; } }

static rasbery::cohort::Descriptor make(int tag) {
    return rasbery::cohort::Descriptor{
        std::string("geometry-") + std::to_string(tag), "LIB", 2, 2, 17};
}

int main() {
    using namespace rasbery;
    // Set BEFORE the first acquire: the limit is a function-local static and
    // resolves on first use, exactly as it does in the solver.
#ifdef _WIN32
    _putenv_s("RASBERY_COHORT_CACHE_ENTRIES", "2");
#else
    setenv("RASBERY_COHORT_CACHE_ENTRIES", "2", 1);
#endif
    for (int i = 0; i < 5; ++i) (void)cohort::acquire(make(i));
    cohort::Stats s = cohort::snapshot();
    std::cout << "bounded " << (s.cohorts <= 2) << "\n";
    std::cout << "limit_reported " << (s.limit == 2) << "\n";
    std::cout << "evicted " << (s.evictions == 3) << "\n";
    std::cout << "built_five " << (s.builds == 5) << "\n";

    // The most recently used cohort must have SURVIVED: an LRU that evicts the
    // entry every case is still asking for is a cache that costs and gives
    // nothing.
    (void)cohort::acquire(make(4));
    s = cohort::snapshot();
    std::cout << "mru_survived " << (s.builds == 5 && s.hits == 1) << "\n";

    // ...and the oldest must be GONE, rebuilt rather than silently served.
    (void)cohort::acquire(make(0));
    s = cohort::snapshot();
    std::cout << "lru_evicted " << (s.builds == 6) << "\n";

    // A rebuild is bit-identical: the Context is a pure function of the key's
    // inputs, which is the whole argument for sharing it -- and therefore also
    // the argument that dropping one costs a rebuild and nothing else.
    auto a = cohort::acquire(make(3));
    auto b = cohort::acquire(make(3));
    std::cout << "rebuild_is_shared " << (a.get() == b.get()) << "\n";
    return 0;
}
"""

EXPECTED = {
    "bounded": "the cohort registry did not respect RASBERY_COHORT_CACHE_ENTRIES=2",
    "limit_reported": "cohort::snapshot() does not report the limit it is enforcing",
    "evicted": "five cohorts at a cap of two must have evicted exactly three",
    "built_five": "five distinct descriptors must have built five Contexts",
    "mru_survived": "the most recently used cohort was evicted; an LRU that drops the "
                    "entry every case is still asking for costs a rebuild per case and "
                    "saves nothing",
    "lru_evicted": "the least recently used cohort was still resident, so the bound is "
                   "not actually trimming",
    "rebuild_is_shared": "a rebuilt cohort is not shared by the next acquisition, so an "
                         "eviction costs more than one rebuild",
}


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
    compiler = find_compiler()
    if compiler is None:
        return False
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        stub = tmp / "stub"
        stub.mkdir()
        (stub / "Geometry.h").write_text(STUB_GEOMETRY_H, encoding="utf-8")
        # Copied beside the stub for the same reason test_cohort_context_contract
        # copies them: a quoted include resolves relative to the including file
        # first, so a header left in src/ would reach the real Geometry.h and
        # drag pch.h -> HighFive -> the HDF5 C headers in behind it.
        for name in ("CohortKey.h", "CohortContext.h", "PprQuadrature.h"):
            (stub / name).write_text(read("src/" + name), encoding="utf-8")
        cpp = tmp / "mem_bound_harness.cpp"
        cpp.write_text(HARNESS_CPP, encoding="utf-8")
        exe = tmp / ("mem_bound_harness.exe" if os.name == "nt" else "mem_bound_harness")
        includes = [stub, ROOT / "src", ROOT / "include", ROOT / "include" / "chiffon"]
        try:
            if compiler.lower().endswith("vcvars64.bat"):
                # QUOTED include paths: this repository's own path contains an
                # `&`, which cmd would otherwise read as a command separator.
                script = tmp / "build_mem_bound_harness.bat"
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
            failures.append("the bounded-cache harness does not compile:\n"
                            + (failure.stdout or "") + (failure.stderr or ""))
            return True

        done = subprocess.run([str(exe)], capture_output=True, universal_newlines=True)
        if done.returncode != 0:
            failures.append(f"the bounded-cache harness failed: {done.stdout}{done.stderr}")
            return True
        results = dict(line.split() for line in done.stdout.split("\n") if line.strip())
        for name, why in EXPECTED.items():
            if results.get(name) != "1":
                failures.append(f"compiled bound contract [{name}]: {why} "
                                f"(harness said {results.get(name)!r})")
    return True


failures += check_receipt(SERVER)
failures += check_live_cases(SERVER)
failures += check_xslib_bound(XSSET)
failures += check_cohort_bound(COHORT)
failures += check_memo_bound(LIGHT)
failures += check_pin_bytes(PINREG)
failures += check_soak_attribution()

broken_controls = controls()
if broken_controls:
    failures.append("NEGATIVE CONTROLS FAILED -- these checks cannot fail and are "
                    "therefore comments:\n    " + "\n    ".join(broken_controls))

compiled = compiled_contract()

if failures:
    raise SystemExit("evaluator memory receipt: FAIL\n  " + "\n  ".join(failures))
if not compiled:
    print("evaluator memory receipt: NOTE -- no C++ compiler found; the bound was "
          "checked by source scan only", file=sys.stderr)
print(f"evaluator memory receipt: PASS ({len(MEM_FIELDS)} fields, "
      f"{len(MEM_CACHES)} sized containers, 3 bounded caches"
      + (", 7 compiled" if compiled else "") + ")")
