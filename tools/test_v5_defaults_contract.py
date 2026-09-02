#!/usr/bin/env python3
"""The four v5 defaults are ON, the four off switches survive, and nothing else moved.

WHY THIS FILE EXISTS.  On 2026-08-30 four features stopped being opt-in and
became the shipped behaviour of the binary:

    RASBERY_GPU_CMFD_FUSE   0  ->  15   (src/CudaBICGBackend.cu, cmfdFuseMask)
    RASBERY_GPU_XE_TXN      0  ->   1   (src/CudaXsReconBackend.cu)
    RASBERY_RESULT_ASYNC    0  ->   1   (src/IoWriter.h, resultAsyncRequested)
    RASBERY_GPU_FLATXS_CTA  0  ->   1   (src/CudaXsReconBackend.cu)

All four are B0: each was measured bit-identical to its own off arm on BOTH
238 (E:/rasbery_runs/2026-08-30/238/pricing_388e8f2.md blocks 4, 5, 11, 12) and
181 (E:/rasbery_runs/20260830/181/gates_8919331.md blocks 3, 11, 5, 13) -- by
h5diff for the three that touch HDF5, and by `cmp` on the 119 MB pin-power CSV
for the one that h5diff cannot see at all.

A DEFAULT IS THE ONE PROPERTY NO RUN REPORTS.  Every other claim in this
campaign is checked by a receipt printed by the run that made it.  "What does
this binary do when the operator types nothing" is different: the run that would
reveal a silent revert is the run nobody thinks to make, because it looks
exactly like the run before it and prints a wall that is merely disappointing.
So the defaults are pinned HERE, in a scan of the source, with a negative
control for each direction of drift -- back to opt-in, and forward to
no-off-switch.

WHAT THIS FILE DELIBERATELY DOES NOT ASSERT.

  * CRAM and PPR stay OPT-IN.  They are N1 -- RASBERY_GPU_CRAM moves the
    trajectory by design (238 digest 1f36e75dc00ed2b4 / 4377 against the v3
    arm's 0d15abf29d222a02 / 4382) and the PPR device loop moves the pin-derived
    datasets.  An N1 feature does not become a default because it passed its
    gate; per tools/promotion_gate.py it is HOLD until a recorded project
    acceptance says otherwise.  The v5 production env names them EXPLICITLY, and
    this file checks their resolvers are still opt-in so that a later edit
    cannot promote them alongside the four that were.
  * XFER_ELIDE and XFER_LEDGER stay OPT-IN, for the simpler reason that neither
    has been priced at all.
  * Anything about WALL TIME.  A default is a statement about which code runs,
    not about how fast it is; the walls live in the manifest and the docs.

USAGE
    tools/test_v5_defaults_contract.py
"""
from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BICG = ROOT / "src" / "CudaBICGBackend.cu"
XSRECON = ROOT / "src" / "CudaXsReconBackend.cu"
XSRECON_H = ROOT / "src" / "CudaXsReconBackend.h"
IO_WRITER = ROOT / "src" / "IoWriter.h"
STUB = ROOT / "src" / "CudaXsReconBackendStub.cpp"
XFER = ROOT / "src" / "XferLedger.h"
DRIVER = ROOT / "src" / "Driver.h"
CRAM = ROOT / "src" / "CudaCramBackend.cu"
PPR = ROOT / "src" / "CudaPprBackend.cu"
MANIFEST = ROOT / "test" / "reference" / "validation_baseline_manifest_v5.json"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        failures.append("cannot read %s (%s)" % (path, exc))
        return ""


def body_of(text: str, opener: str) -> str:
    """The braced body that follows `opener`, by brace counting.

    Brace counting rather than a regex because these bodies contain lambdas and
    nested initialisers, and a regex that stopped at the first `}` would score
    half a resolver.
    """
    start = text.find(opener)
    if start < 0:
        return ""
    i = text.find("{", start)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return ""


BICG_SRC = read(BICG)
XSRECON_SRC = read(XSRECON)
XSRECON_H_SRC = read(XSRECON_H)
IO_WRITER_SRC = read(IO_WRITER)
STUB_SRC = read(STUB)
XFER_SRC = read(XFER)
DRIVER_SRC = read(DRIVER)


# ===========================================================================
# 1. RASBERY_GPU_CMFD_FUSE defaults to 15, and 0 is still reachable.
# ===========================================================================
#
# The mask is not a bool, so "default on" has to be said as a VALUE.  It is said
# once, as kFuseDefaultMask, and that constant is pinned to a NAMED SET rather
# than to a literal: mask 15 is the member of the family that cleared the 0.2 s
# adoption threshold (-0.396 s interleaved on 238 block 4b, where mask 4 managed
# -0.183 s and was not adopted), so a default that quietly became a different
# member would be a wall claim nobody measured.
#
# THE NAME USED TO BE kFuseAllBits AND IS NOW kFusePricedBits.  Until WP21-A the
# two were the same set, so pinning the default to "every declared bit" was also
# pinning it to "every priced bit" -- by coincidence, not by construction.  WP21-A
# declared a fifth bit (kFuseNorm) that it deliberately did NOT price, and at that
# moment the coincidence broke: kFuseAllBits is the VALIDATION set (what a user may
# ask for, so an undeclared bit cannot arm an undeclared path) and kFusePricedBits
# is the ADOPTION set (what the tree claims).  This rule follows the split rather
# than papering over it -- it still refuses a literal, still refuses a silent
# return to the reference mask, and now also refuses the reverse mistake of
# adopting an unpriced bit by writing kFuseAllBits back into the default.
def rule_fuse_default(src: str) -> list[str]:
    bad: list[str] = []
    default = re.search(r"kFuseDefaultMask = ([^;}]*)", src)
    if default is None:
        bad.append("kFuseDefaultMask is gone -- the default is no longer said "
                   "anywhere as a value")
    else:
        spelling = default.group(1).strip()
        if spelling != "kFusePricedBits":
            bad.append("kFuseDefaultMask is not defined as kFusePricedBits (found "
                       "%r) -- the adopted default is mask 15 and it must be named "
                       "once, not spelled as a literal in the resolver and not "
                       "widened to the validation set" % spelling)
    priced = re.search(r"kFusePricedBits = ([^,;}]*)", src)
    if priced is None:
        bad.append("kFusePricedBits is gone -- the adopted set has no name, so the "
                   "default cannot be checked against anything")
    else:
        for bit in ("kFuseDot", "kFuseDot2", "kFuseWiel", "kFuseSweepPre"):
            if not re.search(r"\b%s\b" % bit, priced.group(1)):
                bad.append("kFusePricedBits no longer names %s -- the adopted set is "
                           "the four bits both hosts measured at digest "
                           "0d15abf29d222a02 / 1d897e3f77204799" % bit)
        if re.search(r"\bkFuseNorm\b", priced.group(1)):
            bad.append("kFuseNorm is inside kFusePricedBits -- bit 4 has not been "
                       "priced on 238, so it may be REQUESTABLE but not ADOPTED")
    allbits = re.search(r"kFuseAllBits\s*=\s*([^,;}]*)", src)
    if allbits is None:
        bad.append("kFuseAllBits is gone -- there is nothing to validate a parsed "
                   "mask against")
    elif not re.search(r"\bkFusePricedBits\b", allbits.group(1)):
        bad.append("kFuseAllBits is not built from kFusePricedBits -- the validation "
                   "set must be a superset of the adopted set, or a default the tree "
                   "claims could be masked away by the resolver")
    gate = body_of(src, "unsigned cmfdFuseMask(")
    if not gate:
        return bad + ["cmfdFuseMask() is gone -- there is no latch for the mask"]
    if "kFuseDefaultMask" not in gate:
        bad.append("cmfdFuseMask() does not resolve an unset variable to "
                   "kFuseDefaultMask")
    if re.search(r"return\s+0u", gate):
        bad.append("cmfdFuseMask() still falls back to 0; since v5 that value is "
                   "reachable only by typing RASBERY_GPU_CMFD_FUSE=0, which strtoul "
                   "already produces")
    if "kFuseAllBits" not in gate:
        bad.append("cmfdFuseMask() no longer masks the parsed value to the declared "
                   "bits -- an unknown bit could arm an undeclared path")
    return bad


failures += rule_fuse_default(BICG_SRC)


# ===========================================================================
# 2. The two boolean device arms default ON through !envFlagDisabled.
# ===========================================================================
#
# envFlagEnabled is "absent means off"; envFlagDisabled is "present AND falsy
# means off".  The whole flip is which of the two the resolver calls, so that is
# what gets pinned -- not a comment, because a comment cannot be wrong in a way
# that changes the binary.
def rule_boolean_default_on(src: str, name: str, spelling: str) -> list[str]:
    bad: list[str] = []
    if spelling not in src:
        bad.append("%s is not resolved with %s -- absent must mean ON since the v5 "
                   "freeze" % (name, spelling))
    if ('envFlagEnabled("%s")' % name) in src:
        bad.append("%s is still read with envFlagEnabled somewhere: two readers of "
                   "one knob will disagree about the default" % name)
    return bad


FLIPPED_BOOLEANS = (
    ("RASBERY_GPU_FLATXS_CTA", '!envFlagDisabled("RASBERY_GPU_FLATXS_CTA")'),
    ("RASBERY_GPU_XE_TXN", '!envFlagDisabled("RASBERY_GPU_XE_TXN")'),
)
for _name, _spelling in FLIPPED_BOOLEANS:
    failures += rule_boolean_default_on(XSRECON_SRC, _name, _spelling)


# ===========================================================================
# 3. RASBERY_RESULT_ASYNC defaults ON, and an EMPTY string is still OFF.
# ===========================================================================
#
# The empty case is not pedantry.  `export RASBERY_RESULT_ASYNC=` is how a shell
# script spells "off" by accident; every other RASBERY_* gate reads it as falsy
# (envFlagDisabled treats "" as falsy), and a gate that read it as "set, so
# default" would be the one knob in the tree that disagreed with the others
# about the same shell.
def rule_result_async_default(src: str) -> list[str]:
    bad: list[str] = []
    gate = body_of(src, "inline bool resultAsyncRequested()")
    if not gate:
        return ["resultAsyncRequested() is gone -- the gate has no default"]
    flat = " ".join(gate.split())
    if "if (value == nullptr) return true;" not in flat:
        bad.append("resultAsyncRequested() does not resolve an UNSET variable to "
                   "true; async is the v5 default")
    if "return false" not in gate:
        bad.append("resultAsyncRequested() has no falsy path -- RASBERY_RESULT_ASYNC=0 "
                   "must still reach the inline emitter, because that arm IS the "
                   "reference the byte-identity claim is measured against")
    if "mode() == Mode::Thread" not in src:
        bad.append("async is no longer conditioned on the writer thread existing; "
                   "under RASBERY_IO_WRITER=inline the request must degrade to sync "
                   "rather than spawn a thread the goldens were never frozen on")
    return bad


failures += rule_result_async_default(IO_WRITER_SRC)


# ===========================================================================
# 4. The off switches are written where the arms are declared.
# ===========================================================================
#
# An arm whose off switch is not named beside it is an arm nobody will think to
# turn off, and for both of these the off arm is load-bearing: CTA=0 is the
# reference kernel the B0 replay scores against, and TXN=0 is the only arm the
# forms audit can run on.
for _token in ("RASBERY_GPU_FLATXS_CTA=0", "RASBERY_GPU_XE_TXN=0"):
    check(_token in XSRECON_H_SRC,
          "CudaXsReconBackend.h does not name %s beside the arm it turns off" % _token)


# ===========================================================================
# 5. The no-CUDA stub still reports both device arms disabled.
# ===========================================================================
#
# NOT AN INCONSISTENCY WITH SECTION 2.  The default answers "which device kernel
# runs"; the stub answers "is there a device at all".  A stub that returned true
# would claim a CTA kernel exists in a build that compiled no CUDA.
for _stub_line in ("bool rasberyGpuFlatXsCtaEnabled() { return false; }",
                   "bool rasberyGpuXeTxnEnabled() { return false; }"):
    check(_stub_line in STUB_SRC,
          "the no-CUDA stub no longer reports the arm disabled: " + _stub_line)


# ===========================================================================
# 6. What did NOT flip.
# ===========================================================================
STILL_OPT_IN = (
    ("RASBERY_GPU_CRAM", CRAM, 'truthy(std::getenv("RASBERY_GPU_CRAM"))'),
    ("RASBERY_GPU_PPR", PPR, 'truthy(std::getenv("RASBERY_GPU_PPR"))'),
    ("RASBERY_GPU_PPR_GRAPH", PPR, 'truthy(std::getenv("RASBERY_GPU_PPR_GRAPH"))'),
)
for _name, _path, _spelling in STILL_OPT_IN:
    check(_spelling in read(_path),
          "%s is no longer resolved as %s.  It is N1 (or its rider) and stays "
          "EXPLICIT in the production env; a default-on N1 feature changes the "
          "answer for operators who never accepted the change" % (_name, _spelling))

for _name in ("RASBERY_GPU_XFER_ELIDE", "RASBERY_XFER_LEDGER"):
    check(('std::getenv("%s")' % _name) in XFER_SRC,
          "%s is no longer read in src/XferLedger.h" % _name)
check(XFER_SRC.count("if (v == nullptr) return false;") >= 2,
      "src/XferLedger.h no longer resolves an unset transfer knob to false; "
      "neither XFER_ELIDE nor XFER_LEDGER has been priced, and an unpriced "
      "feature does not get a default")


# ===========================================================================
# 7. The digest's `env` field still reports SET-VS-UNSET, and says so.
# ===========================================================================
#
# armEnvJson() prints the raw exported string, or `null` when the variable is
# unset.  That is unchanged by the flip and must stay unchanged: the receipt
# reports what the run was ASKED for, so it can never disagree with the solver
# about what the run was.  The CONSEQUENCE -- that an env-free v5 run prints
# `null` where the old explicit-ON env printed "1", while printing the SAME
# trajectory digest -- is the thing a reader has to be told, so Driver.h has to
# say it beside the list.
_env_json = body_of(DRIVER_SRC, "inline std::string armEnvJson()")
check('out += "null"' in _env_json,
      "armEnvJson() no longer prints null for an unset knob -- a receipt that "
      "resolved defaults would be a second interpretation of the environment and "
      "could drift from the solver's")
for _phrase in ("SET-VS-UNSET", "case_key"):
    check(_phrase in DRIVER_SRC,
          "src/Driver.h does not document %s beside kArmEnv: after the v5 flip two "
          "runs of ONE arm can print two different `env` objects and two different "
          "case keys, and that has to be written where the list is" % _phrase)

# And the digest itself must still fold the trajectory and nothing else.
_digest_step = body_of(DRIVER_SRC, "void step(int step_number, int outer, int th_steps")
check(_digest_step != "" and "getenv" not in _digest_step,
      "the trajectory digest now reads the environment.  It must not: the whole "
      "point of the v5 flip being B0 is that an env-free run prints the SAME "
      "digest as the old explicit-ON env, and a digest that folded `env` would "
      "make that false by construction")


# ===========================================================================
# 8. The v5 manifest names all seven production knobs EXPLICITLY.
# ===========================================================================
#
# Four of them because a manifest that relied on a default would describe a
# BINARY rather than a RUN; and three (CRAM, PPR, PPR_GRAPH) because they are
# not defaults at all.  There is also an operational reason for the first four:
# src/CaseKey.h digests the RAW env strings, so an unset knob and an explicit `1`
# are two different case keys for one arm, and a controller that stopped
# exporting them would miss every cached case.
if MANIFEST.is_file():
    _manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    _env_text = _manifest.get("benchmark_input", {}).get("env", "")
    for _token in ("RASBERY_GPU_CMFD_FUSE=15", "RASBERY_GPU_XE_TXN=1",
                   "RASBERY_RESULT_ASYNC=1", "RASBERY_GPU_FLATXS_CTA=1",
                   "RASBERY_GPU_CRAM=1", "RASBERY_GPU_PPR=1",
                   "RASBERY_GPU_PPR_GRAPH=1"):
        check(_token in _env_text,
              "validation_baseline_manifest_v5.json benchmark_input.env does not "
              "name %s -- the manifest has to be self-describing" % _token)
    check(_manifest.get("frozen") is True,
          "validation_baseline_manifest_v5.json is not frozen:true")
else:
    failures.append("test/reference/validation_baseline_manifest_v5.json is missing")


# ===========================================================================
# NEGATIVE CONTROLS.  Each scan, fed a source that breaks it, must FAIL.
#
# Two directions per knob wherever both are possible: BACK (the default quietly
# returns to opt-in, which is how a rebase loses a flip) and FORWARD (the off
# switch is deleted as dead code, which is how a reference arm stops being
# measurable).  A scan that cannot fail its own control proves nothing, and that
# is reported here as a failure rather than as a pass.
# ===========================================================================
CONTROL_FUSE_BACK_TO_ZERO = """
enum : unsigned { kFusePricedBits = kFuseDot | kFuseDot2 | kFuseWiel | kFuseSweepPre,
                  kFuseAllBits = kFusePricedBits | kFuseNorm };
enum : unsigned { kFuseDefaultMask = kFusePricedBits };
unsigned cmfdFuseMask() {
    static const unsigned mask = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_FUSE");
        if (v == nullptr) return 0u;
        return 1u & kFuseAllBits;
    }();
    return mask;
}
"""

CONTROL_FUSE_LITERAL_DEFAULT = """
enum : unsigned { kFusePricedBits = kFuseDot | kFuseDot2 | kFuseWiel | kFuseSweepPre,
                  kFuseAllBits = kFusePricedBits | kFuseNorm };
enum : unsigned { kFuseDefaultMask = 15u };
unsigned cmfdFuseMask() {
    static const unsigned mask = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_FUSE");
        if (v == nullptr) return kFuseDefaultMask;
        return 1u & kFuseAllBits;
    }();
    return mask;
}
"""

# The reverse mistake WP21-A made possible: the default widened from the adopted
# set to the validation set, which adopts a bit nobody priced.
CONTROL_FUSE_UNPRICED_ADOPTED = """
enum : unsigned { kFusePricedBits = kFuseDot | kFuseDot2 | kFuseWiel | kFuseSweepPre,
                  kFuseAllBits = kFusePricedBits | kFuseNorm };
enum : unsigned { kFuseDefaultMask = kFuseAllBits };
unsigned cmfdFuseMask() {
    static const unsigned mask = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_FUSE");
        if (v == nullptr) return static_cast<unsigned>(kFuseDefaultMask);
        return 1u & kFuseAllBits;
    }();
    return mask;
}
"""

CONTROL_BOOL_BACK_TO_OPT_IN = """
bool rasberyGpuFlatXsCtaEnabled() {
    static const bool on = envFlagEnabled("RASBERY_GPU_FLATXS_CTA");
    return on;
}
"""

CONTROL_BOOL_NO_OFF_SWITCH = """
bool rasberyGpuXeTxnEnabled() {
    return true;
}
"""

CONTROL_ASYNC_BACK_TO_SYNC = """
inline bool resultAsyncRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("RASBERY_RESULT_ASYNC");
        if (value == nullptr) return false;
        return true;
    }();
    return requested;
}
inline bool resultAsyncEnabled() { return resultAsyncRequested() && mode() == Mode::Thread; }
"""

CONTROL_ASYNC_NO_OFF_SWITCH = """
inline bool resultAsyncRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("RASBERY_RESULT_ASYNC");
        if (value == nullptr) return true;
        return true;
    }();
    return requested;
}
inline bool resultAsyncEnabled() { return resultAsyncRequested() && mode() == Mode::Thread; }
"""

CONTROL_ASYNC_NO_THREAD_GUARD = """
inline bool resultAsyncRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("RASBERY_RESULT_ASYNC");
        if (value == nullptr) return true;
        return false;
    }();
    return requested;
}
inline bool resultAsyncEnabled() { return resultAsyncRequested(); }
"""

CONTROLS = (
    ("the fuse default silently back to the reference mask",
     lambda: rule_fuse_default(CONTROL_FUSE_BACK_TO_ZERO)),
    ("the fuse default spelled as a literal, not kFusePricedBits",
     lambda: rule_fuse_default(CONTROL_FUSE_LITERAL_DEFAULT)),
    ("the fuse default widened to the validation set, adopting an unpriced bit",
     lambda: rule_fuse_default(CONTROL_FUSE_UNPRICED_ADOPTED)),
    ("a flipped boolean back to envFlagEnabled",
     lambda: rule_boolean_default_on(CONTROL_BOOL_BACK_TO_OPT_IN,
                                     "RASBERY_GPU_FLATXS_CTA",
                                     '!envFlagDisabled("RASBERY_GPU_FLATXS_CTA")')),
    ("a flipped boolean with its off switch deleted",
     lambda: rule_boolean_default_on(CONTROL_BOOL_NO_OFF_SWITCH,
                                     "RASBERY_GPU_XE_TXN",
                                     '!envFlagDisabled("RASBERY_GPU_XE_TXN")')),
    ("RESULT_ASYNC back to default-sync",
     lambda: rule_result_async_default(CONTROL_ASYNC_BACK_TO_SYNC)),
    ("RESULT_ASYNC with no off switch",
     lambda: rule_result_async_default(CONTROL_ASYNC_NO_OFF_SWITCH)),
    ("RESULT_ASYNC no longer degrading to sync without the writer thread",
     lambda: rule_result_async_default(CONTROL_ASYNC_NO_THREAD_GUARD)),
)


def main() -> int:
    for label, run in CONTROLS:
        if not run():
            failures.append("negative control PASSED -- the scan is vacuous for: "
                            + label)
    if failures:
        print("FAIL  tools/test_v5_defaults_contract.py")
        for problem in failures:
            print("  - " + problem)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    print("PASS  tools/test_v5_defaults_contract.py  "
          "(4 defaults, 4 off switches, 3 opt-in knobs held, %d negative controls)"
          % len(CONTROLS))
    print("  the DEFAULTS are a source property and are pinned here; the B0 claims")
    print("  behind them are hardware gates -- 238 pricing blocks 4/5/11/12 and")
    print("  181 gates blocks 3/5/11/13.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
