#!/usr/bin/env python3
"""Contract for tools/_cxx_toolchain.py, the compiler/std-flag discovery the
four compiled-harness contract tests (case_key, gpu_physics_interface,
arena_persist, device_block_pool) now share instead of each re-guessing.

WHY THIS EXISTS.  RHEL 238's system compiler is g++ 8.5, which has a
`-std=c++2a` mode (GCC's pre-standardization name for the C++20 feature set)
but rejects `-std=c++20` outright:

    g++: error: unrecognized command line option '-std=c++20'; did you mean
    '-std=c++2a'?

Every one of the four tests used to hard-code `-std=c++20`, so a compiler
that was perfectly capable of building the harness (with the c++2a spelling)
failed loudly as though the harness itself did not compile.  This checks the
fix at the unit the bug actually lives in -- discovery and probing -- with a
FAKE compiler script standing in for g++ 8.5, g++ 8.5's future replacement,
and a compiler too old for either, so the behaviour is pinned without
needing three real toolchains on hand.

WHAT THIS CHECKS
  1. find_compiler() picks in order: $RASBERY_TEST_CXX, then $CXX, then a
     build dir's CMakeCache.txt, then PATH.
  2. probe_std_flag() prefers -std=c++20 when a compiler takes it, falls
     back to -std=c++2a when it does not (the 238 case), and returns None
     when a compiler takes neither.
  3. discover() wires the two together: a working compiler+flag pair, or
     (None, reason) -- and the reason names what was missing.
  4. is_msvc() keys off the vcvars64.bat path, nothing else.
  5. NEGATIVE CONTROLS: a scan that always finds a compiler regardless of
     $RASBERY_TEST_CXX/$CXX would hide a broken override; a probe that
     accepts a flag it was told to reject would hide the 238 symptom itself.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import _cxx_toolchain  # noqa: E402

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)


# ---------------------------------------------------------------------------
# A fake compiler.  GNU-style argv (-std=..., -O0, <src>, -o <exe>), so it
# exercises the same code path _cxx_toolchain.py drives real g++/clang++
# through, regardless of the host this test itself runs on.
# ---------------------------------------------------------------------------

_LOGIC = '''\
import os, sys

accepted = set(filter(None, os.environ.get("FAKE_CC_ACCEPTED_STDS", "").split(",")))
argv = sys.argv[1:]
std_flags = [a for a in argv if a.startswith("-std=")]
if not std_flags:
    sys.stderr.write("fake_cc: no -std flag given\\n")
    sys.exit(1)
flag = std_flags[0]
if flag not in accepted:
    hint = " -- did you mean \\'-std=c++2a\\'?" if flag == "-std=c++20" else ""
    sys.stderr.write(
        "fake_cc: error: unrecognized command line option \\'%s\\'%s\\n" % (flag, hint))
    sys.exit(1)
if "-o" in argv:
    out = argv[argv.index("-o") + 1]
    with open(out, "wb"):
        pass
sys.exit(0)
'''


def write_fake_compiler(tmp: Path, accepted: set) -> str:
    """A compiler binary at tmp/fake_cc(.bat) that accepts exactly the
    -std=... flags in `accepted` and refuses every other one, the way a real
    compiler refuses a standard mode it was not built to support."""
    logic = tmp / "fake_cc_logic.py"
    logic.write_text(_LOGIC, encoding="utf-8")
    if os.name == "nt":
        script = tmp / "fake_cc.bat"
        script.write_text(
            "@echo off\r\n"
            + '"%s" "%s" %%*\r\n' % (sys.executable, logic),
            encoding="utf-8")
        compiler = str(script)
    else:
        script = tmp / "fake_cc"
        script.write_text(
            "#!/bin/sh\n"
            + 'exec "%s" "%s" "$@"\n' % (sys.executable, logic),
            encoding="utf-8")
        script.chmod(0o755)
        compiler = str(script)
    return compiler


class EnvSandbox:
    """Save/restore the handful of env vars + PATH these tests poke, so a
    real toolchain on the machine running this test cannot leak in and mask
    a broken override, and this test cannot leak out and break later ones."""

    KEYS = ("RASBERY_TEST_CXX", "CXX", "PATH")

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False

    @staticmethod
    def clear_all():
        for k in EnvSandbox.KEYS:
            os.environ.pop(k, None)
        # An empty PATH still has to be a string on some platforms' getenv.
        os.environ["PATH"] = ""


class NoMSVCFallback:
    """Suppresses _cxx_toolchain's Program-Files scan for vcvars64.bat, which
    (unlike PATH/$CXX/$RASBERY_TEST_CXX) is not something this test can
    sandbox by clearing an env var -- a dev box that happens to have Visual
    Studio installed would otherwise make the 'nothing is configured' checks
    below find a real compiler and pass for the wrong reason. Host 238 is
    Linux, where that scan is a no-op anyway; this stands in for that on any
    OS so the negative control means the same thing everywhere."""

    def __enter__(self):
        self._orig = _cxx_toolchain._msvc_vcvars
        _cxx_toolchain._msvc_vcvars = lambda: None
        return self

    def __exit__(self, *exc):
        _cxx_toolchain._msvc_vcvars = self._orig
        return False


# ---------------------------------------------------------------------------
# 1. probe_std_flag(): prefers c++20, falls back to c++2a, None if neither.
# ---------------------------------------------------------------------------

def check_probe_prefers_cxx20():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        compiler = write_fake_compiler(tmp, {"-std=c++20", "-std=c++2a"})
        os.environ["FAKE_CC_ACCEPTED_STDS"] = "-std=c++20,-std=c++2a"
        try:
            flag = _cxx_toolchain.probe_std_flag(compiler)
        finally:
            os.environ.pop("FAKE_CC_ACCEPTED_STDS", None)
        if flag != "-std=c++20":
            fail(f"probe_std_flag preferred {flag!r} over -std=c++20 when both work")


def check_probe_falls_back_to_cxx2a():
    """The host 238 case: g++ 8.5 takes -std=c++2a, not -std=c++20."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        compiler = write_fake_compiler(tmp, {"-std=c++2a"})
        os.environ["FAKE_CC_ACCEPTED_STDS"] = "-std=c++2a"
        try:
            flag = _cxx_toolchain.probe_std_flag(compiler)
        finally:
            os.environ.pop("FAKE_CC_ACCEPTED_STDS", None)
        if flag != "-std=c++2a":
            fail(f"probe_std_flag returned {flag!r}, not the c++2a fallback, for a "
                "compiler that only accepts -std=c++2a (the host 238 symptom)")


def check_probe_none_when_neither_works():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        compiler = write_fake_compiler(tmp, set())
        os.environ["FAKE_CC_ACCEPTED_STDS"] = ""
        try:
            flag = _cxx_toolchain.probe_std_flag(compiler)
        finally:
            os.environ.pop("FAKE_CC_ACCEPTED_STDS", None)
        if flag is not None:
            fail(f"probe_std_flag returned {flag!r} for a compiler that accepts "
                "neither -std=c++20 nor -std=c++2a")


# ---------------------------------------------------------------------------
# 2. find_compiler(): $RASBERY_TEST_CXX > $CXX > CMakeCache.txt > PATH.
# ---------------------------------------------------------------------------

def check_find_compiler_env_priority():
    with tempfile.TemporaryDirectory() as raw, EnvSandbox():
        tmp = Path(raw)
        EnvSandbox.clear_all()
        cxx_dir = tmp / "cxx"
        cxx_dir.mkdir()
        cxx_cc = write_fake_compiler(cxx_dir, set())

        os.environ["CXX"] = cxx_cc
        found = _cxx_toolchain.find_compiler(tmp)
        if found != cxx_cc:
            fail(f"find_compiler ignored $CXX ({cxx_cc!r}), returned {found!r}")

        # A distinct compiler under $RASBERY_TEST_CXX must win over $CXX.
        override_dir = tmp / "override"
        override_dir.mkdir()
        override_cc = write_fake_compiler(override_dir, set())
        os.environ["RASBERY_TEST_CXX"] = override_cc
        found = _cxx_toolchain.find_compiler(tmp)
        if found != override_cc:
            fail(f"find_compiler did not prefer $RASBERY_TEST_CXX ({override_cc!r}) "
                f"over $CXX, returned {found!r}")


def check_find_compiler_cmake_cache():
    with tempfile.TemporaryDirectory() as raw, EnvSandbox():
        tmp = Path(raw)
        EnvSandbox.clear_all()
        build_dir = tmp / "build_release"
        build_dir.mkdir()
        cache_cc = write_fake_compiler(tmp, set())
        (build_dir / "CMakeCache.txt").write_text(
            "// comment line the parser must skip\n"
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            f"CMAKE_CXX_COMPILER:FILEPATH={cache_cc}\n",
            encoding="utf-8")
        found = _cxx_toolchain.find_compiler(tmp)
        if found != cache_cc:
            fail(f"find_compiler did not read CMAKE_CXX_COMPILER from "
                f"build_release/CMakeCache.txt, returned {found!r}")


def check_find_compiler_none_when_nothing_present():
    """Negative control: with every override cleared, PATH emptied, and no
    build dir, find_compiler must say None -- not silently find a real
    compiler leaked in from this machine's own PATH."""
    with tempfile.TemporaryDirectory() as raw, EnvSandbox(), NoMSVCFallback():
        tmp = Path(raw)
        EnvSandbox.clear_all()
        found = _cxx_toolchain.find_compiler(tmp)
        if found is not None:
            fail(f"find_compiler found {found!r} with $RASBERY_TEST_CXX, $CXX, PATH "
                "all cleared and no build dir present -- the negative control is "
                "vacuous")


# ---------------------------------------------------------------------------
# 3. discover(): the compiler + probe wired together.
# ---------------------------------------------------------------------------

def check_discover_success_with_fallback():
    with tempfile.TemporaryDirectory() as raw, EnvSandbox():
        tmp = Path(raw)
        EnvSandbox.clear_all()
        compiler = write_fake_compiler(tmp, {"-std=c++2a"})
        os.environ["RASBERY_TEST_CXX"] = compiler
        os.environ["FAKE_CC_ACCEPTED_STDS"] = "-std=c++2a"
        try:
            toolchain, reason = _cxx_toolchain.discover(tmp)
        finally:
            os.environ.pop("FAKE_CC_ACCEPTED_STDS", None)
        if toolchain is None:
            fail(f"discover() returned None with reason {reason!r} for a fake "
                "compiler that accepts -std=c++2a")
        elif toolchain.std_flag != "-std=c++2a" or toolchain.is_msvc:
            fail(f"discover() returned an unexpected toolchain: {toolchain!r}")


def check_discover_skip_reason_when_neither_std_works():
    with tempfile.TemporaryDirectory() as raw, EnvSandbox():
        tmp = Path(raw)
        EnvSandbox.clear_all()
        compiler = write_fake_compiler(tmp, set())
        os.environ["RASBERY_TEST_CXX"] = compiler
        os.environ["FAKE_CC_ACCEPTED_STDS"] = ""
        try:
            toolchain, reason = _cxx_toolchain.discover(tmp)
        finally:
            os.environ.pop("FAKE_CC_ACCEPTED_STDS", None)
        if toolchain is not None:
            fail(f"discover() returned a toolchain ({toolchain!r}) for a compiler "
                "that accepts neither standard flag")
        if not reason or "c++20" not in reason or "c++2a" not in reason:
            fail(f"discover()'s SKIP reason does not name what was tried: {reason!r}")


def check_discover_reason_when_no_compiler():
    with tempfile.TemporaryDirectory() as raw, EnvSandbox(), NoMSVCFallback():
        tmp = Path(raw)
        EnvSandbox.clear_all()
        toolchain, reason = _cxx_toolchain.discover(tmp)
        if toolchain is not None:
            fail(f"discover() found a toolchain ({toolchain!r}) with nothing on "
                "PATH and no override set")
        if not reason or "compiler" not in reason.lower():
            fail(f"discover()'s no-compiler reason is not clear: {reason!r}")


# ---------------------------------------------------------------------------
# 4. is_msvc(): keyed off the vcvars64.bat path, nothing else.
# ---------------------------------------------------------------------------

def check_is_msvc():
    if not _cxx_toolchain.is_msvc(r"C:\VS\VC\Auxiliary\Build\vcvars64.bat"):
        fail("is_msvc() said False for a real vcvars64.bat path")
    if not _cxx_toolchain.is_msvc(r"C:\VS\VC\Auxiliary\Build\VCVARS64.BAT"):
        fail("is_msvc() is case-sensitive and should not be")
    if _cxx_toolchain.is_msvc("/usr/bin/g++"):
        fail("is_msvc() said True for g++")
    if _cxx_toolchain.is_msvc(""):
        fail("is_msvc() said True for an empty string")


def main() -> int:
    if shutil.which(sys.executable) is None and not Path(sys.executable).is_file():
        fail("sys.executable does not resolve -- the fake compiler wrapper "
            "cannot invoke python, so this whole test is vacuous")

    check_probe_prefers_cxx20()
    check_probe_falls_back_to_cxx2a()
    check_probe_none_when_neither_works()
    check_find_compiler_env_priority()
    check_find_compiler_cmake_cache()
    check_find_compiler_none_when_nothing_present()
    check_discover_success_with_fallback()
    check_discover_skip_reason_when_neither_std_works()
    check_discover_reason_when_no_compiler()
    check_is_msvc()

    if failures:
        print("cxx toolchain contract: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("cxx toolchain contract: PASS (probe fallback + priority order + "
          "discover reasons + is_msvc, all against a fake compiler)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
