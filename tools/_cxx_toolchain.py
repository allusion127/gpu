"""Shared C++ toolchain discovery for the compiled halves of the contract
tests (test_case_key_contract.py, test_gpu_physics_interface_contract.py,
test_arena_persist_contract.py, test_device_block_pool_contract.py).

Each of those tests compiles a small harness against src/ to check a C++
implementation against its python twin.  They used to each hand-roll their
own "find a g++/clang++/MSVC and hard-code -std=c++20" logic, which is fine
until the host's system compiler does not have a c++20 mode: RHEL 238's
system g++ 8.5 only goes up to `-std=c++2a` (GCC's pre-standardization name
for the same C++20 feature set) and rejects `-std=c++20` outright with
"unrecognized command line option ... did you mean '-std=c++2a'?" -- which
every caller was previously treating as a genuine harness compile failure.

This module separates the two questions a caller actually has:

  1. WHICH COMPILER.  find_compiler() tries, in order: $RASBERY_TEST_CXX (an
     explicit override for whoever is running the tests), $CXX (the ambient
     convention), the CMAKE_CXX_COMPILER recorded in a build*/CMakeCache.txt
     next to the repo root (so a project that already configured a build
     reuses that compiler rather than guessing again), then g++/c++/clang++
     on PATH, then (Windows only) the newest MSVC's vcvars64.bat.

  2. WHICH STANDARD FLAG.  probe_std_flag() actually compiles a trivial
     translation unit with -std=c++20 and, if that is rejected, -std=c++2a,
     and returns whichever one worked (or None if neither did).

discover() combines both and returns a Toolchain the caller can compile
its real harness with, or (None, reason) when no usable toolchain exists --
callers use that to print SKIP instead of FAIL, since "this host has no
c++20-or-c++2a-capable compiler" is a fact about the host, not a defect the
harness compile step found.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Newest first: c++20 is what every harness is written against; c++2a is
# GCC/Clang's spelling of the same feature set from before C++20 was
# ratified, and it is what a GCC 8-10 system compiler actually understands.
_GNU_STD_CANDIDATES: Tuple[str, ...] = ("-std=c++20", "-std=c++2a")
_MSVC_STD_CANDIDATES: Tuple[str, ...] = ("/std:c++20",)


class Toolchain(NamedTuple):
    compiler: str
    std_flag: str
    is_msvc: bool


def is_msvc(compiler: str) -> bool:
    return compiler.lower().endswith("vcvars64.bat")


def _env_compiler(var: str) -> Optional[str]:
    val = os.environ.get(var)
    if not val:
        return None
    resolved = shutil.which(val)
    if resolved:
        return resolved
    return val if Path(val).is_file() else None


def _cmake_cache_compiler(root: Path) -> Optional[str]:
    for cache in sorted(root.glob("build*/CMakeCache.txt")):
        try:
            text = cache.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:"):
                _, _, value = line.partition("=")
                value = value.strip()
                if value and Path(value).is_file():
                    return value
    return None


def _msvc_vcvars() -> Optional[str]:
    if os.name != "nt":
        return None
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        done = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, universal_newlines=True)
        found_roots = done.stdout.strip().splitlines()
        if done.returncode == 0 and found_roots:
            bat = Path(found_roots[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if bat.is_file():
                return str(bat)
    for base in (r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio"):
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            if "vcvars64.bat" in files:
                return os.path.join(dirpath, "vcvars64.bat")
    return None


def find_compiler(root: Optional[Path] = None) -> Optional[str]:
    """The compiler the contract tests should use, or None if nothing usable
    was found anywhere.  See the module docstring for the search order."""
    root = root or REPO_ROOT
    for var in ("RASBERY_TEST_CXX", "CXX"):
        found = _env_compiler(var)
        if found:
            return found
    found = _cmake_cache_compiler(root)
    if found:
        return found
    for name in ("g++", "c++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return _msvc_vcvars()


def probe_std_flag(compiler: str) -> Optional[str]:
    """The newest standard flag `compiler` actually accepts: -std=c++20,
    falling back to -std=c++2a (MSVC: /std:c++20 only).  None if it accepts
    neither -- an actual compile, not a version-string guess, because
    version strings lie about mode support often enough not to trust."""
    candidates = _MSVC_STD_CANDIDATES if is_msvc(compiler) else _GNU_STD_CANDIDATES
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        cpp = tmp / "probe.cpp"
        cpp.write_text("int main() { return 0; }\n", encoding="utf-8")
        for flag in candidates:
            exe = tmp / ("probe.exe" if os.name == "nt" else "probe")
            if is_msvc(compiler):
                script = tmp / "probe_build.bat"
                script.write_text(
                    "@echo off\r\n"
                    + 'call "%s" >nul 2>&1\r\n' % compiler
                    + 'cd /d "%s"\r\n' % tmp
                    + 'cl /nologo %s /EHsc "%s" /Fe:"%s"\r\n' % (flag, cpp, exe),
                    encoding="utf-8")
                result = subprocess.run(["cmd", "/c", str(script)], cwd=str(tmp),
                                        capture_output=True, universal_newlines=True)
            else:
                result = subprocess.run(
                    [compiler, flag, "-O0", str(cpp), "-o", str(exe)],
                    capture_output=True, universal_newlines=True)
            if result.returncode == 0 and exe.is_file():
                return flag
    return None


def discover(root: Optional[Path] = None) -> Tuple[Optional[Toolchain], Optional[str]]:
    """(Toolchain, None) when a compiler and a working standard flag were
    both found; (None, reason) otherwise, where reason is a human-readable
    string a caller can print alongside a SKIP.  Note the difference from
    "compile failed": this only ever returns a reason for two host-level
    facts (no compiler at all, or the one found accepts neither c++20 nor
    c++2a) -- an error compiling the caller's own harness is a separate,
    still-FAIL, condition callers should keep detecting themselves."""
    compiler = find_compiler(root)
    if compiler is None:
        return None, ("no C++ compiler found ($RASBERY_TEST_CXX, $CXX, a build dir's "
                      "CMakeCache.txt, or g++/c++/clang++ on PATH)")
    std_flag = probe_std_flag(compiler)
    if std_flag is None:
        return None, f"{compiler} accepts neither -std=c++20 nor -std=c++2a"
    return Toolchain(compiler=compiler, std_flag=std_flag, is_msvc=is_msvc(compiler)), None
