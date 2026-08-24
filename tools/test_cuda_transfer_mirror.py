#!/usr/bin/env python3
"""Compile and execute the byte-exact transfer mirror contract."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "CudaTransferMirror.h"


def main() -> int:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("no C++ compiler available")

    source = r'''
#include "CudaTransferMirror.h"
#include <array>
#include <bit>
#include <cstdint>

int main() {
    using rasbery::cuda_transfer::ByteExactMirror;
    ByteExactMirror<double> mirror;
    std::array<double, 4> a{1.0, 2.0, 0.0, 4.0};
    if (mirror.matches(a.data(), a.size())) return 1;
    mirror.commit(a.data(), a.size());
    if (!mirror.valid() || !mirror.matches(a.data(), a.size())) return 2;

    auto b = a;
    b[1] = 3.0;
    if (mirror.matches(b.data(), b.size())) return 3;

    // Numeric equality is not enough: +0 and -0 have different bytes.
    auto signed_zero = a;
    signed_zero[2] = -0.0;
    if (mirror.matches(signed_zero.data(), signed_zero.size())) return 4;

    // Distinct quiet-NaN payloads must remain distinguishable too.
    auto nan1 = a;
    auto nan2 = a;
    nan1[0] = std::bit_cast<double>(std::uint64_t{0x7ff8000000000001ULL});
    nan2[0] = std::bit_cast<double>(std::uint64_t{0x7ff8000000000002ULL});
    mirror.commit(nan1.data(), nan1.size());
    if (!mirror.matches(nan1.data(), nan1.size())) return 5;
    if (mirror.matches(nan2.data(), nan2.size())) return 6;

    mirror.invalidate();
    if (mirror.valid() || mirror.matches(nan1.data(), nan1.size())) return 7;
    return 0;
}
'''

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cpp = tmp_path / "mirror_test.cpp"
        exe = tmp_path / "mirror_test"
        cpp.write_text(source, encoding="utf-8")
        subprocess.run(
            [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
             "-I", str(ROOT / "src"), str(cpp), "-o", str(exe)],
            check=True,
        )
        subprocess.run([str(exe)], check=True)
    print("cuda transfer mirror: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
