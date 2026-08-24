#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
source = (root / "src" / "Nodal.cpp").read_text(encoding="utf-8")
header = (root / "src" / "Nodal.h").read_text(encoding="utf-8-sig")

assert '#include "NodalConstantKernel.h"' in source
assert "bool updateConstant(const int& lk);" in header
body = re.search(r"bool\s+Nodal::updateConstant\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
assert body, "updateConstant body not found"
text = body.group(1)
assert "nodal::nodalConstantCoefficients" in text
assert "xsrf_node" in text and "xsdf_node" in text and "hmesh_node" in text
assert "std::exp" not in text
assert "xs.xsrf(ig, lk) * _g.hmesh" not in text
assert "return true;" in text and "return false;" in text
assert source.count("reduction(| : constants_changed)") == 2
assert source.count("++_const_generation;") == 2
assert "Racy increments from the omp for are fine" not in source
print("PASS Nodal::updateConstant integration contract")
