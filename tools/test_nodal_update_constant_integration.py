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
# Rev.7.1 W3 item 1 MERGED THE TWO SWEEPS.  TryDriveGpu and driveBody each ran
# their own copy of the updateConstant loop; the sweep became CONDITIONAL (it is
# skipped when XSSet::macroXsGeneration() has not moved, which is bit-exact
# because every node would then take the early-out), and a gate spelled twice is
# a gate that can be answered twice differently.  So there is now exactly one
# sweep, in Nodal::updateConstantsIfMoved, and both drives call it.
assert source.count("reduction(| : constants_changed)") == 1
assert "void Nodal::updateConstantsIfMoved()" in source
for caller in ("bool Nodal::TryDriveGpu()", "void Nodal::driveBody()"):
    at = source.index(caller)
    assert "updateConstantsIfMoved()" in source[at:source.index(chr(10) + "}", at)], caller
# Two sites still advance the residency generation: the ordinary one, and the
# RASBERY_NODAL_CONST_VERIFY path, which has just rebuilt the coefficients after
# catching a writer that did not announce itself and must tell the device.
assert source.count("++_const_generation;") == 2
assert "Racy increments from the omp for are fine" not in source
print("PASS Nodal::updateConstant integration contract")
