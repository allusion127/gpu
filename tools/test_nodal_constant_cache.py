#!/usr/bin/env python3
"""Static contract for the bit-exact Nodal material-constant cache."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
header = (root / "src" / "Nodal.h").read_text(encoding="utf-8")
source = (root / "src" / "Nodal.cpp").read_text(encoding="utf-8")

required_header = ("_constant_xsrf", "_constant_xsdf")
required_source = (
    "std::numeric_limits<double>::quiet_NaN()",
    "if (unchanged) return;",
    "_constant_xsrf[lkg0 + ig] == xs.xsrf(ig, lk)",
    "_constant_xsdf[lkg0 + ig] == xs.xsdf(ig, lk)",
    "delete[] _constant_xsrf;",
    "delete[] _constant_xsdf;",
)

missing = [token for token in required_header if token not in header]
missing += [token for token in required_source if token not in source]
if missing:
    raise SystemExit("nodal constant cache contract missing: " + ", ".join(missing))

# The cache key intentionally uses exact double equality.  A changed material
# input must force recomputation; toleranced equality would change physics.
if "std::abs(_constant_xs" in source or "memcmp" in source:
    raise SystemExit("nodal constant cache must use scalar bit-stable equality")

print("nodal constant cache: PASS")
