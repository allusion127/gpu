#!/usr/bin/env python3
"""Static exactness contract for the equilibrium-Xe reconstruction fast path."""

from pathlib import Path


source = (Path(__file__).resolve().parents[1] / "src" / "XSSet.cpp").read_text(
    encoding="utf-8"
)
start = source.index("double XSSet::UpdateEquilibriumXenon")
stop = source.index("void XSSet::DepleteNode", start)
body = source[start:stop]
required = (
    "Only fuel-node Xe-chain densities changed",
    "ReconstructNode(static_cast<size_t>(l))",
)
missing = [token for token in required if token not in body]
if "UpdateFlatXS();" in body:
    missing.append("redundant UpdateFlatXS remains in equilibrium-Xe path")
if missing:
    raise SystemExit("Xe reconstruction fast-path contract missing: " + ", ".join(missing))
print("equilibrium-Xe reconstruction fast path: PASS")
