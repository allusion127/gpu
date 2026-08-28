#!/usr/bin/env python3
"""Static and arithmetic contract for CMFD's immutable geometry cache.

The production regression still owns the physics gate (full CPU golden h5diff).
This fast test protects the reason the cache is numerically inert:
  * geometry lookups occur only while CMFD is constructed;
  * the hot loops preserve their historical direction and group order; and
  * cached flat-index layouts reproduce the legacy formulas bit-for-bit on a
    deterministic synthetic two-group problem.
"""
from __future__ import annotations

from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src" / "CMFD.h").read_text(encoding="utf-8-sig")
SOURCE = (ROOT / "src" / "CMFD.cpp").read_text(encoding="utf-8-sig")


def body(text: str, signature: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise SystemExit(f"cmfd geometry cache: FAIL missing signature {signature!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit(f"cmfd geometry cache: FAIL no body for {signature!r}")
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : i + 1]
    raise SystemExit(f"cmfd geometry cache: FAIL unterminated body for {signature!r}")


required_header = (
    "_surface_node",
    "_surface_dir",
    "_node_surface",
    "_node_neighbor",
    "_node_hmesh",
    "_node_face_area",
    "_node_volume",
    "_boundary_albedo",
    "cachedSurfaceNode",
    "cachedSurfaceDirection",
    "cachedNodeSurface",
    "cachedNodeNeighbor",
    "cachedNodeHmesh",
    "cachedNodeFaceArea",
    "cachedNodeVolume",
    "cachedBoundaryAlbedo",
)
missing = [token for token in required_header if token not in HEADER]
if missing:
    raise SystemExit(f"cmfd geometry cache: FAIL missing header tokens={missing}")

constructor = body(SOURCE, "CMFD::CMFD(Geometry& g, XSSet& x)")
for token in ("_g.lklr", "_g.idirlr", "_g.lktosfc", "_g.neib", "_g.hmesh", "_g.vol", "_g.albedo"):
    if token not in constructor:
        raise SystemExit(f"cmfd geometry cache: FAIL constructor does not populate {token}")

for signature, forbidden in (
    ("void CMFD::upddtil(const int& ls)", ("_g.lklr", "_g.idirlr", "_g.hmesh", "_g.albedo")),
    ("void CMFD::upddhat(const int& ls, const double* flux, double* jnet)", ("_g.lklr",)),
    ("void CMFD::setls(const int& l)", ("_g.lktosfc", "_g.hmesh", "_g.vol")),
    ("void CMFD::updpsi(const int& l, const double* flux)", ("_g.vol",)),
):
    hot = body(SOURCE, signature)
    leaked = [token for token in forbidden if token in hot]
    if leaked:
        raise SystemExit(f"cmfd geometry cache: FAIL {signature} still uses {leaked}")

updjnet = body(HEADER, "void updjnet(const int& ls, const double* flux, double* jnet)")
if "_g.lklr" in updjnet or "_g.idirlr" in updjnet:
    raise SystemExit("cmfd geometry cache: FAIL updjnet still re-indexes Geometry")

axb = body(HEADER, "double axb(const int& ig, const int& l, const double* flux)")
if "_g.neib" in axb:
    raise SystemExit("cmfd geometry cache: FAIL axb still re-indexes Geometry")

setls = body(SOURCE, "void CMFD::setls(const int& l)")
for token in (
    "for (int idir = NDIRMAX - 1; idir >= 0; --idir)",
    "for (int idir = 0; idir < NDIRMAX; ++idir)",
    "diag_l[ige * ng + igs] = -_x.xssm(igs, ige, l) * volume",
    "diagonal += _x.xsrf(ige, l) * volume",
):
    if token not in setls:
        raise SystemExit(f"cmfd geometry cache: FAIL setls order token missing: {token}")


def bits(values: list[float]) -> list[bytes]:
    return [struct.pack("=d", value) for value in values]


# Arithmetic/layout equivalence on a deterministic synthetic 2-group mesh.
rng = random.Random(20260824)
NG, NDIR, LR = 2, 3, 2
NXYZ, NSURF = 23, 71
NEWSBT = NDIR * LR

hmesh = [[0.5 + rng.random() * 20.0 for _ in range(NDIR)] for _ in range(NXYZ)]
volume = [h[0] * h[1] * h[2] for h in hmesh]
area = [[h[1] * h[2], h[0] * h[2], h[0] * h[1]] for h in hmesh]
node_surface = [[[rng.randrange(NSURF) for _ in range(LR)] for _ in range(NDIR)] for _ in range(NXYZ)]
flat_node_surface = [node_surface[l][d][lr] for l in range(NXYZ) for d in range(NDIR) for lr in range(LR)]

dtil = [rng.uniform(0.01, 2.0) for _ in range(NSURF * NG)]
dhat = [rng.uniform(-0.5, 0.5) for _ in range(NSURF * NG)]
xssm = [rng.uniform(0.0, 0.4) for _ in range(NXYZ * NG * NG)]
xsrf = [rng.uniform(0.1, 2.0) for _ in range(NXYZ * NG)]


def dt(ig: int, ls: int) -> float:
    return dtil[ls * NG + ig]


def dh(ig: int, ls: int) -> float:
    return dhat[ls * NG + ig]


def sm(igs: int, ige: int, l: int) -> float:
    return xssm[l * NG * NG + ige * NG + igs]


def rf(ig: int, l: int) -> float:
    return xsrf[l * NG + ig]


def legacy_setls(l: int) -> tuple[list[float], list[float]]:
    diag = [0.0] * (NG * NG)
    cc = [0.0] * (NG * NEWSBT)
    old_area = [hmesh[l][1] * hmesh[l][2], hmesh[l][0] * hmesh[l][2], hmesh[l][0] * hmesh[l][1]]
    for ige in range(NG):
        for igs in range(NG):
            diag[ige * NG + igs] = -sm(igs, ige, l) * volume[l]
        diag[ige * NG + ige] += rf(ige, l) * volume[l]
        for idir in range(NDIR - 1, -1, -1):
            ls = node_surface[l][idir][0]
            cc[ige * NEWSBT + idir * LR] = (-dt(ige, ls) + dh(ige, ls)) * old_area[idir]
            diag[ige * NG + ige] += (dt(ige, ls) + dh(ige, ls)) * old_area[idir]
        for idir in range(NDIR):
            ls = node_surface[l][idir][1]
            cc[ige * NEWSBT + idir * LR + 1] = (-dt(ige, ls) - dh(ige, ls)) * old_area[idir]
            diag[ige * NG + ige] += (dt(ige, ls) - dh(ige, ls)) * old_area[idir]
    return diag, cc


def cached_setls(l: int) -> tuple[list[float], list[float]]:
    diag = [0.0] * (NG * NG)
    cc = [0.0] * (NG * NEWSBT)
    for ige in range(NG):
        for igs in range(NG):
            diag[ige * NG + igs] = -sm(igs, ige, l) * volume[l]
        diagonal = ige * NG + ige
        diag[diagonal] += rf(ige, l) * volume[l]
        for idir in range(NDIR - 1, -1, -1):
            ls = flat_node_surface[(l * NDIR + idir) * LR]
            cc[ige * NEWSBT + idir * LR] = (-dt(ige, ls) + dh(ige, ls)) * area[l][idir]
            diag[diagonal] += (dt(ige, ls) + dh(ige, ls)) * area[l][idir]
        for idir in range(NDIR):
            ls = flat_node_surface[(l * NDIR + idir) * LR + 1]
            cc[ige * NEWSBT + idir * LR + 1] = (-dt(ige, ls) - dh(ige, ls)) * area[l][idir]
            diag[diagonal] += (dt(ige, ls) - dh(ige, ls)) * area[l][idir]
    return diag, cc


for node in range(NXYZ):
    legacy_diag, legacy_cc = legacy_setls(node)
    cached_diag, cached_cc = cached_setls(node)
    if bits(legacy_diag) != bits(cached_diag) or bits(legacy_cc) != bits(cached_cc):
        raise SystemExit(f"cmfd geometry cache: FAIL arithmetic mismatch at node {node}")

print("cmfd geometry cache: PASS")
