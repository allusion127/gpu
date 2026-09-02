#!/usr/bin/env python3
"""Contract gate for WP21-C: the nodal storage orders, and the one that is shared.

WHAT WP21-C MEASURED.  238 (RTX PRO 6000 Blackwell, 188 SMs), v6 single deck,
ncu `--launch-skip 10 --launch-count 5` per kernel (pricing block 39):
`kNodalJnet<0>` loads at **16.7 sectors per request** against an ideal of 2 for
8-byte accesses -- the worst load in the tree (st 7.7).

WHERE IT COMES FROM.  The kernel runs one thread per SURFACE and reaches its
node ids through `lklr[ls*NLR + side]`.  Almost every load it then issues is
based on `lkd = lk*NDIR + idir`, and the arrays behind it are NODE-MAJOR: the
nine updateConstant products and six of the working arrays at
`[(lk*NDIR + idir)*NG + ig]` (6 doubles per node, 48 B between lanes), mu/tau
at `[(lk*NDIR + idir)*NG2 + j*NG + i]` (12 per node, 96 B), the four per-node
matrices at `[lk*NG2 + j*NG + i]` (32 B), hmesh at `[lk*NDIR + dir]` (24 B).
Node-innermost would put those at 8 B and the model at ~2-3.

WHY THE PERMUTATION IS NOT IN THIS COMMIT, and it is not a shrug.  Every one of
those index sites lives in src/NodalKernel.h, which is the SHARED host/device
body -- src/Nodal.cpp runs the same functions over its own host arrays as a
production arm, so a compile-time flip makes the host arm read its own arrays
through the wrong map.  WP21-A did not hit this because CMFD's CPU reference
solver is a different function in a different file, so only the device side
needed pack lanes.  The safe shape here is strides carried in NodalViewT with
the CURRENT values as struct defaults (so Nodal::MakeView, the replay tools and
test/canonical_state.cpp do not change a character) -- plus pack lanes for the
nine constants' H2D and for the hybrid arm's trlcff0/trlcff2/matM D2H.  That is
a NodalKernel.h + Nodal.cpp change, and doing it blind, with no compiler and no
device, is the exact recipe for a silent index error.  See
docs/WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md §4.

SO WHAT THIS GATE IS FOR.  Two things, and the first is the one that has never
been held anywhere:

  THE CANONICAL HANDOFF IS A TWO-SIDED INVARIANT AND ONLY ONE SIDE WAS WRITTEN
  DOWN.  In canonical mode flux/jnet/phis are not the nodal drive's buffers,
  they ARE the CMFD backend's (src/GpuCanonicalState.h: "the same bytes both
  backends already index, not a transposed copy that would need a conversion
  kernel on every handover").  WP21-A moved cc/diag/udiag/neighbors/
  node_surface/face_area to SoA and DELIBERATELY left phi, src and the Krylov
  state at the element index [2l+ig] -- and the reason is recorded only in
  WP21-A's own document, with nothing on the nodal side enforcing it.  A later
  campaign permuting phi would leave the nodal kernels reading the same bytes
  as `flux[lk*NG + ig]`.  Finite, plausible, wrong, silent.  This gate holds
  both sides of that handoff in one place.

  THE RESIDUAL IS ATTRIBUTED, NOT FORGOTTEN.  [RASBERY][NODAL][GPU] prints
  `private_layout`, and this gate makes that string and the actual index
  spelling in NodalKernel.h agree.  Convert the arrays and the receipt fails
  here until it is updated; change the receipt without converting and it fails
  the same way.

NEGATIVE CONTROLS.  Every rule is also run against a synthetic snippet that
violates it, so a rule that has quietly stopped matching anything fails here
rather than passing for ever.

Pure python, no build, no device.

Run:  python tools/test_nodal_soa_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODAL = os.path.join("src", "NodalKernel.h")
CANON = os.path.join("src", "GpuCanonicalState.h")
BICG = os.path.join("src", "CudaBICGBackend.cu")
BACKEND = os.path.join("src", "CudaXsReconBackend.cu")
DOC = os.path.join("docs", "WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md")

# The three canonical regions and the index each side must keep.  `flux` is
# reached through lk / lkl / lkr; jnet and phis only ever through ls.
CANONICAL_FORMS = (
    ("flux", r"v\.flux\[\s*lk[lr]?\s*\*\s*NG\s*\+"),
    ("jnet", r"v\.jnet\[\s*ls\s*\*\s*NG\s*\+"),
    ("phis", r"v\.phis\[\s*ls\s*\*\s*NG\s*\+"),
)

# No layout helper may be applied to a canonical pointer on EITHER side: the
# handoff is a pointer swap and a helper is how a transpose gets introduced.
LAYOUT_HELPERS = ("cmfd_layout::", "block_layout::", "nodal_layout::")

# The node-major residual, by array class.  (label, regex, minimum sites)
PRIVATE_FORMS = (
    ("per-(node,dir,group) constants and working arrays",
     r"v\.(eta1|eta2|diagD|diagDI|m260|m251|m253|m262|m264|"
     r"trlcff0|trlcff1|trlcff2|dsncff2|dsncff4|dsncff6)\[\s*lkd\w*\s*\*\s*NG\s*\+",
     40),
    ("per-node matrices (matM/matMI/matMs/matMf)",
     r"v\.(matM|matMI|matMs|matMf)\[\s*lk\w*\s*\*\s*NG2\s*\+", 8),
    ("per-(node,dir) matrices (mu/tau)",
     r"v\.(mu|tau)\[\s*lkd\w*\s*\*\s*NG2\s*\+", 4),
    ("hmesh", r"v\.hmesh\[\s*lk\w*\s*\*\s*NDIR\s*\+", 4),
)

RECEIPT_FIELDS = ("canonical_layout", "private_layout")


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ---------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------


def rule_canonical_index(nodal: str) -> list:
    """The nodal side of the handoff keeps the element index."""
    problems = []
    code = strip_comments(nodal)
    for name, pattern in CANONICAL_FORMS:
        if not re.search(pattern, code):
            problems.append(
                "src/NodalKernel.h: `%s` is no longer indexed as the element "
                "index the CMFD backend uses.  In canonical mode this pointer "
                "IS the CMFD buffer; a nodal-side permutation makes the two "
                "backends read the same bytes through two maps" % name)
        for helper in LAYOUT_HELPERS:
            if re.search(r"v\.%s\[\s*[A-Za-z_:]*%s" % (name, re.escape(helper)), code):
                problems.append(
                    "src/NodalKernel.h: `%s` is indexed through %s.  The three "
                    "canonical regions are a pointer swap, not a layout -- a "
                    "helper here is how a transpose gets introduced by accident"
                    % (name, helper))
    return problems


def rule_canonical_note(canon: str) -> list:
    """GpuCanonicalState.h states the shared layout; it may not go quiet."""
    problems = []
    flat = re.sub(r"\s+", " ", canon)
    if "[ls*ng + ig]" not in flat or "[l*ng + ig]" not in flat:
        problems.append(
            "src/GpuCanonicalState.h: canonicalFromSlotView's layout note no "
            "longer spells `dtil/dhat/jnet are [ls*ng + ig] and flux is "
            "[l*ng + ig]`.  That sentence is the whole reason a canonical "
            "buffer needs no conversion kernel on handover")
    if "not a transposed copy" not in flat:
        problems.append(
            "src/GpuCanonicalState.h: the 'not a transposed copy' claim is "
            "gone from canonicalFromSlotView; if a transpose HAS been "
            "introduced the sharing is no longer a pointer swap and every "
            "handover costs a kernel")
    return problems


def rule_cmfd_side_untouched(bicg: str) -> list:
    """WP21-A's exclusion: phi / dtil / dhat keep the element index."""
    problems = []
    code = strip_comments(bicg)
    if len(re.findall(r"2\s*\*\s*l\s*\+", code)) < 20:
        problems.append(
            "src/CudaBICGBackend.cu: the `2*l + ig` element index has largely "
            "disappeared.  WP21-A left phi, src and the Krylov state at it on "
            "purpose -- they are reduction operands (N1 if permuted) AND the "
            "nodal/PPR pointer handoff (WP21-C's invariant)")
    for name in ("phi", "dtil", "dhat"):
        for helper in LAYOUT_HELPERS:
            if re.search(r"\b%s\[\s*[A-Za-z_:]*%s" % (name, re.escape(helper)), code):
                problems.append(
                    "src/CudaBICGBackend.cu: `%s` is indexed through %s.  It is "
                    "handed to the nodal drive as flux/jnet/phis and to PPR and "
                    "the outer segment through CmfdResidentView; permuting it "
                    "here breaks readers in files WP21-A never opened"
                    % (name, helper))
    return problems


def rule_private_residual(nodal: str) -> list:
    """The node-major residual is where the receipt and the doc say it is."""
    problems = []
    code = strip_comments(nodal)
    for label, pattern, minimum in PRIVATE_FORMS:
        found = len(re.findall(pattern, code))
        if found < minimum:
            problems.append(
                "src/NodalKernel.h: the node-major spelling for %s has "
                "%d sites, fewer than the %d WP21-C inventoried.  Either the "
                "arrays were converted -- in which case "
                "kNodalPrivateLayout in CudaXsReconBackend.cu and §4 of the "
                "WP21-BC doc are now lying about a 16.7-sector residual -- or "
                "this gate has gone blind" % (label, found, minimum))
    return problems


def rule_receipt(backend: str) -> list:
    problems = []
    code = strip_comments(backend)
    if "kNodalCanonicalLayout" not in code or "kNodalPrivateLayout" not in code:
        problems.append(
            "src/CudaXsReconBackend.cu: kNodalCanonicalLayout / "
            "kNodalPrivateLayout are missing; the nodal receipt must state "
            "which storage order a run used, or a profile read against the "
            "wrong body has nothing to contradict it")
        return problems
    if not re.search(r'kNodalCanonicalLayout\s*=\s*"element"', code):
        problems.append(
            'kNodalCanonicalLayout is not "element".  Any other value means a '
            "conversion kernel now sits on the canonical handoff, and that is "
            "a change to CudaBICGBackend.cu's contract as much as to this one")
    if not re.search(r'kNodalPrivateLayout\s*=\s*"(node_major|soa)"', code):
        problems.append(
            'kNodalPrivateLayout must be "node_major" (the inventoried '
            'residual) or "soa" (converted); anything else is not a state this '
            "tree knows how to be in")
    receipt = code[code.find("[RASBERY][NODAL][GPU]"):]
    receipt = receipt[:1800]
    for field in RECEIPT_FIELDS:
        if '\\"%s\\"' % field not in receipt:
            problems.append("[RASBERY][NODAL][GPU] does not carry \"%s\"" % field)
    for symbol in ("kNodalCanonicalLayout", "kNodalPrivateLayout"):
        if symbol not in receipt:
            problems.append(
                "[RASBERY][NODAL][GPU] quotes a literal instead of %s; a "
                "receipt that cannot disagree with the code it describes is "
                "decoration" % symbol)
    return problems


def rule_doc(doc: str) -> list:
    problems = []
    for needle in ("16.7", "kNodalJnet", "NodalKernel.h", "Nodal.cpp",
                   "canonical_layout", "private_layout"):
        if needle not in doc:
            problems.append(
                "docs/WP21_BC_...: `%s` is not discussed.  The inventory and "
                "the reason the permutation was not landed ARE the WP21-C "
                "deliverable; without them the residual is unattributed"
                % needle)
    return problems


# ---------------------------------------------------------------------------
# Negative controls.
# ---------------------------------------------------------------------------

CONTROLS = (
    ("canonical index",
     lambda s: rule_canonical_index(s),
     "v.jnet[nodal_layout::surf(nsurf, ls, ig)] = x;\n"
     "v.phis[ls * NG + ig] = y;\n"
     "double f = v.flux[lk * NG + ig];\n"),
    ("canonical note",
     lambda s: rule_canonical_note(s),
     "inline CanonicalSlotBuffers canonicalFromSlotView(const DeviceSlotView& v) {\n"
     "    return CanonicalSlotBuffers{};\n"
     "}\n"),
    ("cmfd side untouched",
     lambda s: rule_cmfd_side_untouched(s),
     "__global__ void matvec_two_group(int nxyz) {\n"
     "    double x0 = phi[cmfd_layout::mat(nxyz, l, 0)];\n"
     "}\n"),
    ("private residual",
     lambda s: rule_private_residual(s),
     "double a = v.eta1[(0 * NG + ig) * nxyz + lk];\n"),
    ("receipt",
     lambda s: rule_receipt(s),
     'std::cout << "[RASBERY][NODAL][GPU] {\\"canonical_layout\\":\\"element\\","\n'
     '             "\\"private_layout\\":\\"node_major\\"}";\n'),
    ("receipt canonical value",
     lambda s: rule_receipt(s),
     'constexpr const char* kNodalCanonicalLayout = "soa";\n'
     'constexpr const char* kNodalPrivateLayout   = "node_major";\n'
     'std::cout << "[RASBERY][NODAL][GPU] {\\"canonical_layout\\":\\""\n'
     '          << kNodalCanonicalLayout << "\\",\\"private_layout\\":\\""\n'
     '          << kNodalPrivateLayout << "\\"}";\n'),
    ("doc",
     lambda s: rule_doc(s),
     "# WP21-C\n\nnothing to see here\n"),
)


def self_test() -> list:
    failures = []
    for label, rule, snippet in CONTROLS:
        if not rule(snippet):
            failures.append("negative control did not fire: %s" % label)
    return failures


def main() -> int:
    nodal = read(NODAL)
    canon = read(CANON)
    bicg = read(BICG)
    backend = read(BACKEND)
    try:
        doc = read(DOC)
    except OSError:
        doc = ""

    problems = []
    problems += rule_canonical_index(nodal)
    problems += rule_canonical_note(canon)
    problems += rule_cmfd_side_untouched(bicg)
    problems += rule_private_residual(nodal)
    problems += rule_receipt(backend)
    if not doc:
        problems.append(
            "docs/WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md is missing: "
            "the nodal inventory and the attribution of the 16.7 residual are "
            "the deliverable, not a nicety")
    else:
        problems += rule_doc(doc)

    controls = self_test()
    if problems or controls:
        print("FAIL: nodal layout / canonical handoff contract")
        for problem in problems:
            print("  - " + problem)
        for control in controls:
            print("  - " + control)
        return 1

    code = strip_comments(nodal)
    print("PASS: nodal layout / canonical handoff contract")
    print("  canonical (two-sided with WP21-A, may not move): flux [lk*NG+ig], "
          "jnet/phis [ls*NG+ig] == CMFD phi/dtil/dhat [2l+ig] / [ls*ng+ig]")
    for label, pattern, _ in PRIVATE_FORMS:
        print("  residual (node-major, inventoried NOT converted): %s -- %d sites"
              % (label, len(re.findall(pattern, code))))
    print("  receipt: [RASBERY][NODAL][GPU] canonical_layout + private_layout, "
          "quoted from the constants")
    print("  negative controls: %d, all fired" % len(CONTROLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
