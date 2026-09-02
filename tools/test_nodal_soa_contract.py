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

WP21-C2 CONVERTED IT.  What follows describes why the conversion took the shape
it did, and what this gate now holds.  The paragraph below is kept verbatim
because it is the design constraint the conversion had to satisfy, not a
historical note.

WHY THE PERMUTATION WAS NOT IN WP21-C, and it was not a shrug.  Every one of
those index sites lives in src/NodalKernel.h, which is the SHARED host/device
body -- src/Nodal.cpp runs the same functions over its own host arrays as a
production arm, so a compile-time flip makes the host arm read its own arrays
through the wrong map.  WP21-A did not hit this because CMFD's CPU reference
solver is a different function in a different file, so only the device side
needed pack lanes.  The safe shape here is strides carried in NodalViewT with
the CURRENT values as struct defaults (so Nodal::MakeView, the replay tools and
test/canonical_state.cpp do not change a character) -- plus pack lanes for the
nine constants' H2D and for the hybrid arm's trlcff0/trlcff2/matM D2H.  That is
exactly the shape WP21-C2 built, and it is checked here AND by a gate that runs
the whole thing twice: test/nodal_layout_equivalence.cpp builds a mesh, runs all
five phases over node-major arrays and over the same values packed
node-innermost, and compares the outputs BIT FOR BIT -- with a sabotage arm that
swaps two strides and requires the comparison to FAIL.  That is the answer to
"doing it blind is the recipe for a silent index error": it is no longer blind.
See docs/WP21_B2C2_COALESCING_20260831_KO.md §4.

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
DOC2 = os.path.join("docs", "WP21_B2C2_COALESCING_20260831_KO.md")

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

# WP21-C2: THE CONVERTED CENSUS.  Every private-array index must go through the
# view's own accessor, and no inline node-major spelling may survive anywhere in
# the body.  The second half is the one that matters: a single site left at
# `[(lk*NDIR + idir)*NG + ig]` reads the node-major address out of a
# node-innermost array -- in range, finite, plausible, wrong, silent.
#
# (label, regex, minimum sites)
PRIVATE_FORMS = (
    ("per-(node,dir,group) constants and working arrays",
     r"v\.(eta1|eta2|diagD|diagDI|m260|m251|m253|m262|m264|"
     r"trlcff0|trlcff1|trlcff2|dsncff2|dsncff4|dsncff6)\[\s*v\.ndg\(", 100),
    ("per-node matrices (matM/matMI/matMs/matMf)",
     r"v\.(matM|matMI|matMs|matMf)\[\s*v\.ng2\(", 8),
    ("per-(node,dir) matrices (mu/tau)", r"v\.(mu|tau)\[\s*v\.dg2\(", 4),
    ("hmesh", r"v\.hmesh\[\s*v\.hm\(", 4),
)

# The spellings that must be GONE: any private array indexed by an inline
# node-major expression rather than by the view.
BANNED_FORMS = (
    ("per-(node,dir,group) inline",
     r"v\.(eta1|eta2|diagD|diagDI|m260|m251|m253|m262|m264|"
     r"trlcff0|trlcff1|trlcff2|dsncff2|dsncff4|dsncff6)\[[^\]]*\*\s*NG\s*\+"),
    ("per-node matrix inline", r"v\.(matM|matMI|matMs|matMf)\[[^\]]*\*\s*NG2\s*\+"),
    ("per-(node,dir) matrix inline", r"v\.(mu|tau)\[[^\]]*\*\s*NG2\s*\+"),
    ("hmesh inline", r"v\.hmesh\[[^\]]*\*\s*NDIR\s*\+"),
)

# The ten strides, and the values they must DEFAULT to.  The defaults are the
# node-major layout, which is what makes `NodalView v{}` -- how Nodal::MakeView,
# nodalDumpState, test/nodal_replay.cpp and test/canonical_state.cpp all build
# one -- byte-for-byte the view it was before WP21-C2.
STRIDE_DEFAULTS = (
    ("ndg_node", "NDIR * NG"), ("ndg_dir", "NG"), ("ndg_grp", "1"),
    ("dg2_node", "NDIR * NG2"), ("dg2_dir", "NG2"), ("dg2_elem", "1"),
    ("ng2_node", "NG2"), ("ng2_elem", "1"),
    ("hm_node", "NDIR"), ("hm_dir", "1"),
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


def rule_private_converted(nodal: str) -> list:
    """Every private-array index goes through the view, and none is inline."""
    problems = []
    code = strip_comments(nodal)
    for label, pattern, minimum in PRIVATE_FORMS:
        found = len(re.findall(pattern, code))
        if found < minimum:
            problems.append(
                "src/NodalKernel.h: %s has %d accessor sites, fewer than the %d "
                "WP21-C2 converted.  Either sites were deleted or one has gone "
                "back to an inline index" % (label, found, minimum))
    for label, pattern in BANNED_FORMS:
        hits = re.findall(pattern, code)
        if hits:
            problems.append(
                "src/NodalKernel.h: %d %s index site(s) survive.  Under the "
                "node-innermost layout an inline node-major index reads a "
                "different element -- in range, finite, plausible, wrong"
                % (len(hits), label))
    return problems


def rule_strides_default_to_node_major(nodal: str) -> list:
    """The struct's defaults ARE the host layout, and the accessors are the only
    place an index is composed."""
    problems = []
    for name, value in STRIDE_DEFAULTS:
        if not re.search(r"int\s+%s\s*=\s*%s\s*;" % (name, re.escape(value)),
                         nodal):
            problems.append(
                "src/NodalKernel.h: NodalViewT::%s does not default to `%s`.  "
                "Every host builder in the tree makes its view with `NodalView "
                "v{}` and expects the node-major layout; a changed default "
                "silently re-points Nodal.cpp's own arrays" % (name, value))
    for acc, body in (("ndg", "lk * ndg_node + idir * ndg_dir + ig * ndg_grp"),
                      ("dg2", "lk * dg2_node + idir * dg2_dir + e * dg2_elem"),
                      ("ng2", "lk * ng2_node + e * ng2_elem"),
                      ("hm", "lk * hm_node + idir * hm_dir")):
        if body not in nodal:
            problems.append(
                "src/NodalKernel.h: the `%s` accessor is not `%s`.  One "
                "composition per class, or the pack and the kernel can disagree "
                "about one array" % (acc, body))
    if "nodalNodeInnermostView" not in nodal:
        problems.append(
            "src/NodalKernel.h: nodalNodeInnermostView is gone; the device "
            "layout must have exactly one definition")
    # nodalWideShell enumerates the fields that are NOT narrowed.  A stride
    # left behind there gives the FP32 twin a different map for the same bytes.
    a = nodal.find("nodalWideShell")
    shell = nodal[a:] if a >= 0 else ""
    b = shell.find("return v;")
    shell = shell[:b] if b >= 0 else shell
    for name, _ in STRIDE_DEFAULTS:
        if "v.%s" % name not in shell:
            problems.append(
                "src/NodalKernel.h: nodalWideShell does not carry `%s`.  The "
                "layout is not narrowed -- the FP32 twin holds the same "
                "elements in the same order at half the width" % name)
    return problems


def rule_pack_pairs_with_layout(backend: str) -> list:
    """A view that says node-innermost over bytes that are still node-major is
    the failure this whole design exists to avoid.  The pack and the layout are
    ONE decision, so they are checked as one."""
    problems = []
    code = strip_comments(backend)
    if "kNodalPermute" not in code or "nodalPermuteLaunch" not in code:
        problems.append(
            "src/CudaXsReconBackend.cu: the permutation kernel is gone.  With "
            "no pack, a node-innermost view reads uploaded node-major bytes")
        return problems
    if code.count("ndl::nodalNodeInnermostView(") < 3:
        problems.append(
            "src/CudaXsReconBackend.cu: fewer than three views are switched to "
            "the node-innermost layout.  solveNodal, solveNodalPost (it "
            "finishes the SAME drive over the SAME arrays) and the arena's "
            "slot-0 base all need it")
    # The strides a pack uses must come from the same two helpers the views use.
    if "nodalAosStrides" not in code or "nodalSoaStrides" not in code:
        problems.append(
            "src/CudaXsReconBackend.cu: the packs no longer take their strides "
            "from nodalAosStrides / nodalSoaStrides.  A second spelling is how "
            "a pack and a kernel end up disagreeing about one array")
    # Every permuted transfer keeps its byte count: the copies are unchanged and
    # the reorder happens on the device between them.
    for tag in ('"hmesh"', '"consts"', '"trlcff0"', '"trlcff2"', '"matM"'):
        if tag not in code:
            problems.append(
                "src/CudaXsReconBackend.cu: the %s transfer is gone; WP21-C2 "
                "moves no bytes, so every xfer:: site must survive" % tag)
    if "rasberyGpuNodalSoaEnabled" not in code:
        problems.append(
            "src/CudaXsReconBackend.cu: the layout has no off switch, so a "
            "digest disagreement has no one-variable bisect")
    return problems


def rule_equivalence_gate_is_registered(cmake: str, test_src: str) -> list:
    """The claim is an equivalence between two RUNS, so a source gate cannot
    decide it -- but it can refuse to let the gate go missing."""
    problems = []
    if "rasbery_nodal_layout_equivalence" not in cmake:
        problems.append(
            "CMakeLists.txt: the layout-equivalence gate is not built.  It is "
            "the only thing that checks the 142 converted sites agree with the "
            "node-major ones, and it needs no GPU")
    if "nodal_layout_equivalence_negative" not in cmake \
            or "RASBERY_NODAL_LAYOUT_SABOTAGE=1" not in cmake:
        problems.append(
            "CMakeLists.txt: the sabotage arm is not registered.  A gate that "
            "compares two runs which agree for the wrong reason passes for ever")
    for needle in ("nodalNodeInnermostView", "sabotage", "sameBits"):
        if needle not in test_src:
            problems.append(
                "test/nodal_layout_equivalence.cpp no longer uses `%s`; the "
                "gate is not comparing what it claims" % needle)
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
    # WP21-C2 made the private layout an ARM, so the receipt must print the one
    # that ran rather than a constant.  Both spellings have to exist and the
    # selector has to be the only thing that turns the flag into a word.
    if not re.search(r'kNodalPrivateLayoutAos\s*=\s*"node_major"', code):
        problems.append(
            'kNodalPrivateLayoutAos is missing or is not "node_major".  '
            "RASBERY_GPU_NODAL_SOA=0 is the bisect arm and the receipt has to "
            "be able to say so")
    if "nodalPrivateLayoutName" not in code:
        problems.append(
            "there is no nodalPrivateLayoutName() selector: a run could print "
            "one layout and index another")
    receipt = code[code.find("[RASBERY][NODAL][GPU]"):]
    receipt = receipt[:1800]
    for field in RECEIPT_FIELDS:
        if '\\"%s\\"' % field not in receipt:
            problems.append("[RASBERY][NODAL][GPU] does not carry \"%s\"" % field)
    for symbol in ("kNodalCanonicalLayout", "nodalPrivateLayoutName"):
        if symbol not in receipt:
            problems.append(
                "[RASBERY][NODAL][GPU] quotes a literal instead of %s; a "
                "receipt that cannot disagree with the code it describes is "
                "decoration" % symbol)
    return problems


def rule_doc2(doc: str) -> list:
    """The WP21-C2 doc: the conversion, its gate and its runbook."""
    problems = []
    for needle in ("RASBERY_GPU_NODAL_SOA", "nodal_layout_equivalence",
                   "16.7", "NodalViewT", "nodalNodeInnermostView",
                   "hybrid", "XFER"):
        if needle not in doc:
            problems.append(
                "docs/WP21_B2C2_...: `%s` is not discussed.  The conversion is "
                "the deliverable and its gate is the argument" % needle)
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
    ("private converted -- an inline node-major site survives",
     lambda s: rule_private_converted(s),
     "double a = v.eta1[lkd * NG + ig];\n"),
    ("stride defaults moved off the host layout",
     lambda s: rule_strides_default_to_node_major(s),
     "int ndg_node = 1;\nint ndg_dir = NG;\nint ndg_grp = 1;\n"),
    ("a layout with no pack behind it",
     lambda s: rule_pack_pairs_with_layout(s),
     "v = ndl::nodalNodeInnermostView(v);\n"),
    ("the equivalence gate unregistered",
     lambda s: rule_equivalence_gate_is_registered(s, ""),
     "add_test(NAME canonical_state COMMAND rasbery_canonical_state)\n"),
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
    ("doc2",
     lambda s: rule_doc2(s),
     "# WP21-C2\n\nnothing to see here\n"),
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
    problems += rule_private_converted(nodal)
    problems += rule_strides_default_to_node_major(nodal)
    problems += rule_pack_pairs_with_layout(backend)
    problems += rule_receipt(backend)
    try:
        cmake = read("CMakeLists.txt")
    except OSError:
        cmake = ""
    try:
        eqsrc = read(os.path.join("test", "nodal_layout_equivalence.cpp"))
    except OSError:
        eqsrc = ""
    problems += rule_equivalence_gate_is_registered(cmake, eqsrc)
    try:
        doc2 = read(DOC2)
    except OSError:
        doc2 = ""
    if not doc2:
        problems.append(
            "docs/WP21_B2C2_COALESCING_20260831_KO.md is missing: the "
            "conversion, its equivalence gate and its 238 runbook are the "
            "WP21-C2 deliverable")
    else:
        problems += rule_doc2(doc2)
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
        print("  converted (node-innermost, through the view): %s -- %d sites"
              % (label, len(re.findall(pattern, code))))
    print("  receipt: [RASBERY][NODAL][GPU] canonical_layout + private_layout, "
          "quoted from the constants")
    print("  negative controls: %d, all fired" % len(CONTROLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
