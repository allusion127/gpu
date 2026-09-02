#!/usr/bin/env python3
"""Contract gate for WP21-B: the micx/lmpx block is node-innermost (SoA).

WHAT WP21-B FOUND.  238 (RTX PRO 6000 Blackwell, 188 SMs), v6 single deck, ncu
`--launch-skip 10 --launch-count 5` per kernel (pricing block 39):
`kernelFlatXsCta` stores at **25.2 sectors per request** against an ideal of 2
for 8-byte accesses -- the worst store in the tree and the only kernel that
leans on bandwidth at all (dram 23.1 %, occupancy 62.4 %).  The campaign brief
attributed that to a node-major block and asked for a permutation to SoA.

THE BLOCK WAS ALREADY SoA.  Component c of node l has lived at `c*nxyz + l`
since the arm was written -- FlatXsView's own indexing note, XSSet's `micx()` /
`micxssm()` accessors, and CudaCramBackend's D2D fill all agree on it.  So the
25.2 is NOT a layout defect and no permutation removes it:

    kernelFlatXsCta is CTA-PER-NODE.  blockIdx.x is the node, threadIdx.x is
    the component ordinal q, and under `c*nxyz + l` consecutive lanes of a warp
    write addresses nxyz*elem_bytes APART -- one sector each, 32 sectors per
    request.  The thread-per-node twin (kernelFlatXs) is perfectly coalesced on
    the same bytes, and so is every other device consumer, because they all
    walk NODES.

    A layout is a transpose and a transpose is zero-sum between two axes: any
    order that coalesces the CTA arm's component walk un-coalesces the node
    walk of kernelFlatXs, the CRAM D2D fill, the Xe commit and the host
    accessors.  See docs/WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md §2.

So WP21-B does not permute anything.  It gives the convention a NAME
(`rasbery::flatxs::block_layout`), states it in a receipt, and pins it here --
because the live risk is not the sector count, it is a future reader "fixing"
the CTA store by flipping the block and silently breaking eleven consumers, two
of which live in files this campaign may not edit.

WHAT THIS GATE PINS

  1. THE LAYOUT HAS ONE DEFINITION.  `rasbery::flatxs::block_layout` in
     src/FlatXsKernel.h owns kNodeInnermost, kLayoutVersion, name() and the
     four index helpers (lmp/lsm/mic/msm).  It is a COMPILE-TIME constant, so
     one process cannot hold two orders.

  2. THE HELPERS ARE THE OLD EXPRESSION.  The shipping branch must be
     `c * nxyz + l` character for character -- WP21-B is a naming change and a
     helper that quietly reassociated the address would make it something else.

  3. NO FLAT-XS BODY SPELLS THE INDEX INLINE.  Both bodies -- flatxsSolveNode
     (src/FlatXsKernel.h) and flatxsSolveNodeCta (src/FlatXsCtaKernel.cuh) --
     reach the four blocks through block_layout, never through a bare
     `e * nxyz + l`, and never through a node-major `l * NMIC + e`.

  4. THE EIGHT WIDTH ACCESSORS STILL OWN THE BLOCK.  fxsRefLmp/Lsm/Mic/Msm and
     fxsStoreLmp/Lsm/Mic/Msm remain the only spellings of v.ref_mic / v.mic /
     v.lmp / v.lsm / v.msm inside a kernel (WP20.1's rule), so the layout
     helper and the width flag compose at one place instead of eight.

  5. THE MACROSCOPIC ARRAYS ARE NOT IN THE BLOCK.  xs / xs_ssm / iden are the
     FP64 authority the nodal drive, the CMFD operator and the host read.  They
     are already `ig*nxyz + l` and they are DELIBERATELY not routed through
     block_layout: they are not part of the four blocks and a helper that
     covered them would invite someone to permute them with the blocks.

  6. THE HOST ACCESSORS AGREE.  XSSet::micx / refMicx / micxssm compute
     `(iso*ng + ig)*nxyz + l` and `(iso*ng*ng + igs*ng + ige)*nxyz + l`.  They
     are the same order the device writes, which is what makes
     EnsureMicxHost's download a straight copy and FillCramMicDevice's handoff
     an ADDRESS rather than a transpose.

  7. THE RECEIPT SAYS WHICH ORDER RAN.  [RASBERY][MICX][LAYOUT] carries
     "layout", "layout_version" and "elem_bytes".

NEGATIVE CONTROLS.  Every rule is also run against a synthetic snippet that
violates it, so a rule that has quietly stopped matching anything fails here
rather than passing for ever.

Pure python, no build, no device.

Run:  python tools/test_micx_layout_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = os.path.join("src", "FlatXsKernel.h")
CTA = os.path.join("src", "FlatXsCtaKernel.cuh")
BACKEND = os.path.join("src", "CudaXsReconBackend.cu")
XSSET_H = os.path.join("src", "XSSet.h")
DOC = os.path.join("docs", "WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md")

# The four block classes and the helper each one must be addressed through.
BLOCK_HELPERS = ("lmp", "lsm", "mic", "msm")

# WP20.1's eight width accessors.  Each takes a block-layout index and nothing
# else may name the block pointers inside a kernel.
ACCESSORS = (
    "fxsRefLmp", "fxsRefLsm", "fxsRefMic", "fxsRefMsm",
    "fxsStoreLmp", "fxsStoreLsm", "fxsStoreMic", "fxsStoreMsm",
)

# The bodies that touch the four blocks.
BLOCK_BODIES = (
    (LAYOUT, "inline void flatxsSolveNode(const FlatXsView& v, int i,"),
    (CTA, "__device__ inline void flatxsSolveNodeCta(const FlatXsView& v, int i,"),
)

# Node-major spellings.  `l * NMIC`, `l * 78`, `l * 880` and friends are what a
# well-meaning "fix the CTA store" patch would introduce.
AOS_PATTERNS = (
    r"\bl\s*\*\s*NMIC\b",
    r"\bl\s*\*\s*NMSM\b",
    r"\bl\s*\*\s*NLSM\b",
    r"\bNMIC\s*\*\s*l\b",
    r"\bNMSM\s*\*\s*l\b",
    r"\b880\s*\*\s*l\b",
    r"\bl\s*\*\s*880\b",
)

RECEIPT_FIELDS = ("layout", "layout_version", "elem_bytes")


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def body_of(code: str, signature: str) -> str:
    """The braced body that follows `signature`, by brace matching."""
    start = code.find(signature)
    if start < 0:
        return ""
    open_brace = code.find("{", start)
    if open_brace < 0:
        return ""
    depth = 0
    for i in range(open_brace, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_brace:i + 1]
    return ""


# ---------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------


def rule_single_definition(layout: str) -> list:
    """One namespace owns the order, and it is a compile-time constant."""
    problems = []
    if "namespace block_layout {" not in layout:
        problems.append(
            "rasbery::flatxs::block_layout is missing from src/FlatXsKernel.h: "
            "the micx/lmpx storage order must have ONE definition, shared by "
            "both kernel bodies and quoted by the receipt")
        return problems
    if "constexpr bool kNodeInnermost" not in layout:
        problems.append(
            "block_layout::kNodeInnermost is not a compile-time constant; a "
            "runtime layout switch puts a branch on every block access and "
            "lets one process hold two orders at once")
    if "constexpr int  kLayoutVersion" not in layout \
            and "constexpr int kLayoutVersion" not in layout:
        problems.append("block_layout::kLayoutVersion is missing (the receipt "
                        "quotes it, so it may not be a literal at the receipt)")
    if not re.search(r"kNodeInnermost\s*=\s*true", layout):
        problems.append(
            "block_layout::kNodeInnermost is not true: the shipping order is "
            "node-innermost, and every host accessor, the CRAM D2D fill and "
            "the D2H download assume it")
    for helper in BLOCK_HELPERS:
        if not re.search(r"\bint\s+%s\s*\(int nxyz, int l, int" % helper, layout):
            problems.append("block_layout::%s() is missing: each of the four "
                            "block classes needs its own named helper so a "
                            "call site cannot pass the wrong component count"
                            % helper)
    return problems


def rule_helper_is_the_old_expression(layout: str) -> list:
    """WP21-B is a naming change; the address arithmetic may not move."""
    problems = []
    elem = body_of(layout, "int elem(int nxyz, int l, int c, int per_node)")
    if not elem:
        problems.append("block_layout::elem() is missing")
        return problems
    elem = strip_comments(elem)
    if not re.search(r"kNodeInnermost\s*\?\s*c\s*\*\s*nxyz\s*\+\s*l", elem):
        problems.append(
            "block_layout::elem()'s node-innermost branch is not `c * nxyz + l`: "
            "WP21-B claims byte identity with the inline spelling it replaced, "
            "and any other association is a different claim")
    if not re.search(r":\s*l\s*\*\s*per_node\s*\+\s*c", elem):
        problems.append(
            "block_layout::elem() has lost its AoS branch: it is kept as live, "
            "reachable text so a layout question stays bisectable (same reason "
            "cmfd_layout keeps its own)")
    if "long long" in elem or "size_t" in elem:
        problems.append(
            "block_layout::elem() no longer returns `int`: the eight width "
            "accessors take an int and every call site passed one, so widening "
            "here changes the addressing instruction sequence that was mined")
    return problems


def rule_bodies_use_the_helper(sources: dict) -> list:
    """Neither flat-XS body may spell a block index inline."""
    problems = []
    for rel, signature in BLOCK_BODIES:
        code = sources.get(rel, "")
        body = strip_comments(body_of(code, signature))
        if not body:
            problems.append("%s: the body `%s` is gone or was renamed; this "
                            "gate cannot see what it does not find"
                            % (rel, signature.split("(")[0]))
            continue
        for accessor in ACCESSORS:
            for call in re.finditer(re.escape(accessor) + r"\s*\(", body):
                tail = body[call.end():call.end() + 200]
                if "block_layout::" not in tail.split(";")[0]:
                    problems.append(
                        "%s: %s() is called without a block_layout:: index -- "
                        "an inline `e * nxyz + l` is the spelling WP21-B "
                        "replaced and the one a permutation would miss"
                        % (rel, accessor))
                    break
        for pattern in AOS_PATTERNS:
            if re.search(pattern, body):
                problems.append(
                    "%s: node-major spelling /%s/ is back in the flat-XS body. "
                    "The block is shared BY ADDRESS with CudaCramBackend's D2D "
                    "fill and aliased by the eleven host _micx vectors; a "
                    "node-major producer against SoA consumers is a silent "
                    "wrong answer, not a compile error" % (rel, pattern))
    return problems


def rule_accessors_own_the_block(layout: str, cta: str) -> list:
    """WP20.1's rule: the eight accessors are the only spelling of the block."""
    problems = []
    for accessor in ACCESSORS:
        if accessor not in layout:
            problems.append("src/FlatXsKernel.h: %s is gone; the block width "
                            "and the block layout compose at these eight sites "
                            "and nowhere else" % accessor)
    for rel, code, signature in ((LAYOUT, layout, BLOCK_BODIES[0][1]),
                                 (CTA, cta, BLOCK_BODIES[1][1])):
        body = strip_comments(body_of(code, signature))
        if not body:
            continue
        for field in ("v.ref_mic[", "v.ref_msm[", "v.ref_lmp[", "v.ref_lsm[",
                      "v.mic[", "v.msm[", "v.lmp[", "v.lsm[",
                      "v.mic_f[", "v.msm_f[", "v.lmp_f[", "v.lsm_f["):
            if field in body:
                problems.append(
                    "%s: `%s` is dereferenced outside the eight accessors -- a "
                    "ninth spelling reads the wide block on the narrow arm "
                    "(WP20.1) and skips the layout helper (WP21-B)"
                    % (rel, field))
    return problems


def rule_macroscopic_not_in_the_block(layout: str, cta: str) -> list:
    """xs / xs_ssm / iden are the FP64 authority and stay outside block_layout."""
    problems = []
    for rel, code in ((LAYOUT, layout), (CTA, cta)):
        stripped = strip_comments(code)
        for name in ("v.xs[", "v.xs_ssm[", "v.iden["):
            for hit in re.finditer(re.escape(name), stripped):
                index = stripped[hit.end():hit.end() + 120]
                index = index.split("]")[0] if name.endswith("_ssm[") or name.endswith("iden[") \
                    else index
                if "block_layout::" in index:
                    problems.append(
                        "%s: %s is indexed through block_layout. xs / xs_ssm / "
                        "iden are the macroscopic FP64 authority shared with "
                        "the nodal drive and the CMFD operator; they are not "
                        "part of the four blocks and must not be permuted with "
                        "them" % (rel, name))
                    break
    return problems


def rule_host_accessors_agree(xsset: str) -> list:
    """XSSet's value readers compute the same order the device writes."""
    problems = []
    micx = body_of(xsset, "double micx(Chiffon::XSTYPE xt, size_t iso, int ig, int l) const")
    ssm = body_of(xsset, "double micxssm(size_t iso, int igs, int ige, int l) const")
    if not micx or not ssm:
        problems.append("src/XSSet.h: micx()/micxssm() are gone; they are the "
                        "host half of this layout and the reason the D2H "
                        "download is a copy and not a transpose")
        return problems
    for name, body in (("micx", micx), ("micxssm", ssm)):
        flat = re.sub(r"\s+", "", strip_comments(body))
        if "*static_cast<size_t>(_g.nxyz())+static_cast<size_t>(l)" not in flat:
            problems.append(
                "src/XSSet.h: %s() no longer ends in `* nxyz + l`.  The host "
                "arrays ARE the download destination and the upload source; a "
                "host order that disagrees with the device order corrupts "
                "every cross section without raising anything" % name)
    return problems


def rule_receipt(backend: str) -> list:
    problems = []
    # Comments stripped first: the tag is discussed in prose right above the
    # struct, and a rule that matched the prose would pass on a deleted line.
    backend = strip_comments(backend)
    if "[RASBERY][MICX][LAYOUT]" not in backend:
        problems.append(
            "src/CudaXsReconBackend.cu: the [RASBERY][MICX][LAYOUT] receipt is "
            "missing.  A profile, a digest or a census read against the wrong "
            "block order is the failure this line exists to prevent")
        return problems
    line_at = backend.find("[RASBERY][MICX][LAYOUT]")
    window = backend[line_at:line_at + 900]
    for field in RECEIPT_FIELDS:
        if '\\"%s\\"' % field not in window:
            problems.append("[RASBERY][MICX][LAYOUT] does not carry \"%s\"" % field)
    if "block_layout::name()" not in window:
        problems.append(
            "[RASBERY][MICX][LAYOUT] quotes a literal layout name instead of "
            "block_layout::name(); a receipt that cannot disagree with the "
            "code it describes is decoration")
    if "block_layout::kLayoutVersion" not in window:
        problems.append(
            "[RASBERY][MICX][LAYOUT] quotes a literal layout_version instead of "
            "block_layout::kLayoutVersion")
    return problems


def rule_doc(doc: str) -> list:
    problems = []
    for needle in ("25.2", "kernelFlatXsCta", "block_layout", "kNodalJnet"):
        if needle not in doc:
            problems.append("docs/WP21_BC_...: `%s` is not discussed; the "
                            "inventory and the residual are the deliverable"
                            % needle)
    return problems


# ---------------------------------------------------------------------------
# Negative controls.
# ---------------------------------------------------------------------------

CONTROLS = (
    ("single definition",
     lambda s: rule_single_definition(s),
     "namespace block_layout {\n"
     "constexpr bool kNodeInnermost = false;\n"
     "}\n"),
    ("helper is the old expression",
     lambda s: rule_helper_is_the_old_expression(s),
     "constexpr long long elem(int nxyz, int l, int c, int per_node) {\n"
     "    return kNodeInnermost ? nxyz * c + l : l * per_node + c;\n"
     "}\n"),
    ("bodies use the helper",
     lambda s: rule_bodies_use_the_helper(
         {LAYOUT: "inline void flatxsSolveNode(const FlatXsView& v, int i,\n"
                  "                            const POL& pol) {\n"
                  "    bm[t * NMIC + e] = fxsRefMic(v, t, e * nxyz + l);\n"
                  "}\n",
          CTA: ""}),
     None),
    ("no node-major spelling",
     lambda s: rule_bodies_use_the_helper(
         {LAYOUT: "inline void flatxsSolveNode(const FlatXsView& v, int i,\n"
                  "                            const POL& pol) {\n"
                  "    fxsStoreMic(v, t, block_layout::mic(nxyz, l, e), x);\n"
                  "    double q = raw[l * NMIC + e];\n"
                  "}\n",
          CTA: ""}),
     None),
    ("accessors own the block",
     lambda s: rule_accessors_own_the_block(
         "inline void flatxsSolveNode(const FlatXsView& v, int i,\n"
         "                            const POL& pol) {\n"
         "    double x = v.mic[t][block_layout::mic(nxyz, l, e)];\n"
         "}\n"
         "fxsRefLmp fxsRefLsm fxsRefMic fxsRefMsm "
         "fxsStoreLmp fxsStoreLsm fxsStoreMic fxsStoreMsm\n", ""),
     None),
    ("macroscopic not in the block",
     lambda s: rule_macroscopic_not_in_the_block(
         "v.xs_ssm[block_layout::lsm(nxyz, l, q)] = val;\n", ""),
     None),
    ("host accessors agree",
     lambda s: rule_host_accessors_agree(s),
     "double micx(Chiffon::XSTYPE xt, size_t iso, int ig, int l) const {\n"
     "    return _micx[xt][l * 78 + iso * 2 + ig];\n"
     "}\n"
     "double micxssm(size_t iso, int igs, int ige, int l) const {\n"
     "    return _micx.xssm[l * 156 + iso * 4 + igs * 2 + ige];\n"
     "}\n"),
    ("receipt",
     lambda s: rule_receipt(s),
     'std::cout << "[RASBERY][MICX][LAYOUT] {\\"layout\\":\\"soa\\"}";\n'),
    ("doc",
     lambda s: rule_doc(s),
     "# WP21-B\n\nnothing to see here\n"),
)


def self_test() -> list:
    failures = []
    for label, rule, snippet in CONTROLS:
        if not rule(snippet):
            failures.append("negative control did not fire: %s" % label)
    return failures


def main() -> int:
    layout = read(LAYOUT)
    cta = read(CTA)
    backend = read(BACKEND)
    xsset = read(XSSET_H)
    try:
        doc = read(DOC)
    except OSError:
        doc = ""

    problems = []
    problems += rule_single_definition(layout)
    problems += rule_helper_is_the_old_expression(layout)
    problems += rule_bodies_use_the_helper({LAYOUT: layout, CTA: cta})
    problems += rule_accessors_own_the_block(layout, cta)
    problems += rule_macroscopic_not_in_the_block(layout, cta)
    problems += rule_host_accessors_agree(xsset)
    problems += rule_receipt(backend)
    if not doc:
        problems.append(
            "docs/WP21_BC_FLATXS_NODAL_COALESCING_20260831_KO.md is missing: "
            "the inventory, the zero-sum argument and the 238 runbook are the "
            "deliverable, not a nicety")
    else:
        problems += rule_doc(doc)

    controls = self_test()
    if problems or controls:
        print("FAIL: micx/lmpx block layout contract")
        for problem in problems:
            print("  - " + problem)
        for control in controls:
            print("  - " + control)
        return 1

    version = re.search(r"kLayoutVersion = kNodeInnermost \? (\d+)", layout)
    print("PASS: micx/lmpx block layout contract")
    print("  layout: flatxs::block_layout::kNodeInnermost -> version %s"
          % (version.group(1) if version else "?"))
    print("  four blocks pinned SoA: lmp[NG] lsm[NLSM] mic[NMIC] msm[NMSM], "
          "component c of node l at c*nxyz + l")
    print("  NOT in the block (FP64 macroscopic authority): xs, xs_ssm, iden")
    print("  width accessors held: %d" % len(ACCESSORS))
    print("  residual, by construction: kernelFlatXsCta is CTA-per-node, so its "
          "stores stay uncoalesced under ANY order that coalesces the "
          "node-walking consumers -- see the doc's zero-sum argument")
    print("  negative controls: %d, all fired" % len(CONTROLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
