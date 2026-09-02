#!/usr/bin/env python3
"""Contract gate for WP21-A: the CMFD/BiCG node-innermost (SoA) layout.

WHAT WP21-A IS.  238 (RTX PRO 6000 Blackwell, 188 SMs), v6 single deck, ncu
`--launch-skip 10 --launch-count 5` per kernel (pricing block 39): every CMFD
kernel is uncoalesced.  `colored_block_sweep` ld 6.55 / st 6.90 sectors per
request against an ideal of 2 for 8-byte accesses, `matvec_two_group` ld 9.92 /
st 6.38, `prepare_p_jacobi` 6.33 / 6.38, `reduce_dot_stage1` ld 6.55,
`reduce_dot2_fused` ld 7.05.  One thread owns one node, and every node-indexed
array was packed NODE-MAJOR -- diag[4*l+k], cc[12*l+j], neighbors[6*l+s],
face_area[3*l+d] -- so consecutive threads read addresses 32, 96 and 24 bytes
apart and each one lands in its own sector.

WP21-A stores the same values COMPONENT-MAJOR: component k of node l at
k*nxyz + l (rasbery::cmfd_layout).  Consecutive nodes become adjacent doubles.

WHY IT IS B0 (bit-identical), and what this gate exists to keep true:

  A pure permutation of STORAGE is B0 only while it stays a permutation of
  storage.  Two things could break that and neither is visible from a diff of
  one kernel:

    * a kernel left indexing the OLD way against a rewritten producer -- a
      silent wrong-answer, not a compile error, because both spellings are
      valid arithmetic on the same buffer;
    * a permuted REDUCTION operand.  The dots walk the element index i over
      [0, n) with a fixed `chunk = ceil(n/gridDim.x)` partition and a fixed
      256-lane tree.  Permuting a Krylov vector re-chunks that partition and
      moves additions, which is N1, not B0.  So the Krylov vectors, the flux
      and the source are DELIBERATELY LEFT node-major and this gate pins that.

WHAT THIS GATE PINS

  1. THE LAYOUT HAS ONE DEFINITION.  rasbery::cmfd_layout in
     src/CmfdAssemblyKernel.h owns kNodeInnermost, kLayoutVersion, layoutName()
     and the four index helpers (mat/cpl/face/dir).  Nothing computes a
     node-indexed address any other way.

  2. NO DEVICE KERNEL STILL SPELLS THE AoS INDEX.  `4 * l +`, `12 * l +`,
     `6 * l +` and `l * 4 +` are gone from every kernel body that touches the
     operator.  (They survive on the HOST, where the CPU reference solver and
     the greedy colouring read the node-major host arrays; that is the point of
     the pack lanes.)

  3. EVERY KERNEL USES THE HELPER ITS ARRAY CLASS CALLS FOR.  diag/dinv/udiag
     and their float mirrors go through cmfd_layout::mat, cc/cc_f through
     cmfd_layout::cpl, neighbors through cmfd_layout::face.

  4. PRODUCERS AND CONSUMERS AGREE.  The device assembly (assembleNode2G), the
     Wielandt rewrite (cmfd_updls), the FP32 mirror refresh, the one-time
     geometry uploads and the per-outer H2D/D2H all move through the same
     helpers or the pack lanes.  A producer that writes AoS into an array a
     consumer reads as SoA is exactly the silent failure above.

  5. THE TRANSFERS ARE THE SAME SIZE.  packMat/packCpl/unpackMat/unpackCpl are
     permutations in place of a copy: matrix_count and coupling_count doubles,
     the same xfer::memcpy site names, the same leaves.  The ledger must not
     move.

  6. THE REDUCTION OPERANDS ARE UNTOUCHED.  The Krylov vectors, phi and src
     keep the 2*l+ig element index; the reduce kernels keep `chunk`, the
     strided walk and the strict ascending fold.

  7. THE RECEIPT SAYS WHICH LAYOUT RAN.  [RASBERY][CMFD][GRAPH] carries
     "layout" and "layout_version".

  8. FUSE BIT 4 IS ARMED, OFF BY DEFAULT, AND KEEPS THE OVERRUN TALLY.
     reduce_norm_accumulate_fused folds the one-block/one-thread stage 2 into
     the last block of its stage 1.  A naive fusion would drop kOverrunCount
     for a halted slot; both fused kernels keep the halt branch and the
     `active` test, and kFuseNorm is NOT part of kFuseDefaultMask.

NEGATIVE CONTROLS.  Every rule is also run against a synthetic snippet that
violates it, so a rule that has quietly stopped matching anything fails here
rather than passing for ever.

Pure python, no build, no device.

Run:  python tools/test_cmfd_soa_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join("src", "CudaBICGBackend.cu")
LAYOUT = os.path.join("src", "CmfdAssemblyKernel.h")
DOC = os.path.join("docs", "WP21_A_CMFD_COALESCING_20260831_KO.md")

# Device kernels (and the two persistent-arm device helpers) that index at
# least one node-indexed operator array.  Every one of them must be free of the
# AoS spelling and must use the helpers.
OPERATOR_KERNELS = (
    "__global__ void matvec_two_group(",
    "__global__ void begin_outer_fused(",
    "__global__ void prepare_p_jacobi(",
    "__global__ void colored_block_sweep(",
    "__global__ void update_s_jacobi(",
    "__device__ inline void persistentColourSweepNode(",
    "__device__ inline void persistentMatvecNode(",
    "__global__ void bicg_iteration_persistent(",
    "__global__ void refresh_operator_mirror_f32(",
    "__global__ void begin_outer_fused_f32(",
    "__global__ void matvec_two_group_f32(",
    "__global__ void colored_block_sweep_f32(",
    "__global__ void prepare_p_jacobi_f32(",
    "__global__ void update_s_jacobi_f32(",
    "__global__ void cmfd_updls(",
)

# Which helper each array class must be addressed through.  The rule below does
# NOT trust pointer names -- `cm` is chif in cmfd_updls and the counter row in
# the persistent kernel -- it finds the local pointers actually BOUND to an
# operator array and checks only those.
MAT_ARRAYS = ("diag_f", "dinv_f", "udiag_dev", "diag", "dinv", "udiag",
              "a.diag", "a.dinv")
CPL_ARRAYS = ("cc_f", "cc", "a.cc")

# The AoS spellings.  `l * 4 +` is the cmfd_updls form; the rest are the
# multiply-first spellings every other kernel used.
AOS_PATTERNS = (
    r"\b4\s*\*\s*l\s*\+",
    r"\b12\s*\*\s*l\s*\+",
    r"\b6\s*\*\s*l\s*\+",
    r"\bl\s*\*\s*4\s*\+",
    r"\bl\s*\*\s*12\s*\+",
    r"\bl\s*\*\s*6\s*\+",
)

# The reduce family, whose operands may not be permuted.
REDUCE_KERNELS = (
    "__global__ void reduce_dot_stage1(",
    "__global__ void reduce_dot_fused(",
    "__global__ void reduce_dot2_stage1(",
    "__global__ void reduce_dot2_fused(",
    "__global__ void reduce_dot_stage1_f32(",
    "__global__ void reduce_dot2_stage1_f32(",
    "__global__ void reduce_norm_accumulate_fused(",
    "__global__ void reduce_norm_accumulate_fused_f32(",
)

# Kernels that carry the ELEMENT index and therefore still spell 2*l or a bare
# i: the flux, the source and the Krylov state are not permuted.
ELEMENT_KERNELS = (
    "__global__ void update_solution(",
    "__global__ void update_solution_f32(",
    "__global__ void cmfd_src_build(",
)

GRAPH_FIELDS = ("layout", "layout_version")


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
                return code[open_brace: i + 1]
    return ""


# ---------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------


def rule_single_definition(layout: str) -> list:
    """One namespace owns the layout, and it is a compile-time constant."""
    problems = []
    if "namespace rasbery::cmfd_layout {" not in layout:
        problems.append(
            "rasbery::cmfd_layout is missing from src/CmfdAssemblyKernel.h: the "
            "layout must have ONE definition shared by the device kernels, the "
            "host pack lanes and the receipt")
        return problems
    if "constexpr bool kNodeInnermost" not in layout:
        problems.append(
            "cmfd_layout::kNodeInnermost must be a `constexpr bool`: a runtime "
            "switch would put a branch on every operator access and would let one "
            "process hold two layouts at once")
    if "constexpr int  kLayoutVersion" not in layout and \
       "constexpr int kLayoutVersion" not in layout:
        problems.append("cmfd_layout::kLayoutVersion is missing")
    if "layoutName()" not in layout:
        problems.append("cmfd_layout::layoutName() is missing: the receipt needs a "
                        "spelling, not just a number")
    for helper, per_node in (("mat(", "4"), ("cpl(", "12"),
                             ("face(", "6"), ("dir(", "3")):
        body = body_of(layout, "inline long long " + helper)
        if not body:
            problems.append("cmfd_layout::%s is missing" % helper.rstrip("("))
        elif ("component(nxyz, l, " not in body) or (", " + per_node + ")" not in body):
            problems.append(
                "cmfd_layout::%s must delegate to component(nxyz, l, k, %s); a "
                "hand-rolled second formula is how the two layouts drift apart"
                % (helper.rstrip("("), per_node))
    core = body_of(layout, "inline long long component(int nxyz, int l, int k,")
    if not core:
        problems.append("cmfd_layout::component() is missing")
    else:
        if "kNodeInnermost" not in core:
            problems.append("cmfd_layout::component() does not branch on "
                            "kNodeInnermost, so the AoS reference is not reachable "
                            "and the permutation cannot be bisected")
        if "* nxyz + l" not in core:
            problems.append("cmfd_layout::component() does not compute the "
                            "component-major address k*nxyz + l")
        if "* per_node + k" not in core:
            problems.append("cmfd_layout::component() has lost the node-major "
                            "reference address l*per_node + k; the AoS arm is what "
                            "makes this a bisectable permutation rather than a "
                            "rewrite")
    return problems


def rule_no_aos_in_kernels(code: str) -> list:
    """The AoS spelling is gone from every operator kernel body."""
    problems = []
    clean = strip_comments(code)
    for sig in OPERATOR_KERNELS:
        body = body_of(clean, sig)
        if not body:
            problems.append("kernel not found: %s" % sig)
            continue
        for pattern in AOS_PATTERNS:
            hit = re.search(pattern, body)
            if hit:
                problems.append(
                    "%s still spells a node-major index (%r): one kernel left on "
                    "the old addressing against a producer on the new one is a "
                    "silent wrong answer, not a compile error"
                    % (sig.strip("("), hit.group(0)))
                break
    return problems


def _operator_pointers(body: str) -> list:
    """(local pointer name, required helper) for every operator array bound in
    this body.  Longest array name first so `diag_f` is not read as `diag`."""
    out = []
    for arrays, helper in ((MAT_ARRAYS, "mat"), (CPL_ARRAYS, "cpl")):
        for array in arrays:
            pattern = (r"(\w+)\s*=\s*" + re.escape(array) + r"\s*\+\s*m\s*\*")
            for hit in re.finditer(pattern, body):
                name = hit.group(1)
                if not any(name == n for n, _ in out):
                    out.append((name, helper))
    return out


def _layout_locals(body: str) -> set:
    """Identifiers assigned from a cmfd_layout helper, so `dm[idx]` where
    `idx = cmfd_layout::mat(...)` is an address computed the right way."""
    return set(re.findall(
        r"(?:const\s+)?(?:long long|int|size_t|auto)\s+(\w+)\s*=\s*cmfd_layout::",
        body))


def rule_helper_per_class(code: str) -> list:
    """Each array class goes through the helper it calls for."""
    problems = []
    clean = strip_comments(code)
    for sig in OPERATOR_KERNELS:
        body = body_of(clean, sig)
        if not body:
            continue
        indirect = _layout_locals(body)
        for name, helper in _operator_pointers(body):
            want = "cmfd_layout::%s(" % helper
            for hit in re.finditer(r"\b" + re.escape(name) + r"\[([^\]]*)\]", body):
                expr = hit.group(1)
                if want in expr:
                    continue
                if expr.strip() in indirect:
                    continue
                problems.append(
                    "%s indexes %s[%s] without %s"
                    % (sig.strip("("), name, expr.strip()[:40], want))
        for hit in re.finditer(r"neighbors\[([^\]]*)\]", body):
            if "cmfd_layout::face(" not in hit.group(1):
                problems.append(
                    "%s indexes neighbors[%s] without cmfd_layout::face()"
                    % (sig.strip("("), hit.group(1)[:40]))
    return problems


def rule_producers_agree(code: str, layout: str) -> list:
    """Every writer of a node-indexed array writes the layout the readers read."""
    problems = []
    assembly = body_of(layout, "inline void assembleNode2G(")
    if not assembly:
        problems.append("assembleNode2G() is missing from src/CmfdAssemblyKernel.h")
    else:
        for want, why in (
                ("v.diag[cmfd_layout::mat(", "diag"),
                ("v.cc[cmfd_layout::cpl(", "cc"),
                ("v.udiag[at]", "udiag"),
                ("v.node_surface[cmfd_layout::face(", "node_surface"),
                ("v.face_area[cmfd_layout::dir(", "face_area")):
            if want not in assembly:
                problems.append(
                    "assembleNode2G() no longer writes/reads %s through "
                    "cmfd_layout: it is the DEVICE producer of the operator the "
                    "five kernels consume" % why)
        for pattern in AOS_PATTERNS:
            if re.search(pattern, strip_comments(assembly)):
                problems.append("assembleNode2G() still spells a node-major index")
                break
    clean = strip_comments(code)
    init = body_of(clean, "void init(")
    haystack = init if init else clean
    for want, why in (
            ("soa_neighbors", "neighbors"),
            ("soa_node_surface", "assembly_node_surface"),
            ("soa_face_area", "assembly_face_area")):
        if want not in haystack:
            problems.append(
                "the one-time upload of %s is not packed into the component-major "
                "order the kernels read" % why)
    for want, why in (
            ("packMat(xpose_diag, m, sl.host_diag)", "issueUploads:diag"),
            ("packCpl(m, sl.host_cc)", "cc"),
            ("packMat(xpose_udiag, m, sl.host_udiag)", "issueSweepUploads:udiag")):
        if want not in clean:
            problems.append(
                "the H2D of %s does not go through the pack lane: it would push "
                "node-major bytes into a component-major array" % why)
    for want in ("unpackMat(xpose_diag, m, sl.host_diag_out)",
                 "unpackCpl(m, sl.host_cc_out)",
                 "unpackMat(xpose_udiag, m, sl.host_udiag)"):
        if want not in clean:
            problems.append(
                "the exceptional D2H is missing its scatter back to node-major "
                "(%s): the CPU reference solver reads those arrays" % want)
    if "unpack_operator" not in clean:
        problems.append("the D2H scatter must be deferred to after the drain; the "
                        "unpack_operator list is gone")
    return problems


def rule_transfer_sizes(code: str) -> list:
    """The permutation does not resize anything."""
    problems = []
    clean = strip_comments(code)
    for helper, count in (("const double* packMat(", "matrix_count"),
                          ("const double* packCpl(", "coupling_count"),
                          ("void unpackMat(", "matrix_count"),
                          ("void unpackCpl(", "coupling_count")):
        body = body_of(clean, helper)
        if not body:
            problems.append("%s is missing" % helper)
            continue
        if count not in body:
            problems.append(
                "%s does not stride its lane by %s: the lane must hold exactly "
                "the doubles the DMA moves, or the transfer changes size"
                % (helper, count))
        if "cmfd_layout::" not in body:
            problems.append("%s does not use the shared index helper" % helper)
    for want in ('"diag", diag + m * mat_stride()',
                 'matrix_count * sizeof(double)',
                 'coupling_count * sizeof(double)'):
        if want not in clean:
            problems.append("the operator transfer no longer moves %s" % want)
    if "packMat(xpose_diag, m, sl.host_diag), bytes" not in clean:
        problems.append(
            "the issueUploads diag push must keep its `bytes` argument -- the "
            "byte count the xfer ledger records is not allowed to move")
    return problems


def rule_reduction_operands(code: str) -> list:
    """The Krylov vectors keep the element index, so no sum is re-associated."""
    problems = []
    clean = strip_comments(code)
    for sig in REDUCE_KERNELS:
        body = body_of(clean, sig)
        if not body:
            problems.append("reduce kernel not found: %s" % sig)
            continue
        if "cmfd_layout::" in body:
            problems.append(
                "%s indexes through cmfd_layout: permuting a reduction operand "
                "re-chunks `chunk = ceil(n/gridDim.x)` and moves additions -- "
                "that is N1, not B0" % sig.strip("("))
        if "const int chunk = (n + static_cast<int>(gridDim.x) - 1) / " \
                "static_cast<int>(gridDim.x);" not in body:
            problems.append("%s no longer fixes its partition from n and gridDim.x"
                            % sig.strip("("))
        if "i += static_cast<int>(blockDim.x)" not in body:
            problems.append("%s no longer walks its chunk with the blockDim stride"
                            % sig.strip("("))
    for sig in ELEMENT_KERNELS:
        body = body_of(clean, sig)
        if not body:
            problems.append("element kernel not found: %s" % sig)
            continue
        if "cmfd_layout::" in body:
            problems.append(
                "%s indexes through cmfd_layout: the flux, the source and the "
                "Krylov state are DELIBERATELY left node-major -- phi is a "
                "pointer handoff to the nodal/PPR readers (CmfdResidentView) and "
                "the vectors are reduction operands" % sig.strip("("))
    view = body_of(code, "struct CmfdResidentView {")
    if view and "phi" not in view:
        problems.append("CmfdResidentView no longer exposes phi")
    return problems


def rule_receipt(code: str) -> list:
    """The graph census says which layout it profiled."""
    problems = []
    census = body_of(code, "static void reportCmfdGraphCensus(")
    if not census:
        problems.append("reportCmfdGraphCensus() is missing")
        return problems
    for field in GRAPH_FIELDS:
        if '\\"%s\\"' % field not in census:
            problems.append(
                "[RASBERY][CMFD][GRAPH] does not carry %r: a profile and a digest "
                "must never be readable against the wrong kernel bodies" % field)
    if "cmfd_layout::layoutName()" not in census:
        problems.append("the receipt hard-codes the layout name instead of asking "
                        "cmfd_layout for it")
    if "cmfd_layout::kLayoutVersion" not in census:
        problems.append("the receipt hard-codes the layout version")
    return problems


def rule_fuse_norm(code: str) -> list:
    """Bit 4 exists, is off by default, and keeps the over-run telemetry."""
    problems = []
    clean = strip_comments(code)
    if "kFuseNorm     = 1u << 4" not in clean and "kFuseNorm = 1u << 4" not in clean:
        problems.append("kFuseNorm (FUSE bit 4) is missing")
    # The default names the ADOPTED set (kFusePricedBits); kFuseAllBits is the
    # strictly larger VALIDATION set that makes `RASBERY_GPU_CMFD_FUSE=31`
    # parse.  Bit 4 belongs in the second and not the first, so both the
    # default and the set it names are checked.
    default = re.search(r"kFuseDefaultMask = ([^;}]*)", clean)
    priced = re.search(r"kFusePricedBits = ([^,;}]*)", clean)
    if not default:
        problems.append("kFuseDefaultMask is missing")
    elif "kFuseAllBits" in default.group(1):
        problems.append(
            "kFuseDefaultMask is kFuseAllBits, which includes bit 4: that adopts "
            "an arm nobody priced on 238")
    elif "kFuseNorm" in default.group(1):
        problems.append(
            "kFuseNorm is in kFuseDefaultMask: a default is a claim, and bit 4 "
            "has not been priced on 238 -- it ships armed and OFF")
    if not priced:
        problems.append("kFusePricedBits is missing: the adopted set has no name")
    elif "kFuseNorm" in priced.group(1):
        problems.append(
            "kFuseNorm is in kFusePricedBits: the default names that set, so bit 4 "
            "would be adopted through the indirection")
    for sig in ("__global__ void reduce_norm_accumulate_fused(",
                "__global__ void reduce_norm_accumulate_fused_f32("):
        body = body_of(clean, sig)
        if not body:
            problems.append("%s is missing" % sig.strip("("))
            continue
        if "kOverrunCount" not in body:
            problems.append(
                "%s drops the over-run tally.  In the two-node form a halted slot "
                "ran stage 1 for nothing and stage 2 STILL counted the overrun; "
                "that counter is the evidence the halt gating is complete"
                % sig.strip("("))
        if "active[m] == 0u" not in body:
            problems.append("%s has lost stage 2's `active` guard" % sig.strip("("))
        if "atomicInc(retire + m, gridDim.x - 1u)" not in body:
            problems.append(
                "%s does not gate its fold on the retire counter, so the fold can "
                "read partials that are not written yet" % sig.strip("("))
        if "for (int i = 0; i < blocks; ++i) fold += vpm[i];" not in body:
            problems.append("%s no longer folds in strict ascending index order"
                            % sig.strip("("))
        if "HALT_GUARD" in body:
            problems.append(
                "%s uses HALT_GUARD, which returns before the counter: the halted "
                "path has to BE stage 2's halted path, not a guard" % sig.strip("("))
    if "if (fuse_norm && scalar_fusion) {" not in clean:
        problems.append("bit 4 is never dispatched, or it is dispatched without "
                        "the scalar_fusion precondition its tail needs")
    return problems


def rule_doc(doc: str) -> list:
    problems = []
    for want in ("sectors", "6.55", "9.92", "cmfd_layout", "B0", "runbook"):
        if want not in doc:
            problems.append("docs/WP21_A_CMFD_COALESCING_20260831_KO.md does not "
                            "mention %r" % want)
    return problems


# ---------------------------------------------------------------------------
# Negative controls.
# ---------------------------------------------------------------------------

CONTROLS = (
    ("single definition",
     lambda s: rule_single_definition(s),
     "namespace rasbery::cmfd_layout {\n"
     "inline long long mat(int nxyz, int l, int k) { return l * 4 + k; }\n"
     "}\n"),
    ("no AoS in kernels",
     lambda s: rule_no_aos_in_kernels(s),
     "__global__ void matvec_two_group(int nxyz) {\n"
     "    double y0 = dm[4 * l + 0];\n"
     "}\n"),
    ("helper per class",
     lambda s: rule_helper_per_class(s),
     "__global__ void matvec_two_group(int nxyz) {\n"
     "    const double* dm = diag + m * mat_stride;\n"
     "    double y0 = dm[k * nxyz + l];\n"
     "}\n"),
    ("producers agree",
     lambda s: rule_producers_agree(s, s),
     "inline void assembleNode2G(const View& v, int l) {\n"
     "    v.diag[l * 4 + 0] = 1.0;\n"
     "}\n"),
    ("transfer sizes",
     lambda s: rule_transfer_sizes(s),
     "const double* packMat(double* lane, int m, const double* a) {\n"
     "    return lane;\n"
     "}\n"),
    ("reduction operands",
     lambda s: rule_reduction_operands(s),
     "__global__ void reduce_dot_stage1(int n) {\n"
     "    sum += am[cmfd_layout::mat(nxyz, i, 0)];\n"
     "}\n"),
    ("receipt",
     lambda s: rule_receipt(s),
     "static void reportCmfdGraphCensus(const char* tag) {\n"
     '    line << "[RASBERY][CMFD][GRAPH] {\\"nodes\\":" << count;\n'
     "}\n"),
    ("fuse bit 4 adopted through the default",
     lambda s: rule_fuse_norm(s),
     "enum CmfdFuseBit : unsigned { kFuseNorm = 1u << 4 };\n"
     "enum : unsigned { kFuseDefaultMask = kFuseAllBits };\n"),
    ("fuse bit 4 adopted through the priced set",
     lambda s: rule_fuse_norm(s),
     "enum CmfdFuseBit : unsigned { kFuseNorm = 1u << 4,\n"
     "  kFusePricedBits = kFuseDot | kFuseNorm };\n"
     "enum : unsigned { kFuseDefaultMask = kFusePricedBits };\n"),
)

DOC_CONTROL = "WP21-A moved some arrays around and it went fine."


def self_test() -> list:
    failures = []
    for label, rule, snippet in CONTROLS:
        if not rule(snippet):
            failures.append("negative control did not fire: %s" % label)
    if not rule_doc(DOC_CONTROL):
        failures.append("negative control did not fire: doc")
    return failures


def main() -> int:
    code = read(BACKEND)
    layout = read(LAYOUT)
    try:
        doc = read(DOC)
    except OSError:
        doc = ""

    problems = []
    problems += rule_single_definition(layout)
    problems += rule_no_aos_in_kernels(code)
    problems += rule_helper_per_class(code)
    problems += rule_producers_agree(code, layout)
    problems += rule_transfer_sizes(code)
    problems += rule_reduction_operands(code)
    problems += rule_receipt(code)
    problems += rule_fuse_norm(code)
    if not doc:
        problems.append("docs/WP21_A_CMFD_COALESCING_20260831_KO.md is missing: the "
                        "inventory table and the 238 runbook are the deliverable, "
                        "not a nicety")
    else:
        problems += rule_doc(doc)

    controls = self_test()
    if problems or controls:
        print("FAIL: CMFD SoA layout contract")
        for problem in problems:
            print("  - " + problem)
        for control in controls:
            print("  - " + control)
        return 1

    print("PASS: CMFD SoA layout contract")
    print("  layout: cmfd_layout::kNodeInnermost -> version %s"
          % (re.search(r"kLayoutVersion = kNodeInnermost \? (\d+)", layout).group(1)
             if re.search(r"kLayoutVersion = kNodeInnermost \? (\d+)", layout)
             else "?"))
    print("  permuted (B0): diag, dinv, udiag, cc, diag_f, dinv_f, cc_f, "
          "neighbors, node_surface, face_area")
    print("  NOT permuted (N1 if moved): phi, src, r, r0, p, v, s, t, y, z, ax, "
          "psi, dtil, dhat -- reduction operands and the nodal/PPR pointer handoff")
    print("  operator kernels checked: %d" % len(OPERATOR_KERNELS))
    print("  fuse bit 4 (kFuseNorm): armed, OFF by default, over-run tally kept")
    print("  negative controls: %d, all fired" % (len(CONTROLS) + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
