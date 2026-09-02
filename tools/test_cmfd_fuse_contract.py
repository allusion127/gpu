#!/usr/bin/env python3
"""Contract gate for WP7 stage B: the CMFD graph-node fusion bitmask.

WHAT STAGE B IS.  The single-deck GPU profile puts 23.7% of device time in
`colored_block_sweep` at 2.23 us per launch over 701k launches and 15.5% in the
reduce-dot family, and the captured CMFD graphs dispatch ~90 nodes per BiCG
outer.  The lever is the NODE COUNT, and the only safe way to lower it is to
concatenate ADJACENT kernels that can be concatenated without moving a single
floating-point operation.  `RASBERY_GPU_CMFD_FUSE` is the bitmask that arms
those concatenations, one bit at a time, default 0.

THE THREE PROPERTIES THIS GATE PINS

  1. EVERY FUSED KERNEL CARRIES A WRITTEN ORDER-PRESERVATION ARGUMENT.  A
     fusion whose bit-identity argument lives only in a reviewer's head is a
     fusion nobody can re-check when the reference kernel next changes.  So
     each fused kernel must be preceded by a comment block naming the bit, and
     that block must contain the words ORDER-PRESERVATION NOTE.

  2. MASK 0 LEAVES THE GRAPH NODE SET IDENTICAL TO THE CENSUS.  Each fused
     kernel is reachable only through a `fuse_*` boolean, and the reference
     kernels it replaces are still launched on the other side of that branch --
     in the SAME function.  A fusion that deletes its reference has removed the
     thing the B0 replay compares against.

  3. THE CENSUS AND THE CODE CANNOT DRIFT.  The node counts written in
     docs/WP7_CMFD_GRAPH_CENSUS_20260831_KO.md are recomputed here from a model
     whose structural inputs (how many dots, how many colour sweeps, how many
     elementwise kernels per iteration) are COUNTED OUT OF THE SOURCE.  Add a
     kernel to enqueue_iteration and this test fails until the census is
     updated -- which is the only way a node census stays true.

Plus the receipt: `[RASBERY][CMFD][GRAPH]` must carry the five fields the plan
names, must be emitted where a graph is instantiated, and must be
trajectory-neutral -- its emitter may not enqueue, launch, copy or synchronise
anything.

NEGATIVE CONTROLS.  Every rule is also run against a synthetic snippet that
violates it, so a rule that has quietly stopped matching anything fails here
rather than passing forever.

Pure python, no build, no device.

Run:  python tools/test_cmfd_fuse_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join("src", "CudaBICGBackend.cu")
CENSUS = os.path.join("docs", "WP7_CMFD_GRAPH_CENSUS_20260831_KO.md")

# bit -> (fused kernel, the reference kernels it stands in for, dispatch guard,
#         the function whose body must hold both paths)
FUSIONS = {
    0: (
        "reduce_dot_fused",
        ("reduce_dot_stage1", "reduce_dot_stage2"),
        "fuse_dot",
        "void dot(",
    ),
    1: (
        "reduce_dot2_fused",
        ("reduce_dot2_stage1", "reduce_dot2_stage2"),
        "fuse_dot2",
        "void dot2(",
    ),
    2: (
        "cmfd_wiel_fused",
        ("cmfd_wiel_stage1", "cmfd_wiel_finalize_chunked"),
        "fuse_wiel",
        "void enqueue_sweeps(",
    ),
    3: (
        "cmfd_sweep_gate_patch",
        ("cmfd_sweep_gate", "cmfd_sweep_patch"),
        "fuse_sweep_pre",
        "void enqueueSweepPreamble(",
    ),
}

RECEIPT_FIELDS = (
    "nodes_per_sweep",
    "kernel_nodes",
    "memcpy_nodes",
    "memset_nodes",
    "launches_per_outer",
)

# The deck the census tabulates: KNGR, nmax = 3, rb_sweeps = 4, ncolors = 2,
# scalar fusion on (the default), FP64 inner, chunked Wielandt fold.
REF_NMAX = 3
REF_RB = 4


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
                return code[open_brace : i + 1]
    return ""


def preamble_of(code: str, kernel: str) -> str:
    """The comment block immediately above `__global__ void <kernel>`."""
    hit = code.find("__global__ void " + kernel)
    if hit < 0:
        return ""
    head = code[:hit]
    # Walk back over the contiguous run of `//` lines.
    lines = head.split("\n")
    out = []
    for line in reversed(lines):
        if line.strip() == "" and not out:
            continue
        if line.lstrip().startswith("//"):
            out.append(line)
            continue
        break
    return "\n".join(reversed(out))


# ---------------------------------------------------------------------------
# The rules.  Each takes the source text (and, where it needs it, the census)
# and returns a list of problem strings.  Each is also fed a synthetic
# violation in self_test() below.
# ---------------------------------------------------------------------------


def rule_flag_family(code: str) -> list:
    """One env name, read once, masked to the declared bits, default 15.

    THE DEFAULT FLIPPED AT THE v5 FREEZE and this rule flipped with it.  What it
    still enforces is the part that was never about which value is the default:
    ONE read, LATCHED, MASKED to the declared bits, and a named constant rather
    than a literal -- so that "which mask is the default" is a question with one
    answer in the source instead of a number repeated in three places.
    """
    problems = []
    reads = re.findall(r'getenv\s*\(\s*"RASBERY_GPU_CMFD_FUSE"\s*\)', code)
    if len(reads) != 1:
        problems.append(
            "RASBERY_GPU_CMFD_FUSE must be read exactly once (found %d): the mask "
            "fixes the captured graph topology, so it cannot be allowed to change "
            "between two outers of the same run" % len(reads)
        )
    gate = body_of(code, "unsigned cmfdFuseMask(")
    if not gate:
        problems.append("cmfdFuseMask() is missing: there is no single latch for the mask")
        return problems
    if "static const" not in gate:
        problems.append(
            "cmfdFuseMask() must latch its value in a `static const` initialiser; "
            "an unlatched getenv can return a different mask per call"
        )
    if "kFuseAllBits" not in gate:
        problems.append(
            "cmfdFuseMask() must mask the parsed value with kFuseAllBits, so an "
            "unknown bit cannot arm an undeclared path"
        )
    if "kFuseDefaultMask" not in gate:
        problems.append(
            "cmfdFuseMask() must fall back to the NAMED default kFuseDefaultMask "
            "when the variable is unset or unparseable -- a literal here is a "
            "second answer to 'what is the default mask'"
        )
    if "return 0u" in gate:
        problems.append(
            "cmfdFuseMask() still falls back to 0.  Since the v5 freeze the "
            "default is kFuseDefaultMask (= kFuseAllBits = 15); 0 survives only "
            "as the EXPLICIT off switch, which strtoul already produces"
        )
    return problems


# The four bits that were priced and adopted.  kFuseDefaultMask must be the ONE
# name that enumerates exactly these (kFusePricedBits) -- never a literal, and
# never kFuseAllBits, which since WP21-A is the strictly larger VALIDATION set
# (what a user may request) rather than the ADOPTION set (what the tree claims).
ADOPTED_BITS = ("kFuseDot", "kFuseDot2", "kFuseWiel", "kFuseSweepPre")


def rule_default_mask_is_all_bits(code: str) -> list:
    """The named default exists and is the mask both hosts gated.

    238 measured every mask in {0,1,2,4,8,15} at digest 0d15abf29d222a02 / 4382
    and 181 measured every one of them at 1d897e3f77204799 with h5diff 0 lines
    against mask 0, so B0 is established for the whole family; 15 is the member
    that also cleared the 0.2 s adoption threshold (-0.396 s interleaved, where
    mask 4 managed -0.183 s).  A default that quietly became a DIFFERENT member
    of the family would be a wall claim nobody measured, so the constant is
    pinned to the four adopted bits BY NAME.

    WP21-A ADDED A FIFTH BIT AND DID NOT ADOPT IT.  kFuseNorm (bit 4, the
    residual-norm stage-2 fold) is B0 by the same textual argument as the other
    four, but it has not been priced on 238, so it is in kFuseAllBits -- which
    is what cmfdFuseMask() validates a user's request against -- and OUT of the
    default.  That is why this rule can no longer be "== kFuseAllBits": the two
    constants deliberately differ now, and the gate has to say which is which.
    """
    problems = []
    default = re.search(r"kFuseDefaultMask = ([^;}]*)", code)
    if not default:
        problems.append("kFuseDefaultMask is missing")
        return problems
    spelling = default.group(1).strip()
    if "kFuseAllBits" in spelling:
        problems.append(
            "kFuseDefaultMask is kFuseAllBits, which since WP21-A also contains "
            "the unpriced bit 4 (kFuseNorm): the default would become a wall "
            "claim nobody measured")
        return problems
    if spelling != "kFusePricedBits":
        problems.append(
            "kFuseDefaultMask is %r rather than the named adopted set "
            "kFusePricedBits: the default has to be ONE name, so a literal "
            "cannot change the arm without changing the receipt's vocabulary"
            % spelling)
        return problems
    # Follow the indirection.  kFusePricedBits is where the four adopted bits
    # are actually enumerated, and it is the constant a rebase could hollow out.
    priced = re.search(r"kFusePricedBits = ([^,;}]*)", code)
    if not priced:
        problems.append("kFusePricedBits is missing: the adopted set has no name")
        return problems
    members = priced.group(1)
    for bit in ADOPTED_BITS:
        if not re.search(r"(?<![A-Za-z0-9_])" + bit + r"(?![A-Za-z0-9_])", members):
            problems.append(
                "kFusePricedBits no longer names %s: the adopted default is the "
                "four priced bits, spelled by name" % bit)
    if re.search(r"(?<![A-Za-z0-9_])kFuseNorm(?![A-Za-z0-9_])", members):
        problems.append(
            "kFuseNorm is in kFusePricedBits: bit 4 ships armed and OFF until "
            "the 238 runbook in docs/WP21_A_CMFD_COALESCING_20260831_KO.md "
            "prices mask 31 against mask 15")
    allbits = re.search(r"kFuseAllBits[ ]*=[ ]*([^,;}]*)", code)
    if not allbits:
        problems.append("kFuseAllBits is missing: nothing validates a parsed mask")
    elif "kFusePricedBits" not in allbits.group(1):
        problems.append(
            "kFuseAllBits is not built from kFusePricedBits: the validation set "
            "must be a superset of the adopted set, or cmfdFuseMask() could mask "
            "away a bit the default claims")
    return problems


def rule_order_notes(code: str) -> list:
    """Every fused kernel has a documented order-preservation argument."""
    problems = []
    for bit, (fused, _refs, _guard, _site) in sorted(FUSIONS.items()):
        if "__global__ void " + fused not in code:
            problems.append("bit %d: fused kernel %s does not exist" % (bit, fused))
            continue
        note = preamble_of(code, fused)
        if "ORDER-PRESERVATION NOTE" not in note:
            problems.append(
                "bit %d: %s has no ORDER-PRESERVATION NOTE above it -- a fusion "
                "whose bit-identity argument is not written down cannot be "
                "re-checked when its reference kernel next changes" % (bit, fused)
            )
        if ("bit %d" % bit) not in note:
            problems.append(
                "bit %d: %s's note does not name the bit that arms it" % (bit, fused)
            )
    return problems


def rule_reference_survives(code: str) -> list:
    """Mask 0 must still launch the reference kernels, from the same function."""
    problems = []
    for bit, (fused, refs, guard, site) in sorted(FUSIONS.items()):
        for ref in refs:
            if ref + "<<<" not in code and "__global__ void " + ref not in code:
                problems.append(
                    "bit %d: reference kernel %s is gone -- the fused path has "
                    "nothing left to be compared against" % (bit, ref)
                )
        body = strip_comments(body_of(code, site))
        if not body:
            problems.append("bit %d: dispatch site %s not found" % (bit, site))
            continue
        if guard not in body:
            problems.append(
                "bit %d: %s does not consult %s, so the fusion is not behind the "
                "flag family" % (bit, site, guard)
            )
        if fused + "<<<" not in body:
            problems.append(
                "bit %d: %s never launches %s" % (bit, site, fused)
            )
        for ref in refs:
            launched = (ref + "<<<") in body or ("enqueueSweep" in body and ref in code)
            if not launched:
                problems.append(
                    "bit %d: %s no longer contains the reference launch of %s; the "
                    "mask-0 path must stay in the same function as the fused one"
                    % (bit, site, ref)
                )
    return problems


def rule_colour_sweep_refusal(code: str) -> list:
    """colored_block_sweep is NOT fusable and the source must say why."""
    problems = []
    for bit, (fused, _refs, _guard, _site) in sorted(FUSIONS.items()):
        body = body_of(code, "__global__ void " + fused)
        if "colored_block_sweep" in body:
            problems.append(
                "bit %d: %s absorbs colored_block_sweep.  Sweep k+1 reads x at "
                "NEIGHBOURING nodes, so it depends on sweep k across the whole "
                "grid; the kernel boundary IS that barrier and the colour order "
                "IS the Gauss-Seidel semantics." % (bit, fused)
            )
    gate = code
    if "NOT FUSABLE" not in gate:
        problems.append(
            "the source must state, in the FUSE section, that the colour sweeps "
            "are NOT FUSABLE and why -- it is the largest single item in the "
            "single-run Amdahl table and the first thing a reader will ask about"
        )
    return problems


def rule_receipt(code: str) -> list:
    problems = []
    if "[RASBERY][CMFD][GRAPH]" not in code:
        problems.append("the [RASBERY][CMFD][GRAPH] receipt is missing")
        return problems
    emitter = body_of(code, "static void reportCmfdGraphCensus(")
    if not emitter:
        problems.append("reportCmfdGraphCensus() is missing")
        return problems
    for field in RECEIPT_FIELDS:
        if ('\\"%s\\"' % field) not in emitter and ('"%s"' % field) not in emitter:
            problems.append("the [CMFD][GRAPH] receipt does not carry %s" % field)
    # Trajectory neutrality: a receipt that enqueues, copies or synchronises is
    # not an observation of the run, it is part of it.
    forbidden = ("cudaMemcpy", "cudaLaunch", "Synchronize", "<<<", "cudaStreamBegin")
    stripped = strip_comments(emitter)
    for token in forbidden:
        if token in stripped:
            problems.append(
                "reportCmfdGraphCensus() contains `%s`: a census that touches the "
                "stream is not trajectory-neutral" % token
            )
    # Emitted where a graph is instantiated, on BOTH graphs.
    for site, tag in (("void launch_outer(", '"outer"'), ("void launch_sweeps(", '"sweep"')):
        body = strip_comments(body_of(code, site))
        if "reportCmfdGraphCensus(" not in body:
            problems.append("%s does not emit the [CMFD][GRAPH] census" % site)
        elif tag not in body:
            problems.append("%s emits the census under the wrong graph tag" % site)
        if "cudaGraphInstantiate" not in body:
            problems.append(
                "%s no longer instantiates a graph; the census emission has moved "
                "away from the one place a node set is created" % site
            )
    return problems


# ---------------------------------------------------------------------------
# The node model.  Structural inputs are COUNTED OUT OF THE SOURCE, so the
# census cannot drift away from the code without this failing.
# ---------------------------------------------------------------------------

# What each body must contain.  These are the numbers the model is built on;
# changing the solver changes them and forces the census to be rewritten.
EXPECTED_STRUCTURE = {
    "void enqueue_iteration(": {
        "dot(": 2,        # rho_new and r0.v
        "dot2(": 1,       # the (s.t, t.t) pair
        "precondition_sweeps(": 2,
        # 5 elementwise + reduce_dot_stage1 + 3 scalar-tail variants, plus
        # WP21-A's bit-4 branch (reduce_norm_accumulate_fused) which replaces
        # the stage1 + stage2 pair with one node when the bit is set.
        "<<<": 10,
    },
    "void enqueue_outer(": {
        "<<<": 9,         # init, fp32 mirror, 2 begin variants, stage1, 3 stage2
                          # variants, finalize_status
        # The status D2H.  Spelled `xfer::memcpyAsync(` since WP13.1 routed
        # every transfer in src/ through the site-tagged wrapper; the token is
        # the SUFFIX so this counts the call whichever of the two it is, because
        # what the node model cares about is that there is exactly ONE copy node
        # here -- not which header the call went through.
        "memcpyAsync(": 1,
    },
    "void enqueue_sweeps(": {
        "<<<": 11,        # assemble + 6 tail kernels + 4 Wielandt-fold variants
    },
}


def rule_structure(code: str) -> list:
    problems = []
    for site, wants in EXPECTED_STRUCTURE.items():
        body = strip_comments(body_of(code, site))
        if not body:
            problems.append("structure: %s not found" % site)
            continue
        for token, want in wants.items():
            got = body.count(token)
            if got != want:
                problems.append(
                    "structure: %s contains %d x `%s`, the node model assumes %d. "
                    "The model and the census in %s must be updated together with "
                    "the solver." % (site, got, token, want, CENSUS)
                )
    return problems


def model(nmax: int, rb: int, fuse: int, scalar_fusion: bool = True,
          fp32: bool = False, chunked_wiel: bool = True):
    """Node counts for one capture.  Returns (outer, per_sweep)."""
    captured = 1 + nmax
    dot_nodes = 1 if (fuse & 1) else 2
    dot2_nodes = 1 if (fuse & 2) else 2
    tail = 1 if scalar_fusion else 2
    # prologue: initialize_solver_state, [refresh_operator_mirror_f32,]
    #           begin_outer_fused, reduce_dot_stage1, stage-2 tail
    prologue = 2 + (1 if fp32 else 0) + 1 + tail
    # iteration: 2 dots + 1 dot2 + 2 x rb colour sweeps + prepare_p_jacobi +
    #            2 matvecs + update_s_jacobi + update_solution +
    #            reduce_dot_stage1 + stage-2 tail
    iteration = 2 * dot_nodes + dot2_nodes + 2 * rb + 6 + tail
    # epilogue: finalize_status + the status D2H memcpy node
    outer = prologue + captured * iteration + 2
    if chunked_wiel:
        wiel = 1 if (fuse & 4) else 2
    else:
        wiel = 1
    # sweep tail: begin, src_build, wiel_terms, updls, negative_scan, sweep_end
    per_sweep = 6 + wiel + outer
    return outer, per_sweep


def rule_census_numbers(census: str) -> list:
    """The doc must state the numbers the model produces."""
    problems = []
    checks = [
        ("outer, FUSE=0", model(REF_NMAX, REF_RB, 0)[0]),
        ("outer, FUSE=3", model(REF_NMAX, REF_RB, 3)[0]),
        ("per sweep, FUSE=0", model(REF_NMAX, REF_RB, 0)[1]),
        ("per sweep, FUSE=7", model(REF_NMAX, REF_RB, 7)[1]),
        ("outer, FUSE=0, scalar fusion off", model(REF_NMAX, REF_RB, 0, False)[0]),
    ]
    for label, value in checks:
        if not re.search(r"(?<![0-9])%d(?![0-9])" % value, census):
            problems.append(
                "the census does not state %d (%s); doc and model have drifted"
                % (value, label)
            )
    return problems


def rule_census_runbook(census: str) -> list:
    """The 238 runbook must name the gates the plan requires."""
    problems = []
    required = {
        "h5diff": "h5diff -c 0/644 byte identity",
        "0d15abf29d222a02": "the [TRAJECTORY] digest",
        "4382": "the outer count the digest is paired with",
        "ctest": "the ctest suite",
        "cuda_api_sum": "the nsys cuda_api_sum table",
        "RASBERY_GPU_CMFD_FUSE": "the flag the arms are taken over",
        "16.9": "the single-run wall reference",
        "878": "the batch c/h reference",
    }
    for token, what in required.items():
        if token not in census:
            problems.append("the 238 runbook does not mention %s (%s)" % (what, token))
    return problems


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def self_test() -> list:
    """Each rule must reject a snippet that violates it."""
    failures = []

    def expect_fail(rule, text, label, *extra):
        if not rule(text, *extra):
            failures.append("negative control did not fire: " + label)

    # 1. Unlatched env read.
    expect_fail(
        rule_flag_family,
        'unsigned cmfdFuseMask() {\n'
        '    const char* v = std::getenv("RASBERY_GPU_CMFD_FUSE");\n'
        '    return v ? 1u : 0u;\n'
        '}\n',
        "cmfdFuseMask() without a static const latch",
    )
    # 2. Env read twice.
    expect_fail(
        rule_flag_family,
        'unsigned cmfdFuseMask() {\n'
        '    static const unsigned m = [] {\n'
        '        std::getenv("RASBERY_GPU_CMFD_FUSE");\n'
        '        std::getenv("RASBERY_GPU_CMFD_FUSE");\n'
        '        return 0u & kFuseAllBits; }();\n'
        '    return m;\n}\n',
        "RASBERY_GPU_CMFD_FUSE read twice",
    )
    # 2b. The default silently back to the reference mask.  This is the control
    # the v5 flip needs: a gate that still reads the variable once, still
    # latches it and still masks it, and is wrong ONLY in which value an unset
    # variable resolves to.
    expect_fail(
        rule_flag_family,
        'unsigned cmfdFuseMask() {\n'
        '    static const unsigned m = [] {\n'
        '        const char* v = std::getenv("RASBERY_GPU_CMFD_FUSE");\n'
        '        if (v == nullptr) return 0u;\n'
        '        return 1u & kFuseAllBits; }();\n'
        '    return m;\n}\n',
        "cmfdFuseMask() defaulting back to the reference mask 0",
    )
    # 2c. The default spelled as a literal instead of the named constants.
    expect_fail(
        rule_default_mask_is_all_bits,
        "enum : unsigned { kFuseDefaultMask = 15u };\n",
        "kFuseDefaultMask spelled as a literal rather than the adopted bits",
    )
    # 2d. The unpriced bit 4 folded into the ADOPTED set.
    expect_fail(
        rule_default_mask_is_all_bits,
        "enum : unsigned { kFusePricedBits = kFuseDot | kFuseDot2 | kFuseWiel"
        " | kFuseSweepPre | kFuseNorm,\n"
        "                  kFuseAllBits = kFusePricedBits };\n"
        "enum : unsigned { kFuseDefaultMask = kFusePricedBits };\n",
        "the unpriced kFuseNorm adopted into kFusePricedBits",
    )
    # 2e. The default widened from the adopted set to the validation set.
    expect_fail(
        rule_default_mask_is_all_bits,
        "enum : unsigned { kFusePricedBits = kFuseDot | kFuseDot2 | kFuseWiel"
        " | kFuseSweepPre,\n"
        "                  kFuseAllBits = kFusePricedBits | kFuseNorm };\n"
        "enum : unsigned { kFuseDefaultMask = kFuseAllBits };\n",
        "the fuse default widened to the validation set kFuseAllBits",
    )
    # 2f. The validation set narrowed until it no longer covers the default.
    expect_fail(
        rule_default_mask_is_all_bits,
        "enum : unsigned { kFusePricedBits = kFuseDot | kFuseDot2 | kFuseWiel"
        " | kFuseSweepPre,\n"
        "                  kFuseAllBits = kFuseNorm };\n"
        "enum : unsigned { kFuseDefaultMask = kFusePricedBits };\n",
        "kFuseAllBits no longer a superset of the adopted set",
    )
    # 3. Fused kernel with no order note.
    expect_fail(
        rule_order_notes,
        "// just some comment\n__global__ void reduce_dot_fused(int n) {}\n",
        "fused kernel without an ORDER-PRESERVATION NOTE",
    )
    # 4. Reference launch deleted from the dispatch site.
    expect_fail(
        rule_reference_survives,
        "__global__ void reduce_dot_fused(int n) {}\n"
        "__global__ void reduce_dot_stage1(int n) {}\n"
        "__global__ void reduce_dot_stage2(int n) {}\n"
        "void dot(const double* a) {\n"
        "    if (fuse_dot) { reduce_dot_fused<<<1, 1>>>(0); return; }\n"
        "}\n",
        "dispatch site whose reference launches were removed",
    )
    # 5. A fused kernel that swallowed a colour sweep.
    expect_fail(
        rule_colour_sweep_refusal,
        "NOT FUSABLE\n__global__ void cmfd_wiel_fused(int n) { colored_block_sweep(); }\n",
        "fused kernel containing colored_block_sweep",
    )
    # 6. A census emitter that synchronises.
    expect_fail(
        rule_receipt,
        '[RASBERY][CMFD][GRAPH]\n'
        'static void reportCmfdGraphCensus(int a) {\n'
        '    cudaDeviceSynchronize();\n'
        '}\n',
        "census emitter that synchronises",
    )
    # 7. A solver body that grew a kernel without updating the model.
    expect_fail(
        rule_structure,
        "void enqueue_iteration(int a) {\n"
        "    dot(r0, r, kRhoNew);\n"
        "}\n",
        "enqueue_iteration whose launch count no longer matches the model",
    )
    # 8. A census doc missing one of the model's numbers.
    expect_fail(
        rule_census_numbers,
        "no numbers here at all\n",
        "census with none of the model's node counts",
    )
    # 9. A runbook with no gates.
    expect_fail(
        rule_census_runbook,
        "just a heading\n",
        "runbook that names no acceptance gate",
    )
    return failures


def main() -> int:
    problems = []
    try:
        code = read(BACKEND)
    except OSError as exc:
        print("FAIL: cannot read %s (%s)" % (BACKEND, exc))
        return 1

    problems += rule_flag_family(code)
    problems += rule_default_mask_is_all_bits(code)
    problems += rule_order_notes(code)
    problems += rule_reference_survives(code)
    problems += rule_colour_sweep_refusal(code)
    problems += rule_receipt(code)
    problems += rule_structure(code)

    try:
        census = read(CENSUS)
    except OSError:
        census = ""
        problems.append(
            "%s is missing: the receipt's field meanings and the 238 runbook live "
            "there, and the node model has nothing to be checked against" % CENSUS
        )
    if census:
        problems += rule_census_numbers(census)
        problems += rule_census_runbook(census)

    controls = self_test()

    if problems or controls:
        print("FAIL: CMFD fuse contract")
        for problem in problems:
            print("  - " + problem)
        for control in controls:
            print("  - " + control)
        return 1

    outer0, sweep0 = model(REF_NMAX, REF_RB, 0)
    outer_f, sweep_f = model(REF_NMAX, REF_RB, 7)
    print("PASS: CMFD fuse contract")
    print("  bits checked: " + ", ".join(
        "%d=%s" % (b, FUSIONS[b][0]) for b in sorted(FUSIONS)))
    print("  reference deck (nmax=%d, rb_sweeps=%d, scalar fusion on):" % (REF_NMAX, REF_RB))
    print("    outer graph      FUSE=0 %3d nodes -> FUSE=7 %3d nodes" % (outer0, outer_f))
    print("    sweep graph      FUSE=0 %3d nodes/sweep -> FUSE=7 %3d nodes/sweep"
          % (sweep0, sweep_f))
    print("  negative controls: %d, all fired" % 12)
    return 0


if __name__ == "__main__":
    sys.exit(main())
