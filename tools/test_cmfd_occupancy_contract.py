#!/usr/bin/env python3
"""Contract gate for WP17: the CMFD occupancy knobs.

WHAT WP17 IS.  238 (RTX PRO 6000 Blackwell, 188 SMs), v6 single deck, 35
statepoints: `colored_block_sweep` runs 379,027 times at 2.55 us on a grid of
**34** blocks x 256 threads, `matvec_two_group` 95,405 times at 2.75 us on the
same 34, `update_solution` on 67, and the whole reduce-dot family on 17 with a
1-block/1-thread stage 2.  Every one of them is under a fifth of the SM array
and about 2.5 us long -- the dispatch floor, not the arithmetic.  WP17 opens two
knobs against that, both OFF by default:

  RASBERY_GPU_CMFD_BLOCK=<32|64|128|192|256>  narrows the block of the FIVE
      per-iteration elementwise classes, so 34 blocks becomes 133 (64) or 265
      (32).  B0: those five are elementwise in the index the launch hands them.

  RASBERY_GPU_CMFD_PERSISTENT=1  runs one whole BiCGSTAB iteration inside ONE
      cooperative launch with grid.sync() between stages, replacing 18
      dispatches with 1.  A spike: gated, refused by name, receipted, off.

WHAT THIS GATE PINS

  1. THE DEFAULT IS UNCHANGED.  Unset environment -> cmfd_block_threads() IS
     block_size, so the grids are the 34/67 the profile measured.  A knob whose
     "off" is not the old behaviour is not a knob, it is a rewrite.

  2. THE COLOUR ORDER IS PRESERVED.  The block partition is NOT the colouring:
     colored_block_sweep filters on `colors[l] != target_color` against a
     per-NODE array, so re-blocking splits a colour across more blocks and
     changes nothing about which nodes are updated together.  If that filter
     ever moved into the launch geometry, re-blocking would become N1 -- so the
     filter, and the `sweep % ncolors` order at the dispatch site, are pinned.

  3. THE REDUCTION ORDER IS PRESERVED.  The reductions are NOT re-blocked:
     their partition is `chunk = (n + gridDim.x - 1) / gridDim.x` and their
     tree is 256 lanes wide, so a different block count is a different sum.
     They keep kReduceThreads and reduce_blocks_for(n), and the persistent arm
     pins its own partition to that same reduce_blocks rather than to its grid.

  4. THE PERSISTENT ARM REFUSES BY NAME.  Every PersistentRefusal enumerator
     has a name, is reachable, and the switch has no `default:` so a new one
     cannot be added silently.  PERSISTENT and the captured graph are mutually
     exclusive and the source says so where it refuses.

  5. THE RECEIPTS CARRY THE FIELDS.  block_threads, the per-class
     blocks_per_launch, persistent_arm, cooperative_supported and
     launches_per_iteration.

NEGATIVE CONTROLS.  Every rule is also run against a synthetic snippet that
violates it, so a rule that has quietly stopped matching anything fails here
rather than passing for ever.

Pure python, no build, no device.

Run:  python tools/test_cmfd_occupancy_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join("src", "CudaBICGBackend.cu")
DOC = os.path.join("docs", "WP17_CMFD_OCCUPANCY_20260830_KO.md")

# The five per-iteration elementwise classes, FP64 and FP32 spellings.  These
# are the ones the block knob may move.
NODE_CLASSES = (
    "colored_block_sweep", "colored_block_sweep_f32",
    "prepare_p_jacobi", "prepare_p_jacobi_f32",
    "matvec_two_group", "matvec_two_group_f32",
    "update_s_jacobi", "update_s_jacobi_f32",
)
VECTOR_CLASSES = ("update_solution", "update_solution_f32")

# Per-node kernels that are NOT per-iteration and must keep the arena width:
# they run once per outer or once per sweep, so re-blocking them buys nothing
# and widens the blast radius of the knob for no measured reason.
UNTOUCHED_NODE_CLASSES = (
    "cmfd_assemble_operator_2g", "cmfd_src_build", "cmfd_wiel_terms",
    "cmfd_updls", "begin_outer_fused", "begin_outer_fused_f32",
    "refresh_operator_mirror_f32",
)

# The reduce family, which must stay on kReduceThreads / reduce_blocks_for(n).
REDUCE_CLASSES = (
    "reduce_dot_stage1", "reduce_dot_fused", "reduce_dot2_stage1",
    "reduce_dot2_fused", "reduce_dot_stage1_f32", "reduce_dot2_stage1_f32",
)

OCCUPANCY_FIELDS = (
    "block_threads", "sweep_block_threads", "node_blocks", "vector_blocks",
    "reduce_blocks", "scalar_blocks", "launches_per_iteration",
    "persistent_arm", "persistent_blocks", "cooperative_supported",
    "persistent_refusal",
)
GRAPH_FIELDS = ("block_threads", "node_blocks", "vector_blocks", "persistent_arm")


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


def preamble_of(code: str, kernel: str) -> str:
    """The contiguous run of `//` lines immediately above a declaration."""
    hit = code.find(kernel)
    if hit < 0:
        return ""
    out = []
    for line in reversed(code[:hit].split("\n")):
        if line.strip() == "" and not out:
            continue
        if line.lstrip().startswith("//") or line.lstrip().startswith("///"):
            out.append(line)
            continue
        break
    return "\n".join(reversed(out))


# ---------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------


def rule_default_unchanged(code: str) -> list:
    """One latched read, and OFF is byte-for-byte the old geometry."""
    problems = []
    reads = re.findall(r'getenv\s*\(\s*"RASBERY_GPU_CMFD_BLOCK"\s*\)', code)
    if len(reads) != 1:
        problems.append(
            "RASBERY_GPU_CMFD_BLOCK must be read exactly once (found %d): the "
            "block width fixes the captured graph topology, so it cannot change "
            "between two outers of one run" % len(reads))
    gate = body_of(code, "inline int cmfdBlockThreads(")
    if not gate:
        problems.append("cmfdBlockThreads() is missing: there is no single latch "
                        "for the width")
    else:
        if "static const" not in gate:
            problems.append("cmfdBlockThreads() must latch its value in a `static "
                            "const` initialiser; an unlatched getenv can return a "
                            "different width per call")
        if "return 0;" not in gate:
            problems.append("cmfdBlockThreads() must return 0 -- the sentinel for "
                            "UNCHANGED -- when the variable is unset or unparseable")
        if "32" not in gate:
            problems.append("cmfdBlockThreads() no longer admits 32; the 265-block "
                            "arm of the campaign needs it")
    fallback = body_of(code, "int cmfd_block_threads() const")
    if not fallback:
        problems.append("cmfd_block_threads() is missing")
    elif "block_size" not in fallback or "cmfd_block" not in fallback:
        problems.append(
            "cmfd_block_threads() must fall back to block_size when cmfd_block is "
            "the 0 sentinel: with the variable unset the five classes have to "
            "dispatch the SAME grid they did before WP17")
    for helper, base in (("int cmfd_node_blocks() const", "nxyz"),
                         ("int cmfd_vector_blocks() const", "n")):
        body = body_of(code, helper)
        if not body:
            problems.append("%s is missing" % helper)
        elif "cmfd_block_threads()" not in body:
            problems.append("%s does not derive its width from cmfd_block_threads()"
                            % helper)
        elif base not in body:
            problems.append("%s no longer covers `%s`" % (helper, base))
    return problems


def rule_five_classes(code: str) -> list:
    """Exactly the five per-iteration classes move; nothing else does."""
    problems = []
    for name in NODE_CLASSES:
        want = name + "<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>("
        if want not in code:
            problems.append(
                "%s is not launched on the WP17 grid.  The knob is worth nothing "
                "if one of the five per-iteration classes keeps the old width -- "
                "the chain is only as parallel as its narrowest dispatch." % name)
    for name in VECTOR_CLASSES:
        want = name + "<<<cmfd_vector_grid(), cmfd_block_threads(), 0, stream>>>("
        if want not in code:
            problems.append("%s is not launched on the WP17 vector grid" % name)
    for name in UNTOUCHED_NODE_CLASSES:
        if name + "<<<cmfd_node_grid()" in code:
            problems.append(
                "%s was re-blocked.  It runs once per outer or per sweep, not per "
                "BiCGSTAB iteration, so it is not on the launch-floor critical "
                "path and widening the knob's blast radius buys nothing measured."
                % name)
    return problems


def rule_colour_order(code: str) -> list:
    """The colouring is data, not launch geometry -- and stays that way."""
    problems = []
    sweep = body_of(code, "__global__ void colored_block_sweep(")
    if not sweep:
        problems.append("colored_block_sweep is gone")
        return problems
    if "colors[l] != target_color" not in sweep:
        problems.append(
            "colored_block_sweep no longer selects its colour from the per-NODE "
            "`colors` array against `target_color`.  That filter is the ONLY "
            "reason re-blocking is B0: if a block ever became a colour group, a "
            "different block count would be a different Gauss-Seidel order (N1).")
    for hazard in ("__shared__", "__syncthreads", "__shfl", "atomicAdd"):
        if hazard in sweep:
            problems.append(
                "colored_block_sweep contains `%s`: an intra-block reduction or "
                "ordering dependence would make the block partition part of the "
                "arithmetic, and re-blocking would stop being B0" % hazard)
    site = strip_comments(body_of(code, "void precondition_sweeps("))
    if not site:
        problems.append("precondition_sweeps() is gone")
    else:
        if "sweep % ncolors" not in site:
            problems.append(
                "precondition_sweeps no longer walks the colours as "
                "`sweep % ncolors`; the colour ORDER is the Gauss-Seidel "
                "semantics and is not a WP17 degree of freedom")
        if "rb_sweeps" not in site:
            problems.append("precondition_sweeps no longer runs rb_sweeps sweeps")
    # The persistent arm's copy of the same filter.
    helper = body_of(code, "__device__ inline void persistentColourSweepNode(")
    if not helper:
        problems.append("persistentColourSweepNode() is missing")
    elif "colors[l] != target_color" not in helper:
        problems.append(
            "the persistent arm's colour sweep does not carry the same "
            "`colors[l] != target_color` filter as colored_block_sweep")
    return problems


def rule_reduction_order(code: str) -> list:
    """The reductions keep their fixed partition, tree and fold."""
    problems = []
    for needle, why in (
        ("const int chunk = (n + static_cast<int>(gridDim.x) - 1) / "
         "static_cast<int>(gridDim.x);",
         "the reduce stage-1 partition is a pure function of (n, gridDim.x)"),
        ("for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order",
         "reduce_dot_stage2's strict ascending fold"),
    ):
        if needle not in code:
            problems.append("lost `%s...` -- %s" % (needle[:56], why))
    for name in REDUCE_CLASSES:
        for launch in re.findall(re.escape(name) + r"<<<[^>]*>>>", code):
            if "cmfd_block_threads()" in launch or "cmfd_node_grid()" in launch \
                    or "cmfd_vector_grid()" in launch:
                problems.append(
                    "%s is launched on the WP17 block width.  Its partition is "
                    "`chunk = ceil(n / gridDim.x)` and its tree is kReduceThreads "
                    "lanes wide, so a different block count or a different block "
                    "width is a DIFFERENT SUM: N1, not B0." % name)
            if "kReduceThreads" not in launch:
                problems.append(
                    "%s is not launched with kReduceThreads threads; the 256-lane "
                    "binary tree is a compile-time shape" % name)
        if name + "<<<" in code and "reduce_blocks_for(n)" not in code:
            problems.append("reduce_blocks_for(n) is gone; the 17-way partition "
                            "has no single definition any more")
    # The persistent arm must pin its partition to reduce_blocks, NOT its grid.
    stage1 = body_of(code, "__device__ inline void persistentDotStage1(")
    if not stage1:
        problems.append("persistentDotStage1() is missing")
    else:
        if "(n + reduce_blocks - 1) / reduce_blocks" not in stage1:
            problems.append(
                "persistentDotStage1 does not pin its chunk to `reduce_blocks`.  "
                "The persistent grid is sized for occupancy, not for the fold, so "
                "reading gridDim.x here would silently re-partition the sum.")
        if "gridDim.x" in stage1:
            problems.append(
                "persistentDotStage1 reads gridDim.x; the whole point of the "
                "pinned partition is that the fold does not follow the grid")
    fold = body_of(code, "__device__ inline double persistentFold(")
    if not fold:
        problems.append("persistentFold() is missing")
    elif "for (int i = 0; i < blocks; ++i) sum += pm[i];   // strict index order" \
            not in fold:
        problems.append(
            "persistentFold is not reduce_dot_stage2's strict ascending fold, "
            "verbatim; anything else re-associates the sum")
    return problems


def rule_refusal_ladder(code: str) -> list:
    """Every refusal has a name, is reachable, and nothing is defaulted."""
    problems = []
    enum = body_of(code, "enum class PersistentRefusal : int")
    if not enum:
        problems.append("PersistentRefusal is missing: the persistent arm has no "
                        "way to say WHY it did not run")
        return problems
    names = [m.group(1) for m in
             re.finditer(r"^\s*([A-Z][A-Za-z0-9]*)\s*(?:[,=]|$)", enum, re.M)]
    namer = body_of(code, "inline const char* persistentRefusalName(")
    if not namer:
        problems.append("persistentRefusalName() is missing")
        return problems
    if "default:" in namer:
        problems.append(
            "persistentRefusalName() has a `default:` label.  With one, adding an "
            "enumerator compiles and prints the wrong thing; without one the "
            "compiler names the omission.")
    if "Count" not in names:
        problems.append("PersistentRefusal has no Count sentinel")
    for name in names:
        if ("PersistentRefusal::" + name) not in namer:
            problems.append("PersistentRefusal::%s has no name in "
                            "persistentRefusalName()" % name)
        if name in ("None", "Count"):
            continue
        assigned = re.search(
            r"(persistent_refusal\s*=\s*PersistentRefusal::%s\b)" % name, code)
        if assigned is None:
            problems.append(
                "PersistentRefusal::%s is never assigned: a refusal nobody can "
                "reach is a name the receipt can never print" % name)
    return problems


def rule_persistent_guards(code: str) -> list:
    """The arm probes the capability, and refuses instead of guessing."""
    problems = []
    arm = body_of(code, "void armPersistent(")
    if not arm:
        problems.append("armPersistent() is missing")
        return problems
    if "cudaOccupancyMaxActiveBlocksPerMultiprocessor" not in arm:
        problems.append(
            "armPersistent does not call "
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor.  A cooperative grid "
            "that is not co-resident fails with "
            "cudaErrorCooperativeLaunchTooLarge; the arm has to know before it "
            "launches, not after.")
    if "cooperativeLaunch" not in arm:
        problems.append("armPersistent does not consult "
                        "cudaDeviceProp::cooperativeLaunch")
    if "use_graph" not in arm:
        problems.append(
            "armPersistent does not refuse when the captured graph is armed.  A "
            "cooperative launch cannot be recorded into a stream capture, so "
            "PERSISTENT and OUTER_GRAPH are MUTUALLY EXCLUSIVE and one of them "
            "has to decline by name.")
    if "PersistentRefusal::OuterGraphActive" not in arm:
        problems.append("armPersistent does not name the graph exclusion refusal")
    if "slots != 1" not in arm:
        problems.append(
            "armPersistent does not refuse a batch.  One grid barrier spans the "
            "batch axis too, so a lane that halts while its neighbours do not "
            "would strand the grid for ever.")
    launch = body_of(code, "bool enqueuePersistentIteration(")
    if not launch:
        problems.append("enqueuePersistentIteration() is missing")
        return problems
    if "cudaLaunchCooperativeKernel" not in launch:
        problems.append("enqueuePersistentIteration does not use "
                        "cudaLaunchCooperativeKernel")
    if "<<<" in strip_comments(launch):
        problems.append(
            "enqueuePersistentIteration uses a `<<<` launch.  grid.sync() is "
            "undefined unless the kernel came in through "
            "cudaLaunchCooperativeKernel.")
    if "graphCaptureActive" not in launch:
        problems.append(
            "enqueuePersistentIteration does not re-test for an open capture at "
            "launch time; the stand-up test cannot see a capture the sweep "
            "segment opens later")
    if "persistent_armed   = false;" not in launch and \
            "persistent_armed = false;" not in launch:
        problems.append(
            "a refused cooperative launch does not latch the arm off; the refusal "
            "would then be paid once per iteration instead of once")
    if "return false;" not in launch:
        problems.append("enqueuePersistentIteration cannot decline")
    caller = strip_comments(body_of(code, "void enqueue_iteration("))
    if "enqueuePersistentIteration(" not in caller:
        problems.append("enqueue_iteration never offers the iteration to the "
                        "persistent arm")
    if "dot(r0, r, kRhoNew);" not in caller:
        problems.append(
            "enqueue_iteration no longer holds the reference launch chain; the "
            "arm that mask 0 -- and every refusal -- falls back to must stay in "
            "the same function as the one that replaces it")
    # The note is the banner block that opens the whole persistent section, so
    # look in the window above the struct rather than at the doc comment glued
    # to it.
    hit = code.find("struct PersistentBicgParams")
    note = code[max(0, hit - 6000):hit] if hit >= 0 else ""
    if "ORDER-PRESERVATION NOTE" not in note:
        problems.append(
            "the persistent arm has no ORDER-PRESERVATION NOTE above it -- a "
            "bit-identity argument that lives only in a reviewer's head cannot "
            "be re-checked when a reference kernel next changes")
    return problems


def rule_receipts(code: str) -> list:
    problems = []
    if "[RASBERY][CMFD][OCCUPANCY]" not in code:
        problems.append("the [RASBERY][CMFD][OCCUPANCY] receipt is missing")
        return problems
    emitter = body_of(code, "inline void reportCmfdOccupancy(")
    if not emitter:
        problems.append("reportCmfdOccupancy() is missing")
        return problems
    for field in OCCUPANCY_FIELDS:
        if ('\\"%s\\"' % field) not in emitter:
            problems.append("the [CMFD][OCCUPANCY] receipt does not carry %s"
                            % field)
    forbidden = ("cudaMemcpy", "cudaLaunch", "Synchronize", "<<<")
    stripped = strip_comments(emitter)
    for token in forbidden:
        if token in stripped:
            problems.append(
                "reportCmfdOccupancy() contains `%s`: a receipt that touches the "
                "stream is not an observation of the run, it is part of it"
                % token)
    if "exchange(true" not in emitter:
        problems.append(
            "reportCmfdOccupancy() is not one-shot; it is called from "
            "enqueue_iteration, which runs 74k times")
    census = body_of(code, "static void reportCmfdGraphCensus(")
    for field in GRAPH_FIELDS:
        if ('\\"%s\\"' % field) not in census:
            problems.append("the [CMFD][GRAPH] receipt does not carry %s" % field)
    model = body_of(code, "int launchesPerIteration() const")
    if not model:
        problems.append("launchesPerIteration() is missing; the receipt cannot "
                        "state what one iteration costs in dispatches")
    elif "rb_sweeps" not in model or "fuse_dot" not in model:
        problems.append(
            "launchesPerIteration() is not derived from the same structural "
            "inputs the graph census uses (rb_sweeps, the fuse bits, the scalar "
            "tail); two answers to one question is how a census drifts")
    return problems


def rule_doc(doc: str) -> list:
    problems = []
    for needle in ("B0", "N1", "RASBERY_GPU_CMFD_BLOCK",
                   "RASBERY_GPU_CMFD_PERSISTENT", "1f36e75dc00ed2b4", "h5diff",
                   "nsys"):
        if needle not in doc:
            problems.append("the WP17 doc does not mention `%s`" % needle)
    if "34" not in doc or "133" not in doc or "265" not in doc:
        problems.append("the WP17 doc does not tabulate the 34 / 133 / 265 block "
                        "counts the knob produces")
    return problems


# ---------------------------------------------------------------------------
# Negative controls.  Each rule is fed a snippet that violates exactly it.
# ---------------------------------------------------------------------------

CONTROLS = (
    ("default unchanged",
     rule_default_unchanged,
     'inline int cmfdBlockThreads() {\n'
     '    const char* v = std::getenv("RASBERY_GPU_CMFD_BLOCK");\n'
     '    return v ? std::atoi(v) : 64;\n'
     '}\n'
     'int cmfd_block_threads() const { return cmfd_block; }\n'
     'int cmfd_node_blocks() const { return (nxyz + 255) / 256; }\n'
     'int cmfd_vector_blocks() const { return (n + 255) / 256; }\n'),
    ("five classes retargeted",
     rule_five_classes,
     "colored_block_sweep<<<node_grid(), block_size, 0, stream>>>(x);\n"),
    ("colour order preserved",
     rule_colour_order,
     "__global__ void colored_block_sweep(int nxyz) {\n"
     "    __shared__ double partial[256];\n"
     "    const int l = blockIdx.x * blockDim.x + threadIdx.x;\n"
     "    if (blockIdx.x % 2 != target_color) return;\n"
     "}\n"
     "void precondition_sweeps(const double* b, double* x) {\n"
     "    colored_block_sweep<<<g, t>>>(0);\n"
     "}\n"),
    ("reduction order preserved",
     rule_reduction_order,
     "reduce_dot_stage1<<<cmfd_node_grid(), cmfd_block_threads(), 0, stream>>>(n);\n"
     "__device__ inline void persistentDotStage1(int n) {\n"
     "    const int chunk = (n + static_cast<int>(gridDim.x) - 1) /"
     " static_cast<int>(gridDim.x);\n"
     "}\n"
     "__device__ inline double persistentFold(const int blocks, const double* pm) {\n"
     "    double sum = 0.0;\n"
     "    for (int i = blocks - 1; i >= 0; --i) sum += pm[i];\n"
     "    return sum;\n"
     "}\n"),
    ("refusal ladder",
     rule_refusal_ladder,
     "enum class PersistentRefusal : int {\n"
     "    None = 0,\n"
     "    ArmOff,\n"
     "    Count\n"
     "};\n"
     "inline const char* persistentRefusalName(PersistentRefusal r) {\n"
     "    switch (r) {\n"
     "        default: return \"?\";\n"
     "    }\n"
     "}\n"),
    ("persistent guards",
     rule_persistent_guards,
     "void armPersistent(const cudaDeviceProp& p) { persistent_armed = true; }\n"
     "bool enqueuePersistentIteration(int a, int f) {\n"
     "    bicg_iteration_persistent<<<g, t, 0, stream>>>(args);\n"
     "    return true;\n"
     "}\n"
     "void enqueue_iteration(int allow_halt, int force_halt = 0) {\n"
     "    enqueuePersistentIteration(allow_halt, force_halt);\n"
     "}\n"
     "struct PersistentBicgParams { int nxyz; };\n"),
    ("receipts",
     rule_receipts,
     'inline void reportCmfdOccupancy(int b) {\n'
     '    std::cout << "[RASBERY][CMFD][OCCUPANCY] {\\"block_threads\\":" << b'
     ' << "}";\n'
     '}\n'
     'static void reportCmfdGraphCensus(const char* tag) {\n'
     '    line << "[RASBERY][CMFD][GRAPH] {\\"nodes\\":" << count;\n'
     '}\n'),
)

DOC_CONTROL = "WP17 was a good idea and we did it."


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
    try:
        doc = read(DOC)
    except OSError:
        doc = ""

    problems = []
    problems += rule_default_unchanged(code)
    problems += rule_five_classes(code)
    problems += rule_colour_order(code)
    problems += rule_reduction_order(code)
    problems += rule_refusal_ladder(code)
    problems += rule_persistent_guards(code)
    problems += rule_receipts(code)
    if not doc:
        problems.append("docs/WP17_CMFD_OCCUPANCY_20260830_KO.md is missing: the "
                        "partition analysis and the 238 runbook are the "
                        "deliverable, not a nicety")
    else:
        problems += rule_doc(doc)

    controls = self_test()
    if problems or controls:
        print("FAIL: CMFD occupancy contract")
        for problem in problems:
            print("  - " + problem)
        for control in controls:
            print("  - " + control)
        return 1

    print("PASS: CMFD occupancy contract")
    print("  re-blocked classes (B0): " + ", ".join(NODE_CLASSES + VECTOR_CLASSES))
    print("  left alone (N1 if moved): " + ", ".join(REDUCE_CLASSES))
    print("  default: RASBERY_GPU_CMFD_BLOCK unset -> cmfd_block_threads() == "
          "block_size (34/67 blocks, unchanged)")
    print("  persistent arm: %d named refusals, cooperative launch guarded, "
          "mutually exclusive with the captured graph"
          % len(re.findall(r"case PersistentRefusal::", code)))
    print("  negative controls: %d, all fired" % (len(CONTROLS) + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
