#!/usr/bin/env python3
"""WP21-D: the pole-parallel CRAM node mapping (src/CudaCramBackend.cu).

The 238 ncu profile's block 39 is kPredictor/kCorrector: 133 blocks x 64
threads, 4.38 / 4.49 ms per launch, DRAM at 0.45 % of peak, 4.15 % warps active,
occupancy capped at 14 by 70 registers.  Nothing there is bandwidth or FLOPs.
One thread walked a whole node -- 4 CRAM poles x up to 64 Gauss-Seidel sweeps
over 36 unknowns -- so the kernel was a serial latency chain on 2 % of the card.

WP21-D widens the NODE and not the SWEEP.  Gauss-Seidel stays sequential in
`row` and its inner product stays sequential in `i`; what moves is the pole
loop, because order 8 is four INDEPENDENT complex solves whose only coupling is
milk.h's closing `accum[row] += x_pole[row]` -- a fixed left-to-right walk over
pole = 0, 1, 2, 3.  Four lanes per node, lane == pole, and that walk re-formed
with __shfl_sync in the same order is the SAME sequence of double additions.

Which is a claim, and a claim about arithmetic order is exactly the kind that a
numerical A/B keeps confirming right up until the day it stops.  So it is
asserted here, eleven ways:

  1. LANE PER POLE, BY CONSTRUCTION.  kLanesPerNode is defined AS kPoleCount, so
     a fifth pole cannot leave a four-lane group behind, and the node/pole split
     of the thread id is spelled in terms of it rather than in literals.

  2. THE GROUP IS FOUR LANES OF ONE WARP, AND EVERY WARP OP NAMES IT.  base is
     `(threadIdx.x & 31) & ~(kLanesPerNode - 1)` and mask is those four lanes;
     no legacy __shfl/__ballot, no __activemask() -- an implicit mask is how a
     group op silently starts reading a neighbouring node's pole.

  3. THE POLE-SUM ORDER IS THE SERIAL ONE.  The pole-parallel accumulation is
     the serial driver's loop nest with one substitution: p outer, row inner,
     into a zero-initialised accumulator, source lane `base + p`.  Both drivers
     then finish through the SAME cramFinish, so the closing
     `alpha0 * N + 2 * accum` is one piece of code and not two.

  4. THE STATUS IS THE FIRST FAILING POLE'S.  The serial loop stopped at the
     first pole that failed; the parallel one runs all four, so it must pick the
     lowest by __ffs over the group ballot -- and must not count the sweeps of a
     pole the serial arm never reached, or gs_iters_mean stops comparing.

  5. THE STORES ARE NODE-INNERMOST.  `iden_out[iso * nxyz + l]` was ALREADY the
     coalesced layout -- 32 lanes, one isotope, 256 contiguous bytes, the
     8-sectors-per-request floor for fp64 -- so ncu's 8.7 st sectors/request was
     never this store but the local-memory traffic of `cval`/`ccol`.  This rule
     exists so that a future "SoA transpose" cannot be applied to the one array
     that is already SoA, and so the WP15.1 consumer's layout stays untouched.

  6. ONE PUBLISHER PER NODE.  Only lane 0 runs the epilogue.  Four lanes writing
     the same node's outputs is not a data race that shows up in a comparison.

  7. THE TWO ARMS SHARE THE PROLOGUE AND THE EPILOGUE.  An A/B whose arms differ
     in the condensation as well as in the solve measures two things at once.

  8. THE DEFAULT IS THE WIDE ARM, AND AN UNKNOWN SPELLING SAYS SO.  Silently
     running a mapping other than the one the operator typed is the failure this
     knob exists to avoid.

  9. THE ENUM IS EXACTLY TWO ARMS.  There is no `jacobi` arm: a Jacobi sweep
     changes Gauss-Seidel semantics and would be N1, and lane-per-pole preserves
     the order without it.  A third arm has to be declared, not smuggled.

 10. THE PER-LAUNCH TIMER IS BEHIND A FLAG AND MEASURES THE KERNEL.  wall_ms is
     the whole call; ncu's block 39 is not.  Two extra event records per launch
     are two extra ordering points, so they are opt-in.

 11. THE MAPPING IS IN THE RECEIPT AND *NOT* IN kArmEnv.  It does not move the
     trajectory (it is B0 against serial), exactly like RASBERY_GPU_CRAM_BLOCK
     and unlike RASBERY_GPU_CRAM -- but it does move the time, and a per-launch
     figure quoted without the mapping it was measured under is unattributable.

Every rule runs against a deliberately broken copy of the same text, so a rule
that has stopped discriminating fails loudly instead of passing vacuously.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "cu": "src/CudaCramBackend.cu",
    "hdr": "src/CudaCramBackend.h",
    "driver": "src/Driver.h",
}


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise AssertionError(f"missing file: {rel}")
    return path.read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def region(text: str, start: str, end: str, what: str) -> str:
    i = text.find(start)
    assert i >= 0, f"{what}: opening marker {start!r} not found"
    j = text.find(end, i + len(start))
    assert j >= 0, f"{what}: closing marker {end!r} not found"
    return text[i : j + len(end)]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def r_lane_per_pole(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "constexpr int kLanesPerNode = kPoleCount;" in cu, (
        "kLanesPerNode must be DEFINED as kPoleCount.  A literal 4 would let a "
        "different CRAM order leave a four-lane group solving three poles and "
        "shuffling a fourth that nobody wrote."
    )
    assert cu.count("const int l    = gtid / kLanesPerNode;") == 2, \
        "both pole-parallel kernels must derive the node from kLanesPerNode"
    assert cu.count("const int pole = gtid % kLanesPerNode;") == 2, \
        "both pole-parallel kernels must derive the pole from kLanesPerNode"
    assert "return (variant == Variant::kPole4) ? kLanesPerNode : 1;" in cu, \
        "lanesPerNode() must report the mapping the launch actually uses"
    assert "static_cast<long long>(nxyz_in) * lanesPerNode();" in cu, (
        "the grid must be sized in LANES, not nodes -- a grid still sized in "
        "nodes would leave three quarters of every deck unsolved"
    )
    assert cu.count("const int blocks = s.gridFor(v.nxyz);") == 2, \
        "both entry points must size their grid through gridFor()"


def r_group_mask(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert cu.count("const int          base = (threadIdx.x & 31) & ~(kLanesPerNode - 1);") == 2, \
        "the group's first lane must be derived from the WARP lane, not the block index"
    assert cu.count("const unsigned int mask = ((1u << kLanesPerNode) - 1u) << base;") == 2, \
        "the mask must be exactly the group's lanes"
    # Every warp-collective names that mask explicitly.
    for m in re.finditer(r"__(?:shfl|ballot)\w*_sync\(\s*([A-Za-z_]\w*)", cu):
        assert m.group(1) == "mask", \
            f"a warp collective is synchronised on {m.group(1)!r}, not the group mask"
    for banned in ("__activemask(", "__shfl(", "__shfl_up(", "__shfl_down(",
                   "__shfl_xor(", "__ballot(", "__any(", "__all("):
        assert banned not in cu, (
            f"{banned} has no mask: on independent thread scheduling it reads "
            "whatever lanes happen to be converged, which is a neighbouring "
            "node's pole as often as not"
        )


def r_pole_sum_order(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    serial = region(cu, "unsigned int cramSolveNode(", "\n}\n", "cramSolveNode")
    par = region(cu, "unsigned int cramSolveNodePole4(", "\n}\n", "cramSolveNodePole4")

    for body, what in ((serial, "serial"), (par, "pole-parallel")):
        assert "for (int row = 0; row < n; ++row) accr[row] = 0.0;" in body, \
            f"{what}: the accumulator must start at 0.0 for every row, as milk.h's std::fill does"
        assert "return cramFinish(x, iden, accr);\n}" in body, \
            f"{what}: the closing alpha0*N + 2*accum must go through the shared cramFinish"

    assert "for (int row = first; row < n; ++row) accr[row] += xr[row];" in serial, \
        "the serial driver's pole accumulation moved"
    i_pole = serial.find("for (int pole = 0; pole < kPoleCount; ++pole) {")
    assert 0 <= i_pole < serial.find("accr[row] += xr[row];"), \
        "the serial accumulation must sit INSIDE the ascending pole loop"

    # The parallel accumulation is that nest with one substitution.
    nest = ("    for (int p = 0; p < kPoleCount; ++p) {\n"
            "        for (int row = first; row < n; ++row)\n"
            "            accr[row] += __shfl_sync(mask, xr[row], base + p);\n"
            "    }\n")
    assert nest in par, (
        "the pole-parallel accumulation is no longer the serial loop nest with "
        "pole p's value fetched from lane base + p -- ascending p outside, "
        "ascending row inside.  Any other nesting or any other source lane is a "
        "different sequence of double additions, and the arm stops being B0."
    )
    assert cu.count("return cramFinish(x, iden, accr);") == 2, \
        "cramFinish must be the ONE closing loop both arms run"


def r_first_failing_pole(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    par = region(cu, "unsigned int cramSolveNodePole4(", "\n}\n", "cramSolveNodePole4")
    assert "const unsigned int failed      = __ballot_sync(mask, st != kOk) & mask;" in par, \
        "the group must ballot its per-pole status"
    assert "const int lane = __ffs(static_cast<int>(failed)) - 1;" in par, (
        "the node's status must be the LOWEST failing lane's.  The serial loop "
        "returned at the first failing pole, so a pole-3 non-convergence behind "
        "a pole-0 zero diagonal must still decline as a zero diagonal"
    )
    assert "last_run       = lane - base;" in par, \
        "the sweep count must stop at the pole the serial arm stopped at"
    assert "if (p < last_run || (p == last_run && sp != kZeroDiag))" in par, (
        "a zero diagonal refuses BEFORE its first sweep and contributes none; "
        "counting it would move gs_iters_mean, which is the receipt observable "
        "that says the device solved the host's iteration"
    )
    assert "if (node_status != kOk) return node_status;" in par, \
        "a failed node must not reach the accumulation with lanes holding garbage"


def r_stores_soa(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    subs = re.findall(r"x\.iden_(?:in|out|pred)\[([^\]]+)\]", cu)
    assert subs, "the inventory load/store subscripts vanished"
    for s in subs:
        assert re.search(r"\*\s*nxyz\s*\+\s*(?:l|node)\s*$", s.strip()), (
            f"inventory subscript {s.strip()!r} is not node-innermost.  "
            "[iso * nxyz + l] is what makes a warp's write 32 contiguous doubles "
            "-- 256 bytes, the 8-sectors-per-request floor for fp64 -- and it is "
            "also the layout the D2H of rows [first, niso) copies in ONE go and "
            "the layout every XSSet host reader and WP15.1's device consumer "
            "already agree on.  There is no transpose to do here."
        )
    # A bare `* x.niso +` elsewhere is legitimate and deliberately NOT flagged:
    # dep_decay and dep_trans are niso x niso row-major LIBRARY tables, not
    # per-node arrays, and they have no node index to put innermost.


def r_one_publisher(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "if (pole == 0) predictorEpilogue(x, l, iden, cond4, sumflux, status, gs);" in cu, \
        "the predictor's node outputs must be published by ONE lane of the group"
    assert "if (pole == 0) correctorEpilogue(x, l, iden, cond4, sumflux, burn, status, gs);" in cu, \
        "the corrector's node outputs must be published by ONE lane of the group"
    # And the counters go with them: four lanes each doing atomicAdd(stats[1])
    # would quadruple gs_iters without changing a single density.
    for fn in ("predictorEpilogue", "correctorEpilogue"):
        body = region(cu, f"void {fn}(const DevCtx& x", "\n}\n", fn)
        assert "atomicAdd(&x.stats[1], gs);" in body, \
            f"{fn} must be the only place the sweep counter is accumulated"


def r_shared_body(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    for call, what in (
        ("predictorPrologue(x, l, cond4, iden, &sumflux);", "predictor prologue"),
        ("correctorPrologue(x, l, cond4, iden, &sumflux, &burn);", "corrector prologue"),
        ("predictorEpilogue(x, l, iden, cond4, sumflux, status, gs);", "predictor epilogue"),
        ("correctorEpilogue(x, l, iden, cond4, sumflux, burn, status, gs);", "corrector epilogue"),
        ("cramBuildSplit(x, cond4, sumflux, dt, cval, ccol, cend, mdiag);", "matrix split"),
    ):
        assert cu.count(call) == 2, (
            f"the {what} is not shared by both arms (found {cu.count(call)} call sites, "
            "expected one serial and one pole-parallel).  An A/B whose arms differ "
            "outside the solve measures two things at once"
        )


def r_default_pole4(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert "Variant     variant      = Variant::kPole4;" in cu, \
        "the pole-parallel mapping is the default arm"
    assert 'if (want == "serial") {' in cu, \
        "RASBERY_GPU_CRAM_PARALLEL=serial must select the pre-WP21-D body"
    assert 'want == "pole" || want == "pole4"' in cu, \
        "RASBERY_GPU_CRAM_PARALLEL=pole4 must select the wide body"
    assert '", using pole4)"' in cu, (
        "an unrecognised RASBERY_GPU_CRAM_PARALLEL must SAY it fell back to the "
        "default: silently running a mapping other than the one the operator "
        "typed is the failure this knob exists to prevent"
    )


def r_two_arms(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    body = region(cu, "enum class Variant : int {", "};", "Variant")
    names = set(re.findall(r"\b(k\w+)\s*=", body))
    assert names == {"kSerial", "kPole4"}, (
        f"Variant now declares {sorted(names)}.  A Jacobi sweep inside a pole "
        "reassociates Gauss-Seidel and is N1; lane-per-pole preserves the order "
        "without it, so there is no jacobi arm.  A third arm has to be declared "
        "here, gated, and receipted as N1 -- not smuggled in as a default."
    )
    hdr = src["hdr"]
    assert "There is no `jacobi` arm" in hdr, \
        "the header must record why no N1 arm is offered"


def r_timing_flag(src: dict[str, str]) -> None:
    cu = strip_comments(src["cu"])
    assert cu.count("if (s.timing) cudaEventRecord(s.ev_k0, s.stream);") == 2, \
        "the per-launch timer must be opt-in at BOTH launch sites"
    assert cu.count("if (s.timing) cudaEventRecord(s.ev_k1, s.stream);") == 2, \
        "the per-launch timer must close at BOTH launch sites"
    assert cu.count("s.noteLaunch();") == 2, \
        "every launch must be counted, including the ones whose nodes declined"
    assert 'truthy(std::getenv("RASBERY_GPU_CRAM_TIMING"))' in cu, \
        "the timer is gated on RASBERY_GPU_CRAM_TIMING"
    assert "if (!_impl->timing || _impl->n_launches == 0) return -1.0;" in cu, (
        "launch_us_mean must be -1 when nothing measured it: 0.0 reads as a "
        "kernel that took no time"
    )
    # ev_k1 is only readable after a sync, and noteLaunch must come after one.
    for entry in ("bool CramBackend::predictor(", "bool CramBackend::corrector("):
        i = cu.find(entry)
        assert i >= 0, f"{entry} vanished"
        body = cu[i : cu.find("\n}\n", i)]
        sync = body.find('"stats drain"')
        note = body.find("s.noteLaunch();")
        assert 0 <= sync < note, \
            f"{entry}: cudaEventElapsedTime is read before the sync that makes it readable"


def r_receipt_and_arm(src: dict[str, str]) -> None:
    driver = strip_comments(src["driver"])
    for field in ('\\"kernel_variant\\":', '\\"lanes_per_node\\":',
                  '\\"launches\\":', '\\"launch_us_mean\\":'):
        assert field in driver, f"the [RASBERY][CRAM_GPU] receipt is missing {field}"
    for call in ("c.kernelVariant()", "c.lanesPerNode()", "c.launches()",
                 "c.launchUsMean()"):
        assert call in driver, f"the receipt declares a field it never fills: {call}"
    hdr = src["hdr"]
    for decl in ("const std::string& kernelVariant() const;",
                 "int lanesPerNode() const;",
                 "unsigned long long launches() const;",
                 "double launchUsMean() const;"):
        assert decl in hdr, f"CudaCramBackend.h does not declare {decl!r}"

    m = re.search(r"kArmEnv\[\]\s*=\s*\{(.*?)\};", driver, re.S)
    assert m, "trajectory::kArmEnv vanished"
    arms = m.group(1)
    assert "RASBERY_GPU_CRAM" in arms, \
        "RASBERY_GPU_CRAM must stay an arm knob -- depletion output IS the next statepoint"
    for knob in ("RASBERY_GPU_CRAM_PARALLEL", "RASBERY_GPU_CRAM_BLOCK"):
        assert f'"{knob}"' not in arms, (
            f"{knob} is in trajectory::kArmEnv.  It selects a MAPPING, not a "
            "method: the pole-parallel arm is B0 against the serial one, so two "
            "runs that differ only in it are the same trajectory and must "
            "compare as the same arm.  Putting it here would fork the case key "
            "and throw away every cached result for no physical reason."
        )


RULES = [
    ("lane-per-pole", r_lane_per_pole, "cu",
     ("constexpr int kLanesPerNode = kPoleCount;",
      "constexpr int kLanesPerNode = 4;")),
    ("group-mask", r_group_mask, "cu",
     ("const unsigned int mask = ((1u << kLanesPerNode) - 1u) << base;",
      "const unsigned int mask = __activemask();")),
    ("pole-sum-order", r_pole_sum_order, "cu",
     ("            accr[row] += __shfl_sync(mask, xr[row], base + p);",
      "            accr[row] += __shfl_sync(mask, xr[row], base + (kPoleCount - 1 - p));")),
    ("first-failing-pole", r_first_failing_pole, "cu",
     ("const int lane = __ffs(static_cast<int>(failed)) - 1;",
      "const int lane = 31 - __clz(static_cast<int>(failed));")),
    ("stores-soa", r_stores_soa, "cu",
     ("x.iden_out[static_cast<size_t>(i) * nxyz + l] = iden[i];",
      "x.iden_out[static_cast<size_t>(l) * x.niso + i] = iden[i];")),
    ("one-publisher", r_one_publisher, "cu",
     ("    if (pole == 0) predictorEpilogue(x, l, iden, cond4, sumflux, status, gs);",
      "    predictorEpilogue(x, l, iden, cond4, sumflux, status, gs);")),
    ("shared-body", r_shared_body, "cu",
     ("    correctorPrologue(x, l, cond4, iden, &sumflux, &burn);\n\n    unsigned long long gs = 0;",
      "\n\n    unsigned long long gs = 0;")),
    ("default-pole4", r_default_pole4, "cu",
     ("Variant     variant      = Variant::kPole4;",
      "Variant     variant      = Variant::kSerial;")),
    ("two-arms", r_two_arms, "cu",
     ("    kPole4  = 1, ///< default: kLanesPerNode lanes per node, lane == pole\n};",
      "    kPole4  = 1, ///< default: kLanesPerNode lanes per node, lane == pole\n"
      "    kJacobi = 2, ///< lanes over isotopes\n};")),
    ("timing-flag", r_timing_flag, "cu",
     ("    if (s.timing) cudaEventRecord(s.ev_k0, s.stream);",
      "    cudaEventRecord(s.ev_k0, s.stream);")),
    ("receipt-and-arm", r_receipt_and_arm, "driver",
     ('    "RASBERY_GPU_CRAM",\n',
      '    "RASBERY_GPU_CRAM",\n    "RASBERY_GPU_CRAM_PARALLEL",\n')),
]


def main() -> int:
    failures: list[str] = []
    try:
        src = {k: read(v) for k, v in FILES.items()}
    except AssertionError as exc:
        print(f"CRAM parallel contract: FAIL {exc}")
        return 1

    for name, rule, _target, _control in RULES:
        try:
            rule(src)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")

    for name, rule, target, (needle, replacement) in RULES:
        broken = dict(src)
        if needle not in broken[target]:
            failures.append(
                f"{name}: negative control is stale, {needle!r} not in {FILES[target]}")
            continue
        broken[target] = broken[target].replace(needle, replacement, 1)
        try:
            rule(broken)
        except AssertionError:
            continue
        failures.append(f"{name}: negative control PASSED the rule -- the rule is vacuous")

    if failures:
        print("CRAM parallel contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"CRAM parallel contract: PASS ({len(RULES)} rules, each with a negative control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
