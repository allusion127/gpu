#!/usr/bin/env python3
"""Source contract for WP5 stage B -- the FlatXS CTA-per-node kernel.

WHAT THIS FILE CAN AND CANNOT DECIDE.

It can decide the SHAPE of the change, and for this work package the shape IS
the bit-identity argument.  kernelFlatXsCta claims to write the same bytes as
kernelFlatXs, and that claim rests on exactly three structural properties
(spelled out at the top of src/FlatXsCtaKernel.cuh):

  P1  fixed lane ownership -- every workspace element's accumulation chain
      lives inside ONE lane, because every phase walks its ordinal space with
      the same `for (q = tid; q < N; q += T)` form;
  P2  the isotope fold is never parallelised -- 39 dependent single-rounding
      FMAs per output, one lane per output, iso ascending, no tree, no shuffle,
      no atomic;
  P3  the per-delta scalars are recomputed per lane, not broadcast, so the
      stream loop carries no barrier at all.

Every one of those can be broken by an edit that still compiles, still runs,
and still produces plausible cross sections -- on the deck the author happened
to try.  A parallel tree reduction over iso would look like a tidy
optimisation and would silently stop reproducing the reference, because
fma(a,b,c) is single-rounding and the reference's chain never materialises the
partial products a tree would sum.  So those three properties are asserted
here, in pure python, on every run.

IT CANNOT DECIDE THE CLAIM ITSELF.  "kernelFlatXsCta writes the same bytes as
kernelFlatXs" is a 238 GATE, not a source property: it needs nvcc, a
RASBERY_FLATXS_DUMP capture and a device.  test/flatxs_device_replay.cu --cta
is that gate and docs/WP5_FLATXS_CTA_20260831_KO.md is its runbook.  What this
file does is make sure that when the gate runs, it runs against the code the
doc describes -- and that the reference arm it is scored against is still
there, still reachable, and still the default.

NEGATIVE CONTROLS.  Every structural rule is also run against a synthetic
snippet that violates it, so a rule that has quietly stopped matching anything
fails here rather than passing forever.

Pure python, no build, no device.

Run:  python tools/test_flatxs_cta_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []
checks = 0


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def strip_comments(text: str) -> str:
    """Rules are about CODE.  These headers describe the banned constructs by
    name ("no atomicAdd", "no tree reduction"), so a rule that searched the raw
    text would fire on its own documentation."""
    return re.sub(r"//[^\n]*", "", text)


def blank_subscripts(text: str) -> str:
    """Erase the contents of every [...] so INDEX arithmetic is not mistaken for
    floating-point arithmetic.  `v.ref_lmp[t][ig * nxyz + l]` is a load, not a
    multiply-add, and every index expression in these kernels lives inside
    brackets."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\[[^\[\]]*\]", "[]", text)
    return text


def check(ok: bool, what: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(what)


CTA = read("src", "FlatXsCtaKernel.cuh")
REF = read("src", "FlatXsKernel.h")
XSR = read("src", "XsReconKernel.h")
BACKEND = read("src", "CudaXsReconBackend.cu")
BACKEND_H = read("src", "CudaXsReconBackend.h")
STUB = read("src", "CudaXsReconBackendStub.cpp")
DRIVER = read("src", "Driver.h")
REPLAY = read("test", "flatxs_device_replay.cu")
CMAKE = read("CMakeLists.txt")
DOC = read("docs", "WP5_FLATXS_CTA_20260831_KO.md")


def body_of(text: str, start_marker: str, end_marker: str) -> str:
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    return text[a:b]


# The two bodies being compared, sliced so a site in a neighbouring helper
# cannot be mistaken for a site in the kernel.
REF_BODY = body_of(REF, "inline void flatxsSolveNode(const FlatXsView& v, int i,",
                   "/// Host-side helper for the NodeSpectralIndex")
CTA_BODY = body_of(CTA, "flatxsSolveNodeCta(const FlatXsView& v, int i,",
                   "/// One CTA per node.")
# The delta-stream loop alone: P3's "no barriers in here" is about this region.
CTA_STREAM = body_of(CTA_BODY, "for (int s = s0; s < s1; ++s) {",
                     "// --- 3. Densities")


# ---------------------------------------------------------------------------
# 1. THE FLAG IS ON BY DEFAULT (SINCE v5) AND DECLARED WHERE ARMS ARE DECLARED.
#
# The rule this replaces was "default-on would ship the claim unchecked".  The
# claim is no longer unchecked: 238 priced the arm under PROD on 2026-08-30
# (pricing block 12 -- h5diff 0 differences, digest 1f36e75dc00ed2b4 / 4377 on
# both arms, 12.102 -> 11.189 s) and 181 gated it on a second toolchain
# (gates block 5 -- cta_vs_ref_mismatches 0 at 64/128/256 and byte-identical
# production h5 at every thread count).
#
# So what this rule now holds is the OTHER half, which never changed: the off
# switch has to survive, because RASBERY_GPU_FLATXS_CTA=0 is the live reference
# the bit-identity claim is measured against.
# ---------------------------------------------------------------------------
def rule_flag_default_on(backend: str) -> bool:
    return '!envFlagDisabled("RASBERY_GPU_FLATXS_CTA")' in backend


check(
    rule_flag_default_on(BACKEND),
    "RASBERY_GPU_FLATXS_CTA is resolved with !envFlagDisabled, i.e. absent means ON "
    "and =0 is the off switch (envFlagEnabled would put it back to default-OFF)",
)
check(
    "bool rasberyGpuFlatXsCtaEnabled() { return false; }" in STUB,
    "the no-CUDA stub reports the CTA arm disabled",
)
check(
    "int rasberyGpuFlatXsCtaThreads()" in STUB,
    "the no-CUDA stub defines the threads accessor too (a call site must never "
    "need an #ifdef)",
)
check(
    "bool rasberyGpuFlatXsCtaEnabled();" in BACKEND_H
    and "int rasberyGpuFlatXsCtaThreads();" in BACKEND_H,
    "both accessors are declared in CudaXsReconBackend.h beside the other arm flags",
)
for env in ('"RASBERY_GPU_FLATXS_CTA",', '"RASBERY_GPU_FLATXS_CTA_THREADS",'):
    check(
        env in DRIVER,
        "%s is listed in Driver.h's kArmEnv -- a knob that selects which kernel "
        "runs has to be declared where the receipt reads the arms from" % env,
    )


# ---------------------------------------------------------------------------
# 2. THE REFERENCE ARM SURVIVES AND STAYS THE DEFAULT.
#
# WP7-B's rule and the reason a B0 replay is worth anything: the code CTA=0
# runs must still be present and still be reached, so the comparison is against
# LIVE CODE and not against a memory of it.
# ---------------------------------------------------------------------------
check(
    "__global__ void kernelFlatXs(fxs::FlatXsView v)" in BACKEND,
    "the thread-per-node reference kernel is still defined (CTA=0 launches it)",
)
check(
    "fxs::flatxsSolveNode(v, i, fxs::StaticForms{});" in BACKEND,
    "the reference kernel still calls the shared body -- the mask-0 reference is "
    "the SAME code the host replay scores",
)


def rule_dispatch_is_guarded(arm: str) -> bool:
    return ("static const bool cta = rasberyGpuFlatXsCtaEnabled();" in arm
            and re.search(r"if\s*\(cta\)\s*\{", arm) is not None
            and "} else {" in arm
            and "kernelFlatXs<<<grid, block, 0, d.stream>>>(v);" in arm)


SOLVE = body_of(BACKEND, "bool XsReconBackend::solveFlatXs(",
                "bool XsReconBackend::solveNodal(")
check(
    rule_dispatch_is_guarded(SOLVE),
    "the CTA arm is one guarded branch inside solveFlatXs, the flag is read ONCE "
    "into a cached bool, and the else-branch still launches kernelFlatXs",
)
check(
    SOLVE.count("rasberyGpuFlatXsCtaEnabled()") == 1,
    "there is exactly ONE dispatch point -- a second one is a second policy",
)
# Everything downstream of the launch is arm-agnostic: the two arms write the
# same bytes, so the downloads and the residency bookkeeping must not branch.
POST = SOLVE[SOLVE.index("--- results the host needs back"):]
check(
    "cta" not in re.sub(r"//[^\n]*", "", POST),
    "nothing after the launch knows which arm ran (the downloads, the residency "
    "generations and the counters are identical by construction)",
)


# ---------------------------------------------------------------------------
# 3. THE SHARED WORKSPACE IS SIZED FROM THE SAME CONSTANTS AS THE REFERENCE.
#
# The reference declares bl/bls/bm/bms/iden from N_ACTIVE, NG, NLSM, NMIC,
# NMSM and NISO.  A hand-copied 702 or 7352 in the CTA header would be a
# workspace that silently stops tracking the isotope registry -- and the
# failure mode is an out-of-bounds shared write, i.e. corruption of a
# NEIGHBOURING array in the same struct, which is exactly the kind of bug a
# 0-ULP replay on one deck can miss.
# ---------------------------------------------------------------------------
WS = body_of(CTA, "struct CtaWorkspace {", "};")
EXPECTED_DIMS = {
    "bl": "N_ACTIVE * NG",
    "bls": "NLSM",
    "bm": "N_ACTIVE * NMIC",
    "bms": "NMSM",
    "iden": "NISO",
}
for name, dim in EXPECTED_DIMS.items():
    check(
        re.search(r"double\s+%s\[%s\]" % (name, re.escape(dim)), WS) is not None,
        "CtaWorkspace::%s is sized [%s] -- the same expression the reference's "
        "local array uses" % (name, dim),
    )
for name, dim in EXPECTED_DIMS.items():
    check(
        re.search(r"double\s+%s\[%s\]" % (name, re.escape(dim)), REF_BODY) is not None
        or re.search(r"double\s+%s\[%s\]" % (name, re.escape(dim)), REF) is not None,
        "the reference still declares %s[%s] -- if it moved, the CTA workspace "
        "must move with it" % (name, dim),
    )


def rule_no_literal_dims(ws: str) -> bool:
    """No array bound in the shared workspace may be a bare number."""
    return re.search(r"double\s+\w+\[\s*\d+\s*\]", ws) is None


check(rule_no_literal_dims(WS),
      "no array in CtaWorkspace has a literal bound")

# The ordinal-space extents are the loop bounds P1 depends on; they must be the
# same products, not respellings.
for name, dim in (("Q_LMP", "N_ACTIVE * NG"), ("Q_LSM", "NLSM"),
                  ("Q_MIC", "N_ACTIVE * NMIC"), ("Q_MSM", "NMSM")):
    check(
        re.search(r"constexpr int %s\s*=\s*%s\s*;" % (name, re.escape(dim)), CTA)
        is not None,
        "%s is defined as %s" % (name, dim),
    )


def rule_no_respelled_constants(cta: str) -> bool:
    """The physics constants must be USED from FlatXsKernel.h, never redefined.

    A second definition of WATER_NUMBER_DENSITY or BORON_DENSITY_FACTOR on the
    device side is how two arms end up conditioning differently while both look
    right in review.
    """
    for name in ("WATER_NUMBER_DENSITY", "BORON_DENSITY_FACTOR", "FLATXS_FORMS",
                 "NISO", "NG", "NLSM", "NMIC", "NMSM", "N_ACTIVE"):
        if re.search(r"constexpr\s+(double|int)\s+%s\s*=" % name, cta):
            return False
    return True


check(
    rule_no_respelled_constants(CTA),
    "FlatXsCtaKernel.cuh redefines no constant of FlatXsKernel.h / XsReconKernel.h",
)
check(
    "IH1" in CTA and "IB10" in CTA and "IO16" in CTA
    and re.search(r"constexpr\s+int\s+I(H1|B10|O16)\s*=", CTA) is None,
    "the light-isotope row indices are USED from the reference header, not "
    "respelled",
)


# ---------------------------------------------------------------------------
# 4. P1 -- FIXED LANE OWNERSHIP.
#
# Every ordinal loop must be `for (int <v> = tid; <v> < Q_x; <v> += T)`.  A
# loop that started at 0, or strided by something other than the block size,
# would either duplicate work across lanes or split one element's accumulation
# chain across two lanes.  The second one is the dangerous one: it does not
# crash, it just rounds differently.
# ---------------------------------------------------------------------------
ORDINAL_LOOP = re.compile(
    r"for\s*\(int\s+(\w+)\s*=\s*tid;\s*\1\s*<\s*(Q_LMP|Q_LSM|Q_MIC|Q_MSM|NISO|NG);"
    r"\s*\1\s*\+=\s*T\)")
ANY_ORDINAL_LOOP = re.compile(
    r"for\s*\(int\s+(\w+)\s*=\s*[^;]+;\s*\1\s*<\s*(Q_LMP|Q_LSM|Q_MIC|Q_MSM);")


def rule_all_ordinal_loops_are_lane_owned(body: str) -> bool:
    good = {m.start() for m in ORDINAL_LOOP.finditer(body)}
    every = {m.start() for m in ANY_ORDINAL_LOOP.finditer(body)}
    return every <= good and len(good) > 0


check(
    rule_all_ordinal_loops_are_lane_owned(CTA_BODY),
    "every Q_* loop in flatxsSolveNodeCta has the lane-owned form "
    "`for (int q = tid; q < Q_x; q += T)` (P1)",
)
# There are five phases that walk an ordinal space (gather, apply, scatter,
# macro-scalar, macro-scatter); the count keeps a deleted phase visible.
check(
    len(ORDINAL_LOOP.findall(CTA_BODY)) >= 14,
    "all five ordinal-space phases are present and lane-owned "
    "(gather 4 + apply 4 + scatter 4 + macro 2, plus the iden/XSRF walks)",
)
# WP20 appended a third template parameter (the workspace type, so the FP32 and
# FP64 arms compile from one body).  The check is on the BLOCK SIZE being a
# template parameter and on it coming FIRST, which is what makes the strides
# fold; it must not also pin the arity, or every later parameter becomes a
# contract change.  `template <int T, class POL` is therefore a prefix match.
check(
    "int T" in CTA and "template <int T, class POL" in CTA,
    "the block size is a TEMPLATE parameter, so the stride is a compile-time "
    "constant and the ladder cannot silently disagree with blockDim",
)
check(
    "__launch_bounds__(T)" in CTA,
    "the kernel carries __launch_bounds__(T) so ptxas budgets registers for the "
    "block size the launcher actually uses",
)
check(
    "const int i = static_cast<int>(blockIdx.x);" in CTA
    and "const int grid = v.n_nodes;" in CTA,
    "one CTA per node: the node ordinal is blockIdx.x and the grid is n_nodes",
)


# ---------------------------------------------------------------------------
# 5. P2 -- THE ISOTOPE FOLD IS SEQUENTIAL, ASCENDING, AND UNSPLIT.
#
# This is the rule the whole B0 claim hangs on and the one an optimiser is most
# likely to "fix".
# ---------------------------------------------------------------------------
ISO_FOLD = re.compile(r"for\s*\(int\s+iso\s*=\s*0;\s*iso\s*<\s*NISO;\s*\+\+iso\)")


def rule_iso_fold_is_sequential(body: str) -> bool:
    return len(ISO_FOLD.findall(body)) == 2


check(
    rule_iso_fold_is_sequential(CTA_BODY),
    "both macro-XS folds run `for (int iso = 0; iso < NISO; ++iso)` -- ascending, "
    "sequential, one lane per output chain (P2)",
)
# The reference has THREE such loops: the two folds plus the plain `iden`
# snapshot copy.  The CTA arm strides that copy across lanes instead -- it is a
# copy and carries no arithmetic -- which is why the counts are 3 and 2.
check(
    len(ISO_FOLD.findall(REF_BODY)) == 3,
    "the reference still runs both macro folds ascending (plus its plain iden "
    "snapshot copy) -- if IT changed, this contract is asserting the wrong order",
)


BANNED_REDUCTIONS = ("__shfl", "atomicAdd", "atomicSub", "cub::", "__reduce_add",
                     "cg::reduce", "thrust::")


def rule_no_parallel_reduction(body: str) -> bool:
    return not any(tok in body for tok in BANNED_REDUCTIONS)


check(
    rule_no_parallel_reduction(strip_comments(CTA)),
    "the CTA kernel uses no shuffle, no atomic and no library reduction: a tree "
    "over iso would not merely re-round, it would sum products the reference's "
    "single-rounding FMA chain never materialises",
)
# The reduction over ige in the XSRF pass is a plain += chain, per ig, in one
# lane, for the same reason.
check(
    re.search(r"for\s*\(int\s+ige\s*=\s*0;\s*ige\s*<\s*NG;\s*\+\+ige\)\s*\n?\s*"
              r"rf\s*\+=", CTA_BODY) is not None,
    "the XSRF sum stays a per-ig sequential `rf +=` chain",
)


# ---------------------------------------------------------------------------
# 6. CONTRACTION SITES: SAME BITS, SAME COUNT, NOTHING BARE.
#
# The device TU builds --fmad=false, so a bare `a * b + c` is an unfused
# multiply-add whatever the mined mask says.  Every multiply-add must therefore
# go through pol.ma(<bit>, ...), and the SET of bits must be exactly the
# reference's -- a site spelled with the wrong bit rounds differently at
# whichever mask entry differs.
# ---------------------------------------------------------------------------
FORM_BITS = ("F_HORNER_LMP", "F_ACC_LMP", "F_HORNER_LSM", "F_ACC_LSM",
             "F_HORNER_MIC", "F_ACC_MIC", "F_HORNER_MSM", "F_ACC_MSM",
             "F_MACRO_SCAL", "F_MACRO_SSM")


def rule_form_bits_match(ref_body: str, cta_body: str) -> bool:
    for bit in FORM_BITS:
        if ref_body.count("pol.ma(%s," % bit) != cta_body.count("pol.ma(%s," % bit):
            return False
    return True


check(
    rule_form_bits_match(REF_BODY, CTA_BODY),
    "every FormBit appears the same number of times in flatxsSolveNodeCta as in "
    "flatxsSolveNode -- same sites, same forms",
)
for bit in FORM_BITS:
    check(
        ("pol.ma(%s," % bit) in CTA_BODY,
        "contraction site %s is present in the CTA body" % bit,
    )


def rule_no_bare_multiply_add(body: str) -> bool:
    """No `x = a * b + c;` outside pol.ma.

    The three products RefreshLightIsotopes forms (2.0*nH2O, dmod*wvfr*W,
    boron*wvfr*bppm*F) are pure multiplies with no addend and are exempt by
    inspection -- the reference says so in its own comment.
    """
    stripped = blank_subscripts(strip_comments(body))
    return re.search(r"=\s*[\w.\[\]]+\s*\*\s*[\w.\[\]]+\s*\+\s*[\w.\[\]]", stripped) \
        is None


check(
    rule_no_bare_multiply_add(CTA_BODY),
    "no bare `a * b + c` in the CTA body: every multiply-add goes through "
    "pol.ma() so the mined mask decides the form",
)
check(
    "StaticForms{}" in CTA and "FLATXS_FORMS" not in CTA_BODY,
    "the kernel instantiates the SAME StaticForms policy over the SAME mined "
    "FLATXS_FORMS mask -- there is one definition of every form in the tree",
)


# ---------------------------------------------------------------------------
# 7. P3 -- THE STREAM LOOP CARRIES NO BARRIER, AND THE BARRIERS THAT EXIST
#    ARE THE ONES THAT PUBLISH CROSS-LANE STATE.
#
# A __syncthreads() inside the delta loop is not WRONG, it is slow -- one
# barrier per stream entry.  But its absence is only safe BECAUSE of P1, so the
# two rules are checked together: if someone breaks lane ownership, rule 4
# fires; if someone "fixes" it by adding a barrier per delta instead, this one
# does.
# ---------------------------------------------------------------------------
def rule_stream_loop_has_no_barrier(stream: str) -> bool:
    return "__syncthreads" not in stream


check(
    rule_stream_loop_has_no_barrier(CTA_STREAM),
    "the delta-stream loop contains NO __syncthreads(): by P1 there is no "
    "cross-lane dependency inside it, so its cost is independent of the stream "
    "length",
)
check(
    CTA_BODY.count("__syncthreads();") == 2,
    "exactly two barriers: one publishing the workspace + sh_iden, one "
    "publishing the macro stores the XSDF/XSRF pass re-reads from global",
)
check(
    "// Publish the workspace and sh_iden" in CTA_BODY,
    "the first barrier says what it publishes",
)
check(
    "if (tid == 0) {" in CTA_BODY and "iso > IO16" in CTA_BODY,
    "the light-isotope refresh splits rows 0..2 (lane 0) from rows 3.. (strided) "
    "so no lane reads a row another lane is about to overwrite",
)


# ---------------------------------------------------------------------------
# 8. THE TU IS BUILT --fmad=false, AND THE HEADER IS DEVICE-ONLY.
# ---------------------------------------------------------------------------
check(
    '"--fmad=false"' in CMAKE and "RASBERY_BITEXACT_CUDA_OPTS" in CMAKE,
    "CMakeLists still spells --fmad=false for the bit-exact TUs",
)
check(
    re.search(r"set_source_files_properties\(\"\$\{CMAKE_CURRENT_SOURCE_DIR\}/src/"
              r"CudaXsReconBackend\.cu\"\s*\n\s*PROPERTIES COMPILE_OPTIONS "
              r"\"\$\{RASBERY_BITEXACT_CUDA_OPTS\}\"\)", CMAKE) is not None,
    "the production TU that now contains the CTA kernel still carries the "
    "bit-exact option list",
)


def rule_ptxas_verbose_only_appends(cmake: str) -> bool:
    blk = body_of(cmake, 'set(RASBERY_BITEXACT_CUDA_OPTS "--fmad=false")', "endif ()")
    return "list(APPEND RASBERY_BITEXACT_CUDA_OPTS" in blk and "set(RASBER" \
        not in blk[blk.index("if (RASBERY_PTXAS_VERBOSE)"):]


check(
    rule_ptxas_verbose_only_appends(CMAKE),
    "RASBERY_PTXAS_VERBOSE only APPENDS to the bit-exact option list -- measuring "
    "must never be able to switch --fmad=false off",
)
check(
    '#error "FlatXsCtaKernel.cuh is device-only' in CTA,
    "the header refuses to be compiled by a host-only compiler rather than "
    "failing later on __shared__",
)


# ---------------------------------------------------------------------------
# 9. THE REPLAY GATE IS WIRED AND ITS PASS CONDITION INCLUDES CTA-vs-REFERENCE.
# ---------------------------------------------------------------------------
check(
    '#include "../src/FlatXsCtaKernel.cuh"' in REPLAY,
    "test/flatxs_device_replay.cu builds the CTA kernel",
)
check(
    'std::string(argv[2]) == "--cta"' in REPLAY,
    "the replay tool has a --cta mode",
)
check(
    "fxs::flatxsCtaLaunch(c, cta_threads, nullptr, false, cta_tile);" in REPLAY,
    "the replay tool launches through the SAME ladder helper the backend uses, "
    "so the gate cannot certify a block size production never runs",
)


def rule_replay_scores_cta_against_reference(replay: str) -> bool:
    return ('score(hc, h, tag_, ab_bad, sink, "cta-vs-ref");' in replay
            and "ab_bad == 0" in replay)


check(
    rule_replay_scores_cta_against_reference(REPLAY),
    "the replay compares the CTA arm against the REFERENCE KERNEL's own output "
    "(not only against the capture) and requires 0 mismatches to exit 0",
)
check(
    "c.iden   = up(iden);" in REPLAY and "c.xs_ssm = up(xs_ssm);" in REPLAY,
    "the CTA arm replays on its OWN copies of every mutable array, so the two "
    "arms cannot read each other's results",
)


# ---------------------------------------------------------------------------
# 10. THE DOC CARRIES THE 238 RUNBOOK THIS FILE DEFERS TO.
# ---------------------------------------------------------------------------
for token in ("RASBERY_GPU_FLATXS_CTA", "0d15abf29d222a02", "4382", "878",
              "--cta", "flatxs_resource_report.py", "8×M8", "h5diff"):
    check(token in DOC,
          "docs/WP5_FLATXS_CTA_20260831_KO.md names %s" % token)
check(
    "R1" in DOC and "39.9" in DOC,
    "the doc records that the 39.9 % share is the tracker's R1 figure and must "
    "be re-measured before adoption",
)


# ---------------------------------------------------------------------------
# 11. WP21-B2 -- THE TILED ARM.
#
# The tiled body is a SECOND copy of the arithmetic, deliberately (the reason
# is written at the top of the WP21-B2 block in src/FlatXsCtaKernel.cuh: the
# untiled body has to stay the bisectable A/B reference, and two arms that are
# one text cannot be bisected).  Duplicated arithmetic is a drift risk and this
# section is the price paid for it: every structural rule sections 4-7 apply to
# `flatxsSolveNodeCta` is applied AGAIN to `flatxsSolveTileCta`, against the
# same reference `flatxsSolveNode`.  A site that changes in one body and not
# the other fails the census here.
#
# The one rule that is NOT the same is lane ownership, because permuting lane
# ownership is exactly what the tile does.  (P1) is preserved in the stronger
# form: every phase walks a lane-owned space, but there are now TWO spaces --
# the delta-stream phase walks ONE node's ordinals with `q = lane; q < Q_x;
# q += T` (which is what keeps the coefficient reads stride-1), and every other
# phase walks the (ordinal, node) product with `p = tid; p < Q_x * TILE;
# p += NT` and node INNERMOST (which is what makes the store contiguous).
# ---------------------------------------------------------------------------
TILE_BODY = body_of(CTA, "flatxsSolveTileCta(const FlatXsView& v, int i0,",
                    "/// The static-__shared__ ceiling")
TILE_STREAM = body_of(TILE_BODY, "for (int s = s0; s < s1; ++s) {",
                      "// --- 3. Densities")

check(
    rule_form_bits_match(REF_BODY, TILE_BODY),
    "every FormBit appears the same number of times in flatxsSolveTileCta as in "
    "flatxsSolveNode -- the tiled arm is the same sites in the same forms",
)
check(
    rule_form_bits_match(CTA_BODY, TILE_BODY),
    "the tiled body and the untiled body carry the SAME FormBit census -- the "
    "duplication cannot drift in one direction only",
)
check(
    rule_iso_fold_is_sequential(TILE_BODY),
    "both macro-XS folds of the tiled body run `for (int iso = 0; iso < NISO; "
    "++iso)` -- ascending, sequential, one lane per output chain (P2)",
)
check(
    rule_no_bare_multiply_add(TILE_BODY),
    "no bare `a * b + c` in the tiled body either",
)
check(
    rule_stream_loop_has_no_barrier(TILE_STREAM),
    "the tiled delta-stream loop contains NO __syncthreads(): the tile slots "
    "have different stream lengths, so a barrier here would make every slot "
    "wait for the longest one, once per delta (P3)",
)
check(
    re.search(r"for\s*\(int\s+ige\s*=\s*0;\s*ige\s*<\s*NG;\s*\+\+ige\)\s*\n?\s*"
              r"rf\s*\+=", TILE_BODY) is not None,
    "the tiled XSRF sum stays a per-ig sequential `rf +=` chain",
)

# (P1), tiled form.  The delta-stream phase is group-owned over ONE node.
TILE_LANE_LOOP = re.compile(
    r"for\s*\(int\s+(\w+)\s*=\s*lane;\s*\1\s*<\s*(Q_LMP|Q_LSM|Q_MIC|Q_MSM);"
    r"\s*\1\s*\+=\s*T\)")
# Every other phase walks (ordinal, node) with the NODE INNERMOST.
TILE_PROD_LOOP = re.compile(
    r"for\s*\(int\s+(\w+)\s*=\s*tid;\s*\1\s*<\s*(Q_LMP|Q_LSM|Q_MIC|Q_MSM|NISO|NG)"
    r"\s*\*\s*TILE;\s*\1\s*\+=\s*NT\)")
ANY_TILE_LOOP = re.compile(r"for\s*\(int\s+(\w+)\s*=\s*[^;]+;\s*\1\s*<\s*"
                           r"(Q_LMP|Q_LSM|Q_MIC|Q_MSM)")


def rule_tile_loops_are_lane_owned(body: str) -> bool:
    good = {m.start() for m in TILE_LANE_LOOP.finditer(body)}
    good |= {m.start() for m in TILE_PROD_LOOP.finditer(body)}
    every = {m.start() for m in ANY_TILE_LOOP.finditer(body)}
    return every <= good and len(good) > 0


check(
    rule_tile_loops_are_lane_owned(TILE_BODY),
    "every Q_* loop in flatxsSolveTileCta is lane-owned: `q = lane; q < Q_x; "
    "q += T` in the delta-stream phase, `p = tid; p < Q_x * TILE; p += NT` "
    "everywhere else (P1)",
)
check(
    len(TILE_LANE_LOOP.findall(TILE_BODY)) == 4,
    "the delta-stream phase keeps ALL FOUR of its ordinal walks group-owned -- "
    "moving any of them to the node-innermost mapping would give each lane a "
    "different `base` and scatter the stride-1 coefficient reads",
)
check(
    len(TILE_PROD_LOOP.findall(TILE_BODY)) == 12,
    "the twelve node-innermost walks are present (gather 4 + density 1 + "
    "scatter 4 + macro 2 + XSDF/XSRF 1)",
)


def rule_tile_decomposes_node_innermost(body: str) -> bool:
    """`q = p / TILE; j = p - q * TILE` and nothing else.

    The transpose IS this decomposition.  `q = p % TILE` with `j = p / TILE`
    compiles, runs, produces plausible cross sections -- and puts the ORDINAL
    innermost, which is the layout the kernel already had.  The whole WP would
    then be a barrier and a shared-memory copy for nothing, with no error
    anywhere.
    """
    dec = re.findall(r"const int\s+(\w+)\s*=\s*p\s*/\s*TILE;", body)
    off = re.findall(r"const int\s+(\w+)\s*=\s*p\s*-\s*\w+\s*\*\s*TILE;", body)
    if len(dec) != len(off) or not dec:
        return False
    return all(o == "j" for o in off)


check(
    rule_tile_decomposes_node_innermost(TILE_BODY),
    "every node-innermost phase decomposes `p` as (q = p / TILE, j = p % TILE) "
    "-- the NODE is the fast axis, which is the entire point of the transpose",
)
check(
    TILE_BODY.count("__syncthreads();") == 3,
    "exactly three barriers in the tiled body: one publishing the transposed "
    "gather, one publishing the workspace + iden, one publishing the macro "
    "stores the XSDF/XSRF pass re-reads from global",
)
check(
    "sh_l[j]" in TILE_BODY and "v.nodes[" not in TILE_BODY,
    "the tile reads its node ids ONCE into __shared__ and indexes sh_l[] "
    "afterwards -- a per-element `v.nodes[i0+j]` would be a node-strided load "
    "reintroduced inside the phase that exists to remove one",
)

# --- the shared-memory budget ---------------------------------------------
#
# Recomputed from the SAME constants the workspace is sized from, so a change
# to the isotope registry moves this budget instead of silently overflowing the
# 48 KiB static ceiling (which is a compile error, but one that would appear
# for the first time on the benchmark host).
NISO_N = int(re.search(r"constexpr int NISO\s*=\s*(\d+)", XSR).group(1))
NG_N = int(re.search(r"constexpr int NG\s*=\s*(\d+)", XSR).group(1))
N_ACTIVE_N = int(re.search(r"constexpr int N_ACTIVE\s*=\s*(\d+)", REF).group(1))
NMIC_N, NLSM_N, NMSM_N = NISO_N * NG_N, NG_N * NG_N, NISO_N * NG_N * NG_N
WS_ELEMS = (N_ACTIVE_N * NG_N + NLSM_N + N_ACTIVE_N * NMIC_N + NMSM_N + NISO_N)
SMEM_MAX = 48 * 1024
TILE_D = int(re.search(r"constexpr int CTA_TILE_DEFAULT\s*=\s*(\d+)", CTA).group(1))
TILE_F = int(re.search(r"constexpr int CTA_TILE_DEFAULT_F32\s*=\s*(\d+)",
                       CTA).group(1))

check(
    "constexpr int CTA_SMEM_STATIC_MAX = 48 * 1024;" in CTA,
    "the 48 KiB static-__shared__ ceiling is a named constant, not a literal at "
    "the dispatch sites",
)
check(
    TILE_D * WS_ELEMS * 8 <= SMEM_MAX and TILE_F * WS_ELEMS * 4 <= SMEM_MAX,
    "both default tiles fit the static ceiling (FP64 %d x %d B, FP32 %d x %d B, "
    "against %d)" % (TILE_D, WS_ELEMS * 8, TILE_F, WS_ELEMS * 4, SMEM_MAX),
)

# --- and the budget that actually picks the default ------------------------
#
# 238 block 40 (ddd0ccc): the FP32 arm of THIS kernel ran 280 -> 379 us with
# occupancy 62.4 -> 39.8 % on HALF the shared memory.  Under a shared-memory
# model that cannot happen; under a register model it is arithmetic.  So the
# tile ladder is scored against REGISTERS, and the default is the entry that
# leaves the resident thread count where the untiled arm left it -- because the
# one thing this campaign will not repeat is guessing this kernel's occupancy
# from the wrong resource.
REGS_PER_SM = 65536      # sm_80..sm_120
THREADS_PER_SM = 2048
REGS_FP64 = 48           # tools/flatxs_resource_report.py, and 62.4 % confirms it
REGS_FP32 = 80           # implied by the measured 39.8 %
BLOCK = 128              # CTA_THREADS_DEFAULT


def resident_threads(tile: int, regs: int, block: int = BLOCK) -> int:
    per_block = block * tile * regs
    blocks = REGS_PER_SM // per_block
    return min(blocks * block * tile, THREADS_PER_SM)


def rule_tile_is_occupancy_neutral(tile_d: int, tile_f: int) -> bool:
    return (resident_threads(tile_d, REGS_FP64) >= resident_threads(1, REGS_FP64)
            and resident_threads(tile_f, REGS_FP32) >= resident_threads(1, REGS_FP32))


check(
    rule_tile_is_occupancy_neutral(TILE_D, TILE_F),
    "the default tiles are REGISTER-occupancy neutral: FP64 %d -> %d resident "
    "threads/SM at %d regs, FP32 %d -> %d at %d.  A default that lowered either "
    "would be WP20's mistake made again with a different resource"
    % (resident_threads(1, REGS_FP64), resident_threads(TILE_D, REGS_FP64),
       REGS_FP64, resident_threads(1, REGS_FP32),
       resident_threads(TILE_F, REGS_FP32), REGS_FP32),
)
check(
    not rule_tile_is_occupancy_neutral(2 * TILE_D, 2 * TILE_F),
    "and the NEXT rung is not neutral -- if it were, the default would be "
    "leaving store coalescing on the table for nothing",
)
check(
    TILE_D * BLOCK <= 1024 and TILE_F * BLOCK <= 1024,
    "both defaults keep blockDim.x = TILE * 128 inside the 1,024-thread block "
    "cap at the default block size",
)
# The register finding is a claim about __launch_bounds__ as much as about the
# tile, and the header has to carry it or the next reader repeats the analysis.
check(
    "minBlocksPerMultiprocessor" in CTA
    and "maxThreadsPerBlock and NOTHING ELSE" in CTA,
    "the header records WHY __launch_bounds__(T) alone cannot deliver the block "
    "count a shared-memory argument predicts",
)


def rule_tile_dispatch_is_constexpr_guarded(cta: str) -> bool:
    """Every tile instantiation is behind `if constexpr (ctaTileFits<...>)`.

    `kernelFlatXsCtaTile<256, 8, CtaWorkspace>` is not a slow kernel, it is a
    COMPILE ERROR (2,048 threads and 58 KiB of shared memory).  A runtime clamp
    cannot prevent an instantiation; only `if constexpr` can.
    """
    disp = body_of(cta, "inline void flatxsCtaDispatchTile(",
                   "/// The largest tile the ladder will actually run")
    calls = re.findall(r"flatxsCtaTiledLaunch<T,\s*(\d+),\s*WS>", disp)
    guards = re.findall(r"if constexpr \(ctaTileFits<T,\s*(\d+),\s*WS>\(\)\)", disp)
    asserts = re.findall(r"static_assert\(ctaTileFits<T,\s*(\d+),\s*WS>\(\)", disp)
    return bool(calls) and sorted(calls) == sorted(guards + asserts)


check(
    rule_tile_dispatch_is_constexpr_guarded(CTA),
    "every tile on the ladder is instantiated only inside `if constexpr "
    "(ctaTileFits<T, TILE, WS>())`",
)
check(
    "__launch_bounds__(T * TILE) kernelFlatXsCtaTile" in CTA,
    "the tiled kernel carries __launch_bounds__(T * TILE): the block scales with "
    "the tile, which is what keeps shared bytes PER THREAD unchanged",
)
check(
    "kernelFlatXsCtaAt<T, WS><<<tail, T, 0, stream>>>(v, n_tiles * TILE);" in CTA,
    "the tail nodes go to the untiled body through kernelFlatXsCtaAt -- the "
    "certified single-node path, offset, not a partially-masked tile",
)
TAIL = body_of(CTA, "kernelFlatXsCtaAt(FlatXsView v, int node_base)",
               "/// WP21-B2 tile ladder.")
check(
    "flatxsSolveNodeCta<T>(v, i, StaticForms{}, w);" in TAIL,
    "kernelFlatXsCtaAt calls the SAME untiled body -- the tail is not a third "
    "copy of the arithmetic",
)
check(
    "__shared__ WS w;" in TAIL
    and "template <int T, class WS>" in CTA
    and "kernelFlatXsCtaAt<T, WS>" in CTA,
    "the tail carries the TILE'S workspace type: on the narrow arm it keeps the "
    "float workspace and the float block accessors, so it cannot reintroduce "
    "the per-element widening 238 block 40 is measuring",
)

# --- the knob, the receipt and the gate ------------------------------------
check(
    'std::getenv("RASBERY_GPU_FLATXS_CTA_TILE")' in BACKEND,
    "RASBERY_GPU_FLATXS_CTA_TILE is read in the backend, once, into a cached int",
)
check(
    "int rasberyGpuFlatXsCtaTile();" in BACKEND_H
    and re.search(r"int\s+rasberyGpuFlatXsCtaTile\(\) \{ return 0; \}", STUB)
    is not None,
    "the tile accessor is declared beside the other arm flags and defined by the "
    "no-CUDA stub (a call site must never need an #ifdef)",
)
check(
    '"RASBERY_GPU_FLATXS_CTA_TILE"' not in DRIVER,
    "RASBERY_GPU_FLATXS_CTA_TILE is deliberately NOT in trajectory::kArmEnv: it "
    "changes WHICH LANE stores a byte, never which byte, so folding it into the "
    "case key would split one arm's cache into a family (the "
    "RASBERY_GPU_MICX_RESIDENT precedent).  If the 238 gate ever shows the tile "
    "moving a trajectory, THIS is the line that has to change first",
)
check(
    SOLVE.count("rasberyGpuFlatXsCtaTile()") == 1
    and "fxs::flatxsCtaTileResolved(" in SOLVE,
    "the tile is resolved ONCE at the same dispatch point the arm is, through "
    "the launcher's own ladder helper, so the receipt states the tile that RAN",
)
MAIN = read("src", "main.cpp")
for field in ('\\"tile\\":', '\\"tiles_launched\\":', '\\"tail_nodes\\":'):
    check(
        MAIN.count(field) == 3,
        "the [RASBERY][FLATXS][GPU] receipt carries %s on all three arms "
        "(batch, single, drive)" % field,
    )
check(
    "g_flatxs_cta_tiles.fetch_add" in BACKEND
    and "g_flatxs_cta_tail_nodes.fetch_add" in BACKEND,
    "tiles_launched and tail_nodes are counted at the launch site, so "
    "tiles*tile + tail reconstructs nodes_solved",
)
check(
    'std::getenv("RASBERY_GPU_FLATXS_CTA_TILE")' in REPLAY
    and "fxs::flatxsCtaTileResolved(" in REPLAY
    and "fxs::flatxsCtaLaunch(c, cta_threads, nullptr, false, cta_tile);" in REPLAY,
    "the replay gate selects the tile through the SAME resolver and the SAME "
    "launcher the backend uses -- it cannot certify a tile production never runs",
)
DOC_B2 = read("docs", "WP21_B2C2_COALESCING_20260831_KO.md")
for token in ("RASBERY_GPU_FLATXS_CTA_TILE", "1f36e75dc00ed2b4", "4377", "25.2",
              "h5diff", "1,321", "16.7",
              # 238 block 40: the FP32 regression and its register diagnosis.
              "379", "39.8", "65,536", "__launch_bounds__"):
    check(token in DOC_B2,
          "docs/WP21_B2C2_COALESCING_20260831_KO.md names %s" % token)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.
# ---------------------------------------------------------------------------
NEGATIVES: list[tuple[str, object]] = [
    ("flag back to default-off",
     lambda: rule_flag_default_on(
         'static const bool on = envFlagEnabled("RASBERY_GPU_FLATXS_CTA");')),
    ("off switch removed -- the reference arm becomes unreachable",
     lambda: rule_flag_default_on(
         "static const bool on = true;")),
    ("dispatch not guarded / reference arm gone",
     lambda: rule_dispatch_is_guarded(
         "    fxs::flatxsCtaLaunch(v, 128, d.stream);\n")),
    ("literal array bound in the shared workspace",
     lambda: rule_no_literal_dims("    double bm[702];\n")),
    ("constant respelled on the device",
     lambda: rule_no_respelled_constants(
         "constexpr double WATER_NUMBER_DENSITY = 0.033427699;")),
    ("an ordinal loop that is not lane-owned",
     lambda: rule_all_ordinal_loops_are_lane_owned(
         "for (int q = 0; q < Q_MIC; ++q) { w.bm[q] = 0.0; }")),
    ("an ordinal loop strided by something other than the block size",
     lambda: rule_all_ordinal_loops_are_lane_owned(
         "for (int q = tid; q < Q_MIC; q += 32) { w.bm[q] = 0.0; }")),
    ("isotope fold no longer sequential/ascending",
     lambda: rule_iso_fold_is_sequential(
         "for (int iso = NISO - 1; iso >= 0; --iso) val = f(val);")),
    ("a parallel tree reduction over iso",
     lambda: rule_no_parallel_reduction(
         "val += __shfl_down_sync(0xffffffff, val, 16);")),
    ("a contraction site dropped from the CTA body",
     lambda: rule_form_bits_match(REF_BODY,
                                  CTA_BODY.replace("pol.ma(F_MACRO_SSM,", "fma("))),
    ("a bare multiply-add",
     lambda: rule_no_bare_multiply_add("val = a * b + c;")),
    ("a barrier per delta in the stream loop",
     lambda: rule_stream_loop_has_no_barrier(
         "for (int s = s0; s < s1; ++s) { apply(s); __syncthreads(); }")),
    ("ptxas verbose replacing the bit-exact options",
     lambda: rule_ptxas_verbose_only_appends(
         'set(RASBERY_BITEXACT_CUDA_OPTS "--fmad=false")\n'
         "if (RASBERY_PTXAS_VERBOSE)\n"
         '    set(RASBERY_BITEXACT_CUDA_OPTS "--ptxas-options=-v")\n'
         "endif ()\n")),
    ("a tile default that costs resident threads (WP20 mistake, new resource)",
     lambda: rule_tile_is_occupancy_neutral(8, 8)),
    ("a tile phase that puts the ORDINAL innermost -- the transpose undone",
     lambda: rule_tile_decomposes_node_innermost(
         "const int j = p / TILE;\n" "const int q = p - j * TILE;\n")),
    ("a tiled ordinal loop that is not lane-owned",
     lambda: rule_tile_loops_are_lane_owned(
         "for (int p = 0; p < Q_MIC * TILE; ++p) { w[0].bm[p] = 0.0; }")),
    ("the delta-stream phase moved to the node-innermost mapping",
     lambda: rule_tile_loops_are_lane_owned(
         "for (int q = tid; q < Q_MIC; q += NT) { w[0].bm[q] = 0.0; }")),
    ("a contraction site dropped from the TILED body",
     lambda: rule_form_bits_match(
         REF_BODY, TILE_BODY.replace("pol.ma(F_ACC_MIC,", "fma("))),
    ("the tiled isotope fold no longer sequential/ascending",
     lambda: rule_iso_fold_is_sequential(
         "for (int iso = NISO - 1; iso >= 0; --iso) val = f(val);")),
    ("a barrier per delta in the tiled stream loop",
     lambda: rule_stream_loop_has_no_barrier(
         "for (int s = s0; s < s1; ++s) { apply(s); __syncthreads(); }")),
    ("a tile instantiated without its if-constexpr fit guard",
     lambda: rule_tile_dispatch_is_constexpr_guarded(
         "inline void flatxsCtaDispatchTile(\n"
         "    flatxsCtaTiledLaunch<T, 8, WS>(v, stream);\n"
         "/// The largest tile the ladder will actually run")),
    ("replay that only scores against the capture",
     lambda: rule_replay_scores_cta_against_reference(
         'score(hc, want, tag_, cta_bad, cta_worst, "cta-vs-capture");')),
]
for label, probe in NEGATIVES:
    checks += 1
    try:
        fired = not probe()  # type: ignore[operator]
    except Exception:
        fired = True
    if not fired:
        failures.append("negative control did not fire: " + label)


# ---------------------------------------------------------------------------
if failures:
    print("FAIL (%d of %d checks)" % (len(failures), checks))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("PASS  tools/test_flatxs_cta_contract.py  (%d checks, %d negative controls)"
      % (checks, len(NEGATIVES)))
print("  the bit-identity claim (CTA=0 vs CTA=1) is a 238 gate, not a source")
print("  property: test/flatxs_device_replay.cu --cta, and section 6 of")
print("  docs/WP5_FLATXS_CTA_20260831_KO.md.")
