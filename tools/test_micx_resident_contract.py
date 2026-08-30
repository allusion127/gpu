#!/usr/bin/env python3
"""Contract gate for WP15's RASBERY_GPU_MICX_RESIDENT arm and the [RASBERY][MICX] receipt.

WHAT THIS PROTECTS.  With the arm on, solveFlatXs computes the 59.5 MB micx/lmpx
block and does NOT download it -- 85 % of the run's D2H side (docs/WP13 §4.4)
stops crossing PCIe.  The download is not skipped, it is OWED, and the whole B0
claim is one sentence:

    WHILE THE DEBT IS OUTSTANDING NO HOST CODE READS OR WRITES _micx / _lmpx.

That sentence is not visible in a diff and it is one careless edit away from
false.  A new host reader added six months from now -- a diagnostic, a restart
path, an export -- would silently read the PREVIOUS flat-XS epoch's cross
sections and produce a plausible, wrong answer.  So:

M1  THE FLAG IS NOT A TRAJECTORY KNOB.  It changes WHEN a copy happens, never
    WHICH bytes it moves: the lazy download is the same copy list off the same
    device offsets, taken before the read instead of after the solve.  So
    RASBERY_GPU_MICX_RESIDENT must NOT be in trajectory::kArmEnv -- putting it
    there forks the evaluator's case key for two runs that are one run.

M2  EVERY HOST TOUCH OF THE LIVE BLOCK GOES THROUGH THE ACCESSOR.  Every
    function in XSSet.cpp whose body names `_micx`/`_lmpx` must either call
    XSSet::EnsureMicxHost or be on the short, justified allow-list of sites that
    take an ADDRESS and never a value.  This is the check that catches the
    reader nobody thought of, and it is the reason it is written as an
    enumeration over the file rather than a search for known names.

M3  THE DEBT IS RECORDED AT EVERY DEVICE WRITE, AND READ BACK FROM THE BACKEND.
    solveFlatXs must set BOTH pending flags on the deferral path, must refuse to
    defer when `mark_micx_resident` is false (that caller is about to have the
    block re-uploaded FROM the host, which would pay the debt with the bytes the
    upload overwrote), and XSSet must take the flags from the backend rather
    than assume them from its own reading of the env.

M4  THE OFF ARM IS THE CODE THAT SHIPPED.  The deferral must sit behind
    rasberyGpuMicxResidentEnabled(), and EnsureMicxHost's fast path must return
    before it touches anything, so an A/B measures the deferral and not the
    bookkeeping.

M5  THE TWO UPLOAD SITES CARRY THE ALARM.  solveFlatXs and stage() are the only
    places that can overwrite an undownloaded device block; both must say so
    loudly rather than let a silent wrong answer out.

M6  THE RECEIPT EXISTS AND CARRIES THE FOUR COUNTERS.  A deferral with no
    measurement is a claim with no evidence -- and `resident_hits == 0` with the
    arm on is the G0 check that says the flag never reached a solve.

M7  THE LAZY DOWNLOAD USES THE LEDGER WRAPPERS.  A raw cudaMemcpy* at the new
    site would leave 19 GB of moved-or-not-moved bytes outside
    [RASBERY][XFER][LEDGER], which is the instrument this work is measured with.

WP15.1 added three more, all of them about handing one backend's device memory
to another instead of routing it through the host:

M8  THE CRAM D2D IS ORDERED AND ALL-OR-NOTHING.  The depletion backend reads the
    four condensation slots out of the flat-XS backend's resident block; the two
    live on different streams, so without the event it can read a block the
    flat-XS kernel is still writing.  It must also take all four from the same
    place -- a mixed fill puts two flat-XS epochs into one condensation, and no
    generation check downstream can see that.  The offer side must check that
    the resident block is THIS epoch, because micxDeviceSlot hands out an
    address, not a generation.

M9  THE BATCH ARENA'S jnet SHADOW IS COMMITTED AT THE DOWNLOAD, NOT THE UPLOAD,
    and consulted only for LEGACY slots.  jnet has a device writer (kNodalJnet),
    so the usual "what I last sent" shadow would be a false statement about
    device content; what makes this one true is that the drive ends by
    downloading that same buffer.  An adopted slot's jnet is the CMFD backend's
    canonical buffer, which a device sweep can write between two nodal drives --
    there the canonicalElides* predicate owns the decision.

M10 THE BATCH ARM REACHES THE DEFERRAL THROUGH THE SAME solveFlatXs.  WP15's
    batch claim rests on there being exactly ONE flat-XS solve in the tree: the
    nodal arena is nodal-only, so the per-instance deferral covers 8x M8 for
    free.  A micx path growing inside the arena would silently need its own.

NEGATIVE CONTROLS.  Each check is run a second time against a MUTATED copy of
the source that breaks exactly the property it tests, and the check must fail.
A gate that cannot fail is not a gate.

Run:  python tools/test_micx_resident_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

XSSET = "src/XSSet.cpp"
XSSET_H = "src/XSSet.h"
XSR = "src/CudaXsReconBackend.cu"
XSR_H = "src/CudaXsReconBackend.h"
CRAM = "src/CudaCramBackend.cu"
CRAM_H = "src/CudaCramBackend.h"
STUB = "src/CudaXsReconBackendStub.cpp"
DRIVER = "src/Driver.h"

FLAG = "RASBERY_GPU_MICX_RESIDENT"

RECEIPT_FIELDS = (
    "resident_hits",
    "lazy_downloads",
    "slice_downloads",
    "bytes_saved",
    # WP15.1
    "cram_micx_h2d_mb",
    "cram_micx_d2d_mb",
    "nodal_jnet_elided_mb",
    "nodal_jnet_elision_tests",
)

# Functions in XSSet.cpp that name the live block and legitimately do NOT
# materialise it.  Every entry needs a reason, and the reason is the point: an
# allow-list without one is a place to hide a bug.
ADDRESS_ONLY = {
    # Allocates the arrays.  There is nothing on the device yet.
    "Initialize": "allocates the storage; no device block exists",
    # Releases the page-locks at teardown.  Takes base addresses.
    "~XSSet": "unpins base addresses; reads no element",
    # Builds the view EnsureMicxHost itself uses to name the destinations.  A
    # call here would be an infinite regress.
    "MakeFlatXsHostView": "builds the payer's own destination view",
    # Page-locks the same base addresses and then hands the view to the device.
    # The debt is RECORDED here (M3 checks that separately).
    "TryUpdateFlatXSGpu": "pins base addresses and records the debt",
}


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    """Comments explain the contract; they must never SATISFY it."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def strip_strings(text: str) -> str:
    """A diagnostic that MENTIONS _micx is not a read of _micx."""
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)


# `_micx` but not `_micx_bos`, `_micx_generation`, `_micx_device_*`, `_ref_micx`.
LIVE_TOKEN = re.compile(r"(?<![A-Za-z0-9_])_(?:micx|lmpx)(?![A-Za-z0-9_])")

FUNC_HEAD = re.compile(
    r"(?m)^[A-Za-z_][A-Za-z0-9_:<>,&* \t]*?\bXSSet::(~?[A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def xsset_functions(code: str) -> list[tuple[str, str]]:
    """(name, body) for every out-of-line XSSet member in the file, in order."""
    clean = strip_strings(strip_comments(code))
    heads = [(m.start(), m.group(1)) for m in FUNC_HEAD.finditer(clean)]
    out: list[tuple[str, str]] = []
    for i, (start, name) in enumerate(heads):
        stop = heads[i + 1][0] if i + 1 < len(heads) else len(clean)
        out.append((name, clean[start:stop]))
    return out


def body_of(code: str, signature: str, end: str) -> str:
    start = code.find(signature)
    if start < 0:
        return ""
    stop = code.find(end, start)
    return code[start:stop] if stop > start else code[start:]


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def check_m1_not_an_arm_knob(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    driver = src[DRIVER]
    anchor = "inline constexpr const char* kArmEnv[] = {"
    at = driver.find(anchor)
    if at < 0:
        return [f"M1: cannot find kArmEnv in {DRIVER}; the absence test is void"]
    block = driver[at + len(anchor) : driver.find("};", at)]
    names = re.findall(r'"([A-Za-z0-9_]+)"', block)
    if "RASBERY_GPU_CRAM" not in names:
        problems.append(
            "M1: kArmEnv parse looks wrong -- RASBERY_GPU_CRAM missing from "
            f"{len(names)} parsed names; the absence test cannot be trusted"
        )
    if FLAG in names:
        problems.append(
            f"M1: {FLAG} is in trajectory::kArmEnv.  The lazy download is the "
            "same copy list off the same offsets as the eager one, so every "
            "host-visible value is identical and the knob cannot move a "
            "trajectory.  If it CAN now, say so in the doc and delete this check."
        )
    return problems


def check_m2_every_reader_materialises(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    funcs = xsset_functions(src[XSSET])
    if not funcs:
        return ["M2: parsed no XSSet member functions; the enumeration is void"]
    # Trust the parse only if it found members we KNOW touch the block.
    seen = {name for name, _ in funcs}
    for anchor in ("ReconstructNode", "UpdateUnroddedNodeXS", "DepleteNode"):
        if anchor not in seen:
            return [
                f"M2: the function parse missed XSSet::{anchor}; the "
                "enumeration cannot be trusted"
            ]
    touched = 0
    for name, body in funcs:
        if not LIVE_TOKEN.search(body):
            continue
        touched += 1
        if "EnsureMicxHost" in body:
            continue
        if name in ADDRESS_ONLY:
            continue
        problems.append(
            f"M2: XSSet::{name} names the live _micx/_lmpx block and never calls "
            "EnsureMicxHost.  With RASBERY_GPU_MICX_RESIDENT on those arrays can "
            "hold the PREVIOUS flat-XS epoch, so this reads (or overwrites) cross "
            "sections that are one solve stale.  Add the call, or -- if it only "
            "takes addresses -- add it to ADDRESS_ONLY here with the reason."
        )
    if touched < 10:
        problems.append(
            f"M2: only {touched} functions were found to touch the block; the "
            "census had 17.  The token filter has stopped matching and this "
            "check is measuring nothing."
        )
    return problems


def check_m2b_value_accessors(src: dict[str, str]) -> list[str]:
    """The two VALUE accessors are how IO and NodeSpectralIndex reach the block."""
    problems: list[str] = []
    header = strip_comments(src[XSSET_H])
    for accessor, need in (("double micx(", "Scalars"), ("double micxssm(", "All")):
        body = body_of(header, accessor, "\n    }")
        if not body:
            problems.append(f"M2b: XSSet::{accessor.split()[1]} is gone")
            continue
        if "EnsureMicxHost" not in body:
            problems.append(
                f"M2b: {accessor.split()[1]} returns an element of the live block "
                "without paying the deferred download.  IO's node-monitor dump "
                "and NodeSpectralIndex reach _micx through here and nowhere else."
            )
        elif f"MicxNeed::{need}" not in body:
            problems.append(
                f"M2b: {accessor.split()[1]} materialises the wrong half "
                f"(expected MicxNeed::{need})"
            )
    return problems


def check_m3_debt_recorded(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    cu = strip_comments(src[XSR])

    # The deferral branch of solveFlatXs must raise BOTH flags.
    if "d.micx_scalars_pending = true;" not in cu:
        problems.append("M3: solveFlatXs never raises micx_scalars_pending")
    if "d.micx_scatter_pending = true;" not in cu:
        problems.append("M3: solveFlatXs never raises micx_scatter_pending")

    # ... and must refuse to defer when the caller is about to re-upload.
    if "mark_micx_resident" not in cu:
        problems.append("M3: solveFlatXs lost its mark_micx_resident parameter")
    elif not re.search(
        r"micx_resident\s*=\s*micx_resident_arm\s*&&\s*mark_micx_resident", cu
    ):
        problems.append(
            "M3: the deferral no longer requires mark_micx_resident.  With it "
            "false the next solve re-uploads the block FROM THE HOST, and a debt "
            "carried across that upload is paid with the bytes it overwrote."
        )

    # XSSet must take the flags from the backend, not from its own env reading.
    xs = strip_comments(src[XSSET])
    if "micxScalarsPending()" not in xs or "micxScatterPending()" not in xs:
        problems.append(
            "M3: XSSet no longer reads the pending flags back from the backend.  "
            "The backend declines to defer in cases this side does not enumerate "
            "(the SKIP_MICX_DL experiment, any_rodded); assuming is how the two "
            "sides drift."
        )
    return problems


def check_m4_off_arm_untouched(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    cu = strip_comments(src[XSR])
    if "rasberyGpuMicxResidentEnabled()" not in cu:
        return ["M4: the flag reader is not called anywhere in the backend"]
    if not re.search(
        r"micx_resident_arm\s*=\s*rasberyGpuMicxResidentEnabled\(\)", cu
    ):
        problems.append(
            "M4: the deferral is no longer gated on rasberyGpuMicxResidentEnabled()"
        )
    if f'envFlagEnabled("{FLAG}")' not in cu:
        problems.append(
            f"M4: {FLAG} is no longer ABSENT-MEANS-OFF.  It becomes default-on "
            "only after the 238 runbook in docs/WP15_MICX_RESIDENCY_20260830_KO.md "
            "has been run."
        )
    # EnsureMicxHost's fast path must return before doing anything at all.
    xs = strip_comments(src[XSSET])
    body = body_of(xs, "void XSSet::EnsureMicxHost(", "\nbool XSSet::")
    if not body:
        problems.append("M4: XSSet::EnsureMicxHost is gone")
    else:
        head = body[: body.find("XSSet& self")] if "XSSet& self" in body else body
        if "return;" not in head:
            problems.append(
                "M4: EnsureMicxHost no longer returns early when nothing is "
                "owed.  It is called per node from inside OpenMP loops; the "
                "nothing-owed path has to be two atomic loads and a return."
            )
    return problems


def check_m5_upload_alarms(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    cu = strip_comments(src[XSR])
    alarms = cu.count("[RASBERY][ERROR][micx]")
    if alarms < 2:
        problems.append(
            f"M5: {alarms} of the 2 micx upload alarms remain.  solveFlatXs and "
            "stage() are the only places that can overwrite a device block whose "
            "results no host reader has collected; both must say so."
        )
    for site, marker in (
        ("solveFlatXs", "d.micx_scalars_pending || d.micx_scatter_pending"),
        ("stage", "micx_scalars_pending || micx_scatter_pending"),
    ):
        if marker not in cu:
            problems.append(f"M5: the {site} upload no longer tests the debt")
    return problems


def check_m6_receipt(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    driver = src[DRIVER]
    if "[RASBERY][MICX]" not in driver:
        return ["M6: Driver.h emits no [RASBERY][MICX] receipt"]
    at = driver.find("[RASBERY][MICX]")
    block = driver[at : at + 1600]
    for field in RECEIPT_FIELDS:
        if f'\\"{field}\\"' not in block and f'"{field}"' not in block:
            problems.append(f"M6: the receipt no longer carries {field}")
    header = src[XSR_H]
    for fn in (
        "micxResidentHits",
        "micxLazyDownloads",
        "micxSliceDownloads",
        "micxBytesSaved",
    ):
        if fn not in header:
            problems.append(f"M6: XsReconBackend::{fn} is gone from the header")
        if fn not in src[STUB]:
            problems.append(
                f"M6: {fn} has no stub definition -- the CUDA-less build breaks"
            )
    return problems


def check_m7_ledger_wrappers(src: dict[str, str]) -> list[str]:
    problems: list[str] = []
    cu = strip_comments(src[XSR])
    body = body_of(cu, "bool XsReconBackend::downloadFlatXsMicx(", "\nunsigned long long")
    if not body:
        return ["M7: XsReconBackend::downloadFlatXsMicx is gone"]
    if "d.download(" not in body:
        problems.append(
            "M7: the lazy download no longer goes through Impl::download, so its "
            "bytes are outside [RASBERY][XFER][LEDGER]"
        )
    if re.search(r"(?<!xfer::)cudaMemcpy", body):
        problems.append(
            "M7: a raw cudaMemcpy* at the lazy-download site.  Every copy in "
            "this tree goes through the rasbery::xfer wrappers, which is what "
            "makes the ledger a census rather than a sample."
        )
    if "streamSync" not in body:
        problems.append(
            "M7: the lazy download does not drain.  Its caller is a host reader "
            "that is about to dereference these arrays."
        )
    return problems


def check_m8_cram_d2d(src: dict[str, str]) -> list[str]:
    """WP15.1: CRAM reads the resident block instead of re-uploading a copy of it."""
    problems: list[str] = []
    cu = strip_comments(src[CRAM])
    if "bool fillMic(" not in cu:
        return ["M8: CudaCramBackend::fillMic is gone; the D2D path does not exist"]
    if cu.count("fillMic(v.mic, v.mic_device, v.mic_device_ready") != 2:
        problems.append(
            "M8: the predictor and the corrector no longer BOTH go through "
            "fillMic -- one of them is back to an unconditional H2D"
        )
    body = body_of(cu, "bool fillMic(", "\n    /// Fill the DevCtx")
    if "cudaStreamWaitEvent" not in body:
        problems.append(
            "M8: fillMic does the D2D without waiting on the producer's event.  "
            "The two backends are on different streams, so this reads a block "
            "the flat-XS kernel may still be writing -- finite, plausible, wrong."
        )
    if "ready != nullptr" not in body:
        problems.append(
            "M8: fillMic no longer requires an ordering event before the D2D.  "
            "No event, no D2D: the host copy is always the safe answer."
        )
    if "kSlot[0]" not in body:
        problems.append(
            "M8: the all-or-nothing decision is gone.  A mixed fill puts two "
            "flat-XS epochs into one condensation and no generation check can "
            "see it."
        )
    if "kSlot[k]" not in body:
        problems.append(
            "M8: fillMic no longer indexes the caller's eleven-slot table "
            "through kSlot; the four blocks it uploads are no longer provably "
            "{XSAF, XSFF, XS2N, XS3N}"
        )
    xs = strip_comments(src[XSSET])
    offer = body_of(xs, "bool XSSet::FillCramMicDevice(", "\nbool XSSet::TryUpdateFlatXSGpu")
    if not offer:
        return problems + ["M8: XSSet::FillCramMicDevice is gone"]
    if "micxResidentGeneration() != _micx_generation" not in offer:
        problems.append(
            "M8: the offer no longer checks that the resident block is THIS "
            "epoch.  micxDeviceSlot hands out an address, not a generation; "
            "without the check CRAM condenses one flat-XS epoch behind."
        )
    if "micxReadyEvent()" not in offer:
        problems.append("M8: the offer no longer supplies the ordering event")
    return problems


def check_m9_jnet_shadow(src: dict[str, str]) -> list[str]:
    """WP15.1: the batch nodal arena's jnet upload shadow."""
    problems: list[str] = []
    cu = strip_comments(src[XSR])
    if "jnet_mirror" not in cu:
        return ["M9: the NodalArena jnet shadow is gone"]
    # Committed at the DOWNLOAD, which is the only event that makes it true.
    if "sl.jnet_mirror.commit(sl.h_jnet" not in cu:
        problems.append("M9: the jnet shadow is never committed")
    if "downloaded_jnet" not in cu:
        problems.append(
            "M9: the shadow no longer records whether the D2H actually ran.  "
            "An ELIDED download leaves h_jnet describing an older device state, "
            "and committing that licenses an upload elision the device needs."
        )
    at = cu.find("sl.jnet_mirror.commit")
    if at >= 0:
        # The commit must stand AFTER launchBatch's drain, not between the
        # queued D2H and it.  Anchor on the drain's own ledger site string.
        drain = cu.find('"CudaXsReconBackend.cu:NodalArena::launchBatch", "drain"')
        if drain < 0 or drain > at:
            problems.append(
                "M9: the shadow is committed before launchBatch's drain.  The "
                "D2H has only been ISSUED at that point, so h_jnet is not yet "
                "the device's jnet."
            )
    # Legacy slots only: an adopted jnet has a second device writer.
    if "_jnet_mirror && canon.jnet == nullptr &&" not in cu:
        problems.append(
            "M9: the shadow is consulted for ADOPTED slots too.  Their v.jnet "
            "is the CMFD backend's canonical buffer, which a device sweep can "
            "write between two nodal drives -- the shadow would be a statement "
            "about a buffer someone else has since changed."
        )
    # The per-slot reset audit has to cover it.
    at_def = cu.find("bool nodalSlotIsReset(const Slot&")
    reset = cu[at_def : at_def + 1800] if at_def >= 0 else ""
    if "jnet_mirror" not in reset:
        problems.append(
            "M9: nodalSlotIsReset does not check the jnet shadow.  A shadow "
            "inherited by the next tenant describes another deck's currents."
        )
    return problems


def check_m10_batch_path_covered(src: dict[str, str]) -> list[str]:
    """WP15.1: the batch arm reaches the deferral through the SAME solveFlatXs."""
    problems: list[str] = []
    cu = strip_comments(src[XSR])
    start = cu.find("class NodalArena {")
    if start < 0:
        return ["M10: NodalArena is gone; the batch-coverage claim is void"]
    stop = cu.find("\nbool nodalArenaWanted()", start)
    arena = cu[start:stop] if stop > start else cu[start:]
    if "micx" in arena or "flatxs" in arena:
        problems.append(
            "M10: the batch nodal arena has grown its own micx/flat-XS path.  "
            "WP15's whole batch claim is that there is only ONE flat-XS solve "
            "-- the per-instance solveFlatXs -- so the deferral covers the "
            "batch arm for free.  A second path needs its own deferral."
        )
    # The two copy lists, and only two: solveFlatXs (eager) and the payer.
    if cu.count('d.download("flatxs micx mic"') != 2:
        problems.append(
            "M10: expected exactly two call sites of the "
            "micx download list (solveFlatXs's eager one and "
            "downloadFlatXsMicx's lazy one).  A third is a third opinion about "
            "what the block is."
        )
    return problems


CHECKS = (
    ("M1", check_m1_not_an_arm_knob),
    ("M2", check_m2_every_reader_materialises),
    ("M2b", check_m2b_value_accessors),
    ("M3", check_m3_debt_recorded),
    ("M4", check_m4_off_arm_untouched),
    ("M5", check_m5_upload_alarms),
    ("M6", check_m6_receipt),
    ("M7", check_m7_ledger_wrappers),
    ("M8", check_m8_cram_d2d),
    ("M9", check_m9_jnet_shadow),
    ("M10", check_m10_batch_path_covered),
)

NEGATIVE_CONTROLS = (
    (
        "M1",
        "the flag added to kArmEnv",
        DRIVER,
        lambda t: t.replace(
            "inline constexpr const char* kArmEnv[] = {",
            'inline constexpr const char* kArmEnv[] = {\n    "' + FLAG + '",',
            1,
        ),
    ),
    (
        "M2",
        "ReconstructNode stops materialising",
        XSSET,
        lambda t: t.replace(
            "    EnsureMicxHost(MicxNeed::All);\n    const int    ng   = _g.ng();",
            "    const int    ng   = _g.ng();",
            1,
        ),
    ),
    (
        "M2",
        "a new host reader of _micx appears with no accessor call",
        XSSET,
        lambda t: t
        + "\ndouble XSSet::WP15NegativeControlReader(int l) const {\n"
        "    return _micx[Chiffon::XSAF][static_cast<size_t>(l)] +\n"
        "           _lmpx[Chiffon::XSAF][static_cast<size_t>(l)];\n}\n",
    ),
    (
        "M2b",
        "the micx() value accessor stops materialising",
        XSSET_H,
        lambda t: t.replace(
            "        EnsureMicxHost(MicxNeed::Scalars);\n        const size_t elem = (iso",
            "        const size_t elem = (iso",
            1,
        ),
    ),
    (
        "M3",
        "the deferral stops honouring mark_micx_resident",
        XSR,
        lambda t: t.replace(
            "const bool        micx_resident     = micx_resident_arm && mark_micx_resident;",
            "const bool        micx_resident     = micx_resident_arm;",
            1,
        ),
    ),
    (
        "M3",
        "XSSet assumes the debt instead of reading it back",
        XSSET,
        lambda t: t.replace(
            "    _micx_device_scalars.store(_xsrecon_backend->micxScalarsPending(),\n"
            "                               std::memory_order_release);\n"
            "    _micx_device_scatter.store(_xsrecon_backend->micxScatterPending(),\n"
            "                               std::memory_order_release);",
            "    _micx_device_scalars.store(true, std::memory_order_release);\n"
            "    _micx_device_scatter.store(true, std::memory_order_release);",
            1,
        ),
    ),
    (
        "M4",
        "the deferral escapes the flag",
        XSR,
        lambda t: t.replace(
            "static const bool micx_resident_arm = rasberyGpuMicxResidentEnabled();",
            "static const bool micx_resident_arm = true;",
            1,
        ),
    ),
    (
        "M4",
        "the flag becomes default-on before the runbook",
        XSR,
        lambda t: t.replace(
            'static const bool on = envFlagEnabled("RASBERY_GPU_MICX_RESIDENT");',
            'static const bool on = !envFlagDisabled("RASBERY_GPU_MICX_RESIDENT");',
            1,
        ),
    ),
    (
        "M5",
        "the stage() upload alarm removed",
        XSR,
        lambda t: t.replace(
            "            if (micx_scalars_pending || micx_scatter_pending) {",
            "            if (false) {",
            1,
        ),
    ),
    (
        "M6",
        "a receipt counter dropped",
        DRIVER,
        lambda t: t.replace('\\"slice_downloads\\":{}', '\\"slices\\":{}', 1),
    ),
    (
        "M8",
        "the CRAM D2D stops waiting on the producer's event",
        CRAM,
        lambda t: t.replace("cudaStreamWaitEvent(stream,", "cudaSuccess; (void)(", 1),
    ),
    (
        "M8",
        "the offer stops checking that the resident block is this epoch",
        XSSET,
        lambda t: t.replace(
            "    if (_xsrecon_backend->micxResidentGeneration() != _micx_generation) return false;",
            "    // generation check removed",
            1,
        ),
    ),
    (
        "M9",
        "the jnet shadow is consulted for adopted slots too",
        XSR,
        lambda t: t.replace(
            "} else if (_jnet_mirror && canon.jnet == nullptr &&",
            "} else if (_jnet_mirror &&",
            1,
        ),
    ),
    (
        "M9",
        "the slot reset audit stops covering the jnet shadow",
        XSR,
        lambda t: t.replace(
            "        if (sl.jnet_mirror.valid() || sl.downloaded_jnet) return false;\n",
            "",
            1,
        ),
    ),
    (
        "M10",
        "the batch nodal arena grows its own flat-XS micx path",
        XSR,
        lambda t: t.replace(
            "    int acquireSlot(const ndl::NodalView& p) {",
            "    void arena_flatxs_micx_download() {}\n"
            "    int acquireSlot(const ndl::NodalView& p) {",
            1,
        ),
    ),
    (
        "M7",
        "the lazy download bypasses the ledger wrappers",
        XSR,
        lambda t: t.replace(
            '            if (!d.download("flatxs micx mic", host.mic[t], d.off_mic[ACTIVE_XT9[t]],\n'
            "                            mic))\n"
            "                return false;\n"
            "        }\n"
            "        moved += static_cast<unsigned long long>(fxs::N_ACTIVE) * (lmp + mic) *",
            "            if (cudaMemcpyAsync(host.mic[t], d.dev_block + d.off_mic[ACTIVE_XT9[t]],\n"
            "                                mic * sizeof(double), cudaMemcpyDeviceToHost,\n"
            "                                d.stream) != cudaSuccess)\n"
            "                return false;\n"
            "        }\n"
            "        moved += static_cast<unsigned long long>(fxs::N_ACTIVE) * (lmp + mic) *",
            1,
        ),
    ),
)


def main() -> int:
    src = {p: read(p) for p in (XSSET, XSSET_H, XSR, XSR_H, STUB, DRIVER, CRAM, CRAM_H)}

    problems: list[str] = []
    for _, check in CHECKS:
        problems.extend(check(src))

    control_failures: list[str] = []
    for name, description, path, mutate in NEGATIVE_CONTROLS:
        mutated = dict(src)
        mutated[path] = mutate(src[path])
        if mutated[path] == src[path]:
            control_failures.append(
                f"negative control [{name}] {description}: the mutation did not "
                f"apply to {path} -- the anchor text has moved, so this control "
                "is measuring nothing.  Update the anchor."
            )
            continue
        fired = False
        for check_name, check in CHECKS:
            if check_name != name:
                continue
            if check(mutated):
                fired = True
        if not fired:
            control_failures.append(
                f"negative control [{name}] {description}: {name} PASSED on a "
                "source that breaks it -- the check does not test what it says"
            )

    for line in problems:
        print(f"FAIL {line}")
    for line in control_failures:
        print(f"FAIL {line}")

    if problems or control_failures:
        print(
            f"\ntest_micx_resident_contract: {len(problems)} contract violation(s), "
            f"{len(control_failures)} dead negative control(s)"
        )
        return 1

    print(
        f"test_micx_resident_contract: OK "
        f"({len(CHECKS)} checks, {len(NEGATIVE_CONTROLS)} negative controls)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
