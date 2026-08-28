#!/usr/bin/env python3
"""Contract gate for the CMFD sweep transfer plan (audit items C3/C4/C5).

Three claims are load-bearing here and none of them is visible in a diff:

C3  psi does not round-trip.  It is uploaded on the FIRST launch of a drive and
    downloaded only on the exceptional (state == 2) one.  Both halves rest on
    the same invariant: every caller of BICGCMFD::drive regenerates _psi with
    CMFD::updpsi(Phif()) first, so nothing ever reads the bytes the removed
    download used to write.  If a call site ever drives CMFD without that
    updpsi in front of it, the download has to come back -- so the invariant is
    asserted against Driver.h here, where it can be caught by reading rather
    than by a k_eff that moved in the sixth digit.

C4  chif / xsnf / vol are ALIASED, not staged.  The rebuild loop is gone
    because the accessors it read are already the layout the device wants.  A
    reintroduced copy would be silently correct and permanently wasteful, so
    the staging loop must stay absent.  (The generation-counter guard the audit
    proposed is explicitly NOT used: hoststateGeneration means "the device
    mirror of _xs is stale", not "the host bytes changed", and XSSet.cpp does
    not bump it when the GPU XS arm downloads a new _xs into host memory.)

C5  the flux mirror stays, and stays measurable.  The counters that decide it
    must exist and be emitted.

Run:  python tools/test_cmfd_sweep_transfer_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def main() -> int:
    problems: list[str] = []
    cu = read("src", "CudaBICGBackend.cu")
    cu_code = strip_comments(cu)
    hdr = read("src", "CudaBICGBackend.h")
    cmfd = read("src", "BICGCMFD.cpp")
    cmfd_code = strip_comments(cmfd)
    driver = strip_comments(read("src", "Driver.h"))
    solver = read("src", "BICGSolver.cpp")

    # ---- C3a: the download is gone from the per-launch path ---------------
    down = cu_code[cu_code.find("void issueSweepDownloads"):]
    down = down[: down.find("void issueFluxDownloads")]
    if "psi_dev" in down:
        problems.append(
            "issueSweepDownloads still copies psi back on every launch; it belongs on "
            "the state == 2 path only (issueExceptionalOperatorDownloads)")
    exc = cu_code[cu_code.find("void issueExceptionalOperatorDownloads"):]
    exc = exc[: exc.find("void drain(")]
    if "psi_dev" not in exc or "host_psi" not in exc:
        problems.append(
            "issueExceptionalOperatorDownloads does not pull psi; the degenerate-gamma "
            "hand-back in BICGCMFD reads psi(l) and would read stale host bytes")
    if "state != 2" not in exc:
        problems.append("the exceptional download no longer keys on state == 2")
    # The psi pull must not inherit the device_assembly filter: the host-assembly
    # arm needs the new fission source just as much.
    psi_at = exc.find("host_psi")
    asm_at = exc.find("if (!sl.device_assembly) continue;")
    if psi_at >= 0 and asm_at >= 0 and asm_at < psi_at:
        problems.append(
            "the psi download sits behind the device_assembly filter; a host-assembly "
            "slot would take the Rayleigh branch on stale psi")

    # ---- C3b: the upload is first-launch-only, and the flag is wired ------
    if "bool          psi_dirty = true;" not in hdr:
        problems.append("CmfdSweepIO has no psi_dirty flag")
    if "sl.push_psi   = io.psi_dirty;" not in cu_code:
        problems.append("stageSweeps does not carry psi_dirty into the slot")
    if "if (sl.push_psi) {" not in cu_code:
        problems.append("issueSweepUploads pushes psi unconditionally")
    if "bool psi_dirty = true;" not in cmfd_code:
        problems.append("driveDeviceSweeps does not arm psi_dirty for the first launch")
    if "psi_dirty          = false;" not in cmfd_code:
        problems.append(
            "driveDeviceSweeps never clears psi_dirty; every launch would re-upload the "
            "host copy over the psi the device just advanced")
    # It has to be cleared AFTER the launch, not before -- otherwise the very
    # first launch of a drive skips the one upload that matters.
    launch_at = cmfd_code.find("_ls->driveSweepsCuda(flux, io)")
    clear_at = cmfd_code.find("psi_dirty          = false;")
    if launch_at >= 0 and clear_at >= 0 and clear_at < launch_at:
        problems.append("psi_dirty is cleared before the launch that consumes it")

    # ---- C3c: the invariant the removal rests on --------------------------
    # Every cmfd_solver.drive(...) must be preceded by a REGENERATION of psi.
    #
    # On the host path that is cmfd_solver.updpsi() with no other cmfd_solver
    # call in between.  Rev.7.1 Task 9 link 2 adds a second producer: the device
    # outer segment runs updpsi as a KERNEL over the arena buffer the sweep
    # reads, and mirrors the result back into _psi before its sweep hook (see
    # CudaOuterGraph.cu, "mirror psi to the host").  The invariant this test
    # protects -- "the psi this drive reads was regenerated from the current
    # flux" -- holds on both, so the check admits the hook by name rather than
    # by relaxing the rule.
    #
    # NAMED, NOT PATTERN-MATCHED, deliberately.  A rule like "or any drive()
    # inside a function whose name contains Hook" would pass for a hook that
    # never regenerated anything.  There is exactly one such producer today; a
    # second one has to be added here, which is the point.
    DEVICE_PSI_PRODUCERS = ("outerSweepHook",)

    def enclosing_function(text, pos):
        head = text.rfind("static bool ", 0, pos)
        if head < 0:
            return ""
        open_paren = text.find("(", head)
        return text[head + len("static bool "):open_paren].strip() if open_paren > head else ""

    calls = [(m.start(), m.group(1))
             for m in re.finditer(r"cmfd_solver\.(updpsi|drive)\s*\(", driver)]
    for i, (pos, name) in enumerate(calls):
        if name != "drive":
            continue
        if enclosing_function(driver, pos) in DEVICE_PSI_PRODUCERS:
            continue
        if i == 0 or calls[i - 1][1] != "updpsi":
            problems.append(
                f"Driver.h offset {pos}: cmfd_solver.drive() is not immediately preceded "
                "by cmfd_solver.updpsi(), and is not one of the named device-psi "
                f"producers {DEVICE_PSI_PRODUCERS}. The psi download was removed BECAUSE "
                "psi is regenerated from the flux before every drive; without that this "
                "drive reads a host psi the device has moved past.")
    # The named producer has to actually exist, or the exemption is a hole.
    for producer in DEVICE_PSI_PRODUCERS:
        if f"static bool {producer}(" not in driver:
            problems.append(
                f"Driver.h: the C3c exemption names {producer}, which does not exist; an "
                "exemption for a function nobody can find exempts everything")

    # ---- C4: no staging copy, no generation guard -------------------------
    # Scoped to driveDeviceSweeps: updls and the offline dump capture read the
    # same accessors legitimately.
    drive_dev = cmfd_code[cmfd_code.find("bool BICGCMFD::driveDeviceSweeps"):]
    drive_dev = drive_dev[: drive_dev.find("void BICGCMFD::drive(")]
    if not drive_dev:
        problems.append("BICGCMFD.cpp: cannot locate driveDeviceSweeps")
    for banned, why in (
        (r"_sweep_xsnf\[", "xsnf is aliased from XSSet::xsnfData(); a copy is dead work"),
        (r"_sweep_vol\[", "vol is aliased from Geometry's _vol; a copy is dead work"),
        (r"_x\.xsnf\(ig", "the strided per-node gather is the loop that was removed"),
    ):
        if re.search(banned, drive_dev):
            problems.append(f"BICGCMFD::driveDeviceSweeps: `{banned}` is back -- {why}")
    if "_x.xsnfData()" not in cmfd_code or "&_g.vol(0)" not in cmfd_code:
        problems.append(
            "driveDeviceSweeps no longer aliases xsnf/vol from their owners")
    if "hoststateGeneration" in drive_dev or "refGeneration" in drive_dev:
        problems.append(
            "BICGCMFD.cpp guards a host-array read on an XSSet generation counter. "
            "hoststateGeneration tracks DEVICE-MIRROR staleness, not host content: "
            "XSSet.cpp does not bump it when the GPU XS arm downloads a fresh _xs "
            "into host memory, which is exactly the case such a guard must catch.")
    # The aliased ranges are owned by XSSet/Geometry, so BICGCMFD must not take a
    # lease it will release in its own destructor (the registry deduplicates a
    # repeat request for the same base instead of counting a second owner).
    for tag in ('"xs.xsnf@sweep"', '"geom.vol@sweep"'):
        if tag not in cmfd_code:
            problems.append(f"the aliased range is not page-locked under {tag}")
        line = next((ln for ln in cmfd.splitlines() if tag in ln), "")
        if "lease_vector" in line:
            problems.append(
                f"{tag} is taken with lease_vector; ~BICGCMFD would then release the "
                "only owner record for a base XSSet/Geometry still uses")

    # ---- C5: the mirror decision stays measurable -------------------------
    for field in ("cmfd_psi_h2d_elided_bytes", "cmfd_psi_d2h_elided_bytes",
                  "cmfd_phi_mirror_ns", "cmfd_phi_mirror_calls",
                  "cmfd_phi_mirror_bypassed", "cmfd_phi_h2d_elided_bytes"):
        if field not in hdr:
            problems.append(f"BackendCounters has no {field}")
        if field not in solver:
            problems.append(f"the BICGSolver telemetry line does not emit {field}")
        if field not in cu:
            problems.append(f"the arena telemetry line does not emit {field}")
    gate = cu_code[cu_code.find("bool phiMirrorEnabled"):][:400]
    if "RASBERY_GPU_PHI_MIRROR" not in gate:
        problems.append("phiMirrorEnabled does not read RASBERY_GPU_PHI_MIRROR")
    if 'std::string(v) != "0"' not in gate or "v == nullptr ||" not in gate:
        problems.append(
            "phiMirrorEnabled must default ON (measured: 3.69 us of mirror against a "
            "19.52 us H2D issue it elides on every continuation launch) and treat '0' "
            "as off, like every other RASBERY_* gate")
    if "sl.push_phi  = !phiMirrorMatches(" not in cu_code:
        problems.append("the flux mirror comparison does not go through the timed gate")

    if problems:
        for problem in problems:
            print(f"cmfd sweep transfer contract: FAIL {problem}", file=sys.stderr)
        return 1
    print("cmfd sweep transfer contract: PASS (psi round trip removed with its "
          "updpsi invariant checked, xs/vol aliased, mirror measurable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
