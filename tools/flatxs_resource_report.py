#!/usr/bin/env python3
"""WP5 stage A -- the resource proof for kernelFlatXs, parsed into a receipt.

WHY THIS EXISTS.  The WP5 hypothesis is a STATIC count: flatxsSolveNode()
declares ~7.2 KiB of per-thread workspace, kernelFlatXs runs one thread per
node, therefore ptxas must be spilling it to local memory and occupancy must be
poor.  That is a plausible story and plausible stories are exactly what this
campaign has been burned by (see docs/WP_PLAN_REVIEW_AND_TRACKER_20260831_KO.md
R1: the "FlatXS = 39.9 % of batch GPU time" figure that motivates WP5 came from
a harness that starved the arrival width).  So stage B does not get built on
the story.  It gets built on these numbers, or it does not get adopted.

WHAT IT READS.  Three artifact kinds, any subset:

  1. `nvcc --ptxas-options=-v` build logs (cmake -DRASBERY_PTXAS_VERBOSE=ON).
     Gives registers/thread, stack frame bytes, spill store/load bytes and
     static smem per kernel, at COMPILE time, with no device involved.
  2. `cuobjdump -res-usage <binary>` output.  Same class of facts, read back
     out of a binary that is already built -- REG/STACK/SHARED/LOCAL.
  3. Nsight Compute CSV (`ncu --csv ...`, or `ncu -i report.ncu-rep --csv
     --page details`).  The only source for the RUNTIME facts: achieved
     occupancy, the occupancy limiter, actual local load/store traffic,
     eligible warps, and kernel duration.

WHAT IT DECIDES.  The stage-A branch that the WP5 plan writes as prose:

    local memory traffic HIGH  -> PROCEED_CTA   (build/adopt the CTA kernel)
    local memory traffic LOW   -> REDESIGN      (the spill story is wrong; go
                                                 look at the stream H2D, the
                                                 61 MB download, divergence,
                                                 and the serial isotope fold
                                                 instead -- and do NOT adopt
                                                 the CTA kernel just because it
                                                 exists)

A verdict of REDESIGN is a real, useful outcome.  It is not a failure of this
tool and it must not be argued around.

WHAT IT CANNOT DECIDE.  Whether the CTA kernel is FASTER, and whether it is
BIT-IDENTICAL.  The first is the 238 runbook in
docs/WP5_FLATXS_CTA_20260831_KO.md; the second is
test/flatxs_device_replay.cu --cta.  This file never launches anything.

Usage (on 238):
    python3 tools/flatxs_resource_report.py \
        --ptxas-log  "$OUT/build_ptxas.log" \
        --res-usage  "$OUT/res_usage.txt" \
        --ncu-csv    "$OUT/flatxs_ncu.csv" \
        --json-out   "$OUT/flatxs_stageA.json"
    python3 tools/flatxs_resource_report.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The kernels this report is about.  Names are matched as substrings of the
# (possibly mangled) symbol, so both `kernelFlatXs` and
# `_Z13kernelFlatXsN7rasbery6flatxs11FlatXsViewE` hit.
REFERENCE_KERNEL = "kernelFlatXs"
CTA_KERNEL = "kernelFlatXsCta"

# ---------------------------------------------------------------------------
# The stage-A branch thresholds.
#
# LOCAL_BYTES_GATE: bytes of local/stack frame per thread above which "the
# workspace did not fit" is established rather than suspected.  A kernel with a
# handful of spilled registers sits in the tens of bytes; the WP5 hypothesis
# predicts thousands.  1 KiB is far above the noise and far below the claim, so
# it separates the two stories without being tuned to either.
LOCAL_BYTES_GATE = 1024
# OCCUPANCY_GATE: achieved occupancy (percent) under which the kernel is
# corroborated as occupancy-starved.  Reported always; it only turns a verdict
# when the local-memory evidence is already there.
OCCUPANCY_GATE = 30.0
# ---------------------------------------------------------------------------

# ncu metric names, in the spelling `ncu --csv` emits.  Several have historical
# aliases; each entry is a list and the first one present wins.
NCU_METRICS: dict[str, list[str]] = {
    "registers_per_thread": ["launch__registers_per_thread"],
    "static_smem_per_block": ["launch__shared_mem_per_block_static"],
    "dynamic_smem_per_block": ["launch__shared_mem_per_block_dynamic"],
    "achieved_occupancy_pct": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "achieved_occupancy",
    ],
    "theoretical_occupancy_pct": [
        "sm__maximum_warps_avg_per_active_cycle_pct",
        "launch__occupancy_limit_warps",
    ],
    "eligible_warps_per_scheduler": [
        "smsp__warps_eligible.avg.per_cycle_active",
    ],
    "local_load_bytes": [
        "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    ],
    "local_store_bytes": [
        "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    ],
    "local_load_inst": ["smsp__inst_executed_op_local_ld.sum"],
    "local_store_inst": ["smsp__inst_executed_op_local_st.sum"],
    "l1_hit_rate_pct": ["l1tex__t_sector_hit_rate.pct"],
    "l2_hit_rate_pct": ["lts__t_sector_hit_rate.pct"],
    "dram_pct_of_peak": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "duration_ns": ["gpu__time_duration.sum"],
    "branch_efficiency_pct": ["smsp__sass_average_branch_targets_threads_uniform.pct"],
}


def _f(text: str) -> float | None:
    """Parse an ncu metric value: '7,392' / '3.08' / 'n/a' / ''."""
    t = text.strip().replace(",", "").replace("%", "")
    if not t or t.lower() in ("n/a", "nan", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 1. The static workspace prediction, derived from the HEADERS.
#
# The 7,388-byte figure in the plan is a product of constants that live in
# src/XsReconKernel.h and src/FlatXsKernel.h.  Hard-coding it here would let
# the report keep quoting it after the isotope registry or the group count
# moved.  So it is recomputed from the source every run, and the report prints
# the factors, not just the product.
# ---------------------------------------------------------------------------
def _const(text: str, name: str) -> int:
    m = re.search(r"constexpr\s+int\s+%s\s*=\s*(\d+)\s*;" % re.escape(name), text)
    if m is None:
        raise KeyError(name)
    return int(m.group(1))


def static_workspace(root: str = ROOT) -> dict[str, Any]:
    with io.open(os.path.join(root, "src", "XsReconKernel.h"), encoding="utf-8") as fh:
        xsr = fh.read()
    with io.open(os.path.join(root, "src", "FlatXsKernel.h"), encoding="utf-8") as fh:
        fxs = fh.read()
    niso = _const(xsr, "NISO")
    ng = _const(xsr, "NG")
    n_active = _const(fxs, "N_ACTIVE")
    nmic = niso * ng
    nlsm = ng * ng
    nmsm = niso * ng * ng
    arrays = {
        "bl": n_active * ng,
        "bls": nlsm,
        "bm": n_active * nmic,
        "bms": nmsm,
        "iden": niso,
    }
    doubles = sum(arrays.values())
    return {
        "NISO": niso,
        "NG": ng,
        "N_ACTIVE": n_active,
        "NMIC": nmic,
        "NLSM": nlsm,
        "NMSM": nmsm,
        "arrays_doubles": arrays,
        "workspace_doubles": doubles,
        "workspace_bytes": doubles * 8,
        # active_xt[N_ACTIVE] ints, the one non-double local the body declares.
        "plus_active_xt_bytes": n_active * 4,
        "predicted_local_bytes_per_thread": doubles * 8 + n_active * 4,
    }


# ---------------------------------------------------------------------------
# 2. ptxas -v build log.
# ---------------------------------------------------------------------------
PTXAS_ENTRY = re.compile(
    r"ptxas info\s*:\s*Compiling entry function '([^']+)' for '([^']+)'")
PTXAS_PROPS = re.compile(r"ptxas info\s*:\s*Function properties for (\S+)")
PTXAS_FRAME = re.compile(
    r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,"
    r"\s*(\d+)\s+bytes\s+spill\s+loads")
PTXAS_USED = re.compile(r"ptxas info\s*:\s*Used\s+(\d+)\s+registers(.*)")
PTXAS_SMEM = re.compile(r"(\d+)\s+bytes\s+smem")


def parse_ptxas_log(text: str) -> dict[str, dict[str, Any]]:
    """symbol -> {arch, registers, stack_frame_bytes, spill_*_bytes, smem}.

    ptxas emits three lines per kernel in a fixed order (entry / properties /
    used), but interleaves kernels when compiling for several architectures, so
    the parser tracks the CURRENT symbol rather than assuming adjacency.
    """
    out: dict[str, dict[str, Any]] = {}
    current: str | None = None
    arch: str | None = None
    for line in text.splitlines():
        m = PTXAS_ENTRY.search(line)
        if m:
            current = m.group(1)
            arch = m.group(2)
            out.setdefault(current, {})["arch"] = arch
            continue
        m = PTXAS_PROPS.search(line)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            if arch:
                out[current].setdefault("arch", arch)
            continue
        if current is None:
            continue
        m = PTXAS_FRAME.search(line)
        if m:
            out[current]["stack_frame_bytes"] = int(m.group(1))
            out[current]["spill_store_bytes"] = int(m.group(2))
            out[current]["spill_load_bytes"] = int(m.group(3))
            continue
        m = PTXAS_USED.search(line)
        if m:
            out[current]["registers"] = int(m.group(1))
            sm = PTXAS_SMEM.search(m.group(2))
            if sm:
                out[current]["smem_bytes"] = int(sm.group(1))
    return out


# ---------------------------------------------------------------------------
# 3. cuobjdump -res-usage.
# ---------------------------------------------------------------------------
RES_FUNC = re.compile(r"^\s*Function\s+(\S+?):?\s*$")
RES_LINE = re.compile(r"\b(REG|STACK|SHARED|LOCAL)\s*:\s*(\d+)")
RES_ARCH = re.compile(r"^\s*arch\s*=\s*(\S+)")


def parse_res_usage(text: str) -> dict[str, dict[str, Any]]:
    """symbol -> {arch, registers, stack_frame_bytes, smem_bytes, local_bytes}."""
    out: dict[str, dict[str, Any]] = {}
    current: str | None = None
    arch: str | None = None
    for line in text.splitlines():
        m = RES_ARCH.match(line)
        if m:
            arch = m.group(1)
            continue
        m = RES_FUNC.match(line)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            if arch:
                out[current]["arch"] = arch
            continue
        if current is None:
            continue
        pairs = RES_LINE.findall(line)
        if not pairs:
            continue
        rec = out[current]
        for key, val in pairs:
            n = int(val)
            if key == "REG":
                rec["registers"] = n
            elif key == "STACK":
                rec["stack_frame_bytes"] = n
            elif key == "SHARED":
                rec["smem_bytes"] = n
            elif key == "LOCAL":
                rec["local_bytes"] = n
    return out


# ---------------------------------------------------------------------------
# 4. Nsight Compute CSV.
# ---------------------------------------------------------------------------
def parse_ncu_csv(text: str) -> dict[str, dict[str, float]]:
    """kernel name -> {friendly metric key: value}.

    ncu's CSV carries a preamble of `==PROF==` lines before the header, and the
    header column spelling drifts between versions, so the header is FOUND by
    looking for the three columns this parser needs rather than assumed to be
    row 0.
    """
    rows = list(csv.reader(io.StringIO(text)))
    header_idx = -1
    for i, row in enumerate(rows):
        low = [c.strip().lower() for c in row]
        if "kernel name" in low and "metric name" in low and "metric value" in low:
            header_idx = i
            break
    if header_idx < 0:
        return {}
    head = [c.strip().lower() for c in rows[header_idx]]
    k_i = head.index("kernel name")
    m_i = head.index("metric name")
    v_i = head.index("metric value")

    alias: dict[str, str] = {}
    for friendly, names in NCU_METRICS.items():
        for n in names:
            alias[n.lower()] = friendly

    out: dict[str, dict[str, float]] = {}
    for row in rows[header_idx + 1:]:
        if len(row) <= max(k_i, m_i, v_i):
            continue
        kern = row[k_i].strip()
        if not kern:
            continue
        friendly = alias.get(row[m_i].strip().lower())
        if friendly is None:
            continue
        val = _f(row[v_i])
        if val is None:
            continue
        rec = out.setdefault(kern, {})
        # ncu reports one row per launch; a kernel measured many times gets the
        # LAST launch for launch-config metrics and the MAX for traffic, which
        # is the conservative reading for a spill claim.
        if friendly in ("local_load_bytes", "local_store_bytes",
                        "local_load_inst", "local_store_inst", "duration_ns"):
            rec[friendly] = max(rec.get(friendly, 0.0), val)
        else:
            rec[friendly] = val
    return out


# ---------------------------------------------------------------------------
# 5. Fold the sources into one per-kernel record and decide the branch.
# ---------------------------------------------------------------------------
def _pick(table: dict[str, dict[str, Any]], needle: str,
          exclude: str | None = None) -> dict[str, Any]:
    """Merge every symbol whose name contains `needle` (and not `exclude`).

    `kernelFlatXs` is a substring of `kernelFlatXsCta`, so the reference lookup
    must exclude the CTA one explicitly -- a silent merge of the two would make
    the whole report meaningless in exactly the direction that would flatter
    the CTA arm.
    """
    merged: dict[str, Any] = {}
    for sym, rec in sorted(table.items()):
        if needle not in sym:
            continue
        if exclude is not None and exclude in sym:
            continue
        for k, v in rec.items():
            merged.setdefault(k, v)
        merged.setdefault("symbols", [])
        merged["symbols"].append(sym)
    return merged


def build_report(ptxas: dict[str, dict[str, Any]],
                 resusage: dict[str, dict[str, Any]],
                 ncu: dict[str, dict[str, float]],
                 root: str = ROOT) -> dict[str, Any]:
    static = static_workspace(root)
    kernels: dict[str, Any] = {}
    for label, needle, exclude in (("reference", REFERENCE_KERNEL, CTA_KERNEL),
                                   ("cta", CTA_KERNEL, None)):
        rec: dict[str, Any] = {}
        rec["ptxas"] = _pick(ptxas, needle, exclude)
        rec["res_usage"] = _pick(resusage, needle, exclude)
        rec["ncu"] = _pick(ncu, needle, exclude)  # type: ignore[arg-type]
        # The per-thread local footprint, taken from whichever source has it.
        local = None
        for src in ("res_usage", "ptxas"):
            for key in ("local_bytes", "stack_frame_bytes"):
                if rec[src].get(key):
                    local = max(local or 0, int(rec[src][key]))
        rec["local_bytes_per_thread"] = local
        rec["registers"] = (rec["res_usage"].get("registers")
                            or rec["ptxas"].get("registers")
                            or rec["ncu"].get("registers_per_thread"))
        rec["spill_bytes"] = None
        if rec["ptxas"].get("spill_store_bytes") is not None:
            rec["spill_bytes"] = (int(rec["ptxas"]["spill_store_bytes"])
                                  + int(rec["ptxas"]["spill_load_bytes"]))
        rec["smem_bytes"] = (rec["res_usage"].get("smem_bytes")
                             or rec["ptxas"].get("smem_bytes")
                             or rec["ncu"].get("static_smem_per_block"))
        rec["achieved_occupancy_pct"] = rec["ncu"].get("achieved_occupancy_pct")
        lb = rec["ncu"].get("local_load_bytes")
        sb = rec["ncu"].get("local_store_bytes")
        rec["local_traffic_bytes"] = (
            (lb or 0.0) + (sb or 0.0) if (lb is not None or sb is not None) else None)
        rec["present"] = bool(rec["ptxas"] or rec["res_usage"] or rec["ncu"])
        kernels[label] = rec

    ref = kernels["reference"]
    verdict, why = _verdict(ref, static)
    return {
        "static_workspace": static,
        "kernels": kernels,
        "gates": {
            "local_bytes_gate": LOCAL_BYTES_GATE,
            "occupancy_gate_pct": OCCUPANCY_GATE,
        },
        "stage_a_verdict": verdict,
        "stage_a_reason": why,
    }


def _verdict(ref: dict[str, Any], static: dict[str, Any]) -> tuple[str, str]:
    if not ref.get("present"):
        return ("INSUFFICIENT_DATA",
                "no ptxas / cuobjdump / ncu record for %s was supplied"
                % REFERENCE_KERNEL)
    local = ref.get("local_bytes_per_thread")
    traffic = ref.get("local_traffic_bytes")
    occ = ref.get("achieved_occupancy_pct")
    if local is None and traffic is None:
        return ("INSUFFICIENT_DATA",
                "neither a local/stack-frame figure nor local memory traffic was "
                "found; run with --ptxas-log or --res-usage, or add the local "
                "l1tex byte metrics to the ncu run")
    bits = []
    high = False
    if local is not None:
        bits.append("local/stack frame = %d B/thread (gate %d, static prediction %d)"
                    % (local, LOCAL_BYTES_GATE,
                       static["predicted_local_bytes_per_thread"]))
        high = high or local >= LOCAL_BYTES_GATE
    if traffic:
        bits.append("measured local traffic = %.0f B" % traffic)
        high = high or traffic > 0
    if occ is not None:
        bits.append("achieved occupancy = %.1f %% (gate %.1f)" % (occ, OCCUPANCY_GATE))
    reason = "; ".join(bits)
    if high:
        return ("PROCEED_CTA",
                reason + " -- the workspace does not fit, so the CTA-per-node arm "
                         "is addressing the real cost")
    return ("REDESIGN",
            reason + " -- the spill hypothesis is NOT established.  Do not adopt "
                     "the CTA kernel on this evidence; look at the delta-stream "
                     "H2D, the ~61 MB per-call download, divergence, and the "
                     "serial isotope fold instead")


def render(report: dict[str, Any]) -> str:
    s = report["static_workspace"]
    out: list[str] = []
    out.append("=== WP5 stage A -- kernelFlatXs resource proof ===")
    out.append("")
    out.append("static per-thread workspace, recomputed from the headers:")
    out.append("  NISO=%d NG=%d N_ACTIVE=%d  ->  NMIC=%d NLSM=%d NMSM=%d"
               % (s["NISO"], s["NG"], s["N_ACTIVE"], s["NMIC"], s["NLSM"], s["NMSM"]))
    for name, n in s["arrays_doubles"].items():
        out.append("    %-5s %6d doubles = %7d B" % (name, n, n * 8))
    out.append("    %-5s %6s   %9d B (active_xt ints)"
               % ("", "", s["plus_active_xt_bytes"]))
    out.append("    TOTAL %6d doubles = %7d B/thread"
               % (s["workspace_doubles"], s["predicted_local_bytes_per_thread"]))
    out.append("")
    for label in ("reference", "cta"):
        k = report["kernels"][label]
        if not k.get("present"):
            out.append("%-10s : (not present in the supplied artifacts)" % label)
            continue
        out.append("%-10s : registers=%s  local/stack=%s B/thread  spill=%s B  "
                   "smem=%s B/CTA"
                   % (label, k["registers"], k["local_bytes_per_thread"],
                      k["spill_bytes"], k["smem_bytes"]))
        n = k["ncu"]
        if n:
            out.append("             occupancy=%s %%  eligible_warps=%s  "
                       "local_traffic=%s B  dur=%s ns"
                       % (n.get("achieved_occupancy_pct"),
                          n.get("eligible_warps_per_scheduler"),
                          k["local_traffic_bytes"], n.get("duration_ns")))
            out.append("             l1_hit=%s %%  l2_hit=%s %%  dram=%s %%peak  "
                       "branch_uniform=%s %%"
                       % (n.get("l1_hit_rate_pct"), n.get("l2_hit_rate_pct"),
                          n.get("dram_pct_of_peak"),
                          n.get("branch_efficiency_pct")))
    out.append("")
    out.append("stage-A verdict: %s" % report["stage_a_verdict"])
    out.append("  %s" % report["stage_a_reason"])
    out.append("")
    out.append("NOT decided here: whether the CTA arm is faster (238 runbook, "
               "docs/WP5_FLATXS_CTA_20260831_KO.md) and whether it is bit-identical")
    out.append("(test/flatxs_device_replay.cu --cta).  A PROCEED_CTA verdict is "
               "permission to measure, not permission to adopt.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Selftest: synthetic artifacts in every format, plus the negative case.
# ---------------------------------------------------------------------------
PTXAS_FIXTURE = """
ptxas info    : 0 bytes gmem, 24 bytes cmem[3]
ptxas info    : Compiling entry function '_Z13kernelFlatXsN7rasbery6flatxs11FlatXsViewE' for 'sm_89'
ptxas info    : Function properties for _Z13kernelFlatXsN7rasbery6flatxs11FlatXsViewE
    7392 bytes stack frame, 116 bytes spill stores, 128 bytes spill loads
ptxas info    : Used 72 registers, 396 bytes cmem[0]
ptxas info    : Compiling entry function '_Z16kernelFlatXsCtaILi128EEvN7rasbery6flatxs11FlatXsViewE' for 'sm_89'
ptxas info    : Function properties for _Z16kernelFlatXsCtaILi128EEvN7rasbery6flatxs11FlatXsViewE
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, 7352 bytes smem, 396 bytes cmem[0]
"""

RES_USAGE_FIXTURE = """
Fatbin elf code:
================
arch = sm_89
code version = [1,7]

Function _Z13kernelFlatXsN7rasbery6flatxs11FlatXsViewE:
REG:72 STACK:7392 SHARED:0 LOCAL:7392 CONSTANT[0]:396 TEXTURE:0

Function _Z16kernelFlatXsCtaILi128EEvN7rasbery6flatxs11FlatXsViewE:
REG:48 STACK:0 SHARED:7352 LOCAL:0 CONSTANT[0]:396 TEXTURE:0
"""

NCU_FIXTURE = '''==PROF== Connected to process 1
"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","kernelFlatXs","Launch Statistics","launch__registers_per_thread","register/thread","72"
"0","kernelFlatXs","Occupancy","sm__warps_active.avg.pct_of_peak_sustained_active","%","11.40"
"0","kernelFlatXs","Memory","l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum","byte","1,234,567"
"0","kernelFlatXs","Memory","l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum","byte","987,654"
"0","kernelFlatXs","Speed Of Light","gpu__time_duration.sum","nsecond","3,080,000"
"1","kernelFlatXsCta","Launch Statistics","launch__registers_per_thread","register/thread","48"
"1","kernelFlatXsCta","Occupancy","sm__warps_active.avg.pct_of_peak_sustained_active","%","44.10"
"1","kernelFlatXsCta","Memory","l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum","byte","0"
"1","kernelFlatXsCta","Memory","l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum","byte","0"
'''


def selftest() -> int:
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)

    st = static_workspace()
    # The prediction must be derived, not typed: NISO/NG/N_ACTIVE come from the
    # headers, so this asserts the DERIVATION, not the number.
    check(st["workspace_doubles"] ==
          st["N_ACTIVE"] * st["NG"] + st["NLSM"] +
          st["N_ACTIVE"] * st["NMIC"] + st["NMSM"] + st["NISO"],
          "static workspace total is the sum of its five arrays")
    check(6000 < st["predicted_local_bytes_per_thread"] < 9000,
          "the derived per-thread workspace is in the ~7 KiB range WP5 claims "
          "(got %d B)" % st["predicted_local_bytes_per_thread"])
    # The CTA arm's shared workspace is the same five arrays.
    check(st["workspace_bytes"] == st["workspace_doubles"] * 8,
          "shared workspace bytes = doubles * 8")

    p = parse_ptxas_log(PTXAS_FIXTURE)
    ref_syms = [s for s in p if "kernelFlatXs" in s and "Cta" not in s]
    check(len(ref_syms) == 1, "ptxas log yields exactly one reference symbol")
    check(p[ref_syms[0]]["stack_frame_bytes"] == 7392, "ptxas stack frame parsed")
    check(p[ref_syms[0]]["spill_store_bytes"] == 116, "ptxas spill stores parsed")
    check(p[ref_syms[0]]["registers"] == 72, "ptxas registers parsed")
    cta_syms = [s for s in p if "Cta" in s]
    check(p[cta_syms[0]].get("smem_bytes") == 7352, "ptxas smem parsed for the CTA arm")

    r = parse_res_usage(RES_USAGE_FIXTURE)
    ref_syms = [s for s in r if "kernelFlatXs" in s and "Cta" not in s]
    check(r[ref_syms[0]]["local_bytes"] == 7392, "res-usage LOCAL parsed")
    check(r[ref_syms[0]]["arch"] == "sm_89", "res-usage arch parsed")

    n = parse_ncu_csv(NCU_FIXTURE)
    check("kernelFlatXs" in n and "kernelFlatXsCta" in n,
          "ncu csv yields both kernels")
    check(abs(n["kernelFlatXs"]["achieved_occupancy_pct"] - 11.40) < 1e-9,
          "ncu occupancy parsed")
    check(n["kernelFlatXs"]["local_load_bytes"] == 1234567.0,
          "ncu comma-separated byte counts parsed")

    rep = build_report(p, r, n)
    check(rep["stage_a_verdict"] == "PROCEED_CTA",
          "a 7,392 B/thread local frame with real local traffic reads PROCEED_CTA")
    check(rep["kernels"]["reference"]["local_bytes_per_thread"] == 7392,
          "the reference record does not absorb the CTA kernel's 0 B frame")
    check(rep["kernels"]["cta"]["smem_bytes"] == 7352,
          "the CTA record carries the shared workspace")
    check(rep["kernels"]["reference"]["registers"] == 72,
          "registers survive the fold")

    # NEGATIVE CONTROL 1: the substring trap.  `kernelFlatXs` is a prefix of
    # `kernelFlatXsCta`, so a lookup without the exclusion would merge them and
    # could report the CTA arm's 0 B frame as the reference's.
    merged = _pick(r, REFERENCE_KERNEL, None)
    check(len(merged.get("symbols", [])) == 2,
          "negative control: without the exclusion the lookup DOES merge both "
          "kernels (so the exclusion in build_report is load-bearing)")

    # NEGATIVE CONTROL 2: no spill -> the verdict must NOT be PROCEED_CTA.
    clean_ptxas = PTXAS_FIXTURE.replace(
        "7392 bytes stack frame, 116 bytes spill stores, 128 bytes spill loads",
        "24 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
    clean_res = RES_USAGE_FIXTURE.replace("STACK:7392", "STACK:24").replace(
        "LOCAL:7392", "LOCAL:0")
    clean_ncu = NCU_FIXTURE.replace('"1,234,567"', '"0"').replace('"987,654"', '"0"')
    rep2 = build_report(parse_ptxas_log(clean_ptxas), parse_res_usage(clean_res),
                        parse_ncu_csv(clean_ncu))
    check(rep2["stage_a_verdict"] == "REDESIGN",
          "negative control: a kernel with no spill and no local traffic must "
          "read REDESIGN, not PROCEED_CTA (got %s)" % rep2["stage_a_verdict"])

    # NEGATIVE CONTROL 3: nothing supplied -> INSUFFICIENT_DATA, never a verdict.
    rep3 = build_report({}, {}, {})
    check(rep3["stage_a_verdict"] == "INSUFFICIENT_DATA",
          "negative control: an empty artifact set decides nothing")

    # NEGATIVE CONTROL 4: an ncu CSV with no recognisable header decides nothing
    # rather than silently returning an empty (and therefore 'clean') kernel.
    check(parse_ncu_csv("garbage,not,a,report\n1,2,3,4\n") == {},
          "negative control: an unrecognisable CSV yields no metrics")

    if failures:
        print("FAIL (%d)" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("PASS  tools/flatxs_resource_report.py --selftest  "
          "(4 negative controls)")
    print(render(rep))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ptxas-log", action="append", default=[],
                    help="nvcc build log built with -DRASBERY_PTXAS_VERBOSE=ON")
    ap.add_argument("--res-usage", action="append", default=[],
                    help="output of `cuobjdump -res-usage <binary>`")
    ap.add_argument("--ncu-csv", action="append", default=[],
                    help="Nsight Compute CSV (ncu --csv, or -i rep --csv --page details)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    ptxas: dict[str, dict[str, Any]] = {}
    resusage: dict[str, dict[str, Any]] = {}
    ncu: dict[str, dict[str, float]] = {}
    for path in args.ptxas_log:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            for k, v in parse_ptxas_log(fh.read()).items():
                ptxas.setdefault(k, {}).update(v)
    for path in args.res_usage:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            for k, v in parse_res_usage(fh.read()).items():
                resusage.setdefault(k, {}).update(v)
    for path in args.ncu_csv:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            for k, v in parse_ncu_csv(fh.read()).items():
                ncu.setdefault(k, {}).update(v)

    report = build_report(ptxas, resusage, ncu)
    print(render(report))
    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print("\nwrote %s" % args.json_out)
    # Exit code carries the verdict so a runbook can branch on it without
    # re-parsing: 0 PROCEED_CTA, 2 REDESIGN, 3 INSUFFICIENT_DATA.
    return {"PROCEED_CTA": 0, "REDESIGN": 2}.get(report["stage_a_verdict"], 3)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
