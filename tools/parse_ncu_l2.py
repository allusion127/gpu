#!/usr/bin/env python3
"""Aggregate the width-sweep artifacts of tools/probe_l2_width.sh into a receipt.

W0 decision spike 4/5, parser half.  The shell script produces raw Nsight
Compute CSV (or, on the fallback path, `nvidia-smi dmon` samples); this turns
them into per-kernel-class L2 hit rates and DRAM throughput at batch widths
1 / 8 / 22 / 64 and evaluates the W2 gate.

THE GATE
    dram__throughput.avg.pct_of_peak_sustained_elapsed <= 60 % of peak.
Rev.7 5.8 forbids running heavy phases concurrently until NCU shows memory
bandwidth headroom.  Above 60 % there is nothing to overlap into and 5.8 stays
as written; below it, heavy-phase overlap becomes a legitimate W2+ candidate.

WHAT THE NUMBERS DO AND DO NOT MEAN
  * The ncu path measures real hardware counters and the gate applies directly.
  * The dmon fallback measures utilisation DUTY CYCLE, not fraction of peak
    sustained bandwidth.  A memory controller can read 95 % busy while moving
    half of peak bytes.  Fallback results are emitted under
    "dram_proxy_pct_utilisation" -- a deliberately different key from the ncu
    path's "dram_pct_of_peak" -- and the gate verdict is reported as "PROXY",
    never PASS or FAIL.  Do not quote a dmon number against the 60 % gate.
  * `-c 40` means each width's sample is roughly 0.4 of one 95-node CMFD sweep.
    Class means are stable at that sample size; per-kernel means are not, and
    are emitted with their sample count so a thin one is visible.

KERNEL CLASSES.  Assigned from the kernel name, because ncu reports names and
nothing else identifies the phase.  The BiCGSTAB inner kernels do not have
"cmfd" in their names (matvec_two_group, colored_block_sweep, update_solution,
reduce_dot*, ...), so a name-substring rule of "cmfd" would file the entire CMFD
solver under "other".  See KERNEL_CLASSES below.

Usage:
    python3 tools/parse_ncu_l2.py --outdir /tmp/w0_l2 [--json-out receipt.json]
    python3 tools/parse_ncu_l2.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

# The W2 gate, in percent of peak sustained DRAM throughput.
DRAM_GATE_PCT_OF_PEAK = 60.0

WIDTHS_EXPECTED = [1, 8, 22, 64]

HIT_RATE_METRIC = "lts__t_sector_hit_rate.pct"
DRAM_METRIC = "dram__throughput.avg.pct_of_peak_sustained_elapsed"

# Ordered: the first pattern that matches wins, so the specific CMFD stems are
# tried before the generic ones.
KERNEL_CLASSES: list[tuple[str, str]] = [
    ("nodal", r"[Nn]odal"),
    ("xsrecon", r"kernelXsRecon"),
    ("flatxs", r"kernelFlatXs"),
    (
        "cmfd",
        r"cmfd|matvec_two_group|colored_block_sweep|update_solution|update_s_jacobi"
        r"|prepare_p_jacobi|reduce_dot|reduce_norm|begin_outer|accumulate_iteration"
        r"|initialize_solver_state|store_reference_norm|refresh_operator_mirror"
        r"|finalize_status",
    ),
]

_CLASS_RE = [(name, re.compile(pat)) for name, pat in KERNEL_CLASSES]


def classify(kernel_name: str) -> str:
    for name, rx in _CLASS_RE:
        if rx.search(kernel_name):
            return name
    return "other"


def _to_float(text: str) -> float | None:
    """ncu prints thousands separators and the odd 'n/a'."""
    cleaned = text.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"n/a", "nan", "-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ncu_csv(path: Path) -> list[dict[str, Any]]:
    """Rows of {kernel, klass, metric, value} from one ncu --csv artifact.

    ncu interleaves `==PROF==` progress lines with the CSV, and the CSV header
    is not necessarily the first line.  Both are handled here rather than by
    asking the caller to pre-clean the file.
    """
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln for ln in raw if not ln.startswith("==") and ln.strip()]
    header_at = next(
        (i for i, ln in enumerate(lines) if "Metric Name" in ln and "," in ln), None
    )
    if header_at is None:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_at:])))
    out: list[dict[str, Any]] = []
    for row in reader:
        name = (row.get("Kernel Name") or row.get("Kernel") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        if not name or not metric:
            continue
        value = _to_float(row.get("Metric Value") or "")
        if value is None:
            continue
        out.append({"kernel": name, "klass": classify(name), "metric": metric,
                    "value": value})
    return out


def parse_dmon(path: Path) -> dict[str, Any]:
    """SM and memory-controller utilisation samples from `nvidia-smi dmon -s um`.

    Columns are `gpu sm mem enc dec fb bar1`.  Samples with sm == 0 are the
    idle head and tail around the run; both the full and the active-only
    aggregates are returned so a short run is not flattered by its own idle time.
    """
    if not path.is_file():
        return {}
    sm: list[float] = []
    mem: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        a, b = _to_float(parts[1]), _to_float(parts[2])
        if a is None or b is None:
            continue
        sm.append(a)
        mem.append(b)
    if not sm:
        return {}
    active = [(a, b) for a, b in zip(sm, mem) if a > 0.0]
    out: dict[str, Any] = {
        "samples": len(sm),
        "sm_pct_mean": round(statistics.fmean(sm), 3),
        "dram_proxy_pct_utilisation_mean": round(statistics.fmean(mem), 3),
        "dram_proxy_pct_utilisation_max": round(max(mem), 3),
    }
    if active:
        out["active_samples"] = len(active)
        out["sm_pct_mean_active"] = round(statistics.fmean(a for a, _ in active), 3)
        out["dram_proxy_pct_utilisation_mean_active"] = round(
            statistics.fmean(b for _, b in active), 3
        )
    return out


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "p50": round(statistics.median(values), 3),
    }


def aggregate_ncu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, list[float]]] = {}
    by_kernel: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_class.setdefault(r["klass"], {}).setdefault(r["metric"], []).append(r["value"])
        by_kernel.setdefault(r["kernel"], {}).setdefault(r["metric"], []).append(r["value"])

    classes: dict[str, Any] = {}
    for klass, metrics in sorted(by_class.items()):
        entry: dict[str, Any] = {}
        if HIT_RATE_METRIC in metrics:
            entry["l2_hit_rate_pct"] = _stats(metrics[HIT_RATE_METRIC])
        if DRAM_METRIC in metrics:
            entry["dram_pct_of_peak"] = _stats(metrics[DRAM_METRIC])
        classes[klass] = entry

    # Per-kernel detail, capped: the point is to spot an outlier kernel, not to
    # reproduce the whole ncu report in the receipt.
    kernels: dict[str, Any] = {}
    ranked = sorted(
        by_kernel.items(),
        key=lambda kv: -sum(len(v) for v in kv[1].values()),
    )[:20]
    for name, metrics in ranked:
        kernels[name] = {
            "klass": classify(name),
            "l2_hit_rate_pct_mean": (
                round(statistics.fmean(metrics[HIT_RATE_METRIC]), 3)
                if HIT_RATE_METRIC in metrics else None
            ),
            "dram_pct_of_peak_mean": (
                round(statistics.fmean(metrics[DRAM_METRIC]), 3)
                if DRAM_METRIC in metrics else None
            ),
            "samples": sum(len(v) for v in metrics.values()),
        }

    all_dram = [r["value"] for r in rows if r["metric"] == DRAM_METRIC]
    all_hit = [r["value"] for r in rows if r["metric"] == HIT_RATE_METRIC]
    return {
        "classes": classes,
        "kernels": kernels,
        "all_kernels": {
            "l2_hit_rate_pct": _stats(all_hit) if all_hit else None,
            "dram_pct_of_peak": _stats(all_dram) if all_dram else None,
        },
    }


def build_receipt(outdir: Path) -> dict[str, Any]:
    meta_path = outdir / "l2_width_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tool = meta.get("tool", "unknown")
    widths = meta.get("widths") or WIDTHS_EXPECTED

    receipt: dict[str, Any] = {
        "probe": "l2_width",
        "tool": tool,
        "deck": meta.get("deck"),
        "bin": meta.get("bin"),
        "kernel_regex": meta.get("kernel_regex"),
        "launch_count": meta.get("launch_count"),
        "widths": widths,
        "widths_expected": WIDTHS_EXPECTED,
        "widths_match_expected": list(widths) == WIDTHS_EXPECTED,
        "gate": {
            "name": "W2 dram headroom",
            "metric": DRAM_METRIC,
            "threshold_pct_of_peak": DRAM_GATE_PCT_OF_PEAK,
        },
        "by_width": {},
    }

    verdicts: dict[str, str] = {}
    for w in widths:
        key = str(w)
        if tool == "ncu":
            rows = parse_ncu_csv(outdir / f"ncu_w{w}.csv")
            if not rows:
                receipt["by_width"][key] = {"status": "no_data"}
                verdicts[key] = "UNKNOWN"
                continue
            agg = aggregate_ncu(rows)
            agg["status"] = "ok"
            receipt["by_width"][key] = agg
            headline = agg["all_kernels"]["dram_pct_of_peak"]
            if headline is None:
                verdicts[key] = "UNKNOWN"
            else:
                verdicts[key] = (
                    "PASS" if headline["mean"] <= DRAM_GATE_PCT_OF_PEAK else "FAIL"
                )
        else:
            samples = parse_dmon(outdir / f"dmon_w{w}.txt")
            if not samples:
                receipt["by_width"][key] = {"status": "no_data"}
                verdicts[key] = "UNKNOWN"
                continue
            samples["status"] = "ok"
            samples["note"] = (
                "utilisation duty cycle, NOT fraction of peak sustained bandwidth; "
                "not comparable to the 60% gate"
            )
            receipt["by_width"][key] = samples
            verdicts[key] = "PROXY"

    receipt["gate"]["verdict_by_width"] = verdicts
    # The headline is the widest width that produced data: that is the width the
    # campaign intends to ship (M64).
    shipping = None
    for w in sorted((int(x) for x in widths), reverse=True):
        if receipt["by_width"].get(str(w), {}).get("status") == "ok":
            shipping = w
            break
    receipt["gate"]["shipping_width"] = shipping
    receipt["gate"]["verdict"] = verdicts.get(str(shipping), "UNKNOWN") if shipping else "UNKNOWN"
    return receipt


# ---------------------------------------------------------------------------
# Self-test: synthesise both artifact shapes and parse them, so the parser is
# exercised on a machine with no GPU, no ncu and no binary.
# ---------------------------------------------------------------------------
SELFTEST_NCU_CSV = '''==PROF== Connected to process 12345
"ID","Process ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","12345","matvec_two_group","Memory","lts__t_sector_hit_rate.pct","%","71.20"
"0","12345","matvec_two_group","Memory","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","44.10"
"1","12345","update_solution","Memory","lts__t_sector_hit_rate.pct","%","66.00"
"1","12345","update_solution","Memory","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","51.30"
"2","12345","kNodalMatEven","Memory","lts__t_sector_hit_rate.pct","%","88.40"
"2","12345","kNodalMatEven","Memory","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","12.00"
"3","12345","cmfd_src_build","Memory","lts__t_sector_hit_rate.pct","%","55.00"
"3","12345","cmfd_src_build","Memory","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","1,002.00"
==PROF== Disconnected
'''

SELFTEST_DMON = """# gpu    sm   mem   enc   dec    fb  bar1
# Idx     %     %     %     %    MB    MB
    0     0     0     0     0   500     3
    0    91    58     0     0  9000     5
    0    88    61     0     0  9100     5
    0     0     0     0     0   500     3
"""


def selftest() -> int:
    failures = 0

    def check(cond: bool, message: str) -> None:
        nonlocal failures
        if not cond:
            print(f"parse_ncu_l2 selftest: FAIL: {message}", file=sys.stderr)
            failures += 1

    # Classification is the load-bearing part: a CMFD inner kernel filed under
    # "other" would make the whole receipt say nothing about CMFD.
    check(classify("matvec_two_group_f32") == "cmfd", "matvec_two_group_f32 not cmfd")
    check(classify("colored_block_sweep") == "cmfd", "colored_block_sweep not cmfd")
    check(classify("update_solution") == "cmfd", "update_solution not cmfd")
    check(classify("reduce_dot2_stage1") == "cmfd", "reduce_dot2_stage1 not cmfd")
    check(classify("cmfd_updls") == "cmfd", "cmfd_updls not cmfd")
    check(classify("kNodalMatEven") == "nodal", "kNodalMatEven not nodal")
    check(classify("kernelXsRecon") == "xsrecon", "kernelXsRecon not xsrecon")
    check(classify("kernelFlatXs") == "flatxs", "kernelFlatXs not flatxs")
    check(classify("some_unrelated_kernel") == "other", "unknown kernel not other")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "l2_width_meta.json").write_text(
            json.dumps({"probe": "l2_width", "tool": "ncu", "widths": [1, 8, 22, 64],
                        "deck": "cy02_step1.json", "bin": "RASBERY",
                        "kernel_regex": "regex:...", "launch_count": 40}),
            encoding="utf-8",
        )
        for w in (1, 8, 22, 64):
            (d / f"ncu_w{w}.csv").write_text(SELFTEST_NCU_CSV, encoding="utf-8")
        r = build_receipt(d)
        check(r["tool"] == "ncu", "tool not carried through")
        check(r["widths_match_expected"], "width list did not match 1/8/22/64")
        cmfd = r["by_width"]["64"]["classes"]["cmfd"]
        check(cmfd["l2_hit_rate_pct"]["n"] == 3, "cmfd hit-rate sample count wrong")
        # 1,002.00 must survive the thousands separator.
        check(cmfd["dram_pct_of_peak"]["max"] == 1002.0, "thousands separator not stripped")
        check(r["by_width"]["64"]["classes"]["nodal"]["l2_hit_rate_pct"]["mean"] == 88.4,
              "nodal hit rate wrong")
        check(r["gate"]["shipping_width"] == 64, "shipping width not the widest with data")
        check(r["gate"]["verdict"] == "FAIL",
              "a mean dram of 365.8% must not pass the 60% gate")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "l2_width_meta.json").write_text(
            json.dumps({"probe": "l2_width", "tool": "nvidia-smi-dmon",
                        "widths": [1, 8, 22, 64]}),
            encoding="utf-8",
        )
        for w in (1, 8, 22, 64):
            (d / f"dmon_w{w}.txt").write_text(SELFTEST_DMON, encoding="utf-8")
        r = build_receipt(d)
        check(r["tool"] == "nvidia-smi-dmon", "fallback tool not recorded")
        check(r["gate"]["verdict"] == "PROXY",
              "the dmon fallback must never report PASS or FAIL against the gate")
        w64 = r["by_width"]["64"]
        check(w64["samples"] == 4, "dmon sample count wrong")
        check(w64["active_samples"] == 2, "dmon active-sample filter wrong")
        check("dram_pct_of_peak" not in w64,
              "the fallback must not emit the ncu-path key name")

    if failures:
        print(f"parse_ncu_l2 selftest: {failures} failure(s)", file=sys.stderr)
        return 1
    print("parse_ncu_l2 selftest: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, help="artifact directory from probe_l2_width.sh")
    ap.add_argument("--json-out", type=Path, help="also write the receipt here")
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline parser self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.outdir is None:
        ap.error("--outdir is required (or use --selftest)")
    if not args.outdir.is_dir():
        print(f"parse_ncu_l2: not a directory: {args.outdir}", file=sys.stderr)
        return 2

    receipt = build_receipt(args.outdir)
    text = json.dumps(receipt, indent=2, sort_keys=False)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
