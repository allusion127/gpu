#!/usr/bin/env python3
"""Read `[RASBERY][CMFD][COMPACT]` receipts -- with the SOURCE as the ladder.

WHY THIS MODULE EXISTS (review doc R2).  The plan's WP3 says "bucket은
`1,2,4,8,16,32,64`로 제한" -- seven rungs.  The tree has NINE:
`{1, 2, 4, 8, 16, 24, 32, 48, 64}`, in three places that must agree
(`src/GpuPhaseScheduler.h` kDispatchBuckets, `src/CudaBICGBackend.cu` kBuckets,
`src/CudaXsReconBackend.cu` kBuckets) and a `bucket_histogram` printed with one
entry per rung.  **The source is the truth**: 24 and 48 are the rungs that
cover the 238 M64 arrival width of 14.5 and its neighbourhood, and dropping
them would make the padding this WP is pricing WORSE, not the receipt shorter.

So a parser must not carry a rung count of its own.  This one reads the ladder
out of the header and zips it with whatever the receipt printed:

  * a histogram the same length as the ladder is labelled rung by rung;
  * a histogram of a DIFFERENT length is reported as a mismatch and returned
    unlabelled -- never truncated to fit, never zero-padded to fit.  Silently
    fitting is how a receipt and a parser drift apart for a whole campaign;
  * a run with no ladder available (no source tree) is labelled by index.

USE
    from cmfd_compact_receipt import parse_compact_receipts, bucket_ladder
    for r in parse_compact_receipts(log_text):
        print(r["padding_fraction"], r["buckets"])

    python tools/cmfd_compact_receipt.py run.log      # the WP3 pricing table
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPACT_RECEIPT = re.compile(r"\[RASBERY\]\[CMFD\]\[COMPACT\]\s*(\{.*\})")

# Where each copy of the ladder lives.  The scheduler's is the one the others
# have to equal (tools/test_cmfd_slot_compaction_contract.py holds them to it).
LADDER_SOURCES = (
    (os.path.join("src", "GpuPhaseScheduler.h"), r"kDispatchBuckets\[\]\s*=\s*\{([^}]*)\}"),
    (os.path.join("src", "CudaBICGBackend.cu"), r"kBuckets\[\]\s*=\s*\{([^}]*)\}"),
    (os.path.join("src", "CudaXsReconBackend.cu"), r"kBuckets\[\d*\]\s*=\s*\{([^}]*)\}"),
)

# The fields src/CudaBICGBackend.cu:5620-5645 prints.  Listed so a receipt that
# LOST one is visible as missing rather than as a zero.
COMPACT_FIELDS = ("enabled", "logical_drives", "physical_slot_blocks", "padding_blocks",
                  "padding_fraction", "bucket_graphs", "bucket_histogram")


def _read(path: str) -> str | None:
    try:
        with open(os.path.join(ROOT, path), "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except OSError:
        return None


def bucket_ladder(root: str | None = None) -> list[int]:
    """The dispatch ladder, read from the source.  [] when it cannot be read.

    No default list: a parser that falls back to a hard-coded ladder is exactly
    the thing R2 warns about, because it will keep parsing after the source
    changes and will label the rungs wrong.
    """
    global ROOT
    saved = ROOT
    if root is not None:
        ROOT = root
    try:
        for path, pattern in LADDER_SOURCES:
            text = _read(path)
            if text is None:
                continue
            match = re.search(pattern, text)
            if not match:
                continue
            rungs = [int(t.strip()) for t in match.group(1).split(",") if t.strip()]
            if rungs:
                return rungs
        return []
    finally:
        ROOT = saved


def label_histogram(histogram, ladder) -> tuple[list[tuple[str, int]], str]:
    """(labelled rungs, mismatch reason).  Tolerant of any ladder length."""
    counts = [int(x) for x in (histogram or [])]
    if not ladder:
        return ([(f"[{i}]", n) for i, n in enumerate(counts)],
                "no ladder in the source tree: rungs are labelled by index")
    if len(counts) != len(ladder):
        return ([(f"[{i}]", n) for i, n in enumerate(counts)],
                f"receipt has {len(counts)} bucket(s), the source ladder has "
                f"{len(ladder)} ({ladder}). The SOURCE is the truth (review R2); this "
                "histogram is reported unlabelled rather than truncated or padded to fit")
    return ([(f"<={rung}", n) for rung, n in zip(ladder, counts)], "")


def parse_compact_receipts(text: str, ladder: list[int] | None = None) -> list[dict]:
    """Every [CMFD][COMPACT] receipt in *text*, labelled and checked."""
    rungs = bucket_ladder() if ladder is None else list(ladder)
    out: list[dict] = []
    for match in COMPACT_RECEIPT.finditer(text):
        try:
            raw = json.loads(match.group(1))
        except ValueError:
            out.append({"parse_error": match.group(1)[:200], "missing": list(COMPACT_FIELDS)})
            continue
        record = dict(raw)
        record["missing"] = [f for f in COMPACT_FIELDS if f not in raw]
        buckets, mismatch = label_histogram(raw.get("bucket_histogram"), rungs)
        record["buckets"] = buckets
        record["ladder"] = rungs
        record["ladder_mismatch"] = mismatch
        physical = float(raw.get("physical_slot_blocks", 0) or 0)
        padding = float(raw.get("padding_blocks", 0) or 0)
        total = physical + padding
        # Recomputed rather than trusted: `padding_fraction` and the two block
        # counts are three numbers for one fact, and a disagreement between
        # them is a finding, not a rounding.
        record["padding_fraction_recomputed"] = padding / total if total > 0 else 0.0
        out.append(record)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: {os.path.basename(argv[0])} <log> [<log> ...]", file=sys.stderr)
        return 2
    rungs = bucket_ladder()
    print(f"# bucket ladder from source: {rungs or '(unavailable)'}")
    status = 0
    for path in argv[1:]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            status = 2
            continue
        records = parse_compact_receipts(text, rungs)
        if not records:
            print(f"{path}: no [RASBERY][CMFD][COMPACT] receipt")
            continue
        for record in records:
            if record.get("parse_error"):
                print(f"{path}: unparseable receipt {record['parse_error']}")
                status = 1
                continue
            print("%s: enabled=%s logical_drives=%s physical=%s padding=%s "
                  "padding_fraction=%.4f (recomputed %.4f) bucket_graphs=%s"
                  % (path, record.get("enabled"), record.get("logical_drives"),
                     record.get("physical_slot_blocks"), record.get("padding_blocks"),
                     float(record.get("padding_fraction", 0.0) or 0.0),
                     record["padding_fraction_recomputed"], record.get("bucket_graphs")))
            print("    buckets: " + ", ".join(f"{name}={n}" for name, n in record["buckets"]))
            if record["ladder_mismatch"]:
                print("    LADDER: " + record["ladder_mismatch"])
                status = 1
            if record["missing"]:
                print("    MISSING FIELDS: " + ", ".join(record["missing"]))
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
