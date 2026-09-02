#!/usr/bin/env python3
"""WP11 -- the long-stability soak, and the receipts that must be zero at exit.

WHAT A SOAK IS FOR, AND WHY A THROUGHPUT RUN IS NOT ONE.  Every number the
campaign has measured came off a wave or two.  The defects WP8-WP10 can
introduce are not wave-shaped: a slot the refill ledger hands out twice, a
pinned host range whose lease outlived its Driver, a graph capture that unwound
and left an allocation behind, a cohort key that started covering something a
candidate changes.  Each of those is invisible for one generation and fatal over
twenty, and each is already COUNTED by a receipt this tree prints.  So the soak
does not measure anything new.  It runs long enough for the counters to have
something to say, and then it reads them.

THE PLAN'S LIST, VERBATIM (Sec WP11 "종료 시 0이어야 하는 receipt"), and where
each one actually comes from:

    duplicates                  [REFILL].duplicates / [EVALUATOR].slot_duplicates
    stale_tenants               [REFILL].stale_tenants / [EVALUATOR].slot_stale_tenants
    double_releases             [REFILL].double_releases / [EVALUATOR].slot_double_releases
    alloc_in_capture            [CUDA][CAPTURE_ARBITER].alloc_in_capture
    captures_unwound            [CUDA][CAPTURE_ARBITER].captures_unwound
    graph/cmfd/nodal/flatxs/ppr fallbacks    [GPU_FULL].*_fallbacks, under RASBERY_GPU_FULL=1
    queue duplicate claims      [EVALUATOR].slot_duplicates (one queue, one counter)
    output collisions           this driver's own: two cases on one --raso
    cross_case_digest_mismatch  [EVALUATOR].isolation_mismatches
    pin_live_ranges_between_waves           [EVALUATOR], asserted per wave by the binary

TWO OF THEM ARE CONDITIONAL AND SAYING SO IS THE POINT.  The GPU-full fallbacks
are only a zero-assertion when RASBERY_GPU_FULL=1 -- without it a fallback is
legal and counting it as a failure would make the soak refuse a legitimate arm.
And `restarts` is bounded by `--expect-restarts`, which is ZERO by default -- not
because restarts are impossible but because the poison this soak plants is a
deck that does not PARSE, and that is supposed to fail one case and leave the
process answering.  A restart there is the poison taking sixty-three other
candidates down with it, which is the failure-isolation defect the poison exists
to find.  A run that means to kill the child raises the bound and thereby says
so.

WHAT IS MEASURED RATHER THAN ASSERTED.

  c/h per generation      A drift budget, not a target.  The first generation is
                          the reference; every later one must stay within
                          --drift (default 3 %) of the MEDIAN.  A soak whose
                          throughput decays 0.5 % per generation ends 10 % down
                          and every individual step looks like noise -- which is
                          exactly what a slow leak looks like from inside.
  VRAM                    THE PROCESS'S OWN device footprint, sampled between
                          generations: `[EVALUATOR][MEM] vram_mb` first, then
                          nvidia-smi's per-compute-app row for this pid, and a
                          board total only as a last resort and only ever
                          LABELLED as one.  WP10.6: the 238 soak reported a
                          298 MB <-> 47,000 MB "sawtooth" that was an
                          eight-process MPS batch on the OTHER board, sampled
                          because `--gpu` defaulted to 0 while the child ran
                          under CUDA_VISIBLE_DEVICES=1.  A board-scoped sample
                          with other tenants on it is now reported and NOT
                          convicted.  A LEAK IS A SLOPE, not a level: warm
                          plateau is expected and high, so the test is MB per
                          generation over the second half of the run, after the
                          caches have stopped growing.
  host RSS                /proc/<pid>/status VmRSS, same rule.  Unavailable off
                          Linux, and the report says `null` rather than 0 --
                          "not measured" and "measured zero" are different
                          claims and only one of them is evidence.

THE WORKLOAD.  N generations x W cases, and every case in it is doing something
the plan asks a soak to exercise:

    mixed light/full      --light-fraction of each generation is `light`, the
                          rest `full`.  Both write; only one writes HDF5.
    warm-start chain      case i of generation g warm-starts from case i of
                          generation g-1 and saves its own state for g+1.  A
                          chain is what finds a warm-state file that is written
                          but never closed.
    mixed fidelity        --screen-fraction of each generation runs L3coarse on
                          the coarse burnup grid (WP10.3), and one promoted
                          strict re-run per generation links back to the screen
                          it replaces.  A soak on ONE fidelity would never
                          exercise the path where two cases in one wave converge
                          differently.
    one poisoned case     a deck that does not parse.  The real binary reaches
                          this through IO::ReadInput throwing inside
                          runOneCase's try: ONE case fails, the process keeps
                          answering, and the wave receipt says 63/64.  If it
                          instead takes the process down, the dispatcher
                          restarts it and `restarts` exceeds --expect-restarts,
                          which is the assertion.

USE

    python tools/soak_run.py --deck kngr_238.json --workdir /tmp/soak \\
        --binary ./RASBERY --generations 20 --width 64

    # the contract-test shape: no GPU, no CUDA, no 40 s case
    python tools/soak_run.py --deck any.json --workdir /tmp/soak \\
        --command "python tools/fake_rasbery_child.py" --generations 3 --width 4

Exit status is 0 only when every zero-receipt is zero, the drift budget holds
and no leak slope is above its threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exact_audit  # noqa: E402
from run_multi_gpu_batch import EvaluatorSession  # noqa: E402

# ---------------------------------------------------------------------------
# The zero-receipt table.  ONE list, and the report prints it whole -- including
# the ones that were zero, because "we checked and it was zero" and "we did not
# check" are different statements and a report that only shows failures cannot
# distinguish them.
# ---------------------------------------------------------------------------

#: (name, receipt tag, field).  Read from the session's whole transcript.
ZERO_RECEIPTS: tuple[tuple[str, str, str], ...] = (
    ("duplicates", "[RASBERY][EVALUATOR]", "slot_duplicates"),
    ("stale_tenants", "[RASBERY][EVALUATOR]", "slot_stale_tenants"),
    ("double_releases", "[RASBERY][EVALUATOR]", "slot_double_releases"),
    ("refill_duplicates", "[RASBERY][REFILL]", "duplicates"),
    ("refill_stale_tenants", "[RASBERY][REFILL]", "stale_tenants"),
    ("refill_double_releases", "[RASBERY][REFILL]", "double_releases"),
    ("alloc_in_capture", "[RASBERY][CUDA][CAPTURE_ARBITER]", "alloc_in_capture"),
    ("captures_unwound", "[RASBERY][CUDA][CAPTURE_ARBITER]", "captures_unwound"),
    ("cross_case_digest_mismatch", "[RASBERY][EVALUATOR]", "isolation_mismatches"),
    ("pin_live_ranges_between_waves", "[RASBERY][EVALUATOR]",
     "pin_live_ranges_between_waves"),
)

#: Only a zero-assertion under RASBERY_GPU_FULL=1.  Without the gate a fallback
#: is a legal thing for the binary to do and failing the soak on one would make
#: it refuse an arm nobody said was wrong.
#: WP10.7.  `outer_fallbacks` JOINED THIS LIST, and its absence was not an
#: oversight anybody had argued for.  The 238 GPU1 arm-A soak reported
#: `outer_fallbacks:9` -- NINE cases killed by the fail-closed outer seam, a
#: materially larger category than the four `flatxs_fallbacks` the same receipt
#: carried -- and the soak's verdict never looked at the number, because the
#: only counter it did not read was that one.  Every other subsystem in the
#: GPU_FULL receipt was asserted; the outer is not a special case, and the
#: allowance list (gpufull::kGpuFullAllowedOuterRefusals) is exactly the
#: mechanism that keeps its legitimate host regions -- the Wielandt warm-up --
#: out of this counter in the first place.
GPU_FULL_FALLBACKS: tuple[str, ...] = (
    "cmfd_fallbacks", "outer_fallbacks", "nodal_fallbacks", "xsrecon_fallbacks",
    "flatxs_fallbacks", "xe_fallbacks", "ppr_fallbacks", "cram_fallbacks",
    "graph_fallbacks",
)


def receipts_of(text: str, tag: str) -> list[dict]:
    """Every `<tag> {json}` object in *text*.

    Tag matching is on the literal tag followed by whitespace and `{`, so
    `[RASBERY][EVALUATOR]` cannot swallow `[RASBERY][EVALUATOR][CASE]` -- the
    latter has `[` where this needs a space, which is the same discrimination
    run_multi_gpu_batch's EVALUATOR_PROCESS regex makes.
    """
    out: list[dict] = []
    start = 0
    while True:
        index = text.find(tag, start)
        if index < 0:
            return out
        cursor = index + len(tag)
        start = cursor
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue
        end = text.find("\n", cursor)
        blob = text[cursor:end if end >= 0 else len(text)]
        try:
            value = json.loads(blob)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sampled_gpu(env: "dict[str, str]", requested: str) -> str:
    """The board the CHILD is on, which is not always the one --gpu names.

    WP10.6.  The 238 20-generation soak was launched with `CUDA_VISIBLE_DEVICES=1`
    in the environment and no `--gpu` flag.  `env.setdefault` therefore left the
    child on GPU1 -- correctly -- and the sampler went on querying
    `nvidia-smi -i 0`, because that is what `--gpu` still defaulted to.  GPU0 was
    running an eight-process MPS batch at the time, so the VRAM trace the report
    published (298 MB <-> 47,000 MB, "a sawtooth") was a faithful measurement of
    somebody else's work.  `CUDA_VISIBLE_DEVICES` wins here: nvidia-smi's `-i`
    is a board index and ignores it, so the translation has to be done once,
    explicitly, rather than assumed to be the identity.
    """
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return str(requested)
    first = visible.split(",")[0].strip()
    return first or str(requested)


class VramSample:
    """A VRAM reading, and WHOSE it is.

    `scope` is the whole point.  A board reading with other tenants on it is not
    this process's footprint, and a leak gate that cannot tell the two apart
    reports the neighbours' allocations as its own growth -- which is exactly
    the finding WP10.6 had to withdraw.
    """

    __slots__ = ("used_mb", "scope", "foreign_pids", "board_mb")

    def __init__(self, used_mb: float | None, scope: str,
                 foreign_pids: "list[int]", board_mb: float | None) -> None:
        self.used_mb = used_mb
        self.scope = scope                  # "process" | "board" | "none"
        self.foreign_pids = foreign_pids
        self.board_mb = board_mb


def _nvidia_smi(args: "list[str]") -> "list[str] | None":
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", *args], capture_output=True, text=True,
            timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip().splitlines()


def sample_vram(gpu: str, pid: int | None) -> VramSample:
    """This process's device memory on *gpu*, MB, and every other tenant's pid.

    None rather than 0.0 on every failure path.  A soak that reported 0 MB
    because nvidia-smi was missing would report a perfectly flat VRAM trace and
    pass its leak check for the one reason that proves nothing.
    """
    board = None
    rows = _nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits",
                        "-i", str(gpu)])
    if rows:
        try:
            board = float(rows[0].strip())
        except ValueError:
            board = None

    mine: float | None = None
    foreign: list[int] = []
    apps = _nvidia_smi(["--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits", "-i", str(gpu)])
    if apps is not None:
        for row in apps:
            parts = [f.strip() for f in row.split(",")]
            if len(parts) < 2:
                continue
            try:
                row_pid, row_mb = int(parts[0]), float(parts[1])
            except ValueError:
                continue
            if pid is not None and row_pid == pid:
                mine = (mine or 0.0) + row_mb
            else:
                foreign.append(row_pid)
        if mine is not None:
            return VramSample(mine, "process", foreign, board)
        # The child is on the board but the driver did not list it (MPS routes
        # every client's memory through the server process, and some drivers
        # report nothing for a compute app at all).  Say "board", loudly,
        # rather than silently offering a board number as a process number.
    if board is None:
        return VramSample(None, "none", foreign, None)
    return VramSample(board, "board", foreign, board)


def sample_vram_mb(gpu: str, pid: int | None = None) -> float | None:
    """Back-compatible scalar, used by the contract test's negative controls."""
    return sample_vram(gpu, pid).used_mb


def sample_rss_mb(pid: int | None) -> float | None:
    """The child's resident set, MB, or None where /proc does not exist."""
    if pid is None:
        return None
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0
                except ValueError:
                    return None
    return None


def leak_slope_mb_per_generation(series: Sequence[float | None]) -> float | None:
    """MB per generation over the SECOND HALF of *series*, or None.

    THE SECOND HALF, deliberately.  A warm plateau is expected and it is steep:
    the library loads, the arenas stand up, the graph cache fills, and the first
    few generations climb for reasons that are the design working.  A slope
    fitted over the whole run would report that climb as a leak on every healthy
    soak, and the threshold would then have to be set so high that it caught
    nothing.  What a leak looks like is a line that is still climbing after the
    caches have stopped, which is what the second half is.
    """
    values = [(i, v) for i, v in enumerate(series) if v is not None]
    if len(values) < 4:
        return None
    tail = values[len(values) // 2:]
    if len(tail) < 2:
        return None
    xs = [float(i) for i, _ in tail]
    ys = [v for _, v in tail]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def linear_slope_mb_per_generation(series: Sequence[float | None]) -> float | None:
    """MB per generation over EVERY present sample of *series*, or None.

    Reported beside the gated slope, never gated on.  It is the number a reader
    computes by eye from the table, and publishing it is what stops an argument
    about whether the tool and the eye disagree: at 55c0dce they did, and only
    the tool's number was in the report.
    """
    values = [(i, v) for i, v in enumerate(series) if v is not None]
    if len(values) < 2:
        return None
    xs = [float(i) for i, _ in values]
    ys = [v for _, v in values]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def growth_slopes(series: Sequence[float | None], warmup: int
                  ) -> "dict[str, float | None]":
    """Every slope worth reporting, and which one the gate uses.

    WHY THE WARM-UP GENERATIONS COME OUT.  The host 181 soak at 55c0dce measured
    RSS 1263 -> 4251 -> 4502 -> 4582 -> 4557 MB: a ONE-TIME +2988 MB step as the
    first generation stood the arenas, the device library and the graph cache
    up, and then a curve that is nearly flat and ends by going DOWN.  Fitting a
    line through the step reports a leak that is a stand-up cost, and no
    threshold can be set that tells those apart -- the step is 300x the budget.
    `leak_slope_mb_per_generation` already dropped the first half, which was
    enough at five generations and stops being enough as a soak gets longer (at
    50 generations the second half starts at 25 and the step is long forgotten,
    but so is everything else the gate wanted to see).  Dropping a named number
    of WARM-UP generations is the rule that holds at both lengths, and the raw
    slope is published beside it so nothing is hidden by the choice.
    """
    tail = list(series)[max(0, warmup):]
    return {
        # THE GATE: a straight fit through the POST-WARM-UP series, all of it.
        #
        # Not its second half.  Halving was the old way of dodging the stand-up
        # step, and once the step is dropped by name it only throws samples
        # away: at five generations the post-warm-up second half is TWO points,
        # and a gate that turns on two points is a coin. The 55c0dce numbers are
        # the worked example -- second-half-of-post-warm-up reads -25.4 MB/gen
        # (PASS) from the last two samples while the run actually gained 305 MB
        # over the three post-warm-up generations, which is +99.6 MB/gen and a
        # correct FAIL.  The halved figures stay in the receipt below; they are
        # not what decides.
        "slope_mb_per_generation": linear_slope_mb_per_generation(tail),
        "slope_post_warmup_second_half_mb_per_generation":
            leak_slope_mb_per_generation(tail),
        # Everything a reader might otherwise recompute by hand.
        "slope_raw_mb_per_generation": linear_slope_mb_per_generation(series),
        "slope_post_warmup_mb_per_generation": linear_slope_mb_per_generation(tail),
        "slope_second_half_all_mb_per_generation":
            leak_slope_mb_per_generation(series),
        "warmup_generations": warmup,
    }


#: A mover has to account for at least this share of the observed RSS growth
#: before it is offered as an explanation rather than as a note.  5% is chosen
#: to be forgiving -- a real container is usually most of the growth -- while
#: still refusing the case that made this necessary: `case_samples` grew by 72
#: doubles (576 bytes) on a run whose RSS grew by 3.3 GB, and was reported as
#: THE moving container because it was the only counter that moved at all.
ATTRIBUTION_SHARE = 0.05


def _mb(byte_delta: float) -> float:
    return byte_delta / (1024.0 * 1024.0)


def _paired(first: dict, last: dict, key: str) -> "tuple[float, float] | None":
    """(first, last) for a numeric receipt field, or None when either is absent.

    None rather than 0.0 for a missing field: an older binary that never printed
    the field and a run where the field really was zero are different facts, and
    only one of them is evidence.
    """
    head, tail = first.get(key), last.get(key)
    if isinstance(head, (int, float)) and isinstance(tail, (int, float)):
        return float(head), float(tail)
    return None


def attribute_rss_growth(generations: "Sequence[GenerationResult]",
                         rss_growth_mb: float | None = None) -> list[str]:
    """Name the container -- and say whether it is big enough to be the cause.

    WHY THIS EXISTS.  The host 181 soak at `91004f7` found RSS climbing
    +17.41 MB/generation with every mechanism receipt at zero, and the finding
    said only that.  A number with no suspects cannot be closed on a host nobody
    can attach a profiler to, and at 10k generations that slope is 170 GB.  So
    the evaluator prints `[RASBERY][EVALUATOR][MEM]` between generations with
    the size of every process-lifetime container beside its own RSS, and this
    walks the first and last of them.

    WHY IT WAS NOT ENOUGH, AND WHAT WP10.5 CHANGED.  At `55c0dce` it worked and
    was wrong: it named `cache_entries.case_samples 18 -> 72` as the container
    that moved on a run that gained 3.3 GB.  It was the only counter that moved,
    and it moved by 576 bytes.  A reader who trusts an attribution spends a day
    on the wrong container, which is worse than no attribution -- so a mover is
    now WEIGHED.  Anything the receipt can price is compared against the growth
    it is supposed to explain, and a mover that cannot explain it is reported as
    exactly that, by name and by size.  All of them flat, or all of them too
    small, is the same answer and a useful one: the growth is not in the caches
    this receipt can see.
    """
    receipts = [g.mem for g in generations if isinstance(g.mem, dict)]
    if len(receipts) < 2:
        return ["and the binary printed no [RASBERY][EVALUATOR][MEM] receipt, so the "
                "growth cannot be attributed to a container from this run: rebuild at "
                "a commit that carries WP10.4 and repeat"]
    first, last = receipts[0], receipts[-1]

    # What the process itself says it gained, when the caller did not say.
    if rss_growth_mb is None:
        head, tail = first.get("rss_mb"), last.get("rss_mb")
        if isinstance(head, (int, float)) and isinstance(tail, (int, float)):
            rss_growth_mb = float(tail) - float(head)

    def moved_in(group: str) -> list[tuple[str, float, float]]:
        before, after = first.get(group) or {}, last.get(group) or {}
        if not isinstance(before, dict) or not isinstance(after, dict):
            return []
        out = []
        for name in sorted(set(before) | set(after)):
            a, b = before.get(name, 0), after.get(name, 0)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != b:
                out.append((name, float(a), float(b)))
        return out

    byte_moves = {name: (a, b) for name, a, b in moved_in("cache_bytes")}
    pinned = (first.get("cuda_host_bytes"), last.get("cuda_host_bytes"))
    if all(isinstance(v, (int, float)) for v in pinned) and pinned[0] != pinned[1]:
        byte_moves["cuda_host_bytes"] = (float(pinned[0]), float(pinned[1]))

    explains: list[str] = []
    too_small: list[str] = []
    unpriced: list[str] = []

    def weigh(label: str, delta_bytes: float) -> None:
        gained = _mb(delta_bytes)
        if (rss_growth_mb is None or rss_growth_mb <= 0.0
                or gained >= ATTRIBUTION_SHARE * rss_growth_mb):
            explains.append(f"{label} (+{gained:.3f} MB)")
        else:
            too_small.append(f"{label} (+{gained:.3f} MB of the "
                             f"{rss_growth_mb:.1f} MB gained)")

    for name, (a, b) in byte_moves.items():
        weigh(f"cache_bytes.{name} {a:.0f} -> {b:.0f}", b - a)

    for group in ("cache_entries", "evictions"):
        for name, a, b in moved_in(group):
            label = f"{group}.{name} {a:.0f} -> {b:.0f}"
            # A counter the receipt also prices is already weighed above; do not
            # report the same container twice under two units.
            if group == "cache_entries" and name in byte_moves:
                continue
            unpriced.append(label)

    notes: list[str] = []
    live = last.get("live_cases")
    if isinstance(live, (int, float)) and live != 0:
        notes.append(f"and live_cases={live} between generations: a Driver outlived its "
                     "case, which leaks everything a case owns -- look here FIRST, it "
                     "outranks every cache below")
    if explains:
        notes.append("and the process's own [EVALUATOR][MEM] receipts show these growing "
                     "by enough to matter: " + "; ".join(explains))
    if too_small:
        notes.append("and these moved but CANNOT be the cause -- they are too small by "
                     "orders of magnitude, so do not spend a day on them: "
                     + "; ".join(too_small))
    if unpriced:
        notes.append("and these counters moved with no byte price in the receipt: "
                     + "; ".join(unpriced))
    if not (explains or too_small or unpriced):
        notes.append("and every container [EVALUATOR][MEM] can see was FLAT across the "
                     "run -- not one counter or byte total moved")
    # WP10.6.  THE ALLOCATOR, ASKED DIRECTLY.  The 238 soak ended on "look at
    # the allocator next" because the receipt could not look at it.  It can now:
    # `malloc_retained_mb` is what glibc has taken from the kernel and no case
    # is using, i.e. free lists a per-thread arena will hand to the next case
    # and will never hand back to the OS.  Retention that tracks the growth is a
    # HIGH-WATER MARK, which converges; a leak does not, and the two need
    # opposite repairs.  Saying which one it is here is the difference between a
    # finding and a hand-off.
    retained = _paired(first, last, "malloc_retained_mb")
    if retained is not None:
        head, tail = retained
        share = None
        if rss_growth_mb and rss_growth_mb > 0.0:
            share = (tail - head) / rss_growth_mb
        if share is not None and share >= ATTRIBUTION_SHARE:
            notes.append(
                f"and malloc_retained_mb went {head:.1f} -> {tail:.1f} MB, "
                f"{share:.0%} of the {rss_growth_mb:.1f} MB gained: this is ALLOCATOR "
                "RETENTION, not a leak -- glibc is holding free lists at the "
                "concurrent-case high-water mark. Compare the second-half slope; a "
                "high-water mark flattens and a leak does not")
        else:
            notes.append(f"and malloc_retained_mb went {head:.1f} -> {tail:.1f} MB, "
                         "which does NOT account for the growth")
    elif last.get("malloc_readable") is False:
        notes.append("and malloc_retained_mb is unreadable on this C library, so "
                     "allocator retention could not be ruled in or out from the receipt")
    # The device half, for the same reason: a receipt that reports its own VRAM
    # cannot be contaminated by a neighbour on the board.
    device = _paired(first, last, "vram_mb")
    if device is not None:
        head, tail = device
        rebuilds = last.get("arena_rebuilds")
        notes.append(f"and the process's OWN device footprint went {head:.1f} -> "
                     f"{tail:.1f} MB with arena_rebuilds="
                     f"{rebuilds if rebuilds is not None else 'absent'} "
                     "(0 means the arena was stood up once and never rebuilt)")
    if not explains:
        notes.append("so nothing [EVALUATOR][MEM] can see accounts for the growth: look "
                     "at the allocator (compare rss_peak_mb with rss_mb -- a large gap "
                     "is retention, not a leak), HDF5's own free lists, and the device "
                     "runtime's host-side allocations next")
    return notes


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    index: int
    cases: int = 0
    ok: int = 0
    failed: int = 0
    wall_s: float = 0.0
    cases_per_hour: float = 0.0
    vram_mb: float | None = None
    #: WP10.6.  "process", "board" or "none".  A board reading is NOT this
    #: process's footprint and is labelled so no later reader can mistake one
    #: for the other, which is how a neighbour's 47 GB became a soak finding.
    vram_scope: str = "none"
    vram_foreign: int = 0
    vram_board_mb: float | None = None
    rss_mb: float | None = None
    poisoned: int = 0
    promotions: int = 0
    screens: int = 0
    alive: bool = True
    refused: list[dict] = field(default_factory=list)
    fidelity_problems: list[str] = field(default_factory=list)
    #: WP10.4.  The generation's `[RASBERY][EVALUATOR][MEM]` receipt, or None
    #: from a binary that predates it.  The soak measures RSS from OUTSIDE; this
    #: is the process's own view of the same quantity plus the size of every
    #: container that could explain a change -- which is what turns "something
    #: grew 17 MB" into a named suspect.
    mem: dict | None = None


def build_generation(*, generation: int, width: int, deck: Path, workdir: Path,
                     light_fraction: float, screen_fraction: float,
                     poison: bool, promote: bool, bad_deck: Path
                     ) -> tuple[list[dict], dict[str, str]]:
    """One generation's case requests, and the fidelity each was DECLARED at.

    The declaration map is returned beside the requests because the per-case
    audit needs it: after WP10.3 "what was this wave declared as" has as many
    answers as the wave has cases, and reconstructing them from the receipts
    afterwards would be asking the run to grade its own homework.
    """
    cases: list[dict] = []
    declared: dict[str, str] = {}

    def declare(word: str, *names: str) -> None:
        """Declare one case under EVERY name a receipt could identify it by.

        WP10.5.  The map used to be keyed on the client `key` alone, which is the
        only name `[RASBERY][EVALUATOR][CASE]` carries -- and the Driver's own
        `[RASBERY][CASE]` line does not carry it.  exact_audit reads BOTH tags,
        so half the receipts in every run were unmatchable and reported as
        undeclared: 82 of them on 181 at 55c0dce.  src/Driver.h now prints
        `output`; this declares under it, and the two halves meet.

        Declaring under several names is safe because they all name ONE case:
        the evaluator refuses duplicate `--raso` paths within a wave, so an
        output path can never carry two declarations at once.
        """
        for name in names:
            if name:
                declared[name] = word
    n_light = int(round(width * light_fraction))
    n_screen = int(round(width * screen_fraction))
    for i in range(width):
        key = f"g{generation:04d}c{i:04d}"
        request: dict[str, object] = {
            "op": "case",
            "key": key,
            "deck": str(deck),
            # One output path per (generation, case).  Reusing a path ACROSS
            # generations would be legal for the evaluator (it scopes the
            # namespace rule to the wave) and would hide an output collision
            # from this driver, which is one of the things it is here to find.
            "output": str(workdir / "out" / f"{key}.h5"),
            "result_mode": "light" if i < n_light else "full",
            # The chain.  Generation 0 has no parent and starts cold, which is
            # also the control: if the chain is what leaks, generation 0 is the
            # only clean one and the trace says so.
            "save_warm_state": str(workdir / "warm" / f"{key}.warm"),
        }
        if generation > 0:
            request["warm_start_from"] = str(
                workdir / "warm" / f"g{generation - 1:04d}c{i:04d}.warm")
        if i < n_screen:
            request["fidelity"] = "L3coarse"
            request["statepoint_grid"] = "coarse"
            declare("L3coarse", key, str(request["output"]))
        else:
            request["fidelity"] = "strict"
            declare("strict", key, str(request["output"]))
        cases.append(request)

    if poison:
        key = f"g{generation:04d}poison"
        cases.append({"op": "case", "key": key, "deck": str(bad_deck),
                      "output": str(workdir / "out" / f"{key}.h5"),
                      "result_mode": "light"})
        # No declaration: the poisoned case is EXPECTED to fold no receipt, and
        # audit_case_fidelity skips a failed case that reported none.  Declaring
        # one would ask the audit to grade a case that never ran.
    if promote and n_screen > 0:
        parent = f"g{generation:04d}c0000"
        key = f"g{generation:04d}promote"
        cases.append({
            "op": "promote",
            "key": key,
            "deck": str(deck),
            "output": str(workdir / "out" / f"{key}.h5"),
            # The link.  `promoted_from` is the SCREENING case's key here rather
            # than its case_key, because the soak drives its own requests and
            # knows what it sent; a GA reads the case_key back off the screen's
            # receipt and sends that.  Both are strings the receipts carry.
            "promoted_from": parent,
        })
        declare("strict", key, str(workdir / "out" / f"{key}.h5"))
    return cases, declared


def write_bad_deck(path: Path) -> None:
    """A deck that cannot be parsed.

    NOT a deck that is merely wrong -- a deck the JSON parser itself refuses, so
    the failure lands in IO::ReadInput inside runOneCase's try and exercises the
    ONE path that matters: a case that throws must fail alone.  A semantically
    bad deck could be caught later, in a place with different unwinding, and
    would be testing something else.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schedule": [ this is not json', encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines = ["# RASBERY WP11 soak report", ""]
    lines.append(f"- verdict: **{'PASS' if report['pass'] else 'FAIL'}**")
    lines.append(f"- generations: {report['generations']} x {report['width']} "
                 f"= {report['cases_requested']} cases requested, "
                 f"{report['cases_reported']} reported")
    lines.append(f"- wall: {report['wall_s']:.1f} s")
    lines.append(f"- restarts: {report['restarts']} "
                 f"(injected poison: {report['poisoned']})")
    lines.append(f"- command: `{report['command']}`")
    lines.append("")
    lines.append("## Zero receipts")
    lines.append("")
    lines.append("| receipt | value | required |")
    lines.append("|---|---|---|")
    for name, value in report["zero_receipts"].items():
        required = "0" if name not in report["not_asserted"] else "not asserted"
        shown = "n/a" if value is None else value
        lines.append(f"| {name} | {shown} | {required} |")
    if report["not_asserted"]:
        lines.append("")
        lines.append("> Not asserted, and why: "
                     + "; ".join(f"`{k}` -- {v}"
                                 for k, v in report["not_asserted"].items()))
    lines.append("")
    lines.append("## Throughput and growth")
    lines.append("")
    # WP10.6.  The SCOPE column is not decoration.  A board reading and a
    # process reading in the same column with no label is how eight MPS clients
    # on another board became a soak finding about this evaluator.
    lines.append("| gen | cases | ok | failed | wall s | c/h | VRAM MB | scope | RSS MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for gen in report["per_generation"]:
        vram = "-" if gen["vram_mb"] is None else format(gen["vram_mb"], ".0f")
        rss = "-" if gen["rss_mb"] is None else format(gen["rss_mb"], ".0f")
        lines.append(
            f"| {gen['index']} | {gen['cases']} | {gen['ok']} | {gen['failed']} | "
            f"{gen['wall_s']:.2f} | {gen['cases_per_hour']:.1f} | {vram} | "
            f"{gen.get('vram_scope', '?')} | {rss} |")
    lines.append("")
    drift = report["throughput"]
    lines.append(f"- c/h median {drift['median']:.1f}, "
                 f"min {drift['min']:.1f}, max {drift['max']:.1f}, "
                 f"worst drift {drift['worst_drift']:.3%} "
                 f"(budget {drift['budget']:.1%})")
    for what in ("vram", "rss"):
        block = report["growth"][what]
        limit = block["limit_mb_per_generation"]
        warm = block.get("warmup_generations", 0)

        def shown(key: str) -> str:
            value = block.get(key)
            return "not measured" if value is None else f"{value:+.2f} MB/gen"

        # Gated number first, raw beside it.  Printing only one of them is how
        # the 55c0dce report ended up arguing with the table above it.
        lines.append(
            f"- {what} growth (gate, post-warm-up, {warm} gen dropped): "
            f"{shown('slope_mb_per_generation')}, limit {limit} MB/gen "
            f"[raw over every generation {shown('slope_raw_mb_per_generation')}; "
            f"second half of the whole run "
            f"{shown('slope_second_half_all_mb_per_generation')}]")
    if report["problems"]:
        lines.append("")
        lines.append("## Problems")
        lines.append("")
        for problem in report["problems"]:
            lines.append(f"- {problem}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, required=True,
                    help="the deck every case runs (one cohort, which is what the "
                         "plan's soak 1 asks for: 64 decks, ONE cohort, 10,000 cases)")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=None,
                    help="the RASBERY executable")
    ap.add_argument("--command", type=str, default=None,
                    help="the full child command instead of --binary (the contract "
                         "test passes the fake child here)")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--width", type=int, default=64,
                    help="cases per generation AND --batch-mode; the arena is one "
                         "allocation latched on the first wave")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--light-fraction", type=float, default=0.5)
    ap.add_argument("--screen-fraction", type=float, default=0.25,
                    help="fraction of each generation run at L3coarse on the coarse "
                         "burnup grid; 0 makes the soak single-fidelity, which does "
                         "not exercise WP10.3 at all")
    ap.add_argument("--no-poison", action="store_true",
                    help="do not inject a failing case per generation. Off by "
                         "default: a soak with no failure has tested the easy half")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--drift", type=float, default=0.03,
                    help="allowed c/h deviation from the median, per generation")
    ap.add_argument("--vram-leak-mb", type=float, default=8.0,
                    help="MB per generation of VRAM growth over the run's second "
                         "half that counts as a leak")
    ap.add_argument("--rss-leak-mb", type=float, default=8.0)
    ap.add_argument("--warmup-generations", type=int, default=1,
                    help="generations dropped before the leak slope is fitted. The "
                         "first generation stands the arenas, the device library and "
                         "the graph cache up -- on 181 that was a one-time +2988 MB "
                         "step -- and a line fitted through it reports a stand-up cost "
                         "as a leak. The raw slope is reported either way.")
    ap.add_argument("--max-restarts", type=int, default=3,
                    help="how many times the dispatcher may replace a dead child "
                         "before giving up; this is the RECOVERY budget")
    ap.add_argument("--expect-restarts", type=int, default=0,
                    help="how many restarts this run INTENDS to cause. The default "
                         "poison fails one case and leaves the process alive, so 0 "
                         "is the honest expectation and any restart is a finding")
    ap.add_argument("--report", type=Path, default=None,
                    help="JSON report path (default <workdir>/soak_report.json); the "
                         "markdown goes beside it with a .md suffix")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="an environment variable for the child, repeatable")
    ap.add_argument("--result", type=str, default="full",
                    help="the child's default --result; per-case modes override it")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    workdir: Path = args.workdir
    (workdir / "out").mkdir(parents=True, exist_ok=True)
    (workdir / "warm").mkdir(parents=True, exist_ok=True)
    (workdir / "log").mkdir(parents=True, exist_ok=True)
    bad_deck = workdir / "poison" / "unparseable.json"
    write_bad_deck(bad_deck)

    if args.command:
        # posix=False on Windows, because posix mode treats a backslash as an
        # escape and turns C:\Users\MK\python.exe into C:UsersMKpython.exe --
        # a path that does not exist, reported as "the child never became
        # ready", which is the least informative possible spelling of "your
        # command was mangled by the argument parser".  The soak itself runs on
        # Linux; its contract test runs here.
        command = shlex.split(args.command, posix=(os.name != "nt"))
        if os.name == "nt":
            command = [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t
                       for t in command]
    elif args.binary:
        command = [str(args.binary)]
    else:
        print("soak_run: one of --binary or --command is required", file=sys.stderr)
        return 2
    command += ["--evaluator-jsonl", "-", "--batch-mode", str(args.width),
                "--result", args.result]

    env = dict(os.environ)
    for item in args.set:
        key, _, value = item.partition("=")
        env[key] = value
    env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    gpu_full = env.get("RASBERY_GPU_FULL") not in (None, "", "0")
    # The board the CHILD will be on -- see sampled_gpu().  Resolved AFTER the
    # setdefault above, because the setdefault is what decides it.
    vram_gpu = sampled_gpu(env, args.gpu)

    session = EvaluatorSession(command=command, env=env, cwd=str(ROOT),
                               log_path=workdir / "log" / "soak.log",
                               max_restarts=args.max_restarts)
    started = time.time()
    if not session.start():
        print("soak_run: the child never became ready", file=sys.stderr)
        return 2

    generations: list[GenerationResult] = []
    declared_all: dict[str, str] = {}
    cases_requested = 0
    poisoned = 0
    transcript = session.preamble

    for g in range(args.generations):
        cases, declared = build_generation(
            generation=g, width=args.width, deck=args.deck, workdir=workdir,
            light_fraction=args.light_fraction, screen_fraction=args.screen_fraction,
            poison=not args.no_poison, promote=not args.no_promote, bad_deck=bad_deck)
        declared_all.update(declared)
        cases_requested += len(cases)
        if not args.no_poison:
            poisoned += 1
        result = GenerationResult(index=g)
        result.poisoned = 0 if args.no_poison else 1
        result.promotions = 0 if args.no_promote else 1
        result.screens = sum(1 for c in cases if c.get("fidelity") == "L3coarse")

        if not session.alive and not session.restart():
            result.alive = False
            generations.append(result)
            break
        wave_start = time.time()
        outcome = session.wave(wave_id=g + 1, cases=cases)
        transcript += outcome.text
        result.wall_s = time.time() - wave_start
        result.alive = outcome.alive
        result.refused = list(outcome.refused)
        result.cases = len(outcome.cases)
        result.ok = sum(1 for c in outcome.cases if c.get("status") == "ok")
        result.failed = result.cases - result.ok
        if outcome.receipt:
            result.wall_s = float(outcome.receipt.get("wall_s", result.wall_s))
            result.cases_per_hour = float(outcome.receipt.get("cases_per_hour", 0.0))
        elif result.wall_s > 0:
            result.cases_per_hour = 3600.0 * result.cases / result.wall_s
        # The per-case fidelity audit, ON THIS GENERATION's declarations.  Done
        # here rather than at the end because a mixed wave's declarations are
        # per generation and folding them all into one map would let generation
        # 7's key answer for generation 3's case.
        result.fidelity_problems = exact_audit.audit_case_fidelity(
            outcome.text, declared, require_any=result.cases > 0)
        # VRAM/RSS BETWEEN generations, not during: a sample taken mid-wave
        # measures where the wave happened to be, and the question is what is
        # left behind when it is over.
        vram = sample_vram(vram_gpu, session.pid)
        result.vram_mb = vram.used_mb
        result.vram_scope = vram.scope
        result.vram_foreign = len(vram.foreign_pids)
        result.vram_board_mb = vram.board_mb
        result.rss_mb = sample_rss_mb(session.pid)
        mem_receipts = receipts_of(outcome.text, "[RASBERY][EVALUATOR][MEM]")
        result.mem = mem_receipts[-1] if mem_receipts else None
        # WP10.6.  THE PROCESS'S OWN NUMBER WINS for VRAM, which is the opposite
        # of the RSS rule three lines down, and deliberately so.  RSS is read
        # from outside because an outside reading cannot be wrong in the same
        # direction as the thing it measures.  VRAM's outside reading is the one
        # that WAS wrong: `nvidia-smi` answers for a board, and a board can have
        # eight other tenants on it.  `[EVALUATOR][MEM] vram_mb` is what this
        # process holds, by its own accounting, and nothing else can move it.
        if isinstance(result.mem, dict):
            reported = result.mem.get("vram_mb")
            if isinstance(reported, (int, float)) and reported >= 0:
                result.vram_mb = float(reported)
                result.vram_scope = "receipt"
        # FALL BACK TO THE PROCESS'S OWN RSS, and only then.  /proc/<pid>/status
        # is the primary because it is an OUTSIDE measurement and cannot be
        # wrong in the same direction as the thing it is measuring; where there
        # is no /proc for the child, a self-report is strictly better than the
        # `None` that silently disables the whole leak gate.
        if result.rss_mb is None and isinstance(result.mem, dict):
            reported = result.mem.get("rss_mb")
            if isinstance(reported, (int, float)) and reported > 0:
                result.rss_mb = float(reported)
        generations.append(result)
        if not outcome.alive and not session.restart():
            break

    transcript += session.close()
    wall = time.time() - started

    # -- the assertions ----------------------------------------------------
    problems: list[str] = []
    zero_values: dict[str, int | None] = {}
    not_asserted: dict[str, str] = {}
    for name, tag, field_name in ZERO_RECEIPTS:
        found = [r.get(field_name) for r in receipts_of(transcript, tag)
                 if field_name in r]
        if not found:
            zero_values[name] = None
            not_asserted[name] = (f"no {tag} receipt carried {field_name!r} in this "
                                  "run's output")
            problems.append(
                f"{name}: no {tag} receipt carried {field_name!r}. A counter that was "
                "never printed is not a counter that was zero, and this soak's whole "
                "claim is that it read them.")
            continue
        worst = max(int(v) for v in found if isinstance(v, (int, float)))
        zero_values[name] = worst
        if worst != 0:
            problems.append(f"{name} = {worst}, must be 0 at exit")

    for field_name in GPU_FULL_FALLBACKS:
        found = [r.get(field_name) for r in receipts_of(transcript, "[RASBERY][GPU_FULL]")
                 if field_name in r]
        if not found:
            continue
        worst = max(int(v) for v in found if isinstance(v, (int, float)))
        zero_values[field_name] = worst
        if not gpu_full:
            not_asserted[field_name] = ("RASBERY_GPU_FULL is not set, so a host "
                                        "fallback is legal in this arm")
        elif worst != 0:
            problems.append(f"{field_name} = {worst} under RASBERY_GPU_FULL=1")

    # WP10.7.  THE RECEIPT'S OWN VERDICT, read at last.
    #
    # `contract_pass` has been in the [RASBERY][GPU_FULL] receipt since WP1(b)
    # and this soak has never read it.  On 238 arm A that cost nothing only
    # because the counters happened to catch the same failure -- but the run
    # ALSO printed `first_violation` naming a site, and the soak's report,
    # which is what a campaign quotes, carried neither.  Reading the verdict
    # instead of only re-deriving it from the counters is also how a future
    # count-only seam (the WP1 defect, which gpufull::enforceExitCode exists to
    # backstop) shows up here rather than passing.
    gpu_full_receipts = receipts_of(transcript, "[RASBERY][GPU_FULL]")
    contract: dict = {}
    if gpu_full_receipts:
        last = gpu_full_receipts[-1]
        contract = {
            "contract_pass": last.get("contract_pass"),
            "first_violation": last.get("first_violation"),
            # WP10.7 fields; absent from a binary that predates them, which is
            # why they are read with .get and reported as None rather than
            # asserted into existence.
            "first_violation_seq": last.get("first_violation_seq"),
            "violations": last.get("violations"),
            "allowed_refusals": last.get("allowed_refusals"),
        }
        if gpu_full and last.get("contract_pass") is False:
            problems.append(
                "the [RASBERY][GPU_FULL] receipt reports contract_pass:false under "
                "RASBERY_GPU_FULL=1, first_violation="
                f"{last.get('first_violation')!r}. A fail-closed contract that the "
                "run itself declares broken is not a soak this campaign may quote.")

    # RESTARTS ARE BOUNDED BY WHAT WAS INJECTED, and this soak's injection is
    # zero.  The poison it plants is a deck that does not PARSE: the real binary
    # reaches that through IO::ReadInput throwing inside runOneCase's try, which
    # fails one case and leaves the process answering.  So a restart is not "the
    # poison working", it is the poison taking the process with it -- the exact
    # failure-isolation defect the poison is planted to find.  --expect-restarts
    # raises the bound for a run that deliberately kills the child (a CUDA abort
    # rehearsal), so the number is a stated intention rather than a slack budget.
    if session.restarts > args.expect_restarts:
        problems.append(
            f"the evaluator restarted {session.restarts} times, expecting at most "
            f"{args.expect_restarts}. The injected poison is a deck that fails its OWN "
            "case (EvaluatorServer::runOneCase catches the throw); a restart means it "
            "took the process down with it, and the sixty-three candidates that were "
            "in flight went with it.")

    # Output collisions: this driver's own bookkeeping, because the evaluator
    # scopes its namespace rule to a wave and would not see a cross-generation
    # reuse at all.
    # WP10.5.  COMPARE LIKE WITH LIKE.  This used to fold `key` and `case_key`
    # into one identity with an `or`, which was harmless only while exactly one
    # of the two case tags carried an `output` field.  Now both do, and the two
    # tags name the SAME case by different identities on purpose -- the client's
    # opaque label and the solver's canonical duplicate key -- so folding them
    # reported every case in the run as colliding with itself.  A collision is
    # two DIFFERENT cases on one path, and the only way to see that is to
    # compare a key against a key and a case_key against a case_key.
    #
    # `case_key` alone can never prove a collision, either: it is a duplicate
    # class, and two cases that legitimately share one (same deck, same
    # fidelity, same warm token) share it by design.  So its map is kept for the
    # receipts that carry nothing else, and a disagreement there is reported in
    # the weaker language it deserves.
    by_key: dict[str, str] = {}
    by_case_key: dict[str, str] = {}
    for receipt in exact_audit.parse_case_receipts(transcript):
        out_path = receipt.get("output")
        if not (isinstance(out_path, str) and out_path):
            continue
        label = receipt.get("key")
        if isinstance(label, str) and label:
            if by_key.get(out_path, label) != label:
                problems.append(
                    f"output collision: {out_path!r} was written by both "
                    f"{by_key[out_path]!r} and {label!r}")
            by_key[out_path] = label
            continue
        canonical = receipt.get("case_key")
        if isinstance(canonical, str) and canonical:
            if by_case_key.get(out_path, canonical) != canonical:
                problems.append(
                    f"output collision: {out_path!r} was written by two different "
                    f"case_keys, {by_case_key[out_path]!r} and {canonical!r} -- two "
                    "cases that are not duplicates of each other wrote one --raso path")
            by_case_key[out_path] = canonical

    for gen in generations:
        for problem in gen.fidelity_problems:
            problems.append(f"generation {gen.index}: {problem}")
        for refusal in gen.refused:
            problems.append(f"generation {gen.index}: refused -- "
                            f"{refusal.get('what', refusal)}")

    # Throughput drift.
    rates = [g.cases_per_hour for g in generations if g.cases_per_hour > 0]
    median = statistics.median(rates) if rates else 0.0
    worst_drift = 0.0
    if median > 0:
        for gen in generations:
            if gen.cases_per_hour <= 0:
                continue
            drift = abs(gen.cases_per_hour - median) / median
            worst_drift = max(worst_drift, drift)
            if drift > args.drift:
                problems.append(
                    f"generation {gen.index}: {gen.cases_per_hour:.1f} c/h is "
                    f"{drift:.2%} off the median {median:.1f}, budget {args.drift:.1%}")

    # WP10.6.  A VRAM SLOPE IS ONLY THIS PROCESS'S SLOPE WHEN THE SAMPLE WAS
    # THIS PROCESS'S.  Where the reading is board-scoped and other compute apps
    # were on the board, the trace is the board's and the slope is not evidence
    # about this process at all -- so it is reported, named, and NOT convicted.
    # This is the 238 finding, in the form that would have prevented it: the
    # sawtooth was eight MPS clients on the other board, and no amount of
    # arithmetic on that series was ever going to be about the evaluator.
    vram_scopes = {g.vram_scope for g in generations}
    contaminated = [g for g in generations
                    if g.vram_scope == "board" and g.vram_foreign > 0]
    if contaminated:
        problems.append(
            f"{len(contaminated)} of {len(generations)} VRAM samples were BOARD-scoped "
            f"with up to {max(g.vram_foreign for g in contaminated)} other compute "
            f"process(es) on the board: that trace is the board's, not this "
            f"process's, and no slope taken from it is evidence about this "
            f"evaluator. Run the soak on an idle board, or use a build whose "
            f"[EVALUATOR][MEM] receipt carries vram_mb.")

    growth = {}
    for what, series, limit in (
            ("vram", [g.vram_mb for g in generations], args.vram_leak_mb),
            ("rss", [g.rss_mb for g in generations], args.rss_leak_mb)):
        if what == "vram" and contaminated:
            growth[what] = dict(growth_slopes(series, args.warmup_generations),
                                limit_mb_per_generation=limit, samples=series,
                                scopes=sorted(vram_scopes),
                                gated=False,
                                gate_skipped_because="board-scoped with other tenants")
            continue
        slopes = growth_slopes(series, args.warmup_generations)
        slope = slopes["slope_mb_per_generation"]
        growth[what] = dict(slopes,
                            limit_mb_per_generation=limit,
                            samples=series,
                            gated=True)
        if what == "vram":
            growth[what]["scopes"] = sorted(vram_scopes)
        if slope is not None and slope > limit:
            raw = slopes["slope_raw_mb_per_generation"]
            post = slopes["slope_post_warmup_mb_per_generation"]
            problems.append(
                f"{what} grew {slope:.2f} MB/generation over the post-warm-up run "
                f"({args.warmup_generations} generation(s) dropped, limit {limit}); "
                f"after the warm plateau nothing should still be climbing. Raw slope "
                f"over every generation including the stand-up step "
                f"{'n/a' if raw is None else format(raw, '.2f')} MB/gen"
                + ("" if post is None or slope is None or abs(post - slope) < 1e-9
                   else f", post-warm-up {post:.2f} MB/gen"))
            if what == "rss":
                present = [v for v in series if v is not None]
                observed = (present[-1] - present[0]) if len(present) >= 2 else None
                problems.extend(attribute_rss_growth(generations, observed))

    cases_reported = sum(g.cases for g in generations)
    if cases_reported < cases_requested:
        problems.append(
            f"{cases_requested - cases_reported} of {cases_requested} cases were never "
            "reported. A generation that silently lost candidates is not a generation.")

    report = {
        "schema": "rasbery-soak/v1",
        "pass": not problems,
        "command": " ".join(command),
        "generations": args.generations,
        "width": args.width,
        "cases_requested": cases_requested,
        "cases_reported": cases_reported,
        "poisoned": poisoned,
        "restarts": session.restarts,
        "expect_restarts": args.expect_restarts,
        "starts": session.starts,
        "returncode": session.returncode,
        "wall_s": wall,
        "gpu_full": gpu_full,
        "gpu_sampled": vram_gpu,
        "gpu_requested": args.gpu,
        "vram_scopes": sorted(vram_scopes),
        "zero_receipts": zero_values,
        "not_asserted": not_asserted,
        # WP10.7.  The GPU_FULL receipt's own verdict and the site it names, so
        # the report a campaign quotes answers "which seam, first" without a
        # 16k-line log.
        "gpu_full_contract": contract,
        "throughput": {
            "median": median,
            "min": min(rates) if rates else 0.0,
            "max": max(rates) if rates else 0.0,
            "worst_drift": worst_drift,
            "budget": args.drift,
        },
        "growth": growth,
        "per_generation": [
            {"index": g.index, "cases": g.cases, "ok": g.ok, "failed": g.failed,
             "wall_s": g.wall_s, "cases_per_hour": g.cases_per_hour,
             "vram_mb": g.vram_mb, "vram_scope": g.vram_scope,
             "vram_foreign_procs": g.vram_foreign, "vram_board_mb": g.vram_board_mb,
             "rss_mb": g.rss_mb, "screens": g.screens,
             "promotions": g.promotions, "poisoned": g.poisoned, "alive": g.alive,
             "mem": g.mem}
            for g in generations],
        "problems": problems,
    }

    report_path = args.report or (workdir / "soak_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    report_path.with_suffix(".md").write_text(render_markdown(report),
                                              encoding="utf-8", newline="\n")
    print(f"soak: {'PASS' if report['pass'] else 'FAIL'} "
          f"({cases_reported}/{cases_requested} cases, {len(problems)} problems)")
    for problem in problems:
        print("  - " + problem)
    print(f"wrote {report_path} and {report_path.with_suffix('.md')}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
