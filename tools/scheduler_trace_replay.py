#!/usr/bin/env python3
"""Offline replay of five case-phase scheduler policies over an M64 trace.

W0 decision spike 5/5, implementing Rev.7 8.8 ("Scheduler trace replay before
production").  Rev.7 makes this a precondition, not an option: the conditional
case-phase scheduler is adopted only if a replay over the EXISTING M64 trace
predicts a >= 20 % reduction in GPU idle, a smaller cohort tail, better tail
efficiency, and an acceptable scheduler operation count.  This is that replay.

THIS IS A RANKING TOOL, NOT A PREDICTOR.  It exists to order five policies, not
to forecast a wall clock.  Every number it prints is conditional on the
assumptions below; if a policy wins here it has earned a prototype, nothing more.

=============================================================================
ASSUMPTIONS -- all of them, stated once, in one place
=============================================================================

A1. ONE SERVER.  Rev.7 5.8: "the first production version does not run heavy
    phases concurrently.  The scheduler picks one heavy phase; only control and
    transfer overlap."  The device is therefore a single server servicing
    exactly one phase kind per step.  Light phases get no free overlap lane by
    default (--light-overlap turns that on), because granting it by default
    would flatter every policy equally and hide the ranking.

A2. FIXED INTRA-CASE ORDER.  Rev.7 global constraint 33: a scheduler may reorder
    cases, never the phases within a case.  Every job walks PHASE_ORDER in a
    fixed cycle, once per outer, then a statepoint-boundary item.  A policy
    chooses WHICH jobs run, never which phase a given job does next.

A3. PER-OUTER SERVICE TIME FROM MEASURED SHARES.  A statepoint reporting
    `outers=O` and `phase_wall={...}` becomes O passes through the seven phases,
    each costing phase_wall[p]/O.  This assumes every outer of a statepoint
    costs the same.  It does not -- the first outer is dearer -- but the trace
    records no per-outer walls, and the error is common to all five policies.

A4. THE STATEPOINT BOUNDARY IS THE RESIDUAL.  Each statepoint gets one extra
    work item costing `wall - sum(phase_wall)`, floored at zero: the Xe,
    boron/rod search, T-H control, depletion, burnup and convergence work that
    Rev.7 6.15-6.19 puts on the device but that phase_wall does not itemise.
      This is a clean residual, not a guess: Driver.h stops the statepoint clock
    at line 2303 and starts the I/O clock at 2307, so `wall` EXCLUDES `io_wall`.
    Subtracting phase_wall from it therefore leaves device-side compute and no
    host writer time -- which is what we want, since Rev.7 Task 19 keeps I/O on
    a CPU writer thread and off the critical path.
      The item matters out of proportion to its size, because statepoints have
    DIFFERENT outer counts across jobs, so it is where cases desynchronise --
    and desynchronisation is the entire thing the queue policies exist to
    exploit.  A model without it has all five policies collapse into two.

A5. BATCH WIDTH SCALING IS A ONE-PARAMETER GUESS.  A phase step over w slots
    costs

        service(p, w) = max_{j in step} d_j * (s + (1 - s) * w / W_ref)

    where d_j is job j's measured per-outer phase time at the reference width
    W_ref (default 64) and s is --dispatch-share: the fraction of a width-64
    batched step that does NOT shrink when the batch narrows.  s = 1 is pure
    dispatch bound (narrowing saves nothing); s = 0 is pure throughput bound.
      The default s = 0.6 is a JUDGEMENT from the fact that single-deck CMFD
    grids are 34 and 67 blocks on a 188-SM device (Phase 5 doc 1.1) and that
    drive time is dominated by dispatch and memory latency, not arithmetic.  It
    is NOT measured.  W0 probe 1 (probe_dispatch_floor.cu) measures the quantity
    that should replace it.  Re-run with --dispatch-share 0.3 and 0.9 and check
    the ranking is unchanged before quoting anything; the receipt records the
    value used.

A6. `max` OVER A STEP.  Slots in one batched launch finish together, so a step
    costs what its slowest participant costs.  This is why narrower, more
    homogeneous cohorts win here -- and it is the modelling choice most
    responsible for that win.  Defensible (grid.y = slots, one kernel, one
    completion) but a choice.

A7. IDLE MEANS WASTED SLOT CAPACITY, NOT IDLE SMs.  Rev.7 8.6 defines
    tail_efficiency = active_slot_phase_work / physical_bucket_phase_work.  The
    idle figure the 20 % gate is taken against is
    1 - (time-weighted mean active width / W).  It says nothing about SM
    occupancy; probe 4 measures that instead.

A8. NO ADMISSION, TRANSFER, OR REFILL LATENCY.  Slot refill is instantaneous and
    free.  Real refill costs a host round trip on the epoch backend (Rev.7 8.7).
    This biases the refill policies (3, 4, 5) OPTIMISTICALLY.  Scheduler
    operation count is reported so the bias is visible: a policy that wins on
    idle while doing 50x the scheduler operations has not won.

A9. NO CONVERGENCE COUPLING.  Outer counts replay from the trace as fixed.  A
    different schedule cannot change how many outers a case needs -- true for
    this code, since same-case numerical order is preserved, and false for
    anything that changed the physics.

=============================================================================

POLICIES (Rev.7 8.8 list, in order)
  1 current rendezvous       fleet-level cohort barrier; within a statepoint the
                             whole batch runs max(outers) outers and a converged
                             slot is padding; no refill until the cohort drains
  2 fixed-cohort conditional cohort barrier kept, but each slot runs its OWN
                             outer count (the conditional WHILE), so padding
                             inside a statepoint disappears and phase queues form
  3 track-slot refill only   lockstep and padding kept; a freed slot refills at
                             once, joining at the next statepoint boundary
  4 largest-ready-first      no lockstep, immediate refill; each step services
                             the largest ready queue in the highest non-empty
                             cost class (Rev.7 8.5)
  5 phase queues + max-age   policy 4 plus: any non-empty queue that has aged
                             MAX_AGE steps without service is serviced first
                             (Rev.7 8.5 verbatim)

ADOPTION GATE (Rev.7 8.8): policy 4 or 5 must predict >= 20 % idle reduction
against policy 1.

SCHEDULER OPS are counted as real scheduler decisions, so the metric means the
same thing across policies that do different amounts of thinking: lockstep
policies are charged one op per cohort round plus one per admission; queue
policies are charged one op per launch (they re-rank the queues every time) plus
one per admission.

INPUT
  --sptelem FILE   run log or JSONL carrying [RASBERY][SPTELEM] receipts.  The
                   real input: job_id, statepoint, outers, wall and phase_wall
                   are all already there (Driver.h:2362).
  --walls FILE     fallback: `TOTAL DRIVER TIME=  NNN.NNN s` lines, or bare
                   floats, one per job.  Phase timelines are synthesised from
                   PHASE_SHARE_DEFAULT, the measured single-deck split in
                   PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md 1.2.  Strictly worse
                   input: every job gets the same phase mix and the same
                   statepoint structure, so the desynchronisation of A4 is
                   synthetic.  Use --statepoints to give it some.
  --selftest       synthetic fleet; no input needed.

Usage:
    python3 tools/scheduler_trace_replay.py --sptelem m64_run.log --width 64
    python3 tools/scheduler_trace_replay.py --walls walls.txt --statepoints 12
    python3 tools/scheduler_trace_replay.py --selftest
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# The adoption gate, in percent.  Rev.7 8.8.
IDLE_REDUCTION_GATE_PCT = 20.0

# Driver.h:2377-2379 emits phase_wall in exactly this order, and it is the order
# an outer executes them in (Phase 5 doc 1.2).
PHASE_ORDER = ["updpsi", "setls", "drive", "updjnet", "nodal", "cusping", "upddhat"]

# The statepoint-boundary work item of assumption A4.
BOUNDARY_PHASE = "statepoint"

# Rev.7 8.5 cost classes.  Higher rank is serviced first when queues tie.
COST_CLASS = {
    "drive": "heavy",
    "nodal": "heavy",
    BOUNDARY_PHASE: "heavy",   # depletion CRAM is Heavy in Rev.7 8.5
    "setls": "medium",
    "cusping": "medium",
    "upddhat": "medium",
    "updpsi": "light",
    "updjnet": "light",
}
CLASS_RANK = {"heavy": 2, "medium": 1, "light": 0}

# Measured single-deck phase split, PLAN_PHASE5_PERSISTENT_RESIDENCY_KO.md 1.2:
# setls 12.1 s, drive 55.5 s, updjnet 2.5 s, nodal 6.1 s, upddhat 5.8 s.  updpsi
# and cusping are not itemised there and are given small residuals.  Used only
# by the --walls fallback and by --selftest.
PHASE_SHARE_DEFAULT = {
    "updpsi": 1.0,
    "setls": 12.1,
    "drive": 55.5,
    "updjnet": 2.5,
    "nodal": 6.1,
    "cusping": 1.0,
    "upddhat": 5.8,
}
# Share of a statepoint's wall attributed to the boundary item on the synthetic
# and --walls paths.  On the --sptelem path this is measured, not assumed.
BOUNDARY_SHARE_DEFAULT = 0.08

SPTELEM_RE = re.compile(r"\[RASBERY\]\[SPTELEM\]\s*(\{.*\})")
TOTAL_DRIVER_RE = re.compile(r"TOTAL DRIVER TIME\s*=\s*([0-9.]+)\s*s")

DEFAULT_MAX_AGE = 8            # Rev.7 8.7 tuning candidates: 4, 8, 16
DEFAULT_DISPATCH_SHARE = 0.6   # assumption A5
DEFAULT_MAX_ITEMS = 200_000


# --------------------------------------------------------------------------
# Fleet model
# --------------------------------------------------------------------------

@dataclass
class Statepoint:
    """One statepoint: `blocks` passes of PHASE_ORDER, then one boundary item."""
    blocks: int
    per_block: dict[str, float]
    boundary_s: float
    outers: int = 0

    def items(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for _ in range(self.blocks):
            for p in PHASE_ORDER:
                out.append((p, self.per_block.get(p, 0.0)))
        out.append((BOUNDARY_PHASE, self.boundary_s))
        return out


@dataclass
class Job:
    job_id: str
    statepoints: list[Statepoint] = field(default_factory=list)
    wall: float = 0.0

    @property
    def outers(self) -> int:
        return sum(sp.outers for sp in self.statepoints)

    def items(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for sp in self.statepoints:
            out.extend(sp.items())
        return out


def make_statepoint(outers: int, phase_wall: dict[str, float], boundary_s: float,
                    coarsen: int) -> Statepoint | None:
    if outers <= 0:
        return None
    blocks = max(1, math.ceil(outers / coarsen))
    per_block = {p: float(phase_wall.get(p, 0.0)) / blocks for p in PHASE_ORDER}
    return Statepoint(blocks=blocks, per_block=per_block,
                      boundary_s=max(0.0, boundary_s), outers=outers)


def load_sptelem(path: Path, coarsen: int = 1) -> list[Job]:
    """Jobs from SPTELEM statepoint receipts.

    Accepts a raw run log (receipts are picked out by their tag) or plain JSONL
    of the same objects.  Statepoints are grouped by job_id and kept in the order
    they appear, which is the order they ran.
    """
    jobs: dict[str, Job] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SPTELEM_RE.search(line)
        payload = m.group(1) if m else (line.strip() if line.strip().startswith("{") else None)
        if not payload:
            continue
        try:
            rec = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "phase_wall" not in rec or "outers" not in rec:
            continue
        jid = str(rec.get("job_id", f"slot{rec.get('slot', 0)}"))
        job = jobs.setdefault(jid, Job(job_id=jid))
        phase_wall = rec.get("phase_wall") or {}
        wall = float(rec.get("wall") or 0.0)
        # Assumption A4: the boundary item is the measured residual.  `wall` is
        # Driver.h's step_seconds, which stops before the I/O clock starts
        # (Driver.h:2303 vs 2307), so io_wall is already excluded and must NOT
        # be subtracted again.
        boundary = wall - sum(float(v) for v in phase_wall.values())
        sp = make_statepoint(int(rec.get("outers") or 0), phase_wall, boundary, coarsen)
        if sp is not None:
            job.statepoints.append(sp)
            job.wall += wall
    return [j for j in jobs.values() if j.statepoints]


def load_walls(path: Path, statepoints: int, outers_per_statepoint: int,
               coarsen: int = 1) -> list[Job]:
    """Jobs synthesised from a TOTAL DRIVER TIME distribution."""
    walls: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TOTAL_DRIVER_RE.search(line)
        if m:
            walls.append(float(m.group(1)))
            continue
        s = line.strip()
        if s:
            try:
                walls.append(float(s))
            except ValueError:
                continue

    share_total = sum(PHASE_SHARE_DEFAULT.values())
    jobs: list[Job] = []
    for i, w in enumerate(walls):
        job = Job(job_id=f"synth{i:04d}", wall=w)
        sp_wall = w / max(1, statepoints)
        phase_budget = sp_wall * (1.0 - BOUNDARY_SHARE_DEFAULT)
        phase_wall = {p: phase_budget * s / share_total
                      for p, s in PHASE_SHARE_DEFAULT.items()}
        for _ in range(statepoints):
            sp = make_statepoint(outers_per_statepoint, phase_wall,
                                 sp_wall * BOUNDARY_SHARE_DEFAULT, coarsen)
            if sp is not None:
                job.statepoints.append(sp)
        jobs.append(job)
    return jobs


def synthetic_fleet(n_jobs: int, statepoints: int, outers_mean: int, seed: int,
                    coarsen: int = 1) -> list[Job]:
    """A deliberately heterogeneous fake fleet for --selftest.

    Heterogeneity is the point.  Outer counts vary BOTH between jobs and between
    statepoints of one job, which is what makes cases drift apart (A4); with a
    homogeneous fleet every policy has the same tail and the self-test would pass
    vacuously.
    """
    rng = random.Random(seed)
    share_total = sum(PHASE_SHARE_DEFAULT.values())
    jobs: list[Job] = []
    for i in range(n_jobs):
        # Long-tailed on purpose: a handful of cases needing far more outers is
        # the pathology the whole scheduler programme exists to attack.
        job_scale = rng.lognormvariate(0.0, 0.45)
        job = Job(job_id=f"fake{i:04d}")
        n_sp = max(1, int(round(statepoints * rng.uniform(0.6, 1.4))))
        for _ in range(n_sp):
            outers = max(1, int(rng.lognormvariate(math.log(outers_mean), 0.5) * job_scale))
            unit = rng.uniform(0.08, 0.30)
            mix = {p: s * rng.uniform(0.7, 1.4) for p, s in PHASE_SHARE_DEFAULT.items()}
            mix_total = sum(mix.values())
            phase_budget = outers * unit * (1.0 - BOUNDARY_SHARE_DEFAULT)
            phase_wall = {p: phase_budget * v / mix_total * share_total / share_total
                          for p, v in mix.items()}
            scale = phase_budget / max(sum(phase_wall.values()), 1e-12)
            phase_wall = {p: v * scale for p, v in phase_wall.items()}
            boundary = outers * unit * BOUNDARY_SHARE_DEFAULT
            sp = make_statepoint(outers, phase_wall, boundary, coarsen)
            if sp is not None:
                job.statepoints.append(sp)
                job.wall += phase_budget + boundary
        jobs.append(job)
    return jobs


def auto_coarsen(total_outers: int, max_items: int) -> int:
    raw_items = total_outers * len(PHASE_ORDER)
    if raw_items <= max_items:
        return 1
    return math.ceil(raw_items / max_items)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    key: str
    name: str
    kind: str      # "lockstep" | "queued"
    refill: str    # "cohort" | "immediate"
    select: str    # "none" | "largest_ready" | "largest_ready_maxage"
    max_age: int | None = None


POLICIES: list[Policy] = [
    Policy("p1_rendezvous", "current rendezvous (batch barrier per sweep)",
           kind="lockstep", refill="cohort", select="none"),
    Policy("p2_fixed_cohort_conditional", "fixed-cohort conditional outer",
           kind="queued", refill="cohort", select="largest_ready"),
    Policy("p3_track_slot_refill", "track-slot refill only",
           kind="lockstep", refill="immediate", select="none"),
    Policy("p4_largest_ready_first", "largest-ready-first phase queues",
           kind="queued", refill="immediate", select="largest_ready"),
    Policy("p5_phase_queue_max_age", "phase queues + max-age",
           kind="queued", refill="immediate", select="largest_ready_maxage",
           max_age=DEFAULT_MAX_AGE),
]


# --------------------------------------------------------------------------
# Shared metering
# --------------------------------------------------------------------------

class Meter:
    """Accumulates the Rev.7 8.6 tail metrics over a run of launches."""

    def __init__(self, width: int, dispatch_share: float, ref_width: int,
                 light_overlap: bool) -> None:
        self.width = width
        self.dispatch_share = dispatch_share
        self.ref_width = max(1, ref_width)
        self.light_overlap = light_overlap
        self.time = 0.0
        self.service_total = 0.0
        self.weighted_width = 0.0
        self.active_width_sum = 0
        self.launches = 0
        self.ops = 0
        self.max_age_triggers = 0
        self.done_times: list[float] = []

    def launch(self, phase: str, peak: float, w: int) -> None:
        if w <= 0:
            return
        factor = self.dispatch_share + (1.0 - self.dispatch_share) * (w / float(self.ref_width))
        service = peak * factor
        if self.light_overlap and COST_CLASS.get(phase) == "light":
            service = 0.0
        self.time += service
        self.service_total += service
        self.weighted_width += service * w
        self.active_width_sum += w
        self.launches += 1

    def retire(self) -> None:
        self.done_times.append(self.time)

    def result(self, policy: Policy, jobs: int) -> dict[str, Any]:
        makespan = self.time
        busy_capacity = self.weighted_width / float(self.width) if self.width else 0.0
        busy = (busy_capacity / makespan) if makespan > 0 else 0.0
        launches = max(1, self.launches)
        return {
            "policy": policy.key,
            "name": policy.name,
            "makespan_s": round(makespan, 4),
            "gpu_busy_fraction": round(busy, 5),
            "idle_fraction": round(1.0 - busy, 5),
            "tail_s": round(max(self.done_times) - min(self.done_times), 4) if self.done_times else 0.0,
            "first_done_s": round(min(self.done_times), 4) if self.done_times else 0.0,
            "last_done_s": round(max(self.done_times), 4) if self.done_times else 0.0,
            "mean_active_width": (round(self.weighted_width / self.service_total, 3)
                                  if self.service_total > 0 else 0.0),
            "tail_efficiency": (round(self.active_width_sum / float(launches * self.width), 5)
                                if self.width else 0.0),
            "launches": self.launches,
            "scheduler_ops": self.ops,
            # Zero here on policy 5 means the max-age rule never fired and
            # policy 5 IS policy 4 on this trace -- a result, not a bug.
            "max_age_triggers": self.max_age_triggers,
            "jobs_completed": len(self.done_times),
            "jobs_total": jobs,
        }


# --------------------------------------------------------------------------
# Lockstep simulator: policies 1 and 3
# --------------------------------------------------------------------------

@dataclass
class Seat:
    job: Job
    sp: int = 0

    def current(self) -> Statepoint | None:
        return self.job.statepoints[self.sp] if self.sp < len(self.job.statepoints) else None


def simulate_lockstep(jobs: list[Job], policy: Policy, m: Meter) -> None:
    """The batch runs one phase at a time and pads for the slowest slot.

    Within a round every seated slot works its CURRENT statepoint.  The round is
    max(blocks) long; a slot whose statepoint is shorter contributes nothing to
    the remaining passes and is padding -- which is exactly the M64 cohort tail.
    Refill timing is the only difference between policy 1 and policy 3.
    """
    pending = list(jobs)
    seats: list[Seat | None] = [None] * m.width

    def admit() -> None:
        for i in range(m.width):
            if seats[i] is None and pending:
                seats[i] = Seat(pending.pop(0))
                m.ops += 1

    admit()
    while True:
        active = [s for s in seats if s is not None and s.current() is not None]
        if not active:
            if not pending:
                break
            admit()
            continue

        m.ops += 1  # one cohort-round decision
        sps = [s.current() for s in active]
        n_blocks = max(sp.blocks for sp in sps if sp is not None)

        for b in range(n_blocks):
            for p in PHASE_ORDER:
                chosen = [sp for sp in sps if sp is not None and b < sp.blocks]
                if not chosen:
                    continue
                m.launch(p, max(sp.per_block.get(p, 0.0) for sp in chosen), len(chosen))

        boundary_peak = max((sp.boundary_s for sp in sps if sp is not None), default=0.0)
        if boundary_peak > 0.0:
            m.launch(BOUNDARY_PHASE, boundary_peak, len(active))

        for i, s in enumerate(seats):
            if s is None:
                continue
            s.sp += 1
            if s.current() is None:
                m.retire()
                seats[i] = None

        if policy.refill == "immediate":
            admit()
        elif all(s is None for s in seats):
            admit()


# --------------------------------------------------------------------------
# Queued simulator: policies 2, 4 and 5
# --------------------------------------------------------------------------

@dataclass
class Cursor:
    items: list[tuple[str, float]]
    at: int = 0

    def done(self) -> bool:
        return self.at >= len(self.items)


def _largest_ready(ready: dict[str, list[int]]) -> str:
    """Rev.7 8.5: largest ready count within the highest non-empty cost class."""
    best_rank = max(CLASS_RANK[COST_CLASS[p]] for p in ready)
    candidates = [p for p in ready if CLASS_RANK[COST_CLASS[p]] == best_rank]
    return max(candidates, key=lambda p: (len(ready[p]), p))


def simulate_queued(jobs: list[Job], policy: Policy, m: Meter,
                    max_steps: int = 20_000_000) -> None:
    """Each slot advances independently; a step services one phase queue."""
    pending = [Cursor(j.items()) for j in jobs]
    seats: list[Cursor | None] = [None] * m.width
    ages: dict[str, int] = {p: 0 for p in list(PHASE_ORDER) + [BOUNDARY_PHASE]}

    def admit() -> None:
        for i in range(m.width):
            if seats[i] is None and pending:
                seats[i] = pending.pop(0)
                m.ops += 1

    admit()
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("scheduler_trace_replay: step budget exhausted; "
                               "lower --max-items or raise the budget")

        # Retire whatever finished, then refill per policy.
        for i, c in enumerate(seats):
            if c is not None and c.done():
                m.retire()
                seats[i] = None
        if policy.refill == "immediate":
            admit()
        elif all(s is None for s in seats):
            admit()

        ready: dict[str, list[int]] = {}
        for i, c in enumerate(seats):
            if c is None or c.done():
                continue
            ready.setdefault(c.items[c.at][0], []).append(i)

        if not ready:
            if not pending:
                break
            admit()
            continue

        m.ops += 1  # a real queue re-ranking every launch
        if policy.select == "largest_ready_maxage" and policy.max_age is not None:
            aged = [p for p in ready if ages[p] >= policy.max_age]
            if aged:
                phase = max(aged, key=lambda p: (ages[p], len(ready[p])))
                if phase != _largest_ready(ready):
                    m.max_age_triggers += 1
            else:
                phase = _largest_ready(ready)
        else:
            phase = _largest_ready(ready)

        chosen = ready[phase]
        peak = max(seats[i].items[seats[i].at][1] for i in chosen)  # type: ignore[union-attr]
        m.launch(phase, peak, len(chosen))
        for i in chosen:
            c = seats[i]
            assert c is not None
            c.at += 1

        for p in ages:
            ages[p] += 1
        ages[phase] = 0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_policy(jobs: list[Job], policy: Policy, width: int, dispatch_share: float,
               ref_width: int, light_overlap: bool) -> dict[str, Any]:
    m = Meter(width, dispatch_share, ref_width, light_overlap)
    if policy.kind == "lockstep":
        simulate_lockstep(jobs, policy, m)
    else:
        simulate_queued(jobs, policy, m)
    return m.result(policy, len(jobs))


def replay(jobs: list[Job], width: int, dispatch_share: float, ref_width: int,
           light_overlap: bool = False, max_age: int = DEFAULT_MAX_AGE) -> dict[str, Any]:
    policies = [p if p.max_age is None else dataclasses.replace(p, max_age=max_age)
                for p in POLICIES]
    results = [run_policy(jobs, p, width, dispatch_share, ref_width, light_overlap)
               for p in policies]
    by_key = {r["policy"]: r for r in results}
    base = by_key["p1_rendezvous"]
    base_idle = base["idle_fraction"]

    for r in results:
        r["idle_reduction_pct_vs_p1"] = (
            round((base_idle - r["idle_fraction"]) / base_idle * 100.0, 3)
            if base_idle > 0 else None
        )
        r["makespan_vs_p1"] = (round(r["makespan_s"] / base["makespan_s"], 4)
                               if base["makespan_s"] > 0 else None)
        r["tail_vs_p1"] = (round(r["tail_s"] / base["tail_s"], 4)
                           if base["tail_s"] > 0 else None)
        r["scheduler_ops_vs_p1"] = (round(r["scheduler_ops"] / base["scheduler_ops"], 4)
                                    if base["scheduler_ops"] > 0 else None)

    p4 = by_key["p4_largest_ready_first"]["idle_reduction_pct_vs_p1"]
    p5 = by_key["p5_phase_queue_max_age"]["idle_reduction_pct_vs_p1"]
    candidates = [v for v in (p4, p5) if v is not None]
    verdict = ("UNKNOWN" if not candidates
               else "PASS" if max(candidates) >= IDLE_REDUCTION_GATE_PCT else "FAIL")

    notes: list[str] = []
    if len(jobs) <= width:
        notes.append(
            "fleet size <= slot width: no slot ever refills, so policy 3 can only "
            "differ from policy 1 through refill at the fleet tail.  Replay a fleet "
            "larger than the width to exercise refill (Rev.7 8.10 asks for 1,280)."
        )
    if dispatch_share >= 0.95:
        notes.append("dispatch_share near 1: narrowing a batch is modelled as free, "
                     "which flatters every queue policy.")
    if max((len(j.statepoints) for j in jobs), default=0) <= 1:
        notes.append(
            "every job has one statepoint: the boundary item of assumption A4 cannot "
            "desynchronise anything, so policies 2/4/5 will look like policy 1."
        )
    if verdict == "FAIL" and len(jobs) <= width:
        notes.append(
            "the gate FAILS on a fleet no larger than the slot width, and that is the "
            "expected shape of this model rather than a verdict on the policy: without "
            "refill headroom, letting slots advance independently FRAGMENTS the batch "
            "across phase queues, so mean active width falls and the queue policies "
            "lose to lockstep.  Rev.7 8.2 pairs the conditional outer with immediate "
            "slot refill precisely for this reason.  Re-run against a fleet several "
            "times the width (Rev.7 8.10 asks for 1,280 jobs) before concluding "
            "anything about policies 4 and 5."
        )
    if base["tail_s"] <= 1e-9:
        notes.append(
            "policy 1 reports tail 0 and that is not a win: under a cohort barrier "
            "nothing is released early, so every case completes at the same instant "
            "and the first-done -> last-done gap collapses.  It happens whenever the "
            "fleet's jobs have the same statepoint count, which an M64 campaign of one "
            "cycle deck does.  Rank on makespan, idle_fraction and tail_efficiency "
            "here; tail is only informative once cases are released independently."
        )
    if by_key["p5_phase_queue_max_age"]["max_age_triggers"] == 0:
        notes.append(
            f"the max-age rule never fired at MAX_AGE={max_age}: policy 5 is policy 4 "
            "on this trace.  A queue only stays hungry while higher-class work keeps "
            "arriving, and in this cost model a job blocked on a light phase generates "
            "no heavy work, so heavy queues drain and light ones are serviced anyway.  "
            "Sweep --max-age 4/8/16 (Rev.7 8.7) before implementing the rule."
        )

    return {
        "probe": "scheduler_replay",
        "policies": results,
        "gate": {
            "name": "Rev.7 8.8 adoption gate",
            "rule": "policy 4 or 5 predicts >= 20% idle reduction vs policy 1",
            "threshold_pct": IDLE_REDUCTION_GATE_PCT,
            "p4_idle_reduction_pct": p4,
            "p5_idle_reduction_pct": p5,
            "verdict": verdict,
        },
        "model": {
            "dispatch_share": dispatch_share,
            "reference_width": ref_width,
            "slot_width": width,
            "light_overlap": light_overlap,
            "max_age": max_age,
            "phase_order": PHASE_ORDER,
            "boundary_phase": BOUNDARY_PHASE,
            "cost_class": COST_CLASS,
            "single_heavy_phase_at_a_time": True,
        },
        "fleet": {
            "jobs": len(jobs),
            "statepoints": sum(len(j.statepoints) for j in jobs),
            "total_outers": sum(j.outers for j in jobs),
            "total_items": sum(len(j.items()) for j in jobs),
            "wall_sum_s": round(sum(j.wall for j in jobs), 3),
            "wall_p50_s": round(statistics.median([j.wall for j in jobs]), 3) if jobs else 0.0,
            "wall_max_s": round(max((j.wall for j in jobs), default=0.0), 3),
        },
        "notes": notes,
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def selftest() -> int:
    failures = 0

    def check(cond: bool, message: str) -> None:
        nonlocal failures
        if not cond:
            print(f"scheduler_trace_replay selftest: FAIL: {message}", file=sys.stderr)
            failures += 1

    jobs = synthetic_fleet(n_jobs=96, statepoints=6, outers_mean=30,
                           seed=20260828, coarsen=3)
    check(len(jobs) == 96, "synthetic fleet size wrong")
    check(all(j.statepoints for j in jobs), "a synthetic job has no statepoints")
    check(len({len(j.statepoints) for j in jobs}) > 1,
          "synthetic fleet has a uniform statepoint count; A4 could not desynchronise it")

    out = replay(jobs, width=16, dispatch_share=DEFAULT_DISPATCH_SHARE, ref_width=64)
    check(len(out["policies"]) == 5, "not five policies")
    keys = [p["policy"] for p in out["policies"]]
    check(keys == [p.key for p in POLICIES], f"policy order changed: {keys}")

    by = {p["policy"]: p for p in out["policies"]}
    for k, r in by.items():
        check(r["makespan_s"] > 0.0, f"{k}: non-positive makespan")
        check(r["jobs_completed"] == len(jobs),
              f"{k}: {r['jobs_completed']} of {len(jobs)} jobs finished")
        check(0.0 <= r["gpu_busy_fraction"] <= 1.0, f"{k}: busy fraction out of range")
        check(r["mean_active_width"] <= 16.0 + 1e-9, f"{k}: active width exceeds the slot count")

    # A replay whose policies all score alike is not measuring anything.
    idles = {k: r["idle_fraction"] for k, r in by.items()}
    check(len(set(round(v, 4) for v in idles.values())) >= 3,
          f"policies are not distinguishable: {idles}")

    # The monotonic sanity the task names.
    check(by["p5_phase_queue_max_age"]["tail_s"] <= by["p1_rendezvous"]["tail_s"] + 1e-9,
          f'policy 5 tail {by["p5_phase_queue_max_age"]["tail_s"]} > policy 1 tail '
          f'{by["p1_rendezvous"]["tail_s"]}')
    check(by["p5_phase_queue_max_age"]["idle_fraction"] <= by["p1_rendezvous"]["idle_fraction"] + 1e-9,
          "policy 5 wastes more slot capacity than policy 1")
    check(by["p4_largest_ready_first"]["idle_fraction"] <= by["p1_rendezvous"]["idle_fraction"] + 1e-9,
          "policy 4 wastes more slot capacity than policy 1")
    # Queue policies think harder than lockstep ones; if they do not, the ops
    # metric is not measuring scheduler work and assumption A8's warning is void.
    check(by["p4_largest_ready_first"]["scheduler_ops"] > by["p1_rendezvous"]["scheduler_ops"],
          "queue policies are not charged more scheduler ops than rendezvous")

    check(out["gate"]["verdict"] in {"PASS", "FAIL", "UNKNOWN"}, "gate verdict missing")
    check(out["gate"]["threshold_pct"] == IDLE_REDUCTION_GATE_PCT, "gate threshold moved")

    small = replay(jobs[:8], width=16, dispatch_share=DEFAULT_DISPATCH_SHARE, ref_width=64)
    check(any("refill" in n for n in small["notes"]),
          "no note when the fleet cannot exercise refill")

    # The ranking must survive the assumption-A5 sweep, or it is an artifact of
    # the guessed dispatch share rather than of the policies.
    for s in (0.3, 0.6, 0.9):
        o = replay(jobs, width=16, dispatch_share=s, ref_width=64)
        b = {p["policy"]: p for p in o["policies"]}
        check(b["p5_phase_queue_max_age"]["tail_s"] <= b["p1_rendezvous"]["tail_s"] + 1e-9,
              f"dispatch_share={s}: policy 5 tail exceeds policy 1 tail")
        check(b["p5_phase_queue_max_age"]["idle_fraction"]
              <= b["p1_rendezvous"]["idle_fraction"] + 1e-9,
              f"dispatch_share={s}: policy 5 idle exceeds policy 1 idle")

    # SPTELEM parsing must survive the real receipt shape, residual and all.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "run.log"
        p.write_text(
            '[RASBERY][SPTELEM] {"schema_version":1,"job_id":"deck0","slot":0,'
            '"statepoint":0,"outers":10,"wall":12.0,"phase_wall":{"updpsi":0.1,'
            '"setls":1.2,"drive":5.5,"updjnet":0.25,"nodal":0.6,"cusping":0.1,'
            '"upddhat":0.58}}\n'
            'noise line that is not a receipt\n'
            '[RASBERY][SPTELEM] {"schema_version":1,"job_id":"deck0","slot":0,'
            '"statepoint":1,"outers":4,"wall":6.0,"phase_wall":{"updpsi":0.05,'
            '"setls":0.6,"drive":2.7,"updjnet":0.12,"nodal":0.3,"cusping":0.0,'
            '"upddhat":0.29}}\n',
            encoding="utf-8")
        parsed = load_sptelem(p)
        check(len(parsed) == 1, f"expected 1 job from the receipts, got {len(parsed)}")
        check(len(parsed[0].statepoints) == 2, "both statepoints must be kept")
        check(parsed[0].outers == 14, f"outers not summed: {parsed[0].outers}")
        b0 = parsed[0].statepoints[0].boundary_s
        check(abs(b0 - (12.0 - 8.33)) < 1e-6,
              f"boundary residual wrong: {b0}")

    if failures:
        print(f"scheduler_trace_replay selftest: {failures} failure(s)", file=sys.stderr)
        return 1
    print("scheduler_trace_replay selftest: PASS")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline scheduler policy replay (Rev.7 8.8)")
    ap.add_argument("--sptelem", type=Path,
                    help="run log or JSONL with [RASBERY][SPTELEM] receipts")
    ap.add_argument("--walls", type=Path,
                    help="fallback: TOTAL DRIVER TIME lines, one per job")
    ap.add_argument("--statepoints", type=int, default=12,
                    help="statepoints per job on the --walls path (default: 12)")
    ap.add_argument("--outers-per-statepoint", type=int, default=28,
                    help="outers per statepoint on the --walls path (default: 28)")
    ap.add_argument("--width", type=int, default=64, help="physical slot count (default: 64)")
    ap.add_argument("--ref-width", type=int, default=64,
                    help="width the trace was recorded at (default: 64)")
    ap.add_argument("--dispatch-share", type=float, default=DEFAULT_DISPATCH_SHARE,
                    help="assumption A5; sweep 0.3/0.6/0.9 before quoting a result")
    ap.add_argument("--light-overlap", action="store_true",
                    help="give light phases a free control-stream lane (off by default)")
    ap.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE,
                    help="policy 5 queue-age bound; Rev.7 8.7 sweeps 4/8/16 (default: 8)")
    ap.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS,
                    help="auto-coarsen outers so total work items stay under this")
    ap.add_argument("--json-out", type=Path, help="also write the receipt here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        return selftest()

    if args.width <= 0:
        ap.error("--width must be positive")
    if not (0.0 <= args.dispatch_share <= 1.0):
        ap.error("--dispatch-share must be in [0, 1]")

    if args.sptelem:
        if not args.sptelem.is_file():
            print(f"scheduler_trace_replay: no such file: {args.sptelem}", file=sys.stderr)
            return 2
        jobs = load_sptelem(args.sptelem, coarsen=1)
        if not jobs:
            print("scheduler_trace_replay: no SPTELEM receipts with phase_wall found; "
                  "was the run made with RASBERY_STATEPOINT_TELEMETRY=1?", file=sys.stderr)
            return 2
        c = auto_coarsen(sum(j.outers for j in jobs), args.max_items)
        if c > 1:
            jobs = load_sptelem(args.sptelem, coarsen=c)
            print(f"scheduler_trace_replay: coarsening {c} outers per work item "
                  f"to stay under --max-items {args.max_items}", file=sys.stderr)
    elif args.walls:
        if not args.walls.is_file():
            print(f"scheduler_trace_replay: no such file: {args.walls}", file=sys.stderr)
            return 2
        probe = load_walls(args.walls, args.statepoints, args.outers_per_statepoint, 1)
        if not probe:
            print("scheduler_trace_replay: no walls parsed", file=sys.stderr)
            return 2
        c = auto_coarsen(sum(j.outers for j in probe), args.max_items)
        jobs = (load_walls(args.walls, args.statepoints, args.outers_per_statepoint, c)
                if c > 1 else probe)
    else:
        ap.error("one of --sptelem, --walls or --selftest is required")

    out = replay(jobs, args.width, args.dispatch_share, args.ref_width,
                 args.light_overlap, args.max_age)
    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
