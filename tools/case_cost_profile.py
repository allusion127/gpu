#!/usr/bin/env python3
"""Fold RASBERY run logs into a per-case cost ledger.

The GA evaluator question is not "how fast is the solver" but "what does one
case cost, and how much of that cost would a persistent evaluator never pay
again".  This tool answers exactly that, from receipts the binary already
prints, and it refuses to guess: every column below is a receipt field or a
difference of two receipt fields, never a model.

Receipts consumed (all optional; missing ones become blank cells):

  ``  [TIMING] Init+IO=<s>``          Driver.h:3950  deck+XSLIB parse, solver
                                      construction, arena admission, OpenResult
  ``  [TIMING] IO write=<s>``         Driver.h:4232  driver-thread I/O charge
  ``  TOTAL DRIVER TIME=<s>``         Driver.h:4233  Drive() entry -> exit
  ``[RASBERY][SPTELEM][SUMMARY]``     Driver.h:4249  the full per-run counters,
                                      including library_seconds and solve_wall
  ``[RASBERY][IO_WRITER][SUMMARY]``   IoWriter.h     bytes, writer_busy_ms
  ``[RASBERY][TRAJECTORY]``           Driver.h:4305  outers, statepoints, digest
  ``[RASBERY][CUDA][BATCH_OCCUPANCY]`` CudaBICGBackend.cu:5525  mean_width
  ``[RASBERY][REFILL]``               BatchRefill.h  tail_idle_s, slot_busy
  ``[RASBERY][XSLIB_CACHE]``          XSSet.cpp      loads, hits, bytes,
                                      lock_wait_ms -- the host library parse
  ``[RASBERY][PROCESS]``              main.cpp       exec_s, pre_drive_s,
                                      drive_s, post_drive_s: the decomposition
                                      of what `wall - drive` used to lump
  ``[RASBERY][READINPUT]``            IO.cpp         deck_s, geometry_s, xs_s,
                                      rest_s: the decomposition of Init+IO

The process wall is NOT a receipt -- the Driver cannot observe its own startup
or teardown.  Pass it with ``--wall-dir DIR``: for a log ``NAME.log`` the tool
reads ``DIR/NAME.wall`` (the first float in it), which is what
``/usr/bin/time -f "%e" -o NAME.wall`` writes.  ``outside_drive`` is then
``wall - TOTAL DRIVER TIME``: process start, CUDA context creation, arena
teardown, context destruction.  Without a .wall file that column stays blank
rather than being invented.

The ledger splits one case into four buckets:

  fixed_startup   outside_drive          once per PROCESS in batch, once per
                                         case in one-case-per-process runs
  fixed_percase   Init+IO                once per CASE today; a persistent
                                         evaluator amortises library_seconds
                                         and the arena admission inside it
  physics         solve_wall             the only bucket that is the answer
  output          IO write               driver-thread charge for results;
                                         writer_busy_ms is the overlapped part

WP9-A splits `physics` further, from the SPTELEM summary's four wall objects:

  phase_wall      the seven outer-body phases (the `d x outer` half)
  loop_wall       host work inside SolveLoop, outside the outer body: T/H, the
                  Xe Picard step, the secant proposal, the trial application
  floor_wall      the statepoint boundary: PPR, depletion, result packing/write
  nested_wall     UpdateFlatXS, which runs INSIDE the two above and is therefore
                  reported beside them and never summed with them
  floor_transfer  the H2D/D2H the BOUNDARY paid, i.e. after the last SolveLoop

`residual` is `solve_wall - (phase + loop + floor)`: the part still unnamed.

Usage
-----
    tools/case_cost_profile.py ~/gaplan/*.log --wall-dir ~/gaplan
    tools/case_cost_profile.py run.log --json > ledger.json
    tools/case_cost_profile.py batchrun.log --batch
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TIMING_INIT = re.compile(r"\[TIMING\]\s+Init\+IO=([0-9.]+)\s*s")
TIMING_IO = re.compile(r"\[TIMING\]\s+IO write=([0-9.]+)\s*s")
TIMING_TOTAL = re.compile(r"TOTAL DRIVER TIME=\s*([0-9.]+)\s*s")
STEP_LINE = re.compile(
    r"NO\.=\s*(\d+)\s+EFPD=\s*([0-9.eE+-]+)\s+K-EFF=([0-9.]+)\s+PPM=\s*([0-9.eE+-]+)"
    r"\s+outer=\s*(\d+)\s+TH=\s*(\d+)\s+t=\s*([0-9.]+)s")


def receipts(text: str, tag: str) -> list[dict]:
    """Every JSON object printed after `tag` on its own line."""
    out = []
    for line in text.splitlines():
        idx = line.find(tag)
        if idx < 0:
            continue
        brace = line.find("{", idx)
        if brace < 0:
            continue
        try:
            out.append(json.loads(line[brace:]))
        except json.JSONDecodeError:
            continue
    return out


def first(seq: list[dict]) -> dict:
    return seq[0] if seq else {}


def read_wall(wall_dir: Path | None, log: Path) -> float | None:
    if wall_dir is None:
        return None
    cand = wall_dir / (log.stem + ".wall")
    if not cand.exists():
        return None
    for token in cand.read_text(errors="replace").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def profile_log(log: Path, wall_dir: Path | None) -> dict:
    text = log.read_text(errors="replace")
    rec: dict = {"log": str(log), "name": log.stem}

    # A batch log holds one [TIMING] set PER CASE.  Charging the whole process
    # wall against the first case's Drive() would invent an "outside_drive" of
    # the entire batch, so a multi-case log gets a different ledger: the wall is
    # the batch's, the per-case numbers are a distribution, and the amortisation
    # split (which is a statement about ONE case) is not printed at all.
    inits = [float(x) for x in TIMING_INIT.findall(text)]
    ios = [float(x) for x in TIMING_IO.findall(text)]
    drives = [float(x) for x in TIMING_TOTAL.findall(text)]
    rec["cases"] = max(len(drives), 1)
    batch = len(drives) > 1
    rec["batch"] = batch
    if batch:
        rec["init_seconds_min"] = min(inits) if inits else None
        rec["init_seconds_max"] = max(inits) if inits else None
        rec["driver_seconds_min"] = min(drives)
        rec["driver_seconds_max"] = max(drives)
        rec["driver_seconds_mean"] = sum(drives) / len(drives)
        rec["io_driver_seconds_sum"] = sum(ios)
    else:
        if inits:
            rec["init_seconds"] = inits[0]
        if ios:
            rec["io_driver_seconds"] = ios[0]
        if drives:
            rec["driver_seconds"] = drives[0]

    sp = {} if batch else first(receipts(text, "[RASBERY][SPTELEM][SUMMARY]"))
    for key in ("library_seconds", "solve_wall", "io_wall", "total_seconds",
                "outers", "statepoints", "cmfd_sweeps", "bicg_iters",
                "graph_launches_delta", "h2d_bytes_delta", "d2h_bytes_delta",
                "d2h_calls_delta", "search_trials", "th_updates",
                "xe_cascades", "xe_updates", "xe_outers", "search_outers",
                "settle_outers", "outers_initial", "th_outers", "solve_loops"):
        if key in sp:
            rec[key] = sp[key]
    if "solve_wall" in sp:
        rec["driver_seconds"] = sp.get("total_seconds", rec.get("driver_seconds"))
        rec["init_seconds"] = sp.get("init_seconds", rec.get("init_seconds"))
        rec["io_driver_seconds"] = sp.get("io_wall", rec.get("io_driver_seconds"))

    # WP9-A: the host floor, named.  THREE objects, and they do not overlap:
    #   phase_wall     the seven outer-body phases -- the `d x outer` half of
    #                  the GA plan Sec 2.2 cost model.
    #   loop_wall      host work inside SolveLoop but outside the outer body:
    #                  T/H, the Xe step, the secant, the trial application.
    #                  This is the 0.377 ms/outer the plan's Sec 3.2 could name
    #                  but not measure.
    #   floor_wall     the statepoint boundary -- PPR, depletion, result -- the
    #                  `c` half.
    # `nested_wall` is DELIBERATELY excluded from the sum: UpdateFlatXS runs
    # inside loop_wall/floor_wall buckets, so adding it would double count.
    for obj, prefix in (("phase_wall", "phase_"), ("loop_wall", "loop_"),
                        ("floor_wall", "floor_"), ("nested_wall", "nested_"),
                        ("floor_transfer", "floortx_"), ("search", "srch_")):
        block = sp.get(obj) or {}
        for name, value in block.items():
            rec[prefix + name] = value
        if block and obj not in ("floor_transfer", "nested_wall", "search"):
            rec[prefix + "sum"] = sum(float(v) for v in block.values())
    if "nested_flatxs" in rec:
        rec["nested_sum"] = float(rec["nested_flatxs"])

    traj = first(receipts(text, "[RASBERY][TRAJECTORY]"))
    for key in ("digest", "outers", "statepoints", "th_updates", "telemetry"):
        if key in traj:
            rec.setdefault(key, traj[key])
    rec["digest"] = traj.get("digest", rec.get("digest"))

    iow = first(receipts(text, "[RASBERY][IO_WRITER][SUMMARY]"))
    for src, dst in (("bytes", "out_bytes"), ("writer_busy_ms", "writer_busy_ms"),
                     ("enqueue_block_ms", "enqueue_block_ms"), ("ops", "io_ops"),
                     ("failures", "io_failures"), ("skipped", "io_skipped")):
        if src in iow:
            rec[dst] = iow[src]

    lock = first(receipts(text, "[RASBERY][HDF5][LOCK]"))
    if lock:
        rec["hdf5_acquires"] = lock.get("acquires")
        rec["hdf5_wait_ms"] = lock.get("wait_ms")

    xslib = first(receipts(text, "[RASBERY][XSLIB_CACHE]"))
    for src, dst in (("loads", "xslib_loads"), ("hits", "xslib_hits"),
                     ("bytes", "xslib_bytes"), ("lock_wait_ms", "xslib_lock_wait_ms")):
        if src in xslib:
            rec[dst] = xslib[src]

    proc = first(receipts(text, "[RASBERY][PROCESS]"))
    for src, dst in (("exec_s", "process_exec_s"), ("pre_drive_s", "process_pre_drive_s"),
                     ("drive_s", "process_drive_s"), ("post_drive_s", "process_post_drive_s"),
                     ("in_main_s", "process_in_main_s")):
        if src in proc:
            rec[dst] = proc[src]

    # The Init+IO staircase, as ONE number.  Eight workers that queue behind one
    # 34 MB parse report a rising Init+IO with a constant slope (4.108 .. 14.506
    # s, ~1.49 s/case before the host XSLIB cache).  `init_ramp_seconds` is the
    # spread and `init_ramp_slope` is that spread per case: both go to ~0 when
    # the parse stops being per-case, and neither can be read off a mean.
    if len(inits) > 1:
        rec["init_ramp_seconds"] = max(inits) - min(inits)
        rec["init_ramp_slope"] = rec["init_ramp_seconds"] / (len(inits) - 1)
        rec["init_seconds_sum"] = sum(inits)

    readin = receipts(text, "[RASBERY][READINPUT]")
    if readin:
        for key in ("deck_s", "geometry_s", "xs_s", "rest_s", "total_s"):
            values = [r[key] for r in readin if key in r]
            if values:
                rec["readinput_" + key] = sum(values) / len(values)
                if len(values) > 1:
                    rec["readinput_" + key + "_max"] = max(values)

    occ = receipts(text, "[RASBERY][CUDA][BATCH_OCCUPANCY]")
    if occ:
        widest = max(occ, key=lambda o: o.get("launches", 0))
        rec["slots"] = widest.get("slots")
        rec["mean_width"] = widest.get("mean_width")
        rec["effective_mean_width"] = widest.get("effective_mean_width")
        rend = widest.get("claim_rendezvous") or {}
        rec["rendezvous_wait_ms"] = rend.get("wait_ms")

    ref = first(receipts(text, "[RASBERY][REFILL]"))
    for key in ("jobs", "slots", "lanes", "refills", "wall_s", "tail_idle_s",
                "slot_busy_fraction", "duplicates", "stale_tenants",
                "double_releases"):
        if key in ref:
            rec["refill_" + key] = ref[key]

    steps = [(int(a), float(b), float(c), float(d), int(e), int(f), float(g))
             for a, b, c, d, e, f, g in STEP_LINE.findall(text)]
    if steps:
        rec["step_count"] = len(steps)
        rec["step_seconds_sum"] = sum(s[6] for s in steps)
        rec["step_seconds_max"] = max(s[6] for s in steps)
        rec["step_seconds_min"] = min(s[6] for s in steps)
        rec["efpd_last"] = steps[-1][1]

    wall = read_wall(wall_dir, log)
    if wall is not None:
        rec["wall_seconds"] = wall
        if batch:
            rec["cases_per_hour"] = 3600.0 * rec["cases"] / wall
            rec["effective_seconds_per_case"] = wall / rec["cases"]
        elif rec.get("driver_seconds") is not None:
            rec["outside_drive_seconds"] = wall - rec["driver_seconds"]

    if batch:
        # No per-case ledger from a batch log: the fields above are the batch's.
        return rec

    solve = rec.get("solve_wall")
    if solve is None and "driver_seconds" in rec:
        # Without the telemetry summary, solve is what is left of Drive().
        known = (rec.get("init_seconds") or 0.0) + (rec.get("io_driver_seconds") or 0.0)
        solve = rec["driver_seconds"] - known
        rec["solve_wall_inferred"] = solve
    if solve is not None and rec.get("outers"):
        rec["ms_per_outer"] = 1000.0 * solve / float(rec["outers"])
    if rec.get("outers") and rec.get("statepoints"):
        rec["outers_per_statepoint"] = rec["outers"] / float(rec["statepoints"])

    # The residual is the point of the decomposition: it is what the three
    # named objects still do not account for inside solve_wall.  A residual that
    # stays large after WP9-A is a phase nobody has named yet, and saying so is
    # the whole job of this line.
    if solve is not None and ("phase_sum" in rec or "floor_sum" in rec):
        named = (rec.get("phase_sum") or 0.0) + (rec.get("loop_sum") or 0.0) + \
                (rec.get("floor_sum") or 0.0)
        rec["named_seconds"] = named
        rec["residual_seconds"] = solve - named
        if solve > 0.0:
            rec["named_fraction"] = named / solve

    base = rec.get("wall_seconds") or rec.get("driver_seconds")
    if base:
        amort = (rec.get("outside_drive_seconds") or 0.0) + (rec.get("init_seconds") or 0.0)
        rec["amortisable_seconds"] = amort
        rec["amortisable_fraction"] = amort / base
        if solve is not None:
            rec["physics_fraction"] = solve / base
        if rec.get("io_driver_seconds") is not None:
            rec["output_fraction"] = rec["io_driver_seconds"] / base
    return rec


COLUMNS = [
    ("name", "run", "{}", 16),
    ("cases", "n", "{:d}", 3),
    ("wall_seconds", "wall", "{:.2f}", 8),
    ("driver_seconds", "drive", "{:.2f}", 7),
    ("outside_drive_seconds", "outside", "{:.2f}", 8),
    ("init_seconds", "init+io", "{:.2f}", 8),
    ("library_seconds", "library", "{:.2f}", 8),
    ("solve_wall", "solve", "{:.2f}", 8),
    ("io_driver_seconds", "io(drv)", "{:.2f}", 8),
    ("writer_busy_ms", "wr_busy_ms", "{:.0f}", 11),
    ("out_bytes", "out_bytes", "{:d}", 12),
    ("statepoints", "sp", "{:d}", 4),
    ("outers", "outers", "{:d}", 7),
    ("outers_per_statepoint", "out/sp", "{:.1f}", 7),
    ("ms_per_outer", "ms/outer", "{:.2f}", 9),
    ("mean_width", "width", "{:.2f}", 7),
    ("init_ramp_slope", "ramp/case", "{:.2f}", 9),
    ("xslib_loads", "xsloads", "{:d}", 7),
    ("digest", "digest", "{}", 17),
]


def render(rows: list[dict], stream=sys.stdout) -> None:
    header = "  ".join(f"{label:>{w}}" for _, label, _, w in COLUMNS)
    stream.write(header + "\n")
    stream.write("-" * len(header) + "\n")
    for row in rows:
        cells = []
        for key, _, fmt, w in COLUMNS:
            value = row.get(key)
            if value is None:
                cells.append(" " * w)
                continue
            try:
                cells.append(f"{fmt.format(value):>{w}}")
            except (ValueError, TypeError):
                cells.append(f"{str(value):>{w}}")
        stream.write("  ".join(cells) + "\n")
    stream.write("\n")
    for row in rows:
        if row.get("batch"):
            stream.write(
                f"{row['name']:>16}  BATCH  cases {row['cases']}  "
                f"c/h {row.get('cases_per_hour', float('nan')):8.1f}  "
                f"eff {row.get('effective_seconds_per_case', float('nan')):7.2f} s/case  "
                f"drive/case {row.get('driver_seconds_min', 0):6.1f}–"
                f"{row.get('driver_seconds_max', 0):.1f} s  "
                f"init/case {row.get('init_seconds_min') or 0:5.2f}–"
                f"{row.get('init_seconds_max') or 0:.2f} s  "
                f"hdf5_lock_wait {row.get('hdf5_wait_ms', 0) / 1000.0:8.1f} s  "
                f"init_ramp {row.get('init_ramp_seconds') or 0:5.2f} s "
                f"({row.get('init_ramp_slope') or 0:.2f} s/case)  "
                f"xslib loads {row.get('xslib_loads', '-')}"
                f"/hits {row.get('xslib_hits', '-')}\n")
            continue
        if "amortisable_fraction" not in row:
            continue
        stream.write(
            f"{row['name']:>16}  amortisable {row['amortisable_seconds']:6.2f} s "
            f"({100 * row['amortisable_fraction']:5.1f} %)   "
            f"physics {100 * row.get('physics_fraction', float('nan')):5.1f} %   "
            f"output {100 * row.get('output_fraction', float('nan')):5.1f} %\n")
        if row.get("process_in_main_s") is not None:
            stream.write(
                f"{row['name']:>16}  process  exec "
                f"{row.get('process_exec_s') if row.get('process_exec_s') is not None else float('nan'):.2f} s  "
                f"pre_drive {row.get('process_pre_drive_s', float('nan')):.2f} s  "
                f"drive {row.get('process_drive_s', float('nan')):.2f} s  "
                f"post_drive {row.get('process_post_drive_s', float('nan')):.2f} s  "
                f"unaccounted "
                f"{(row.get('wall_seconds') or float('nan')) - row['process_in_main_s'] - (row.get('process_exec_s') or 0.0):.2f} s\n")
        # WP9-A floor split.  Printed only when the SPTELEM summary carried it,
        # so a log from a run without RASBERY_STATEPOINT_TELEMETRY is silent
        # here rather than showing a row of zeros.
        if row.get("floor_sum") is not None or row.get("loop_sum") is not None:
            stream.write(
                f"{row['name']:>16}  floor      ppr "
                f"{(row.get('floor_ppr_reset') or 0) + (row.get('floor_ppr_drive') or 0) + (row.get('floor_ppr_recon') or 0):6.2f} s  "
                f"depl {(row.get('floor_depl_predictor') or 0) + (row.get('floor_depl_corrector') or 0):5.2f} s  "
                f"result {(row.get('floor_result_add') or 0) + (row.get('floor_result_write') or 0):5.2f} s  "
                f"= {row.get('floor_sum') or 0:6.2f} s\n")
            stream.write(
                f"{row['name']:>16}  loop       th "
                f"{row.get('loop_th_update') or 0:6.2f} s  "
                f"xe {row.get('loop_xe_step') or 0:6.2f} s  "
                f"search {(row.get('loop_search_propose') or 0) + (row.get('loop_search_apply') or 0):5.2f} s  "
                f"= {row.get('loop_sum') or 0:6.2f} s\n")
            stream.write(
                f"{row['name']:>16}  nested     flatxs {row.get('nested_flatxs') or 0:6.2f} s "
                f"in {int(row.get('nested_flatxs_calls') or 0):6d} calls "
                f"({1000.0 * (row.get('nested_flatxs') or 0) / max(1, int(row.get('nested_flatxs_calls') or 0)):.2f} ms/call)"
                f"   [inside loop/floor, NOT additive]\n")
            stream.write(
                f"{row['name']:>16}  accounted  outer {row.get('phase_sum') or 0:6.2f} s  "
                f"loop {row.get('loop_sum') or 0:6.2f} s  "
                f"floor {row.get('floor_sum') or 0:6.2f} s  "
                f"residual {row.get('residual_seconds', float('nan')):6.2f} s  "
                f"({100 * (row.get('named_fraction') or 0):.1f} % of solve named)\n")
        # WP9-D.  The search ledger.  `outers/trial` is the number the plan's
        # trial-reduction options are priced against: a trial is not one outer,
        # it is a Xe cascade's worth of re-convergence, and the ratio says how
        # many.  The method split says WHICH lever could apply -- a run that is
        # mostly `probe` has no slope to carry and wants a seeded bracket; one
        # that is mostly `bisect` has a bracket that is not narrowing.
        if row.get("srch_trials") is not None:
            trials = int(row.get("srch_trials") or 0)
            outers = float(row.get("search_outers") or 0)
            stream.write(
                f"{row['name']:>16}  search     trials {trials:5d}  "
                f"outers {int(outers):6d}  "
                f"({outers / max(1, trials):6.1f} outers/trial, "
                f"{100.0 * outers / max(1.0, float(row.get('outers') or 1)):4.1f} % of run)  "
                f"proposals {int(row.get('srch_proposals') or 0):5d}  "
                f"refused {int(row.get('srch_refused') or 0):4d}\n")
            stream.write(
                f"{row['name']:>16}  search     method  probe {int(row.get('srch_probe') or 0):5d}"
                f"  carry {int(row.get('srch_carry_secant') or 0):5d}"
                f"  secant {int(row.get('srch_secant') or 0):5d}"
                f"  bisect {int(row.get('srch_bisect') or 0):5d}"
                f"  iterations {int(row.get('srch_iterations') or 0):5d}\n")
        if row.get("floortx_d2h_bytes") is not None:
            stream.write(
                f"{row['name']:>16}  boundary   H2D "
                f"{(row.get('floortx_h2d_bytes') or 0) / 1e6:8.1f} MB in "
                f"{int(row.get('floortx_h2d_calls') or 0):6d} calls   D2H "
                f"{(row.get('floortx_d2h_bytes') or 0) / 1e6:8.1f} MB in "
                f"{int(row.get('floortx_d2h_calls') or 0):6d} calls\n")
        if row.get("readinput_total_s") is not None:
            stream.write(
                f"{row['name']:>16}  readinput  deck {row.get('readinput_deck_s', float('nan')):.2f} s  "
                f"geometry {row.get('readinput_geometry_s', float('nan')):.2f} s  "
                f"xs {row.get('readinput_xs_s', float('nan')):.2f} s  "
                f"rest {row.get('readinput_rest_s', float('nan')):.2f} s  "
                f"total {row['readinput_total_s']:.2f} s\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--wall-dir", type=Path, default=None,
                    help="directory holding NAME.wall sidecars from /usr/bin/time -f %%e")
    ap.add_argument("--json", action="store_true", help="emit the rows as JSON")
    ap.add_argument("--sort", default=None, help="sort rows by this key")
    args = ap.parse_args(argv)

    logs: list[Path] = []
    for entry in args.logs:
        if entry.is_dir():
            logs.extend(sorted(entry.glob("*.log")))
        else:
            logs.append(entry)

    rows = [profile_log(log, args.wall_dir) for log in logs if log.exists()]
    if args.sort:
        rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort)))
    if args.json:
        json.dump(rows, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        render(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
