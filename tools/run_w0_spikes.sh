#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# W0 decision-spike runner: build and run all five probes, emit ONE receipt.
#
# The five probes gate the Rev.7.1 programme's first waves.  Each is standalone
# and each prints its own JSON; this builds them, runs them, and folds the
# results into a single receipt on stdout (progress goes to stderr, so stdout is
# pure JSON and can be piped).
#
#   1 probe_dispatch_floor.cu     graph-node dispatch cost      -> c_dispatch
#   2 probe_gridsync_cost.cu      cooperative grid.sync() cost  -> c_barrier
#   3 probe_conditional_graph.cu  WHILE/SWITCH legality + cost  -> backend choice
#   4 probe_l2_width.sh           L2 / DRAM at widths 1/8/22/64 -> W2 gate
#   5 scheduler_trace_replay.py   offline policy replay         -> Rev.7 8.8 gate
#
# Probes 1-3 need only a GPU.  Probe 4 needs the production binary (--bin) and,
# ideally, Nsight Compute.  Probe 5 needs an M64 run log carrying SPTELEM
# receipts (--sptelem).  Anything not supplied is recorded as "skipped" with a
# reason -- never silently dropped, and never faked.
#
# USAGE (server 238)
#   tools/run_w0_spikes.sh --outdir /tmp/w0 \
#       [--arch sm_120] [--gpu 0] \
#       [--bin /path/to/RASBERY] \
#       [--sptelem /path/to/m64_run.log] \
#       [--receipt /tmp/w0/w0_spikes_receipt.json] \
#       [--skip l2_width,scheduler_replay]
#
# Total runtime: ~2 min for probes 1-3; probe 4 adds 10-40 min under ncu.
# ---------------------------------------------------------------------------
set -u

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TOOLS/.." && pwd)"

OUTDIR="/tmp/w0"
ARCH="sm_120"
GPU="0"
BIN=""
SPTELEM=""
RECEIPT=""
SKIP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --arch)    ARCH="$2"; shift 2 ;;
        --gpu)     GPU="$2"; shift 2 ;;
        --bin)     BIN="$2"; shift 2 ;;
        --sptelem) SPTELEM="$2"; shift 2 ;;
        --receipt) RECEIPT="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "run_w0_spikes: unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
[ -n "$RECEIPT" ] || RECEIPT="$OUTDIR/w0_spikes_receipt.json"

skipped() { case ",$SKIP," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }
note()    { echo "run_w0_spikes: $*" >&2; }

set_status() { printf '%s\n' "$2" > "$OUTDIR/status_$1.txt"; }
set_reason() { printf '%s\n' "$2" > "$OUTDIR/reason_$1.txt"; }
set_cmds()   { printf '%s\n' "$2" > "$OUTDIR/cmd_$1.txt"; }

# --------------------------------------------------------------------------
# Probes 1-3: nvcc build, then run.
# --------------------------------------------------------------------------
build_and_run() {
    local name="$1" src="$2" pre="$3" post="$4"
    if skipped "$name"; then
        set_status "$name" "skipped"; set_reason "$name" "--skip $name"
        note "$name: skipped by request"
        return 0
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        set_status "$name" "skipped"; set_reason "$name" "nvcc not on PATH"
        note "$name: nvcc not found"
        return 0
    fi

    local exe="$OUTDIR/$name"
    # An array, not a string: $TOOLS or $OUTDIR may contain a space, and a
    # word-split build line would then compile the wrong file or nothing.
    # $extra is our own literal and IS meant to split into separate flags.
    local -a build=(nvcc -O3 -std=c++17 "-arch=$ARCH")
    [ -n "$pre" ] && build+=($pre)
    build+=(-o "$exe" "$TOOLS/$src")
    # Libraries go AFTER the translation unit: GNU ld resolves left to right, so
    # -lcudadevrt ahead of the source can leave the device-runtime symbols
    # undefined even though the library was named.
    [ -n "$post" ] && build+=($post)
    local run="CUDA_VISIBLE_DEVICES=$GPU $exe"
    set_cmds "$name" "${build[*]}
$run"

    note "$name: building"
    if ! "${build[@]}" > "$OUTDIR/build_$name.log" 2>&1; then
        set_status "$name" "build_failed"
        set_reason "$name" "$(tail -n 20 "$OUTDIR/build_$name.log" | tr '\n' ' ')"
        note "$name: BUILD FAILED (see $OUTDIR/build_$name.log)"
        return 0
    fi

    note "$name: running"
    # Outer bound on top of probe 3's own SIGALRM watchdog.  A wedged conditional
    # graph must never be able to hang this script: if the probe's internal
    # watchdog is itself blocked (or the probe is one without one), `timeout`
    # ends it.  180 s > the probe's 120 s so the probe's own error record wins
    # when it can produce one.
    local rc=0
    if timeout --kill-after=10 180 \
           env CUDA_VISIBLE_DEVICES="$GPU" "$exe" \
           > "$OUTDIR/$name.jsonl" 2> "$OUTDIR/run_$name.log"; then
        set_status "$name" "ok"; set_reason "$name" ""
    else
        rc=$?
        if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
            set_status "$name" "timeout"
            set_reason "$name" "killed by timeout after 180 s (rc=$rc); the probe hung"
            note "$name: TIMED OUT after 180 s"
        else
            set_status "$name" "run_failed"
            set_reason "$name" "rc=$rc $(tail -n 20 "$OUTDIR/run_$name.log" | tr '\n' ' ')"
            note "$name: RUN FAILED rc=$rc (see $OUTDIR/run_$name.log)"
        fi
    fi
}

build_and_run "probe_dispatch_floor"    "probe_dispatch_floor.cu"    ""          ""
build_and_run "probe_gridsync_cost"     "probe_gridsync_cost.cu"     "-rdc=true" "-lcudadevrt"
build_and_run "probe_conditional_graph" "probe_conditional_graph.cu" "-rdc=true" "-lcudadevrt"

# --------------------------------------------------------------------------
# Probe 4: the L2 / DRAM width sweep needs the production binary.
# --------------------------------------------------------------------------
if skipped "l2_width"; then
    set_status "l2_width" "skipped"; set_reason "l2_width" "--skip l2_width"
    note "l2_width: skipped by request"
elif [ -z "$BIN" ]; then
    set_status "l2_width" "skipped"
    set_reason "l2_width" "no --bin given; the width sweep profiles the production binary"
    note "l2_width: skipped (no --bin)"
else
    set_cmds "l2_width" "$TOOLS/probe_l2_width.sh --bin $BIN --outdir $OUTDIR/l2 --gpu $GPU
python3 $TOOLS/parse_ncu_l2.py --outdir $OUTDIR/l2 --json-out $OUTDIR/probe_l2_width.json"
    note "l2_width: sweeping widths 1/8/22/64 (this is the long one)"
    if bash "$TOOLS/probe_l2_width.sh" --bin "$BIN" --outdir "$OUTDIR/l2" --gpu "$GPU" \
            > "$OUTDIR/run_l2_width.log" 2>&1 \
       && python3 "$TOOLS/parse_ncu_l2.py" --outdir "$OUTDIR/l2" \
            --json-out "$OUTDIR/probe_l2_width.json" > /dev/null 2>>"$OUTDIR/run_l2_width.log"; then
        set_status "l2_width" "ok"; set_reason "l2_width" ""
    else
        set_status "l2_width" "run_failed"
        set_reason "l2_width" "$(tail -n 20 "$OUTDIR/run_l2_width.log" | tr '\n' ' ')"
        note "l2_width: FAILED (see $OUTDIR/run_l2_width.log)"
    fi
fi

# --------------------------------------------------------------------------
# Probe 5: the offline replay needs an M64 trace with SPTELEM receipts.
# --------------------------------------------------------------------------
if skipped "scheduler_replay"; then
    set_status "scheduler_replay" "skipped"; set_reason "scheduler_replay" "--skip scheduler_replay"
    note "scheduler_replay: skipped by request"
elif [ -z "$SPTELEM" ]; then
    set_status "scheduler_replay" "skipped"
    set_reason "scheduler_replay" "no --sptelem trace given; run M64 with RASBERY_STATEPOINT_TELEMETRY=1 first"
    note "scheduler_replay: skipped (no --sptelem)"
else
    set_cmds "scheduler_replay" \
        "python3 $TOOLS/scheduler_trace_replay.py --sptelem $SPTELEM --width 64 --json-out $OUTDIR/probe_scheduler_replay.json"
    note "scheduler_replay: replaying five policies"
    if python3 "$TOOLS/scheduler_trace_replay.py" --sptelem "$SPTELEM" --width 64 \
            --json-out "$OUTDIR/probe_scheduler_replay.json" \
            > /dev/null 2> "$OUTDIR/run_scheduler_replay.log"; then
        set_status "scheduler_replay" "ok"; set_reason "scheduler_replay" ""
    else
        set_status "scheduler_replay" "run_failed"
        set_reason "scheduler_replay" "$(tail -n 20 "$OUTDIR/run_scheduler_replay.log" | tr '\n' ' ')"
        note "scheduler_replay: FAILED (see $OUTDIR/run_scheduler_replay.log)"
    fi
fi

# --------------------------------------------------------------------------
# Consolidate.  The receipt schema lives here and in tools/test_w0_spikes.py.
# --------------------------------------------------------------------------
NVCC_VERSION="$(command -v nvcc >/dev/null 2>&1 && nvcc --version 2>/dev/null | tail -n 1 || echo 'nvcc not found')"
DRIVER_VERSION="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo 'unknown')"
GPU_NAME="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo 'unknown')"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

python3 - "$OUTDIR" "$ARCH" "$GPU" "$RECEIPT" "$NVCC_VERSION" "$DRIVER_VERSION" "$GPU_NAME" "$GIT_SHA" <<'PY'
import datetime, json, os, platform, sys

outdir, arch, gpu, receipt_path, nvcc, driver, gpu_name, git_sha = sys.argv[1:9]

# Gate constants.  These are the numbers the whole W0 exists to compare against;
# they are duplicated in the probes that own them and asserted equal by
# tools/test_w0_spikes.py.
C_BARRIER_GATE_US = 0.384
DRAM_GATE_PCT = 60.0
IDLE_REDUCTION_GATE_PCT = 20.0

PROBES = ["probe_dispatch_floor", "probe_gridsync_cost", "probe_conditional_graph",
          "l2_width", "scheduler_replay"]


def read(name, default=""):
    p = os.path.join(outdir, name)
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return default


def jsonl(name):
    out = []
    p = os.path.join(outdir, name)
    if not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_json(name):
    p = os.path.join(outdir, name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def summary_of(records):
    for r in reversed(records):
        if r.get("record") == "summary":
            return r
    return None


probes = {}
for name in PROBES:
    entry = {
        "status": read(f"status_{name}.txt", "skipped") or "skipped",
        "reason": read(f"reason_{name}.txt", ""),
        "commands": read(f"cmd_{name}.txt", ""),
    }
    if name.startswith("probe_"):
        recs = jsonl(f"{name}.jsonl")
        entry["records"] = recs
        entry["summary"] = summary_of(recs)
    elif name == "l2_width":
        entry["receipt"] = load_json("probe_l2_width.json")
    elif name == "scheduler_replay":
        entry["receipt"] = load_json("probe_scheduler_replay.json")
    probes[name] = entry


def verdict(measured, threshold, lower_is_better=True):
    if measured is None:
        return "UNKNOWN"
    if lower_is_better:
        return "GO" if measured <= threshold else "NO-GO"
    return "GO" if measured >= threshold else "NO-GO"


disp = (probes["probe_dispatch_floor"].get("summary") or {})
grid = (probes["probe_gridsync_cost"].get("summary") or {})
cond = (probes["probe_conditional_graph"].get("summary") or {})
l2 = probes["l2_width"].get("receipt") or {}
sched = probes["scheduler_replay"].get("receipt") or {}

c_dispatch = disp.get("c_dispatch_us_measured")
c_barrier = grid.get("c_barrier_us_gate_value")
if isinstance(c_barrier, (int, float)) and c_barrier < 0:
    c_barrier = None

l2_gate = l2.get("gate") or {}
l2_shipping = l2_gate.get("shipping_width")
l2_measured = None
if l2_shipping is not None:
    w = (l2.get("by_width") or {}).get(str(l2_shipping)) or {}
    allk = (w.get("all_kernels") or {}).get("dram_pct_of_peak") or {}
    l2_measured = allk.get("mean")

sched_gate = sched.get("gate") or {}
idle_candidates = [v for v in (sched_gate.get("p4_idle_reduction_pct"),
                               sched_gate.get("p5_idle_reduction_pct"))
                   if isinstance(v, (int, float))]
idle_best = max(idle_candidates) if idle_candidates else None

gates = {
    "c_dispatch_us": {
        "gates_wave": "W10 / Phase 5 Stage 1b (persistent CMFD)",
        "measured": c_dispatch,
        "inferred_prior": disp.get("c_dispatch_us_inferred"),
        "note": "no pass/fail of its own; it is the numerator of the persistent-kernel gain",
    },
    "c_barrier_us": {
        "gates_wave": "W10 / Phase 5 Stage 1b (persistent CMFD)",
        "threshold": C_BARRIER_GATE_US,
        "measured": c_barrier,
        "verdict": verdict(c_barrier, C_BARRIER_GATE_US),
    },
    "conditional_scheduler": {
        "gates_wave": "W3 (CUDA conditional scheduler backend)",
        "while_ok": cond.get("while_ok"),
        "switch_ok": cond.get("switch_ok"),
        "nested_if_fallback": cond.get("nested_if_fallback"),
        "coop_in_conditional": cond.get("coop_in_conditional"),
        "verdict": ("GO" if cond.get("while_ok") and
                    (cond.get("switch_ok") or cond.get("nested_if_fallback"))
                    else "UNKNOWN" if not cond else "NO-GO"),
    },
    "dram_pct_of_peak": {
        "gates_wave": "W2 (resident CMFD/Nodal) and Rev.7 5.8 heavy-phase overlap",
        "threshold": DRAM_GATE_PCT,
        "measured": l2_measured,
        "measurement_tool": l2.get("tool"),
        "verdict": ("PROXY" if l2.get("tool") not in (None, "ncu")
                    else verdict(l2_measured, DRAM_GATE_PCT)),
    },
    "idle_reduction_pct": {
        "gates_wave": "W1 (trace replay) and W3/W4 (scheduler adoption)",
        "threshold": IDLE_REDUCTION_GATE_PCT,
        "measured": idle_best,
        "verdict": verdict(idle_best, IDLE_REDUCTION_GATE_PCT, lower_is_better=False),
    },
}

receipt = {
    "schema": "w0_spikes",
    "schema_version": 1,
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "host": platform.node(),
    "git_sha": git_sha,
    "arch": arch,
    "gpu_index": gpu,
    "gpu_name": gpu_name,
    "nvcc": nvcc,
    "driver": driver,
    "outdir": outdir,
    "probes": probes,
    "gates": gates,
}

text = json.dumps(receipt, indent=2)
print(text)
with open(receipt_path, "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
print(f"run_w0_spikes: receipt written to {receipt_path}", file=sys.stderr)
PY
