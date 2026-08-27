#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# W0 decision spike 4/5 -- L2 hit rate and DRAM throughput at batch widths
# 1 / 8 / 22 / 64.
#
# WHAT THIS DECIDES.  The W2 exit gate and, through it, whether Rev.7 5.8's
# "first production version does not run heavy phases concurrently" stays a
# restriction or is lifted.  That section is explicit: heavy-phase overlap is
# allowed ONLY after NCU confirms memory-bandwidth and register headroom.  This
# is the NCU run that confirms it or does not.
#
# It also settles a question the FP32 campaign left open.  The CMFD kernels were
# measured at 0.13 FLOP/B -- bandwidth bound, not compute bound.  If L2 hit rate
# collapses between width 22 and width 64 then the batched arena has outgrown
# L2, and the slot-compaction work in Phase 5 Stage 1a is buying less than the
# active-width arithmetic predicts.  The width sweep is the only way to see it.
#
# GATE (printed by tools/parse_ncu_l2.py):
#     dram__throughput.avg.pct_of_peak_sustained_elapsed <= 60 % of peak
# at the width the campaign intends to ship.  Above that, heavy-phase overlap
# has no headroom to overlap into and Rev.7 5.8 stays as written.
#
# WHAT IS HELD FIXED.  Only the width varies:
#   * --batch-mode W for EVERY width, W=1 included.  A bare single run takes a
#     different code path (Driver.h:512 -- Anderson defaults differ by mode),
#     and a sweep whose W=1 point came from another path is not a sweep.
#   * RASBERY_XE_ANDERSON=0 everywhere, so the mode-dependent adoption default
#     cannot move the trajectory between widths.
#   * RASBERY_STATEPOINT_TELEMETRY unset: SPTELEM formats per statepoint and
#     this run is being profiled.
#
# HOW LONG THE RUN IS.  Under ncu the run is bounded by `-c 40`: profiling stops
# after 40 matching kernel launches, so a full deck is fine and the wall is
# dominated by ncu replay, not by the deck.  The DEFAULT deck is nonetheless the
# single-step one (test/7_i-SMR_Validation/cy02_step1.json) so the fallback path
# -- which has no such bound -- also terminates.  The fallback additionally
# enforces --timeout.
#
# FALLBACK.  If `ncu` is absent (or refuses for lack of permission, which on a
# shared box is the common case: NVreg_RestrictProfilingToAdminUsers), the
# script samples `nvidia-smi dmon` instead and says so.  dmon gives SM and
# memory-controller utilisation, which is a PROXY: it is a duty-cycle percent,
# not a fraction of peak sustained bandwidth, and the receipt labels it as such
# so nobody quotes it against the 60% gate.
#
# USAGE
#   tools/probe_l2_width.sh --bin /path/to/RASBERY \
#       [--deck test/7_i-SMR_Validation/cy02_step1.json] \
#       [--outdir /tmp/w0_l2] [--widths "1 8 22 64"] [--gpu 0] [--timeout 600]
#
#   python3 tools/parse_ncu_l2.py --outdir /tmp/w0_l2
#
# Runtime: 10-40 min under ncu (replay is slow and there are four widths);
# a few minutes on the dmon fallback.
# ---------------------------------------------------------------------------
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN=""
DECK="$REPO_ROOT/test/7_i-SMR_Validation/cy02_step1.json"
OUTDIR="/tmp/w0_l2"
WIDTHS="1 8 22 64"
GPU="0"
TIMEOUT="600"

# The W2 gate.  Kept here as well as in the parser so the value is visible in
# the run log, not only in the receipt.
DRAM_GATE_PCT="60"

NCU_METRICS="lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed"

# THE KERNEL FILTER, AND WHY IT IS NOT `regex:cmfd|nodal`.
#
# The obvious filter misses almost everything that matters.  Measured against
# the __global__ names actually in this tree, `cmfd|nodal` matches exactly seven
# kernels -- cmfd_assemble_operator_2g, cmfd_negative_scan, cmfd_src_build,
# cmfd_sweep_begin, cmfd_sweep_end, cmfd_updls, cmfd_wiel_* -- and NONE of the
# BiCGSTAB inner kernels, which are the bandwidth-bound ones the 0.13 FLOP/B
# finding was about: matvec_two_group, colored_block_sweep, update_solution,
# update_s_jacobi, prepare_p_jacobi, reduce_dot*, reduce_norm*.  It also misses
# every Nodal kernel, because they are spelled kNodalEven / kNodalMat / ... with
# a capital N and ncu's regex: filter is case-sensitive.
#
# A sweep run with `cmfd|nodal` would come back green on kernels that are not
# the bottleneck.  The list below is the real kernel set; the _f32 variants are
# picked up by the same stems.  Override with --kernel-regex if the kernel names
# move.
NCU_KERNEL_REGEX='regex:cmfd|[Nn]odal|matvec_two_group|colored_block_sweep|update_solution|update_s_jacobi|prepare_p_jacobi|reduce_dot|reduce_norm|begin_outer|accumulate_iteration|initialize_solver_state|store_reference_norm|refresh_operator_mirror|finalize_status|kernelFlatXs|kernelXsRecon'

# `-c 40` bounds profiling to the first 40 matching launches.  One CMFD sweep is
# 95 graph nodes (CudaBICGBackend.cu:2629), so 40 is roughly 0.4 of a sweep:
# enough for per-class hit-rate and bandwidth ratios, thin for per-kernel means.
# Raise with --count when a tighter per-kernel estimate is wanted.
NCU_LAUNCH_COUNT="40"

usage() {
    sed -n '2,60p' "${BASH_SOURCE[0]}"
    exit "${1:-1}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --bin)     BIN="$2"; shift 2 ;;
        --deck)    DECK="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --widths)  WIDTHS="$2"; shift 2 ;;
        --gpu)     GPU="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --count)   NCU_LAUNCH_COUNT="$2"; shift 2 ;;
        --kernel-regex) NCU_KERNEL_REGEX="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "probe_l2_width: unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [ -z "$BIN" ]; then
    echo "probe_l2_width: --bin <RASBERY binary> is required" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "probe_l2_width: not executable: $BIN" >&2
    exit 2
fi
if [ ! -f "$DECK" ]; then
    echo "probe_l2_width: deck not found: $DECK" >&2
    exit 2
fi

mkdir -p "$OUTDIR"
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
DECK="$(cd "$(dirname "$DECK")" && pwd)/$(basename "$DECK")"
OUTDIR="$(cd "$OUTDIR" && pwd)"

# Which measurement path is available?
TOOL="none"
if command -v ncu >/dev/null 2>&1; then
    TOOL="ncu"
elif command -v nvidia-smi >/dev/null 2>&1; then
    TOOL="nvidia-smi-dmon"
fi

if [ "$TOOL" = "none" ]; then
    echo "probe_l2_width: neither ncu nor nvidia-smi is on PATH; nothing to measure" >&2
    exit 4
fi
echo "probe_l2_width: measurement path = $TOOL" >&2

# Production GPU configuration, copied from tools/run_single_gpu_batch.py
# DEFAULT_ENV.  A profile of a different configuration answers a different
# question.
export RASBERY_GPU=1
export RASBERY_GPU_RB_SWEEPS=4
export RASBERY_GPU_XSRECON=1
export RASBERY_GPU_FLATXS=1
export RASBERY_GPU_NODAL=1
export RASBERY_GPU_NODAL_FULL=1
export RASBERY_GPU_CMFD_SWEEP=1
export RASBERY_PPR_MODE=master
export RASBERY_PC_MODE=decart
export RASBERY_BATCH_WAIT_US=0
export RASBERY_NODAL_BATCH_WAIT_US=0
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
# Uniformity across the width sweep -- see the header.
export RASBERY_XE_ANDERSON=0
unset RASBERY_STATEPOINT_TELEMETRY
export CUDA_VISIBLE_DEVICES="$GPU"

DECK_BASE="$(basename "$DECK")"
DECK_DIR="$(dirname "$DECK")"

run_one_width() {
    local w="$1"
    local work="$OUTDIR/w$w"
    rm -rf "$work"
    mkdir -p "$work"

    # A deck is read relative to its own directory (see .regress.sh:196), so the
    # copies go next to the original and the run happens there.
    local -a rasi=()
    local -a raso=()
    local i=0
    while [ "$i" -lt "$w" ]; do
        cp "$DECK" "$DECK_DIR/.w0probe_${w}_${i}.json"
        rasi+=(".w0probe_${w}_${i}.json")
        raso+=("$work/out_${i}.h5")
        i=$((i + 1))
    done

    local rc=0
    if [ "$TOOL" = "ncu" ]; then
        ( cd "$DECK_DIR" && \
          ncu --target-processes all \
              --metrics "$NCU_METRICS" \
              -k "$NCU_KERNEL_REGEX" \
              -c "$NCU_LAUNCH_COUNT" \
              --csv \
              -- "$BIN" --rasi "${rasi[@]}" --raso "${raso[@]}" --batch-mode "$w" \
        ) > "$OUTDIR/ncu_w${w}.csv" 2> "$OUTDIR/ncu_w${w}.log"
        rc=$?
    else
        # Sample while the run proceeds.  `-s um`: u = SM / memory-controller
        # utilisation, m = framebuffer usage.
        nvidia-smi dmon -i "$GPU" -s um -d 1 -c "$TIMEOUT" \
            > "$OUTDIR/dmon_w${w}.txt" 2>"$OUTDIR/dmon_w${w}.log" &
        local dmon_pid=$!
        ( cd "$DECK_DIR" && \
          timeout "$TIMEOUT" "$BIN" --rasi "${rasi[@]}" --raso "${raso[@]}" \
                  --batch-mode "$w" \
        ) > "$OUTDIR/run_w${w}.log" 2>&1
        rc=$?
        kill "$dmon_pid" 2>/dev/null
        wait "$dmon_pid" 2>/dev/null
    fi

    rm -f "$DECK_DIR"/.w0probe_${w}_*.json
    echo "probe_l2_width: width $w finished rc=$rc" >&2
    return 0
}

for W in $WIDTHS; do
    echo "probe_l2_width: width $W ..." >&2
    run_one_width "$W"
done

# Metadata the parser needs and cannot infer.
{
    printf '{"probe":"l2_width","tool":"%s","widths":[' "$TOOL"
    first=1
    for W in $WIDTHS; do
        if [ "$first" -eq 1 ]; then first=0; else printf ','; fi
        printf '%s' "$W"
    done
    printf '],"deck":"%s","bin":"%s","gpu":"%s",' "$DECK_BASE" "$(basename "$BIN")" "$GPU"
    printf '"metrics":"%s","kernel_regex":"%s","launch_count":%s,' \
        "$NCU_METRICS" "$NCU_KERNEL_REGEX" "$NCU_LAUNCH_COUNT"
    printf '"dram_gate_pct_of_peak":%s,' "$DRAM_GATE_PCT"
    printf '"anderson":0,"sptelem":false,"batch_mode_forced":true}\n'
} > "$OUTDIR/l2_width_meta.json"

echo "probe_l2_width: W2 gate = dram <= ${DRAM_GATE_PCT}% of peak sustained" >&2
echo "probe_l2_width: artifacts in $OUTDIR" >&2
echo "probe_l2_width: now run  python3 tools/parse_ncu_l2.py --outdir $OUTDIR" >&2
