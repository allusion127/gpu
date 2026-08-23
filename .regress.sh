#!/usr/bin/env bash
# RASBERY regression gate (rewritten 2026-08-10, IISC port Phase 4).
#
# Replaces the dead upstream script, which hard-coded /home/shj/Rasbery and
# referenced test/9-1_IISC/{cases,cross_sections}/ -- directories that do not
# exist in this fork. Two further upstream assumptions are gone:
#
#   * byte-identical h5diff is unattainable here. The CUDA backend recombines
#     floating point differently, and our BiCGSTAB residual definition changed
#     (r2 = ||s - w t||). The gate is a pcm tolerance instead.
#   * IISC has no on/off switch; it is enabled by library content alone. Tier 0
#     therefore asserts that the fitted terms are actually present, so a silently
#     empty library fails loudly instead of scoring as "traditional".
#
# Usage:
#   .regress.sh [--nobuild] [--tier 0|1|2|all]
#
# Environment:
#   RASBERY_BIN   solver binary        (default ./build/RASBERY)
#   PYTHON        interpreter with h5py+numpy+matplotlib
#   IISC_JOBS     run_iisc.py case parallelism (default 6)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${RASBERY_BIN:-$ROOT/build/RASBERY}"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-/tmp/ras_regress}"
TIER=all
NOBUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --nobuild) NOBUILD=1 ;;
    --tier)    TIER="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$OUT"
fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; }

# RASBERY overrides OMP_NUM_THREADS and re-execs itself; this is the real knob.
export RASBERY_OMP_THREADS="${RASBERY_OMP_THREADS:-8}"
export MPLBACKEND=Agg

if [ "$NOBUILD" -eq 0 ]; then
  echo "=== build ==="
  if ! cmake --build "$ROOT/build" -j > "$OUT/build.log" 2>&1; then
    echo "BUILD FAILED"; tail -20 "$OUT/build.log"; exit 1
  fi
  tail -1 "$OUT/build.log"
fi
[ -x "$BIN" ] || { echo "solver binary not found: $BIN"; exit 1; }

# ---------------------------------------------------------------------------
# Tier 0 -- library integrity (R6). Cheap, and it is the only thing standing
# between "IISC is off" and a silently wrong pass.
# ---------------------------------------------------------------------------
tier0() {
  echo "=== T0: IISC library integrity ==="
  local case_dir="$ROOT/test/9-1_IISC/iSMR_AIC/CASE1_WH17_6pct_HIGA"
  local deck="$case_dir/input/chiffon/iisc_rhst.json"
  if [ ! -f "$deck" ]; then
    note SKIP "IISC decks absent (test/CrossSections/5_IISC not staged)"
    return
  fi
  ( cd "$(dirname "$deck")" && "$BIN" --chiffoni "./$(basename "$deck")" \
      --chiffono "$OUT/t0_iisc.h5" ) > "$OUT/t0_build.log" 2>&1
  if grep -q "CHIFFON\]\[WARNING\]" "$OUT/t0_build.log"; then
    note FAIL "library build emitted degeneracy warnings"
    grep "CHIFFON\]\[WARNING\]" "$OUT/t0_build.log" | head -3 | sed 's/^/        /'
    fail=1
  else
    note PASS "library build clean (no pden/pairing warnings)"
  fi
  "$PYTHON" - "$OUT/t0_iisc.h5" <<'PY'
import sys, h5py
bad = []
with h5py.File(sys.argv[1]) as f:
    for k, g in f["Models"].items():
        nt = int(g["spectral_history"]["num_terms"][()]) if "spectral_history" in g else 0
        ne = int(g["rod_depletion_delt"]["num_entries"][()]) if "rod_depletion_delt" in g else 0
        if nt <= 0 or ne <= 0:
            bad.append(f"{k}: shct_terms={nt} rdpl_entries={ne}")
if bad:
    print("  FAIL   IISC terms missing -- the library would score as traditional")
    for b in bad[:5]:
        print("        " + b)
    sys.exit(1)
print("  PASS   IISC terms present (SHCT > 0 and RDPL > 0)")
PY
  [ $? -ne 0 ] && fail=1
}

# ---------------------------------------------------------------------------
# Tier 1 -- IISC accuracy. Reference values are the methodology document, not
# the committed AIC/B4C summary.csv, which predates the ctype surface split.
# ---------------------------------------------------------------------------
tier1() {
  echo "=== T1: IISC accuracy (3 x CASE1 vs methodology doc) ==="
  local harness="$ROOT/test/9-1_IISC/run_iisc.py"
  if [ ! -f "$harness" ] || [ ! -d "$ROOT/test/CrossSections/5_IISC" ]; then
    note SKIP "IISC battery assets absent"
    return
  fi
  ( cd "$ROOT/test/9-1_IISC" && "$PYTHON" run_iisc.py --rasbery "$BIN" \
      --jobs "${IISC_JOBS:-6}" \
      --case PWR/CASE1_CE16_2pct_Plain \
             iSMR_AIC/CASE1_WH17_6pct_HIGA \
             iSMR_B4C/CASE1_WH17_6pct_HIGA ) > "$OUT/t1.log" 2>&1
  "$PYTHON" - "$ROOT/test/9-1_IISC" <<'PY'
import csv, os, sys
root = sys.argv[1]
# (family, case, method) -> (expected mean k_inf RMS pcm, tolerance)
EXPECT = {
    ("PWR", "CASE1_CE16_2pct_Plain", "traditional"): (275.5, 5.0),
    ("PWR", "CASE1_CE16_2pct_Plain", "iisc_rhst"):   (63.2, 5.0),
    ("iSMR_AIC", "CASE1_WH17_6pct_HIGA", "traditional"): (478.5, 5.0),
    ("iSMR_AIC", "CASE1_WH17_6pct_HIGA", "iisc_rhst"):   (93.1, 5.0),
    ("iSMR_B4C", "CASE1_WH17_6pct_HIGA", "traditional"): (718.9, 5.0),
    ("iSMR_B4C", "CASE1_WH17_6pct_HIGA", "iisc_rhst"):   (103.7, 5.0),
}
# The decisive IISC litmus: M10 rod withdrawal must collapse to ~123 pcm.
M10 = {("iSMR_AIC", "CASE1_WH17_6pct_HIGA", "iisc_rhst"): (123.0, 10.0)}

def rows(fam, case):
    out = []
    for rel in ("summary/summary.csv", "branch_closure/summary.csv"):
        p = os.path.join(root, fam, case, rel)
        if os.path.exists(p):
            out += list(csv.DictReader(open(p)))
    return out

bad = 0
for (fam, case, meth), (want, tol) in EXPECT.items():
    v = [float(r["keff_rms_pcm"]) for r in rows(fam, case)
         if r["method"] == meth and r["keff_rms_pcm"] not in ("", "nan")]
    got = sum(v) / len(v) if v else float("nan")
    ok = abs(got - want) <= tol
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL':6s} {fam}/{case} {meth:12s} "
          f"got={got:8.2f} want={want:8.2f} +-{tol}")
for (fam, case, meth), (want, tol) in M10.items():
    v = [float(r["keff_rms_pcm"]) for r in rows(fam, case)
         if r["method"] == meth and r["scenario"].startswith("branch_r_rodout")]
    got = sum(v) / len(v) if v else float("nan")
    ok = abs(got - want) <= tol
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL':6s} {fam}/{case} M10 withdrawal "
          f"got={got:8.2f} want={want:8.2f} +-{tol}")
sys.exit(1 if bad else 0)
PY
  [ $? -ne 0 ] && fail=1
}

# ---------------------------------------------------------------------------
# Tier 2 -- core depletion + rod search against the stored baseline.
# Populate the baseline with: BASELINE=<dir> .regress.sh --tier 2 --record
# ---------------------------------------------------------------------------
tier2() {
  echo "=== T2: core cycles CY01-CY04 (rod search) ==="
  local base="${BASELINE:-$ROOT/test/7_i-SMR_Validation/.regress_baseline}"
  local deck_dir="$ROOT/test/7_i-SMR_Validation"
  if [ ! -d "$base" ]; then
    note SKIP "no baseline at $base (set BASELINE= to record one)"
    return
  fi
  local w="$OUT/cy"; rm -rf "$w"; mkdir -p "$w"; cp "$deck_dir"/i-SMR_CY0*.json "$w/" 2>/dev/null
  cp "${IISC_CORE_LIB:-$ROOT/test/CrossSections/i-SMR_Validation.h5}" "$w/i-SMR_Validation.h5" 2>/dev/null || {
    note SKIP "core library not found (set IISC_CORE_LIB=)"; return; }
  for c in 01 02 03 04; do
    ( cd "$w" && "$BIN" --rasi "i-SMR_CY$c.json" --raso "out_CY$c.h5" ) > "$w/log_$c.txt" 2>&1
  done
  "$PYTHON" - "$base" "$w" <<'PY'
import sys, os, h5py, numpy as np
base, new = sys.argv[1], sys.argv[2]
TOL_K, TOL_ROD = 1.0, 1e-3   # pcm, rod-step units
bad = 0
for c in ("01", "02", "03", "04"):
    a, b = os.path.join(base, f"out_CY{c}.h5"), os.path.join(new, f"out_CY{c}.h5")
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"  SKIP   CY{c} (missing output)"); continue
    with h5py.File(a) as fa, h5py.File(b) as fb:
        ka, kb = np.array(fa["summary"]["keff"]), np.array(fb["summary"]["keff"])
        ra, rb = np.array(fa["summary"]["rod_step"]), np.array(fb["summary"]["rod_step"])
        n = min(len(ka), len(kb))
        dk = np.abs(kb[:n] - ka[:n]) * 1e5
        dr = np.abs(rb[:n] - ra[:n])
    ok = dk.max() <= TOL_K and dr.max() <= TOL_ROD
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL':6s} CY{c} max|dk|={dk.max():8.4f} pcm "
          f"max|d rod|={dr.max():.6f}")
sys.exit(1 if bad else 0)
PY
  [ $? -ne 0 ] && fail=1
}

# ---------------------------------------------------------------------------
# Tier 3 -- whole-output golden h5diff on the CPU arm (adopted from the rev00
# tree's regression, where it caught PPR/isotopic regressions Tier 2's
# keff/rod-only check passes silently).  CPU-only (RASBERY_GPU=0), so the
# byte-identity objection to h5diff does not apply; /summary/reactivity is
# excluded because its OpenMP reduction order is thread-count dependent.
# Record with: GOLDEN=<dir> .regress.sh --tier 3 (missing golden = record)
# ---------------------------------------------------------------------------
tier3() {
  echo "=== T3: CPU golden h5diff (whole output) ==="
  command -v h5diff >/dev/null || { note SKIP "h5diff not on PATH"; return; }
  local golden="${GOLDEN:-$ROOT/test/7_i-SMR_Validation/.golden_cpu}"
  local deck_dir="$ROOT/test/7_i-SMR_Validation"
  local w="$OUT/golden"; rm -rf "$w"; mkdir -p "$w"
  cp "$deck_dir"/i-SMR_CY01.json "$w/" 2>/dev/null || { note SKIP "deck missing"; return; }
  cp "${IISC_CORE_LIB:-$ROOT/test/CrossSections/i-SMR_Validation.h5}" "$w/i-SMR_Validation.h5" 2>/dev/null || {
    note SKIP "core library not found (set IISC_CORE_LIB=)"; return; }
  ( cd "$w" && RASBERY_GPU=0 "$BIN" --rasi i-SMR_CY01.json --raso out_CY01.h5 ) > "$w/log.txt" 2>&1
  if [ ! -f "$golden/out_CY01.h5" ]; then
    mkdir -p "$golden"; cp "$w/out_CY01.h5" "$golden/"
    note RECORD "golden recorded at $golden"; return
  fi
  if h5diff -q -p 1e-9 --exclude-path /summary/reactivity \
        "$golden/out_CY01.h5" "$w/out_CY01.h5" > "$w/h5diff.txt" 2>&1; then
    note PASS "CY01 whole-output identical to golden (p<=1e-9)"
  else
    note FAIL "CY01 differs from golden -- see $w/h5diff.txt"
    fail=1
  fi
}

case "$TIER" in
  0) tier0 ;;
  1) tier1 ;;
  2) tier2 ;;
  3) tier3 ;;
  all) tier0; tier1; tier2; tier3 ;;
  *) echo "unknown tier: $TIER" >&2; exit 2 ;;
esac

echo
if [ "$fail" -eq 0 ]; then echo "REGRESSION: PASS"; else echo "REGRESSION: FAIL"; fi
exit "$fail"
