#!/usr/bin/env python3
"""The host-wait knobs must be applied BEFORE the things that make them moot.

WHY THIS TEST EXISTS.  On the 238 host a batch of 8 processes x
RASBERY_OMP_THREADS=8 on 24 CPUs shows ~79 % host CPU while the XFER ledger
says ~92 % of process time is inside cudaStreamSynchronize.  Both numbers are
real and they cannot both be work: CUDA's default schedule (Auto) SPINS while
the context count is below the core count, so a thread "busy" in a synchronise
is polling a fence with a core, and eight processes' pollers contend with the
OpenMP teams that have real work.  RASBERY_CUDA_SYNC_MODE and RASBERY_OMP_WAIT
exist to take the other arms and measure the trade.

THE DEFECT CLASS THIS CLOSES IS ORDER, NOT SPELLING -- which is what `4415254`
had to teach three other contract tests.  Both knobs are one-shot process-wide
settings with a deadline:

  * cudaSetDeviceFlags is honoured only while the process has NO CUDA context.
    Nothing in this tree calls cudaSetDevice, so the primary context is created
    lazily by whichever backend's first runtime call wins during Drive().  A
    call moved below any of them returns cudaErrorSetOnActiveProcess -- and
    STILL PRINTS A RECEIPT.  The campaign row would then say `blocking` while
    the process spun.
  * libgomp reads OMP_WAIT_POLICY / GOMP_SPINCOUNT in a library constructor,
    before main() is entered.  Exporting them anywhere except before the
    OpenMP re-exec changes an environment variable and nothing else.

So every check here is about WHERE, and every one of them has a negative
control: the same checker is run against a mutated copy of the source that
makes exactly that mistake, and the check must fail on it.  A checker that
cannot fail is not a gate.

Run: python tools/test_cuda_sched_contract.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def strip_comments(text: str) -> str:
    """Comments mention the very names this test orders on; drop them."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ===========================================================================
# 0. The files exist and are wired the way every other CUDA arm is
# ===========================================================================
HEADER = ROOT / "src" / "CudaHostSchedule.h"
CU = ROOT / "src" / "CudaHostSchedule.cu"
STUB = ROOT / "src" / "CudaHostScheduleStub.cpp"
MAIN = ROOT / "src" / "main.cpp"
CMAKE = ROOT / "CMakeLists.txt"
RUNNER = ROOT / "tools" / "run_multi_gpu_batch.py"

for path in (HEADER, CU, STUB, MAIN, CMAKE, RUNNER):
    check(path.is_file(), f"missing {path.relative_to(ROOT).as_posix()}")
if failures:
    for problem in failures:
        print(f"[RASBERY][CUDA_SCHED_CONTRACT][FAIL] {problem}", file=sys.stderr)
    raise SystemExit(1)

header_text = HEADER.read_text(encoding="utf-8")
cu_text = CU.read_text(encoding="utf-8")
stub_text = STUB.read_text(encoding="utf-8")
main_text = MAIN.read_text(encoding="utf-8")
cmake_text = CMAKE.read_text(encoding="utf-8")
runner_text = RUNNER.read_text(encoding="utf-8")

check("CudaHostScheduleReceipt ApplyCudaHostSchedule();" in header_text,
      "src/CudaHostSchedule.h must declare ApplyCudaHostSchedule()")
check("ApplyCudaHostSchedule()" in cu_text and "ApplyCudaHostSchedule()" in stub_text,
      "the CUDA arm and the CPU-only stub must define the SAME symbol, so main() "
      "never needs an #ifdef -- the rule every other Cuda*Backend pair follows")
check('src/CudaHostScheduleStub.cpp"' in cmake_text
      and 'src/CudaHostSchedule.cu"' in cmake_text,
      "CMakeLists.txt must REMOVE_ITEM the stub and APPEND the .cu in the "
      "RASBERY_ENABLE_CUDA arm; src/*.cpp is globbed, so a stub that is not "
      "removed is a duplicate definition and a .cu that is not appended is never "
      "compiled at all")
stub_removed = cmake_text.find('src/CudaHostScheduleStub.cpp"')
cu_added = cmake_text.find('src/CudaHostSchedule.cu"')
check(0 < stub_removed < cu_added,
      "the stub removal must come before the .cu append, i.e. be inside "
      "list(REMOVE_ITEM ...) rather than list(APPEND ...)")


# ===========================================================================
# 1. The CUDA arm: the flag is set before this TU can create a context
# ===========================================================================
#
# The strongest available guarantee for "before context creation" is that this
# translation unit creates NOTHING.  A cudaMalloc, a stream, even a
# cudaGetDeviceCount in this file would create the primary context on the line
# above the one that tries to configure it.

CONTEXT_CREATORS = (
    "cudaMalloc",
    "cudaMallocHost",
    "cudaSetDevice(",
    "cudaStreamCreate",
    "cudaGetDeviceCount",
    "cudaGetDeviceProperties",
    "cudaMemcpy",
    "<<<",
)


def problems_cu(text: str) -> list[str]:
    body = strip_comments(text)
    out: list[str] = []
    if body.count("cudaSetDeviceFlags(") != 1:
        out.append("cudaSetDeviceFlags must be called exactly once")
    for symbol in ("cudaDeviceScheduleAuto", "cudaDeviceScheduleSpin",
                   "cudaDeviceScheduleYield", "cudaDeviceScheduleBlockingSync"):
        if symbol not in body:
            out.append(
                f"{symbol} is never requested, so one of the four modes is unreachable")
    for word in ("auto", "spin", "yield", "blocking"):
        if f'"{word}"' not in body:
            out.append(f'RASBERY_CUDA_SYNC_MODE="{word}" is not parsed')
    for creator in CONTEXT_CREATORS:
        if creator in body:
            out.append(
                f"{creator} in this TU creates the primary context that "
                "cudaSetDeviceFlags has to precede")
    set_at = body.find("cudaSetDeviceFlags(")
    if re.search(r'getenv\("RASBERY_CUDA_SYNC_MODE"\)', body) is None:
        out.append("RASBERY_CUDA_SYNC_MODE is never read")
    # The unset default must be a GUARDED EARLY RETURN, not merely some return
    # that happens to sit above the call: the invalid-value arm returns there
    # too, and a check that accepted any of them would pass a source in which
    # the unset path falls straight through into the runtime.
    guard = re.search(r"if \(env == nullptr\b[^)]*\)\s*\{", body)
    if guard is None:
        out.append(
            "the unset default must be guarded by an explicit `env == nullptr` "
            "test. Without it there is no path that leaves ApplyCudaHostSchedule() "
            "without entering the CUDA runtime, and 'default = unchanged' is an "
            "assertion rather than a property")
    else:
        block_end = body.find("\n    }", guard.end())
        block = body[guard.end():block_end] if block_end > 0 else ""
        if "return receipt;" not in block or not (0 <= guard.start() < set_at):
            out.append(
                "the unset default must RETURN before cudaSetDeviceFlags: "
                "'unchanged behaviour' is only provable if the default path "
                "enters the CUDA runtime not at all")
    if "cudaGetErrorName(" not in body:
        out.append("the cudaSetDeviceFlags rc is not recorded by name")
    if "cudaGetDeviceFlags(" not in body:
        out.append(
            "`applied` must be READ BACK, not assumed: a call made after a "
            "context exists fails and the bits in force are the context's")
    return out


check(problems_cu(cu_text) == [], f"src/CudaHostSchedule.cu: {problems_cu(cu_text)}")

# --- negative controls for the .cu checks ---------------------------------
mutant = cu_text.replace("if (env == nullptr || *env == '\\0') {", "if (false) {", 1)
check(any("must be guarded" in p for p in problems_cu(mutant)),
      "negative control: neutering the unset test must be reported -- that guard "
      "is the whole of the 'the default calls nothing' claim")

mutant = cu_text.replace("        return receipt;\n    }",
                         "        (void) receipt;\n    }", 1)
check(any("must RETURN before" in p for p in problems_cu(mutant)),
      "negative control: a default path that falls through into "
      "cudaSetDeviceFlags must be reported. That is the mutation that turns "
      "'default = unchanged' into 'default = one more CUDA entry point'")

mutant = cu_text.replace("    unsigned int flags = cudaDeviceScheduleAuto;",
                         "    void* p = nullptr; cudaMalloc(&p, 1);\n"
                         "    unsigned int flags = cudaDeviceScheduleAuto;", 1)
check(any("creates the primary context" in p for p in problems_cu(mutant)),
      "negative control: an allocation in this TU must be reported -- it would "
      "create the context on the line above the one configuring it")

mutant = cu_text.replace("cudaDeviceScheduleBlockingSync", "cudaDeviceScheduleYield")
check(any("BlockingSync" in p for p in problems_cu(mutant)),
      "negative control: losing one of the four schedule symbols must be reported")

mutant = cu_text.replace("cudaGetDeviceFlags(&actual)", "cudaSuccess, (actual = flags)")
check(any("READ BACK" in p for p in problems_cu(mutant)),
      "negative control: assuming `applied` instead of querying it must be "
      "reported -- an arm that asked too late would otherwise tabulate as the "
      "arm it asked for")


# ===========================================================================
# 2. main.cpp: the call is the first thing after the re-exec
# ===========================================================================
#
# Tokens that stand for "the program has started doing things".  Every one of
# them can reach a CUDA runtime call, directly or through a Driver, and the
# ordering claim is worth nothing if none of them is actually present -- so the
# count is asserted too.

STARTED_DOING_THINGS = ("plog::init", "IO::", "Driver", "rasbery::gpu", "Importer")


def problems_main(text: str) -> list[str]:
    out: list[str] = []
    stripped = strip_comments(text)
    start = stripped.find("int main(int argc, char* argv[])")
    if start < 0:
        return ["main() not found"]
    body = stripped[start:]

    call = body.find("ApplyCudaHostSchedule()")
    if call < 0:
        return ["main() never calls ApplyCudaHostSchedule()"]

    prepare = body.find("rasberyPrepareOpenMPStartup(argv);")
    if not (0 <= prepare < call):
        out.append(
            "ApplyCudaHostSchedule() must run AFTER rasberyPrepareOpenMPStartup(): "
            "the OpenMP startup may execvp, and flags set in the image that is "
            "about to be replaced are set for a process that never runs CUDA")

    present = [t for t in STARTED_DOING_THINGS if t in body]
    if len(present) < 3:
        out.append(
            "the ordering check has no teeth: fewer than three of "
            f"{STARTED_DOING_THINGS} appear in main(), so the tokens no longer "
            "stand for 'the program has started'")
    for token in present:
        if body.find(token) < call:
            out.append(
                f"`{token}` appears in main() BEFORE ApplyCudaHostSchedule(). "
                "cudaSetDeviceFlags is honoured only while the process has no "
                "CUDA context; after one exists it fails and still prints a "
                "receipt, so the arm would be mislabelled rather than refused")

    # The receipt, and the keys a campaign table reads out of it.
    if "[RASBERY][CUDA][SCHED]" not in body:
        out.append("the [RASBERY][CUDA][SCHED] receipt is not printed")
    for key in ("mode", "requested", "applied", "rc",
                "omp_wait", "omp_wait_policy", "gomp_spincount"):
        if '\\"' + key + '\\"' not in body:
            out.append(f"the receipt does not report `{key}`")

    # The OMP hook: read before the exec, not after it.
    startup = stripped.find("void rasberyPrepareOpenMPStartup(")
    if startup < 0:
        out.append("rasberyPrepareOpenMPStartup() not found")
    else:
        startup_body = stripped[startup:start]
        wait = startup_body.find("rasberyResolvedOmpWait()")
        exec_at = startup_body.find("execvp(")
        if wait < 0:
            out.append(
                "RASBERY_OMP_WAIT is not resolved inside rasberyPrepareOpenMPStartup(): "
                "libgomp reads OMP_WAIT_POLICY in a constructor before main(), so "
                "the re-exec is the only hook that reaches the running runtime")
        elif not (0 <= exec_at and wait < exec_at):
            out.append(
                "RASBERY_OMP_WAIT is resolved AFTER the execvp, i.e. in an image "
                "whose libgomp has already read the old policy")
        if 'rasberySetEnvIfNeeded("OMP_WAIT_POLICY", "ACTIVE", true)' not in startup_body:
            out.append("the active arm must OVERWRITE OMP_WAIT_POLICY: DEFAULT_ENV "
                       "already exports PASSIVE, so a non-overwriting set is a no-op")
        if 'unsetenv("GOMP_SPINCOUNT")' not in startup_body:
            out.append("the active arm must drop GOMP_SPINCOUNT: libgomp's spin count "
                       "beats the policy word, and 0 means 'do not spin'")
        if 'rasberySetEnvIfNeeded("GOMP_SPINCOUNT", "0", true)' not in startup_body:
            out.append("the passive arm must overwrite GOMP_SPINCOUNT=0")
        # The default arm, byte for byte what it was before the knob existed.
        if ('rasberySetEnvIfNeeded("OMP_WAIT_POLICY", "PASSIVE");' not in startup_body
                or 'rasberySetEnvIfNeeded("GOMP_SPINCOUNT", "0");' not in startup_body):
            out.append(
                "the unset default must keep the ORIGINAL non-overwriting "
                "PASSIVE/0 pair. Overwriting there would silently take an "
                "operator's exported policy away from them -- a behaviour change "
                "in the arm that is supposed to be unchanged")
    return out


check(problems_main(main_text) == [], f"src/main.cpp: {problems_main(main_text)}")

# --- negative controls for the main.cpp checks ----------------------------
mutant = main_text.replace(
    "    const rasbery::CudaHostScheduleReceipt cuda_sched = "
    "rasbery::ApplyCudaHostSchedule();", "", 1)
check(any("never calls" in p for p in problems_main(mutant)),
      "negative control: removing the call must be reported")

mutant = main_text.replace(
    "    rasberyPrepareOpenMPStartup(argv);\n",
    "    rasberyPrepareOpenMPStartup(argv);\n    plog::init(plog::error, nullptr);\n", 1)
check(any("BEFORE ApplyCudaHostSchedule" in p for p in problems_main(mutant)),
      "negative control: hoisting ANY startup work above the call must be "
      "reported. This is the whole defect: the knob keeps working, the receipt "
      "keeps printing, and the arm is silently `auto`")

mutant = main_text.replace(
    '        changed |= rasberySetEnvIfNeeded("OMP_WAIT_POLICY", "PASSIVE");',
    '        changed |= rasberySetEnvIfNeeded("OMP_WAIT_POLICY", "PASSIVE", true);', 1)
check(any("non-overwriting" in p for p in problems_main(mutant)),
      "negative control: turning the DEFAULT arm into an overwriting set must be "
      "reported -- the default arm is the B0 arm")

mutant = main_text.replace('unsetenv("GOMP_SPINCOUNT")',
                           'setenv("GOMP_SPINCOUNT", "0", 1)', 1)
check(any("must drop GOMP_SPINCOUNT" in p for p in problems_main(mutant)),
      "negative control: an `active` arm that leaves GOMP_SPINCOUNT=0 in place "
      "is a PASSIVE run wearing an ACTIVE label; it must be reported")

mutant = main_text.replace('\\"applied\\"', '\\"got\\"', 1)
check(any("does not report `applied`" in p for p in problems_main(mutant)),
      "negative control: renaming a receipt key breaks every parser reading the "
      "campaign table and must be reported")


# ===========================================================================
# 3. The stub keeps the CPU-only build honest
# ===========================================================================
check("no-cuda" in stub_text,
      "the CPU-only stub must report rc:'no-cuda'. A stub that printed the "
      "operator's requested mode with a success-looking rc would make a row "
      "that never touched a device look like a device arm")
check('getenv("RASBERY_CUDA_SYNC_MODE")' in stub_text,
      "the stub must still echo what was REQUESTED, so the receipt says the "
      "knob was set on a build that cannot honour it")


# ===========================================================================
# 4. The dispatcher forwards both keys, and neither is a harness default
# ===========================================================================
import run_single_gpu_batch as sg  # noqa: E402

KNOBS = ("RASBERY_CUDA_SYNC_MODE", "RASBERY_OMP_WAIT")

for key in KNOBS:
    check(key not in sg.DEFAULT_ENV,
          f"{key} must NOT be in DEFAULT_ENV. DEFAULT_ENV is the 238 reference "
          "line key for key (test/reference/batch_reference_env_238.json) and the "
          "reference sets neither; a harness default the reference never had is "
          "exactly what RASBERY_PPR_MODE=master was")
    check(key in runner_text,
          f"{key} must be DOCUMENTED in tools/run_multi_gpu_batch.py -- a knob "
          "nobody can find is a knob nobody takes")

baseline = sg.resolve_profile_env(
    batch_width=8, driver_workers=8, solver_threads=8, gpu="0")
for key in KNOBS:
    check(key not in baseline,
          f"{key} leaked into the resolved env without --set; the default arm "
          "must be the reference environment untouched")

overrides = sg.parse_overrides(
    ["RASBERY_CUDA_SYNC_MODE=blocking", "RASBERY_OMP_WAIT=passive"])
forwarded = sg.resolve_profile_env(
    batch_width=8, driver_workers=8, solver_threads=8, gpu="0", overrides=overrides)
check(forwarded.get("RASBERY_CUDA_SYNC_MODE") == "blocking"
      and forwarded.get("RASBERY_OMP_WAIT") == "passive",
      "`--set` must forward both knobs verbatim to the child environment")
check({k: v for k, v in forwarded.items() if k not in KNOBS} == baseline,
      "`--set RASBERY_CUDA_SYNC_MODE=... --set RASBERY_OMP_WAIT=...` must change "
      "NOTHING else in the child environment, or the arms of the WP16 matrix are "
      "not comparable to each other")

# The active arm is the one that has to beat a DEFAULT_ENV value, so check it.
check(sg.DEFAULT_ENV.get("OMP_WAIT_POLICY") == "PASSIVE",
      "DEFAULT_ENV no longer exports OMP_WAIT_POLICY=PASSIVE, so the note in "
      "main.cpp about the active arm needing an overwriting set is stale")


if failures:
    for problem in failures:
        print(f"[RASBERY][CUDA_SCHED_CONTRACT][FAIL] {problem}", file=sys.stderr)
    raise SystemExit(1)

print("[RASBERY][CUDA_SCHED_CONTRACT][OK] the schedule flag precedes every "
      "context creator, the OMP wait policy is exported before the re-exec, the "
      "default path enters the CUDA runtime not at all, and --set forwards both "
      "knobs without disturbing the reference environment")
