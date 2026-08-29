#!/usr/bin/env python3
"""Static contract for Rev.7.1 Task 20 -- immediate slot refill.

Refill is not a new code path.  It is the OpenMP dynamic queue in main.cpp plus
the two arenas' acquire/release, and it has worked since the batch branch
existed.  What this pins is the part that CAN silently stop being true:

  1. the job queue is admitted from a manifest, not only from argv, and the
     manifest is expanded BEFORE the output-namespace check (otherwise a
     manifest could smuggle two jobs onto one --raso and they would race);
  2. the Driver is destroyed -- i.e. the slot is released -- before the tenancy
     is stamped closed, so refill latency includes the teardown it is measuring;
  3. both arenas audit the tenant reset and refuse a slot that is queued twice;
  4. the receipt carries every field the gate reads.

A regression in any of these produces a run that still LOOKS right: the decks
finish, the numbers are finite, and only the receipt would have said otherwise.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
main_cpp = (root / "src" / "main.cpp").read_text(encoding="utf-8")
refill_h = (root / "src" / "BatchRefill.h").read_text(encoding="utf-8")
cmfd_cu = (root / "src" / "CudaBICGBackend.cu").read_text(encoding="utf-8")
nodal_cu = (root / "src" / "CudaXsReconBackend.cu").read_text(encoding="utf-8")

failures: list[str] = []


def require(source: str, name: str, *tokens: str) -> None:
    for token in tokens:
        if token not in source:
            failures.append(f"{name}: missing {token!r}")


# --- 1. the job queue -------------------------------------------------------
require(
    main_cpp,
    "main.cpp --jobs",
    'option == "--jobs"',
    'option != "--jobs"',
    "rasberyReadJobManifest(",
    "job_manifests",
)

# The manifest has to be expanded before the counts-match test, which is itself
# before the distinct-output test.  If it moved after either, a manifest's jobs
# would skip the namespace rule that keeps two Drivers out of one HDF5 file.
expand = main_cpp.find("rasberyReadJobManifest(manifest, rasbery_inputs")
counts = main_cpp.find('std::cerr << "The number of --rasi and --raso paths must match."')
namespace_rule = main_cpp.find("--raso paths must be distinct, one per deck: entry")
if not (0 < expand < counts < namespace_rule):
    failures.append(
        "main.cpp: --jobs manifests must be expanded before the counts-match and "
        "distinct-output checks (expand=%d counts=%d namespace=%d)"
        % (expand, counts, namespace_rule)
    )

# CRLF: a manifest authored on Windows and read in WSL would otherwise put \r
# into every output path.
require(main_cpp, "main.cpp manifest", "line.back() == '\\r'")

# --- 2. the ledger, and the order it is stamped in --------------------------
require(
    main_cpp,
    "main.cpp ledger",
    "rasbery::refill::ledger().begin(jobs, batch_width, host_threads)",
    "rasbery::refill::ledger().jobStarted(i, lane)",
    "rasbery::refill::ledger().jobFinished(i)",
    "rasbery::refill::ledger().report(std::cout)",
    "schedule(dynamic, 1)",
)

# The Driver must be in its own scope inside the try, so the destructor (which
# releases the arena slot) runs before jobFinished stamps the tenancy end.
driver_block = main_cpp[main_cpp.find("ledger().jobStarted(i, lane)"): main_cpp.find("ledger().jobFinished(i)")]
if "{\n                    rasbery::Driver driver(" not in driver_block:
    failures.append(
        "main.cpp: the Driver must be scoped inside the try so its destructor "
        "(the slot release) runs before jobFinished() closes the tenancy"
    )

# --- 3. the tenancy audit ---------------------------------------------------
require(
    refill_h,
    "BatchRefill.h",
    "queue_duplicates",
    "stale_tenants",
    "double_releases",
    "admissions",
    "[RASBERY][REFILL]",
)
for key in (
    "jobs",
    "slots",
    "refills",
    "tail_idle_s",
    "slot_busy_fraction",
    "refill_latency_p50_ms",
    "duplicates",
    "stale_tenants",
):
    if f'\\"{key}\\":' not in refill_h:
        failures.append(f"BatchRefill.h: the [RASBERY][REFILL] receipt is missing {key!r}")

require(
    cmfd_cu,
    "CudaBICGBackend.cu",
    "batchSlotIsReset(sl)",
    "tenancy().stale_tenants.fetch_add",
    "tenancy().double_releases.fetch_add",
    "tenancy().queue_duplicates.fetch_add",
    "tenancy().admissions.fetch_add",
)
require(
    nodal_cu,
    "CudaXsReconBackend.cu",
    "nodalSlotIsReset(sl, _canon[static_cast<std::size_t>(m)])",
    "tenancy().stale_tenants.fetch_add",
    "tenancy().double_releases.fetch_add",
    "tenancy().queue_duplicates.fetch_add",
    "tenancy().admissions.fetch_add",
)

# The nodal audit has to cover the OUT-OF-STRUCT canonical binding: that is the
# field that actually leaked between tenants once (Task 18), and an audit that
# only walked Slot would not have caught it.
if "canon.jnet == nullptr" not in nodal_cu:
    failures.append(
        "CudaXsReconBackend.cu: nodalSlotIsReset must audit the out-of-struct "
        "CanonicalSlotBuffers, which is the per-slot state that has actually "
        "survived a tenancy before"
    )

# Both arenas must still do the whole-struct reset the audit checks.
require(cmfd_cu, "CudaBICGBackend.cu reset", "sl        = BatchCore::Slot{};")
require(nodal_cu, "CudaXsReconBackend.cu reset", "sl              = Slot{};")

if failures:
    raise SystemExit("batch refill contract: FAIL\n  " + "\n  ".join(failures))
print("batch refill contract: PASS")
