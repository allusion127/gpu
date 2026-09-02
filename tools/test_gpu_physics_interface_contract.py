#!/usr/bin/env python3
"""Backend-neutral GPU physics interface contract (plan Rev.7.1 Task 1).

The bit-golden gates cannot see any of this: these are types, not arithmetic.
What they CAN hide is exactly the failure Rev.7.1 amendment 3 was written for --
a device control packet that silently has no room for two thirds of a case's
state, so a recycled slot inherits the previous tenant's search bracket.  So the
contract covers the properties a numerical A/B is blind to:

  1. the four-way split EXISTS and Rev.7's single `DeviceSlotControl` is gone --
     DeviceSlotPhase (hot 32 B) / DeviceSlotState / DeviceSearchState /
     DeviceScheduleParams, with the Task 1 Step 4 size and alignment asserts;
  2. FIELD COMPLETENESS (Step 4b): every field of Scheduler.h's SearchMemory,
     of the Schedule search block, and of the runtime/termination search block
     has a named counterpart on the device.  A field added to Scheduler.h and
     not to the device struct fails here, which is the recurrence guard for
     amendment 3;
  3. the five phases Rev.7.1 promoted out of host arithmetic and the two CRAM
     escape codes exist -- device code cannot throw, so milk.h's two throws must
     land in DeviceEscape or M2 is unreachable;
  4. device views are POINTERS AND SCALARS: no std::vector, std::string, std::map
     or virtual anywhere inside a `Device*` struct, and `burn_key` is an int*
     (Sec 3.5 / 6.16(3): a floating-point bracket compare splits boundary nodes
     between host and device);
  5. NO persistent / cooperative scaffolding anywhere in these files.  W0
     measured c_barrier = 0.78 us against the 0.384 us kill threshold, so the
     persistent track is closed; scaffolding for it would be dead weight that
     invites use;
  6. stub parity: every symbol GpuPhysicsTypes.h declares is defined in the
     no-CUDA translation unit, so a RASBERY_ENABLE_CUDA=OFF build links;
  7. compiled behaviour, when a compiler is available: the layout asserts fire,
     the refill reset leaves no field of any of the four structs at a previous
     tenant's value, and the Sec 5.2 duplicate-queue predicate agrees with the
     epoch rule.
"""
from __future__ import annotations

import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SLOT_CONTROL = (SRC / "GpuSlotControl.h").read_text(encoding="utf-8-sig")
PHYSICS_TYPES = (SRC / "GpuPhysicsTypes.h").read_text(encoding="utf-8-sig")
STUB = (SRC / "GpuPhysicsBackendStub.cpp").read_text(encoding="utf-8-sig")
CUDA_ARM = (SRC / "GpuPhysicsBackendCuda.cu").read_text(encoding="utf-8-sig")
SCHEDULER = (SRC / "Scheduler.h").read_text(encoding="utf-8-sig")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments so prose can never satisfy a code assertion."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


SLOT_CODE = strip_comments(SLOT_CONTROL)
TYPES_CODE = strip_comments(PHYSICS_TYPES)
STUB_CODE = strip_comments(STUB)
CUDA_ARM_CODE = strip_comments(CUDA_ARM)


def struct_body(name: str, text: str) -> str:
    """The brace-matched body of `struct <name>` / `class <name>`."""
    match = re.search(r"\b(?:struct|class)\s+(?:alignas\s*\([^)]*\)\s*)?" + name + r"\b", text)
    if match is None:
        raise SystemExit(f"gpu physics interface: FAIL: no definition of {name}")
    open_at = text.find("{", match.end())
    if open_at < 0:
        raise SystemExit(f"gpu physics interface: FAIL: {name} has no body")
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    raise SystemExit(f"gpu physics interface: FAIL: unbalanced body for {name}")


DECLARATION = re.compile(
    r"^\s*(?:const\s+|static\s+|inline\s+|constexpr\s+|mutable\s+)*"
    r"(?:unsigned\s+|signed\s+|long\s+|short\s+)*"
    r"[A-Za-z_][\w:]*"                                     # type
    r"(?:\s*<[^;{}]*>)?"                                   # template arguments
    r"\s*\**\s+\**"                                        # pointer stars, either side
    r"(?P<names>[A-Za-z_]\w*(?:\s*\[[^\]]*\])?"
    r"(?:\s*,\s*\**\s*[A-Za-z_]\w*(?:\s*\[[^\]]*\])?)*)"
    r"\s*(?:=[^;]*)?;"
)


def members(body: str) -> list[str]:
    """Declared member names of a POD body, in declaration order.

    Comment-stripped first, so an aligned `= false;` initialiser or a trailing
    `// comment` cannot hide (or invent) a field.  Function declarations are not
    matched -- none of the bodies this is pointed at has any.
    """
    names: list[str] = []
    for line in strip_comments(body).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("}"):
            continue
        # A "(" LEFT of the initialiser means a function declaration; one to the
        # right is a cast or constant expression in a default value
        # (`= static_cast<int>(SearchExit::NONE);`), which is still a field.
        if "(" in line.split("=", 1)[0]:
            continue
        match = DECLARATION.match(line)
        if match is None:
            continue
        for raw in match.group("names").split(","):
            names.append(re.sub(r"\s*\[[^\]]*\]", "", raw).strip().lstrip("*").strip())
    return names


def between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"gpu physics interface: FAIL: Scheduler.h lost {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"gpu physics interface: FAIL: Scheduler.h lost {end_marker!r}")
    return text[start:end]


problems: list[str] = []


def need(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


# ---------------------------------------------------------------------------
# 1. The four-way split exists; Rev.7's single packet is gone.
# ---------------------------------------------------------------------------
for name in ("DeviceSlotPhase", "DeviceSlotState", "DeviceSearchState", "DeviceScheduleParams"):
    need(re.search(r"\bstruct\s+(?:alignas\s*\([^)]*\)\s*)?" + name + r"\s*\{", SLOT_CODE) is not None,
         f"GpuSlotControl.h does not define struct {name}")

for path, text in (("GpuSlotControl.h", SLOT_CODE), ("GpuPhysicsTypes.h", TYPES_CODE)):
    need("DeviceSlotControl" not in text,
         f"{path} still mentions Rev.7's single DeviceSlotControl (Rev.7.1 deletes it)")
    need("search_history" not in text,
         f"{path} still carries `search_history`; Sec 3.2(C) replaces it with DeviceSearchState")

for assertion in (
    "static_assert(sizeof(DeviceSlotPhase) == 32",
    "static_assert(alignof(DeviceSlotPhase) == 32)",
    "static_assert(std::is_trivially_copyable_v<DeviceSlotPhase>)",
    "static_assert(alignof(DeviceSlotState) == 128)",
    "static_assert(sizeof(DeviceSlotState) % 128 == 0)",
    "static_assert(std::is_trivially_copyable_v<DeviceSlotState>)",
    "static_assert(std::is_trivially_copyable_v<DeviceSearchState>)",
    "static_assert(std::is_trivially_copyable_v<DeviceScheduleParams>)",
):
    need(assertion in SLOT_CODE, f"GpuSlotControl.h lost the Task 1 Step 4 assert: {assertion}")

phase_body = struct_body("DeviceSlotPhase", SLOT_CODE)
for field in ("phase", "queued_phase", "escape", "flags", "state_epoch", "queued_epoch",
              "phase_age", "input_id", "job_id", "error_code"):
    need(field in members(phase_body), f"DeviceSlotPhase lacks {field}")
for helper in ("slotActive", "slotInFlight", "slotAlreadyQueued"):
    need(helper in SLOT_CODE, f"GpuSlotControl.h lacks the {helper} accessor")

state_fields = members(struct_body("DeviceSlotState", SLOT_CODE))
generations = [f for f in state_fields if f.endswith("_generation")]
# The four that mirror a counter that EXISTS on the host today.  A device
# generation with no host counter behind it can never go stale, so gating an
# upload on one is a silent "never re-upload"; the header says which is which and
# this pins the four real ones.
for real, host_file, host_field in (
        ("micx_generation", "XSSet.h", "_micx_generation"),
        ("ref_generation", "XSSet.h", "_ref_generation"),
        ("hoststate_generation", "XSSet.h", "_hoststate_generation"),
        ("nodal_constant_generation", "Nodal.h", "_const_generation")):
    need(real in state_fields, f"DeviceSlotState lacks the host-backed counter {real}")
    host_text = (SRC / host_file).read_text(encoding="utf-8-sig")
    need(host_field in host_text,
         f"{host_file} no longer declares {host_field}; the device mirror {real} is now dead")
need(len(generations) == 12,
     f"DeviceSlotState carries {len(generations)} generation counters, Sec 3.2(B) lists 12")
need("SPECULATIVE" in SLOT_CONTROL,
     "the generation block does not say which counters have no host backing yet")
for field in ("substep", "substep_index", "clean_iters", "stall_sample_taken",
              "xe_aa_ncol", "xe_aa_have_prev", "xe_relax", "eigv_before_segment"):
    need(field in state_fields, f"DeviceSlotState lacks {field}")

# ---------------------------------------------------------------------------
# 2. Field completeness against Scheduler.h (Task 1 Step 4b).
#
# The host name -> device name map IS the contract.  Extracting the host side
# from Scheduler.h means a field added there and not here fails this test.
# ---------------------------------------------------------------------------
SEARCH_MEMORY_MAP = {
    "has_boron_secant": "has_boron_secant",
    "boron_secant_dkdx": "boron_secant_dkdx",
    "has_rod_secant": "has_rod_secant",
    "rod_secant_x": "rod_secant_x",
    "rod_secant_dkdx": "rod_secant_dkdx",
}
RUNTIME_SEARCH_MAP = {
    "search_initialized": "initialized",
    "search_seeded_from_previous_step": "seeded_from_previous_step",
    "search_has_prev": "has_prev",
    "search_has_bracket": "has_bracket",
    "search_has_best": "has_best",
    "search_slope_frozen": "slope_frozen",
    "search_iteration": "iteration",
    "search_current_x": "current_x",
    "search_prev_x": "prev_x",
    "search_prev_eigv": "prev_eigv",
    "search_frozen_slope": "frozen_slope",
    "search_best_x": "best_x",
    "search_best_residual": "best_residual",
    "search_bracket_lo_x": "bracket_lo_x",
    "search_bracket_lo_residual": "bracket_lo_residual",
    "search_bracket_hi_x": "bracket_hi_x",
    "search_bracket_hi_residual": "bracket_hi_residual",
    "search_exit_status": "exit_status",
    "search_exit_dk": "exit_dk",
    "search_exit_tol": "exit_tol",
    "search_stall_count": "stall_count",
}
SCHEDULE_PARAM_MAP = {
    "searchType": "search_type",
    "max_outer_iter": "max_outer_iter",
    "max_search_iter": "max_search_iter",
    "max_th_iter": "max_th_iter",
    "tolerance_keff": "tolerance_keff",
    "tolerance_search": "tolerance_search",
    "rodcrit_search_floor": "rodcrit_search_floor",
    "rodcrit_search_cap": "rodcrit_search_cap",
    "tolerance_th": "tolerance_th",
    "target_keff": "target_keff",
    "tolerance_tmod": "tolerance_tmod",
    "tolerance_tfuel": "tolerance_tfuel",
    "search_boron_ppm": "search_boron_ppm",
    "tolerance_boron": "tolerance_boron",
    "tolerance_rodsearch": "tolerance_rodsearch",
    "search_relaxation": "search_relaxation",
    "search_low": "search_low",
    "search_hi": "search_hi",
    "slope_freeze_thres": "slope_freeze_thres",
    "min_secant_denom": "min_secant_denom",
    "bracket_min_span": "bracket_min_span",
    "search_boron_probe": "search_boron_probe",
    "search_rod_probe": "search_rod_probe",
}

sched_code = strip_comments(SCHEDULER)
host_search_memory = members(struct_body("SearchMemory", sched_code))
host_runtime = members(
    between(SCHEDULER, "// Runtime search state managed by Driver", "// OUTPUT parameters")
)
host_params = members(between(SCHEDULER, "// Search condition", "// Rod insertion parameters"))

device_search = members(struct_body("DeviceSearchState", SLOT_CODE))
device_params = members(struct_body("DeviceScheduleParams", SLOT_CODE))

need(len(host_search_memory) == 5,
     f"Scheduler.h SearchMemory now has {len(host_search_memory)} fields, the map covers 5")
need(len(host_runtime) == 21,
     f"Scheduler.h runtime/termination search block now has {len(host_runtime)} fields, the map covers 21")
# WP24 took this to 23: Schedule::rodcrit_search_cap, the RODCRIT clamp that
# used to be the kRodCritSearchTol literal inside criticalSearchTolerance() and
# had to become a value for a fidelity preset to be able to move it.
need(len(host_params) == 23,
     f"Scheduler.h search-condition block now has {len(host_params)} fields, the map covers 23")

for host_name in host_search_memory + host_runtime:
    mapped = {**SEARCH_MEMORY_MAP, **RUNTIME_SEARCH_MAP}.get(host_name)
    need(mapped is not None, f"Scheduler.h search field {host_name!r} has no DeviceSearchState counterpart")
    if mapped is not None:
        need(mapped in device_search, f"DeviceSearchState lacks {mapped!r} (Scheduler.h {host_name})")

need(len(device_search) == 26,
     f"DeviceSearchState has {len(device_search)} fields; Sec 3.2(C) is exactly 26")

for host_name in host_params:
    mapped = SCHEDULE_PARAM_MAP.get(host_name)
    need(mapped is not None,
         f"Scheduler.h search parameter {host_name!r} has no DeviceScheduleParams counterpart")
    if mapped is not None:
        need(mapped in device_params,
             f"DeviceScheduleParams lacks {mapped!r} (Scheduler.h {host_name})")

# The Sec 3.2(D) physics half, which Rev.7 assumed was shared and is not.
for field in ("schedule_type", "th_mode", "rated_power", "actual_power", "substeps",
              "xenon_transient", "bppm0", "tful0", "tmod0", "dmod0", "pressure",
              "inlet_temp", "outlet_temp", "mass_flow_rate", "fuel_temp_rise_scale",
              "delta_tful", "delta_tmod", "delta_dmod", "delta_bppm", "delta_xe", "delta_sm"):
    need(field in device_params, f"DeviceScheduleParams lacks the per-deck field {field!r}")

# Device-side default constants must equal the Scheduler.h ones NUMERICALLY --
# the duplication is only safe while that holds.
CONSTANT_PAIRS = (
    ("kMaxEigenIter", "kDevMaxEigenIter"),
    ("kMaxSearchIter", "kDevMaxSearchIter"),
    ("kMaxThIter", "kDevMaxThIter"),
    ("kEigvTol", "kDevEigvTol"),
    ("kCritSearchTol", "kDevCritSearchTol"),
    ("kRodCritSearchTol", "kDevRodCritSearchTol"),
    ("kThTol", "kDevThTol"),
    ("kTempSearchTol", "kDevTempSearchTol"),
    ("kBoronSearchTol", "kDevBoronSearchTol"),
    ("kRodSearchTol", "kDevRodSearchTol"),
    ("kSearchRelax", "kDevSearchRelax"),
    ("kSearchLow", "kDevSearchLow"),
    ("kSearchHigh", "kDevSearchHigh"),
    ("kSlopeFreezeThres", "kDevSlopeFreezeThres"),
    ("kMinSecantDenom", "kDevMinSecantDenom"),
    ("kBracketMinSpan", "kDevBracketMinSpan"),
    ("kBoronProbe", "kDevBoronProbe"),
    ("kRodProbe", "kDevRodProbe"),
)


def constant_value(name: str, text: str) -> float | None:
    match = re.search(r"\b" + name + r"\s*=\s*([-\w.+eE]+)\s*[u;]", text)
    if match is None:
        return None
    literal = match.group(1).rstrip("uU")
    try:
        return float(literal)
    except ValueError:
        return None


for host_name, device_name in CONSTANT_PAIRS:
    host_value = constant_value(host_name, sched_code)
    device_value = constant_value(device_name, SLOT_CODE)
    need(host_value is not None, f"Scheduler.h lost {host_name}")
    need(device_value is not None, f"GpuSlotControl.h lost {device_name}")
    if host_value is not None and device_value is not None:
        need(host_value == device_value,
             f"{device_name}={device_value} != Scheduler.h {host_name}={host_value}")

# ---------------------------------------------------------------------------
# 3. Phases and escapes (Sec 3.1).
# ---------------------------------------------------------------------------
for phase in ("NormalizeFluxSign", "Derivative", "RodOp", "Ppr", "ResultAggregate"):
    need(phase in SLOT_CODE, f"DevicePhase lacks the Rev.7.1 phase {phase}")
for phase in ("Empty", "Import", "Material", "Outer", "Xenon", "ThermalHydraulics", "Search",
              "DepletionPredictor", "DepletionCorrector", "OutputPack", "Done", "Failed"):
    need(phase in SLOT_CODE, f"DevicePhase lacks {phase}")
for escape in ("FluxLimitCycleSample", "FluxStallFatal", "CramNotConverged", "CramZeroDiagonal"):
    need(escape in SLOT_CODE, f"DeviceEscape lacks {escape}")

# ---------------------------------------------------------------------------
# 4. Device views are pointers and scalars; burn_key is an int*.
# ---------------------------------------------------------------------------
DEVICE_STRUCTS = (
    ("DeviceSlotPhase", SLOT_CODE),
    ("DeviceSlotState", SLOT_CODE),
    ("DeviceSearchState", SLOT_CODE),
    ("DeviceScheduleParams", SLOT_CODE),
    ("DeviceGeometryView", TYPES_CODE),
    ("DeviceXsLibraryView", TYPES_CODE),
    ("DeviceSlotView", TYPES_CODE),
)
BANNED_IN_VIEW = ("std::vector", "std::string", "std::map", "std::array", "std::unique_ptr",
                  "std::shared_ptr", "virtual", "std::function")
for name, text in DEVICE_STRUCTS:
    body = struct_body(name, text)
    for token in BANNED_IN_VIEW:
        need(token not in body, f"{name} contains {token} -- a device view is pointers and scalars only")

slot_view = struct_body("DeviceSlotView", TYPES_CODE)
need(re.search(r"\bint\s*\*\s*burn_key\s*;", slot_view) is not None,
     "DeviceSlotView.burn_key is not an `int*` (Sec 3.5 / 6.16(3): the bracket key is integer)")
need(re.search(r"\bdouble\s*\*?\s*burnup\b", slot_view) is None,
     "DeviceSlotView still carries a floating-point `burnup`; Rev.7.1 replaces it with burn_key")
for pointer in ("DeviceSlotPhase*", "DeviceSlotState*", "DeviceSearchState*", "DeviceScheduleParams*"):
    need(pointer in slot_view, f"DeviceSlotView does not point at {pointer}")
for array in ("ref_micx", "ref_lmpx", "xe_aa_history", "bos_micx"):
    need(array in slot_view, f"DeviceSlotView lacks the Rev.7.1 array {array} (Sec 3.5)")

# Every pointer in the library view must have an arena region behind it.  A view
# field with no backing reads as available and dereferences to whatever the
# arena last wrote there.
library_view = struct_body("DeviceXsLibraryView", TYPES_CODE)
need("knot_offsets" not in library_view,
     "DeviceXsLibraryView still declares knot_offsets, which no host array backs "
     "(the offsets are an int field inside the branch descriptors)")
LAYOUT = (SRC / "GpuPhysicsArenaLayout.h").read_text(encoding="utf-8-sig")
for backed in ("lib_flux", "lib_chix"):
    need(backed in library_view, f"DeviceXsLibraryView lost {backed}")
need("LibraryRegion::LibFlux" in LAYOUT and "LibraryRegion::LibChix" in LAYOUT,
     "lib_flux / lib_chix have no arena region behind them")

# The same rule for the GEOMETRY view, where it had actually been broken.
# `DeviceGeometryView::comps` had an arena region (GeometryRegion::Comps, nxyz
# ints of VRAM per cohort) whose only host backing was Geometry::_comps --
# `new int[nxyz]`, so uninitialised, and written by NOTHING.  Nothing read it on
# the host either: the composition index the solver uses is XSSet::_comp
# (XSSet.h), a std::size_t array on a different object.  So a kernel that
# believed the name would have read whatever the arena last held, and the
# pointer was non-null, which is the way this kind of hole survives review.
LAYOUT_CODE = strip_comments(LAYOUT)
GEOMETRY_CODE = strip_comments((SRC / "Geometry.h").read_text(encoding="utf-8-sig"))
geometry_view = struct_body("DeviceGeometryView", TYPES_CODE)
need("comps" not in geometry_view,
     "DeviceGeometryView declares `comps` again; no host array fills it "
     "(Geometry::_comps was allocated and never written)")
need("Comps" not in LAYOUT_CODE,
     "the arena catalogues a Comps geometry region again; nothing imports into it, "
     "so the view would publish uninitialised device memory")
need("_comps" not in GEOMETRY_CODE,
     "Geometry::_comps is back -- an uninitialised [nxyz] array no code writes or "
     "reads; XSSet::_comp is the composition map")

# The BOS microscopic snapshot is FOUR slots, not eleven (Sec 6.18).
need("kBosMicroXtCount = 4" in TYPES_CODE,
     "GpuPhysicsTypes.h does not pin the BOS microscopic snapshot at 4 slots")
need("kBosMicroXt[4]" in TYPES_CODE and "kXtXsaf" in TYPES_CODE and "kXtXsff" in TYPES_CODE
     and "kXtXs2n" in TYPES_CODE and "kXtXs3n" in TYPES_CODE,
     "the BOS slot table is not XSAF/XSFF/XS2N/XS3N")

need("GpuBackendKind" in TYPES_CODE and "GpuCapabilityReceipt" in TYPES_CODE,
     "GpuPhysicsTypes.h lacks the Sec 1.5 backend kind / capability receipt")
kind_enum = TYPES_CODE[TYPES_CODE.find("enum class GpuBackendKind"):]
kind_enum = kind_enum[: kind_enum.find("}")]
for kind in ("Cuda", "Hip", "Sycl", "None"):
    need(kind in kind_enum, f"GpuBackendKind lacks {kind}")
tier_enum = TYPES_CODE[TYPES_CODE.find("enum class GpuSupportTier"):]
tier_enum = tier_enum[: tier_enum.find("}")]
for tier in ("Unsupported", "G0", "G1", "G2", "G3", "G4"):
    need(tier in tier_enum, f"GpuSupportTier lacks the Rev.7 Sec 1.5 rung {tier}")

# ---------------------------------------------------------------------------
# 5. No persistent / cooperative scaffolding, and no CUDA in the headers.
#
# W0: c_barrier = 0.78 us vs the 0.384 us kill threshold -> the persistent track
# is closed (constraint 17).  Nothing here may carry scaffolding for it.
# ---------------------------------------------------------------------------
FORBIDDEN_PERSISTENT = ("cooperative_groups", "grid_group", "grid.sync", "this_grid",
                        "cudaLaunchCooperativeKernel", "PersistentKernel", "persistent_kernel",
                        "blocksPerSM", "cudaOccupancyMaxActiveBlocksPerMultiprocessor")
for path, text in (("GpuSlotControl.h", SLOT_CODE), ("GpuPhysicsTypes.h", TYPES_CODE),
                   ("GpuPhysicsBackendStub.cpp", STUB_CODE)):
    for token in FORBIDDEN_PERSISTENT:
        need(token not in text,
             f"{path} carries persistent/cooperative scaffolding ({token}); W0 closed that track")
    need("cuda_runtime.h" not in text, f"{path} includes cuda_runtime.h; these types are backend-neutral")
    need(re.search(r"\bcuda[A-Z]\w*\s*\(", text) is None,
         f"{path} calls a CUDA API directly; the types layer is backend-neutral")

# A capability flag for cooperative launch would invite exactly the code W0
# ruled out.
need("cooperative" not in TYPES_CODE.lower(),
     "GpuCapabilityReceipt exposes a cooperative-launch capability; W0 closed that track")

# ---------------------------------------------------------------------------
# 6. Stub parity: every declared backend symbol is defined without CUDA.
# ---------------------------------------------------------------------------
backend_body = struct_body("GpuPhysicsBackend", TYPES_CODE)
declared = set(re.findall(r"\b(\w+)\s*\([^)]*\)\s*(?:const\s*)?;", backend_body))
declared.discard("GpuPhysicsBackend")
# BOTH arms, not just the stub.  A CUDA build REMOVES the stub from the source
# list, so a symbol defined only there is an undefined reference at the end of a
# twenty-minute build -- which is exactly what happened.
ARMS = (("GpuPhysicsBackendStub.cpp", STUB_CODE), ("GpuPhysicsBackendCuda.cu", CUDA_ARM_CODE))
for arm_name, arm in ARMS:
    for method in sorted(declared):
        need(re.search(r"GpuPhysicsBackend::" + method + r"\s*\(", arm) is not None,
             f"{arm_name} does not define GpuPhysicsBackend::{method}")
    for ctor in ("GpuPhysicsBackend::GpuPhysicsBackend(",
                 "GpuPhysicsBackend::~GpuPhysicsBackend("):
        need(ctor in arm, f"{arm_name} does not define {ctor})")
    need("rasberyGpuPhysicsEnabled" in arm,
         f"{arm_name} does not define rasberyGpuPhysicsEnabled")
    need("struct GpuPhysicsBackend::Impl" in arm, f"{arm_name} has no Impl definition")

# The CUDA arm reports what it MEASURED, not what it hopes.
need("cudaGetDeviceCount" in CUDA_ARM_CODE,
     "the CUDA backend does not probe for a device")
need("cudaGetDeviceProperties" in CUDA_ARM_CODE,
     "the CUDA backend does not read the device properties it reports")
need("cudaDevAttrMemoryPoolsSupported" in CUDA_ARM_CODE,
     "the CUDA backend infers memory-pool support instead of asking the driver; "
     "the arena's single allocation depends on it")
need("GpuSupportTier::G3" in CUDA_ARM_CODE and "GpuSupportTier::G2" in CUDA_ARM_CODE,
     "the CUDA backend does not report the two Sec 1.5 CUDA tiers")
need("CUDART_VERSION" in CUDA_ARM_CODE,
     "conditional-graph support is claimed without a runtime-version check")
need("cooperative" not in CUDA_ARM_CODE.lower(),
     "the CUDA backend exposes a cooperative-launch capability; W0 closed that track")
need(re.search(r"bool GpuPhysicsBackend::available\(\)\s*const\s*\{\s*return false;\s*\}",
               STUB_CODE) is not None,
     "the stub's available() does not return false")

# ---------------------------------------------------------------------------
# 7. Build wiring: the stub is compiled when CUDA is off, dropped when it is on.
# ---------------------------------------------------------------------------
need("GpuPhysicsBackendStub.cpp" in CMAKE,
     "CMakeLists.txt does not mention GpuPhysicsBackendStub.cpp")
remove_block = CMAKE[CMAKE.find("list(REMOVE_ITEM RASBERY_SOURCES"):]
remove_block = remove_block[: remove_block.find(")")]
need("GpuPhysicsBackendStub.cpp" in remove_block,
     "GpuPhysicsBackendStub.cpp is not removed from the sources in a CUDA build")
need("GpuPhysicsBackendCuda.cu" in CMAKE,
     "the CUDA build removes the stub but never compiles GpuPhysicsBackendCuda.cu")

# ---------------------------------------------------------------------------
# 8. Compiled behaviour.
# ---------------------------------------------------------------------------
HARNESS = r'''
#include "GpuPhysicsTypes.h"

#include <cstdio>
#include <cstring>

using namespace rasbery::gpu;

#define CHECK(cond, code)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);         \
            return (code);                                                       \
        }                                                                        \
    } while (false)

int main() {
    // Layout: the hot struct is 32 B so 64 slots are 2 KiB in L1.
    CHECK(sizeof(DeviceSlotPhase) == 32, 1);
    CHECK(alignof(DeviceSlotPhase) == 32, 2);
    CHECK(alignof(DeviceSlotState) == 128, 3);
    CHECK(sizeof(DeviceSlotState) % 128 == 0, 4);
    CHECK(64 * sizeof(DeviceSlotPhase) == 2048, 5);

    // The Scheduler.h enum values the device mirrors have to agree.
    CHECK(static_cast<unsigned>(DeviceSearchExit::None) == 0, 10);
    CHECK(static_cast<unsigned>(DeviceSearchExit::Converged) == 1, 11);
    CHECK(static_cast<unsigned>(DeviceSearchExit::BestFallback) == 2, 12);
    CHECK(static_cast<unsigned>(DeviceSearchExit::Unconverged) == 3, 13);
    CHECK(static_cast<unsigned>(DeviceSearchType::Keff) == 0, 14);
    CHECK(static_cast<unsigned>(DeviceScheduleType::Standard) == 0, 15);
    CHECK(static_cast<unsigned>(DeviceThMode::None) == 0, 16);

    // Xe Anderson history sizing, Driver.h:732 / 999-1007.
    CHECK(kDevXeAndersonDepth == 2, 20);
    CHECK(kDevXeAndersonTriples == 10, 21);

    // BOS snapshot is four microscopic slots, not eleven.
    CHECK(kBosMicroXtCount == 4, 25);
    CHECK(kBosMicroXt[0] == kXtXsaf && kBosMicroXt[1] == kXtXsff, 26);
    CHECK(kBosMicroXt[2] == kXtXs2n && kBosMicroXt[3] == kXtXs3n, 27);

    // REFILL RESET.  Fill all four structs with a previous tenant's bytes, reset
    // them, and check that nothing recognisable survives.  This is the property
    // the Task 20 audit kernel will check on the device.
    DeviceSlotPhase      p;
    DeviceSlotState      s;
    DeviceSearchState    q;
    DeviceScheduleParams d;
    std::memset(&p, 0x5a, sizeof(p));
    std::memset(&s, 0x5a, sizeof(s));
    std::memset(&q, 0x5a, sizeof(q));
    std::memset(&d, 0x5a, sizeof(d));

    deviceSlotPhaseReset(p, 7u);
    CHECK(p.phase == static_cast<unsigned char>(DevicePhase::Empty), 30);
    CHECK(p.flags == 0u, 31);
    CHECK(p.state_epoch == 7u, 32);
    CHECK(!slotActive(p) && !slotInFlight(p) && !slotFatal(p) && !slotInputReady(p), 33);
    // A refilled slot must not look queued.
    CHECK(!slotAlreadyQueued(p), 34);
    CHECK(p.input_id == 0u && p.job_id == 0u && p.error_code == 0u && p.reserved == 0u, 35);
    CHECK(p.phase_age == 0u, 36);

    deviceSlotStateReset(s);
    CHECK(s.geometry_generation == 0u && s.hoststate_generation == 0u, 40);
    CHECK(s.isotope_generation == 0u && s.th_generation == 0u, 41);
    CHECK(s.xe_streak == 0u && s.xe_cap_charged == 0u && s.clean_iters == 0u, 42);
    CHECK(s.xe_relax == 1.0 && s.eigv == 1.0, 43);
    CHECK(s.substep == 1u && s.substep_index == 0u, 44);
    CHECK(s.flux_stall == 0u && s.stall_events == 0u && s.stall_sample_taken == 0u, 45);

    deviceSearchStateReset(q);
    // SearchMemory must NOT survive a refill: a carried secant slope is a wrong
    // first trial for a different deck.
    CHECK(q.has_boron_secant == 0u && q.boron_secant_dkdx == 0.0, 50);
    CHECK(q.has_rod_secant == 0u && q.rod_secant_dkdx == 0.0, 51);
    CHECK(q.rod_secant_x == 1.0, 52);
    CHECK(q.has_bracket == 0u && q.has_best == 0u && q.slope_frozen == 0u, 53);
    CHECK(q.bracket_lo_x == 0.0 && q.bracket_hi_x == 0.0, 54);
    CHECK(q.exit_status == static_cast<unsigned>(DeviceSearchExit::None), 55);
    CHECK(q.stall_count == 0u && q.iteration == 0u, 56);

    deviceScheduleParamsReset(d);
    CHECK(d.max_outer_iter == 200u && d.max_search_iter == 300u && d.max_th_iter == 10u, 60);
    CHECK(d.tolerance_keff == 1.0e-6 && d.tolerance_search == 1.0e-5, 61);
    CHECK(d.min_secant_denom == 1.0e-12 && d.bracket_min_span == 1.0e-6, 62);
    CHECK(d.search_boron_probe == 50.0 && d.search_rod_probe == 0.25, 63);
    CHECK(d.substeps == 1u && d.xenon_transient == 0u, 64);

    // Sec 5.2 ownership: an entry is live only while its captured epoch is
    // current, and a phase transition invalidates it without touching a queue.
    p.phase        = static_cast<unsigned char>(DevicePhase::Outer);
    p.queued_phase = p.phase;
    p.queued_epoch = p.state_epoch;
    CHECK(slotAlreadyQueued(p), 70);
    ++p.state_epoch;                       // phase transition
    CHECK(!slotAlreadyQueued(p), 71);
    p.queued_epoch = p.state_epoch;
    p.phase        = static_cast<unsigned char>(DevicePhase::Xenon);
    CHECK(!slotAlreadyQueued(p), 72);      // same epoch, different phase

    p.flags = kSlotFlagActive | kSlotFlagInFlight;
    CHECK(slotActive(p) && slotInFlight(p) && !slotFatal(p), 73);

    // Views: the four control pointers plus an int burn key.
    DeviceSlotView v{};
    v.phase  = &p;
    v.state  = &s;
    v.search = &q;
    v.params = &d;
    int burn_keys[4] = {0, 1, 2, 3};
    v.burn_key = burn_keys;
    CHECK(v.phase->state_epoch == p.state_epoch, 80);
    CHECK(v.burn_key[3] == 3, 81);

    // The backend links and refuses, without CUDA.
    GpuPhysicsBackend backend;
    CHECK(!backend.available(), 90);
    CHECK(!backend.status().empty(), 91);
    CHECK(backend.capability().tier == GpuSupportTier::Unsupported, 92);
    CHECK(GpuPhysicsBackend::compiledKind() == GpuBackendKind::None, 93);
    CHECK(std::strcmp(GpuPhysicsBackend::backendName(GpuBackendKind::Cuda), "cuda") == 0, 94);
    CHECK(backend.receiptJson().find("\"available\":false") != std::string::npos, 95);
    CHECK(!rasberyGpuPhysicsEnabled(), 96);
    return 0;
}
'''


def msvc_vcvars() -> str | None:
    """vcvars64.bat of the newest MSVC install, or None off Windows."""
    if os.name != "nt":
        return None
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    done = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, universal_newlines=True)
    root = done.stdout.strip().splitlines()
    if done.returncode != 0 or not root:
        return None
    bat = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return str(bat) if bat.is_file() else None


def compile_harness(compiler: str, cpp: Path, exe: Path, stub: Path) -> None:
    if compiler.lower().endswith("vcvars64.bat"):
        script = cpp.parent / "build_gpu_physics_types_test.bat"
        script.write_text(
            "@echo off\r\n"
            + 'call "%s" >nul\r\n' % compiler
            + 'cl /nologo /std:c++20 /EHsc /W4 /WX "%s" "%s" /I "%s" /Fe:"%s"\r\n'
              % (cpp, stub, SRC, exe),
            encoding="utf-8")
        subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(cpp.parent),
                       capture_output=True, universal_newlines=True)
        return
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
         "-I", str(SRC), str(cpp), str(stub), "-o", str(exe)],
        check=True, capture_output=True, universal_newlines=True,
    )


def run_harness(compiler: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cpp = tmp_path / "gpu_physics_types_test.cpp"
        exe = tmp_path / ("gpu_physics_types_test.exe" if os.name == "nt"
                          else "gpu_physics_types_test")
        cpp.write_text(HARNESS, encoding="utf-8")
        try:
            compile_harness(compiler, cpp, exe, SRC / "GpuPhysicsBackendStub.cpp")
        except subprocess.CalledProcessError as failure:
            output = (failure.stdout or "") + (failure.stderr or "")
            return ["harness did not compile: " + output.strip()[-2000:]]
        done = subprocess.run([str(exe)], capture_output=True, universal_newlines=True)
        if done.returncode != 0:
            return ["compiled harness exited %d: %s" % (done.returncode, done.stderr.strip())]
    return []


def main() -> int:
    compiler = (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
                or msvc_vcvars())
    if compiler is not None:
        problems.extend(run_harness(compiler))
    if problems:
        for problem in problems:
            print("gpu physics interface: FAIL " + problem, file=sys.stderr)
        return 1
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    if compiler is None:
        print("gpu physics interface: static contract PASS "
              "(no C++ compiler here -- the compiled harness was skipped)")
    else:
        print("gpu physics interface: PASS (static contract + compiled harness, %s)"
              % Path(compiler).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
