#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "test" / "7_i-SMR_Validation"
REF_DIR = CASE_DIR / "Reference_output"
PLOT_DIR = CASE_DIR / "plots"

R_BANK = ["R4", "R3", "R2", "R1"]
REF_ROD_ORDER = ["S1", "R3", "S3", "S2", "R4", "R1", "S4", "R2", "S5", "S6"]
REF_ROD_START_COL = 10
ASSEMBLY_STATES = ("BOC", "MOC", "EOC")
SKIP_FIRST_MASTER_RESULT = True


def numeric_row(line: str) -> list[float] | None:
    tokens = line.split()
    if not tokens:
        return None
    try:
        return [float(tok) for tok in tokens]
    except ValueError:
        return None


def parse_reference_sum(path: Path) -> dict[str, object]:
    lines = path.read_text(errors="replace").splitlines()
    rod_rows: list[list[float]] = []
    peak_rows: list[list[float]] = []
    assembly_maps: list[dict[str, object]] = []

    in_rod = False
    in_peak = False
    in_assembly = False
    saw_rod_order = False
    saw_peak_header = False

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "(CONTROL ROD POSITIONS)" in line:
            in_rod = True
            in_peak = False
            in_assembly = False
            saw_rod_order = False
            idx += 1
            continue
        if "SUMMARY EDIT 2" in line:
            in_rod = False
        if "SUMMARY EDIT" in line and "SUMMARY EDIT 3" not in line:
            in_peak = False
        if "SUMMARY EDIT 5" in line:
            in_assembly = True
            in_rod = False
            in_peak = False
            idx += 1
            continue
        if "SUMMARY EDIT 6" in line:
            in_assembly = False
        if "SUMMARY EDIT 3" in line:
            in_peak = True
            in_rod = False
            in_assembly = False
            saw_peak_header = False
            idx += 1
            continue

        if in_rod:
            if all(name in line for name in REF_ROD_ORDER):
                saw_rod_order = True
                idx += 1
                continue
            if not saw_rod_order:
                idx += 1
                continue
            row = numeric_row(line)
            if row and len(row) >= REF_ROD_START_COL + len(REF_ROD_ORDER):
                rod_rows.append(row)

        if in_peak:
            if re.search(r"\bNO\.\s+DAY\s+EFPD\s+AO\b", line):
                saw_peak_header = True
                idx += 1
                continue
            if not saw_peak_header:
                idx += 1
                continue
            row = numeric_row(line)
            if row and len(row) == 12:
                peak_rows.append(row)

        if in_assembly:
            match = re.search(
                r"NO\.\=\s*(\d+)\s+DAY\s*=\s*([0-9.Ee+-]+)\s+EFPD\s*=\s*([0-9.Ee+-]+)",
                line,
            )
            if match:
                power_rows: list[list[float]] = []
                burn_rows: list[list[float]] = []
                idx += 1
                while idx < len(lines):
                    row_line = lines[idx]
                    if "NO.=" in row_line or "SUMMARY EDIT" in row_line:
                        idx -= 1
                        break
                    tokens = row_line.split()
                    if tokens and tokens[0].isdigit():
                        count = len(tokens) - 1
                        if idx + 2 < len(lines):
                            power = numeric_row(lines[idx + 1])
                            burn = numeric_row(lines[idx + 2])
                            if power and burn and len(power) == count and len(burn) == count:
                                power_rows.append(power)
                                burn_rows.append(burn)
                                idx += 3
                                continue
                    idx += 1
                assembly_maps.append(
                    {
                        "no": int(match.group(1)),
                        "efpd": float(match.group(3)),
                        "power": ragged_to_matrix(power_rows),
                        "burn": ragged_to_matrix(burn_rows),
                    }
                )

        idx += 1

    if SKIP_FIRST_MASTER_RESULT:
        # The first MASTER result is an unrodded state in these validation outputs.
        if rod_rows:
            rod_rows = rod_rows[1:]
        if peak_rows:
            peak_rows = peak_rows[1:]
        if assembly_maps:
            assembly_maps = assembly_maps[1:]

    rod = {
        "efpd": [row[2] for row in rod_rows],
        "position": {
            name: [row[REF_ROD_START_COL + i] for row in rod_rows]
            for i, name in enumerate(REF_ROD_ORDER)
        },
    }
    peak = {
        "efpd": [row[2] for row in peak_rows],
        "ao": [row[3] for row in peak_rows],
        "fqp": [row[6] for row in peak_rows],
    }
    return {"rod": rod, "peak": peak, "assembly": assembly_maps}


def ragged_to_matrix(rows: list[list[float]]) -> np.ndarray:
    nrow = len(rows)
    ncol = max((len(row) for row in rows), default=0)
    matrix = np.full((nrow, ncol), np.nan)
    for row, values in enumerate(rows):
        matrix[row, : len(values)] = values
    return matrix


def read_rasbery_h5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        efpd = h5["summary/efpd"][()]
        local_efpd = efpd - efpd[0]
        ao = h5["summary/ao"][()]
        fqp = h5["summary/fqp"][()]

        groups = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in h5["rods/groups"][()]
        ]
        insertions = h5["rods/insertions"][()]
        step_keys = sorted(h5["steps"].keys())
        assembly_power = [h5[f"steps/{step}/assembly/power"][()] for step in step_keys]
        assembly_burn = [h5[f"steps/{step}/assembly/burn"][()] for step in step_keys]
        ijtola = h5["geometry/ijtola"][()]
        nxa = int(h5["geometry/nxa"][()])

    group_index = {name: i for i, name in enumerate(groups)}
    position = {}
    for name in R_BANK:
        if name in group_index:
            position[name] = 240.0 - insertions[:, group_index[name]]

    return {
        "efpd": local_efpd,
        "ao": ao,
        "fqp": fqp,
        "rod": position,
        "assembly_power": assembly_power,
        "assembly_burn": assembly_burn,
        "ijtola": ijtola.reshape((-1, nxa)),
    }


def rasbery_matrix(flat: np.ndarray, ijtola: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    matrix = np.full(ref_shape, np.nan)
    for row in range(ref_shape[0]):
        for col in range(ref_shape[1]):
            if row >= ijtola.shape[0] or col >= ijtola.shape[1]:
                continue
            la = int(ijtola[row, col])
            if la >= 0:
                matrix[row, col] = flat[la]
    return matrix


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )


def save_line_plot(path: Path, title: str, ylabel: str, series: list[tuple[object, object, str, str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for x, y, label, color, linestyle in series:
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Cycle EFPD")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.86", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heatmap_triplet(path: Path, title: str, ref: np.ndarray, ras: np.ndarray, label: str) -> None:
    mask = ~np.isfinite(ref)
    diff = ras - ref
    ras = ras.copy()
    diff = diff.copy()
    ras[mask] = np.nan
    diff[mask] = np.nan

    values = np.concatenate([ref[np.isfinite(ref)], ras[np.isfinite(ras)]])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    dmax = float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else 1.0
    if dmax == 0.0:
        dmax = 1.0

    cmap_main = plt.get_cmap("viridis").copy()
    cmap_diff = plt.get_cmap("coolwarm").copy()
    cmap_main.set_bad("white")
    cmap_diff.set_bad("white")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), constrained_layout=True)
    panels = [
        ("MASTER", ref, cmap_main, None, vmin, vmax),
        ("RASBERY", ras, cmap_main, None, vmin, vmax),
        ("Difference", diff, cmap_diff, TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax), None, None),
    ]

    for ax, (subtitle, data, cmap, norm, lo, hi) in zip(axes, panels):
        image = ax.imshow(data, cmap=cmap, norm=norm, vmin=lo, vmax=hi)
        ax.set_title(subtitle)
        ax.set_xticks(range(data.shape[1]), labels=["E", "F", "G", "H", "J"][: data.shape[1]])
        ax.set_yticks(range(data.shape[0]), labels=[str(5 + i) for i in range(data.shape[0])])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if np.isfinite(data[row, col]):
                    ax.text(col, row, f"{data[row, col]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)

    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def nearest_index(values: object, target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - target)))


def reference_assembly_for_state(assembly_maps: list[dict[str, object]], state: str) -> dict[str, object]:
    efpds = np.asarray([float(item["efpd"]) for item in assembly_maps])
    if state == "BOC":
        target = float(np.min(efpds))
    elif state == "EOC":
        target = float(np.max(efpds))
    else:
        target = 0.5 * (float(np.min(efpds)) + float(np.max(efpds)))

    candidates = np.flatnonzero(np.isclose(np.abs(efpds - target), np.min(np.abs(efpds - target))))
    return assembly_maps[int(candidates[-1])]


def rasbery_assembly_for_efpd(ras: dict[str, object], target_efpd: float) -> int:
    efpd = np.asarray(ras["efpd"], dtype=float)
    nstep = len(ras["assembly_power"])
    if len(efpd) != nstep:
        efpd = efpd[:nstep]
    return nearest_index(efpd, target_efpd)


def plot_cycle(cycle: int) -> None:
    ref = parse_reference_sum(REF_DIR / f"depf_{cycle:02d}.sum")
    ras = read_rasbery_h5(CASE_DIR / f"i-SMR_CY{cycle:02d}_out.h5")

    prefix = PLOT_DIR / f"cy{cycle:02d}"

    rod_series = []
    colors = {"R4": "#b91c1c", "R3": "#1d4ed8", "R2": "#c2410c", "R1": "#15803d"}
    for name in R_BANK:
        ref_pos = ref["rod"]["position"].get(name, [])
        if len(ref_pos) > 0:
            rod_series.append((ref["rod"]["efpd"], ref_pos, f"MASTER {name}", colors[name], "--"))
        if name in ras["rod"]:
            rod_series.append((ras["efpd"], ras["rod"][name], f"RAS {name}", colors[name], "-"))

    save_line_plot(
        prefix.with_name(prefix.name + "_rod_rbank.png"),
        f"Cycle {cycle:02d} R-bank Rod Position",
        "Rod position (cm)",
        rod_series,
    )

    save_line_plot(
        prefix.with_name(prefix.name + "_fqp.png"),
        f"Cycle {cycle:02d} FQP",
        "FQP",
        [
            (ref["peak"]["efpd"], ref["peak"]["fqp"], "MASTER", "#111111", "--"),
            (ras["efpd"], ras["fqp"], "RASBERY", "#003f8c", "-"),
        ],
    )

    save_line_plot(
        prefix.with_name(prefix.name + "_ao.png"),
        f"Cycle {cycle:02d} AO",
        "AO",
        [
            (ref["peak"]["efpd"], ref["peak"]["ao"], "MASTER", "#111111", "--"),
            (ras["efpd"], ras["ao"], "RASBERY", "#003f8c", "-"),
        ],
    )

    if ref["assembly"]:
        for state in ASSEMBLY_STATES:
            ref_map = reference_assembly_for_state(ref["assembly"], state)
            ref_power = ref_map["power"]
            ref_burn = ref_map["burn"]
            ras_idx = rasbery_assembly_for_efpd(ras, float(ref_map["efpd"]))
            ras_power = rasbery_matrix(ras["assembly_power"][ras_idx], ras["ijtola"], ref_power.shape)
            ras_burn = rasbery_matrix(ras["assembly_burn"][ras_idx], ras["ijtola"], ref_burn.shape)

            power_path = prefix.with_name(prefix.name + f"_{state.lower()}_radial_power.png")
            burn_path = prefix.with_name(prefix.name + f"_{state.lower()}_burnup.png")
            save_heatmap_triplet(
                power_path,
                f"Cycle {cycle:02d} {state} Radial Power",
                ref_power,
                ras_power,
                "Power",
            )
            save_heatmap_triplet(
                burn_path,
                f"Cycle {cycle:02d} {state} Burnup",
                ref_burn,
                ras_burn,
                "Burnup",
            )

            if state == "EOC":
                save_heatmap_triplet(
                    prefix.with_name(prefix.name + "_radial_power.png"),
                    f"Cycle {cycle:02d} EOC Radial Power",
                    ref_power,
                    ras_power,
                    "Power",
                )
                save_heatmap_triplet(
                    prefix.with_name(prefix.name + "_burnup.png"),
                    f"Cycle {cycle:02d} EOC Burnup",
                    ref_burn,
                    ras_burn,
                    "Burnup",
                )


def main() -> None:
    setup_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for cycle in range(1, 5):
        plot_cycle(cycle)
    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
