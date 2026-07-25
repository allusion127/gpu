#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np


def load_isotope_names(group):
    names = group["isotope_names"][...]
    return [x.decode() if hasattr(x, "decode") else str(x) for x in names]


def step_groups(h5, node):
    root = h5["steps"] if "steps" in h5 else h5
    for key in sorted(root.keys(), key=lambda x: int(x) if x.isdigit() else -1):
        path = f"{key}/node_monitor/{node}"
        if path in root:
            yield int(key), root[path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--node", default="0")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--xs", nargs="*", default=["xstf", "xsaf", "xsnf", "xssf"])
    parser.add_argument("--reference", choices=["ref", "baseline"], default="ref")
    args = parser.parse_args()

    rows = []
    with h5py.File(args.h5, "r") as h5:
        first = True
        for step, group in step_groups(h5, args.node):
            names = load_isotope_names(group)
            density = group["isotope_density"][...]
            burn = float(group["burn"][()]) / 1000.0
            for xs_name in args.xs:
                cur_name = f"mic_{xs_name}"
                ref_name = f"{'baseline' if args.reference == 'baseline' else 'ref'}_mic_{xs_name}"
                if cur_name not in group or ref_name not in group:
                    continue
                cur = group[cur_name][...]
                ref = group[ref_name][...]
                diff = cur - ref
                rel = diff / np.maximum(np.abs(ref), 1.0e-30)
                weighted = diff * density[:, None]
                for iso, iso_name in enumerate(names):
                    group_idx = int(np.argmax(np.abs(weighted[iso])))
                    signed_macro = float(weighted[iso, group_idx])
                    abs_macro = abs(signed_macro)
                    if abs_macro <= 0.0:
                        continue
                    rows.append(
                        {
                            "burn": burn,
                            "step": step,
                            "xs": xs_name,
                            "iso": iso_name,
                            "density": density[iso],
                            "group": group_idx,
                            "signed_micro": float(diff[iso, group_idx]),
                            "signed_rel_micro_pct": float(rel[iso, group_idx] * 100.0),
                            "max_abs_micro": float(np.abs(diff[iso]).max()),
                            "max_rel_micro_pct": float(np.abs(rel[iso]).max() * 100.0),
                            "signed_macro_contrib": signed_macro,
                            "max_abs_macro_contrib": float(abs_macro),
                        }
                    )
            if first:
                first = False

    rows.sort(key=lambda r: r["max_abs_macro_contrib"], reverse=True)
    print("rank,burn,step,xs,isotope,group,density,signed_micro,signed_rel_micro_pct,signed_macro_contrib,max_abs_macro_contrib")
    for rank, row in enumerate(rows[: args.top], 1):
        print(
            f"{rank},{row['burn']:.3f},{row['step']},{row['xs']},{row['iso']},{row['group']},"
            f"{row['density']:.8e},{row['signed_micro']:.8e},"
            f"{row['signed_rel_micro_pct']:.3f},{row['signed_macro_contrib']:.8e},"
            f"{row['max_abs_macro_contrib']:.8e}"
        )


if __name__ == "__main__":
    main()
