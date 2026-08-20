#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import h5py


def read_sum_keff(path):
    burns = []
    keff = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 11 and parts[0].isdigit():
            burns.append(float(parts[1]))
            keff.append(float(parts[8]))
    return burns, keff


def read_h5_keff(path):
    with h5py.File(path, "r") as h5:
        return [float(x) for x in h5["summary/keff"][...]]


def stats(values):
    n = len(values)
    return (
        sum(values) / n,
        math.sqrt(sum(v * v for v in values) / n),
        max(abs(v) for v in values),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--sum", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    burns, ref = read_sum_keff(args.sum)
    pred = read_h5_keff(args.h5)
    n = min(len(ref), len(pred))
    burns = burns[:n]
    ref = ref[:n]
    pred = pred[:n]

    # The project comparison convention is signed absolute delta-k in pcm:
    #     delta_k_pcm = (k_calc - k_reference) * 1e5
    # Keep relative delta-k/k only as a secondary diagnostic.
    raw = [(p - r) * 1.0e5 for p, r in zip(pred, ref)]
    rel = [(p - r) / r * 1.0e5 for p, r in zip(pred, ref)]
    rho = [(p - r) / (p * r) * 1.0e5 for p, r in zip(pred, ref)]

    title = f"{args.label} " if args.label else ""
    print(f"{title}n={n}")
    for name, values in [
        ("delta_k_pcm", raw),
        ("relative_delta_k_over_k_ref_x1e5_secondary", rel),
        ("reactivity_pcm", rho),
    ]:
        bias, rms, max_abs = stats(values)
        print(f"  {name}: bias={bias:+.1f} RMS={rms:.1f} max={max_abs:.1f}")
    print("  burnup,ref,calc,delta_k_pcm")
    for bu, r, p, e in zip(burns, ref, pred, raw):
        print(f"  {bu:g},{r:.6f},{p:.6f},{e:+.1f}")


if __name__ == "__main__":
    main()
