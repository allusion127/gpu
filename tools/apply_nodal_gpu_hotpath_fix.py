#!/usr/bin/env python3
"""Remove per-batch heap allocation from the nodal XS mirror path.

This is a second, fail-closed normalization pass. The first refactor revision
used a temporary std::vector to remember which XS arrays were uploaded. That
allocation sits on every nodal batch launch and can erase the latency saved by
skipping H2D copies. Store the three decisions in the already-owned Slot
instead. The script accepts only the reviewed V1 source and is idempotent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SOURCE_MARKER = "// RASBERY_NODAL_REFACTOR_V1"
HOTPATH_MARKER = "// RASBERY_NODAL_XS_MIRROR_NO_BATCH_ALLOCATION"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def validate(text: str) -> None:
    required = (
        SOURCE_MARKER,
        HOTPATH_MARKER,
        "bool pushed_xsrf = false",
        "sl.pushed_xsrf = _mirror_xs && push_xsrf;",
        "if (sl.pushed_xsrf) sl.xsrf_mirror.commit",
        "sl.pushed_xsrf = sl.pushed_xsnf = sl.pushed_xssm = false;",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"allocation-free source validation failed; missing={missing}")
    forbidden = ("PendingXsMirror", "pending_xs.reserve", "pending_xs.push_back")
    stale = [token for token in forbidden if token in text]
    if stale:
        raise RuntimeError(f"per-batch allocation remains; tokens={stale}")

    sync = text.find("const cudaError_t src = cudaStreamSynchronize(_stream);")
    commit = text.find("if (sl.pushed_xsrf) sl.xsrf_mirror.commit", sync)
    if sync < 0 or commit < sync:
        raise RuntimeError("allocation-free mirror commit moved before the successful drain")


def apply(text: str) -> str:
    if HOTPATH_MARKER in text:
        validate(text)
        return text
    if SOURCE_MARKER not in text:
        raise RuntimeError("nodal V1 refactor marker is missing")

    text = replace_once(
        text,
        """        cuda_transfer::ByteExactMirror<double> xsrf_mirror;
        cuda_transfer::ByteExactMirror<double> xsnf_mirror;
        cuda_transfer::ByteExactMirror<double> xssm_mirror;
""",
        """        cuda_transfer::ByteExactMirror<double> xsrf_mirror;
        cuda_transfer::ByteExactMirror<double> xsnf_mirror;
        cuda_transfer::ByteExactMirror<double> xssm_mirror;
        """ + HOTPATH_MARKER + """
        bool pushed_xsrf = false, pushed_xsnf = false, pushed_xssm = false;
""",
        "slot upload flags",
    )
    text = replace_once(
        text,
        """        struct PendingXsMirror {
            int  slot = -1;
            bool xsrf = false;
            bool xsnf = false;
            bool xssm = false;
        };
        std::vector<PendingXsMirror> pending_xs;
        pending_xs.reserve(part.size());
        unsigned long long xs_h2d_bytes = 0;
        unsigned long long xs_h2d_skipped_bytes = 0;
""",
        """        unsigned long long xs_h2d_bytes = 0;
        unsigned long long xs_h2d_skipped_bytes = 0;
""",
        "temporary upload-decision vector",
    )
    text = replace_once(
        text,
        """            if (_mirror_xs)
                pending_xs.push_back({m, push_xsrf, push_xsnf, push_xssm});
""",
        """            sl.pushed_xsrf = _mirror_xs && push_xsrf;
            sl.pushed_xsnf = _mirror_xs && push_xsnf;
            sl.pushed_xssm = _mirror_xs && push_xssm;
""",
        "slot upload decisions",
    )
    text = replace_once(
        text,
        """        for (const PendingXsMirror& pending : pending_xs) {
            Slot& sl = _slot[static_cast<std::size_t>(pending.slot)];
            if (pending.xsrf) sl.xsrf_mirror.commit(sl.h_xsrf, _cnt_ng1);
            if (pending.xsnf) sl.xsnf_mirror.commit(sl.h_xsnf, _cnt_ng1);
            if (pending.xssm) sl.xssm_mirror.commit(sl.h_xssm, _cnt_ng2);
        }
""",
        """        for (int m : part) {
            Slot& sl = _slot[static_cast<std::size_t>(m)];
            if (sl.pushed_xsrf) sl.xsrf_mirror.commit(sl.h_xsrf, _cnt_ng1);
            if (sl.pushed_xsnf) sl.xsnf_mirror.commit(sl.h_xsnf, _cnt_ng1);
            if (sl.pushed_xssm) sl.xssm_mirror.commit(sl.h_xssm, _cnt_ng2);
            sl.pushed_xsrf = sl.pushed_xsnf = sl.pushed_xssm = false;
        }
""",
        "allocation-free mirror commit",
    )
    validate(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.in_place and args.output is not None:
        parser.error("choose --in-place or --output, not both")
    original = args.source.read_text(encoding="utf-8")
    patched = apply(original)
    if args.check:
        print("nodal XS mirror hot-path anchors: PASS")
        return 0
    target = args.source if args.in_place else args.output
    if target is None:
        parser.error("provide --in-place or --output")
    target.write_text(patched, encoding="utf-8", newline="\n")
    print(f"nodal XS mirror hot path: {'UNCHANGED' if patched == original else 'APPLIED'} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
