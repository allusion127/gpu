#!/usr/bin/env python3
"""WP13.1 -- the site ledger's contract, and the negative controls for it.

WHAT THIS DEFENDS.  `[RASBERY][XFER][LEDGER]` is only worth reading if its
totals are the run's totals.  The moment one raw `cudaMemcpyAsync` or
`cudaStreamSynchronize` survives outside the wrappers, the receipt silently
becomes a subset again -- exactly the condition WP13 spent a whole document
failing to escape -- and nothing in the output says so.  So the invariant is
mechanical and absolute:

    NO source file under src/ may call cudaMemcpy / cudaMemcpyAsync /
    cudaMemcpyToSymbol / cudaStreamSynchronize / cudaDeviceSynchronize /
    cudaEventSynchronize directly.  src/XferLedger.h is the one implementation
    that may, and a site that genuinely cannot be wrapped must say so in an
    explicit allow-list entry here, with the reason.

Every check is re-run against a MUTATED copy of the source that breaks the
property it defends (the negative controls at the bottom); a check that cannot
fail is not a check.

Run:  python tools/test_xfer_ledger_contract.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# The raw CUDA entry points the ledger owns.
RAW = (
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
)
RAW_CALL = re.compile(r"(?<![\w:])(" + "|".join(RAW) + r")\s*\(")

# The wrapper implementation itself.  Nothing else is exempt wholesale.
WRAPPER_FILE = "XferLedger.h"

# Sites that CANNOT go through a wrapper.  Empty is the correct state and the
# entry format exists so that a future exception has to be argued in writing:
#   (relative path, substring that must appear on the line, reason)
ALLOW: list[tuple[str, str, str]] = []

# Files with no GPU transfer surface at all are not scanned differently; the
# scan simply finds nothing in them.
EXTS = (".cu", ".cuh", ".h", ".cpp")


# ---------------------------------------------------------------------------
# lexing: comments and string literals are NOT code
# ---------------------------------------------------------------------------

def strip_noncode(text: str) -> list[str]:
    """Return the file's lines with comments and string/char literals blanked.

    `fail("cudaMemcpy(nodal arena slot map)", rc)` is an error MESSAGE, not a
    call, and a scanner that cannot tell the two apart would either fail on
    honest code or force the messages to be reworded to please it.
    """
    out: list[str] = []
    in_block = False
    for line in text.split("\n"):
        buf = []
        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if in_block:
                if c == "*" and i + 1 < n and line[i + 1] == "/":
                    in_block = False
                    buf.append("  ")
                    i += 2
                    continue
                buf.append(" ")
                i += 1
                continue
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                buf.append(" " * (n - i))
                break
            if c == "/" and i + 1 < n and line[i + 1] == "*":
                in_block = True
                buf.append("  ")
                i += 2
                continue
            if c in "\"'":
                quote = c
                buf.append(" ")
                i += 1
                while i < n:
                    if line[i] == "\\":
                        buf.append("  ")
                        i += 2
                        continue
                    if line[i] == quote:
                        buf.append(" ")
                        i += 1
                        break
                    buf.append(" ")
                    i += 1
                continue
            buf.append(c)
            i += 1
        out.append("".join(buf))
    return out


def sources() -> list[str]:
    found = []
    for base, _dirs, files in os.walk(SRC):
        for name in sorted(files):
            if name.endswith(EXTS):
                found.append(os.path.join(base, name))
    return sorted(found)


def read(path: str) -> str:
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace").replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def scan_raw(path: str, text: str) -> list[tuple[int, str]]:
    """Raw CUDA transfer/sync calls in one file, as (line number, source)."""
    hits = []
    raw_lines = text.split("\n")
    for idx, code in enumerate(strip_noncode(text), start=1):
        m = RAW_CALL.search(code)
        if not m:
            continue
        line = raw_lines[idx - 1]
        allowed = any(
            os.path.basename(path) == os.path.basename(rel) and needle in line
            for rel, needle, _why in ALLOW
        )
        if not allowed:
            hits.append((idx, line.strip()))
    return hits


def check_no_raw_calls(files: dict[str, str]) -> list[str]:
    """1. Every cudaMemcpy*/cuda*Synchronize goes through the wrappers."""
    bad = []
    for path, text in files.items():
        if os.path.basename(path) == WRAPPER_FILE:
            continue
        for line_no, src in scan_raw(path, text):
            bad.append("%s:%d raw CUDA transfer/sync outside the wrapper: %s"
                       % (os.path.relpath(path, ROOT).replace("\\", "/"), line_no, src))
    return bad


def check_wrappers_exist(files: dict[str, str]) -> list[str]:
    """2. The wrapper header actually declares the five entry points."""
    text = files.get(os.path.join(SRC, WRAPPER_FILE))
    if text is None:
        return ["src/%s is missing" % WRAPPER_FILE]
    bad = []
    for sig in ("inline cudaError_t memcpyAsync(",
                "inline cudaError_t memcpy(",
                "inline cudaError_t streamSync(",
                "inline cudaError_t deviceSync(",
                "inline cudaError_t eventSync(",
                "inline void note("):
        if sig not in text:
            bad.append("src/%s does not declare `%s`" % (WRAPPER_FILE, sig))
    return bad


def check_wrapper_forwards(files: dict[str, str]) -> list[str]:
    """3. B0: each wrapper forwards to the SAME CUDA call it is named for.

    A wrapper that quietly dropped its copy, or synchronised a different
    stream, would be a behaviour change wearing a receipt's clothes.
    """
    text = files.get(os.path.join(SRC, WRAPPER_FILE), "")
    bad = []
    pairs = [
        ("memcpyAsync", "::cudaMemcpyAsync(dst, src, bytes, kind, stream)"),
        ("memcpy", "::cudaMemcpy(dst, src, bytes, kind)"),
        ("streamSync", "::cudaStreamSynchronize(stream)"),
        ("deviceSync", "::cudaDeviceSynchronize()"),
        ("eventSync", "::cudaEventSynchronize(ev)"),
    ]
    for name, call in pairs:
        if call not in text:
            bad.append("src/%s: %s does not forward to `%s`" % (WRAPPER_FILE, name, call))
    return bad


def check_ledger_is_opt_in(files: dict[str, str]) -> list[str]:
    """4. The per-site table costs one branch when the env var is unset."""
    text = files.get(os.path.join(SRC, WRAPPER_FILE), "")
    bad = []
    if 'std::getenv("RASBERY_XFER_LEDGER")' not in text:
        bad.append("src/%s: RASBERY_XFER_LEDGER is not read" % WRAPPER_FILE)
    if "if (ledgerEnabled()) recordSite(" not in text:
        bad.append("src/%s: the async copy path records unconditionally "
                   "(the per-site table must sit behind ledgerEnabled())" % WRAPPER_FILE)
    if "if (!ledgerEnabled()) return ::cudaStreamSynchronize(stream);" not in text:
        bad.append("src/%s: streamSync reads the clock even with the ledger off"
                   % WRAPPER_FILE)
    if "if (!ledgerEnabled()) return;" not in text:
        bad.append("src/%s: printLedgerReceipt/note is not gated on the flag"
                   % WRAPPER_FILE)
    return bad


def check_receipt_shape(files: dict[str, str]) -> list[str]:
    """5. The receipt emits totals plus the three orderings, with per-row fields."""
    text = files.get(os.path.join(SRC, WRAPPER_FILE), "")
    bad = []
    for tag in ("[RASBERY][XFER][LEDGER] ",
                "[RASBERY][XFER][LEDGER][BY_BYTES] ",
                "[RASBERY][XFER][LEDGER][BY_CALLS] ",
                "[RASBERY][XFER][LEDGER][BY_SYNC_NS] "):
        if tag not in text:
            bad.append("src/%s: receipt line `%s` is missing" % (WRAPPER_FILE, tag))
    for field in ('\\"site\\":', '\\"dir\\":', '\\"calls\\":', '\\"bytes\\":', '\\"ns\\":'):
        if field not in text:
            bad.append("src/%s: row field %s is missing" % (WRAPPER_FILE, field))
    return bad


def check_receipt_is_printed(files: dict[str, str]) -> list[str]:
    """6. All three of main.cpp's exit paths print it.

    A receipt that only the single-deck path emits would make the batch and
    evaluator arms silently unmeasurable.
    """
    text = files.get(os.path.join(SRC, "main.cpp"), "")
    # Commented out is NOT printed, so the census runs on code, not on text.
    n = sum(code.count("rasbery::xfer::printLedgerReceipt(")
            for code in strip_noncode(text))
    if n != 3:
        return ["src/main.cpp: printLedgerReceipt appears %d times, expected 3 "
                "(single, batch, evaluator)" % n]
    return []


def check_no_double_count(files: dict[str, str]) -> list[str]:
    """7. No site adds to the aggregate by hand any more.

    The wrappers own countH2D/countD2H/countSync now.  A leftover manual call
    beside a wrapped copy would double-count that site and make the aggregate
    disagree with the sum of the rows.
    """
    bad = []
    for path, text in files.items():
        if os.path.basename(path) == WRAPPER_FILE:
            continue
        for idx, code in enumerate(strip_noncode(text), start=1):
            for fn in ("countH2D(", "countD2H(", "countSync("):
                if fn in code:
                    bad.append("%s:%d calls xfer::%s by hand; the wrapper counts it"
                               % (os.path.relpath(path, ROOT).replace("\\", "/"), idx,
                                  fn[:-1]))
    return bad


def check_sites_are_literals(files: dict[str, str]) -> list[str]:
    """8. Every wrapper call names a scope, and the scope starts with a file.

    The table is keyed by POINTER, so a scope that is not a string literal
    with static storage would key rows by a stack address and scatter one site
    across the table.  Requiring the literal to begin with the file's own name
    also keeps rows self-locating in the receipt.
    """
    call = re.compile(r"xfer::(memcpyAsync|memcpy|streamSync|deviceSync|eventSync|note)\s*\(\s*"
                      r"(\"[^\"]*\"|[A-Za-z_]\w*)")
    bad = []
    for path, text in files.items():
        if os.path.basename(path) == WRAPPER_FILE:
            continue
        base = os.path.basename(path)
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        # WHOLE-FILE, not per line: the scope literal is routinely on the line
        # AFTER the call, and a per-line scan would pass every one of those
        # sites without ever having looked at it.
        for m in call.finditer(text):
            idx = text.count("\n", 0, m.start()) + 1
            arg = m.group(2)
            if not arg.startswith('"'):
                bad.append("%s:%d wrapper scope is not a string literal: %s"
                           % (rel, idx, arg))
                continue
            scope = arg.strip('"')
            if not scope.startswith(base.split(".")[0]):
                bad.append("%s:%d scope %r does not name its own file" % (rel, idx, scope))
    return bad


CHECKS = [
    ("no raw cudaMemcpy*/cuda*Synchronize outside the wrapper", check_no_raw_calls),
    ("the wrapper header declares all six entry points", check_wrappers_exist),
    ("each wrapper forwards to its own CUDA call (B0)", check_wrapper_forwards),
    ("the per-site table is opt-in on RASBERY_XFER_LEDGER", check_ledger_is_opt_in),
    ("the receipt carries totals and the three orderings", check_receipt_shape),
    ("all three main.cpp exit paths print the receipt", check_receipt_is_printed),
    ("no site double-counts the aggregate by hand", check_no_double_count),
    ("every wrapper call names its file in a literal scope", check_sites_are_literals),
]


# ---------------------------------------------------------------------------
# negative controls: break the property, prove the check notices
# ---------------------------------------------------------------------------

def mutate(files: dict[str, str], path_base: str, old: str, new: str) -> dict[str, str]:
    out = dict(files)
    for path in out:
        if os.path.basename(path) == path_base:
            assert old in out[path], "control anchor missing in %s: %r" % (path_base, old[:80])
            out[path] = out[path].replace(old, new, 1)
            return out
    raise AssertionError("control target %s not found" % path_base)


def controls(files: dict[str, str]):
    xl = "XferLedger.h"
    return [
        ("a raw cudaStreamSynchronize creeps back into a backend",
         check_no_raw_calls,
         mutate(files, "CudaPprBackend.cu",
                "bool syncStream(const char* what) {",
                "bool syncStream(const char* what) {\n"
                "        (void)cudaStreamSynchronize(stream);")),
        ("a raw cudaMemcpyAsync creeps back into a backend",
         check_no_raw_calls,
         mutate(files, "CudaCramBackend.cu",
                "    bool h2d(void* d, const void* h, size_t bytes, const char* name) {",
                "    bool h2d(void* d, const void* h, size_t bytes, const char* name) {\n"
                "        (void)cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream);")),
        ("the streamSync wrapper stops declaring itself",
         check_wrappers_exist,
         mutate(files, xl, "inline cudaError_t streamSync(",
                "inline cudaError_t streamSyncRenamed(")),
        ("memcpyAsync stops forwarding the copy (B0 broken)",
         check_wrapper_forwards,
         mutate(files, xl, "::cudaMemcpyAsync(dst, src, bytes, kind, stream)",
                "cudaSuccess")),
        ("the per-site table is recorded unconditionally",
         check_ledger_is_opt_in,
         mutate(files, xl, "if (ledgerEnabled()) recordSite(", "if (true) recordSite(")),
        ("streamSync reads the clock with the ledger off",
         check_ledger_is_opt_in,
         mutate(files, xl, "if (!ledgerEnabled()) return ::cudaStreamSynchronize(stream);",
                "// hoisted")),
        ("the BY_SYNC_NS ordering is dropped from the receipt",
         check_receipt_shape,
         mutate(files, xl, "[RASBERY][XFER][LEDGER][BY_SYNC_NS] ", "[GONE] ")),
        ("one of main.cpp's three exits stops printing it",
         check_receipt_is_printed,
         mutate(files, "main.cpp", "rasbery::xfer::printLedgerReceipt(",
                "// rasbery::xfer::printLedgerReceipt(")),
        ("a site adds to the aggregate by hand beside its wrapper",
         check_no_double_count,
         mutate(files, "CudaBICGBackend.cu",
                "    void pushOrSkip(const char* leaf,",
                "    void pushOrSkipCounted() { rasbery::xfer::countH2D(0); }\n"
                "    void pushOrSkip(const char* leaf,")),
        ("a wrapper call is given a runtime scope instead of a literal",
         check_sites_are_literals,
         mutate(files, "CudaCramBackend.cu",
                '"CudaCramBackend.cu:h2d", name', 'name, name')),
    ]


def main() -> int:
    files = {p: read(p) for p in sources()}

    failures = 0
    for label, fn in CHECKS:
        bad = fn(files)
        if bad:
            failures += 1
            print("FAIL  %s" % label)
            for line in bad[:40]:
                print("        %s" % line)
            if len(bad) > 40:
                print("        ... and %d more" % (len(bad) - 40))
        else:
            print("ok    %s" % label)

    print()
    for label, fn, mutated in controls(files):
        if fn(mutated):
            print("ok    control caught: %s" % label)
        else:
            failures += 1
            print("FAIL  control NOT caught: %s" % label)

    # A census, so the receipt's row count can be sanity-checked against the
    # source without running anything.
    print()
    per_file = {}
    call = re.compile(r"xfer::(memcpyAsync|memcpy|streamSync|deviceSync|eventSync|note)\s*\(")
    for path, text in sorted(files.items()):
        if os.path.basename(path) == WRAPPER_FILE:
            continue
        n = sum(1 for code in strip_noncode(text) if call.search(code))
        if n:
            per_file[os.path.relpath(path, ROOT).replace("\\", "/")] = n
    total = sum(per_file.values())
    print("tagged call sites: %d" % total)
    for path, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
        print("    %-34s %3d" % (path, n))

    print()
    print("FAILED (%d)" % failures if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
