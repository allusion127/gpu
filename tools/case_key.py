#!/usr/bin/env python3
"""The canonical duplicate key of a case, computed from the deck (WP10.1).

THE POINT.  A GA that evaluates 2.56M candidates re-proposes the same core many
times, and two loading patterns related by a symmetry operation of the core are
the SAME physics.  The key folds that: symmetric-equivalent decks hash equal, and
anything that can move a published number -- the rest of the deck, the effective
fidelity, every trajectory-affecting environment string, the cross-section
library's CONTENT -- hashes differently.

WHY THIS FILE EXISTS BESIDE src/CaseKey.h.  The solver prints the key it used
(`[RASBERY][CASE]`), and a controller that wants to SKIP a case has to compute
the key BEFORE running anything.  Two implementations of one key is a real risk,
so they are held to one fixture by tools/test_case_key_contract.py, and this
tool can print the exact payload bytes for a byte-compare on a cache hit.

USAGE
    tools/case_key.py deck.json                       # the key
    tools/case_key.py deck.json --components          # the five component digests
    tools/case_key.py deck.json --payload             # the bytes it digested
    tools/case_key.py deck.json --json                # key + parts, machine-readable
    tools/case_key.py deck.json --no-xslib            # skip the 34 MB library read

ENVIRONMENT.  The arm knobs and RASBERY_PHYSICS_FIDELITY are read from THIS
process's environment, because that is what a launcher would export to the
solver it is about to start.  `--env NAME=VALUE` overrides one for a what-if.
RASBERY_XSLIB_DIGEST is honoured exactly as src/XSSet.cpp honours it, including
`off`, which makes the library digest EMPTY on both sides.

WHY --components EXISTS.  WP10.1's first live gate (host 181, 2026-08-30) had
the solver and this tool disagree on kngr_238.json while agreeing on two other
decks, and there was nothing to read: one 64-hex digest each, six inputs behind
it.  `--components` prints the same five keys the solver's [RASBERY][CASE]
receipt now prints -- deck_digest, env_digest, xslib_digest, warm_start,
code_sha -- so which component moved is a diff and not a bisect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "rasbery-case-key/v1"
CODE_SHA_ENV = "RASBERY_CODE_SHA"

# src/Driver.h trajectory::kArmEnv, IN ORDER -- READ FROM THE SOURCE, not copied.
#
# The order is part of the payload, so this is not a set, and a copy of it here
# would be a second answer to "which knobs can move an iteration": the day
# someone adds a knob to the C++ list and not to this one, two runs with
# different physics quietly share a cache entry.  So there is no copy.  The list
# is parsed out of src/Driver.h, which is the campaign's one maintained answer
# (the trajectory receipt's, pinned by its own contracts).  A checkout without
# that file cannot compute a key that means anything, so this refuses rather
# than falling back to a stale copy.
ARM_ENV_SOURCE = Path(__file__).resolve().parents[1] / "src" / "Driver.h"


def _read_arm_env(path: Path = ARM_ENV_SOURCE) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    anchor = "inline constexpr const char* kArmEnv[] = {"
    if anchor not in text:
        raise SystemExit(f"case_key: cannot find kArmEnv in {path}")
    block = text[text.index(anchor):]
    block = block[:block.index("};")]
    names = re.findall(r'"([A-Z_0-9]+)"', block)
    if not names:
        raise SystemExit(f"case_key: kArmEnv in {path} parsed empty")
    return names


ARM_ENV = _read_arm_env()


# src/IO.cpp NormalizeInputPath() -- READ FROM THE SOURCE, for the same reason
# the knob list is.
#
# THIS IS THE COMPONENT THAT BROKE.  The solver does not open the string the
# deck wrote: IO::ReadInput runs it through NormalizeInputPath (backslashes to
# forward slashes, and one WSL UNC prefix rewritten to the POSIX path it names)
# and then joins it to the DECK's directory lexically.  A tool that opened the
# raw string instead finds nothing for a deck authored on Windows or through the
# WSL share, records an EMPTY library digest, and publishes a key the solver
# never computes -- while every deck that names no library at all still agrees,
# which is exactly the two-of-three pattern host 181 reported.
IO_SOURCE = Path(__file__).resolve().parents[1] / "src" / "IO.cpp"


def _read_wsl_unc_prefix(path: Path = IO_SOURCE) -> str:
    text = path.read_text(encoding="utf-8-sig")
    found = re.search(r'kWslUncPrefix\[\]\s*=\s*"([^"]*)"', text)
    if not found:
        raise SystemExit(f"case_key: cannot find kWslUncPrefix in {path}")
    return found.group(1)


WSL_UNC_PREFIX = _read_wsl_unc_prefix()


def normalize_input_path(text: str) -> str:
    """src/IO.cpp NormalizeInputPath(), character for character."""
    text = text.replace("\\", "/")
    if text.startswith(WSL_UNC_PREFIX):
        return "/" + text[len(WSL_UNC_PREFIX):]
    return text


def resolve_xs_path(deck: Path, raw: str) -> str:
    """src/IO.cpp's `_xs_path` for the same deck and the same raw string.

    LEXICAL, not `Path.resolve()`: the solver uses `lexically_normal()`, which
    folds `..` WITHOUT consulting the filesystem.  Through a symlinked data
    mount the two land on different files, and then the two sides digest
    different bytes and neither is obviously wrong.
    """
    if not raw:
        return ""
    text = normalize_input_path(raw)
    if not text.startswith("/"):
        # IO::ReadInput's `_input_dir` is the deck's parent with a trailing
        # separator; joining and lexically normalising is what it does next.
        text = os.path.normpath(os.path.join(str(deck.parent), text))
    return str(text).replace("\\", "/")


def xslib_digest_policy(env) -> str:
    """src/XSSet.cpp XsLibraryDigestMode(), value for value."""
    value = env.get("RASBERY_XSLIB_DIGEST", "") or "cached"
    if value in ("0", "off"):
        return "off"
    if value == "always":
        return "always"
    return "cached"


# src/RunContract.h kFidelityTraits, in rank order (coarsest last).
FIDELITY_TRAITS = [
    ("strict", "full_exact"),
    ("A2", "staged_a2"),
    ("L3coarse", "coarse10"),
    ("feedback_limited", "feedback_limited"),
]


# ---------------------------------------------------------------------------
# Canonical value tokens -- src/CaseKey.h appendValue(), spelling for spelling
# ---------------------------------------------------------------------------
def _string_token(text: str) -> str:
    raw = text.encode("utf-8")
    return f"s{len(raw)}:{text}"


def canonical(value) -> str:
    if value is None:
        return "~"
    if value is True:
        return "T"
    if value is False:
        return "F"
    if isinstance(value, int):
        return f"i{value}"
    if isinstance(value, float):
        # %.17g: the one float spelling C++ std::format and Python both produce.
        return "d%.17g" % value
    if isinstance(value, str):
        return _string_token(value)
    if isinstance(value, list):
        return "[" + "".join(canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        # Sorted by key BYTES, so an object's spelling cannot depend on the
        # order the deck happened to write it in.
        return "{" + "".join(_string_token(k) + canonical(value[k])
                             for k in sorted(value.keys())) + "}"
    raise TypeError(f"a deck cannot contain {type(value).__name__}")


# ---------------------------------------------------------------------------
# Symmetry canonicalisation -- src/CaseKey.h canonicalCore()
# ---------------------------------------------------------------------------
def rectangular(core):
    width = max((len(r) for r in core), default=0)
    return [list(r) + ["XX"] * (width - len(r)) for r in core]


def transposed(m):
    if not m:
        return m
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]


def flipped_rows(m):
    return list(reversed(m))


def flipped_cols(m):
    return [list(reversed(r)) for r in m]


def canonical_core(core, symang: int):
    """The lexicographically smallest member of the pattern's symmetry orbit.

    The legal operations, and the argument for each, are in src/CaseKey.h.  The
    short version: a quarter map (angle 90) admits only the transpose, because
    every other square operation moves the fold corner and a quarter folded
    about a different corner is a different core; a full map (angle 360) admits
    the whole dihedral group; anything else admits nothing, and says so.
    """
    rect = rectangular(core)
    square = bool(rect) and len(rect) == len(rect[0])
    orbit = [("identity", rect)]
    if symang == 90:
        if square:
            orbit.append(("transpose", transposed(rect)))
    elif symang == 360:
        orbit.append(("rot180", flipped_rows(flipped_cols(rect))))
        orbit.append(("flip_rows", flipped_rows(rect)))
        orbit.append(("flip_cols", flipped_cols(rect)))
        if square:
            tr = transposed(rect)
            orbit.append(("transpose", tr))
            orbit.append(("rot90", flipped_cols(tr)))
            orbit.append(("rot270", flipped_rows(tr)))
            orbit.append(("antitranspose", flipped_rows(flipped_cols(tr))))
    best_op, best_map, best_text = None, None, None
    for name, m in orbit:
        text = canonical(m)
        if best_text is None or text < best_text:
            best_op, best_map, best_text = name, m, text
    return best_op, best_map


# ---------------------------------------------------------------------------
# The two halves of the key -- src/CaseKey.h deckPayload() / payloadOf()
# ---------------------------------------------------------------------------
def deck_payload(config: dict) -> tuple[str, str]:
    core = config.get("core", [])
    sym = config.get("geometry", {}).get("symmetry", {})
    symang = int(sym.get("angle", 0))
    mirror = bool(sym.get("mirror", False))
    symdiv = 1 if sym.get("center assembly divided", False) else 0
    op, canon_map = canonical_core(core, symang)

    rest = {k: v for k, v in config.items() if k not in ("core", "batch")}
    # core_op is NOT in the payload: it is HOW the canonical member was reached,
    # not WHAT the key identifies.  Folding it in would give a pattern and its
    # transpose two different keys.  See src/CaseKey.h.
    payload = (
        f"sym\t{symang}\t{1 if mirror else 0}\t{symdiv}\n"
        f"core\t{canonical(canon_map)}\n"
        f"batch\t{canonical(config['batch']) if 'batch' in config else '~'}\n"
        f"rest\t{canonical(rest)}\n"
    )
    return payload, op


def effective_fidelity(env) -> int:
    """src/RunContract.h effectivePhysicsFidelity(), rank for rank."""
    passes = env.get("RASBERY_GA_FEEDBACK_PASSES", "")
    detected = 0
    try:
        if passes and int(passes) > 0:
            detected = 3
    except ValueError:
        pass
    if detected == 0:
        for name in ("RASBERY_STAGED_FLUX_TOL", "RASBERY_STAGED_XE_TOL"):
            try:
                if float(env.get(name, "1") or "1") > 1.0:
                    detected = 1
                    break
            except ValueError:
                pass
    declared_text = env.get("RASBERY_PHYSICS_FIDELITY", "")
    for i, (policy, fidelity) in enumerate(FIDELITY_TRAITS):
        if declared_text in (policy, fidelity) and declared_text:
            # A declaration can only make the effective policy COARSER.
            detected = max(detected, i)
            break
    return detected


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _token(text: str) -> str:
    return text if text else "~"


def env_payload(env) -> str:
    """src/CaseKey.h envPayload(): the env half's bytes, newline-terminated.

    The SAME line spelling payload_of() uses below, so its digest is a digest of
    the real payload bytes rather than of a paraphrase.  The compiled half of
    tools/test_case_key_contract.py compares both against src/CaseKey.h.
    """
    return "".join(f"env\t{name}\t{_token(env.get(name, ''))}\n" for name in ARM_ENV)


def env_digest(env) -> str:
    return hashlib.sha256(env_payload(env).encode("utf-8")).hexdigest()


def payload_of(deck_digest: str, fidelity: str, policy: str,
               env, xslib_digest: str, warm_start: str) -> str:
    lines = [SCHEMA,
             f"deck\t{_token(deck_digest)}",
             f"fidelity\t{_token(fidelity)}",
             f"policy\t{_token(policy)}"]
    for name in ARM_ENV:
        lines.append(f"env\t{name}\t{_token(env.get(name, ''))}")
    lines.append(f"xslib\t{_token(xslib_digest)}")
    lines.append(f"warm_start\t{_token(warm_start)}")
    lines.append(f"code_sha\t{_token(env.get(CODE_SHA_ENV, ''))}")
    return "\n".join(lines) + "\n"


def case_key(deck: Path, env=None, xslib: bool = True, warm_start: str = "") -> dict:
    env = dict(os.environ if env is None else env)
    config = json.loads(deck.read_text(encoding="utf-8-sig"))
    dpayload, core_op = deck_payload(config)
    deck_digest = hashlib.sha256(dpayload.encode("utf-8")).hexdigest()

    # The solver resolves a relative XS path against the DECK's directory
    # (IO.cpp NormalizeInputPath + lexically_normal) and digests the file it
    # actually opened; a missing file is an empty digest there and here, not a
    # guess.  `--no-xslib` is the tool's own shortcut and is reported as such,
    # because a key computed without the library half is not the run's key.
    xs_path = resolve_xs_path(deck, config.get("data", {}).get("cross-section", ""))
    policy_name = xslib_digest_policy(env)
    xslib_digest = ""
    if not xslib:
        xslib_source = "skipped"
    elif not xs_path:
        xslib_source = "no library in deck"
    elif policy_name == "off":
        # src/XSSet.cpp: RASBERY_XSLIB_DIGEST=off returns an EMPTY digest, so
        # the solver's key carries no library provenance.  A tool that hashed
        # the file anyway would publish a key no run can produce -- and would
        # do it only for decks that name a library, which is how this hid.
        xslib_source = "RASBERY_XSLIB_DIGEST=off"
    elif not Path(xs_path).is_file():
        xslib_source = "not a file"
    else:
        xslib_digest = sha256_file(Path(xs_path))
        xslib_source = "content"

    rank = effective_fidelity(env)
    policy, fidelity = FIDELITY_TRAITS[rank]
    payload = payload_of(deck_digest, fidelity, policy, env,
                         xslib_digest, warm_start)
    return {
        "deck": str(deck),
        "key_schema": SCHEMA,
        "case_key": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "deck_digest": deck_digest,
        "core_op": core_op,
        "fidelity": fidelity,
        "policy": policy,
        "env_digest": env_digest(env),
        # THE PAYLOAD'S SPELLING, not the raw string.  src/CaseKey.h digests
        # `tokenOrTilde(xslib_digest)` and the [RASBERY][CASE] receipt prints
        # the same token, so a component whose empty case printed as `""` here
        # and `~` there would report a mismatch on every deck that names no
        # library -- a false alarm in exactly the place the real one hid.
        "xslib_digest": _token(xslib_digest),
        "xslib_digest_raw": xslib_digest,
        "xslib_path": xs_path,
        "xslib_policy": policy_name,
        "xslib_source": xslib_source,
        "warm_start_token": _token(warm_start),
        "code_sha": _token(env.get(CODE_SHA_ENV, "")),
        "warm_start": warm_start,
        "payload": payload,
        "deck_payload": dpayload,
    }


# The components the solver's [RASBERY][CASE] receipt prints under the same
# names.  ORDER IS THE PAYLOAD'S, so reading a diff top to bottom reads the key
# left to right.
COMPONENT_FIELDS = ("case_key", "key_schema", "core_op", "deck_digest",
                    "env_digest", "xslib_digest", "xslib_policy",
                    "warm_start_token", "code_sha", "fidelity", "policy")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck", type=Path)
    ap.add_argument("--payload", action="store_true",
                    help="print the exact bytes the key digests, for a byte-compare")
    ap.add_argument("--deck-payload", action="store_true",
                    help="print the deck half's bytes (what deck_digest digests)")
    ap.add_argument("--components", action="store_true",
                    help="print the component digests the [RASBERY][CASE] receipt "
                         "prints, one per line, for a direct diff")
    ap.add_argument("--json", action="store_true", help="print every part")
    ap.add_argument("--no-xslib", action="store_true",
                    help="leave the library digest empty (skips a 34 MB read)")
    ap.add_argument("--warm-start", default="",
                    help="warm-start provenance token to fold into the key")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                    help="override one environment variable for a what-if")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    for item in args.env:
        name, _, value = item.partition("=")
        env[name] = value

    result = case_key(args.deck, env=env, xslib=not args.no_xslib,
                      warm_start=args.warm_start)
    if args.components:
        for field in COMPONENT_FIELDS:
            print(f"{field}\t{result[field]}")
        print(f"xslib_path\t{result['xslib_path']}")
        print(f"xslib_source\t{result['xslib_source']}")
    elif args.deck_payload:
        sys.stdout.write(result["deck_payload"])
    elif args.payload:
        sys.stdout.write(result["payload"])
    elif args.json:
        printable = {k: v for k, v in result.items()
                     if k not in ("payload", "deck_payload")}
        json.dump(printable, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(result["case_key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
