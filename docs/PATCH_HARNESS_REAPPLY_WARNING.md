# Do not re-apply `02_variants/patch_harness` over this tree

`Codex_consolidated_20260820/02_variants/patch_harness/` is the *input* staging
area for the patches that were already applied and receipted in
`combined_patch_receipt_20260820.md`. It is a record of what went in, not a
source you can overlay again. Three specific traps live in it.

## 1. Truncated SHA-256 round constant (`batch_light_variant`)

`patch_harness/batch_light_variant/BatchLightResult.h` still contains

```
0x4ed8aa4u
```

for `K[53]`. The constant is `0x4ed8aa4au` — the staged copy is one hex digit
short, so every digest it produces is wrong from round 53 onward.

The shipped header in this tree has the corrected value:

```
$ grep -c '0x4ed8aa4au' include/chiffon/BatchLightResult.h
1
```

`combined_patch_receipt_20260820.md` records the fix and its external
`hashlib` verification as PASS. **Re-applying the staged variant silently
reverts the receipt hashes to the broken state**, and because the failure is a
wrong-but-well-formed hex string it will not fail to compile, will not throw,
and will not be caught by any test that does not compare against an independent
SHA-256 implementation.

If you ever need to touch that file, the check is:

```
python -c "import hashlib;print(hashlib.sha256(b'abc').hexdigest())"
# ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
```

and the solver's own `Sha256FileCached()` must agree on the same bytes.

## 2. `hdf5_scope_variant_20260820/src/CudaBICGBackend.cu` predates adaptive rendezvous

That copy is from before the adaptive-wait and topology-guard patches. It is
not the shipped backend. `src/CudaBICGBackend.cu` in this tree is the
post-apply artifact the receipt hashes.

## 3. `narrow_hdf5_scope/brute_hash.py` is not an optimization

It is a SHA-256 reverse-search tool written to recover a receipt digest. It
does not belong in any build or benchmark path.

## Why this file lives here and not next to the harness

`Codex_consolidated_20260820/` is a read-only snapshot with its own
`SHA256_MANIFEST.csv`. Adding a file inside it would leave the manifest
incomplete and make the snapshot no longer self-describing, so the warning is
recorded in the working tree instead. If you want the note physically beside
the harness, copy this file there **and** regenerate `SHA256_MANIFEST.csv`.

Reference: `ANALYSIS_STRUCTURE_20260820.md` §4 "variant 관련 주의 사항" items 1,
2 and 4; §5.5 issue 4.
