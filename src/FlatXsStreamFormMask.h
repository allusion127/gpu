#pragma once

// Host-side resolution of the WP23 branch-stream contraction mask and of the
// libm policy -- WP23.1.
//
// Three declarations, and the split between them is the one GpuFormMask.h
// draws: MINING is a measurement this binary makes of itself, RESOLUTION is
// what the run ends up using after the environment has had its say.  Kept out
// of FlatXsStreamKernel.h because that header is compiled by nvcc and must stay
// free of host-only machinery (getenv has no device implementation); both
// values reach device code as kernel ARGUMENTS, never as calls.

namespace rasbery::flatxs_stream {

/// Derive this host's mask from the shipped bodies and the verbatim quotation
/// (src/FlatXsStreamReference.cpp).  `sound` is false when no seed reached zero
/// mismatches, which is the only form in which "this build's device branch
/// stream is bit-identical to its host branch stream on the three lerp sites"
/// can be disbelieved before a run.
unsigned mineStreamFormsOnThisHost(bool& sound);

/// The mask this process runs under.  Resolved ONCE, on first call, with the
/// precedence GpuFormMask.h::resolveCalibratedFormMask documents:
///
///   1. RASBERY_FLATXS_STREAM_FORMS, when it parses -- a binary built on one
///      host and validated against a reference produced on another must be able
///      to say so, and a human who typed the variable meant it;
///   2. the MINED value, when the mining is sound;
///   3. kStreamFormsDefault, only when the mining could not reach zero, and then
///      LOUDLY, because at that point nothing here knows the contract.
///
/// A run that never arms RASBERY_GPU_FLATXS_STREAM never calls this, so it never
/// mines and never prints a [RASBERY][FORMS] line about a feature it did not
/// use.
unsigned streamFormMask();

/// Whether the mining behind streamFormMask() reached zero mismatches.  A
/// SECOND accessor and not a second mining: the receipt has to report the
/// soundness beside the value, and re-deriving it would be a second answer to a
/// question that already has one.  Calling it resolves the mask if that has not
/// happened yet, so `forms_source` and `forms_sound` can never disagree about
/// which run they describe.
bool streamFormsSound();

/// Which of `mined` / `env` / `build_default` the resolved mask came from, as
/// the literal string the receipt prints.
const char* streamFormsSource();

/// FS_LIBM_FAST or FS_LIBM_EXACT, from RASBERY_GPU_FLATXS_STREAM_LIBM.
///
/// `exact` is one env var away and is measurably NEARER the host than any
/// vendor pair (see FlatXsStreamExactMath.h), but it is not the default: the
/// default has to be the arm 238 has a digest for, and bit equality with the
/// flag-off path is not reachable either way until the host's own log/cbrt are
/// the ones being reproduced.
unsigned streamLibmMode();

/// "exact" or "fast", for the receipt.
const char* streamLibmName();

} // namespace rasbery::flatxs_stream
