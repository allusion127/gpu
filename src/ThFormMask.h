#pragma once

// Host-side resolution of the T/H contraction mask -- WP22.
//
// Two declarations, and the split between them is the one GpuFormMask.h draws:
// MINING is a measurement this binary makes of itself, RESOLUTION is what the
// run ends up using after the environment has had its say.  Kept out of
// ThKernel.h because that header is compiled by nvcc and must stay free of
// host-only machinery (getenv has no device implementation); the mask reaches
// device code as a kernel ARGUMENT, never as a call.

namespace rasbery::th {

/// Derive this host's mask from the shipped bodies and the verbatim quotation
/// (src/ThReference.cpp).  `sound` is false when no seed reached zero
/// mismatches, which is the only form in which "this build's device T/H is
/// bit-identical to its host T/H" can be disbelieved before a run.
unsigned long long mineThFormsOnThisHost(bool& sound);

/// The mask this process runs under.  Resolved ONCE, on first call, with the
/// precedence GpuFormMask.h::resolveCalibratedFormMask documents:
///
///   1. RASBERY_TH_FORMS, when it parses -- a binary built on one host and
///      validated against a reference produced on another must be able to say
///      so, and a human who typed the variable meant it;
///   2. the MINED value, when the mining is sound -- a measurement of THIS
///      binary on THIS machine beats a constant measured on another;
///   3. TH_FORMS_DEFAULT, only when the mining could not reach zero, and then
///      LOUDLY, because at that point nothing here knows the contract.
///
/// A run that never arms RASBERY_GPU_TH never calls this, so it never mines and
/// never prints a [RASBERY][FORMS] line about a feature it did not use.
unsigned long long thFormMask();

} // namespace rasbery::th
