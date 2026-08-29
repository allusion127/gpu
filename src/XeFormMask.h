#pragma once

// Host-side resolution of the Xe Anderson contraction mask -- Rev.7.1 Task 13.
//
// Two declarations, and the split between them is the same one GpuFormMask.h
// draws: MINING is a measurement this binary makes of itself, RESOLUTION is
// what the run ends up using after the environment has had its say.  Kept out
// of XeKernel.h because that header is compiled by nvcc and must stay free of
// host-only machinery (getenv has no device implementation); the mask reaches
// device code as a kernel ARGUMENT, never as a call.

namespace rasbery::xe {

/// Derive this host's mask from the shipped bodies and the verbatim quotation.
///
/// TWO VERDICTS, ONE PER CHANNEL.  `sound` covers the sites the PRODUCTION
/// device Xe arm reads -- the fixed-partition dot and the candidate loop, bits
/// 0..4 -- and is the only one the resolution below is allowed to act on.
/// `algebra_sound` covers WP7-C's normal-equations sites (bits 5..12), which
/// only RASBERY_GPU_XE_TXN evaluates.  Merging the two, as 71092e2 did, lets a
/// site nobody shipped demote a mask everybody runs.
unsigned long long mineXeFormsOnThisHost(bool& sound, bool& algebra_sound);

/// The mask this process runs under: the mined value, overridden by
/// RASBERY_XE_FORMS, falling back to XE_FORMS_DEFAULT only when the mining
/// failed.  Resolved once, on first call, and one [RASBERY][FORMS] receipt line
/// is emitted then -- so a run that never touches the device Xe arm never mines
/// and never prints.
unsigned long long xeFormMask();

} // namespace rasbery::xe
