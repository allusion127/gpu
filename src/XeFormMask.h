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

/// THE MASK THE PRODUCTION SPLIT ARM'S KERNELS ARE LAUNCHED WITH: exactly
/// `xeFormMask() & XE_SHIPPED_FORMS`, bits 0..4 and nothing else.
///
/// WHY THIS IS A SECOND FUNCTION AND NOT A COMMENT ASKING CALLERS TO MASK.  The
/// resolved mask is one number that answers TWO questions -- "how does this
/// build contract the dot and the candidate" (bits 0..4, every Xe step of the
/// RASBERY_GPU_XE arm) and "how does it contract WP7-C's normal equations"
/// (bits 5..12, RASBERY_GPU_XE_TXN only).  On 238 the mined value moved from
/// 0xd to 0xd2d when WP7-C's sites became minable, and the split arm's kernels
/// were handed the whole 0xd2d.  Their bodies happened to extract only their
/// own fields, so nothing moved; but the ARGUMENT carried bits from a channel
/// no gate had cleared for that arm, and a single future `forms != DEFAULT` or
/// `popcount(forms)` inside those kernels would have turned a TXN-only mining
/// result into a production trajectory change with no flag moved.
///
/// The receipt still prints the FULL mined and resolved value -- a mask that is
/// measured must be reported whole -- plus the two sub-masks, so a log says
/// which bits each arm actually ran under.
unsigned long long xeShippedFormMask();

} // namespace rasbery::xe
