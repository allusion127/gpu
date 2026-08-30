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

/// The mask this process runs under.  COMPOSED FROM TWO SOURCES, one per
/// channel, and that composition is what makes RASBERY_GPU_XE_TXN=1 agree with
/// TXN=0 without a hand-typed pin:
///
///     bits 0..4  (XE_SHIPPED_FORMS)  the MINED value -- the device dot and
///                                    candidate sites, which the fixture
///                                    legitimately determines; XE_FORMS_DEFAULT
///                                    only when the mining failed;
///     bits 5..12 (XE_ALGEBRA_FORMS)  xeHostFormMask() -- the four normal-
///                                    equations sites, taken from the HOST call
///                                    site because TXN=1 has to reproduce what
///                                    that call site computes.
///
/// WHY THE MINED ALGEBRA BITS ARE DISCARDED.  On host 181 the mining answered
/// 0xd3d while Driver.h's own block ran 0xac0, and TXN=1 diverged from TXN=0
/// (1190 versus 1195 Xe steps) until somebody typed RASBERY_XE_FORMS=0xadd --
/// which is precisely `(0xd3d & 0x1f) | 0xac0`.  The mining scores the four
/// algebra sites against `xeref::refAlgebra`, a QUOTATION in another translation
/// unit; the value that has to be reproduced lives at a call site no fixture can
/// reach (src/XeFormAudit.h).  So for those four sites the host mask is the
/// ground truth and the mined bits are dropped rather than shipped.
///
/// RASBERY_XE_FORMS still wins VERBATIM, algebra bits included -- the 81-point
/// sweep in docs/WP7C_XE_TXN_20260831_KO.md section 9.6 is the procedure of
/// disagreeing with this composition, and it stays possible.  The receipt names
/// the source (`build_default_composed` or `env`) and prints both components,
/// so `0xd3d` + `0xac0` -> `0xadd` is checkable on one line.
///
/// Resolved once, on first call, and one [RASBERY][FORMS] receipt line is
/// emitted then -- so a run that never touches the device Xe arm never mines and
/// never prints.
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

/// THE MASK THE HOST'S OWN NORMAL EQUATIONS ARE SPELLED UNDER --
/// `Driver.h::TryAndersonXeStepGpu`, bits 5..12, XE_SITE_* encoding.
///
/// A THIRD FUNCTION AND NOT A THIRD READING OF THE SAME NUMBER.  xeFormMask()
/// answers "how does the DEVICE spell the algebra", and its value is MINED --
/// it is a measurement of this build machine scored against a quotation in
/// another translation unit.  This one answers "how does the HOST spell the
/// algebra", and there is no fixture that can measure it: the production call
/// site is only reachable inside the production inline, which is the whole
/// argument of src/XeFormAudit.h.  So this is a build constant
/// (XE_HOST_FORMS_DEFAULT) with an environment override, and nothing else; it
/// never mines, so a run that never touches the device Xe arm never pays for
/// it and never prints a line about it.
///
/// Resolved once, on first call.  One receipt line then:
///
///     [RASBERY][FORMS] {"mask":"XE_HOST_FORMS","value":"0x...","source":"...",
///                       "build_default":"0x...","det":n,"g0":n,"g1":n,"proj":n}
///
/// The four per-site digits are printed because a sweep over 81 combinations
/// reads them, not the hex: a log line that only said `0x6a0` would have to be
/// decoded by hand at every step of the sweep.
///
/// Bits outside XE_ALGEBRA_FORMS are masked off: a typo in the override may not
/// reach into the shipped dot/candidate channel.
unsigned long long xeHostFormMask();

} // namespace rasbery::xe
