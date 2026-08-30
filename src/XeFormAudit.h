#pragma once

// WP7-C.  Does the mined XE_FORMS mask describe the PRODUCTION Anderson
// algebra, or only the quotation it was mined against?
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS -- HOST 181, 2026-08-30
// ---------------------------------------------------------------------------
//
// `RASBERY_GPU_XE_TXN=1` claimed B0 against the current device arm: the four
// normal-equation expressions move from Driver.h (g++ -O3, -ffp-contract=fast)
// onto the device (--fmad=false, forms written out explicitly), and the mined
// mask is what is supposed to make those two the same bits.  On 181 they are
// not: TXN=0 digest 1d897e3f77204799, TXN=1 7f32414a742623b9, h5diff 854 lines,
// and 1195 versus 1190 Xe steps end to end.
//
// THE MASK WAS NOT UNSOUND, WHICH IS THE POINT.  Both arms printed the same
// `[RASBERY][FORMS]` line (0xd3d, mined, no `[WARN][FORMS]`), so
// XeFormMine.h's algebra channel reached zero mismatches from all four seeds
// and all four sites moved off the all-zero seed -- the fixture discriminates
// every one of them.  The mask reproduces `xeref::refAlgebra` exactly.  What it
// was never able to reproduce is `Driver.h::TryAndersonXeStepGpu`, because
// `refAlgebra` is a DIFFERENT FUNCTION in a DIFFERENT TRANSLATION UNIT with
// different operand provenance and different register pressure, and which
// multiply gcc folds into an add is a per-call-site decision.  No fixture can
// close that gap: the production call site is only reachable inside the
// production inline.
//
// So the gap is MEASURED instead, at the call site itself.  With
// `RASBERY_XE_FORMS_AUDIT=1`, every step of the round-tripping arm
// (RASBERY_GPU_XE=1, TXN=0) computes its gammas the way it always has and then
// hands them here, where they are recomputed through the SAME shipped body the
// device runs (`xe::xeAndersonFit`) under the SAME resolved mask, and compared
// bit for bit.  `forms_audit_mismatch: 0` on a real deck is the B0 claim,
// established on the operands the run actually produces; anything else is the
// N1 verdict with a number attached.  One run, no h5diff, no bisect.
//
// ---------------------------------------------------------------------------
// ITS OWN TRANSLATION UNIT, AND THAT IS THE WHOLE INSTRUMENT
// ---------------------------------------------------------------------------
//
// If this were an inline function in a header, gcc would be free to common
// subexpression the PRODUCTION `a * c - b * b` into the audit's recomputation
// -- and then the audit compares a value with itself and reports zero
// mismatches on every host, forever.  A false negative from an instrument whose
// only job is to detect a false claim is worse than no instrument.  XeKernel.h
// is therefore not included here; it is included by src/XeFormAudit.cpp and by
// nothing that also carries the production block.
//
// The same argument, one level up, is why src/XeAlgebraReference.cpp exists.
// This file is that argument applied to the measurement instead of to the
// reference.
//
// ---------------------------------------------------------------------------
// WHAT CHANGED WHEN THE HOST SITES WERE BARRIERED (238, 2026-08-31)
// ---------------------------------------------------------------------------
//
// The paragraph above says the production spelling "is only reachable inside
// the production inline".  That was true of a spelling gcc chose; it is no
// longer true of the spelling this tree ships.  The four host expressions in
// Driver.h::TryAndersonXeStepGpu now go through xe::xeSiteSub / xe::xeSiteAdd
// under RASBERY_XE_HOST_FORMS (XeFormMask.h::xeHostFormMask), for the reason
// docs/REGRESSION_7cfe3a4_d7b81af_20260831_KO.md section 8 gives: with them
// unbarriered, ANY change of inlining context around that arm re-rolled the
// flag-off trajectory, and 048c6c1's noinline attribute re-rolled it the wrong
// way.
//
// So this audit now compares TWO WRITTEN-DOWN SPELLINGS at the run's own
// operands -- the device's `xeFormMask()` against the host's
// `xeHostFormMask()` -- and B0 for RASBERY_GPU_XE_TXN=1 has a stateable
// precondition for the first time: the algebra channels must be the same
// number (`forms_audit_mask & XE_ALGEBRA_FORMS == forms_audit_host_mask` in
// the receipt) and `forms_audit_mismatch` must be zero on a real deck.  A
// mismatch now names which of the two it is.  What is NOT weakened: the audit
// still lives in its own translation unit, because CSE between the production
// block and the recomputation would still be a false negative -- barriered
// operands do not stop gcc from reusing an identical barriered result.

namespace rasbery::xe {

/// RASBERY_XE_FORMS_AUDIT, read once.  Off by default: the audit calls into
/// another translation unit on every Anderson step, and a run that did not ask
/// for it must be byte-for-byte the run it was before this file existed.
bool xeFormAuditEnabled();

/// Recompute one solved fit through the shipped body under the resolved mask
/// and compare it, bit for bit, with what the production block just produced.
///
/// `solved`, `gamma0`, `gamma1` and `proj` are the PRODUCTION values, passed by
/// value so this function cannot reach back into the caller's operands.  `dots`
/// is the same six-slot array the production block read.
///
/// Charges `forms_audits` always and `forms_audit_mismatch` on any disagreement
/// -- including a disagreement about `solved` itself, because the conditioning
/// test divides by a determinant one of the mined sites decides.  The first
/// mismatch prints one line naming the slot; after that only the counter moves,
/// so a run that diverges on every step does not drown its own log.
void auditAndersonFit(const double* dots, int ncol, double min_gram, bool solved,
                      double gamma0, double gamma1, double proj);

} // namespace rasbery::xe
