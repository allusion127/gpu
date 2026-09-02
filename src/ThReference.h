#pragma once

// A VERBATIM QUOTATION of the T/H arithmetic in XSSet.cpp, and a fixture to
// drive it -- WP22, reshaped after the 238 measurement.
//
// DELIBERATELY IN ITS OWN TRANSLATION UNIT, and this header must never be
// included by one that also includes ThKernel.h.  With both in one TU gcc
// common-subexpressions across them and changes the QUOTATION's contraction, so
// the mining would score a reference that is no longer the production one.
// CmfdOuterReference.h says the same thing and Task 4 paid for the lesson.
//
// ---------------------------------------------------------------------------
// THE SHAPE IS PART OF THE QUOTATION -- AND THE FIRST SPELLING GOT IT WRONG
// ---------------------------------------------------------------------------
//
// WHAT 238 MEASURED (block 48 of the 2026-08-30 pricing log).  With the device
// arm on and the MINED mask (0x54) the deck moved: h5diff rc=1, 866 lines, a
// different digest.  With `RASBERY_TH_FORMS=0x57` the SAME build reproduced the
// flag-off digest `1f36e75dc00ed2b4` EXACTLY -- rc=0, 0 lines.  0x57 differs
// from 0x54 in bits 0 and 1 alone: TH_LERP_X0 and TH_LERP_X1, the two x-lerps of
// milk::Table::Get.  The host DOES fuse them; the mining said it does not.
//
// WHY THE MINING SAID THAT, AND IT IS NOT WHAT THIS FILE FIRST GUESSED.  The
// arithmetic was quoted correctly -- it is character-for-character
// milk::Table::Get.  The suspicion was the CALL GRAPH, because which multiply gcc
// folds into an add is decided per inlining context and this file's first
// spelling gave the x-lerps a context production does not have (external-linkage
// `refTableGet` / `refGetTfuel` with out-of-line bodies, and a PER-CHANNEL
// `refChannelSweep` where SolveTH holds the channel loop itself).  That shape is
// wrong and is fixed below -- but it was MEASURED not to be the cause: with the
// operand repair alone and the old shape, the descent mines 0x57 with a zero
// residual and identical scores.
//
// THE CAUSE WAS THAT THE FIXTURE NEVER REACHED THE TWO SITES.  scoreMask does not
// read `Fixture::node_power` -- nothing does.  It derives node power from
// xskf/phif/vol through the shipped fold, exactly as XSSet::UpdateTH does, and
// hands THAT to the sweep.  buildFixture set `norm` to 1.0e-3 where SolveTH's is
// of order 1e2, so every one of the 1,280 fuel nodes came out at `lpd < 0.03`
// W/cm, every tf query clamped to the first LPD knot, and `fx` was 0 at all of
// them.  With the x-lerps unreachable, 0x54 and 0x57 BOTH scored zero: the
// coordinate descent returned whichever bit pattern its seed started from, and
// `mineStable` reported sound because the residual really was zero.  A second,
// smaller version of the same fault sat in the tf table's burnup axis, which ran
// 0..60000 (MWd/tHM) against a query in GWd/tHM, pinning `fy` into [0, 0.012] and
// attenuating TH_LERP_X1 below the final rounding -- with `norm` fixed but the
// axis still wrong, the mining returned 0x55.
//
// SO THE CHECK THAT WAS MISSING IS NOW THERE: thmine::dontCareMask asks, of every
// site, whether the fixture can tell its alternative forms apart, and
// ThFormMiner.cpp warns when the answer is anything but TH_HAVG.  "Residual zero"
// and "the reference could distinguish the alternatives" are different facts, and
// only the second one makes a mined mask a measurement.
//
// SO THE QUOTATION BELOW IS THE CALL GRAPH, not just the expressions:
//
//     XSSet::UpdateTH        the power fold                    -> refNodePower
//       -> XSSet::SolveTH    channel loop + 3 axial loops      -> refSolveTH
//            -> XSSet::GetTmod/GetDmod  (class-body inline)    -> getTmod/getDmod
//            -> XSSet::GetTfuel         (class-body inline)    -> getTfuel
//                 -> milk::Table::Get   (class-body inline)    -> tableGet
//       -> the under-relaxation blend                          -> refRelax
//
// getTmod / getDmod / getTfuel / tableGet are `inline` in an anonymous namespace
// in ThReference.cpp and are deliberately NOT declared here: an external
// declaration is what made gcc keep an out-of-line body last time, and nothing
// outside the quotation has any business calling half of it.  refSolveTH carries
// the channel loop for the same reason SolveTH does.
//
// WHAT IS STILL NOT QUOTED, WRITTEN DOWN SO IT CAN BE CHECKED RATHER THAN
// DISCOVERED:
//
//   * SolveTH's PREAMBLE -- the serial total_power fold, the two GetHmod calls
//     and the total_area loop.  Those resolve scalars (inlet_h, norm,
//     flow_per_channel) that the fixture SUPPLIES, exactly as the device arm's
//     ThView supplies them, so quoting them would change the operands rather
//     than the shape.  They put two more inlined Table::Get copies in the real
//     function; that is a named remaining difference, not a claim of identity.
//   * the `#pragma omp parallel for schedule(static)` on UpdateTH's power fold.
//     refNodePower is a plain loop.  The 238-validated mask has TH_POWER_ACC
//     CLEAR and the plain loop mines it clear, so the two agree today; if that
//     bit ever moves, this is the first line to read.
//   * anything that is not arithmetic: the warning print, the scratch-vector
//     resizing, the Geometry accessor calls.  A quotation of those would be a
//     second implementation of them, which is the thing this file exists to
//     avoid being.
//
// NOTHING IN THIS FILE MAY BE "TIDIED".

#include <cstddef>
#include <vector>

namespace thref {

/// One table, in the shape milk::Table stores it (row-major over y then x).
struct Table {
    std::vector<double> x, y, v;
    int                 nx = 0, ny = 0;
};

/// A deck-shaped operand set: small enough to mine in microseconds, wide enough
/// that every branch of the quoted bodies is reached.
struct Fixture {
    int nxy = 0, nz = 0, nxyz = 0, ng = 0, kbc = 0, kec = 0;

    double pressure             = 0.0;
    double inlet_h              = 0.0;
    double h_table_max          = 0.0;
    double norm                 = 0.0;
    double flow_per_channel     = 0.0;
    double fuel_temp_rise_scale = 0.0;
    double th_relaxation        = 0.0;

    std::vector<double> xskf, phif, vol, hmesh_x, hmesh_y, hz;
    std::vector<int>    burn;
    std::vector<double> node_power;
    std::vector<double> tful_old, tmod_old, dmod_old;
    std::vector<double> tful, tmod, dmod;

    Table mod_t, mod_rho, tf;
};

/// Build the fixture.  Deterministic, no I/O: the operands are a reproducible
/// pseudo-random field over ranges the real deck occupies, including channels
/// driven past the table ceiling so the clamp branch is scored too.
Fixture buildFixture(int nxy, int nz, int kbc, int kec);

// --- the quotations -------------------------------------------------------
//
// Each takes the fixture and writes into caller-owned output, so the miner can
// run the shipped body over the same operands and compare bit patterns.
//
// THREE ENTRY POINTS, ONE PER PRODUCTION FUNCTION.  There is no entry point for
// one channel and none for the table interpolation, because production has no
// such function; giving them one is what pinned the wrong mask.

/// The UpdateTH per-node power fold, quoted.  `out` is [nxyz].
void refNodePower(const Fixture& f, double* out);

/// The water-property table ceiling triple SolveTH accumulates over its WHOLE
/// channel loop: how many nodes ran off the enthalpy axis, the worst enthalpy
/// and the node it happened at, under the host's FIRST-WINS tie-break in
/// ascending (channel, plane) order.
struct Overflow {
    int    count = 0;
    double worst = 0.0;
    int    node  = -1;
};

/// XSSet::SolveTH, quoted -- THE WHOLE FUNCTION, channel loop included, because
/// the channel loop is part of the context gcc contracts in.  Writes into the
/// caller's tmod/dmod/tful and returns the core-wide ceiling triple.
Overflow refSolveTH(const Fixture& f, const double* node_power, double* tmod, double* dmod,
                    double* tful);

/// The under-relaxation blend, quoted.
double refRelax(double old_value, double new_value, double w);

} // namespace thref
