#pragma once

// WP24 -- A NAMED FIDELITY IS ONE ROW OF ONE TABLE, OR IT IS NOT REPRODUCIBLE.
//
// WHAT WAS WRONG.  The A2 arm is five environment exports
// (RASBERY_STAGED_FLUX_TOL=50, _XE_TOL=1000, _LOOSE_SETTLE=1, and whatever the
// campaign shell happened to be carrying).  Nothing names that set, so no
// receipt can say which arm produced a number, and CaseFidelity.h's "WHY AN
// `A2` DECLARATION MAY NOT INVENT MULTIPLIERS" paragraph already writes down
// the consequence: "A2 is a FAMILY: 50/1000 is the measured arm,
// 5/10 is a different one, and a receipt saying A2 without saying which is a
// number that cannot be reproduced."  A screening preset built the same way --
// six more exports, none of them named -- would repeat that defect with a wider
// blast radius, because a screening preset moves the PUBLISHED tolerances and
// not only the loose stage.
//
// SO THE TABLE IS THE ARTIFACT.  One row per named preset, every knob in the
// preset's space spelled out even where the value is the built-in default,
// because a knob a preset does not assert is a knob a campaign environment can
// assert for it.  RASBERY_SEARCH_CARRY_SLOPE is the worked example: 238
// measured it at +8.11 % outers (block 33), so a screen100 run inside a shell
// that already exported it would be EIGHT PERCENT SLOWER under a receipt saying
// screen100.  Pinning it off in the row is the whole fix.
//
// THE THREE ROWS.
//
//   strict     the production tolerances, i.e. every built-in constant this
//              header restates.  Naming it CLEARS an A2 environment, which is
//              the promotion lane CaseFidelity.h's "WHY A `strict`
//              DECLARATION IS ALLOWED TO CLEAR AN A2 ENVIRONMENT" argues for.
//   A2         the measured production screening arm, 50/1000/settle, with the
//              production tolerances untouched -- staging only ever moves WHEN
//              a loop stops on the way to an answer converged at production
//              tolerance (Driver.h `polishing`).
//   screen100  the GA screening preset.  Target: MASTER-relative |dkeff| <=
//              100 pcm and pin power error < 1 % RMS and max.
//
// WHY screen100 MOVES THE POLISH TOLERANCES AND A2 DOES NOT.  Under staging
// only a convergence reached at the POLISH tolerance can end a solve
// (Driver::SolveLoop's `polishing` transition), so the published answer is
// always the polish tolerance's answer and the staged multipliers buy outers
// without spending accuracy budget.  A 100 pcm preset therefore cannot be built out of
// multipliers at all: it has to move the production numbers, and that is the
// one thing this table does that the A2 arm never did.
//
// THE FOUR TOLERANCE DECISIONS, AND WHY THEY ARE NOT ONE DECADE EACH.
//
//   keff_tol_mult 10  (1e-6 -> 1e-5).  Eigenvalue error after exit is about
//     rho/(1-rho) x the tolerance, so one decade leaves 1-10 pcm per statepoint
//     against a 100 pcm budget.  TWO decades would put the search tolerance at
//     1e-3 = 100 pcm, i.e. the search tolerance would EQUAL the acceptance
//     envelope and the preset could not pass its own gate.
//   search_tol_mult 10 WITH AN ABSOLUTE CAP OF 1e-4.  The shipped 10:1
//     search-to-keff ratio is preserved exactly, and at the measured boron
//     slope of -5.4 pcm/ppm (Driver::SolveLoop's boron-slope comment) a 10 pcm
//     search tolerance admits ~1.9 ppm of CBC error.  THE CAP IS NOT
//     DECORATION -- see the next paragraph.
//   flux_l2_tol 1e-4  (two decades).  errl2 is a RELATIVE fission-source change
//     per sweep (BICGCMFD::wiel), so the remaining node-power error is about
//     rho/(1-rho) x errl2 = 0.01-0.1 % -- one to two orders under the 1 % pin
//     target even at two decades.
//   xe_tol 1e-5  (ONE decade, not two).  The tree has a MEASURED datum for
//     exactly this relaxation: Driver::SolveLoop's XE_EQUILIBRIUM_TOLERANCE
//     comment records four statepoints stopping at ~1e-5 instead of 1e-6 and
//     k_eff moving up to 8.9 pcm.  That is 9 % of the budget and is affordable;
//     1e-4 would be ~10x that.
//
// WHY THE SEARCH MULTIPLIER NEEDED AN ABSOLUTE CEILING BESIDE IT.  A multiplier
// scales `schedule.tolerance_search`, which IO::ReadInput fills from the DECK's
// `search_tol` (or `search_pcm_tolerance` x 1e-5) -- so x10 means 10 pcm only
// for a deck that states the built-in 1e-5.  A deck at `search_pcm_tolerance: 2`
// becomes 20 pcm and one at 5 becomes 50 pcm: half this preset's entire budget
// spent by one knob, silently, with the row's own ppm arithmetic (and the
// contract test that checks it) computing 10 pcm from the BUILT-IN and never
// looking at the deck.  And the tree's own dominant-risk finding is that this
// is already the binding knob: docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md
// records that at search_tol = 2e-5 the published k_eff is not the root but a
// function of the search PATH, and names that one mechanism as what killed
// boron_bracket, warm start and every acceptance lever it was measuring.  So
// the row carries `search_tol_cap` as an ABSOLUTE column, min()'d in after the
// scale, exactly as it already had to carry the RODCRIT cap as one: no deck can
// spend more than the 10 pcm the ppm column assumes.  0.0 means "no ceiling",
// which is the shipped behaviour and what the strict and A2 rows carry.
//
// THE COUPLING THAT HAD TO BE BROKEN BY HAND, AND WHAT PINNING IT COSTS.
// XE_OSCILLATION_FLOOR is written as 100 x XE_EQUILIBRIUM_TOLERANCE (Driver.h,
// beside the tolerance itself), so relaxing the Xe tolerance would float the
// damper's engagement floor from 1e-4 to 1e-3 -- and the documented pathology
// (APR1400 cy01 oscillating at ~1e-3 for 80+ steps) sits exactly there, so the
// damper would stop engaging on the case it was tuned for.  screen100 therefore
// pins the floor as an ABSOLUTE 1e-4, the same number every other row carries.
//
// THAT PIN IS A TRADE AND NOT A CLEAN WIN, and the trade is worth writing down
// because the invariant Driver.h actually states is a RATIO, not an absolute:
// "only an iteration still two orders of magnitude from its tolerance is stuck
// rather than finishing".  Against the production 1e-6 the floor is 100x the
// tolerance; against screen100's 1e-5 it is 10x.  The measured jitter band the
// floor exists to stay above is the i-SMR CY03 record in the same comment --
// contracted to 1.46e-6, then read 1.65 / 1.92 / 3.64e-6, i.e. up to ~3.6x the
// tolerance -- so the residual margin under screen100 is ~2.8x rather than the
// ~28x production enjoys, and a misfire below the floor is what moved CY03 and
// (through the restart chain) CY04 by several pcm for nothing.  Floating to
// 1e-3 is not the answer -- that IS the APR1400 pathology -- but ~3e-4 would
// restore ~30x while staying ~3x below it, and is the third knob to sweep.
//
// THE TRAP THAT MAKES A PRESET SILENTLY INERT, AND WHY screen100 WALKS INTO IT
// ON PURPOSE.  Scheduler::criticalSearchTolerance() returns
// max(min(tolerance_search, rodcrit_search_cap), rodcrit_search_floor) for
// RODCRIT.  The min() DISCARDS any relaxation of tolerance_search, so a preset
// that moved only the search tolerance is a complete no-op on every rod-crit
// deck (the iSMR / CY families) while its receipt names the preset.  The CAP
// and the cusping FLOOR are therefore columns of this table, so a row CAN move
// them -- and screen100 deliberately does NOT.  See the screen100 row.
//
// WHAT screen100 REFUSES TO TOUCH, on the record, because each looks free:
//
//   SEARCH_SETTLE_ITERS = 2.  Driver::SolveLoop's settling-gate comment
//     measures an unsettled sample at APR1400 BOC publishing k=1.000000 at
//     2346.04 ppm where a settled solve gives 0.998882: +112 pcm.  One
//     unsettled sample spends the ENTIRE budget.
//   TH_DOPPLER_TOLERANCE = 1e-2.  A 1e-2 relative Doppler residual on a ~300 K
//     rise is ~3 K, and at ~-2.5 pcm/K that is already ~7.5 pcm of the budget;
//     3e-2 would be ~22 pcm for a few percent of outers, and the error feeds
//     the depletion history rather than staying local.
//   RASBERY_GA_FEEDBACK_PASSES.  Any positive value latches the PROCESS floor
//     to FeedbackLimited (CaseFidelity::processFidelityFloor), which no
//     per-case request can undo and which main.cpp refuses without
//     --light-result.  It is a pass-count truncation in a different fidelity
//     lane, not a tolerance.
//   statepoint_grid.  The GA statepoint-count screening knob is a column here
//     so it is expressible, and it is EMPTY in every row: coarse() outranks
//     staged() (CaseFidelity::solved) and would make screen100 report L3coarse.
//     StatepointGrid.h also measures the 3-point grid costing MORE outers
//     (5,104) than the 35-point deck (4,609) -- cost is superlinear in step.
//
// WHY A NAMED PRESET OVERRIDES THE ENVIRONMENT RATHER THAN DEFERRING TO IT.
// The alternative -- preset supplies defaults, exported knobs win -- means a
// screen100 case inside run_single_gpu_batch.DEFAULT_ENV solves at 50/1000
// (the A2 arm's multipliers) under a receipt that says screen100.  That is the
// unnamed-arm defect again.  A named preset is a COMPLETE declaration of its
// own space; only an explicit per-request knob ("staged_flux_tol": N in the
// evaluator JSON) may override one, and resolveCaseFidelity() applies those
// after the preset so the equality check still judges the result.
//
// FEATURE-OFF IS THE OLD PATH, BY CONSTRUCTION.  With RASBERY_FIDELITY unset
// and no fidelity_preset in the request, lookupFidelityPreset() answers
// nullptr, no row is applied, every tolerance is the built-in constant it was,
// and no [RASBERY][FIDELITY] line is printed.  tools/test_fidelity_preset_contract.py
// pins the strict and A2 rows against those constants so the table cannot drift
// away from the tree it claims to restate.

#include <cstdlib>
#include <cstring>
#include <string>

namespace rasbery {

/// A tri-state for a boolean knob a preset asserts.  `Inherit` is the only
/// value that lets the environment decide, and no shipped row uses it: see the
/// carry-slope paragraph above.
enum class PresetFlag : int { Inherit = -1, Off = 0, On = 1 };

/// ONE ROW.  Every field is a number a receipt can print and a contract test
/// can pin; nothing here is derived at read time.
struct FidelityPresetSpec {
    const char* name;

    // --- the staged (loose-stage) knobs, i.e. the pre-WP24 A2 space ---------
    double staged_flux_mult;
    double staged_xe_mult;
    bool   loose_settle;

    // --- the PRODUCTION / polish tolerances, which only WP24 can move -------
    /// Multiplier on schedule.tolerance_keff (deck value, kEigvTol = 1e-6).
    /// A MULTIPLIER and not an absolute, because the deck is allowed to state
    /// its own eigenvalue tolerance and rewriting the deck would move the case
    /// key's DECK digest instead of its ENV digest (IO::ReadInput folds the
    /// canonical deck digest immediately after the parse, Driver.h ReadInput).
    double keff_tol_mult;
    /// Multiplier on schedule.tolerance_search (kCritSearchTol = 1e-5), same
    /// argument.
    double search_tol_mult;
    /// ABSOLUTE CEILING on the scaled search tolerance, applied as a min()
    /// after `search_tol_mult` and BEFORE the RODCRIT clamp; 0.0 means none,
    /// which is the shipped behaviour.
    ///
    /// WHY A MULTIPLIER ALONE WAS NOT SAFE HERE, when it is for keff_tol_mult.
    /// Both scale a DECK-STATED number, but only this one is bounded by the
    /// acceptance envelope in its own units: `search_tol_mult` x a deck at
    /// `search_pcm_tolerance: 5` is 50 pcm, half the screening budget, while
    /// the row's ppm arithmetic and the contract test that checks it both
    /// compute 10 pcm from the BUILT-IN 1e-5 and never see the deck.  The
    /// eigenvalue tolerance has no such bound -- it is spent through
    /// rho/(1-rho) and is checked as a fraction of the budget, not as the
    /// budget -- which is why it stays a bare multiplier and this does not.
    double search_tol_cap;
    /// Absolute.  Driver::CMFD_FLUX_L2_TOLERANCE.
    double flux_l2_tol;
    /// Absolute.  Driver::XE_EQUILIBRIUM_TOLERANCE.
    double xe_tol;
    /// Absolute, and deliberately NOT 100 x xe_tol -- see the coupling
    /// paragraph above.  Driver::XE_OSCILLATION_FLOOR.
    double xe_oscillation_floor;
    /// Absolute.  BICGCMFD::_epsl2, the CMFD SWEEP-loop exit inside drive()
    /// (the setEpsl2 call in Driver::Run).  Must be AT OR BELOW the resulting
    /// flux tolerance -- a sweep allowed to stop LOOSER than the outer verdict
    /// requires would make the sweep, not the preset, the thing that decides
    /// the published flux.
    ///
    /// THE MECHANISM, STATED CORRECTLY, BECAUSE TWO EARLIER DRAFTS HAD IT
    /// BACKWARDS.  BICGCMFD::drive() breaks on `errl2 < _epsl2` and otherwise
    /// exhausts `_ncmfd = 5` sweeps.  Read the outer's L2 half
    /// (`residual < flux_tol_now`) against that:
    ///
    ///   epsl2 == flux_tol  ->  break implies errl2 < flux_tol (outer L2 PASS);
    ///                          cap-exhaust implies errl2 >= flux_tol (FAIL).
    ///                          The outer test is then exactly "the sweep loop
    ///                          converged", which is the STRICTEST reading
    ///                          available and the shipped tree's own 1:1 state.
    ///   epsl2 <  flux_tol  ->  break still passes the outer test (trivially),
    ///                          AND a cap-exhausted sweep landing anywhere in
    ///                          [epsl2, flux_tol) now passes it too although
    ///                          the sweep did not converge.
    ///
    /// So a TIGHTER epsl2 makes the L2 half strictly MORE permissive, not less,
    /// and it costs inner sweeps.  On those two axes alone epsl2 == flux_tol
    /// dominates, and the earlier claim that 1e-5 kept the outer verdict from
    /// being "true by construction" was simply wrong.
    ///
    /// WHY screen100 STILL CARRIES 1e-5 AGAINST A 1e-4 FLUX TOLERANCE.  There
    /// is a real trade here, and it is a different one: a better-converged flux
    /// per outer is what lets the |dk| half of the outer verdict be met in
    /// FEWER OUTERS, and the cost model is outers x inner cost, so the sign of
    /// the total is not decidable from the ratio.  Nobody has measured it.
    /// 1e-4 is the cheaper-per-outer and stricter-verdict choice and is the
    /// FIRST knob to sweep once screen100 has a measured outer count; 1e-5 is
    /// the accuracy-conservative starting point, and it is called out in the
    /// docs as an unmeasured cost rather than defended as free.  The contract
    /// test therefore pins `epsl2 <= flux_tol` -- the invariant that survives
    /// the correction -- and deliberately does NOT forbid the sweep to 1e-4.
    double cmfd_sweep_epsl2;
    /// Absolute.  The RODCRIT clamp, Scheduler::kRodCritSearchTol.
    double rodcrit_search_cap;
    /// Absolute.  The rod-crit search floor applied when axial rod cusping is
    /// active (5e-5 in the shipped tree).
    double rodcrit_search_floor_cusping;

    // --- the search policy the preset asserts -------------------------------
    /// SearchPolicy::staged_margin; 0 leaves STAGED_SEARCH_MARGIN = 4.0.
    double     staged_search_margin;
    PresetFlag boron_bracket;
    PresetFlag carry_slope;
    PresetFlag warm_boron;
    /// SearchPolicy::max_trials; 0 = no cap (the deck's own limit stands).
    int        max_trials;

    // --- the deck transform -------------------------------------------------
    /// Empty is the deck as written.  EMPTY IN EVERY SHIPPED ROW -- the knob
    /// exists so a future screening preset can express it, not so this one can.
    const char* statepoint_grid;
};

/// The built-in production tolerances, restated once so the `strict` row and
/// the contract test have a single thing to agree with.  These MUST equal
/// Driver::CMFD_FLUX_L2_TOLERANCE, Driver::XE_EQUILIBRIUM_TOLERANCE,
/// Driver::XE_OSCILLATION_FLOOR, the setEpsl2 argument, Scheduler::kRodCritSearchTol
/// and the cusping floor; tools/test_fidelity_preset_contract.py reads both
/// sides out of the source and compares them.
/// 0.0 is "no absolute ceiling on the deck-stated search tolerance", which is
/// what the shipped tree does: nothing between IO::ReadInput and
/// Scheduler::criticalSearchTolerance() bounds it except the RODCRIT clamp.
inline constexpr double kProdSearchTolCap      = 0.0;
inline constexpr double kProdFluxL2Tol          = 1.0e-6;
inline constexpr double kProdXeTol              = 1.0e-6;
inline constexpr double kProdXeOscillationFloor = 1.0e-4;
inline constexpr double kProdCmfdSweepEpsl2     = 1.0e-6;
inline constexpr double kProdRodCritSearchCap   = 1.0e-5;
inline constexpr double kProdRodCritFloorCusp   = 5.0e-5;

/// THE TABLE.  Nothing else in this tree may spell a preset's numbers.
inline constexpr FidelityPresetSpec kFidelityPresets[] = {
    {
        /*name*/                         "strict",
        /*staged_flux_mult*/             1.0,
        /*staged_xe_mult*/               1.0,
        /*loose_settle*/                 false,
        /*keff_tol_mult*/                1.0,
        /*search_tol_mult*/              1.0,
        /*search_tol_cap*/               kProdSearchTolCap,
        /*flux_l2_tol*/                  kProdFluxL2Tol,
        /*xe_tol*/                       kProdXeTol,
        /*xe_oscillation_floor*/         kProdXeOscillationFloor,
        /*cmfd_sweep_epsl2*/             kProdCmfdSweepEpsl2,
        /*rodcrit_search_cap*/           kProdRodCritSearchCap,
        /*rodcrit_search_floor_cusping*/ kProdRodCritFloorCusp,
        /*staged_search_margin*/         0.0,
        /*boron_bracket*/                PresetFlag::Off,
        /*carry_slope*/                  PresetFlag::Off,
        /*warm_boron*/                   PresetFlag::Off,
        /*max_trials*/                   0,
        /*statepoint_grid*/              "",
    },
    // The measured production screening arm: run_single_gpu_batch.DEFAULT_ENV's
    // 50 / 1000 / 1, with EVERY production tolerance untouched.  That identity
    // is the point of the row -- it is what makes "A2 changed nothing but the
    // loose stage" a statement a test can check rather than a claim.
    {
        /*name*/                         "A2",
        /*staged_flux_mult*/             50.0,
        /*staged_xe_mult*/               1000.0,
        /*loose_settle*/                 true,
        /*keff_tol_mult*/                1.0,
        /*search_tol_mult*/              1.0,
        /*search_tol_cap*/               kProdSearchTolCap,
        /*flux_l2_tol*/                  kProdFluxL2Tol,
        /*xe_tol*/                       kProdXeTol,
        /*xe_oscillation_floor*/         kProdXeOscillationFloor,
        /*cmfd_sweep_epsl2*/             kProdCmfdSweepEpsl2,
        /*rodcrit_search_cap*/           kProdRodCritSearchCap,
        /*rodcrit_search_floor_cusping*/ kProdRodCritFloorCusp,
        /*staged_search_margin*/         0.0,
        /*boron_bracket*/                PresetFlag::Off,
        /*carry_slope*/                  PresetFlag::Off,
        /*warm_boron*/                   PresetFlag::Off,
        /*max_trials*/                   0,
        /*statepoint_grid*/              "",
    },
    // WP24.  The 100 pcm / 1 % screening preset.
    //
    // THE STAGED MULTIPLIERS ARE SMALLER THAN A2'S, AND THAT IS NOT A TYPO.
    // The multipliers scale the PRODUCTION numbers, which screen100 has already
    // moved.  A2's loose stage sits at loose_xe_tol = 1e-6 x 1000 = 1e-3; keeping
    // that ABSOLUTE loose stage on a 1e-5 base needs a multiplier of 100, not
    // 1000.  Carrying 1000 forward would put the loose Xe tolerance at 1e-2,
    // where a cascade exits after roughly one step and every polish transition
    // re-runs the whole cascade with prev_xe_change re-armed to infinity --
    // thrash, counted as staged_relapses, and the WP9-D D3 discard rule is that
    // relapses reach the same order as trials.  Same argument for flux:
    // 5 x 1e-4 = 5e-4 keeps the loose stage a useful factor above the polish
    // tolerance without making loose agreement meaningless.
    //
    // staged_search_margin 2.0 is the other half of the search change: the cap
    // loose_keff_tol = min(keff_tol x flux_mult, search_tol / margin) BINDS, and
    // at margin 4 with search_tol 1e-4 it would clip the loose keff tolerance to
    // 2.5e-5 whatever the flux multiplier said.  Margin 1.0 would put the sample
    // AT the search tolerance -- the noise the cap exists to keep out -- so 2.0
    // is the deliberate half-step and not an endpoint.
    //
    // WHAT IT BUYS, HONESTLY, AND WHAT IT SPENDS.  loose_flux_tol =
    // max(loose_keff_tol, flux_tol x staged_flux_mult) = max(2.5e-5 or 5e-5,
    // 5e-4) = 5e-4 either way, so the margin moves ONLY loose_keff_tol, and only
    // by a factor of 2.  Against that it HALVES the separation between the loose
    // sample and the tolerance the secant reads: the shipped relationship is a
    // factor 4, not 2 -- an earlier draft of this comment claimed 2 was "the
    // same relationship the shipped tree has", and it is not.  The tree's own
    // rod-crit record (Driver.h) is that a search reading k_eff off a flux
    // converged near its own tolerance bounces and spends back more outers than
    // the loosening saved.  So this is the SECOND knob to sweep once screen100
    // has a measured outer count, and reverting it to the built-in 4.0 is the
    // conservative move if the trial count comes back up.
    //
    // boron_bracket ON is the one lever 238 measured and REJECTED ONLY ON THE
    // STRICT ENVELOPE (block 33: 4,377 -> 4,300 outers, -1.76 %, bisect=3
    // confirming the bracket engaged; Gate A 2.27 pcm against a 1.905 pcm
    // screen).
    //
    // AND THE REASON THE REJECTION DOES NOT SURVIVE HERE IS NOT THAT 2.27 IS
    // 2 % OF 100.  An earlier draft of this row argued exactly that, i.e. it
    // treated the 2.27 pcm as real accuracy error spent out of the budget.
    // docs/A2_OUTER_REDUCTION_DESIGN_20260902_KO.md says otherwise: at
    // search_tol = 2e-5 (2 pcm) the published k_eff is not the root but a
    // function of the search PATH, so a 2.27 pcm Gate A delta against a
    // 1.905 pcm screen is a SEARCH-PATH ARTEFACT and not accuracy at all --
    // the same single mechanism that document names as having killed
    // boron_bracket, warm start and every other acceptance lever it measured.
    // What changes under screen100 is that the search tolerance is 10 pcm
    // (capped absolutely), so the path artefact is inside the tolerance the
    // arm is asking for rather than five times it.  Right conclusion, correct
    // reason.
    //
    // NOTE THE OPEN DEFECT IT DOES NOT RETIRE: 8xM16 + boron_bracket produced
    // 5/128 silent worker failures where 8xM8 was 128/128 clean, so screen100
    // batches are pinned to M8 until that forensics closes.
    //
    // AND THAT PIN IS PROSE.  Nothing in the binary, the evaluator or the
    // contract test stops an 8xM16 screen100 batch; the constraint lives in
    // this comment and in the runbook.  For a screening arm whose whole purpose
    // is throughput, 1.76 % of outers against a known silent-failure mode is a
    // thin bargain, and the two honest ways out are to drop the flag from the
    // row until block 34 closes or to make the batch-width constraint
    // enforceable.  It is kept ON, unenforced, because the measurement that
    // would settle it (boron_bracket-OFF at 8xM16) has never been run and
    // dropping the flag would also drop the only measured outer saving the row
    // has -- but a reader deciding whether to trust a screen100 batch should
    // know the guard is a sentence, not a check.
    //
    // THE TWO RODCRIT COLUMNS ARE AT THE PRODUCTION VALUES, DELIBERATELY, AND
    // THAT MAKES screen100 A DOCUMENTED NO-OP ON THE ROD-CRIT SEARCH.
    //
    // An earlier draft raised rodcrit_search_cap 1e-5 -> 1e-4 and the cusping
    // floor 5e-5 -> 1e-4.  Every measurement quoted anywhere in this row comes
    // from BORON decks: KNGR's 4,377 outers, the APR1400 slope, the -1.76 %
    // bracket result.  The rod-crit families (iSMR / CY) had a 10x cap
    // relaxation and a doubled cusping floor with no measurement at all -- and
    // the gate over those decks is blinder than anywhere else in the campaign:
    //
    //   delta_pcm   is the SEARCH RESIDUAL, not physics (k_eff is driven to
    //               target by construction), and is bounded by the same cap;
    //   delta_ppm   is the deck's own fixed boron on both sides, ~0 (the ROD is
    //               searched, the boron is not);
    //   delta_ao    is ADVISORY under this envelope -- and rod misposition maps
    //               primarily into AXIAL shape, so the one metric that would
    //               catch it is the one screen100 stopped failing on;
    //   Fq / Fr     are in the compare tool's DELTA_PAIRS but in NEITHER
    //               envelope's METRICS.
    //
    // So `compare_master_rasbery.py --envelope screen100` over an iSMR / CY
    // deck could print `GATE B scalars: PASS` having judged only columns that
    // cannot fail: loosest exactly where the gate is blindest.  Two things
    // follow, and both are done rather than written down.  (1) The row stops
    // relaxing the rod-crit search until somebody measures it -- screen100 then
    // solves the rod search FINER than a naive reading of its name suggests,
    // which is the safe direction, and the `resolved_entry0.search_tol` field
    // of the [RASBERY][FIDELITY] receipt prints the 1e-5 it actually used so
    // the no-op is visible rather than inferred.  (2) gate_b_envelope.report()
    // refuses to call a run SCORED when every judged column is one the caller
    // declared structurally pinned, so the hollow PASS cannot be printed even
    // if a later row does relax these.
    //
    // TO RAISE THEM: measure the rod-crit families against their own Gate B
    // first, and give that Gate B a metric that can actually fail on a rod
    // deck.  A prose guard here would be a comment; the two production values
    // below are a check.
    {
        /*name*/                         "screen100",
        /*staged_flux_mult*/             5.0,
        /*staged_xe_mult*/               100.0,
        /*loose_settle*/                 true,
        /*keff_tol_mult*/                10.0,
        /*search_tol_mult*/              10.0,
        /*search_tol_cap*/               1.0e-4,
        /*flux_l2_tol*/                  1.0e-4,
        /*xe_tol*/                       1.0e-5,
        /*xe_oscillation_floor*/         kProdXeOscillationFloor,
        /*cmfd_sweep_epsl2*/             1.0e-5,
        /*rodcrit_search_cap*/           kProdRodCritSearchCap,
        /*rodcrit_search_floor_cusping*/ kProdRodCritFloorCusp,
        /*staged_search_margin*/         2.0,
        /*boron_bracket*/                PresetFlag::On,
        /*carry_slope*/                  PresetFlag::Off,
        /*warm_boron*/                   PresetFlag::Off,
        /*max_trials*/                   0,
        /*statepoint_grid*/              "",
    },
};

inline constexpr int kFidelityPresetCount =
    static_cast<int>(sizeof(kFidelityPresets) / sizeof(kFidelityPresets[0]));

/// The row named *name*, or nullptr.  An empty or absent name is NOT an error
/// and is NOT the strict row: it is "no preset", which is the pre-WP24 path.
inline const FidelityPresetSpec* lookupFidelityPreset(const char* name) {
    if (name == nullptr || *name == '\0') return nullptr;
    for (int i = 0; i < kFidelityPresetCount; ++i)
        if (std::strcmp(kFidelityPresets[i].name, name) == 0) return &kFidelityPresets[i];
    return nullptr;
}

inline const FidelityPresetSpec* lookupFidelityPreset(const std::string& name) {
    return name.empty() ? nullptr : lookupFidelityPreset(name.c_str());
}

/// RASBERY_FIDELITY, raw, or empty.  ONE spelling of the read, so the case key,
/// the receipt and the solver cannot form three opinions about which arm ran.
inline std::string fidelityPresetEnvName() {
    const char* value = std::getenv("RASBERY_FIDELITY");
    return (value != nullptr && *value != '\0') ? std::string(value) : std::string();
}

/// True when RASBERY_FIDELITY named something this build has no row for.
/// main.cpp refuses on it, for the reason declaredPhysicsFidelityIsUnknown()
/// exists: a typo'd preset is a preset that silently did not happen, and the
/// run would then be a production-tolerance run wearing a screening name.
inline bool fidelityPresetEnvIsUnknown() {
    const std::string name = fidelityPresetEnvName();
    return !name.empty() && lookupFidelityPreset(name) == nullptr;
}

/// The names, for an error message and for the evaluator's hello line.
inline std::string fidelityPresetNames() {
    std::string out;
    for (int i = 0; i < kFidelityPresetCount; ++i) {
        if (i) out += " | ";
        out += kFidelityPresets[i].name;
    }
    return out;
}

/// WHAT ONE SOLVE CONVERGES TO.  Resolved once per case from the preset (or
/// from nothing, which is the built-ins) and carried on SolverContext, so every
/// consumer -- the outer verdict, the Xe cascade, the Anderson arming gate, the
/// device Xe request, the CMFD sweep exit -- reads ONE value and they cannot
/// disagree about what the case was asked for.
struct SolveTolerances {
    double keff_tol_mult                = 1.0;
    double search_tol_mult              = 1.0;
    double search_tol_cap               = kProdSearchTolCap;
    double flux_l2_tol                  = kProdFluxL2Tol;
    double xe_tol                       = kProdXeTol;
    double xe_oscillation_floor         = kProdXeOscillationFloor;
    double cmfd_sweep_epsl2             = kProdCmfdSweepEpsl2;
    double rodcrit_search_cap           = kProdRodCritSearchCap;
    double rodcrit_search_floor_cusping = kProdRodCritFloorCusp;
};

/// The tolerances *preset* asks for; the built-ins when it is nullptr.  A
/// default-constructed SolveTolerances IS the nullptr answer, so the no-preset
/// path multiplies nothing and compares against the same constants it always
/// did -- multiplication by exactly 1.0 is exact in IEEE-754, which is what
/// makes the feature-off identity a property of the arithmetic rather than of a
/// diff somebody remembered to run.
inline SolveTolerances presetTolerances(const FidelityPresetSpec* preset) {
    SolveTolerances t;
    if (preset == nullptr) return t;
    t.keff_tol_mult                = preset->keff_tol_mult;
    t.search_tol_mult              = preset->search_tol_mult;
    t.search_tol_cap               = preset->search_tol_cap;
    t.flux_l2_tol                  = preset->flux_l2_tol;
    t.xe_tol                       = preset->xe_tol;
    t.xe_oscillation_floor         = preset->xe_oscillation_floor;
    t.cmfd_sweep_epsl2             = preset->cmfd_sweep_epsl2;
    t.rodcrit_search_cap           = preset->rodcrit_search_cap;
    t.rodcrit_search_floor_cusping = preset->rodcrit_search_floor_cusping;
    return t;
}

} // namespace rasbery
