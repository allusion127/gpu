#pragma once

// WP1(a): WHAT A CASE WRITES AND HOW A CASE SOLVES ARE TWO DIFFERENT AXES.
// Plan Sec 6.2, priority P0, risk ledger Sec 10 row 1.
//
// THE DEFECT.  main.cpp computed
//
//     const bool screening = (ga_feedback_passes > 0) || light_result;
//
// so `--result light` -- a switch that changes only what LEAVES the process --
// made the run declare `physics_mode:ga_screen_feedback_limited`,
// `screening:true`, and refuse to start without RASBERY_ALLOW_SCREENING.
// Nothing about the physics differs: full, pin-off and light run the same
// solve, the same PPR and the same feedback loops, and the campaign measured
// all three to ONE trajectory digest (814201df0583e1d2, GA evaluator plan
// Sec 2.1).  What differed was ~301.6 MB, ~12.0 MB and ~25.1 kB per case.
//
// The cost was paid twice, in opposite directions.  A strict run that wrote
// scalars was voided by the acceptance audit on `full_hdf5:false`.  And the
// harness worked around that by EXPECTING the screening receipt for a light
// chunk -- so a receipt that said "this was screening" was accepted for a run
// that was not screening at all, which is the failure mode the exact-only
// contract exists to prevent, arrived at from the other side.
//
// THE TWO AXES.
//
//   ResultMode       full | pin-off | light           include/chiffon/BatchLightResult.h
//   PhysicsFidelity  strict | A2 | L3coarse | feedback_limited   here
//
// `screening` and `acceptance_eligible` are functions of the FIDELITY alone.
// `light + strict` is legal and acceptance-eligible.  `full + L3coarse` is
// screening whatever it wrote.
//
// WHY THE FIDELITY IS DETECTED AND NOT DECLARED.  An operator-declared mode is
// a mode somebody forgets to declare, and the receipt would then be a
// confident lie.  Three of the four are derivable from the environment the
// solver itself reads:
//
//   feedback_limited   RASBERY_GA_FEEDBACK_PASSES > 0
//   A2                 RASBERY_STAGED_FLUX_TOL or _XE_TOL above 1.0, read
//                      exactly as Driver.h's SolveLoop reads them
//   strict             neither
//
// The fourth cannot be.  L3 coarse is a DECK property -- tools/make_screening_deck.py
// rewrites the burnup grid to ten statepoints and sets no environment variable
// -- and this receipt is emitted before any deck is parsed.  So it has one
// declaration channel, RASBERY_PHYSICS_FIDELITY, with a rule that keeps a
// declaration from ever flattering a run: it can only make the effective
// fidelity COARSER.  Declaring `strict` on a staged-tolerance run changes
// nothing; the raw declaration is reported beside the effective one so the
// attempt is visible.

#include "FidelityPreset.h"

#include <cstdlib>
#include <string>

namespace rasbery {

/// Plan Sec 6.2.  ORDER IS RANK, coarsest last: effectivePhysicsFidelity()
/// combines the detected and declared values by taking the higher.
enum class PhysicsFidelity : int {
    FullExact       = 0,
    StagedA2        = 1,
    Coarse10State   = 2,
    FeedbackLimited = 3
};

/// ONE TABLE, and everything else reads it.  `policy` is the campaign's
/// shorthand (it is what the receipt's `policy` field and the acceptance audit
/// key on); `physics_fidelity` is the plan Sec 6.2 spelling of the same value.
/// tools/test_result_fidelity_contract.py parses these four rows.
struct FidelityTraits {
    const char* policy;
    const char* physics_fidelity;
    bool        screening;
    bool        acceptance_eligible;
};

inline constexpr FidelityTraits kFidelityTraits[4] = {
    {"strict",           "full_exact",       false, true },
    {"A2",               "staged_a2",        false, false},
    {"L3coarse",         "coarse10",         true,  false},
    {"feedback_limited", "feedback_limited", true,  false},
};

inline const FidelityTraits& fidelityTraits(PhysicsFidelity f) {
    return kFidelityTraits[static_cast<int>(f)];
}

inline const char* physicsPolicyName(PhysicsFidelity f) { return fidelityTraits(f).policy; }
inline const char* physicsFidelityName(PhysicsFidelity f) {
    return fidelityTraits(f).physics_fidelity;
}

/// Sec 6.2: ONLY Coarse10State and FeedbackLimited are screening, whatever the
/// run wrote.  A2 is not screening -- it is a different convergence policy with
/// its own acceptance flag, and the plan is explicit that it must not be mixed
/// into a strict table either.
inline bool fidelityIsScreening(PhysicsFidelity f) { return fidelityTraits(f).screening; }
inline bool fidelityIsAcceptanceEligible(PhysicsFidelity f) {
    return fidelityTraits(f).acceptance_eligible;
}
inline bool fidelityRequiresExactRerun(PhysicsFidelity f) { return fidelityIsScreening(f); }

inline bool parsePhysicsFidelity(const std::string& text, PhysicsFidelity& out) {
    for (int i = 0; i < 4; ++i) {
        const auto f = static_cast<PhysicsFidelity>(i);
        if (text == kFidelityTraits[i].policy || text == kFidelityTraits[i].physics_fidelity) {
            out = f;
            return true;
        }
    }
    return false;
}

namespace detail {

/// Driver.h's SolveLoop reads these two as multipliers clamped at 1.0 and calls
/// the run staged when either exceeds 1.0.  This is the SAME reading, and
/// tools/test_result_fidelity_contract.py pins both so they cannot drift: a
/// receipt that re-derived the knob differently would disagree with the solver
/// about what the run was asked for, which is the one thing it must not do.
inline double stagedMultiplier(const char* name) {
    const char*  value = std::getenv(name);
    const double m     = (value != nullptr) ? std::atof(value) : 1.0;
    return (m >= 1.0) ? m : 1.0; // a multiplier below 1 would TIGHTEN
}

} // namespace detail

/// WP24.  The two staged multipliers the PROCESS is configured with, PRESET
/// INCLUDED, and there is exactly one implementation of that sentence because a
/// second one is how a receipt and a solver come to disagree about which arm
/// ran.
///
/// A NAMED PRESET REPLACES THE TWO KNOBS RATHER THAN DEFAULTING THEM.  The
/// argument is in FidelityPreset.h and it is short: `run_single_gpu_batch`'s
/// DEFAULT_ENV exports 50/1000, so a preset that merely supplied defaults would
/// mean a screen100 case solving at the A2 arm's multipliers under a receipt
/// that says screen100 -- the unnamed-arm defect this whole table exists to end.
///
/// WITH NO PRESET THESE ARE THE PRE-WP24 EXPRESSIONS, character for character,
/// so every fidelity this tree has ever detected is unchanged.
inline double processStagedFluxMult() {
    const FidelityPresetSpec* preset = lookupFidelityPreset(fidelityPresetEnvName());
    return preset != nullptr ? preset->staged_flux_mult
                             : detail::stagedMultiplier("RASBERY_STAGED_FLUX_TOL");
}
inline double processStagedXeMult() {
    const FidelityPresetSpec* preset = lookupFidelityPreset(fidelityPresetEnvName());
    return preset != nullptr ? preset->staged_xe_mult
                             : detail::stagedMultiplier("RASBERY_STAGED_XE_TOL");
}

/// What the environment actually configures.  Read once: the receipt is the
/// first reader and every later reader must agree with it.
inline PhysicsFidelity detectedPhysicsFidelity() {
    static const PhysicsFidelity detected = [] {
        const char* passes = std::getenv("RASBERY_GA_FEEDBACK_PASSES");
        if (passes != nullptr && *passes != '\0' && std::atoi(passes) > 0)
            return PhysicsFidelity::FeedbackLimited;
        // WP24.  A preset is not a fifth PhysicsFidelity -- kFidelityTraits is a
        // CLOSED four-row table that tools/case_key.py, exact_audit.py and
        // test_result_fidelity_contract.py all mirror by index.  screen100
        // resolves to StagedA2 like any other staged arm; WHICH staged arm is
        // the preset name, carried in the case key and the receipts.
        if (processStagedFluxMult() > 1.0 || processStagedXeMult() > 1.0)
            return PhysicsFidelity::StagedA2;
        return PhysicsFidelity::FullExact;
    }();
    return detected;
}

/// RASBERY_PHYSICS_FIDELITY, raw and unparsed, or nullptr.  Reported in the
/// receipt so a declaration that changed nothing is still visible.
inline const char* declaredPhysicsFidelityText() {
    const char* value = std::getenv("RASBERY_PHYSICS_FIDELITY");
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

/// The coarser of what was detected and what was declared.  A declaration can
/// only ADD an approximation; it can never take one away, so no environment can
/// make an A2 or feedback-limited run report as strict.
inline PhysicsFidelity effectivePhysicsFidelity() {
    static const PhysicsFidelity effective = [] {
        PhysicsFidelity f    = detectedPhysicsFidelity();
        const char*     text = declaredPhysicsFidelityText();
        PhysicsFidelity declared_fidelity = PhysicsFidelity::FullExact;
        if (text != nullptr && parsePhysicsFidelity(std::string(text), declared_fidelity)) {
            // COARSER ONLY.  The rank IS the enum order.
            if (static_cast<int>(declared_fidelity) > static_cast<int>(f))
                f = declared_fidelity;
        }
        return f;
    }();
    return effective;
}

/// True when RASBERY_PHYSICS_FIDELITY was set to something this build does not
/// know.  main.cpp refuses on it: a typo'd declaration is a declaration that
/// silently did not happen, and the case it exists for (L3coarse) is the one
/// case nothing else can detect.
inline bool declaredPhysicsFidelityIsUnknown() {
    const char*     declared = declaredPhysicsFidelityText();
    PhysicsFidelity ignored  = PhysicsFidelity::FullExact;
    return declared != nullptr && !parsePhysicsFidelity(std::string(declared), ignored);
}

} // namespace rasbery
