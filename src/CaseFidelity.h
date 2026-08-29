#pragma once

// WP10.3 -- FIDELITY BECOMES A PROPERTY OF THE CASE, WHERE THE BINARY CAN KEEP
// THAT PROMISE WITHOUT RE-STANDING THE ARENAS.
//
// WHAT WAS WRONG.  WP8 stage 1 wrote it down honestly and refused: "a per-case
// fidelity is a claim this process cannot keep" (EvaluatorServer.h), because
// PhysicsFidelity resolved ONCE per process from function-local statics in
// RunContract.h, and Driver::SolveLoop read RASBERY_STAGED_FLUX_TOL /
// _XE_TOL / _LOOSE_SETTLE into three MORE function-local statics of its own
// (Driver.h).  Six reads, all latched on the first case, so a mixed-fidelity
// wave was a wave in which sixty-three cases silently inherited the first
// one's convergence policy.
//
// WHAT ACTUALLY BINDS THE FIDELITY TO THE PROCESS, once those statics are
// gone.  Exactly two things, and neither is the arena:
//
//   RASBERY_GA_FEEDBACK_PASSES   caps the T/H feedback passes inside the solve
//                                and is read through BatchLightResult; the
//                                whole GA screening lane keys on it and
//                                main.cpp refuses it without a light result.
//                                Left process-level: it is a FLOOR here.
//   RASBERY_PHYSICS_FIDELITY     the operator's declaration.  Coarser-only by
//                                construction (RunContract.h), so leaving it
//                                process-level cannot dress any case up.
//
// Everything else -- the two staged-tolerance multipliers, the loose-settle
// gate, and the burnup grid -- is per-case state that costs one member each and
// nothing at all when it is not set.  The arena is untouched by all four: the
// tolerances change WHEN a loop stops, and the grid changes WHAT the deck says,
// neither of which resizes anything.  That is why this is a stage-1-compatible
// hook and `batch_width` still is not.
//
// THE RULE, and the only one worth remembering: WHAT A CASE DECLARES AND WHAT
// IT SOLVES ARE CHECKED FOR EQUALITY, BOTH WAYS.  resolveCaseFidelity() builds
// the configuration from the request, computes the fidelity that configuration
// actually produces, and refuses unless it is the word the request used.  A
// case cannot run coarser than it declared (the acceptance defect) and cannot
// run finer either (the mixed-table defect tools/exact_audit.py exists for).
// A request that declares nothing inherits the process default and is byte
// identical to what this tree did before -- feature-off is the old path.
//
// WHY A `strict` DECLARATION IS ALLOWED TO CLEAR AN A2 ENVIRONMENT.  Because
// that is the promotion lane.  `run_single_gpu_batch.DEFAULT_ENV` IS the A2 arm,
// so an evaluator standing in a production campaign has staged tolerances in
// its environment; a promoted elite has to be re-run at strict, and standing a
// second process up to do it throws away the CUDA context, the library and the
// warm cohort -- i.e. the entire reason WP8 exists.  Clearing two multipliers
// for one case costs nothing and the receipt says exactly what happened.
//
// WHY AN `A2` DECLARATION MAY NOT INVENT MULTIPLIERS.  The other direction is
// not symmetric.  `strict` is a well-defined point -- the production tolerances
// the deck states -- and A2 is a FAMILY: 50/1000 is the measured arm, 5/10 is a
// different one, and a receipt saying `A2` without saying which is a number
// that cannot be reproduced.  So A2 must find its multipliers in the process
// environment or be given them by name in the request, and a request that
// declares A2 against a strict process with no numbers is refused rather than
// guessed at.

#include "RunContract.h"
#include "StatepointGrid.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace rasbery {

/// RASBERY_STAGED_LOOSE_SETTLE, read exactly as Driver::SolveLoop read it when
/// the read was a function-local static.  ONE spelling of the truthiness test,
/// here, so the receipt and the solver cannot form two opinions.
inline bool parseStagedLooseSettle(const char* value) {
    if (value == nullptr) return false;
    const std::string s(value);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
             s == "FALSE");
}

/// The floor no case may climb above: the two fidelity inputs that stay
/// process-level.  A case's own configuration is combined with this by taking
/// the COARSER, the same rule effectivePhysicsFidelity() uses for a
/// declaration.
inline PhysicsFidelity processFidelityFloor() {
    static const PhysicsFidelity floor_value = [] {
        const char* passes = std::getenv("RASBERY_GA_FEEDBACK_PASSES");
        if (passes != nullptr && *passes != '\0' && std::atoi(passes) > 0)
            return PhysicsFidelity::FeedbackLimited;
        PhysicsFidelity declared_fidelity = PhysicsFidelity::FullExact;
        const char*     text              = declaredPhysicsFidelityText();
        if (text != nullptr && parsePhysicsFidelity(std::string(text), declared_fidelity))
            return declared_fidelity;
        return PhysicsFidelity::FullExact;
    }();
    return floor_value;
}

/// The coarser of two, the rank being the enum order (RunContract.h).
inline PhysicsFidelity coarserOf(PhysicsFidelity a, PhysicsFidelity b) {
    return (static_cast<int>(b) > static_cast<int>(a)) ? b : a;
}

/// HOW ONE CASE SOLVES.  Per Driver, carried into SolveLoop through
/// SolverContext, and into IO::ReadInput as the grid.
struct CaseFidelity {
    /// Driver::SolveLoop's staged tolerance multipliers, clamped at 1.0 the way
    /// detail::stagedMultiplier clamps them: below 1 would TIGHTEN.
    double staged_flux_mult = 1.0;
    double staged_xe_mult   = 1.0;
    /// The A2 settling-gate skip.  A staged knob but NOT a fidelity one -- with
    /// a single stage it is inert, which is why it alone never makes a run A2.
    bool loose_settle = false;
    /// Empty or "full" is the deck as written; anything else rewrites the
    /// burnup grid at deck load (StatepointGrid.h) and makes the case coarse.
    std::string statepoint_grid;
    /// The word the request used, raw, or empty.  Reported so a declaration
    /// that changed nothing is still visible.
    std::string declared;
    /// Set when the case was produced by a `promote` request: the case_key of
    /// the screening result it is the strict re-run of.  Pure provenance --
    /// nothing in the solve reads it.
    std::string promoted_from;

    [[nodiscard]] bool staged() const {
        return staged_flux_mult > 1.0 || staged_xe_mult > 1.0;
    }
    [[nodiscard]] bool coarse() const { return !spgrid::isFullGrid(statepoint_grid); }
    /// "full" for the default, so a receipt never carries an empty grid token.
    [[nodiscard]] std::string gridToken() const {
        return statepoint_grid.empty() ? std::string("full") : statepoint_grid;
    }

    /// WHAT THIS CONFIGURATION ACTUALLY SOLVES AT.  The same ranking rule the
    /// process receipt uses, over the per-case inputs plus the process floor.
    [[nodiscard]] PhysicsFidelity solved() const {
        PhysicsFidelity f = PhysicsFidelity::FullExact;
        if (staged()) f = PhysicsFidelity::StagedA2;
        // Coarse outranks A2 and is not exclusive with it: a coarse deck solved
        // under staged tolerances is coarse, and the receipt says the coarsest
        // true thing about it.
        if (coarse()) f = PhysicsFidelity::Coarse10State;
        return coarserOf(f, processFidelityFloor());
    }

    [[nodiscard]] const char* policy() const { return physicsPolicyName(solved()); }
    [[nodiscard]] const char* physicsFidelity() const {
        return physicsFidelityName(solved());
    }
    [[nodiscard]] bool acceptanceEligible() const {
        return fidelityIsAcceptanceEligible(solved());
    }
};

/// The process default: what a request that declares nothing gets, and what
/// this tree did before WP10.3 existed.  Read ONCE -- these are environment
/// facts and re-reading them per case would let a mid-run setenv split a wave.
inline const CaseFidelity& processCaseFidelity() {
    static const CaseFidelity value = [] {
        CaseFidelity f;
        f.staged_flux_mult = detail::stagedMultiplier("RASBERY_STAGED_FLUX_TOL");
        f.staged_xe_mult   = detail::stagedMultiplier("RASBERY_STAGED_XE_TOL");
        f.loose_settle     = parseStagedLooseSettle(std::getenv("RASBERY_STAGED_LOOSE_SETTLE"));
        return f;
    }();
    return value;
}

/// What one request may ask for.  Every field is OPTIONAL and every `has_`
/// flag defaults false, so an empty request resolves to processCaseFidelity()
/// unchanged.
struct FidelityRequest {
    std::string fidelity;        ///< strict | A2 | L3coarse | feedback_limited
    std::string statepoint_grid; ///< full | coarse | three | a GWd/t list
    bool        has_grid = false;
    bool        has_flux_mult = false;
    double      flux_mult     = 1.0;
    bool        has_xe_mult   = false;
    double      xe_mult       = 1.0;
    bool        has_loose_settle = false;
    bool        loose_settle     = false;
    std::string promoted_from;

    [[nodiscard]] bool empty() const {
        return fidelity.empty() && !has_grid && !has_flux_mult && !has_xe_mult &&
               !has_loose_settle && promoted_from.empty();
    }
};

/// Resolve *request* against *base*.  False with *error* set when the process
/// cannot honour it -- and REFUSING is the point: the alternative is a case
/// that runs at one fidelity while its receipt claims another.
inline bool resolveCaseFidelity(const FidelityRequest& request, const CaseFidelity& base,
                                CaseFidelity& out, std::string& error) {
    out = base;
    out.declared      = request.fidelity;
    out.promoted_from = request.promoted_from;

    // 1. The explicit knobs, applied first, so a declaration is checked against
    //    what they produced rather than the other way round.
    if (request.has_flux_mult) {
        if (!(request.flux_mult >= 1.0)) {
            error = "\"staged_flux_tol\" must be >= 1.0 (a multiplier below 1 would "
                    "TIGHTEN the tolerance, which is not a fidelity this campaign has "
                    "a lane for)";
            return false;
        }
        out.staged_flux_mult = request.flux_mult;
    }
    if (request.has_xe_mult) {
        if (!(request.xe_mult >= 1.0)) {
            error = "\"staged_xe_tol\" must be >= 1.0";
            return false;
        }
        out.staged_xe_mult = request.xe_mult;
    }
    if (request.has_loose_settle) out.loose_settle = request.loose_settle;
    if (request.has_grid) {
        std::vector<double> ignored;
        if (!spgrid::parseGrid(request.statepoint_grid, ignored, error)) return false;
        out.statepoint_grid = spgrid::isFullGrid(request.statepoint_grid)
                                  ? std::string()
                                  : request.statepoint_grid;
    }

    if (request.fidelity.empty()) return true;

    PhysicsFidelity want = PhysicsFidelity::FullExact;
    if (!parsePhysicsFidelity(request.fidelity, want)) {
        error = "\"fidelity\":\"" + request.fidelity + "\" is not a fidelity this build knows";
        return false;
    }

    // 2. What the declaration is allowed to CHANGE.  Only `strict` changes
    //    anything, and only by clearing -- see the header comment.
    if (want == PhysicsFidelity::FullExact) {
        if (request.has_flux_mult && request.flux_mult > 1.0) {
            error = "\"fidelity\":\"strict\" with \"staged_flux_tol\":" +
                    std::to_string(request.flux_mult) +
                    " contradicts itself -- a staged multiplier above 1.0 IS the A2 policy";
            return false;
        }
        if (request.has_xe_mult && request.xe_mult > 1.0) {
            error = "\"fidelity\":\"strict\" with \"staged_xe_tol\":" +
                    std::to_string(request.xe_mult) + " contradicts itself";
            return false;
        }
        out.staged_flux_mult = 1.0;
        out.staged_xe_mult   = 1.0;
        out.loose_settle     = false;
        if (!request.has_grid) out.statepoint_grid.clear();
    }
    if (want == PhysicsFidelity::StagedA2 && !out.staged()) {
        error = "\"fidelity\":\"A2\" but neither this process's environment "
                "(RASBERY_STAGED_FLUX_TOL / _XE_TOL) nor this request names a staged "
                "multiplier above 1.0. A2 is a FAMILY of convergence policies, not a "
                "point, so a receipt that said A2 without saying which one would not be "
                "reproducible: name \"staged_flux_tol\" / \"staged_xe_tol\" in the "
                "request, or start a process whose environment carries the arm.";
        return false;
    }
    if (want == PhysicsFidelity::Coarse10State && !out.coarse()) {
        error = "\"fidelity\":\"L3coarse\" but no \"statepoint_grid\" was named and this "
                "process has none by default. L3coarse is a DECK property (the burnup "
                "grid), so ask for the grid: \"statepoint_grid\":\"coarse\".";
        return false;
    }

    // 3. THE EQUALITY.  Both directions are failures, and the message says which
    //    one happened, because they are different defects with different fixes.
    const PhysicsFidelity got = out.solved();
    if (got != want) {
        const bool coarser = static_cast<int>(got) > static_cast<int>(want);
        error = std::string("\"fidelity\":\"") + request.fidelity + "\" but this case "
                "resolves to " + physicsPolicyName(got) + ". " +
                (coarser ? "A case may never solve COARSER than it declared -- that is an "
                           "approximation walking into an acceptance table. "
                         : "A case may never solve FINER than it declared either -- a "
                           "number filed in one column that was measured under another "
                           "policy is the mixing plan Sec 6.2 forbids. ") +
                "The process floor is " + physicsPolicyName(processFidelityFloor()) +
                " (RASBERY_GA_FEEDBACK_PASSES / RASBERY_PHYSICS_FIDELITY), which no "
                "request can undo.";
        return false;
    }
    return true;
}

} // namespace rasbery
