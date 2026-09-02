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
// WP24 (review).  For SearchPolicy / presetSearchPolicy / processSearchPolicy.
// A preset row asserts FIVE search knobs as well as eight tolerances, and a
// case that resolved only the tolerance half ran boron_bracket and the staged
// search margin off the ENVIRONMENT while its receipt named the row -- see
// searchPolicy() below.
#include "Scheduler.h"
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
    /// WP24.  The NAMED convergence policy this case runs, or empty for "no
    /// preset" -- which is the pre-WP24 path and not the same thing as
    /// "strict".  Empty means the tolerances are the built-in constants and the
    /// two multipliers came from the environment; `strict` means a row was
    /// applied that happens to restate those constants and CLEARED the
    /// environment's multipliers.  Two different facts, so two different
    /// values.
    std::string preset;

    /// The row this case names, or nullptr.  ONE lookup, so nothing downstream
    /// can resolve the name a second (different) way.
    [[nodiscard]] const FidelityPresetSpec* presetSpec() const {
        return lookupFidelityPreset(preset);
    }
    /// The production tolerances this case converges to.  The built-ins when no
    /// preset is named, bit for bit.
    [[nodiscard]] SolveTolerances tolerances() const {
        return presetTolerances(presetSpec());
    }
    /// WP24 (review).  HOW THIS CASE'S CRITICAL SEARCH PROPOSES -- the other
    /// half of the row, resolved from the SAME lookup `tolerances()` uses.
    ///
    /// THE DEFECT THIS CLOSES.  SolverContext::search_policy was initialised
    /// from processSearchPolicy() and never reassigned, and that function reads
    /// RASBERY_FIDELITY -- the ENVIRONMENT.  So a GA case asking for
    /// `"fidelity_preset":"screen100"` over the evaluator socket got the row's
    /// eight tolerances and the environment's five search knobs: boron_bracket
    /// Off where the row says On, and staged_search_margin falling back to the
    /// built-in 4.0, which clips loose_keff_tol to min(1e-5 x 5, 1e-4/4) =
    /// 2.5e-5 instead of the row's intended 5e-5.  A different secant sequence,
    /// a different converged answer -- under a receipt that printed the ROW's
    /// five knobs and therefore asserted knobs the run did not use.  Worse, the
    /// case key folded the same "screen100" string from both paths, so the
    /// env-level arm and the per-case arm collided on ONE key: a wrong cache
    /// hit, which is the one direction Driver.h's kArmEnv comment says the list
    /// must never allow.
    ///
    /// NO ROW MEANS NO ROW -- environmentSearchPolicy(), NOT
    /// processSearchPolicy().
    ///
    /// THE SECOND DEFECT, WHICH THE FIRST FIX LEFT STANDING.  This used to
    /// answer processSearchPolicy() for a case with no preset, and that
    /// function resolves the row from RASBERY_FIDELITY (Scheduler.h).  So in
    /// the deployment WP24's runbook prescribes -- a process started with
    /// `--set RASBERY_FIDELITY=screen100` -- a case that CLEARED the preset
    /// got the built-in TOLERANCES (correct) and screen100's SEARCH POLICY
    /// (wrong): boron_bracket ON, staged margin 2.0.  The two ways to clear are
    /// `op:"promote"` (EvaluatorServer.h) and any `"fidelity":"strict"` request
    /// -- i.e. the ACCEPTANCE lane, the one row in a generation whose entire
    /// job is to be quotable against the production envelope, running the one
    /// lever 238 measured at Gate A 2.27 pcm and rejected on that very
    /// envelope.  And the key collided: armEnvValue() folded "" for the preset
    /// and "" for all five search knobs (the row, not the shell, was supplying
    /// them), which is byte-identical to a genuine strict run in a preset-free
    /// process whose bracket is OFF.  One key, two solves.
    ///
    /// With environmentSearchPolicy() the two are the same solve again, so the
    /// shared key is a true hit; and in a preset-free process the two functions
    /// answer the same bits, so every existing run is unchanged.
    [[nodiscard]] SearchPolicy searchPolicy() const {
        const FidelityPresetSpec* spec = presetSpec();
        return spec != nullptr ? presetSearchPolicy(spec) : environmentSearchPolicy();
    }
    /// "none" for a receipt, so a preset field is never an empty string.
    [[nodiscard]] std::string presetToken() const {
        return preset.empty() ? std::string("none") : preset;
    }

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
        // WP24.  RASBERY_FIDELITY names a ROW, and a row is a complete
        // declaration of its own space -- it replaces the three knobs rather
        // than defaulting them (FidelityPreset.h, "WHY A NAMED PRESET
        // OVERRIDES THE ENVIRONMENT").  With it unset the three reads below are
        // the pre-WP24 reads, character for character.
        f.preset = fidelityPresetEnvName();
        if (const FidelityPresetSpec* spec = lookupFidelityPreset(f.preset)) {
            f.staged_flux_mult = spec->staged_flux_mult;
            f.staged_xe_mult   = spec->staged_xe_mult;
            f.loose_settle     = spec->loose_settle;
            f.statepoint_grid  = spec->statepoint_grid;
            return f;
        }
        // An UNKNOWN name is dropped here rather than honoured: main.cpp and
        // the evaluator both refuse it up front, and a silently-kept typo would
        // reach the case key as a payload nothing in this process acts on.
        f.preset.clear();
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
    /// WP24.  `"fidelity_preset":"screen100"` -- the NAME of a row in
    /// FidelityPreset.h's table.  Set `has_preset` to ask for "no preset" (the
    /// empty string) explicitly, which is what a promotion does.
    std::string preset;
    bool        has_preset = false;
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
        return fidelity.empty() && !has_preset && !has_grid && !has_flux_mult &&
               !has_xe_mult && !has_loose_settle && promoted_from.empty();
    }
};

/// Resolve *request* against *base*.  False with *error* set when the process
/// cannot honour it -- and REFUSING is the point: the alternative is a case
/// that runs at one fidelity while its receipt claims another.
inline bool resolveCaseFidelity(const FidelityRequest& in_request, const CaseFidelity& base,
                                CaseFidelity& out, std::string& error) {
    // WP24 (review).  `"none"` IS THE OUTPUT VOCABULARY; ACCEPT IT AS INPUT.
    //
    // presetToken() writes "none" into [CASE], [PHYSICS_MODE], the light JSONL
    // and the evaluator's per-case line, because a receipt field that is
    // sometimes an empty string is a field a reader has to special-case.  But
    // the INPUT spelling for "no preset" is the empty string, so a GA that
    // round-tripped a light-result field straight back into its next request --
    // the most natural code path there is -- hit "is not a preset this build
    // knows" on the value this binary had just printed.  One alias, here, so
    // the two vocabularies cannot diverge again; `""` still means the same
    // thing and no shipped row is named `none`.
    FidelityRequest request = in_request;
    if (request.has_preset && request.preset == "none") request.preset.clear();

    out = base;
    out.declared      = request.fidelity;
    out.promoted_from = request.promoted_from;

    // 0. WP24.  THE PRESET, FIRST, because it is the only input that sets more
    //    than one knob and the explicit knobs below have to be able to override
    //    it.  A row REPLACES the staged half of *base* wholesale -- half a
    //    preset is exactly the unnamed arm the table exists to end -- and an
    //    empty `preset` with `has_preset` set is how a promotion says "no
    //    preset", which is not the same request as `"fidelity_preset":"strict"`.
    if (request.has_preset) {
        if (request.preset.empty()) {
            out.preset           = std::string();
            out.staged_flux_mult = 1.0;
            out.staged_xe_mult   = 1.0;
            out.loose_settle     = false;
        } else {
            const FidelityPresetSpec* spec = lookupFidelityPreset(request.preset);
            if (spec == nullptr) {
                error = "\"fidelity_preset\":\"" + request.preset +
                        "\" is not a preset this build knows. Use one of: " +
                        fidelityPresetNames() +
                        ". A preset is a NAMED row of src/FidelityPreset.h, not a free "
                        "set of multipliers: a receipt that named an arm this binary "
                        "cannot reproduce would be the defect the table exists to end.";
                return false;
            }
            out.preset           = spec->name;
            out.staged_flux_mult = spec->staged_flux_mult;
            out.staged_xe_mult   = spec->staged_xe_mult;
            out.loose_settle     = spec->loose_settle;
            out.statepoint_grid  = spec->statepoint_grid;
        }
    }

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
        // WP24.  A preset the REQUEST named is a contradiction the same way an
        // explicit multiplier is, and it has to be refused rather than cleared:
        // clearing would silently discard a relaxation the caller asked for by
        // name, and a preset also carries POLISH tolerances -- the numbers the
        // published answer is converged to -- which `strict` cannot honour.  A
        // preset merely INHERITED from the process is cleared, because that is
        // the promotion lane (see the header comment).
        if (request.has_preset && !request.preset.empty() &&
            request.preset != std::string("strict")) {
            error = "\"fidelity\":\"strict\" with \"fidelity_preset\":\"" + request.preset +
                    "\" contradicts itself -- a preset carries the PUBLISHED (polish) "
                    "tolerances, not only the loose stage, so it cannot be reconciled with "
                    "a strict declaration by dropping multipliers. Ask for one or the other.";
            return false;
        }
        out.preset           = request.has_preset ? request.preset : std::string();
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
