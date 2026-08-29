#pragma once

// WP10.3 -- THE BURNUP GRID IS A DECK PROPERTY, SO IT IS APPLIED AT DECK LOAD.
//
// WHAT THIS IS.  `tools/make_screening_deck.py` rewrites the `schedule` block of
// a full depletion deck onto a coarse burnup grid and writes a second file.
// That works for a launcher that owns a directory and does not work for the
// evaluator: WP8's process answers a stream of case requests, and a request
// that wants a coarse lane cannot be made to first materialise a deck on disk
// that the controller then has to name, clean up and keep in step with the
// original.  So the SAME transform lives here, runs on the parsed deck inside
// IO::ReadInput, and is selected by one request field.
//
// WHY IT MUST RUN BEFORE THE DECK DIGEST, AND NOT AFTER.  IO::ReadInput folds
// casekey::deckPayload(config) into `_deck_key_digest` immediately after the
// parse.  A coarse case and a full case of the same candidate are DIFFERENT
// PHYSICS -- ten statepoints against thirty-five -- and if the transform ran
// after the fold they would share a case key, which is a screening answer
// served out of the cache to an acceptance request.  Running here makes the
// coarse deck the deck the key is taken of, so the two lanes cannot collide,
// with no extra field in the key and no rule anybody has to remember.
//
// WHAT IS AND IS NOT COPIED FROM make_screening_deck.py.  Copied verbatim, and
// pinned by tools/test_statepoint_grid_contract.py:
//
//   the coarse grid            0.5 1 2 4 6 8 10 13 16 GWd/t
//   the three grid             8 16 GWd/t
//   the time-key set           the keys a depletion entry states its length in
//   the entry template         the MODAL depletion entry, minus its time keys
//   the rebuild                one entry per grid point, steps=1, burnup=delta
//
// NOT copied, because they are launcher policy rather than deck arithmetic:
// `--pin-output` (the request's ResultMode already decides what is written),
// `--max-substep-burnup`, `--keep-until-boron` and `--eoc-restate`.  A request
// that needs one of those asks for a deck built by the tool and names it.
//
// THE UNTIL-BORON TAIL IS DROPPED, and that is the point of a screening lane:
// `"until boron ppm"` re-queues the terminal entry at runtime until the
// converged critical boron reaches the target, so the statepoint count -- and
// the wall -- becomes a function of the candidate.  A screening arm must have a
// FIXED cost or its throughput number means nothing.
//
// THE MEASURED WARNING, restated where a C++ caller can trip over it: cost is
// SUPERLINEAR in the burnup step.  The 3-statepoint grid measured 5,104 outers
// against the 35-statepoint full deck's 4,609 -- MORE work from FEWER
// statepoints -- because the boron secant and the xenon cascade both restart
// from the previous statepoint's solution.  `coarse` (max step 3 GWd/t) is the
// measured optimum.  A caller naming its own grid is not stopped; it is warned
// by the same threshold the tool uses.

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace rasbery::spgrid {

/// tools/make_screening_deck.py COARSE_BURNUPS, cumulative GWd/t past BOC.
inline constexpr double kCoarseBurnups[] = {0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0};
/// tools/make_screening_deck.py THREE_BURNUPS.
inline constexpr double kThreeBurnups[] = {8.0, 16.0};
/// tools/make_screening_deck.py WARN_BURNUP_STEP_GWD.
inline constexpr double kWarnBurnupStepGwd = 4.0;

/// tools/make_screening_deck.py DEPLETION_TIME_KEYS.  A depletion entry states
/// its LENGTH in one of these; the template is the entry minus all of them.
inline constexpr const char* kDepletionTimeKeys[] = {
    "time", "steps", "substeps", "burnup", "bu", "burnup_increment",
    "burnup increment", "until boron ppm", "until_boron_ppm",
};

inline bool isTimeKey(const std::string& key) {
    for (const char* name : kDepletionTimeKeys)
        if (key == name) return true;
    return false;
}

/// True for the two spellings that mean "the deck exactly as it was written".
/// An EMPTY spec is `full`: no request field and `"statepoint_grid":"full"` must
/// be the same run, or the default path would depend on whether a controller
/// bothered to spell the default out.
inline bool isFullGrid(const std::string& spec) {
    return spec.empty() || spec == "full";
}

/// `full` | `coarse` | `three` | an explicit cumulative GWd/t list.
///
/// The list accepts commas or whitespace, the way `--burnups` does, and is
/// required to be STRICTLY INCREASING and positive: a grid is a set of
/// cumulative burnups and a non-increasing one would ask a depletion entry to
/// consume a negative burnup, which Scheduler converts to a negative time.
inline bool parseGrid(const std::string& spec, std::vector<double>& burnups,
                      std::string& error) {
    burnups.clear();
    if (isFullGrid(spec)) return true;
    if (spec == "coarse") {
        burnups.assign(std::begin(kCoarseBurnups), std::end(kCoarseBurnups));
        return true;
    }
    if (spec == "three") {
        burnups.assign(std::begin(kThreeBurnups), std::end(kThreeBurnups));
        return true;
    }
    std::string token;
    auto        flush = [&burnups, &token, &error]() -> bool {
        if (token.empty()) return true;
        try {
            std::size_t  used  = 0;
            const double value = std::stod(token, &used);
            if (used != token.size()) {
                error = "\"statepoint_grid\": \"" + token + "\" is not a number";
                return false;
            }
            burnups.push_back(value);
        } catch (const std::exception&) {
            error = "\"statepoint_grid\": \"" + token + "\" is not a number";
            return false;
        }
        token.clear();
        return true;
    };
    for (const char c : spec) {
        if (c == ',' || c == ' ' || c == '\t') {
            if (!flush()) return false;
            continue;
        }
        token += c;
    }
    if (!flush()) return false;
    if (burnups.empty()) {
        error = "\"statepoint_grid\": \"" + spec +
                "\" named no grid (use full | coarse | three | a cumulative GWd/t list)";
        return false;
    }
    double previous = 0.0;
    for (const double value : burnups) {
        if (!(value > previous)) {
            error = "\"statepoint_grid\": the grid must be strictly increasing and "
                    "positive cumulative GWd/t; \"" +
                    spec + "\" is not";
            return false;
        }
        previous = value;
    }
    return true;
}

/// The largest single burnup step, or 0 for an empty grid.  A caller prints the
/// measured warning above this; nothing refuses on it.
inline double largestStep(const std::vector<double>& burnups) {
    double previous = 0.0;
    double worst    = 0.0;
    for (const double value : burnups) {
        worst    = std::max(worst, value - previous);
        previous = value;
    }
    return worst;
}

namespace detail {

/// make_screening_deck.depletion_template: the MODAL depletion entry, minus its
/// time keys.  Ties break toward the FIRST bucket seen, which is what Python's
/// `max(buckets.values(), key=len)` over an insertion-ordered dict does.
template <class Json>
inline Json depletionTemplate(const std::vector<const Json*>& entries) {
    std::vector<std::string> keys;
    std::vector<Json>        firsts;
    std::vector<std::size_t> counts;
    for (const Json* entry : entries) {
        Json stripped = Json::object();
        for (auto it = entry->begin(); it != entry->end(); ++it)
            if (!isTimeKey(it.key())) stripped[it.key()] = it.value();
        // The bucket key only has to separate DIFFERENT settings consistently,
        // and nlohmann's dump of an ordered_json preserves the deck's own key
        // order -- which is the same for every entry of one deck, because they
        // came out of one file written by one generator.
        const std::string bucket = stripped.dump();
        std::size_t       found  = keys.size();
        for (std::size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == bucket) {
                found = i;
                break;
            }
        if (found == keys.size()) {
            keys.push_back(bucket);
            firsts.push_back(stripped);
            counts.push_back(1);
        } else {
            ++counts[found];
        }
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < counts.size(); ++i)
        if (counts[i] > counts[best]) best = i;
    return firsts.empty() ? Json::object() : firsts[best];
}

/// Python's round(x, 10), which is what the tool writes into the deck.
inline double round10(double value) { return std::round(value * 1.0e10) / 1.0e10; }

} // namespace detail

/// Rewrite *deck*'s `schedule` onto *burnups*.  False with *error* set when the
/// deck has no depletion entry to coarsen -- which is a REFUSAL and not a
/// silent pass-through: a request that asked for a screening lane and got the
/// full deck would be a full-cost case wearing a screening receipt.
template <class Json>
inline bool applyGrid(Json& deck, const std::vector<double>& burnups, std::string& error) {
    if (burnups.empty()) return true;
    if (!deck.contains("schedule") || !deck["schedule"].is_array()) {
        error = "\"statepoint_grid\" was asked for but this deck has no \"schedule\" array";
        return false;
    }
    const Json& schedule = deck["schedule"];

    std::size_t first = schedule.size();
    std::size_t last  = 0;
    bool        any   = false;
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const auto& item = schedule[i];
        if (item.is_object() && item.contains("type") && item["type"].is_string() &&
            item["type"].template get<std::string>() == "depletion") {
            if (!any) first = i;
            last = i;
            any  = true;
        }
    }
    if (!any) {
        error = "\"statepoint_grid\" was asked for but this deck has no depletion entry "
                "-- there is nothing to coarsen, and running the full deck under a "
                "screening declaration would file a full-cost case in the screening lane";
        return false;
    }

    std::vector<const Json*> depletion;
    for (std::size_t i = first; i <= last; ++i) depletion.push_back(&schedule[i]);
    const Json entry_template = detail::depletionTemplate<Json>(depletion);

    Json rebuilt = Json::array();
    for (std::size_t i = 0; i < first; ++i) rebuilt.push_back(schedule[i]);
    double previous = 0.0;
    for (const double cumulative : burnups) {
        Json entry      = entry_template;
        entry["type"]   = "depletion";
        entry["steps"]  = 1;
        entry["burnup"] = detail::round10(cumulative - previous);
        rebuilt.push_back(entry);
        previous = cumulative;
    }
    for (std::size_t i = last + 1; i < schedule.size(); ++i) rebuilt.push_back(schedule[i]);

    deck["schedule"] = rebuilt;
    return true;
}

/// parseGrid + applyGrid, the form IO::ReadInput calls.
template <class Json>
inline bool applyGridSpec(Json& deck, const std::string& spec, std::string& error) {
    std::vector<double> burnups;
    if (!parseGrid(spec, burnups, error)) return false;
    return applyGrid(deck, burnups, error);
}

} // namespace rasbery::spgrid
