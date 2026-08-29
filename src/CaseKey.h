#pragma once

// WP10.1 -- the canonical duplicate key of a case.
//
// WHY THE EVALUATOR HAS TO MAKE IT.  A GA that evaluates 2.56M candidates
// re-proposes the same core many times, and the plan's largest single lever is
// not to solve those again.  But the cache is the GA's and the KEY is not: two
// loading patterns that differ only by a symmetry operation of the core are the
// SAME physics, and whether they are depends on `symmetry.angle` /
// `symmetry.mirror` -- deck facts the controller does not interpret and must
// not start interpreting, because a second interpretation is a second answer.
// GA evaluator plan Sec 5.2 says exactly this: the cache is the GA's, the key
// is the evaluator's.
//
// WHAT THE KEY COVERS.  Everything that can move a published number:
//
//   the loading pattern, after canonicalisation under the core's own symmetry
//   the assembly inventory (`batch`) the pattern indexes into
//   the rest of the deck -- schedule, burnup grid, T/H, rods, tolerances
//   the effective physics fidelity and its policy name
//   every trajectory-affecting environment string, raw and in a fixed order
//   the cross-section library's CONTENT digest, not its path
//   the warm-start provenance, because a warm start can pick a root
//   the code identity, as far as the process can know it (see kCodeShaEnv)
//
// WHAT IT DELIBERATELY DOES NOT COVER.  The result mode (full / pin-off /
// light) and the output paths.  The campaign measured all three output modes to
// one trajectory digest, so a cached scalar answer is valid for a later request
// that wants the same physics written differently.  Anything a cache hit CANNOT
// serve -- a pin map that was never written -- is a miss on the artefact, not
// on the key, and the requester asks for a rerun in the mode it needs.
//
// COLLISIONS.  SHA-256 over a payload that is itself reproducible: the key is
// the digest, and `tools/case_key.py --payload` prints the exact bytes it
// digested, so a cache can store them and byte-compare on a hit rather than
// trusting 256 bits it did not verify.
//
// THE CANONICAL FORM IS A TOKEN STREAM, NOT JSON TEXT.  Two implementations
// have to agree byte for byte -- this one and tools/case_key.py -- and JSON
// text does not survive that: key order, whitespace and above all float
// formatting differ between any two serialisers.  So the deck is walked and
// emitted as tokens with exactly one spelling per value, floats through
// `%.17g`, which C++ and Python both produce identically.
// tools/test_case_key_contract.py holds both sides to one fixture.

#include "RunContract.h"
#include "Sha256.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "CompatFormat.h"

namespace rasbery::casekey {

/// The payload's version.  Bump it when the FIELD SET changes: a key computed
/// under a different field set is a different key and must not silently collide
/// with, or miss against, one computed under this one.
inline constexpr const char* kSchema = "rasbery-case-key/v1";

/// The harness declares the build identity here.  It is not derivable inside
/// the process -- there is no embedded commit -- and a key that pretended
/// otherwise would let two builds share a cache entry.  Unset prints as `~`,
/// and the key then does NOT distinguish two builds, which is a fact the
/// payload states rather than hides.
inline constexpr const char* kCodeShaEnv = "RASBERY_CODE_SHA";

// ---------------------------------------------------------------------------
// Canonical value tokens
// ---------------------------------------------------------------------------

inline void appendString(std::string& out, const std::string& text) {
    out += 's';
    out += std::to_string(text.size());
    out += ':';
    out += text;
}

/// One JSON value, as a token stream with exactly one spelling per value.
/// Templated on the JSON type because the deck is parsed as `ordered_json`
/// (order-preserving) while a map built here is a plain `json`; both must give
/// the same bytes, which sorting the object keys below is what guarantees.
template <class Json>
inline void appendValue(std::string& out, const Json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::null:
            out += '~';
            return;
        case nlohmann::json::value_t::boolean:
            out += value.template get<bool>() ? 'T' : 'F';
            return;
        case nlohmann::json::value_t::number_integer:
            out += 'i';
            out += std::to_string(value.template get<long long>());
            return;
        case nlohmann::json::value_t::number_unsigned:
            out += 'i';
            out += std::to_string(value.template get<unsigned long long>());
            return;
        case nlohmann::json::value_t::number_float:
            out += 'd';
            // %.17g, the one float spelling both languages produce identically.
            out += std::format("{:.17g}", value.template get<double>());
            return;
        case nlohmann::json::value_t::string:
            appendString(out, value.template get<std::string>());
            return;
        case nlohmann::json::value_t::array:
            out += '[';
            for (const auto& item : value) appendValue(out, item);
            out += ']';
            return;
        case nlohmann::json::value_t::object: {
            // Sorted by key BYTES, so an object's spelling cannot depend on the
            // order the deck happened to write it in.
            std::vector<std::string> keys;
            keys.reserve(value.size());
            for (auto it = value.begin(); it != value.end(); ++it) keys.push_back(it.key());
            std::sort(keys.begin(), keys.end());
            out += '{';
            for (const auto& key : keys) {
                appendString(out, key);
                appendValue(out, value.at(key));
            }
            out += '}';
            return;
        }
        default:
            // binary / discarded: a deck cannot contain them, and inventing a
            // spelling for one would be a spelling nothing agrees with.
            out += '?';
            return;
    }
}

template <class Json>
inline std::string canonical(const Json& value) {
    std::string out;
    appendValue(out, value);
    return out;
}

// ---------------------------------------------------------------------------
// Symmetry canonicalisation of the loading pattern
// ---------------------------------------------------------------------------

using CoreMap = std::vector<std::vector<std::string>>;

/// The result of folding a loading pattern onto its symmetry orbit.
struct CoreCanon {
    CoreMap     map;          ///< the lexicographically smallest member of the orbit
    std::string op = "identity"; ///< which operation produced it, for the receipt
};

/// Rows padded to a rectangle with "XX".  A deck may write ragged rows, and
/// Geometry treats a missing column exactly as an absent assembly
/// (Geometry.cpp's core scan skips "XX"), so the padded map is the same core --
/// and a rectangle is what a rotation can be defined on at all.
inline CoreMap rectangular(const CoreMap& core) {
    std::size_t width = 0;
    for (const auto& row : core) width = std::max(width, row.size());
    CoreMap out = core;
    for (auto& row : out) row.resize(width, "XX");
    return out;
}

inline CoreMap transposed(const CoreMap& m) {
    if (m.empty()) return m;
    CoreMap out(m[0].size(), std::vector<std::string>(m.size()));
    for (std::size_t r = 0; r < m.size(); ++r)
        for (std::size_t c = 0; c < m[r].size(); ++c) out[c][r] = m[r][c];
    return out;
}

inline CoreMap flippedRows(const CoreMap& m) {
    CoreMap out = m;
    std::reverse(out.begin(), out.end());
    return out;
}

inline CoreMap flippedCols(const CoreMap& m) {
    CoreMap out = m;
    for (auto& row : out) std::reverse(row.begin(), row.end());
    return out;
}

/// Fold a loading pattern onto the canonical member of its symmetry orbit.
///
/// WHICH OPERATIONS ARE LEGAL, AND WHY.
///
///   symmetry.angle == 90 (a quarter map).  The full core is generated from
///   the quarter by mirroring (`mirror: true`) or by 90-degree rotation
///   (`mirror: false`) about the SAME corner.  Transposing the quarter
///   therefore produces the full core reflected in its diagonal -- an isometry
///   of a square core, which two-group diffusion is invariant under, so keff,
///   the critical boron, Fq and FdH are identical.  The other seven square
///   operations are NOT legal here: they move the fold corner, and a quarter
///   folded about a different corner is a different core.
///
///   symmetry.angle == 360 (a full map).  The whole dihedral group applies --
///   four rotations and four reflections on a square map, and the four that
///   preserve a rectangle otherwise.
///
///   anything else (a half core, an unusual sector).  IDENTITY ONLY, and the
///   `op` field says so.  A fold that has not been argued is a fold that can
///   silently merge two different cores into one cache entry, which is the one
///   failure this must not have.  Adding a sector means adding it here with the
///   isometry that justifies it.
inline CoreCanon canonicalCore(const CoreMap& core, int symang, bool /*mirror*/) {
    const CoreMap rect   = rectangular(core);
    const bool    square = !rect.empty() && rect.size() == rect[0].size();

    std::vector<std::pair<const char*, CoreMap>> orbit;
    orbit.emplace_back("identity", rect);
    if (symang == 90) {
        if (square) orbit.emplace_back("transpose", transposed(rect));
    } else if (symang == 360) {
        const CoreMap rot180 = flippedRows(flippedCols(rect));
        orbit.emplace_back("rot180", rot180);
        orbit.emplace_back("flip_rows", flippedRows(rect));
        orbit.emplace_back("flip_cols", flippedCols(rect));
        if (square) {
            const CoreMap tr = transposed(rect);
            orbit.emplace_back("transpose", tr);
            orbit.emplace_back("rot90", flippedCols(tr));
            orbit.emplace_back("rot270", flippedRows(tr));
            orbit.emplace_back("antitranspose", flippedRows(flippedCols(tr)));
        }
    }

    CoreCanon best;
    std::string best_text;
    for (const auto& [name, map] : orbit) {
        const std::string text = canonical(nlohmann::json(map));
        if (best_text.empty() || text < best_text) {
            best_text = text;
            best.map  = map;
            best.op   = name;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// The two halves of the key
// ---------------------------------------------------------------------------

/// The DECK half: everything the key takes from the input file, with the
/// loading pattern replaced by its canonical member.  Computed where the deck
/// JSON is live (IO::ReadInput) and carried as one digest, so nothing has to
/// keep the parsed deck alive to name the case later.
template <class Json>
inline std::string deckPayload(const Json& config, std::string* core_op = nullptr) {
    const CoreMap core =
        config.contains("core") ? config.at("core").template get<CoreMap>() : CoreMap{};
    int  symang = 0;
    bool mirror = false;
    int  symdiv = 0;
    if (config.contains("geometry") && config.at("geometry").contains("symmetry")) {
        const auto& sym = config.at("geometry").at("symmetry");
        symang = sym.value("angle", 0);
        mirror = sym.value("mirror", false);
        symdiv = sym.value("center assembly divided", false) ? 1 : 0;
    }
    const CoreCanon canon = canonicalCore(core, symang, mirror);
    if (core_op != nullptr) *core_op = canon.op;

    // The deck MINUS the two blocks that are re-added canonically.  Everything
    // else -- schedule, burnup grid, geometry, T/H, rods, tolerances -- goes in
    // verbatim, because all of it can move a published number.
    Json rest = config;
    rest.erase("core");
    rest.erase("batch");

    std::string payload;
    payload += "sym\t";
    payload += std::to_string(symang);
    payload += '\t';
    payload += mirror ? '1' : '0';
    payload += '\t';
    payload += std::to_string(symdiv);
    // core_op is NOT in the payload, and that is the whole point: the operation
    // that carried a pattern onto the canonical member of its orbit is HOW the
    // key was reached, not WHAT it identifies.  Folding it in would give a
    // pattern and its transpose two different keys -- exactly the miss the
    // canonicalisation exists to remove.  It is REPORTED instead, in
    // [RASBERY][CASE], so a reader can still see which member was chosen.
    payload += "\ncore\t";
    payload += canonical(nlohmann::json(canon.map));
    payload += "\nbatch\t";
    payload += config.contains("batch") ? canonical(config.at("batch")) : std::string("~");
    payload += "\nrest\t";
    payload += canonical(rest);
    payload += '\n';
    return payload;
}

/// What the case key needs beyond the deck.  Every field is a string so the
/// payload can never disagree with itself about how a value was spelled.
struct Provenance {
    std::string deck_digest;   ///< sha256 of deckPayload()
    std::string fidelity;      ///< effective PhysicsFidelity, plan Sec 6.2 spelling
    std::string policy;        ///< its campaign shorthand
    std::string xslib_digest;  ///< sha256 of the cross-section library CONTENT
    std::string warm_start;    ///< warm-start provenance token, or empty
    std::vector<std::pair<std::string, std::string>> env; ///< arm knobs, raw, in order
};

inline std::string tokenOrTilde(const std::string& text) {
    return text.empty() ? std::string("~") : text;
}

/// The full payload.  Line-oriented and readable on purpose: when two runs
/// disagree about a key, the answer has to be visible in a diff.
inline std::string payloadOf(const Provenance& p) {
    std::string out = kSchema;
    out += "\ndeck\t";
    out += tokenOrTilde(p.deck_digest);
    out += "\nfidelity\t";
    out += tokenOrTilde(p.fidelity);
    out += "\npolicy\t";
    out += tokenOrTilde(p.policy);
    for (const auto& [name, value] : p.env) {
        out += "\nenv\t";
        out += name;
        out += '\t';
        out += tokenOrTilde(value);
    }
    out += "\nxslib\t";
    out += tokenOrTilde(p.xslib_digest);
    out += "\nwarm_start\t";
    out += tokenOrTilde(p.warm_start);
    out += "\ncode_sha\t";
    const char* code = std::getenv(kCodeShaEnv);
    out += tokenOrTilde(code != nullptr ? std::string(code) : std::string());
    out += '\n';
    return out;
}

inline std::string keyOf(const Provenance& p) { return Sha256::hexOf(payloadOf(p)); }

} // namespace rasbery::casekey
