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
//   the EXECUTION MODE (single / batch), which is an argv fact no env line
//     carries and which selects the Xe Anderson default
//   the EFFECTIVE Xe Anderson and Xe transaction states, which are resolved
//     from the environment AND the execution mode, so the raw env strings
//     under-describe them
//   the RESOLVED CONTRACTION MASKS (`forms`), because the mask a host MINES
//     selects the rounding of production arithmetic and no env string says so
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
/// v2 (2026-09-04) ADDED FOUR LINES -- `exec_mode`, `xe_anderson`, `xe_txn` and
/// `forms`.  EVERY KEY IN THE CAMPAIGN MOVES ONCE, including a single run with
/// the AA default and no forms channel resolved, because the payload gained
/// lines rather than values.  That is the point: the v1 field set could not tell
/// a `--batch-mode 1` run from a single run of the same deck and env, and on 238
/// those two converged on different trajectories (1f36e75dc00ed2b4 / 4377 outers
/// against 4c663ff538b28f82 / 7087) under ONE key -- the batch execution mode
/// turns the Xe Anderson default off (Driver.h xeAndersonGate) and nothing in
/// the key said so.  A cache built under v1 must be re-keyed, not merged.
/// v3 (2026-09-04) ADDED ONE LINE -- `th_fuel_rods`, the EFFECTIVE fuel-rod
/// count per node that divides SolveTH's linear power density
/// (src/ThFuelRods.h).  It was the literal 62.0 in three bodies, it is wrong for
/// both campaign decks (i-SMR 65, APR1400 59), and it is now a resolved value a
/// deck key or RASBERY_TH_FUEL_RODS can move.  Moving it moves every fuel
/// temperature and therefore every cross section, so two runs that differ only
/// in it are two physics and must not share a cache entry.  EVERY KEY MOVES
/// ONCE AGAIN: the payload gained a LINE, and the legacy default is now spelled
/// out rather than assumed.  The VALUE is folded and the SOURCE is not -- a deck
/// that declares 62 and a legacy default are one arithmetic and key alike.
/// v4 (2026-09-04) ADDED ONE MORE LINE -- `th_tf_table`, the IDENTITY of the
/// fuel-temperature dT(LPD, burnup) grid GetTfuel interpolates, spelled
/// `<name>:<sha256>` (src/ThTfTable.h).  The shipped include/Database/tf.csv is
/// MASTER's WH table (isolth 11); the APR1400/KNGR decks are isolth 12 (ABB-CE),
/// whose burnup slope is ~5x steeper -- about -14.6 K on tfavg at BOC and +71 K
/// at EOC.  A deck key or RASBERY_TH_TF_TABLE can now move it, and moving it
/// moves every fuel temperature and therefore every cross section, so two runs
/// that differ only in it are two physics and must not share a cache entry.
/// EVERY KEY MOVES ONCE AGAIN: the payload gained a LINE, and the shipped table
/// is now named and digested rather than assumed.  The IDENTITY is folded and
/// the SOURCE is not -- a deck that names `wh` and the legacy default are one
/// table and key alike.
inline constexpr const char* kSchema = "rasbery-case-key/v4";

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

    // ---- v2 -------------------------------------------------------------
    //
    // WHY THESE ARE NOT COVERED BY `env`.  The env half digests the RAW STRINGS
    // a launcher exported, which is the right thing for a knob whose value IS
    // its meaning.  These three are not that: their effective state is resolved
    // from the environment AND from facts the environment does not carry.
    //
    //   exec_mode   `--batch-mode` is an ARGV flag, not a variable, and it
    //               selects the Xe Anderson default (Driver.h xeAndersonGate).
    //               No env line can see it.
    //   xe_anderson the EFFECTIVE state.  Unset + single = ON, unset + batch =
    //               OFF, and both spell `~` in the env half.  This is the field
    //               that closes the 238 hole.
    //   xe_txn      the EFFECTIVE state.  Default ON, `=0` off, and a stub
    //               (CUDA=OFF) build resolves it OFF whatever the variable says.
    //
    // THE SOURCE NAMES ARE CARRIED BUT NOT FOLDED.  `RASBERY_XE_ANDERSON` is
    // already on trajectory::kArmEnv, so "somebody asked for it" is already in
    // the key through its raw value; folding the source again would give the
    // batch harness (which exports RASBERY_XE_ANDERSON=1 --
    // tools/run_single_gpu_batch.py DEFAULT_ENV) and a single run that takes
    // the same state by default two keys for one arithmetic, which is a cache
    // miss bought for nothing.  They are printed in [RASBERY][CASE] instead,
    // because "AA was on" and "AA was on because somebody asked" are different
    // facts about a measurement.
    std::string exec_mode;          ///< "single" | "batch"
    std::string xe_anderson;        ///< effective: "on" | "off"
    std::string xe_anderson_source; ///< "default" | "env" -- REPORTED, not folded
    std::string xe_txn;             ///< effective: "on" | "off"
    std::string xe_txn_source;      ///< "default" | "env" -- REPORTED, not folded
    /// sha256 of gpu::formsPayloadFrozen(), or empty when no contraction
    /// channel had resolved by the time the key was computed.  The mask a host
    /// MINES selects the rounding of production arithmetic and is not derivable
    /// from any environment string, so it has to be measured and folded.
    std::string forms_digest;

    // ---- v3 -------------------------------------------------------------
    //
    // THE FUEL-TEMPERATURE DIVISOR.  `%.17g` of Geometry::fuel_rods_per_node(),
    // the rods-per-node that SolveTH divides the linear power density by.  It
    // is not an env line because it can come from the DECK, and it is not a
    // deck line either because RASBERY_TH_FUEL_RODS can override the deck --
    // the EFFECTIVE value is the only thing that describes the arithmetic.
    // The source is carried for the receipt and deliberately NOT folded, for
    // the same reason `xe_anderson_source` is not.
    std::string th_fuel_rods;        ///< effective rods per node, "%.17g"
    std::string th_fuel_rods_source; ///< "legacy_62" | "deck" | "env" -- REPORTED

    // ---- v4 -------------------------------------------------------------
    //
    // THE FUEL-TEMPERATURE TABLE.  `<name>:<sha256>` of the grid actually
    // interpolated -- the CONTENT, so a re-fitted tf_ce.csv is a different case
    // even under the same name, and the same table read from another path is
    // still one case.  The source is carried for the receipt and deliberately
    // NOT folded, for the same reason `th_fuel_rods_source` is not.
    std::string th_tf_table;        ///< "<name>:<sha256>"
    std::string th_tf_table_source; ///< "legacy" | "deck" | "env" -- REPORTED
};

inline std::string tokenOrTilde(const std::string& text) {
    return text.empty() ? std::string("~") : text;
}

/// The build identity token, spelled ONCE.  `payloadOf` digests it and the
/// [RASBERY][CASE] receipt prints it, and those two reading the environment
/// through two expressions is how a receipt starts describing a key it did not
/// name.
inline std::string codeShaToken() {
    const char* code = std::getenv(kCodeShaEnv);
    return tokenOrTilde(code != nullptr ? std::string(code) : std::string());
}

/// The ENV HALF of the payload as its own bytes: one `env\tNAME\tVALUE` line
/// per knob, in kArmEnv order, newline-terminated.
///
/// WHY THIS EXISTS SEPARATELY FROM payloadOf.  WP10.1's first live failure
/// (host 181, kngr_238.json) was a key mismatch with no way to say WHICH of the
/// six components moved -- the solver published one 64-hex digest and the tool
/// published another, and the whole diagnosis budget went into guessing.  A
/// digest per component is what turns that into one line of output: the deck
/// half already had one (`deck_digest`), and this gives the half that is read
/// from the environment the same treatment.  Thirty raw values in a receipt
/// would be unreadable and would leak whatever a launcher exported; one digest
/// answers "is the env part the same" and nothing else.
///
/// THE LINE SPELLING IS payloadOf's, DELIBERATELY, so the component digest is a
/// digest of the actual payload bytes and not of a paraphrase of them.  The two
/// are held together by tools/test_case_key_contract.py, which compiles this
/// header and compares both against tools/case_key.py byte for byte.
inline std::string envPayload(const Provenance& p) {
    std::string out;
    for (const auto& [name, value] : p.env) {
        out += "env\t";
        out += name;
        out += '\t';
        out += tokenOrTilde(value);
        out += '\n';
    }
    return out;
}

inline std::string envDigest(const Provenance& p) { return Sha256::hexOf(envPayload(p)); }

/// WHICH knobs were set, by NAME, comma-joined in kArmEnv order.
///
/// WHY A DIGEST WAS NOT ENOUGH.  `env_digest` answers "is the env half the
/// same" and, when the answer is no, nothing else -- which on host 181
/// (kngr_238.json, 2026-08-30) left the reading of the failure open between two
/// very different stories: the solver folds a value the tool does not model
/// (a code defect), or the two simply ran under different environments (a
/// harness fact, and the far likelier one, since the tool reads the env of
/// whatever shell invoked it while the run under comparison carried the
/// production arm).  Those need opposite responses and the receipt could not
/// tell them apart.
///
/// This separates them in one line.  Two runs whose `env_set` DIFFERS were
/// configured differently -- an arm on in one and off in the other -- and the
/// key is doing its job.  Two runs whose `env_set` MATCHES while `env_digest`
/// does not have the same knobs set to different VALUES, which is the case
/// worth reading code over.
///
/// NAMES ONLY, NEVER VALUES: the values are whatever a launcher exported, they
/// are what `env_digest` is for, and a receipt is not the place to publish
/// them.  `~` (unset or empty) is the state that is left out.
inline std::string envSetToken(const Provenance& p) {
    std::string out;
    for (const auto& [name, value] : p.env) {
        if (value.empty()) continue;
        if (!out.empty()) out += ',';
        out += name;
    }
    return out;
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
    out += codeShaToken();
    // ---- v2, appended AFTER code_sha so a v1 payload is a prefix of a v2 one
    // and a diff of the two reads as four added lines rather than a reshuffle.
    out += "\nexec_mode\t";
    out += tokenOrTilde(p.exec_mode);
    out += "\nxe_anderson\t";
    out += tokenOrTilde(p.xe_anderson);
    out += "\nxe_txn\t";
    out += tokenOrTilde(p.xe_txn);
    out += "\nforms\t";
    out += tokenOrTilde(p.forms_digest);
    // ---- v3, appended after forms for the same reason ------------------
    out += "\nth_fuel_rods\t";
    out += tokenOrTilde(p.th_fuel_rods);
    // ---- v4, appended after th_fuel_rods for the same reason -----------
    out += "\nth_tf_table\t";
    out += tokenOrTilde(p.th_tf_table);
    out += '\n';
    return out;
}

inline std::string keyOf(const Provenance& p) { return Sha256::hexOf(payloadOf(p)); }

} // namespace rasbery::casekey
