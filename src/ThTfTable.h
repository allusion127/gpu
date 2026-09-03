#pragma once

// WHICH FUEL-TEMPERATURE TABLE.  The dT(LPD, burnup) grid that GetTfuel
// interpolates, SELECTED rather than assumed.
//
// THE FINDING.  include/Database/tf.csv is a 9-LPD x 10-burnup table of
// dT = Tfuel - Tmod, read by XSSet::LoadTHTables, interpolated by
// XSSet::GetTfuel (src/XSSet.h:516-526) at src/XSSet.cpp:6378-6381, and
// mirrored on the device by thGetTfuel (src/ThKernel.h) and in the
// src/ThReference quotation.  It is MASTER's BUILT-IN WH-TYPE table --
// `isolth = 11` -- and its top-left corner is the %DEF_TFT example printed in
// the MASTER-3.0 manual.  The APR1400 / KNGR decks this campaign runs do NOT
// use it: their `isolth` is 12, the ABB-CE table, whose burnup slope of the
// rise is about five times steeper.  Over one cycle MASTER's rise falls by
// 29.6 % while this table falls by 5.8 %, so KNGR tfavg comes out about
// -14.6 K at BOC and +71 K at EOC -- a drift the boron search silently
// absorbs, and which is where the -15.3 ppm Gate B residual has been sitting.
//
// WHY THE DEFAULT IS STILL THE WH TABLE.  Every published number in this
// campaign was produced with tf.csv (the trajectory digest 1f36e75dc00ed2b4 at
// 4377 outers above all).  Swapping the table silently would move every fuel
// temperature, every cross section and every published number at once, and
// destroy the baseline the swap is supposed to be measured against.  So the
// table becomes a RESOLVED IDENTITY with a source, the default source is
// `legacy`, and moving it is an explicit act: a deck key plus
// RASBERY_TH_TF_TABLE=deck, or a name, or a path.
//
// NOTHING HERE GUESSES A TABLE.  `ce` names include/Database/tf_ce.csv, which
// DOES NOT EXIST in the tree: the ABB-CE grid has to be REGRESSED from MASTER
// per-node edits (tools/fit_tf_table.py), not invented from a slope ratio.
// Asking for it before it exists is REFUSED, loudly.  A guessed fuel
// temperature is a wrong cross section that no receipt would flag.
//
// LEGACY IS `wh`, DELIBERATELY.  tf.csv IS the WH table, so `legacy` and an
// explicit `wh` resolve to the SAME identity and therefore the SAME case key.
// Only the SOURCE differs, and the source is reported, never folded -- the
// same rule th_fuel_rods follows (src/ThFuelRods.h).

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace rasbery::th {

/// The override.  `legacy` (or unset) keeps the shipped WH table, `deck` takes
/// whatever the deck declared (and REFUSES when it declared nothing), `wh` and
/// `ce` name the two built-in tables, and anything else is taken as a path --
/// with or without the `file:` prefix the deck key also accepts.
inline constexpr const char* kTfTableEnv = "RASBERY_TH_TF_TABLE";

/// The two built-in file names, relative to include/Database.
inline constexpr const char* kTfTableWhFile = "tf.csv";
inline constexpr const char* kTfTableCeFile = "tf_ce.csv";

/// What the deck asked for.  `name` is "" when the deck said nothing.
struct TfTableSpec {
    std::string         name; ///< "" | "wh" | "ce" | "file" | "inline"
    std::string         path; ///< for name == "file"
    std::vector<double> lpd;  ///< inline: the LPD axis    [W/cm], ascending
    std::vector<double> bu;   ///< inline: the burnup axis [GWd/t], ascending
    std::vector<double> dt;   ///< inline: dT row-major over bu, nbu * nlpd

    [[nodiscard]] bool declared() const { return !name.empty(); }
};

/// The resolved REQUEST -- what to load, and who asked.  No file has been
/// touched yet; loadTfTable() does that, once, and adds the digest.
struct TfChoice {
    std::string         name   = "wh";     ///< "wh" | "ce" | "file" | "inline"
    std::string         source = "legacy"; ///< "legacy" | "deck" | "env"
    std::string         path;              ///< for name == "file"
    std::vector<double> lpd, bu, dt;       ///< for name == "inline"
};

/// The loaded table plus its identity.  PLAIN VECTORS, deliberately: this
/// header is included by Geometry.h and must not drag milk.h behind it.
struct TfTableData {
    std::string         name;    ///< "wh" | "ce" | "file" | "inline"
    std::string         source;  ///< "legacy" | "deck" | "env" -- REPORTED, not keyed
    std::string         path;    ///< the file actually read, or "" for inline
    std::string         sha256;  ///< of the file BYTES, or of the inline canonical form
    std::vector<double> lpd, bu; ///< the axes as loaded
    std::vector<double> dt;      ///< row-major over bu, nbu * nlpd
    int                 nlpd = 0;
    int                 nbu  = 0;

    /// What the case key folds: the table's identity, blind to who asked for it.
    [[nodiscard]] std::string identity() const { return name + ":" + sha256; }
};

inline std::string tfTrimmed(const char* raw) {
    std::string text = raw != nullptr ? std::string(raw) : std::string();
    std::size_t b    = 0;
    while (b < text.size() && std::isspace(static_cast<unsigned char>(text[b]))) ++b;
    std::size_t e = text.size();
    while (e > b && std::isspace(static_cast<unsigned char>(text[e - 1]))) --e;
    return text.substr(b, e - b);
}

/// Parse one spec STRING -- the deck key's value or the environment's -- into a
/// choice with no source yet.  Throws on a spelling that names nothing.
inline TfChoice tfChoiceOfString(const std::string& raw, const char* whence) {
    const std::string want = tfTrimmed(raw.c_str());
    TfChoice          c;
    if (want == "wh" || want == "legacy") {
        c.name = "wh";
        return c;
    }
    if (want == "ce") {
        c.name = "ce";
        return c;
    }
    const std::string path = want.rfind("file:", 0) == 0 ? want.substr(5) : want;
    if (path.empty())
        throw std::runtime_error(std::string(whence) +
                                 " names no fuel-temperature table; use \"wh\", \"ce\", "
                                 "\"file:<path>\" or an inline table.");
    c.name = "file";
    c.path = path;
    return c;
}

/// THE ONE RESOLUTION.  `deck` is what IO parsed; the environment wins.
inline TfChoice resolveTfTable(const TfTableSpec& deck) {
    const std::string want = tfTrimmed(std::getenv(kTfTableEnv));

    if (want.empty() || want == "legacy") {
        TfChoice c;
        c.name   = "wh"; // tf.csv IS the WH table -- same identity, same key
        c.source = "legacy";
        return c;
    }

    if (want == "deck") {
        if (!deck.declared())
            throw std::runtime_error(
                "RASBERY_TH_TF_TABLE=deck, but the deck declares no fuel-temperature "
                "table: add a \"tf table\" key (\"wh\" | \"ce\" | \"file:<path>\" | an "
                "inline {lpd, bu, dt} table) under geometry.dimensions or \"default "
                "parameters\".  Nothing here infers one -- the shipped table is "
                "MASTER's WH (isolth 11) and a KNGR deck is CE (isolth 12); guessing "
                "between them is a wrong fuel temperature no receipt would flag.");
        TfChoice c;
        c.name   = deck.name;
        c.path   = deck.path;
        c.lpd    = deck.lpd;
        c.bu     = deck.bu;
        c.dt     = deck.dt;
        c.source = "deck";
        return c;
    }

    TfChoice c = tfChoiceOfString(want, kTfTableEnv);
    c.source   = "env";
    return c;
}

/// include/Database, as the build points at it.  ONE spelling for every caller,
/// so the loader and the case key can never read different trees.
[[nodiscard]] std::string tfDatabaseDir();

/// Load (or return the process-cached) table for `choice`, print the
/// [RASBERY][TH][TFTABLE] receipt once per distinct identity, and refuse loudly
/// when the named table does not exist.  IDEMPOTENT: every caller -- XSSet, the
/// case key -- gets the same object, so the identity is resolved exactly once.
[[nodiscard]] const TfTableData& loadTfTable(const TfChoice& choice);

/// The inline table's canonical serialisation -- what its sha256 digests.
/// tools/case_key.py mirrors this byte for byte.
[[nodiscard]] std::string tfInlinePayload(const std::vector<double>& lpd,
                                          const std::vector<double>& bu,
                                          const std::vector<double>& dt);

} // namespace rasbery::th
