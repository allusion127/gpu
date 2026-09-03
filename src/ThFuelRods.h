#pragma once

// The fuel-temperature linear power density's DIVISOR: how many fuel rods a
// node carries.
//
// THE DEFECT.  XSSet::SolveTH computes the linear power density that indexes
// the Tfuel table as
//
//     lpd = 1000 * P_node / (RODS * hz[k])
//
// and RODS was the literal `62.0`, hard-coded in three places (the host body in
// XSSet.cpp, the ported body in ThKernel.h and the verbatim quotation in
// ThReference.cpp) and inherited unexamined from the CPU baseline (e76d40d).
// It is not 62 for either deck this campaign runs:
//
//     APR1400  16x16, 236 fuel rods / assembly, ndivxy=2 -> 236/4 = 59
//     i-SMR    17x17, 260 fuel rods / assembly, ndivxy=2 -> 260/4 = 65
//                     (MASTER's depf card comments it "npin, nfrod (289-29)")
//
// and the error is a straight scale on the fuel temperature RISE: the measured
// RASBERY/MASTER rise ratio on i-SMR is 1.047-1.062, which is 65/62 = 1.048 to
// within the noise of the comparison, and costs about +9 K on tfavg (~ -25 pcm).
//
// WHY THE DEFAULT IS STILL 62.  Every published number in this campaign -- the
// 238 trajectory digest 1f36e75dc00ed2b4 at 4377 outers above all -- was
// produced with the literal, and a fix that silently moved every number would
// destroy the B0 baseline it is supposed to be measured against.  So the divisor
// becomes a resolved VALUE with a source, the default source is `legacy_62`,
// and moving it is an explicit act: a deck key plus RASBERY_TH_FUEL_RODS=deck,
// or a number.  The A/B that pays for the change is then one run against one
// run, and the receipt says which arm each was.
//
// NOTHING HERE GUESSES.  `npins^2 - guide tubes` is not derivable from anything
// the deck states (the guide-tube count is a lattice fact that lives in the
// assembly description, not in `geometry.dimensions`), so a request for the
// deck value that the deck does not carry is REFUSED, loudly, rather than
// approximated.  An approximated divisor is a wrong fuel temperature that no
// receipt would flag.

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rasbery::th {

/// THE ONLY 62.0 IN THE TREE.  tools/test_th_fuel_rods_contract.py holds it
/// there: every body reads the resolved field, and the literal survives only
/// here, as the name of the baseline it is.
inline constexpr double kLegacyFuelRodsPerNode = 62.0;

/// The override.  `legacy` (or unset) keeps kLegacyFuelRodsPerNode, `deck`
/// takes the deck's declared count, and a bare number is the rods-per-NODE
/// value itself (an escape hatch for a bisect, not a deck format).
inline constexpr const char* kFuelRodsEnv = "RASBERY_TH_FUEL_RODS";

/// What the divisor resolved to, and where it came from.  The SOURCE is
/// reported and never folded into the case key: a deck that declares 62 and a
/// legacy default are the same arithmetic and must key alike.
struct FuelRods {
    double      value  = kLegacyFuelRodsPerNode;
    std::string source = "legacy_62"; ///< "legacy_62" | "deck" | "env"
};

inline std::string trimmed(const char* raw) {
    std::string text = raw != nullptr ? std::string(raw) : std::string();
    std::size_t b    = 0;
    while (b < text.size() && std::isspace(static_cast<unsigned char>(text[b]))) ++b;
    std::size_t e = text.size();
    while (e > b && std::isspace(static_cast<unsigned char>(text[e - 1]))) --e;
    return text.substr(b, e - b);
}

/// @param deck_rods_per_node  the deck's rods-per-node (nfrod / ndivxy^2), or
///                            <= 0 when the deck declares nothing.
inline FuelRods resolveFuelRodsPerNode(double deck_rods_per_node) {
    const std::string want = trimmed(std::getenv(kFuelRodsEnv));

    if (want.empty() || want == "legacy")
        return FuelRods{kLegacyFuelRodsPerNode, "legacy_62"};

    if (want == "deck") {
        if (!(deck_rods_per_node > 0.0))
            throw std::runtime_error(
                "RASBERY_TH_FUEL_RODS=deck, but the deck declares no fuel-rod count: add "
                "\"nfrod\" (fuel rods per assembly) under geometry.dimensions.  Nothing "
                "here infers it from npins -- the guide-tube count is not in the deck, and "
                "a guessed divisor is a wrong fuel temperature no receipt would flag.");
        return FuelRods{deck_rods_per_node, "deck"};
    }

    char*        end   = nullptr;
    const double value = std::strtod(want.c_str(), &end);
    if (end == want.c_str() || end == nullptr || *end != '\0' || !(value > 0.0))
        throw std::runtime_error("RASBERY_TH_FUEL_RODS must be \"legacy\", \"deck\" or a "
                                 "positive number of fuel rods per node; got \"" +
                                 want + "\".");
    return FuelRods{value, "env"};
}

} // namespace rasbery::th
