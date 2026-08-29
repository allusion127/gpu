#pragma once

// WP8 stage 2 -- the cohort key: which cases may share built state.
//
// WHY A SECOND KEY, WHEN CaseKey.h ALREADY EXISTS.  They answer different
// questions, and a cache that used one for the other would be wrong in both
// directions.
//
//   casekey  "are these the same CASE?"  -- may I return a stored ANSWER?
//   cohort   "are these the same CORE?"  -- may I share built STATE?
//
// The case key is deliberately total: it folds the loading pattern onto its
// symmetry orbit and covers the schedule, the tolerances, the fidelity, the
// environment and the code identity, because every one of those can move a
// published number.  The cohort key is narrower in content and longer in
// lifetime.  Two candidates of one GA generation differ in exactly the thing
// the case key exists to distinguish -- which assembly sits where -- and agree
// on exactly the thing this key exists to identify: the mesh, the neighbour
// maps, the surface tables, the quadrature.  A generation is 64 cases and ONE
// cohort.  That is the lever.
//
// IT IS KEYED ON `GeometryInput`, NOT ON THE DECK JSON.  Deliberately, and it
// matters: `GeometryInput` is what `Geometry::Initialize` is actually called
// with, AFTER the shuffle resolver has rewritten `core` in place (IO.cpp) and
// after a missing geometry block has been recovered from a restart file
// (IO::LoadGeometryFromRestart).  A key read off `config["geometry"]` would
// miss both -- it would give one cohort to two cases whose topology differs
// because a shuffle moved an assembly, and it would have no value at all for a
// restart-driven case, whose deck has no geometry block to read.  The argument
// of the build is the honest key for the build.
//
// WHAT IS IN IT.
//
//   ng, nz, ndivxy, npins       the counts every map is sized by
//   hx, hy, hz[]                the mesh
//   symang, symopt, symdiv      the symmetry fold
//   albedo[6]                   the boundary condition
//   the core OCCUPANCY MASK     which lattice positions hold an assembly at
//                               all -- NOT which assembly.  See below.
//   the library's CONTENT digest and ng (in Provenance)
//
// WHY THE MASK AND NOT THE MAP.  Geometry's index and neighbour maps are a
// function of which lattice positions are OCCUPIED -- the core scan skips
// "XX" -- and not of what occupies them.  A GA that permutes assembly types
// across a fixed footprint is therefore ONE cohort, which is the case worth
// optimising.  A candidate that EMPTIES a position changes the footprint and
// gets its own cohort.  Keying on the whole map would make every candidate its
// own cohort and the lever would be exactly zero; keying on the dimensions
// alone would let a core with a hole share the maps of a core without one,
// which is a wrong answer rather than a slow one.
//
// WHY `batch` IS NOT IN IT.  `batch` is the assembly inventory the pattern
// indexes into, and what Geometry derives from it -- `_is_fuel`, `_kbc`,
// `_kec`, `_hzcore` -- is candidate state that a case refills, not topology a
// cohort shares (src/EvaluatorContext.h names the same six fields).  Putting it
// in would split a cohort for a reason nothing shared depends on.
//
// WHY THE LIBRARY IS IN IT.  Anything sized by `ng` or built from the library's
// group structure is not a function of geometry alone.  Two decks with
// identical geometry against two different libraries are two cohorts, and the
// CONTENT digest is what says so -- a path is a name, and WP8 stage 2 hardened
// the XSLIB key (XsLibrary.h) for precisely the reason that a name is not an
// identity in a process that outlives a library rebuild.
//
// THE SYMMETRY FIELDS ARE IN, AND ARE NOT CANONICALISED.  casekey folds a
// pattern onto its orbit because two patterns related by an isometry are the
// same physics.  A cohort must NOT do that: the state it shares is INDEXED BY
// LATTICE POSITION, and a core and its transpose are the same physics through
// different index maps.  Sharing a neighbour map across a transpose would be a
// silent aliasing bug, so the mask goes in exactly as it was built.

#include "CaseKey.h"
#include "Geometry.h"
#include "Sha256.h"

#include "CompatFormat.h"

#include <string>

namespace rasbery::cohort {

/// The payload's version.  Bump it when the FIELD SET changes: state shared
/// under a different field set is state shared on a different argument.
inline constexpr const char* kSchema = "rasbery-cohort-key/v1";

/// The token a lattice position carries when no assembly sits there.
/// Geometry's core scan skips it, which is what makes the mask -- and not the
/// map -- the thing the topology depends on.
inline constexpr const char* kEmptyPosition = "XX";

/// One character per lattice position, rows separated by `/`.
///
/// Rectangular first, for the reason casekey::rectangular exists: a deck may
/// write ragged rows and Geometry treats a missing column exactly as an absent
/// assembly, so the padded map IS the same core -- and a mask that depended on
/// how many trailing "XX" the author typed would split one cohort in two for a
/// whitespace reason.
inline std::string occupancyMask(const casekey::CoreMap& core) {
    const casekey::CoreMap rect = casekey::rectangular(core);
    std::string            mask;
    for (std::size_t row = 0; row < rect.size(); ++row) {
        if (row > 0) mask += '/';
        for (const auto& cell : rect[row]) mask += (cell == kEmptyPosition) ? '.' : '#';
    }
    return mask;
}

/// The GEOMETRY half of the cohort key, as a line-oriented token stream.
///
/// Line-oriented and readable on purpose, like casekey::payloadOf: when two
/// cases that should share a cohort do not, the answer has to be visible in a
/// diff of two payloads and not merely inferable from two digests that differ.
/// Floats go through `%.17g`, the one spelling every reader produces
/// identically (CaseKey.h states the same rule and for the same reason).
inline std::string geometryPayload(const GeometryInput& gin) {
    std::string out = kSchema;
    out += "\ncounts\t";
    out += std::to_string(gin.ng);
    out += '\t';
    out += std::to_string(gin.nz);
    out += '\t';
    out += std::to_string(gin.ndivxy);
    out += '\t';
    out += std::to_string(gin.npins);
    out += "\npitch\t";
    out += std::format("{:.17g}", gin.hx);
    out += '\t';
    out += std::format("{:.17g}", gin.hy);
    out += "\nhz";
    for (const double h : gin.hz) {
        out += '\t';
        out += std::format("{:.17g}", h);
    }
    out += "\nsym\t";
    out += std::to_string(gin.symang);
    out += '\t';
    out += gin.symopt ? '1' : '0';
    out += '\t';
    out += gin.symdiv ? '1' : '0';
    out += "\nalbedo";
    for (const double a : gin.albedo) {
        out += '\t';
        out += std::format("{:.17g}", a);
    }
    out += "\nmask\t";
    out += occupancyMask(gin.core);
    out += '\n';
    return out;
}

/// What the cohort key needs beyond the geometry.
struct Provenance {
    std::string geometry_digest; ///< sha256 of geometryPayload()
    std::string xslib_digest;    ///< sha256 of the library CONTENT (XsLibrary.h)
    int         ng = 0;          ///< energy groups the flatten was laid out for
};

inline std::string payloadOf(const Provenance& p) {
    std::string out = kSchema;
    out += "\ngeometry\t";
    out += casekey::tokenOrTilde(p.geometry_digest);
    out += "\nxslib\t";
    out += casekey::tokenOrTilde(p.xslib_digest);
    out += "\nng\t";
    out += std::to_string(p.ng);
    out += '\n';
    return out;
}

inline std::string keyOf(const Provenance& p) { return Sha256::hexOf(payloadOf(p)); }

/// One step, for callers that hold the build's own argument.
inline std::string keyOf(const GeometryInput& gin, const std::string& xslib_digest) {
    return keyOf(Provenance{Sha256::hexOf(geometryPayload(gin)), xslib_digest, gin.ng});
}

} // namespace rasbery::cohort
