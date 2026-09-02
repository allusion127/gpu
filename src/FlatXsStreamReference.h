#pragma once

// A VERBATIM QUOTATION of the three burnup-lerp sites in
// XSSet::ResolveSpectralHistoryDeltas, and a fixture to drive it -- WP23.1.
//
// DELIBERATELY IN ITS OWN TRANSLATION UNIT, and this header must never be
// included by one that also includes FlatXsStreamKernel.h.  With both in one TU
// gcc common-subexpressions across them and changes the QUOTATION's
// contraction, so the mining would score a reference that is no longer the
// production one.  ThReference.h and CmfdOuterReference.h say the same thing
// and WP22 paid for the lesson twice.
//
// ---------------------------------------------------------------------------
// WHAT IS QUOTED, AND IN WHAT SHAPE
// ---------------------------------------------------------------------------
//
// Three lambdas of XSSet::ResolveSpectralHistoryDeltas -- `referenceDensity`,
// `referenceDensity0` and `referenceCondition` -- written exactly as XSSet.cpp
// writes them: plain `+`, `*` and `/`, no barriers, no std::fma, nothing that
// tells the compiler what to do.  Whatever gcc contracts HERE is what the
// production host loop contracts, because it is the same text under the same
// flags.
//
// THE SHAPE IS PART OF THE QUOTATION, and that is WP22's expensive lesson:
// which multiply gcc folds into an add is a decision it re-makes per inlining
// context, so a fixture that calls a quoted expression from a different call
// graph pins a different mask.  The production context is
//
//     BuildFlatXsStream                  #pragma omp parallel for, over nodes
//       -> ResolveNodeApplications       one node
//         -> ResolveSpectralHistoryDeltas   holds all three lambdas
//           -> the lambdas                  called from the coordinate chain
//
// and refBuildStream / refResolveNode / refResolveSpectralHistoryDeltas below
// are that graph, with the same OpenMP pragma and the same schedule clause.
// The three lambdas live in ONE function here because they live in one function
// there; splitting them into three would be a different inlining problem for
// gcc to solve.
//
// WHAT IS NOT QUOTED.  Anything that is not the arithmetic of those three
// sites: Chiffon's term structs, the twenty coordinate forms, the delta
// bookkeeping.  Those carry no contraction site (they are libm calls, integer
// work and single multiplies with no add to fuse into), so quoting them would
// be a second implementation of them, which is the thing this file exists to
// avoid being.
//
// NOTHING IN THIS FILE MAY BE "TIDIED".

#include <cstddef>
#include <vector>

namespace fsref {

/// A deck-shaped operand set for the three sites: small enough to mine in
/// milliseconds, wide enough that every branch of the quoted lambdas is
/// reached -- the empty key list, the single-key list (loIndex == hiIndex, so
/// the lerp is skipped), the equal-burnup guard, the out-of-range isotope, the
/// below-first and above-last burnup clamps, and the ordinary interior
/// bracket.  A mask mined on operands that never reach a branch would be mined
/// on a fraction of the function.
struct Fixture {
    int       nmodel = 0;
    int       niso   = 0; ///< Isotope::niso, the REGISTRY size
    int       nnode  = 0;
    int       nprobe = 0; ///< isotopes probed per node
    long long n_rows = 0;

    // The library tables, in exactly the flat layout StreamLibView points at,
    // so the shipped device body and the quotation below read THE SAME BYTES
    // and can differ only in how they round.
    std::vector<int>       refr0_key_off, refr0_key_cnt, refr0_present, refr0_keys;
    std::vector<long long> refr0_base, refr_burn_stride;
    std::vector<double>    lib_iden;         ///< [n_rows * niso]
    std::vector<double>    lib_burn;         ///< [n_rows], GWd/THM
    std::vector<double>    lib_ref_branch_x; ///< [n_rows * 3]

    // Per-node operands.
    std::vector<int> node_model; ///< [nnode]
    std::vector<int> node_burn;  ///< [nnode], MWd/tHM
    std::vector<int> probe_iso;  ///< [nnode * nprobe]

    /// Words the driver writes, so the miner can size its buffers without
    /// re-deriving the loop bounds.  TWO per probe for the two density sites:
    /// each is evaluated at `burn` and at the bracket key, which is the
    /// production call pattern and also the only way a difference in one call
    /// cannot be cancelled by the next.
    std::size_t densWords() const {
        return static_cast<std::size_t>(nnode) * static_cast<std::size_t>(nprobe) * 2u;
    }
    std::size_t condWords() const { return static_cast<std::size_t>(nnode) * 3u; }
};

Fixture buildFixture(int nmodel, int nnode, int nprobe);

/// THE SHAPE THE MINING RUNS AT, defined ONCE.  Both the production miner
/// (src/FlatXsStreamFormMiner.cpp) and the gate (test/flatxs_stream_form_probe
/// .cpp) call this rather than each writing its own three numbers: a gate that
/// mined a different fixture from the binary would pass while asserting nothing
/// about what the binary does.
Fixture buildProductionFixture();

/// Drive the quotation over every node, in the production call graph.
///
/// `dens_out` and `dens0_out` are [densWords()]; `cond_out` is [condWords()].
/// Caller-owned, so scoring never writes the fixture.
void refBuildStream(const Fixture& f, double* dens_out, double* dens0_out,
                    double* cond_out);

} // namespace fsref
