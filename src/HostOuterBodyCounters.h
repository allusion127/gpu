#pragma once

// How many times the HOST ran a step of the CMFD outer -- Rev.7.1 Task 9.
//
// ---------------------------------------------------------------------------
// WHAT THIS IS FOR, AND WHY A COUNTER RATHER THAN A REVIEW
// ---------------------------------------------------------------------------
//
// Tasks 4, 5 and 7 built device bodies for updpsi, upddtil, updjnet, upddhat
// and the nodal constants, and each one is replay-verified bit-identical to the
// host loop it replaces.  None of that says the PRODUCTION solve ever calls
// them.  An audit of 8be6bee found exactly that: every one of those kernels had
// zero production callers, so a run with the device features on still executed
// BICGCMFD::updpsi/updjnet/upddhat and Nodal::updateConstant on the CPU, and no
// receipt in the tree could have told anyone.
//
// This is the counter that can.  It is read at the end of a run and printed in
// the [RASBERY][OUTER_GPU] receipt, so "the device outer is on" and "the host
// still did the arithmetic" are different, visible numbers rather than the same
// silent one.  The claim a device outer makes is `host_body_calls == 0` over the
// segment; a claim with no counter behind it is a review comment, not a gate.
//
// ---------------------------------------------------------------------------
// WHY IT IS CHEAP ENOUGH TO LEAVE ON
// ---------------------------------------------------------------------------
//
// ONE RELAXED ATOMIC PER SWEEP, NOT PER NODE OR PER SURFACE.  BICGCMFD::updpsi
// walks nxyz nodes and BICGCMFD::upddhat walks nsurf surfaces; the counter is
// bumped ONCE around each of those loops, so an APR1400 outer pays four
// increments, not 100k.  Nodal::updateConstant is the same shape -- it is called
// per node from inside an OpenMP loop, so the count is taken around the loop,
// where it is also the number a reader wants ("did this drive recompute the
// constants at all", not "how many nodes were dirty").
//
// Relaxed ordering is exactly the guarantee needed: nothing reads a counter to
// make a decision, they exist only to be printed after the solve.
//
// HEADER-ONLY, so BICGCMFD.cpp, Nodal.cpp, CudaOuterGraph.cu and main.cpp all
// see the SAME counters with no new translation unit and no link-order
// question.  `inline` on the function-local static gives one instance per
// process in every build, which is the same idiom rasberyOuterSegment() and
// rasberyGpuNodalFullEnabled() already use.

#include <atomic>
#include <cstdint>

namespace rasbery::hostouter {

/// One counter per host outer body.  Names are the METHOD names, so a reader
/// grepping the receipt lands on the function.
struct Counters {
    std::atomic<std::uint64_t> updpsi{0};
    std::atomic<std::uint64_t> updjnet{0};
    std::atomic<std::uint64_t> upddhat{0};
    std::atomic<std::uint64_t> upddtil{0};
    /// Nodal::updateConstant, counted once per full sweep over the nodes.
    std::atomic<std::uint64_t> nodal_constants{0};
};

[[nodiscard]] inline Counters& counters() {
    static Counters c;
    return c;
}

inline void bumpHostBody(std::atomic<std::uint64_t>& c) {
    c.fetch_add(1, std::memory_order_relaxed);
}

/// A plain snapshot, for the receipt.  Loading through this rather than
/// touching the atomics at the print site keeps the one place that knows the
/// memory order in this header.
struct Snapshot {
    std::uint64_t updpsi          = 0;
    std::uint64_t updjnet         = 0;
    std::uint64_t upddhat         = 0;
    std::uint64_t upddtil         = 0;
    std::uint64_t nodal_constants = 0;

    [[nodiscard]] std::uint64_t total() const {
        return updpsi + updjnet + upddhat + upddtil + nodal_constants;
    }
};

[[nodiscard]] inline Snapshot snapshotOf(const Counters& c) {
    Snapshot s;
    s.updpsi          = c.updpsi.load(std::memory_order_relaxed);
    s.updjnet         = c.updjnet.load(std::memory_order_relaxed);
    s.upddhat         = c.upddhat.load(std::memory_order_relaxed);
    s.upddtil         = c.upddtil.load(std::memory_order_relaxed);
    s.nodal_constants = c.nodal_constants.load(std::memory_order_relaxed);
    return s;
}

[[nodiscard]] inline Snapshot snapshot() { return snapshotOf(counters()); }

// ---------------------------------------------------------------------------
// Rev.7.1 W3 item 1: THE CLAIM IS ABOUT THE SEGMENT, SO MEASURE THE SEGMENT
// ---------------------------------------------------------------------------
//
// The run-wide counters above cannot state the claim a device outer actually
// makes.  `upddtil` is the example that proves it: 69 of them on kngr_238, all
// from Driver.h:1288 and Driver.h:1988 -- the once-per-SolveLoop /
// once-per-ReconvergeFlux d-tilde build that runs BEFORE any segment exists and
// is the segment's own precondition.  A run-wide 69 reads as "the host still
// does upddtil" when the truthful statement is "no device outer ran one".
//
// The same distinction is what makes the constants gate reportable: after the
// gate, the sweeps that remain are the ones a material change genuinely
// required, and every one of them happens OUTSIDE a segment (a material change
// is what ends one).  In-segment is therefore the number that has to be zero.
//
// Measured as a DELTA around the segment rather than by a mode flag, so a body
// called from anywhere -- a hook, a nested call, a thread the runner does not
// know about -- is still attributed to the segment that was running.

[[nodiscard]] inline Counters& segmentCounters() {
    static Counters c;
    return c;
}

/// Adds the run-wide movement since `before` into the segment counters.
inline void accumulateSegment(const Snapshot& before) {
    const Snapshot now = snapshot();
    Counters&      s   = segmentCounters();
    s.updpsi.fetch_add(now.updpsi - before.updpsi, std::memory_order_relaxed);
    s.updjnet.fetch_add(now.updjnet - before.updjnet, std::memory_order_relaxed);
    s.upddhat.fetch_add(now.upddhat - before.upddhat, std::memory_order_relaxed);
    s.upddtil.fetch_add(now.upddtil - before.upddtil, std::memory_order_relaxed);
    s.nodal_constants.fetch_add(now.nodal_constants - before.nodal_constants,
                                std::memory_order_relaxed);
}

/// RAII around one segment.  A guard and not a pair of calls because runSegment
/// has a dozen early returns (every launchFailed/hookFailed path), and a claim
/// that is only measured on the success path is not a claim.
class SegmentScope {
public:
    SegmentScope() : _before(snapshot()) {}
    ~SegmentScope() { accumulateSegment(_before); }
    SegmentScope(const SegmentScope&)            = delete;
    SegmentScope& operator=(const SegmentScope&) = delete;

private:
    Snapshot _before;
};

} // namespace rasbery::hostouter
