#pragma once

// Canonical CMFD-Nodal device state -- Rev.7.1 plan Task 7.
//
// THE HOLE THIS CLOSES.  CMFD and Nodal each keep their own device copy of the
// same physics.  Every outer therefore pays a round trip that exists only
// because neither side knows the other already put the bytes on the device:
//
//     CMFD  updjnet   ->  jnet on the device
//           D2H jnet                                (CudaBICGBackend)
//     Nodal H2D jnet, H2D flux                      (CudaXsReconBackend:1974-1979)
//           drive     ->  jnet, phis on the device
//           D2H jnet, D2H phis                      (CudaXsReconBackend:1989-1992)
//     CMFD  H2D jnet   (upddhat reads it back)
//
// At APR1400 size that is ~1.7 MB of surface traffic per outer in each
// direction, for data that never left the device conceptually.  The fix is not
// a faster copy: it is to make ONE buffer canonical and have both backends
// borrow the pointer.
//
// WHAT THIS HEADER IS.  The contract, and nothing else: which regions are
// canonical, who owns each one at which point of the outer, which transfers
// that makes redundant, and when the host is still allowed to look.  It is
// CUDA-free and pure so the decisions are testable without a device -- the
// backends ask it what to do rather than each deciding for themselves, which is
// the only way two backends can agree about a buffer they share.
//
// DEFAULT OFF (RASBERY_GPU_SHARED_STATE).  Sharing changes which memory the
// host's Geometry arrays reflect, so it is opt-in and the feature-off path is
// byte-identical by construction: canonicalSlotBuffers() returns nulls, every
// elision predicate returns false, and both backends take exactly the branches
// they took before.

#include "GpuPhysicsTypes.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace rasbery::gpu {

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// The six regions Task 7 makes canonical.  These are exactly the ones BOTH
/// backends touch; everything else stays private to its owner, because sharing
/// a buffer only one side reads buys nothing and costs a coupling.
enum class CanonicalRegion : int {
    Flux = 0,  ///< Geometry::Phif, [l*ng + ig]
    Jnet,      ///< Geometry::Jnet, [ls*ng + ig]
    Phis,      ///< Geometry::Phis, [ls*ng + ig]
    Dtil,      ///< CMFD::_dtil,    [ls*ng + ig]
    Dhat,      ///< CMFD::_dhat,    [ls*ng + ig]
    LiveXs,    ///< XSSet live macroscopic block, [ig*nxyz + l] per scalar slot
    Count
};

inline constexpr int kCanonicalRegionCount = static_cast<int>(CanonicalRegion::Count);

inline const char* canonicalRegionName(CanonicalRegion r) {
    switch (r) {
        case CanonicalRegion::Flux:   return "flux";
        case CanonicalRegion::Jnet:   return "jnet";
        case CanonicalRegion::Phis:   return "phis";
        case CanonicalRegion::Dtil:   return "dtil";
        case CanonicalRegion::Dhat:   return "dhat";
        case CanonicalRegion::LiveXs: return "live_xs";
        case CanonicalRegion::Count:  break;
    }
    return "?";
}

/// Which side wrote a region last.  Ownership is the whole mechanism: a
/// transfer is redundant exactly when the side about to read already owns the
/// bytes, or when the side about to write is the one that produced them.
enum class CanonicalOwner : int {
    Host = 0, ///< the host arrays are authoritative (legacy, or after a host write)
    Cmfd,     ///< last written by the CMFD backend on the device
    Nodal     ///< last written by the Nodal backend on the device
};

inline const char* canonicalOwnerName(CanonicalOwner o) {
    switch (o) {
        case CanonicalOwner::Host:  return "host";
        case CanonicalOwner::Cmfd:  return "cmfd";
        case CanonicalOwner::Nodal: return "nodal";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// RASBERY_GPU_SHARED_STATE, default OFF.  Read once, like every other gate.
inline bool canonicalSharedStateEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_SHARED_STATE");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

// ---------------------------------------------------------------------------
// Borrowed pointers
// ---------------------------------------------------------------------------

/// One slot's canonical device pointers, borrowed from GpuPhysicsArena.
///
/// BORROWED, NOT OWNED -- and that is the reason the arena exists in the shape
/// it does.  The arena fixes every per-slot address in reserve() and never
/// allocates again (GpuPhysicsArena.h), so a backend may bake these into a
/// captured graph and they stay valid for the run.  A backend that allocated
/// its own and handed the pointer over would break the moment it resized.
///
/// A null member means "not canonical for this slot": the backend keeps its
/// private allocation for that region and behaves exactly as before.  That is
/// what makes MIXED MODE work -- one slot shared, another legacy, in the same
/// process, with no third code path.
struct CanonicalSlotBuffers {
    double* flux    = nullptr;
    double* jnet    = nullptr;
    double* phis    = nullptr;
    double* dtil    = nullptr;
    double* dhat    = nullptr;
    double* live_xs = nullptr;

    [[nodiscard]] double* get(CanonicalRegion r) const {
        switch (r) {
            case CanonicalRegion::Flux:   return flux;
            case CanonicalRegion::Jnet:   return jnet;
            case CanonicalRegion::Phis:   return phis;
            case CanonicalRegion::Dtil:   return dtil;
            case CanonicalRegion::Dhat:   return dhat;
            case CanonicalRegion::LiveXs: return live_xs;
            case CanonicalRegion::Count:  break;
        }
        return nullptr;
    }

    /// True when this slot shares ANY region.  A slot that shares none is a
    /// legacy slot and every predicate below is inert for it.
    [[nodiscard]] bool shared() const {
        for (int i = 0; i < kCanonicalRegionCount; ++i)
            if (get(static_cast<CanonicalRegion>(i)) != nullptr) return true;
        return false;
    }
};

/// Build the borrowed set for one slot from an arena slot view.
///
/// The layouts are the HOST's -- dtil/dhat/jnet are [ls*ng + ig] and flux is
/// [l*ng + ig] (see the note on DeviceSlotView) -- so a canonical buffer is the
/// same bytes both backends already index, not a transposed copy that would
/// need a conversion kernel on every handover.
inline CanonicalSlotBuffers canonicalFromSlotView(const DeviceSlotView& v) {
    CanonicalSlotBuffers b;
    b.flux    = v.phif;
    b.jnet    = v.jnet;
    b.phis    = v.phis;
    b.dtil    = v.dtil;
    b.dhat    = v.dhat;
    b.live_xs = v.xs;
    return b;
}

// ---------------------------------------------------------------------------
// Sec 6.4  Transfer elision
// ---------------------------------------------------------------------------

/// May the upload of `region` into `reader` be skipped?
///
/// Yes exactly when the region is canonical AND the device already holds the
/// current bytes -- that is, the last writer was a DEVICE side.  If the host
/// wrote last (a perturbation, a restart, a rod move) the upload is still
/// required, and saying so here rather than in each backend is what stops the
/// two from disagreeing about it.
///
/// `reader` is passed but deliberately unused for the decision: it does not
/// matter WHICH device side wrote, only that a device side did.  It is in the
/// signature because the call sites read better with it and because a future
/// per-region ownership rule (Task 8's nodal phase graph) will need it.
[[nodiscard]] inline bool canonicalElidesUpload(const CanonicalSlotBuffers& b,
                                                CanonicalRegion region,
                                                CanonicalOwner last_writer,
                                                CanonicalOwner reader) {
    (void)reader;
    if (b.get(region) == nullptr) return false;      // not shared -> copy as before
    return last_writer != CanonicalOwner::Host;      // host wrote last -> must upload
}

/// May the download of `region` back to the host be skipped?
///
/// Yes exactly when the region is canonical and the host has not ASKED to see
/// it.  This is the half that actually saves the traffic, and it is also the
/// half that can produce a silently stale host array -- so the request is
/// explicit (materialize) rather than inferred.
[[nodiscard]] inline bool canonicalElidesDownload(const CanonicalSlotBuffers& b,
                                                  CanonicalRegion region,
                                                  std::uint32_t materialize_mask) {
    if (b.get(region) == nullptr) return false;
    return (materialize_mask & (1u << static_cast<int>(region))) == 0u;
}

// ---------------------------------------------------------------------------
// Sec 3.3  The materialize mask -- the observation API
// ---------------------------------------------------------------------------

[[nodiscard]] inline constexpr std::uint32_t canonicalBit(CanonicalRegion r) {
    return 1u << static_cast<int>(r);
}

/// Who on the host still reads these arrays, and what each one needs.
///
/// This table IS the safety argument for eliding the downloads: with shared
/// state the Geometry arrays no longer track the device automatically, so
/// every host consumer has to be named and served deliberately.  Missing one
/// is a stale read that produces plausible numbers.
enum class CanonicalConsumer : int {
    /// Driver.h: pin_power_reconstruction.reset(1/eigv, Jnet(), Phif(), Phis())
    PinPowerReconstruction = 0,
    /// Driver.h: input_output.AddResult(geometry, ...) and the output pack
    ResultOutput,
    /// CMFD::upddhat / updjnet on the HOST path (the warm-up sweeps, and any
    /// fallback out of the resident path)
    HostCmfdOuter,
    /// RASBERY_NODAL_DUMP / RASBERY_CMFD_DUMP and the debug hashes
    Diagnostics,
    Count
};

/// What each consumer must see before it runs.
[[nodiscard]] inline constexpr std::uint32_t canonicalConsumerMask(CanonicalConsumer c) {
    switch (c) {
        case CanonicalConsumer::PinPowerReconstruction:
            return canonicalBit(CanonicalRegion::Jnet) |
                   canonicalBit(CanonicalRegion::Flux) |
                   canonicalBit(CanonicalRegion::Phis);
        case CanonicalConsumer::ResultOutput:
            return canonicalBit(CanonicalRegion::Flux);
        case CanonicalConsumer::HostCmfdOuter:
            // The host outer reads flux and jnet and WRITES dtil/dhat, so it
            // needs the first two materialised; the two it writes become
            // host-owned and are uploaded again by the elision rule above.
            return canonicalBit(CanonicalRegion::Flux) |
                   canonicalBit(CanonicalRegion::Jnet);
        case CanonicalConsumer::Diagnostics:
            // A dump is supposed to show everything, so it asks for everything.
            return canonicalBit(CanonicalRegion::Flux) |
                   canonicalBit(CanonicalRegion::Jnet) |
                   canonicalBit(CanonicalRegion::Phis) |
                   canonicalBit(CanonicalRegion::Dtil) |
                   canonicalBit(CanonicalRegion::Dhat) |
                   canonicalBit(CanonicalRegion::LiveXs);
        case CanonicalConsumer::Count:
            break;
    }
    return 0u;
}

/// Every region some consumer needs.  A region that appears in no consumer mask
/// would never be materialised, so it must not be shared -- checked by the
/// contract test rather than left to inspection.
[[nodiscard]] inline constexpr std::uint32_t canonicalAllConsumerMask() {
    std::uint32_t m = 0u;
    for (int i = 0; i < static_cast<int>(CanonicalConsumer::Count); ++i)
        m |= canonicalConsumerMask(static_cast<CanonicalConsumer>(i));
    return m;
}

// ---------------------------------------------------------------------------
// Generation ownership
// ---------------------------------------------------------------------------

/// The four generation counters that are REAL -- each mirrors a counter the
/// host already maintains, so gating an upload on one is safe today.
///
/// DeviceSlotState carries eight more (geometry_, material_, operator_, flux_,
/// current_, dhat_, isotope_, th_generation) which nothing on the host bumps
/// yet; GpuSlotControl.h calls them speculative for exactly this reason.  An
/// upload gated on one of those would be suppressed forever, because the
/// counter never moves.  Task 7 therefore uses only these four, and the
/// contract test enforces it.
enum class CanonicalGeneration : int {
    Micx = 0,     ///< XSSet.h:180 _micx_generation -- host rebuilt _micx/_lmpx
    Ref,          ///< XSSet.h:198 _ref_generation  -- reference blocks rebuilt
    HostState,    ///< XSSet.h:190 _hoststate_generation -- host wrote _xs/_iden
    NodalConstant,///< Nodal.h:124 _const_generation -- updateConstant products stale
    Count
};

inline const char* canonicalGenerationName(CanonicalGeneration g) {
    switch (g) {
        case CanonicalGeneration::Micx:          return "micx_generation";
        case CanonicalGeneration::Ref:           return "ref_generation";
        case CanonicalGeneration::HostState:     return "hoststate_generation";
        case CanonicalGeneration::NodalConstant: return "nodal_constant_generation";
        case CanonicalGeneration::Count:         break;
    }
    return "?";
}

/// Read one of the four from the device control block.
[[nodiscard]] inline std::uint64_t canonicalGenerationOf(const DeviceSlotState& s,
                                                         CanonicalGeneration g) {
    switch (g) {
        case CanonicalGeneration::Micx:          return s.micx_generation;
        case CanonicalGeneration::Ref:           return s.ref_generation;
        case CanonicalGeneration::HostState:     return s.hoststate_generation;
        case CanonicalGeneration::NodalConstant: return s.nodal_constant_generation;
        case CanonicalGeneration::Count:         break;
    }
    return 0;
}

/// One slot's shared-state bookkeeping.  Small, trivially copyable, and host
/// side: the device never reads it, because ownership is a HOST decision about
/// which transfers to issue.
struct CanonicalSlotState {
    CanonicalSlotBuffers buffers{};
    /// Last writer per region.  Starts Host: nothing is on the device yet.
    CanonicalOwner owner[kCanonicalRegionCount] = {};
    /// The generations the device copy was built from, so a host-side rebuild
    /// forces the upload back on.
    std::uint64_t seen[static_cast<int>(CanonicalGeneration::Count)] = {};

    [[nodiscard]] CanonicalOwner ownerOf(CanonicalRegion r) const {
        return owner[static_cast<int>(r)];
    }
    void setOwner(CanonicalRegion r, CanonicalOwner o) { owner[static_cast<int>(r)] = o; }

    /// A host-side rebuild invalidates the device copy of everything the host
    /// touched.  Returns true when something actually moved, so the caller can
    /// log a transition instead of guessing at one.
    bool noteGenerations(const DeviceSlotState& s) {
        bool moved = false;
        for (int i = 0; i < static_cast<int>(CanonicalGeneration::Count); ++i) {
            const auto g   = static_cast<CanonicalGeneration>(i);
            const auto now = canonicalGenerationOf(s, g);
            if (now != seen[i]) {
                seen[i] = now;
                moved   = true;
            }
        }
        if (moved)
            for (int r = 0; r < kCanonicalRegionCount; ++r)
                owner[r] = CanonicalOwner::Host;
        return moved;
    }
};

} // namespace rasbery::gpu
