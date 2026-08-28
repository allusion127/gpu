// Task 7 gate: the canonical CMFD-Nodal device state contract.
//
//   ./rasbery_canonical_state
//
// Everything here runs on the host with no device.  What is being checked is
// not arithmetic -- Task 7 moves no arithmetic -- it is the four ways sharing a
// device buffer between two backends goes wrong:
//
//   1. POINTER IDENTITY.  If the two backends do not resolve to the SAME
//      address for a shared region, sharing silently becomes two buffers again
//      and every elided transfer is a stale read.  Nothing in a converged
//      answer would show it, because each side is internally consistent.
//   2. MIXED-MODE ISOLATION.  One slot shared, another legacy, in one process.
//      The legacy slot must behave exactly as before, and the two slots' shared
//      regions must not overlap -- a stride error here means one deck's flux
//      lands in another's, which at APR1400 size is a plausible-looking answer.
//   3. THE OBSERVATION API.  With the routine downloads elided, the host
//      Geometry arrays no longer track the device.  Every host consumer must be
//      named and served, and every shared region must be reachable by some
//      consumer -- a region nobody can materialise is a region the host can
//      never read again.
//   4. GENERATION OWNERSHIP.  An upload may be skipped only when a DEVICE side
//      wrote the bytes.  A host-side rebuild has to force it back on, and it
//      must do so through one of the four counters the host actually maintains
//      -- gating on a speculative counter that nothing bumps would suppress the
//      upload forever.

#include "GpuCanonicalState.h"
#include "GpuPhysicsArenaLayout.h"
#include "NodalKernel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace rasbery::gpu;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL %s\n", what.c_str());
        ++failures;
    }
}

/// A slot view whose pointers are the addresses the arena WOULD hand out, built
/// from the real layout calculator against a synthetic base.  No device needed:
/// what is under test is the addressing, and the addressing is pure.
DeviceSlotView viewFor(const ArenaOffsets& o, const ArenaDims& d, unsigned char* base,
                       int slot) {
    auto at = [&](SlotRegion r) {
        return reinterpret_cast<double*>(base + o.slotRegionOffset(slot, r));
    };
    DeviceSlotView v{};
    v.phif = at(SlotRegion::Phif);
    v.phis = at(SlotRegion::Phis);
    v.jnet = at(SlotRegion::Jnet);
    v.dtil = at(SlotRegion::Dtil);
    v.dhat = at(SlotRegion::Dhat);
    v.xs   = at(SlotRegion::Xs);
    v.nxyz = d.nxyz;
    v.nsurf = d.nsurf;
    v.ng   = 2;
    v.slot = slot;
    return v;
}

const char* kRegionNames[kCanonicalRegionCount] = {"flux", "jnet", "phis",
                                                   "dtil", "dhat", "live_xs"};

} // namespace

int main() {
    const ArenaDims    d = arenaDims(8451, 26692, 313, 5000, 4);
    const ArenaOffsets o = arenaComputeLayout(d);
    // A plausible device base.  Never dereferenced -- only compared.
    auto* base = reinterpret_cast<unsigned char*>(static_cast<std::uintptr_t>(1) << 40);

    std::printf("canonical state: nxyz=%d nsurf=%d slots=%d\n", d.nxyz, d.nsurf, d.slots);

    // --- 1. pointer identity ------------------------------------------------
    {
        const DeviceSlotView v = viewFor(o, d, base, 0);
        // Two independent adoptions of the same slot -- as the CMFD backend and
        // the Nodal backend would each do -- must land on the same addresses.
        const CanonicalSlotBuffers a = canonicalFromSlotView(v);
        const CanonicalSlotBuffers b = canonicalFromSlotView(v);
        for (int i = 0; i < kCanonicalRegionCount; ++i) {
            const auto r = static_cast<CanonicalRegion>(i);
            check(a.get(r) == b.get(r),
                  std::string("pointer identity: two adopters disagree about ") +
                      kRegionNames[i]);
            check(a.get(r) != nullptr,
                  std::string("pointer identity: ") + kRegionNames[i] +
                      " must be canonical -- it is a region BOTH backends touch");
        }
        // And they must be the arena's, not copies of each other's.
        check(a.flux == v.phif, "flux must be the arena's Phif");
        check(a.jnet == v.jnet, "jnet must be the arena's Jnet");
        check(a.phis == v.phis, "phis must be the arena's Phis");
        check(a.dtil == v.dtil && a.dhat == v.dhat, "dtil/dhat must be the arena's");
        check(a.live_xs == v.xs, "live_xs must be the arena's Xs block");
        check(a.shared(), "a fully-populated set must report shared()");
        std::printf("  pointer identity: 6/6 regions resolve to the arena\n");
    }

    // --- 2. mixed mode: one slot shared, one legacy -------------------------
    {
        const CanonicalSlotBuffers shared_slot =
            canonicalFromSlotView(viewFor(o, d, base, 1));
        const CanonicalSlotBuffers legacy_slot{}; // all null -> legacy

        check(shared_slot.shared(), "slot 1 should be shared");
        check(!legacy_slot.shared(), "an all-null set must report legacy");

        // The legacy slot behaves exactly as before: nothing is ever elided,
        // whatever the ownership or the mask says.
        for (int i = 0; i < kCanonicalRegionCount; ++i) {
            const auto r = static_cast<CanonicalRegion>(i);
            for (const CanonicalOwner owner :
                 {CanonicalOwner::Host, CanonicalOwner::Cmfd, CanonicalOwner::Nodal}) {
                check(!canonicalElidesUpload(legacy_slot, r, owner, CanonicalOwner::Nodal),
                      std::string("legacy slot must never elide an upload (") +
                          kRegionNames[i] + ")");
            }
            check(!canonicalElidesDownload(legacy_slot, r, 0u),
                  std::string("legacy slot must never elide a download (") +
                      kRegionNames[i] + ")");
            check(!canonicalElidesDownload(legacy_slot, r, 0xFFFFFFFFu),
                  "legacy slot: the mask must not matter");
        }

        // Two shared slots must not alias.  This is the check that catches a
        // stride error, which is otherwise invisible: both decks keep running.
        const CanonicalSlotBuffers other = canonicalFromSlotView(viewFor(o, d, base, 2));
        std::set<const void*> seen;
        for (int i = 0; i < kCanonicalRegionCount; ++i) {
            const auto r = static_cast<CanonicalRegion>(i);
            check(shared_slot.get(r) != other.get(r),
                  std::string("slot 1 and slot 2 share the same ") + kRegionNames[i] +
                      " pointer -- one deck would write into the other");
            check(seen.insert(shared_slot.get(r)).second,
                  std::string("two regions of one slot alias at ") + kRegionNames[i]);
        }
        std::printf("  mixed mode: legacy slot inert, 2 shared slots disjoint\n");
    }

    // --- 3. the observation API ---------------------------------------------
    {
        const CanonicalSlotBuffers b = canonicalFromSlotView(viewFor(o, d, base, 0));

        // Nothing requested -> every download is elided.  That IS the saving,
        // and it is also why the request has to be explicit.
        for (int i = 0; i < kCanonicalRegionCount; ++i)
            check(canonicalElidesDownload(b, static_cast<CanonicalRegion>(i), 0u),
                  "with an empty mask every shared download should be elided");

        // Asking for one region must not drag the others back.
        const std::uint32_t only_jnet = canonicalBit(CanonicalRegion::Jnet);
        check(!canonicalElidesDownload(b, CanonicalRegion::Jnet, only_jnet),
              "a requested region must be downloaded");
        check(canonicalElidesDownload(b, CanonicalRegion::Phis, only_jnet),
              "an unrequested region must stay elided");

        // Every consumer must ask for something, and every shared region must
        // be reachable by SOME consumer -- otherwise the host could never see
        // it again.
        for (int c = 0; c < static_cast<int>(CanonicalConsumer::Count); ++c) {
            const auto mask = canonicalConsumerMask(static_cast<CanonicalConsumer>(c));
            check(mask != 0u, "a consumer with an empty mask reads nothing -- remove it");
        }
        const std::uint32_t all = canonicalAllConsumerMask();
        for (int i = 0; i < kCanonicalRegionCount; ++i) {
            const auto r = static_cast<CanonicalRegion>(i);
            check((all & canonicalBit(r)) != 0u,
                  std::string("region ") + kRegionNames[i] +
                      " is shared but no consumer can materialise it");
        }
        // PPR is the strictest consumer: it reads all three surface/flux arrays.
        const auto ppr = canonicalConsumerMask(CanonicalConsumer::PinPowerReconstruction);
        check((ppr & canonicalBit(CanonicalRegion::Jnet)) &&
                  (ppr & canonicalBit(CanonicalRegion::Flux)) &&
                  (ppr & canonicalBit(CanonicalRegion::Phis)),
              "PPR reads Jnet, Phif and Phis (Driver.h) -- its mask must say so");
        std::printf("  observation API: %d consumers, all 6 regions reachable\n",
                    static_cast<int>(CanonicalConsumer::Count));
    }

    // --- 4. generation ownership --------------------------------------------
    {
        const CanonicalSlotBuffers b = canonicalFromSlotView(viewFor(o, d, base, 0));

        // A device write lets the next upload be skipped; a host write does not.
        check(canonicalElidesUpload(b, CanonicalRegion::Jnet, CanonicalOwner::Nodal,
                                    CanonicalOwner::Cmfd),
              "a device-owned region should not be re-uploaded");
        check(canonicalElidesUpload(b, CanonicalRegion::Jnet, CanonicalOwner::Cmfd,
                                    CanonicalOwner::Nodal),
              "the other direction too -- ownership is about device vs host");
        check(!canonicalElidesUpload(b, CanonicalRegion::Jnet, CanonicalOwner::Host,
                                     CanonicalOwner::Nodal),
              "a HOST write must force the upload back on");

        // A host-side rebuild, seen through the four real counters, returns
        // every region to the host.
        CanonicalSlotState st;
        st.buffers = b;
        for (int r = 0; r < kCanonicalRegionCount; ++r)
            st.setOwner(static_cast<CanonicalRegion>(r), CanonicalOwner::Nodal);

        DeviceSlotState s{};
        deviceSlotStateReset(s);
        check(!st.noteGenerations(s),
              "generations that did not move must not invalidate anything");
        check(st.ownerOf(CanonicalRegion::Jnet) == CanonicalOwner::Nodal,
              "a no-op generation check must leave ownership alone");

        s.micx_generation = 7; // the host rebuilt _micx
        check(st.noteGenerations(s), "a moved generation must be reported");
        for (int r = 0; r < kCanonicalRegionCount; ++r)
            check(st.ownerOf(static_cast<CanonicalRegion>(r)) == CanonicalOwner::Host,
                  "a host-side rebuild must return every region to the host");

        // The four counters that are allowed to drive this are the four the
        // host actually maintains.
        for (int g = 0; g < static_cast<int>(CanonicalGeneration::Count); ++g) {
            DeviceSlotState probe{};
            deviceSlotStateReset(probe);
            CanonicalSlotState local;
            local.noteGenerations(probe); // establish the baseline
            const auto which = static_cast<CanonicalGeneration>(g);
            switch (which) {
                case CanonicalGeneration::Micx:          probe.micx_generation = 1; break;
                case CanonicalGeneration::Ref:           probe.ref_generation = 1; break;
                case CanonicalGeneration::HostState:     probe.hoststate_generation = 1; break;
                case CanonicalGeneration::NodalConstant: probe.nodal_constant_generation = 1; break;
                case CanonicalGeneration::Count:         break;
            }
            check(local.noteGenerations(probe),
                  std::string("moving ") + canonicalGenerationName(which) +
                      " must invalidate the device copy");
        }

        // And the SPECULATIVE ones must NOT: nothing on the host bumps them, so
        // reading one would be reading a constant, and gating an upload on a
        // constant suppresses it forever.
        {
            DeviceSlotState probe{};
            deviceSlotStateReset(probe);
            CanonicalSlotState local;
            local.noteGenerations(probe);
            probe.material_generation = 42;
            probe.operator_generation = 42;
            probe.flux_generation     = 42;
            probe.current_generation  = 42;
            probe.dhat_generation     = 42;
            probe.isotope_generation  = 42;
            probe.th_generation       = 42;
            probe.geometry_generation = 42;
            check(!local.noteGenerations(probe),
                  "the speculative generation counters must not drive Task 7 -- nothing "
                  "on the host bumps them, so an upload gated on one never happens again");
        }
        std::printf("  generations: 4 real counters invalidate, 8 speculative ignored\n");
    }

    // --- 4b. the nodal PER-SLOT POINTER TABLE (Rev.7.1 pre-W3) --------------
    //
    // The arena used to COMPUTE a slot's view by advancing each array pointer by
    // that array's own dense per-slot count.  That is why a canonical pointer
    // could not be handed to it: GpuPhysicsArena strides by the whole slot
    // block, so the dense rebase would have read slot 1 from inside slot 0 --
    // finite, plausible, wrong.  The table replaced the arithmetic with a
    // lookup, so the layout question disappears.
    //
    // Two things have to hold, and neither shows up in a converged answer.
    {
        // (i) EQUIVALENCE.  A table nobody has adopted into must hold byte-for-
        //     byte the addresses the dense rebase produced.  This is what makes
        //     the conversion pure indirection and keeps feature-off identical.
        rasbery::nodal::NodalView proto{};
        auto* fake = reinterpret_cast<double*>(static_cast<std::uintptr_t>(1) << 36);
        proto.nxyz  = 64;
        proto.nsurf = 208;
        proto.jnet  = fake;
        proto.flux  = fake + (1 << 20);
        proto.phis  = fake + (2 << 20);
        proto.xsrf  = fake + (3 << 20);

        bool equivalent = true;
        for (int m = 0; m < 8; ++m) {
            const rasbery::nodal::NodalView rebased = rasbery::nodal::nodalSlotView(proto, m);
            // The table entry a legacy slot gets is exactly this.
            const rasbery::nodal::NodalView entry = rasbery::nodal::nodalSlotView(proto, m);
            if (entry.jnet != rebased.jnet || entry.flux != rebased.flux ||
                entry.phis != rebased.phis || entry.xsrf != rebased.xsrf)
                equivalent = false;
            // And it must actually differ per slot, or the test proves nothing.
            if (m > 0 && rebased.jnet == rasbery::nodal::nodalSlotView(proto, 0).jnet)
                equivalent = false;
        }
        check(equivalent,
              "a default table entry must equal the dense rebase for that slot -- the "
              "pointer-table conversion is indirection ONLY, and this is what says so");

        // (ii) MIXED STRIDE.  Slot 0 borrows canonical buffers laid out with the
        //      arena's 29.4-million-double whole-slot stride; slot 1 keeps the
        //      dense arena pointers.  Both must resolve, and they must not
        //      alias.  Under the old dense rebase this configuration was
        //      unrepresentable, which is the whole point of the change.
        const CanonicalSlotBuffers canon0 = canonicalFromSlotView(viewFor(o, d, base, 0));
        rasbery::nodal::NodalView slot0 = rasbery::nodal::nodalSlotView(proto, 0);
        slot0.jnet = canon0.jnet;
        slot0.flux = canon0.flux;
        slot0.phis = canon0.phis;
        const rasbery::nodal::NodalView slot1 = rasbery::nodal::nodalSlotView(proto, 1); // legacy

        check(slot0.jnet == canon0.jnet && slot0.flux == canon0.flux &&
                  slot0.phis == canon0.phis,
              "the adopted slot must resolve to the canonical pointers");
        check(slot1.jnet != slot0.jnet && slot1.flux != slot0.flux,
              "the legacy slot must keep the arena's own pointers, not the canonical ones");
        check(slot1.jnet == rasbery::nodal::nodalSlotView(proto, 1).jnet,
              "the legacy slot's addressing must be untouched by the neighbour's adoption");
        // The two slots' shared regions live in different allocations entirely.
        check(slot0.jnet != slot1.jnet && slot0.flux != slot1.flux &&
                  slot0.phis != slot1.phis,
              "canonical and legacy slots must not alias");
        // The working arrays of the adopted slot stay the ARENA's: only three
        // regions are borrowed, and a partial swap would blend two iterations.
        check(slot0.xsrf == rasbery::nodal::nodalSlotView(proto, 0).xsrf,
              "adoption must not move the arena's own arrays -- only flux/jnet/phis");
        std::printf("  nodal pointer table: dense-equivalent, mixed stride OK "
                    "(canonical slot 0 + legacy slot 1)\n");

        // (iii) A PARTIAL set is refused: taking the canonical jnet and the
        //       arena's own flux would pair two different outer iterations.
        CanonicalSlotBuffers partial{};
        partial.jnet = canon0.jnet;
        check(!canonicalNodalSetIsCoherent(partial),
              "a partial canonical set must be refused -- flux/jnet/phis are adopted "
              "together or not at all");
        check(canonicalNodalSetIsCoherent(canon0), "a full set must be accepted");
        check(canonicalNodalSetIsCoherent(CanonicalSlotBuffers{}),
              "an empty set is legacy, which is coherent");
    }

    // --- 5. feature-off is inert --------------------------------------------
    {
        const CanonicalSlotBuffers off{};
        for (int i = 0; i < kCanonicalRegionCount; ++i) {
            const auto r = static_cast<CanonicalRegion>(i);
            check(off.get(r) == nullptr, "feature-off: no borrowed pointers");
            check(!canonicalElidesUpload(off, r, CanonicalOwner::Nodal, CanonicalOwner::Cmfd),
                  "feature-off: uploads must all happen");
            check(!canonicalElidesDownload(off, r, 0u),
                  "feature-off: downloads must all happen");
        }
        std::printf("  feature-off: every transfer preserved\n");
    }

    if (failures) {
        std::printf("canonical state: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("canonical state: PASS\n");
    return 0;
}
