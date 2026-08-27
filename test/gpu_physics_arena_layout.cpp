// Arena layout gate -- Rev.7.1 plan Task 2 Step 1.
//
// The layout calculator is pure arithmetic, so it is checkable WITHOUT a
// device, and this harness is the only place the invariants get checked at all.
// It includes GpuPhysicsArenaLayout.h and nothing else: no CUDA header, no
// runtime, no HDF5.  `cmake -DRASBERY_ENABLE_TESTS=ON` builds it everywhere.
//
// What it proves, at APR1400 dimensions (nxyz 8451, nsurf 26692, nxy 313,
// n_fuel ~5000) and at 1, 8 and 64 slots:
//
//   1. ALIGNMENT -- every region, in every slot, starts on 256 B.  Not just
//      slot 0: the stride has to be a multiple of 256 or slot 1 onwards drift.
//   2. NON-OVERLAP inside a slot -- regions in declaration order are disjoint
//      and all of them fit inside the stride.
//   3. SLOT ISOLATION -- slot i's stride and slot j's never intersect, and no
//      slot reaches the immutable geometry or library blocks.  This is Sec 5.9
//      and it is what makes the per-slot alias rule sound.
//   4. ALIAS LIFETIME, PER SLOT PER PHASE (Sec 4.2) -- scratch users in the
//      same band share bytes; a user is reachable only from the phases that own
//      it.  The Rev.7 failure this replaces: a GLOBAL phase lifetime, which is
//      wrong the moment slot A is in Outer while slot B is in Depletion.
//   5. BUDGET -- per-slot bytes land in the Sec 3.6 band of 200-225 MiB and the
//      64-slot total lands in the 12.8-14.4 GiB the plan predicts.  Rev.7's
//      22-35 MiB compressed tier is gone, and this is the arithmetic that says
//      it did not need to be there.
//
// Run: rasbery_gpu_arena_layout          (silent, exit 0)
//      rasbery_gpu_arena_layout --print  (dumps the region table)

#include "GpuPhysicsArenaLayout.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace rasbery::gpu;

namespace {

int g_failures = 0;

void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

constexpr double kMiB = 1024.0 * 1024.0;

/// APR1400, the deck every gate in this campaign is measured on.
ArenaDims apr1400(int slots) {
    ArenaDims d = arenaDims(8451, 26692, 313, 5000, slots);
    // A representative library: the counts only move the immutable block, which
    // is shared and therefore does not scale with slots.
    d.n_ref_points   = 240;
    d.n_coeff_points = 480;
    d.n_knots        = 4096;
    return d;
}

struct Span {
    std::size_t begin;
    std::size_t end;
    std::string name;
};

bool disjoint(const Span& a, const Span& b) {
    return a.end <= b.begin || b.end <= a.begin;
}

void checkOneLayout(int slots, bool print) {
    const ArenaDims    d = apr1400(slots);
    const ArenaOffsets o = arenaComputeLayout(d);

    CHECK(o.slot_count == static_cast<std::size_t>(slots));
    CHECK(o.per_slot_bytes > 0);
    CHECK(o.total_bytes == o.slot_base + o.slot_count * o.per_slot_bytes);

    // ---- 1. alignment ----------------------------------------------------
    CHECK(o.slot_base % kArenaAlignment == 0);
    CHECK(o.per_slot_bytes % kArenaAlignment == 0);
    for (int i = 0; i < kGeometryRegionCount; ++i)
        CHECK(o.geometry[i].offset % kArenaAlignment == 0);
    for (int i = 0; i < kLibraryRegionCount; ++i)
        CHECK(o.library[i].offset % kArenaAlignment == 0);
    for (int i = 0; i < kSlotRegionCount; ++i)
        CHECK(o.slot[i].offset % kArenaAlignment == 0);
    for (int i = 0; i < kScratchBandCount; ++i)
        CHECK(o.scratch[i].offset % kArenaAlignment == 0);

    // Alignment has to hold for EVERY slot, which is a property of the stride.
    for (int s = 0; s < slots; ++s) {
        CHECK(o.slotBase(s) % kArenaAlignment == 0);
        for (int i = 0; i < kSlotRegionCount; ++i)
            CHECK(o.slotRegionOffset(s, static_cast<SlotRegion>(i)) % kArenaAlignment == 0);
    }

    // ---- 2. non-overlap inside one slot ----------------------------------
    std::vector<Span> inside;
    for (int i = 0; i < kSlotRegionCount; ++i) {
        const ArenaRegion& r = o.slot[i];
        // A zero-byte region is still PLACED (its offset is inside the stride
        // and 256-aligned); it simply spans nothing, so it is trivially
        // disjoint from everything.  Dropping it from the sweep entirely would
        // hide a region that silently lost its elements.
        CHECK(r.offset <= o.per_slot_bytes);
        CHECK(r.offset + r.bytes <= o.per_slot_bytes);
        if (r.bytes == 0) continue;
        inside.push_back({r.offset, r.offset + r.bytes, "slot_region"});
    }
    for (int b = 0; b < kScratchBandCount; ++b) {
        const ArenaRegion& r = o.scratch[b];
        if (r.bytes == 0) continue;
        inside.push_back({r.offset, r.offset + r.bytes,
                          arenaScratchBandName(static_cast<ScratchBand>(b))});
        CHECK(r.offset + r.bytes <= o.per_slot_bytes);
    }
    for (std::size_t i = 0; i < inside.size(); ++i)
        for (std::size_t j = i + 1; j < inside.size(); ++j)
            CHECK(disjoint(inside[i], inside[j]));

    // ---- 3. slot isolation, and immutables out of reach -------------------
    const std::size_t immutable_end =
        o.shared_geometry_bytes + o.shared_library_bytes + o.control_block_bytes;
    CHECK(immutable_end <= o.slot_base);

    // THE CONTROL BLOCK IS OUT OF clearSlotAsync's REACH.  This is the whole
    // reason the four structs left the slot stride: a bulk clear memsets
    // [slotBase, slotBase + per_slot_bytes) and, while they lived inside it,
    // wiped the defaults a refill had just written.
    for (int i = 0; i < kControlRegionCount; ++i) {
        const ArenaRegion& r = o.control[i];
        CHECK(r.offset % kArenaAlignment == 0);
        CHECK(r.offset + r.bytes <= o.slot_base);
        CHECK(r.bytes == static_cast<std::size_t>(slots) *
                             arenaControlElementBytes(static_cast<ControlRegion>(i)));
        for (int s = 0; s < slots; ++s) CHECK(!o.insideSlotStride(s, r.offset, r.bytes));
    }
    // Control arrays are dense and pairwise disjoint, and every slot's element
    // sits inside its own array -- the property `phases[tid]` depends on.
    for (int i = 0; i < kControlRegionCount; ++i) {
        const ControlRegion r     = static_cast<ControlRegion>(i);
        const std::size_t   width = arenaControlElementBytes(r);
        for (int s = 0; s < slots; ++s) {
            const std::size_t at = o.controlOffset(s, r);
            CHECK(at == o.control[i].offset + static_cast<std::size_t>(s) * width);
            CHECK(at + width <= o.control[i].offset + o.control[i].bytes);
            if (s > 0) CHECK(at == o.controlOffset(s - 1, r) + width);
        }
    }
    // The 32-byte hot struct exists so 64 of them are 2 KiB of CONTIGUOUS
    // memory.  Strided by the slot pitch that claim was simply false.
    if (slots == 64)
        CHECK(o.control[static_cast<int>(ControlRegion::SlotPhase)].bytes == 2048);
    for (int i = 0; i < kGeometryRegionCount; ++i)
        CHECK(o.geometry[i].offset + o.geometry[i].bytes <= o.slot_base);
    for (int i = 0; i < kLibraryRegionCount; ++i)
        CHECK(o.library[i].offset + o.library[i].bytes <= o.slot_base);
    // Immutable regions are disjoint from each other too.
    std::vector<Span> immutable;
    for (int i = 0; i < kGeometryRegionCount; ++i)
        if (o.geometry[i].bytes > 0)
            immutable.push_back({o.geometry[i].offset, o.geometry[i].offset + o.geometry[i].bytes,
                                 "geometry"});
    for (int i = 0; i < kLibraryRegionCount; ++i)
        if (o.library[i].bytes > 0)
            immutable.push_back({o.library[i].offset, o.library[i].offset + o.library[i].bytes,
                                 "library"});
    for (int i = 0; i < kControlRegionCount; ++i)
        if (o.control[i].bytes > 0)
            immutable.push_back({o.control[i].offset, o.control[i].offset + o.control[i].bytes,
                                 "control"});
    for (std::size_t i = 0; i < immutable.size(); ++i)
        for (std::size_t j = i + 1; j < immutable.size(); ++j)
            CHECK(disjoint(immutable[i], immutable[j]));

    // Sec 5.9: slot ranges never intersect, and the last one stays in bounds.
    for (int s = 0; s + 1 < slots; ++s)
        CHECK(o.slotBase(s) + o.per_slot_bytes == o.slotBase(s + 1));
    if (slots > 0)
        CHECK(o.slotBase(slots - 1) + o.per_slot_bytes == o.total_bytes);

    // The property stated directly rather than inferred from the stride: no two
    // slots share a byte of ANY region, including the aliased scratch bands.
    for (int a = 0; a < slots; ++a) {
        for (int b = a + 1; b < slots; ++b) {
            const Span sa{o.slotBase(a), o.slotBase(a) + o.per_slot_bytes, "a"};
            const Span sb{o.slotBase(b), o.slotBase(b) + o.per_slot_bytes, "b"};
            CHECK(disjoint(sa, sb));
            for (int u = 0; u < kScratchIdCount; ++u) {
                const ScratchId id = static_cast<ScratchId>(u);
                CHECK(o.scratchOffset(a, id) != o.scratchOffset(b, id));
            }
        }
    }

    // ---- 4. alias lifetime, per slot per phase ---------------------------
    //
    // THE RULE, corrected.  This test used to assert that ALL users of a band
    // share one offset -- which pinned a bug as expected behaviour: bicg_ax and
    // bicg_s are both owned by Outer and are live in the same BiCGSTAB
    // iteration, as are the nodal trl and matrix scratches.  Sharing bytes
    // there means the solver reads its own overwritten intermediates.
    //
    // So: users that can be live together (their owner-phase sets intersect)
    // must be DISJOINT; only users that can never be live together may alias.
    for (int s = 0; s < slots; ++s) {
        for (int u = 0; u < kScratchIdCount; ++u) {
            for (int v = u + 1; v < kScratchIdCount; ++v) {
                const ScratchId a = static_cast<ScratchId>(u);
                const ScratchId b = static_cast<ScratchId>(v);
                const std::size_t abeg = o.scratchOffset(s, a);
                const std::size_t aend = abeg + o.scratchBytes(a);
                const std::size_t bbeg = o.scratchOffset(s, b);
                const std::size_t bend = bbeg + o.scratchBytes(b);
                const bool overlap = abeg < bend && bbeg < aend;

                if (arenaScratchBand(a) != arenaScratchBand(b)) {
                    CHECK(!overlap); // different bands never share bytes
                } else if (arenaScratchCoResident(a, b)) {
                    // Same band, same owner phase -> simultaneously live.
                    if (overlap) {
                        std::fprintf(stderr,
                                     "FAIL: %s and %s share an owner phase AND overlap "
                                     "[%zu,%zu) vs [%zu,%zu)\n",
                                     arenaScratchName(a), arenaScratchName(b), abeg, aend, bbeg,
                                     bend);
                        ++g_failures;
                    }
                } else {
                    // Same band, disjoint phases -> aliasing is the point.
                    CHECK(overlap);
                }
            }
        }
        // Every user fits inside its band.
        for (int u = 0; u < kScratchIdCount; ++u) {
            const ScratchId   id   = static_cast<ScratchId>(u);
            const ScratchBand band = arenaScratchBand(id);
            const std::size_t band_begin = o.slotBase(s) + o.scratch[static_cast<int>(band)].offset;
            const std::size_t band_end   = band_begin + o.scratch[static_cast<int>(band)].bytes;
            CHECK(o.scratchOffset(s, id) >= band_begin);
            CHECK(o.scratchOffset(s, id) + o.scratchBytes(id) <= band_end);
            CHECK(arenaScratchUserElements(id, d) * sizeof(double) <= o.scratchBytes(id));
        }
    }

    // Band size == max over phases of that phase's summed users, computed here
    // independently of arenaScratchBandBytes' greedy assignment.
    for (int b = 0; b < kScratchBandCount; ++b) {
        std::size_t widest = 0;
        for (int p = 0; p < kDevicePhaseCount; ++p) {
            std::size_t sum = 0;
            for (int u = 0; u < kScratchIdCount; ++u) {
                if (static_cast<int>(kScratchSpecs[u].band) != b) continue;
                if (!arenaScratchPhaseAllowed(kScratchSpecs[u].id, static_cast<DevicePhase>(p)))
                    continue;
                sum += arenaScratchUserBytes(kScratchSpecs[u].id, d);
            }
            if (sum > widest) widest = sum;
        }
        CHECK(o.scratch[b].bytes == widest);
    }

    // Every user is owned by at least one phase, and only by phases that make
    // sense for it.  The Rev.7.1 correction in one assertion: the nodal scratch
    // and the depletion temporary share a band and NEVER share a phase, so slot
    // A in Outer and slot B in Depletion cannot collide on the same slot.
    for (int u = 0; u < kScratchIdCount; ++u)
        CHECK(kScratchSpecs[u].owner_phases != 0u);

    CHECK(arenaScratchPhaseAllowed(ScratchId::NodalTrlScratch, DevicePhase::Outer));
    CHECK(!arenaScratchPhaseAllowed(ScratchId::NodalTrlScratch, DevicePhase::DepletionPredictor));
    CHECK(arenaScratchPhaseAllowed(ScratchId::DepletionTemp, DevicePhase::DepletionPredictor));
    CHECK(arenaScratchPhaseAllowed(ScratchId::DepletionTemp, DevicePhase::DepletionCorrector));
    CHECK(!arenaScratchPhaseAllowed(ScratchId::DepletionTemp, DevicePhase::Outer));

    CHECK(arenaScratchPhaseAllowed(ScratchId::BicgAx, DevicePhase::Outer));
    CHECK(!arenaScratchPhaseAllowed(ScratchId::BicgAx, DevicePhase::Xenon));

    CHECK(arenaScratchPhaseAllowed(ScratchId::ThChannel, DevicePhase::ThermalHydraulics));
    CHECK(!arenaScratchPhaseAllowed(ScratchId::ThChannel, DevicePhase::Ppr));
    CHECK(arenaScratchPhaseAllowed(ScratchId::PprCorner, DevicePhase::Ppr));
    CHECK(!arenaScratchPhaseAllowed(ScratchId::PprCorner, DevicePhase::OutputPack));
    CHECK(arenaScratchPhaseAllowed(ScratchId::OutputPackScratch, DevicePhase::OutputPack));

    CHECK(arenaScratchPhaseAllowed(ScratchId::CramPredictor, DevicePhase::DepletionPredictor));
    CHECK(!arenaScratchPhaseAllowed(ScratchId::CramPredictor, DevicePhase::DepletionCorrector));
    CHECK(arenaScratchPhaseAllowed(ScratchId::CramCorrector, DevicePhase::DepletionCorrector));

    // No scratch is reachable from a phase that does no numerical work.
    for (int u = 0; u < kScratchIdCount; ++u) {
        const ScratchId id = static_cast<ScratchId>(u);
        CHECK(!arenaScratchPhaseAllowed(id, DevicePhase::Empty));
        CHECK(!arenaScratchPhaseAllowed(id, DevicePhase::Done));
        CHECK(!arenaScratchPhaseAllowed(id, DevicePhase::Failed));
    }

    // ---- 5. budget (Sec 3.6) ---------------------------------------------
    const double per_slot_mib = static_cast<double>(o.per_slot_bytes) / kMiB;
    CHECK(per_slot_mib >= 200.0);
    CHECK(per_slot_mib <= 225.0);
    CHECK(o.per_slot_bytes <= kArenaPerSlotByteCeiling);
    // The four sizeable Sec 3.6 rows, each against the plan's own figure.
    CHECK(o.slotRegionBytes(SlotRegion::RefMicx) + o.slotRegionBytes(SlotRegion::RefMicxSsm)
              >= 60ull * 1024 * 1024);
    CHECK(o.slotRegionBytes(SlotRegion::BosMicx) >= 18ull * 1024 * 1024);
    CHECK(o.slotRegionBytes(SlotRegion::XeAaHistory) ==
          static_cast<std::size_t>(kDevXeAndersonTriples) * 3 * 5000 * sizeof(double));
    CHECK(o.slotRegionBytes(SlotRegion::Iden) == 39ull * 8451 * sizeof(double));
    // Sec 6.18: FOUR microscopic BOS slots, not eleven.
    CHECK(o.slotRegionBytes(SlotRegion::BosMicx) * 11 / 4 ==
          o.slotRegionBytes(SlotRegion::Micx));
    // The four control structs are NOT slot regions any more.
    CHECK(o.control_block_bytes > 0);
    // The burn key is an int array, not a double one.
    CHECK(o.slotRegionBytes(SlotRegion::BurnKey) == 8451ull * sizeof(int));

    if (slots == 64) {
        const double total_gib = static_cast<double>(o.total_bytes) / (kMiB * 1024.0);
        CHECK(total_gib >= 12.8);
        CHECK(total_gib <= 14.4);
    }

    // Admission: refuse when the device is too small, grant when it is not.
    const ArenaAdmission tight = arenaAdmit(o, o.total_bytes, o.total_bytes);
    CHECK(!tight.granted); // 10% + 10% cannot fit in exactly the layout size
    const std::size_t roomy = static_cast<std::size_t>(static_cast<double>(o.total_bytes) * 4.0);
    const ArenaAdmission ok = arenaAdmit(o, roomy, roomy);
    CHECK(ok.granted);
    CHECK(ok.driver_reserve_bytes == static_cast<std::size_t>(static_cast<double>(roomy) * 0.10));
    CHECK(ok.requested_bytes ==
          o.total_bytes + static_cast<std::size_t>(static_cast<double>(o.total_bytes) * 0.10));
    CHECK(!ok.per_slot_over_ceiling);

    if (print) {
        std::printf("slots=%d  geometry=%.3f MiB  library=%.3f MiB  control=%zu B  "
                    "per_slot=%.3f MiB (scratch %.3f MiB)  total=%.3f GiB\n",
                    slots, static_cast<double>(o.shared_geometry_bytes) / kMiB,
                    static_cast<double>(o.shared_library_bytes) / kMiB, o.control_block_bytes,
                    per_slot_mib, static_cast<double>(o.per_slot_scratch_bytes) / kMiB,
                    static_cast<double>(o.total_bytes) / (kMiB * 1024.0));
        if (slots == 64) {
            for (int i = 0; i < kSlotRegionCount; ++i) {
                const double mib = static_cast<double>(o.slot[i].bytes) / kMiB;
                if (mib >= 0.5)
                    std::printf("    slot region %2d  %10.3f MiB  @%zu\n", i, mib,
                                o.slot[i].offset);
            }
            for (int b = 0; b < kScratchBandCount; ++b)
                std::printf("    scratch %-16s %10.3f MiB  @%zu\n",
                            arenaScratchBandName(static_cast<ScratchBand>(b)),
                            static_cast<double>(o.scratch[b].bytes) / kMiB, o.scratch[b].offset);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;

    for (const int slots : {1, 8, 64}) checkOneLayout(slots, print);

    // A layout with no slots is empty, not a crash: reserve() refuses it.
    const ArenaOffsets empty = arenaComputeLayout(arenaDims(8451, 26692, 313, 5000, 0));
    CHECK(empty.slot_count == 0);
    CHECK(empty.total_bytes == empty.slot_base);
    CHECK(empty.valid);
    CHECK(!empty.slots_exceed_cap);
    for (int i = 0; i < kControlRegionCount; ++i) CHECK(empty.control[i].bytes == 0);

    // More slots than the scheduler can classify is a REFUSAL, not a truncation.
    const ArenaOffsets too_wide =
        arenaComputeLayout(arenaDims(8451, 26692, 313, 5000, kMaxDeviceSlots + 1));
    CHECK(!too_wide.valid);
    CHECK(too_wide.slots_exceed_cap);
    CHECK(too_wide.slot_count == 0);
    CHECK(!arenaAdmit(too_wide, ~0ull / 2, ~0ull / 2).granted);

    // The scratch table has no orphan: every ScratchId belongs to a real band
    // and every band has at least one user.
    for (int b = 0; b < kScratchBandCount; ++b) {
        int users = 0;
        for (int u = 0; u < kScratchIdCount; ++u)
            if (static_cast<int>(kScratchSpecs[u].band) == b) ++users;
        CHECK(users > 0);
    }
    for (int u = 0; u < kScratchIdCount; ++u) {
        CHECK(static_cast<int>(kScratchSpecs[u].id) == u); // table order == enum order
        CHECK(kScratchSpecs[u].name != nullptr);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "gpu arena layout: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("gpu arena layout: PASS\n");
    return 0;
}
