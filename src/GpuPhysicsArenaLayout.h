#pragma once

// Fixed-address arena layout calculator -- Rev.7.1 plan Task 2, Sec 3.3-3.6 / 4.
//
// WHAT THIS IS.  Pure arithmetic over problem dimensions: given (nxyz, nsurf,
// nxy, n_fuel, slots) it returns every byte offset the arena will ever hand
// out.  No CUDA, no allocation, no I/O -- so the whole layout is unit-testable
// on a machine with no GPU (test/gpu_physics_arena_layout.cpp), which is the
// only way the non-overlap and aliasing properties get checked at all.  The
// CUDA arm (GpuPhysicsArenaCuda.cu) does exactly one thing with this: add the
// offsets to one base pointer.
//
// WHY THE ADDRESSES ARE FIXED.  A captured CUDA graph bakes its kernel
// arguments.  If a slot's arrays could move -- because an allocator recycled a
// block, or a slot was resized -- every captured graph would have to be
// recaptured, which is the cost the graph exists to avoid.  So: ONE allocation,
// laid out before any graph work, uniform slot stride, and no allocation on any
// hot path ever (Sec 4.1).  A uniform stride is also what makes
// `slotView(slot)` a pure index rebase, exactly like nodalSlotView() in
// NodalKernel.h.
//
// SCRATCH ALIASING IS PER SLOT AND PER PHASE (Sec 4.2, Rev.7.1 correction).
// Rev.7's aliasing table assumed a GLOBAL phase lifetime -- "after the Outer
// phase ends, depletion reuses the nodal scratch".  Under an asynchronous
// case-phase scheduler that assumption is simply false: slot A is in Outer
// while slot B is in Depletion, so a global-lifetime alias hands B's depletion
// kernel the bytes A's nodal kernel is still reading.  The rule here is
// therefore:
//
//   * aliasing happens only INSIDE one slot, never between slots (Sec 5.9:
//     the slot stride is the isolation boundary), and
//   * every scratch request names the phase that owns it, and a debug build
//     traps when the requesting phase is not the slot's current phase.
//
// The alias groups are DATA (kScratchSpecs below), not comments, so the debug
// trap and the contract test read the same table.
//
// NO COMPRESSION TIER.  Rev.7 put a 22-35 MiB/slot compressed tier on the
// critical path.  Re-measured (Sec 3.6): the per-slot budget is ~200-225 MiB
// because `_ref_micx` (~65 MiB) and the Xe Anderson history cannot be removed,
// and 64 slots at that size is ~14 GiB -- 15% of a 96 GiB RTX PRO 6000.  Memory
// is not the risk; L2 is.  So every array here is full size and the compression
// tier is gone from the plan.

// GpuPhysicsTypes.h (which itself pulls in GpuSlotControl.h) for two things and
// nothing else: the problem-fixed shape constants (kDevNg / kDevNiso / kDevNxs /
// kDevNdirMax / kDevLr / kDevNews / kDevNewsbt, and the four-slot BOS table) and
// the sizes of the four control structs.  No CUDA arrives with it.
#include "GpuPhysicsTypes.h"

#include <cstddef>
#include <cstdint>

namespace rasbery::gpu {

/// Every region starts on a 256-byte boundary.  256 is the CUDA global-memory
/// transaction/alignment granularity that makes a coalesced load of the first
/// element of a region a single sector, and it is what cudaMalloc itself
/// guarantees -- so a region carved out of one big block is as well aligned as
/// one that had its own allocation.
inline constexpr std::size_t kArenaAlignment = 256;

/// Sec 4.4 VRAM admission.  Two reserves, both 10%:
///   - the DRIVER reserve, held back from total VRAM for the context, the
///     module images and whatever else the driver needs to stay alive;
///   - the FRAGMENTATION reserve, added on top of the request, because a pool
///     that is asked for exactly what is free has no room to satisfy the next
///     stream-ordered suballocation.
/// Admission failure is a hard failure.  It does NOT silently reduce the slot
/// count: a run that quietly became 24 slots wide would answer the throughput
/// question with the wrong experiment.
inline constexpr double kArenaDriverReserveFraction        = 0.10;
inline constexpr double kArenaFragmentationReserveFraction = 0.10;

/// Sec 3.6 / 4.4 per-slot ceiling.
///
/// The MEASURED layout at APR1400 size is 224.06 MiB/slot, at the very top of
/// Sec 3.6's 200-225 MiB band -- 0.4% of headroom, which is not headroom.  Any
/// array W2 adds (Task 12a's resolved delta stream, Task 13's Xe staging) would
/// trip admission on arithmetic alone, before a single kernel ran.  The ceiling
/// is therefore 256 MiB: the measured footprint plus room for the W2 additions,
/// still 16 GiB at 64 slots and still ~17% of a 96 GiB RTX PRO 6000.
///
/// This is a budget, not a target.  If the layout ever approaches 256 MiB the
/// question to ask is which array stopped being necessary, not whether the
/// number can move again.
inline constexpr std::size_t kArenaPerSlotByteCeiling = 256ull * 1024ull * 1024ull;

inline constexpr std::size_t arenaAlignUp(std::size_t bytes) {
    return (bytes + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Everything the layout depends on.  The library counts default to 0, which
/// makes the immutable library block empty -- useful for a layout test that
/// only cares about geometry and slots, and honest about a run that has not
/// loaded a library yet.
struct ArenaDims {
    int nxyz   = 0;
    int nsurf  = 0;
    int nxy    = 0;
    int nz     = 0;
    int nxya   = 0; ///< radial assemblies; `vola` is [nxya * nz]
    int n_fuel = 0;
    int slots  = 0;

    int ng   = kDevNg;
    int niso = kDevNiso;

    // Immutable library element counts (Sec 3.4).
    int n_ref_points   = 0;
    int n_coeff_points = 0;
    int n_knots        = 0;
    int cram_poles     = 4; ///< milk.h:1676-1701 pole_count for CRAM order 8

    /// Non-diagonal entries per isotope row of the CRAM transition matrix.
    /// milk.h:1742-1758 stores the diagonal separately and compresses the rest;
    /// Sec 6.18 measures 3-6 per row.  Used only to size the depletion scratch.
    int cram_row_nnz = 4;
};

/// The required 5-argument spelling.  `nz` and `nxya` are derived rather than
/// demanded: the node stack is rectangular (nz = nxyz / nxy -- 8451/313 = 27 at
/// APR1400 size), and there can never be more assemblies in a plane than radial
/// nodes, so nxy is a safe upper bound for nxya.  Callers that know the real
/// values set them on the returned struct.
inline constexpr ArenaDims arenaDims(int nxyz, int nsurf, int nxy, int n_fuel, int slots) {
    ArenaDims d;
    d.nxyz   = nxyz;
    d.nsurf  = nsurf;
    d.nxy    = nxy;
    d.nz     = (nxy > 0) ? nxyz / nxy : 0;
    d.nxya   = nxy;
    d.n_fuel = n_fuel;
    d.slots  = slots;
    return d;
}

// ---------------------------------------------------------------------------
// Region catalogues
// ---------------------------------------------------------------------------

/// Immutable shared geometry (Sec 3.3): ONE copy for the whole cohort, placed
/// below the slot base so no slot stride can reach it.
///
/// There is deliberately NO `Comps` region, for the same reason LibraryRegion
/// has no `knot_offsets`.  `Geometry::_comps` was allocated (`new int[_nxyz]`,
/// so uninitialised) and then never written by anything, and never read either:
/// the composition index the solver actually uses is `XSSet::_comp`
/// (XSSet.h:587), a different array of a different type.  A region for it
/// reserved nxyz*4 bytes of VRAM and, worse, made `DeviceGeometryView::comps`
/// a NON-NULL pointer into arena bytes no import had ever touched -- a field
/// that reads as available and dereferences to whatever the arena last held.
/// If a kernel ever needs the composition map, the region to add is XSSet's,
/// with an importGeometryAsync that actually fills it.
enum class GeometryRegion : int {
    Hmesh = 0,
    Hz,
    Vol,
    Vola,
    Albedo,
    Neib,
    Neibr,
    Neibrb,
    Lklr,
    Idirlr,
    Sgnlr,
    Lktosfc,
    Ltola,
    Ltolc,
    FuelNodes,
    IsFuel,
    Count
};

/// Immutable shared XS library (Sec 3.4): also one copy, re-uploaded only when
/// the library file itself changes.
///
/// Every region here has a host array behind it -- LibFlux and LibChix are
/// XSSet.cpp:705-706 `_lib_flux`/`_lib_chix`, both [dpt*ng].  There is
/// deliberately no `knot_offsets` region: the knot offsets are an `int` field
/// inside the per-branch descriptor structs (XSSet.cpp:419, 451), not a
/// standalone array, so a device pointer for them would have nothing to point
/// at.  A view field with no backing is worse than a missing one -- it reads as
/// available and dereferences to whatever the arena last wrote.
enum class LibraryRegion : int {
    LibMicxScalar = 0,
    LibMicxScatter,
    LibLmpxScalar,
    LibLmpxScatter,
    CoeffMicxScalar,
    CoeffMicxScatter,
    CoeffLmpxScalar,
    CoeffLmpxScatter,
    LibIden,
    LibBurn,
    LibWvfr,
    LibFlux,
    LibChix,
    Knots,
    DepDecay,
    DepTrans,
    CramAlpha,
    CramTheta,
    Count
};

/// Sec 3.2 control packet, hoisted OUT of the per-slot stride.
///
/// THE BUG THIS FIXES.  These four were laid out inside the slot block, so
/// slot s's DeviceSlotPhase sat at `slot_base + s*224MiB`.  Every kernel indexes
/// them densely -- `phases[tid]`, `states[s]` -- which is not merely a different
/// spelling of the same address, it is a different address.  It also destroyed
/// the entire reason DeviceSlotPhase is 32 bytes: "64 slots = 2 KiB resident in
/// L1" is only true if the 64 structs are CONTIGUOUS.  Strided by 224 MiB they
/// are 64 separate cache lines in 64 separate pages, which is the traffic
/// pattern the split was built to remove.
///
/// So the control block is four dense arrays, sized `slots` each, placed with
/// geometry and library BELOW slot_base.  Two consequences worth stating:
/// `slotView` is a dense index, and `clearSlotAsync` -- which memsets the whole
/// slot stride -- can no longer reach the control structs, so a bulk clear
/// cannot wipe the defaults a refill just wrote.
enum class ControlRegion : int {
    SlotPhase = 0,
    SlotState,
    SearchState,
    ScheduleParams,
    Count
};

/// Per-slot mutable state (Sec 3.5).  Every one of these repeats once per slot
/// at a uniform stride.
enum class SlotRegion : int {
    // NOTE: the four Sec 3.2 control structs are NOT here -- they live in the
    // contiguous control block below slot_base (ControlRegion above).
    // flux / current
    Phif = 0,
    Phis,
    Jnet,
    Psi,
    Phic,
    // CMFD operator + Krylov
    Dtil,
    Dhat,
    Diag,
    Cc,
    Src,
    CmfdPsi,
    BicgVec,
    BicgDinv,
    // nodal
    Trlcff,
    NodalConst,
    ConstantXs,
    Dsncff,
    MuTau,
    MatM,
    // isotopics and cross sections
    Iden,
    Xs,
    XsSsm,
    Lmpx,
    LmpxSsm,
    Micx,
    MicxSsm,
    RefMicx,
    RefMicxSsm,
    RefLmpx,
    RefLmpxSsm,
    RefIden,
    // beginning-of-step snapshot (Sec 6.18: FOUR microscopic slots, not eleven)
    BosMicx,
    BosXskf,
    BosIden,
    BosFlux,
    BosBurnKey,
    // Xe Anderson history (Sec 3.5, Driver.h:999-1020)
    XeAaHistory,
    // thermal-hydraulics / rod
    Bppm,
    Tful,
    Tmod,
    Dmod,
    RodFraction,
    NodeWvfr,
    // pin power reconstruction (Sec 6.22)
    PprP,
    PprA,
    PprC,
    PprQ,
    PprL,
    PprBt,
    // integer node keys
    BurnKey,
    CtypKey,
    // output packing
    OutPack,
    Count
};

/// Aliasable scratch bands.  One band is one byte range; the ScratchIds that
/// map to it share those bytes (Sec 4.2 "허용 alias 쌍").
enum class ScratchBand : int {
    KrylovOuter = 0,    ///< BiCG ax/s <-> Wielandt terms <-> outer partials
    NodalDepletion,     ///< nodal trl/matrix scratch <-> depletion temporary
    ThOutputPpr,        ///< TH channel <-> output pack <-> PPR corner
    Cram,               ///< predictor CRAM workspace <-> corrector CRAM workspace
    Count
};

/// Scratch USERS.  A request names one of these plus the phase that owns it.
enum class ScratchId : int {
    BicgAx = 0,
    BicgS,
    WielandtTerms,
    OuterPartials,
    NodalTrlScratch,
    NodalMatrixScratch,
    DepletionTemp,
    ThChannel,
    OutputPackScratch,
    PprCorner,
    CramPredictor,
    CramCorrector,
    Count
};

inline constexpr int kGeometryRegionCount = static_cast<int>(GeometryRegion::Count);
inline constexpr int kLibraryRegionCount  = static_cast<int>(LibraryRegion::Count);
inline constexpr int kControlRegionCount  = static_cast<int>(ControlRegion::Count);
inline constexpr int kSlotRegionCount     = static_cast<int>(SlotRegion::Count);
inline constexpr int kScratchBandCount    = static_cast<int>(ScratchBand::Count);
inline constexpr int kScratchIdCount      = static_cast<int>(ScratchId::Count);

static_assert(kDevicePhaseCount <= 32, "the scratch owner-phase mask is a uint32");

/// One bit per DevicePhase.
inline constexpr std::uint32_t phaseBit(DevicePhase p) {
    return 1u << static_cast<std::uint32_t>(p);
}

/// The Sec 4.2 alias table AS DATA.  `owner_phases` is the whole contract: a
/// scratch pointer may only be taken while the slot is in one of these phases,
/// which is what makes it safe for two users of the same band to exist at all.
///
/// Read the four rows against Sec 4.2:
///   BiCG ax/s <-> Wielandt <-> outer partials              (Outer)
///   nodal trl/matrix scratch <-> depletion temporary       (Outer / Depletion*)
///   TH channel <-> output pack <-> PPR corner              (TH / OutputPack / Ppr)
///   predictor CRAM workspace <-> corrector CRAM workspace  (Depletion*)
struct ScratchSpec {
    ScratchId     id;
    ScratchBand   band;
    const char*   name;
    std::uint32_t owner_phases;
};

inline constexpr ScratchSpec kScratchSpecs[kScratchIdCount] = {
    {ScratchId::BicgAx, ScratchBand::KrylovOuter, "bicg_ax", phaseBit(DevicePhase::Outer)},
    {ScratchId::BicgS, ScratchBand::KrylovOuter, "bicg_s", phaseBit(DevicePhase::Outer)},
    {ScratchId::WielandtTerms, ScratchBand::KrylovOuter, "wielandt_terms",
     phaseBit(DevicePhase::Outer)},
    {ScratchId::OuterPartials, ScratchBand::KrylovOuter, "outer_partials",
     phaseBit(DevicePhase::Outer)},

    {ScratchId::NodalTrlScratch, ScratchBand::NodalDepletion, "nodal_trl_scratch",
     phaseBit(DevicePhase::Outer)},
    {ScratchId::NodalMatrixScratch, ScratchBand::NodalDepletion, "nodal_matrix_scratch",
     phaseBit(DevicePhase::Outer)},
    {ScratchId::DepletionTemp, ScratchBand::NodalDepletion, "depletion_temp",
     phaseBit(DevicePhase::DepletionPredictor) | phaseBit(DevicePhase::DepletionCorrector)},

    {ScratchId::ThChannel, ScratchBand::ThOutputPpr, "th_channel",
     phaseBit(DevicePhase::ThermalHydraulics)},
    {ScratchId::OutputPackScratch, ScratchBand::ThOutputPpr, "output_pack_scratch",
     phaseBit(DevicePhase::OutputPack)},
    {ScratchId::PprCorner, ScratchBand::ThOutputPpr, "ppr_corner", phaseBit(DevicePhase::Ppr)},

    {ScratchId::CramPredictor, ScratchBand::Cram, "cram_predictor",
     phaseBit(DevicePhase::DepletionPredictor)},
    {ScratchId::CramCorrector, ScratchBand::Cram, "cram_corrector",
     phaseBit(DevicePhase::DepletionCorrector)},
};

/// The debug trap's first half: is this scratch id reachable from this phase at
/// all?  (The second half -- is this the slot's CURRENT phase -- needs the slot,
/// so it lives in GpuPhysicsArena.)
inline constexpr bool arenaScratchPhaseAllowed(ScratchId id, DevicePhase phase) {
    return (kScratchSpecs[static_cast<int>(id)].owner_phases & phaseBit(phase)) != 0u;
}

inline constexpr ScratchBand arenaScratchBand(ScratchId id) {
    return kScratchSpecs[static_cast<int>(id)].band;
}

/// Do two scratch users ever have the same owner phase?  If so they can be live
/// at the same moment in the same slot and MUST NOT share bytes.
inline constexpr bool arenaScratchCoResident(ScratchId a, ScratchId b) {
    return (kScratchSpecs[static_cast<int>(a)].owner_phases &
            kScratchSpecs[static_cast<int>(b)].owner_phases) != 0u;
}

// ---------------------------------------------------------------------------
// Element counts.  One switch per catalogue so the tables and the tests read
// the same arithmetic; every count is annotated with the host array it mirrors.
// ---------------------------------------------------------------------------

inline constexpr std::size_t arenaGeometryElements(GeometryRegion r, const ArenaDims& d) {
    const std::size_t nxyz  = static_cast<std::size_t>(d.nxyz);
    const std::size_t nsurf = static_cast<std::size_t>(d.nsurf);
    const std::size_t nxy   = static_cast<std::size_t>(d.nxy);
    switch (r) {
        case GeometryRegion::Hmesh:     return nxyz * kDevNdirMax;      // Geometry.h:121
        case GeometryRegion::Hz:        return static_cast<std::size_t>(d.nz);
        case GeometryRegion::Vol:       return nxyz;
        case GeometryRegion::Vola:      return static_cast<std::size_t>(d.nxya) * d.nz;
        case GeometryRegion::Albedo:    return kDevNdirMax * kDevLr;
        case GeometryRegion::Neib:      return nxyz * kDevNewsbt;       // Geometry.h:134
        case GeometryRegion::Neibr:     return nxy * kDevNews;
        case GeometryRegion::Neibrb:    return nxy * kDevNews;
        case GeometryRegion::Lklr:      return nsurf * kDevLr;          // Geometry.h:138
        case GeometryRegion::Idirlr:    return nsurf * kDevLr;
        case GeometryRegion::Sgnlr:     return nsurf * kDevLr;
        case GeometryRegion::Lktosfc:   return nxyz * kDevNdirMax * kDevLr;
        case GeometryRegion::Ltola:     return nxy;
        case GeometryRegion::Ltolc:     return nxy * kDevNews;
        case GeometryRegion::FuelNodes: return static_cast<std::size_t>(d.n_fuel);
        case GeometryRegion::IsFuel:    return nxyz;
        case GeometryRegion::Count:     break;
    }
    return 0;
}

/// 8 for a double, 4 for an int.  Kept as a function rather than a column in a
/// table so a new region cannot be added without deciding its element type.
inline constexpr std::size_t arenaGeometryElementBytes(GeometryRegion r) {
    switch (r) {
        case GeometryRegion::Hmesh:
        case GeometryRegion::Hz:
        case GeometryRegion::Vol:
        case GeometryRegion::Vola:
        case GeometryRegion::Albedo: return sizeof(double);
        default:                     return sizeof(int);
    }
}

inline constexpr std::size_t arenaLibraryElements(LibraryRegion r, const ArenaDims& d) {
    const std::size_t ref   = static_cast<std::size_t>(d.n_ref_points);
    const std::size_t coeff = static_cast<std::size_t>(d.n_coeff_points);
    const std::size_t niso  = static_cast<std::size_t>(d.niso);
    const std::size_t ng    = static_cast<std::size_t>(d.ng);
    switch (r) {
        // XSSet.cpp:700  _lib_micx.allocate(dpt*niso*ng, dpt*niso*ng*ng)
        case LibraryRegion::LibMicxScalar:    return kDevNxs * ref * niso * ng;
        case LibraryRegion::LibMicxScatter:   return ref * niso * ng * ng;
        case LibraryRegion::LibLmpxScalar:    return kDevNxs * ref * ng;
        case LibraryRegion::LibLmpxScatter:   return ref * ng * ng;
        // XSSet.cpp:708  _lib_coeff_micx.allocate(coeff*niso*ng, coeff*niso*ng*ng)
        case LibraryRegion::CoeffMicxScalar:  return kDevNxs * coeff * niso * ng;
        case LibraryRegion::CoeffMicxScatter: return coeff * niso * ng * ng;
        case LibraryRegion::CoeffLmpxScalar:  return kDevNxs * coeff * ng;
        case LibraryRegion::CoeffLmpxScatter: return coeff * ng * ng;
        case LibraryRegion::LibIden:          return ref * niso;
        case LibraryRegion::LibBurn:          return ref;
        case LibraryRegion::LibWvfr:          return ref;
        // XSSet.cpp:705-706  _lib_flux / _lib_chix, both [dpt*ng].
        case LibraryRegion::LibFlux:          return ref * ng;
        case LibraryRegion::LibChix:          return ref * ng;
        case LibraryRegion::Knots:            return static_cast<std::size_t>(d.n_knots);
        case LibraryRegion::DepDecay:         return niso * niso;
        case LibraryRegion::DepTrans:         return niso * niso;
        case LibraryRegion::CramAlpha:        return static_cast<std::size_t>(d.cram_poles) * 2;
        case LibraryRegion::CramTheta:        return static_cast<std::size_t>(d.cram_poles) * 2;
        case LibraryRegion::Count:            break;
    }
    return 0;
}

inline constexpr std::size_t arenaLibraryElementBytes(LibraryRegion) { return sizeof(double); }

inline constexpr std::size_t arenaSlotElements(SlotRegion r, const ArenaDims& d) {
    const std::size_t nxyz   = static_cast<std::size_t>(d.nxyz);
    const std::size_t nsurf  = static_cast<std::size_t>(d.nsurf);
    const std::size_t nfuel  = static_cast<std::size_t>(d.n_fuel);
    const std::size_t ng     = static_cast<std::size_t>(d.ng);
    const std::size_t ng2    = ng * ng;
    const std::size_t niso   = static_cast<std::size_t>(d.niso);
    switch (r) {
        case SlotRegion::Phif: return ng * nxyz;                                // Geometry.h:390
        case SlotRegion::Phis: return kDevLr * ng * kDevNdirMax * nxyz;         // Geometry.h:392
        case SlotRegion::Jnet: return kDevLr * ng * kDevNdirMax * nxyz;         // Geometry.h:391
        case SlotRegion::Psi:  return nxyz;
        case SlotRegion::Phic: return nxyz * ng * kDevNews;                     // Geometry.cpp:400

        case SlotRegion::Dtil:     return nsurf * ng;                           // CMFD.cpp:19
        case SlotRegion::Dhat:     return nsurf * ng;
        case SlotRegion::Diag:     return nxyz * ng2;
        case SlotRegion::Cc:       return nxyz * ng * kDevNewsbt;
        case SlotRegion::Src:      return nxyz * ng;
        case SlotRegion::CmfdPsi:  return nxyz;
        // BICGSolver.cpp:32-43  vz vy vr vr0 vp vv vs vt ssor_tmp, each ng*nxyz
        case SlotRegion::BicgVec:  return 9 * ng * nxyz;
        case SlotRegion::BicgDinv: return ng2 * nxyz;

        // Nodal.cpp:48-79
        case SlotRegion::Trlcff:     return 3 * nxyz * kDevNdirMax * ng;
        case SlotRegion::NodalConst: return 9 * nxyz * kDevNdirMax * ng;
        case SlotRegion::ConstantXs: return 2 * nxyz * ng;
        case SlotRegion::Dsncff:     return 3 * nxyz * kDevNdirMax * ng;
        case SlotRegion::MuTau:      return 2 * nxyz * kDevNdirMax * ng2;
        case SlotRegion::MatM:       return 4 * nxyz * ng2;

        case SlotRegion::Iden:       return niso * nxyz;                        // XSSet.cpp:612
        case SlotRegion::Xs:         return kDevNxs * ng * nxyz;
        case SlotRegion::XsSsm:      return ng2 * nxyz;
        case SlotRegion::Lmpx:       return kDevNxs * ng * nxyz;
        case SlotRegion::LmpxSsm:    return ng2 * nxyz;
        // XSSet.cpp:606-607  micn = niso*ng*nxyz, mism = niso*ng*ng*nxyz
        case SlotRegion::Micx:       return kDevNxs * niso * ng * nxyz;
        case SlotRegion::MicxSsm:    return niso * ng2 * nxyz;
        case SlotRegion::RefMicx:    return kDevNxs * niso * ng * nxyz;         // XSSet.h:211
        case SlotRegion::RefMicxSsm: return niso * ng2 * nxyz;
        case SlotRegion::RefLmpx:    return kDevNxs * ng * nxyz;                // XSSet.h:210
        case SlotRegion::RefLmpxSsm: return ng2 * nxyz;
        case SlotRegion::RefIden:    return niso * nxyz;                        // XSSet.h:212

        // Sec 6.18: FOUR microscopic slots, not eleven.  BuildTransitionMatrix
        // (XSSet.cpp:3380-3407) reads only XSAF/XSFF/XS2N/XS3N of the condensed
        // block; the macroscopic XSKF row is read separately (XSSet.cpp:4189).
        case SlotRegion::BosMicx:    return kBosMicroXtCount * niso * ng * nxyz;
        case SlotRegion::BosXskf:    return ng * nxyz;
        case SlotRegion::BosIden:    return niso * nxyz;
        case SlotRegion::BosFlux:    return ng * nxyz;
        case SlotRegion::BosBurnKey: return nxyz;

        // (6 + 2*XE_ANDERSON_DEPTH) triples of (I, Xe, Xem) over the fuel nodes.
        case SlotRegion::XeAaHistory: return static_cast<std::size_t>(kDevXeAndersonTriples) * 3 * nfuel;

        case SlotRegion::Bppm:        return nxyz;                              // Geometry.cpp:395-399
        case SlotRegion::Tful:        return nxyz;
        case SlotRegion::Tmod:        return nxyz;
        case SlotRegion::Dmod:        return nxyz;
        case SlotRegion::RodFraction: return nxyz;
        case SlotRegion::NodeWvfr:    return 2 * nxyz;                          // XSSet.h:213-214

        case SlotRegion::PprP:  return nxyz * ng * 15;                          // Geometry.cpp:401-406
        case SlotRegion::PprA:  return nxyz * ng * 8;
        case SlotRegion::PprC:  return nxyz * ng * 15;
        case SlotRegion::PprQ:  return nxyz * ng * 15;
        case SlotRegion::PprL:  return nxyz * ng * 9;
        case SlotRegion::PprBt: return nxyz * ng;

        case SlotRegion::BurnKey: return nxyz;                                  // XSSet.h:161
        case SlotRegion::CtypKey: return nxyz;                                  // XSSet.h:162

        case SlotRegion::OutPack: return nxyz * ng * 2 + nxyz;
        case SlotRegion::Count:   break;
    }
    return 0;
}

inline constexpr std::size_t arenaSlotElementBytes(SlotRegion r) {
    switch (r) {
        case SlotRegion::BosBurnKey:
        case SlotRegion::BurnKey:
        case SlotRegion::CtypKey: return sizeof(int);
        default:                  return sizeof(double);
    }
}

inline constexpr std::size_t arenaControlElementBytes(ControlRegion r) {
    switch (r) {
        case ControlRegion::SlotPhase:      return sizeof(DeviceSlotPhase);
        case ControlRegion::SlotState:      return sizeof(DeviceSlotState);
        case ControlRegion::SearchState:    return sizeof(DeviceSearchState);
        case ControlRegion::ScheduleParams: return sizeof(DeviceScheduleParams);
        case ControlRegion::Count:          break;
    }
    return 0;
}

inline constexpr const char* arenaControlRegionName(ControlRegion r) {
    switch (r) {
        case ControlRegion::SlotPhase:      return "slot_phase";
        case ControlRegion::SlotState:      return "slot_state";
        case ControlRegion::SearchState:    return "search_state";
        case ControlRegion::ScheduleParams: return "schedule_params";
        case ControlRegion::Count:          break;
    }
    return "?";
}

/// Byte size of one scratch user's own range.
///
/// A band is NOT simply "the largest of its users" -- that was wrong, and the
/// old layout gate pinned the bug as expected behaviour.  bicg_ax and bicg_s
/// are both owned by Outer and are live in the SAME BiCGSTAB iteration; so are
/// the nodal trl and matrix scratches.  Giving them one shared range meant the
/// solver would have been reading its own overwritten intermediates.
///
/// The rule is therefore: users that share an owner phase get DISJOINT
/// sub-ranges within the band; only users whose owner-phase sets are disjoint
/// may overlap.  See arenaScratchSubOffset below.
inline constexpr std::size_t arenaScratchUserElements(ScratchId id, const ArenaDims& d) {
    const std::size_t nxyz = static_cast<std::size_t>(d.nxyz);
    const std::size_t ng   = static_cast<std::size_t>(d.ng);
    const std::size_t niso = static_cast<std::size_t>(d.niso);
    switch (id) {
        case ScratchId::BicgAx:        return ng * nxyz;
        case ScratchId::BicgS:         return ng * nxyz;
        case ScratchId::WielandtTerms: return ng * nxyz;
        case ScratchId::OuterPartials: return nxyz;

        case ScratchId::NodalTrlScratch:    return nxyz * kDevNdirMax * ng * ng;
        case ScratchId::NodalMatrixScratch: return nxyz * ng * ng;
        // milk.h:1742-1758 compresses the off-diagonal entries per row; Sec 6.18
        // measures 3-6 of them.  One row per isotope, one matrix per node.
        case ScratchId::DepletionTemp:      return niso * static_cast<std::size_t>(d.cram_row_nnz) * nxyz;

        case ScratchId::ThChannel:         return static_cast<std::size_t>(d.nxy) * d.nz * 4;
        case ScratchId::OutputPackScratch: return nxyz * ng;
        case ScratchId::PprCorner:         return nxyz * ng * kDevNews;

        // Sec 6.18 per-node global SoA: base_diag/pole_diag/rhs/x/accum, complex
        // double, 39 x 5 -> ~3.1 KiB/node.  Shared memory cannot hold it
        // (128 threads x 3.1 KiB = 397 KiB >> 128 KB/SM on sm_120).
        case ScratchId::CramPredictor: return niso * 5 * 2 * nxyz;
        case ScratchId::CramCorrector: return niso * 5 * 2 * nxyz;
        case ScratchId::Count:         break;
    }
    return 0;
}

inline constexpr std::size_t arenaScratchUserBytes(ScratchId id, const ArenaDims& d) {
    return arenaAlignUp(arenaScratchUserElements(id, d) * sizeof(double));
}

/// Offset of a user's range WITHIN its band.
///
/// Greedy interval assignment over the owner-phase sets, in enum order: a user
/// starts after everything already placed that shares one of its phases, and at
/// zero when nothing does.  Two users overlap exactly when their phase sets are
/// disjoint, which is the aliasing rule stated as an algorithm.
///
/// At the current table this gives, per band:
///   krylov_outer     ax | s | wielandt | partials      all Outer -> all disjoint
///   nodal_depletion  trl | matrix   (Outer, disjoint)
///                    depletion_temp (Depletion*, overlaps both -- never co-live)
///   th_output_ppr    th | outpack | ppr    three phases -> all at offset 0
///   cram             predictor | corrector two phases -> both at offset 0
inline constexpr std::size_t arenaScratchSubOffset(ScratchId id, const ArenaDims& d) {
    const ScratchBand band  = arenaScratchBand(id);
    const int         target = static_cast<int>(id);

    std::size_t phase_end[kDevicePhaseCount] = {};
    for (int u = 0; u < kScratchIdCount; ++u) {
        const ScratchSpec& spec = kScratchSpecs[u];
        if (spec.band != band) continue;

        std::size_t start = 0;
        for (int p = 0; p < kDevicePhaseCount; ++p)
            if ((spec.owner_phases & (1u << static_cast<std::uint32_t>(p))) != 0u &&
                phase_end[p] > start)
                start = phase_end[p];

        if (u == target) return start;

        const std::size_t end = start + arenaScratchUserBytes(spec.id, d);
        for (int p = 0; p < kDevicePhaseCount; ++p)
            if ((spec.owner_phases & (1u << static_cast<std::uint32_t>(p))) != 0u)
                phase_end[p] = end;
    }
    return 0;
}

/// Band size: the widest any single phase's co-resident set gets.  Equivalently
/// max over phases of the summed sizes of that phase's users in the band, which
/// the layout gate asserts against this independently.
inline constexpr std::size_t arenaScratchBandBytes(ScratchBand band, const ArenaDims& d) {
    std::size_t widest = 0;
    for (int u = 0; u < kScratchIdCount; ++u) {
        if (kScratchSpecs[u].band != band) continue;
        const ScratchId   id  = kScratchSpecs[u].id;
        const std::size_t end = arenaScratchSubOffset(id, d) + arenaScratchUserBytes(id, d);
        if (end > widest) widest = end;
    }
    return widest;
}

// ---------------------------------------------------------------------------
// The computed layout
// ---------------------------------------------------------------------------

struct ArenaRegion {
    std::size_t offset = 0; ///< bytes from the arena base (slot-relative for slot regions)
    std::size_t bytes  = 0;
};

/// Every offset the arena will ever hand out.  Trivially copyable and free of
/// any owning type, so it can be computed on the host, checked in a unit test,
/// and (if it ever needs to be) uploaded.
struct ArenaOffsets {
    ArenaDims dims{};

    ArenaRegion geometry[kGeometryRegionCount]{};
    ArenaRegion library[kLibraryRegionCount]{};
    /// Dense arrays of `slots` entries each, OUTSIDE the slot stride.
    ArenaRegion control[kControlRegionCount]{};
    /// Slot-RELATIVE: absolute offset is slot_base + slot*slot_stride + offset.
    ArenaRegion slot[kSlotRegionCount]{};
    ArenaRegion scratch[kScratchBandCount]{};

    std::size_t geometry_base = 0;
    std::size_t library_base  = 0;
    std::size_t control_base  = 0;
    std::size_t slot_base     = 0;

    std::size_t shared_geometry_bytes = 0;
    std::size_t shared_library_bytes  = 0;
    std::size_t control_block_bytes   = 0;
    std::size_t per_slot_bytes        = 0; ///< the uniform slot stride
    std::size_t per_slot_scratch_bytes = 0;
    std::size_t slot_count            = 0;
    std::size_t total_bytes           = 0;

    /// False when the request was rejected outright (see arenaComputeLayout).
    /// A caller must not use any offset from an invalid layout.
    bool valid            = true;
    bool slots_exceed_cap = false;

    [[nodiscard]] constexpr std::size_t slotBase(int s) const {
        return slot_base + static_cast<std::size_t>(s) * per_slot_bytes;
    }
    [[nodiscard]] constexpr std::size_t slotRegionOffset(int s, SlotRegion r) const {
        return slotBase(s) + slot[static_cast<int>(r)].offset;
    }
    [[nodiscard]] constexpr std::size_t slotRegionBytes(SlotRegion r) const {
        return slot[static_cast<int>(r)].bytes;
    }
    /// Dense: element `s` of the control array, not a slot-strided address.
    [[nodiscard]] constexpr std::size_t controlOffset(int s, ControlRegion r) const {
        return control[static_cast<int>(r)].offset +
               static_cast<std::size_t>(s) * arenaControlElementBytes(r);
    }
    [[nodiscard]] constexpr std::size_t scratchOffset(int s, ScratchId id) const {
        return slotBase(s) + scratch[static_cast<int>(arenaScratchBand(id))].offset +
               arenaScratchSubOffset(id, dims);
    }
    [[nodiscard]] constexpr std::size_t scratchBytes(ScratchId id) const {
        return arenaScratchUserBytes(id, dims);
    }
    /// True when `bytes` at `offset` lies inside slot `s`'s stride.  Used by the
    /// gate to prove the control block is out of clearSlotAsync's reach.
    [[nodiscard]] constexpr bool insideSlotStride(int s, std::size_t offset,
                                                  std::size_t bytes) const {
        const std::size_t lo = slotBase(s);
        const std::size_t hi = lo + per_slot_bytes;
        return offset < hi && (offset + bytes) > lo;
    }
};

static_assert(std::is_trivially_copyable_v<ArenaOffsets>);

/// The whole calculator.  Immutable blocks first (below the slot base, so a
/// slot stride cannot reach them), then `slots` copies of an identical slot
/// block.  Every region is aligned up to 256 B, which is also what makes the
/// stride a multiple of 256 and every slot's copy of a region equally aligned.
inline constexpr ArenaOffsets arenaComputeLayout(const ArenaDims& d) {
    ArenaOffsets o{};
    o.dims       = d;
    o.slot_count = static_cast<std::size_t>(d.slots < 0 ? 0 : d.slots);

    // Fail loud rather than truncate.  The scheduler classifies at most
    // kMaxDeviceSlots slots in one CTA, so a wider arena would silently run
    // slots nothing ever schedules -- which looks like a throughput result and
    // is not one.
    if (d.slots > kMaxDeviceSlots) {
        o.valid            = false;
        o.slots_exceed_cap = true;
        o.slot_count       = 0;
        return o;
    }

    std::size_t cursor = 0;

    o.geometry_base = cursor;
    for (int i = 0; i < kGeometryRegionCount; ++i) {
        const GeometryRegion r     = static_cast<GeometryRegion>(i);
        const std::size_t    bytes = arenaGeometryElements(r, d) * arenaGeometryElementBytes(r);
        o.geometry[i].offset       = cursor;
        o.geometry[i].bytes        = bytes;
        cursor += arenaAlignUp(bytes);
    }
    o.shared_geometry_bytes = cursor - o.geometry_base;

    o.library_base = cursor;
    for (int i = 0; i < kLibraryRegionCount; ++i) {
        const LibraryRegion r     = static_cast<LibraryRegion>(i);
        const std::size_t   bytes = arenaLibraryElements(r, d) * arenaLibraryElementBytes(r);
        o.library[i].offset       = cursor;
        o.library[i].bytes        = bytes;
        cursor += arenaAlignUp(bytes);
    }
    o.shared_library_bytes = cursor - o.library_base;

    // The control block: four DENSE arrays of `slots` entries, below slot_base.
    // Contiguity is the point -- 64 DeviceSlotPhase in 2 KiB is what keeps
    // Level-1's classify pass in L1, and being outside the slot stride is what
    // keeps clearSlotAsync from wiping a refill's freshly written defaults.
    o.control_base = cursor;
    for (int i = 0; i < kControlRegionCount; ++i) {
        const ControlRegion r     = static_cast<ControlRegion>(i);
        const std::size_t   bytes = o.slot_count * arenaControlElementBytes(r);
        o.control[i].offset       = cursor;
        o.control[i].bytes        = bytes;
        cursor += arenaAlignUp(bytes);
    }
    o.control_block_bytes = cursor - o.control_base;

    o.slot_base = arenaAlignUp(cursor);

    // One slot, laid out at offset 0; every slot repeats it at slot_base +
    // slot*stride, which is what makes slotView() a pure index rebase.
    std::size_t rel = 0;
    for (int i = 0; i < kSlotRegionCount; ++i) {
        const SlotRegion  r     = static_cast<SlotRegion>(i);
        const std::size_t bytes = arenaSlotElements(r, d) * arenaSlotElementBytes(r);
        o.slot[i].offset        = rel;
        o.slot[i].bytes         = bytes;
        rel += arenaAlignUp(bytes);
    }

    const std::size_t scratch_begin = rel;
    for (int b = 0; b < kScratchBandCount; ++b) {
        const std::size_t bytes = arenaScratchBandBytes(static_cast<ScratchBand>(b), d);
        o.scratch[b].offset     = rel;
        o.scratch[b].bytes      = bytes;
        rel += arenaAlignUp(bytes);
    }
    o.per_slot_scratch_bytes = rel - scratch_begin;

    o.per_slot_bytes = rel;
    o.total_bytes    = o.slot_base + o.slot_count * o.per_slot_bytes;
    return o;
}

// ---------------------------------------------------------------------------
// Sec 4.4  VRAM admission
// ---------------------------------------------------------------------------

struct ArenaAdmission {
    bool        granted = false;
    std::size_t required_bytes             = 0; ///< the layout total
    std::size_t requested_bytes            = 0; ///< required + fragmentation reserve
    std::size_t driver_reserve_bytes       = 0;
    std::size_t fragmentation_reserve_bytes = 0;
    std::size_t usable_bytes               = 0; ///< free - driver reserve
    std::size_t per_slot_bytes             = 0;
    bool        per_slot_over_ceiling      = false;
};

/// Decide, from the layout and what the driver says is free.  Two independent
/// refusals, both hard:
///   1. per-slot footprint above the Sec 3.6 ceiling -- the layout grew past
///      what the plan budgeted, which is a design question, not a runtime one;
///   2. requested (= required + 10%) above free minus the 10% driver reserve.
/// Neither reduces the slot count.  A run that quietly became narrower would
/// answer the throughput question with a different experiment.
inline constexpr ArenaAdmission arenaAdmit(const ArenaOffsets& o, std::size_t free_bytes,
                                           std::size_t total_bytes) {
    ArenaAdmission a{};
    a.required_bytes  = o.total_bytes;
    a.per_slot_bytes  = o.per_slot_bytes;
    a.per_slot_over_ceiling = o.per_slot_bytes > kArenaPerSlotByteCeiling;

    a.driver_reserve_bytes =
        static_cast<std::size_t>(static_cast<double>(total_bytes) * kArenaDriverReserveFraction);
    a.fragmentation_reserve_bytes = static_cast<std::size_t>(
        static_cast<double>(o.total_bytes) * kArenaFragmentationReserveFraction);
    a.requested_bytes = o.total_bytes + a.fragmentation_reserve_bytes;
    a.usable_bytes    = (free_bytes > a.driver_reserve_bytes) ? free_bytes - a.driver_reserve_bytes : 0;

    a.granted = o.valid && !a.per_slot_over_ceiling && a.requested_bytes <= a.usable_bytes;
    return a;
}

// ---------------------------------------------------------------------------
// Names, for receipts and for test failure messages.
// ---------------------------------------------------------------------------

inline constexpr const char* arenaScratchName(ScratchId id) {
    return kScratchSpecs[static_cast<int>(id)].name;
}

inline constexpr const char* arenaScratchBandName(ScratchBand b) {
    switch (b) {
        case ScratchBand::KrylovOuter:    return "krylov_outer";
        case ScratchBand::NodalDepletion: return "nodal_depletion";
        case ScratchBand::ThOutputPpr:    return "th_output_ppr";
        case ScratchBand::Cram:           return "cram";
        case ScratchBand::Count:          break;
    }
    return "?";
}

} // namespace rasbery::gpu
