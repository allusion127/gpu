#pragma once

// Backend-neutral GPU physics types -- Rev.7.1 plan Task 1, Sec 3.3-3.5 / 1.5.
//
// WHAT THIS HEADER IS.  The vocabulary every GPU phase kernel, the arena and
// the scheduler agree on, with no CUDA in it.  Three device views (immutable
// geometry, immutable library, per-slot mutable bulk) plus the backend
// capability receipt.  The per-slot CONTROL packet is the four-way split in
// GpuSlotControl.h; Rev.7's single `DeviceSlotControl` is deleted, not renamed.
//
// WHY BACKEND-NEUTRAL WHEN THE ONLY HARDWARE IS NVIDIA.  Rev.7.1 Sec 1.5 is
// explicit that HIP and SYCL are a trailing track and contribute zero to the
// speed target: the hardware is one NVIDIA GPU (238, GPU0).  What is adopted
// NOW is the naming and the pure-body discipline (constraint 35), because those
// fix the UPPER BOUND on a later port at zero cost today.  The tier ladder and
// GpuCapabilityReceipt below are the Rev.7 Sec 1.5 shapes kept intact so
// Task 23 has something to fill in; the only two tiers this campaign will
// actually produce are the two CUDA ones.
//
// A DEVICE VIEW IS POINTERS AND SCALARS, NOTHING ELSE.  Every struct whose name
// starts with `Device` is memcpy'd to the device or passed by value into a
// kernel.  No std::vector, no std::string, no owning handle, no virtual: the
// arena owns the storage and a view only says where it is.  The contract test
// enforces this by grep, because the failure mode -- a std::vector member that
// compiles fine on the host and dereferences a host pointer on the device -- is
// silent until it is a wrong answer.
//
// SLOT VIEWS ARE REBASED, NOT REALLOCATED.  As in nodalSlotView (NodalKernel.h)
// a slot view is the slot-0 view with every per-slot pointer advanced by the
// slot's stride.  The arena fixes those addresses once, before any graph work,
// so a captured graph's baked kernel arguments stay valid for the run
// (GpuPhysicsArena.h).
//
// NO PERSISTENT / COOPERATIVE PATH.  W0 measured c_barrier = 0.78 us against
// the 0.384 us kill threshold, so constraint 17's persistent track is closed
// and nothing here may grow cooperative-groups or grid-barrier scaffolding.
// Dispatch is c_dispatch = 0.783 us per launch, which is the cost the SWITCH
// scheduler (Sec 5.5) is designed against.

#include "GpuSlotControl.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace rasbery::gpu {

// ---------------------------------------------------------------------------
// Sec 1.5  Backend identity and capability receipt
// ---------------------------------------------------------------------------

enum class GpuBackendKind : std::uint32_t {
    None = 0, ///< built without any device backend (stub)
    Cuda,
    Hip,  ///< Task 24, trailing track: interface only, never `available()`
    Sycl  ///< Task 25, trailing track: interface only, never `available()`
};

/// Rev.7 Sec 1.5 support ladder, kept whole so Task 23 can populate it.
/// Rev.7.1 narrows the SCOPE of what this campaign produces to the two CUDA
/// rungs: ConditionalGraph (device-side control flow, Sec 5.5) and
/// EpochScheduler (host-driven epoch loop, Sec 5.6).  The remaining rungs are
/// reserved labels, not promises.
enum class GpuSupportTier : std::uint32_t {
    Unsupported = 0, ///< "U": no device, or a device this build cannot drive
    G0,              ///< device present, no async arena
    G1,              ///< compute only, host-driven, no graphs
    G2,              ///< EpochScheduler: graphs, host-driven epoch loop
    G3,              ///< ConditionalGraph: device-side phase selection
    G4               ///< reserved (Sec 1.5 top rung)
};

/// "U" / "G0".."G4".  The receipt carries this as well as the numeric value,
/// because the ordinal and the label do NOT agree -- Unsupported is 0, so G2 is
/// ordinal 3 -- and a receipt reading `"tier":3` invites exactly the wrong
/// conclusion.
inline const char* gpuSupportTierName(GpuSupportTier tier) {
    switch (tier) {
        case GpuSupportTier::Unsupported: return "U";
        case GpuSupportTier::G0:          return "G0";
        case GpuSupportTier::G1:          return "G1";
        case GpuSupportTier::G2:          return "G2";
        case GpuSupportTier::G3:          return "G3";
        case GpuSupportTier::G4:          return "G4";
    }
    return "U";
}

/// What the process learned about the device it is actually running on.  Emitted
/// once into the final receipt (Sec 9.3) so a run can be told apart from a run
/// on different silicon without reading the log body.
///
/// Trivially copyable except for the two names, which are host-only strings:
/// this struct never crosses onto the device.
struct GpuCapabilityReceipt {
    GpuBackendKind kind = GpuBackendKind::None;
    GpuSupportTier tier = GpuSupportTier::Unsupported;

    std::string backend_name; ///< "cuda" / "hip" / "sycl" / "none"
    std::string device_name;  ///< as reported by the driver, "" when none

    int  compute_major = 0;
    int  compute_minor = 0;
    int  multiprocessors = 0;
    long long total_global_bytes = 0;
    long long l2_cache_bytes     = 0;
    int  shared_bytes_per_block  = 0;

    bool has_graphs             = false;
    bool has_conditional_graphs = false; ///< Sec 5.5 backend gate
    bool has_memory_pools       = false; ///< Sec 4.1 cudaMemPool allocation policy
    bool has_stream_ordered_alloc = false;

    /// Deliberately absent: any "supports cooperative launch" field.  The
    /// persistent track is closed by W0 (c_barrier = 0.78 us > 0.384 us kill
    /// threshold); a capability flag for it would invite code that reads it.
};

// ---------------------------------------------------------------------------
// Problem-fixed shape constants, mirroring the host solver.
// ---------------------------------------------------------------------------

inline constexpr int kDevNg      = 2;  ///< xsrecon::NG   -- every accepted deck is 2-group
inline constexpr int kDevNg2     = 4;  ///< ng * ng
inline constexpr int kDevNiso    = 39; ///< xsrecon::NISO, Chiffon::Isotope::isotopeIds.size()
inline constexpr int kDevNxs     = 11; ///< xsrecon::NXS, XSTF..XS3N scalar slots
inline constexpr int kDevNdirMax = 3;  ///< pch.h NDIRMAX
inline constexpr int kDevLr      = 2;  ///< pch.h LR
inline constexpr int kDevNews    = 4;  ///< pch.h NEWS
inline constexpr int kDevNewsbt  = 6;  ///< pch.h NEWSBT

/// Chiffon::XSTYPE scalar slot indices, mirroring xsrecon's copy so a device TU
/// never has to include Model.h.
inline constexpr int kXtXstf = 0, kXtXsdf = 1, kXtXsaf = 2, kXtXsff = 3, kXtXsnf = 4,
                     kXtXskf = 5, kXtXssf = 6, kXtXsrf = 7, kXtFyld = 8, kXtXs2n = 9,
                     kXtXs3n = 10;

/// The four microscopic slots the predictor/corrector actually reads back at
/// beginning-of-step.  BuildTransitionMatrix (XSSet.cpp:3380-3407) touches only
/// XSAF, XSFF, XS2N and XS3N of the condensed micro block; the macroscopic
/// XSKF row is read separately for the power average (XSSet.cpp:4189).  Rev.7.1
/// Sec 6.18 replaces Rev.7's "remove the BOS copy" with "keep 4 slots, not 11",
/// and this array is that decision as data.
inline constexpr int kBosMicroXt[4] = {kXtXsaf, kXtXsff, kXtXs2n, kXtXs3n};
inline constexpr int kBosMicroXtCount = 4;

// ---------------------------------------------------------------------------
// Sec 3.3  Immutable shared geometry -- one copy for the whole cohort
// ---------------------------------------------------------------------------

/// Geometry is fixed for a cohort, so it lives OUTSIDE the slot stride: one
/// copy, shared by every slot, never written after import.  All pointers are
/// const for that reason -- a phase kernel that needs to write geometry is a
/// design error, not a missing const_cast.
struct DeviceGeometryView {
    const double* hmesh;   ///< [l*NDIRMAX + dir]
    const double* hz;      ///< [nz]
    const double* vol;     ///< [nxyz]
    const double* vola;    ///< [nxya * nz]
    const double* albedo;  ///< [dir*LR + side]

    const int* neib;    ///< [l*NEWSBT + dir*LR + side]
    const int* neibr;   ///< [l2d*NEWS + news]
    const int* neibrb;  ///< [l2d*NEWS + news], reflective BC
    const int* lklr;    ///< [ls*LR + side]
    const int* idirlr;  ///< [ls*LR + side]
    const int* sgnlr;   ///< [ls*LR + side]
    const int* lktosfc; ///< [(l*NDIRMAX + dir)*LR + side]
    const int* ltola;   ///< [nxy]
    const int* ltolc;   ///< [NEWS*nxy]
    const int* comps;   ///< [nxyz]

    const int* fuel_nodes; ///< [n_fuel], built once at setup
    const int* is_fuel;    ///< [nxyz], 0/1 (int, not bool: bool has no device ABI guarantee)

    int nxyz;
    int nsurf;
    int nxy;
    int nz;
    int nxya;
    int n_fuel;
    int ng;
    int symang;
};

// ---------------------------------------------------------------------------
// Sec 3.4  Immutable shared XS library -- one copy for the whole cohort
// ---------------------------------------------------------------------------

/// The parameterised library (reference depletion points + branch delta
/// coefficients) is identical for every slot running the same library file, so
/// it is shared exactly like geometry.  Slot-specific reconstruction products
/// live in DeviceSlotView.
struct DeviceXsLibraryView {
    const double* lib_lmpx;       ///< [dpt*NXS-slot-major], reference lumped XS
    const double* lib_lmpx_ssm;
    const double* lib_micx;       ///< [(dpt*niso + iso)*ng + ig] per scalar slot
    const double* lib_micx_ssm;
    const double* lib_iden;       ///< [dpt*niso + iso]
    const double* lib_burn;       ///< [dpt], reference burnup [GWd/tHM]
    const double* lib_wvfr;       ///< [dpt]
    const double* lib_flux;       ///< [dpt*ng + ig]
    const double* lib_chix;       ///< [dpt*ng + ig]

    const double* coeff_lmpx;     ///< branch delta coefficients, lumped
    const double* coeff_lmpx_ssm;
    const double* coeff_micx;     ///< branch delta coefficients, microscopic
    const double* coeff_micx_ssm;
    const double* knots;          ///< concatenated spline knots
    // No `knot_offsets`: the per-branch knot offset is an int FIELD inside the
    // branch descriptor structs (XSSet.cpp:419, 451), not a standalone array.
    // A view pointer with no host array behind it is worse than a missing one --
    // it reads as available and dereferences to whatever the arena last wrote
    // there.  Add it when a descriptor block exists to point at.

    const double* dep_decay;      ///< [niso*niso] decay matrix
    const double* dep_trans;      ///< [niso*niso] transmutation topology
    const double* cram_alpha;     ///< [pole_count], milk.h:1676-1701 order-8 constants
    const double* cram_theta;     ///< [pole_count]
    double        cram_alpha0;

    int n_ref_points;
    int n_coeff_points;
    int n_knots;
    int niso;
    int ng;
    int cram_order;  ///< XSSet.cpp:4008 CRAM_ORDER = 8
    int cram_poles;  ///< milk.h:1676-1701 pole_count = 4 (not 8)
    int cram_first;  ///< XSSet.cpp:4008 first = iI135 = 3
};

// ---------------------------------------------------------------------------
// Sec 3.5  Per-slot mutable bulk state
// ---------------------------------------------------------------------------

/// One slot's writable arrays plus the four control structs.  Rev.7's
/// `DeviceSlotControl* control` and `double* search_history` are replaced by the
/// four pointers at the top; Rev.7.1 adds the reference micro/lumped blocks and
/// the Xe Anderson history, both of which Rev.7's Sec 3.6 table omitted while
/// the Sec 6.14 flat-XS path depends on the first and Sec 6.15 on the second.
struct DeviceSlotView {
    // --- Sec 3.2 control packet, four-way split ---
    DeviceSlotPhase*      phase;
    DeviceSlotState*      state;
    DeviceSearchState*    search;
    DeviceScheduleParams* params;

    // --- flux / current state ---
    double* phif; ///< [l*ng + ig]  (AoS, matching Geometry::Phif)
    double* phis; ///< [((l*NDIRMAX + dir)*LR + side)*ng + ig]
    double* jnet; ///< same shape as phis
    double* psi;  ///< [nxyz]
    double* phic; ///< [(l*ng + ig)*NEWS + corner]

    // --- CMFD operator + Krylov workspace ---
    double* dtil;      ///< [ig*nsurf + ls]
    double* dhat;      ///< [ig*nsurf + ls]
    double* diag;      ///< [l*ng2 + ...]
    double* cc;        ///< [(l*ng + ig)*NEWSBT + dir]
    double* src;       ///< [ig*nxyz + l]
    double* cmfd_psi;  ///< [nxyz], the CMFD fission source (distinct from `psi`)
    double* bicg_vec;  ///< 9 packed BiCGSTAB vectors, each [ng*nxyz]
    double* bicg_dinv; ///< [ng2*nxyz] SSOR preconditioner

    // --- nodal state and updateConstant products ---
    double* trlcff;     ///< 3 packed arrays, each [nxyz*NDIRMAX*ng]
    double* nodal_const;///< 9 packed arrays (eta1/eta2/m260/m251/m253/m262/m264/diagDI/diagD)
    double* constant_xs;///< 2 packed arrays (xsrf, xsdf), each [nxyz*ng]
    double* dsncff;     ///< 3 packed arrays, each [nxyz*NDIRMAX*ng]
    double* mutau;      ///< 2 packed arrays, each [nxyz*NDIRMAX*ng2]
    double* matm;       ///< 4 packed arrays (M, MI, Ms, Mf), each [nxyz*ng2]

    // --- isotopics and cross sections ---
    double* iden;      ///< [iso*nxyz + l], live isotope densities
    double* xs;        ///< live macroscopic, NXS packed scalar slots [ig*nxyz + l]
    double* xs_ssm;    ///< [(igs*ng + ige)*nxyz + l]
    double* lmpx;      ///< live lumped, same packing as xs
    double* lmpx_ssm;
    double* micx;      ///< live microscopic, [(iso*ng + ig)*nxyz + l] per slot
    double* micx_ssm;  ///< [(iso*ng*ng + igs*ng + ige)*nxyz + l]

    /// Reference micro/lumped blocks (XSSet.h:210-212).  Device-resident and
    /// re-uploaded only when the LIBRARY changes -- Rev.7 claimed these could be
    /// removed; Sec 6.14's Task 12a apply path reads them every flat-XS update,
    /// so they cannot.  ~65 MiB/slot at APR1400 size, and the reason Sec 3.6
    /// names L2 (not DRAM) as the real risk.
    const double* ref_micx;
    const double* ref_micx_ssm;
    const double* ref_lmpx;
    const double* ref_lmpx_ssm;
    const double* ref_iden;

    // --- beginning-of-step snapshot for predictor/corrector (Sec 6.18) ---
    double* bos_micx;  ///< kBosMicroXtCount packed slots, each [(iso*ng + ig)*nxyz + l]
    double* bos_xskf;  ///< [ig*nxyz + l], macroscopic kappa-fission only
    double* bos_iden;  ///< [iso*nxyz + l]
    double* bos_flux;  ///< [l*ng + ig]
    int*    bos_burn_key;

    /// Xe Anderson history (Driver.h:999-1020).  kDevXeAndersonTriples triples
    /// of (I, Xe, Xem) over the fuel nodes:
    ///   (6 + 2*XE_ANDERSON_DEPTH) * 3 * n_fuel doubles.
    double* xe_aa_history;

    // --- thermal-hydraulics / rod state ---
    double* bppm;
    double* tful;
    double* tmod;
    double* dmod;
    double* rod_fraction;
    double* node_wvfr;

    // --- pin power reconstruction (Sec 6.22, Task 19b) ---
    double* ppr_p;  ///< [nxyz*ng*15]
    double* ppr_a;  ///< [nxyz*ng*8]
    double* ppr_c;  ///< [nxyz*ng*15]
    double* ppr_q;  ///< [nxyz*ng*15]
    double* ppr_l;  ///< [nxyz*ng*9]
    double* ppr_bt; ///< [nxyz*ng]

    // --- integer node keys ---
    /// Burnup BRACKET key, an int (XSSet.h:161 `std::vector<int> _burn`).
    /// Sec 6.16(3): the bracket search must not be a floating-point compare, or
    /// a boundary node can land in different brackets on host and device.  The
    /// device reproduces the host's INTEGER key; Rev.7's `double burnup` here
    /// was the bug.
    int* burn_key;
    int* ctyp_key;

    // --- output packing ---
    double* out_pack;

    // --- shapes ---
    int nxyz;
    int nsurf;
    int nxy;
    int n_fuel;
    int ng;
    int slot; ///< physical slot index this view is rebased onto
};

// ---------------------------------------------------------------------------
// Sec 5.9 / Sec 4.1  What a phase kernel is handed  -- [Rev.7.1 Task 4]
// ---------------------------------------------------------------------------

/// The device-side face of the arena: the per-slot views plus the contiguous
/// Sec 3.2 control block.  ONE of these is built by the host after reserve()
/// and passed by value into every phase kernel.
///
/// WHY A TABLE OF VIEWS AND NOT A STRIDE.  slotView(s) on the host is a pure
/// index rebase, so a kernel could recompute it from a slot-0 view and a byte
/// stride.  It must not: the stride is layout knowledge, and a kernel that
/// carries layout knowledge is a second opinion about the arena that will drift
/// from GpuPhysicsArenaLayout.h.  The arena fixes every per-slot address in
/// reserve() and never allocates again, so the host builds this table once,
/// uploads it once, and a captured graph's baked argument stays valid for the
/// whole run -- which is the entire reason the arena has that property.
///
/// THE RESOLUTION ORDER IS PART OF THE CONTRACT.  Every phase kernel does:
///
///     if (gpuDispatchIsPadding(logical, queue.count)) return;
///     const int slot = queue.slots[logical];        // NEVER blockIdx as a slot
///     const DeviceSlotView v = arena.slotView(slot);
///
/// `logical` is the lane's index into the queue; the queue is dispatched at the
/// BUCKET width, so lanes past `count` read kQueueEmptySlot and must return
/// before touching it.
///
/// WHAT A PHASE KERNEL MAY WRITE.  `states[slot]` and the slot's bulk arrays.
/// It may NOT write `phases[slot].queued_phase` or `.queued_epoch`: those are
/// captured by classify and are what makes a stale queue entry go stale by
/// itself (Sec 5.2).  A phase kernel that stamps them re-validates the entry it
/// is currently consuming, and the slot is queued twice on the next epoch.
struct DeviceArenaView {
    const DeviceSlotView* slot_views; ///< [slot_count], built once after reserve()

    // Sec 3.2 control block, four dense arrays below slot_base.
    DeviceSlotPhase*      phases;
    DeviceSlotState*      states;
    DeviceSearchState*    searches;
    DeviceScheduleParams* params;

    int slot_count;

    RASBERY_GPU_HD const DeviceSlotView& slotView(int slot) const {
        return slot_views[slot];
    }
};

// ---------------------------------------------------------------------------
// Device view contract.  These are the properties that make a view safe to hand
// to a kernel; asserting them here means a bad member is a compile error in
// every build, not a grep miss.
// ---------------------------------------------------------------------------

static_assert(std::is_trivially_copyable_v<DeviceArenaView>);
static_assert(std::is_standard_layout_v<DeviceArenaView>);

static_assert(std::is_trivially_copyable_v<DeviceGeometryView>);
static_assert(std::is_trivially_copyable_v<DeviceXsLibraryView>);
static_assert(std::is_trivially_copyable_v<DeviceSlotView>);
static_assert(std::is_standard_layout_v<DeviceGeometryView>);
static_assert(std::is_standard_layout_v<DeviceXsLibraryView>);
static_assert(std::is_standard_layout_v<DeviceSlotView>);
static_assert(std::is_same_v<decltype(DeviceSlotView::burn_key), int*>,
              "Sec 3.5: the burnup bracket key is an INTEGER key, not a double burnup");

// ---------------------------------------------------------------------------
// Backend facade
// ---------------------------------------------------------------------------

/// The one object that knows whether a device exists and what it can do.
///
/// Every backend arm -- the CUDA .cu and the no-CUDA stub -- defines exactly
/// these symbols, so call sites never need an #ifdef.  Same shape as
/// XsReconBackend / CudaBICGBackend, which is the convention in this tree.
class GpuPhysicsBackend {
public:
    GpuPhysicsBackend();
    ~GpuPhysicsBackend();

    GpuPhysicsBackend(const GpuPhysicsBackend&)            = delete;
    GpuPhysicsBackend& operator=(const GpuPhysicsBackend&) = delete;

    /// True when a device was found and every probe so far has succeeded.
    /// Always false in a build without a device backend.
    [[nodiscard]] bool available() const;

    /// Human-readable reason when available() is false.
    [[nodiscard]] const std::string& status() const;

    /// What the probe found.  Valid whether or not available() is true; when it
    /// is false the tier is Unsupported and the numeric fields are zero.
    [[nodiscard]] const GpuCapabilityReceipt& capability() const;

    /// Which backend this binary was COMPILED with, independent of whether a
    /// device was found at run time.
    [[nodiscard]] static GpuBackendKind compiledKind();

    /// "cuda" / "hip" / "sycl" / "none" for `kind`.
    [[nodiscard]] static const char* backendName(GpuBackendKind kind);

    /// One line of JSON for the run receipt (Sec 9.3).
    [[nodiscard]] std::string receiptJson() const;

private:
    struct Impl;
    Impl* _impl;
};

/// Process-wide gate, read once, mirroring rasberyGpuNodalEnabled() and friends.
bool rasberyGpuPhysicsEnabled();

} // namespace rasbery::gpu
