#pragma once

// Per-slot device control packet -- Rev.7.1 plan Sec 3.1 / 3.2.
//
// THE HOLE THIS CLOSES.  Rev.7 carried ONE 128-byte `DeviceSlotControl` and
// called it the case state.  It is not.  A real case carries ~60 more fields
// than that packet has room for: the whole pre-search state machine
// (Scheduler.h:155-177), the cross-statepoint secant memory (Scheduler.h:55-61),
// the per-deck tolerances and limits Rev.7 assumed were shared constants
// (Scheduler.h:127-148), and the Xe damper / ONCE / Anderson scalars
// (Driver.h:1434-1500).  A slot refilled from a packet that never held those
// fields inherits the previous tenant's search bracket, its frozen secant slope
// and its Xe streak counter -- silently, because nothing in the packet says the
// value is stale.  That is the stale-tenant class the campaign already paid for
// once in --batch-mode (see HostPinRegistry.h), arriving a second time through
// a different door.
//
// WHY FOUR STRUCTS AND NOT ONE BIG ONE.  Level-1 of the scheduler is a 1-CTA
// classify/compact pass that runs once per epoch and reads NOTHING but the
// phase word of every slot.  At 64 slots a 32-byte hot struct is 2 KiB: it
// stays resident in L1 across the ~68k epochs a single deck spends in the
// scheduler.  The 128-byte packet Rev.7 proposed is 8 KiB, which does not, so
// every epoch re-fetches it through L2.  The split is therefore not tidiness --
// it is the difference between classify traffic that disappears into L1 and
// classify traffic that competes with the CMFD hot set for the same L2 that
// Sec 3.6 already names as the real risk.
//
//   (A) DeviceSlotPhase       32 B   hot; the ONLY struct Level-1 reads
//   (B) DeviceSlotState      cold;   phase kernels only
//   (C) DeviceSearchState    cold;   the 26 fields of Scheduler.h:55-61,155-177
//   (D) DeviceScheduleParams read-mostly; per-slot deck parameters
//
// OWNERSHIP AND EPOCHS (Sec 5.2).  A queue entry is valid only while the epoch
// it captured still equals the slot's `state_epoch`.  `queued_epoch ==
// state_epoch && queued_phase == phase` means the slot is ALREADY queued and
// must not be inserted again; queueing a slot whose `in_flight` bit is set is a
// fatal scheduler error, not a retry.  Every phase transition and every tenant
// refill bumps `state_epoch`, which is what invalidates the stale entries
// without walking the queues.
//
// REFILL IS A FULL RESET (Sec 3.2, "재활용 감사와 epoch 규칙의 결합").  All
// four structs are reset -- not just the hot one.  DeviceSearchState in
// particular carries SearchMemory, which is deliberately carried ACROSS
// statepoints within one deck and must never survive into a different deck's
// tenancy.  The reset helpers at the bottom of this header are the single
// definition of "reset"; the Task 20 audit kernel checks their post-condition,
// so the bytes have to exist here before the audit can exist at all (that is
// why Task 1 precedes Task 20).
//
// NO PERSISTENT / COOPERATIVE SCAFFOLDING.  W0 measured c_barrier = 0.78 us on
// sm_120 against a kill threshold of 0.384 us (constraint 17, Rev.7.1).  The
// persistent case-phase scheduler is dead and W3.7 is skipped, so nothing in
// this header or anything downstream of it may introduce cooperative groups,
// grid-wide barriers or resident-block bookkeeping.  Dispatch is measured at
// c_dispatch = 0.783 us per launch and that is the cost model the scheduler is
// designed against.
//
// LAYERING.  Header-only, CUDA-free, and free of every non-trivially-copyable
// type: these structs are memcpy'd to the device as bytes and are read by
// device code, so no std::vector, no std::string, no virtuals, no references.
// Compiles in the no-CUDA stub build exactly as it compiles under nvcc.

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
    #define RASBERY_GPU_HD __host__ __device__
#else
    #define RASBERY_GPU_HD
#endif

namespace rasbery::gpu {

/// Physical slot cap, shared by the arena layout and the scheduler so the two
/// cannot disagree about how wide the fleet may be.  64 is the campaign width
/// and the widest conditional-graph bucket (Sec 5.5).
///
/// It lives here, with the control structs, because both consumers already
/// include this header and neither includes the other: the arena sizes the
/// control block by it, the scheduler sizes its queues by it, and a layout that
/// asked for more slots than the scheduler can classify would be a silent
/// truncation.
inline constexpr int kMaxDeviceSlots = 64;

// ---------------------------------------------------------------------------
// Sec 3.1  Phase and exit states
// ---------------------------------------------------------------------------

/// One unit of case work.  The scheduler picks ONE phase per epoch and runs it
/// for every slot that is in it (Sec 5.5 SWITCH), so this enum is also the
/// queue set.
///
/// The five entries between `Search` and `DepletionPredictor` are the Rev.7.1
/// additions: they are the points where Driver.h's statepoint loop still did
/// host arithmetic after Rev.7's list was exhausted, and M2 (Sec 9.2) is not
/// reachable while any of them runs on the CPU.
enum class DevicePhase : std::uint32_t {
    Empty = 0,
    Import,
    Material,
    Outer,
    Xenon,
    ThermalHydraulics,
    Search,
    NormalizeFluxSign, ///< Driver.h:2235, 2239, 2261, 2274
    Derivative,        ///< Driver.h:2245 UpdateDerivative + 2268-2274 re-converge retry
    RodOp,             ///< Driver.h:2252-2255 SetRod / ResetFluxAndCurrents / resetDhat / eigv=1
    Ppr,               ///< Driver.h:2278-2295 reset / drive / reconstructPinPower
    ResultAggregate,   ///< Driver.h:2308 AddResult
    DepletionPredictor,
    DepletionCorrector,
    OutputPack,
    Done,
    Failed
};

/// Number of DevicePhase values; sized so a phase can index a fixed table.
inline constexpr int kDevicePhaseCount = static_cast<int>(DevicePhase::Failed) + 1;

/// Why a phase stopped.  Device code cannot throw, so every host `throw` on a
/// numerical path has to arrive here instead -- see CramZeroDiagonal and
/// CramNotConverged, which are milk.h:1765 and milk.h:1807 verbatim.  A slot
/// that takes either one goes to DevicePhase::Failed; the other slots keep
/// running (Task 16 Step 4).
enum class DeviceEscape : std::uint32_t {
    None = 0,
    FluxConverged,
    FluxLimitCycleSample, ///< noisy limit cycle: sample taken, solve continues
    FluxStallFatal,       ///< stall budget exhausted
    NegativeFlux,
    RayleighFallback,
    SegmentBudget,
    MaterialChanged,
    NonFinite,
    MaxIteration,
    CramZeroDiagonal, ///< milk.h:1765  "CRAM Gauss-Seidel zero diagonal"
    CramNotConverged  ///< milk.h:1807  "CRAM Gauss-Seidel did not converge"
};

inline constexpr int kDeviceEscapeCount = static_cast<int>(DeviceEscape::CramNotConverged) + 1;

// ---------------------------------------------------------------------------
// Sec 3.2 (A)  DeviceSlotPhase -- hot 32 B, the only struct Level-1 reads
// ---------------------------------------------------------------------------

/// `DeviceSlotPhase::flags` bits.  Kept as bits rather than four bytes because
/// the struct has exactly 32 bytes to spend and `active` is read by the same
/// load that reads `phase`.
inline constexpr std::uint8_t kSlotFlagActive     = 0x1u;
inline constexpr std::uint8_t kSlotFlagInFlight   = 0x2u;
inline constexpr std::uint8_t kSlotFlagFatal      = 0x4u;
inline constexpr std::uint8_t kSlotFlagInputReady = 0x8u;

struct alignas(32) DeviceSlotPhase {
    std::uint8_t phase;        ///< DevicePhase
    std::uint8_t queued_phase; ///< DevicePhase captured at queue insertion
    std::uint8_t escape;       ///< DeviceEscape
    std::uint8_t flags;        ///< kSlotFlag* bits

    std::uint32_t state_epoch;  ///< bumped on every phase transition and every refill
    std::uint32_t queued_epoch; ///< state_epoch captured at queue insertion
    std::uint32_t phase_age;    ///< epochs spent in `phase`; fairness input (Sec 5.4)
    std::uint32_t input_id;
    std::uint32_t job_id;
    std::uint32_t error_code;
    std::uint32_t reserved;
};

RASBERY_GPU_HD inline bool slotActive(const DeviceSlotPhase& p) {
    return (p.flags & kSlotFlagActive) != 0u;
}
RASBERY_GPU_HD inline bool slotInFlight(const DeviceSlotPhase& p) {
    return (p.flags & kSlotFlagInFlight) != 0u;
}
RASBERY_GPU_HD inline bool slotFatal(const DeviceSlotPhase& p) {
    return (p.flags & kSlotFlagFatal) != 0u;
}
RASBERY_GPU_HD inline bool slotInputReady(const DeviceSlotPhase& p) {
    return (p.flags & kSlotFlagInputReady) != 0u;
}

/// Sec 5.2 duplicate-prevention predicate, in one place so the scheduler core
/// and the contract test cannot drift apart: a slot is already queued when the
/// entry it captured is still current.
RASBERY_GPU_HD inline bool slotAlreadyQueued(const DeviceSlotPhase& p) {
    return p.queued_epoch == p.state_epoch && p.queued_phase == p.phase;
}

// ---------------------------------------------------------------------------
// Sec 3.2 (B)  DeviceSlotState -- cold, phase kernels only
// ---------------------------------------------------------------------------

// The tail padding alignas(128) adds is the POINT (Task 1 Step 4 asserts
// sizeof % 128 == 0, so a slot's cold block always starts on a 128-byte
// boundary and never straddles two sectors).  MSVC warns about it anyway.
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4324)
#endif

struct alignas(128) DeviceSlotState {
    // --- schedule / statepoint position ---
    std::uint32_t schedule_index;
    std::uint32_t statepoint;
    std::uint32_t substep;         ///< Driver.h:2227  nsub = max(1, schedule.substep)
    std::uint32_t substep_index;   ///< Driver.h:2229  isub (predictor/corrector position)
    std::uint32_t solve_call_kind; ///< isub>0 pre-solve / predictor / final / derivative-retry

    // --- outer / convergence counters ---
    std::uint32_t outer_in_segment;
    std::uint32_t total_outer;
    std::uint32_t flux_stall;
    std::uint32_t stall_events;
    std::uint32_t stall_sample_taken; ///< limit-cycle sample already consumed
    std::uint32_t cmfd_sweeps;
    std::uint32_t bicg_iterations;
    std::uint32_t clean_iters; ///< consecutive outers with no stall

    // --- Xe damper / ONCE / Anderson scalars (Driver.h:1434-1500) ---
    std::uint32_t xe_iterations;
    std::uint32_t xe_total;           ///< settled Xe steps over the whole SolveLoop
    std::uint32_t xe_interim_count;   ///< RASBERY_XE_INTERIM_L2 loose-flux steps
    std::uint32_t xe_no_progress;     ///< consecutive contraction failures
    std::uint32_t xe_streak;          ///< oscillation streak vs xe_streak_limit
    std::uint32_t xe_cap_charged;     ///< one starvation charge per cascade
    std::uint32_t xe_cascade_index;
    std::uint32_t xe_aa_ncol;         ///< usable Anderson difference columns (Driver.h:1008)
    std::uint32_t xe_aa_have_prev;    ///<                                    (Driver.h:1009)
    double        xe_relax;           ///< 1.0 or XE_DAMPED_RELAX
    double        prev_xe_change;
    double        xe_residual;

    // --- TH / search residuals ---
    std::uint32_t th_iterations;
    std::uint32_t search_iterations;
    double        th_residual;
    double        search_residual;

    // --- eigenvalue ---
    double eigv;
    double previous_eigv;
    double eigv_before_segment; ///< Driver.h:2259 eigv_before (re-converge decision basis)
    double flux_l2;
    double keff_tolerance;
    double flux_tolerance;

    // --- generation counters ------------------------------------------------
    //
    // FOUR OF THESE MIRROR A COUNTER THAT EXISTS ON THE HOST TODAY.  The rest
    // are placeholders for state the device phases will need to invalidate once
    // they exist, and they are called out as such so nobody wires an upload
    // decision to a counter that nothing ever bumps:
    //
    //   REAL, host-backed (the device MUST keep these in step):
    //     micx_generation      <- XSSet.h:180  _micx_generation
    //                             (host rebuilt _micx/_lmpx; re-run reconstruct)
    //     ref_generation       <- XSSet.h:198  _ref_generation
    //                             (PrecomputeBranchCoefficients rebuilt the
    //                              reference blocks; re-upload _ref_micx/_ref_lmpx)
    //     hoststate_generation <- XSSet.h:190  _hoststate_generation
    //                             (host wrote _xs/_iden outside the device path)
    //     nodal_constant_generation <- Nodal.h:124 _const_generation
    //                             (updateConstant products are stale)
    //
    //   SPECULATIVE (no host counter yet; device-side only until a phase owns
    //   one).  Do not gate an upload on these until the owning task lands.
    //     geometry_generation   Task 4    operator_generation  Task 5
    //     flux_generation       Task 9    current_generation   Task 5
    //     dhat_generation       Task 7    isotope_generation   Task 16
    //     th_generation         Task 14
    //
    //   material_generation LEFT THAT LIST IN Rev.7.1 TASK 9 (link 5).
    //   XSSet::hoststateGeneration() is its host counter -- every site that
    //   writes `_xs` already bumps it (XSSet.cpp:1060, 1474, 2774, 2801,
    //   3180, 3280, 3759, 3955, 4043; SetBoron/SetRod reach it through
    //   UpdateFlatXS) -- and rasberyStandUpOuterSegment stamps it into the
    //   slot at stand-up.  It is therefore SAFE to gate on, which is what
    //   nodalConstantSlotIsCurrent does; before the stamp both counters sat
    //   at zero and that gate read `current` for the whole run.
    //
    // Every one of them is a refill reset target either way: a survivor here
    // suppresses a rebuild the NEW tenant needs, which is the failure mode the
    // whole four-struct reset exists to prevent.
    std::uint64_t micx_generation;
    std::uint64_t ref_generation;
    std::uint64_t hoststate_generation;
    std::uint64_t nodal_constant_generation;

    std::uint64_t geometry_generation;
    std::uint64_t material_generation;
    std::uint64_t operator_generation;
    std::uint64_t flux_generation;
    std::uint64_t current_generation;
    std::uint64_t dhat_generation;
    std::uint64_t isotope_generation;
    std::uint64_t th_generation;
};

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

// ---------------------------------------------------------------------------
// Sec 3.2 (C)  DeviceSearchState -- Scheduler.h:55-61 + 155-177, all 26 fields
// ---------------------------------------------------------------------------

/// Rev.7 spent one line on this ("double* search_history; // bounded scalar
/// packet").  There is no history array to point at: the critical search keeps
/// exactly 26 scalars, which is also why the Rev.7.1 thread mapping is one
/// thread per slot and not 32 lanes scanning a history (Sec 6.17).
struct DeviceSearchState {
    // Scheduler.h:55-61  SearchMemory -- secant memory carried across statepoints
    std::uint32_t has_boron_secant;
    double        boron_secant_dkdx;
    std::uint32_t has_rod_secant;
    double        rod_secant_x;
    double        rod_secant_dkdx;

    // Scheduler.h:155-171  runtime search state kept across SolveLoop calls
    std::uint32_t initialized;
    std::uint32_t seeded_from_previous_step;
    std::uint32_t has_prev;
    std::uint32_t has_bracket;
    std::uint32_t has_best;
    std::uint32_t slope_frozen;
    std::uint32_t iteration;
    double        current_x;
    double        prev_x;
    double        prev_eigv;
    double        frozen_slope;
    double        best_x;
    double        best_residual;
    double        bracket_lo_x;
    double        bracket_lo_residual;
    double        bracket_hi_x;
    double        bracket_hi_residual;

    // Scheduler.h:174-177  termination bookkeeping (published to the result file)
    std::uint32_t exit_status; ///< SearchExit{NONE, CONVERGED, BEST_FALLBACK, UNCONVERGED}
    double        exit_dk;
    double        exit_tol;
    std::uint32_t stall_count;
};

/// Mirror of rasbery::SearchExit (Scheduler.h:48-53).  Duplicated rather than
/// included because Scheduler.h drags in Geometry.h, std::map and std::string,
/// none of which may reach a device translation unit.  The contract test pins
/// the two enumerations to the same values.
enum class DeviceSearchExit : std::uint32_t {
    None         = 0,
    Converged    = 1,
    BestFallback = 2,
    Unconverged  = 3
};

// ---------------------------------------------------------------------------
// Sec 3.2 (D)  DeviceScheduleParams -- per-slot deck parameters (read-mostly)
// ---------------------------------------------------------------------------

/// Mirrors of rasbery::SearchType / ScheduleType / THMode (Scheduler.h:24-43),
/// duplicated for the same reason DeviceSearchExit is.
enum class DeviceSearchType : std::uint32_t { Keff = 0, Boron = 1, RodCrit = 2 };
enum class DeviceScheduleType : std::uint32_t {
    Standard   = 0,
    Depletion  = 1,
    Derivative = 2,
    Rod        = 3
};
enum class DeviceThMode : std::uint32_t { None = 0, Steady = 1, Transient = 2 };

/// Rev.7 assumed every slot shared one set of tolerances and limits.  Under a
/// large-input campaign each slot is a DIFFERENT deck, so the tolerances differ
/// per slot and a shared constant silently applies deck A's search tolerance to
/// deck B.  These are the Scheduler.h:127-148 search parameters plus the
/// Scheduler.h:91-124 physical conditions and perturbations.
struct DeviceScheduleParams {
    // Scheduler.h:127-148  search / convergence parameters
    std::uint32_t search_type;     ///< DeviceSearchType
    std::uint32_t schedule_type;   ///< DeviceScheduleType
    std::uint32_t th_mode;         ///< DeviceThMode
    std::uint32_t max_outer_iter;  ///< default kMaxEigenIter  = 200
    std::uint32_t max_search_iter; ///< default kMaxSearchIter = 300
    std::uint32_t max_th_iter;     ///< default kMaxThIter     = 10
    double        tolerance_keff;      ///< kEigvTol          = 1e-6
    double        tolerance_search;    ///< kCritSearchTol    = 1e-5
    double        rodcrit_search_floor;
    // WP24.  The RODCRIT clamp, whose host twin stopped being a literal when
    // the fidelity preset had to be able to move it (Scheduler.h
    // criticalSearchTolerance).  It is here for the SAME reason
    // rodcrit_search_floor is -- the host and device search parameter blocks are
    // pinned field-for-field by tools/test_gpu_physics_interface_contract.py, so
    // a host field with no device twin is a mirror that has quietly stopped
    // being one.
    //
    // TODO(WP24-device-search): RESET-ONLY TODAY, AND THAT IS A HOLE THE
    // MIRROR CONTRACT CANNOT SEE.  resetDeviceScheduleParams() writes
    // kDevRodCritSearchTol here and nothing ever feeds it from
    // ctx.tolerances.rodcrit_search_cap, so the day the device search goes
    // live the device would clamp at the BUILT-IN while the host clamps at the
    // preset's -- two search tolerances inside one solve, which is exactly the
    // "half applied" failure src/FidelityPreset.h exists to prevent, arriving
    // on the axis nobody is watching.  `search_tol_cap` has no device twin at
    // all for the same reason (it is an argument to
    // Scheduler::criticalSearchTolerance, not a Schedule field), so it lands
    // here too.  The contract test pins this marker, so the reminder cannot be
    // deleted without wiring the value.
    double        rodcrit_search_cap;
    double        tolerance_th;        ///< kThTol            = 1e-6
    double        target_keff;
    double        tolerance_tmod;      ///< kTempSearchTol    = 0.01
    double        tolerance_tfuel;     ///< kTempSearchTol    = 0.01
    double        search_boron_ppm;
    double        tolerance_boron;     ///< kBoronSearchTol   = 0.01
    double        tolerance_rodsearch; ///< kRodSearchTol     = 0.01
    double        search_relaxation;   ///< kSearchRelax      = 1.0
    double        search_low;          ///< kSearchLow        = 0.0
    double        search_hi;           ///< kSearchHigh       = 1.0
    double        slope_freeze_thres;  ///< kSlopeFreezeThres = 0.01
    double        min_secant_denom;    ///< kMinSecantDenom   = 1e-12
    double        bracket_min_span;    ///< kBracketMinSpan   = 1e-6
    double        search_boron_probe;  ///< kBoronProbe       = 50.0
    double        search_rod_probe;    ///< kRodProbe         = 0.25

    // Scheduler.h:91-124  physical conditions and derivative perturbations
    double        time;
    double        burnup;
    double        rate;
    double        rated_power;
    double        actual_power;
    double        step_dt;
    std::uint32_t substeps;
    std::uint32_t xenon_transient;
    std::uint32_t use_burnup_time;
    double        bppm0;
    double        tful0;
    double        tmod0;
    double        dmod0;
    double        pressure;
    double        inlet_temp;
    double        outlet_temp;
    double        mass_flow_rate;
    double        fuel_temp_rise_scale;
    double        delta_tful;
    double        delta_tmod;
    double        delta_dmod;
    double        delta_bppm;
    double        delta_xe;
    double        delta_sm;
};

// ---------------------------------------------------------------------------
// Device-side mirrors of the Scheduler.h defaults.
//
// Scheduler.h cannot be included here (Geometry.h, std::map, std::string), so
// the constants are restated.  The contract test compares these to the
// `inline constexpr` definitions in Scheduler.h numerically, which is what
// makes the duplication safe.
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kDevMaxEigenIter     = 200u;
inline constexpr std::uint32_t kDevMaxSearchIter    = 300u;
inline constexpr std::uint32_t kDevMaxThIter        = 10u;
inline constexpr double        kDevEigvTol          = 1.0e-6;
inline constexpr double        kDevCritSearchTol    = 1.0e-5;
inline constexpr double        kDevRodCritSearchTol = 1.0e-5;
inline constexpr double        kDevThTol            = 1.0e-6;
inline constexpr double        kDevTempSearchTol    = 0.01;
inline constexpr double        kDevBoronSearchTol   = 0.01;
inline constexpr double        kDevRodSearchTol     = 0.01;
inline constexpr double        kDevSearchRelax      = 1.0;
inline constexpr double        kDevSearchLow        = 0.0;
inline constexpr double        kDevSearchHigh       = 1.0;
inline constexpr double        kDevSlopeFreezeThres = 0.01;
inline constexpr double        kDevMinSecantDenom   = 1.0e-12;
inline constexpr double        kDevBracketMinSpan   = 1.0e-6;
inline constexpr double        kDevBoronProbe       = 50.0;
inline constexpr double        kDevRodProbe         = 0.25;

/// Anderson depth for the Xe fixed point, mirroring Driver.h:732
/// (`XE_ANDERSON_DEPTH = 2`).  Sizes the per-slot AA history block:
/// (6 + 2*DEPTH) triples of (I, Xe, Xem) over n_fuel nodes.
inline constexpr int kDevXeAndersonDepth = 2;

/// Triples held by the AA history: x, f, g, f_prev, g_prev, cand, plus
/// df[DEPTH] and dg[DEPTH] (Driver.h:999-1007).
inline constexpr int kDevXeAndersonTriples = 6 + 2 * kDevXeAndersonDepth;

// ---------------------------------------------------------------------------
// Refill reset -- the single definition of "a slot has no tenant".
//
// Sec 8.2 immediate refill hands a Done slot straight to the next input.  Every
// byte of all four structs is reset here; the Task 20 audit kernel re-reads
// them right after and any non-reset byte is `stale_tenant_errors++` and fatal.
// The helpers are the post-condition that audit checks, so they live with the
// structs and not with the scheduler.
// ---------------------------------------------------------------------------

/// Zero every byte, PADDING INCLUDED.  Each reset below starts here.
///
/// Resetting only the declared fields is not enough and the difference is not
/// academic: alignas(128) gives DeviceSlotState 120 bytes of tail padding, and
/// every uint32-before-double pair adds four more.  Those bytes still belong to
/// the previous tenant, so a byte-level audit -- which is exactly what the
/// Task 20 `k_audit_tenant_reset` does, and what any snapshot hash or
/// determinism comparison does -- sees a recycled slot that is not clean.
/// Caught by test/gpu_phase_compaction.cpp's two-fill reset postcondition.
///
/// Written as a byte loop rather than memset so the header needs no <cstring>
/// and compiles identically under nvcc; these run once per refill, never on a
/// hot path.
RASBERY_GPU_HD inline void deviceZeroBytes(void* p, unsigned int bytes) {
    unsigned char* b = static_cast<unsigned char*>(p);
    for (unsigned int i = 0; i < bytes; ++i) b[i] = 0u;
}

RASBERY_GPU_HD inline void deviceSlotPhaseReset(DeviceSlotPhase& p, std::uint32_t next_epoch) {
    deviceZeroBytes(&p, static_cast<unsigned int>(sizeof(p)));
    p.phase        = static_cast<std::uint8_t>(DevicePhase::Empty);
    p.queued_phase = static_cast<std::uint8_t>(DevicePhase::Empty);
    p.escape       = static_cast<std::uint8_t>(DeviceEscape::None);
    p.flags        = 0u;
    p.state_epoch  = next_epoch;
    // A refilled slot must not look queued: an epoch that cannot equal
    // state_epoch is the cheapest way to say "no live queue entry".
    p.queued_epoch = next_epoch - 1u;
    p.phase_age    = 0u;
    p.input_id     = 0u;
    p.job_id       = 0u;
    p.error_code   = 0u;
    p.reserved     = 0u;
}

RASBERY_GPU_HD inline void deviceSlotStateReset(DeviceSlotState& s) {
    deviceZeroBytes(&s, static_cast<unsigned int>(sizeof(s)));
    s.schedule_index     = 0u;
    s.statepoint         = 0u;
    s.substep            = 1u;
    s.substep_index      = 0u;
    s.solve_call_kind    = 0u;
    s.outer_in_segment   = 0u;
    s.total_outer        = 0u;
    s.flux_stall         = 0u;
    s.stall_events       = 0u;
    s.stall_sample_taken = 0u;
    s.cmfd_sweeps        = 0u;
    s.bicg_iterations    = 0u;
    s.clean_iters        = 0u;
    s.xe_iterations      = 0u;
    s.xe_total           = 0u;
    s.xe_interim_count   = 0u;
    s.xe_no_progress     = 0u;
    s.xe_streak          = 0u;
    s.xe_cap_charged     = 0u;
    s.xe_cascade_index   = 0u;
    s.xe_aa_ncol         = 0u;
    s.xe_aa_have_prev    = 0u;
    s.xe_relax           = 1.0;
    s.prev_xe_change     = 0.0;
    s.xe_residual        = 0.0;
    s.th_iterations      = 0u;
    s.search_iterations  = 0u;
    s.th_residual        = 0.0;
    s.search_residual    = 0.0;
    s.eigv                = 1.0;
    s.previous_eigv       = 1.0;
    s.eigv_before_segment = 1.0;
    s.flux_l2             = 0.0;
    s.keff_tolerance      = kDevEigvTol;
    s.flux_tolerance      = kDevEigvTol;
    s.micx_generation            = 0u;
    s.ref_generation             = 0u;
    s.hoststate_generation       = 0u;
    s.nodal_constant_generation  = 0u;
    s.geometry_generation        = 0u;
    s.material_generation        = 0u;
    s.operator_generation        = 0u;
    s.flux_generation            = 0u;
    s.current_generation         = 0u;
    s.dhat_generation            = 0u;
    s.isotope_generation         = 0u;
    s.th_generation              = 0u;
}

/// SearchMemory is carried across statepoints WITHIN a deck and must never
/// survive into a different deck's tenancy: this resets it with everything else.
RASBERY_GPU_HD inline void deviceSearchStateReset(DeviceSearchState& q) {
    deviceZeroBytes(&q, static_cast<unsigned int>(sizeof(q)));
    q.has_boron_secant  = 0u;
    q.boron_secant_dkdx = 0.0;
    q.has_rod_secant    = 0u;
    q.rod_secant_x      = 1.0; // Scheduler.h:59 SearchMemory default
    q.rod_secant_dkdx   = 0.0;

    q.initialized               = 0u;
    q.seeded_from_previous_step = 0u;
    q.has_prev                  = 0u;
    q.has_bracket               = 0u;
    q.has_best                  = 0u;
    q.slope_frozen              = 0u;
    q.iteration                 = 0u;
    q.current_x                 = 0.0;
    q.prev_x                    = 0.0;
    q.prev_eigv                 = 0.0;
    q.frozen_slope              = 0.0;
    q.best_x                    = 0.0;
    q.best_residual             = 0.0;
    q.bracket_lo_x              = 0.0;
    q.bracket_lo_residual       = 0.0;
    q.bracket_hi_x              = 0.0;
    q.bracket_hi_residual       = 0.0;

    q.exit_status = static_cast<std::uint32_t>(DeviceSearchExit::None);
    q.exit_dk     = 0.0;
    q.exit_tol    = 0.0;
    q.stall_count = 0u;
}

/// Defaults only: the import phase overwrites every field from the new deck.
/// Reset still runs first so a field the importer forgets is a Scheduler.h
/// default rather than the previous tenant's value.
RASBERY_GPU_HD inline void deviceScheduleParamsReset(DeviceScheduleParams& d) {
    deviceZeroBytes(&d, static_cast<unsigned int>(sizeof(d)));
    d.search_type          = static_cast<std::uint32_t>(DeviceSearchType::Keff);
    d.schedule_type        = static_cast<std::uint32_t>(DeviceScheduleType::Standard);
    d.th_mode              = static_cast<std::uint32_t>(DeviceThMode::Steady);
    d.max_outer_iter       = kDevMaxEigenIter;
    d.max_search_iter      = kDevMaxSearchIter;
    d.max_th_iter          = kDevMaxThIter;
    d.tolerance_keff       = kDevEigvTol;
    d.tolerance_search     = kDevCritSearchTol;
    d.rodcrit_search_floor = 0.0;
    d.rodcrit_search_cap   = kDevRodCritSearchTol;
    d.tolerance_th         = kDevThTol;
    d.target_keff          = 1.0;
    d.tolerance_tmod       = kDevTempSearchTol;
    d.tolerance_tfuel      = kDevTempSearchTol;
    d.search_boron_ppm     = 0.0;
    d.tolerance_boron      = kDevBoronSearchTol;
    d.tolerance_rodsearch  = kDevRodSearchTol;
    d.search_relaxation    = kDevSearchRelax;
    d.search_low           = kDevSearchLow;
    d.search_hi            = kDevSearchHigh;
    d.slope_freeze_thres   = kDevSlopeFreezeThres;
    d.min_secant_denom     = kDevMinSecantDenom;
    d.bracket_min_span     = kDevBracketMinSpan;
    d.search_boron_probe   = kDevBoronProbe;
    d.search_rod_probe     = kDevRodProbe;

    d.time                 = 0.0;
    d.burnup               = 0.0;
    d.rate                 = 100.0;
    d.rated_power          = 4200.0;
    d.actual_power         = 4200.0;
    d.step_dt              = 0.0;
    d.substeps             = 1u;
    d.xenon_transient      = 0u;
    d.use_burnup_time      = 0u;
    d.bppm0                = 500.0;
    d.tful0                = 900.0;
    d.tmod0                = 580.0;
    d.dmod0                = 0.71;
    d.pressure             = 15.0;
    d.inlet_temp           = 580.0;
    d.outlet_temp          = 600.0;
    d.mass_flow_rate       = 1000.0;
    d.fuel_temp_rise_scale = 1.0;
    d.delta_tful           = 0.0;
    d.delta_tmod           = 0.0;
    d.delta_dmod           = 0.0;
    d.delta_bppm           = 0.0;
    d.delta_xe             = 0.0;
    d.delta_sm             = 0.0;
}

// ---------------------------------------------------------------------------
// Task 1 Step 4 layout contract (Rev.7.1).
// ---------------------------------------------------------------------------

static_assert(sizeof(DeviceSlotPhase) == 32, "Level-1 reads 64 slots x 32 B = 2 KiB; it must fit L1");
static_assert(alignof(DeviceSlotPhase) == 32);
static_assert(std::is_trivially_copyable_v<DeviceSlotPhase>);

static_assert(alignof(DeviceSlotState) == 128);
static_assert(sizeof(DeviceSlotState) % 128 == 0);
static_assert(std::is_trivially_copyable_v<DeviceSlotState>);

static_assert(std::is_trivially_copyable_v<DeviceSearchState>);
static_assert(std::is_trivially_copyable_v<DeviceScheduleParams>);

static_assert(std::is_standard_layout_v<DeviceSlotPhase>);
static_assert(std::is_standard_layout_v<DeviceSlotState>);
static_assert(std::is_standard_layout_v<DeviceSearchState>);
static_assert(std::is_standard_layout_v<DeviceScheduleParams>);

} // namespace rasbery::gpu
