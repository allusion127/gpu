#pragma once
#include "CudaCramBackend.h"
#include "CudaThBackend.h"
#include "CudaXsReconBackend.h"
#include "FlatXsKernel.h"
// For xsrecon::NISO and xsrecon::BatchView: the device Xe entry points
// below take a view by reference and size the depTrans rows from the
// kernel's own constant, rather than restating 39 here.
#include "XsReconKernel.h"
#include "Geometry.h"
#include "Model.h"
// XSArraySet, the library-derived flatten, and the process-wide parse cache.
#include "XsLibrary.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rasbery {

// XSArraySet now lives in XsLibrary.h: the library flatten and the node-wise
// live blocks are the same shape, and the flatten had to move to a header that
// carries no Geometry.

struct DepletionWorkspace;

// Rod group definition

struct RodGroup {
    int                 ctype = 0;
    std::vector<int>    ctype_segments;
    std::vector<double> length_segments;
    std::vector<int>    xy_nodes;
    double              insertion = 0.0;
};

// Flat XS update control. Empty nodes means "all nodes".
struct XSUpdateOptions {
    std::vector<int> nodes;
};

struct AxialTransverseLeakageView {
    const double* c0;
    const double* c1;
    const double* c2;

    [[nodiscard]] double value(int lk, int ig, int ng, double zeta) const {
        const size_t idx = static_cast<size_t>((lk * NDIRMAX + ZDIR) * ng + ig);
        const double p2  = 0.5 * (3.0 * zeta * zeta - 1.0);
        return c0[idx] + c1[idx] * zeta + c2[idx] * p2;
    }
};

// XSSet

class XSSet {
private:
    Geometry& _g;

    // T/H property tables
    milk::Table _mod_cp_table;
    milk::Table _mod_rho_table;
    milk::Table _mod_h_table;
    milk::Table _mod_t_table;
    milk::Table _tf_table;

    void LoadTHTables();

    // Chiffon database.
    //
    // The parse and its flatten are shared, immutable, and owned by the
    // process-wide cache in XsLibrary.h -- every field that used to live here
    // as `_models`, `_lib_*`, `_refr_*` or `_brch_*` is now `_lib->...`.  The
    // pointer is null until Initialize() runs, which is the same window in
    // which the old members were empty.
    std::shared_ptr<const XsLibrary> _lib;

    // Node-wise data
    std::vector<int>                 _node_refr_lo;    // node -> lower reference depletion point id
    std::vector<int>                 _node_refr_hi;    // node -> upper reference depletion point id
    std::vector<std::vector<int>>    _node_delta_lo;   // branch, node -> lower burnup delta id
    std::vector<std::vector<int>>    _node_delta_hi;   // branch, node -> upper burnup delta id
    std::vector<std::vector<double>> _node_delta_frac; // branch, node -> burnup interpolation fraction

    // Gd effective-number-density lookup axis (MASTER/PROLOG `BP01 MICRO` equivalent).
    // When enabled, the lumped-Gd (virtual isotope "640000") microscopic XS is interpolated on
    // the effective Gd number density N_eff = sum_i w_i N_i instead of on burnup.
    bool                _gd_neff_axis = false; // RASBERY_GD_AXIS=neff enables it
    std::vector<int>    _node_gd_lo;           // node -> lower reference point flat id on the N_eff axis
    std::vector<int>    _node_gd_hi;           // node -> upper reference point flat id on the N_eff axis
    std::vector<double> _node_gd_frac;         // node -> N_eff interpolation fraction

    // Depletion-history blend. A fuel may declare a rodded-depletion twin; the
    // node then reads a weighted mix of the two libraries, the weight being how
    // far its own U235/Pu239 balance has been driven toward the rodded
    // trajectory. Everything here is inert (partner < 0, weight 0) unless the
    // library carries the twin, so single-library runs are untouched.
    // (the model -> partner map is library data: _lib->lib_history_partner)
    std::vector<double>              _node_hw;             // node -> blend weight in [0,1]
    std::vector<int>                 _node_refr_lo_p;      // node -> partner lower reference id
    std::vector<int>                 _node_refr_hi_p;      // node -> partner upper reference id
    std::vector<std::vector<int>>    _node_delta_lo_p;     // branch, node -> partner lower delta id
    std::vector<std::vector<int>>    _node_delta_hi_p;     // branch, node -> partner upper delta id
    std::vector<std::vector<double>> _node_delta_frac_p;   // branch, node -> partner burnup fraction

    std::vector<size_t> _asmb;          // assembly index of each node
    std::vector<size_t> _comp;          // composition index of each node
    std::vector<int>    _burn;          // burnup key of each node
    std::vector<int>    _ctyp;          // control rod type of each node
    std::vector<char>   _external_iden; // to preserve restart/shuffle isotope densities after InitXS

    // Live node-wise SoA cross-section storage.
    XSArraySet           _xs;   // current macroscopic XS after isotope reconstruction
    XSArraySet           _micx; // current node microscopic XS after branch/rod updates
    XSArraySet           _lmpx; // current node lumped XS after branch/rod updates
    milk::Vector<double> _iden; // current node isotope densities

    /// Shared setup for every device Xe entry point: dimension check, backend
    /// creation, host page-locking, the depTrans rows and the pointer view.
    /// The two dep arrays are the CALLER's locals because the view holds their
    /// addresses and the backend reads them after this returns.
    bool PrepareXeDeviceCall(double power, double relax, xsrecon::BatchView& view,
                             std::array<double, xsrecon::NISO>& dep_i135,
                             std::array<double, xsrecon::NISO>& dep_xe135);

    // Equilibrium-Xe device backend (RASBERY_GPU_XSRECON, default off).  The
    // generation counter advances whenever a host path rebuilds _micx/_lmpx
    // (UpdateFlatXS / Update), so the backend re-uploads its resident copy
    // only when the host actually mutated it; between rebuilds a whole
    // Xe<->flux cascade reuses the device copy.  Rod cusping blends _xs only,
    // never _micx/_lmpx, so it needs no bump (and the Xe reconstruct
    // overwrites the blended _xs of fuel nodes identically on both paths).
    std::unique_ptr<XsReconBackend> _xsrecon_backend;
    std::vector<int>                _fuel_nodes;      // built once; geometry-fixed
    unsigned long long              _micx_generation = 1;
    bool                            _xsrecon_pinned  = false;

    // WP15.  "The device holds micx/lmpx values this host copy does not."
    //
    // ATOMIC, and mutable, for one reason each.  Atomic because the readers are
    // per-node functions called from inside OpenMP loops (ReconstructNode,
    // UpdateUnroddedNodeXS) and the payer flips the flag while the other
    // threads are still testing it -- a plain bool there is a data race whether
    // or not the machine notices.  Mutable because EnsureMicxHost is const:
    // every consumer of a cross section is.
    //
    // TWO FLAGS, NOT ONE, because the scatter block is 10.5 MB of the 59.5 and
    // the depletion / Xe / CRAM readers never touch it.  See MicxNeed.
    mutable std::atomic<bool> _micx_device_scalars{false};
    mutable std::atomic<bool> _micx_device_scatter{false};

    // CRAM depletion device backend (RASBERY_GPU_CRAM, default off).  GA
    // evaluator plan Task 16.  Created on first use like _xsrecon_backend, and
    // owned here for the same reason: one XSSet, one Driver, one batch slot.
    //
    // UNLIKE THE PPR ARM, THIS ONE MOVES THE TRAJECTORY.  Its output is the
    // isotope inventory the next statepoint's XS reconstruction reads, so
    // RASBERY_GPU_CRAM is listed in trajectory::kArmEnv (Driver.h) and an A/B
    // that changes it is an arm change, not a timing change.
    std::unique_ptr<CramBackend> _cram_backend;
    /// Predictor/corrector calls that ran on the host because the device
    /// declined or failed.  With the arm off this is every one of them.
    unsigned long long _cram_host_fallbacks = 0;
    /// Token published by the last successful device predictor; the corrector
    /// presents it so a mid-statepoint fallback cannot pair a device corrector
    /// with a host predictor's BOS state.  0 = no device predictor yet.
    unsigned long long _cram_bos_token = 0;
    /// Node-invariant burnup-key normalisation, built once (geometry- and
    /// composition-fixed) with the host's own expression so the device reads a
    /// number the host would have produced, not a re-derivation of it.
    std::vector<double> _cram_dfac;
    std::vector<double> _cram_vol;
    unsigned long long  _cram_lib_generation = 0;

    // WP22.  The T/H device arm (RASBERY_GPU_TH, default off).  Created on
    // first use like _cram_backend, and owned here for the same reason: one
    // XSSet, one Driver, one batch slot.
    //
    // THIS ONE MOVES THE TRAJECTORY, on the same footing RASBERY_GPU_CRAM does
    // and more directly: its outputs are tful/tmod/dmod, which the very next
    // UpdateFlatXS reconstructs every macroscopic cross section from.  So
    // RASBERY_GPU_TH is listed in trajectory::kArmEnv (Driver.h), and an A/B
    // that changes it is an arm change and not a timing change.
    std::unique_ptr<ThBackend> _th_backend;
    /// T/H updates that ran on the host because the device declined or failed.
    /// With the arm off this is every one of them.
    unsigned long long _th_host_fallbacks = 0;
    /// The hmesh X/Y columns, gathered once.  Geometry stores hmesh node-major
    /// and direction-minor (`_hmesh[lk * NDIRMAX + dir]`), so neither column is
    /// contiguous and neither can be uploaded as-is.  Built on first use and
    /// never rebuilt: the mesh does not move inside a run, which is what
    /// `_th_geom_generation` asserts rather than assumes.
    std::vector<double> _th_hmesh_x;
    std::vector<double> _th_hmesh_y;
    unsigned long long  _th_geom_generation = 0;

    /// The device T/H arm.  Returns false -- WITHOUT having touched tful, tmod,
    /// dmod or `delta_dop` -- when the arm is off, the device refused, or the
    /// deck is one the kernel does not serve, and the caller must then run the
    /// unchanged host body.
    bool TryUpdateTHGpu(double power_rate, double& delta_dop);

    /// True and the views filled when the device arm can run this deck at all.
    bool PrepareCramLib(cram::LibView& lib);
    bool DepleteGpu(double dt, double power, bool xe_transient);
    /// WP15.1: fill a cram view's eleven-slot DEVICE micro-XS table from the
    /// flat-XS backend's resident block, plus the ordering event.  Returns false
    /// (and leaves the table all-null) when the block is not resident, its
    /// generation does not match `_micx_generation`, or no event is available --
    /// in which case the caller must materialise the host copy and let the
    /// backend upload it.  All eleven or none: see the definition.
    /// WP20.1: `const void**` and an out `elem_bytes`, because the resident
    /// block's element width is now the FP32 arm's decision and the consumer
    /// must be told rather than assume.
    bool FillCramMicDevice(const void** dev, void*& ready, int& elem_bytes);
    bool CorrectorStepGpu(double dt, double power, bool xe_transient,
                          bool density_average, bool xe_equilibrium_fix,
                          int substeps);

    // Advances whenever HOST code writes _xs or _iden outside the device
    // backends' own downloads (CPU reference loops, cusping blends, depletion,
    // Update, the reference rebuild).  While it holds still, the device copies
    // of _xs/_iden are bit-identical to the host's (every solve downloads what
    // it wrote), so the backends skip their per-call re-uploads.  Missing a
    // bump site corrupts physics silently -- the full-deck debug-hash A/B is
    // the gate for any new host-side writer.
    unsigned long long _hoststate_generation = 1;

    // ---------------------------------------------------------------------
    // Rev.7.1 Task 9/W3: THE MACRO-XS WRITE GENERATION
    // ---------------------------------------------------------------------
    //
    // "Did the HOST BYTES of _xs.xsrf / _xs.xsdf move" -- and nothing else.
    //
    // WHY IT IS NOT _hoststate_generation.  That counter means "the device
    // MIRROR of _xs is stale", which is a different question and deliberately
    // NOT bumped when a device arm rewrites the host array (UpdateFlatXS's GPU
    // branch bumps only `if (!gpu_ok || !rodded.empty())`; UpdateEquilibriumXenon's
    // returns before its bump), because after such a write host and mirror
    // agree.  tools/test_device_outer_exactness_contract.py invariant 5 is the
    // record of what that cost when a gate believed otherwise.
    //
    // WHY IT CANNOT MISS A WRITER.  It is bumped AT THE WRITE, inside the four
    // functions that assign _xs.xsrf / _xs.xsdf (Reconstruct, ReconstructNode,
    // UpdateUnroddedNodeXS, ApplyRodCuspingStencil), not at their callers.  A
    // caller's policy -- which arm ran, whether a mirror agrees -- cannot
    // therefore change the answer, and a new writer that does not bump is
    // caught by tools/test_nodal_constant_host_gate_contract.py, which greps
    // for exactly those assignments.
    //
    // ATOMIC because ReconstructNode and UpdateUnroddedNodeXS are called from
    // inside OpenMP loops over nodes.  Relaxed is the right order: nothing
    // BRANCHES on this value concurrently with a write -- Nodal reads it between
    // drives, on one thread -- and a redundant bump is conservative (it can only
    // make a reader recompute something that had not moved).
    std::atomic<unsigned long long> _macroxs_generation{1};

    // Flat-XS device arm (RASBERY_GPU_FLATXS, default off).  The reference
    // blocks (_ref_lmpx/_ref_micx) are device-resident and re-upload only
    // when PrecomputeBranchCoefficients rebuilds them (_ref_generation); the
    // library coefficient tables upload once per process (content-hashed,
    // shared across instances).  The stream scratch below carries the
    // host-resolved (did, x, scale) applications of one UpdateFlatXS call.
    unsigned long long  _ref_generation = 1;
    bool                _flatxs_pinned  = false;
    std::vector<int>    _flatxs_stream_did;
    std::vector<double> _flatxs_stream_x;
    std::vector<double> _flatxs_stream_scale;
    std::vector<int>    _flatxs_off;
    std::vector<int>    _flatxs_cnt;
    std::vector<int>    _flatxs_unrodded;
    std::vector<int>    _flatxs_rodded;

    // Flat branch delta storage
    // Pre-flattened reference XS (burnup-interpolated, SoA layout)
    XSArraySet           _ref_lmpx;  // [ig*nxyz + l]
    XSArraySet           _ref_micx;  // [(iso*ng+ig)*nxyz + l]
    milk::Vector<double> _ref_iden;  // [niso*nxyz] (non-H/B/O isotopes)
    std::vector<double>  _ref_wvfr;  // unrodded AD_WVFR per node
    std::vector<double>  _node_wvfr; // live AD_WVFR per node
    // Page-exclusive storage (HostPinRegistry.h): chifData() is page-locked by
    // NodalArena::pinSlot, and a plain std::vector shares its boundary pages
    // with whatever the allocator put next to it, which cudaHostRegister
    // refuses.
    PageExclusiveVector<double> _ref_chix; // burnup-interpolated fission spectrum [ig*nxyz + l]
    double               _boron_dmod_average = 0.0;
    double               _current_power_rate = 1.0;

    bool _simd_ready = false;

    // Beginning-of-step state for predictor-corrector depletion
    XSArraySet           _xs_bos;
    XSArraySet           _micx_bos;
    milk::Vector<double> _iden_bos;
    std::vector<int>     _burn_bos;
    milk::Vector<double> _flux_bos;

    std::vector<double> _node_power_scratch;
    std::vector<double> _cum_bot_scratch;
    std::vector<double> _th_tful_old_scratch;
    std::vector<double> _old_rod_fraction_scratch;
    std::vector<int>    _old_rod_ctyp_scratch;
    std::vector<int>    _dirty_nodes_scratch;
    std::vector<int>    _rod_cusping_nodes_scratch;
    int                 _axial_rod_division             = 10;
    double              _rod_cusping_relaxation         = 1.0;  // 1.0 = full PARCS flux-weighted cusp; <1 dilutes (see ApplyRodCusping)
    double              _th_relaxation                  = 0.85; // T/H feedback under-relaxation: damps temperature oscillation, ~30% fewer outer iters (RASBERY_TH_RELAX overrides)
    std::vector<int>    _fine_rod_type;
    std::vector<double> _fine_rod_frac; // rodded fraction of each fine cell (0..1); <1 only at the tip cell
    // PROBE: cumulative Pu239 gain, split by how much of it accrued while the
    // node was rodded. The ratio is a history weight; the instantaneous rod
    // fraction is a spectrum weight, and the two differ only in transitions.
    std::vector<double> _pu_prev;
    std::vector<double> _pu_gain_tot;
    std::vector<double> _pu_gain_rod;
    std::vector<double> _fine_rod_thermal_fluence;
    std::vector<double> _fine_rod_thermal_fluence_bos;
    std::vector<int>    _rod_node_segment_offset;
    std::vector<int>    _rod_node_segment_ctype;
    std::vector<char>   _node_uses_rod; // cached UsesRodXS answer, rebuilt in Initialize/SetRod

    // Cusping resets the touched nodes to their base (pre-cusping) flat XS every outer iteration so
    // the relaxation blend is stable. That base is a deterministic function of (bppm, tful, dmod,
    // burn, rod fraction, ctype, rod fluence) and does NOT depend on the flux, so it is reconstructed
    // once per input change and restored by copy thereafter (skips the dominant branch-delta cost).
    struct CuspingBaseSnapshot {
        std::vector<double> scalar;      // N_XS_SCALAR * ng
        std::vector<double> scatter;     // ng * ng
        double              sig[7] = {}; // burn, bppm, tful, dmod, rod_fraction, ctyp, fluence
        bool                valid  = false;
    };
    std::vector<CuspingBaseSnapshot> _cusping_base_snapshot; // indexed by node l
    void                             ResetCuspingNodesToBase(const std::vector<int>& nodes);
    std::vector<double>              _rod_node_segment_fraction;

    std::map<std::string, RodGroup> _rod_groups;

    // Rod profile matrix: row = rod group, col = step position
    // Ordered by _rod_group_order (same order as matrix rows)
    std::vector<std::string> _rod_group_order; // group names in row order
    milk::Matrix<double>     _rod_profile;     // (ngroups, ncols)
    int                      _rod_ncols = 0;

    void Deplete(double dt, double power, bool xe_transient);
    void DepleteNode(DepletionWorkspace& ws, size_t l, const double* abs_flux, size_t ngrp, double dt, bool xe_transient);

    void BuildTransitionMatrix(const std::vector<double>& condensed, double sumflux,
                               milk::Matrix<double>& mat) const;

    [[nodiscard]] double NormFactor(double power, const XSArraySet& xs_arr, const double* flux) const;

    void               unpackXS(const Chiffon::CrossSection& xs, size_t l, size_t ngrp, size_t nxyz, size_t niso_count);
    // The four Flatten* helpers moved to XSSet.cpp as free functions over
    // XsLibrary: they never read anything but (models, ng, niso), which is
    // what made the parse shareable in the first place.
    void               RebuildUsesRodCache();
    [[nodiscard]] bool UsesRodXS(int l) const;
    /// @brief One fitted spectral-history surface to apply.
    struct DeltaApplication {
        int    did;
        double x;
        double scale;
        int    iso;
    };
    /// @brief Ratio of resonance to 1/v thermal absorption on the node's
    /// pre-correction cross sections; a stand-in for the intra-group spectrum.
    /// @param micWork when non-null, the node's XSAF microscopic workspace
    /// block laid out [iso*ng+ig]; supplies the base+branch state that is not
    /// yet scattered back into `_micx`. Falls back to `_micx` when null.
    [[nodiscard]] double NodeSpectralIndex(
        int l, const double* micWork = nullptr) const;

    /// @brief Share of the node flux in the fast (ig=0) or thermal (last) group.
    [[nodiscard]] double NodeFluxShare(int l, bool thermal) const;

    /// @brief Collect the spectral-history delta applications for one node.
    /// @param modelIndex Library to resolve against; SIZE_MAX uses the node's own.
    /// @param weight Extra factor on every returned scale, for the history blend.
    void ResolveSpectralHistoryDeltas(
        int l, std::vector<DeltaApplication>& out,
        const double* micWork = nullptr,
        size_t modelIndex = static_cast<size_t>(-1), double weight = 1.0) const;

    // Flat-XS device arm helpers (see the _flatxs_* scratch above).
    // BuildFlatXsStream resolves every applyDelta call of the unrodded nodes
    // into the flat stream scratch, in exactly the CPU loop's order;
    // TryUpdateFlatXSGpu ships it to the backend and downloads the results.
    std::vector<std::vector<DeltaApplication>> _flatxs_node_apps;
    std::vector<flatxs::DeltaMeta>             _flatxs_deltas; // flattened _lib_deltas
    flatxs::FlatXsView MakeFlatXsHostView();
    FlatXsLibShape     MakeFlatXsLibShape() const;
    void               BuildFlatXsStream(const std::vector<int>& nodes);
    bool               TryUpdateFlatXSGpu(const std::vector<int>& unrodded,
                                          bool any_rodded);
    /// @brief Blend weight toward the rodded-depletion twin, 0 when there is none.
    [[nodiscard]] double HistoryBlendWeight(int l) const {
        return _node_hw.empty() ? 0.0 : _node_hw[static_cast<size_t>(l)];
    }
    /// @brief Interpolate one isotope density on a model's ctype reference trajectory.
    /// @param mi Model index. @param ctype Rod state (0 = out). @param burn Burnup key.
    /// @param iso Isotope index. @return Density [1/barn-cm], 0 when unavailable.
    [[nodiscard]] double ReferenceIden(size_t mi, int ctype, int burn, size_t iso) const;
    /// @brief Fill node `l`'s partner brackets and depletion-history blend weight.
    ///
    /// The weight is how far the node's own U235-vs-Pu239 balance has been driven
    /// from the unrodded reference toward the partner's rod-in reference, clamped
    /// to [0,1] so a node outside the two trajectories takes one of them whole.
    void BuildHistoryBlend(int l, size_t mi);
    void UpdateUnroddedNodeXS(int l);
    void ApplyBranchDeltaIdToNode(int l, int did, double x, double scale);
    /// @brief Accumulate the few-group MACRO delta-sigma [barn] of fitted surface `did` at scaled
    /// indicator coordinate `x` into `scalar`, laid out [N_XS_SCALAR][ng] (xt-major).
    /// Microscopic coefficients are folded through node densities.
    void                 AccumulateDeltaMacro(int l, int did, double x, double scale,
                                              std::vector<double>& scalar) const;
    void                 ApplySpectralHistoryToNode(int l);
    void                 FillRodNodeXS(int l);
    void                 RefreshLightIsotopes(int l);
    void                 Reconstruct();
    void                 ReconstructNode(size_t l);
    [[nodiscard]] int    RodCTypeAtDistance(const RodGroup& group, double distance_from_tip) const;
    [[nodiscard]] double RodTotalLength(const RodGroup& group) const;
    void                 FillCuspingMacroXS(int l, int ctype, double fluence,
                                            std::vector<double>& scalar,
                                            std::vector<double>& scatter) const;
    void                 ApplyRodCuspingStencil(int tip_l, double reigv,
                                                const AxialTransverseLeakageView& leakage,
                                                std::vector<int>&                 touched_nodes);
    void                 RebuildFineRodOccupancy();
    void                 DepleteRodMaterials(double dt, double power, bool corrected_flux);
    void                 AccumulatePuHistory();
    [[nodiscard]] double RoddedPuFraction(int l) const;
    [[nodiscard]] double RodBlendWeight(int l) const;

    // T/H steady-state solve (moved from THSolver)
    void SolveTH(const double* node_power, const int* burnup, double power_rate);

public:
    explicit XSSet(Geometry& g) noexcept;

    ~XSSet();

    XSSet(const XSSet&)            = delete;
    XSSet& operator=(const XSSet&) = delete;
    XSSet(XSSet&&)                 = delete;
    XSSet& operator=(XSSet&&)      = delete;

    void Initialize(const std::string& xs_path);
    void Update();

    /// @brief Few-group macroscopic contribution of one spectral-history term.
    struct TermContribution {
        int                 iso;
        std::vector<double> scalar;
    };
    /// @brief Per-term spectral-history contributions applied at a node.
    void ResolveTermContributions(int l, std::vector<TermContribution>& out) const;
    void PrecomputeBranchCoefficients();
    void UpdateFlatXS(const XSUpdateOptions& options = XSUpdateOptions{});
    void UpdateBurnup(double dt, double power);

    // T/H property queries
    [[nodiscard]] double GetCpmod(double temperature_k, double pressure_mpa) const { return _mod_cp_table.Get(pressure_mpa, temperature_k); }
    [[nodiscard]] double GetHmod(double temperature_k, double pressure_mpa) const { return _mod_h_table.Get(pressure_mpa, temperature_k); }
    [[nodiscard]] double GetTmod(double enthalpy_kJ, double pressure_mpa) const { return _mod_t_table.Get(pressure_mpa, enthalpy_kJ); }
    [[nodiscard]] double GetDmod(double enthalpy_kJ, double pressure_mpa) const { return _mod_rho_table.Get(pressure_mpa, enthalpy_kJ); }
    [[nodiscard]] double GetTfuel(double burnup, double LPD) const {
        // tf.csv starts at 50 W/cm.  Table::Get clamps below that knot,
        // which leaves a non-zero ~80 K fuel rise as local power tends to
        // zero and distorts the axial power shape.  The conduction solution
        // is linear to first order at low heat rate, so continue the first
        // tabulated point to the physical origin.
        constexpr double first_lpd = 50.0;
        const double     safe_lpd  = std::max(0.0, LPD);
        const double rise = _tf_table.Get(std::max(first_lpd, safe_lpd), burnup);
        return safe_lpd < first_lpd ? rise * safe_lpd / first_lpd : rise;
    }

    [[nodiscard]] double pressure() const { return _g.pressure(); }

    void InitXS(double bppm, double tful, double tmod, double pres,
                double dmod = 0.0, bool overwrite_th = true);

    void SetPowerRate(double power_rate);
    // Applies T/H feedback; returns the max relative Doppler (fuel) temperature change delta_Dop
    // used as the T/H convergence metric.
    double UpdateTH(double power_rate);
    void   UpdateDerivative(double delta_bppm, double delta_tful, double delta_tmod, double delta_dmod);
    void   ResetFluxAndCurrents(double flux_value = 1.0);
    void   NormalizeFluxSign();

    // Recompute the BOC/equilibrium I-135/Xe-135 inventory from the current
    // power-normalized flux and rebuild macroscopic cross sections.
    //
    // `relax` damps the APPLIED update: x <- x + relax*(F(x) - x).  That has
    // exactly the fixed points of x = F(x), so the converged inventory is
    // unchanged; only the path to it is damped.  The undamped Xe<->flux map
    // limit-cycles on some cores (see the oscillation detector in Driver.h).
    // relax == 1.0 is the original update, bit for bit.
    //
    // Returns the maximum RAW relative Xe-135 density change over fuel nodes --
    // |F(x) - x| / |F(x)|, measured before the damping is applied.  Reporting
    // the damped step instead would let a small relax push the change under the
    // convergence tolerance on its own and declare a convergence that never
    // happened.
    double UpdateEquilibriumXenon(double power, double relax = 1.0);

    /// Device arm of UpdateEquilibriumXenon (XsReconKernel.h), attempted only
    /// when RASBERY_GPU_XSRECON is set.  Returns false on any unavailability,
    /// in which case the caller runs the unchanged CPU loop.
    bool TryUpdateEquilibriumXenonGpu(double power, double relax, double& max_change);

    // --- Rev.7.1 Task 13: the SPLIT device Xe arm (RASBERY_GPU_XE) ---------
    //
    // TryUpdateEquilibriumXenonGpu above is the FUSED device step and it is the
    // whole of UpdateEquilibriumXenon.  These take the same node body apart at
    // the SAME seam the host API is split at -- Snapshot / Evaluate / Commit --
    // so the safeguarded Anderson arm can run on the device without its history
    // (ten triples over the fuel nodes) crossing the bus every step.
    //
    // THE HISTORY LIVES IN THE BACKEND, WHICH IS PER XSSet AND THEREFORE PER
    // DRIVER.  A --batch-mode run gives each deck its own Driver, its own XSSet
    // and its own backend, so the histories are independent by construction and
    // there is nothing process-wide to share by accident.  That is deliberate:
    // the batch bug this tree just fixed was a process-wide slot-0 buffer that
    // every Driver adopted.
    //
    // Every one returns false when the arm is unavailable, having written
    // nothing, and the caller then runs the untouched host path.

    /// Evaluate F(x) on the device without applying it, and report the RAW
    /// (pre-damping) maximum relative Xe-135 change -- the same number
    /// UpdateEquilibriumXenon and EvaluateEquilibriumXenon return.
    bool XeGpuEvaluate(double power, double& picard);

    /// Anderson window bookkeeping, one primitive per line of the host arm's
    /// history roll, so the two read the same way side by side.
    bool XeGpuRotateHistory();
    bool XeGpuRecordColumn(int col);
    bool XeGpuSaveEvaluation();

    /// The six inner products of the depth-2 normal equations, in
    /// xe::XeDotSlot order.  THIS is the N1 step: a fixed partition, not the
    /// host's single serial fold (see XeKernel.h).
    bool XeGpuDots(int ncol, double* out_six);

    /// Build the candidate and measure the two things the safeguards need.
    bool XeGpuCandidate(const double* gamma, int ncol, double& step, bool& physics_ok);

    /// Commit the accepted candidate: writes the three Xe-chain rows of every
    /// fuel node and reconstructs them, exactly like CommitXenon.
    bool XeGpuCommitCandidate(double power);

    /// Commit the damped Picard image x + relax*(F - x), skipping the nodes the
    /// fused loop skips, exactly like UpdateEquilibriumXenon.
    bool XeGpuCommitPicard(double power, double relax);

    /// WP7 stage C.  The SEVEN calls above, as one -- evaluate, history, dots,
    /// solve, candidate, gate and commit, with the three decisions between them
    /// taken on the device and one host observation at the end.  The seven stay
    /// because RASBERY_GPU_XE_TXN=0 still runs them and the bit-identity gate
    /// compares this against them.
    bool XeGpuTransaction(double power, const XsReconBackend::XeTxnRequest& req,
                          xe::XeTxnControl& out);

    /// Evaluate + commit, i.e. the whole of UpdateEquilibriumXenon through the
    /// split kernels.  This is the B0 (bit-gated) half of Task 13.
    bool TryUpdateEquilibriumXenonGpuSplit(double power, double relax,
                                           double& max_change);

    // --- Raw fixed-point API (plan Rev.4 Sec 10.1) ------------------------
    //
    // UpdateEquilibriumXenon above evaluates the map, damps it, writes the
    // three Xe-chain rows and reconstructs, all in one call.  These three take
    // that apart so a caller can look at F(x) BEFORE committing anything --
    // which is what the safeguarded Anderson arm in Driver.h needs.  Nothing
    // in the default solve path calls them, and UpdateEquilibriumXenon is
    // unchanged, so RASBERY_XE_ANDERSON unset is byte-golden.
    //
    // All three index their vectors by FUEL-NODE ORDINAL -- position in
    // fuel_nodes(), NOT node index -- so the caller never touches the SoA
    // stride and the halves cannot disagree about the layout.

    /// Read the current Xe-chain inventory on fuel nodes: the iterate x.
    /// Resizes the three outputs to fuel_nodes().size().
    void SnapshotXenon(std::vector<double>& iodine_out, std::vector<double>& xenon_out,
                       std::vector<double>& xe135m_out);

    /// Evaluate the equilibrium map F(x) at the current inventory and flux
    /// WITHOUT applying it: no _iden row, no _xs entry, no node reconstruction
    /// and no generation bump.  Returns the same RAW (pre-damping) maximum
    /// relative Xe-135 change UpdateEquilibriumXenon reports, so the two
    /// numbers are directly comparable.  Always the host closed form, even
    /// with RASBERY_GPU_XSRECON set: the device kernel fuses evaluate and
    /// apply and has no evaluate-only entry point (see the XSSet.cpp header).
    double EvaluateEquilibriumXenon(double power, std::vector<double>& iodine_out,
                                    std::vector<double>& xenon_out,
                                    std::vector<double>& xe135m_out);

    /// Commit an accepted Xe-chain state: write the I-135/Xe-135/Xe-135m rows
    /// of every fuel node, reconstruct those nodes, and bump the host-state
    /// generation so the device arms re-upload.  Writes those three rows and
    /// nothing else.  Throws if a vector length does not match fuel_nodes().
    void CommitXenon(const std::vector<double>& iodine, const std::vector<double>& xenon,
                     const std::vector<double>& xe135m);

    /// Fuel-node index list in ascending node order; built once on first use
    /// and geometry-fixed.  The single owner of the list -- the xsrecon device
    /// arm batches over it too, so an independently built copy would be a
    /// silent layout fork.
    [[nodiscard]] const std::vector<int>& fuel_nodes();

    void PredictorStep(double dt, double power, bool xe_transient);
    void CorrectorStep(double dt, double power, bool xe_transient);
    void DecayIsotopeDensityFlat(std::vector<double>& iden_flat,
                                 int nxyz, double cooling_days, int substeps) const;

    [[nodiscard]] double NormFactor(double power) const;
    [[nodiscard]] double CoreHeavyMetalMassKg() const;
    [[nodiscard]] double NodeHeavyMetalMassGrams(int l) const;

    [[nodiscard]] const int& ng() const { return _g.ng(); }
    [[nodiscard]] const int& nxyz() const { return _g.nxyz(); }

    // SoA accessors
    double&       iden(size_t iso, int l) { return _iden[iso * _g.nxyz() + l]; }
    const double& iden(size_t iso, int l) const { return _iden[iso * _g.nxyz() + l]; }

    milk::Vector<double>&       iden_data() { return _iden; }
    const milk::Vector<double>& iden_data() const { return _iden; }

    size_t&                     assm(const int& la) { return _asmb[la]; }
    [[nodiscard]] const size_t& assm(const int& la) const { return _asmb[la]; }

    size_t&                     comp(const int& l) { return _comp[l]; }
    [[nodiscard]] const size_t& comp(const int& l) const { return _comp[l]; }

    [[nodiscard]] int        burn(const int& l) const { return _burn[l]; }
    int&                     burn(const int& l) { return _burn[l]; }
    [[nodiscard]] int*       burn_data() { return _burn.data(); }
    [[nodiscard]] const int* burn_data() const { return _burn.data(); }
    [[nodiscard]] int ctyp(const int& l) const { return _ctyp[l]; }
    int&              ctyp(const int& l) { return _ctyp[l]; }
    /// @brief Mean thermal fluence of the active fine-mesh rod cells [n/cm2].
    [[nodiscard]] double FineRodThermalFluenceAverage(int l, int ctype) const;
    [[nodiscard]] std::vector<double>& fine_rod_thermal_fluence_data() {
        return _fine_rod_thermal_fluence;
    }
    [[nodiscard]] const std::vector<double>& fine_rod_thermal_fluence_data() const {
        return _fine_rod_thermal_fluence;
    }

    std::map<std::string, RodGroup>&                     rod_groups() { return _rod_groups; }
    [[nodiscard]] const std::map<std::string, RodGroup>& rod_groups() const { return _rod_groups; }

    void                 SetRod(const std::map<std::string, double>& insertions);
    void                 SetBoron(double bppm);
    void                 SetRod(double step);
    void                 SetAxialRodDivision(int division);
    [[nodiscard]] int    axial_rod_division() const { return _axial_rod_division; }
    [[nodiscard]] double rod_cusping_relaxation() const { return _rod_cusping_relaxation; }
    bool                 ApplyRodCusping(double eigv, const AxialTransverseLeakageView& leakage = {});

    /// Rev.7.1 Task 10 part 3: can ApplyRodCusping do ANYTHING right now?
    ///
    /// THE QUESTION A DEVICE OUTER SEGMENT HAS TO ASK BEFORE IT STOPS ASKING.
    /// Cusping is a host call that reads the nodal drive's leakage and writes
    /// the macroscopic cross sections, so a segment that runs its outers without
    /// looking at the exit cannot afford to call it -- on an outer past the exit
    /// the leakage was never produced and the blend could not be undone.  It can
    /// afford to SKIP it exactly when skipping is an identity, and this is that
    /// condition, read off ApplyRodCusping's own body:
    ///
    ///   * no axial rod division  -> it returns false on its first line;
    ///   * no node with a partial rod -> the stencil never runs, so nothing is
    ///     written and the generation is not bumped;
    ///   * an EMPTY carry-over set -> nothing "left" the cusped set either, so
    ///     ResetCuspingNodesToBase does not run and the return value is false.
    ///
    /// The third term is the one that is easy to miss and it is what made i-SMR
    /// CY02 diverge under a Stage A eligibility check: ApplyRodCusping can
    /// answer YES from its own previous scratch with no node fractional now.
    ///
    /// AND THE ANSWER HOLDS FOR A WHOLE SEGMENT.  Geometry::rod_fraction moves
    /// only when the rod bank moves, which is the Search phase and therefore not
    /// an Outer -> Outer transition; the scratch is written only by
    /// ApplyRodCusping itself, which this answer forbids from running.
    [[nodiscard]] bool   RodCuspingQuiescent() const;

    // Rod profile matrix interface
    void                 BuildRodProfileMatrix(const std::map<std::string, std::vector<double>>& profiles);
    [[nodiscard]] double rod_max_step() const { return _rod_ncols > 1 ? static_cast<double>(_rod_ncols - 1) : 0.0; }

    /// The parsed library models.  CONST ONLY: the parse is shared by every
    /// Driver in the process (XsLibrary.h), so a mutable handle to it is a
    /// data race with every other deck that named the same library file.
    [[nodiscard]] const std::vector<Chiffon::Model>& models() const { return _lib->models; }

    /// The shared parse itself, for callers that need its provenance.
    [[nodiscard]] const std::shared_ptr<const XsLibrary>& library() const { return _lib; }

    // Macroscopic XS accessors
    [[nodiscard]] double xsnf(const int& ig, const int& l) const { return _xs.xsnf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xsdf(const int& ig, const int& l) const { return _xs.xsdf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xstf(const int& ig, const int& l) const { return _xs.xstf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xsrf(const int& ig, const int& l) const { return _xs.xsrf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xskf(const int& ig, const int& l) const { return _xs.xskf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xsaf(const int& ig, const int& l) const { return _xs.xsaf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double fyld(const int& ig, const int& l) const { return _xs.fyld[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xssf(const int& ig, const int& l) const { return _xs.xssf[ig * _g.nxyz() + l]; }
    [[nodiscard]] double xsff(const int& ig, const int& l) const { return _xs.xsff[ig * _g.nxyz() + l]; }
    [[nodiscard]] double chif(const int& ig, const int& l) const {
        if (_ref_chix.empty()) return ig == 0 ? 1.0 : 0.0;
        return _ref_chix[static_cast<size_t>(ig) * _g.nxyz() + l];
    }
    [[nodiscard]] double xssm(const int& igs, const int& ige, const int& l) const { return _xs.xssm[(igs * _g.ng() + ige) * _g.nxyz() + l]; }

    // Raw SoA pointers for the nodal device arm (same arrays the accessors
    // above index; chif may legitimately be absent).
    [[nodiscard]] const double* xsrfData() const { return _xs.xsrf.data(); }
    [[nodiscard]] const double* xsnfData() const { return _xs.xsnf.data(); }
    [[nodiscard]] const double* xssmData() const { return _xs.xssm.data(); }
    [[nodiscard]] const double* chifData() const {
        return _ref_chix.empty() ? nullptr : _ref_chix.data();
    }

    /// Shared device backend for the xs/nodal arms (created on first use;
    /// null in stub builds is fine -- the caller falls back to the CPU body).
    XsReconBackend* EnsureBackend() {
        if (!_xsrecon_backend)
            _xsrecon_backend = std::make_unique<XsReconBackend>();
        return _xsrecon_backend.get();
    }

    /// Task 16 device depletion backend, created on first use.  Never null, so
    /// the receipt in Driver.h can read its counters without asking whether the
    /// arm was ever reached.
    CramBackend& cram() {
        if (!_cram_backend) _cram_backend = std::make_unique<CramBackend>();
        return *_cram_backend;
    }
    [[nodiscard]] const CramBackend& cram() const {
        const_cast<XSSet*>(this)->cram();
        return *_cram_backend;
    }
    /// Predictor/corrector halves that ran on the host.  The receipt's
    /// `host_fallbacks`; non-zero with the arm on means the device declined.
    [[nodiscard]] unsigned long long cramHostFallbacks() const {
        return _cram_host_fallbacks;
    }

    /// WP22 device T/H backend, created on first use.  Never null, so the
    /// receipt in Driver.h can read its counters without asking whether the arm
    /// was ever reached.
    ThBackend& th() {
        if (!_th_backend) _th_backend = std::make_unique<ThBackend>();
        return *_th_backend;
    }
    [[nodiscard]] const ThBackend& th() const {
        const_cast<XSSet*>(this)->th();
        return *_th_backend;
    }
    /// T/H updates that ran on the host.  The receipt's `host_fallbacks`;
    /// non-zero with the arm on means the device declined.
    [[nodiscard]] unsigned long long thHostFallbacks() const {
        return _th_host_fallbacks;
    }
    [[nodiscard]] unsigned long long hoststateGeneration() const {
        return _hoststate_generation;
    }

    // --- WP15: micx/lmpx residency (RASBERY_GPU_MICX_RESIDENT) -------------
    //
    // With the arm on, a flat-XS device solve leaves its micx/lmpx result ON
    // THE DEVICE and owes the host a download.  EnsureMicxHost is the SINGLE
    // payer of that debt and every host reader and writer of `_micx`/`_lmpx`
    // calls it first.  That sentence is the whole safety argument, and
    // tools/test_micx_resident_contract.py is what keeps it true.
    //
    // WITH THE ARM OFF NOTHING IS EVER OWED, `_micx_device_scalars` and
    // `_micx_device_scatter` are never set, and every call here is one relaxed
    // atomic load -- so the OFF arm is the code that shipped and the A/B
    // measures the deferral, not the bookkeeping.

    /// How much of the block the caller is about to look at.
    ///
    ///   Scalars  the eleven scalar slots only -- what depletion, the Xe
    ///            algebra, the CRAM condensation and the result export read.
    ///            Leaves the 10.5 MB scatter block on the device.
    ///   All      those plus `_micx.xssm` / `_lmpx.xssm`.  What the
    ///            reconstruction and every host branch-delta path need.
    ///
    /// WHEN IN DOUBT PASS `All`.  Under-declaring is the one error this type
    /// cannot catch, and over-declaring costs 10.5 MB, not a wrong answer.
    enum class MicxNeed { Scalars, All };

    /// Bring `_micx`/`_lmpx` back from the device if a solve left them there.
    ///
    /// A no-op (one atomic load) unless the arm is on AND a solve deferred.
    /// const because every reader is: the arrays it refreshes are caches of a
    /// device block, and refreshing a cache does not change what the object
    /// means.  Safe to call from inside an OpenMP region -- the download runs
    /// under a critical section and the flag is atomic, so the first thread in
    /// pays and the rest see the debt already settled.
    void EnsureMicxHost(MicxNeed need = MicxNeed::All) const;

    /// True while any part of the block is device-only.  For receipts and for
    /// the device paths that must NOT materialise (they read the same block).
    [[nodiscard]] bool micxDeviceResident() const {
        return _micx_device_scalars.load(std::memory_order_acquire) ||
               _micx_device_scatter.load(std::memory_order_acquire);
    }

    /// "Have the host bytes of _xs.xsrf / _xs.xsdf moved since you last asked?"
    ///
    /// The ONLY sound gate for Nodal::updateConstant's node sweep: those two
    /// arrays plus Geometry::hmesh (immutable after stand-up) are the whole
    /// input of nodalConstantCoefficients, so a generation that held still means
    /// the sweep would have found `unchanged` on every node and written nothing.
    /// See _macroxs_generation for why this is not hoststateGeneration().
    [[nodiscard]] unsigned long long macroXsGeneration() const {
        return _macroxs_generation.load(std::memory_order_relaxed);
    }

    /// Bumped at the point of the write, by the four writers of xsrf/xsdf.
    void noteMacroXsWrite() {
        _macroxs_generation.fetch_add(1, std::memory_order_relaxed);
    }
    [[nodiscard]] unsigned long long refGeneration() const {
        return _ref_generation;
    }

    // Microscopic XS accessors: [(iso, group), node]
    //
    // WP15: these two are the VALUE readers of the live block -- IO's
    // node-monitor dump and NodeSpectralIndex reach `_micx` through them and
    // through nothing else -- so the debt is paid here rather than at each call
    // site.  A scalar read needs no scatter block; micxssm does.
    [[nodiscard]] double micx(Chiffon::XSTYPE xt, size_t iso, int ig, int l) const {
        EnsureMicxHost(MicxNeed::Scalars);
        const size_t elem = (iso * static_cast<size_t>(_g.ng()) + static_cast<size_t>(ig)) * static_cast<size_t>(_g.nxyz()) + static_cast<size_t>(l);
        return _micx[xt][elem];
    }
    [[nodiscard]] double refMicx(Chiffon::XSTYPE xt, size_t iso, int ig, int l) const {
        const size_t elem = (iso * static_cast<size_t>(_g.ng()) + static_cast<size_t>(ig)) * static_cast<size_t>(_g.nxyz()) + static_cast<size_t>(l);
        return _ref_micx[xt][elem];
    }
    [[nodiscard]] double micxssm(size_t iso, int igs, int ige, int l) const {
        EnsureMicxHost(MicxNeed::All);
        const size_t elem = (iso * static_cast<size_t>(_g.ng()) * static_cast<size_t>(_g.ng()) +
                             static_cast<size_t>(igs) * static_cast<size_t>(_g.ng()) + static_cast<size_t>(ige)) *
                                static_cast<size_t>(_g.nxyz()) +
                            static_cast<size_t>(l);
        return _micx.xssm[elem];
    }

    // fmap()/gmap() are gone.  They handed out `double&` into the parsed
    // models, had no caller anywhere in the tree, and indexed `_fmap` with a
    // node id against an assembly-sized array -- so they could not have had
    // one.  A mutable alias into a shared parse is the one thing this cache
    // cannot allow; PPR reads the same form functions through const models().

    // Node-level isotope density helpers
    [[nodiscard]] std::vector<double> getNodeIden(int l) const;

    void setNodeIden(int l, const std::vector<double>& v);
};

} // namespace rasbery
