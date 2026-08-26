#pragma once
#include "CudaXsReconBackend.h"
#include "FlatXsKernel.h"
#include "Geometry.h"
#include "Model.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rasbery {

// SoA cross-section array set
struct XSArraySet {
    milk::Vector<double> xstf, xsdf, xsaf, xsff, xsnf, xskf, xssf, xsrf;
    milk::Vector<double> fyld, xs2n, xs3n;
    milk::Vector<double> xssm;

    void allocate(size_t scalar_size, size_t sm_size);
    void clear();
    void fill(double v);

    milk::Vector<double>&       operator[](Chiffon::XSTYPE xt);
    const milk::Vector<double>& operator[](Chiffon::XSTYPE xt) const;
};

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

    // Chiffon database

    std::vector<Chiffon::Model> _models;

    static constexpr int BRANCH_BPPM = 0;
    static constexpr int BRANCH_TFUL = 1;
    static constexpr int BRANCH_DMOD = 2;
    static constexpr int NUM_SCALAR_BRANCHES = 3;

    struct DeltaInfo {
        int nord        = 0;
        int mode        = 0;
        int ncoeff      = 0;
        int coeff_base  = 0;
        int knot_offset = 0;
        int knot_count  = 0;
    };

    struct SpectralHistoryInfo {
        Chiffon::SpectralTerm term;
        size_t                delta_base = 0;
        std::vector<int>      burnups;
        bool                  rod_scaled = false;
    };

    bool _lib_has_coeff_micx = false; // any delta has microscopic XS

    std::vector<size_t>                        _refr_base;        // model -> first reference flat id
    std::vector<size_t>                        _refr_ctyp_stride; // model -> flat ids per reference ctype row
    std::vector<size_t>                        _refr_burn_stride; // model -> flat id step per reference burn point
    std::vector<std::vector<int>>              _refr_ctyp;        // model -> ordered reference ctype list
    std::vector<std::vector<std::vector<int>>> _refr_burn;        // model -> ctype -> burn keys

    std::vector<std::vector<size_t>>                        _brch_base;        // model, branch -> first delta id
    std::vector<std::vector<size_t>>                        _brch_ctyp_stride; // model, branch -> ids per ctype row
    std::vector<std::vector<size_t>>                        _brch_burn_stride; // model, branch -> id step per burn point
    std::vector<std::vector<std::vector<int>>>              _brch_ctyp;        // model, branch -> ctype list
    std::vector<std::vector<std::vector<std::vector<int>>>> _brch_burn;        // model, branch, ctype -> burn keys

    XSArraySet                         _lib_lmpx;         // library reference lumped XS, indexed by depletion point
    XSArraySet                         _lib_micx;         // library reference microscopic XS, indexed by depletion point
    milk::Vector<double>               _lib_iden;         // library reference isotope densities
    std::vector<double>                _lib_burn;         // library reference burnup in GWd/THM
    std::vector<double>                _lib_wvfr;         // library reference water volume fraction
    std::vector<std::array<double, 3>> _lib_ref_branch_x; // reference bppm/tful/dmod coordinates
    std::vector<double>                _lib_flux;         // library reference average flux [dpt*ng + ig]
    std::vector<double>                _lib_chix;         // library reference fission spectrum [dpt*ng + ig]

    XSArraySet                                      _lib_coeff_lmpx; // branch delta coefficients for lumped XS
    XSArraySet                                      _lib_coeff_micx; // branch delta coefficients for microscopic XS
    std::vector<DeltaInfo>                          _lib_deltas;     // branch delta metadata and coefficient offsets
    std::vector<double>                             _lib_knots;      // concatenated spline knots for branch deltas
    std::vector<double>                             _lib_model_volu; // model reference volume for burnup normalization
    std::vector<double>                             _lib_model_hmas; // model heavy metal mass for burnup normalization
    std::vector<std::vector<SpectralHistoryInfo>> _lib_spectral_history;

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
    std::vector<int>                 _lib_history_partner; // model -> partner model index (-1 none)
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

    // Advances whenever HOST code writes _xs or _iden outside the device
    // backends' own downloads (CPU reference loops, cusping blends, depletion,
    // Update, the reference rebuild).  While it holds still, the device copies
    // of _xs/_iden are bit-identical to the host's (every solve downloads what
    // it wrote), so the backends skip their per-call re-uploads.  Missing a
    // bump site corrupts physics silently -- the full-deck debug-hash A/B is
    // the gate for any new host-side writer.
    unsigned long long _hoststate_generation = 1;

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
    void               FlattenReferenceCrossSection(size_t flat, const Chiffon::DepletionPoint& dpt);
    void               FlattenDeltaCrossSection(size_t coeff_base, const Chiffon::DeltaCrossSection& dxs);
    void               FlattenBranchDelta(const Chiffon::BranchDelta& bd, size_t mi, int branch,
                                          size_t& delta_slot_idx, size_t& coeff_idx, size_t& knot_offset);
    void               FlattenSpectralHistory(
        const Chiffon::SpectralHistoryCorrection& correction,
        SpectralHistoryInfo& info, size_t& delta_slot_idx,
        size_t& coeff_idx, size_t& knot_offset);
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

    // Rod profile matrix interface
    void                 BuildRodProfileMatrix(const std::map<std::string, std::vector<double>>& profiles);
    [[nodiscard]] double rod_max_step() const { return _rod_ncols > 1 ? static_cast<double>(_rod_ncols - 1) : 0.0; }

    std::vector<Chiffon::Model>&                     models() { return _models; }
    [[nodiscard]] const std::vector<Chiffon::Model>& models() const { return _models; }

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
    [[nodiscard]] unsigned long long hoststateGeneration() const {
        return _hoststate_generation;
    }
    [[nodiscard]] unsigned long long refGeneration() const {
        return _ref_generation;
    }

    // Microscopic XS accessors: [(iso, group), node]
    [[nodiscard]] double micx(Chiffon::XSTYPE xt, size_t iso, int ig, int l) const {
        const size_t elem = (iso * static_cast<size_t>(_g.ng()) + static_cast<size_t>(ig)) * static_cast<size_t>(_g.nxyz()) + static_cast<size_t>(l);
        return _micx[xt][elem];
    }
    [[nodiscard]] double refMicx(Chiffon::XSTYPE xt, size_t iso, int ig, int l) const {
        const size_t elem = (iso * static_cast<size_t>(_g.ng()) + static_cast<size_t>(ig)) * static_cast<size_t>(_g.nxyz()) + static_cast<size_t>(l);
        return _ref_micx[xt][elem];
    }
    [[nodiscard]] double micxssm(size_t iso, int igs, int ige, int l) const {
        const size_t elem = (iso * static_cast<size_t>(_g.ng()) * static_cast<size_t>(_g.ng()) +
                             static_cast<size_t>(igs) * static_cast<size_t>(_g.ng()) + static_cast<size_t>(ige)) *
                                static_cast<size_t>(_g.nxyz()) +
                            static_cast<size_t>(l);
        return _micx.xssm[elem];
    }

    [[nodiscard]] double& fmap(const int& ig, const int& l, const int& pinx, const int& piny);
    [[nodiscard]] double& gmap(const int& l, const int& pinx, const int& piny);

    // Node-level isotope density helpers
    [[nodiscard]] std::vector<double> getNodeIden(int l) const;

    void setNodeIden(int l, const std::vector<double>& v);
};

} // namespace rasbery
