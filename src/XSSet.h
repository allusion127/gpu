#pragma once
#include "Geometry.h"
#include "Model.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <map>
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
    std::vector<int>    nodes;
    std::vector<double> old_bppm; // used only when boron_difference is true
    bool                restore_reference = true;
    bool                apply_bppm        = true;
    bool                apply_tful        = true;
    bool                apply_dmod        = true;
    bool                boron_difference  = false;
};

struct AxialTransverseLeakageView {
    const double* c0 = nullptr;
    const double* c1 = nullptr;
    const double* c2 = nullptr;

    [[nodiscard]] bool valid() const { return c0 != nullptr && c1 != nullptr && c2 != nullptr; }

    [[nodiscard]] double value(int lk, int ig, int ng, double zeta) const {
        if (!valid())
            return 0.0;

        const size_t idx = static_cast<size_t>((lk * NDIRMAX + ZDIR) * ng + ig);
        const double p2  = 0.5 * (3.0 * zeta * zeta - 1.0);
        const double val = c0[idx] + c1[idx] * zeta + c2[idx] * p2;
        return std::isfinite(val) ? val : 0.0;
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

    static constexpr int BRANCH_BPPM = 0, BRANCH_TFUL = 1, BRANCH_DMOD = 2, BRANCH_HISTORY_BASE = 3;

    struct DeltaInfo {
        int nord        = 0;
        int mode        = 0;
        int ncoeff      = 0;
        int coeff_base  = 0;
        int knot_offset = 0;
        int knot_count  = 0;
    };

    struct HistoryCorrectionInfo {
        Chiffon::BRANCHTYPE          branch_type = Chiffon::BRANCHTYPE::SPCT;
        Chiffon::CorrectionComponent component   = Chiffon::CorrectionComponent::UNKNOWN;
        int                          kind        = 0;
        int                          state_field = Chiffon::AD_TMOD;
        std::vector<size_t>          vector_isotopes;
        std::vector<int>             vector_powers;
    };

    size_t _lib_ngrp = 0, _lib_ndat = 0, _lib_nmem = 0, _lib_niso = 0; // library dimensions
    size_t _lib_ndpt = 0, _lib_ndelta = 0, _lib_total_coeff = 0;       // flat table sizes
    bool   _lib_has_coeff_micx = false;                                // any delta has microscopic XS

    std::vector<size_t>                        _refr_base;        // model -> first reference flat id
    std::vector<size_t>                        _refr_ctyp_stride; // model -> flat ids per reference ctype row
    std::vector<size_t>                        _refr_burn_stride; // model -> flat id step per reference burn point
    std::vector<std::vector<int>>              _refr_ctyp;        // model -> ordered reference ctype list
    std::vector<std::vector<std::vector<int>>> _refr_burn;        // model -> ctype -> burn keys

    size_t                                                  _num_delta_branches = BRANCH_HISTORY_BASE;
    std::vector<std::vector<size_t>>                        _brch_base;        // model, branch -> first delta id
    std::vector<std::vector<size_t>>                        _brch_ctyp_stride; // model, branch -> ids per ctype row
    std::vector<std::vector<size_t>>                        _brch_burn_stride; // model, branch -> id step per burn point
    std::vector<std::vector<std::vector<int>>>              _brch_ctyp;        // model, branch -> ctype list
    std::vector<std::vector<std::vector<std::vector<int>>>> _brch_burn;        // model, branch, ctype -> burn keys

    XSArraySet           _lib_lmpx; // library reference lumped XS, indexed by depletion point
    XSArraySet           _lib_micx; // library reference microscopic XS, indexed by depletion point
    milk::Vector<double> _lib_iden; // library reference isotope densities
    std::vector<double>  _lib_burn; // library reference burnup in GWd/THM
    std::vector<double>  _lib_wvfr; // library reference water volume fraction
    std::vector<double>  _lib_tmod; // library reference average moderator temperature
    std::vector<double>  _lib_dmod; // library reference average moderator density
    std::vector<double>  _lib_flux; // library reference average flux [dpt*ng + ig]
    std::vector<double>  _lib_chix; // library reference fission spectrum [dpt*ng + ig]

    XSArraySet                                      _lib_coeff_lmpx; // branch delta coefficients for lumped XS
    XSArraySet                                      _lib_coeff_micx; // branch delta coefficients for microscopic XS
    std::vector<DeltaInfo>                          _lib_deltas;     // branch delta metadata and coefficient offsets
    std::vector<double>                             _lib_knots;      // concatenated spline knots for branch deltas
    std::vector<double>                             _lib_model_volu; // model reference volume for burnup normalization
    std::vector<double>                             _lib_model_hmas; // model heavy metal mass for burnup normalization
    std::vector<std::vector<HistoryCorrectionInfo>> _lib_history_corrections;

    // Node-wise data
    std::vector<int>                 _node_refr_lo;    // node -> lower reference depletion point id
    std::vector<int>                 _node_refr_hi;    // node -> upper reference depletion point id
    std::vector<std::vector<int>>    _node_delta_lo;   // branch, node -> lower burnup delta id
    std::vector<std::vector<int>>    _node_delta_hi;   // branch, node -> upper burnup delta id
    std::vector<std::vector<double>> _node_delta_frac; // branch, node -> burnup interpolation fraction

    std::vector<size_t> _asmb;           // assembly index of each node
    std::vector<size_t> _comp;           // composition index of each node
    std::vector<int>    _burn;           // burnof of each node
    std::vector<int>    _ctyp;           // control rod type of each node
    std::vector<int>    _history_ctyp;   // depletion trajectory ctype; IISC-RHST runtime only gates on its nonzero-ness (the surface is keyed on the current ctype)
    std::vector<double> _rodded_fluence; // node-level rod-history trajectory fluence [n/cm2]
    std::vector<char>   _external_iden;  // to preserve restart/shuffle isotope densities after InitXS

    // Live node-wise SoA cross-section storage.
    XSArraySet           _xs;   // current macroscopic XS after isotope reconstruction
    XSArraySet           _micx; // current node microscopic XS after branch/rod updates
    XSArraySet           _lmpx; // current node lumped XS after branch/rod updates
    milk::Vector<double> _iden; // current node isotope densities

    // Flat branch delta storage
    // Pre-flattened reference XS (burnup-interpolated, SoA layout)
    XSArraySet           _ref_lmpx;  // [ig*nxyz + l]
    XSArraySet           _ref_micx;  // [(iso*ng+ig)*nxyz + l]
    milk::Vector<double> _ref_iden;  // [niso*nxyz] (non-H/B/O isotopes)
    std::vector<double>  _ref_wvfr;  // unrodded AD_WVFR per node
    std::vector<double>  _ref_tmod;  // burnup-interpolated reference average moderator temperature
    std::vector<double>  _ref_dmod;  // burnup-interpolated reference average moderator density
    std::vector<double>  _node_wvfr; // live AD_WVFR per node
    std::vector<double>  _ref_flux;  // burnup-interpolated reference flux per node [ig*nxyz + l]
    std::vector<double>  _ref_chix;  // burnup-interpolated fission spectrum per node [ig*nxyz + l]
    double               _boron_dmod_average   = 0.0;
    double               _history_tmod_average = 0.0;
    double               _history_dmod_average = 0.0;
    double               _current_power_rate   = 1.0;

    bool _simd_ready = false;

    // Beginning-of-step state for predictor-corrector depletion
    XSArraySet           _xs_bos;
    XSArraySet           _micx_bos;
    milk::Vector<double> _iden_bos;
    std::vector<int>     _burn_bos;
    std::vector<int>     _history_ctyp_bos;
    std::vector<double>  _rodded_fluence_bos;
    milk::Vector<double> _flux_bos;

    std::vector<double> _node_power_scratch;
    std::vector<double> _cum_bot_scratch;
    std::vector<double> _th_tful_old_scratch;
    std::vector<double> _old_rod_fraction_scratch;
    std::vector<int>    _old_rod_ctyp_scratch;
    std::vector<int>    _dirty_nodes_scratch;
    std::vector<double> _old_bppm_scratch;
    std::vector<int>    _rod_cusping_nodes_scratch;
    int                 _axial_rod_division     = 10;
    double              _rod_cusping_relaxation = 1.0; // 1.0 = full PARCS flux-weighted cusp; <1 dilutes (see ApplyRodCusping)
    double              _th_relaxation          = 0.85; // T/H feedback under-relaxation: damps temperature oscillation, ~30% fewer outer iters (RASBERY_TH_RELAX overrides)
    std::vector<int>    _fine_rod_type;
    std::vector<double> _fine_rod_frac; // rodded fraction of each fine cell (0..1); <1 only at the tip cell
    std::vector<double> _fine_rod_fluence;
    std::vector<double> _fine_rod_fluence_bos;
    std::vector<int>    _rod_node_segment_offset;
    std::vector<int>    _rod_node_segment_ctype;

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

    void                 unpackXS(const Chiffon::CrossSection& xs, size_t l, size_t ngrp, size_t nxyz, size_t niso_count);
    void                 FlattenReferenceCrossSection(size_t flat, const Chiffon::DepletionPoint& dpt);
    void                 FlattenDeltaCrossSection(size_t coeff_base, const Chiffon::DeltaCrossSection& dxs);
    void                 FlattenBranchDelta(const Chiffon::BranchDelta& bd, size_t mi, int branch,
                                            size_t& delta_slot_idx, size_t& coeff_idx, size_t& knot_offset);
    [[nodiscard]] bool   UsesRodXS(int l) const;
    void                 RestoreReferenceNode(int l);
    void                 ApplyBranchDeltaIdToNode(int l, int did, double x, double scale, bool clamp_x = false);
    void                 ApplyBranchDeltaToNode(int l, int branch, double x, double scale);
    void                 ApplyHistoryDeltasToNode(int l);
    void                 ApplyBranchDeltasToNode(int l);
    void                 FillRodNodeXS(int l);
    void                 RefreshLightIsotopes(int l);
    void                 Reconstruct();
    void                 ReconstructNode(size_t l);
    [[nodiscard]] int    RodCTypeAtDistance(const RodGroup& group, double distance_from_tip) const;
    [[nodiscard]] double RodTotalLength(const RodGroup& group) const;
    [[nodiscard]] double FineRodFluenceAverage(int l, int ctype) const;
    void                 FillCuspingMacroXS(int l, int ctype, double fluence,
                                            std::vector<double>& scalar,
                                            std::vector<double>& scatter) const;
    void                 ApplyRodCuspingStencil(int tip_l, double reigv,
                                                const AxialTransverseLeakageView& leakage,
                                                std::vector<int>&                 touched_nodes);
    void                 RebuildFineRodOccupancy();
    void                 DepleteRodMaterials(double dt, double power, bool corrected_flux);

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
    void PrecomputeBranchCoefficients();
    void UpdateFlatXS(const XSUpdateOptions& options = XSUpdateOptions{});
    void UpdateBurnup(double dt, double power);

    // T/H property queries
    [[nodiscard]] double GetCpmod(double temperature_k, double pressure_mpa) const { return _mod_cp_table.Get(pressure_mpa, temperature_k); }
    [[nodiscard]] double GetHmod(double temperature_k, double pressure_mpa) const { return _mod_h_table.Get(pressure_mpa, temperature_k); }
    [[nodiscard]] double GetTmod(double enthalpy_kJ, double pressure_mpa) const { return _mod_t_table.Get(pressure_mpa, enthalpy_kJ); }
    [[nodiscard]] double GetDmod(double enthalpy_kJ, double pressure_mpa) const { return _mod_rho_table.Get(pressure_mpa, enthalpy_kJ); }
    [[nodiscard]] double GetTfuel(double burnup, double LPD) const { return _tf_table.Get(LPD, burnup); }

    [[nodiscard]] double pressure() const { return _g.pressure(); }

    void RemoveUpscattering();

    void InitXS(double bppm, double tful, double tmod, double pres,
                double dmod = 0.0, bool overwrite_th = true);

    void SetPowerRate(double power_rate);
    // Applies T/H feedback; returns the max relative Doppler (fuel) temperature change delta_Dop
    // used as the T/H convergence metric.
    double UpdateTH(double power_rate);
    void UpdateDerivative(double delta_bppm, double delta_tful, double delta_tmod, double delta_dmod);
    void ResetFluxAndCurrents(double flux_value = 1.0);
    void NormalizeFluxSign();

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

    [[nodiscard]] int                        ctyp(const int& l) const { return _ctyp[l]; }
    int&                                     ctyp(const int& l) { return _ctyp[l]; }
    [[nodiscard]] int                        history_ctyp(const int& l) const { return _history_ctyp[l]; }
    int&                                     history_ctyp(const int& l) { return _history_ctyp[l]; }
    [[nodiscard]] int*                       history_ctyp_data() { return _history_ctyp.data(); }
    [[nodiscard]] const int*                 history_ctyp_data() const { return _history_ctyp.data(); }
    [[nodiscard]] double                     rodded_fluence(const int& l) const { return _rodded_fluence[l]; }
    double&                                  rodded_fluence(const int& l) { return _rodded_fluence[l]; }
    [[nodiscard]] double*                    rodded_fluence_data() { return _rodded_fluence.data(); }
    [[nodiscard]] const double*              rodded_fluence_data() const { return _rodded_fluence.data(); }
    [[nodiscard]] std::vector<double>&       fine_rod_fluence_data() { return _fine_rod_fluence; }
    [[nodiscard]] const std::vector<double>& fine_rod_fluence_data() const { return _fine_rod_fluence; }

    std::map<std::string, RodGroup>&                     rod_groups() { return _rod_groups; }
    [[nodiscard]] const std::map<std::string, RodGroup>& rod_groups() const { return _rod_groups; }

    void              SetRod(const std::map<std::string, double>& insertions);
    void              SetBoron(double bppm);
    void              SetRod(double step);
    void              SetAxialRodDivision(int division);
    [[nodiscard]] int axial_rod_division() const { return _axial_rod_division; }
    [[nodiscard]] double rod_cusping_relaxation() const { return _rod_cusping_relaxation; }
    bool              ApplyRodCusping(double eigv, const AxialTransverseLeakageView& leakage = {});

    // Rod profile matrix interface
    void                 BuildRodProfileMatrix(const std::map<std::string, std::vector<double>>& profiles);
    [[nodiscard]] double rod_max_step() const { return _rod_ncols > 1 ? static_cast<double>(_rod_ncols - 1) : 0.0; }

    std::vector<Chiffon::Model>&                     models() { return _models; }
    [[nodiscard]] const std::vector<Chiffon::Model>& models() const { return _models; }
    [[nodiscard]] double                             xsadf(const int&, const int&) const { return 1.0; }

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
