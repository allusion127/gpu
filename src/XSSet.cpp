#include "XSSet.h"

#include "Importer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

using namespace rasbery;
using namespace Chiffon;

namespace {
constexpr double WATER_NUMBER_DENSITY       = 0.033427699;
constexpr double BORON_DENSITY_FACTOR       = 5.5707678E-8;
constexpr int    OMP_THRESHOLD              = 64;
constexpr bool   USE_AVERAGE_DMOD_FOR_BORON = false;
constexpr int    CRAM_ORDER                 = 8;

// Locate a ctype in a per-node ctype list; -1 if absent.
int findCtype(const std::vector<int>& ctypes, int ctype) {
    const auto it = std::find(ctypes.begin(), ctypes.end(), ctype);
    return it == ctypes.end() ? -1 : static_cast<int>(std::distance(ctypes.begin(), it));
}

// Lower/upper burnup-bracket indices for burn_key in a sorted key list; -1 if empty.
int findLoBurn(const std::vector<int>& keys, int burn_key) {
    if (keys.empty()) return -1;
    auto it = std::lower_bound(keys.begin(), keys.end(), burn_key);
    if (it == keys.end()) return static_cast<int>(keys.size() - 1);
    if (it == keys.begin()) return 0;
    return static_cast<int>(std::distance(keys.begin(), std::prev(it)));
}

int findHiBurn(const std::vector<int>& keys, int burn_key) {
    if (keys.empty()) return -1;
    auto it = std::lower_bound(keys.begin(), keys.end(), burn_key);
    if (it == keys.end()) return static_cast<int>(keys.size() - 1);
    return static_cast<int>(std::distance(keys.begin(), it));
}

// Burn keys of an inner burn map, in map (sorted) order.
template <class BurnMap>
std::vector<int> burnKeys(const BurnMap& burn_map) {
    std::vector<int> keys;
    keys.reserve(burn_map.size());
    for (const auto& [burn_key, value] : burn_map)
        keys.push_back(burn_key);
    return keys;
}

// Sorted ctype keys of a per-ctype table (Reference or BranchDelta).
template <class ByCtype>
std::vector<int> ctypeKeys(const ByCtype& by_ctype) {
    std::vector<int> keys;
    keys.reserve(by_ctype.size());
    for (const auto& [ctype, bmap] : by_ctype)
        keys.push_back(ctype);
    std::sort(keys.begin(), keys.end());
    return keys;
}

// Largest per-ctype burn count in a per-ctype table.
template <class ByCtype>
size_t maxBurnCount(const ByCtype& by_ctype) {
    size_t max_count = 0;
    for (const auto& [ctype, bmap] : by_ctype)
        max_count = std::max(max_count, bmap.size());
    return max_count;
}

size_t countReferenceSlots(const Chiffon::Reference& ref) {
    if (ref.empty()) return 0;
    return ref.size() * maxBurnCount(ref);
}

size_t countDeltaSlots(const Chiffon::BranchDelta& bd) {
    if (bd.empty()) return 0;
    return bd.size() * maxBurnCount(bd);
}

size_t countDeltaCoefficients(const Chiffon::BranchDelta& bd) {
    size_t coeff_count = 0;
    for (const auto& [ctype, bmap] : bd)
        for (const auto& [bkey, dxs] : bmap)
            coeff_count += dxs.nord();
    return coeff_count;
}

double FuelVolumeAverageDmod(Geometry& geometry) {
    const int nxyz = geometry.nxyz();

    double weighted_sum = 0.0;
    double volume_sum   = 0.0;
    for (int l = 0; l < nxyz; ++l) {
        if (!geometry.IsFuel(l)) continue;

        const double volume = geometry.vol(l);
        if (volume <= 1.0e-20) continue;

        weighted_sum += geometry.dmod(l) * volume;
        volume_sum += volume;
    }

    if (volume_sum > 0.0) return weighted_sum / volume_sum;
    return (nxyz > 0) ? geometry.dmod(0) : 0.0;
}

double FuelVolumeAverageTmod(Geometry& geometry) {
    const int nxyz = geometry.nxyz();

    double weighted_sum = 0.0;
    double volume_sum   = 0.0;
    for (int l = 0; l < nxyz; ++l) {
        if (!geometry.IsFuel(l)) continue;

        const double volume = geometry.vol(l);
        if (volume <= 1.0e-20) continue;

        weighted_sum += geometry.tmod(l) * volume;
        volume_sum += volume;
    }

    if (volume_sum > 0.0) return weighted_sum / volume_sum;
    return (nxyz > 0) ? geometry.tmod(0) : 0.0;
}

double BoronDmod(Geometry& geometry, double average_dmod, int l) {
    return USE_AVERAGE_DMOD_FOR_BORON ? average_dmod : geometry.dmod(l);
}

} // namespace

using ScalarXSRefs      = std::array<milk::Vector<double>*, N_XS_SCALAR>;
using ConstScalarXSRefs = std::array<const milk::Vector<double>*, N_XS_SCALAR>;
using ConstScalarPtrs   = std::array<const double*, N_XS_SCALAR>;

static ScalarXSRefs ScalarXS(XSArraySet& xs) {
    return {&xs.xstf, &xs.xsdf, &xs.xsaf, &xs.xsff, &xs.xsnf, &xs.xskf,
            &xs.xssf, &xs.xsrf, &xs.fyld, &xs.xs2n, &xs.xs3n};
}

static ConstScalarXSRefs ScalarXS(const XSArraySet& xs) {
    return {&xs.xstf, &xs.xsdf, &xs.xsaf, &xs.xsff, &xs.xsnf, &xs.xskf,
            &xs.xssf, &xs.xsrf, &xs.fyld, &xs.xs2n, &xs.xs3n};
}

static ConstScalarPtrs ScalarData(const XSArraySet& xs) {
    ConstScalarPtrs ptrs{};
    const auto      vectors = ScalarXS(xs);
    for (size_t i = 0; i < vectors.size(); ++i)
        ptrs[i] = vectors[i]->data();
    return ptrs;
}

static void CopyDoubles(size_t count, const double* src, double* dst) {
    if (count != 0)
        std::copy_n(src, count, dst);
}

static bool SolveDenseLinearSystem(std::vector<double>& a, std::vector<double>& b, int n) {
    for (int col = 0; col < n; ++col) {
        int    pivot = col;
        double best  = std::abs(a[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const double cand = std::abs(a[row * n + col]);
            if (cand > best) {
                best  = cand;
                pivot = row;
            }
        }
        if (best < 1.0e-18 || !std::isfinite(best))
            return false;
        if (pivot != col) {
            for (int j = col; j < n; ++j)
                std::swap(a[col * n + j], a[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }

        const double diag = a[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            const double factor = a[row * n + col] / diag;
            if (std::abs(factor) < 1.0e-30)
                continue;
            a[row * n + col] = 0.0;
            for (int j = col + 1; j < n; ++j)
                a[row * n + j] -= factor * a[col * n + j];
            b[row] -= factor * b[col];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        double sum = b[row];
        for (int col = row + 1; col < n; ++col)
            sum -= a[row * n + col] * b[col];
        b[row] = sum / a[row * n + row];
        if (!std::isfinite(b[row]))
            return false;
    }
    return true;
}

void XSArraySet::allocate(size_t scalar_size, size_t sm_size) {
    for (auto* xs : ScalarXS(*this))
        xs->assign(scalar_size, 0.0);
    xssm.assign(sm_size, 0.0);
}

void XSArraySet::clear() {
    for (auto* xs : ScalarXS(*this))
        xs->clear();
    xssm.clear();
}

void XSArraySet::fill(double value) {
    for (auto* xs : ScalarXS(*this))
        xs->fill(value);
    xssm.fill(value);
}

milk::Vector<double>& XSArraySet::operator[](Chiffon::XSTYPE xt) {
    if (xt == Chiffon::XSSM)
        return xssm;

    const auto index = static_cast<size_t>(xt);
    auto       xs    = ScalarXS(*this);
    if (index < xs.size())
        return *xs[index];
    throw std::out_of_range("XSArraySet: invalid scalar XS type");
}

const milk::Vector<double>& XSArraySet::operator[](Chiffon::XSTYPE xt) const {
    if (xt == Chiffon::XSSM)
        return xssm;

    const auto index = static_cast<size_t>(xt);
    auto       xs    = ScalarXS(*this);
    if (index < xs.size())
        return *xs[index];
    throw std::out_of_range("XSArraySet: invalid scalar XS type");
}

namespace rasbery {

struct DepletionWorkspace {
    milk::Vector<double> iden;
    milk::Matrix<double> matrix;
    std::vector<double>  condensed;
    milk::CramWorkspace  cram;

    void ensure(size_t n);
};

} // namespace rasbery

void XSSet::unpackXS(const Chiffon::CrossSection& xs, size_t l, size_t ngrp, size_t nxyz, size_t niso_count) {
    auto lmpx = ScalarXS(_lmpx);
    auto micx = ScalarXS(_micx);

    for (size_t ig = 0; ig < ngrp; ++ig) {
        const size_t off = ig * nxyz + l;
        for (size_t xt = 0; xt < lmpx.size(); ++xt)
            (*lmpx[xt])[off] = xs.lmpxs(ig, static_cast<XSTYPE>(xt));
        for (size_t ige = 0; ige < ngrp; ++ige)
            _lmpx.xssm[(ig * ngrp + ige) * nxyz + l] = xs.lmpxssm(ig, ige);
    }

    for (size_t iso = 0; iso < niso_count; ++iso) {
        for (size_t ig = 0; ig < ngrp; ++ig) {
            const size_t off  = (iso * ngrp + ig) * nxyz + l;
            const int    iiso = static_cast<int>(iso);
            for (size_t xt = 0; xt < micx.size(); ++xt) {
                (*micx[xt])[off] = (xt == static_cast<size_t>(XSRF))
                                       ? 0.0
                                       : xs.mixs(iiso, ig, static_cast<XSTYPE>(xt));
            }
            for (size_t ige = 0; ige < ngrp; ++ige)
                _micx.xssm[(iso * ngrp * ngrp + ig * ngrp + ige) * nxyz + l] = xs.mixssm(iiso, ig, ige);
        }
    }
}

void XSSet::FlattenReferenceCrossSection(size_t flat, const Chiffon::DepletionPoint& dpt) {
    const auto&  xs   = dpt._xs;
    const size_t ng   = static_cast<size_t>(_g.ng());
    const size_t niso = Isotope::niso;

    // 1. Copy lumped scalar cross sections and scattering matrices.
    for (size_t ig = 0; ig < ng; ++ig) {
        const size_t off = flat * ng + ig;
        for (int xt = XSTF; xt <= XS3N; ++xt)
            _lib_lmpx[static_cast<XSTYPE>(xt)][off] = xs.lmpxs(ig, static_cast<XSTYPE>(xt));
        for (size_t ige = 0; ige < ng; ++ige)
            _lib_lmpx.xssm[flat * ng * ng + ig * ng + ige] = xs.lmpxssm(ig, ige);
    }

    // 2. Copy microscopic scalar cross sections and scattering matrices isotope by isotope.
    for (size_t iso = 0; iso < niso; ++iso) {
        const int iiso = static_cast<int>(iso);
        for (size_t ig = 0; ig < ng; ++ig) {
            const size_t off = (flat * niso + iso) * ng + ig;
            for (int xt = XSTF; xt <= XS3N; ++xt)
                _lib_micx[static_cast<XSTYPE>(xt)][off] = xs.mixs(iiso, ig, static_cast<XSTYPE>(xt));
            for (size_t ige = 0; ige < ng; ++ige)
                _lib_micx.xssm[(flat * niso + iso) * ng * ng + ig * ng + ige] =
                    xs.mixssm(iiso, ig, ige);
        }
    }

    // 3. Copy isotope densities and scalar depletion-state data.
    for (size_t iso = 0; iso < niso; ++iso)
        _lib_iden[flat * niso + iso] = dpt._iden[iso];
    _lib_burn[flat] = dpt._data[AD_BURN];
    _lib_wvfr[flat] = dpt._data[AD_WVFR];
    _lib_tmod[flat] = dpt._data[AD_TMOD];
    _lib_dmod[flat] = dpt._data[AD_DMOD];

    // 4. Copy the average flux and normalize the fission spectrum.
    const size_t flux_count = std::min(ng, dpt._aflx.size());
    for (size_t ig = 0; ig < ng; ++ig)
        _lib_flux[flat * ng + ig] = ig < flux_count ? dpt._aflx[ig] : 0.0;

    const size_t chi_count = std::min(ng, dpt._chix.size());
    double       chi_sum   = 0.0;
    for (size_t ig = 0; ig < chi_count; ++ig)
        chi_sum += dpt._chix[ig];

    for (size_t ig = 0; ig < ng; ++ig) {
        double value = ig < chi_count ? dpt._chix[ig] : 0.0;
        if (chi_sum > 1.0e-30)
            value /= chi_sum;
        else if (ig == 0)
            value = 1.0;
        _lib_chix[flat * ng + ig] = value;
    }
}

void XSSet::FlattenDeltaCrossSection(size_t coeff_base, const Chiffon::DeltaCrossSection& dxs) {
    const size_t ng   = static_cast<size_t>(_g.ng());
    const size_t niso = Isotope::niso;

    for (size_t p = 0; p < dxs.nord(); ++p) {
        const auto&  coeff_xs = dxs[p];
        const size_t coeff    = coeff_base + p;

        // 1. Copy lumped coefficient cross sections.
        for (size_t ig = 0; ig < ng; ++ig) {
            const size_t off = coeff * ng + ig;
            for (int xt = XSTF; xt <= XS3N; ++xt)
                _lib_coeff_lmpx[static_cast<XSTYPE>(xt)][off] =
                    coeff_xs.lmpxs(ig, static_cast<XSTYPE>(xt));
            for (size_t ige = 0; ige < ng; ++ige)
                _lib_coeff_lmpx.xssm[coeff * ng * ng + ig * ng + ige] =
                    coeff_xs.lmpxssm(ig, ige);
        }

        // 2. Copy microscopic coefficient cross sections when the delta carries them.
        if (!coeff_xs.has_micx()) continue;

        _lib_has_coeff_micx = true;
        for (size_t iso = 0; iso < niso; ++iso) {
            const int iiso = static_cast<int>(iso);
            for (size_t ig = 0; ig < ng; ++ig) {
                const size_t off = (coeff * niso + iso) * ng + ig;
                for (int xt = XSTF; xt <= XS3N; ++xt)
                    _lib_coeff_micx[static_cast<XSTYPE>(xt)][off] =
                        coeff_xs.mixs(iiso, ig, static_cast<XSTYPE>(xt));
                for (size_t ige = 0; ige < ng; ++ige)
                    _lib_coeff_micx.xssm[(coeff * niso + iso) * ng * ng + ig * ng + ige] =
                        coeff_xs.mixssm(iiso, ig, ige);
            }
        }
    }
}

void XSSet::FlattenBranchDelta(const Chiffon::BranchDelta& bd, size_t mi, int branch,
                               size_t& delta_slot_idx, size_t& coeff_idx, size_t& knot_offset) {
    _brch_base[mi][branch]        = delta_slot_idx;
    _brch_ctyp_stride[mi][branch] = bd.empty() ? 0 : maxBurnCount(bd);
    _brch_burn_stride[mi][branch] = 1;
    _brch_ctyp[mi][branch]        = ctypeKeys(bd);
    _brch_burn[mi][branch].resize(_brch_ctyp[mi][branch].size());
    if (bd.empty()) return;

    for (size_t ci = 0; ci < _brch_ctyp[mi][branch].size(); ++ci) {
        const int   ctype = _brch_ctyp[mi][branch][ci];
        const auto& bmap  = bd.at(ctype);
        auto&       keys  = _brch_burn[mi][branch][ci];
        keys              = burnKeys(bmap);
        for (size_t bi = 0; bi < keys.size(); ++bi) {
            const int    burn_key = keys[bi];
            const size_t flat_did = _brch_base[mi][branch] +
                                    ci * _brch_ctyp_stride[mi][branch] +
                                    bi * _brch_burn_stride[mi][branch];
            const auto& dxs = bmap.at(burn_key);

            auto& info       = _lib_deltas[flat_did];
            info.nord        = static_cast<int>(dxs.nord());
            info.mode        = (dxs.mode() == Chiffon::SPLINE_MODE) ? 1 : 0;
            info.ncoeff      = static_cast<int>(dxs.ncoeff());
            info.coeff_base  = static_cast<int>(coeff_idx);
            info.knot_offset = static_cast<int>(knot_offset);
            info.knot_count  = static_cast<int>(dxs.knots().size());

            // Copy knots
            for (double k : dxs.knots())
                _lib_knots.push_back(k);
            knot_offset += dxs.knots().size();

            FlattenDeltaCrossSection(coeff_idx, dxs);
            coeff_idx += info.nord;
        }
    }
    delta_slot_idx += _brch_ctyp[mi][branch].size() * _brch_ctyp_stride[mi][branch];
}

static void LoadNodeIden(const milk::Vector<double>& soa, int nxyz, int l, milk::Vector<double>& iden) {
    const size_t niso = Chiffon::Isotope::niso;
    if (iden.size() != niso)
        iden.assign(niso, 0.0);
    for (size_t iso = 0; iso < niso; ++iso)
        iden[iso] = soa[iso * static_cast<size_t>(nxyz) + static_cast<size_t>(l)];
}

// Lifetime

XSSet::XSSet(Geometry& g) noexcept : _g(g) {
    if (const char* r = std::getenv("RASBERY_CUSP_RELAX")) {
        const double v = std::atof(r);
        if (v > 0.0 && v <= 2.0) // allow over-relaxation (>1) for cusp-convergence experiments
            _rod_cusping_relaxation = v;
    }
    if (const char* r = std::getenv("RASBERY_TH_RELAX")) {
        const double v = std::atof(r);
        if (v > 0.0 && v <= 1.0)
            _th_relaxation = v;
    }
}

XSSet::~XSSet() = default;

void XSSet::LoadTHTables() {
#ifdef DATA_DIR
    const auto base = std::filesystem::path(DATA_DIR) / "include" / "Database";
#else
    const auto base = std::filesystem::path("include") / "Database";
#endif
    _mod_cp_table  = milk::Table::ParseFromCSV(base / "mod_cp.csv");
    _mod_rho_table = milk::Table::ParseFromCSV(base / "mod_rho.csv");
    _mod_h_table   = milk::Table::ParseFromCSV(base / "mod_h.csv");
    _mod_t_table   = milk::Table::ParseFromCSV(base / "mod_t.csv");
    _tf_table      = milk::Table::ParseFromCSV(base / "tf.csv");
}

void DepletionWorkspace::ensure(size_t n) {
    if (iden.size() == n) return;

    iden   = milk::Vector<double>(n);
    matrix = milk::Matrix<double>(n, n);
    condensed.resize(n * N_XS_SCALAR);
}

// Initialization

void XSSet::SetAxialRodDivision(int division) {
    _axial_rod_division     = std::max(1, division);
    const size_t fine_count = static_cast<size_t>(_g.nxy()) *
                              static_cast<size_t>(_g.nz()) *
                              static_cast<size_t>(_axial_rod_division);
    _fine_rod_type.assign(fine_count, 0);
    _fine_rod_fluence.assign(fine_count, 0.0);
    _fine_rod_fluence_bos.assign(fine_count, 0.0);
    RebuildFineRodOccupancy();
}

void XSSet::Initialize(const std::string& xs_path) {
    const int nz    = _g.nz();
    const int nxyz  = _g.nxyz();
    const int nxyza = _g.nxyza();

    // 1. Allocate node-level storage and load external library data.
    _comp.assign(nxyz, 0);
    _asmb.assign(nxyza, 0);
    _ctyp.assign(nxyz, 0);
    _history_ctyp.assign(nxyz, 0);
    _rodded_fluence.assign(nxyz, 0.0);
    _burn.assign(nxyz, 0);
    _external_iden.assign(nxyz, 0);

    _burn_bos.resize(nxyz);
    _history_ctyp_bos.resize(nxyz);
    _rodded_fluence_bos.resize(nxyz);
    _flux_bos = milk::Vector<double>(static_cast<size_t>(nxyz * _g.ng()));

    _node_power_scratch.assign(nxyz, 0.0);
    _cum_bot_scratch.assign(nz + 1, 0.0);
    _rod_node_segment_offset.assign(nxyz + 1, 0);
    _rod_node_segment_ctype.clear();
    _rod_node_segment_fraction.clear();
    _fine_rod_type.assign(static_cast<size_t>(_g.nxy()) * static_cast<size_t>(nz) *
                              static_cast<size_t>(_axial_rod_division),
                          0);
    _fine_rod_fluence.assign(_fine_rod_type.size(), 0.0);
    _fine_rod_fluence_bos.assign(_fine_rod_type.size(), 0.0);

    // Load T/H property tables
    LoadTHTables();

    // Load Chiffon XS library (sets Isotope::niso)
    Importer importer;
    _models = importer.LoadHDF(xs_path);

    // Load unified depletion matrices from CSV if not already loaded from HDF5
    if (Isotope::depDecay.size() == 0) {
#ifdef DATA_DIR
        const auto chainDir = std::filesystem::path(DATA_DIR) / "include" / "Database";
#else
        const auto chainDir = std::filesystem::path("include") / "Database";
#endif
        if (std::filesystem::exists(chainDir / "dep_decay.csv"))
            Isotope::Initialize(chainDir);
    }

    const size_t niso = Isotope::niso;
    const size_t ng   = static_cast<size_t>(_g.ng());

    // Allocate SoA arrays.
    const size_t ngn  = ng * nxyz;             // scalar XS size
    const size_t smsz = ng * ng * nxyz;        // scatter matrix size
    const size_t micn = niso * ng * nxyz;      // microscopic scalar size
    const size_t mism = niso * ng * ng * nxyz; // microscopic scatter size

    _xs.allocate(ngn, smsz);
    _lmpx.allocate(ngn, smsz);
    _micx.allocate(micn, mism);
    _iden.assign(niso * nxyz, 0.0);

    // Pre-flattened reference + coefficient arrays
    _ref_lmpx.allocate(ngn, smsz);
    _ref_micx.allocate(micn, mism);
    _ref_iden.assign(niso * nxyz, 0.0);
    _ref_wvfr.assign(nxyz, 0.0);
    _ref_tmod.assign(nxyz, 0.0);
    _ref_dmod.assign(nxyz, 0.0);
    _node_wvfr.assign(nxyz, 0.0);
    _ref_flux.assign(static_cast<size_t>(ng) * nxyz, 0.0);
    _ref_chix.assign(static_cast<size_t>(ng) * nxyz, 0.0);
    _simd_ready = false;

    // Beginning-of-step snapshots for predictor-corrector depletion.
    _xs_bos.allocate(ngn, 0);
    _micx_bos.allocate(micn, 0);
    _iden_bos.assign(niso * nxyz, 0.0);

    // 2. Build model-name lookup and assign each node to its XS model.
    std::map<std::string, size_t> modelIndexMap;
    size_t                        modelId = 0;
    for (const auto& model : _models)
        modelIndexMap[model.name()] = modelId++;

    const auto& core                    = _g.core();
    const auto& batch                   = _g.batch();
    const int   nxy                     = _g.nxy();
    const int   nxya                    = _g.nxya();
    const int   ndivxy                  = _g.ndivxy();
    const int   symang                  = _g.symang();
    const bool  symdiv                  = _g.symdiv();
    const bool  use_half_symmetric_edge = (symang == 90 && symdiv);

    for (int z = 0; z < nz; ++z) {
        for (size_t row = 0; row < core.size(); ++row) {
            for (size_t col = 0; col < core[row].size(); ++col) {
                if (core[row][col] == "XX") continue;

                const std::string& asmb = batch.at(core[row][col])[nz - 1 - z];
                int                lka  = _g.ijtola(static_cast<int>(col), static_cast<int>(row)) + nxya * z;

                if (lka < 0 || lka >= _g.nxyza()) {
                    PLOG_ERROR << "Assembly index out of range (lka=" << lka << ").";
                    continue;
                }

                _asmb[lka] = modelIndexMap.at(asmb);

                int rowend = ndivxy;
                int colend = ndivxy;
                if (use_half_symmetric_edge && row == 0) rowend = ndivxy / 2;
                if (use_half_symmetric_edge && col == 0) colend = ndivxy / 2;
                for (int j = 0; j < rowend; ++j) {
                    for (int i = 0; i < colend; ++i) {
                        int idx = static_cast<int>(col) * ndivxy + i;
                        int idy = static_cast<int>(row) * ndivxy + j;
                        if (use_half_symmetric_edge && col > 0) idx -= ndivxy / 2;
                        if (use_half_symmetric_edge && row > 0) idy -= ndivxy / 2;
                        int lk = _g.ijtol(idx, idy) + nxy * z;
                        if (lk >= 0 && lk < nxyz)
                            _comp[lk] = modelIndexMap.at(asmb);
                    }
                }
            }
        }
    }

    // 3. Flatten model data into library-wide SoA storage.
    {
        const size_t nmodels = _models.size();
        _lib_ngrp            = ng;
        _lib_ndat            = XSSM + ng;
        _lib_nmem            = _lib_ndat * ng;
        _lib_niso            = niso;

        // Count totals across all models
        size_t total_dpts  = 0;
        size_t total_delta = 0;
        size_t total_coeff = 0;
        for (const auto& m : _models) {
            total_dpts += countReferenceSlots(m._refr_dpts);
            total_delta += countDeltaSlots(m._bppm_delt);
            total_delta += countDeltaSlots(m._tful_delt);
            total_delta += countDeltaSlots(m._dmod_delt);
            total_coeff += countDeltaCoefficients(m._bppm_delt);
            total_coeff += countDeltaCoefficients(m._tful_delt);
            total_coeff += countDeltaCoefficients(m._dmod_delt);
            for (const auto& history : m._history_deltas) {
                total_delta += countDeltaSlots(history.delta);
                total_coeff += countDeltaCoefficients(history.delta);
            }
        }

        _lib_ndpt        = total_dpts;
        _lib_ndelta      = total_delta;
        _lib_total_coeff = total_coeff;

        // Allocate flat arrays
        _lib_lmpx.allocate(total_dpts * ng, total_dpts * ng * ng);
        _lib_micx.allocate(total_dpts * niso * ng, total_dpts * niso * ng * ng);
        _lib_iden.assign(total_dpts * niso, 0.0);
        _lib_burn.assign(total_dpts, 0.0);
        _lib_wvfr.assign(total_dpts, 0.0);
        _lib_tmod.assign(total_dpts, 0.0);
        _lib_dmod.assign(total_dpts, 0.0);
        _lib_flux.assign(total_dpts * ng, 0.0);
        _lib_chix.assign(total_dpts * ng, 0.0);
        _lib_coeff_lmpx.allocate(total_coeff * ng, total_coeff * ng * ng);
        _lib_coeff_micx.allocate(total_coeff * niso * ng, total_coeff * niso * ng * ng);
        _lib_has_coeff_micx = false;
        _lib_deltas.resize(total_delta);
        _lib_knots.clear();
        _refr_base.resize(nmodels, 0);
        _refr_ctyp_stride.resize(nmodels, 0);
        _refr_burn_stride.resize(nmodels, 1);
        _refr_ctyp.resize(nmodels);
        _refr_burn.resize(nmodels);
        _num_delta_branches = BRANCH_HISTORY_BASE;
        for (const auto& m : _models)
            _num_delta_branches = std::max(_num_delta_branches, BRANCH_HISTORY_BASE + m._history_deltas.size());
        _brch_base.assign(nmodels, std::vector<size_t>(_num_delta_branches, 0));
        _brch_ctyp_stride.assign(nmodels, std::vector<size_t>(_num_delta_branches, 0));
        _brch_burn_stride.assign(nmodels, std::vector<size_t>(_num_delta_branches, 1));
        _brch_ctyp.assign(nmodels, std::vector<std::vector<int>>(_num_delta_branches));
        _brch_burn.assign(nmodels, std::vector<std::vector<std::vector<int>>>(_num_delta_branches));
        _lib_model_volu.resize(nmodels, 1.0);
        _lib_model_hmas.resize(nmodels, 1.0);
        _lib_history_corrections.assign(nmodels, {});

        // Flatten reference depletion points in model/ctype/burn stride order.
        size_t dpt_idx = 0;
        for (size_t mi = 0; mi < nmodels; ++mi) {
            const auto& m    = _models[mi];
            const auto& dpts = m.Dpts();
            _lib_history_corrections[mi].reserve(m._history_deltas.size());
            for (const auto& history : m._history_deltas) {
                _lib_history_corrections[mi].push_back(
                    HistoryCorrectionInfo{history.branch_type,
                                          Chiffon::CorrectionComponentFromHistoryKind(history.kind),
                                          history.kind, history.state_field,
                                          history.vector_isotopes, history.vector_powers});
            }

            _refr_base[mi] = dpt_idx;
            if (m._refr_dpts.empty())
                continue;
            _refr_ctyp_stride[mi] = maxBurnCount(m._refr_dpts);
            _refr_burn_stride[mi] = 1;
            _refr_ctyp[mi]        = ctypeKeys(m._refr_dpts);
            _refr_burn[mi].resize(_refr_ctyp[mi].size());

            // Per-model constants from dpt(0, first burn), used by UpdateBurnup.
            if (!m._refr_dpts.empty() && m._refr_dpts.count(0)) {
                auto         it            = m._refr_dpts.at(0).begin();
                const auto&  dpt0          = dpts[it->second];
                const double assembly_area = dpt0._data[AD_APIT] * dpt0._data[AD_APIT];
                _lib_model_volu[mi]        = (assembly_area > 0.0) ? assembly_area : dpt0._data[AD_VOLU];
                _lib_model_hmas[mi]        = dpt0._data[AD_HMAS];
            }

            for (size_t ci = 0; ci < _refr_ctyp[mi].size(); ++ci) {
                const int   ctype = _refr_ctyp[mi][ci];
                const auto& bmap  = m._refr_dpts.at(ctype);
                auto&       keys  = _refr_burn[mi][ci];
                keys              = burnKeys(bmap);
                for (size_t bi = 0; bi < keys.size(); ++bi) {
                    const int    burn_key = keys[bi];
                    const size_t flat     = _refr_base[mi] + ci * _refr_ctyp_stride[mi] +
                                        bi * _refr_burn_stride[mi];
                    FlattenReferenceCrossSection(flat, dpts[bmap.at(burn_key)]);
                }
            }

            dpt_idx += _refr_ctyp[mi].size() * _refr_ctyp_stride[mi];
        }

        // Flatten branch delta coefficients.
        size_t delta_slot_idx = 0;
        size_t coeff_idx      = 0;
        size_t knot_offset    = 0;

        for (size_t mi = 0; mi < nmodels; ++mi) {
            FlattenBranchDelta(_models[mi]._bppm_delt, mi, BRANCH_BPPM, delta_slot_idx, coeff_idx, knot_offset);
            FlattenBranchDelta(_models[mi]._tful_delt, mi, BRANCH_TFUL, delta_slot_idx, coeff_idx, knot_offset);
            FlattenBranchDelta(_models[mi]._dmod_delt, mi, BRANCH_DMOD, delta_slot_idx, coeff_idx, knot_offset);
            for (size_t h = 0; h < _models[mi]._history_deltas.size(); ++h) {
                FlattenBranchDelta(_models[mi]._history_deltas[h].delta, mi,
                                   static_cast<int>(BRANCH_HISTORY_BASE + h), delta_slot_idx, coeff_idx, knot_offset);
            }
        }

        _node_refr_lo.assign(nxyz, -1);
        _node_refr_hi.assign(nxyz, -1);
        _node_delta_lo.assign(_num_delta_branches, std::vector<int>(nxyz, -1));
        _node_delta_hi.assign(_num_delta_branches, std::vector<int>(nxyz, -1));
        _node_delta_frac.assign(_num_delta_branches, std::vector<double>(nxyz, 0.0));
    }
}

// Reconstruct: rmcx = lmpx + sum(micx_i * iden_i), then derive D and removal.
void XSSet::Reconstruct() {
    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    // 1. Rebuild scalar macroscopic XS from lumped and microscopic terms.
    for (int xt = XSTF; xt <= XS3N; ++xt) {
        if (xt == XSDF || xt == XSRF) continue;
        auto        xtype = static_cast<XSTYPE>(xt);
        auto&       dst   = _xs[xtype];
        const auto& lmp   = _lmpx[xtype];
        const auto& mic   = _micx[xtype];

        // dst = lmpx (copy)
        CopyDoubles(static_cast<size_t>(ng) * static_cast<size_t>(nxyz), lmp.data(), dst.data());

        // dst += Σ_iso (micx[iso] * iden[iso]).
        // iden varies by node, so each isotope/group block is an element-wise stride-1 loop.
        for (size_t iso = 0; iso < niso; ++iso) {
            for (int ig = 0; ig < ng; ++ig) {
                const double* mic_ptr  = mic.data() + (iso * ng + ig) * nxyz;
                const double* iden_ptr = _iden.data() + iso * nxyz;
                double*       dst_ptr  = dst.data() + ig * nxyz;

#pragma omp simd
                for (int l = 0; l < nxyz; ++l)
                    dst_ptr[l] += mic_ptr[l] * iden_ptr[l];
            }
        }
    }

    // 2. Rebuild the scattering matrix with the same SoA accumulation pattern.
    {
        auto&       dst = _xs.xssm;
        const auto& lmp = _lmpx.xssm;
        const auto& mic = _micx.xssm;
        CopyDoubles(static_cast<size_t>(ng) * static_cast<size_t>(ng) * static_cast<size_t>(nxyz),
                    lmp.data(), dst.data());

        for (size_t iso = 0; iso < niso; ++iso) {
            for (int igs = 0; igs < ng; ++igs) {
                for (int ige = 0; ige < ng; ++ige) {
                    const double* mic_ptr  = mic.data() + (iso * ng * ng + igs * ng + ige) * nxyz;
                    const double* iden_ptr = _iden.data() + iso * nxyz;
                    double*       dst_ptr  = dst.data() + (igs * ng + ige) * nxyz;

#pragma omp simd
                    for (int l = 0; l < nxyz; ++l)
                        dst_ptr[l] += mic_ptr[l] * iden_ptr[l];
                }
            }
        }
    }

    // 3. Derive diffusion coefficients from the transport cross section.
    for (int ig = 0; ig < ng; ++ig) {
        double* xstf_ptr = _xs.xstf.data() + ig * nxyz;
        double* xsdf_ptr = _xs.xsdf.data() + ig * nxyz;
#pragma omp simd
        for (int l = 0; l < nxyz; ++l) {
            double tr   = xstf_ptr[l];
            xsdf_ptr[l] = (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;
        }
    }

    // 4. Recompute removal as absorption plus outgoing scattering.
    for (int igs = 0; igs < ng; ++igs) {
        double*       xsrf_ptr = _xs.xsrf.data() + igs * nxyz;
        const double* xsaf_ptr = _xs.xsaf.data() + igs * nxyz;
#pragma omp simd
        for (int l = 0; l < nxyz; ++l)
            xsrf_ptr[l] = xsaf_ptr[l];

        for (int ige = 0; ige < ng; ++ige) {
            const double* sm_ptr = _xs.xssm.data() + (igs * ng + ige) * nxyz;
#pragma omp simd
            for (int l = 0; l < nxyz; ++l)
                xsrf_ptr[l] += sm_ptr[l];
        }
    }
}

void XSSet::ReconstructNode(size_t l) {
    const int    ng   = _g.ng();
    const int    nxyz = _g.nxyz();
    const size_t niso = Isotope::niso;

    // Scalar types
    for (int xt = XSTF; xt <= XS3N; ++xt) {
        if (xt == XSDF || xt == XSRF) continue;
        auto        xtype = static_cast<XSTYPE>(xt);
        auto&       dst   = _xs[xtype];
        const auto& lmp   = _lmpx[xtype];
        const auto& mic   = _micx[xtype];

        for (int ig = 0; ig < ng; ++ig) {
            double val = lmp[ig * nxyz + l];
            for (size_t iso = 0; iso < niso; ++iso)
                val += mic[(iso * ng + ig) * nxyz + l] * _iden[iso * nxyz + l];
            dst[ig * nxyz + l] = val;
        }
    }

    // Scattering matrix
    for (int igs = 0; igs < ng; ++igs) {
        for (int ige = 0; ige < ng; ++ige) {
            double val = _lmpx.xssm[(igs * ng + ige) * nxyz + l];
            for (size_t iso = 0; iso < niso; ++iso)
                val += _micx.xssm[(iso * ng * ng + igs * ng + ige) * nxyz + l] * _iden[iso * nxyz + l];
            _xs.xssm[(igs * ng + ige) * nxyz + l] = val;
        }
    }

    // D = 1/(3*Σ_tr)
    for (int ig = 0; ig < ng; ++ig) {
        double tr               = _xs.xstf[ig * nxyz + l];
        _xs.xsdf[ig * nxyz + l] = (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;
    }

    // Removal = absorption + outgoing scatter
    for (int igs = 0; igs < ng; ++igs) {
        double rf = _xs.xsaf[igs * nxyz + l];
        for (int ige = 0; ige < ng; ++ige)
            rf += _xs.xssm[(igs * ng + ige) * nxyz + l];
        _xs.xsrf[igs * nxyz + l] = rf;
    }
}

// Update: fetch XS from the library, unpack into SoA, then reconstruct.
void XSSet::Update() {
    const int    nxyz = _g.nxyz();
    const size_t niso = Isotope::niso;
    const size_t ng   = static_cast<size_t>(_g.ng());

#pragma omp parallel for schedule(static) if (nxyz > OMP_THRESHOLD)
    for (int l = 0; l < nxyz; ++l) {
        // Thread-local scratch: allocated once, reused across iterations
        static thread_local CrossSection         tls_xs, tls_delta;
        static thread_local milk::Vector<double> tls_iden;

        const auto& model = _models[_comp[l]];

        if (UsesRodXS(l)) {
            FillRodNodeXS(l);
        } else {
            model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                                   0, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));

            unpackXS(tls_xs, l, ng, nxyz, niso);

            // Merge: overwrite etc isotopes (H, B, O) from library; keep depleted.
            for (size_t i = 0; i < Isotope::iI135 && i < niso; ++i)
                _iden[i * nxyz + l] = tls_iden[i];
        }
    }

    Reconstruct();
}

// Pre-compute node lookup and burnup-interpolated reference XS.

void XSSet::PrecomputeBranchCoefficients() {
    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    // 1. Build the per-node index table used by the hot update loop.
    for (int l = 0; l < nxyz; ++l) {
        const size_t mi     = _comp[l];
        const int    eff_ct = 0;

        // Reference depletion points (lo/hi burnup bracket)
        const int refr_ci = findCtype(_refr_ctyp[mi], eff_ct);
        if (refr_ci < 0)
            throw std::runtime_error("XSSet: missing unrodded reference depletion table");
        const auto& refr_burn  = _refr_burn[mi][refr_ci];
        const int   refr_lo_bi = findLoBurn(refr_burn, _burn[l]);
        const int   refr_hi_bi = findHiBurn(refr_burn, _burn[l]);
        if (refr_lo_bi < 0 || refr_hi_bi < 0)
            throw std::runtime_error("XSSet: empty unrodded reference burn table");
        _node_refr_lo[l] = static_cast<int>(_refr_base[mi] +
                                            static_cast<size_t>(refr_ci) * _refr_ctyp_stride[mi] +
                                            static_cast<size_t>(refr_lo_bi) * _refr_burn_stride[mi]);
        _node_refr_hi[l] = static_cast<int>(_refr_base[mi] +
                                            static_cast<size_t>(refr_ci) * _refr_ctyp_stride[mi] +
                                            static_cast<size_t>(refr_hi_bi) * _refr_burn_stride[mi]);

        // Delta polynomial entries per branch
        for (size_t b = 0; b < _num_delta_branches; ++b) {
            const int brch_ci = findCtype(_brch_ctyp[mi][b], eff_ct);
            if (brch_ci < 0) {
                _node_delta_lo[b][l]   = -1;
                _node_delta_hi[b][l]   = -1;
                _node_delta_frac[b][l] = 0.0;
                continue;
            }
            const auto& brch_burn  = _brch_burn[mi][b][brch_ci];
            const int   brch_lo_bi = findLoBurn(brch_burn, _burn[l]);
            const int   brch_hi_bi = findHiBurn(brch_burn, _burn[l]);
            if (brch_lo_bi < 0 || brch_hi_bi < 0) {
                _node_delta_lo[b][l]   = -1;
                _node_delta_hi[b][l]   = -1;
                _node_delta_frac[b][l] = 0.0;
                continue;
            }

            _node_delta_lo[b][l] = static_cast<int>(_brch_base[mi][b] +
                                                    static_cast<size_t>(brch_ci) * _brch_ctyp_stride[mi][b] +
                                                    static_cast<size_t>(brch_lo_bi) * _brch_burn_stride[mi][b]);
            _node_delta_hi[b][l] = static_cast<int>(_brch_base[mi][b] +
                                                    static_cast<size_t>(brch_ci) * _brch_ctyp_stride[mi][b] +
                                                    static_cast<size_t>(brch_hi_bi) * _brch_burn_stride[mi][b]);
            _node_delta_frac[b][l] =
                (brch_lo_bi == brch_hi_bi)
                    ? 0.0
                    : static_cast<double>(_burn[l] - brch_burn[brch_lo_bi]) /
                          static_cast<double>(brch_burn[brch_hi_bi] - brch_burn[brch_lo_bi]);
        }
    }

    // 2. Scatter reference data from the flat library cache into node SoA buffers.

    _ref_lmpx.fill(0.0);
    _ref_micx.fill(0.0);
    _ref_iden.fill(0.0);
    std::fill(_ref_tmod.begin(), _ref_tmod.end(), 0.0);
    std::fill(_ref_dmod.begin(), _ref_dmod.end(), 0.0);

    // Cache XSArraySet data pointers to avoid switch dispatch inside the loop
    const auto    lib_lmpx_ptrs = ScalarData(_lib_lmpx);
    const auto    lib_micx_ptrs = ScalarData(_lib_micx);
    const double* lib_lmpx_sm   = _lib_lmpx.xssm.data();
    const double* lib_micx_sm   = _lib_micx.xssm.data();
    const double* lib_iden      = _lib_iden.data();
    const double* lib_burn      = _lib_burn.data();
    const double* lib_wvfr      = _lib_wvfr.data();
    const double* lib_tmod      = _lib_tmod.data();
    const double* lib_dmod      = _lib_dmod.data();
    const double* lib_flux      = _lib_flux.data();
    const double* lib_chix      = _lib_chix.data();
    auto          ref_lmpx      = ScalarXS(_ref_lmpx);
    auto          ref_micx      = ScalarXS(_ref_micx);

#pragma omp parallel for schedule(static) if (nxyz > OMP_THRESHOLD)
    for (int l = 0; l < nxyz; ++l) {
        const int lo = _node_refr_lo[l];
        const int hi = _node_refr_hi[l];

        // Scatter reference lumped XS from the flat cache.
        for (int ig = 0; ig < ng; ++ig) {
            size_t dst_off = ig * nxyz + l;
            size_t src_off = lo * ng + ig;
            for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                if (xt == XSDF || xt == XSRF) continue;
                (*ref_lmpx[xt])[dst_off] = lib_lmpx_ptrs[xt][src_off];
            }
            for (int ige = 0; ige < ng; ++ige)
                _ref_lmpx.xssm[(ig * ng + ige) * nxyz + l] = lib_lmpx_sm[lo * ng * ng + ig * ng + ige];
        }

        // Scatter reference microscopic XS from the flat cache.
        for (size_t iso = 0; iso < niso; ++iso) {
            for (int ig = 0; ig < ng; ++ig) {
                size_t dst_off = (iso * ng + ig) * nxyz + l;
                size_t src_off = (static_cast<size_t>(lo) * niso + iso) * ng + ig;
                for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                    if (xt == XSDF || xt == XSRF) continue;
                    (*ref_micx[xt])[dst_off] = lib_micx_ptrs[xt][src_off];
                }
                for (int ige = 0; ige < ng; ++ige)
                    _ref_micx.xssm[(iso * ng * ng + ig * ng + ige) * nxyz + l] =
                        lib_micx_sm[(static_cast<size_t>(lo) * niso + iso) * ng * ng + ig * ng + ige];
            }
        }

        // Scatter reference isotope densities.
        for (size_t i = 0; i < niso; ++i)
            _ref_iden[i * nxyz + l] = lib_iden[lo * niso + i];

        _ref_wvfr[l]  = lib_wvfr[lo];
        _ref_tmod[l]  = lib_tmod[lo];
        _ref_dmod[l]  = lib_dmod[lo];
        _node_wvfr[l] = _ref_wvfr[l];

        for (int ig = 0; ig < ng; ++ig) {
            _ref_flux[static_cast<size_t>(ig) * nxyz + l] = lib_flux[lo * ng + ig];
            _ref_chix[static_cast<size_t>(ig) * nxyz + l] = lib_chix[lo * ng + ig];
        }

        // Interpolate between burnup brackets when the node sits between two reference points.
        if (lo != hi) {
            const double f = (_burn[l] / 1000.0 - lib_burn[lo]) / (lib_burn[hi] - lib_burn[lo]);

            for (int ig = 0; ig < ng; ++ig) {
                size_t dst_off = ig * nxyz + l;
                size_t lo_off  = lo * ng + ig;
                size_t hi_off  = hi * ng + ig;
                for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                    if (xt == XSDF || xt == XSRF) continue;
                    (*ref_lmpx[xt])[dst_off] += f * (lib_lmpx_ptrs[xt][hi_off] - lib_lmpx_ptrs[xt][lo_off]);
                }
                for (int ige = 0; ige < ng; ++ige)
                    _ref_lmpx.xssm[(ig * ng + ige) * nxyz + l] +=
                        f * (lib_lmpx_sm[hi * ng * ng + ig * ng + ige] - lib_lmpx_sm[lo * ng * ng + ig * ng + ige]);
            }

            for (size_t iso = 0; iso < niso; ++iso) {
                for (int ig = 0; ig < ng; ++ig) {
                    size_t dst_off = (iso * ng + ig) * nxyz + l;
                    size_t lo_off  = (static_cast<size_t>(lo) * niso + iso) * ng + ig;
                    size_t hi_off  = (static_cast<size_t>(hi) * niso + iso) * ng + ig;
                    for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                        if (xt == XSDF || xt == XSRF) continue;
                        (*ref_micx[xt])[dst_off] += f * (lib_micx_ptrs[xt][hi_off] - lib_micx_ptrs[xt][lo_off]);
                    }
                    for (int ige = 0; ige < ng; ++ige)
                        _ref_micx.xssm[(iso * ng * ng + ig * ng + ige) * nxyz + l] +=
                            f * (lib_micx_sm[(static_cast<size_t>(hi) * niso + iso) * ng * ng + ig * ng + ige] -
                                 lib_micx_sm[(static_cast<size_t>(lo) * niso + iso) * ng * ng + ig * ng + ige]);
                }
            }

            for (size_t i = 0; i < niso; ++i)
                _ref_iden[i * nxyz + l] += f * (lib_iden[hi * niso + i] - lib_iden[lo * niso + i]);
            _ref_tmod[l] += f * (lib_tmod[hi] - lib_tmod[lo]);
            _ref_dmod[l] += f * (lib_dmod[hi] - lib_dmod[lo]);

            for (int ig = 0; ig < ng; ++ig) {
                _ref_flux[static_cast<size_t>(ig) * nxyz + l] +=
                    f * (lib_flux[hi * ng + ig] - lib_flux[lo * ng + ig]);
                _ref_chix[static_cast<size_t>(ig) * nxyz + l] +=
                    f * (lib_chix[hi * ng + ig] - lib_chix[lo * ng + ig]);
            }
        }
    }

    // Overwrite non-depleted isotope densities from the reference state.
    for (int l = 0; l < nxyz; ++l) {
        if (l < static_cast<int>(_external_iden.size()) && _external_iden[l])
            continue;
        for (size_t i = 0; i < Isotope::iI135 && i < niso; ++i)
            _iden[i * nxyz + l] = _ref_iden[i * nxyz + l];
    }

    _simd_ready = true;
}

// Flat XS update helpers.

bool XSSet::UsesRodXS(int l) const {
    if (_g.rod_fraction(l) <= EPS) return false;

    const auto& model = _models[_comp[l]];
    if (_rod_node_segment_offset.size() == static_cast<size_t>(_g.nxyz() + 1)) {
        const int begin = _rod_node_segment_offset[static_cast<size_t>(l)];
        const int end   = _rod_node_segment_offset[static_cast<size_t>(l + 1)];
        for (int i = begin; i < end; ++i) {
            const int ctype = _rod_node_segment_ctype[static_cast<size_t>(i)];
            if (ctype != 0 && model._refr_dpts.count(ctype) != 0)
                return true;
        }
        return false;
    }

    const int ctype = _ctyp[l];
    return ctype != 0 && model._refr_dpts.count(ctype) != 0;
}

void XSSet::RestoreReferenceNode(int l) {
    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    _node_wvfr[l] = _ref_wvfr[l];

    auto ref_lmpx = ScalarXS(_ref_lmpx);
    auto ref_micx = ScalarXS(_ref_micx);
    auto lmpx     = ScalarXS(_lmpx);
    auto micx     = ScalarXS(_micx);

    for (int ig = 0; ig < ng; ++ig) {
        const size_t off = ig * nxyz + l;
        for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
            if (xt == XSDF || xt == XSRF) continue;
            (*lmpx[xt])[off] = (*ref_lmpx[xt])[off];
        }
        for (int ige = 0; ige < ng; ++ige) {
            const size_t sm_off = (ig * ng + ige) * nxyz + l;
            _lmpx.xssm[sm_off]  = _ref_lmpx.xssm[sm_off];
        }
    }

    for (size_t iso = 0; iso < niso; ++iso) {
        for (int ig = 0; ig < ng; ++ig) {
            const size_t off = (iso * ng + ig) * nxyz + l;
            for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
                if (xt == XSDF || xt == XSRF) continue;
                (*micx[xt])[off] = (*ref_micx[xt])[off];
            }
            for (int ige = 0; ige < ng; ++ige) {
                const size_t sm_off = (iso * ng * ng + ig * ng + ige) * nxyz + l;
                _micx.xssm[sm_off]  = _ref_micx.xssm[sm_off];
            }
        }
    }
}

void XSSet::ApplyBranchDeltaIdToNode(int l, int did, double x, double scale, bool clamp_x) {
    if (did < 0 || scale == 0.0) return;

    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    const auto& dinfo = _lib_deltas[did];
    int         base  = dinfo.coeff_base;
    int         nord  = dinfo.nord;

    if (clamp_x && dinfo.knot_count >= 2) {
        const double xmin = _lib_knots[dinfo.knot_offset];
        const double xmax = _lib_knots[dinfo.knot_offset + dinfo.knot_count - 1];
        x                 = std::clamp(x, xmin, xmax);
    }

    double xloc = x;

    if (dinfo.mode == 1) {
        const int nintervals = dinfo.nord / dinfo.ncoeff;
        int       interval   = nintervals - 1;
        for (int i = 0; i < nintervals - 1; ++i) {
            if (x < _lib_knots[dinfo.knot_offset + i + 1]) {
                interval = i;
                break;
            }
        }
        xloc = x - _lib_knots[dinfo.knot_offset + interval];
        base += interval * dinfo.ncoeff;
        nord = dinfo.ncoeff;
    }

    const auto    coeff_lmpx    = ScalarData(_lib_coeff_lmpx);
    const auto    coeff_micx    = ScalarData(_lib_coeff_micx);
    const double* coeff_lmpx_sm = _lib_coeff_lmpx.xssm.data();
    const double* coeff_micx_sm = _lib_coeff_micx.xssm.data();
    auto          lmpx          = ScalarXS(_lmpx);
    auto          micx          = ScalarXS(_micx);

    for (int ig = 0; ig < ng; ++ig) {
        const size_t dst_lmpx = ig * nxyz + l;

        for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
            if (xt == XSDF || xt == XSRF) continue;
            const double* cdata = coeff_lmpx[xt];
            double        val   = cdata[(base + nord - 1) * ng + ig];
            for (int p = nord - 2; p >= 0; --p)
                val = val * xloc + cdata[(base + p) * ng + ig];
            (*lmpx[xt])[dst_lmpx] += scale * val;
        }

        for (int ige = 0; ige < ng; ++ige) {
            double val = coeff_lmpx_sm[(base + nord - 1) * ng * ng + ig * ng + ige];
            for (int p = nord - 2; p >= 0; --p)
                val = val * xloc + coeff_lmpx_sm[(base + p) * ng * ng + ig * ng + ige];
            _lmpx.xssm[(ig * ng + ige) * nxyz + l] += scale * val;
        }
    }

    if (!_lib_has_coeff_micx) return;

    const size_t scalar_stride  = niso * static_cast<size_t>(ng);
    const size_t scatter_stride = niso * static_cast<size_t>(ng) * static_cast<size_t>(ng);
    for (size_t iso = 0; iso < niso; ++iso) {
        for (int ig = 0; ig < ng; ++ig) {
            const size_t elem     = iso * ng + ig;
            const size_t dst_micx = elem * nxyz + l;

            for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                if (xt == XSDF || xt == XSRF) continue;
                const double* cdata = coeff_micx[xt];
                double        val   = cdata[(base + nord - 1) * scalar_stride + elem];
                for (int p = nord - 2; p >= 0; --p)
                    val = val * xloc + cdata[(base + p) * scalar_stride + elem];
                (*micx[xt])[dst_micx] += scale * val;
            }

            for (int ige = 0; ige < ng; ++ige) {
                const size_t elem_sm = iso * ng * ng + ig * ng + ige;
                double       val     = coeff_micx_sm[(base + nord - 1) * scatter_stride + elem_sm];
                for (int p = nord - 2; p >= 0; --p)
                    val = val * xloc + coeff_micx_sm[(base + p) * scatter_stride + elem_sm];
                _micx.xssm[elem_sm * nxyz + l] += scale * val;
            }
        }
    }
}

void XSSet::ApplyBranchDeltaToNode(int l, int branch, double x, double scale) {
    if (branch < 0 || static_cast<size_t>(branch) >= _node_delta_lo.size()) return;
    const int lo = _node_delta_lo[branch][l];
    if (lo < 0 || scale == 0.0) return;

    const int    hi      = _node_delta_hi[branch][l];
    const double f       = _node_delta_frac[branch][l];
    const bool   clamp_x = branch >= BRANCH_HISTORY_BASE;
    ApplyBranchDeltaIdToNode(l, lo, x, scale * (1.0 - f), clamp_x);
    if (hi != lo)
        ApplyBranchDeltaIdToNode(l, hi, x, scale * f, clamp_x);
}

void XSSet::ApplyHistoryDeltasToNode(int l) {
    const size_t mi = _comp[l];
    if (mi >= _lib_history_corrections.size()) return;

    // Resolve the lo/hi burnup-bracketed delta-surface indices for a (branch, ctype) pair.
    auto resolveDelta = [&](int branch, int ctype, int burn_key, int& lo, int& hi, double& frac) {
        if (branch < 0 || static_cast<size_t>(branch) >= _brch_ctyp[mi].size())
            return false;
        const int brch_ci = findCtype(_brch_ctyp[mi][branch], ctype);
        if (brch_ci < 0)
            return false;
        const auto& burns = _brch_burn[mi][branch][brch_ci];
        const int   lo_bi = findLoBurn(burns, burn_key);
        const int   hi_bi = findHiBurn(burns, burn_key);
        if (lo_bi < 0 || hi_bi < 0)
            return false;
        lo   = static_cast<int>(_brch_base[mi][branch] +
                                static_cast<size_t>(brch_ci) * _brch_ctyp_stride[mi][branch] +
                                static_cast<size_t>(lo_bi) * _brch_burn_stride[mi][branch]);
        hi   = static_cast<int>(_brch_base[mi][branch] +
                                static_cast<size_t>(brch_ci) * _brch_ctyp_stride[mi][branch] +
                                static_cast<size_t>(hi_bi) * _brch_burn_stride[mi][branch]);
        frac = (lo_bi == hi_bi) ? 0.0
                                : static_cast<double>(burn_key - burns[lo_bi]) /
                                      static_cast<double>(burns[hi_bi] - burns[lo_bi]);
        return true;
    };

    const int    current_ctype    = UsesRodXS(l) ? _ctyp[l] : 0;
    const int    trajectory_ctype = (l < static_cast<int>(_history_ctyp.size())) ? _history_ctyp[l] : 0;
    const size_t stride           = static_cast<size_t>(_g.nxyz());
    const size_t node             = static_cast<size_t>(l);

    // Indicator-vector coordinate at the current node state and at the reference state.
    // rod_fluence is the only synthetic coordinate; every other entry is a real isotope density.
    auto histCur = [&](size_t iso) -> double {
        if (iso == Chiffon::Hv::ROD_FLU)
            return Chiffon::hvRodFluCoord(FineRodFluenceAverage(l, current_ctype));
        const size_t idx = iso * stride + node;
        return idx < _iden.size() ? _iden[idx] : 0.0;
    };
    auto histRef = [&](size_t iso) -> double {
        if (iso == Chiffon::Hv::ROD_FLU)
            return 0.0;
        const size_t idx = iso * stride + node;
        return idx < _ref_iden.size() ? _ref_iden[idx] : 0.0;
    };

    const int   delta_burn  = _burn[l];
    const auto& corrections = _lib_history_corrections[mi];
    for (size_t h = 0; h < corrections.size(); ++h) {
        const auto& history = corrections[h];
        const int   branch  = static_cast<int>(BRANCH_HISTORY_BASE + h);

        // Only the two runtime IISC kinds are applied here: CTYPE_INDEP_VEC is a
        // ctype-independent surface applied to every node, and RHST_UNIT is a current-ctype
        // surface applied only where the fuel carries rodded history. A ctype-keyed IISC
        // surface (legacy Hk::VEC, emitted when ctype_independent is false) is NOT handled by
        // this path and falls through the else below, so an applied IISC must be ctype-independent.
        int storage_ctype = 0;
        if (history.kind == Chiffon::Hk::CTYPE_INDEP_VEC) {
            storage_ctype = 0;
        } else if (history.kind == Chiffon::Hk::RHST_UNIT) {
            if (trajectory_ctype <= 0) continue;
            storage_ctype = current_ctype;
        } else {
            continue;
        }

        int    lo = -1, hi = -1;
        double frac = 0.0;
        if (!resolveDelta(branch, storage_ctype, delta_burn, lo, hi, frac))
            continue;

        const double x = Interpolator::EvalVectorTermFromAccessors(
            history.vector_isotopes, history.vector_powers, histCur, histRef);
        ApplyBranchDeltaIdToNode(l, lo, x, 1.0 - frac, true);
        if (hi != lo)
            ApplyBranchDeltaIdToNode(l, hi, x, frac, true);
    }
}

void XSSet::ApplyBranchDeltasToNode(int l) {
    // Unrodded nodes use the ordinary delta chain:
    // base XS + BPPM + TFUEL + DMOD + IISC/RHST history deltas.
    const double boron_dmod                  = BoronDmod(_g, _boron_dmod_average, l);
    const double x_vals[BRANCH_HISTORY_BASE] = {
        boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR,
        std::sqrt(_g.tful(l)),
        _g.dmod(l)};

    for (int branch = 0; branch < BRANCH_HISTORY_BASE; ++branch)
        ApplyBranchDeltaToNode(l, branch, x_vals[branch], 1.0);
    ApplyHistoryDeltasToNode(l);
}

void XSSet::FillRodNodeXS(int l) {
    // Rodded nodes start from the rodded reference surface and apply explicit
    // rod-material depletion before IISC/RHST history deltas are added by the
    // caller. Fine rod fluence is rod-material state, not fuel history memory.
    const int    nxyz  = _g.nxyz();
    const size_t ng    = static_cast<size_t>(_g.ng());
    const size_t niso  = Isotope::niso;
    const auto&  model = _models[_comp[l]];

    static thread_local CrossSection         tls_xs, tls_delta, tls_xs2;
    static thread_local milk::Vector<double> tls_iden, tls_iden2;

    int begin = 0;
    int end   = 0;
    if (_rod_node_segment_offset.size() == static_cast<size_t>(nxyz + 1)) {
        begin = _rod_node_segment_offset[static_cast<size_t>(l)];
        end   = _rod_node_segment_offset[static_cast<size_t>(l + 1)];
    }

    if (begin < end) {
        double rodded_frac = 0.0;
        for (int i = begin; i < end; ++i)
            rodded_frac += _rod_node_segment_fraction[static_cast<size_t>(i)];
        rodded_frac = std::clamp(rodded_frac, 0.0, 1.0);

        bool   has_xs        = false;
        double unrodded_frac = std::max(0.0, 1.0 - rodded_frac);
        if (unrodded_frac > EPS) {
            model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                                   0, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
            tls_xs *= unrodded_frac;
            has_xs = true;
        }

        for (int i = begin; i < end; ++i) {
            const double frac = _rod_node_segment_fraction[static_cast<size_t>(i)];
            if (frac <= EPS) continue;

            const int    input_ctype = _rod_node_segment_ctype[static_cast<size_t>(i)];
            const int    solve_ctype = (input_ctype != 0 && model._refr_dpts.count(input_ctype) != 0)
                                           ? input_ctype
                                           : 0;
            const double fluence     = FineRodFluenceAverage(l, input_ctype);
            if (!has_xs) {
                model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                                       solve_ctype, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
                model.ApplyRodDepletion(tls_xs, tls_delta, solve_ctype, fluence);
                tls_xs *= frac;
                has_xs = true;
            } else {
                model.FillCrossSection(tls_xs2, tls_iden2, tls_delta,
                                       solve_ctype, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
                model.ApplyRodDepletion(tls_xs2, tls_delta, solve_ctype, fluence);
                tls_xs2 *= frac;
                tls_xs += tls_xs2;
            }
        }

        if (!has_xs)
            model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                                   0, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
    } else {
        const int    ctype   = _ctyp[l];
        const double frac    = _g.rod_fraction(l);
        const double fluence = FineRodFluenceAverage(l, ctype);
        model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                               (frac >= 1.0 - EPS) ? ctype : 0,
                               _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
        if (frac >= 1.0 - EPS)
            model.ApplyRodDepletion(tls_xs, tls_delta, ctype, fluence);
        if (frac < 1.0 - EPS) {
            model.FillCrossSection(tls_xs2, tls_iden2, tls_delta,
                                   ctype, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
            model.ApplyRodDepletion(tls_xs2, tls_delta, ctype, fluence);
            tls_xs *= (1.0 - frac);
            tls_xs2 *= frac;
            tls_xs += tls_xs2;
        }
    }

    unpackXS(tls_xs, l, ng, nxyz, niso);

    for (size_t i = 0; i < Isotope::iI135 && i < niso; ++i)
        _iden[i * nxyz + l] = tls_iden[i];

    if (_g.dmod(l) > 1.0e-30)
        _node_wvfr[l] = tls_iden[Isotope::iO16] / (_g.dmod(l) * WATER_NUMBER_DENSITY);
}

void XSSet::RefreshLightIsotopes(int l) {
    const int    nxyz       = _g.nxyz();
    const double nH2O       = _g.dmod(l) * _node_wvfr[l] * WATER_NUMBER_DENSITY;
    const double boron_dmod = BoronDmod(_g, _boron_dmod_average, l);

    _iden[Isotope::iH1 * nxyz + l]  = 2.0 * nH2O;
    _iden[Isotope::iO16 * nxyz + l] = nH2O;
    _iden[Isotope::iB10 * nxyz + l] = boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR;
}

// Update flat XS arrays using pre-computed coefficients.

void XSSet::UpdateFlatXS(const XSUpdateOptions& options) {
    if (!_simd_ready) {
        Update();
        return;
    }

    const int    nxyz       = _g.nxyz();
    const int    ng         = _g.ng();
    const size_t niso       = Isotope::niso;
    const bool   all_nodes  = options.nodes.empty();
    const int    node_count = all_nodes ? nxyz : static_cast<int>(options.nodes.size());
    _boron_dmod_average     = FuelVolumeAverageDmod(_g);
    _history_dmod_average   = _boron_dmod_average;
    _history_tmod_average   = FuelVolumeAverageTmod(_g);

    if (options.restore_reference && all_nodes) {
        const size_t ngn  = static_cast<size_t>(ng) * static_cast<size_t>(nxyz);
        const size_t smsz = static_cast<size_t>(ng) * static_cast<size_t>(ng) * static_cast<size_t>(nxyz);

        std::copy(_ref_wvfr.begin(), _ref_wvfr.end(), _node_wvfr.begin());

        auto ref_lmpx = ScalarXS(_ref_lmpx);
        auto ref_micx = ScalarXS(_ref_micx);
        auto lmpx     = ScalarXS(_lmpx);
        auto micx     = ScalarXS(_micx);
        for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
            if (xt == XSDF || xt == XSRF) continue;
            CopyDoubles(ngn, ref_lmpx[xt]->data(), lmpx[xt]->data());
            CopyDoubles(niso * ngn, ref_micx[xt]->data(), micx[xt]->data());
        }
        CopyDoubles(smsz, _ref_lmpx.xssm.data(), _lmpx.xssm.data());
        CopyDoubles(niso * smsz, _ref_micx.xssm.data(), _micx.xssm.data());
    }

#pragma omp parallel for schedule(static) if (node_count > OMP_THRESHOLD)
    for (int i = 0; i < node_count; ++i) {
        const int l = all_nodes ? i : options.nodes[i];

        if (options.boron_difference) {
            if (UsesRodXS(l)) {
                FillRodNodeXS(l);
                ApplyHistoryDeltasToNode(l);
            } else {
                const double old_bppm   = (static_cast<size_t>(l) < options.old_bppm.size())
                                              ? options.old_bppm[l]
                                              : _g.bppm(l);
                const double boron_dmod = BoronDmod(_g, _boron_dmod_average, l);
                const double old_x      = boron_dmod * _node_wvfr[l] * old_bppm * BORON_DENSITY_FACTOR;
                const double new_x      = boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR;
                ApplyBranchDeltaToNode(l, BRANCH_BPPM, old_x, -1.0);
                ApplyBranchDeltaToNode(l, BRANCH_BPPM, new_x, 1.0);
                RefreshLightIsotopes(l);
            }
            continue;
        }

        if (options.restore_reference && !all_nodes)
            RestoreReferenceNode(l);

        if (UsesRodXS(l)) {
            FillRodNodeXS(l);
            ApplyHistoryDeltasToNode(l);
        } else {
            if (options.apply_bppm && options.apply_tful && options.apply_dmod) {
                ApplyBranchDeltasToNode(l);
            } else {
                if (options.apply_bppm) {
                    const double boron_dmod = BoronDmod(_g, _boron_dmod_average, l);
                    const double x          = boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR;
                    ApplyBranchDeltaToNode(l, BRANCH_BPPM, x, 1.0);
                }
                if (options.apply_tful)
                    ApplyBranchDeltaToNode(l, BRANCH_TFUL, std::sqrt(_g.tful(l)), 1.0);
                if (options.apply_dmod)
                    ApplyBranchDeltaToNode(l, BRANCH_DMOD, _g.dmod(l), 1.0);
                ApplyHistoryDeltasToNode(l);
            }
            RefreshLightIsotopes(l);
        }
    }

    if (all_nodes) {
        Reconstruct();
    } else {
#pragma omp parallel for schedule(static) if (node_count > OMP_THRESHOLD)
        for (int i = 0; i < node_count; ++i)
            ReconstructNode(static_cast<size_t>(options.nodes[i]));
    }
}

double XSSet::FineRodFluenceAverage(int l, int ctype) const {
    if (ctype <= 0 || _fine_rod_type.empty() || _fine_rod_fluence.empty())
        return 0.0;

    const int nxy = _g.nxy();
    const int div = std::max(1, _axial_rod_division);
    if (nxy <= 0 || l < 0 || l >= _g.nxyz())
        return 0.0;

    const int k    = l / nxy;
    const int l2d  = l % nxy;
    double    sum  = 0.0;
    int       nrod = 0;
    // Rod material fluence lives on the fine axial rod-state mesh. The current
    // single-rod history model uses the arithmetic mean over fine cells of the
    // requested ctype that overlap this coarse node.
    for (int m = 0; m < div; ++m) {
        const int idx = (k * div + m) * nxy + l2d;
        if (idx < 0 || idx >= static_cast<int>(_fine_rod_type.size()) ||
            idx >= static_cast<int>(_fine_rod_fluence.size()))
            continue;
        if (_fine_rod_type[static_cast<size_t>(idx)] != ctype)
            continue;
        // Count a cell as rodded for fluence purposes when it is at least half rodded,
        // matching the historical center-in-rod criterion (frac > 0.5).
        if (_fine_rod_frac[static_cast<size_t>(idx)] < 0.5)
            continue;
        sum += _fine_rod_fluence[static_cast<size_t>(idx)];
        ++nrod;
    }
    return nrod > 0 ? sum / static_cast<double>(nrod) : 0.0;
}

void XSSet::FillCuspingMacroXS(int l, int ctype, double fluence,
                               std::vector<double>& scalar,
                               std::vector<double>& scatter) const {
    const int ng = _g.ng();
    scalar.assign(static_cast<size_t>(ng) * N_XS_SCALAR, 0.0);
    scatter.assign(static_cast<size_t>(ng) * static_cast<size_t>(ng), 0.0);

    const auto& model     = _models[_comp[l]];
    const int   solve_ctp = (ctype != 0 && model._refr_dpts.count(ctype) != 0) ? ctype : 0;

    static thread_local CrossSection         tls_xs, tls_delta;
    static thread_local milk::Vector<double> tls_iden;
    model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                           solve_ctp, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
    model.ApplyRodDepletion(tls_xs, tls_delta, solve_ctp, fluence);

    for (int ig = 0; ig < ng; ++ig) {
        for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt)
            scalar[static_cast<size_t>(ig) * N_XS_SCALAR + static_cast<size_t>(xt)] =
                tls_xs.maxs(ig, static_cast<XSTYPE>(xt));
        for (int ige = 0; ige < ng; ++ige)
            scatter[static_cast<size_t>(ig) * static_cast<size_t>(ng) + static_cast<size_t>(ige)] =
                tls_xs.maxssm(ig, ige);
    }
}

int XSSet::RodCTypeAtDistance(const RodGroup& group, double distance_from_tip) const {
    if (distance_from_tip < 0.0)
        return 0;
    if (group.ctype_segments.empty() || group.length_segments.empty())
        return group.ctype;

    double remaining = distance_from_tip;
    for (int i = static_cast<int>(group.ctype_segments.size()) - 1; i >= 0; --i) {
        const double segment_length = group.length_segments[static_cast<size_t>(i)];
        if (remaining <= segment_length + 1.0e-10)
            return group.ctype_segments[static_cast<size_t>(i)];
        remaining -= segment_length;
    }
    return group.ctype_segments.front();
}

double XSSet::RodTotalLength(const RodGroup& group) const {
    if (group.length_segments.empty())
        return std::numeric_limits<double>::infinity();

    double total = 0.0;
    for (double length : group.length_segments)
        total += std::max(0.0, length);
    return total;
}

void XSSet::ApplyRodCuspingStencil(int tip_l, double reigv,
                                   const AxialTransverseLeakageView& leakage,
                                   std::vector<int>&                 touched_nodes) {
    const int nxy = _g.nxy();
    const int nz  = _g.nz();
    const int ng  = _g.ng();
    const int div = std::max(1, _axial_rod_division);

    const int tip_k = tip_l / nxy;
    const int l2d   = tip_l % nxy;
    if (tip_k <= _g.kbc() || tip_k >= _g.kec() - 1)
        return;
    if (tip_k <= 0 || tip_k >= nz - 1)
        return;

    const int coarse_l[3] = {
        (tip_k - 1) * nxy + l2d,
        tip_k * nxy + l2d,
        (tip_k + 1) * nxy + l2d};
    for (int i = 0; i < 3; ++i) {
        if (!_g.IsFuel(coarse_l[i]))
            return;
    }

    bool has_rod_fine_cell = false;
    for (int c = 0; c < 3 && !has_rod_fine_cell; ++c) {
        for (int m = 0; m < div; ++m) {
            const int fine_idx = ((tip_k - 1 + c) * div + m) * nxy + l2d;
            if (fine_idx >= 0 &&
                fine_idx < static_cast<int>(_fine_rod_type.size()) &&
                _fine_rod_type[static_cast<size_t>(fine_idx)] != 0) {
                has_rod_fine_cell = true;
                break;
            }
        }
    }
    if (!has_rod_fine_cell)
        return;

    std::vector<int>                 coarse_ctypes[3];
    std::vector<std::vector<double>> coarse_scalar[3];
    std::vector<std::vector<double>> coarse_scatter[3];
    for (int c = 0; c < 3; ++c) {
        coarse_ctypes[c].push_back(0);
        for (int m = 0; m < div; ++m) {
            const int fine_idx = ((tip_k - 1 + c) * div + m) * nxy + l2d;
            if (fine_idx < 0 || fine_idx >= static_cast<int>(_fine_rod_type.size()))
                continue;
            const int ctype = _fine_rod_type[static_cast<size_t>(fine_idx)];
            if (std::find(coarse_ctypes[c].begin(), coarse_ctypes[c].end(), ctype) == coarse_ctypes[c].end())
                coarse_ctypes[c].push_back(ctype);
        }
        coarse_scalar[c].resize(coarse_ctypes[c].size());
        coarse_scatter[c].resize(coarse_ctypes[c].size());
        for (int s = 0; s < static_cast<int>(coarse_ctypes[c].size()); ++s) {
            const double fluence = FineRodFluenceAverage(coarse_l[c], coarse_ctypes[c][s]);
            FillCuspingMacroXS(coarse_l[c], coarse_ctypes[c][s], fluence,
                               coarse_scalar[c][s], coarse_scatter[c][s]);
        }
    }

    const int           fine_count = 3 * div;
    std::vector<double> fine_h(static_cast<size_t>(fine_count), 0.0);
    std::vector<int>    fine_ctype(static_cast<size_t>(fine_count), 0);
    std::vector<double> fine_frac(static_cast<size_t>(fine_count), 0.0);
    std::vector<double> fine_scalar(static_cast<size_t>(fine_count) * static_cast<size_t>(ng) * N_XS_SCALAR, 0.0);
    std::vector<double> fine_scatter(static_cast<size_t>(fine_count) *
                                         static_cast<size_t>(ng) * static_cast<size_t>(ng),
                                     0.0);

    for (int c = 0; c < 3; ++c) {
        const int    lk = coarse_l[c];
        const double h  = _g.hmesh(ZDIR, lk);
        for (int m = 0; m < div; ++m) {
            const int f        = c * div + m;
            int       ctype    = 0;
            double    frac     = 0.0;
            const int fine_idx = ((tip_k - 1 + c) * div + m) * nxy + l2d;
            if (fine_idx >= 0 && fine_idx < static_cast<int>(_fine_rod_type.size())) {
                ctype = _fine_rod_type[static_cast<size_t>(fine_idx)];
                frac  = _fine_rod_frac[static_cast<size_t>(fine_idx)];
            }
            int state = 0;
            for (int s = 0; s < static_cast<int>(coarse_ctypes[c].size()); ++s) {
                if (coarse_ctypes[c][s] == ctype) {
                    state = s;
                    break;
                }
            }
            fine_h[f]     = h / static_cast<double>(div);
            fine_ctype[f] = ctype;
            fine_frac[f]  = frac;
            // Volume-weight the boundary cell (tip inside): rodded weight w blends the rodded
            // state with unrodded (state 0) so the fine-mesh XS is continuous in tip position.
            const double w = (ctype != 0) ? std::clamp(frac, 0.0, 1.0) : 0.0;

            for (int ig = 0; ig < ng; ++ig) {
                const size_t fine_group =
                    static_cast<size_t>(f) * static_cast<size_t>(ng) + static_cast<size_t>(ig);
                for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                    const size_t gx = static_cast<size_t>(ig) * N_XS_SCALAR + static_cast<size_t>(xt);
                    fine_scalar[fine_group * N_XS_SCALAR + static_cast<size_t>(xt)] =
                        w * coarse_scalar[c][state][gx] + (1.0 - w) * coarse_scalar[c][0][gx];
                }
                for (int ige = 0; ige < ng; ++ige) {
                    const size_t gs = static_cast<size_t>(ig) * static_cast<size_t>(ng) + static_cast<size_t>(ige);
                    fine_scatter[fine_group * static_cast<size_t>(ng) + static_cast<size_t>(ige)] =
                        w * coarse_scatter[c][state][gs] + (1.0 - w) * coarse_scatter[c][0][gs];
                }
            }
        }
    }

    const int           matrix_size = fine_count * ng;
    std::vector<double> a(static_cast<size_t>(matrix_size) * static_cast<size_t>(matrix_size), 0.0);
    std::vector<double> rhs(static_cast<size_t>(matrix_size), 0.0);

    for (int f = 0; f < fine_count; ++f) {
        const int coarse_idx = f / div;
        const int lk         = coarse_l[coarse_idx];
        for (int ig = 0; ig < ng; ++ig) {
            const int    row        = f * ng + ig;
            const size_t fine_group = static_cast<size_t>(f) * static_cast<size_t>(ng) + static_cast<size_t>(ig);
            a[static_cast<size_t>(row) * matrix_size + row] += fine_scalar[fine_group * N_XS_SCALAR + XSRF];

            for (int srcg = 0; srcg < ng; ++srcg)
                a[static_cast<size_t>(row) * matrix_size + f * ng + srcg] -=
                    fine_scatter[(static_cast<size_t>(f) * static_cast<size_t>(ng) + static_cast<size_t>(srcg)) *
                                     static_cast<size_t>(ng) +
                                 static_cast<size_t>(ig)];

            double fiss_src = 0.0;
            for (int srcg = 0; srcg < ng; ++srcg)
                fiss_src += fine_scalar[(static_cast<size_t>(f) * static_cast<size_t>(ng) +
                                         static_cast<size_t>(srcg)) *
                                            N_XS_SCALAR +
                                        XSNF] *
                            _g.Phif()[lk * ng + srcg];
            const int    local_m = f % div;
            const double zeta    = 2.0 * (static_cast<double>(local_m) + 0.5) /
                                    static_cast<double>(div) -
                                1.0;
            rhs[row] = reigv * chif(ig, lk) * fiss_src - leakage.value(lk, ig, ng, zeta);
        }
    }

    for (int f = 0; f < fine_count - 1; ++f) {
        for (int ig = 0; ig < ng; ++ig) {
            const size_t left_group =
                static_cast<size_t>(f) * static_cast<size_t>(ng) + static_cast<size_t>(ig);
            const size_t right_group =
                static_cast<size_t>(f + 1) * static_cast<size_t>(ng) + static_cast<size_t>(ig);
            const double dl    = fine_scalar[left_group * N_XS_SCALAR + XSDF];
            const double dr    = fine_scalar[right_group * N_XS_SCALAR + XSDF];
            const double bl    = (fine_h[f] > 0.0) ? dl / fine_h[f] : 0.0;
            const double br    = (fine_h[f + 1] > 0.0) ? dr / fine_h[f + 1] : 0.0;
            const double dtil  = (bl + br > 1.0e-30) ? 2.0 * bl * br / (bl + br) : 0.0;
            const int    left  = f * ng + ig;
            const int    right = (f + 1) * ng + ig;

            a[static_cast<size_t>(left) * matrix_size + left] += dtil / fine_h[f];
            a[static_cast<size_t>(left) * matrix_size + right] -= dtil / fine_h[f];
            a[static_cast<size_t>(right) * matrix_size + right] += dtil / fine_h[f + 1];
            a[static_cast<size_t>(right) * matrix_size + left] -= dtil / fine_h[f + 1];
        }
    }

    // §9.3 option-2 closure: the two outer boundary-cell rows become node-average-flux
    // constraints for the outer coarse nodes (R, U), taken from the previous CMFD solution.
    for (int outer = 0; outer < 2; ++outer) {
        const int    node_c = (outer == 0) ? 0 : 2;
        const int    bnd_f  = (outer == 0) ? 0 : fine_count - 1;
        const int    lk     = coarse_l[node_c];
        const double hnode  = _g.hmesh(ZDIR, lk);
        for (int ig = 0; ig < ng; ++ig) {
            const int row = bnd_f * ng + ig;
            for (int col = 0; col < matrix_size; ++col)
                a[static_cast<size_t>(row) * matrix_size + col] = 0.0;
            for (int m = 0; m < div; ++m) {
                const int f                                               = node_c * div + m;
                a[static_cast<size_t>(row) * matrix_size + (f * ng + ig)] = fine_h[f];
            }
            rhs[row] = _g.Phif()[lk * ng + ig] * hnode;
        }
    }

    if (!SolveDenseLinearSystem(a, rhs, matrix_size))
        return;

    double max_flux = 0.0;
    for (double v : rhs)
        max_flux = std::max(max_flux, v);
    if (max_flux <= 0.0 || !std::isfinite(max_flux))
        return;
    const double flux_floor = max_flux * 1.0e-12;

    // The fine-mesh FDM flux is used ONLY to update the cross-section weighting factor (alpha,
    // Eq. 9.5) below. The fine-mesh interface currents are NOT injected into the CMFD d-hat: the
    // homogenized node's nonlinear coupling is left to the regular nodal/CMFD update, which keeps
    // eigenvalue convergence stable (forcing the FDM current into d-hat runs away catastrophically).
    for (int c = 0; c < 3; ++c) {
        const int lk = coarse_l[c];
        if (std::find(touched_nodes.begin(), touched_nodes.end(), lk) == touched_nodes.end())
            touched_nodes.push_back(lk);

        std::vector<double> alpha(coarse_ctypes[c].size() * static_cast<size_t>(ng), 0.0);
        for (int ig = 0; ig < ng; ++ig) {
            double denom = 0.0;
            for (int m = 0; m < div; ++m) {
                const int    f   = c * div + m;
                const double phi = std::max(rhs[f * ng + ig], flux_floor);
                denom += phi * fine_h[f];
            }
            if (denom <= 1.0e-30)
                continue;

            for (int m = 0; m < div; ++m) {
                const int ctype = fine_ctype[static_cast<size_t>(c * div + m)];
                if (ctype == 0)
                    continue;

                int state = 0;
                for (int s = 1; s < static_cast<int>(coarse_ctypes[c].size()); ++s) {
                    if (coarse_ctypes[c][s] == ctype) {
                        state = s;
                        break;
                    }
                }
                if (state == 0)
                    continue;

                const int    f   = c * div + m;
                const double phi = std::max(rhs[f * ng + ig], flux_floor);
                // Boundary cell is only partially rodded: weight its flux-volume by frac so the
                // rodded weight (alpha) is continuous in tip position.
                const double w = std::clamp(fine_frac[static_cast<size_t>(f)], 0.0, 1.0);
                alpha[static_cast<size_t>(state) * static_cast<size_t>(ng) + static_cast<size_t>(ig)] +=
                    w * phi * fine_h[f] / denom;
            }

            for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                if (xt == XSDF || xt == XSRF)
                    continue;
                const size_t group_base = static_cast<size_t>(ig) * N_XS_SCALAR + static_cast<size_t>(xt);
                double       new_xs     = coarse_scalar[c][0][group_base];
                for (int s = 1; s < static_cast<int>(coarse_ctypes[c].size()); ++s) {
                    const double astate =
                        std::clamp(alpha[static_cast<size_t>(s) * static_cast<size_t>(ng) +
                                         static_cast<size_t>(ig)],
                                   0.0, 1.0);
                    new_xs += astate * (coarse_scalar[c][s][group_base] - coarse_scalar[c][0][group_base]);
                }
                const size_t xs_idx = static_cast<size_t>(ig) * _g.nxyz() + lk;
                const double old_xs = _xs[static_cast<XSTYPE>(xt)][xs_idx];
                _xs[static_cast<XSTYPE>(xt)][xs_idx] =
                    _rod_cusping_relaxation * new_xs +
                    (1.0 - _rod_cusping_relaxation) * old_xs;
            }

            for (int ige = 0; ige < ng; ++ige) {
                const size_t scatter_idx =
                    static_cast<size_t>(ig) * static_cast<size_t>(ng) + static_cast<size_t>(ige);
                double new_xs = coarse_scatter[c][0][scatter_idx];
                for (int s = 1; s < static_cast<int>(coarse_ctypes[c].size()); ++s) {
                    const double astate =
                        std::clamp(alpha[static_cast<size_t>(s) * static_cast<size_t>(ng) +
                                         static_cast<size_t>(ig)],
                                   0.0, 1.0);
                    new_xs += astate * (coarse_scatter[c][s][scatter_idx] - coarse_scatter[c][0][scatter_idx]);
                }
                const size_t xs_idx = (ig * ng + ige) * _g.nxyz() + lk;
                const double old_xs = _xs.xssm[xs_idx];
                _xs.xssm[xs_idx] =
                    _rod_cusping_relaxation * new_xs +
                    (1.0 - _rod_cusping_relaxation) * old_xs;
            }
        }

        for (int ig = 0; ig < ng; ++ig) {
            const double tr               = _xs.xstf[ig * _g.nxyz() + lk];
            _xs.xsdf[ig * _g.nxyz() + lk] = (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;

            double rf = _xs.xsaf[ig * _g.nxyz() + lk];
            for (int ige = 0; ige < ng; ++ige)
                rf += _xs.xssm[(ig * ng + ige) * _g.nxyz() + lk];
            _xs.xsrf[ig * _g.nxyz() + lk] = rf;
        }
    }
}

void XSSet::ResetCuspingNodesToBase(const std::vector<int>& nodes) {
    const int ng   = _g.ng();
    const int nxyz = _g.nxyz();
    if (static_cast<int>(_cusping_base_snapshot.size()) < nxyz)
        _cusping_base_snapshot.resize(static_cast<size_t>(nxyz));

    auto sig_of = [&](int l, double(&sig)[7]) {
        sig[0] = _burn[l];
        sig[1] = _g.bppm(l);
        sig[2] = _g.tful(l);
        sig[3] = _g.dmod(l);
        sig[4] = _g.rod_fraction(l);
        sig[5] = static_cast<double>(_ctyp[l]);
        sig[6] = FineRodFluenceAverage(l, _ctyp[l]);
    };

    std::vector<int> miss;
    for (int l : nodes) {
        CuspingBaseSnapshot& snap = _cusping_base_snapshot[static_cast<size_t>(l)];
        double               sig[7];
        sig_of(l, sig);
        bool hit = snap.valid;
        for (int i = 0; i < 7 && hit; ++i)
            if (snap.sig[i] != sig[i])
                hit = false;
        if (!hit) {
            miss.push_back(l);
            continue;
        }
        for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt)
            for (int ig = 0; ig < ng; ++ig)
                _xs[static_cast<XSTYPE>(xt)][static_cast<size_t>(ig) * nxyz + l] =
                    snap.scalar[static_cast<size_t>(xt) * ng + ig];
        for (int ig = 0; ig < ng; ++ig)
            for (int ige = 0; ige < ng; ++ige)
                _xs.xssm[(static_cast<size_t>(ig) * ng + ige) * nxyz + l] =
                    snap.scatter[static_cast<size_t>(ig) * ng + ige];
    }

    if (miss.empty())
        return;

    XSUpdateOptions options;
    options.nodes = miss;
    UpdateFlatXS(options);
    for (int l : miss) {
        CuspingBaseSnapshot& snap = _cusping_base_snapshot[static_cast<size_t>(l)];
        snap.scalar.resize(N_XS_SCALAR * static_cast<size_t>(ng));
        snap.scatter.resize(static_cast<size_t>(ng) * static_cast<size_t>(ng));
        for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt)
            for (int ig = 0; ig < ng; ++ig)
                snap.scalar[static_cast<size_t>(xt) * ng + ig] =
                    _xs[static_cast<XSTYPE>(xt)][static_cast<size_t>(ig) * nxyz + l];
        for (int ig = 0; ig < ng; ++ig)
            for (int ige = 0; ige < ng; ++ige)
                snap.scatter[static_cast<size_t>(ig) * ng + ige] =
                    _xs.xssm[(static_cast<size_t>(ig) * ng + ige) * nxyz + l];
        sig_of(l, snap.sig);
        snap.valid = true;
    }
}

bool XSSet::ApplyRodCusping(double eigv, const AxialTransverseLeakageView& leakage) {
    if (_axial_rod_division <= 0 || eigv <= 1.0e-20)
        return false;

    const int    nxyz  = _g.nxyz();
    const double reigv = 1.0 / eigv;

    // Accumulating damped fixed-point. The blend in ApplyRodCuspingStencil is
    //   _xs = relax*full_cusp + (1-relax)*old_xs,   old_xs = current _xs.
    // The previous implementation reset every cusped node back to base at the TOP of this
    // function, so old_xs was always base and the iteration's fixed point was
    //   _xs* = base + relax*(full_cusp - base)
    // i.e. only a `relax` FRACTION of the cusp was ever applied (relax=0.1 -> 10% cusp). That
    // made the converged cross-section, and hence the critical rod position, depend on relax.
    // Instead we keep the previous cusped _xs as old_xs for nodes that are STILL cusped, so the
    // recursion is x_{n+1}=relax*full+(1-relax)*x_n whose fixed point is x=full_cusp regardless
    // of relax (relax now only sets the convergence rate). We must still un-cusp nodes that LEFT
    // the cusped set. The cusped set is the 3-node stencil neighbourhood recorded in
    // _rod_cusping_nodes_scratch (not just the partial tip nodes), so "left" is membership-based.
    std::vector<int> prev_scratch;
    prev_scratch.swap(_rod_cusping_nodes_scratch);

    for (int l = 0; l < nxyz; ++l) {
        const double frac = _g.rod_fraction(l);
        if (frac > EPS && frac < 1.0 - EPS)
            ApplyRodCuspingStencil(l, reigv, leakage, _rod_cusping_nodes_scratch);
    }

    if (!prev_scratch.empty()) {
        std::vector<int> left;
        for (int l : prev_scratch)
            if (std::find(_rod_cusping_nodes_scratch.begin(),
                          _rod_cusping_nodes_scratch.end(), l) == _rod_cusping_nodes_scratch.end())
                left.push_back(l);
        if (!left.empty())
            ResetCuspingNodesToBase(left);
    }

    return !prev_scratch.empty() || !_rod_cusping_nodes_scratch.empty();
}

// Norm factor

double XSSet::NormFactor(double power, const XSArraySet& xs_arr, const double* flux) const {
    const int ng   = _g.ng();
    const int nxyz = _g.nxyz();

    double reaction_sum = 0.0;
#pragma omp parallel for reduction(+ : reaction_sum) schedule(static) if (nxyz > rasbery_omp_gate)
    for (int l = 0; l < nxyz; ++l)
        for (int ig = 0; ig < ng; ++ig)
            reaction_sum += xs_arr.xskf[ig * nxyz + l] * flux[l * ng + ig] * _g.vol(l);

    return 1.0e6 * power / reaction_sum;
}

double XSSet::NormFactor(double power) const {
    return NormFactor(power, _xs, _g.Phif());
}

double XSSet::CoreHeavyMetalMassKg() const {
    const int nxyz   = _g.nxyz();
    double    mass_g = 0.0;
    for (int l = 0; l < nxyz; ++l)
        mass_g += NodeHeavyMetalMassGrams(l);
    return mass_g * 1.0e-3;
}

double XSSet::NodeHeavyMetalMassGrams(int l) const {
    if (l < 0 || l >= _g.nxyz() || !_g.IsFuel(l))
        return 0.0;

    const size_t mi = _comp[l];
    if (mi >= _lib_model_volu.size() || mi >= _lib_model_hmas.size())
        return 0.0;
    if (_lib_model_volu[mi] <= 0.0)
        return 0.0;

    return (_g.vol(l) / _lib_model_volu[mi]) * _lib_model_hmas[mi];
}

// Burnup update

void XSSet::UpdateBurnup(double dt, double power) {
    const int     ng   = _g.ng();
    const int     nxyz = _g.nxyz();
    double* const flux = _g.Phif();

    const double norm_factor = NormFactor(power);

#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
    for (int l = 0; l < nxyz; ++l) {
        double burn       = 0.0;
        double scalarFlux = 0.0;
        for (int ig = 0; ig < ng; ++ig) {
            const double normalizedFlux = flux[l * ng + ig] * norm_factor;
            burn += _xs.xskf[ig * nxyz + l] * normalizedFlux * _g.vol(l) * dt;
            scalarFlux += normalizedFlux;
        }

        if (burn >= 1.0e-10) {
            const double dfac               = 8.64e7 * (_g.vol(l) / _lib_model_volu[_comp[l]]) * _lib_model_hmas[_comp[l]];
            const double burn_key_increment = burn / dfac * 1000.0;
            _burn[l] += static_cast<int>(burn_key_increment + 0.5);
            const bool   currentRod  = UsesRodXS(l);
            const double rodFraction = currentRod ? std::clamp(_g.rod_fraction(l), 0.0, 1.0) : 0.0;
            if (currentRod)
                _history_ctyp[l] = _ctyp[l];
            if (currentRod && rodFraction > 0.0 && scalarFlux > 0.0 && std::isfinite(scalarFlux))
                _rodded_fluence[l] += scalarFlux * dt * rodFraction;
        }
    }

    PrecomputeBranchCoefficients();
    UpdateFlatXS();
}

// Depletion (moved from Chiffon::DepletionSolver)

static double FluxScale(const double* flux, int ng) {
    double sum = 0.0;
    for (int ig = 0; ig < ng; ++ig)
        sum += flux[ig];
    return sum * 1.0e-24;
}

void XSSet::BuildTransitionMatrix(const std::vector<double>& cond, double sumflux,
                                  milk::Matrix<double>& mat) const {
    using namespace Isotope;

    mat = depDecay; // niso × niso (H/B/O rows are zero)

    for (size_t p = 0; p < niso; ++p) {
        double xsaf_val = cond[p * N_XS_SCALAR + XSAF];
        double xsff_val = cond[p * N_XS_SCALAR + XSFF];
        double xs2n_val = cond[p * N_XS_SCALAR + XS2N];
        double xs3n_val = cond[p * N_XS_SCALAR + XS3N];
        double xscf     = xsaf_val - xsff_val + xs2n_val + 2.0 * xs3n_val;

        mat(p, p) -= xsaf_val * sumflux;

        for (size_t d = 0; d < niso; ++d) {
            if (d == p) continue;
            const double topo = depTrans(d, p);
            if (topo == 0.0) continue;

            // Actinide parent → non-actinide daughter: fission yield (σ_f)
            // Otherwise: capture chain (σ_c)
            bool isActinide = (p >= iAcFirst && p <= iAcLast);
            bool dIsNonAc   = (d < iAcFirst || d > iAcLast);
            if (isActinide && dIsNonAc) {
                mat(d, p) += topo * xsff_val * sumflux;
            } else {
                mat(d, p) += topo * xscf * sumflux;
            }
        }
    }

    // (n,2n) special cases: U-235→U-234, U-238→Np-237
    mat(iU234, iU235) += cond[iU235 * N_XS_SCALAR + XS2N] * sumflux;
    mat(iNp237, iU238) += cond[iU238 * N_XS_SCALAR + XS2N] * sumflux;
}

/// Apply Xe-135 equilibrium overwrite on the isotope density vector.
static void ApplyXeEquilibrium(milk::Vector<double>& iden, const std::vector<double>& cond,
                               double sumflux) {
    using namespace Chiffon::Isotope;
    constexpr double lambdaI     = 2.930607e-05;
    constexpr double lambdaXe    = 2.106574e-05;
    constexpr double lambdaXem   = 7.555561e-04;
    constexpr double brItoXe135m = 1.650900e-01;

    double fissSourceI = 0.0, fissSourceXe = 0.0;
    for (size_t j = iAcFirst; j <= iAcLast; ++j) {
        double xsff  = cond[j * N_XS_SCALAR + XSFF];
        double fRate = iden[j] * xsff * sumflux;
        fissSourceI += fRate * depTrans(iI135, j);
        fissSourceXe += fRate * depTrans(iXe135, j);
    }

    double sigaXe = cond[iXe135 * N_XS_SCALAR + XSAF] * sumflux;
    double Ieq    = fissSourceI / lambdaI;
    double Xeeq   = (lambdaI * Ieq + fissSourceXe) / (lambdaXe + sigaXe);

    iden[iI135]   = Ieq;
    iden[iXe135]  = Xeeq;
    iden[iXe135m] = brItoXe135m * lambdaI * Ieq / lambdaXem;
}

void XSSet::DepleteNode(DepletionWorkspace& ws, size_t l,
                        const double* abs_flux, size_t ngrp, double dt, bool xe_transient) {
    using namespace Isotope;
    if (depDecay.size() == 0) return;

    const size_t nxyz_val = static_cast<size_t>(_g.nxyz());

    const size_t condensed_size = niso * N_XS_SCALAR;
    if (ws.condensed.size() < condensed_size) ws.condensed.resize(condensed_size, 0.0);

    double raw_sumflux = 0.0;
    for (size_t ig = 0; ig < ngrp; ++ig)
        raw_sumflux += abs_flux[ig];
    const double invflux = (raw_sumflux > 0.0) ? 1.0 / raw_sumflux : 0.0;

    const double* mic_ptrs[N_XS_SCALAR] = {
        _micx.xstf.data(), _micx.xsdf.data(), _micx.xsaf.data(), _micx.xsff.data(),
        _micx.xsnf.data(), _micx.xskf.data(), _micx.xssf.data(), _micx.xsrf.data(),
        _micx.fyld.data(), _micx.xs2n.data(), _micx.xs3n.data()};

    if (ngrp == 2) {
        const double f0 = abs_flux[0] * invflux;
        const double f1 = abs_flux[1] * invflux;
        for (size_t iso = 0; iso < niso; ++iso) {
            const size_t base0 = (iso * 2) * nxyz_val + l;
            const size_t base1 = (iso * 2 + 1) * nxyz_val + l;
            double*      dst   = ws.condensed.data() + iso * N_XS_SCALAR;
            for (size_t xt = 0; xt < N_XS_SCALAR; ++xt)
                dst[xt] = mic_ptrs[xt][base0] * f0 + mic_ptrs[xt][base1] * f1;
        }
    } else {
        for (size_t iso = 0; iso < niso; ++iso) {
            double* dst = ws.condensed.data() + iso * N_XS_SCALAR;
            for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
                double sum = 0.0;
                for (size_t ig = 0; ig < ngrp; ++ig)
                    sum += mic_ptrs[xt][(iso * ngrp + ig) * nxyz_val + l] * abs_flux[ig];
                dst[xt] = sum * invflux;
            }
        }
    }

    double sumflux = FluxScale(abs_flux, static_cast<int>(ngrp));

    // Load full isotope density vector from SoA
    for (size_t i = 0; i < niso; ++i)
        ws.iden[i] = _iden[i * nxyz_val + l];

    // Build & solve (niso × niso)
    BuildTransitionMatrix(ws.condensed, sumflux, ws.matrix);
    milk::Solver<double>::solveBatemanCRAM(ws.matrix, ws.iden, dt, ws.iden, ws.cram, CRAM_ORDER, iI135);

    // Xe equilibrium overwrite
    if (!xe_transient)
        ApplyXeEquilibrium(ws.iden, ws.condensed, sumflux);

    // Store back (skip H, B, O — overwritten by library in Update)
    for (size_t i = iI135; i < niso; ++i)
        _iden[i * nxyz_val + l] = ws.iden[i];
}

void XSSet::Deplete(double dt, double power, bool xe_transient) {
    const int     ng   = _g.ng();
    const int     nxyz = _g.nxyz();
    double* const flux = _g.Phif();

    const double norm_factor = NormFactor(power);

#pragma omp parallel if (nxyz > OMP_THRESHOLD)
    {
        static thread_local DepletionWorkspace  ws_tls;
        static thread_local std::vector<double> abs_flux_tls;

        ws_tls.ensure(Isotope::niso);
        if (abs_flux_tls.size() != static_cast<size_t>(ng))
            abs_flux_tls.resize(ng);

#pragma omp for schedule(dynamic, 8)
        for (int l = 0; l < nxyz; ++l) {
            for (int ig = 0; ig < ng; ++ig)
                abs_flux_tls[ig] = flux[l * ng + ig] * norm_factor;

            DepleteNode(ws_tls, l, abs_flux_tls.data(), ng, dt, xe_transient);
        }
    }
}

void XSSet::DecayIsotopeDensityFlat(std::vector<double>& iden_flat,
                                    int nxyz, double cooling_days, int substeps) const {
    using namespace Isotope;
    const size_t niso = Isotope::niso;
    if (cooling_days <= 0.0 || nxyz <= 0 || niso == 0 || depDecay.size() == 0)
        return;
    if (iden_flat.size() < static_cast<size_t>(nxyz) * niso)
        throw std::runtime_error("XSSet: restart isotope density size is smaller than nxyz*niso.");

    const int    ndecay = std::max(1, substeps);
    const double dt     = cooling_days * 86400.0 / static_cast<double>(ndecay);

#pragma omp parallel if (nxyz > OMP_THRESHOLD)
    {
        static thread_local DepletionWorkspace ws_tls;
        ws_tls.ensure(niso);

#pragma omp for schedule(dynamic, 8)
        for (int l = 0; l < nxyz; ++l) {
            const auto off = static_cast<size_t>(l) * niso;
            for (size_t iso = 0; iso < niso; ++iso)
                ws_tls.iden[iso] = iden_flat[off + iso];

            for (int isub = 0; isub < ndecay; ++isub)
                milk::Solver<double>::solveBatemanCRAM(depDecay, ws_tls.iden, dt, ws_tls.iden,
                                                       ws_tls.cram, CRAM_ORDER, iI135);

            for (size_t iso = iI135; iso < niso; ++iso)
                iden_flat[off + iso] = ws_tls.iden[iso];
        }
    }
}

// Predictor / Corrector

void XSSet::PredictorStep(double dt, double power, bool xe_transient) {
    const int    nxyz            = _g.nxyz();
    const int    ng              = _g.ng();
    const size_t niso            = Isotope::niso;
    const size_t scalar_size     = static_cast<size_t>(ng) * static_cast<size_t>(nxyz);
    const size_t isotope_size    = niso * static_cast<size_t>(nxyz);
    const size_t mic_scalar_size = niso * scalar_size;

    // Predictor stage overview:
    // 1. Snapshot BOS material, microscopic XS, macro fission XS, and flux.
    // 2. Deplete once with BOS reaction rates to obtain provisional EOS compositions.
    // 3. Temporarily advance burnup so the EOS neutronics solve sees a consistent predicted state.
    auto xs       = ScalarXS(_xs);
    auto xs_bos   = ScalarXS(_xs_bos);
    auto micx     = ScalarXS(_micx);
    auto micx_bos = ScalarXS(_micx_bos);
    for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
        CopyDoubles(scalar_size, xs[xt]->data(), xs_bos[xt]->data());
        CopyDoubles(mic_scalar_size, micx[xt]->data(), micx_bos[xt]->data());
    }

    CopyDoubles(isotope_size, _iden.data(), _iden_bos.data());
    std::copy(_burn.begin(), _burn.begin() + nxyz, _burn_bos.begin());
    std::copy(_history_ctyp.begin(), _history_ctyp.begin() + nxyz, _history_ctyp_bos.begin());
    std::copy(_rodded_fluence.begin(), _rodded_fluence.begin() + nxyz, _rodded_fluence_bos.begin());
    _fine_rod_fluence_bos = _fine_rod_fluence;
    CopyDoubles(scalar_size, _g.Phif(), _flux_bos.data());

    // Apply the beginning-of-step operator A_1 to obtain the predictor inventory N^P.
    Deplete(dt, power, xe_transient);
    DepleteRodMaterials(dt, power, false);

    // Temporarily advance burnup so the transport solve between predictor and corrector uses the
    // same end-of-step reference point as the predicted densities.
    UpdateBurnup(dt, power);
}

void XSSet::CorrectorStep(double dt, double power, bool xe_transient) {
    using namespace Isotope;

    // Predictor-corrector flavour.  Default: RASBERY's midpoint scheme -- build one
    // transition matrix from the BOS/EOS *rate* average and do a single CRAM solve from
    // the BOS inventory.  RASBERY_PC_MODE=decart selects DeCART2D's Eq. (6.20) instead:
    // the corrector solves with EOS rates only, and the final inventory is the average of
    // the predictor (BOS-rate) and corrector (EOS-rate) *densities*.
    static const bool pcDensityAverage = []() {
        const char* m = std::getenv("RASBERY_PC_MODE");
        return m != nullptr && std::string(m) == "decart";
    }();

    const int           nxyz           = _g.nxyz();
    const int           ng             = _g.ng();
    const double        bos_norm       = NormFactor(power, _xs_bos, _flux_bos.data());
    const double        eos_norm       = NormFactor(power);
    const double* const eos_flux       = _g.Phif();
    const size_t        condensed_size = niso * N_XS_SCALAR;
    const size_t        nxyz_size      = static_cast<size_t>(nxyz);

#pragma omp parallel if (nxyz > OMP_THRESHOLD)
    {
        static thread_local DepletionWorkspace  ws_tls;
        static thread_local std::vector<double> corrected_flux_tls;

        ws_tls.ensure(Isotope::niso);
        if (corrected_flux_tls.size() != static_cast<size_t>(ng))
            corrected_flux_tls.resize(ng);

        const double* bos_ptrs[N_XS_SCALAR] = {
            _micx_bos.xstf.data(), _micx_bos.xsdf.data(), _micx_bos.xsaf.data(), _micx_bos.xsff.data(),
            _micx_bos.xsnf.data(), _micx_bos.xskf.data(), _micx_bos.xssf.data(), _micx_bos.xsrf.data(),
            _micx_bos.fyld.data(), _micx_bos.xs2n.data(), _micx_bos.xs3n.data()};
        const double* eos_ptrs[N_XS_SCALAR] = {
            _micx.xstf.data(), _micx.xsdf.data(), _micx.xsaf.data(), _micx.xsff.data(),
            _micx.xsnf.data(), _micx.xskf.data(), _micx.xssf.data(), _micx.xsrf.data(),
            _micx.fyld.data(), _micx.xs2n.data(), _micx.xs3n.data()};

#pragma omp for schedule(dynamic, 8)
        for (int l = 0; l < nxyz; ++l) {
            double burn = 0.0;
            for (int ig = 0; ig < ng; ++ig)
                corrected_flux_tls[ig] = pcDensityAverage
                    ? eos_flux[l * ng + ig] * eos_norm
                    : 0.5 * (_flux_bos[l * ng + ig] * bos_norm + eos_flux[l * ng + ig] * eos_norm);

            for (int ig = 0; ig < ng; ++ig) {
                const double sigma_corrected = pcDensityAverage
                    ? _xs.xskf[ig * nxyz + l]
                    : 0.5 * (_xs_bos.xskf[ig * nxyz + l] + _xs.xskf[ig * nxyz + l]);
                burn += sigma_corrected * corrected_flux_tls[ig] * _g.vol(l) * dt;
            }

            if (ws_tls.condensed.size() < condensed_size)
                ws_tls.condensed.resize(condensed_size, 0.0);

            double raw_sumflux = 0.0;
            for (int ig = 0; ig < ng; ++ig)
                raw_sumflux += corrected_flux_tls[ig];
            const double invflux           = (raw_sumflux > 0.0) ? 1.0 / raw_sumflux : 0.0;
            const double corrected_sumflux = FluxScale(corrected_flux_tls.data(), ng);

            const size_t node = static_cast<size_t>(l);
            for (size_t iso = 0; iso < niso; ++iso) {
                double* dst = ws_tls.condensed.data() + iso * N_XS_SCALAR;
                for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
                    double sum = 0.0;
                    for (int ig = 0; ig < ng; ++ig) {
                        const size_t off = (iso * static_cast<size_t>(ng) + static_cast<size_t>(ig)) *
                                               nxyz_size +
                                           node;
                        const double sigma_corrected = pcDensityAverage
                            ? eos_ptrs[xt][off]
                            : 0.5 * (bos_ptrs[xt][off] + eos_ptrs[xt][off]);
                        sum += sigma_corrected * corrected_flux_tls[ig];
                    }
                    dst[xt] = sum * invflux;
                }
            }

            for (size_t i = 0; i < niso; ++i)
                ws_tls.iden[i] = _iden_bos[i * nxyz_size + node];

            BuildTransitionMatrix(ws_tls.condensed, corrected_sumflux, ws_tls.matrix);
            milk::Solver<double>::solveBatemanCRAM(ws_tls.matrix, ws_tls.iden, dt, ws_tls.iden,
                                                   ws_tls.cram, CRAM_ORDER, iI135);

            if (!xe_transient)
                ApplyXeEquilibrium(ws_tls.iden, ws_tls.condensed, corrected_sumflux);

            // ws_tls.iden now holds the corrector inventory N^C.  In DeCART mode _iden still
            // carries the predictor inventory N^P (PredictorStep wrote it, the transport solve
            // did not touch it), so average the two per Eq. (6.20); otherwise take N^C directly.
            for (size_t i = iI135; i < niso; ++i) {
                const double n_corr = ws_tls.iden[i];
                _iden[i * nxyz_size + node] =
                    pcDensityAverage ? 0.5 * (_iden[i * nxyz_size + node] + n_corr) : n_corr;
            }

            _burn[l]           = _burn_bos[l];
            _history_ctyp[l]   = _history_ctyp_bos[l];
            _rodded_fluence[l] = _rodded_fluence_bos[l];
            if (burn >= 1.0e-10) {
                const double dfac               = 8.64e7 * (_g.vol(l) / _lib_model_volu[_comp[l]]) * _lib_model_hmas[_comp[l]];
                const double burn_key_increment = burn / dfac * 1000.0;
                _burn[l] += static_cast<int>(burn_key_increment + 0.5);
                const bool   currentRod  = UsesRodXS(l);
                const double rodFraction = currentRod ? std::clamp(_g.rod_fraction(l), 0.0, 1.0) : 0.0;
                if (currentRod)
                    _history_ctyp[l] = _ctyp[l];
                if (currentRod && rodFraction > 0.0 && raw_sumflux > 0.0 && std::isfinite(raw_sumflux))
                    _rodded_fluence[l] += raw_sumflux * dt * rodFraction;
            }
        }
    }

    // Rebuild XS on the corrected EOS composition and burnup.
    _fine_rod_fluence = _fine_rod_fluence_bos;
    DepleteRodMaterials(dt, power, true);
    PrecomputeBranchCoefficients();
    UpdateFlatXS();
}

// Rod insertion

void XSSet::RebuildFineRodOccupancy() {
    const int nxy = _g.nxy();
    const int nz  = _g.nz();
    const int kbc = _g.kbc();
    const int kec = _g.kec();
    const int div = std::max(1, _axial_rod_division);

    const size_t expected_size =
        static_cast<size_t>(nxy) * static_cast<size_t>(nz) * static_cast<size_t>(div);
    if (_fine_rod_type.size() != expected_size) {
        _fine_rod_type.assign(expected_size, 0);
        _fine_rod_frac.assign(expected_size, 0.0);
        _fine_rod_fluence.assign(expected_size, 0.0);
        _fine_rod_fluence_bos.assign(expected_size, 0.0);
    }
    if (_fine_rod_type.empty()) return;
    if (_fine_rod_frac.size() != _fine_rod_type.size())
        _fine_rod_frac.assign(_fine_rod_type.size(), 0.0);

    std::fill(_fine_rod_type.begin(), _fine_rod_type.end(), 0);
    std::fill(_fine_rod_frac.begin(), _fine_rod_frac.end(), 0.0);

    auto& cum_bot = _cum_bot_scratch;
    if (cum_bot.size() != static_cast<size_t>(nz + 1)) cum_bot.resize(nz + 1);
    std::fill(cum_bot.begin(), cum_bot.end(), 0.0);
    for (int k = 0; k < nz; ++k)
        cum_bot[k + 1] = cum_bot[k] + _g.hz(k);

    const double fuel_top    = cum_bot[kec];
    const double fuel_bot    = cum_bot[kbc];
    const double fuel_height = fuel_top - fuel_bot;

    for (const auto& rod_entry : _rod_groups) {
        const auto& group = rod_entry.second;
        if (group.insertion < EPS) continue;

        const double rod_length = RodTotalLength(group);
        const double depth      = std::min(std::min(std::max(group.insertion, 0.0), fuel_height), rod_length);
        const double rod_tip    = fuel_top - depth;

        for (int l2d : group.xy_nodes) {
            for (int k = kbc; k < kec; ++k) {
                const double node_bot = cum_bot[k];
                const double fine_h   = _g.hz(k) / static_cast<double>(div);
                for (int m = 0; m < div; ++m) {
                    const double cell_bot = node_bot + static_cast<double>(m) * fine_h;
                    const double cell_top = cell_bot + fine_h;
                    // Rodded region of this node is [rod_tip, node_top] (rod inserts top-down).
                    // Volume-weight the one boundary cell the tip sits inside so the occupancy
                    // is continuous in tip position (no fixed-grid staircase).
                    const double rodded_len = std::max(0.0, cell_top - std::max(cell_bot, rod_tip));
                    if (rodded_len <= 0.0 || fine_h <= 0.0)
                        continue;
                    const double frac = std::min(1.0, rodded_len / fine_h);
                    // Use the midpoint of the rodded sub-span to pick the rod ctype/segment.
                    const double rodded_mid                  = std::max(cell_bot, rod_tip) + 0.5 * rodded_len;
                    const int    idx                         = (k * div + m) * nxy + l2d;
                    _fine_rod_type[static_cast<size_t>(idx)] = RodCTypeAtDistance(group, rodded_mid - rod_tip);
                    _fine_rod_frac[static_cast<size_t>(idx)] = frac;
                }
            }
        }
    }
}

void XSSet::DepleteRodMaterials(double dt, double power, bool corrected_flux) {
    if (dt <= 0.0 || power <= 0.0 || _fine_rod_type.empty())
        return;

    const int    nxy = _g.nxy();
    const int    nz  = _g.nz();
    const int    ng  = _g.ng();
    const int    div = std::max(1, _axial_rod_division);
    const size_t expected_size =
        static_cast<size_t>(nxy) * static_cast<size_t>(nz) * static_cast<size_t>(div);
    if (_fine_rod_fluence.size() != expected_size)
        _fine_rod_fluence.assign(expected_size, 0.0);
    if (_fine_rod_type.size() != expected_size)
        return;

    const double eos_norm = NormFactor(power);
    double       bos_norm = 0.0;
    if (corrected_flux)
        bos_norm = NormFactor(power, _xs_bos, _flux_bos.data());

#pragma omp parallel for schedule(static) if (static_cast<int>(expected_size) > OMP_THRESHOLD)
    for (int idx = 0; idx < static_cast<int>(expected_size); ++idx) {
        // Accumulate fluence only for cells at least half rodded (matches the historical
        // center-in-rod criterion); the cusping fractional weighting does not affect depletion.
        if (_fine_rod_type[static_cast<size_t>(idx)] == 0 ||
            _fine_rod_frac[static_cast<size_t>(idx)] < 0.5)
            continue;

        const int fine_axial = idx / nxy;
        const int l2d        = idx - fine_axial * nxy;
        const int k          = fine_axial / div;
        const int l          = k * nxy + l2d;
        if (l < 0 || l >= _g.nxyz())
            continue;

        double scalar_flux = 0.0;
        for (int ig = 0; ig < ng; ++ig) {
            const size_t flux_idx = static_cast<size_t>(l) * static_cast<size_t>(ng) +
                                    static_cast<size_t>(ig);
            if (corrected_flux) {
                scalar_flux += 0.5 * (_flux_bos[flux_idx] * bos_norm +
                                      _g.Phif()[flux_idx] * eos_norm);
            } else {
                scalar_flux += _g.Phif()[flux_idx] * eos_norm;
            }
        }
        if (scalar_flux > 0.0 && std::isfinite(scalar_flux))
            _fine_rod_fluence[static_cast<size_t>(idx)] += scalar_flux * dt;
    }
}

void XSSet::SetRod(const std::map<std::string, double>& insertions) {
    const int nxy  = _g.nxy();
    const int nz   = _g.nz();
    const int nxyz = _g.nxyz();
    const int kbc  = _g.kbc();
    const int kec  = _g.kec();

    const std::vector<int>    old_segment_offset   = _rod_node_segment_offset;
    const std::vector<int>    old_segment_ctype    = _rod_node_segment_ctype;
    const std::vector<double> old_segment_fraction = _rod_node_segment_fraction;

    auto& old_fraction = _old_rod_fraction_scratch;
    auto& old_ctyp     = _old_rod_ctyp_scratch;
    old_fraction.resize(nxyz);
    old_ctyp.resize(nxyz);
    for (int lk = 0; lk < nxyz; ++lk) {
        old_fraction[lk] = _g.rod_fraction(lk);
        old_ctyp[lk]     = _ctyp[lk];
    }

    for (const auto& [name, depth_cm] : insertions) {
        auto it = _rod_groups.find(name);
        if (it != _rod_groups.end())
            it->second.insertion = depth_cm;
    }

    for (int lk = 0; lk < nxyz; ++lk) {
        _g.rod_fraction(lk) = 0.0;
        _ctyp[lk]           = 0;
    }

    auto& cum_bot = _cum_bot_scratch;
    if (cum_bot.size() != static_cast<size_t>(nz + 1)) cum_bot.resize(nz + 1);
    std::fill(cum_bot.begin(), cum_bot.end(), 0.0);
    for (int k = 0; k < nz; ++k)
        cum_bot[k + 1] = cum_bot[k] + _g.hz(k);

    // Insertion depth is measured from the top of the fuel region downward;
    // top and bottom axial reflectors are excluded so that depth=0 places the rod
    // tip at the top of fuel and depth=fuel_height places it at the bottom.
    const double fuel_top    = cum_bot[kec];
    const double fuel_bot    = cum_bot[kbc];
    const double fuel_height = fuel_top - fuel_bot;

    std::vector<int>    segment_count(static_cast<size_t>(nxyz), 0);
    std::vector<double> dominant_fraction(static_cast<size_t>(nxyz), 0.0);

    for (const auto& [name, group] : _rod_groups) {
        if (group.insertion < EPS) continue;

        const double rod_length = RodTotalLength(group);
        const double depth      = std::min(std::min(std::max(group.insertion, 0.0), fuel_height), rod_length);
        const double rod_tip    = fuel_top - depth;

        for (int l : group.xy_nodes) {
            for (int k = kbc; k < kec; ++k) {
                const int    lk       = k * nxy + l;
                const double node_bot = cum_bot[k];
                const double node_top = cum_bot[k + 1];
                const double node_h   = node_top - node_bot;

                const double rodded_bot = std::max(rod_tip, node_bot);
                const double rodded_top = node_top;
                if (rodded_top <= rodded_bot || node_h <= 0.0)
                    continue;

                double frac         = (rodded_top - rodded_bot) / node_h;
                _g.rod_fraction(lk) = std::min(1.0, frac);

                if (group.ctype_segments.empty()) {
                    segment_count[static_cast<size_t>(lk)] += 1;
                    if (_g.rod_fraction(lk) > dominant_fraction[static_cast<size_t>(lk)]) {
                        dominant_fraction[static_cast<size_t>(lk)] = _g.rod_fraction(lk);
                        _ctyp[lk]                                  = group.ctype;
                    }
                    continue;
                }

                double segment_bot_distance = 0.0;
                for (int si = static_cast<int>(group.ctype_segments.size()) - 1; si >= 0; --si) {
                    const double segment_top_distance =
                        segment_bot_distance + group.length_segments[static_cast<size_t>(si)];
                    const double segment_bot = rod_tip + segment_bot_distance;
                    const double segment_top = rod_tip + segment_top_distance;
                    const double overlap_bot = std::max(rodded_bot, segment_bot);
                    const double overlap_top = std::min(rodded_top, segment_top);
                    if (overlap_top > overlap_bot) {
                        const double segment_frac = (overlap_top - overlap_bot) / node_h;
                        segment_count[static_cast<size_t>(lk)] += 1;
                        if (segment_frac > dominant_fraction[static_cast<size_t>(lk)]) {
                            dominant_fraction[static_cast<size_t>(lk)] = segment_frac;
                            _ctyp[lk]                                  = group.ctype_segments[static_cast<size_t>(si)];
                        }
                    }
                    segment_bot_distance = segment_top_distance;
                }
            }
        }
    }

    _rod_node_segment_offset.assign(static_cast<size_t>(nxyz + 1), 0);
    for (int lk = 0; lk < nxyz; ++lk)
        _rod_node_segment_offset[static_cast<size_t>(lk + 1)] =
            _rod_node_segment_offset[static_cast<size_t>(lk)] + segment_count[static_cast<size_t>(lk)];
    const int total_segments = _rod_node_segment_offset[static_cast<size_t>(nxyz)];
    _rod_node_segment_ctype.assign(static_cast<size_t>(total_segments), 0);
    _rod_node_segment_fraction.assign(static_cast<size_t>(total_segments), 0.0);

    std::vector<int> write_pos = _rod_node_segment_offset;
    for (const auto& [name, group] : _rod_groups) {
        if (group.insertion < EPS) continue;

        const double rod_length = RodTotalLength(group);
        const double depth      = std::min(std::min(std::max(group.insertion, 0.0), fuel_height), rod_length);
        const double rod_tip    = fuel_top - depth;

        for (int l : group.xy_nodes) {
            for (int k = kbc; k < kec; ++k) {
                const int    lk       = k * nxy + l;
                const double node_bot = cum_bot[k];
                const double node_top = cum_bot[k + 1];
                const double node_h   = node_top - node_bot;

                const double rodded_bot = std::max(rod_tip, node_bot);
                const double rodded_top = node_top;
                if (rodded_top <= rodded_bot || node_h <= 0.0)
                    continue;

                if (group.ctype_segments.empty()) {
                    const int idx                                     = write_pos[static_cast<size_t>(lk)]++;
                    _rod_node_segment_ctype[static_cast<size_t>(idx)] = group.ctype;
                    _rod_node_segment_fraction[static_cast<size_t>(idx)] =
                        std::min(1.0, (rodded_top - rodded_bot) / node_h);
                    continue;
                }

                double segment_bot_distance = 0.0;
                for (int si = static_cast<int>(group.ctype_segments.size()) - 1; si >= 0; --si) {
                    const double segment_top_distance =
                        segment_bot_distance + group.length_segments[static_cast<size_t>(si)];
                    const double segment_bot = rod_tip + segment_bot_distance;
                    const double segment_top = rod_tip + segment_top_distance;
                    const double overlap_bot = std::max(rodded_bot, segment_bot);
                    const double overlap_top = std::min(rodded_top, segment_top);
                    if (overlap_top > overlap_bot) {
                        const int idx = write_pos[static_cast<size_t>(lk)]++;
                        _rod_node_segment_ctype[static_cast<size_t>(idx)] =
                            group.ctype_segments[static_cast<size_t>(si)];
                        _rod_node_segment_fraction[static_cast<size_t>(idx)] =
                            (overlap_top - overlap_bot) / node_h;
                    }
                    segment_bot_distance = segment_top_distance;
                }
            }
        }
    }

    auto& dirty_nodes = _dirty_nodes_scratch;
    dirty_nodes.clear();
    if (dirty_nodes.capacity() < static_cast<size_t>(nxyz / 8 + 1))
        dirty_nodes.reserve(nxyz / 8 + 1);
    for (int lk = 0; lk < nxyz; ++lk) {
        bool segment_changed = old_segment_offset.size() != _rod_node_segment_offset.size();
        if (!segment_changed) {
            const int old_begin = old_segment_offset[static_cast<size_t>(lk)];
            const int old_end   = old_segment_offset[static_cast<size_t>(lk + 1)];
            const int new_begin = _rod_node_segment_offset[static_cast<size_t>(lk)];
            const int new_end   = _rod_node_segment_offset[static_cast<size_t>(lk + 1)];
            segment_changed     = (old_end - old_begin) != (new_end - new_begin);
            for (int i = 0; !segment_changed && i < old_end - old_begin; ++i) {
                const int old_idx = old_begin + i;
                const int new_idx = new_begin + i;
                if (old_segment_ctype[static_cast<size_t>(old_idx)] !=
                        _rod_node_segment_ctype[static_cast<size_t>(new_idx)] ||
                    std::abs(old_segment_fraction[static_cast<size_t>(old_idx)] -
                             _rod_node_segment_fraction[static_cast<size_t>(new_idx)]) > 1.0e-14) {
                    segment_changed = true;
                }
            }
        }
        if (old_ctyp[lk] != _ctyp[lk] ||
            std::abs(old_fraction[lk] - _g.rod_fraction(lk)) > 1.0e-14 ||
            segment_changed)
            dirty_nodes.push_back(lk);
    }

    RebuildFineRodOccupancy();

    if (dirty_nodes.empty()) return;

    XSUpdateOptions options;
    options.nodes.swap(dirty_nodes);
    UpdateFlatXS(options);
    options.nodes.swap(dirty_nodes);
}

void XSSet::SetBoron(double bppm) {
    const int nxyz = _g.nxyz();

    for (int l = 0; l < nxyz; ++l)
        _g.bppm(l) = bppm;

    UpdateFlatXS();
}

void XSSet::SetRod(double step) {
    if (_rod_ncols <= 0) return;

    const double clamped   = std::clamp(step, 0.0, static_cast<double>(_rod_ncols - 1));
    const int    lower_col = static_cast<int>(std::floor(clamped));
    const int    upper_col = std::min(lower_col + 1, _rod_ncols - 1);
    const double frac      = clamped - static_cast<double>(lower_col);

    std::map<std::string, double> depths;
    for (int r = 0; r < static_cast<int>(_rod_group_order.size()); ++r) {
        const double lower_depth    = _rod_profile(r, lower_col);
        const double upper_depth    = _rod_profile(r, upper_col);
        depths[_rod_group_order[r]] = (upper_col == lower_col || frac <= 1.0e-12)
                                          ? lower_depth
                                          : (1.0 - frac) * lower_depth + frac * upper_depth;
    }

    SetRod(depths);
}

// Build rod profile matrix from per-group profiles.
// Pads shorter profiles by repeating their last value. Prepends column 0 (all zeros).
void XSSet::BuildRodProfileMatrix(const std::map<std::string, std::vector<double>>& profiles) {
    if (profiles.empty()) {
        _rod_ncols = 0;
        return;
    }

    // Find max profile length
    int max_len = 0;
    for (const auto& [name, prof] : profiles)
        max_len = std::max(max_len, static_cast<int>(prof.size()));

    _rod_ncols        = max_len + 1; // +1 for column 0 (all zeros)
    const int ngroups = static_cast<int>(profiles.size());

    _rod_group_order.clear();
    _rod_group_order.reserve(ngroups);
    _rod_profile.assign(ngroups, _rod_ncols, 0.0);

    int row = 0;
    for (const auto& [name, prof] : profiles) {
        _rod_group_order.push_back(name);
        // Column 0 is already 0.0
        for (int c = 0; c < static_cast<int>(prof.size()); ++c)
            _rod_profile(row, c + 1) = prof[c];
        // Pad with last value
        const double last = prof.empty() ? 0.0 : prof.back();
        for (int c = static_cast<int>(prof.size()) + 1; c < _rod_ncols; ++c)
            _rod_profile(row, c) = last;
        ++row;
    }
}

// T/H steady-state solve (moved from THSolver)

void XSSet::SolveTH(const double* node_power, const int* burnup, double power_rate) {
    const int nxy  = _g.nxy();
    const int nz   = _g.nz();
    const int nxyz = _g.nxyz();

    const double inlet_temp      = _g.inlet_temp();
    const double outlet_temp     = _g.outlet_temp();
    const double pressure        = _g.pressure();
    const double rated_power     = _g.rated_power();
    const double input_mass_flux = _g.mass_flow_rate();

    double total_raw_power = 0.0;
    for (int lk = 0; lk < nxyz; ++lk)
        total_raw_power += node_power[lk];

    const double actual_power        = 1000.0 * rated_power * power_rate;
    const double inlet_h             = GetHmod(inlet_temp, pressure);
    const double outlet_h            = GetHmod(outlet_temp, pressure);
    const bool   use_input_mass_flux = _g.use_mass_flow_rate() && input_mass_flux > 0.0;
    const double total_flow          = use_input_mass_flux
                                           ? 0.0
                                           : actual_power / (outlet_h - inlet_h);
    const double norm                = (total_raw_power > 0.0) ? actual_power / total_raw_power : 0.0;

    double total_area = 0.0;
    int    k_mid      = (_g.kbc() + _g.kec() - 1) / 2;
    for (int l = 0; l < nxy; ++l) {
        int idx = k_mid * nxy + l;
        if (node_power[idx] > 0.0) total_area += _g.hmesh(XDIR, idx) * _g.hmesh(YDIR, idx);
    }

    const double flow_per_channel = use_input_mass_flux
                                        ? input_mass_flux * 1.0e-4
                                        : total_flow / total_area;
    const int    kbc              = _g.kbc();
    const int    kec              = _g.kec();

    for (int l = 0; l < nxy; ++l) {
        double h_cur = inlet_h;

        // Bottom reflector
        for (int k = 0; k < kbc; ++k) {
            const int lk = l + k * nxy;
            _g.tmod(lk)  = GetTmod(h_cur, pressure);
            _g.dmod(lk)  = GetDmod(h_cur, pressure);
        }

        // Fuel region
        for (int k = kbc; k < kec; ++k) {
            const int    lk     = l + k * nxy;
            const double P_node = node_power[lk] * norm;
            if (P_node <= 0.0) {
                _g.tmod(lk) = GetTmod(h_cur, pressure);
                _g.dmod(lk) = GetDmod(h_cur, pressure);
                continue;
            }
            const double dh    = P_node / (flow_per_channel * _g.hmesh(XDIR, l) * _g.hmesh(YDIR, l));
            const double h_out = h_cur + dh;
            const double h_avg = 0.5 * (h_cur + h_out);
            _g.tmod(lk)        = GetTmod(h_avg, pressure);
            _g.dmod(lk)        = GetDmod(h_avg, pressure);
            const double lpd   = 1000.0 * P_node / (62.0 * _g.hz(k));
            const double bu    = burnup[lk] / 1000.0;
            _g.tful(lk)        = _g.tmod(lk) + GetTfuel(bu, lpd);
            h_cur              = h_out;
        }

        // Top reflector
        for (int k = kec; k < nz; ++k) {
            const int lk = l + k * nxy;
            _g.tmod(lk)  = GetTmod(h_cur, pressure);
            _g.dmod(lk)  = GetDmod(h_cur, pressure);
        }
    }
}

void XSSet::RemoveUpscattering() {
    const int ng   = _g.ng();
    const int nxyz = _g.nxyz();

    for (int igs = 0; igs < ng; ++igs) {
        for (int ige = igs + 1; ige < ng; ++ige) {
#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
            for (int l = 0; l < nxyz; ++l) {
                const size_t upscat_idx = (ige * ng + igs) * nxyz + l;
                double       upscat     = _xs.xssm[upscat_idx];
                if (upscat != 0.0) {
                    const double phi_src = _g.Phif()[l * ng + ige];
                    const double phi_dst = _g.Phif()[l * ng + igs];
                    if (phi_dst > 1.0e-24)
                        _xs.xssm[(igs * ng + ige) * nxyz + l] -= upscat * (phi_src / phi_dst);
                    _xs.xssm[upscat_idx] = 0.0;
                }
            }
        }
    }

    for (int igs = 0; igs < ng; ++igs) {
#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
        for (int l = 0; l < nxyz; ++l) {
            double outgoing = 0.0;
            for (int ige = 0; ige < ng; ++ige)
                outgoing += _xs.xssm[(igs * ng + ige) * nxyz + l];
            _xs.xsrf[igs * nxyz + l] = _xs.xsaf[igs * nxyz + l] + outgoing;
        }
    }
}

void XSSet::InitXS(double bppm, double tful, double tmod, double pres,
                   double dmod, bool overwrite_th) {
    const int    nxyz      = _g.nxyz();
    const double hmod_base = GetHmod(tmod, pres);
    const double dmod_base = GetDmod(hmod_base, pres);

    if (overwrite_th) {
        for (int l = 0; l < nxyz; ++l) {
            _g.bppm(l) = bppm;
            _g.tful(l) = tful;
            _g.tmod(l) = tmod;
            _g.dmod(l) = dmod_base + dmod;
        }
    }

    const size_t niso                    = Chiffon::Isotope::niso;
    const size_t ng                      = static_cast<size_t>(_g.ng());
    bool         preserved_external_iden = false;
    for (int l = 0; l < nxyz; ++l) {
        const auto&          model = _models[_comp[l]];
        milk::Vector<double> lib_iden;
        auto                 xs_obj = model.GetCrossSection(0, _burn[l], _g.bppm(l), _g.tful(l),
                                                            _g.dmod(l), &lib_iden);

        const bool preserve_external_iden =
            l < static_cast<int>(_external_iden.size()) && _external_iden[l];
        if (preserve_external_iden) {
            preserved_external_iden = true;
        } else {
            for (size_t iso = 0; iso < niso; ++iso)
                _iden[iso * nxyz + l] = lib_iden[iso];
        }

        if (_g.dmod(l) > 1.0e-30)
            _node_wvfr[l] = lib_iden[Isotope::iO16] / (_g.dmod(l) * WATER_NUMBER_DENSITY);

        unpackXS(xs_obj, l, ng, nxyz, niso);
    }

    PrecomputeBranchCoefficients();
    UpdateFlatXS();
    if (preserved_external_iden)
        std::fill(_external_iden.begin(), _external_iden.end(), 0);
}

void XSSet::SetPowerRate(double power_rate) {
    _current_power_rate = power_rate;
    UpdateFlatXS();
}

void XSSet::ResetFluxAndCurrents(double flux_value) {
    std::fill_n(_g.Phif(), static_cast<size_t>(_g.ngxyz()), flux_value);

    const size_t surface_size = static_cast<size_t>(LR) * static_cast<size_t>(_g.ng()) *
                                static_cast<size_t>(NDIRMAX) * static_cast<size_t>(_g.nxyz());
    std::fill_n(_g.Jnet(), surface_size, 0.0);
    std::fill_n(_g.Phis(), surface_size, 0.0);
}

void XSSet::NormalizeFluxSign() {
    double    fission_source = 0.0;
    const int ng             = _g.ng();
    const int nxyz           = _g.nxyz();
    for (int lk = 0; lk < nxyz; ++lk) {
        if (!_g.IsFuel(lk)) continue;
        for (int ig = 0; ig < ng; ++ig)
            fission_source += _xs.xskf[ig * nxyz + lk] *
                              _g.Phif()[lk * ng + ig] *
                              _g.vol(lk);
    }
    if (fission_source >= 0.0)
        return;

    const size_t flux_size = static_cast<size_t>(_g.ngxyz());
    for (size_t i = 0; i < flux_size; ++i)
        _g.Phif()[i] = -_g.Phif()[i];

    const size_t surface_size = static_cast<size_t>(LR) * static_cast<size_t>(ng) *
                                static_cast<size_t>(NDIRMAX) * static_cast<size_t>(nxyz);
    for (size_t i = 0; i < surface_size; ++i) {
        _g.Jnet()[i] = -_g.Jnet()[i];
        _g.Phis()[i] = -_g.Phis()[i];
    }

    for (int lk = 0; lk < nxyz; ++lk)
        _g.Psi()[lk] = -_g.Psi()[lk];
}

double XSSet::UpdateTH(double power_rate) {
    const int ng         = _g.ng();
    const int nxyz       = _g.nxyz();
    auto&     node_power = _node_power_scratch;
    _current_power_rate  = power_rate;

    if (node_power.size() != static_cast<size_t>(nxyz))
        node_power.assign(nxyz, 0.0);
    else
        std::fill(node_power.begin(), node_power.end(), 0.0);

#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
    for (int lk = 0; lk < nxyz; ++lk) {
        double power_density = 0.0;
        for (int ig = 0; ig < ng; ++ig)
            power_density += _xs.xskf[ig * nxyz + lk] * _g.Phif()[lk * ng + ig];
        node_power[lk] = power_density * _g.vol(lk);
    }

    double total_power = 0.0;
    for (int lk = 0; lk < nxyz; ++lk)
        total_power += node_power[lk];
    if (total_power < 0.0) {
        for (int lk = 0; lk < nxyz; ++lk)
            node_power[lk] = -node_power[lk];
    }

    // Snapshot fuel temperature to report the Doppler-temperature change (delta_Dop) used as the
    // T/H feedback convergence metric (PARCS Eq. 10.8) -- looser and more physical than k_eff.
    auto& tful_old = _th_tful_old_scratch;
    if (tful_old.size() != static_cast<size_t>(nxyz))
        tful_old.assign(nxyz, 0.0);
    std::vector<double> tmod_old(static_cast<size_t>(nxyz)), dmod_old(static_cast<size_t>(nxyz));
    for (int lk = 0; lk < nxyz; ++lk) {
        tful_old[lk]                        = _g.tful(lk);
        tmod_old[static_cast<size_t>(lk)]   = _g.tmod(lk);
        dmod_old[static_cast<size_t>(lk)]   = _g.dmod(lk);
    }

    SolveTH(node_power.data(), _burn.data(), power_rate);

    // Under-relax the temperature feedback: SolveTH overwrites the temperatures directly, so the
    // undamped temperature<->flux Picard loop can overshoot and oscillate (delta_Dop grows on the
    // 2nd iteration), needing ~18 iterations to settle. Damping toward the previous temperatures
    // converges in a few iterations to the SAME fixed point (relaxation cancels at convergence).
    if (_th_relaxation < 1.0) {
        const double w = _th_relaxation;
        for (int lk = 0; lk < nxyz; ++lk) {
            _g.tful(lk) = (1.0 - w) * tful_old[lk] + w * _g.tful(lk);
            _g.tmod(lk) = (1.0 - w) * tmod_old[static_cast<size_t>(lk)] + w * _g.tmod(lk);
            _g.dmod(lk) = (1.0 - w) * dmod_old[static_cast<size_t>(lk)] + w * _g.dmod(lk);
        }
    }

    double delta_dop = 0.0;
    for (int lk = 0; lk < nxyz; ++lk) {
        const double t = _g.tful(lk);
        if (t > 1.0e-30)
            delta_dop = std::max(delta_dop, std::abs(t - tful_old[lk]) / t);
    }

    UpdateFlatXS();
    return delta_dop;
}

void XSSet::UpdateDerivative(double delta_bppm, double delta_tful, double delta_tmod, double delta_dmod) {
    const int nxyz = _g.nxyz();

    for (int l = 0; l < nxyz; ++l) {
        _g.bppm(l) += delta_bppm;
        _g.tful(l) += delta_tful;
        if (delta_tmod != 0.0) {
            _g.dmod(l) = GetDmod(GetHmod(_g.tmod(l) + delta_tmod, _g.pressure()), _g.pressure());
            _g.tmod(l) += delta_tmod;
        }
        _g.dmod(l) += delta_dmod;
    }

    UpdateFlatXS();
}

double& XSSet::fmap(const int& ig, const int& l, const int& pinx, const int& piny) {
    const int npins = _g.npins();
    const int npina = npins * npins;
    auto&     dpt   = _models[_comp[l]].GetDepletionPoint(0, 0);
    return dpt._fmap[l * _g.ng() * npina + ig * npina + piny * npins + pinx];
}

double& XSSet::gmap(const int& l, const int& pinx, const int& piny) {
    const int npins = _g.npins();
    auto&     dpt   = _models[_comp[l]].GetDepletionPoint(0, 0);
    return dpt._gmap[piny * npins + pinx];
}

std::vector<double> XSSet::getNodeIden(int l) const {
    const size_t        niso = Chiffon::Isotope::niso;
    const int           nxyz = _g.nxyz();
    std::vector<double> out(niso);

    for (size_t iso = 0; iso < niso; ++iso)
        out[iso] = _iden[iso * nxyz + l];

    return out;
}

void XSSet::setNodeIden(int l, const std::vector<double>& values) {
    const size_t niso = Chiffon::Isotope::niso;
    const int    nxyz = _g.nxyz();

    for (size_t iso = 0; iso < niso && iso < values.size(); ++iso)
        _iden[iso * nxyz + l] = values[iso];
    if (l >= 0 && l < static_cast<int>(_external_iden.size()))
        _external_iden[l] = 1;
}
