#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "milk.h"

#include "highfive/highfive.hpp"
#include "nlohmann/json.hpp"

namespace Chiffon {

/// @brief Global isotope registry, chain definitions, and depletion matrices.
namespace Isotope {
/// @brief Map isotope ID strings to the global isotope order.
inline std::map<std::string, size_t> iidx;

/// @brief Number of isotopes in the global depletion basis.
inline size_t niso = 0;

/// @brief Isotope IDs in the fixed depletion order.
///
/// Order: H-1, natural boron, O-16, iodine/xenon, Nd/Pm/Sm,
/// lumped Gd, and actinides.
inline constexpr std::array<const char*, 39> isotopeIds = {
    "10010", "50002", "80160",
    "531350", "541350", "541351",
    "601470", "601480", "601490",
    "611470", "611480", "611481", "611490",
    "621470", "621480", "621490",
    "640000",
    "922340", "922350", "922360", "922380",
    "932370", "932380", "932390",
    "942380", "942390", "942400", "942410", "942420", "942430",
    "952410", "952420", "952421", "952430", "952440",
    "962420", "962430", "962440", "962450"};

/// @name Common isotope indices
/// @{
inline constexpr size_t iH1     = 0;  // H-1
inline constexpr size_t iB10    = 1;  // B-10
inline constexpr size_t iO16    = 2;  // O-16
inline constexpr size_t iI135   = 3;  // I-135
inline constexpr size_t iXe135  = 4;  // Xe-135
inline constexpr size_t iXe135m = 5;  // Xe-135m
inline constexpr size_t iNd148  = 7;  // Nd-148
inline constexpr size_t iPm148g = 10; // Pm-148
inline constexpr size_t iPm148m = 11; // Pm-148m
inline constexpr size_t iPm149  = 12; // Pm-149
inline constexpr size_t iSm149  = 15; // Sm-149
inline constexpr size_t iGd     = 16; // lumped Gd
/// @}

/// @brief First actinide index used by depletion matrix assembly.
inline constexpr size_t iAcFirst = 17; // first actinide in isotopeIds
/// @brief Last actinide index used by depletion matrix assembly.
inline constexpr size_t iAcLast = 38; // last actinide in isotopeIds

/// @name Isotopes with explicit (n,2n) handling
/// @{
inline constexpr size_t iU234  = 17;
inline constexpr size_t iU235  = 18;
inline constexpr size_t iU236  = 19;
inline constexpr size_t iU238  = 20;
inline constexpr size_t iNp237 = 21;
inline constexpr size_t iNp239 = 23;
inline constexpr size_t iPu238 = 24;
inline constexpr size_t iPu239 = 25;
inline constexpr size_t iPu240 = 26;
inline constexpr size_t iPu241 = 27;
/// @}

/// Decay matrix (niso × niso). Non-depleting isotopes have zero rows/columns.
inline milk::Matrix<double> depDecay;

/// Transmutation topology (niso × niso).
inline milk::Matrix<double> depTrans;

/// @brief Initialize isotope indices and load depletion matrices.
/// @param isotopeDataDir Directory containing dep_decay.csv and dep_trans.csv.
inline void Initialize(const std::filesystem::path& isotopeDataDir) {
    iidx.clear();
    niso = isotopeIds.size();
    for (size_t i = 0; i < isotopeIds.size(); ++i)
        iidx.emplace(isotopeIds[i], i);

    const std::vector<std::string> labels(isotopeIds.begin(), isotopeIds.end());
    depDecay = milk::LabeledMatrixFromCSV(isotopeDataDir / "dep_decay.csv", labels, labels);
    depTrans = milk::LabeledMatrixFromCSV(isotopeDataDir / "dep_trans.csv", labels, labels);
}
} // namespace Isotope

/// @brief Sentinel "isotope" ids marking special history-vector coordinates (near SIZE_MAX).
/// @note Only ROD_FLU is evaluated by the RASBERY runtime (XSSet::ApplyHistoryDeltasToNode);
/// the remaining synthetic coordinates are honored only during CHIFFON fitting
/// (Interpolator::PointDensity) and evaluate to 0 at query time.
namespace Hv {
enum : size_t {
    ROD_DEPL      = std::numeric_limits<size_t>::max() - 8,  ///< explicit rod-depletion factor
    LOG_XE        = std::numeric_limits<size_t>::max() - 10, ///< log Xe-135 density
    LOG_PU        = std::numeric_limits<size_t>::max() - 7,  ///< log Pu-239 density
    LOG_U_PU      = std::numeric_limits<size_t>::max() - 9,  ///< log U-238/Pu-239 ratio
    TOT_BURN      = std::numeric_limits<size_t>::max() - 11, ///< total fuel burnup
    CUR_ROD_FRAC  = std::numeric_limits<size_t>::max() - 12, ///< current local rod insertion fraction
    FUEL_ROD_FLU  = std::numeric_limits<size_t>::max() - 6,  ///< fuel exposure under rodded spectrum
    ROD_BURN_FRAC = std::numeric_limits<size_t>::max() - 5,  ///< rodded fraction of burnup
    FUEL_ROD_BURN = std::numeric_limits<size_t>::max() - 4,  ///< fuel burnup under rodded spectrum
    ROD_BURN      = std::numeric_limits<size_t>::max() - 3,  ///< rod-material burnup
    ROD_FLU       = std::numeric_limits<size_t>::max() - 2,  ///< rod-material fluence
    FAST_THERM    = std::numeric_limits<size_t>::max() - 1,  ///< fast-to-thermal flux ratio
    THERM_FRAC    = std::numeric_limits<size_t>::max()       ///< thermal flux fraction
};
/// @brief Scale rod fluence [n/cm2] to an O(1) coordinate for vector fitting.
inline constexpr double ROD_FLU_SCALE = 1.0e-22;
} // namespace Hv

inline double hvRodFluCoord(double fluence) {
    return std::isfinite(fluence) ? fluence * Hv::ROD_FLU_SCALE : 0.0;
}

inline double hvLogIso(double density) {
    return std::log(std::max(density, 1.0e-30));
}

inline double hvLogRatio(double numerator, double denominator) {
    return hvLogIso(numerator) -
           hvLogIso(denominator);
}

inline double hvBurnCoord(double burnupKey) {
    return std::isfinite(burnupKey) ? burnupKey * 1.0e-3 : 0.0;
}

inline double hvRoddedBurnFrac(double roddedBurnupKey, double totalBurnupKey) {
    if (!std::isfinite(roddedBurnupKey) || !std::isfinite(totalBurnupKey) || totalBurnupKey <= 0.0)
        return 0.0;
    return std::clamp(roddedBurnupKey / totalBurnupKey, 0.0, 1.0);
}

/// @brief Correction-kind tags stored in HistoryDeltaCorrection::kind.
/// Values are fixed for HDF backward-read compatibility. CTYPE_INDEP_VEC (ctype-independent
/// IISC vector) and RHST_UNIT (rod-history residual vector) are the only kinds the RASBERY
/// runtime applies (see XSSet::ApplyHistoryDeltasToNode). VEC is the legacy ctype-keyed
/// spectral vector: it is still fit, classified as IISC, and serialized, but the current
/// runtime apply path skips it, so an applied IISC block must set ctype_independent.
namespace Hk {
enum : int {
    VEC             = 4,
    RHST_UNIT       = 7,
    CTYPE_INDEP_VEC = 26
};
} // namespace Hk

inline double hvFastThermRatio(double fastFlux, double thermalFlux) {
    if (!std::isfinite(fastFlux) || !std::isfinite(thermalFlux) || thermalFlux <= 1.0e-30)
        return 0.0;
    return std::max(0.0, fastFlux) / std::max(thermalFlux, 1.0e-30);
}

inline double hvThermFluxFrac(double fastFlux, double thermalFlux) {
    if (!std::isfinite(fastFlux) || !std::isfinite(thermalFlux))
        return 0.0;
    const double fast    = std::max(0.0, fastFlux);
    const double thermal = std::max(0.0, thermalFlux);
    const double total   = fast + thermal;
    return total > 1.0e-30 ? thermal / total : 0.0;
}

inline double hvFastThermRatio(const std::vector<double>& flux) {
    return flux.size() >= 2 ? hvFastThermRatio(flux[0], flux[1]) : 0.0;
}

inline double hvThermFluxFrac(const std::vector<double>& flux) {
    return flux.size() >= 2 ? hvThermFluxFrac(flux[0], flux[1]) : 0.0;
}

inline bool IsHistoryVectorSpecial(size_t iso) {
    return iso == Hv::ROD_DEPL ||
           iso == Hv::LOG_XE ||
           iso == Hv::LOG_PU ||
           iso == Hv::LOG_U_PU ||
           iso == Hv::TOT_BURN ||
           iso == Hv::CUR_ROD_FRAC ||
           iso == Hv::FUEL_ROD_FLU ||
           iso == Hv::ROD_BURN_FRAC ||
           iso == Hv::FUEL_ROD_BURN ||
           iso == Hv::ROD_BURN ||
           iso == Hv::ROD_FLU ||
           iso == Hv::FAST_THERM ||
           iso == Hv::THERM_FRAC;
}

/// @brief Surface and corner direction indices for discontinuity factors.
namespace Direction {
constexpr int SOUTH = 0;
constexpr int EAST  = 1;
constexpr int NORTH = 2;
constexpr int WEST  = 3;

constexpr int SW = 1;
constexpr int SE = 2;
constexpr int NE = 3;
constexpr int NW = 4;
} // namespace Direction

using namespace Isotope;
using namespace Direction;

/// @brief Return the flat pin index for an (i, j) pin position.
/// @param i Pin x-index.
/// @param j Pin y-index.
/// @param npin Number of pins per assembly side.
inline size_t pidx(size_t i, size_t j, size_t npin) { return j * npin + i; }

/// @brief Return the flat pin/group index for an (i, j) pin and energy group.
/// @param g Energy group index.
/// @param i Pin x-index.
/// @param j Pin y-index.
/// @param npin Number of pins per assembly side.
inline size_t pidg(size_t g, size_t i, size_t j, size_t npin) { return g * npin * npin + j * npin + i; }

/// @brief Cross-section reaction type indices within each energy group.
enum XSTYPE { XSTF, // Transport
              XSDF, // Diffusion coefficient
              XSAF, // Absorption
              XSFF, // Fission
              XSNF, // Nu-fission
              XSKF, // Kappa-fission (energy release)
              XSSF, // Total scattering
              XSRF, // Removal (absorption + out-scatter)
              FYLD, // Fission yield
              XS2N, // (n,2n) reaction
              XS3N, // (n,3n) reaction
              XSSM  // Start of scattering matrix (group-to-group)
};

/// @brief Branch perturbation types in the parameterized cross-section library.
enum BRANCHTYPE { REFR, // Reference (base depletion)
                  BPPM, // Boron concentration
                  TFUL, // Fuel temperature
                  DMOD, // Moderator density
                  TMOD, // Moderator temperature
                  SPCT, // Spectral history correction
                  RHST, // Rod-history correction
                  RDEP  // Rod-material depletion correction
};

/// @brief Human-readable correction components used by the current delta-sum path.
enum class CorrectionComponent {
    REFERENCE,
    BPPM,
    TFUEL,
    DMOD,
    TMOD,
    ROD_DEPLETION,
    IISC,
    IISC_RHST,
    UNKNOWN
};

inline CorrectionComponent CorrectionComponentFromBranch(BRANCHTYPE branchType) {
    if (branchType == BRANCHTYPE::BPPM) return CorrectionComponent::BPPM;
    if (branchType == BRANCHTYPE::TFUL) return CorrectionComponent::TFUEL;
    if (branchType == BRANCHTYPE::DMOD) return CorrectionComponent::DMOD;
    if (branchType == BRANCHTYPE::TMOD) return CorrectionComponent::TMOD;
    if (branchType == BRANCHTYPE::RDEP) return CorrectionComponent::ROD_DEPLETION;
    if (branchType == BRANCHTYPE::REFR) return CorrectionComponent::REFERENCE;
    return CorrectionComponent::UNKNOWN;
}

inline CorrectionComponent CorrectionComponentFromHistoryKind(int kind) {
    if (kind == Hk::RHST_UNIT)
        return CorrectionComponent::IISC_RHST;
    if (kind == Hk::CTYPE_INDEP_VEC || kind == Hk::VEC)
        return CorrectionComponent::IISC;
    return CorrectionComponent::UNKNOWN;
}

inline CorrectionComponent CorrectionComponentFromBranchAndKind(BRANCHTYPE branchType, int kind) {
    if (branchType == BRANCHTYPE::SPCT || branchType == BRANCHTYPE::RHST)
        return CorrectionComponentFromHistoryKind(kind);
    return CorrectionComponentFromBranch(branchType);
}

inline const char* CorrectionComponentName(CorrectionComponent component) {
    switch (component) {
    case CorrectionComponent::REFERENCE: return "reference";
    case CorrectionComponent::BPPM: return "bppm";
    case CorrectionComponent::TFUEL: return "tfuel";
    case CorrectionComponent::DMOD: return "dmod";
    case CorrectionComponent::TMOD: return "tmod";
    case CorrectionComponent::ROD_DEPLETION: return "rod_depletion";
    case CorrectionComponent::IISC: return "iisc";
    case CorrectionComponent::IISC_RHST: return "iisc_rhst";
    default: return "unknown";
    }
}

class CrossSection;
class DeltaCrossSection;
class DepletionPoint;
class Model;

/// @brief Number of scalar XS entries before the scattering matrix.
constexpr size_t N_XS_SCALAR = XSSM; // 11

/// @brief Branch sample index map: control-rod type -> burnup key -> point indices.
using Branch = std::unordered_map<int, std::map<int, std::vector<size_t>>>;

/// @brief Branch delta map: control-rod type -> burnup key -> fitted XS delta.
using BranchDelta = std::unordered_map<int, std::map<int, DeltaCrossSection>>;

/// @brief Settings and fitted deltas for spectral or rod-history corrections.
struct HistoryDeltaCorrection {
    /// @brief Fitted correction deltas keyed by correction ctype and burnup.
    BranchDelta delta;
    /// @brief Branch family that owns this correction.
    BRANCHTYPE branch_type = BRANCHTYPE::SPCT;
    /// @brief Legacy HDF kind. New code should classify this through CorrectionComponentFromHistoryKind().
    int kind = 4;
    /// @brief Serialized assembly-data-field tag (HDF `state_fields`); kept for on-disk
    /// compatibility and metadata equality only, not read by any correction at query time.
    int state_field = 9; // AD_TMOD
    /// @brief Isotope indices used by vector corrections; special ids mean fuel/rod fluence.
    std::vector<size_t> vector_isotopes;
    /// @brief Powers applied to vector_isotopes in the correction variable.
    std::vector<int> vector_powers;
};

/// @brief Reference point index map: control-rod type -> burnup key -> point index.
using Reference = std::unordered_map<int, std::map<int, size_t>>;

/// @brief Return a copy of a string with leading and trailing whitespace removed.
/// @param s Input string.
inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(start, end - start + 1);
}

/// @brief Multigroup cross-section container with macroscopic, microscopic, and lumped terms.
class CrossSection {
    friend class Importer;
    friend class Exporter;
    friend class DeltaCrossSection;
    friend class Interpolator;
    friend class Model;

private:
    size_t _ngrp = 0; ///< Number of energy groups.
    size_t _ndat = 0; ///< Elements per group: scalar XS entries plus group-to-group scattering.
    size_t _nmem = 0; ///< Total flat array size: ngrp * ndat.

    milk::Vector<double> _macx; ///< Original macroscopic XS from the lattice calculation.
    milk::Vector<double> _micx; ///< Per-isotope microscopic XS in global isotope order.
    milk::Vector<double> _lmpx; ///< Lumped remainder: macx - sum(micx * iden).

    // Index helper for microscopic cross-section
    size_t idx(const std::string& iso, int ig, XSTYPE xt) const { return iidx.at(iso) * _nmem + ig * _ndat + xt; }
    size_t idx(const std::string& iso, int igs, int ige) const { return iidx.at(iso) * _nmem + igs * _ndat + XSSM + ige; }
    size_t idx(int igs, int ige) const { return igs * _ndat + XSSM + ige; } // Index helper for scattering cross-section
    size_t idx(int ig, XSTYPE xt) const { return ig * _ndat + xt; }         // Index helper for macroscopic cross-section

    /// @brief Allocate and zero-initialize all cross-section storage arrays.
    /// @param nmem Number of flat entries per macroscopic/lumped XS vector.
    void allocate(const size_t nmem) {
        _macx.assign(nmem, 0.0);
        _lmpx.assign(nmem, 0.0);
        if (niso > 0) {
            _micx.assign(niso * nmem, 0.0);
        } else {
            _micx.clear();
        }
    }

    /// @brief Release all cross-section storage.
    void deallocate() {
        _macx.clear();
        _micx.clear();
        _lmpx.clear();
    }

public:
    /// @brief Construct an empty cross-section container.
    CrossSection()                                         = default;
    virtual ~CrossSection()                                = default;
    CrossSection(const CrossSection& other)                = default;
    CrossSection(CrossSection&& other) noexcept            = default;
    CrossSection& operator=(const CrossSection& other)     = default;
    CrossSection& operator=(CrossSection&& other) noexcept = default;

    /// @brief Construct an allocated cross-section container.
    /// @param ngrp Number of energy groups.
    explicit CrossSection(size_t ngrp) : _ngrp(ngrp), _ndat(XSSM + _ngrp), _nmem(_ndat * _ngrp) {
        allocate(_nmem);
    }

    /// @brief Return the entrywise sum of two cross-section containers.
    /// @param other Cross-section container to add.
    CrossSection operator+(const CrossSection& other) const {
        CrossSection result(*this);
        result += other;
        return result;
    }

    /// @brief Add another cross-section container in place.
    /// @param other Cross-section container to add.
    CrossSection& operator+=(const CrossSection& other) {
        if (_ngrp != other._ngrp) {
            throw std::runtime_error("CrossSection energy group mismatch.");
        }
        _macx += other._macx;
        _micx += other._micx;
        _lmpx += other._lmpx;
        return *this;
    }

    /// @brief Return the entrywise difference of two cross-section containers.
    /// @param other Cross-section container to subtract.
    CrossSection operator-(const CrossSection& other) const {
        CrossSection result(*this);
        result -= other;
        return result;
    }

    /// @brief Subtract another cross-section container in place.
    /// @param other Cross-section container to subtract.
    CrossSection& operator-=(const CrossSection& other) {
        if (_ngrp != other._ngrp) {
            throw std::runtime_error("CrossSection energy group mismatch.");
        }
        _macx -= other._macx;
        _micx -= other._micx;
        _lmpx -= other._lmpx;
        return *this;
    }

    /// @brief Return this cross-section container scaled by a scalar.
    /// @param scaler Multiplicative scale factor.
    CrossSection operator*(double scaler) const {
        CrossSection result(*this);
        result *= scaler;
        return result;
    }

    /// @brief Scale all macroscopic, microscopic, and lumped entries in place.
    /// @param scaler Multiplicative scale factor.
    CrossSection& operator*=(double scaler) {
        _macx *= scaler;
        _micx *= scaler;
        _lmpx *= scaler;
        return *this;
    }

    /// @brief Add other*w to this container without allocating a temporary.
    /// @param other Cross-section container to scale and add.
    /// @param w Weight applied to other.
    void addScaled(const CrossSection& other, double w) {
        if (_nmem == 0) return;
        milk::addScaled(_nmem, w, other._macx.data(), 1, _macx.data(), 1);
        milk::addScaled(_nmem, w, other._lmpx.data(), 1, _lmpx.data(), 1);
        if (_micx.size() > 0 && other._micx.size() > 0)
            milk::addScaled(_micx.size(), w, other._micx.data(), 1, _micx.data(), 1);
    }

    /// @brief Return the number of energy groups.
    size_t ngrp() const { return _ngrp; }
    /// @brief Return the number of flat XS entries per group.
    size_t ndat() const { return _ndat; }
    /// @brief Return the total flat XS vector size.
    size_t nmem() const { return _nmem; }

    /// @brief Set macroscopic XS entries to zero.
    void ClearMacroscopic() {
        _macx.fill(0);
    }

    /// @brief Apply a discontinuity factor correction to one energy group.
    /// @param df Discontinuity factor.
    /// @param ig Energy group index.
    void ApplyDF(const double& df, int ig) {
        double ddf = df * df;
        double rdf = 1.0 / df;

        for (size_t i = ig * _ndat; i < (ig + 1) * _ndat; ++i) {
            _macx[i] *= rdf;
            _lmpx[i] *= rdf;
            for (auto& [iso, isoidx] : iidx) {
                _micx[isoidx * _nmem + i] *= rdf;
            }
        }
        _macx[idx(ig, XSTF)] *= (ddf);
        _lmpx[idx(ig, XSTF)] *= (ddf);
        for (auto& [iso, isoidx] : iidx) {
            _micx[idx(iso, ig, XSTF)] *= (ddf);
        }
    }

    /// @brief Compute lumped XS as macx - sum_i(micx_i * iden_i).
    /// @param iden Isotope number densities in global isotope order.
    void CalcLmpx(const milk::Vector<double>& iden) {
        milk::copy(_nmem, _macx.data(), 1, _lmpx.data(), 1);
        const size_t n = _nmem;
        for (size_t i = 0; i < niso; ++i)
            milk::addScaled(n, -iden[i], &_micx[i * _nmem], 1, _lmpx.data(), 1);
    }

    /// @brief Set lumped XS entries to zero.
    void ClearLumped() {
        _lmpx.fill(0.0);
    }

    /// @brief Reconstruct macroscopic XS from microscopic XS and isotope densities.
    /// @param iden Isotope number densities in global isotope order.
    /// @param includeLumped Include the lumped remainder before adding microscopic terms.
    void ReconstructMacroscopicFromMicroscopic(const milk::Vector<double>& iden,
                                               bool                        includeLumped = true) {
        if (iden.size() != niso)
            throw std::runtime_error("CrossSection: isotope density vector size mismatch.");

        const size_t n = _nmem;
        if (includeLumped)
            milk::copy(n, _lmpx.data(), 1, _macx.data(), 1);
        else
            _macx.fill(0.0);

        for (size_t i = 0; i < niso; ++i)
            milk::addScaled(n, iden[i], &_micx[i * _nmem], 1, _macx.data(), 1);
    }

    // Macroscopic XS accessors (original from lattice code)
    double&       maxs(int ig, XSTYPE xt) { return _macx[idx(ig, xt)]; }
    const double& maxs(int ig, XSTYPE xt) const { return _macx[idx(ig, xt)]; }
    double&       maxssm(const int igs, const int ige) { return _macx[idx(igs, ige)]; };
    const double& maxssm(const int igs, const int ige) const { return _macx[idx(igs, ige)]; }

    // Microscopic XS accessors (by isotope name or integer index)
    double&       mixs(const std::string& iso, int ig, XSTYPE xt) { return _micx[idx(iso, ig, xt)]; }
    const double& mixs(const std::string& iso, int ig, XSTYPE xt) const { return _micx[idx(iso, ig, xt)]; }
    double&       mixssm(const std::string& iso, int igs, int ige) { return _micx[idx(iso, igs, ige)]; }
    const double& mixssm(const std::string& iso, int igs, int ige) const { return _micx[idx(iso, igs, ige)]; }
    double&       mixs(int iso, int ig, XSTYPE xt) { return _micx[iso * _nmem + ig * _ndat + xt]; }
    const double& mixs(int iso, int ig, XSTYPE xt) const { return _micx[iso * _nmem + ig * _ndat + xt]; }
    double&       mixssm(int iso, int igs, int ige) { return _micx[iso * _nmem + igs * _ndat + XSSM + ige]; }
    const double& mixssm(int iso, int igs, int ige) const { return _micx[iso * _nmem + igs * _ndat + XSSM + ige]; }

    // Lumped XS accessors (remainder = macx - sum(micx*iden))
    double&       lmpxs(int ig, XSTYPE xt) { return _lmpx[idx(ig, xt)]; }
    const double& lmpxs(int ig, XSTYPE xt) const { return _lmpx[idx(ig, xt)]; }
    double&       lmpxssm(int igs, int ige) { return _lmpx[idx(igs, ige)]; }
    const double& lmpxssm(int igs, int ige) const { return _lmpx[idx(igs, ige)]; }

    // Raw data accessors for flat XS pre-flattening
    bool   has_micx() const { return _micx.size() > 0; }
    size_t micx_size() const { return _micx.size(); }
};

/// @brief Interpolation representation for fitted branch delta XS.
enum InterpolMode { POLY_MODE,
                    SPLINE_MODE };

/// @brief Polynomial or spline coefficients for branch delta cross-sections.
class DeltaCrossSection {
    friend class Importer;
    friend class Exporter;

private:
    size_t              _ngrp   = 0;
    size_t              _nord   = 0;
    CrossSection*       _u      = nullptr;
    InterpolMode        _mode   = POLY_MODE;
    size_t              _ncoeff = 0;
    std::vector<double> _knots;

    /// @brief Allocate the coefficient array.
    /// @param order Number of stored CrossSection coefficients.
    void allocate(size_t order) {
        _u    = order > 0 ? new CrossSection[order] : nullptr;
        _nord = order;
    }

    /// @brief Release the coefficient array.
    void deallocate() {
        delete[] _u;
        _u    = nullptr;
        _nord = _ngrp = 0;
    }

public:
    /// @brief Return the number of energy groups.
    size_t ngrp() const { return _ngrp; }
    /// @brief Return the number of stored polynomial or spline coefficients.
    size_t nord() const { return _nord; }
    /// @brief Return the interpolation representation.
    InterpolMode mode() const { return _mode; }
    /// @brief Return the number of coefficients per spline interval.
    size_t ncoeff() const { return _ncoeff; }
    /// @brief Return spline knot locations.
    const std::vector<double>& knots() const { return _knots; }

    /// @brief Construct an empty delta cross-section.
    DeltaCrossSection() {
    }

    /// @brief Construct polynomial-mode delta coefficients.
    /// @param ngrp Number of energy groups.
    /// @param order Number of polynomial coefficients.
    explicit DeltaCrossSection(size_t ngrp, size_t order = 0)
        : _ngrp(ngrp), _mode(POLY_MODE) {
        if (order > 0) {
            allocate(order);
            for (size_t i = 0; i < _nord; ++i)
                _u[i] = CrossSection(_ngrp);
        }
    }

    /// @brief Construct spline-mode delta coefficients.
    /// @param ngrp Number of energy groups.
    /// @param totalCoeffs Total number of stored coefficients over all intervals.
    /// @param coeffPerInterval Number of polynomial coefficients per interval.
    /// @param knotPoints Spline knot locations.
    DeltaCrossSection(size_t ngrp, size_t totalCoeffs, size_t coeffPerInterval, const std::vector<double>& knotPoints)
        : _ngrp(ngrp), _mode(SPLINE_MODE), _ncoeff(coeffPerInterval), _knots(knotPoints) {
        allocate(totalCoeffs);
        for (size_t i = 0; i < _nord; ++i)
            _u[i] = CrossSection(_ngrp);
    }

    /// @brief Release dynamically allocated coefficient storage.
    ~DeltaCrossSection() { deallocate(); }

    /// @brief Deep-copy delta coefficients.
    /// @param o Source delta cross-section.
    DeltaCrossSection(const DeltaCrossSection& o)
        : _ngrp(o._ngrp), _mode(o._mode), _ncoeff(o._ncoeff), _knots(o._knots) {
        allocate(o._nord);
        for (size_t i = 0; i < _nord; ++i)
            _u[i] = o._u[i];
    }

    /// @brief Move delta coefficient ownership.
    /// @param o Source delta cross-section.
    DeltaCrossSection(DeltaCrossSection&& o) noexcept
        : _ngrp(o._ngrp), _nord(o._nord), _u(o._u),
          _mode(o._mode), _ncoeff(o._ncoeff), _knots(std::move(o._knots)) {
        o._u      = nullptr;
        o._nord   = 0;
        o._ngrp   = 0;
        o._ncoeff = 0;
    }

    /// @brief Deep-copy assign delta coefficients.
    /// @param o Source delta cross-section.
    DeltaCrossSection& operator=(const DeltaCrossSection& o) {
        if (this == &o) return *this;
        if (_nord != o._nord) {
            deallocate();
            allocate(o._nord);
        }
        _ngrp   = o._ngrp;
        _mode   = o._mode;
        _ncoeff = o._ncoeff;
        _knots  = o._knots;
        for (size_t i = 0; i < _nord; ++i)
            _u[i] = o._u[i];
        return *this;
    }

    /// @brief Move-assign delta coefficient ownership.
    /// @param o Source delta cross-section.
    DeltaCrossSection& operator=(DeltaCrossSection&& o) noexcept {
        if (this == &o) return *this;
        deallocate();
        _ngrp     = o._ngrp;
        _nord     = o._nord;
        _u        = o._u;
        _mode     = o._mode;
        _ncoeff   = o._ncoeff;
        _knots    = std::move(o._knots);
        o._u      = nullptr;
        o._nord   = 0;
        o._ngrp   = 0;
        o._ncoeff = 0;
        return *this;
    }

    CrossSection&       operator[](size_t i) { return _u[i]; }
    const CrossSection& operator[](size_t i) const { return _u[i]; }

    /// @brief Evaluate this delta and return a new cross-section container.
    /// @param x Branch state variable.
    CrossSection Delta(double x) const {
        CrossSection r(_ngrp);
        DeltaInto(x, r);
        return r;
    }

    /// @brief Evaluate this delta into caller-owned storage.
    /// @param x Branch state variable.
    /// @param result Output cross-section container.
    void DeltaInto(double x, CrossSection& result) const {
        if (_nord == 0) {
            result *= 0.0;
            return;
        }
        // Evaluate polynomial by Horner's method (i.e., (a₃ * x + a₂) * x + a₁) * x + a₀)
        if (_mode == POLY_MODE) {
            result = _u[_nord - 1];
            for (int p = static_cast<int>(_nord) - 2; p >= 0; --p) {
                result *= x;
                result += _u[p];
            }
        }
        // Evaluate polynomial between spline intervals by Horner's method
        else {
            int nintervals = static_cast<int>(_nord / _ncoeff);
            int interval   = nintervals - 1;
            for (int i = 0; i < nintervals - 1; ++i) {
                if (x < _knots[i + 1]) {
                    interval = i;
                    break;
                }
            }
            double t    = x - _knots[interval];
            int    base = interval * static_cast<int>(_ncoeff);
            result      = _u[base + _ncoeff - 1];
            for (int p = static_cast<int>(_ncoeff) - 2; p >= 0; --p) {
                result *= t;
                result += _u[base + p];
            }
        }
    }
};

/// @brief Indices into the per-depletion-point assembly data array.
enum AssemblyDataIndex {
    AD_APIT = 0,  // Assembly pitch (cm)
    AD_PPIT = 1,  // Pin pitch (cm)
    AD_VOLU = 2,  // Domain volume (cm^2)
    AD_PDEN = 3,  // Power density (W/gHM)
    AD_BURN = 4,  // Burnup (GWd/THM), map key = int(burn*1000)
    AD_KEFF = 5,  // Effective multiplication factor
    AD_KINF = 6,  // Infinite multiplication factor
    AD_BSQU = 7,  // Critical spectrum correction buckling (cm^-2)
    AD_TFUL = 8,  // Fuel temperature (K)
    AD_TMOD = 9,  // Moderator temperature (K)
    AD_DMOD = 10, // Moderator density (g/cm^3)
    AD_BPPM = 11, // Soluble boron concentration (ppm)
    AD_VFRA = 12, // Void fraction of moderator
    AD_PRES = 13, // System pressure (bar)
    AD_HMAS = 14, // Heavy metal mass (g)
    AD_WVFR = 15, // Water volume fraction (estimated from dmod and H/O densities)
    AD_SIZE = 16  // Total number of assembly data fields
};

/// @brief Single library state point with XS, isotope, pin-map, and branch data.
class DepletionPoint {
    friend class Importer;
    friend class Exporter;

public:
    size_t _ngrp; ///< Number of energy groups.
    size_t _npin; ///< Number of pins per assembly side.

    BRANCHTYPE _btyp;                 ///< Branch type represented by this point.
    int        _ctyp;                 ///< Control-rod type.
    int        _trajectory_ctyp;      ///< Control-rod type of the depletion trajectory.
    bool       _trajectory_reference; ///< Main point of a rod-history depletion set.
    double     _fuel_rod_fluence;     ///< Fuel exposure under rodded spectrum [n/cm2].
    double     _rod_fluence;          ///< Rod-material fluence coordinate [n/cm2].

    std::array<double, AD_SIZE> _data = {}; ///< Assembly scalar data indexed by AssemblyDataIndex.

    std::vector<double> _aflx; ///< Average flux by energy group.
    std::vector<double> _gmap; ///< Gamma smeared power map indexed by j * npin + i.
    std::vector<double> _fmap; ///< Groupwise pin flux map indexed by ig * npin * npin + j * npin + i.
    std::vector<double> _chix; ///< Fission neutron spectrum by energy group.
    std::vector<double> _sdfa; ///< Surface discontinuity factor indexed by ig * 4 + side.
    std::vector<double> _pdfa; ///< Corner discontinuity factor indexed by ig * 4 + corner.

    milk::Vector<double> _iden; ///< Isotope number densities in global isotope order.
    CrossSection         _xs;   ///< Cross-section data at this depletion point.

    ~DepletionPoint() = default;

    /// @brief Construct a depletion point with allocated arrays and default state data.
    /// @param ngrp Number of energy groups.
    /// @param npin Number of pins per assembly side.
    /// @param btyp Branch type represented by this point.
    /// @param ctyp Control-rod type.
    explicit DepletionPoint(int ngrp, int npin, const BRANCHTYPE btyp, int ctyp)
        : _ngrp(ngrp), _npin(npin), _btyp(btyp), _ctyp(ctyp),
          _trajectory_ctyp(ctyp), _trajectory_reference(false),
          _fuel_rod_fluence(std::numeric_limits<double>::quiet_NaN()),
          _rod_fluence(std::numeric_limits<double>::quiet_NaN()), _xs(_ngrp) {
        if (_ngrp <= 0) {
            throw std::out_of_range("DepletionPoint: the number of energy group less than zero");
        }
        _data.fill(0.0);
        _data[AD_APIT] = 1.0;
        _data[AD_PPIT] = 1.0;
        _data[AD_VOLU] = 1.0;
        _data[AD_KEFF] = 1.0;
        _data[AD_KINF] = 1.0;
        _data[AD_BSQU] = 1.0;
        _data[AD_TFUL] = 900.0;
        _data[AD_TMOD] = 300.0;
        _data[AD_DMOD] = 1.0;
        _data[AD_PRES] = 1.0;
        _data[AD_HMAS] = 1.0;

        _aflx.resize(_ngrp);
        _gmap.resize(_npin * _npin);
        _fmap.resize(_ngrp * _npin * _npin);
        _chix.resize(_ngrp);
        _sdfa.resize(_ngrp * 4);
        _pdfa.resize(_ngrp * 4);

        if (niso > 0) {
            _iden.assign(niso, 0.0);
        }
    }

    DepletionPoint(const DepletionPoint&)                = default;
    DepletionPoint(DepletionPoint&&) noexcept            = default;
    DepletionPoint& operator=(const DepletionPoint&)     = default;
    DepletionPoint& operator=(DepletionPoint&&) noexcept = default;

    /// @brief Return mutable assembly scalar data by field index.
    /// @param idx Assembly data field.
    double& data(AssemblyDataIndex idx) { return _data[idx]; }
    /// @brief Return assembly scalar data by field index.
    /// @param idx Assembly data field.
    double data(AssemblyDataIndex idx) const { return _data[idx]; }

    /// @brief Return isotope density by isotope ID string.
    /// @param iso Isotope ID string.
    double iden(const std::string& iso) const { return _iden[iidx.at(iso)]; }
    /// @brief Return mutable isotope density by isotope ID string.
    /// @param iso Isotope ID string.
    double& iden(const std::string& iso) { return _iden[iidx.at(iso)]; }
    /// @brief Return isotope density by isotope index.
    /// @param iso Isotope index in global isotope order.
    double iden(int iso) const { return _iden[iso]; }
    /// @brief Return mutable isotope density by isotope index.
    /// @param iso Isotope index in global isotope order.
    double& iden(int iso) { return _iden[iso]; }

    /// @brief Convert burnup in GWd/THM to the integer burnup map key.
    int burnKey() const { return static_cast<int>(_data[AD_BURN] * 1000.0); }
};

/// @brief Assembly model containing depletion points and fitted branch deltas.
class Model {
    friend class Importer;
    friend class Exporter;
    friend class Interpolator;

private:
    int         _id   = 0;
    std::string _name = "NULL";

public:
    /// @brief Return the mutable model ID.
    int& id() { return _id; }
    /// @brief Return the model ID.
    const int& id() const { return _id; }
    /// @brief Return the mutable model name.
    std::string& name() { return _name; }
    /// @brief Return the model name.
    const std::string& name() const { return _name; }

private:
    std::vector<DepletionPoint> _dpts;               ///< Database of depletion points.
    Branch                      _bppm_dpts;          ///< Boron branch point indices by ctype and burnup.
    Branch                      _tful_dpts;          ///< Fuel-temperature branch point indices by ctype and burnup.
    Branch                      _tmod_dpts;          ///< Moderator-temperature branch point indices by ctype and burnup.
    Branch                      _dmod_dpts;          ///< Moderator-density branch point indices by ctype and burnup.
    std::vector<DepletionPoint> _spct_dpts;          ///< Extra HGC state points used for SPCT corrections.
    std::vector<DepletionPoint> _rhst_dpts;          ///< Rod-in history points used for RHST corrections.
    std::vector<DepletionPoint> _rod_depletion_dpts; ///< Rod fluence points used for rod-material depletion.
public:
    /// @brief Return all base branch depletion points.
    const std::vector<DepletionPoint>& Dpts() const { return _dpts; }
    /// @brief Return spectral history correction points.
    const std::vector<DepletionPoint>& SpctDpts() const { return _spct_dpts; }
    /// @brief Return rod-history correction points.
    const std::vector<DepletionPoint>& RhstDpts() const { return _rhst_dpts; }
    /// @brief Return rod-material depletion points.
    const std::vector<DepletionPoint>& RodDepletionDpts() const { return _rod_depletion_dpts; }

    /// @brief Return the number of base branch depletion points.
    int NumPoints() const { return _dpts.size(); }

    /// @brief Return mutable boron branch point map.
    Branch& BppmDpts() { return _bppm_dpts; }
    /// @brief Return mutable boron branch burnup map for a control-rod type.
    /// @param cType Control-rod type.
    std::map<int, std::vector<size_t>>& BppmDpts(const int cType) { return _bppm_dpts[cType]; }
    /// @brief Return mutable boron branch point indices for a control-rod type and burnup.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    std::vector<size_t>& BppmDpts(const int cType, const int burnup) { return _bppm_dpts[cType][burnup]; }

    /// @brief Return mutable fuel-temperature branch point map.
    Branch& TfulDpts() { return _tful_dpts; }
    /// @brief Return mutable fuel-temperature branch burnup map for a control-rod type.
    /// @param cType Control-rod type.
    std::map<int, std::vector<size_t>>& TfulDpts(const int cType) { return _tful_dpts[cType]; }
    /// @brief Return mutable fuel-temperature branch point indices for a control-rod type and burnup.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    std::vector<size_t>& TfulDpts(const int cType, const int burnup) { return _tful_dpts[cType][burnup]; }

    /// @brief Return mutable moderator-density branch point map.
    Branch& DmodDpts() { return _dmod_dpts; }
    /// @brief Return mutable moderator-density branch burnup map for a control-rod type.
    /// @param cType Control-rod type.
    std::map<int, std::vector<size_t>>& DmodDpts(const int cType) { return _dmod_dpts[cType]; }
    /// @brief Return mutable moderator-density branch point indices for a control-rod type and burnup.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    std::vector<size_t>& DmodDpts(const int cType, const int burnup) { return _dmod_dpts[cType][burnup]; }

public:
    /// @brief Construct a model with the default name.
    Model() { _name = "NULL"; }

    ~Model() = default;

    /// @brief Construct a model with a name.
    /// @param name Model name.
    explicit Model(const std::string& name) { _name = name; }

    Model(const Model&) = default;

    Model(Model&&) noexcept = default;

    Model& operator=(const Model&) = default;

    Model& operator=(Model&&) noexcept = default;

    Reference   _refr_dpts;          ///< Reference depletion point indices by ctype and burnup.
    BranchDelta _bppm_delt;          ///< Boron branch delta XS by ctype and burnup.
    BranchDelta _tful_delt;          ///< Fuel-temperature branch delta XS by ctype and burnup.
    BranchDelta _tmod_delt;          ///< Moderator-temperature branch delta XS by ctype and burnup.
    BranchDelta _dmod_delt;          ///< Moderator-density branch delta XS by ctype and burnup.
    BranchDelta _rod_depletion_delt; ///< Rod-material delta XS by ctype at fluence key 0.
    /// @brief Spectral and rod-history correction deltas.
    std::vector<HistoryDeltaCorrection> _history_deltas;

    /// @brief Create a depletion point and register it in the matching branch map.
    /// @param ngrp Number of energy groups.
    /// @param npin Number of pins per assembly side.
    /// @param burn Integer burnup map key.
    /// @param bType Branch type represented by the point.
    /// @param cType Control-rod type.
    DepletionPoint& AddDepletionPoint(const int ngrp, const int npin, const int burn, const BRANCHTYPE bType,
                                      const int cType) {
        _dpts.push_back(DepletionPoint(ngrp, npin, bType, cType));

        const int idx = _dpts.size() - 1;
        if (bType == BRANCHTYPE::REFR)
            _refr_dpts[cType][burn] = idx;
        else if (bType == BRANCHTYPE::BPPM)
            _bppm_dpts[cType][burn].push_back(idx);
        else if (bType == BRANCHTYPE::TFUL)
            _tful_dpts[cType][burn].push_back(idx);
        else if (bType == BRANCHTYPE::DMOD)
            _dmod_dpts[cType][burn].push_back(idx);
        else if (bType == BRANCHTYPE::TMOD)
            _tmod_dpts[cType][burn].push_back(idx);
        else
            _refr_dpts[cType][burn] = idx;
        return _dpts.back();
    }

    /// @brief Return mutable depletion point by flat point index.
    /// @param index Flat index in the depletion point vector.
    DepletionPoint& operator()(const size_t index) { return _dpts[index]; }
    /// @brief Return depletion point by flat point index.
    /// @param index Flat index in the depletion point vector.
    const DepletionPoint& operator()(const size_t index) const { return _dpts[index]; }
    /// @brief Return mutable reference depletion point by control-rod type and burnup.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    DepletionPoint& operator()(const int cType, const int burnup) { return _dpts[_refr_dpts.at(cType).at(burnup)]; }

    /// @brief Return mutable depletion point by flat point index.
    /// @param index Flat index in the depletion point vector.
    DepletionPoint& GetDepletionPoint(const size_t index) { return _dpts[index]; }
    /// @brief Return depletion point by flat point index.
    /// @param index Flat index in the depletion point vector.
    const DepletionPoint& GetDepletionPoint(const size_t index) const { return _dpts[index]; }
    /// @brief Return mutable reference depletion point by control-rod type and burnup.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    DepletionPoint& GetDepletionPoint(const int cType, const int burnup) { return _dpts[_refr_dpts.at(cType).at(burnup)]; }
    /// @brief Return reference depletion point by control-rod type and burnup.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    const DepletionPoint& GetDepletionPoint(const int cType, const int burnup) const { return _dpts[_refr_dpts.at(cType).at(burnup)]; }

    /// @brief Return the branch point map for a branch type.
    /// @param bType Branch type selector.
    Branch& GetBranch(const BRANCHTYPE bType) {
        if (bType == BRANCHTYPE::BPPM) return _bppm_dpts;
        if (bType == BRANCHTYPE::TFUL) return _tful_dpts;
        if (bType == BRANCHTYPE::DMOD) return _dmod_dpts;
        if (bType == BRANCHTYPE::TMOD) return _tmod_dpts;
        return _bppm_dpts;
    }

    /// @brief Return the const branch point map for a branch type.
    /// @param bType Branch type selector.
    const Branch& GetBranch(const BRANCHTYPE bType) const {
        if (bType == BRANCHTYPE::BPPM) return _bppm_dpts;
        if (bType == BRANCHTYPE::TFUL) return _tful_dpts;
        if (bType == BRANCHTYPE::DMOD) return _dmod_dpts;
        if (bType == BRANCHTYPE::TMOD) return _tmod_dpts;
        return _bppm_dpts;
    }

    /// @brief Fill reference XS, isotope density, state data, and optional flux at a burnup.
    /// @param xs Output reference cross-section.
    /// @param iden Output isotope densities.
    /// @param data Output assembly state data.
    /// @param flux Optional output reference average flux.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    void FillReferenceState(CrossSection& xs, milk::Vector<double>& iden,
                            std::array<double, AD_SIZE>& data,
                            std::vector<double>*         flux,
                            int cType, int burnup) const {
        const auto& refrMap = _refr_dpts.at(cType);

        auto refrHi = refrMap.lower_bound(burnup);
        auto refrLo = refrHi;
        if (refrHi == refrMap.end()) {
            refrLo = std::prev(refrMap.end());
            refrHi = refrLo;
        } else if (refrHi == refrMap.begin()) {
            refrLo = refrMap.begin();
        } else {
            refrLo = std::prev(refrHi);
        }

        const DepletionPoint& loDpt = _dpts[refrLo->second];
        const DepletionPoint& hiDpt = _dpts[refrHi->second];

        xs   = loDpt._xs;
        iden = loDpt._iden;
        data = loDpt._data;
        if (flux)
            *flux = loDpt._aflx;

        if (refrLo == refrHi)
            return;

        const double rawBurn = burnup / 1000.0;
        const double denom   = hiDpt._data[AD_BURN] - loDpt._data[AD_BURN];
        const double frac    = std::abs(denom) > 0.0 ? (rawBurn - loDpt._data[AD_BURN]) / denom : 0.0;

        xs.addScaled(hiDpt._xs, frac);
        xs.addScaled(loDpt._xs, -frac);
        for (size_t i = 0; i < niso; ++i)
            iden[i] += (hiDpt._iden[i] - loDpt._iden[i]) * frac;
        for (size_t i = 0; i < data.size(); ++i)
            data[i] += (hiDpt._data[i] - loDpt._data[i]) * frac;
        if (flux) {
            for (size_t i = 0; i < flux->size() && i < hiDpt._aflx.size(); ++i)
                (*flux)[i] += (hiDpt._aflx[i] - (*flux)[i]) * frac;
        }
    }

    /// @brief Fill caller-owned workspaces with branch-corrected XS and isotope densities.
    /// @param xs      [in/out] Scratch CrossSection, resized if needed. Filled with result.
    /// @param iden    [in/out] Scratch isotope density vector, resized if needed. Filled with result.
    /// @param delta   [in/out] Scratch CrossSection for branch delta evaluation.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    /// @param bppm Soluble boron concentration in ppm.
    /// @param tfuel Fuel temperature in K.
    /// @param dmod Moderator density in g/cm^3.
    void FillCrossSection(CrossSection& xs, milk::Vector<double>& iden, CrossSection& delta,
                          int cType, int burnup, double bppm, double tfuel, double dmod) const {
        auto& refrMap = _refr_dpts.at(cType);

        // 1. Find reference depletion point
        auto refrHi = refrMap.lower_bound(burnup);
        auto refrLo = refrHi;
        if (refrHi == refrMap.end()) {
            refrLo = std::prev(refrMap.end());
            refrHi = refrLo;
        } else if (refrHi == refrMap.begin()) {
            refrLo = refrMap.begin();
        } else {
            refrLo = std::prev(refrHi);
        }
        const DepletionPoint& loDpt = _dpts[refrLo->second];
        const DepletionPoint& hiDpt = _dpts[refrHi->second];

        // Copy reference into workspace (single copy, reuses allocation)
        xs   = loDpt._xs;
        iden = loDpt._iden;

        // Burnup interpolation
        if (refrLo != refrHi) {
            double rawBurn = burnup / 1000.0;
            double frac    = (rawBurn - loDpt._data[AD_BURN]) / (hiDpt._data[AD_BURN] - loDpt._data[AD_BURN]);
            xs.addScaled(hiDpt._xs, frac);
            xs.addScaled(loDpt._xs, -frac);
            for (size_t i = 0; i < niso; ++i)
                iden[i] += (hiDpt._iden[i] - loDpt._iden[i]) * frac;
        }

        // Calculate H, B, O isotope density
        // N_A / M_H2O = 0.6022 / 18.015 = 0.033427699
        // nB = dmod * wvfr * C_B (ppm, 10^-6) * N_A / M_B
        // C_B * N_A / M_B = bppm * 10^-6 * 0.6022 / 10.81 = 5.5707678E-8
        double nH2O = dmod * loDpt._data[AD_WVFR] * 0.033427699;
        double nH   = 2.0 * nH2O;
        double nO   = nH2O;
        double nB   = dmod * loDpt._data[AD_WVFR] * bppm * 5.5707678E-8;

        if (_bppm_delt.contains(cType) && !_bppm_delt.at(cType).empty()) {
            const auto& m  = _bppm_delt.at(cType);
            auto        it = m.lower_bound(burnup);
            it             = (it == m.end()) ? std::prev(it) : it;
            it->second.DeltaInto(nB, delta);
            xs += delta;
        }
        if (_tful_delt.contains(cType) && !_tful_delt.at(cType).empty()) {
            const auto& m  = _tful_delt.at(cType);
            auto        it = m.lower_bound(burnup);
            it             = (it == m.end()) ? std::prev(it) : it;
            it->second.DeltaInto(sqrt(tfuel), delta);
            xs += delta;
        }
        if (_dmod_delt.contains(cType) && !_dmod_delt.at(cType).empty()) {
            const auto& m  = _dmod_delt.at(cType);
            auto        it = m.lower_bound(burnup);
            it             = (it == m.end()) ? std::prev(it) : it;
            it->second.DeltaInto(dmod, delta);
            xs += delta;
        }

        iden[iH1]  = nH;
        iden[iO16] = nO;
        iden[iB10] = nB;
    }

    /// @brief Fill reference average flux at a burnup by reference interpolation.
    /// @param flux Output average flux by energy group.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    void FillReferenceFlux(std::vector<double>& flux, int cType, int burnup) const {
        flux.clear();
        auto mapIt = _refr_dpts.find(cType);
        if (mapIt == _refr_dpts.end() || mapIt->second.empty())
            return;

        const auto& refrMap = mapIt->second;
        auto        refrHi  = refrMap.lower_bound(burnup);
        auto        refrLo  = refrHi;
        if (refrHi == refrMap.end()) {
            refrLo = std::prev(refrMap.end());
            refrHi = refrLo;
        } else if (refrHi == refrMap.begin()) {
            refrLo = refrMap.begin();
        } else {
            refrLo = std::prev(refrHi);
        }

        const DepletionPoint& loDpt = _dpts[refrLo->second];
        const DepletionPoint& hiDpt = _dpts[refrHi->second];
        flux                        = loDpt._aflx;
        if (refrLo != refrHi && hiDpt.burnKey() != loDpt.burnKey()) {
            const double frac = static_cast<double>(burnup - loDpt.burnKey()) /
                                static_cast<double>(hiDpt.burnKey() - loDpt.burnKey());
            for (size_t i = 0; i < flux.size() && i < hiDpt._aflx.size(); ++i)
                flux[i] += (hiDpt._aflx[i] - flux[i]) * frac;
        }
    }

    /// @brief Return branch-corrected XS using thread-local workspaces.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    /// @param bppm Soluble boron concentration in ppm.
    /// @param tfuel Fuel temperature in K.
    /// @param dmod Moderator density in g/cm^3.
    /// @param iden_out Optional isotope density output.
    CrossSection GetCrossSection(int cType, int burnup, double bppm, double tfuel,
                                 double dmod, milk::Vector<double>* iden_out = nullptr) const {
        static thread_local CrossSection         ws_xs, ws_delta;
        static thread_local milk::Vector<double> ws_iden;
        FillCrossSection(ws_xs, ws_iden, ws_delta, cType, burnup, bppm, tfuel, dmod);
        if (iden_out) *iden_out = ws_iden;
        return ws_xs;
    }

    /// @brief Apply ctype-specific rod-material depletion delta at a fluence point.
    /// @param xs Cross-section state to correct in place.
    /// @param delta Caller-owned scratch CrossSection.
    /// @param cType Rod control type.
    /// @param fluence Integrated scalar flux [n/cm2].
    void ApplyRodDepletion(CrossSection& xs, CrossSection& delta, int cType, double fluence) const {
        if (cType <= 0 || fluence <= 0.0)
            return;
        auto ctypeIt = _rod_depletion_delt.find(cType);
        if (ctypeIt == _rod_depletion_delt.end() || ctypeIt->second.empty())
            return;

        const auto& bmap = ctypeIt->second;
        const auto  it   = bmap.begin();
        double      x    = fluence;
        if (it->second.knots().size() >= 2) {
            const auto& knots = it->second.knots();
            x                 = std::clamp(x, knots.front(), knots.back());
        }
        it->second.DeltaInto(x, delta);
        xs += delta;
    }
};
} // namespace Chiffon
