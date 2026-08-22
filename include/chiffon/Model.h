#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "milk.h"

#include "highfive/highfive.hpp"
#include "nlohmann/json.hpp"

namespace Chiffon {

inline constexpr char HDF_VERSION[] = "3.0.0";
inline constexpr double SPECTRAL_LOG_DENSITY_FLOOR = 1.0e-12;

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

/// @brief Serialises every write to the four globals above.
///
/// The registry is process-global but is (re)built by each XSSet as it loads
/// its cross-section library. That is harmless while one solver owns the
/// process and fatal once several instances run as threads in one process
/// (--batch-mode): two concurrent `iidx.clear()` + `emplace` sequences corrupt
/// the map's heap blocks and the run dies inside malloc() a fraction of a
/// second after start. The content rebuilt is canonical -- byte-identical on
/// every call, and validated against `isotopeIds` before it is written -- so
/// serialising the write costs nothing measurable and changes no result.
inline std::mutex registryMutex;

/// @brief Rebuild the isotope index map. Caller must hold registryMutex.
inline void RebuildIndexLocked() {
    iidx.clear();
    niso = isotopeIds.size();
    for (size_t i = 0; i < isotopeIds.size(); ++i)
        iidx.emplace(isotopeIds[i], i);
}

/// @brief Initialize isotope indices and load depletion matrices.
/// @param isotopeDataDir Directory containing dep_decay.csv and dep_trans.csv.
inline void Initialize(const std::filesystem::path& isotopeDataDir) {
    std::lock_guard<std::mutex> guard(registryMutex);
    RebuildIndexLocked();

    const std::vector<std::string> labels(isotopeIds.begin(), isotopeIds.end());
    depDecay = milk::LabeledMatrixFromCSV(isotopeDataDir / "dep_decay.csv", labels, labels);
    depTrans = milk::LabeledMatrixFromCSV(isotopeDataDir / "dep_trans.csv", labels, labels);
}

/// @brief Load the depletion matrices once, whichever instance gets there first.
///
/// The previous form -- take the lock, test `depDecay.size()`, *drop the lock*,
/// then call Initialize() -- is the textbook broken double-checked lock. The
/// window between the two critical sections is exactly what the lock was there
/// to close: in --batch-mode every instance can observe size()==0, all of them
/// fall through, and Initialize() then re-runs LabeledMatrixFromCSV() and
/// reassigns depDecay/depTrans once per instance. Each rebuild is serialised and
/// canonical, so nothing was ever *observed* torn -- but only because no reader
/// happens to touch the matrices during startup. That is an accident of
/// scheduling, not a guarantee, and the "replaces a check-then-act race" comment
/// claimed otherwise.
///
/// std::call_once gives the real guarantee: exactly one thread runs the
/// initialiser, every other caller blocks until it has completed, and that
/// completion synchronises-with their return, so the matrices they go on to read
/// are fully published.
///
/// main() calls Isotope::Initialize() directly before any solver is constructed
/// (src/main.cpp). That call is load-bearing: it is what populates the matrices
/// for the single-instance path, and it does not set this flag -- hence the
/// size() test inside the initialiser, without which the first XSSet would redo
/// the CSV load for nothing. The test is safe here because it runs inside
/// call_once (single-threaded by construction) and main()'s Initialize
/// happens-before any thread that could reach this point.
inline void EnsureInitialized(const std::filesystem::path& isotopeDataDir) {
    static std::once_flag initialised;
    std::call_once(initialised, [&isotopeDataDir] {
        if (depDecay.size() == 0)
            Initialize(isotopeDataDir);
    });
}
} // namespace Isotope

/// @brief Composition coordinate used by one spectral-history term.
enum class SpectralCoordinate : int {
    Density    = 0,
    LogDensity = 1,
    /// @brief PROBE: square root of the density.
    SqrtDensity = 5,
    /// @brief PROBE: ratio of a resonance absorber's thermal cross section to a
    /// 1/v absorber's, evaluated on the base+branch state. A dimensionless
    /// stand-in for the intra-group spectrum shape.
    SpectralIndex = 7,
    /// @brief PROBE: the spectral index times the isotope's deviation from the
    /// reference inventory — the composition x spectrum cross term. Vanishes at
    /// nominal composition, so it cannot encroach on the branch layer.
    SpectralIndexInteraction = 8,
    /// @brief PROBE: density weighted by the thermal share of the flux.
    ThermalWeighted = 2,
    /// @brief PROBE: density weighted by the fast share of the flux.
    FastWeighted = 3,
    /// @brief PROBE: log(psi_th/psi_fast) times the density deviation. Vanishes
    /// when the composition sits on the reference, so it cannot absorb the
    /// branch layer's own condition dependence.
    FluxRatioInteraction = 4,
    /// @brief PROBE: log of one isotope's relative depletion divided by
    /// another's, both measured against the burnup reference. Isotopes that
    /// live in different pins (Gd in the Gd rods, Pu/U in the fuel rods) burn
    /// at rates set by the *spatial* flux distribution, so their relative rate
    /// carries information about the isotope profile p_i(r) that the node
    /// average N_i throws away. A linear combination of the two deviations
    /// cannot represent this ratio.
    RelativeBurnRatio = 9,
    /// @brief PROBE: (branch coordinate - its reference) x (density - reference
    /// density), one enumerator per condition axis. Both factors are centered, so
    /// the column is exactly zero at the reference condition AND at the reference
    /// composition: it can impersonate neither the branch layer nor a plain
    /// composition column, and fires only where both deviate. This is the term
    /// that lets the history correction tilt a branch slope.
    /// The values start at 12 because 6/10/11 were briefly written by an
    /// experimental build with different semantics; reusing them would let such a
    /// library load and apply the coefficient to the wrong axis.
    BppmInteraction = 12,
    TfulInteraction = 13,
    DmodInteraction = 14,
    /// @brief PROBE: squared centered log deviation, (log N - log N_ref)^2. A
    /// monotone transform of N adds nothing a linear fit cannot already reach
    /// through N itself; only curvature does. This asks whether the response is
    /// actually linear in log N or still bends.
    LogDeviationSquared = 15,
    /// @brief PROBE: N_a / (N_a + N_partner). Genuinely bivariate, but its
    /// first-order expansion already lives in the two density columns, so only
    /// its curvature is new information.
    FissileFraction = 16,
    /// @brief PROBE: N_ref / N. Narrow-resonance asymptotics motivate an inverse
    /// dependence; around the reference it is mostly another quadratic, and it
    /// gives extreme leverage to rows where the inventory is nearly exhausted.
    InverseRatio = 17,
    /// @brief PROBE: (N / N_ref)^(1/3). For a burnable absorber consumed by a
    /// receding front the absorbing interface scales as a power of the remaining
    /// inventory; the exponent is geometry-dependent and not derived here.
    CubeRootRatio = 18,
    /// @brief PROBE: N / (1 + N/N_ref), a saturating form for the fission-product
    /// poisons whose equilibrium concentration is a rational function of flux.
    SaturatingRatio = 19,
    /// @brief PROBE: (branch coordinate - its reference) x control-rod fluence,
    /// one enumerator per condition axis. The rodded branch tables are built from
    /// a fresh rod and evaluated on a burned one; this lets the condition slope
    /// depend on how burned the rod is. Zero at the reference condition, so it
    /// cannot disturb the fresh-rod branch, and zero with the rod withdrawn.
    BppmRodAge = 20,
    TfulRodAge = 21,
    DmodRodAge = 22,
};

/// @brief The condition axis a rod-age cross term rides on, or -1. Same axis
/// numbering as BranchAxisOf.
[[nodiscard]] inline int RodAgeAxisOf(SpectralCoordinate coordinate) {
    switch (coordinate) {
    case SpectralCoordinate::BppmRodAge: return 0;
    case SpectralCoordinate::TfulRodAge: return 1;
    case SpectralCoordinate::DmodRodAge: return 2;
    default: return -1;
    }
}

/// @brief Rod fluence is O(1e21); scale it so the design column is O(1) before the
/// fit's own column normalisation, and so the stored coefficient stays readable.
inline constexpr double ROD_AGE_SCALE = 1.0e-21;

/// @brief Evaluate one of the reference-ratio coordinate forms.
/// @param coordinate Which form; anything else returns the saturating form.
/// @param now Node or sample density, already floored away from zero.
/// @param ref Reference-inventory density at the same burnup, likewise floored.
/// @return The coordinate value.
/// Lives here rather than in the fitter so the Chiffon fit and the RASBERY
/// runtime evaluate one expression, not two that can drift apart.
[[nodiscard]] inline double RatioFormOf(SpectralCoordinate coordinate,
                                        double now, double ref) {
    if (coordinate == SpectralCoordinate::LogDeviationSquared) {
        const double d = std::log(now / ref);
        return d * d;
    }
    if (coordinate == SpectralCoordinate::InverseRatio)
        return ref / now;
    if (coordinate == SpectralCoordinate::CubeRootRatio)
        return std::cbrt(now / ref);
    return now / (1.0 + now / ref);
}

/// @brief Whether a serialized integer names a coordinate this build implements.
/// An explicit list, not a range: unassigned values must be rejected at load
/// rather than falling through to plain density.
[[nodiscard]] inline bool IsKnownSpectralCoordinate(int value) {
    switch (static_cast<SpectralCoordinate>(value)) {
    case SpectralCoordinate::Density:
    case SpectralCoordinate::LogDensity:
    case SpectralCoordinate::ThermalWeighted:
    case SpectralCoordinate::FastWeighted:
    case SpectralCoordinate::FluxRatioInteraction:
    case SpectralCoordinate::SqrtDensity:
    case SpectralCoordinate::SpectralIndex:
    case SpectralCoordinate::SpectralIndexInteraction:
    case SpectralCoordinate::RelativeBurnRatio:
    case SpectralCoordinate::BppmInteraction:
    case SpectralCoordinate::TfulInteraction:
    case SpectralCoordinate::DmodInteraction:
    case SpectralCoordinate::LogDeviationSquared:
    case SpectralCoordinate::FissileFraction:
    case SpectralCoordinate::InverseRatio:
    case SpectralCoordinate::CubeRootRatio:
    case SpectralCoordinate::SaturatingRatio:
    case SpectralCoordinate::BppmRodAge:
    case SpectralCoordinate::TfulRodAge:
    case SpectralCoordinate::DmodRodAge:
        return true;
    }
    return false;
}

/// @brief The condition axis a cross term rides on, or -1 if the coordinate is
/// not one of the three centered branch interactions. Matches the branch order
/// used by the runtime scalar-branch table.
[[nodiscard]] inline int BranchAxisOf(SpectralCoordinate coordinate) {
    switch (coordinate) {
    case SpectralCoordinate::BppmInteraction: return 0;
    case SpectralCoordinate::TfulInteraction: return 1;
    case SpectralCoordinate::DmodInteraction: return 2;
    default: return -1;
    }
}

/// @brief One isotope coordinate in the spectral-history regression basis.
struct SpectralTerm {
    size_t             isotope   = 0;
    SpectralCoordinate coordinate = SpectralCoordinate::Density;
    /// @brief Second isotope for two-isotope coordinates; unused (SIZE_MAX)
    /// otherwise.
    size_t partner = static_cast<size_t>(-1);
    bool operator==(const SpectralTerm&) const = default;
};

using SpectralBasis = std::vector<SpectralTerm>;

using namespace Isotope;

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
enum BRANCHTYPE : int {
    REFR            = 0, // Reference (base depletion)
    BPPM            = 1, // Boron concentration
    TFUL            = 2, // Fuel temperature
    DMOD            = 3, // Moderator density
    TMOD            = 4, // Moderator temperature
    SPECTRAL_HISTORY = 5, // Spectral-history input
    ROD_HISTORY     = 6, // Rodded depletion history input
    COMBO           = 8  // Multi-axis instantaneous branch (validation-only)
};

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

using BurnupDelta = std::map<int, DeltaCrossSection>;

/// @brief One ctype-independent spectral-history correction term.
struct SpectralHistoryCorrection {
    BurnupDelta delta;
    SpectralTerm term;
    /// @brief PROBE: scale this term by the node rod fraction.
    bool rod_scaled = false;
};

/// @brief Reference point index map: control-rod type -> burnup key -> point index.
using Reference = std::unordered_map<int, std::map<int, size_t>>;

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
    bool       _trajectory_reference; ///< Fresh-absorber member of a rod-depletion pair.
    bool       _nondepleted = false;  ///< Frozen-absorber (RODNONDEPL) counterfactual point.
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
    int         _history_partner = -1;
    std::string _history_partner_name;

public:
    /// @brief Return the mutable model ID.
    int& id() { return _id; }
    /// @brief Return the model ID.
    const int& id() const { return _id; }
    /// @brief Return the mutable model name.
    std::string& name() { return _name; }
    /// @brief Return the model name.
    const std::string& name() const { return _name; }
    /// @brief Index of the rodded-depletion partner model, or -1 when this model
    /// carries no depletion-history blend.
    int& history_partner() { return _history_partner; }
    /// @brief Index of the rodded-depletion partner model (-1 when absent).
    [[nodiscard]] const int& history_partner() const { return _history_partner; }
    /// @brief Name of the rodded-depletion partner, resolved to an index after
    /// every model has been read.
    std::string& history_partner_name() { return _history_partner_name; }
    /// @brief Name of the rodded-depletion partner model.
    [[nodiscard]] const std::string& history_partner_name() const {
        return _history_partner_name;
    }

private:
    std::vector<DepletionPoint> _dpts;               ///< Database of depletion points.
    Branch                      _bppm_dpts;          ///< Boron branch point indices by ctype and burnup.
    Branch                      _tful_dpts;          ///< Fuel-temperature branch point indices by ctype and burnup.
    Branch                      _dmod_dpts;          ///< Moderator-density branch point indices by ctype and burnup.
    std::vector<DepletionPoint> _spectral_history_dpts; ///< Spectral-history input points.
    std::vector<DepletionPoint> _rod_history_dpts;   ///< Rodded depletion history points.
    std::vector<DepletionPoint> _rod_depletion_dpts; ///< Rod fluence points used for rod-material depletion.
public:
    /// @brief Return all base branch depletion points.
    const std::vector<DepletionPoint>& Dpts() const { return _dpts; }
    /// @brief Return spectral history correction points.
    const std::vector<DepletionPoint>& SpectralHistoryDpts() const {
        return _spectral_history_dpts;
    }
    /// @brief Return rodded depletion history points.
    const std::vector<DepletionPoint>& RodHistoryDpts() const {
        return _rod_history_dpts;
    }

    /// @brief Return the number of base branch depletion points.
    int NumPoints() const { return _dpts.size(); }

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
    BranchDelta _dmod_delt;          ///< Moderator-density branch delta XS by ctype and burnup.
    BranchDelta _rod_depletion_delt; ///< Rod-material delta XS by ctype and thermal fluence.
    /// Rod-age branch residual curves over rod fluence, per axis
    /// (0 = bppm as B-10eq density x1e12, 1 = sqrt(Tfuel) x1e6, 2 = dmod x1e9);
    /// int key = quantized branch coordinate.
    std::array<BranchDelta, 3> _rod_depletion_branch;
    /// @brief Ctype-independent spectral-history correction terms.
    std::vector<SpectralHistoryCorrection> _spectral_history;

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
        else if (bType != BRANCHTYPE::TMOD &&
                 bType != BRANCHTYPE::COMBO)
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

    /// @brief Return the branch point map for a fitted scalar branch type.
    /// @param bType Branch type selector.
    Branch& GetBranch(const BRANCHTYPE bType) {
        if (bType == BRANCHTYPE::BPPM) return _bppm_dpts;
        if (bType == BRANCHTYPE::TFUL) return _tful_dpts;
        if (bType == BRANCHTYPE::DMOD) return _dmod_dpts;
        throw std::logic_error("GetBranch: unsupported branch type.");
    }

    /// @brief Return the branch point map for a fitted scalar branch type.
    /// @param bType Branch type selector.
    const Branch& GetBranch(const BRANCHTYPE bType) const {
        if (bType == BRANCHTYPE::BPPM) return _bppm_dpts;
        if (bType == BRANCHTYPE::TFUL) return _tful_dpts;
        if (bType == BRANCHTYPE::DMOD) return _dmod_dpts;
        throw std::logic_error("GetBranch: unsupported branch type.");
    }

    /// @brief Apply a burnup-interpolated scalar branch delta in place.
    /// @param xs Cross-section state to correct in place.
    /// @param delta Caller-owned scratch CrossSection.
    /// @param branch Scalar branch delta table.
    /// @param cType Control-rod type.
    /// @param burnup Integer burnup map key.
    /// @param x Scalar branch coordinate.
    void ApplyScalarBranchDelta(CrossSection& xs, CrossSection& delta,
                                const BranchDelta& branch,
                                int cType, int burnup, double x) const {
        auto cIt = branch.find(cType);
        if (cIt == branch.end() || cIt->second.empty())
            return;
        const auto& m  = cIt->second;
        auto        hi = m.lower_bound(burnup);
        auto        lo = hi;
        if (hi == m.end()) {
            lo = std::prev(m.end());
            hi = lo;
        } else if (hi == m.begin()) {
            lo = hi;
        } else if (hi->first != burnup) {
            lo = std::prev(hi);
        }

        auto applyOne = [&](const DeltaCrossSection& dxs, double burnupScale) {
            if (burnupScale == 0.0)
                return;
            dxs.DeltaInto(x, delta);
            delta *= burnupScale;
            xs += delta;
        };

        if (lo != hi && hi->first != lo->first) {
            const double frac = static_cast<double>(burnup - lo->first) /
                                static_cast<double>(hi->first - lo->first);
            applyOne(lo->second, 1.0 - frac);
            applyOne(hi->second, frac);
        } else {
            applyOne(lo->second, 1.0);
        }
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
        const double frac    = (rawBurn - loDpt._data[AD_BURN]) /
                            (hiDpt._data[AD_BURN] - loDpt._data[AD_BURN]);

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
                          int cType, int burnup, double bppm, double tfuel,
                          double dmod) const {
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

        ApplyScalarBranchDelta(xs, delta, _bppm_delt, cType, burnup, nB);
        ApplyScalarBranchDelta(xs, delta, _tful_delt, cType, burnup, sqrt(tfuel));
        ApplyScalarBranchDelta(xs, delta, _dmod_delt, cType, burnup, dmod);

        iden[iH1]  = nH;
        iden[iO16] = nO;
        iden[iB10] = nB;
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

    /// @brief Nominal rod-material fluence [n/cm2] of the ctype reference trajectory at a burnup.
    /// Rodded reference points carry reconstructed fluence; other references return zero.
    [[nodiscard]] double ReferenceRodFluence(int cType, int burnup) const {
        auto it = _refr_dpts.find(cType);
        if (it == _refr_dpts.end() || it->second.empty())
            return 0.0;
        const auto& bmap = it->second;
        auto        hi   = bmap.lower_bound(burnup);
        auto        lo   = hi;
        if (hi == bmap.end()) {
            lo = std::prev(bmap.end());
            hi = lo;
        } else if (hi == bmap.begin()) {
            lo = bmap.begin();
        } else if (hi->first != burnup) {
            lo = std::prev(hi);
        }
        const double fLo = _dpts[lo->second]._rod_fluence;
        const double fHi = _dpts[hi->second]._rod_fluence;
        if (!std::isfinite(fLo) || !std::isfinite(fHi))
            return 0.0;
        if (lo == hi || hi->first == lo->first)
            return std::max(0.0, fLo);
        const double frac = static_cast<double>(burnup - lo->first) /
                            static_cast<double>(hi->first - lo->first);
        return std::max(0.0, fLo + frac * (fHi - fLo));
    }

    /// @brief Corner-to-surface DF consistency ratio (sdfa/pdfa) of the ctype-0
    /// reference trajectory, burnup-interpolated.  The PPR corner-balance flux is
    /// a heterogeneous corner estimate; a node whose expansion works in the
    /// surface-DF-folded (SET) space must rescale it by sdfa/pdfa before use --
    /// the same physics MASTER carries through its CRADF corner factors.  The
    /// lattice HGCs write side-uniform factors, so side/corner index 0 is used.
    /// Returns 1.0 whenever either factor is absent or outside the physical
    /// window (DeCART emits garbage on zero-current surfaces).
    [[nodiscard]] double CornerToSurfaceDFRatio(int ig, int burnup) const {
        auto it = _refr_dpts.find(0);
        if (it == _refr_dpts.end() || it->second.empty())
            return 1.0;
        const auto& bmap = it->second;
        auto        hi   = bmap.lower_bound(burnup);
        auto        lo   = hi;
        if (hi == bmap.end()) {
            lo = std::prev(bmap.end());
            hi = lo;
        } else if (hi == bmap.begin()) {
            lo = bmap.begin();
        } else if (hi->first != burnup) {
            lo = std::prev(hi);
        }
        auto ratioOf = [ig](const DepletionPoint& p) {
            const size_t k = static_cast<size_t>(ig) * 4;
            if (p._sdfa.size() <= k || p._pdfa.size() <= k)
                return 1.0;
            const double sd = p._sdfa[k];
            const double pd = p._pdfa[k];
            const bool   ok = sd > 0.05 && sd < 20.0 && pd > 0.05 && pd < 20.0;
            return ok ? sd / pd : 1.0;
        };
        const double rLo = ratioOf(_dpts[lo->second]);
        const double rHi = ratioOf(_dpts[hi->second]);
        if (lo == hi || hi->first == lo->first)
            return rLo;
        const double frac = static_cast<double>(burnup - lo->first) /
                            static_cast<double>(hi->first - lo->first);
        return rLo + frac * (rHi - rLo);
    }

    /// @brief Apply the relative rod-material depletion correction.
    /// @param xs Cross-section state to correct in place.
    /// @param delta Caller-owned scratch cross section.
    /// @param cType Control-rod type.
    /// @param fluence Integrated thermal flux [n/cm2].
    void ApplyRodDepletion(
        CrossSection& xs, CrossSection& delta, int cType,
        double fluence, int burnup, double uBppm,
        double uTful, double uDmod) const {
        if (cType <= 0)
            return;
        const auto ctypeIt = _rod_depletion_delt.find(cType);
        if (ctypeIt == _rod_depletion_delt.end())
            return;
        const auto curve = ctypeIt->second.find(0);
        if (curve == ctypeIt->second.end())
            return;

        const double referenceFluence =
            ReferenceRodFluence(cType, burnup);
        if (fluence <= 0.0 && referenceFluence <= 0.0)
            return;

        curve->second.DeltaInto(std::max(fluence, 0.0), delta);
        xs += delta;
        if (referenceFluence > 0.0) {
            curve->second.DeltaInto(referenceFluence, delta);
            xs -= delta;
        }
        ApplyRodDepletionBranchResiduals(
            xs, delta, cType, fluence, uBppm, uTful, uDmod);
    }

    /// Condition dependence of the rod-age worth. Each axis stores
    /// residual-vs-fluence curves at the pair decks' branch displacements (zero at the
    /// nominal knot); the node's displacement interpolates linearly between knot curves.
    void ApplyRodDepletionBranchResiduals(CrossSection& xs, CrossSection& delta, int cType,
                                          double fluence, double uBppm, double uTful,
                                          double uDmod) const {
        static constexpr double kUScale[3] = {1.0e12, 1.0e6, 1.0e9};
        const double            u[3]       = {uBppm, uTful, uDmod};
        for (int a = 0; a < 3; ++a) {
            if (u[a] < 0.0)
                continue;
            auto cIt = _rod_depletion_branch[a].find(cType);
            if (cIt == _rod_depletion_branch[a].end() || cIt->second.size() < 2)
                continue;
            const auto&  m   = cIt->second;
            const double key = u[a] * kUScale[a];
            auto         hi  = m.lower_bound(static_cast<int>(std::llround(key)));
            auto         lo  = hi;
            if (hi == m.end()) {
                lo = std::prev(m.end());
                hi = lo;
            } else if (hi == m.begin()) {
                lo = hi;
            } else {
                lo = std::prev(hi);
            }
            auto applyOne = [&](const DeltaCrossSection& dxs, double w, double x) {
                if (w == 0.0)
                    return;
                dxs.DeltaInto(std::max(x, 0.0), delta);
                delta *= w;
                xs += delta;
            };
            if (lo != hi && hi->first != lo->first) {
                const double frac = (key - lo->first) /
                                    static_cast<double>(hi->first - lo->first);
                applyOne(lo->second, 1.0 - frac, fluence);
                applyOne(hi->second, frac, fluence);
            } else {
                applyOne(lo->second, 1.0, fluence);
            }
        }
    }
};
} // namespace Chiffon
