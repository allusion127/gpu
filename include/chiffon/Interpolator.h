#pragma once

#include "Model.h"
#include <iostream>
#include <set>
#include "milk.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace Chiffon {

// Every (model, branch) pair the fit dropped for a degenerate abscissa, in the
// order they were found. A warning on the build console is only as durable as
// the terminal it scrolled past; the library it produced then looks like any
// other library that simply has no response on that branch. Exporter writes
// this list into the HDF5 Metadata so a library carries its own record of which
// branches are missing on purpose.
inline std::vector<std::string>& DegenerateBranchLog() {
    static std::vector<std::string> log;
    return log;
}

// The fit path is entered concurrently by --batch-mode workers.  Keep the
// legacy reference accessor for source compatibility, but publish a locked
// snapshot for readers and serialize the one writer below.  The log is
// process-global by design so an exported library retains one merged audit
// trail, rather than one record per worker thread.
inline std::mutex& DegenerateBranchLogMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::vector<std::string> SnapshotDegenerateBranchLog() {
    std::lock_guard<std::mutex> lock(DegenerateBranchLogMutex());
    return DegenerateBranchLog();
}

// Fits delta cross-sections as functions of branch variables (boron, fuel temp, etc.)
class Interpolator {

private:
    Interpolator() = delete;

    // Fit delta cross-sections as a piecewise-linear curve in the branch variable.
    // Sorts and de-duplicates x (keeping the last duplicate); a single unique point
    // yields a constant delta, otherwise a linear spline with two coefficients
    // (intercept, slope) per interval. Coefficients are CrossSection-valued and
    // evaluated by Horner at query time (DeltaCrossSection::DeltaInto).
    static DeltaCrossSection FitCurve(const std::vector<double>&       x_in,
                                      const std::vector<CrossSection>& y_in) {
        if (x_in.empty() || x_in.size() != y_in.size())
            throw std::invalid_argument(
                "FitCurve requires equally sized, nonempty samples.");
        const size_t ngrp = y_in.front().ngrp();
        for (size_t i = 0; i < x_in.size(); ++i) {
            if (!std::isfinite(x_in[i]))
                throw std::invalid_argument(
                    "FitCurve coordinates must be finite.");
            if (y_in[i].ngrp() != ngrp)
                throw std::invalid_argument(
                    "FitCurve cross sections must use one group structure.");
        }

        // Sort by x and drop duplicate x values (keep the last occurrence).
        const int        n0 = static_cast<int>(x_in.size());
        std::vector<int> sortIdx(n0);
        std::iota(sortIdx.begin(), sortIdx.end(), 0);
        std::sort(sortIdx.begin(), sortIdx.end(),
                  [&](int a, int b) { return x_in[a] < x_in[b]; });

        std::vector<double>       x;
        std::vector<CrossSection> y;
        x.reserve(n0);
        y.reserve(n0);
        for (int i = 0; i < n0; ++i) {
            if (!x.empty() && std::abs(x_in[sortIdx[i]] - x.back()) < 1e-15) {
                y.back() = y_in[sortIdx[i]];
            } else {
                x.push_back(x_in[sortIdx[i]]);
                y.push_back(y_in[sortIdx[i]]);
            }
        }

        const int N = static_cast<int>(x.size());

        // One unique point: constant delta.
        if (N < 2) {
            DeltaCrossSection delta(ngrp, 1);
            delta[0] = y[0];
            return delta;
        }

        // Linear spline: coefficients [intercept, slope] per interval.
        const int         nintervals = N - 1;
        const int         ncoeff     = 2;
        DeltaCrossSection delta(ngrp, static_cast<size_t>(nintervals * ncoeff), static_cast<size_t>(ncoeff), x);
        for (int i = 0; i < nintervals; ++i) {
            delta[2 * i]       = y[i];
            CrossSection slope = y[i + 1] - y[i];
            slope *= (1.0 / (x[i + 1] - x[i]));
            delta[2 * i + 1] = std::move(slope);
        }
        return delta;
    }

    // A branch fit needs its abscissa to actually vary. When every sample shares one x
    // (to within `tol`), FitCurve de-duplicates down to a single knot and returns a
    // CONSTANT delta equal to the last branch point -- i.e. the full branch perturbation
    // is then added unconditionally at every state, whatever the real condition is.
    //
    // This is not hypothetical. The iSMR_HIGA DeCART decks omit isotope 5000 from their
    // EDIT/isotope card, so both the reference and the 500 ppm branch point report
    // iden("50002") == 0; the boron branch collapsed to a constant +500 ppm delta and
    // biased the whole cycle by about -3500 pcm. Under CHIFFON 2.4.0 that was worked
    // around with "settings": {"bppm": {"apply": false}}; the settings block no longer
    // exists in 3.0.0, so the degeneracy has to be detected from the data instead.
    //
    // Detecting it is also the physically correct answer: a fit with no abscissa range
    // carries no information about the branch and must contribute nothing. Libraries whose
    // branch axis does vary (all the built-in i-SMR / CE16 / WH17 sets) are untouched.
    static bool DegenerateBranchAxis(const std::vector<double>& x, double tol = 1.0e-12) {
        if (x.size() < 2) return true;
        const double lo = *std::min_element(x.begin(), x.end());
        const double hi = *std::max_element(x.begin(), x.end());
        const double scale = std::max({std::abs(lo), std::abs(hi), 1.0e-30});
        return (hi - lo) <= tol * scale || (hi - lo) == 0.0;
    }

    // Emit the "branch dropped" notice once per (model, branch) pair.
    static void WarnDegenerateBranch(const std::string& modelName, const char* branchName) {
        static std::set<std::string> seen;
        const std::string key = modelName + "/" + branchName;
        {
            std::lock_guard<std::mutex> lock(DegenerateBranchLogMutex());
            if (!seen.insert(key).second) return;
            DegenerateBranchLog().push_back(key);
        }
        std::cerr << "[CHIFFON][warn] model '" << modelName << "': the " << branchName
                  << " branch abscissa is constant across all samples; the branch carries no "
                     "information and is dropped (no delta applied). Check that the DeCART "
                     "EDIT/isotope card exports the branch nuclide."
                  << std::endl;
    }

    static double RodDepletionAxis(const DepletionPoint& point) {
        if (std::isfinite(point._rod_fluence))
            return point._rod_fluence;
        return point._data[AD_BURN];
    }

    static void SubtractBranchDeltaTable(const Model& model, CrossSection& delta,
                                         const BranchDelta& stored, int cType, int burnup,
                                         double x, bool isRodDepletion) {
        if (!stored.contains(cType) || stored.at(cType).empty())
            return;

        const auto& bmap = stored.at(cType);
        auto        hi   = bmap.lower_bound(burnup);
        auto        lo   = hi;
        if (hi == bmap.end()) {
            lo = std::prev(bmap.end());
            hi = lo;
        } else if (hi == bmap.begin()) {
            lo = hi;
        } else if (hi->first != burnup) {
            lo = std::prev(hi);
        }

        CrossSection branchDelta = lo->second.Delta(x);
        if (lo != hi && hi->first != lo->first) {
            const double frac = static_cast<double>(burnup - lo->first) /
                                static_cast<double>(hi->first - lo->first);
            branchDelta *= (1.0 - frac);
            CrossSection hiDelta = hi->second.Delta(x);
            hiDelta *= frac;
            branchDelta += hiDelta;
        }

        // Express rod depletion relative to the rodded reference fluence.
        if (isRodDepletion) {
            const double refFlu = model.ReferenceRodFluence(cType, burnup);
            if (refFlu > 0.0) {
                CrossSection refDelta = lo->second.Delta(refFlu);
                if (lo != hi && hi->first != lo->first) {
                    const double frac = static_cast<double>(burnup - lo->first) /
                                        static_cast<double>(hi->first - lo->first);
                    refDelta *= (1.0 - frac);
                    CrossSection refHi = hi->second.Delta(refFlu);
                    refHi *= frac;
                    refDelta += refHi;
                }
                branchDelta -= refDelta;
            }
        }
        delta -= branchDelta;
    }

    enum class FittedEffect {
        Boron,
        FuelTemperature,
        ModeratorDensity,
        RodDepletion,
    };

    // Remove effects that were fitted before the current residual.
    static void SubtractStoredDeltas(Model& model, CrossSection& delta, int cType, int burnup,
                                     const DepletionPoint& point,
                                     std::initializer_list<FittedEffect> effects) {
        for (const FittedEffect effect : effects) {
            const BranchDelta* stored = nullptr;
            double             x      = 0.0;
            bool               isRodDepletion = false;
            if (effect == FittedEffect::Boron) {
                stored = &model._bppm_delt;
                x      = point.iden("50002");
            } else if (effect == FittedEffect::FuelTemperature) {
                stored = &model._tful_delt;
                x      = std::sqrt(point._data[AD_TFUL]);
            } else if (effect == FittedEffect::ModeratorDensity) {
                stored = &model._dmod_delt;
                x      = point._data[AD_DMOD];
            } else if (effect == FittedEffect::RodDepletion) {
                stored         = &model._rod_depletion_delt;
                x              = RodDepletionAxis(point);
                isRodDepletion = true;
            }
            SubtractBranchDeltaTable(model, delta, *stored, cType, burnup, x, isRodDepletion);
        }
    }

    static double EvaluateSpectralTerm(const SpectralTerm& term,
                                       const milk::Vector<double>& densities) {
        const double density =
            term.isotope < densities.size() ? densities[term.isotope] : 0.0;
        if (term.coordinate == SpectralCoordinate::LogDensity)
            return std::log(
                std::max(density, SPECTRAL_LOG_DENSITY_FLOOR));
        if (term.coordinate == SpectralCoordinate::SqrtDensity)
            return std::sqrt(std::max(0.0, density));
        return std::max(0.0, density);
    }

    static std::vector<double> EvaluateSpectralBasis(
        const SpectralBasis& terms, const milk::Vector<double>& densities) {
        std::vector<double> x;
        x.reserve(terms.size());
        for (const SpectralTerm& term : terms)
            x.push_back(EvaluateSpectralTerm(term, densities));
        return x;
    }

private:
    static constexpr double SPECTRAL_SVD_RCOND = 1.0e-2;

    // PROBE (revert before merge): the production rcond was tuned for a pooled
    // fit with far fewer rows per unknown than the split surfaces have.
    static double SpectralRcond() {
        static const double r = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RCOND");
            if (env == nullptr)
                return SPECTRAL_SVD_RCOND;
            const double v = std::atof(env);
            return (v > 0.0 && v < 1.0) ? v : SPECTRAL_SVD_RCOND;
        }();
        return r;
    }

    // PROBE: the rodded surface is fitted on fewer rows, so its singular values
    // decay faster and the shared truncation reaches directions the unrodded
    // surface still resolves. Allow a separate value for it.
    static double SpectralRcondRodded() {
        static const double r = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RCOND_ROD");
            if (env == nullptr)
                return -1.0;
            const double v = std::atof(env);
            return (v > 0.0 && v < 1.0) ? v : -1.0;
        }();
        return r;
    }

    static double& ActiveRcond() {
        static double active = -1.0;
        return active;
    }

    // PROBE: which surface and burnup key the next fit belongs to, so the
    // design/response dump can be split per key offline. Follows the same
    // set-before-call idiom as ActiveRank/ActiveRcond above.
    static std::string& ActiveFitTag() {
        static std::string tag;
        return tag;
    }
    // PROBE: response channels written beside the design matrix. Two lumped
    // reactions per group plus four microscopic thermal absorptions, chosen to
    // span slow (Pu239, U235), fast (Xe135) and self-shielded (Gd) behaviour.
    static constexpr std::size_t SPECTRAL_DUMP_CHANNELS = 8;

    // addScaled with a separate weight for the lumped and microscopic blocks, so
    // the two can be truncated at different ranks. Mirrors CrossSection::addScaled;
    // the macroscopic block is cleared on every fit row and stays untouched.
    static void addScaledSplit(CrossSection& dst, const CrossSection& src,
                               double lumpedWeight, double microWeight) {
        if (dst._nmem == 0)
            return;
        milk::addScaled(dst._nmem, lumpedWeight, src._lmpx.data(), 1,
                        dst._lmpx.data(), 1);
        if (dst._micx.size() > 0 && src._micx.size() > 0)
            milk::addScaled(dst._micx.size(), microWeight, src._micx.data(), 1,
                            dst._micx.data(), 1);
    }

    static std::vector<double> DumpChannels(const CrossSection& y) {
        const int ith = static_cast<int>(y.ngrp()) - 1;
        return {y.lmpxs(0, XSAF),
                y.lmpxs(ith, XSAF),
                y.lmpxs(0, XSNF),
                y.lmpxs(ith, XSNF),
                y.mixs(static_cast<int>(Isotope::iGd), ith, XSAF),
                y.mixs(static_cast<int>(Isotope::iU235), ith, XSAF),
                y.mixs(static_cast<int>(Isotope::iPu239), ith, XSAF),
                y.mixs(static_cast<int>(Isotope::iXe135), ith, XSAF)};
    }

    // An isotope thinner than this in the reference carries no correction.
    static constexpr double MACRO_FIT_DENSITY_FLOOR = 1.0e-14;
    static constexpr double SPECTRAL_WEIGHT_CUTOFF = 1.0e-14;

    struct VectorFitData {
        std::vector<std::vector<double>> xvals;
        std::vector<CrossSection>        yvals;
        std::vector<int>                 ctypes;
        std::vector<double>              weights;
        bool                             has_reference = false;
    };

    using VectorFitMap = std::map<int, VectorFitData>;

    // Share of the lattice-average flux carried by one end of the group
    // structure. Shares are dimensionless, so the lattice and solver flux
    // normalisations cancel; absolute fluxes do not and cannot be used here.
    static double FluxShare(const std::vector<double>& aflx, bool thermal) {
        double total = 0.0;
        for (const double value : aflx)
            total += value;
        if (!(total > 0.0) || aflx.empty())
            return 0.0;
        return (thermal ? aflx.back() : aflx.front()) / total;
    }

    static double FluxRatioIndex(const std::vector<double>& aflx) {
        if (aflx.size() < 2)
            return 0.0;
        const double fast    = std::max(aflx.front(), 1.0e-30);
        const double thermal = std::max(aflx.back(), 1.0e-30);
        return std::log(thermal / fast);
    }

    // The condition coordinates the branch layer itself rides on, in the runtime
    // scalar-branch order. Kept identical to XSSet's `_lib_ref_branch_x` so a
    // centered cross term tilts exactly the slope the branch layer fitted.
    static double BranchAxisValue(const DepletionPoint& point, int axis) {
        if (axis == 0)
            return point.iden("50002");
        if (axis == 1)
            return std::sqrt(point._data[AD_TFUL]);
        return point._data[AD_DMOD];
    }

    static std::vector<double> EvaluateSpectralBasis(
        const DepletionPoint& point, const SpectralBasis& terms,
        const DepletionPoint* reference) {
        std::vector<double> x = EvaluateSpectralBasis(point, terms);
        if (reference == nullptr)
            return x;
        const double h = FluxRatioIndex(point._aflx);
        // Condition x rod age: the rodded branch tables are fitted on a fresh rod
        // and used on a burned one, so let the condition slope carry a term that
        // grows with rod exposure. Vanishes at the reference condition, and with
        // the rod out there is no exposure to report.
        for (std::size_t k = 0; k < terms.size(); ++k) {
            const int axis = RodAgeAxisOf(terms[k].coordinate);
            if (axis < 0)
                continue;
            // Gated on the rod being present, because the runtime reads exposure
            // from the currently-rodded cells and reports zero once the rod is
            // out. A withdrawn row still carries its deck's fluence, so without
            // this gate the fit and the runtime disagree on the same state.
            const double age = (point._ctyp > 0 &&
                                std::isfinite(point._rod_fluence))
                                   ? point._rod_fluence * ROD_AGE_SCALE
                                   : 0.0;
            x[k] = (BranchAxisValue(point, axis) -
                    BranchAxisValue(*reference, axis)) *
                   age;
        }
        for (std::size_t k = 0; k < terms.size(); ++k) {
            const int  axis = BranchAxisOf(terms[k].coordinate);
            const bool flux =
                terms[k].coordinate == SpectralCoordinate::FluxRatioInteraction;
            if (!flux && axis < 0)
                continue;
            const std::size_t iso = terms[k].isotope;
            const double now = iso < point._iden.size() ? point._iden[iso] : 0.0;
            const double ref =
                iso < reference->_iden.size() ? reference->_iden[iso] : 0.0;
            const double weight = flux ? h
                                       : BranchAxisValue(point, axis) -
                                             BranchAxisValue(*reference, axis);
            x[k] = weight * (now - ref);
        }
        // Forms measured against the burnup reference inventory. Each is a
        // monotone reshaping of one density, so its new content in a linear fit
        // is curvature, not a new latent variable.
        for (std::size_t k = 0; k < terms.size(); ++k) {
            const SpectralCoordinate c = terms[k].coordinate;
            if (c != SpectralCoordinate::LogDeviationSquared &&
                c != SpectralCoordinate::InverseRatio &&
                c != SpectralCoordinate::CubeRootRatio &&
                c != SpectralCoordinate::SaturatingRatio)
                continue;
            const std::size_t iso = terms[k].isotope;
            const double fl  = SPECTRAL_LOG_DENSITY_FLOOR;
            const double now =
                iso < point._iden.size() ? std::max(point._iden[iso], fl) : fl;
            const double ref = iso < reference->_iden.size()
                                   ? std::max(reference->_iden[iso], fl)
                                   : fl;
            x[k] = RatioFormOf(c, now, ref);
        }
        return x;
    }

    template <typename V>
    static double FissileFractionOf(const V& iden, std::size_t a,
                                    std::size_t b) {
        auto at = [&](std::size_t i) {
            return i < iden.size() ? std::max(0.0, iden[i]) : 0.0;
        };
        const double na = at(a);
        const double sum = na + at(b);
        return sum > 1.0e-300 ? na / sum : 0.0;
    }

    static std::vector<double> EvaluateSpectralBasis(
        const DepletionPoint& point, const SpectralBasis& terms) {
        std::vector<double> x = EvaluateSpectralBasis(terms, point._iden);
        // Evaluated here, not at the call site, so the reference row that defines
        // the centering gets the same expression as every other row.
        for (std::size_t k = 0; k < terms.size(); ++k)
            if (terms[k].coordinate == SpectralCoordinate::FissileFraction)
                x[k] = FissileFractionOf(point._iden, terms[k].isotope,
                                         terms[k].partner);
        for (std::size_t k = 0; k < terms.size(); ++k) {
            const bool thermal =
                terms[k].coordinate == SpectralCoordinate::ThermalWeighted;
            const bool fast =
                terms[k].coordinate == SpectralCoordinate::FastWeighted;
            if (!thermal && !fast)
                continue;
            const double density =
                terms[k].isotope < point._iden.size()
                    ? std::max(0.0, point._iden[terms[k].isotope])
                    : 0.0;
            x[k] = FluxShare(point._aflx, thermal) * density;
        }
        return x;
    }

    // PROBE (revert before merge): "xe135,pu239" or, with a flux weight,
    // "th_xe135,fa_xe135". Unset keeps the production spec.
    static SpectralBasis ParseBasisSpec(const std::string& spec) {
        static const std::map<std::string, const char*> isotopes = {
            {"i135", "531350"},  {"nd149", "601490"}, {"pm149", "611490"},
            {"xe135", "541350"}, {"sm149", "621490"}, {"gd", "640000"},
            {"u235", "922350"},  {"u236", "922360"},  {"u238", "922380"},
            {"np237", "932370"}, {"nd148", "601480"}, {"pu238", "942380"}, {"pu239", "942390"},
            {"pu240", "942400"}, {"pu241", "942410"}, {"pu242", "942420"},
            {"am241", "952410"},
        };

        SpectralBasis     basis;
        std::stringstream stream(spec);
        std::string       token;
        while (std::getline(stream, token, ',')) {
            if (token.empty())
                continue;
            SpectralCoordinate coordinate = SpectralCoordinate::Density;
            if (token.rfind("th_", 0) == 0) {
                coordinate = SpectralCoordinate::ThermalWeighted;
                token.erase(0, 3);
            } else if (token.rfind("fa_", 0) == 0) {
                coordinate = SpectralCoordinate::FastWeighted;
                token.erase(0, 3);
            } else if (token.rfind("hp_", 0) == 0) {
                coordinate = SpectralCoordinate::FluxRatioInteraction;
                token.erase(0, 3);
            } else if (token == "spx") {
                basis.push_back(SpectralTerm{
                    Isotope::iPu239, SpectralCoordinate::SpectralIndex});
                continue;
            } else if (token.rfind("rr_", 0) == 0) {
                const std::string body = token.substr(3);
                const auto        cut  = body.find('_');
                if (cut == std::string::npos)
                    throw std::runtime_error(
                        "CHIFFON_PROBE_BASIS: 'rr_' needs two isotopes, got '" +
                        token + "'.");
                const auto a = isotopes.find(body.substr(0, cut));
                const auto b = isotopes.find(body.substr(cut + 1));
                if (a == isotopes.end() || b == isotopes.end() ||
                    !Isotope::iidx.contains(a->second) ||
                    !Isotope::iidx.contains(b->second))
                    throw std::runtime_error(
                        "CHIFFON_PROBE_BASIS: unusable pair in '" + token + "'.");
                basis.push_back(
                    SpectralTerm{Isotope::iidx.at(a->second),
                                 SpectralCoordinate::RelativeBurnRatio,
                                 Isotope::iidx.at(b->second)});
                continue;
            } else if (token.rfind("ff_", 0) == 0) {
                const std::string body = token.substr(3);
                const auto        cut  = body.find('_');
                if (cut == std::string::npos)
                    throw std::runtime_error(
                        "CHIFFON_PROBE_BASIS: 'ff_' needs two isotopes, got '" +
                        token + "'.");
                const auto a = isotopes.find(body.substr(0, cut));
                const auto b = isotopes.find(body.substr(cut + 1));
                if (a == isotopes.end() || b == isotopes.end() ||
                    !Isotope::iidx.contains(a->second) ||
                    !Isotope::iidx.contains(b->second))
                    throw std::runtime_error(
                        "CHIFFON_PROBE_BASIS: unusable pair in '" + token + "'.");
                basis.push_back(
                    SpectralTerm{Isotope::iidx.at(a->second),
                                 SpectralCoordinate::FissileFraction,
                                 Isotope::iidx.at(b->second)});
                continue;
            } else if (token.rfind("sq_", 0) == 0) {
                coordinate = SpectralCoordinate::LogDeviationSquared;
                token.erase(0, 3);
            } else if (token.rfind("iv_", 0) == 0) {
                coordinate = SpectralCoordinate::InverseRatio;
                token.erase(0, 3);
            } else if (token.rfind("cb_", 0) == 0) {
                coordinate = SpectralCoordinate::CubeRootRatio;
                token.erase(0, 3);
            } else if (token.rfind("mm_", 0) == 0) {
                coordinate = SpectralCoordinate::SaturatingRatio;
                token.erase(0, 3);
            } else if (token.rfind("sx_", 0) == 0) {
                coordinate = SpectralCoordinate::SpectralIndexInteraction;
                token.erase(0, 3);
            } else if (token.rfind("xb_", 0) == 0) {
                coordinate = SpectralCoordinate::BppmInteraction;
                token.erase(0, 3);
            } else if (token.rfind("xt_", 0) == 0) {
                coordinate = SpectralCoordinate::TfulInteraction;
                token.erase(0, 3);
            } else if (token.rfind("xd_", 0) == 0) {
                coordinate = SpectralCoordinate::DmodInteraction;
                token.erase(0, 3);
            } else if (token.rfind("fb_", 0) == 0) {
                coordinate = SpectralCoordinate::BppmRodAge;
                token.erase(0, 3);
            } else if (token.rfind("ft_", 0) == 0) {
                coordinate = SpectralCoordinate::TfulRodAge;
                token.erase(0, 3);
            } else if (token.rfind("fd_", 0) == 0) {
                coordinate = SpectralCoordinate::DmodRodAge;
                token.erase(0, 3);
            } else if (token.rfind("sqrt_", 0) == 0) {
                coordinate = SpectralCoordinate::SqrtDensity;
                token.erase(0, 5);
            } else if (token.rfind("lin_", 0) == 0) {
                token.erase(0, 4);
            } else if (token.rfind("log_", 0) == 0) {
                coordinate = SpectralCoordinate::LogDensity;
                token.erase(0, 4);
            } else if (token == "pu239") {
                coordinate = SpectralCoordinate::LogDensity;
            }
            auto it = isotopes.find(token);
            if (it == isotopes.end() || !Isotope::iidx.contains(it->second))
                throw std::runtime_error(
                    "CHIFFON_PROBE_BASIS: unusable token '" + token + "'.");
            basis.push_back(
                SpectralTerm{Isotope::iidx.at(it->second), coordinate});
        }
        if (basis.empty())
            throw std::runtime_error("CHIFFON_PROBE_BASIS: empty basis.");
        return basis;
    }

    // PROBE (revert before merge): fit Gd as its macroscopic contribution
    // N_eff * sigma_eff instead of the microscopic sigma. The lumped effective
    // sigma is path dependent at fixed N_eff; the product is the quantity the
    // balance actually sees. Stored in the lumped slot so the runtime applies
    // it directly to the macro without multiplying by the node inventory.
    // 0 = microscopic residual, 1 = Gd only as N*sigma, 2 = every isotope,
    // i.e. fit the change in reaction rate per unit flux that the balance sees.
    static int MacroFitMode() {
        static const int mode = [] {
            const char* env = std::getenv("CHIFFON_PROBE_GDMACRO");
            if (env == nullptr)
                return 0;
            const std::string value(env);
            if (value == "all")
                return 2;
            if (value == "rr")
                return 3;
            if (value == "psi")
                return 4;
            if (value == "rrpsi")
                return 5;
            return std::atoi(env) != 0 ? 1 : 0;
        }();
        return mode;
    }

    // Weight the residual by the row's own inventory so the least squares is
    // driven by reaction rate rather than by microscopic cross section. The
    // channel stays in place; only the target is rescaled.
    // Relative reaction-rate weighting: the row's own inventory divided by the
    // burnup reference. Every channel-constant factor cancels once the fit is
    // converted back, so the ratio is the only weighting that can change the
    // answer, and unlike the raw inventory it does not let U238 and O16 own the
    // least squares.
    static void WeightByRelativeInventory(CrossSection& residual,
                                          const DepletionPoint& point,
                                          const DepletionPoint& reference) {
        const std::size_t nmem = residual._lmpx.size();
        if (nmem == 0)
            return;
        const std::size_t niso = residual._micx.size() / nmem;
        for (std::size_t iso = 0; iso < niso; ++iso) {
            const double now =
                iso < point._iden.size() ? std::max(0.0, point._iden[iso]) : 0.0;
            const double ref = iso < reference._iden.size()
                                   ? std::max(0.0, reference._iden[iso])
                                   : 0.0;
            if (!(ref > MACRO_FIT_DENSITY_FLOOR))
                continue;
            const double weight = now / ref;
            const std::size_t base = iso * nmem;
            for (std::size_t m = 0; m < nmem; ++m)
                residual._micx[base + m] *= weight;
        }
    }

    // Reaction-rate objective: weight each channel's row by the group's flux
    // share relative to the reference row, and optionally by the isotope's
    // relative inventory. Both factors vary by row, which is the only kind of
    // weighting a shared design matrix can feel.
    static void WeightByFluxShare(CrossSection& residual,
                                  const DepletionPoint& point,
                                  const DepletionPoint& reference,
                                  bool alsoInventory) {
        const std::size_t nmem = residual._lmpx.size();
        const std::size_t ndat = residual._ndat;
        if (nmem == 0 || ndat == 0)
            return;
        const std::size_t ngrp = nmem / ndat;

        std::vector<double> share(ngrp, 1.0);
        for (std::size_t g = 0; g < ngrp; ++g) {
            const double now = g < point._aflx.size() ? point._aflx[g] : 0.0;
            const double ref =
                g < reference._aflx.size() ? reference._aflx[g] : 0.0;
            double sumNow = 0.0, sumRef = 0.0;
            for (const double v : point._aflx) sumNow += v;
            for (const double v : reference._aflx) sumRef += v;
            if (sumNow > 0.0 && sumRef > 0.0 && ref > 0.0)
                share[g] = (now / sumNow) / (ref / sumRef);
        }

        for (std::size_t m = 0; m < nmem; ++m)
            residual._lmpx[m] *= share[m / ndat];

        const std::size_t niso = residual._micx.size() / nmem;
        for (std::size_t iso = 0; iso < niso; ++iso) {
            double inv = 1.0;
            if (alsoInventory) {
                const double now =
                    iso < point._iden.size() ? std::max(0.0, point._iden[iso]) : 0.0;
                const double ref = iso < reference._iden.size()
                                       ? std::max(0.0, reference._iden[iso])
                                       : 0.0;
                if (!(ref > MACRO_FIT_DENSITY_FLOOR))
                    continue;
                inv = now / ref;
            }
            const std::size_t base = iso * nmem;
            for (std::size_t m = 0; m < nmem; ++m)
                residual._micx[base + m] *= inv * share[m / ndat];
        }
    }

    static void WeightByInventory(CrossSection& residual,
                                  const DepletionPoint& point,
                                  bool allIsotopes) {
        const std::size_t nmem = residual._lmpx.size();
        if (nmem == 0)
            return;
        const std::size_t niso = residual._micx.size() / nmem;
        for (std::size_t iso = 0; iso < niso; ++iso) {
            if (!allIsotopes && iso != Isotope::iGd)
                continue;
            const double density =
                iso < point._iden.size()
                    ? std::max(0.0, point._iden[iso])
                    : 0.0;
            const std::size_t base = iso * nmem;
            for (std::size_t m = 0; m < nmem; ++m)
                residual._micx[base + m] *= density;
        }
    }

    // PROBE (revert before merge): keep only rows whose control state matches
    // their own trajectory, so an instantaneous rod branch stops acting as a
    // composition sample and is left to define the reference it is subtracted
    // from.
    static bool DropPromotedBranchRows() {
        static const bool drop = [] {
            const char* env = std::getenv("CHIFFON_PROBE_NOBRANCH");
            return env != nullptr && std::atoi(env) != 0;
        }();
        return drop;
    }

    // PROBE (revert before merge): row selection.
    //   "match"  - control state must equal the row's own trajectory
    //   "rodout" - keep only rod-out evaluations, whatever the trajectory, so a
    //              rodded composition seen with the rod withdrawn is paired
    //              against the unrodded base in one common frame
    //   "all"    - no filter
    static const std::string& RowMode() {
        static const std::string mode = [] {
            const char* env = std::getenv("CHIFFON_PROBE_ROWS");
            return std::string(env != nullptr ? env : "");
        }();
        return mode;
    }

    // PROBE: retained rank for the lumped channels alone. The control-rod
    // absorber is not a tracked nuclide, so the whole rod response lands in the
    // lumped remainder; measured per channel, the lumped reactions keep paying
    // out to r=6 while the tracked microscopic thermal absorptions stop at r=2.
    // -1 keeps one rank for every channel.
    static int SpectralRankLumped() {
        static const int rank = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RANK_LMP");
            return env == nullptr ? -1 : std::atoi(env);
        }();
        return rank;
    }

    // PROBE: keep the rod-depletion layer's fluence axis but drop its condition
    // axis. Rod exposure genuinely changes the absorber inventory and is
    // load-bearing; an instantaneous condition change does not, yet the condition
    // sub-layer fires on it and is measured to be the dominant error wherever it
    // is active. Its fuel-temperature and boron axes carry only two quantised
    // knots each, so there is little there to interpolate between.
    static bool UseRodDepletionCondition() {
        static const bool on = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RDPLCOND");
            return env == nullptr || std::atoi(env) != 0;
        }();
        return on;
    }

    // PROBE: fit the ctype>0 condition branches on the rodded decks' own branch
    // rows instead of the base deck's fresh-rod CR rows. Those rows are imported
    // but no layer reads them, because GetBranch indexes the main deck only —
    // and the tables they would replace are fitted on a fresh rod yet evaluated
    // on a burned one.
    static bool UseRoddedBranchTables() {
        static const bool on = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RODBRTAB");
            return env != nullptr && std::atoi(env) != 0;
        }();
        return on;
    }

    // PROBE: least-squares weight on the rodded depletion's own reference rows.
    // 1 is plain LSQ; large values approach an equality constraint on that
    // trajectory without needing a constrained solver.
    static double AnchorWeight() {
        static const double w = [] {
            const char* env = std::getenv("CHIFFON_PROBE_ANCHORW");
            if (env == nullptr)
                return 1.0;
            const double v = std::atof(env);
            return v > 0.0 ? v : 1.0;
        }();
        return w;
    }

    // PROBE: admit the rodded decks' condition-branch rows as SHCT fit rows.
    // They are imported today but reach neither layer: the branch surfaces index
    // only the main HGC, and addRow takes REFR rows alone.
    static bool AllowRodBranchRows() {
        static const bool on = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RODBRANCH");
            return env != nullptr && std::atoi(env) != 0;
        }();
        return on;
    }

    static bool KeepRow(const DepletionPoint& point) {
        const std::string& mode = RowMode();
        if (mode == "rodout")
            return point._ctyp == 0;
        // Keep both members of a rodded trajectory - the rod-out one is the
        // state a withdrawal lands in - and drop only the instantaneous
        // insertion taken on an unrodded trajectory, which adds no composition.
        if (mode == "pair")
            return !(point._trajectory_ctyp == 0 && point._ctyp > 0);
        if (mode == "all")
            return true;
        if (mode == "match" || DropPromotedBranchRows())
            return point._ctyp == point._trajectory_ctyp;
        return true;
    }

    static void AddReferenceSample(VectorFitData& data,
                                   const DepletionPoint& reference,
                                   const SpectralBasis& terms) {
        if (data.has_reference)
            return;
        data.xvals.push_back(
            EvaluateSpectralBasis(reference, terms, &reference));
        for (std::size_t k = 0; k < terms.size(); ++k) {
            if (terms[k].coordinate == SpectralCoordinate::SpectralIndex)
                data.xvals.back()[k] = 0.0;
            if (terms[k].coordinate ==
                    SpectralCoordinate::SpectralIndexInteraction ||
                terms[k].coordinate == SpectralCoordinate::RelativeBurnRatio)
                data.xvals.back()[k] = 0.0;
        }
        CrossSection zeroDelta = reference._xs - reference._xs;
        zeroDelta.ClearMacroscopic();
        data.yvals.push_back(std::move(zeroDelta));
        data.ctypes.push_back(0);
        data.weights.push_back(1.0);
        data.has_reference = true;
    }




    // PROBE: macro-equivalent thermal absorption of a residual, i.e. the number
    // the balance actually feels, used to trace what each peel step removes.
    static double MacroAbsorption(const CrossSection& xs,
                                  const DepletionPoint& point) {
        double v = xs.lmpxs(1, XSAF);
        for (std::size_t i = 0; i < Isotope::niso; ++i) {
            const double n = i < point._iden.size() ? point._iden[i] : 0.0;
            v += n * xs.mixs(static_cast<int>(i), 1, XSAF);
        }
        return v;
    }

    static std::vector<double>& PeelTrace() {
        // This is a per-BuildSpectralResidual scratch trace, not shared
        // metadata.  Thread-local storage preserves the CSV columns while
        // preventing concurrent workers from overwriting one another.
        static thread_local std::vector<double> trace;
        return trace;
    }

    // PROBE: sigma(Pu239,th)/sigma(B10,th) on the base+branch state. Pu239 is
    // resonant at 0.3 eV and B10 is 1/v, so the ratio moves with the shape of
    // the thermal group rather than with its magnitude. Dimensionless, so the
    // lattice and solver normalisations cancel.
    // log[(N_a/N_a,ref) / (N_b/N_b,ref)]: zero when both isotopes sit on the
    // reference trajectory, so the term cannot displace the branch layer.
    template <typename V>
    static double RelativeBurnRatioOf(
        const V& iden, const V& ref, std::size_t a, std::size_t b) {
        const double fl = Chiffon::SPECTRAL_LOG_DENSITY_FLOOR;
        auto         at = [&](const V& v, std::size_t i) {
            return i < v.size() ? std::max(v[i], fl) : fl;
        };
        return std::log(at(iden, a) / at(ref, a)) -
               std::log(at(iden, b) / at(ref, b));
    }

    static double SpectralIndexOf(const CrossSection& xs) {
        const int    ith = xs.ngrp() - 1;
        const double num = xs.mixs(static_cast<int>(Isotope::iPu239), ith, XSAF);
        const double den = xs.mixs(static_cast<int>(Isotope::iB10), ith, XSAF);
        return std::abs(den) > 1.0e-30 ? num / den : 0.0;
    }

    static double RowSpectralIndex(
        Model& model, const DepletionPoint& point, int currentCtype,
        int burnup, std::initializer_list<FittedEffect> fittedEffects) {
        CrossSection                refXs;
        milk::Vector<double>        refIden;
        std::array<double, AD_SIZE> refData;
        model.FillReferenceState(refXs, refIden, refData, nullptr,
                                 currentCtype, burnup);
        // SubtractStoredDeltas on a zero container yields -sum(branch deltas).
        CrossSection negBranch = refXs - refXs;
        SubtractStoredDeltas(model, negBranch, currentCtype, burnup, point,
                             fittedEffects);
        CrossSection baseBranch = refXs;
        baseBranch.addScaled(negBranch, -1.0);
        // Centre on the unbranched base so the coordinate is the *deviation* of
        // the spectrum, zero at nominal conditions. The raw ratio is dominated
        // by moderator temperature, which gives the cross term enormous
        // extrapolation leverage on cold branches the fit never saw.
        const double base = SpectralIndexOf(refXs);
        const double now  = SpectralIndexOf(baseBranch);
        return (base > 1.0e-30 && now > 1.0e-30) ? std::log(now / base) : 0.0;
    }

    static CrossSection BuildSpectralResidual(
        Model& model, const DepletionPoint& point, int currentCtype,
        int burnup, std::initializer_list<FittedEffect> fittedEffects) {
        CrossSection                mainRefXs;
        milk::Vector<double>        mainRefIden;
        std::array<double, AD_SIZE> mainRefData;
        std::vector<double>         mainRefFlux;
        model.FillReferenceState(
            mainRefXs, mainRefIden, mainRefData, &mainRefFlux,
            currentCtype, burnup);

        CrossSection delta = point._xs - mainRefXs;
        auto& trace = PeelTrace();
        trace.assign(4, 0.0);
        trace[0] = MacroAbsorption(delta, point);

        std::vector<FittedEffect> conditions;
        bool                      hasRodDepletion = false;
        for (const FittedEffect e : fittedEffects) {
            if (e == FittedEffect::RodDepletion)
                hasRodDepletion = true;
            else
                conditions.push_back(e);
        }
        for (const FittedEffect e : conditions)
            SubtractStoredDeltas(model, delta, currentCtype, burnup, point, {e});
        trace[1] = MacroAbsorption(delta, point);

        if (hasRodDepletion)
            SubtractStoredDeltas(model, delta, currentCtype, burnup, point,
                                 {FittedEffect::RodDepletion});
        trace[2] = MacroAbsorption(delta, point);
        trace[3] = std::isfinite(point._rod_fluence) ? point._rod_fluence : -1.0;

        delta.ClearMacroscopic();
        return delta;
    }

    static bool HasExactReferenceBurnup(const Model& model, int cType, int burnup) {
        auto cIt = model._refr_dpts.find(cType);
        if (cIt == model._refr_dpts.end())
            return false;
        return cIt->second.find(burnup) != cIt->second.end();
    }

    // PROBE (revert before merge): one CSV row per fit sample, rich enough to
    // refit any coordinate shape offline.
    static void DumpRow(const Model& model, const DepletionPoint& point,
                        int burnup, const CrossSection& residual) {
        static std::unique_ptr<std::ofstream> out = [] {
            const char* path = std::getenv("CHIFFON_PROBE_DUMP");
            if (path == nullptr)
                return std::unique_ptr<std::ofstream>();
            auto file = std::make_unique<std::ofstream>(path);
            *file << "ctype,traj,burnup,dens,refdens,y_gd,y_u5,y_pu9,"
                     "y_lmp_a,y_lmp_nf,y_macro_a,y_macro_nf,"
                     "peel_raw,peel_cond,peel_rdpl,rod_flu\n";
            return file;
        }();
        if (!out)
            return;

        const DepletionPoint& ref = model.GetDepletionPoint(0, burnup);
        *out << point._ctyp << ',' << point._trajectory_ctyp << ',' << burnup
             << ',';
        for (std::size_t i = 0; i < Isotope::niso; ++i)
            *out << (i ? " " : "")
                 << (i < point._iden.size() ? point._iden[i] : 0.0);
        *out << ',';
        for (std::size_t i = 0; i < Isotope::niso; ++i)
            *out << (i ? " " : "")
                 << (i < ref._iden.size() ? ref._iden[i] : 0.0);

        double macroA = residual.lmpxs(1, XSAF);
        double macroN = residual.lmpxs(1, XSNF);
        for (std::size_t i = 0; i < Isotope::niso; ++i) {
            const double n = i < point._iden.size() ? point._iden[i] : 0.0;
            macroA += n * residual.mixs(static_cast<int>(i), 1, XSAF);
            macroN += n * residual.mixs(static_cast<int>(i), 1, XSNF);
        }
        *out << ',' << residual.mixs(static_cast<int>(Isotope::iGd), 1, XSAF)
             << ',' << residual.mixs(static_cast<int>(Isotope::iU235), 1, XSAF)
             << ',' << residual.mixs(static_cast<int>(Isotope::iPu239), 1, XSAF)
             << ',' << residual.lmpxs(1, XSAF)
             << ',' << residual.lmpxs(1, XSNF)
             << ',' << macroA << ',' << macroN;
        for (const double v : PeelTrace())
            *out << ',' << v;
        *out << '\n';
    }

    // PROBE (revert before merge): dump every imported point with its state and
    // spectrum so the history sensitivity can be tracked across condition axes.
    static void DumpAllPoints(const Model& model) {
        const char* path = std::getenv("CHIFFON_PROBE_POINTS");
        if (path == nullptr)
            return;
        std::ofstream out(path);
        out << "src,ctyp,traj,btyp,burnup,bppm,tful,dmod,psi_th,"
               "macro_a_th,macro_nf_th,dens,mic_a_th,mic_nf_th\n";

        auto emit = [&](const char* src, const DepletionPoint& p) {
            if (!p._xs.has_micx())
                return;
            double tot = 0.0;
            for (const double v : p._aflx)
                tot += v;
            const double psi = tot > 0.0 ? p._aflx.back() / tot : 0.0;
            double a = p._xs.lmpxs(1, XSAF), nf = p._xs.lmpxs(1, XSNF);
            for (std::size_t i = 0; i < Isotope::niso; ++i) {
                const double n = i < p._iden.size() ? p._iden[i] : 0.0;
                a  += n * p._xs.mixs(static_cast<int>(i), 1, XSAF);
                nf += n * p._xs.mixs(static_cast<int>(i), 1, XSNF);
            }
            out << src << ',' << p._ctyp << ',' << p._trajectory_ctyp << ','
                << static_cast<int>(p._btyp) << ',' << p.burnKey() << ','
                << p.iden("50002") << ',' << p._data[AD_TFUL] << ','
                << p._data[AD_DMOD] << ',' << psi << ',' << a << ',' << nf
                << ',';
            for (std::size_t i = 0; i < Isotope::niso; ++i)
                out << (i ? " " : "")
                    << (i < p._iden.size() ? p._iden[i] : 0.0);
            out << ',';
            for (std::size_t i = 0; i < Isotope::niso; ++i)
                out << (i ? " " : "")
                    << p._xs.mixs(static_cast<int>(i), 1, XSAF);
            out << ',';
            for (std::size_t i = 0; i < Isotope::niso; ++i)
                out << (i ? " " : "")
                    << p._xs.mixs(static_cast<int>(i), 1, XSNF);
            out << '\n';
        };
        for (const auto& p : model._dpts) emit("base", p);
        for (const auto& p : model._spectral_history_dpts) emit("spct", p);
        for (const auto& p : model._rod_history_dpts) emit("rodh", p);
        for (const auto& p : model._rod_depletion_dpts) emit("rdep", p);
    }

    /// Candidate rows for one (ctype, burnup): several rodded decks land here.
    struct RodBranchRows {
        std::vector<const DepletionPoint*> references;
        std::vector<const DepletionPoint*> samples;
    };

    static std::map<int, std::map<int, RodBranchRows>> CollectRodBranchRows(
        const Model& model, BRANCHTYPE btyp) {
        std::map<int, std::map<int, RodBranchRows>> out;
        for (const auto& point : model.RodHistoryDpts()) {
            if (point._trajectory_ctyp <= 0 || !point._xs.has_micx())
                continue;
            if (point._btyp == REFR)
                out[point._ctyp][point.burnKey()].references.push_back(&point);
            else if (point._btyp == btyp)
                out[point._ctyp][point.burnKey()].samples.push_back(&point);
        }
        return out;
    }

    /// A condition branch is taken at fixed composition, so a row and its own
    /// deck's reference share an inventory exactly. Every rodded deck registered
    /// as an extra lands in the same (ctype, burnup) bucket, and pairing a row
    /// against the wrong deck's reference turns a condition delta into a
    /// composition difference — this is what keeps the decks apart.
    /// Compared over the depleted state only. The first three entries are H-1,
    /// B-10 and O-16, which a condition branch moves by construction: perturbing
    /// boron changes B-10, perturbing moderator density changes H-1 and O-16.
    /// Including them would reject every row against its own reference.
    static bool SameInventory(const DepletionPoint& a, const DepletionPoint& b) {
        const std::size_t n = std::min(a._iden.size(), b._iden.size());
        double            num = 0.0, den = 0.0;
        for (std::size_t i = Isotope::iO16 + 1; i < n; ++i) {
            num += std::abs(a._iden[i] - b._iden[i]);
            den += std::abs(b._iden[i]);
        }
        return den <= 0.0 ? true : num / den < 1.0e-9;
    }

    /// Of the rodded decks present at this burnup, the one depleted closest to the
    /// base deck's own conditions. The off-nominal rodded decks perturb around a
    /// different operating point, so their branch slopes are not the ones the
    /// nominal rodded table wants.
    static const DepletionPoint* NominalRodReference(
        const Model& model, const std::vector<const DepletionPoint*>& refs,
        int burnup) {
        const DepletionPoint& base = model.GetDepletionPoint(0, burnup);
        const DepletionPoint* best     = nullptr;
        double                bestDist = std::numeric_limits<double>::max();
        for (const DepletionPoint* r : refs) {
            double dist = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double b = BranchAxisValue(base, axis);
                dist += std::abs(BranchAxisValue(*r, axis) - b) /
                        std::max(std::abs(b), 1.0e-30);
            }
            if (dist < bestDist) {
                bestDist = dist;
                best     = r;
            }
        }
        return best;
    }

    // Refit the rodded-side condition branches on the rodded trajectory. A branch
    // delta is a difference at fixed composition, so referencing each sample to
    // its own deck's reference keeps it a pure condition response — measured on a
    // depleted rod, which is the state the table is used at.
    static void FitRoddedBranchTables(Model& model) {
        if (!UseRoddedBranchTables())
            return;
        std::size_t refit = 0, thin = 0, noref = 0, dropped = 0, degenerate = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const BRANCHTYPE btyp =
                axis == 0 ? BPPM : (axis == 1 ? TFUL : DMOD);
            const char* const axisName =
                axis == 0 ? "bppm (rodded refit)"
                          : (axis == 1 ? "tful (rodded refit)"
                                       : "dmod (rodded refit)");
            BranchDelta& table = axis == 0   ? model._bppm_delt
                                 : axis == 1 ? model._tful_delt
                                             : model._dmod_delt;
            for (const auto& [cType, byBurn] : CollectRodBranchRows(model, btyp)) {
                if (cType <= 0)
                    continue;
                for (const auto& [burnup, group] : byBurn) {
                    const DepletionPoint* chosen =
                        NominalRodReference(model, group.references, burnup);
                    if (chosen == nullptr || group.samples.empty()) {
                        ++noref;
                        continue;
                    }
                    const DepletionPoint&     ref = *chosen;
                    std::vector<double>       xvals{BranchAxisValue(ref, axis)};
                    std::vector<CrossSection> yvals;
                    CrossSection              zero = ref._xs - ref._xs;
                    zero.ClearMacroscopic();
                    yvals.push_back(std::move(zero));
                    for (const DepletionPoint* p : group.samples) {
                        if (!SameInventory(*p, ref)) {
                            ++dropped;
                            continue;
                        }
                        xvals.push_back(BranchAxisValue(*p, axis));
                        CrossSection delta = p->_xs - ref._xs;
                        // Same peel order as the base-deck tables: the density
                        // branch is fitted on what boron has not explained.
                        if (btyp == DMOD)
                            SubtractStoredDeltas(model, delta, cType, burnup, *p,
                                                 {FittedEffect::Boron});
                        delta.ClearMacroscopic();
                        yvals.push_back(std::move(delta));
                    }
                    // Never replace a fitted table with a degenerate one: if the
                    // inventory filter left nothing, the base-deck table stands.
                    if (xvals.size() < 2) {
                        ++thin;
                        continue;
                    }
                    // ...and "degenerate" means the same thing here as it does on
                    // the main path: a fit whose abscissa does not move carries no
                    // information about the branch, and FitCurve() turns it into a
                    // CONSTANT delta applied unconditionally at every state. The
                    // comment above claimed this case was covered; only the row
                    // *count* was. Two surviving rows at an identical boron
                    // density (the iSMR_HIGA missing-isotope-5000 signature) pass
                    // xvals.size() >= 2 and would silently overwrite a good
                    // base-deck table with a constant offset.
                    if (DegenerateBranchAxis(xvals)) {
                        WarnDegenerateBranch(model.name(), axisName);
                        ++degenerate;
                        continue;
                    }
                    table[cType][burnup] = FitCurve(xvals, yvals);
                    ++refit;
                }
            }
        }
        std::cerr << "[rodbrtab] refit " << refit << " tables, thin " << thin
                  << ", degenerate axis " << degenerate << ", no reference "
                  << noref << ", inventory-dropped rows " << dropped << "\n";
    }

    // Fit one pooled spectral-history surface at every burnup.
    static void FitSpectralHistoryCorrection(Model& model) {
        static const SpectralBasis terms = [] {
            const char* spec = std::getenv("CHIFFON_PROBE_BASIS");
            if (spec != nullptr)
                return ParseBasisSpec(spec);
            // Four columns, one per independent depletion clock: prompt
            // poison, delayed poison, fissile buildup, fuel depletion. The Gd
            // column was measured to be 99.95% explained by these four on the
            // fit rows; it adds no direction, only reshapes the geometry, and
            // it is a net loss on Gd-free cores.
            return SpectralBasis{
                {Isotope::iXe135, SpectralCoordinate::Density},
                {Isotope::iSm149, SpectralCoordinate::Density},
                {Isotope::iPu239, SpectralCoordinate::LogDensity},
                {Isotope::iU235, SpectralCoordinate::Density},
            };
        }();

        VectorFitMap samples;
        auto addRow = [&](const DepletionPoint& point) {
            const int burnup = point.burnKey();
            const bool branchRow =
                AllowRodBranchRows() && point._trajectory_ctyp > 0 &&
                (point._btyp == BPPM || point._btyp == TFUL ||
                 point._btyp == DMOD);
            if (!point._xs.has_micx() || point._nondepleted ||
                (point._btyp != REFR && !branchRow))
                return;
            if (!HasExactReferenceBurnup(model, point._ctyp, burnup) ||
                !HasExactReferenceBurnup(model, 0, burnup))
                return;
            if (!KeepRow(point))
                return;

            CrossSection residual =
                point._ctyp > 0
                    ? BuildSpectralResidual(
                          model, point, point._ctyp, burnup,
                          {FittedEffect::ModeratorDensity,
                           FittedEffect::Boron,
                           FittedEffect::FuelTemperature,
                           FittedEffect::RodDepletion})
                    : BuildSpectralResidual(
                          model, point, point._ctyp, burnup,
                          {FittedEffect::ModeratorDensity,
                           FittedEffect::Boron,
                           FittedEffect::FuelTemperature});

            if (MacroFitMode() == 4 || MacroFitMode() == 5)
                WeightByFluxShare(residual, point,
                                  model.GetDepletionPoint(0, burnup),
                                  MacroFitMode() == 5);
            else if (MacroFitMode() == 3)
                WeightByRelativeInventory(
                    residual, point, model.GetDepletionPoint(0, burnup));
            else if (MacroFitMode() > 0)
                WeightByInventory(residual, point, MacroFitMode() == 2);

            auto& data = samples[burnup];
            if (!data.has_reference)
                AddReferenceSample(
                    data, model.GetDepletionPoint(0, burnup), terms);
            data.xvals.push_back(EvaluateSpectralBasis(
                point, terms,
                &model.GetDepletionPoint(point._ctyp, burnup)));
            for (std::size_t k = 0; k < terms.size(); ++k) {
                const bool raw =
                    terms[k].coordinate == SpectralCoordinate::SpectralIndex;
                const bool cross =
                    terms[k].coordinate ==
                    SpectralCoordinate::SpectralIndexInteraction;
                if (terms[k].coordinate ==
                    SpectralCoordinate::RelativeBurnRatio) {
                    data.xvals.back()[k] = RelativeBurnRatioOf(
                        point._iden, model.GetDepletionPoint(0, burnup)._iden,
                        terms[k].isotope, terms[k].partner);
                    continue;
                }
                if (!raw && !cross)
                    continue;
                const double index =
                        point._ctyp > 0
                            ? RowSpectralIndex(
                                  model, point, point._ctyp, burnup,
                                  {FittedEffect::ModeratorDensity,
                                   FittedEffect::Boron,
                                   FittedEffect::FuelTemperature,
                                   FittedEffect::RodDepletion})
                            : RowSpectralIndex(
                                  model, point, point._ctyp, burnup,
                                  {FittedEffect::ModeratorDensity,
                                   FittedEffect::Boron,
                                   FittedEffect::FuelTemperature});
                if (raw) {
                    data.xvals.back()[k] = index;
                    continue;
                }
                const DepletionPoint& ref =
                    model.GetDepletionPoint(point._ctyp, burnup);
                const std::size_t     iso = terms[k].isotope;
                const double          now =
                    iso < point._iden.size() ? point._iden[iso] : 0.0;
                const double rf =
                    iso < ref._iden.size() ? ref._iden[iso] : 0.0;
                data.xvals.back()[k] = index * (now - rf);
            }
            DumpRow(model, point, burnup, residual);
            data.yvals.push_back(std::move(residual));
            data.ctypes.push_back(point._ctyp);
            // The unrodded reference is already exact: it is row 0 and the design
            // is centered on it, so its prediction is zero by construction. The
            // rodded depletion has no such guarantee -- it is one ordinary least
            // squares row. Weighting its reference points is the soft form of
            // "reproduce that trajectory too".
            const bool roddedAnchor =
                point._trajectory_ctyp > 0 && point._btyp == REFR;
            data.weights.push_back(roddedAnchor ? AnchorWeight() : 1.0);
        };

        for (const auto& point : model.SpectralHistoryDpts())
            addRow(point);
        for (const auto& point : model.RodHistoryDpts())
            addRow(point);
        for (const auto& point : model.Dpts())
            if (point._ctyp > 0)
                addRow(point);

        StoreSpectralHistoryFit(model, samples, terms);
    }

    // Undo the reaction-rate weighting on a fitted coefficient: divide each
    // isotope block by that isotope's reference inventory at this burnup, so the
    // stored quantity is a microscopic delta again and scales with the node's
    // own inventory when the runtime rebuilds the macro.
    static void MacroToMicro(CrossSection& coefficient,
                             const DepletionPoint& reference, bool allIsotopes) {
        const std::size_t nmem = coefficient._lmpx.size();
        if (nmem == 0)
            return;
        const std::size_t niso = coefficient._micx.size() / nmem;
        for (std::size_t iso = 0; iso < niso; ++iso) {
            if (!allIsotopes && iso != Isotope::iGd)
                continue;
            const double density =
                iso < reference._iden.size()
                    ? std::max(0.0, reference._iden[iso])
                    : 0.0;
            const std::size_t base = iso * nmem;
            const double      scale =
                density > MACRO_FIT_DENSITY_FLOOR ? 1.0 / density : 0.0;
            for (std::size_t m = 0; m < nmem; ++m)
                coefficient._micx[base + m] *= scale;
        }
    }

    // PROBE (revert before merge): channel-selective rod-state multiplier.
    // Fit one surface per rod ctype and apply a0*x + f_rod*(a1-a0)*x. This is
    // the discrete form of the composition x flux cross term of the group
    // constant expansion; without it the always-rodded response is unfittable.
    static bool UseFullSplit() {
        static const bool on = [] {
            const char* env = std::getenv("CHIFFON_PROBE_SPLIT");
            return env == nullptr || std::atoi(env) != 0;
        }();
        return on;
    }

    static bool UseChannelKappa() {
        static const bool on = [] {
            const char* env = std::getenv("CHIFFON_PROBE_KAPPA");
            return env != nullptr && std::atoi(env) != 0;
        }();
        return on;
    }

    static double KappaSpreadTolerance() {
        static const double tol = [] {
            const char* env = std::getenv("CHIFFON_PROBE_KAPPA_TOL");
            const double value = env != nullptr ? std::atof(env) : 0.25;
            return (value > 0.0 && value < 10.0) ? value : 0.25;
        }();
        return tol;
    }

    static VectorFitData RowsOfCtype(const VectorFitData& data, int ctype) {
        VectorFitData out;
        out.xvals.push_back(data.xvals.front());
        out.yvals.push_back(data.yvals.front());
        out.weights.push_back(data.weights.front());
        for (std::size_t i = 1; i < data.xvals.size(); ++i) {
            const bool rodded = data.ctypes[i] > 0;
            if ((ctype > 0) != rodded)
                continue;
            out.xvals.push_back(data.xvals[i]);
            out.yvals.push_back(data.yvals[i]);
            out.weights.push_back(data.weights[i]);
        }
        return out;
    }

    // Per channel, test whether the rodded coefficients are one multiple of the
    // unrodded ones across every basis direction. Channels that pass carry the
    // multiplier; channels that fail keep the pooled fit untouched.
    static void BlendChannelKappa(std::vector<CrossSection>& pooled,
                                  const std::vector<CrossSection>& unrodded,
                                  const std::vector<CrossSection>& rodded,
                                  std::vector<CrossSection>& increment) {
        const double tol   = KappaSpreadTolerance();
        const std::size_t nslope = pooled.size() - 1;
        increment = pooled;
        for (auto& coefficient : increment)
            for (auto& v : coefficient._lmpx) v = 0.0;
        for (auto& coefficient : increment)
            for (auto& v : coefficient._micx) v = 0.0;

        auto blend = [&](auto member) {
            auto&       target = pooled[0].*member;
            const std::size_t n = target.size();
            for (std::size_t c = 0; c < n; ++c) {
                double num = 0.0, den = 0.0;
                for (std::size_t k = 1; k <= nslope; ++k) {
                    const double a0 = (unrodded[k].*member)[c];
                    const double a1 = (rodded[k].*member)[c];
                    if (!(std::abs(a0) > 0.0))
                        continue;
                    num += std::abs(a0) * (a1 / a0);
                    den += std::abs(a0);
                }
                if (!(den > 0.0))
                    continue;
                const double kappa = num / den;
                if (!(kappa > 0.2 && kappa < 1.5))
                    continue;
                double spread = 0.0;
                for (std::size_t k = 1; k <= nslope; ++k) {
                    const double a0 = (unrodded[k].*member)[c];
                    const double a1 = (rodded[k].*member)[c];
                    if (!(std::abs(a0) > 0.0))
                        continue;
                    spread += std::abs(a0) * std::abs(a1 / a0 - kappa);
                }
                if (spread / (den * kappa) > tol)
                    continue;
                for (std::size_t k = 0; k <= nslope; ++k) {
                    (pooled[k].*member)[c]    = (unrodded[k].*member)[c];
                    (increment[k].*member)[c] =
                        (kappa - 1.0) * (unrodded[k].*member)[c];
                }
            }
        };
        blend(&CrossSection::_lmpx);
        blend(&CrossSection::_micx);
    }

    // PROBE (revert before merge): margin in units of the training span beyond
    // which a term stops growing. 0 = disabled (pure linear extrapolation).
    static double SaturationMargin() {
        static const double m = [] {
            const char* env = std::getenv("CHIFFON_PROBE_SAT");
            const double v = env != nullptr ? std::atof(env) : 0.0;
            return (v > 0.0 && v < 100.0) ? v : 0.0;
        }();
        return m;
    }

    // Hold the term flat outside the range the rows actually covered, so a node
    // far from the fit cloud stops being corrected harder and harder.
    static DeltaCrossSection SaturatingTerm(const CrossSection& constant,
                                            const CrossSection& slope,
                                            double xlo, double xhi,
                                            double margin) {
        const double span = xhi - xlo;
        const double lo   = xlo - (margin - 1.0) * span;
        const double hi   = xhi + (margin - 1.0) * span;
        const double pad  = std::max(span, 1.0e-30);

        auto value = [&](double x) {
            CrossSection v = constant;
            v.addScaled(slope, x);
            return v;
        };
        std::vector<double>       xs = {lo - pad, lo, hi, hi + pad};
        std::vector<CrossSection> ys = {value(lo), value(lo), value(hi),
                                        value(hi)};
        return FitCurve(xs, ys);
    }

    static void StoreSpectralHistoryFit(
        Model& model, VectorFitMap& samples, const SpectralBasis& terms) {
        const std::size_t nterm = terms.size();
        std::vector<SpectralHistoryCorrection> corrections(nterm);
        std::vector<SpectralHistoryCorrection> rodTerms(nterm);
        for (size_t termIndex = 0; termIndex < nterm; ++termIndex) {
            corrections[termIndex].term = terms[termIndex];
            rodTerms[termIndex].term       = terms[termIndex];
            rodTerms[termIndex].rod_scaled = true;
        }

        for (auto& [burnup, data] : samples) {
            if (data.xvals.size() < 2)
                continue;

            ActiveFitTag() = "pooled " + std::to_string(burnup);
            auto coefficients =
                FitVectorTermCoefficients(data.xvals, data.yvals, data.weights);

            std::vector<CrossSection> increment;
            // Full-rank rod dependence: keep both surfaces and let the node's
            // rod fraction interpolate. Algebraically the same model as a
            // gated increment, without the rank-1 constraint kappa imposes.
            if (UseFullSplit()) {
                const VectorFitData ct0 = RowsOfCtype(data, 0);
                const VectorFitData ct1 = RowsOfCtype(data, 1);
                if (ct0.xvals.size() > 2 && ct1.xvals.size() > 2) {
                    ActiveRcond()  = -1.0;
                    ActiveRank()   = -1;
                    ActiveFitTag() = "ct0 " + std::to_string(burnup);
                    auto a0 = FitVectorTermCoefficients(ct0.xvals, ct0.yvals, ct0.weights);
                    ActiveRcond()  = SpectralRcondRodded();
                    ActiveRank()   = SpectralRankRodded();
                    ActiveFitTag() = "ct1 " + std::to_string(burnup);
                    auto a1 = FitVectorTermCoefficients(ct1.xvals, ct1.yvals, ct1.weights);
                    ActiveRcond() = -1.0;
                    ActiveRank()  = -1;
                    increment = a1;
                    for (std::size_t k = 0; k < increment.size(); ++k)
                        increment[k].addScaled(a0[k], -1.0);
                    coefficients = std::move(a0);
                }
            } else if (UseChannelKappa()) {
                const VectorFitData ct0 = RowsOfCtype(data, 0);
                const VectorFitData ct1 = RowsOfCtype(data, 1);
                if (ct0.xvals.size() > 2 && ct1.xvals.size() > 2) {
                    auto a0 = FitVectorTermCoefficients(ct0.xvals, ct0.yvals, ct0.weights);
                    auto a1 = FitVectorTermCoefficients(ct1.xvals, ct1.yvals, ct1.weights);
                    BlendChannelKappa(coefficients, a0, a1, increment);
                }
            }
            if (MacroFitMode() == 1 || MacroFitMode() == 2) {
                const DepletionPoint& reference =
                    model.GetDepletionPoint(0, burnup);
                for (auto& coefficient : coefficients)
                    MacroToMicro(coefficient, reference,
                                 MacroFitMode() == 2);
            }
            for (size_t termIndex = 0;
                 termIndex < terms.size(); ++termIndex) {
                CrossSection constant =
                    termIndex == 0
                        ? std::move(coefficients[0])
                        : coefficients[termIndex + 1] -
                              coefficients[termIndex + 1];
                coefficients[termIndex + 1].ClearMacroscopic();
                constant.ClearMacroscopic();

                const double margin = SaturationMargin();
                double xlo = data.xvals.front()[termIndex];
                double xhi = xlo;
                for (const auto& row : data.xvals) {
                    xlo = std::min(xlo, row[termIndex]);
                    xhi = std::max(xhi, row[termIndex]);
                }
                if (margin > 0.0 && xhi > xlo) {
                    corrections[termIndex].delta[burnup] = SaturatingTerm(
                        constant, coefficients[termIndex + 1], xlo, xhi,
                        margin);
                } else {
                    DeltaCrossSection delta(
                        coefficients[termIndex + 1].ngrp(), 2);
                    delta[0] = std::move(constant);
                    delta[1] = std::move(coefficients[termIndex + 1]);
                    corrections[termIndex].delta[burnup] = std::move(delta);
                }

                if (increment.empty())
                    continue;
                CrossSection rodConstant =
                    termIndex == 0
                        ? increment[0]
                        : increment[0] - increment[0];
                CrossSection rodSlope = increment[termIndex + 1];
                rodConstant.ClearMacroscopic();
                rodSlope.ClearMacroscopic();
                DeltaCrossSection rodDelta(rodSlope.ngrp(), 2);
                rodDelta[0] = std::move(rodConstant);
                rodDelta[1] = std::move(rodSlope);
                rodTerms[termIndex].delta[burnup] = std::move(rodDelta);
            }
        }

        for (auto& correction : corrections)
            if (!correction.delta.empty())
                model._spectral_history.push_back(std::move(correction));
        for (auto& correction : rodTerms)
            if (!correction.delta.empty())
                model._spectral_history.push_back(std::move(correction));
    }

    static void FitRodDepletionDeltas(Model& model) {
        model._rod_depletion_delt.clear();
        for (auto& branch : model._rod_depletion_branch)
            branch.clear();
        if (model._rod_depletion_dpts.empty())
            return;

        std::map<int, std::vector<const DepletionPoint*>> pointsByCtype;
        for (const auto& dpt : model._rod_depletion_dpts) {
            if (dpt._btyp != REFR || !dpt._xs.has_micx() || dpt._ctyp <= 0)
                continue;
            pointsByCtype[dpt._ctyp].push_back(&dpt);
        }

        for (auto& [ctype, points] : pointsByCtype) {
            if (points.size() < 2)
                continue;

            std::sort(points.begin(), points.end(),
                      [](const DepletionPoint* a, const DepletionPoint* b) {
                          return RodDepletionAxis(*a) < RodDepletionAxis(*b);
                      });

            bool hasPairedReference = false;
            for (const auto* dpt : points) {
                if (dpt->_trajectory_reference) {
                    hasPairedReference = true;
                    break;
                }
            }

            // Paired data isolate rod-material depletion from fuel depletion.
            if (hasPairedReference) {
                // Pair HGCs tag the rodded base as ctype 0 and withdrawal as ctype > 0.
                // Select matching references by trajectory identity.
                std::map<int, const DepletionPoint*> nondepletedByBurn, depletedByBurn;
                for (const auto& raw : model._rod_depletion_dpts) {
                    if (raw._btyp != REFR || !raw._xs.has_micx() ||
                        raw._ctyp != 0 || raw._trajectory_ctyp != 0)
                        continue;
                    if (raw._trajectory_reference)
                        nondepletedByBurn.emplace(raw.burnKey(), &raw);
                    else
                        depletedByBurn.emplace(raw.burnKey(), &raw);
                }

                std::vector<double>       xvals;
                std::vector<CrossSection> yvals;
                xvals.reserve(depletedByBurn.size() + 1);
                yvals.reserve(depletedByBurn.size() + 1);
                if (!nondepletedByBurn.empty()) {
                    const DepletionPoint* ref  = nondepletedByBurn.begin()->second;
                    CrossSection          zero = ref->_xs - ref->_xs;
                    zero.ClearMacroscopic();
                    xvals.push_back(0.0);
                    yvals.push_back(std::move(zero));
                }
                for (const auto& [burnKey, dpt] : depletedByBurn) {
                    auto refIt = nondepletedByBurn.find(burnKey);
                    if (refIt == nondepletedByBurn.end())
                        continue;
                    CrossSection delta = dpt->_xs - refIt->second->_xs;
                    delta.ClearMacroscopic();
                    xvals.push_back(RodDepletionAxis(*dpt));
                    yvals.push_back(std::move(delta));
                }

                if (yvals.empty())
                    continue;
                model._rod_depletion_delt[ctype][0] =
                    FitCurve(xvals, yvals);

                // Fit condition-dependent rod-age residuals from paired branch blocks.
                // Each curve is zero-anchored at fresh-rod fluence.
                {
                    static constexpr double kUScale[3] = {1.0e12, 1.0e6, 1.0e9};
                    auto axisOf = [](BRANCHTYPE b) {
                        if (b == BRANCHTYPE::BPPM)
                            return 0;
                        if (b == BRANCHTYPE::TFUL)
                            return 1;
                        if (b == BRANCHTYPE::DMOD)
                            return 2;
                        return -1;
                    };
                    auto coordOf = [](const DepletionPoint& d, int a) {
                        if (a == 0)
                            return d.iden("50002");
                        if (a == 1)
                            return std::sqrt(d._data[AD_TFUL]);
                        return d._data[AD_DMOD];
                    };
                    std::map<int, CrossSection> nomDelta;
                    for (const auto& [burnKey, dpt] : depletedByBurn) {
                        auto refIt = nondepletedByBurn.find(burnKey);
                        if (refIt != nondepletedByBurn.end())
                            nomDelta.emplace(burnKey, dpt->_xs - refIt->second->_xs);
                    }
                    double maxFlu = 1.0;
                    for (const auto& [burnKey, dpt] : depletedByBurn)
                        maxFlu = std::max(maxFlu, RodDepletionAxis(*dpt));
                    std::array<std::map<int, std::map<int, const DepletionPoint*>>, 3> depBr, ndeBr;
                    for (const auto& raw : model._rod_depletion_dpts) {
                        const int a = axisOf(raw._btyp);
                        if (a < 0 || !raw._xs.has_micx() || raw._ctyp != 0 ||
                            raw._trajectory_ctyp != 0)
                            continue;
                        const int uKey = static_cast<int>(
                            std::llround(coordOf(raw, a) * kUScale[a]));
                        (raw._trajectory_reference ? ndeBr : depBr)[a][uKey][raw.burnKey()] = &raw;
                    }
                    for (int a = 0; a < 3; ++a) {
                        for (const auto& [uKey, byBurn] : depBr[a]) {
                            if (ndeBr[a].empty())
                                continue;
                            auto nIt = ndeBr[a].lower_bound(uKey);
                            if (nIt == ndeBr[a].end() ||
                                (nIt != ndeBr[a].begin() &&
                                 uKey - std::prev(nIt)->first < nIt->first - uKey))
                                nIt = std::prev(nIt);
                            std::vector<double>       xv;
                            std::vector<CrossSection> yv;
                            for (const auto& [burnKey, dDpt] : byBurn) {
                                auto n2 = nIt->second.find(burnKey);
                                auto nm = nomDelta.find(burnKey);
                                if (n2 == nIt->second.end() || nm == nomDelta.end())
                                    continue;
                                CrossSection r = dDpt->_xs - n2->second->_xs;
                                r.addScaled(nm->second, -1.0);
                                r.ClearMacroscopic();
                                if (yv.empty()) {
                                    CrossSection z = r - r;
                                    z.ClearMacroscopic();
                                    xv.push_back(0.0);
                                    yv.push_back(std::move(z));
                                }
                                xv.push_back(RodDepletionAxis(*dDpt));
                                yv.push_back(std::move(r));
                            }
                            if (yv.size() < 3)
                                continue;
                            model._rod_depletion_branch[a][ctype][uKey] = FitCurve(xv, yv);
                        }
                        auto stored = model._rod_depletion_branch[a].find(ctype);
                        if (stored != model._rod_depletion_branch[a].end() &&
                            !stored->second.empty() && !nondepletedByBurn.empty()) {
                            const DepletionPoint& refRow = *nondepletedByBurn.begin()->second;
                            const int             uRefKey = static_cast<int>(
                                std::llround(coordOf(refRow, a) * kUScale[a]));
                            CrossSection z = refRow._xs - refRow._xs;
                            z.ClearMacroscopic();
                            stored->second[uRefKey] =
                                FitCurve({0.0, maxFlu}, {z, z});
                        }
                    }
                }
                continue;
            }

            // Unpaired data use the lowest-exposure point as the reference.
            const DepletionPoint* ref = points.front();
            for (const auto* dpt : points) {
                if (std::abs(dpt->_data[AD_BURN]) < std::abs(ref->_data[AD_BURN]))
                    ref = dpt;
            }

            std::vector<double>       xvals;
            std::vector<CrossSection> yvals;
            xvals.reserve(points.size());
            yvals.reserve(points.size());
            for (const auto* dpt : points) {
                CrossSection delta = dpt->_xs - ref->_xs;
                delta.ClearMacroscopic();
                xvals.push_back(RodDepletionAxis(*dpt));
                yvals.push_back(std::move(delta));
            }

            if (yvals.empty())
                continue;
            model._rod_depletion_delt[ctype][0] =
                FitCurve(xvals, yvals);
        }
    }

    // PROBE: keep a fixed number of singular directions instead of a relative
    // threshold, so the retained rank does not vary from one burnup key to the
    // next. -1 keeps the rcond behaviour.
    // Retained rank of the spectral-history solve. Held FIXED rather than set
    // by a relative threshold, so the model complexity does not vary from one
    // burnup key to the next. Three is matched to the four-column basis: the
    // measured singular spectrum is 1 / 0.48 / 0.022 / 0.011, and the fourth
    // direction is not populated by the training trajectories.
    static int SpectralRank() {
        static const int rank = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RANK");
            return env == nullptr ? 3 : std::atoi(env);
        }();
        return rank;
    }

    // PROBE: the unrodded and rodded surfaces are fitted on different row sets
    // and want opposite regularization -- the unrodded one resolves a fourth
    // direction the rodded one does not. Allow a separate retained rank.
    static int SpectralRankRodded() {
        static const int rank = [] {
            const char* env = std::getenv("CHIFFON_PROBE_RANK_ROD");
            return env == nullptr ? -1 : std::atoi(env);
        }();
        return rank;
    }

    static int& ActiveRank() {
        static int active = -1;
        return active;
    }

    // PROBE: dump the singular spectrum of every spectral-history design matrix
    // so the retained rank can be counted per burnup key.
    static std::ofstream& SvDump() {
        static std::ofstream out = [] {
            const char* path = std::getenv("CHIFFON_PROBE_SVDUMP");
            std::ofstream f;
            if (path != nullptr) {
                f.open(path);
                f << "# design matrix rows: nrow ncol then the rows\n";
            }
            return f;
        }();
        return out;
    }

    // Fit a reference-centered, column-scaled linear map with truncated SVD.
    // `rowWeights` is the least-squares weight per row; empty means all ones.
    // Weighted LSQ is solved as pinv(W^1/2 Z) * W^1/2 y, so the row scaling has
    // to appear on both the design and the response.
    static std::vector<CrossSection> FitVectorTermCoefficients(
        const std::vector<std::vector<double>>& xvals,
        const std::vector<CrossSection>&        yvals,
        const std::vector<double>&              rowWeights = {}) {
        if (xvals.empty() || xvals.size() != yvals.size() ||
            xvals.front().empty())
            throw std::invalid_argument(
                "Spectral fit requires equally sized, nonempty samples.");
        const std::size_t          ndata    = yvals.size();
        const std::size_t          nfeature = xvals[0].size();
        const std::vector<double>& xref     = xvals[0];
        const size_t               ngrp     = yvals.front().ngrp();
        for (size_t i = 0; i < ndata; ++i) {
            if (xvals[i].size() != nfeature)
                throw std::invalid_argument(
                    "Spectral fit rows must have equal width.");
            if (yvals[i].ngrp() != ngrp)
                throw std::invalid_argument(
                    "Spectral fit cross sections must use one group structure.");
            for (const double value : xvals[i])
                if (!std::isfinite(value))
                    throw std::invalid_argument(
                        "Spectral fit coordinates must be finite.");
        }

        milk::Matrix<double> z(ndata, nfeature); // reference-centered, column-normalized design
        std::vector<double>  scale(nfeature, 1.0);
        for (std::size_t j = 0; j < nfeature; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < ndata; ++i)
                s = std::max(s, std::abs(xvals[i][j] - xref[j]));
            if (s > 1.0e-300)
                scale[j] = s;
        }
        std::vector<double> rootWeight(ndata, 1.0);
        if (!rowWeights.empty()) {
            if (rowWeights.size() != ndata)
                throw std::invalid_argument(
                    "Spectral fit weights must match the row count.");
            for (std::size_t i = 0; i < ndata; ++i) {
                if (!(rowWeights[i] > 0.0) || !std::isfinite(rowWeights[i]))
                    throw std::invalid_argument(
                        "Spectral fit weights must be finite and positive.");
                rootWeight[i] = std::sqrt(rowWeights[i]);
            }
        }
        for (std::size_t i = 0; i < ndata; ++i)
            for (std::size_t j = 0; j < nfeature; ++j)
                z(i, j) = rootWeight[i] * (xvals[i][j] - xref[j]) / scale[j];
        // Discard singular directions below 1% of the leading direction.
        if (SvDump().is_open()) {
            SvDump() << "K " << (ActiveFitTag().empty() ? "-" : ActiveFitTag())
                     << "\n";
            SvDump() << "M " << ndata << " " << nfeature << "\n";
            for (std::size_t i = 0; i < ndata; ++i) {
                for (std::size_t j = 0; j < nfeature; ++j)
                    SvDump() << z(i, j) << (j + 1 < nfeature ? " " : "");
                SvDump() << "\n";
            }
            // Representative response channels, so an offline pass can ask
            // whether one retained rank suits every cross section or only some.
            SvDump() << "Y " << ndata << " " << SPECTRAL_DUMP_CHANNELS << "\n";
            for (std::size_t i = 0; i < ndata; ++i) {
                const std::vector<double> row = DumpChannels(yvals[i]);
                for (std::size_t c = 0; c < row.size(); ++c)
                    SvDump() << row[c] << (c + 1 < row.size() ? " " : "");
                SvDump() << "\n";
            }
            SvDump().flush();
        }
        const double rcond =
            ActiveRcond() > 0.0 ? ActiveRcond() : SpectralRcond();
        const int microRank =
            ActiveRank() >= 0 ? ActiveRank() : SpectralRank();
        const int lumpedRank =
            SpectralRankLumped() >= 0 ? SpectralRankLumped() : microRank;
        const milk::Matrix<double> zpinv = z.pseudoInverse(rcond, microRank);
        // Only pay for a second decomposition when the two ranks differ.
        const milk::Matrix<double> zpinvLumped =
            lumpedRank == microRank ? milk::Matrix<double>()
                                    : z.pseudoInverse(rcond, lumpedRank);
        const bool splitRank = lumpedRank != microRank;

        // Transform normalized slopes back to physical coordinates.
        std::vector<CrossSection> coeffs;
        coeffs.reserve(nfeature + 1);
        CrossSection              constant = yvals[0] - yvals[0];
        std::vector<CrossSection> slopes;
        slopes.reserve(nfeature);
        for (std::size_t j = 0; j < nfeature; ++j) {
            CrossSection slope(yvals[0].ngrp());
            for (std::size_t i = 0; i < ndata; ++i) {
                const double w = zpinv(j, i) * rootWeight[i];
                if (!splitRank) {
                    if (std::abs(w) > SPECTRAL_WEIGHT_CUTOFF)
                        slope.addScaled(yvals[i], w / scale[j]);
                    continue;
                }
                const double wl = zpinvLumped(j, i) * rootWeight[i];
                if (std::abs(w) > SPECTRAL_WEIGHT_CUTOFF ||
                    std::abs(wl) > SPECTRAL_WEIGHT_CUTOFF)
                    addScaledSplit(slope, yvals[i], wl / scale[j],
                                   w / scale[j]);
            }
            constant.addScaled(slope, -xref[j]);
            slopes.push_back(std::move(slope));
        }
        coeffs.push_back(std::move(constant));
        for (auto& slope : slopes)
            coeffs.push_back(std::move(slope));
        return coeffs;
    }

public:
    // Fit delta cross-sections for all branch variations in a model.
    // Only micx and lmpx are interpolated; macroscopic XS is reconstructed at query time.
    static void Interpolate(Model& model) {
        model._bppm_delt.clear();
        model._tful_delt.clear();
        model._dmod_delt.clear();

        // BPPM branch: fit XS residuals against boron number density.
        for (auto& [cType, maps] : model.GetBranch(BPPM)) {
            for (auto& [burnup, points] : maps) {
                if (points.empty()) continue;
                const DepletionPoint&     refDpt = model.GetDepletionPoint(cType, burnup);
                std::vector<double>       xvals;
                std::vector<CrossSection> yvals;

                xvals.push_back(refDpt.iden("50002"));
                CrossSection zeroDelta = refDpt._xs - refDpt._xs;
                zeroDelta.ClearMacroscopic();
                yvals.push_back(std::move(zeroDelta));

                for (const auto& idx : points) {
                    DepletionPoint& brDpt = model.GetDepletionPoint(idx);
                    xvals.push_back(brDpt.iden("50002"));
                    CrossSection delta = brDpt._xs - refDpt._xs;
                    delta.ClearMacroscopic();
                    yvals.push_back(std::move(delta));
                }
                if (DegenerateBranchAxis(xvals)) {
                    WarnDegenerateBranch(model.name(), "bppm");
                    continue;
                }
                model._bppm_delt[cType][burnup] =
                    FitCurve(xvals, yvals);
            }
        }

        // TFUL branch: fit XS residuals against sqrt(fuel temperature).
        for (auto& [cType, maps] : model.GetBranch(TFUL)) {
            for (auto& [burnup, points] : maps) {
                if (points.empty()) continue;
                const DepletionPoint&     refDpt = model.GetDepletionPoint(cType, burnup);
                std::vector<double>       xvals;
                std::vector<CrossSection> yvals;

                xvals.push_back(sqrt(refDpt._data[AD_TFUL]));
                CrossSection zeroDelta = refDpt._xs - refDpt._xs;
                zeroDelta.ClearMacroscopic();
                yvals.push_back(std::move(zeroDelta));

                for (const auto& idx : points) {
                    DepletionPoint& brDpt = model.GetDepletionPoint(idx);
                    xvals.push_back(sqrt(brDpt._data[AD_TFUL]));
                    CrossSection delta = brDpt._xs - refDpt._xs;
                    delta.ClearMacroscopic();
                    yvals.push_back(std::move(delta));
                }
                if (DegenerateBranchAxis(xvals)) {
                    WarnDegenerateBranch(model.name(), "tful");
                    continue;
                }
                model._tful_delt[cType][burnup] =
                    FitCurve(xvals, yvals);
            }
        }

        // DMOD branch: subtract configured prior branch effects first, then
        // fit the remaining residual against moderator density.
        for (auto& [cType, maps] : model.GetBranch(DMOD)) {
            for (auto& [burnup, points] : maps) {
                if (points.empty()) continue;
                const DepletionPoint&     refDpt = model.GetDepletionPoint(cType, burnup);
                std::vector<double>       xvals;
                std::vector<CrossSection> yvals;

                xvals.push_back(refDpt._data[AD_DMOD]);
                CrossSection zeroDelta = refDpt._xs - refDpt._xs;
                zeroDelta.ClearMacroscopic();
                yvals.push_back(std::move(zeroDelta));

                for (const auto& idx : points) {
                    DepletionPoint& brDpt = model.GetDepletionPoint(idx);
                    xvals.push_back(brDpt._data[AD_DMOD]);
                    CrossSection delta = brDpt._xs - refDpt._xs;
                    // The moderator-density branch implicitly removes the boron branch first.
                    SubtractStoredDeltas(
                        model, delta, cType, burnup, brDpt,
                        {FittedEffect::Boron});

                    delta.ClearMacroscopic();
                    yvals.push_back(std::move(delta));
                }
                if (DegenerateBranchAxis(xvals)) {
                    WarnDegenerateBranch(model.name(), "dmod");
                    continue;
                }
                model._dmod_delt[cType][burnup] = FitCurve(xvals, yvals);
            }
        }

        // Override the rodded-side tables before anything peels against them.
        FitRoddedBranchTables(model);

        // Fit RDPL first so each spectral-history sample contains only
        // the residual that is not represented by the ordinary branch corrections.
        DumpAllPoints(model);
        FitRodDepletionDeltas(model);
        if (!UseRodDepletionCondition())
            for (auto& branch : model._rod_depletion_branch)
                branch.clear();

        model._spectral_history.clear();
        if (model.SpectralHistoryDpts().empty() &&
            model.RodHistoryDpts().empty())
            return;
        FitSpectralHistoryCorrection(model);
    }
};
} // namespace Chiffon
