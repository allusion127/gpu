#pragma once

#include "Model.h"
#include "milk.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>

namespace Chiffon {

inline constexpr double HISTORY_DELTA_EPS       = 1.0e-20;
inline constexpr double HISTORY_RATIO_DENOM_EPS = 1.0e-10;

// Branch correction settings for BPPM, TFUL, and DMOD.
struct BRCH_SETTINGS {
    bool                     apply = true;
    int                      order = 1;
    std::string              type  = "polynomial"; // "polynomial", "poly", or "spline"
    std::vector<std::string> pre_remove;
};

// Spectral-history vector correction settings, shared by IISC and IISC-RHST.
// Each power vector defines one basis term such as N_Pu239, N_U238 / N_Pu239, or rod fluence.
struct SPCT_SETTINGS {
    bool                          apply             = true;
    bool                          ctype_independent = true;
    std::vector<size_t>           isotopes;
    std::vector<std::vector<int>> powers;
    std::vector<std::string>      pre_remove = {"dmod"};
};

// Fits delta cross-sections as functions of branch variables (boron, fuel temp, etc.)
class Interpolator {
public:
    Interpolator() = default;

    // Constrained least-squares polynomial fit via KKT system.
    // Returns DeltaCrossSection with polynomial coefficients in Horner form.
    static DeltaCrossSection InterpolatePoly(const std::vector<double>& x, const std::vector<CrossSection>& y,
                                             int order, const std::vector<int>& fixed_indices = {}) {
        if (y.empty()) throw std::runtime_error("Empty y vector");

        size_t ngrp    = y[0].ngrp();
        int    numData = static_cast<int>(x.size());
        int    npoly   = order + 1;
        int    nconst  = static_cast<int>(fixed_indices.size());
        int    kktsize = npoly + nconst;

        // Vandermonde matrix V (numData x npoly)
        milk::Matrix<double> V(numData, npoly);
        for (int k = 0; k < numData; ++k) {
            double val = 1.0;
            for (int i = 0; i < npoly; ++i) {
                V(k, i) = val;
                val *= x[k];
            }
        }

        // Constraint matrix C (nconst x npoly): rows of V at fixed points
        milk::Matrix<double> C(std::max(nconst, 1), npoly);
        for (int i = 0; i < nconst; ++i)
            for (int j = 0; j < npoly; ++j)
                C(i, j) = V(fixed_indices[i], j);

        // KKT matrix: [ V^T*V  C^T ]
        //              [ C      0   ]
        milk::Matrix<double> KKT(kktsize, kktsize);

        // V^T * V -> KKT[0:npoly, 0:npoly]
        milk::multiply(milk::Transpose::Yes, milk::Transpose::No,
                       static_cast<size_t>(npoly), static_cast<size_t>(npoly), static_cast<size_t>(numData),
                       1.0, V.data(), static_cast<size_t>(npoly), V.data(), static_cast<size_t>(npoly),
                       0.0, KKT.data(), static_cast<size_t>(kktsize));

        if (nconst > 0) {
            for (int i = 0; i < npoly; ++i)
                for (int j = 0; j < nconst; ++j)
                    KKT(i, npoly + j) = C(j, i);
            for (int i = 0; i < nconst; ++i)
                for (int j = 0; j < npoly; ++j)
                    KKT(npoly + i, j) = C(i, j);
        }

        // Invert KKT via LU factorization
        std::vector<int> ipiv(kktsize);
        milk::factorizeLU(static_cast<size_t>(kktsize), KKT.data(), static_cast<size_t>(kktsize), ipiv.data());
        milk::invertLU(static_cast<size_t>(kktsize), KKT.data(), static_cast<size_t>(kktsize), ipiv.data());
        // KKT now holds KKT^{-1}

        // Weight matrix P = KKT_inv[0:npoly, 0:npoly] * V^T  (npoly x numData)
        milk::Matrix<double> P(npoly, numData);
        milk::multiply(milk::Transpose::No, milk::Transpose::Yes,
                       static_cast<size_t>(npoly), static_cast<size_t>(numData), static_cast<size_t>(npoly),
                       1.0, KKT.data(), static_cast<size_t>(kktsize),
                       V.data(), static_cast<size_t>(npoly),
                       0.0, P.data(), static_cast<size_t>(numData));

        // Build polynomial coefficients as weighted sums of y vectors (fused multiply-add)
        DeltaCrossSection delta(ngrp, npoly);
        for (int i = 0; i < npoly; ++i) {
            CrossSection coeff(ngrp);
            for (int k = 0; k < numData; ++k) {
                double w = P(i, k);
                if (std::abs(w) > 1e-12) coeff.addScaled(y[k], w);
            }
            for (int k = 0; k < nconst; ++k) {
                double w = KKT(i, npoly + k);
                if (std::abs(w) > 1e-12) coeff.addScaled(y[fixed_indices[k]], w);
            }
            delta[i] = std::move(coeff);
        }
        return delta;
    }

    static double FallingFactorial(int n, int j) {
        double f = 1.0;
        for (int i = 0; i < j; ++i)
            f *= (n - i);
        return f;
    }

    static double Factorial(int n) {
        double f = 1.0;
        for (int i = 2; i <= n; ++i)
            f *= i;
        return f;
    }

    // Piecewise polynomial (spline) fit with natural boundary conditions.
    // Builds constraint system for continuity and solves via LU.
    static DeltaCrossSection InterpolateSpline(const std::vector<double>&       x_in,
                                               const std::vector<CrossSection>& y_in, int order) {
        if (y_in.empty()) throw std::runtime_error("Empty y vector");

        int N = static_cast<int>(x_in.size());
        if (N < 2) throw std::runtime_error("Need at least 2 data points for spline");

        // Sort by x and remove duplicate x values (keep last occurrence)
        std::vector<int> sortIdx(N);
        std::iota(sortIdx.begin(), sortIdx.end(), 0);
        struct XIndexLess {
            const std::vector<double>& x;
            bool                       operator()(int a, int b) const { return x[a] < x[b]; }
        };
        std::sort(sortIdx.begin(), sortIdx.end(), XIndexLess{x_in});

        std::vector<double>       x;
        std::vector<CrossSection> y;
        x.reserve(N);
        y.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (!x.empty() && std::abs(x_in[sortIdx[i]] - x.back()) < 1e-15) {
                y.back() = y_in[sortIdx[i]];
            } else {
                x.push_back(x_in[sortIdx[i]]);
                y.push_back(y_in[sortIdx[i]]);
            }
        }
        N = static_cast<int>(x.size());
        if (N < 2) return InterpolatePoly(x, y, std::min(order, N - 1));

        int k          = std::clamp(order, 1, std::max(N - 2, 1));
        int ncoeff     = k + 1;
        int nintervals = N - 1;
        int M          = nintervals * ncoeff;

        std::vector<double> h(nintervals);
        for (int i = 0; i < nintervals; ++i) {
            h[i] = x[i + 1] - x[i];
            if (h[i] <= 0) throw std::runtime_error("Spline requires strictly increasing x values");
        }

        // Constraint system: A * coeffs = B * y_values
        milk::Matrix<double> A(M, M);
        milk::Matrix<double> B(M, N);
        int                  row = 0;

        // Left interpolation: S_i(0) = y_i
        for (int i = 0; i < nintervals; ++i) {
            A(row, i * ncoeff) = 1.0;
            B(row, i)          = 1.0;
            row++;
        }

        // Right interpolation of last interval: S_{N-2}(h_{N-2}) = y_{N-1}
        {
            int    i  = nintervals - 1;
            double hp = 1.0;
            for (int p = 0; p <= k; ++p) {
                A(row, i * ncoeff + p) = hp;
                hp *= h[i];
            }
            B(row, N - 1) = 1.0;
            row++;
        }

        // Right interpolation at interior knots: S_i(h_i) = y_{i+1}
        for (int i = 0; i < nintervals - 1; ++i) {
            double hp = 1.0;
            for (int p = 0; p <= k; ++p) {
                A(row, i * ncoeff + p) = hp;
                hp *= h[i];
            }
            B(row, i + 1) = 1.0;
            row++;
        }

        // Derivative continuity: S_i^(j)(h_i) = S_{i+1}^(j)(0) for j=1..k-1
        for (int j = 1; j < k; ++j) {
            for (int i = 0; i < nintervals - 1; ++i) {
                double hp = 1.0;
                for (int p = j; p <= k; ++p) {
                    A(row, i * ncoeff + p) = FallingFactorial(p, j) * hp;
                    hp *= h[i];
                }
                A(row, (i + 1) * ncoeff + j) -= Factorial(j);
                row++;
            }
        }

        // Natural boundary conditions
        int nLeftBC  = (k - 1) / 2;
        int nRightBC = (k - 1) - nLeftBC;

        for (int bc = 0; bc < nLeftBC; ++bc) {
            int j     = bc + 2;
            A(row, j) = Factorial(j);
            row++;
        }

        for (int bc = 0; bc < nRightBC; ++bc) {
            int    j  = bc + 2;
            int    i  = nintervals - 1;
            double hp = 1.0;
            for (int p = j; p <= k; ++p) {
                A(row, i * ncoeff + p) = FallingFactorial(p, j) * hp;
                hp *= h[i];
            }
            row++;
        }

        // Solve A * P = B via LU factorization (P = A^{-1} * B, M x N)
        milk::Matrix<double> P     = B;
        milk::Matrix<double> Acopy = A;
        std::vector<int>     ipiv(M);
        milk::solveLinearSystem(static_cast<size_t>(M), static_cast<size_t>(N),
                                Acopy.data(), static_cast<size_t>(M),
                                ipiv.data(), P.data(), static_cast<size_t>(N));

        // Build spline coefficients
        size_t            ngrp = y[0].ngrp();
        DeltaCrossSection delta(ngrp, static_cast<size_t>(M), static_cast<size_t>(ncoeff), x);

        for (int idx = 0; idx < M; ++idx) {
            CrossSection coeff(ngrp);
            for (int j = 0; j < N; ++j) {
                double w = P(idx, j);
                if (std::abs(w) > 1e-12) coeff.addScaled(y[j], w);
            }
            delta[idx] = std::move(coeff);
        }
        return delta;
    }

    // Dispatch to polynomial or spline based on type string.
    // Spline with two unique points is a linear spline; one point falls back to constant.
    static DeltaCrossSection Interpolate(const std::vector<double>& x, const std::vector<CrossSection>& y,
                                         int order, const std::string& type,
                                         const std::vector<int>& fixed_indices = {}) {
        if ((type == "spline" || type == "piecewise linear") && x.size() >= 2)
            return InterpolateSpline(x, y, order);
        return InterpolatePoly(x, y, order, fixed_indices);
    }

    static double RodDepletionAxis(const DepletionPoint& point) {
        if (std::isfinite(point._rod_fluence))
            return point._rod_fluence;
        return point._data[AD_BURN];
    }

    static double RodDepletionAxis(const Model& model, int cType, int burnup,
                                   const DepletionPoint& point) {
        if (std::isfinite(point._rod_fluence))
            return point._rod_fluence;

        for (const auto& dpt : model.RodDepletionDpts()) {
            if (dpt._btyp != REFR || dpt._trajectory_reference)
                continue;
            if (dpt._ctyp != cType || dpt.burnKey() != burnup)
                continue;
            if (std::isfinite(dpt._rod_fluence))
                return dpt._rod_fluence;
        }

        return point._data[AD_BURN];
    }

    static void SubtractStoredDelta(Model& model, CrossSection& delta, int cType, int burnup,
                                    const DepletionPoint& point, const std::string& name) {
        const BranchDelta* stored = nullptr;
        double             x      = 0.0;
        if (name == "bppm" || name == "boron") {
            stored = &model._bppm_delt;
            x      = point.iden("50002");
        } else if (name == "tful" || name == "tfuel") {
            stored = &model._tful_delt;
            x      = std::sqrt(point._data[AD_TFUL]);
        } else if (name == "dmod") {
            stored = &model._dmod_delt;
            x      = point._data[AD_DMOD];
        } else if (name == "rod depletion" || name == "rod_depletion") {
            stored = &model._rod_depletion_delt;
            x      = RodDepletionAxis(model, cType, burnup, point);
        } else {
            return;
        }

        if (!stored->contains(cType) || stored->at(cType).empty())
            return;
        const auto& bmap = stored->at(cType);
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
        delta -= branchDelta;
    }

    // Remove already-fitted branch effects before fitting a later correction.
    // This keeps DMOD, SPCT, and RHST residuals from refitting the same XS change.
    static void SubtractStoredDeltas(Model& model, CrossSection& delta, int cType, int burnup,
                                     const DepletionPoint& point, const std::vector<std::string>& names) {
        for (const auto& name : names)
            SubtractStoredDelta(model, delta, cType, burnup, point, name);
    }

    // Scalar x coordinate for state-history correction terms.
    // Recreate the vector x coordinate used when an earlier correction was stored.
    static double StoredHistoryX(const HistoryDeltaCorrection& settings,
                                 const DepletionPoint&         dpt,
                                 const DepletionPoint&         refDpt) {
        return EvalVectorTerm(settings.vector_isotopes, settings.vector_powers, dpt, refDpt);
    }

    // IISC is stored ctype-independent (ctype 0); every other correction is keyed by current ctype.
    static int HistoryDeltaCTypeForPoint(const HistoryDeltaCorrection& settings, int currentCType) {
        return settings.kind == Hk::CTYPE_INDEP_VEC ? 0 : currentCType;
    }

    static bool IsCenteredProductTerm(const std::vector<int>& powers) {
        int degree = 0;
        for (const int power : powers) {
            if (power < 0)
                return false;
            degree += power;
        }
        return degree > 1;
    }

    // Evaluate one absolute isotope-density basis term from a caller-provided isotope-density accessor.
    template <typename CurrentAt>
    static double EvalAbsoluteVectorTermFromAccessor(const std::vector<size_t>& isotopes,
                                                     const std::vector<int>&    powers,
                                                     CurrentAt                  currentAt) {
        // Detect ratio-like terms first; a negative power means the isotope belongs to the denominator.
        bool ratio_term = false;
        for (const int power : powers) {
            if (power < 0) {
                ratio_term = true;
                break;
            }
        }

        if (ratio_term) {
            // Build numerator and denominator separately so denominator protection is explicit.
            double current_num = 1.0;
            double current_den = 1.0;
            for (size_t i = 0; i < isotopes.size(); ++i) {
                // Missing powers are treated as zero, so the isotope does not contribute.
                const int power = i < powers.size() ? powers[i] : 0;
                if (power == 0)
                    continue;

                // Out-of-range isotope indices are treated as zero density for incomplete input vectors.
                const size_t iso     = isotopes[i];
                const double raw     = currentAt(iso);
                const double current = std::max(0.0, raw);
                if (power > 0) {
                    // Positive powers multiply the numerator.
                    for (int p = 0; p < power; ++p)
                        current_num *= current;
                } else {
                    // Negative powers multiply the denominator by the absolute exponent.
                    for (int p = 0; p < -power; ++p)
                        current_den *= current;
                }
            }

            // Avoid large artificial ratios when the denominator isotope is absent.
            if (current_den < HISTORY_RATIO_DENOM_EPS)
                return 0.0;

            // Return the ratio basis term, for example N_Pu239 / N_U238.
            return current_num / current_den;
        }

        // Non-ratio terms are ordinary products of isotope densities and powers.
        double       value = 1.0;
        const size_t n     = std::min(isotopes.size(), powers.size());
        for (size_t i = 0; i < n; ++i) {
            const size_t iso     = isotopes[i];
            const double raw     = currentAt(iso);
            const double current = std::max(0.0, raw);
            for (int p = 0; p < powers[i]; ++p)
                value *= current;
        }
        return value;
    }

    // Evaluate one vector basis term from current and reference accessors.
    // Linear terms and ratio terms keep the historical absolute-coordinate form.
    // Pure products above first order are centered around the reference state,
    // e.g. (N_Pu239 - Nref_Pu239)^2, so a quadratic term is not absorbed by
    // the corresponding first-order isotope axis.
    template <typename CurrentAt, typename ReferenceAt>
    static double EvalVectorTermFromAccessors(const std::vector<size_t>& isotopes,
                                              const std::vector<int>&    powers,
                                              CurrentAt                  currentAt,
                                              ReferenceAt                referenceAt) {
        if (!IsCenteredProductTerm(powers))
            return EvalAbsoluteVectorTermFromAccessor(isotopes, powers, currentAt);

        double       value = 1.0;
        const size_t n     = std::min(isotopes.size(), powers.size());
        for (size_t i = 0; i < n; ++i) {
            const int power = powers[i];
            if (power == 0)
                continue;
            const size_t iso     = isotopes[i];
            const double current = std::max(0.0, currentAt(iso));
            const double ref     = std::max(0.0, referenceAt(iso));
            const double delta   = current - ref;
            for (int p = 0; p < power; ++p)
                value *= delta;
        }
        return value;
    }

    static double EvalVectorTerm(const std::vector<size_t>& isotopes,
                                 const std::vector<int>&    powers,
                                 const DepletionPoint&      dpt,
                                 const DepletionPoint&      refDpt) {
        return EvalVectorTermFromAccessors(isotopes, powers, PointDensity{dpt}, PointDensity{refDpt});
    }

    static bool HasSpectralVectorInputs(const DepletionPoint& dpt,
                                        const SPCT_SETTINGS&  settings) {
        for (const size_t iso : settings.isotopes) {
            if (iso == Hv::ROD_DEPL) {
                const bool roddedPoint = dpt._ctyp > 0 || dpt._trajectory_ctyp > 0;
                if (roddedPoint && !std::isfinite(dpt._rod_fluence))
                    throw std::runtime_error("CHIFFON Interpolator: missing rod material fluence for rodded spectral point.");
                continue;
            }
            if (iso == Hv::LOG_PU) {
                if (dpt._iden.size() <= Isotope::iPu239)
                    return false;
                continue;
            }
            if (iso == Hv::LOG_XE) {
                if (dpt._iden.size() <= Isotope::iXe135)
                    return false;
                continue;
            }
            if (iso == Hv::LOG_U_PU) {
                if (dpt._iden.size() <= Isotope::iU238 || dpt._iden.size() <= Isotope::iPu239)
                    return false;
                continue;
            }
            if (iso == Hv::TOT_BURN)
                continue;
            if (iso == Hv::CUR_ROD_FRAC)
                continue;
            if (iso == Hv::FUEL_ROD_FLU) {
                const bool roddedTrajectoryPoint = dpt._ctyp > 0 || dpt._trajectory_ctyp > 0;
                if (roddedTrajectoryPoint && !std::isfinite(dpt._fuel_rod_fluence))
                    throw std::runtime_error("CHIFFON Interpolator: missing fuel rod fluence for rodded spectral point.");
                continue;
            }
            if (iso == Hv::ROD_BURN_FRAC)
                continue;
            if (iso == Hv::FUEL_ROD_BURN)
                continue;
            if (iso == Hv::ROD_BURN)
                continue;
            if (iso == Hv::ROD_FLU) {
                const bool roddedPoint = dpt._ctyp > 0;
                if (roddedPoint && !std::isfinite(dpt._rod_fluence))
                    throw std::runtime_error("CHIFFON Interpolator: missing rod fluence for rodded spectral point.");
                continue;
            }
            if (iso == Hv::FAST_THERM ||
                iso == Hv::THERM_FRAC) {
                if (dpt._aflx.size() < 2)
                    return false;
                continue;
            }
            if (dpt._iden.size() <= iso)
                return false;
        }
        return true;
    }

private:
    // Isotope-vector density accessor shared by EvalAbsoluteVectorTermFromAccessor and EvalVectorTerm.
    // Maps a basis isotope index to the point's coordinate value, including the
    // synthetic Hv:: history coordinates (rod depletion, log densities, fluence, etc.).
    struct PointDensity {
        const DepletionPoint& dpt;
        double                operator()(size_t iso) const {
            if (iso == Hv::ROD_DEPL)
                // ROD_DEPL deliberately reuses the rod-fluence O(1) scaling.
                return hvRodFluCoord(dpt._rod_fluence);
            if (iso == Hv::LOG_XE)
                return dpt._iden.size() > Isotope::iXe135
                                          ? hvLogIso(dpt._iden[Isotope::iXe135])
                                          : hvLogIso(0.0);
            if (iso == Hv::LOG_PU)
                return dpt._iden.size() > Isotope::iPu239
                                          ? hvLogIso(dpt._iden[Isotope::iPu239])
                                          : hvLogIso(0.0);
            if (iso == Hv::LOG_U_PU) {
                const double u238  = dpt._iden.size() > Isotope::iU238 ? dpt._iden[Isotope::iU238] : 0.0;
                const double pu239 = dpt._iden.size() > Isotope::iPu239 ? dpt._iden[Isotope::iPu239] : 0.0;
                return hvLogRatio(u238, pu239);
            }
            if (iso == Hv::TOT_BURN)
                return hvBurnCoord(static_cast<double>(dpt.burnKey()));
            if (iso == Hv::CUR_ROD_FRAC)
                return dpt._ctyp > 0 ? 1.0 : 0.0;
            if (iso == Hv::FUEL_ROD_FLU)
                return hvRodFluCoord(dpt._fuel_rod_fluence);
            if (iso == Hv::ROD_BURN_FRAC) {
                if (dpt._trajectory_ctyp <= 0)
                    return 0.0;
                return hvRoddedBurnFrac(static_cast<double>(dpt.burnKey()),
                                                       static_cast<double>(dpt.burnKey()));
            }
            if (iso == Hv::FUEL_ROD_BURN) {
                if (dpt._trajectory_ctyp <= 0)
                    return 0.0;
                return hvBurnCoord(static_cast<double>(dpt.burnKey()));
            }
            if (iso == Hv::ROD_BURN) {
                if (dpt._ctyp <= 0)
                    return 0.0;
                if (std::isfinite(dpt._rod_fluence) && dpt._rod_fluence <= 0.0)
                    return 0.0;
                return hvBurnCoord(static_cast<double>(dpt.burnKey()));
            }
            if (iso == Hv::ROD_FLU) {
                if (dpt._ctyp <= 0)
                    return 0.0;
                return hvRodFluCoord(dpt._rod_fluence);
            }
            if (iso == Hv::FAST_THERM)
                return hvFastThermRatio(dpt._aflx);
            if (iso == Hv::THERM_FRAC)
                return hvThermFluxFrac(dpt._aflx);
            return iso < dpt._iden.size() ? dpt._iden[iso] : 0.0;
        }
    };

    struct ScalarHistoryFitData {
        std::vector<double>       xvals;
        std::vector<CrossSection> yvals;
        bool                      hasZero = false;
    };

    struct VectorFitData {
        std::vector<std::vector<double>> xvals;
        std::vector<CrossSection>        yvals;
        bool                             has_reference = false;
    };

    using HistoryFitKey       = std::pair<int, int>;
    using ScalarHistoryFitMap = std::map<HistoryFitKey, ScalarHistoryFitData>;
    using VectorFitMap        = std::map<HistoryFitKey, VectorFitData>;

    static std::vector<double> EvalVectorXForBasis(const DepletionPoint&                dpt,
                                                   const DepletionPoint&                refDpt,
                                                   const std::vector<size_t>&           isotopes,
                                                   const std::vector<std::vector<int>>& basisPowers) {
        // Allocate one scalar coordinate per requested isotope basis term.
        std::vector<double> x(basisPowers.size(), 0.0);
        for (size_t i = 0; i < basisPowers.size(); ++i) {
            // Evaluate the i-th basis term from the point's isotope density vector.
            x[i] = EvalVectorTerm(isotopes, basisPowers[i], dpt, refDpt);
        }
        return x;
    }

    static std::vector<double> EvalSpectralVectorX(const DepletionPoint& dpt,
                                                   const DepletionPoint& refDpt,
                                                   const SPCT_SETTINGS&  settings) {
        // Spectral settings define both the isotope list and the power vectors.
        return EvalVectorXForBasis(dpt, refDpt, settings.isotopes, settings.powers);
    }

    static void AddScalarReferenceZero(ScalarHistoryFitData& data,
                                       const DepletionPoint& refDpt) {
        if (data.hasZero)
            return;
        data.xvals.push_back(0.0);
        CrossSection zeroDelta = refDpt._xs - refDpt._xs;
        zeroDelta.ClearMacroscopic();
        data.yvals.push_back(std::move(zeroDelta));
        data.hasZero = true;
    }

    static void AddVectorReferenceZero(VectorFitData&        data,
                                       const DepletionPoint& refDpt,
                                       const SPCT_SETTINGS&  settings) {
        if (data.has_reference)
            return;
        // The first sample is the zero-residual anchor; vector fitting uses it as x_ref.
        data.xvals.push_back(EvalSpectralVectorX(refDpt, refDpt, settings));
        CrossSection zeroDelta = refDpt._xs - refDpt._xs;
        zeroDelta.ClearMacroscopic();
        data.yvals.push_back(std::move(zeroDelta));
        data.has_reference = true;
    }

    static void SubtractPriorHistoryDeltas(Model&                model,
                                           CrossSection&         delta,
                                           const DepletionPoint& dpt,
                                           const DepletionPoint& refDpt,
                                           int                   refCType,
                                           int                   refBurnup) {
        // History corrections are fitted sequentially, so later residuals must remove earlier ones.
        // The prior corrections are the IISC vector(s) appended above: ctype-independent
        // (CTYPE_INDEP_VEC, stored under ctype 0) or ctype-keyed (VEC, stored under the current
        // ctype). HistoryDeltaCTypeForPoint selects the matching storage key.
        for (const auto& priorSettings : model._history_deltas) {
            const auto& priorDelta = priorSettings.delta;
            const int   priorCType = HistoryDeltaCTypeForPoint(priorSettings, refCType);
            if (!priorDelta.contains(priorCType) || !priorDelta.at(priorCType).contains(refBurnup))
                continue;
            const double priorX = StoredHistoryX(priorSettings, dpt, refDpt);
            delta -= priorDelta.at(priorCType).at(refBurnup).Delta(priorX);
        }
    }

    static void AddCurrentSpectralFitPoint(Model&                model,
                                           const SPCT_SETTINGS&  spectralSettings,
                                           VectorFitMap&         fitData,
                                           const DepletionPoint& dpt) {
        // Normal SPCT points are reference-like extra points from the current state.
        if (dpt._btyp != REFR)
            return;
        // The fit cannot evaluate requested vector coordinates if the point data are incomplete.
        if (!HasSpectralVectorInputs(dpt, spectralSettings) || !dpt._xs.has_micx())
            return;
        // The ordinary main reference is the zero-history state for the same ctype/burnup.
        const DepletionPoint& mainRef      = model.GetDepletionPoint(dpt._ctyp, dpt.burnKey());
        const int             currentCType = mainRef._ctyp;
        const int             burnup       = mainRef.burnKey();
        CrossSection          delta        = dpt._xs - mainRef._xs;
        SubtractStoredDeltas(model, delta, currentCType, burnup, dpt, spectralSettings.pre_remove);
        SubtractPriorHistoryDeltas(model, delta, dpt, mainRef, currentCType, burnup);
        delta.ClearMacroscopic();
        // Keyless design: the IISC vector is always ctype-independent (stored under ctype 0).
        const int storageCType = 0;
        auto&     data         = fitData[{storageCType, burnup}];
        AddVectorReferenceZero(data, mainRef, spectralSettings);
        data.xvals.push_back(EvalSpectralVectorX(dpt, mainRef, spectralSettings));
        data.yvals.push_back(std::move(delta));
    }

    // Result of the shared head used by every AddRhst*FitPoint variant: the
    // ordinary main reference for the point plus the pre-removed,
    // prior-history-removed, macroscopic-cleared residual cross-section.
    struct HistoryMainReference {
        DepletionPoint mainRef;
        CrossSection   delta;
    };

    // Build the ordinary main reference for currentCType/burnup (via reference
    // burnup interpolation), then form the residual relative to it: subtract the
    // configured prior branch deltas, optionally subtract earlier history
    // corrections, and clear macroscopic XS. This is the identical opening
    // sequence shared by the AddRhst*FitPoint helpers.
    static HistoryMainReference BuildHistoryMainReferenceResidual(Model&                          model,
                                                                  const DepletionPoint&           dpt,
                                                                  int                             currentCType,
                                                                  int                             burnup,
                                                                  const std::vector<std::string>& preRemove,
                                                                  bool                            subtractPriorHistory) {
        CrossSection                mainRefXs;
        milk::Vector<double>        mainRefIden;
        std::array<double, AD_SIZE> mainRefData;
        std::vector<double>         mainRefFlux;
        model.FillReferenceState(mainRefXs, mainRefIden, mainRefData, &mainRefFlux, currentCType, burnup);
        DepletionPoint mainRef(dpt._ngrp, dpt._npin, BRANCHTYPE::REFR, currentCType);
        mainRef._xs   = std::move(mainRefXs);
        mainRef._iden = std::move(mainRefIden);
        mainRef._data = mainRefData;
        mainRef._aflx = std::move(mainRefFlux);

        CrossSection delta = dpt._xs - mainRef._xs;
        SubtractStoredDeltas(model, delta, currentCType, burnup, dpt, preRemove);
        if (subtractPriorHistory)
            SubtractPriorHistoryDeltas(model, delta, dpt, mainRef, currentCType, burnup);
        delta.ClearMacroscopic();
        return {std::move(mainRef), std::move(delta)};
    }

    static void AddRhstCurrentSpectralFitPoint(Model&                model,
                                               const SPCT_SETTINGS&  spectralSettings,
                                               VectorFitMap&         fitData,
                                               const DepletionPoint& dpt) {
        // Unified SPCT/RHST mode treats a rod-history HGC point as another
        // current-state isotope-vector sample, not as a separate trajectory residual.
        if (dpt._btyp != REFR)
            return;
        if (!HasSpectralVectorInputs(dpt, spectralSettings) || !dpt._xs.has_micx())
            return;

        const int trajectoryCType = dpt._trajectory_ctyp >= 0 ? dpt._trajectory_ctyp : 0;
        if (trajectoryCType <= 0)
            return;

        const int currentCType = dpt._ctyp;
        const int burnup       = dpt.burnKey();

        HistoryMainReference built = BuildHistoryMainReferenceResidual(
            model, dpt, currentCType, burnup, spectralSettings.pre_remove, true);
        DepletionPoint& mainRef = built.mainRef;
        CrossSection    delta   = std::move(built.delta);

        // Keyless design: RHST residuals are stored under the current ctype (no pair key).
        const int storageCType = currentCType;
        auto&     data         = fitData[{storageCType, burnup}];
        AddVectorReferenceZero(data, mainRef, spectralSettings);
        data.xvals.push_back(EvalSpectralVectorX(dpt, mainRef, spectralSettings));
        data.yvals.push_back(std::move(delta));
    }

    static void AddCurrentSpectralResidualZeroPoint(Model&                model,
                                                    const SPCT_SETTINGS&  spectralSettings,
                                                    VectorFitMap&         fitData,
                                                    const DepletionPoint& dpt,
                                                    int                   storageCType) {
        if (dpt._btyp != REFR)
            return;
        if (!HasSpectralVectorInputs(dpt, spectralSettings) || !dpt._xs.has_micx())
            return;

        const DepletionPoint& mainRef = model.GetDepletionPoint(dpt._ctyp, dpt.burnKey());
        auto&                 data    = fitData[{storageCType, mainRef.burnKey()}];
        AddVectorReferenceZero(data, mainRef, spectralSettings);

        CrossSection zeroDelta = mainRef._xs - mainRef._xs;
        zeroDelta.ClearMacroscopic();
        data.xvals.push_back(EvalSpectralVectorX(dpt, mainRef, spectralSettings));
        data.yvals.push_back(std::move(zeroDelta));
    }

    static void AppendVectorCorrections(Model&                               model,
                                        int                                  kind,
                                        VectorFitMap&                        fitData,
                                        const std::vector<size_t>&           correctionIsotopes,
                                        const std::vector<std::vector<int>>& correctionPowers) {
        // Keep one HistoryDeltaCorrection per isotope-vector basis term.
        std::vector<HistoryDeltaCorrection> vectorCorrections(correctionPowers.size());
        for (size_t t = 0; t < correctionPowers.size(); ++t) {
            // Spectral vector corrections are stored as SPCT-family history deltas.
            vectorCorrections[t].branch_type = BRANCHTYPE::SPCT;
            // The kind distinguishes ordinary isotope-vector corrections from other history corrections.
            vectorCorrections[t].kind = kind;
            // Store the isotope index list needed to evaluate this correction at query time.
            vectorCorrections[t].vector_isotopes = correctionIsotopes;
            // Store the power vector for this one basis term.
            vectorCorrections[t].vector_powers = correctionPowers[t];
        }

        for (auto& [key, data] : fitData) {
            // Empty datasets cannot produce coefficients.
            if (data.yvals.empty())
                continue;
            // Fit CrossSection-valued coefficients for c0 + c1*x1 + ...
            auto coeffs = FitVectorTermCoefficients(data.xvals, data.yvals);
            // A successful fit returns one constant plus one coefficient per basis term.
            if (coeffs.size() != correctionPowers.size() + 1)
                continue;

            for (size_t t = 0; t < correctionPowers.size(); ++t) {
                // Start with a zero constant for every basis term.
                CrossSection constant = coeffs[t + 1] - coeffs[t + 1];
                // Store the global constant only with the first term to avoid duplicating c0.
                if (t == 0)
                    constant = std::move(coeffs[0]);
                // Vector corrections should not store macroscopic XS deltas.
                coeffs[t + 1].ClearMacroscopic();
                // The optional constant term follows the same microscopic/lumped-only rule.
                constant.ClearMacroscopic();

                // A two-coefficient polynomial encodes constant + slope * x for this basis term.
                DeltaCrossSection dxs(coeffs[t + 1].ngrp(), 2);
                // Coefficient 0 is the optional constant contribution.
                dxs[0] = std::move(constant);
                // Coefficient 1 is the fitted slope for this isotope-vector feature.
                dxs[1] = std::move(coeffs[t + 1]);
                // Store the fitted delta under the matching ctype/burnup key.
                vectorCorrections[t].delta[key.first][key.second] = std::move(dxs);
            }
        }

        for (auto& correction : vectorCorrections) {
            // Drop empty correction objects so the model only stores usable deltas.
            if (!correction.delta.empty())
                model._history_deltas.push_back(std::move(correction));
        }
    }

    static void FitRodDepletionDeltas(Model& model, const BRCH_SETTINGS& settings) {
        model._rod_depletion_delt.clear();
        if (!settings.apply || model._rod_depletion_dpts.empty())
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

            if (hasPairedReference) {
                std::map<int, const DepletionPoint*> nondepletedByBurn;
                for (const auto* dpt : points) {
                    if (dpt->_trajectory_reference)
                        nondepletedByBurn[dpt->burnKey()] = dpt;
                }

                std::vector<double>       xvals;
                std::vector<CrossSection> yvals;
                xvals.reserve(points.size());
                yvals.reserve(points.size());
                if (!nondepletedByBurn.empty()) {
                    const DepletionPoint* ref  = nondepletedByBurn.begin()->second;
                    CrossSection          zero = ref->_xs - ref->_xs;
                    zero.ClearMacroscopic();
                    xvals.push_back(0.0);
                    yvals.push_back(std::move(zero));
                }
                for (const auto* dpt : points) {
                    if (dpt->_trajectory_reference)
                        continue;
                    auto refIt = nondepletedByBurn.find(dpt->burnKey());
                    if (refIt == nondepletedByBurn.end())
                        continue;

                    CrossSection delta = dpt->_xs - refIt->second->_xs;
                    delta.ClearMacroscopic();
                    xvals.push_back(RodDepletionAxis(*dpt));
                    yvals.push_back(std::move(delta));
                }

                const int order = std::min(settings.order, static_cast<int>(yvals.size()) - 1);
                if (order < 0)
                    continue;
                model._rod_depletion_delt[ctype][0] =
                    Interpolate(xvals, yvals, order, settings.type);
                continue;
            }

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

            const int order = std::min(settings.order, static_cast<int>(yvals.size()) - 1);
            if (order < 0)
                continue;
            model._rod_depletion_delt[ctype][0] =
                Interpolate(xvals, yvals, order, settings.type);
        }
    }

    // Solve y = c1*(x1-xref1) + ... for CrossSection-valued coefficients,
    // then store the equivalent c0 + c1*x1 + ... form. The first fitting
    // sample is the reference isotope vector with zero residual.
    // A modified Gram-Schmidt QR solves the centered design after dropping zero-variance
    // and collinear features; if the remaining design is still rank-deficient the fit is
    // abandoned (returns empty) rather than regularized.
    static std::vector<CrossSection> FitVectorTermCoefficients(
        const std::vector<std::vector<double>>& xvals,
        const std::vector<CrossSection>&        yvals) {
        if (yvals.empty() || xvals.empty()) return {};

        const int ndata    = static_cast<int>(yvals.size());
        const int nfeature = static_cast<int>(xvals[0].size());
        if (nfeature == 0) return {};
        const std::vector<double>& xref = xvals[0];

        std::vector<double> rawDiff(static_cast<size_t>(ndata) * static_cast<size_t>(nfeature), 0.0);
        std::vector<double> rawScale(nfeature, 0.0);
        for (int i = 0; i < ndata; ++i) {
            if (static_cast<int>(xvals[i].size()) != nfeature)
                return {};
            for (int j = 0; j < nfeature; ++j) {
                const double raw                               = xvals[i][j] - xref[j];
                rawDiff[static_cast<size_t>(i) * nfeature + j] = raw;
                rawScale[j]                                    = std::max(rawScale[j], std::abs(raw));
            }
        }

        std::vector<int> activeFeature;
        activeFeature.reserve(static_cast<size_t>(nfeature));
        for (int j = 0; j < nfeature; ++j) {
            if (rawScale[j] > 1.0e-30)
                activeFeature.push_back(j);
        }

        CrossSection zeroCoeff = yvals[0] - yvals[0];
        zeroCoeff.ClearMacroscopic();
        if (activeFeature.empty()) {
            std::vector<CrossSection> coeffs;
            coeffs.reserve(static_cast<size_t>(nfeature) + 1);
            for (int j = 0; j < nfeature + 1; ++j)
                coeffs.push_back(zeroCoeff);
            return coeffs;
        }

        if (activeFeature.size() > 1) {
            std::vector<int>    independentFeature;
            std::vector<double> qcols;
            independentFeature.reserve(activeFeature.size());
            qcols.reserve(static_cast<size_t>(ndata) * activeFeature.size());

            for (const int feature : activeFeature) {
                std::vector<double> v(ndata, 0.0);
                const double        scale = std::max(1.0, rawScale[feature]);
                for (int i = 0; i < ndata; ++i)
                    v[i] = rawDiff[static_cast<size_t>(i) * nfeature + feature] / scale;

                for (size_t k = 0; k < independentFeature.size(); ++k) {
                    double dot = 0.0;
                    for (int i = 0; i < ndata; ++i)
                        dot += qcols[k * static_cast<size_t>(ndata) + static_cast<size_t>(i)] * v[i];
                    for (int i = 0; i < ndata; ++i)
                        v[i] -= dot * qcols[k * static_cast<size_t>(ndata) + static_cast<size_t>(i)];
                }

                double norm2 = 0.0;
                for (double value : v)
                    norm2 += value * value;
                const double norm = std::sqrt(norm2);
                if (norm < 1.0e-12)
                    continue;

                independentFeature.push_back(feature);
                for (int i = 0; i < ndata; ++i)
                    qcols.push_back(v[i] / norm);
            }

            activeFeature = std::move(independentFeature);
            if (activeFeature.empty()) {
                std::vector<CrossSection> coeffs;
                coeffs.reserve(static_cast<size_t>(nfeature) + 1);
                for (int j = 0; j < nfeature + 1; ++j)
                    coeffs.push_back(zeroCoeff);
                return coeffs;
            }
        }

        const int           nterm = static_cast<int>(activeFeature.size());
        std::vector<double> colScale(nterm, 1.0);
        std::vector<double> design(static_cast<size_t>(ndata) * static_cast<size_t>(nterm), 0.0);
        for (int t = 0; t < nterm; ++t) {
            const int j = activeFeature[t];
            colScale[t] = std::max(1.0, rawScale[j]);
            for (int i = 0; i < ndata; ++i)
                design[static_cast<size_t>(i) * nterm + t] =
                    rawDiff[static_cast<size_t>(i) * nfeature + j];
        }
        for (int i = 0; i < ndata; ++i)
            for (int j = 0; j < nterm; ++j)
                design[static_cast<size_t>(i) * nterm + j] /= colScale[j];

        std::vector<double> weights(static_cast<size_t>(nterm) * static_cast<size_t>(ndata), 0.0);
        bool                solved = false;
        std::vector<double> q(static_cast<size_t>(nterm) * static_cast<size_t>(ndata), 0.0);
        std::vector<double> r(static_cast<size_t>(nterm) * static_cast<size_t>(nterm), 0.0);
        std::vector<double> v(ndata, 0.0);
        bool                fullRank = true;

        for (int j = 0; j < nterm && fullRank; ++j) {
            for (int i = 0; i < ndata; ++i)
                v[i] = design[static_cast<size_t>(i) * nterm + j];

            for (int k = 0; k < j; ++k) {
                double dot = 0.0;
                for (int i = 0; i < ndata; ++i)
                    dot += q[static_cast<size_t>(k) * ndata + i] * v[i];
                r[static_cast<size_t>(k) * nterm + j] = dot;
                for (int i = 0; i < ndata; ++i)
                    v[i] -= dot * q[static_cast<size_t>(k) * ndata + i];
            }

            double norm2 = 0.0;
            for (double value : v)
                norm2 += value * value;
            const double norm = std::sqrt(norm2);
            if (norm < 1.0e-12) {
                fullRank = false;
                break;
            }

            r[static_cast<size_t>(j) * nterm + j] = norm;
            for (int i = 0; i < ndata; ++i)
                q[static_cast<size_t>(j) * ndata + i] = v[i] / norm;
        }

        if (fullRank) {
            for (int i = 0; i < ndata; ++i) {
                for (int j = nterm - 1; j >= 0; --j) {
                    double sum = q[static_cast<size_t>(j) * ndata + i];
                    for (int k = j + 1; k < nterm; ++k)
                        sum -= r[static_cast<size_t>(j) * nterm + k] *
                               weights[static_cast<size_t>(k) * ndata + i];
                    weights[static_cast<size_t>(j) * ndata + i] =
                        sum / r[static_cast<size_t>(j) * nterm + j];
                }
            }
            solved = true;
        }

        if (!solved)
            return {};

        std::vector<CrossSection> slopes;
        slopes.reserve(static_cast<size_t>(nfeature));
        for (int j = 0; j < nfeature; ++j)
            slopes.push_back(zeroCoeff);

        for (int j = 0; j < nterm; ++j) {
            CrossSection coeff(yvals[0].ngrp());
            for (int i = 0; i < ndata; ++i) {
                const double w = weights[static_cast<size_t>(j) * ndata + i];
                if (std::abs(w) > 1.0e-14)
                    coeff.addScaled(yvals[i], w);
            }
            coeff *= (1.0 / colScale[j]);
            slopes[activeFeature[j]] = std::move(coeff);
        }

        std::vector<CrossSection> coeffs;
        coeffs.reserve(static_cast<size_t>(nfeature) + 1);
        CrossSection constant = yvals[0] - yvals[0];
        for (int j = 0; j < nfeature; ++j)
            constant.addScaled(slopes[j], -xref[j]);
        coeffs.push_back(std::move(constant));
        for (auto& slope : slopes)
            coeffs.push_back(std::move(slope));
        return coeffs;
    }

public:
    // Fit delta cross-sections for all branch variations in a model.
    // Only micx and lmpx are interpolated; macroscopic XS is reconstructed at query time.
    static void Interpolate(Model& model, bool useBppm, bool useTful, bool useDmod,
                            const BRCH_SETTINGS&       bppmSettings,
                            const BRCH_SETTINGS&       tfulSettings,
                            const BRCH_SETTINGS&       dmodSettings,
                            const SPCT_SETTINGS&       spectralSettings         = {},
                            const BRCH_SETTINGS&       rodDepletionSettings     = {},
                            std::vector<SPCT_SETTINGS> rhstSpectralSettingsList = {}) {

        // BPPM branch: fit XS residuals against boron number density.
        if (!useBppm) {
            model._bppm_delt.clear();
        }
        for (auto& [cType, maps] : model.GetBranch(BPPM)) {
            if (!useBppm) break;
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
                model._bppm_delt[cType][burnup] =
                    Interpolate(xvals, yvals, bppmSettings.order, bppmSettings.type);
            }
        }

        // TFUL branch: fit XS residuals against sqrt(fuel temperature).
        if (!useTful) {
            model._tful_delt.clear();
        }
        for (auto& [cType, maps] : model.GetBranch(TFUL)) {
            if (!useTful) break;
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
                model._tful_delt[cType][burnup] =
                    Interpolate(xvals, yvals, tfulSettings.order, tfulSettings.type);
            }
        }

        // DMOD branch: subtract configured prior branch effects first, then
        // fit the remaining residual against moderator density.
        if (!useDmod) {
            model._dmod_delt.clear();
        }
        for (auto& [cType, maps] : model.GetBranch(DMOD)) {
            if (!useDmod) break;
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
                    SubtractStoredDeltas(model, delta, cType, burnup, brDpt, dmodSettings.pre_remove);

                    delta.ClearMacroscopic();
                    yvals.push_back(std::move(delta));
                }
                int nsize                       = static_cast<int>(points.size());
                model._dmod_delt[cType][burnup] = Interpolate(
                    xvals, yvals, dmodSettings.order, dmodSettings.type,
                    (dmodSettings.type == "polynomial" || dmodSettings.type == "poly")
                        ? std::vector<int>{0, nsize}
                        : std::vector<int>{});
            }
        }

        // Rod-material depletion must be available before SPCT/RHST residuals
        // remove already-fitted effects.
        FitRodDepletionDeltas(model, rodDepletionSettings);

        // IISC / IISC-RHST: the two fixed spectral-history vector corrections, fitted
        // after the ordinary branch deltas and rod-material depletion.
        model._history_deltas.clear();
        if (model.SpctDpts().empty() && model.RhstDpts().empty())
            return;

        const bool useSpctVector =
            spectralSettings.apply && !spectralSettings.isotopes.empty() && !spectralSettings.powers.empty();
        if (!useSpctVector)
            return;

        // IISC: ordinary spectral-history vector, measured from the current-state main reference.
        VectorFitMap iiscFitData;
        for (const auto& spctDpt : model.SpctDpts())
            AddCurrentSpectralFitPoint(model, spectralSettings, iiscFitData, spctDpt);
        // Keyless design: the IISC vector is always ctype-independent (CTYPE_INDEP_VEC).
        AppendVectorCorrections(model, Hk::CTYPE_INDEP_VEC,
                                iiscFitData, spectralSettings.isotopes, spectralSettings.powers);

        // IISC-RHST: keyless rod-history residual, fitted on what remains after the ordinary
        // branches, rod depletion and IISC are removed (handled inside AddRhstCurrentSpectralFitPoint).
        if (!model.RhstDpts().empty()) {
            if (rhstSpectralSettingsList.empty())
                rhstSpectralSettingsList.push_back(spectralSettings);
            for (const auto& stageSettings : rhstSpectralSettingsList) {
                if (!stageSettings.apply || stageSettings.isotopes.empty() || stageSettings.powers.empty())
                    continue;

                VectorFitMap residualFitData;
                for (const auto& rhstDpt : model.RhstDpts())
                    AddRhstCurrentSpectralFitPoint(model, stageSettings, residualFitData, rhstDpt);

                // Anchor the residual surface at zero for each current ctype/burnup that has rod history.
                std::set<HistoryFitKey> residualFitKeys;
                for (const auto& rhstDpt : model.RhstDpts()) {
                    const int trajectoryCType = rhstDpt._trajectory_ctyp >= 0 ? rhstDpt._trajectory_ctyp : 0;
                    if (rhstDpt._btyp != REFR || trajectoryCType <= 0)
                        continue;
                    residualFitKeys.emplace(rhstDpt._ctyp, rhstDpt.burnKey());
                }
                for (const auto& spctDpt : model.SpctDpts())
                    for (const auto& key : residualFitKeys)
                        if (key.second == spctDpt.burnKey() && key.first == spctDpt._ctyp)
                            AddCurrentSpectralResidualZeroPoint(model, stageSettings, residualFitData,
                                                                spctDpt, key.first);

                AppendVectorCorrections(model, Hk::RHST_UNIT, residualFitData,
                                        stageSettings.isotopes, stageSettings.powers);
            }
        }
    }
};
} // namespace Chiffon
