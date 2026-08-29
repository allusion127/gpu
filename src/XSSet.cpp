#include "XSSet.h"

#include "GpuFullContract.h"
#include "Importer.h"
#include "XSTiming.h"
#include "XeGpuReceipt.h"
#include "XeKernel.h"
#include "XsReconKernel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace rasbery;
using namespace Chiffon;

// XsReconKernel.h mirrors these library constants instead of including the
// HDF5-bearing Model.h; a drift here must fail the build, not the physics.
static_assert(XSTF == xsrecon::T_XSTF && XSDF == xsrecon::T_XSDF &&
              XSAF == xsrecon::T_XSAF && XSFF == xsrecon::T_XSFF &&
              XSNF == xsrecon::T_XSNF && XSKF == xsrecon::T_XSKF &&
              XSSF == xsrecon::T_XSSF && XSRF == xsrecon::T_XSRF &&
              FYLD == xsrecon::T_FYLD && XS2N == xsrecon::T_XS2N &&
              XS3N == xsrecon::T_XS3N,
              "XSTYPE order drifted from XsReconKernel.h");
static_assert(Isotope::isotopeIds.size() == static_cast<size_t>(xsrecon::NISO) &&
              static_cast<int>(Isotope::iI135) == xsrecon::I135 &&
              static_cast<int>(Isotope::iXe135) == xsrecon::XE135 &&
              static_cast<int>(Isotope::iXe135m) == xsrecon::XE135M &&
              static_cast<int>(Isotope::iAcFirst) == xsrecon::AC_FIRST &&
              static_cast<int>(Isotope::iAcLast) == xsrecon::AC_LAST,
              "isotope registry drifted from XsReconKernel.h");
static_assert(N_XS_SCALAR == xsrecon::NXS,
              "scalar XS slot count drifted from XsReconKernel.h");

namespace {
constexpr double WATER_NUMBER_DENSITY       = 0.033427699;
constexpr double BORON_DENSITY_FACTOR       = 5.5707678E-8;
constexpr int    OMP_THRESHOLD              = 64;
constexpr bool   USE_AVERAGE_DMOD_FOR_BORON = false;
constexpr int    CRAM_ORDER                 = 8;
// The device depletion arm (Task 16) transcribes ONE pole set.  A change here
// must break the build, not leave CudaCramBackend.cu quietly running the old one.
static_assert(CRAM_ORDER == cram::kOrder,
              "CRAM order drifted from CudaCramBackend.h's kOrder");
// cram::PredictorView / CorrectorView carry the eleven scalar-XS pointers as a
// fixed-size array; the fill loops below index it with N_XS_SCALAR.
static_assert(N_XS_SCALAR == 11,
              "N_XS_SCALAR drifted from cram::PredictorView::mic's extent");

// Stored scalar XS types: all but the derived XSDF/XSRF (rebuilt from transport/scatter).
constexpr int ACTIVE_XT[] = {XSTF, XSAF, XSFF, XSNF, XSKF, XSSF, FYLD, XS2N, XS3N};
constexpr int N_ACTIVE_XT = static_cast<int>(std::size(ACTIVE_XT));

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

size_t countDeltaCoefficients(const Chiffon::BurnupDelta& delta) {
    size_t count = 0;
    for (const auto& [burnup, crossSection] : delta) {
        (void)burnup;
        count += crossSection.nord();
    }
    return count;
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
    return *ScalarXS(*this)[static_cast<size_t>(xt)];
}

const milk::Vector<double>& XSArraySet::operator[](Chiffon::XSTYPE xt) const {
    if (xt == Chiffon::XSSM)
        return xssm;
    return *ScalarXS(*this)[static_cast<size_t>(xt)];
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

static void FlattenReferenceCrossSection(XsLibrary& lib, size_t flat,
                                         const Chiffon::DepletionPoint& dpt) {
    const auto&  xs   = dpt._xs;
    const size_t ng   = static_cast<size_t>(lib.ng);
    const size_t niso = lib.niso;

    // 1. Copy lumped scalar cross sections and scattering matrices.
    for (size_t ig = 0; ig < ng; ++ig) {
        const size_t off = flat * ng + ig;
        for (int xt = XSTF; xt <= XS3N; ++xt)
            lib.lib_lmpx[static_cast<XSTYPE>(xt)][off] = xs.lmpxs(ig, static_cast<XSTYPE>(xt));
        for (size_t ige = 0; ige < ng; ++ige)
            lib.lib_lmpx.xssm[flat * ng * ng + ig * ng + ige] = xs.lmpxssm(ig, ige);
    }

    // 2. Copy microscopic scalar cross sections and scattering matrices isotope by isotope.
    for (size_t iso = 0; iso < niso; ++iso) {
        const int iiso = static_cast<int>(iso);
        for (size_t ig = 0; ig < ng; ++ig) {
            const size_t off = (flat * niso + iso) * ng + ig;
            for (int xt = XSTF; xt <= XS3N; ++xt)
                lib.lib_micx[static_cast<XSTYPE>(xt)][off] = xs.mixs(iiso, ig, static_cast<XSTYPE>(xt));
            for (size_t ige = 0; ige < ng; ++ige)
                lib.lib_micx.xssm[(flat * niso + iso) * ng * ng + ig * ng + ige] =
                    xs.mixssm(iiso, ig, ige);
        }
    }

    // 3. Copy isotope densities and scalar depletion-state data.
    for (size_t iso = 0; iso < niso; ++iso)
        lib.lib_iden[flat * niso + iso] = dpt._iden[iso];
    lib.lib_burn[flat]         = dpt._data[AD_BURN];
    lib.lib_wvfr[flat]         = dpt._data[AD_WVFR];
    lib.lib_ref_branch_x[flat] = {
        (Isotope::iB10 < dpt._iden.size()) ? dpt._iden[Isotope::iB10] : 0.0,
        std::sqrt(dpt._data[AD_TFUL]),
        dpt._data[AD_DMOD]};

    // 4. Copy the average flux and normalize the fission spectrum.
    const size_t flux_count = std::min(ng, dpt._aflx.size());
    for (size_t ig = 0; ig < ng; ++ig)
        lib.lib_flux[flat * ng + ig] = ig < flux_count ? dpt._aflx[ig] : 0.0;

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
        lib.lib_chix[flat * ng + ig] = value;
    }
}

static void FlattenDeltaCrossSection(XsLibrary& lib, size_t coeff_base,
                                     const Chiffon::DeltaCrossSection& dxs) {
    const size_t ng   = static_cast<size_t>(lib.ng);
    const size_t niso = lib.niso;

    for (size_t p = 0; p < dxs.nord(); ++p) {
        const auto&  coeff_xs = dxs[p];
        const size_t coeff    = coeff_base + p;

        // 1. Copy lumped coefficient cross sections.
        for (size_t ig = 0; ig < ng; ++ig) {
            const size_t off = coeff * ng + ig;
            for (int xt = XSTF; xt <= XS3N; ++xt)
                lib.lib_coeff_lmpx[static_cast<XSTYPE>(xt)][off] =
                    coeff_xs.lmpxs(ig, static_cast<XSTYPE>(xt));
            for (size_t ige = 0; ige < ng; ++ige)
                lib.lib_coeff_lmpx.xssm[coeff * ng * ng + ig * ng + ige] =
                    coeff_xs.lmpxssm(ig, ige);
        }

        // 2. Copy microscopic coefficient cross sections when the delta carries them.
        if (!coeff_xs.has_micx()) continue;

        lib.has_coeff_micx = true;
        for (size_t iso = 0; iso < niso; ++iso) {
            const int iiso = static_cast<int>(iso);
            for (size_t ig = 0; ig < ng; ++ig) {
                const size_t off = (coeff * niso + iso) * ng + ig;
                for (int xt = XSTF; xt <= XS3N; ++xt)
                    lib.lib_coeff_micx[static_cast<XSTYPE>(xt)][off] =
                        coeff_xs.mixs(iiso, ig, static_cast<XSTYPE>(xt));
                for (size_t ige = 0; ige < ng; ++ige)
                    lib.lib_coeff_micx.xssm[(coeff * niso + iso) * ng * ng + ig * ng + ige] =
                        coeff_xs.mixssm(iiso, ig, ige);
            }
        }
    }
}

static void FlattenBranchDelta(XsLibrary& lib, const Chiffon::BranchDelta& bd, size_t mi,
                               int branch, size_t& delta_slot_idx, size_t& coeff_idx,
                               size_t& knot_offset) {
    lib.brch_base[mi][branch]        = delta_slot_idx;
    lib.brch_ctyp_stride[mi][branch] = bd.empty() ? 0 : maxBurnCount(bd);
    lib.brch_burn_stride[mi][branch] = 1;
    lib.brch_ctyp[mi][branch]        = ctypeKeys(bd);
    lib.brch_burn[mi][branch].resize(lib.brch_ctyp[mi][branch].size());
    if (bd.empty()) return;

    for (size_t ci = 0; ci < lib.brch_ctyp[mi][branch].size(); ++ci) {
        const int   ctype = lib.brch_ctyp[mi][branch][ci];
        const auto& bmap  = bd.at(ctype);
        auto&       keys  = lib.brch_burn[mi][branch][ci];
        keys              = burnKeys(bmap);
        for (size_t bi = 0; bi < keys.size(); ++bi) {
            const int    burn_key = keys[bi];
            const size_t flat_did = lib.brch_base[mi][branch] +
                                    ci * lib.brch_ctyp_stride[mi][branch] +
                                    bi * lib.brch_burn_stride[mi][branch];
            const auto& dxs = bmap.at(burn_key);

            auto& info       = lib.lib_deltas[flat_did];
            info.nord        = static_cast<int>(dxs.nord());
            info.mode        = (dxs.mode() == Chiffon::SPLINE_MODE) ? 1 : 0;
            info.ncoeff      = static_cast<int>(dxs.ncoeff());
            info.coeff_base  = static_cast<int>(coeff_idx);
            info.knot_offset = static_cast<int>(knot_offset);
            info.knot_count  = static_cast<int>(dxs.knots().size());

            // Copy knots
            for (double k : dxs.knots())
                lib.lib_knots.push_back(k);
            knot_offset += dxs.knots().size();

            FlattenDeltaCrossSection(lib, coeff_idx, dxs);
            coeff_idx += info.nord;
        }
    }
    delta_slot_idx += lib.brch_ctyp[mi][branch].size() * lib.brch_ctyp_stride[mi][branch];
}

static void FlattenSpectralHistory(
    XsLibrary& lib, const Chiffon::SpectralHistoryCorrection& correction,
    SpectralHistoryInfo& info, size_t& delta_slot_idx,
    size_t& coeff_idx, size_t& knot_offset) {
    info.term       = correction.term;
    info.delta_base = delta_slot_idx;
    info.burnups    = burnKeys(correction.delta);
    info.rod_scaled = correction.rod_scaled;

    for (const int burnup : info.burnups) {
        const auto& delta = correction.delta.at(burnup);
        auto&       flat  = lib.lib_deltas[delta_slot_idx++];
        flat.nord         = static_cast<int>(delta.nord());
        flat.mode =
            delta.mode() == Chiffon::SPLINE_MODE ? 1 : 0;
        flat.ncoeff      = static_cast<int>(delta.ncoeff());
        flat.coeff_base  = static_cast<int>(coeff_idx);
        flat.knot_offset = static_cast<int>(knot_offset);
        flat.knot_count  = static_cast<int>(delta.knots().size());

        lib.lib_knots.insert(
            lib.lib_knots.end(), delta.knots().begin(), delta.knots().end());
        knot_offset += delta.knots().size();
        FlattenDeltaCrossSection(lib, coeff_idx, delta);
        coeff_idx += flat.nord;
    }
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

XSSet::~XSSet() {
    // Tear the device backend down FIRST: ~Impl hands the nodal arena slot back
    // and frees this instance's device allocations, so nothing is left holding
    // a handle to the host arrays whose leases are released below.  Every solve
    // is synchronous from the instance's side (solve/solveNodal return after
    // their downloads landed), so no DMA is in flight here either -- plan Sec
    // 6.4's conservative drain, in place of per-stream event tracking.
    _xsrecon_backend.reset();

    // Release every host pin lease the two pin arms take, enumerated from
    // TryUpdateFlatXSGpu (the flatxs arm, the superset) and TryUpdateXsGpu (the
    // xsrecon arm, whose live-block bases are the same allocations).  Releasing
    // a base that was never leased -- a build that never ran either arm, or a
    // request that took the pageable fallback -- is a no-op, so the list is
    // unconditional.  _g.Phif() is deliberately NOT here: Geometry owns it and
    // ~Geometry releases it.
    for (int xt = 0; xt < xsrecon::NXS; ++xt) {
        const auto t = static_cast<XSTYPE>(xt);
        rasberyUnpinHost(_micx[t].data());
        rasberyUnpinHost(_lmpx[t].data());
        rasberyUnpinHost(_xs[t].data());
    }
    rasberyUnpinHost(_micx.xssm.data());
    rasberyUnpinHost(_lmpx.xssm.data());
    rasberyUnpinHost(_xs.xssm.data());
    rasberyUnpinHost(_iden.data());
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        const auto xt = static_cast<XSTYPE>(ACTIVE_XT[t]);
        rasberyUnpinHost(_ref_micx[xt].data());
        rasberyUnpinHost(_ref_lmpx[xt].data());
    }
    rasberyUnpinHost(_ref_micx.xssm.data());
    rasberyUnpinHost(_ref_lmpx.xssm.data());
    // Not from either pin arm: the nodal batch arena page-locks chifData()
    // (NodalArena::pinSlot's h_chif).  XSSet owns that buffer, so XSSet
    // releases it -- the same rule that puts jnet/phis/phif in ~Geometry.
    rasberyUnpinHost(_ref_chix.data());
}

void XSSet::LoadTHTables() {
#ifdef DATA_DIR
    const auto base = std::filesystem::path(DATA_DIR) / "include" / "Database";
#else
    const auto base = std::filesystem::path("include") / "Database";
#endif
    // Five CSV parses that depend on the BUILD, not on the deck -- the same
    // shape of per-case waste the XSLIB cache removed one level up, and the
    // same fix: parse once per process, copy the parsed tables per Driver.
    //
    // COPIED, NOT SHARED, deliberately.  milk::Table is small (four property
    // curves and a fuel-temperature table, kilobytes), the copy is measured in
    // microseconds, and XSSet::GetCpmod and friends are called from inside the
    // T/H sweep -- a shared_ptr indirection there buys nothing and a shared
    // mutable table would be the bug the XSLIB cache had to design around.
    struct ThTables {
        milk::Table cp, rho, h, t, tf;
    };
    static const ThTables tables = [&base] {
        ThTables loaded;
        loaded.cp  = milk::Table::ParseFromCSV(base / "mod_cp.csv");
        loaded.rho = milk::Table::ParseFromCSV(base / "mod_rho.csv");
        loaded.h   = milk::Table::ParseFromCSV(base / "mod_h.csv");
        loaded.t   = milk::Table::ParseFromCSV(base / "mod_t.csv");
        loaded.tf  = milk::Table::ParseFromCSV(base / "tf.csv");
        return loaded;
    }();

    _mod_cp_table  = tables.cp;
    _mod_rho_table = tables.rho;
    _mod_h_table   = tables.h;
    _mod_t_table   = tables.t;
    _tf_table      = tables.tf;
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
    _fine_rod_thermal_fluence.assign(fine_count, 0.0);
    _pu_prev.assign(static_cast<size_t>(_g.nxyz()), -1.0);
    _pu_gain_tot.assign(static_cast<size_t>(_g.nxyz()), 0.0);
    _pu_gain_rod.assign(static_cast<size_t>(_g.nxyz()), 0.0);
    _fine_rod_thermal_fluence_bos.assign(fine_count, 0.0);
    RebuildFineRodOccupancy();
}


// ---------------------------------------------------------------------------
// The shared parse: build, cache, receipt.  See XsLibrary.h.
// ---------------------------------------------------------------------------

namespace rasbery {

std::shared_ptr<const XsLibrary> BuildXsLibrary(const std::string& xs_path, int ng_in) {
    auto       owned = std::make_shared<XsLibrary>();
    XsLibrary& lib   = *owned;
    lib.ng           = ng_in;

    // Provenance.  Recorded from the file as it was read, so the cache key and
    // the receipt quote the same bytes.
    {
        std::error_code ec;
        const auto      canonical =
            std::filesystem::weakly_canonical(std::filesystem::path(xs_path), ec);
        lib.path = ec ? xs_path : canonical.lexically_normal().string();
        ec.clear();
        const auto size = std::filesystem::file_size(std::filesystem::path(xs_path), ec);
        lib.file_size   = ec ? 0 : static_cast<std::uint64_t>(size);
        ec.clear();
        const auto stamp = std::filesystem::last_write_time(std::filesystem::path(xs_path), ec);
        lib.mtime        = ec ? 0 : static_cast<std::int64_t>(stamp.time_since_epoch().count());
    }

    // 1. Parse.  LoadHDF takes its own Chiffon::Hdf5Guard and publishes the
    //    isotope registry (niso) under the registry lock.
    {
        Importer importer;
        lib.models = importer.LoadHDF(xs_path);
    }

    // Load unified depletion matrices from CSV if not already loaded from HDF5.
    // EnsureInitialized replaces the bare `if (depDecay.size() == 0)` check:
    // that was a check-then-act on a process global, so two instances starting
    // together could both enter and assign the matrices concurrently.
    {
#ifdef DATA_DIR
        const auto chainDir = std::filesystem::path(DATA_DIR) / "include" / "Database";
#else
        const auto chainDir = std::filesystem::path("include") / "Database";
#endif
        if (std::filesystem::exists(chainDir / "dep_decay.csv"))
            Isotope::EnsureInitialized(chainDir);
    }

    lib.niso          = Isotope::niso;
    const size_t niso = lib.niso;
    const size_t ng   = static_cast<size_t>(lib.ng);

    const size_t nmodels = lib.models.size();

    // Count totals across all models
    size_t total_dpts  = 0;
    size_t total_delta = 0;
    size_t total_coeff = 0;
    for (const auto& m : lib.models) {
        total_dpts += countReferenceSlots(m._refr_dpts);
        total_delta += countDeltaSlots(m._bppm_delt);
        total_delta += countDeltaSlots(m._tful_delt);
        total_delta += countDeltaSlots(m._dmod_delt);
        total_coeff += countDeltaCoefficients(m._bppm_delt);
        total_coeff += countDeltaCoefficients(m._tful_delt);
        total_coeff += countDeltaCoefficients(m._dmod_delt);
        for (const auto& correction : m._spectral_history) {
            total_delta += correction.delta.size();
            total_coeff += countDeltaCoefficients(correction.delta);
        }
    }

    // Allocate flat arrays
    lib.lib_lmpx.allocate(total_dpts * ng, total_dpts * ng * ng);
    lib.lib_micx.allocate(total_dpts * niso * ng, total_dpts * niso * ng * ng);
    lib.lib_iden.assign(total_dpts * niso, 0.0);
    lib.lib_burn.assign(total_dpts, 0.0);
    lib.lib_wvfr.assign(total_dpts, 0.0);
    lib.lib_ref_branch_x.assign(total_dpts, {});
    lib.lib_flux.assign(total_dpts * ng, 0.0);
    lib.lib_chix.assign(total_dpts * ng, 0.0);
    lib.lib_coeff_lmpx.allocate(total_coeff * ng, total_coeff * ng * ng);
    lib.lib_coeff_micx.allocate(total_coeff * niso * ng, total_coeff * niso * ng * ng);
    lib.has_coeff_micx = false;
    lib.lib_deltas.resize(total_delta);
    lib.lib_knots.clear();
    lib.refr_base.resize(nmodels, 0);
    lib.refr_ctyp_stride.resize(nmodels, 0);
    lib.refr_burn_stride.resize(nmodels, 1);
    lib.refr_ctyp.resize(nmodels);
    lib.refr_burn.resize(nmodels);
    lib.brch_base.assign(
        nmodels, std::vector<size_t>(NUM_SCALAR_BRANCHES, 0));
    lib.brch_ctyp_stride.assign(
        nmodels, std::vector<size_t>(NUM_SCALAR_BRANCHES, 0));
    lib.brch_burn_stride.assign(
        nmodels, std::vector<size_t>(NUM_SCALAR_BRANCHES, 1));
    lib.brch_ctyp.assign(
        nmodels,
        std::vector<std::vector<int>>(NUM_SCALAR_BRANCHES));
    lib.brch_burn.assign(
        nmodels,
        std::vector<std::vector<std::vector<int>>>(
            NUM_SCALAR_BRANCHES));
    lib.lib_model_volu.resize(nmodels, 1.0);
    lib.lib_model_hmas.resize(nmodels, 1.0);
    lib.lib_spectral_history.assign(nmodels, {});

    // Flatten reference depletion points in model/ctype/burn stride order.
    size_t dpt_idx = 0;
    for (size_t mi = 0; mi < nmodels; ++mi) {
        const auto& m    = lib.models[mi];
        const auto& dpts = m.Dpts();

        lib.refr_base[mi] = dpt_idx;
        if (m._refr_dpts.empty())
            continue;
        lib.refr_ctyp_stride[mi] = maxBurnCount(m._refr_dpts);
        lib.refr_burn_stride[mi] = 1;
        lib.refr_ctyp[mi]        = ctypeKeys(m._refr_dpts);
        lib.refr_burn[mi].resize(lib.refr_ctyp[mi].size());

        // Per-model constants from dpt(0, first burn), used by UpdateBurnup.
        if (!m._refr_dpts.empty() && m._refr_dpts.count(0)) {
            auto         it            = m._refr_dpts.at(0).begin();
            const auto&  dpt0          = dpts[it->second];
            const double assembly_area = dpt0._data[AD_APIT] * dpt0._data[AD_APIT];
            lib.lib_model_volu[mi]        = (assembly_area > 0.0) ? assembly_area : dpt0._data[AD_VOLU];
            lib.lib_model_hmas[mi]        = dpt0._data[AD_HMAS];
        }

        for (size_t ci = 0; ci < lib.refr_ctyp[mi].size(); ++ci) {
            const int   ctype = lib.refr_ctyp[mi][ci];
            const auto& bmap  = m._refr_dpts.at(ctype);
            auto&       keys  = lib.refr_burn[mi][ci];
            keys              = burnKeys(bmap);
            for (size_t bi = 0; bi < keys.size(); ++bi) {
                const int    burn_key = keys[bi];
                const size_t flat     = lib.refr_base[mi] + ci * lib.refr_ctyp_stride[mi] +
                                    bi * lib.refr_burn_stride[mi];
                FlattenReferenceCrossSection(lib, flat, dpts[bmap.at(burn_key)]);
            }
        }

        dpt_idx += lib.refr_ctyp[mi].size() * lib.refr_ctyp_stride[mi];
    }

    // Flatten branch delta coefficients.
    size_t delta_slot_idx = 0;
    size_t coeff_idx      = 0;
    size_t knot_offset    = 0;

    for (size_t mi = 0; mi < nmodels; ++mi) {
        FlattenBranchDelta(lib, lib.models[mi]._bppm_delt, mi, BRANCH_BPPM, delta_slot_idx, coeff_idx, knot_offset);
        FlattenBranchDelta(lib, lib.models[mi]._tful_delt, mi, BRANCH_TFUL, delta_slot_idx, coeff_idx, knot_offset);
        FlattenBranchDelta(lib, lib.models[mi]._dmod_delt, mi, BRANCH_DMOD, delta_slot_idx, coeff_idx, knot_offset);
        auto& history = lib.lib_spectral_history[mi];
        history.resize(lib.models[mi]._spectral_history.size());
        for (size_t h = 0;
             h < lib.models[mi]._spectral_history.size(); ++h) {
            FlattenSpectralHistory(
                lib, lib.models[mi]._spectral_history[h], history[h],
                delta_slot_idx, coeff_idx, knot_offset);
        }
    }

    // The N_eff axis reads sigma_a,Gd by inverting N_table(Bu), so that trajectory has to
    // be strictly decreasing or the inverse map does not exist.  Assert it here, on load,
    // for every reference row -- not at the point of use.  PrecomputeBranchCoefficients
    // can only compare the two endpoints of a row it is already walking, and when that
    // test fails it has nothing to do but keep the burnup bracket: the node would then
    // read sigma at a burnup with no relation to its Gd inventory, quietly.  A library
    // whose lumped-Gd trajectory is not invertible has to be rejected, loudly, once.
    //
    // Rows with N_eff == 0 throughout carry no Gd (reflectors, non-BP fuels); zero is not
    // strictly decreasing, and the axis never indexes them, so they are exempt.
    for (size_t mi = 0; mi < nmodels; ++mi) {
        for (size_t ci = 0; ci < lib.refr_ctyp[mi].size(); ++ci) {
            const auto&         keys = lib.refr_burn[mi][ci];
            std::vector<double> n_table(keys.size(), 0.0);
            for (size_t bi = 0; bi < keys.size(); ++bi) {
                const size_t flat = lib.refr_base[mi] + ci * lib.refr_ctyp_stride[mi] +
                                    bi * lib.refr_burn_stride[mi];
                n_table[bi] = lib.lib_iden[flat * niso + Isotope::iGd];
            }
            if (std::all_of(n_table.begin(), n_table.end(),
                            [](double v) { return v == 0.0; }))
                continue;
            for (size_t bi = 1; bi < n_table.size(); ++bi)
                if (!(n_table[bi] < n_table[bi - 1]))
                    throw std::runtime_error(std::format(
                        "model '{}' (ctype {}): lumped-Gd N_eff(Bu) is not strictly "
                        "decreasing at burnup {} (index {}): {:.6e} -> {:.6e} -- the "
                        "N_eff axis inverse map is invalid for this library.",
                        lib.models[mi].name(), lib.refr_ctyp[mi][ci], keys[bi], bi,
                        n_table[bi - 1], n_table[bi]));
        }
    }

    lib.lib_history_partner.assign(lib.models.size(), -1);
    for (size_t mi = 0; mi < lib.models.size(); ++mi) {
        const int p = lib.models[mi].history_partner();
        if (p >= 0 && static_cast<size_t>(p) < lib.models.size() &&
            static_cast<size_t>(p) != mi)
            lib.lib_history_partner[mi] = p;
    }
    // Approximate resident footprint, for the receipt.
    {
        const auto set_bytes = [](const XSArraySet& s) {
            size_t n = s.xssm.size();
            for (const auto* v : {&s.xstf, &s.xsdf, &s.xsaf, &s.xsff, &s.xsnf, &s.xskf,
                                  &s.xssf, &s.xsrf, &s.fyld, &s.xs2n, &s.xs3n})
                n += v->size();
            return n * sizeof(double);
        };
        lib.bytes = set_bytes(lib.lib_lmpx) + set_bytes(lib.lib_micx) +
                    set_bytes(lib.lib_coeff_lmpx) + set_bytes(lib.lib_coeff_micx) +
                    lib.lib_iden.size() * sizeof(double) +
                    lib.lib_burn.size() * sizeof(double) +
                    lib.lib_wvfr.size() * sizeof(double) +
                    lib.lib_flux.size() * sizeof(double) +
                    lib.lib_chix.size() * sizeof(double) +
                    lib.lib_knots.size() * sizeof(double) +
                    lib.lib_deltas.size() * sizeof(DeltaInfo) +
                    lib.lib_ref_branch_x.size() * sizeof(std::array<double, 3>);
    }

    return owned;
}

namespace {

struct XsLibraryCacheEntry {
    std::string                      path;
    std::uint64_t                    file_size = 0;
    std::int64_t                     mtime     = 0;
    int                              ng        = 0;
    std::shared_ptr<const XsLibrary> value;
    bool                             building = false; ///< one worker is parsing this key
};

// Deliberately a plain mutex and a small vector: every run this campaign has
// names one library, and a miss costs a 34 MB HDF5 parse.  What matters is that
// this lock is NOT Chiffon::hdf5Mutex -- a hit must not queue behind another
// worker's parse, which is the whole point of the cache.
std::mutex& XsLibraryCacheMutex() {
    static std::mutex m;
    return m;
}

std::vector<XsLibraryCacheEntry>& XsLibraryCacheEntries() {
    static std::vector<XsLibraryCacheEntry> entries;
    return entries;
}

std::condition_variable& XsLibraryCacheReady() {
    static std::condition_variable cv;
    return cv;
}

std::atomic<std::uint64_t> g_xslib_loads{0};
std::atomic<std::uint64_t> g_xslib_hits{0};
std::atomic<std::uint64_t> g_xslib_waits{0};
std::atomic<std::uint64_t> g_xslib_lock_wait_ns{0};

struct XsLibraryKeyFields {
    std::string   path;
    std::uint64_t file_size = 0;
    std::int64_t  mtime     = 0;
};

XsLibraryKeyFields XsLibraryKeyOf(const std::string& xs_path) {
    XsLibraryKeyFields key;
    std::error_code    ec;
    const auto canonical = std::filesystem::weakly_canonical(std::filesystem::path(xs_path), ec);
    key.path             = ec ? xs_path : canonical.lexically_normal().string();
    ec.clear();
    const auto size = std::filesystem::file_size(std::filesystem::path(xs_path), ec);
    key.file_size   = ec ? 0 : static_cast<std::uint64_t>(size);
    ec.clear();
    const auto stamp = std::filesystem::last_write_time(std::filesystem::path(xs_path), ec);
    key.mtime        = ec ? 0 : static_cast<std::int64_t>(stamp.time_since_epoch().count());
    return key;
}

bool XsLibraryCacheDisabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("RASBERY_XSLIB_CACHE");
        return v && *v && std::string(v) == "0";
    }();
    return disabled;
}

} // namespace

std::shared_ptr<const XsLibrary> AcquireXsLibrary(const std::string& xs_path, int ng) {
    if (XsLibraryCacheDisabled())
        return BuildXsLibrary(xs_path, ng);

    const XsLibraryKeyFields key = XsLibraryKeyOf(xs_path);
    const auto               find = [&key, ng]() -> XsLibraryCacheEntry* {
        for (auto& e : XsLibraryCacheEntries())
            if (e.ng == ng && e.file_size == key.file_size && e.mtime == key.mtime &&
                e.path == key.path)
                return &e;
        return nullptr;
    };

    // SINGLE FLIGHT, and the parse is NOT under the cache mutex.
    //
    // Two failure modes had to be avoided at once, and only this shape avoids
    // both.  Parsing while holding the mutex rebuilds the queue this cache
    // exists to remove, one level down: a HIT would wait behind somebody else's
    // 34 MB parse, and the Init+IO staircase would be unchanged.  Letting every
    // miss parse instead removes that -- and replaces it with M concurrent
    // parses on a cold M-wide wave: at M=64 that is 64 simultaneous 34 MB
    // reads, ~2 GB of transient copies and 2.7x CPU oversubscription on the
    // 24-core host, to produce 64 identical results.  Measured at width 4 it
    // reported loads=4.
    //
    // So: exactly one worker builds a given key, the others WAIT ON THIS KEY
    // (not on the mutex, and not on the HDF5 lock), and a hit takes the mutex
    // for the length of a vector scan.  Workers that want a different library,
    // or a library already built, are never delayed by a build at all.
    const auto                   wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(XsLibraryCacheMutex());
    while (true) {
        XsLibraryCacheEntry* entry = find();
        if (entry == nullptr) {
            XsLibraryCacheEntries().push_back(
                XsLibraryCacheEntry{key.path, key.file_size, key.mtime, ng, nullptr, true});
            break; // this thread builds it
        }
        if (entry->value != nullptr) {
            g_xslib_hits.fetch_add(1, std::memory_order_relaxed);
            g_xslib_lock_wait_ns.fetch_add(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wait_start)
                        .count()),
                std::memory_order_relaxed);
            return entry->value;
        }
        // Being built by somebody else: wait for THAT key, then re-check.  The
        // re-check is what makes a failed build (which erases its entry)
        // recoverable rather than a permanent hang.
        g_xslib_waits.fetch_add(1, std::memory_order_relaxed);
        XsLibraryCacheReady().wait(lock);
    }
    // Charged here, BEFORE the parse: the counter means "time this acquisition
    // lost to the cache", and the one worker that parses would have parsed
    // anyway.  What is charged is the mutex, and -- for every worker that
    // arrived while somebody else was parsing -- the wait for that one parse.
    g_xslib_lock_wait_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_start)
                .count()),
        std::memory_order_relaxed);
    lock.unlock();

    std::shared_ptr<const XsLibrary> built;
    try {
        built = BuildXsLibrary(xs_path, ng);
    } catch (...) {
        // Erase the placeholder before rethrowing, or every waiter on this key
        // waits for a parse that will never arrive.
        lock.lock();
        auto& entries = XsLibraryCacheEntries();
        for (auto it = entries.begin(); it != entries.end(); ++it)
            if (it->ng == ng && it->file_size == key.file_size && it->mtime == key.mtime &&
                it->path == key.path && it->value == nullptr) {
                entries.erase(it);
                break;
            }
        lock.unlock();
        XsLibraryCacheReady().notify_all();
        throw;
    }
    g_xslib_loads.fetch_add(1, std::memory_order_relaxed);

    lock.lock();
    if (XsLibraryCacheEntry* entry = find()) {
        entry->value    = built;
        entry->building = false;
    }
    lock.unlock();
    XsLibraryCacheReady().notify_all();
    return built;
}

XsLibraryCacheStats XsLibraryCacheSnapshot() {
    XsLibraryCacheStats s;
    s.loads        = g_xslib_loads.load(std::memory_order_relaxed);
    s.hits         = g_xslib_hits.load(std::memory_order_relaxed);
    s.waits        = g_xslib_waits.load(std::memory_order_relaxed);
    s.lock_wait_ms = g_xslib_lock_wait_ns.load(std::memory_order_relaxed) / 1000000ULL;
    std::lock_guard<std::mutex> guard(XsLibraryCacheMutex());
    s.entries = XsLibraryCacheEntries().size();
    for (const auto& e : XsLibraryCacheEntries())
        s.bytes += e.value ? e.value->bytes : 0;
    return s;
}

void PrintXsLibraryCacheReceipt(std::ostream& out) {
    const XsLibraryCacheStats s = XsLibraryCacheSnapshot();
    out << "[RASBERY][XSLIB_CACHE] {\"loads\":" << s.loads << ",\"hits\":" << s.hits
        << ",\"waits\":" << s.waits
        << ",\"entries\":" << s.entries << ",\"bytes\":" << s.bytes
        << ",\"lock_wait_ms\":" << s.lock_wait_ms
        << ",\"enabled\":" << (XsLibraryCacheDisabled() ? "false" : "true") << "}\n";
}

} // namespace rasbery

void XSSet::Initialize(const std::string& xs_path) {
    const int nz    = _g.nz();
    const int nxyz  = _g.nxyz();
    const int nxyza = _g.nxyza();

    // 1. Allocate node-level storage and load external library data.
    _comp.assign(nxyz, 0);
    _asmb.assign(nxyza, 0);
    _ctyp.assign(nxyz, 0);
    _burn.assign(nxyz, 0);
    _external_iden.assign(nxyz, 0);

    _burn_bos.resize(nxyz);
    _flux_bos = milk::Vector<double>(static_cast<size_t>(nxyz * _g.ng()));

    _node_power_scratch.assign(nxyz, 0.0);
    _cum_bot_scratch.assign(nz + 1, 0.0);
    _rod_node_segment_offset.assign(nxyz + 1, 0);
    _rod_node_segment_ctype.clear();
    _rod_node_segment_fraction.clear();
    _fine_rod_type.assign(static_cast<size_t>(_g.nxy()) * static_cast<size_t>(nz) *
                              static_cast<size_t>(_axial_rod_division),
                          0);
    _fine_rod_thermal_fluence.assign(_fine_rod_type.size(), 0.0);
    _fine_rod_thermal_fluence_bos.assign(_fine_rod_type.size(), 0.0);

    // Load T/H property tables
    LoadTHTables();

    // Acquire the Chiffon XS library.  Parsed and flattened at most once per
    // (file, ng) per process; every later deck naming the same library takes a
    // shared_ptr to the same immutable parse and never enters HDF5 at all.
    // See XsLibrary.h for why that is sound and what stays per-Driver.
    _lib = AcquireXsLibrary(xs_path, _g.ng());

    const size_t niso = _lib->niso;
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
    _node_wvfr.assign(nxyz, 0.0);
    _ref_chix.assign(static_cast<size_t>(ng) * nxyz, 0.0);
    _simd_ready = false;

    // Beginning-of-step snapshots for predictor-corrector depletion.
    _xs_bos.allocate(ngn, 0);
    _micx_bos.allocate(micn, 0);
    _iden_bos.assign(niso * nxyz, 0.0);

    // 2. Build model-name lookup and assign each node to its XS model.
    std::map<std::string, size_t> modelIndexMap;
    size_t                        modelId = 0;
    for (const auto& model : _lib->models)
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

                if (lka < 0 || lka >= _g.nxyza())
                    throw std::runtime_error(std::format(
                        "XSSet: assembly index out of range (lka={}, core row={}, col={})", lka, row, col));

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

    // 3. Bracket every node against the shared flatten.
    //
    // Everything the library file itself determines -- the SoA flatten, the
    // per-model reference/branch index tables, the N_eff monotonicity check and
    // the history-partner map -- was done once, in BuildXsLibrary.  What is left
    // here is per-deck: arrays sized by nxyz, and the checks that read this
    // deck's own _comp.
    {
        _node_refr_lo.assign(nxyz, -1);
        _node_refr_hi.assign(nxyz, -1);
        _node_delta_lo.assign(
            NUM_SCALAR_BRANCHES, std::vector<int>(nxyz, -1));
        _node_delta_hi.assign(
            NUM_SCALAR_BRANCHES, std::vector<int>(nxyz, -1));
        _node_delta_frac.assign(
            NUM_SCALAR_BRANCHES, std::vector<double>(nxyz, 0.0));

        // Gd lookup axis selection: burnup (default, historical) or effective Gd number density
        // (MASTER/PROLOG `BP01 MICRO` + IOPT(1)=1 equivalent).
        _gd_neff_axis = false;
        if (const char* env = std::getenv("RASBERY_GD_AXIS")) {
            std::string mode(env);
            std::transform(mode.begin(), mode.end(), mode.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            _gd_neff_axis = (mode == "neff" || mode == "nd" || mode == "1");
        }

        _node_gd_lo.assign(nxyz, -1);
        _node_gd_hi.assign(nxyz, -1);
        _node_gd_frac.assign(nxyz, 0.0);

        _node_hw.assign(nxyz, 0.0);
        _node_refr_lo_p.assign(nxyz, -1);
        _node_refr_hi_p.assign(nxyz, -1);
        _node_delta_lo_p.assign(
            NUM_SCALAR_BRANCHES, std::vector<int>(nxyz, -1));
        _node_delta_hi_p.assign(
            NUM_SCALAR_BRANCHES, std::vector<int>(nxyz, -1));
        _node_delta_frac_p.assign(
            NUM_SCALAR_BRANCHES, std::vector<double>(nxyz, 0.0));

        // Boundary validation: every model referenced by a node must carry a non-empty
        // unrodded reference burn table (PrecomputeBranchCoefficients relies on this).
        for (int l = 0; l < nxyz; ++l) {
            const size_t mi = _comp[l];
            const int    ci = findCtype(_lib->refr_ctyp[mi], 0);
            if (ci < 0 || _lib->refr_burn[mi][ci].empty())
                throw std::runtime_error("XSSet: model lacks an unrodded reference depletion table");
        }

        RebuildUsesRodCache();
    }
}

// Reconstruct: rmcx = lmpx + sum(micx_i * iden_i), then derive D and removal.
void XSSet::Reconstruct() {
    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    // Every step is element-wise in l, so slicing the node axis across threads keeps each
    // element's accumulation order identical to the serial loop (bit-identical), while the
    // ~20 MB streaming pass no longer runs on one thread with the others parked.
    auto reconstructSlice = [&](int ls, int le) {
        const size_t len = static_cast<size_t>(le - ls);

        // 1. Rebuild scalar macroscopic XS from lumped and microscopic terms.
        for (int xt = XSTF; xt <= XS3N; ++xt) {
            if (xt == XSDF || xt == XSRF) continue;
            auto        xtype = static_cast<XSTYPE>(xt);
            auto&       dst   = _xs[xtype];
            const auto& lmp   = _lmpx[xtype];
            const auto& mic   = _micx[xtype];

            for (int ig = 0; ig < ng; ++ig)
                CopyDoubles(len, lmp.data() + ig * nxyz + ls, dst.data() + ig * nxyz + ls);

            // dst += Σ_iso (micx[iso] * iden[iso]).
            // iden varies by node, so each isotope/group block is an element-wise stride-1 loop.
            for (size_t iso = 0; iso < niso; ++iso) {
                for (int ig = 0; ig < ng; ++ig) {
                    const double* mic_ptr  = mic.data() + (iso * ng + ig) * nxyz + ls;
                    const double* iden_ptr = _iden.data() + iso * nxyz + ls;
                    double*       dst_ptr  = dst.data() + ig * nxyz + ls;

#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
                    for (size_t l = 0; l < len; ++l)
                        dst_ptr[l] += mic_ptr[l] * iden_ptr[l];
                }
            }
        }

        // 2. Rebuild the scattering matrix with the same SoA accumulation pattern.
        {
            auto&       dst = _xs.xssm;
            const auto& lmp = _lmpx.xssm;
            const auto& mic = _micx.xssm;
            for (int sm = 0; sm < ng * ng; ++sm)
                CopyDoubles(len, lmp.data() + sm * nxyz + ls, dst.data() + sm * nxyz + ls);

            for (size_t iso = 0; iso < niso; ++iso) {
                for (int igs = 0; igs < ng; ++igs) {
                    for (int ige = 0; ige < ng; ++ige) {
                        const double* mic_ptr  = mic.data() + (iso * ng * ng + igs * ng + ige) * nxyz + ls;
                        const double* iden_ptr = _iden.data() + iso * nxyz + ls;
                        double*       dst_ptr  = dst.data() + (igs * ng + ige) * nxyz + ls;

#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
                        for (size_t l = 0; l < len; ++l)
                            dst_ptr[l] += mic_ptr[l] * iden_ptr[l];
                    }
                }
            }
        }

        // 3. Derive diffusion coefficients from the transport cross section.
        for (int ig = 0; ig < ng; ++ig) {
            double* xstf_ptr = _xs.xstf.data() + ig * nxyz + ls;
            double* xsdf_ptr = _xs.xsdf.data() + ig * nxyz + ls;
#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
            for (size_t l = 0; l < len; ++l) {
                double tr   = xstf_ptr[l];
                xsdf_ptr[l] = (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;
            }
        }

        // 4. Recompute removal as absorption plus outgoing scattering.
        for (int igs = 0; igs < ng; ++igs) {
            double*       xsrf_ptr = _xs.xsrf.data() + igs * nxyz + ls;
            const double* xsaf_ptr = _xs.xsaf.data() + igs * nxyz + ls;
#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
            for (size_t l = 0; l < len; ++l)
                xsrf_ptr[l] = xsaf_ptr[l];

            for (int ige = 0; ige < ng; ++ige) {
                const double* sm_ptr = _xs.xssm.data() + (igs * ng + ige) * nxyz + ls;
#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
                for (size_t l = 0; l < len; ++l)
                    xsrf_ptr[l] += sm_ptr[l];
            }
        }
    };

    constexpr int RECON_BLOCK = 512;
    const int     nblk        = (nxyz + RECON_BLOCK - 1) / RECON_BLOCK;
#pragma omp parallel for schedule(static) if (nxyz > OMP_THRESHOLD)
    for (int b = 0; b < nblk; ++b)
        reconstructSlice(b * RECON_BLOCK, std::min(nxyz, (b + 1) * RECON_BLOCK));
    // Steps 3 and 4 above assigned _xs.xsdf and _xs.xsrf.  Said HERE, once,
    // rather than inside the slice lambda: the bump is per CALL, and the
    // reader (Nodal::updateConstant's gate) only ever compares equality.
    noteMacroXsWrite();
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
    noteMacroXsWrite();
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

        const auto& model = _lib->models[_comp[l]];

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
    ++_micx_generation; // full rebuild; see the note in UpdateFlatXS
    ++_hoststate_generation;
}

// Pre-compute node lookup and burnup-interpolated reference XS.

double XSSet::ReferenceIden(size_t mi, int ctype, int burn, size_t iso) const {
    if (iso >= Isotope::niso || mi >= _lib->refr_ctyp.size())
        return 0.0;
    const int ci = findCtype(_lib->refr_ctyp[mi], ctype);
    if (ci < 0)
        return 0.0;
    const auto& burns = _lib->refr_burn[mi][ci];
    const int   lo    = findLoBurn(burns, burn);
    const int   hi    = findHiBurn(burns, burn);
    if (lo < 0 || hi < 0)
        return 0.0;
    const size_t base = _lib->refr_base[mi] + static_cast<size_t>(ci) * _lib->refr_ctyp_stride[mi];
    const size_t lb   = base + static_cast<size_t>(lo) * _lib->refr_burn_stride[mi];
    const size_t hb   = base + static_cast<size_t>(hi) * _lib->refr_burn_stride[mi];
    double       v    = _lib->lib_iden[lb * Isotope::niso + iso];
    if (lb != hb && _lib->lib_burn[hb] != _lib->lib_burn[lb]) {
        const double f = (static_cast<double>(burn) / 1000.0 - _lib->lib_burn[lb]) /
                         (_lib->lib_burn[hb] - _lib->lib_burn[lb]);
        v += f * (_lib->lib_iden[hb * Isotope::niso + iso] - v);
    }
    return v;
}

void XSSet::BuildHistoryBlend(int l, size_t mi) {
    _node_hw[l]        = 0.0;
    _node_refr_lo_p[l] = -1;
    _node_refr_hi_p[l] = -1;
    for (size_t b = 0; b < NUM_SCALAR_BRANCHES; ++b) {
        _node_delta_lo_p[b][l]   = -1;
        _node_delta_hi_p[b][l]   = -1;
        _node_delta_frac_p[b][l] = 0.0;
    }
    const int partner = mi < _lib->lib_history_partner.size() ? _lib->lib_history_partner[mi] : -1;
    if (partner < 0)
        return;
    const size_t pi = static_cast<size_t>(partner);

    // PROBE: pin the weight to a constant so the required w can be recovered by
    // scanning k(w) against the reference. Negative (default) = measure it.
    static const double fixw = [] {
        const char* env = std::getenv("RASBERY_HB_FIXW");
        return env != nullptr ? std::atof(env) : -1.0;
    }();

    const int refr_ci = findCtype(_lib->refr_ctyp[pi], 0);
    if (refr_ci < 0 || _lib->refr_burn[pi][refr_ci].empty())
        return;
    const auto& refr_burn  = _lib->refr_burn[pi][refr_ci];
    const int   refr_lo_bi = findLoBurn(refr_burn, _burn[l]);
    const int   refr_hi_bi = findHiBurn(refr_burn, _burn[l]);
    if (refr_lo_bi < 0 || refr_hi_bi < 0)
        return;
    const size_t refr_base = _lib->refr_base[pi] + static_cast<size_t>(refr_ci) * _lib->refr_ctyp_stride[pi];
    _node_refr_lo_p[l] =
        static_cast<int>(refr_base + static_cast<size_t>(refr_lo_bi) * _lib->refr_burn_stride[pi]);
    _node_refr_hi_p[l] =
        static_cast<int>(refr_base + static_cast<size_t>(refr_hi_bi) * _lib->refr_burn_stride[pi]);

    for (size_t b = 0; b < NUM_SCALAR_BRANCHES; ++b) {
        const int brch_ci = findCtype(_lib->brch_ctyp[pi][b], 0);
        if (brch_ci < 0)
            continue;
        const auto& brch_burn  = _lib->brch_burn[pi][b][brch_ci];
        const int   brch_lo_bi = findLoBurn(brch_burn, _burn[l]);
        const int   brch_hi_bi = findHiBurn(brch_burn, _burn[l]);
        if (brch_lo_bi < 0 || brch_hi_bi < 0)
            continue;
        const size_t brch_base = _lib->brch_base[pi][b] +
                                 static_cast<size_t>(brch_ci) * _lib->brch_ctyp_stride[pi][b];
        _node_delta_lo_p[b][l] =
            static_cast<int>(brch_base + static_cast<size_t>(brch_lo_bi) * _lib->brch_burn_stride[pi][b]);
        _node_delta_hi_p[b][l] =
            static_cast<int>(brch_base + static_cast<size_t>(brch_hi_bi) * _lib->brch_burn_stride[pi][b]);
        _node_delta_frac_p[b][l] =
            (brch_lo_bi == brch_hi_bi)
                ? 0.0
                : static_cast<double>(_burn[l] - brch_burn[brch_lo_bi]) /
                      static_cast<double>(brch_burn[brch_hi_bi] - brch_burn[brch_lo_bi]);
    }

    // Weight: the node's U235-vs-Pu239 balance, measured from the unrodded
    // reference and normalized by the partner's rod-in reference at the same
    // burnup. Both endpoints come from the library, so nothing extra is carried
    // across a restart -- the fuel's own inventory is the clock.
    if (fixw >= 0.0) {
        _node_hw[l] = std::clamp(fixw, 0.0, 1.0);
        return;
    }

    // Every endpoint is guarded for finiteness: a NaN would slip through the
    // ordinary comparisons and reach applyDelta as a NaN scale.
    const double fl = Chiffon::SPECTRAL_LOG_DENSITY_FLOOR;
    const double ru = ReferenceIden(mi, 0, _burn[l], Isotope::iU235);
    const double rp = ReferenceIden(mi, 0, _burn[l], Isotope::iPu239);
    const double su = ReferenceIden(pi, 1, _burn[l], Isotope::iU235);
    const double sp = ReferenceIden(pi, 1, _burn[l], Isotope::iPu239);
    if (!std::isfinite(ru) || !std::isfinite(rp) || !std::isfinite(su) ||
        !std::isfinite(sp) || ru <= fl || rp <= fl || su <= fl || sp <= fl)
        return;
    const double span = std::log(su / ru) - std::log(sp / rp);
    if (!std::isfinite(span) || std::abs(span) < 1.0e-6)
        return;

    const size_t stride  = static_cast<size_t>(_g.nxyz());
    const size_t node    = static_cast<size_t>(l);
    const double raw_nu  = _iden[Isotope::iU235 * stride + node];
    const double raw_np  = _iden[Isotope::iPu239 * stride + node];
    if (!std::isfinite(raw_nu) || !std::isfinite(raw_np))
        return;
    const double rr    = std::log(std::max(raw_nu, fl) / ru) -
                      std::log(std::max(raw_np, fl) / rp);
    const double ratio = rr / span;
    if (!std::isfinite(ratio))
        return;
    // PROBE: shape of the approach to the rodded library. 1.0 is the linear
    // reading of the composition coordinate; >1 delays it, <1 hastens it.
    static const double gamma = [] {
        const char* env = std::getenv("RASBERY_HB_GAMMA");
        const double v = env != nullptr ? std::atof(env) : 1.0;
        return (v > 0.05 && v < 20.0) ? v : 1.0;
    }();
    const double clamped = std::clamp(ratio, 0.0, 1.0);
    _node_hw[l] = gamma == 1.0 ? clamped : std::pow(clamped, gamma);
}

void XSSet::PrecomputeBranchCoefficients() {
    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    // 1. Build the per-node index table used by the hot update loop.
    // The unrodded reference tables are validated once at Initialize, so the
    // ctype/burn lookups here cannot fail.
    for (int l = 0; l < nxyz; ++l) {
        const size_t mi     = _comp[l];
        const int    eff_ct = 0;

        // Reference depletion points (lo/hi burnup bracket)
        const int   refr_ci    = findCtype(_lib->refr_ctyp[mi], eff_ct);
        const auto& refr_burn  = _lib->refr_burn[mi][refr_ci];
        const int   refr_lo_bi = findLoBurn(refr_burn, _burn[l]);
        const int   refr_hi_bi = findHiBurn(refr_burn, _burn[l]);
        _node_refr_lo[l]       = static_cast<int>(_lib->refr_base[mi] +
                                                  static_cast<size_t>(refr_ci) * _lib->refr_ctyp_stride[mi] +
                                                  static_cast<size_t>(refr_lo_bi) * _lib->refr_burn_stride[mi]);
        _node_refr_hi[l]       = static_cast<int>(_lib->refr_base[mi] +
                                                  static_cast<size_t>(refr_ci) * _lib->refr_ctyp_stride[mi] +
                                                  static_cast<size_t>(refr_hi_bi) * _lib->refr_burn_stride[mi]);

        // Delta polynomial entries per branch
        for (size_t b = 0; b < NUM_SCALAR_BRANCHES; ++b) {
            const int brch_ci = findCtype(_lib->brch_ctyp[mi][b], eff_ct);
            if (brch_ci < 0) {
                _node_delta_lo[b][l]   = -1;
                _node_delta_hi[b][l]   = -1;
                _node_delta_frac[b][l] = 0.0;
                continue;
            }
            const auto& brch_burn  = _lib->brch_burn[mi][b][brch_ci];
            const int   brch_lo_bi = findLoBurn(brch_burn, _burn[l]);
            const int   brch_hi_bi = findHiBurn(brch_burn, _burn[l]);
            if (brch_lo_bi < 0 || brch_hi_bi < 0) {
                _node_delta_lo[b][l]   = -1;
                _node_delta_hi[b][l]   = -1;
                _node_delta_frac[b][l] = 0.0;
                continue;
            }

            _node_delta_lo[b][l] = static_cast<int>(_lib->brch_base[mi][b] +
                                                    static_cast<size_t>(brch_ci) * _lib->brch_ctyp_stride[mi][b] +
                                                    static_cast<size_t>(brch_lo_bi) * _lib->brch_burn_stride[mi][b]);
            _node_delta_hi[b][l] = static_cast<int>(_lib->brch_base[mi][b] +
                                                    static_cast<size_t>(brch_ci) * _lib->brch_ctyp_stride[mi][b] +
                                                    static_cast<size_t>(brch_hi_bi) * _lib->brch_burn_stride[mi][b]);
            _node_delta_frac[b][l] =
                (brch_lo_bi == brch_hi_bi)
                    ? 0.0
                    : static_cast<double>(_burn[l] - brch_burn[brch_lo_bi]) /
                          static_cast<double>(brch_burn[brch_hi_bi] - brch_burn[brch_lo_bi]);
        }

        // Gd effective number density axis: locate the bracket on the library N_eff trajectory.
        if (_gd_neff_axis) {
            _node_gd_lo[l]   = _node_refr_lo[l];
            _node_gd_hi[l]   = _node_refr_hi[l];
            _node_gd_frac[l] = 0.0;

            const size_t nbp   = refr_burn.size();
            const double n_now = _iden[Isotope::iGd * nxyz + l];
            if (nbp >= 2 && n_now > 0.0) {
                auto flat_id = [&](size_t bi) {
                    return _lib->refr_base[mi] + static_cast<size_t>(refr_ci) * _lib->refr_ctyp_stride[mi] +
                           bi * _lib->refr_burn_stride[mi];
                };
                auto neff_at = [&](size_t bi) { return _lib->lib_iden[flat_id(bi) * niso + Isotope::iGd]; };

                // n0 > 0 says this model's reference row carries Gd at all -- an applicability
                // test, not a failure path.  For any such row Initialize() has already asserted
                // strict decrease, so the bracket below always exists and na > nb.  There is no
                // fallback here, because a library that could need one never gets this far.
                const double n0 = neff_at(0);
                const double nl = neff_at(nbp - 1);
                if (n0 > 0.0) {
                    size_t bi = 0;
                    if (n_now >= n0) {
                        bi = 0; // fresher than the table start -> clamp to the first interval
                    } else if (n_now <= nl) {
                        bi = nbp - 2; // more depleted than the table end -> clamp to the last interval
                    } else {
                        // n_now is bracketed; N_eff is monotonically decreasing with burnup.
                        size_t lo = 0, hi = nbp - 1;
                        while (hi - lo > 1) {
                            const size_t mid = (lo + hi) / 2;
                            if (neff_at(mid) > n_now)
                                lo = mid;
                            else
                                hi = mid;
                        }
                        bi = lo;
                    }
                    const double na = neff_at(bi);
                    const double nb = neff_at(bi + 1);
                    // Clamp instead of extrapolating (same convention as the burnup axis).
                    const double f = std::min(1.0, std::max(0.0, (na - n_now) / (na - nb)));
                    _node_gd_lo[l]   = static_cast<int>(flat_id(bi));
                    _node_gd_hi[l]   = static_cast<int>(flat_id(bi + 1));
                    _node_gd_frac[l] = f;
                }
            }
        }

        BuildHistoryBlend(l, mi);
    }

    // 2. Scatter reference data from the flat library cache into node SoA buffers.

    _ref_lmpx.fill(0.0);
    _ref_micx.fill(0.0);
    _ref_iden.fill(0.0);

    // Cache XSArraySet data pointers to avoid switch dispatch inside the loop
    const auto    lib_lmpx_ptrs = ScalarData(_lib->lib_lmpx);
    const auto    lib_micx_ptrs = ScalarData(_lib->lib_micx);
    const double* lib_lmpx_sm   = _lib->lib_lmpx.xssm.data();
    const double* lib_micx_sm   = _lib->lib_micx.xssm.data();
    const double* lib_iden      = _lib->lib_iden.data();
    const double* lib_burn      = _lib->lib_burn.data();
    const double* lib_wvfr      = _lib->lib_wvfr.data();
    const double* lib_chix      = _lib->lib_chix.data();
    auto          ref_lmpx      = ScalarXS(_ref_lmpx);
    auto          ref_micx      = ScalarXS(_ref_micx);

#pragma omp parallel for schedule(static) if (nxyz > OMP_THRESHOLD)
    for (int l = 0; l < nxyz; ++l) {
        const int lo = _node_refr_lo[l];
        const int hi = _node_refr_hi[l];

        // Single fused pass: val = lib[lo] (+ f*(lib[hi]-lib[lo]) between burnup brackets),
        // written once. Same operation sequence as scatter-then-interpolate, kept in a register.
        const bool   interp = (lo != hi);
        const double f      = interp ? (_burn[l] / 1000.0 - lib_burn[lo]) / (lib_burn[hi] - lib_burn[lo]) : 0.0;

        // Depletion-history blend: the same gather on the rodded twin, mixed in
        // by weight. w is 0 (and lo_p < 0) whenever the fuel declares no twin,
        // so the arithmetic below collapses to the single-library case.
        const double w      = _node_hw.empty() ? 0.0 : _node_hw[l];
        const int    lo_p   = _node_refr_lo_p.empty() ? -1 : _node_refr_lo_p[l];
        const int    hi_p   = _node_refr_hi_p.empty() ? -1 : _node_refr_hi_p[l];
        const bool   blend  = (w > 0.0 && lo_p >= 0 && hi_p >= 0);
        const bool   interp_p = blend && (lo_p != hi_p);
        const double span_p = interp_p ? lib_burn[hi_p] - lib_burn[lo_p] : 0.0;
        const double f_p =
            (interp_p && std::abs(span_p) > 1.0e-30)
                ? (_burn[l] / 1000.0 - lib_burn[lo_p]) / span_p
                : 0.0;
        const double wu     = blend ? 1.0 - w : 1.0;

        auto mix = [&](double primary, double partner) {
            return blend ? wu * primary + w * partner : primary;
        };

        // Reference lumped XS from the flat cache.
        for (int ig = 0; ig < ng; ++ig) {
            size_t dst_off = ig * nxyz + l;
            size_t lo_off  = lo * ng + ig;
            size_t hi_off  = hi * ng + ig;
            for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                if (xt == XSDF || xt == XSRF) continue;
                double val = lib_lmpx_ptrs[xt][lo_off];
                if (interp) val += f * (lib_lmpx_ptrs[xt][hi_off] - lib_lmpx_ptrs[xt][lo_off]);
                double valp = 0.0;
                if (blend) {
                    valp = lib_lmpx_ptrs[xt][lo_p * ng + ig];
                    if (interp_p)
                        valp += f_p * (lib_lmpx_ptrs[xt][hi_p * ng + ig] - valp);
                }
                (*ref_lmpx[xt])[dst_off] = mix(val, valp);
            }
            for (int ige = 0; ige < ng; ++ige) {
                double val = lib_lmpx_sm[lo * ng * ng + ig * ng + ige];
                if (interp)
                    val += f * (lib_lmpx_sm[hi * ng * ng + ig * ng + ige] - lib_lmpx_sm[lo * ng * ng + ig * ng + ige]);
                double valp = 0.0;
                if (blend) {
                    valp = lib_lmpx_sm[lo_p * ng * ng + ig * ng + ige];
                    if (interp_p)
                        valp += f_p * (lib_lmpx_sm[hi_p * ng * ng + ig * ng + ige] - valp);
                }
                _ref_lmpx.xssm[(ig * ng + ige) * nxyz + l] = mix(val, valp);
            }
        }

        // Reference microscopic XS from the flat cache.
        for (size_t iso = 0; iso < niso; ++iso) {
            for (int ig = 0; ig < ng; ++ig) {
                size_t dst_off = (iso * ng + ig) * nxyz + l;
                size_t lo_off  = (static_cast<size_t>(lo) * niso + iso) * ng + ig;
                size_t hi_off  = (static_cast<size_t>(hi) * niso + iso) * ng + ig;
                for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                    if (xt == XSDF || xt == XSRF) continue;
                    double val = lib_micx_ptrs[xt][lo_off];
                    if (interp) val += f * (lib_micx_ptrs[xt][hi_off] - lib_micx_ptrs[xt][lo_off]);
                    double valp = 0.0;
                    if (blend) {
                        valp = lib_micx_ptrs[xt][(static_cast<size_t>(lo_p) * niso + iso) * ng + ig];
                        if (interp_p)
                            valp += f_p * (lib_micx_ptrs[xt][(static_cast<size_t>(hi_p) * niso + iso) * ng + ig] - valp);
                    }
                    (*ref_micx[xt])[dst_off] = mix(val, valp);
                }
                for (int ige = 0; ige < ng; ++ige) {
                    const size_t sm_lo = (static_cast<size_t>(lo) * niso + iso) * ng * ng + ig * ng + ige;
                    const size_t sm_hi = (static_cast<size_t>(hi) * niso + iso) * ng * ng + ig * ng + ige;
                    double       val   = lib_micx_sm[sm_lo];
                    if (interp) val += f * (lib_micx_sm[sm_hi] - lib_micx_sm[sm_lo]);
                    double valp = 0.0;
                    if (blend) {
                        const size_t p_lo = (static_cast<size_t>(lo_p) * niso + iso) * ng * ng + ig * ng + ige;
                        const size_t p_hi = (static_cast<size_t>(hi_p) * niso + iso) * ng * ng + ig * ng + ige;
                        valp              = lib_micx_sm[p_lo];
                        if (interp_p) valp += f_p * (lib_micx_sm[p_hi] - valp);
                    }
                    _ref_micx.xssm[(iso * ng * ng + ig * ng + ige) * nxyz + l] = mix(val, valp);
                }
            }
        }

        // Reference isotope densities and water fraction stay on the unrodded
        // trajectory. They define the coordinate the blend weight is measured
        // in, so blending them would feed the weight back into itself.
        for (size_t i = 0; i < niso; ++i) {
            double val = lib_iden[lo * niso + i];
            if (interp) val += f * (lib_iden[hi * niso + i] - lib_iden[lo * niso + i]);
            _ref_iden[i * nxyz + l] = val;
        }

        _ref_wvfr[l]  = lib_wvfr[lo];
        _node_wvfr[l] = _ref_wvfr[l];

        for (int ig = 0; ig < ng; ++ig) {
            double chix_val = lib_chix[lo * ng + ig];
            if (interp)
                chix_val += f * (lib_chix[hi * ng + ig] - lib_chix[lo * ng + ig]);
            double chix_p = 0.0;
            if (blend) {
                chix_p = lib_chix[lo_p * ng + ig];
                if (interp_p) chix_p += f_p * (lib_chix[hi_p * ng + ig] - chix_p);
            }
            _ref_chix[static_cast<size_t>(ig) * nxyz + l] = mix(chix_val, chix_p);
        }

        // Gd axis override: re-evaluate the lumped-Gd microscopic XS on the effective number
        // density axis. Everything else (including the Gd reference density itself, which is only
        // used to initialise fresh nodes) stays on the burnup axis.
        if (_gd_neff_axis && _node_gd_lo[l] >= 0) {
            const int    glo = _node_gd_lo[l];
            const int    ghi = _node_gd_hi[l];
            const double gf  = _node_gd_frac[l];
            const size_t iso = Isotope::iGd;
            for (int ig = 0; ig < ng; ++ig) {
                const size_t dst_off = (iso * ng + ig) * nxyz + l;
                const size_t lo_off  = (static_cast<size_t>(glo) * niso + iso) * ng + ig;
                const size_t hi_off  = (static_cast<size_t>(ghi) * niso + iso) * ng + ig;
                for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                    if (xt == XSDF || xt == XSRF) continue;
                    (*ref_micx[xt])[dst_off] =
                        lib_micx_ptrs[xt][lo_off] + gf * (lib_micx_ptrs[xt][hi_off] - lib_micx_ptrs[xt][lo_off]);
                }
                for (int ige = 0; ige < ng; ++ige) {
                    const size_t sm_lo = (static_cast<size_t>(glo) * niso + iso) * ng * ng + ig * ng + ige;
                    const size_t sm_hi = (static_cast<size_t>(ghi) * niso + iso) * ng * ng + ig * ng + ige;
                    _ref_micx.xssm[(iso * ng * ng + ig * ng + ige) * nxyz + l] =
                        lib_micx_sm[sm_lo] + gf * (lib_micx_sm[sm_hi] - lib_micx_sm[sm_lo]);
                }
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

    // The flat-XS device arm keeps _ref_lmpx/_ref_micx resident; this rebuild
    // is the only writer, so the bump here is the complete invalidation set.
    ++_ref_generation;
    ++_hoststate_generation; // the _iden overwrite above is a host write

    _simd_ready = true;
}

// Flat XS update helpers.

// Rod occupancy only changes in SetRod, so the per-node answer is cached there and the
// hot paths (UpdateFlatXS / history deltas / depletion) read a flat byte instead of
// re-scanning segments and probing the model's reference map per call.
void XSSet::RebuildUsesRodCache() {
    const int nxyz = _g.nxyz();
    _node_uses_rod.assign(static_cast<size_t>(nxyz), 0);
    for (int l = 0; l < nxyz; ++l) {
        if (_g.rod_fraction(l) <= EPS) continue;

        const auto& model = _lib->models[_comp[l]];
        const int   begin = _rod_node_segment_offset[static_cast<size_t>(l)];
        const int   end   = _rod_node_segment_offset[static_cast<size_t>(l + 1)];
        for (int i = begin; i < end; ++i) {
            const int ctype = _rod_node_segment_ctype[static_cast<size_t>(i)];
            if (ctype != 0 && model._refr_dpts.count(ctype) != 0) {
                _node_uses_rod[static_cast<size_t>(l)] = 1;
                break;
            }
        }
    }
}

bool XSSet::UsesRodXS(int l) const {
    return _node_uses_rod[static_cast<size_t>(l)] != 0;
}

void XSSet::ApplyBranchDeltaIdToNode(int l, int did, double x, double scale) {
    if (did < 0 || scale == 0.0) return;

    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    const auto& dinfo = _lib->lib_deltas[did];
    int         base  = dinfo.coeff_base;
    int         nord  = dinfo.nord;

    double xloc = x;

    if (dinfo.mode == 1) {
        const int nintervals = dinfo.nord / dinfo.ncoeff;
        int       interval   = nintervals - 1;
        for (int i = 0; i < nintervals - 1; ++i) {
            if (x < _lib->lib_knots[dinfo.knot_offset + i + 1]) {
                interval = i;
                break;
            }
        }
        xloc = x - _lib->lib_knots[dinfo.knot_offset + interval];
        base += interval * dinfo.ncoeff;
        nord = dinfo.ncoeff;
    }

    const auto    coeff_lmpx    = ScalarData(_lib->lib_coeff_lmpx);
    const auto    coeff_micx    = ScalarData(_lib->lib_coeff_micx);
    const double* coeff_lmpx_sm = _lib->lib_coeff_lmpx.xssm.data();
    const double* coeff_micx_sm = _lib->lib_coeff_micx.xssm.data();
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

    if (!_lib->has_coeff_micx) return;

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

void XSSet::ResolveSpectralHistoryDeltas(
    int l, std::vector<DeltaApplication>& out,
    const double* micWork, size_t requestedModel, double weight) const {
    out.clear();
    if (weight == 0.0)
        return;

    const size_t modelIndex =
        requestedModel == static_cast<size_t>(-1) ? _comp[l] : requestedModel;
    const size_t stride     = static_cast<size_t>(_g.nxyz());
    const size_t node       = static_cast<size_t>(l);
    const int    burn       = _burn[l];
    const int currentCtype  = UsesRodXS(l) ? _ctyp[l] : 0;
    const int ctypeIndex =
        findCtype(_lib->refr_ctyp[modelIndex], currentCtype);
    if (ctypeIndex < 0)
        return;

    const auto& referenceBurnups =
        _lib->refr_burn[modelIndex][ctypeIndex];
    const size_t referenceBase =
        _lib->refr_base[modelIndex] +
        static_cast<size_t>(ctypeIndex) *
            _lib->refr_ctyp_stride[modelIndex];

    const int ctypeIndex0 = findCtype(_lib->refr_ctyp[modelIndex], 0);
    const size_t referenceBase0 =
        _lib->refr_base[modelIndex] +
        static_cast<size_t>(ctypeIndex0 < 0 ? ctypeIndex : ctypeIndex0) *
            _lib->refr_ctyp_stride[modelIndex];

    // Reference inventory on the *unrodded* ctype-0 trajectory, independent of
    // the node's current rod state.
    auto referenceDensity0 = [&](size_t isotope, int burnup) {
        if (isotope >= Isotope::niso)
            return 0.0;
        const auto& burns = _lib->refr_burn[modelIndex][ctypeIndex0 < 0 ? ctypeIndex : ctypeIndex0];
        const int   lo    = findLoBurn(burns, burnup);
        const int   hi    = findHiBurn(burns, burnup);
        if (lo < 0 || hi < 0)
            return 0.0;
        const size_t lb = referenceBase0 + static_cast<size_t>(lo) * _lib->refr_burn_stride[modelIndex];
        const size_t hb = referenceBase0 + static_cast<size_t>(hi) * _lib->refr_burn_stride[modelIndex];
        double       v  = _lib->lib_iden[lb * Isotope::niso + isotope];
        if (lb != hb && _lib->lib_burn[hb] != _lib->lib_burn[lb]) {
            const double f = (static_cast<double>(burnup) / 1000.0 - _lib->lib_burn[lb]) /
                             (_lib->lib_burn[hb] - _lib->lib_burn[lb]);
            v += f * (_lib->lib_iden[hb * Isotope::niso + isotope] - v);
        }
        return v;
    };

    auto burnRatioCoordinate = [&](const Chiffon::SpectralTerm& term, int burnup) {
        const size_t b = term.partner;
        if (b >= Isotope::niso)
            return 0.0;
        const double fl = Chiffon::SPECTRAL_LOG_DENSITY_FLOOR;
        const double na = std::max(_iden[term.isotope * stride + node], fl);
        const double nb = std::max(_iden[b * stride + node], fl);
        const double ra = std::max(referenceDensity0(term.isotope, burnup), fl);
        const double rb = std::max(referenceDensity0(b, burnup), fl);
        return std::log(na / ra) - std::log(nb / rb);
    };

    auto fissileFraction = [&](const Chiffon::SpectralTerm& term) {
        const size_t b = term.partner;
        if (b >= Isotope::niso)
            return 0.0;
        const double na = std::max(0.0, _iden[term.isotope * stride + node]);
        const double sum = na + std::max(0.0, _iden[b * stride + node]);
        return sum > 1.0e-300 ? na / sum : 0.0;
    };

    auto referenceDensity = [&](size_t isotope, int burnup) {
        if (isotope >= Isotope::niso)
            return 0.0;

        const int loIndex = findLoBurn(referenceBurnups, burnup);
        const int hiIndex = findHiBurn(referenceBurnups, burnup);
        if (loIndex < 0 || hiIndex < 0)
            return 0.0;

        const size_t lo =
            referenceBase +
            static_cast<size_t>(loIndex) *
                _lib->refr_burn_stride[modelIndex];
        const size_t hi =
            referenceBase +
            static_cast<size_t>(hiIndex) *
                _lib->refr_burn_stride[modelIndex];

        double value =
            _lib->lib_iden[lo * Isotope::niso + isotope];
        if (lo != hi && _lib->lib_burn[hi] != _lib->lib_burn[lo]) {
            const double fraction =
                (static_cast<double>(burnup) / 1000.0 -
                 _lib->lib_burn[lo]) /
                (_lib->lib_burn[hi] - _lib->lib_burn[lo]);
            value +=
                fraction *
                (_lib->lib_iden[hi * Isotope::niso + isotope] -
                 _lib->lib_iden[lo * Isotope::niso + isotope]);
        }
        return value;
    };

    // Condition the reference depletion ran at, on the same three coordinates the
    // scalar branch tables use. Centering the cross term on it keeps the column
    // exactly zero wherever the node sits on the reference condition.
    auto referenceCondition = [&](int axis, int burnup) {
        const int loIndex = findLoBurn(referenceBurnups, burnup);
        const int hiIndex = findHiBurn(referenceBurnups, burnup);
        if (loIndex < 0 || hiIndex < 0)
            return 0.0;
        const size_t lo = referenceBase +
                          static_cast<size_t>(loIndex) *
                              _lib->refr_burn_stride[modelIndex];
        const size_t hi = referenceBase +
                          static_cast<size_t>(hiIndex) *
                              _lib->refr_burn_stride[modelIndex];
        const size_t a = static_cast<size_t>(axis);
        double value   = _lib->lib_ref_branch_x[lo][a];
        if (lo != hi && _lib->lib_burn[hi] != _lib->lib_burn[lo]) {
            const double fraction =
                (static_cast<double>(burnup) / 1000.0 - _lib->lib_burn[lo]) /
                (_lib->lib_burn[hi] - _lib->lib_burn[lo]);
            value += fraction * (_lib->lib_ref_branch_x[hi][a] - value);
        }
        return value;
    };

    // The node's own value on those same three coordinates, matching the
    // `x_vals` table in UpdateUnroddedNodeXS exactly.
    const double nodeBranchX[NUM_SCALAR_BRANCHES] = {
        BoronDmod(_g, _boron_dmod_average, l) * _node_wvfr[node] *
            _g.bppm(l) * BORON_DENSITY_FACTOR,
        std::sqrt(_g.tful(l)),
        _g.dmod(l)};

    for (const auto& correction :
         _lib->lib_spectral_history[modelIndex]) {
        const auto& burnups = correction.burnups;
        const int loIndex = findLoBurn(burnups, burn);
        const int hiIndex = findHiBurn(burnups, burn);
        if (loIndex < 0 || hiIndex < 0)
            continue;

        const bool logarithmic =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::LogDensity;
        const bool rooted =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::SqrtDensity;
        const bool thermalWeighted =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::ThermalWeighted;
        const bool fastWeighted =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::FastWeighted;
        const bool ratioInteraction =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::FluxRatioInteraction;
        const bool burnRatio =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::RelativeBurnRatio;
        const bool spectralIndex =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::SpectralIndex;
        const bool spectralCross =
            correction.term.coordinate ==
            Chiffon::SpectralCoordinate::SpectralIndexInteraction;
        // Centered condition x composition cross term: zero on the reference
        // condition and zero on the reference composition, so it tilts a branch
        // slope without being able to stand in for either layer.
        const int branchAxis =
            Chiffon::BranchAxisOf(correction.term.coordinate);
        const Chiffon::SpectralCoordinate coord = correction.term.coordinate;
        const bool ratioForm =
            coord == Chiffon::SpectralCoordinate::LogDeviationSquared ||
            coord == Chiffon::SpectralCoordinate::InverseRatio ||
            coord == Chiffon::SpectralCoordinate::CubeRootRatio ||
            coord == Chiffon::SpectralCoordinate::SaturatingRatio;
        const bool fissile =
            coord == Chiffon::SpectralCoordinate::FissileFraction;
        // Rod exposure of this node, on the same rod-material fluence the rod
        // depletion layer uses. Zero with the rod out, which is where this term
        // is meant to be silent anyway.
        const int rodAgeAxis = Chiffon::RodAgeAxisOf(coord);
        const bool fluxWeighted = thermalWeighted || fastWeighted ||
                                  ratioInteraction || spectralIndex ||
                                  spectralCross || branchAxis >= 0 ||
                                  ratioForm || fissile || rodAgeAxis >= 0;
        const double density =
            _iden[correction.term.isotope * stride + node];
        const double fl = Chiffon::SPECTRAL_LOG_DENSITY_FLOOR;
        const double coordinate =
            burnRatio
                ? burnRatioCoordinate(correction.term, burn)
                : rodAgeAxis >= 0
                ? (nodeBranchX[rodAgeAxis] -
                   referenceCondition(rodAgeAxis, burn)) *
                      FineRodThermalFluenceAverage(l, currentCtype) *
                      Chiffon::ROD_AGE_SCALE
                : fissile
                ? fissileFraction(correction.term)
                : ratioForm
                ? Chiffon::RatioFormOf(
                      coord, std::max(density, fl),
                      std::max(referenceDensity(correction.term.isotope, burn),
                               fl))
                : branchAxis >= 0
                ? (nodeBranchX[branchAxis] -
                   referenceCondition(branchAxis, burn)) *
                      (density - referenceDensity(correction.term.isotope, burn))
                : spectralCross
                ? NodeSpectralIndex(l, micWork) *
                      (density - referenceDensity(correction.term.isotope, burn))
                : spectralIndex
                ? NodeSpectralIndex(l, micWork)
                : ratioInteraction
                ? std::log(std::max(NodeFluxShare(l, true), 1.0e-30) /
                           std::max(NodeFluxShare(l, false), 1.0e-30)) *
                      (density - referenceDensity(correction.term.isotope, burn))
                : fluxWeighted
                ? NodeFluxShare(l, thermalWeighted) *
                      std::max(0.0, density)
                : rooted
                ? std::sqrt(std::max(0.0, density))
                : (logarithmic
                       ? std::log(std::max(
                             density, Chiffon::SPECTRAL_LOG_DENSITY_FLOOR))
                       : std::max(0.0, density));
        const double fraction =
            loIndex == hiIndex
                ? 0.0
                : static_cast<double>(burn - burnups[loIndex]) /
                      static_cast<double>(
                          burnups[hiIndex] - burnups[loIndex]);

        auto keyCoordinate = [&](int keyBurnup) {
            if (loIndex == hiIndex || fluxWeighted)
                return coordinate;

            const double referenceNow =
                referenceDensity(
                    correction.term.isotope, burn);
            const double referenceAtKey =
                referenceDensity(
                    correction.term.isotope, keyBurnup);
            if (logarithmic) {
                return coordinate -
                       std::log(std::max(
                           referenceNow,
                           Chiffon::SPECTRAL_LOG_DENSITY_FLOOR)) +
                       std::log(std::max(
                           referenceAtKey,
                           Chiffon::SPECTRAL_LOG_DENSITY_FLOOR));
            }
            if (rooted) {
                return coordinate -
                       std::sqrt(std::max(0.0, referenceNow)) +
                       std::sqrt(std::max(0.0, referenceAtKey));
            }
            return coordinate - referenceNow + referenceAtKey;
        };

        // Rod-state increment terms carry the instantaneous rod fraction.
        const double rodWeight =
            correction.rod_scaled ? RodBlendWeight(l) : 1.0;
        if (rodWeight == 0.0)
            continue;

        const int loDelta =
            static_cast<int>(
                correction.delta_base +
                static_cast<size_t>(loIndex));
        out.push_back(
            {loDelta, keyCoordinate(burnups[loIndex]),
             weight * rodWeight * (1.0 - fraction),
             static_cast<int>(correction.term.isotope)});
        if (hiIndex != loIndex) {
            const int hiDelta =
                static_cast<int>(
                    correction.delta_base +
                    static_cast<size_t>(hiIndex));
            out.push_back(
                {hiDelta, keyCoordinate(burnups[hiIndex]),
                 weight * rodWeight * fraction,
                 static_cast<int>(correction.term.isotope)});
        }
    }
}

// PROBE: which weight blends the rodded surface in. "frod" is the
// instantaneous spectrum state, "pu" the share of Pu that accrued while rodded.
double XSSet::RodBlendWeight(int l) const {
    static const int mode = [] {
        const char* env = std::getenv("CHIFFON_PROBE_BLEND");
        if (env == nullptr) return 0;
        const std::string v(env);
        if (v == "pu") return 1;
        if (v == "both") return 2;
        return 0;
    }();
    const double f = _g.rod_fraction(l);
    if (mode == 0) return f;
    const double p = RoddedPuFraction(l);
    return mode == 1 ? p : f * p;
}

// PROBE: the node's base+branch thermal cross sections are already in place
// when the spectral-history terms are resolved (FillRodNodeXS /
// UpdateUnroddedNodeXS run first, ReconstructNode after), so this reads the
// pre-correction state and is not self-referential.
double XSSet::NodeSpectralIndex(int l, const double* micWork) const {
    const int    ng  = _g.ng();
    const int    ith = ng - 1;
    // `_micx` still holds the previous step's corrected state at the point the
    // history terms are resolved, so reading it here would feed the correction
    // back into its own coordinate. The workspace is the base+branch state.
    const double num =
        micWork != nullptr
            ? micWork[Isotope::iPu239 * ng + ith]
            : micx(XSAF, Isotope::iPu239, ith, l);
    const double den =
        micWork != nullptr
            ? micWork[Isotope::iB10 * ng + ith]
            : micx(XSAF, Isotope::iB10, ith, l);
    if (!(std::abs(den) > 1.0e-30))
        return 0.0;
    // Centre on the unbranched base state (`_ref_micx`), matching the fit side:
    // the coordinate is the spectrum *deviation*, zero at nominal conditions.
    const double bnum = refMicx(XSAF, Isotope::iPu239, ith, l);
    const double bden = refMicx(XSAF, Isotope::iB10, ith, l);
    if (!(std::abs(bden) > 1.0e-30))
        return 0.0;
    const double now  = num / den;
    const double base = bnum / bden;
    return (now > 1.0e-30 && base > 1.0e-30) ? std::log(now / base) : 0.0;
}

double XSSet::NodeFluxShare(int l, bool thermal) const {
    const int     ng  = _g.ng();
    const double* phi = _g.Phif();
    double        tot = 0.0;
    for (int ig = 0; ig < ng; ++ig)
        tot += phi[static_cast<size_t>(l) * ng + ig];
    if (!(tot > 0.0))
        return 0.0;
    const int ig = thermal ? ng - 1 : 0;
    return phi[static_cast<size_t>(l) * ng + ig] / tot;
}

void XSSet::ApplySpectralHistoryToNode(int l) {
    static thread_local std::vector<DeltaApplication> deltas;
    const double hw = _node_hw.empty() ? 0.0 : _node_hw[l];
    ResolveSpectralHistoryDeltas(l, deltas, nullptr,
                                 static_cast<size_t>(-1), 1.0 - hw);
    for (const auto& d : deltas)
        ApplyBranchDeltaIdToNode(l, d.did, d.x, d.scale);
    if (hw <= 0.0)
        return;
    const int partner =
        _lib->lib_history_partner.empty() ? -1 : _lib->lib_history_partner[_comp[l]];
    if (partner < 0)
        return;
    ResolveSpectralHistoryDeltas(l, deltas, nullptr,
                                 static_cast<size_t>(partner), hw);
    for (const auto& d : deltas)
        ApplyBranchDeltaIdToNode(l, d.did, d.x, d.scale);
}

// Accumulate the few-group MACRO contribution of a fitted delta surface.
// Scatter and the derived XSDF/XSRF channels are not reported here.
void XSSet::AccumulateDeltaMacro(int l, int did, double x, double scale,
                                 std::vector<double>& scalar) const {
    // The descending Horner reduction below MUST match ApplyBranchDeltaIdToNode's lumped loop
    // bit-for-bit (same order: val = val*xloc + cdata[(base+p)*ng+ig]); reassociating / fma /
    // power-precompute would break the term_contrib == live-XS contract.
    if (did < 0 || scale == 0.0) return;

    const int    ng    = _g.ng();
    const int    nxyz  = _g.nxyz();
    const size_t niso  = Isotope::niso;
    const auto&  dinfo = _lib->lib_deltas[did];
    int          base  = dinfo.coeff_base;
    int          nord  = dinfo.nord;
    double       xloc  = x;
    if (dinfo.mode == 1) {
        const int nintervals = dinfo.nord / dinfo.ncoeff;
        int       interval   = nintervals - 1;
        for (int i = 0; i < nintervals - 1; ++i) {
            if (x < _lib->lib_knots[dinfo.knot_offset + i + 1]) {
                interval = i;
                break;
            }
        }
        xloc = x - _lib->lib_knots[dinfo.knot_offset + interval];
        base += interval * dinfo.ncoeff;
        nord = dinfo.ncoeff;
    }

    const auto coeff_lmpx = ScalarData(_lib->lib_coeff_lmpx);
    for (int ig = 0; ig < ng; ++ig) {
        for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
            if (xt == XSDF || xt == XSRF) continue;
            const double* cdata = coeff_lmpx[xt];
            double        val   = cdata[(base + nord - 1) * ng + ig];
            for (int p = nord - 2; p >= 0; --p)
                val = val * xloc + cdata[(base + p) * ng + ig];
            scalar[xt * ng + ig] += scale * val;
        }
    }

    if (!_lib->has_coeff_micx)
        return;

    const auto   coeff_micx          = ScalarData(_lib->lib_coeff_micx);
    const size_t scalar_stride       = niso * static_cast<size_t>(ng);
    auto         densityForMacroFold = [&](size_t iso) {
        if (!UsesRodXS(l) && (iso == Isotope::iH1 || iso == Isotope::iO16 || iso == Isotope::iB10)) {
            const double nH2O       = _g.dmod(l) * _node_wvfr[l] * WATER_NUMBER_DENSITY;
            const double boron_dmod = BoronDmod(_g, _boron_dmod_average, l);
            if (iso == Isotope::iH1) return 2.0 * nH2O;
            if (iso == Isotope::iO16) return nH2O;
            return boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR;
        }
        return _iden[iso * nxyz + l];
    };

    for (size_t iso = 0; iso < niso; ++iso) {
        const double ndens = densityForMacroFold(iso);
        for (int ig = 0; ig < ng; ++ig) {
            const size_t elem = iso * static_cast<size_t>(ng) + static_cast<size_t>(ig);
            for (int xt = 0; xt < static_cast<int>(N_XS_SCALAR); ++xt) {
                if (xt == XSDF || xt == XSRF) continue;
                const double* cdata = coeff_micx[xt];
                double        val   = cdata[(base + nord - 1) * scalar_stride + elem];
                for (int p = nord - 2; p >= 0; --p)
                    val = val * xloc + cdata[(base + p) * scalar_stride + elem];
                scalar[xt * ng + ig] += scale * val * ndens;
            }
        }
    }
}

// Resolve per-isotope spectral-history contributions for diagnostics.
// Scattering contributions are omitted from this compact report.
void XSSet::ResolveTermContributions(int l, std::vector<TermContribution>& out) const {
    out.clear();
    std::vector<DeltaApplication> deltas;
    ResolveSpectralHistoryDeltas(l, deltas);

    const size_t scalar_len = N_XS_SCALAR * static_cast<size_t>(_g.ng());
    for (const auto& d : deltas) {
        auto it = std::find_if(out.begin(), out.end(), [&](const TermContribution& t) {
            return t.iso == d.iso;
        });
        if (it == out.end()) {
            out.push_back({d.iso, std::vector<double>(scalar_len, 0.0)});
            it = std::prev(out.end());
        }
        AccumulateDeltaMacro(l, d.did, d.x, d.scale, it->scalar);
    }
}

void XSSet::FillRodNodeXS(int l) {
    // Rodded nodes start from the rodded reference surface and apply explicit
    // rod-material depletion before spectral-history deltas are added. Fine rod
    // fluence is rod-material state, not fuel history memory.
    const int    nxyz  = _g.nxyz();
    const size_t ng    = static_cast<size_t>(_g.ng());
    const size_t niso  = Isotope::niso;
    const auto&  model = _lib->models[_comp[l]];

    static thread_local CrossSection         tls_xs, tls_delta, tls_xs2;
    static thread_local milk::Vector<double> tls_iden, tls_iden2;

    // Depletion-history blend on the base+branch state. RDPL stays on the
    // primary library: it is rod-material burnout, not fuel history.
    const int    partner = _lib->lib_history_partner.empty()
                               ? -1
                               : _lib->lib_history_partner[_comp[l]];
    const double hw      = (partner >= 0 && !_node_hw.empty()) ? _node_hw[l] : 0.0;
    static thread_local CrossSection         tls_xsp;
    static thread_local milk::Vector<double> tls_idenp;
    // Only the cross sections blend; `iden` stays on the unrodded trajectory
    // because the blend weight is measured against it.
    //
    // RDPL is applied to the PRIMARY side only. It carries a fresh rod toward a
    // burned one, and the partner library was itself depleted with the rod in,
    // so its base already holds that burnout -- adding RDPL there double-counts.
    auto fill = [&](CrossSection& xs, milk::Vector<double>& iden, int ctype,
                    double fluence, bool rodded) {
        model.FillCrossSection(xs, iden, tls_delta, ctype, _burn[l],
                               _g.bppm(l), _g.tful(l), _g.dmod(l));
        if (rodded)
            model.ApplyRodDepletion(xs, tls_delta, ctype, fluence, _burn[l],
                                    iden[Chiffon::Isotope::iB10],
                                    std::sqrt(_g.tful(l)), _g.dmod(l));
        if (hw <= 0.0)
            return;
        _lib->models[static_cast<size_t>(partner)].FillCrossSection(
            tls_xsp, tls_idenp, tls_delta, ctype, _burn[l], _g.bppm(l),
            _g.tful(l), _g.dmod(l));
        xs *= (1.0 - hw);
        tls_xsp *= hw;
        xs += tls_xsp;
    };

    const int begin = _rod_node_segment_offset[static_cast<size_t>(l)];
    const int end   = _rod_node_segment_offset[static_cast<size_t>(l + 1)];

    double rodded_frac = 0.0;
    for (int i = begin; i < end; ++i)
        rodded_frac += _rod_node_segment_fraction[static_cast<size_t>(i)];
    rodded_frac = std::clamp(rodded_frac, 0.0, 1.0);

    bool   has_xs        = false;
    double unrodded_frac = std::max(0.0, 1.0 - rodded_frac);
    if (unrodded_frac > EPS) {
        fill(tls_xs, tls_iden, 0, 0.0, false);
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
        const double fluence =
            FineRodThermalFluenceAverage(l, input_ctype);
        if (!has_xs) {
            fill(tls_xs, tls_iden, solve_ctype, fluence, true);
            tls_xs *= frac;
            has_xs = true;
        } else {
            fill(tls_xs2, tls_iden2, solve_ctype, fluence, true);
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

// Per-node update through a node-local contiguous workspace: gather the burnup-interpolated
// reference once, apply every fitted delta surface with stride-1 reads AND writes (the SoA
// destination stride is nxyz*8B, so the former per-delta read-modify-write touched a distinct
// cache line per element), then scatter back and rebuild this node's macroscopic XS in one pass.
void XSSet::UpdateUnroddedNodeXS(int l) {
    const int    nxyz = _g.nxyz();
    const int    ng   = _g.ng();
    const size_t niso = Isotope::niso;

    // Workspace layout: [active-xt lmpx | lmpx sm | active-xt micx | micx sm].
    const size_t nlsm    = static_cast<size_t>(ng) * ng;
    const size_t nmic    = niso * static_cast<size_t>(ng);
    const size_t nmsm    = niso * nlsm;
    const size_t off_lsm = static_cast<size_t>(N_ACTIVE_XT) * ng;
    const size_t off_mic = off_lsm + nlsm;
    const size_t off_msm = off_mic + static_cast<size_t>(N_ACTIVE_XT) * nmic;
    const size_t total   = off_msm + nmsm;

    static thread_local std::vector<double> buf;
    if (buf.size() != total) buf.resize(total);
    double* bl  = buf.data();           // lmpx scalars [t*ng + ig]
    double* bls = buf.data() + off_lsm; // lmpx scatter [ig*ng + ige]
    double* bm  = buf.data() + off_mic; // micx scalars [t*nmic + iso*ng + ig]
    double* bms = buf.data() + off_msm; // micx scatter [iso*ng*ng + ig*ng + ige]

    const auto ref_lmpx = ScalarData(_ref_lmpx);
    const auto ref_micx = ScalarData(_ref_micx);
    auto       lmpx     = ScalarXS(_lmpx);
    auto       micx     = ScalarXS(_micx);
    auto       xs       = ScalarXS(_xs);

    _node_wvfr[l] = _ref_wvfr[l];

    // 1. Gather the reference state into the workspace.
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        const double* src = ref_lmpx[ACTIVE_XT[t]];
        for (int ig = 0; ig < ng; ++ig)
            bl[t * ng + ig] = src[static_cast<size_t>(ig) * nxyz + l];
    }
    for (size_t sm = 0; sm < nlsm; ++sm)
        bls[sm] = _ref_lmpx.xssm[sm * nxyz + l];
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        const double* src = ref_micx[ACTIVE_XT[t]];
        double*       dst = bm + static_cast<size_t>(t) * nmic;
        for (size_t e = 0; e < nmic; ++e)
            dst[e] = src[e * nxyz + l];
    }
    for (size_t e = 0; e < nmsm; ++e)
        bms[e] = _ref_micx.xssm[e * nxyz + l];

    // 2. Delta applicator: Horner per element, coefficient reads and accumulation
    //    writes both stride-1 in the element direction.
    const auto    coeff_lmpx    = ScalarData(_lib->lib_coeff_lmpx);
    const auto    coeff_micx    = ScalarData(_lib->lib_coeff_micx);
    const double* coeff_lmpx_sm = _lib->lib_coeff_lmpx.xssm.data();
    const double* coeff_micx_sm = _lib->lib_coeff_micx.xssm.data();

    auto applyDelta = [&](int did, double x, double scale) {
        if (did < 0 || scale == 0.0) return;

        const auto& dinfo = _lib->lib_deltas[did];
        int         base  = dinfo.coeff_base;
        int         nord  = dinfo.nord;
        double      xloc  = x;
        if (dinfo.mode == 1) {
            const int nintervals = dinfo.nord / dinfo.ncoeff;
            int       interval   = nintervals - 1;
            for (int i = 0; i < nintervals - 1; ++i) {
                if (x < _lib->lib_knots[dinfo.knot_offset + i + 1]) {
                    interval = i;
                    break;
                }
            }
            xloc = x - _lib->lib_knots[dinfo.knot_offset + interval];
            base += interval * dinfo.ncoeff;
            nord = dinfo.ncoeff;
        }

        for (int t = 0; t < N_ACTIVE_XT; ++t) {
            const double* cdata = coeff_lmpx[ACTIVE_XT[t]];
            double*       dst   = bl + static_cast<size_t>(t) * ng;
            for (int e = 0; e < ng; ++e) {
                double val = cdata[(base + nord - 1) * ng + e];
                for (int p = nord - 2; p >= 0; --p)
                    val = val * xloc + cdata[(base + p) * ng + e];
                dst[e] += scale * val;
            }
        }
        for (size_t e = 0; e < nlsm; ++e) {
            double val = coeff_lmpx_sm[(base + nord - 1) * nlsm + e];
            for (int p = nord - 2; p >= 0; --p)
                val = val * xloc + coeff_lmpx_sm[(base + p) * nlsm + e];
            bls[e] += scale * val;
        }

        if (!_lib->has_coeff_micx) return;
        for (int t = 0; t < N_ACTIVE_XT; ++t) {
            const double* cdata = coeff_micx[ACTIVE_XT[t]];
            double*       dst   = bm + static_cast<size_t>(t) * nmic;
#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
            for (size_t e = 0; e < nmic; ++e) {
                double val = cdata[(base + nord - 1) * nmic + e];
                for (int p = nord - 2; p >= 0; --p)
                    val = val * xloc + cdata[(base + p) * nmic + e];
                dst[e] += scale * val;
            }
        }
#ifndef _MSC_VER // MSVC needs -openmp:experimental for simd, which conflicts with the llvm runtime the max-reduction needs
#pragma omp simd
#endif
        for (size_t e = 0; e < nmsm; ++e) {
            double val = coeff_micx_sm[(base + nord - 1) * nmsm + e];
            for (int p = nord - 2; p >= 0; --p)
                val = val * xloc + coeff_micx_sm[(base + p) * nmsm + e];
            bms[e] += scale * val;
        }
    };

    // Apply scalar branches before the spectral-history terms.
    const double boron_dmod                  = BoronDmod(_g, _boron_dmod_average, l);
    const double x_vals[NUM_SCALAR_BRANCHES] = {
        boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR,
        std::sqrt(_g.tful(l)),
        _g.dmod(l)};
    // Depletion-history blend: each library contributes its own branch and
    // spectral-history surfaces, weighted. wu is 1 and w is 0 without a twin.
    const double hw = _node_hw.empty() ? 0.0 : _node_hw[l];
    const double wu = 1.0 - hw;
    for (int branch = 0; branch < NUM_SCALAR_BRANCHES; ++branch) {
        const int lo = _node_delta_lo[branch][l];
        if (lo >= 0 && wu > 0.0) {
            const int    hi = _node_delta_hi[branch][l];
            const double f  = _node_delta_frac[branch][l];
            applyDelta(lo, x_vals[branch], wu * (1.0 - f));
            if (hi != lo)
                applyDelta(hi, x_vals[branch], wu * f);
        }
        if (hw <= 0.0 || _node_delta_lo_p.empty()) continue;
        const int lo_p = _node_delta_lo_p[branch][l];
        if (lo_p < 0) continue;
        const int    hi_p = _node_delta_hi_p[branch][l];
        const double f_p  = _node_delta_frac_p[branch][l];
        applyDelta(lo_p, x_vals[branch], hw * (1.0 - f_p));
        if (hi_p != lo_p)
            applyDelta(hi_p, x_vals[branch], hw * f_p);
    }

    static thread_local std::vector<DeltaApplication> history;
    ResolveSpectralHistoryDeltas(l, history, bm + nmic,
                                 static_cast<size_t>(-1), wu);
    for (const auto& d : history)
        applyDelta(d.did, d.x, d.scale);
    if (hw > 0.0) {
        const int partner = _lib->lib_history_partner.empty()
                                ? -1
                                : _lib->lib_history_partner[_comp[l]];
        if (partner >= 0) {
            ResolveSpectralHistoryDeltas(l, history, bm + nmic,
                                         static_cast<size_t>(partner), hw);
            for (const auto& d : history)
                applyDelta(d.did, d.x, d.scale);
        }
    }

    RefreshLightIsotopes(l);

    // 4. Scatter the workspace back to the SoA arrays (one strided pass).
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        double* dst = lmpx[ACTIVE_XT[t]]->data();
        for (int ig = 0; ig < ng; ++ig)
            dst[static_cast<size_t>(ig) * nxyz + l] = bl[t * ng + ig];
    }
    for (size_t sm = 0; sm < nlsm; ++sm)
        _lmpx.xssm[sm * nxyz + l] = bls[sm];
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        double*       dst = micx[ACTIVE_XT[t]]->data();
        const double* src = bm + static_cast<size_t>(t) * nmic;
        for (size_t e = 0; e < nmic; ++e)
            dst[e * nxyz + l] = src[e];
    }
    for (size_t e = 0; e < nmsm; ++e)
        _micx.xssm[e * nxyz + l] = bms[e];

    // 5. Rebuild this node's macroscopic XS from the workspace
    //    (same per-element isotope order as ReconstructNode).
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        const double* mt = bm + static_cast<size_t>(t) * nmic;
        for (int ig = 0; ig < ng; ++ig) {
            double val = bl[t * ng + ig];
            for (size_t iso = 0; iso < niso; ++iso)
                val += mt[iso * ng + ig] * _iden[iso * nxyz + l];
            (*xs[ACTIVE_XT[t]])[static_cast<size_t>(ig) * nxyz + l] = val;
        }
    }
    for (int igs = 0; igs < ng; ++igs) {
        for (int ige = 0; ige < ng; ++ige) {
            double val = bls[igs * ng + ige];
            for (size_t iso = 0; iso < niso; ++iso)
                val += bms[iso * nlsm + igs * ng + ige] * _iden[iso * nxyz + l];
            _xs.xssm[(static_cast<size_t>(igs) * ng + ige) * nxyz + l] = val;
        }
    }
    for (int ig = 0; ig < ng; ++ig) {
        const double tr                              = _xs.xstf[static_cast<size_t>(ig) * nxyz + l];
        _xs.xsdf[static_cast<size_t>(ig) * nxyz + l] = (tr > 1.0e-30) ? 0.333333333333333 / tr : 0.0;

        double rf = _xs.xsaf[static_cast<size_t>(ig) * nxyz + l];
        for (int ige = 0; ige < ng; ++ige)
            rf += _xs.xssm[(static_cast<size_t>(ig) * ng + ige) * nxyz + l];
        _xs.xsrf[static_cast<size_t>(ig) * nxyz + l] = rf;
    }
    noteMacroXsWrite();
}

// Fill the flat-XS pointer view with this instance's host arrays.  The stream
// and node-list fields stay null; BuildFlatXsStream's caller wires them.
flatxs::FlatXsView XSSet::MakeFlatXsHostView() {
    namespace fxs = flatxs;
    static_assert(fxs::N_ACTIVE == N_ACTIVE_XT, "ACTIVE_XT list drifted");

    fxs::FlatXsView v{};
    const auto coeff_lmpx = ScalarData(_lib->lib_coeff_lmpx);
    const auto coeff_micx = ScalarData(_lib->lib_coeff_micx);
    const auto ref_lmpx   = ScalarData(_ref_lmpx);
    const auto ref_micx   = ScalarData(_ref_micx);
    auto       lmpx       = ScalarXS(_lmpx);
    auto       micx       = ScalarXS(_micx);
    auto       xs         = ScalarXS(_xs);
    for (int t = 0; t < N_ACTIVE_XT; ++t) {
        v.coeff_lmp[t] = coeff_lmpx[ACTIVE_XT[t]];
        v.coeff_mic[t] = coeff_micx[ACTIVE_XT[t]];
        v.ref_lmp[t]   = ref_lmpx[ACTIVE_XT[t]];
        v.ref_mic[t]   = ref_micx[ACTIVE_XT[t]];
        v.lmp[t]       = lmpx[static_cast<size_t>(ACTIVE_XT[t])]->data();
        v.mic[t]       = micx[static_cast<size_t>(ACTIVE_XT[t])]->data();
    }
    for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
        v.xs[xt]      = xs[xt]->data();
        v.mic_all[xt] = micx[xt]->data();
        v.lmp_all[xt] = lmpx[xt]->data();
    }
    v.coeff_lsm = _lib->lib_coeff_lmpx.xssm.data();
    v.coeff_msm = _lib->lib_coeff_micx.xssm.data();
    v.ref_lsm   = _ref_lmpx.xssm.data();
    v.ref_msm   = _ref_micx.xssm.data();
    v.lsm       = _lmpx.xssm.data();
    v.msm       = _micx.xssm.data();
    v.xs_ssm    = _xs.xssm.data();
    v.iden      = _iden.data();
    v.wvfr      = _node_wvfr.data();
    v.dmod      = &_g.dmod(0);
    v.bppm      = &_g.bppm(0);

    if (_flatxs_deltas.size() != _lib->lib_deltas.size()) {
        _flatxs_deltas.resize(_lib->lib_deltas.size());
        for (size_t i = 0; i < _lib->lib_deltas.size(); ++i) {
            const auto& d     = _lib->lib_deltas[i];
            _flatxs_deltas[i] = {d.nord, d.mode, d.ncoeff, d.coeff_base,
                                 d.knot_offset};
        }
    }
    v.deltas             = _flatxs_deltas.data();
    v.knots              = _lib->lib_knots.data();
    v.has_coeff_micx     = _lib->has_coeff_micx ? 1 : 0;
    v.nxyz               = _g.nxyz();
    v.boron_dmod_average = _boron_dmod_average;
    v.use_average_dmod   = USE_AVERAGE_DMOD_FOR_BORON ? 1 : 0;
    return v;
}

FlatXsLibShape XSSet::MakeFlatXsLibShape() const {
    FlatXsLibShape s{};
    s.lmp_slot = _lib->lib_coeff_lmpx.xstf.size();
    s.lsm      = _lib->lib_coeff_lmpx.xssm.size();
    s.mic_slot = _lib->has_coeff_micx ? _lib->lib_coeff_micx.xstf.size() : 0;
    s.msm      = _lib->has_coeff_micx ? _lib->lib_coeff_micx.xssm.size() : 0;
    s.n_knots  = _lib->lib_knots.size();
    s.n_deltas = _lib->lib_deltas.size();
    return s;
}

// Capture for offline replay (RASBERY_FLATXS_DUMP=<path>): the first
// all-unrodded UpdateFlatXS call's full inputs (<path>.in, incl. the resolved
// stream) and outputs (<path>.out), raw doubles.  test/flatxs_replay.cpp
// applies the shared body to the captured inputs and reports elementwise ULP
// against the captured outputs; its sweep mode mines the contraction mask.
static void flatxsDumpState(const char* path, const flatxs::FlatXsView& v,
                            const FlatXsLibShape& shape, int ng, bool full) {
    namespace fxs = flatxs;
    std::FILE* f = std::fopen(path, "wb");
    if (!f)
        return;
    const std::size_t nx  = static_cast<std::size_t>(v.nxyz);
    const std::size_t mic = static_cast<std::size_t>(xsrecon::NISO) * xsrecon::NG * nx;
    const std::size_t lmp = static_cast<std::size_t>(xsrecon::NG) * nx;
    const std::size_t msm = static_cast<std::size_t>(xsrecon::NISO) * xsrecon::NG * xsrecon::NG * nx;
    const std::size_t ssm = static_cast<std::size_t>(xsrecon::NG) * xsrecon::NG * nx;
    const std::size_t stream_len =
        v.n_nodes > 0 ? static_cast<std::size_t>(v.node_off[v.n_nodes - 1]) +
                            static_cast<std::size_t>(v.node_cnt[v.n_nodes - 1])
                      : 0;

    const std::int64_t hdr[16] = {
        ng, v.nxyz, xsrecon::NISO, v.n_nodes,
        static_cast<std::int64_t>(stream_len),
        static_cast<std::int64_t>(shape.n_deltas),
        static_cast<std::int64_t>(shape.n_knots),
        static_cast<std::int64_t>(shape.lmp_slot),
        static_cast<std::int64_t>(shape.lsm),
        static_cast<std::int64_t>(shape.mic_slot),
        static_cast<std::int64_t>(shape.msm),
        v.has_coeff_micx, v.use_average_dmod, full ? 1 : 0, 0, 0};
    std::fwrite(hdr, sizeof hdr[0], 16, f);
    std::fwrite(&v.boron_dmod_average, sizeof(double), 1, f);

    if (full) {
        std::fwrite(v.nodes, sizeof(int), static_cast<std::size_t>(v.n_nodes), f);
        std::fwrite(v.node_off, sizeof(int), static_cast<std::size_t>(v.n_nodes), f);
        std::fwrite(v.node_cnt, sizeof(int), static_cast<std::size_t>(v.n_nodes), f);
        std::fwrite(v.stream_did, sizeof(int), stream_len, f);
        std::fwrite(v.stream_x, sizeof(double), stream_len, f);
        std::fwrite(v.stream_scale, sizeof(double), stream_len, f);
        std::fwrite(v.deltas, sizeof(fxs::DeltaMeta), shape.n_deltas, f);
        std::fwrite(v.knots, sizeof(double), shape.n_knots, f);
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            std::fwrite(v.coeff_lmp[t], sizeof(double), shape.lmp_slot, f);
        std::fwrite(v.coeff_lsm, sizeof(double), shape.lsm, f);
        if (v.has_coeff_micx) {
            for (int t = 0; t < fxs::N_ACTIVE; ++t)
                std::fwrite(v.coeff_mic[t], sizeof(double), shape.mic_slot, f);
            std::fwrite(v.coeff_msm, sizeof(double), shape.msm, f);
        }
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            std::fwrite(v.ref_mic[t], sizeof(double), mic, f);
        std::fwrite(v.ref_msm, sizeof(double), msm, f);
        for (int t = 0; t < fxs::N_ACTIVE; ++t)
            std::fwrite(v.ref_lmp[t], sizeof(double), lmp, f);
        std::fwrite(v.ref_lsm, sizeof(double), ssm, f);
        std::fwrite(v.wvfr, sizeof(double), nx, f);
        std::fwrite(v.dmod, sizeof(double), nx, f);
        std::fwrite(v.bppm, sizeof(double), nx, f);
        std::fwrite(v.iden, sizeof(double), static_cast<std::size_t>(xsrecon::NISO) * nx, f);
    }
    // Live arrays: the .in file carries the pre-call state, the .out file the
    // post-call state, in the same order.
    for (int t = 0; t < fxs::N_ACTIVE; ++t)
        std::fwrite(v.lmp[t], sizeof(double), lmp, f);
    std::fwrite(v.lsm, sizeof(double), ssm, f);
    for (int t = 0; t < fxs::N_ACTIVE; ++t)
        std::fwrite(v.mic[t], sizeof(double), mic, f);
    std::fwrite(v.msm, sizeof(double), msm, f);
    for (size_t xt = 0; xt < N_XS_SCALAR; ++xt)
        std::fwrite(v.xs[xt], sizeof(double), lmp, f);
    std::fwrite(v.xs_ssm, sizeof(double), ssm, f);
    if (!full)
        std::fwrite(v.iden, sizeof(double), 3 * nx, f); // H-1/B-10/O-16 rows
    std::fclose(f);
}

// Resolve every applyDelta call of the given (unrodded) nodes into the flat
// stream scratch, in exactly the order UpdateUnroddedNodeXS makes them: the
// three scalar branches (wu arm then history-twin arm per branch), then the
// spectral-history terms of the node's own library, then the twin's.  The
// history resolution is the very ResolveSpectralHistoryDeltas the CPU arm
// calls, fed a two-element workspace probe for NodeSpectralIndex (computed
// with the mined contraction forms, see FlatXsKernel.h).
void XSSet::BuildFlatXsStream(const std::vector<int>& nodes) {
    namespace fxs = flatxs;
    const int    ng  = _g.ng();
    const int    ith = ng - 1;
    const size_t n   = nodes.size();
    if (_flatxs_node_apps.size() < n)
        _flatxs_node_apps.resize(n);
    const fxs::FlatXsView hv = MakeFlatXsHostView();

#pragma omp parallel for schedule(dynamic, 64) if (static_cast<int>(n) > OMP_THRESHOLD)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const int l    = nodes[i];
        auto&     apps = _flatxs_node_apps[static_cast<size_t>(i)];
        apps.clear();

        const double boron_dmod = BoronDmod(_g, _boron_dmod_average, l);
        const double x_vals[NUM_SCALAR_BRANCHES] = {
            boron_dmod * _node_wvfr[l] * _g.bppm(l) * BORON_DENSITY_FACTOR,
            std::sqrt(_g.tful(l)),
            _g.dmod(l)};
        const double hw = _node_hw.empty() ? 0.0 : _node_hw[l];
        const double wu = 1.0 - hw;
        for (int branch = 0; branch < NUM_SCALAR_BRANCHES; ++branch) {
            const int lo = _node_delta_lo[branch][l];
            if (lo >= 0 && wu > 0.0) {
                const int    hi = _node_delta_hi[branch][l];
                const double f  = _node_delta_frac[branch][l];
                apps.push_back({lo, x_vals[branch], wu * (1.0 - f), 0});
                if (hi != lo)
                    apps.push_back({hi, x_vals[branch], wu * f, 0});
            }
            if (hw <= 0.0 || _node_delta_lo_p.empty())
                continue;
            const int lo_p = _node_delta_lo_p[branch][l];
            if (lo_p < 0)
                continue;
            const int    hi_p = _node_delta_hi_p[branch][l];
            const double f_p  = _node_delta_frac_p[branch][l];
            apps.push_back({lo_p, x_vals[branch], hw * (1.0 - f_p), 0});
            if (hi_p != lo_p)
                apps.push_back({hi_p, x_vals[branch], hw * f_p, 0});
        }

        // Two-element XSAF workspace probe (base + branch state) for
        // NodeSpectralIndex; ACTIVE_XT index 1 is XSAF.
        static thread_local std::vector<double>            micprobe;
        static thread_local std::vector<int>               p_did;
        static thread_local std::vector<double>            p_x, p_scale;
        static thread_local std::vector<DeltaApplication>  hist;
        if (micprobe.size() != Isotope::niso * static_cast<size_t>(ng))
            micprobe.assign(Isotope::niso * static_cast<size_t>(ng), 0.0);
        p_did.clear(); p_x.clear(); p_scale.clear();
        for (const auto& a : apps) {
            p_did.push_back(a.did);
            p_x.push_back(a.x);
            p_scale.push_back(a.scale);
        }
        const int nb  = static_cast<int>(p_did.size());
        const int ePu = static_cast<int>(Isotope::iPu239) * ng + ith;
        const int eB  = static_cast<int>(Isotope::iB10) * ng + ith;
        micprobe[static_cast<size_t>(ePu)] = fxs::flatxsProbeMicElement(
            hv, l, 1, ePu, p_did.data(), p_x.data(), p_scale.data(), nb,
            fxs::StaticForms{});
        micprobe[static_cast<size_t>(eB)] = fxs::flatxsProbeMicElement(
            hv, l, 1, eB, p_did.data(), p_x.data(), p_scale.data(), nb,
            fxs::StaticForms{});

        ResolveSpectralHistoryDeltas(l, hist, micprobe.data(),
                                     static_cast<size_t>(-1), wu);
        for (const auto& d : hist)
            apps.push_back(d);
        if (hw > 0.0) {
            const int partner = _lib->lib_history_partner.empty()
                                    ? -1
                                    : _lib->lib_history_partner[_comp[l]];
            if (partner >= 0) {
                ResolveSpectralHistoryDeltas(l, hist, micprobe.data(),
                                             static_cast<size_t>(partner), hw);
                for (const auto& d : hist)
                    apps.push_back(d);
            }
        }
    }

    // Serial concatenation keeps the stream in node order.
    _flatxs_off.resize(n);
    _flatxs_cnt.resize(n);
    _flatxs_stream_did.clear();
    _flatxs_stream_x.clear();
    _flatxs_stream_scale.clear();
    for (size_t i = 0; i < n; ++i) {
        _flatxs_off[i] = static_cast<int>(_flatxs_stream_did.size());
        for (const auto& a : _flatxs_node_apps[i]) {
            _flatxs_stream_did.push_back(a.did);
            _flatxs_stream_x.push_back(a.x);
            _flatxs_stream_scale.push_back(a.scale);
        }
        _flatxs_cnt[i] = static_cast<int>(_flatxs_stream_did.size()) - _flatxs_off[i];
    }
}

bool XSSet::TryUpdateFlatXSGpu(const std::vector<int>& unrodded, bool any_rodded) {
    namespace fxs = flatxs;
    if (_g.ng() != xsrecon::NG || static_cast<int>(Isotope::niso) != xsrecon::NISO)
        return false;

    if (!_xsrecon_backend)
        _xsrecon_backend = std::make_unique<XsReconBackend>();
    if (!_xsrecon_backend->available()) {
        static std::once_flag warn_once;
        std::call_once(warn_once, [this] {
            std::cerr << "[RASBERY][WARN][flatxs] RASBERY_GPU_FLATXS set but device "
                         "path unavailable ("
                      << _xsrecon_backend->status() << ") -- CPU loop\n";
        });
        return false;
    }

    const int nxyz = _g.nxyz();
    if (!_flatxs_pinned) {
        // The live micx/lmpx/xs/iden pins are shared with the xsrecon arm
        // (pinHost is idempotent); the reference blocks are this arm's own.
        const size_t ngn = static_cast<size_t>(_g.ng()) * nxyz;
        const size_t ssn = static_cast<size_t>(_g.ng()) * _g.ng() * nxyz;
        for (int xt = 0; xt < xsrecon::NXS; ++xt) {
            const auto t = static_cast<XSTYPE>(xt);
            XsReconBackend::pinHost(_micx[t].data(), xsrecon::NISO * ngn * sizeof(double),
                                    "xs.micx@flatxs");
            XsReconBackend::pinHost(_lmpx[t].data(), ngn * sizeof(double), "xs.lmpx@flatxs");
            XsReconBackend::pinHost(_xs[t].data(), ngn * sizeof(double), "xs.xs@flatxs");
        }
        XsReconBackend::pinHost(_micx.xssm.data(), xsrecon::NISO * ssn * sizeof(double),
                                "xs.micx_ssm@flatxs");
        XsReconBackend::pinHost(_lmpx.xssm.data(), ssn * sizeof(double), "xs.lmpx_ssm@flatxs");
        XsReconBackend::pinHost(_xs.xssm.data(), ssn * sizeof(double), "xs.xssm@flatxs");
        XsReconBackend::pinHost(_iden.data(),
                                static_cast<size_t>(xsrecon::NISO) * nxyz * sizeof(double),
                                "xs.iden@flatxs");
        for (int t = 0; t < N_ACTIVE_XT; ++t) {
            const auto xt = static_cast<XSTYPE>(ACTIVE_XT[t]);
            XsReconBackend::pinHost(_ref_micx[xt].data(), xsrecon::NISO * ngn * sizeof(double),
                                    "xs.ref_micx@flatxs");
            XsReconBackend::pinHost(_ref_lmpx[xt].data(), ngn * sizeof(double),
                                    "xs.ref_lmpx@flatxs");
        }
        XsReconBackend::pinHost(_ref_micx.xssm.data(), xsrecon::NISO * ssn * sizeof(double),
                                "xs.ref_micx_ssm@flatxs");
        XsReconBackend::pinHost(_ref_lmpx.xssm.data(), ssn * sizeof(double),
                                "xs.ref_lmpx_ssm@flatxs");
        _flatxs_pinned = true;
    }

    fxs::FlatXsView v = MakeFlatXsHostView();
    v.nodes           = unrodded.data();
    v.n_nodes         = static_cast<int>(unrodded.size());
    v.node_off        = _flatxs_off.data();
    v.node_cnt        = _flatxs_cnt.data();
    v.stream_did      = _flatxs_stream_did.data();
    v.stream_x        = _flatxs_stream_x.data();
    v.stream_scale    = _flatxs_stream_scale.data();

    // THE DEVICE ARM IS A MACRO-XS WRITER TOO.  solveFlatXs downloads whole
    // arrays into the host _xs columns pinned above (that is what the "runs
    // after the device download so the device's whole-array downloads cannot
    // clobber these columns" note in UpdateFlatXS is about), and xsdf/xsrf are
    // among them.  _hoststate_generation is deliberately NOT bumped for this
    // write -- the device mirror and the host agree afterwards -- which is
    // exactly why the constants gate needs its own counter.  Bumped
    // UNCONDITIONALLY: a refusal that had already run part of the download
    // would otherwise leave a write unannounced.
    noteMacroXsWrite();
    return _xsrecon_backend->solveFlatXs(v, MakeFlatXsLibShape(), _micx_generation,
                                         _micx_generation + 1, _ref_generation,
                                         _hoststate_generation, !any_rodded);
}

void XSSet::UpdateFlatXS(const XSUpdateOptions& options) {
    if (!_simd_ready) {
        Update();
        return;
    }

    const int  nxyz       = _g.nxyz();
    const bool all_nodes  = options.nodes.empty();
    const int  node_count = all_nodes ? nxyz : static_cast<int>(options.nodes.size());
    xsphase::Scope flatxs_scope(xsphase::tallies().flatxs,
                                static_cast<std::uint64_t>(node_count));
    _boron_dmod_average   = FuelVolumeAverageDmod(_g);

    // Device arm (RASBERY_GPU_FLATXS) and capture (RASBERY_FLATXS_DUMP): both
    // need the applyDelta stream resolved up front.  With neither set this
    // whole block costs two cached getenv reads.
    static const char* dump_path = std::getenv("RASBERY_FLATXS_DUMP");
    static bool        dump_done = false;
    const bool         want_dump = dump_path != nullptr && !dump_done;
    if (rasberyGpuFlatXsEnabled() || want_dump) {
        auto& unrodded = _flatxs_unrodded;
        auto& rodded   = _flatxs_rodded;
        unrodded.clear();
        rodded.clear();
        for (int i = 0; i < node_count; ++i) {
            const int l = all_nodes ? i : options.nodes[i];
            (UsesRodXS(l) ? rodded : unrodded).push_back(l);
        }
        if (!unrodded.empty()) {
            // The CPU loop refreshes each node's wvfr before anything reads
            // it, and nothing reads a neighbour's, so the bulk refresh sees
            // the same values.
            for (int l : unrodded)
                _node_wvfr[l] = _ref_wvfr[l];
            BuildFlatXsStream(unrodded);

            // A capture forces the CPU reference loop so the .out file is the
            // ground truth the replay gate scores against.
            const bool dump_this = want_dump && rodded.empty();
            bool       gpu_ok    = false;
            if (rasberyGpuFlatXsEnabled() && !dump_this)
                gpu_ok = TryUpdateFlatXSGpu(unrodded, !rodded.empty());

            if (gpu_ok || dump_this) {
                if (dump_this) {
                    dump_done = true;
                    flatxs::FlatXsView dv = MakeFlatXsHostView();
                    dv.nodes              = unrodded.data();
                    dv.n_nodes            = static_cast<int>(unrodded.size());
                    dv.node_off           = _flatxs_off.data();
                    dv.node_cnt           = _flatxs_cnt.data();
                    dv.stream_did         = _flatxs_stream_did.data();
                    dv.stream_x           = _flatxs_stream_x.data();
                    dv.stream_scale       = _flatxs_stream_scale.data();
                    flatxsDumpState((std::string(dump_path) + ".in").c_str(), dv,
                                    MakeFlatXsLibShape(), _g.ng(), true);

                    const int un = static_cast<int>(unrodded.size());
#pragma omp parallel for schedule(dynamic, 16) if (un > OMP_THRESHOLD)
                    for (int i = 0; i < un; ++i) {
                        xsphase::Scope unrod_scope(xsphase::tallies().flatxs_unrodded, 1);
                        UpdateUnroddedNodeXS(unrodded[static_cast<size_t>(i)]);
                    }
                    flatxsDumpState((std::string(dump_path) + ".out").c_str(), dv,
                                    MakeFlatXsLibShape(), _g.ng(), false);
                }

                // Rodded remainder on the CPU, exactly the reference loop's
                // rodded branch.  Runs after the device download so the
                // device's whole-array downloads cannot clobber these columns.
                const int rn = static_cast<int>(rodded.size());
#pragma omp parallel for schedule(dynamic, 16) if (rn > OMP_THRESHOLD)
                for (int i = 0; i < rn; ++i) {
                    xsphase::Scope rod_scope(xsphase::tallies().flatxs_rodded, 1);
                    const int l = rodded[static_cast<size_t>(i)];
                    FillRodNodeXS(l);
                    ApplySpectralHistoryToNode(l);
                    ReconstructNode(static_cast<size_t>(l));
                }
                ++_micx_generation;
                // The CPU wrote host arrays in this block only when the GPU
                // declined (dump run) or rodded nodes ran after the download.
                if (!gpu_ok || !rodded.empty())
                    ++_hoststate_generation;
                return;
            }
            // Device arm declined: fall through to the reference loop.
            // WP1 (plan Sec 6.3).  This seam had no counter of any kind: a
            // FlatXS arm that refused every node looked exactly like an arm
            // that was never set.
            RASBERY_GPU_FULL_GUARD_IF(rasberyGpuFlatXsEnabled() && !dump_this, FlatXs,
                                      "XSSet::UpdateFlatXS",
                                      "the FlatXS device arm declined; the reference "
                                      "reconstruction loop runs");
        }
    }

    // Rodded nodes cost far more than unrodded ones (two full Chiffon fills + rod
    // depletion), so dynamic chunks keep the rod-bank threads from straggling.
#pragma omp parallel for schedule(dynamic, 16) if (node_count > OMP_THRESHOLD)
    for (int i = 0; i < node_count; ++i) {
        const int l = all_nodes ? i : options.nodes[i];

        if (UsesRodXS(l)) {
            xsphase::Scope rod_scope(xsphase::tallies().flatxs_rodded, 1);
            FillRodNodeXS(l);
            ApplySpectralHistoryToNode(l);
            ReconstructNode(static_cast<size_t>(l));
        } else {
            xsphase::Scope unrod_scope(xsphase::tallies().flatxs_unrodded, 1);
            UpdateUnroddedNodeXS(l);
        }
    }

    // Even a partial pass may have rebuilt _micx/_lmpx rows; the device copy
    // re-uploads whole, which is conservative but never stale.
    ++_micx_generation;
    ++_hoststate_generation;
}

double XSSet::FineRodThermalFluenceAverage(int l, int ctype) const {
    if (ctype <= 0 || _fine_rod_type.empty())
        return 0.0;

    const int nxy  = _g.nxy();
    const int div  = std::max(1, _axial_rod_division);
    const int k    = l / nxy;
    const int l2d  = l % nxy;
    double    sum  = 0.0;
    int       nrod = 0;
    // Rod material fluence lives on the fine axial rod-state mesh. The current
    // single-rod history model uses the arithmetic mean over fine cells of the
    // requested ctype that overlap this coarse node.
    for (int m = 0; m < div; ++m) {
        const int idx = (k * div + m) * nxy + l2d;
        if (_fine_rod_type[static_cast<size_t>(idx)] != ctype)
            continue;
        // Count a cell as rodded for fluence purposes when it is at least half rodded,
        // matching the historical center-in-rod criterion (frac > 0.5).
        if (_fine_rod_frac[static_cast<size_t>(idx)] < 0.5)
            continue;
        sum += _fine_rod_thermal_fluence[static_cast<size_t>(idx)];
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

    const auto& model     = _lib->models[_comp[l]];
    const int   solve_ctp = (ctype != 0 && model._refr_dpts.count(ctype) != 0) ? ctype : 0;

    static thread_local CrossSection         tls_xs, tls_delta;
    static thread_local milk::Vector<double> tls_iden;
    model.FillCrossSection(tls_xs, tls_iden, tls_delta,
                           solve_ctp, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l));
    model.ApplyRodDepletion(
        tls_xs, tls_delta, solve_ctp, fluence, _burn[l],
        tls_iden[Chiffon::Isotope::iB10],
        std::sqrt(_g.tful(l)), _g.dmod(l));

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
            const double fluence =
                FineRodThermalFluenceAverage(
                    coarse_l[c], coarse_ctypes[c][s]);
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

    double max_flux = 0.0;
    if (SolveDenseLinearSystem(a, rhs, matrix_size))
        for (double v : rhs)
            max_flux = std::max(max_flux, v);
    if (max_flux <= 0.0 || !std::isfinite(max_flux)) {
        // Singular/degenerate fine-mesh solve: skip the cusp correction for this tip node.
        PLOG_ERROR << "rod cusping: singular fine-mesh solve at node " << tip_l << ", cusp skipped";
        return;
    }
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
    noteMacroXsWrite();
}

void XSSet::ResetCuspingNodesToBase(const std::vector<int>& nodes) {
    ++_hoststate_generation; // restores _xs columns from host snapshots
    // The restore below writes EVERY N_XS_SCALAR column, XSDF and XSRF among
    // them, through the generic _xs[xtype] form -- so this is a macro-XS writer
    // even though it names neither array.
    noteMacroXsWrite();
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
        sig[6] = FineRodThermalFluenceAverage(l, _ctyp[l]);
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

bool XSSet::RodCuspingQuiescent() const {
    // Term 1 and term 3 are free; term 2 is one pass over rod_fraction, paid
    // once per SEGMENT rather than once per outer -- which is cheaper than the
    // call it replaces, since ApplyRodCusping walks the same array.
    if (_axial_rod_division <= 0) return true;
    if (!_rod_cusping_nodes_scratch.empty()) return false;
    const int nxyz = _g.nxyz();
    for (int l = 0; l < nxyz; ++l) {
        const double frac = _g.rod_fraction(l);
        if (frac > EPS && frac < 1.0 - EPS) return false;
    }
    return true;
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

    // BUMPED ONLY WHEN _xs ACTUALLY MOVED.
    //
    // It used to bump unconditionally, on the argument that a redundant
    // invalidation only costs a redundant upload.  That was true while nothing
    // gated on it per outer.  The device outer segment now does: it syncs xsnf
    // when the generation changes, and this function is CALLED every outer -- so
    // an unconditional bump made every single one of those uploads unskippable.
    // Measured on kngr_238, which has an axial rod division but no fractional
    // node: 661 outers, 661 xsnf uploads, 0 elided, on a deck where cusping
    // never once wrote a cross section.
    //
    // The stencil is the only writer here, and ResetCuspingNodesToBase bumps for
    // itself (:3180), so `the stencil ran` is the complete condition.
    if (!_rod_cusping_nodes_scratch.empty())
        ++_hoststate_generation; // cusping blends _xs in place
    return !prev_scratch.empty() || !_rod_cusping_nodes_scratch.empty();
}

// Norm factor

double XSSet::NormFactor(double power, const XSArraySet& xs_arr, const double* flux) const {
    const int ng   = _g.ng();
    const int nxyz = _g.nxyz();

    // Deterministic reduction: the node range is cut into a fixed number of
    // chunks that does NOT depend on the thread count, each chunk is summed
    // serially, and the partials are accumulated in ascending chunk order.
    // This makes reaction_sum bitwise identical for any OMP_NUM_THREADS,
    // unlike `reduction(+:)` whose combine order follows the scheduler.
    const int           nchunk = rasbery_det_chunks(nxyz);
    std::vector<double> partial(static_cast<size_t>(nchunk), 0.0);

#pragma omp parallel for schedule(dynamic) if (nxyz > rasbery_omp_gate)
    for (int c = 0; c < nchunk; ++c) {
        const int lb  = rasbery_det_chunk_begin(nxyz, nchunk, c);
        const int le  = rasbery_det_chunk_begin(nxyz, nchunk, c + 1);
        double    acc = 0.0;
        for (int l = lb; l < le; ++l)
            for (int ig = 0; ig < ng; ++ig)
                acc += xs_arr.xskf[ig * nxyz + l] * flux[l * ng + ig] * _g.vol(l);
        partial[static_cast<size_t>(c)] = acc;
    }

    double reaction_sum = 0.0;
    for (int c = 0; c < nchunk; ++c)
        reaction_sum += partial[static_cast<size_t>(c)];

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
    if (!_g.IsFuel(l))
        return 0.0;

    const size_t mi = _comp[l];
    return (_g.vol(l) / _lib->lib_model_volu[mi]) * _lib->lib_model_hmas[mi];
}

// Burnup update

void XSSet::UpdateBurnup(double dt, double power) {
    xsphase::Scope burnup_scope(xsphase::tallies().update_burnup,
                                static_cast<std::uint64_t>(_g.nxyz()));
    const int     ng   = _g.ng();
    const int     nxyz = _g.nxyz();
    const double* const flux = _g.Phif();

    const double norm_factor = NormFactor(power);

#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
    for (int l = 0; l < nxyz; ++l) {
        double burn = 0.0;
        for (int ig = 0; ig < ng; ++ig)
            burn += _xs.xskf[ig * nxyz + l] * flux[l * ng + ig] * norm_factor * _g.vol(l) * dt;

        if (burn >= 1.0e-10) {
            const double dfac               = 8.64e7 * (_g.vol(l) / _lib->lib_model_volu[_comp[l]]) * _lib->lib_model_hmas[_comp[l]];
            const double burn_key_increment = burn / dfac * 1000.0;
            _burn[l] += static_cast<int>(burn_key_increment + 0.5);
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

/// The closed-form equilibrium image of the I-135/Xe-135/Xe-135m chain at one
/// node.  COMPUTE ONLY: it writes nothing at all.
///
/// Split out of ApplyXeEquilibrium -- which is now the three-line wrapper below
/// and keeps every one of its callers -- so the safeguarded Anderson arm of the
/// Xe fixed point (Driver.h, plan Rev.4 Sec 10.1) can look at F(x) BEFORE
/// deciding what to commit.  The arithmetic and the order it is evaluated in are
/// unchanged, so every applied path is bit-for-bit what it was.
struct XeEquilibriumImage {
    double i135   = 0.0;
    double xe135  = 0.0;
    double xe135m = 0.0;
};

static XeEquilibriumImage ComputeXeEquilibrium(const milk::Vector<double>& iden,
                                               const std::vector<double>& cond,
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

    XeEquilibriumImage img;
    img.i135   = Ieq;
    img.xe135  = Xeeq;
    img.xe135m = brItoXe135m * lambdaI * Ieq / lambdaXem;
    return img;
}

/// Apply Xe-135 equilibrium overwrite on the isotope density vector.
static void ApplyXeEquilibrium(milk::Vector<double>& iden, const std::vector<double>& cond,
                               double sumflux) {
    using namespace Chiffon::Isotope;
    const XeEquilibriumImage img = ComputeXeEquilibrium(iden, cond, sumflux);
    iden[iI135]   = img.i135;
    iden[iXe135]  = img.xe135;
    iden[iXe135m] = img.xe135m;
}

// Divergence probe for the xsrecon A/B, behind RASBERY_XSRECON_DEBUG_HASH:
// one FNV-1a line per equilibrium-Xe call over everything the call writes
// (_xs incl. scatter, the three Xe-chain _iden rows, max_change).  Diffing the
// two arms' streams pinpoints the first call whose outputs differ, which a
// 6-digit console trace cannot.
static void xsreconDebugHash(const XSArraySet& xs, const milk::Vector<double>& iden,
                             int ng, int nxyz, double max_change) {
    static const bool on = std::getenv("RASBERY_XSRECON_DEBUG_HASH") != nullptr;
    if (!on)
        return;
    auto mix = [](const double* p, size_t n, std::uint64_t s) {
        for (size_t i = 0; i < n; ++i) {
            std::uint64_t b;
            std::memcpy(&b, &p[i], sizeof b);
            s = (s ^ b) * 1099511628211ULL;
        }
        return s;
    };
    const size_t  ngn  = static_cast<size_t>(ng) * static_cast<size_t>(nxyz);
    const std::uint64_t seed = 1469598103934665603ULL;
    static int    call = 0;
    ++call;
    for (int xt = XSTF; xt <= XS3N; ++xt)
        std::fprintf(stderr, "[XSRECON][HASH] call=%d xs%d=%016llx\n", call, xt,
                     static_cast<unsigned long long>(
                         mix(xs[static_cast<XSTYPE>(xt)].data(), ngn, seed)));
    std::fprintf(stderr, "[XSRECON][HASH] call=%d ssm=%016llx\n", call,
                 static_cast<unsigned long long>(
                     mix(xs.xssm.data(), static_cast<size_t>(ng) * ng * nxyz, seed)));
    std::fprintf(stderr, "[XSRECON][HASH] call=%d iden=%016llx\n", call,
                 static_cast<unsigned long long>(
                     mix(iden.data() + static_cast<size_t>(Isotope::iI135) * nxyz,
                         3 * static_cast<size_t>(nxyz), seed)));
    std::fprintf(stderr, "[XSRECON][HASH] call=%d max=%016llx\n", call,
                 static_cast<unsigned long long>(mix(&max_change, 1, seed)));
}

// Capture for offline replay (RASBERY_XSRECON_DUMP=<path>): the first Xe
// call's full inputs before the loop and outputs after it, raw doubles.  A
// replay tool applies the shared kernel body to the captured inputs and
// reports elementwise ULP against the captured outputs -- production data,
// production codegen, no synthetic-coverage gap.
static void xsreconDumpArrays(const char* path, Geometry& g,
                              const XSArraySet& micx, const XSArraySet& lmpx,
                              const XSArraySet& xs, const milk::Vector<double>& iden,
                              double norm_factor, double relax) {
    using namespace Isotope;
    std::FILE* f = std::fopen(path, "wb");
    if (!f)
        return;
    const int ng = g.ng(), nxyz = g.nxyz();
    std::int64_t hdr[2] = {ng, nxyz};
    std::fwrite(hdr, sizeof hdr[0], 2, f);
    std::fwrite(&norm_factor, sizeof(double), 1, f);
    std::fwrite(&relax, sizeof(double), 1, f);
    std::vector<char> isf(static_cast<size_t>(nxyz));
    for (int l = 0; l < nxyz; ++l)
        isf[static_cast<size_t>(l)] = g.IsFuel(l) ? 1 : 0;
    std::fwrite(isf.data(), 1, isf.size(), f);
    std::vector<double> dep(2 * niso);
    for (size_t j = 0; j < niso; ++j) {
        dep[j]        = depTrans(iI135, j);
        dep[niso + j] = depTrans(iXe135, j);
    }
    std::fwrite(dep.data(), sizeof(double), dep.size(), f);
    const size_t ngn = static_cast<size_t>(ng) * nxyz;
    for (int xt = XSTF; xt <= XS3N; ++xt)
        std::fwrite(micx[static_cast<XSTYPE>(xt)].data(), sizeof(double),
                    niso * ngn, f);
    std::fwrite(micx.xssm.data(), sizeof(double), niso * static_cast<size_t>(ng) * ng * nxyz, f);
    for (int xt = XSTF; xt <= XS3N; ++xt)
        std::fwrite(lmpx[static_cast<XSTYPE>(xt)].data(), sizeof(double), ngn, f);
    std::fwrite(lmpx.xssm.data(), sizeof(double), static_cast<size_t>(ng) * ng * nxyz, f);
    for (int xt = XSTF; xt <= XS3N; ++xt)
        std::fwrite(xs[static_cast<XSTYPE>(xt)].data(), sizeof(double), ngn, f);
    std::fwrite(xs.xssm.data(), sizeof(double), static_cast<size_t>(ng) * ng * nxyz, f);
    std::fwrite(iden.data(), sizeof(double), niso * static_cast<size_t>(nxyz), f);
    std::fwrite(g.Phif(), sizeof(double), static_cast<size_t>(nxyz) * ng, f);
    std::fclose(f);
}

bool XSSet::PrepareXeDeviceCall(double power, double relax, xsrecon::BatchView& view,
                                std::array<double, xsrecon::NISO>& dep_i135,
                                std::array<double, xsrecon::NISO>& dep_xe135) {
    using namespace Isotope;

    // The kernel fixes NG/NISO at compile time (registers, full unroll); a
    // deck outside those dimensions falls back before touching the device.
    if (_g.ng() != xsrecon::NG || static_cast<int>(niso) != xsrecon::NISO)
        return false;

    if (!_xsrecon_backend)
        _xsrecon_backend = std::make_unique<XsReconBackend>();
    if (!_xsrecon_backend->available()) {
        static std::once_flag warn_once;
        std::call_once(warn_once, [this] {
            std::cerr << "[RASBERY][WARN][xsrecon] a device Xe arm was requested "
                         "(RASBERY_GPU_XSRECON / RASBERY_GPU_FLATXS / RASBERY_GPU_XE) "
                         "but the device path is unavailable ("
                      << _xsrecon_backend->status() << ") -- CPU loop\n";
        });
        return false;
    }

    const int nxyz = _g.nxyz();
    // One owner for the list (see XSSet::fuel_nodes): the Anderson arm indexes
    // its Xe-chain vectors by the SAME ordinals this kernel batches over, so a
    // second, independently-built list would be a silent layout fork.
    if (fuel_nodes().empty())
        return false; // no fuel: the CPU loop is an equally empty pass

    // Page-lock the host arrays every call memcpys (~6 MB/call, thousands of
    // calls per case); pageable async copies block, pinned ones stream.
    if (!_xsrecon_pinned) {
        const size_t ngn = static_cast<size_t>(_g.ng()) * nxyz;
        const size_t ssn = static_cast<size_t>(_g.ng()) * _g.ng() * nxyz;
        // Deliberately the SAME bases at the SAME byte counts as the flatxs arm
        // above: identical requests deduplicate into one lease (one owner base),
        // where a narrower or wider repeat would be a Sec 6.2 refusal and drop
        // the buffer to pageable copies for the rest of the run.
        for (int xt = 0; xt < xsrecon::NXS; ++xt) {
            const auto t = static_cast<XSTYPE>(xt);
            XsReconBackend::pinHost(_micx[t].data(), xsrecon::NISO * ngn * sizeof(double),
                                    "xs.micx@xsrecon");
            XsReconBackend::pinHost(_lmpx[t].data(), ngn * sizeof(double), "xs.lmpx@xsrecon");
            XsReconBackend::pinHost(_xs[t].data(), ngn * sizeof(double), "xs.xs@xsrecon");
        }
        XsReconBackend::pinHost(_micx.xssm.data(), xsrecon::NISO * ssn * sizeof(double),
                                "xs.micx_ssm@xsrecon");
        XsReconBackend::pinHost(_lmpx.xssm.data(), ssn * sizeof(double), "xs.lmpx_ssm@xsrecon");
        XsReconBackend::pinHost(_xs.xssm.data(), ssn * sizeof(double), "xs.xssm@xsrecon");
        XsReconBackend::pinHost(_iden.data(),
                                static_cast<size_t>(xsrecon::NISO) * nxyz * sizeof(double),
                                "xs.iden@xsrecon");
        XsReconBackend::pinHost(_g.Phif(), ngn * sizeof(double), "geom.phif@xsrecon");
        _xsrecon_pinned = true;
    }

    for (int j = 0; j < xsrecon::NISO; ++j) {
        dep_i135[static_cast<size_t>(j)]  = depTrans(iI135, static_cast<size_t>(j));
        dep_xe135[static_cast<size_t>(j)] = depTrans(iXe135, static_cast<size_t>(j));
    }

    for (int xt = 0; xt < xsrecon::NXS; ++xt) {
        const auto t = static_cast<XSTYPE>(xt);
        view.mic[xt] = _micx[t].data();
        view.lmp[xt] = _lmpx[t].data();
        view.xs[xt]  = _xs[t].data();
    }
    view.mic_ssm     = _micx.xssm.data();
    view.lmp_ssm     = _lmpx.xssm.data();
    view.xs_ssm      = _xs.xssm.data();
    view.iden        = _iden.data();
    view.phif        = _g.Phif();
    view.fuel        = _fuel_nodes.data();
    view.n_fuel      = static_cast<int>(_fuel_nodes.size());
    view.nxyz        = nxyz;
    view.norm_factor = NormFactor(power);
    view.relax       = relax;
    view.dep_i135    = dep_i135.data();
    view.dep_xe135   = dep_xe135.data();
    return true;
}

bool XSSet::TryUpdateEquilibriumXenonGpu(double power, double relax, double& max_change) {
    std::array<double, xsrecon::NISO> dep_i135{}, dep_xe135{};
    xsrecon::BatchView                v{};
    if (!PrepareXeDeviceCall(power, relax, v, dep_i135, dep_xe135))
        return false;

    // Same reason as TryUpdateFlatXSGpu: this arm downloads into the host _xs
    // columns and UpdateEquilibriumXenon's GPU branch returns BEFORE the
    // _hoststate_generation bump, so this counter is the only announcement the
    // constants gate can see.
    noteMacroXsWrite();
    return _xsrecon_backend->solve(v, _micx_generation, _hoststate_generation,
                                   &max_change);
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 13 -- the SPLIT device Xe arm (RASBERY_GPU_XE)
// ---------------------------------------------------------------------------
//
// The header on the raw fixed-point API below says the device arm "fuses
// evaluate + apply + reconstruct into a single kernel, so there is no
// evaluate-only device entry point to borrow".  THAT IS WHAT THESE ADD.  They
// do not replace the fused arm -- it is still what RASBERY_GPU_XSRECON runs,
// bit for bit -- they add the seam, out of the same node body, so the Anderson
// cascade can run without the host evaluating a 39-isotope condensation over
// every fuel node on every step.
//
// WHAT STAYS ON THE HOST, AND WHY.  The 2x2 normal equations and all four
// safeguards: they are eight doubles of arithmetic, they are the part a reader
// has to be able to check against Sec 10, and running them on the host means
// the device and host Anderson arms differ in EXACTLY ONE PLACE -- the inner
// products.  That is what makes the N1 classification a statement about one
// identified thing rather than about "the algebra moved".

bool XSSet::XeGpuEvaluate(double power, double& picard) {
    // THE SAME TWO TERMS UpdateEquilibriumXenon returns 0.0 on, and refusing
    // here is behaviourally the same thing.  The host Anderson arm would run an
    // IDENTITY evaluation instead (EvaluateEquilibriumXenon seeds its outputs
    // with the snapshot and returns 0.0), record a zero difference column, and
    // then fail its own arming test because 0.0 is under the tolerance -- so on
    // both arms the outcome is "no candidate, this step is a plain Picard step",
    // and a cascade at zero power converges on its first step either way.  What
    // differs is a history the arming test can never reach.  The `depDecay`
    // term is not optional: with no depletion data depTrans has no rows and the
    // kernel's dep_i135/dep_xe135 would be built out of nothing.
    if (depDecay.size() == 0 || power <= 0.0)
        return false;
    std::array<double, xsrecon::NISO> dep_i135{}, dep_xe135{};
    xsrecon::BatchView                v{};
    if (!PrepareXeDeviceCall(power, 1.0, v, dep_i135, dep_xe135))
        return false;
    xsphase::Scope eqxe_scope(xsphase::tallies().eqxe,
                              static_cast<std::uint64_t>(v.n_fuel));
    // Nothing is written: no _iden row, no _xs entry, no generation bump, no
    // node reconstructed -- the same contract EvaluateEquilibriumXenon keeps.
    return _xsrecon_backend->xeEvaluate(v, _micx_generation, _hoststate_generation,
                                        &picard);
}

bool XSSet::XeGpuRotateHistory() {
    return _xsrecon_backend && _xsrecon_backend->xeRotateHistory();
}

bool XSSet::XeGpuRecordColumn(int col) {
    return _xsrecon_backend && _xsrecon_backend->xeRecordColumn(col);
}

bool XSSet::XeGpuSaveEvaluation() {
    return _xsrecon_backend && _xsrecon_backend->xeSaveEvaluation();
}

bool XSSet::XeGpuDots(int ncol, double* out_six) {
    return _xsrecon_backend && _xsrecon_backend->xeDots(ncol, out_six);
}

bool XSSet::XeGpuCandidate(const double* gamma, int ncol, double& step,
                           bool& physics_ok) {
    return _xsrecon_backend &&
           _xsrecon_backend->xeCandidate(gamma, ncol, &step, &physics_ok);
}

bool XSSet::XeGpuCommitCandidate(double power) {
    std::array<double, xsrecon::NISO> dep_i135{}, dep_xe135{};
    xsrecon::BatchView                v{};
    if (!PrepareXeDeviceCall(power, 1.0, v, dep_i135, dep_xe135))
        return false;
    xsphase::Scope recon_scope(xsphase::tallies().eqxe_recon,
                               static_cast<std::uint64_t>(v.n_fuel));
    // The download makes the host arrays equal to the device copy again, so --
    // exactly like the fused arm -- this returns BEFORE any host-state
    // generation bump and the macro-XS write counter is the only announcement.
    noteMacroXsWrite();
    return _xsrecon_backend->xeCommit(v, xe::XE_T_CAND, 1.0, /*picard_skip=*/false,
                                      _hoststate_generation);
}

bool XSSet::XeGpuCommitPicard(double power, double relax) {
    std::array<double, xsrecon::NISO> dep_i135{}, dep_xe135{};
    xsrecon::BatchView                v{};
    if (!PrepareXeDeviceCall(power, relax, v, dep_i135, dep_xe135))
        return false;
    xsphase::Scope recon_scope(xsphase::tallies().eqxe_recon,
                               static_cast<std::uint64_t>(v.n_fuel));
    noteMacroXsWrite();
    return _xsrecon_backend->xeCommit(v, xe::XE_T_F, relax, /*picard_skip=*/true,
                                      _hoststate_generation);
}

bool XSSet::TryUpdateEquilibriumXenonGpuSplit(double power, double relax,
                                              double& max_change) {
    // Evaluate then commit, which composes to EXACTLY the fused body: the
    // seam between them carries doubles and nothing else, so no rounding can
    // hide in it.  A failure of the second half after the first is the one
    // asymmetry -- the evaluate wrote nothing, but a commit whose download
    // failed may have left the host arrays half refreshed.  That is the same
    // exposure solve() has had since it shipped, and it is a CUDA error, not a
    // control-flow path: the instance is dead by then and every later call
    // fails open.
    if (!XeGpuEvaluate(power, max_change))
        return false;
    return XeGpuCommitPicard(power, relax);
}

double XSSet::UpdateEquilibriumXenon(double power, double relax) {
    using namespace Isotope;

    if (depDecay.size() == 0 || power <= 0.0)
        return 0.0;

    xsphase::Scope eqxe_scope(xsphase::tallies().eqxe,
                              static_cast<std::uint64_t>(_g.nxyz()));

    // Rev.7.1 Task 13.  The split arm runs the same node body through three
    // kernels instead of one, so it is the SAME numbers -- what it buys is the
    // seam the Anderson arm needs, and running the Picard cascade through it
    // too is what makes that seam bit-gateable against this very loop.  It is
    // tried first because a run that asked for RASBERY_GPU_XE asked for the
    // split kernels; with the flag unset not one line of this is reached.
    if (rasberyGpuXeEnabled()) {
        double          gpu_max = 0.0;
        xe::XeGpuTally& tally   = xe::xeGpuTally();
        // Charged BEFORE the attempt, so xe_updates counts steps ASKED FOR and
        // device_updates + host_fallbacks always adds up to it.  A receipt whose
        // parts do not sum to its whole cannot be used to find the fallbacks.
        tally.xe_updates.fetch_add(1, std::memory_order_relaxed);
        if (TryUpdateEquilibriumXenonGpuSplit(power, relax, gpu_max)) {
            tally.device_updates.fetch_add(1, std::memory_order_relaxed);
            xsreconDebugHash(_xs, _iden, _g.ng(), _g.nxyz(), gpu_max);
            return gpu_max;
        }
        tally.host_fallbacks.fetch_add(1, std::memory_order_relaxed);
        // WP1 (plan Sec 6.3).  The split arm is the one Xe seam that was
        // already counted; the guard rides on the same line so the two cannot
        // drift apart.
        RASBERY_GPU_FULL_GUARD(Xe, "XSSet::UpdateEquilibriumXenon(split)",
                               "the split device Xe arm declined");
        // any failure falls through to the fused arm, then to the CPU loop
    }

    if (rasberyGpuXsReconEnabled()) {
        double          gpu_max = 0.0;
        xe::XeGpuTally& tally   = xe::xeGpuTally();
        // F10 (review doc Sec 3).  Charged BEFORE the attempt, the same way the
        // split arm above charges xe_updates, so
        // `fused_updates == fused_device_updates + fused_host_fallbacks` is an
        // accounting identity and a receipt whose parts do not sum to its whole
        // is a bug rather than a reading.
        tally.fused_updates.fetch_add(1, std::memory_order_relaxed);
        if (TryUpdateEquilibriumXenonGpu(power, relax, gpu_max)) {
            tally.fused_device_updates.fetch_add(1, std::memory_order_relaxed);
            xsreconDebugHash(_xs, _iden, _g.ng(), _g.nxyz(), gpu_max);
            return gpu_max;
        }
        tally.fused_host_fallbacks.fetch_add(1, std::memory_order_relaxed);
        // any failure falls through to the unchanged CPU loop
        // WP1 (plan Sec 6.3).  Unlike the split arm above, this one had NO
        // tally at all -- the fused RASBERY_GPU_XSRECON arm could decline every
        // step of every statepoint and no receipt would say so.  The counter on
        // the line above is F10's half of that; the guard is WP1's.
        RASBERY_GPU_FULL_GUARD(Xe, "XSSet::UpdateEquilibriumXenon(fused)",
                               "the fused device Xe arm declined; the host Xe loop "
                               "runs");
    }

    static const char* dump_path = std::getenv("RASBERY_XSRECON_DUMP");
    static bool        dump_done = false;
    const bool         dump_this = (dump_path != nullptr && !dump_done);
    if (dump_this) {
        dump_done = true;
        xsreconDumpArrays((std::string(dump_path) + ".in").c_str(), _g, _micx,
                          _lmpx, _xs, _iden, NormFactor(power), relax);
    }

    const int    ng          = _g.ng();
    const int    nxyz        = _g.nxyz();
    const size_t niso        = Isotope::niso;
    const double norm_factor = NormFactor(power);
    double       max_change  = 0.0;

#pragma omp parallel if (nxyz > OMP_THRESHOLD) reduction(max : max_change)
    {
        static thread_local DepletionWorkspace  ws_tls;
        static thread_local std::vector<double> abs_flux_tls;

        ws_tls.ensure(niso);
        if (abs_flux_tls.size() != static_cast<size_t>(ng))
            abs_flux_tls.resize(static_cast<size_t>(ng));
        if (ws_tls.condensed.size() < niso * N_XS_SCALAR)
            ws_tls.condensed.resize(niso * N_XS_SCALAR, 0.0);

        const double* mic_ptrs[N_XS_SCALAR] = {
            _micx.xstf.data(), _micx.xsdf.data(), _micx.xsaf.data(), _micx.xsff.data(),
            _micx.xsnf.data(), _micx.xskf.data(), _micx.xssf.data(), _micx.xsrf.data(),
            _micx.fyld.data(), _micx.xs2n.data(), _micx.xs3n.data()};

#pragma omp for schedule(dynamic, 8)
        for (int l = 0; l < nxyz; ++l) {
            if (!_g.IsFuel(l))
                continue;

            double raw_sumflux = 0.0;
            for (int ig = 0; ig < ng; ++ig) {
                abs_flux_tls[static_cast<size_t>(ig)] =
                    _g.Phif()[l * ng + ig] * norm_factor;
                raw_sumflux += abs_flux_tls[static_cast<size_t>(ig)];
            }
            if (raw_sumflux <= 0.0)
                continue;

            const double invflux = 1.0 / raw_sumflux;
            xsphase::Scope condense_scope(xsphase::tallies().eqxe_condense, 1);
            for (size_t iso = 0; iso < niso; ++iso) {
                double* dst = ws_tls.condensed.data() + iso * N_XS_SCALAR;
                for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
                    double sum = 0.0;
                    for (int ig = 0; ig < ng; ++ig) {
                        const size_t off =
                            (iso * static_cast<size_t>(ng) + static_cast<size_t>(ig)) *
                                static_cast<size_t>(nxyz) +
                            static_cast<size_t>(l);
                        sum += mic_ptrs[xt][off] * abs_flux_tls[static_cast<size_t>(ig)];
                    }
                    dst[xt] = sum * invflux;
                }
                ws_tls.iden[iso] = _iden[iso * static_cast<size_t>(nxyz) +
                                         static_cast<size_t>(l)];
            }

            const double old_i   = ws_tls.iden[iI135];
            const double old_xe  = ws_tls.iden[iXe135];
            const double old_xem = ws_tls.iden[iXe135m];
            ApplyXeEquilibrium(ws_tls.iden, ws_tls.condensed,
                               FluxScale(abs_flux_tls.data(), ng));
            const double new_xe = ws_tls.iden[iXe135];
            const double scale  = std::max(std::abs(new_xe), 1.0e-30);
            // Measured on the RAW step, before damping -- see the header.
            max_change          = std::max(max_change, std::abs(new_xe - old_xe) / scale);

            // x <- x + relax*(F(x) - x).  Same fixed point, damped path.  The
            // `relax < 1.0` guard keeps the undamped case arithmetically
            // identical to what it was rather than relying on 1.0*(a-b)+b == a.
            if (relax < 1.0) {
                ws_tls.iden[iI135]   = old_i + relax * (ws_tls.iden[iI135] - old_i);
                ws_tls.iden[iXe135]  = old_xe + relax * (ws_tls.iden[iXe135] - old_xe);
                ws_tls.iden[iXe135m] = old_xem + relax * (ws_tls.iden[iXe135m] - old_xem);
            }

            _iden[iI135 * static_cast<size_t>(nxyz) + static_cast<size_t>(l)] =
                ws_tls.iden[iI135];
            _iden[iXe135 * static_cast<size_t>(nxyz) + static_cast<size_t>(l)] =
                ws_tls.iden[iXe135];
            _iden[iXe135m * static_cast<size_t>(nxyz) + static_cast<size_t>(l)] =
                ws_tls.iden[iXe135m];

            condense_scope.stop();

            // Only fuel-node Xe-chain densities changed.  Reconstruct this
            // node while its data is hot instead of opening a second
            // all-node pass after the equilibrium update.
            {
                xsphase::Scope recon_scope(xsphase::tallies().eqxe_recon, 1);
                ReconstructNode(static_cast<size_t>(l));
            }
        }
    }
    if (dump_this) {
        std::FILE* f = std::fopen((std::string(dump_path) + ".out").c_str(), "wb");
        if (f) {
            const size_t ngn = static_cast<size_t>(ng) * nxyz;
            for (int xt = XSTF; xt <= XS3N; ++xt)
                std::fwrite(_xs[static_cast<XSTYPE>(xt)].data(), sizeof(double), ngn, f);
            std::fwrite(_xs.xssm.data(), sizeof(double),
                        static_cast<size_t>(ng) * ng * nxyz, f);
            std::fwrite(_iden.data(), sizeof(double), niso * static_cast<size_t>(nxyz), f);
            std::fwrite(&max_change, sizeof(double), 1, f);
            std::fclose(f);
        }
    }
    ++_hoststate_generation; // CPU arm wrote _xs and the Xe-chain _iden rows
    xsreconDebugHash(_xs, _iden, ng, nxyz, max_change);
    return max_change;
}

// ---------------------------------------------------------------------------
// Raw fixed-point API for the safeguarded Anderson arm (plan Rev.4 Sec 10.1).
//
// UpdateEquilibriumXenon above is ONE fused operation: it evaluates the
// closed-form image F(x), damps it, writes the three Xe-chain _iden rows and
// reconstructs the touched nodes.  Anderson needs those halves apart, because it
// has to see F(x) before it can decide what to commit:
//
//   SnapshotXenon             the current iterate x, read-only
//   EvaluateEquilibriumXenon  F(x) and the raw residual, SIDE-EFFECT FREE
//   CommitXenon               write an accepted x and reconstruct
//
// UpdateEquilibriumXenon itself is untouched, so with the Anderson gate unset
// (the default) not one byte of the production path changes -- these three are
// simply never called.
//
// FUEL-NODE ORDINALS, NOT NODE INDICES.  The three vectors all of them take and
// return are indexed 0 .. fuel_nodes().size()-1, in the order fuel_nodes() lists
// them (ascending node index, built once, geometry-fixed).  A caller therefore
// never has to know the SoA stride, and Evaluate and Commit cannot disagree
// about the layout because they read the same list.
//
// HOST ONLY, DELIBERATELY.  The device arm (XsReconKernel.h) fuses evaluate +
// damp + apply + reconstruct into a single kernel, so there is no evaluate-only
// device entry point to borrow and splitting the kernel would fork the bit-exact
// A/B this campaign already gated.  Evaluate therefore always runs the host
// closed form, even with RASBERY_GPU_XSRECON set.  That is affordable: the map
// is ~8.5k nodes of micro-XS condensation, one to two orders below the flux
// re-convergence an accelerated cascade removes.  Commit writes the host _iden
// and bumps _hoststate_generation, which is exactly the signal the device
// backend re-uploads its resident copy on, so the two arms cannot drift.
// ---------------------------------------------------------------------------

const std::vector<int>& XSSet::fuel_nodes() {
    if (_fuel_nodes.empty()) {
        const int nxyz = _g.nxyz();
        for (int l = 0; l < nxyz; ++l)
            if (_g.IsFuel(l))
                _fuel_nodes.push_back(l);
    }
    return _fuel_nodes;
}

void XSSet::SnapshotXenon(std::vector<double>& iodine_out, std::vector<double>& xenon_out,
                          std::vector<double>& xe135m_out) {
    using namespace Isotope;

    const std::vector<int>& fuel = fuel_nodes();
    const size_t            nf   = fuel.size();
    const size_t            nxyz = static_cast<size_t>(_g.nxyz());
    iodine_out.resize(nf);
    xenon_out.resize(nf);
    xe135m_out.resize(nf);
    for (size_t k = 0; k < nf; ++k) {
        const size_t l = static_cast<size_t>(fuel[k]);
        iodine_out[k]  = _iden[iI135 * nxyz + l];
        xenon_out[k]   = _iden[iXe135 * nxyz + l];
        xe135m_out[k]  = _iden[iXe135m * nxyz + l];
    }
}

double XSSet::EvaluateEquilibriumXenon(double power, std::vector<double>& iodine_out,
                                       std::vector<double>& xenon_out,
                                       std::vector<double>& xe135m_out) {
    using namespace Isotope;

    // Identity seed.  F(x) = x on every node the production loop SKIPS -- no
    // depletion data, zero power, or a node whose normalized flux is not
    // positive -- so committing this image writes back exactly what is already
    // there, and a skipped node contributes nothing to the residual.  Same
    // no-op semantics the fused update has, expressed as data.
    SnapshotXenon(iodine_out, xenon_out, xe135m_out);
    if (depDecay.size() == 0 || power <= 0.0)
        return 0.0;

    const std::vector<int>& fuel = fuel_nodes();
    const int               nf   = static_cast<int>(fuel.size());
    if (nf == 0)
        return 0.0;

    xsphase::Scope eqxe_scope(xsphase::tallies().eqxe, static_cast<std::uint64_t>(nf));

    const int    ng          = _g.ng();
    const int    nxyz        = _g.nxyz();
    const size_t niso        = Isotope::niso;
    const double norm_factor = NormFactor(power);
    double       max_change  = 0.0;

#pragma omp parallel if (nf > OMP_THRESHOLD) reduction(max : max_change)
    {
        static thread_local DepletionWorkspace  ws_tls;
        static thread_local std::vector<double> abs_flux_tls;

        ws_tls.ensure(niso);
        if (abs_flux_tls.size() != static_cast<size_t>(ng))
            abs_flux_tls.resize(static_cast<size_t>(ng));
        if (ws_tls.condensed.size() < niso * N_XS_SCALAR)
            ws_tls.condensed.resize(niso * N_XS_SCALAR, 0.0);

        const double* mic_ptrs[N_XS_SCALAR] = {
            _micx.xstf.data(), _micx.xsdf.data(), _micx.xsaf.data(), _micx.xsff.data(),
            _micx.xsnf.data(), _micx.xskf.data(), _micx.xssf.data(), _micx.xsrf.data(),
            _micx.fyld.data(), _micx.xs2n.data(), _micx.xs3n.data()};

#pragma omp for schedule(dynamic, 8)
        for (int k = 0; k < nf; ++k) {
            const int l = fuel[static_cast<size_t>(k)];

            double raw_sumflux = 0.0;
            for (int ig = 0; ig < ng; ++ig) {
                abs_flux_tls[static_cast<size_t>(ig)] =
                    _g.Phif()[l * ng + ig] * norm_factor;
                raw_sumflux += abs_flux_tls[static_cast<size_t>(ig)];
            }
            if (raw_sumflux <= 0.0)
                continue;

            const double invflux = 1.0 / raw_sumflux;
            xsphase::Scope condense_scope(xsphase::tallies().eqxe_condense, 1);
            for (size_t iso = 0; iso < niso; ++iso) {
                double* dst = ws_tls.condensed.data() + iso * N_XS_SCALAR;
                for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
                    double sum = 0.0;
                    for (int ig = 0; ig < ng; ++ig) {
                        const size_t off =
                            (iso * static_cast<size_t>(ng) + static_cast<size_t>(ig)) *
                                static_cast<size_t>(nxyz) +
                            static_cast<size_t>(l);
                        sum += mic_ptrs[xt][off] * abs_flux_tls[static_cast<size_t>(ig)];
                    }
                    dst[xt] = sum * invflux;
                }
                ws_tls.iden[iso] = _iden[iso * static_cast<size_t>(nxyz) +
                                         static_cast<size_t>(l)];
            }

            const double             old_xe = ws_tls.iden[iXe135];
            const XeEquilibriumImage img =
                ComputeXeEquilibrium(ws_tls.iden, ws_tls.condensed,
                                     FluxScale(abs_flux_tls.data(), ng));
            const double scale = std::max(std::abs(img.xe135), 1.0e-30);
            // THE SAME metric UpdateEquilibriumXenon returns: raw, measured on
            // the undamped image, |F(x)-x|/|F(x)| over Xe-135.  The Anderson
            // trust region compares its candidate against this number, so the
            // two have to be the same measurement or the comparison is meaningless.
            max_change = std::max(max_change, std::abs(img.xe135 - old_xe) / scale);

            iodine_out[static_cast<size_t>(k)] = img.i135;
            xenon_out[static_cast<size_t>(k)]  = img.xe135;
            xe135m_out[static_cast<size_t>(k)] = img.xe135m;
        }
    }
    // Nothing was written: no _iden row, no _xs entry, no generation bump, no
    // node reconstructed.  That is the whole contract of this function -- a
    // caller that rejects the image leaves the solver exactly as it found it.
    return max_change;
}

void XSSet::CommitXenon(const std::vector<double>& iodine, const std::vector<double>& xenon,
                        const std::vector<double>& xe135m) {
    using namespace Isotope;

    const std::vector<int>& fuel = fuel_nodes();
    const size_t            nf   = fuel.size();
    if (nf == 0)
        return;
    // A short vector would commit a partial inventory and leave the rest of the
    // core on the previous iterate -- a silently mixed Xe state that no
    // downstream check could attribute.  Refuse instead of publishing it.
    if (iodine.size() != nf || xenon.size() != nf || xe135m.size() != nf)
        throw std::runtime_error(
            "XSSet::CommitXenon: Xe-chain vector length does not match the fuel-node count");

    const int    nfi  = static_cast<int>(nf);
    const size_t nxyz = static_cast<size_t>(_g.nxyz());

#pragma omp parallel for schedule(dynamic, 8) if (nfi > OMP_THRESHOLD)
    for (int k = 0; k < nfi; ++k) {
        const size_t idx = static_cast<size_t>(k);
        const size_t l   = static_cast<size_t>(fuel[idx]);
        // EXACTLY the three Xe-chain rows ApplyXeEquilibrium owns, and nothing
        // else: the rest of the isotope vector belongs to depletion.
        _iden[iI135 * nxyz + l]   = iodine[idx];
        _iden[iXe135 * nxyz + l]  = xenon[idx];
        _iden[iXe135m * nxyz + l] = xe135m[idx];
        // The same per-node reconstruction the fused update does, into the same
        // timing bucket: only fuel-node Xe-chain densities moved, and
        // ReconstructNode reads nothing but node l.
        xsphase::Scope recon_scope(xsphase::tallies().eqxe_recon, 1);
        ReconstructNode(l);
    }
    ++_hoststate_generation; // host wrote _xs and the Xe-chain _iden rows
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

    const double sumflux = FluxScale(abs_flux, static_cast<int>(ngrp));

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
    const double* const flux = _g.Phif();

    const double norm_factor = NormFactor(power);

    // GA evaluator plan Task 16.  RASBERY_GPU_CRAM=1 runs the whole node loop on
    // the device; anything else -- arm off, no CUDA, ng != 2, no depletion data,
    // a CUDA failure, or ANY node that hit milk.h's two throw conditions --
    // returns false having written nothing, and the host loop below runs exactly
    // as it did before.
    if (DepleteGpu(dt, power, xe_transient)) {
        ++_hoststate_generation; // depletion rewrites _iden
        return;
    }
    ++_cram_host_fallbacks;
    // WP1 (plan Sec 6.3).  Guarded BEFORE the OpenMP region below: an exception
    // must not have to leave a parallel region to reach main.cpp's per-case
    // catch.
    RASBERY_GPU_FULL_GUARD_IF(rasbery::gpufull::armRequested("RASBERY_GPU_CRAM"), Cram,
                              "XSSet::Deplete",
                              "the device CRAM predictor declined; the host DepleteNode "
                              "loop runs");

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
    ++_hoststate_generation; // depletion rewrites _iden
}

void XSSet::DecayIsotopeDensityFlat(
    std::vector<double>& idenFlat, int nxyz, double coolingDays,
    int substeps) const {
    using namespace Isotope;
    const size_t isotopeCount = Isotope::niso;
    if (coolingDays <= 0.0 || nxyz <= 0 || isotopeCount == 0 ||
        depDecay.size() == 0)
        return;
    if (idenFlat.size() < static_cast<size_t>(nxyz) * isotopeCount)
        throw std::runtime_error(
            "XSSet: restart isotope density is smaller than nxyz*niso.");

    const int    decaySteps = std::max(1, substeps);
    const double dt =
        coolingDays * 86400.0 / static_cast<double>(decaySteps);

#pragma omp parallel if (nxyz > OMP_THRESHOLD)
    {
        static thread_local DepletionWorkspace workspace;
        workspace.ensure(isotopeCount);

#pragma omp for schedule(dynamic, 8)
        for (int l = 0; l < nxyz; ++l) {
            const size_t offset = static_cast<size_t>(l) * isotopeCount;
            for (size_t isotope = 0; isotope < isotopeCount; ++isotope)
                workspace.iden[isotope] = idenFlat[offset + isotope];

            for (int step = 0; step < decaySteps; ++step)
                milk::Solver<double>::solveBatemanCRAM(
                    depDecay, workspace.iden, dt, workspace.iden,
                    workspace.cram, CRAM_ORDER, iI135);

            for (size_t isotope = iI135; isotope < isotopeCount; ++isotope)
                idenFlat[offset + isotope] = workspace.iden[isotope];
        }
    }
}

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
    _fine_rod_thermal_fluence_bos = _fine_rod_thermal_fluence;
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

    // Predictor-corrector flavour.  Default: CELI/RASBERY midpoint scheme -- build one
    // transition matrix from the BOS/EOS *rate* average and do a single CRAM solve from
    // the BOS inventory.  RASBERY_PC_MODE=decart selects DeCART2D's Eq. (6.20) instead:
    // the corrector solves with EOS rates only, and the final inventory is the average of
    // the predictor (BOS-rate) and corrector (EOS-rate) *densities*.
    static const bool pcDensityAverage = []() {
        const char* m = std::getenv("RASBERY_PC_MODE");
        return m != nullptr && std::string(m) == "decart";
    }();
    // Isotalo-style corrector substepping (ANE 38 (2011)): split the corrector CRAM solve
    // into k substeps whose one-group rates are linearly interpolated BOS->EOS at each
    // substep midpoint.  No extra transport solves.  k=1 (default) takes the original
    // code path untouched, so both existing PC modes stay bit-identical.
    static const int pcSubsteps = []() {
        const char* m = std::getenv("RASBERY_PC_SUBSTEPS");
        const int   k = (m != nullptr) ? std::atoi(m) : 1;
        return std::clamp(k, 1, 64);
    }();
    // RASBERY_PC_XE_EQUILIBRIUM_FIX: restore the Xe fixed-point property that the
    // Eq. (6.20) density average destroys.  See the block at the end of the node
    // loop.  Opt-in for now -- with the gate unset the corrector is byte-for-byte
    // the old one -- and it does nothing outside decart mode or with transient Xe.
    static const bool pcXeEquilibriumFix = []() {
        const char* m = std::getenv("RASBERY_PC_XE_EQUILIBRIUM_FIX");
        if (m == nullptr) return false;
        const std::string s(m);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
                 s == "false" || s == "FALSE");
    }();

    const int           nxyz           = _g.nxyz();
    const int           ng             = _g.ng();
    const double        bos_norm       = NormFactor(power, _xs_bos, _flux_bos.data());
    const double        eos_norm       = NormFactor(power);
    const double* const eos_flux       = _g.Phif();
    const size_t        condensed_size = niso * N_XS_SCALAR;
    const size_t        nxyz_size      = static_cast<size_t>(nxyz);

    // GA evaluator plan Task 16.  Same fail-open contract as the predictor's,
    // plus one extra decline: a corrector whose BOS micro-XS snapshot on the
    // device did not come from THIS statepoint's device predictor.
    const bool corrector_on_device =
        CorrectorStepGpu(dt, power, xe_transient, pcDensityAverage,
                         pcXeEquilibriumFix, pcSubsteps);
    if (!corrector_on_device) ++_cram_host_fallbacks;
    // WP1 (plan Sec 6.3).  Same rule as the predictor's, and before the
    // corrector's own OpenMP region for the same reason.
    RASBERY_GPU_FULL_GUARD_IF(!corrector_on_device &&
                                  rasbery::gpufull::armRequested("RASBERY_GPU_CRAM"),
                              Cram, "XSSet::PredictorCorrectorStep",
                              "the device CRAM corrector declined; the host corrector "
                              "loop runs");

    if (!corrector_on_device) {
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

            const size_t node = static_cast<size_t>(l);
            for (size_t i = 0; i < niso; ++i)
                ws_tls.iden[i] = _iden_bos[i * nxyz_size + node];

            // The corrector half's own one-group flux scale, kept where the Eq. (6.20)
            // average below can still see it (both branches scope theirs privately).
            double xe_sumflux = 0.0;

            if (pcSubsteps == 1) {
                double raw_sumflux = 0.0;
                for (int ig = 0; ig < ng; ++ig)
                    raw_sumflux += corrected_flux_tls[ig];
                const double invflux           = (raw_sumflux > 0.0) ? 1.0 / raw_sumflux : 0.0;
                const double corrected_sumflux = FluxScale(corrected_flux_tls.data(), ng);
                xe_sumflux                     = corrected_sumflux;

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

                BuildTransitionMatrix(ws_tls.condensed, corrected_sumflux, ws_tls.matrix);
                milk::Solver<double>::solveBatemanCRAM(ws_tls.matrix, ws_tls.iden, dt, ws_tls.iden,
                                                       ws_tls.cram, CRAM_ORDER, iI135);

                if (!xe_transient)
                    ApplyXeEquilibrium(ws_tls.iden, ws_tls.condensed, corrected_sumflux);
            } else {
                // Substep chain: k CRAM solves of dt/k each, rates linearly interpolated
                // BOS->EOS at each substep midpoint.  Applies to both PC modes (in decart
                // mode the chained result replaces the pure-EOS-rate corrector before the
                // Eq. (6.20) density average below).
                static thread_local std::vector<double> sub_flux_tls;
                if (sub_flux_tls.size() != static_cast<size_t>(ng))
                    sub_flux_tls.resize(ng);
                double last_sumflux = 0.0;
                for (int sub = 0; sub < pcSubsteps; ++sub) {
                    const double w = (static_cast<double>(sub) + 0.5) / pcSubsteps;
                    double raw_sumflux = 0.0;
                    for (int ig = 0; ig < ng; ++ig) {
                        sub_flux_tls[ig] = (1.0 - w) * _flux_bos[l * ng + ig] * bos_norm +
                                           w * eos_flux[l * ng + ig] * eos_norm;
                        raw_sumflux += sub_flux_tls[ig];
                    }
                    const double invflux     = (raw_sumflux > 0.0) ? 1.0 / raw_sumflux : 0.0;
                    const double sub_sumflux = FluxScale(sub_flux_tls.data(), ng);
                    last_sumflux             = sub_sumflux;

                    for (size_t iso = 0; iso < niso; ++iso) {
                        double* dst = ws_tls.condensed.data() + iso * N_XS_SCALAR;
                        for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) {
                            double sum = 0.0;
                            for (int ig = 0; ig < ng; ++ig) {
                                const size_t off = (iso * static_cast<size_t>(ng) + static_cast<size_t>(ig)) *
                                                       nxyz_size +
                                                   node;
                                sum += ((1.0 - w) * bos_ptrs[xt][off] + w * eos_ptrs[xt][off]) *
                                       sub_flux_tls[ig];
                            }
                            dst[xt] = sum * invflux;
                        }
                    }

                    BuildTransitionMatrix(ws_tls.condensed, sub_sumflux, ws_tls.matrix);
                    milk::Solver<double>::solveBatemanCRAM(ws_tls.matrix, ws_tls.iden,
                                                           dt / pcSubsteps, ws_tls.iden,
                                                           ws_tls.cram, CRAM_ORDER, iI135);
                }

                // Xe equilibrium at the final (EOS-nearest) substep rates.
                xe_sumflux = last_sumflux;
                if (!xe_transient)
                    ApplyXeEquilibrium(ws_tls.iden, ws_tls.condensed, last_sumflux);
            }

            // ws_tls.iden now holds the corrector inventory N^C.  In DeCART mode _iden still
            // carries the predictor inventory N^P (PredictorStep wrote it, the transport solve
            // did not touch it), so average the two per Eq. (6.20); otherwise take N^C directly.
            for (size_t i = iI135; i < niso; ++i) {
                const double n_corr = ws_tls.iden[i];
                _iden[i * nxyz_size + node] =
                    pcDensityAverage ? 0.5 * (_iden[i * nxyz_size + node] + n_corr) : n_corr;
            }

            // The average above ran over EVERY row from iI135 up, the three
            // equilibrium-Xe rows included -- and both halves had already been put on
            // their own Xe equilibrium (the predictor inside Deplete, the corrector
            // just above).  The average of two fixed points is not a fixed point: the
            // published I-135/Xe-135/Xe-135m come out off the equilibrium of the
            // published (averaged) actinide inventory by roughly half the step's Xe
            // swing, so the next SolveLoop opens its re-convergence cascade two orders
            // of magnitude from the 1e-6 tolerance and spends about twice the Picard
            // steps walking back to a point the corrector had already reached.
            //
            // Re-solve the equilibrium on the AVERAGED inventory.  ApplyXeEquilibrium
            // is an explicit formula in the actinide densities, the condensed
            // one-group XS and the flux scale -- no iteration, and nothing here that
            // CorrectorStep does not already hold: ws_tls.condensed still carries the
            // corrector half's rates and xe_sumflux its flux scale, which are exactly
            // the rates the averaged burnup is about to be reconstructed at.  The
            // actinide rows are all >= iI135, so copying the averaged tail back into
            // ws_tls.iden gives the formula the published state it must be consistent
            // with; only the three Xe-chain rows are written back.
            if (pcXeEquilibriumFix && pcDensityAverage && !xe_transient) {
                for (size_t i = iI135; i < niso; ++i)
                    ws_tls.iden[i] = _iden[i * nxyz_size + node];
                ApplyXeEquilibrium(ws_tls.iden, ws_tls.condensed, xe_sumflux);
                _iden[iI135 * nxyz_size + node]   = ws_tls.iden[iI135];
                _iden[iXe135 * nxyz_size + node]  = ws_tls.iden[iXe135];
                _iden[iXe135m * nxyz_size + node] = ws_tls.iden[iXe135m];
            }

            _burn[l] = _burn_bos[l];
            if (burn >= 1.0e-10) {
                const double dfac               = 8.64e7 * (_g.vol(l) / _lib->lib_model_volu[_comp[l]]) * _lib->lib_model_hmas[_comp[l]];
                const double burn_key_increment = burn / dfac * 1000.0;
                _burn[l] += static_cast<int>(burn_key_increment + 0.5);
            }
        }
    }
    } // !corrector_on_device

    // Rebuild XS on the corrected EOS composition and burnup.  The device arm
    // stops at _iden/_burn: rod materials, branch coefficients and the flat-XS
    // rebuild are the same host calls on both paths.
    _fine_rod_thermal_fluence = _fine_rod_thermal_fluence_bos;
    DepleteRodMaterials(dt, power, true);
    PrecomputeBranchCoefficients();
    UpdateFlatXS();
}

// --- Task 16: the CRAM depletion device arm --------------------------------
//
// Three functions, and every one of them can say no.  The rule is the PPR arm's
// rule: a `false` return means NOTHING has been written, so the host loop that
// follows is the loop that always ran.  The difference from PPR is downstream,
// not here -- these densities are the next statepoint's input, which is why
// RASBERY_GPU_CRAM sits in trajectory::kArmEnv.

bool XSSet::PrepareCramLib(cram::LibView& lib) {
    using namespace Isotope;

    const int nxyz = _g.nxyz();
    if (nxyz <= 0) return false;
    // depDecay/depTrans are the process-wide Chiffon parse.  A deck without
    // them has no depletion at all; a deck whose registry is not the canonical
    // 39 is not the registry the device pattern was mined from.
    if (depDecay.rows() != niso || depDecay.cols() != niso ||
        depTrans.rows() != niso || depTrans.cols() != niso)
        return false;

    if (_cram_dfac.size() != static_cast<size_t>(nxyz)) {
        _cram_dfac.resize(static_cast<size_t>(nxyz));
        _cram_vol.resize(static_cast<size_t>(nxyz));
        for (int l = 0; l < nxyz; ++l) {
            _cram_vol[static_cast<size_t>(l)] = _g.vol(l);
            // EXACTLY the host expression from CorrectorStep, evaluated here so
            // the device reads a number the host produced rather than a
            // re-derivation of it in a different association.
            _cram_dfac[static_cast<size_t>(l)] =
                8.64e7 * (_g.vol(l) / _lib->lib_model_volu[_comp[l]]) *
                _lib->lib_model_hmas[_comp[l]];
        }
        ++_cram_lib_generation;
    }

    lib.generation = _cram_lib_generation;
    lib.niso       = static_cast<int>(niso);
    lib.nxs        = N_XS_SCALAR;
    lib.first      = static_cast<int>(iI135);
    lib.ac_first   = static_cast<int>(iAcFirst);
    lib.ac_last    = static_cast<int>(iAcLast);
    lib.i135       = static_cast<int>(iI135);
    lib.xe135      = static_cast<int>(iXe135);
    lib.xe135m     = static_cast<int>(iXe135m);
    lib.u234       = static_cast<int>(iU234);
    lib.u235       = static_cast<int>(iU235);
    lib.u238       = static_cast<int>(iU238);
    lib.np237      = static_cast<int>(iNp237);
    lib.dep_decay  = depDecay.data();
    lib.dep_trans  = depTrans.data();
    lib.dfac       = _cram_dfac.data();
    lib.vol        = _cram_vol.data();
    return true;
}

bool XSSet::DepleteGpu(double dt, double power, bool xe_transient) {
    CramBackend& g = cram();
    if (!g.available()) return false;
    // DepleteNode's own first line: with no depletion data the host body is a
    // no-op, and a device arm that "succeeded" on it would skip a no-op while
    // pretending it ran.
    if (depDecay.size() == 0) return false;

    cram::LibView lib;
    if (!PrepareCramLib(lib)) return false;

    cram::PredictorView v;
    v.ng              = _g.ng();
    v.nxyz            = _g.nxyz();
    v.dt              = dt;
    v.norm_factor     = NormFactor(power);
    v.xe_transient    = xe_transient ? 1 : 0;
    v.micx_generation = _micx_generation;
    v.phif            = _g.Phif();
    v.iden            = _iden.data();
    // The same eleven pointers, in the same order, DepleteNode builds.  The
    // backend uploads only the four slots the transition matrix and the Xe
    // equilibrium actually read; the list stays complete so a future reader of
    // a fifth slot is a compile-time edit here, not a silent wrong answer.
    const double* mic_ptrs[N_XS_SCALAR] = {
        _micx.xstf.data(), _micx.xsdf.data(), _micx.xsaf.data(), _micx.xsff.data(),
        _micx.xsnf.data(), _micx.xskf.data(), _micx.xssf.data(), _micx.xsrf.data(),
        _micx.fyld.data(), _micx.xs2n.data(), _micx.xs3n.data()};
    for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) v.mic[xt] = mic_ptrs[xt];

    // Retire the previous statepoint's token BEFORE the call: a predictor that
    // declines must not leave a token this statepoint's corrector can match.
    _cram_bos_token          = 0;
    unsigned long long token = 0;
    if (!g.predictor(lib, v, &token)) return false;
    _cram_bos_token = token;
    return true;
}

bool XSSet::CorrectorStepGpu(double dt, double power, bool xe_transient,
                             bool density_average, bool xe_equilibrium_fix,
                             int substeps) {
    // The Isotalo substep chain is a second device path nobody has measured.
    // Declining is the cheap, honest answer; see CudaCramBackend.h.
    if (substeps != 1) return false;
    CramBackend& g = cram();
    if (!g.available()) return false;
    if (depDecay.size() == 0) return false;
    if (_cram_bos_token == 0) return false; // this statepoint's predictor was host-side

    cram::LibView lib;
    if (!PrepareCramLib(lib)) return false;

    cram::CorrectorView v;
    v.ng                 = _g.ng();
    v.nxyz               = _g.nxyz();
    v.dt                 = dt;
    v.bos_norm           = NormFactor(power, _xs_bos, _flux_bos.data());
    v.eos_norm           = NormFactor(power);
    v.xe_transient       = xe_transient ? 1 : 0;
    v.density_average    = density_average ? 1 : 0;
    v.xe_equilibrium_fix = xe_equilibrium_fix ? 1 : 0;
    v.micx_generation    = _micx_generation;
    v.bos_token          = _cram_bos_token;
    v.flux_bos           = _flux_bos.data();
    v.flux_eos           = _g.Phif();
    v.xskf_bos           = _xs_bos.xskf.data();
    v.xskf_eos           = _xs.xskf.data();
    v.iden_bos           = _iden_bos.data();
    v.iden               = _iden.data();
    v.burn_bos           = _burn_bos.data();
    v.burn               = _burn.data();

    const double* mic_ptrs[N_XS_SCALAR] = {
        _micx.xstf.data(), _micx.xsdf.data(), _micx.xsaf.data(), _micx.xsff.data(),
        _micx.xsnf.data(), _micx.xskf.data(), _micx.xssf.data(), _micx.xsrf.data(),
        _micx.fyld.data(), _micx.xs2n.data(), _micx.xs3n.data()};
    for (size_t xt = 0; xt < N_XS_SCALAR; ++xt) v.mic[xt] = mic_ptrs[xt];

    if (!g.corrector(lib, v)) return false;
    _cram_bos_token = 0; // one corrector consumes one predictor's snapshot
    return true;
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
        _fine_rod_thermal_fluence.assign(expected_size, 0.0);
        _fine_rod_thermal_fluence_bos.assign(expected_size, 0.0);
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

void XSSet::AccumulatePuHistory() {
    const int    nxyz = _g.nxyz();
    const size_t iso  = Isotope::iPu239;
    if (_pu_prev.size() != static_cast<size_t>(nxyz))
        _pu_prev.assign(static_cast<size_t>(nxyz), -1.0);
    if (_pu_gain_tot.size() != static_cast<size_t>(nxyz)) {
        _pu_gain_tot.assign(static_cast<size_t>(nxyz), 0.0);
        _pu_gain_rod.assign(static_cast<size_t>(nxyz), 0.0);
    }
    for (int l = 0; l < nxyz; ++l) {
        const double now = _iden[iso * static_cast<size_t>(nxyz) + l];
        const double was = _pu_prev[l];
        if (was >= 0.0) {
            const double gain = now - was;
            if (gain > 0.0) {
                _pu_gain_tot[l] += gain;
                _pu_gain_rod[l] += gain * _g.rod_fraction(l);
            }
        }
        _pu_prev[l] = now;
    }
}

double XSSet::RoddedPuFraction(int l) const {
    if (_pu_gain_tot.size() <= static_cast<size_t>(l))
        return 0.0;
    const double tot = _pu_gain_tot[l];
    return tot > 0.0 ? std::clamp(_pu_gain_rod[l] / tot, 0.0, 1.0) : 0.0;
}

void XSSet::DepleteRodMaterials(double dt, double power, bool corrected_flux) {
    AccumulatePuHistory();
    if (dt <= 0.0 || power <= 0.0 || _fine_rod_type.empty())
        return;

    const int    nxy           = _g.nxy();
    const int    ng            = _g.ng();
    const int    div           = std::max(1, _axial_rod_division);
    const size_t expected_size = _fine_rod_type.size();

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

        const size_t th_idx = static_cast<size_t>(l) * static_cast<size_t>(ng) +
                              static_cast<size_t>(ng - 1);
        const double th_flux = corrected_flux
                                   ? 0.5 * (_flux_bos[th_idx] * bos_norm +
                                            _g.Phif()[th_idx] * eos_norm)
                                   : _g.Phif()[th_idx] * eos_norm;
        if (th_flux > 0.0)
            _fine_rod_thermal_fluence[static_cast<size_t>(idx)] +=
                th_flux * dt;
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
    auto&                     old_fraction         = _old_rod_fraction_scratch;
    auto&                     old_ctyp             = _old_rod_ctyp_scratch;
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
    RebuildUsesRodCache();

    if (dirty_nodes.empty()) return;

    XSUpdateOptions options;
    options.nodes.swap(dirty_nodes);
    UpdateFlatXS(options);
    options.nodes.swap(dirty_nodes);
}

void XSSet::SetBoron(double bppm) {
    xsphase::Scope boron_scope(xsphase::tallies().set_boron,
                               static_cast<std::uint64_t>(_g.nxyz()));
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

    // Water-property table ceiling.  milk::Table::Get() CLAMPS a query that runs off
    // the tabulated enthalpy axis, so a channel driven past the table simply freezes
    // at the last knot and leaves no other trace.  That is a silent failure: the
    // coolant properties of the hottest nodes stop responding to power, and every
    // quantity fed from them (moderator density, and through it the boron worth and
    // the whole radial power shape) is quietly wrong.  Count the nodes that land
    // outside and say so.  Everything below is read-only -- the solve is untouched.
    const double h_table_max =
        _mod_t_table.y_axis[_mod_t_table.y_axis.size() - 1];
    int          n_over      = 0;
    double       worst_h     = 0.0;
    int          worst_node  = -1;

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
            if (h_avg > h_table_max) {
                ++n_over;
                if (h_avg > worst_h) {
                    worst_h    = h_avg;
                    worst_node = lk;
                }
            }
            _g.tmod(lk)        = GetTmod(h_avg, pressure);
            _g.dmod(lk)        = GetDmod(h_avg, pressure);
            const double lpd   = 1000.0 * P_node / (62.0 * _g.hz(k));
            const double bu    = burnup[lk] / 1000.0;
            _g.tful(lk)        = _g.tmod(lk) +
                                 _g.fuel_temp_rise_scale() * GetTfuel(bu, lpd);
            h_cur              = h_out;
        }

        // Top reflector
        for (int k = kec; k < nz; ++k) {
            const int lk = l + k * nxy;
            if (k == kec && h_cur > h_table_max) {
                ++n_over;
                if (h_cur > worst_h) {
                    worst_h    = h_cur;
                    worst_node = lk;
                }
            }
            _g.tmod(lk)  = GetTmod(h_cur, pressure);
            _g.dmod(lk)  = GetDmod(h_cur, pressure);
        }
    }

    if (n_over > 0) {
        std::cerr << std::format(
            "[RASBERY][WARN][th] {} of {} nodes ran off the water-property table "
            "(enthalpy axis ends at {:.1f} kJ/kg); their coolant temperature and "
            "density are CLAMPED at the table edge and no longer respond to power. "
            "Worst {:.1f} kJ/kg at node {} (excess {:.1f}). The single-phase "
            "closed-channel model is outside its range here -- check the radial "
            "peaking against the core flow.\n",
            n_over, nxyz, h_table_max, worst_h, worst_node,
            worst_h - h_table_max);
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
        const auto&          model = _lib->models[_comp[l]];
        milk::Vector<double> lib_iden;
        auto                 xs_obj = model.GetCrossSection(
            0, _burn[l], _g.bppm(l), _g.tful(l), _g.dmod(l), &lib_iden);

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
    std::fill_n(_g.PhifMutable(), static_cast<size_t>(_g.ngxyz()), flux_value);

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
    // Hoisted: PhifMutable() bumps the flux generation, and this is ONE write
    // of the array, not ngxyz of them.
    double* const phif = _g.PhifMutable();
    for (size_t i = 0; i < flux_size; ++i)
        phif[i] = -phif[i];

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
    xsphase::Scope th_scope(xsphase::tallies().update_th,
                            static_cast<std::uint64_t>(_g.nxyz()));
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
        tful_old[lk]                      = _g.tful(lk);
        tmod_old[static_cast<size_t>(lk)] = _g.tmod(lk);
        dmod_old[static_cast<size_t>(lk)] = _g.dmod(lk);
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
