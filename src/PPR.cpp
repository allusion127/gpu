#include "PPR.h"

#include "CohortContext.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>

using namespace rasbery;

namespace {

constexpr int    kSourceSweepsPerIteration = 3;
constexpr double kCornerFluxTolerance      = 1.0E-5;

struct CornerFluxSums {
    double nw = 0.0;
    double sw = 0.0;
    double ne = 0.0;
    double se = 0.0;
};

double RelativeChange(double current, double previous) {
    return (previous != 0.0) ? std::abs((current - previous) / previous) : 1.0;
}

int To3DNeighborIndex(int neighbor2d, int axialPlane, int nxy) {
    return (neighbor2d >= 0) ? neighbor2d + axialPlane * nxy : -1;
}

} // namespace

PPR::PPR(Geometry& g, XSSet& xs)
    : _g(g), _xs(xs) {
    _nxyz = _g.nxyz();
    _ng   = _g.ng();

    // Corner-DF consistency correction (sdfa/pdfa on the corner constraints).
    // Opt-in: the MASTER benchmarks carry no CRADF.INP, so this is for
    // physical-accuracy studies, not code-to-code matching.
    if (const char* e = std::getenv("RASBERY_PPR_CRDF")) {
        std::string v(e);
        _crdf_on = (v == "1" || v == "on" || v == "ON" || v == "ratio");
    }
    _crdf.assign(static_cast<size_t>(_nxyz) * _ng, 1.0);

    // Reconstruction mode (see PPR.h).
    if (const char* e = std::getenv("RASBERY_PPR_MODE")) {
        std::string v(e);
        _mode_master = (v == "master" || v == "MASTER" || v == "afen");
    }

    // All coefficient arrays are owned by Geometry; just cache pointers.
    _phic = _g.Phic();
    _p    = _g.CoeffPart();
    _a    = _g.CoeffHom();
    _c    = _g.CoeffFit();
    _q    = _g.CoeffExp();
    _l    = _g.CoeffLeak();
    _bt   = _g.CoeffBuckling();

    // Task 10 device arm.  Constructed unconditionally and cheaply: with
    // RASBERY_GPU_PPR unset the constructor reads one environment variable and
    // stops.  One instance per PPR, i.e. per Driver, i.e. per batch slot.
    _gpu = std::make_unique<PprBackend>();
}

bool PPR::resetAndDriveGpu(const double reigv, double* jnet, const double* phif,
                           double* phis, int niter) {
    // Lowered before every decision below, so that the WP6 stage D
    // reconstruction refuses on any statepoint whose drive did not put
    // coefficients on the device -- including the ones that decline here.
    _gpu_drove = false;
    if (_gpu == nullptr) return false;
    // WP6 stage F.  EVERY REFUSAL ON THIS SIDE NAMES ITSELF TOO.  These
    // statepoints never reach PprBackend::resetAndDrive, so a ladder kept only
    // inside the backend would report "none" for the two commonest reasons a
    // production run falls back -- which is exactly how
    // `if (_mode_master) return false;` survived a whole campaign printing
    // nothing but `host_fallbacks:35`.
    if (!_gpu->available()) {
        _gpu->noteHostFallback(ppr::Refusal::ArmOff);
        return false;
    }
    if (_g.ng() != 2) {
        _gpu->noteHostFallback(ppr::Refusal::NotTwoGroup);
        return false;
    }
    // RASBERY_PPR_MODE=master USED TO REFUSE HERE.  It no longer does: the
    // MASTER MM 6.1 scheme -- the interpolant, the corner-point-balance solve
    // and the cross terms -- is ported (CudaPprBackend.cu, "WP6 stage F"), and
    // the mode travels in StepView so the backend selects the body rather than
    // declining the statepoint.

    // The pointer/scalar half of reset(), verbatim: everything downstream
    // (reconstructPinPower, phig, getPhis) reads these.
    _reigv = reigv;
    _jnet  = jnet;
    _phif  = phif;
    _phis  = phis;

    if (_crdf_on) {
        for (int lk = 0; lk < _nxyz; lk++) {
            const auto& model = _xs.models()[_xs.comp(lk)];
            const int   burn  = _xs.burn(lk);
            for (int g = 0; g < _ng; g++)
                _crdf[static_cast<size_t>(lk) * _ng + g] = model.CornerToSurfaceDFRatio(g, burn);
        }
        // WP6 stage C.  The array was just rewritten, so the device copy is
        // stale -- bumped HERE, at the write, and nowhere else: a bump at the
        // caller would be a policy that a second writer could silently break.
        // With the correction off this line never runs and the generation holds
        // at 1 forever, which is what turns crdf into a one-time upload.
        ++_crdf_generation;
    }

    const size_t nng = static_cast<size_t>(_nxyz) * _ng;
    if (_xsdf_stage.size() != nng) _xsdf_stage.resize(nng);
    for (int g = 0; g < _ng; ++g)
        for (int lk = 0; lk < _nxyz; ++lk)
            _xsdf_stage[static_cast<size_t>(g) * _nxyz + lk] = _xs.xsdf(g, lk);

    if (_isfuel_stage.size() != static_cast<size_t>(_nxyz)) {
        _isfuel_stage.assign(static_cast<size_t>(_nxyz), 0);
        for (int lk = 0; lk < _nxyz; ++lk)
            _isfuel_stage[static_cast<size_t>(lk)] = _g.IsFuel(lk) ? 1u : 0u;
    }

    ppr::GeomView geom;
    geom.ng      = _ng;
    geom.nxyz    = _nxyz;
    geom.nxy     = _g.nxy();
    geom.nsurf   = _g.nsurf();
    geom.hmesh   = &_g.hmesh(XDIR, 0);
    geom.lktosfc = &_g.lktosfc(LEFT, XDIR, 0);
    geom.neibrb  = &_g.neibrb(WEST, 0);
    geom.is_fuel = _isfuel_stage.data();

    ppr::StepView step;
    step.reigv = _reigv;
    step.phif  = _phif;
    step.phis  = _phis;
    step.jnet  = _jnet;
    step.xsdf  = _xsdf_stage.data();
    step.xsrf  = _xs.xsrfData();
    step.xsnf  = _xs.xsnfData();
    step.xssm  = _xs.xssmData();
    step.chif  = _xs.chifData();
    step.crdf  = _crdf.data();
    // WP6 stage C.  chif is the burnup-interpolated fission spectrum, rebuilt
    // by PrecomputeBranchCoefficients (XSSet::_ref_generation) and NOT once per
    // statepoint; crdf carries PPR's own counter.  Everything else in StepView
    // moves every statepoint and is uploaded every statepoint.
    step.chif_generation = _xs.refGeneration();
    step.crdf_generation = _crdf_generation;
    step.mode_master     = _mode_master;
    // The borrowed nodal set, if the caller offered one this statepoint.  The
    // completeness test is repeated here rather than trusted: a partial set
    // would pair a device jnet with an uploaded phif.
    if (_canonical.mode != ppr::CanonicalMode::Off && _canonical.complete()) {
        step.canonical = _canonical.mode;
        step.dev_phif  = _canonical.phif;
        step.dev_phis  = _canonical.phis;
        step.dev_jnet  = _canonical.jnet;
    }
    step.phic  = _phic;
    step.p     = _p;
    step.a     = _a;
    step.c     = _c;
    step.q     = _q;
    step.l     = _l;
    step.bt    = _bt;

    // WP6 stage D.  Decided HERE, before the drive, because the drive is what
    // issues (or does not issue) the 9 MB coefficient download.
    _last_niter                       = niter;
    const bool plan                   = reconPlanned();
    step.coefficients_stay_on_device  = plan;

    // PPR::drive's own floor for the MASTER cap, applied at the seam so both
    // arms iterate against the same bound: `driveMaster(std::max(niter, 200))`.
    // The device arm would otherwise stop 100 rounds earlier than the host it
    // is being compared against.
    const int gpu_niter = _mode_master ? std::max(niter, 200) : niter;

    int iters = 0;
    _gpu_drove = _gpu->resetAndDrive(geom, step, gpu_niter, &iters);
    _coeffs_device_only = plan && _gpu_drove;
    return _gpu_drove;
}

bool PPR::reconPlanned() {
    if (_gpu == nullptr || !_gpu->available()) return false;
    if (!ppr::reconEnabledFromEnv()) return false;
    if (_ng != 2) return false;
    // The table and the registry are built here rather than lazily at the
    // reconstruction, because "can this arm run" has to be answerable before
    // the drive commits to eliding the download.
    if (!_pin_quad_table) acquireQuadratureTable();
    return buildReconStaging();
}

// ---------------------------------------------------------------------------
// WP6 stage D: staging for the device reconstruction
// ---------------------------------------------------------------------------

bool PPR::buildReconStaging() {
    if (_recon.declined) return false;
    if (_recon.built) return true;
    if (!_pin_quad_table) return false;

    const int npins = _g.npins();
    const int npina = npins * npins;
    const int ng    = _ng;

    if (static_cast<int>(_pin_quad_table->size()) != npina) {
        _recon.declined = true;
        _recon.reason   = "quadrature table size != npins^2";
        return false;
    }

    // --- the quadrature table, flattened -----------------------------------
    //
    // SoA and not the host's vector-of-structs, because a kernel reading
    // `qpts[q].leg[t]` through a 162-double stride would serialise on the
    // stride; the VALUES are copied, not recomputed, so nothing here can move a
    // bit of the table the host built.
    _recon.pin_off.assign(static_cast<size_t>(npina) + 1, 0);
    int total = 0;
    for (int i = 0; i < npina; ++i) {
        _recon.pin_off[static_cast<size_t>(i)] = total;
        total += static_cast<int>((*_pin_quad_table)[static_cast<size_t>(i)].overlaps.size());
    }
    _recon.pin_off[static_cast<size_t>(npina)] = total;
    _recon.n_overlaps                          = total;

    _recon.ovl_di.assign(static_cast<size_t>(total), 0);
    _recon.ovl_dj.assign(static_cast<size_t>(total), 0);
    _recon.ovl_dxh.assign(static_cast<size_t>(total), 0.0);
    _recon.ovl_dyh.assign(static_cast<size_t>(total), 0.0);
    _recon.q_xq.assign(static_cast<size_t>(total) * 9, 0.0);
    _recon.q_yq.assign(static_cast<size_t>(total) * 9, 0.0);
    _recon.q_wt.assign(static_cast<size_t>(total) * 9, 0.0);
    _recon.q_leg.assign(static_cast<size_t>(total) * 9 * 15, 0.0);

    int o = 0;
    for (int i = 0; i < npina; ++i) {
        for (const auto& ovl : (*_pin_quad_table)[static_cast<size_t>(i)].overlaps) {
            _recon.ovl_di[static_cast<size_t>(o)]  = ovl.di;
            _recon.ovl_dj[static_cast<size_t>(o)]  = ovl.dj;
            _recon.ovl_dxh[static_cast<size_t>(o)] = ovl.dx_h;
            _recon.ovl_dyh[static_cast<size_t>(o)] = ovl.dy_h;
            for (int q = 0; q < 9; ++q) {
                const auto&  qp = ovl.qpts[q];
                const size_t qi = static_cast<size_t>(o) * 9 + q;
                _recon.q_xq[qi] = qp.xq;
                _recon.q_yq[qi] = qp.yq;
                _recon.q_wt[qi] = qp.wt;
                for (int tt = 0; tt < 15; ++tt)
                    _recon.q_leg[qi * 15 + tt] = qp.leg[tt];
            }
            ++o;
        }
    }

    // --- the form-function registry ----------------------------------------
    //
    // EVERY reference depletion point of every model, once.  The host
    // interpolates a fresh (fmap, gmap) pair per (plane, assembly) per
    // statepoint out of exactly these; putting the endpoints on the device and
    // sending three numbers per plane is the same arithmetic with 260x less
    // traffic.
    _recon.slot_of.clear();
    _recon.gmap.clear();
    _recon.fmap.clear();
    _recon.slots = 0;
    const auto& models = _xs.models();
    for (size_t mi = 0; mi < models.size(); ++mi) {
        const auto& model = models[mi];
        const auto  it    = model._refr_dpts.find(0);
        if (it == model._refr_dpts.end()) continue;
        for (const auto& entry : it->second) {
            const size_t di  = static_cast<size_t>(entry.second);
            const auto   key = std::make_pair(mi, di);
            if (_recon.slot_of.count(key) != 0) continue;
            const auto& dpt = model.GetDepletionPoint(di);
            if (static_cast<int>(dpt._gmap.size()) != npina ||
                static_cast<int>(dpt._fmap.size()) != ng * npina) {
                // A library whose pin map is not this geometry's is not a case
                // the kernels can index; the host loop reads the same vectors
                // and would be equally wrong, but it is not this arm's business
                // to decide that -- it declines and lets the host run.
                _recon.declined = true;
                _recon.reason   = "form-function map size != npins^2";
                return false;
            }
            _recon.slot_of[key] = _recon.slots++;
            _recon.gmap.insert(_recon.gmap.end(), dpt._gmap.begin(), dpt._gmap.end());
            _recon.fmap.insert(_recon.fmap.end(), dpt._fmap.begin(), dpt._fmap.end());
        }
    }
    if (_recon.slots == 0) {
        _recon.declined = true;
        _recon.reason   = "no reference depletion points";
        return false;
    }

    const size_t nplan = static_cast<size_t>(_g.nz()) * _g.nxya();
    _recon.plane_lo.assign(nplan, -1);
    _recon.plane_hi.assign(nplan, -1);
    _recon.plane_alpha.assign(nplan, 0.0);
    _recon.xskf.assign(static_cast<size_t>(_nxyz) * ng, 0.0);

    _recon.built = true;
    return true;
}

bool PPR::reconstructPinPowerGpu(bool use_quadrature, bool reconstruct_flux,
                                 bool materialize_pin_map) {
    if (_gpu == nullptr || !_gpu->available()) return false;
    if (!_gpu_drove) return false;
    if (!ppr::reconEnabledFromEnv()) return false;
    // The pointwise `phig` sampling is a different scheme -- the kernels
    // transcribe the QUADRATURE path only -- so it declines rather than
    // silently reconstructing by another method.  MASTER mode is no longer in
    // this list: its expansion (a 15-term dot product of `c` with the
    // pre-computed Legendre products, no `exp`) is ported alongside the SENM
    // one in PprReconstructionKernel.cuh.
    if (!use_quadrature || _ng != 2) return false;
    if (!_pin_quad_table) acquireQuadratureTable();
    if (!buildReconStaging()) return false;

    const int ng    = _ng;
    const int nxy   = _g.nxy();
    const int nz    = _g.nz();
    const int nxya  = _g.nxya();
    const int ndiv  = _g.ndivxy();
    const int ndiv2 = ndiv * ndiv;
    const int kbc   = _g.kbc();
    const int kec   = _g.kec();

    for (int g = 0; g < ng; ++g)
        for (int lk = 0; lk < _nxyz; ++lk)
            _recon.xskf[static_cast<size_t>(g) * _nxyz + lk] = _xs.xskf(g, lk);

    // The host's own bracketing lookup, per (plane, assembly), verbatim -- the
    // map walk stays on the host because it is a std::map walk and because
    // getting it wrong here is invisible.  What changes is that the RESULT is
    // three numbers instead of two interpolated maps.
    std::fill(_recon.plane_lo.begin(), _recon.plane_lo.end(), -1);
    std::fill(_recon.plane_hi.begin(), _recon.plane_hi.end(), -1);
    std::fill(_recon.plane_alpha.begin(), _recon.plane_alpha.end(), 0.0);

    for (int k = kbc; k < kec; ++k) {
        for (int la = 0; la < nxya; ++la) {
            int ref_lk = -1;
            for (int li = 0; li < ndiv2; ++li) {
                const int l = _g.latol(li, la);
                if (l >= 0) {
                    const int lk_tmp = l + nxy * k;
                    if (_g.IsFuel(lk_tmp)) {
                        ref_lk = lk_tmp;
                        break;
                    }
                }
            }
            if (ref_lk < 0) continue;

            const size_t mi    = _xs.comp(ref_lk);
            const auto&  model = _xs.models()[mi];
            const int    burn  = _xs.burn(ref_lk);

            const auto& refrMap = model._refr_dpts.at(0);
            auto        hiIt    = refrMap.lower_bound(burn);
            auto        loIt    = hiIt;

            if (hiIt == refrMap.end()) {
                loIt = std::prev(refrMap.end());
                hiIt = loIt;
            } else if (hiIt == refrMap.begin()) {
                loIt = refrMap.begin();
            } else {
                loIt = std::prev(hiIt);
            }

            const auto& loDpt = model.GetDepletionPoint(loIt->second);
            const auto& hiDpt = model.GetDepletionPoint(hiIt->second);

            double alpha = 0.0;
            if (loIt != hiIt && hiDpt.burnKey() != loDpt.burnKey())
                alpha = static_cast<double>(burn - loDpt.burnKey()) /
                        static_cast<double>(hiDpt.burnKey() - loDpt.burnKey());

            const auto lo_it =
                _recon.slot_of.find(std::make_pair(mi, static_cast<size_t>(loIt->second)));
            const auto hi_it =
                _recon.slot_of.find(std::make_pair(mi, static_cast<size_t>(hiIt->second)));
            if (lo_it == _recon.slot_of.end() || hi_it == _recon.slot_of.end()) return false;

            const size_t lka             = static_cast<size_t>(la) + static_cast<size_t>(nxya) * k;
            _recon.plane_lo[lka]         = lo_it->second;
            _recon.plane_hi[lka]         = hi_it->second;
            _recon.plane_alpha[lka]      = alpha;
        }
    }

    ppr::ReconGeomView geom;
    geom.nxya       = nxya;
    geom.nz         = nz;
    geom.kbc        = kbc;
    geom.kec        = kec;
    geom.ndiv       = ndiv;
    geom.npins      = _g.npins();
    geom.latol      = &_g.latol(0, 0);
    geom.vol        = &_g.vol(0);
    geom.hz         = &_g.hz(0);
    geom.n_overlaps = _recon.n_overlaps;
    geom.pin_off    = _recon.pin_off.data();
    geom.ovl_di     = _recon.ovl_di.data();
    geom.ovl_dj     = _recon.ovl_dj.data();
    geom.ovl_dxh    = _recon.ovl_dxh.data();
    geom.ovl_dyh    = _recon.ovl_dyh.data();
    geom.q_xq       = _recon.q_xq.data();
    geom.q_yq       = _recon.q_yq.data();
    geom.q_wt       = _recon.q_wt.data();
    geom.q_leg      = _recon.q_leg.data();

    ppr::ReconStepView step;
    step.xskf             = _recon.xskf.data();
    step.n_form_slots     = _recon.slots;
    step.gmap             = _recon.gmap.data();
    step.fmap             = _recon.fmap.data();
    step.plane_lo         = _recon.plane_lo.data();
    step.plane_hi         = _recon.plane_hi.data();
    step.plane_alpha      = _recon.plane_alpha.data();
    step.mode_master      = _mode_master;
    step.reconstruct_flux = reconstruct_flux;
    step.materialize_pin  = materialize_pin_map;
    step.pin_power        = _g.PinPower();
    step.pin_flux         = reconstruct_flux ? _g.PinFlux() : nullptr;
    step.frp              = &_g.frp();
    step.fqp              = &_g.fqp();

    return _gpu->reconstructPinPower(geom, step);
}

// WP8 stage 2.  Was `PPR::buildQuadratureTable()`, a member that filled a
// per-object vector.  It is now a free function of the only two things it ever
// read, so the result can be built once per process and shared by every case of
// the cohort.  THE ARITHMETIC IS UNCHANGED, line for line: the gate for the
// whole work package is that the trajectory digest does not move, and a
// reordered floating-point expression would move it.
PinQuadTable rasbery::buildPinQuadratureTable(int ndiv, int npins) {
    PinQuadTable            table;
    const double            inv_npins = 1.0 / npins;
    const double            inv_ndiv  = 1.0 / ndiv;
    static constexpr double xi3[3]    = {-0.7745966692414834, 0.0, 0.7745966692414834};
    static constexpr double wi3[3]    = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

    table.resize(npins * npins);

    for (int py = 0; py < npins; ++py) {
        const double y0    = py * inv_npins;
        const double y1    = (py + 1) * inv_npins;
        const int    dj_lo = std::min(static_cast<int>(y0 * ndiv), ndiv - 1);
        const int    dj_hi = std::min(static_cast<int>(y1 * ndiv - 1e-12), ndiv - 1);

        for (int px = 0; px < npins; ++px) {
            const double x0    = px * inv_npins;
            const double x1    = (px + 1) * inv_npins;
            const int    di_lo = std::min(static_cast<int>(x0 * ndiv), ndiv - 1);
            const int    di_hi = std::min(static_cast<int>(x1 * ndiv - 1e-12), ndiv - 1);

            auto& info = table[py * npins + px];
            info.overlaps.clear();

            for (int dj = dj_lo; dj <= dj_hi; ++dj) {
                const double oy0  = std::max(y0, dj * inv_ndiv);
                const double oy1  = std::min(y1, (dj + 1) * inv_ndiv);
                const double yp0  = 2.0 * ndiv * oy0 - 2.0 * dj - 1.0;
                const double yp1  = 2.0 * ndiv * oy1 - 2.0 * dj - 1.0;
                const double dy_h = (yp1 - yp0) * 0.5;
                const double yc   = (yp0 + yp1) * 0.5;

                for (int di = di_lo; di <= di_hi; ++di) {
                    const double ox0  = std::max(x0, di * inv_ndiv);
                    const double ox1  = std::min(x1, (di + 1) * inv_ndiv);
                    const double xp0  = 2.0 * ndiv * ox0 - 2.0 * di - 1.0;
                    const double xp1  = 2.0 * ndiv * ox1 - 2.0 * di - 1.0;
                    const double dx_h = (xp1 - xp0) * 0.5;
                    const double xc   = (xp0 + xp1) * 0.5;

                    PinOverlap ovl;
                    ovl.di   = di;
                    ovl.dj   = dj;
                    ovl.dx_h = dx_h;
                    ovl.dy_h = dy_h;

                    for (int qi = 0; qi < 3; ++qi) {
                        const double xq    = dx_h * xi3[qi] + xc;
                        const double x2    = xq * xq;
                        const double Lx[5] = {1.0, xq, 0.5 * (3 * x2 - 1), 0.5 * (5 * x2 - 3) * xq, 0.125 * (35 * x2 * x2 - 30 * x2 + 3)};

                        for (int qj = 0; qj < 3; ++qj) {
                            const double yq    = dy_h * xi3[qj] + yc;
                            const double y2    = yq * yq;
                            const double Ly[5] = {1.0, yq, 0.5 * (3 * y2 - 1), 0.5 * (5 * y2 - 3) * yq, 0.125 * (35 * y2 * y2 - 30 * y2 + 3)};

                            auto& qp = ovl.qpts[qi * 3 + qj];
                            qp.xq    = xq;
                            qp.yq    = yq;
                            qp.wt    = wi3[qi] * wi3[qj];

                            // Pre-compute Legendre products: Lx[i]*Ly[j] in upper-triangular order
                            int t = 0;
                            for (int i = 0; i < 5; ++i)
                                for (int j = 0; j < 5 - i; ++j)
                                    qp.leg[t++] = Lx[i] * Ly[j];
                        }
                    }
                    info.overlaps.push_back(std::move(ovl));
                }
            }
        }
    }
    return table;
}

void PPR::acquireQuadratureTable() {
    _pin_quad_table = cohort::acquirePinQuadrature(_g.ndivxy(), _g.npins());
}

/// @brief Build 3×3 neighbor stencil and reflection flags from neibrb.
void PPR::buildStencil(int lk, int idx[3][3], bool xrev[3][3], bool yrev[3][3]) {
    int nxy = _g.nxy();
    int l2d = lk % nxy;
    int k   = lk / nxy;

    // 2D stencil via neibrb (cardinal + diagonal by composition)
    int nb[3][3];
    nb[1][1] = l2d;
    nb[0][1] = _g.neibrb(WEST, l2d);
    nb[2][1] = _g.neibrb(EAST, l2d);
    nb[1][0] = _g.neibrb(NORTH, l2d);
    nb[1][2] = _g.neibrb(SOUTH, l2d);
    nb[0][0] = (nb[1][0] >= 0) ? _g.neibrb(WEST, nb[1][0]) : -1;
    nb[2][0] = (nb[1][0] >= 0) ? _g.neibrb(EAST, nb[1][0]) : -1;
    nb[0][2] = (nb[1][2] >= 0) ? _g.neibrb(WEST, nb[1][2]) : -1;
    nb[2][2] = (nb[1][2] >= 0) ? _g.neibrb(EAST, nb[1][2]) : -1;

    // Reflection flags: neibrb closes a zero-albedo boundary by mapping the node
    // onto itself, so "the closure neighbour is me" is exactly "this face is
    // reflective".  This used to be written as (neibr != neibrb), which reads the
    // same today but stops being equivalent once neibr carries a real neighbour
    // across a symmetry line (the 90-degree rotational fold stitches node (0,0)'s
    // west face to its own north face, making neibr == l2d legitimately).
    bool xr_w = (_g.neibrb(WEST, l2d) == l2d);
    bool xr_e = (_g.neibrb(EAST, l2d) == l2d);
    bool yr_n = (_g.neibrb(NORTH, l2d) == l2d);
    bool yr_s = (_g.neibrb(SOUTH, l2d) == l2d);

    const bool x_reflection[3] = {xr_w, false, xr_e};
    const bool y_reflection[3] = {yr_n, false, yr_s};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            idx[i][j]  = (nb[i][j] >= 0) ? nb[i][j] + k * nxy : -1;
            xrev[i][j] = x_reflection[i];
            yrev[i][j] = y_reflection[j];
        }
    }
}

void PPR::reset(const double reigv, double* jnet, const double* phif, double* phis) {

    // 1. Get mesh information and initialize variables
    _reigv = reigv;
    _jnet  = jnet;
    _phif  = phif;
    _phis  = phis;

    // Refresh the per-node corner-DF ratios at the current burnup state.
    if (_crdf_on) {
        for (int lk = 0; lk < _nxyz; lk++) {
            const auto& model = _xs.models()[_xs.comp(lk)];
            const int   burn  = _xs.burn(lk);
            for (int g = 0; g < _ng; g++)
                _crdf[static_cast<size_t>(lk) * _ng + g] = model.CornerToSurfaceDFRatio(g, burn);
        }
        // WP6 stage C.  The array was just rewritten, so the device copy is
        // stale -- bumped HERE, at the write, and nowhere else: a bump at the
        // caller would be a policy that a second writer could silently break.
        // With the correction off this line never runs and the generation holds
        // at 1 forever, which is what turns crdf into a one-time upload.
        ++_crdf_generation;
    }

    for (int lk = 0; lk < _nxyz; lk++) {
        for (int g = 0; g < _ng; g++) {
            _hmesh = _g.hmesh(XDIR, lk);
            Bt     = std::sqrt((_hmesh * _hmesh * _xs.xsrf(g, lk)) / (4.0 * _xs.xsdf(g, lk)));
        }
    }

    // 2. Update corner flux
    for (int lk = 0; lk < _nxyz; lk++) {
        int  idx[3][3];
        bool xrev[3][3], yrev[3][3];
        buildStencil(lk, idx, xrev, yrev);

        for (int g = 0; g < _ng; g++) {
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    double sur_flux = 0.0;
                    double avg_flux = 0.0;
                    int    dir      = j * 2 + i;

                    sur_flux += (getPhis(RIGHT ^ xrev[i + 0][j + 0], XDIR, idx[i + 0][j + 0], g) +
                                 getPhis(RIGHT ^ xrev[i + 0][j + 1], XDIR, idx[i + 0][j + 1], g)) *
                                0.5;

                    sur_flux += (getPhis(RIGHT ^ yrev[i + 0][j + 0], YDIR, idx[i + 0][j + 0], g) +
                                 getPhis(RIGHT ^ yrev[i + 1][j + 0], YDIR, idx[i + 1][j + 0], g)) *
                                0.5;

                    avg_flux += (idx[i + 0][j + 0] < 0 || idx[i + 0][j + 0] >= _nxyz) ? 0.0 : _phif[idx[i + 0][j + 0] * _ng + g];
                    avg_flux += (idx[i + 1][j + 0] < 0 || idx[i + 1][j + 0] >= _nxyz) ? 0.0 : _phif[idx[i + 1][j + 0] * _ng + g];
                    avg_flux += (idx[i + 0][j + 1] < 0 || idx[i + 0][j + 1] >= _nxyz) ? 0.0 : _phif[idx[i + 0][j + 1] * _ng + g];
                    avg_flux += (idx[i + 1][j + 1] < 0 || idx[i + 1][j + 1] >= _nxyz) ? 0.0 : _phif[idx[i + 1][j + 1] * _ng + g];

                    phic(dir) = sur_flux - avg_flux * 0.25;
                }
            }
        }
    }

    // 3. Flux fitting from nodal constraints
    for (int lk = 0; lk < _nxyz; lk++) {
        for (int g = 0; g < _ng; g++) {
            _hmesh      = _g.hmesh(XDIR, lk);
            double invD = 1.0 / _xs.xsdf(g, lk);

            c(4, 0) = 0.2142857143 * (2.0 * aflux - phis(XDIR, LEFT) - phis(XDIR, RIGHT)) + (0.03571428571 * _hmesh * invD) * (jnet(XDIR, RIGHT) - jnet(XDIR, LEFT));
            c(0, 4) = 0.2142857143 * (2.0 * aflux - phis(YDIR, LEFT) - phis(YDIR, RIGHT)) + (0.03571428571 * _hmesh * invD) * (jnet(YDIR, RIGHT) - jnet(YDIR, LEFT));

            c(3, 1) = 0.0;
            c(1, 3) = 0.0;

            c(3, 0) = 0.1 * (phis(XDIR, LEFT) - phis(XDIR, RIGHT)) - (0.05 * _hmesh * invD) * (jnet(XDIR, RIGHT) + jnet(XDIR, LEFT));
            c(0, 3) = 0.1 * (phis(YDIR, LEFT) - phis(YDIR, RIGHT)) - (0.05 * _hmesh * invD) * (jnet(YDIR, RIGHT) + jnet(YDIR, LEFT));

            // The corner-balance phic is heterogeneous; this node's folded (SET)
            // expansion needs sdfa/pdfa on it (all-1 unless RASBERY_PPR_CRDF on
            // a library with corner factors).
            const double rc = crdf(lk, g);

            c(2, 2) = aflux + 0.25 * rc * (phic(NE) + phic(NW) + phic(SE) + phic(SW)) - 0.5 * (phis(XDIR, RIGHT) + phis(XDIR, LEFT) + phis(YDIR, RIGHT) + phis(YDIR, LEFT));

            c(2, 1) = 0.25 * rc * (phic(SE) + phic(SW) - phic(NE) - phic(NW)) + 0.5 * (phis(YDIR, RIGHT) - phis(YDIR, LEFT));
            c(1, 2) = 0.25 * rc * (phic(SE) - phic(SW) + phic(NE) - phic(NW)) + 0.5 * (phis(XDIR, RIGHT) - phis(XDIR, LEFT));

            c(2, 0) = 0.7142857143 * (-2 * aflux + phis(XDIR, LEFT) + phis(XDIR, RIGHT)) + (0.03571428571 * _hmesh * invD) * (jnet(XDIR, LEFT) - jnet(XDIR, RIGHT));
            c(0, 2) = 0.7142857143 * (-2 * aflux + phis(YDIR, LEFT) + phis(YDIR, RIGHT)) + (0.03571428571 * _hmesh * invD) * (jnet(YDIR, LEFT) - jnet(YDIR, RIGHT));

            c(1, 1) = 0.25 * rc * (phic(SE) - phic(SW) - phic(NE) + phic(NW));

            c(1, 0) = 0.6 * (phis(XDIR, LEFT) - phis(XDIR, RIGHT)) + (0.05 * _hmesh * invD) * (jnet(XDIR, RIGHT) + jnet(XDIR, LEFT));
            c(0, 1) = 0.6 * (phis(YDIR, LEFT) - phis(YDIR, RIGHT)) + (0.05 * _hmesh * invD) * (jnet(YDIR, RIGHT) + jnet(YDIR, LEFT));

            c(0, 0) = aflux;
        }
    }
    updateAxialLeakage();
    updateSource();
}

double PPR::getPhis(int lr, int dir, int lk, int g) {
    if (lk < 0 || lk >= _g.nxyz()) {
        return 0.0;
    }
    return _phis[_g.lktosfc(lr, dir, lk) * _ng + g];
}

// Fused update: updateParticular + updateHomogeneous + projectFlux in one pass
// Eliminates 2 extra sweeps over nxyz and shares sinh/cosh between Homogeneous and projectFlux
void PPR::updateFused(int lk, int g) {
    // updateParticular
    double D  = _xs.xsdf(g, lk);
    double rr = 1.0 / _xs.xsrf(g, lk);
    double rh = 1.0 / _g.hmesh(XDIR, lk);

    double Drh2r2  = D * (rr * rr) * (rh * rh);
    double D2rh4r3 = (D * D) * (rr * rr * rr) * (rh * rh * rh * rh);

    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5 - i; j++)
            p(i, j) = q(lk, g, i, j) * rr;

    p(0, 2) += Drh2r2 * (140.0 * q(lk, g, 0, 4) + 12.0 * q(lk, g, 2, 2));
    p(2, 0) += Drh2r2 * (140.0 * q(lk, g, 4, 0) + 12.0 * q(lk, g, 2, 2));
    p(1, 1) += 60.0 * Drh2r2 * (q(lk, g, 1, 3) + q(lk, g, 3, 1));
    p(0, 1) += Drh2r2 * (60.0 * q(lk, g, 0, 3) + 12.0 * q(lk, g, 2, 1));
    p(1, 0) += Drh2r2 * (60.0 * q(lk, g, 3, 0) + 12.0 * q(lk, g, 1, 2));
    p(0, 0) += Drh2r2 * (12.0 * q(lk, g, 0, 2) + 12.0 * q(lk, g, 2, 0) + 40.0 * q(lk, g, 0, 4) + 40.0 * q(lk, g, 4, 0));
    p(0, 0) += D2rh4r3 * (1680.0 * q(lk, g, 0, 4) + 288.0 * q(lk, g, 2, 2) + 1680.0 * q(lk, g, 4, 0));

    // Shared sinh/cosh (used by both Homogeneous and projectFlux)
    double hmesh  = _g.hmesh(XDIR, lk);
    double rhmesh = rh;
    double _2DrH  = 2.0 * D * rhmesh;
    double bt     = Bt;

    // Compute exp pairs instead of sinh/cosh (2 exp calls instead of 4 sinh+cosh)
    double e1 = std::exp(bt), re1 = 1.0 / e1;
    double e2 = std::exp(bt * rsq2), re2 = 1.0 / e2;
    double S   = 0.5 * (e1 - re1);
    double C   = 0.5 * (e1 + re1);
    double Sp  = 0.5 * (e2 - re2);
    double Cp  = 0.5 * (e2 + re2);
    double Sp2 = Sp * Sp;
    double Cp2 = Cp * Cp;

    // updateHomogeneous (rc = corner-DF ratio; see reset())
    const double rc = crdf(lk, g);
    double hFlux_SW = rc * phic(SW) - (p(0, 0) + p(0, 1) + p(0, 2) + p(0, 3) + p(0, 4) - p(1, 0) - p(1, 1) - p(1, 2) - p(1, 3) + p(2, 0) + p(2, 1) + p(2, 2) - p(3, 0) - p(3, 1) + p(4, 0));
    double hFlux_SE = rc * phic(SE) - (p(0, 0) + p(0, 1) + p(0, 2) + p(0, 3) + p(0, 4) + p(1, 0) + p(1, 1) + p(1, 2) + p(1, 3) + p(2, 0) + p(2, 1) + p(2, 2) + p(3, 0) + p(3, 1) + p(4, 0));
    double hFlux_NW = rc * phic(NW) - (p(0, 0) - p(0, 1) + p(0, 2) - p(0, 3) + p(0, 4) - p(1, 0) + p(1, 1) - p(1, 2) + p(1, 3) + p(2, 0) - p(2, 1) + p(2, 2) - p(3, 0) + p(3, 1) + p(4, 0));
    double hFlux_NE = rc * phic(NE) - (p(0, 0) - p(0, 1) + p(0, 2) - p(0, 3) + p(0, 4) + p(1, 0) - p(1, 1) + p(1, 2) - p(1, 3) + p(2, 0) - p(2, 1) + p(2, 2) + p(3, 0) - p(3, 1) + p(4, 0));

    double hCurr_xl = jnet(XDIR, LEFT) + _2DrH * (p(1, 0) - 3. * p(2, 0) + 6. * p(3, 0) - 10. * p(4, 0));
    double hCurr_xr = jnet(XDIR, RIGHT) + _2DrH * (p(1, 0) + 3. * p(2, 0) + 6. * p(3, 0) + 10. * p(4, 0));
    double hCurr_yl = jnet(YDIR, LEFT) + _2DrH * (p(0, 1) - 3. * p(0, 2) + 6. * p(0, 3) - 10. * p(0, 4));
    double hCurr_yr = jnet(YDIR, RIGHT) + _2DrH * (p(0, 1) + 3. * p(0, 2) + 6. * p(0, 3) + 10. * p(0, 4));

    double Dph_1 = D * (hFlux_SE - hFlux_SW + hFlux_NE - hFlux_NW);
    double Dph_2 = D * (hFlux_SE - hFlux_SW - hFlux_NE + hFlux_NW);
    double Dph_3 = D * (hFlux_SE + hFlux_SW - hFlux_NE - hFlux_NW);
    double Dph_4 = D * (hFlux_SE + hFlux_SW + hFlux_NE + hFlux_NW);

    double hj_xs = hmesh * (hCurr_xr + hCurr_xl);
    double hj_xd = hmesh * (hCurr_xr - hCurr_xl);
    double hj_ys = hmesh * (hCurr_yr + hCurr_yl);
    double hj_yd = hmesh * (hCurr_yr - hCurr_yl);

    double denom1 = 1.0 / (4.0 * D * (S - bt * C));
    a(1)          = (Dph_1 + hj_xs) * denom1;
    a(3)          = (Dph_3 + hj_ys) * denom1;

    double denom2 = 1.0 / (4.0 * D * bt * S * (bt * S * Cp2 - 2 * C * Sp2));
    a(2)          = (C * Sp2 * (hj_xd - hj_yd) - bt * S * (Cp2 * hj_xd + Sp2 * Dph_4)) * denom2;
    a(4)          = (C * Sp2 * (hj_yd - hj_xd) - bt * S * (Cp2 * hj_yd + Sp2 * Dph_4)) * denom2;

    double denom3 = 1.0 / (4.0 * D * Cp * Sp * (bt * C - S));
    a(5)          = (bt * C * Dph_1 + S * hj_xs) * denom3;
    a(7)          = (bt * C * Dph_3 + S * hj_ys) * denom3;

    a(6) = Dph_2 / (4.0 * D * Sp2);

    double denom4 = 1.0 / (4.0 * D * (bt * S * Cp2 - 2 * C * Sp2));
    a(8)          = denom4 * (bt * S * Dph_4 + C * (hj_xd + hj_yd));

    // projectFlux (reuses S, C, Sp, Cp from above)
    double bt2 = bt * bt;
    double bt3 = bt * bt2;
    double bt4 = bt2 * bt2;
    double bt5 = bt2 * bt3;

    double rbt  = 1.0 / bt;
    double rbt2 = rbt * rbt;
    double rbt3 = rbt * rbt2;
    double rbt4 = rbt2 * rbt2;
    double rbt5 = rbt3 * rbt2;
    double rbt6 = rbt3 * rbt3;

    double CpSp   = Cp * Sp;
    double sq2Sp2 = sq2 * Sp2;
    double BtCpSp = bt * CpSp;
    double BtSp2  = bt * Sp2;

    c(4, 0) = 9.0 * rbt6 * (a(2) * bt * (S * (105 + 45 * bt2 + bt4) - 5 * C * bt * (21 + 2 * bt2)) + 2.0 * a(8) * Sp * ((420 + 90 * bt2 + bt4) * Sp - (10 * sq2 * bt * (21 + bt2)) * Cp));
    c(0, 4) = 9.0 * rbt6 * (a(4) * bt * (S * (105 + 45 * bt2 + bt4) - 5 * C * bt * (21 + 2 * bt2)) + 2.0 * a(8) * Sp * ((420 + 90 * bt2 + bt4) * Sp - (10 * sq2 * bt * (21 + bt2)) * Cp));

    c(3, 1) = 42 * rbt6 * a(6) * (30 * bt2 + bt4 - 60 * sq2 * BtCpSp - 7 * sq2 * bt2 * BtCpSp + 60 * Sp2 + 42 * bt2 * Sp2 + bt4 * Sp2);
    c(1, 3) = c(3, 1);

    c(3, 0) = 7 * rbt5 * (a(1) * bt * (C * bt3 - 6 * S * bt2 + 15 * C * bt - 15 * S) + 2 * a(5) * Sp * (bt * Cp * (30 + bt2) - (6 * (5 + bt2) * sq2 * Sp)));
    c(0, 3) = 7 * rbt5 * (a(3) * bt * (C * bt3 - 6 * S * bt2 + 15 * C * bt - 15 * S) + 2 * a(7) * Sp * (bt * Cp * (30 + bt2) - (6 * (5 + bt2) * sq2 * Sp)));

    c(2, 2) = 50 * rbt6 * a(8) * (18 * bt2 - 36 * sq2 * BtCpSp - 6 * sq2 * bt2 * BtCpSp + 36 * Sp2 + 30 * bt2 * Sp2 + bt4 * Sp2);

    c(2, 1) = 30 * rbt5 * a(7) * (-3 * sq2 * bt2 + bt * (12 + bt2) * CpSp - 2 * sq2 * (3 + 2 * bt2) * Sp2);
    c(1, 2) = 30 * rbt5 * a(5) * (-3 * sq2 * bt2 + bt * (12 + bt2) * CpSp - 2 * sq2 * (3 + 2 * bt2) * Sp2);

    c(2, 0) = 5 * rbt4 * (a(2) * bt * (3 * S - 3 * C * bt + S * bt2) + 2 * Sp * a(8) * (Sp * bt2 + 6 * Sp - 3 * sq2 * bt * Cp));
    c(0, 2) = 5 * rbt4 * (a(4) * bt * (3 * S - 3 * C * bt + S * bt2) + 2 * Sp * a(8) * (Sp * bt2 + 6 * Sp - 3 * sq2 * bt * Cp));

    c(1, 1) = 18 * rbt4 * a(6) * (2 * Sp2 + bt2 * (1 + Sp2) - 2 * sq2 * bt * CpSp);

    c(1, 0) = 3 * rbt3 * (a(1) * bt * (C * bt - S) + 2 * Sp * a(5) * (bt * Cp - sq2 * Sp));
    c(0, 1) = 3 * rbt3 * (a(3) * bt * (C * bt - S) + 2 * Sp * a(7) * (bt * Cp - sq2 * Sp));

    c(0, 0) = rbt2 * (S * bt * (a(2) + a(4)) + 2 * a(8) * Sp2);

    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5 - i; j++)
            c(i, j) += p(i, j);
}

// MASTER 4.0 MM section 6.1 reconstruction.  The intranodal shape is a 13-term
// Legendre interpolant of the nodal solution (Eq. 6.1): even-parity terms follow
// directly from surface fluxes/currents and the node average (Eq. 6.6); the four
// cross terms need the corner fluxes, which solve the corner-point-balance
// system Eq. 6.7 with the per-node leakage Eq. 6.8 -- a diagonally dominant
// linear system swept with Gauss-Seidel.  No source iteration is involved: the
// polynomial interpolates the converged nodal solution, it does not re-solve
// the diffusion equation.  Coefficients land in the shared _c array using the
// same 15-slot layout the SENM path uses (slots (1,3) and (3,1) stay zero), so
// reconstructPinPower can evaluate either mode from _c/_p uniformly.
void PPR::driveMaster(int niter) {
    constexpr double r10 = 1.0 / 10.0;
    constexpr double r14 = 1.0 / 14.0;

    // 1. Even-parity coefficients (Eq. 6.2 and the first eight of Eq. 6.6).
    for (int lk = 0; lk < _nxyz; lk++) {
        for (int g = 0; g < _ng; g++) {
            const double pb  = aflux;
            const double pxr = phis(XDIR, RIGHT), pxl = phis(XDIR, LEFT);
            const double pyr = phis(YDIR, RIGHT), pyl = phis(YDIR, LEFT);
            const double jxr = getJoutRed(RIGHT, XDIR, lk, g), jxl = getJoutRed(LEFT, XDIR, lk, g);
            const double jyr = getJoutRed(RIGHT, YDIR, lk, g), jyl = getJoutRed(LEFT, YDIR, lk, g);

            c(0, 0) = pb;
            c(1, 0) = r10 * (6.0 * (pxr - pxl) + (jxr - jxl));
            c(2, 0) = r14 * (10.0 * (pxr + pxl) + (jxr + jxl) - 20.0 * pb);
            c(3, 0) = -r10 * ((pxr - pxl) + (jxr - jxl));
            c(4, 0) = -r14 * (3.0 * (pxr + pxl) + (jxr + jxl) - 6.0 * pb);
            c(0, 1) = r10 * (6.0 * (pyr - pyl) + (jyr - jyl));
            c(0, 2) = r14 * (10.0 * (pyr + pyl) + (jyr + jyl) - 20.0 * pb);
            c(0, 3) = -r10 * ((pyr - pyl) + (jyr - jyl));
            c(0, 4) = -r14 * (3.0 * (pyr + pyl) + (jyr + jyl) - 6.0 * pb);
            c(1, 3) = 0.0;
            c(3, 1) = 0.0;
            c(1, 1) = 0.0;
            c(1, 2) = 0.0;
            c(2, 1) = 0.0;
            c(2, 2) = 0.0;
        }
    }

    // 2. CPB Gauss-Seidel sweep on the per-node corner-flux copies.  Every copy
    // of a shared corner point evaluates the same 4-node balance, so the copies
    // converge to a common value.  Stencil-position (di,dj): di=0 west of the
    // corner (its own out-x face is EAST), di=1 east; dj=0 north (out-y SOUTH),
    // dj=1 south.  Reflection flags flip the logical surface/corner indices the
    // same way the SENM path's getPhis(RIGHT ^ xrev, ...) pattern does.
    for (int citer = 0; citer < niter; citer++) {
        double maxrel = 0.0;

        for (int lk = 0; lk < _nxyz; lk++) {
            int  idx[3][3];
            bool xrev[3][3], yrev[3][3];
            buildStencil(lk, idx, xrev, yrev);

            for (int g = 0; g < _ng; g++) {
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
                        const int dir = j * 2 + i;
                        double    num = 0.0;
                        double    den = 0.0;

                        for (int dj = 0; dj < 2; dj++) {
                            for (int di = 0; di < 2; di++) {
                                const int m = idx[i + di][j + dj];
                                if (m < 0 || m >= _nxyz) continue;

                                const bool xr = xrev[i + di][j + dj];
                                const bool yr = yrev[i + di][j + dj];

                                const int out_x = ((di == 0) ? RIGHT : LEFT) ^ (xr ? 1 : 0);
                                const int out_y = ((dj == 0) ? RIGHT : LEFT) ^ (yr ? 1 : 0);

                                const double pox = getPhis(out_x, XDIR, m, g);
                                const double pix = getPhis(out_x ^ 1, XDIR, m, g);
                                const double poy = getPhis(out_y, YDIR, m, g);
                                const double piy = getPhis(out_y ^ 1, YDIR, m, g);
                                const double jox = getJoutRed(out_x, XDIR, m, g);
                                const double joy = getJoutRed(out_y, YDIR, m, g);
                                const double pbm = _phif[m * _ng + g];

                                // Logical corner index of this corner point in node m.
                                const int icl = ((di == 0) ? 1 : 0) ^ (xr ? 1 : 0);
                                const int jcl = ((dj == 0) ? 1 : 0) ^ (yr ? 1 : 0);
                                const int adj1 = jcl * 2 + (1 - icl); // shares the y-edge
                                const int adj2 = (1 - jcl) * 2 + icl; // shares the x-edge
                                const double fadj1 = _phic[(m * 4 * _ng) + (g * 4) + adj1];
                                const double fadj2 = _phic[(m * 4 * _ng) + (g * 4) + adj2];

                                const double w = _xs.xsdf(g, m) / _g.hmesh(XDIR, m);
                                num += w * (5.0 * (pox + poy) + (pix + piy) + jox + joy - 6.0 * pbm - fadj1 - fadj2);
                                den += 4.0 * w;
                            }
                        }
                        if (den <= 0.0) continue;

                        const double fnew = num / den;
                        const double fold = phic(dir);
                        if (fold != 0.0)
                            maxrel = std::max(maxrel, std::abs((fnew - fold) / fold));
                        phic(dir) = fnew;
                    }
                }
            }
        }
        // The device arm reports PprBackend::iterations(); the host has to
        // report its own or the "did the two schemes stop in the same place?"
        // comparison has only one side.  Counted for the same reason drive()
        // counts, and read only through hostIterations().
        ++_host_iters;
        if (maxrel < kCornerFluxTolerance) break;
    }

    // 3. Cross terms from the converged corners (last four of Eq. 6.6).
    // MM axes: xi=+1 east, eta=+1 south, so phi1=SE, phi2=SW, phi3=NW, phi4=NE.
    for (int lk = 0; lk < _nxyz; lk++) {
        for (int g = 0; g < _ng; g++) {
            const double f1 = phic(SE), f2 = phic(SW), f3 = phic(NW), f4 = phic(NE);
            const double pxr = phis(XDIR, RIGHT), pxl = phis(XDIR, LEFT);
            const double pyr = phis(YDIR, RIGHT), pyl = phis(YDIR, LEFT);

            c(1, 1) = 0.25 * (f1 - f2 + f3 - f4);
            c(1, 2) = 0.25 * (f1 - f2 - f3 + f4 - 2.0 * (pxr - pxl));
            c(2, 1) = 0.25 * (f1 + f2 - f3 - f4 - 2.0 * (pyr - pyl));
            c(2, 2) = 0.25 * (f1 + f2 + f3 + f4 - 2.0 * (pxr + pxl + pyr + pyl) + 4.0 * aflux);
        }
    }
}

void PPR::drive(int niter) {
    if (_mode_master) {
        driveMaster(std::max(niter, 200));
        return;
    }

    CornerFluxSums previousCornerFlux;

    // 1. Iterate until the corner-flux balance is stable.
    for (int citer = 0; citer < niter; citer++) {
        // 2. Refresh fused coefficients a few times before re-enforcing continuity.
        for (int f = 0; f < kSourceSweepsPerIteration; f++) {
            for (int g = 0; g < _ng; g++) {
                for (int lk = 0; lk < _nxyz; lk++) {
                    updateFused(lk, g);
                }
            }
            updateSource();
        }
        updateCorner();

        // 3. Accumulate fuel-only corner fluxes and stop once all corners converge.
        CornerFluxSums currentCornerFlux;

        for (int lk = 0; lk < _nxyz; lk++) {
            if (!_g.IsFuel(lk)) continue;
            for (int g = 0; g < _ng; g++) {
                currentCornerFlux.nw += phic(NW);
                currentCornerFlux.sw += phic(SW);
                currentCornerFlux.ne += phic(NE);
                currentCornerFlux.se += phic(SE);
            }
        }

        const double err_nw = RelativeChange(currentCornerFlux.nw, previousCornerFlux.nw);
        const double err_sw = RelativeChange(currentCornerFlux.sw, previousCornerFlux.sw);
        const double err_ne = RelativeChange(currentCornerFlux.ne, previousCornerFlux.ne);
        const double err_se = RelativeChange(currentCornerFlux.se, previousCornerFlux.se);

        ++_host_iters;

        if (err_nw < kCornerFluxTolerance && err_sw < kCornerFluxTolerance && err_ne < kCornerFluxTolerance && err_se < kCornerFluxTolerance) {
            break;
        }

        previousCornerFlux = currentCornerFlux;
    }
}

void PPR::updateAxialLeakage() {
    const int nxy = _g.nxy();
    for (int lk = 0; lk < _nxyz; lk++) {
        const int l2d = lk % nxy;
        const int k   = lk / nxy;

        const int west2d  = _g.neibrb(WEST, l2d);
        const int east2d  = _g.neibrb(EAST, l2d);
        const int north2d = _g.neibrb(NORTH, l2d);
        const int south2d = _g.neibrb(SOUTH, l2d);

        const int northWest2d = (north2d >= 0) ? _g.neibrb(WEST, north2d) : -1;
        const int northEast2d = (north2d >= 0) ? _g.neibrb(EAST, north2d) : -1;
        const int southWest2d = (south2d >= 0) ? _g.neibrb(WEST, south2d) : -1;
        const int southEast2d = (south2d >= 0) ? _g.neibrb(EAST, south2d) : -1;

        const int west      = To3DNeighborIndex(west2d, k, nxy);
        const int east      = To3DNeighborIndex(east2d, k, nxy);
        const int north     = To3DNeighborIndex(north2d, k, nxy);
        const int south     = To3DNeighborIndex(south2d, k, nxy);
        const int northWest = To3DNeighborIndex(northWest2d, k, nxy);
        const int northEast = To3DNeighborIndex(northEast2d, k, nxy);
        const int southWest = To3DNeighborIndex(southWest2d, k, nxy);
        const int southEast = To3DNeighborIndex(southEast2d, k, nxy);

        for (int g = 0; g < _ng; g++) {
            // 1. Collect leakage for the center cell and its 8 in-plane neighbors.
            const double leak11 = zleak(lk);
            const double leak01 = getLeakage(west, g);
            const double leak21 = getLeakage(east, g);
            const double leak10 = getLeakage(north, g);
            const double leak12 = getLeakage(south, g);
            const double leak00 = getLeakage(northWest, g);
            const double leak20 = getLeakage(northEast, g);
            const double leak02 = getLeakage(southWest, g);
            const double leak22 = getLeakage(southEast, g);

            // 2. Fit the 2D leakage expansion used by the source update.
            l(0, 0) = leak11;
            l(1, 0) = (0.25) * (leak21 - leak01);
            l(0, 1) = (0.25) * (leak12 - leak10);
            l(2, 0) = 0.0833333333 * (leak01 - 2 * leak11 + leak21);
            l(0, 2) = 0.0833333333 * (leak10 - 2 * leak11 + leak12);
            l(1, 1) = 0.0625000000 * (leak00 - leak02 - leak20 + leak22);
            l(1, 2) = 0.0208333333 * (-leak00 + 2 * leak01 - leak02 + leak20 - 2 * leak21 + leak22);
            l(2, 1) = 0.0208333333 * (-leak00 + 2 * leak10 + leak02 - leak20 - 2 * leak12 + leak22);
            l(2, 2) = 0.0069444444 * (leak00 - 2 * leak01 + leak02 - 2 * leak10 + 4 * leak11 - 2 * leak12 + leak20 - 2 * leak21 + leak22);
        }
    }
}

void PPR::updateSource() {
    const int    ng    = _ng;
    const int    nxyz  = _nxyz;
    const double reigv = _reigv;

    for (int lk = 0; lk < nxyz; ++lk) {
        // Precompute all 4 XS coefficients for this node (ng=2: 2×2 combos)
        double coeff[2][2]; // coeff[g][to_g]
        for (int g = 0; g < ng; ++g)
            for (int to_g = 0; to_g < ng; ++to_g)
                coeff[g][to_g] = _xs.xssm(g, to_g, lk) + _xs.chif(to_g, lk) * _xs.xsnf(g, lk) * reigv;

        // Pointers to c arrays for g=0 and g=1
        const double* c0 = &_c[(lk * 15 * ng) + (0 * 15)];
        const double* c1 = &_c[(lk * 15 * ng) + (1 * 15)];

        // Pointers to q arrays for to_g=0 and to_g=1
        double* q0 = &_q[(lk * 15 * ng) + (0 * 15)];
        double* q1 = &_q[(lk * 15 * ng) + (1 * 15)];

        // q(to_g) = sum_g coeff[g][to_g] * c(g) - leakage
        // Directly assign instead of zero + accumulate
        const double c00 = coeff[0][0], c10 = coeff[1][0]; // → to_g=0
        const double c01 = coeff[0][1], c11 = coeff[1][1]; // → to_g=1

        for (int k = 0; k < 15; ++k) {
            q0[k] = c00 * c0[k] + c10 * c1[k];
            q1[k] = c01 * c0[k] + c11 * c1[k];
        }

        // Subtract leakage: q(lk, g, i, j) -= l(i, j) for i<3, j<3
        // l layout: _l[(lk * 9 * ng) + (g * 9) + i*3 + j]
        const double* l0 = &_l[(lk * 9 * ng) + (0 * 9)];
        const double* l1 = &_l[(lk * 9 * ng) + (1 * 9)];

        // Map upper-triangular (i,j) for i<3,j<3 to flat index:
        // (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=5, (1,1)=6, (1,2)=7, (2,0)=9, (2,1)=10, (2,2)=11
        static constexpr int qidx[9] = {0, 1, 2, 5, 6, 7, 9, 10, 11};
        for (int k = 0; k < 9; ++k) {
            q0[qidx[k]] -= l0[k];
            q1[qidx[k]] -= l1[k];
        }
    }
}

void PPR::updateCorner() {
    for (int lk = 0; lk < _nxyz; lk++) {
        int  idx[3][3];
        bool xrev[3][3], yrev[3][3];
        buildStencil(lk, idx, xrev, yrev);

        for (int g = 0; g < _ng; g++) {
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    int    dir      = j * 2 + i;
                    double jnet_x11 = jnetX(idx[i + 0][j + 0], g, 1, 1, xrev[i + 0][j + 0], yrev[i + 0][j + 0]);
                    double jnet_x12 = jnetX(idx[i + 0][j + 1], g, 1, -1, xrev[i + 0][j + 1], yrev[i + 0][j + 1]);
                    double jnet_x21 = jnetX(idx[i + 1][j + 0], g, -1, 1, xrev[i + 1][j + 0], yrev[i + 1][j + 0]);
                    double jnet_x22 = jnetX(idx[i + 1][j + 1], g, -1, -1, xrev[i + 1][j + 1], yrev[i + 1][j + 1]);

                    double jnet_y11 = jnetY(idx[i + 0][j + 0], g, 1, 1, xrev[i + 0][j + 0], yrev[i + 0][j + 0]);
                    double jnet_y12 = jnetY(idx[i + 0][j + 1], g, 1, -1, xrev[i + 0][j + 1], yrev[i + 0][j + 1]);
                    double jnet_y21 = jnetY(idx[i + 1][j + 0], g, -1, 1, xrev[i + 1][j + 0], yrev[i + 1][j + 0]);
                    double jnet_y22 = jnetY(idx[i + 1][j + 1], g, -1, -1, xrev[i + 1][j + 1], yrev[i + 1][j + 1]);

                    double xdiff = jnet_x11 - jnet_x21 + jnet_x12 - jnet_x22;
                    double ydiff = jnet_y11 - jnet_y12 + jnet_y21 - jnet_y22;

                    phic(dir) = phic(dir) + 0.25 * (xdiff + ydiff);
                }
            }
        }
    }
}

double PPR::phig(int lk, int g, double x, double y) {
    x = std::clamp(x, -1.0, 1.0);
    y = std::clamp(y, -1.0, 1.0);

    // Pre-compute Legendre polynomials for x and y (avoids per-term switch dispatch)
    const double x2 = x * x, y2 = y * y;
    const double Lx[5] = {1.0, x, 0.5 * (3.0 * x2 - 1.0), 0.5 * (5.0 * x2 - 3.0) * x,
                          0.125 * (35.0 * x2 * x2 - 30.0 * x2 + 3.0)};
    const double Ly[5] = {1.0, y, 0.5 * (3.0 * y2 - 1.0), 0.5 * (5.0 * y2 - 3.0) * y,
                          0.125 * (35.0 * y2 * y2 - 30.0 * y2 + 3.0)};

    if (_mode_master) {
        double cflux = 0.0;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5 - i; ++j)
                cflux += c(i, j) * Lx[i] * Ly[j];
        return cflux;
    }

    double particularFlux = 0.0;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5 - i; ++j)
            particularFlux += p(i, j) * Lx[i] * Ly[j];

    // Compute sinh/cosh via exp pairs (4 exp calls instead of 8 sinh/cosh)
    const double bt = Bt;
    const double e1 = std::exp(bt * x), re1 = 1.0 / e1;
    const double e2 = std::exp(bt * y), re2 = 1.0 / e2;
    const double e3 = std::exp(bt * x * rsq2), re3 = 1.0 / e3;
    const double e4 = std::exp(bt * y * rsq2), re4 = 1.0 / e4;

    const double sx = 0.5 * (e1 - re1), cx = 0.5 * (e1 + re1);
    const double sy = 0.5 * (e2 - re2), cy = 0.5 * (e2 + re2);
    const double sxr = 0.5 * (e3 - re3), cxr = 0.5 * (e3 + re3);
    const double syr = 0.5 * (e4 - re4), cyr = 0.5 * (e4 + re4);

    double homogeneousFlux =
        a(1) * sx + a(2) * cx + a(3) * sy + a(4) * cy + a(5) * sxr * cyr + a(6) * sxr * syr + a(7) * cxr * syr + a(8) * cxr * cyr;

    return particularFlux + homogeneousFlux;
}

// Surface current in direction dir (XDIR/YDIR): shared bounds-check + reflection
// + scaling; only the polynomial and the mesh direction differ per component.
double PPR::jnetDir(int dir, int lk, int g, double x, double y, bool xrev, bool yrev) {
    if (lk < 0 || lk >= _g.nxyz()) return 0.0;
    if (xrev) x = -x;
    if (yrev) y = -y;
    double j = (dir == XDIR)
                   ? c(1, 0) + y * c(1, 1) + 0.5 * (3 * y * y - 1) * c(1, 2) + 3 * x * c(2, 0) + 3 * x * y * c(2, 1) + 1.5 * x * (3 * y * y - 1) * c(2, 2) + 0.5 * (15 * x * x - 3) * c(3, 0) + 0.125 * (-60 * x + 140 * x * x * x) * c(4, 0)
                   : c(0, 1) + 3 * y * c(0, 2) + 0.5 * (15 * y * y - 3) * c(0, 3) + 0.125 * (140 * y * y * y - 60 * y) * c(0, 4) + x * c(1, 1) + 3 * x * y * c(1, 2) + 0.5 * (3 * x * x - 1) * c(2, 1) + 1.5 * (3 * x * x - 1) * y * c(2, 2);
    return j * -2.0 * _xs.xsdf(g, lk) / _g.hmesh(dir, lk);
}

double PPR::jnetX(int lk, int g, double x, double y, bool xrev, bool yrev) {
    return jnetDir(XDIR, lk, g, x, y, xrev, yrev);
}

double PPR::jnetY(int lk, int g, double x, double y, bool xrev, bool yrev) {
    return jnetDir(YDIR, lk, g, x, y, xrev, yrev);
}

void PPR::reconstructPinPower(bool use_quadrature, bool reconstruct_flux,
                              bool materialize_pin_map) {
    // Take the cohort's quadrature table once (geometry-dependent, never
    // changes).  Lazily, exactly as before: a run that never reconstructs a pin
    // power still does not pay for the table.
    if (use_quadrature && !_pin_quad_table) acquireQuadratureTable();

    // WP6 stage D.  Returns false having written nothing on every refusal --
    // arm off, drive on the host, MASTER mode, a CUDA failure -- and then the
    // loop below runs exactly as it always did.
    if (reconstructPinPowerGpu(use_quadrature, reconstruct_flux, materialize_pin_map)) return;

    // THE REPAIR, AND WHY IT IS NOT OPTIONAL.  The drive elided the coefficient
    // download on the promise that this call would consume p/a/bt on the
    // device.  It did not -- a CUDA failure, or a caller that asked for the
    // pointwise mode -- so the host arrays hold the PREVIOUS statepoint's
    // coefficients and the loop below would reconstruct last statepoint's pin
    // power with this statepoint's normalisation.  Rebuilding them is exact and
    // costs one host reset+drive; the receipt counts it, because a run that
    // repairs often should not have elided.
    if (_coeffs_device_only) {
        _coeffs_device_only = false;
        if (_gpu != nullptr) _gpu->noteReconRepair();
        reset(_reigv, _jnet, _phif, _phis);
        drive(_last_niter > 0 ? _last_niter : 100);
    }

    const int ng    = _ng;
    const int nxy   = _g.nxy();
    const int nz    = _g.nz();
    const int nxya  = _g.nxya();
    const int ndiv  = _g.ndivxy();
    const int ndiv2 = ndiv * ndiv;
    const int npins = _g.npins();
    const int npina = npins * npins;
    const int kbc   = _g.kbc();
    const int kec   = _g.kec();

    double* pphif  = reconstruct_flux ? _g.PinFlux() : nullptr;
    double* ppower = _g.PinPower();

    // Clear arrays
    if (reconstruct_flux)
        std::fill_n(pphif, static_cast<size_t>(nxya) * nz * ng * npina, 0.0);
    std::fill_n(ppower, static_cast<size_t>(nxya) * nz * npina, 0.0);

    const double inv_npins  = 1.0 / npins;
    const double pin_area   = inv_npins * inv_npins;
    const double area_coeff = 1.0 / (4.0 * ndiv * ndiv);

// Process each fuel axial plane and assembly
#pragma omp parallel for collapse(2) schedule(dynamic) if ((kec - kbc) * nxya > 64)
    for (int k = kbc; k < kec; ++k) {
        for (int la = 0; la < nxya; ++la) {
            const int lka = la + nxya * k;

            // Find a valid fuel node to get the composition / form functions
            int ref_lk = -1;
            for (int li = 0; li < ndiv2; ++li) {
                int l = _g.latol(li, la);
                if (l >= 0) {
                    int lk_tmp = l + nxy * k;
                    if (_g.IsFuel(lk_tmp)) {
                        ref_lk = lk_tmp;
                        break;
                    }
                }
            }
            if (ref_lk < 0) continue;

            // Get burn-interpolated form functions from Chiffon model
            size_t      mi    = _xs.comp(ref_lk);
            // CONST, and .at(0): the parsed models are shared by every Driver
            // in the process (XsLibrary.h).  `auto&` selected the mutating
            // overloads -- `_refr_dpts[0]` inserts an empty ctype row when the
            // key is missing -- which is a write into another deck's library.
            // XSSet::Initialize already refuses any library whose node models
            // lack ctype 0, so .at(0) cannot be the throwing case here.
            const auto& model = _xs.models()[mi];
            int         burn  = _xs.burn(ref_lk);

            // Interpolate fmap/gmap between bounding burnup points
            const auto& refrMap = model._refr_dpts.at(0);
            auto  hiIt    = refrMap.lower_bound(burn);
            auto  loIt    = hiIt;

            if (hiIt == refrMap.end()) {
                loIt = std::prev(refrMap.end());
                hiIt = loIt;
            } else if (hiIt == refrMap.begin()) {
                loIt = refrMap.begin();
            } else {
                loIt = std::prev(hiIt);
            }

            const auto& loDpt = model.GetDepletionPoint(loIt->second);
            const auto& hiDpt = model.GetDepletionPoint(hiIt->second);

            // Compute interpolation factor
            double alpha = 0.0;
            if (loIt != hiIt && hiDpt.burnKey() != loDpt.burnKey())
                alpha = static_cast<double>(burn - loDpt.burnKey()) / static_cast<double>(hiDpt.burnKey() - loDpt.burnKey());

            // Build interpolated fmap / gmap. Reuse thread-local buffers to avoid
            // allocating once per assembly-plane in the OpenMP loop.
            static thread_local std::vector<double> gmap_interp;
            static thread_local std::vector<double> fmap_interp;
            gmap_interp.resize(loDpt._gmap.size());
            if (reconstruct_flux)
                fmap_interp.resize(loDpt._fmap.size());
            else
                fmap_interp.clear();

            for (size_t fi = 0; fi < fmap_interp.size(); ++fi)
                fmap_interp[fi] = loDpt._fmap[fi] + alpha * (hiDpt._fmap[fi] - loDpt._fmap[fi]);
            for (size_t gi = 0; gi < gmap_interp.size(); ++gi)
                gmap_interp[gi] = loDpt._gmap[gi] + alpha * (hiDpt._gmap[gi] - loDpt._gmap[gi]);

            // For each pin in the assembly
            for (int py = 0; py < npins; ++py) {
                for (int px = 0; px < npins; ++px) {
                    const int pin_idx = py * npins + px;

                    double hom_flux[8]    = {};
                    double power_integral = 0.0;
                    bool   has_valid_node = false;

                    if (!use_quadrature) {
                        // === POINTWISE MODE: evaluate phig at pin center ===
                        // Pin center in assembly-normalised [0,1] coords
                        const double xc_asm = (px + 0.5) * inv_npins;
                        const double yc_asm = (py + 0.5) * inv_npins;

                        // Find which sub-node contains the center
                        const int di = std::min(static_cast<int>(xc_asm * ndiv), ndiv - 1);
                        const int dj = std::min(static_cast<int>(yc_asm * ndiv), ndiv - 1);
                        const int li = dj * ndiv + di;
                        const int l  = _g.latol(li, la);
                        if (l >= 0) {
                            has_valid_node = true;
                            const int lk   = l + nxy * k;

                            // Map to local [-1,1] coords within sub-node
                            const double inv_ndiv = 1.0 / ndiv;
                            const double xlocal   = 2.0 * ndiv * xc_asm - 2.0 * di - 1.0;
                            const double ylocal   = 2.0 * ndiv * yc_asm - 2.0 * dj - 1.0;

                            for (int g = 0; g < ng; ++g) {
                                double phi_val = phig(lk, g, xlocal, ylocal);
                                hom_flux[g]    = phi_val;
                                power_integral += phi_val * _xs.xskf(g, lk);
                            }
                        }
                    } else {
                        // === QUADRATURE MODE: 3x3 Gauss-Legendre integration ===
                        const auto& pin_info = (*_pin_quad_table)[pin_idx];

                        for (const auto& ovl : pin_info.overlaps) {
                            const int li = ovl.dj * ndiv + ovl.di;
                            const int l  = _g.latol(li, la);
                            if (l < 0) continue;
                            has_valid_node = true;
                            const int lk   = l + nxy * k;

                            if (_mode_master) {
                                // MASTER mode: pure Legendre interpolant in _c.
                                for (int g = 0; g < ng; ++g) {
                                    const double* c_base = &_c[(lk * 15 * _ng) + (g * 15)];
                                    double        integ  = 0.0;
                                    for (int qq = 0; qq < 9; ++qq) {
                                        const auto& qp    = ovl.qpts[qq];
                                        double      cflux = 0.0;
                                        for (int t = 0; t < 15; ++t)
                                            cflux += c_base[t] * qp.leg[t];
                                        integ += qp.wt * cflux;
                                    }
                                    const double flux_contrib = integ * ovl.dx_h * ovl.dy_h * area_coeff;
                                    hom_flux[g] += flux_contrib;
                                    power_integral += flux_contrib * _xs.xskf(g, lk);
                                }
                                continue;
                            }

                            for (int g = 0; g < ng; ++g) {
                                const double  bt     = _bt[lk * _ng + g];
                                const double* p_base = &_p[(lk * 15 * _ng) + (g * 15)];
                                const double* a_base = &_a[(lk * 8 * _ng) + (g * 8)];

                                double sx_arr[3], cx_arr[3], sxr_arr[3], cxr_arr[3];
                                double sy_arr[3], cy_arr[3], syr_arr[3], cyr_arr[3];

                                for (int qi = 0; qi < 3; ++qi) {
                                    const double ex   = std::exp(bt * ovl.qpts[qi * 3].xq);
                                    const double rex  = 1.0 / ex;
                                    const double exr  = std::exp(bt * ovl.qpts[qi * 3].xq * rsq2);
                                    const double rexr = 1.0 / exr;
                                    sx_arr[qi]        = 0.5 * (ex - rex);
                                    cx_arr[qi]        = 0.5 * (ex + rex);
                                    sxr_arr[qi]       = 0.5 * (exr - rexr);
                                    cxr_arr[qi]       = 0.5 * (exr + rexr);
                                }
                                for (int qj = 0; qj < 3; ++qj) {
                                    const double ey   = std::exp(bt * ovl.qpts[qj].yq);
                                    const double rey  = 1.0 / ey;
                                    const double eyr  = std::exp(bt * ovl.qpts[qj].yq * rsq2);
                                    const double reyr = 1.0 / eyr;
                                    sy_arr[qj]        = 0.5 * (ey - rey);
                                    cy_arr[qj]        = 0.5 * (ey + rey);
                                    syr_arr[qj]       = 0.5 * (eyr - reyr);
                                    cyr_arr[qj]       = 0.5 * (eyr + reyr);
                                }

                                double integ = 0.0;
                                for (int qi = 0; qi < 3; ++qi) {
                                    for (int qj = 0; qj < 3; ++qj) {
                                        const auto& qp = ovl.qpts[qi * 3 + qj];

                                        double pFlux = 0.0;
                                        for (int t = 0; t < 15; ++t)
                                            pFlux += p_base[t] * qp.leg[t];

                                        double hFlux =
                                            a_base[0] * sx_arr[qi] + a_base[1] * cx_arr[qi] + a_base[2] * sy_arr[qj] + a_base[3] * cy_arr[qj] + a_base[4] * sxr_arr[qi] * cyr_arr[qj] + a_base[5] * sxr_arr[qi] * syr_arr[qj] + a_base[6] * cxr_arr[qi] * syr_arr[qj] + a_base[7] * cxr_arr[qi] * cyr_arr[qj];

                                        integ += qp.wt * (pFlux + hFlux);
                                    }
                                }
                                const double flux_contrib = integ * ovl.dx_h * ovl.dy_h * area_coeff;
                                hom_flux[g] += flux_contrib;
                                power_integral += flux_contrib * _xs.xskf(g, lk);
                            }
                        }
                    } // end quadrature vs pointwise

                    if (!has_valid_node) {
                        if (reconstruct_flux) {
                            for (int g = 0; g < ng; ++g)
                                pphif[static_cast<size_t>(lka) * ng * npina + g * npina + pin_idx] = std::numeric_limits<double>::quiet_NaN();
                        }
                        ppower[static_cast<size_t>(lka) * npina + pin_idx] = std::numeric_limits<double>::quiet_NaN();
                        continue;
                    }

                    if (use_quadrature) {
                        const double inv_pin_area = 1.0 / pin_area;
                        if (reconstruct_flux) {
                            for (int g = 0; g < ng; ++g) {
                                double phi_hom                                                     = hom_flux[g] * inv_pin_area;
                                double fval                                                        = fmap_interp[g * npina + pin_idx];
                                pphif[static_cast<size_t>(lka) * ng * npina + g * npina + pin_idx] = phi_hom * fval;
                            }
                        }
                        double gmap_val                                    = gmap_interp[pin_idx];
                        ppower[static_cast<size_t>(lka) * npina + pin_idx] = power_integral * inv_pin_area * gmap_val;
                    } else {
                        if (reconstruct_flux) {
                            for (int g = 0; g < ng; ++g) {
                                double fval                                                        = fmap_interp[g * npina + pin_idx];
                                pphif[static_cast<size_t>(lka) * ng * npina + g * npina + pin_idx] = hom_flux[g] * fval;
                            }
                        }
                        double gmap_val                                    = gmap_interp[pin_idx];
                        ppower[static_cast<size_t>(lka) * npina + pin_idx] = power_integral * gmap_val;
                    }
                }
            }
        }
    }

    // Normalise pin power by the volume-weighted average nodal power density.
    // This keeps the node-wise power distribution on the same 1.0-based scale as
    // the nodal output, and pin-wise variation comes only from the reconstructed
    // pin factor (homogeneous pin power × gmap).
    // Deterministic reduction (see rasbery_det_chunks in Geometry.h): fixed
    // chunking + ordered accumulation of the partials, so the normalisation
    // constant -- and therefore every pin power written out -- is bitwise
    // identical for any OMP_NUM_THREADS.
    const int           nchunk_ppr = rasbery_det_chunks(_nxyz);
    std::vector<double> pw_partial(static_cast<size_t>(nchunk_ppr), 0.0);
    std::vector<double> vol_partial(static_cast<size_t>(nchunk_ppr), 0.0);

#pragma omp parallel for schedule(dynamic) if (_nxyz > rasbery_omp_gate)
    for (int c = 0; c < nchunk_ppr; ++c) {
        const int lb   = rasbery_det_chunk_begin(_nxyz, nchunk_ppr, c);
        const int le   = rasbery_det_chunk_begin(_nxyz, nchunk_ppr, c + 1);
        double    apw  = 0.0;
        double    avol = 0.0;
        for (int lk = lb; lk < le; ++lk) {
            if (!_g.IsFuel(lk)) continue;
            const double vol = _g.vol(lk);
            if (vol <= 1.0e-20) continue;

            double node_power = 0.0;
            for (int g = 0; g < ng; ++g)
                node_power += _xs.xskf(g, lk) * _phif[lk * ng + g];

            apw += node_power * vol;
            avol += vol;
        }
        pw_partial[static_cast<size_t>(c)]  = apw;
        vol_partial[static_cast<size_t>(c)] = avol;
    }

    double nodal_power_sum = 0.0;
    double fuel_vol_sum    = 0.0;
    for (int c = 0; c < nchunk_ppr; ++c) {
        nodal_power_sum += pw_partial[static_cast<size_t>(c)];
        fuel_vol_sum += vol_partial[static_cast<size_t>(c)];
    }

    const double avg_nodal_power = (fuel_vol_sum > 1.0e-30) ? nodal_power_sum / fuel_vol_sum : 1.0;
    const double inv_avg_power   = 1.0 / avg_nodal_power;

#pragma omp parallel for collapse(2) schedule(static) if ((kec - kbc) * nxya > 500)
    for (int k = kbc; k < kec; ++k) {
        for (int la = 0; la < nxya; ++la) {
            const int lka = la + nxya * k;
            for (int pi = 0; pi < npina; ++pi)
                ppower[static_cast<size_t>(lka) * npina + pi] *= inv_avg_power;
        }
    }

    // Compute peaking factors.
    // FRP: max of Z-averaged radial pin power
    std::vector<double> radial_power(static_cast<size_t>(nxya) * npina, 0.0);
    std::vector<double> radial_hz(nxya, 0.0);
    for (int k = kbc; k < kec; ++k) {
        const double hz_k = _g.hz(k);
        for (int la = 0; la < nxya; ++la) {
            const int lka   = la + nxya * k;
            bool      valid = false;
            for (int li = 0; li < ndiv2 && !valid; ++li) {
                int l = _g.latol(li, la);
                if (l >= 0 && _g.IsFuel(l + nxy * k)) valid = true;
            }
            if (!valid) continue;
            radial_hz[la] += hz_k;
            for (int pi = 0; pi < npina; ++pi)
                radial_power[static_cast<size_t>(la) * npina + pi] +=
                    ppower[static_cast<size_t>(lka) * npina + pi] * hz_k;
        }
    }

    double frp = 0.0;
    for (int la = 0; la < nxya; ++la) {
        if (radial_hz[la] <= 0.0) continue;
        const double inv_hz = 1.0 / radial_hz[la];
        for (int pi = 0; pi < npina; ++pi)
            frp = std::max(frp, radial_power[static_cast<size_t>(la) * npina + pi] * inv_hz);
    }
    _g.frp() = frp;

    // FQP: max 3-D pin power
    double fqp = 0.0;
    for (int k = kbc; k < kec; ++k)
        for (int la = 0; la < nxya; ++la) {
            const int lka = la + nxya * k;
            for (int pi = 0; pi < npina; ++pi)
                fqp = std::max(fqp, ppower[static_cast<size_t>(lka) * npina + pi]);
        }
    _g.fqp() = fqp;
}
