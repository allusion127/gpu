#include "BICGCMFD.h"

#include "HostPinRegistry.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define flux(ig, l)  (flux[(l) * _g.ng() + ig])
#define aflux(ig, l) (aflux[(l) * _g.ng() + ig])

using namespace rasbery;

namespace {
// Capture for the device-sweep port (RASBERY_CMFD_DUMP=<prefix>): the first
// post-warmup drive() call's entry state plus a record after every sweep.
// The offline form probe replays these to mine the contraction forms gcc
// actually emitted for wiel/updls -- same methodology, same reason as
// RASBERY_XSRECON_DUMP (quoting source does not pin contraction).
struct CmfdDump {
    std::FILE* f       = nullptr;
    bool       armed   = false;
    bool       done    = false;
    CmfdDump() {
        armed = std::getenv("RASBERY_CMFD_DUMP") != nullptr;
    }
    void openEntry() {
        const char* p = std::getenv("RASBERY_CMFD_DUMP");
        f             = std::fopen((std::string(p) + ".cmfd").c_str(), "wb");
    }
    void write(const double* p, size_t n) {
        if (f) std::fwrite(p, sizeof(double), n, f);
    }
    void close() {
        if (f) std::fclose(f);
        f    = nullptr;
        done = true;
    }
};
CmfdDump g_cmfd_dump;

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string s(value);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
             s == "false" || s == "FALSE");
}

bool envFlagEnabledDefaultOn(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return true;
    const std::string s(value);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
             s == "false" || s == "FALSE");
}
} // namespace

BICGCMFD::BICGCMFD(Geometry& g, XSSet& x)
    : CMFD(g, x),
      _ls(std::make_unique<BICGSolver>(_g)),
      _nodal(nullptr),
      _udiag(static_cast<size_t>(_g.ng2()) * static_cast<size_t>(_g.nxyz())) {
    // Inner BiCGSTAB budget. The defaults are the historical ones; the two
    // overrides exist so the pair can be A/B-ed against outer-iteration count
    // and k_eff without a rebuild. On the GPU the inner loop is one graph
    // launch, so a deeper, tighter inner solve amortises the launch latency
    // that a 2-iteration loop cannot.
    _epsbicg  = 0.1;
    _nmaxbicg = 3;
    if (const char* nmax_env = std::getenv("RASBERY_BICG_NMAX")) {
        const int requested = std::atoi(nmax_env);
        if (requested > 0) _nmaxbicg = requested;
    }
    if (const char* eps_env = std::getenv("RASBERY_BICG_EPS")) {
        const double requested = std::atof(eps_env);
        if (requested > 0.0) _epsbicg = requested;
    }

    _eshift     = 0.04;
    iter        = 0;
    _wiel_sweep = 0;
    _bicg_iters = 0;
}

BackendCounters BICGCMFD::backendCounters() const {
    if (_ls == nullptr) return BackendCounters{};
    if (CudaBatchArena* shared = _ls->arena(); shared != nullptr)
        return shared->counters();
    return _ls->cudaCounters();
}

int BICGCMFD::batchSlot() const {
    return _ls != nullptr ? _ls->batchSlot() : -1;
}

void BICGCMFD::setIterLim(int maxls, double epsls) {
    _nmaxbicg = maxls;
    _epsbicg  = epsls;
}

BICGCMFD::~BICGCMFD() {
    // Order matters, and it is the reason this is no longer `= default`.
    //
    // Members are destroyed in reverse declaration order, which would free the
    // four page-locked vectors BEFORE _ls (declared first) hands its arena slot
    // back.  Tear the backend handles down first, then release the leases.
    //
    // What makes this the conservative drain Sec 6.4 permits in place of
    // per-stream event tracking: every solve this instance issues is
    // synchronous from the instance's side -- solveSweeps/drive return only
    // after the batch that carried this slot completed and its downloads
    // landed -- so once the last one returned, no DMA is in flight on these
    // buffers.  Releasing the slot first additionally stops a lingering
    // launcher from waiting on an instance that is going away.
    _nodal.reset();
    _ls.reset();

    rasberyUnpinHost(_pin_udiag);
    rasberyUnpinHost(_pin_sweep_chif);
    rasberyUnpinHost(_pin_sweep_xsnf);
    rasberyUnpinHost(_pin_sweep_vol);
}

void BICGCMFD::setEshift(double eshift) {
    _eshift = eshift;
}

void BICGCMFD::wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, double& errl2) {

    double gamman = 0;
    double gammad = 0;
    double err    = 0;

    for (int l = 0; l < _g.nxyz(); l++) {
        double psid = psi(l);
        psi(l)      = _x.xsnf(0, l) * flux(0, l) + _x.xsnf(1, l) * flux(1, l);
        psi(l)      = psi(l) * _g.vol(l);

        double err1 = psi(l) - psid;
        err         = err + err1 * err1;
        gammad += psid * psi(l);
        gamman += psi(l) * psi(l);
    }

    errl2 = err;

    // compute new eigenvalue
    //
    // The Wielandt update needs gamma = gammad/gamman with gammad = <psi_old,psi_new>.
    // On the first call after a state reset psi_old is identically zero, so gammad is
    // exactly 0: gamma becomes 0 and the shift extrapolation degenerates (and the
    // errl2 normalisation below divides by zero). gamman is a sum of squares and can
    // never be negative, so the old `gamman < 0` test was dead code; what actually
    // needs guarding is gamman == 0 (null fission source). Both degenerate cases now
    // fall back to the Rayleigh-quotient branch, which is well defined there.
    const bool gamma_usable = (gammad > 0.0) && (gamman > 0.0);
    if (icy < 0 || !gamma_usable) {
        double sumf = 0;
        double summ = 0;
        for (int l = 0; l < _g.nxyz(); l++) {
            for (int ig = 0; ig < _g.ng(); ig++) {
                double ab = CMFD::axb(ig, l, flux);
                summ      = summ + ab;
            }
            sumf += psi(l);
            summ += psi(l) * reigvs;
        }
        eigv = sumf / summ;
    } else {
        double gamma = gammad / gamman;
        eigv         = 1 / (reigv * gamma + (1 - gamma) * reigvs);
    }
    reigv = 1 / eigv;

    // Normalise the fission-source change by the fission-source norm. gammad is the
    // natural scale but is 0 on the first call (see above) and can turn negative when
    // the iterate flips sign; fall back to gamman = ||psi_new||^2, which is positive
    // whenever there is any fission source at all. Without this, errl2 came out as
    // inf/NaN on the first sweep and the outer loop's `errl2 < _epsl2` test was
    // evaluating a NaN comparison.
    const double err_scale = (gammad > 0.0) ? gammad : gamman;
    errl2                  = (err_scale > 0.0) ? sqrt(abs(errl2 / err_scale)) : 0.0;

    double eigvs = eigv;
    reigvs       = 0;

    if (icy >= 0) {
        eigvs += _eshift;
        if (_eshift != 0.0) reigvs = 1. / eigvs;
    }
}

void BICGCMFD::upddtil() {

    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}

void BICGCMFD::upddhat(double* flux, double* jnet) {

    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddhat(ls, flux, jnet);
    }
}

bool BICGCMFD::canUseDeviceAssembly() const {
    // The arena assembly is deliberately coupled to the resident multi-sweep
    // path.  Otherwise setls() could skip the host operator and the pristine
    // host drive loop would observe stale diag/cc arrays.
    return envFlagEnabledDefaultOn("RASBERY_GPU_CMFD_ASSEMBLY") &&
           envFlagEnabled("RASBERY_GPU_CMFD_SWEEP") &&
           std::getenv("RASBERY_CMFD_DUMP") == nullptr &&
           _g.ng() == 2 && _ls->usingCuda() && _ls->arena() != nullptr &&
           _wiel_sweep >= WIELANDT_WARMUP_SWEEPS;
}

void BICGCMFD::assembleHostLinearSystem(const double& eigv) {
    // Zero shift does not need an extra unshifted-diagonal copy/update pass.
    if (_eshift == 0.0) {
        for (int l = 0; l < _g.nxyz(); ++l) setls(l);
        _ls->facilu(_diag, _cc);
        return;
    }

    const double reigvs = 1. / (eigv + _eshift);
    for (int l = 0; l < _g.nxyz(); ++l) {
        setls(l);
        updls(l, reigvs);
    }
    _ls->facilu(_diag, _cc);
}

void BICGCMFD::setls(const double& eigv) {
    _device_assembly_pending = canUseDeviceAssembly();
    if (_device_assembly_pending) return;
    assembleHostLinearSystem(eigv);
}

void BICGCMFD::setls(const int& l) {
    CMFD::setls(l);
    if (_eshift == 0.0) return;

    for (int ige = 0; ige < _g.ng(); ++ige) {
        for (int igs = 0; igs < _g.ng(); ++igs) {
            udiag(igs, ige, l) = diag(igs, ige, l);
        }
    }
}

void BICGCMFD::updls(const double& reigvs) {
    if (_eshift == 0.0) return;

    for (int l = 0; l < _g.nxyz(); ++l) {
        updls(l, reigvs);
    }
}
void BICGCMFD::updls(const int& l, const double& reigvs) {
    if (_eshift == 0.0) return;

    for (int ige = 0; ige < _g.ng(); ++ige) {
        for (int igs = 0; igs < _g.ng(); ++igs) {
            diag(igs, ige, l) = udiag(igs, ige, l) - (_x.chif(ige, l) * _x.xsnf(igs, l) * reigvs * _g.vol(l));
        }
    }
}

void BICGCMFD::updjnet(double* flux, double* jnet) {

    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::updjnet(ls, flux, jnet);
    }
}

void BICGCMFD::updpsi(const double* flux) {

    for (int l = 0; l < _g.nxyz(); ++l) {
        CMFD::updpsi(l, flux);
    }
}

// The device-resident sweep loop (RASBERY_GPU_CMFD_SWEEP).  One graph launch
// runs up to the remaining sweep budget; the loop below only spins again for
// the negative-flux retry pathology or the degenerate-gamma hand-back, both
// of which the plain host loop also treats as exceptional.  Delegation only
// happens in the Wielandt regime (the caller checks the warm-up), so the
// Rayleigh branch here is the gamma-degenerate fallback, not the schedule.
bool BICGCMFD::driveDeviceSweeps(double& eigv, double* flux, double& errl2) {
    const int nxyz = _g.nxyz();
    const int ng   = _g.ng();
    const bool use_device_assembly = _device_assembly_pending;
    auto finish = [&](bool result) {
        _device_assembly_pending = false;
        return result;
    };
    _sweep_chif.resize(static_cast<size_t>(ng) * nxyz);
    _sweep_xsnf.resize(static_cast<size_t>(ng) * nxyz);
    _sweep_vol.resize(static_cast<size_t>(nxyz));
    for (int l = 0; l < nxyz; ++l) {
        _sweep_vol[static_cast<size_t>(l)] = _g.vol(l);
        for (int ig = 0; ig < ng; ++ig) {
            _sweep_chif[static_cast<size_t>(ig) * nxyz + l] = _x.chif(ig, l);
            _sweep_xsnf[static_cast<size_t>(ig) * nxyz + l] = _x.xsnf(ig, l);
        }
    }

    // Page-lock every buffer the sweep launcher memcpys, once per instance:
    // pageable async copies stage through the driver ON the launcher's
    // critical path, pinned ones run at bus speed and actually overlap.
    //
    // The registrations are LEASED (HostPinRegistry.h): the nine fixed-address
    // ranges below are owned by Geometry/CMFD/XSSet and released in THEIR
    // destructors, the four vector-backed ones by ~BICGCMFD.
    if (_ls->arena() != nullptr) {
        auto*        ar   = _ls->arena();
        const size_t nd   = static_cast<size_t>(nxyz);
        const size_t nn   = static_cast<size_t>(_g.ngxyz());
        if (!_sweep_pinned) {
            // Raw arrays, allocated once by their owner and never reallocated.
            // Byte counts are the OWNING allocation's full width, which is what
            // the other arms ask for too -- xsrf/xssm are the same ranges the
            // XSSet arms and the nodal arena page-lock, so those three requests
            // deduplicate instead of colliding (plan Sec 6.2).
            ar->pinHost(_diag, static_cast<size_t>(_g.ng2()) * nd * sizeof(double),
                        "cmfd.diag@sweep");
            ar->pinHost(_cc, static_cast<size_t>(ng) * NEWSBT * nd * sizeof(double),
                        "cmfd.cc@sweep");
            ar->pinHost(_src, nn * sizeof(double), "cmfd.src@sweep");
            ar->pinHost(_psi, nd * sizeof(double), "cmfd.psi@sweep");
            ar->pinHost(flux, nn * sizeof(double), "geom.phif@sweep");
            ar->pinHost(_x.xsrfData(), nn * sizeof(double), "xs.xsrf@sweep");
            ar->pinHost(_x.xssmData(), static_cast<size_t>(ng * ng) * nd * sizeof(double),
                        "xs.xssm@sweep");
            ar->pinHost(_dtil, static_cast<size_t>(_g.nsurf()) * ng * sizeof(double),
                        "cmfd.dtil@sweep");
            ar->pinHost(_dhat, static_cast<size_t>(_g.nsurf()) * ng * sizeof(double),
                        "cmfd.dhat@sweep");
            _sweep_pinned = true;
        }
        // Vector-reallocation contract (plan Sec 6.5).  The three _sweep_*
        // vectors are resized at the top of this function and _udiag is sized
        // in the constructor, so on the steady path .data() never moves and
        // these four comparisons are the whole cost.  If a resize ever DID
        // reallocate, a lease left on the old address would be a registration
        // pointing at freed memory: release it, then lease the new address.
        // "Size before pin, never realloc while leased" is the rule; this is
        // what enforces it rather than assuming it.
        auto lease_vector = [ar](const void*& leased, const void* data, size_t bytes,
                                 const char* tag) {
            if (leased == data) return;
            if (leased != nullptr) rasberyUnpinHost(leased);
            ar->pinHost(data, bytes, tag);
            leased = data;
        };
        lease_vector(_pin_udiag, _udiag.data(), _udiag.size() * sizeof(double),
                     "bicg.udiag@sweep");
        lease_vector(_pin_sweep_chif, _sweep_chif.data(), _sweep_chif.size() * sizeof(double),
                     "bicg.sweep_chif@sweep");
        lease_vector(_pin_sweep_xsnf, _sweep_xsnf.data(), _sweep_xsnf.size() * sizeof(double),
                     "bicg.sweep_xsnf@sweep");
        lease_vector(_pin_sweep_vol, _sweep_vol.data(), _sweep_vol.size() * sizeof(double),
                     "bicg.sweep_vol@sweep");
    }

    double reigv  = 1. / eigv;
    double reigvs = (_eshift != 0.0) ? 1. / (eigv + _eshift) : 0.0;
    int    iout = 0, icmfd = 0;

    while (iout < _ncmfd) {
        // Stage this outer's operator exactly as a host sweep would; the
        // device builds src itself, so _src rides along unused.
        double r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);
        _ls->solveInner(_nmaxbicg, _epsbicg);

        CudaBatchArena::CmfdSweepIO io;
        io.chif         = _sweep_chif.data();
        io.xsnf         = _sweep_xsnf.data();
        io.xsrf         = _x.xsrfData();
        io.xssm         = _x.xssmData();
        io.dtil         = _dtil;
        io.dhat         = _dhat;
        io.vol          = _sweep_vol.data();
        io.udiag        = _udiag.data();
        io.psi          = _psi;
        io.device_assembly = use_device_assembly;
        io.eigv         = eigv;
        io.reigv        = reigv;
        io.reigvs       = reigvs;
        io.errl2        = errl2;
        io.epsl2        = _epsl2;
        io.eshift       = _eshift;
        io.sweep_budget = _ncmfd - iout;
        io.icmfd_budget = 20 * _ncmfd;
        io.icmfd_done   = icmfd;
        io.ngxyz        = _g.ngxyz();

        if (!_ls->driveSweepsCuda(flux, io)) {
            // Nothing ran. Rebuild the host operator before allowing the
            // pristine host loop to take over; stale diag/cc is fail-closed.
            if (use_device_assembly) assembleHostLinearSystem(eigv);
            return finish(false);
        }

        const int attempts = io.icmfd_done - icmfd;
        icmfd              = io.icmfd_done;
        iout += io.sweeps_done;
        _wiel_sweep += attempts;
        iter += attempts;
        // Each device sweep runs the same captured inner loop the host path
        // launches: one unconditional iteration plus _nmaxbicg more.
        _bicg_iters += static_cast<long long>(attempts) * (1 + _nmaxbicg);
        eigv   = io.eigv;
        reigv  = io.reigv;
        reigvs = io.reigvs;
        errl2  = io.errl2;

        if (io.state == 1 || io.state == 3) return finish(true); // converged / budget spent

        if (io.state == 2) {
            // Degenerate gamma: the device ran this sweep's source/BiCG/psi
            // update and exported the wiel sums; finish the sweep with the
            // Rayleigh branch of wiel, then keep looping.  psi already holds
            // the NEW fission source, exactly as it does at this point of the
            // host wiel.
            double sumf = 0;
            double summ = 0;
            for (int l = 0; l < nxyz; ++l) {
                for (int ig2 = 0; ig2 < ng; ++ig2) {
                    double ab = CMFD::axb(ig2, l, flux);
                    summ      = summ + ab;
                }
                sumf += psi(l);
                summ += psi(l) * reigvs;
            }
            eigv  = sumf / summ;
            reigv = 1 / eigv;
            const double err_scale = (io.gammad > 0.0) ? io.gammad : io.gamman;
            errl2 = (err_scale > 0.0) ? sqrt(abs(io.err_acc / err_scale)) : 0.0;
            double eigvs = eigv + _eshift; // Wielandt regime: icy >= 0
            reigvs       = (_eshift != 0.0) ? 1. / eigvs : 0.0;

            if (_eshift != 0.0) {
                updls(reigvs);
                _ls->facilu(_diag, _cc);
            }
            int negative = 0;
            for (int l = 0; l < nxyz; ++l)
                for (int ig2 = 0; ig2 < ng; ++ig2)
                    if (flux(ig2, l) < 0) ++negative;
            if (negative == _g.ngxyz()) negative = 0;
            if (!(negative != 0 && icmfd < 20 * _ncmfd)) ++iout;
            if (errl2 < _epsl2) return finish(true);
            continue;
        }
        // state 0: the launch unroll was spent on retries; go again with the
        // remaining budget.
    }
    return finish(true);
}

void BICGCMFD::drive(double& eigv, double* flux, double& errl2) {

    int    icmfd  = 0;
    double reigv  = 1. / eigv;
    double reigvs = 0.0;

    if (_eshift != 0.0) reigvs = 1. / (eigv + _eshift);

    // One-shot capture of the first fully post-warmup drive (every sweep in
    // the Wielandt regime), for the offline form probe.
    const bool cap = g_cmfd_dump.armed && !g_cmfd_dump.done &&
                     (_wiel_sweep >= WIELANDT_WARMUP_SWEEPS);
    if (cap) {
        g_cmfd_dump.openEntry();
        const double hdr[8] = {static_cast<double>(_g.ng()), static_cast<double>(_g.nxyz()),
                               static_cast<double>(_g.ng2()), static_cast<double>(_ncmfd),
                               eigv, reigvs, _eshift, _epsl2};
        g_cmfd_dump.write(hdr, 8);
        g_cmfd_dump.write(_psi, static_cast<size_t>(_g.nxyz()));
        g_cmfd_dump.write(_udiag.data(), static_cast<size_t>(_g.ng2()) * _g.nxyz());
        std::vector<double> chif_mat(static_cast<size_t>(_g.ng()) * _g.nxyz());
        std::vector<double> xsnf_mat(static_cast<size_t>(_g.ng()) * _g.nxyz());
        std::vector<double> vol_mat(static_cast<size_t>(_g.nxyz()));
        for (int l = 0; l < _g.nxyz(); ++l) {
            vol_mat[static_cast<size_t>(l)] = _g.vol(l);
            for (int ig = 0; ig < _g.ng(); ++ig) {
                chif_mat[static_cast<size_t>(ig) * _g.nxyz() + l] = _x.chif(ig, l);
                xsnf_mat[static_cast<size_t>(ig) * _g.nxyz() + l] = _x.xsnf(ig, l);
            }
        }
        g_cmfd_dump.write(chif_mat.data(), chif_mat.size());
        g_cmfd_dump.write(xsnf_mat.data(), xsnf_mat.size());
        g_cmfd_dump.write(vol_mat.data(), vol_mat.size());
    }

    // Device-resident sweeps, once the Wielandt regime is reached (the warm-up
    // and its Rayleigh schedule stay on the host, so the device never needs
    // the icy < 0 branch).  A false return means nothing ran on the device and
    // the pristine host loop below takes over.
    static const bool sweep_dev = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_SWEEP");
        return v != nullptr && std::string(v) != "0";
    }();
    if (sweep_dev && !cap && _ls->usingCuda() && _g.ng() == 2 &&
        _wiel_sweep >= WIELANDT_WARMUP_SWEEPS) {
        if (driveDeviceSweeps(eigv, flux, errl2)) return;
    }

    int negative = 0;
    int iout     = 0;
    for (; iout < _ncmfd; ++iout) {
        ++iter;
        ++_wiel_sweep;
        ++icmfd;
        double reigvdel = reigv - reigvs;

        for (int l = 0; l < _g.nxyz(); ++l) {
            double fs = psi(l) * reigvdel;
            for (int ig = 0; ig < _g.ng(); ++ig) {
                src(ig, l) = _x.chif(ig, l) * fs;
            }
        }

        // r20 is the reference residual norm ||b - A*phi0|| for this outer and must
        // stay fixed for the whole inner loop. The first solve used to pass r20 as
        // BOTH the reference (3rd arg) and the output residual (5th arg), so the
        // reference was overwritten by the first iteration's residual and every
        // subsequent relative test was measured against a moving denominator.
        double r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);

        if (_ls->usingCuda()) {
            // The device runs the identical loop -- one unconditional iteration
            // then up to _nmaxbicg more, each followed by the same relative test
            // against the same frozen r20 -- but without reporting to the host
            // in between, so the whole thing is one CUDA graph launch instead of
            // a per-iteration status copy plus stream drain.
            _ls->solveInner(_nmaxbicg, _epsbicg);
            _bicg_iters += 1 + _nmaxbicg;
        } else {
            double r2 = r20;
            _ls->solve(_diag, _cc, r20, flux, r2);
            ++_bicg_iters;

            for (int iin = 0; iin < _nmaxbicg; ++iin) {
                // solve linear system A*phi = src
                _ls->solve(_diag, _cc, r20, flux, r2);
                ++_bicg_iters;
                PLOG(plog::debug) << iin << "-th Inner Solver Error " << r2;
                // One relative exit, measured against the fixed reference r20.  The companion
                // `if (r2 < 1.E-6 && iin > 2) break;` is gone: since fix4 un-aliased r20 from r2,
                // r2 is the true absolute residual ||b - A*phi||, so a fixed absolute threshold
                // tested the source normalisation (core size / power level) rather than
                // convergence -- it would fire immediately on a small or lightly-normalised
                // problem and never on a large one.  It was also unreachable at the default
                // _nmaxbicg = 3, where iin never exceeds 2.  The degenerate r20 == 0 case (the
                // entry flux already satisfies the linear system exactly) now exits explicitly
                // instead of being caught by that absolute floor.
                if (r20 <= 0.0 || r2 / r20 < _epsbicg) break;
            }
        }

        // Keep the bulk flux resident through all BiCGSTAB inner iterations.
        // The CPU Wielandt/nodal stages observe it once at this boundary.
        _ls->synchronizeCudaFlux(flux);

        if (cap) {
            // Everything the wiel/updls of THIS sweep consumes, before it runs.
            const double pre[4] = {reigv, reigvs, eigv,
                                   static_cast<double>(_wiel_sweep - WIELANDT_WARMUP_SWEEPS)};
            g_cmfd_dump.write(pre, 4);
            g_cmfd_dump.write(flux, static_cast<size_t>(_g.ng()) * _g.nxyz());
            g_cmfd_dump.write(_psi, static_cast<size_t>(_g.nxyz()));
        }

        // wielandt shift
        wiel(_wiel_sweep - WIELANDT_WARMUP_SWEEPS, flux, reigvs, eigv, reigv, errl2);

        if (_eshift != 0.0) {
            updls(reigvs);
            // B1: updls just modified _diag via the Wielandt shift; rebuild the SSOR
            // factorization so the next iteration's solve preconditions on the current
            // diagonal instead of a stale _dinv (fewer BiCGSTAB iterations).
            _ls->facilu(_diag, _cc);
        }

        negative = 0;
        for (int l = 0; l < _g.nxyz(); ++l) {
            for (int ig = 0; ig < _g.ng(); ++ig) {
                if (flux(ig, l) < 0) {
                    ++negative;
                }
            }
        }
        if (negative == _g.ngxyz()) {
            negative = 0;
        }

        if (negative != 0 && icmfd < 20 * _ncmfd) iout--;

        PLOG(plog::debug) << "IOUT : " << iter << ", EIGV : " << eigv << ", ERRL2 : " << errl2 << ", NEGATIVE : " << negative;

        if (cap) {
            // Everything this sweep produced.
            const double post[5] = {eigv, reigv, reigvs, errl2,
                                    static_cast<double>(negative)};
            g_cmfd_dump.write(post, 5);
            g_cmfd_dump.write(_psi, static_cast<size_t>(_g.nxyz()));
            g_cmfd_dump.write(_diag, static_cast<size_t>(_g.ng2()) * _g.nxyz());
        }

        if (errl2 < _epsl2) break;
    }
    if (cap) g_cmfd_dump.close();
}

void BICGCMFD::resetIteration() {
    iter                     = 0;
    _wiel_sweep              = 0;
    _bicg_iters              = 0;
    _device_assembly_pending = false;
}

