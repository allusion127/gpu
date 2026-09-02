#include "BICGCMFD.h"

#include "GpuFullContract.h"
#include "HostOuterBodyCounters.h"
#include "HostPinRegistry.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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
    // The only writer of _dtil.  The device outer segment reads the device twin
    // every outer; the generation is what lets it skip the copy on the outers
    // where nothing moved.
    ++_dtil_generation;
    // Rev.7.1 Task 9: one relaxed increment per SWEEP, so a receipt can say
    // whether the device outer actually replaced this loop (HostOuterBodyCounters.h).
    hostouter::bumpHostBody(hostouter::counters().upddtil);
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddtil(ls);
    }
}

void BICGCMFD::upddhat(const double* flux, double* jnet) {
    // Rev.7.1 Task 9: one relaxed increment per SWEEP, so a receipt can say
    // whether the device outer actually replaced this loop (HostOuterBodyCounters.h).
    hostouter::bumpHostBody(hostouter::counters().upddhat);
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::upddhat(ls, flux, jnet);
    }
}

bool BICGCMFD::deviceSweepResident() const {
    // canUseDeviceAssembly() is NOT reused here, and the difference is the whole
    // point: it carries `_wiel_sweep >= WIELANDT_WARMUP_SWEEPS`, which is a
    // PER-DRIVE state that is false at the top of every SolveLoop entry.  Arming
    // the segment on it would arm nothing, ever.  What the segment needs to know
    // is whether this run is CONFIGURED for the resident sweep at all; whether
    // any individual drive takes it is drive()'s business, and the segment stays
    // correct either way because it mirrors dhat and psi back to the host.
    return envFlagEnabled("RASBERY_GPU_CMFD_SWEEP") &&
           envFlagEnabledDefaultOn("RASBERY_GPU_CMFD_ASSEMBLY") &&
           std::getenv("RASBERY_CMFD_DUMP") == nullptr && _g.ng() == 2 &&
           _ls != nullptr && _ls->usingCuda() && _ls->arena() != nullptr &&
           _ls->batchSlot() >= 0;
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

void BICGCMFD::updjnet(const double* flux, double* jnet) {
    // Rev.7.1 Task 9: one relaxed increment per SWEEP, so a receipt can say
    // whether the device outer actually replaced this loop (HostOuterBodyCounters.h).
    hostouter::bumpHostBody(hostouter::counters().updjnet);
    for (int ls = 0; ls < _g.nsurf(); ++ls) {
        CMFD::updjnet(ls, flux, jnet);
    }
}

void BICGCMFD::updpsi(const double* flux) {
    // Rev.7.1 Task 9: one relaxed increment per SWEEP, so a receipt can say
    // whether the device outer actually replaced this loop (HostOuterBodyCounters.h).
    hostouter::bumpHostBody(hostouter::counters().updpsi);
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
//
// Rev.7.1 Task 10 part 2 SPLIT IT IN THREE, and the split is what keeps the
// stream-ordered arm honest.  A device outer segment needs the SAME drive
// enqueued instead of driven, and the obvious way to get that -- a second copy
// of the launch sequence with the drain taken out -- would be two spellings of
// one physics loop, free to disagree about the Rayleigh hand-back or the retry
// packing.  So the pieces are named and shared: prepareDeviceSweeps is the
// prologue (aliasing and page-locking), stageSweepIO fills one launch's IO
// block, absorbSweepLaunch is the whole post-launch decision including the
// Rayleigh branch, and BOTH arms are those three in a loop.
bool BICGCMFD::prepareDeviceSweeps(double* flux, SweepPrep& p) {
    // Rev.7.1 Task 10: the sweep signals describe THIS drive, so clear them here.
    //
    // HERE BECAUSE THIS IS THE ONE PROLOGUE BOTH ARMS SHARE -- driveDeviceSweeps
    // and enqueueDrive each call it before anything else -- so neither can begin
    // a drive holding an older drive's verdict.
    //
    // The latches are only written when a device sweep actually ran, and drive()
    // takes the host loop for the whole Wielandt warm-up and whenever this path
    // declines.  A reader arriving in either window used to see the signals of
    // some earlier drive, and a stale 'negative flux' is worse than no signal:
    // it names a condition that has already been resolved.  That is the escape
    // kngr_238 raised on its very first device outer.
    _last_sweep_negative = 0;
    _last_sweep_state    = 0;

    const int nxyz = _g.nxyz();
    const int ng   = _g.ng();
    // chif / xsnf / vol are ALIASED, not copied.
    //
    // The staging loop that used to live here rebuilt three vectors on every
    // drive -- 2 x ng x nxyz strided reads plus nxyz -- and every element it
    // wrote was already sitting contiguously in the layout the device wants:
    //
    //   XSSet::xsnf(ig, l) == _xs.xsnf[ig * nxyz + l]  == xsnfData()[ig*nxyz+l]
    //   XSSet::chif(ig, l) == _ref_chix[ig * nxyz + l] == chifData()[ig*nxyz+l]
    //   Geometry::vol(l)   == _vol[l]
    //
    // so the copy was a pure identity and the mirror memcmp that followed it
    // was comparing a fresh copy of an array against a shadow of the same
    // array.  Pointing at the originals is byte-identical by construction and
    // costs nothing, and it needs no generation counter to stay honest.
    //
    // (A generation guard was the other candidate and is NOT usable here:
    // XSSet::hoststateGeneration() means "the device mirror of _xs is stale",
    // not "the host bytes changed".  XSSet.cpp deliberately does NOT bump it
    // when the GPU XS arm downloads a freshly reconstructed _xs into host
    // memory -- exactly the case a rebuild guard has to catch.)
    const double* sweep_xsnf = _x.xsnfData();
    const double* sweep_chif = _x.chifData();
    const double* sweep_vol  = (nxyz > 0) ? &_g.vol(0) : nullptr;
    if (sweep_chif == nullptr) {
        // No burnup-interpolated fission spectrum: chif(ig,l) is the constant
        // (1, 0, ...) of the accessor, so stage it once and keep it.
        const size_t want = static_cast<size_t>(ng) * nxyz;
        if (_sweep_chif.size() != want) {
            _sweep_chif.assign(want, 0.0);
            for (int l = 0; l < nxyz; ++l) _sweep_chif[static_cast<size_t>(l)] = 1.0;
        }
        sweep_chif = _sweep_chif.data();
    }
    if (sweep_xsnf == nullptr || sweep_vol == nullptr) return false;

    // Page-lock every buffer the sweep launcher memcpys, once per instance:
    // pageable async copies stage through the driver ON the launcher's
    // critical path, pinned ones run at bus speed and actually overlap.
    //
    // The registrations are LEASED (HostPinRegistry.h): the nine fixed-address
    // ranges below are owned by Geometry/CMFD/XSSet and released in THEIR
    // destructors, the four vector-backed ones by ~BICGCMFD.
    //
    // AND THEY ARE ARBITRATED (Rev.7.1 Task 18d).  This block is the FIRST
    // TOUCH: it runs once per instance on that instance's own Driver thread, so
    // in `--batch-mode M` the late decks reach it while the early decks are
    // already capturing the shared arena's CMFD graph.  Measured, this block is
    // the ACCOMPLICE rather than the trigger -- 73 of these registrations
    // landed inside a capture window without killing it, while a single
    // cudaDeviceSynchronize from CudaOuterSegment::bindResidency killed it
    // every time -- but it is what puts a late deck's thread on the device API
    // at exactly that moment, and cudaHostRegister is on CUDA's unsafe list too.
    // Nothing here changed; the serialisation lives one level down, in the pin
    // hook that every one of these calls lands in (CudaBICGBackend.cu /
    // CudaXsReconBackend.cu, rasbery::AllocWindow).  See src/GpuCaptureArbiter.h.
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
            // xsnf / vol joined this list when the staging copies went away:
            // they are now the OWNERS' arrays (XSSet's _xs.xsnf, Geometry's
            // _vol), released by ~XSSet / ~Geometry like xsrf and xssm above.
            // A lease taken here and released in ~BICGCMFD would drop the only
            // owner record for a base the owner still uses -- the registry
            // deduplicates a repeat request for the SAME base instead of
            // counting a second owner -- so these must NOT be lease_vector'd.
            ar->pinHost(sweep_xsnf, static_cast<size_t>(ng) * nd * sizeof(double),
                        "xs.xsnf@sweep");
            ar->pinHost(sweep_vol, nd * sizeof(double), "geom.vol@sweep");
            if (sweep_chif != _sweep_chif.data())
                ar->pinHost(sweep_chif, static_cast<size_t>(ng) * nd * sizeof(double),
                            "xs.chif@sweep");
            _sweep_pinned = true;
        }
        // Reallocation contract (plan Sec 6.5).  _udiag is sized in the
        // constructor, so on the steady path .data() never moves and this
        // comparison is the whole cost.  If a resize ever DID reallocate, a
        // lease left on the old address would be a registration pointing at
        // freed memory: release it, then lease the new address.  "Size before
        // pin, never realloc while leased" is the rule; this enforces it
        // rather than assuming it.
        auto lease_vector = [ar](const void*& leased, const void* data, size_t bytes,
                                 const char* tag) {
            if (leased == data) return;
            if (leased != nullptr) rasberyUnpinHost(leased);
            ar->pinHost(data, bytes, tag);
            leased = data;
        };
        lease_vector(_pin_udiag, _udiag.data(), _udiag.size() * sizeof(double),
                     "bicg.udiag@sweep");
        // chif is XSSet-owned too (pinned once above) UNLESS the deck has no
        // fission spectrum, in which case the constant fallback is OURS and is
        // the one chif range ~BICGCMFD still has to release.
        if (sweep_chif == _sweep_chif.data())
            lease_vector(_pin_sweep_chif, sweep_chif,
                         _sweep_chif.size() * sizeof(double), "bicg.sweep_chif@sweep");
    }

    p.xsnf = sweep_xsnf;
    p.chif = sweep_chif;
    p.vol  = sweep_vol;
    p.ok   = true;
    return true;
}

/// Fill one launch's IO block exactly as the host sweep loop always has.
void BICGCMFD::stageSweepIO(CudaBatchArena::CmfdSweepIO& io, const SweepPrep& p,
                            double eigv, double reigv, double reigvs, double errl2,
                            int iout, int icmfd, bool psi_dirty) {
        io.chif         = p.chif;
        io.xsnf         = p.xsnf;
        io.xsrf         = _x.xsrfData();
        io.xssm         = _x.xssmData();
        io.dtil         = _dtil;
        io.dhat         = _dhat;
        io.vol          = p.vol;
        io.udiag        = _udiag.data();
        io.psi          = _psi;
        io.psi_dirty    = psi_dirty;
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
        // Rev.7.1 Task 9 link 2: when the device outer segment owns these, the
        // host arrays are one outer behind by construction and the upload would
        // copy them over the bytes the segment just produced.
        io.dhat_device_resident = _outer_segment_resident;
        io.psi_device_resident  = _outer_segment_resident;
}

/// Absorb one launch's outputs and say whether the DRIVE is over.
///
/// True = this drive is finished (converged, or the sweep budget is spent, or
/// the Rayleigh hand-back converged).  False = the caller must launch again:
/// state 0 means the launch's slot budget ran out mid-retry, state 2 means the
/// gamma degenerated and the branch below has just finished that sweep on the
/// host, exactly as the host loop always did.
bool BICGCMFD::absorbSweepLaunch(CudaBatchArena::CmfdSweepIO& io, double& eigv,
                                 double* flux, double& errl2, double& reigv,
                                 double& reigvs, int& iout, int& icmfd) {
    const int nxyz = _g.nxyz();
    const int ng   = _g.ng();
        const int attempts = io.icmfd_done - icmfd;
        icmfd              = io.icmfd_done;
        iout += io.sweeps_done;
        // A2 S0 Sec 1.6: the sweeps nothing charges for.  `attempts` is every
        // sweep this launch ran and `sweeps_done` is the subset that consumed
        // the drive's budget, so the difference is the negative-flux retry tail
        // -- real launches that appear in no outer-count receipt.
        //
        // GUARDED ON THE SAME WINDOW `attempts` MEASURES.  finishDeferredDrives
        // charges a whole segment's attempts from the device accumulator and
        // then re-enters here for the ABANDONED launch with `icmfd` seeded to
        // that launch's own icmfd_done, so `attempts` is deliberately zero
        // there and its sweeps are already in the accumulator's total.  Without
        // the guard that call would subtract them twice.
        if (attempts > 0) {
            _drive_exits.sweeps_charged        += io.sweeps_done;
            _drive_exits.negative_retry_sweeps += attempts - io.sweeps_done;
        }
        _wiel_sweep += attempts;
        iter += attempts;
        // Each device sweep runs the same captured inner loop the host path
        // launches: one unconditional iteration plus _nmaxbicg more.
        _bicg_iters += static_cast<long long>(attempts) * (1 + _nmaxbicg);
        eigv   = io.eigv;
        reigv  = io.reigv;
        reigvs = io.reigvs;
        errl2  = io.errl2;
        // What the segment's transition has to rank, recorded where the sweep
        // status is known rather than re-derived from a later observation.
        _last_sweep_negative = io.negative_last;
        _last_sweep_state    = io.state;

        if (io.state == 1 || io.state == 3) {
            // issueFluxDownloads wrote Geometry::Phif FROM the device phi, so
            // the two agree byte for byte.  Only these two states qualify:
            // state 0 and state 2 hand back to the host loop, which then moves
            // the host flux without the device seeing it.
            _last_drive_device_flux = true;
            return true; // converged / budget spent
        }

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
            if (errl2 < _epsl2) return true;
            return false;
        }
    // state 0: the launch unroll was spent on retries; the caller goes again
    // with the remaining budget.
    return false;
}

bool BICGCMFD::driveDeviceSweeps(double& eigv, double* flux, double& errl2) {
    const bool use_device_assembly = _device_assembly_pending;
    _device_assembly_pending       = false;

    SweepPrep p;
    if (!prepareDeviceSweeps(flux, p)) return false;

    double reigv  = 1. / eigv;
    double reigvs = (_eshift != 0.0) ? 1. / (eigv + _eshift) : 0.0;
    int    iout = 0, icmfd = 0;

    // Only the FIRST launch of this drive carries host-written psi.  The host
    // regenerates _psi wholesale (CMFD::updpsi from the flux) before every
    // BICGCMFD::drive and never writes it between two launches of one drive,
    // so a later launch would be handing the device back exactly the bytes it
    // produced.  See CmfdSweepIO::psi_dirty.
    bool psi_dirty = true;

    while (iout < _ncmfd) {
        // Stage this outer's operator exactly as a host sweep would; the
        // device builds src itself, so _src rides along unused.
        double r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);
        _ls->solveInner(_nmaxbicg, _epsbicg);

        CudaBatchArena::CmfdSweepIO io;
        stageSweepIO(io, p, eigv, reigv, reigvs, errl2, iout, icmfd, psi_dirty);
        io.device_assembly = use_device_assembly;

        if (!_ls->driveSweepsCuda(flux, io)) {
            // Nothing ran. Rebuild the host operator before allowing the
            // pristine host loop to take over; stale diag/cc is fail-closed.
            if (use_device_assembly) assembleHostLinearSystem(eigv);
            ++_drive_exits.aborted;
            return false;
        }
        // The device now owns psi: it advanced it, and nothing on the host
        // touches it until the next drive's updpsi.
        psi_dirty = false;
        if (absorbSweepLaunch(io, eigv, flux, errl2, reigv, reigvs, iout, icmfd)) {
            chargeDriveExit(errl2);
            return true;
        }
    }
    chargeDriveExit(errl2);
    return true;
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 10 part 2: the drive as a stream-ordered enqueue
// ---------------------------------------------------------------------------

BICGCMFD::EnqueueRefusal BICGCMFD::enqueueRefusal() const {
    // The SAME gate drive() applies, minus the one-shot form-probe capture,
    // which is a property of the call and not of the run.  Spelled once, here,
    // so the segment cannot arm an arm drive() would not have taken.
    //
    // WP1 follow-up: it returns the REASON rather than a boolean, because the
    // outer segment's fail-closed guard has to tell the Rayleigh warm-up (host
    // by design -- see drive() and the WIELANDT_WARMUP_SWEEPS note) apart from
    // an arm that was asked for and is simply not there.  canEnqueueDrive() is
    // `== None`, so there is still exactly one predicate.
    static const bool sweep_dev = [] {
        const char* v = std::getenv("RASBERY_GPU_CMFD_SWEEP");
        return v != nullptr && std::string(v) != "0";
    }();
    if (!sweep_dev) return EnqueueRefusal::SweepArmOff;
    if (_ls == nullptr || !_ls->usingCuda()) return EnqueueRefusal::NoCudaSolver;
    if (_g.ng() != 2) return EnqueueRefusal::NotTwoGroup;
    if (_wiel_sweep < WIELANDT_WARMUP_SWEEPS) return EnqueueRefusal::WielandtWarmup;
    return EnqueueRefusal::None;
}

bool BICGCMFD::enqueueDrive(double& eigv, double* flux, double& errl2,
                            const CudaBatchArena::CmfdSweepProbeSink& probe,
                            void* caller_stream) {
    _enqueued = EnqueuedDrive{};
    // ASKED BEFORE ANYTHING IS CONSUMED.  A refusal here must leave the solver
    // exactly as setls() left it, because the caller's fallback is the blocking
    // drive() -- which reads `_device_assembly_pending` to decide whether the
    // operator is the arena's or the host's.
    //
    // WP1 follow-up: both refusal points RECORD why, so the caller that already
    // paid for this call reads the reason that actually decided instead of
    // re-asking a ladder whose per-drive state may have moved.
    _enqueue_refusal = enqueueRefusal();
    if (_enqueue_refusal != EnqueueRefusal::None) return false;

    const bool use_device_assembly = _device_assembly_pending;
    _device_assembly_pending       = false;

    SweepPrep p;
    if (!prepareDeviceSweeps(flux, p)) {
        // NOT a designed host region: the gate said yes and the staging refused.
        _enqueue_refusal = EnqueueRefusal::StagePrepFailed;
        return false;
    }

    const double reigv  = 1. / eigv;
    const double reigvs = (_eshift != 0.0) ? 1. / (eigv + _eshift) : 0.0;

    // Exactly the first pass of driveDeviceSweeps' loop: the operator stage, the
    // inner budget, the IO block at iout = icmfd = 0 with psi_dirty set.
    double r20 = 0.0;
    _ls->reset(_diag, _cc, flux, _src, r20);
    _ls->solveInner(_nmaxbicg, _epsbicg);

    CudaBatchArena::CmfdSweepIO io;
    stageSweepIO(io, p, eigv, reigv, reigvs, errl2, 0, 0, /*psi_dirty=*/true);
    io.device_assembly = use_device_assembly;

    if (!_ls->enqueueSweepsCuda(flux, io, probe, caller_stream)) {
        // Nothing was enqueued.  Rebuild the host operator before the caller
        // falls back, for the same fail-closed reason driveDeviceSweeps does.
        if (use_device_assembly) assembleHostLinearSystem(eigv);
        return false;
    }

    _enqueued.active              = true;
    _enqueued.use_device_assembly = use_device_assembly;
    _enqueued.prep                = p;
    _enqueued.io                  = io;
    _enqueued.flux                = flux;
    _enqueued.reigv               = reigv;
    _enqueued.reigvs              = reigvs;
    return true;
}

bool BICGCMFD::finishDrive(double& eigv, double* flux, double& errl2,
                           bool& host_continued) {
    host_continued = false;
    if (!_enqueued.active) return false;
    EnqueuedDrive d = _enqueued;
    _enqueued       = EnqueuedDrive{};

    if (!_ls->finishSweepsCuda(d.io))
        throw std::runtime_error("CUDA BiCGSTAB detected a non-finite value");

    int iout = 0, icmfd = 0;
    if (absorbSweepLaunch(d.io, eigv, flux, errl2, d.reigv, d.reigvs, iout, icmfd)) {
        chargeDriveExit(errl2);
        return true;
    }

    // THE DRIVE IS NOT OVER, and this is the one place the host still has to
    // spin: sweep state 0 (the launch's slot budget was spent on negative-flux
    // retries) and state 2 (the Wielandt gamma degenerated, and absorbSweepLaunch
    // has just finished that sweep on the Rayleigh branch).  Both are exceptional
    // -- 0 of 1690 outers on i-SMR CY01 -- and both are the host loop verbatim,
    // because it is the same three calls in the same order.
    //
    // The caller is told so it can republish the segment's probe: the device
    // verdict kernel published the eigenvalue of a HALF drive and latched the
    // segment's halt on it, and what follows here moves both.
    host_continued = true;
    bool psi_dirty = false;
    while (iout < _ncmfd) {
        double r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);
        _ls->solveInner(_nmaxbicg, _epsbicg);

        CudaBatchArena::CmfdSweepIO io;
        stageSweepIO(io, d.prep, eigv, d.reigv, d.reigvs, errl2, iout, icmfd, psi_dirty);
        io.device_assembly = d.use_device_assembly;

        if (!_ls->driveSweepsCuda(flux, io)) {
            if (d.use_device_assembly) assembleHostLinearSystem(eigv);
            ++_drive_exits.aborted;
            return false;
        }
        psi_dirty = false;
        if (absorbSweepLaunch(io, eigv, flux, errl2, d.reigv, d.reigvs, iout, icmfd)) {
            chargeDriveExit(errl2);
            return true;
        }
    }
    chargeDriveExit(errl2);
    return true;
}


// ---------------------------------------------------------------------------
// Rev.7.1 Task 10 part 3: the observation, once per SEGMENT
// ---------------------------------------------------------------------------

bool BICGCMFD::finishDeferredDrives(
    const CudaBatchArena::CmfdSweepProbeSink::Accum& acc, double& eigv, double* flux,
    double& errl2, bool& host_continued) {
    host_continued = false;
    if (!_enqueued.active) return false;
    EnqueuedDrive d = _enqueued;
    _enqueued       = EnqueuedDrive{};

    // THE ABANDONED LAUNCH, IF THERE WAS ONE.  `saved` is that launch's whole
    // scalar block, copied by the verdict kernel at the moment it raised the
    // segment's halt -- the only moment it was still readable, because every
    // outer enqueued behind it uploads its own staged block over the same
    // sixteen doubles before the host is ever allowed to look.
    const bool                     exceptional = acc.exceptional != 0.0;
    CudaBatchArena::CmfdSweepIO    saved{};
    if (exceptional) CudaBatchArena::unpackSavedSweepBlock(acc, saved);

    // The arena's end-of-launch bookkeeping, once.  `state` decides only whether
    // the Rayleigh hand-back's operator downloads run, so it is the ABANDONED
    // launch's state that matters and not the last one's.
    if (!_ls->finishSweepsDeferredCuda(exceptional ? saved.state
                                                   : static_cast<int>(acc.state)))
        throw std::runtime_error("CUDA BiCGSTAB detected a non-finite value");

    // ---- the counters absorbSweepLaunch would have advanced, summed on the
    // ---- device instead of once per outer on the host.
    //
    // ONE SUM, NOT N SUMS, AND THE SAME TOTAL.  Every launch's contribution is
    // `io.icmfd_done - icmfd`, and `icmfd` is 0 at the top of each enqueued
    // drive (stageSweepIO is called with icmfd = 0), so each contribution is
    // just that launch's kIcmfdDone -- which is exactly what the verdict kernel
    // added.  A launch enqueued behind a halt that had already fired ran
    // nothing and contributed nothing, because the verdict returns first.
    const int attempts = static_cast<int>(acc.attempts);
    _wiel_sweep += attempts;
    iter += attempts;
    _bicg_iters += static_cast<long long>(attempts) * (1 + _nmaxbicg);

    // A2 S0.  THE SEGMENT'S DRIVES, AND WHY THEIR EXITS ARE NOT SPLIT.  The
    // verdict kernel sums a whole segment into one Accum and keeps only the
    // LAST launch's `state`, so the exit of every earlier drive is gone before
    // the host sees anything.  Recovering it means a per-state histogram in the
    // accumulator, which is a .cu change this instrumentation deliberately does
    // not make.  So they are reported as their own bucket rather than folded
    // into `converged`/`budget`, and `deferred_sweeps / deferred_drives` is the
    // proxy that answers S3's K0 gate anyway: a ratio sitting at `_ncmfd` is a
    // segment whose drives all spent the budget.
    //
    // The exceptional launch is EXCLUDED from the count because the host tail
    // below finishes that drive and charges its exit for real; its partial
    // sweeps are excluded from `deferred_sweeps` with it, so the ratio is over
    // whole drives.  Its ATTEMPTS stay in the totals above, which is why
    // absorbSweepLaunch is re-entered for it with attempts deliberately zero.
    const long long acc_sweeps   = static_cast<long long>(acc.sweeps);
    const long long saved_sweeps = exceptional ? static_cast<long long>(saved.sweeps_done) : 0;
    _drive_exits.sweeps_charged        += acc_sweeps;
    _drive_exits.negative_retry_sweeps += static_cast<long long>(attempts) - acc_sweeps;
    _drive_exits.deferred_drives += static_cast<long long>(acc.launches) - (exceptional ? 1 : 0);
    _drive_exits.deferred_sweeps += acc_sweeps - saved_sweeps;

    _last_sweep_negative = (acc.negative != 0.0) ? 1 : 0;
    _last_sweep_state    = static_cast<int>(acc.state);

    if (!exceptional) {
        // States 1 and 3 only.  issueFluxDownloads wrote Geometry::Phif FROM
        // the device phi at the end of every launch and the host never touched
        // it in between, so the two agree byte for byte -- which is
        // absorbSweepLaunch's own condition for this flag, reached by the same
        // reasoning on a whole segment instead of one launch.
        _last_drive_device_flux = true;
        return true;
    }

    // ---- the drive the device could not finish, finished verbatim ----------
    //
    // FROM HERE THIS IS BICGCMFD::finishDrive'S TAIL, ON THE SAVED BLOCK.  The
    // only difference is where the block came from, and the two counter
    // seedings that follow from it: `icmfd` starts at the abandoned launch's
    // own kIcmfdDone so absorbSweepLaunch's `attempts` for it is zero -- the
    // accumulator has already charged those attempts -- while `iout` starts at
    // zero so the drive's sweep count is built up exactly as the host's is.
    int    iout   = 0;
    int    icmfd  = saved.icmfd_done;
    double reigv  = d.reigv;
    double reigvs = d.reigvs;
    _last_drive_device_flux = false;
    if (absorbSweepLaunch(saved, eigv, flux, errl2, reigv, reigvs, iout, icmfd)) {
        chargeDriveExit(errl2);
        return true;
    }

    host_continued = true;
    bool psi_dirty = false;
    while (iout < _ncmfd) {
        double r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);
        _ls->solveInner(_nmaxbicg, _epsbicg);

        CudaBatchArena::CmfdSweepIO io;
        stageSweepIO(io, d.prep, eigv, reigv, reigvs, errl2, iout, icmfd, psi_dirty);
        io.device_assembly = d.use_device_assembly;

        if (!_ls->driveSweepsCuda(flux, io)) {
            if (d.use_device_assembly) assembleHostLinearSystem(eigv);
            ++_drive_exits.aborted;
            return false;
        }
        psi_dirty = false;
        if (absorbSweepLaunch(io, eigv, flux, errl2, reigv, reigvs, iout, icmfd)) {
            chargeDriveExit(errl2);
            return true;
        }
    }
    chargeDriveExit(errl2);
    return true;
}

void BICGCMFD::drive(double& eigv, double* flux, double& errl2) {
    // Assume the HOST will own the flux.  Every path out of this function
    // leaves that true except the device-sweep states that download it, and
    // absorbSweepLaunch says so for itself.
    _last_drive_device_flux = false;

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
    if (!cap && canEnqueueDrive()) {
        if (driveDeviceSweeps(eigv, flux, errl2)) return;
        // WP1 (plan Sec 6.3).  canEnqueueDrive() was true, so the device arm
        // was armed and ASKED to run; a false from driveDeviceSweeps is a
        // decline, and the loop below is the CPU.  BackendCounters has carried
        // a `cmfd_cpu_fallbacks` field since the backend was written and
        // nothing ever incremented it -- this is the seam it was for.
        RASBERY_GPU_FULL_GUARD(Cmfd, "BICGCMFD::drive",
                               "the device sweep loop declined; the host BiCGSTAB "
                               "outer loop takes the drive");
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

        if (negative != 0 && icmfd < 20 * _ncmfd) {
            iout--;
            // A2 S0 Sec 1.6.  This sweep ran and is not charged against the
            // budget; `icmfd - sweeps_charged` on the device side is the same
            // quantity, counted the same way.
            ++_drive_exits.negative_retry_sweeps;
        } else {
            ++_drive_exits.sweeps_charged;
        }

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
    // A2 S0.  The host loop's only break IS `errl2 < _epsl2`, so re-reading it
    // here separates a drive that stopped on tolerance from one that ran out of
    // budget without a second flag to keep in step with the break.
    chargeDriveExit(errl2);
    if (cap) g_cmfd_dump.close();
}

void BICGCMFD::resetIteration() {
    iter                     = 0;
    _wiel_sweep              = 0;
    _bicg_iters              = 0;
    _drive_exits             = DriveExits{};
    _device_assembly_pending = false;
}

void BICGCMFD::chargeDriveExit(double errl2) {
    ++_drive_exits.drives;
    if (errl2 < _epsl2)
        ++_drive_exits.converged;
    else
        ++_drive_exits.budget;
}
