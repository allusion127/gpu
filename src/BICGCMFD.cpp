#include "BICGCMFD.h"

#define flux(ig, l)  (flux[(l) * _g.ng() + ig])
#define aflux(ig, l) (aflux[(l) * _g.ng() + ig])

using namespace rasbery;

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
}

void BICGCMFD::setIterLim(int maxls, double epsls) {
    _nmaxbicg = maxls;
    _epsbicg  = epsls;
}

BICGCMFD::~BICGCMFD() = default;

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

void BICGCMFD::setls(const double& eigv) {
    // Zero shift does not need an extra unshifted-diagonal copy/update pass.
    if (_eshift == 0.0) {
        for (int l = 0; l < _g.nxyz(); ++l) {
            setls(l);
        }
        _ls->facilu(_diag, _cc);
        return;
    }

    double reigvs = 0.0;
    if (_eshift != 0.0) reigvs = 1. / (eigv + _eshift);

    for (int l = 0; l < _g.nxyz(); ++l) {
        setls(l);
        updls(l, reigvs);
    }
    _ls->facilu(_diag, _cc);
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

void BICGCMFD::drive(double& eigv, double* flux, double& errl2) {

    int    icmfd  = 0;
    double reigv  = 1. / eigv;
    double reigvs = 0.0;

    if (_eshift != 0.0) reigvs = 1. / (eigv + _eshift);

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
        } else {
            double r2 = r20;
            _ls->solve(_diag, _cc, r20, flux, r2);

            for (int iin = 0; iin < _nmaxbicg; ++iin) {
                // solve linear system A*phi = src
                _ls->solve(_diag, _cc, r20, flux, r2);
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

        if (errl2 < _epsl2) break;
    }
}

void BICGCMFD::resetIteration() {
    iter        = 0;
    _wiel_sweep = 0;
}

