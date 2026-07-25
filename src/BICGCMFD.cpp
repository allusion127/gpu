#include "BICGCMFD.h"

#define flux(ig, l)  (flux[(l) * _g.ng() + ig])
#define aflux(ig, l) (aflux[(l) * _g.ng() + ig])

using namespace rasbery;

BICGCMFD::BICGCMFD(Geometry& g, XSSet& x)
    : CMFD(g, x),
      _ls(std::make_unique<BICGSolver>(_g)),
      _nodal(nullptr),
      _udiag(static_cast<size_t>(_g.ng2()) * static_cast<size_t>(_g.nxyz())) {
    _epsbicg  = 0.1;
    _nmaxbicg = 3;

    _eshift = 0.04;
    iter    = 0;
}

void BICGCMFD::setIterLim(int maxls, float epsls) {
    _nmaxbicg = maxls;
    _epsbicg  = epsls;
}

BICGCMFD::~BICGCMFD() = default;

void BICGCMFD::setEshift(float eshift) {
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
    if (icy < 0 || gammad < 0 || gamman < 0) {
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

    errl2 = sqrt(abs(errl2 / gammad));

    double eigvs = eigv;
    reigvs       = 0;

    if (icy >= 0) {
        eigvs += _eshift;
        if (_eshift != 0.0) reigvs = 1. / eigvs;
    }
}

double BICGCMFD::residual(const double& reigv, const double& reigvs, const double* flux) {

    double reigvdel = reigv - reigvs;

    double r    = 0.0;
    double psi2 = 0.0;

    for (int l = 0; l < _g.nxyz(); ++l) {
        double fs = psi(l) * reigvdel;

        for (int ig = 0; ig < _g.ng(); ++ig) {
            double ab = CMFD::axb(ig, l, flux);

            double err = _x.chif(ig, l) * fs - ab;
            r += err * err;

            double ps = _x.chif(ig, l) * psi(l);
            psi2 += ps * ps;
        }
    }

    return sqrt(r / psi2);
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

void BICGCMFD::axb(double* flux, double* aflux) {

    for (int l = 0; l < _g.nxyz(); ++l) {
        for (int ig = 0; ig < _g.ng(); ++ig) {
            aflux(ig, l) = CMFD::axb(ig, l, flux);
        }
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
        ++icmfd;
        double reigvdel = reigv - reigvs;

        for (int l = 0; l < _g.nxyz(); ++l) {
            double fs = psi(l) * reigvdel;
            for (int ig = 0; ig < _g.ng(); ++ig) {
                src(ig, l) = _x.chif(ig, l) * fs;
            }
        }

        double r20 = 0.0;
        _ls->reset(_diag, _cc, flux, _src, r20);
        _ls->solve(_diag, _cc, r20, flux, r20);

        double r2 = 0.0;
        for (int iin = 0; iin < _nmaxbicg; ++iin) {
            // solve linear system A*phi = src
            _ls->solve(_diag, _cc, r20, flux, r2);
            PLOG(plog::debug) << iin << "-th Inner Solver Error " << r2;
            if (r2 / r20 < _epsbicg) break;
            if (r2 < 1.E-6 && iin > 2) break;
        }

        // wielandt shift
        wiel(iter - 5, flux, reigvs, eigv, reigv, errl2);

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
    iter = 0;
}

void BICGCMFD::updnodal(double& eigv, double* flux, double* jnet, double* phis) {
    if (_nodal == nullptr)
        _nodal = std::make_unique<Nodal>(_g, _x);

    updjnet(flux, jnet);
    _nodal->reset(1. / eigv, jnet, flux, phis);
    _nodal->drive();
    upddhat(flux, jnet);
}
