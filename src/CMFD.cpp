#include "CMFD.h"

#define flux(ig, l)  (flux[(l) * _g.ng() + ig])
#define jnet(ig, ls) (jnet[(ls) * _g.ng() + ig])

using namespace rasbery;

CMFD::CMFD(Geometry& g, XSSet& x)
    : _g(g), _x(x) {
    _epsl2 = 1.E-5;
    _dtil  = new double[_g.nsurf() * _g.ng()]{};
    _dhat  = new double[_g.nsurf() * _g.ng()]{};
    _diag  = new double[_g.nxyz() * _g.ng2()]{};
    _cc    = new double[_g.nxyz() * _g.ng() * NEWSBT]{};
    _src   = new double[_g.nxyz() * _g.ng()]{};
    _psi   = new double[_g.nxyz()]{};

    if (const char* c = std::getenv("RASBERY_DHAT_CLAMP"))
        _dhat_clamp_enabled = (std::atoi(c) != 0);
}

CMFD::~CMFD() {
    delete[] _dtil;
    delete[] _dhat;
    delete[] _diag;
    delete[] _cc;
    delete[] _src;
    delete[] _psi;
}

void CMFD::resetDhat() {
    std::fill(_dhat, _dhat + static_cast<size_t>(_g.nsurf()) * static_cast<size_t>(_g.ng()), 0.0);
}

void CMFD::upddtil(const int& ls) {
    int ll    = _g.lklr(LEFT, ls);
    int lr    = _g.lklr(RIGHT, ls);
    int idirl = _g.idirlr(LEFT, ls);
    int idirr = _g.idirlr(RIGHT, ls);

    double betal, betar;

    for (int ig = 0; ig < _g.ng(); ig++) {
        if (ll < 0) {
            betal = _g.albedo(LEFT, idirl) * 0.5;
        } else {
            betal = _x.xsdf(ig, ll) / _g.hmesh(idirl, ll);
        }
        if (lr < 0) {
            betar = _g.albedo(RIGHT, idirr) * 0.5;
        } else {
            betar = _x.xsdf(ig, lr) / _g.hmesh(idirr, lr);
        }
        dtil(ig, ls) = 2 * betal * betar / (betal + betar);
    }
}

void CMFD::upddhat(const int& ls, double* flux, double* jnet) {
    int ll = _g.lklr(LEFT, ls);
    int lr = _g.lklr(RIGHT, ls);

    for (int ig = 0; ig < _g.ng(); ig++) {
        double fdiff, fsum;
        if (ll < 0) {
            fdiff = flux(ig, lr);
            fsum  = flux(ig, lr);
        } else if (lr < 0) {
            fdiff = -flux(ig, ll);
            fsum  = flux(ig, ll);
        } else {
            fdiff = flux(ig, lr) - flux(ig, ll);
            fsum  = flux(ig, lr) + flux(ig, ll);
        }
        double jnet_fdm = -dtil(ig, ls) * (fdiff);

        // Guard 1: fsum is a sum of (nominally positive) fluxes, but during the
        // early outers it can collapse toward zero or go negative, which turns
        // the division into a spike that poisons the coupling coefficients.
        // Scale the floor to dtil so the test is dimensionally consistent.
        ++_dhat_total;
        const double dtl   = dtil(ig, ls);
        const double floor = 1.0e-12 * std::max(1.0, std::abs(dtl));
        if (!(std::abs(fsum) > floor) || !std::isfinite(fsum)) {
            ++_dhat_fsum_guard;
            dhat(ig, ls) = 0.0;
            continue;
        }

        double dh = (jnet_fdm - jnet(ig, ls)) / fsum;
        if (!std::isfinite(dh)) {
            ++_dhat_fsum_guard;
            dhat(ig, ls) = 0.0;
            continue;
        }

        // Diagnostic 2: |dhat| > |dtil| makes one of the two CMFD coupling
        // coefficients change sign, which breaks diagonal dominance and is the
        // classic source of oscillating/negative CMFD flux.
        //
        // Measured on i-SMR CY01 this fires on 2.72% of all surface/group updates
        // with max|dhat/dtil| ~ 2.8, and -- crucially -- it is still firing at the
        // converged state, not only during the early outers. That means these are
        // not transients: the converged nodal solution genuinely wants |dhat| >
        // |dtil| at those surfaces (reflector and strong-absorber interfaces).
        // Clamping them unconditionally therefore does NOT remove an error, it
        // breaks the CNCC consistency condition (dhat is defined so that CMFD
        // reproduces the nodal net current exactly) and biases the answer by
        // ~+100 pcm at BOC on CY01. So the envelope is counted by default and only
        // enforced when RASBERY_DHAT_CLAMP=1 is set, which keeps the diagnostic
        // available without silently changing converged results.
        const double cap = std::abs(dtl);
        if (cap > 0.0) {
            const double ratio = std::abs(dh) / cap;
            if (ratio > _dhat_ratio_max) _dhat_ratio_max = ratio;
            if (ratio > 1.0) {
                ++_dhat_clamped;
                if (_dhat_clamp_enabled) dh = (dh > 0.0 ? cap : -cap);
            }
        }
        dhat(ig, ls) = dh;
    }
}

void CMFD::reportDhatGuardStats(const char* tag) const {
    if (_dhat_total == 0) return;
    std::fprintf(stderr,
                 "[RASBERY][dhat-guard]%s%s clamp=%s total=%lld fsum_guard=%lld (%.3g%%) "
                 "over_envelope=%lld (%.3g%%) max|dhat/dtil|=%.6g\n",
                 (*tag ? " " : ""), tag, _dhat_clamp_enabled ? "on" : "off(count-only)",
                 _dhat_total, _dhat_fsum_guard,
                 100.0 * static_cast<double>(_dhat_fsum_guard) / static_cast<double>(_dhat_total),
                 _dhat_clamped,
                 100.0 * static_cast<double>(_dhat_clamped) / static_cast<double>(_dhat_total),
                 _dhat_ratio_max);
}

void CMFD::setls(const int& l) {
    // determine the area of surfaces at coarse meshes that is normal to directions
    double area[NDIRMAX];

    area[XDIR] = _g.hmesh(YDIR, l) * _g.hmesh(ZDIR, l);
    area[YDIR] = _g.hmesh(XDIR, l) * _g.hmesh(ZDIR, l);
    area[ZDIR] = _g.hmesh(XDIR, l) * _g.hmesh(YDIR, l);

    for (int ige = 0; ige < _g.ng(); ++ige) {

        for (int igs = 0; igs < _g.ng(); ++igs) {
            diag(igs, ige, l) = -_x.xssm(igs, ige, l) * _g.vol(l);
        }
        diag(ige, ige, l) += _x.xsrf(ige, l) * _g.vol(l);

        for (int idir = NDIRMAX - 1; idir >= 0; --idir) {
            int ls = _g.lktosfc(LEFT, idir, l);

            cc(LEFT, idir, ige, l) = (-dtil(ige, ls) + dhat(ige, ls)) * area[idir];
            diag(ige, ige, l) += (dtil(ige, ls) + dhat(ige, ls)) * area[idir];
        }

        for (int idir = 0; idir < NDIRMAX; idir++) {
            int ls                  = _g.lktosfc(RIGHT, idir, l);
            cc(RIGHT, idir, ige, l) = (-dtil(ige, ls) - dhat(ige, ls)) * area[idir];
            diag(ige, ige, l) += (dtil(ige, ls) - dhat(ige, ls)) * area[idir];
        }
    }
}

void CMFD::setNcmfd(int ncmfd) {
    _ncmfd = ncmfd;
}

void CMFD::setEpsl2(double epsl2) {
    _epsl2 = epsl2;
}

void CMFD::updpsi(const int& l, const double* flux) {

    _psi[l] = 0.0;

    for (int ig = 0; ig < _g.ng(); ig++) {
        _psi[l] += flux(ig, l) * _x.xsnf(ig, l);
    }
    _psi[l] = _psi[l] * _g.vol(l);
}
