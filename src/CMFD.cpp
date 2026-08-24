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

    // Geometry is fixed for the lifetime of a Driver.  Build the compact maps
    // once instead of re-running the node/surface indexing helpers in every
    // setls/upddhat/updjnet call of every concurrent instance.
    const int nsurf = _g.nsurf();
    _surface_node.resize(static_cast<size_t>(nsurf) * LR);
    _surface_dir.resize(static_cast<size_t>(nsurf) * LR);
    for (int ls = 0; ls < nsurf; ++ls) {
        for (int lr = 0; lr < LR; ++lr) {
            const size_t i = static_cast<size_t>(ls) * LR + lr;
            _surface_node[i] = _g.lklr(lr, ls);
            _surface_dir[i]  = _g.idirlr(lr, ls);
        }
    }

    const int nxyz = _g.nxyz();
    _node_surface.resize(static_cast<size_t>(nxyz) * NDIRMAX * LR);
    _node_neighbor.resize(static_cast<size_t>(nxyz) * NDIRMAX * LR);
    _node_hmesh.resize(static_cast<size_t>(nxyz) * NDIRMAX);
    _node_face_area.resize(static_cast<size_t>(nxyz) * NDIRMAX);
    _node_volume.resize(static_cast<size_t>(nxyz));
    for (int l = 0; l < nxyz; ++l) {
        const size_t dir_base = static_cast<size_t>(l) * NDIRMAX;
        for (int idir = 0; idir < NDIRMAX; ++idir) {
            _node_hmesh[dir_base + idir] = _g.hmesh(idir, l);
            for (int lr = 0; lr < LR; ++lr) {
                const size_t i = (dir_base + idir) * LR + lr;
                _node_surface[i]  = _g.lktosfc(lr, idir, l);
                _node_neighbor[i] = _g.neib(lr, idir, l);
            }
        }
        _node_face_area[dir_base + XDIR] =
            _node_hmesh[dir_base + YDIR] * _node_hmesh[dir_base + ZDIR];
        _node_face_area[dir_base + YDIR] =
            _node_hmesh[dir_base + XDIR] * _node_hmesh[dir_base + ZDIR];
        _node_face_area[dir_base + ZDIR] =
            _node_hmesh[dir_base + XDIR] * _node_hmesh[dir_base + YDIR];
        _node_volume[static_cast<size_t>(l)] = _g.vol(l);
    }

    _boundary_albedo.resize(static_cast<size_t>(NDIRMAX) * LR);
    for (int idir = 0; idir < NDIRMAX; ++idir)
        for (int lr = 0; lr < LR; ++lr)
            _boundary_albedo[static_cast<size_t>(idir) * LR + lr] = _g.albedo(lr, idir);

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
    const int ll    = cachedSurfaceNode(LEFT, ls);
    const int lr    = cachedSurfaceNode(RIGHT, ls);
    const int idirl = cachedSurfaceDirection(LEFT, ls);
    const int idirr = cachedSurfaceDirection(RIGHT, ls);

    double betal, betar;

    for (int ig = 0; ig < _g.ng(); ig++) {
        if (ll < 0) {
            betal = cachedBoundaryAlbedo(LEFT, idirl) * 0.5;
        } else {
            betal = _x.xsdf(ig, ll) / cachedNodeHmesh(idirl, ll);
        }
        if (lr < 0) {
            betar = cachedBoundaryAlbedo(RIGHT, idirr) * 0.5;
        } else {
            betar = _x.xsdf(ig, lr) / cachedNodeHmesh(idirr, lr);
        }
        dtil(ig, ls) = 2 * betal * betar / (betal + betar);
    }
}

void CMFD::upddhat(const int& ls, double* flux, double* jnet) {
    const int ll = cachedSurfaceNode(LEFT, ls);
    const int lr = cachedSurfaceNode(RIGHT, ls);
    const int ng = _g.ng();

    // APR1400 production decks are two-group.  Keep a generic fallback, but
    // make the hot path a fixed-trip body so the compiler can keep the surface
    // state in registers instead of repeatedly evaluating accessor strides.
    const auto update_group = [&](const int ig) {
        double fdiff, fsum;
        if (ll < 0) {
            fdiff = flux[lr * ng + ig];
            fsum  = flux[lr * ng + ig];
        } else if (lr < 0) {
            fdiff = -flux[ll * ng + ig];
            fsum  = flux[ll * ng + ig];
        } else {
            fdiff = flux[lr * ng + ig] - flux[ll * ng + ig];
            fsum  = flux[lr * ng + ig] + flux[ll * ng + ig];
        }
        const size_t surface_group = static_cast<size_t>(ls) * ng + ig;
        const double dtl           = _dtil[surface_group];
        const double jnet_fdm      = -dtl * fdiff;

        // Guard 1: fsum is a sum of (nominally positive) fluxes, but during the
        // early outers it can collapse toward zero or go negative, which turns
        // the division into a spike that poisons the coupling coefficients.
        // Scale the floor to dtil so the test is dimensionally consistent.
        ++_dhat_total;
        const double floor = 1.0e-12 * std::max(1.0, std::abs(dtl));
        if (!(std::abs(fsum) > floor) || !std::isfinite(fsum)) {
            ++_dhat_fsum_guard;
            _dhat[surface_group] = 0.0;
            return;
        }

        double dh = (jnet_fdm - jnet[surface_group]) / fsum;
        if (!std::isfinite(dh)) {
            ++_dhat_fsum_guard;
            _dhat[surface_group] = 0.0;
            return;
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
        _dhat[surface_group] = dh;
    };

    if (ng == 2) {
        update_group(0);
        update_group(1);
    } else {
        for (int ig = 0; ig < ng; ++ig) update_group(ig);
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
    // Surface areas and volume are immutable geometry.  Reuse the constructor
    // cache, while keeping the historical accessor/arithmetic order below so
    // the optimisation remains numerically inert.
    double area[NDIRMAX];
    area[XDIR] = cachedNodeFaceArea(XDIR, l);
    area[YDIR] = cachedNodeFaceArea(YDIR, l);
    area[ZDIR] = cachedNodeFaceArea(ZDIR, l);
    const double volume = cachedNodeVolume(l);

    for (int ige = 0; ige < _g.ng(); ++ige) {

        for (int igs = 0; igs < _g.ng(); ++igs) {
            diag(igs, ige, l) = -_x.xssm(igs, ige, l) * volume;
        }
        diag(ige, ige, l) += _x.xsrf(ige, l) * volume;

        for (int idir = NDIRMAX - 1; idir >= 0; --idir) {
            const int ls = cachedNodeSurface(LEFT, idir, l);

            cc(LEFT, idir, ige, l) = (-dtil(ige, ls) + dhat(ige, ls)) * area[idir];
            diag(ige, ige, l) += (dtil(ige, ls) + dhat(ige, ls)) * area[idir];
        }

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            const int ls = cachedNodeSurface(RIGHT, idir, l);
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
    const int ng = _g.ng();
    if (ng == 2) {
        const int nxyz = _g.nxyz();
        const double* const xsnf = _x.xsnfData();
        double value = 0.0;
        value += flux[static_cast<size_t>(l) * 2 + 0] * xsnf[l];
        value += flux[static_cast<size_t>(l) * 2 + 1] *
                 xsnf[static_cast<size_t>(nxyz) + l];
        _psi[l] = value * cachedNodeVolume(l);
        return;
    }

    _psi[l] = 0.0;
    for (int ig = 0; ig < ng; ++ig)
        _psi[l] += flux[static_cast<size_t>(l) * ng + ig] * _x.xsnf(ig, l);
    _psi[l] = _psi[l] * cachedNodeVolume(l);
}
