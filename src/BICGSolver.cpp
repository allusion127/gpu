#include "BICGSolver.h"

#include <algorithm>
#include <cmath>

#define diag(igs, ige, l)   diag[(l) * _g.ng2() + (ige) * _g.ng() + (igs)]
#define cc(lr, idir, ig, l) cc[(l) * _g.ng() * NDIRMAX * LR + (ig) * NDIRMAX * LR + (idir) * LR + (lr)]
#define src(ig, l)          src[(l) * _g.ng() + (ig)]
#define aphi(ig, l)         aphi[(l) * _g.ng() + (ig)]
#define phi(ig, l)          phi[(l) * _g.ng() + (ig)]

#define vr(ig, l)  _vr[((l) * _g.ng()) + (ig)]
#define vr0(ig, l) _vr0[((l) * _g.ng()) + (ig)]
#define vp(ig, l)  _vp[((l) * _g.ng()) + (ig)]
#define vv(ig, l)  _vv[((l) * _g.ng()) + (ig)]
#define vs(ig, l)  _vs[((l) * _g.ng()) + (ig)]
#define vt(ig, l)  _vt[((l) * _g.ng()) + (ig)]
#define vy(ig, l)  _vy[((l) * _g.ng()) + (ig)]
#define vz(ig, l)  _vz[((l) * _g.ng()) + (ig)]

using namespace rasbery;

BICGSolver::BICGSolver(Geometry& g)
    : _g(g), _diag_ptr(nullptr) {
    const size_t nv = static_cast<size_t>(_g.ng()) * _g.nxyz();
    const size_t nm = static_cast<size_t>(_g.ng2()) * _g.nxyz();

    // BiCGSTAB workspace
    _vz.assign(nv, 0.0);
    _vy.assign(nv, 0.0);
    _vr.assign(nv, 0.0);
    _vr0.assign(nv, 0.0);
    _vp.assign(nv, 0.0);
    _vv.assign(nv, 0.0);
    _vs.assign(nv, 0.0);
    _vt.assign(nv, 0.0);

    // SSOR preconditioner
    _dinv.assign(nm, 0.0);
    _ssor_tmp.assign(nv, 0.0);
}

BICGSolver::~BICGSolver() = default;

// ============================================================
// Reset: compute initial residual r = b - A*x
// ============================================================

double BICGSolver::reset(const int& ig, const int& l, double* diag, double* cc, double* phi, double* src) {
    double aphi = axb(ig, l, diag, cc, phi);
    vr(ig, l)   = src(ig, l) - aphi;
    vr0(ig, l)  = vr(ig, l);
    vp(ig, l)   = 0.0;
    vv(ig, l)   = 0.0;

    return vr(ig, l) * vr(ig, l);
}

void BICGSolver::reset(double* diag, double* cc, double* phi, double* src, double& r20) {
    _calpha = 1;
    _crho   = 1;
    _comega = 1;

    r20 = 0;
    for (int l = 0; l < _g.nxyz(); ++l)
        for (int ig = 0; ig < _g.ng(); ++ig)
            r20 += reset(ig, l, diag, cc, phi, src);

    r20 = sqrt(r20);
}

// ============================================================
// SSOR Preconditioner: facilu computes D^{-1} blocks
// ============================================================

void BICGSolver::facilu(double* diag, double* cc) {
    _diag_ptr      = diag;
    const int nxyz = _g.nxyz();

    // Precompute inverse of 2x2 diagonal blocks
    for (int l = 0; l < nxyz; ++l) {
        const double a00 = diag[l * 4 + 0];
        const double a10 = diag[l * 4 + 1];
        const double a01 = diag[l * 4 + 2];
        const double a11 = diag[l * 4 + 3];

        const double rdet = 1.0 / (a00 * a11 - a10 * a01);

        _dinv[l * 4 + 0] = rdet * a11;
        _dinv[l * 4 + 1] = -rdet * a10;
        _dinv[l * 4 + 2] = -rdet * a01;
        _dinv[l * 4 + 3] = rdet * a00;
    }
}

// ============================================================
// SSOR Preconditioner Application: x = M^{-1} * b
// Symmetric Gauss-Seidel (ω=1):
//   Forward:  (D + L) * tmp = b
//   Backward: (D + U) * x   = D * tmp
// ============================================================

void BICGSolver::minv(double* cc, double* b, double* x) {
    const int     nxyz = _g.nxyz();
    double*       tmp  = _ssor_tmp.data();
    const double* dinv = _dinv.data();
    const double* dptr = _diag_ptr;

    // Forward sweep: for l = 0..nxyz-1
    //   tmp(l) = D^{-1} * (b(l) - sum_{ln < l} cc(l,ln) * tmp(ln))
    for (int l = 0; l < nxyz; ++l) {
        double b0 = b[l * 2 + 0];
        double b1 = b[l * 2 + 1];

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g.neib(lr, idir, l);
                if (ln >= 0 && ln < l) {
                    b0 -= cc[l * 12 + 0 * 6 + idir * 2 + lr] * tmp[ln * 2 + 0];
                    b1 -= cc[l * 12 + 1 * 6 + idir * 2 + lr] * tmp[ln * 2 + 1];
                }
            }
        }

        tmp[l * 2 + 0] = dinv[l * 4 + 0] * b0 + dinv[l * 4 + 1] * b1;
        tmp[l * 2 + 1] = dinv[l * 4 + 2] * b0 + dinv[l * 4 + 3] * b1;
    }

    // Backward sweep: for l = nxyz-1..0
    //   x(l) = D^{-1} * (D*tmp(l) - sum_{ln > l} cc(l,ln) * x(ln))
    for (int l = nxyz - 1; l >= 0; --l) {
        // D * tmp
        double b0 = dptr[l * 4 + 0] * tmp[l * 2 + 0] + dptr[l * 4 + 1] * tmp[l * 2 + 1];
        double b1 = dptr[l * 4 + 2] * tmp[l * 2 + 0] + dptr[l * 4 + 3] * tmp[l * 2 + 1];

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g.neib(lr, idir, l);
                if (ln >= 0 && ln > l) {
                    b0 -= cc[l * 12 + 0 * 6 + idir * 2 + lr] * x[ln * 2 + 0];
                    b1 -= cc[l * 12 + 1 * 6 + idir * 2 + lr] * x[ln * 2 + 1];
                }
            }
        }

        x[l * 2 + 0] = dinv[l * 4 + 0] * b0 + dinv[l * 4 + 1] * b1;
        x[l * 2 + 1] = dinv[l * 4 + 2] * b0 + dinv[l * 4 + 3] * b1;
    }
}

// ============================================================
// BiCGSTAB Iteration
// ============================================================

void BICGSolver::solve(double* diag, double* cc, double& r20, double* phi, double& r2) {
    int n = _g.nxyz() * _g.ng();

    // Preconditioned BiCGSTAB
    double crhod = _crho;
    _crho        = milk::dot(static_cast<size_t>(n), _vr0.data(), 1, _vr.data(), 1);
    _cbeta       = _crho * _calpha / (crhod * _comega);

    for (int i = 0; i < n; ++i) {
        _vp[i] = _vr[i] + _cbeta * (_vp[i] - _comega * _vv[i]);
    }

    minv(cc, _vp.data(), _vy.data());
    axb(diag, cc, _vy.data(), _vv.data());

    double r0v = milk::dot(static_cast<size_t>(n), _vr0.data(), 1, _vv.data(), 1);

    if (abs(r0v) < 1.E-10) {
        return;
    }

    _calpha = _crho / r0v;

    for (int i = 0; i < n; ++i) {
        _vs[i] = _vr[i] - _calpha * _vv[i];
    }

    minv(cc, _vs.data(), _vz.data());
    axb(diag, cc, _vz.data(), _vt.data());

    double pts = milk::dot(static_cast<size_t>(n), _vs.data(), 1, _vt.data(), 1);
    double ptt = milk::dot(static_cast<size_t>(n), _vt.data(), 1, _vt.data(), 1);

    _comega = 0.0;
    if (ptt != 0.0) {
        _comega = pts / ptt;
    }

    for (int i = 0; i < n; ++i) {
        phi[i] += _calpha * _vy[i] + _comega * _vz[i];
        _vr[i] = _vs[i] - _comega * _vt[i];
    }

    if (r20 != 0.0) {
        r2 = sqrt(ptt) / r20;
    }
}

void BICGSolver::axb(double* diag, double* cc, double* phi, double* aphi) {
    const int    ng   = _g.ng();
    const int    ng2  = _g.ng2();
    const int    nxyz = _g.nxyz();
    const int    ncc  = ng * NDIRMAX * LR;
#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
    for (int l = 0; l < nxyz; ++l) {
        // Neighbor topology is identical for all groups; resolve it once per node.
        int       nln[NDIRMAX * LR];
        int       nslot[NDIRMAX * LR];
        int       nn = 0;
        for (int idir = 0; idir < NDIRMAX; ++idir)
            for (int lr = 0; lr < LR; ++lr) {
                const int neighbor = _g.neib(lr, idir, l);
                if (neighbor != -1) {
                    nln[nn]   = neighbor;
                    nslot[nn] = idir * LR + lr;
                    ++nn;
                }
            }
        const double* diag_l = diag + static_cast<size_t>(l) * ng2;
        const double* cc_l   = cc + static_cast<size_t>(l) * ncc;
        const double* phi_l  = phi + static_cast<size_t>(l) * ng;
        for (int ig = 0; ig < ng; ++ig) {
            double ab = 0.0;
            for (int igs = 0; igs < ng; ++igs)
                ab += diag_l[ig * ng + igs] * phi_l[igs];
            for (int k = 0; k < nn; ++k)
                ab += cc_l[ig * NDIRMAX * LR + nslot[k]] * phi[static_cast<size_t>(nln[k]) * ng + ig];
            aphi[static_cast<size_t>(l) * ng + ig] = ab;
        }
    }
}

double BICGSolver::axb(const int& ig, const int& l, double* diag, double* cc, double* phi) {
    double ab = 0.0;
    for (int igs = 0; igs < _g.ng(); ++igs) {
        ab += diag(igs, ig, l) * phi(igs, l);
    }

    for (int idir = 0; idir < NDIRMAX; ++idir) {
        for (int lr = 0; lr < LR; ++lr) {
            int ln = _g.neib(lr, idir, l);
            if (ln != -1)
                ab += cc(lr, idir, ig, l) * phi(ig, ln);
        }
    }

    return ab;
}
