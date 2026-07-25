#include "PPR.h"
#include <algorithm>
#include <limits>

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

    // All coefficient arrays are owned by Geometry; just cache pointers.
    _phic = _g.Phic();
    _p    = _g.CoeffPart();
    _a    = _g.CoeffHom();
    _c    = _g.CoeffFit();
    _q    = _g.CoeffExp();
    _l    = _g.CoeffLeak();
    _bt   = _g.CoeffBuckling();
}

void PPR::buildQuadratureTable() {
    const int               ndiv      = _g.ndivxy();
    const int               npins     = _g.npins();
    const double            inv_npins = 1.0 / npins;
    const double            inv_ndiv  = 1.0 / ndiv;
    static constexpr double xi3[3]    = {-0.7745966692414834, 0.0, 0.7745966692414834};
    static constexpr double wi3[3]    = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

    _pin_quad_table.resize(npins * npins);

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

            auto& info = _pin_quad_table[py * npins + px];
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
    _quad_table_built = true;
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

    // Reflection flags: compare neibr vs neibrb
    bool xr_w = (_g.neibr(WEST, l2d) != _g.neibrb(WEST, l2d));
    bool xr_e = (_g.neibr(EAST, l2d) != _g.neibrb(EAST, l2d));
    bool yr_n = (_g.neibr(NORTH, l2d) != _g.neibrb(NORTH, l2d));
    bool yr_s = (_g.neibr(SOUTH, l2d) != _g.neibrb(SOUTH, l2d));

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

void PPR::reset(const double reigv, double* jnet, double* phif, double* phis) {

    // 1. Get mesh information and initialize variables
    _reigv = reigv;
    _jnet  = jnet;
    _phif  = phif;
    _phis  = phis;

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

            c(2, 2) = aflux + 0.25 * (phic(NE) + phic(NW) + phic(SE) + phic(SW)) - 0.5 * (phis(XDIR, RIGHT) + phis(XDIR, LEFT) + phis(YDIR, RIGHT) + phis(YDIR, LEFT));

            c(2, 1) = 0.25 * (phic(SE) + phic(SW) - phic(NE) - phic(NW)) + 0.5 * (phis(YDIR, RIGHT) - phis(YDIR, LEFT));
            c(1, 2) = 0.25 * (phic(SE) - phic(SW) + phic(NE) - phic(NW)) + 0.5 * (phis(XDIR, RIGHT) - phis(XDIR, LEFT));

            c(2, 0) = 0.7142857143 * (-2 * aflux + phis(XDIR, LEFT) + phis(XDIR, RIGHT)) + (0.03571428571 * _hmesh * invD) * (jnet(XDIR, LEFT) - jnet(XDIR, RIGHT));
            c(0, 2) = 0.7142857143 * (-2 * aflux + phis(YDIR, LEFT) + phis(YDIR, RIGHT)) + (0.03571428571 * _hmesh * invD) * (jnet(YDIR, LEFT) - jnet(YDIR, RIGHT));

            c(1, 1) = 0.25 * (phic(SE) - phic(SW) - phic(NE) + phic(NW));

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

    // updateHomogeneous
    double hFlux_SW = phic(SW) - (p(0, 0) + p(0, 1) + p(0, 2) + p(0, 3) + p(0, 4) - p(1, 0) - p(1, 1) - p(1, 2) - p(1, 3) + p(2, 0) + p(2, 1) + p(2, 2) - p(3, 0) - p(3, 1) + p(4, 0));
    double hFlux_SE = phic(SE) - (p(0, 0) + p(0, 1) + p(0, 2) + p(0, 3) + p(0, 4) + p(1, 0) + p(1, 1) + p(1, 2) + p(1, 3) + p(2, 0) + p(2, 1) + p(2, 2) + p(3, 0) + p(3, 1) + p(4, 0));
    double hFlux_NW = phic(NW) - (p(0, 0) - p(0, 1) + p(0, 2) - p(0, 3) + p(0, 4) - p(1, 0) + p(1, 1) - p(1, 2) + p(1, 3) + p(2, 0) - p(2, 1) + p(2, 2) - p(3, 0) + p(3, 1) + p(4, 0));
    double hFlux_NE = phic(NE) - (p(0, 0) - p(0, 1) + p(0, 2) - p(0, 3) + p(0, 4) + p(1, 0) - p(1, 1) + p(1, 2) - p(1, 3) + p(2, 0) - p(2, 1) + p(2, 2) + p(3, 0) - p(3, 1) + p(4, 0));

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

void PPR::drive(int niter) {
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

void PPR::reconstructPinPower(bool use_quadrature, bool reconstruct_flux) {
    // Build quadrature table once (geometry-dependent, never changes)
    if (use_quadrature && !_quad_table_built) buildQuadratureTable();

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
            size_t mi    = _xs.comp(ref_lk);
            auto&  model = _xs.models()[mi];
            int    burn  = _xs.burn(ref_lk);

            // Interpolate fmap/gmap between bounding burnup points
            auto& refrMap = model._refr_dpts[0];
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
                        const auto& pin_info = _pin_quad_table[pin_idx];

                        for (const auto& ovl : pin_info.overlaps) {
                            const int li = ovl.dj * ndiv + ovl.di;
                            const int l  = _g.latol(li, la);
                            if (l < 0) continue;
                            has_valid_node = true;
                            const int lk   = l + nxy * k;

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
    double nodal_power_sum = 0.0;
    double fuel_vol_sum    = 0.0;
#pragma omp parallel for reduction(+ : nodal_power_sum, fuel_vol_sum) schedule(static) if (_nxyz > rasbery_omp_gate)
    for (int lk = 0; lk < _nxyz; ++lk) {
        if (!_g.IsFuel(lk)) continue;
        const double vol = _g.vol(lk);
        if (vol <= 1.0e-20) continue;

        double node_power = 0.0;
        for (int g = 0; g < ng; ++g)
            node_power += _xs.xskf(g, lk) * _phif[lk * ng + g];

        nodal_power_sum += node_power * vol;
        fuel_vol_sum += vol;
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
