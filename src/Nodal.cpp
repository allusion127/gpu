#include "Nodal.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define jnet(ig, lks)    (_jnet[(lks) * _ng + ig])
#define trlcff0(ig, lkd) (_trlcff0[(lkd) * _ng + ig])
#define trlcff1(ig, lkd) (_trlcff1[(lkd) * _ng + ig])
#define trlcff2(ig, lkd) (_trlcff2[(lkd) * _ng + ig])
#define eta1(ig, lkd)    (_eta1[(lkd) * _ng + ig])
#define eta2(ig, lkd)    (_eta2[(lkd) * _ng + ig])
#define m260(ig, lkd)    (_m260[(lkd) * _ng + ig])
#define m251(ig, lkd)    (_m251[(lkd) * _ng + ig])
#define m253(ig, lkd)    (_m253[(lkd) * _ng + ig])
#define m262(ig, lkd)    (_m262[(lkd) * _ng + ig])
#define m264(ig, lkd)    (_m264[(lkd) * _ng + ig])
#define diagD(ig, lkd)   (_diagD[(lkd) * _ng + ig])
#define diagDI(ig, lkd)  (_diagDI[(lkd) * _ng + ig])
#define mu(i, j, lkd)    (_mu[(lkd) * _ng2 + (j) * _ng + i])
#define tau(i, j, lkd)   (_tau[(lkd) * _ng2 + (j) * _ng + i])
#define matM(i, j, lk)   (_matM[(lk) * _ng2 + (j) * _ng + i])
#define matMI(i, j, lk)  (_matMI[(lk) * _ng2 + (j) * _ng + i])
#define matMs(i, j, lk)  (_matMs[(lk) * _ng2 + (j) * _ng + i])
#define matMf(i, j, lk)  (_matMf[(lk) * _ng2 + (j) * _ng + i])
#define flux(ig, lk)     (_flux[(lk) * _ng + ig])

#define dsncff2(ig, lkd) (_dsncff2[(lkd) * _ng + ig])
#define dsncff4(ig, lkd) (_dsncff4[(lkd) * _ng + ig])
#define dsncff6(ig, lkd) (_dsncff6[(lkd) * _ng + ig])

using namespace rasbery;

Nodal::Nodal(Geometry& g, XSSet& xs)
    : _g(g), xs(xs) {

    _ng    = _g.ng();
    _nxyz  = _g.nxyz();
    _nsurf = _g.nsurf();
    _ng2   = _ng * _ng;

    _trlcff0 = new double[_nxyz * NDIRMAX * _ng];
    _trlcff1 = new double[_nxyz * NDIRMAX * _ng];
    _trlcff2 = new double[_nxyz * NDIRMAX * _ng];
    _eta1    = new double[_nxyz * NDIRMAX * _ng];
    _eta2    = new double[_nxyz * NDIRMAX * _ng];
    _m260    = new double[_nxyz * NDIRMAX * _ng];
    _m251    = new double[_nxyz * NDIRMAX * _ng];
    _m253    = new double[_nxyz * NDIRMAX * _ng];
    _m262    = new double[_nxyz * NDIRMAX * _ng];
    _m264    = new double[_nxyz * NDIRMAX * _ng];
    _diagDI  = new double[_nxyz * NDIRMAX * _ng];
    _diagD   = new double[_nxyz * NDIRMAX * _ng];
    _constant_xsrf = new double[_nxyz * _ng];
    _constant_xsdf = new double[_nxyz * _ng];
    std::fill_n(_constant_xsrf, _nxyz * _ng, std::numeric_limits<double>::quiet_NaN());
    std::fill_n(_constant_xsdf, _nxyz * _ng, std::numeric_limits<double>::quiet_NaN());
    _dsncff2 = new double[_nxyz * NDIRMAX * _ng];
    _dsncff4 = new double[_nxyz * NDIRMAX * _ng];
    _dsncff6 = new double[_nxyz * NDIRMAX * _ng];
    _mu      = new double[_nxyz * NDIRMAX * _ng2];
    _tau     = new double[_nxyz * NDIRMAX * _ng2];
    _matM    = new double[_nxyz * _ng2];
    _matMI   = new double[_nxyz * _ng2];
    _matMs   = new double[_nxyz * _ng2];
    _matMf   = new double[_nxyz * _ng2];
}

Nodal::~Nodal() {
    delete[] _trlcff0;
    delete[] _trlcff1;
    delete[] _trlcff2;
    delete[] _eta1;
    delete[] _eta2;
    delete[] _m260;
    delete[] _m251;
    delete[] _m253;
    delete[] _m262;
    delete[] _m264;
    delete[] _diagDI;
    delete[] _diagD;
    delete[] _constant_xsrf;
    delete[] _constant_xsdf;
    delete[] _dsncff2;
    delete[] _dsncff4;
    delete[] _dsncff6;
    delete[] _mu;
    delete[] _tau;
    delete[] _matM;
    delete[] _matMI;
    delete[] _matMs;
    delete[] _matMf;
}

void Nodal::updateConstant(const int& lk) {
    const int lkg0 = lk * _ng;
    bool unchanged = true;
    for (int ig = 0; ig < _ng; ++ig) {
        unchanged = unchanged &&
                    _constant_xsrf[lkg0 + ig] == xs.xsrf(ig, lk) &&
                    _constant_xsdf[lkg0 + ig] == xs.xsdf(ig, lk);
    }
    if (unchanged) return;

    int lkd0 = lk * NDIRMAX;

    for (int idir = 0; idir < NDIRMAX; idir++) {
        int lkd = lkd0 + idir;

        for (int ig = 0; ig < _ng; ig++) {
            double kp2    = xs.xsrf(ig, lk) * _g.hmesh(idir, lk) * _g.hmesh(idir, lk) / (4 * xs.xsdf(ig, lk));
            double kp     = sqrt(kp2);
            double kp3    = kp2 * kp;
            double kp4    = kp2 * kp2;
            double kp5    = kp2 * kp3;
            double rkp    = 1 / kp;
            double rkp2   = rkp * rkp;
            double rkp3   = rkp2 * rkp;
            double rkp4   = rkp2 * rkp2;
            double rkp5   = rkp2 * rkp3;
            // sinh/cosh via a single exp (glibc sinh/cosh are slow and call expm1/hypot); same
            // trick as PPR.cpp. One exp + reciprocal instead of two transcendental library calls.
            double ekp    = std::exp(kp);
            double iekp   = 1.0 / ekp;
            double sinhkp = 0.5 * (ekp - iekp);
            double coshkp = 0.5 * (ekp + iekp);

            // calculate coefficient of basic functions P5and P6
            double bfcff0 = -sinhkp * rkp;
            double bfcff2 = -5 * (-3 * kp * coshkp + 3 * sinhkp + kp2 * sinhkp) * rkp3;
            double bfcff4 =
                -9. * (-105 * kp * coshkp - 10 * kp3 * coshkp + 105 * sinhkp + 45 * kp2 * sinhkp + kp4 * sinhkp) *
                rkp5;
            double bfcff1 = -3 * (kp * coshkp - sinhkp) * rkp2;
            double bfcff3 = -7 * (15 * kp * coshkp + kp3 * coshkp - 15 * sinhkp - 6 * kp2 * sinhkp) * rkp4;

            double oddtemp  = 1 / (sinhkp + bfcff1 + bfcff3);
            double eventemp = 1 / (coshkp + bfcff0 + bfcff2 + bfcff4);

            // eta1, eta2
            eta1(ig, lkd) = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
            eta2(ig, lkd) = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;

            // set to variables that depends on node properties by integrating of Pi* pj over - 1 ~1
            m260(ig, lkd) = 2 * eta2(ig, lkd);
            m251(ig, lkd) = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
            m253(ig, lkd) = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) * oddtemp * rkp2;
            m262(ig, lkd) = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) * eventemp * rkp;
            m264(ig, lkd) = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp + (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;

            diagD(ig, lkd)  = 4 * xs.xsdf(ig, lk) / (_g.hmesh(idir, lk) * _g.hmesh(idir, lk));
            diagDI(ig, lkd) = 1.0 / diagD(ig, lkd);
        }
    }

    for (int ig = 0; ig < _ng; ++ig) {
        _constant_xsrf[lkg0 + ig] = xs.xsrf(ig, lk);
        _constant_xsdf[lkg0 + ig] = xs.xsdf(ig, lk);
    }

    // Any recompute makes the device copy of the nine coefficient arrays
    // stale.  Racy increments from the omp for are fine: the consumer only
    // compares against its resident value, so any change forces a re-upload.
    ++_const_generation;
}

void Nodal::updateMatrix(const int& lk) {
    int lkd0 = lk * NDIRMAX;

    for (int ige = 0; ige < _ng; ige++) {
        for (int igs = 0; igs < _ng; igs++) {
            matMs(igs, ige, lk) = -xs.xssm(igs, ige, lk);
            matMf(igs, ige, lk) = xs.chif(ige, lk) * xs.xsnf(igs, lk);
        }
        matMs(ige, ige, lk) += xs.xsrf(ige, lk);

        for (int igs = 0; igs < _ng; igs++) {
            matM(igs, ige, lk) = matMs(igs, ige, lk) - _reigv * matMf(igs, ige, lk);
        }
    }

    double det = matM(0, 0, lk) * matM(1, 1, lk) - matM(1, 0, lk) * matM(0, 1, lk);

    if (abs(det) > 1.E-10) {
        double rdet     = 1 / det;
        matMI(0, 0, lk) = rdet * matM(1, 1, lk);
        matMI(1, 0, lk) = -rdet * matM(1, 0, lk);
        matMI(0, 1, lk) = -rdet * matM(0, 1, lk);
        matMI(1, 1, lk) = rdet * matM(0, 0, lk);
    } else {
        matMI(0, 0, lk) = 0;
        matMI(1, 0, lk) = 0;
        matMI(0, 1, lk) = 0;
        matMI(1, 1, lk) = 0;
    }

    double rm011 = 1. / m011;

    for (int idir = 0; idir < NDIRMAX; idir++) {
        int lkd = lkd0 + idir;

        double tempz[2][2] = {};

        for (int igd = 0; igd < _ng; igd++) {
            double tau1 = m033 * (diagDI(igd, lkd) / m253(igd, lkd));

            tempz[igd][igd] = tempz[igd][igd] + m231;

            for (int igs = 0; igs < _ng; igs++) {
                tau(igs, igd, lkd) = tau1 * matM(igs, igd, lk);

                // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
                tempz[igs][igd] += m251(igd, lkd) * tau(igs, igd, lkd);

                // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
                tempz[igs][igd] *= diagD(igd, lkd);
            }
        }

        // mu=m011_inv*M_inv*D*(m231*I+m251*tau)
        mu(0, 0, lkd) = rm011 * (matMI(0, 0, lk) * tempz[0][0] + matMI(1, 0, lk) * tempz[0][1]);
        mu(1, 0, lkd) = rm011 * (matMI(0, 0, lk) * tempz[1][0] + matMI(1, 0, lk) * tempz[1][1]);
        mu(0, 1, lkd) = rm011 * (matMI(0, 1, lk) * tempz[0][0] + matMI(1, 1, lk) * tempz[0][1]);
        mu(1, 1, lkd) = rm011 * (matMI(0, 1, lk) * tempz[1][0] + matMI(1, 1, lk) * tempz[1][1]);
    }
}

void Nodal::trlcffbyintg(double* avgtrl3, double* hmesh3, double& trlcff1, double& trlcff2) {
    double sh[4];

    double rh = (1 / ((hmesh3[LEFT] + hmesh3[CENTER] + hmesh3[RIGHT]) * (hmesh3[LEFT] + hmesh3[CENTER]) *
                      (hmesh3[CENTER] + hmesh3[RIGHT])));
    sh[0]     = (2 * hmesh3[LEFT] + hmesh3[CENTER]) * (hmesh3[LEFT] + hmesh3[CENTER]);
    sh[1]     = hmesh3[LEFT] + hmesh3[CENTER];
    sh[2]     = (hmesh3[CENTER] + 2 * hmesh3[RIGHT]) * (hmesh3[CENTER] + hmesh3[RIGHT]);
    sh[3]     = hmesh3[CENTER] + hmesh3[RIGHT];

    if (hmesh3[LEFT] == 0.0) {
        trlcff1 = 0.125 * (5. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
        trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[RIGHT]);
    } else if (hmesh3[RIGHT] == 0.0) {
        trlcff1 = -0.125 * (5. * avgtrl3[CENTER] + avgtrl3[LEFT]);
        trlcff2 = 0.125 * (-3. * avgtrl3[CENTER] + avgtrl3[LEFT]);
    } else {
        trlcff1 = 0.5 * rh * hmesh3[CENTER] *
                  ((avgtrl3[CENTER] - avgtrl3[LEFT]) * sh[2] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[0]);
        trlcff2 = 0.5 * rh * (hmesh3[CENTER] * hmesh3[CENTER]) *
                  ((avgtrl3[LEFT] - avgtrl3[CENTER]) * sh[3] + (avgtrl3[RIGHT] - avgtrl3[CENTER]) * sh[1]);
    }
}

void Nodal::caltrlcff0(const int& lk) {
    int lkd0 = lk * NDIRMAX;

    double avgjnet[NDIRMAX];

    for (int ig = 0; ig < _ng; ig++) {
        for (int idir = 0; idir < NDIRMAX; idir++) {
            int lsl = _g.lktosfc(LEFT, idir, lk);
            int lsr = _g.lktosfc(RIGHT, idir, lk);

            avgjnet[idir] = (jnet(ig, lsr) - jnet(ig, lsl)) / _g.hmesh(idir, lk);
        }

        trlcff0(ig, lkd0 + XDIR) = avgjnet[YDIR] + avgjnet[ZDIR];
        trlcff0(ig, lkd0 + YDIR) = avgjnet[XDIR] + avgjnet[ZDIR];
        trlcff0(ig, lkd0 + ZDIR) = avgjnet[XDIR] + avgjnet[YDIR];
    }
}

void Nodal::caltrlcff12(const int& lk) {
    int lkd0 = lk * NDIRMAX;

    for (int idir = 0; idir < NDIRMAX; idir++) {
        int lkd = lkd0 + idir;

        int lkl = _g.neib(LEFT, idir, lk);
        int lkr = _g.neib(RIGHT, idir, lk);

        for (int ig = 0; ig < _ng; ig++) {
            double avgtrl3[LRC]{};
            double hmesh3[LRC]{};
            hmesh3[CENTER]  = _g.hmesh(idir, lk);
            avgtrl3[CENTER] = trlcff0(ig, lkd);

            if (lkl > -1) {
                int lsl       = _g.lktosfc(LEFT, idir, lk);
                int idirl     = _g.idirlr(LEFT, lsl);
                hmesh3[LEFT]  = _g.hmesh(idirl, lkl);
                avgtrl3[LEFT] = trlcff0(ig, lkl * NDIRMAX + idirl);
            } else if (_g.albedo(LEFT, idir) < 1.0E-10) {
                hmesh3[LEFT]  = hmesh3[CENTER];
                avgtrl3[LEFT] = avgtrl3[CENTER];
            }

            if (lkr > -1) {
                int lsr        = _g.lktosfc(RIGHT, idir, lk);
                int idirr      = _g.idirlr(RIGHT, lsr);
                hmesh3[RIGHT]  = _g.hmesh(idirr, lkr);
                avgtrl3[RIGHT] = trlcff0(ig, lkr * NDIRMAX + idirr);
            } else if (_g.albedo(RIGHT, idir) < 1.0E-10) {
                hmesh3[RIGHT]  = hmesh3[CENTER];
                avgtrl3[RIGHT] = avgtrl3[CENTER];
            }

            trlcffbyintg(avgtrl3, hmesh3, trlcff1(ig, lkd), trlcff2(ig, lkd));
        }
    }
}

void Nodal::calculateEven(const int& lk) {
    int lkd0 = lk * NDIRMAX;

    for (int idir = 0; idir < NDIRMAX; idir++) {
        int    lkd = lkd0 + idir;
        double at2[2][2], a[2][2], rm4464[2], bt1[2], bt2[2], b[2];

        for (int igd = 0; igd < _ng; igd++) {
            rm4464[igd] = 0.0;
            if (abs(m264(igd, lkd)) > 1.0E-10)
                rm4464[igd] = m044 / m264(igd, lkd);

            double mu2 = rm4464[igd] * m260(igd, lkd) * diagDI(igd, lkd);

            for (int igs = 0; igs < _ng; igs++) {
                at2[igs][igd] = m022 * rm220 * mu2 * matM(igs, igd, lk);
            }
            at2[igd][igd] += m022 * rm220 * m240;
        }

        for (int igd = 0; igd < _ng; igd++) {
            double mu1 = rm4464[igd] * m262(igd, lkd);
            for (int igs = 0; igs < _ng; igs++) {
                a[igs][igd] =
                    mu1 * matM(igs, igd, lk) + matM(0, igd, lk) * at2[igs][0] + matM(1, igd, lk) * at2[igs][1];
            }
            a[igd][igd] += diagD(igd, lkd) * m242;
            bt2[igd] = 2 * (matM(0, igd, lk) * flux(0, lk) + matM(1, igd, lk) * flux(1, lk) + trlcff0(igd, lkd));
            bt1[igd] = m022 * rm220 * diagDI(igd, lkd) * bt2[igd];
        }

        for (int ig = 0; ig < _ng; ig++) {
            b[ig] = m022 * trlcff2(ig, lkd) + (matM(0, ig, lk) * bt1[0] + matM(1, ig, lk) * bt1[1]);
        }

        double rdet = (a[0][0] * a[1][1] - a[1][0] * a[0][1]);

        if (rdet != 0.0) {
            rdet            = 1. / rdet;
            dsncff4(0, lkd) = rdet * (a[1][1] * b[0] - a[1][0] * b[1]);
            dsncff4(1, lkd) = rdet * (a[0][0] * b[1] - a[0][1] * b[0]);
        } else {
            dsncff4(0, lkd) = 0.0;
            dsncff4(1, lkd) = 0.0;
        }

        for (int ig = 0; ig < _ng; ig++) {
            dsncff6(ig, lkd) = diagDI(ig, lkd) * rm4464[ig] *
                               (matM(0, ig, lk) * dsncff4(0, lkd) + matM(1, ig, lk) * dsncff4(1, lkd));
            dsncff2(ig, lkd) =
                rm220 * (diagDI(ig, lkd) * bt2[ig] - m240 * dsncff4(ig, lkd) - m260(ig, lkd) * dsncff6(ig, lkd));
        }
    }
}

void Nodal::calculateJnet(const int& ls) {
    int lkl = _g.lklr(LEFT, ls);
    int lkr = _g.lklr(RIGHT, ls);

    if (lkl < 0) {
        int idirr = _g.idirlr(RIGHT, ls);
        calculateJnet1n(ls, RIGHT, _g.albedo(LEFT, idirr));
    } else if (lkr < 0) {
        int idirl = _g.idirlr(LEFT, ls);
        calculateJnet1n(ls, LEFT, _g.albedo(RIGHT, idirl));
    } else {
        calculateJnet2n(ls);
    }
}

void Nodal::calculateJnet1n(const int& ls, const int& lr, const double& alb) {
    int lk   = _g.lklr(lr, ls);
    int idir = _g.idirlr(lr, ls);
    // int sgn = sgnlr[lsfclr + lr];
    int lkd = lk * NDIRMAX + idir;
    int sgn = 1;
    if (lr == RIGHT)
        sgn = -1;

    double diagDj[2]{};

    double a11[2][2], a12[2], a13[2], a22[2][2], a23[2], a31[2], a32[2], a33[2];
    double b1[2], b2[2];

    // 1, 1
    for (int ige = 0; ige < _ng; ige++) {
        for (int igs = 0; igs < _ng; igs++) {
            a11[igs][ige] = matM(igs, ige, lk) * m011;
        }
    }

    // 1, 2
    for (int ig = 0; ig < _ng; ig++) {
        a12[ig] = -diagD(ig, lkd) * m231;
    }

    // 1, 3
    for (int ig = 0; ig < _ng; ig++) {
        a13[ig] = -diagD(ig, lkd) * m251(ig, lkd);
    }

    // 2,2
    for (int ige = 0; ige < _ng; ige++) {
        for (int igs = 0; igs < _ng; igs++) {
            a22[igs][ige] = matM(igs, ige, lk) * m033;
        }
    }

    // 2, 3
    for (int ig = 0; ig < _ng; ig++) {
        a23[ig] = -diagD(ig, lkd) * m253(ig, lkd);
    }

    for (int ig = 0; ig < _ng; ig++) {
        diagDj[ig] = 0.5 * _g.hmesh(idir, lk) * diagD(ig, lkd);
    }

    // 3,1
    for (int ig = 0; ig < _ng; ig++) {
        a31[ig] = diagDj[ig] + alb;
    }

    // 3,2
    for (int ig = 0; ig < _ng; ig++) {
        a32[ig] = 6. * diagDj[ig] + alb;
    }

    // 3,3
    for (int ig = 0; ig < _ng; ig++) {
        a33[ig] = diagDj[ig] * eta1(ig, lkd) + alb;
    }

    // make right vector
    for (int ig = 0; ig < _ng; ig++) {
        b1[ig] = -m011 * trlcff1(ig, lkd);
    }
    for (int ig = 0; ig < _ng; ig++) {
        b2[ig] = -sgn *
                 (diagDj[ig] * (3 * dsncff2(ig, lkd) + 10 * dsncff4(ig, lkd) + eta2(ig, lkd) * dsncff6(ig, lkd)) +
                  alb * (flux(ig, lk) + dsncff2(ig, lkd) + dsncff4(ig, lkd) + dsncff6(ig, lkd)));
    }

    for (int ige = 0; ige < _ng; ige++) {
        for (int igs = 0; igs < _ng; igs++) {
            a22[igs][ige] = -a22[igs][ige] / a23[ige];
            a11[igs][ige] = a11[igs][ige] / a31[igs];
        }
    }

    double a[2][2] = {};
    for (int ige = 0; ige < _ng; ige++) {
        for (int igs = 0; igs < _ng; igs++) {
            a[igs][ige] = a13[ige] * a22[igs][ige] - a11[igs][ige] * a32[igs];
        }
    }

    for (int ige = 0; ige < _ng; ige++) {
        a[ige][ige] = a[ige][ige] + a12[ige];
    }

    for (int ige = 0; ige < _ng; ige++) {
        for (int igs = 0; igs < _ng; igs++) {
            a[igs][ige] = a[igs][ige] - (a11[0][ige] * a33[0] * a22[igs][0] + a11[1][ige] * a33[1] * a22[igs][1]);
        }

        b1[ige] = b1[ige] - (a11[0][ige] * b2[0] + a11[1][ige] * b2[1]);
    }

    double oddcff[3][2];

    double rdet = 1.0 / (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
    a11[0][0]   = rdet * a[1][1];
    a11[1][0]   = -rdet * a[1][0];
    a11[0][1]   = -rdet * a[0][1];
    a11[1][1]   = rdet * a[0][0];

    for (int ig = 0; ig < _ng; ig++) {
        oddcff[1][ig] = a11[0][ig] * b1[0] + a11[1][ig] * b1[1];
    }

    for (int ig = 0; ig < _ng; ig++) {
        oddcff[2][ig] = a22[0][ig] * oddcff[1][0] + a22[1][ig] * oddcff[1][1];
    }

    for (int ig = 0; ig < _ng; ig++) {
        oddcff[0][ig] = (b2[ig] - a32[ig] * oddcff[1][ig] - a33[ig] * oddcff[2][ig]) / a31[ig];
    }

    for (int ig = 0; ig < _ng; ig++) {
        jnet(ig, ls) = -_g.hmesh(idir, lk) * 0.5 * diagD(ig, lkd) * (oddcff[0][ig] + 6 * oddcff[1][ig] + eta1(ig, lkd) * oddcff[2][ig] + sgn * (3 * dsncff2(ig, lkd) + 10 * dsncff4(ig, lkd) + eta2(ig, lkd) * dsncff6(ig, lkd)));

        phis(ig, ls) = flux(ig, lk) + dsncff2(ig, lkd) + dsncff4(ig, lkd) + dsncff6(ig, lkd) + sgn * (oddcff[0][ig] + oddcff[1][ig] + oddcff[2][ig]);
    }
}

void Nodal::calculateJnet2n(const int& ls) {
    int lkl = _g.lklr(LEFT, ls);
    int lkr = _g.lklr(RIGHT, ls);

    int idirl = _g.idirlr(LEFT, ls);
    int idirr = _g.idirlr(RIGHT, ls);
    int sgnl  = _g.sgnlr(LEFT, ls);
    int sgnr  = _g.sgnlr(RIGHT, ls);
    int lkdl  = lkl * NDIRMAX + idirl;
    int lkdr  = lkr * NDIRMAX + idirr;

    double diagDj[2][LR], tempz[2][2], tempzI[2][2], zeta1[2][2], zeta2[2], bfc[2], mat1g[2][2];

    for (int ig = 0; ig < _ng; ig++) {
        diagDj[ig][LEFT]  = 0.5 * _g.hmesh(idirl, lkl) * diagD(ig, lkdl);
        diagDj[ig][RIGHT] = 0.5 * _g.hmesh(idirr, lkr) * diagD(ig, lkdr);
    }

    // zeta1 = (mur + I + taur)_inv * (mul + I + taul)
    tempz[0][0] = mu(0, 0, lkdr) + tau(0, 0, lkdr) + 1;
    tempz[1][0] = mu(1, 0, lkdr) + tau(1, 0, lkdr);
    tempz[0][1] = mu(0, 1, lkdr) + tau(0, 1, lkdr);
    tempz[1][1] = mu(1, 1, lkdr) + tau(1, 1, lkdr) + 1;

    double rdet  = 1 / (tempz[0][0] * tempz[1][1] - tempz[1][0] * tempz[0][1]);
    tempzI[0][0] = rdet * tempz[1][1];
    tempzI[1][0] = -rdet * tempz[1][0];
    tempzI[0][1] = -rdet * tempz[0][1];
    tempzI[1][1] = rdet * tempz[0][0];

    tempz[0][0] = mu(0, 0, lkdl) + tau(0, 0, lkdl) + 1;
    tempz[1][0] = mu(1, 0, lkdl) + tau(1, 0, lkdl);
    tempz[0][1] = mu(0, 1, lkdl) + tau(0, 1, lkdl);
    tempz[1][1] = mu(1, 1, lkdl) + tau(1, 1, lkdl) + 1;

    zeta1[0][0] = tempzI[0][0] * tempz[0][0] + tempzI[1][0] * tempz[0][1];
    zeta1[1][0] = tempzI[0][0] * tempz[1][0] + tempzI[1][0] * tempz[1][1];
    zeta1[0][1] = tempzI[0][1] * tempz[0][0] + tempzI[1][1] * tempz[0][1];
    zeta1[1][1] = tempzI[0][1] * tempz[1][0] + tempzI[1][1] * tempz[1][1];

    for (int ig = 0; ig < _ng; ig++) {
        bfc[ig] = (dsncff2(ig, lkdr) + dsncff4(ig, lkdr) + dsncff6(ig, lkdr) + flux(ig, lkr) + matMI(0, ig, lkr) * sgnr * trlcff1(0, lkdr) + matMI(1, ig, lkr) * sgnr * trlcff1(1, lkdr)) + (-dsncff2(ig, lkdl) - dsncff4(ig, lkdl) - dsncff6(ig, lkdl) - flux(ig, lkl) + matMI(0, ig, lkl) * sgnl * trlcff1(0, lkdl) + matMI(1, ig, lkl) * sgnl * trlcff1(1, lkdl));
    }

    for (int ig = 0; ig < _ng; ig++) {
        zeta2[ig] = tempzI[0][ig] * bfc[0] + tempzI[1][ig] * bfc[1];
    }

    // tempz = mur + 6 * I + eta1 * taur
    tempz[0][0] = diagDj[0][RIGHT] * (mu(0, 0, lkdr) + 6 + eta1(0, lkdr) * tau(0, 0, lkdr));
    tempz[1][0] = diagDj[0][RIGHT] * (mu(1, 0, lkdr) + eta1(0, lkdr) * tau(1, 0, lkdr));
    tempz[0][1] = diagDj[1][RIGHT] * (mu(0, 1, lkdr) + eta1(1, lkdr) * tau(0, 1, lkdr));
    tempz[1][1] = diagDj[1][RIGHT] * (mu(1, 1, lkdr) + 6 + eta1(1, lkdr) * tau(1, 1, lkdr));

    // mat1g = mul + 6 * I + eta1 * taul - tempzI
    mat1g[0][0] =
        -diagDj[0][LEFT] * (mu(0, 0, lkdl) + 6 + eta1(0, lkdl) * tau(0, 0, lkdl)) - tempz[0][0] * zeta1[0][0] -
        tempz[1][0] * zeta1[0][1];
    mat1g[1][0] = -diagDj[0][LEFT] * (mu(1, 0, lkdl) + eta1(0, lkdl) * tau(1, 0, lkdl)) - tempz[0][0] * zeta1[1][0] -
                  tempz[1][0] * zeta1[1][1];
    mat1g[0][1] = -diagDj[1][LEFT] * (mu(0, 1, lkdl) + eta1(1, lkdl) * tau(0, 1, lkdl)) - tempz[0][1] * zeta1[0][0] -
                  tempz[1][1] * zeta1[0][1];
    mat1g[1][1] =
        -diagDj[1][LEFT] * (mu(1, 1, lkdl) + 6 + eta1(1, lkdl) * tau(1, 1, lkdl)) - tempz[0][1] * zeta1[1][0] -
        tempz[1][1] * zeta1[1][1];

    double bcc[2], vec1g[2];

    for (int ig = 0; ig < _ng; ig++) {
        bcc[ig] =
            diagDj[ig][LEFT] * (3 * dsncff2(ig, lkdl) + 10 * dsncff4(ig, lkdl) + eta2(ig, lkdl) * dsncff6(ig, lkdl)) + diagDj[ig][RIGHT] *
                                                                                                                           (3 * dsncff2(ig, lkdr) + 10 * dsncff4(ig, lkdr) + eta2(ig, lkdr) * dsncff6(ig, lkdr));
        vec1g[ig] = bcc[ig] - diagDj[ig][LEFT] * (matMI(0, ig, lkl) * sgnl * trlcff1(0, lkdl) + matMI(1, ig, lkl) * sgnl * trlcff1(1, lkdl)) + diagDj[ig][RIGHT] * (matMI(0, ig, lkr) * sgnr * trlcff1(0, lkdr) + matMI(1, ig, lkr) * sgnr * trlcff1(1, lkdr)) - (tempz[0][ig] * zeta2[0] + tempz[1][ig] * zeta2[1]);
    }

    rdet        = 1 / (mat1g[0][0] * mat1g[1][1] - mat1g[1][0] * mat1g[0][1]);
    double tmp  = mat1g[0][0];
    mat1g[0][0] = rdet * mat1g[1][1];
    mat1g[1][0] = -rdet * mat1g[1][0];
    mat1g[0][1] = -rdet * mat1g[0][1];
    mat1g[1][1] = rdet * tmp;

    double oddcff[3][2];

    oddcff[1][0] = zeta2[0] - (zeta1[0][0] * (mat1g[0][0] * vec1g[0] + mat1g[1][0] * vec1g[1]) + zeta1[1][0] * (mat1g[0][1] * vec1g[0] + mat1g[1][1] * vec1g[1]));
    oddcff[1][1] = zeta2[1] - (zeta1[0][1] * (mat1g[0][0] * vec1g[0] + mat1g[1][0] * vec1g[1]) + zeta1[1][1] * (mat1g[0][1] * vec1g[0] + mat1g[1][1] * vec1g[1]));

    oddcff[2][0] = tau(0, 0, lkdr) * oddcff[1][0] + tau(1, 0, lkdr) * oddcff[1][1];
    oddcff[2][1] = tau(0, 1, lkdr) * oddcff[1][0] + tau(1, 1, lkdr) * oddcff[1][1];

    oddcff[0][0] = mu(0, 0, lkdr) * oddcff[1][0] - matMI(0, 0, lkr) * sgnr * trlcff1(0, lkdr) + mu(1, 0, lkdr) * oddcff[1][1] - matMI(1, 0, lkr) * sgnr * trlcff1(1, lkdr);
    oddcff[0][1] = mu(0, 1, lkdr) * oddcff[1][0] - matMI(0, 1, lkr) * sgnr * trlcff1(0, lkdr) + mu(1, 1, lkdr) * oddcff[1][1] - matMI(1, 1, lkr) * sgnr * trlcff1(1, lkdr);

    for (int ig = 0; ig < _ng; ig++) {
        jnet(ig, ls) = sgnr * _g.hmesh(idirr, lkr) * 0.5 * diagD(ig, lkdr) * (-1.0 * oddcff[0][ig] + 3 * dsncff2(ig, lkdr) - 6 * oddcff[1][ig] + 10 * dsncff4(ig, lkdr) - eta1(ig, lkdr) * oddcff[2][ig] + eta2(ig, lkdr) * dsncff6(ig, lkdr));

        phis(ig, ls) = flux(ig, lkr) - (oddcff[0][ig] + oddcff[1][ig] + oddcff[2][ig]) + dsncff2(ig, lkdr) + dsncff4(ig, lkdr) + dsncff6(ig, lkdr);
    }
}

void Nodal::reset(const double& reigv, double* jnet, double* phif, double* phis) {
    _flux  = phif;
    _jnet  = jnet;
    _phis  = phis;
    _reigv = reigv;
}

nodal::NodalView Nodal::MakeView() {
    nodal::NodalView v{};
    v.hmesh   = &_g.hmesh(0, 0);
    v.lktosfc = &_g.lktosfc(0, 0, 0);
    v.neib    = &_g.neib(0, 0, 0);
    v.lklr    = &_g.lklr(0, 0);
    v.idirlr  = &_g.idirlr(0, 0);
    v.sgnlr   = &_g.sgnlr(0, 0);
    v.albedo  = &_g.albedo(0, 0);

    v.xsrf       = xs.xsrfData();
    v.xsnf       = xs.xsnfData();
    v.xssm       = xs.xssmData();
    v.chif       = xs.chifData();
    v.chif_empty = v.chif == nullptr ? 1 : 0;

    v.eta1   = _eta1;
    v.eta2   = _eta2;
    v.m260   = _m260;
    v.m251   = _m251;
    v.m253   = _m253;
    v.m262   = _m262;
    v.m264   = _m264;
    v.diagD  = _diagD;
    v.diagDI = _diagDI;

    v.trlcff0 = _trlcff0;
    v.trlcff1 = _trlcff1;
    v.trlcff2 = _trlcff2;
    v.mu      = _mu;
    v.tau     = _tau;
    v.matM    = _matM;
    v.matMI   = _matMI;
    v.matMs   = _matMs;
    v.matMf   = _matMf;
    v.dsncff2 = _dsncff2;
    v.dsncff4 = _dsncff4;
    v.dsncff6 = _dsncff6;

    v.flux  = _flux;
    v.jnet  = _jnet;
    v.phis  = _phis;
    v.reigv = _reigv;
    v.nxyz  = _nxyz;
    v.nsurf = _nsurf;
    return v;
}

// Capture for offline replay (RASBERY_NODAL_DUMP=<path>): one drive() call's
// full inputs -- geometry tables, xs rows, the nine constants, jnet/flux --
// plus every intermediate array and the outputs, raw doubles.  The call index
// is RASBERY_NODAL_DUMP_CALL (default 5: the first calls run on a rough flux).
// test/nodal_replay.cpp replays the shared body phase by phase against it.
static void nodalDumpState(const char* path, const nodal::NodalView& v,
                           const double* jnet_in) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f)
        return;
    namespace nk = rasbery::nodal;
    const std::int64_t hdr[8] = {v.nxyz, v.nsurf, nk::NDIR, nk::NG,
                                 nk::NEWSB, v.chif_empty, 0, 0};
    std::fwrite(hdr, sizeof hdr[0], 8, f);
    std::fwrite(&v.reigv, sizeof(double), 1, f);

    const std::size_t nx = static_cast<std::size_t>(v.nxyz);
    const std::size_t ns = static_cast<std::size_t>(v.nsurf);
    std::fwrite(v.lktosfc, sizeof(int), nx * nk::NDIR * nk::NLR, f);
    std::fwrite(v.neib, sizeof(int), nx * nk::NEWSB, f);
    std::fwrite(v.lklr, sizeof(int), ns * nk::NLR, f);
    std::fwrite(v.idirlr, sizeof(int), ns * nk::NLR, f);
    std::fwrite(v.sgnlr, sizeof(int), ns * nk::NLR, f);
    std::fwrite(v.hmesh, sizeof(double), nx * nk::NDIR, f);
    std::fwrite(v.albedo, sizeof(double), nk::NDIR * nk::NLR, f);

    std::fwrite(v.xsrf, sizeof(double), nk::NG * nx, f);
    std::fwrite(v.xsnf, sizeof(double), nk::NG * nx, f);
    std::fwrite(v.xssm, sizeof(double), nk::NG2 * nx, f);
    if (!v.chif_empty)
        std::fwrite(v.chif, sizeof(double), nk::NG * nx, f);

    const double* consts[9] = {v.eta1, v.eta2, v.m260, v.m251, v.m253,
                               v.m262, v.m264, v.diagD, v.diagDI};
    for (const double* c : consts)
        std::fwrite(c, sizeof(double), nx * nk::NDIR * nk::NG, f);

    std::fwrite(jnet_in, sizeof(double), ns * nk::NG, f);
    std::fwrite(v.flux, sizeof(double), nx * nk::NG, f);

    // Post-drive state: intermediates then outputs.
    std::fwrite(v.trlcff0, sizeof(double), nx * nk::NDIR * nk::NG, f);
    std::fwrite(v.trlcff1, sizeof(double), nx * nk::NDIR * nk::NG, f);
    std::fwrite(v.trlcff2, sizeof(double), nx * nk::NDIR * nk::NG, f);
    std::fwrite(v.matMs, sizeof(double), nx * nk::NG2, f);
    std::fwrite(v.matMf, sizeof(double), nx * nk::NG2, f);
    std::fwrite(v.matM, sizeof(double), nx * nk::NG2, f);
    std::fwrite(v.matMI, sizeof(double), nx * nk::NG2, f);
    std::fwrite(v.mu, sizeof(double), nx * nk::NDIR * nk::NG2, f);
    std::fwrite(v.tau, sizeof(double), nx * nk::NDIR * nk::NG2, f);
    std::fwrite(v.dsncff2, sizeof(double), nx * nk::NDIR * nk::NG, f);
    std::fwrite(v.dsncff4, sizeof(double), nx * nk::NDIR * nk::NG, f);
    std::fwrite(v.dsncff6, sizeof(double), nx * nk::NDIR * nk::NG, f);
    std::fwrite(v.jnet, sizeof(double), ns * nk::NG, f);
    std::fwrite(v.phis, sizeof(double), ns * nk::NG, f);
    std::fclose(f);
}

void Nodal::drive() {
    static const char* dump_path = std::getenv("RASBERY_NODAL_DUMP");
    static const int   dump_call = [] {
        const char* v = std::getenv("RASBERY_NODAL_DUMP_CALL");
        return v ? std::atoi(v) : 5;
    }();
    static std::atomic<int>  drive_calls{0};
    static std::atomic<bool> dump_done{false};
    std::vector<double>      jnet_snapshot;
    bool                     dump_this = false;
    if (dump_path != nullptr && !dump_done.load(std::memory_order_relaxed)) {
        const int call = drive_calls.fetch_add(1) + 1;
        if (call == dump_call && !dump_done.exchange(true)) {
            dump_this = true;
            jnet_snapshot.assign(_jnet, _jnet + static_cast<std::size_t>(_nsurf) * _ng);
        }
    }
    driveBody();
    if (dump_this) {
        nodal::NodalView v = MakeView();
        nodalDumpState(dump_path, v, jnet_snapshot.data());
        std::fprintf(stderr, "[RASBERY][NODAL][DUMP] call=%d -> %s\n", dump_call,
                     dump_path);
    }
}

void Nodal::driveBody() {
    // Each per-node / per-surface routine is independent (writes its own node/surface data; reads
    // neighbours only across the implicit barrier between phases). One parallel region with a
    // barrier per phase amortizes fork/join; results are bit-identical (no cross-node reduction).
#pragma omp parallel if (_nxyz > rasbery_omp_gate)
    {
#pragma omp for schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            updateConstant(lk);
#pragma omp for schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            caltrlcff0(lk);
#pragma omp for schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            caltrlcff12(lk);
#pragma omp for schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            updateMatrix(lk);
#pragma omp for schedule(static)
        for (int lk = 0; lk < _nxyz; ++lk)
            calculateEven(lk);
#pragma omp for schedule(static)
        for (int ls = 0; ls < _nsurf; ++ls)
            calculateJnet(ls);
    }
}
