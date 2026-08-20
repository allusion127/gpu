#pragma once

#include "Geometry.h"
#include "XSSet.h"
#include "pch.h"

#define m011  0.666666667
#define m022  0.4
#define m033  0.285714286
#define m044  0.222222222
#define m220  6.
#define rm220 0.166666667
#define m240  20.
#define m231  10.
#define m242  14.

/**
 * @brief Nodal class for implementing SENM(FENM) Nodal method
 *
 */
namespace rasbery {
class Nodal {
private:
    /// @brief Geometry object
    Geometry& _g;

    /// @brief CrossSection object
    XSSet& xs;

    /// @brief the number of energy groups
    int _ng;

    /// @brief the number of energy groups squared
    int _ng2;

    /// @brief the number of nodes in 2d plane
    int _nxyz;

    /// @brief the number of surfaces in 2d plane
    int _nsurf;

    /// @brief the 0th order coefficient of transverse leakage
    double* _trlcff0;

    /// @brief the 1st order coefficient of transverse leakage
    double* _trlcff1;

    /// @brief the 2nd order coefficient of transverse leakage
    double* _trlcff2;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _eta1;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _eta2;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _mu;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _tau;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _m260;
    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _m251;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _m253;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _m262;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _m264;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _diagDI;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _diagD;

    /// Last material inputs used by updateConstant().  Nodal::drive() is called
    /// repeatedly while the flux converges, but these coefficients only change
    /// when removal/diffusion XS changes.  Keeping a per-node bit-exact shadow
    /// avoids recomputing exp/sinh/cosh-derived constants on unchanged outers.
    double* _constant_xsrf;
    double* _constant_xsdf;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _matM;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _matMI;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _matMs;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _matMf;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _dsncff2;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _dsncff4;

    /// @brief variable in FENM method (see the paper : https://doi.org/10.1080/18811248.2008.9711467)
    double* _dsncff6;

    /// @brief net current at left or right of each surface
    double* _jnet;

    /// @brief flux in each node
    double* _flux;

    /// @brief surface flux at left or right of each surface
    double* _phis;

    /// @brief the reciprocal of eigenvalue
    double _reigv;

public:
    /// @brief the maximum number of iteration for updating the fission source shape
    int nmaxswp;

public:
    Nodal(Geometry& g, XSSet& xs);
    virtual ~Nodal();

    /// @brief reset the nodal calculation
    /// @param reigv the reciprocal of eigenvalue
    /// @param jnet the net current
    /// @param phif the node flux
    /// @param phis the surface flux
    void reset(const double& reigv, double* jnet, double* phif, double* phis);

    /// @brief run the nodal calculation and update the net current
    void drive();

    /// @brief update the constant in the nodal calculation
    /// @param lk the node index
    void updateConstant(const int& lk);

    /// @brief update the matrix in the nodal calculation
    /// @param lk the node index
    void updateMatrix(const int& lk);

    /// @brief calculate the transverse leakage in the nodal calculation
    /// @param avgtrl3 the average transverse leakage
    /// @param hmesh3 the mesh size
    /// @param trlcff1 the 1st order transverse leakage coefficient
    /// @param trlcff2 the 2nd order transverse leakage coefficient
    void trlcffbyintg(double* avgtrl3, double* hmesh3, double& trlcff1, double& trlcff2);

    /// @brief calculate the 0th order transverse leakage coefficient
    /// @param lk the node index
    void caltrlcff0(const int& lk);

    /// @brief calculate the 1st and 2nd order transverse leakage coefficient
    /// @param lk the node index
    void caltrlcff12(const int& lk);

    /// @brief calculate the even order coefficient of 1d nodal function
    /// @param lk the node index
    void calculateEven(const int& lk);

    /// @brief calculate the odd order coefficient of 1d nodal function and update the net current
    /// @param ls the surface index
    void calculateJnet(const int& ls);

    /// @brief update the net current at the boundary surface
    /// @param ls the surface index
    /// @param lr the left or right surface
    /// @param alb the albedo boundary condition
    void calculateJnet1n(const int& ls, const int& lr, const double& alb);

    /// @brief update the net current at the interior surface
    void calculateJnet2n(const int& ls);

    /// @brief the surface flux
    /// @param ig the energy group index
    /// @param lks the surface index
    inline double& phis(const int& ig, const int& lks) {
        return _phis[(lks)*_ng + ig];
    };

    double* trlcff0() {
        return _trlcff0;
    }

    [[nodiscard]] AxialTransverseLeakageView axialTransverseLeakage() const {
        return {_trlcff0, _trlcff1, _trlcff2};
    }

    inline double tl(const int& ig, const int& lkd) {
        return _trlcff0[(lkd)*_ng + ig] + _trlcff1[(lkd)*_ng + ig] + _trlcff2[(lkd)*_ng + ig];
    }
};
} // namespace rasbery
