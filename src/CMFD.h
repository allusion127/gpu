#pragma once
#include "Geometry.h"
#include "XSSet.h"
#include "pch.h"

namespace rasbery {
/**
 * @brief Class for implementing Coarse Mesh Finite Difference Method (CMFD)
 */
class CMFD {
protected:
    /// @brief Geometry object
    Geometry& _g;

    /// @brief CrossSection object
    XSSet& _x;

    /// @brief the number of CMFD iteration
    int _ncmfd;

    /// @brief d_tilde for CMFD
    double* _dtil;

    /// @brief d_hat for CMFD
    double* _dhat;

    /// @brief diagonal matrix for CMFD
    double* _diag;

    /// @brief coupling coefficient for CMFD
    double* _cc;

    /// @brief source term for CMFD
    double* _src;

    /// @brief fission source term for CMFD
    double* _psi;

    /// @brief criterion for convergence
    double _epsl2;

public:
    CMFD(Geometry& g, XSSet& x);

    virtual ~CMFD();

    /// @brief  get the geometry object
    Geometry& g() {
        return _g;
    };

    /// @brief get the cross section object
    XSSet& x() {
        return _x;
    };

    /// @brief update the d_tilde for CMFD
    virtual void upddtil() = 0;

    /// @brief update the d_hat for CMFD
    virtual void upddhat(double* flux, double* jnet) = 0;

    /// @brief set the linear system for CMFD
    /// @param eigv the eigenvalue
    virtual void setls(const double& eigv) = 0;

    /// @brief update the net current for CMFD
    /// @param flux the flux
    /// @param jnet the net current
    virtual void updjnet(double* flux, double* jnet) = 0;

    /// @brief update the fission source term for CMFD
    /// @param flux the flux
    virtual void updpsi(const double* flux) = 0;

    /// @brief drive the CMFD calculation
    /// @param eigv the eigenvalue
    /// @param flux the flux
    /// @param errl2 the L2 norm of the error
    virtual void drive(double& eigv, double* flux, double& errl2) = 0;

    /// @brief update fission source term for each node
    /// @param l the node index
    /// @param flux the flux
    void updpsi(const int& l, const double* flux);

    /// @brief update d_tilde for each surface
    /// @param ls th surface index
    void upddtil(const int& ls);

    /// @brief update d_hat for each surface
    /// @param ls the surface index
    /// @param flux the flux
    /// @param jnet the net current
    void upddhat(const int& ls, double* flux, double* jnet);

    /// @brief setup the linear system element for each node for CMFD calculation
    /// @param l the node index
    void setls(const int& l);

    /// @brief set the number of CMFD iteration
    void setNcmfd(int ncmfd);

    /// @brief set the criterion for convergence
    void setEpsl2(double epsl2);

    /// @brief reset nonlinear current correction after an imposed state change
    void resetDhat();

    /// @brief d_tilde for CMFD
    /// @param ig group index
    /// @param ls surface index
    double& dtil(const int& ig, const int& ls) {
        return _dtil[ls * _g.ng() + ig];
    };

    /// @brief d_hat for CMFD
    /// @param ig group index
    /// @param ls surface index
    double& dhat(const int& ig, const int& ls) {
        return _dhat[ls * _g.ng() + ig];
    };

    /// @brief diagonal element of CMFD
    /// @param igs from-group index
    /// @param ige to-group index
    /// @param l the node index
    double& diag(const int& igs, const int& ige, const int& l) {
        return _diag[l * _g.ng2() + ige * _g.ng() + igs];
    };

    /// @brief coupling coefficient for CMFD
    /// @param lr left or right
    /// @param idir X,Y or Z direction
    /// @param ig group index
    /// @param l the node index
    double& cc(const int& lr, const int& idir, const int& ig, const int& l) {
        return _cc[l * _g.ng() * NDIRMAX * LR + ig * NDIRMAX * LR + idir * LR + lr];
    };

    /// @brief source term for CMFD
    /// @param ig group index
    /// @param l the node index
    double& src(const int& ig, const int& l) {
        return _src[l * _g.ng() + ig];
    };

    /// @brief fission source term for CMFD
    /// @param l the node index
    double& psi(const int& l) {
        return _psi[l];
    };

    /// @brief calculate Axb calculation for each group and node
    /// @param ig group index
    /// @param l the node index
    /// @param flux the flux
    /// @return the element of Axb calculation
    double axb(const int& ig, const int& l, const double* flux) {

        double ab = 0.0;
        for (int igs = 0; igs < _g.ng(); ++igs) {
            ab += diag(igs, ig, l) * flux[l * _g.ng() + igs];
        }

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g.neib(lr, idir, l);
                if (ln != -1)
                    ab += cc(lr, idir, ig, l) * flux[ln * _g.ng() + ig];
            }
        }

        return ab;
    };

    /// @brief update net current at each surface
    /// @param ls the surface index
    /// @param flux the flux
    /// @param jnet the net current to be updated
    void updjnet(const int& ls, const double* flux, double* jnet) {
        int ll    = _g.lklr(LEFT, ls);
        int lr    = _g.lklr(RIGHT, ls);
        int idirl = _g.idirlr(LEFT, ls);
        int idirr = _g.idirlr(RIGHT, ls);

        for (int ig = 0; ig < _g.ng(); ig++) {
            if (ll < 0) {
                jnet[ls * _g.ng() + ig] = -(dtil(ig, ls) + dhat(ig, ls)) * flux[lr * _g.ng() + ig];
            } else if (lr < 0) {
                jnet[ls * _g.ng() + ig] = (dtil(ig, ls) - dhat(ig, ls)) * flux[ll * _g.ng() + ig];
            } else {
                jnet[ls * _g.ng() + ig] = -dtil(ig, ls) * (flux[lr * _g.ng() + ig] - flux[ll * _g.ng() + ig]) - dhat(ig, ls) * (flux[lr * _g.ng() + ig] + flux[ll * _g.ng() + ig]);
            }
        }
    }
};
} // namespace rasbery
