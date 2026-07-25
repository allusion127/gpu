#pragma once

#include "Geometry.h"
#include "milk.h"

namespace rasbery {

/**
 * @brief BICGStab solver class
 * see : https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
 *
 */
class BICGSolver {
private:
    /// @brief Geometry object
    Geometry& _g;

    /// @brief alpha in BICGStab
    double _calpha;

    /// @brief beta in BICGStab
    double _cbeta;

    /// @brief rho in BICGStab
    double _crho;

    /// @brief omega in BICGStab
    double _comega;

    /// y in BICGStab
    milk::Vector<double> _vy;

    /// z in BICGStab
    milk::Vector<double> _vz;

    /// @brief r in BICGStab
    milk::Vector<double> _vr;

    /// @brief r0 in BICGStab
    milk::Vector<double> _vr0;

    /// @brief p in BICGStab
    milk::Vector<double> _vp;

    /// @brief v in BICGStab
    milk::Vector<double> _vv;

    /// @brief s in BICGStab
    milk::Vector<double> _vs;

    /// @brief t in BICGStab
    milk::Vector<double> _vt;

    /// @brief SSOR: inverse of 2x2 diagonal blocks [nxyz * ng2]
    milk::Vector<double> _dinv;

    /// @brief SSOR: forward sweep workspace [nxyz * ng]
    milk::Vector<double> _ssor_tmp;

    /// @brief Pointer to current diagonal (saved from facilu)
    double* _diag_ptr;

public:
    /// @brief Construct a new BICGStab object
    BICGSolver(Geometry& g);

    /// @brief  Destroy the BICGStab object
    virtual ~BICGSolver();

    /// @brief reset the BICGStab calculation
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    /// @param src the source term
    /// @param r20 the initial residual
    void reset(double* diag, double* cc, double* phi, double* src, double& r20);

    /// @brief reset the BICGStab calculation for each group and node
    /// @param ig group index
    /// @param l node index
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    /// @param src the source term
    /// @return the initial residual
    double reset(const int& ig, const int& l, double* diag, double* cc, double* phi, double* src);

    /// @brief Apply SSOR preconditioner: sol = M^{-1} * b
    /// @param cc the coupling coefficient
    /// @param b the right-hand side
    /// @param x the solution
    void minv(double* cc, double* b, double* x);

    /// @brief Compute SSOR diagonal block inverses
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    void facilu(double* diag, double* cc);

    /// @brief solve the BICGStab calculation
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param r20 the initial residual
    /// @param phi the flux
    /// @param r2 the final residual
    void solve(double* diag, double* cc, double& r20, double* phi, double& r2);

    /// @brief calculate Axb in the BICGStab calculation
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    /// @param aphi the result of Axb
    void axb(double* diag, double* cc, double* phi, double* aphi);


    /// @brief calculate an element of Axb calculation in the BICGStab calculation
    /// @param ig group index
    /// @param l node index
    /// @param diag the diagonal matrix
    /// @param cc the coupling coefficient
    /// @param phi the flux
    double axb(const int& ig, const int& l, double* diag, double* cc, double* phi);
};
}