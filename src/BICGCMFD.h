#pragma once
#include "BICGSolver.h"
#include "CMFD.h"
#include "Nodal.h"
#include <memory>
#include <vector>

namespace rasbery {
/**
 * @brief Class for implementing combination of BICGCMFD + Nodal method
 *
 */

class BICGCMFD : public CMFD {
protected:
    /// @brief BICG solver object
    std::unique_ptr<BICGSolver> _ls;

    /// @brief Nodal object
    std::unique_ptr<Nodal> _nodal;

    /// @brief total number of CMFD sweeps since the last resetIteration(); diagnostic only
    int iter;

    /// @brief Number of CMFD sweeps since the last resetIteration(), used ONLY to decide
    /// when the Wielandt extrapolation may take over from the Rayleigh-quotient warm-up.
    ///
    /// This used to be `iter` itself, which made a diagnostic counter silently steer the
    /// eigenvalue update: any caller that read, reset, or reused `iter` for reporting
    /// would have changed the Wielandt activation point. The two are now separate
    /// variables with the same reset point, so the numerical schedule is unchanged but
    /// no longer coupled to the reporting counter.
    int _wiel_sweep;

    /// @brief number of warm-up sweeps solved with the Rayleigh quotient before the
    /// Wielandt extrapolation is allowed to drive the eigenvalue
    static constexpr int WIELANDT_WARMUP_SWEEPS = 5;

    /// @brief the maximum number of bicg iteration (inner iteration)
    int _nmaxbicg;

    /// @brief the convergence criterion of bicg iteration (inner iteration)
    double _epsbicg;

    /// @brief the Wielandt shift value in CMFD acceleration
    double _eshift;

    /// @brief the unshifted diagonal matrix
    std::vector<double> _udiag;

public:
    BICGCMFD(Geometry& g, XSSet& x);

    ~BICGCMFD() override;

    [[nodiscard]] int innerIterations() const { return iter; }

    /// @brief set the iteration limit for BICGCMFD
    /// @param maxls maximum number of iteration in BICG calculation
    /// @param epsls convergence criterion in BICG calculation
    void setIterLim(int maxls, double epsls);

    /// @brief update d_tilde for CMFD
    void upddtil() override;

    /// @brief update d_hat for CMFD
    void upddhat(double* flux, double* jnet) override;

    /// @brief setup the linear system for CMFD
    /// @param eigv the eigenvalue
    void setls(const double& eigv) override;

    /// @brief update the net current for CMFD
    /// @param flux the flux
    /// @param jnet the net current
    void updjnet(double* flux, double* jnet) override;

    /// @brief update the fission source term for CMFD
    /// @param flux the flux
    void updpsi(const double* flux) override;

    /// @brief drive the CMFD calculation
    /// @param eigv the eigenvalue
    /// @param flux the flux
    /// @param errl2 the L2 norm of the error
    void drive(double& eigv, double* flux, double& errl2) override;


    /// @brief reset the iteration count
    void resetIteration();

    /// @brief set the Wielandt shift value
    void setEshift(double eshift0);

    /// @brief update linear system of CMFD with Wielandt shift value
    /// @param reigvs the reciprocal of shifted eigenvalue
    void updls(const double& reigvs);

    /// @brief setup the element of linear system for CMFD
    /// @param l the node index
    void setls(const int& l);

    /// @brief update the element of the linear system for CMFD
    /// @param l the node index
    /// @param reigvs the reciprocal of shifted eigenvalue
    void updls(const int& l, const double& reigvs);



    /// @brief perform the Wielandt shift to accelerate CMFD calculation
    /// @param icy the iteration count
    /// @param flux the flux
    /// @param reigvs the reciprocal of shifted eigenvalue
    /// @param eigv the eigenvalue
    /// @param reigv the reciprocal of eigenvalue
    /// @param errl2 the L2 norm of the error
    void wiel(const int& icy, const double* flux, double& reigvs, double& eigv, double& reigv, double& errl2);

    /// @brief the unshifted diagonal matrix
    /// @param igs from-group index
    /// @param ige to-group index
    /// @param l the node index
    double& udiag(const int& igs, const int& ige, const int& l) {
        return _udiag[l * _g.ng2() + ige * _g.ng() + igs];
    };

    /// @brief the Wielandt shift value
    double& eshift() {
        return _eshift;
    };
};
} // namespace rasbery
