#pragma once

#include "Geometry.h"
#include "NodalKernel.h"
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
    const double* _flux; // -> Geometry::Phif(), READ-ONLY

    /// @brief surface flux at left or right of each surface
    double* _phis;

    /// @brief the reciprocal of eigenvalue
    double _reigv;

    unsigned long long _const_generation = 1;

    // ---------------------------------------------------------------------
    // Rev.7.1 W3 item 1: THE CONSTANTS GATE
    // ---------------------------------------------------------------------
    //
    // WHAT IT REMOVES.  Every drive re-ran the whole updateConstant sweep --
    // nxyz*ng cache comparisons -- and on the device outer segment that sweep is
    // HOST ARITHMETIC inside a body that is otherwise entirely enqueued.  The
    // [RASBERY][OUTER_GPU] receipt said so: host_body_calls.nodal_constants was
    // equal to device_outers (12017 of 12017 on kngr_238), and it was the only
    // non-zero body left besides the out-of-segment upddtil.
    //
    // WHY THE SWEEP CAN BE SKIPPED RATHER THAN MOVED.  updateConstant's entire
    // input is xs.xsrf, xs.xsdf and Geometry::hmesh (immutable after stand-up).
    // When none of the three moved, every node takes the node-scoped early-out,
    // the function returns false everywhere, _const_generation does not advance
    // and NOT ONE COEFFICIENT IS WRITTEN.  Skipping it is therefore bit-exact by
    // construction, which porting it to the device would NOT be: the coefficient
    // body is class N1 (CUDA exp differs from glibc by 1 ulp on 3.34% of
    // arguments, NodalConstantKernel.h), so a device producer changes the
    // trajectory on every outer that genuinely recomputes.
    //
    // THE GENERATION IT WATCHES.  XSSet::macroXsGeneration(), NOT
    // hoststateGeneration() -- see XSSet.h::_macroxs_generation.  0 means "never
    // built", which no live generation can equal (XSSet starts it at 1), so the
    // first drive always sweeps.
    //
    // RASBERY_NODAL_CONST_VERIFY=1 runs the sweep even when the gate says skip
    // and complains if any node reports a change, so the gate's premise is
    // measurable on a real deck rather than argued.
    unsigned long long _constants_macroxs_generation = 0;

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
    void reset(const double& reigv, double* jnet, const double* phif, double* phis);

    /// @brief run the nodal calculation and update the net current
    void drive();

    /// Pointer view of this instance's nodal state for the shared kernel body
    /// (host pointers; the device backend repoints them).
    nodal::NodalView MakeView();

    /// Advances whenever updateConstant actually recomputed any node, so the
    /// device copy of the nine coefficient arrays re-uploads only then.
    unsigned long long const_generation() const { return _const_generation; }

    /// The six solve phases (capture wrapper lives in drive()).
    void driveBody();

    /// Rev.7.1 W3 item 1: run the updateConstant sweep only when the macro XS it
    /// reads can have moved.  Shared by BOTH drives, so the gate has exactly one
    /// spelling.  See _constants_macroxs_generation.
    void updateConstantsIfMoved();

    /// Rev.7.1 Task 18-lite: would TryDriveGpu take the device path right now?
    ///
    /// TryDriveGpu's OWN refusal test, asked without running anything -- the
    /// same relationship BICGCMFD::canEnqueueDrive() has to drive().  The device
    /// outer segment needs the answer BEFORE the drive, because it decides
    /// whether the jnet bridge around the nodal hook can be dropped: with the
    /// bridge gone, a drive that falls back to the CPU body reads a
    /// Geometry::Jnet the device stopped sending home.
    ///
    /// A PREDICATE AND NOT A COPY OF ONE.  TryDriveGpu calls this, so the two
    /// cannot drift; a second spelling of `does this deck have a fractional
    /// rod` is a second chance to answer it differently.
    [[nodiscard]] bool DeviceDriveEligible() const;

    /// Device arm behind RASBERY_GPU_NODAL; false = run the CPU body.
    bool TryDriveGpu();

    /// @brief update the constant in the nodal calculation
    /// @param lk the node index
    bool updateConstant(const int& lk);

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

/// How many times RASBERY_NODAL_CONST_VERIFY caught the constants gate skipping
/// a sweep that had work to do.  Zero on every gated run so far; printed in the
/// receipt so "verify found nothing" and "verify was off" are different numbers.
[[nodiscard]] std::uint64_t nodalConstantGateViolations();

} // namespace rasbery
