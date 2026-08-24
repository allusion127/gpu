#pragma once

// Pure coefficient body for Nodal::updateConstant.
//
// This is deliberately host-only for production today. std::exp/std::sqrt
// must keep the same libm and contraction choices as the historical CPU body;
// a future CUDA arm needs its own replay/mining gate before it may call this
// arithmetic on the device. Keeping the body isolated gives that experiment
// one testable boundary without changing the current numerical contract.

#include <cmath>

namespace rasbery::nodal {

struct NodalConstantCoefficients {
    double eta1;
    double eta2;
    double m260;
    double m251;
    double m253;
    double m262;
    double m264;
    double diagD;
    double diagDI;
};

[[nodiscard]] inline NodalConstantCoefficients
nodalConstantCoefficients(double xsrf, double xsdf, double hmesh) {
    // Keep the statement order and association of Nodal::updateConstant.
    double kp2    = xsrf * hmesh * hmesh / (4 * xsdf);
    double kp     = std::sqrt(kp2);
    double kp3    = kp2 * kp;
    double kp4    = kp2 * kp2;
    double rkp    = 1 / kp;
    double rkp2   = rkp * rkp;
    double rkp3   = rkp2 * rkp;
    double rkp4   = rkp2 * rkp2;
    double rkp5   = rkp2 * rkp3;
    double ekp    = std::exp(kp);
    double iekp   = 1.0 / ekp;
    double sinhkp = 0.5 * (ekp - iekp);
    double coshkp = 0.5 * (ekp + iekp);

    double bfcff0 = -sinhkp * rkp;
    double bfcff2 = -5 * (-3 * kp * coshkp + 3 * sinhkp + kp2 * sinhkp) * rkp3;
    double bfcff4 =
        -9. * (-105 * kp * coshkp - 10 * kp3 * coshkp + 105 * sinhkp +
               45 * kp2 * sinhkp + kp4 * sinhkp) * rkp5;
    double bfcff1 = -3 * (kp * coshkp - sinhkp) * rkp2;
    double bfcff3 =
        -7 * (15 * kp * coshkp + kp3 * coshkp - 15 * sinhkp -
              6 * kp2 * sinhkp) * rkp4;

    double oddtemp  = 1 / (sinhkp + bfcff1 + bfcff3);
    double eventemp = 1 / (coshkp + bfcff0 + bfcff2 + bfcff4);

    NodalConstantCoefficients out{};
    out.eta1 = (kp * coshkp + bfcff1 + 6 * bfcff3) * oddtemp;
    out.eta2 = (kp * sinhkp + 3 * bfcff2 + 10 * bfcff4) * eventemp;

    out.m260 = 2 * out.eta2;
    out.m251 = 2 * (kp * coshkp - sinhkp + 5 * bfcff3) * oddtemp;
    out.m253 = 2 * (kp * (15 + kp2) * coshkp - 3 * (5 + 2 * kp2) * sinhkp) *
               oddtemp * rkp2;
    out.m262 = 2 * (-3 * kp * coshkp + (3 + kp2) * sinhkp + 7 * kp * bfcff4) *
               eventemp * rkp;
    out.m264 = 2 * (-5 * kp * (21 + 2 * kp2) * coshkp +
                    (105 + 45 * kp2 + kp4) * sinhkp) * eventemp * rkp3;

    out.diagD  = 4 * xsdf / (hmesh * hmesh);
    out.diagDI = 1.0 / out.diagD;
    return out;
}

} // namespace rasbery::nodal
