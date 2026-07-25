#pragma once

#include "Model.h"

namespace Chiffon {

// Boundary condition types for reflector interface coupling
enum BOUNDARY {
    REFLECTIVE,
    VACUUM,
    FLUXZERO,
    NODE
};

// Applies assembly discontinuity factors (ADF) to fuel and reflector cross-sections.
// ADF corrects for the flux discontinuity at node interfaces in nodal diffusion.
class ReflectorSolver {
public:
    ReflectorSolver() {
    }

    // Apply ADF from surface DF data to all depletion points in a fuel model
    void ApplyDF(Model& fuel) {
        for (int i = 0; i < fuel.NumPoints(); i++) {
            DepletionPoint& fuelNode = fuel(i);
            fuelNode._xs.ApplyDF(fuelNode._sdfa[0], 0);
            fuelNode._xs.ApplyDF(fuelNode._sdfa[4], 1);
            fuelNode._xs.CalcLmpx(fuelNode._iden);
        }
    }

    // Apply ADF ratio (reflector/fuel) to radial reflector model
    void ApplyDF(const Model& fuel, Model& refl, Model* lNeighbor, Model* rNeighbor, BOUNDARY lBoundaryCond,
                 BOUNDARY rBoundaryCond) {
        for (int i = 0; i < fuel.NumPoints(); i++) {
            const DepletionPoint& fuelNode = fuel(i);
            DepletionPoint&       reflNode = refl(i);
            // reflNode._xs.ApplyDF(reflNode._sdfa[0 * 4 + WEST] / fuelNode._sdfa[0 * 4 + EAST], 0);
            // reflNode._xs.ApplyDF(reflNode._sdfa[1 * 4 + WEST] / fuelNode._sdfa[1 * 4 + EAST], 1);
            reflNode._xs.CalcLmpx(reflNode._iden);
        }
    }

    // Apply ADF ratio for axial bottom reflector (reversed interface direction)
    void ApplyDF2(Model& refl, Model& fuel, Model* lNeighbor, Model* rNeighbor, BOUNDARY lBoundaryCond,
                  BOUNDARY rBoundaryCond) {
        for (int i = 0; i < fuel.NumPoints(); i++) {
            const DepletionPoint& fuelNode = fuel(i);
            DepletionPoint&       reflNode = refl(i);
            // reflNode._xs.ApplyDF(reflNode._sdfa[0 * 4 + EAST] / fuelNode._sdfa[0 * 4 + WEST], 0);
            // reflNode._xs.ApplyDF(reflNode._sdfa[1 * 4 + EAST] / fuelNode._sdfa[1 * 4 + WEST], 1);
            reflNode._xs.CalcLmpx(reflNode._iden);
        }
    }
};
} // namespace Chiffon