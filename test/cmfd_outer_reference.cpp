// Verbatim CPU quotations of the four CMFD outer bodies, compiled ALONE.
//
// Nothing else may go in this file.  It is the reference the shared bodies in
// src/CmfdOuterKernel.h are scored against, and the whole point of giving it a
// translation unit of its own is that gcc must make its contraction choices for
// THIS code and nothing else -- see the note in cmfd_outer_reference.h for what
// happens otherwise.
//
// Each body is a statement-for-statement transcription of the source it names,
// with only the array plumbing changed (CMFD's members and accessors become
// explicit pointers with the same indexing).  Do not "simplify" any of it: the
// association, the branch order and the counter order are the contract.

#include "cmfd_outer_reference.h"

#include <cmath>

namespace cmfdref {

// --- CMFD::upddtil(ls), src/CMFD.cpp:103-124 ---------------------------------
void refUpdDtil(const Mesh& m, double* dtil_out) {
    for (int ls = 0; ls < m.nsurf; ++ls) {
        const int ll    = m.surface_node[ls * NLR + 0];
        const int lr    = m.surface_node[ls * NLR + 1];
        const int idirl = m.surface_dir[ls * NLR + 0];
        const int idirr = m.surface_dir[ls * NLR + 1];

        double betal, betar;

        for (int ig = 0; ig < m.ng; ig++) {
            if (ll < 0) {
                betal = m.boundary_albedo[idirl * NLR + 0] * 0.5;
            } else {
                betal = m.xsdf[ig * m.nxyz + ll] / m.node_hmesh[ll * NDIR + idirl];
            }
            if (lr < 0) {
                betar = m.boundary_albedo[idirr * NLR + 1] * 0.5;
            } else {
                betar = m.xsdf[ig * m.nxyz + lr] / m.node_hmesh[lr * NDIR + idirr];
            }
            dtil_out[ls * m.ng + ig] = 2 * betal * betar / (betal + betar);
        }
    }
}

// --- CMFD::updpsi(l, flux), src/CMFD.cpp:246-254 -----------------------------
void refUpdPsi(const Mesh& m, double* psi_out) {
    for (int l = 0; l < m.nxyz; ++l) {
        psi_out[l] = 0.0;

        for (int ig = 0; ig < m.ng; ig++) {
            psi_out[l] += m.flux[l * m.ng + ig] * m.xsnf[ig * m.nxyz + l];
        }
        psi_out[l] = psi_out[l] * m.node_volume[l];
    }
}

// --- CMFD::updjnet(ls, flux, jnet), src/CMFD.h:240-256 -----------------------
void refUpdJnet(const Mesh& m, double* jnet_out) {
    const double* flux = m.flux;
    for (int ls = 0; ls < m.nsurf; ++ls) {
        const int ll = m.surface_node[ls * NLR + 0];
        const int lr = m.surface_node[ls * NLR + 1];

        for (int ig = 0; ig < m.ng; ig++) {
            const double dtil = m.dtil[ls * m.ng + ig];
            const double dhat = m.dhat[ls * m.ng + ig];
            if (ll < 0) {
                jnet_out[ls * m.ng + ig] = -(dtil + dhat) * flux[lr * m.ng + ig];
            } else if (lr < 0) {
                jnet_out[ls * m.ng + ig] = (dtil - dhat) * flux[ll * m.ng + ig];
            } else {
                jnet_out[ls * m.ng + ig] =
                    -dtil * (flux[lr * m.ng + ig] - flux[ll * m.ng + ig]) -
                    dhat * (flux[lr * m.ng + ig] + flux[ll * m.ng + ig]);
            }
        }
    }
}

// --- CMFD::upddhat(ls, flux, jnet), src/CMFD.cpp:126-190 ---------------------
void refUpdDhat(const Mesh& m, bool clamp_enabled, double* dhat_out,
                DhatCounters* counters) {
    const double* flux = m.flux;
    const double* jnet = m.jnet;
    for (int ls = 0; ls < m.nsurf; ++ls) {
        const int ll = m.surface_node[ls * NLR + 0];
        const int lr = m.surface_node[ls * NLR + 1];

        for (int ig = 0; ig < m.ng; ig++) {
            double fdiff, fsum;
            if (ll < 0) {
                fdiff = flux[lr * m.ng + ig];
                fsum  = flux[lr * m.ng + ig];
            } else if (lr < 0) {
                fdiff = -flux[ll * m.ng + ig];
                fsum  = flux[ll * m.ng + ig];
            } else {
                fdiff = flux[lr * m.ng + ig] - flux[ll * m.ng + ig];
                fsum  = flux[lr * m.ng + ig] + flux[ll * m.ng + ig];
            }
            double jnet_fdm = -m.dtil[ls * m.ng + ig] * (fdiff);

            ++counters->total;
            const double dtl   = m.dtil[ls * m.ng + ig];
            const double floor = 1.0e-12 * std::max(1.0, std::abs(dtl));
            if (!(std::abs(fsum) > floor) || !std::isfinite(fsum)) {
                ++counters->fsum_guard;
                dhat_out[ls * m.ng + ig] = 0.0;
                continue;
            }

            double dh = (jnet_fdm - jnet[ls * m.ng + ig]) / fsum;
            if (!std::isfinite(dh)) {
                ++counters->fsum_guard;
                dhat_out[ls * m.ng + ig] = 0.0;
                continue;
            }

            const double cap = std::abs(dtl);
            if (cap > 0.0) {
                const double ratio = std::abs(dh) / cap;
                if (ratio > counters->ratio_max) counters->ratio_max = ratio;
                if (ratio > 1.0) {
                    ++counters->clamped;
                    if (clamp_enabled) dh = (dh > 0.0 ? cap : -cap);
                }
            }
            dhat_out[ls * m.ng + ig] = dh;
        }
    }
}

} // namespace cmfdref
