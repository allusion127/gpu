#pragma once
#include "Geometry.h"
#include "XSSet.h"
#include "pch.h"

#include <vector>

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

    // Geometry is immutable after a Driver is constructed.  CMFD's hottest
    // host loops used to re-index Geometry for every surface/node of every
    // outer iteration.  Cache only immutable topology/metrics here; stateful
    // fluxes, currents, cross sections and CMFD coefficients remain per-call.
    std::vector<int>    _surface_node;     ///< [surface][left/right]
    std::vector<int>    _surface_dir;      ///< [surface][left/right]
    std::vector<int>    _node_surface;     ///< [node][direction][left/right]
    std::vector<int>    _node_neighbor;    ///< [node][direction][left/right]
    std::vector<double> _node_hmesh;       ///< [node][direction]
    std::vector<double> _node_face_area;   ///< [node][direction]
    std::vector<double> _node_volume;      ///< [node]
    std::vector<double> _boundary_albedo;  ///< [direction][left/right]

    [[nodiscard]] inline int cachedSurfaceNode(const int lr, const int ls) const {
        return _surface_node[static_cast<size_t>(ls) * LR + lr];
    }

    [[nodiscard]] inline int cachedSurfaceDirection(const int lr, const int ls) const {
        return _surface_dir[static_cast<size_t>(ls) * LR + lr];
    }

    [[nodiscard]] inline int cachedNodeSurface(const int lr, const int idir, const int l) const {
        return _node_surface[(static_cast<size_t>(l) * NDIRMAX + idir) * LR + lr];
    }

    [[nodiscard]] inline int cachedNodeNeighbor(const int lr, const int idir, const int l) const {
        return _node_neighbor[(static_cast<size_t>(l) * NDIRMAX + idir) * LR + lr];
    }

    [[nodiscard]] inline double cachedNodeHmesh(const int idir, const int l) const {
        return _node_hmesh[static_cast<size_t>(l) * NDIRMAX + idir];
    }

    [[nodiscard]] inline double cachedNodeFaceArea(const int idir, const int l) const {
        return _node_face_area[static_cast<size_t>(l) * NDIRMAX + idir];
    }

    [[nodiscard]] inline double cachedNodeVolume(const int l) const {
        return _node_volume[static_cast<size_t>(l)];
    }

    [[nodiscard]] inline double cachedBoundaryAlbedo(const int lr, const int idir) const {
        return _boundary_albedo[static_cast<size_t>(idir) * LR + lr];
    }

    /// @brief criterion for convergence
    double _epsl2;

    /// @brief number of dhat updates skipped because |fsum| fell below the floor
    long long _dhat_fsum_guard = 0;

    /// @brief number of dhat updates damped by the |dhat| <= |dtil| clamp
    long long _dhat_clamped = 0;

    /// @brief total number of dhat updates attempted
    long long _dhat_total = 0;

    /// @brief largest pre-clamp |dhat|/|dtil| ratio observed
    double _dhat_ratio_max = 0.0;

    /// @brief whether the |dhat| <= |dtil| envelope is enforced (RASBERY_DHAT_CLAMP=1)
    /// or only counted. Off by default: see the rationale in CMFD::upddhat.
    bool _dhat_clamp_enabled = false;

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
    virtual void upddhat(const double* flux, double* jnet) = 0;

    /// @brief set the linear system for CMFD
    /// @param eigv the eigenvalue
    virtual void setls(const double& eigv) = 0;

    /// @brief update the net current for CMFD
    /// @param flux the flux
    /// @param jnet the net current
    virtual void updjnet(const double* flux, double* jnet) = 0;

    /// @brief update the fission source term for CMFD
    /// @param flux the flux
    virtual void updpsi(const double* flux) = 0;

    /// @brief drive the CMFD calculation
    /// @param eigv the eigenvalue
    /// @param flux the flux
    /// @param errl2 the L2 norm of the error
    virtual void drive(double& eigv, double* flux, double& errl2) = 0;

    /// @brief update fission source term for each node
    void updpsi(const int& l, const double* flux);

    /// @brief update d_tilde for each surface
    void upddtil(const int& ls);

    /// @brief update d_hat for each surface
    void upddhat(const int& ls, const double* flux, double* jnet);

    /// @brief setup the linear system element for each node for CMFD calculation
    void setls(const int& l);

    /// @brief set the number of CMFD iteration
    void setNcmfd(int ncmfd);

    /// @brief set the criterion for convergence
    void setEpsl2(double epsl2);

    // -----------------------------------------------------------------
    // Base pointers of the immutable topology cache (Rev.7.1 Task 9)
    // -----------------------------------------------------------------
    //
    // cmfd::CmfdGeometryView (CmfdOuterKernel.h:190) is the device twin of
    // exactly these five arrays, in exactly this layout, and the arena
    // uploads them once per run.  Handing out the BASE POINTER rather than
    // adding a copy accessor matters: the device view is byte-shareable with
    // the host cache only while both sides address the same layout, and a
    // copy would be a second layout nobody would notice drifting.
    //
    // They are const and read-only.  The cache is built once per Driver
    // (CMFD.cpp:31-73) and is immutable for the run, which is the whole
    // reason it exists -- so exposing it cannot create a second writer.
    [[nodiscard]] const int* surfaceNodeData() const { return _surface_node.data(); }
    [[nodiscard]] const int* surfaceDirData() const { return _surface_dir.data(); }
    [[nodiscard]] const double* nodeHmeshData() const { return _node_hmesh.data(); }
    [[nodiscard]] const double* nodeVolumeData() const { return _node_volume.data(); }
    /// RASBERY_DHAT_CLAMP as this CMFD resolved it (CMFD.cpp:76).  The device
    /// upddhat body takes the same flag, and reading it from here rather than
    /// re-parsing the environment is what stops the two arms disagreeing about
    /// a clamp that changes the answer.
    [[nodiscard]] bool dhatClampEnabled() const { return _dhat_clamp_enabled; }

    /// Rev.7.1 Task 9 link 2: the HOST twins of the two arrays the device outer
    /// segment writes.
    ///
    /// THE SEGMENT HAS TO KEEP THEM CURRENT, and this is why.  drive() does not
    /// always take the resident sweep: the Wielandt warm-up runs on the host
    /// (BICGCMFD.cpp:558-565), and the device sweep can decline.  The host loop
    /// reads _dhat through setls/axb and _psi through wiel, so a segment that
    /// wrote only the device copies would hand the host path arrays one outer
    /// stale -- silently, and only on the decks that warm up differently.
    [[nodiscard]] double* dhatData() { return _dhat; }
    [[nodiscard]] double* psiData() { return _psi; }
    /// The CMFD operator's d-tilde.  The segment READS it (updjnet, upddhat)
    /// and never writes it -- upddtil is host-only -- but the device copy it
    /// reads is refreshed by the sweep only on the device-assembly path, so
    /// the segment has to sync it like the flux and xsnf.  See
    /// OuterSegmentResidency::host_dtil.
    [[nodiscard]] const double* dtilData() const { return _dtil; }

    [[nodiscard]] const double* boundaryAlbedoData() const {
        return _boundary_albedo.data();
    }

    /// @brief reset nonlinear current correction after an imposed state change
    void resetDhat();

    /// @brief emit a one-line summary of the dhat guard/clamp statistics to stderr
    void reportDhatGuardStats(const char* tag = "") const;

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
                const int ln = cachedNodeNeighbor(lr, idir, l);
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
        const int ll = cachedSurfaceNode(LEFT, ls);
        const int lr = cachedSurfaceNode(RIGHT, ls);

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
