#pragma once
#include "pch.h"

#include <array>
#include <string>
#include <vector>

namespace rasbery {

/// Node-count threshold above which the per-node solver loops (Nodal, BiCGSTAB matvec, and the
/// node-reduction loops in XSSet/PPR) run with OpenMP. Below it the fork/join overhead dominates,
/// so they stay serial. Set once at startup from RASBERY_OMP_GATE (default 256); see main.cpp.
extern int rasbery_omp_gate;

/// Thread-count-independent chunking used by the deterministic node reductions.
///
/// `reduction(+:)` combines per-thread partials in an order OpenMP is free to
/// choose, so a sum over the same data changes in the last bits when
/// OMP_NUM_THREADS changes. These helpers instead cut the index range into a
/// FIXED number of chunks (a function of the range length only), let threads
/// compute the chunk partials in any order, and then add the partials back in
/// ascending chunk index. The result is bitwise reproducible for any thread
/// count, while keeping the parallelism.
constexpr int RASBERY_DET_CHUNK_TARGET = 256;

inline int rasbery_det_chunks(int n) {
    if (n <= 1) return 1;
    return (n < RASBERY_DET_CHUNK_TARGET) ? n : RASBERY_DET_CHUNK_TARGET;
}

/// First index of chunk `c` in a `nchunk`-way even split of [0, n).
inline int rasbery_det_chunk_begin(int n, int nchunk, int c) {
    if (c >= nchunk) return n;
    const long long nn = n;
    return static_cast<int>((nn * c) / nchunk);
}

/// @brief Raw input parameters extracted from the JSON deck for Geometry initialization.
struct GeometryInput {
    int                                             ng;
    int                                             nz;
    int                                             ndivxy;
    int                                             npins;
    double                                          hx, hy; // assembly pitch [cm]
    std::vector<double>                             hz;     // axial mesh heights (bottom→top) [cm]
    int                                             symang; // symmetry angle (90 or 360)
    bool                                            symopt; // mirror symmetry
    bool                                            symdiv; // center assembly divided
    std::array<double, 6>                           albedo; // W, E, N, S, B, T
    std::vector<std::vector<std::string>>           core;   // assembly layout (row-major)
    std::map<std::string, std::vector<std::string>> batch;  // assembly name → axial layer IDs (JSON order)
};

/// @brief 3D cartesian geometry with multi-level index maps for node handling.
///
/// Naming convention:
///   Count    : n + (x,y) + (a) + (f)        e.g. nxy, nxya
///   1D index : (i,j,k) + (s,e) + (a) + (f)  e.g. nxs, kbc
///   2D index : la (assembly), l (node)
///   3D index : lk (node)
///   C-style  : start index 0 (included), end index n (excluded)
class Geometry {

    // Private indexing helpers

    /// @brief 2D node-map index from (column, row)
    inline int idx2d(int i, int j) const { return j * _nx + i; }

    /// @brief 3D node index from 2D node index and axial plane
    inline int idx3d(int l2d, int k) const { return k * _nxy + l2d; }

    /// @brief 2D assembly-map index from (column, row)
    inline int idxAsm2d(int ia, int ja) const { return ja * _nxa + ia; }

    /// @brief Directional offset inside per-node arrays with NDIRMAX stride
    inline int idxDir(int lk, int dir) const { return lk * NDIRMAX + dir; }

    /// @brief Surface-map offset (node, direction, side)
    inline int idxSfc(int lk, int dir, int side) const { return (lk * NDIRMAX + dir) * LR + side; }

    /// @brief Neighbor offset in 2D NEWS array
    inline int idxNeib2d(int l, int dir) const { return l * NEWS + dir; }

    /// @brief Neighbor offset in 3D NEWSBT array
    inline int idxNeib3d(int lk, int dir) const { return lk * NEWSBT + dir; }

    /// @brief Assembly sub-node offset
    inline int idxAsmSub(int la, int li) const { return la * _ndivxy2 + li; }

// Basic core information
#pragma region basic core information

private:
    int _ng;      // energy groups
    int _nx;      // nodes in x
    int _ny;      // nodes in y
    int _nz;      // nodes in z
    int _ndivxy;  // sub-divisions per assembly side
    int _ndivxy2; // sub-divisions per assembly (ndivxy^2)
    int _npins;   // pins per assembly side
    int _ncellxy; // cells per assembly
    int _ng2;     // ng * ng
    int _nxy;     // nodes in 2D plane
    int _nxyz;    // nodes in 3D
    int _ngxy;    // nxy * ng
    int _ngxyz;   // nxyz * ng
    int _kbc;     // fuel-start axial index
    int _kec;     // fuel-end axial index

    double _pressure           = 15.5;  // system pressure [MPa]
    double _inlet_temp         = 580.0; // coolant inlet temperature [K]
    double _outlet_temp        = 600.0; // coolant outlet temperature [K]
    double _mass_flow_rate     = 1.0;   // coolant mass flux [kg/s/m^2]
    double _rated_power        = 1.0;   // rated thermal power [MW]
    double _fuel_temp_rise_scale = 1.0; // multiplier on tabulated Tfuel-Tcoolant
    bool   _use_mass_flow_rate = false; // use input flow instead of outlet-derived flow
    double _part;                       // geometry fraction (1.0 full, 0.25 quarter)
    double _hzcore;                     // active core axial height [cm]

    double* _albedo; // boundary albedo [LR * NDIRMAX]
    double* _hmesh;  // mesh size per direction [NDIRMAX * nxyz]
    double* _hz;     // axial mesh height [nz]
    double* _vol;    // node volume [nxyz]
    double* _vola;   // assembly volume [nxya * nz]

    // Node index maps
    int* _nxs;    // x-start per row [ny]
    int* _nxe;    // x-end per row [ny]
    int* _nys;    // y-start per column [nx]
    int* _nye;    // y-end per column [nx]
    int* _neibr;  // 2D Neighbor indices [nxy * NEWS]
    int* _neibrb; // 2D Neighbor indices with reflection BC [nxy * NEWS]
    int* _ijtol;  // (i,j) -> l [nx * ny]
    int* _neib;   // 3D Neighbor indices [nxyz * NEWSBT]

    // Surface maps
    int  _nsurf;   // total number of surfaces
    int* _lklr;    // node index on each side of a surface [nsurf * LR]
    int* _idirlr;  // direction on each side [nsurf * LR]
    int* _sgnlr;   // current sign on each side [nsurf * LR]
    int* _lktosfc; // node -> surface index [(nxyz * NDIRMAX) * LR]

    // Assembly index maps
    int  _nxa;     // assemblies in x
    int  _nya;     // assemblies in y
    int  _nxya;    // assemblies in 2D plane
    int  _nxyfa;   // fuel assemblies in 2D
    int  _ncellfa; // cells in a fuel assembly
    int* _nxsa;    // assembly x-start per row [nya]
    int* _nxea;    // assembly x-end per row [nya]
    int* _nysa;    // assembly y-start per column [nxa]
    int* _nyea;    // assembly y-end per column [nxa]
    int* _ijtola;  // (ia,ja) -> la [nxa * nya]
    int* _ltola;   // node -> assembly [nxy]
    int* _latol;   // assembly -> node [nxya * ndivxy2]
    int* _larot;   // assembly rotation map [nxya * ndivxy2]

    // Corner maps
    int  _ncorn; // corner points in 2D
    int* _ltolc; // node -> corner index [NEWS * nxy]
    int* _lctol; // corner -> node index [NEWS * ncorn]

    // Symmetry
    int  _symopt; // symmetry option
    int  _symang; // symmetry angle (90 / 360)
    bool _symdiv; // center assembly divided

    // Composition / form-function
    int*  _comps;   // composition index per node [nxyz]
    int   _ncomp;   // number of compositions
    bool* _is_fuel; // fuel flag per node [nxyz]

    // Raw layout stored so consumers (XSSet) can map models without re-receiving gin.
    std::vector<std::vector<std::string>>           _core;
    std::map<std::string, std::vector<std::string>> _batch;

public:
    /// @brief Number of energy groups
    inline int& ng() { return _ng; }

    /// @brief Energy groups squared
    inline int& ng2() { return _ng2; }

    /// @brief Nodes in x direction
    inline int& nx() { return _nx; }

    /// @brief Nodes in y direction
    inline int& ny() { return _ny; }

    /// @brief Nodes in z direction
    inline int& nz() { return _nz; }

    /// @brief Nodes in 2D plane
    inline int& nxy() { return _nxy; }

    /// @brief Nodes in 3D
    inline int& nxyz() { return _nxyz; }

    /// @brief nxy * ng
    inline int& ngxy() { return _ngxy; }

    /// @brief nxyz * ng
    inline int& ngxyz() { return _ngxyz; }

    /// @brief Divisions per assembly side
    inline int& ndivxy() { return _ndivxy; }

    /// @brief Cells per assembly
    inline int& ncellxy() { return _ncellxy; }

    /// @brief Cells per fuel assembly
    inline int& ncellfa() { return _ncellfa; }

    /// @brief Pins per assembly side
    inline int& npins() { return _npins; }

    /// @brief Fuel assemblies in 2D plane
    inline int& nxyfa() { return _nxyfa; }

    /// @brief Assemblies in 2D plane
    inline int& nxya() { return _nxya; }

    /// @brief Assemblies in x direction
    inline int& nxa() { return _nxa; }

    /// @brief Assemblies in y direction
    inline int& nya() { return _nya; }

    /// @brief Total 3D assemblies (nxya * nz)
    inline int nxyza() const { return _nxya * _nz; }

    /// @brief Number of surfaces
    inline int& nsurf() { return _nsurf; }

    /// @brief Symmetry option
    inline int& symopt() { return _symopt; }

    /// @brief Symmetry angle (90 or 360)
    inline int& symang() { return _symang; }

    /// @brief Center assembly divided
    inline bool symdiv() const { return _symdiv; }

    /// @brief Raw core layout (assembly names, row-major)
    [[nodiscard]] const std::vector<std::vector<std::string>>& core() const { return _core; }
    /// @brief Batch map: assembly name → axial layer model IDs
    [[nodiscard]] const std::map<std::string, std::vector<std::string>>& batch() const { return _batch; }

    /// @brief Fuel-start axial index
    inline int& kbc() { return _kbc; }

    /// @brief Fuel-end axial index
    inline int& kec() { return _kec; }

    /// @brief Number of corner points in 2D plane
    inline int& ncorn() { return _ncorn; }

    /// @brief Geometry fraction (1.0 full, 0.25 quarter)
    inline double& part() { return _part; }

    /// @brief Active core axial height [cm]
    inline double& hzcore() { return _hzcore; }

    /// @brief Number of compositions
    inline int& ncomp() { return _ncomp; }

    /// @brief System pressure [MPa]
    inline double&                     pressure() { return _pressure; }
    [[nodiscard]] inline const double& pressure() const { return _pressure; }

    /// @brief Coolant inlet temperature [K]
    inline double&                     inlet_temp() { return _inlet_temp; }
    [[nodiscard]] inline const double& inlet_temp() const { return _inlet_temp; }

    /// @brief Coolant outlet temperature [K]
    inline double&                     outlet_temp() { return _outlet_temp; }
    [[nodiscard]] inline const double& outlet_temp() const { return _outlet_temp; }

    /// @brief Coolant mass flux [kg/s/m^2]
    inline double&                     mass_flow_rate() { return _mass_flow_rate; }
    [[nodiscard]] inline const double& mass_flow_rate() const { return _mass_flow_rate; }

    /// @brief Whether TH should use the input mass flow rate directly
    inline bool&                     use_mass_flow_rate() { return _use_mass_flow_rate; }
    [[nodiscard]] inline const bool& use_mass_flow_rate() const { return _use_mass_flow_rate; }

    /// @brief Rated thermal power [MW]
    inline double&                     rated_power() { return _rated_power; }
    [[nodiscard]] inline const double& rated_power() const { return _rated_power; }

    /// @brief Scale applied to the tabulated fuel-to-coolant temperature rise
    inline double& fuel_temp_rise_scale() { return _fuel_temp_rise_scale; }
    [[nodiscard]] inline const double& fuel_temp_rise_scale() const { return _fuel_temp_rise_scale; }

    /// @brief Albedo boundary condition for a given direction and side
    inline double& albedo(int side, int dir) { return _albedo[dir * LR + side]; }

    /// @brief Mesh size in a given direction at a 3D node
    inline double& hmesh(int dir, int lk) { return _hmesh[idxDir(lk, dir)]; }

    /// @brief Axial mesh height at plane k
    inline double& hz(int k) { return _hz[k]; }

    /// @brief Node volume
    inline double& vol(int lk) { return _vol[lk]; }

    /// @brief Assembly volume (3D assembly index = la + nxya*k)
    inline double& vola(int lka) { return _vola[lka]; }

    /// @brief X-start index for row j
    inline int& nxs(int j) { return _nxs[j]; }

    /// @brief X-end index for row j
    inline int& nxe(int j) { return _nxe[j]; }

    /// @brief Y-start index for column i
    inline int& nys(int i) { return _nys[i]; }

    /// @brief Y-end index for column i
    inline int& nye(int i) { return _nye[i]; }

    /// @brief 2D Neighbor node index (WEST/EAST/NORTH/SOUTH)
    inline int& neibr(int dir, int l) { return _neibr[idxNeib2d(l, dir)]; }

    /// @brief 2D Neighbor with reflection BC (reflects at zero-albedo boundaries)
    inline int& neibrb(int dir, int l) { return _neibrb[idxNeib2d(l, dir)]; }

    /// @brief (i,j) to 2D node index l
    inline int& ijtol(int i, int j) { return _ijtol[idx2d(i, j)]; }

    /// @brief 3D Neighbor node index (NEWSBT directions)
    inline int& neib(int dir, int lk) { return _neib[idxNeib3d(lk, dir)]; }

    /// @brief 3D Neighbor by side and direction
    inline int& neib(int side, int dir, int lk) { return _neib[lk * NEWSBT + dir * LR + side]; }

    /// @brief Node index on a given side of a surface
    inline int& lklr(int side, int ls) { return _lklr[ls * LR + side]; }

    /// @brief Direction on a given side of a surface
    inline int& idirlr(int side, int ls) { return _idirlr[ls * LR + side]; }

    /// @brief Current sign on a given side of a surface
    inline int& sgnlr(int side, int ls) { return _sgnlr[ls * LR + side]; }

    /// @brief Node to surface index
    inline int& lktosfc(int side, int dir, int lk) { return _lktosfc[idxSfc(lk, dir, side)]; }

    /// @brief Assembly x-start for assembly-row ja
    inline int& nxsa(int ja) { return _nxsa[ja]; }

    /// @brief Assembly x-end for assembly-row ja
    inline int& nxea(int ja) { return _nxea[ja]; }

    /// @brief Assembly y-start for assembly-column ia
    inline int& nysa(int ia) { return _nysa[ia]; }

    /// @brief Assembly y-end for assembly-column ia
    inline int& nyea(int ia) { return _nyea[ia]; }

    /// @brief (ia,ja) to 2D assembly index la
    inline int& ijtola(int ia, int ja) { return _ijtola[idxAsm2d(ia, ja)]; }

    /// @brief Node to assembly index
    inline int& ltola(int l) { return _ltola[l]; }

    /// @brief Assembly sub-node to node index
    inline int& latol(int li, int la) { return _latol[idxAsmSub(la, li)]; }

    /// @brief Assembly rotation for sub-node
    inline int& larot(int li, int la) { return _larot[idxAsmSub(la, li)]; }

    /// @brief Assembly rotation array for a given assembly
    inline int* larot(int la) { return &_larot[la * _ndivxy2]; }

    /// @brief Composition index per node
    inline int& comp(int lk) { return _comps[lk]; }

    /// @brief Composition index array
    inline int* comps() { return _comps; }

    /// @brief Whether a 3D node is fuel
    [[nodiscard]] inline bool IsFuel(int lk) const { return _is_fuel[lk]; }
#pragma endregion

// Nodal information (flux, current, source)
#pragma region nodal information

private:
    double* _phif; // node-averaged scalar flux [ng * nxyz]
    double* _jnet; // surface-averaged net current [LR * ng * NDIRMAX * nxyz]
    double* _phis; // surface-averaged scalar flux [LR * ng * NDIRMAX * nxyz]
    double* _psi;  // fission source [nxyz]

public:
    /// @brief Node-averaged scalar flux array [ng * nxyz]
    inline double* Phif() { return _phif; }

    /// @brief Surface-averaged net current array
    inline double* Jnet() { return _jnet; }

    /// @brief Surface-averaged scalar flux array
    inline double* Phis() { return _phis; }

    /// @brief Fission source array
    inline double* Psi() { return _psi; }
#pragma endregion

// Pin power reconstruction
#pragma region pin power reconstruction

private:
    double* _phic;   // corner flux [nxyz * ng * 4]
    double* _ppr_p;  // particular coefficients [nxyz * ng * 15]
    double* _ppr_a;  // homogeneous coefficients [nxyz * ng * 8]
    double* _ppr_c;  // polynomial fitting coefficients [nxyz * ng * 15]
    double* _ppr_q;  // source expansion coefficients [nxyz * ng * 15]
    double* _ppr_l;  // axial leakage expansion [nxyz * ng * 9]
    double* _ppr_bt; // transverse buckling [nxyz * ng]

public:
    /// @brief Corner flux array [nxyz * ng * 4]
    inline double* Phic() { return _phic; }

    /// @brief PPR particular coefficients (p_i) [nxyz * ng * 15]
    inline double* CoeffPart() { return _ppr_p; }

    /// @brief PPR homogeneous coefficients (a_i) [nxyz * ng * 8]
    inline double* CoeffHom() { return _ppr_a; }

    /// @brief PPR polynomial fitting coefficients (c_i) [nxyz * ng * 15]
    inline double* CoeffFit() { return _ppr_c; }

    /// @brief PPR source expansion coefficients (q_i) [nxyz * ng * 15]
    inline double* CoeffExp() { return _ppr_q; }

    /// @brief PPR axial leakage expansion [nxyz * ng * 9]
    inline double* CoeffLeak() { return _ppr_l; }

    /// @brief PPR transverse buckling [nxyz * ng]
    inline double* CoeffBuckling() { return _ppr_bt; }

private:
    // Pin-wise reconstruction results
    double* _pphif;  // pin-wise heterogeneous flux [nxya * nz * ng * npins^2]
    double* _ppower; // pin-wise heterogeneous power [nxya * nz * npins^2]
    double  _frp;    // radial pin peaking factor
    double  _fqp;    // 3-D pin peaking factor

public:
    /// @brief Pin-wise heterogeneous flux array [nxya * nz * ng * npins^2]
    inline double* PinFlux() { return _pphif; }

    /// @brief Pin-wise heterogeneous power array [nxya * nz * npins^2]
    inline double* PinPower() { return _ppower; }

    /// @brief Radial pin peaking factor (Frp)
    inline double&                     frp() { return _frp; }
    [[nodiscard]] inline const double& frp() const { return _frp; }

    /// @brief 3-D pin peaking factor (Fqp)
    inline double&                     fqp() { return _fqp; }
    [[nodiscard]] inline const double& fqp() const { return _fqp; }
#pragma endregion

// Cross-section state variables
#pragma region cross-section information

private:
    double* _bppm; // boron concentration per node [ppm]
    double* _tful; // fuel temperature per node [K]
    double* _tmod; // moderator temperature per node [K]
    double* _dmod; // moderator density per node [g/cc]

public:
    /// @brief Boron concentration [ppm]
    inline double&                     bppm(int l) { return _bppm[l]; }
    [[nodiscard]] inline const double& bppm(int l) const { return _bppm[l]; }
    /// @brief Fuel temperature [K]
    inline double&                     tful(int l) { return _tful[l]; }
    [[nodiscard]] inline const double& tful(int l) const { return _tful[l]; }
    /// @brief Moderator temperature [K]
    inline double&                     tmod(int l) { return _tmod[l]; }
    [[nodiscard]] inline const double& tmod(int l) const { return _tmod[l]; }
    /// @brief Moderator density [g/cc]
    inline double&                     dmod(int l) { return _dmod[l]; }
    [[nodiscard]] inline const double& dmod(int l) const { return _dmod[l]; }
#pragma endregion

// Rod insertion
#pragma region rod insertion

private:
    double* _rod_fraction; // rod insertion fraction per node (0=out, 1=full)

public:
    /// @brief Rod insertion fraction (0.0 = out, 1.0 = fully inserted)
    inline double&                     rod_fraction(int l) { return _rod_fraction[l]; }
    [[nodiscard]] inline const double& rod_fraction(int l) const { return _rod_fraction[l]; }
#pragma endregion

public:
    Geometry();

    virtual ~Geometry();

    // Initialization

    /// @brief Initialize geometry from raw input parameters.
    void Initialize(const GeometryInput& in);

    /// @brief Initialize corner-point index maps.
    void initCorner(const int& ncorn, const int* lctol, const int* ltolc);

private:
    /// @brief Per-row and per-column node spans for the core map under quarter symmetry.
    /// Fills @p rowSpan (one entry per core row) and @p colSpan (one entry per column,
    /// sized to the widest row). Under quarter symmetry the first row/column is halved.
    void quarterSpans(const std::vector<std::vector<std::string>>& core, bool qsym, std::vector<int>& rowSpan, std::vector<int>& colSpan) const;
};

} // namespace rasbery
