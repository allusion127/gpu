#include "Geometry.h"

#include "CompatFormat.h"
#include "HostPinRegistry.h"

using namespace rasbery;

Geometry::Geometry() {
    _albedo       = nullptr;
    _neibr        = nullptr;
    _neibrb       = nullptr;
    _ijtol        = nullptr;
    _nxs          = nullptr;
    _nxe          = nullptr;
    _nys          = nullptr;
    _nye          = nullptr;
    _neib         = nullptr;
    _hmesh        = nullptr;
    _lktosfc      = nullptr;
    _vol          = nullptr;
    _hz           = nullptr;
    _idirlr       = nullptr;
    _sgnlr        = nullptr;
    _lklr         = nullptr;
    _nxsa         = nullptr;
    _nxea         = nullptr;
    _nysa         = nullptr;
    _nyea         = nullptr;
    _ijtola       = nullptr;
    _ltola        = nullptr;
    _latol        = nullptr;
    _larot        = nullptr;
    _vola         = nullptr;
    _lctol        = nullptr;
    _ltolc        = nullptr;
    _bppm         = nullptr;
    _tful         = nullptr;
    _tmod         = nullptr;
    _dmod         = nullptr;
    _rod_fraction = nullptr;
    _phif         = nullptr;
    _jnet         = nullptr;
    _phis         = nullptr;
    _psi          = nullptr;
    _phic         = nullptr;
    _ppr_p        = nullptr;
    _ppr_a        = nullptr;
    _ppr_c        = nullptr;
    _ppr_q        = nullptr;
    _ppr_l        = nullptr;
    _ppr_bt       = nullptr;
    _symdiv       = false;
    _is_fuel      = nullptr;
    _pphif        = nullptr;
    _ppower       = nullptr;
    _frp          = 0.0;
    _fqp          = 0.0;
}

Geometry::~Geometry() {
    // Release the host pin leases the device backends took on Geometry-owned
    // buffers BEFORE the delete[]s below hand the pages back to the allocator.
    // _phif/_jnet/_phis are the three the nodal arm and the CMFD sweep path
    // page-lock (Phif also under its BICGCMFD alias `flux`); unpinning an
    // address that was never leased is a no-op, so this list is unconditional.
    // Without it the next deck on this worker inherits a live registration at
    // its own fresh addresses -- see HostPinRegistry.h.
    //
    // _vol JOINED THE LIST when the CMFD sweep stopped staging a private copy
    // of it: BICGCMFD::driveDeviceSweeps now aliases &_g.vol(0) and page-locks
    // it under the OWNER's tag, `geom.vol@sweep`.  The owner is us, so the
    // release is ours -- and it was missing, which every run reported as
    // `[RASBERY][WARN][PIN] {"leaked_ranges":1,...}` at shutdown.  Harmless for
    // a single deck (the process exits), but a --batch-mode worker that
    // recycles hands the next deck's fresh allocation an address the driver
    // still holds registered, which is precisely the dead-tenant aliasing the
    // lease contract exists to prevent.
    rasberyUnpinHost(_phif);
    rasberyUnpinHost(_jnet);
    rasberyUnpinHost(_phis);
    rasberyUnpinHost(_vol);

    delete[] _albedo;
    delete[] _neibr;
    delete[] _neibrb;
    delete[] _ijtol;
    delete[] _nxs;
    delete[] _nxe;
    delete[] _nys;
    delete[] _nye;
    delete[] _neib;
    delete[] _hmesh;
    delete[] _lktosfc;
    rasberyPageExclusiveDeleteArray(_vol);
    delete[] _hz;
    delete[] _idirlr;
    delete[] _sgnlr;
    delete[] _lklr;
    delete[] _nxsa;
    delete[] _nxea;
    delete[] _nysa;
    delete[] _nyea;
    delete[] _ijtola;
    delete[] _ltola;
    delete[] _latol;
    delete[] _larot;
    delete[] _vola;
    delete[] _lctol;
    delete[] _ltolc;
    delete[] _bppm;
    delete[] _tful;
    delete[] _tmod;
    delete[] _dmod;
    delete[] _rod_fraction;
    rasberyPageExclusiveDeleteArray(_phif);
    rasberyPageExclusiveDeleteArray(_jnet);
    rasberyPageExclusiveDeleteArray(_phis);
    delete[] _psi;
    delete[] _phic;
    delete[] _ppr_p;
    delete[] _ppr_a;
    delete[] _ppr_c;
    delete[] _ppr_q;
    delete[] _ppr_l;
    delete[] _ppr_bt;
    delete[] _is_fuel;
    delete[] _pphif;
    delete[] _ppower;
}

void Geometry::quarterSpans(const std::vector<std::vector<std::string>>& core, bool qsym, std::vector<int>& rowSpan, std::vector<int>& colSpan) const {
    size_t ncol = 0;
    for (auto& row : core)
        ncol = std::max(ncol, row.size());

    rowSpan.resize(core.size());
    for (size_t r = 0; r < core.size(); r++)
        rowSpan[r] = (qsym && r == 0) ? (_ndivxy / 2) : _ndivxy;

    colSpan.resize(ncol);
    for (size_t c = 0; c < ncol; c++)
        colSpan[c] = (qsym && c == 0) ? (_ndivxy / 2) : _ndivxy;
}

void Geometry::Initialize(const GeometryInput& in) {
    _ng      = in.ng;
    _nz      = in.nz;
    _ndivxy  = in.ndivxy;
    _ndivxy2 = _ndivxy * _ndivxy;
    _ng2     = _ng * _ng;
    _npins   = in.npins;

    // THE FUEL-TEMPERATURE DIVISOR, RESOLVED ONCE, WITH A RECEIPT.  `nfrod` is
    // rods per ASSEMBLY and the T/H bodies divide per NODE, so the deck value is
    // folded by ndivxy^2 here -- the one place that knows both.  The default is
    // the legacy literal (ThFuelRods.h says why), and the line below is printed
    // unconditionally because a divisor nobody can see in a log is exactly how
    // 62 survived from e76d40d to this campaign.
    {
        const double deck_rods = (in.nfrod > 0 && _ndivxy2 > 0)
                                     ? static_cast<double>(in.nfrod) /
                                           static_cast<double>(_ndivxy2)
                                     : 0.0;
        const th::FuelRods rods = th::resolveFuelRodsPerNode(deck_rods);
        _fuel_rods_per_node     = rods.value;
        _fuel_rods_source       = rods.source;
        std::cout << std::format(
            "  [RASBERY][TH][NFROD] {{\"schema_version\":1,\"rods_per_node\":{:.17g},"
            "\"source\":\"{}\",\"deck_nfrod\":{},\"ndivxy\":{},"
            "\"deck_rods_per_node\":{:.17g},\"legacy\":{:.17g}}}\n",
            _fuel_rods_per_node, _fuel_rods_source, in.nfrod, _ndivxy, deck_rods,
            th::kLegacyFuelRodsPerNode);
    }

    // Store raw layout so XSSet can access it without re-receiving gin.
    _core  = in.core;
    _batch = in.batch;

    // Boundary condition
    _symopt = in.symopt;
    _symang = in.symang;
    _symdiv = in.symdiv;
    _part   = _symang / 360.0;

    // A quarter core can be folded onto the full core in two inequivalent ways,
    // and "mirror" selects between them.
    //
    //   mirror: true  -- MIRROR (reflective) fold. The west and north boundaries
    //                    reflect onto themselves (the zero-albedo self-mapping in
    //                    the neighbour build below) and IO::ApplyShuffle completes
    //                    symmetry-cut assemblies by mirror reflection.
    //   mirror: false -- 90-degree ROTATIONAL fold (MASTER's %GEN_DIM nsym=1). The
    //                    first node row and the first node column are stitched to
    //                    each other: the north face of node (i,0) IS the west face
    //                    of node (0,i), seen from the other side of a 90-degree
    //                    rotation about the core centre.
    //
    // The two agree only when the quadrant loading is diagonally symmetric;
    // otherwise they describe different cores. See rotationalFold() below.
    const bool rotfold = (_symang == 90 && !_symopt);

    _albedo = new double[LR * NDIRMAX];
    std::copy_n(in.albedo.data(), LR * NDIRMAX, _albedo);

    // Check quarter symmetry
    const bool qsym = (_symang == 90 && _symdiv);
    const int  rn   = qsym ? _ndivxy / 2 : 0;

    // Per-row / per-column node spans for the core map (first row/col halved under qsym).
    std::vector<int> rowSpan, colSpan;
    quarterSpans(in.core, qsym, rowSpan, colSpan);

    // Derive assembly counts from core map.
    int nxa_a = 0;
    int nya_a = static_cast<int>(in.core.size());
    for (auto& row : in.core)
        nxa_a = std::max(nxa_a, static_cast<int>(row.size()));

    // Number of nodes = number of assemblies * ndivxy - nodes excluded by mirror
    _nx = nxa_a * _ndivxy - rn;
    _ny = nya_a * _ndivxy - rn;

    // Build node-wise index start/end (nxs, nxe, nys, nye) from core map
    std::vector<int> v_nxs, v_nxe;
    for (size_t rowIdx = 0; rowIdx < in.core.size(); rowIdx++) {
        const auto& row    = in.core[rowIdx];
        int         rowend = rowSpan[rowIdx];
        for (int r = 0; r < rowend; r++) {
            int ixs = 0, ixe = 0;
            for (size_t colIdx = 0; colIdx < row.size(); colIdx++) {
                int span = colSpan[colIdx];
                if (row[colIdx] == "XX") ixs += span;
                ixe += span;
            }
            v_nxs.push_back(ixs);
            v_nxe.push_back(ixe);
        }
    }

    std::vector<int> v_nys(_nx), v_nye(_nx);
    for (int i = 0; i < _nx; i++) {
        // A column can have void rows at both its north and south ends in a
        // full-core ragged map. Counting rows with i<nxs/i<nxe conflates the
        // two ends and can make [nys,nye) include cells whose ijtol is -1.
        // Derive the actual first/last occupied row from the row intervals.
        int first = _ny;
        int last  = -1;
        for (int row = 0; row < static_cast<int>(v_nxs.size()); row++) {
            if (v_nxs[row] <= i && i < v_nxe[row]) {
                first = std::min(first, row);
                last  = std::max(last, row);
            }
        }
        if (last < first)
            throw std::runtime_error("Geometry: empty fine-mesh column in core map.");
        v_nys[i] = first;
        v_nye[i] = last + 1;
    }

    // Copy row/column start-end into member arrays
    _nxs = new int[_ny];
    _nxe = new int[_ny];
    _nys = new int[_nx];
    _nye = new int[_nx];
    std::copy_n(v_nxs.data(), _ny, _nxs);
    std::copy_n(v_nxe.data(), _ny, _nxe);
    std::copy_n(v_nys.data(), _nx, _nys);
    std::copy_n(v_nye.data(), _nx, _nye);

    // Build (i,j) → l map and count nxy
    int* loc_ijtol = new int[_nx * _ny];
    std::fill_n(loc_ijtol, _nx * _ny, -1);
    _nxy = 0;
    for (int j = 0; j < _ny; j++)
        for (int i = v_nxs[j]; i < v_nxe[j]; i++)
            loc_ijtol[j * _nx + i] = _nxy++;

    _nxyz  = _nxy * _nz;
    _ngxy  = _nxy * _ng;
    _ngxyz = _nxyz * _ng;
    _ijtol = new int[_nx * _ny];
    std::copy_n(loc_ijtol, _nx * _ny, _ijtol);

    // Build 2D neighbor map
    _neibr = new int[_nxy * NEWS];
    std::fill_n(_neibr, _nxy * NEWS, -1);
    for (int j = 0; j < _ny; j++) {
        for (int i = v_nxs[j]; i < v_nxe[j]; i++) {
            int l = loc_ijtol[j * _nx + i];
            if (i > v_nxs[j]) neibr(WEST, l) = l - 1;
            if (i < v_nxe[j] - 1) neibr(EAST, l) = l + 1;
            if (j > v_nys[i]) neibr(NORTH, l) = loc_ijtol[(j - 1) * _nx + i];
            if (j < v_nye[i] - 1) neibr(SOUTH, l) = loc_ijtol[(j + 1) * _nx + i];
        }
    }

    // Build 2D neighbor map with reflection BC
    // At zero-albedo (reflecting) boundaries, map to self instead of -1.
    _neibrb = new int[_nxy * NEWS];
    std::copy_n(_neibr, _nxy * NEWS, _neibrb);
    for (int l = 0; l < _nxy; l++) {
        if (_neibrb[l * NEWS + WEST] < 0 && _albedo[XDIR * LR + LEFT] < EPS) _neibrb[l * NEWS + WEST] = l;
        if (_neibrb[l * NEWS + EAST] < 0 && _albedo[XDIR * LR + RIGHT] < EPS) _neibrb[l * NEWS + EAST] = l;
        if (_neibrb[l * NEWS + NORTH] < 0 && _albedo[YDIR * LR + LEFT] < EPS) _neibrb[l * NEWS + NORTH] = l;
        if (_neibrb[l * NEWS + SOUTH] < 0 && _albedo[YDIR * LR + RIGHT] < EPS) _neibrb[l * NEWS + SOUTH] = l;
    }

    // 90-degree rotational fold: stitch the first node row to the first node column.
    //
    // The quadrant is the region x >= 0, y >= 0 with the core centre at the corner
    // of node (0,0); x runs east, y runs south.  The C4 rotation R: (x,y) -> (-y,x)
    // carries the modelled quadrant onto its neighbour, and its inverse carries the
    // ghost strip west of column 0 onto node row 0:
    //
    //      west ghost of node (0,j)  ==  R^-1 image of node (j,0)
    //      north ghost of node (i,0) ==  R    image of node (0,i)
    //
    // so the WEST neighbour of (0,j) is (j,0) and the NORTH neighbour of (j,0) is
    // (0,j) -- one mutual link per centreline index, degenerating to self-coupling
    // at (0,0), whose west face is its own north face rotated.
    //
    // Nothing else in the geometry has to change.  The surface tables below already
    // carry a per-side direction and sign (idirlr/sgnlr) and already give the row's
    // west-boundary surface idirlr(LEFT)=YDIR,sgnlr(LEFT)=MINUS and the column's
    // north-boundary surface idirlr(LEFT)=XDIR,sgnlr(LEFT)=MINUS -- which are
    // exactly the rotational partner's direction and orientation (the partner's
    // local axis is anti-parallel to this surface's axis).  They pick their left
    // node up from neib(WEST/NORTH,...), so stitching _neibr is enough for the
    // surfaces, for CMFD's coupling (CMFD::setls indexes cc by direction slot, and
    // a stitched slot still holds exactly one neighbour) and for the nodal
    // transverse leakage (Nodal::caltrlcff12 already reads the neighbour's
    // direction from idirlr(LEFT,lsl) rather than assuming its own).
    //
    // _neibrb is left on the reflective closure on purpose: it feeds only PPR's
    // 3x3 pin-reconstruction stencil, which has no way to express a rotated
    // neighbour (its flags encode reversal, not a direction swap).  Mirror closure
    // there is the existing behaviour and keeps the boundary surface flux correct;
    // the far diagonal cells of the stencil stay approximate.  Pin reconstruction
    // under the rotational fold is tracked separately.
    if (rotfold) {
        if (std::abs(in.hx - in.hy) > 1.0E-9 * std::max(1.0, std::abs(in.hx)))
            throw std::runtime_error(
                "Geometry: the 90-degree rotational quarter-core fold maps the x axis onto "
                "the y axis, so it needs a square assembly pitch (hx == hy).");
        if (_albedo[XDIR * LR + LEFT] >= EPS || _albedo[YDIR * LR + LEFT] >= EPS)
            throw std::runtime_error(
                "Geometry: under the 90-degree rotational quarter-core fold the west and "
                "north faces are interior surfaces, not boundaries, so a non-zero west or "
                "north albedo cannot be honoured. Declare them 0.0.");

        const int nline = std::max(_nx, _ny);
        for (int t = 0; t < nline; t++) {
            // node (0,t): column 0 of row t.  node (t,0): column t of row 0.
            const int lcol = (t < _ny && v_nxs[t] == 0 && v_nxe[t] > 0) ? loc_ijtol[t * _nx + 0] : -1;
            const int lrow = (t < _nx && v_nxs[0] <= t && t < v_nxe[0]) ? loc_ijtol[0 * _nx + t] : -1;
            if ((lcol < 0) != (lrow < 0))
                throw std::runtime_error(
                    "Geometry: the 90-degree rotational quarter-core fold needs the core map to "
                    "be diagonally consistent, but node (0," + std::to_string(t) + ") and node (" +
                    std::to_string(t) + ",0) do not both exist.");
            if (lcol < 0) continue;
            _neibr[lcol * NEWS + WEST]  = lrow;
            _neibr[lrow * NEWS + NORTH] = lcol;
        }
    }

    // hmesh: fine-mesh sizes per direction
    double hx = in.hx / _ndivxy;
    double hy = in.hy / _ndivxy;

    _neib    = new int[NEWSBT * _nxyz];
    _hmesh   = new double[NDIRMAX * _nxyz];
    _lktosfc = new int[NEWSBT * _nxyz];
    // PAGE-EXCLUSIVE, because the CMFD sweep page-locks this array directly
    // (`geom.vol@sweep`, since BICGCMFD stopped staging a private copy of it).
    // cudaHostRegister works on whole PAGES, and `new double[]` leaves _vol's
    // first and last pages shared with its neighbours -- `_lktosfc` above and
    // `_hz` below, because a general-purpose allocator packs its chunks 16
    // bytes apart.  Whether the registration then succeeds is ALLOCATOR LUCK:
    // it is refused as an overlap the moment anything else on one of those two
    // pages is registered first, and the failure is silent -- the buffer just
    // takes the pageable path on every launch and the only trace is
    // overlap_rejections in the shutdown receipt.  Owning our pages outright
    // makes it deterministic instead of lucky.  Same treatment as
    // _jnet/_phis/_phif below; see the page-exclusive section of
    // HostPinRegistry.h.
    //
    // Zeroed rather than raw is free here and byte-identical: the loop below
    // writes every one of the _nxyz elements before anything reads them.
    _vol = rasberyPageExclusiveZeroedArray<double>(static_cast<size_t>(_nxyz));

    for (int z = 0; z < _nz; z++) {
        for (int l2d = 0; l2d < _nxy; l2d++) {
            int lk          = z * _nxy + l2d;
            hmesh(XDIR, lk) = hx;
            hmesh(YDIR, lk) = hy;
            hmesh(ZDIR, lk) = in.hz[z];
        }
    }

    // Half-node at symmetry boundary for odd subdivision.
    if (qsym && _ndivxy % 2 != 0) {
        for (int z = 0; z < _nz; z++) {
            for (int j = 0; j < _ny; j++)
                hmesh(XDIR, loc_ijtol[j * _nx + 0] + _nxy * z) = hx / 2;
            for (int i = 0; i < _nx; i++)
                hmesh(YDIR, loc_ijtol[0 * _nx + i] + _nxy * z) = hy / 2;
        }
    }

    // 3D neighbor, volume
    for (int k = 0; k < _nz; k++) {
        int l0 = k * _nxy;
        for (int l2d = 0; l2d < _nxy; l2d++) {
            int l = l0 + l2d;
            for (int d = 0; d < NEWS; d++)
                neib(d, l) = (neibr(d, l2d) < 0) ? -1 : l0 + neibr(d, l2d);
            neib(BOT, l) = (k > 0) ? (k - 1) * _nxy + l2d : -1;
            neib(TOP, l) = (k < _nz - 1) ? (k + 1) * _nxy + l2d : -1;
        }
    }

    for (int l = 0; l < _nxyz; l++)
        vol(l) = hmesh(XDIR, l) * hmesh(YDIR, l) * hmesh(ZDIR, l);

    _hz = new double[_nz];
    for (int k = 0; k < _nz; k++)
        hz(k) = hmesh(ZDIR, k * _nxy);

    // Allocate solution arrays.  jnet/phis/phif are the three Geometry-owned
    // buffers both nodal arms and the CMFD sweep path page-lock, so they get
    // page-exclusive storage: adjacent heap arrays share their boundary pages,
    // and cudaHostRegister refuses the second of any two that do.  See the
    // page-exclusive section of HostPinRegistry.h.
    _jnet = rasberyPageExclusiveZeroedArray<double>(
        static_cast<size_t>(LR) * _ng * NDIRMAX * _nxyz);
    _phis = rasberyPageExclusiveZeroedArray<double>(
        static_cast<size_t>(LR) * _ng * NDIRMAX * _nxyz);
    _phif = rasberyPageExclusiveZeroedArray<double>(static_cast<size_t>(_ng) * _nxyz);
    std::fill_n(_phif, _ng * _nxyz, 1.0);
    _psi          = new double[_nxyz];
    _bppm         = new double[_nxyz]{};
    _tful         = new double[_nxyz]{};
    _tmod         = new double[_nxyz]{};
    _dmod         = new double[_nxyz]{};
    _rod_fraction = new double[_nxyz]{};
    _phic         = new double[_nxyz * _ng * 4]{};
    _ppr_p        = new double[_nxyz * _ng * 15]{};
    _ppr_a        = new double[_nxyz * _ng * 8]{};
    _ppr_c        = new double[_nxyz * _ng * 15]{};
    _ppr_q        = new double[_nxyz * _ng * 15]{};
    _ppr_l        = new double[_nxyz * _ng * 9]{};
    _ppr_bt       = new double[_nxyz * _ng]{};

    // Surface maps
    _nsurf = 0;
    for (int j = 0; j < _ny; j++) {
        ++_nsurf;
        for (int i = nxs(j); i < nxe(j); i++)
            ++_nsurf;
    }
    for (int i = 0; i < _nx; i++) {
        ++_nsurf;
        for (int j = nys(i); j < nye(i); j++)
            ++_nsurf;
    }
    _nsurf *= _nz;
    _nsurf += _nxy * (_nz + 1);
    _idirlr = new int[_nsurf * LR];
    _sgnlr  = new int[_nsurf * LR];
    _lklr   = new int[_nsurf * LR];

    int ls = -1;
    for (int k = 0; k < _nz; k++) {
        int lk0 = k * _nxy;
        for (int j = 0; j < _ny; j++) {
            ++ls;
            idirlr(LEFT, ls)  = YDIR;
            idirlr(RIGHT, ls) = XDIR;
            sgnlr(LEFT, ls)   = MINUS;
            sgnlr(RIGHT, ls)  = PLUS;
            int l             = ijtol(_nxs[j], j);
            int lk            = lk0 + l;
            lklr(LEFT, ls)    = neib(WEST, lk);
            lklr(RIGHT, ls)   = lk;
            for (int i = nxs(j); i < nxe(j); i++) {
                l                        = ijtol(i, j);
                lk                       = lk0 + l;
                lktosfc(LEFT, XDIR, lk)  = ls;
                lktosfc(RIGHT, XDIR, lk) = ++ls;
                lklr(LEFT, ls)           = lk;
                lklr(RIGHT, ls)          = neib(EAST, lk);
                idirlr(LEFT, ls)         = XDIR;
                idirlr(RIGHT, ls)        = XDIR;
                sgnlr(LEFT, ls)          = PLUS;
                sgnlr(RIGHT, ls)         = PLUS;
            }
        }
        for (int i = 0; i < _nx; i++) {
            ++ls;
            int l             = ijtol(i, nys(i));
            int lk            = lk0 + l;
            idirlr(LEFT, ls)  = XDIR;
            idirlr(RIGHT, ls) = YDIR;
            sgnlr(LEFT, ls)   = MINUS;
            sgnlr(RIGHT, ls)  = PLUS;
            lklr(LEFT, ls)    = neib(NORTH, lk);
            lklr(RIGHT, ls)   = lk;
            for (int j = nys(i); j < nye(i); j++) {
                l                        = ijtol(i, j);
                lk                       = lk0 + l;
                lktosfc(LEFT, YDIR, lk)  = ls;
                lktosfc(RIGHT, YDIR, lk) = ++ls;
                lklr(LEFT, ls)           = lk;
                lklr(RIGHT, ls)          = neib(SOUTH, lk);
                idirlr(LEFT, ls)         = YDIR;
                idirlr(RIGHT, ls)        = YDIR;
                sgnlr(LEFT, ls)          = PLUS;
                sgnlr(RIGHT, ls)         = PLUS;
            }
        }
    }
    for (int l2d = 0; l2d < _nxy; l2d++) {
        ++ls;
        idirlr(LEFT, ls)  = ZDIR;
        idirlr(RIGHT, ls) = ZDIR;
        sgnlr(LEFT, ls)   = PLUS;
        sgnlr(RIGHT, ls)  = PLUS;
        lklr(LEFT, ls)    = -1;
        for (int k = 0; k < _nz; k++) {
            int lk                   = k * _nxy + l2d;
            lktosfc(LEFT, ZDIR, lk)  = ls;
            lklr(RIGHT, ls)          = lk;
            lktosfc(RIGHT, ZDIR, lk) = ++ls;
            lklr(LEFT, ls)           = lk;
            idirlr(LEFT, ls)         = ZDIR;
            idirlr(RIGHT, ls)        = ZDIR;
            sgnlr(LEFT, ls)          = PLUS;
            sgnlr(RIGHT, ls)         = PLUS;
        }
        lklr(RIGHT, ls) = -1;
    }
    assert(ls + 1 == _nsurf);

    // Assembly index maps
    int nxyfa = 0;
    for (auto& row : in.core)
        for (auto& col : row)
            if (col != "XX") nxyfa++;

    _nxyfa   = nxyfa;
    _ncellfa = _npins * _npins;

    int nsub = (_part == 1.0 || _ndivxy == 1) ? 0 : _ndivxy / 2;
    _nya     = (_ny + nsub) / _ndivxy;
    _nxa     = (_nx + nsub) / _ndivxy;

    _nxsa = new int[_nya]{};
    _nxea = new int[_nya]{};
    _nysa = new int[_nxa]{};
    _nyea = new int[_nxa]{};

    _nxya = 0;
    for (int ja = 0; ja < _nya; ja++) {
        int j    = ja * _ndivxy;
        nxsa(ja) = (nxs(j) + nsub) / _ndivxy;
        nxea(ja) = (nxe(j) + nsub) / _ndivxy;
        _nxya += nxea(ja) - nxsa(ja);
    }

    for (int ia = 0; ia < _nxa; ia++) {
        int i    = ia * _ndivxy;
        nysa(ia) = (nys(i) + nsub) / _ndivxy;
        nyea(ia) = (nye(i) + nsub) / _ndivxy;
    }

    _ijtola = new int[_nxa * _nya];
    std::fill(_ijtola, _ijtola + _nxa * _nya, -1);
    _nxya = 0;
    for (int ja = 0; ja < _nya; ja++)
        for (int ia = nxsa(ja); ia < nxea(ja); ia++)
            ijtola(ia, ja) = _nxya++;

    _ltola = new int[_nxy]{};
    _latol = new int[_nxya * _ndivxy2];
    std::fill_n(_latol, _nxya * _ndivxy2, -1);
    _vola = new double[_nxya * _nz]{};

    for (int j = 0; j < _ny; j++) {
        int ja = (j + nsub) / _ndivxy;
        int dj = j - ja * _ndivxy + nsub;
        for (int i = nxs(j); i < nxe(j); i++) {
            int ia                       = (i + nsub) / _ndivxy;
            int di                       = i - ia * _ndivxy + nsub;
            int l                        = ijtol(i, j);
            int la                       = ijtola(ia, ja);
            ltola(l)                     = la;
            latol(dj * _ndivxy + di, la) = l;
            for (int k = 0; k < _nz; k++)
                vola(la + _nxya * k) += vol(l + _nxy * k);
        }
    }

    // Fuel flag per 3D node — "R"-prefixed batch layers are reflector.
    _is_fuel = new bool[_nxyz]{};
    for (int z = 0; z < _nz; z++) {
        int joff = 0;
        for (size_t row = 0; row < in.core.size(); row++) {
            int rowend = rowSpan[row];
            int ioff   = 0;
            for (size_t col = 0; col < in.core[row].size(); col++) {
                int         colend = colSpan[col];
                const auto& name   = in.core[row][col];
                if (name == "XX") {
                    ioff += colend;
                    continue;
                }
                const auto& layer = in.batch.at(name)[_nz - 1 - z];
                bool        fuel  = (layer.empty() || layer[0] != 'R');
                for (int j = 0; j < rowend; j++)
                    for (int i = 0; i < colend; i++) {
                        int lk       = ijtol(ioff + i, joff + j) + _nxy * z;
                        _is_fuel[lk] = fuel;
                    }
                ioff += colend;
            }
            joff += rowend;
        }
    }

    // kbc/kec from per-layer fuel status.
    _kbc = _nz;
    _kec = 0;
    for (int k = 0; k < _nz; k++)
        for (int l = 0; l < _nxy; l++)
            if (_is_fuel[k * _nxy + l]) {
                _kbc = std::min(_kbc, k);
                _kec = std::max(_kec, k + 1);
                break;
            }
    if (_kbc > _kec) {
        _kbc = 0;
        _kec = _nz;
    }

    _hzcore = 0.0;
    for (int k = _kbc; k < _kec; k++)
        _hzcore += hmesh(ZDIR, k * _nxy);

    // Pin-wise reconstruction arrays (must be after _nxya and _npins are finalised)
    {
        const int npina = _npins * _npins;
        _pphif          = new double[static_cast<size_t>(_nxya) * _nz * _ng * npina]{};
        _ppower         = new double[static_cast<size_t>(_nxya) * _nz * npina]{};
        _frp            = 0.0;
        _fqp            = 0.0;
    }

    delete[] loc_ijtol;
}

void Geometry::initCorner(const int& ncorn, const int* lctol, const int* ltolc) {
    _ncorn = ncorn;

    _lctol = new int[NEWS * ncorn];
    _ltolc = new int[NEWS * nxy()];

    std::copy(lctol, lctol + NEWS * ncorn, _lctol);
    std::copy(ltolc, ltolc + NEWS * nxy(), _ltolc);

    for (int i = 0; i < NEWS * ncorn; i++) {
        --_lctol[i];
    }

    for (int i = 0; i < NEWS * nxy(); i++) {
        --_ltolc[i];
    }
}
