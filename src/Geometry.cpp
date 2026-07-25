#include "Geometry.h"

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
    _comps        = nullptr;
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
    delete[] _albedo;
    delete[] _neibr;
    delete[] _neibrb;
    delete[] _ijtol;
    delete[] _nxs;
    delete[] _nxe;
    delete[] _nys;
    delete[] _nye;
    delete[] _comps;
    delete[] _neib;
    delete[] _hmesh;
    delete[] _lktosfc;
    delete[] _vol;
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
    delete[] _phif;
    delete[] _jnet;
    delete[] _phis;
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

    // Store raw layout so XSSet can access it without re-receiving gin.
    _core  = in.core;
    _batch = in.batch;

    // Boundary condition
    _symopt = in.symopt;
    _symang = in.symang;
    _symdiv = in.symdiv;
    _part   = _symang / 360.0;
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
        int iys = 0, iye = 0;
        for (size_t row = 0; row < v_nxs.size(); row++) {
            if (i < v_nxs[row]) iys++;
            if (i < v_nxe[row]) iye++;
        }
        v_nys[i] = iys;
        v_nye[i] = iye;
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

    // hmesh: fine-mesh sizes per direction
    double hx = in.hx / _ndivxy;
    double hy = in.hy / _ndivxy;

    _comps   = new int[_nxyz];
    _neib    = new int[NEWSBT * _nxyz];
    _hmesh   = new double[NDIRMAX * _nxyz];
    _lktosfc = new int[NEWSBT * _nxyz];
    _vol     = new double[_nxyz];

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

    // Allocate solution arrays
    _jnet = new double[LR * _ng * NDIRMAX * _nxyz]{};
    _phis = new double[LR * _ng * NDIRMAX * _nxyz]{};
    _phif = new double[_ng * _nxyz]{};
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