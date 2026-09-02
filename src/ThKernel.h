#pragma once

// The thermal-hydraulics node/channel bodies, compiled by BOTH g++ and nvcc
// from this one text -- WP22 (Rev.7 plan Tasks 13-15, "M2: no host numerics
// inside a statepoint").
//
// WHAT MOVED, AND WHAT DID NOT.  XSSet::UpdateTH is four phases:
//
//   1. the per-node power density fold   sum_ig xskf[ig][l] * phif[l][ig], * vol
//   2. a serial total, and a global sign flip when it comes out negative
//   3. SolveTH: per CHANNEL, an axial sweep that carries the coolant enthalpy
//      from the inlet up through the reflector/fuel/reflector stack, reading
//      the water-property tables at each node and setting tmod/dmod/tful
//   4. the under-relaxation blend and the delta_Dop max
//
// Phases 1, 3 and 4 are per-node or per-channel and are what this header ports.
// Phase 2 is a SERIAL FOLD IN NODE ORDER and stays one: `total_power +=
// node_power[lk]` for lk = 0..nxyz is not associative in floating point, so a
// tree reduction is a different number.  It runs in ONE device lane, iterating
// the same range in the same direction.  20k adds in one lane is microseconds;
// a wrong sum is a wrong core.
//
// THE CHANNEL SWEEP IS INHERENTLY SERIAL AND IS MAPPED LANE-PER-CHANNEL.  The
// enthalpy `h_cur` at node k is the enthalpy at k-1 plus that node's rise, so
// the axial direction cannot be parallelised without changing the arithmetic.
// One thread owns one radial position l and walks k = 0..nz-1 exactly as the
// host loop does.  The parallelism is the nxy channels, which is what the host
// loop's OUTER index already was.
//
// CONTRACTION IS EXPLICIT, AND THAT IS THE WHOLE B0 CLAIM.  gcc at -O3
// (-ffp-contract=fast) fuses a*b+c into an fma at some of the multiply-adds
// below and not at others, and WHICH ones is a property of the build host --
// exactly the situation CmfdOuterKernel.h and XeKernel.h are in.  So every
// multiply-add site below reads its form out of a MASK passed in as a kernel
// argument, the mask is MINED on this host at startup (src/ThFormMiner.cpp)
// against a verbatim quotation compiled in its own translation unit
// (src/ThReference.cpp), and the device TU is built with --fmad=false so nvcc
// cannot fuse anything the mask did not ask for.
//
// This header must stay compilable by both compilers: no STL containers, no
// exceptions, no allocation.  Same rule and same reason as XsReconKernel.h.

#include <cmath>

#if defined(__CUDACC__)
    #define RASBERY_TH_HD __host__ __device__
#else
    #define RASBERY_TH_HD
#endif

namespace rasbery::th {

// ---------------------------------------------------------------------------
// Contraction sites
// ---------------------------------------------------------------------------
//
// One bit per multiply-add whose form gcc is free to choose.  Bit set = the
// site is a single-rounding fma; bit clear = a separately-rounded product
// followed by an add.  The names are the expressions in XSSet.cpp, so a mined
// hex value can be read back to a line.

enum ThFormBit : int {
    /// milk::Table::Get: `z0 = z00 + fx * (z01 - z00)`
    TH_LERP_X0 = 0,
    /// milk::Table::Get: `z1 = z10 + fx * (z11 - z10)`
    TH_LERP_X1 = 1,
    /// milk::Table::Get: `return z0 + fy * (z1 - z0)`
    TH_LERP_Y = 2,
    /// UpdateTH: `power_density += xskf[ig*nxyz+lk] * Phif()[lk*ng+ig]`
    TH_POWER_ACC = 3,
    /// SolveTH: `tful = tmod + fuel_temp_rise_scale() * GetTfuel(bu, lpd)`
    TH_TFUEL = 4,
    /// UpdateTH: the three under-relaxation blends `(1-w)*old + w*new`.  TWO
    /// BITS, not one, because this site has THREE forms and not two: the
    /// expression is a sum of two products, so gcc may contract either one of
    /// them into the add or neither.  A one-bit site would have been unable to
    /// spell the form the host actually ran and the mining would have reported a
    /// residual it could not remove.  0 = neither, 1 = fma on `w * new`,
    /// 2 = fma on `(1-w) * old`; 3 is unused and reads as 0.
    ///
    /// ONE FIELD FOR THREE STATEMENTS, deliberately: they are the same
    /// expression on three different arrays, compiled adjacently, and no
    /// compiler this tree has met contracts one and not the others.  If one ever
    /// does, the mining reports a non-zero residual rather than a wrong answer --
    /// which is the property the mask exists to give.
    TH_RELAX = 5,
    TH_RELAX_HI = 6,
    /// GetTfuel: `rise * safe_lpd / first_lpd` -- a product feeding a divide, so
    /// the only choice is whether the product is rounded before the divide.
    TH_TFUEL_LINEAR = 7,
    /// SolveTH: `h_avg = 0.5 * (h_cur + h_out)`.  Exact in binary floating point
    /// either way, so this site is a DON'T-CARE and the mining will legitimately
    /// settle either way; it carries a bit anyway so the receipt's mask has one
    /// field per multiply-add rather than one per multiply-add the author
    /// believed mattered.
    TH_HAVG = 8,
    TH_BIT_COUNT = 9
};

/// The bit index of the first ONE-BIT site's successor: sites [0, TH_ONE_BIT_END)
/// are single-bit, TH_RELAX is the one two-bit field, and TH_TFUEL_LINEAR
/// onwards are single-bit again.  The miner reads this rather than hard-coding a
/// site table in a second place.
inline constexpr int TH_ONE_BIT_END = TH_RELAX;

inline constexpr unsigned long long TH_ALL_FORMS = (1ull << TH_BIT_COUNT) - 1ull;

/// The mask MEASURED on the authoring box (WSL2 / g++ 13.3, -O3 -march=native):
/// LERP_Y and TFUEL fused, POWER_ACC and both x-lerps not, RELAX contracting the
/// `(1 - w) * old` product.  It is the value the receipt COMPARES AGAINST, not a
/// value the run depends on: the production binary mines its own
/// (src/ThFormMiner.cpp) for exactly the reason CmfdOuterFormMiner.cpp exists --
/// a baked constant is a record of the machine it was measured on, not of the
/// physics, and the same probe on a box WITHOUT -march=native mines 0x0 because
/// an ISA with no FMA lets gcc contract nothing at all.
inline constexpr unsigned long long TH_FORMS_DEFAULT = 0x54ull;

RASBERY_TH_HD inline bool thForm(unsigned long long forms, int bit) {
    return ((forms >> bit) & 1ull) != 0ull;
}

/// Single-rounding a*b+c on both compilers.
RASBERY_TH_HD inline double thFma(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

/// Separately-ROUNDED product, immune to contraction into a following add on
/// both compilers: the device TU is built with --fmad=false, and the host build
/// pins the rounding with an optimization barrier because gcc fuses across
/// single-use temporaries and a named variable alone is not a barrier.
RASBERY_TH_HD inline double thMul(double a, double b) {
#if defined(__CUDA_ARCH__)
    return a * b;
#elif defined(__GNUC__)
    double p = a * b;
    asm volatile("" : "+x"(p));
    return p;
#else
    volatile double p = a * b;
    return p;
#endif
}

/// `a*b + c` under the mask: one site, two spellings, chosen by one bit.
RASBERY_TH_HD inline double thMulAdd(unsigned long long forms, int bit, double a,
                                     double b, double c) {
    return thForm(forms, bit) ? thFma(a, b, c) : thMul(a, b) + c;
}

// ---------------------------------------------------------------------------
// The water-property tables
// ---------------------------------------------------------------------------

/// One milk::Table, flattened.  `v` is row-major over y then x, which is
/// milk::Matrix's own layout, so the device reads the same element the host's
/// `values(row, col)` reads.
struct ThTable {
    const double* x  = nullptr; ///< [nx], the pressure axis (mod_*) or LPD axis (tf)
    const double* y  = nullptr; ///< [ny], the enthalpy axis (mod_*) or burnup axis (tf)
    const double* v  = nullptr; ///< [ny * nx]
    int           nx = 0;
    int           ny = 0;
};

/// milk::Table::FindLowerIndex, transcribed.  A binary search over a sorted
/// axis: integer arithmetic only, so this is exact by construction and carries
/// no form bit.
RASBERY_TH_HD inline int thFindLowerIndex(const double* axis, int n, double value) {
    int lo = 0;
    int hi = n - 1;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (axis[mid] <= value)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/// std::clamp's contract, spelled out: `lo` when v < lo, `hi` when hi < v, and v
/// otherwise -- NOT min(max(...)), which differs on a NaN.
RASBERY_TH_HD inline double thClamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (hi < v) return hi;
    return v;
}

/// milk::Table::Get, transcribed statement for statement.  CLAMPS off-axis
/// queries, which is the behaviour SolveTH's table-ceiling warning is about.
RASBERY_TH_HD inline double thTableGet(const ThTable& t, double x, double y,
                                       unsigned long long forms) {
    const int nx = t.nx;
    const int ny = t.ny;

    x = thClamp(x, t.x[0], t.x[nx - 1]);
    y = thClamp(y, t.y[0], t.y[ny - 1]);

    int ix = thFindLowerIndex(t.x, nx, x);
    int iy = thFindLowerIndex(t.y, ny, y);
    if (ix >= nx - 1) ix = nx - 2;
    if (iy >= ny - 1) iy = ny - 2;

    const double x0 = t.x[ix], x1 = t.x[ix + 1];
    const double y0 = t.y[iy], y1 = t.y[iy + 1];
    const double fx = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
    const double fy = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;

    const double z00 = t.v[static_cast<long long>(iy) * nx + ix];
    const double z01 = t.v[static_cast<long long>(iy) * nx + ix + 1];
    const double z10 = t.v[static_cast<long long>(iy + 1) * nx + ix];
    const double z11 = t.v[static_cast<long long>(iy + 1) * nx + ix + 1];

    const double z0 = thMulAdd(forms, TH_LERP_X0, fx, z01 - z00, z00);
    const double z1 = thMulAdd(forms, TH_LERP_X1, fx, z11 - z10, z10);
    return thMulAdd(forms, TH_LERP_Y, fy, z1 - z0, z0);
}

/// XSSet::GetTfuel, transcribed -- including the below-first-knot linear
/// continuation, which is NOT milk::Table behaviour and is the reason this is a
/// function rather than a table read.
RASBERY_TH_HD inline double thGetTfuel(const ThTable& tf, double burnup, double lpd,
                                       unsigned long long forms) {
    const double first_lpd = 50.0;
    const double safe_lpd  = (lpd > 0.0) ? lpd : 0.0; // std::max(0.0, LPD)
    const double query     = (safe_lpd > first_lpd) ? safe_lpd : first_lpd;
    const double rise      = thTableGet(tf, query, burnup, forms);
    if (safe_lpd < first_lpd) {
        const double scaled =
            thForm(forms, TH_TFUEL_LINEAR) ? rise * safe_lpd : thMul(rise, safe_lpd);
        return scaled / first_lpd;
    }
    return rise;
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

/// Everything one TH update reads and writes.  Pointers are DEVICE addresses in
/// the device arm and HOST addresses in the mining harness; the bodies never
/// learn which.
struct ThView {
    // --- shapes ---
    int nxy  = 0;
    int nz   = 0;
    int nxyz = 0;
    int ng   = 0;
    int kbc  = 0;
    int kec  = 0;

    // --- scalars from Geometry, resolved host-side exactly as SolveTH does ---
    double pressure            = 0.0;
    double inlet_h             = 0.0; ///< GetHmod(inlet_temp, pressure)
    double actual_power        = 0.0; ///< 1000.0 * rated_power * power_rate
    double total_flow          = 0.0; ///< 0 when the deck supplies a mass flux
    double input_mass_flux     = 0.0;
    int    use_input_mass_flux = 0;
    double fuel_temp_rise_scale = 1.0;
    double th_relaxation        = 1.0;
    double h_table_max          = 0.0; ///< mod_t.y_axis.back()

    // --- inputs ---
    const double* xskf    = nullptr; ///< [ig * nxyz + l]
    const double* phif    = nullptr; ///< [l * ng + ig]
    const double* vol     = nullptr; ///< [nxyz]
    const double* hmesh_x = nullptr; ///< [nxyz], hmesh(XDIR, l)
    const double* hmesh_y = nullptr; ///< [nxyz]
    const double* hz      = nullptr; ///< [nz]
    const int*    burn    = nullptr; ///< [nxyz], the integer burnup key

    // --- tables ---
    ThTable mod_t;
    ThTable mod_rho;
    ThTable tf;

    // --- scratch and outputs ---
    double* node_power = nullptr; ///< [nxyz]
    double* tful_old   = nullptr; ///< [nxyz]
    double* tmod_old   = nullptr; ///< [nxyz]
    double* dmod_old   = nullptr; ///< [nxyz]
    double* tful       = nullptr; ///< [nxyz], in and out
    double* tmod       = nullptr; ///< [nxyz], in and out
    double* dmod       = nullptr; ///< [nxyz], in and out
};

// ---------------------------------------------------------------------------
// Phase 1: the per-node power density
// ---------------------------------------------------------------------------

/// One node of UpdateTH's first loop.  The ig accumulation is in ascending
/// group order, which is the host's; the volume multiply is the host's separate
/// statement and is therefore NOT a multiply-add site.
RASBERY_TH_HD inline double thNodePower(const ThView& v, int lk, unsigned long long forms) {
    double power_density = 0.0;
    for (int ig = 0; ig < v.ng; ++ig)
        power_density = thMulAdd(forms, TH_POWER_ACC,
                                 v.xskf[static_cast<long long>(ig) * v.nxyz + lk],
                                 v.phif[static_cast<long long>(lk) * v.ng + ig],
                                 power_density);
    return power_density * v.vol[lk];
}

// ---------------------------------------------------------------------------
// Phase 2: the serial folds
// ---------------------------------------------------------------------------

/// `total_power` -- ascending node order, one lane.  Not a reduction: the host
/// loop's addition order is part of the answer.
RASBERY_TH_HD inline double thTotalPowerSerial(const double* node_power, int nxyz) {
    double total = 0.0;
    for (int lk = 0; lk < nxyz; ++lk)
        total += node_power[lk];
    return total;
}

/// SolveTH's `k_mid`, which is integer arithmetic and must not be re-derived
/// differently anywhere.
RASBERY_TH_HD inline int thKMid(int kbc, int kec) { return (kbc + kec - 1) / 2; }

/// `total_area` -- the mid-plane flow area, ascending channel order, one lane.
RASBERY_TH_HD inline double thTotalAreaSerial(const ThView& v, const double* node_power,
                                              int k_mid) {
    double total_area = 0.0;
    for (int l = 0; l < v.nxy; ++l) {
        const int idx = k_mid * v.nxy + l;
        if (node_power[idx] > 0.0)
            total_area += v.hmesh_x[idx] * v.hmesh_y[idx];
    }
    return total_area;
}

// ---------------------------------------------------------------------------
// Phase 3: one channel's axial sweep
// ---------------------------------------------------------------------------

/// What a channel reports back about the water-property table ceiling.  The
/// host prints ONE warning built from a running (count, worst, node) triple
/// scanned in channel order; a lane cannot do that, so it reports its own and
/// the reducer replays the host's first-wins tie-break over channels in order.
struct ThChannelOverflow {
    int    count = 0;
    double worst = 0.0;
    int    node  = -1;
};

/// One radial position of SolveTH's outer loop, transcribed.  `norm` and
/// `flow_per_channel` are the host's, computed once from the serial folds and
/// passed in -- a lane must not re-derive a global.
RASBERY_TH_HD inline ThChannelOverflow thChannelSweep(const ThView& v, int l, double norm,
                                                      double flow_per_channel,
                                                      unsigned long long forms) {
    ThChannelOverflow over;
    double            h_cur = v.inlet_h;

    // Bottom reflector
    for (int k = 0; k < v.kbc; ++k) {
        const int lk = l + k * v.nxy;
        v.tmod[lk]   = thTableGet(v.mod_t, v.pressure, h_cur, forms);
        v.dmod[lk]   = thTableGet(v.mod_rho, v.pressure, h_cur, forms);
    }

    // Fuel region
    for (int k = v.kbc; k < v.kec; ++k) {
        const int    lk     = l + k * v.nxy;
        const double P_node = v.node_power[lk] * norm;
        if (P_node <= 0.0) {
            v.tmod[lk] = thTableGet(v.mod_t, v.pressure, h_cur, forms);
            v.dmod[lk] = thTableGet(v.mod_rho, v.pressure, h_cur, forms);
            continue;
        }
        const double dh    = P_node / (flow_per_channel * v.hmesh_x[l] * v.hmesh_y[l]);
        const double h_out = h_cur + dh;
        const double h_avg =
            thForm(forms, TH_HAVG) ? 0.5 * (h_cur + h_out) : thMul(0.5, h_cur + h_out);
        if (h_avg > v.h_table_max) {
            ++over.count;
            if (h_avg > over.worst) {
                over.worst = h_avg;
                over.node  = lk;
            }
        }
        v.tmod[lk]       = thTableGet(v.mod_t, v.pressure, h_avg, forms);
        v.dmod[lk]       = thTableGet(v.mod_rho, v.pressure, h_avg, forms);
        const double lpd = 1000.0 * P_node / (62.0 * v.hz[k]);
        const double bu  = v.burn[lk] / 1000.0;
        v.tful[lk]       = thMulAdd(forms, TH_TFUEL, v.fuel_temp_rise_scale,
                                    thGetTfuel(v.tf, bu, lpd, forms), v.tmod[lk]);
        h_cur            = h_out;
    }

    // Top reflector
    for (int k = v.kec; k < v.nz; ++k) {
        const int lk = l + k * v.nxy;
        if (k == v.kec && h_cur > v.h_table_max) {
            ++over.count;
            if (h_cur > over.worst) {
                over.worst = h_cur;
                over.node  = lk;
            }
        }
        v.tmod[lk] = thTableGet(v.mod_t, v.pressure, h_cur, forms);
        v.dmod[lk] = thTableGet(v.mod_rho, v.pressure, h_cur, forms);
    }

    return over;
}

// ---------------------------------------------------------------------------
// Phase 4: the relaxation blend and the Doppler metric
// ---------------------------------------------------------------------------

/// `(1 - w) * old + w * new`, one array, one node.  The host writes the three
/// arrays in one loop body; the order between the three does not matter because
/// no one of them reads another.
RASBERY_TH_HD inline double thRelaxNode(double old_value, double new_value, double w,
                                        unsigned long long forms) {
    const unsigned long long form = (forms >> TH_RELAX) & 3ull;
    if (form == 1ull) return thFma(w, new_value, thMul(1.0 - w, old_value));
    if (form == 2ull) return thFma(1.0 - w, old_value, thMul(w, new_value));
    return thMul(1.0 - w, old_value) + thMul(w, new_value);
}

/// One node's contribution to delta_Dop.  MAX is associative and exact, so this
/// one reduction is allowed to be a tree.
RASBERY_TH_HD inline double thDeltaDopNode(double tful_new, double tful_old) {
    if (!(tful_new > 1.0e-30)) return 0.0;
    const double d = tful_new - tful_old;
    return (d < 0.0 ? -d : d) / tful_new;
}

} // namespace rasbery::th
