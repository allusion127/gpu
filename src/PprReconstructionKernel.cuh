#pragma once

// WP6 stage D: PPR::reconstructPinPower on the device.
//
// WHAT THIS IS.  A line-for-line transcription of PPR.cpp's
// reconstructPinPower() quadrature path -- the pin expansion, the deterministic
// normalisation, the radial fold and the two peaking factors -- in the host's
// statement order, on the host's operands, with the host's loop nesting.  It is
// included by exactly one translation unit (CudaPprBackend.cu), which is
// compiled with --fmad=false, so nvcc may not contract an a*b+c the host
// compiler did not.
//
// WHY IT IS WORTH PORTING AT ALL.  The floor_wall receipt put reconstruction at
// 8.48 ms/statepoint against reset+drive's 149.59, i.e. 5.4 % of PPR -- which
// is why c502856 left it alone and said so.  Two things changed:
//
//   * stages B and C shrink reset+drive, so the 5.4 % is a growing share;
//   * the reconstruction is the ONLY reason the seven coefficient arrays come
//     back at all.  p, a, c, bt, phic, q and l are 9,059,472 B/statepoint of
//     D2H that exists to feed a host loop.  Moving the consumer removes the
//     transfer, and the transfer is bigger than the loop.
//
// THE FORM FUNCTIONS ARE A REGISTRY, NOT AN UPLOAD.  fmap/gmap live in
// std::map-keyed Chiffon depletion points and the host interpolates a fresh
// pair per (plane, assembly) every statepoint.  Uploading the INTERPOLATED maps
// would be nxya*nz*(1+ng)*npina doubles -- 16 MB/statepoint at KNGR size, which
// costs more than the loop it saves.  So the DEPLETION POINTS go up once
// (they are library data and never move) and what travels per statepoint is
// three small arrays per (plane, assembly): the two bracketing slots and alpha,
// 24 B * nxya * nz = 63 KB.  The interpolation itself is the host's expression,
// evaluated on the device:
//
//     gmap = lo + alpha * (hi - lo)
//
// CLASS N1, and for exactly the reason the rest of this arm is: the expansion
// calls `exp` twelve times per (pin, overlap, group) and the device libm's is
// not glibc's.  Everything else -- the accumulation order over overlaps and
// groups, the 256-chunk normalisation partition and its ascending fold, the
// ascending-k radial sum -- is the host's, deliberately, so that the deviation
// stays attributable to `exp` and Gate A stays a measurement of one thing.
//
// THE PEAK REDUCTIONS ARE `fmax`, WHICH IS NOT AN APPROXIMATION OF `std::max`
// HERE, IT IS THE SAME ANSWER.  The host writes `frp = std::max(frp, v)` from
// 0.0, and `std::max(a, NaN)` returns `a`: NaN pins (the reconstruction pads
// invalid pins with quiet_NaN) are DROPPED, never propagated.  `fmax` drops NaN
// too, and max is exactly associative and commutative in floating point, so a
// chunked reduction gives the same bits as the host's sequential scan.  That is
// why this one reduction may be chunked freely while the corner sums may not.

#include <cmath>

namespace rasbery {
namespace ppr {

/// pch.h's `rsq2`, restated so this header does not include the host headers.
/// tools/test_ppr_gpu_contract.py compares the two literals.
constexpr double kReconRsq2 = 0.707106781186;

/// Everything the reconstruction kernels read.  Pointers are device pointers,
/// and every one of them is owned by PprBackend::Impl or borrowed from the
/// canonical set -- nothing here allocates.
struct ReconCtx {
    // --- shape ---
    int ng;
    int nxyz;
    int nxy;
    int nxya;
    int nz;
    int kbc;
    int kec;
    int ndiv;
    int ndiv2;
    int npins;
    int npina;
    int nchunk;
    int reconstruct_flux; ///< 1 = also fill pin flux with fmap
    /// WP6 stage F.  1 = RASBERY_PPR_MODE=master: the intranodal shape is the
    /// 13-term Legendre interpolant already in `c`, so the expansion is a
    /// 15-term dot product with the pre-computed products and p/a/bt are not
    /// read at all.  It is PPR.cpp's own branch inside the overlap loop.
    int mode_master;

    // --- geometry, uploaded once ---
    const int*           latol;   ///< [nxya * ndiv2], index la*ndiv2 + li
    const double*        vol;     ///< [nxyz]
    const double*        hz;      ///< [nz]
    const unsigned char* is_fuel; ///< [nxyz]

    // --- the quadrature table, uploaded once per (ndiv, npins) ---
    const int*    pin_off; ///< [npina + 1] offsets into the overlap arrays
    const int*    ovl_di;
    const int*    ovl_dj;
    const double* ovl_dxh;
    const double* ovl_dyh;
    const double* q_xq;  ///< [n_ovl * 9]
    const double* q_yq;  ///< [n_ovl * 9]
    const double* q_wt;  ///< [n_ovl * 9]
    const double* q_leg; ///< [n_ovl * 9 * 15]

    // --- the coefficients the drive left on the device ---
    const double* p;    ///< [nxyz * ng * 15]
    const double* a;    ///< [nxyz * ng * 8]
    const double* bt;   ///< [nxyz * ng]
    const double* c;    ///< [nxyz * ng * 15], MASTER mode's interpolant
    const double* phif; ///< [nxyz * ng]  (borrowed or uploaded)
    const double* xskf; ///< [ng * nxyz]

    // --- the form-function registry and this statepoint's index into it ---
    const double* gmap;        ///< [n_slots * npina]
    const double* fmap;        ///< [n_slots * ng * npina], may be null
    const int*    plane_lo;    ///< [nz * nxya], -1 = this (k, la) has no fuel
    const int*    plane_hi;
    const double* plane_alpha;

    // --- outputs ---
    double* pin_power;    ///< [nxya * nz * npina]
    double* pin_flux;     ///< [nxya * nz * ng * npina], may be null
    double* norm_partial; ///< [2 * nchunk] power then volume
    double* peak_partial; ///< [2 * nchunk] frp then fqp
    double* radial_power; ///< [nxya * npina]
    double* radial_hz;    ///< [nxya]
    double* scalars;      ///< [4]: inv_avg_power, frp, fqp, spare
};

// ---------------------------------------------------------------------------
// 1. the pin expansion
// ---------------------------------------------------------------------------

/// One thread per (fuel plane, assembly, pin) -- the host's innermost body,
/// whose (overlap, group) accumulation order is the dependence that had to be
/// preserved and the (k, la, py, px) nesting is the one that did not.
__global__ void kReconPins(ReconCtx x) {
    const int planes = (x.kec - x.kbc) * x.nxya;
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<long long>(planes) * x.npina) return;

    const int pin_idx = static_cast<int>(t % x.npina);
    const int plane   = static_cast<int>(t / x.npina);
    const int la      = plane % x.nxya;
    const int k       = x.kbc + plane / x.nxya;
    const int lka     = la + x.nxya * k;

    const int lo = x.plane_lo[lka];
    // The host `continue`s the whole (k, la) when it finds no fuel node, which
    // leaves the zeros fill_n put there.  The arrays are memset before this
    // launch, so returning is the same thing.
    if (lo < 0) return;
    const int    hi    = x.plane_hi[lka];
    const double alpha = x.plane_alpha[lka];

    const double inv_npins  = 1.0 / x.npins;
    const double pin_area   = inv_npins * inv_npins;
    const double area_coeff = 1.0 / (4.0 * x.ndiv * x.ndiv);

    double hom_flux[8]     = {};
    double power_integral  = 0.0;
    bool   has_valid_node  = false;

    const int ob = x.pin_off[pin_idx];
    const int oe = x.pin_off[pin_idx + 1];

    for (int o = ob; o < oe; ++o) {
        const int li = x.ovl_dj[o] * x.ndiv + x.ovl_di[o];
        const int l  = x.latol[la * x.ndiv2 + li];
        if (l < 0) continue;
        has_valid_node = true;
        const int lk   = l + x.nxy * k;

        const double dx_h = x.ovl_dxh[o];
        const double dy_h = x.ovl_dyh[o];
        const double* q_xq  = &x.q_xq[o * 9];
        const double* q_yq  = &x.q_yq[o * 9];
        const double* q_wt  = &x.q_wt[o * 9];
        const double* q_leg = &x.q_leg[static_cast<long long>(o) * 9 * 15];

        for (int g = 0; g < x.ng; ++g) {
            if (x.mode_master) {
                // PPR.cpp's MASTER branch, statement for statement.  NO `exp`
                // anywhere in it, so unlike the SENM path below this is a plain
                // reduction in the host's order on the host's operands -- class
                // B0 given its input, and its input is the N1 corner fluxes the
                // device CPB solve produced.
                const double* c_base = &x.c[(lk * 15 * x.ng) + (g * 15)];
                double        integ  = 0.0;
                for (int qq = 0; qq < 9; ++qq) {
                    const double* leg   = &q_leg[qq * 15];
                    double        cflux = 0.0;
                    for (int tt = 0; tt < 15; ++tt) cflux += c_base[tt] * leg[tt];
                    integ += q_wt[qq] * cflux;
                }
                const double flux_contrib = integ * dx_h * dy_h * area_coeff;
                hom_flux[g] += flux_contrib;
                power_integral += flux_contrib * x.xskf[g * x.nxyz + lk];
                continue;
            }

            const double  bt     = x.bt[lk * x.ng + g];
            const double* p_base = &x.p[(lk * 15 * x.ng) + (g * 15)];
            const double* a_base = &x.a[(lk * 8 * x.ng) + (g * 8)];

            double sx_arr[3], cx_arr[3], sxr_arr[3], cxr_arr[3];
            double sy_arr[3], cy_arr[3], syr_arr[3], cyr_arr[3];

            for (int qi = 0; qi < 3; ++qi) {
                const double ex   = exp(bt * q_xq[qi * 3]);
                const double rex  = 1.0 / ex;
                const double exr  = exp(bt * q_xq[qi * 3] * kReconRsq2);
                const double rexr = 1.0 / exr;
                sx_arr[qi]        = 0.5 * (ex - rex);
                cx_arr[qi]        = 0.5 * (ex + rex);
                sxr_arr[qi]       = 0.5 * (exr - rexr);
                cxr_arr[qi]       = 0.5 * (exr + rexr);
            }
            for (int qj = 0; qj < 3; ++qj) {
                const double ey   = exp(bt * q_yq[qj]);
                const double rey  = 1.0 / ey;
                const double eyr  = exp(bt * q_yq[qj] * kReconRsq2);
                const double reyr = 1.0 / eyr;
                sy_arr[qj]        = 0.5 * (ey - rey);
                cy_arr[qj]        = 0.5 * (ey + rey);
                syr_arr[qj]       = 0.5 * (eyr - reyr);
                cyr_arr[qj]       = 0.5 * (eyr + reyr);
            }

            double integ = 0.0;
            for (int qi = 0; qi < 3; ++qi) {
                for (int qj = 0; qj < 3; ++qj) {
                    const int     qq  = qi * 3 + qj;
                    const double* leg = &q_leg[qq * 15];

                    double pFlux = 0.0;
                    for (int tt = 0; tt < 15; ++tt) pFlux += p_base[tt] * leg[tt];

                    const double hFlux =
                        a_base[0] * sx_arr[qi] + a_base[1] * cx_arr[qi] +
                        a_base[2] * sy_arr[qj] + a_base[3] * cy_arr[qj] +
                        a_base[4] * sxr_arr[qi] * cyr_arr[qj] +
                        a_base[5] * sxr_arr[qi] * syr_arr[qj] +
                        a_base[6] * cxr_arr[qi] * syr_arr[qj] +
                        a_base[7] * cxr_arr[qi] * cyr_arr[qj];

                    integ += q_wt[qq] * (pFlux + hFlux);
                }
            }
            const double flux_contrib = integ * dx_h * dy_h * area_coeff;
            hom_flux[g] += flux_contrib;
            power_integral += flux_contrib * x.xskf[g * x.nxyz + lk];
        }
    }

    const long long pp = static_cast<long long>(lka) * x.npina + pin_idx;

    if (!has_valid_node) {
        // The host's NaN padding, and it is load-bearing: IO writes NaN for the
        // pins a plane does not cover, and Gate A compares the NaN POSITIONS
        // before it compares any value.
        // The host writes std::numeric_limits<double>::quiet_NaN(), whose object
        // representation is this constant.  Spelled as bits, not as nan(""),
        // because the pin map is written to HDF5 and an h5diff of two arms
        // compares payloads, not just is-a-NaN.
        const double nan_v = __longlong_as_double(0x7ff8000000000000LL);
        if (x.reconstruct_flux && x.pin_flux != nullptr)
            for (int g = 0; g < x.ng; ++g)
                x.pin_flux[(static_cast<long long>(lka) * x.ng + g) * x.npina + pin_idx] = nan_v;
        x.pin_power[pp] = nan_v;
        return;
    }

    const double inv_pin_area = 1.0 / pin_area;
    if (x.reconstruct_flux && x.pin_flux != nullptr && x.fmap != nullptr) {
        for (int g = 0; g < x.ng; ++g) {
            const double phi_hom = hom_flux[g] * inv_pin_area;
            const long long fi   = static_cast<long long>(g) * x.npina + pin_idx;
            const double f_lo    = x.fmap[static_cast<long long>(lo) * x.ng * x.npina + fi];
            const double f_hi    = x.fmap[static_cast<long long>(hi) * x.ng * x.npina + fi];
            const double fval    = f_lo + alpha * (f_hi - f_lo);
            x.pin_flux[(static_cast<long long>(lka) * x.ng + g) * x.npina + pin_idx] =
                phi_hom * fval;
        }
    }
    const double g_lo     = x.gmap[static_cast<long long>(lo) * x.npina + pin_idx];
    const double g_hi     = x.gmap[static_cast<long long>(hi) * x.npina + pin_idx];
    const double gmap_val = g_lo + alpha * (g_hi - g_lo);
    x.pin_power[pp]       = power_integral * inv_pin_area * gmap_val;
}

// ---------------------------------------------------------------------------
// 2. the normalisation -- the host's partition, the host's fold
// ---------------------------------------------------------------------------

__device__ inline int reconChunkBegin(int n, int nchunk, int c) {
    if (c >= nchunk) return n;
    const long long nn = n;
    return static_cast<int>((nn * c) / nchunk);
}

/// rasbery_det_chunks over nxyz, one thread per chunk, ascending inside it --
/// PPR.cpp's own `#pragma omp parallel for` over the same partition.
__global__ void kReconNormPartials(ReconCtx x) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= x.nchunk) return;
    const int lb = reconChunkBegin(x.nxyz, x.nchunk, c);
    const int le = reconChunkBegin(x.nxyz, x.nchunk, c + 1);

    double apw  = 0.0;
    double avol = 0.0;
    for (int lk = lb; lk < le; ++lk) {
        if (!x.is_fuel[lk]) continue;
        const double v = x.vol[lk];
        if (v <= 1.0e-20) continue;

        double node_power = 0.0;
        for (int g = 0; g < x.ng; ++g)
            node_power += x.xskf[g * x.nxyz + lk] * x.phif[lk * x.ng + g];

        apw += node_power * v;
        avol += v;
    }
    x.norm_partial[c]            = apw;
    x.norm_partial[x.nchunk + c] = avol;
}

/// The host folds the two partial arrays in ascending chunk index; so does
/// this, on one thread, for the same reason the corner fold does.
__global__ void kReconNormFold(ReconCtx x) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double nodal_power_sum = 0.0;
    double fuel_vol_sum    = 0.0;
    for (int c = 0; c < x.nchunk; ++c) nodal_power_sum += x.norm_partial[c];
    for (int c = 0; c < x.nchunk; ++c) fuel_vol_sum += x.norm_partial[x.nchunk + c];

    const double avg_nodal_power =
        (fuel_vol_sum > 1.0e-30) ? nodal_power_sum / fuel_vol_sum : 1.0;
    x.scalars[0] = 1.0 / avg_nodal_power;
}

__global__ void kReconScale(ReconCtx x) {
    const int planes = (x.kec - x.kbc) * x.nxya;
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<long long>(planes) * x.npina) return;
    const int pi    = static_cast<int>(t % x.npina);
    const int plane = static_cast<int>(t / x.npina);
    const int la    = plane % x.nxya;
    const int k     = x.kbc + plane / x.nxya;
    const int lka   = la + x.nxya * k;
    x.pin_power[static_cast<long long>(lka) * x.npina + pi] *= x.scalars[0];
}

// ---------------------------------------------------------------------------
// 3. the radial fold and the two peaking factors
// ---------------------------------------------------------------------------

/// PPR.cpp's `valid` test for (k, la): some sub-node of this assembly maps to a
/// fuel node on this plane.
__device__ inline bool reconPlaneValid(const ReconCtx& x, int k, int la) {
    for (int li = 0; li < x.ndiv2; ++li) {
        const int l = x.latol[la * x.ndiv2 + li];
        if (l >= 0 && x.is_fuel[l + x.nxy * k]) return true;
    }
    return false;
}

__global__ void kReconRadialHz(ReconCtx x) {
    const int la = blockIdx.x * blockDim.x + threadIdx.x;
    if (la >= x.nxya) return;
    double s = 0.0;
    for (int k = x.kbc; k < x.kec; ++k)
        if (reconPlaneValid(x, k, la)) s += x.hz[k];
    x.radial_hz[la] = s;
}

/// One thread per (assembly, pin), summing over k ASCENDING -- the host's own
/// order, because `radial_power[la][pi] += ppower * hz_k` is accumulated with k
/// as the outer loop.
__global__ void kReconRadial(ReconCtx x) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<long long>(x.nxya) * x.npina) return;
    const int pi = static_cast<int>(t % x.npina);
    const int la = static_cast<int>(t / x.npina);

    double s = 0.0;
    for (int k = x.kbc; k < x.kec; ++k) {
        if (!reconPlaneValid(x, k, la)) continue;
        const int lka = la + x.nxya * k;
        s += x.pin_power[static_cast<long long>(lka) * x.npina + pi] * x.hz[k];
    }
    const double hz_sum = x.radial_hz[la];
    // `radial_hz <= 0` is the host's `continue`, i.e. this (la, pi) never
    // reaches the max at all.  0.0 is the identity of a max whose accumulator
    // starts at 0.0, so writing it is the same thing.
    x.radial_power[t] = (hz_sum > 0.0) ? s * (1.0 / hz_sum) : 0.0;
}

/// Both peaks, chunked.  fmax is exactly associative and commutative and drops
/// NaN, so the partition does not move the answer -- unlike the corner sums,
/// which is why that one is pinned to 256 and this one is not.
__global__ void kReconPeakPartials(ReconCtx x) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= x.nchunk) return;

    const long long nrad = static_cast<long long>(x.nxya) * x.npina;
    double          frp  = 0.0;
    for (long long i = (nrad * c) / x.nchunk; i < (nrad * (c + 1)) / x.nchunk; ++i)
        frp = fmax(frp, x.radial_power[i]);

    const long long planes = static_cast<long long>(x.kec - x.kbc) * x.nxya;
    const long long nfq    = planes * x.npina;
    double          fqp    = 0.0;
    for (long long t = (nfq * c) / x.nchunk; t < (nfq * (c + 1)) / x.nchunk; ++t) {
        const int pi    = static_cast<int>(t % x.npina);
        const int plane = static_cast<int>(t / x.npina);
        const int la    = plane % x.nxya;
        const int k     = x.kbc + plane / x.nxya;
        const int lka   = la + x.nxya * k;
        fqp = fmax(fqp, x.pin_power[static_cast<long long>(lka) * x.npina + pi]);
    }

    x.peak_partial[c]            = frp;
    x.peak_partial[x.nchunk + c] = fqp;
}

__global__ void kReconPeakFold(ReconCtx x) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double frp = 0.0, fqp = 0.0;
    for (int c = 0; c < x.nchunk; ++c) frp = fmax(frp, x.peak_partial[c]);
    for (int c = 0; c < x.nchunk; ++c) fqp = fmax(fqp, x.peak_partial[x.nchunk + c]);
    x.scalars[1] = frp;
    x.scalars[2] = fqp;
}

} // namespace ppr
} // namespace rasbery
