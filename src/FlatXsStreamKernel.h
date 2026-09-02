#pragma once

// WP23: the flat-XS BRANCH STREAM RESOLVER, as a shared host/device body.
//
// ---------------------------------------------------------------------------
// WHAT THIS IS THE DEVICE TWIN OF
// ---------------------------------------------------------------------------
//
// XSSet::BuildFlatXsStream (src/XSSet.cpp) turns one statepoint's unrodded node
// list into the flat (did, x, scale) application stream that kernelFlatXsCta
// consumes.  Per node it does exactly four things, in this order:
//
//   1. the three SCALAR BRANCH coordinates -- boron*wvfr*bppm, sqrt(tful), dmod
//      -- each interpolated between the node's two burnup-bracket delta ids,
//      with the unrodded weight `wu = 1 - hw` and then, when the node carries a
//      rodded-depletion twin, the same three against the partner tables at `hw`;
//   2. a TWO-ELEMENT WORKSPACE PROBE: the XSAF Pu-239 and B-10 thermal micro
//      cross sections after the branch prefix built in (1), which is what
//      NodeSpectralIndex is centred on;
//   3. the SPECTRAL-HISTORY terms of the node's own library
//      (ResolveSpectralHistoryDeltas at weight `wu`), each of whose coordinates
//      is one of twenty enumerated FORMS;
//   4. the same terms against the history twin's library at weight `hw`.
//
// Steps 1-4 are PER NODE AND ONLY PER NODE.  `_flatxs_node_apps[i]` is written
// by one OpenMP iteration and read by no other; the thread_local `micprobe` /
// `p_did` / `hist` workspaces are cleared at the top of every node; the ONLY
// serial pass in BuildFlatXsStream is the concatenation, which is a copy in node
// order and introduces no arithmetic.  There is no running accumulator across
// nodes.  That is the property that makes a device port reachable at B0 at all,
// and it is asserted by tools/test_flatxs_stream_contract.py rather than
// assumed here.
//
// ---------------------------------------------------------------------------
// THE PACKING IS NOT THE HOST'S, AND THAT IS DELIBERATE
// ---------------------------------------------------------------------------
//
// The host emits a DENSE stream and an exclusive scan (`node_off[i]` is the
// running length).  A device port of a dense scan needs a two-pass count/scan or
// an atomic bump, and BOTH change the ORDER in which a node's entries land --
// which is the one thing the CTA kernel's determinism contract forbids.
//
// So this arm does not pack: node `i` owns the FIXED slot
// `[i*stride, i*stride + cnt_i)` and writes `node_off[i] = i*stride` itself.
// The consumer already reads `node_off`/`node_cnt` and never assumes they are
// contiguous, so nothing downstream changes; a node's OWN entries stay in the
// host's exact order, which is what exactness depends on.  The cost is
// `stride * n_nodes` of slack in three device arrays and nothing else.
//
// ---------------------------------------------------------------------------
// EXACTNESS: WHY THIS FILE CANNOT CLAIM B0 THE WAY FlatXsKernel.h DOES
// ---------------------------------------------------------------------------
//
// FlatXsKernel.h opens with the sentence this header has to qualify: "everything
// transcendental ... happens on the HOST ... glibc's log() is not correctly
// rounded and does not match CUDA's, so any design that evaluates coordinates on
// the device is unverifiable; this one never has to."  THIS one has to -- that
// is the whole work package -- so the qualification is stated in the two places
// it bites and nowhere hidden:
//
//   * SEVEN of the twenty forms call `log` or `cbrt` (kFormUsesLibm below).  On
//     a deck whose library uses one of them the arm is CLASS N1 by construction,
//     ~1 ulp on the coordinate, and the receipt names the form that forced it.
//     The other thirteen are +, -, *, / and sqrt only -- sqrt is correctly
//     rounded in IEEE-754 on both sides -- and are B0-CAPABLE.
//   * B0-capable is not B0.  The three multiply-add sites below (the burnup
//     interpolations) have a contraction choice, and this work package does NOT
//     ship a miner for them; `kStreamFormsDefault` is 0 (nothing fused), which is
//     a GUESS about what gcc did and not a measurement.  Until a miner in the
//     shape of src/ThFormMiner.cpp pins these three bits against a quotation, the
//     arm is N1 even on a libm-free library.  Said again in
//     src/FlatXsStreamReceipt.h, in the string a gate script reads.
//
// This header must stay compilable by both g++ and nvcc, and it must not include
// anything from Chiffon: `Model.h` reaches HighFive and HDF5, which nvcc has no
// business parsing.  The coordinate enumerators are therefore RESTATED here as
// plain ints and the restatement is held to Chiffon's by static_assert in
// src/XSSet.cpp -- the one translation unit that legitimately sees both.

#include "FlatXsKernel.h" // FlatXsView, DeltaMeta, StaticForms, flatxsProbeMicElement

namespace rasbery::flatxs_stream {

using xsrecon::NG;
using xsrecon::NISO;
using xsrecon::xsrFma;
using xsrecon::xsrMul;

/// The three scalar branch axes, in the order UpdateUnroddedNodeXS applies them.
constexpr int kNumScalarBranches = 3;

/// Chiffon::SpectralCoordinate, restated (see the header note).  The gaps at
/// 6/10/11 are Chiffon's own -- an experimental build wrote them with different
/// semantics and reusing them would apply a coefficient to the wrong axis.
enum Form : int {
    kDensity                  = 0,
    kLogDensity               = 1,
    kThermalWeighted          = 2,
    kFastWeighted             = 3,
    kFluxRatioInteraction     = 4,
    kSqrtDensity              = 5,
    kSpectralIndex            = 7,
    kSpectralIndexInteraction = 8,
    kRelativeBurnRatio        = 9,
    kBppmInteraction          = 12,
    kTfulInteraction          = 13,
    kDmodInteraction          = 14,
    kLogDeviationSquared      = 15,
    kFissileFraction          = 16,
    kInverseRatio             = 17,
    kCubeRootRatio            = 18,
    kSaturatingRatio          = 19,
    kBppmRodAge               = 20,
    kTfulRodAge               = 21,
    kDmodRodAge               = 22,
    /// One past the largest enumerator: the width of every per-form census array.
    kFormCount                = 23,
};

/// Chiffon::SPECTRAL_LOG_DENSITY_FLOOR / ROD_AGE_SCALE, restated for the same
/// reason the enumerators are, and static_asserted against Chiffon in XSSet.cpp.
constexpr double kSpectralLogDensityFloor = 1.0e-12;
constexpr double kRodAgeScale             = 1.0e-21;

/// Chiffon::BranchAxisOf.
RASBERY_XSR_HD constexpr int branchAxisOf(int coord) {
    return coord == kBppmInteraction   ? 0
           : coord == kTfulInteraction ? 1
           : coord == kDmodInteraction ? 2
                                       : -1;
}

/// Chiffon::RodAgeAxisOf.
RASBERY_XSR_HD constexpr int rodAgeAxisOf(int coord) {
    return coord == kBppmRodAge   ? 0
           : coord == kTfulRodAge ? 1
           : coord == kDmodRodAge ? 2
                                  : -1;
}

/// Chiffon::RatioFormOf's domain.
RASBERY_XSR_HD constexpr bool isRatioForm(int coord) {
    return coord == kLogDeviationSquared || coord == kInverseRatio ||
           coord == kCubeRootRatio || coord == kSaturatingRatio;
}

/// THE SEVEN FORMS WHOSE COORDINATE CALLS A LIBM FUNCTION, and therefore the
/// seven that make a run of this arm CLASS N1 no matter how well the contraction
/// mask is mined.  `log` (glibc vs CUDA: not correctly rounded on either side,
/// differing arguments-to-1-ulp) and `cbrt` are the whole list.
///
/// kSpectralIndex / kSpectralIndexInteraction are HERE and it is not obvious
/// why: their coordinate is NodeSpectralIndex, whose last line is
/// `log(now / base)`.
RASBERY_XSR_HD constexpr bool formUsesLibm(int coord) {
    return coord == kLogDensity || coord == kFluxRatioInteraction ||
           coord == kSpectralIndex || coord == kSpectralIndexInteraction ||
           coord == kRelativeBurnRatio || coord == kLogDeviationSquared ||
           coord == kCubeRootRatio;
}

/// Every enumerator this body evaluates.  A library carrying a coordinate that
/// is NOT here is refused BY NAME with a counted host fallback, which is the
/// whole point of having a list rather than a `default:` arm.
RASBERY_XSR_HD constexpr bool formImplemented(int coord) {
    return coord == kDensity || coord == kLogDensity || coord == kThermalWeighted ||
           coord == kFastWeighted || coord == kFluxRatioInteraction ||
           coord == kSqrtDensity || coord == kSpectralIndex ||
           coord == kSpectralIndexInteraction || coord == kRelativeBurnRatio ||
           branchAxisOf(coord) >= 0 || rodAgeAxisOf(coord) >= 0 ||
           isRatioForm(coord) || coord == kFissileFraction;
}

/// Human name of a form, for the receipt and for the refusal message.  Host-only
/// by construction (it returns a string literal); never called from a kernel.
inline const char* formName(int coord) {
    switch (coord) {
    case kDensity:                  return "Density";
    case kLogDensity:               return "LogDensity";
    case kThermalWeighted:          return "ThermalWeighted";
    case kFastWeighted:             return "FastWeighted";
    case kFluxRatioInteraction:     return "FluxRatioInteraction";
    case kSqrtDensity:              return "SqrtDensity";
    case kSpectralIndex:            return "SpectralIndex";
    case kSpectralIndexInteraction: return "SpectralIndexInteraction";
    case kRelativeBurnRatio:        return "RelativeBurnRatio";
    case kBppmInteraction:          return "BppmInteraction";
    case kTfulInteraction:          return "TfulInteraction";
    case kDmodInteraction:          return "DmodInteraction";
    case kLogDeviationSquared:      return "LogDeviationSquared";
    case kFissileFraction:          return "FissileFraction";
    case kInverseRatio:             return "InverseRatio";
    case kCubeRootRatio:            return "CubeRootRatio";
    case kSaturatingRatio:          return "SaturatingRatio";
    case kBppmRodAge:               return "BppmRodAge";
    case kTfulRodAge:               return "TfulRodAge";
    case kDmodRodAge:               return "DmodRodAge";
    default:                        return "Unassigned";
    }
}

// ---------------------------------------------------------------------------
// The refusal ladder
// ---------------------------------------------------------------------------
//
// A refused node is NOT an error and NOT a clamp: `node_cnt[i]` comes back
// negative, the host resolves that node with the very function the flag-off run
// uses (XSSet::ResolveNodeApplications) and writes it into the same slot, and
// the receipt counts it by reason.  An arm that silently produced a shorter
// stream would produce finite, plausible, wrong cross sections.
enum Refusal : int {
    kRefusalNone    = 0,
    kRefusalCapacity = 1, ///< the node needs more entries than `stride`
    kRefusalForm     = 2, ///< a coordinate enumerator formImplemented() rejects
    kRefusalModel    = 3, ///< the model's reference trajectory did not flatten
    kRefusalRodded   = 4, ///< a rodded node reached a body that serves unrodded only
    kRefusalCount    = 5,
};

/// `node_cnt` encoding.  Non-negative is a length; negative is `-1 - reason`, so
/// the reason survives the one array the host has to read back anyway.
RASBERY_XSR_HD constexpr int encodeRefusal(int reason) { return -1 - reason; }
RASBERY_XSR_HD constexpr int decodeRefusal(int cnt) { return cnt < 0 ? -1 - cnt : 0; }

inline const char* refusalName(int reason) {
    switch (reason) {
    case kRefusalNone:     return "none";
    case kRefusalCapacity: return "capacity";
    case kRefusalForm:     return "form";
    case kRefusalModel:    return "model";
    case kRefusalRodded:   return "rodded";
    default:               return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Contraction sites
// ---------------------------------------------------------------------------

/// The three multiply-add sites in this body.  All three are the SAME shape --
/// a burnup lerp `acc += f * (hi - lo)` -- and gcc still chooses per statement,
/// which is the finding the xsrecon campaign paid for and the reason these are
/// three bits and not one.
enum StreamFormBit : unsigned {
    FS_REFDENS  = 1u << 0, ///< referenceDensity:   value += fraction * (hi - lo)
    FS_REFDENS0 = 1u << 1, ///< referenceDensity0:  v     += f * (hi - v)
    FS_REFCOND  = 1u << 2, ///< referenceCondition: value += fraction * (hi - value)
    FS_ALL      = 0x7u,
};

/// NOT MINED.  Zero means "nothing fused", which is what gcc does on an ISA
/// without FMA and what a reader must not mistake for a measurement -- see the
/// header note and src/FlatXsStreamReceipt.h.  The value is carried at RUNTIME
/// (not baked) so a miner can be added later without changing one call site.
constexpr unsigned kStreamFormsDefault = 0u;

/// Runtime-mask contraction policy, same rounding guarantees per arm as
/// flatxs::StaticForms.
struct StreamForms {
    unsigned mask = kStreamFormsDefault;

    RASBERY_XSR_HD double ma(unsigned bit, double a, double b, double c) const {
#if defined(__CUDA_ARCH__)
        return (mask & bit) ? fma(a, b, c) : a * b + c;
#else
        return (mask & bit) ? xsrFma(a, b, c) : xsrMul(a, b) + c;
#endif
    }
};

// ---------------------------------------------------------------------------
// The views
// ---------------------------------------------------------------------------

/// The library half: everything ResolveSpectralHistoryDeltas reads out of
/// `_lib`, flattened.  Immutable for the life of a run, uploaded once, keyed by
/// the library's own content digest generation.
///
/// THE ctype-0 COLLAPSE, and why it is sound.  BuildFlatXsStream is called with
/// the UNRODDED node list only, so `currentCtype` in the host resolver is
/// literally `UsesRodXS(l) ? _ctyp[l] : 0` == 0, which makes `ctypeIndex ==
/// ctypeIndex0`, `referenceBase == referenceBase0` and `referenceBurnups` the
/// ctype-0 key list -- for every node this body will ever see.  So only the
/// ctype-0 row of each model is flattened here.  A rodded node reaching this
/// body is kRefusalRodded, not a wrong answer.
struct StreamLibView {
    // spectral-history corrections, concatenated over models
    const int* model_sh_off;  ///< [nmodel + 1], into the seven arrays below
    const int* sh_iso;        ///< [ncorr] term.isotope
    const int* sh_partner;    ///< [ncorr] term.partner, or >= niso when unused
    const int* sh_coord;      ///< [ncorr] term.coordinate
    const int* sh_delta_base; ///< [ncorr]
    const int* sh_rod_scaled; ///< [ncorr]
    const int* sh_burn_off;   ///< [ncorr] into sh_burnups
    const int* sh_burn_cnt;   ///< [ncorr]
    const int* sh_burnups;    ///< concatenated burnup key lists

    // ctype-0 reference trajectory, per model
    const int*       refr0_key_off; ///< [nmodel] into refr0_keys
    const int*       refr0_key_cnt; ///< [nmodel]; MAY be 0 with the ctype present
    /// [nmodel]: 1 when findCtype(refr_ctyp[m], 0) >= 0.  IT IS NOT THE SAME
    /// FACT AS `refr0_key_cnt > 0`, and conflating the two is a wrong answer,
    /// not a refusal: the host returns with NO terms when the ctype is absent,
    /// and emits terms whose reference density is 0.0 when the ctype is there
    /// with an empty burn list.
    const int*       refr0_present;
    const int*       refr0_keys;    ///< concatenated
    const long long* refr0_base;    ///< [nmodel] flat id of the ctype-0 block
    const long long* refr_burn_stride; ///< [nmodel]
    const int*       history_partner;  ///< [nmodel], -1 when the model has no twin
    const int*       model_ok;         ///< [nmodel], 0 -> kRefusalModel

    // library tables
    const double* lib_iden;         ///< [nrow * niso]
    const double* lib_burn;         ///< [nrow] GWd/THM
    const double* lib_ref_branch_x; ///< [nrow * 3]

    int nmodel;
    int niso; ///< Isotope::niso, which is a REGISTRY size and not xsrecon::NISO
};

/// Element counts of the arrays StreamLibView points at.  The view carries
/// pointers only -- it is the same struct on the host and on the device, exactly
/// like thgpu::TableView -- so the LENGTHS have to travel beside it for the
/// backend to know what to copy.  Kept as its own struct rather than fields on
/// the view so that a device-side body cannot read a length it has no business
/// knowing and start bounds-checking the host's data structures.
struct StreamLibShape {
    long long n_corr      = 0; ///< entries in sh_* (the seven parallel arrays)
    long long n_burnups   = 0; ///< entries in sh_burnups
    long long n_refr0_key = 0; ///< entries in refr0_keys
    long long n_rows      = 0; ///< reference flat ids: lib_burn / lib_ref_branch_x rows
    long long n_iden      = 0; ///< entries in lib_iden (n_rows * niso, as loaded)
    int       nmodel      = 0;
};

/// The per-node half.  `wvfr`, `dmod`, `bppm` and `iden` are NOT here: they are
/// already in flatxs::FlatXsView, already device-resident, and a second opinion
/// about where they live is exactly the bug this arm must not introduce.
struct StreamNodeView {
    const int*    comp;    ///< [nxyz] model index
    const int*    burn;    ///< [nxyz] burnup key
    const double* hw;      ///< [nxyz] history blend weight, or null for none
    const double* rodfrac; ///< [nxyz] Geometry::rod_fraction
    const double* tful;    ///< [nxyz]
    const double* phif;    ///< [nxyz * ng], [l*ng + ig]

    // PrecomputeBranchCoefficients' per-branch burnup brackets, flattened
    // branch-major: [branch * nxyz + l].
    const int*    delta_lo;
    const int*    delta_hi;
    const double* delta_frac;
    const int*    delta_lo_p;   ///< null when the deck carries no history twin
    const int*    delta_hi_p;
    const double* delta_frac_p;

    int ng;
    int i_pu239; ///< Isotope::iPu239 -- a REGISTRY index, so it travels as data
    int i_b10;   ///< Isotope::iB10
};

/// Where the resolved stream lands.  `stride` slots per node; see the packing
/// note at the top.
struct StreamOutView {
    int*    node_off;
    int*    node_cnt;
    int*    stream_did;
    double* stream_x;
    double* stream_scale;
    int     stride;
};

/// One statepoint's ask, as it crosses from XSSet into the backend.  HOST
/// POINTERS throughout -- the backend uploads what it must and rebinds the two
/// views onto its own device block, exactly as thgpu::TableView / GeomView work.
///
/// IT TRAVELS AS A POINTER PARAMETER OF solveFlatXs, and NOT as a field of
/// FlatXsView, because of an ordering fact that took a rewrite to notice: the
/// stream builder reads the reference micx block, the library coefficient
/// tables, `iden` and the three per-node coordinate columns -- all of which
/// solveFlatXs itself is what makes resident.  A separate public entry point
/// called before it would run against null device pointers on the first call of
/// a run.  So the build is a PHASE of solveFlatXs, ordered after the residency
/// work and before the flat-XS launch, and a null request is the flag-off path
/// down to the byte.
struct StreamRequest {
    StreamLibView      lib{};
    StreamLibShape     shape{};
    unsigned long long lib_generation = 0;
    StreamNodeView     nodes{};
    /// Slots per node.  The caller computes a STATIC UPPER BOUND for it (12
    /// scalar-branch entries plus two per spectral term per pass), so
    /// kRefusalCapacity is unreachable rather than merely unlikely.
    int                stride = 0;
};

// ---------------------------------------------------------------------------
// Bracket search -- std::lower_bound semantics, transcribed
// ---------------------------------------------------------------------------

/// First index with `keys[i] >= key`, or `n` -- std::lower_bound on a sorted
/// int range.  Integer arithmetic throughout: no rounding to disagree about.
RASBERY_XSR_HD inline int fsLowerBound(const int* keys, int n, int key) {
    int lo = 0;
    int len = n;
    while (len > 0) {
        const int half = len / 2;
        const int mid  = lo + half;
        if (keys[mid] < key) {
            lo  = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return lo;
}

/// findLoBurn (src/XSSet.cpp), transcribed.
RASBERY_XSR_HD inline int fsFindLoBurn(const int* keys, int n, int key) {
    if (n <= 0) return -1;
    const int it = fsLowerBound(keys, n, key);
    if (it == n) return n - 1;
    if (it == 0) return 0;
    return it - 1;
}

/// findHiBurn (src/XSSet.cpp), transcribed.
RASBERY_XSR_HD inline int fsFindHiBurn(const int* keys, int n, int key) {
    if (n <= 0) return -1;
    const int it = fsLowerBound(keys, n, key);
    if (it == n) return n - 1;
    return it;
}

RASBERY_XSR_HD inline double fsMax(double a, double b) { return a > b ? a : b; }
RASBERY_XSR_HD inline double fsAbs(double a) { return a < 0.0 ? -a : a; }

RASBERY_XSR_HD inline double fsSqrt(double a) {
#if defined(__CUDA_ARCH__)
    return sqrt(a);
#else
    return std::sqrt(a);
#endif
}

RASBERY_XSR_HD inline double fsLog(double a) {
#if defined(__CUDA_ARCH__)
    return log(a);
#else
    return std::log(a);
#endif
}

RASBERY_XSR_HD inline double fsCbrt(double a) {
#if defined(__CUDA_ARCH__)
    return cbrt(a);
#else
    return std::cbrt(a);
#endif
}

/// Chiffon::RatioFormOf, transcribed.  `now` and `ref` arrive already floored.
RASBERY_XSR_HD inline double fsRatioFormOf(int coord, double now, double ref) {
    if (coord == kLogDeviationSquared) {
        const double d = fsLog(now / ref);
        return d * d;
    }
    if (coord == kInverseRatio) return ref / now;
    if (coord == kCubeRootRatio) return fsCbrt(now / ref);
    return now / (1.0 + now / ref);
}

// ---------------------------------------------------------------------------
// The reference-trajectory lookups
// ---------------------------------------------------------------------------

/// XSSet::ResolveSpectralHistoryDeltas' `referenceDensity` lambda.
RASBERY_XSR_HD inline double fsReferenceDensity(const StreamLibView& lib, int model,
                                                int isotope, int burnup,
                                                const StreamForms& pol) {
    if (isotope >= lib.niso) return 0.0;
    const int* keys = lib.refr0_keys + lib.refr0_key_off[model];
    const int  nkey = lib.refr0_key_cnt[model];
    const int  loIndex = fsFindLoBurn(keys, nkey, burnup);
    const int  hiIndex = fsFindHiBurn(keys, nkey, burnup);
    if (loIndex < 0 || hiIndex < 0) return 0.0;

    const long long base   = lib.refr0_base[model];
    const long long stride = lib.refr_burn_stride[model];
    const long long lo     = base + static_cast<long long>(loIndex) * stride;
    const long long hi     = base + static_cast<long long>(hiIndex) * stride;

    double value = lib.lib_iden[lo * lib.niso + isotope];
    if (lo != hi && lib.lib_burn[hi] != lib.lib_burn[lo]) {
        const double fraction = (static_cast<double>(burnup) / 1000.0 - lib.lib_burn[lo]) /
                                (lib.lib_burn[hi] - lib.lib_burn[lo]);
        value = pol.ma(FS_REFDENS, fraction,
                       lib.lib_iden[hi * lib.niso + isotope] -
                           lib.lib_iden[lo * lib.niso + isotope],
                       value);
    }
    return value;
}

/// The `referenceDensity0` lambda.  IT IS A DIFFERENT SPELLING OF THE SAME
/// NUMBER and it is kept as its own function for exactly that reason: the host
/// writes `v += f * (hi - v)` there and `value += fraction * (hi - lo)` above,
/// and gcc contracts the two independently.  Collapsing them would be a
/// simplification that changes a bit.
RASBERY_XSR_HD inline double fsReferenceDensity0(const StreamLibView& lib, int model,
                                                 int isotope, int burnup,
                                                 const StreamForms& pol) {
    if (isotope >= lib.niso) return 0.0;
    const int* keys = lib.refr0_keys + lib.refr0_key_off[model];
    const int  nkey = lib.refr0_key_cnt[model];
    const int  lo_i = fsFindLoBurn(keys, nkey, burnup);
    const int  hi_i = fsFindHiBurn(keys, nkey, burnup);
    if (lo_i < 0 || hi_i < 0) return 0.0;

    const long long base   = lib.refr0_base[model];
    const long long stride = lib.refr_burn_stride[model];
    const long long lb     = base + static_cast<long long>(lo_i) * stride;
    const long long hb     = base + static_cast<long long>(hi_i) * stride;

    double v = lib.lib_iden[lb * lib.niso + isotope];
    if (lb != hb && lib.lib_burn[hb] != lib.lib_burn[lb]) {
        const double f = (static_cast<double>(burnup) / 1000.0 - lib.lib_burn[lb]) /
                         (lib.lib_burn[hb] - lib.lib_burn[lb]);
        v = pol.ma(FS_REFDENS0, f, lib.lib_iden[hb * lib.niso + isotope] - v, v);
    }
    return v;
}

/// The `referenceCondition` lambda: the bppm/tful/dmod the reference depletion
/// ran at, interpolated on the same bracket.
RASBERY_XSR_HD inline double fsReferenceCondition(const StreamLibView& lib, int model,
                                                  int axis, int burnup,
                                                  const StreamForms& pol) {
    const int* keys = lib.refr0_keys + lib.refr0_key_off[model];
    const int  nkey = lib.refr0_key_cnt[model];
    const int  loIndex = fsFindLoBurn(keys, nkey, burnup);
    const int  hiIndex = fsFindHiBurn(keys, nkey, burnup);
    if (loIndex < 0 || hiIndex < 0) return 0.0;

    const long long base   = lib.refr0_base[model];
    const long long stride = lib.refr_burn_stride[model];
    const long long lo     = base + static_cast<long long>(loIndex) * stride;
    const long long hi     = base + static_cast<long long>(hiIndex) * stride;

    double value = lib.lib_ref_branch_x[lo * 3 + axis];
    if (lo != hi && lib.lib_burn[hi] != lib.lib_burn[lo]) {
        const double fraction = (static_cast<double>(burnup) / 1000.0 - lib.lib_burn[lo]) /
                                (lib.lib_burn[hi] - lib.lib_burn[lo]);
        value = pol.ma(FS_REFCOND, fraction,
                       lib.lib_ref_branch_x[hi * 3 + axis] - value, value);
    }
    return value;
}

/// XSSet::NodeFluxShare, transcribed.
RASBERY_XSR_HD inline double fsNodeFluxShare(const StreamNodeView& nd, int l, bool thermal) {
    const int ng = nd.ng;
    double    tot = 0.0;
    for (int ig = 0; ig < ng; ++ig)
        tot += nd.phif[static_cast<long long>(l) * ng + ig];
    if (!(tot > 0.0)) return 0.0;
    const int ig = thermal ? ng - 1 : 0;
    return nd.phif[static_cast<long long>(l) * ng + ig] / tot;
}

/// XSSet::NodeSpectralIndex, transcribed, with the workspace probe values passed
/// in rather than re-derived: the host reads them out of a two-element `micprobe`
/// and this body has them in registers.
RASBERY_XSR_HD inline double fsNodeSpectralIndex(const flatxs::FlatXsView& v,
                                                 const StreamNodeView& nd, int l,
                                                 double probe_pu, double probe_b) {
    const int ng  = nd.ng;
    const int ith = ng - 1;
    const double num = probe_pu;
    const double den = probe_b;
    if (!(fsAbs(den) > 1.0e-30)) return 0.0;
    const double bnum = flatxs::fxsRefMic(
        v, 1, flatxs::block_layout::mic(v.nxyz, l, nd.i_pu239 * ng + ith));
    const double bden = flatxs::fxsRefMic(
        v, 1, flatxs::block_layout::mic(v.nxyz, l, nd.i_b10 * ng + ith));
    if (!(fsAbs(bden) > 1.0e-30)) return 0.0;
    const double now  = num / den;
    const double base = bnum / bden;
    return (now > 1.0e-30 && base > 1.0e-30) ? fsLog(now / base) : 0.0;
}

/// XSSet::FineRodThermalFluenceAverage, for the nodes this body serves.
///
/// IT IS EXACTLY ZERO AND THE REASON IS STRUCTURAL, not a shortcut: the host
/// function returns 0.0 on its first line when `ctype <= 0`, and the ctype of
/// every node in the unrodded list is 0 (see the ctype-0 collapse note on
/// StreamLibView).  Kept as a named function so that the day a rodded node is
/// admitted, there is one place that has to grow the fine-mesh average, and the
/// three rod-age forms do not have to be found again.
RASBERY_XSR_HD inline double fsRodThermalFluence(int /*l*/, int currentCtype) {
    return currentCtype <= 0 ? 0.0 : 0.0;
}

/// XSSet::RodBlendWeight in its production mode.  CHIFFON_PROBE_BLEND selects
/// two other modes that read RoddedPuFraction; the host refuses the whole arm
/// when that variable is set rather than porting a probe.
RASBERY_XSR_HD inline double fsRodBlendWeight(const StreamNodeView& nd, int l) {
    return nd.rodfrac[l];
}

// ---------------------------------------------------------------------------
// The body
// ---------------------------------------------------------------------------

/// One node's whole stream, in XSSet::BuildFlatXsStream's order.
///
/// @param i  index into `v.nodes` -- the node's slot, NOT its node id.
/// @return   the entry count on success, `encodeRefusal(reason)` on a refusal.
///
/// Writes `out.node_off[i]` and `out.node_cnt[i]` and its own `stride` slots of
/// the three stream arrays.  Touches no other node's memory, reads no other
/// node's outputs, and takes no lock: node independence is the contract.
template <class POL>
RASBERY_XSR_HD inline int flatxsStreamResolveNode(const flatxs::FlatXsView& v,
                                                  const StreamLibView& lib,
                                                  const StreamNodeView& nd,
                                                  const StreamOutView& out, int i,
                                                  const POL& pol,
                                                  const StreamForms& spol) {
    const int l      = v.nodes[i];
    const int nxyz   = v.nxyz;
    const int ng     = nd.ng;
    const int ith    = ng - 1;
    const int base   = i * out.stride;
    const int stride = out.stride;

    out.node_off[i] = base;

    const int model = nd.comp[l];
    if (model < 0 || model >= lib.nmodel || lib.model_ok[model] == 0) {
        out.node_cnt[i] = encodeRefusal(kRefusalModel);
        return out.node_cnt[i];
    }

    int n = 0;

    // -- 1. the three scalar branches, wu arm then history-twin arm ----------
    const double boron_dmod = flatxs::fxsBoronDmod(v, l);
    const double x_vals[kNumScalarBranches] = {
        boron_dmod * v.wvfr[l] * v.bppm[l] * flatxs::BORON_DENSITY_FACTOR,
        fsSqrt(nd.tful[l]),
        v.dmod[l]};
    const double hw = nd.hw != nullptr ? nd.hw[l] : 0.0;
    const double wu = 1.0 - hw;

    for (int branch = 0; branch < kNumScalarBranches; ++branch) {
        const int lo = nd.delta_lo[branch * nxyz + l];
        if (lo >= 0 && wu > 0.0) {
            const int    hi = nd.delta_hi[branch * nxyz + l];
            const double f  = nd.delta_frac[branch * nxyz + l];
            if (n >= stride) { out.node_cnt[i] = encodeRefusal(kRefusalCapacity); return out.node_cnt[i]; }
            out.stream_did[base + n]   = lo;
            out.stream_x[base + n]     = x_vals[branch];
            out.stream_scale[base + n] = wu * (1.0 - f);
            ++n;
            if (hi != lo) {
                if (n >= stride) { out.node_cnt[i] = encodeRefusal(kRefusalCapacity); return out.node_cnt[i]; }
                out.stream_did[base + n]   = hi;
                out.stream_x[base + n]     = x_vals[branch];
                out.stream_scale[base + n] = wu * f;
                ++n;
            }
        }
        if (hw <= 0.0 || nd.delta_lo_p == nullptr) continue;
        const int lo_p = nd.delta_lo_p[branch * nxyz + l];
        if (lo_p < 0) continue;
        const int    hi_p = nd.delta_hi_p[branch * nxyz + l];
        const double f_p  = nd.delta_frac_p[branch * nxyz + l];
        if (n >= stride) { out.node_cnt[i] = encodeRefusal(kRefusalCapacity); return out.node_cnt[i]; }
        out.stream_did[base + n]   = lo_p;
        out.stream_x[base + n]     = x_vals[branch];
        out.stream_scale[base + n] = hw * (1.0 - f_p);
        ++n;
        if (hi_p != lo_p) {
            if (n >= stride) { out.node_cnt[i] = encodeRefusal(kRefusalCapacity); return out.node_cnt[i]; }
            out.stream_did[base + n]   = hi_p;
            out.stream_x[base + n]     = x_vals[branch];
            out.stream_scale[base + n] = hw * f_p;
            ++n;
        }
    }

    // -- 2. the two-element workspace probe ---------------------------------
    //
    // The host builds a whole NISO*ng scratch and fills two entries of it; the
    // two entries are all NodeSpectralIndex reads, so they live in registers
    // here.  Same function, same prefix, same forms policy.
    const int    ePu = nd.i_pu239 * ng + ith;
    const int    eB  = nd.i_b10 * ng + ith;
    const double probe_pu = flatxs::flatxsProbeMicElement(
        v, l, 1, ePu, out.stream_did + base, out.stream_x + base,
        out.stream_scale + base, n, pol);
    const double probe_b = flatxs::flatxsProbeMicElement(
        v, l, 1, eB, out.stream_did + base, out.stream_x + base,
        out.stream_scale + base, n, pol);

    // -- 3./4. the spectral-history terms, own library then the twin's -------
    const int burn         = nd.burn[l];
    const int currentCtype = 0; // unrodded list only; see the ctype-0 note
    const int partner      = lib.history_partner[model];

    for (int pass = 0; pass < 2; ++pass) {
        const double weight = (pass == 0) ? wu : hw;
        if (weight == 0.0) continue;
        int m = model;
        if (pass == 1) {
            if (!(hw > 0.0) || partner < 0) continue;
            m = partner;
            if (m >= lib.nmodel || lib.model_ok[m] == 0) {
                out.node_cnt[i] = encodeRefusal(kRefusalModel);
                return out.node_cnt[i];
            }
        }
        if (lib.refr0_present[m] == 0) continue; // findCtype(..., 0) < 0 -> return

        const double nodeBranchX[kNumScalarBranches] = {
            boron_dmod * v.wvfr[l] * v.bppm[l] * flatxs::BORON_DENSITY_FACTOR,
            fsSqrt(nd.tful[l]),
            v.dmod[l]};

        const int c0 = lib.model_sh_off[m];
        const int c1 = lib.model_sh_off[m + 1];
        for (int c = c0; c < c1; ++c) {
            const int* burnups = lib.sh_burnups + lib.sh_burn_off[c];
            const int  nburn   = lib.sh_burn_cnt[c];
            const int  loIndex = fsFindLoBurn(burnups, nburn, burn);
            const int  hiIndex = fsFindHiBurn(burnups, nburn, burn);
            if (loIndex < 0 || hiIndex < 0) continue;

            const int coord = lib.sh_coord[c];
            if (!formImplemented(coord)) {
                out.node_cnt[i] = encodeRefusal(kRefusalForm);
                return out.node_cnt[i];
            }
            const int isotope = lib.sh_iso[c];
            const int partner_iso = lib.sh_partner[c];

            const bool logarithmic     = coord == kLogDensity;
            const bool rooted          = coord == kSqrtDensity;
            const bool thermalWeighted = coord == kThermalWeighted;
            const bool fastWeighted    = coord == kFastWeighted;
            const bool ratioInteraction = coord == kFluxRatioInteraction;
            const bool burnRatio       = coord == kRelativeBurnRatio;
            const bool spectralIndex   = coord == kSpectralIndex;
            const bool spectralCross   = coord == kSpectralIndexInteraction;
            const int  branchAxis      = branchAxisOf(coord);
            const bool ratioForm       = isRatioForm(coord);
            const bool fissile         = coord == kFissileFraction;
            const int  rodAgeAxis      = rodAgeAxisOf(coord);
            const bool fluxWeighted = thermalWeighted || fastWeighted ||
                                      ratioInteraction || spectralIndex ||
                                      spectralCross || branchAxis >= 0 ||
                                      ratioForm || fissile || rodAgeAxis >= 0;

            const double density = v.iden[static_cast<long long>(isotope) * nxyz + l];
            const double fl      = kSpectralLogDensityFloor;

            double coordinate;
            if (burnRatio) {
                // burnRatioCoordinate
                if (partner_iso >= lib.niso) {
                    coordinate = 0.0;
                } else {
                    const double na = fsMax(v.iden[static_cast<long long>(isotope) * nxyz + l], fl);
                    const double nb = fsMax(v.iden[static_cast<long long>(partner_iso) * nxyz + l], fl);
                    const double ra = fsMax(fsReferenceDensity0(lib, m, isotope, burn, spol), fl);
                    const double rb = fsMax(fsReferenceDensity0(lib, m, partner_iso, burn, spol), fl);
                    coordinate = fsLog(na / ra) - fsLog(nb / rb);
                }
            } else if (rodAgeAxis >= 0) {
                coordinate = (nodeBranchX[rodAgeAxis] -
                              fsReferenceCondition(lib, m, rodAgeAxis, burn, spol)) *
                             fsRodThermalFluence(l, currentCtype) * kRodAgeScale;
            } else if (fissile) {
                if (partner_iso >= lib.niso) {
                    coordinate = 0.0;
                } else {
                    const double na = fsMax(0.0, v.iden[static_cast<long long>(isotope) * nxyz + l]);
                    const double sum =
                        na + fsMax(0.0, v.iden[static_cast<long long>(partner_iso) * nxyz + l]);
                    coordinate = sum > 1.0e-300 ? na / sum : 0.0;
                }
            } else if (ratioForm) {
                coordinate = fsRatioFormOf(
                    coord, fsMax(density, fl),
                    fsMax(fsReferenceDensity(lib, m, isotope, burn, spol), fl));
            } else if (branchAxis >= 0) {
                coordinate = (nodeBranchX[branchAxis] -
                              fsReferenceCondition(lib, m, branchAxis, burn, spol)) *
                             (density - fsReferenceDensity(lib, m, isotope, burn, spol));
            } else if (spectralCross) {
                coordinate = fsNodeSpectralIndex(v, nd, l, probe_pu, probe_b) *
                             (density - fsReferenceDensity(lib, m, isotope, burn, spol));
            } else if (spectralIndex) {
                coordinate = fsNodeSpectralIndex(v, nd, l, probe_pu, probe_b);
            } else if (ratioInteraction) {
                coordinate = fsLog(fsMax(fsNodeFluxShare(nd, l, true), 1.0e-30) /
                                   fsMax(fsNodeFluxShare(nd, l, false), 1.0e-30)) *
                             (density - fsReferenceDensity(lib, m, isotope, burn, spol));
            } else if (fluxWeighted) {
                coordinate = fsNodeFluxShare(nd, l, thermalWeighted) * fsMax(0.0, density);
            } else if (rooted) {
                coordinate = fsSqrt(fsMax(0.0, density));
            } else if (logarithmic) {
                coordinate = fsLog(fsMax(density, kSpectralLogDensityFloor));
            } else {
                coordinate = fsMax(0.0, density);
            }

            const double fraction =
                loIndex == hiIndex
                    ? 0.0
                    : static_cast<double>(burn - burnups[loIndex]) /
                          static_cast<double>(burnups[hiIndex] - burnups[loIndex]);

            const double rodWeight = lib.sh_rod_scaled[c] ? fsRodBlendWeight(nd, l) : 1.0;
            if (rodWeight == 0.0) continue;

            // keyCoordinate, inlined twice exactly as the host calls it twice.
            double referenceNow = 0.0;
            if (!(loIndex == hiIndex || fluxWeighted))
                referenceNow = fsReferenceDensity(lib, m, isotope, burn, spol);

            for (int side = 0; side < 2; ++side) {
                const int idx = side == 0 ? loIndex : hiIndex;
                if (side == 1 && hiIndex == loIndex) break;
                double key = coordinate;
                if (!(loIndex == hiIndex || fluxWeighted)) {
                    const double referenceAtKey =
                        fsReferenceDensity(lib, m, isotope, burnups[idx], spol);
                    if (logarithmic) {
                        key = coordinate -
                              fsLog(fsMax(referenceNow, kSpectralLogDensityFloor)) +
                              fsLog(fsMax(referenceAtKey, kSpectralLogDensityFloor));
                    } else if (rooted) {
                        key = coordinate - fsSqrt(fsMax(0.0, referenceNow)) +
                              fsSqrt(fsMax(0.0, referenceAtKey));
                    } else {
                        key = coordinate - referenceNow + referenceAtKey;
                    }
                }
                if (n >= stride) {
                    out.node_cnt[i] = encodeRefusal(kRefusalCapacity);
                    return out.node_cnt[i];
                }
                out.stream_did[base + n] = lib.sh_delta_base[c] + idx;
                out.stream_x[base + n]   = key;
                out.stream_scale[base + n] =
                    side == 0 ? weight * rodWeight * (1.0 - fraction)
                              : weight * rodWeight * fraction;
                ++n;
            }
        }
    }

    out.node_cnt[i] = n;
    return n;
}

} // namespace rasbery::flatxs_stream
