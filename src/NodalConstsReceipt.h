#pragma once

// The [RASBERY][NODAL_CONSTS] receipt -- WP23 item 3.
//
// ---------------------------------------------------------------------------
// WHAT THE ARM IS, IN ONE PARAGRAPH
// ---------------------------------------------------------------------------
//
// Nodal::updateConstant computes nine SENM coefficient arrays on the host from
// xsrf/xsdf/hmesh, and XsReconBackend::solveNodal uploads all nine every time
// `_const_generation` moves: 9,675 copies and 3.9 GB over a KNGR run (1.96 GB
// under RASBERY_GPU_FP32), because the Xe device step keeps moving the
// macroscopic cross sections that the coefficients are a function of.  With
// RASBERY_GPU_NODAL_CONSTS=1 the nine arrays are computed ON THE DEVICE from
// xsrf/xsdf/hmesh -- all three of which are already device-resident -- and the
// nine uploads become one xsdf upload (or none, when the state generation
// matches).
//
// THE HOST SWEEP STILL RUNS, AND THAT IS THE POINT OF THIS RECEIPT.  The arm
// removes UPLOADS, not the host arithmetic: `Nodal`'s own CPU phases and the
// hybrid drive read `_eta1 ... _diagDI`, so a run whose device drive declines
// must still find them correct.  Keeping them also makes the self-check below
// possible at all, because the host answer is right there to compare against.
// Moving the host sweep as well is a separate arm with a separate obligation
// (the host arrays would then be stale for the fallback), and this one does not
// claim it.
//
// ---------------------------------------------------------------------------
// CLASS N1, MEASURED, NOT ASSUMED -- AND THE MEASUREMENT IS IN THE RECEIPT
// ---------------------------------------------------------------------------
//
// src/NodalConstantKernel.h records the B0-rescue spike that FAILED: over
// 4,000,000 swept arguments CUDA's `sqrt` matched glibc's everywhere, and its
// `exp` differed on 3.34 % (5.34 % inside the physical kp2 band), always by
// exactly 1 ulp.  So this arm is a trajectory-changing transition by
// construction and is priced with Gate A/B, never with a digest equality.
//
// What a receipt can add is whether THIS run behaved like that spike.  The arm
// samples a strided subset of the nine device arrays, compares each element to
// the host value the sweep already computed, and reports the largest ULP
// distance seen.  `max_ulp <= 1` is the spike's claim reproduced on the deck;
// anything larger is a different phenomenon -- a contraction the --fmad=false
// build did not stop, a stale xsdf, a layout disagreement -- and it is better to
// read it in the receipt than to discover it as a Gate B pin-power miss.
//
// ---------------------------------------------------------------------------
// WHY THE METRIC IS PER-ARRAY AND RELATIVE, AND WHAT IT COST TO FIND OUT
// ---------------------------------------------------------------------------
//
// The 238 run of block 49 reported `max_ulp:9545195, max_ulp_array:7,
// over_1ulp:107279 (58 %)` while Gate A and Gate B both PASSED.  Nine and a half
// million ulp is not "exp differs by 1 ulp", and a receipt that can print that
// number next to a passing gate is a receipt nobody can act on.  Array 7 is
// `diagD = 4 * xsdf / (hmesh * hmesh)`: no exp, no sqrt, no contraction site, one
// IEEE divide, under --fmad=false.  Host and device CANNOT legitimately disagree
// there by one ulp, let alone by nine million -- so the number was never about
// the arithmetic.  It was the read-back racing the build kernel (see the
// self-check in CudaXsReconBackend.cu, which now drains the stream first).
//
// Three things changed here so the next surprise reads correctly the first time:
//
//   * PER-ARRAY.  One global maximum and one "which array carried it" scalar
//     cannot say whether ONE array is broken or all nine are drifting, and the
//     scalar was being stored under a predicate (`worst >= max_ulp` re-read
//     AFTER the compare-exchange) that is true whenever a later check merely
//     TIES.  It is now DERIVED as the argmax of the per-array table, which
//     cannot disagree with the maximum it names.
//   * RELATIVE, WITH A FLOOR.  A ULP distance is meaningless across a sign
//     change and unbounded near zero, and these arrays legitimately hold values
//     that straddle zero.  `rel = |a - b| / max(|a|, |b|, floor)` says how wrong
//     the value is; the ULP figure is kept beside it because the exp spike's own
//     bound is stated in ulp and that bound is still the thing to check.
//   * THE FLOOR IS THE ARRAY'S OWN SCALE, not a constant somebody picked.  Each
//     sampled array's largest host magnitude times kRelFloorFrac is the floor for
//     that array, so a coefficient that is 1e-30 in an array whose scale is 1e2
//     is compared absolutely instead of reporting a spectacular relative error
//     about a number that does not matter.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>

namespace rasbery::nodalconsts {

/// ULP distance between two doubles, on the standard monotone-integer mapping.
/// Returns UINT64_MAX for a NaN on either side -- which is a difference the
/// receipt must not report as "0 ulp".
inline unsigned long long ulpDistance(double a, double b) {
    if (a == b) return 0ULL;
    if (!(a == a) || !(b == b)) return ~0ULL;
    std::uint64_t ua = 0, ub = 0;
    std::memcpy(&ua, &a, sizeof ua);
    std::memcpy(&ub, &b, sizeof ub);
    // IEEE-754 doubles are sign-magnitude; this maps them onto the total order
    // in UNSIGNED arithmetic, where the -0.0/+0.0 boundary and the extremes
    // cannot overflow.  Doing it in int64 needs 0x8000000000000000 as a
    // POSITIVE value, which int64 does not have.
    auto key = [](std::uint64_t u) -> std::uint64_t {
        return (u & 0x8000000000000000ULL)
                   ? (0x8000000000000000ULL - (u & 0x7FFFFFFFFFFFFFFFULL))
                   : (u + 0x8000000000000000ULL);
    };
    const std::uint64_t ka = key(ua);
    const std::uint64_t kb = key(ub);
    return static_cast<unsigned long long>(ka > kb ? ka - kb : kb - ka);
}

/// The nine arrays, in the order the upload uses and the device kernel stores
/// them: index i here is `dst[i * ndg + idx]` in kNodalConstsDevice.
///
/// NAMED, NOT NUMBERED, BECAUSE THE NUMBERS COLLIDE.  CudaNodalConstantKernel.h
/// has an enum for the ARENA kernel in which 7 is diagDI and 8 is diagD -- the
/// other way round from this table.  238 reported `max_ulp_array:7` and reading
/// it against the wrong table sends the reader to `1/D` when the array is `D`.
inline constexpr int         kNodalConstsCount   = 9;
inline constexpr const char* kNodalConstsNames[] = {"eta1", "eta2",  "m260",
                                                    "m251", "m253",  "m262",
                                                    "m264", "diagD", "diagDI"};

/// The relative-error floor, AS A FRACTION OF THE ARRAY'S OWN SAMPLED SCALE.
/// Values below it are compared absolutely: a relative error on a coefficient
/// twelve orders of magnitude under everything else in its own array is a
/// statement about rounding noise, not about the physics.
inline constexpr double kRelFloorFrac = 1.0e-12;

/// `|a - b| / max(|a|, |b|, floor)`.  Never NaN: two NaNs agree, one NaN is
/// infinitely wrong.  Defined so a sign change reports ~2 rather than the
/// astronomical integer a ULP distance across zero produces.
inline double relError(double a, double b, double floor_abs) {
    const bool na = !(a == a), nb = !(b == b);
    if (na && nb) return 0.0;
    if (na || nb) return HUGE_VAL;
    if (a == b) return 0.0;
    const double aa = a < 0.0 ? -a : a;
    const double ab = b < 0.0 ? -b : b;
    double       den = aa > ab ? aa : ab;
    if (!(den > floor_abs)) den = floor_abs;
    if (!(den > 0.0)) return 0.0;
    const double d = a - b;
    return (d < 0.0 ? -d : d) / den;
}

/// A non-negative double as a monotone unsigned key, so a lock-free running
/// maximum can be kept in the same atomic<unsigned long long> the ULP maxima
/// use.  Monotone for non-negative IEEE doubles by construction.
inline unsigned long long relKey(double r) {
    if (!(r == r)) return ~0ULL; // NaN cannot outrank a real disagreement quietly
    if (r < 0.0) r = 0.0;
    std::uint64_t u = 0;
    std::memcpy(&u, &r, sizeof u);
    return static_cast<unsigned long long>(u);
}

inline double relFromKey(unsigned long long k) {
    if (k == ~0ULL) return HUGE_VAL;
    const std::uint64_t u = static_cast<std::uint64_t>(k);
    double              r = 0.0;
    std::memcpy(&r, &u, sizeof r);
    return r;
}

/// JSON HAS NO INFINITY, so a relative error that came back non-finite -- which
/// relError produces for exactly one case, a NaN on one side and not the other --
/// is printed as the sentinel -2.  -1 already means "no sample"; a bare `inf` in
/// the receipt would make the line unparseable for the gate script that reads it,
/// which is a worse failure than the one it is trying to report.
inline constexpr double kRelNonFinite = -2.0;

inline double relForJson(double r) { return (r - r == 0.0) ? r : kRelNonFinite; }

/// Lock-free running maximum.
inline void bumpMax(std::atomic<unsigned long long>& slot, unsigned long long value) {
    unsigned long long prev = slot.load(std::memory_order_relaxed);
    while (value > prev && !slot.compare_exchange_weak(prev, value, std::memory_order_relaxed))
        ;
}

struct NodalConstsTally {
    /// solveNodal calls that reached the constants gate with the arm armed.
    std::atomic<unsigned long long> gates{0};
    /// Of those, the ones the device computed.
    std::atomic<unsigned long long> device_builds{0};
    /// Of those, the ones that fell back to the nine uploads.
    std::atomic<unsigned long long> host_uploads{0};
    /// Copies the arm removed, and their bytes AT THE ARM'S WIDTH (float under
    /// RASBERY_GPU_FP32) -- a figure quoted in doubles on a narrowed run would
    /// hide the saving the FP32 arm already took.
    std::atomic<unsigned long long> uploads_elided{0};
    std::atomic<unsigned long long> bytes_elided{0};
    /// Bytes the arm still moves: the xsdf column it needs and the self-check
    /// read-back.  Present so bytes_elided is never read as a net.
    std::atomic<unsigned long long> bytes_h2d{0};
    std::atomic<unsigned long long> bytes_d2h{0};
    /// Self-check: how many builds were sampled, how many elements in total, and
    /// the largest ULP distance to the host answer seen across all of them.
    std::atomic<unsigned long long> checks{0};
    std::atomic<unsigned long long> checked_elems{0};
    /// PER-ARRAY, and the scalars below are DERIVED from these rather than kept
    /// beside them.  A separately-stored "which array" scalar is a second source
    /// of truth that drifts: the one this replaced was written under a predicate
    /// that any later check could satisfy by TYING the maximum, so it named the
    /// last array to tie rather than the array that carried the max.
    std::atomic<unsigned long long> max_ulp_by_array[kNodalConstsCount]{};
    /// Relative error, as the monotone bit key relKey() produces.
    std::atomic<unsigned long long> max_rel_by_array[kNodalConstsCount]{};
    /// Elements whose ULP distance exceeded 1 -- the spike's own bound.  A
    /// non-zero count here is the thing to look at before any gate is quoted.
    std::atomic<unsigned long long> over_1ulp{0};

    unsigned long long maxUlp() const {
        unsigned long long m = 0;
        for (int i = 0; i < kNodalConstsCount; ++i) {
            const unsigned long long v = max_ulp_by_array[i].load(std::memory_order_relaxed);
            if (v > m) m = v;
        }
        return m;
    }
    /// Argmax, ties going to the LOWEST index, so the answer is a function of
    /// the table and not of the order the checks happened to run in.
    int maxUlpArray() const {
        int                m = 0;
        unsigned long long best = max_ulp_by_array[0].load(std::memory_order_relaxed);
        for (int i = 1; i < kNodalConstsCount; ++i) {
            const unsigned long long v = max_ulp_by_array[i].load(std::memory_order_relaxed);
            if (v > best) { best = v; m = i; }
        }
        return m;
    }
    unsigned long long maxRelKey() const {
        unsigned long long m = 0;
        for (int i = 0; i < kNodalConstsCount; ++i) {
            const unsigned long long v = max_rel_by_array[i].load(std::memory_order_relaxed);
            if (v > m) m = v;
        }
        return m;
    }
    int maxRelArray() const {
        int                m = 0;
        unsigned long long best = max_rel_by_array[0].load(std::memory_order_relaxed);
        for (int i = 1; i < kNodalConstsCount; ++i) {
            const unsigned long long v = max_rel_by_array[i].load(std::memory_order_relaxed);
            if (v > best) { best = v; m = i; }
        }
        return m;
    }
};

inline NodalConstsTally& nodalConstsTally() {
    static NodalConstsTally t;
    return t;
}

inline constexpr const char* kNodalConstsPolicyNote =
    "RASBERY_GPU_NODAL_CONSTS=1 is CLASS N1 BY MEASUREMENT, not by caution: "
    "test/nodal_constant_exp_probe.cu found CUDA's exp differs from glibc's on "
    "3.34% of the arguments this body evaluates, always by 1 ulp "
    "(src/NodalConstantKernel.h). It therefore moves the trajectory and is priced "
    "with Gate A/B, never with a digest equality; max_ulp/max_rel above are this "
    "run's own check of that bound, measured per array AFTER the build stream is "
    "drained -- 238 block 49 reported max_ulp 9545195 on diagD, an array with no "
    "libm and no contraction site, because the read-back was racing the kernel "
    "(docs/WP23_FLATXS_STREAM_GPU_20260902_KO.md section 6)";

inline bool nodalConstsReceiptWanted() {
    const NodalConstsTally& t = nodalConstsTally();
    return t.gates.load(std::memory_order_relaxed) != 0;
}

inline void appendNodalConstsReceiptFields(std::ostream& os) {
    const NodalConstsTally&  t     = nodalConstsTally();
    const unsigned long long gates = t.gates.load(std::memory_order_relaxed);
    const unsigned long long dev   = t.device_builds.load(std::memory_order_relaxed);
    const double share =
        (gates > 0) ? static_cast<double>(dev) / static_cast<double>(gates) : -1.0;
    const unsigned long long checks = t.checks.load(std::memory_order_relaxed);

    os << "\"arm\":" << (gates > 0 ? 1 : 0) << ",\"gates\":" << gates
       << ",\"device_builds\":" << dev
       << ",\"host_uploads\":" << t.host_uploads.load(std::memory_order_relaxed)
       << ",\"device_share\":" << share
       << ",\"uploads_elided\":" << t.uploads_elided.load(std::memory_order_relaxed)
       << ",\"bytes_elided\":" << t.bytes_elided.load(std::memory_order_relaxed)
       << ",\"bytes_h2d\":" << t.bytes_h2d.load(std::memory_order_relaxed)
       << ",\"bytes_d2h\":" << t.bytes_d2h.load(std::memory_order_relaxed)
       << ",\"self_checks\":" << checks
       << ",\"checked_elems\":" << t.checked_elems.load(std::memory_order_relaxed)
       << ",\"max_ulp\":";
    // Never 0 on an unchecked run: 0 ulp is a RESULT and "no sample" is not one.
    if (checks > 0)
        os << t.maxUlp() << ",\"max_ulp_array\":" << t.maxUlpArray();
    else
        os << "-1,\"max_ulp_array\":-1";
    os << ",\"max_rel\":";
    if (checks > 0)
        os << relForJson(relFromKey(t.maxRelKey())) << ",\"max_rel_array\":" << t.maxRelArray();
    else
        os << "-1,\"max_rel_array\":-1";
    // AND THE TABLE THE TWO SCALARS ARE THE ARGMAX OF.  A maximum without its
    // distribution cannot tell ONE broken array from nine drifting ones, which
    // is exactly the question 238's `max_ulp_array:7` left open.
    os << ",\"by_array\":{";
    for (int i = 0; i < kNodalConstsCount; ++i) {
        if (i != 0) os << ",";
        os << "\"" << kNodalConstsNames[i]
           << "\":{\"ulp\":" << t.max_ulp_by_array[i].load(std::memory_order_relaxed)
           << ",\"rel\":"
           << relForJson(relFromKey(t.max_rel_by_array[i].load(std::memory_order_relaxed)))
           << "}";
    }
    os << "}";
    os << ",\"over_1ulp\":" << t.over_1ulp.load(std::memory_order_relaxed)
       << ",\"policy_note\":\"" << kNodalConstsPolicyNote << "\"";
}

} // namespace rasbery::nodalconsts
