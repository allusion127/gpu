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

#include <atomic>
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
    std::atomic<unsigned long long> max_ulp{0};
    /// Which of the nine arrays carried that maximum (0..8 in the eta1, eta2,
    /// m260, m251, m253, m262, m264, diagD, diagDI order the upload uses).
    std::atomic<unsigned long long> max_ulp_array{0};
    /// Elements whose ULP distance exceeded 1 -- the spike's own bound.  A
    /// non-zero count here is the thing to look at before any gate is quoted.
    std::atomic<unsigned long long> over_1ulp{0};
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
    "with Gate A/B, never with a digest equality; max_ulp above is this run's own "
    "check of that bound (docs/WP23_FLATXS_STREAM_GPU_20260902_KO.md section 6)";

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
        os << t.max_ulp.load(std::memory_order_relaxed) << ",\"max_ulp_array\":"
           << t.max_ulp_array.load(std::memory_order_relaxed);
    else
        os << "-1,\"max_ulp_array\":-1";
    os << ",\"over_1ulp\":" << t.over_1ulp.load(std::memory_order_relaxed)
       << ",\"policy_note\":\"" << kNodalConstsPolicyNote << "\"";
}

} // namespace rasbery::nodalconsts
