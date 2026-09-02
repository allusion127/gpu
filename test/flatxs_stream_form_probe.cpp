// WP23.1 gate: the branch-stream contraction mask, and the libm residual.
//
// SELF-CALIBRATING, LIKE xe_form_probe and cmfd_outer_form_probe.  It does not
// assert a literal mask -- a literal is a record of the machine it was measured
// on and would fail on the next host for a reason that has nothing to do with
// the code under test.  It MINES this host's mask, checks the derivation is
// sound, and scores against that.  The shipped record is printed beside it so a
// host that contracts differently says so in one line.
//
// Five things are checked, and they are five different claims:
//
//   1. THE MINING IS SOUND.  Every one of four seeds reaches zero mismatching
//      words.  A seed that cannot is a site this fixture does not pin, and then
//      nothing here knows the contract.
//
//   2. EACH SITE IS DECISIVE.  Moving a site OFF the mined state must break
//      something.  Without this, a mask of all zeros passes on a host whose
//      compiler fused nothing, and the gate would be asserting that the fixture
//      is boring rather than that the mask is right.  P1 and P2 are exempt from
//      each other -- IEEE fma is commutative in its first two arguments, so
//      they are the same bits at these three single-product sites and the
//      REPORT says which sites are decisive only against FS_SITE_NONE.
//
//   3. THE PRODUCTION RESOLVER AGREES WITH THE MINING.  flatxs_stream::
//      streamFormMask() is what the kernel is launched with; a gate that mined
//      its own mask and never asked the resolver would pass while the run used
//      something else.
//
//   4. THE EXACT log/cbrt ARE CORRECTLY ROUNDED, and their residual against
//      glibc is MEASURED rather than assumed.  10^6 sampled arguments over the
//      ranges the resolver evaluates; mismatch count, max ulp, and -- for every
//      mismatch -- WHICH side is the one that is not correctly rounded, decided
//      against the double-double value itself.  Reported, not asserted to be
//      zero: it is exactly the N1 difference reason (b) names, and a gate that
//      demanded it be zero would be demanding the feature not exist.
//
//   5. THE EXACT PATH IS DECIDABLE.  No sampled argument may leave the
//      double-double too close to a rounding boundary for its own accuracy to
//      choose.  Over 10^6 arguments the expected count is ~10^6 * 2^-47, so a
//      non-zero count is a bug in FlatXsStreamExactMath.h and not a tie.

#include "FlatXsStreamFormMask.h"
#include "FlatXsStreamFormMine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fss = rasbery::flatxs_stream;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::uint64_t bitsOf(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

/// Signed distance in representable doubles.  Both arguments are finite and of
/// the same sign everywhere this is used.
long ulpDistance(double a, double b) {
    if (a == b) return 0;
    std::int64_t x = static_cast<std::int64_t>(bitsOf(a));
    std::int64_t y = static_cast<std::int64_t>(bitsOf(b));
    if (x < 0) x = static_cast<std::int64_t>(0x8000000000000000ull) - x;
    if (y < 0) y = static_cast<std::int64_t>(0x8000000000000000ull) - y;
    const long d = static_cast<long>(x - y);
    return d < 0 ? -d : d;
}

struct Rng {
    std::uint64_t s;
    double        next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((s >> 11) & ((1ull << 53) - 1ull)) /
               static_cast<double>(1ull << 53);
    }
};

/// The argument a coordinate form actually hands to `log`: a floored number
/// density, a now/reference inventory ratio, a wide-decade ratio, and the
/// 1e-30-guarded thermal/fast flux share.  Sampling uniform doubles instead
/// would report a residual for a function this arm never evaluates.
double logArgument(Rng& rng) {
    const int mode = static_cast<int>(rng.next() * 4.0);
    if (mode == 0) return 1.0e-12 + rng.next() * 1.0e-2;
    if (mode == 1) return 0.2 + rng.next() * 5.0;
    if (mode == 2) return 1.0e-4 * std::pow(10.0, rng.next() * 8.0);
    return 1.0e-30 + rng.next() * 1.0e-24;
}

/// kCubeRootRatio's argument: `now / reference`, both floored at 1e-12.
double cbrtArgument(Rng& rng) {
    const int mode = static_cast<int>(rng.next() * 3.0);
    if (mode == 0) return 0.2 + rng.next() * 5.0;
    if (mode == 1) return 1.0e-6 * std::pow(10.0, rng.next() * 12.0);
    return 1.0e-12 + rng.next() * 1.0e-2;
}

struct LibmReport {
    long mismatch = 0;
    long max_ulp  = 0;
    long vendor_not_cr = 0; ///< of the mismatches, the ones where GLIBC is wrong
    long ours_not_cr   = 0; ///< of the mismatches, the ones where WE are wrong
    long undecidable   = 0;
};

LibmReport measureLog(long n) {
    LibmReport r;
    Rng        rng{0x9E3779B97F4A7C15ull};
    for (long i = 0; i < n; ++i) {
        const double x = logArgument(rng);
        if (!(x > 0.0)) continue;
        const fss::FsDD d = fss::fsExactLogDD(x);
        if (fss::fsDDRoundingUncertain(d, 1.0e-28)) ++r.undecidable;
        const double ours   = fss::fsDDToDouble(d);
        const double vendor = std::log(x);
        if (ours == vendor) continue;
        ++r.mismatch;
        const long u = ulpDistance(ours, vendor);
        if (u > r.max_ulp) r.max_ulp = u;
        // Which one is not correctly rounded?  The double-double IS the answer
        // to ~2^-100, so the nearer of the two candidates to it is the
        // correctly-rounded one.  This is the question the doc reports, and it
        // is not the same question as "do they differ".
        const double ro = std::fabs(fss::fsDDAddD(d, -ours).hi);
        const double rv = std::fabs(fss::fsDDAddD(d, -vendor).hi);
        if (ro < rv)
            ++r.vendor_not_cr;
        else if (rv < ro)
            ++r.ours_not_cr;
    }
    return r;
}

LibmReport measureCbrt(long n) {
    LibmReport r;
    Rng        rng{0xA16E13ull};
    for (long i = 0; i < n; ++i) {
        const double x = cbrtArgument(rng);
        const fss::FsDD d = fss::fsExactCbrtDD(x);
        if (fss::fsDDRoundingUncertain(d, 1.0e-28)) ++r.undecidable;
        const double ours   = fss::fsDDToDouble(d);
        const double vendor = std::cbrt(x);
        if (ours == vendor) continue;
        ++r.mismatch;
        const long u = ulpDistance(ours, vendor);
        if (u > r.max_ulp) r.max_ulp = u;
        const double ro = std::fabs(fss::fsDDAddD(d, -ours).hi);
        const double rv = std::fabs(fss::fsDDAddD(d, -vendor).hi);
        if (ro < rv)
            ++r.vendor_not_cr;
        else if (rv < ro)
            ++r.ours_not_cr;
    }
    return r;
}

const char* siteName(unsigned bit) {
    if (bit == fss::FS_REFDENS) return "FS_REFDENS";
    if (bit == fss::FS_REFDENS0) return "FS_REFDENS0";
    return "FS_REFCOND";
}

const char* stateName(unsigned st) {
    if (st == fss::FS_SITE_P1) return "P1";
    if (st == fss::FS_SITE_P2) return "P2";
    return "NONE";
}

} // namespace

int main() {
    const fsref::Fixture     f   = fsref::buildProductionFixture();
    const fssmine::Reference ref = fssmine::referenceOf(f);

    // --- 1. the mining is sound -------------------------------------------
    bool           sound = false;
    const unsigned mined = fssmine::mineStable(f, sound);
    std::printf("FLATXS_STREAM_FORMS mined 0x%x (shipped record 0x%x), sound=%d\n", mined,
                fss::kStreamFormsDefault, sound ? 1 : 0);
    std::printf("  sites: %s=%s %s=%s %s=%s\n", siteName(fss::FS_REFDENS),
                stateName((mined >> fss::FS_REFDENS) & 3u), siteName(fss::FS_REFDENS0),
                stateName((mined >> fss::FS_REFDENS0) & 3u), siteName(fss::FS_REFCOND),
                stateName((mined >> fss::FS_REFCOND) & 3u));
    check(sound, "the four-seed descent did not reach a bit-exact mask on this host");
    check(fssmine::scoreMask(f, ref, mined) == 0,
          "the mined mask does not reproduce the quotation bit for bit");

    // --- 2. each site is decisive against FS_SITE_NONE ---------------------
    const unsigned sites[3] = {fss::FS_REFDENS, fss::FS_REFDENS0, fss::FS_REFCOND};
    for (unsigned bit : sites) {
        const unsigned st = (mined >> bit) & 3u;
        const unsigned other =
            (st == fss::FS_SITE_NONE) ? fss::FS_SITE_P1 : fss::FS_SITE_NONE;
        const long score = fssmine::scoreSiteState(f, ref, mined, bit, other);
        std::printf("  %-12s %s -> %s : %ld mismatching words\n", siteName(bit),
                    stateName(st), stateName(other), score);
        check(score > 0,
              "a site is a DON'T-CARE against FS_SITE_NONE on this fixture -- the "
              "mask is not being measured, the fixture is");
    }
    // P1 vs P2 must be indistinguishable, and that is an ASSERTION and not a
    // hope: it is what lets `sound` be defined as "every seed reached zero"
    // rather than "every seed agreed".
    for (unsigned bit : sites) {
        const long a = fssmine::scoreSiteState(f, ref, mined, bit, fss::FS_SITE_P1);
        const long b = fssmine::scoreSiteState(f, ref, mined, bit, fss::FS_SITE_P2);
        check(a == b,
              "FS_SITE_P1 and FS_SITE_P2 differ at a single-product site; IEEE fma "
              "is commutative in its first two arguments, so this cannot happen "
              "unless the encoding has drifted");
    }

    // --- 3. the production resolver agrees --------------------------------
    const unsigned resolved = fss::streamFormMask();
    std::printf("  resolver: mask 0x%x source %s sound %d libm %s\n", resolved,
                fss::streamFormsSource(), fss::streamFormsSound() ? 1 : 0,
                fss::streamLibmName());
    if (std::getenv("RASBERY_FLATXS_STREAM_FORMS") == nullptr)
        check(resolved == mined,
              "the production resolver did not return the mined mask; the kernel "
              "would run under a contract this gate never scored");

    // --- 4./5. the libm residual ------------------------------------------
    const long       N  = 1000000;
    const LibmReport lg = measureLog(N);
    const LibmReport cb = measureCbrt(N);
    std::printf("libm exact vs glibc over %ld sampled arguments:\n", N);
    std::printf("  log : mismatch %ld  max_ulp %ld  glibc_not_CR %ld  ours_not_CR %ld  "
                "undecidable %ld\n",
                lg.mismatch, lg.max_ulp, lg.vendor_not_cr, lg.ours_not_cr, lg.undecidable);
    std::printf("  cbrt: mismatch %ld  max_ulp %ld  glibc_not_CR %ld  ours_not_CR %ld  "
                "undecidable %ld\n",
                cb.mismatch, cb.max_ulp, cb.vendor_not_cr, cb.ours_not_cr, cb.undecidable);
    check(lg.ours_not_cr == 0,
          "the exact log is not correctly rounded on some sampled argument");
    check(cb.ours_not_cr == 0,
          "the exact cbrt is not correctly rounded on some sampled argument");
    check(lg.undecidable == 0 && cb.undecidable == 0,
          "a sampled argument left the double-double unable to decide its own "
          "rounding; that is an accuracy bug in FlatXsStreamExactMath.h");

    // THE GRADE, said out loud rather than left to be inferred from the counts.
    if (lg.mismatch == 0 && cb.mismatch == 0)
        std::printf("libm verdict: EXACT reproduces glibc on this fixture -- reason (b) "
                    "retires on these ranges\n");
    else
        std::printf("libm verdict: EXACT does NOT reproduce glibc (%ld + %ld mismatches); "
                    "reason (b) stays live and the default remains \"fast\"\n",
                    lg.mismatch, cb.mismatch);

    std::printf(failures == 0 ? "flatxs stream form probe: PASS\n"
                              : "flatxs stream form probe: FAIL (%d)\n",
                failures);
    return failures == 0 ? 0 : 1;
}
