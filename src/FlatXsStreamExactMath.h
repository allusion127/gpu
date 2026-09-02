#pragma once

// WP23.1: `log` and `cbrt` WITHOUT A LIBM, so the device and the host can be
// asked to agree bit for bit instead of being assumed not to.
//
// ---------------------------------------------------------------------------
// THE PROBLEM THIS SOLVES, STATED EXACTLY
// ---------------------------------------------------------------------------
//
// src/FlatXsStreamKernel.h moved the coordinate arithmetic to the device, and
// seven of the twenty coordinate forms end in `std::log` or `std::cbrt` on the
// host and in CUDA's `log` / `cbrt` on the device.  Neither vendor's function is
// correctly rounded, both are within about one ulp of the true value, and they
// are within one ulp of DIFFERENT SIDES of it on some arguments.  That is
// reason (b) of the arm's N1 grade and no amount of contraction mining touches
// it: the two functions are different functions.
//
// The escape is not to make CUDA's log better.  It is to STOP CALLING EITHER
// ONE.  Everything below is built out of `+`, `-`, `*`, `/` and `fma` only --
// five operations IEEE-754 requires to be correctly rounded, which every
// conforming host and every CUDA device implements identically, and which the
// repo already routes through xsrMul/xsrFma so no compiler may re-associate or
// re-contract them.  So `fsExactLog(x)` computes THE SAME BITS on g++ and on
// nvcc by construction, not by measurement.
//
// That is a weaker claim than the one the arm needs, and the difference is the
// whole reason this file carries a measurement instead of an assertion:
//
//     BY CONSTRUCTION:  device fsExactLog(x) == host fsExactLog(x).
//     WHAT B0 NEEDS:    device fsExactLog(x) == host std::log(x),
//                       because the FLAG-OFF host path still calls glibc.
//
// The second follows from the first only where glibc's log is itself correctly
// rounded, since these functions are.  glibc's dbl-64 `log` is the ARM
// optimized-routines implementation, whose worst-case error is ~0.519 ulp --
// so it is correctly rounded on almost every argument and provably not on a
// few.  `cbrt` is weaker still (glibc's own libm-test-ulps records 1 ulp for
// it on x86_64).  WHICH of those two situations this deck is in is a question
// with a number for an answer, and test/flatxs_stream_form_probe.cpp is where
// the number is produced: 10^6 sampled arguments over the ranges the resolver
// actually evaluates, max ulp and mismatch count reported, and
// RASBERY_GPU_FLATXS_STREAM_LIBM defaulting to `exact` only when the count is
// zero.  Section 4.2 of docs/WP23_FLATXS_STREAM_GPU_20260902_KO.md carries the
// measured numbers.
//
// ---------------------------------------------------------------------------
// WHY DOUBLE-DOUBLE AND NOT A ZIV LOOP ON TOP OF THE VENDOR FUNCTION
// ---------------------------------------------------------------------------
//
// A Ziv correction needs a SECOND, more accurate evaluation to correct the
// first, so it needs this file anyway; and it would still start from a vendor
// function whose bits differ between the two compilers, which is the thing
// being removed.  Starting from the accurate evaluation and rounding it once is
// the same work with one fewer dependency.
//
// The accuracy target is ~2^-100 relative before the final rounding.  A result
// that lands closer than that to a halfway point cannot be rounded with
// certainty; `fsExactLogHard` reports those rather than pretending, and the
// probe counts them.  Over 10^6 arguments the expected count is ~10^6 * 2^-47,
// i.e. zero, and a non-zero count is a bug in this file and not a tie.
//
// HOST/DEVICE, NO STL, NO ALLOCATION -- the same rules FlatXsStreamKernel.h
// runs under, because this header is included by it.

#include "XsReconKernel.h" // xsrFma / xsrMul / RASBERY_XSR_HD

#include <cstring>

namespace rasbery::flatxs_stream {

using xsrecon::xsrFma;
using xsrecon::xsrMul;

// ---------------------------------------------------------------------------
// Bit access
// ---------------------------------------------------------------------------
//
// The exponent split is INTEGER work.  Doing it with frexp/ldexp would reach a
// libm on the host and an intrinsic on the device -- two functions again, for a
// job that is a shift and a mask.

RASBERY_XSR_HD inline unsigned long long fsToBits(double d) {
#if defined(__CUDA_ARCH__)
    return static_cast<unsigned long long>(__double_as_longlong(d));
#else
    unsigned long long b = 0;
    std::memcpy(&b, &d, sizeof b);
    return b;
#endif
}

RASBERY_XSR_HD inline double fsFromBits(unsigned long long b) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(b));
#else
    double d = 0.0;
    std::memcpy(&d, &b, sizeof d);
    return d;
#endif
}

/// 2^e for |e| <= 1023, EXACTLY, by writing the exponent field.  No ldexp, for
/// the reason above; no loop, so it is one instruction's worth of work.
RASBERY_XSR_HD inline double fsPow2(int e) {
    return fsFromBits(static_cast<unsigned long long>(e + 1023) << 52);
}

// ---------------------------------------------------------------------------
// Double-double, out of correctly-rounded primitives only
// ---------------------------------------------------------------------------
//
// EVERY multiply below is xsrMul and every fused multiply-add is xsrFma.  That
// is not style: twoProd's error term is `fma(a, b, -p)` and it is the exact
// residual of `p = a*b` ONLY IF `p` really was the rounded product.  A compiler
// permitted to contract `a*b` into a neighbouring add -- which -ffp-contract=
// fast is -- would silently make `p` something else and every digit below the
// leading one would be noise.  xsrMul's barrier is what forbids that.
//
// The additions and subtractions carry no barrier and need none: contraction
// only ever consumes a multiply, and no compiler is allowed to re-associate
// floating-point addition without -ffast-math, which this tree never sets.

struct FsDD {
    double hi;
    double lo;
};

/// Knuth's two-sum: `hi + lo` is a + b EXACTLY, for any finite a and b.
RASBERY_XSR_HD inline FsDD fsTwoSum(double a, double b) {
    const double s  = a + b;
    const double bb = s - a;
    const double e  = (a - (s - bb)) + (b - bb);
    return {s, e};
}

/// Dekker's fast two-sum.  Valid only for |a| >= |b|, which every use below
/// establishes rather than assumes.
RASBERY_XSR_HD inline FsDD fsQuickTwoSum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

/// Exact product, via the fma residual.
RASBERY_XSR_HD inline FsDD fsTwoProd(double a, double b) {
    const double p = xsrMul(a, b);
    return {p, xsrFma(a, b, -p)};
}

RASBERY_XSR_HD inline FsDD fsDDAdd(FsDD a, FsDD b) {
    const FsDD   s = fsTwoSum(a.hi, b.hi);
    const double e = s.lo + (a.lo + b.lo);
    return fsQuickTwoSum(s.hi, e);
}

RASBERY_XSR_HD inline FsDD fsDDAddD(FsDD a, double b) {
    const FsDD   s = fsTwoSum(a.hi, b);
    const double e = s.lo + a.lo;
    return fsQuickTwoSum(s.hi, e);
}

RASBERY_XSR_HD inline FsDD fsDDNeg(FsDD a) { return {-a.hi, -a.lo}; }

RASBERY_XSR_HD inline FsDD fsDDSub(FsDD a, FsDD b) { return fsDDAdd(a, fsDDNeg(b)); }

RASBERY_XSR_HD inline FsDD fsDDMul(FsDD a, FsDD b) {
    const FsDD   p = fsTwoProd(a.hi, b.hi);
    const double e = p.lo + xsrFma(a.hi, b.lo, xsrMul(a.lo, b.hi));
    return fsQuickTwoSum(p.hi, e);
}

RASBERY_XSR_HD inline FsDD fsDDMulD(FsDD a, double b) {
    const FsDD   p = fsTwoProd(a.hi, b);
    const double e = xsrFma(a.lo, b, p.lo);
    return fsQuickTwoSum(p.hi, e);
}

/// Newton-corrected division.  `q1` is the double quotient, the remainder is
/// formed exactly and its own quotient is the correction.
RASBERY_XSR_HD inline FsDD fsDDDiv(FsDD a, FsDD b) {
    const double q1 = a.hi / b.hi;
    const FsDD   r  = fsDDSub(a, fsDDMulD(b, q1));
    const double q2 = r.hi / b.hi;
    const FsDD   t  = fsQuickTwoSum(q1, q2);
    const FsDD   r2 = fsDDSub(a, fsDDMul(b, t));
    const double q3 = r2.hi / b.hi;
    return fsQuickTwoSum(t.hi, t.lo + q3);
}

/// `1/c` to double-double accuracy, for a c that is exact as a double.
///
/// WHY THE SERIES COEFFICIENTS ARE NOT PLAIN DOUBLES.  `1/(2i+1)` rounded to a
/// double carries 2^-53 of relative error, and it multiplies z^i where
/// |z| <= 2^-4.54 -- so the i-th coefficient's own error lands in the sum at
/// 2^(-53-4.54i).  The i=2 term alone puts it at 2^-64, forty bits above the
/// target, and no amount of double-double accumulation removes an error that
/// was already in the coefficient.  `fma(-c, hi, 1.0)` is the exact residual of
/// the rounded reciprocal, so one division corrects it.
RASBERY_XSR_HD inline FsDD fsRecipDD(double c) {
    const double hi = 1.0 / c;
    const double lo = xsrFma(-c, hi, 1.0) / c;
    return {hi, lo};
}

/// Round a normalised double-double to the nearest double.  `hi + lo` IS the
/// value, so one correctly-rounded addition is the rounding.
RASBERY_XSR_HD inline double fsDDToDouble(FsDD a) { return a.hi + a.lo; }

/// TRUE when the double-double is too close to a rounding boundary for its own
/// accuracy to decide which way to go -- see the header note.  `rel` is the
/// relative accuracy the caller believes it achieved.
///
/// The residual of the final rounding is exactly `(hi - y) + lo`; the decision
/// is safe when that residual's distance from the half-ulp boundary exceeds the
/// evaluation's own uncertainty.
RASBERY_XSR_HD inline bool fsDDRoundingUncertain(FsDD a, double rel) {
    const double y = a.hi + a.lo;
    if (!(y == y) || y == 0.0) return false;
    const double e   = (a.hi - y) + a.lo;
    const double ay  = y < 0.0 ? -y : y;
    const double ae  = e < 0.0 ? -e : e;
    // ulp(y), for a NORMAL y: 2^(exponent(y) - 52).  A subnormal or a value
    // within 52 binades of the bottom has no representable ulp scale here, and
    // fsPow2 would shift a negative exponent field; those are not arguments this
    // resolver's log or cbrt can return, so the honest answer is "no verdict"
    // rather than a computed one from an expression that does not hold.
    const int    ey  = static_cast<int>((fsToBits(ay) >> 52) & 0x7ffull) - 1023;
    if (ey < -970) return false;
    const double ulp = fsPow2(ey - 52);
    const double d   = ae - 0.5 * ulp;
    return (d < 0.0 ? -d : d) < rel * ay;
}

// ---------------------------------------------------------------------------
// log
// ---------------------------------------------------------------------------

/// ln 2, to ~107 bits.
RASBERY_XSR_HD inline FsDD fsLn2DD() {
    return {6.93147180559945286227e-01, 2.31904681384629956e-17};
}

/// The correctly-rounded natural logarithm, or as near to it as ~2^-100 of
/// working accuracy reaches.
///
/// THE ARGUMENT REDUCTION.  `x = m * 2^k` with m in [sqrt(1/2), sqrt(2)), so
/// `r = m - 1` lies in [-0.2929, 0.4143] and is EXACT (Sterbenz: 1/2 <= m <= 2).
/// Then log x = k*ln2 + log1p(r), and log1p is evaluated through the atanh
/// form, whose series has only ODD powers and therefore converges twice as fast
/// as log1p's own:
///
///     s = r / (2 + r),   |s| <= 0.2071
///     log1p(r) = 2*atanh(s) = 2s * (1 + s^2/3 + s^4/5 + ...)
///
/// With s^2 <= 0.0429 = 2^-4.54, twenty-six terms take the tail below 2^-118,
/// which is under the double-double arithmetic's own noise floor.  The Horner
/// fold runs in double-double for exactly that reason: a double-precision tail
/// would cap the whole evaluation at ~2^-59 and no rounding decision could be
/// made from it.
///
/// Domain: x must be finite and strictly positive.  The callers in
/// FlatXsStreamKernel.h floor their arguments (kSpectralLogDensityFloor, the
/// 1e-30 flux guard) before reaching here, exactly as the host does, so this
/// body does not re-guard what the quotation does not.
RASBERY_XSR_HD inline FsDD fsExactLogDD(double x) {
    unsigned long long b = fsToBits(x);
    int                k = static_cast<int>((b >> 52) & 0x7ffull) - 1023;
    if (k == -1023) { // subnormal: scale into the normal range, exactly
        const double xs = xsrMul(x, fsPow2(54));
        b               = fsToBits(xs);
        k               = static_cast<int>((b >> 52) & 0x7ffull) - 1023 - 54;
    }
    double m = fsFromBits((b & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
    if (m > 1.41421356237309514547e+00) { // sqrt(2), rounded down
        m = xsrMul(m, 0.5);               // exact
        k += 1;
    }

    const double r = m - 1.0; // exact
    const FsDD   s = fsDDDiv(FsDD{r, 0.0}, fsTwoSum(2.0, r));
    const FsDD   z = fsDDMul(s, s);

    // P(z) = sum_{i=0..kLogTerms} z^i / (2i+1), Horner from the tail.
    //
    // TWENTY-FIVE TERMS, and the count is set by the tail and not by taste:
    // z^26 <= 2^-118, which is under the double-double arithmetic's own noise
    // floor, so a twenty-sixth term could not be observed.  The coefficients up
    // to i = 9 are double-doubles (fsRecipDD says why); past that a rounded
    // double's error is already below 2^-95 and a second division would buy
    // nothing.
    constexpr int kLogTerms = 25;
    FsDD          p         = fsRecipDD(2.0 * kLogTerms + 1.0);
    for (int i = kLogTerms - 1; i >= 0; --i) {
        p              = fsDDMul(p, z);
        const double c = 2.0 * static_cast<double>(i) + 1.0;
        p              = (i <= 9) ? fsDDAdd(p, fsRecipDD(c)) : fsDDAddD(p, 1.0 / c);
    }

    FsDD l = fsDDMul(s, p);
    l      = fsDDAdd(l, l); // 2*s*(1 + ...)
    if (k != 0) l = fsDDAdd(l, fsDDMulD(fsLn2DD(), static_cast<double>(k)));
    return l;
}

RASBERY_XSR_HD inline double fsExactLog(double x) {
    return fsDDToDouble(fsExactLogDD(x));
}

// ---------------------------------------------------------------------------
// cbrt
// ---------------------------------------------------------------------------

/// The correctly-rounded cube root.
///
/// THE REDUCTION IS EXACT AND THE REFINEMENT IS ONE STEP.  `x = f * 2^e` with f
/// in [1, 2); writing e = 3q + t with t in {0, 1, 2} makes `g = f * 2^t` exact
/// and in [1, 8), so cbrt(x) = 2^q * cbrt(g) and the outer scaling is a
/// power of two -- exact, and unable to move the rounding.
///
/// The seed is the classic exponent-third bit trick, three double Newton steps
/// take it to ~2^-52, and ONE double-double Newton step squares that to ~2^-104
/// -- which is the arithmetic's own floor, so a second step would buy nothing.
RASBERY_XSR_HD inline FsDD fsExactCbrtDD(double x) {
    const bool   neg = x < 0.0;
    const double a   = neg ? -x : x;

    unsigned long long b = fsToBits(a);
    int                e = static_cast<int>((b >> 52) & 0x7ffull) - 1023;
    int                sub = 0;
    if (e == -1023) { // subnormal
        const double as = xsrMul(a, fsPow2(60));
        b               = fsToBits(as);
        e               = static_cast<int>((b >> 52) & 0x7ffull) - 1023;
        sub             = 20; // 60/3
    }
    const double f = fsFromBits((b & 0x000fffffffffffffull) | 0x3ff0000000000000ull);

    const int q = (e >= 0) ? (e / 3) : -((-e + 2) / 3);
    const int t = e - 3 * q; // 0, 1 or 2
    const double g = xsrMul(f, fsPow2(t)); // exact, g in [1, 8)

    // Seed: divide the biased exponent by three in the bit pattern.  Good to
    // ~5 %, which three Newton steps take to full double accuracy.
    double y = fsFromBits(fsToBits(g) / 3ull + 0x2a9f76253119d328ull);
    for (int i = 0; i < 4; ++i) {
        const double y2 = xsrMul(y, y);
        y               = y - (y - g / y2) * (1.0 / 3.0);
    }

    // One Newton step in double-double: y <- y + (g - y^3) / (3 y^2).
    const FsDD Y  = FsDD{y, 0.0};
    const FsDD Y2 = fsDDMul(Y, Y);
    const FsDD Y3 = fsDDMul(Y2, Y);
    const FsDD num = fsDDAddD(fsDDNeg(Y3), g);
    const FsDD den = fsDDMulD(Y2, 3.0);
    FsDD       res = fsDDAdd(Y, fsDDDiv(num, den));

    const double scale = fsPow2(q - sub);
    res                = fsDDMulD(res, scale); // exact: scale is a power of two
    return neg ? fsDDNeg(res) : res;
}

RASBERY_XSR_HD inline double fsExactCbrt(double x) {
    if (x == 0.0) return x; // preserves the signed zero, as cbrt must
    return fsDDToDouble(fsExactCbrtDD(x));
}

} // namespace rasbery::flatxs_stream
