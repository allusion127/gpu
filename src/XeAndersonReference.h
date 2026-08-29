#pragma once

// Verbatim CPU quotation of the Anderson algebra -- Rev.7.1 Task 13.
//
// WHAT A "QUOTATION" IS FOR.  XeKernel.h's xeDotChunk and xeCandidateOrdinal
// are the SHIPPED bodies: one source, built by g++ for the host harness and by
// nvcc for the device, with every multiply-add written out explicitly so the
// two agree.  Writing them out explicitly is only correct if it MATCHES what
// the host compiler did to the production loops in Driver.h, and that is a
// property of the build machine, not of the physics (CmfdOuterFormMiner.cpp
// documents the 0x6-vs-0x7 case that cost a campaign).  So the mask is mined,
// and mining needs a reference to score against: these two functions, written
// exactly as Driver.h writes them, compiled at the project's own -O3.
//
// ITS OWN TRANSLATION UNIT, AND THAT IS NOT AN ACCIDENT.  With the quotation
// and the shipped body in one TU, gcc common-subexpressions across them and
// changes the QUOTATION's contraction -- so the mining would score a reference
// that is no longer the production one.  CmfdOuterReference.h says the same
// thing and Task 4 paid for the lesson.  Hence: declarations here, definitions
// in XeAndersonReference.cpp, and nothing in that file includes XeKernel.h.
//
// THE QUOTATION IS NOT THE PRODUCTION CODE, AND THAT GAP IS GATED.  Driver.h's
// XeDot and candidate loop stay exactly where they are -- moving them into this
// TU would change the production path's own inlining and contraction, which is
// the one thing a bit-identity claim cannot survive.  What keeps the two in
// step instead is tools/test_xe_gpu_contract.py, which reads the expressions
// out of both files and fails when they stop being the same text.

#include <vector>

namespace xeref {

/// Operands with the shape and the DYNAMIC RANGE the production arrays have:
/// Xe-chain densities are ~1e-9 atoms/barn-cm and the residual differences the
/// window holds are many orders below them, which is exactly the regime where a
/// fused multiply-add and a rounded one part company.  A fixture of same-sized
/// operands would score every mask identically and mine nothing.
struct Fixture {
    int                 n = 0;
    std::vector<double> a_i, a_x, a_m; ///< left operand of the inner product
    std::vector<double> b_i, b_x, b_m; ///< right operand
    std::vector<double> f_i, f_x, f_m; ///< F(x_k), the candidate's base point
    std::vector<double> d_i, d_x, d_m; ///< dF columns, [j*n + k], j = 0..1
    double              gamma[2]       = {0.0, 0.0};
    /// WP7-C.  Gram matrices and right-hand sides for the normal equations,
    /// six doubles per case, [6*case + slot] in XeDotSlot order.  A SEPARATE
    /// FIXTURE FROM THE VECTORS ABOVE because what it has to discriminate is
    /// different: the four sites in the 2x2 solve are only evaluated when the
    /// two-column branch is TAKEN, so every case here is deliberately
    /// well-conditioned.  A fixture of near-singular Gram matrices would fall
    /// to the one-column secant every time, which has no site, and would mine
    /// three of the four as don't-cares -- a statement about the fixture and
    /// not about the compiler.
    std::vector<double> alg;
    int                 alg_cases = 0;
};

/// Deterministic; the same fixture on every host and every run.
Fixture buildFixture(int n);

/// Driver.h XeDot over the ordinals [i0, i1), quoted verbatim.
double refDot(const Fixture& f, int i0, int i1);

/// TryAndersonXeStep's candidate loop, quoted verbatim.  Outputs are resized to
/// f.n.
void refCandidate(const Fixture& f, int ncol, std::vector<double>& ci,
                  std::vector<double>& cx, std::vector<double>& cm);

/// WP7-C.  TryAndersonXeStepGpu's normal equations, quoted verbatim: the two
/// coefficients, the projection, the determinant, and the `solved` flag the
/// conditioning test sets.  `det_out` is an output only so the mining can score
/// that site directly instead of through two divisions; the production arm
/// materialises det anyway, because it divides by it twice.
///
/// `min_gram` is a parameter for the same reason the shipped body takes one:
/// Driver.h owns XE_ANDERSON_MIN_GRAM and a second spelling here would be a
/// second opinion.
bool refAlgebra(const Fixture& f, int idx, int ncol, double min_gram, double gamma_out[2],
                double* proj_out, double* det_out);

} // namespace xeref
