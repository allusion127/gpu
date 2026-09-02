#pragma once

// A VERBATIM QUOTATION of the T/H arithmetic in XSSet.cpp, and a fixture to
// drive it -- WP22.
//
// DELIBERATELY IN ITS OWN TRANSLATION UNIT, and this header must never be
// included by one that also includes ThKernel.h.  With both in one TU gcc
// common-subexpressions across them and changes the QUOTATION's contraction, so
// the mining would score a reference that is no longer the production one.
// CmfdOuterReference.h says the same thing and Task 4 paid for the lesson.
//
// WHAT IS QUOTED.  The floating-point expressions of milk::Table::Get,
// XSSet::GetTfuel, the UpdateTH power fold, the SolveTH channel sweep and the
// under-relaxation blend, written exactly as those files write them: plain `+`,
// `*` and `/`, no barriers, no std::fma, nothing that would tell the compiler
// what to do.  Whatever gcc contracts HERE is what the production host loop
// contracts, because it is the same text under the same flags.
//
// WHAT IS NOT QUOTED.  Anything that is not arithmetic: the OpenMP pragmas, the
// warning print, the scratch-vector resizing, the Geometry accessor calls.  A
// quotation of those would be a second implementation of them, which is the
// thing this file exists to avoid being.

#include <cstddef>
#include <vector>

namespace thref {

/// One table, in the shape milk::Table stores it (row-major over y then x).
struct Table {
    std::vector<double> x, y, v;
    int                 nx = 0, ny = 0;
};

/// A deck-shaped operand set: small enough to mine in microseconds, wide enough
/// that every branch of the quoted bodies is reached.
struct Fixture {
    int nxy = 0, nz = 0, nxyz = 0, ng = 0, kbc = 0, kec = 0;

    double pressure             = 0.0;
    double inlet_h              = 0.0;
    double h_table_max          = 0.0;
    double norm                 = 0.0;
    double flow_per_channel     = 0.0;
    double fuel_temp_rise_scale = 0.0;
    double th_relaxation        = 0.0;

    std::vector<double> xskf, phif, vol, hmesh_x, hmesh_y, hz;
    std::vector<int>    burn;
    std::vector<double> node_power;
    std::vector<double> tful_old, tmod_old, dmod_old;
    std::vector<double> tful, tmod, dmod;

    Table mod_t, mod_rho, tf;
};

/// Build the fixture.  Deterministic, no I/O: the operands are a reproducible
/// pseudo-random field over ranges the real deck occupies, including channels
/// driven past the table ceiling so the clamp branch is scored too.
Fixture buildFixture(int nxy, int nz, int kbc, int kec);

// --- the quotations -------------------------------------------------------
//
// Each takes the fixture and writes into caller-owned output, so the miner can
// run the shipped body over the same operands and compare bit patterns.

/// milk::Table::Get(x, y), quoted.
double refTableGet(const Table& t, double x, double y);

/// XSSet::GetTfuel(burnup, LPD), quoted.
double refGetTfuel(const Table& tf, double burnup, double lpd);

/// The UpdateTH per-node power fold, quoted.  `out` is [nxyz].
void refNodePower(const Fixture& f, double* out);

/// The SolveTH channel sweep for one radial position, quoted.  Writes into the
/// caller's tmod/dmod/tful and returns the channel's table-ceiling triple.
struct Overflow {
    int    count = 0;
    double worst = 0.0;
    int    node  = -1;
};
Overflow refChannelSweep(const Fixture& f, int l, const double* node_power, double* tmod,
                         double* dmod, double* tful);

/// The under-relaxation blend, quoted.
double refRelax(double old_value, double new_value, double w);

} // namespace thref
