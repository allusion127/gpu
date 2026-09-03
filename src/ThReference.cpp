// The verbatim T/H quotation.  See ThReference.h for why this is its own
// translation unit, what may and may not appear in it, and why the CALL GRAPH
// below is as much a part of the quotation as the expressions are.
//
// NOTHING IN THIS FILE MAY INCLUDE ThKernel.h.  That is the whole contract.

#include "ThReference.h"

#include "ThFuelRods.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace thref {

namespace {

/// milk::Table::FindLowerIndex, quoted.  Integer arithmetic only, so it carries
/// no contraction site; it is here because Table::Get calls it and a quotation
/// that called something else would be quoting a different function.
std::size_t findLowerIndex(const std::vector<double>& axis, double value) {
    std::size_t lo = 0;
    std::size_t hi = axis.size() - 1;
    while (lo + 1 < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (axis[mid] <= value)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/// milk::Table::Get(x, y), quoted.
///
/// INLINE AND INTERNAL LINKAGE, WHICH IS HARDENING AND WAS *NOT* THE BUG.
/// milk::Table::Get is a class-body inline in include/milk.h, so every production
/// call site -- GetTmod, GetDmod and the tf lookup inside GetTfuel, all of them
/// inlined into XSSet::SolveTH -- gets its own copy for gcc to contract in
/// context, and CudaThBackend.h records that an out-of-line spelling pins
/// TH_LERP_X0 differently (17 mismatches versus 0 over a 20k sweep).  Matching
/// that shape is right on its own merits.
///
/// BUT IT DID NOT FIX 0x54, AND THAT WAS MEASURED RATHER THAN ASSUMED.  With the
/// operand fix in buildFixture below and the OLD out-of-line, per-channel shape,
/// the descent still mines 0x57 with a zero residual on g++ 13.3 -O3
/// -march=native, scoring identically to this shape (0x00: 415, 0x54: 37, 0x57: 0,
/// 0x1f3: 186).  The bug was never the inlining context.  It was that the fixture
/// never reached the two sites at all -- see buildFixture's `norm` and tf-axis
/// comments, and thmine::dontCareMask, which is the check that now says so.
inline double tableGet(const Table& t, double x, double y) {
    const std::size_t nx = t.x.size();
    const std::size_t ny = t.y.size();

    x = std::clamp(x, t.x[0], t.x[nx - 1]);
    y = std::clamp(y, t.y[0], t.y[ny - 1]);

    std::size_t ix = findLowerIndex(t.x, x);
    std::size_t iy = findLowerIndex(t.y, y);
    if (ix >= nx - 1) ix = nx - 2;
    if (iy >= ny - 1) iy = ny - 2;

    const double x0 = t.x[ix], x1 = t.x[ix + 1];
    const double y0 = t.y[iy], y1 = t.y[iy + 1];
    const double fx = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
    const double fy = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;

    const double z00 = t.v[iy * nx + ix];
    const double z01 = t.v[iy * nx + ix + 1];
    const double z10 = t.v[(iy + 1) * nx + ix];
    const double z11 = t.v[(iy + 1) * nx + ix + 1];

    const double z0 = z00 + fx * (z01 - z00);
    const double z1 = z10 + fx * (z11 - z10);
    return z0 + fy * (z1 - z0);
}

/// XSSet::GetTmod / XSSet::GetDmod, quoted -- INCLUDING THE ARGUMENT SWAP.  The
/// production wrappers take (enthalpy, pressure) and hand Table::Get
/// (pressure, enthalpy); a quotation that called the table directly with the
/// operands already in table order would be one inlining step shorter than the
/// thing it claims to record.
inline double getTmod(const Fixture& f, double enthalpy_kJ, double pressure_mpa) {
    return tableGet(f.mod_t, pressure_mpa, enthalpy_kJ);
}

inline double getDmod(const Fixture& f, double enthalpy_kJ, double pressure_mpa) {
    return tableGet(f.mod_rho, pressure_mpa, enthalpy_kJ);
}

/// XSSet::GetTfuel(burnup, LPD), quoted -- also a class-body inline in
/// production, also inlined into SolveTH's fuel loop.
///
/// THIS IS THE ONLY QUERY THAT CAN PIN TH_LERP_X0 / TH_LERP_X1, which is why
/// buildFixture works as hard as it does to reach it.  The mod_t / mod_rho
/// queries cannot: the deck's pressure axis has a knot at 15.5 MPa exactly
/// (`fx == 0`) and the fixture's puts 15.5 dead centre (`fx == 0.5`, and
/// `0.5 * d` is exact), so both x-lerps are the identity there in both.  Only the
/// tf table's LPD axis is queried off-knot -- and only when the node powers put
/// `lpd` between the first and last knot, which until this commit they never did.
inline double getTfuel(const Table& tf, double burnup, double LPD) {
    constexpr double first_lpd = 50.0;
    const double     safe_lpd  = std::max(0.0, LPD);
    const double     rise      = tableGet(tf, std::max(first_lpd, safe_lpd), burnup);
    return safe_lpd < first_lpd ? rise * safe_lpd / first_lpd : rise;
}

/// A deterministic 64-bit LCG.  Not a physics choice: the fixture just has to be
/// the same field on every host and in every process, or the mined mask would
/// depend on which operands the run happened to see.
struct Rng {
    std::uint64_t s;
    double        next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((s >> 11) & ((1ull << 53) - 1ull)) /
               static_cast<double>(1ull << 53);
    }
};

Table buildTable(Rng& rng, double x0, double x1, int nx, double y0, double y1, int ny,
                 double v0, double v1) {
    Table t;
    t.nx = nx;
    t.ny = ny;
    t.x.resize(static_cast<std::size_t>(nx));
    t.y.resize(static_cast<std::size_t>(ny));
    t.v.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    for (int i = 0; i < nx; ++i)
        t.x[static_cast<std::size_t>(i)] = x0 + (x1 - x0) * i / (nx - 1);
    for (int j = 0; j < ny; ++j)
        t.y[static_cast<std::size_t>(j)] = y0 + (y1 - y0) * j / (ny - 1);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            t.v[static_cast<std::size_t>(j) * nx + i] = v0 + (v1 - v0) * rng.next();
    return t;
}

} // namespace

// ---------------------------------------------------------------------------
// The quotations
// ---------------------------------------------------------------------------

void refNodePower(const Fixture& f, double* out) {
    const int ng   = f.ng;
    const int nxyz = f.nxyz;
    for (int lk = 0; lk < nxyz; ++lk) {
        double power_density = 0.0;
        for (int ig = 0; ig < ng; ++ig)
            power_density += f.xskf[static_cast<std::size_t>(ig) * nxyz + lk] *
                             f.phif[static_cast<std::size_t>(lk) * ng + ig];
        out[lk] = power_density * f.vol[static_cast<std::size_t>(lk)];
    }
}

/// XSSet::SolveTH, quoted.  THE CHANNEL LOOP IS INSIDE THIS FUNCTION because it
/// is inside that one: gcc's contraction choices are made over the whole
/// function it has after inlining, and a per-channel entry point is a smaller
/// function than SolveTH with fewer inlined copies of Table::Get in it.  The
/// ceiling triple is accumulated in SolveTH's own locals, in its own order, so
/// the first-wins tie-break the host prints is the one that is scored.
Overflow refSolveTH(const Fixture& f, const double* node_power, double* tmod, double* dmod,
                    double* tful) {
    const int    nxy              = f.nxy;
    const int    nz               = f.nz;
    const int    kbc              = f.kbc;
    const int    kec              = f.kec;
    const double pressure         = f.pressure;
    const double norm             = f.norm;
    const double flow_per_channel = f.flow_per_channel;
    const double h_table_max      = f.h_table_max;

    int    n_over     = 0;
    double worst_h    = 0.0;
    int    worst_node = -1;

    for (int l = 0; l < nxy; ++l) {
        double h_cur = f.inlet_h;

        // Bottom reflector
        for (int k = 0; k < kbc; ++k) {
            const int lk = l + k * nxy;
            tmod[lk]     = getTmod(f, h_cur, pressure);
            dmod[lk]     = getDmod(f, h_cur, pressure);
        }

        // Fuel region
        for (int k = kbc; k < kec; ++k) {
            const int    lk     = l + k * nxy;
            const double P_node = node_power[lk] * norm;
            if (P_node <= 0.0) {
                tmod[lk] = getTmod(f, h_cur, pressure);
                dmod[lk] = getDmod(f, h_cur, pressure);
                continue;
            }
            const double dh =
                P_node / (flow_per_channel * f.hmesh_x[static_cast<std::size_t>(l)] *
                          f.hmesh_y[static_cast<std::size_t>(l)]);
            const double h_out = h_cur + dh;
            const double h_avg = 0.5 * (h_cur + h_out);
            if (h_avg > h_table_max) {
                ++n_over;
                if (h_avg > worst_h) {
                    worst_h    = h_avg;
                    worst_node = lk;
                }
            }
            tmod[lk]         = getTmod(f, h_avg, pressure);
            dmod[lk]         = getDmod(f, h_avg, pressure);
            const double lpd = 1000.0 * P_node / (f.fuel_rods_per_node * f.hz[static_cast<std::size_t>(k)]);
            const double bu  = f.burn[static_cast<std::size_t>(lk)] / 1000.0;
            tful[lk]         = tmod[lk] + f.fuel_temp_rise_scale * getTfuel(f.tf, bu, lpd);
            h_cur            = h_out;
        }

        // Top reflector
        for (int k = kec; k < nz; ++k) {
            const int lk = l + k * nxy;
            if (k == kec && h_cur > h_table_max) {
                ++n_over;
                if (h_cur > worst_h) {
                    worst_h    = h_cur;
                    worst_node = lk;
                }
            }
            tmod[lk] = getTmod(f, h_cur, pressure);
            dmod[lk] = getDmod(f, h_cur, pressure);
        }
    }

    Overflow over;
    over.count = n_over;
    over.worst = worst_h;
    over.node  = worst_node;
    return over;
}

double refRelax(double old_value, double new_value, double w) {
    return (1.0 - w) * old_value + w * new_value;
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

Fixture buildFixture(int nxy, int nz, int kbc, int kec) {
    Fixture f;
    f.nxy  = nxy;
    f.nz   = nz;
    f.nxyz = nxy * nz;
    f.ng   = 2;
    f.kbc  = kbc;
    f.kec  = kec;

    Rng rng{0x9e3779b97f4a7c15ull};

    // THE AXES ARE CHOSEN TO PIN SITES, NOT TO IMITATE A DECK, AND THAT IS THE
    // WHOLE JOB OF A MINING FIXTURE.  Three sites were measured to be
    // DON'T-CARES under the first spelling of this function, for three different
    // and equally uninteresting reasons, and a don't-care site is a site the
    // mask cannot pin on this host:
    //
    //   * the pressure axis contained 15.5 exactly, so `fx` came out 0 at every
    //     mod_t / mod_rho query and both x-lerps were the identity.  The axis
    //     below straddles 15.5 instead.
    //   * the tf table's LPD axis stopped at the heat rates the fixture thought
    //     it was generating, so that lerp was an identity too.  The table now
    //     runs to 900 W/cm.  (The other half of that repair -- "a tenth of the
    //     nodes are driven below 50" -- was written into `Fixture::node_power`,
    //     WHICH NOTHING READS.  See below.)
    //   * `fuel_temp_rise_scale` was 1.0, which makes `tmod + 1.0 * rise` exact
    //     under either form.  The production default is also 1.0, so the site is
    //     a genuine don't-care THERE -- but a deck that sets the multiplier
    //     would run a site nobody had pinned, so the fixture uses 1.03.
    //
    // AND ALL THREE OF THOSE REPAIRS MISSED THE ONE THAT MATTERED, which is why
    // this block now ends in a MEASUREMENT rather than an argument.  Every claim
    // above was about an AXIS.  None of them checked what the sweep actually
    // QUERIES, and the sweep does not read `Fixture::node_power` at all -- it
    // derives node power from xskf/phif/vol through the shipped fold, exactly as
    // XSSet::UpdateTH does.  With `norm` at 1e-3 the resulting `lpd` came out
    // under 0.03 W/cm at all 1,280 fuel nodes, so every tf query clamped to the
    // first LPD knot with `fx == 0`, and TH_LERP_X0 / TH_LERP_X1 were UNREACHABLE:
    // 0x54 and 0x57 both scored ZERO, and the descent returned whichever its seed
    // began at.  238 then ran the deck and disagreed by 866 lines.
    //
    // STRADDLING A KNOT IS NOT THE SAME AS REACHING A SITE.  15.5 sits DEAD CENTRE
    // of the 13..17 MPa five-knot axis below, so `fx` is exactly 0.5 and
    // `0.5 * (z01 - z00)` is exact under either form: the mod_t / mod_rho x-lerps
    // are still don't-cares here.  The axis is left alone anyway, and
    // deliberately, because the production deck is degenerate the same way --
    // include/Database/mod_t.csv has a knot at 15.5 exactly, so the deck's `fx` is
    // 0 and its x-lerps are identities too.  Pinning bits 0-1 from a site the deck
    // cannot observe would impose a constraint production does not have, and if
    // gcc contracted the mod_* copies differently from the tf copy the single
    // two-bit field could not express both -- the mining would then report an
    // unremovable residual on a mask that reproduces the deck exactly.  A DECK
    // WITH AN OFF-KNOT PRESSURE would run a site pinned only through the tf
    // context; that is a named residual risk, recorded in
    // docs/WP22_TH_SEARCH_GPU_20260902_KO.md section 2.2, not a claim.
    //
    // TWO SITES STAY DON'T-CARES AND ALWAYS WILL, both by arithmetic rather than
    // by coverage: `0.5 * (a + b)` is exact in binary floating point under either
    // spelling, and GetTfuel's `rise * safe_lpd` has no add to be contracted into.
    // No fixture can pin either, so TH_EXPECTED_DONT_CARE names them both --
    // TH_TFUEL_LINEAR because thmine::dontCareMask put it there on the first run
    // against this repaired fixture, not because anybody expected it.  The census
    // runs at every startup and ThFormMiner.cpp warns on anything else, because
    // "residual zero" and "the reference could tell the alternatives apart" are
    // different facts and only the second one makes a mined mask a measurement.
    f.mod_t   = buildTable(rng, 13.0, 17.0, 5, 1000.0, 1600.0, 41, 550.0, 620.0);
    f.mod_rho = buildTable(rng, 13.0, 17.0, 5, 1000.0, 1600.0, 41, 0.60, 0.78);
    // THE tf BURNUP AXIS IS IN GWd/tHM, WHICH IS THE UNIT THE QUERY ARRIVES IN.
    // SolveTH passes `bu = burnup[lk] / 1000.0`, so bu runs 0..60 -- and
    // include/Database/tf.csv's first column agrees (0, 5.66, 10.27, ...).  This
    // axis used to run 0..60000, i.e. MWd/tHM, so EVERY query landed inside the
    // first interval and `fy` came out in [0, 0.012].  That does not make the
    // y-lerp unreachable, but it attenuates the second x-lerp by a factor of ~2^7:
    // a difference in `z1` reaches the answer only through `fy * (z1 - z0)`, so
    // TH_LERP_X1 was measurably a DON'T-CARE (mined 0x55, and 0x57 scored 0 too)
    // while TH_LERP_X0 was pinned.  With the axis in the right unit `fy` is
    // generic and both x-lerps are pinned.
    f.tf      = buildTable(rng, 50.0, 900.0, 18, 0.0, 60.0, 13, 80.0, 900.0);

    f.pressure             = 15.5;
    f.inlet_h              = 1290.0;
    f.h_table_max          = f.mod_t.y.back();
    // `norm` IS NOT A FREE PARAMETER, AND THE OLD VALUE MADE THE FIXTURE INERT.
    // SolveTH's norm is `actual_power / total_raw_power`, which on a deck is of
    // order 1e2: it scales a raw fold of order 10 up to a node power of order
    // 1e2..1e3 kW.  This fixture used 1.0e-3, five orders of magnitude the other
    // way, and the consequence was measured rather than argued -- every one of
    // the 1,280 fuel nodes came out at `lpd < 0.03`, so EVERY tf query clamped to
    // the first LPD knot, `fx` was 0 at every one of them, and both x-lerps were
    // the identity.  That, and not the inlining shape, is why the mining could
    // return either 0x54 or 0x57 with a zero residual and returned whichever its
    // seed started from.
    f.norm                 = 50.0;
    f.flow_per_channel     = 0.37;
    // The LEGACY divisor, deliberately: the mined form mask must not move
    // when a deck changes the rod count, and the fixture is the norm.
    f.fuel_rods_per_node   = rasbery::th::kLegacyFuelRodsPerNode;
    f.fuel_temp_rise_scale = 1.03;
    f.th_relaxation        = 0.85;

    const std::size_t n = static_cast<std::size_t>(f.nxyz);
    f.xskf.resize(static_cast<std::size_t>(f.ng) * n);
    f.phif.resize(n * static_cast<std::size_t>(f.ng));
    f.vol.resize(n);
    f.hmesh_x.resize(n);
    f.hmesh_y.resize(n);
    f.hz.resize(static_cast<std::size_t>(nz));
    f.burn.resize(n);
    f.node_power.resize(n);
    f.tful_old.resize(n);
    f.tmod_old.resize(n);
    f.dmod_old.resize(n);
    f.tful.resize(n);
    f.tmod.resize(n);
    f.dmod.resize(n);

    for (std::size_t i = 0; i < f.xskf.size(); ++i) f.xskf[i] = 1.0e-3 + 4.0e-3 * rng.next();
    for (std::size_t i = 0; i < f.phif.size(); ++i) f.phif[i] = 0.2 + 2.0 * rng.next();
    for (std::size_t i = 0; i < n; ++i) {
        f.vol[i]     = 800.0 + 400.0 * rng.next();
        f.hmesh_x[i] = 20.0 + 2.0 * rng.next();
        f.hmesh_y[i] = 20.0 + 2.0 * rng.next();
        f.burn[i]    = static_cast<int>(60000.0 * rng.next());
        f.tful_old[i] = 600.0 + 600.0 * rng.next();
        f.tmod_old[i] = 560.0 + 40.0 * rng.next();
        f.dmod_old[i] = 0.60 + 0.15 * rng.next();
        f.tful[i]     = f.tful_old[i];
        f.tmod[i]     = f.tmod_old[i];
        f.dmod[i]     = f.dmod_old[i];
    }
    for (int k = 0; k < nz; ++k) f.hz[static_cast<std::size_t>(k)] = 10.0 + 10.0 * rng.next();

    // THE POWER POPULATIONS GO IN xskf, BECAUSE xskf IS WHAT THE SWEEP READS.
    //
    // `Fixture::node_power` is written by nobody and read by nobody: scoreMask
    // derives the node power from xskf/phif/vol through the SHIPPED fold, which
    // is what XSSet::UpdateTH does, and hands THAT to the sweep.  The previous
    // spelling wrote its three carefully-argued populations into
    // `f.node_power` -- a field the miner never looks at -- and then mined
    // against whatever the fold happened to produce.  A fixture population that
    // reaches no branch is a comment, not a fixture, and this one did not even
    // reach the array.
    //
    // Each population below is here to reach a branch, and every one of them was
    // CONFIRMED reached (P<=0: 79 nodes, sub-50: 158, interior: 967, LPD clamp:
    // 76, of 1,280 fuel nodes) rather than asserted:
    //
    //   scale 0     an unfuelled node: the `P_node <= 0` skip in the fuel loop,
    //               which the old fixture reached ZERO times;
    //   scale 0.02  GetTfuel's sub-50 W/cm linear continuation (TH_TFUEL_LINEAR);
    //   scale 3     LPD past the tf table's last knot, so the x-clamp path runs;
    //   scale 30    one hot channel, whose enthalpy runs off the water-property
    //               table so the ceiling clamp and the (count, worst, node)
    //               overflow triple are scored -- also never reached before;
    //   scale 1     the interior bulk, whose `fx` is GENERIC.  This is the only
    //               population that pins TH_LERP_X0 / TH_LERP_X1 at all, and the
    //               old fixture had none of it.
    for (int lk = 0; lk < f.nxyz; ++lk) {
        const int    l     = lk % nxy;
        const double r     = rng.next();
        double       scale = 1.0;
        if (l == 0)
            scale = 30.0;
        else if (r < 0.08)
            scale = 0.0;
        else if (r < 0.20)
            scale = 0.02;
        else if (r < 0.24)
            scale = 3.0;
        for (int ig = 0; ig < f.ng; ++ig)
            f.xskf[static_cast<std::size_t>(ig) * n + static_cast<std::size_t>(lk)] *= scale;
    }

    return f;
}

} // namespace thref
