// The verbatim T/H quotation.  See ThReference.h for why this is its own
// translation unit and what may and may not appear in it.
//
// NOTHING IN THIS FILE MAY INCLUDE ThKernel.h.  That is the whole contract.

#include "ThReference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace thref {

namespace {

/// milk::Table::FindLowerIndex, quoted.
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

double refTableGet(const Table& t, double x, double y) {
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

double refGetTfuel(const Table& tf, double burnup, double LPD) {
    constexpr double first_lpd = 50.0;
    const double     safe_lpd  = std::max(0.0, LPD);
    const double     rise      = refTableGet(tf, std::max(first_lpd, safe_lpd), burnup);
    return safe_lpd < first_lpd ? rise * safe_lpd / first_lpd : rise;
}

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

Overflow refChannelSweep(const Fixture& f, int l, const double* node_power, double* tmod,
                         double* dmod, double* tful) {
    Overflow     over;
    const int    nxy               = f.nxy;
    const int    nz                = f.nz;
    const int    kbc               = f.kbc;
    const int    kec               = f.kec;
    const double pressure          = f.pressure;
    const double norm              = f.norm;
    const double flow_per_channel  = f.flow_per_channel;
    const double h_table_max       = f.h_table_max;

    double h_cur = f.inlet_h;

    for (int k = 0; k < kbc; ++k) {
        const int lk = l + k * nxy;
        tmod[lk]     = refTableGet(f.mod_t, pressure, h_cur);
        dmod[lk]     = refTableGet(f.mod_rho, pressure, h_cur);
    }

    for (int k = kbc; k < kec; ++k) {
        const int    lk     = l + k * nxy;
        const double P_node = node_power[lk] * norm;
        if (P_node <= 0.0) {
            tmod[lk] = refTableGet(f.mod_t, pressure, h_cur);
            dmod[lk] = refTableGet(f.mod_rho, pressure, h_cur);
            continue;
        }
        const double dh =
            P_node / (flow_per_channel * f.hmesh_x[static_cast<std::size_t>(l)] *
                      f.hmesh_y[static_cast<std::size_t>(l)]);
        const double h_out = h_cur + dh;
        const double h_avg = 0.5 * (h_cur + h_out);
        if (h_avg > h_table_max) {
            ++over.count;
            if (h_avg > over.worst) {
                over.worst = h_avg;
                over.node  = lk;
            }
        }
        tmod[lk]         = refTableGet(f.mod_t, pressure, h_avg);
        dmod[lk]         = refTableGet(f.mod_rho, pressure, h_avg);
        const double lpd = 1000.0 * P_node / (62.0 * f.hz[static_cast<std::size_t>(k)]);
        const double bu  = f.burn[static_cast<std::size_t>(lk)] / 1000.0;
        tful[lk]         = tmod[lk] + f.fuel_temp_rise_scale * refGetTfuel(f.tf, bu, lpd);
        h_cur            = h_out;
    }

    for (int k = kec; k < nz; ++k) {
        const int lk = l + k * nxy;
        if (k == kec && h_cur > h_table_max) {
            ++over.count;
            if (h_cur > over.worst) {
                over.worst = h_cur;
                over.node  = lk;
            }
        }
        tmod[lk] = refTableGet(f.mod_t, pressure, h_cur);
        dmod[lk] = refTableGet(f.mod_rho, pressure, h_cur);
    }

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
    // WHOLE JOB OF A MINING FIXTURE.  Three of them were measured to be
    // DON'T-CARES under the first spelling of this function, for three different
    // and equally uninteresting reasons, and a don't-care site is a site the
    // mask cannot pin on this host:
    //
    //   * the pressure axis contained 15.5 exactly, so `fx` came out 0 at every
    //     mod_t / mod_rho query and both x-lerps were the identity.  The
    //     PRODUCTION table (include/Database/mod_t.csv) has 0.05 MPa knots
    //     around 15.5 and is degenerate the same way, which is precisely why the
    //     fixture must not be: a mask mined where a site is unreachable pins
    //     nothing there, and the day a deck runs off-knot the device would round
    //     somewhere the mining never looked.  The axis below straddles 15.5.
    //   * every node's linear heat rate landed ABOVE the tf table's last LPD
    //     knot, so `fx` came out 1 exactly and that lerp was the identity too --
    //     and GetTfuel's sub-50 W/cm linear continuation was never reached at
    //     all, though the comment claimed it was.  The table now runs to 900
    //     W/cm and a tenth of the nodes are driven below 50.
    //   * `fuel_temp_rise_scale` was 1.0, which makes `tmod + 1.0 * rise` exact
    //     under either form.  The production default is also 1.0, so the site is
    //     a genuine don't-care THERE -- but a deck that sets the multiplier
    //     would run a site nobody had pinned, so the fixture uses 1.03.
    //
    // TH_HAVG stays a don't-care and always will: `0.5 * (a + b)` is exact in
    // binary floating point under either spelling.  That is a property of the
    // arithmetic and not of the fixture, and mineStable() reports soundness by
    // residual rather than by mask equality for exactly this reason.
    f.mod_t   = buildTable(rng, 13.0, 17.0, 5, 1000.0, 1600.0, 41, 550.0, 620.0);
    f.mod_rho = buildTable(rng, 13.0, 17.0, 5, 1000.0, 1600.0, 41, 0.60, 0.78);
    f.tf      = buildTable(rng, 50.0, 900.0, 18, 0.0, 60000.0, 13, 80.0, 900.0);

    f.pressure             = 15.5;
    f.inlet_h              = 1290.0;
    f.h_table_max          = f.mod_t.y.back();
    f.norm                 = 1.0e-3;
    f.flow_per_channel     = 0.37;
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

    // Node powers, and every one of these three populations is here to reach a
    // branch rather than to look like a core:
    //
    //   zeros            the `P_node <= 0` skip in the fuel loop;
    //   very low powers  GetTfuel's sub-50 W/cm linear continuation
    //                    (lpd = 1000 * P_node / (62 * hz), so P_node of order
    //                    10 kW puts lpd well under the first tabulated knot);
    //   one hot channel  the water-property table ceiling, so the clamp path and
    //                    the (count, worst, node) overflow triple are scored.
    for (std::size_t i = 0; i < n; ++i) {
        const double r = rng.next();
        if (r < 0.08)
            f.node_power[i] = 0.0;
        else if (r < 0.20)
            f.node_power[i] = 1.0e3 + 2.0e4 * rng.next();
        else
            f.node_power[i] = 1.0e5 + 5.0e5 * rng.next();
    }
    for (int k = kbc; k < kec; ++k)
        f.node_power[static_cast<std::size_t>(k) * nxy] = 3.0e7;

    return f;
}

} // namespace thref
