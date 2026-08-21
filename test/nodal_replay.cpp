// Replay gate + contraction-form miner for the nodal device arm.
//
//   ./nodal_replay <capture>            end-to-end replay with production
//                                       forms; per-phase ULP score
//   ./nodal_replay <capture> --sweep    mine each phase's mask (brute force
//                                       up to 12 bits, coordinate descent
//                                       above) against captured phase inputs
//
// <capture> comes from RASBERY_NODAL_DUMP (Nodal.cpp nodalDumpState).

#include "../src/NodalKernel.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace nk = rasbery::nodal;

namespace {

struct Cap {
    std::int64_t nxyz = 0, nsurf = 0, chif_empty = 0;
    double       reigv = 0;
    std::vector<int>    lktosfc, neib, lklr, idirlr, sgnlr;
    std::vector<double> hmesh, albedo;
    std::vector<double> xsrf, xsnf, xssm, chif;
    std::vector<double> C[9]; // eta1,eta2,m260,m251,m253,m262,m264,diagD,diagDI
    std::vector<double> jnet_in, flux;
    // captured post-drive state
    std::vector<double> trl0, trl1, trl2, matMs, matMf, matM, matMI, mu, tau,
        c2, c4, c6, jnet_out, phis_out;
};

template <class T>
bool rd(std::FILE* f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    return std::fread(v.data(), sizeof(T), n, f) == n;
}

bool load(const char* path, Cap& c) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::int64_t hdr[8];
    if (std::fread(hdr, 8, 8, f) != 8) return false;
    c.nxyz = hdr[0]; c.nsurf = hdr[1]; c.chif_empty = hdr[5];
    if (hdr[2] != nk::NDIR || hdr[3] != nk::NG || hdr[4] != nk::NEWSB) {
        std::fprintf(stderr, "dimension mismatch\n");
        return false;
    }
    if (std::fread(&c.reigv, 8, 1, f) != 1) return false;
    const std::size_t nx = c.nxyz, ns = c.nsurf;
    bool ok = rd(f, c.lktosfc, nx * nk::NDIR * nk::NLR) && rd(f, c.neib, nx * nk::NEWSB) &&
              rd(f, c.lklr, ns * nk::NLR) && rd(f, c.idirlr, ns * nk::NLR) &&
              rd(f, c.sgnlr, ns * nk::NLR) && rd(f, c.hmesh, nx * nk::NDIR) &&
              rd(f, c.albedo, nk::NDIR * nk::NLR) && rd(f, c.xsrf, nk::NG * nx) &&
              rd(f, c.xsnf, nk::NG * nx) && rd(f, c.xssm, nk::NG2 * nx);
    if (ok && !c.chif_empty) ok = rd(f, c.chif, nk::NG * nx);
    for (int i = 0; ok && i < 9; ++i) ok = rd(f, c.C[i], nx * nk::NDIR * nk::NG);
    ok = ok && rd(f, c.jnet_in, ns * nk::NG) && rd(f, c.flux, nx * nk::NG);
    ok = ok && rd(f, c.trl0, nx * nk::NDIR * nk::NG) && rd(f, c.trl1, nx * nk::NDIR * nk::NG) &&
         rd(f, c.trl2, nx * nk::NDIR * nk::NG) && rd(f, c.matMs, nx * nk::NG2) &&
         rd(f, c.matMf, nx * nk::NG2) && rd(f, c.matM, nx * nk::NG2) &&
         rd(f, c.matMI, nx * nk::NG2) && rd(f, c.mu, nx * nk::NDIR * nk::NG2) &&
         rd(f, c.tau, nx * nk::NDIR * nk::NG2) && rd(f, c.c2, nx * nk::NDIR * nk::NG) &&
         rd(f, c.c4, nx * nk::NDIR * nk::NG) && rd(f, c.c6, nx * nk::NDIR * nk::NG) &&
         rd(f, c.jnet_out, ns * nk::NG) && rd(f, c.phis_out, ns * nk::NG);
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "truncated capture\n");
    return ok;
}

struct Work {
    std::vector<double> trl0, trl1, trl2, matMs, matMf, matM, matMI, mu, tau,
        c2, c4, c6, jnet, phis;
};

nk::NodalView makeView(const Cap& c, Work& w) {
    nk::NodalView v{};
    v.hmesh   = c.hmesh.data();
    v.lktosfc = c.lktosfc.data();
    v.neib    = c.neib.data();
    v.lklr    = c.lklr.data();
    v.idirlr  = c.idirlr.data();
    v.sgnlr   = c.sgnlr.data();
    v.albedo  = c.albedo.data();
    v.xsrf    = c.xsrf.data();
    v.xsnf    = c.xsnf.data();
    v.xssm    = c.xssm.data();
    v.chif    = c.chif_empty ? nullptr : c.chif.data();
    v.chif_empty = static_cast<int>(c.chif_empty);
    const std::vector<double>* C = c.C;
    v.eta1 = C[0].data(); v.eta2 = C[1].data(); v.m260 = C[2].data();
    v.m251 = C[3].data(); v.m253 = C[4].data(); v.m262 = C[5].data();
    v.m264 = C[6].data(); v.diagD = C[7].data(); v.diagDI = C[8].data();
    v.trlcff0 = w.trl0.data(); v.trlcff1 = w.trl1.data(); v.trlcff2 = w.trl2.data();
    v.mu = w.mu.data(); v.tau = w.tau.data();
    v.matM = w.matM.data(); v.matMI = w.matMI.data();
    v.matMs = w.matMs.data(); v.matMf = w.matMf.data();
    v.dsncff2 = w.c2.data(); v.dsncff4 = w.c4.data(); v.dsncff6 = w.c6.data();
    v.flux = c.flux.data();
    v.jnet = w.jnet.data();
    v.phis = w.phis.data();
    v.reigv = c.reigv;
    v.nxyz = static_cast<int>(c.nxyz);
    v.nsurf = static_cast<int>(c.nsurf);
    return v;
}

void allocWork(const Cap& c, Work& w) {
    const std::size_t nx = c.nxyz, ns = c.nsurf;
    w.trl0.assign(nx * nk::NDIR * nk::NG, 0); w.trl1 = w.trl0; w.trl2 = w.trl0;
    w.matMs.assign(nx * nk::NG2, 0); w.matMf = w.matMs; w.matM = w.matMs; w.matMI = w.matMs;
    w.mu.assign(nx * nk::NDIR * nk::NG2, 0); w.tau = w.mu;
    w.c2 = w.trl0; w.c4 = w.trl0; w.c6 = w.trl0;
    w.jnet.assign(ns * nk::NG, 0);
    w.phis.assign(ns * nk::NG, 0);
}

std::uint64_t ulp(double a, double b) {
    if (a == b) return 0;
    std::int64_t ia, ib;
    std::memcpy(&ia, &a, 8); std::memcpy(&ib, &b, 8);
    if (ia < 0) ia = INT64_MIN - ia;
    if (ib < 0) ib = INT64_MIN - ib;
    const std::int64_t d = ia - ib;
    return static_cast<std::uint64_t>(d < 0 ? -d : d);
}

std::uint64_t score(const std::vector<double>& got, const std::vector<double>& want,
                    std::uint64_t* worst = nullptr, std::size_t* first = nullptr) {
    std::uint64_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const std::uint64_t u = ulp(got[i], want[i]);
        if (u) {
            if (first && bad == 0) *first = i;
            ++bad;
            if (worst && u > *worst) *worst = u;
        }
    }
    return bad;
}

// Phase runners.  seed==true copies the captured UPSTREAM intermediates into
// the work arrays first, isolating the phase for mining.
template <class POL>
void runPhase(int phase, const Cap& c, Work& w, const POL& pol, bool seed) {
    nk::NodalView v = makeView(c, w);
    const int     nx = v.nxyz, ns = v.nsurf;
    switch (phase) {
        case 1: // trlcff0 (no mask)
            std::copy(c.jnet_in.begin(), c.jnet_in.end(), w.jnet.begin());
            for (int lk = 0; lk < nx; ++lk) nk::nodalTrlcff0(v, lk);
            break;
        case 2: // trlcff12
            if (seed) w.trl0 = c.trl0;
            for (int lk = 0; lk < nx; ++lk) nk::nodalTrlcff12(v, lk, pol);
            break;
        case 3: // updateMatrix
            for (int lk = 0; lk < nx; ++lk) nk::nodalUpdateMatrix(v, lk, pol);
            break;
        case 4: // calculateEven
            if (seed) {
                w.trl0 = c.trl0;
                w.trl2 = c.trl2;
                w.matM = c.matM;
            }
            for (int lk = 0; lk < nx; ++lk) nk::nodalCalculateEven(v, lk, pol);
            break;
        case 5: // calculateJnet
            if (seed) {
                std::copy(c.jnet_in.begin(), c.jnet_in.end(), w.jnet.begin());
                w.trl1 = c.trl1; w.mu = c.mu; w.tau = c.tau;
                w.matM = c.matM; w.matMI = c.matMI;
                w.c2 = c.c2; w.c4 = c.c4; w.c6 = c.c6;
            }
            for (int ls = 0; ls < ns; ++ls) nk::nodalCalculateJnet(v, ls, pol);
            break;
    }
}

} // namespace

namespace {

struct Site { int bit; int width; }; // width 1: states {0,1}; width 2: {0,1,2}

const std::vector<Site> SITES_TRL12 = {{0,1},{1,1},{2,1},{3,1},{4,1},{5,1}};
const std::vector<Site> SITES_EVEN = {{0,2},{2,1},{3,2},{5,2},{7,1},{8,2},
                                      {10,2},{12,2},{14,2},{16,1},{17,1}};
const std::vector<Site> SITES_1N = {{0,1},{1,2},{3,1},{4,2},{6,2},{8,2},{10,2},
                                    {12,2},{14,2},{16,2},{18,1},{19,1},{20,2},
                                    {22,1},{23,1},{24,1},{25,1},{26,1}};
const std::vector<Site> SITES_2N = {{0,2},{2,2},{4,1},{5,1},{6,2},{8,1},{9,1},
                                    {10,1},{11,2},{13,1},{14,2},{16,2},{18,2},
                                    {20,1},{21,1},{22,2},{24,2},{26,2},{28,2},
                                    {30,2},{32,1},{33,1},{34,2},{36,1},{37,1},
                                    {38,1},{39,1}};

unsigned long long setSite(unsigned long long m, const Site& st, unsigned val) {
    const unsigned long long width_mask = st.width == 2 ? 3ull : 1ull;
    m &= ~(width_mask << st.bit);
    m |= (static_cast<unsigned long long>(val) & width_mask) << st.bit;
    return m;
}

unsigned long long uniformMask(const std::vector<Site>& sites, unsigned val) {
    unsigned long long m = 0;
    for (const auto& st : sites)
        m = setSite(m, st, st.width == 1 ? (val ? 1u : 0u) : val);
    return m;
}

/// Coordinate descent over sites; returns final mask, sets *out_bad.
unsigned long long descend(const std::vector<Site>& sites,
                           unsigned long long seed,
                           const std::function<std::uint64_t(unsigned long long)>& trial,
                           std::uint64_t* out_bad) {
    unsigned long long m   = seed;
    std::uint64_t      cur = trial(m);
    bool improved = true;
    while (cur != 0 && improved) {
        improved = false;
        for (const auto& st : sites) {
            const unsigned nstates = st.width == 2 ? 3u : 2u;
            const unsigned cur_val =
                static_cast<unsigned>((m >> st.bit) & (st.width == 2 ? 3u : 1u));
            for (unsigned v2 = 0; v2 < nstates; ++v2) {
                if (v2 == cur_val) continue;
                const unsigned long long cand = setSite(m, st, v2);
                const std::uint64_t      bad  = trial(cand);
                if (bad < cur) { m = cand; cur = bad; improved = true; break; }
            }
            if (cur == 0) break;
        }
    }
    *out_bad = cur;
    return m;
}

/// Try full enumeration when the state product is small, else multi-seed descent.
unsigned long long mineMask(const char* name, const std::vector<Site>& sites,
                            const std::function<std::uint64_t(unsigned long long)>& trial) {
    double combos = 1;
    for (const auto& st : sites) combos *= (st.width == 2 ? 3 : 2);
    if (combos <= 250000) {
        std::vector<unsigned> idx(sites.size(), 0);
        unsigned long long best = 0; std::uint64_t bestbad = ~0ull; int nwin = 0;
        unsigned long long firstwin = 0;
        while (true) {
            unsigned long long m = 0;
            for (std::size_t i = 0; i < sites.size(); ++i) m = setSite(m, sites[i], idx[i]);
            const std::uint64_t bad = trial(m);
            if (bad == 0) { if (nwin == 0) firstwin = m; ++nwin; }
            if (bad < bestbad) { bestbad = bad; best = m; }
            std::size_t k = 0;
            for (; k < sites.size(); ++k) {
                const unsigned lim = sites[k].width == 2 ? 3u : 2u;
                if (++idx[k] < lim) break;
                idx[k] = 0;
            }
            if (k == sites.size()) break;
        }
        if (nwin > 0)
            std::printf("phase %-12s: %d exact masks (brute), first=0x%llx\n", name,
                        nwin, static_cast<unsigned long long>(firstwin));
        else
            std::printf("phase %-12s: NO exact (brute), best=0x%llx bad=%" PRIu64 "\n",
                        name, static_cast<unsigned long long>(best), bestbad);
        return nwin > 0 ? firstwin : best;
    }
    unsigned long long bestm = 0; std::uint64_t bestbad = ~0ull;
    for (unsigned seedv = 0; seedv < 3; ++seedv) {
        std::uint64_t      bad = 0;
        const unsigned long long m =
            descend(sites, uniformMask(sites, seedv), trial, &bad);
        std::printf("  %s seed=%u -> mask=0x%llx bad=%" PRIu64 "\n", name, seedv,
                    static_cast<unsigned long long>(m), bad);
        if (bad < bestbad) { bestbad = bad; bestm = m; }
        if (bad == 0) break;
    }
    std::printf("phase %-12s: %s mask=0x%llx bad=%" PRIu64 "\n", name,
                bestbad == 0 ? "EXACT (descent)" : "STUCK",
                static_cast<unsigned long long>(bestm), bestbad);
    return bestm;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <capture> [--sweep]\n", argv[0]); return 2; }
    Cap c;
    if (!load(argv[1], c)) return 2;
    std::printf("capture: nxyz=%" PRId64 " nsurf=%" PRId64 " chif_empty=%" PRId64 "\n",
                c.nxyz, c.nsurf, c.chif_empty);

    const bool sweep = argc > 2 && std::string(argv[2]) == "--sweep";
    Work w;
    allocWork(c, w);

    // Surface type split: 1n (boundary) vs 2n (interior) score independently.
    std::vector<char> is1n(c.nsurf, 0);
    for (std::int64_t ls = 0; ls < c.nsurf; ++ls)
        is1n[ls] = (c.lklr[ls * nk::NLR] < 0 || c.lklr[ls * nk::NLR + 1] < 0) ? 1 : 0;

    auto scoreArr = [&](const std::vector<double>& got, const std::vector<double>& want,
                        int only_type /*-1 all, 0 2n, 1 1n*/) {
        std::uint64_t bad = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (only_type >= 0) {
                const std::size_t ls = i / nk::NG;
                if (static_cast<int>(is1n[ls]) != only_type) continue;
            }
            if (ulp(got[i], want[i])) ++bad;
        }
        return bad;
    };

    nk::RuntimeForms pol;

    auto runEven = [&](unsigned long long m) {
        pol.mask[2] = m;
        runPhase(4, c, w, pol, true);
        return scoreArr(w.c2, c.c2, -1) + scoreArr(w.c4, c.c4, -1) +
               scoreArr(w.c6, c.c6, -1);
    };
    auto runJnet = [&](int type, unsigned long long m3, unsigned long long m4) {
        pol.mask[3] = m3; pol.mask[4] = m4;
        runPhase(5, c, w, pol, true);
        return scoreArr(w.jnet, c.jnet_out, type) + scoreArr(w.phis, c.phis_out, type);
    };

    if (!sweep) {
        nk::StaticForms sp;
        std::uint64_t   grand = 0;
        runPhase(1, c, w, sp, true);
        grand += scoreArr(w.trl0, c.trl0, -1) * 0; // trl0 scored below with others
        std::uint64_t bad;
        runPhase(1, c, w, sp, true);
        bad = 0;
        for (std::size_t i = 0; i < w.trl0.size(); ++i) if (ulp(w.trl0[i], c.trl0[i])) ++bad;
        std::printf("phase trlcff0     : %s (%" PRIu64 ")\n", bad ? "FAIL" : "PASS", bad); grand += bad;
        runPhase(2, c, w, sp, true);
        bad = 0;
        for (std::size_t i = 0; i < w.trl1.size(); ++i) if (ulp(w.trl1[i], c.trl1[i]) + ulp(w.trl2[i], c.trl2[i])) ++bad;
        std::printf("phase trlcff12    : %s (%" PRIu64 ")\n", bad ? "FAIL" : "PASS", bad); grand += bad;
        runPhase(3, c, w, sp, true);
        bad = 0;
        for (std::size_t i = 0; i < w.matM.size(); ++i)
            if (ulp(w.matM[i], c.matM[i]) + ulp(w.matMI[i], c.matMI[i]) +
                ulp(w.matMs[i], c.matMs[i]) + ulp(w.matMf[i], c.matMf[i])) ++bad;
        for (std::size_t i = 0; i < w.mu.size(); ++i)
            if (ulp(w.mu[i], c.mu[i]) + ulp(w.tau[i], c.tau[i])) ++bad;
        std::printf("phase updateMatrix: %s (%" PRIu64 ")\n", bad ? "FAIL" : "PASS", bad); grand += bad;
        runPhase(4, c, w, sp, true);
        bad = 0;
        for (std::size_t i = 0; i < w.c2.size(); ++i)
            if (ulp(w.c2[i], c.c2[i]) + ulp(w.c4[i], c.c4[i]) + ulp(w.c6[i], c.c6[i])) ++bad;
        std::printf("phase even        : %s (%" PRIu64 ")\n", bad ? "FAIL" : "PASS", bad); grand += bad;
        runPhase(5, c, w, sp, true);
        bad = 0;
        for (std::size_t i = 0; i < w.jnet.size(); ++i)
            if (ulp(w.jnet[i], c.jnet_out[i]) + ulp(w.phis[i], c.phis_out[i])) ++bad;
        for (std::size_t i = 0; i < w.jnet.size(); ++i)
            if (ulp(w.jnet[i], c.jnet_out[i]) + ulp(w.phis[i], c.phis_out[i])) {
                std::printf("  first: elem=%zu ls=%zu type=%d jnet %.17g vs %.17g phis %.17g vs %.17g\n",
                            i, i / nk::NG, (int)is1n[i / nk::NG], w.jnet[i],
                            c.jnet_out[i], w.phis[i], c.phis_out[i]);
                break;
            }
        std::printf("phase jnet        : %s (%" PRIu64 ")\n", bad ? "FAIL" : "PASS", bad); grand += bad;
        // Static-vs-runtime instantiation check: same masks, same seeded
        // inputs -- any element difference is a codegen difference between
        // the two template instantiations, not a mask problem.
        {
            std::vector<double> jnet_static = w.jnet, phis_static = w.phis;
            nk::RuntimeForms rp;
            for (int q = 0; q < 5; ++q) rp.mask[q] = nk::NODAL_FORMS[q];
            runPhase(5, c, w, rp, true);
            std::uint64_t d1 = 0, d2 = 0;
            for (std::size_t k = 0; k < w.jnet.size(); ++k) {
                if (ulp(jnet_static[k], w.jnet[k])) ++d1;
                if (ulp(phis_static[k], w.phis[k])) ++d2;
            }
            std::printf("static-vs-runtime: jnet=%llu phis=%llu\n",
                        (unsigned long long)d1, (unsigned long long)d2);
        }
        std::printf("[nodal_replay] production masks: %s\n", grand == 0 ? "PASS" : "FAIL");
        return grand == 0 ? 0 : 1;
    }

    pol.mask[1] = 0xFull; // updateMatrix mined exact on the first pass
    const unsigned long long m_trl = mineMask(
        "trlcff12", SITES_TRL12,
        std::function<std::uint64_t(unsigned long long)>([&](unsigned long long m) {
            pol.mask[0] = m;
            runPhase(2, c, w, pol, true);
            return scoreArr(w.trl1, c.trl1, -1) + scoreArr(w.trl2, c.trl2, -1);
        }));
    pol.mask[0] = m_trl;
    runPhase(3, c, w, pol, true);

    const unsigned long long m_even =
        mineMask("even", SITES_EVEN, std::function<std::uint64_t(unsigned long long)>(runEven));
    pol.mask[2] = m_even;

    const unsigned long long m3 = mineMask(
        "jnet1n", SITES_1N,
        std::function<std::uint64_t(unsigned long long)>(
            [&](unsigned long long m) { return runJnet(1, m, pol.mask[4]); }));
    pol.mask[3] = m3;
    const unsigned long long m4 = mineMask(
        "jnet2n", SITES_2N,
        std::function<std::uint64_t(unsigned long long)>(
            [&](unsigned long long m) { return runJnet(0, pol.mask[3], m); }));
    pol.mask[4] = m4;

    // Combined full-surface check with the mined masks, split to localize.
    runPhase(5, c, w, pol, true);
    std::printf("combined: jnet1n=%llu phis1n=%llu jnet2n=%llu phis2n=%llu\n",
                (unsigned long long)scoreArr(w.jnet, c.jnet_out, 1),
                (unsigned long long)scoreArr(w.phis, c.phis_out, 1),
                (unsigned long long)scoreArr(w.jnet, c.jnet_out, 0),
                (unsigned long long)scoreArr(w.phis, c.phis_out, 0));
    for (std::size_t i2 = 0; i2 < w.jnet.size(); ++i2)
        if (ulp(w.jnet[i2], c.jnet_out[i2])) {
            std::printf("first jnet mm: elem=%zu ls=%zu type=%d got=%.17g want=%.17g\n",
                        i2, i2 / nk::NG, (int)is1n[i2 / nk::NG], w.jnet[i2],
                        c.jnet_out[i2]);
            break;
        }
    std::printf("[nodal_replay] mined masks: {0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx}\n",
                (unsigned long long)pol.mask[0], (unsigned long long)pol.mask[1],
                (unsigned long long)pol.mask[2], (unsigned long long)pol.mask[3],
                (unsigned long long)pol.mask[4]);
    return 0;
}
