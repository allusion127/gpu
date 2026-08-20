// Mine the contraction forms production gcc emitted for BICGCMFD::wiel and
// BICGCMFD::updls, from a RASBERY_CMFD_DUMP capture.  Same method and same
// justification as xsrecon_form_probe: gcc -ffp-contract=fast fuses the same
// statements differently per translation unit, so the device sweep kernels
// must copy the MEASURED forms, not the source text.
//
// Usage: rasbery_cmfd_form_probe <prefix>   (reads <prefix>.cmfd)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

inline double opaque(double x) {
    asm volatile("" : "+x"(x));
    return x;
}

/// a*b + c, fused (f=1) or with the product rounded separately (f=0).
inline double mac(int f, double a, double b, double c) {
    if (f) return std::fma(a, b, c);
    return opaque(a * b) + c;
}

std::vector<double> rd(std::FILE* f, size_t n) {
    std::vector<double> v(n);
    if (std::fread(v.data(), sizeof(double), n, f) != n) {
        std::fprintf(stderr, "short read (%zu)\n", n);
        std::exit(2);
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    std::FILE* f = std::fopen((std::string(argv[1]) + ".cmfd").c_str(), "rb");
    if (!f) {
        std::printf("no capture\n");
        return 2;
    }

    std::vector<double> hdr = rd(f, 8);
    const int    ng    = static_cast<int>(hdr[0]);
    const int    nxyz  = static_cast<int>(hdr[1]);
    const int    ng2   = static_cast<int>(hdr[2]);
    const int    nsw   = static_cast<int>(hdr[3]);
    const double eshift = hdr[6];
    if (ng != 2) {
        std::printf("ng != 2\n");
        return 2;
    }
    const size_t nx = static_cast<size_t>(nxyz);

    std::vector<double> psi0  = rd(f, nx);
    std::vector<double> udiag = rd(f, static_cast<size_t>(ng2) * nx);
    std::vector<double> chif  = rd(f, 2 * nx);
    std::vector<double> xsnf  = rd(f, 2 * nx);
    std::vector<double> vol   = rd(f, nx);

    struct Sweep {
        double              reigv_in, reigvs_in, eigv_in, icy;
        std::vector<double> flux, psi_in;
        double              eigv_out, reigv_out, reigvs_out, errl2_out, negative;
        std::vector<double> psi_out, diag_out;
    };
    std::vector<Sweep> sweeps;
    for (int s = 0; s < nsw; ++s) {
        Sweep sw;
        double pre[4];
        if (std::fread(pre, sizeof(double), 4, f) != 4) break; // early break in drive
        sw.reigv_in  = pre[0];
        sw.reigvs_in = pre[1];
        sw.eigv_in   = pre[2];
        sw.icy       = pre[3];
        sw.flux      = rd(f, 2 * nx);
        sw.psi_in    = rd(f, nx);
        double post[5];
        if (std::fread(post, sizeof(double), 5, f) != 5) break;
        sw.eigv_out   = post[0];
        sw.reigv_out  = post[1];
        sw.reigvs_out = post[2];
        sw.errl2_out  = post[3];
        sw.negative   = post[4];
        sw.psi_out    = rd(f, nx);
        sw.diag_out   = rd(f, static_cast<size_t>(ng2) * nx);
        sweeps.push_back(std::move(sw));
    }
    std::fclose(f);
    std::printf("sweeps captured: %zu (eshift=%g)\n", sweeps.size(), eshift);

    // ---- wiel: psi pair-sum form (p), and the three accumulations (e,d,n).
    // p0: fma(xsnf1,f1, xsnf0*f0)  p1: rounded products, plain add
    // p2: fma(xsnf0,f0, xsnf1*f1)  -- gcc may evaluate/fuse either operand
    static const char* kPsiName[3] = {"fma2nd", "plain", "fma1st"};
    for (int p = 0; p < 3; ++p) {
        size_t ok_psi = 0, tot_psi = 0, first_bad = static_cast<size_t>(-1),
               last_bad = 0, tail_bad = 0;
        for (const Sweep& sw : sweeps)
            for (size_t l = 0; l < nx; ++l) {
                double pv;
                if (p == 0)
                    pv = std::fma(xsnf[nx + l], sw.flux[l * 2 + 1],
                                  opaque(xsnf[l] * sw.flux[l * 2 + 0]));
                else if (p == 1)
                    pv = opaque(xsnf[l] * sw.flux[l * 2 + 0]) +
                         opaque(xsnf[nx + l] * sw.flux[l * 2 + 1]);
                else
                    pv = std::fma(xsnf[l], sw.flux[l * 2 + 0],
                                  opaque(xsnf[nx + l] * sw.flux[l * 2 + 1]));
                pv = pv * vol[l];
                ++tot_psi;
                if (!std::memcmp(&pv, &sw.psi_out[l], 8)) {
                    ++ok_psi;
                } else {
                    if (first_bad == static_cast<size_t>(-1)) first_bad = l;
                    last_bad = l;
                    if (l >= nx - 8) ++tail_bad;
                }
            }
        std::printf("psi form %s : %zu/%zu  first_bad=%zd last_bad=%zu tail8=%zu\n",
                    kPsiName[p], ok_psi, tot_psi, static_cast<long>(first_bad),
                    last_bad, tail_bad);
    }

    // Accumulations + eigenvalue update.  e/d/n: fused or plain; u: the
    // eigv denominator reigv*gamma + (1-gamma)*reigvs, second product fused
    // into the add (u=1) or rounded then added (u=0).
    for (int pf = 2; pf < 3; ++pf) // psi form settled: fma1st
        for (int e = 0; e < 1; ++e)     // errl2 settled: plain
            for (int d = 0; d < 2; ++d)
                for (int n = 0; n < 2; ++n)
                    for (int u = 0; u < 3; ++u) {
                        size_t ok_e = 0, ok_g = 0, tot = 0;
                        for (const Sweep& sw : sweeps) {
                            double err = 0, gammad = 0, gamman = 0;
                            for (size_t l = 0; l < nx; ++l) {
                                double psid = sw.psi_in[l];
                                double pv   = std::fma(xsnf[l], sw.flux[l * 2 + 0],
                                                       opaque(xsnf[nx + l] * sw.flux[l * 2 + 1]));
                                pv          = pv * vol[l];
                                double err1 = pv - psid;
                                err         = mac(e, err1, err1, err);
                                gammad      = mac(d, psid, pv, gammad);
                                gamman      = mac(n, pv, pv, gamman);
                            }
                            ++tot;
                            const bool usable = (gammad > 0.0) && (gamman > 0.0);
                            if (sw.icy < 0 || !usable) continue; // fallback branch: skip
                            double gamma = gammad / gamman;
                            double den;
                            if (u == 0)
                                den = opaque(sw.reigv_in * gamma) +
                                      opaque((1 - gamma) * sw.reigvs_in);
                            else if (u == 1)
                                den = std::fma((1 - gamma), sw.reigvs_in,
                                               opaque(sw.reigv_in * gamma));
                            else
                                den = std::fma(sw.reigv_in, gamma,
                                               opaque((1 - gamma) * sw.reigvs_in));
                            double eigv      = 1 / den;
                            double err_scale = (gammad > 0.0) ? gammad : gamman;
                            double errl2 =
                                (err_scale > 0.0) ? sqrt(std::abs(err / err_scale)) : 0.0;
                            if (!std::memcmp(&eigv, &sw.eigv_out, 8)) {
                                ++ok_e;
                            } else if (d == 0 && n == 0 && u == 2) {
                                std::int64_t ba, bb;
                                std::memcpy(&ba, &eigv, 8);
                                std::memcpy(&bb, &sw.eigv_out, 8);
                                std::printf("  eigv miss: got=%.17g prod=%.17g ulp=%lld "
                                            "(gamma=%.17g gd=%.17g gn=%.17g)\n",
                                            eigv, sw.eigv_out,
                                            static_cast<long long>(bb - ba), gamma,
                                            gammad, gamman);
                            }
                            if (!std::memcmp(&errl2, &sw.errl2_out, 8)) ++ok_g;
                        }
                        if (ok_e || ok_g)
                            std::printf("acc pf=%d e=%d d=%d n=%d u=%d : eigv %zu/%zu errl2 %zu/%zu\n",
                                        pf, e, d, n, u, ok_e, tot, ok_g, tot);
                    }

    // ---- updls: diag = udiag - chif*xsnf*reigvs*vol.
    // c=0: product chain rounded, plain subtract.  c=1: last multiply fused
    // into the subtract as fma(-c2, vol, udiag).
    if (eshift != 0.0) {
        for (int c = 0; c < 2; ++c) {
            size_t ok = 0, tot = 0;
            for (const Sweep& sw : sweeps)
                for (size_t l = 0; l < nx; ++l)
                    for (int ige = 0; ige < 2; ++ige)
                        for (int igs = 0; igs < 2; ++igs) {
                            double c2 = chif[static_cast<size_t>(ige) * nx + l] *
                                        xsnf[static_cast<size_t>(igs) * nx + l] *
                                        sw.reigvs_out;
                            double dv =
                                c ? std::fma(-c2, vol[l],
                                             udiag[l * 4 + static_cast<size_t>(ige) * 2 + igs])
                                  : udiag[l * 4 + static_cast<size_t>(ige) * 2 + igs] -
                                        opaque(c2 * vol[l]);
                            ++tot;
                            if (!std::memcmp(&dv,
                                             &sw.diag_out[l * 4 + static_cast<size_t>(ige) * 2 + igs],
                                             8))
                                ++ok;
                        }
            std::printf("updls form %s : %zu/%zu\n", c ? "fused-sub" : "plain", ok, tot);
        }
    }
    return 0;
}
