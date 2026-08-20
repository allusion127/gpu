#pragma once

#include "Importer.h"
#include "Model.h"
#include <algorithm>
#include <cctype>
#include "CompatFormat.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace Chiffon {

class Benchmark {
private:
    /// Reconstruct one XS value: lmpx + sum(micx * iden)
    static double Recon(const CrossSection& xs, const milk::Vector<double>& iden, int ig, XSTYPE xt) {
        double val = xs.lmpxs(ig, xt);
        for (size_t iso = 0; iso < niso; ++iso)
            val += xs.mixs(static_cast<int>(iso), ig, xt) * iden[iso];
        return val;
    }

public:
    // Offline statistics export (friend of Importer): load each HGC with the production
    // loader, reconstruct its rod-material fluence, and write ONE CSV row per depletion
    // point with the perturbation tag (file stem), state scalars, rod coordinates, and all
    // node-average isotope densities. Used by the isotope correlation / VIF study; keeps the
    // main classes free of any export code. ZZAAAM isotope IDs are the density column names.
    static void DumpIsotopeStates(const std::vector<std::string>& hgcPaths,
                                  const std::string&              outCsv) {
        std::ofstream os(outCsv);
        if (!os) {
            std::cerr << "[isodump] cannot open " << outCsv << std::endl;
            return;
        }
        os << "tag,ctype,btyp,traj_ctyp,traj_ref,burnup,pden,tmod,tful,dmod,bppm,keff,kinf,"
           << "tot_flux,rod_fluence,fuel_rod_fluence,macro_abs_rate";
        for (size_t iso = 0; iso < niso; ++iso)  // node-average isotope densities
            os << "," << Isotope::isotopeIds[iso];
        for (size_t iso = 0; iso < niso; ++iso)  // 1-group flux-weighted absorption CONTRIBUTION n*sigma_a
            os << ",ac_" << Isotope::isotopeIds[iso];
        os << "\n";
        os << std::scientific;

        for (const std::string& path : hgcPaths) {
            std::string stem = path;
            if (auto s = stem.find_last_of("/\\"); s != std::string::npos) stem = stem.substr(s + 1);
            if (auto d = stem.rfind("_0101.HGC"); d != std::string::npos) stem = stem.substr(0, d);

            Importer imp;
            Model    m("isodump");
            try {
                imp.ReadHGC(m, path);
            } catch (const std::exception& e) {
                std::cerr << "[isodump][skip] " << stem << " : " << e.what() << std::endl;
                continue;
            }
            const std::map<int, double> fluence = Importer::ReconstructFluenceFromHGC(m);

            for (const DepletionPoint& dpt : m.Dpts()) {
                if (!dpt._xs.has_micx())
                    continue;
                double totFlux = 0.0;
                for (double f : dpt._aflx) totFlux += f;
                // node-average macroscopic absorption reaction rate sum_g (Sigma_a,g * phi_g),
                // reconstructed from micro XS x densities (an explicit rod-history exposure axis).
                double macroAbs = 0.0;
                for (size_t ig = 0; ig < dpt._ngrp; ++ig) {
                    const double phi = ig < dpt._aflx.size() ? dpt._aflx[ig] : 0.0;
                    macroAbs += Recon(dpt._xs, dpt._iden, static_cast<int>(ig), XSAF) * phi;
                }
                const double rodFlu = Importer::InterpolateRodFluence(dpt, fluence);
                os << stem << "," << dpt._ctyp << "," << static_cast<int>(dpt._btyp) << ","
                   << dpt._trajectory_ctyp << "," << (dpt._trajectory_reference ? 1 : 0) << ","
                   << dpt._data[AD_BURN] << "," << dpt._data[AD_PDEN] << "," << dpt._data[AD_TMOD] << ","
                   << dpt._data[AD_TFUL] << "," << dpt._data[AD_DMOD] << "," << dpt._data[AD_BPPM] << ","
                   << dpt._data[AD_KEFF] << "," << dpt._data[AD_KINF] << "," << totFlux << ","
                   << rodFlu << "," << dpt._fuel_rod_fluence << "," << macroAbs;
                for (size_t iso = 0; iso < niso; ++iso)
                    os << "," << dpt._iden[iso];
                // per-isotope 1-group flux-weighted absorption contribution c_i = n_i * <sigma_a,i>_phi,
                // i.e. each isotope's share of the macroscopic absorption (reactivity-weighted importance).
                for (size_t iso = 0; iso < niso; ++iso) {
                    double num = 0.0, den = 0.0;
                    for (size_t ig = 0; ig < dpt._ngrp; ++ig) {
                        const double phi = ig < dpt._aflx.size() ? dpt._aflx[ig] : 0.0;
                        num += dpt._xs.mixs(static_cast<int>(iso), static_cast<int>(ig), XSAF) * phi;
                        den += phi;
                    }
                    os << "," << (den > 0.0 ? dpt._iden[iso] * num / den : 0.0);
                }
                os << "\n";
            }
            os.flush();
            std::cout << "[isodump] " << stem << "  (" << m.Dpts().size() << " pts)" << std::endl;
        }
        std::cout << "[isodump] wrote " << outCsv << std::endl;
    }

    /// Tidy CSV dump for interpolation-validation plots at burnup 0, control type 0.
    /// For each branch axis (BPPM / TFUL / DMOD) three series are written:
    ///   source=fine   : dense reference branch points  (the "truth")
    ///   source=coarse : sparse branch points used for the fit (the fit nodes)
    ///   source=interp : coarse.GetCrossSection() swept across the axis (CHIFFON interpolation)
    /// Columns: branch,source,group,xval,xstype,value  (value = macroscopic XS).
    /// NOTE: TMOD is intentionally absent - the test assemblies carry no TMOD branch and
    /// GetCrossSection() does not interpolate a moderator-temperature axis (only BPPM/TFUL/DMOD).
    static void ValidateCSV(const Model& fine, const Model& coarse, const std::string& outPath) {
        std::ofstream os(outPath);
        if (!os) {
            std::cerr << "[validate] cannot open " << outPath << "\n";
            return;
        }
        os << "branch,source,group,xval,xstype,value\n";
        os << std::scientific;

        const size_t NGRP  = coarse.GetDepletionPoint(0, 0)._ngrp;
        const auto&  refC  = coarse.GetDepletionPoint(0, 0);
        const double bppm0 = refC._data[AD_BPPM];
        const double tful0 = refC._data[AD_TFUL];
        const double dmod0 = refC._data[AD_DMOD];

        const XSTYPE      xss[] = {XSAF, XSNF, XSSF};
        const char* const xsn[] = {"XSAF", "XSNF", "XSSF"};

        struct Axis {
            BRANCHTYPE        btyp;
            AssemblyDataIndex ad;
            const char*       name;
        };
        const Axis axes[] = {{BPPM, AD_BPPM, "BPPM"}, {TFUL, AD_TFUL, "TFUL"}, {DMOD, AD_DMOD, "DMOD"}};

        auto emit = [&](const char* branch, const char* src, double xval, const CrossSection& xs) {
            for (size_t ig = 0; ig < NGRP; ++ig)
                for (int k = 0; k < 3; ++k)
                    os << branch << ',' << src << ',' << ig << ',' << xval << ','
                       << xsn[k] << ',' << xs.maxs(static_cast<int>(ig), xss[k]) << '\n';
        };

        const std::pair<const Model*, const char*> srcs[] = {{&fine, "fine"}, {&coarse, "coarse"}};

        for (const Axis& ax : axes) {
            // fine + coarse branch nodes at control type 0, burnup key 0 (plus the reference node).
            for (const auto& [mp, src] : srcs) {
                const Model& m   = *mp;
                const auto&  ref = m.GetDepletionPoint(0, 0);
                emit(ax.name, src, ref._data[ax.ad], ref._xs);
                const auto& br    = m.GetBranch(ax.btyp);
                const auto  brIt  = br.find(0);
                if (brIt == br.end()) continue;
                const auto buIt = brIt->second.find(0);
                if (buIt == brIt->second.end()) continue;
                for (size_t idx : buIt->second) {
                    const auto& dpt = m.GetDepletionPoint(idx);
                    emit(ax.name, src, dpt._data[ax.ad], dpt._xs);
                }
            }

            // interp sweep: vary this axis across the fine range, hold the others at reference.
            double      lo   = refC._data[ax.ad];
            double      hi   = refC._data[ax.ad];
            const auto& fbr  = fine.GetBranch(ax.btyp);
            const auto  fbIt = fbr.find(0);
            if (fbIt != fbr.end()) {
                const auto buIt = fbIt->second.find(0);
                if (buIt != fbIt->second.end())
                    for (size_t idx : buIt->second) {
                        const double v = fine.GetDepletionPoint(idx)._data[ax.ad];
                        lo             = std::min(lo, v);
                        hi             = std::max(hi, v);
                    }
            }
            const int N = 81;
            for (int i = 0; i < N; ++i) {
                const double xval = (hi > lo) ? lo + (hi - lo) * i / (N - 1) : lo;
                double       bppm = bppm0, tful = tful0, dmod = dmod0;
                if (ax.btyp == BPPM)
                    bppm = xval;
                else if (ax.btyp == TFUL)
                    tful = xval;
                else
                    dmod = xval;
                emit(ax.name, "interp", xval, coarse.GetCrossSection(0, 0, bppm, tful, dmod));
            }
        }
        std::cout << "[validate] wrote " << outPath << " (NGRP=" << NGRP << ")\n";
    }
};
} // namespace Chiffon
