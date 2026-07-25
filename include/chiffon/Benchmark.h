#pragma once

#include "Model.h"
#include <algorithm>
#include <cctype>
#include <format>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace Chiffon {

class Benchmark {
private:
    static constexpr XSTYPE      fields[]     = {XSTF, XSDF, XSAF, XSNF, XSSF};
    static constexpr const char* fieldNames[] = {"XSTF", "XSDF", "XSAF", "XSNF", "XSSF"};

    /// Reconstruct one XS value: lmpx + sum(micx * iden)
    static double Recon(const CrossSection& xs, const milk::Vector<double>& iden, int ig, XSTYPE xt) {
        double val = xs.lmpxs(ig, xt);
        for (size_t iso = 0; iso < niso; ++iso)
            val += xs.mixs(static_cast<int>(iso), ig, xt) * iden[iso];
        return val;
    }

    static void PrintHeader(const char* refLabel, const char* cmpLabel) {
        std::string hdr = std::format("{:<8} {:<8} {:<8} {:<8} {:<5} ", "BURN", "BPPM", "TFUL", "DMOD", "GRP");
        for (const auto& name : fieldNames)
            hdr += std::format("{:<11} {:<11} {:<11} ",
                               std::format("{}({})", name, refLabel),
                               std::format("{}({})", name, cmpLabel), "ERR(%)");
        std::cout << hdr << "\n";
    }

    static std::string FormatRow(int bu, double bppm, double tful, double dmod, size_t ig,
                                 auto getRef, auto getCmp) {
        std::string out = std::format("{:<8} {:<8.1f} {:<8.1f} {:<8.6f} {:<5} ", bu, bppm, tful, dmod, ig);
        for (auto xt : fields) {
            double ref = getRef(ig, xt);
            double cmp = getCmp(ig, xt);
            double err = (std::abs(ref) > 1e-30) ? 100.0 * (cmp - ref) / ref : 0.0;
            out += std::format("{:<11.4e} {:<11.4e} {:<11.4f} ", ref, cmp, err);
        }
        return out;
    }

    static std::string NormalizeName(std::string name) {
        name.erase(std::remove_if(name.begin(), name.end(),
                                  [](unsigned char c) { return std::isspace(c) || c == '-' || c == '_'; }),
                   name.end());
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return name;
    }

    static const char* IsotopeName(size_t iso) {
        return iso < isotopeIds.size() ? isotopeIds[iso] : "UNKNOWN";
    }

    static const char* XSTypeName(XSTYPE xsType) {
        switch (xsType) {
        case XSTF: return "XSTF";
        case XSDF: return "XSDF";
        case XSAF: return "XSAF";
        case XSFF: return "XSFF";
        case XSNF: return "XSNF";
        case XSKF: return "XSKF";
        case XSSF: return "XSSF";
        case XSRF: return "XSRF";
        case XS2N: return "XS2N";
        case XS3N: return "XS3N";
        default: return "UNKNOWN";
        }
    }

    /// Sweep one branch variable and print self-check rows.
    static void SweepBranch(const Model& model, BRANCHTYPE btyp, AssemblyDataIndex adIdx,
                            double step, double baseBppm, double baseTful, double baseDmod, size_t ngrp) {
        for (size_t ig = 0; ig < ngrp; ++ig) {
            for (const auto& [crType, buMap] : model.GetBranch(btyp)) {
                for (const auto& [bu, idxList] : buMap) {
                    std::vector<double> vals;
                    for (auto idx : idxList)
                        vals.push_back(model.GetDepletionPoint(idx)._data[adIdx]);
                    std::sort(vals.begin(), vals.end());
                    for (double v = vals.front(); v <= vals.back(); v += step) {
                        double bppm = baseBppm, tful = baseTful, dmod = baseDmod;
                        if (btyp == BPPM)
                            bppm = v;
                        else if (btyp == TFUL)
                            tful = v;
                        else if (btyp == DMOD)
                            dmod = v;
                        milk::Vector<double> iden;
                        auto                 xs = model.GetCrossSection(crType, bu, bppm, tful, dmod, &iden);
                        std::cout << FormatRow(bu, bppm, tful, dmod, ig, [&](int g, XSTYPE t) { return xs.maxs(g, t); }, [&](int g, XSTYPE t) { return Recon(xs, iden, g, t); }) << "\n";
                    }
                }
            }
        }
    }

    /// Print validation rows for one branch type.
    static void ValidateBranch(const Model& fine, const Model& coarse, BRANCHTYPE btyp, size_t ngrp,
                               bool compact) {
        for (size_t ig = 0; ig < ngrp; ++ig) {
            for (const auto& branchEntry : fine.GetBranch(btyp)) {
                const int   cr    = branchEntry.first;
                const auto& buMap = branchEntry.second;
                if (compact && !buMap.empty()) {
                    const int edgeCount = buMap.size() > 1 ? 2 : 1;
                    for (int edge = 0; edge < edgeCount; ++edge) {
                        const auto buIt = edge == 0 ? buMap.begin() : std::prev(buMap.end());
                        for (auto idx : buIt->second) {
                            const auto&          dpt = fine.GetDepletionPoint(idx);
                            milk::Vector<double> ciden;
                            auto                 cxs = coarse.GetCrossSection(cr, buIt->first,
                                                                              dpt._data[AD_BPPM], dpt._data[AD_TFUL],
                                                                              dpt._data[AD_DMOD], &ciden);
                            std::cout << FormatRow(buIt->first, dpt._data[AD_BPPM], dpt._data[AD_TFUL], dpt._data[AD_DMOD], ig, [&](int g, XSTYPE t) { return dpt._xs.maxs(g, t); }, [&](int g, XSTYPE t) { return Recon(cxs, ciden, g, t); }) << "\n";
                        }
                    }
                    continue;
                }
                for (const auto& [bu, idxList] : buMap) {
                    for (auto idx : idxList) {
                        const auto&          dpt = fine.GetDepletionPoint(idx);
                        milk::Vector<double> ciden;
                        auto                 cxs = coarse.GetCrossSection(cr, bu,
                                                                          dpt._data[AD_BPPM], dpt._data[AD_TFUL], dpt._data[AD_DMOD], &ciden);
                        std::cout << FormatRow(bu, dpt._data[AD_BPPM], dpt._data[AD_TFUL], dpt._data[AD_DMOD], ig, [&](int g, XSTYPE t) { return dpt._xs.maxs(g, t); }, [&](int g, XSTYPE t) { return Recon(cxs, ciden, g, t); }) << "\n";
                    }
                }
            }
        }
    }

public:
    struct ExtraIsotopeXSDumpSpec {
        bool                apply = false;
        std::string         fuel;
        size_t              isotope           = 0;
        int                 ctype             = 0;
        int                 burnup            = 0;
        int                 burnup_tolerance  = 0;
        int                 group             = -1;
        double              reference_density = std::numeric_limits<double>::quiet_NaN();
        std::vector<XSTYPE> xs_types          = {XSAF};
    };

    static XSTYPE ParseXSTypeName(const std::string& name, XSTYPE fallback = XSAF) {
        const std::string key = NormalizeName(name);
        if (key == "XSTF" || key == "TRANSPORT") return XSTF;
        if (key == "XSDF" || key == "DIFFUSION" || key == "D") return XSDF;
        if (key == "XSAF" || key == "ABSORPTION" || key == "ABS") return XSAF;
        if (key == "XSFF" || key == "FISSION" || key == "FIS") return XSFF;
        if (key == "XSNF" || key == "NUFISSION" || key == "NUFIS") return XSNF;
        if (key == "XSKF" || key == "KAPPAFISSION" || key == "KAPPAFIS") return XSKF;
        if (key == "XSSF" || key == "SCATTER" || key == "SCATTERING") return XSSF;
        if (key == "XSRF" || key == "REMOVAL") return XSRF;
        if (key == "XS2N" || key == "N2N") return XS2N;
        if (key == "XS3N" || key == "N3N") return XS3N;
        return fallback;
    }

    static void DumpExtraIsotopeXS(const Model& model, const ExtraIsotopeXSDumpSpec& spec) {
        if (!spec.apply)
            return;
        if (spec.isotope >= niso) {
            std::cout << "[Benchmark][extra isotope xs] skip: isotope index out of range\n";
            return;
        }
        if (model.SpctDpts().empty()) {
            std::cout << std::format("[Benchmark][extra isotope xs] model={} has no extra depletion points\n",
                                     model.name());
            return;
        }

        CrossSection                refXs;
        milk::Vector<double>        refIden;
        std::array<double, AD_SIZE> refData = {};
        std::vector<double>         refFlux;
        try {
            model.FillReferenceState(refXs, refIden, refData, &refFlux, spec.ctype, spec.burnup);
        } catch (const std::exception& e) {
            std::cout << std::format("[Benchmark][extra isotope xs] model={} reference lookup failed: {}\n",
                                     model.name(), e.what());
            return;
        }

        const double refDensity   = std::isfinite(spec.reference_density)
                                        ? spec.reference_density
                                        : refIden[spec.isotope];
        const int    firstGroup   = spec.group >= 0 ? spec.group : 0;
        const int    lastGroup    = spec.group >= 0 ? spec.group + 1 : static_cast<int>(refXs.ngrp());
        const double densityDenom = std::abs(refDensity) + 1.0e-30;

        std::cout << std::format(
            "[Benchmark][extra isotope xs] model={} isotope={}({}) ctype={} reference_burnup={} reference_density={:.8e}\n",
            model.name(), IsotopeName(spec.isotope), spec.isotope, spec.ctype, spec.burnup, refDensity);
        std::cout << "model,source,ctype,burnup,group,xs_type,isotope,isotope_index,"
                     "reference_burnup,reference_density,density,delta_density,relative_delta_density,"
                     "extra_macro_xs,ref_macro_xs,delta_macro_xs,extra_micro_xs,ref_micro_xs,delta_micro_xs\n";

        int printed = 0;
        for (const auto& dpt : model.SpctDpts()) {
            if (dpt._btyp != REFR || dpt._ctyp != spec.ctype)
                continue;
            const int bu = dpt.burnKey();
            if (std::abs(bu - spec.burnup) > spec.burnup_tolerance)
                continue;
            if (dpt._iden.size() <= spec.isotope)
                continue;

            const double density      = dpt._iden[spec.isotope];
            const double deltaDensity = density - refDensity;
            const double relDensity   = deltaDensity / densityDenom;

            for (int ig = firstGroup; ig < lastGroup; ++ig) {
                if (ig < 0 || ig >= static_cast<int>(dpt._xs.ngrp()))
                    continue;
                for (XSTYPE xsType : spec.xs_types) {
                    const double extraMacro = dpt._xs.maxs(ig, xsType);
                    const double refMacro   = refXs.maxs(ig, xsType);
                    const double extraMicro = dpt._xs.has_micx()
                                                  ? dpt._xs.mixs(static_cast<int>(spec.isotope), ig, xsType)
                                                  : std::numeric_limits<double>::quiet_NaN();
                    const double refMicro   = refXs.has_micx()
                                                  ? refXs.mixs(static_cast<int>(spec.isotope), ig, xsType)
                                                  : std::numeric_limits<double>::quiet_NaN();
                    std::cout << std::format(
                        "{},{},{},{},{},{},{},{},{},{:.8e},{:.8e},{:.8e},{:.8e},"
                        "{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}\n",
                        model.name(), "extra", dpt._ctyp, bu, ig, XSTypeName(xsType),
                        IsotopeName(spec.isotope), spec.isotope, spec.burnup,
                        refDensity, density, deltaDensity, relDensity,
                        extraMacro, refMacro, extraMacro - refMacro,
                        extraMicro, refMicro, extraMicro - refMicro);
                    ++printed;
                }
            }
        }

        if (printed == 0) {
            std::cout << std::format(
                "[Benchmark][extra isotope xs] no matching extra points: model={} ctype={} burnup={} tolerance={}\n",
                model.name(), spec.ctype, spec.burnup, spec.burnup_tolerance);
        }
    }

    /// Self-check: macx (original) vs lmpx+micx*iden (reconstructed from interpolation)
    static void Verificate(const Model& model, int cType = 0) {
        std::cout << " [Benchmark] Model Interpolation Verification \n";

        const size_t NGRP  = model.GetDepletionPoint(0, 0)._ngrp;
        const double bppm0 = model.GetDepletionPoint(0, 0)._data[AD_BPPM];
        const double tful0 = model.GetDepletionPoint(0, 0)._data[AD_TFUL];
        const double dmod0 = model.GetDepletionPoint(0, 0)._data[AD_DMOD];

        PrintHeader("MAC", "REC");

        // Burnup sweep
        for (size_t ig = 0; ig < NGRP; ++ig) {
            for (int bu = 0; bu < 60000; bu += 500) {
                milk::Vector<double> iden;
                auto                 xs = model.GetCrossSection(cType, bu, bppm0, tful0, dmod0, &iden);
                std::cout << FormatRow(bu, bppm0, tful0, dmod0, ig, [&](int g, XSTYPE t) { return xs.maxs(g, t); }, [&](int g, XSTYPE t) { return Recon(xs, iden, g, t); }) << "\n";
            }
        }

        SweepBranch(model, BRANCHTYPE::BPPM, AD_BPPM, 100.0, bppm0, tful0, dmod0, NGRP);
        SweepBranch(model, BRANCHTYPE::TFUL, AD_TFUL, 100.0, bppm0, tful0, dmod0, NGRP);
        SweepBranch(model, BRANCHTYPE::DMOD, AD_DMOD, 0.05, bppm0, tful0, dmod0, NGRP);
    }

    /// Cross-model: fine macx vs coarse reconstructed
    static void Validate(const Model& fine, const Model& coarse, bool compact = false) {
        std::cout << " [Benchmark] Model Interpolation Validation \n";

        const size_t NGRP = coarse.GetDepletionPoint(0, 0)._ngrp;
        PrintHeader("fine", "cors");

        std::cout << "================ BPPM VALIDATION ================\n";
        ValidateBranch(fine, coarse, BPPM, NGRP, compact);
        std::cout << "================ TFUEL VALIDATION ================\n";
        ValidateBranch(fine, coarse, TFUL, NGRP, compact);
        std::cout << "================ DMOD VALIDATION ================\n";
        ValidateBranch(fine, coarse, DMOD, NGRP, compact);
    }
};
} // namespace Chiffon
