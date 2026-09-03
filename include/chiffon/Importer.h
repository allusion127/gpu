#pragma once

#include "Interpolator.h"
#include "Hdf5Guard.h"
#include "Model.h"
#include "Parser.h"
#include "ReflectorSolver.h"
#include <cstdio>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace Chiffon {

// Reads HGC lattice code output and HDF5 files into Model objects
class Importer {
public:
    // Benchmark reuses the HGC loader + rod-fluence reconstruction (both below) to export
    // per-point isotope/coordinate state for offline statistics, without adding any I/O
    // surface to the main classes. See Benchmark::DumpIsotopeStates.
    friend class Benchmark;

    Importer() = default;

private:
    static int BurnKeyFromDouble(double burn) {
        const double key = std::floor(1000.0 * burn);
        if (key > static_cast<double>(std::numeric_limits<int>::max()))
            return std::numeric_limits<int>::max();
        if (key < static_cast<double>(std::numeric_limits<int>::min()))
            return std::numeric_limits<int>::min();
        return static_cast<int>(key);
    }

    static bool IsSeparatorChar(unsigned char c) { return std::isspace(c) || c == '-' || c == '_'; }

    // Normalize a token for tolerant matching: drop separators (space/-/_) and uppercase.
    static std::string ToUpper(std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return IsSeparatorChar(c); }), s.end());
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return s;
    }

    // Normalize a token for tolerant matching: drop separators (space/-/_) and lowercase.
    static std::string ToLower(std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return IsSeparatorChar(c); }), s.end());
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    // Reconstruct accumulated thermal fluence [n/cm2] along an HGC's depletion
    // trajectory by integrating the thermal-group flux over EFPD (derived from burnup
    // and power density). This replaces the old external rod-fluence CSV: the HGC
    // already carries the flux and burnup needed for flux x time.
    static std::map<int, double> ReconstructFluenceFromHGC(const Model& hgc) {
        struct Step {
            int    key;
            double burn;
            double flux;
            double pden;
        };
        std::vector<Step> steps;
        for (const auto& dpt : hgc.Dpts()) {
            if (dpt._btyp != REFR)
                continue;
            double flux = 0.0;
            if (!dpt._aflx.empty())
                flux = dpt._aflx.back();
            steps.push_back({dpt.burnKey(), dpt._data[AD_BURN], flux, dpt._data[AD_PDEN]});
        }
        std::sort(steps.begin(), steps.end(),
                  [](const Step& a, const Step& b) { return a.burn < b.burn; });

        std::map<int, double> fluenceByBurnup;
        double                fluence  = 0.0;
        double                prevEfpd = 0.0;
        double                prevFlux = 0.0;
        bool                  first    = true;
        // R6 guard: the RDPL abscissa is thermal fluence, reconstructed as flux x EFPD with
        // EFPD = burn * 1000 / pden. A missing or zero %TITL pden (or an absent %FLUX block)
        // silently collapses the fluence to zero, which disables the rod-depletion layer with
        // no other symptom. Count the degenerate points and warn once per HGC model.
        {
            int badPden = 0, zeroFlux = 0, burnedSteps = 0;
            for (const auto& s : steps) {
                if (!(s.burn > 0.0))
                    continue;
                ++burnedSteps;
                if (!(s.pden > 0.0))
                    ++badPden;
                if (!(s.flux > 0.0))
                    ++zeroFlux;
            }
            if (burnedSteps > 0 && (badPden > 0 || zeroFlux == burnedSteps)) {
                std::fprintf(stderr,
                             "[CHIFFON][WARNING] rod-fluence reconstruction is degenerate for "
                             "model '%s': %d/%d depleted points have pden<=0 and %d have zero "
                             "thermal flux. The accumulated fluence stays 0, so the RDPL (rod "
                             "absorber depletion) layer is silently inactive. Check the HGC "
                             "%%TITL pden field and the %%FLUX block.\n",
                             hgc.name().c_str(), badPden, burnedSteps, zeroFlux);
            }
        }
        for (const auto& s : steps) {
            // EFPD [d] = burnup [MWd/kgHM] * 1000 / power density [W/gHM].
            const double efpd = s.pden > 0.0 ? s.burn * 1000.0 / s.pden : 0.0;
            if (!first) {
                const double dt = (efpd - prevEfpd) * 86400.0; // days -> seconds
                if (dt > 0.0)
                    fluence += 0.5 * (prevFlux + s.flux) * dt;
            }
            fluenceByBurnup[s.key] = fluence;
            prevEfpd               = efpd;
            prevFlux               = s.flux;
            first                  = false;
        }
        return fluenceByBurnup;
    }

    // Burnup-interpolate a reconstructed fluence table at a depletion point's burnup.
    static double InterpolateRodFluence(const DepletionPoint&        dpt,
                                        const std::map<int, double>& fluenceByBurnup) {
        return InterpolateFluenceAtKey(dpt.burnKey(), fluenceByBurnup);
    }

    static double InterpolateFluenceAtKey(int burnKey, const std::map<int, double>& fluenceByBurnup) {
        if (fluenceByBurnup.empty())
            return std::numeric_limits<double>::quiet_NaN();
        auto hiIt = fluenceByBurnup.lower_bound(burnKey);
        if (hiIt != fluenceByBurnup.end() && hiIt->first == burnKey)
            return hiIt->second;
        if (hiIt == fluenceByBurnup.end())
            return std::prev(hiIt)->second;
        if (hiIt == fluenceByBurnup.begin())
            return hiIt->second;
        auto         loIt = std::prev(hiIt);
        const double span = static_cast<double>(hiIt->first - loIt->first);
        const double frac = span > 0.0 ? static_cast<double>(burnKey - loIt->first) / span : 0.0;
        return loIt->second + frac * (hiIt->second - loIt->second);
    }

    // Assign reconstructed fuel-exposure and rod-material fluence to one point. The
    // nondepleted counterfactual keeps a zero rod-material fluence (fresh absorber).
    static void AssignReconstructedFluence(DepletionPoint& dpt, const std::map<int, double>& fluenceByBurnup,
                                           bool nondepleted) {
        const double flu      = InterpolateRodFluence(dpt, fluenceByBurnup);
        dpt._fuel_rod_fluence = flu;
        dpt._rod_fluence      = nondepleted ? 0.0 : flu;
    }

    static void AddVectorInto(std::vector<double>& dst, const std::vector<double>& src) {
        if (dst.size() != src.size())
            throw std::runtime_error("CHIFFON reflector average: vector size mismatch.");
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] += src[i];
    }

    static void ScaleVector(std::vector<double>& dst, double scale) {
        for (double& value : dst)
            value *= scale;
    }

    static void AddMilkVectorInto(milk::Vector<double>& dst, const milk::Vector<double>& src) {
        if (dst.size() != src.size())
            throw std::runtime_error("CHIFFON reflector average: isotope vector size mismatch.");
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] += src[i];
    }

    static void ScaleMilkVector(milk::Vector<double>& dst, double scale) {
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] *= scale;
    }

    static void AddDepletionPointInto(DepletionPoint& dst, const DepletionPoint& src) {
        if (dst._ngrp != src._ngrp || dst._npin != src._npin || dst._btyp != src._btyp || dst._ctyp != src._ctyp)
            throw std::runtime_error("CHIFFON reflector average: depletion point metadata mismatch.");
        if (dst._trajectory_ctyp != src._trajectory_ctyp || dst._trajectory_reference != src._trajectory_reference)
            throw std::runtime_error("CHIFFON reflector average: history metadata mismatch.");

        for (size_t i = 0; i < dst._data.size(); ++i)
            dst._data[i] += src._data[i];
        AddVectorInto(dst._aflx, src._aflx);
        AddVectorInto(dst._gmap, src._gmap);
        AddVectorInto(dst._fmap, src._fmap);
        AddVectorInto(dst._chix, src._chix);
        AddVectorInto(dst._sdfa, src._sdfa);
        AddVectorInto(dst._pdfa, src._pdfa);
        AddMilkVectorInto(dst._iden, src._iden);
        dst._xs += src._xs;
    }

    /// @brief Resolve every "history partner" name to a model index.
    ///
    /// The partner is the rodded-depletion twin a fuel blends toward as its
    /// nodes accumulate rod exposure. Names are resolved only after all models
    /// exist, so the partner may be declared before it is read.
    static void ResolveHistoryPartners(std::vector<Model>& models) {
        for (auto& model : models) {
            if (model.history_partner_name().empty())
                continue;
            const auto it = std::find_if(
                models.begin(), models.end(), [&](const Model& candidate) {
                    return candidate.name() == model.history_partner_name();
                });
            if (it == models.end())
                throw std::runtime_error(
                    "Fuel '" + model.name() + "': history partner '" +
                    model.history_partner_name() + "' is not a declared fuel.");
            model.history_partner() = static_cast<int>(it - models.begin());
        }
    }

    static void ScaleDepletionPoint(DepletionPoint& dpt, double scale) {
        for (double& value : dpt._data)
            value *= scale;
        ScaleVector(dpt._aflx, scale);
        ScaleVector(dpt._gmap, scale);
        ScaleVector(dpt._fmap, scale);
        ScaleVector(dpt._chix, scale);
        ScaleVector(dpt._sdfa, scale);
        ScaleVector(dpt._pdfa, scale);
        ScaleMilkVector(dpt._iden, scale);
        dpt._xs *= scale;
    }

    static void AddDeltaCrossSectionInto(DeltaCrossSection& dst, const DeltaCrossSection& src) {
        if (dst.ngrp() != src.ngrp() || dst.nord() != src.nord() || dst.mode() != src.mode() ||
            dst.ncoeff() != src.ncoeff() || dst.knots() != src.knots()) {
            throw std::runtime_error("CHIFFON reflector average: branch delta shape mismatch.");
        }
        for (size_t i = 0; i < dst.nord(); ++i)
            dst[i] += src[i];
    }

    static void ScaleDeltaCrossSection(DeltaCrossSection& dxs, double scale) {
        for (size_t i = 0; i < dxs.nord(); ++i)
            dxs[i] *= scale;
    }

    static void AddBranchDeltaInto(BranchDelta& dst, const BranchDelta& src) {
        if (dst.empty() && src.empty())
            return;
        if (dst.size() != src.size())
            throw std::runtime_error("CHIFFON reflector average: branch delta map mismatch.");

        for (const auto& [ctype, srcBurnMap] : src) {
            auto dstCtype = dst.find(ctype);
            if (dstCtype == dst.end() || dstCtype->second.size() != srcBurnMap.size())
                throw std::runtime_error("CHIFFON reflector average: branch delta ctype mismatch.");

            for (const auto& [burnup, srcDelta] : srcBurnMap) {
                auto dstBurn = dstCtype->second.find(burnup);
                if (dstBurn == dstCtype->second.end())
                    throw std::runtime_error("CHIFFON reflector average: branch delta burnup mismatch.");
                AddDeltaCrossSectionInto(dstBurn->second, srcDelta);
            }
        }
    }

    static void ScaleBranchDelta(BranchDelta& delta, double scale) {
        for (auto& [ctype, burnMap] : delta)
            for (auto& [burnup, dxs] : burnMap)
                ScaleDeltaCrossSection(dxs, scale);
    }

    static void AddSpectralHistoryInto(
        std::vector<SpectralHistoryCorrection>& dst,
        const std::vector<SpectralHistoryCorrection>& src) {
        if (dst.size() != src.size())
            throw std::runtime_error(
                "CHIFFON reflector average: spectral-history term count mismatch.");

        for (size_t i = 0; i < dst.size(); ++i) {
            if (!(dst[i].term == src[i].term) ||
                dst[i].delta.size() != src[i].delta.size()) {
                throw std::runtime_error(
                    "CHIFFON reflector average: spectral-history layout mismatch.");
            }
            for (const auto& [burnup, sourceDelta] : src[i].delta) {
                auto target = dst[i].delta.find(burnup);
                if (target == dst[i].delta.end()) {
                    throw std::runtime_error(
                        "CHIFFON reflector average: spectral-history burnup mismatch.");
                }
                AddDeltaCrossSectionInto(target->second, sourceDelta);
            }
        }
    }

    static void ScaleSpectralHistory(
        std::vector<SpectralHistoryCorrection>& corrections,
        double scale) {
        for (auto& correction : corrections)
            for (auto& [burnup, delta] : correction.delta) {
                (void)burnup;
                ScaleDeltaCrossSection(delta, scale);
            }
    }

    static void CheckAverageModelLayout(const Model& dst, const Model& src) {
        if (dst._dpts.size() != src._dpts.size())
            throw std::runtime_error("CHIFFON reflector average: depletion point count mismatch.");
        if (dst._refr_dpts != src._refr_dpts ||
            dst._bppm_dpts != src._bppm_dpts ||
            dst._tful_dpts != src._tful_dpts ||
            dst._dmod_dpts != src._dmod_dpts) {
            throw std::runtime_error("CHIFFON reflector average: branch point layout mismatch.");
        }
    }

    static Model AverageReflectorModels(const std::string& reflName, const std::vector<Model*>& nodes) {
        if (nodes.empty())
            throw std::runtime_error("CHIFFON reflector average: empty node list for '" + reflName + "'.");

        Model averaged = *nodes.front();
        averaged._name = reflName;
        for (size_t i = 1; i < nodes.size(); ++i) {
            CheckAverageModelLayout(averaged, *nodes[i]);
            for (size_t d = 0; d < averaged._dpts.size(); ++d)
                AddDepletionPointInto(averaged._dpts[d], nodes[i]->_dpts[d]);
            AddBranchDeltaInto(averaged._bppm_delt, nodes[i]->_bppm_delt);
            AddBranchDeltaInto(averaged._tful_delt, nodes[i]->_tful_delt);
            AddBranchDeltaInto(averaged._dmod_delt, nodes[i]->_dmod_delt);
            AddSpectralHistoryInto(
                averaged._spectral_history,
                nodes[i]->_spectral_history);
        }

        const double scale = 1.0 / static_cast<double>(nodes.size());
        for (auto& dpt : averaged._dpts)
            ScaleDepletionPoint(dpt, scale);
        ScaleBranchDelta(averaged._bppm_delt, scale);
        ScaleBranchDelta(averaged._tful_delt, scale);
        ScaleBranchDelta(averaged._dmod_delt, scale);
        ScaleSpectralHistory(averaged._spectral_history, scale);
        return averaged;
    }

    static Model& FindReflectorNode(std::unordered_map<std::string, Model>& reflModels, const std::string& nodeName) {
        auto nodeIt = reflModels.find(nodeName);
        if (nodeIt == reflModels.end())
            throw std::runtime_error("CHIFFON input: unknown reflector node '" + nodeName + "'.");
        return nodeIt->second;
    }

    // Resolve an optional "left/right neighbor" field of a reflector spec to either a
    // neighbouring reflector model or an outer boundary condition. Upstream v3.0.0 deleted
    // this together with the commented-out reflector DF; our WF4 work restored the DF path
    // (RASBERY_REFL_DF, default off), so the parsing has to come back with it.
    static void ParseOptionalReflectorNeighbor(std::unordered_map<std::string, Model>& reflModels,
                                               const nlohmann::json&                   specs,
                                               const std::string&                      field,
                                               Model*&                                 neighbor,
                                               BOUNDARY&                               boundary) {
        if (!specs.contains(field)) {
            neighbor = nullptr;
            boundary = NODE;
            return;
        }

        const std::string name = specs.at(field).get<std::string>();
        if (name == "VACUUM") {
            neighbor = nullptr;
            boundary = VACUUM;
        } else if (name == "REFLECTIVE") {
            neighbor = nullptr;
            boundary = REFLECTIVE;
        } else {
            // A name that is neither a boundary keyword nor a declared reflector
            // node used to resolve to nullptr with boundary=NODE -- "there is a
            // neighbouring node, and it is nothing". A typo in the node name was
            // then indistinguishable from an omitted field. Four lines above,
            // FindReflectorNode() throws on exactly this mistake for the
            // "left node"/"right node" fields; the same rule applies here.
            neighbor = &FindReflectorNode(reflModels, name);
            boundary = NODE;
        }
    }

    // Resolve one HGC file reference from a --chiffoni JSON document. An absolute
    // path (POSIX leading '/', a Windows drive letter like "C:\" or "C:/", or a
    // leading '\\' UNC/drive-relative marker) is used exactly as given; anything
    // else is resolved relative to the json's own directory (baseDir). Using
    // std::filesystem::path::is_absolute() alone is not enough here because its
    // notion of "absolute" is platform-dependent (a POSIX build does not treat
    // "C:\foo" as absolute), so both conventions are checked explicitly -- the
    // input json may reference paths from either OS regardless of what this
    // binary is compiled for.
    static bool IsAbsoluteHGCPath(const std::string& fileName) {
        if (fileName.empty())
            return false;
        if (fileName[0] == '/' || fileName[0] == '\\')
            return true;
        if (fileName.size() >= 2 && std::isalpha(static_cast<unsigned char>(fileName[0])) &&
            fileName[1] == ':')
            return true;
        return false;
    }

    static std::string ResolveHGCPath(const std::filesystem::path& baseDir,
                                       const std::string& fileName) {
        if (IsAbsoluteHGCPath(fileName))
            return fileName;
        return (baseDir / fileName).string();
    }

    // Append one HGC file as spectral or rodded-history fitting data.
    void AppendHGCPoints(Model& targetModel, const std::filesystem::path& baseDir,
                         const std::string& fileName, ReflectorSolver& reflSolver,
                         BRANCHTYPE targetType, int trajectoryCtype = -1,
                         bool nondepleted = false,
                         const std::string& adfReferenceFile = "",
                         bool invertCtype = false) {
        Model hgcModel(targetModel.name() + "#branch");
        ReadHGC(hgcModel, ResolveHGCPath(baseDir, fileName), invertCtype);
        if (adfReferenceFile.empty()) {
            reflSolver.ApplyDF(hgcModel);
        } else {
            Model adfReference(targetModel.name() + "#adf_reference");
            ReadHGC(adfReference, ResolveHGCPath(baseDir, adfReferenceFile),
                    invertCtype);
            std::map<int, std::pair<double, double>> referenceAdf;
            for (const auto& dpt : adfReference._dpts)
                if (dpt._btyp == REFR && dpt._sdfa.size() >= 5)
                    referenceAdf.emplace(dpt.burnKey(),
                                         std::make_pair(dpt._sdfa[0], dpt._sdfa[4]));
            for (auto& dpt : hgcModel._dpts) {
                auto it = referenceAdf.find(dpt.burnKey());
                if (it == referenceAdf.end() || dpt._sdfa.size() < 5)
                    continue;
                dpt._sdfa[0] = it->second.first;
                dpt._sdfa[4] = it->second.second;
            }
            reflSolver.ApplyDF(hgcModel);
        }

        if (targetType == BRANCHTYPE::SPECTRAL_HISTORY) {
            // Keep instantaneous insertion points with the rodded-history samples.
            const auto spectralFluence = ReconstructFluenceFromHGC(hgcModel);
            for (auto& dpt : hgcModel._dpts) {
                if (dpt._ctyp > 0) {
                    AssignReconstructedFluence(
                        dpt, spectralFluence, /*nondepleted=*/true);
                    dpt._trajectory_ctyp      = 0;
                    dpt._trajectory_reference = false;
                    targetModel._rod_history_dpts.push_back(std::move(dpt));
                    continue;
                }
                targetModel._spectral_history_dpts.push_back(std::move(dpt));
            }
            return;
        }

        if (targetType != BRANCHTYPE::ROD_HISTORY)
            return;

        // Rod fluence is reconstructed from this HGC's own flux x time (no external CSV).
        const auto fluence = ReconstructFluenceFromHGC(hgcModel);

        for (auto& dpt : hgcModel._dpts) {
            AssignReconstructedFluence(dpt, fluence, nondepleted);

            const int hgcCtype        = dpt._ctyp;
            dpt._trajectory_ctyp      = trajectoryCtype;
            dpt._trajectory_reference = false;
            dpt._nondepleted          = nondepleted;

            if (trajectoryCtype >= 0) {
                // In a rod-history HGC the base reference was depleted rodded.
                // Its CR* branch is therefore the rod-out current state.
                dpt._ctyp = (hgcCtype == 0) ? trajectoryCtype : 0;
            }

            targetModel._rod_history_dpts.push_back(std::move(dpt));
        }
    }

    void AppendRodDepletionHGC(Model& targetModel, const std::filesystem::path& baseDir,
                               const std::string& fileName, ReflectorSolver& reflSolver) {
        Model hgcModel(targetModel.name() + "#rod_depletion");
        ReadHGC(hgcModel, ResolveHGCPath(baseDir, fileName));
        reflSolver.ApplyDF(hgcModel);

        for (auto& dpt : hgcModel._dpts)
            targetModel._rod_depletion_dpts.push_back(std::move(dpt));
    }

    void AppendRodDepletionPairHGC(Model& targetModel, const std::filesystem::path& baseDir,
                                   const std::string& referenceFile, const std::string& depletedFile,
                                   ReflectorSolver& reflSolver) {
        Model referenceModel(targetModel.name() + "#rod_depletion_reference");
        ReadHGC(referenceModel, ResolveHGCPath(baseDir, referenceFile));
        reflSolver.ApplyDF(referenceModel);

        // Nondepleted reference: fresh absorber, so rod-material fluence is zero.
        const auto referenceFluence = ReconstructFluenceFromHGC(referenceModel);
        for (auto& dpt : referenceModel._dpts) {
            AssignReconstructedFluence(dpt, referenceFluence, /*nondepleted=*/true);
            dpt._trajectory_reference = true;
            targetModel._rod_depletion_dpts.push_back(std::move(dpt));
        }

        Model depletedModel(targetModel.name() + "#rod_depletion_depleted");
        ReadHGC(depletedModel, ResolveHGCPath(baseDir, depletedFile));
        // Common-gauge ADF for the pair: the aged deck's own thermal ADF drifts up to
        // -4% from the fresh deck's, which would pollute every channel of the pair delta
        // with O(full sigma x 4%) convention noise — larger than the physics it stores.
        // Divide both decks by the REFERENCE deck's ADF at the same burnup instead.
        std::map<int, std::pair<double, double>> refAdf;
        for (const auto& dpt : targetModel._rod_depletion_dpts)
            if (dpt._trajectory_reference && dpt._sdfa.size() >= 5)
                refAdf.emplace(dpt.burnKey(), std::make_pair(dpt._sdfa[0], dpt._sdfa[4]));
        for (auto& dpt : depletedModel._dpts) {
            auto it = refAdf.find(dpt.burnKey());
            if (it == refAdf.end() || dpt._sdfa.size() < 5)
                continue;
            dpt._sdfa[0] = it->second.first;
            dpt._sdfa[4] = it->second.second;
        }
        reflSolver.ApplyDF(depletedModel);

        const auto depletedFluence = ReconstructFluenceFromHGC(depletedModel);
        for (auto& dpt : depletedModel._dpts) {
            AssignReconstructedFluence(dpt, depletedFluence, /*nondepleted=*/false);
            dpt._trajectory_reference = false;
            targetModel._rod_depletion_dpts.push_back(std::move(dpt));
        }
    }

    // Dispatch a fuel "rod depletion" / "rod_depletion" spec (string, array, or object;
    // an object may describe a reference/depleted pair) to the matching HGC appender.
    void AppendFuelRodDepletion(Model& targetModel, const std::filesystem::path& baseDir,
                                const nlohmann::json& block, ReflectorSolver& reflSolver) {
        // R6 guard: only the object form {"nondepleted": ..., "depleted": ...} establishes the
        // counterfactual pair that RDPL is defined from. The string/array forms fall back to an
        // unpaired trajectory, which yields a physically wrong (but silent) rod-depletion delta.
        if (!block.is_object() || !(block.contains("reference") || block.contains("nondepleted"))) {
            std::fprintf(stderr,
                         "[CHIFFON][WARNING] model '%s': \"rod depletion\" is not given as a "
                         "{nondepleted, depleted} pair. RDPL falls back to the unpaired path and "
                         "the resulting absorber-depletion delta is not the RODDEPL-RODNONDEPL "
                         "difference the method is defined by.\n",
                         targetModel.name().c_str());
        }
        if (block.is_string()) {
            AppendRodDepletionHGC(targetModel, baseDir, block.get<std::string>(), reflSolver);
        } else if (block.is_array()) {
            for (const auto& item : block)
                AppendRodDepletionHGC(targetModel, baseDir,
                                      item.is_string() ? item.get<std::string>() : item.at("file").get<std::string>(),
                                      reflSolver);
        } else if (block.is_object()) {
            if (block.contains("reference") || block.contains("nondepleted")) {
                const std::string referenceFile =
                    block.contains("reference") ? block.at("reference").get<std::string>()
                                                : block.at("nondepleted").get<std::string>();
                AppendRodDepletionPairHGC(targetModel, baseDir, referenceFile,
                                          block.at("depleted").get<std::string>(),
                                          reflSolver);
            } else {
                AppendRodDepletionHGC(targetModel, baseDir, block.at("file").get<std::string>(), reflSolver);
            }
        }
    }

    // Parse the first contiguous integer in a rod-type label.
    static int ParseRodType(const std::string& label) {
        const auto begin = std::find_if(
            label.begin(), label.end(), [](unsigned char value) {
                return std::isdigit(value);
            });
        if (begin == label.end())
            throw std::runtime_error(
                "CHIFFON input: rod type must contain an integer.");
        const auto end = std::find_if(
            begin, label.end(), [](unsigned char value) {
                return !std::isdigit(value);
            });
        return std::stoi(std::string(begin, end));
    }

    // Append one rodded depletion trajectory and its optional branch files.
    void AppendRodHistorySpec(Model& model, const std::filesystem::path& baseDir,
                              const std::string& rodKey, const nlohmann::json& rodSpec,
                              ReflectorSolver& reflSolver) {
        int trajectoryCtype = ParseRodType(rodKey);

        if (rodSpec.contains("ctype")) {
            if (rodSpec["ctype"].is_number_integer()) {
                trajectoryCtype = rodSpec["ctype"].get<int>();
            } else if (rodSpec["ctype"].is_string()) {
                trajectoryCtype =
                    ParseRodType(rodSpec["ctype"].get<std::string>());
            } else {
                throw std::runtime_error("CHIFFON input: rod history ctype must be an integer or string.");
            }
        }
        if (trajectoryCtype <= 0)
            throw std::runtime_error(
                "CHIFFON input: rod type must be positive.");

        // The nondepleted counterfactual (fresh absorber) keeps a zero rod-material fluence.
        // Mark it with "nondepleted": true, or name the entry "..._non_depleted". Rod fluence
        // is reconstructed from each HGC's flux x time, so no fluence files are needed.
        bool nondepleted = rodSpec.value("nondepleted", false);
        if (!nondepleted && ToLower(rodKey).find("nondepl") != std::string::npos)
            nondepleted = true;

        const std::string adfReference =
            rodSpec.value("adf reference", std::string{});

        if (rodSpec.contains("main"))
            throw std::runtime_error(
                "CHIFFON input: rod-history 'main' is unsupported; "
                "provide branch fitting decks through 'extra'.");

        if (rodSpec.contains("extra"))
            for (const auto& item : rodSpec["extra"])
                AppendHGCPoints(model, baseDir, item.get<std::string>(),
                                reflSolver, BRANCHTYPE::ROD_HISTORY,
                                trajectoryCtype, nondepleted,
                                adfReference);
    }

    // Case-insensitive lookup of a fixed structural settings key. The canonical
    // form is lowercase words separated by spaces (e.g. "rod depletion"); inputs
    // may differ only in letter case. Returns nullptr if the key is absent.
    static const nlohmann::json* FindSetting(const nlohmann::json& obj, const char* lowerKey) {
        if (!obj.is_object()) return nullptr;
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            std::string k = it.key();
            std::transform(k.begin(), k.end(), k.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (k == lowerKey) return &it.value();
        }
        return nullptr;
    }

public:
    // Parse JSON input, read HGC files, apply discontinuity factors, then interpolate all models
    void ReadInput(const std::filesystem::path& filepath, std::vector<Model>& models) {
        ReflectorSolver reflSolver;
        std::ifstream   file(filepath);
        if (!file.is_open()) throw std::runtime_error("Failed to open input file: " + filepath.string());
        nlohmann::json input;
        file >> input;

        std::filesystem::path baseDir = filepath.parent_path();

        std::vector<std::string> globalRodDepletionFiles;
        const nlohmann::json*    globalRodDepletionBlock = FindSetting(input, "rod depletion");
        if (globalRodDepletionBlock != nullptr) {
            const auto& block = *globalRodDepletionBlock;
            if (block.is_string()) {
                globalRodDepletionFiles.push_back(block.get<std::string>());
            } else if (block.is_array()) {
                for (const auto& item : block)
                    globalRodDepletionFiles.push_back(item.is_string() ? item.get<std::string>() : item.at("file").get<std::string>());
            } else if (block.is_object()) {
                globalRodDepletionFiles.push_back(block.at("file").get<std::string>());
            }
        }

        // Read reference, spectral-history, and rodded-history HGC files.
        for (const auto& [fuelName, fuelSpec] : input.at("fuels").items()) {
            std::string mainFile;

            if (fuelSpec.is_string()) {
                mainFile = fuelSpec.get<std::string>();
            } else {
                mainFile = fuelSpec.at("main").get<std::string>();
            }

            const bool roddedBase =
                fuelSpec.is_object() && fuelSpec.value("rodded base", false);

            Model model(fuelName);
            ReadHGC(model, ResolveHGCPath(baseDir, mainFile), roddedBase);
            reflSolver.ApplyDF(model);

            if (fuelSpec.is_object()) {
                if (fuelSpec.contains("history partner"))
                    model.history_partner_name() =
                        fuelSpec["history partner"].get<std::string>();

                if (fuelSpec.contains("extra")) {
                    for (const auto& item : fuelSpec["extra"]) {
                        // The off-nominal decks of a rodded-base fuel carry the
                        // same inverted rod tag as its main HGC.
                        AppendHGCPoints(model, baseDir, item.get<std::string>(),
                                        reflSolver,
                                        BRANCHTYPE::SPECTRAL_HISTORY, -1, false,
                                        "", roddedBase);
                    }
                }

                if (fuelSpec.contains("rod history")) {
                    for (const auto& [rodKey, rodSpec] : fuelSpec["rod history"].items())
                        AppendRodHistorySpec(model, baseDir, rodKey, rodSpec, reflSolver);
                }

                if (const nlohmann::json* rodDepl = FindSetting(fuelSpec, "rod depletion"))
                    AppendFuelRodDepletion(model, baseDir, *rodDepl, reflSolver);
            }

            for (const auto& file : globalRodDepletionFiles)
                AppendRodDepletionHGC(model, baseDir, file, reflSolver);
            models.push_back(std::move(model));
        }

        ResolveHistoryPartners(models);

        // Reflectors parsing
        if (input.contains("reflectors")) {
            std::unordered_map<std::string, Model> reflModels;
            for (const auto& [reflName, reflFile] : input["reflectors"]["files"].items()) {
                std::string filename = reflFile.get<std::string>();
                Model       model(reflName);
                ReadHGC(model, ResolveHGCPath(baseDir, filename));
                reflModels[reflName] = std::move(model);
            }

            // Parse reflector neighbors information
            if (input["reflectors"].contains("neighbors")) {
                for (const auto& [reflName, specs] : input["reflectors"]["neighbors"].items()) {
                    std::string typeStr = specs.at("type").get<std::string>();

                    if (typeStr == "averaged" || typeStr == "average") {
                        // Optional fuel-facing surface per node. When given, the reflector node's
                        // own interface DF is folded into its XS before averaging, mirroring what
                        // ApplyDF/ApplyDF2 do for the paired axial reflectors. Without it these
                        // nodes carry no DF at all (historical behaviour). ReflectorSolver itself
                        // no-ops unless RASBERY_REFL_DF is set, so the default is unchanged.
                        std::vector<std::string> fuelSides;
                        if (specs.contains("fuel sides")) {
                            for (const auto& item : specs.at("fuel sides"))
                                fuelSides.push_back(item.get<std::string>());
                            // The two arrays are positional: fuelSides[i] names the
                            // fuel-facing surface of nodes[i].  A short list left the
                            // trailing nodes with no DF at all (the `nodeIdx <
                            // fuelSides.size()` guard below silently skipped them),
                            // and a long one dropped the surplus -- either way the
                            // averaged reflector came out of a different recipe than
                            // the input describes, with nothing said.
                            if (fuelSides.size() != specs.at("nodes").size())
                                throw std::runtime_error(
                                    "CHIFFON input: reflector '" + reflName + "' declares " +
                                    std::to_string(fuelSides.size()) + " 'fuel sides' for " +
                                    std::to_string(specs.at("nodes").size()) +
                                    " nodes; the two lists are positional and must have the "
                                    "same length (use the scalar 'fuel side' to apply one "
                                    "surface to every node).");
                        } else if (specs.contains("fuel side")) {
                            const std::string sd = specs.at("fuel side").get<std::string>();
                            fuelSides.assign(specs.at("nodes").size(), sd);
                        }

                        std::vector<Model>  dfNodes; // owns the DF-corrected copies
                        std::vector<Model*> avgNodes;
                        dfNodes.reserve(specs.at("nodes").size());
                        size_t nodeIdx = 0;
                        for (const auto& nodeNameValue : specs.at("nodes")) {
                            const std::string nodeName = nodeNameValue.get<std::string>();
                            auto              nodeIt   = reflModels.find(nodeName);
                            if (nodeIt == reflModels.end())
                                throw std::runtime_error("CHIFFON input: unknown averaged reflector node '" + nodeName + "'.");
                            dfNodes.push_back(nodeIt->second);
                            if (nodeIdx < fuelSides.size()) {
                                const int side = ReflectorSolver::ParseSide(fuelSides[nodeIdx]);
                                if (side < 0)
                                    throw std::runtime_error("CHIFFON input: bad 'fuel side' value '" +
                                                             fuelSides[nodeIdx] + "' for reflector '" + reflName + "'.");
                                reflSolver.ApplySelfDF(dfNodes.back(), side);
                            }
                            ++nodeIdx;
                        }
                        for (auto& node : dfNodes)
                            avgNodes.push_back(&node);
                        models.push_back(AverageReflectorModels(reflName, avgNodes));
                        continue;
                    }

                    // "left neighbor"/"right neighbor" describe the *outer* face of
                    // a paired axial reflector -- what lies beyond it, or that
                    // nothing does.  Nothing implements them: ApplyDF/ApplyDF2 take
                    // the two Model* and the two BOUNDARY values and immediately
                    // (void)-cast all four away, so the outer face is handled by
                    // ReflectorSolver::Usable()'s value window and by nothing else.
                    // Accepting the fields makes the input file claim a boundary
                    // treatment the library does not have, and every reader of that
                    // file -- human or script -- is entitled to believe it.
                    //
                    // Wiring them up is a numerical change that has not been
                    // validated, so this refuses the configuration instead.  Delete
                    // the fields: the library the input produces is unchanged.
                    // Verified on
                    // test/CrossSections/2_i-SMR_Validation/2_i-SMR_Validation.json,
                    // whose RB/RT specs have carried them since the import --
                    // rebuilt with and without, h5diff -c reports 0 differences.
                    for (const char* dead : {"left neighbor", "right neighbor"}) {
                        if (!specs.contains(dead)) continue;
                        throw std::runtime_error(
                            std::string("CHIFFON input: reflector '") + reflName + "' sets '" +
                            dead + "', which is not implemented for reflector type '" + typeStr +
                            "'.  ReflectorSolver::ApplyDF/ApplyDF2 discard the neighbour and "
                            "its boundary condition, so the field changes nothing and only "
                            "misdescribes the library.  Remove it from the input.");
                    }

                    std::string leftNode  = specs.at("left node").get<std::string>();
                    std::string rightNode = specs.at("right node").get<std::string>();

                    Model&   lNode = FindReflectorNode(reflModels, leftNode);
                    Model&   rNode = FindReflectorNode(reflModels, rightNode);
                    Model*   lNeih = nullptr;
                    Model*   rNeih = nullptr;
                    BOUNDARY lBound;
                    BOUNDARY rBound;

                    ParseOptionalReflectorNeighbor(reflModels, specs, "left neighbor", lNeih, lBound);
                    ParseOptionalReflectorNeighbor(reflModels, specs, "right neighbor", rNeih, rBound);

                    if (typeStr == "axial bottom") {
                        reflSolver.ApplyDF2(lNode, rNode, lNeih, rNeih, lBound, rBound);
                        lNode._name = reflName;
                        models.push_back(lNode);
                    } else {
                        reflSolver.ApplyDF(lNode, rNode, lNeih, rNeih, lBound, rBound);
                        rNode._name = reflName;
                        models.push_back(rNode);
                    }
                }
            }
        }

        for (auto& model : models) {
            Interpolator::Interpolate(model);
        }
    }

#pragma region Load HDF file
    // Load all models from a CHIFFON HDF5 file (metadata, depletion points, branches, deltas)
    std::vector<Model> LoadHDF(const std::string& filename) {
        Hdf5Guard hdf5_guard;
        try {
            HighFive::File file(filename, HighFive::File::ReadOnly);

            // Load and validate metadata
            auto metaGroup = file.getGroup("Metadata");

            if (!metaGroup.exist("format"))
                throw std::runtime_error(
                    "missing CHIFFON HDF5 format marker.");
            std::string format;
            metaGroup.getDataSet("format").read(format);
            if (format != "CHIFFON_HDF5")
                throw std::runtime_error(
                    "Invalid HDF5 format: expected 'CHIFFON_HDF5', got '" +
                    format + "'");
            std::string version;
            if (!metaGroup.exist("version"))
                throw std::runtime_error(
                    "missing CHIFFON HDF5 version — regenerate the library.");
            metaGroup.getDataSet("version").read(version);
            if (version != HDF_VERSION)
                throw std::runtime_error(
                    "unsupported CHIFFON HDF5 version '" + version +
                    "'; expected " + HDF_VERSION +
                    " — regenerate the library.");

            // Load isotope registry.
            //
            // `niso` and `iidx` are process globals.  Read the library's copy
            // into locals, validate it, and only then publish it under the
            // registry lock: in --batch-mode several instances load their
            // libraries at the same time, and an unguarded clear()/emplace()
            // pair on the shared map corrupts the heap (the run then dies in
            // malloc() a fraction of a second after start).  Every library
            // that passes the validation below yields the same registry, so
            // the publish is idempotent and the lock is uncontended once the
            // first instance is through.
            std::vector<std::string> isoNames;
            std::vector<size_t>      isoIndices;
            metaGroup.getDataSet("isotope_names").read(isoNames);
            metaGroup.getDataSet("isotope_indices").read(isoIndices);
            size_t libraryNiso = 0;
            metaGroup.getDataSet("niso").read(libraryNiso);

            const size_t canonicalCount = isotopeIds.size();
            if (libraryNiso != canonicalCount ||
                isoNames.size() != canonicalCount ||
                isoIndices.size() != canonicalCount)
                throw std::runtime_error(
                    "Isotope registry does not match the schema-3 basis.");

            std::vector<bool> seen(canonicalCount, false);
            for (size_t i = 0; i < canonicalCount; ++i) {
                const size_t index = isoIndices[i];
                if (index >= canonicalCount || seen[index])
                    throw std::runtime_error(
                        "Isotope registry contains an invalid or duplicate index.");
                if (isoNames[i] != isotopeIds[index])
                    throw std::runtime_error(
                        "Isotope registry name/index mapping is not canonical.");
                seen[index] = true;
            }
            {
                std::lock_guard<std::mutex> guard(registryMutex);
                if (niso != canonicalCount || iidx.size() != canonicalCount)
                    RebuildIndexLocked();
            }

            size_t numModels;
            metaGroup.getDataSet("num_models").read(numModels);

            std::vector<Model> models;
            models.reserve(numModels);

            auto modelsGroup = file.getGroup("Models");

            for (size_t m = 0; m < numModels; ++m) {
                auto  modelGroup = modelsGroup.getGroup("Model_" + std::to_string(m));
                Model model;

                modelGroup.getDataSet("id").read(model._id);
                modelGroup.getDataSet("name").read(model._name);
                if (modelGroup.exist("history_partner"))
                    modelGroup.getDataSet("history_partner").read(model.history_partner());

                size_t numDpts;
                modelGroup.getDataSet("num_dpts").read(numDpts);

                // Runtime libraries use the compact flat layout.
                auto dptsGroup = modelGroup.getGroup("DepletionPoints");
                if (!dptsGroup.exist("layout"))
                    throw std::runtime_error(
                        "Unsupported DepletionPoints layout; regenerate the library.");
                model._dpts = LoadFlatDepletionPoints(dptsGroup, numDpts);

                LoadBranch(
                    modelGroup, "refr_dpts", model._refr_dpts,
                    model._dpts);
                // Branch dpts are only needed for fitting BranchDelta polynomials.
                // Since BranchDelta is already stored in HDF5 and loaded below,
                // skip loading branch dpts to save I/O time and memory.

                LoadBranchDelta(modelGroup, "bppm_delt", model._bppm_delt);
                LoadBranchDelta(modelGroup, "tful_delt", model._tful_delt);
                LoadBranchDelta(modelGroup, "dmod_delt", model._dmod_delt);
                if (modelGroup.exist("rod_depletion_delt"))
                    LoadBranchDelta(modelGroup, "rod_depletion_delt", model._rod_depletion_delt);
                if (modelGroup.exist("rod_depletion_branch_bppm"))
                    LoadBranchDelta(modelGroup, "rod_depletion_branch_bppm", model._rod_depletion_branch[0]);
                if (modelGroup.exist("rod_depletion_branch_tful"))
                    LoadBranchDelta(modelGroup, "rod_depletion_branch_tful", model._rod_depletion_branch[1]);
                if (modelGroup.exist("rod_depletion_branch_dmod"))
                    LoadBranchDelta(modelGroup, "rod_depletion_branch_dmod", model._rod_depletion_branch[2]);
                LoadSpectralHistory(modelGroup, model._spectral_history);

                models.push_back(std::move(model));
            }

            return models;

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load HDF5 file '" + filename + "': " + e.what());
        }
    }

private:
    struct FlatFieldData {
        std::vector<size_t> offsets = {0};
        std::vector<double> values;
    };

    FlatFieldData LoadFlatFieldData(
        const HighFive::Group& parent, const std::string& field_name,
        size_t slotCount) {
        if (slotCount == std::numeric_limits<size_t>::max())
            throw std::runtime_error(
                "corrupt flat field '" + field_name + "'.");
        FlatFieldData field;
        parent.getDataSet(field_name + "_offsets").read(field.offsets);
        parent.getDataSet(field_name + "_values").read(field.values);
        if (field.offsets.size() != slotCount + 1 ||
            field.offsets.front() != 0 ||
            field.offsets.back() != field.values.size())
            throw std::runtime_error(
                "corrupt flat field '" + field_name + "'.");
        for (size_t i = 1; i < field.offsets.size(); ++i)
            if (field.offsets[i] < field.offsets[i - 1])
                throw std::runtime_error(
                    "corrupt flat field '" + field_name + "'.");
        return field;
    }

    // Corrupt flat-field offsets fail loudly at load time (LoadHDF wraps with the filename)
    // instead of leaving zero-filled physics state behind.
    static void CheckFlatFieldSlot(const FlatFieldData& field, size_t slot) {
        if (slot + 1 >= field.offsets.size() ||
            field.offsets[slot] > field.offsets[slot + 1] ||
            field.offsets[slot + 1] > field.values.size())
            throw std::runtime_error("corrupt flat field (slot " + std::to_string(slot) + ").");
    }

    static void AssignFlatField(std::vector<double>& dst, const FlatFieldData& field, size_t slot) {
        CheckFlatFieldSlot(field, slot);
        dst.assign(field.values.begin() + static_cast<ptrdiff_t>(field.offsets[slot]),
                   field.values.begin() + static_cast<ptrdiff_t>(field.offsets[slot + 1]));
    }

    static void AssignFlatField(milk::Vector<double>& dst, const FlatFieldData& field, size_t slot) {
        CheckFlatFieldSlot(field, slot);
        const size_t start = field.offsets[slot];
        const size_t end   = field.offsets[slot + 1];
        if (start == end) {
            dst.clear();
            return;
        }

        std::vector<double> tmp(field.values.begin() + start, field.values.begin() + end);
        dst.fromVector(tmp);
    }

    std::vector<DepletionPoint> LoadFlatDepletionPoints(const HighFive::Group& parent, size_t numDpts) {
        std::string         layout;
        std::vector<int>    meta;
        std::vector<double> data;
        std::vector<size_t> heavy_indices;

        parent.getDataSet("layout").read(layout);
        if (layout != "flat_v1") {
            throw std::runtime_error("Unsupported depletion layout: '" + layout + "'");
        }

        parent.getDataSet("meta").read(meta);
        parent.getDataSet("data").read(data);
        std::vector<double> rod_fluence;
        parent.getDataSet("rod_fluence").read(rod_fluence);
        parent.getDataSet("heavy_indices").read(heavy_indices);

        if (numDpts > std::numeric_limits<size_t>::max() / 4 ||
            numDpts > std::numeric_limits<size_t>::max() / AD_SIZE)
            throw std::runtime_error("Invalid flat depletion point count");
        if (meta.size() != numDpts * 4)
            throw std::runtime_error("Invalid flat depletion metadata size");
        if (data.size() != numDpts * AD_SIZE)
            throw std::runtime_error("Invalid flat depletion data size");
        if (rod_fluence.size() != numDpts)
            throw std::runtime_error("Invalid rod-fluence vector size");

        std::vector<DepletionPoint> dpts;
        dpts.reserve(numDpts);
        for (size_t idx = 0; idx < numDpts; ++idx) {
            const size_t meta_off = idx * 4;
            const int    ngrp     = meta[meta_off + 0];
            const int    npin     = meta[meta_off + 1];
            const int    rawType  = meta[meta_off + 2];
            const int    ctype    = meta[meta_off + 3];
            const bool validType =
                rawType == BRANCHTYPE::REFR ||
                rawType == BRANCHTYPE::BPPM ||
                rawType == BRANCHTYPE::TFUL ||
                rawType == BRANCHTYPE::DMOD ||
                rawType == BRANCHTYPE::TMOD ||
                rawType == BRANCHTYPE::SPECTRAL_HISTORY ||
                rawType == BRANCHTYPE::ROD_HISTORY ||
                rawType == BRANCHTYPE::COMBO;
            if (ngrp <= 0 || npin <= 0 || ctype < 0 || !validType)
                throw std::runtime_error(
                    "Invalid flat depletion point metadata.");

            DepletionPoint dpt(
                ngrp, npin, static_cast<BRANCHTYPE>(rawType), ctype);

            const size_t data_off = idx * AD_SIZE;
            for (size_t i = 0; i < AD_SIZE; ++i)
                dpt._data[i] = data[data_off + i];
            dpt._rod_fluence = rod_fluence[idx];

            dpts.push_back(std::move(dpt));
        }

        std::vector<bool> heavySeen(numDpts, false);
        size_t            referenceCount = 0;
        for (const auto& dpt : dpts)
            if (dpt._btyp == BRANCHTYPE::REFR)
                ++referenceCount;
        if (heavy_indices.size() != referenceCount)
            throw std::runtime_error(
                "Reference depletion point payload is incomplete.");
        for (const size_t index : heavy_indices) {
            if (index >= numDpts || heavySeen[index] ||
                dpts[index]._btyp != BRANCHTYPE::REFR)
                throw std::runtime_error(
                    "Invalid or duplicate reference depletion point index.");
            heavySeen[index] = true;
        }
        for (size_t i = 0; i < numDpts; ++i)
            if (dpts[i]._btyp == BRANCHTYPE::REFR && !heavySeen[i])
                throw std::runtime_error(
                    "Reference depletion point payload is incomplete.");

        const size_t heavyCount = heavy_indices.size();
        const FlatFieldData aflx =
            LoadFlatFieldData(parent, "aflx", heavyCount);
        const FlatFieldData gmap =
            LoadFlatFieldData(parent, "gmap", heavyCount);
        const FlatFieldData fmap =
            LoadFlatFieldData(parent, "fmap", heavyCount);
        const FlatFieldData chix =
            LoadFlatFieldData(parent, "chix", heavyCount);
        const FlatFieldData sdfa =
            LoadFlatFieldData(parent, "sdfa", heavyCount);
        const FlatFieldData pdfa =
            LoadFlatFieldData(parent, "pdfa", heavyCount);
        const FlatFieldData iden =
            LoadFlatFieldData(parent, "iden", heavyCount);

        std::vector<size_t> xs_meta;
        std::vector<double> xs_macx, xs_lmpx, xs_micx;
        parent.getDataSet("xs_meta").read(xs_meta);
        parent.getDataSet("xs_macx").read(xs_macx);
        parent.getDataSet("xs_lmpx").read(xs_lmpx);
        parent.getDataSet("xs_micx").read(xs_micx);
        if (xs_meta.size() != heavyCount * 3)
            throw std::runtime_error(
                "Invalid reference cross-section metadata size.");

        size_t macx_offset = 0;
        size_t lmpx_offset = 0;
        size_t micx_offset = 0;

        for (size_t slot = 0; slot < heavy_indices.size(); ++slot) {
            const size_t dpt_index = heavy_indices[slot];
            if (dpt_index >= dpts.size())
                throw std::runtime_error("corrupt heavy_indices (slot " + std::to_string(slot) + ").");

            auto& dpt = dpts[dpt_index];
            AssignFlatField(dpt._aflx, aflx, slot);
            AssignFlatField(dpt._gmap, gmap, slot);
            AssignFlatField(dpt._fmap, fmap, slot);
            AssignFlatField(dpt._chix, chix, slot);
            AssignFlatField(dpt._sdfa, sdfa, slot);
            AssignFlatField(dpt._pdfa, pdfa, slot);
            AssignFlatField(dpt._iden, iden, slot);

            const size_t ngrp = static_cast<size_t>(dpt._ngrp);
            const size_t npin = static_cast<size_t>(dpt._npin);
            if (npin > std::numeric_limits<size_t>::max() / npin)
                throw std::runtime_error(
                    "Invalid reference depletion point dimensions.");
            const size_t pinCount = npin * npin;
            if (ngrp > std::numeric_limits<size_t>::max() / pinCount ||
                ngrp > std::numeric_limits<size_t>::max() / 4)
                throw std::runtime_error(
                    "Invalid reference depletion point dimensions.");
            if (dpt._aflx.size() != ngrp ||
                dpt._gmap.size() != pinCount ||
                dpt._fmap.size() != ngrp * pinCount ||
                dpt._chix.size() != ngrp ||
                dpt._sdfa.size() != ngrp * 4 ||
                dpt._pdfa.size() != ngrp * 4 ||
                dpt._iden.size() != niso)
                throw std::runtime_error(
                    "Invalid reference depletion point field dimensions.");

            const size_t xs_meta_off = slot * 3;
            const size_t xsNgrp = xs_meta[xs_meta_off + 0];
            const size_t ndat = xs_meta[xs_meta_off + 1];
            const size_t nmem = xs_meta[xs_meta_off + 2];
            if (ngrp > std::numeric_limits<size_t>::max() -
                           N_XS_SCALAR)
                throw std::runtime_error(
                    "Invalid reference cross-section dimensions.");
            const size_t expectedNdat = N_XS_SCALAR + ngrp;
            if (ngrp > std::numeric_limits<size_t>::max() /
                           expectedNdat ||
                xsNgrp != ngrp || ndat != expectedNdat ||
                nmem != ngrp * expectedNdat)
                throw std::runtime_error(
                    "Invalid reference cross-section dimensions.");
            if (niso != 0 &&
                nmem > std::numeric_limits<size_t>::max() / niso)
                throw std::runtime_error(
                    "Reference microscopic cross-section size overflow.");

            dpt._xs.deallocate();
            dpt._xs._ngrp = xsNgrp;
            dpt._xs._ndat = ndat;
            dpt._xs._nmem = nmem;

            Parser::LoadVector(dpt._xs._macx, xs_macx, macx_offset, nmem);
            Parser::LoadVector(dpt._xs._lmpx, xs_lmpx, lmpx_offset, nmem);
            Parser::LoadVector(dpt._xs._micx, xs_micx, micx_offset, niso * nmem);
        }
        if (macx_offset != xs_macx.size() ||
            lmpx_offset != xs_lmpx.size() ||
            micx_offset != xs_micx.size())
            throw std::runtime_error(
                "Trailing reference cross-section payload data.");

        return dpts;
    }

    void LoadBranch(const HighFive::Group& parent, const std::string& name,
                    Reference& ref,
                    const std::vector<DepletionPoint>& points) {
        auto                group = parent.getGroup(name);
        std::vector<int>    cTypes, burnups;
        std::vector<size_t> indices;

        group.getDataSet("cTypes").read(cTypes);
        group.getDataSet("burnups").read(burnups);
        group.getDataSet("indices").read(indices);
        if (cTypes.size() != burnups.size() ||
            cTypes.size() != indices.size())
            throw std::runtime_error(
                name + ": reference key count mismatch.");

        ref.clear();
        for (size_t i = 0; i < cTypes.size(); ++i) {
            if (indices[i] >= points.size())
                throw std::runtime_error(
                    name + ": reference point index is out of range.");
            const DepletionPoint& point = points[indices[i]];
            if (point._btyp != BRANCHTYPE::REFR ||
                point._ctyp != cTypes[i] ||
                point.burnKey() != burnups[i])
                throw std::runtime_error(
                    name + ": reference point metadata does not match its key.");
            auto& burnupMap = ref[cTypes[i]];
            const auto [entry, inserted] =
                burnupMap.emplace(burnups[i], indices[i]);
            (void)entry;
            if (!inserted)
                throw std::runtime_error(
                    name + ": duplicate reference key.");
        }
    }

    std::vector<DeltaCrossSection> LoadDeltaPayload(
        const HighFive::Group& group, const std::string& name) {
        size_t entryCount = 0;
        group.getDataSet("num_entries").read(entryCount);
        if (entryCount == 0)
            return {};

        std::vector<size_t> dxsMeta;
        std::vector<size_t> xsMeta;
        std::vector<size_t> knotOffsets;
        std::vector<double> knots;
        std::vector<double> macroscopic;
        std::vector<double> lumped;
        std::vector<double> microscopic;
        group.getDataSet("dxs_meta").read(dxsMeta);
        group.getDataSet("knots_flat").read(knots);
        group.getDataSet("knots_offsets").read(knotOffsets);
        group.getDataSet("xs_meta").read(xsMeta);
        group.getDataSet("all_macx").read(macroscopic);
        group.getDataSet("all_lmpx").read(lumped);
        group.getDataSet("all_micx").read(microscopic);

        if (entryCount > std::numeric_limits<size_t>::max() / 4 ||
            entryCount == std::numeric_limits<size_t>::max() ||
            dxsMeta.size() != entryCount * 4 ||
            knotOffsets.size() != entryCount + 1 ||
            knotOffsets.front() != 0 || knotOffsets.back() != knots.size() ||
            xsMeta.size() % 3 != 0) {
            throw std::runtime_error(name + ": invalid delta metadata.");
        }

        std::vector<DeltaCrossSection> deltas;
        deltas.reserve(entryCount);
        size_t xsMetaIndex       = 0;
        size_t macroscopicOffset = 0;
        size_t lumpedOffset      = 0;
        size_t microscopicOffset = 0;
        for (size_t i = 0; i < entryCount; ++i) {
            const size_t ngrp             = dxsMeta[i * 4 + 0];
            const size_t order            = dxsMeta[i * 4 + 1];
            const size_t rawMode          = dxsMeta[i * 4 + 2];
            const size_t coefficientCount = dxsMeta[i * 4 + 3];
            if (ngrp == 0 || order == 0 ||
                ngrp > std::numeric_limits<size_t>::max() -
                           N_XS_SCALAR)
                throw std::runtime_error(
                    name + ": delta dimensions must be positive.");
            if (rawMode > static_cast<size_t>(SPLINE_MODE))
                throw std::runtime_error(name + ": unknown interpolation mode.");

            const size_t knotBegin = knotOffsets[i];
            const size_t knotEnd   = knotOffsets[i + 1];
            if (knotBegin > knotEnd || knotEnd > knots.size())
                throw std::runtime_error(name + ": invalid knot offsets.");
            std::vector<double> entryKnots(
                knots.begin() + static_cast<ptrdiff_t>(knotBegin),
                knots.begin() + static_cast<ptrdiff_t>(knotEnd));

            const auto mode = static_cast<InterpolMode>(rawMode);
            if (mode == SPLINE_MODE) {
                if (coefficientCount == 0 ||
                    order % coefficientCount != 0 ||
                    entryKnots.size() != order / coefficientCount + 1)
                    throw std::runtime_error(
                        name + ": invalid spline dimensions.");
                for (size_t knot = 0; knot < entryKnots.size(); ++knot) {
                    if (!std::isfinite(entryKnots[knot]) ||
                        (knot > 0 &&
                         entryKnots[knot] <= entryKnots[knot - 1]))
                        throw std::runtime_error(
                            name + ": spline knots must be finite and increasing.");
                }
            } else if (!entryKnots.empty()) {
                throw std::runtime_error(
                    name + ": polynomial delta must not contain knots.");
            }

            DeltaCrossSection delta;
            if (mode == SPLINE_MODE)
                delta = DeltaCrossSection(
                    ngrp, order, coefficientCount, entryKnots);
            else {
                delta        = DeltaCrossSection(ngrp, order);
                delta._knots = std::move(entryKnots);
            }

            const size_t xsEntryCount = xsMeta.size() / 3;
            if (xsMetaIndex > xsEntryCount ||
                order > xsEntryCount - xsMetaIndex)
                throw std::runtime_error(
                    name + ": incomplete cross-section metadata.");

            for (size_t coefficient = 0; coefficient < order;
                 ++coefficient, ++xsMetaIndex) {
                CrossSection& xs             = delta._u[coefficient];
                const size_t metadataOffset  = xsMetaIndex * 3;
                const size_t xsNgrp          = xsMeta[metadataOffset + 0];
                const size_t xsNdat          = xsMeta[metadataOffset + 1];
                const size_t xsNmem          = xsMeta[metadataOffset + 2];
                const size_t expectedNdat    = N_XS_SCALAR + ngrp;
                if (xsNgrp != ngrp ||
                    xsNdat != expectedNdat ||
                    xsNgrp > std::numeric_limits<size_t>::max() /
                                  expectedNdat ||
                    xsNmem != xsNgrp * expectedNdat)
                    throw std::runtime_error(
                        name + ": inconsistent cross-section dimensions.");
                if (niso != 0 &&
                    xsNmem > std::numeric_limits<size_t>::max() / niso)
                    throw std::runtime_error(
                        name + ": microscopic cross-section size overflow.");

                xs.deallocate();
                xs._ngrp = xsNgrp;
                xs._ndat = xsNdat;
                xs._nmem = xsNmem;
                Parser::LoadVector(
                    xs._macx, macroscopic, macroscopicOffset, xsNmem);
                Parser::LoadVector(
                    xs._lmpx, lumped, lumpedOffset, xsNmem);
                Parser::LoadVector(
                    xs._micx, microscopic, microscopicOffset,
                    niso * xsNmem);
            }
            deltas.push_back(std::move(delta));
        }

        if (xsMetaIndex * 3 != xsMeta.size() ||
            macroscopicOffset != macroscopic.size() ||
            lumpedOffset != lumped.size() ||
            microscopicOffset != microscopic.size()) {
            throw std::runtime_error(name + ": trailing delta payload data.");
        }
        return deltas;
    }

    void LoadBranchDelta(
        const HighFive::Group& parent, const std::string& name,
        BranchDelta& delta) {
        auto             group = parent.getGroup(name);
        std::vector<int> cTypes;
        std::vector<int> burnups;
        group.getDataSet("cTypes").read(cTypes);
        group.getDataSet("burnups").read(burnups);
        auto entries = LoadDeltaPayload(group, name);
        if (cTypes.size() != entries.size() ||
            burnups.size() != entries.size())
            throw std::runtime_error(name + ": branch key count mismatch.");

        delta.clear();
        for (size_t i = 0; i < entries.size(); ++i) {
            auto& burnupMap = delta[cTypes[i]];
            const auto [entry, inserted] =
                burnupMap.emplace(burnups[i], std::move(entries[i]));
            (void)entry;
            if (!inserted)
                throw std::runtime_error(name + ": duplicate branch key.");
        }
    }

    void LoadBurnupDelta(
        const HighFive::Group& parent, const std::string& name,
        BurnupDelta& delta) {
        auto             group = parent.getGroup(name);
        std::vector<int> burnups;
        group.getDataSet("burnups").read(burnups);
        auto entries = LoadDeltaPayload(group, name);
        if (burnups.size() != entries.size())
            throw std::runtime_error(name + ": burnup key count mismatch.");

        delta.clear();
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto [entry, inserted] =
                delta.emplace(burnups[i], std::move(entries[i]));
            (void)entry;
            if (!inserted)
                throw std::runtime_error(name + ": duplicate burnup key.");
        }
    }

    void LoadSpectralHistory(
        const HighFive::Group& parent,
        std::vector<SpectralHistoryCorrection>& corrections) {
        auto group = parent.getGroup("spectral_history");

        size_t              termCount = 0;
        std::vector<size_t> isotopes;
        std::vector<int>    coordinates;
        group.getDataSet("num_terms").read(termCount);
        group.getDataSet("isotopes").read(isotopes);
        group.getDataSet("coordinates").read(coordinates);
        std::vector<size_t> partners(termCount, static_cast<size_t>(-1));
        if (group.exist("partners"))
            group.getDataSet("partners").read(partners);
        if (partners.size() != termCount)
            throw std::runtime_error("spectral_history: partners mismatch.");
        std::vector<int> rodScaled(termCount, 0);
        if (group.exist("rod_scaled"))
            group.getDataSet("rod_scaled").read(rodScaled);
        if (rodScaled.size() != termCount)
            throw std::runtime_error("spectral_history: rod_scaled mismatch.");
        if (isotopes.size() != termCount || coordinates.size() != termCount)
            throw std::runtime_error(
                "spectral_history: term metadata size mismatch.");

        corrections.clear();
        corrections.reserve(termCount);
        for (size_t i = 0; i < termCount; ++i) {
            if (isotopes[i] >= niso)
                throw std::runtime_error(
                    "spectral_history: isotope index is out of range.");
            if (partners[i] != static_cast<size_t>(-1) && partners[i] >= niso)
                throw std::runtime_error(
                    "spectral_history: partner index is out of range.");
            if (!IsKnownSpectralCoordinate(coordinates[i])) {
                throw std::runtime_error(
                    "spectral_history: unknown spectral coordinate.");
            }

            SpectralHistoryCorrection correction;
            correction.term = SpectralTerm{
                isotopes[i],
                static_cast<SpectralCoordinate>(coordinates[i]),
                partners[i]};
            correction.rod_scaled = rodScaled[i] != 0;
            LoadBurnupDelta(
                group, "term_" + std::to_string(i), correction.delta);
            corrections.push_back(std::move(correction));
        }
    }

#pragma endregion

#pragma region Read HGC file
private:
    // Maps HGC isotope IDs to internal IDs (metastable remapping, Gd lumping to virtual "640000")
    const std::unordered_map<std::string, std::string> isochange = {
        {"953420", "952421"},
        {"606470", "601470"},
        {"606480", "601480"},
        {"606490", "601490"},
        {"616470", "611470"},
        {"616480", "611480"},
        {"617480", "611481"},
        {"616490", "611490"},
        {"626470", "621470"},
        {"626480", "621480"},
        {"626490", "621490"},
        {"536350", "531350"},
        {"546350", "541350"},
        // Map both supported Gd encodings to the lumped Gd isotope.
        {"641520", "640000"},
        {"641540", "640000"},
        {"641550", "640000"},
        {"641560", "640000"},
        {"641570", "640000"},
        {"641580", "640000"},
        {"641590", "640000"},
        {"641600", "640000"},
        {"646520", "640000"},
        {"646540", "640000"},
        {"646550", "640000"},
        {"646560", "640000"},
        {"646570", "640000"},
        {"646580", "640000"},
        {"646590", "640000"},
        {"646600", "640000"}
    };

    // Remaining-capture weights define the effective lumped Gd density.
    const std::unordered_map<std::string, double> gdWeight = {
        {"641520", 0.0},
        {"641540", 5.0},
        {"641550", 4.0},
        {"641560", 3.0},
        {"641570", 2.0},
        {"641580", 1.0},
        {"641590", 0.0},
        {"641600", 0.0},
        {"646520", 0.0},
        {"646540", 5.0},
        {"646550", 4.0},
        {"646560", 3.0},
        {"646570", 2.0},
        {"646580", 1.0},
        {"646590", 0.0},
        {"646600", 0.0}
    };

public:
    // Block-wise HGC file parser: splits on '%' delimiters and dispatches TITL/FLUX/DIST/MACX/MICX/ADFT blocks
    /// @brief Read one HGC into a model.
    /// @param invertCtype Invert the deck's rod tag. A rodded-depletion deck
    /// marks its own rod-in reference as ctype 0 and its CR withdrawal branch as
    /// ctype 1; inverting makes ctype absolute (0 = rod out) across models. The
    /// inversion must happen here, before AddDepletionPoint keys the branch
    /// index maps by ctype.
    void ReadHGC(Model& model, const std::string& fileName,
                 bool invertCtype = false) {
        // Check file size
        std::ifstream file(fileName, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Failed to open HGC file: " + fileName);

        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Load entire file onto the buffer, +1 for null terminator safety
        std::vector<char> buffer(fileSize + 1);
        if (!file.read(buffer.data(), fileSize))
            throw std::runtime_error("Failed to read HGC file: " + fileName);
        buffer[fileSize] = '\0';

        const char* fileStart = buffer.data();
        const char* fileEnd   = fileStart + fileSize;
        const char* cur       = fileStart;

        DepletionPoint* dpnt = nullptr;

        // find '%' character, block-wise treatment
        while (cur < fileEnd) {
            // find next '%'
            const char* nextDelim = static_cast<const char*>(memchr(cur, '%', fileEnd - cur));
            const char* blockEnd  = (nextDelim) ? nextDelim : fileEnd;

            if (blockEnd > cur) {
                Parser::FastParser blockStream(cur, blockEnd - cur);

                std::string blockType;
                blockStream >> blockType;

                if (!blockType.empty()) {
                    if (blockType == "TITL")
                        dpnt = &ReadTitl(model, blockStream, invertCtype);
                    else if (dpnt) {
                        if (blockType == "FLUX")
                            ReadFlux(*dpnt, blockStream);
                        else if (blockType == "DIST")
                            ReadDist(*dpnt, blockStream);
                        else if (blockType == "MACX")
                            ReadMacx(*dpnt, blockStream);
                        else if (blockType == "MICX")
                            ReadMicx(*dpnt, blockStream);
                        else if (blockType == "FYLD")
                            ReadFyld(*dpnt, blockStream);
                        else if (blockType == "XSMN")
                            ReadXsmn(*dpnt, blockStream);
                        else if (blockType == "ADFT")
                            ReadAdft(*dpnt, blockStream);
                    }
                }
            }

            // skip
            cur = blockEnd;
            if (cur < fileEnd && *cur == '%') cur++;
        }

        // The HGC files used here carry the fixed two-group fission spectrum.
        for (auto& dpt : model._dpts) {
            for (int ig = 0; ig < dpt._ngrp; ++ig)
                dpt._chix[ig] = ig == 0 ? 1.0 : 0.0;
        }

        // Ensure lmpx is computed for every depletion point (lmpx = macx - sum(micx*iden))
        for (auto& dpt : model._dpts) {
            if (dpt._iden.size() > 0)
                dpt._xs.CalcLmpx(dpt._iden);
        }
    }

private:
    // Parse depletion point header: branch type, control rod state, burnup, temperatures, geometry
    DepletionPoint& ReadTitl(Model& model, Parser::FastParser& blockData,
                             bool invertCtype = false) {
        std::string _line;
        std::string _case;

        blockData.getline(_line); // Remaining title line
        blockData.getline(_line); // Title separator
        blockData.getline(_line); // Empty line
        blockData.getline(_case);

        auto hasShortTmodToken = [](const std::string& s) {
            for (size_t i = 0; i + 1 < s.size(); ++i) {
                const bool boundary = (i == 0) || s[i - 1] == '_' ||
                                      std::isspace(static_cast<unsigned char>(s[i - 1]));
                if (boundary && s[i] == 'T' &&
                    std::isdigit(static_cast<unsigned char>(s[i + 1])))
                    return true;
            }
            return false;
        };

        BRANCHTYPE bType = BRANCHTYPE::REFR;
        if (_case.find("COMBO") != std::string::npos)
            bType = BRANCHTYPE::COMBO; // multi-axis block: usable as validation only, never a fit point
        else if (_case.find("BOR") != std::string::npos)
            bType = BRANCHTYPE::BPPM;
        else if (_case.find("TFUEL") != std::string::npos)
            bType = BRANCHTYPE::TFUL;
        else if (_case.find("DMOD") != std::string::npos)
            bType = BRANCHTYPE::DMOD;
        else if (_case.find("TMOD") != std::string::npos || hasShortTmodToken(_case))
            bType = BRANCHTYPE::TMOD;
        else
            bType = BRANCHTYPE::REFR;

        // Rod-state markers identify the changed state. Rodded-history import
        // later inverts this tag so the base is rod-in and the branch is rod-out.
        int cType = 0;
        if (_case.find("ROD_IN") != std::string::npos || _case.find("ROD_OUT") != std::string::npos) {
            cType = 1;
        } else if (size_t _crstringIdx = _case.find("CR"); _crstringIdx != std::string::npos) {
            size_t digitBegin = _crstringIdx + 2;
            size_t digitEnd   = digitBegin;
            while (digitEnd < _case.size() &&
                   std::isdigit(static_cast<unsigned char>(_case[digitEnd])))
                ++digitEnd;
            if (digitEnd > digitBegin)
                cType = stoi(_case.substr(digitBegin, digitEnd - digitBegin));
        }
        if (invertCtype)
            cType = cType == 0 ? 1 : 0;

        std::string _ignore;
        int         ngrp = 0, npin = 0;
        double      apit = 0.0, ppit = 0.0, volu = 0.0, pden = 0.0, burn = 0.0;

        blockData >> ngrp >> _ignore >> npin >> apit >> ppit >> volu >> pden >> burn;
        const int burnup = BurnKeyFromDouble(burn);

        DepletionPoint& dpnt = model.AddDepletionPoint(ngrp, npin, burnup, bType, cType);
        blockData >>
            dpnt._data[AD_KEFF] >> dpnt._data[AD_KINF] >> dpnt._data[AD_BSQU] >> dpnt._data[AD_TFUL] >> dpnt._data[AD_TMOD] >>
            dpnt._data[AD_BPPM] >> dpnt._data[AD_VFRA] >> dpnt._data[AD_PRES] >> dpnt._data[AD_DMOD] >> dpnt._data[AD_HMAS];
        dpnt._data[AD_APIT] = apit;
        dpnt._data[AD_PPIT] = ppit;
        dpnt._data[AD_VOLU] = volu;
        dpnt._data[AD_PDEN] = pden;
        dpnt._data[AD_BURN] = burn;

        return dpnt;
    }

    // Read average flux using the production 12-value auxiliary stride.
    // Fit and runtime fluence definitions must be revalidated before changing it.
    void ReadFlux(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        for (int ig = 0; ig < dpnt._ngrp; ig++) {
            blockData >> dpnt._aflx[ig];
            // Skip sflx(4), cflx(4), and jnet(4).
            for (int i = 0; i < 12; i++)
                blockData.skipToken();
        }
    }

    // Read pin-wise power and flux maps, skipping unused pmap/rmap/bmap fields.
    void ReadDist(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        const int nij  = dpnt._npin * dpnt._npin;
        const int ngij = dpnt._ngrp * nij;
        for (int lk = 0; lk < nij; lk++)
            blockData >> dpnt._gmap[lk];
        for (int lk = 0; lk < nij; lk++)
            blockData.skipToken();
        // Skip rmap (ngij tokens, field removed)
        for (int lk = 0; lk < ngij; lk++)
            blockData.skipToken();
        for (int lk = 0; lk < ngij; lk++)
            blockData >> dpnt._fmap[lk];
        for (int lk = 0; lk < nij; lk++)
            blockData.skipToken();
    }

    // Read macroscopic XS per group (D, abs, fission, nu-fission, kappa-fission, scatter) + scattering matrix
    void ReadMacx(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        CrossSection& xs = dpnt._xs;
        for (int ig = 0; ig < dpnt._ngrp; ig++) {
            blockData >> xs.maxs(ig, XSDF) >> xs.maxs(ig, XSAF) >> xs.maxs(ig, XSFF);
            blockData >> xs.maxs(ig, XSNF) >> xs.maxs(ig, XSKF) >> xs.maxs(ig, XSSF);
            xs.maxs(ig, XSTF) = 0.333333333 / (xs.maxs(ig, XSDF));
        }
        for (int igs = 0; igs < dpnt._ngrp; igs++) {
            for (int ige = 0; ige < dpnt._ngrp; ige++) {
                blockData >> xs.maxssm(igs, ige);
            }
        }
    }

    // Read microscopic XS and isotope densities; Gd isotopes are lumped into virtual "640000"
    void ReadMicx(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        CrossSection& xs = dpnt._xs;
        std::string   nid, rid;

        // Track whether any Gd isotope was read, for final normalization
        bool gdFound = false;

        while (true) {
            nid.clear();
            blockData >> nid;
            if (nid.empty()) break;
            blockData >> rid;
            std::string rawIso = nid + rid;
            std::string iso    = rawIso;

            if (isochange.contains(iso)) {
                iso = isochange.at(iso);
            }
            if (!iidx.contains(iso)) {
                int n = dpnt._ngrp * dpnt._ngrp + dpnt._ngrp * 6 + 1;
                for (int i = 0; i < n; i++) {
                    blockData.skipToken();
                }
                continue;
            };

            // Gd lumping: accumulate N*sigma into "640000" slot.
            if (iso == "640000") {
                gdFound = true;
                double nden;
                blockData >> nden;

                double weight = gdWeight.contains(rawIso) ? gdWeight.at(rawIso) : 0.0;
                if (std::getenv("RASBERY_GD_DEBUG"))
                    std::fprintf(stderr, "[GDDBG] raw=%s nden=%.4E w=%.1f before=%.4E\n",
                                 rawIso.c_str(), nden, weight, dpnt.iden("640000"));
                // N_eff += w_i * N_i
                dpnt.iden("640000") += weight * nden;

                // mixs("640000", ig, xt) += N_i * sigma_i (numerator accumulation)
                for (int ig = 0; ig < dpnt._ngrp; ig++) {
                    double xstf = 0.0, xsaf = 0.0, xsff = 0.0;
                    double xsnf = 0.0, xskf = 0.0, xssf = 0.0;
                    blockData >> xstf >> xsaf >> xsff >> xsnf >> xskf >> xssf;
                    xs.mixs("640000", ig, XSTF) += nden * xstf;
                    xs.mixs("640000", ig, XSAF) += nden * xsaf;
                    xs.mixs("640000", ig, XSFF) += nden * xsff;
                    xs.mixs("640000", ig, XSNF) += nden * xsnf;
                    xs.mixs("640000", ig, XSKF) += nden * xskf;
                    xs.mixs("640000", ig, XSSF) += nden * xssf;
                }

                for (int igs = 0; igs < dpnt._ngrp; igs++) {
                    for (int ige = 0; ige < dpnt._ngrp; ige++) {
                        double sm = 0.0;
                        blockData >> sm;
                        xs.mixssm("640000", igs, ige) += nden * sm;
                    }
                }
            }
            // Normal isotope
            else {
                blockData >> dpnt.iden(iso);

                for (int ig = 0; ig < dpnt._ngrp; ig++) {
                    blockData >> xs.mixs(iso, ig, XSTF) >> xs.mixs(iso, ig, XSAF) >> xs.mixs(iso, ig, XSFF);
                    blockData >> xs.mixs(iso, ig, XSNF) >> xs.mixs(iso, ig, XSKF) >> xs.mixs(iso, ig, XSSF);
                }

                for (int igs = 0; igs < dpnt._ngrp; igs++) {
                    for (int ige = 0; ige < dpnt._ngrp; ige++) {
                        blockData >> xs.mixssm(iso, igs, ige);
                    }
                }
            }

            if (blockData.p >= blockData.end) break;
        }

        // Gd normalization: sigma_eff = sum(N_i * sigma_i) / N_eff
        if (gdFound) {
            double neff = dpnt.iden("640000");
            if (neff > 0.0) {
                double inv = 1.0 / neff;
                for (int ig = 0; ig < dpnt._ngrp; ig++) {
                    xs.mixs("640000", ig, XSTF) *= inv;
                    xs.mixs("640000", ig, XSAF) *= inv;
                    xs.mixs("640000", ig, XSFF) *= inv;
                    xs.mixs("640000", ig, XSNF) *= inv;
                    xs.mixs("640000", ig, XSKF) *= inv;
                    xs.mixs("640000", ig, XSSF) *= inv;
                }
                for (int igs = 0; igs < dpnt._ngrp; igs++) {
                    for (int ige = 0; ige < dpnt._ngrp; ige++) {
                        xs.mixssm("640000", igs, ige) *= inv;
                    }
                }
            }
        }
    }

    // Read fission yield data per isotope
    void ReadFyld(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        CrossSection& xs = dpnt._xs;
        std::string   iso;

        while (true) {
            iso.clear();
            blockData >> iso;
            if (iso.empty()) break;

            iso = iso + "0";
            if (isochange.contains(iso)) {
                iso = isochange.at(iso);
            }
            if (!iidx.contains(iso)) {
                for (int ityp = 0; ityp < 2; ityp++) {
                    blockData.skipToken();
                }
                continue;
            }
            double yield = 0.0;
            for (int ityp = 0; ityp < 2; ityp++) {
                blockData >> yield;
                if (yield > 1.0e-20) {
                    xs.mixs(iso, 0, FYLD) = yield;
                }
            }
            if (blockData.p >= blockData.end) break;
        }
    }

    // Read (n,2n) and (n,3n) microscopic cross-sections per isotope
    void ReadXsmn(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        CrossSection& xs = dpnt._xs;
        std::string   iso;

        while (true) {
            iso.clear();
            blockData >> iso;
            if (iso.empty()) break;

            iso = iso + "0";
            if (isochange.contains(iso)) {
                iso = isochange.at(iso);
            }
            if (!iidx.contains(iso)) {
                for (int ityp = 0; ityp < dpnt._ngrp * 2; ityp++) {
                    blockData.skipToken();
                }
                continue;
            }
            for (int ig = 0; ig < dpnt._ngrp; ig++) {
                blockData >> xs.mixs(iso, ig, XS2N) >> xs.mixs(iso, ig, XS3N);
            }
            if (blockData.p >= blockData.end) break;
        }
    }

    // Read surface/partial discontinuity factors and compute water volume fraction
    void ReadAdft(DepletionPoint& dpnt, Parser::FastParser& blockData) {
        for (int ig = 0; ig < dpnt._ngrp; ig++) {
            for (int is = 0; is < 4; is++)
                blockData >> dpnt._sdfa[ig * 4 + is];
        }
        for (int ig = 0; ig < dpnt._ngrp; ig++) {
            for (int is = 0; is < 4; is++)
                blockData >> dpnt._pdfa[ig * 4 + is];
        }
        // CalcLmpx is called once for all dpts after ReadHGC completes.

        // Compute water volume fraction: wvfr = N_H * M_H2O / (2 * dmod * N_A_barn)
        if (iidx.contains("10010") && dpnt._data[AD_DMOD] > 0.0) {
            double NH           = dpnt._iden[iidx.at("10010")];
            dpnt._data[AD_WVFR] = NH * 18.015 / (2.0 * dpnt._data[AD_DMOD] * 0.602214);
        }
    }
#pragma endregion
};
} // namespace Chiffon
