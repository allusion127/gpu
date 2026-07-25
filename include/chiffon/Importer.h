#pragma once

#include "Interpolator.h"
#include "Model.h"
#include "ReflectorSolver.h"
#include <cstdio>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

namespace Chiffon {

namespace ImporterDetail {

inline void LoadChainMatrix(const HighFive::Group& chainGroup, const std::string& name,
                            milk::Matrix<double>& mat) {
    if (!chainGroup.exist(name)) return;
    size_t rows, cols;
    chainGroup.getDataSet(name + "_rows").read(rows);
    chainGroup.getDataSet(name + "_cols").read(cols);
    std::vector<double> flat;
    chainGroup.getDataSet(name).read(flat);
    mat.assign(rows, cols, 0.0);
    std::copy(flat.begin(), flat.end(), mat.data());
}

inline void LoadFlatVector(milk::Vector<double>& dst, const std::vector<double>& src, size_t& offset, size_t count) {
    if (count == 0 || offset + count > src.size()) return;
    std::vector<double> tmp(src.begin() + offset, src.begin() + offset + count);
    dst.fromVector(tmp);
    offset += count;
}

} // namespace ImporterDetail

// Lightweight zero-copy text parser for memory-mapped HGC file blocks
class FastStream {
public:
    const char* p;
    const char* end;

    FastStream(const char* start, size_t len)
        : p(start), end(start + len) {
    }

    void skipWhitespace() {
        while (p < end && std::isspace(static_cast<unsigned char>(*p)))
            ++p;
    }

    FastStream& operator>>(std::string& out) {
        skipWhitespace();
        if (p >= end) return *this;
        const char* start = p;
        while (p < end && !std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        out.assign(start, p - start);
        return *this;
    }

    FastStream& operator>>(double& out) {
        skipWhitespace();
        if (p >= end) return *this;
        char* next;
        out = std::strtod(p, &next);
        p   = next;
        return *this;
    }

    FastStream& operator>>(int& out) {
        skipWhitespace();
        if (p >= end) return *this;
        char* next;
        out = static_cast<int>(std::strtol(p, &next, 10));
        p   = next;
        return *this;
    }

    bool getline(std::string& out) {
        if (p >= end) return false;
        const char* start = p;
        while (p < end && *p != '\n' && *p != '\r')
            ++p;
        out.assign(start, p - start);
        if (p < end && *p == '\r') ++p;
        if (p < end && *p == '\n') ++p;
        return true;
    }

    void skipToken() {
        skipWhitespace();
        if (p >= end) return;
        while (p < end && !std::isspace(static_cast<unsigned char>(*p)))
            ++p;
    }

    operator bool() const { return p < end; }
};

// Reads HGC lattice code output and HDF5 files into Model objects
class Importer {
public:
    explicit Importer() {
    }

private:
    // Current-ctype resolution mode for rod-history HGC points (negative so they never
    // collide with real ctypes >= 0). AUTO derives the current ctype from the HGC branch
    // (the CR* branch of a depleted-rodded base is the rod-out state); FROM_HGC keeps the
    // HGC ctype verbatim.
    static constexpr int RHST_CURRENT_AUTO     = -1;
    static constexpr int RHST_CURRENT_FROM_HGC = -2;

    static int BurnKeyFromDouble(double burn) {
        const double key = std::floor(1000.0 * burn);
        if (key > static_cast<double>(std::numeric_limits<int>::max()))
            return std::numeric_limits<int>::max();
        if (key < static_cast<double>(std::numeric_limits<int>::min()))
            return std::numeric_limits<int>::min();
        return static_cast<int>(key);
    }

    static bool IsSeparatorChar(unsigned char c) { return std::isspace(c) || c == '-' || c == '_'; }

    static std::string StripSeparatorsUpper(std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return IsSeparatorChar(c); }), s.end());
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return s;
    }

    static std::string StripSeparatorsLower(std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return IsSeparatorChar(c); }), s.end());
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    // Reconstruct accumulated fast+thermal fluence [n/cm2] along an HGC's depletion
    // trajectory by integrating the average flux over EFPD (derived from burnup and
    // power density). This replaces the old external rod-fluence CSV: the HGC already
    // carries the flux and burnup needed for flux x time.
    static std::map<int, double> ReconstructFluenceFromHGC(const Model& hgc) {
        struct Step {
            int    key;
            double burn;
            double flux;
            double pden;
        };
        // Integrand for the rod-depletion regressor: total flux (default) or the
        // macroscopic absorption reaction rate sum_g(Sigma_a,g * phi_g).  On a rodded
        // HGC the homogenized absorption is absorber-dominated, so its cumulative
        // integral tracks absorber burnout more directly than a bare fluence.
        // Select with RASBERY_ROD_FLUENCE_MODE=absorption.
        static const bool useAbsorptionRR = []() {
            const char* m = std::getenv("RASBERY_ROD_FLUENCE_MODE");
            return m != nullptr && std::string(m) == "absorption";
        }();
        std::vector<Step> steps;
        for (const auto& dpt : hgc.Dpts()) {
            if (dpt._btyp != REFR)
                continue;
            double flux = 0.0;
            if (useAbsorptionRR) {
                for (size_t g = 0; g < dpt._aflx.size(); ++g)
                    flux += dpt._xs._macx[dpt._xs.idx(static_cast<int>(g), XSAF)] * dpt._aflx[g];
            } else {
                for (const double phi : dpt._aflx)
                    flux += phi;
            }
            steps.push_back({dpt.burnKey(), dpt._data[AD_BURN], flux, dpt._data[AD_PDEN]});
        }
        std::sort(steps.begin(), steps.end(),
                  [](const Step& a, const Step& b) { return a.burn < b.burn; });

        std::map<int, double> fluenceByBurnup;
        double                fluence  = 0.0;
        double                prevEfpd = 0.0;
        double                prevFlux = 0.0;
        bool                  first    = true;
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
        if (fluenceByBurnup.empty())
            return std::numeric_limits<double>::quiet_NaN();
        const int burnKey = dpt.burnKey();
        auto      hiIt    = fluenceByBurnup.lower_bound(burnKey);
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

    static void AddHistoryDeltasInto(std::vector<HistoryDeltaCorrection>&       dst,
                                     const std::vector<HistoryDeltaCorrection>& src) {
        if (dst.size() != src.size())
            throw std::runtime_error("CHIFFON reflector average: history delta count mismatch.");

        for (size_t i = 0; i < dst.size(); ++i) {
            if (dst[i].branch_type != src[i].branch_type || dst[i].kind != src[i].kind ||
                dst[i].state_field != src[i].state_field ||
                dst[i].vector_isotopes != src[i].vector_isotopes ||
                dst[i].vector_powers != src[i].vector_powers) {
                throw std::runtime_error("CHIFFON reflector average: history delta metadata mismatch.");
            }
            AddBranchDeltaInto(dst[i].delta, src[i].delta);
        }
    }

    static void ScaleHistoryDeltas(std::vector<HistoryDeltaCorrection>& deltas, double scale) {
        for (auto& correction : deltas)
            ScaleBranchDelta(correction.delta, scale);
    }

    static void CheckAverageModelLayout(const Model& dst, const Model& src) {
        if (dst._dpts.size() != src._dpts.size())
            throw std::runtime_error("CHIFFON reflector average: depletion point count mismatch.");
        if (dst._refr_dpts != src._refr_dpts || dst._bppm_dpts != src._bppm_dpts ||
            dst._tful_dpts != src._tful_dpts || dst._tmod_dpts != src._tmod_dpts ||
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
            AddBranchDeltaInto(averaged._tmod_delt, nodes[i]->_tmod_delt);
            AddBranchDeltaInto(averaged._dmod_delt, nodes[i]->_dmod_delt);
            AddHistoryDeltasInto(averaged._history_deltas, nodes[i]->_history_deltas);
        }

        const double scale = 1.0 / static_cast<double>(nodes.size());
        for (auto& dpt : averaged._dpts)
            ScaleDepletionPoint(dpt, scale);
        ScaleBranchDelta(averaged._bppm_delt, scale);
        ScaleBranchDelta(averaged._tful_delt, scale);
        ScaleBranchDelta(averaged._tmod_delt, scale);
        ScaleBranchDelta(averaged._dmod_delt, scale);
        ScaleHistoryDeltas(averaged._history_deltas, scale);
        return averaged;
    }

    static Model& FindReflectorNode(std::unordered_map<std::string, Model>& reflModels, const std::string& nodeName) {
        auto nodeIt = reflModels.find(nodeName);
        if (nodeIt == reflModels.end())
            throw std::runtime_error("CHIFFON input: unknown reflector node '" + nodeName + "'.");
        return nodeIt->second;
    }

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
            auto nodeIt = reflModels.find(name);
            neighbor    = nodeIt != reflModels.end() ? &nodeIt->second : nullptr;
            boundary    = NODE;
        }
    }

    // Read one HGC file, apply DF if requested, and append its depletion points
    // to the existing fuel model as either SPCT or RHST fitting data.
    void AppendHGCPoints(Model& targetModel, const std::filesystem::path& baseDir,
                         const std::string& fileName, bool useDF, ReflectorSolver& reflSolver,
                         BRANCHTYPE targetType, int trajectoryCtype = -1,
                         int currentCtype = RHST_CURRENT_AUTO, int sourceCtype = -1,
                         bool                 trajectoryReference = false,
                         bool                 nondepleted         = false,
                         int                  burnupMinKey        = std::numeric_limits<int>::min(),
                         int                  burnupMaxKey        = std::numeric_limits<int>::max(),
                         const std::set<int>& burnupKeys          = std::set<int>()) {
        Model hgcModel(targetModel.name() + "#branch");
        ReadHGC(hgcModel, baseDir.string() + "/" + fileName);
        if (useDF)
            reflSolver.ApplyDF(hgcModel);

        if (targetType == BRANCHTYPE::SPCT) {
            for (auto& dpt : hgcModel._dpts) {
                const int burnKey = dpt.burnKey();
                if (burnKey < burnupMinKey || burnKey > burnupMaxKey)
                    continue;
                if (!burnupKeys.empty() && burnupKeys.find(burnKey) == burnupKeys.end())
                    continue;
                targetModel._spct_dpts.push_back(std::move(dpt));
            }
            return;
        }

        if (targetType != BRANCHTYPE::RHST)
            return;

        // Rod fluence is reconstructed from this HGC's own flux x time (no external CSV).
        const auto fluence = ReconstructFluenceFromHGC(hgcModel);

        for (auto& dpt : hgcModel._dpts) {
            const int burnKey = dpt.burnKey();
            if (burnKey < burnupMinKey || burnKey > burnupMaxKey)
                continue;
            if (!burnupKeys.empty() && burnupKeys.find(burnKey) == burnupKeys.end())
                continue;

            if (sourceCtype >= 0 && dpt._ctyp != sourceCtype)
                continue;

            AssignReconstructedFluence(dpt, fluence, nondepleted);

            const int hgcCtype        = dpt._ctyp;
            dpt._trajectory_ctyp      = trajectoryCtype;
            dpt._trajectory_reference = trajectoryReference;

            if (currentCtype >= 0) {
                dpt._ctyp = currentCtype;
            } else if (currentCtype == RHST_CURRENT_FROM_HGC) {
                dpt._ctyp = hgcCtype;
            } else if (trajectoryCtype >= 0) {
                // In a rod-history HGC the base reference was depleted rodded.
                // Its CR* branch is therefore the rod-out current state.
                dpt._ctyp = (hgcCtype == 0) ? trajectoryCtype : 0;
            }

            targetModel._rhst_dpts.push_back(std::move(dpt));
        }
    }

    void AppendRodDepletionHGC(Model& targetModel, const std::filesystem::path& baseDir,
                               const std::string& fileName, bool useDF, ReflectorSolver& reflSolver) {
        Model hgcModel(targetModel.name() + "#rod_depletion");
        ReadHGC(hgcModel, baseDir.string() + "/" + fileName);
        if (useDF)
            reflSolver.ApplyDF(hgcModel);

        for (auto& dpt : hgcModel._dpts)
            targetModel._rod_depletion_dpts.push_back(std::move(dpt));
    }

    void AppendRodDepletionPairHGC(Model& targetModel, const std::filesystem::path& baseDir,
                                   const std::string& referenceFile, const std::string& depletedFile,
                                   bool useDF, ReflectorSolver& reflSolver) {
        Model referenceModel(targetModel.name() + "#rod_depletion_reference");
        ReadHGC(referenceModel, baseDir.string() + "/" + referenceFile);
        if (useDF)
            reflSolver.ApplyDF(referenceModel);

        // Nondepleted reference: fresh absorber, so rod-material fluence is zero.
        const auto referenceFluence = ReconstructFluenceFromHGC(referenceModel);
        for (auto& dpt : referenceModel._dpts) {
            AssignReconstructedFluence(dpt, referenceFluence, /*nondepleted=*/true);
            dpt._trajectory_reference = true;
            targetModel._rod_depletion_dpts.push_back(std::move(dpt));
        }

        Model depletedModel(targetModel.name() + "#rod_depletion_depleted");
        ReadHGC(depletedModel, baseDir.string() + "/" + depletedFile);
        if (useDF)
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
                                const nlohmann::json& block, bool useDF, ReflectorSolver& reflSolver) {
        if (block.is_string()) {
            AppendRodDepletionHGC(targetModel, baseDir, block.get<std::string>(), useDF, reflSolver);
        } else if (block.is_array()) {
            for (const auto& item : block)
                AppendRodDepletionHGC(targetModel, baseDir,
                                      item.is_string() ? item.get<std::string>() : item.at("file").get<std::string>(),
                                      useDF, reflSolver);
        } else if (block.is_object()) {
            if (block.contains("reference") || block.contains("nondepleted")) {
                const std::string referenceFile =
                    block.contains("reference") ? block.at("reference").get<std::string>()
                                                : block.at("nondepleted").get<std::string>();
                AppendRodDepletionPairHGC(targetModel, baseDir, referenceFile,
                                          block.at("depleted").get<std::string>(),
                                          useDF, reflSolver);
            } else {
                AppendRodDepletionHGC(targetModel, baseDir, block.at("file").get<std::string>(), useDF, reflSolver);
            }
        }
    }

    // Resolve a JSON isotope spec (numeric index, isotope ID, or symbolic name) to an
    // isotope slot, accepting the fixed history-vector special names. Unknown names throw.
    static size_t parseVectorIsotope(const nlohmann::json& item, const std::string& context) {
        static const std::map<std::string, std::string> isotopeNameToId = {
            {      "H1",  "10010"},
            {     "B10",  "50002"},
            {       "B",  "50002"},
            {     "O16",  "80160"},
            {    "I135", "531350"},
            {   "XE135", "541350"},
            {  "XE135M", "541351"},
            {   "ND147", "601470"},
            {   "ND148", "601480"},
            {   "ND149", "601490"},
            {   "PM147", "611470"},
            {   "PM148", "611480"},
            {  "PM148G", "611480"},
            {  "PM148M", "611481"},
            {   "PM149", "611490"},
            {   "SM147", "621470"},
            {   "SM148", "621480"},
            {   "SM149", "621490"},
            {      "GD", "640000"},
            {   "GD000", "640000"},
            {"GDLUMPED", "640000"},
            {    "U234", "922340"},
            {    "U235", "922350"},
            {    "U236", "922360"},
            {    "U238", "922380"},
            {   "NP237", "932370"},
            {   "NP238", "932380"},
            {   "NP239", "932390"},
            {   "PU238", "942380"},
            {   "PU239", "942390"},
            {   "PU240", "942400"},
            {   "PU241", "942410"},
            {   "PU242", "942420"},
            {   "PU243", "942430"},
            {   "AM241", "952410"},
            {   "AM242", "952420"},
            {  "AM242M", "952421"},
            {   "AM243", "952430"},
            {   "AM244", "952440"},
            {   "CM242", "962420"},
            {   "CM243", "962430"},
            {   "CM244", "962440"},
            {   "CM245", "962450"}
        };

        size_t iso = Isotope::niso;
        if (item.is_number_unsigned()) {
            iso = item.get<size_t>();
        } else if (item.is_number_integer()) {
            const int index = item.get<int>();
            if (index >= 0)
                iso = static_cast<size_t>(index);
        } else if (item.is_string()) {
            const std::string raw  = item.get<std::string>();
            const std::string name = StripSeparatorsUpper(raw);

            if (name == "RODDEPLETIONFACTOR" || name == "RDPLFACTOR" ||
                name == "RODDEPLFACTOR" || name == "ABSORBERDEPLETIONFACTOR") {
                iso = Hv::ROD_DEPL;
            } else if (name == "LOGXE135" || name == "LNXE135" || name == "XE135LOG") {
                iso = Hv::LOG_XE;
            } else if (name == "LOGPU239" || name == "LNPU239" || name == "PU239LOG") {
                iso = Hv::LOG_PU;
            } else if (name == "LOGU238PU239" || name == "LNU238PU239" ||
                       name == "LOGU238OVERPU239" || name == "LNU238OVERPU239") {
                iso = Hv::LOG_U_PU;
            } else if (name == "TOTALBURNUP" || name == "CURRENTBURNUP" ||
                       name == "BURNUP" || name == "BURN") {
                iso = Hv::TOT_BURN;
            } else if (name == "CURRENTRODFRACTION" || name == "CURRENTRODSHARE" ||
                       name == "CURRENTRODINSERTION" || name == "RODFRACTION" ||
                       name == "RODINSERTIONFRACTION") {
                iso = Hv::CUR_ROD_FRAC;
            } else if (name == "FUELRODFLUENCE" || name == "FUELFLUENCE") {
                iso = Hv::FUEL_ROD_FLU;
            } else if (name == "RODDEDBURNUPFRACTION" || name == "RODDEDBURNUPRATIO" ||
                       name == "FUELRODBURNUPFRACTION" || name == "FUELRODBURNUPRATIO" ||
                       name == "RODDEDTOTALBURNUPRATIO" || name == "RODDEDBURNUPSHARE") {
                iso = Hv::ROD_BURN_FRAC;
            } else if (name == "FUELRODBURNUP" || name == "FUELRODDEDBURNUP" ||
                       name == "FUELBURNUP" || name == "RODDEDBURNUP") {
                iso = Hv::FUEL_ROD_BURN;
            } else if (name == "RODBURNUP" || name == "RODMATERIALBURNUP" ||
                       name == "ABSORBERBURNUP") {
                iso = Hv::ROD_BURN;
            } else if (name == "RODFLUENCE" || name == "RODMATERIALFLUENCE" ||
                       name == "RODDEPLETIONFLUENCE" || name == "FLUENCE" || name == "NRN") {
                iso = Hv::ROD_FLU;
            } else if (name == "FASTTHERMALRATIO" || name == "FASTTOTHERMALRATIO" ||
                       name == "FLUXRATIO" || name == "FASTFLUXRATIO") {
                iso = Hv::FAST_THERM;
            } else if (name == "THERMALFLUXFRACTION" || name == "THERMALFRACTION" ||
                       name == "THERMALFLUXSHARE") {
                iso = Hv::THERM_FRAC;
            } else if (Isotope::iidx.contains(raw)) {
                iso = Isotope::iidx.at(raw);
            } else if (Isotope::iidx.contains(name)) {
                iso = Isotope::iidx.at(name);
            } else if (name.size() == 5 && name[0] == '6' && Isotope::iidx.contains(name + "0")) {
                iso = Isotope::iidx.at(name + "0");
            } else if (isotopeNameToId.contains(name) && Isotope::iidx.contains(isotopeNameToId.at(name))) {
                iso = Isotope::iidx.at(isotopeNameToId.at(name));
            }
        }

        if (iso >= Isotope::niso && !IsHistoryVectorSpecial(iso))
            throw std::runtime_error("CHIFFON input: unknown isotope in " + context + ".");
        return iso;
    }

    // Read one SPCT/RHST vector settings block: the apply / ctype_independent flags, the
    // isotope basis, isotope-ratio and product cross-terms, and pre_remove. Vector
    // corrections have no polynomial order/type or weighting fields (any such keys are ignored).
    static void readVectorSettingsBlock(SPCT_SETTINGS& target, const nlohmann::json& block, const std::string& context) {
        target.apply = block.value("apply", true);
        // Keyless design: vector corrections default to ctype-independent.
        target.ctype_independent =
            block.value("ctype_independent", block.value("ctype independent", true));
        target.isotopes.clear();
        target.powers.clear();
        if (!target.apply)
            return;
        if (!block.contains("isotopes"))
            throw std::runtime_error("CHIFFON input: " + context + " needs an isotopes array.");
        for (const auto& item : block.at("isotopes")) {
            const size_t iso = parseVectorIsotope(item, context + ".isotopes");
            target.isotopes.push_back(iso);
        }

        for (size_t i = 0; i < target.isotopes.size(); ++i) {
            std::vector<int> powers(target.isotopes.size(), 0);
            powers[i] = 1;
            target.powers.push_back(std::move(powers));
        }

        if (block.contains("isotope ratio")) {
            for (const auto& term : block["isotope ratio"]) {
                if (!term.is_array())
                    throw std::runtime_error("CHIFFON input: " + context + ".isotope ratio entries must be arrays.");

                std::vector<size_t> termIsotopes;
                for (const auto& item : term) {
                    const size_t iso = parseVectorIsotope(item, context + ".isotope ratio");
                    auto         it  = std::find(target.isotopes.begin(), target.isotopes.end(), iso);
                    if (it == target.isotopes.end()) {
                        target.isotopes.push_back(iso);
                        for (auto& powers : target.powers)
                            powers.push_back(0);
                        termIsotopes.push_back(target.isotopes.size() - 1);
                    } else {
                        termIsotopes.push_back(static_cast<size_t>(std::distance(target.isotopes.begin(), it)));
                    }
                }

                std::vector<int> powers(target.isotopes.size(), 0);
                if (termIsotopes.size() == 2 && termIsotopes[0] != termIsotopes[1]) {
                    powers[termIsotopes[0]] = 1;
                    powers[termIsotopes[1]] = -1;
                } else {
                    for (const size_t iso : termIsotopes)
                        powers[iso]++;
                }

                bool hasTerm = false;
                for (const int power : powers) {
                    if (power != 0) {
                        hasTerm = true;
                        break;
                    }
                }
                if (hasTerm)
                    target.powers.push_back(std::move(powers));
            }
        }

        const std::vector<std::string> productKeys = {
            "isotope product",
            "isotope products",
            "product",
            "products",
            "cross term",
            "cross terms",
        };
        for (const auto& keyName : productKeys) {
            if (!block.contains(keyName))
                continue;

            for (const auto& term : block[keyName]) {
                if (!term.is_array())
                    throw std::runtime_error("CHIFFON input: " + context + "." + keyName + " entries must be arrays.");

                std::vector<int> powers(target.isotopes.size(), 0);
                for (const auto& item : term) {
                    const size_t iso = parseVectorIsotope(item, context + "." + keyName);
                    auto         it  = std::find(target.isotopes.begin(), target.isotopes.end(), iso);
                    size_t       idx = 0;
                    if (it == target.isotopes.end()) {
                        target.isotopes.push_back(iso);
                        for (auto& existingPowers : target.powers)
                            existingPowers.push_back(0);
                        powers.push_back(0);
                        idx = target.isotopes.size() - 1;
                    } else {
                        idx = static_cast<size_t>(std::distance(target.isotopes.begin(), it));
                    }
                    if (idx >= powers.size())
                        powers.resize(idx + 1, 0);
                    powers[idx]++;
                }

                bool hasTerm = false;
                for (const int power : powers) {
                    if (power != 0) {
                        hasTerm = true;
                        break;
                    }
                }
                if (hasTerm)
                    target.powers.push_back(std::move(powers));
            }
        }

        if (block.contains("pre_remove")) {
            target.pre_remove.clear();
            for (const auto& item : block["pre_remove"])
                target.pre_remove.push_back(item.get<std::string>());
        }
    }

    // Parse one fixed CHIFFON branch interpolation block (bppm/tful/dmod).
    static void readBranchSettingsBlock(BRCH_SETTINGS& target, const nlohmann::json& block) {
        target.apply = block.at("apply").get<bool>();
        target.order = block.at("order").get<int>();
        target.type  = block.at("type").get<std::string>();
        if (block.contains("pre_remove")) {
            target.pre_remove.clear();
            for (const auto& item : block["pre_remove"])
                target.pre_remove.push_back(item.get<std::string>());
        }
    }

    // Parse a rod-depletion settings block (accepts either key spelling at the call site).
    static void readRodDepletionSettingsBlock(BRCH_SETTINGS& target, const nlohmann::json& block) {
        target.apply = block.value("apply", true);
        target.order = block.value("order", 1);
        target.type  = block.value("type", std::string("spline"));
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

        bool          useDF = true;
        BRCH_SETTINGS bppmSettings;
        BRCH_SETTINGS tfulSettings;
        BRCH_SETTINGS dmodSettings;
        bppmSettings.order = 1;
        tfulSettings.order = 1;
        dmodSettings.order = 3;
        SPCT_SETTINGS              spectralSettings;
        SPCT_SETTINGS              rhstSpectralSettings;
        std::vector<SPCT_SETTINGS> rhstSpectralSettingsList;
        BRCH_SETTINGS              rodDepletionSettings;
        rodDepletionSettings.order = 1;
        rodDepletionSettings.type  = "spline";

        // Read the fixed CHIFFON branch interpolation blocks.
        const auto& settings = input.at("settings");
        useDF                = settings.at("discontinuity factor").get<bool>();

        readBranchSettingsBlock(bppmSettings, settings.at("bppm"));
        readBranchSettingsBlock(tfulSettings, settings.at("tful"));
        readBranchSettingsBlock(dmodSettings, settings.at("dmod"));
        if (dmodSettings.pre_remove.empty())
            dmodSettings.pre_remove.push_back("bppm");

        // Read SPCT settings. Isotope names are accepted only as known IDs or as
        // the fixed symbol names recognized by parseVectorIsotope; unknown names are input errors.
        if (settings.contains("spectral"))
            readVectorSettingsBlock(spectralSettings, settings["spectral"], "settings.spectral");

        // Keyless design: the IISC (spectral) vector must be ctype-independent. A ctype-keyed
        // IISC (legacy Hk::VEC) would be fit and stored but skipped by the RASBERY runtime.
        if (spectralSettings.apply && !spectralSettings.ctype_independent)
            throw std::runtime_error(
                "CHIFFON input: settings.spectral must be ctype-independent "
                "(ctype-keyed IISC is no longer supported in the keyless design).");

        // Existing inputs use the SPCT vector basis for RHST vector residuals.
        // A separate RHST vector block overrides this default.
        rhstSpectralSettings = spectralSettings;
        if (settings.contains("rhst spectral")) {
            readVectorSettingsBlock(rhstSpectralSettings, settings["rhst spectral"], "settings.rhst spectral");
        } else if (settings.contains("rhst_spectral")) {
            readVectorSettingsBlock(rhstSpectralSettings, settings["rhst_spectral"], "settings.rhst_spectral");
        } else if (settings.contains("rod history spectral")) {
            readVectorSettingsBlock(rhstSpectralSettings, settings["rod history spectral"], "settings.rod history spectral");
        } else if (settings.contains("rod_history_spectral")) {
            readVectorSettingsBlock(rhstSpectralSettings, settings["rod_history_spectral"], "settings.rod_history_spectral");
        }
        rhstSpectralSettingsList.push_back(rhstSpectralSettings);

        if (settings.contains("rod depletion")) {
            readRodDepletionSettingsBlock(rodDepletionSettings, settings["rod depletion"]);
        } else if (settings.contains("rod_depletion")) {
            readRodDepletionSettingsBlock(rodDepletionSettings, settings["rod_depletion"]);
        }

        std::vector<std::string> globalRodDepletionFiles;
        const nlohmann::json*    globalRodDepletionBlock = nullptr;
        if (input.contains("rod depletion")) {
            globalRodDepletionBlock = &input["rod depletion"];
        } else if (input.contains("rod_depletion")) {
            globalRodDepletionBlock = &input["rod_depletion"];
        }
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

        // Read fuel HGC files. "main" is the reference depletion set, "extra"
        // supplies SPCT fitting points, and "rod history" supplies RHST points.
        for (const auto& [fuelName, fuelSpec] : input.at("fuels").items()) {
            std::string mainFile;

            if (fuelSpec.is_string()) {
                mainFile = fuelSpec.get<std::string>();
            } else {
                mainFile = fuelSpec.at("main").get<std::string>();
            }

            Model model(fuelName);
            ReadHGC(model, baseDir.string() + "/" + mainFile);
            if (useDF) reflSolver.ApplyDF(model);

            if (fuelSpec.is_object()) {
                if (fuelSpec.contains("extra")) {
                    for (const auto& item : fuelSpec["extra"]) {
                        AppendHGCPoints(model, baseDir, item.get<std::string>(),
                                        useDF, reflSolver, BRANCHTYPE::SPCT);
                    }
                }

                if (fuelSpec.contains("rod history")) {
                    for (const auto& [rodKey, rodSpec] : fuelSpec["rod history"].items()) {
                        std::string digits;
                        for (const char c : rodKey) {
                            if (std::isdigit(static_cast<unsigned char>(c)))
                                digits.push_back(c);
                        }
                        if (digits.empty())
                            throw std::runtime_error("CHIFFON input: rod history key must contain ctype digits.");

                        int trajectoryCtype = std::stoi(digits);
                        int currentCtype    = RHST_CURRENT_AUTO;
                        int sourceCtype     = -1;

                        if (rodSpec.contains("ctype")) {
                            if (rodSpec["ctype"].is_number_integer()) {
                                trajectoryCtype = rodSpec["ctype"].get<int>();
                            } else if (rodSpec["ctype"].is_string()) {
                                digits.clear();
                                for (const char c : rodSpec["ctype"].get<std::string>()) {
                                    if (std::isdigit(static_cast<unsigned char>(c)))
                                        digits.push_back(c);
                                }
                                if (digits.empty())
                                    throw std::runtime_error("CHIFFON input: rod history ctype must contain digits.");
                                trajectoryCtype = std::stoi(digits);
                            } else {
                                throw std::runtime_error("CHIFFON input: rod history ctype must be an integer or string.");
                            }
                        }

                        if (rodSpec.contains("current_ctype")) {
                            if (rodSpec["current_ctype"].is_string() && rodSpec["current_ctype"].get<std::string>() == "hgc") {
                                currentCtype = RHST_CURRENT_FROM_HGC;
                            } else if (rodSpec["current_ctype"].is_number_integer()) {
                                currentCtype = rodSpec["current_ctype"].get<int>();
                            } else if (rodSpec["current_ctype"].is_string()) {
                                digits.clear();
                                for (const char c : rodSpec["current_ctype"].get<std::string>()) {
                                    if (std::isdigit(static_cast<unsigned char>(c)))
                                        digits.push_back(c);
                                }
                                if (digits.empty())
                                    throw std::runtime_error("CHIFFON input: rod history current_ctype must contain digits or be hgc.");
                                currentCtype = std::stoi(digits);
                            } else {
                                throw std::runtime_error("CHIFFON input: rod history current_ctype must be an integer or string.");
                            }
                        }

                        if (rodSpec.contains("source_ctype")) {
                            if (rodSpec["source_ctype"].is_number_integer()) {
                                sourceCtype = rodSpec["source_ctype"].get<int>();
                            } else if (rodSpec["source_ctype"].is_string()) {
                                digits.clear();
                                for (const char c : rodSpec["source_ctype"].get<std::string>()) {
                                    if (std::isdigit(static_cast<unsigned char>(c)))
                                        digits.push_back(c);
                                }
                                if (digits.empty())
                                    throw std::runtime_error("CHIFFON input: rod history source_ctype must contain digits.");
                                sourceCtype = std::stoi(digits);
                            } else {
                                throw std::runtime_error("CHIFFON input: rod history source_ctype must be an integer or string.");
                            }
                        }

                        // The nondepleted counterfactual (fresh absorber) keeps a zero
                        // rod-material fluence. Mark it with "nondepleted": true, or name
                        // the entry "..._non_depleted". Rod fluence is reconstructed from
                        // each HGC's flux x time, so no fluence files are needed.
                        bool nondepleted = rodSpec.value("nondepleted", false);
                        if (!nondepleted &&
                            StripSeparatorsLower(rodKey).find("nondepl") != std::string::npos)
                            nondepleted = true;

                        int burnupMinKey = std::numeric_limits<int>::min();
                        int burnupMaxKey = std::numeric_limits<int>::max();
                        if (rodSpec.contains("burnup_min_key"))
                            burnupMinKey = rodSpec.at("burnup_min_key").get<int>();
                        if (rodSpec.contains("burnup_max_key"))
                            burnupMaxKey = rodSpec.at("burnup_max_key").get<int>();
                        std::set<int> burnupKeys;
                        if (rodSpec.contains("burnup_keys")) {
                            for (const auto& item : rodSpec.at("burnup_keys"))
                                burnupKeys.insert(item.get<int>());
                        }

                        if (rodSpec.contains("main")) {
                            AppendHGCPoints(model, baseDir, rodSpec.at("main").get<std::string>(),
                                            useDF, reflSolver, BRANCHTYPE::RHST,
                                            trajectoryCtype, currentCtype, sourceCtype, true,
                                            nondepleted, burnupMinKey, burnupMaxKey, burnupKeys);
                        }

                        if (rodSpec.contains("extra")) {
                            for (const auto& item : rodSpec["extra"]) {
                                AppendHGCPoints(model, baseDir, item.get<std::string>(),
                                                useDF, reflSolver, BRANCHTYPE::RHST,
                                                trajectoryCtype, currentCtype, sourceCtype, false,
                                                nondepleted, burnupMinKey, burnupMaxKey, burnupKeys);
                            }
                        }
                    }
                }

                if (fuelSpec.contains("rod depletion")) {
                    AppendFuelRodDepletion(model, baseDir, fuelSpec["rod depletion"], useDF, reflSolver);
                } else if (fuelSpec.contains("rod_depletion")) {
                    AppendFuelRodDepletion(model, baseDir, fuelSpec["rod_depletion"], useDF, reflSolver);
                }
            }

            for (const auto& file : globalRodDepletionFiles)
                AppendRodDepletionHGC(model, baseDir, file, useDF, reflSolver);
            models.push_back(std::move(model));
        }

        // Reflectors parsing
        if (input.contains("reflectors")) {
            std::unordered_map<std::string, Model> reflModels;
            for (const auto& [reflName, reflFile] : input["reflectors"]["files"].items()) {
                std::string filename = reflFile.get<std::string>();
                Model       model(reflName);
                ReadHGC(model, baseDir.string() + "/" + filename);
                reflModels[reflName] = std::move(model);
            }

            // Parse reflector neighbors information
            if (input["reflectors"].contains("neighbors")) {
                for (const auto& [reflName, specs] : input["reflectors"]["neighbors"].items()) {
                    std::string typeStr = specs.at("type").get<std::string>();

                    if (typeStr == "averaged" || typeStr == "average") {
                        std::vector<Model*> avgNodes;
                        for (const auto& nodeNameValue : specs.at("nodes")) {
                            const std::string nodeName = nodeNameValue.get<std::string>();
                            auto              nodeIt   = reflModels.find(nodeName);
                            if (nodeIt == reflModels.end())
                                throw std::runtime_error("CHIFFON input: unknown averaged reflector node '" + nodeName + "'.");
                            avgNodes.push_back(&nodeIt->second);
                        }
                        models.push_back(AverageReflectorModels(reflName, avgNodes));
                        continue;
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
                        if (useDF) reflSolver.ApplyDF2(lNode, rNode, lNeih, rNeih, lBound, rBound);
                        lNode._name = reflName;
                        models.push_back(lNode);
                    } else {
                        if (useDF) reflSolver.ApplyDF(lNode, rNode, lNeih, rNeih, lBound, rBound);
                        rNode._name = reflName;
                        models.push_back(rNode);
                    }
                }
            }
        }

        for (auto& model : models) {
            Interpolator::Interpolate(model,
                                      bppmSettings.apply, tfulSettings.apply, dmodSettings.apply,
                                      bppmSettings, tfulSettings, dmodSettings,
                                      spectralSettings, rodDepletionSettings, rhstSpectralSettingsList);
        }
    }

#pragma region Load HDF file
    // Load all models from a CHIFFON HDF5 file (metadata, depletion points, branches, deltas)
    std::vector<Model> LoadHDF(const std::string& filename) {
        try {
            HighFive::File file(filename, HighFive::File::ReadOnly);

            // Load and validate metadata
            auto metaGroup = file.getGroup("Metadata");

            // Check version compatibility
            std::string version, format;
            if (metaGroup.exist("version")) {
                metaGroup.getDataSet("version").read(version);
            }
            if (metaGroup.exist("format")) {
                metaGroup.getDataSet("format").read(format);
                if (format != "CHIFFON_HDF5") {
                    throw std::runtime_error("Invalid HDF5 format: expected 'CHIFFON_HDF5', got '" + format + "'");
                }
            }

            // Load isotope registry
            std::vector<std::string> isoNames;
            std::vector<size_t>      isoIndices;
            metaGroup.getDataSet("isotope_names").read(isoNames);
            metaGroup.getDataSet("isotope_indices").read(isoIndices);
            metaGroup.getDataSet("niso").read(niso);

            if (isoNames.size() != isoIndices.size()) {
                throw std::runtime_error("Isotope registry mismatch: names and indices have different sizes");
            }

            iidx.clear();
            for (size_t i = 0; i < isoNames.size(); ++i) {
                iidx[isoNames[i]] = isoIndices[i];
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

                size_t numDpts;
                modelGroup.getDataSet("num_dpts").read(numDpts);

                auto dptsGroup = modelGroup.getGroup("DepletionPoints");
                if (dptsGroup.exist("layout")) {
                    model._dpts = LoadFlatDepletionPoints(dptsGroup, numDpts);
                } else {
                    model._dpts.reserve(numDpts);
                    for (size_t d = 0; d < numDpts; ++d) {
                        model._dpts.push_back(LoadLegacyDepletionPoint(dptsGroup, d, true));
                    }
                }

                LoadBranch(modelGroup, "refr_dpts", model._refr_dpts);
                // Branch dpts are only needed for fitting BranchDelta polynomials.
                // Since BranchDelta is already stored in HDF5 and loaded below,
                // skip loading branch dpts to save I/O time and memory.

                if (modelGroup.exist("branch_deltas")) {
                    LoadBranchCorrections(modelGroup, model);
                } else {
                    LoadBranchDelta(modelGroup, "bppm_delt", model._bppm_delt);
                    LoadBranchDelta(modelGroup, "tful_delt", model._tful_delt);
                    LoadBranchDelta(modelGroup, "tmod_delt", model._tmod_delt);
                    LoadBranchDelta(modelGroup, "dmod_delt", model._dmod_delt);
                    if (modelGroup.exist("history_deltas"))
                        LoadHistoryDeltas(modelGroup, model);
                }
                if (modelGroup.exist("rod_depletion_delt"))
                    LoadBranchDelta(modelGroup, "rod_depletion_delt", model._rod_depletion_delt);

                models.push_back(std::move(model));
            }

            // Load unified depletion matrices (for depletion) if present
            if (file.exist("IsotopeChains")) {
                auto chainGroup = file.getGroup("IsotopeChains");
                ImporterDetail::LoadChainMatrix(chainGroup, "depDecay", Isotope::depDecay);
                ImporterDetail::LoadChainMatrix(chainGroup, "depTrans", Isotope::depTrans);
            }

            return models;

        } catch (const HighFive::Exception& e) {
            throw std::runtime_error("Failed to load HDF5 file '" + filename + "': " + e.what());
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load HDF5 file '" + filename + "': " + e.what());
        }
    }

private:
    struct FlatFieldData {
        std::vector<size_t> offsets = {0};
        std::vector<double> values;
    };

    FlatFieldData LoadFlatFieldData(const HighFive::Group& parent, const std::string& field_name) {
        FlatFieldData field;
        if (parent.exist(field_name + "_offsets"))
            parent.getDataSet(field_name + "_offsets").read(field.offsets);
        if (parent.exist(field_name + "_values"))
            parent.getDataSet(field_name + "_values").read(field.values);
        if (field.offsets.empty())
            field.offsets.push_back(0);
        return field;
    }

    static void AssignFlatField(std::vector<double>& dst, const FlatFieldData& field, size_t slot) {
        if (slot + 1 >= field.offsets.size()) return;

        const size_t start = field.offsets[slot];
        const size_t end   = field.offsets[slot + 1];
        if (start > end || end > field.values.size()) return;

        dst.assign(field.values.begin() + start, field.values.begin() + end);
    }

    static void AssignFlatField(milk::Vector<double>& dst, const FlatFieldData& field, size_t slot) {
        if (slot + 1 >= field.offsets.size()) return;

        const size_t start = field.offsets[slot];
        const size_t end   = field.offsets[slot + 1];
        if (start > end || end > field.values.size()) return;
        if (start == end) {
            dst.clear();
            return;
        }

        std::vector<double> tmp(field.values.begin() + start, field.values.begin() + end);
        dst.fromVector(tmp);
    }

    static void LoadVectorSlice(milk::Vector<double>& dst, const std::vector<double>& src, size_t& off, size_t n) {
        if (n == 0 || off + n > src.size()) return;

        std::vector<double> tmp(src.begin() + off, src.begin() + off + n);
        dst.fromVector(tmp);
        off += n;
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
        if (parent.exist("heavy_indices"))
            parent.getDataSet("heavy_indices").read(heavy_indices);

        if (meta.size() != numDpts * 4) {
            throw std::runtime_error("Invalid flat depletion metadata size");
        }
        if (data.size() < numDpts * AD_SIZE) {
            throw std::runtime_error("Invalid flat depletion data size");
        }

        std::vector<DepletionPoint> dpts;
        dpts.reserve(numDpts);
        for (size_t idx = 0; idx < numDpts; ++idx) {
            const size_t   meta_off = idx * 4;
            DepletionPoint dpt(static_cast<size_t>(meta[meta_off + 0]),
                               static_cast<size_t>(meta[meta_off + 1]),
                               static_cast<BRANCHTYPE>(meta[meta_off + 2]),
                               meta[meta_off + 3]);

            const size_t data_off = idx * AD_SIZE;
            for (size_t i = 0; i < AD_SIZE; ++i)
                dpt._data[i] = data[data_off + i];

            dpts.push_back(std::move(dpt));
        }

        const FlatFieldData aflx = LoadFlatFieldData(parent, "aflx");
        const FlatFieldData gmap = LoadFlatFieldData(parent, "gmap");
        const FlatFieldData fmap = LoadFlatFieldData(parent, "fmap");
        const FlatFieldData chix = LoadFlatFieldData(parent, "chix");
        const FlatFieldData sdfa = LoadFlatFieldData(parent, "sdfa");
        const FlatFieldData pdfa = LoadFlatFieldData(parent, "pdfa");
        const FlatFieldData iden = LoadFlatFieldData(parent, "iden");

        std::vector<size_t> xs_meta;
        std::vector<double> xs_macx, xs_lmpx, xs_micx;
        if (parent.exist("xs_meta")) parent.getDataSet("xs_meta").read(xs_meta);
        if (parent.exist("xs_macx")) parent.getDataSet("xs_macx").read(xs_macx);
        if (parent.exist("xs_lmpx")) parent.getDataSet("xs_lmpx").read(xs_lmpx);
        if (parent.exist("xs_micx")) parent.getDataSet("xs_micx").read(xs_micx);

        size_t macx_offset = 0;
        size_t lmpx_offset = 0;
        size_t micx_offset = 0;

        for (size_t slot = 0; slot < heavy_indices.size(); ++slot) {
            const size_t dpt_index = heavy_indices[slot];
            if (dpt_index >= dpts.size()) continue;

            auto& dpt = dpts[dpt_index];
            AssignFlatField(dpt._aflx, aflx, slot);
            AssignFlatField(dpt._gmap, gmap, slot);
            AssignFlatField(dpt._fmap, fmap, slot);
            AssignFlatField(dpt._chix, chix, slot);
            AssignFlatField(dpt._sdfa, sdfa, slot);
            AssignFlatField(dpt._pdfa, pdfa, slot);
            AssignFlatField(dpt._iden, iden, slot);

            const size_t xs_meta_off = slot * 3;
            if (xs_meta_off + 2 >= xs_meta.size()) continue;

            const size_t ngrp = xs_meta[xs_meta_off + 0];
            const size_t ndat = xs_meta[xs_meta_off + 1];
            const size_t nmem = xs_meta[xs_meta_off + 2];

            dpt._xs.deallocate();
            dpt._xs._ngrp = ngrp;
            dpt._xs._ndat = ndat;
            dpt._xs._nmem = nmem;
            dpt._xs.allocate(nmem);

            LoadVectorSlice(dpt._xs._macx, xs_macx, macx_offset, nmem);
            LoadVectorSlice(dpt._xs._lmpx, xs_lmpx, lmpx_offset, nmem);
            LoadVectorSlice(dpt._xs._micx, xs_micx, micx_offset, niso * nmem);
        }

        return dpts;
    }

    // Read a single DepletionPoint from the legacy per-group HDF5 layout.
    // If skip_heavy=true, only load metadata (skip XS, pin maps, isotope data)
    // Used for branch dpts that are only needed for BranchDelta fitting (already done)
    DepletionPoint LoadLegacyDepletionPoint(const HighFive::Group& parent, size_t idx, bool skip_heavy = false) {
        auto dptGroup = parent.getGroup("DPT_" + std::to_string(idx));

        std::vector<int> meta;
        dptGroup.getDataSet("meta").read(meta);
        size_t ngrp = static_cast<size_t>(meta[0]);
        size_t npin = static_cast<size_t>(meta[1]);
        int    btyp = meta[2];
        int    ctyp = meta[3];

        DepletionPoint dpt(ngrp, npin, static_cast<BRANCHTYPE>(btyp), ctyp);

        std::vector<double> data;
        dptGroup.getDataSet("data").read(data);
        for (size_t i = 0; i < AD_SIZE && i < data.size(); ++i)
            dpt._data[i] = data[i];

        // Skip heavy data for non-reference dpts (branch points)
        // Reference dpts always need full XS/pin/isotope data
        if (skip_heavy && btyp != BRANCHTYPE::REFR) return dpt;

        if (dptGroup.exist("aflx")) dptGroup.getDataSet("aflx").read(dpt._aflx);
        if (dptGroup.exist("gmap")) dptGroup.getDataSet("gmap").read(dpt._gmap);
        if (dptGroup.exist("fmap")) dptGroup.getDataSet("fmap").read(dpt._fmap);
        if (dptGroup.exist("chix")) dptGroup.getDataSet("chix").read(dpt._chix);
        if (dptGroup.exist("sdfa")) dptGroup.getDataSet("sdfa").read(dpt._sdfa);
        if (dptGroup.exist("pdfa")) dptGroup.getDataSet("pdfa").read(dpt._pdfa);

        LoadCrossSection(dptGroup, "xs", dpt._xs);

        // Load isotope densities
        std::vector<double> iden;
        dptGroup.getDataSet("iden").read(iden);
        if (iden.size() == niso && niso > 0) {
            dpt._iden.fromVector(iden);
        }

        return dpt;
    }

    // Read CrossSection arrays from HDF5 group
    void LoadCrossSection(const HighFive::Group& parent, const std::string& name, CrossSection& xs) {
        auto xsGroup = parent.getGroup(name);

        std::vector<size_t> meta;
        xsGroup.getDataSet("meta").read(meta);
        size_t ngrp = meta[0];
        size_t ndat = meta[1];
        size_t nmem = meta[2];

        xs.deallocate();
        xs._ngrp = ngrp;
        xs._ndat = ndat;
        xs._nmem = nmem;
        xs.allocate(nmem);

        // Read arrays; rmcx is reconstructed at runtime.
        std::vector<double> macx, lmpx, micx;
        xsGroup.getDataSet("macx").read(macx);
        xsGroup.getDataSet("lmpx").read(lmpx);
        xsGroup.getDataSet("micx").read(micx);

        if (macx.size() == nmem) xs._macx.fromVector(macx);
        if (lmpx.size() == nmem) xs._lmpx.fromVector(lmpx);
        if (micx.size() == niso * nmem && niso > 0) xs._micx.fromVector(micx);
    }

    void LoadBranch(const HighFive::Group& parent, const std::string& name, Reference& ref) {
        auto                group = parent.getGroup(name);
        std::vector<int>    cTypes, burnups;
        std::vector<size_t> indices;

        group.getDataSet("cTypes").read(cTypes);
        group.getDataSet("burnups").read(burnups);
        group.getDataSet("indices").read(indices);

        ref.clear();
        for (size_t i = 0; i < cTypes.size(); ++i) {
            ref[cTypes[i]][burnups[i]] = indices[i];
        }
    }

    void LoadBranchMap(const HighFive::Group& parent, const std::string& name, Branch& branch) {
        auto group = parent.getGroup(name);
        branch.clear();

        if (!group.exist("cTypes")) return;

        std::vector<int>    cTypes, burnups;
        std::vector<size_t> offsets, indices_flat;

        group.getDataSet("cTypes").read(cTypes);
        group.getDataSet("burnups").read(burnups);
        group.getDataSet("offsets").read(offsets);
        group.getDataSet("indices_flat").read(indices_flat);

        for (size_t i = 0; i < cTypes.size(); ++i) {
            size_t start                  = offsets[i];
            size_t end                    = offsets[i + 1];
            branch[cTypes[i]][burnups[i]] = std::vector<size_t>(indices_flat.begin() + start, indices_flat.begin() + end);
        }
    }

    void LoadBranchDelta(const HighFive::Group& parent, const std::string& name, BranchDelta& delta) {
        auto group = parent.getGroup(name);
        delta.clear();

        size_t numEntries = 0;
        if (group.exist("num_entries")) group.getDataSet("num_entries").read(numEntries);

        if (numEntries > 0) {
            std::vector<int>    cTypes, burnups;
            std::vector<size_t> dxs_meta, xs_meta, knots_offsets;
            std::vector<double> knots_flat, all_macx, all_lmpx, all_micx;

            group.getDataSet("cTypes").read(cTypes);
            group.getDataSet("burnups").read(burnups);
            group.getDataSet("dxs_meta").read(dxs_meta);
            group.getDataSet("knots_flat").read(knots_flat);
            group.getDataSet("knots_offsets").read(knots_offsets);
            group.getDataSet("xs_meta").read(xs_meta);
            group.getDataSet("all_macx").read(all_macx);
            group.getDataSet("all_lmpx").read(all_lmpx);
            // all_rmcx is reconstructed at runtime.
            group.getDataSet("all_micx").read(all_micx);

            size_t xs_meta_idx = 0;
            size_t macx_offset = 0, lmpx_offset = 0, micx_offset = 0;

            for (size_t i = 0; i < numEntries; ++i) {
                size_t ngrp   = dxs_meta[i * 4 + 0];
                size_t nord   = dxs_meta[i * 4 + 1];
                size_t mode   = dxs_meta[i * 4 + 2];
                size_t ncoeff = dxs_meta[i * 4 + 3];

                size_t              kstart = knots_offsets[i];
                size_t              kend   = knots_offsets[i + 1];
                std::vector<double> knots(knots_flat.begin() + kstart, knots_flat.begin() + kend);
                DeltaCrossSection   dxs;
                if (static_cast<InterpolMode>(mode) == SPLINE_MODE)
                    dxs = DeltaCrossSection(ngrp, nord, ncoeff, knots);
                else {
                    dxs        = DeltaCrossSection(ngrp, nord);
                    dxs._knots = std::move(knots);
                }

                for (size_t c = 0; c < nord; ++c) {
                    CrossSection& xs      = dxs._u[c];
                    size_t        xs_ngrp = xs_meta[xs_meta_idx * 3 + 0];
                    size_t        xs_ndat = xs_meta[xs_meta_idx * 3 + 1];
                    size_t        xs_nmem = xs_meta[xs_meta_idx * 3 + 2];

                    xs.deallocate();
                    xs._ngrp = xs_ngrp;
                    xs._ndat = xs_ndat;
                    xs._nmem = xs_nmem;
                    xs.allocate(xs_nmem);

                    ImporterDetail::LoadFlatVector(xs._macx, all_macx, macx_offset, xs_nmem);
                    ImporterDetail::LoadFlatVector(xs._lmpx, all_lmpx, lmpx_offset, xs_nmem);
                    ImporterDetail::LoadFlatVector(xs._micx, all_micx, micx_offset, niso * xs_nmem);
                    xs_meta_idx++;
                }

                delta[cTypes[i]][burnups[i]] = std::move(dxs);
            }
        }
    }

    void LoadHistoryDeltas(const HighFive::Group& parent, Model& model) {
        auto   group      = parent.getGroup("history_deltas");
        size_t numHistory = 0;
        if (group.exist("num_history_deltas"))
            group.getDataSet("num_history_deltas").read(numHistory);

        std::vector<int>    kinds, stateFields;
        std::vector<size_t> vectorIsotopesFlat, vectorOffsets;
        std::vector<int>    vectorPowersFlat;
        if (group.exist("kinds")) group.getDataSet("kinds").read(kinds);
        if (group.exist("state_fields")) group.getDataSet("state_fields").read(stateFields);
        if (group.exist("vector_isotopes_flat")) group.getDataSet("vector_isotopes_flat").read(vectorIsotopesFlat);
        if (group.exist("vector_powers_flat")) group.getDataSet("vector_powers_flat").read(vectorPowersFlat);
        if (group.exist("vector_offsets")) group.getDataSet("vector_offsets").read(vectorOffsets);

        model._history_deltas.clear();
        model._history_deltas.reserve(numHistory);
        for (size_t i = 0; i < numHistory; ++i) {
            HistoryDeltaCorrection history;
            history.kind        = i < kinds.size() ? kinds[i] : 0;
            history.branch_type = BRANCHTYPE::SPCT;
            history.state_field = i < stateFields.size() ? stateFields[i] : AD_TMOD;
            if (vectorOffsets.size() == numHistory + 1) {
                const size_t begin = std::min(vectorOffsets[i], vectorIsotopesFlat.size());
                const size_t end   = std::min(vectorOffsets[i + 1], vectorIsotopesFlat.size());
                history.vector_isotopes.assign(vectorIsotopesFlat.begin() + begin,
                                               vectorIsotopesFlat.begin() + end);
                if (vectorPowersFlat.size() >= end) {
                    history.vector_powers.assign(vectorPowersFlat.begin() + begin,
                                                 vectorPowersFlat.begin() + end);
                } else {
                    history.vector_powers.assign(history.vector_isotopes.size(), 0);
                }
            }
            LoadBranchDelta(group, "delta_" + std::to_string(i), history.delta);
            model._history_deltas.push_back(std::move(history));
        }
    }

    void LoadBranchCorrections(const HighFive::Group& parent, Model& model) {
        auto group = parent.getGroup("branch_deltas");

        size_t              numBranchDeltas = 0;
        std::vector<int>    branchTypes, kinds, stateFields;
        std::vector<size_t> vectorIsotopesFlat, vectorOffsets;
        std::vector<int>    vectorPowersFlat;

        group.getDataSet("num_branch_deltas").read(numBranchDeltas);
        group.getDataSet("branch_types").read(branchTypes);
        group.getDataSet("kinds").read(kinds);
        group.getDataSet("state_fields").read(stateFields);
        group.getDataSet("vector_isotopes_flat").read(vectorIsotopesFlat);
        group.getDataSet("vector_powers_flat").read(vectorPowersFlat);
        group.getDataSet("vector_offsets").read(vectorOffsets);

        if (branchTypes.size() != numBranchDeltas ||
            kinds.size() != numBranchDeltas ||
            stateFields.size() != numBranchDeltas ||
            vectorOffsets.size() != numBranchDeltas + 1) {
            throw std::runtime_error("branch_deltas metadata size mismatch.");
        }

        model._bppm_delt.clear();
        model._tful_delt.clear();
        model._tmod_delt.clear();
        model._dmod_delt.clear();
        model._history_deltas.clear();

        for (size_t i = 0; i < numBranchDeltas; ++i) {
            BranchDelta delta;
            LoadBranchDelta(group, "delta_" + std::to_string(i), delta);

            const BRANCHTYPE branchType = static_cast<BRANCHTYPE>(branchTypes[i]);
            if (branchType == BRANCHTYPE::BPPM) {
                model._bppm_delt = std::move(delta);
            } else if (branchType == BRANCHTYPE::TFUL) {
                model._tful_delt = std::move(delta);
            } else if (branchType == BRANCHTYPE::TMOD) {
                model._tmod_delt = std::move(delta);
            } else if (branchType == BRANCHTYPE::DMOD) {
                model._dmod_delt = std::move(delta);
            } else {
                HistoryDeltaCorrection history;
                history.delta       = std::move(delta);
                history.branch_type = branchType;
                history.kind        = kinds[i];
                history.state_field = stateFields[i];
                const size_t begin  = vectorOffsets[i];
                const size_t end    = vectorOffsets[i + 1];
                if (end > vectorIsotopesFlat.size() || end > vectorPowersFlat.size()) {
                    throw std::runtime_error("branch_deltas vector metadata is out of range.");
                }
                history.vector_isotopes.assign(vectorIsotopesFlat.begin() + begin,
                                               vectorIsotopesFlat.begin() + end);
                history.vector_powers.assign(vectorPowersFlat.begin() + begin,
                                             vectorPowersFlat.begin() + end);
                model._history_deltas.push_back(std::move(history));
            }
        }
    }

    void LoadDoubleMap(const HighFive::Group& parent, const std::string& name,
                       std::unordered_map<int, std::map<int, double>>& data) {
        auto                group = parent.getGroup(name);
        std::vector<int>    cTypes, burnups;
        std::vector<double> values;

        group.getDataSet("cTypes").read(cTypes);
        group.getDataSet("burnups").read(burnups);
        group.getDataSet("values").read(values);

        data.clear();
        for (size_t i = 0; i < cTypes.size(); ++i) {
            data[cTypes[i]][burnups[i]] = values[i];
        }
    }

#pragma endregion

#pragma region Load specific model from HDF file
public:
    // Load a single model by index from a CHIFFON HDF5 file
    Model LoadSingleModel(const std::string& filename, size_t modelIndex) {
        try {
            HighFive::File file(filename, HighFive::File::ReadOnly);

            // Load isotope registry
            auto                     metaGroup = file.getGroup("Metadata");
            std::vector<std::string> isoNames;
            std::vector<size_t>      isoIndices;
            metaGroup.getDataSet("isotope_names").read(isoNames);
            metaGroup.getDataSet("isotope_indices").read(isoIndices);
            metaGroup.getDataSet("niso").read(niso);

            iidx.clear();
            for (size_t i = 0; i < isoNames.size(); ++i) {
                iidx[isoNames[i]] = isoIndices[i];
            }

            size_t numModels;
            metaGroup.getDataSet("num_models").read(numModels);

            if (modelIndex >= numModels) {
                throw std::out_of_range("Model index " + std::to_string(modelIndex) +
                                        " out of range (total: " + std::to_string(numModels) + ")");
            }

            auto modelsGroup = file.getGroup("Models");
            auto modelGroup  = modelsGroup.getGroup("Model_" + std::to_string(modelIndex));

            Model model;
            modelGroup.getDataSet("id").read(model._id);
            modelGroup.getDataSet("name").read(model._name);

            size_t numDpts;
            modelGroup.getDataSet("num_dpts").read(numDpts);

            auto dptsGroup = modelGroup.getGroup("DepletionPoints");
            if (dptsGroup.exist("layout")) {
                model._dpts = LoadFlatDepletionPoints(dptsGroup, numDpts);
            } else {
                model._dpts.reserve(numDpts);
                for (size_t d = 0; d < numDpts; ++d) {
                    model._dpts.push_back(LoadLegacyDepletionPoint(dptsGroup, d));
                }
            }

            LoadBranch(modelGroup, "refr_dpts", model._refr_dpts);
            LoadBranchMap(modelGroup, "bppm_dpts", model._bppm_dpts);
            LoadBranchMap(modelGroup, "tful_dpts", model._tful_dpts);
            LoadBranchMap(modelGroup, "tmod_dpts", model._tmod_dpts);
            LoadBranchMap(modelGroup, "dmod_dpts", model._dmod_dpts);

            if (modelGroup.exist("branch_deltas")) {
                LoadBranchCorrections(modelGroup, model);
            } else {
                LoadBranchDelta(modelGroup, "bppm_delt", model._bppm_delt);
                LoadBranchDelta(modelGroup, "tful_delt", model._tful_delt);
                LoadBranchDelta(modelGroup, "tmod_delt", model._tmod_delt);
                LoadBranchDelta(modelGroup, "dmod_delt", model._dmod_delt);
                if (modelGroup.exist("history_deltas"))
                    LoadHistoryDeltas(modelGroup, model);
            }
            if (modelGroup.exist("rod_depletion_delt"))
                LoadBranchDelta(modelGroup, "rod_depletion_delt", model._rod_depletion_delt);

            return model;

        } catch (const HighFive::Exception& e) {
            throw std::runtime_error("Failed to load model " + std::to_string(modelIndex) +
                                     " from HDF5 file '" + filename + "': " + e.what());
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model " + std::to_string(modelIndex) +
                                     " from HDF5 file '" + filename + "': " + e.what());
        }
    }

    Model LoadModelByName(const std::string& filename, const std::string& modelName) {
        try {
            HighFive::File file(filename, HighFive::File::ReadOnly);

            auto   metaGroup = file.getGroup("Metadata");
            size_t numModels;
            metaGroup.getDataSet("num_models").read(numModels);

            auto modelsGroup = file.getGroup("Models");
            for (size_t m = 0; m < numModels; ++m) {
                auto        modelGroup = modelsGroup.getGroup("Model_" + std::to_string(m));
                std::string name;
                modelGroup.getDataSet("name").read(name);

                if (name == modelName) {
                    return LoadSingleModel(filename, m);
                }
            }

            throw std::runtime_error("Model '" + modelName + "' not found in HDF5 file '" + filename + "'");

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model '" + modelName +
                                     "' from HDF5 file '" + filename + "': " + e.what());
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
        // Lumped Gd: individual Gd isotopes → virtual "640000"
        {"641540", "640000"},
        {"641550", "640000"},
        {"641560", "640000"},
        {"641570", "640000"},
        {"641580", "640000"},
        {"641590", "640000"}
    };

    // Gd lumping weights for effective number density: N_eff = 5*N154 + 4*N155 + 3*N156 + 2*N157 + 1*N158
    // Gd159 has weight 0 (fast decay, not included in effective number density)
    const std::unordered_map<std::string, double> gdWeight = {
        {"641540", 5.0},
        {"641550", 4.0},
        {"641560", 3.0},
        {"641570", 2.0},
        {"641580", 1.0},
        {"641590", 0.0}
    };

public:
    // Block-wise HGC file parser: splits on '%' delimiters and dispatches TITL/FLUX/DIST/MACX/MICX/ADFT blocks
    void ReadHGC(Model& model, const std::string& fileName) {
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
                FastStream blockStream(cur, blockEnd - cur);

                std::string blockType;
                blockStream >> blockType;

                if (!blockType.empty()) {
                    if (blockType == "TITL")
                        dpnt = &ReadTitl(model, blockStream);
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
    DepletionPoint& ReadTitl(Model& model, FastStream& blockData) {
        std::string _line;
        std::string _case;

        blockData.getline(_line); // TITL (현재 줄의 나머지)
        blockData.getline(_line); // ===
        blockData.getline(_line); // Empty line
        blockData.getline(_case);

        BRANCHTYPE bType = BRANCHTYPE::REFR;
        if (_case.find("BOR") != std::string::npos)
            bType = BRANCHTYPE::BPPM;
        else if (_case.find("TFUEL") != std::string::npos)
            bType = BRANCHTYPE::TFUL;
        else if (_case.find("DMOD") != std::string::npos)
            bType = BRANCHTYPE::DMOD;
        else if (_case.find("TMOD") != std::string::npos)
            bType = BRANCHTYPE::TMOD;
        else
            bType = BRANCHTYPE::REFR;

        size_t _crstringIdx = _case.find("CR");
        int    cType        = 0;
        if (_crstringIdx != std::string::npos) {
            size_t digitBegin = _crstringIdx + 2;
            size_t digitEnd   = digitBegin;
            while (digitEnd < _case.size() &&
                   std::isdigit(static_cast<unsigned char>(_case[digitEnd])))
                ++digitEnd;
            if (digitEnd > digitBegin)
                cType = stoi(_case.substr(digitBegin, digitEnd - digitBegin));
        }

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

    // Read average flux per group, skip removed surface/corner/current data (12 tokens each)
    void ReadFlux(DepletionPoint& dpnt, FastStream& blockData) {
        for (int ig = 0; ig < dpnt._ngrp; ig++) {
            blockData >> dpnt._aflx[ig];
            // Skip sflx(4) + cflx(4) + jnet(4) = 12 tokens per group
            for (int i = 0; i < 12; i++)
                blockData.skipToken();
        }
    }

    // Read pin-wise power and flux maps, skipping unused pmap/rmap/bmap fields.
    void ReadDist(DepletionPoint& dpnt, FastStream& blockData) {
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
    void ReadMacx(DepletionPoint& dpnt, FastStream& blockData) {
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
    void ReadMicx(DepletionPoint& dpnt, FastStream& blockData) {
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
    void ReadFyld(DepletionPoint& dpnt, FastStream& blockData) {
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
    void ReadXsmn(DepletionPoint& dpnt, FastStream& blockData) {
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
    void ReadAdft(DepletionPoint& dpnt, FastStream& blockData) {
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
