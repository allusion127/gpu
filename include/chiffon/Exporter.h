#pragma once

#include "Model.h"
#include "Interpolator.h"
#include "Hdf5Guard.h"

#include <algorithm>
#include <string>
#include <vector>
#include "CompatFormat.h"

namespace Chiffon {
// Serializes Model objects to CHIFFON HDF5 format
class Exporter {
public:
    // Save all models to CHIFFON HDF5: metadata (isotope registry), depletion points, branches, deltas
    static void SaveHDF(const std::string& filename, const std::vector<Model>& models) {
        Hdf5Guard hdf5_guard;
        try {
            HighFive::File file(filename, HighFive::File::Overwrite);

            // Metadata group - isotope registry and version info
            auto metaGroup = file.createGroup("Metadata");

            // Version information for compatibility checking
            metaGroup.createDataSet("version", std::string(HDF_VERSION));
            metaGroup.createDataSet("format", std::string("CHIFFON_HDF5"));

            // Isotope registry
            std::vector<std::string> isoNames;
            std::vector<size_t>      isoIndices;
            isoNames.reserve(iidx.size());
            isoIndices.reserve(iidx.size());

            for (const auto& [name, idx] : iidx) {
                isoNames.push_back(name);
                isoIndices.push_back(idx);
            }

            metaGroup.createDataSet("isotope_names", isoNames);
            metaGroup.createDataSet("isotope_indices", isoIndices);
            metaGroup.createDataSet("niso", niso);
            metaGroup.createDataSet("num_models", models.size());

            // Branches the fit dropped because their abscissa never moved.
            //
            // Interpolator warns about these on stderr once per (model, branch)
            // pair, which is exactly as durable as the terminal it scrolled
            // past.  The library it produced then looks like any other library
            // that simply has no boron/fuel-temperature/density response, and
            // months later nobody can say whether a flat branch was physics or
            // a missing DeCART EDIT/isotope entry.  Written as "model/branch"
            // strings, newline-joined so an empty list is still a well-formed
            // dataset; "" therefore means "nothing was dropped", while an
            // absent dataset means "written by a build that did not record it".
            {
                const std::vector<std::string> dropped = SnapshotDegenerateBranchLog();
                std::string joined;
                for (size_t i = 0; i < dropped.size(); ++i) {
                    if (i) joined += '\n';
                    joined += dropped[i];
                }
                metaGroup.createDataSet("degenerate_branches", joined);
                metaGroup.createDataSet("degenerate_branch_count", dropped.size());
            }

            // Models group
            auto modelsGroup = file.createGroup("Models");

            for (size_t m = 0; m < models.size(); ++m) {
                const Model& model      = models[m];
                auto         modelGroup = modelsGroup.createGroup("Model_" + std::to_string(m));

                // Model attributes
                modelGroup.createDataSet("id", model._id);
                modelGroup.createDataSet("name", model._name);
                modelGroup.createDataSet("num_dpts", model._dpts.size());
                // -1 when the fuel carries no rodded-depletion twin; absent in
                // libraries written before the blend existed.
                modelGroup.createDataSet("history_partner", model.history_partner());

                // Depletion points
                SaveDepletionPoints(modelGroup, model._dpts);

                // Branch indices (only the reference map is read back; branch-point
                // maps are fit-time-only state and are not serialized).
                SaveBranch(modelGroup, "refr_dpts", model._refr_dpts);

                SaveBranchDelta(modelGroup, "bppm_delt", model._bppm_delt);
                SaveBranchDelta(modelGroup, "tful_delt", model._tful_delt);
                SaveBranchDelta(modelGroup, "dmod_delt", model._dmod_delt);
                SaveBranchDelta(modelGroup, "rod_depletion_delt", model._rod_depletion_delt);
                if (!model._rod_depletion_branch[0].empty())
                    SaveBranchDelta(modelGroup, "rod_depletion_branch_bppm", model._rod_depletion_branch[0]);
                if (!model._rod_depletion_branch[1].empty())
                    SaveBranchDelta(modelGroup, "rod_depletion_branch_tful", model._rod_depletion_branch[1]);
                if (!model._rod_depletion_branch[2].empty())
                    SaveBranchDelta(modelGroup, "rod_depletion_branch_dmod", model._rod_depletion_branch[2]);
                SaveSpectralHistory(modelGroup, model._spectral_history);
            }

            file.flush();

        } catch (const std::exception& e) {
            throw std::runtime_error(std::format("Failed to save HDF5 file '{}': {}", filename, e.what()));
        }
    }

private:
    static constexpr const char* DepletionLayout() { return "flat_v1"; }

    static void AppendFlat(std::vector<double>& dst, const std::vector<double>& src) {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    static void AppendFlat(std::vector<double>& dst, const milk::Vector<double>& src) {
        if (src.size() > 0)
            dst.insert(dst.end(), src.data(), src.data() + src.size());
    }

    template <typename Getter>
    static void SaveIndexedField(HighFive::Group& parent, const std::string& field_name,
                                 const std::vector<const DepletionPoint*>& heavy_dpts, Getter&& getter) {
        std::vector<size_t> offsets;
        std::vector<double> values;

        offsets.reserve(heavy_dpts.size() + 1);
        offsets.push_back(0);

        for (const auto* dpt : heavy_dpts) {
            const auto& field = getter(*dpt);
            AppendFlat(values, field);
            offsets.push_back(values.size());
        }

        parent.createDataSet(field_name + "_offsets", offsets);
        parent.createDataSet(field_name + "_values", values);
    }

    template <typename Field>
    static void SaveMemberField(HighFive::Group& parent, const std::string& field_name,
                                const std::vector<const DepletionPoint*>& heavy_dpts, Field DepletionPoint::*member) {
        SaveIndexedField(parent, field_name, heavy_dpts, [member](const DepletionPoint& dpt) -> const Field& {
            return dpt.*member;
        });
    }

    static void SaveReferenceCrossSections(HighFive::Group&                          parent,
                                           const std::vector<const DepletionPoint*>& heavy_dpts) {
        std::vector<size_t> xs_meta;
        std::vector<double> xs_macx;
        std::vector<double> xs_lmpx;
        std::vector<double> xs_micx;

        xs_meta.reserve(heavy_dpts.size() * 3);
        for (const auto* dpt : heavy_dpts) {
            const auto& xs = dpt->_xs;
            xs_meta.push_back(xs._ngrp);
            xs_meta.push_back(xs._ndat);
            xs_meta.push_back(xs._nmem);
            AppendFlat(xs_macx, xs._macx);
            AppendFlat(xs_lmpx, xs._lmpx);
            AppendFlat(xs_micx, xs._micx);
        }

        parent.createDataSet("xs_meta", xs_meta);
        parent.createDataSet("xs_macx", xs_macx);
        parent.createDataSet("xs_lmpx", xs_lmpx);
        parent.createDataSet("xs_micx", xs_micx);
    }

    static void SaveDepletionPoints(
        HighFive::Group& parent,
        const std::vector<DepletionPoint>& dpts) {
        auto                               group = parent.createGroup("DepletionPoints");
        std::vector<int>                   meta;
        std::vector<double>                data;
        std::vector<double>                rod_fluence;
        std::vector<size_t>                heavy_indices;
        std::vector<const DepletionPoint*> heavy_dpts;

        meta.reserve(dpts.size() * 4);
        data.reserve(dpts.size() * AD_SIZE);
        rod_fluence.reserve(dpts.size());

        for (size_t idx = 0; idx < dpts.size(); ++idx) {
            const auto& dpt = dpts[idx];

            meta.push_back(static_cast<int>(dpt._ngrp));
            meta.push_back(static_cast<int>(dpt._npin));
            meta.push_back(static_cast<int>(dpt._btyp));
            meta.push_back(dpt._ctyp);
            data.insert(data.end(), dpt._data.begin(), dpt._data.end());
            rod_fluence.push_back(dpt._rod_fluence);

            if (dpt._btyp == BRANCHTYPE::REFR) {
                heavy_indices.push_back(idx);
                heavy_dpts.push_back(&dpt);
            }
        }

        group.createDataSet("layout", std::string(DepletionLayout()));
        group.createDataSet("meta", meta);
        group.createDataSet("data", data);
        // Rodded references retain their reconstructed rod fluence.
        group.createDataSet("rod_fluence", rod_fluence);
        group.createDataSet("heavy_indices", heavy_indices);

        SaveMemberField(group, "aflx", heavy_dpts, &DepletionPoint::_aflx);
        SaveMemberField(group, "gmap", heavy_dpts, &DepletionPoint::_gmap);
        SaveMemberField(group, "fmap", heavy_dpts, &DepletionPoint::_fmap);
        SaveMemberField(group, "chix", heavy_dpts, &DepletionPoint::_chix);
        SaveMemberField(group, "sdfa", heavy_dpts, &DepletionPoint::_sdfa);
        SaveMemberField(group, "pdfa", heavy_dpts, &DepletionPoint::_pdfa);
        SaveMemberField(group, "iden", heavy_dpts, &DepletionPoint::_iden);
        SaveReferenceCrossSections(group, heavy_dpts);
    }

    static void SaveBranch(HighFive::Group& parent, const std::string& name, const Reference& ref) {
        auto                group = parent.createGroup(name);
        std::vector<int>    cTypes, burnups;
        std::vector<size_t> indices;
        std::vector<int>    orderedTypes;
        orderedTypes.reserve(ref.size());
        for (const auto& [cType, burnupMap] : ref) {
            (void)burnupMap;
            orderedTypes.push_back(cType);
        }
        std::sort(orderedTypes.begin(), orderedTypes.end());
        for (const int cType : orderedTypes) {
            const auto& buMap = ref.at(cType);
            for (const auto& [bu, idx] : buMap) {
                cTypes.push_back(cType);
                burnups.push_back(bu);
                indices.push_back(idx);
            }
        }

        group.createDataSet("cTypes", cTypes);
        group.createDataSet("burnups", burnups);
        group.createDataSet("indices", indices);
    }

    struct DeltaPayload {
        std::vector<size_t> dxsMeta;
        std::vector<double> knots;
        std::vector<size_t> knotOffsets = {0};
        std::vector<size_t> xsMeta;
        std::vector<double> macroscopic;
        std::vector<double> lumped;
        std::vector<double> microscopic;
    };

    static DeltaPayload PackDeltas(
        const std::vector<const DeltaCrossSection*>& deltas) {
        DeltaPayload payload;
        payload.dxsMeta.reserve(deltas.size() * 4);

        size_t coefficientCount = 0;
        size_t knotCount        = 0;
        size_t macroscopicCount = 0;
        size_t lumpedCount      = 0;
        size_t microscopicCount = 0;
        for (const DeltaCrossSection* delta : deltas) {
            coefficientCount += delta->_nord;
            knotCount += delta->_knots.size();
            for (size_t i = 0; i < delta->_nord; ++i) {
                macroscopicCount += delta->_u[i]._macx.size();
                lumpedCount += delta->_u[i]._lmpx.size();
                microscopicCount += delta->_u[i]._micx.size();
            }
        }

        payload.xsMeta.reserve(coefficientCount * 3);
        payload.knots.reserve(knotCount);
        payload.macroscopic.reserve(macroscopicCount);
        payload.lumped.reserve(lumpedCount);
        payload.microscopic.reserve(microscopicCount);

        for (const DeltaCrossSection* delta : deltas) {
            payload.dxsMeta.push_back(delta->_ngrp);
            payload.dxsMeta.push_back(delta->_nord);
            payload.dxsMeta.push_back(static_cast<size_t>(delta->_mode));
            payload.dxsMeta.push_back(delta->_ncoeff);
            payload.knots.insert(
                payload.knots.end(), delta->_knots.begin(),
                delta->_knots.end());
            payload.knotOffsets.push_back(payload.knots.size());

            for (size_t i = 0; i < delta->_nord; ++i) {
                const CrossSection& coefficient = delta->_u[i];
                payload.xsMeta.push_back(coefficient._ngrp);
                payload.xsMeta.push_back(coefficient._ndat);
                payload.xsMeta.push_back(coefficient._nmem);
                AppendFlat(payload.macroscopic, coefficient._macx);
                AppendFlat(payload.lumped, coefficient._lmpx);
                AppendFlat(payload.microscopic, coefficient._micx);
            }
        }
        return payload;
    }

    static void SaveDeltaPayload(
        HighFive::Group& group,
        const std::vector<const DeltaCrossSection*>& deltas) {
        DeltaPayload payload = PackDeltas(deltas);
        group.createDataSet("num_entries", deltas.size());
        group.createDataSet("dxs_meta", payload.dxsMeta);
        group.createDataSet("knots_flat", payload.knots);
        group.createDataSet("knots_offsets", payload.knotOffsets);
        group.createDataSet("xs_meta", payload.xsMeta);
        group.createDataSet("all_macx", payload.macroscopic);
        group.createDataSet("all_lmpx", payload.lumped);
        group.createDataSet("all_micx", payload.microscopic);
    }

    static void SaveBranchDelta(
        HighFive::Group& parent, const std::string& name,
        const BranchDelta& delta) {
        auto                                  group = parent.createGroup(name);
        std::vector<int>                      cTypes;
        std::vector<int>                      burnups;
        std::vector<const DeltaCrossSection*> entries;
        std::vector<int>                      orderedTypes;
        orderedTypes.reserve(delta.size());
        for (const auto& [cType, burnupMap] : delta) {
            (void)burnupMap;
            orderedTypes.push_back(cType);
        }
        std::sort(orderedTypes.begin(), orderedTypes.end());
        for (const int cType : orderedTypes) {
            const auto& burnupMap = delta.at(cType);
            for (const auto& [burnup, crossSection] : burnupMap) {
                cTypes.push_back(cType);
                burnups.push_back(burnup);
                entries.push_back(&crossSection);
            }
        }

        group.createDataSet("cTypes", cTypes);
        group.createDataSet("burnups", burnups);
        SaveDeltaPayload(group, entries);
    }

    static void SaveBurnupDelta(
        HighFive::Group& parent, const std::string& name,
        const BurnupDelta& delta) {
        auto                                  group = parent.createGroup(name);
        std::vector<int>                      burnups;
        std::vector<const DeltaCrossSection*> entries;
        burnups.reserve(delta.size());
        entries.reserve(delta.size());
        for (const auto& [burnup, crossSection] : delta) {
            burnups.push_back(burnup);
            entries.push_back(&crossSection);
        }

        group.createDataSet("burnups", burnups);
        SaveDeltaPayload(group, entries);
    }

    static void SaveSpectralHistory(
        HighFive::Group& parent,
        const std::vector<SpectralHistoryCorrection>& corrections) {
        auto group = parent.createGroup("spectral_history");
        std::vector<size_t> isotopes;
        std::vector<int>    coordinates;
        std::vector<size_t> partners;
        std::vector<int>    rodScaled;
        isotopes.reserve(corrections.size());
        coordinates.reserve(corrections.size());
        partners.reserve(corrections.size());
        rodScaled.reserve(corrections.size());

        for (size_t i = 0; i < corrections.size(); ++i) {
            isotopes.push_back(corrections[i].term.isotope);
            partners.push_back(corrections[i].term.partner);
            coordinates.push_back(
                static_cast<int>(corrections[i].term.coordinate));
            rodScaled.push_back(corrections[i].rod_scaled ? 1 : 0);
            SaveBurnupDelta(
                group, "term_" + std::to_string(i),
                corrections[i].delta);
        }
        group.createDataSet("num_terms", corrections.size());
        group.createDataSet("isotopes", isotopes);
        group.createDataSet("coordinates", coordinates);
        group.createDataSet("partners", partners);
        group.createDataSet("rod_scaled", rodScaled);
    }
};
} // namespace Chiffon
