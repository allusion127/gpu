#pragma once

#include "Model.h"

#include <format>

namespace Chiffon {
// Serializes Model objects to CHIFFON HDF5 format
class Exporter {
public:
    static constexpr const char* HdfVersion() { return "2.4.0"; }

    // Save all models to CHIFFON HDF5: metadata (isotope registry), depletion points, branches, deltas
    static void SaveHDF(const std::string& filename, const std::vector<Model>& models) {
        try {
            HighFive::File file(filename, HighFive::File::Overwrite);

            // Metadata group - isotope registry and version info
            auto metaGroup = file.createGroup("Metadata");

            // Version information for compatibility checking
            metaGroup.createDataSet("version", std::string(HdfVersion()));
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

            // Models group
            auto modelsGroup = file.createGroup("Models");

            for (size_t m = 0; m < models.size(); ++m) {
                const Model& model      = models[m];
                auto         modelGroup = modelsGroup.createGroup("Model_" + std::to_string(m));

                // Model attributes
                modelGroup.createDataSet("id", model._id);
                modelGroup.createDataSet("name", model._name);
                modelGroup.createDataSet("num_dpts", model._dpts.size());

                // Depletion points
                SaveDepletionPoints(modelGroup, model._dpts, false);

                // Branch indices
                SaveBranch(modelGroup, "refr_dpts", model._refr_dpts);
                SaveBranchMap(modelGroup, "bppm_dpts", model._bppm_dpts);
                SaveBranchMap(modelGroup, "tful_dpts", model._tful_dpts);
                SaveBranchMap(modelGroup, "tmod_dpts", model._tmod_dpts);
                SaveBranchMap(modelGroup, "dmod_dpts", model._dmod_dpts);

                // Delta cross-sections
                SaveBranchDelta(modelGroup, "bppm_delt", model._bppm_delt);
                SaveBranchDelta(modelGroup, "tful_delt", model._tful_delt);
                SaveBranchDelta(modelGroup, "tmod_delt", model._tmod_delt);
                SaveBranchDelta(modelGroup, "dmod_delt", model._dmod_delt);
                SaveBranchDelta(modelGroup, "rod_depletion_delt", model._rod_depletion_delt);
                SaveBranchCorrections(modelGroup, model);
            }

            file.flush();

        } catch (const HighFive::Exception& e) {
            throw std::runtime_error(std::format("Failed to save HDF5 file '{}': {}", filename, e.what()));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::format("Failed to save HDF5 file '{}': {}", filename, e.what()));
        }
    }

    // Utility: Get HDF5 file metadata without loading full data
    static void PrintHDF5Info(const std::string& filename) {
        try {
            HighFive::File file(filename, HighFive::File::ReadOnly);

            auto metaGroup = file.getGroup("Metadata");

            std::string version, format;
            size_t      niso, num_models;

            if (metaGroup.exist("version")) {
                metaGroup.getDataSet("version").read(version);
                std::cout << "Version: " << version << std::endl;
            }
            if (metaGroup.exist("format")) {
                metaGroup.getDataSet("format").read(format);
                std::cout << "Format: " << format << std::endl;
            }

            metaGroup.getDataSet("niso").read(niso);
            metaGroup.getDataSet("num_models").read(num_models);

            std::cout << "Number of isotopes: " << niso << std::endl;
            std::cout << "Number of models: " << num_models << std::endl;

            auto modelsGroup = file.getGroup("Models");
            for (size_t m = 0; m < num_models; ++m) {
                auto        modelGroup = modelsGroup.getGroup("Model_" + std::to_string(m));
                std::string name;
                size_t      num_dpts;
                modelGroup.getDataSet("name").read(name);
                modelGroup.getDataSet("num_dpts").read(num_dpts);
                std::cout << "  Model[" << m << "]: " << name << " (" << num_dpts << " depletion points)" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Failed to read HDF5 info: " << e.what() << std::endl;
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

    static void SaveDepletionPoints(HighFive::Group& parent, const std::vector<DepletionPoint>& dpts,
                                    bool include_branch_heavy) {
        auto                               group = parent.createGroup("DepletionPoints");
        std::vector<int>                   meta;
        std::vector<double>                data;
        std::vector<size_t>                heavy_indices;
        std::vector<const DepletionPoint*> heavy_dpts;

        meta.reserve(dpts.size() * 4);
        data.reserve(dpts.size() * AD_SIZE);

        for (size_t idx = 0; idx < dpts.size(); ++idx) {
            const auto& dpt = dpts[idx];

            meta.push_back(static_cast<int>(dpt._ngrp));
            meta.push_back(static_cast<int>(dpt._npin));
            meta.push_back(static_cast<int>(dpt._btyp));
            meta.push_back(dpt._ctyp);
            data.insert(data.end(), dpt._data.begin(), dpt._data.end());

            if (dpt._btyp == BRANCHTYPE::REFR || include_branch_heavy) {
                heavy_indices.push_back(idx);
                heavy_dpts.push_back(&dpt);
            }
        }

        group.createDataSet("layout", std::string(DepletionLayout()));
        group.createDataSet("meta", meta);
        group.createDataSet("data", data);
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

        for (const auto& [cType, buMap] : ref) {
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

    static void SaveBranchMap(HighFive::Group& parent, const std::string& name, const Branch& branch) {
        // Flatten: instead of creating a group per entry, store as parallel arrays
        // cTypes[i], burnups[i] are the keys; offsets[i] is start index into indices_flat
        // offsets has size num_entries+1 (last element = total size of indices_flat)
        auto                group = parent.createGroup(name);
        std::vector<int>    cTypes, burnups;
        std::vector<size_t> offsets;
        std::vector<size_t> indices_flat;

        offsets.push_back(0);
        for (const auto& [cType, buMap] : branch) {
            for (const auto& [bu, idxVec] : buMap) {
                cTypes.push_back(cType);
                burnups.push_back(bu);
                indices_flat.insert(indices_flat.end(), idxVec.begin(), idxVec.end());
                offsets.push_back(indices_flat.size());
            }
        }

        group.createDataSet("cTypes", cTypes);
        group.createDataSet("burnups", burnups);
        group.createDataSet("offsets", offsets);
        group.createDataSet("indices_flat", indices_flat);
    }

    // Save BranchDelta as bulk flat arrays: DXS meta, knots, and concatenated XS coefficient data
    static void SaveBranchDelta(HighFive::Group& parent, const std::string& name, const BranchDelta& delta) {
        // Flatten: store keys as parallel arrays and all DeltaCrossSection data compactly.
        // For each entry we store the DXS metadata + flattened coefficient XS data.
        auto             group = parent.createGroup(name);
        std::vector<int> cTypes, burnups;

        struct DxsEntry {
            int                      cType;
            int                      burnup;
            const DeltaCrossSection* dxs;
        };
        std::vector<DxsEntry> entries;

        for (const auto& [cType, buMap] : delta) {
            for (const auto& [bu, deltaXS] : buMap) {
                entries.push_back({cType, bu, &deltaXS});
            }
        }

        size_t numEntries = entries.size();

        // Per-entry metadata: {ngrp, nord, mode, ncoeff} as flat array of size numEntries*4
        std::vector<size_t> dxs_meta;
        dxs_meta.reserve(numEntries * 4);

        // Per-entry knots: stored as flat array with offsets
        std::vector<double> knots_flat;
        std::vector<size_t> knots_offsets;
        knots_offsets.push_back(0);

        // Per-entry XS metadata: for each coefficient, {ngrp, ndat, nmem} stored flat
        // Total coefficients across all entries
        size_t totalCoeffs = 0;
        for (const auto& entry : entries) {
            totalCoeffs += entry.dxs->_nord;
        }

        // XS meta: totalCoeffs * 3 values
        std::vector<size_t> xs_meta;
        xs_meta.reserve(totalCoeffs * 3);

        // Pass 1: compute total sizes for reserve (avoid reallocation)
        size_t total_macx = 0, total_lmpx = 0, total_micx = 0, total_knots = 0;
        for (const auto& entry : entries) {
            const auto& dxs = *entry.dxs;
            total_knots += dxs._knots.size();
            for (size_t i = 0; i < dxs._nord; ++i) {
                const CrossSection& xs = dxs._u[i];
                total_macx += xs._macx.size();
                total_lmpx += xs._lmpx.size();
                total_micx += xs._micx.size();
            }
        }

        std::vector<double> all_macx, all_lmpx, all_rmcx, all_micx;
        all_macx.reserve(total_macx);
        all_lmpx.reserve(total_lmpx);
        all_micx.reserve(total_micx);
        knots_flat.reserve(total_knots);

        // Pass 2: fill metadata + append XS data directly from raw pointers (no toVector)
        for (const auto& entry : entries) {
            const auto& dxs = *entry.dxs;
            cTypes.push_back(entry.cType);
            burnups.push_back(entry.burnup);

            dxs_meta.push_back(dxs._ngrp);
            dxs_meta.push_back(dxs._nord);
            dxs_meta.push_back(static_cast<size_t>(dxs._mode));
            dxs_meta.push_back(dxs._ncoeff);

            knots_flat.insert(knots_flat.end(), dxs._knots.begin(), dxs._knots.end());
            knots_offsets.push_back(knots_flat.size());

            for (size_t i = 0; i < dxs._nord; ++i) {
                const CrossSection& xs = dxs._u[i];
                xs_meta.push_back(xs._ngrp);
                xs_meta.push_back(xs._ndat);
                xs_meta.push_back(xs._nmem);

                AppendFlat(all_macx, xs._macx);
                AppendFlat(all_lmpx, xs._lmpx);
                // rmcx is reconstructed at runtime in the SoA layer.
                AppendFlat(all_micx, xs._micx);
            }
        }

        group.createDataSet("num_entries", numEntries);
        group.createDataSet("cTypes", cTypes);
        group.createDataSet("burnups", burnups);
        group.createDataSet("dxs_meta", dxs_meta);
        group.createDataSet("knots_flat", knots_flat);
        group.createDataSet("knots_offsets", knots_offsets);
        group.createDataSet("xs_meta", xs_meta);
        group.createDataSet("all_macx", all_macx);
        group.createDataSet("all_lmpx", all_lmpx);
        group.createDataSet("all_rmcx", all_rmcx);
        group.createDataSet("all_micx", all_micx);
    }

    static void SaveBranchCorrections(HighFive::Group& parent, const Model& model) {
        auto group = parent.createGroup("branch_deltas");

        struct Entry {
            BRANCHTYPE                 branch_type;
            int                        kind;
            int                        state_field;
            const std::vector<size_t>* vector_isotopes;
            const std::vector<int>*    vector_powers;
            const BranchDelta*         delta;
        };

        std::vector<Entry> entries;
        if (!model._bppm_delt.empty())
            entries.push_back({BRANCHTYPE::BPPM, 0, 0, nullptr, nullptr, &model._bppm_delt});
        if (!model._tful_delt.empty())
            entries.push_back({BRANCHTYPE::TFUL, 0, 0, nullptr, nullptr, &model._tful_delt});
        if (!model._tmod_delt.empty())
            entries.push_back({BRANCHTYPE::TMOD, 0, 0, nullptr, nullptr, &model._tmod_delt});
        if (!model._dmod_delt.empty())
            entries.push_back({BRANCHTYPE::DMOD, 0, 0, nullptr, nullptr, &model._dmod_delt});

        for (const auto& history : model._history_deltas) {
            entries.push_back({history.branch_type, history.kind, history.state_field,
                               &history.vector_isotopes, &history.vector_powers,
                               &history.delta});
        }

        std::vector<int>         branch_types, kinds, state_fields;
        std::vector<std::string> component_names;
        std::vector<size_t>      vector_isotopes_flat, vector_offsets;
        std::vector<int>         vector_powers_flat;
        vector_offsets.push_back(0);

        branch_types.reserve(entries.size());
        kinds.reserve(entries.size());
        state_fields.reserve(entries.size());
        component_names.reserve(entries.size());

        for (const auto& entry : entries) {
            branch_types.push_back(static_cast<int>(entry.branch_type));
            kinds.push_back(entry.kind);
            state_fields.push_back(entry.state_field);
            component_names.emplace_back(
                CorrectionComponentName(CorrectionComponentFromBranchAndKind(entry.branch_type, entry.kind)));
            if (entry.vector_isotopes != nullptr) {
                vector_isotopes_flat.insert(vector_isotopes_flat.end(),
                                            entry.vector_isotopes->begin(), entry.vector_isotopes->end());
            }
            if (entry.vector_powers != nullptr) {
                vector_powers_flat.insert(vector_powers_flat.end(),
                                          entry.vector_powers->begin(), entry.vector_powers->end());
            }
            vector_offsets.push_back(vector_isotopes_flat.size());
        }

        group.createDataSet("num_branch_deltas", entries.size());
        group.createDataSet("branch_types", branch_types);
        group.createDataSet("kinds", kinds);
        group.createDataSet("component_names", component_names);
        group.createDataSet("state_fields", state_fields);
        group.createDataSet("vector_isotopes_flat", vector_isotopes_flat);
        group.createDataSet("vector_powers_flat", vector_powers_flat);
        group.createDataSet("vector_offsets", vector_offsets);

        for (size_t i = 0; i < entries.size(); ++i)
            SaveBranchDelta(group, "delta_" + std::to_string(i), *entries[i].delta);
    }
};
} // namespace Chiffon
