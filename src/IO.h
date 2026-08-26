#pragma once

#include "nlohmann/json.hpp"
#include "highfive/highfive.hpp"

#include "Model.h"
#include "Hdf5Guard.h"

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <filesystem>

#include "pch.h"
#include "Geometry.h"
#include "XSSet.h"
#include "Scheduler.h"

namespace rasbery {

/// @brief Handles input parsing from JSON and HDF5 result output.
class IO {
private:
    Geometry&   _g;
    XSSet&      _xs;
    Scheduler&  _s;
    std::string _xs_path;
    std::string _input_dir;
    GeometryInput _gin;
    std::string _restart_path;
    std::map<int, std::string> _restart_files;
    std::map<int, double> _restart_cooling_days;
    std::map<int, int> _restart_cooling_substeps;
    int _primary_restart_cycle = 1;
    double _restart_efpd = 0.0;

    struct ShuffleSpec {
        int target_row, target_col;
        int source_row, source_col;
        int cycle;
        int rotation;
    };
    std::vector<ShuffleSpec> _shuffle_specs;

    std::unique_ptr<HighFive::File> _result_file;
    std::filesystem::path _result_path;
    std::filesystem::path _pin_power_csv_path;
    bool _pin_power_csv_started = false;

    static bool TryParseShuffleEntry(const std::string& entry,
                                     int tgt_row, int tgt_col,
                                     ShuffleSpec& out);
    void ApplyShuffle();

    static PrintOpt ParsePrintOpt(const nlohmann::ordered_json& item);
    void ParseSchedule(const nlohmann::ordered_json& config);

public:
    IO(Geometry& g, XSSet& xs, Scheduler& sched);
    ~IO();

    const std::string& input_dir() const { return _input_dir; }

    /// Directory the result HDF5 was opened in, with a trailing separator, so
    /// it composes exactly like input_dir().  Empty before OpenResult().
    ///
    /// This is the base of a job's OUTPUT namespace (plan Rev.4 Sec 7): decks
    /// may share an input file and an output parent directory, but never an
    /// output path -- so deriving restart/scratch from the output keeps every
    /// job's namespace distinct by construction, whereas deriving them from the
    /// input makes N decks that share one input collide on restart_1.h5.
    std::string result_dir() const {
        if (_result_path.empty()) return {};
        std::string dir = _result_path.parent_path().string();
        if (dir.empty()) return "./";
        if (dir.back() != '/' && dir.back() != '\\') dir += '/';
        return dir;
    }

    /// Stem of the result HDF5 ("out0" for .../run/out0.h5), the second half of
    /// the output namespace: Sec 7 allows two jobs to SHARE an output parent
    /// directory, so the directory alone does not separate their restart files.
    /// Same derivation OpenResult() already uses for the pin-power CSV.
    std::string result_stem() const {
        if (_result_path.empty()) return {};
        const std::string stem = _result_path.stem().string();
        return stem.empty() ? std::string("result") : stem;
    }

    const std::string& xs_path() const { return _xs_path; }
    const std::string& restart_path() const { return _restart_path; }
    bool has_restart() const { return !_restart_path.empty(); }
    double restart_efpd() const { return _restart_efpd; }
    const std::map<int, std::string>& restart_files() const { return _restart_files; }
    bool has_shuffle() const { return !_shuffle_specs.empty(); }

    void ReadInput(const std::string& filepath);
    void AddResult(Geometry& g, double keff,
                   int schedule_index, int step_no, double efpd);

    void OpenResult(const std::string& filepath);
    void WriteStepToResult(Geometry& g, const XSSet& xs, int schedule_index);
    void CloseResult();

    void SaveRestart(const std::string& filepath,
                     Geometry& g, XSSet& xs,
                     double eigv, double efpd, int step) const;

    static GeometryInput LoadGeometryFromRestart(const std::string& filepath);
};

} // namespace rasbery
