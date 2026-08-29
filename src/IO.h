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
#include "CohortContext.h"
#include "Geometry.h"
#include "XSSet.h"
#include "Scheduler.h"
#include "IoWriter.h"

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

    // WP10.1: the DECK half of the canonical case key, folded where the parsed
    // deck is live and carried as two short strings.  Keeping the whole parsed
    // deck alive instead would cost a copy per batch slot for a value that is
    // only ever hashed.
    std::string _deck_key_digest;   ///< sha256 of casekey::deckPayload(config)
    std::string _deck_key_core_op;  ///< which symmetry op canonicalised the LP

    // WP8 stage 2: which COHORT this case belongs to -- the middle lifetime.
    // Acquired where GeometryInput is final (after the shuffle resolver and the
    // restart fallback have both had their say) and before XSSet::Initialize,
    // so a case is attached to its cohort before anything that could consume
    // the cohort's state runs.  The shared_ptr keeps the Context alive for the
    // life of the process, which is what it is for.
    std::shared_ptr<const cohort::Context> _cohort;

    struct ShuffleSpec {
        int target_row, target_col;
        int source_row, source_col;
        int cycle;
        int rotation;
    };
    std::vector<ShuffleSpec> _shuffle_specs;

    /// The result file's writer session.  Non-null between OpenResult() and
    /// CloseResult(); the HighFive::File inside it belongs to whichever thread
    /// is allowed inside HDF5 for it (this one in inline mode, the writer thread
    /// otherwise) -- never read it here without fencing first.
    std::shared_ptr<iowriter::FileSession> _result_session;
    /// Restart snapshots are fire-and-forget on the thread path, so their
    /// sessions are kept to be fenced and error-checked at job end: a restart
    /// write that failed must fail ITS job, not vanish into a process counter.
    mutable std::vector<std::shared_ptr<iowriter::FileSession>> _restart_sessions;
    std::filesystem::path _result_path;
    std::filesystem::path _pin_power_csv_path;
    bool _pin_power_csv_started = false;

    /// Wait for every write this job queued and rethrow the writer's error.
    /// No-op in inline mode (nothing is ever in flight).
    void FenceJobWrites() const;

    /// Rethrow a writer error that has ALREADY happened, without waiting.  One
    /// mutex-protected bool read per statepoint: it stops a deck whose output
    /// file could not even be created from computing its whole run before
    /// CloseResult() finds out.  Fencing here instead would re-serialise the
    /// very hand-off this class exists to make asynchronous.
    void ThrowIfWritesFailed() const;

    /// The old `if (!_result_file)` precondition.  The handle is only
    /// inspectable on the inline path -- in thread mode the writer owns it, so
    /// the session's existence is the test: OpenResult queued the open ahead of
    /// anything recorded later, and the queue is FIFO.
    bool HasOpenResult() const;

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
    /// WP10.1.  Empty until ReadInput() has run.
    const std::string& deck_key_digest() const { return _deck_key_digest; }
    const std::string& deck_key_core_op() const { return _deck_key_core_op; }
    /// WP8 stage 2.  Null until ReadInput() has run.
    const std::shared_ptr<const cohort::Context>& cohort_context() const { return _cohort; }
    std::string cohort_key() const { return _cohort ? _cohort->key : std::string(); }
    const std::string& restart_path() const { return _restart_path; }
    bool has_restart() const { return !_restart_path.empty(); }
    double restart_efpd() const { return _restart_efpd; }
    const std::map<int, std::string>& restart_files() const { return _restart_files; }
    bool has_shuffle() const { return !_shuffle_specs.empty(); }

    /// WP10.3.  `statepoint_grid` is a DECK transform applied between the JSON
    /// parse and the canonical deck digest: empty (or "full") is the deck as
    /// written and costs one string compare, anything else rewrites the burnup
    /// schedule (src/StatepointGrid.h).  It is a parameter and not a member
    /// because a coarse case and a full case are two cases, and the digest that
    /// tells them apart has to be taken of the deck that was actually solved.
    void ReadInput(const std::string& filepath, const std::string& statepoint_grid = "");
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
