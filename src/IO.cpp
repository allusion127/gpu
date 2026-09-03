#include "IO.h"

#include "CaseKey.h"
#include "StatepointGrid.h"
#include "CohortContext.h"
#include "Model.h"
#include "XsLibrary.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include "CompatFormat.h"
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace rasbery;

namespace {

constexpr double kMinimumScheduleValue = 1.0e-10;
constexpr double kMaximumPowerRate     = 100.0;
constexpr char   kWslUncPrefix[]       = "//wsl.localhost/Ubuntu/";

SearchType ParseSearchTypeString(const std::string& text) {
    if (text == "boron") { return SearchType::BORON; }
    if (text == "rod") { return SearchType::RODCRIT; }
    return SearchType::KEFF;
}

THMode ParseThModeString(const std::string& text) {
    if (text == "steady") return THMode::STEADY;
    if (text == "transient") return THMode::TRANSIENT;
    return THMode::NONE;
}

std::string LowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// Return the value of the first key present in `obj` (in listed precedence), or nullptr.
const nlohmann::ordered_json* FirstPresentKey(const nlohmann::ordered_json&      obj,
                                              std::initializer_list<const char*> keys) {
    for (const char* key : keys)
        if (obj.contains(key))
            return &obj[key];
    return nullptr;
}

// The fuel-temperature-table deck key, parsed from wherever it was declared.
//
// TWO PLACES ARE ACCEPTED and exactly one may be USED: `geometry.dimensions`,
// beside `nfrod`, because the table is a property of the fuel the geometry
// describes; and "default parameters", because that is where a deck author
// looks for a physics default.  A deck that says it in BOTH is refused rather
// than silently resolved by precedence -- two declarations are two intents and
// the loser would be invisible.  See src/ThTfTable.h for the WH/CE finding.
const char* const kTfTableKeys[] = {"tf table", "tf_table", "tfuel table", "tfuel_table"};

rasbery::th::TfTableSpec ParseTfTableSpec(const nlohmann::ordered_json& value,
                                          const std::string&            whence) {
    if (value.is_string())
        return [&] {
            const auto c = rasbery::th::tfChoiceOfString(value.get<std::string>(),
                                                         whence.c_str());
            rasbery::th::TfTableSpec spec;
            spec.name = c.name;
            spec.path = c.path;
            return spec;
        }();

    if (!value.is_object())
        throw std::runtime_error("IO::ReadInput: " + whence +
                                 " must be \"wh\", \"ce\", \"file:<path>\" or an "
                                 "inline object {\"lpd\": [...], \"bu\": [...], "
                                 "\"dt\": [[...]]}.");

    rasbery::th::TfTableSpec spec;
    spec.name = "inline";
    const auto* lpd = FirstPresentKey(value, {"lpd", "LPD", "power", "linear power density"});
    const auto* bu  = FirstPresentKey(value, {"bu", "burnup", "BU"});
    const auto* dt  = FirstPresentKey(value, {"dt", "dT", "deltat", "delta_t", "tf"});
    if (lpd == nullptr || bu == nullptr || dt == nullptr)
        throw std::runtime_error("IO::ReadInput: an inline " + whence +
                                 " needs \"lpd\", \"bu\" and \"dt\" -- the %DEF_TFT "
                                 "shape: an LPD axis, a burnup axis and one dT row per "
                                 "burnup.");
    spec.lpd = lpd->get<std::vector<double>>();
    spec.bu  = bu->get<std::vector<double>>();
    for (const auto& row : *dt) {
        const auto values = row.is_array() ? row.get<std::vector<double>>()
                                           : std::vector<double>{row.get<double>()};
        if (values.size() != spec.lpd.size())
            throw std::runtime_error("IO::ReadInput: every dT row of " + whence +
                                     " must have one entry per LPD knot.");
        spec.dt.insert(spec.dt.end(), values.begin(), values.end());
    }
    if (spec.dt.size() != spec.lpd.size() * spec.bu.size())
        throw std::runtime_error("IO::ReadInput: " + whence +
                                 " needs exactly one dT row per burnup knot.");
    return spec;
}

// True if any listed key is present and truthy; stops at the first truthy key.
bool AnyKeyTrue(const nlohmann::ordered_json& obj, std::initializer_list<const char*> keys) {
    for (const char* key : keys)
        if (obj.value(key, false))
            return true;
    return false;
}

bool ParseXenonTransientString(const std::string& text) {
    const std::string mode = LowerAscii(text);
    if (mode == "transient" || mode == "tr")
        return true;
    if (mode == "equilibrium" || mode == "eq")
        return false;
    throw std::runtime_error("IO: xenon mode must be 'transient' or 'equilibrium'.");
}

bool ParseXenonTransientValue(const nlohmann::ordered_json& value) {
    if (value.is_string())
        return ParseXenonTransientString(value.get<std::string>());
    if (value.is_object()) {
        if (value.contains("mode"))
            return ParseXenonTransientValue(value["mode"]);
        if (value.contains("xenon_mode"))
            return ParseXenonTransientValue(value["xenon_mode"]);
        if (value.contains("transient"))
            return value["transient"].get<bool>();
    }
    throw std::runtime_error("IO: xenon mode must be a string or an object with a mode field.");
}

bool ReadXenonTransientOverride(const nlohmann::ordered_json& item, bool current) {
    if (item.contains("xenon"))
        return ParseXenonTransientValue(item["xenon"]);
    if (item.contains("xenon_mode"))
        return ParseXenonTransientValue(item["xenon_mode"]);
    if (item.contains("xenon mode"))
        return ParseXenonTransientValue(item["xenon mode"]);
    return current;
}

std::string NormalizeInputPath(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    if (path.rfind(kWslUncPrefix, 0) == 0) {
        path = "/" + path.substr(std::char_traits<char>::length(kWslUncPrefix));
    }
    return path;
}

std::string ResolveInputPath(const std::string& input_directory, std::string path) {
    path = NormalizeInputPath(std::move(path));
    if (!path.empty() && path.front() != '/') {
        path = input_directory + path;
    }
    return path;
}

void RegisterRestartEntry(std::map<int, std::string>& restart_files,
                          std::string&                primary_restart_path,
                          int&                        primary_restart_cycle,
                          const std::string&          input_directory,
                          int                         cycle,
                          std::string                 raw_path) {
    std::string resolved_path = ResolveInputPath(input_directory, std::move(raw_path));
    restart_files[cycle]      = resolved_path;
    if (primary_restart_path.empty()) {
        primary_restart_path  = resolved_path;
        primary_restart_cycle = cycle;
    }
}

double RestartCoolingDays(const nlohmann::ordered_json& entry) {
    if (const auto* v = FirstPresentKey(entry, {"cooling time", "cooling_time", "cooling days", "cooling_days"}))
        return v->get<double>();
    return 0.0;
}

int RestartCoolingSubsteps(const nlohmann::ordered_json& entry) {
    if (const auto* v = FirstPresentKey(entry, {"cooling substeps", "cooling_substeps"}))
        return std::max(1, v->get<int>());
    return 10;
}

bool IsSymmetryCutAssembly(int symang, bool symdiv, int row, int col) {
    return symang == 90 && symdiv && (row == 0 || col == 0);
}

int VisibleAssemblyRows(int symang, bool symdiv, int row, int ndivxy) {
    return (row == 0 && symang == 90 && symdiv) ? ndivxy / 2 : ndivxy;
}

int VisibleAssemblyCols(int symang, bool symdiv, int col, int ndivxy) {
    return (col == 0 && symang == 90 && symdiv) ? ndivxy / 2 : ndivxy;
}

int AssemblyNodeX(int symang, bool symdiv, int assembly_col, int local_col, int ndivxy) {
    int x = assembly_col * ndivxy + local_col;
    if (assembly_col > 0 && symang == 90 && symdiv) x -= ndivxy / 2;
    return x;
}

int AssemblyNodeY(int symang, bool symdiv, int assembly_row, int local_row, int ndivxy) {
    int y = assembly_row * ndivxy + local_row;
    if (assembly_row > 0 && symang == 90 && symdiv) y -= ndivxy / 2;
    return y;
}

// One 90-degree step of the quarter-core rotation, expressed on an assembly's
// full-size local index grid (row index i runs south, column index j runs east).
// This is the same step the shuffle's rotation=90 case applies, so the two stay
// consistent by construction.
void RotateAssemblyIndex(int& i, int& j, int ndivxy) {
    const int ti = j;
    const int tj = ndivxy - 1 - i;
    i            = ti;
    j            = tj;
}

void RotateAssemblyIndexInverse(int& i, int& j, int ndivxy) {
    const int ti = ndivxy - 1 - j;
    const int tj = i;
    i            = ti;
    j            = tj;
}

// Apply optional per-entry search and exposed convergence overrides directly to the Schedule fields.
// Internal convergence tolerances and iteration caps stay at Scheduler constexpr defaults.
void ApplyScheduleOverrides(const nlohmann::ordered_json& item, Schedule& entry) {
    if (item.contains("search"))
        entry.searchType = ParseSearchTypeString(item["search"].get<std::string>());
    entry.xenon_transient  = ReadXenonTransientOverride(item, entry.xenon_transient);
    bool has_rate_override = false;
    if (item.contains("rate")) {
        entry.rate        = item["rate"].get<double>();
        has_rate_override = true;
    }
    if (item.contains("rated power percent")) {
        entry.rate        = item["rated power percent"].get<double>();
        has_rate_override = true;
    }
    if (item.contains("rated_power_percent")) {
        entry.rate        = item["rated_power_percent"].get<double>();
        has_rate_override = true;
    }
    if (has_rate_override)
        entry.rate = std::clamp(entry.rate, kMinimumScheduleValue, kMaximumPowerRate);
    entry.actual_power = entry.rated_power * entry.rate;
    if (item.contains("max_eigen_iterations"))
        entry.max_outer_iter = item["max_eigen_iterations"].get<int>();
    if (item.contains("search_max_iter"))
        entry.max_search_iter = item["search_max_iter"].get<int>();
    if (item.contains("eigv_tolerance"))
        entry.tolerance_keff = item["eigv_tolerance"].get<double>();
    if (item.contains("search_min"))
        entry.search_low = item["search_min"].get<double>();
    if (item.contains("search_max"))
        entry.search_hi = item["search_max"].get<double>();
    if (item.contains("search_target"))
        entry.target_keff = item["search_target"].get<double>();
    if (item.contains("search_tol"))
        entry.tolerance_search = item["search_tol"].get<double>();
    if (item.contains("search_pcm_tolerance"))
        entry.tolerance_search = item["search_pcm_tolerance"].get<double>() * 1.0e-5;
    if (item.contains("search_relaxation"))
        entry.search_relaxation = item["search_relaxation"].get<double>();
    if (item.contains("search_boron_probe"))
        entry.search_boron_probe = item["search_boron_probe"].get<double>();
    if (item.contains("search_rod_probe"))
        entry.search_rod_probe = item["search_rod_probe"].get<double>();
    if (const auto* v = FirstPresentKey(
            item, {"fuel temperature rise scale", "fuel_temperature_rise_scale"}))
        entry.fuel_temp_rise_scale = v->get<double>();
}

/// @brief Write spectral-history macro delta-sigma terms for one monitored node.
void writeTermContrib(const iowriter::Node& ngrp, const XSSet& xs, int lk, int ng) {
    std::vector<XSSet::TermContribution> terms;
    xs.ResolveTermContributions(lk, terms);
    const size_t nt    = terms.size();
    const size_t ng_sz = static_cast<size_t>(ng);
    if (!nt) return;

    auto                     tc_grp = ngrp.createGroup("term_contrib");
    std::vector<std::string> isotope(nt);
    for (size_t t = 0; t < nt; ++t) {
        isotope[t] =
            terms[t].iso >= 0 ? std::string(Chiffon::Isotope::isotopeIds[terms[t].iso]) : "unknown";
    }
    tc_grp.createDataSet("isotope", isotope);
    const iowriter::Dims tc_space{nt, ng_sz};
    auto                 emit_term = [&](const std::string& name, Chiffon::XSTYPE xt) {
        std::vector<double> v(nt * ng_sz, 0.0);
        for (size_t t = 0; t < nt; ++t)
            for (int ig = 0; ig < ng; ++ig)
                v[t * ng_sz + ig] = terms[t].scalar[xt * ng + ig];
        tc_grp.createDataSet<double>(name, tc_space).write_raw(v.data());
    };
    emit_term("xstf", Chiffon::XSTF);
    emit_term("xsaf", Chiffon::XSAF);
    emit_term("xsnf", Chiffon::XSNF);
    emit_term("xskf", Chiffon::XSKF);
    emit_term("xssf", Chiffon::XSSF);
}

} // namespace

// Construct an IO helper bound to the main geometry, XS, and scheduler objects.
IO::IO(Geometry& g, XSSet& xs, Scheduler& sched)
    : _g(g), _xs(xs), _s(sched) {
}

IO::~IO() {
    // HighFive::File destruction also enters the non-thread-safe HDF5 runtime.
    // Keep exception paths safe when a Driver fails before CloseResult().  On
    // the writer-thread path the queued batches must drain FIRST -- the session
    // outlives this object through its shared_ptr, but the file handle may not
    // be touched here while a batch for it is still in flight.
    iowriter::fence(_result_session);
    for (const auto& session : _restart_sessions) iowriter::fence(session);
    Chiffon::Hdf5Guard hdf5_guard;
    if (_result_session) _result_session->file.reset();
    _restart_sessions.clear();
}

// Parse the JSON "print" block of a schedule entry into a PrintOpt.
PrintOpt IO::ParsePrintOpt(const nlohmann::ordered_json& item) {
    PrintOpt opt;
    if (!item.contains("print")) return opt;

    const auto& pr = item["print"];

    // "summary" controls whether this step appears in the summary table.
    opt.summary = pr.value("summary", true);

    // Pin-wise output keys keep legacy spellings while accepting the corrected form.
    opt.pin_info = AnyKeyTrue(pr, {"pin-wise information", "pin-wise infomration",
                                   "pin-wise power information", "pin-wise power infomration",
                                   "pin information"});

    // Optional pin-flux reconstruction switch for steps that need fmap-based flux output.
    opt.pin_flux = AnyKeyTrue(pr, {"pin flux", "pin-wise flux", "pin-wise flux information",
                                   "pin-wise flux infomration"});
    if (opt.pin_flux) opt.pin_info = true;

    // "cross-section information" triggers per-node XS dump.
    opt.xs_info = pr.value("cross-section information", false);

    // "save" triggers restart file export at this step.
    opt.save = pr.value("save", false);

    // "node monitor" is an array of flat node indices to monitor individually.
    if (pr.contains("node monitor")) {
        opt.node_monitor = pr["node monitor"].get<std::vector<int>>();
    }

    return opt;
}

// Parse the JSON schedule block and expand it into concrete Scheduler entries.
void IO::ParseSchedule(const nlohmann::ordered_json& config) {
    auto& schedule_entries = _s.schedule();
    schedule_entries.clear();

    // 1. Read global defaults from the optional top-level blocks.
    double     default_boron_ppm        = 0.0;
    double     default_fuel_temperature = 900.0;
    double     default_mod_temperature  = 600.0;
    SearchType default_search           = SearchType::KEFF;
    bool       default_xenon_transient  = false; // equilibrium Xe by default; opt in with "xenon":"transient"
    default_xenon_transient             = ReadXenonTransientOverride(config, default_xenon_transient);
    if (config.contains("default parameters")) {
        const auto& dp           = config["default parameters"];
        default_boron_ppm        = dp.value("boron_ppm", 0.0);
        default_fuel_temperature = dp.value("fuel_temperature", 900.0);
        default_mod_temperature  = dp.value("moderator_temperature", 600.0);
        default_search           = ParseSearchTypeString(dp.value("search", std::string("keff")));
        default_xenon_transient  = ReadXenonTransientOverride(dp, default_xenon_transient);
    }

    double              default_pressure           = 15.5;
    double              default_inlet_temperature  = 580.0;
    double              default_outlet_temperature = 610.0;
    double              default_mass_flow_rate     = 0.0;
    double              default_rated_power        = 520.0;
    double              default_fuel_temp_rise_scale = 1.0;
    bool                default_use_mass_flow_rate = false;
    THMode              default_th_mode            = THMode::STEADY;
    std::vector<double> default_profile_power;
    std::vector<double> default_inlet_profile;
    std::vector<double> default_outlet_profile;
    if (config.contains("TH")) {
        const auto& th_config      = config["TH"];
        default_pressure           = th_config.value("global pressure", 15.5);
        default_inlet_temperature  = th_config.value("inlet temperature", 580.0);
        default_outlet_temperature = th_config.value("outlet temperature", 610.0);
        if (const auto* v = FirstPresentKey(th_config, {"profile_power", "profile power"}))
            default_profile_power = v->get<std::vector<double>>();
        if (const auto* v = FirstPresentKey(th_config, {"inlet profile", "inlet_profile"}))
            default_inlet_profile = v->get<std::vector<double>>();
        if (const auto* v = FirstPresentKey(th_config, {"outlet profile", "Outlet profile", "outlet_profile"}))
            default_outlet_profile = v->get<std::vector<double>>();
        if (const auto* v = FirstPresentKey(th_config, {"mass flux", "mass_flux", "mass flow rate", "mass_flow_rate"})) {
            default_mass_flow_rate     = v->get<double>();
            default_use_mass_flow_rate = default_mass_flow_rate > 0.0;
        }
        default_rated_power = th_config.value("rated power", 520.0);
        if (const auto* v = FirstPresentKey(
                th_config, {"fuel temperature rise scale", "fuel_temperature_rise_scale"}))
            default_fuel_temp_rise_scale = v->get<double>();
        default_th_mode     = ParseThModeString(th_config.value("TH_mode", "steady"));
    }
    if (!default_inlet_profile.empty() || !default_outlet_profile.empty()) {
        if (default_profile_power.empty())
            throw std::runtime_error("IO: TH temperature profile needs non-empty profile_power.");
        if (!default_inlet_profile.empty() &&
            default_inlet_profile.size() != default_profile_power.size())
            throw std::runtime_error("IO: inlet profile length must match profile_power length.");
        if (!default_outlet_profile.empty() &&
            default_outlet_profile.size() != default_profile_power.size())
            throw std::runtime_error("IO: outlet profile length must match profile_power length.");
        for (std::size_t i = 1; i < default_profile_power.size(); ++i) {
            if (default_profile_power[i] <= default_profile_power[i - 1])
                throw std::runtime_error("IO: profile_power must be strictly increasing.");
        }
    }

    // 2. Seed Scheduler defaults. `rate` is always 100% for the implicit initial step.
    _s.SetDefaultCondition(default_boron_ppm, default_fuel_temperature, default_mod_temperature,
                           default_rated_power, 100.0);
    _s.SetDefaultTH(default_pressure, default_inlet_temperature, default_outlet_temperature,
                    default_mass_flow_rate, default_use_mass_flow_rate,
                    default_fuel_temp_rise_scale);
    _s.SetDefaultTHProfile(default_profile_power, default_inlet_profile, default_outlet_profile);
    _s.SetDefaultXenonTransient(default_xenon_transient);

    // Convergence block feeds exposed Scheduler defaults and a cached json applied to every entry.
    nlohmann::ordered_json convergence_defaults;
    if (config.contains("convergence")) {
        convergence_defaults = config["convergence"];
        _s.SetDefaultConvergence(
            convergence_defaults.value("max_eigen_iterations", 0),
            convergence_defaults.value("eigv_tolerance", 0.0));
    }

    // 3. Expand each JSON schedule item into one or more concrete scheduler entries.
    if (!config.contains("schedule")) return;

    for (const auto& item : config["schedule"]) {
        const std::string entry_type = item.value("type", "");
        if (entry_type.empty()) break;

        const PrintOpt   print_options = ParsePrintOpt(item);
        const SearchType entry_search  = item.contains("search")
                                             ? ParseSearchTypeString(item["search"].get<std::string>())
                                             : default_search;

        auto finalize_entry = [&](Schedule& e, bool first_in_group) {
            e.thmode = default_th_mode;
            ApplyScheduleOverrides(convergence_defaults, e);
            ApplyScheduleOverrides(item, e);
            if (first_in_group)
                e.print_opt = print_options;
            else if (!print_options.node_monitor.empty())
                e.print_opt.node_monitor = print_options.node_monitor;
        };

        if (entry_type == "depletion") {
            const int steps            = item.value("steps", 1);
            int       substeps         = std::max(1, item.value("substeps", 1));
            double    burnup_increment = 0.0;
            bool      use_burnup_time  = false;
            if (const auto* v = FirstPresentKey(item, {"burnup", "burnup_increment", "burnup increment", "bu"})) {
                burnup_increment = v->get<double>();
                use_burnup_time  = burnup_increment > kMinimumScheduleValue;
            }

            for (int step_index = 0; step_index < steps; ++step_index) {
                const double time_days  = std::max(item.value("time", kMinimumScheduleValue),
                                                   kMinimumScheduleValue);
                double       power_rate = kMinimumScheduleValue;
                if (const auto* v = FirstPresentKey(item, {"rated power percent", "rated_power_percent", "rate"}))
                    power_rate = v->get<double>();
                power_rate = std::clamp(power_rate, kMinimumScheduleValue, kMaximumPowerRate);

                _s.AddDepletionSchedule(time_days, power_rate, substeps, entry_search,
                                        burnup_increment, use_burnup_time);
                finalize_entry(_s.schedule().back(), step_index == 0);
                // MASTER %EXE_DEP tgobj-boron analogue: the entry re-queues itself at
                // runtime until the converged critical boron reaches this target.
                if (const auto* v = FirstPresentKey(item, {"until boron ppm", "until_boron_ppm"}))
                    _s.schedule().back().until_boron_ppm = v->get<double>();
            }

        } else if (entry_type == "derivative") {
            const double delta_tful = item.value("delta fuel temperature", 0.0);
            const double delta_tmod = item.value("delta mod temperature", 0.0);
            const double delta_dmod = item.value("delta mod density", 0.0);
            const double delta_bppm = item.value("delta boron ppm", 0.0);
            const double delta_xe   = item.value("delta xe density", 0.0);
            const double delta_sm   = item.value("delta sm density", 0.0);

            // Workaround: Scheduler lacks a working AddDerivativeSchedule — shape an added
            // DEPLETION entry into a DERIVATIVE entry.
            _s.AddDepletionSchedule(0.0, 100.0, 1, entry_search);
            Schedule& e  = _s.schedule().back();
            e.type       = ScheduleType::DERIVATIVE;
            e.time       = 0.0;
            e.substep    = 1;
            e.delta_tful = delta_tful;
            e.delta_tmod = delta_tmod;
            e.delta_dmod = delta_dmod;
            e.delta_bppm = delta_bppm;
            e.delta_xe   = delta_xe;
            e.delta_sm   = delta_sm;
            finalize_entry(e, true);

        } else if (entry_type == "standard") {
            // "standard" entry: an up-front configuration step. Creates a STANDARD
            // scheduler entry (one-shot solve at current state) AND promotes its own
            // settings to Scheduler defaults so subsequent entries inherit them.
            _s.AddDepletionSchedule(0.0, 100.0, 1, entry_search);
            Schedule& e = _s.schedule().back();
            e.type      = ScheduleType::STANDARD;
            e.time      = 0.0;
            e.substep   = 1;
            finalize_entry(e, true);

            // Promote settings so later entries without explicit keys pick them up.
            if (item.contains("search")) default_search = entry_search;
            default_xenon_transient = ReadXenonTransientOverride(item, default_xenon_transient);
            _s.SetDefaultXenonTransient(default_xenon_transient);
            if (item.contains("TH_mode"))
                default_th_mode = ParseThModeString(item["TH_mode"].get<std::string>());
            _s.SetDefaultConvergence(
                item.value("max_eigen_iterations", 0),
                item.value("eigv_tolerance", 0.0));
            // Merge long-tail keys (search_min/max/target, probes, etc.) into the cached
            // defaults block so they continue to be applied by finalize_entry().
            for (auto it = item.begin(); it != item.end(); ++it) {
                if (it.key() == "type" || it.key() == "print") continue;
                convergence_defaults[it.key()] = it.value();
            }

        } else if (entry_type == "rod insertion") {
            std::map<std::string, double> insertions;
            for (auto& [key, val] : item.items()) {
                if (!val.is_number())
                    continue;
                if (key == "type" || key == "rate" || key == "rated power percent" ||
                    key == "rated_power_percent" || key == "search" || key == "search_min" ||
                    key == "search_max" || key == "search_target" || key == "search_tol" ||
                    key == "search_max_iter" || key == "search_relaxation" ||
                    key == "max_eigen_iterations" || key == "eigv_tolerance" ||
                    key == "max_th_iterations" || key == "th_tolerance" ||
                    key == "minimum_keff" || key == "minimum_carry_slope" ||
                    key == "search_minimum_secant_denominator" || key == "search_minimum_span" ||
                    key == "search_pcm_tolerance" ||
                    key == "search_slope_freeze_dx_threshold" || key == "print")
                    continue;
                insertions[key] = val.get<double>();
            }

            // Workaround: Scheduler has no AddRodSchedule — reshape an added DEPLETION entry.
            _s.AddDepletionSchedule(0.0, 100.0, 1, entry_search);
            Schedule& e      = _s.schedule().back();
            e.type           = ScheduleType::ROD;
            e.time           = 0.0;
            e.substep        = 1;
            e.rod_insertions = insertions;
            finalize_entry(e, true);
        }
    }

    // Ensure at least one entry exists so Driver can index schedule(0) for InitXS.
    if (_s.schedule().empty()) {
        _s.AddDepletionSchedule(0.0, 100.0, 1, default_search);
        Schedule& e = _s.schedule().back();
        e.type      = ScheduleType::STANDARD;
        e.time      = 0.0;
        e.substep   = 1;
        e.thmode    = default_th_mode;
        ApplyScheduleOverrides(convergence_defaults, e);
    }
}

// Read one JSON input deck and initialize geometry, XS, and scheduler state.
void IO::ReadInput(const std::string& filepath, const std::string& statepoint_grid) {
    // THE GUARD IS NOT HELD HERE, AND THAT IS THE POINT.
    //
    // It used to wrap this whole function -- "restart reads, shuffle reads and
    // XSSet::Initialize as one transaction".  But the guard exists for one
    // reason only, that HDF5 1.10.x is not thread-safe (Hdf5Guard.h); nothing
    // in this function needs a transaction, and the transaction is what turned
    // eight concurrent workers into a queue.  Under it sat the 34 MB CHIFFON
    // parse, per deck, per worker: Init+IO 4.108 .. 14.506 s across a width-8
    // wave, ~1.49 s of slope per case, 63.8 s of lock wait for eight decks
    // (GA evaluator plan Sec 3.1(a)).
    //
    // The parse itself is now shared (XsLibrary.h), so the second worker has
    // nothing to parse.  What is left here is JSON parsing, the geometry build
    // and the nxyz-sized allocations -- pure per-deck host work that touches no
    // HDF5 and has no business being serialised.  So the guard moves DOWN, onto
    // the four places that actually enter the HDF5 runtime:
    //
    //   * the restart EFPD probe below,
    //   * the shuffle-spec source reads,
    //   * LoadGeometryFromRestart and ApplyShuffle (both already take their own
    //     guard -- it is recursive, so a caller's scope nests harmlessly),
    //   * the trailing restart-state restore.
    //
    // XSSet::Initialize needs none: Importer::LoadHDF takes its own guard, and
    // a cache hit never reaches it.
    //
    // Sec 2.2's `T_percase` (1.1 s) had no breakdown either -- Init+IO was one
    // number covering deck parse, geometry build, library acquire, the
    // nxyz-sized allocations and OpenResult.  These five stamps are the
    // breakdown, printed once per deck as [RASBERY][READINPUT].
    const auto t_enter = std::chrono::steady_clock::now();
    auto       t_deck  = t_enter;
    auto       t_geom  = t_enter;
    auto       t_xs    = t_enter;

    // 1. Resolve the input directory and load the JSON deck.
    {
        std::filesystem::path input_path(filepath);
        _input_dir = input_path.parent_path().string();
        if (!_input_dir.empty() && _input_dir.back() != '/' && _input_dir.back() != '\\')
            _input_dir += '/';
    }

    std::ifstream input_stream(filepath);
    if (!input_stream.is_open())
        throw std::runtime_error("IO::ReadInput: cannot open input file: " + filepath);
    nlohmann::ordered_json config;
    input_stream >> config;

    // WP10.3: THE BURNUP GRID, BEFORE THE DIGEST AND BEFORE ANY CONSUMER.
    //
    // This is the L3coarse lane, and it is the one fidelity RunContract.h says
    // "cannot be detected" -- it is a DECK property, not an environment one.
    // Applying it here makes it a deck property in the only sense that matters:
    // the coarse schedule is what the case key is taken of two statements
    // below, so a ten-statepoint screening answer and a thirty-five-statepoint
    // acceptance answer of the same candidate cannot land on one key.  Empty is
    // the deck as written and is one string compare -- feature-off is the old
    // path, byte for byte.
    //
    // A REFUSAL, not a silent pass-through, when the grid cannot be applied: a
    // deck with no depletion entry run under a screening declaration would be a
    // full-cost case filed in the screening lane, which is the mirror image of
    // the defect WP1's contract exists for.
    if (!rasbery::spgrid::isFullGrid(statepoint_grid)) {
        std::string grid_error;
        if (!rasbery::spgrid::applyGridSpec(config, statepoint_grid, grid_error))
            throw std::runtime_error("IO::ReadInput: " + grid_error);
    }

    // WP10.1: fold the DECK half of the canonical case key HERE, and nowhere
    // else, because this is the only place the parsed deck exists.  Two
    // loading patterns related by a symmetry operation of the core fold to one
    // digest; everything else in the deck goes in verbatim.  Cost is one walk
    // of a deck-sized JSON tree, once per case, next to the parse that produced
    // it -- and it is unconditional, because a key that only some runs computed
    // is a key no cache could trust.
    _deck_key_digest =
        Sha256::hexOf(casekey::deckPayload(config, &_deck_key_core_op));

    _xs_path.clear();
    _restart_path.clear();
    _restart_files.clear();
    _restart_cooling_days.clear();
    _restart_cooling_substeps.clear();
    _primary_restart_cycle = 1;
    if (config.contains("data")) {
        _xs_path = NormalizeInputPath(config["data"].value("cross-section", ""));

        if (!_xs_path.empty()) {
            std::filesystem::path xs_path(_xs_path);
            if (!xs_path.is_absolute()) {
                _xs_path = (std::filesystem::path(_input_dir) / xs_path).lexically_normal().string();
            }
        }

        if (config["data"].contains("restart")) {
            const auto& rv = config["data"]["restart"];

            if (rv.is_string()) {
                RegisterRestartEntry(_restart_files, _restart_path, _primary_restart_cycle,
                                     _input_dir, 1, rv.get<std::string>());
            } else if (rv.is_object()) {
                const int         cycle = rv.value("cycle", 1);
                const std::string path  = rv.value("path", "");
                if (path.empty()) {
                    throw std::runtime_error("IO::ReadInput: restart object missing \"path\".");
                }
                RegisterRestartEntry(_restart_files, _restart_path, _primary_restart_cycle,
                                     _input_dir, cycle, path);
                _restart_cooling_days[cycle]     = RestartCoolingDays(rv);
                _restart_cooling_substeps[cycle] = RestartCoolingSubsteps(rv);
            } else if (rv.is_array()) {
                for (const auto& entry : rv) {
                    const int         cycle = entry.value("cycle", 1);
                    const std::string path  = entry.value("path", "");
                    if (path.empty()) {
                        throw std::runtime_error("IO::ReadInput: restart array entry missing \"path\".");
                    }
                    RegisterRestartEntry(_restart_files, _restart_path, _primary_restart_cycle,
                                         _input_dir, cycle, path);
                    _restart_cooling_days[cycle]     = RestartCoolingDays(entry);
                    _restart_cooling_substeps[cycle] = RestartCoolingSubsteps(entry);
                }
            } else {
                throw std::runtime_error(
                    "IO::ReadInput: \"restart\" must be a string, object, or array.");
            }
        }
    }

    if (!_restart_files.empty()) {
        Chiffon::Hdf5Guard hdf5_guard;
        const std::string& efpd_restart_path = _restart_files.rbegin()->second;
        HighFive::File     rfile(efpd_restart_path, HighFive::File::ReadOnly);
        if (rfile.exist("metadata")) {
            rfile.getGroup("metadata").getDataSet("efpd").read(_restart_efpd);
        }
    }

    // 2. Parse the schedule and geometry definitions.
    ParseSchedule(config);

    GeometryInput geometry_input;
    if (config.contains("geometry")) {
        auto geom             = config["geometry"];
        geometry_input.ng     = geom["dimensions"]["ng"];
        geometry_input.ndivxy = geom["dimensions"]["xydivision"];
        geometry_input.npins  = geom["dimensions"]["npins"];
        // The fuel-rod count per ASSEMBLY, optional.  Absent (0) resolves to the
        // legacy 62 rods/node divisor; see src/ThFuelRods.h for why the default
        // is the wrong number and why it is still the default.  Spelled `nfrod`
        // like MASTER's depf card ("npin, nfrod"), with the long form accepted
        // the way every other deck key here accepts one.
        if (const auto* v = FirstPresentKey(geom["dimensions"],
                                            {"nfrod", "fuel rods per assembly",
                                             "fuel_rods_per_assembly"})) {
            geometry_input.nfrod = v->get<int>();
            if (geometry_input.nfrod <= 0)
                throw std::runtime_error(
                    "IO::ReadInput: geometry.dimensions.nfrod must be a positive fuel-rod "
                    "count per assembly.");
        }
        // WHICH FUEL-TEMPERATURE TABLE.  Optional; absent resolves to the
        // shipped WH grid (tf.csv), which is what every published number in
        // this campaign was produced with.  Declaring it here does NOT move
        // the run: RASBERY_TH_TF_TABLE=deck does.  src/ThTfTable.h says why.
        {
            const auto* from_geom = FirstPresentKey(
                geom["dimensions"], {kTfTableKeys[0], kTfTableKeys[1], kTfTableKeys[2],
                                     kTfTableKeys[3]});
            const nlohmann::ordered_json* from_defaults = nullptr;
            if (config.contains("default parameters"))
                from_defaults = FirstPresentKey(
                    config["default parameters"],
                    {kTfTableKeys[0], kTfTableKeys[1], kTfTableKeys[2], kTfTableKeys[3]});
            if (from_geom != nullptr && from_defaults != nullptr)
                throw std::runtime_error(
                    "IO::ReadInput: \"tf table\" is declared BOTH under "
                    "geometry.dimensions and under \"default parameters\".  Two "
                    "declarations are two intents; delete one.");
            if (from_geom != nullptr)
                geometry_input.tf_table =
                    ParseTfTableSpec(*from_geom, "geometry.dimensions.\"tf table\"");
            else if (from_defaults != nullptr)
                geometry_input.tf_table =
                    ParseTfTableSpec(*from_defaults, "\"default parameters\".\"tf table\"");
        }
        geometry_input.hx     = geom["size"]["hx"];
        geometry_input.hy     = geom["size"]["hy"];

        {
            const auto& hz_src = geom.contains("size") && geom["size"].contains("hz")
                                     ? geom["size"]["hz"]
                                     : config["hz"];
            geometry_input.hz.clear();
            for (const auto& entry : hz_src) {
                if (entry.is_object()) {
                    double h = entry["height"];
                    int    n = entry.value("node", 1);
                    geometry_input.hz.insert(geometry_input.hz.end(), n, h);
                } else {
                    geometry_input.hz.push_back(entry.get<double>());
                }
            }
        }
        std::reverse(geometry_input.hz.begin(), geometry_input.hz.end());
        geometry_input.nz = static_cast<int>(geometry_input.hz.size());

        geometry_input.symang = geom["symmetry"]["angle"];
        geometry_input.symopt = geom["symmetry"]["mirror"];
        geometry_input.symdiv = geom["symmetry"]["center assembly divided"];
        geometry_input.albedo = {
            geom["albedo"]["west"], geom["albedo"]["east"],
            geom["albedo"]["north"], geom["albedo"]["south"],
            geom["albedo"]["bottom"], geom["albedo"]["up"]};

        geometry_input.core = config["core"].get<std::vector<std::vector<std::string>>>();

        std::map<std::string, std::vector<std::string>> batches;
        for (const auto& [name, layers] : config["batch"].items()) {
            std::vector<std::string> zlayer;
            for (const auto& item : layers) {
                std::string id    = item["id"];
                int         count = item["count"];
                zlayer.insert(zlayer.end(), count, id);
            }
            batches.emplace(name, std::move(zlayer));
        }
        geometry_input.batch = batches;

        _shuffle_specs.clear();
        // One guard over the whole shuffle scan, not one per dataset: the
        // HighFive::File below is read in two places and its DESTRUCTOR also
        // enters the HDF5 runtime.  Still far narrower than the function-wide
        // guard this replaced, and a deck with no shuffle never enters here.
        Chiffon::Hdf5Guard shuffle_guard;
        for (int row = 0; row < static_cast<int>(geometry_input.core.size()); ++row) {
            for (int col = 0; col < static_cast<int>(geometry_input.core[row].size()); ++col) {
                ShuffleSpec spec;
                if (!TryParseShuffleEntry(geometry_input.core[row][col], row, col, spec))
                    continue;

                if (_restart_files.count(spec.cycle) == 0)
                    throw std::runtime_error(std::format(
                        "IO: shuffle at ({},{}) references cycle {} which is not in \"restart\".",
                        row, col, spec.cycle));

                HighFive::File src(_restart_files.at(spec.cycle), HighFive::File::ReadOnly);
                auto           src_geo = src.getGroup("geometry");

                std::vector<int>         row_lens;
                std::vector<std::string> core_flat;
                src_geo.getDataSet("core_row_len").read(row_lens);
                src_geo.getDataSet("core_flat").read(core_flat);

                if (spec.source_row >= static_cast<int>(row_lens.size()) ||
                    spec.source_col >= row_lens[spec.source_row])
                    throw std::runtime_error(std::format(
                        "IO: shuffle source ({},{}) is out of bounds.", spec.source_row, spec.source_col));

                int flat_idx = 0;
                for (int r = 0; r < spec.source_row; ++r)
                    flat_idx += row_lens[r];
                const std::string assembly_name = core_flat[flat_idx + spec.source_col];

                geometry_input.core[row][col] = assembly_name;

                if (geometry_input.batch.count(assembly_name) == 0) {
                    std::vector<std::string> bkeys, bflat;
                    std::vector<int>         blen;
                    src_geo.getDataSet("batch_keys").read(bkeys);
                    src_geo.getDataSet("batch_len").read(blen);
                    src_geo.getDataSet("batch_flat").read(bflat);
                    int bidx = 0;
                    for (size_t i = 0; i < bkeys.size(); ++i) {
                        if (bkeys[i] == assembly_name) {
                            geometry_input.batch[assembly_name] = std::vector<std::string>(
                                bflat.begin() + bidx,
                                bflat.begin() + bidx + blen[i]);
                            break;
                        }
                        bidx += blen[i];
                    }
                }
                _shuffle_specs.push_back(spec);
            }
        }
    } else if (!_restart_path.empty()) {
        geometry_input = LoadGeometryFromRestart(_restart_path);
    } else {
        throw std::runtime_error(
            "IO::ReadInput: 'geometry' block missing and no restart file specified.");
    }

    // 3. Initialize Geometry, XS, and optional rod maps from the parsed input.
    _gin = geometry_input;

    t_deck = std::chrono::steady_clock::now();
    _g.Initialize(geometry_input);
    t_geom = std::chrono::steady_clock::now();

    if (_xs_path.empty()) {
        std::filesystem::path in(filepath);
        _xs_path = (in.parent_path() / "results.h5").string();
    }

    // WP8 stage 2: attach this case to its COHORT.
    //
    // HERE, and not earlier, because the cohort key is a function of
    // `geometry_input` AS BUILT -- after the shuffle resolver rewrote `core` in
    // place and after a missing geometry block was recovered from a restart --
    // and of the cross-section library's CONTENT, which needs `_xs_path`
    // resolved.  Earlier would key on something Geometry was not built from;
    // later would be after XSSet::Initialize, which is the first thing that
    // could want cohort state.
    //
    // The library digest is memoised by (path, size, mtime) inside XsLibrary,
    // and XSSet::Initialize on the next line asks for the same value, so this
    // costs one file read for the whole process and a vector scan per case.
    _cohort = cohort::acquire(cohort::Descriptor{
        Sha256::hexOf(cohort::geometryPayload(geometry_input)),
        XsLibraryContentDigest(_xs_path),
        geometry_input.ng, geometry_input.ndivxy, geometry_input.npins});

    _xs.Initialize(_xs_path);
    t_xs = std::chrono::steady_clock::now();
    if (config.contains("axial_rod_division")) {
        _xs.SetAxialRodDivision(config["axial_rod_division"].get<int>());
    } else if (config.contains("rod cusping")) {
        const auto& cusping = config["rod cusping"];
        if (cusping.contains("axial_rod_division"))
            _xs.SetAxialRodDivision(cusping["axial_rod_division"].get<int>());
        else if (cusping.contains("axial rod division"))
            _xs.SetAxialRodDivision(cusping["axial rod division"].get<int>());
    }
    if (!_shuffle_specs.empty()) ApplyShuffle();

    // Parse rod configuration and rod map → store into XSSet as RodGroup
    std::map<std::string, std::vector<double>> rod_profiles;
    if (config.contains("rod configuration")) {
        const auto& rod_conf = config["rod configuration"];
        for (const auto& [name, data] : rod_conf.items()) {
            auto& grp = _xs.rod_groups()[name];
            if (data.contains("ctype") && data["ctype"].is_array()) {
                grp.ctype_segments = data["ctype"].get<std::vector<int>>();
                if (!data.contains("length") || !data["length"].is_array())
                    throw std::runtime_error(std::format(
                        "IO: rod configuration \"{}\" uses segmented ctype without length.", name));
                grp.length_segments = data["length"].get<std::vector<double>>();
                if (grp.ctype_segments.size() != grp.length_segments.size())
                    throw std::runtime_error(std::format(
                        "IO: rod configuration \"{}\" ctype and length arrays have different sizes.", name));
                if (grp.ctype_segments.empty())
                    throw std::runtime_error(std::format(
                        "IO: rod configuration \"{}\" ctype array is empty.", name));
                for (double length : grp.length_segments) {
                    if (length <= 0.0)
                        throw std::runtime_error(std::format(
                            "IO: rod configuration \"{}\" has non-positive segment length.", name));
                }
                grp.ctype = grp.ctype_segments.back();
            } else {
                grp.ctype = data.value("ctype", 1);
                grp.ctype_segments.clear();
                grp.length_segments.clear();
            }
            if (data.contains("profile"))
                rod_profiles[name] = data["profile"].get<std::vector<double>>();
        }
    }

    if (config.contains("rod map")) {
        std::vector<std::vector<std::string>> rod_map = config["rod map"];

        const int  ndivxy = _g.ndivxy();
        const int  symang = _g.symang();
        const bool symdiv = _g.symdiv();

        for (size_t row = 0; row < rod_map.size(); row++) {
            for (size_t col = 0; col < rod_map[row].size(); col++) {
                const std::string& name = rod_map[row][col];
                if (name == "XX") continue;

                int rowend = (row == 0 && symang == 90 && symdiv) ? ndivxy / 2 : ndivxy;
                int colend = (col == 0 && symang == 90 && symdiv) ? ndivxy / 2 : ndivxy;
                for (int j = 0; j < rowend; j++) {
                    for (int i = 0; i < colend; i++) {
                        int idx = static_cast<int>(col) * ndivxy + i;
                        int idy = static_cast<int>(row) * ndivxy + j;
                        if (col > 0 && symang == 90 && symdiv) idx -= (ndivxy / 2);
                        if (row > 0 && symang == 90 && symdiv) idy -= (ndivxy / 2);
                        int l = _g.ijtol(idx, idy);
                        if (l >= 0)
                            _xs.rod_groups()[name].xy_nodes.push_back(l);
                    }
                }
            }
        }
    }

    if (!rod_profiles.empty())
        _xs.BuildRodProfileMatrix(rod_profiles);

    // 4. Restore restart-state fields after the model library is ready.
    if (!_restart_path.empty() && _shuffle_specs.empty()) {
        Chiffon::Hdf5Guard hdf5_guard;
        const int nxyz = _g.nxyz();
        const int niso = static_cast<int>(Chiffon::Isotope::niso);

        HighFive::File rfile(_restart_path, HighFive::File::ReadOnly);

        if (rfile.exist("metadata")) {
            rfile.getGroup("metadata").getDataSet("efpd").read(_restart_efpd);
        }

        {
            std::vector<int> burn;
            rfile.getDataSet("burnup").read(burn);
            if (static_cast<int>(burn.size()) != nxyz)
                throw std::runtime_error(
                    std::format("Restart: burnup size {} != nxyz {}", burn.size(), nxyz));
            std::copy_n(burn.data(), nxyz, _xs.burn_data());
        }

        if (rfile.exist("axial_rod_division")) {
            int saved_division = _xs.axial_rod_division();
            rfile.getDataSet("axial_rod_division").read(saved_division);
            _xs.SetAxialRodDivision(saved_division);
        }

        if (rfile.exist("rod_state")) {
            const auto               state = rfile.getGroup("rod_state");
            std::vector<std::string> names;
            std::vector<double>      insertions;
            state.getDataSet("names").read(names);
            state.getDataSet("insertions").read(insertions);
            if (names.size() != insertions.size())
                throw std::runtime_error(
                    "Restart: rod-state name and insertion counts differ.");

            std::map<std::string, double> saved;
            for (size_t i = 0; i < names.size(); ++i) {
                if (_xs.rod_groups().count(names[i]) == 0)
                    throw std::runtime_error(
                        "Restart: rod group '" + names[i] +
                        "' is absent from the input configuration.");
                saved.emplace(names[i], insertions[i]);
            }
            _xs.SetRod(saved);
        }

        // RDPL restart state. The v2.4.0 `fine_rod_fluence` was a total-group fluence;
        // the RDPL axis is the THERMAL-group fluence, so a legacy file cannot seed it.
        if (rfile.exist("fine_rod_thermal_fluence")) {
            std::vector<double> saved;
            rfile.getDataSet("fine_rod_thermal_fluence").read(saved);
            auto& current = _xs.fine_rod_thermal_fluence_data();
            if (saved.size() != current.size())
                throw std::runtime_error(std::format(
                    "Restart: fine_rod_thermal_fluence size {} != expected {}",
                    saved.size(), current.size()));
            current = std::move(saved);
        } else if (rfile.exist("fine_rod_fluence")) {
            PLOG_WARNING
                << "Restart: legacy fine_rod_fluence contains total fluence and cannot "
                   "initialize the thermal-fluence RDPL coordinate; starting that coordinate at zero.";
        }

        if (niso > 0 && rfile.exist("isotope_density")) {
            std::vector<double> iden_flat(static_cast<size_t>(nxyz) * niso, 0.0);
            auto                      density = rfile.getDataSet("isotope_density");
            const std::vector<size_t> expectedShape = {
                static_cast<size_t>(nxyz), static_cast<size_t>(niso)};
            if (density.getDimensions() != expectedShape ||
                density.getElementCount() != iden_flat.size())
                throw std::runtime_error(
                    "Restart: isotope_density dimensions do not match nxyz and niso.");
            density.read_raw<double>(iden_flat.data());
            const double cooling_days = _restart_cooling_days[_primary_restart_cycle];
            if (cooling_days > 0.0) {
                const int substeps = _restart_cooling_substeps[_primary_restart_cycle];
                _xs.DecayIsotopeDensityFlat(iden_flat, nxyz, cooling_days, substeps);
                PLOG_INFO << std::format("Restart: isotope densities cooled for {:.6g} days in {} substeps.",
                                         cooling_days, substeps);
            }
            for (int l = 0; l < nxyz; ++l) {
                std::vector<double> node_iden(
                    iden_flat.begin() + static_cast<ptrdiff_t>(l) * niso,
                    iden_flat.begin() + static_cast<ptrdiff_t>(l + 1) * niso);
                _xs.setNodeIden(l, node_iden);
            }
        }

        if (rfile.exist("th_state")) {
            auto                thst = rfile.getGroup("th_state");
            std::vector<double> bppm, tful, tmod, dmod;
            thst.getDataSet("bppm").read(bppm);
            thst.getDataSet("tful").read(tful);
            thst.getDataSet("tmod").read(tmod);
            thst.getDataSet("dmod").read(dmod);
            if (static_cast<int>(bppm.size()) != nxyz || static_cast<int>(tful.size()) != nxyz ||
                static_cast<int>(tmod.size()) != nxyz || static_cast<int>(dmod.size()) != nxyz)
                throw std::runtime_error(std::format("Restart: th_state size != nxyz {}", nxyz));
            for (int l = 0; l < nxyz; ++l) {
                _g.bppm(l) = bppm[l];
                _g.tful(l) = tful[l];
                _g.tmod(l) = tmod[l];
                _g.dmod(l) = dmod[l];
            }
        }

        if (!config.contains("TH") && rfile.exist("th_params")) {
            auto   thp = rfile.getGroup("th_params");
            double pres, inlet, outlet, power;
            double flow     = 0.0;
            int    use_flow = 0;
            thp.getDataSet("pressure").read(pres);
            thp.getDataSet("inlet_temp").read(inlet);
            thp.getDataSet("outlet_temp").read(outlet);
            thp.getDataSet("rated_power").read(power);
            if (thp.exist("mass_flow_rate")) thp.getDataSet("mass_flow_rate").read(flow);
            if (thp.exist("use_mass_flow_rate")) thp.getDataSet("use_mass_flow_rate").read(use_flow);

            for (auto& sched : _s.schedule()) {
                sched.pressure           = pres;
                sched.inlet_temp         = inlet;
                sched.outlet_temp        = outlet;
                sched.rated_power        = power;
                sched.mass_flow_rate     = flow;
                sched.use_mass_flow_rate = use_flow != 0 && flow > 0.0;
            }
            _g.pressure()           = pres;
            _g.inlet_temp()         = inlet;
            _g.outlet_temp()        = outlet;
            _g.mass_flow_rate()     = flow;
            _g.use_mass_flow_rate() = use_flow != 0 && flow > 0.0;
            _g.rated_power()        = power;

            PLOG_INFO << std::format(
                "Restart: TH params restored  pressure={:.2f} MPa  inlet={:.2f} K"
                "  outlet={:.2f} K  mass_flux={:.3f} kg/s/m^2  power={:.1f} MW",
                pres, inlet, outlet, flow, power);
        }

        if (rfile.exist("flux")) {
            const size_t        sz = static_cast<size_t>(_g.ng()) * nxyz;
            std::vector<double> flux;
            rfile.getDataSet("flux").read(flux);
            if (flux.size() == sz)
                std::copy_n(flux.data(), sz, _g.PhifMutable());
        }

        PLOG_INFO << "Restart: burnup, isotope densities and flux loaded from " << _restart_path;
    }

    // The Init+IO breakdown (GA evaluator plan Sec 2.2 Task C).  `deck_s`
    // includes the JSON parse and the schedule/geometry parse; `geometry_s` is
    // Geometry::Initialize; `xs_s` is XSSet::Initialize, whose library half is
    // the cached parse and whose other half is the nxyz-sized allocations;
    // `rest_s` is rod maps, shuffle and the restart restore.  One line per
    // deck, beside [TIMING] Init+IO, so a batch shows all M of them.
    {
        const auto to_s = [](auto a, auto b) {
            return std::chrono::duration<double>(b - a).count();
        };
        const auto t_exit = std::chrono::steady_clock::now();
        std::cout << std::format(
            "[RASBERY][READINPUT] {{\"deck_s\":{:.3f},\"geometry_s\":{:.3f},"
            "\"xs_s\":{:.3f},\"rest_s\":{:.3f},\"total_s\":{:.3f}}}\n",
            to_s(t_enter, t_deck), to_s(t_deck, t_geom), to_s(t_geom, t_xs),
            to_s(t_xs, t_exit), to_s(t_enter, t_exit));
    }
}

// Store the calculated results of one solved schedule step back into Scheduler.
void IO::AddResult(Geometry& g, double keff,
                   int schedule_index, int step_no, double efpd) {
    if (schedule_index < 0 || schedule_index >= static_cast<int>(_s.schedule().size())) return;

    Schedule& d = _s.schedule()[schedule_index];

    d.step = step_no;
    d.efpd = efpd;
    d.eigv = keff;
    d.rho  = (keff > 1.0e-20) ? (keff - 1.0) / keff : 0.0;
    d.ppm  = _g.bppm(0);

    // Snapshot current rod insertion depths from XSSet. Covers ROD-type entries,
    // RODCRIT-search entries (which update XSSet via SetRod(step) but not the map),
    // and any step that merely inherits a prior rod configuration.
    d.rod_insertions.clear();
    for (const auto& [name, group] : _xs.rod_groups())
        d.rod_insertions[name] = group.insertion;

    int nxy  = g.nxy();
    int nxyz = g.nxyz();
    int ng   = g.ng();
    int kbc  = g.kbc();
    int kec  = g.kec();
    int nxya = g.nxya();

    // Half-core split used by every axial offset below.  The active stack is
    // [kbc, kec); its geometric midplane generally falls INSIDE a node -- with
    // the 25 equal 15.24 cm nodes of APR1400 it bisects the 13th -- and that
    // node belongs half to each half-core.  bot_frac[k] is the fraction of node
    // k lying below the midplane, so an odd stack is handled exactly and an
    // even one degenerates to the plain 0/1 split.
    //
    // Assigning the central node wholly to one side (the previous
    // `k < (kbc+kec)/2` test) biases the reported AO by the central node's
    // power share, +0.037..+0.056 for APR1400, and compares two half-cores of
    // unequal height (182.88 vs 198.12 cm).  MASTER splits it 50/50; that is
    // also the only definition consistent with AO's own geometry.
    std::vector<double> bot_frac(g.nz(), 0.0);
    {
        double h_tot = 0.0;
        for (int k = kbc; k < kec; ++k)
            h_tot += g.hz(k);
        const double h_half = 0.5 * h_tot;
        double       h_lo   = 0.0;
        for (int k = kbc; k < kec; ++k) {
            const double hz = g.hz(k);
            if (hz <= 0.0) continue;
            bot_frac[k] = std::min(1.0, std::max(0.0, (h_half - h_lo) / hz));
            h_lo += hz;
        }
    }

    // 1. Build node-wise power data used by the rest of the summary metrics.
    double              p_tot = 0.0, v_tot = 0.0;
    std::vector<double> pdens(nxyz, 0.0);
    for (int lk = 0; lk < nxyz; ++lk) {
        double vol = g.vol(lk);
        if (!g.IsFuel(lk)) continue;
        double p = 0.0;
        for (int ig = 0; ig < ng; ++ig)
            p += _xs.xskf(ig, lk) * g.Phif()[lk * ng + ig];
        pdens[lk] = p;
        p_tot += p * vol;
        v_tot += vol;
    }
    double p_avg = (v_tot > 0.0 && p_tot > 1.0e-30) ? p_tot / v_tot : 1.0;

    {
        double fqn = 0.0;
        for (int lk = 0; lk < nxyz; ++lk)
            fqn = std::max(fqn, pdens[lk] / p_avg);
        d.fqn = fqn;
    }

    {
        double bu_sum = 0.0, hm_sum = 0.0;
        for (int k = kbc; k < kec; ++k)
            for (int l = 0; l < nxy; ++l) {
                int          lk = k * nxy + l;
                const double hm = _xs.NodeHeavyMetalMassGrams(lk);
                if (hm <= 1.0e-20) continue;
                bu_sum += static_cast<double>(_xs.burn(lk)) * hm;
                hm_sum += hm;
            }
        d.bu_avg = (hm_sum > 0.0) ? bu_sum / (hm_sum * 1000.0) : 0.0;
    }

    // 2. Collapse node data into axial, assembly, isotope, and TH summaries.
    {
        std::vector<double> ax_pv(g.nz(), 0.0);
        std::vector<double> ax_v(g.nz(), 0.0);
        for (int k = 0; k < g.nz(); ++k) {
            for (int l = 0; l < nxy; ++l) {
                int    lk  = k * nxy + l;
                double vol = g.vol(lk);
                if (vol <= 1.0e-20 || !g.IsFuel(lk)) continue;
                ax_pv[k] += (pdens[lk] / p_avg) * vol;
                ax_v[k] += vol;
            }
        }
        d.ax_power.resize(g.nz(), 0.0);
        for (int k = 0; k < g.nz(); ++k) {
            if (ax_v[k] > 0.0) d.ax_power[k] = ax_pv[k] / ax_v[k];
        }
        double p_bot = 0.0, p_top = 0.0;
        for (int k = kbc; k < kec; ++k) {
            double       pw = d.ax_power[k] * g.hz(k);
            const double fb = bot_frac[k];
            // A node lying wholly on one side is accumulated unscaled, so on a
            // mesh whose midplane falls on a node boundary (iSMR: 24 x 10 cm)
            // every term is the one the old code summed.  The two half-sums can
            // still differ in the last bit there, because the loop is no longer
            // splittable into two contiguous ranges and the reduction order
            // changes; measured at 1.2e-16 on AO, xe_ao/sm_ao bit-identical.
            if (fb >= 1.0)
                p_bot += pw;
            else if (fb <= 0.0)
                p_top += pw;
            else {
                p_bot += pw * fb;
                p_top += pw * (1.0 - fb);
            }
        }
        d.ao  = (p_bot + p_top > 0.0) ? (p_top - p_bot) / (p_top + p_bot) : 0.0;
        d.asi = -d.ao;
    }

    {
        std::vector<double> tf_s(g.nz(), 0.0), tf_v(g.nz(), 0.0);
        std::vector<double> tm_s(g.nz(), 0.0), tm_v(g.nz(), 0.0);
        std::vector<double> dm_s(g.nz(), 0.0), dm_v(g.nz(), 0.0);
        for (int k = kbc; k < kec; ++k)
            for (int l = 0; l < nxy; ++l) {
                int    lk  = k * nxy + l;
                double vol = g.vol(lk);
                if (vol <= 1.0e-20 || !g.IsFuel(lk)) continue;
                tf_s[k] += _g.tful(lk) * vol;
                tm_s[k] += _g.tmod(lk) * vol;
                dm_s[k] += _g.dmod(lk) * vol;
                tf_v[k] += vol;
                tm_v[k] += vol;
                dm_v[k] += vol;
            }
        d.ax_tful.resize(g.nz(), 0.0);
        d.ax_tmod.resize(g.nz(), 0.0);
        d.ax_dmod.resize(g.nz(), 0.0);
        for (int k = kbc; k < kec; ++k) {
            if (tf_v[k] > 0.0) d.ax_tful[k] = tf_s[k] / tf_v[k];
            if (tm_v[k] > 0.0) d.ax_tmod[k] = tm_s[k] / tm_v[k];
            if (dm_v[k] > 0.0) d.ax_dmod[k] = dm_s[k] / dm_v[k];
        }
    }

    {
        std::vector<double> ap(nxya, 0.0), av(nxya, 0.0);
        std::vector<double> an(nxya, 0.0), aa(nxya, 0.0);
        std::vector<double> ab(nxya, 0.0), ahm(nxya, 0.0);
        std::vector<double> rp(nxy, 0.0), rv(nxy, 0.0);

        for (int k = kbc; k < kec; ++k)
            for (int l = 0; l < nxy; ++l) {
                int la = g.ltola(l);
                if (la < 0 || la >= nxya) continue;
                int    lk  = k * nxy + l;
                double vol = g.vol(lk);
                if (vol <= 1.0e-20 || !g.IsFuel(lk)) continue;
                double       p  = pdens[lk];
                double       nf = 0.0, af = 0.0;
                const double hm = _xs.NodeHeavyMetalMassGrams(lk);
                for (int ig = 0; ig < ng; ++ig) {
                    double phi = g.Phif()[lk * ng + ig];
                    nf += _xs.xsnf(ig, lk) * phi;
                    af += _xs.xsaf(ig, lk) * phi;
                }
                ap[la] += p * vol;
                av[la] += vol;
                rp[l] += p * vol;
                rv[l] += vol;
                an[la] += nf * vol;
                aa[la] += af * vol;
                if (hm > 1.0e-20) {
                    ab[la] += static_cast<double>(_xs.burn(lk)) * hm;
                    ahm[la] += hm;
                }
            }

        d.asm_type.resize(nxya);
        d.asm_power.resize(nxya, 0.0);
        d.asm_burn.resize(nxya, 0.0);
        d.asm_kinf.resize(nxya, 0.0);
        double frn = 0.0;

        // MASTER's FRN is the maximum axially integrated radial *nodal*
        // power, not the maximum assembly-average power.  With xydivision=2
        // the old assembly collapse erased the intra-assembly radial peak.
        for (int l = 0; l < nxy; ++l)
            if (rv[l] > 0.0)
                frn = std::max(frn, (rp[l] / rv[l]) / p_avg);

        for (int la = 0; la < nxya; ++la) {
            // `_asmb` is stored per assembly-plane (la + nxya*k).  Indexing it
            // with `la` alone selects the top reflector layer (RT) and mislabeled
            // every fuel assembly in result files.  Identify the model from an
            // active fuel subnode instead.
            size_t mi = _xs.models().size();
            for (int li = 0; li < g.ndivxy() * g.ndivxy(); ++li) {
                const int l = g.latol(li, la);
                if (l < 0) continue;
                const int lk = kbc * nxy + l;
                if (g.IsFuel(lk)) {
                    mi = _xs.comp(lk);
                    break;
                }
            }
            d.asm_type[la] = (mi < _xs.models().size()) ? _xs.models()[mi].name() : "?";
            if (av[la] > 0.0) {
                d.asm_power[la] = (ap[la] / av[la]) / p_avg;
            }
            if (aa[la] > 1.0e-30)
                d.asm_kinf[la] = an[la] / aa[la];
            if (ahm[la] > 0.0)
                d.asm_burn[la] = ab[la] / (ahm[la] * 1000.0);
        }
        d.frn = frn;
        d.frp = g.frp();
        d.fqp = g.fqp();
    }

    {
        static const std::string xeKey = "541350";
        static const std::string smKey = "621490";
        static const std::string gdKey = "640000";
        bool                     hasXe = Chiffon::Isotope::iidx.count(xeKey) > 0;
        bool                     hasSm = Chiffon::Isotope::iidx.count(smKey) > 0;
        bool                     hasGd = Chiffon::Isotope::iidx.count(gdKey) > 0;

        double xe_s = 0, xe_v = 0, sm_s = 0, sm_v = 0, gd_s = 0, gd_v = 0;
        double xe_bot = 0, xe_bv = 0, xe_top = 0, xe_tv = 0;
        double sm_bot = 0, sm_bv = 0, sm_top = 0, sm_tv = 0;
        for (int k = kbc; k < kec; ++k) {
            // As for the power AO: a node wholly on one side is accumulated
            // unscaled, so a boundary-aligned midplane sums the same terms.
            const double fb   = bot_frac[k];
            const double ft   = 1.0 - fb;
            const bool   full = (fb >= 1.0 || fb <= 0.0);
            for (int l = 0; l < nxy; ++l) {
                int    lk  = k * nxy + l;
                double vol = g.vol(lk);
                if (vol <= 1.0e-20 || !g.IsFuel(lk)) continue;
                if (hasXe) {
                    double xe = _xs.iden(Chiffon::Isotope::iidx.at(xeKey), lk);
                    xe_s += xe * vol;
                    xe_v += vol;
                    if (full) {
                        if (fb >= 1.0) {
                            xe_bot += xe * vol;
                            xe_bv += vol;
                        } else {
                            xe_top += xe * vol;
                            xe_tv += vol;
                        }
                    } else {
                        xe_bot += xe * vol * fb;
                        xe_bv += vol * fb;
                        xe_top += xe * vol * ft;
                        xe_tv += vol * ft;
                    }
                }
                if (hasSm) {
                    double sm = _xs.iden(Chiffon::Isotope::iidx.at(smKey), lk);
                    sm_s += sm * vol;
                    sm_v += vol;
                    if (full) {
                        if (fb >= 1.0) {
                            sm_bot += sm * vol;
                            sm_bv += vol;
                        } else {
                            sm_top += sm * vol;
                            sm_tv += vol;
                        }
                    } else {
                        sm_bot += sm * vol * fb;
                        sm_bv += vol * fb;
                        sm_top += sm * vol * ft;
                        sm_tv += vol * ft;
                    }
                }
                if (hasGd) {
                    gd_s += _xs.iden(Chiffon::Isotope::iidx.at(gdKey), lk) * vol;
                    gd_v += vol;
                }
            }
        }
        d.xe_avg    = (xe_v > 0.0) ? xe_s / xe_v : 0.0;
        d.sm_avg    = (sm_v > 0.0) ? sm_s / sm_v : 0.0;
        d.gd_avg    = (gd_v > 0.0) ? gd_s / gd_v : 0.0;
        double xe_b = (xe_bv > 0) ? xe_bot / xe_bv : 0;
        double xe_t = (xe_tv > 0) ? xe_top / xe_tv : 0;
        d.xe_ao     = (xe_b + xe_t > 0) ? (xe_t - xe_b) / (xe_b + xe_t) : 0;
        double sm_b = (sm_bv > 0) ? sm_bot / sm_bv : 0;
        double sm_t = (sm_tv > 0) ? sm_top / sm_tv : 0;
        d.sm_ao     = (sm_b + sm_t > 0) ? (sm_t - sm_b) / (sm_b + sm_t) : 0;
    }

    // 3. Finish with volume-weighted thermal-hydraulic summaries.
    {
        double tf_s = 0, tf_v = 0, tm_s = 0, tm_v = 0, dm_s = 0, dm_v = 0;
        d.tf_max = 0;
        d.tm_max = 0;
        d.dm_max = 0;
        for (int k = kbc; k < kec; ++k)
            for (int l = 0; l < nxy; ++l) {
                int    lk  = k * nxy + l;
                double vol = g.vol(lk);
                if (vol <= 1.0e-20 || !g.IsFuel(lk)) continue;
                double tf = _g.tful(lk);
                tf_s += tf * vol;
                tf_v += vol;
                d.tf_max  = std::max(d.tf_max, tf);
                double tm = _g.tmod(lk);
                tm_s += tm * vol;
                tm_v += vol;
                d.tm_max  = std::max(d.tm_max, tm);
                double dm = _g.dmod(lk);
                dm_s += dm * vol;
                dm_v += vol;
                d.dm_max = std::max(d.dm_max, dm);
            }
        d.tf_avg = (tf_v > 0) ? tf_s / tf_v : 0;
        d.tm_avg = (tm_v > 0) ? tm_s / tm_v : 0;
        d.dm_avg = (dm_v > 0) ? dm_s / dm_v : 0;
    }
}
// HDF5 result output

// ---------------------------------------------------------------------------
// WP12: the pin-power CSV of one statepoint, as a VALUE.
//
// WHY A TYPE AND NOT A LAMBDA.  The CSV is the expensive half of the result
// path -- Driver.h prices it at ~119 MB per `--result full` case, formatted one
// double at a time through `ostream <<` -- and it used to run on the solver
// thread inside PH_RESULT_WRITE.  To let the writer thread run it instead, the
// emitter must not name a single thing the solver still owns: Geometry, XSSet,
// Schedule and IO's own `_pin_power_csv_*` members are all mutated the instant
// WriteStepToResult returns.  So every number it needs is COPIED into this
// struct, and the struct is what the closure captures.  A reviewer can check
// that claim by reading the field list: there is no reference and no pointer.
//
// The emitter body is the pre-WP12 block verbatim, with exactly two
// substitutions -- `g.hz(k)` -> `hz[k]` and `g.ijtola(ia, ja)` ->
// `ijtola[ja * nxa + ia]`, both of which are lookups into copies of the same
// tables.  Same manipulators, same order, same `<<` calls, therefore same
// bytes; that is what makes the gate B0.
// ---------------------------------------------------------------------------
namespace {

struct PinPowerCsvRecord {
    std::filesystem::path path;
    /// Decided by the SOLVER, at record time, from `_pin_power_csv_started` --
    /// never re-derived here.  The writer thread must not have an opinion about
    /// whether this is the first block of the run.
    bool                append = false;
    int                 step   = 0;
    double              efpd   = 0.0;
    int                 nz = 0, nxya = 0, npins = 0, npina = 0;
    int                 nxa = 0, nya = 0, kbc = 0, kec = 0;
    std::vector<double> hz;        ///< [nz]
    std::vector<int>    ijtola;    ///< [nya * nxa], -1 where there is no assembly
    std::vector<double> pin_data;  ///< [nz * nxya * npina]

    /// Resident footprint, for the writer queue's byte bound.
    [[nodiscard]] std::size_t bytes() const {
        return pin_data.size() * sizeof(double) + hz.size() * sizeof(double) +
               ijtola.size() * sizeof(int);
    }

    void Emit() const;
};

void PinPowerCsvRecord::Emit() const {
    const auto started = std::chrono::steady_clock::now();

    // Byte accounting by file size, not by counting `<<` calls: it costs two
    // stats and it cannot drift from what actually landed.  A trunc block
    // starts from zero by definition; an append block starts from what is
    // already there.
    std::uintmax_t before = 0;
    if (append) {
        std::error_code ec;
        const std::uintmax_t sized = std::filesystem::file_size(path, ec);
        if (!ec) before = sized;
    }

    {
        std::ofstream csv(path, append ? std::ios::app : std::ios::trunc);
        if (!csv)
            throw std::runtime_error("IO: failed to open pin-power CSV: " + path.string());
        csv << std::setprecision(6);
        if (append) csv << '\n';

        std::vector<double> zavg(static_cast<size_t>(nxya) * npina, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> zsum(static_cast<size_t>(nxya), 0.0);
        for (int k = kbc; k < kec; ++k) {
            const double hz_k = hz[k];
            for (int la = 0; la < nxya; ++la) {
                const size_t src = (static_cast<size_t>(k) * nxya + la) * npina;
                if (std::isnan(pin_data[src])) continue;
                if (zsum[la] == 0.0) {
                    for (int pi = 0; pi < npina; ++pi)
                        zavg[static_cast<size_t>(la) * npina + pi] = 0.0;
                }
                zsum[la] += hz_k;
                for (int pi = 0; pi < npina; ++pi)
                    zavg[static_cast<size_t>(la) * npina + pi] += pin_data[src + pi] * hz_k;
            }
        }
        for (int la = 0; la < nxya; ++la) {
            if (zsum[la] <= 0.0) continue;
            const double inv_hz = 1.0 / zsum[la];
            for (int pi = 0; pi < npina; ++pi)
                zavg[static_cast<size_t>(la) * npina + pi] *= inv_hz;
        }

        csv << std::format("Pin Power (Z-averaged) -- Step {} (EFPD={:.2f})\n", step, efpd);
        for (int ja = 0; ja < nya; ++ja) {
            for (int py = 0; py < npins; ++py) {
                bool first_col = true;
                for (int ia = 0; ia < nxa; ++ia) {
                    const int la = ijtola[static_cast<size_t>(ja) * nxa + ia];
                    for (int px = 0; px < npins; ++px) {
                        if (!first_col) csv << ',';
                        first_col = false;
                        if (la < 0 || la >= nxya) continue;
                        const int    pi  = py * npins + px;
                        const double val = zavg[static_cast<size_t>(la) * npina + pi];
                        if (!std::isnan(val)) csv << val;
                    }
                }
                csv << '\n';
            }
        }
        csv << '\n';

        for (int k = kbc; k < kec; ++k) {
            const size_t src = static_cast<size_t>(k) * nxya * npina;
            csv << std::format("Pin Power k={} (hz={:.4f}) -- Step {} (EFPD={:.2f})\n",
                               k, hz[k], step, efpd);
            for (int ja = 0; ja < nya; ++ja) {
                for (int py = 0; py < npins; ++py) {
                    bool first_col = true;
                    for (int ia = 0; ia < nxa; ++ia) {
                        const int la = ijtola[static_cast<size_t>(ja) * nxa + ia];
                        for (int px = 0; px < npins; ++px) {
                            if (!first_col) csv << ',';
                            first_col = false;
                            if (la < 0 || la >= nxya) continue;
                            const int    pi  = py * npins + px;
                            const double val = pin_data[src + static_cast<size_t>(la) * npina + pi];
                            if (!std::isnan(val)) csv << val;
                        }
                    }
                    csv << '\n';
                }
            }
            csv << '\n';
        }
    }

    std::error_code      after_ec;
    const std::uintmax_t after = std::filesystem::file_size(path, after_ec);
    auto&                tally = rasbery::iowriter::resultIoCounters();
    if (!after_ec && after >= before)
        tally.bytes.fetch_add(static_cast<std::uint64_t>(after - before),
                                 std::memory_order_relaxed);
    tally.records.fetch_add(1, std::memory_order_relaxed);
    tally.writer_ns.fetch_add(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - started)
                                       .count()),
        std::memory_order_relaxed);
}

}  // namespace

/// @brief Create the result HDF5 file and write the geometry group.
void IO::OpenResult(const std::string& filepath) {
    _result_path                     = filepath;
    std::filesystem::path result_dir = _result_path.parent_path();
    if (result_dir.empty()) result_dir = ".";
    const std::string result_stem = _result_path.stem().empty() ? "result" : _result_path.stem().string();
    _pin_power_csv_path           = result_dir / (result_stem + "_pinpower.csv");
    _pin_power_csv_started        = false;
    std::error_code ec;
    std::filesystem::remove(_pin_power_csv_path, ec);

    _result_session      = std::make_shared<iowriter::FileSession>();
    _result_session->job = result_stem;

    iowriter::Recorder rec(_result_session);
    rec.openOverwrite(filepath);

    auto geo = rec.root().createGroup("geometry");

    const int nx = _g.nx(), ny = _g.ny(), nxy = _g.nxy();
    const int nxa = _g.nxa(), nya = _g.nya(), nxya = _g.nxya();
    const int nz = _g.nz(), ng = _g.ng();
    const int kbc = _g.kbc(), kec = _g.kec();
    const int npins = _g.npins(), ndivxy = _g.ndivxy();

    geo.createDataSet("nx", nx);
    geo.createDataSet("ny", ny);
    geo.createDataSet("nxy", nxy);
    geo.createDataSet("nxa", nxa);
    geo.createDataSet("nya", nya);
    geo.createDataSet("nxya", nxya);
    geo.createDataSet("nz", nz);
    geo.createDataSet("ng", ng);
    geo.createDataSet("kbc", kbc);
    geo.createDataSet("kec", kec);
    geo.createDataSet("npins", npins);
    geo.createDataSet("ndivxy", ndivxy);

    std::vector<double> hz(nz);
    for (int k = 0; k < nz; ++k)
        hz[k] = _g.hz(k);
    geo.createDataSet("hz", hz);

    std::vector<int> row_ixs(nya), row_ixe(nya);
    for (int ja = 0; ja < nya; ++ja) {
        row_ixs[ja] = _g.nxsa(ja);
        row_ixe[ja] = _g.nxea(ja);
    }
    geo.createDataSet("row_ixs", row_ixs);
    geo.createDataSet("row_ixe", row_ixe);

    std::vector<int> ijtola(static_cast<size_t>(nya) * nxa, -1);
    for (int ja = 0; ja < nya; ++ja)
        for (int ia = row_ixs[ja]; ia < row_ixe[ja]; ++ia)
            ijtola[ja * nxa + ia] = _g.ijtola(ia, ja);
    geo.createDataSet("ijtola", ijtola);

    // Compact fine-node index -> Cartesian (i,j).  The internal mesh omits
    // leading/trailing void cells, so nxy can be smaller than nx*ny.
    std::vector<int> node_i(nxy, -1), node_j(nxy, -1);
    for (int j = 0; j < ny; ++j)
        for (int i = _g.nxs(j); i < _g.nxe(j); ++i) {
            const int l = _g.ijtol(i, j);
            if (l >= 0 && l < nxy) {
                node_i[l] = i;
                node_j[l] = j;
            }
        }
    geo.createDataSet("node_i", node_i);
    geo.createDataSet("node_j", node_j);

    rec.submit();
    PLOG_INFO << "Result file opened: " << filepath;
}

/// @brief Whether OpenResult() has a live result file for this job.
bool IO::HasOpenResult() const {
    if (!_result_session) return false;
    return iowriter::mode() != iowriter::Mode::Inline || _result_session->file != nullptr;
}

/// @brief Write one step's data to the open HDF5 result file.
void IO::WriteStepToResult(Geometry& g, const XSSet& xs, int schedule_index) {
    if (!HasOpenResult() || schedule_index < 0 ||
        schedule_index >= static_cast<int>(_s.schedule().size()))
        return;

    const Schedule& d = _s.schedule()[schedule_index];
    if (d.step <= 0) return;

    // Cheap, non-blocking: if a write this deck already queued has failed (the
    // result file could not be created at all, say), stop here rather than
    // compute another statepoint whose output can never land.
    ThrowIfWritesFailed();

    iowriter::Recorder rec(_result_session);
    auto step_grp = rec.root().createGroup(std::format("steps/{:04d}", d.step));

    // Assembly data
    {
        auto grp = step_grp.createGroup("assembly");
        grp.createDataSet("type", d.asm_type);
        grp.createDataSet("power", d.asm_power);
        grp.createDataSet("burn", d.asm_burn);
        grp.createDataSet("kinf", d.asm_kinf);
    }

    // Axial data
    {
        auto grp = step_grp.createGroup("axial");
        grp.createDataSet("power", d.ax_power);
        grp.createDataSet("tful", d.ax_tful);
        grp.createDataSet("tmod", d.ax_tmod);
        grp.createDataSet("dmod", d.ax_dmod);
    }

    // Full nodal fields used for independent power / local-k / burnup audits.
    // These are compact for a nodal core model and avoid trying to reconstruct
    // spatial distributions from peak factors alone after the run.
    {
        const int nxyz = g.nxyz();
        const int ng   = g.ng();
        std::vector<double> power(nxyz, 0.0);
        std::vector<double> burnup(nxyz, 0.0);
        std::vector<double> kinf(nxyz, 0.0);
        std::vector<double> flux(static_cast<size_t>(nxyz) * ng, 0.0);

        double total_power = 0.0;
        double fuel_volume = 0.0;
        double total_flux  = 0.0;
        for (int lk = 0; lk < nxyz; ++lk) {
            for (int ig = 0; ig < ng; ++ig)
                flux[static_cast<size_t>(lk) * ng + ig] = g.Phif()[lk * ng + ig];
            if (!g.IsFuel(lk)) continue;

            double fission_power = 0.0;
            double production    = 0.0;
            double absorption    = 0.0;
            double node_flux     = 0.0;
            for (int ig = 0; ig < ng; ++ig) {
                const double phi = g.Phif()[lk * ng + ig];
                fission_power += xs.xskf(ig, lk) * phi;
                production += xs.xsnf(ig, lk) * phi;
                absorption += xs.xsaf(ig, lk) * phi;
                node_flux += phi;
            }
            total_flux += node_flux * g.vol(lk);
            power[lk]  = fission_power;
            kinf[lk]   = (absorption > 1.0e-30) ? production / absorption : 0.0;
            burnup[lk] = static_cast<double>(xs.burn(lk)) / 1000.0;
            total_power += fission_power * g.vol(lk);
            fuel_volume += g.vol(lk);
        }
        const double average_power =
            (fuel_volume > 0.0 && total_power > 1.0e-30) ? total_power / fuel_volume : 1.0;
        // /steps/*/node/power is written on a 1.0-based scale: each node's power
        // density divided by the fuel-volume-averaged power density. /steps/*/node/flux
        // used to be dumped as the raw solver Phif, whose absolute magnitude is an
        // artefact of the eigenvalue normalisation and drifts from step to step, so
        // the two datasets in the same group were not on a common scale and flux
        // could not be compared against power (or against another step) without an
        // out-of-band rescale.
        //
        // "Same normalisation" here means the same *recipe*, not the same constant:
        // dividing flux by average_power would be dimensionally wrong (average_power
        // is a power density, so flux/average_power is neither a flux nor a ratio,
        // and on CY01 it inflates the stored values by ~1e10). Flux is therefore
        // divided by the fuel-volume-averaged group-summed flux, which puts it on
        // the same 1.0-based footing as power while preserving the group spectrum
        // ratio within each node.
        const double average_flux =
            (fuel_volume > 0.0 && total_flux > 1.0e-30) ? total_flux / fuel_volume : 1.0;
        // Kept as a division (not a reciprocal multiply) so `power` stays bitwise
        // identical to the previous output and this commit shows up in a regression
        // diff only on `flux`.
        for (int lk = 0; lk < nxyz; ++lk)
            power[lk] /= average_power;
        for (size_t i = 0; i < flux.size(); ++i)
            flux[i] /= average_flux;

        auto grp = step_grp.createGroup("node");
        // Geometry stores only active row spans: use [z, compact-node] and
        // `/geometry/node_i,node_j` rather than pretending nxy == nx*ny.
        const iowriter::Dims scalar_space(
            {static_cast<size_t>(g.nz()), static_cast<size_t>(g.nxy())});
        grp.createDataSet<double>("power", scalar_space).write_raw(power.data());
        grp.createDataSet<double>("burnup", scalar_space).write_raw(burnup.data());
        grp.createDataSet<double>("kinf", scalar_space).write_raw(kinf.data());
        const iowriter::Dims flux_space(
            {static_cast<size_t>(g.nz()), static_cast<size_t>(g.nxy()),
             static_cast<size_t>(ng)});
        grp.createDataSet<double>("flux", flux_space).write_raw(flux.data());
    }

    // Pin-wise PPR output (only when pin_info is set)
    if (d.print_opt.pin_info) {
        const int nxya  = g.nxya();
        const int nz    = g.nz();
        const int kbc   = g.kbc();
        const int kec   = g.kec();
        const int npins = g.npins();
        const int npina = npins * npins;
        const int nxy   = g.nxy();
        const int ndiv2 = g.ndivxy() * g.ndivxy();
        const int ng    = g.ng();
        const int nxa   = g.nxa();
        const int nya   = g.nya();

        double*             ppower = g.PinPower();
        const size_t        total  = static_cast<size_t>(nz) * nxya * npina;
        std::vector<double> pin_data(total, std::numeric_limits<double>::quiet_NaN());

        for (int k = 0; k < nz; ++k) {
            for (int la = 0; la < nxya; ++la) {
                bool valid = false;
                if (k >= kbc && k < kec) {
                    for (int li = 0; li < ndiv2 && !valid; ++li) {
                        int l = g.latol(li, la);
                        if (l >= 0 && g.IsFuel(l + nxy * k)) valid = true;
                    }
                }
                if (!valid) continue;

                const size_t lka = static_cast<size_t>(la + nxya * k);
                const size_t dst = (static_cast<size_t>(k) * nxya + la) * npina;
                for (int pi = 0; pi < npina; ++pi)
                    pin_data[dst + pi] = ppower[lka * npina + pi];
            }
        }

        const iowriter::Dims space({static_cast<size_t>(nz),
                                    static_cast<size_t>(nxya),
                                    static_cast<size_t>(npins),
                                    static_cast<size_t>(npins)});
        step_grp.createDataSet<double>("pin_power", space).write_raw(pin_data.data());

        if (d.print_opt.pin_flux) {
            double*             pphif      = g.PinFlux();
            const size_t        flux_total = static_cast<size_t>(nz) * nxya * ng * npina;
            std::vector<double> flux_data(flux_total, std::numeric_limits<double>::quiet_NaN());

            for (int k = 0; k < nz; ++k) {
                for (int la = 0; la < nxya; ++la) {
                    bool valid = false;
                    if (k >= kbc && k < kec) {
                        for (int li = 0; li < ndiv2 && !valid; ++li) {
                            int l = g.latol(li, la);
                            if (l >= 0 && g.IsFuel(l + nxy * k)) valid = true;
                        }
                    }
                    if (!valid) continue;

                    const size_t lka = static_cast<size_t>(la + nxya * k);
                    const size_t dst = (static_cast<size_t>(k) * nxya + la) * ng * npina;
                    for (int ig = 0; ig < ng; ++ig) {
                        const size_t src_g = lka * ng * npina + static_cast<size_t>(ig) * npina;
                        const size_t dst_g = dst + static_cast<size_t>(ig) * npina;
                        for (int pi = 0; pi < npina; ++pi)
                            flux_data[dst_g + pi] = pphif[src_g + pi];
                    }
                }
            }

            const iowriter::Dims flux_space({static_cast<size_t>(nz),
                                             static_cast<size_t>(nxya),
                                             static_cast<size_t>(ng),
                                             static_cast<size_t>(npins),
                                             static_cast<size_t>(npins)});
            step_grp.createDataSet<double>("pin_flux", flux_space).write_raw(flux_data.data());
        }

        // WP12.  Snapshot, then emit -- the two halves that used to be one.
        // Building the record is a memcpy of numbers the solver already has;
        // Emit() is the ~15 MB of text formatting that used to sit here on the
        // solver thread, and is what the gate moves.
        PinPowerCsvRecord csv_record;
        csv_record.path   = _pin_power_csv_path;
        csv_record.append = _pin_power_csv_started;
        csv_record.step   = d.step;
        csv_record.efpd   = d.efpd;
        csv_record.nz     = nz;
        csv_record.nxya   = nxya;
        csv_record.npins  = npins;
        csv_record.npina  = npina;
        csv_record.nxa    = nxa;
        csv_record.nya    = nya;
        csv_record.kbc    = kbc;
        csv_record.kec    = kec;
        csv_record.hz.resize(static_cast<size_t>(nz));
        for (int k = 0; k < nz; ++k)
            csv_record.hz[static_cast<size_t>(k)] = g.hz(k);
        csv_record.ijtola.resize(static_cast<size_t>(nya) * nxa);
        for (int ja = 0; ja < nya; ++ja)
            for (int ia = 0; ia < nxa; ++ia)
                csv_record.ijtola[static_cast<size_t>(ja) * nxa + ia] = g.ijtola(ia, ja);
        // MOVED, not copied: `pin_data` has no reader left -- the HDF5
        // write_raw above already took its own copy into the batch.
        csv_record.pin_data = std::move(pin_data);

        // The flag advances HERE in both modes, because it is the SOLVER's
        // question ("has this run written a block yet?") and the answer must be
        // the same whether or not the emitter has run yet.
        _pin_power_csv_started = true;

        if (iowriter::resultAsyncEnabled()) {
            const std::size_t csv_bytes = csv_record.bytes();
            rec.pushSideTask(
                [record = std::move(csv_record)]() { record.Emit(); }, csv_bytes);
        } else {
            csv_record.Emit();
        }
    }

    // XS data (only when xs_info is set)
    if (d.print_opt.xs_info) {
        const int ng   = g.ng();
        const int nxyz = g.nxyz();

        std::vector<int> fuel_indices;
        for (int lk = 0; lk < nxyz; ++lk)
            if (g.IsFuel(lk)) fuel_indices.push_back(lk);

        const int nfuel = static_cast<int>(fuel_indices.size());
        if (nfuel > 0) {
            std::vector<int>    burn(nfuel);
            std::vector<double> bppm(nfuel), tful(nfuel), dmod(nfuel);
            std::vector<double> xsdf(static_cast<size_t>(nfuel) * ng);
            std::vector<double> xsaf(static_cast<size_t>(nfuel) * ng);
            std::vector<double> xsnf(static_cast<size_t>(nfuel) * ng);
            std::vector<double> xssf(static_cast<size_t>(nfuel) * ng);

            for (int i = 0; i < nfuel; ++i) {
                int lk  = fuel_indices[i];
                burn[i] = xs.burn(lk);
                bppm[i] = g.bppm(lk);
                tful[i] = g.tful(lk);
                dmod[i] = g.dmod(lk);
                for (int ig = 0; ig < ng; ++ig) {
                    xsdf[i * ng + ig] = xs.xsdf(ig, lk);
                    xsaf[i * ng + ig] = xs.xsaf(ig, lk);
                    xsnf[i * ng + ig] = xs.xsnf(ig, lk);
                    xssf[i * ng + ig] = xs.xssf(ig, lk);
                }
            }

            auto grp = step_grp.createGroup("xs");
            grp.createDataSet("indices", fuel_indices);
            grp.createDataSet("burn", burn);
            grp.createDataSet("bppm", bppm);
            grp.createDataSet("tful", tful);
            grp.createDataSet("dmod", dmod);

            const iowriter::Dims xs_space({static_cast<size_t>(nfuel),
                                           static_cast<size_t>(ng)});
            grp.createDataSet<double>("xsdf", xs_space).write_raw(xsdf.data());
            grp.createDataSet<double>("xsaf", xs_space).write_raw(xsaf.data());
            grp.createDataSet<double>("xsnf", xs_space).write_raw(xsnf.data());
            grp.createDataSet<double>("xssf", xs_space).write_raw(xssf.data());
        }
    }

    // Node monitor is an explicit per-step diagnostic. It is not propagated
    // because schedule-only restore/check steps can otherwise pollute the CSV.
    if (!d.print_opt.node_monitor.empty()) {
        const int ng     = g.ng();
        const int nxyz   = g.nxyz();
        auto      nm_grp = step_grp.createGroup("node_monitor");

        // Absolute-flux normalisation: phif * norm_factor → n/cm²/s
        const double power_rate  = d.rate / 100.0;
        const double power       = g.rated_power() * power_rate;
        const double norm_factor = xs.NormFactor(power);

        for (int lk : d.print_opt.node_monitor) {
            if (lk < 0 || lk >= nxyz || !g.IsFuel(lk)) continue;
            auto ngrp = nm_grp.createGroup(std::format("{}", lk));

            const int    burn = xs.burn(lk);
            const double bppm = g.bppm(lk);
            const double tful = g.tful(lk);
            const double tmod = g.tmod(lk);
            const double dmod = g.dmod(lk);
            // The reference cross section must be taken on the rod branch the node
            // actually sits on, otherwise a rodded node is compared against an
            // unrodded library point.  Fall back to rod-out when the model has no
            // depletion trajectory for that control type.
            const auto& model  = xs.models()[xs.comp(lk)];
            int         ref_ct = (g.rod_fraction(lk) > 1.0e-12) ? xs.ctyp(lk) : 0;
            if (ref_ct != 0 && model._refr_dpts.count(ref_ct) == 0)
                ref_ct = 0;

            ngrp.createDataSet("burn", burn);
            ngrp.createDataSet("bppm", bppm);
            ngrp.createDataSet("tful", tful);
            ngrp.createDataSet("tmod", tmod);
            ngrp.createDataSet("dmod", dmod);
            ngrp.createDataSet("ref_ctype", ref_ct);
            ngrp.createDataSet("keff", d.eigv);
            ngrp.createDataSet("step", d.step);
            ngrp.createDataSet("efpd", d.efpd);
            ngrp.createDataSet("summary_burnup", d.bu_avg);

            // Compare the runtime and library thermal-fluence coordinates used by RDPL.
            const int ct = xs.ctyp(lk);
            ngrp.createDataSet("rod_thermal_fluence",
                               xs.FineRodThermalFluenceAverage(lk, ct));
            ngrp.createDataSet("ref_rod_thermal_fluence",
                               model.ReferenceRodFluence(ref_ct, burn));

            // Calculated (reconstructed) XS and flux
            std::vector<double> xsdf_v(ng), xsaf_v(ng), xsnf_v(ng), xskf_v(ng), xssf_v(ng), flux_v(ng);
            for (int ig = 0; ig < ng; ++ig) {
                xsdf_v[ig] = xs.xsdf(ig, lk);
                xsaf_v[ig] = xs.xsaf(ig, lk);
                xsnf_v[ig] = xs.xsnf(ig, lk);
                xskf_v[ig] = xs.xskf(ig, lk);
                xssf_v[ig] = xs.xssf(ig, lk);
                flux_v[ig] = g.Phif()[lk * ng + ig] * norm_factor;
            }
            ngrp.createDataSet("xsdf", xsdf_v);
            ngrp.createDataSet("xsaf", xsaf_v);
            ngrp.createDataSet("xsnf", xsnf_v);
            ngrp.createDataSet("xskf", xskf_v);
            ngrp.createDataSet("xssf", xssf_v);
            ngrp.createDataSet("flux", flux_v);

            // Calculated isotope density
            ngrp.createDataSet("isotope_density", xs.getNodeIden(lk));

            // Reference library XS at the same burnup, boron, fuel temperature, and density.
            milk::Vector<double> ref_iden_local;
            auto                 ref_xs =
                model.GetCrossSection(ref_ct, burn, bppm, tful, dmod, &ref_iden_local);

            std::vector<double> ref_xsdf(ng), ref_xsaf(ng), ref_xsnf(ng), ref_xskf(ng), ref_xssf(ng);
            for (int ig = 0; ig < ng; ++ig) {
                ref_xsdf[ig] = ref_xs.maxs(ig, Chiffon::XSDF);
                ref_xsaf[ig] = ref_xs.maxs(ig, Chiffon::XSAF);
                ref_xsnf[ig] = ref_xs.maxs(ig, Chiffon::XSNF);
                ref_xskf[ig] = ref_xs.maxs(ig, Chiffon::XSKF);
                ref_xssf[ig] = ref_xs.maxs(ig, Chiffon::XSSF);
            }
            ngrp.createDataSet("ref_xsdf", ref_xsdf);
            ngrp.createDataSet("ref_xsaf", ref_xsaf);
            ngrp.createDataSet("ref_xsnf", ref_xsnf);
            ngrp.createDataSet("ref_xskf", ref_xskf);
            ngrp.createDataSet("ref_xssf", ref_xssf);

            // Isotope-wise microscopic XS dump for delta-correction diagnostics.
            const size_t         niso = Chiffon::Isotope::niso;
            const iowriter::Dims mic_space({niso, static_cast<size_t>(ng)});
            auto                 dump_mic = [&](const std::string& name, Chiffon::XSTYPE xt) {
                std::vector<double> cur(niso * static_cast<size_t>(ng), 0.0);
                std::vector<double> ref(niso * static_cast<size_t>(ng), 0.0);
                std::vector<double> baseline(niso * static_cast<size_t>(ng), 0.0);
                for (size_t iso = 0; iso < niso; ++iso) {
                    for (int ig = 0; ig < ng; ++ig) {
                        const size_t off = iso * static_cast<size_t>(ng) + static_cast<size_t>(ig);
                        cur[off]         = xs.micx(xt, iso, ig, lk);
                        ref[off]         = ref_xs.mixs(static_cast<int>(iso), ig, xt);
                        baseline[off]    = xs.refMicx(xt, iso, ig, lk);
                    }
                }
                ngrp.createDataSet<double>("mic_" + name, mic_space).write_raw(cur.data());
                ngrp.createDataSet<double>("ref_mic_" + name, mic_space).write_raw(ref.data());
                ngrp.createDataSet<double>("baseline_mic_" + name, mic_space).write_raw(baseline.data());
            };
            dump_mic("xstf", Chiffon::XSTF);
            dump_mic("xsaf", Chiffon::XSAF);
            dump_mic("xsff", Chiffon::XSFF);
            dump_mic("xsnf", Chiffon::XSNF);
            dump_mic("xskf", Chiffon::XSKF);
            dump_mic("xssf", Chiffon::XSSF);
            dump_mic("xs2n", Chiffon::XS2N);

            // Spectral-history contributions to the macro group constants.
            writeTermContrib(ngrp, xs, lk, ng);

            // Reference isotope density and kinf: interpolate between bounding depletion points
            {
                const auto& refrMap = model._refr_dpts.at(ref_ct);
                auto        hiIt    = refrMap.lower_bound(burn);
                auto        loIt    = hiIt;
                if (hiIt == refrMap.end()) {
                    loIt = std::prev(refrMap.end());
                    hiIt = loIt;
                } else if (hiIt == refrMap.begin()) {
                    loIt = refrMap.begin();
                } else {
                    loIt = std::prev(hiIt);
                }
                const auto& loDpt    = model.GetDepletionPoint(loIt->second);
                const auto& hiDpt    = model.GetDepletionPoint(hiIt->second);
                auto        ref_iden = loDpt._iden.toVector();
                double      ref_kinf = loDpt._data[Chiffon::AD_KINF];
                auto        ref_flux = loDpt._aflx; // reference average flux per group
                if (loIt != hiIt && hiDpt.burnKey() != loDpt.burnKey()) {
                    double     alpha   = static_cast<double>(burn - loDpt.burnKey()) / static_cast<double>(hiDpt.burnKey() - loDpt.burnKey());
                    const auto hi_iden = hiDpt._iden.toVector();
                    for (size_t ii = 0; ii < ref_iden.size() && ii < hi_iden.size(); ++ii)
                        ref_iden[ii] += alpha * (hi_iden[ii] - ref_iden[ii]);
                    ref_kinf += alpha * (hiDpt._data[Chiffon::AD_KINF] - loDpt._data[Chiffon::AD_KINF]);
                    for (size_t ii = 0; ii < ref_flux.size() && ii < hiDpt._aflx.size(); ++ii)
                        ref_flux[ii] += alpha * (hiDpt._aflx[ii] - ref_flux[ii]);
                }
                ngrp.createDataSet("ref_isotope_density", ref_iden);
                ngrp.createDataSet("ref_kinf", ref_kinf);
                ngrp.createDataSet("ref_flux", ref_flux);
            }

            // Isotope names in global index order
            {
                using namespace Chiffon::Isotope;
                std::vector<std::string> iso_names;
                iso_names.reserve(niso);
                for (const auto& id : isotopeIds)
                    iso_names.emplace_back(id);
                ngrp.createDataSet("isotope_names", iso_names);
            }
        }
    }

    // One statepoint == one batch: the queue is FIFO and this Driver thread is
    // the only recorder for this file, so the file's batches replay in the same
    // order the run produced them.
    rec.submit();
}

/// @brief Write accumulated summary arrays and close the HDF5 result file.
void IO::CloseResult() {
    if (!HasOpenResult()) return;

    iowriter::Recorder rec(_result_session);
    auto sum = rec.root().createGroup("summary");

    std::vector<int>    steps;
    std::vector<double> efpd, bu_avg, ppm, keff, reactivity;
    std::vector<double> ao, asi, fqn, frn, fqp, frp;
    std::vector<double> xe_avg, xe_ao, sm_avg, sm_ao, gd_avg;
    std::vector<double> tf_avg, tf_max, tm_avg, tm_max, dm_avg, dm_max;
    std::vector<double> rod_step;
    // Critical-search termination quality, per step.  search_status is the SearchExit enum
    // (0 none / 1 converged / 2 best-point fallback / 3 unconverged); search_dk is the
    // accepted |k_eff - target| residual and search_tol the tolerance it was judged against.
    // Without these a step that terminated on a flux limit cycle or an exhausted search was
    // indistinguishable in the result file from a properly converged one.
    std::vector<int>    search_status;
    std::vector<double> search_dk, search_tol;

    // Collect the union of rod-group names across all reported steps (stable order).
    std::vector<std::string>   rod_groups;
    std::map<std::string, int> rod_group_index;
    for (const auto& d : _s.schedule()) {
        if (d.step <= 0 || !d.print_opt.summary) continue;
        for (const auto& [name, depth] : d.rod_insertions) {
            if (rod_group_index.emplace(name, static_cast<int>(rod_groups.size())).second)
                rod_groups.push_back(name);
        }
    }

    for (const auto& d : _s.schedule()) {
        if (d.step <= 0 || !d.print_opt.summary) continue;
        steps.push_back(d.step);
        efpd.push_back(d.efpd);
        bu_avg.push_back(d.bu_avg);
        ppm.push_back(d.ppm);
        keff.push_back(d.eigv);
        reactivity.push_back(d.rho);
        ao.push_back(d.ao);
        asi.push_back(d.asi);
        fqn.push_back(d.fqn);
        frn.push_back(d.frn);
        fqp.push_back(d.fqp);
        frp.push_back(d.frp);
        xe_avg.push_back(d.xe_avg);
        xe_ao.push_back(d.xe_ao);
        sm_avg.push_back(d.sm_avg);
        gd_avg.push_back(d.gd_avg);
        sm_ao.push_back(d.sm_ao);
        tf_avg.push_back(d.tf_avg);
        tf_max.push_back(d.tf_max);
        tm_avg.push_back(d.tm_avg);
        tm_max.push_back(d.tm_max);
        dm_avg.push_back(d.dm_avg);
        dm_max.push_back(d.dm_max);
        rod_step.push_back(d.rod_step);
        search_status.push_back(d.search_exit_status);
        search_dk.push_back(d.search_exit_dk);
        search_tol.push_back(d.search_exit_tol);
    }

    if (!steps.empty()) {
        sum.createDataSet("step", steps);
        sum.createDataSet("efpd", efpd);
        sum.createDataSet("bu_avg", bu_avg);
        sum.createDataSet("ppm", ppm);
        sum.createDataSet("keff", keff);
        sum.createDataSet("reactivity", reactivity);
        sum.createDataSet("ao", ao);
        sum.createDataSet("asi", asi);
        sum.createDataSet("fqn", fqn);
        sum.createDataSet("frn", frn);
        sum.createDataSet("fqp", fqp);
        sum.createDataSet("frp", frp);
        sum.createDataSet("xe_avg", xe_avg);
        sum.createDataSet("xe_ao", xe_ao);
        sum.createDataSet("sm_avg", sm_avg);
        sum.createDataSet("sm_ao", sm_ao);
        sum.createDataSet("gd_avg", gd_avg);
        sum.createDataSet("tf_avg", tf_avg);
        sum.createDataSet("tf_max", tf_max);
        sum.createDataSet("tm_avg", tm_avg);
        sum.createDataSet("tm_max", tm_max);
        sum.createDataSet("dm_avg", dm_avg);
        sum.createDataSet("dm_max", dm_max);
        sum.createDataSet("rod_step", rod_step);
        sum.createDataSet("search_status", search_status);
        sum.createDataSet("search_dk", search_dk);
        sum.createDataSet("search_tol", search_tol);
    }

    // Rod insertion table lives outside `summary/` because its datasets do not match
    // the per-step scalar shape that downstream tools (ViewerCSV) assume for everything
    // under `summary/`. Layout: rods/groups (names) + rods/insertions [nsteps, ngroups].
    if (!steps.empty() && !rod_groups.empty()) {
        const size_t        nsteps  = steps.size();
        const size_t        ngroups = rod_groups.size();
        std::vector<double> rod_depths(nsteps * ngroups, 0.0);

        size_t row = 0;
        for (const auto& d : _s.schedule()) {
            if (d.step <= 0 || !d.print_opt.summary) continue;
            for (const auto& [name, depth] : d.rod_insertions) {
                const int col                   = rod_group_index.at(name);
                rod_depths[row * ngroups + col] = depth;
            }
            ++row;
        }

        auto rods = rec.root().createGroup("rods");
        rods.createDataSet("groups", rod_groups);
        const iowriter::Dims rod_space({nsteps, ngroups});
        rods.createDataSet<double>("insertions", rod_space).write_raw(rod_depths.data());
    }

    rec.closeFile();
    rec.submit();

    // End of the job's output: block here until everything this deck queued has
    // actually landed, and re-raise a writer-thread failure on the Driver thread
    // so it fails THIS job rather than disappearing into a process counter.
    FenceJobWrites();
    _result_session.reset();
    _restart_sessions.clear();
    PLOG_INFO << "Result HDF5 closed.";
}

/// @brief Drain this job's queued writes and rethrow the first writer error.
void IO::FenceJobWrites() const {
    iowriter::fence(_result_session);
    for (const auto& session : _restart_sessions) iowriter::fence(session);
    ThrowIfWritesFailed();
}

/// @brief Rethrow an already-recorded writer error, without waiting for the queue.
void IO::ThrowIfWritesFailed() const {
    const auto broke = [](const std::shared_ptr<iowriter::FileSession>& session) {
        if (!session) return false;
        std::lock_guard<std::mutex> lock(session->mtx);
        return session->failed;
    };

    const iowriter::FileSession* failed = nullptr;
    if (broke(_result_session)) failed = _result_session.get();
    if (failed == nullptr)
        for (const auto& session : _restart_sessions)
            if (broke(session)) {
                failed = session.get();
                break;
            }
    if (failed == nullptr) return;

    // `error`/`path` are only read after `failed` was observed set under the
    // same mutex that published them, so they are stable by then.
    std::lock_guard<std::mutex> lock(const_cast<iowriter::FileSession*>(failed)->mtx);
    throw std::runtime_error("IO: HDF5 writer failed for '" + failed->path + "' (job " +
                             failed->job + "): " + failed->error);
}

/// @brief Save a restart snapshot to an HDF5 file.
void IO::SaveRestart(const std::string& filepath,
                     Geometry& g, XSSet& xs,
                     double eigv, double efpd, int step) const {
    const int nxyz = g.nxyz();
    const int ng   = g.ng();
    const int niso = static_cast<int>(Chiffon::Isotope::niso);

    // A restart snapshot is a whole file: open, write, close, all in ONE batch,
    // so it can never interleave with another file's ops.  Its namespace is
    // job-local by construction (Driver::RestartPath derives it from the OUTPUT
    // path, plan Rev.4 Sec 7), and nothing in a run reads back a restart the
    // same run wrote -- restart INPUTS come from the deck and are read during
    // ReadInput, before any write is queued.
    auto session  = std::make_shared<iowriter::FileSession>();
    session->job  = result_stem().empty() ? _input_dir : result_stem();
    _restart_sessions.push_back(session);

    iowriter::Recorder rec(session);
    rec.openOverwrite(filepath);
    auto file = rec.root();

    // /metadata
    {
        auto meta = file.createGroup("metadata");
        meta.createDataSet("eigv", eigv);
        meta.createDataSet("efpd", efpd);
        meta.createDataSet("step", step);
        meta.createDataSet("nxyz", nxyz);
        meta.createDataSet("ng", ng);
        meta.createDataSet("niso", niso);
    }

    // /geometry (GeometryInput needed for verification on load)
    {
        auto geo = file.createGroup("geometry");
        geo.createDataSet("ng", _gin.ng);
        geo.createDataSet("nz", _gin.nz);
        geo.createDataSet("ndivxy", _gin.ndivxy);
        geo.createDataSet("npins", _gin.npins);
        // Written since 2026-09-04.  A restart file older than that has no
        // `nfrod`, and LoadGeometryFromRestart reads it back as 0 -- the legacy
        // divisor, which is what such a run was computed with.
        geo.createDataSet("nfrod", _gin.nfrod);
        // Written since 2026-09-04 too: WHICH fuel-temperature table.  A restart
        // older than that carries none and reads back empty -- the shipped WH
        // grid, which is what such a run was computed with.
        // Only when the deck DECLARED one: a zero-length HDF5 dataset is a
        // portability question nobody needs answered, and "absent" already means
        // exactly what an undeclared table means.
        if (_gin.tf_table.declared()) {
            geo.createDataSet("tf_table_name", std::vector<std::string>{_gin.tf_table.name});
            geo.createDataSet("tf_table_path",
                              std::vector<std::string>{_gin.tf_table.path.empty()
                                                           ? std::string("-")
                                                           : _gin.tf_table.path});
            if (!_gin.tf_table.lpd.empty()) {
                geo.createDataSet("tf_table_lpd", _gin.tf_table.lpd);
                geo.createDataSet("tf_table_bu", _gin.tf_table.bu);
                geo.createDataSet("tf_table_dt", _gin.tf_table.dt);
            }
        }
        geo.createDataSet("hx", _gin.hx);
        geo.createDataSet("hy", _gin.hy);
        geo.createDataSet("hz", _gin.hz);
        geo.createDataSet("symang", _gin.symang);
        geo.createDataSet("symopt", static_cast<int>(_gin.symopt));
        geo.createDataSet("symdiv", static_cast<int>(_gin.symdiv));

        std::vector<double> alb(_gin.albedo.begin(), _gin.albedo.end());
        geo.createDataSet("albedo", alb);

        // Core map (2-D jagged → flatten with row lengths)
        std::vector<int>         core_row_len;
        std::vector<std::string> core_flat;
        for (const auto& row : _gin.core) {
            core_row_len.push_back(static_cast<int>(row.size()));
            for (const auto& c : row)
                core_flat.push_back(c);
        }
        geo.createDataSet("core_row_len", core_row_len);
        geo.createDataSet("core_flat", core_flat);

        // Batch map
        std::vector<std::string> batch_keys;
        std::vector<int>         batch_len;
        std::vector<std::string> batch_flat;
        for (const auto& [key, layers] : _gin.batch) {
            batch_keys.push_back(key);
            batch_len.push_back(static_cast<int>(layers.size()));
            for (const auto& lid : layers)
                batch_flat.push_back(lid);
        }
        geo.createDataSet("batch_keys", batch_keys);
        geo.createDataSet("batch_len", batch_len);
        geo.createDataSet("batch_flat", batch_flat);
    }

    // /burnup [nxyz]
    {
        std::vector<int> burn(xs.burn_data(), xs.burn_data() + nxyz);
        file.createDataSet("burnup", burn);
    }

    // Fine rod thermal fluence is the RDPL restart state.
    {
        file.createDataSet("axial_rod_division", xs.axial_rod_division());
        file.createDataSet("fine_rod_thermal_fluence",
                           xs.fine_rod_thermal_fluence_data());
    }

    if (!xs.rod_groups().empty()) {
        auto                     rods = file.createGroup("rod_state");
        std::vector<std::string> names;
        std::vector<double>      insertions;
        names.reserve(xs.rod_groups().size());
        insertions.reserve(xs.rod_groups().size());
        for (const auto& [name, group] : xs.rod_groups()) {
            names.push_back(name);
            insertions.push_back(group.insertion);
        }
        rods.createDataSet("names", names);
        rods.createDataSet("insertions", insertions);
    }

    // /isotope_density [nxyz x niso]
    if (niso > 0) {
        std::vector<double> iden_flat(static_cast<size_t>(nxyz) * niso, 0.0);
        for (int l = 0; l < nxyz; ++l) {
            for (size_t i = 0; i < static_cast<size_t>(niso); ++i)
                iden_flat[static_cast<size_t>(l) * niso + i] = xs.iden(i, l);
        }
        const iowriter::Dims space({static_cast<size_t>(nxyz), static_cast<size_t>(niso)});
        auto                 ds = file.createDataSet<double>("isotope_density", space);
        ds.write_raw(iden_flat.data());
    }

    // /th_state: bppm, tful, tmod, dmod [nxyz each]
    {
        auto                th = file.createGroup("th_state");
        std::vector<double> bppm(nxyz), tful(nxyz), tmod(nxyz), dmod(nxyz);
        for (int l = 0; l < nxyz; ++l) {
            bppm[l] = g.bppm(l);
            tful[l] = g.tful(l);
            tmod[l] = g.tmod(l);
            dmod[l] = g.dmod(l);
        }
        th.createDataSet("bppm", bppm);
        th.createDataSet("tful", tful);
        th.createDataSet("tmod", tmod);
        th.createDataSet("dmod", dmod);
    }

    // /th_params: global TH input scalars (pressure, temperatures, mass flux, rated power)
    {
        auto thp = file.createGroup("th_params");
        thp.createDataSet("pressure", g.pressure());
        thp.createDataSet("inlet_temp", g.inlet_temp());
        thp.createDataSet("outlet_temp", g.outlet_temp());
        thp.createDataSet("mass_flow_rate", g.mass_flow_rate());
        thp.createDataSet("use_mass_flow_rate", g.use_mass_flow_rate() ? 1 : 0);
        thp.createDataSet("rated_power", g.rated_power());
    }

    // /flux: Phif [ng * nxyz]
    {
        const size_t        sz = static_cast<size_t>(ng) * nxyz;
        std::vector<double> flux(g.Phif(), g.Phif() + sz);
        file.createDataSet("flux", flux);
    }

    rec.closeFile();
    rec.submit();

    PLOG_INFO << "Restart file saved: " << filepath
              << "  (step=" << step << ", efpd=" << efpd << ", keff=" << eigv << ")";
}

// Parse "i,j/cycle/rotation" shuffle notation.
// Returns false for plain assembly names and throws on malformed input.
bool IO::TryParseShuffleEntry(const std::string& entry, int tgt_row, int tgt_col,
                              ShuffleSpec& out) {
    const size_t slash1 = entry.find('/');
    if (slash1 == std::string::npos) return false; // plain name, not a shuffle entry

    const size_t slash2 = entry.find('/', slash1 + 1);
    if (slash2 == std::string::npos || entry.find('/', slash2 + 1) != std::string::npos)
        throw std::runtime_error(std::format(
            "IO: malformed shuffle entry \"{}\" at core({},{}). Expected \"i,j/cycle/rotation\".",
            entry, tgt_row, tgt_col));

    try {
        const std::string ij_str  = entry.substr(0, slash1);
        const std::string cyc_str = entry.substr(slash1 + 1, slash2 - slash1 - 1);
        const std::string rot_str = entry.substr(slash2 + 1);

        const size_t comma = ij_str.find(',');
        if (comma == std::string::npos)
            throw std::runtime_error("missing comma in i,j");

        const int si  = std::stoi(ij_str.substr(0, comma));
        const int sj  = std::stoi(ij_str.substr(comma + 1));
        const int cyc = std::stoi(cyc_str);
        const int rot = std::stoi(rot_str);

        if (rot != 0 && rot != 90 && rot != 180 && rot != 270)
            throw std::runtime_error(std::format(
                "IO: invalid rotation {} in \"{}\". Must be 0, 90, 180, or 270.",
                rot, entry));

        out = {tgt_row, tgt_col, si, sj, cyc, rot};
        return true;
    } catch (const std::runtime_error&) {
        throw;
    } catch (...) {
        throw std::runtime_error(std::format(
            "IO: failed to parse shuffle entry \"{}\" at core({},{}).",
            entry, tgt_row, tgt_col));
    }
}

// Copy fuel-carried restart state into the shuffled target positions. Fine
// rod-material fluence is intentionally not shuffled with fuel assemblies.
void IO::ApplyShuffle() {
    Chiffon::Hdf5Guard hdf5_guard;
    const int  ndivxy = _g.ndivxy();
    const int  symang = _g.symang();
    const bool symdiv = _g.symdiv();
    // Which quarter-core fold the geometry was built with (see Geometry::Initialize).
    const bool rotfold = (symang == 90 && _g.symopt() == 0);
    const int  nz     = _g.nz();
    const int  nxy    = _g.nxy();
    const int  niso   = static_cast<int>(Chiffon::Isotope::niso);
    const int  n      = ndivxy;

    // 1. Cache each referenced restart cycle once.
    struct CycleSnapshot {
        std::vector<int>    burn;
        std::vector<double> iden;
        int                 nxyz = 0;
    };
    std::map<int, CycleSnapshot> cache;
    for (const auto& spec : _shuffle_specs) {
        if (cache.contains(spec.cycle)) continue;
        HighFive::File rfile(_restart_files.at(spec.cycle), HighFive::File::ReadOnly);
        CycleSnapshot& snap = cache[spec.cycle];
        rfile.getDataSet("burnup").read(snap.burn);
        snap.nxyz = static_cast<int>(snap.burn.size());
        snap.iden.resize(static_cast<size_t>(snap.nxyz) * niso, 0.0);
        if (niso > 0 && rfile.exist("isotope_density")) {
            auto                      density = rfile.getDataSet("isotope_density");
            const std::vector<size_t> expectedShape = {
                static_cast<size_t>(snap.nxyz), static_cast<size_t>(niso)};
            if (density.getDimensions() != expectedShape ||
                density.getElementCount() != snap.iden.size())
                throw std::runtime_error(std::format(
                    "Shuffle: cycle {} isotope_density dimensions do not "
                    "match burnup and isotope counts.",
                    spec.cycle));
            density.read_raw<double>(snap.iden.data());
            const double cooling_days = _restart_cooling_days[spec.cycle];
            if (cooling_days > 0.0) {
                const int substeps = _restart_cooling_substeps[spec.cycle];
                _xs.DecayIsotopeDensityFlat(snap.iden, snap.nxyz, cooling_days, substeps);
                PLOG_INFO << std::format(
                    "Shuffle: cycle {} isotope densities cooled for {:.6g} days in {} substeps.",
                    spec.cycle, cooling_days, substeps);
            }
        }
    }

    // 2. Apply the cached burnup and isotope data to the shuffled target locations.
    std::vector<double> nd_buf(niso);
    for (const auto& spec : _shuffle_specs) {
        const auto& snap = cache.at(spec.cycle);

        const bool source_cut = IsSymmetryCutAssembly(symang, symdiv,
                                                      spec.source_row, spec.source_col);
        const bool target_cut = IsSymmetryCutAssembly(symang, symdiv,
                                                      spec.target_row, spec.target_col);
        if ((source_cut || target_cut) && n % 2 != 0)
            throw std::runtime_error(std::format(
                "IO: cut-assembly shuffle requires even xydivision; got {} at source ({},{}) target ({},{}).",
                n, spec.source_row, spec.source_col, spec.target_row, spec.target_col));

        const int src_row_count = VisibleAssemblyRows(symang, symdiv, spec.source_row, n);
        const int src_col_count = VisibleAssemblyCols(symang, symdiv, spec.source_col, n);
        const int tgt_row_count = VisibleAssemblyRows(symang, symdiv, spec.target_row, n);
        const int tgt_col_count = VisibleAssemblyCols(symang, symdiv, spec.target_col, n);
        const int src_row_off   = (spec.source_row == 0 && symang == 90 && symdiv) ? n - src_row_count : 0;
        const int src_col_off   = (spec.source_col == 0 && symang == 90 && symdiv) ? n - src_col_count : 0;
        const int tgt_row_off   = (spec.target_row == 0 && symang == 90 && symdiv) ? n - tgt_row_count : 0;
        const int tgt_col_off   = (spec.target_col == 0 && symang == 90 && symdiv) ? n - tgt_col_count : 0;
        const int full_nodes    = n * n * nz;

        std::vector<char>   full_valid(full_nodes, 0);
        std::vector<int>    full_burn(full_nodes, 0);
        std::vector<double> full_iden(static_cast<size_t>(full_nodes) * niso, 0.0);

        // Copy the visible nodes of one assembly of the source cycle into the
        // full-assembly buffer. `rot_steps` rotates each node's local index by that
        // many 90-degree steps on the way in, which is how the rotational fold's
        // partner half (and the centre assembly's own other three quarters) are
        // expressed in this assembly's frame. Slots already written are kept.
        auto gather = [&](int a_row, int a_col, int rot_steps) {
            const int row_count = VisibleAssemblyRows(symang, symdiv, a_row, n);
            const int col_count = VisibleAssemblyCols(symang, symdiv, a_col, n);
            const int row_off   = (a_row == 0 && symang == 90 && symdiv) ? n - row_count : 0;
            const int col_off   = (a_col == 0 && symang == 90 && symdiv) ? n - col_count : 0;

            for (int li = 0; li < row_count; ++li) {
                for (int lj = 0; lj < col_count; ++lj) {
                    const int src_x = AssemblyNodeX(symang, symdiv, a_col, lj, n);
                    const int src_y = AssemblyNodeY(symang, symdiv, a_row, li, n);
                    if (src_x < 0 || src_x >= _g.nx() || src_y < 0 || src_y >= _g.ny()) continue;
                    const int src_l2d = _g.ijtol(src_x, src_y);
                    if (src_l2d < 0) continue;

                    int full_i = row_off + li;
                    int full_j = col_off + lj;
                    for (int s = 0; s < rot_steps; ++s)
                        RotateAssemblyIndexInverse(full_i, full_j, n);

                    for (int k = 0; k < nz; ++k) {
                        const int src_lk = src_l2d + k * nxy;
                        if (src_lk >= snap.nxyz) continue;

                        const int full_lk = (k * n + full_i) * n + full_j;
                        if (full_valid[full_lk]) continue;
                        full_valid[full_lk] = 1;
                        full_burn[full_lk]  = snap.burn[src_lk];
                        if (niso > 0) {
                            const auto src_off  = static_cast<ptrdiff_t>(src_lk) * niso;
                            const auto full_off = static_cast<ptrdiff_t>(full_lk) * niso;
                            std::copy_n(snap.iden.data() + src_off, niso,
                                        full_iden.data() + full_off);
                        }
                    }
                }
            }
        };

        gather(spec.source_row, spec.source_col, 0);

        if (source_cut && rotfold) {
            // Under the 90-degree fold a symmetry-cut assembly's missing part is not
            // its own mirror image: it is the rotated image of the partner entry on
            // the other arm of the quarter map. Quarter-map row 0 column c and row c
            // column 0 are the two halves of ONE physical assembly, and the centre
            // assembly's quarter carries all four of its own.
            if (spec.source_row == 0 && spec.source_col == 0) {
                gather(0, 0, 1);
                gather(0, 0, 2);
                gather(0, 0, 3);
            } else if (spec.source_row == 0) {
                // partner (c,0) node p sits at this assembly's index rot^-1(p)
                gather(spec.source_col, 0, 1);
            } else {
                // partner (0,r) node p sits at this assembly's index rot(p)
                gather(0, spec.source_row, 3);
            }

            for (int full_lk = 0; full_lk < full_nodes; ++full_lk)
                if (!full_valid[full_lk])
                    throw std::runtime_error(std::format(
                        "IO: the 90-degree rotational fold could not complete the shuffle source "
                        "at cycle{}/({},{}) -- its partner half on the other arm of the quarter "
                        "map is missing.",
                        spec.cycle, spec.source_row, spec.source_col));
        } else if (source_cut) {
            for (int k = 0; k < nz; ++k) {
                for (int full_i = 0; full_i < n; ++full_i) {
                    const int mirror_i = (spec.source_row == 0 && symang == 90 && symdiv &&
                                          full_i < src_row_off)
                                             ? n - 1 - full_i
                                             : full_i;
                    for (int full_j = 0; full_j < n; ++full_j) {
                        const int mirror_j  = (spec.source_col == 0 && symang == 90 && symdiv &&
                                              full_j < src_col_off)
                                                  ? n - 1 - full_j
                                                  : full_j;
                        const int full_lk   = (k * n + full_i) * n + full_j;
                        const int mirror_lk = (k * n + mirror_i) * n + mirror_j;
                        if (full_valid[full_lk] || !full_valid[mirror_lk]) continue;

                        full_valid[full_lk]   = 1;
                        full_burn[full_lk]    = full_burn[mirror_lk];
                        if (niso > 0) {
                            const auto full_off   = static_cast<ptrdiff_t>(full_lk) * niso;
                            const auto mirror_off = static_cast<ptrdiff_t>(mirror_lk) * niso;
                            std::copy_n(full_iden.data() + mirror_off, niso,
                                        full_iden.data() + full_off);
                        }
                    }
                }
            }
        }

        const int rot = (n == 1) ? 0 : spec.rotation;

        for (int li = 0; li < tgt_row_count; ++li) {
            for (int lj = 0; lj < tgt_col_count; ++lj) {
                const int tgt_full_i = tgt_row_off + li;
                const int tgt_full_j = tgt_col_off + lj;

                const int tgt_x = AssemblyNodeX(symang, symdiv, spec.target_col, lj, n);
                const int tgt_y = AssemblyNodeY(symang, symdiv, spec.target_row, li, n);
                if (tgt_x < 0 || tgt_x >= _g.nx() || tgt_y < 0 || tgt_y >= _g.ny()) continue;
                const int tgt_l2d = _g.ijtol(tgt_x, tgt_y);
                if (tgt_l2d < 0) continue;

                // Mirror fold: a symmetry-cut target node stands for itself AND its
                // mirror image inside the same assembly, so it receives their average.
                // Rotational fold: the visible nodes are a fundamental domain -- the
                // other half of the physical assembly is carried by the partner entry
                // on the other arm of the quarter map -- so each node takes its own
                // value and nothing is averaged.
                int target_i[2] = {tgt_full_i, tgt_full_i};
                int target_j[2] = {tgt_full_j, tgt_full_j};
                int ni          = 1;
                int nj          = 1;
                if (!rotfold && spec.target_row == 0 && symang == 90 && symdiv) {
                    const int mirror_i = n - 1 - tgt_full_i;
                    if (mirror_i != tgt_full_i) target_i[ni++] = mirror_i;
                }
                if (!rotfold && spec.target_col == 0 && symang == 90 && symdiv) {
                    const int mirror_j = n - 1 - tgt_full_j;
                    if (mirror_j != tgt_full_j) target_j[nj++] = mirror_j;
                }

                for (int k = 0; k < nz; ++k) {
                    int    count    = 0;
                    double burn_sum = 0.0;
                    std::fill(nd_buf.begin(), nd_buf.end(), 0.0);

                    for (int ii = 0; ii < ni; ++ii) {
                        for (int jj = 0; jj < nj; ++jj) {
                            int src_full_i, src_full_j;
                            switch (rot) {
                            case 90:
                                src_full_i = target_j[jj];
                                src_full_j = n - 1 - target_i[ii];
                                break;
                            case 180:
                                src_full_i = n - 1 - target_i[ii];
                                src_full_j = n - 1 - target_j[jj];
                                break;
                            case 270:
                                src_full_i = n - 1 - target_j[jj];
                                src_full_j = target_i[ii];
                                break;
                            default:
                                src_full_i = target_i[ii];
                                src_full_j = target_j[jj];
                                break;
                            }

                            const int src_full_lk = (k * n + src_full_i) * n + src_full_j;
                            if (!full_valid[src_full_lk]) continue;

                            ++count;
                            burn_sum += static_cast<double>(full_burn[src_full_lk]);
                            if (niso > 0) {
                                const auto off = static_cast<ptrdiff_t>(src_full_lk) * niso;
                                for (int iso = 0; iso < niso; ++iso)
                                    nd_buf[iso] += full_iden[off + iso];
                            }
                        }
                    }
                    if (count == 0) continue;

                    const int    tgt_lk        = tgt_l2d + k * nxy;
                    const double inv_count     = 1.0 / static_cast<double>(count);
                    _xs.burn(tgt_lk) = static_cast<int>(std::llround(burn_sum * inv_count));

                    if (niso > 0) {
                        for (double& value : nd_buf) value *= inv_count;
                        _xs.setNodeIden(tgt_lk, nd_buf);
                    }
                }
            }
        }

        PLOG_INFO << std::format("Shuffle: core({},{}) ← cycle{}/({},{}) rot={}°",
                                 spec.target_row, spec.target_col,
                                 spec.cycle,
                                 spec.source_row, spec.source_col,
                                 rot);
    }
}

// Reconstruct a GeometryInput from the geometry group saved in a restart HDF5 file.
GeometryInput IO::LoadGeometryFromRestart(const std::string& filepath) {
    Chiffon::Hdf5Guard hdf5_guard;
    HighFive::File file(filepath, HighFive::File::ReadOnly);
    auto           geo = file.getGroup("geometry");

    GeometryInput gin;
    geo.getDataSet("ng").read(gin.ng);
    geo.getDataSet("nz").read(gin.nz);
    geo.getDataSet("ndivxy").read(gin.ndivxy);
    geo.getDataSet("npins").read(gin.npins);
    // Optional: restart files written before 2026-09-04 carry no fuel-rod count,
    // and 0 is the honest answer for one -- it resolves to the legacy divisor
    // those runs actually used.
    if (geo.exist("nfrod")) geo.getDataSet("nfrod").read(gin.nfrod);
    if (geo.exist("tf_table_name")) {
        std::vector<std::string> name, path;
        geo.getDataSet("tf_table_name").read(name);
        geo.getDataSet("tf_table_path").read(path);
        gin.tf_table.name = name.empty() ? std::string() : name.front();
        gin.tf_table.path = (path.empty() || path.front() == "-") ? std::string()
                                                                  : path.front();
        if (geo.exist("tf_table_lpd")) {
            geo.getDataSet("tf_table_lpd").read(gin.tf_table.lpd);
            geo.getDataSet("tf_table_bu").read(gin.tf_table.bu);
            geo.getDataSet("tf_table_dt").read(gin.tf_table.dt);
        }
    }
    geo.getDataSet("hx").read(gin.hx);
    geo.getDataSet("hy").read(gin.hy);
    geo.getDataSet("hz").read(gin.hz);
    geo.getDataSet("symang").read(gin.symang);

    int symopt_i = 0, symdiv_i = 0;
    geo.getDataSet("symopt").read(symopt_i);
    geo.getDataSet("symdiv").read(symdiv_i);
    gin.symopt = static_cast<bool>(symopt_i);
    gin.symdiv = static_cast<bool>(symdiv_i);

    std::vector<double> alb;
    geo.getDataSet("albedo").read(alb);
    for (size_t i = 0; i < alb.size() && i < gin.albedo.size(); ++i)
        gin.albedo[i] = alb[i];

    // Reconstruct core map from flat arrays.
    std::vector<int>         core_row_len;
    std::vector<std::string> core_flat;
    geo.getDataSet("core_row_len").read(core_row_len);
    geo.getDataSet("core_flat").read(core_flat);
    int flat_idx = 0;
    for (int rlen : core_row_len) {
        gin.core.emplace_back(core_flat.begin() + flat_idx,
                              core_flat.begin() + flat_idx + rlen);
        flat_idx += rlen;
    }

    // Reconstruct batch map from flat arrays.
    std::vector<std::string> batch_keys, batch_flat;
    std::vector<int>         batch_len;
    geo.getDataSet("batch_keys").read(batch_keys);
    geo.getDataSet("batch_len").read(batch_len);
    geo.getDataSet("batch_flat").read(batch_flat);
    flat_idx = 0;
    for (size_t i = 0; i < batch_keys.size(); ++i) {
        gin.batch[batch_keys[i]] = std::vector<std::string>(
            batch_flat.begin() + flat_idx,
            batch_flat.begin() + flat_idx + batch_len[i]);
        flat_idx += batch_len[i];
    }

    PLOG_INFO << "Geometry loaded from restart file: " << filepath;
    return gin;
}
