#pragma once
#include "Geometry.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rasbery {

/// @brief Per-step output options parsed from the JSON "print" block.
struct PrintOpt {
    bool             summary  = true;  // write this step into the summary table
    bool             pin_info = false; // run PPR and export pin-wise power map
    bool             pin_flux = false; // also reconstruct pin-wise flux with fmap when explicitly requested
    bool             xs_info  = false; // export cross-section information per node
    bool             save     = false; // save restart file at this step
    std::vector<int> node_monitor;     // list of node indices to export detailed XS comparison
};

/// @brief TH coupling modes
enum class THMode {
    NONE,     // No TH feedback, fixed to global values
    STEADY,   // Only consider enthalpy rise, ignore transient effects
    TRANSIENT // Time-dependent TH coupling
};

/// @brief Determines what unknown the eigenvalue solver searches for
enum class SearchType {
    KEFF,   // Search eigenvalue (no critical search)
    BORON,  // Search critical soluble boron concentration [ppm]
    RODCRIT // Search critical continuous rod-step parameter
};

/// @brief Identifies the kind of calculation each schedule entry requests
enum class ScheduleType {
    STANDARD,   // Setup calculation standard for next steps
    DEPLETION,  // burn fuel for a given time at a given power
    DERIVATIVE, // perturb state variables and re-solve (for reactivity coefficients)
    ROD         // insert or move the rod
};

/// @brief How a critical search terminated.  Published to the result file so a step whose
/// rod position / boron was accepted without meeting the search tolerance is identifiable
/// downstream instead of being indistinguishable from a converged one.
enum class SearchExit {
    NONE          = 0, // no critical search on this step
    CONVERGED     = 1, // |k_eff - target| < search tolerance at the accepted point
    BEST_FALLBACK = 2, // not converged; fell back to the best observed trial point
    UNCONVERGED   = 3  // not converged and no better observed point was available
};

struct SearchMemory {
    bool   has_boron_secant  = false;
    double boron_secant_dkdx = 0.0;
    bool   has_rod_secant    = false;
    double rod_secant_x      = 1.0;
    double rod_secant_dkdx   = 0.0;
};

inline constexpr int    kMaxEigenIter     = 200;
inline constexpr int    kMaxSearchIter    = 300;
inline constexpr int    kMaxThIter        = 10;
inline constexpr double kEigvTol          = 1.0e-6;
inline constexpr double kCritSearchTol    = 1.0e-5;
inline constexpr double kRodCritSearchTol = 1.0e-5;
inline constexpr double kThTol            = 1.0e-6;
inline constexpr double kTempSearchTol    = 0.01;
inline constexpr double kBoronSearchTol   = 0.01;
inline constexpr double kRodSearchTol     = 0.01;
inline constexpr double kSearchRelax      = 1.0;
inline constexpr double kSearchLow        = 0.0;
inline constexpr double kSearchHigh       = 1.0;
inline constexpr double kSlopeFreezeThres = 0.01;
inline constexpr double kMinSecantDenom   = 1.0e-12;
// Rod-step span below which a sign-change bracket is indistinguishable from rod-cusping
// k_eff noise: bisecting it cannot resolve the root, it only re-proposes the same point.
inline constexpr double kBracketMinSpan   = 1.0e-6;
inline constexpr double kBoronProbe       = 50.0;
inline constexpr double kRodProbe         = 0.25;

/// @brief one step in the calculation sequence; The Driver reads these sequentially and executes each one.
///        Contains both input parameters (what to do) and output results (what happened).
struct Schedule {
    PrintOpt     print_opt;
    ScheduleType type = ScheduleType::STANDARD;

    // Depletion parameters
    double time            = 0.0;    // time per schedule step [days]
    double until_boron_ppm = 0.0;    // >0: repeat this depletion entry until critical boron reaches this target (natural EOC)
    double burnup          = 0.0;    // burnup increment per schedule step [MWd/kgHM]
    double rate            = 100.0;  // thermal power rate (%)
    double rated_power     = 4200.0; // rated thermal power for the whole core [MW]
    double actual_power    = 4200.0; // rated power * rate
    bool   use_burnup_time = false;  // convert burnup to depletion time at runtime
    bool   xenon_transient = false;  // default: overwrite equilibrium Xe (set "xenon":"transient" to opt in)

    // Basic core conidition
    double bppm = 500.0; // initial soluble boron [ppm]
    double tful = 900.0; // initial fuel temperature [K]
    double tmod = 580.0; // initial moderator temperature [K]
    double dmod = 0.71;  // initial moderator density [g/cc]

    // TH condition
    THMode              thmode             = THMode::STEADY;
    double              pressure           = 15.0;   // system pressure [MPa]
    double              inlet_temp         = 580.0;  // coolant inlet temperature [K]
    double              outlet_temp        = 600.0;  // coolant outlet temperature [K]
    double              mass_flow_rate     = 1000.0; // coolant mass flux [kg/s/m^2]
    bool                use_mass_flow_rate = false;  // true when mass flow rate is explicitly provided
    double              fuel_temp_rise_scale = 1.0;  // multiplier on tabulated Tfuel-Tcoolant
    std::vector<double> th_profile_power;            // rated power percent breakpoints for TH temperature profiles
    std::vector<double> th_inlet_profile;            // inlet temperature at th_profile_power breakpoints [K]
    std::vector<double> th_outlet_profile;           // outlet temperature at th_profile_power breakpoints [K]

    // Derivative perturbation delta
    double delta_tful = 0.0; // fuel temperature perturbation [K]
    double delta_tmod = 0.0; // moderator temperature perturbation [K]
    double delta_dmod = 0.0; // moderator density perturbation [g/cc]
    double delta_bppm = 0.0; // boron concentration perturbation [ppm]
    double delta_xe   = 0.0; // Xe-135 number density perturbation [at/b-cm]
    double delta_sm   = 0.0; // Sm-149 number density perturbation [at/b-cm]

    // Search condition
    SearchType searchType          = SearchType::KEFF;
    int        max_outer_iter      = kMaxEigenIter;  // Max CMFD+Nodal eigen iterations
    int        max_search_iter     = kMaxSearchIter; // Max critical-search iterations
    int        max_th_iter         = kMaxThIter;     // Max TH-neutronics feedback iterations
    double     tolerance_keff      = kEigvTol;       // Eigenvalue iteration tolerance
    double     tolerance_search    = kCritSearchTol; // Critical-search tolerance on |k_eff - target|
    double     rodcrit_search_floor = 0.0;           // Lower bound on rod-crit search tol (set when cusping kinks add keff noise)
    double     tolerance_th        = kThTol;         // TH feedback convergence on |Δk_eff|
    double     target_keff         = 1.0;            // Target eigenvalue for search
    double     tolerance_tmod      = kTempSearchTol; // Search tolerance for moderator temperature (K)
    double     tolerance_tfuel     = kTempSearchTol; // Search tolerance for fuel temperature (K)
    double     search_boron_ppm    = 0.0;
    double     tolerance_boron     = kBoronSearchTol; // Search tolerance for boron concentration (PPM)
    double     tolerance_rodsearch = kRodSearchTol;   // Search tolerance for rod insertion profile
    double     search_relaxation   = kSearchRelax;
    double     search_low          = kSearchLow;
    double     search_hi           = kSearchHigh;
    double     slope_freeze_thres  = kSlopeFreezeThres;
    double     min_secant_denom    = kMinSecantDenom; // Minimum |Δx| or |Δk_eff| for secant update
    double     bracket_min_span    = kBracketMinSpan; // Below this a bracket is noise, not a root
    double     search_boron_probe  = kBoronProbe;     // Probe step for bootstrapping boron search [ppm]
    double     search_rod_probe    = kRodProbe;       // Probe step for bootstrapping rod search [rod-step]

    // Rod insertion parameters (group name → insertion depth in cm, measured from the top)
    std::map<std::string, double> rod_insertions;

    // Runtime search state managed by Driver and persisted across multiple SolveLoop calls
    // within the same scheduler step (e.g. depletion predictor/corrector + final solve).
    bool   search_initialized               = false;
    bool   search_seeded_from_previous_step = false;
    bool   search_has_prev                  = false;
    bool   search_has_bracket               = false;
    bool   search_has_best                  = false;
    bool   search_slope_frozen              = false;
    int    search_iteration                 = 0;
    double search_current_x                 = 0.0;
    double search_prev_x                    = 0.0;
    double search_prev_eigv                 = 0.0;
    double search_frozen_slope              = 0.0;
    double search_best_x                    = 0.0;
    double search_best_residual             = 0.0;
    double search_bracket_lo_x              = 0.0;
    double search_bracket_lo_residual       = 0.0;
    double search_bracket_hi_x              = 0.0;
    double search_bracket_hi_residual       = 0.0;

    // Termination bookkeeping for the critical search (published to the result file).
    int    search_exit_status = static_cast<int>(SearchExit::NONE);
    double search_exit_dk     = 0.0; // |k_eff - target| actually accepted
    double search_exit_tol    = 0.0; // tolerance that dk was judged against
    int    search_stall_count = 0;   // flux limit-cycle events survived during this solve

    // OUTPUT parameters
    // 1. Calculation options
    int    step    = 0;   // global step
    int    substep = 1;   // The number of substeps (1 = no subdivision)
    double efpd    = 0.0; // cumulative effective full-power days at end of step
    double bu_avg  = 0.0; // core-average burnup [MWd/kgHM]

    // 2. Reactivity
    double eigv     = 1.0; // effective multiplication factor
    double rho      = 0.0; // reactivity (k-1)/k
    double ppm      = 0.0; // soluble boron concentration [ppm]
    double rod_step = 0.0; // converged continuous rod-step parameter

    // 3. safety parameters
    double ao     = 0.0; // axial offset
    double asi    = 0.0; // axial shape index
    double fqn    = 1.0; // 3-D nodal peaking factor (hot spot)
    double frn    = 1.0; // radial (assembly) peaking factor
    double fqp    = 1.0; // 3-D pin peaking factor
    double frp    = 1.0; // radial pin peaking factor
    double xe_avg = 0.0; // Xe-135 average number density [at/b-cm]
    double xe_ao  = 0.0; // Xe-135 axial offset
    double sm_avg = 0.0; // Sm-149 average number density [at/b-cm]
    double sm_ao  = 0.0; // Sm-149 axial offset
    double gd_avg = 0.0; // lumped-Gd ("640000") average number density [at/b-cm]

    // 4. Thermal-hydraulics parameters
    double tf_avg = 0.0; // volume-averaged fuel temperature [K]
    double tf_max = 0.0; // peak fuel temperature [K]
    double tm_avg = 0.0; // volume-averaged moderator temperature [K]
    double tm_max = 0.0; // peak moderator temperature [K]
    double dm_avg = 0.0; // volume-averaged moderator density [g/cc]
    double dm_max = 0.0; // peak moderator density [g/cc]

    // 5. Assembly-wise information
    std::vector<std::string> asm_type;  // assembly batch name per location
    std::vector<double>      asm_power; // normalized assembly power
    std::vector<double>      asm_burn;  // assembly-average burnup [MWd/kgHM]
    std::vector<double>      asm_kinf;  // assembly k-infinity

    // 6. Axial information
    std::vector<double> ax_power;
    std::vector<double> ax_tful;
    std::vector<double> ax_tmod;
    std::vector<double> ax_dmod;

    // 7. WP9-D: the critical search's convergence history.  TELEMETRY ONLY.
    //
    // ON THE OUTPUT SIDE OF THE LINE, and that placement is the point.  The
    // block above `// OUTPUT parameters` is the search's STATE, and
    // tools/test_gpu_physics_interface_contract.py requires every field of it
    // to have a DeviceSearchState counterpart, because the device solver
    // mirrors all of it.  These are not state: nothing reads them, least of
    // all the device.  They are what the search DID.
    //
    // WHY IT IS HERE AND NOT IN THE DRIVER.  The GA evaluator plan's outer
    // census charges 707 outers -- 15.3 % of a case -- to the boron search,
    // spread over 137 committed trials, and says the cost is CANDIDATE
    // DEPENDENT: the worse a loading pattern is in reactivity, the more trials
    // it buys.  Before anything is done about that, the 238 runner has to be
    // able to say how many outers a statepoint's search actually costs and WHY
    // the proposals were what they were -- a bootstrap probe, a carried slope, a
    // secant, or a bisection inside a bracket are four different stories about
    // the same trial count.  The classification can only be made where `method`
    // is decided, which is here.
    //
    // COST AND NEUTRALITY.  Plain integer members of an object that already
    // exists per statepoint, incremented once per PROPOSAL (137 per run).
    // Nothing here is read by the solver: no branch, no tolerance and no
    // proposal depends on any of these fields, which is what makes the receipt
    // trajectory-neutral rather than merely cheap.
    long long search_n_proposals  = 0;  ///< ProposeNextSearchPoint calls
    long long search_n_refused    = 0;  ///< proposals that returned no next point
    long long search_n_secant     = 0;  ///< two-sample secant steps
    long long search_n_carry      = 0;  ///< first steps taken on a CARRIED slope
    long long search_n_probe      = 0;  ///< bootstrap/probe steps (no slope yet)
    long long search_n_bisect     = 0;  ///< bisections inside a sign-change bracket
    double    search_first_x      = 0.0; ///< the initial guess this statepoint started from
    double    search_last_dx      = 0.0; ///< |x_new - x_old| of the LAST committed step

    /// Re-armed per statepoint by the Driver.  Deliberately NOT cleared by
    /// ResetSearchState(): that runs at every SolveLoop entry, and a statepoint
    /// with substeps enters SolveLoop several times -- clearing there would
    /// report the last substep's search as the statepoint's.
    void ResetSearchTelemetry() {
        search_n_proposals = 0;
        search_n_refused   = 0;
        search_n_secant    = 0;
        search_n_carry     = 0;
        search_n_probe     = 0;
        search_n_bisect    = 0;
        search_first_x     = 0.0;
        search_last_dx     = 0.0;
    }

    [[nodiscard]] bool hasCriticalSearch() const {
        return searchType == SearchType::BORON || searchType == SearchType::RODCRIT;
    }

    [[nodiscard]] bool usesTHFeedback() const {
        return thmode == THMode::STEADY;
    }

    [[nodiscard]] bool keepSearchBetweenSolves() const {
        return type == ScheduleType::DEPLETION && hasCriticalSearch();
    }

    [[nodiscard]] double powerFraction() const {
        return rate * 0.01;
    }

    [[nodiscard]] double thermalPower() const {
        return rated_power * powerFraction();
    }

    void ResetSearchState() {
        search_initialized               = false;
        search_seeded_from_previous_step = false;
        search_has_prev                  = false;
        search_iteration                 = 0;
        search_has_best                  = false;
        search_current_x                 = 0.0;
        search_prev_x                    = 0.0;
        search_prev_eigv                 = 0.0;
        search_best_x                    = 0.0;
        search_best_residual             = 0.0;
        search_has_bracket               = false;
        search_bracket_lo_x              = 0.0;
        search_bracket_lo_residual       = 0.0;
        search_bracket_hi_x              = 0.0;
        search_bracket_hi_residual       = 0.0;
        ResetSearchExitStatus();
    }

    void ApplyTHProfile() {
        if (th_profile_power.empty())
            return;

        const int profile_count  = static_cast<int>(th_profile_power.size());
        int       lower_index    = 0;
        int       upper_index    = 0;
        double    profile_weight = 0.0;

        if (profile_count == 1 || rate <= th_profile_power.front()) {
            lower_index = 0;
            upper_index = 0;
        } else if (rate >= th_profile_power.back()) {
            lower_index = profile_count - 1;
            upper_index = profile_count - 1;
        } else {
            for (int profile_index = 1; profile_index < profile_count; ++profile_index) {
                if (rate <= th_profile_power[profile_index]) {
                    lower_index             = profile_index - 1;
                    upper_index             = profile_index;
                    const double power_span = th_profile_power[upper_index] - th_profile_power[lower_index];
                    profile_weight          = (rate - th_profile_power[lower_index]) / power_span;
                    break;
                }
            }
        }

        if (!th_inlet_profile.empty()) {
            inlet_temp = th_inlet_profile[lower_index] +
                         profile_weight * (th_inlet_profile[upper_index] -
                                           th_inlet_profile[lower_index]);
        }
        if (!th_outlet_profile.empty()) {
            outlet_temp = th_outlet_profile[lower_index] +
                          profile_weight * (th_outlet_profile[upper_index] -
                                            th_outlet_profile[lower_index]);
        }
    }

    void PrepareForStep(double core_hm_kg = 0.0) {
        ApplyTHProfile();
        actual_power = thermalPower();
        if (type != ScheduleType::DEPLETION || !use_burnup_time)
            return;
        if (core_hm_kg <= 0.0)
            throw std::runtime_error("Schedule: burnup-based depletion needs positive heavy metal mass.");
        if (actual_power <= 0.0)
            throw std::runtime_error("Schedule: burnup-based depletion needs positive thermal power.");
        time = burnup * core_hm_kg / actual_power;
    }

    void ApplyToGeometry(Geometry& geometry) const {
        geometry.pressure()           = pressure;
        geometry.inlet_temp()         = inlet_temp;
        geometry.outlet_temp()        = outlet_temp;
        geometry.mass_flow_rate()     = mass_flow_rate;
        geometry.use_mass_flow_rate() = use_mass_flow_rate;
        geometry.rated_power()        = rated_power;
        geometry.fuel_temp_rise_scale() = fuel_temp_rise_scale;
    }

    void StartCriticalSearch(SearchMemory& memory, double current_bppm, double rod_max_step) {
        if (search_initialized || !hasCriticalSearch())
            return;
        search_initialized = true;

        if (searchType == SearchType::BORON) {
            search_current_x = (search_boron_ppm > 0.0) ? search_boron_ppm : std::max(0.0, current_bppm);
        } else {
            search_seeded_from_previous_step = memory.has_rod_secant;
            search_current_x                 = std::clamp(memory.has_rod_secant ? memory.rod_secant_x : 1.0,
                                          0.0, rod_max_step);
        }
        // WP9-D: the point the statepoint STARTED from.  Warm start (WP10.2)
        // and any future bracket seeding both move exactly this number, so an
        // A/B has to be able to read it rather than infer it from the first
        // committed trial -- which is already one proposal downstream.
        search_first_x = search_current_x;
    }

    // Drop everything learned about k_eff(x) while keeping the current trial point and the
    // carried secant slope.  Used when a schedule step re-enters the search after the material
    // state moved (Xe / T-H / depletion): the old trial residuals were measured on a different
    // problem, so a bracket or best-point inherited from them is not just stale but actively
    // misleading -- a bracket that survived a restart could pin the search onto a rod position
    // it had already left (CY02 step 10: bracket collapsed to [1.808854, 1.808854], every
    // proposal bisected back onto it, 40+ wasted trials ending 14.8 pcm off critical).
    void ResetSearchTrials() {
        search_has_prev    = false;
        search_has_bracket = false;
        search_has_best    = false;
        search_iteration   = 0;
        search_best_x        = 0.0;
        search_best_residual = 0.0;
    }

    void ResetSearchExitStatus() {
        search_exit_status = static_cast<int>(SearchExit::NONE);
        search_exit_dk     = 0.0;
        search_exit_tol    = 0.0;
        search_stall_count = 0;
    }

    [[nodiscard]] double searchResidual(double eigv) const {
        return eigv - target_keff;
    }

    void UpdateBestSearchPoint(double k_residual) {
        if (!hasCriticalSearch())
            return;
        if (!search_has_best || std::abs(k_residual) < std::abs(search_best_residual)) {
            search_has_best      = true;
            search_best_x        = search_current_x;
            search_best_residual = k_residual;
        }
    }

    [[nodiscard]] double criticalSearchTolerance() const {
        return (searchType == SearchType::RODCRIT)
                   ? std::max(std::min(tolerance_search, kRodCritSearchTol), rodcrit_search_floor)
                   : tolerance_search;
    }

    void UpdateRodBracket(double k_residual) {
        if (searchType != SearchType::RODCRIT || !search_has_prev)
            return;

        const double prev_residual = search_prev_eigv - target_keff;
        if (!search_has_bracket && prev_residual * k_residual < 0.0) {
            if (search_prev_x < search_current_x) {
                search_bracket_lo_x        = search_prev_x;
                search_bracket_lo_residual = prev_residual;
                search_bracket_hi_x        = search_current_x;
                search_bracket_hi_residual = k_residual;
            } else {
                search_bracket_lo_x        = search_current_x;
                search_bracket_lo_residual = k_residual;
                search_bracket_hi_x        = search_prev_x;
                search_bracket_hi_residual = prev_residual;
            }
            search_has_bracket = true;
        }

        if (!search_has_bracket)
            return;
        if (k_residual * search_bracket_lo_residual > 0.0) {
            search_bracket_lo_x        = search_current_x;
            search_bracket_lo_residual = k_residual;
        } else if (k_residual * search_bracket_hi_residual > 0.0) {
            search_bracket_hi_x        = search_current_x;
            search_bracket_hi_residual = k_residual;
        }
        if (search_bracket_lo_x > search_bracket_hi_x) {
            std::swap(search_bracket_lo_x, search_bracket_hi_x);
            std::swap(search_bracket_lo_residual, search_bracket_hi_residual);
        }

        // Restore the defining invariant: a usable bracket has positive span and endpoint
        // residuals of opposite sign.  Endpoint replacement above is driven by residuals
        // re-measured at the same rod position across T/H sub-iterations, so a sign flip
        // inside the k_eff noise band can drag both endpoints onto the same point.  Such a
        // bracket carries no information and would trap bisection there forever, so drop it
        // and let the search re-establish one.
        if (search_bracket_hi_x - search_bracket_lo_x <= 0.0 ||
            search_bracket_lo_residual * search_bracket_hi_residual >= 0.0)
            search_has_bracket = false;
    }

    [[nodiscard]] bool   hasRodBracket() const { return search_has_bracket; }
    [[nodiscard]] double rodBracketSpan() const {
        return search_has_bracket ? (search_bracket_hi_x - search_bracket_lo_x) : 0.0;
    }

    [[nodiscard]] bool rodBracketNarrowEnough(double k_residual, double keff_tolerance) const {
        if (searchType != SearchType::RODCRIT || !search_has_bracket)
            return false;
        const double bracket_span = search_bracket_hi_x - search_bracket_lo_x;
        return bracket_span <= std::max(tolerance_rodsearch, 1.0e-10) &&
               std::abs(k_residual) <= keff_tolerance;
    }

    // Differing inputs between the RODCRIT and BORON secant searches.
    struct SecantSearchParams {
        bool*       has_secant        = nullptr; // memory flag written on each secant update
        double*     secant_dkdx       = nullptr; // memory slope written on each secant update
        bool        carry_available   = false;   // a carried-over slope may seed the first step
        double      probe             = 0.0;     // signed probe magnitude when bootstrapping
        const char* probe_method      = "";      // method label used for the probe step
        bool        clamp_carry       = false;   // restrict carried secant to current_x +-1
        bool        enforce_rod_clamp = false;   // clamp to [0, rod_max_step] and detect a stuck point
        double      rod_max_step      = 0.0;
        bool        use_bracket       = false;   // fall back to bisection inside a known sign-change bracket
    };

    // Shared secant / carry-secant / probe / bracket logic for both critical searches.
    bool AdvanceSecantSearch(double eigv, double k_residual,
                             const SecantSearchParams& params, double& next_x,
                             std::string& method, bool& rod_bracket_not_found) {
        if (!search_has_prev) {
            if (params.carry_available && std::abs(*params.secant_dkdx) >= min_secant_denom) {
                next_x = search_current_x - search_relaxation * k_residual / *params.secant_dkdx;
                if (params.clamp_carry)
                    next_x = std::clamp(next_x, search_current_x - 1.0, search_current_x + 1.0);
                method = "carry-secant";
            } else {
                next_x = search_current_x +
                         ((k_residual > 0.0) ? params.probe : -params.probe);
                method = params.probe_method;
            }
            if (params.enforce_rod_clamp) {
                next_x = std::clamp(next_x, 0.0, params.rod_max_step);
                if (std::abs(next_x - search_current_x) < 1.0e-10) {
                    rod_bracket_not_found = true;
                    return false;
                }
            }
        } else {
            const double dx = search_current_x - search_prev_x;
            const double dk = eigv - search_prev_eigv;

            bool secant_ok = std::abs(dx) >= min_secant_denom && std::abs(dk) >= min_secant_denom;
            if (secant_ok) {
                *params.secant_dkdx = dk / dx;
                *params.has_secant  = std::isfinite(*params.secant_dkdx);
                next_x              = search_current_x - search_relaxation * k_residual * dx / dk;
                method              = "secant";
                secant_ok           = std::isfinite(next_x);
            }

            // Secant/bisection hybrid.  UpdateRodBracket() maintains a sign-change bracket
            // [lo, hi] that used to be recorded and then never consulted: the proposal was
            // always the raw secant.  Two consecutive trials separated by less than the
            // rod-cusping k_eff noise give a badly conditioned slope, so the secant regularly
            // leaves the bracket (CY02 step 1: bracket [2.1034, 2.1845], proposal 2.2159) or
            // fails outright, and the caller then treated that as "cannot bracket" and
            // accepted whatever point it happened to be standing on.  Once the root is known
            // to be bracketed, reject any proposal that leaves the interval and take the
            // guaranteed-progress midpoint instead.
            if (params.use_bracket && search_has_bracket) {
                const double lo = search_bracket_lo_x;
                const double hi = search_bracket_hi_x;
                if (!secant_ok || next_x < lo || next_x > hi) {
                    const double mid = 0.5 * (lo + hi);
                    // Only bisect while the midpoint is a genuinely new point.  Once the
                    // bracket has narrowed to the rod-position resolution the remaining
                    // k_eff spread is cusping noise, not a resolvable root, and bisecting
                    // further just re-proposes the point we are standing on.
                    if (hi - lo > bracket_min_span &&
                        std::abs(mid - search_current_x) > bracket_min_span) {
                        method    = secant_ok ? "bisection(secant-left-bracket)"
                                              : "bisection(secant-failed)";
                        next_x    = mid;
                        secant_ok = true;
                    } else {
                        search_has_bracket = false; // exhausted: fall back to the raw secant
                    }
                }
            }
            if (!secant_ok)
                return false;
        }
        if (params.enforce_rod_clamp)
            next_x = std::clamp(next_x, 0.0, params.rod_max_step);
        else
            next_x = std::max(0.0, next_x);
        return true;
    }

    bool ProposeNextSearchPoint(double eigv, SearchMemory& memory,
                                double rod_max_step, double& next_x,
                                std::string& method, bool& rod_bracket_not_found) {
        rod_bracket_not_found   = false;
        const double k_residual = searchResidual(eigv);

        SecantSearchParams params;
        if (searchType == SearchType::RODCRIT) {
            params.has_secant        = &memory.has_rod_secant;
            params.secant_dkdx       = &memory.rod_secant_dkdx;
            params.carry_available   = search_seeded_from_previous_step;
            params.probe             = std::max(search_rod_probe, 1.0e-6);
            params.probe_method      = "probe";
            params.clamp_carry       = true;
            params.enforce_rod_clamp = true;
            params.rod_max_step      = rod_max_step;
            params.use_bracket       = true;
        } else {
            params.has_secant      = &memory.has_boron_secant;
            params.secant_dkdx     = &memory.boron_secant_dkdx;
            params.carry_available = memory.has_boron_secant;
            params.probe           = search_boron_probe;
            params.probe_method    = "bootstrap";
        }
        // ONE call site, and the tally below is why the two arms were folded
        // into one: a classification that had to be repeated per arm is a
        // classification that can disagree with itself.
        const bool proposed = AdvanceSecantSearch(eigv, k_residual, params, next_x, method,
                                                  rod_bracket_not_found);
        TallyProposal(proposed, method);
        return proposed;
    }

    /// WP9-D telemetry.  Reads `method` and writes counters nothing else reads.
    /// `method` is the string the proposal ALREADY built for the trace, so this
    /// invents no second name for what happened.
    void TallyProposal(bool proposed, const std::string& method) {
        ++search_n_proposals;
        if (!proposed) {
            ++search_n_refused;
            return;
        }
        if (method == "secant")             ++search_n_secant;
        else if (method == "carry-secant")  ++search_n_carry;
        else if (method.rfind("bisection", 0) == 0) ++search_n_bisect;
        else                                ++search_n_probe;
    }

    void CommitSearchPoint(double eigv, double next_x, SearchMemory& memory) {
        // WP9-D: the size of the step being taken.  At the end of the search
        // this is the LAST one, i.e. how far the secant still had to move when
        // it stopped -- the |dppm| the plan's trial-reduction options are judged
        // against, and a number that says whether the tolerance or the noise
        // floor ended the search.
        search_last_dx   = std::abs(next_x - search_current_x);
        search_prev_x    = search_current_x;
        search_prev_eigv = eigv;
        search_has_prev  = true;
        search_current_x = next_x;
        if (searchType == SearchType::RODCRIT)
            memory.rod_secant_x = next_x;
        ++search_iteration;
    }
};

class Scheduler {
private:
    Schedule              default_schedule;
    std::vector<Schedule> _schedule;

public:
    Scheduler() = default;

    std::vector<Schedule>&       schedule() { return _schedule; }
    const std::vector<Schedule>& schedule() const { return _schedule; }

    Schedule&       schedule(const size_t step) { return _schedule[step]; }
    const Schedule& schedule(const size_t step) const { return _schedule[step]; }

    void SetDefaultPrintOpt(PrintOpt print_opt) {
        default_schedule.print_opt = print_opt;
    }

    void SetDefaultCondition(const double bppm = 0.0, const double tful = 0.0, const double tmod = 0.0,
                             const double power = 0.0, const double rate = 0.0) {
        if (bppm >= 0.0) default_schedule.bppm = bppm;
        if (tful > 0.0) default_schedule.tful = tful;
        if (tmod > 0.0) default_schedule.tmod = tmod;
        if (power > 0.0) default_schedule.rated_power = power;
        if (rate > 0.0) default_schedule.rate = rate;
        if (power * rate > 0.0) default_schedule.actual_power = power * rate;
    }

    void SetDefaultTH(const double pressure = 0.0, const double inlet_temp = 0.0,
                      const double outlet_temp = 0.0, const double flow_rate = 0.0,
                      const bool use_flow_rate = false,
                      const double fuel_temp_rise_scale = 1.0) {
        if (pressure > 0.0) default_schedule.pressure = pressure;
        if (inlet_temp > 0.0) default_schedule.inlet_temp = inlet_temp;
        if (outlet_temp > 0.0) default_schedule.outlet_temp = outlet_temp;
        if (flow_rate > 0.0) default_schedule.mass_flow_rate = flow_rate;
        default_schedule.use_mass_flow_rate = use_flow_rate && default_schedule.mass_flow_rate > 0.0;
        if (fuel_temp_rise_scale > 0.0)
            default_schedule.fuel_temp_rise_scale = fuel_temp_rise_scale;
    }

    void SetDefaultTHProfile(const std::vector<double>& power,
                             const std::vector<double>& inlet,
                             const std::vector<double>& outlet) {
        default_schedule.th_profile_power  = power;
        default_schedule.th_inlet_profile  = inlet;
        default_schedule.th_outlet_profile = outlet;
    }

    /// @brief Set default Xe-135 depletion treatment.
    /// @param transient True for transient iodine/xenon, false for equilibrium Xe.
    void SetDefaultXenonTransient(bool transient) {
        default_schedule.xenon_transient = transient;
    }

    /// @brief Set default convergence criteria exposed in the JSON input.
    /// Each argument is applied only when > 0, so callers may override a subset.
    void SetDefaultConvergence(const int max_outer = 0, const double tol_keff = 0.0) {
        if (max_outer > 0) default_schedule.max_outer_iter = max_outer;
        if (tol_keff > 0.0) default_schedule.tolerance_keff = tol_keff;
    }

    /// @brief Add basic depletion schedule
    /// @param time Depletion time in days
    /// @param rate Rated power (%)
    /// @param substeps The number of depletion substeps
    /// @param type Search type (eigenvalue, boron, temperature, rod)
    void AddDepletionSchedule(const double time, const double rate, int substeps = 1,
                              SearchType   type   = SearchType::KEFF,
                              const double burnup = 0.0, const bool use_burnup_time = false) {
        Schedule temp_schedule        = default_schedule;
        temp_schedule.type            = ScheduleType::DEPLETION;
        temp_schedule.searchType      = type;
        temp_schedule.time            = time;
        temp_schedule.burnup          = burnup;
        temp_schedule.rate            = rate;
        temp_schedule.actual_power    = temp_schedule.rated_power * rate;
        temp_schedule.substep         = substeps;
        temp_schedule.use_burnup_time = use_burnup_time;
        _schedule.push_back(temp_schedule);
    }

    /// @brief Add a derivative (perturbation + re-solve) schedule step.
    /// Only non-zero deltas / non-empty rod_insertions override the defaults,
    /// matching the SetDefault*-style guards.
    void AddDerivativeSchedule(const double delta_tful = 0.0, const double delta_tmod = 0.0,
                               const double delta_dmod = 0.0, const double delta_bppm = 0.0,
                               const double delta_xe = 0.0, const double delta_sm = 0.0,
                               SearchType                           type           = SearchType::KEFF,
                               const std::map<std::string, double>& rod_insertions = {}) {
        Schedule temp_schedule   = default_schedule;
        temp_schedule.type       = ScheduleType::DERIVATIVE;
        temp_schedule.searchType = type;
        if (delta_tful != 0.0) temp_schedule.delta_tful = delta_tful;
        if (delta_tmod != 0.0) temp_schedule.delta_tmod = delta_tmod;
        if (delta_dmod != 0.0) temp_schedule.delta_dmod = delta_dmod;
        if (delta_bppm != 0.0) temp_schedule.delta_bppm = delta_bppm;
        if (delta_xe != 0.0) temp_schedule.delta_xe = delta_xe;
        if (delta_sm != 0.0) temp_schedule.delta_sm = delta_sm;
        if (!rod_insertions.empty()) temp_schedule.rod_insertions = rod_insertions;
        _schedule.push_back(temp_schedule);
    }
};
} // namespace rasbery
