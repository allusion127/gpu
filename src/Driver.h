#pragma once
#include "BICGCMFD.h"
#include "IO.h"
#include "Nodal.h"
#include "PPR.h"
#include "Scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>

namespace rasbery {

class Driver {
private:
    std::string _input;
    std::string _result_output;

    static constexpr double CMFD_FLUX_L2_TOLERANCE = 1.0e-6;
    // Search / T-H feedbacks engage once the CMFD flux L2 residual drops below this, so they
    // act on a meaningful (not rough-early) flux while still co-converging in the single loop.
    static constexpr double FEEDBACK_TRIGGER = 1.0e-3;
    // T/H feedback convergence on the relative Doppler (fuel) temperature change (PARCS eps_Dop).
    static constexpr double TH_DOPPLER_TOLERANCE = 1.0e-2;

    struct SolverContext {
        Geometry&    geometry;
        XSSet&       cross_sections;
        BICGCMFD&    cmfd_solver;
        Nodal&       nodal_solver;
        SearchMemory search_memory;
    };

    // Single-loop eigenvalue solve following PARCS Fig 10.1. One outer iteration performs a
    // CMFD/BiCGSTAB flux update (Wielandt inside drive), then the nodal correction + CNCC
    // (d-hat) + rod cusping, then — once the flux is meaningful — one damped critical-search
    // step and one T/H feedback update. The sub-problems co-converge and are checked together
    // ("All Converged?"), instead of fully converging each in a nested inner loop.
    static void SolveLoop(SolverContext& ctx, double& eigv, Schedule& schedule,
                          int& total_outer, int& total_th, bool keepSearch = false,
                          bool /*requireFullSearchValidation*/ = true) {
        const double power_fraction = schedule.powerFraction();
        const bool   has_th         = schedule.usesTHFeedback();
        const bool   has_search     = schedule.hasCriticalSearch();
        const bool   trace_search   = (std::getenv("RASBERY_SEARCH_TRACE") != nullptr);
        const bool   trace_sl       = (std::getenv("RASBERY_SL_TRACE") != nullptr);
        const int    sl_outer0      = total_outer;
        const int    sl_th0         = total_th;

        if (has_search) {
            if (keepSearch) {
                schedule.search_iteration = 0;
                schedule.search_has_prev  = false;
            } else {
                schedule.ResetSearchState();
            }
            // First entry into this schedule step: pick an initial guess and apply it.
            if (!schedule.search_initialized) {
                schedule.StartCriticalSearch(ctx.search_memory, ctx.geometry.bppm(0),
                                             ctx.cross_sections.rod_max_step());
                if (schedule.searchType == SearchType::BORON)
                    ctx.cross_sections.SetBoron(schedule.search_current_x);
                else {
                    ctx.cross_sections.SetRod(schedule.search_current_x);
                    schedule.rod_step = schedule.search_current_x;
                }
            }
        }

        // Rod-crit search over a cusping-enabled core: the fractional fine-cell stencil smooths
        // the keff(rod) staircase but residual kinks at fine-cell boundaries leave a ~3e-5 keff
        // noise floor. A 1e-5 search tolerance sits inside that noise, so the secant bounces
        // (8-13 trials/step). Raise the floor above the noise so it converges in ~half as many.
        if (has_search && schedule.searchType == SearchType::RODCRIT)
            schedule.rodcrit_search_floor = (ctx.cross_sections.axial_rod_division() > 0) ? 5.0e-5 : 0.0;

        const double keff_tol   = schedule.tolerance_keff;
        const double search_tol = schedule.criticalSearchTolerance();
        const double flux_tol   = std::max(keff_tol, CMFD_FLUX_L2_TOLERANCE);

        ctx.cmfd_solver.resetIteration();
        ctx.cmfd_solver.upddtil();
        double residual   = 1.0;
        double prev_inner = eigv + 1.0;
        double th_dop     = 1.0; // last Doppler-temperature change from UpdateTH
        int    th_count   = 0;
        int    flux_stall = 0;
        bool   used_best  = false;

        // Hard safety bound. Each feedback (search/T-H) step is followed by a bounded flux
        // re-convergence (flux_stall guard), and the search/T-H step counts are bounded too.
        const int max_iter = 50 * std::max({schedule.max_outer_iter, schedule.max_th_iter,
                                            has_search ? schedule.max_search_iter : 0});
        for (int iout = 0; iout < max_iter; ++iout) {
            // 1. Flux: CMFD BiCGSTAB iterations + Wielandt shift.
            ctx.cmfd_solver.updpsi(ctx.geometry.Phif());
            ctx.cmfd_solver.setls(eigv);
            ctx.cmfd_solver.drive(eigv, ctx.geometry.Phif(), residual);
            ++total_outer;
            const bool flux_converged = std::abs(prev_inner - eigv) < keff_tol && residual < flux_tol;
            prev_inner                = eigv;

            // 2. Nodal correction -> CNCC (d-hat) + rod cusping macro-XS update. The cusping blend
            //    co-converges with the flux, so its settledness is implied by flux_converged.
            ctx.cmfd_solver.updjnet(ctx.geometry.Phif(), ctx.geometry.Jnet());
            ctx.nodal_solver.reset(1.0 / eigv, ctx.geometry.Jnet(),
                                   ctx.geometry.Phif(), ctx.geometry.Phis());
            ctx.nodal_solver.drive();
            if (ctx.cross_sections.ApplyRodCusping(eigv, ctx.nodal_solver.axialTransverseLeakage()))
                ctx.cmfd_solver.upddtil();
            ctx.cmfd_solver.upddhat(ctx.geometry.Phif(), ctx.geometry.Jnet());

            // Keep iterating flux + nodal/cusping until the flux is converged; the feedbacks
            // (search, T/H) are root-finds on k_eff / power and must act on a clean flux.
            if (!flux_converged) {
                if (++flux_stall > schedule.max_outer_iter)
                    break; // flux not converging: give up on this solve
                continue;
            }
            flux_stall = 0;

            // 3. Critical search ("CBC Search?").
            bool         search_converged = !has_search;
            const double k_residual       = has_search ? schedule.searchResidual(eigv) : 0.0;
            if (has_search) {
                schedule.UpdateBestSearchPoint(k_residual);
                if (schedule.searchType == SearchType::RODCRIT) {
                    schedule.rod_step = schedule.search_current_x;
                    schedule.UpdateRodBracket(k_residual);
                }
                search_converged = std::abs(k_residual) < search_tol;
            }

            // 4. T/H feedback ("Need T/H?"): converged on the Doppler (fuel) temperature change
            //    (PARCS delta_Dop), which is physical and far above the cusping k_eff noise floor.
            const bool th_converged = !has_th || (th_count > 0 && th_dop < TH_DOPPLER_TOLERANCE) ||
                                      th_count >= schedule.max_th_iter;

            // 5. All converged?
            if (search_converged && th_converged)
                break;

            // Otherwise perturb the unconverged feedbacks, then re-converge the flux.
            if (has_th && !th_converged) {
                th_dop = ctx.cross_sections.UpdateTH(power_fraction);
                ++total_th;
                ++th_count;
                if (trace_sl)
                    std::cout << std::format("        [TH] it={} th_dop={:.3e} eigv={:.6f}\n",
                                             th_count, th_dop, eigv);
            }
            if (has_search && !search_converged) {
                if (schedule.search_iteration >= schedule.max_search_iter) {
                    if (!used_best && schedule.search_has_best &&
                        std::abs(schedule.search_best_residual) < std::abs(k_residual)) {
                        schedule.search_current_x = schedule.search_best_x;
                        used_best                 = true;
                    } else {
                        break; // out of search iterations: accept current point
                    }
                } else {
                    double      next_x = schedule.search_current_x;
                    std::string method;
                    bool        bracket_not_found = false;
                    if (!schedule.ProposeNextSearchPoint(eigv, ctx.search_memory,
                                                         ctx.cross_sections.rod_max_step(),
                                                         next_x, method, bracket_not_found))
                        break; // cannot bracket: accept current point
                    schedule.CommitSearchPoint(eigv, next_x, ctx.search_memory);
                }
                if (trace_search) {
                    const char* nm = (schedule.searchType == SearchType::BORON) ? "BORON" : "ROD";
                    std::cout << std::format("        [SEARCH] {} it={} x={:.6f} k={:.8f} dk={:+.3e} outer={}\n",
                                             nm, schedule.search_iteration, schedule.search_current_x,
                                             eigv, k_residual, total_outer);
                }
                if (schedule.searchType == SearchType::BORON)
                    ctx.cross_sections.SetBoron(schedule.search_current_x);
                else {
                    ctx.cross_sections.SetRod(schedule.search_current_x);
                    schedule.rod_step = schedule.search_current_x;
                }
            }
        }

        if (has_search && schedule.searchType == SearchType::RODCRIT)
            schedule.rod_step = schedule.search_current_x;

        if (trace_sl)
            std::cout << std::format("      [SL] outer+={} th+={} (search={} relax={:.3f})\n",
                                     total_outer - sl_outer0, total_th - sl_th0,
                                     has_search ? 1 : 0, ctx.cross_sections.rod_cusping_relaxation());
    }

public:
    explicit Driver(const std::string& input, const std::string& result_output = "")
        : _input(input),
          _result_output(result_output) {
    }

    int Drive() {
        const auto driver_start = std::chrono::steady_clock::now();

        // 1. Build solver objects and read input deck
        Geometry  geometry;
        Scheduler scheduler;
        XSSet     cross_sections(geometry);

        IO input_output(geometry, cross_sections, scheduler);
        input_output.ReadInput(_input);

        BICGCMFD cmfd_solver(geometry, cross_sections);
        cmfd_solver.setNcmfd(5);
        cmfd_solver.setEpsl2(1.0e-6);
        cmfd_solver.setEshift(0.04);

        Nodal nodal_solver(geometry, cross_sections);
        PPR   pin_power_reconstruction(geometry, cross_sections);

        SolverContext ctx{geometry, cross_sections, cmfd_solver, nodal_solver, SearchMemory{}};

        // 2. Initialize run state
        const bool is_restart_run = input_output.has_restart() && !input_output.has_shuffle();
        const bool is_shuffle_run = input_output.has_shuffle();

        double eigv = 1.0;
        double efpd = 0.0;
        if (input_output.has_restart()) efpd = input_output.restart_efpd();

        auto& initial_schedule = scheduler.schedule(0);
        cross_sections.InitXS(initial_schedule.bppm,
                              initial_schedule.tful,
                              initial_schedule.tmod,
                              initial_schedule.pressure,
                              0.0, !is_restart_run);

        if (!is_restart_run)
            cross_sections.ResetFluxAndCurrents(1.0);

        if (is_restart_run)
            std::cout << std::format("  [RESTART] Continuing from '{}'\n", input_output.restart_path());
        else if (is_shuffle_run)
            std::cout << std::format("  [SHUFFLE] New cycle, {} shuffle spec(s) applied\n", input_output.restart_files().size());

        std::string result_path = _result_output;
        if (result_path.empty())
            result_path = input_output.input_dir() + "result.h5";

        const std::filesystem::path result_file_path(result_path);
        if (result_file_path.has_parent_path())
            std::filesystem::create_directories(result_file_path.parent_path());

        input_output.OpenResult(result_path);

        const double init_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - driver_start).count();
        std::cout << std::format("  [TIMING] Init+IO={:.3f} s\n", init_seconds);

        // 3. Main schedule loop
        double    total_io_seconds = 0.0;
        const int schedule_count   = static_cast<int>(scheduler.schedule().size());

        for (int step_index = 0; step_index < schedule_count; ++step_index) {
            auto& schedule = scheduler.schedule(step_index);
            const auto step_start = std::chrono::steady_clock::now();

            schedule.PrepareForStep(cross_sections.CoreHeavyMetalMassKg());
            schedule.ApplyToGeometry(geometry);

            const double power_fraction = schedule.powerFraction();
            const double thermal_power  = schedule.thermalPower();
            const double step_dt        = schedule.time * 86400.0;
            cross_sections.SetPowerRate(power_fraction);
            efpd += schedule.time * power_fraction;

            int        total_outer = 0;
            int        total_th    = 0;
            const bool keep_search = schedule.keepSearchBetweenSolves();

            // Pre-work by schedule type
            if (schedule.type == ScheduleType::DEPLETION) {
                const int    nsub   = std::max(1, schedule.substep);
                const double sub_dt = step_dt / nsub;
                for (int isub = 0; isub < nsub; ++isub) {
                    // Predictor (BOS) solve. The first substep reuses the flux carried from the
                    // previous step's final solve, which already converged this exact composition
                    // (BOS_k == EOS_{k-1}); only later substeps need a fresh BOS re-solve.
                    if (isub > 0) {
                        SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, false);
                        cross_sections.NormalizeFluxSign();
                    }
                    cross_sections.PredictorStep(sub_dt, thermal_power, schedule.xenon_transient);
                    SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search, false);
                    cross_sections.NormalizeFluxSign();
                    cross_sections.CorrectorStep(sub_dt, thermal_power, schedule.xenon_transient);
                }
            }

            if (schedule.type == ScheduleType::DERIVATIVE) {
                cross_sections.UpdateDerivative(schedule.delta_bppm,
                                                schedule.delta_tful,
                                                schedule.delta_tmod,
                                                schedule.delta_dmod);
            }

            if (schedule.type == ScheduleType::ROD) {
                cross_sections.SetRod(schedule.rod_insertions);
                cross_sections.ResetFluxAndCurrents(1.0);
                cmfd_solver.resetDhat();
                eigv = 1.0;
            }

            // Final solve
            SolveLoop(ctx, eigv, schedule, total_outer, total_th, keep_search);
            cross_sections.NormalizeFluxSign();

            // PPR
            pin_power_reconstruction.reset(1.0 / eigv, geometry.Jnet(), geometry.Phif(), geometry.Phis());
            pin_power_reconstruction.drive(5);
            pin_power_reconstruction.reconstructPinPower(false, schedule.print_opt.pin_flux);

            // Output
            const int step_number = step_index + 1;
            schedule.eigv         = eigv;
            schedule.rho          = (eigv > 1.0e-12) ? (eigv - 1.0) / eigv : 0.0;
            schedule.ppm          = geometry.bppm(0);
            const double step_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
            std::cout << std::format("  NO.={:4d}  EFPD={:10.3f}  K-EFF={:.6f}  PPM={:8.2f}  outer={:3d}  TH={:2d}  t={:5.2f}s\n",
                                     step_number, efpd, eigv, geometry.bppm(0), total_outer, total_th, step_seconds);

            const auto io_start = std::chrono::steady_clock::now();
            input_output.AddResult(geometry, eigv, step_index, step_number, efpd);

            if (schedule.print_opt.save) {
                input_output.SaveRestart(input_output.input_dir() + std::format("restart_{}.h5", step_number),
                                         geometry, cross_sections, eigv, efpd, step_number);
            }

            input_output.WriteStepToResult(geometry, cross_sections, step_index);
            total_io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
        }

        input_output.CloseResult();

        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - driver_start).count();
        std::cout << std::format("  [TIMING] IO write={:.3f} s\n", total_io_seconds);
        std::cout << std::format("  TOTAL DRIVER TIME={:10.3f} s\n", total_seconds);
        return 0;
    }
};

} // namespace rasbery
