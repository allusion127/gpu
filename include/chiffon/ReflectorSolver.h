#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

#include "Model.h"

namespace Chiffon {

// Surface indices of a node, in the DeCART %ADFT column order. Upstream v3.0.0 dropped
// these from Model.h together with the reflector DF path; our WF4 work kept that path
// (RASBERY_REFL_DF), so they live here with their only consumer.
constexpr int SOUTH = 0;
constexpr int EAST  = 1;
constexpr int NORTH = 2;
constexpr int WEST  = 3;

// Boundary condition types for reflector interface coupling
enum BOUNDARY {
    REFLECTIVE,
    VACUUM,
    FLUXZERO,
    NODE
};

// Applies assembly discontinuity factors (ADF) to fuel and reflector cross-sections.
// ADF corrects for the flux discontinuity at node interfaces in nodal diffusion.
//
// Reflector-side DF mode (RASBERY_REFL_DF):
//   "ratio"           - reflector XS is divided by f_refl / f_fuel taken at the shared
//                       interface. This is the classical single-DF-per-interface form: the
//                       fuel node has already absorbed its own f through ApplyDF(Model&),
//                       so the reflector carries the *relative* discontinuity. This is the
//                       treatment that was present but commented out in the imported
//                       baseline (see NOTE below).
//   "self"            - reflector XS is divided by its own interface f only (f_refl).
//   "off"             - legacy behaviour: no DF on reflector nodes at all.
//
// NOTE on the comment history: the single upstream import commit ("Import Rasberry CPU
// baseline") already contains these lines commented out, so no VCS record of why they were
// disabled survives in any of the local trees. The two failure modes the guards below cover
// are the only ones reachable from the shipped decks: (a) reflector models can hold fewer
// depletion points than the paired fuel model, which made the original `fuel.NumPoints()`
// loop read out of range, and (b) DeCART writes the *outer* (vacuum-facing) surface DF of
// an edge reflector node as a garbage value (e.g. 3.13e+01 in dec_iSMR_axial_0105.HGC,
// -3.94e+00 in dec_iSMR_axial_1005.HGC) because the surface flux/current there is ~0; a
// mis-indexed side therefore produced negative or wildly scaled reflector XS and destroyed
// the outer-iteration convergence. Both are now checked explicitly instead of by disabling
// the physics.
class ReflectorSolver {
public:
    enum class ReflDFMode { OFF, RATIO, SELF };

    ReflectorSolver() : _mode(ResolveMode()) {
    }

    ReflDFMode Mode() const { return _mode; }

    // Apply ADF from surface DF data to all depletion points in a fuel model
    void ApplyDF(Model& fuel) {
        for (int i = 0; i < fuel.NumPoints(); i++) {
            DepletionPoint& fuelNode = fuel(i);
            fuelNode._xs.ApplyDF(fuelNode._sdfa[0], 0);
            fuelNode._xs.ApplyDF(fuelNode._sdfa[4], 1);
            fuelNode._xs.CalcLmpx(fuelNode._iden);
        }
    }

    // Apply ADF ratio (reflector/fuel) to a reflector model whose fuel neighbour lies to its
    // WEST, i.e. the reflector's WEST face and the fuel's EAST face are the shared interface.
    // Used for radial reflectors and for the axial *top* reflector.
    void ApplyDF(const Model& fuel, Model& refl, Model* lNeighbor, Model* rNeighbor, BOUNDARY lBoundaryCond,
                 BOUNDARY rBoundaryCond) {
        (void)lNeighbor;
        (void)rNeighbor;
        (void)lBoundaryCond;
        (void)rBoundaryCond;
        ApplyReflectorDF(fuel, refl, WEST, EAST);
    }

    // Apply ADF ratio for the axial bottom reflector: the fuel neighbour lies to the reflector's
    // EAST, so the reflector's EAST face pairs with the fuel's WEST face (reversed interface).
    void ApplyDF2(Model& refl, Model& fuel, Model* lNeighbor, Model* rNeighbor, BOUNDARY lBoundaryCond,
                  BOUNDARY rBoundaryCond) {
        (void)lNeighbor;
        (void)rNeighbor;
        (void)lBoundaryCond;
        (void)rBoundaryCond;
        ApplyReflectorDF(fuel, refl, EAST, WEST);
    }

    // Apply the reflector node's own interface DF (Sigma/f, D/f) on a named surface. Used by the
    // "averaged" reflector path, which has no paired fuel model to form a ratio with: those
    // reflector nodes carry no DF at all otherwise. `side` is a Direction:: index; the fuel-facing
    // surface must be declared in the CHIFFON input ("fuel side"), because a DeCART 1D colorset
    // node's orientation is not recoverable from the HGC file.
    void ApplySelfDF(Model& refl, int side) {
        // Report before the OFF early-return, so a mis-declared "fuel side"
        // shows up in the default (DF-off) build too -- that is where the
        // declaration is written and where the mistake is cheap to fix.
        ReportDeclaredSide(refl, side);
        if (_mode == ReflDFMode::OFF) return;
        DFTally tally;
        for (int i = 0; i < refl.NumPoints(); i++) {
            DepletionPoint& reflNode = refl(i);
            for (int ig = 0; ig < reflNode._ngrp; ig++) {
                const double fr = reflNode._sdfa[ig * 4 + side];
                if (!Usable(fr)) { ++tally.skipped; continue; }
                ++tally.applied;
                reflNode._xs.ApplyDF(fr, ig);
            }
            reflNode._xs.CalcLmpx(reflNode._iden);
        }
        ReportTally(refl.name(), "self", SideName(side), tally);
    }

    // Map a "fuel side" string from the CHIFFON input to a Direction:: index; -1 when absent.
    static int ParseSide(const std::string& name) {
        std::string v(name);
        for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (v == "south") return SOUTH;
        if (v == "east") return EAST;
        if (v == "north") return NORTH;
        if (v == "west") return WEST;
        return -1;
    }

private:
    ReflDFMode _mode;

    // Default is OFF, not RATIO.  Applying the reflector-side DF is the physically
    // correct treatment and it is a large improvement on the iSMR_HIGA benchmark
    // (3D pin RMS 6.83 -> 4.70 %, reflector-adjacent pin RMS 10.06 -> 7.82 %, axial
    // end-plane error 14.5 -> 1.1 %), but on the built-in i-SMR validation set it moves
    // CY01 BOC from -77 to -190 pcm against MASTER, i.e. it makes that deck worse.  The
    // two decks are biased in opposite directions against MASTER and the origin of the
    // built-in set's -77 pcm is not yet identified, so enabling this by default would
    // silently change every already-validated result.  Opt in per run instead:
    //     RASBERY_REFL_DF=ratio   f_refl / f_fuel at the shared interface (recommended)
    //     RASBERY_REFL_DF=self    f_refl only (what "averaged" radial reflectors need)
    // With no environment variable set the generated library is bit-identical to the
    // merged-validated baseline (verified: 0/2114 datasets differ).
    static ReflDFMode ResolveMode() {
        const char* env = std::getenv("RASBERY_REFL_DF");
        if (env == nullptr) return ReflDFMode::OFF;
        std::string v(env);
        for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (v == "off" || v == "none" || v == "0") return ReflDFMode::OFF;
        if (v == "self" || v == "own") return ReflDFMode::SELF;
        if (v == "ratio") return ReflDFMode::RATIO;
        return ReflDFMode::OFF;
    }

    // Reasonable-DF window. DeCART emits ~0 / negative / O(10) values on surfaces whose net
    // current vanishes (outer vacuum faces of edge reflector nodes); those must never be used.
    static constexpr double kDFLow  = 0.05;
    static constexpr double kDFHigh = 20.0;

    static bool Usable(double df) {
        return std::isfinite(df) && df > kDFLow && df < kDFHigh;
    }

    // reflSide : surface index on the reflector node that touches the fuel
    // fuelSide : surface index on the fuel node that touches the reflector
    void ApplyReflectorDF(const Model& fuel, Model& refl, int reflSide, int fuelSide) {
        ReportDeclaredSide(refl, reflSide);
        // Guard (a): a reflector model may hold fewer depletion/branch points than the fuel model
        // it is paired with. The original loop ran to fuel.NumPoints() and indexed refl(i).
        const int npt = std::min(fuel.NumPoints(), refl.NumPoints());

        DFTally tally;
        for (int i = 0; i < npt; i++) {
            DepletionPoint& reflNode = refl(i);
            if (_mode != ReflDFMode::OFF) {
                const DepletionPoint& fuelNode = fuel(i);
                for (int ig = 0; ig < reflNode._ngrp; ig++) {
                    const double fr = reflNode._sdfa[ig * 4 + reflSide];
                    const double ff = fuelNode._sdfa[ig * 4 + fuelSide];
                    if (!Usable(fr)) { ++tally.skipped_refl; continue; }  // guard (b), reflector side
                    double df = fr;
                    if (_mode == ReflDFMode::RATIO) {
                        if (!Usable(ff)) { ++tally.skipped_fuel; continue; }  // guard (b), fuel side
                        df = fr / ff;
                    }
                    if (!Usable(df)) { ++tally.skipped_ratio; continue; }
                    ++tally.applied;
                    reflNode._xs.ApplyDF(df, ig);
                }
            }
            reflNode._xs.CalcLmpx(reflNode._iden);
        }
        // Points beyond the shared range keep the legacy behaviour (lumped XS rebuilt, no DF).
        for (int i = npt; i < refl.NumPoints(); i++)
            refl(i)._xs.CalcLmpx(refl(i)._iden);
        // With the DF path off nothing was filtered and nothing applied, so the
        // counters would read "applied 0, skipped 0" and say nothing at all.
        // The declared-side line above is the diagnostic that matters there.
        if (_mode != ReflDFMode::OFF) {
            tally.unpaired_points = refl.NumPoints() - npt;
            ReportTally(refl.name(), _mode == ReflDFMode::RATIO ? "ratio" : "self",
                        SideName(reflSide), tally);
        }
    }

    // -----------------------------------------------------------------------
    // Instrumentation.
    //
    // Usable() is a silent filter: it drops a surface DF and the node simply
    // keeps its undivided XS.  That is the right behaviour -- DeCART writes
    // garbage on vacuum-facing surfaces -- but it is indistinguishable from the
    // case the operator actually cares about, which is a *mis-declared* fuel
    // side.  Point "fuel side" at the wrong surface and every DF on it is
    // rejected: the run is clean, the library builds, and the reflector quietly
    // has no DF at all.  The counters below make the two cases different on the
    // console, and the one-shot per-(model, side) line prints the DF values the
    // declaration actually selected, so a wrong surface is visible as a set of
    // out-of-window numbers next to a plausible-looking one.
    // -----------------------------------------------------------------------
    struct DFTally {
        long applied         = 0;
        long skipped         = 0;   // ApplySelfDF: reflector-side reject
        long skipped_refl    = 0;
        long skipped_fuel    = 0;
        long skipped_ratio   = 0;
        long unpaired_points = 0;
    };

    static const char* SideName(int side) {
        switch (side) {
            case SOUTH: return "south";
            case EAST:  return "east";
            case NORTH: return "north";
            case WEST:  return "west";
            default:    return "?";
        }
    }

    // One line per (model, side) for the whole run: the group-wise DF the
    // declared surface carries at the first depletion point, and whether the
    // Usable() window accepts it.
    static void ReportDeclaredSide(const Model& refl, int side) {
        static std::set<std::string> seen;
        const std::string key = refl.name() + "/" + SideName(side);
        if (!seen.insert(key).second) return;
        if (refl.NumPoints() <= 0) {
            std::cerr << "[CHIFFON][refl] model '" << refl.name() << "' side "
                      << SideName(side) << ": no depletion points" << std::endl;
            return;
        }
        const DepletionPoint& p = refl(static_cast<size_t>(0));
        std::cerr << "[CHIFFON][refl] model '" << refl.name() << "' fuel-facing side "
                  << SideName(side) << " DF(point 0) =";
        int usable = 0;
        for (int ig = 0; ig < p._ngrp; ig++) {
            const std::size_t k = static_cast<std::size_t>(ig * 4 + side);
            const double      v = k < p._sdfa.size() ? p._sdfa[k] : 0.0;
            std::cerr << ' ' << v;
            if (Usable(v)) ++usable;
        }
        std::cerr << "  (" << usable << '/' << p._ngrp
                  << " inside the usable window " << kDFLow << ".." << kDFHigh << ')'
                  << std::endl;
    }

    static void ReportTally(const std::string& name, const char* mode, const char* side,
                            const DFTally& t) {
        std::cerr << "[CHIFFON][refl] model '" << name << "' side " << side << " mode "
                  << mode << ": applied " << t.applied << ", skipped "
                  << (t.skipped + t.skipped_refl + t.skipped_fuel + t.skipped_ratio)
                  << " (refl " << (t.skipped + t.skipped_refl) << ", fuel " << t.skipped_fuel
                  << ", ratio " << t.skipped_ratio << "), unpaired points "
                  << t.unpaired_points << std::endl;
    }
};
} // namespace Chiffon
