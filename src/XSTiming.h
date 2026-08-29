#pragma once

// Phase wall for the XS-reconstruction block, behind RASBERY_XS_TIMING.
//
// The M1 receipt attributes 37.78% of wall to "XS flat" and the depletion
// probe (docs/DEPLETION_GPU_PROBE_20260821_KO.md, R8) proved the neighbouring
// "depletion 39.34%" bucket was a mislabel: the profiler shares buckets across
// functions (xs_update is UpdateTH+UpdateFlatXS, XSSet.cpp), so no number in
// that table can be trusted as a per-function attribution.  A GPU port of the
// wrong 37.78% would repeat the depletion probe's mistake, so these buckets
// split UpdateEquilibriumXenon / UpdateFlatXS / UpdateTH / SetBoron /
// UpdateBurnup apart, and split the equilibrium-Xe node loop into its
// condense and reconstruct halves -- the two halves a fused GPU kernel would
// merge, hence the two numbers any kernel A/B must be attributed against.
//
// Same carrying contract as DepletionTiming.h on the depletion probe branch:
// the timing lives in the shipped binary so both arms of an A/B are one
// SHA-256; with the variable unset, each timed region costs one relaxed load
// of a function-local static, no clock is read, nothing is allocated, and no
// arithmetic in the solver changes.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>

namespace rasbery::xsphase {

/// One wall-clock bucket, summed across every Driver thread in the process.
struct Tally {
    std::atomic<double>        seconds{0.0};
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nodes{0};

    void add(double s, std::uint64_t n) {
        seconds.fetch_add(s, std::memory_order_relaxed);
        calls.fetch_add(1, std::memory_order_relaxed);
        nodes.fetch_add(n, std::memory_order_relaxed);
    }
};

/// Process-wide buckets.  The eqxe pair decomposes the equilibrium-Xe node
/// loop; the flatxs pair decomposes UpdateFlatXS by its rod branch, which is
/// the divergence boundary a GPU port has to split kernels along.  The
/// remaining whole-call buckets separate the callers the old profiler merged.
struct Tallies {
    Tally eqxe;            ///< UpdateEquilibriumXenon, whole call
    Tally eqxe_condense;   ///< per fuel node: micro-XS condense + Xe equilibrium + _iden write
    Tally eqxe_recon;      ///< per fuel node: ReconstructNode from the Xe loop
    Tally flatxs;          ///< UpdateFlatXS, whole call
    Tally flatxs_rodded;   ///< per node: FillRodNodeXS + spectral history + ReconstructNode
    Tally flatxs_unrodded; ///< per node: UpdateUnroddedNodeXS
    Tally update_th;       ///< UpdateTH, whole call (includes its UpdateFlatXS)
    Tally set_boron;       ///< SetBoron, whole call (includes its UpdateFlatXS)
    Tally update_burnup;   ///< UpdateBurnup, whole call (includes its UpdateFlatXS)
};

inline Tallies& tallies() {
    static Tallies t;
    return t;
}

inline bool timingEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_XS_TIMING");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
                 s == "false" || s == "FALSE");
    }();
    return on;
}

// ---------------------------------------------------------------------------
// WP9 stage A: the same buckets, but PER STATEPOINT and PER DRIVER THREAD.
// ---------------------------------------------------------------------------
//
// The tallies above are process-wide sums printed once at exit, which answers
// "how much UpdateFlatXS did this run do" and cannot answer "how much of THIS
// statepoint's host floor was UpdateFlatXS" -- the question the statepoint
// receipt (Driver.h, sptelem) exists for, and the one WP9-A has to answer
// before anything is ported.  So each bucket that the statepoint receipt wants
// gets a thread-local mirror the Driver can difference across a statepoint
// boundary, exactly the way it already differences its own phase wall.
//
// WHY IT DOES NOT READ AN ENVIRONMENT VARIABLE.  RASBERY_STATEPOINT_TELEMETRY
// has exactly ONE reader in this code base (sptelem::enabled(), asserted by
// tools/test_statepoint_telemetry.py), and a second cached read here would be a
// second interpretation of the same knob that could drift from the solver's.
// Driver::Drive() therefore ARMS this mirror from that one gate, before any
// solve runs.  Unarmed -- the default, and every run of every other binary that
// links XSSet -- a Scope pays one relaxed atomic load more than it did and
// reads no clock.
//
// WHY THE MIRROR IS NOT SUMMED INTO THE FLOOR.  UpdateFlatXS runs INSIDE
// SetBoron / UpdateTH / the depletion steps, i.e. inside buckets the statepoint
// receipt already charges in full.  It is published as `nested_wall`, which is
// explicitly NOT additive with `phase_wall` / `loop_wall` / `floor_wall`; a
// consumer that added it would double-count.
enum LocalBucket : int {
    LB_FLATXS = 0, ///< XSSet::UpdateFlatXS, whole call, wherever it is called from
    LB_COUNT
};

struct LocalWall {
    double    seconds[LB_COUNT]{};
    long long calls[LB_COUNT]{};
};

/// One Driver owns one host thread (batch mode gives every deck its own), so
/// this is per-Driver by construction and needs no atomics.
inline LocalWall& localWall() {
    static thread_local LocalWall wall;
    return wall;
}

inline std::atomic<bool>& localWallArmed() {
    static std::atomic<bool> armed{false};
    return armed;
}

/// Called once, by the single reader of RASBERY_STATEPOINT_TELEMETRY.
inline void armLocalWall(bool on) {
    localWallArmed().store(on, std::memory_order_relaxed);
}

inline bool localWallOn() { return localWallArmed().load(std::memory_order_relaxed); }

/// Scoped wall accumulator.  Inert, and free of any clock read, when disabled.
/// `local` names a LocalBucket for the per-statepoint mirror, or -1 (the
/// default, and every pre-existing call site) for none.
class Scope {
public:
    Scope(Tally& bucket, std::uint64_t nodes, int local = -1)
        : _bucket(timingEnabled() ? &bucket : nullptr),
          _local(localWallOn() ? local : -1),
          _nodes(nodes) {
        if (_bucket != nullptr || _local >= 0) _start = std::chrono::steady_clock::now();
    }
    ~Scope() { stop(); }

    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

    void stop() {
        if (_bucket == nullptr && _local < 0) return;
        const auto   elapsed = std::chrono::steady_clock::now() - _start;
        const double seconds = std::chrono::duration<double>(elapsed).count();
        if (_bucket != nullptr) _bucket->add(seconds, _nodes);
        if (_local >= 0) {
            LocalWall& mirror = localWall();
            mirror.seconds[_local] += seconds;
            ++mirror.calls[_local];
        }
        _bucket = nullptr;
        _local  = -1;
    }

private:
    Tally*                                _bucket;
    int                                   _local;
    std::uint64_t                         _nodes;
    std::chrono::steady_clock::time_point _start{};
};

inline void emit(std::ostream& out, const char* name, const Tally& t, bool last) {
    out << '"' << name << "\":{\"seconds\":"
        << t.seconds.load(std::memory_order_relaxed)
        << ",\"calls\":" << t.calls.load(std::memory_order_relaxed)
        << ",\"nodes\":" << t.nodes.load(std::memory_order_relaxed) << '}';
    if (!last) out << ',';
}

/// One JSON line at end of run.  Silent when the variable is unset.
inline void report(std::ostream& out) {
    if (!timingEnabled()) return;
    const Tallies& t = tallies();
    out << "[RASBERY][XS][PHASE] {\"buckets\":{";
    emit(out, "eqxe", t.eqxe, false);
    emit(out, "eqxe_condense", t.eqxe_condense, false);
    emit(out, "eqxe_recon", t.eqxe_recon, false);
    emit(out, "flatxs", t.flatxs, false);
    emit(out, "flatxs_rodded", t.flatxs_rodded, false);
    emit(out, "flatxs_unrodded", t.flatxs_unrodded, false);
    emit(out, "update_th", t.update_th, false);
    emit(out, "set_boron", t.set_boron, false);
    emit(out, "update_burnup", t.update_burnup, true);
    out << "}}" << std::endl;
}

} // namespace rasbery::xsphase
