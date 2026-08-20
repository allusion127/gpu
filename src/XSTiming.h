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

/// Scoped wall accumulator.  Inert, and free of any clock read, when disabled.
class Scope {
public:
    Scope(Tally& bucket, std::uint64_t nodes)
        : _bucket(timingEnabled() ? &bucket : nullptr), _nodes(nodes) {
        if (_bucket) _start = std::chrono::steady_clock::now();
    }
    ~Scope() { stop(); }

    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

    void stop() {
        if (!_bucket) return;
        const auto elapsed = std::chrono::steady_clock::now() - _start;
        _bucket->add(std::chrono::duration<double>(elapsed).count(), _nodes);
        _bucket = nullptr;
    }

private:
    Tally*                                _bucket;
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
