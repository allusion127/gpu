#pragma once

// WP8 stage 2 -- the cohort's immutable state, built once per process.
//
// THE THREE LIFETIMES (BOTTLENECK plan Sec 5.1) HAD A HOLE IN THE MIDDLE.
// Stage 1 made the PROCESS outlive a case and left the CASE owning everything
// else; `geometry_builds == cases` in the evaluator receipt is the machine
// readable statement that there was no middle lifetime at all.  This is the
// middle one: state that is a pure function of the geometry/topology
// declaration and the library, shared read-only by every case of that cohort,
// and destroyed only with the process.
//
// EVERY MEMBER OF `Context` IS `const`, AND THAT IS THE WHOLE SAFETY ARGUMENT.
// The rule EvaluatorContext.h states -- "PROCESS state is immutable after
// stand-up; CASE state is everything a solve writes to" -- is enforced here by
// the type rather than argued in a comment, because this object is read by up
// to M Driver threads at once with no lock between them.  A mutable member
// would be a data race that the digest would find late, if at all.  That is
// also why the quadrature is built EAGERLY in the builder instead of lazily
// behind a `mutable` cache and a `once_flag`: a lazily-filled member is a
// mutable member, and the cost of being wrong here is worse than the ~1 MB a
// cohort that never reconstructs a pin power holds for nothing.
//
// WHAT IS IN IT TODAY, AND WHY NOT MORE.  Exactly one thing: the PPR pin-power
// quadrature table (PprQuadrature.h), which reads only `ndivxy` and `npins` and
// was rebuilt bit-identically once per case.  The larger prize -- Geometry's
// neighbour, surface and assembly maps -- is NOT here, and the reason is
// specific rather than cautious: `Geometry::Initialize` allocates every array
// with bare `new[]` and has no `delete[]` prologue, so it can be run once but
// cannot be re-run, and a shared Geometry would have to be a Geometry that
// nobody re-initialises.  Its accessors also return mutable `int&`/`double&`
// (the `Phif()`/`PhifMutable()` pair is the only field that has been hardened),
// so a shared Geometry has no compile-time protection against a writer today.
// Both are real refactors with a bit-identity gate attached; claiming the
// sharing before doing them would be claiming the lever without the safety.
// `cohort_builds` / `cohort_hits` are honest about the size of what is shared:
// they count COHORTS, not the bytes saved.
//
// KEY DISCIPLINE.  A Context is looked up by `cohort::keyOf` (CohortKey.h) and
// NOTHING else.  Two cases whose geometry payloads differ by one byte are two
// cohorts.  The registry also RE-CHECKS the shape fields on every hit: if a
// case's (ng, ndivxy, npins) disagree with the cohort it keyed into, that is a
// defect in the key -- not a thing to work around -- and it throws by name
// rather than handing back state built for another core.

#include "CohortKey.h"
#include "PprQuadrature.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rasbery::cohort {

/// What a case knows about its own cohort before the cohort exists.
struct Descriptor {
    std::string geometry_digest; ///< sha256 of cohort::geometryPayload(gin)
    std::string xslib_digest;    ///< sha256 of the library CONTENT
    int         ng     = 0;
    int         ndivxy = 0;
    int         npins  = 0;

    [[nodiscard]] std::string key() const {
        return keyOf(Provenance{geometry_digest, xslib_digest, ng});
    }
};

/// The immutable per-cohort state.  Every member is const; see the header note.
struct Context {
    const std::string key;
    const std::string geometry_digest;
    const std::string xslib_digest;
    const int         ng;
    const int         ndivxy;
    const int         npins;
    /// Pure function of (ndivxy, npins).  `shared_ptr<const>` and not a value
    /// so a second cohort that differs only in its library shares the one table
    /// rather than building a second identical one.
    const std::shared_ptr<const PinQuadTable> ppr_quadrature;
};

struct Stats {
    std::uint64_t builds     = 0; ///< Contexts actually constructed
    std::uint64_t hits       = 0; ///< acquisitions served from the registry
    std::uint64_t cohorts    = 0; ///< distinct keys resident
    std::uint64_t quadrature_builds = 0; ///< PinQuadTables actually computed
    std::uint64_t quadrature_hits   = 0;
    /// WP10.4 -- THE REGISTRY IS BOUNDED NOW, AND SAYS SO.
    ///
    /// The key is (geometry payload digest, library digest, ng), and a GA
    /// candidate IS a different geometry payload: in a campaign the cohort
    /// count grows with the number of distinct candidates, i.e. forever, and
    /// the process holds every Context it ever built.  One Context is small,
    /// but "small and unbounded" is the shape of every leak that took a week to
    /// find -- and `cohorts` was already the number that was supposed to prove
    /// the middle lifetime exists, so letting it grow without limit made it
    /// prove nothing.  Bounded by `RASBERY_COHORT_CACHE_ENTRIES` (default 64),
    /// least-recently-used first, and each eviction counted.
    std::uint64_t evictions  = 0;
    std::uint64_t limit      = 0;
};

namespace detail {

inline std::mutex& registryMutex() {
    static std::mutex m;
    return m;
}

inline std::atomic<std::uint64_t>& counter(int which) {
    static std::atomic<std::uint64_t> builds{0};
    static std::atomic<std::uint64_t> hits{0};
    static std::atomic<std::uint64_t> quad_builds{0};
    static std::atomic<std::uint64_t> quad_hits{0};
    static std::atomic<std::uint64_t> evictions{0};
    switch (which) {
        case 0:  return builds;
        case 1:  return hits;
        case 2:  return quad_builds;
        case 4:  return evictions;
        default: return quad_hits;
    }
}

struct QuadEntry {
    int                                 ndivxy = 0;
    int                                 npins  = 0;
    std::shared_ptr<const PinQuadTable> table;
};

inline std::vector<QuadEntry>& quadEntries() {
    static std::vector<QuadEntry> entries;
    return entries;
}

inline std::mutex& quadMutex() {
    static std::mutex m;
    return m;
}

struct Entry {
    std::string                    key;
    std::shared_ptr<const Context> value;
    /// Monotonic stamp of the last acquire() that used this entry; the victim
    /// is the smallest.  Evicting by insertion order would throw away the
    /// cohort every case is still asking for.
    std::uint64_t                  last_use = 0;
};

inline std::vector<Entry>& entries() {
    static std::vector<Entry> list;
    return list;
}

inline std::atomic<std::uint64_t>& clock() {
    static std::atomic<std::uint64_t> tick{0};
    return tick;
}

inline std::size_t limit() {
    static const std::size_t value = [] {
        const char*     v = std::getenv("RASBERY_COHORT_CACHE_ENTRIES");
        const long long requested = (v && *v) ? std::atoll(v) : 0;
        return requested > 0 ? static_cast<std::size_t>(requested) : std::size_t{64};
    }();
    return value;
}

} // namespace detail

/// The pin-power quadrature table for one (ndivxy, npins), built at most once
/// per process.
///
/// Keyed on its TRUE argument and not on the cohort key: two cohorts that
/// differ only in their cross-section library have the same quadrature, and a
/// table keyed on the wider thing would be built twice for no reason.  The
/// build happens UNDER the mutex -- it is milliseconds, it happens once, and
/// the single-flight dance AcquireXsLibrary needs (where a miss is a 34 MB HDF5
/// parse) would be complexity bought for nothing here.
inline std::shared_ptr<const PinQuadTable> acquirePinQuadrature(int ndivxy, int npins) {
    std::lock_guard<std::mutex> guard(detail::quadMutex());
    for (const auto& e : detail::quadEntries())
        if (e.ndivxy == ndivxy && e.npins == npins) {
            detail::counter(3).fetch_add(1, std::memory_order_relaxed);
            return e.table;
        }
    auto table = std::make_shared<const PinQuadTable>(buildPinQuadratureTable(ndivxy, npins));
    detail::counter(2).fetch_add(1, std::memory_order_relaxed);
    detail::quadEntries().push_back(detail::QuadEntry{ndivxy, npins, table});
    return table;
}

/// Acquire the Context for *d*, building it if this process has not seen the
/// cohort before.
///
/// Throws when a key collides with a different shape.  That is not defensive
/// programming: a hit that handed back maps sized for another core would
/// produce numbers, and they would be wrong in a way no receipt would show.
inline std::shared_ptr<const Context> acquire(const Descriptor& d) {
    const std::string key = d.key();
    std::lock_guard<std::mutex> guard(detail::registryMutex());
    for (auto& e : detail::entries())
        if (e.key == key) {
            const Context& c = *e.value;
            if (c.ng != d.ng || c.ndivxy != d.ndivxy || c.npins != d.npins)
                throw std::runtime_error(
                    "cohort::acquire: key " + key + " already names a cohort of shape (ng=" +
                    std::to_string(c.ng) + ", ndivxy=" + std::to_string(c.ndivxy) +
                    ", npins=" + std::to_string(c.npins) + ") and this case is (ng=" +
                    std::to_string(d.ng) + ", ndivxy=" + std::to_string(d.ndivxy) +
                    ", npins=" + std::to_string(d.npins) +
                    "). The cohort key does not cover something the shared state depends "
                    "on; fix the key (src/CohortKey.h) rather than the symptom.");
            detail::counter(1).fetch_add(1, std::memory_order_relaxed);
            e.last_use = detail::clock().fetch_add(1, std::memory_order_relaxed) + 1;
            return e.value;
        }
    std::shared_ptr<const Context> built(new Context{
        key, d.geometry_digest, d.xslib_digest, d.ng, d.ndivxy, d.npins,
        acquirePinQuadrature(d.ndivxy, d.npins)});
    detail::counter(0).fetch_add(1, std::memory_order_relaxed);
    detail::entries().push_back(detail::Entry{
        key, built, detail::clock().fetch_add(1, std::memory_order_relaxed) + 1});
    // Bounded, least-recently-used first.  Dropping an entry is safe at any
    // moment: every case holds its own shared_ptr, so the Context dies when the
    // last case using it does, and a later case with the same key rebuilds a
    // BIT-IDENTICAL one (the Context is a pure function of the key's inputs --
    // that is the whole argument for sharing it).  What an eviction costs is a
    // rebuild, and `evictions` is what says it happened.
    while (detail::entries().size() > detail::limit()) {
        auto victim = detail::entries().begin();
        for (auto it = detail::entries().begin(); it != detail::entries().end(); ++it)
            if (it->last_use < victim->last_use) victim = it;
        if (victim->key == key) break; // never the one we are about to hand back
        detail::entries().erase(victim);
        detail::counter(4).fetch_add(1, std::memory_order_relaxed);
    }
    return built;
}

inline Stats snapshot() {
    Stats s;
    s.builds            = detail::counter(0).load(std::memory_order_relaxed);
    s.hits              = detail::counter(1).load(std::memory_order_relaxed);
    s.quadrature_builds = detail::counter(2).load(std::memory_order_relaxed);
    s.quadrature_hits   = detail::counter(3).load(std::memory_order_relaxed);
    s.evictions         = detail::counter(4).load(std::memory_order_relaxed);
    s.limit             = static_cast<std::uint64_t>(detail::limit());
    std::lock_guard<std::mutex> guard(detail::registryMutex());
    s.cohorts = static_cast<std::uint64_t>(detail::entries().size());
    return s;
}

/// One line, [RASBERY][COHORT].
///
/// `builds` must equal the number of distinct geometry/library pairs the
/// process saw and must NOT grow with the case count.  In a GA generation of M
/// candidates over one core that is builds=1, hits=M-1 -- and if it is ever
/// builds=M, the cohort key has started covering something a candidate changes
/// and the middle lifetime is gone again without a word from any other number.
inline void printReceipt(std::ostream& out) {
    const Stats s = snapshot();
    out << "[RASBERY][COHORT] {\"builds\":" << s.builds << ",\"hits\":" << s.hits
        << ",\"cohorts\":" << s.cohorts
        << ",\"quadrature_builds\":" << s.quadrature_builds
        << ",\"quadrature_hits\":" << s.quadrature_hits
        << ",\"limit\":" << s.limit
        << ",\"evictions\":" << s.evictions
        << ",\"schema\":\"" << kSchema << "\"}" << std::endl;
}

} // namespace rasbery::cohort
