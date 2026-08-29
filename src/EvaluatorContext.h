#pragma once

// The two lifetimes a GA evaluator has, named.
//
// WHY THIS FILE EXISTS.  `--evaluator` (GA evaluator plan Sec 6.2 Task 6) is a
// process that stands its state up once and then answers cases.  That is only
// a safe thing to build if the boundary between "belongs to this case" and
// "belongs to this process" is already drawn and already correct -- and the
// campaign has twice paid for getting that boundary wrong at a smaller scale
// (the permanent cudaHostRegister that outlived its Driver, HostPinRegistry.h;
// the mutable `double&` into the parsed library, XsLibrary.h).  So the boundary
// is drawn HERE, now, while every object still has one case's lifetime and the
// bit-identity gate is a 40 s run.  Task 6 then changes WHEN the two are
// destroyed, not WHAT is in each -- which is a change one h5diff can gate.
//
// This file names the split.  It does not implement a server, and nothing here
// keeps state alive past `Drive()` yet.
//
// WHAT IS ALREADY PER PROCESS (and needs no further work):
//
//   * the parsed CHIFFON library -- `AcquireXsLibrary`, shared_ptr<const>,
//     keyed by (path, size, mtime, ng).  Measured: loads=1, hits=jobs-1.
//   * the T/H property tables -- five CSVs, function-local static in
//     XSSet::LoadTHTables, copied per Driver because they are kilobytes.
//   * the isotope registry and the depletion chain -- `Isotope::niso`,
//     `Isotope::EnsureInitialized`, both idempotent process globals already.
//   * the DEVICE flat-XS library -- `g_flatxs_libs`, keyed by content hash,
//     process lifetime by construction (CudaXsReconBackend.cu).
//
// WHAT IS STILL PER CASE AND WOULD HAVE TO MOVE FOR `--evaluator`:
//
//   * `Geometry` -- rebuilt per deck.  `Geometry::Initialize` allocates with
//     bare `new` and has no matching release path, so it can be built once but
//     NOT re-initialised; an evaluator has to keep one and refill the parts a
//     candidate actually changes (`_core`, `_batch`, `_is_fuel`, `_kbc`,
//     `_kec`, `_hzcore` -- all cheap scans).
//   * the nxyz-sized half of `XSSet` -- `_comp`/`_asmb`, the live `_xs`,
//     `_micx`, `_lmpx`, `_iden` blocks and their reference and
//     beginning-of-step twins.  These are the arrays HostPinRegistry leases,
//     so they must be rebuilt, or explicitly reset, per case -- never shared.
//   * `Scheduler` -- the deck's schedule, plus whatever natural EOC appended.
//   * the arena slot -- already acquired and released per case.
//
// The one rule that governs the split: PROCESS state is immutable after
// stand-up; CASE state is everything a solve writes to.  A field that is
// written during a solve and lives in ProcessContext is a cross-case leak, and
// the digest is what would find it -- late.

#include "Geometry.h"
#include "IO.h"
#include "Scheduler.h"
#include "XSSet.h"
#include "XsLibrary.h"

#include <string>

namespace rasbery {

/// Everything one case owns and destroys with itself.
///
/// The four objects were `Drive()`'s stack locals, in this order, and the order
/// is load-bearing: `XSSet` holds `Geometry&` and `IO` holds all three, so they
/// must be declared before it and destroyed after it.  Naming the group makes
/// that ordering a property of a type instead of a property of one function's
/// prologue.
struct CaseContext {
    Geometry  geometry;
    Scheduler scheduler;
    XSSet     cross_sections;
    IO        input_output;

    CaseContext() noexcept
        : cross_sections(geometry),
          input_output(geometry, cross_sections, scheduler) {}

    CaseContext(const CaseContext&)            = delete;
    CaseContext& operator=(const CaseContext&) = delete;
};

/// What already survives a case, as one readable snapshot.
///
/// A snapshot, not an owner: the state itself lives in the caches named in the
/// header comment, and this is how a receipt (and, later, an evaluator's
/// stand-up assertion `library_loads == 1`) reads it without reaching into
/// each cache by hand.
struct ProcessContext {
    XsLibraryCacheStats xslib;
};

inline ProcessContext processContext() {
    return ProcessContext{XsLibraryCacheSnapshot()};
}

} // namespace rasbery
