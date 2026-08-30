#pragma once

// XsLibrary -- the parsed-and-flattened CHIFFON cross-section library, split
// out of XSSet so that every Driver in a process can share ONE copy of it.
//
// WHY THIS TYPE EXISTS.  `Importer::LoadHDF` is a pure function of a filename
// (Importer.h) and the flatten that follows it in XSSet::Initialize is a pure
// function of (_models, ng, niso).  Nothing in either reads the deck's core or
// batch map.  So M decks that name the same library produce M bit-identical
// copies of the same ~34 MB parse -- and, because IO::ReadInput holds the
// process-global Chiffon::Hdf5Guard across the whole call, they produce them
// ONE AT A TIME.  The measured shape of that queue is a staircase: eight
// concurrent workers reported Init+IO of 4.108, 4.682, 6.398, 8.879, 10.041,
// 11.358, 13.194, 14.506 s, a slope of ~1.49 s per case, and 63.8 s of
// [HDF5][LOCK] wait for eight decks (GA evaluator plan Sec 3.1).  A width-64
// wave would put the last case ~98 s behind the first before its first outer.
//
// The device side already had this cache (`g_flatxs_libs` in
// CudaXsReconBackend.cu keys a device library by content hash and keeps it for
// the process lifetime).  The host side did not.  Plan Rev.4 Sec 14 specified
// it; `Driver.h`'s `library_seconds` receipt was planted for its A/B.
//
// WHAT IS SHARED AND WHAT IS NOT.  Everything here is written once, by
// BuildXsLibrary, and is const from the moment the shared_ptr is published.
// It is exactly the state whose value depends on the library FILE and on
// nothing else in the deck:
//
//   - `models`      the parsed CHIFFON models (Importer::LoadHDF's return)
//   - `lib_*`       the SoA flatten of those models, indexed by depletion point
//   - `refr_*`      per-model reference (ctype, burnup) index tables
//   - `brch_*`      per-model branch-delta index tables
//
// Every array whose length involves nxyz -- the node-wise `_node_*` brackets,
// `_comp`/`_asmb`, the live `_xs`/`_micx`/`_lmpx`/`_iden` blocks and their
// reference and beginning-of-step twins -- stays in XSSet, per Driver, mutable.
// That split is not a convention: those are the arrays the CUDA backends
// page-lock through HostPinRegistry, and a pinned range whose owner is one
// Driver must not be reachable from another (HostPinRegistry.h).  None of the
// arrays in this struct is ever passed to pinHost; the flatxs library view
// hands their raw pointers to the device path read-only, where the content-hash
// cache now finds the same bytes at the same address and matches on the first
// try.
//
// THE CACHE KEY IS A CONTENT KEY (WP8 stage 2).  It used to be
// (canonical path, size, mtime, ng).  In a one-shot process that is fine: the
// library cannot be replaced between the first deck and the last, because the
// whole process is shorter than the operation that would replace it.  A
// long-lived evaluator breaks that assumption -- it can outlive a library
// rebuild -- and the failure mode is silent and total: every case after the
// swap reuses the OLD parse, produces perfectly self-consistent numbers, and
// nothing in any receipt says which library they came from.
//
// So the key now carries a SHA-256 of the file's bytes, and every parse
// records it in `content_digest`.  The same digest is what casekey::Provenance
// calls `xslib_digest` (src/CaseKey.h) and the same transform computes it
// (Sha256.h), because two hashes of one file is how a cache key ends up
// disagreeing with the receipt that named it.
//
// WHAT IS STILL OPEN, AND IT IS NAMED HERE RATHER THAN PAPERED OVER.  The
// digest itself is memoised by (path, size, mtime) so a wave of M cases reads
// the 34 MB once instead of M times.  A replacement that keeps the same size
// AND lands inside the filesystem's mtime resolution therefore still returns
// the stale digest.  That hole is strictly smaller than the one it replaces --
// it now needs the size to match too -- and `RASBERY_XSLIB_DIGEST=always`
// closes it completely at the cost of one file read per acquisition, for the
// campaigns that actually rewrite libraries underneath a running fleet.
//
// THE ONE MUTABLE ALIAS THAT HAD TO GO.  XSSet::fmap()/gmap() returned
// `double&` into `_models[...]`, and PPR took `auto&` (which selected the
// non-const `GetDepletionPoint` and `_refr_dpts[0]`, an inserting lookup).
// A shared immutable parse cannot have either; fmap/gmap were dead and are
// deleted, and PPR's binding is const.  See XSSet.cpp and PPR.cpp.

#include "Model.h"
#include "milk.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace rasbery {

// SoA cross-section array set
struct XSArraySet {
    milk::Vector<double> xstf, xsdf, xsaf, xsff, xsnf, xskf, xssf, xsrf;
    milk::Vector<double> fyld, xs2n, xs3n;
    milk::Vector<double> xssm;

    void allocate(size_t scalar_size, size_t sm_size);
    void clear();
    void fill(double v);

    milk::Vector<double>&       operator[](Chiffon::XSTYPE xt);
    const milk::Vector<double>& operator[](Chiffon::XSTYPE xt) const;
};

inline constexpr int BRANCH_BPPM          = 0;
inline constexpr int BRANCH_TFUL          = 1;
inline constexpr int BRANCH_DMOD          = 2;
inline constexpr int NUM_SCALAR_BRANCHES  = 3;

/// One flattened branch delta: where its coefficients and knots live.
struct DeltaInfo {
    int nord        = 0;
    int mode        = 0;
    int ncoeff      = 0;
    int coeff_base  = 0;
    int knot_offset = 0;
    int knot_count  = 0;
};

/// One spectral-history correction, flattened.
struct SpectralHistoryInfo {
    Chiffon::SpectralTerm term;
    size_t                delta_base = 0;
    std::vector<int>      burnups;
    bool                  rod_scaled = false;
};

/// Immutable, process-shareable parse of one CHIFFON library file.
struct XsLibrary {
    // Provenance -- the cache key, kept with the value so a receipt can quote it.
    std::string   path;         ///< canonical path the parse came from
    std::uint64_t file_size = 0;
    std::int64_t  mtime      = 0; ///< filesystem clock ticks, as reported by last_write_time
    /// SHA-256 of the library file's bytes -- the part of the key that survives
    /// a same-size, same-mtime replacement.  Empty only when the file could not
    /// be read at all, which is a different fact from "the digest is zero".
    std::string   content_digest;
    int           ng         = 0; ///< energy groups the flatten was laid out for
    size_t        niso       = 0; ///< isotope registry size at parse time
    size_t        bytes      = 0; ///< approximate resident footprint of this parse

    std::vector<Chiffon::Model> models;

    bool has_coeff_micx = false; // any delta carries microscopic XS

    std::vector<size_t>                        refr_base;        // model -> first reference flat id
    std::vector<size_t>                        refr_ctyp_stride; // model -> flat ids per reference ctype row
    std::vector<size_t>                        refr_burn_stride; // model -> flat id step per reference burn point
    std::vector<std::vector<int>>              refr_ctyp;        // model -> ordered reference ctype list
    std::vector<std::vector<std::vector<int>>> refr_burn;        // model -> ctype -> burn keys

    std::vector<std::vector<size_t>>                        brch_base;        // model, branch -> first delta id
    std::vector<std::vector<size_t>>                        brch_ctyp_stride; // model, branch -> ids per ctype row
    std::vector<std::vector<size_t>>                        brch_burn_stride; // model, branch -> id step per burn point
    std::vector<std::vector<std::vector<int>>>              brch_ctyp;        // model, branch -> ctype list
    std::vector<std::vector<std::vector<std::vector<int>>>> brch_burn;        // model, branch, ctype -> burn keys

    XSArraySet                         lib_lmpx;         // library reference lumped XS, indexed by depletion point
    XSArraySet                         lib_micx;         // library reference microscopic XS, indexed by depletion point
    milk::Vector<double>               lib_iden;         // library reference isotope densities
    std::vector<double>                lib_burn;         // library reference burnup in GWd/THM
    std::vector<double>                lib_wvfr;         // library reference water volume fraction
    std::vector<std::array<double, 3>> lib_ref_branch_x; // reference bppm/tful/dmod coordinates
    std::vector<double>                lib_flux;         // library reference average flux [dpt*ng + ig]
    std::vector<double>                lib_chix;         // library reference fission spectrum [dpt*ng + ig]

    XSArraySet                                    lib_coeff_lmpx; // branch delta coefficients for lumped XS
    XSArraySet                                    lib_coeff_micx; // branch delta coefficients for microscopic XS
    std::vector<DeltaInfo>                        lib_deltas;     // branch delta metadata and coefficient offsets
    std::vector<double>                           lib_knots;      // concatenated spline knots for branch deltas
    std::vector<double>                           lib_model_volu; // model reference volume for burnup normalization
    std::vector<double>                           lib_model_hmas; // model heavy metal mass for burnup normalization
    std::vector<std::vector<SpectralHistoryInfo>> lib_spectral_history;
    std::vector<int>                              lib_history_partner; // model -> partner model index (-1 none)
};

/// Cache receipt, printed as [RASBERY][XSLIB_CACHE].
struct XsLibraryCacheStats {
    std::uint64_t loads         = 0; ///< parses actually performed
    std::uint64_t hits          = 0; ///< acquisitions served from the cache
    std::uint64_t waits         = 0; ///< acquisitions that waited for another worker's parse
    std::uint64_t bytes         = 0; ///< resident footprint of the cached parses
    /// Time acquisitions spent inside the cache: the mutex, plus (for the
    /// workers that arrived while one of them was parsing) the wait for that
    /// one parse.  It is the cost the batch still pays for a COLD library, and
    /// it belongs beside [HDF5][LOCK].wait_ms, which is what it replaced.
    std::uint64_t lock_wait_ms  = 0;
    std::uint64_t entries       = 0;
    /// Times a file was actually read to compute its content digest.  THE
    /// WITNESS for the key hardening: it must track the number of distinct
    /// (path, size, mtime) triples the process saw and must NOT grow with the
    /// case count -- if it does, the memoisation is not working and every case
    /// is paying a 34 MB read for a key.
    std::uint64_t digest_computes = 0;
    /// WP10.4 -- THE BOUND, AND ITS WITNESS.
    ///
    /// Every entry here holds a whole ~34 MB parse alive for the process
    /// lifetime, and the key carries the file's size, mtime and content digest:
    /// a library REPLACED under a running evaluator does not overwrite its
    /// entry, it adds one.  In a one-shot process that could not happen (the
    /// process is shorter than the operation that would replace the library);
    /// in a GA evaluator that must survive 10k generations it is 34 MB per
    /// rebuild, kept forever, and nothing in the old receipt named it.  So the
    /// table is bounded (`RASBERY_XSLIB_CACHE_ENTRIES`, default 2 -- enough to
    /// hold both sides of a swap without re-parsing either) and every eviction
    /// is counted.  `evictions > 0` means this process outlived a library.
    std::uint64_t evictions       = 0;
    std::uint64_t entry_limit     = 0;
    /// The (path, size, mtime) -> digest memo behind the key above.  Tiny per
    /// entry and unbounded for the same reason, so bounded for the same reason
    /// (`RASBERY_XSLIB_DIGEST_ENTRIES`, default 64).
    std::uint64_t digest_entries   = 0;
    std::uint64_t digest_evictions = 0;
};

/// Content digest of one library file, memoised by (path, size, mtime).
///
/// Exposed because the cohort key needs the same digest the cache key uses:
/// a cohort whose library provenance came from a second hash of the same file
/// is a cohort that can disagree with its own cache.
std::string XsLibraryContentDigest(const std::string& xs_path);

/// Which RASBERY_XSLIB_DIGEST policy this process resolved: "cached", "always"
/// or "off".
///
/// EXPOSED BECAUSE `off` MAKES THE CASE KEY LIE BY OMISSION.  With the policy
/// off the digest above is the EMPTY STRING, so casekey::Provenance carries no
/// library provenance at all and two runs against two different libraries share
/// a key.  That is a legitimate A/B control (it is the only way to show the
/// hardening is free), but a run that took it has to SAY so in the receipt --
/// otherwise a controller recomputing the key with the policy at its default
/// gets a different answer and has no way to see why.  This is exactly the
/// WP10.1 failure host 181 hit on kngr_238.json.
const char* XsLibraryDigestPolicyName();

/// Parse + flatten one library file.  Pure function of (xs_path, ng); enters HDF5.
std::shared_ptr<const XsLibrary> BuildXsLibrary(const std::string& xs_path, int ng);

/// Process-wide cached acquire, keyed by (canonical path, size, mtime, ng).
///
/// THE LOOKUP IS NOT UNDER Chiffon::Hdf5Guard.  It has its own mutex, and the
/// HDF5 guard is taken only on the miss path, inside BuildXsLibrary -> LoadHDF.
/// A hit therefore costs one mutex and a shared_ptr copy even while another
/// worker holds the HDF5 lock, which is the whole point: the staircase in the
/// header comment is the cost of queueing for a parse that has already been
/// done.
std::shared_ptr<const XsLibrary> AcquireXsLibrary(const std::string& xs_path, int ng);

/// Snapshot of the cache counters.
XsLibraryCacheStats XsLibraryCacheSnapshot();

/// One-line JSON receipt: [RASBERY][XSLIB_CACHE] {...}
void PrintXsLibraryCacheReceipt(std::ostream& out);

} // namespace rasbery
